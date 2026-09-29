"""Dense search over one action round, and the opponent's reply to it.

MCTS spends a fixed number of simulations wherever PUCT sends them. This searcher replaces the
sampling with enumeration: from the decision it is handed it walks EVERY legal line to the end of
the current action round, evaluates each end position with the network's `v_win`, and backs the
values up with max at the searching player's decisions and min at the opponent's. Nothing is
sampled and nothing is skipped, so it cannot miss the one line a 64-simulation tree never visits.

**Why the opponent's reply is not enumerated the same way.** Measured on random-play games, one
action round has 10^4-10^5 distinct end positions after transposition merging (61,130 to 149,666
in the first four rounds probed), and the opponent's reply from each of those is another 10^4-10^5.
Dense over both is ~10^9 network evaluations per decision. So the reply is dense but *pruned at the
seam*: after the own-round pass, the `reply_top_k` best first actions are each followed down their
principal line to the position where the opponent's round begins, and THAT position gets a dense
pass over the opponent's whole round. Its value is the minimum over every reply, read at the start
of the searching player's next round -- the depth the caller asked to see. Among those candidates
the searcher plays the one whose worst reply is best.

    reply_top_k = 0    own round only; leaves are scored at the start of the opponent's round
    reply_top_k = K    the above, with K candidates checked against a dense opponent reply
    reply_top_k = -1   every root action is checked (only sensible on a small round)

**What a value is.** Every value is from the searching player's perspective (`P`), read off the
network at the first state that is no longer inside the round. Terminal states use the exact game
utility. The mover flips mid-round about a third of the time (interrupt events hand the opponent a
decision inside the searcher's round), so the backup takes the max or the min by whoever decides at
each node rather than treating the leaves as a bag.

**Chance.** A die is rolled once per line, from the state's own PRNG, and that PRNG is reseeded at
the root so the search never reads the dice the real game is about to throw. Lines that reach a die
with the same roll count see the same outcome (common random numbers): otherwise the maximum over
10^5 lines would select the ones that rolled well, which is a bias of the enumeration and not
information about the position. It still means a coup's expected value is read from a single die.

**Known limitation, stated because it is the main risk.** An argmax over ~10^5 leaves is an
optimiser's curse on the value head: the line that wins is disproportionately the one whose value
error is most positive. Deeper search averages that error down; a wider one amplifies it. Whether
the reply stage pays for itself is a measurement, and this file does not claim it.

Hidden information follows `batched_mcts.py`: `determinize=False` searches the true state and reads
the opponent's hand (a diagnostic upper bound), `determinize=True` resamples the unseen cards once
per search (deployable).
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.dmcts import determinize, hidden_pool
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle

_UINT64 = 1 << 64
_US = int(ts.Player.US)


@dataclass
class RoundSearchConfig:
    #: See the module docstring. 0 disables the reply stage, -1 checks every root action.
    reply_top_k: int = 8
    #: Distinct positions evaluated per network call, and the capacity of the C++ featuriser.
    eval_batch: int = 4096
    #: Stop expanding one enumeration after this many leaves and score the rest of the frontier
    #: where it stands. A truncation is counted, never silent: `RoundSearch.truncations`.
    max_leaves: int = 400_000
    #: Pruning. Each node keeps the smallest set of its highest-prior actions whose renormalised
    #: policy mass reaches `p_*` (always at least one). 1.0 keeps everything, which is the exact
    #: search. `p_own` applies where the searching player decides, `p_opp` where the opponent
    #: decides inside the searcher's own round, `p_reply` where the opponent decides in the reply
    #: stage. The opponent thresholds are the safer place to be aggressive only if you accept an
    #: optimistic minimum: a refutation the policy thinks unlikely is never looked at.
    p_own: float = 1.0
    p_opp: float = 1.0
    p_reply: float = 1.0
    #: Cap on actions kept per node after the mass rule (0 = no cap).
    max_branch: int = 0
    #: Nodes expanded per layer for each side's choices, best by value for whoever chose them
    #: (0 = all). The rest of the layer is scored where it stands.
    beam: int = 0
    determinize: bool = False
    seed: int = 12345
    #: Follow the line found at the start of a round while the real game stays on it, instead of
    #: paying for a fresh search at every micro-decision of the same round. A die that lands off
    #: the searched line changes the position, the lookup misses, and a fresh search runs.
    follow_plan: bool = True
    plan_capacity: int = 200_000


@dataclass
class _DNode:
    mover: int
    actions: List[int]
    kids: List[bytes]


@dataclass
class _Dag:
    """Every position reachable inside one round, with the value of each leaf.

    Interior positions are keyed by a hash of the board with the PRNG zeroed, so two orderings of
    the same placements are one node. States are NOT kept -- a round holds ~10^5 of them at 4 KB
    each -- only the edges, which is all the backup needs.
    """

    root: bytes
    nodes: Dict[bytes, _DNode] = field(default_factory=dict)
    leaf_val: Dict[bytes, float] = field(default_factory=dict)
    value: Dict[bytes, float] = field(default_factory=dict)
    best: Dict[bytes, int] = field(default_factory=dict)
    truncated: bool = False
    was_truncated: bool = False

    def v(self, h: bytes, p: int) -> float:
        """Value of `h` for player `p`: max where `p` decides, min where the opponent does."""
        if h in self.leaf_val:
            return self.leaf_val[h]
        if h in self.value:
            return self.value[h]
        nd = self.nodes[h]
        bi, bv = -1, 0.0
        for i, c in enumerate(nd.kids):
            x = self.v(c, p)
            if bi < 0 or (x > bv if nd.mover == p else x < bv):
                bi, bv = i, x
        self.value[h] = bv
        self.best[h] = bi
        return bv


@dataclass
class RoundResult:
    action: int
    #: Own-round value of each root action, and the reply-checked value where one was computed.
    v_round: Dict[int, float]
    v_reply: Dict[int, float]
    #: The actions of the chosen line, root first, ending where the opponent's round begins.
    line: List[int]
    leaves: int
    seconds: float


def _round_key(st: ts.GameState) -> Tuple[int, int, int]:
    """Identifies one player's action round. The phase stays ACTION_ROUND throughout, and an
    extra round by the same player is a new `action_round`, so this is what changes at the seam."""
    return (int(st.turn), int(st.action_round), int(st.phasing_player))


def _hash(st: ts.GameState) -> bytes:
    """The board, ignoring the PRNG: dice must not make identical positions distinct."""
    r = st.rng_state
    st.rng_state = 0
    try:
        return hashlib.blake2b(bytes(st.raw_bytes()), digest_size=16).digest()
    finally:
        st.rng_state = r


def _plan_key(st: ts.GameState, me: ts.Player) -> bytes:
    """`_hash` with the cards `me` cannot see moved to one place, so a position reached in a
    determinized world keys the same as the real one it stands for."""
    c = st.clone()
    opp_hand, deck, _ = hidden_pool(st, me)
    for cid in opp_hand + deck:
        c.set_card_location(cid, ts.CardLocation.DRAW_DECK)
    return _hash(c)


def _legal(st: ts.GameState) -> np.ndarray:
    return np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))


class RoundSearch:
    def __init__(self, model, device=None, config: Optional[RoundSearchConfig] = None) -> None:
        self.model = model
        self.device = device or next(model.parameters()).device
        self.cfg = config or RoundSearchConfig()
        self._featuriser = ts.VectorizedBatchRunner(self.cfg.eval_batch, int(self.cfg.seed))
        self._rng = random.Random(self.cfg.seed)
        self._plan: Dict[bytes, int] = {}
        self.truncations = 0
        #: Candidates whose line stopped short of the opponent's round (a truncated own-round pass),
        #: so they were compared on their own-round value alone.
        self.reply_skipped = 0
        self.plan_hits = 0
        self.searches = 0
        self.evaluated = 0
        self.net_seconds = 0.0
        self.model.eval()

    # -- evaluation ------------------------------------------------------------------------

    def _forward(self, states: Sequence[ts.GameState], want_policy: bool
                 ) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """(policy probabilities over the legal actions, `v_win` of whoever moves) per state.

        Chunked to the featuriser's capacity. `probs` is None unless asked for: the softmax over
        220 columns is wasted on a leaf that only needs its value.
        """
        t_net = time.time()
        n = len(states)
        cap = self.cfg.eval_batch
        probs = np.zeros((n, ActionEncoder.FLAT_ACTION_SIZE), dtype=np.float32) if want_policy else None
        vals = np.zeros(n, dtype=np.float32)
        for lo in range(0, n, cap):
            chunk = states[lo:lo + cap]
            m = len(chunk)
            for i, st in enumerate(chunk):
                self._featuriser.set_state(i, st)
            # set_state leaves the cached observation stale (see BatchedMCTS._featurise).
            self._featuriser.refresh_all()
            obs = np.asarray(self._featuriser.get_observations(), dtype=np.float32)[:m]
            masks = np.asarray(self._featuriser.get_action_masks())[:m]
            with torch.no_grad():
                logits, v_win, _ = self.model.forward(torch.from_numpy(obs).to(self.device),
                                                      torch.from_numpy(masks).to(self.device))
                vals[lo:lo + m] = v_win.squeeze(-1).cpu().numpy()
                if probs is not None:
                    probs[lo:lo + m] = torch.softmax(logits, dim=-1).cpu().numpy()
        self.evaluated += n
        self.net_seconds += time.time() - t_net
        return probs, vals

    def _flush(self, hashes: List[bytes], states: List[ts.GameState], p: int,
               out: Dict[bytes, float]) -> None:
        """Score positions in one batch: `v_win` of whoever moves there, read for `p`."""
        if not states:
            return
        _, vals = self._forward(states, want_policy=False)
        for h, st, v in zip(hashes, states, vals):
            out[h] = float(v) if int(acting_player(st)) == p else -float(v)

    # -- enumeration -----------------------------------------------------------------------

    def _keep(self, probs_row: np.ndarray, legal: np.ndarray, mass: float) -> List[int]:
        """Indices into `legal` to expand: highest prior first until `mass` is covered."""
        pr = probs_row[legal].astype(np.float64)
        tot = pr.sum()
        pr = pr / tot if tot > 1e-12 else np.full(len(legal), 1.0 / len(legal))
        order = np.argsort(-pr, kind="stable")
        cap = self.cfg.max_branch or len(order)
        if mass >= 1.0:
            return [int(i) for i in order[:cap]]
        cum = np.cumsum(pr[order])
        n = int(np.searchsorted(cum, mass - 1e-12) + 1)
        return [int(i) for i in order[:max(1, min(n, cap))]]

    def _enumerate(self, root: ts.GameState, p: int, p_opp: float) -> _Dag:
        """Lines from `root` to the end of the round `root` is in, valued for player `p`.

        Layer by layer, so the network sees every node of a layer in one batch. With every
        threshold at 1.0 and no cap, beam or budget pressure this needs no policy at all and is
        the exhaustive search; otherwise a layer is first run through the network to rank it.
        """
        cfg = self.cfg
        k0 = _round_key(root)
        root_h = _hash(root)
        dag = _Dag(root=root_h)
        buf_h: List[bytes] = []
        buf_s: List[ts.GameState] = []
        queued: set = set()

        def add_leaf(h: bytes, st: ts.GameState) -> None:
            if h in dag.leaf_val or h in queued:
                return
            if ts.Engine.is_terminal(st):
                u = float(ts.Engine.get_terminal_utility(st))
                dag.leaf_val[h] = u if p == _US else -u
                return
            queued.add(h)
            buf_h.append(h)
            buf_s.append(st)
            if len(buf_h) >= cfg.eval_batch:
                self._flush(buf_h, buf_s, p, dag.leaf_val)
                queued.clear()
                buf_h.clear()
                buf_s.clear()

        def mass_for(mover: int) -> float:
            return cfg.p_own if mover == p else p_opp

        pruning = (cfg.p_own < 1.0 or p_opp < 1.0 or cfg.max_branch > 0)
        seen = {root_h}
        # (hash, state, reach probability, who chose to arrive here)
        frontier: List[Tuple[bytes, ts.GameState, float, int]] = [(root_h, root, 1.0, p)]
        while frontier:
            legals = [_legal(st) for _, st, _, _ in frontier]
            est = sum(len(x) for x in legals)
            room = cfg.max_leaves - (len(dag.leaf_val) + len(buf_h))
            need_rank = pruning or cfg.beam > 0 or est > room
            probs: Optional[np.ndarray] = None
            vals_p: Optional[np.ndarray] = None
            if need_rank:
                probs, vals = self._forward([st for _, st, _, _ in frontier], want_policy=True)
                vals_p = np.array([float(v) if int(acting_player(st)) == p else -float(v)
                                   for v, (_, st, _, _) in zip(vals, frontier)])

            order = list(range(len(frontier)))
            expand = set(order)
            if vals_p is not None and cfg.beam > 0:
                expand = set()
                for chooser in (p, -p):
                    grp = [i for i in order if frontier[i][3] == chooser]
                    grp.sort(key=lambda i: -vals_p[i] if chooser == p else vals_p[i])
                    expand.update(grp[:cfg.beam])
            if need_rank and est > room:
                # Out of budget: expand the likeliest nodes, and score the rest where they stand.
                ranked = sorted(expand, key=lambda i: -frontier[i][2])
                expand, spent = set(), 0
                for i in ranked:
                    k = len(legals[i])
                    if spent + k > room and expand:
                        break
                    expand.add(i)
                    spent += k
                dag.truncated = True

            nxt: List[Tuple[bytes, ts.GameState, float, int]] = []
            for i, (h, st, reach, _chooser) in enumerate(frontier):
                if i not in expand:
                    assert vals_p is not None
                    dag.leaf_val[h] = float(vals_p[i])
                    continue
                legal = legals[i]
                mover = int(acting_player(st))
                if probs is not None:
                    keep = self._keep(probs[i], legal, mass_for(mover))
                    pr = probs[i][legal].astype(np.float64)
                    pr = pr / pr.sum() if pr.sum() > 1e-12 else np.full(len(legal), 1.0 / len(legal))
                else:
                    keep, pr = list(range(len(legal))), np.full(len(legal), 1.0 / len(legal))
                kids: List[bytes] = []
                acts: List[int] = []
                for ki in keep:
                    a = int(legal[ki])
                    ch = st.clone()
                    ts.Engine.step_flat(ch, a)
                    settle(ch, SettleMode.FORCED)
                    c_h = _hash(ch)
                    kids.append(c_h)
                    acts.append(a)
                    if ts.Engine.is_terminal(ch) or _round_key(ch) != k0:
                        add_leaf(c_h, ch)
                    elif c_h not in seen:
                        seen.add(c_h)
                        nxt.append((c_h, ch, reach * float(pr[ki]), mover))
                dag.nodes[h] = _DNode(mover=mover, actions=acts, kids=kids)
            if dag.truncated:
                dag.truncated = False
                dag.was_truncated = True
            frontier = nxt
        self._flush(buf_h, buf_s, p, dag.leaf_val)
        if dag.was_truncated:
            self.truncations += 1
        return dag

    def _line(self, dag: _Dag, first: int) -> List[int]:
        """The principal line after `first` at the root: best edge at every node below it."""
        nd = dag.nodes[dag.root]
        idx = nd.actions.index(first)
        line = [first]
        h = nd.kids[idx]
        while h in dag.nodes:
            node = dag.nodes[h]
            i = dag.best[h]
            line.append(node.actions[i])
            h = node.kids[i]
        return line

    @staticmethod
    def _replay(root: ts.GameState, line: Sequence[int]) -> ts.GameState:
        st = root.clone()
        for a in line:
            ts.Engine.step_flat(st, int(a))
            settle(st, SettleMode.FORCED)
        return st

    # -- search ----------------------------------------------------------------------------

    def _remember(self, root: ts.GameState, line: Sequence[int], me: ts.Player) -> None:
        """Record the searcher's own decisions along `line`, keyed by the position they were for."""
        if len(self._plan) > self.cfg.plan_capacity:
            self._plan.clear()
        st = root.clone()
        for a in line:
            if acting_player(st) == me:
                self._plan[_plan_key(st, me)] = int(a)
            ts.Engine.step_flat(st, int(a))
            settle(st, SettleMode.FORCED)

    def planned(self, state: ts.GameState) -> Optional[int]:
        """The action the last search planned for exactly this position, if it is still legal."""
        if not self.cfg.follow_plan or not self._plan:
            return None
        a = self._plan.get(_plan_key(state, acting_player(state)))
        if a is None:
            return None
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        return int(a) if 0 <= a < len(mask) and bool(mask[a]) else None

    def should_search(self, state: ts.GameState) -> bool:
        return (not ts.Engine.is_terminal(state)
                and state.current_phase == ts.Phase.ACTION_ROUND)

    def search(self, state: ts.GameState) -> Optional[RoundResult]:
        """Search the round `state` is in. None when there is nothing to choose between."""
        t0 = time.time()
        cfg = self.cfg
        if ts.Engine.is_terminal(state):
            return None
        me = acting_player(state)
        p = int(me)
        root = state.clone()
        if cfg.determinize:
            root = determinize(root, me, self._rng)
        # Reseeded so the enumeration never reads the dice the real game is about to throw.
        root.rng_state = self._rng.getrandbits(64) % _UINT64

        dag = self._enumerate(root, p, cfg.p_opp)
        root_node = dag.nodes.get(dag.root)
        if root_node is None or not root_node.actions:
            return None
        dag.v(dag.root, p)
        v_round = {a: dag.v(k, p) for a, k in zip(root_node.actions, root_node.kids)}
        leaves = len(dag.leaf_val)

        ranked = sorted(v_round, key=lambda a: -v_round[a])
        k = cfg.reply_top_k
        cands = ranked if k <= 0 else ranked[:k]
        v_reply: Dict[int, float] = {}
        opp = -p
        if cfg.reply_top_k != 0 and state.phasing_player == me:
            for a in cands:
                seam = self._replay(root, self._line(dag, a))
                # Only a seam that hands the round to the opponent has a reply to look at.
                if (ts.Engine.is_terminal(seam) or seam.current_phase != ts.Phase.ACTION_ROUND
                        or int(seam.phasing_player) != opp):
                    self.reply_skipped += 1
                    continue
                reply = self._enumerate(seam, p, cfg.p_reply)
                v_reply[a] = reply.v(reply.root, p)
                leaves += len(reply.leaf_val)

        # Candidates compared on the deepest value each has. The others were not checked against a
        # reply, so an own-round value would beat a reply-checked one only by being unexamined.
        pool = cands if v_reply or cfg.reply_top_k == 0 else ranked
        best = max(pool, key=lambda a: (v_reply.get(a, v_round[a]), v_round[a]))
        line = self._line(dag, best)
        self._remember(root, line, me)
        self.searches += 1
        return RoundResult(action=int(best), v_round=v_round, v_reply=v_reply, line=line,
                           leaves=leaves, seconds=time.time() - t0)


class RoundSearchAgent:
    """PlayerAgent-compatible wrapper, so `roundsearch:` specs run through the standard CLIs."""

    def __init__(self, model, name: str = "roundsearch", device=None,
                 config: Optional[RoundSearchConfig] = None) -> None:
        self.model = model
        self.name = name
        self.search = RoundSearch(model, device=device, config=config)
        self.world_mismatch_count = 0
        self.searched_count = 0
        self.decision_count = 0
        self.last: Optional[RoundResult] = None

    def _policy_action(self, state: ts.GameState) -> int:
        """The unsearched fallback: this network's greedy policy over the real mask."""
        obs = np.asarray(ts.extract_observation(state, acting_player(state)),
                         dtype=np.float32)[None]
        mask = np.asarray(ActionEncoder.get_legal_mask(state), dtype=np.uint8)[None]
        dev = self.search.device
        with torch.no_grad():
            logits, _, _ = self.model(torch.from_numpy(obs).to(dev), torch.from_numpy(mask).to(dev))
        return int(torch.argmax(logits, dim=-1).item())

    def select_action(self, state: ts.GameState, player=None, temperature: float = 0.1) -> int:
        self.decision_count += 1
        legal = _legal(state)
        if len(legal) == 1:
            return int(legal[0])
        planned = self.search.planned(state)
        if planned is not None:
            self.search.plan_hits += 1
            return planned
        if not self.search.should_search(state):
            return self._policy_action(state)
        res = self.search.search(state)
        self.last = res
        if res is None:
            return self._policy_action(state)
        self.searched_count += 1
        # A determinized world can offer an action the real state does not (Cambridge Five is the
        # worked example in batched_mcts.py): the search proposes and the true mask disposes.
        if not bool(np.asarray(ActionEncoder.get_legal_mask(state))[res.action]):
            self.world_mismatch_count += 1
            return self._policy_action(state)
        return res.action

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        """Sequential: a round is 10^4-10^5 leaves already, so there is no across-position batch
        to gain; the network is batched inside each search."""
        return [self.select_action(st, acting_player(st)) for st in states]

    def reset(self) -> None:
        self.search._rng = random.Random(self.search.cfg.seed)
        self.search._plan.clear()
