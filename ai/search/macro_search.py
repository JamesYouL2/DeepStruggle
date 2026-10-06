"""Search over complete player decisions (macro-actions) instead of the 220-slot micro-actions.

The question this exists to answer: is the search's weakness the *representation* it branches
over, rather than the network? `BatchedMCTS` branches one micro-action at a time -- card, play
mode, op mode, then each Ops point as its own tree level -- so a 4-Ops placement is five levels
deep before the opponent moves, and the tree has no transpositions. Here the network is frozen and
only the representation changes:

1. **Candidates (`MacroGenerator`).** The network's own policy is rolled forward as a beam until
   each line reaches the end of the current player decision (`macro_boundary`), giving complete
   macros such as "card Z for Ops: Thailand 2, Malaysia 1" or "card X for the event". Each keeps
   the policy mass of every line that reaches it (log probability), lines reaching the same result
   are merged, the network's greedy line is always kept, and a share of the slots is reserved for
   distinct (first two decisions) groups -- in an action round, distinct card x mode -- so the
   candidates are not sixteen orderings of one placement.
2. **Search (`MacroSearch`).** Root candidates, then the next decider's candidates, to `depth`
   macro plies, scored by the frozen value head (minimax: the US maximises its value, the USSR
   minimises it). Honest: each of `worlds` worlds is sampled from the root decider's side, and
   every candidate is played and searched in it, so no leaf sees the opponent's real hand.
3. **Leaf horizon (`rollout_ars`).** Optionally the network plays on greedily from each leaf for
   this many action rounds (one player's turn each) before the value head is read.

A macro ends when (`macro_boundary`):

* the game ends;
* someone else must decide (the opponent's turn, or an opponent's choice inside an event);
* the step drew from the game's random stream -- a coup, a realignment, a die-rolling event, a
  card draw. The macro includes that step, so its dice are rolled afresh in each sampled world and
  the search averages over them rather than trusting the one roll a rollout happened to make; or
* the turn, action round or phase changed, or the step was a headline card choice (a headline is
  the card alone: its event's choices follow the opponent's headline being revealed).

Candidates are generated on the real state. That is honest: the policy reads only the decider's
observation, and the legal masks along a line are the ones the real game would offer the decider
on the way. Values are only ever read in sampled worlds.

Nothing about the network, its action space or a training run changes: the chosen macro is played
point by point, each later step stored as a plan keyed by the decider's observation where it falls
due (as `placement_search.py`), and a plan step that is not legal when it falls due is dropped.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.batched_mcts import settle
from ai.search.dmcts import determinize
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder

_UINT64 = 1 << 64
_US = int(ts.Player.US)
#: A macro is at most an event's worth of points; a line still going after this many is dropped.
MAX_MACRO_STEPS = 48
#: A greedy rollout still in the same game after this many decisions is a fault.
_MAX_ROLLOUT_STEPS = 2000
#: Plans abandoned mid-macro (a game ended, a step became illegal) are dropped wholesale past this.
_MAX_PLANS = 200_000


@dataclass
class MacroSearchConfig:
    #: Candidates at the root.
    k: int = 16
    #: Candidates at every deeper ply.
    k_reply: int = 8
    #: Macro plies searched: 1 = our candidates, then the value head; 2 adds the next decider's
    #: reply (usually the opponent); 3 adds our answer to that.
    depth: int = 1
    #: Unfinished lines kept per position after each beam step (at most twice the candidates
    #: wanted at that ply).
    beam: int = 32
    #: Children expanded per line per step (the most probable legal actions).
    branch: int = 6
    #: A child below this policy probability is not expanded (the argmax always is).
    min_prob: float = 0.01
    #: The same two settings for the first two decisions of a macro (breadth at most the
    #: candidates wanted at that ply) (card and play mode in an
    #: action round), where the network is most confident and breadth matters most: at `branch`
    #: and `min_prob` a typical action round offered only two cards.
    branch_first: int = 16
    min_prob_first: float = 1e-4
    #: The move is argmax(Q + prior_weight * log p). 0, the default, is the value alone -- the
    #: experiment's question; above 0 guards against picking the luckiest of many noisy values.
    prior_weight: float = 0.0
    #: Share of the candidates (and of the beam) reserved for the best line of each distinct
    #: group -- the first two decisions, i.e. card and play mode in an action round.
    diverse: float = 0.5
    #: Worlds sampled from the root decider's side; a candidate's value is its mean over them.
    worlds: int = 4
    #: Before reading the value head at a leaf, the network plays on greedily for this many
    #: action rounds (one player's turn each). 0 reads the value at the leaf itself.
    rollout_ars: int = 0
    #: Resample hidden cards per world. False searches the real state: privileged, for diagnosis
    #: only, never a deployable player.
    determinize: bool = True
    auto_advance: bool = True
    seed: int = 12345


@dataclass
class Macro:
    """One complete candidate decision."""

    actions: List[int]
    #: The decider's observation key at each step, so the later steps can be replayed as a plan.
    keys: List[bytes]
    #: log of the policy mass of every line that reaches this result.
    logp: float
    #: The first two decisions: card and play mode in an action round.
    group: Tuple[int, ...]
    #: The network's own argmax line.
    greedy: bool
    #: What identifies the result (`_result_key`).
    result: bytes
    #: Position among the position's candidates by `logp` (0 = most probable).
    rank: int = 0


@dataclass
class _Line:
    root: int
    state: ts.GameState
    mover: int
    actions: List[int] = field(default_factory=list)
    keys: List[bytes] = field(default_factory=list)
    logp: float = 0.0
    group: Tuple[int, ...] = ()
    greedy: bool = True
    done: bool = False
    drew: bool = False


def _round(state: ts.GameState) -> Tuple[int, int, int]:
    return int(state.turn), int(state.action_round), int(state.current_phase)


def _stage(state: ts.GameState) -> Tuple[int, int, int, int]:
    """Where in the game a decision falls, for `macro_boundary`. The last element marks a headline
    card choice, so a headline macro is the choice alone: the headline event's own choices come
    after the opponent's headline is revealed, which a sampled world would not reproduce."""
    c = state.ctx()
    pick = int(state.current_phase == ts.Phase.HEADLINE and c.decision_type == ts.DecisionType.SELECT_CARD
               and int(c.resolving_card) == 0)
    return int(state.turn), int(state.action_round), int(state.current_phase), pick


def macro_boundary(before_stage: Tuple[int, int, int, int], before_rng: int, mover: int,
                   after: ts.GameState) -> Tuple[bool, bool]:
    """(the macro ends here, the step drew from the random stream)."""
    drew = int(after.rng_state) != before_rng
    if ts.Engine.is_terminal(after):
        return True, drew
    return (drew or int(acting_player(after)) != mover or _stage(after) != before_stage), drew


def obs_key(obs_row: np.ndarray) -> bytes:
    return hashlib.blake2b(np.ascontiguousarray(obs_row, dtype=np.float32).tobytes(),
                           digest_size=16).digest()


def _result_key(line: _Line, obs_after: np.ndarray) -> bytes:
    """What makes two lines the same decision. Without a draw, the decider's view of the result:
    only the decider's own deterministic choices were applied, so equal views are equal states.
    With a draw the result includes one roll, so two different coups could show the same board;
    then the decision is the set of choices plus the one that rolled."""
    if line.drew:
        ident = repr((sorted(line.actions[:-1]), line.actions[-1])).encode()
        return b"d" + hashlib.blake2b(ident, digest_size=16).digest()
    return b"o" + obs_key(obs_after)


def _logsumexp(a: float, b: float) -> float:
    m = max(a, b)
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def _diverse_top(items: Sequence[_Line], n: int, diverse: float) -> List[_Line]:
    """Up to `n` lines: the greedy line, then the best line of each group (most probable groups
    first) for `diverse` of the slots, then the rest by probability."""
    if n <= 0 or not items:
        return []
    by_p = sorted(items, key=lambda ln: -ln.logp)
    out: List[_Line] = [ln for ln in by_p if ln.greedy][:1]
    taken = {id(ln) for ln in out}
    groups_seen = {ln.group for ln in out}
    reserved = max(len(out), int(math.ceil(n * diverse)))
    for ln in by_p:
        if len(out) >= min(n, reserved):
            break
        if ln.group not in groups_seen:
            groups_seen.add(ln.group)
            out.append(ln)
            taken.add(id(ln))
    for ln in by_p:
        if len(out) >= n:
            break
        if id(ln) not in taken:
            out.append(ln)
            taken.add(id(ln))
    return out


class _Net:
    """The frozen network: masked log-policy and value for batches of states, counting the rows it
    evaluates so arms can be compared at equal cost."""

    def __init__(self, model: torch.nn.Module, device: torch.device) -> None:
        from bindings.ts_env import model_obs_features
        self.model = model
        self.device = device
        self.features = int(model_obs_features(model))
        self.rows = 0

    def featurise(self, states: Sequence[ts.GameState]) -> Tuple[np.ndarray, np.ndarray]:
        obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), self.features),
                                   dtype=np.float32) for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        return obs, masks

    def forward(self, obs: np.ndarray, masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(log-policy over the legal actions, -inf elsewhere; value from the decider's side)."""
        self.rows += len(obs)
        with torch.no_grad():
            logits, v, _ = self.model(torch.from_numpy(obs).to(self.device),
                                      torch.from_numpy(masks).to(self.device))
            logits = logits.float().masked_fill(torch.from_numpy(masks).to(self.device) == 0,
                                                float("-inf"))
            logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
        return logp, v.float().reshape(-1).cpu().numpy()

    def values_us(self, states: Sequence[ts.GameState]) -> List[float]:
        """The value head, from the US side; a finished game's real result."""
        out = [0.0] * len(states)
        live = []
        for i, s in enumerate(states):
            if ts.Engine.is_terminal(s):
                out[i] = float(ts.Engine.get_terminal_utility(s))
            else:
                live.append(i)
        if live:
            obs, masks = self.featurise([states[i] for i in live])
            _, v = self.forward(obs, masks)
            for i, val in zip(live, v.tolist()):
                out[i] = val if int(acting_player(states[i])) == _US else -val
        return out


class MacroGenerator:
    """Step 1: the network's top complete decisions for a batch of positions, by beam search."""

    def __init__(self, net: _Net, cfg: MacroSearchConfig) -> None:
        self.net = net
        self.cfg = cfg

    def candidates(self, states: Sequence[ts.GameState], k: int) -> List[List[Macro]]:
        cfg = self.cfg
        # The beam and the first decisions' breadth scale with how many candidates are wanted, so
        # a reply ply asking for 8 does not pay for the root's 16.
        beam = max(1, min(cfg.beam, 2 * k))
        width_first = max(1, min(cfg.branch_first, k))
        lines: List[_Line] = [_Line(root=i, state=s.clone(), mover=int(acting_player(s)))
                              for i, s in enumerate(states) if not ts.Engine.is_terminal(s)]
        finished: List[Dict[bytes, _Line]] = [{} for _ in states]
        for _ in range(MAX_MACRO_STEPS):
            if not lines:
                break
            obs, masks = self.net.featurise([ln.state for ln in lines])
            logp, _ = self.net.forward(obs, masks)
            children: List[_Line] = []
            for row, ln in enumerate(lines):
                lp = logp[row]
                legal = np.flatnonzero(masks[row])
                if legal.size == 0:
                    continue
                order = legal[np.argsort(-lp[legal], kind="stable")]
                key = obs_key(obs[row])
                stage, rng = _stage(ln.state), int(ln.state.rng_state)
                first = len(ln.group) < 2
                width = width_first if first else cfg.branch
                floor = cfg.min_prob_first if first else cfg.min_prob
                for j, a in enumerate(order[:max(1, width)]):
                    a = int(a)
                    if j > 0 and math.exp(float(lp[a])) < floor:
                        break
                    s = ln.state.clone()
                    ts.Engine.step_flat(s, a)
                    settle(s, cfg.auto_advance)
                    done, drew = macro_boundary(stage, rng, ln.mover, s)
                    children.append(_Line(
                        root=ln.root, state=s, mover=ln.mover, actions=ln.actions + [a],
                        keys=ln.keys + [key], logp=ln.logp + float(lp[a]),
                        group=ln.group + (a,) if len(ln.group) < 2 else ln.group,
                        greedy=ln.greedy and j == 0, done=done, drew=drew))
            done_lines = [c for c in children if c.done]
            if done_lines:
                obs_after = np.stack([np.asarray(ts.extract_observation_features(
                    c.state, ts.Player(c.mover), self.net.features), dtype=np.float32)
                    for c in done_lines])
                for c, row in zip(done_lines, obs_after):
                    rk = _result_key(c, row)
                    have = finished[c.root].get(rk)
                    if have is None:
                        finished[c.root][rk] = c
                    else:
                        # Merged: the better line represents the result, carrying both masses.
                        best = c if c.logp > have.logp else have
                        best.greedy = have.greedy or c.greedy
                        best.logp = _logsumexp(have.logp, c.logp)
                        finished[c.root][rk] = best
            # Prune the live lines per position: greedy kept, groups spread, then probability.
            per_root: Dict[int, List[_Line]] = {}
            for c in children:
                if not c.done:
                    per_root.setdefault(c.root, []).append(c)
            lines = []
            for group in per_root.values():
                lines.extend(_diverse_top(group, beam, cfg.diverse))
        out: List[List[Macro]] = []
        for res in finished:
            # Merging can leave a result whose representative line is not the greedy one, so pick
            # first, then rank by probability.
            pick = _diverse_top(list(res.values()), k, self.cfg.diverse)
            keys = {id(ln): rk for rk, ln in res.items()}
            ranked = sorted(res.values(), key=lambda ln: -ln.logp)
            rank = {id(ln): r for r, ln in enumerate(ranked)}
            ms = [Macro(actions=ln.actions, keys=ln.keys, logp=ln.logp, group=ln.group,
                        greedy=ln.greedy, result=keys[id(ln)], rank=rank[id(ln)]) for ln in pick]
            ms.sort(key=lambda m: m.rank)
            out.append(ms)
        return out


def greedy_macro(net: _Net, state: ts.GameState, auto_advance: bool = True) -> Tuple[List[int], bytes]:
    """The network's own argmax line from `state` to the end of the macro, and its result key."""
    s = state.clone()
    mover = int(acting_player(s))
    ln = _Line(root=0, state=s, mover=mover)
    for _ in range(MAX_MACRO_STEPS):
        obs, masks = net.featurise([s])
        logp, _ = net.forward(obs, masks)
        a = int(np.argmax(logp[0]))
        stage, rng = _stage(s), int(s.rng_state)
        ts.Engine.step_flat(s, a)
        settle(s, auto_advance)
        ln.actions.append(a)
        done, drew = macro_boundary(stage, rng, mover, s)
        ln.drew = drew
        if done:
            break
    after = np.asarray(ts.extract_observation_features(s, ts.Player(mover), net.features),
                       dtype=np.float32)
    return ln.actions, _result_key(ln, after)


@dataclass
class _Node:
    state: ts.GameState
    children: List["_Node"] = field(default_factory=list)
    value_us: float = 0.0


class MacroSearch:
    """Step 2 (and 3): choose among the root's macro candidates by a shallow macro-ply search."""

    def __init__(self, model: torch.nn.Module, device: torch.device,
                 cfg: Optional[MacroSearchConfig] = None) -> None:
        self.cfg = cfg or MacroSearchConfig()
        self.net = _Net(model, device)
        self.gen = MacroGenerator(self.net, self.cfg)
        self.reseed(self.cfg.seed)
        #: Instrumentation: the chosen candidate's policy rank, and how many there were.
        self.chosen_ranks: List[int] = []
        self.candidate_counts: List[int] = []
        #: Root candidates that could not be played in a sampled world (legal set differs).
        self.world_misses = 0

    def reseed(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def _replay(self, world: ts.GameState, actions: Sequence[int]) -> Optional[ts.GameState]:
        s = world.clone()
        for a in actions:
            if not np.asarray(ActionEncoder.get_legal_mask(s))[a]:
                return None
            ts.Engine.step_flat(s, a)
            settle(s, self.cfg.auto_advance)
        return s

    def _rollout(self, states: List[ts.GameState]) -> List[ts.GameState]:
        """The network plays on greedily for `rollout_ars` action rounds from each state."""
        n = self.cfg.rollout_ars
        if n <= 0:
            return states
        cur = [s.clone() for s in states]
        left = [n] * len(cur)
        stage = [(_round(s), int(s.phasing_player)) for s in cur]
        for _ in range(_MAX_ROLLOUT_STEPS):
            live = [i for i, s in enumerate(cur) if left[i] > 0 and not ts.Engine.is_terminal(s)]
            if not live:
                break
            obs, masks = self.net.featurise([cur[i] for i in live])
            logp, _ = self.net.forward(obs, masks)
            for row, i in enumerate(live):
                ts.Engine.step_flat(cur[i], int(np.argmax(logp[row])))
                settle(cur[i], self.cfg.auto_advance)
                now = (_round(cur[i]), int(cur[i].phasing_player))
                if now != stage[i]:
                    stage[i] = now
                    left[i] -= 1
        return cur

    def _expand(self, nodes: List[_Node], plies_left: int) -> None:
        """Give each non-terminal node its decider's candidates, `plies_left` plies deep."""
        if plies_left <= 0:
            return
        open_ = [nd for nd in nodes if not ts.Engine.is_terminal(nd.state)]
        if not open_:
            return
        # Each candidate is played on the node's own (sampled) state, so its dice are that
        # world's; the candidates are rebuilt rather than reused across worlds.
        cands = self.gen.candidates([nd.state for nd in open_], self.cfg.k_reply)
        kids: List[_Node] = []
        for nd, ms in zip(open_, cands):
            for m in ms:
                s = self._replay(nd.state, m.actions)
                if s is not None:
                    child = _Node(state=s)
                    nd.children.append(child)
                    kids.append(child)
        self._expand(kids, plies_left - 1)

    def _leaves(self, nodes: Sequence[_Node]) -> List[_Node]:
        out: List[_Node] = []
        for nd in nodes:
            if nd.children:
                out.extend(self._leaves(nd.children))
            else:
                out.append(nd)
        return out

    @staticmethod
    def _backup(nd: _Node) -> float:
        if not nd.children:
            return nd.value_us
        vals = [MacroSearch._backup(c) for c in nd.children]
        nd.value_us = max(vals) if int(acting_player(nd.state)) == _US else min(vals)
        return nd.value_us

    def search(self, states: Sequence[ts.GameState]) -> List[Tuple[List[Macro], List[float]]]:
        """Per position: its root candidates and each one's mean value for the root decider
        (-inf where it could not be played in any world)."""
        cfg = self.cfg
        states = list(states)
        cands = self.gen.candidates(states, cfg.k)
        movers = [int(acting_player(s)) for s in states]
        # (position, candidate, world) -> the node after that candidate in that world.
        roots: List[Tuple[int, int, _Node]] = []
        for i, st in enumerate(states):
            if len(cands[i]) < 2:
                continue
            for _w in range(max(1, cfg.worlds)):
                world = st.clone()
                if cfg.determinize:
                    world = determinize(world, ts.Player(movers[i]), self._rng)
                world.rng_state = self._rng.getrandbits(64) % _UINT64
                for c, m in enumerate(cands[i]):
                    s = self._replay(world, m.actions)
                    if s is None:
                        self.world_misses += 1
                        continue
                    roots.append((i, c, _Node(state=s)))
        self._expand([nd for _, _, nd in roots], cfg.depth - 1)
        leaves = self._leaves([nd for _, _, nd in roots])
        ends = self._rollout([lf.state for lf in leaves])
        for lf, v in zip(leaves, self.net.values_us(ends)):
            lf.value_us = v
        sums = [[0.0] * len(c) for c in cands]
        cnts = [[0] * len(c) for c in cands]
        for i, c, nd in roots:
            v = self._backup(nd)
            sums[i][c] += v if movers[i] == _US else -v
            cnts[i][c] += 1
        out: List[Tuple[List[Macro], List[float]]] = []
        for i in range(len(states)):
            q = [sums[i][c] / cnts[i][c] if cnts[i][c] else -math.inf for c in range(len(cands[i]))]
            out.append((cands[i], q))
        return out

    def choose(self, states: Sequence[ts.GameState]) -> List[Optional[Macro]]:
        """The best candidate per position (by mean value, ties to the more probable); None for a
        position with no candidate."""
        picks: List[Optional[Macro]] = []
        for ms, q in self.search(states):
            if not ms:
                picks.append(None)
                continue
            if len(ms) == 1:
                best = 0
            else:
                w = self.cfg.prior_weight
                best = max(range(len(ms)), key=lambda c: (q[c] + w * ms[c].logp, -ms[c].rank))
            picks.append(ms[best])
            self.chosen_ranks.append(ms[best].rank)
            self.candidate_counts.append(len(ms))
        return picks


class MacroSearchAgent:
    """A tournament player (`macro:` spec): searches at the start of each macro and plays the
    chosen macro's later steps as a plan."""

    def __init__(self, model: torch.nn.Module, name: str = "macro", device: Optional[torch.device] = None,
                 config: Optional[MacroSearchConfig] = None) -> None:
        self.model = model
        self.name = name
        dev = device if device is not None else torch.device("cpu")
        self.search = MacroSearch(model, dev, config)
        self.plans: Dict[bytes, int] = {}
        self.searched_count = 0
        self.planned_count = 0
        self.decision_count = 0

    def reseed(self, seed: int) -> None:
        self.search.reseed(seed)

    def reset(self) -> None:
        self.plans.clear()
        self.search.reseed(self.search.cfg.seed)

    def _key(self, state: ts.GameState) -> bytes:
        return obs_key(np.asarray(ts.extract_observation_features(
            state, acting_player(state), self.search.net.features), dtype=np.float32))

    def select_actions_batch(self, states: Sequence[ts.GameState]) -> List[int]:
        states = list(states)
        out = [-1] * len(states)
        masks = [np.asarray(ActionEncoder.get_legal_mask(s)) for s in states]
        todo: List[int] = []
        for i, s in enumerate(states):
            legal = np.flatnonzero(masks[i])
            if legal.size == 1:
                out[i] = int(legal[0])
                continue
            a = self.plans.pop(self._key(s), None) if self.plans else None
            if a is not None and 0 <= a < len(masks[i]) and masks[i][a]:
                out[i] = int(a)
                self.planned_count += 1
                continue
            todo.append(i)
        if todo:
            if len(self.plans) > _MAX_PLANS:
                self.plans.clear()
            picks = self.search.choose([states[i] for i in todo])
            for i, m in zip(todo, picks):
                if m is None:
                    continue
                out[i] = int(m.actions[0])
                for key, a in zip(m.keys[1:], m.actions[1:]):
                    self.plans[key] = a
            self.searched_count += len(todo)
        for i, a in enumerate(out):
            if a < 0 or not masks[i][a]:
                # Nothing legal chosen: the network's argmax over the real mask.
                obs, mk = self.search.net.featurise([states[i]])
                logp, _ = self.search.net.forward(obs, mk)
                out[i] = int(np.argmax(logp[0]))
        self.decision_count += len(states)
        return out

    def select_action(self, state: ts.GameState, player: ts.Player, temperature: float = 0.1) -> int:
        return self.select_actions_batch([state])[0]
