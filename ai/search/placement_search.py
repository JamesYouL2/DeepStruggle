"""Search an Ops influence placement as one decision: all its points at once.

Point-by-point search spends its budget badly on influence placement, the segment where search
gains most (research/log/E7_search_segments.md): every point is a tree level, so a 4-Ops placement
is four levels deep before the opponent moves, and the tree has no transposition table -- Poland
then Hungary and Hungary then Poland are two paths to one board, searched separately.

Here, at an influence point with no plan yet:

1. **Candidates.** The network's own point-by-point policy is rolled forward to the end of the
   placement, `placement_samples` times sampled and once greedily, on clones of the real state.
   Placing influence draws no dice and reads no hidden card, so a rollout is exact and leaks
   nothing. Rollouts are deduplicated by the multiset of points (the same points in any order give
   the same board); the greedy placement is always kept, then the most often sampled, up to
   `placement_k`.
2. **Sequential halving.** Each phase samples one world consistent with what the PLACER can see
   (shared by every candidate of that position, so they are compared on the same deal), plays each
   surviving candidate in it, and searches the resulting position with the ordinary batched search
   -- undeterminized, since the world is already sampled. The determinization must be the placer's:
   after the last point the opponent usually moves, and the searcher would otherwise resample from
   the opponent's side and see the placer's opponent's real hand. The better half by mean value
   survives each phase.
3. **The budget** is `simulations` per point of the greedy placement, the same total that searching
   every point at `simulations` spends.
4. **The plan.** The winner's first point is played now; the rest are stored by the observation at
   which each is due and played without search when the game reaches it. A plan whose point is not
   legal there is dropped, and that decision is searched as normal.

Nothing about the network, its action space or a training run changes: the placement is still
played point by point.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, settle
from ai.search.dmcts import determinize
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder

_UINT64 = 1 << 64
_US = int(ts.Player.US)
#: A placement is at most a few points; a rollout still placing after this many is a fault.
_MAX_POINTS = 40
#: Plans abandoned mid-placement (a game ended, a point became illegal) are dropped wholesale past
#: this many, rather than growing without bound over a long tournament.
_MAX_PLANS = 200_000


def is_influence_point(state: ts.GameState) -> bool:
    """Placing an Ops point as influence, outside any event (decision_segment's `ops_influence`)."""
    c = state.ctx()
    return (state.current_phase != ts.Phase.SETUP and int(c.resolving_card) == 0
            and c.decision_type == ts.DecisionType.POINT_NODE and int(c.pending_op_card) != 0
            and c.op_mode == ts.OpMode.INFLUENCE)


def _still_placing(state: ts.GameState, mover: int, card: int) -> bool:
    """Is this the same placement -- same player, same card's Ops -- still asking for a point?"""
    return (not ts.Engine.is_terminal(state) and is_influence_point(state)
            and int(acting_player(state)) == mover and int(state.ctx().pending_op_card) == card)


def _obs_key(state: ts.GameState) -> bytes:
    """What the placer sees here. Placing draws no dice, so the game reaches exactly the
    observation a rollout recorded when it plays the same points."""
    obs = np.asarray(ts.extract_observation(state, acting_player(state)), dtype=np.float32)
    return hashlib.blake2b(obs.tobytes(), digest_size=16).digest()


def _legal(state: ts.GameState, action: int) -> bool:
    mask = np.asarray(ActionEncoder.get_legal_mask(state))
    return 0 <= action < len(mask) and bool(mask[action])


class PlacementSearch:
    """Whole-placement search for `BatchedMCTSAgent` (`BatchedMCTSConfig.placement_k`)."""

    def __init__(self, mcts: BatchedMCTS) -> None:
        self.mcts = mcts
        self.k = int(mcts.cfg.placement_k)
        self.samples = int(mcts.cfg.placement_samples)
        #: observation key -> the planned point due there.
        self.plans: Dict[bytes, int] = {}
        #: Undeterminized searchers by simulation count, sharing the model.
        self._subs: Dict[int, BatchedMCTS] = {}
        self.planned_count = 0
        self.placements_searched = 0

    # -- the plan ------------------------------------------------------------------------------

    def planned(self, state: ts.GameState) -> Optional[int]:
        """The stored point for this exact position, if it is still legal here."""
        if not self.plans or not is_influence_point(state):
            return None
        a = self.plans.pop(_obs_key(state), None)
        if a is None or not _legal(state, a):
            return None
        self.planned_count += 1
        return a

    def reset(self) -> None:
        self.plans.clear()

    # -- candidates ----------------------------------------------------------------------------

    def _rollouts(self, states: Sequence[ts.GameState]
                  ) -> List[List[Tuple[List[bytes], List[int]]]]:
        """Per position: (keys, points) of 1 greedy + `samples` sampled placements."""
        mcts = self.mcts
        runs: List[Tuple[int, int, int, ts.GameState, List[bytes], List[int], bool]] = []
        for i, st in enumerate(states):
            mover, card = int(acting_player(st)), int(st.ctx().pending_op_card)
            for j in range(self.samples + 1):
                runs.append((i, mover, card, st.clone(), [], [], j == 0))
        for _ in range(_MAX_POINTS):
            live = [r for r in runs if _still_placing(r[3], r[1], r[2])]
            if not live:
                break
            obs = np.stack([np.asarray(ts.extract_observation(r[3], acting_player(r[3])),
                                       dtype=np.float32) for r in live])
            masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(r[3]), dtype=np.uint8)
                              for r in live])
            with torch.no_grad():
                logits, _v, _vp = mcts.model(torch.from_numpy(obs).to(mcts.device),
                                             torch.from_numpy(masks).to(mcts.device))
                probs = torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)
            for row, r in enumerate(live):
                p = probs[row] * (masks[row] > 0)
                if r[6] or p.sum() <= 0:
                    a = int(np.argmax(p)) if p.sum() > 0 else int(np.flatnonzero(masks[row])[0])
                else:
                    a = int(mcts._np_rng.choice(len(p), p=p / p.sum()))
                r[4].append(_obs_key(r[3]))
                r[5].append(a)
                ts.Engine.step_flat(r[3], a)
                settle(r[3], mcts.cfg.auto_advance)
        out: List[List[Tuple[List[bytes], List[int]]]] = [[] for _ in states]
        for r in runs:
            if r[5]:
                out[r[0]].append((r[4], r[5]))
        return out

    def _candidates(self, rolls: List[Tuple[List[bytes], List[int]]]
                    ) -> List[Tuple[List[bytes], List[int]]]:
        """The greedy placement first, then the most often sampled others, distinct as multisets."""
        if not rolls:
            return []
        ident = [tuple(sorted(pts)) for _keys, pts in rolls]
        counts = Counter(ident[1:])
        chosen: List[Tuple[List[bytes], List[int]]] = [rolls[0]]
        seen = {ident[0]}
        first: Dict[tuple, int] = {}
        for j, idn in enumerate(ident):
            first.setdefault(idn, j)
        for idn, _c in sorted(counts.items(), key=lambda kv: (-kv[1], first[kv[0]])):
            if len(chosen) >= self.k:
                break
            if idn not in seen:
                seen.add(idn)
                chosen.append(rolls[first[idn]])
        return chosen

    # -- evaluation ----------------------------------------------------------------------------

    def _sub(self, sims: int) -> BatchedMCTS:
        sub = self._subs.get(sims)
        if sub is None:
            cfg = replace(self.mcts.cfg, simulations=sims, determinize=False, node_filter="all",
                          subsample=1.0, placement_k=0, reuse_subtree=False, dirichlet_frac=0.0)
            sub = BatchedMCTS(self.mcts.model, device=self.mcts.device, config=cfg,
                              featurise_capacity=self.mcts._featurise_capacity)
            self._subs[sims] = sub
        sub._rng.seed(self.mcts._rng.getrandbits(64))
        return sub

    def choose(self, states: Sequence[ts.GameState]) -> List[int]:
        """The first point of the best placement for each position; the rest become plans."""
        mcts, cfg = self.mcts, self.mcts.cfg
        states = list(states)
        movers = [int(acting_player(st)) for st in states]
        cands = [self._candidates(r) for r in self._rollouts(states)]
        budget = [cfg.simulations * max(1, len(c[0][1])) if c else 0 for c in cands]
        alive = [list(range(len(c))) for c in cands]
        sum_v = [[0.0] * len(c) for c in cands]
        sum_n = [[0.0] * len(c) for c in cands]
        phases = [max(1, math.ceil(math.log2(len(c)))) if len(c) > 1 else 0 for c in cands]

        for ph in range(max(phases, default=0)):
            # (position, candidate, post-placement state, simulations) for this phase.
            jobs: List[Tuple[int, int, ts.GameState, int]] = []
            for i, st in enumerate(states):
                if ph >= phases[i] or len(alive[i]) < 2:
                    continue
                world = st.clone()
                if cfg.determinize:
                    world = determinize(world, ts.Player(movers[i]), mcts._rng)
                world.rng_state = mcts._rng.getrandbits(64) % _UINT64
                sims = max(1, budget[i] // (phases[i] * len(alive[i])))
                for c in alive[i]:
                    s = world.clone()
                    ok = True
                    for a in cands[i][c][1]:
                        if not _legal(s, a):
                            ok = False
                            break
                        ts.Engine.step_flat(s, a)
                        settle(s, cfg.auto_advance)
                    if ok:
                        jobs.append((i, c, s, sims))
            for sims in sorted({j[3] for j in jobs}):
                group = [j for j in jobs if j[3] == sims]
                roots = self._sub(sims)._search([j[2] for j in group])
                for (i, c, _s, n), r in zip(group, roots):
                    if r is None:
                        continue
                    if r.terminal or not r.actions:
                        v_us = float(r.value_us)
                    else:
                        v_us = (float(r.value_us) + float(sum(r.w))) / (1.0 + float(sum(r.n)))
                    v = v_us if movers[i] == _US else -v_us
                    sum_v[i][c] += v * n
                    sum_n[i][c] += n
            for i in range(len(states)):
                if ph >= phases[i] or len(alive[i]) < 2:
                    continue
                mean = {c: (sum_v[i][c] / sum_n[i][c] if sum_n[i][c] > 0 else -math.inf)
                        for c in alive[i]}
                # Stable: on equal values the earlier candidate (the greedy one first) survives.
                ranked = sorted(alive[i], key=lambda c: -mean[c])
                alive[i] = ranked[:max(1, (len(ranked) + 1) // 2)] if ph < phases[i] - 1 \
                    else ranked[:1]

        out: List[int] = []
        if len(self.plans) > _MAX_PLANS:
            self.plans.clear()
        for i, st in enumerate(states):
            if not cands[i]:
                out.append(-1)
                continue
            keys, pts = cands[i][alive[i][0]]
            for key, a in zip(keys[1:], pts[1:]):
                self.plans[key] = a
            out.append(pts[0])
            self.placements_searched += 1
        return out
