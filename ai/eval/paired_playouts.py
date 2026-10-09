"""Paired playouts: what a forced move is worth, by the model's own play afterwards.

Ported from the fork's `ai/eval/branch_oracle.py` and `ai/eval/playout_audit.py`
(`exp/hungary-openings`), the instrument of the E7 expert review (`research/log/expert_review_E7.md`).

To compare moves at a position, each is applied at the start of every *pair*; pair k first redeals
the cards the mover cannot see (`ai.search.dmcts.determinize`, from the mover's side) and fixes the
dice seed, so all branches of one pair face the same hidden hands and the same dice. The model then
plays both sides to the end. Through the rest of the mover's action round an option that loses on
the spot is never taken while another exists (`play_safe`): a forced event's inner choices are
decisions the model rarely reached in training, and a suicide there would score the branch, not
the move.

**What a verdict means.** A branch is valued by how *this model* plays on from it. A move whose
payoff lies in a follow-up the model does not find reads worse than it is, so a human move that
wins a paired comparison is the stronger statement.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.search.dmcts import determinize
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

#: (observations, masks) -> one greedy action per row
PolicyFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
#: states -> one action per state (a searcher's `select_actions_batch`)
StateSelector = Callable[[Sequence[ts.GameState]], List[int]]
_UINT64 = 1 << 64


def decider(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def ar_key(st: ts.GameState) -> Tuple[int, int, int, int]:
    return (int(st.turn), int(st.action_round), int(st.phasing_player), int(st.current_phase))


def pair_start(state: ts.GameState, k: int, seed: int) -> ts.GameState:
    """Pair k's world: the mover's unseen cards redealt, and its own dice."""
    rng = random.Random(seed * 1_000_003 + k)
    st = determinize(state.clone(), decider(state), rng)
    st.rng_state = rng.getrandbits(64) % _UINT64
    return st


def apply_move(state: ts.GameState, action: int) -> ts.GameState:
    """A copy with `action` applied and chance resolved; refuses an illegal action."""
    st = state.clone()
    if not np.asarray(ActionEncoder.get_legal_mask(st))[action]:
        raise ValueError(f"action {action} is illegal here")
    ts.Engine.step_flat(st, int(action))
    drain_chance(st, context="paired_playouts move")
    return st


def loses_now(st: ts.GameState, a: int, mover: ts.Player) -> bool:
    probe = st.clone()
    ts.Engine.step_flat(probe, int(a))
    drain_chance(probe, context="paired_playouts safety probe")
    if not ts.Engine.is_terminal(probe):
        return False
    u = float(ts.Engine.get_terminal_utility(probe))
    return (u < 0) if mover == ts.Player.US else (u > 0)


def play_safe(starts: Sequence[ts.GameState], movers: Sequence[ts.Player],
              keys: Sequence[Tuple[int, int, int, int]], act: PolicyFn, seed: int,
              max_steps: int = 4000, select: Optional[StateSelector] = None) -> List[float]:
    """Play every state to the end with `act` on both sides, in one batch, never taking an
    option that loses on the spot while another exists during the mover's action round (`keys`,
    read before the forced move). The mover's score per state: 1 win, 0.5 draw, 0 loss. `select`,
    if given, chooses instead of `act` (a searched continuation); where its choice is outside a
    row's mask as the guard narrowed it, `act` chooses over that mask."""
    n = len(starts)
    runner = ts.VectorizedBatchRunner(n, seed)
    for i, st in enumerate(starts):
        runner.set_state(i, st)
    runner.refresh_all()
    guarding = [True] * n
    active = np.ones(n, dtype=bool)
    for _ in range(max_steps):
        active &= ~np.array(runner.get_terminals())
        if not active.any():
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks()).copy()
        rows = np.flatnonzero(active)
        for i in (int(x) for x in rows):
            if not guarding[i]:
                continue
            st = runner.get_state(i)
            if ar_key(st) != keys[i] or decider(st) != movers[i]:
                guarding[i] = False
                continue
            if int(st.defcon) > 2:
                continue               # nothing one decision does loses on the spot above DEFCON 2
            legal = np.flatnonzero(masks[i])
            safe = [a for a in legal if not loses_now(st, int(a), movers[i])]
            if safe and len(safe) < len(legal):
                masks[i] = 0
                masks[i, safe] = 1
        acts = np.zeros(n, dtype=np.int32)
        if select is None:
            acts[rows] = act(obs[rows], masks[rows])
        else:
            acts[rows] = select([runner.get_state(int(i)) for i in rows])
            off = rows[masks[rows, acts[rows]] == 0]
            if off.size:
                acts[off] = act(obs[off], masks[off])
        res = runner.step_flat_all(acts.tolist(), auto_advance=True)
        refused = [int(i) for i in rows if res[i] == 0]
        if refused:
            raise RuntimeError(f"the engine refused a playout action in game {refused[0]}")
    active &= ~np.array(runner.get_terminals())
    if active.any():
        # Scoring an unfinished game would read a utility the engine has not decided.
        raise RuntimeError(f"{int(active.sum())} playouts had not ended after {max_steps} steps")
    out = []
    for i in range(n):
        u = float(ts.Engine.get_terminal_utility(runner.get_state(i)))
        u = u if movers[i] == ts.Player.US else -u
        out.append(1.0 if u > 0 else (0.5 if u == 0 else 0.0))
    return out


def compare(positions: Sequence[Tuple[ts.GameState, Sequence[int]]], act: PolicyFn, pairs: int,
            seeds: Sequence[int], chunk: int = 1536,
            select: Optional[StateSelector] = None) -> List[Dict[int, List[float]]]:
    """Per position, every move's score in each pair: {action: [score of pair 0, 1, ...]}. All
    moves of one pair share the redeal and the dice, so differences are paired. `seeds` holds one
    seed per position, so a position's pairs do not depend on what it was batched with -- a
    seed from its index would give two positions in two runs the same redeals. `select` plays
    the continuation instead of `act` (`play_safe`)."""
    if len(seeds) != len(positions):
        raise ValueError(f"{len(seeds)} seeds for {len(positions)} positions")
    out: List[Dict[int, List[float]]] = []
    per_pos = [max(1, len(moves)) * pairs for _, moves in positions]
    lo = 0
    while lo < len(positions):
        hi, size = lo, 0
        while hi < len(positions) and (size == 0 or size + per_pos[hi] <= chunk):
            size += per_pos[hi]
            hi += 1
        starts, movers, keys, index = [], [], [], []
        for p in range(lo, hi):
            st, moves = positions[p]
            for k in range(pairs):
                base = pair_start(st, k, seeds[p])
                for a in moves:
                    starts.append(apply_move(base, a))
                    movers.append(decider(st))
                    keys.append(ar_key(st))
                    index.append((p, a))
        scores = play_safe(starts, movers, keys, act, seeds[lo], select=select)
        res: List[Dict[int, List[float]]] = [{a: [] for a in positions[p][1]} for p in range(lo, hi)]
        for (p, a), s in zip(index, scores):
            res[p - lo][a].append(s)
        out += res
        lo = hi
    return out


def paired_diff(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Mean and standard error of a − b over the pairs."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    if len(d) < 2:
        return float(d.mean()) if len(d) else math.nan, math.nan
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))
