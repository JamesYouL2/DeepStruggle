"""P27: the influence-play enumerator finds every distinct end board (checked against an
unrestricted search on small plays), and the report's measures follow from the boards."""

from __future__ import annotations

from typing import List, Set

import numpy as np
import ts_engine as ts

from ai.eval.ops_block import (CONFIRM_DONE, NODE_OFFSET, N_COUNTRIES, _ctx, _in_block, analyse,
                               country_table, enumerate_block, influence, is_block_start,
                               play_block)
from bindings.ts_env import TsVectorizedEnv


def _starts(n: int, max_ops: int, seed: int = 11) -> List["ts.GameState"]:
    env = TsVectorizedEnv(num_envs=1, base_seed=seed)
    obs, masks, info = env.reset_all()
    rng = np.random.default_rng(0)
    out: List["ts.GameState"] = []
    for _ in range(6000):
        st = env.runner.get_state(0)
        if is_block_start(st) and int(_ctx(st).pending_ops_value) <= max_ops:
            out.append(st.clone())
            if len(out) >= n:
                break
        a = rng.choice(np.flatnonzero(np.asarray(masks)[0]))
        obs, masks, *_ = env.step(np.array([a]))
    assert len(out) >= n
    return out


def _all_orders(start: "ts.GameState") -> Set[bytes]:
    c0 = _ctx(start)
    card, player = int(c0.pending_op_card), int(c0.decision_player)
    ends: Set[bytes] = set()

    def rec(st: "ts.GameState") -> None:
        for a in np.flatnonzero(np.asarray(ts.Engine.get_flat_action_mask(st, False))):
            a = int(a)
            if not (NODE_OFFSET <= a < NODE_OFFSET + N_COUNTRIES or a == CONFIRM_DONE):
                continue
            nxt = st.clone()
            ts.Engine.step_flat(nxt, a, True, False)
            if _in_block(nxt, card, player):
                rec(nxt)
            else:
                ends.add(influence(nxt).tobytes())
    rec(start.clone())
    return ends


def test_canonical_enumeration_finds_every_end_board_of_small_plays() -> None:
    for st in _starts(4, max_ops=2):
        got = {a.inf_end.tobytes() for a in enumerate_block(st)}
        assert got == _all_orders(st)


def test_the_report_counts_what_the_boards_say() -> None:
    stab, bg, _ = country_table()
    for st in _starts(3, max_ops=3, seed=5):
        allocs = enumerate_block(st)
        # the "policy" places every point where the first legal node is, as far as it can
        pol = play_block(st, lambda s: int(np.flatnonzero(
            np.asarray(ts.Engine.get_flat_action_mask(s, False)))[0]))
        rep = analyse(st, allocs, pol)
        side = rep.side
        inf0 = influence(st)
        assert rep.policy_points == int(np.clip(pol.inf_end[side] - inf0[side], 0, None).sum())
        assert rep.max_bg_gain >= rep.policy_bg_gain >= 0
        assert rep.n_allocations == len(allocs) > 0
        vals = np.arange(len(allocs), dtype=np.float64)
        rep2 = analyse(st, allocs, pol, values=vals, policy_value=-1.0)
        assert rep2.critic_rank_of_policy == len(allocs) and rep2.critic_best == len(allocs) - 1
