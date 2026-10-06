"""Macro-action search (ai/search/macro_search.py): candidates, boundaries, honesty, play.

What must hold:

* a macro is one player's contiguous decisions -- it ends where someone else decides, where the
  random stream is drawn from, or where the turn, action round or phase changes (so a headline
  is its own macro);
* the candidates are distinct results, the network's greedy line among them, ranked by policy
  probability, each replaying legally from the position;
* every world is sampled from the ROOT decider's side, so no value is read with the decider
  seeing the opponent's real hand;
* the agent's moves are legal and the chosen macro's later steps are played from its plan.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search import macro_search as M
from ai.search.batched_mcts import settle
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import load_agent


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _cfg(**kw: int) -> M.MacroSearchConfig:
    base = {"k": 6, "k_reply": 3, "depth": 1, "beam": 8, "branch": 3, "branch_first": 6, "worlds": 2}
    base.update(kw)
    return M.MacroSearchConfig(**base)  # type: ignore[arg-type]


def _first_legal(state: ts.GameState) -> int:
    return int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(state)))[0])


def _positions(n: int = 6, seed: int = 5) -> Tuple[List[ts.GameState], List[ts.GameState]]:
    """(headline choices, action-round card choices), reached by playing the first legal action
    (an untrained network's own play can end a game before its first action round)."""
    heads: List[ts.GameState] = []
    cards: List[ts.GameState] = []
    for g in range(n):
        s = ts.GameState()
        ts.Engine.init_game(s, seed + g)
        settle(s, True)
        for _ in range(400):
            if ts.Engine.is_terminal(s):
                break
            c = s.ctx()
            if c.decision_type == ts.DecisionType.SELECT_CARD and int(c.resolving_card) == 0:
                if s.current_phase == ts.Phase.HEADLINE and len(heads) < n:
                    heads.append(s.clone())
                elif s.current_phase == ts.Phase.ACTION_ROUND and len(cards) < 2 * n:
                    cards.append(s.clone())
            ts.Engine.step_flat(s, _first_legal(s))
            settle(s, True)
    return heads, cards


def test_a_headline_is_its_own_macro() -> None:
    heads, _ = _positions(3)
    gen = M.MacroGenerator(M._Net(_model(), torch.device("cpu")), _cfg())
    for ms in gen.candidates(heads, 6):
        assert ms
        assert all(len(m.actions) == 1 for m in ms)


def test_candidates_are_distinct_ranked_and_replay_to_a_boundary() -> None:
    _, cards = _positions(4)
    net = M._Net(_model(), torch.device("cpu"))
    gen = M.MacroGenerator(net, _cfg())
    for st, ms in zip(cards, gen.candidates(cards, 6)):
        assert 1 <= len(ms) <= 6
        assert sum(m.greedy for m in ms) == 1
        assert len({m.result for m in ms}) == len(ms)
        assert [m.rank for m in ms] == sorted(m.rank for m in ms)
        assert all(m.logp <= 1e-9 for m in ms)
        g_actions, g_key = M.greedy_macro(net, st)
        assert [m.actions for m in ms if m.greedy] == [g_actions]
        assert any(m.result == g_key for m in ms)
        for m in ms:
            s = st.clone()
            mover = int(acting_player(s))
            for j, a in enumerate(m.actions):
                assert np.asarray(ActionEncoder.get_legal_mask(s))[a]
                stage, rng = M._stage(s), int(s.rng_state)
                ts.Engine.step_flat(s, a)
                settle(s, True)
                ended, _ = M.macro_boundary(stage, rng, mover, s)
                # Ends exactly at its last step, never before.
                assert ended == (j == len(m.actions) - 1)


def test_diversity_reserves_slots_for_distinct_groups() -> None:
    st = ts.GameState()
    lines = [M._Line(root=0, state=st, mover=0, logp=-0.1 * j, group=(1, 2), greedy=(j == 0))
             for j in range(6)]
    lines += [M._Line(root=0, state=st, mover=0, logp=-5.0 - j, group=(3 + j, 2), greedy=False)
              for j in range(3)]
    top = M._diverse_top(lines, 4, 0.5)
    assert top[0].greedy
    assert len({ln.group for ln in top}) >= 2
    # Without the reservation the four most probable are all one group.
    assert len({ln.group for ln in M._diverse_top(lines, 4, 0.0)}) == 1


def test_worlds_are_sampled_from_the_root_deciders_side(monkeypatch: pytest.MonkeyPatch) -> None:
    _, cards = _positions(3)
    seen: List[Tuple[int, int]] = []
    real = M.determinize

    def spy(state: ts.GameState, me: ts.Player, rng: object) -> ts.GameState:
        seen.append((int(acting_player(state)), int(me)))
        return real(state, me, rng)  # type: ignore[arg-type]

    monkeypatch.setattr(M, "determinize", spy)
    ms = M.MacroSearch(_model(), torch.device("cpu"), _cfg(depth=2))
    ms.search(cards[:3])
    assert seen
    assert all(mover == me for mover, me in seen)


def test_rollout_plays_on_to_a_later_action_round() -> None:
    _, cards = _positions(2)
    ms = M.MacroSearch(_model(), torch.device("cpu"), _cfg(rollout_ars=1))
    for before, after in zip(cards[:3], ms._rollout(cards[:3])):
        assert ts.Engine.is_terminal(after) or \
            (M._round(after), int(after.phasing_player)) != (M._round(before), int(before.phasing_player))


def test_the_agent_plays_legal_moves_and_follows_its_plans() -> None:
    agent = M.MacroSearchAgent(_model(), device=torch.device("cpu"), config=_cfg())
    runner = ts.VectorizedBatchRunner(4, 11)
    for _ in range(250):
        states = [runner.get_state(i) for i in range(4)]
        live = [i for i, s in enumerate(states) if not ts.Engine.is_terminal(s)]
        if not live:
            break
        picks = agent.select_actions_batch([states[i] for i in live])
        acts = [211] * 4
        for i, a in zip(live, picks):
            assert np.asarray(ActionEncoder.get_legal_mask(states[i]))[a]
            acts[i] = int(a)
        runner.step_flat_all(acts, auto_advance=True)
    assert agent.searched_count > 0
    assert agent.planned_count > 0


def test_a_macro_spec_refuses_an_unknown_field() -> None:
    with pytest.raises(ValueError, match="not a MacroSearchConfig field"):
        load_agent("macro:nowhere.pt:depth=2:bogus=1", device="cpu")
