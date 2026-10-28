"""The batched evaluation path must agree with the sequential one, and only be faster.

The trainer used to evaluate snapshots with TournamentEvaluator.play_matchup, one game and
one decision at a time (deleted since; `_play_sequentially` below is that loop, kept as the
reference the batched runner is held to). That was 30x slower than the vectorized path the tournament CLI
uses, and evaluation cost grows with the opponent list, so a 3-hour run spent 37-61% of its
wall clock evaluating and one A/B arm was starved down to 473 iterations against 1024.

It was also wrong. At a ROLL_DIE node ctx().decision_player is NONE, and play_matchup fell
back to phasing_player -- handing the die roll to a policy network, 134 times per 20 games.
The batched runner resolves chance nodes inside the engine. Before the fix the two paths
disagreed by more than 25 points on the same deterministic matchup.
"""

from typing import Dict, List, Mapping

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from bindings.settle import SettleMode, settle
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.player_agent import HeuristicAgent, NeuralAgent, PlayerAgent


def _drain_chance_nodes(state: ts.GameState) -> None:
    """Settle what nobody decides -- chance nodes and forced moves -- as the batch runner does."""
    settle(state, SettleMode.FORCED)


def _agent(seed: int) -> NeuralAgent:
    torch.manual_seed(seed)
    return NeuralAgent(model=create_coldwar_net_v2("cpu"), name=f"n{seed}", device="cpu")


def _play_sequentially(agent_a: PlayerAgent, agent_b: PlayerAgent, games_per_side: int,
                       base_seed: int = 10000, max_steps: int = 2000) -> Dict[str, int]:
    """One game and one decision at a time, greedy: the deleted TournamentEvaluator's loop.

    Games k and k + games_per_side share a deal with the sides swapped, A as the USSR in the
    first half -- the pairing BatchMatchRunner uses (one chunk: seed = base_seed + k).
    """
    out = {"a_wins": 0, "b_wins": 0, "draws": 0, "a_wins_as_us": 0, "a_losses_as_us": 0,
           "a_wins_as_ussr": 0, "a_losses_as_ussr": 0}
    for g in range(2 * games_per_side):
        a_is_ussr = g < games_per_side
        st = ts.GameState()
        ts.Engine.init_game(st, base_seed + g % games_per_side)
        _drain_chance_nodes(st)
        for _ in range(max_steps):
            if ts.Engine.is_terminal(st):
                break
            p = st.ctx().decision_player
            assert p != ts.Player.NONE, "a chance node reached an agent"
            agent = agent_a if (p == ts.Player.USSR) == a_is_ussr else agent_b
            assert ts.Engine.try_step_flat(st, agent.select_action(st, p, temperature=0.0))
            _drain_chance_nodes(st)
        assert ts.Engine.is_terminal(st), f"game {g} did not finish in {max_steps} steps"
        u = float(ts.Engine.get_terminal_utility(st))
        side = "ussr" if a_is_ussr else "us"
        a_won = (u > 0) != a_is_ussr and u != 0
        if u == 0:
            out["draws"] += 1
        elif a_won:
            out["a_wins"] += 1
            out[f"a_wins_as_{side}"] += 1
        else:
            out["b_wins"] += 1
            out[f"a_losses_as_{side}"] += 1
    return out


def test_sequential_path_never_asks_a_policy_to_roll_dice() -> None:
    """A chance node must be resolved by the engine, not chosen by an agent."""
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    _drain_chance_nodes(st)

    seen_none = 0
    for _ in range(1500):
        if ts.Engine.is_terminal(st):
            break
        if st.ctx().decision_player == ts.Player.NONE:
            seen_none += 1
        legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st)))
        ts.Engine.step_flat(st, int(legal[0]) if len(legal) else 0)
        _drain_chance_nodes(st)

    assert seen_none == 0, (
        f"{seen_none} decisions were offered to an agent while decision_player was NONE; "
        f"those are chance nodes and the engine must resolve them"
    )


@pytest.mark.parametrize("opponent_is_neural", [True, False])
def test_batched_and_sequential_agree_exactly_when_deterministic(opponent_is_neural: bool) -> None:
    """Same seeds, same greedy policy: the two paths must produce identical results."""
    a = _agent(1)
    b = _agent(2) if opponent_is_neural else HeuristicAgent()

    seq = _play_sequentially(a, b, games_per_side=6)
    bat: Mapping[str, object] = BatchMatchRunner.play_parallel_matchup(
        a, b, games_per_side=6, device="cpu", temperature=0.0)

    for field in ["a_wins", "b_wins", "draws",
                  "a_wins_as_us", "a_losses_as_us",
                  "a_wins_as_ussr", "a_losses_as_ussr"]:
        assert seq[field] == pytest.approx(bat[field]), (
            f"{field}: sequential={seq[field]} batched={bat[field]}; the two evaluation "
            f"paths must be interchangeable"
        )


def test_batched_path_honours_temperature() -> None:
    """deterministic used to be hard-coded True here, which is why the paths diverged."""
    a, b = _agent(3), _agent(4)
    greedy_1 = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, device="cpu",
                                                      temperature=0.0)
    greedy_2 = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, device="cpu",
                                                      temperature=0.0)
    assert greedy_1["a_wins"] == greedy_2["a_wins"], "greedy play must be reproducible"

    # A high temperature must actually change play, or the parameter is being ignored.
    hot = [BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, device="cpu",
                                                  temperature=5.0)["a_wins"]
           for _ in range(6)]
    assert len(set(hot)) > 1 or hot[0] != greedy_1["a_wins"], (
        "temperature had no effect on the batched path"
    )


def test_matchup_can_resume_from_supplied_positions() -> None:
    """Resuming from saved positions is how the start-pool regime gets measured directly.

    Each supplied position is played twice with the sides swapped, from one pre-deal state,
    so both copies draw the same cards and deal luck cancels between the halves.
    """
    a, b = _agent(11), _agent(12)

    positions: List[ts.GameState] = []
    for seed in range(5):
        st = ts.GameState()
        ts.Engine.init_game(st, 7000 + seed)
        _drain_chance_nodes(st)
        positions.append(st.clone())

    res = BatchMatchRunner.play_parallel_matchup(
        a, b, start_states=positions, device="cpu", temperature=0.0)

    assert res["total_games"] == 2 * len(positions), (
        f"each position must be played from both sides: got {res['total_games']} games "
        f"from {len(positions)} positions"
    )
    assert res["a_wins"] + res["b_wins"] + res["draws"] == res["total_games"]
    # games_per_side is implied by the number of positions, not the default argument.
    assert res["games_per_side"] == len(positions)


def test_supplied_positions_actually_change_the_games() -> None:
    """A mid-game start must not silently fall back to dealing fresh openings."""
    a, b = _agent(13), _agent(14)

    # Advance a few games well past the opening, then resume from the start of the last turn each
    # reached. Not from wherever a fixed step count lands: a walk of first-legal moves ends its
    # game within a few turns, and where it ends moves with every change to the deal or the rules.
    advanced = []
    for seed in range(4):
        st = ts.GameState()
        ts.Engine.init_game(st, 8100 + seed)
        _drain_chance_nodes(st)
        turn_start = None
        for _ in range(220):
            if ts.Engine.is_terminal(st):
                break
            legal = np.flatnonzero(np.asarray(ts.ActionMask.generate_flat_mask(st)))
            ts.Engine.step_flat(st, int(legal[0]) if len(legal) else 0)
            _drain_chance_nodes(st)
            if not ts.Engine.is_terminal(st) and int(st.turn) > (int(turn_start.turn) if turn_start else 1):
                turn_start = st.clone()
        if turn_start is not None:
            advanced.append(turn_start)

    assert advanced, "fixture produced no non-terminal mid-game positions"
    start_turn = min(int(s.turn) for s in advanced)
    assert start_turn > 1, "fixture did not actually advance past turn 1"

    res = BatchMatchRunner.play_parallel_matchup(
        a, b, start_states=advanced, device="cpu", temperature=0.0)
    assert res["avg_turn"] >= start_turn, (
        f"games ended at mean turn {res['avg_turn']:.1f} but were resumed from turn "
        f"{start_turn}; the supplied positions were ignored"
    )
