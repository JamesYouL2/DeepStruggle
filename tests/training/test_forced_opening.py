"""--forced-opening (training) and forced openings in tournaments: games start from a scripted setup."""

from __future__ import annotations

import numpy as np
import pytest

import ts_engine as ts
from ai.training.rollout_buffer import setup_phase_slot
from tools.lib.openings import (EAST_GERMANY, IRAN, ITALY, OPENINGS, POLAND, WEST_GERMANY, YUGOSLAVIA,
                                ScriptedSetupOverride, play_scripted_setup)


def _fresh(seed: int = 7) -> "ts.GameState":
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    return st


def _inf(st: "ts.GameState", cid: int) -> tuple:
    c = st.get_country(cid)
    return int(c.us_influence), int(c.ussr_influence)


def _is_human_opening(st: "ts.GameState") -> bool:
    return (_inf(st, POLAND)[1] == 4 and _inf(st, EAST_GERMANY)[1] == 4 and _inf(st, YUGOSLAVIA)[1] == 1
            and _inf(st, WEST_GERMANY)[0] == 4 and _inf(st, ITALY)[0] == 3 and _inf(st, IRAN)[0] == 3)


def test_the_scripted_setup_is_the_human_opening_and_leaves_setup() -> None:
    # East Germany starts with 3 USSR and Iran with 1 US, so the opening adds up to 4 and 3.
    st = play_scripted_setup(_fresh(), "human")
    assert st.current_phase != ts.Phase.SETUP and int(st.turn) == 1
    assert _is_human_opening(st)


def test_the_training_env_never_shows_a_setup_decision_even_after_auto_resets() -> None:
    from bindings.ts_env import TsVectorizedEnv
    env_ref = []
    injected = []

    def provider(i: int) -> "ts.GameState":
        st = play_scripted_setup(env_ref[0].runner.get_state(i), "human")
        injected.append(st)
        return st

    env = TsVectorizedEnv(num_envs=8, base_seed=3, start_provider=provider)
    env_ref.append(env)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)
    finished = 0
    for _ in range(3000):
        assert (np.asarray(obs)[:, setup_phase_slot()] >= 0.5 / 6.0).all(), "a setup decision reached the policy"
        m = np.asarray(masks)
        acts = np.array([rng.choice(np.flatnonzero(row)) for row in m])
        obs, masks, _r, d, info = env.step(acts)
        for ep in info["completed_episodes"]:
            assert ep["start_turn"] == 1
            finished += 1
        if finished >= 8:
            break
    assert finished >= 8, "no episode ended, so auto-resets were not exercised"
    assert len(injected) == 8 + finished, "every reset, initial and automatic, is scripted"
    assert all(_is_human_opening(st) for st in injected)


@pytest.mark.parametrize("scripted", [("US", "USSR"), ("US",), ("USSR",)])
def test_the_tournament_override_scripts_exactly_the_named_sides(scripted: tuple) -> None:
    n = 6
    runner = ts.VectorizedBatchRunner(n, 11)
    for i in range(n):
        runner.reset_game(i, 100 + i)
    runner.refresh_all()
    override = ScriptedSetupOverride(n)
    rng = np.random.default_rng(1)
    for _ in range(40):
        obs = np.asarray(runner.get_observations())
        if (obs[:, setup_phase_slot()] >= 0.5 / 6.0).all():
            break
        masks = np.asarray(runner.get_action_masks())
        dp = np.array(runner.get_decision_players())
        acts = np.array([rng.choice(np.flatnonzero(row)) for row in masks], dtype=np.int32)
        for side in scripted:
            rows = np.flatnonzero(dp == (1 if side == "US" else -1))
            override.apply(acts, obs, masks, dp, rows, "human")
        runner.step_flat_all(acts.tolist(), auto_advance=True)
    for i in range(n):
        st = runner.get_state(i)
        us_ok = _inf(st, WEST_GERMANY)[0] == 4 and _inf(st, ITALY)[0] == 3 and _inf(st, IRAN)[0] == 3
        ussr_ok = _inf(st, POLAND)[1] == 4 and _inf(st, YUGOSLAVIA)[1] == 1
        # a random side essentially never lands exactly on the human opening
        assert us_ok == ("US" in scripted) and ussr_ok == ("USSR" in scripted)
        if "US" in scripted:
            assert us_ok
        if "USSR" in scripted:
            assert ussr_ok


def test_an_opening_prefix_marks_the_agent_and_its_name() -> None:
    from tools.lib.player_agent import load_agent
    a = load_agent("opening:human:random", device="cpu")
    assert getattr(a, "forced_opening") == "human" and a.name.endswith("+human")
    with pytest.raises(ValueError):
        load_agent("opening:nope:random", device="cpu")


def test_a_packed_tournament_with_a_scripted_side_completes() -> None:
    from tools.lib.batch_tournament import BatchMatchRunner
    from tools.lib.player_agent import load_agent
    a = load_agent("opening:human:random", device="cpu")
    b = load_agent("random", device="cpu")
    res = BatchMatchRunner.play_packed_matchups([(a, b)], games_per_side=4, batch_chunk_size=8, device="cpu")
    assert len(res) == 1


def test_the_training_cli_has_no_forced_opening_by_default() -> None:
    from ai.training.train import build_parser
    assert build_parser().parse_args([]).forced_opening is None
    assert "human" in OPENINGS
