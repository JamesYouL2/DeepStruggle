"""Guards for reading a critic off a recorded game.

The whole method rests on the reconstruction being the same game the replay recorded. A drifted
reconstruction still returns values, and they look like results, so it is verified rather than
assumed -- both here and at every call site, via `verify()`.
"""
from __future__ import annotations

import copy
import json
import os
from typing import Any, Dict

import pytest
import ts_engine as ts

from ai.eval.replay_critic import control, load_actions, replay_to, verify
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from tools.lib.self_play import generate_self_play_replay


@pytest.fixture(scope="module")
def replay_path(tmp_path_factory: pytest.TempPathFactory) -> str:
    """A real replay, generated here -- data/replays is git-ignored and may hold nothing."""
    model = create_coldwar_net_v2()
    out = str(tmp_path_factory.mktemp("replays") / "critic_probe.tslog.json")
    generate_self_play_replay(model=model, seed=4321, temperature=1.0,
                              output_path=out, device="cpu", verbose=False)
    assert os.path.exists(out)
    return out


def test_reconstruction_matches_every_recorded_snapshot(replay_path: str) -> None:
    """Replaying the logged actions must reproduce the logged states exactly."""
    seed, steps = load_actions(replay_path)
    assert steps, "replay has no steps"

    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    checked = 0
    for rec in steps:
        ts.Engine.step_flat(state, rec["flat"])
        while (not ts.Engine.is_terminal(state)
               and state.ctx().decision_player == ts.Player.NONE
               and state.ctx().decision_type == ts.DecisionType.ROLL_DIE):
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
        problem = verify(state, rec)
        assert problem is None, f"diverged at step {rec['step_index']}: {problem}"
        checked += 1
    assert checked == len(steps)


def test_replay_to_is_the_state_after_that_step(replay_path: str) -> None:
    """Step indices are 1-based and the snapshot is the state *after* the action."""
    seed, steps = load_actions(replay_path)
    assert steps[0]["step_index"] == 1, "step indices are 1-based"
    mid = steps[len(steps) // 2]
    st = replay_to(seed, steps, mid["step_index"])
    assert verify(st, mid) is None


def test_verify_catches_a_wrong_state(replay_path: str) -> None:
    """A checker that never fails is not a checker.

    Each field it compares is perturbed by one in an otherwise matching record, so every check
    is exercised whatever game the untrained generator happened to play. (It used to compare
    the first state against the last and skip when the two agreed.)
    """
    seed, steps = load_actions(replay_path)
    first = steps[0]
    state = replay_to(seed, steps, first["step_index"])
    assert verify(state, first) is None, "precondition: the matching record verifies"

    def perturbed(field: str) -> Dict[str, Any]:
        rec = copy.deepcopy(first)
        rec["snapshot"][field] = int(rec["snapshot"][field]) + 1
        return rec

    for field in ("victory_points", "turn", "defcon"):
        assert verify(state, perturbed(field)) is not None, f"a wrong {field} went unnoticed"

    rec = copy.deepcopy(first)
    name, country = next(iter(rec["snapshot"]["countries"].items()))
    country["us_influence"] = int(country.get("us_influence", 0)) + 1
    assert verify(state, rec) is not None, f"wrong influence in {name} went unnoticed"


def test_control_reads_the_stability_threshold() -> None:
    state = ts.GameState()
    ts.Engine.init_game(state, 7)
    # Italy is stability 2: a 2-point lead controls it, a 1-point lead does not.
    assert "Italy" == str(ts.MapData.get_country_info(10)["name"])
    assert int(ts.MapData.get_country_info(10)["stability"]) == 2
    text = control(state, 10)
    assert text.count("/") == 1 and text.split()[-1] in {"US", "USSR", "--"}


def test_replay_metadata_carries_the_seed(replay_path: str) -> None:
    with open(replay_path, encoding="utf-8") as f:
        d = json.load(f)
    assert int(d["metadata"]["seed"]) == 4321
