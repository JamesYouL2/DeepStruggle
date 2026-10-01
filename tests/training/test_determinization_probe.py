"""The determinization probe's arithmetic, on hand-built rows (no model, no search)."""
from typing import Any, Dict, List

import numpy as np

from ai.eval.determinization_targets import _on_legal, report


def _row(worlds: List[List[float]], privileged: List[float], prior: List[float]) -> Dict[str, Any]:
    return {"decision_type": "SELECT_CARD", "side": "US", "turn": 5, "legal": [3, 7, 9],
            "prior": prior, "worlds": worlds, "worlds_small": worlds, "privileged": privileged}


def test_identical_worlds_agree_everywhere() -> None:
    w = [[60.0, 3.0, 1.0]] * 8
    _, js = report([_row(w, [64.0, 0.0, 0.0], [0.8, 0.1, 0.1])])
    g = js["groups"]["all"]
    assert g["one_world_pair_agree"] == 1.0
    assert g["world_sensitive"] == 0.0
    assert g["one_world_vs_rest_tv"] == 0.0
    assert g["privileged_vs_avg_agree"] == 1.0


def test_a_world_dependent_best_move_is_flagged() -> None:
    # Half the worlds prefer action 0, half action 1: every one-world target disagrees with
    # half of the others, and the best move varies by world.
    w = [[50.0, 14.0, 0.0]] * 4 + [[14.0, 50.0, 0.0]] * 4
    _, js = report([_row(w, [0.0, 64.0, 0.0], [0.4, 0.4, 0.2])])
    g = js["groups"]["all"]
    assert g["world_sensitive"] == 1.0
    assert g["modal_world_share"] == 0.5
    assert np.isclose(g["one_world_pair_agree"], 12 / 28)  # C(4,2) * 2 agreeing pairs of C(8,2)


def test_visits_on_actions_illegal_in_the_real_state_are_dropped() -> None:
    assert _on_legal([3, 5, 9], np.array([10.0, 40.0, 14.0]), [3, 7, 9]) == [10.0, 0.0, 14.0]
