"""tools/search_transfer.py's classification on hand-built rows: where the teacher's search departs
the student adopts, rejects or does something else; where it agrees the student may change the move;
rates are weighted by the bank's inverse inclusion probabilities."""
from __future__ import annotations

import pytest

from tools.search_transfer import analyse

ROLES = {"teacher_search": "tg", "student_raw": "sr", "student_search": "sg",
         "teacher_rollout": "trr", "student_rollout": "srr"}


def _bank(i: str, raw: int, weight: float, kind: str = "SELECT_CARD") -> dict:
    return {"id": i, "raw": raw, "weight": weight, "decision_type": kind}


def test_adoption_rejection_and_changes_by_hand() -> None:
    bank = {"a": _bank("a", 1, 1.0), "b": _bank("b", 1, 3.0), "c": _bank("c", 1, 2.0), "d": _bank("d", 1, 2.0)}
    picks = {
        "a": {"tg": 2, "sr": 2, "sg": 2, "trr": 2, "srr": 2},    # search departs; student adopts
        "b": {"tg": 2, "sr": 1, "sg": 2, "trr": 3, "srr": 1},    # departs; student rejects; rollout differs
        "c": {"tg": 1, "sr": 5, "sg": 5, "trr": 1, "srr": 1},    # search agrees; student changes, much worse
        "d": {"tg": 1, "sr": 1, "sg": 1, "trr": 1, "srr": 1},    # all agree
    }
    diffs = {
        "a": {2: (0.10, 0.01)},
        "b": {2: (0.02, 0.01), 3: (0.05, 0.01)},
        "c": {5: (-0.20, 0.02)},
        "d": {},
    }
    res = analyse(bank, picks, diffs, ROLES)
    assert res["teacher_search_departs"] == pytest.approx(4.0 / 8.0)
    dep = res["departures"]
    assert dep["adopt"]["share"] == pytest.approx(0.25) and dep["reject"]["share"] == pytest.approx(0.75)
    assert dep["adopt"]["correction_gain"][0] == pytest.approx(0.10)
    assert dep["reject"]["correction_gain"][0] == pytest.approx(0.02)
    assert dep["reject"]["student_gain"][0] == pytest.approx(0.0)
    agr = res["agreements"]
    assert agr["changed_share"] == pytest.approx(0.5)
    assert agr["worse_by_10pts_share"] == pytest.approx(0.5) and agr["n_worse_10"] == 1
    rv = res["rollout_vs_gumbel"]
    assert rv["n"] == 1 and rv["student_follows"]["neither"] == pytest.approx(1.0)
    assert rv["rollout_minus_gumbel"][0] == pytest.approx(0.03)
    # the student's own Gumbel gap: student search minus student raw, weighted
    g = res["gain_per_decision"]["student_search over student_raw"][0]
    assert g == pytest.approx((1.0 * 0.0 + 3.0 * 0.02 + 2.0 * 0.0 + 2.0 * 0.0) / 8.0)


def test_action_zero_is_a_move_like_any_other() -> None:
    bank = {"a": _bank("a", 3, 1.0)}
    picks = {"a": {"tg": 3, "sr": 0, "sg": 0}}
    diffs = {"a": {0: (-0.15, 0.01)}}
    res = analyse(bank, picks, diffs, {"teacher_search": "tg", "student_raw": "sr", "student_search": "sg"})
    assert res["agreements"]["n_confirmed_worse"] == 1 and res["agreements"]["n_worse_10"] == 1
