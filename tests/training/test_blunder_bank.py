"""The blunder bank's selection and confirmation rules (tools/scripts/blunder_bank.py), on hand-built
rows: the leading alternative is fixed before the confirmation, and a row enters the bank only if
every judge confirms it."""

from __future__ import annotations

from typing import Any, Dict, List

from tools.scripts.blunder_bank import confirmed, leading_alternative, shortlisted


def _census(diffs: Dict[str, List[float]]) -> Dict[str, Any]:
    return {"id": "00", "moves": {"greedy": 5}, "diff_vs_greedy": diffs}


def test_the_leading_alternative_is_the_largest_lead() -> None:
    row = _census({"7": [0.12, 0.05], "9": [0.30, 0.20], "11": [-0.4, 0.1]})
    assert leading_alternative(row) == (9, 0.30, 0.20)
    assert leading_alternative(_census({})) is None


def test_a_shortlist_needs_both_the_gap_and_the_z() -> None:
    assert shortlisted(_census({"7": [0.25, 0.10]}), 0.1, 2.0)
    assert not shortlisted(_census({"7": [0.25, 0.15]}), 0.1, 2.0)    # z 1.7
    assert not shortlisted(_census({"7": [0.08, 0.01]}), 0.1, 2.0)    # sure, but small
    assert not shortlisted(_census({"7": [0.5, 0.0]}), 0.1, 2.0)      # no spread: nothing measured


def _judged(costs: Dict[str, List[float]]) -> Dict[str, Dict[str, Any]]:
    return {rid: {"cost": c} for rid, c in costs.items()}


def test_every_judge_must_confirm_and_the_bank_is_ordered_by_the_smaller_cost() -> None:
    rows = {
        "soup": _judged({"a": [0.40, 0.05], "b": [0.20, 0.03], "c": [0.30, 0.04], "d": [0.5, 0.05]}),
        "r32": _judged({"a": [0.15, 0.04], "b": [0.25, 0.03], "c": [0.05, 0.02]}),
    }
    # c: the second judge sees 5 points; d: the second judge never measured it
    assert confirmed(rows, 0.1, 3.0) == ["b", "a"]
    assert confirmed(rows, 0.1, 4.0) == ["b"]                          # a: 0.15 / 0.04 is z 3.75
