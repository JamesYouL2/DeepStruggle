"""The blunder bank's selection and confirmation rules (tools/scripts/blunder_bank.py), on hand-built
rows: the leading alternative is fixed before the confirmation, and a row enters the bank only if
every judge confirms it; a row decided whatever is played (`prove`) leaves it."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest

from tools.lib.jsonl import read_jsonl, write_jsonl
from tools.scripts.blunder_bank import (_move, confirmed, decided, leading_alternative, search_sims,
                                        settle, settled, shortlisted, strongest)


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


def test_a_row_is_judged_on_its_strongest_measurement() -> None:
    """The census's validation (256 simulations) and its escalation (1,024) carry the same ids;
    given both, in either order, the shortlist reads the escalated row."""
    weak = dict(_census({"7": [0.25, 0.05]}), search={"spec": "gumbel:256:8"})
    strong = dict(_census({"7": [0.01, 0.05]}), search={"spec": "gumbel:1024:8"})
    assert search_sims(weak) == 256 and search_sims(strong) == 1024 and search_sims(_census({})) == 0
    assert strongest([weak, strong])["00"] is strong
    assert strongest([strong, weak])["00"] is strong
    assert not shortlisted(strongest([weak, strong])["00"], 0.1, 2.0)
    first = dict(weak, tag=1)
    assert strongest([first, dict(weak, tag=2)])["00"] is first     # equals: the first given


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


def test_a_bank_row_and_a_shortlist_row_name_their_moves_alike() -> None:
    assert _move(142) == 142 and _move({"action": 142, "name": "x"}) == 142



def test_a_move_matters_unless_it_wins_outright_or_the_position_is_already_lost() -> None:
    assert settled("win", None) == "win"            # a sure win: whatever the playouts charge is later
    assert settled("loss", "loss") == "loss"        # lost whatever the mover plays
    assert settled("loss", "open") is None          # another move keeps a chance: a real blunder
    assert settled("open", None) is None
    assert settled(None, None) is None               # not enumerated


def test_a_row_is_decided_only_where_every_world_is_settled() -> None:
    assert decided(["loss", "loss"]) == "loss"
    assert decided(["win", "win"]) == "win"
    assert decided(["win", "loss"]) == "either"
    assert decided(["loss", None]) == ""             # the move matters in a world, or it was not enumerated
    assert decided([]) == ""


def test_settle_drops_the_decided_rows_and_refuses_a_missing_part(tmp_path: Path) -> None:

    def row(rid: str) -> Dict[str, Any]:
        return {"id": rid, "side": "US", "turn": 10, "ar": 7, "network": {"action": 1, "name": "x"},
                "exact": {"proved": False}}
    bank = tmp_path / "bank.jsonl"
    write_jsonl(str(bank), [row("a"), row("b"), row("c")])
    write_jsonl(str(tmp_path / "prove-1.jsonl.gz"), [{"id": "a", "decided": "win"}, {"id": "b", "decided": ""}])
    write_jsonl(str(tmp_path / "prove-2.jsonl.gz"), [{"id": "c", "decided": "loss"}])
    out = tmp_path / "out.jsonl"
    assert settle(str(bank), [str(tmp_path / "prove-*.jsonl.gz")], str(out)) == 0
    kept = list(read_jsonl([str(out)]))
    assert [r["id"] for r in kept] == ["b"] and kept[0]["exact"] == {"proved": False, "decided": ""}
    with pytest.raises(SystemExit):
        settle(str(bank), [str(tmp_path / "prove-1.jsonl.gz")], str(out))
