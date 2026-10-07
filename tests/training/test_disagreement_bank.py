"""The human-marked disagreement bank (`ai/eval/banks/disagreement_verdicts.jsonl`) holds on the
current engine: every position loads, asks the decision it was reviewed as, and every mark names a
move that is legal there -- so `bank_verdicts.py score` measures a model against what the reviewer
said, not against moves an engine or converter change has made unreachable.
"""

from __future__ import annotations

from tools.scripts.bank_verdicts import load, legal_names
from tools.scripts.disagreement_bank import KINDS, PATTERNS, kind_of, row_id
from tools.scripts.event_play_census import state_from_token


def test_the_bank_holds_one_row_per_position_under_its_own_id() -> None:
    rows = load()
    assert len(rows) >= 100
    assert len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert row_id(r) == r["id"]
        assert r["kind"] in KINDS and r["pattern"] in PATTERNS


def test_every_mark_is_a_legal_move_of_the_decision_reviewed() -> None:
    marked = 0
    for r in load():
        st = state_from_token(r["pos"])
        assert kind_of(st) == r["kind"], r["id"]
        legal = legal_names(st)
        for move in (r["human"], r["network"], r["search"], *r["marks"]):
            assert move in legal, (r["id"], move)
        assert set(r["marks"].values()) <= {"good", "bad"}
        assert not set(r["marks"]) & set(r["stale_marks"])
        marked += bool(r["marks"])
    assert marked >= 70
