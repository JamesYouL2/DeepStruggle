"""The cards the action log names, made hoverable (web/ui/src/game/card_refs.ts).

Each mention -- `Name (#id)`, the headline reveal's `'Name' (#id, n Ops)`, a card move's
`Card #id (Name)`, a deal's bare names -- is wrapped in a `span.log-card[data-card-id]`, whose
hover shows the card's Ops, side and text. The wrapping must change nothing else: with the spans
taken out, every line reads as before, and every card the line names by number is in a span.
Checked under node on a two-turn game's grouped log and on one hand-written line per form.
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import pytest

from tests.web.js_runner import PUBLIC, run_ts

SPAN = re.compile(r'<span class="log-card" data-card-id="(\d+)">(.*?)</span>')
NUMBERED = re.compile(r"\(#(\d{1,3})[,)]|Card #(\d{1,3}) \(")

CASES: List[str] = [
    "USSR plays Card: Fidel (#8)",
    "  ★ Headlines Simultaneously Revealed: US plays 'NATO' (#21, 4 Ops) vs USSR plays 'De-Stalinization' (#33, 3 Ops)",
    "  • Card #4 (Duck and Cover) moved: HAND_US_UNKNOWN -> DISCARD_PILE",
    "  • US draws 3 cards: Duck and Cover, NATO, Not A Card",
    "  🎲 Space Race Attempt (Target Box #2): US rolls 3 -> SUCCESS (Advanced to Box #2)",
    "▸ De-Stalinization (#33) — USSR",
]


@pytest.fixture(scope="module")
def refs(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    tmp = tmp_path_factory.mktemp("card_refs")
    cases = tmp / "cases.json"
    cases.write_text(json.dumps(CASES))
    return run_ts("card_refs.ts", str(tmp), PUBLIC, "6", str(cases))


def _spans(h: str) -> List[tuple]:
    return [(int(i), inner) for i, inner in SPAN.findall(h)]


def _strip(h: str) -> str:
    return SPAN.sub(lambda m: m.group(2), h)


def test_the_spans_change_nothing_else(refs: Dict[str, Any]) -> None:
    for row in refs["game"] + refs["cases"]:
        assert _strip(row["html"]) == row["text"], row


def test_every_card_named_by_number_in_the_game_is_hoverable(refs: Dict[str, Any]) -> None:
    names: Dict[str, List[str]] = refs["names"]
    wrapped = 0
    for row in refs["game"]:
        numbered = [int(a or b) for a, b in NUMBERED.findall(row["text"]) if str(int(a or b)) in names]
        spans = _spans(row["html"])
        assert sorted(i for i, inner in spans if "#" in inner) == sorted(numbered), row
        for i, inner in spans:
            # A span is the card's name, its number, or both -- never another card's name.
            assert f"#{i}" in inner or inner in names[str(i)], (row, inner)
            if f"(#{i}" in inner and not inner.startswith(f"(#{i}"):
                assert any(inner.startswith((n, f"'{n}'")) for n in names[str(i)]), (row, inner)
        wrapped += len(spans)
    assert wrapped > 20, "two turns of play name more cards than that"
    deals = [row for row in refs["game"] if " draws " in row["text"] and " cards: " in row["text"]]
    assert deals and all(_spans(row["html"]) for row in deals), "a deal's cards are hoverable too"


def test_each_form_of_mention(refs: Dict[str, Any]) -> None:
    out = [_spans(row["html"]) for row in refs["cases"]]
    assert out[0] == [(8, "Fidel (#8)")]
    assert out[1] == [(21, "'NATO' (#21, 4 Ops)"), (33, "'De-Stalinization' (#33, 3 Ops)")]
    assert out[2] == [(4, "Card #4 (Duck and Cover)")]
    assert out[3] == [(4, "Duck and Cover"), (21, "NATO")], "a name that is no card is left alone"
    assert out[4] == [], "a Space Race box is not a card"
    assert out[5] == [(33, "De-Stalinization (#33)")]
