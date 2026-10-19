"""The action log as the page shows it: one entry per play (web/ui/src/game/log_view.ts).

The session logs every click, and a play is several: the card, its mode, each Influence point.
Read click by click a play came apart -- its card's discard on its last placement, a point a
line, a deal fifteen lines. The page groups the clicks into plays and tells each once, with the
board's changes merged over the play. Merging must lose nothing and invent nothing: summed over
the entries, the changes are the board's own over the game.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import pytest

from tests.web.js_runner import PUBLIC, run_ts

INFLUENCE = re.compile(r"^  • (.+) Influence: (.+)$")
SIDE = re.compile(r"(US|USSR): (\d+) -> (\d+)")


@pytest.fixture(scope="module")
def game(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    return run_ts("log_view.ts", str(tmp_path_factory.mktemp("log_view")), PUBLIC, "6")


def _lines(entries: List[Dict[str, Any]]) -> List[str]:
    return [line for e in entries for line in e["lines"]]


def test_the_game_reached_the_second_turns_end(game: Dict[str, Any]) -> None:
    assert not game["terminal"], "pick another seed: the game must run through two turns"
    assert sum(1 for e in game["entries"] if e["title"].startswith("End of turn")) == 2


def test_merged_influence_adds_up_to_the_boards_change(game: Dict[str, Any]) -> None:
    total: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for line in _lines(game["entries"]):
        m = INFLUENCE.match(line)
        if not m:
            continue
        for side, a, b in SIDE.findall(m.group(2)):
            total[m.group(1)][0 if side == "US" else 1] += int(b) - int(a)
    want: Dict[str, Tuple[int, int]] = {k: (v[0], v[1]) for k, v in game["influence"].items() if v != [0, 0]}
    assert {k: (v[0], v[1]) for k, v in total.items() if v != [0, 0]} == want


def test_the_entries_vp_adds_up_to_the_score(game: Dict[str, Any]) -> None:
    assert sum(e["vp_delta"] for e in game["entries"]) == game["vp"]


def test_each_play_is_one_entry_headed_by_its_card(game: Dict[str, Any]) -> None:
    titles = [e["title"] for e in game["entries"]]
    played = [t for t in titles if re.match(r"^(US|USSR) plays .+ \(#\d+\)", t)]
    assert len(played) == game["plays"] > 10


def test_card_moves_are_said_in_words(game: Dict[str, Any]) -> None:
    lines = _lines(game["entries"])
    assert not [line for line in lines if " moved: " in line]
    assert any(re.match(r"^  • (US|USSR) draws \d+ cards?: ", line) for line in lines), "the deal is one line"


def test_the_other_sides_deal_stays_hidden_once_grouped(game: Dict[str, Any]) -> None:
    """The USSR's view names no card dealt into the US hand unseen, on a draws line or any other."""
    unseen = set(game["unseen_us"])
    assert len(unseen) >= 5   # turn 2's deal; the opening deal precedes the log
    draws = [line for line in _lines(game["ussr_view"]) if line.startswith("  • US draws")]
    named = {name for line in draws for name in line.split(": ", 1)[1].split(", ")}
    assert not named & unseen, named & unseen
    assert any(line.startswith("  • USSR draws") for line in _lines(game["ussr_view"]))
