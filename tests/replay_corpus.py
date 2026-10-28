"""The ts-replayer corpus as tests read it: one loader, failing -- never skipping -- when absent.

Tests take it through the `corpus` fixture (tests/conftest.py) and annotate the parameter with
`ReplayCorpus`. It replaces a `_game()` / `_entry()` / `_convert()` helper that ~60 test files
each carried a copy of, several of them ending in `pytest.skip("corpus not downloaded")`: a check
that passes on every machine without the data while verifying nothing.

A raw game is returned as a fresh copy on every call, since some tests edit one before
converting it. A conversion is memoized and SHARED (one costs a z3 solve for both hands plus an
entry-by-entry engine drive, and a test file typically asks for the same replay several times),
so treat a `Conversion` from `converted` as read-only. The memo keeps the most recent few only:
a conversion holds every decision's observation, ~9 MB a game, and the suite converts ~100
distinct replays. A loop over the whole corpus uses `every_game` and `convert_game` directly.
"""

from __future__ import annotations

import copy
import gzip
import json
import pathlib
from collections import OrderedDict
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pytest

from tools.lib.corpus_paths import missing_corpus_reason
from tools.lib.ts_replayer_convert import Conversion, convert_game
from tools.lib.ts_replayer_parse import Entry, parse_entry


#: How many conversions `converted` keeps (see the module docstring).
CONVERSIONS_KEPT = 12


class ReplayCorpus:
    """Raw games, parsed entries and conversions of the downloaded human corpus."""

    def __init__(self, directory: pathlib.Path) -> None:
        self.directory = directory
        self._games: Dict[int, Dict[str, Any]] = {}
        self._conversions: OrderedDict[int, Conversion] = OrderedDict()

    def path(self, replay_id: int) -> pathlib.Path:
        """The replay's file, failing the test when the corpus does not hold it."""
        path = self.directory / f"{replay_id}.json.gz"
        if not path.exists():
            pytest.fail(f"replay {replay_id} is not in the corpus at {self.directory}",
                        pytrace=False)
        return path

    def files(self) -> List[pathlib.Path]:
        """Every replay of the corpus, sorted; failing, not empty, when there are none."""
        found = sorted(self.directory.glob("*.json.gz"))
        if not found:
            pytest.fail(missing_corpus_reason() or f"no replays at {self.directory}",
                        pytrace=False)
        return found

    def every_game(self) -> Iterator[Tuple[int, Dict[str, Any]]]:
        """(replay id, JSON) of every replay, in file order, read one at a time and not kept."""
        for path in self.files():
            with gzip.open(path, "rt") as fh:
                yield int(path.name.split(".")[0]), json.load(fh)

    def game(self, replay_id: int) -> Dict[str, Any]:
        """The replay's JSON, a fresh copy the caller may edit."""
        if replay_id not in self._games:
            with gzip.open(self.path(replay_id), "rt") as fh:
                self._games[replay_id] = json.load(fh)
        return copy.deepcopy(self._games[replay_id])

    def turns(self, replay_id: int) -> List[Dict[str, Any]]:
        """The replay's log entries (its `all_turns`), unparsed."""
        return self.game(replay_id)["all_turns"]

    def entry(self, replay_id: int, turn: int, phase: str,
              player: Optional[str] = None) -> Entry:
        """The first entry at (turn, phase) -- and player, when given -- parsed."""
        for raw in self.turns(replay_id):
            e = parse_entry(raw)
            if (e.turn, e.phase) == (turn, phase) and (player is None or e.player == player):
                return e
        who = "" if player is None else f" {player}"
        raise AssertionError(f"replay {replay_id} has no{who} {phase} of turn {turn}")

    def converted(self, replay_id: int) -> Conversion:
        """`convert_game` of the replay, memoized. SHARED: read it, never mutate it."""
        if replay_id in self._conversions:
            self._conversions.move_to_end(replay_id)
        else:
            self._conversions[replay_id] = convert_game(self.game(replay_id))
            while len(self._conversions) > CONVERSIONS_KEPT:
                self._conversions.popitem(last=False)
        return self._conversions[replay_id]


def failed_at(conv: Conversion) -> Optional[str]:
    """Where a conversion stopped, as "T<turn> <phase>", or None if it finished."""
    return None if conv.failure is None else f"T{conv.failure.turn} {conv.failure.phase}"
