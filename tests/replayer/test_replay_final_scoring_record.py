"""The log writes final scoring on a record of its own, uncapped.

ts-replayer hangs it on a header with no card and no play -- "Turn 10, US AR8: :" -- and states
the whole swing in one line. The engine has already played it out and stopped at the cap, so
the record is something to recognise rather than to drive.
"""
from __future__ import annotations

from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_parse import VpGain, parse_entry
from tests.replay_corpus import ReplayCorpus


def test_the_record_states_a_score_past_the_cap(corpus: ReplayCorpus) -> None:
    """Replay 69: "US gains 28 VP. Score is US 34.", where the game is won at 20."""
    raws = cast(List[Dict[str, object]], corpus.game(69)["all_turns"])
    last = parse_entry(raws[-1])
    assert (last.turn, last.phase) == (10, "AR8")
    assert not last.card
    assert last.vp_gains == [VpGain("US", 28, "US", 34)]


def test_replay_69_converts_in_full_and_ends_where_the_log_ends(corpus: ReplayCorpus) -> None:
    """Clamped, the log's 34 and the engine's 20 are the same ending -- so an entry the engine
    has already played out is not one it ended early."""
    conv = corpus.converted(69)
    assert conv.failure is None, f"replay 69 stopped at {conv.failure}"
    assert conv.game_ended is True
    assert conv.entries_converted == conv.entries_total == 154


@pytest.mark.parametrize("replay_id", [154, 105, 109])
def test_the_other_games_that_end_at_the_cap_are_unaffected(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.game_ended is True
