"""A blank line closes the Ops section above it, not just the event's placements.

A coup prints its removal and its placement directly under its own header. A placement below a
blank line is something else -- NORAD's, in every case in the corpus -- and reading it as part
of the coup relied on the victim rule, which only catches a placement by the side that just
lost Influence there.
"""
from __future__ import annotations

import pytest

from tools.lib.ts_replayer_convert import point_queue
from tests.replay_corpus import ReplayCorpus

_ARGENTINA = 82
_UNITED_KINGDOM = 1


def test_a_placement_after_the_gap_is_not_the_coups(corpus: ReplayCorpus) -> None:
    """Replay 281 turn 6 AR4: ABM Treaty's coup on Argentina has a margin of 3, which removes
    the 3 USSR Influence there and leaves nothing over to place. The "US +1 in Argentina" after
    the blank line is NORAD's, and it is the same side and the same country as a coup's own
    placement would be."""
    e = corpus.entry(281, 6, "AR4", "US")
    coup = [s for s in e.sections if s.mode == "coup"]
    assert len(coup) == 1
    assert [row.delta for row in coup[0].influence] == [-3], (
        "the removal, and nothing after the gap")
    assert point_queue(e).count(_ARGENTINA) == 2, "the coup's target, and NORAD's placement"


def test_replay_281_converts_in_full(corpus: ReplayCorpus) -> None:
    conv = corpus.converted(281)
    assert conv.failure is None, f"replay 281 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 145


def test_the_placement_is_queued_where_the_log_prints_it(corpus: ReplayCorpus) -> None:
    """Replay 260 turn 9 AR3: Junta's 2 into Argentina, a coup on Argentina, then NORAD's 1
    into the United Kingdom -- last in the log and last in the queue."""
    e = corpus.entry(260, 9, "AR3", "US")
    queue = point_queue(e)
    assert queue == [_ARGENTINA, _ARGENTINA, _ARGENTINA, _UNITED_KINGDOM]

    conv = corpus.converted(260)
    assert conv.failure is None, f"replay 260 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id", [131, 104, 150, 133, 277, 203])
def test_the_other_norad_games_are_unaffected(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
