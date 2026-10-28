"""An event that may stop asks only for what its own lines record.

Left to fall through to the Ops queue, an event with countries still queued under its header
takes one the entry's coup or realignment was going to use, and the count it owes goes with it.
Only where the event's own queue is empty does the Ops queue answer -- which is how Che's free
coups, recorded under Che's header but spent as Operations, still find their targets.
"""
from __future__ import annotations

import pytest

from tools.lib.ts_replayer_convert import event_point_queues
from tests.replay_corpus import ReplayCorpus

_DE_STALINIZATION = 33
_FINLAND = 5
_CHILE = 81


def test_the_event_queue_holds_the_relocation_and_the_coup_is_the_ops_queues(corpus: ReplayCorpus) -> None:
    """Replay 80 turn 1 AR1: the US plays De-Stalinization for Operations, its event relocates
    one USSR Influence from Finland to Chile, and the Operations coup Egypt."""
    e = corpus.entry(80, 1, "AR1", "US")
    assert event_point_queues(e)[_DE_STALINIZATION] == [_FINLAND, _CHILE]
    assert e.targets == [29], "Egypt is the coup's, not the event's"


def test_replay_80_converts_in_full(corpus: ReplayCorpus) -> None:
    """The event may relocate up to four and relocated one. Falling through to the Ops queue
    took a second out of Egypt -- the country the coup was about to hit -- and two removed
    means two to place, so the entry ended with the engine still asking."""
    conv = corpus.converted(80)
    assert conv.failure is None, f"replay 80 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 114


@pytest.mark.parametrize("replay_id", [131, 105, 150, 123])
def test_a_free_coup_recorded_under_its_event_still_finds_its_target(replay_id: int, corpus: ReplayCorpus) -> None:
    """Che's coups are printed under Che's own header and spent as Operations."""
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
