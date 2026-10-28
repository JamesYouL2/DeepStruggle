"""A card handed over by another one can be played for its own Event.

Grain Sales To Soviets hands the US a card out of the USSR hand; if it is a US or neutral card
the US may play it as its Event. The entry's mode is the header under that Event, not the play
mode -- and on a headline entry there is no card of the entry's own to judge by.
"""
from __future__ import annotations

import pytest

from tests.replay_corpus import ReplayCorpus

_BRAZIL = 79


def test_the_entry_names_the_event_of_the_card_it_was_handed(corpus: ReplayCorpus) -> None:
    """Replay 73 turn 5: Grain Sales hands over Junta, a neutral card, and the log prints
    "Event: Junta" with its 2 Influence into Brazil before the free coup the event grants."""
    e = corpus.entry(73, 5, "Headline")
    assert e.played_card == "Junta"
    assert "Junta" in e.events
    assert e.mode == "coup", "the entry's own mode is the coup the event grants"


def test_replay_73_places_the_events_influence(corpus: ReplayCorpus) -> None:
    """Read as Operations, the event never fired and Brazil stayed empty."""
    conv = corpus.converted(73)
    assert conv.failure is None, f"replay 73 stopped at {conv.failure}"
    assert conv.entries_converted == 74


def test_a_handed_card_spent_on_operations_is_still_operations(corpus: ReplayCorpus) -> None:
    """Replay 304 turn 5: Grain Sales hands over Summit and the US coups Angola with it. The
    log prints no Event line for Summit, so there is no event to play it for."""
    e = corpus.entry(304, 5, "Headline")
    assert e.played_card == "Summit"
    assert "Summit" not in e.events
    conv = corpus.converted(304)
    assert conv.failure is None, f"replay 304 stopped at {conv.failure}"


@pytest.mark.parametrize("replay_id", [96, 270, 119, 14])
def test_the_other_grain_sales_games_are_unaffected(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
