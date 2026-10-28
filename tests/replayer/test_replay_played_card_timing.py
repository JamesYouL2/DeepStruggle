"""Whose timing the log is describing when one card hands over another.

Playing an opponent's card for Operations asks which of the Event and the Operations resolves
first, and the log answers by line order. When the card was handed over by another one -- Grain
Sales To Soviets, Missile Envy -- the "Event:" header above that exchange belongs to the card
that did the handing, so the ordering has to be read from the lines after the handover.
"""
from __future__ import annotations

import pytest

from tests.replay_corpus import ReplayCorpus


def test_the_handing_cards_own_header_is_not_the_answer(corpus: ReplayCorpus) -> None:
    """Replay 270 turn 5 AR1: Grain Sales' "Event:" opens the entry, and OPEC's comes last."""
    e = corpus.entry(270, 5, "AR1", "US")
    assert e.played_card == "OPEC"
    assert e.event_first is True, "read from the top, the entry looks event-first"
    assert e.played_event_first is False, "OPEC's own Event is printed after the coup"


def test_an_entry_with_no_handover_says_nothing_about_a_played_card(corpus: ReplayCorpus) -> None:
    e = corpus.entry(270, 5, "AR6", "USSR")
    assert e.played_card is None
    assert e.played_event_first is None


def test_replay_270_scores_opec_after_the_coup(corpus: ReplayCorpus) -> None:
    """OPEC pays the USSR 1 VP for each oil country they control, and the coup on Venezuela
    takes it out of USSR hands before OPEC counts: 2 VP, not 3."""
    conv = corpus.converted(270)
    assert conv.failure is None, f"replay 270 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id,entries", [(137, 54), (123, 144), (124, 144)])
def test_the_headline_games_this_needed_the_engine_for(replay_id: int, entries: int, corpus: ReplayCorpus) -> None:
    """Ops-first in a headline used to lose the Event outright -- see advance_after_ops.

    Replay 137's count stops short of its total: the recording stops inside turn 5, and that
    turn is a fragment -- see tests/replayer/test_replay_unfinished_final_turn.py.
    """
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert conv.entries_converted == entries
