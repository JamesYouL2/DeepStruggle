"""Grain Sales' play-or-return choice is stated in the log, not inferred from the board.

The entry says "US plays Panama Canal Returned*" or "US returns Brezhnev Doctrine* to USSR".
The return was already read; the play was not, so that half of the choice was decided by how
close the resulting board came out -- which is nothing at all where the card's play barely
moves it.
"""
from __future__ import annotations

import pytest

from tests.replay_corpus import ReplayCorpus


def test_the_log_says_which_of_the_two_happened(corpus: ReplayCorpus) -> None:
    """Replay 105 turn 6 hands Brezhnev Doctrine back; replay 73 turn 5 plays Junta."""
    returned = corpus.entry(105, 6, "Headline")
    assert returned.returned_card == "Brezhnev Doctrine*"
    assert returned.played_card is None

    played = corpus.entry(73, 5, "Headline")
    assert played.played_card == "Junta"
    assert played.returned_card is None


@pytest.mark.parametrize("replay_id", [105, 73, 119, 304, 270, 14, 96])
def test_the_games_either_side_of_the_choice_convert(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"


def test_the_hand_miss_counter_counts_only_real_misses(corpus: ReplayCorpus) -> None:
    """It is diagnostic, and two things that are not misses were drowning out the ones that
    are: the China Card, which no hand list carries, and an entry's own card named again by its
    own discard line -- which is how a Quagmire or Bear Trap escape is recorded."""
    conv = corpus.converted(100)
    assert conv.entries_converted == conv.entries_total == 144
    assert conv.hand_misses == 0, "every card this game plays was in the hand we tracked"

    # Replay 105 turn 6: Grain Sales draws Brezhnev Doctrine and the US hands it straight back,
    # so the USSR still holds it -- the one miss here is that card, marked spent when taken.
    conv = corpus.converted(105)
    assert conv.hand_misses == 1
    assert conv.hands_solved, "and the hands themselves come from the constraint model"
