"""A war whose die the log states is settled by that die, not by a search for a seed.

Searching could not settle it anyway: the expectation the search matches against is the
entry's final board, and another card in the same entry can move the country afterwards.
"""
from __future__ import annotations

import pytest

from tests.replay_corpus import ReplayCorpus

_PANAMA = 70


def test_the_entry_states_the_die_and_moves_the_country_afterwards(corpus: ReplayCorpus) -> None:
    """Replay 30 turn 5: the USSR loses Brush War in Panama, which leaves it untouched, and the
    US's Grain Sales To Soviets then hands over Panama Canal Returned, whose event puts an
    Influence there."""
    e = corpus.entry(30, 5, "Headline", "both")
    assert e.war_targets == [_PANAMA]
    assert e.war_rolls == [(1, 0, False)], "DEFEAT: 1 < 3"
    assert e.influence_by_event["Panama Canal Returned*"][0].country == _PANAMA, (
        "and the Influence in Panama is that card's, printed under its own header")


def test_replay_30_converts_in_full(corpus: ReplayCorpus) -> None:
    conv = corpus.converted(30)
    assert conv.failure is None, f"replay 30 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id", [109, 101, 121, 150])
def test_the_other_war_games_are_unaffected(replay_id: int, corpus: ReplayCorpus) -> None:
    """101's Korean War, 109's, 121's war sharing an entry with a coup, and 150's."""
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
