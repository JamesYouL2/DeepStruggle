"""Olympic Games is the opponent's choice, and the log states it in words.

Participating rolls dice for 2 VP; boycotting degrades DEFCON and hands the player the card's
Operations. Nothing else in an entry distinguishes the two before they resolve, so the branch
search -- which judges a branch by the board and score it reaches -- had nothing to go on.
"""
from __future__ import annotations

from tests.replay_corpus import ReplayCorpus

_SOUTH_KOREA = 38


def test_a_boycott_is_read_from_the_entry(corpus: ReplayCorpus) -> None:
    e = corpus.entry(28, 3, "Headline", "both")
    assert e.olympics_boycotted is True


def test_participating_is_read_the_same_way(corpus: ReplayCorpus) -> None:
    e = corpus.entry(260, 3, "Headline", "both")
    assert e.olympics_boycotted is False


def test_an_entry_without_the_card_says_nothing(corpus: ReplayCorpus) -> None:
    e = corpus.entry(28, 3, "AR1", "USSR")
    assert e.olympics_boycotted is None


def test_replay_28_spends_the_four_operations_the_boycott_grants(corpus: ReplayCorpus) -> None:
    """Turn 3's headline: 2 into South Korea, 1 into Angola, 1 into Indonesia. Taking the
    participation roll instead lost all four."""
    conv = corpus.converted(28)
    assert conv.failure is None, f"replay 28 stopped at {conv.failure}"
    assert conv.entries_converted == 122


def test_the_games_that_participate_are_unaffected(corpus: ReplayCorpus) -> None:
    for replay_id in (260, 161, 101):
        conv = corpus.converted(replay_id)
        assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
