"""Judging a branch by where it can still go, not by how far it has already run.

A two-sided event is chosen by trying each branch on a clone and seeing which one leads where
the log went. Two things flattered the branch that finishes the event outright.

`primary_id` means whatever the decision it belongs to means, and a card selection's is a card
id -- which shares its numeric range with the country ids the log names as targets. A branch
that finished and left the engine asking for the next card scored for every country id that
happened to match a card in hand.

DEFCON improves at a turn end, so a branch that finishes sails on to it and matches the DEFCON
the entry records, while the branch that opens further decisions stops short of it. That is the
same trap the score comparison already guards against by asking only where the log names no
targets, and the DEFCON comparison now does the same.

At turn 8 AR7 of replay 234 -- the last action round of the turn -- the US plays South African
Unrest for Operations and the USSR's event places 1 Influence in South Africa and 2 in Angola.
The other branch, 2 Influence in South Africa, collected both bonuses: 1582 against 1083.
"""
import pytest

from tools.lib.ts_replayer_convert import event_queue
from tools.lib.ts_replayer_parse import InfluenceLine
from tests.replay_corpus import ReplayCorpus

SOUTH_AFRICA, ANGOLA = 63, 59


def test_the_log_takes_the_split_branch(corpus: ReplayCorpus) -> None:
    """1 Influence in South Africa and 2 in Angola, not 2 in South Africa."""
    e = corpus.entry(234, 8, "AR7", "US")
    assert e.card == "South African Unrest"
    assert InfluenceLine("USSR", 1, SOUTH_AFRICA, 3, 1) in e.influence
    assert InfluenceLine("USSR", 2, ANGOLA, 5, 4) in e.influence
    assert event_queue(e) == [SOUTH_AFRICA, ANGOLA, ANGOLA]


def test_replay_234_converts_end_to_end(corpus: ReplayCorpus) -> None:
    conv = corpus.converted(234)
    assert conv.failure is None, f"replay 234 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 143


@pytest.mark.parametrize("replay_id", [302, 113, 119, 101, 104, 164])
def test_the_branch_choices_these_games_turn_on_are_unchanged(replay_id: int, corpus: ReplayCorpus) -> None:
    """Warsaw Pact's two branches, Wargames' ending, and How I Learned's DEFCON.

    Each of these is a branch the search gets right for a different reason -- named targets,
    the score the entry ends on, and a numeric setting with no targets at all -- so between
    them they cover what the two bonuses were there for.
    """
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
