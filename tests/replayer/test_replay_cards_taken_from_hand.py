"""Two events reach into the opponent's hand and take a card, and it is gone from there.

Missile Envy takes the highest Ops card; Grain Sales To Soviets takes one at random. Either way
the card has left that hand for the rest of the turn, while the turn's hand list still names it
-- it was theirs when the turn began and it became visible while they held it.

Missile Envy's transfer was already tracked. Grain Sales' was not, and it shows up as the
opponent still holding cards the log says they have run out of: at turn 8 AR1 of replay 51 the
US plays Grain Sales, draws Puppet Governments out of the USSR hand and plays it, and at the
USSR's own AR7 -- where the log reads "USSR has no cards to discard" -- Five Year Plan found
Puppet Governments still sitting there and fired its event.

A card fired through another card is also a card the entry plays. At turn 8 AR3 of replay 158
the US plays Star Wars, which takes Grain Sales To Soviets out of the discard pile and fires it,
and because the entry's own card is Star Wars nothing recognised that Grain Sales was in play at
all -- so its draw was never steered to the Cuban Missile Crisis the log records. It drew
Blockade, whose event strips every US Influence from West Germany.
"""
import pytest

from tools.lib.ts_replayer_convert import _taken_from_hand, card_id
from tests.replay_corpus import ReplayCorpus


def test_grain_sales_takes_a_card_out_of_the_opponents_hand(corpus: ReplayCorpus) -> None:
    """Replay 51 turn 8 AR1: the US draws Puppet Governments from the USSR."""
    e = corpus.entry(51, 8, "AR1", "US")
    assert e.card == "Grain Sales To Soviets"
    assert _taken_from_hand(e) == [("USSR", card_id("Puppet Governments*"))]


def test_missile_envy_is_still_tracked_too(corpus: ReplayCorpus) -> None:
    """Replay 114 turn 5 AR1: the US takes Nuclear Test Ban."""
    e = corpus.entry(114, 5, "AR1", "US")
    assert _taken_from_hand(e) == [("USSR", card_id("Nuclear Test Ban"))]


def test_an_entry_that_takes_nothing_reports_nothing(corpus: ReplayCorpus) -> None:
    assert _taken_from_hand(corpus.entry(51, 8, "AR2", "USSR")) == []


def test_the_ussr_really_is_out_of_cards_when_the_log_says_so(corpus: ReplayCorpus) -> None:
    """Replay 51 turn 8 AR7: "USSR has no cards to discard"."""
    e = corpus.entry(51, 8, "AR7", "USSR")
    assert e.card == "Five Year Plan"
    assert "USSR has no cards to discard" in (e.text or "")
    conv = corpus.converted(51)
    assert conv.failure is None, f"replay 51 stopped at {conv.failure}"


def test_a_card_fired_through_another_card_counts_as_played(corpus: ReplayCorpus) -> None:
    """Replay 158 turn 8 AR3: Star Wars fires Grain Sales, whose draw needs steering."""
    e = corpus.entry(158, 8, "AR3", "US")
    assert e.card == "Star Wars*"
    assert "Grain Sales To Soviets" in (e.events or []), (
        "the entry fires it even though it never played it")
    assert _taken_from_hand(e) == [("USSR", card_id("Cuban Missile Crisis*"))]


def test_replay_158_gets_past_the_star_wars_entry(corpus: ReplayCorpus) -> None:
    conv = corpus.converted(158)
    assert conv.failure is None or (conv.failure.turn, conv.failure.phase) != (8, "AR3"), (
        f"replay 158 still stops at its Star Wars entry: {conv.failure}")
    assert conv.entries_converted > 105


@pytest.mark.parametrize("replay_id", [51, 52, 145, 300, 114, 141])
def test_the_hand_transfer_games_convert(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
