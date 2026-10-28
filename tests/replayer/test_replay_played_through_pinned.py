"""A card played through another one was in the dealt hand just the same.

"Ask Not What Your Country Can Do For You" discards any number of cards and draws that many
replacements, so a turn's hand list can hold more cards than were dealt. Which of them are the
draws is not recorded, so everything demonstrably held from the start is pinned and the rest is
ordered best-first, with the tail taken as the draws.

What counts as demonstrably held is everything the side already spent -- and a card played
*through* another one is spent just the same. UN Intervention names a card in the player's own
hand and uses its Operations, and the log records that as "US plays Arab-Israeli War" rather
than as the entry's card, so it went unpinned.

At turn 8 AR3 of replay 262 that put Arab-Israeli War among the Ask Not draws four action
rounds later. The US then held no card of the USSR's for UN Intervention to name, so the event
found nothing, fizzled, and its 2 Ops -- one Influence into Canada and one into Greece -- were
never spent.
"""
from typing import List

import pytest

from tools.lib.ts_replayer_convert import _mid_turn_acquisitions, card_id
from tools.lib.ts_replayer_parse import InfluenceLine, parse_entry
from tests.replay_corpus import ReplayCorpus

CANADA, GREECE = 0, 11


def _held(corpus: ReplayCorpus, replay_id: int, turn: int, side: str) -> List[int]:
    hands = corpus.game(replay_id)["hands"][str(turn)]
    return [c for c in (card_id(n) for n in hands[side.lower()]) if c]


def test_the_entry_plays_a_card_through_un_intervention(corpus: ReplayCorpus) -> None:
    raws = corpus.game(262)["all_turns"]
    e = next(parse_entry(r) for r in raws
             if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
             == (8, "AR3", "US"))
    assert e.card == "UN Intervention"
    assert e.played_card == "Arab-Israeli War"
    assert e.influence == [InfluenceLine("US", 1, CANADA, 4, 0),
                       InfluenceLine("US", 1, GREECE, 1, 0)]


def test_that_card_is_not_taken_for_an_ask_not_draw(corpus: ReplayCorpus) -> None:
    raws = corpus.game(262)["all_turns"]
    held = _held(corpus, 262, 8, "US")
    arab_israeli = card_id("Arab-Israeli War")
    assert arab_israeli in held, "the turn's list has it"
    acquired = _mid_turn_acquisitions(raws, 8, "US", held)
    assert arab_israeli not in acquired, (
        "it is played at AR3, so it cannot have been drawn later")


def test_something_is_still_taken_for_the_draw(corpus: ReplayCorpus) -> None:
    """The hand list is one longer than a dealt hand, so a draw there must be."""
    raws = corpus.game(262)["all_turns"]
    held = _held(corpus, 262, 8, "US")
    assert len(held) == 10, "nine dealt and one drawn"
    acquired = _mid_turn_acquisitions(raws, 8, "US", held)
    assert len(acquired) == 1, "exactly as many draws as there were discards"


def test_replay_262_converts_end_to_end(corpus: ReplayCorpus) -> None:
    conv = corpus.converted(262)
    assert conv.failure is None, f"replay 262 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 143


@pytest.mark.parametrize("replay_id", [262, 159, 64])
def test_the_games_that_turn_on_a_named_card_convert(replay_id: int, corpus: ReplayCorpus) -> None:
    """Replay 159 names a card the hand list omits; replay 64 reclaims one with SALT."""
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
