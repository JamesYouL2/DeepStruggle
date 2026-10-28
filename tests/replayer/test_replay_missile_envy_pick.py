"""Missile Envy asks which card to hand over, and only its own reveal answers.

The card it takes is the opponent's highest Ops one, and when two tie the giver chooses. The
log settles that by naming what was revealed -- but an entry can print more than one reveal,
and the driver was answering from whichever came first.

At turn 4's headline of replay 14 the US headlines Grain Sales To Soviets and the USSR
headlines Missile Envy. Grain Sales resolves first (both are 2 Ops and the US wins ties): it
takes Marshall Plan at random from the USSR hand, the US plays it, and it coups Brazil. Missile
Envy then asks the US for a card, the highest Ops in that hand tie, and the reveal at the head
of the queue was Grain Sales' Marshall Plan rather than Missile Envy's own Muslim Revolution.
"""
from tools.lib.ts_replayer_convert import _KNOWN_INCOMPLETE, _missile_envy_took, card_id
from tools.lib.ts_replayer_parse import parse_entry
from tests.replay_corpus import ReplayCorpus


def test_the_entry_prints_two_reveals_and_only_one_is_missile_envys(corpus: ReplayCorpus) -> None:
    e = corpus.entry(14, 4, "Headline")
    names = [nm for _side, nm in e.revealed]
    assert names.count("Marshall Plan*") == 2, "Grain Sales' reveal, printed twice"
    assert "Muslim Revolution" in names
    assert names[0] == "Marshall Plan*", "and it stands at the head of the queue"
    took = _missile_envy_took(e)
    assert took == ("US", card_id("Muslim Revolution")), (
        "the reveal under Event: Missile Envy is the one that answers")


def test_replay_14_converts_end_to_end(corpus: ReplayCorpus) -> None:
    conv = corpus.converted(14)
    assert conv.failure is None, f"replay 14 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total


def test_a_headline_the_log_stops_inside_is_listed_not_guessed(corpus: ReplayCorpus) -> None:
    """Replay 133's last entry is one line: "USSR Headlines Missile Envy".

    No US headline, no reveal, and no answer to the same tie -- ABM Treaty against Muslim
    Revolution, both 4 Ops. Nothing in the record decides it, so the game is converted up to
    that entry and no further rather than picking one.
    """
    raws = corpus.game(133)["all_turns"]
    last = parse_entry(raws[-1])
    assert (last.turn, last.phase) == (10, "Headline")
    assert last.revealed == []
    assert last.headlines == {"USSR": "Missile Envy"}, "the US headline is not recorded"
    assert (10, "Headline", "both") in _KNOWN_INCOMPLETE[133]

    conv = corpus.converted(133)
    assert conv.failure is None
    assert conv.truncated_at is not None
    assert conv.entries_converted == len(raws) - 1
