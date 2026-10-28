"""NORAD's Influence is a placement even when it lands in the country just couped.

A coup takes the opponent's Influence away and puts the couper's down; a realignment only takes
away. The *couped* player gaining Influence in that same country is therefore not a result of
the roll at all, and the only thing in the game that does it is NORAD, which lets the US add 1
Influence after an action round in which it lost some.

The driver queued a coup entry's trailing Influence lines only where they named a country that
was not the target, which is right for everything a coup itself does. At turn 6 AR3 of replay
104 the US put its NORAD Influence into Poland and it was queued without trouble; at turn 6 AR1
of replay 131 the USSR coups Argentina and the US puts it straight back into Argentina, and the
placement was dropped -- leaving the engine asking which of 32 countries the US meant, with the
log's own answer thrown away.
"""
import pytest

from tools.lib.ts_replayer_convert import point_queue
from tools.lib.ts_replayer_parse import InfluenceLine
from tests.replay_corpus import ReplayCorpus

ARGENTINA = 82
PANAMA = 70
POLAND = 15


def test_norad_into_the_couped_country_is_queued(corpus: ReplayCorpus) -> None:
    """Replay 131 turn 6 AR1: the USSR coups Argentina, the US puts 1 back into Argentina."""
    e = corpus.entry(131, 6, "AR1", "USSR")
    assert e.mode == "coup" and e.targets == [ARGENTINA]
    assert InfluenceLine("US", -1, ARGENTINA, 1, 0) in e.influence, "the coup's removal"
    assert InfluenceLine("US", 1, ARGENTINA, 2, 0) in e.influence, "NORAD's placement"
    assert point_queue(e) == [ARGENTINA, ARGENTINA], (
        "the target, and then the placement the log records after it")


def test_norad_into_another_country_still_works(corpus: ReplayCorpus) -> None:
    """Replay 104 turn 6 AR3: couped Panama, NORAD into Poland -- the case that worked."""
    e = corpus.entry(104, 6, "AR3", "USSR")
    assert point_queue(e) == [PANAMA, POLAND]


def test_the_coupers_own_influence_is_not_a_placement(corpus: ReplayCorpus) -> None:
    """A successful coup putting the couper's Influence down is the roll's result.

    Only the couped player gaining is unexplainable by the coup, so only that is queued.
    """
    e = corpus.entry(143, 4, "AR1", "USSR")
    assert e.targets == [PANAMA]
    gains = [(row.side, row.delta) for row in e.influence
             if row.country == PANAMA and row.delta > 0]
    assert gains == [("US", 1)], "the USSR couped, so a US gain is NORAD's"
    assert point_queue(e) == [PANAMA, PANAMA]


@pytest.mark.parametrize("replay_id", [131, 143, 104])
def test_the_norad_games_convert_end_to_end(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
