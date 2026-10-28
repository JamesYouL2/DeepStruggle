"""Whose coup a gain belongs to is decided by who lost Influence, not by whose entry it is.

A coup takes the opponent's Influence away and then puts the couper's down, so within one coup
the side that *loses* Influence is the victim and the side that gains in the same country is
the couper. The victim gaining there is no coup's doing; the only thing that does it is NORAD,
which adds 1 US Influence after an action round in which the US lost some.

Reading the rows against the entry's own player breaks as soon as an entry holds two coups
belonging to two different sides. At turn 7 AR6 of replay 184 the US plays "Lone Gunman", the
USSR coups Nigeria with the Ops the event grants them, and the US then coups Nigeria back with
the card's own -- so the USSR's own gain looked like someone else's placement, and two phantom
points were queued against Nigeria.

A failed coup is the other end of it. It removes nothing, so there is no loss to name a victim
with, and nothing printed afterwards is the coup's: at turn 6 AR3 of replay 150 the USSR's coup
on Angola fails, DEFCON drops to 2, and the "US +1 in Angola" that follows is NORAD's.

Where an entry has several Ops sections the flat queue is cleared and the sections decide, and
a row no section claims has to go somewhere -- NORAD prints its placement loose at the foot of
the entry. Those points are asked for last of all, after every queue that can say what a
decision is for: offered earlier they answered Che's free coup with a country Che may not
touch, and Ortega's the same way.
"""
import pytest

from tools.lib.ts_replayer_convert import point_queue
from tools.lib.ts_replayer_parse import InfluenceLine
from tests.replay_corpus import ReplayCorpus

NIGERIA, PANAMA, ANGOLA, ARGENTINA, POLAND = 53, 70, 59, 82, 15


def test_two_coups_by_two_sides_queue_two_targets_and_one_placement(corpus: ReplayCorpus) -> None:
    """Replay 184 turn 7 AR6: the USSR coups Nigeria, the US coups it back, NORAD to Panama."""
    e = corpus.entry(184, 7, "AR6", "US")
    assert e.card == '"Lone Gunman"*'
    assert e.targets == [NIGERIA, NIGERIA]
    assert [s.mode for s in e.sections] == ["coup", "coup"]
    assert point_queue(e) == [NIGERIA, NIGERIA, PANAMA], (
        "the USSR's own gain in Nigeria is its coup's, not a placement")


def test_a_failed_coup_explains_nothing_that_follows_it(corpus: ReplayCorpus) -> None:
    """Replay 150 turn 6 AR3: the coup on Angola fails and NORAD places there."""
    e = corpus.entry(150, 6, "AR3", "USSR")
    assert e.coup_rolls == [(1, False)]
    assert e.influence == [InfluenceLine("US", 1, ANGOLA, 2, 0)], "no removal: the coup took nothing"
    assert point_queue(e) == [ANGOLA, ANGOLA], "the target, then NORAD's placement"


def test_the_victim_gaining_in_the_couped_country_is_still_norad(corpus: ReplayCorpus) -> None:
    """Replay 131 turn 6 AR1: the USSR coups Argentina, the US puts NORAD back into it."""
    e = corpus.entry(131, 6, "AR1", "USSR")
    assert point_queue(e) == [ARGENTINA, ARGENTINA]


def test_a_placement_in_another_country_is_unchanged(corpus: ReplayCorpus) -> None:
    """Replay 104 turn 6 AR3: couped Panama, NORAD into Poland."""
    assert point_queue(corpus.entry(104, 6, "AR3", "USSR")) == [PANAMA, POLAND]


@pytest.mark.parametrize("replay_id", [184, 150, 131, 104, 143, 123, 105, 173, 129])
def test_the_coup_and_norad_games_convert(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
