"""A coup's die is stated in the log, and is carried on the target choice.

The log prints it on the result line ("SUCCESS: 6 [ + 2 - 2x2 = 4 ]"). The converter used to
ignore it and search the rng for a seed that reproduced the resulting board instead, which
fails outright whenever anything else in the entry touches the same country -- the expectation
then holds that country's *final* value, which no single roll can produce.

The die rides on the target choice in secondary_id rather than being handed to the chance node
that usually follows, because Che's two free coups have no such node: they resolve inside the
event, where secondary_id is the only way in.
"""

from tools.lib.ts_replayer_parse import parse_entry
from tests.replay_corpus import ReplayCorpus, failed_at


def _entry(text: str):
    return parse_entry({"num": "4", "player": "USSR", "phase": "AR4",
                        "card": "Che", "text": text})


def test_every_coup_in_an_entry_keeps_its_own_die() -> None:
    """Che coups twice, and coup_roll held only the last."""
    e = _entry("Turn 4, USSR AR4: Che: Event: Che\n"
               "Coup (3 Ops):\nTarget: Sudan\nSUCCESS: 1 [ + 3 - 2x1 = 2 ]\n"
               "US -1 in Sudan [0][0]\nUSSR +1 in Sudan [0][1]\n"
               "Coup (3 Ops):\nTarget: Saharan States\nSUCCESS: 6 [ + 3 - 2x1 = 7 ]\n"
               "US -4 in Saharan States [0][0]\nUSSR +3 in Saharan States [0][3]\n")
    assert e.coup_rolls == [(1, True), (6, True)]
    assert e.coup_roll == 6, "the single field still holds the last, for existing callers"


def test_a_failed_coup_records_its_die_too() -> None:
    e = _entry("Coup (2 Ops):\nTarget: Panama\nFAILURE: 1 [ + 2 - 2x2 = -1 ]\n")
    assert e.coup_rolls == [(1, False)]


def test_ches_two_coups_reproduce_the_logged_boards(corpus: ReplayCorpus) -> None:
    """Replay 103 turn 4 AR4: dice of 1 and 6 into Sudan and Saharan States.

    The count is the end of turn 5, not the end of the file: replay 103's last turn stops
    short of its final action round, so it is a fragment and is given back. Turn 4 -- the one
    Che's coups are in -- converts either way, which is what this test is about.
    """
    conv = corpus.converted(103)
    assert conv.failure is None, f"replay 103 stopped at {conv.failure}"
    assert conv.entries_converted > 60, "well past the turn 4 AR4 the coups are in"


def test_a_coup_whose_country_is_touched_again_in_the_same_entry(corpus: ReplayCorpus) -> None:
    """Replay 129 turn 4 headline: Junta coups Panama, then Liberation Theology places there.

    No rng seed could ever reproduce Panama's final [1][1] from the coup alone, so this entry
    stopped the game at 39 of 114 entries.
    """
    conv = corpus.converted(129)
    assert failed_at(conv) != "T4 Headline", f"replay 129 stops at its coup: {conv.failure}"
    assert conv.entries_converted > 100
