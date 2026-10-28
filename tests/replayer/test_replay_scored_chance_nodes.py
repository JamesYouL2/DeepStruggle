"""Some chance nodes leave no trace on the board, only on the score.

A die is steered by seeding the engine's stream until the result matches what the log recorded,
and that match was made against country influence. Olympic Games and Summit move no Influence
at all, so nothing matched and the winner was a coin flip -- which puts the points on the wrong
side half the time, a four VP swing from one roll.

The log records who gained them, so the score is what the seed is chosen against instead.
"""
from tools.lib.ts_replayer_parse import parse_entry
from tests.replay_corpus import ReplayCorpus


def _failed_at(conv):
    return None if conv.failure is None else (conv.failure.turn, conv.failure.phase)


def test_the_olympics_go_to_the_side_the_log_says_won(corpus: ReplayCorpus) -> None:
    """Replay 245 turn 6 headline: the US wins and takes 2 VP, reaching 6.

    The reconstruction gave them to the USSR and the score went to 2 instead.
    """
    conv = corpus.converted(245)
    assert _failed_at(conv) != (6, "Headline"), (
        f"replay 245 stops at its Olympics: {conv.failure}")


def test_the_same_holds_when_the_ussr_wins_them(corpus: ReplayCorpus) -> None:
    """Replay 165 turn 3 headline: the USSR wins and takes 2 VP, reaching USSR 8."""
    conv = corpus.converted(165)
    assert _failed_at(conv) != (3, "Headline"), (
        f"replay 165 stops at its Olympics: {conv.failure}")


def test_the_log_records_the_winner_but_never_the_dice(corpus: ReplayCorpus) -> None:
    """Which is why the board cannot settle it and the score has to."""
    raws = corpus.turns(245)
    headline = next(r for r in raws
                    if (lambda e: e.turn == 6 and e.phase == "Headline")(parse_entry(r)))
    entry = parse_entry(headline)
    assert "Olympic Games" in (entry.card or "")
    assert not entry.die_rolls and not entry.bare_rolls, "no dice are printed"
    assert entry.vp_gains, "only the outcome is"
