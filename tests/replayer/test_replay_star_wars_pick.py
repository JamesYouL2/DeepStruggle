"""Star Wars takes the card the log says it fired, not one another card revealed.

Star Wars reaches into the discard pile and plays any non-scoring card there as its event, and
the log names that card on an "Event:" line of its own. A card revealed by something else in
the same entry can be sitting in that pile too.
"""
from __future__ import annotations

import pytest

from tests.replay_corpus import ReplayCorpus

_BEAR_TRAP = 44
_SALT_NEGOTIATIONS = 43


def test_the_entry_names_both_the_reveal_and_the_pick(corpus: ReplayCorpus) -> None:
    """Replay 291 turn 10: Aldrich Ames Remix reads the US hand and discards SALT Negotiations
    out of it; Star Wars then plays Bear Trap out of the pile."""
    e = corpus.entry(291, 10, "Headline", "both")
    assert "SALT Negotiations*" in [nm for _side, nm in e.revealed]
    assert [nm for _side, nm in e.discards] == ["SALT Negotiations*"]
    assert "Bear Trap*" in e.events


def test_replay_291_traps_the_ussr_for_turn_10(corpus: ReplayCorpus) -> None:
    """SALT, freshly discarded and freshly revealed, was offered to Star Wars first. With Bear
    Trap never in play, the USSR spent turn 10 playing cards where the log has them discarding
    Nuclear Subs to the trap and failing to escape."""
    conv = corpus.converted(291)
    assert conv.failure is None, f"replay 291 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 144


@pytest.mark.parametrize("replay_id", [104, 158, 234])
def test_the_other_star_wars_games_are_unaffected(replay_id: int, corpus: ReplayCorpus) -> None:
    """104 takes How I Learned To Stop Worrying, 158 takes Grain Sales To Soviets, and 234's
    entry answers a peek from a queue Star Wars must not touch."""
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
