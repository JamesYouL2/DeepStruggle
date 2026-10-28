"""How I Learned To Stop Worrying is read off the level the log prints, where it prints one.

The card's branches are the five DEFCON levels. Stepping one runs the rest of the action round
with it -- including the turn end, which improves DEFCON by one -- so the board a branch
reaches cannot tell 4 from 5. The branch itself can, and the log names the level outright.
"""
from __future__ import annotations

from typing import Dict, List, cast

import pytest

from tools.lib.ts_replayer_convert import _defcon_after, _defcon_set_under
from tools.lib.ts_replayer_parse import parse_entry
from tests.replay_corpus import ReplayCorpus

_MARKER = "Event: How I Learned To Stop Worrying*"


def test_the_level_is_read_from_under_the_cards_own_header(corpus: ReplayCorpus) -> None:
    """Replay 32 turn 8 AR7: "DEFCON improves to 5", printed by the card itself."""
    e = corpus.entry(32, 8, "AR7", "US")
    assert _defcon_set_under(e, _MARKER) == 5


def test_an_entry_that_does_not_print_one_falls_back_to_the_next(corpus: ReplayCorpus) -> None:
    """Replay 260 turn 5 AR7 narrates only the 5 Military Operations, so the level chosen
    shows up in the entry that follows."""
    e = corpus.entry(260, 5, "AR7", "USSR")
    assert _defcon_set_under(e, _MARKER) is None
    raws = cast(List[Dict[str, object]], corpus.game(260)["all_turns"])
    index = next(i for i, r in enumerate(raws)
                 if (parse_entry(r).turn, parse_entry(r).phase, parse_entry(r).player)
                 == (5, "AR7", "USSR"))
    assert _defcon_after(raws, index, e) == 3


def test_replay_32_pays_the_military_operations_the_level_it_set_is_worth(corpus: ReplayCorpus) -> None:
    """The US sets DEFCON to 5 with the USSR two short of it and the turn ends immediately.
    Taking 4 instead -- which the turn end also improves to 5 -- paid the US 1 VP where the log
    pays 2, and the game was a VP short from there on."""
    conv = corpus.converted(32)
    assert conv.failure is None, f"replay 32 stopped at {conv.failure}"
    assert conv.entries_converted == conv.entries_total == 154


@pytest.mark.parametrize("replay_id", [260, 101, 104])
def test_the_other_games_with_this_card_are_unaffected(replay_id: int, corpus: ReplayCorpus) -> None:
    conv = corpus.converted(replay_id)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
