"""A card removed from the game stays removed.

A starred card whose event occurs leaves the game, and no rule brings it back. The hand solver
knew only that a spent card is in the discard pile until a reshuffle, so at the next *RESHUFFLE*
it freed a removed card like any other, used it to fill a hand, and the converter seated it out of
the engine's removed cards -- in 252 of the corpus's 300 games, at 19% of all converted
decisions. Once unseated it went to the deck, so it stayed in circulation: each one left the
engine's deck a card bigger than the humans', and in four games the deck still held cards at the
turn-7 deal the log reshuffles at (CONV-4).
"""
import gzip
import json
import os
from typing import Dict, List, Set

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import card_id, convert_game
from tools.lib.ts_replayer_hands import GameFacts
from tools.lib.ts_replayer_parse import Entry, parse_entry

CORPUS = str(corpus_dir())
BLOCKADE = 10
OUR_MAN_IN_TEHRAN = 108


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _brought_back(replay_id: int) -> List[str]:
    """Every point of the conversion at which a card the engine had removed is somewhere else."""
    gone: Set[int] = set()
    out: List[str] = []

    def look(st: ts.GameState, e: Entry) -> None:
        for c in range(1, 111):
            loc = st.get_card_location(c)
            if loc == ts.CardLocation.REMOVED_FROM_GAME:
                gone.add(c)
            elif c in gone:
                out.append(f"turn {e.turn} {e.phase} {e.player}: "
                           f"{ts.CardData.get_card_name(c)} at {loc}")

    conv = convert_game(_game(replay_id), on_decision=lambda st, _m, e, _a: look(st, e),
                        on_entry=look)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    return out


@pytest.mark.parametrize("replay_id", [59, 119, 156, 228, 302])
def test_a_removed_card_never_comes_back(replay_id: int) -> None:
    """The five games whose *RESHUFFLE* markers the engine missed, each by exactly as many cards
    as it had brought back by then: 228 Blockade, 302 Romanian Abdication, 59 Nasser and Marshall
    Plan, 119 Blockade and Comecon, 156 Warsaw Pact, Captured Nazi Scientist and Panama Canal
    Returned."""
    assert _brought_back(replay_id) == []


def test_the_solver_knows_blockade_was_removed() -> None:
    """Replay 228: the US plays Blockade at turn 2 AR5, which fires it, and the solved US
    hand held it again at turn 3."""
    game = _game(228)
    facts = GameFacts(game["all_turns"], game.get("hands") or {}, card_id)
    assert facts.removed[BLOCKADE] == 2


@pytest.mark.parametrize("replay_id", [59, 119, 228, 302])
def test_the_turn_7_deal_reshuffles_where_the_log_does(replay_id: int) -> None:
    """The log reshuffles while dealing turn 7, so the whole pile goes into the new deck and the
    first decision of the turn finds it empty. With the removed cards in circulation the engine's
    deck still held cards there, and the pile was not reshuffled until later."""
    game = _game(replay_id)
    first = next(parse_entry(r) for r in game["all_turns"] if parse_entry(r).turn == 7)
    assert "*RESHUFFLE*" in first.text
    piles: List[Set[int]] = []

    def on_decision(st: ts.GameState, _mover: ts.Player, e: Entry, _chosen: int) -> None:
        if e.turn == 7 and not piles:
            piles.append({c for c in range(1, 111)
                          if st.get_card_location(c) == ts.CardLocation.DISCARD_PILE})

    conv = convert_game(game, on_decision=on_decision)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    assert piles == [set()], sorted(ts.CardData.get_card_name(c) for c in piles[0])


def test_our_man_in_tehran_is_not_in_its_own_peek() -> None:
    """Replay 277, turn 8's headline: the US's Five Year Plan discards Our Man in Tehran out of
    the USSR hand, which fires it. The peek was seeded from every discard in the entry, the
    USSR's included, so the removed card was put among the five the US looked at and the US
    discarded it into the pile."""
    assert _brought_back(277) == []
    peeks: List[Set[int]] = []

    def on_decision(st: ts.GameState, _mover: ts.Player, e: Entry, _chosen: int) -> None:
        if (e.turn, e.phase) == (8, "Headline"):
            peek = {c for c in range(1, 111)
                    if st.get_card_location(c) == ts.CardLocation.PEEKED_TEMP}
            if peek:
                peeks.append(peek)
            assert st.get_card_location(OUR_MAN_IN_TEHRAN) != ts.CardLocation.DISCARD_PILE

    conv = convert_game(_game(277), on_decision=on_decision)
    assert conv.failure is None, f"replay 277 stopped at {conv.failure}"
    assert peeks and all(OUR_MAN_IN_TEHRAN not in p for p in peeks)
