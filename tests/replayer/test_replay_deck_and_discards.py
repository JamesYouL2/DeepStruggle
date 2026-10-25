"""A card the log does not put in a hand is in the deck, not the discard pile.

The log never lists a hand in full, so before every entry the converter seats the hands it has
solved for and takes every other card out of them. Those cards used to go to the discard pile.
At 99.7% of the corpus's decisions the pile then held cards nobody had played, 14 on average,
and the engine reshuffled them in where the humans had not: at turn 2, putting back in the deck
the very cards the players had just spent. The observation's discard features read that pile, and
so does every redeal from a converted position.

A card the engine dealt and the humans were never dealt is still in the deck. Anything a hand
really gave up went through the engine, which moved it to the discard pile or out of the game.
"""
import gzip
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Set

import pytest
import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import card_id, convert_game
from tools.lib.ts_replayer_parse import Entry, parse_entry

CORPUS = str(corpus_dir())
THE_CHINA_CARD = 6
RED_SCARE_PURGE = 31
SALT_NEGOTIATIONS = 43


@dataclass(frozen=True)
class _Piles:
    """The engine's piles at one decision, and what the log has spent since its last reshuffle."""
    where: str
    discard: Set[int]
    deck: Set[int]
    spent: Set[int]


def _game(replay_id: int) -> Dict:
    with gzip.open(os.path.join(CORPUS, f"{replay_id}.json.gz"), "rt") as f:
        return json.load(f)


def _spent(e: Entry) -> Set[int]:
    """The cards an entry shows leaving a hand: headlined, played, played through another, discarded."""
    names: List[str] = list(e.headlines.values())
    if e.card:
        names += [n.strip() for n in e.card.split(" & ")]
    if e.played_card:
        names.append(e.played_card)
    names += [nm for _side, nm in e.discards]
    cards = {card_id(nm.replace(" from hand", "").strip(" .")) for nm in names}
    return {c for c in cards if c and c != THE_CHINA_CARD}


def _piles_at_each_decision(replay_id: int, last_turn: int) -> List[_Piles]:
    """The piles at every decision up to `last_turn`."""
    game = _game(replay_id)
    entries = [parse_entry(r) for r in game["all_turns"]]
    spent_by: Dict[int, Set[int]] = {}
    since_reshuffle: Set[int] = set()
    for i, e in enumerate(entries):
        if "*RESHUFFLE*" in e.text:
            since_reshuffle = set()
        since_reshuffle |= _spent(e)
        spent_by[i] = set(since_reshuffle)
    rows: List[_Piles] = []
    cursor = [0]

    def on_decision(st: ts.GameState, mover: ts.Player, e: Entry, chosen: int) -> None:
        if e.turn > last_turn:
            return
        while (entries[cursor[0]].turn, entries[cursor[0]].phase, entries[cursor[0]].player,
               entries[cursor[0]].text) != (e.turn, e.phase, e.player, e.text):
            cursor[0] += 1
        at = {loc: {c for c in range(1, 111) if st.get_card_location(c) == loc}
              for loc in (ts.CardLocation.DISCARD_PILE, ts.CardLocation.DRAW_DECK)}
        rows.append(_Piles(f"turn {e.turn} {e.phase} {e.player}",
                           at[ts.CardLocation.DISCARD_PILE], at[ts.CardLocation.DRAW_DECK],
                           spent_by[cursor[0]]))

    conv = convert_game(game, on_decision=on_decision)
    assert conv.failure is None, f"replay {replay_id} stopped at {conv.failure}"
    return rows


def _name(cards: Set[int]) -> List[str]:
    return sorted(ts.CardData.get_card_name(c) for c in cards)


def test_turn_1_of_replay_51_discards_only_what_was_played() -> None:
    """At turn 1 AR2 of replay 51 the pile held 13 cards, 9 of them never played --
    Middle East Scoring among them."""
    rows = [r for r in _piles_at_each_decision(51, 1) if r.where.startswith("turn 1 AR2")]
    assert rows
    for r in rows:
        assert r.discard <= r.spent, f"{r.where}: never played {_name(r.discard - r.spent)}"


@pytest.mark.parametrize("replay_id", [51, 107, 119, 206])
def test_the_early_war_piles_are_the_logs(replay_id: int) -> None:
    """Through turn 3 the discard pile is exactly the log's: nothing in it the humans did not
    spend, and nothing they spent back in the deck by a reshuffle they never had."""
    rows = _piles_at_each_decision(replay_id, 3)
    assert rows
    for r in rows:
        assert r.discard <= r.spent, f"{r.where}: never played {_name(r.discard - r.spent)}"
        assert not r.spent & r.deck, (
            f"{r.where}: spent and back in the deck {_name(r.spent & r.deck)}")


def test_salt_takes_back_the_card_the_log_names() -> None:
    """Replay 117 turn 9 AR5: "Event: SALT Negotiations* ... US reveals Red Scare/Purge".

    SALT lets its player take a card back out of the discard pile. With the pile padded, the
    engine reshuffled at the wrong times and Red Scare/Purge was back in the deck when SALT
    came, so the logged pick was not legal and the decision was dropped. 24 games lost a
    decision this way.
    """
    picks: List[int] = []

    def on_decision(st: ts.GameState, mover: ts.Player, e: Entry, chosen: int) -> None:
        if st.ctx().resolving_card == SALT_NEGOTIATIONS and (e.turn, e.phase) == (9, "AR5"):
            picks.append(ts.ActionMask.decode_flat_action(st, chosen).primary_id)

    conv = convert_game(_game(117), on_decision=on_decision)
    assert conv.failure is None, f"replay 117 stopped at {conv.failure}"
    assert picks == [RED_SCARE_PURGE]
