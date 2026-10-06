"""The hands a converted position holds are the hands the players had (BUGS.md CONV-1).

The converter checks the board and the score against the log at every entry, but never the
hands, so a hand could hold cards the player did not have without anything failing. Three ways
it did, each pinned on the game that showed it:

* Our Man in Tehran's cards -- seen, never held -- were seated in the US hand from AR1;
* cards revealed out of a hand were left out of the solved deal, replaced by others, and then
  added back as late arrivals, so the hand held both;
* a card that does arrive mid-turn (SALT Negotiations reclaiming one) must arrive then, not
  before.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import List, Set, Tuple

import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import card_id, convert_game

CHINA = 6


def _hands_by_decision(replay: int, turn: int, side: ts.Player) -> List[Tuple[str, int, Set[int]]]:
    """(phase, action round, the side's hand without the China Card) at each of the turn's decisions."""
    with gzip.open(os.path.join(str(corpus_dir()), f"{replay}.json.gz"), "rt") as f:
        game = json.load(f)
    out: List[Tuple[str, int, Set[int]]] = []

    def on(st: ts.GameState, mover: ts.Player, entry: object, chosen: int) -> None:
        if int(st.turn) == turn:
            hand = {c for c in range(1, 111) if c != CHINA and ts.in_hand_of(st.get_card_location(c), side)}
            out.append((str(st.current_phase).split(".")[-1], int(st.action_round), hand))

    convert_game(game, on_decision=on)
    assert out, f"replay {replay} has no decisions in turn {turn}"
    return out


def _ids(*names: str) -> Set[int]:
    ids = {card_id(n) for n in names}
    assert None not in ids
    return {c for c in ids if c is not None}


def test_our_man_in_tehran_cards_are_never_held() -> None:
    # Replay 30, turn 4: the US plays Our Man in Tehran at AR7 and discards three of the cards it
    # shows. They were in the US hand from AR1, three over a mid-war hand.
    peeked = _ids("Africa Scoring", "Portuguese Empire Crumbles*", "Flower Power*")
    for phase, ar, hand in _hands_by_decision(30, 4, ts.Player.US):
        assert not hand & peeked, f"{phase} AR{ar}: the US holds cards it only saw"
        if phase == "ACTION_ROUND":
            assert len(hand) <= 8, f"AR{ar}: {len(hand)} cards after the headline"


def test_revealed_cards_were_dealt() -> None:
    # Replay 147, turn 9: CIA Created reveals eight USSR cards at the headline. Six were left out
    # of the solved deal for cards the log never shows, then added back at AR1: 14 cards.
    revealed = _ids("Solidarity*", "Socialist Governments", "Aldrich Ames Remix*", "South America Scoring",
                    "Asia Scoring", "Latin American Debt Crisis", "Reagan Bombs Libya*",
                    "Portuguese Empire Crumbles*")
    hands = _hands_by_decision(147, 9, ts.Player.USSR)
    headline = [h for phase, _, h in hands if phase == "HEADLINE"]
    assert headline and revealed <= headline[0]
    assert max(len(h) for phase, _, h in hands if phase == "ACTION_ROUND") <= 8


def test_a_card_reclaimed_by_salt_arrives_with_it() -> None:
    # Replay 14, turn 6: the US headlines SALT Negotiations and takes Junta back from the discard
    # pile. Junta is not in the hand at the headline and is from AR1 -- nine cards then is right.
    junta = card_id("Junta")
    hands = _hands_by_decision(14, 6, ts.Player.US)
    assert all(junta not in h for phase, _, h in hands if phase == "HEADLINE")
    first_ar = [h for phase, ar, h in hands if phase == "ACTION_ROUND" and ar == 1]
    assert first_ar and junta in first_ar[0] and len(first_ar[0]) == 9

