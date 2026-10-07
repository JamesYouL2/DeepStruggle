"""The hands a converted position holds are the hands the players had (BUGS.md CONV-1).

The converter checks the board and the score against the log at every entry, but never the
hands, so a hand could hold cards the player did not have without anything failing. Three ways
it did, each pinned on the game that showed it:

* Our Man in Tehran's cards -- seen, never held -- were seated in the US hand from AR1;
* cards revealed out of a hand were left out of the solved deal, replaced by others, and then
  added back as late arrivals, so the hand held both;
* a card that does arrive mid-turn (SALT Negotiations reclaiming one) must arrive then, not
  before -- and an Ask Not draw then, not at its first use;
* a card the turn's list names that nothing shows the side holding is not seated at all;
* a turn whose reveals the rest of the log contradicts is solved without them, reported, and
  kept out of the decisions.
"""

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, List, Set, Tuple

import ts_engine as ts

from tools.lib.corpus_paths import corpus_dir
from tools.lib.ts_replayer_convert import card_id, convert_game

CHINA = 6


def _game(replay: int) -> Dict[str, Any]:
    with gzip.open(os.path.join(str(corpus_dir()), f"{replay}.json.gz"), "rt") as f:
        return json.load(f)


def _hands_by_decision(replay: int, turn: int, side: ts.Player) -> List[Tuple[str, int, Set[int]]]:
    """(phase, action round, the side's hand without the China Card) at each of the turn's decisions."""
    out: List[Tuple[str, int, Set[int]]] = []

    def on(st: ts.GameState, mover: ts.Player, entry: object, chosen: int) -> None:
        if int(st.turn) == turn:
            hand = {c for c in range(1, 111) if c != CHINA and ts.in_hand_of(st.get_card_location(c), side)}
            out.append((str(st.current_phase).split(".")[-1], int(st.action_round), hand))

    convert_game(_game(replay), on_decision=on)
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


def test_a_card_ask_not_draws_is_in_hand_from_the_draw() -> None:
    # Replay 35, turn 7: the US headlines Ask Not, discards six and draws six. Brush War is one of
    # the draws -- the solved deal leaves it out -- and the US plays it at AR6. It was seated just
    # before AR6, so the US hand lacked it through AR1-AR5.
    brush_war = card_id("Brush War")
    rounds = [(ar, h) for phase, ar, h in _hands_by_decision(35, 7, ts.Player.US) if phase == "ACTION_ROUND"]
    assert {ar for ar, _ in rounds} >= {1, 2, 3, 4, 5}
    for ar, hand in rounds:
        if ar <= 5:
            assert brush_war in hand, f"AR{ar}: the US has not drawn Brush War yet"


def test_a_listed_card_nothing_shows_the_side_holding_is_not_seated() -> None:
    # Replay 285, turn 7: both hand lists name ABM Treaty, which the USSR headlines, takes back
    # with SALT Negotiations and plays at AR7. Nothing shows the US with it; it was seated in the
    # US hand from the turn's start.
    abm = card_id("ABM Treaty")
    for phase, ar, hand in _hands_by_decision(285, 7, ts.Player.US):
        assert abm not in hand, f"{phase} AR{ar}: the US holds the USSR's ABM Treaty"
    assert convert_game(_game(285)).listed_never_held == 1


def test_a_turn_whose_reveals_the_log_contradicts_is_reported_and_not_emitted() -> None:
    # Replay 212, turn 6: "Lone Gunman" reveals six US cards and the US later plays a seventh.
    # Replay 321, turn 10: Aldrich Ames reveals a card the model has in the discard pile. The
    # hands are solved without those turns' reveals -- and only those turns' -- so they are not
    # known to be right, and nothing downstream may take a decision from them.
    for replay, turn in ((212, 6), (321, 10)):
        turns: Set[int] = set()

        def on(st: ts.GameState, mover: ts.Player, entry: object, chosen: int) -> None:
            turns.add(int(st.turn))

        conv = convert_game(_game(replay), on_decision=on)
        assert conv.reveal_conflict_turns == [turn]
        assert turn not in turns and turn - 1 in turns
