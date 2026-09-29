"""Doctrine (bot/doctrine): struggler's strategic bot, ported onto ts_engine.

What these pin:

* the evaluator reproduces struggler's own `board_value` exactly on a fixed board -- the golden
  numbers were computed by struggler's evaluator, so the port cannot drift from it silently;
* the bot never reads what its player cannot see: its choice does not depend on how the
  opponent's hidden cards are distributed;
* every action it picks is legal, through a played turn against a fixed opponent;
* it plays its scoring cards when holding them any longer would lose the game.
"""
from __future__ import annotations

import random
from typing import List, Tuple

import pytest

import ts_engine as ts
from bot.doctrine import DoctrineAgent
from bot.doctrine import evaluator as ev
from bot.doctrine.policy import DoctrinePolicy, determinize, legal_actions

# Influence (US, USSR) per country 0..83 and a per-region urgency (plus Southeast Asia's), both
# arbitrary; the values below are struggler's evaluator on exactly this board.
_INF: List[Tuple[int, int]] = [
    (2, 0), (1, 0), (5, 3), (1, 3), (0, 5), (0, 2), (4, 0), (0, 1), (2, 0), (1, 5), (2, 3), (0, 1),
    (0, 1), (0, 1), (0, 3), (2, 4), (0, 0), (3, 0), (1, 1), (0, 5), (4, 4), (0, 0), (3, 3), (0, 0),
    (0, 0), (3, 3), (0, 0), (1, 0), (2, 0), (0, 5), (0, 0), (3, 0), (0, 0), (0, 3), (0, 0), (3, 3),
    (2, 2), (1, 1), (0, 1), (4, 2), (0, 4), (0, 1), (1, 4), (0, 5), (4, 3), (0, 2), (2, 0), (0, 5),
    (4, 4), (4, 4), (0, 4), (0, 4), (0, 5), (2, 0), (2, 4), (2, 1), (0, 0), (3, 2), (0, 0), (5, 0),
    (5, 3), (3, 1), (0, 2), (0, 5), (0, 5), (5, 1), (4, 3), (3, 0), (5, 0), (0, 0), (4, 1), (4, 0),
    (5, 3), (0, 0), (0, 2), (4, 4), (5, 2), (0, 1), (4, 5), (0, 4), (4, 3), (0, 0), (1, 3), (0, 0),
]
_REGION_URGENCY = (0.9, 1.7, 1.2, 0.6, 0.8, 1.1)
_SEA_URGENCY = 0.5
#: struggler `evaluator.board_value` with `StrategicWeights()` defaults on the board above.
_STRUGGLER_US = -226.70104629621054


def _fixed_board() -> ts.GameState:
    state = ts.GameState()
    for i, (us, ussr) in enumerate(_INF):
        state.set_country(i, us, ussr)
    return state


def test_evaluator_matches_struggler_on_a_fixed_board() -> None:
    t = ev.terrain()
    urgency = [_REGION_URGENCY[t.region_of[i]] + (_SEA_URGENCY if i in t.southeast_asia else 0.0)
               for i in range(ev.N_COUNTRIES)]
    pos = ev.Position.of(_fixed_board(), t)
    assert ev.board_value(t, pos, ev.US, ev.Weights(), urgency) == pytest.approx(_STRUGGLER_US, rel=1e-12)
    assert ev.board_value(t, pos, ev.USSR, ev.Weights(), urgency) == pytest.approx(-_STRUGGLER_US, rel=1e-12)


def test_every_country_has_a_fitted_weight() -> None:
    t = ev.terrain()
    assert len(t.fitted[ev.US]) == len(t.fitted[ev.USSR]) == ev.N_COUNTRIES
    # A missing weight would make a country silently worthless; zero is never a fitted value
    # for a battleground.
    assert all(t.fitted[s][i] != 0.0 for s in (ev.US, ev.USSR) for i in range(ev.N_COUNTRIES)
               if t.battleground[i])


def _to_first_action_round(seed: int) -> ts.GameState:
    """Setup and headline played by Doctrine, up to the USSR's first card choice."""
    state = ts.GameState()
    ts.Engine.init_game(state, seed)
    bot = DoctrineAgent(seed=seed)
    while not (state.current_phase == ts.Phase.ACTION_ROUND
               and state.ctx().decision_type == ts.DecisionType.SELECT_CARD):
        c = state.ctx()
        if c.decision_type == ts.DecisionType.ROLL_DIE and c.decision_player == ts.Player.NONE:
            ts.Engine.step_flat(state, 115)
            continue
        assert ts.Engine.try_step_flat(state, bot.select_action(state, c.decision_player))
    return state


def test_choice_does_not_depend_on_the_opponents_hidden_cards() -> None:
    state = _to_first_action_round(5)
    me = state.ctx().decision_player
    hidden = ts.CardLocation.HAND_US_UNKNOWN if me == ts.Player.USSR else ts.CardLocation.HAND_USSR_UNKNOWN
    theirs = [c for c in range(1, 111) if state.get_card_location(c) == hidden]
    deck = [c for c in range(1, 111) if state.get_card_location(c) == ts.CardLocation.DRAW_DECK]
    assert theirs and deck
    # Swap the opponent's hidden hand with cards from the deck: the same information set.
    other = state.clone()
    for a, b in zip(theirs, deck):
        other.set_card_location(a, ts.CardLocation.DRAW_DECK)
        other.set_card_location(b, hidden)
    # The world the search runs on is the same, card for card...
    a = determinize(state, me, random.Random(1))
    b = determinize(other, me, random.Random(1))
    assert [a.get_card_location(c) for c in range(1, 111)] == [b.get_card_location(c) for c in range(1, 111)]
    # ...and so is the choice made on it.
    assert DoctrinePolicy(seed=1).choose(state) == DoctrinePolicy(seed=1).choose(other)


def test_plays_a_turn_with_only_legal_actions() -> None:
    state = _to_first_action_round(7)
    bot = DoctrineAgent(seed=7)
    turn = int(state.turn)
    steps = 0
    while not ts.Engine.is_terminal(state) and int(state.turn) == turn and steps < 400:
        steps += 1
        c = state.ctx()
        if c.decision_type == ts.DecisionType.ROLL_DIE and c.decision_player == ts.Player.NONE:
            ts.Engine.step_flat(state, 115)
            continue
        action = bot.select_action(state, c.decision_player)
        assert action in legal_actions(state)
        assert ts.Engine.try_step_flat(state, action)
    assert ts.Engine.is_terminal(state) or int(state.turn) > turn


def test_plays_scoring_cards_when_it_must() -> None:
    state = _to_first_action_round(3)
    me = state.ctx().decision_player
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), me):
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    # Two scoring cards and one ordinary card, with two rounds left: both must go now.
    state.action_round = 6
    for c in (1, 3, 4):
        state.set_card_location(c, ts.hand_of(me))
    action = DoctrinePolicy(seed=0).choose(state)
    assert action + 1 in (1, 3)


def test_does_not_keep_a_card_it_cannot_play_at_defcon_2() -> None:
    """The shape of a measured loss: the USSR's last card at DEFCON 2 was CIA Created, a US
    Event that hands the US an Op in the USSR's own round -- a coup on a battleground then takes
    DEFCON to 1 and the phasing USSR loses. At DEFCON 3 the card is still safe to play, so it goes
    now and the safe card is kept for the last round."""
    state = _to_first_action_round(101)
    me = state.ctx().decision_player
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), me):
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    state.china_card_holder = ts.Player.US if me == ts.Player.USSR else ts.Player.USSR
    state.set_country(57, 0, 1) if me == ts.Player.USSR else state.set_country(57, 1, 0)  # Zaire
    state.defcon = 3
    state.action_round = 5            # two rounds of ours left this turn
    cia, socialist = 26, 7
    state.set_card_location(cia, ts.hand_of(me))
    state.set_card_location(socialist, ts.hand_of(me))
    assert me == ts.Player.USSR
    assert DoctrinePolicy(seed=0).choose(state) == cia - 1
