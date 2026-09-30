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
from bot.doctrine import schedule
from bot.doctrine.policy import (FATAL, SAFE, SPACE_ONLY, DoctrinePolicy, determinize, hand_survives,
                                 legal_actions)

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


# -- regressions for the port audit (docs/notes/codex/2026-09-28-doctrine-port-audit.md) -------

def _lopsided_for_ussr(state: ts.GameState, t: ev.Terrain) -> None:
    """The audit's F1 board: USSR influence at stability everywhere, the US restored only next to
    home. Struggler's raw value of it is about seven game swings for the USSR."""
    for i in range(ev.N_COUNTRIES):
        state.set_country(i, 0, t.stability[i])
    for i in t.home[ev.US]:
        state.set_country(i, 2 * t.stability[i], t.stability[i])


def test_a_finished_game_outranks_every_ongoing_position() -> None:
    """F1. The raw board value is unbounded; a win or loss is 40 VP at the VP price. Without one
    scale, US preferred losing at once to playing on from this board, and the USSR preferred it
    to winning."""
    state = _to_first_action_round(5)
    state.turn = 9
    pol = DoctrinePolicy(seed=0)
    _lopsided_for_ussr(state, pol.t)
    ctx = pol._prepare(state, ts.Player.US)
    us_lost, us_won = state.clone(), state.clone()
    us_lost.victory_points = -20
    us_won.victory_points = 20
    for s, bad, good in ((ev.US, us_lost, us_won), (ev.USSR, us_won, us_lost)):
        ongoing = pol.static(state, s, ctx)
        assert pol.static(bad, s, ctx) < ongoing < pol.static(good, s, ctx)
    # A dreadful board is still not a certain loss.
    assert not pol._certain_loss(pol.static(state, ev.US, ctx), ctx)
    assert pol._certain_loss(pol.static(us_lost, ev.US, ctx), ctx)


def test_one_space_attempt_cannot_dispose_of_two_cards() -> None:
    """F2. Each card was tested alone, so two cards that each survive only in space both counted
    as safe on one attempt."""
    assert not hand_survives([SPACE_ONLY, SPACE_ONLY], space_slots=1, china=False, rounds=2)
    assert hand_survives([SPACE_ONLY, SPACE_ONLY], space_slots=2, china=False, rounds=2)
    assert hand_survives([SPACE_ONLY, FATAL, SAFE], space_slots=1, china=False, rounds=2)
    assert hand_survives([SPACE_ONLY, SPACE_ONLY], space_slots=1, china=True, rounds=2)


def _two_rounds_after_this(state: ts.GameState, me: ts.Player, hand: Tuple[int, ...]) -> None:
    for c in range(1, 111):
        if ts.in_hand_of(state.get_card_location(c), me):
            state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    for c in hand:
        state.set_card_location(c, ts.hand_of(me))
    state.china_card_holder = ts.Player.US if me == ts.Player.USSR else ts.Player.USSR
    state.defcon = 2
    state.action_round = 5            # this round and two more of ours in a six-round turn
    state.us_space_track = 0
    state.ussr_space_track = 0
    state.set_space_turns_used(me, 0)


def test_duck_and_cover_survives_only_in_space_and_only_while_an_attempt_is_left() -> None:
    """F2, natively: at DEFCON 2 the USSR playing Duck and Cover fires the US Event, which takes
    DEFCON to 1 in the USSR's own round. Spacing it is the one way out."""
    state = _to_first_action_round(101)
    me = state.ctx().decision_player
    assert me == ts.Player.USSR
    duck, cia, socialist = 4, 26, 7
    _two_rounds_after_this(state, me, (duck, cia, socialist))
    pol = DoctrinePolicy(seed=0)
    ctx = pol._prepare(state, me)
    assert pol._exits_at_two(state, duck, ctx) == SPACE_ONLY
    state.set_space_turns_used(me, 1)
    assert pol._exits_at_two(state, duck, ctx) == FATAL


def test_the_hand_check_spends_the_space_attempt_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """F2, through the hazard: two space-only cards and a safe one, two rounds to fill after this
    one. With one attempt every choice leaves an unplayable round; with Animal in Space (two
    attempts) every choice leaves enough. Per-card testing called all three safe either way."""
    state = _to_first_action_round(101)
    me = state.ctx().decision_player
    a, b, safe = 4, 26, 7
    _two_rounds_after_this(state, me, (a, b, safe))
    exits = {a: SPACE_ONLY, b: SPACE_ONLY, safe: SAFE}
    pol = DoctrinePolicy(seed=0)
    monkeypatch.setattr(pol, "_exits_at_two", lambda world, card, ctx: exits[card])
    ctx = pol._prepare(state, me)
    ctx.rounds_left = 3
    choices = [c - 1 for c in (a, b, safe)]
    assert set(pol._defcon_hazard(state, ctx, choices)) == set(choices)
    state.ussr_space_track = 2        # Animal in Space: a second attempt this turn
    assert pol._defcon_hazard(state, ctx, choices) == {}


def test_one_op_price_only_considers_countries_reachable_on_the_board() -> None:
    """F3. Each candidate point was added and removed with a refresh in between, so the next
    candidate saw the reach the previous one created, and eligibility cascaded along the map."""
    state = _to_first_action_round(5)
    for i in range(ev.N_COUNTRIES):
        state.set_country(i, 0, 0)
    pol = DoctrinePolicy(seed=0)
    ctx = pol._prepare(state, ts.Player.US)
    root = ev.Position.of(state, pol.t)
    touched: List[int] = []
    board = pol._board

    def spy(pos: ev.Position, s: int, st: ts.GameState, c: object) -> float:
        touched.extend(i for i in range(ev.N_COUNTRIES) if pos.inf[s][i] > root.inf[s][i])
        return board(pos, s, st, c)  # type: ignore[arg-type]

    pol._board = spy  # type: ignore[method-assign]
    pol._one_op_value(state, ts.Player.US, ctx)
    assert touched, "no candidate was priced"
    assert all(root.reach[ev.US][i] for i in touched), \
        [pol.t.names[i] for i in touched if not root.reach[ev.US][i]]


def _all_cards_to(state: ts.GameState, loc: ts.CardLocation) -> None:
    for c in range(1, 111):
        if c != 6:
            state.set_card_location(c, loc)


def test_a_discarded_scoring_card_can_return_in_the_reshuffle_deal() -> None:
    """F4. Turn 9 with one card left in the draw pile: turn 10's deal takes that card and then
    draws from the reshuffled discards, Asia Scoring among them. The schedule started counting
    recycled draws only from the deal after, so Asia had final scoring alone."""
    state = _to_first_action_round(5)
    state.turn = 9
    _all_cards_to(state, ts.CardLocation.DISCARD_PILE)
    state.set_card_location(50, ts.CardLocation.DRAW_DECK)
    pol = DoctrinePolicy(seed=0)
    u = schedule.urgency(state, ts.Player.US, pol.t, pol.w)
    final_only = schedule.FINAL_SCORING_ODDS[8] * pol.w.scoring_final
    assert u[pol.t.region_anchor[ev.ASIA]] > final_only


def test_seen_opponent_cards_do_not_raise_the_odds_they_hold_an_unseen_one() -> None:
    """F5. Turn 10: eight opponent cards we have seen, one we have not, ten in the pile. An unseen
    Asia Scoring is in their hand with probability 1/11, not 9/19."""
    state = _to_first_action_round(5)
    state.turn = 10
    _all_cards_to(state, ts.CardLocation.DISCARD_PILE)
    for c in range(1, 11):                        # Asia Scoring (1) among the ten in the pile
        state.set_card_location(c, ts.CardLocation.DRAW_DECK)
    for c in range(11, 19):
        state.set_card_location(c, ts.CardLocation.HAND_USSR_KNOWN)
    state.set_card_location(19, ts.CardLocation.HAND_USSR_UNKNOWN)
    pol = DoctrinePolicy(seed=0)
    u = schedule.urgency(state, ts.Player.US, pol.t, pol.w)
    final = schedule.FINAL_SCORING_ODDS[9] * pol.w.scoring_final
    assert u[pol.t.region_anchor[ev.ASIA]] - final == pytest.approx(1 / 11, abs=1e-12)
