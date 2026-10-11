"""A step that draws a card from the engine's PRNG is enumerated exactly (`tools/lib/hidden_draws.py`),
and the blunder bank's exact search averages over those draws instead of taking the one the
world's generator happens to make.

Five Year Plan, Grain Sales, Terrorism and Our Man in Tehran draw inside the step, with no
decision node to branch on. The first proofs of the bank stepped such a move once, so a row whose
rest of the game held one of these cards was "proved" on a single sample of the draw.
"""

from __future__ import annotations

import random
from typing import List, Sequence

import pytest
import ts_engine as ts

from tools.lib.hidden_draws import (GAMMA, MASK, ManyDraws, calls_between, mix, next_u64,
                                    state_for_output, step_outcomes, unmix)
from tools.scripts.blunder_bank import Unprovable, outcomes

FIVE_YEAR_PLAN = 5        # US, 3 Ops: the USSR discards a random card
TERRORISM = 92            # USSR: the opponent discards one card, two under the hostage crisis
NUCLEAR_TEST_BAN = 34
DUCK_AND_COVER = 4
OLYMPIC_GAMES = 20
PLAY_EVENT = 110          # the merged resolution node: event first on an opponent's card
PLAY_OPS_INFLUENCE = 112


def test_the_generator_is_inverted_exactly() -> None:
    rng = random.Random(7)
    for _ in range(2000):
        x = rng.getrandbits(64)
        assert unmix(mix(x)) == x
    for u in (0, 1, 5, 6, 109, 123456789, MASK):
        assert next_u64(state_for_output(u)) == u
    s = 0x123456789ABCDEF0
    assert calls_between(s, s) == 0
    assert calls_between(s, (s + GAMMA) & MASK) == 1
    assert calls_between(s, (s + 5 * GAMMA) & MASK) == 5


def _action_round(mover: ts.Player) -> ts.GameState:
    state = ts.GameState()
    ts.Engine.init_game(state, 12345)
    guard = 0
    while state.current_phase != ts.Phase.ACTION_ROUND and guard < 600:
        guard += 1
        ctx = state.ctx()
        if ctx.decision_player == ts.Player.NONE and ctx.decision_type == ts.DecisionType.ROLL_DIE:
            ts.Engine.step(state, ts.MicroAction(ts.DecisionType.ROLL_DIE, 0, 0, 0))
            continue
        legal = [i for i, v in enumerate(ts.ActionMask.generate_flat_mask(state)) if v]
        ts.Engine.step_flat(state, legal[0])
    state.turn, state.action_round = 9, 6
    state.phasing_player = mover
    ctx = state.ctx()
    ctx.decision_player, ctx.decision_type = mover, ts.DecisionType.SELECT_CARD
    for c in range(1, 111):
        if state.get_card_location(c) in (ts.hand_of(ts.Player.US), ts.hand_of(ts.Player.USSR)):
            state.set_card_location(c, ts.CardLocation.DISCARD_PILE)
    state.china_card_holder = ts.Player.US if mover == ts.Player.USSR else ts.Player.USSR
    state.china_card_playable = 0
    return state


def _selected(mover: ts.Player, card: int, hand: Sequence[int], opp_hand: Sequence[int] = ()) -> ts.GameState:
    """`card` selected by `mover`, holding `hand` beside it; the play mode is the next decision."""
    state = _action_round(mover)
    loc = ts.hand_of(mover)
    state.set_card_location(card, loc)
    for c in hand:
        state.set_card_location(c, loc)
    for c in opp_hand:
        state.set_card_location(c, ts.hand_of(ts.Player.US if mover == ts.Player.USSR else ts.Player.USSR))
    ts.Engine.step_flat(state, card - 1)
    assert state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    return state


def _discarded(st: ts.GameState, cards: List[int]) -> List[int]:
    return [c for c in cards if st.get_card_location(c) == ts.CardLocation.DISCARD_PILE]


def test_a_single_draw_is_enumerated_one_outcome_per_card() -> None:
    """The USSR plays Five Year Plan for Ops holding three other cards: three outcomes, each
    discarding a different one, all equally likely, and the generator put back."""
    hand = [NUCLEAR_TEST_BAN, DUCK_AND_COVER, OLYMPIC_GAMES]
    st = _selected(ts.Player.USSR, FIVE_YEAR_PLAN, hand)
    rng0 = int(st.rng_state)
    outs = step_outcomes(st, lambda c: ts.Engine.step_flat(c, PLAY_EVENT))
    assert len(outs) == 3
    assert sorted(_discarded(o, hand)[0] for o in outs) == sorted(hand)
    assert all(int(o.rng_state) == rng0 for o in outs)
    # the world's own draw is one of them
    own = st.clone()
    ts.Engine.step_flat(own, PLAY_EVENT)
    assert _discarded(own, hand)[0] in {_discarded(o, hand)[0] for o in outs}


def test_a_step_without_a_draw_is_the_plain_step() -> None:
    st = _selected(ts.Player.USSR, FIVE_YEAR_PLAN, [NUCLEAR_TEST_BAN])
    outs = step_outcomes(st, lambda c: ts.Engine.step_flat(c, PLAY_OPS_INFLUENCE))
    plain = st.clone()
    ts.Engine.step_flat(plain, PLAY_OPS_INFLUENCE)
    assert len(outs) == 1 and outs[0].to_save_json() == plain.to_save_json()


def test_two_draws_are_reported_not_sampled() -> None:
    """Terrorism under the hostage crisis discards two US cards: two draws, which the enumeration
    does not untangle, so the search says so rather than take the world's draw."""
    us_hand = [NUCLEAR_TEST_BAN, DUCK_AND_COVER, OLYMPIC_GAMES]
    st = _selected(ts.Player.USSR, TERRORISM, [], us_hand)
    st.set_flag(ts.EffectBits.IRANIAN_HOSTAGE_CRISIS_PLAY)
    # the USSR's own card: the event is the first play mode
    with pytest.raises(ManyDraws) as e:
        step_outcomes(st, lambda c: ts.Engine.step_flat(c, PLAY_EVENT))
    assert e.value.draws == 2
    with pytest.raises(Unprovable, match="2 hidden draws"):
        outcomes(st, lambda c: ts.Engine.step_flat(c, PLAY_EVENT))
    # without the crisis it is one draw: three outcomes
    st.clear_flag(ts.EffectBits.IRANIAN_HOSTAGE_CRISIS_PLAY)
    outs = outcomes(st, lambda c: ts.Engine.step_flat(c, PLAY_EVENT))
    assert sorted(_discarded(o, us_hand)[0] for o in outs) == sorted(us_hand)
