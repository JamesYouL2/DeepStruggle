"""A determinized world must not know the opponent's unrevealed headline.

The engine asks the US for its headline first and keeps the choice in `headline_us_card` while
the card stays in the US hand. Redealing the hand alone left every sampled world resolving the
true US headline, so a search at the USSR's headline knew it (found by the branch oracle: a
+34-point "blind spot" was +20 once the headline was hidden).
"""
import random

import numpy as np
import ts_engine as ts

from ai.search.dmcts import determinize
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle
from tools.lib.openings import play_scripted_setup


def _ussr_headline_after_us_chose(seed: int) -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    st = play_scripted_setup(st, "human")
    settle(st, SettleMode.CHANCE)
    assert st.current_phase == ts.Phase.HEADLINE and st.ctx().decision_player == ts.Player.US
    us_card = int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))[0])
    ts.Engine.step_flat(st, us_card, False)
    assert st.ctx().decision_player == ts.Player.USSR and st.headline_us_card != 0
    return st


def test_the_us_headline_is_forgotten_and_chosen_again() -> None:
    st = _ussr_headline_after_us_chose(31)
    world = determinize(st, ts.Player.USSR, random.Random(0))
    assert world.headline_us_card == 0
    assert world.ctx().decision_player == ts.Player.USSR
    ussr_card = int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(world)))[0])
    ts.Engine.step_flat(world, ussr_card, False)
    assert world.ctx().decision_player == ts.Player.US          # the US chooses again
    assert world.ctx().decision_type == ts.DecisionType.SELECT_CARD


def test_the_true_state_is_untouched() -> None:
    st = _ussr_headline_after_us_chose(32)
    before = st.headline_us_card
    determinize(st, ts.Player.USSR, random.Random(0))
    assert st.headline_us_card == before
