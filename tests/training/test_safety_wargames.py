"""Wargames is a forced win at the play-mode decision, not only at its own "end the game" branch.

classify_legal_actions recognised the win only at Wargames' CHOOSE_BRANCH node -- the second step.
At the decision that matters, play the card for its event or for Ops, the event read "normal",
so the decisive probe never counted a model declining it and the safety layer never took it.
"""
import numpy as np
import ts_engine as ts

from ai.eval.safety import EVENT_ACTION, WARGAMES, classify_legal_actions, find_instant_win
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance


def _wargames_play_mode(lead: int, defcon: int = 2) -> ts.GameState:
    """The US at its first action round with Wargames in hand, then at Wargames' play-mode decision."""
    st = ts.GameState()
    ts.Engine.init_game(st, 7)
    drain_chance(st, context="test_safety_wargames")
    for _ in range(400):
        if st.current_phase == ts.Phase.ACTION_ROUND and st.ctx().decision_type == ts.DecisionType.SELECT_CARD \
                and st.ctx().decision_player == ts.Player.US:
            break
        ts.Engine.step_flat(st, int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))[0]))
        drain_chance(st, context="test_safety_wargames")
    st.set_card_location(WARGAMES, ts.CardLocation.HAND_US_UNKNOWN)
    st.defcon = defcon
    st.victory_points = lead                      # + is the US
    ts.Engine.step_flat(st, WARGAMES - 1)          # select the card
    drain_chance(st, context="test_safety_wargames")
    assert st.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    return st


def test_the_event_is_a_win_at_a_lead_over_six() -> None:
    st = _wargames_play_mode(lead=7)
    assert classify_legal_actions(st)[EVENT_ACTION] == "win"
    assert find_instant_win(st) == EVENT_ACTION
    # And it is: the event, then branch 0, ends the game for the US.
    s = st.clone()
    ts.Engine.step_flat(s, EVENT_ACTION)
    drain_chance(s, context="test_safety_wargames")
    branch = int(np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))[0])
    ts.Engine.step_flat(s, branch)
    assert ts.Engine.is_terminal(s) and ts.Engine.get_terminal_utility(s) > 0


def test_not_a_win_at_six_or_less_or_above_defcon_two() -> None:
    assert classify_legal_actions(_wargames_play_mode(lead=6))[EVENT_ACTION] != "win"
    assert classify_legal_actions(_wargames_play_mode(lead=3))[EVENT_ACTION] != "win"
    assert classify_legal_actions(_wargames_play_mode(lead=9, defcon=3))[EVENT_ACTION] != "win"
