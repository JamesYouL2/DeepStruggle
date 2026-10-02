"""The forced-win and forced-loss cases of `ai/eval/safety.classify_legal_actions`, each on a built
position: Wargames at the play-mode decision, a battleground coup at DEFCON 2 (and with Nuclear
Subs), reaching 20 VP, and Europe Scoring with Europe controlled.

Wargames was recognised only at its CHOOSE_BRANCH node -- the second step. At the decision that
matters, play the card for its event or for Ops, the event read "normal", so the decisive probe
never counted a model declining it and the safety layer never took it. The other cases were already
right; they are pinned here so the classifier cannot regress on any of the game's instant endings.
"""
import numpy as np
import ts_engine as ts

from ai.eval.safety import (EVENT_ACTION, NODE_OFFSET, PLAY_MODE_ACTION, WARGAMES, classify_legal_actions,
                            find_instant_win)
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

COUP_ACTION = PLAY_MODE_ACTION["ops_coup"]
NUCLEAR_SUBS_ACTIVE = int(ts.EffectBits.NUCLEAR_SUBS_ACTIVE)
CA_SCORING, EUROPE_SCORING = 37, 2


def _legal(st: ts.GameState) -> np.ndarray:
    return np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)


def _step(st: ts.GameState, a: int) -> None:
    ts.Engine.step_flat(st, int(a))
    drain_chance(st, context="test_safety_decisive")


def _cid(name: str) -> int:
    return int(ts.MapData.get_country_by_name(name))


def _us_card_choice(seed: int = 7) -> ts.GameState:
    """The US choosing a card at its first action round."""
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    drain_chance(st, context="test_safety_decisive")
    for _ in range(400):
        if st.current_phase == ts.Phase.ACTION_ROUND and st.ctx().decision_type == ts.DecisionType.SELECT_CARD \
                and st.ctx().decision_player == ts.Player.US:
            return st
        _step(st, int(np.flatnonzero(_legal(st))[0]))
    raise AssertionError("no US action round reached")


def _play_card(st: ts.GameState, card: int) -> ts.GameState:
    st.set_card_location(card, ts.CardLocation.HAND_US_UNKNOWN)
    _step(st, card - 1)
    return st


# --- Wargames ------------------------------------------------------------------------------------

def _wargames_play_mode(lead: int, defcon: int = 2) -> ts.GameState:
    st = _us_card_choice()
    st.defcon = defcon
    st.victory_points = lead                      # + is the US
    _play_card(st, WARGAMES)
    assert st.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE
    return st


def test_wargames_event_is_a_win_at_a_lead_over_six() -> None:
    st = _wargames_play_mode(lead=7)
    assert classify_legal_actions(st)[EVENT_ACTION] == "win"
    assert find_instant_win(st) == EVENT_ACTION
    s = st.clone()                                 # and it is: the event, then branch 0, ends the game
    _step(s, EVENT_ACTION)
    _step(s, int(np.flatnonzero(_legal(s))[0]))
    assert ts.Engine.is_terminal(s) and ts.Engine.get_terminal_utility(s) > 0


def test_wargames_is_not_a_win_at_six_or_less_or_above_defcon_two() -> None:
    assert classify_legal_actions(_wargames_play_mode(lead=6))[EVENT_ACTION] != "win"
    assert classify_legal_actions(_wargames_play_mode(lead=3))[EVENT_ACTION] != "win"
    assert classify_legal_actions(_wargames_play_mode(lead=9, defcon=3))[EVENT_ACTION] != "win"


# --- a battleground coup at DEFCON 2 ---------------------------------------------------------------

def _coup_targets(nuclear_subs: bool) -> ts.GameState:
    """The US choosing a coup target at DEFCON 2, a battleground (Angola) and a non-battleground
    (Zimbabwe) both holding USSR influence."""
    st = _us_card_choice()
    angola, zimbabwe = _cid("Angola"), _cid("Zimbabwe")
    st.set_country(angola, 0, 2)
    st.set_country(zimbabwe, 0, 1)
    st.defcon = 2
    if nuclear_subs:
        st.set_flag(NUCLEAR_SUBS_ACTIVE)
    card = next(c for c in range(7, 111) if ts.CardData.get_card_info(c)["side"] == "US"
                and int(ts.CardData.get_card_info(c)["ops"]) >= 2 and c not in (37, 38, 79, 81))
    _play_card(st, card)
    assert _legal(st)[COUP_ACTION]
    _step(st, COUP_ACTION)
    assert st.ctx().decision_type == ts.DecisionType.POINT_NODE and st.ctx().op_mode == ts.OpMode.COUP
    assert _legal(st)[NODE_OFFSET + angola] and _legal(st)[NODE_OFFSET + zimbabwe]
    return st


def test_a_battleground_coup_at_defcon_two_is_a_loss() -> None:
    st = _coup_targets(nuclear_subs=False)
    out = classify_legal_actions(st)
    assert out[NODE_OFFSET + _cid("Angola")] == "loss"
    assert out[NODE_OFFSET + _cid("Zimbabwe")] != "loss"


def test_not_a_loss_under_nuclear_subs() -> None:
    st = _coup_targets(nuclear_subs=True)
    assert classify_legal_actions(st)[NODE_OFFSET + _cid("Angola")] != "loss"
    s = st.clone()                                 # the engine agrees: the coup leaves DEFCON at 2
    _step(s, NODE_OFFSET + _cid("Angola"))
    assert not ts.Engine.is_terminal(s) and int(s.defcon) == 2


# --- 20 VP and Europe ------------------------------------------------------------------------------

def _control(st: ts.GameState, names: list, us: int = 5) -> None:
    for n in names:
        st.set_country(_cid(n), us, 0)


def test_a_scoring_card_that_reaches_twenty_is_a_win() -> None:
    st = _us_card_choice()
    _control(st, ["Mexico", "Cuba", "Panama"])         # Central America: US domination or control
    st.set_card_location(CA_SCORING, ts.CardLocation.HAND_US_UNKNOWN)
    st.victory_points = 18
    assert classify_legal_actions(st)[CA_SCORING - 1] == "win"
    st.victory_points = 10
    assert classify_legal_actions(st)[CA_SCORING - 1] != "win"


def test_europe_scoring_with_europe_controlled_is_a_win() -> None:
    st = _us_card_choice()
    europe = ["West Germany", "France", "Italy", "East Germany", "Poland", "United Kingdom", "Spain/Portugal",
              "Benelux", "Norway", "Denmark", "Greece", "Turkey"]
    _control(st, europe)
    for n in ("Austria", "Finland", "Yugoslavia", "Czechoslovakia", "Hungary", "Romania", "Bulgaria", "Sweden"):
        st.set_country(_cid(n), 0, 0)
    st.set_card_location(EUROPE_SCORING, ts.CardLocation.HAND_US_UNKNOWN)
    st.victory_points = 0
    assert classify_legal_actions(st)[EUROPE_SCORING - 1] == "win"


def _junta_free_coup() -> ts.GameState:
    """The US at Junta's free coup, at DEFCON 2: a South American battleground (Chile) and a
    non-battleground (Bolivia) both holding USSR influence."""
    st = _us_card_choice()
    st.set_country(_cid("Chile"), 0, 2)
    st.set_country(_cid("Bolivia"), 0, 1)
    st.defcon = 2
    _play_card(st, 47)                              # Junta
    _step(st, EVENT_ACTION)
    for _ in range(12):                             # its placement, then the coup-or-realign choice
        c = st.ctx()
        if c.decision_type == ts.DecisionType.POINT_NODE and c.op_mode == ts.OpMode.COUP:
            return st
        legal = [int(a) for a in np.flatnonzero(_legal(st))]
        nxt = None
        for a in legal:                             # prefer whatever leads to the coup
            s = st.clone()
            _step(s, a)
            if s.ctx().decision_type == ts.DecisionType.POINT_NODE and s.ctx().op_mode == ts.OpMode.COUP:
                nxt = a
                break
        _step(st, nxt if nxt is not None else legal[0])
    raise AssertionError("Junta's free coup was not reached")


def test_a_free_battleground_coup_at_defcon_two_is_a_loss() -> None:
    st = _junta_free_coup()
    assert int(st.ctx().pending_op_card) == 47      # Junta's own free coup (the engine grants its 2 Ops)
    out = classify_legal_actions(st)
    assert out[NODE_OFFSET + _cid("Chile")] == "loss"
    assert out[NODE_OFFSET + _cid("Bolivia")] != "loss"
