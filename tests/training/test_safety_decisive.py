"""The forced-win and forced-loss cases of `ai/eval/safety.classify_legal_actions`, each on a built
position, and how the decisive probe counts them.

A win is a line of the mover's own choices: Wargames from hand (card, event, branch), Star Wars
eventing a winning card from the discard -- the mover's own Star Wars, or the opponent's played for
Ops -- and the card Grain Sales drew, whoever played Grain Sales. No opponent decision, die or hidden
draw may sit on the line, so choosing Grain Sales itself, or a war card, is never a win.

The losses pinned here -- a battleground coup at DEFCON 2 (and with Nuclear Subs), and the DEFCON
cards -- were already right; they are pinned so the classifier cannot regress on them.
"""
from typing import List, Tuple

import numpy as np
import pytest
import ts_engine as ts

from ai.eval.decisive_probe import DecisiveStats
from ai.eval.positions import PLAY_MODE_ACTION
from ai.eval.safety import (EVENT_ACTION, NODE_OFFSET, WARGAMES, WinDecision, classify_legal_actions,
                            find_instant_win, fold_win_opportunities)
from bindings.action_encoder import ActionEncoder
from tools.lib.game_step import drain_chance

COUP_ACTION = PLAY_MODE_ACTION["ops_coup"]
OPS_ACTION = PLAY_MODE_ACTION["ops"]
BRANCH_0 = ActionEncoder.BRANCH_OFFSET
NUCLEAR_SUBS_ACTIVE = int(ts.EffectBits.NUCLEAR_SUBS_ACTIVE)
CA_SCORING, EUROPE_SCORING = 37, 2
DUCK_AND_COVER, NUCLEAR_TEST_BAN, BRUSH_WAR, GRAIN_SALES, STAR_WARS = 4, 34, 36, 67, 85
US, USSR = ts.Player.US, ts.Player.USSR
HANDS = {US: (ts.CardLocation.HAND_US_UNKNOWN, ts.CardLocation.HAND_US_KNOWN),
         USSR: (ts.CardLocation.HAND_USSR_UNKNOWN, ts.CardLocation.HAND_USSR_KNOWN)}


def _legal(st: ts.GameState) -> np.ndarray:
    return np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)


def _step(st: ts.GameState, a: int) -> None:
    ts.Engine.step_flat(st, int(a))
    drain_chance(st, context="test_safety_decisive")


def _cid(name: str) -> int:
    return int(ts.MapData.get_country_by_name(name))


def _card_choice(side: ts.Player = US, seed: int = 7) -> ts.GameState:
    """`side` choosing a card at its first action round."""
    st = ts.GameState()
    ts.Engine.init_game(st, seed)
    drain_chance(st, context="test_safety_decisive")
    for _ in range(400):
        if st.current_phase == ts.Phase.ACTION_ROUND and st.ctx().decision_type == ts.DecisionType.SELECT_CARD \
                and st.ctx().decision_player == side:
            return st
        _step(st, int(np.flatnonzero(_legal(st))[0]))
    raise AssertionError(f"no {side} action round reached")


def _us_card_choice(seed: int = 7) -> ts.GameState:
    return _card_choice(US, seed)


def _hand(st: ts.GameState, side: ts.Player, cards: List[int]) -> None:
    """`side` holds exactly `cards`; what it held goes back to the deck."""
    for c in range(1, 111):
        if st.get_card_location(c) in HANDS[side]:
            st.set_card_location(c, ts.CardLocation.DRAW_DECK)
    for c in cards:
        st.set_card_location(c, HANDS[side][0])


def _play_card(st: ts.GameState, card: int) -> ts.GameState:
    st.set_card_location(card, ts.CardLocation.HAND_US_UNKNOWN)
    _step(st, card - 1)
    return st


def _lead(side: ts.Player, lead: int) -> int:
    """`victory_points` for `side` ahead by `lead` (the engine's VP are US-positive)."""
    return lead if side == US else -lead


def _wins_by_taking_them(st: ts.GameState) -> None:
    """From a position with a win, `find_instant_win` at every step reaches the mover's win --
    what the safety layer does with the label."""
    mover = st.ctx().decision_player
    s = st.clone()
    for _ in range(12):
        if ts.Engine.is_terminal(s):
            util = float(ts.Engine.get_terminal_utility(s))
            assert (util if mover == US else -util) > 0
            return
        assert s.ctx().decision_player == mover
        a = find_instant_win(s, mover)
        assert a is not None, f"the line broke off at {s.ctx().decision_type}"
        ts.Engine.step_flat(s, a)
    raise AssertionError("the line did not end the game")


# --- Wargames ------------------------------------------------------------------------------------

def _wargames_in_hand(side: ts.Player, lead: int, defcon: int = 2) -> ts.GameState:
    st = _card_choice(side)
    st.defcon = defcon
    st.victory_points = _lead(side, lead)
    st.set_card_location(WARGAMES, HANDS[side][0])
    return st


@pytest.mark.parametrize("side", [US, USSR])
def test_wargames_is_a_win_at_every_step_at_a_lead_over_six(side: ts.Player) -> None:
    st = _wargames_in_hand(side, lead=7)
    assert classify_legal_actions(st)[WARGAMES - 1] == "win"         # the card, from hand
    _wins_by_taking_them(st)
    _step(st, WARGAMES - 1)
    out = classify_legal_actions(st)                                  # its play mode
    assert out[EVENT_ACTION] == "win" and out[OPS_ACTION] == "normal"
    _step(st, EVENT_ACTION)
    assert st.ctx().decision_type == ts.DecisionType.CHOOSE_BRANCH
    out = classify_legal_actions(st)                                  # its branch, a flat index
    assert out[BRANCH_0] == "win" and out[BRANCH_0 + 1] == "normal"


@pytest.mark.parametrize("side", [US, USSR])
@pytest.mark.parametrize("lead,defcon", [(6, 2), (3, 2), (9, 3)])
def test_wargames_is_not_a_win_at_six_or_less_or_above_defcon_two(side: ts.Player, lead: int,
                                                                  defcon: int) -> None:
    st = _wargames_in_hand(side, lead=lead, defcon=defcon)
    assert classify_legal_actions(st)[WARGAMES - 1] != "win"
    _step(st, WARGAMES - 1)
    assert classify_legal_actions(st)[EVENT_ACTION] == "normal"       # the branch can still decline
    if defcon == 2:
        _step(st, EVENT_ACTION)
        branch = classify_legal_actions(st)[BRANCH_0]
        assert branch == ("normal" if lead == 6 else "loss")          # a tie at 6, a loss below


# --- Grain Sales: the card it drew -----------------------------------------------------------------

def _grain_sales_draw(player: ts.Player, drawn: int, lead: int) -> ts.GameState:
    """`player` plays Grain Sales -- the US for its event, the USSR for Ops, event first -- at
    DEFCON 2 with the USSR holding only `drawn` besides; the US then plays the card it drew."""
    st = _card_choice(player)
    st.defcon = 2
    st.victory_points = lead                          # + is the US
    _hand(st, USSR, [drawn] + ([GRAIN_SALES] if player == USSR else []))
    if player == US:
        st.set_card_location(GRAIN_SALES, HANDS[US][0])
    _step(st, GRAIN_SALES - 1)
    _step(st, EVENT_ACTION)
    c = st.ctx()
    assert (c.decision_type, c.decision_player, int(c.resolving_card), int(c.pending_op_card)) == \
        (ts.DecisionType.SELECT_PLAY_MODE, US, GRAIN_SALES, drawn)
    return st


@pytest.mark.parametrize("player", [US, USSR])
def test_the_wargames_grain_sales_drew_is_a_win(player: ts.Player) -> None:
    st = _grain_sales_draw(player, WARGAMES, lead=7)
    assert classify_legal_actions(st)[EVENT_ACTION] == "win"
    _wins_by_taking_them(st)


def test_choosing_grain_sales_is_a_win_only_when_its_draw_is_certain() -> None:
    """The draw is chance -- unless the USSR holds one card, when the engine draws it without the RNG."""
    st = _us_card_choice()
    st.defcon = 2
    st.victory_points = 7
    st.set_card_location(GRAIN_SALES, HANDS[US][0])
    _hand(st, USSR, [WARGAMES])
    assert classify_legal_actions(st)[GRAIN_SALES - 1] == "win"
    _hand(st, USSR, [WARGAMES, BRUSH_WAR])
    assert classify_legal_actions(st)[GRAIN_SALES - 1] != "win"


def test_the_card_grain_sales_drew_is_judged_as_itself() -> None:
    """Duck and Cover drawn at DEFCON 2: DEFCON 1 defeats the phasing player, so the event loses in
    the US's own round and wins in the USSR's. The card was read as Grain Sales and neither was seen."""
    own = _grain_sales_draw(US, DUCK_AND_COVER, lead=0)
    assert classify_legal_actions(own)[EVENT_ACTION] == "loss"
    theirs = _grain_sales_draw(USSR, DUCK_AND_COVER, lead=0)
    assert classify_legal_actions(theirs)[EVENT_ACTION] == "win"
    _step(theirs, EVENT_ACTION)                       # and the engine agrees
    assert ts.Engine.is_terminal(theirs) and ts.Engine.get_terminal_utility(theirs) > 0


# --- Star Wars: a winning card from the discard ----------------------------------------------------

def _star_wars_position(player: ts.Player, lead: int = 7) -> ts.GameState:
    st = _card_choice(player)
    st.defcon = 2
    st.victory_points = lead
    st.us_space_track, st.ussr_space_track = 3, 0     # Star Wars needs the US ahead in space
    st.set_card_location(WARGAMES, ts.CardLocation.DISCARD_PILE)
    st.set_card_location(BRUSH_WAR, ts.CardLocation.DISCARD_PILE)
    st.set_card_location(STAR_WARS, HANDS[player][0])
    return st


def test_star_wars_from_hand_is_a_win_with_wargames_in_the_discard() -> None:
    st = _star_wars_position(US)
    assert classify_legal_actions(st)[STAR_WARS - 1] == "win"
    _wins_by_taking_them(st)


def test_the_pick_is_a_win_when_the_opponent_plays_star_wars() -> None:
    st = _star_wars_position(USSR)
    _step(st, STAR_WARS - 1)
    _step(st, EVENT_ACTION)                           # the USSR's Ops, event first
    c = st.ctx()
    assert (c.decision_type, c.decision_player, int(c.resolving_card)) == \
        (ts.DecisionType.SELECT_CARD, US, STAR_WARS)
    out = classify_legal_actions(st)
    assert out[WARGAMES - 1] == "win" and out[BRUSH_WAR - 1] == "normal"
    _wins_by_taking_them(st)


def test_star_wars_is_not_a_win_without_a_winning_card_to_pick() -> None:
    st = _star_wars_position(US, lead=6)
    assert classify_legal_actions(st)[STAR_WARS - 1] != "win"


# --- an event's VP, and a die ----------------------------------------------------------------------

def test_an_event_from_hand_that_reaches_twenty_is_a_win() -> None:
    st = _us_card_choice()
    st.defcon = 4                                     # Nuclear Test Ban: DEFCON - 2 = 2 VP
    st.victory_points = 18
    st.set_card_location(NUCLEAR_TEST_BAN, HANDS[US][0])
    assert classify_legal_actions(st)[NUCLEAR_TEST_BAN - 1] == "win"
    _wins_by_taking_them(st)
    st.victory_points = 17
    assert classify_legal_actions(st)[NUCLEAR_TEST_BAN - 1] != "win"


def test_a_win_that_needs_a_die_is_not_one() -> None:
    st = _us_card_choice()
    st.victory_points = 19                            # Brush War's 1 VP would do it -- on a roll
    st.set_card_location(BRUSH_WAR, HANDS[US][0])
    assert classify_legal_actions(st)[BRUSH_WAR - 1] != "win"


# --- counting: one chance to win, however many decisions it takes ----------------------------------

def _counted(decisions: List[WinDecision]) -> Tuple[int, int]:
    s = DecisiveStats()
    s.add_wins(decisions)
    return s.win_available, s.win_taken


def test_a_win_is_counted_once_across_its_decisions() -> None:
    took: List[WinDecision] = [(US, 7, 1, True), (US, 4, 1, True), (US, 2, 1, True)]   # card, event, branch
    assert _counted(took) == (1, 1)
    ops: List[WinDecision] = [(US, 7, 1, True), (US, 4, 1, False)]                     # card, then Ops
    assert _counted(ops) == (1, 0)
    declined: List[WinDecision] = [(US, 7, 1, True), (US, 4, 1, True), (US, 2, 1, False)]
    assert _counted(declined) == (1, 0)


def test_separate_chances_count_separately() -> None:
    seq: List[WinDecision] = [(US, 7, 1, False), (USSR, 5, 0, False), (US, 6, 1, True), (US, 2, 1, True)]
    assert fold_win_opportunities(seq) == [(0, False), (2, True)]


def test_a_win_with_no_way_to_decline_it_is_not_a_chance() -> None:
    assert _counted([(US, 3, 3, True)]) == (0, 0)
    # ...but it does not break a line that is already open
    assert _counted([(US, 7, 1, True), (US, 1, 1, True), (US, 2, 1, True)]) == (1, 1)


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
