"""`headline:<card>:<spec>` -- a scripted headline rule over any agent, for the tournament."""

from __future__ import annotations

import numpy as np

import ts_engine as ts
from tools.lib.batch_tournament import HEADLINE_CONDITIONS, apply_headline_rule
from tools.lib.player_agent import load_agent


def test_the_spec_wraps_an_agent_and_names_the_rule() -> None:
    agent = load_agent("headline:40:heuristic", device="cpu")
    assert getattr(agent, "headline_card") == 40 and agent.name.endswith("+headline40")


def test_the_rule_takes_the_card_at_a_headline_and_nowhere_else() -> None:
    runner = ts.VectorizedBatchRunner(8, 31)
    masks = np.asarray(runner.get_action_masks())
    while not (runner.get_state(0).current_phase == ts.Phase.HEADLINE
               and runner.get_state(0).ctx().decision_type == ts.DecisionType.SELECT_CARD):
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
        masks = np.asarray(runner.get_action_masks())
    st = runner.get_state(0)
    card = next(c for c in range(1, 111) if masks[0][c - 1])        # one the decider holds
    actions = np.full(8, -1, dtype=np.int32)
    apply_headline_rule(card, np.array([0]), masks, runner, actions)
    assert actions[0] == card - 1
    held_elsewhere = next(c for c in range(1, 111) if not masks[0][c - 1])
    actions[0] = -1
    apply_headline_rule(held_elsewhere, np.array([0]), masks, runner, actions)
    assert actions[0] == -1 and st.current_phase == ts.Phase.HEADLINE


def test_a_rule_can_carry_a_condition() -> None:
    agent = load_agent("headline:40+oppcantpay:heuristic", device="cpu")
    assert getattr(agent, "headline_condition") == "oppcantpay"
    assert agent.name.endswith("+headline40-oppcantpay") and "oppcantpay" in HEADLINE_CONDITIONS


def test_cannot_pay_reads_the_opponents_crisis_countries() -> None:
    """On an empty board neither side could pay off a crisis headlined against it."""
    st = ts.GameState()
    assert HEADLINE_CONDITIONS["oppcantpay"](st)
