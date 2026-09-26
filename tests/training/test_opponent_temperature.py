"""--opponent-temperature: a frozen pool opponent can play at its own temperature (P24)."""

from __future__ import annotations

import torch

from ai.training.nash_pg import opponent_actions


def test_greedy_opponent_rows_take_the_argmax_and_learner_rows_are_untouched() -> None:
    logits = torch.tensor([[0.0, 3.0, -1e9], [2.0, 1.0, 0.0], [0.0, 0.0, 5.0]])
    sampled = torch.tensor([0, 2, 1])
    opp = torch.tensor([True, False, True])
    out = opponent_actions(logits, sampled, opp, 0.0)
    assert out.tolist() == [1, 2, 2]


def test_a_sampled_opponent_never_plays_a_masked_action() -> None:
    torch.manual_seed(0)
    logits = torch.tensor([[0.0, 0.0, -1e9]] * 200)
    out = opponent_actions(logits, torch.zeros(200, dtype=torch.long), torch.ones(200, dtype=torch.bool), 0.5)
    assert set(out.tolist()) <= {0, 1}
