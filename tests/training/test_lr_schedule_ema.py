"""P28 step 2: --lr-schedule and --ema-weights (ai/training/schedule.py)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from ai.training.schedule import WeightEMA, scheduled_lr


def test_constant_is_the_base_rate_everywhere() -> None:
    for s in (0, 10**6, 10**9):
        assert scheduled_lr(s, 3e-4) == 3e-4


def test_step_schedule_steps_down_every_interval_from_its_start() -> None:
    kw = dict(kind="step", start=560_000_000, every=60_000_000, values=(1e-4, 3e-5))
    assert scheduled_lr(100_000_000, 3e-4, **kw) == 3e-4          # before the start
    assert scheduled_lr(560_000_000, 3e-4, **kw) == 3e-4
    assert scheduled_lr(619_999_999, 3e-4, **kw) == 3e-4
    assert scheduled_lr(620_000_000, 3e-4, **kw) == 1e-4
    assert scheduled_lr(680_000_000, 3e-4, **kw) == 3e-5
    assert scheduled_lr(900_000_000, 3e-4, **kw) == 3e-5          # the last value holds


def test_cosine_runs_from_the_base_rate_to_the_floor_over_its_span() -> None:
    kw = dict(kind="cosine", start=0, span=200, lr_min=3e-5)
    assert math.isclose(scheduled_lr(0, 3e-4, **kw), 3e-4)
    assert math.isclose(scheduled_lr(100, 3e-4, **kw), (3e-4 + 3e-5) / 2)
    assert math.isclose(scheduled_lr(200, 3e-4, **kw), 3e-5)
    assert math.isclose(scheduled_lr(10_000, 3e-4, **kw), 3e-5)


def test_an_unknown_schedule_is_refused() -> None:
    with pytest.raises(ValueError):
        scheduled_lr(0, 3e-4, kind="linear")


def test_the_average_moves_by_one_minus_exp_of_steps_over_tau() -> None:
    live = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        live.weight.fill_(0.0)
    ema = WeightEMA(live, tau_steps=100.0)
    with torch.no_grad():
        live.weight.fill_(1.0)
    alpha = ema.update(live, 100)
    assert math.isclose(alpha, 1 - math.exp(-1))
    assert torch.allclose(ema.state_dict()["weight"], torch.full((1, 2), alpha))
    # the live weights are untouched, and the average is not trainable
    assert torch.allclose(live.weight, torch.ones(1, 2))
    assert not any(p.requires_grad for p in ema.model.parameters())


def test_the_average_restores_from_a_saved_state() -> None:
    live = nn.Linear(2, 1, bias=False)
    saved = {"weight": torch.full((1, 2), 0.25)}
    ema = WeightEMA(live, tau_steps=10.0, state=saved)
    assert torch.allclose(ema.state_dict()["weight"], torch.full((1, 2), 0.25))
