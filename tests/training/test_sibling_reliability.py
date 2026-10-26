"""tools/scripts/sibling_reliability.py: the split-half reliability of centred candidate means, on a
synthetic slice whose true reliability is known."""

from __future__ import annotations

import numpy as np

from tools.scripts.sibling_reliability import reliability, spearman_brown


def _slice(decisions: int, worlds: int, signal: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Each candidate's true mean plus independent world noise of unit variance; worlds shared."""
    rng = np.random.default_rng(seed)
    true = rng.normal(0.0, signal, (decisions, 4, 1))
    y = true + rng.normal(0.0, 1.0, (decisions, 4, worlds))
    return y, np.ones((decisions, worlds), bool)


def test_reliability_matches_the_signal_to_noise_at_each_world_count() -> None:
    signal = 0.25   # variance 1/16 against unit noise: 16 worlds give reliability 1/2
    y, ok = _slice(6000, 64, signal, 1)
    for w in (16, 64):
        want = w * signal ** 2 / (w * signal ** 2 + 1)
        got = reliability(y, ok, w)
        assert abs(got["reliability"] - want) < 0.03, (w, got, want)
        assert got["decisions"] == 6000


def test_spearman_brown_steps_d2s_16_worlds_up_to_the_plans_numbers() -> None:
    assert abs(spearman_brown(0.41, 4) - 0.735) < 0.005
    assert abs(spearman_brown(0.41, 8) - 0.847) < 0.005


def test_failed_worlds_are_left_out_of_both_halves() -> None:
    y, ok = _slice(500, 32, 0.5, 2)
    ok[:, 5] = False
    y[:, :, 5] = 1e6   # a failed world's result must not reach the means
    got = reliability(y, ok, 32)
    assert 0.5 < got["reliability"] < 1.0
