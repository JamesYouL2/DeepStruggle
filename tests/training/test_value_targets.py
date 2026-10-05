"""The value-targets probe: values come from the mover's side, playouts are reproducible, and the
report's error is the reference's error, noise removed."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ai.eval import value_targets as V
from ai.eval.target_forms import collect_positions
from ai.models.coldwar_net_v2 import create_coldwar_net_v2


def _model():
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


@pytest.fixture(scope="module")
def positions():
    return collect_positions(_model(), 6, seed=4, games=16)


def test_playouts_are_reproducible_and_in_range(positions) -> None:
    a = V.playout_results(_model(), positions, 3, seed=9, batch=64)
    b = V.playout_results(_model(), positions, 3, seed=9, batch=64)
    assert a.shape == (6, 3) and np.array_equal(a, b)
    assert np.all(np.abs(a) <= 1.0)


def test_search_values_are_from_the_movers_side(positions) -> None:
    v = V._search_values(_model(), positions, "search@1", seed=1)
    assert len(v) == 6 and all(-1.0 <= x <= 1.0 for x in v)


def test_a_value_equal_to_the_reference_has_no_error_beyond_noise() -> None:
    rows = [{"segment": "ar_card", "turn": t, "mover": 1, "z_mean": z, "z_var_of_mean": 0.0,
             "values": {"critic": z + 0.3, "perfect": z}} for t, z in zip(range(1, 11), np.linspace(-0.8, 0.8, 10))]
    md = V.report(rows, ["perfect"])
    line = next(l for l in md.splitlines() if l.startswith("| perfect |"))
    assert float(line.split("|")[2]) == pytest.approx(0.0, abs=1e-12)
    crit = next(l for l in md.splitlines() if l.startswith("| critic |"))
    assert float(crit.split("|")[2]) == pytest.approx(0.09, abs=1e-9)
