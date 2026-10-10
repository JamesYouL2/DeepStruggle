"""The ops census's standard error is clustered by game: records of one game move together."""

from __future__ import annotations

import numpy as np

from tools.scripts.ops_census import clustered


def test_independent_records_match_the_plain_standard_error() -> None:
    x = np.array([1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0])
    m, se = clustered(x, list(range(len(x))))
    plain = float(np.sqrt(len(x) / (len(x) - 1)) * x.std() / np.sqrt(len(x)))
    assert m == 0.5 and abs(se - plain) < 1e-12


def test_records_that_move_together_count_as_one() -> None:
    """Four copies of each of two games' records: the same mean, and the standard error of two
    observations, not of eight."""
    x = np.array([1.0] * 4 + [0.0] * 4)
    m, se = clustered(x, ["a"] * 4 + ["b"] * 4)
    assert m == 0.5 and abs(se - 0.5) < 1e-12
    assert clustered(x, list(range(8)))[1] < se
