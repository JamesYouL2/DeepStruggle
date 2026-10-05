"""tools/policy_disagreement.py: names, and a report that reads zero for identical policies."""

from __future__ import annotations

import importlib.util
import os
from types import ModuleType

import numpy as np

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "tools", "policy_disagreement.py")


def _pd() -> ModuleType:
    spec = importlib.util.spec_from_file_location("policy_disagreement", _PATH)
    assert spec is not None and spec.loader is not None
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_labels() -> None:
    pd = _pd()
    assert pd.label("x/E7-20-44_20261004_173440/snapshot_2800025600steps.pt") == "E7-20-44@2800M"
    assert pd.label("data/checkpoints/3fa2c1d0/snapshot_2410020864steps.pt") == "snapshot@2410M"
    assert pd.label("data/checkpoints/E7-75-45@80M.pt") == "E7-75-45@80M"


def test_identical_policies_never_disagree_and_different_ones_do() -> None:
    pd = _pd()
    rng = np.random.default_rng(0)
    a = rng.dirichlet(np.ones(5), size=40)
    b = a[:, ::-1].copy()                                 # a different policy over the same moves
    md = pd.report(["a", "a2", "b"], [a, a.copy(), b], ["ar_card"] * 20 + ["event"] * 20)
    rows = [l for l in md.splitlines() if l.startswith("| **m1** |")]
    top_row, tv_row = rows[0], rows[1]
    assert top_row.split("|")[3].strip() == "0.0" and float(top_row.split("|")[4]) > 0
    assert tv_row.split("|")[3].strip() == "0.000" and float(tv_row.split("|")[4]) > 0
