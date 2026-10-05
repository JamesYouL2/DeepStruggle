"""The target-forms probe (ai/eval/target_forms.py): its targets are distributions over the legal
moves, the improved policy behaves as Gumbel MuZero's, and paired playouts are truly paired."""

from __future__ import annotations

import math
from typing import Dict

import numpy as np
import pytest
import torch

from ai.eval import target_forms as T
from ai.models.coldwar_net_v2 import create_coldwar_net_v2


def _model():
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


@pytest.fixture(scope="module")
def positions():
    return T.collect_positions(_model(), 12, seed=5, games=16)


def test_positions_cover_the_game_not_just_its_start(positions) -> None:
    assert len(positions) == 12
    assert len({p.segment for p in positions}) > 1
    for p in positions:
        assert len(p.legal) > 1
        assert math.isclose(sum(p.prior.values()), 1.0, rel_tol=1e-6)


def test_the_improved_policy() -> None:
    logits: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
    prior = {1: 1 / 3, 2: 1 / 3, 3: 1 / 3}
    # Nothing visited: every move has the mixed value, so the prior comes back.
    ip = T.improved_policy(logits, prior, 0.2, {}, {})
    assert all(math.isclose(ip[a], 1 / 3, rel_tol=1e-9) for a in ip)
    # A better searched value raises a move's probability; an unvisited move sits at the mix.
    ip = T.improved_policy(logits, prior, 0.0, {1: 10.0, 2: 10.0}, {1: 0.3, 2: -0.3})
    assert ip[1] > ip[3] > ip[2]
    # Without rescaling, a tiny value difference moves the target only a little.
    raw = T.improved_policy(logits, prior, 0.0, {1: 10.0, 2: 10.0}, {1: 0.001, 2: 0.0},
                            rescale=False)
    stretched = T.improved_policy(logits, prior, 0.0, {1: 10.0, 2: 10.0}, {1: 0.001, 2: 0.0})
    assert raw[1] - raw[2] < 0.01 < stretched[1] - stretched[2]


@pytest.mark.parametrize("form", ["visits@8", "visits@8,pt1.5", "cq@8", "cq@8,raw", "gumbel@8",
                                  "gchoice@8", "gchoice@8,k3,fpu0.2"])
def test_every_form_is_a_distribution_over_the_legal_moves(positions, form: str) -> None:
    T.TargetBuilder(_model(), seed=1, featurise_capacity=256, gumbel_k=4).build(positions[:6], [form])
    for p in positions[:6]:
        t = p.targets[form]
        assert set(t) == set(p.legal)
        assert math.isclose(sum(t.values()), 1.0, rel_tol=1e-6)
        assert min(t.values()) >= 0.0


def test_a_move_paired_with_itself_has_exactly_no_advantage(positions) -> None:
    """Same redeal and dice in both branches: a pair that plays the same move twice must agree, or
    the pairing is broken and every advantage is mostly noise."""
    items = [(p, p.legal[0], p.legal[0]) for p in positions[:3]]
    for mean, se in T.paired_advantage(_model(), items, pairs=3, seed=11, batch=64):
        assert mean == 0.0 and se == 0.0


def test_the_report_pools_rows(positions) -> None:
    forms = ["visits@8", "gumbel@8"]
    T.TargetBuilder(_model(), seed=2, featurise_capacity=256, gumbel_k=4).build(positions, forms)
    verdicts = {}
    for i, p in enumerate(positions):
        top0 = max(p.prior, key=lambda a: p.prior[a])
        for f in forms:
            t = p.targets[f]
            top = max(t, key=lambda a: (t[a], p.prior[a]))
            if top != top0:
                verdicts[(i, top)] = (0.1, 0.05)
    md = T.report(T.rows(positions, forms, verdicts), forms, T.sims_of(forms))
    assert "| visits@8 | 8 |" in md and "| gumbel@8 | 8 |" in md
    assert np.isfinite(len(md))


def test_a_one_candidate_gumbel_choice_is_the_priors_top_move(positions) -> None:
    T.TargetBuilder(_model(), seed=3, featurise_capacity=256).build(positions[:6], ["gchoice@8,k1"])
    for p in positions[:6]:
        t = p.targets["gchoice@8,k1"]
        assert max(t, key=lambda a: t[a]) == max(p.prior, key=lambda a: p.prior[a])
        assert sorted(t.values())[-1] == 1.0
