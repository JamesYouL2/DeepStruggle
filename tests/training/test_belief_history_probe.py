"""P35a's posterior rebuilds each of the opponent's past decisions with a hypothesised hand. Given the
true hand it must rebuild the position exactly -- every card where it was and the opponent's
observation unchanged -- or the likelihood it computes is of some other position."""

from __future__ import annotations

import numpy as np
import torch

import ts_engine as ts
from ai.eval.belief_history_probe import CHOICES, UNKNOWN, _player, collect, left_since, posterior, world_at
from ai.models.ladder_net import create_ladder_net
from tests.training.test_ladder_tokens import SHALLOW


def _samples() -> list:
    torch.manual_seed(0)
    model = create_ladder_net("cpu", **SHALLOW).eval()
    return [model, collect(model, torch.device("cpu"), 6, 2, 1.0, 11, set(range(6)))]


def test_the_true_hand_rebuilds_every_past_decision_exactly() -> None:
    _, samples = _samples()
    checked = 0
    for s in samples:
        opp = 3 - s.decider
        for d in s.turn_log:
            if d.decider != opp or int(d.state.ctx().decision_type) not in CHOICES:
                continue
            left = left_since(d, s)
            if left is None:
                continue
            w = world_at(d, opp, left | (s.label > 0.5))
            assert [int(w.get_card_location(c)) for c in range(1, 111)] == [int(x) for x in d.locs]
            assert np.array_equal(ts.extract_observation(w, _player(opp)), ts.extract_observation(d.state, _player(opp)))
            assert (d.locs == UNKNOWN[opp]).sum() == left.sum() + s.label.sum()
            checked += 1
    assert checked > 20


def test_the_posterior_is_a_distribution_over_the_pool_with_the_hands_size() -> None:
    model, samples = _samples()
    rng = np.random.default_rng(0)
    some = [s for s in samples if s.turn_log][:5]
    assert some
    for s in some:
        post, _, ess = posterior(model, torch.device("cpu"), s, 32, 1.0, rng)
        assert 1.0 - 1e-6 <= ess <= 32 + 1e-6
        assert np.all(post[s.pool < 0.5] == 0)
        assert abs(float(post.sum()) - float(s.label.sum())) < 1e-4
