"""ai/training/turn_credit.py (P32 B4'): a rollout queues the learner's small decisions, its end
prices them into evidence-weighted improved targets, and the update trains on them."""

import numpy as np
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.turn_credit import turn_credit_loss
from bindings.ts_env import TsVectorizedEnv

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)


def test_the_loss_is_weighted_cross_entropy_on_the_legal_set() -> None:
    logits = torch.tensor([[0.0, 0.0, -1e9], [2.0, 0.0, 0.0]])
    tgt = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]])
    w = torch.tensor([1.0, 0.0])
    # the second row has no weight; the first is -log(1/2)
    assert abs(float(turn_credit_loss(logits, tgt, w)) - 0.5 * float(np.log(2.0))) < 1e-5


def test_a_rollout_prices_and_the_update_trains() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t = NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=4, base_seed=7),
                      num_envs=4, buffer_size=16, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                      device=dev, turn_credit_coef=1.0, turn_credit_budget=6, turn_credit_worlds=4,
                      turn_credit_nested_p=0.5, turn_credit_steps=2, turn_credit_batch=8)
    tc = t.turn_credit
    assert tc is not None
    t.collect_rollouts()
    assert tc.stats["tc_queue"] > 0 and tc.stats["tc_priced"] > 0
    assert tc.stats["tc_priced"] + tc.stats["tc_nested_priced"] <= 6
    assert len(tc.ready) > 0
    for obs, mask, tgt, w in tc.ready:
        assert 0.0 <= w <= 1.0
        assert abs(float(tgt.float().sum()) - 1.0) < 1e-2
        assert float(tgt.float()[~mask].abs().sum()) == 0.0      # no mass off the legal set
    before = [p.detach().clone() for p in t.active_net.parameters()]
    out = tc.update(t.active_net, t.optimizer, t.max_grad_norm)
    assert "tc_loss" in out and np.isfinite(out["tc_loss"])
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, t.active_net.parameters()))
