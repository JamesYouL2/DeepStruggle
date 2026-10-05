"""LadderNet reading the CARD_EFFECTS block: beside the card rows in the trunk, and through the
card-effects head straight into the card and play-mode logits.

A warm start from a parent without the block must play exactly as the parent at step 0, the head
must act only where the block is filled, and the play-mode correction must come from the card being
played alone.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest
import torch

import ts_engine as ts

from ai.models.ladder_net import (CARD_ACTIVE_SLOT, LadderNet, create_ladder_net,
                                  ladder_config_from_state_dict, widen_for_card_effects)
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import TsVectorizedEnv
from tools.lib.player_agent import load_checkpoint_into

CE = int(ts.OBS_FEATURE_CARD_EFFECTS)
BASE = int(ts.OBS_SIZE)

#: E7-04-44's rung at a small width.
M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=0, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _rollout_obs(n_steps: int = 40, seed: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """Observations with the block, and their masks, from random games."""
    env = TsVectorizedEnv(num_envs=32, base_seed=seed)
    env.set_obs_features(CE, CE)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    o_all, m_all = [], []
    for _ in range(n_steps):
        o_all.append(np.array(obs, copy=True))
        m_all.append(np.array(masks, copy=True))
        acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)])
        obs, masks, *_ = env.step(acts)
    return (torch.from_numpy(np.concatenate(o_all)).double(),
            torch.from_numpy(np.concatenate(m_all)).bool())


@pytest.fixture(scope="module")
def rollout() -> tuple[torch.Tensor, torch.Tensor]:
    return _rollout_obs()


def _child(head: int = 32) -> LadderNet:
    torch.manual_seed(1)
    return create_ladder_net("cpu", **{**M2D, "obs_features": CE, "card_effects_head": head})


def test_the_block_is_filled_on_card_decisions_in_the_rollout(rollout) -> None:
    obs, _ = rollout
    assert obs.shape[1] == BASE + 110 * 18
    filled = (obs[:, BASE:] != 0).any(dim=1)
    assert 0.1 < float(filled.double().mean()) < 0.9


def test_the_config_is_recovered_from_the_weights() -> None:
    net = _child()
    cfg = ladder_config_from_state_dict(net.state_dict())
    assert cfg is not None
    assert cfg["card_effects_head"] == 32 and cfg["obs_features"] == CE
    assert cfg == {**net.ladder_config(), "num_attn_heads": 4}


def test_a_warm_start_plays_exactly_as_its_parent(rollout) -> None:
    obs, masks = rollout
    torch.manual_seed(0)
    parent = create_ladder_net("cpu", **M2D).double().eval()
    child = _child().double()
    load_checkpoint_into(child, widen_for_card_effects(child, parent.state_dict()))
    child.eval()
    with torch.no_grad():
        want = parent(obs[:, :BASE], masks)
        got = child(obs, masks)
    for a, b in zip(want, got):
        torch.testing.assert_close(a, b, rtol=0, atol=1e-10)
    # ... and the new paths learn: both the trunk's block columns and the head get a gradient.
    child.train()
    logits, _v, _vp = child(obs.float().double(), None)
    logits[:, :ActionEncoder.PLAY_MODE_OFFSET + 5].sum().backward()
    assert child.ce_head is not None
    out = child.ce_head[-1]
    assert isinstance(out, torch.nn.Linear)
    assert out.weight.grad is not None and float(out.weight.grad.abs().sum()) > 0
    first = child.lad_card[0]
    assert isinstance(first, torch.nn.Linear)
    w = first.weight.grad
    assert w is not None and float(w[:, -110 * 18:].abs().sum()) > 0


def test_the_head_acts_only_where_the_block_is_filled_and_play_modes_follow_the_active_card(
        rollout) -> None:
    obs, _ = rollout
    child = _child().double().eval()
    assert child.ce_head is not None
    torch.manual_seed(3)
    with torch.no_grad():
        for p in child.ce_head[-1].parameters():
            p.normal_()
        h, _attn, _tok = child._encode(obs)
        corr = child._card_effects_logits(obs, h)
    rows = obs[:, BASE:].view(-1, 110, 18)
    filled = (rows != 0).any(dim=-1)
    card = corr[:, :110]
    assert (card[~filled] == 0).all() and (card[filled] != 0).all()
    assert (corr[:, ActionEncoder.PLAY_MODE_OFFSET + 5:] == 0).all()
    cards = obs[:, child.CARD_OFFSET:child.CARD_OFFSET + child.CARD_SIZE].view(-1, 110, child.card_features)
    active_filled = ((cards[..., CARD_ACTIVE_SLOT] > 0) & filled).any(dim=1)
    mode = corr[:, ActionEncoder.PLAY_MODE_OFFSET:ActionEncoder.PLAY_MODE_OFFSET + 5]
    assert (mode[~active_filled] == 0).all()
    assert bool(active_filled.any()) and (mode[active_filled] != 0).any(dim=1).all()


def test_the_head_needs_the_block() -> None:
    with pytest.raises(ValueError, match="card_effects_head"):
        create_ladder_net("cpu", **{**M2D, "card_effects_head": 32})


def test_widening_refuses_what_it_cannot_explain() -> None:
    parent = create_ladder_net("cpu", **{**M2D, "hidden_dim": 32})
    child = _child()
    with pytest.raises(RuntimeError):
        load_checkpoint_into(child, widen_for_card_effects(child, parent.state_dict()))


@pytest.mark.parametrize("mode", ["grouped", "flat", "entity"])
def test_every_input_mode_reads_the_block_and_round_trips(mode: str) -> None:
    cfg: Dict[str, Any] = {**M2D, "input_mode": mode, "obs_features": CE, "card_effects_head": 8}
    if mode == "flat":
        cfg.update(per_entity_heads=0, head_entities="both", head_center=False)
    if mode == "entity":
        cfg.update(entity_dim=16, drop_static=False, aggregation="pool")
    net = create_ladder_net("cpu", **cfg)
    rec = ladder_config_from_state_dict(net.state_dict())
    assert rec == {**net.ladder_config(), "num_attn_heads": 4}
    x = torch.zeros(2, BASE + 110 * 18)
    x[:, BASE:] = torch.randn(2, 110 * 18)
    a, _attn, _tok = net._encode(x)
    x[:, BASE:] = torch.randn(2, 110 * 18)
    b, _attn, _tok = net._encode(x)
    assert not torch.equal(a, b)                              # the trunk reads the block
