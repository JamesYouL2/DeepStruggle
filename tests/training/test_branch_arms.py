"""The two branch arms of 2026-10-02: resuming a saved state into a network with an added training-
only head (C2's card-event head), and --play-mode-temp."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.generic_trainer import load_resume_state, save_resume_state
from ai.training.nash_pg import EVENT_SLOT, PLAY_MODE_LO, play_mode_rows

SHALLOW: dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _trainer(card_aux: bool, **kw: Any) -> NashPGTrainer:
    from bindings.ts_env import TsVectorizedEnv
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_ladder_net(dev, **SHALLOW, card_aux=card_aux)
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    return NashPGTrainer(active_net=model, env=env, num_envs=8, buffer_size=16, lr=3e-4, eta=0.1,
                         ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


def test_a_branch_can_add_the_card_event_head(tmp_path: Any) -> None:
    a = _trainer(False)
    a.train_iteration()                                   # gives the optimizer some state
    path = str(tmp_path / "resume.pt")
    save_resume_state(path, a.active_net, a, iteration=1, total_env_steps=128, elapsed_seconds=1.0, seed=0)

    b = _trainer(True, aux_card_coef=1.0)
    load_resume_state(path, b.active_net, b, seed=0)
    # everything the saved network has is restored exactly
    sa, sb = a.active_net.state_dict(), b.active_net.state_dict()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert any(k.startswith("card_aux_head.") for k in sb) and not any(k.startswith("card_aux_head.") for k in sa)
    # so policy and value are the saved network's
    obs = torch.rand(4, a.active_net.TOTAL_OBS_SIZE)
    mask = torch.ones(4, 220, dtype=torch.bool)
    with torch.no_grad():
        for x, y in zip(a.active_net(obs, mask), b.active_net(obs, mask)):
            assert torch.allclose(x, y)
    # the optimizer kept the saved moments and gives the new head none yet
    n_old = len(list(a.active_net.parameters()))
    st = b.optimizer.state_dict()["state"]
    assert len(st) == len(a.optimizer.state_dict()["state"]) and max(st) < n_old
    b.train_iteration()                                   # and it trains


def test_a_resume_still_refuses_any_other_mismatch(tmp_path: Any) -> None:
    a = _trainer(True, aux_card_coef=1.0)
    path = str(tmp_path / "resume.pt")
    save_resume_state(path, a.active_net, a, iteration=1, total_env_steps=128, elapsed_seconds=1.0, seed=0)
    b = _trainer(False)                                   # the saved state has a head this one lacks
    with pytest.raises(RuntimeError, match="unexpected"):
        load_resume_state(path, b.active_net, b, seed=0)


def test_play_mode_rows() -> None:
    m = torch.zeros(4, 220, dtype=torch.bool)
    m[0, PLAY_MODE_LO:PLAY_MODE_LO + 5] = True            # event, space, three Ops: a play-mode decision
    m[1, PLAY_MODE_LO + 2:PLAY_MODE_LO + 5] = True        # Ops modes only: the deferred Ops choice
    m[2, [EVENT_SLOT, 3]] = True                           # mixed with a card slot
    m[3, 120:130] = True                                   # influence placement
    assert play_mode_rows(m).tolist() == [True, False, False, False]


def test_play_mode_temp_explores_but_stores_the_policys_own_log_prob() -> None:
    t = _trainer(False, play_mode_temp=4.0)
    m = t.collect_rollouts()
    buf = t.buffer
    obs = buf.obs.reshape(-1, buf.obs.shape[-1])
    masks = buf.masks.reshape(-1, buf.masks.shape[-1])
    acts = buf.actions.reshape(-1).long()
    with torch.no_grad():
        lp = torch.log_softmax(t.active_net(obs.float(), masks)[0].float(), -1).gather(1, acts[:, None]).squeeze(1)
    assert torch.allclose(lp, buf.log_probs.reshape(-1).float(), atol=1e-4)
    assert "play_mode_event_frac" in m or not bool(play_mode_rows(masks).any())


def test_play_mode_temp_must_be_positive() -> None:
    with pytest.raises(ValueError):
        _trainer(False, play_mode_temp=0.0)
