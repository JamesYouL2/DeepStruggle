"""--resume-widen (ai/models/grow.py): a LadderNet's grouped projections widened without changing its function."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

import ts_engine as ts
from ai.models.grow import widen_optimizer_state, widen_state_dict
from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.training.generic_trainer import load_resume_state


def _net(proj: int, seed: int = 0) -> LadderNet:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=proj, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True,
                             head_entities="country", head_center=True, identity_dim=0,
                             drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
                             card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False).eval()


def _batch(n: int = 48) -> tuple[torch.Tensor, torch.Tensor]:
    """Real observations and masks from a few random games, so the inputs have the layout's structure."""
    runner = ts.VectorizedBatchRunner(n, 1234)
    rng = np.random.default_rng(5)
    for _ in range(40):
        masks = np.asarray(runner.get_action_masks())
        acts = [int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks]
        runner.step_flat_all(acts, True)
    obs = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32))
    return obs, torch.from_numpy(np.asarray(runner.get_action_masks()))


def _outputs(net: LadderNet, obs: torch.Tensor, masks: torch.Tensor) -> list[torch.Tensor]:
    with torch.no_grad():
        out = net(obs, masks)
    return [t for t in out if isinstance(t, torch.Tensor)]


def _widened(old: LadderNet, k: int, noise: float) -> LadderNet:
    new = _net(old.entity_proj_dim * k, seed=1)
    new.load_state_dict(widen_state_dict(old.state_dict(), k, noise=noise, generator=torch.Generator().manual_seed(3)))
    return new


def test_widening_without_noise_computes_the_same_function() -> None:
    old = _net(32)
    new = _widened(old, 2, noise=0.0)
    obs, masks = _batch()
    for a, b in zip(_outputs(old, obs, masks), _outputs(new, obs, masks)):
        assert a.shape == b.shape
        finite = torch.isfinite(a)
        assert torch.equal(finite, torch.isfinite(b))
        assert torch.allclose(a[finite], b[finite], atol=1e-5, rtol=1e-5)


def test_the_symmetry_breaking_perturbation_moves_the_output_only_at_second_order() -> None:
    old = _net(32)
    obs, masks = _batch()
    base = _outputs(old, obs, masks)[0]
    finite = torch.isfinite(base)
    small = (_outputs(_widened(old, 2, noise=1e-3), obs, masks)[0][finite] - base[finite]).abs().max().item()
    large = (_outputs(_widened(old, 2, noise=1e-2), obs, masks)[0][finite] - base[finite]).abs().max().item()
    assert small < 1e-3
    assert large > 0.0 and small < large / 20          # 10x the perturbation, ~100x the change


def test_copies_differ_and_their_perturbations_sum_to_zero() -> None:
    old = _net(32).state_dict()
    new = widen_state_dict(old, 2, noise=1e-3, generator=torch.Generator().manual_seed(3))
    w, w_new = old["lad_board.0.weight"], new["lad_board.0.weight"]
    a, b = w_new[:32], w_new[32:]
    assert not torch.equal(a, b)
    assert torch.allclose((a + b) / 2, w, atol=1e-6)
    assert torch.equal(new["lad_board.1.weight"][:32], new["lad_board.1.weight"][32:])
    f, f_new = old["fusion_in.0.weight"], new["fusion_in.0.weight"]
    assert f_new.shape == (f.shape[0], f.shape[1] + 64)
    assert torch.allclose(f_new[:, :32] + f_new[:, 32:64], f[:, :32])          # board block
    assert torch.allclose(f_new[:, 64:96] + f_new[:, 96:128], f[:, 32:64])     # card block
    assert torch.equal(f_new[:, 128:], f[:, 64:])                              # the globals, unchanged


def test_adam_moments_follow_the_copies() -> None:
    old = _net(32).train()
    obs, masks = _batch()
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    out = old(obs, masks)
    (out[1].float().sum() + out[0][torch.isfinite(out[0])].sum() * 1e-3).backward()
    opt.step()
    names = [n for n, _ in old.named_parameters()]
    saved = opt.state_dict()
    widened = widen_optimizer_state(saved, names, 32, 2)
    idx = {n: i for i, n in enumerate(names)}
    row = widened["state"][idx["lad_card.0.weight"]]
    orig = saved["state"][idx["lad_card.0.weight"]]
    assert torch.allclose(row["exp_avg"][:32], orig["exp_avg"] / 2)
    assert torch.allclose(row["exp_avg_sq"][32:], orig["exp_avg_sq"] / 4)
    col = widened["state"][idx["fusion_in.0.weight"]]["exp_avg"]
    assert torch.equal(col[:, 32:64], saved["state"][idx["fusion_in.0.weight"]]["exp_avg"][:, :32])
    new = _net(64, seed=1).train()
    new.load_state_dict(widen_state_dict(old.state_dict(), 2, generator=torch.Generator().manual_seed(3)))
    opt_new = torch.optim.Adam(new.parameters(), lr=1e-3)
    opt_new.load_state_dict(widened)
    out = new(obs, masks)
    out[1].float().sum().backward()
    opt_new.step()                                                         # the shapes line up


def test_load_resume_state_widens_the_network_its_reference_and_the_optimizer(tmp_path: Any) -> None:
    old = _net(32)
    ref = _net(32, seed=7)
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    blob: Dict[str, Any] = {"model_state_dict": old.state_dict(), "reference_state_dict": ref.state_dict(),
                            "optimizer_state_dict": opt.state_dict(), "total_env_steps": 1000, "iteration": 3,
                            "elapsed_seconds": 1.0, "seed": 44, "torch_rng_state": torch.get_rng_state(),
                            "numpy_rng_state": np.random.get_state(), "ema_state_dict": None}
    path = os.path.join(str(tmp_path), "resume.pt")
    torch.save(blob, path)
    new, new_ref = _net(64, seed=1), _net(64, seed=2)
    trainer = SimpleNamespace(device="cpu", optimizer=torch.optim.Adam(new.parameters(), lr=1e-3),
                              reference_net=new_ref, total_env_steps=0)
    state = load_resume_state(path, new, trainer, seed=44, widen=True)
    assert state["total_env_steps"] == 1000
    obs, masks = _batch()
    for net_old, net_new in ((old, new), (ref, new_ref)):
        a, b = _outputs(net_old, obs, masks)[1], _outputs(net_new, obs, masks)[1]
        assert torch.allclose(a, b, atol=1e-3)
