"""Cell A (--ladder-region-attn-dim): masked region-and-neighbour attention before the per-country head,
grown in place -- a state written without it resumes into it and plays exactly as before."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict

import numpy as np
import torch

import ts_engine as ts
from ai.models.ladder_net import LadderNet, create_ladder_net, ladder_config_from_state_dict, region_bounds
from ai.training.generic_trainer import load_resume_state
from bindings.action_encoder import ActionEncoder


def _net(region_attn_dim: int = 0, seed: int = 0) -> LadderNet:
    torch.manual_seed(seed)
    return create_ladder_net("cpu", input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=32, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True,
                             head_entities="country", head_center=True, identity_dim=0,
                             drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
                             card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False, branch_head=True,
                             play_mode_head=True, region_attn_dim=region_attn_dim, region_attn_heads=4).eval()


def _batch(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    runner = ts.VectorizedBatchRunner(n, 4321)
    rng = np.random.default_rng(3)
    for _ in range(50):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks], True)
    return (torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32)),
            torch.from_numpy(np.asarray(runner.get_action_masks())))


def test_regions_are_slices_and_neighbours_are_averaged() -> None:
    bounds = region_bounds()
    assert len(bounds) == 6 and bounds[0][0] == 0 and bounds[-1][1] == 84
    assert all(hi == lo2 for (_, hi), (lo2, _) in zip(bounds, bounds[1:]))
    name = {ts.MapData.get_country_info(i)["name"]: i for i in range(84)}
    lo, hi = next(b for b in bounds if b[0] <= name["Iran"] < b[1])
    assert lo <= name["Iraq"] < hi and not lo <= name["France"] < hi
    net = _net(32)
    assert net.region_attn is not None
    nb = net.region_attn.get_buffer("neighbour_mean")
    assert torch.allclose(nb.sum(1), torch.ones(84)) and nb[name["Iran"], name["Afghanistan"]] > 0


def test_a_grown_network_plays_as_the_one_without_it_and_is_detected_from_its_weights() -> None:
    old, new = _net(0), _net(32, seed=9)
    missing, unexpected = new.load_state_dict(old.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith("region_attn.") for k in missing)
    obs, masks = _batch()
    with torch.no_grad():
        a, b = old(obs, masks), new(obs, masks)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    cfg = ladder_config_from_state_dict(new.state_dict())
    assert cfg is not None and cfg["region_attn_dim"] == 32 and cfg["region_attn_heads"] == 4
    again = create_ladder_net("cpu", **cfg)
    again.load_state_dict(new.state_dict())             # strict: the detected architecture is exact


def test_a_state_written_without_it_resumes_into_it(tmp_path: Any) -> None:
    old, ref = _net(0), _net(0, seed=7)
    opt = torch.optim.Adam(old.parameters(), lr=1e-3)
    blob: Dict[str, Any] = {"model_state_dict": old.state_dict(), "reference_state_dict": ref.state_dict(),
                            "optimizer_state_dict": opt.state_dict(), "total_env_steps": 1000, "iteration": 3,
                            "elapsed_seconds": 1.0, "seed": 44, "torch_rng_state": torch.get_rng_state(),
                            "numpy_rng_state": np.random.get_state(), "ema_state_dict": None}
    path = os.path.join(str(tmp_path), "resume.pt")
    torch.save(blob, path)
    new, new_ref = _net(32, seed=1), _net(32, seed=2)
    trainer = SimpleNamespace(device="cpu", optimizer=torch.optim.Adam(new.parameters(), lr=1e-3),
                              reference_net=new_ref, total_env_steps=0)
    load_resume_state(path, new, trainer, seed=44)
    obs, masks = _batch()
    with torch.no_grad():
        assert torch.equal(old(obs, masks)[0], new(obs, masks)[0])
    # One step trains the new layer (its output projection leaves zero).
    new.train()
    # A fresh per-country head starts at zero, which blocks every gradient upstream of it; a trained one
    # (A13+R48's) does not, so give it trained-like weights before the step.
    assert new.pe_country is not None
    last = new.pe_country[-1]
    assert isinstance(last, torch.nn.Linear)
    with torch.no_grad():
        last.weight.normal_(0.0, 0.1)
    logits = new(obs, torch.ones_like(masks))[0]       # every move legal: the country block carries gradient
    logits[:, ActionEncoder.NODE_OFFSET:ActionEncoder.BRANCH_OFFSET].pow(2).mean().backward()
    trainer.optimizer.step()
    assert new.region_attn is not None and new.region_attn.out.weight.abs().sum() > 0
