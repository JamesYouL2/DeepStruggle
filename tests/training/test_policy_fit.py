"""P36c's policy fit (`--mode policy-fit`): on targets a small network can fit, the training fit rises to
near 1 and the run records the three readings; the value heads do not move."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training.policy_fit import PolicyFitConfig, dense_targets, run_policy_fit
from tests.training.test_ladder_tokens import SHALLOW


def _dataset(tmp: Path) -> str:
    """Card choices of random games; the 'teacher' plays each position's lowest legal card, which a
    network can learn from the observation (the hand is in it)."""
    runner = ts.VectorizedBatchRunner(24, 5)
    rng = np.random.default_rng(0)
    states, games, net = [], [], []
    for _ in range(500):
        if all(runner.get_terminals()):
            break
        masks = np.asarray(runner.get_action_masks())
        for i in range(24):
            st = runner.get_state(i)
            if not runner.get_terminals()[i] and int(st.ctx().decision_type) == int(ts.DecisionType.SELECT_CARD) \
                    and masks[i].sum() >= 2:
                states.append(np.frombuffer(st.raw_bytes(), dtype=np.uint8))
                games.append(i)
                net.append(int(np.flatnonzero(masks[i])[-1]))
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks], True)
    n = len(states)
    acts = np.full((n, 16), -1, dtype=np.int16)
    probs = np.zeros((n, 16), dtype=np.float32)
    for k, s in enumerate(states):
        legal = np.flatnonzero(ts.ActionMask.generate_flat_mask(ts.state_from_raw_bytes(s.tobytes())))
        acts[k, 0], probs[k, 0] = legal[0], 1.0
    game = np.asarray(games, dtype=np.int32)
    held = game % 6 == 0
    out = {"states": np.stack(states), "game": game, "heldout": held, "net_argmax": np.asarray(net, np.int16),
           "acts_16": acts, "probs_16": probs, "pick_16": acts[:, 0].copy(),
           "acts2_16": np.where(held[:, None], acts, -1).astype(np.int16), "probs2_16": probs,
           "pick2_16": np.where(held, acts[:, 0], -1).astype(np.int16)}
    path = str(tmp / "targets.npz")
    np.savez(path, **out)
    return path


def test_the_dense_targets_put_each_probability_on_its_move() -> None:
    acts = np.asarray([[3, 7, -1], [0, -1, -1]], dtype=np.int16)
    probs = np.asarray([[0.75, 0.25, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    d = dense_targets(acts, probs)
    assert d.shape == (2, 220) and d[0, 3] == 0.75 and d[0, 7] == 0.25 and d[1, 0] == 1.0 and d.sum() == 2.0


def test_a_fit_converges_and_records_the_three_readings(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **SHALLOW)
    ckpt = str(tmp_path / "init.pt")
    torch.save(net.state_dict(), ckpt)
    before = {k: v.clone() for k, v in net.state_dict().items() if k.startswith("val_")}
    run = run_policy_fit(PolicyFitConfig(dataset=ds, worlds=16, init=ckpt, batch=64, lr=1e-3, warmup_steps=5,
                                         max_epochs=40, eval_every=20, patience=6, fit_sample=500, device="cpu",
                                         bf16=False),
                         run_name="E7-A1-R1-S44", description="test", output_dir=str(tmp_path / "E7-A1-R1-S44_test"))
    rows = [json.loads(line) for line in open(os.path.join(run, "policy_fit_metrics.jsonl"))]
    last = rows[-1]
    for key in ("train_fit", "held_agree_all", "ceiling_all", "held_ce"):
        assert key in last, key
    assert last["train_fit"] > rows[0]["train_fit"] and last["train_fit"] > 0.6
    assert last["ceiling_all"] == 1.0
    after = torch.load(os.path.join(run, "policy_fit_last.pt"), weights_only=True)
    assert all(torch.equal(before[k], after[k]) for k in before), "the value heads moved"
