"""The action-value fit (`--mode q-fit`): on leads a small head can learn, the held-out correlation with
the two seeds' mean rises past the starting network's logits, the Q root takes the stable departures, and
the network's own weights do not move."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training.q_fit import QFitConfig, run_q_fit
from tests.training.test_ladder_tokens import SHALLOW
from tools.lib.engine_fingerprint import fingerprint


def _dataset(tmp: Path) -> str:
    """Card choices of random games. Candidates: the network's move (the highest legal card) first, then up
    to three more legal cards; each move has one fixed value, so a candidate leads by its value less the
    network's move's, read by each seed with its own noise of SE 0.02. A baseline teacher `h0` reads the
    same leads with SE 0.08."""
    runner = ts.VectorizedBatchRunner(24, 5)
    rng = np.random.default_rng(0)
    states, games = [], []
    for _ in range(500):
        if all(runner.get_terminals()):
            break
        masks = np.asarray(runner.get_action_masks())
        for i in range(24):
            st = runner.get_state(i)
            if not runner.get_terminals()[i] and int(st.ctx().decision_type) == int(ts.DecisionType.SELECT_CARD) \
                    and masks[i].sum() >= 3:
                states.append(np.frombuffer(st.raw_bytes(), dtype=np.uint8))
                games.append(i)
        runner.step_flat_all([int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks], True)
    n = len(states)
    value = rng.normal(0, 0.06, 220).astype(np.float32)
    cand = np.full((n, 16), -1, dtype=np.int16)
    true = np.full((n, 16), np.nan, dtype=np.float32)
    for k, s in enumerate(states):
        legal = np.flatnonzero(ts.ActionMask.generate_flat_mask(ts.state_from_raw_bytes(s.tobytes())))
        others = list(legal[:-1][:3])
        cand[k, 0] = legal[-1]
        cand[k, 1:1 + len(others)] = others
        true[k, 0] = 0.0
        true[k, 1:1 + len(others)] = value[others] - value[legal[-1]]
    game = np.asarray(games, dtype=np.int32)
    held = game % 6 == 0
    have = cand >= 0

    def noisy(se: float) -> np.ndarray:
        lead = np.where(have, true + rng.normal(0, se, true.shape), np.nan).astype(np.float32)
        lead[:, 0] = 0.0
        return lead

    def pick_of(lead: np.ndarray) -> np.ndarray:
        best = np.nanargmax(np.where(have, lead, -np.inf), 1)
        return np.where(lead[np.arange(n), best] > 0.04, cand[np.arange(n), best], cand[:, 0]).astype(np.int16)

    l1, l2, lb = noisy(0.02), noisy(0.02), noisy(0.08)
    se = np.where(have, 0.02, np.nan).astype(np.float32)
    out = {"states": np.stack(states), "game": game, "heldout": held, "net_argmax": cand[:, 0].copy(),
           "cand_hq": cand, "lead_hq": l1, "lead_se_hq": se, "worlds_hq": np.full(n, 64, np.int16),
           "pick_hq": pick_of(l1),
           "cand2_hq": np.where(held[:, None], cand, -1).astype(np.int16),
           "lead2_hq": np.where(held[:, None], l2, np.nan).astype(np.float32),
           "lead_se2_hq": np.where(held[:, None], se, np.nan).astype(np.float32),
           "worlds2_hq": np.where(held, 64, 0).astype(np.int16),
           "pick2_hq": np.where(held, pick_of(l2), -1).astype(np.int16),
           "cand_h0": cand, "lead_h0": lb, "lead_se_h0": np.where(have, 0.08, np.nan).astype(np.float32),
           "worlds_h0": np.full(n, 16, np.int16)}
    path = str(tmp / "targets.npz")
    np.savez(path, **out)
    with open(path + ".meta.json", "w") as f:
        json.dump({"engine_fingerprint": fingerprint()}, f)
    return path


def _init(tmp: Path) -> str:
    torch.manual_seed(0)
    path = str(tmp / "init.pt")
    torch.save(create_ladder_net("cpu", **SHALLOW).state_dict(), path)
    return path


def test_the_q_head_learns_the_leads_and_takes_the_stable_departures(tmp_path: Path) -> None:
    data, init = _dataset(tmp_path), _init(tmp_path)
    before = torch.load(init, weights_only=True)
    run = run_q_fit(QFitConfig(dataset=data, target="hq", init=init, batch=64, lr=3e-3, max_epochs=60,
                               eval_every=50, patience=6, warmup_steps=10, device="cpu", bf16=False),
                    run_name=None, description="test", output_dir=str(tmp_path / "run"))
    meta = json.loads((Path(run) / "metadata.json").read_text())
    best, fixed = meta["best"], meta["fixed"]
    assert fixed["reliability_mean"] > 0.9
    assert best["r_dis"] > 0.8 > fixed["r_dis_logits"]
    assert best["r_dis"] > fixed["r_dis_base"]
    assert best["take_0.04"] > 0.7 and best["leave_0.04"] < 0.2  # the teacher departs past 0.04
    after = torch.load(Path(run) / "q_fit_best.pt", weights_only=True)["model"]
    assert all(torch.equal(before[k], after[k]) for k in before if k != "keep_idx")


def test_a_teacher_without_statistics_is_refused(tmp_path: Path) -> None:
    data, init = _dataset(tmp_path), _init(tmp_path)
    with pytest.raises(SystemExit, match="no candidate statistics"):
        run_q_fit(QFitConfig(dataset=data, target="h4w128", init=init, device="cpu", bf16=False),
                  run_name=None, description="test", output_dir=str(tmp_path / "run"))
