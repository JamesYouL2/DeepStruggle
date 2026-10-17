"""P34's value fit (ai/training/value_fit.py) end to end on a tiny playout-labelled dataset: the run
evaluates the initial network on the held-out games first, writes a checkpoint the agent loader
reads strictly (so a rollout root can take it as its leaf network), records its settings where
run_codes.py reads them, and learns a constant it can see in its labels."""
from __future__ import annotations

import json
import pathlib
from typing import Any, Dict

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training.value_fit import ValueFitConfig, parse_arch, run_value_fit
from tests.training.test_ladder_tokens import SHALLOW
from tools.lib.player_agent import NeuralAgent


def _states(n: int, seed: int) -> np.ndarray:
    runner = ts.VectorizedBatchRunner(1, seed)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        for _ in range(int(rng.integers(5, 40))):
            if runner.get_terminals()[0]:
                runner.reset_game(0, int(rng.integers(1 << 30)))
            m = np.asarray(runner.get_action_masks())[0]
            runner.step_flat_all([int(rng.choice(np.flatnonzero(m)))], True)
        if runner.get_terminals()[0]:
            runner.reset_game(0, int(rng.integers(1 << 30)))
        out.append(np.frombuffer(runner.get_state(0).raw_bytes(), dtype=np.uint8))
    return np.stack(out)


def _split(n: int, seed: int, score: float) -> Dict[str, Any]:
    st = _states(n, seed)
    return {"state": st, "id": np.arange(n, dtype=np.int64) + seed * 1000, "score": np.full(n, score, np.float32),
            "score_se": np.zeros(n, np.float32), "final_vp": np.zeros(n, np.float32),
            "turn_vp": np.zeros(n, np.float32), "turn_defcon": np.full(n, 3.0, np.float32),
            "swing": np.full((n, 7), np.nan, np.float32), "swing_frac": np.zeros((n, 7), np.float32),
            "holdout": np.zeros(n, bool)}


def test_a_value_fit_runs_end_to_end(tmp_path: pathlib.Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    np.savez(ds / "train.npz", **_split(48, 1, 0.8))
    np.savez(ds / "holdout.npz", **_split(16, 2, 0.8))
    (ds / "manifest.json").write_text(json.dumps({"dataset": "x~D1(seed=1,games=1)", "engine": "e" * 64,
                                                  "files": {}}))
    init = tmp_path / "init.pt"
    torch.save(create_ladder_net("cpu", **SHALLOW).state_dict(), init)
    out = tmp_path / "run"
    cfg = ValueFitConfig(dataset=str(ds), init=str(init), batch=16, lr=3e-3, warmup_steps=1, max_epochs=40,
                         eval_every=10, patience=100, device="cpu", bf16=False)
    run_dir = run_value_fit(cfg, run_name=None, description="test", output_dir=str(out),
                            cli={"mode": "value-fit", "value_dataset": str(ds), "warmup_checkpoint": str(init)})
    lines = [json.loads(x) for x in (pathlib.Path(run_dir) / "value_fit_metrics.jsonl").read_text().splitlines()]
    assert lines[0]["step"] == 0
    # The labels are a constant 0.8 the network can learn: the held-out error falls well below where
    # the random initial network starts.
    assert lines[-1]["hold_score_mse"] < 0.5 * lines[0]["hold_score_mse"]
    meta = json.loads((pathlib.Path(run_dir) / "metadata.json").read_text())
    assert meta["mode"] == "value-fit" and meta["value_dataset"] == str(ds)
    assert meta["resumed_from"] == str(init)
    agent = NeuralAgent.from_checkpoint(str(pathlib.Path(run_dir) / "value_fit_best.pt"), device="cpu")
    assert agent.model is not None


def test_value_arch_settings_are_typed_and_need_fresh_weights(tmp_path: pathlib.Path) -> None:
    assert parse_arch(["hidden_dim=768", "num_res_blocks=8", "head_center=true", "logit_cap=2.5"]) == {
        "hidden_dim": 768, "num_res_blocks": 8, "head_center": True, "logit_cap": 2.5}
    init = tmp_path / "init.pt"
    torch.save(create_ladder_net("cpu", **SHALLOW).state_dict(), init)
    cfg = ValueFitConfig(dataset=str(tmp_path), init=str(init), arch={"hidden_dim": 96}, device="cpu")
    with pytest.raises(SystemExit, match="value-reinit"):
        run_value_fit(cfg, run_name=None, description=None, output_dir=str(tmp_path / "run"))
