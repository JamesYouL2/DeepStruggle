"""A one-position distillation record (tools/bank_distill_targets.py) streams as exactly that position."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import ts_engine as ts

from ai.training.warmup_dataset_loader import WarmupDataset


def _decision_state() -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, 12345)
    return st


def test_save_record_yields_its_position_and_target(tmp_path: Path) -> None:
    st = _decision_state()
    mask = np.asarray(ts.get_flat_action_mask(st, False))
    legal = [int(i) for i in np.flatnonzero(mask)]
    path = tmp_path / "d.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        # An illegal action in the target is dropped, the legal one carries all the mass.
        f.write(json.dumps({"save": st.to_save_json(),
                            "search_pi": {"a": [legal[0], mask.shape[0] + 5], "v": [1.0, 3.0]}}) + "\n")
    out = list(WarmupDataset(str(path)).stream_policy_transitions())
    assert len(out) == 1
    obs, got_mask, target, dt = out[0]
    np.testing.assert_array_equal(obs, np.asarray(ts.extract_observation(st, st.ctx().decision_player)))
    np.testing.assert_array_equal(got_mask, mask)
    assert target[legal[0]] == 1.0 and target.sum() == 1.0
    assert dt == int(st.ctx().decision_type)


def test_save_records_mix_with_replayed_games(tmp_path: Path) -> None:
    st = _decision_state()
    a0 = int(np.flatnonzero(np.asarray(ts.get_flat_action_mask(st, False)))[0])
    path = tmp_path / "d.jsonl.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        f.write(json.dumps({"seed": 12345, "actions": [
            {"flat_action": a0, "search_pi": {"a": [a0], "v": [1.0]}}]}) + "\n")
        f.write(json.dumps({"save": st.to_save_json(), "search_pi": {"a": [a0], "v": [1.0]}}) + "\n")
    out = list(WarmupDataset(str(path)).stream_policy_transitions())
    assert len(out) == 2
    np.testing.assert_array_equal(out[0][0], out[1][0])
