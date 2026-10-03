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


def _rec(results: dict, priors: dict) -> dict:
    return {"candidates": [{"prefix": [a], "prior": priors[a]} for a in results],
            "results": [results[a] for a in results]}


def test_contrast_target_follows_the_paired_evidence() -> None:
    from tools.bank_distill_targets import EVENT, contrast_target

    # The event wins every pair the influence play loses: P(event) ~ 1.
    t = contrast_target(_rec({112: "0" * 32, EVENT: "2" * 32, 113: "0" * 32}, {112: 0.9, EVENT: 0.0, 113: 0.1}))
    assert t is not None and t["a"][0] == EVENT and t["v"][0] > 0.99
    # Identical results pair by pair: no evidence either way, P(event) = 1/2, and the non-event
    # half keeps the model's own split between its Ops modes.
    tie = contrast_target(_rec({112: "0212" * 8, EVENT: "0212" * 8, 113: "0000" * 8},
                               {112: 0.9, EVENT: 0.0, 113: 0.1}))
    assert tie is not None and tie["v"][0] == 0.5
    rest = dict(zip(tie["a"][1:], tie["v"][1:]))
    assert abs(rest[112] - 0.45) < 1e-9 and abs(rest[113] - 0.05) < 1e-9
    # No event candidate: no target.
    assert contrast_target(_rec({112: "2" * 8, 113: "0" * 8}, {112: 0.5, 113: 0.5})) is None


def test_gap_compares_the_event_with_the_models_own_move() -> None:
    from tools.lib.gap_labels import EVENT, gap

    # The model's own move (candidate 0) is influence; the event wins half the pairs it loses.
    rec = _rec({112: "0000", EVENT: "2200", 113: "2222"}, {112: 0.9, EVENT: 0.0, 113: 0.1})
    assert gap(rec) == 0.5   # against the model's move, not against the luckier coup line
    # The model's own move is the event: compare with its likeliest other mode.
    rec2 = _rec({EVENT: "2222", 112: "0000", 113: "2222"}, {EVENT: 0.6, 112: 0.3, 113: 0.1})
    assert gap(rec2) == 1.0
    assert gap(_rec({112: "22", 113: "00"}, {112: 0.5, 113: 0.5})) is None
