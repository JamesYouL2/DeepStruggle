"""The distillation diagnostics: the teacher ladder of `tools/distill_targets.py` (named teachers, the split
search and its merge), policy-fit's target choice and head re-initialisation, and `tools/distill_probe.py`'s
readings."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from ai.models.ladder_net import create_ladder_net
from ai.training.policy_fit import PolicyFitConfig, reinit_policy_heads, run_policy_fit
from tests.training.test_ladder_tokens import SHALLOW
from tests.training.test_policy_fit import _dataset
from tools import distill_targets as dt
from tools.distill_probe import agree_table, knn_readings, main as probe_main, track_rows


def _ckpt(tmp: Path, seed: int = 0) -> str:
    torch.manual_seed(seed)
    path = str(tmp / f"net{seed}.pt")
    torch.save(create_ladder_net("cpu", **SHALLOW).state_dict(), path)
    return path


def test_a_teacher_spec_is_checked_and_a_root_spec_keeps_its_options() -> None:
    assert dt.parse_teacher("h4=root:4:16:4:z2:earlier:own:keyed:aligned:box4") == (
        "h4", "root:4:16:4:z2:earlier:own:keyed:aligned:box4")
    cfg = dt.root_config("root:16:4:0:argmax:earlier:own:keyed:aligned:box4")
    assert (cfg.k, cfg.worlds, cfg.horizon, cfg.rule, cfg.stop, cfg.leaf_side) == (16, 4, 0, "argmax", "earlier", "own")
    assert cfg.keyed and cfg.aligned and cfg.box4 and not cfg.oracle
    assert dt.parse_teacher("soup=net:/x/soup.pt")[1] == "net:/x/soup.pt"
    for bad in ("h4", "=root:4:16:4:z2", "h4=gumbel:256", "h4=root:4:16", "a b=net:x", "o=root:4:16:4:z2:oracle"):
        with pytest.raises(ValueError):
            dt.parse_teacher(bad)


def test_the_record_teachers_keep_their_seeds() -> None:
    """P36c's targets were searched at seed + W and the held-out second copy at seed + 7919 W: a teacher named by
    its world count with P36c's spec rebuilds them; any other spec gets its own seeds, and a shard offsets both."""
    assert dt.teacher_seeds("128", "root:4:128:4:z2:earlier:own:keyed", 3637, 0) == (3637 + 128, 3637 + 7919 * 128)
    a = dt.teacher_seeds("h4", "root:4:16:4:z2:earlier:own:keyed:aligned:box4", 3637, 0)
    b = dt.teacher_seeds("h2", "root:4:16:2:z2:earlier:own:keyed:aligned:box4", 3637, 0)
    assert a != b and a[0] != a[1]
    assert dt.teacher_seeds("h4", "root:4:16:4:z2:earlier:own:keyed:aligned:box4", 3637, 3)[0] != a[0]


def test_the_shards_cover_every_row_once_and_a_double_cover_is_refused() -> None:
    parts = [dt.shard_rows(10, f"{i}/3") for i in (1, 2, 3)]
    assert np.array_equal(np.sort(np.concatenate(parts)), np.arange(10))
    with pytest.raises(ValueError):
        dt.shard_rows(10, "4/3")
    base = {"states": np.zeros((4, 2), np.uint8)}
    sh = {"rows": np.asarray([0, 1, 2, 3]), "pick_x": np.asarray([5, 6, 7, 8], np.int16)}
    assert dt.merge(base, [sh])["pick_x"].tolist() == [5, 6, 7, 8]
    with pytest.raises(ValueError, match="exactly once"):
        dt.merge(base, [sh, {"rows": np.asarray([0]), "pick_x": np.asarray([1], np.int16)}])
    with pytest.raises(ValueError, match="exactly once"):
        dt.merge(base, [{"rows": np.asarray([0, 1]), "pick_x": np.asarray([1, 2], np.int16)}])


def test_the_split_search_merges_into_the_whole(tmp_path: Path) -> None:
    """Positions collected once; a deterministic teacher (another network's policy) searched in two shards and
    merged equals the unsplit search, and a root teacher's shards fill every row."""
    ck, other = _ckpt(tmp_path, 0), _ckpt(tmp_path, 1)
    pos = str(tmp_path / "pos.npz")
    assert dt.main(["--checkpoint", ck, "--games", "4", "--positions", "40", "--holdout", "2", "--collect-only",
                    "--device", "cpu", "--output", pos]) == 0
    teachers = ["--teacher", f"soup=net:{other}", "--teacher", "h0=root:16:2:0:argmax:earlier:own:keyed"]
    whole = str(tmp_path / "whole.npz")
    assert dt.main(["--checkpoint", ck, "--states", pos, "--device", "cpu", "--output", whole, *teachers]) == 0
    shards = []
    for i in (1, 2):
        shards.append(str(tmp_path / f"s{i}.npz"))
        assert dt.main(["--checkpoint", ck, "--states", pos, "--shard", f"{i}/2", "--device", "cpu",
                        "--output", shards[-1], *teachers]) == 0
    merged = str(tmp_path / "merged.npz")
    assert dt.main(["--checkpoint", ck, "--states", pos, "--merge", *shards, "--output", merged]) == 0
    with np.load(whole) as w, np.load(merged) as m:
        n = w["states"].shape[0]
        for key in ("acts_soup", "probs_soup", "pick_soup", "acts2_soup", "pick2_soup"):
            assert np.array_equal(w[key], m[key]), key
        assert (m["pick_h0"] >= 0).all() and m["acts_h0"].shape == (n, dt.WIDTH)
        held = m["heldout"].astype(bool)
        assert (m["pick2_h0"][held] >= 0).all() and (m["pick2_h0"][~held] == -1).all()
        legal_top = m["acts_soup"][:, 0]
        assert np.allclose(m["probs_soup"].sum(1), 1.0, atol=1e-4) and (legal_top >= 0).all()
    meta = json.load(open(merged + ".meta.json"))
    assert set(meta["teachers"]) == {"soup", "h0"}


def test_reinit_heads_resets_the_policy_heads_and_keeps_the_trunk() -> None:
    torch.manual_seed(0)
    m = create_ladder_net("cpu", **SHALLOW)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    assert reinit_policy_heads(m) > 0
    after = m.state_dict()
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved and all(k.startswith(("policy_head", "pe_")) for k in moved), sorted(moved)
    assert not any(k.startswith("val_") for k in moved)


def test_a_fit_reads_a_named_teacher_and_refuses_a_missing_one(tmp_path: Path) -> None:
    ds = _dataset(tmp_path)
    with np.load(ds) as z:
        d = {(k.replace("_16", "_low") if k.endswith("_16") else k): z[k] for k in z.files}
    np.savez(ds, **d)
    ck = _ckpt(tmp_path)
    cfg = dict(dataset=ds, worlds=16, init=ck, batch=64, lr=1e-3, warmup_steps=5, max_epochs=2, eval_every=10,
               patience=3, fit_sample=200, device="cpu", bf16=False)
    with pytest.raises(SystemExit, match="no targets '16'"):
        run_policy_fit(PolicyFitConfig(**cfg), run_name="E7-A1-R1-S44", description="t",
                       output_dir=str(tmp_path / "E7-A1-R1-S44_a"))
    run = run_policy_fit(PolicyFitConfig(**cfg, target="low", reinit_heads=True), run_name="E7-A1-R1-S44",
                         description="t", output_dir=str(tmp_path / "E7-A1-R1-S44_b"))
    meta = json.load(open(os.path.join(run, "metadata.json")))
    assert meta["config"]["target"] == "low" and meta["config"]["reinit_heads"] is True
    assert os.path.exists(os.path.join(run, "policy_fit_last.pt"))


def _probe_set() -> Dict[str, Any]:
    """Six positions, legal moves 0..3 everywhere; the network plays 0; teacher a departs to 1 at rows 0-2,
    teacher b departs to 1 at rows 0-1 and to 2 at row 3."""
    net = np.zeros(6, np.int16)
    a = np.asarray([1, 1, 1, 0, 0, 0])
    b = np.asarray([1, 1, 0, 2, 0, 0])
    d: Dict[str, Any] = {"net_argmax": net, "heldout": np.asarray([0, 0, 0, 1, 1, 1], bool)}
    for name, top in (("a", a), ("b", b)):
        acts = np.full((6, 16), -1, np.int16)
        acts[:, 0] = top
        d[f"acts_{name}"] = acts
        d[f"acts2_{name}"] = acts.copy()
        d[f"pick_{name}"] = top.astype(np.int16)
        d[f"pick2_{name}"] = top.astype(np.int16)
    return d


def test_agree_reads_departures_and_the_pairs() -> None:
    t = agree_table(_probe_set(), ["a", "b"])
    assert "| a | 50.0% | 50.0% |" in t
    assert "| a (3) | 100.0% (100.0%) | 66.7% (66.7%) |" in t
    assert "| b (3) | 66.7% (66.7%) | 100.0% (100.0%) |" in t


def test_knn_readings_count_what_the_neighbours_play() -> None:
    top = np.asarray([1, 1, 1, 0, 0])
    net = np.zeros(5, np.int64)
    masks = np.zeros((5, 220), np.uint8)
    masks[:, :3] = 1
    masks[2, 1] = 0                             # the teacher's move is not legal at neighbour 2
    r = knn_readings(top, net, masks, np.asarray([0]), np.asarray([[1, 2, 3]]))
    assert r["departures"] == 1 and r["neighbours_legal"] == 2
    assert r["neighbour_teacher_plays_it"] == 0.5 and r["neighbour_network_plays_it"] == 0.0
    assert r["vote_predicts_teacher_on_departures"] == 1.0      # one vote for 1 against none for 0 + the tie bonus
    assert r["vote_predicts_network_move"] == 1.0


def test_track_reads_a_checkpoint_against_the_teacher() -> None:
    top = np.asarray([1, 1, 0, 0])
    base = np.zeros(4, np.int64)
    ck = np.asarray([1, 0, 2, 0])
    r = track_rows(top, base, ck, np.arange(4))
    assert r["takes_teacher_on_departures"] == 0.5 and r["keeps_base_on_departures"] == 0.5
    assert r["leaves_agreed_move"] == 0.5 and r["agrees_with_teacher"] == 0.5


def test_agree_runs_from_the_command_line(tmp_path: Path) -> None:
    path = str(tmp_path / "set.npz")
    np.savez(path, **_probe_set())
    out = str(tmp_path / "r.json")
    assert probe_main(["agree", "--dataset", path, "--json", out]) == 0
    assert "on departures of" in json.load(open(out))["table"]
