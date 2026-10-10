"""Gumbel-choice expert-iteration targets: the generator's `--target gchoice` records, the arms
`tools/gchoice_targets.py` rewrites them into, and its held-out check."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.training.warmup_dataset_loader import WarmupDataset
from tools.gchoice_targets import arm, check, margin, replay, target
from tools.generate_search_targets import generate


def _rec(raw: int = 5, c: int = 7, q_raw: float = 0.1, q_c: float = 0.3) -> Dict[str, Any]:
    return {"raw": raw, "pi": {"a": [5, 7], "v": [0.8, 0.2]}, "value": 0.0,
            "g": {"256": {"c": c, "n": {"5": 4.0, "7": 4.0}, "q": {"5": q_raw, "7": q_c}}}}


def test_a_departure_is_a_one_hot_on_the_choice_and_an_agreement_keeps_the_policy() -> None:
    own = {"a": [5, 7], "v": [0.8, 0.2]}
    assert target(_rec(), "departures", "256", 0.0) == {"a": [7], "v": [1.0]}
    assert target(_rec(), "own", "256", 0.0) == own
    agree = _rec(c=5)
    for form in ("departures", "gated", "own"):
        assert target(agree, form, "256", 0.0) == own


def test_the_gate_reads_the_roots_own_margin_in_win_probability() -> None:
    assert margin(_rec(q_raw=0.1, q_c=0.3), "256") == pytest.approx(0.1)
    assert target(_rec(), "gated", "256", 0.05) == {"a": [7], "v": [1.0]}
    assert target(_rec(), "gated", "256", 0.2) == {"a": [5, 7], "v": [0.8, 0.2]}
    unvisited = _rec()
    unvisited["g"]["256"]["q"]["5"] = None
    assert margin(unvisited, "256") is None
    assert target(unvisited, "gated", "256", 0.0)["a"] == [5, 7]


@pytest.fixture(scope="module")
def targets(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, str]:
    d = tmp_path_factory.mktemp("gchoice")
    torch.manual_seed(0)
    ckpt = str(d / "m.pt")
    torch.save(create_coldwar_net_v2("cpu").state_dict(), ckpt)
    out = str(d / "targets.jsonl.gz")
    assert generate(ckpt, total_games=1, batch_size=1, sims=8, node_filter="all",
                    temperature=1.0, output_path=out, device_str="cpu", target="gchoice",
                    gumbel_sims=[8, 4], gumbel_k=2, subsample=0.2,
                    seed_offset=990_000_000) == 0   # the workflow's held-out offset
    return {"ckpt": ckpt, "targets": out, "dir": str(d)}


def _records(path: str) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [a for line in f for a in json.loads(line)["actions"]]


def test_the_generator_records_every_budget_beside_the_networks_own_policy(targets: Dict[str, str]) -> None:
    recs = [a for a in _records(targets["targets"]) if "gchoice" in a]
    assert recs, "no decision was searched"
    for a in recs:
        g = a["gchoice"]
        assert set(g["g"]) == {"8", "4"}
        assert g["raw"] in g["pi"]["a"] and abs(sum(g["pi"]["v"]) - 1.0) < 1e-2
        c = g["g"]["8"]["c"]
        assert str(c) in g["g"]["8"]["q"]      # the choice was one of the root's candidates
        one_hot = a["search_pi"] == {"a": [c], "v": [1.0]}
        assert one_hot == (c != g["raw"])
    meta = json.load(open(targets["targets"] + ".meta.json"))
    assert meta["searcher"]["target"] == "gchoice" and meta["searcher"]["search_pi_from"] == 8


def test_arms_share_the_positions_and_the_own_arm_carries_no_departure(targets: Dict[str, str]) -> None:
    n_searched = sum("gchoice" in a for a in _records(targets["targets"]))
    rows = {}
    for form in ("departures", "own"):
        out = str(Path(targets["dir"]) / f"{form}.jsonl.gz")
        meta = arm([targets["targets"]], form, 8, 0.0, out)
        assert meta["searched"] == n_searched
        assert all("gchoice" not in a for a in _records(out))
        rows[form] = [a["search_pi"] for a in _records(out) if "search_pi" in a]
        loaded = list(WarmupDataset(out).stream_policy_transitions())
        assert len(loaded) == n_searched
        for _obs, mask, pi, _dt in loaded:
            assert np.all(pi[mask == 0] == 0)
    assert len(rows["departures"]) == len(rows["own"]) == n_searched
    assert meta["targeted"] == 0                 # the last arm written is `own`


def test_the_check_replays_every_searched_position_and_sees_no_move_against_itself(
        targets: Dict[str, str]) -> None:
    n_searched = sum("gchoice" in a for a in _records(targets["targets"]))
    assert sum(1 for _ in replay([targets["targets"]])) == n_searched
    res = check([targets["targets"]], targets["ckpt"], targets["ckpt"], 8)
    assert res["positions"] == n_searched
    assert res["agreements"]["changed"] in (0.0,) or res["agreements"]["n"] == 0
    assert res["departures"]["n"] == 0 or res["departures"]["moved"] == 0.0
    assert res["base_argmax_is_recorded"] == pytest.approx(1.0)
