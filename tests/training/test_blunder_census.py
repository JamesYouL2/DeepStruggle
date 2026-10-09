"""The two-stage catastrophic-blunder census: the screen's cheap checks are sound, a validation
row records what its verdict rests on (exact engine labels apart from estimated regret), and the
pipeline runs end to end on a throwaway checkpoint -- no data under `data/` is assumed to exist.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import state_from_token
from tools.scripts.bank_playouts import (escalation_shortlist, main as bank_main, verdict_of)
from tools.scripts.blunder_census import (DecisionLog, TRIGGERS, legal_actions, main, metrics,
                                          mover_of, root_cause, rescued_by_search, row_id,
                                          safe_alternative, triggers)


def _checkpoint(tmp_path: Path) -> str:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    path = tmp_path / "m.pt"
    torch.save(m.state_dict(), str(path))
    return str(path)


def _screen(tmp_path: Path, games: int = 2, control_rate: float = 0.5) -> str:
    out_dir = str(tmp_path / "screen")
    assert main(["screen", "--model", _checkpoint(tmp_path), "--games", str(games),
                 "--search-sims", "8", "--search-k", "2", "--control-rate", str(control_rate),
                 "--out-dir", out_dir]) == 0
    return out_dir


def _rows(path: str) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


# -- the cheap checks ---------------------------------------------------------------------------


def test_triggers_separate_the_exact_catastrophes_from_the_cheap_disagreement() -> None:
    assert triggers(5, 5, {5: "loss", 6: "normal"}) == ["forced-loss"]
    assert triggers(5, 5, {5: "loss", 6: "loss"}) == []          # nothing else does better
    assert triggers(5, 5, {5: "normal", 6: "win"}) == ["missed-forced-win"]
    assert triggers(5, 5, {5: "win", 6: "normal"}) == []          # it took the win
    assert triggers(5, 6, {5: "normal", 6: "normal"}) == ["g32-disagrees"]
    assert triggers(5, 6, {5: "loss", 6: "normal"}) == ["forced-loss", "g32-disagrees"]
    assert set(triggers(5, 6, {5: "loss", 6: "win"})) <= set(TRIGGERS)


def test_safe_alternative_prefers_a_win_then_the_most_probable_non_loss() -> None:
    probs = [(5, 0.9), (6, 0.5), (7, 0.4)]
    assert safe_alternative({5: "loss", 6: "normal", 7: "normal"}, 5, probs) == 6
    assert safe_alternative({5: "loss", 6: "win", 7: "normal"}, 5, probs) == 6
    assert safe_alternative({5: "loss", 6: "loss", 7: "loss"}, 5, probs) is None
    assert safe_alternative({5: "normal", 6: "normal"}, 5, probs) == 6   # 7 by probability order


def test_verdict_of_keeps_exact_mistakes_apart_from_estimated_regret() -> None:
    labels = {5: "loss", 6: "normal"}
    assert verdict_of(5, (0.5, 0.1), labels, 2.0, 0.05) == "exact-forced-loss"
    assert verdict_of(5, (0.5, 0.1), {5: "normal", 6: "win"}, 2.0, 0.05) == "exact-missed-win"
    assert verdict_of(5, (0.10, 0.03), {5: "normal", 6: "normal"}, 2.0, 0.05) == "confirmed-regret"
    assert verdict_of(5, (0.04, 0.03), {5: "normal", 6: "normal"}, 2.0, 0.05) == "unresolved"
    assert verdict_of(5, (-0.10, 0.03), {5: "normal", 6: "normal"}, 2.0, 0.05) == "refuted"
    assert verdict_of(5, (0.10, 0.0), {5: "normal", 6: "normal"}, 2.0, 0.05) == "unresolved"


def test_escalation_shortlist_takes_the_most_serious_unresolved_only() -> None:
    rows = [{"id": f"{i:016x}"} for i in range(4)]
    prev = [{"id": f"{0:016x}", "verdict": "unresolved", "best": 7, "diff_vs_greedy": {"7": [0.02, 0.05]}},
            {"id": f"{1:016x}", "verdict": "unresolved", "best": 7, "diff_vs_greedy": {"7": [0.30, 0.20]}},
            {"id": f"{2:016x}", "verdict": "confirmed-regret", "best": 7, "diff_vs_greedy": {"7": [0.9, 0.1]}},
            {"id": f"{3:016x}", "verdict": "unresolved", "best": 7, "diff_vs_greedy": {"7": [0.10, 0.09]}}]
    assert [r["id"] for r in escalation_shortlist(rows, prev, 2)] == [f"{1:016x}", f"{3:016x}"]
    assert len(escalation_shortlist(rows, prev, 0)) == 3
    assert escalation_shortlist(rows, [], 2) == []


def test_root_cause_names_the_defcon_mechanism_and_the_estimated_kind() -> None:
    base = {"verdict": "exact-forced-loss", "decision_type": "SELECT_PLAY_MODE", "defcon": 2,
            "card": 4, "phase": "ACTION_ROUND"}
    assert "Duck and Cover" in root_cause(base)
    assert "forced loss" in root_cause(dict(base, card=1, decision_type="POINT_NODE"))
    assert root_cause({"verdict": "exact-missed-win"}) == "missed a forced win"
    assert "headline" in root_cause({"verdict": "confirmed-regret", "decision_type": "SELECT_CARD",
                                     "phase": "HEADLINE"})


def test_rescued_by_search_reads_the_search_move_and_its_score() -> None:
    row = {"moves": {"greedy": 5, "search": 6}, "exact": {"missed_win": False, "forced_loss": False},
           "score": {"5": 0.2, "6": 0.6}}
    assert rescued_by_search(row)
    assert not rescued_by_search(dict(row, moves={"greedy": 5, "search": 5}))
    exact = {"missed_win": True, "forced_loss": False, "labels": {"5": "normal", "6": "win", "7": "normal"}}
    assert rescued_by_search({"moves": {"greedy": 5, "search": 6}, "exact": exact, "score": {}})
    assert not rescued_by_search({"moves": {"greedy": 5, "search": 7}, "exact": exact, "score": {}})


# -- stage 1: the screening output --------------------------------------------------------------


def test_screen_writes_replayable_rows_and_resumes(tmp_path: Path) -> None:
    out_dir = _screen(tmp_path)
    cands, ctrl = _rows(f"{out_dir}/candidates.jsonl.gz"), _rows(f"{out_dir}/control.jsonl.gz")
    assert cands and ctrl
    summary = json.load(open(f"{out_dir}/summary.json"))
    assert summary["decisions"] >= summary["candidates"] + summary["control"] > 0
    assert summary["candidates"] == len(cands) and summary["control"] == len(ctrl)
    assert summary["model_sha256"] and summary["seed"] == 0
    assert sum(summary["triggers"].values()) >= summary["candidates"]   # >=: one row, several triggers
    for r in cands + ctrl:
        if r.get("control"):
            assert r["triggers"] == []
        else:
            assert r["triggers"] and all(t in TRIGGERS for t in r["triggers"])
        st = state_from_token(r["pos"])                       # the exact state travels intact
        assert legal_actions(st) == r["legal"]
        assert r["greedy"] in r["legal"] and r["g32"] in r["legal"] and r["played"] in r["legal"]
        assert r["safety"]["safe"] is None or r["safety"]["safe"] in r["legal"]
        assert r["id"] == row_id(r["pos"])
        assert r["model_sha256"] == summary["model_sha256"]
        assert r["game"]["seed"] == summary["seed"] + r["game"]["index"]
        assert r["decision"] >= 1 and r["pos"]
        assert mover_of(st) == (ts.Player.US if r["side"] == "US" else ts.Player.USSR)
        assert int(st.defcon) == r["defcon"] and int(st.turn) == r["turn"]
        assert set(r["legal"]) <= set(range(ActionEncoder.FLAT_ACTION_SIZE))
    # a resume finishes the remaining games and refuses to start over
    assert main(["screen", "--model", _checkpoint(tmp_path), "--games", "3", "--search-sims", "8",
                 "--search-k", "2", "--control-rate", "0.5", "--out-dir", out_dir, "--resume"]) == 0
    ledger = [json.loads(line) for line in open(f"{out_dir}/games.jsonl")]
    assert sorted(g["game"] for g in ledger) == [0, 1, 2]
    with pytest.raises(SystemExit):
        main(["screen", "--model", _checkpoint(tmp_path), "--games", "3", "--search-sims", "8",
              "--search-k", "2", "--out-dir", out_dir])


# -- stage 2: validation rows and the report ----------------------------------------------------


def test_validate_records_verdicts_and_the_report_merges_the_stages(tmp_path: Path) -> None:
    out_dir = _screen(tmp_path, games=1)
    cands, ctrl = _rows(f"{out_dir}/candidates.jsonl.gz"), _rows(f"{out_dir}/control.jsonl.gz")
    rows = (cands[:2] if len(cands) >= 2 else cands) + ctrl[:1]
    inp = tmp_path / "input.jsonl.gz"
    with gzip.open(inp, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    val = str(tmp_path / "validation.jsonl.gz")
    assert bank_main(["validate", "--input", str(inp), "--model", _checkpoint(tmp_path),
                      "--pairs", "2", "--search-sims", "8", "--search-k", "2", "--search-seeds", "2",
                      "--out", val]) == 0
    vrows = _rows(val)
    assert len(vrows) == len(rows)
    meta = json.load(open(val + ".meta.json"))
    assert meta["budget"] == "validation" and meta["search_sims"] == 8 and meta["rows"] == len(rows)
    for r in vrows:
        assert r["verdict"] in ("exact-missed-win", "exact-forced-loss", "confirmed-regret",
                                "refuted", "unresolved")
        assert set(r["score"]) == {str(a) for a in r["moves"].values()}
        assert r["exact"]["labels"][str(r["moves"]["greedy"])] == r["exact"]["greedy"]
        assert r["search"]["choice"] in [x["pick"] for x in r["search"]["runs"]]
        assert len(r["search"]["runs"]) == 2 and len({x["seed"] for x in r["search"]["runs"]}) == 2
        assert r["seeds"]["continuation"] not in {x["seed"] for x in r["search"]["runs"]}
        for a, d in r["diff_vs_greedy"].items():
            assert a != str(r["moves"]["greedy"]) and len(d) == 2
        assert state_from_token(r["pos"]) is not None
    # escalation: only the unresolved, and at the larger budget
    esc = str(tmp_path / "escalation.jsonl.gz")
    assert bank_main(["validate", "--input", str(inp), "--model", _checkpoint(tmp_path),
                      "--pairs", "2", "--search-sims", "16", "--search-k", "2", "--search-seeds", "1",
                      "--seed", "1", "--escalate-from", val, "--escalate-cap", "1",
                      "--out", esc]) == 0
    erows = _rows(esc)
    assert len(erows) <= 1
    assert json.load(open(esc + ".meta.json"))["budget"] == "escalation"
    if erows:
        assert erows[0]["id"] in {r["id"] for r in vrows if r["verdict"] == "unresolved"}
    # the report merges both stages into the required figures
    report = str(tmp_path / "report.md")
    assert main(["report", "--screen", f"{out_dir}/candidates.jsonl.gz",
                 "--control", f"{out_dir}/control.jsonl.gz", "--summary", f"{out_dir}/summary.json",
                 "--validation", val, "--escalation", esc, "--out", report]) == 0
    text = open(report).read()
    for wanted in ("candidate count", "confirmation rate", "per 1,000 decisions",
                   "the cheap screening found", "rescued by 256-simulation search",
                   "unresolved even under the stronger search", "## Runtime"):
        assert wanted in text


def test_metrics_weights_the_control_sample_by_the_unscreened_decisions() -> None:
    screened = [{"id": "a"}, {"id": "b"}]
    control = [{"id": "c"}]
    validation = [{"id": "a", "verdict": "confirmed-regret", "moves": {"greedy": 5, "search": 6},
                   "exact": {"missed_win": False, "forced_loss": False}, "best": 6,
                   "diff_vs_greedy": {"6": [0.30, 0.10]},
                   "score": {"5": 0.0, "6": 1.0}},
                  {"id": "b", "verdict": "refuted", "moves": {"greedy": 5, "search": 5},
                   "exact": {}, "best": 5, "score": {}},
                  {"id": "c", "verdict": "exact-forced-loss", "moves": {"greedy": 5, "search": 6},
                   "exact": {"missed_win": False, "forced_loss": True, "labels": {"6": "normal"}},
                   "best": 6, "diff_vs_greedy": {"6": [0.30, 0.10]},
                   "score": {}}]
    c = metrics(screened, control, validation, validation, decisions=1000)
    # 998 unscreened decisions over 1 control row: the control stands for 998 positions
    assert c["control_weight"] == 998.0
    assert c["confirmed_candidates"] == 1 and c["confirmed_control"] == 1
    assert c["per_1000_decisions"] == pytest.approx((1 + 998.0) / 1000 * 1000)
    assert c["per_1000_exact"] == pytest.approx(998.0 / 1000 * 1000)
    assert c["per_1000_estimated"] == pytest.approx(1 / 1000 * 1000)
    assert c["detected_by_screen"] == pytest.approx(1 / 999.0)
    assert c["detected_exact"] == 0.0       # the one exact mistake sat in the control sample
    assert c["detected_estimated"] == pytest.approx(1.0)   # only the candidate is estimated-class
    assert c["noise"]["control_with_alternative"] == 1 and c["noise"]["control_z_ge_2"] == 1
    assert c["noise"]["candidate_with_alternative"] == 1 and c["noise"]["candidate_z_ge_2"] == 1
    assert c["rescued_by_256"] == pytest.approx(1.0)     # both confirmed rows are rescued by search
    assert c["confirmation_rate"] == pytest.approx(0.5)
    assert c["unresolved_share"] == 0.0


def test_decision_log_keeps_only_decisions_with_a_choice() -> None:
    log = DecisionLog()
    runner = ts.VectorizedBatchRunner(1, 99)
    for _ in range(30):
        st = runner.get_state(0)
        if ts.Engine.is_terminal(st):
            break
        masks = np.asarray(runner.get_action_masks())
        log.observe(st, int(np.flatnonzero(masks[0])[0]))
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    assert log.kept and all(len(legal_actions(st)) >= 2 for _, st, _ in log.kept)
    assert [d for d, _, _ in log.kept] == sorted(d for d, _, _ in log.kept)
