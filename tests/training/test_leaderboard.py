"""The leaderboard: player ids, the record checks, and the two-stage rating fit."""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys
from typing import Dict, List, Tuple

import pytest

import tools.publish_hf as publish
from tools.lib import leaderboard as lb
from tools.lib.player_agent import search_spec_config
from tools.lib.player_spec import (Bot, Gumbel, Policy, PlayerSpecError, Search, load_spec,
                                   parse_player_id, player_id, spec_from_dict, spec_to_dict)
from tools.leaderboard_play import plan_pairs

ROOT = pathlib.Path(__file__).resolve().parents[2]


# ---- player ids -------------------------------------------------------------------------------


@pytest.mark.parametrize("network,spec,want", [
    ("E7-A4-R1-S44@4800M", Policy(), "E7-A4-R1-S44@4800M"),
    ("E7-A4-R1-S44@4800M", Policy(0.5), "E7-A4-R1-S44@4800M~policy(temperature=0.5)"),
    ("N@1M", Gumbel(), "N@1M~gumbel(sims=256,k=8,fpu=0.2)"),
    ("N@1M", Search(sims=128, determinize=True),
     "N@1M~search(sims=128,determinize=true,node_filter=all,subsample=1,backend=cpp,fpu=0)"),
    (None, Bot("heuristic"), "HeuristicBot"),
])
def test_player_id_round_trips(network, spec, want) -> None:
    assert player_id(network, spec) == want
    assert parse_player_id(want) == (network, spec)
    assert spec_from_dict(spec_to_dict(spec)) == spec


def test_a_search_id_may_omit_fields_but_the_canonical_id_spells_them_all() -> None:
    network, spec = parse_player_id("N@1M~gumbel(k=4)")
    assert player_id(network, spec) == "N@1M~gumbel(sims=256,k=4,fpu=0.2)"


@pytest.mark.parametrize("bad", ["N@1M~gumbel(width=3)", "N@1M~mystery", "N@1M~search(node_filter=some)",
                                 "N@1M~policy(temperature=-1)", "N@1M~search(determinize=yes)"])
def test_bad_ids_are_refused(bad: str) -> None:
    with pytest.raises(PlayerSpecError):
        parse_player_id(bad)


def test_a_stored_spec_must_name_every_field() -> None:
    with pytest.raises(PlayerSpecError):
        spec_from_dict({"kind": "gumbel", "sims": 256, "k": 8})


@pytest.mark.parametrize("spec", [Gumbel(), Gumbel(sims=64, k=4, fpu=0.0), Search(),
                                  Search(sims=128, determinize=True, node_filter="card", subsample=0.125,
                                         backend="python", fpu=0.2)])
def test_load_spec_plays_the_spec_it_names(spec) -> None:
    """The agent string must configure the searcher exactly as the spec says."""
    path, cfg, _ = search_spec_config(load_spec(spec, "x.pt"))
    assert path == "x.pt"
    assert cfg.simulations == spec.sims
    assert cfg.fpu_reduction == spec.fpu
    if isinstance(spec, Gumbel):
        assert cfg.gumbel_k == spec.k
    else:
        assert cfg.determinize == spec.determinize
        assert cfg.node_filter == ("all" if spec.node_filter == "all" else "card_playmode")
        assert cfg.subsample == spec.subsample
        assert cfg.backend == spec.backend


def test_a_policy_pins_its_own_temperature() -> None:
    assert load_spec(Policy(), "x.pt") == "temp:0:x.pt"
    assert load_spec(Policy(0.3), "x.pt") == "temp:0.3:x.pt"


# ---- the fit ----------------------------------------------------------------------------------

ENGINE = "e" * 64


def _board(tmp_path: pathlib.Path, mains: List[str], others: List[str]) -> lb.Board:
    networks = {f"{p}@1M": {"file": f"{p}.pt", "sha256": "0" * 64, "description": "", "old_names": [],
                            "hf": None, "report": None} for p in mains + others}
    players = {f"{p}@1M": {"network": f"{p}@1M", "inference": {"kind": "policy", "temperature": 0.0},
                           "description": ""} for p in mains + others}
    epochs = {"T": {"description": "", "engines": [ENGINE], "anchor": f"{mains[0]}@1M",
                    "anchor_elo": 1500.0, "main": [f"{p}@1M" for p in mains]}}
    (tmp_path / "matches").mkdir(exist_ok=True)
    board = lb.Board(networks, players, epochs, {"T": []}, tmp_path)  # type: ignore[arg-type]
    lb.save_registry(board)
    return board


def _expected(r_a: float, r_b: float, seat: float, n: int) -> Tuple[lb.Seat, lb.Seat]:
    """Win counts that are exactly the model's expectation (no draws), so the fit should return
    the generating ratings up to rounding."""
    def seat_counts(x: float) -> lb.Seat:
        w = round(n / (1 + 10 ** (-x / 400)))
        return {"w": w, "l": n - w, "d": 0}
    return seat_counts(r_a - r_b + seat), seat_counts(r_a - r_b - seat)


def _play(board: lb.Board, truth: Dict[str, float], pairs: List[Tuple[str, str]], seat: float = 40.0,
          n: int = 20000, seed: int = 10000) -> None:
    for a, b in pairs:
        us, ussr = _expected(truth[a], truth[b], seat, n)
        lb.append_record(board, "T", lb.make_record(f"{a}@1M", f"{b}@1M", us, ussr, seed, ENGINE, "c", "d"))


def test_the_fit_recovers_known_ratings_and_the_seat_term(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "C": 1420.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B", "C"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("A", "C"), ("B", "C"), ("D", "A"), ("D", "B"), ("E", "D")])
    assert lb.validate(board) == []
    fit = lb.fit_epoch(board, "T")
    got = {r.player[:-3]: r.elo for r in fit.ratings}
    for p, want in truth.items():
        assert abs(got[p] - want) < 2.0, (p, got[p], want)
    assert abs(fit.seat_us - 40.0) < 2.0
    assert next(r for r in fit.ratings if r.player == "A@1M").se == 0.0
    assert all(r.se > 0 for r in fit.ratings if r.player != "A@1M")


def test_games_without_a_main_player_never_move_the_main_ratings(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "C": 1420.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B", "C"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("A", "C"), ("B", "C"), ("D", "A")])
    before = {r.player: r.elo for r in lb.fit_epoch(board, "T").ratings if r.main}
    truth_shifted = dict(truth, D=1800.0)       # a non-main player who now plays very differently
    _play(board, truth_shifted, [("D", "B"), ("E", "D")], seed=20000)
    after = {r.player: r.elo for r in lb.fit_epoch(board, "T").ratings if r.main}
    assert before == after


def test_a_player_unconnected_to_the_main_players_is_unrated(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("D", "E")])
    fit = lb.fit_epoch(board, "T")
    assert sorted(fit.unrated) == ["D@1M", "E@1M"]
    assert {r.player for r in fit.ratings} == {"A@1M", "B@1M"}


def test_a_main_player_cut_off_from_the_anchor_is_an_error(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B", "C"], [])
    _play(board, {"A": 1500.0, "B": 1600.0, "C": 1400.0}, [("A", "B")])
    with pytest.raises(lb.LeaderboardError, match="C@1M"):
        lb.fit_epoch(board, "T")


def test_views_filter_one_fit(tmp_path: pathlib.Path) -> None:
    truth = {"A": 1500.0, "B": 1600.0, "D": 1550.0, "E": 1700.0}
    board = _board(tmp_path, ["A", "B"], ["D", "E"])
    _play(board, truth, [("A", "B"), ("D", "A"), ("E", "B")])
    fit = lb.fit_epoch(board, "T")
    every = {r.player: r.elo for r in lb.view(board, fit)}
    some = {r.player: r.elo for r in lb.view(board, fit, lineages=["D"])}
    assert set(some) == {"A@1M", "B@1M", "D@1M"}
    assert all(every[p] == some[p] for p in some)
    assert set(r.player for r in lb.view(board, fit, main_only=True)) == {"A@1M", "B@1M"}


# ---- the record checks ------------------------------------------------------------------------


def test_the_same_deals_cannot_be_recorded_twice(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    _play(board, {"A": 1500.0, "B": 1600.0}, [("A", "B")])
    with pytest.raises(lb.LeaderboardError, match="already recorded"):
        _play(board, {"A": 1500.0, "B": 1600.0}, [("B", "A")])       # either side as `a`
    _play(board, {"A": 1500.0, "B": 1600.0}, [("B", "A")], seed=20000)  # other deals are fine


def test_a_line_merged_in_twice_and_an_edited_record_are_caught(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    _play(board, {"A": 1500.0, "B": 1600.0}, [("A", "B")])
    path = tmp_path / "matches" / "T.jsonl"
    line = path.read_text()
    path.write_text(line + line)                                      # what a union merge can leave
    assert any("appears twice" in e for e in lb.validate(lb.load(tmp_path)))
    rec = json.loads(line)
    rec["a_as_us"]["w"] += 1
    path.write_text(json.dumps(rec) + "\n")
    assert any("does not match its content" in e for e in lb.validate(lb.load(tmp_path)))


def test_a_non_canonical_player_id_is_caught(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    board.players["A@1M~gumbel"] = {"network": "A@1M", "description": "",
                                    "inference": spec_to_dict(Gumbel())}  # type: ignore[assignment]
    assert any("canonical id" in e for e in lb.validate(board))


def test_site_data_aggregates_a_pair_from_the_first_players_side(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B"], [])
    lb.append_record(board, "T", lb.make_record("B@1M", "A@1M", {"w": 7, "l": 3, "d": 0},
                                                {"w": 6, "l": 3, "d": 1}, 1, ENGINE, "c", "d"))
    pair = lb.site_data(board)["epochs"]["T"]["pairs"][0]  # type: ignore[index]
    assert (pair["a"], pair["b"]) == ("A@1M", "B@1M")
    # A as US is B as USSR (B won 6, lost 3): A won 3, lost 6.
    assert pair["a_as_us"] == {"w": 3, "l": 6, "d": 1}
    assert pair["a_as_ussr"] == {"w": 3, "l": 7, "d": 0}


def test_plan_pairs_skips_what_is_recorded(tmp_path: pathlib.Path) -> None:
    board = _board(tmp_path, ["A", "B", "C"], ["D"])
    _play(board, {"A": 1500.0, "B": 1600.0, "D": 1550.0}, [("D", "A")])
    play, skipped = plan_pairs(board, "T", ["D@1M"], None, False, None)
    assert skipped == [("D@1M", "A@1M")]
    assert play == [("D@1M", "B@1M"), ("D@1M", "C@1M")]
    play, _ = plan_pairs(board, "T", ["D@1M"], None, False, 20000)
    assert len(play) == 3
    with pytest.raises(lb.LeaderboardError):
        plan_pairs(board, "T", ["D@1M"], None, False, 10000)


# ---- the committed records --------------------------------------------------------------------


def test_the_committed_leaderboard_is_consistent() -> None:
    board = lb.load()
    assert lb.validate(board) == []
    for name in board.epochs:
        if board.matches[name]:
            lb.fit_epoch(board, name)


def test_the_fit_needs_neither_torch_nor_the_engine() -> None:
    """CI builds the page's data with plain python3 (tools/scripts/build_web.sh)."""
    code = ("import sys; import tools.leaderboard; "
            "bad = [m for m in ('torch', 'numpy', 'ts_engine') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=ROOT, env={"PYTHONPATH": str(ROOT)})
    assert math.isfinite(lb.ELO_SCALE)


# ---- publishing -------------------------------------------------------------------------------


def test_the_hf_repo_mirrors_data_checkpoints(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run's files keep their run directory; a model under _models/ keeps its readable name
    even when it is a symlink into _soups/."""
    data = tmp_path / "data"
    (data / "checkpoints" / "R_20260101_000000").mkdir(parents=True)
    (data / "checkpoints" / "_soups").mkdir()
    (data / "checkpoints" / "_models").mkdir()
    (data / "checkpoints" / "_soups" / "soup.pt").write_bytes(b"x")
    (data / "checkpoints" / "_models" / "N@1M+(S1,2)@2M.pt").symlink_to("../_soups/soup.pt")
    monkeypatch.setattr(publish, "data_path", lambda *p: str(data.joinpath(*p)))
    run_file = data / "checkpoints" / "R_20260101_000000" / "snapshot_10steps.pt"
    assert publish.repo_path(str(run_file)) == "R_20260101_000000/snapshot_10steps.pt"
    assert publish.repo_path(str(data / "checkpoints" / "_models" / "N@1M+(S1,2)@2M.pt")) == "_models/N@1M+(S1,2)@2M.pt"
    with pytest.raises(lb.LeaderboardError):
        publish.repo_path(str(tmp_path / "elsewhere.pt"))


def test_only_the_final_snapshot_the_final_swa_and_leaderboard_files_get_an_onnx() -> None:
    files = ["snapshot_10000steps.pt", "snapshot_990000steps.pt", "snapshot_1000000steps.pt",
             "swa_80-160M.pt", "swa_880-960M.pt", "swa_920-1000M.pt", "snapshot_0s.pt"]
    assert publish.onnx_selection(files, set()) == ["snapshot_1000000steps.pt", "swa_920-1000M.pt"]
    assert publish.onnx_selection(files, {"snapshot_10000steps.pt", "elsewhere.pt"}) == [
        "snapshot_1000000steps.pt", "snapshot_10000steps.pt", "swa_920-1000M.pt"]
    assert publish.onnx_selection(files + ["snapshot_final.pt"], set()) == ["snapshot_final.pt", "swa_920-1000M.pt"]
    assert publish.onnx_selection(["snapshot_5steps.pt"], set()) == ["snapshot_5steps.pt"]
