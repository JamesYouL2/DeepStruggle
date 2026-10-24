"""AIVAT over a tournament's game log (ai/eval/aivat.py).

The estimator is only unbiased if its replay is the tournament's game, so the replay is held to the
logged end of every game; the rest checks that the corrections are what they claim to be.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Sequence

import numpy as np
import pytest

import ts_engine as ts
from ai.eval.aivat import (ReplayMismatch, ScoredGame, load_games, main, score_games, summarise,
                           vp_value)
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.player_agent import HeuristicAgent, HeuristicV2Agent
from tools.tournament import run_massive_tournament


@pytest.fixture(scope="module")
def logged(tmp_path_factory: pytest.TempPathFactory) -> List[Dict[str, Any]]:
    path = str(tmp_path_factory.mktemp("aivat") / "games.jsonl")
    BatchMatchRunner.play_parallel_matchup(HeuristicAgent(), HeuristicV2Agent(), games_per_side=3,
                                           batch_chunk_size=4, device="cpu", temperature=0.0,
                                           log_games_file=path)
    return load_games(path)


def test_a_logged_game_replays_to_its_logged_end(logged: List[Dict[str, Any]]) -> None:
    assert len(logged) == 6 and all(e["actions"] for e in logged)
    games = score_games(logged, vp_value, clones=2)
    assert [g.utility for g in games] == [float(e["utility"]) for e in logged]
    assert all(g.chance_steps > 0 for g in games)


def test_a_wrong_action_is_caught_not_scored(logged: List[Dict[str, Any]]) -> None:
    e = dict(logged[0])
    e["actions"] = list(e["actions"][:-5])
    with pytest.raises(ReplayMismatch):
        score_games([e], vp_value, clones=1)


def test_a_constant_value_corrects_nothing(logged: List[Dict[str, Any]]) -> None:
    def const(states: Sequence[ts.GameState]) -> np.ndarray:
        return np.full(len(states), 0.25)
    assert all(g.correction == 0.0 for g in score_games(logged, const, clones=2))


def test_vp_corrections_are_finite_and_seeded(logged: List[Dict[str, Any]]) -> None:
    one = score_games(logged, vp_value, clones=3, salt=1)
    assert all(np.isfinite(g.correction) and g.correction != 0.0 for g in one)
    assert [g.correction for g in score_games(logged, vp_value, clones=3, salt=1)] == \
        [g.correction for g in one]
    assert [g.correction for g in score_games(logged, vp_value, clones=3, salt=2)] != \
        [g.correction for g in one]


def test_summary_is_a_s_score_clustered_by_deal_pair() -> None:
    # Two deals; a wins both seats of the first, splits the second.
    games = [ScoredGame(1, "a", "b", 1.0, 0.5, 3), ScoredGame(1, "b", "a", -1.0, -0.5, 3),
             ScoredGame(2, "a", "b", 1.0, 0.0, 3), ScoredGame(2, "b", "a", 1.0, 0.0, 3),
             ScoredGame(3, "a", "a", 1.0, 0.0, 3)]                  # self-play: left out
    (row,) = summarise(games)
    assert (row["a"], row["b"], row["games"], row["pairs"]) == ("a", "b", 4, 2)
    assert row["raw"]["score"] == pytest.approx(0.75)               # pairs: 1.0 and 0.5
    assert row["raw"]["se"] == pytest.approx(0.25)
    # a's corrected scores: 0.75 both games of deal 1, 1.0 / 0.0 on deal 2 -> pairs 0.75, 0.5
    assert row["aivat"]["score"] == pytest.approx(0.625)
    assert row["variance_ratio"] == pytest.approx(4.0)
    assert row["correction"] == pytest.approx(0.125)
    assert row["raw"]["elo"] == pytest.approx(400 * np.log10(3))


def test_every_matchup_of_a_sequential_tournament_is_logged(tmp_path) -> None:
    path = str(tmp_path / "games.jsonl")
    run_massive_tournament(["heuristic", "heuristic_v2", "random"], games_per_side=2, device="cpu",
                           temperature=0.0, anchor_model="HeuristicBot", pack_pairs=1,
                           log_games=path)
    entries = load_games(path)
    assert len({(min(e["us_agent"], e["ussr_agent"]), max(e["us_agent"], e["ussr_agent"]))
                for e in entries}) == 3
    assert len(entries) == 12


def test_cli_writes_the_table_and_json(logged: List[Dict[str, Any]], tmp_path,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    games, out = tmp_path / "g.jsonl", tmp_path / "out.json"
    games.write_text("".join(json.dumps(e) + "\n" for e in logged))
    assert main(["--games", str(games), "--value", "vp", "--clones", "2",
                 "--output-json", str(out)]) == 0
    assert "variance ratio" in capsys.readouterr().out
    (row,) = json.loads(out.read_text())["matchups"]
    assert row["pairs"] == 3
