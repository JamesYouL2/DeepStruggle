"""A tournament split across processes plays the same games, and an ONNX export plays as its
checkpoint.

The split (tools/lib/parallel_tournament.py) is only worth having if it is invisible: the same
deals, the same games, the same merged numbers. The ONNX agent is only worth having if a rating of
the published file is a rating of the network it was exported from.
"""
from __future__ import annotations

import json
import math
import os
import random
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.batch_tournament import BatchMatchRunner, merge_matchup_results
from tools.lib.game_step import drain_chance
from tools.lib.parallel_tournament import plan_shards, run_matchups_parallel, seed_shard
from tools.lib.player_agent import (HeuristicAgent, HeuristicV2Agent, NeuralAgent, OnnxAgent,
                                    load_agent)


def _assert_same_result(got: Dict[str, Any], want: Dict[str, Any]) -> None:
    """Equal except for timing; per-game averages merged by weight may differ in the last bit."""
    assert set(got) == set(want)
    for k, v in want.items():
        if k == "elapsed_seconds":
            continue
        if isinstance(v, float):
            assert math.isclose(got[k], v, rel_tol=1e-12, abs_tol=1e-12), k
        else:
            assert got[k] == v, k


def _log(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return {e["game_index"]: e for e in map(json.loads, f)}


def test_pair_subsets_merge_into_the_whole_matchup(tmp_path) -> None:
    """Playing the pairs in pieces and merging equals playing them at once -- including across
    chunk boundaries, where a pair's seed depends on the chunking of the whole matchup."""
    a, b = HeuristicAgent(), HeuristicV2Agent()
    kw: Dict[str, Any] = dict(games_per_side=5, batch_chunk_size=4, device="cpu",
                              temperature=0.0, track_choices=True)
    whole = BatchMatchRunner.play_parallel_matchup(
        a, b, log_games_file=str(tmp_path / "whole.jsonl"), **kw)
    subsets = [[0, 3], [1], [2, 4]]
    parts = [BatchMatchRunner.play_parallel_matchup(
        a, b, pairs=ks, log_games_file=str(tmp_path / f"part{n}.jsonl"), **kw)
        for n, ks in enumerate(subsets)]

    _assert_same_result(merge_matchup_results(parts), whole)
    logged: Dict[int, Dict[str, Any]] = {}
    for n in range(len(subsets)):
        logged.update(_log(str(tmp_path / f"part{n}.jsonl")))
    assert logged == _log(str(tmp_path / "whole.jsonl"))


def test_pairs_are_checked() -> None:
    a, b = HeuristicAgent(), HeuristicV2Agent()
    for bad in ([], [0, 0], [5], [-1]):
        with pytest.raises(ValueError, match="pairs must be"):
            BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=5, device="cpu",
                                                   pairs=bad)


def test_merge_refuses_parts_of_different_matchups() -> None:
    a = {"agent_a": "x", "agent_b": "y"}
    b = {"agent_a": "x", "agent_b": "z"}
    with pytest.raises(ValueError, match="different matchups"):
        merge_matchup_results([a, b])


def test_shards_cover_every_pair_of_every_matchup_once() -> None:
    matchups = [(0, 1), (0, 2), (1, 2)]
    shards = plan_shards(matchups, games_per_side=7, shard_pairs=3)
    for m in matchups:
        ks = [k for s in shards if s.pair == m for k in s.pairs]
        assert ks == list(range(7))
    assert [len(s.pairs) for s in shards if s.pair == (0, 1)] == [3, 3, 1]


def test_workers_play_the_same_games_as_one_process(tmp_path) -> None:
    """Deterministic agents: the pool reproduces the unsplit matchup game for game."""
    whole = BatchMatchRunner.play_parallel_matchup(
        HeuristicAgent(), HeuristicV2Agent(), games_per_side=4, device="cpu", temperature=0.0,
        log_games_file=str(tmp_path / "whole.jsonl"))
    got = run_matchups_parallel(["heuristic", "heuristic_v2"], [(0, 1)], games_per_side=4,
                                workers=2, shard_pairs=1, device="cpu", temperature=0.0,
                                log_games=str(tmp_path / "pool.jsonl"))
    _assert_same_result(got[(0, 1)], whole)
    assert _log(str(tmp_path / "pool.jsonl")) == _log(str(tmp_path / "whole.jsonl"))
    assert not [p for p in os.listdir(tmp_path) if ".part-" in p], "shard logs left behind"


def test_workers_play_the_tournament_opening() -> None:
    """--opening scripts the agents the parent loads; workers load their own from specs, so the
    opening has to travel with the spec. Dropped, the workers would play the agents' own setups."""
    from tools.tournament import run_massive_tournament

    kw: Dict[str, Any] = dict(games_per_side=2, device="cpu", temperature=0.0,
                              anchor_model="HeuristicBot", opening="human")
    one = run_massive_tournament(["heuristic", "heuristic_v2"], pack_pairs=1, **kw)
    split = run_massive_tournament(["heuristic", "heuristic_v2"], workers=2, shard_pairs=1, **kw)
    assert all(name.endswith("+human") for name in split["models"])
    _assert_same_result(split["matchup"], one["matchup"])


def test_random_play_does_not_depend_on_the_number_of_workers(tmp_path) -> None:
    """RandomBot draws from the global generator, so its games depend on what ran before them in
    the same process -- unless every shard starts it afresh. At a fixed shard size, one worker
    and three must then play the same games."""
    logs: List[Dict[int, Dict[str, Any]]] = []
    for w in (1, 3):
        path = str(tmp_path / f"w{w}.jsonl")
        run_matchups_parallel(["random", "heuristic"], [(0, 1)], games_per_side=3, workers=w,
                              shard_pairs=1, device="cpu", log_games=path)
        logs.append(_log(path))
    assert logs[0] == logs[1]


def test_seed_shard_restarts_an_agents_own_stream() -> None:
    class _Streamed(HeuristicAgent):
        """An agent that carries a generator between games, as a determinizing search does."""

        def __init__(self) -> None:
            super().__init__()
            self.rng = random.Random(0)

        def reseed(self, seed: int) -> None:
            self.rng = random.Random(seed)

    d = _Streamed()
    d.rng.random()
    seed_shard(1234, [HeuristicAgent(), d])
    assert d.rng.getstate() == random.Random(1234 + 1).getstate()


# -- ONNX -----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, str]:
    """An untrained V2 network as a checkpoint, and its tools/export_onnx.py export."""
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from tools.export_onnx import export

    run = tmp_path_factory.mktemp("onnx") / "T-01-01_20260101_000000"
    run.mkdir()
    torch.manual_seed(7)
    ckpt = run / "snapshot_1000000steps.pt"
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), ckpt)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))
    out = run / "model.onnx"
    export(str(ckpt), str(out), positions=32)
    return {"pt": str(ckpt), "onnx": str(out)}


def test_load_agent_reads_an_export_with_its_label(exported: Dict[str, str]) -> None:
    agent = load_agent(exported["onnx"], device="cpu")
    assert isinstance(agent, OnnxAgent)
    assert agent.name == NeuralAgent.from_checkpoint(exported["pt"], device="cpu").name
    assert agent.obs_size == int(ts.OBS_SIZE)
    assert agent.merged_influence is False


def test_an_export_plays_the_games_its_checkpoint_plays(exported: Dict[str, str]) -> None:
    """Greedy, the export and the checkpoint make the same choice at every decision, so the
    games -- and every number a tournament reports -- are the same."""
    kw: Dict[str, Any] = dict(games_per_side=3, device="cpu", temperature=0.0)
    torch_res = BatchMatchRunner.play_parallel_matchup(
        NeuralAgent.from_checkpoint(exported["pt"], device="cpu"), HeuristicAgent(), **kw)
    onnx_res = BatchMatchRunner.play_parallel_matchup(OnnxAgent(exported["onnx"]),
                                                      HeuristicAgent(), **kw)
    _assert_same_result(onnx_res, torch_res)


def test_sampling_only_ever_picks_legal_actions(exported: Dict[str, str]) -> None:
    agent = OnnxAgent(exported["onnx"])
    rng = random.Random(3)
    obs, masks = [], []
    s = ts.GameState()
    ts.Engine.init_game(s, 99)
    drain_chance(s)
    while not ts.Engine.is_terminal(s) and len(obs) < 200:
        p = s.ctx().decision_player
        if p != ts.Player.NONE:
            obs.append(np.asarray(ts.extract_observation(s, p), dtype=np.float32))
            masks.append(np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8))
        ts.Engine.step_flat(s, rng.choice(ActionEncoder.get_legal_indices(s)))
        drain_chance(s)
    o, m = np.stack(obs), np.stack(masks)
    for t in (0.1, 1.0, 5.0):
        picks = agent.act_batch(o, m, t, greedy=False)
        assert m[np.arange(len(picks)), picks].all(), f"illegal action sampled at tau={t}"


def test_a_determinized_search_does_not_depend_on_the_number_of_workers(
        exported: Dict[str, str], tmp_path) -> None:
    """Search is what the workers are for: it decides in Python, one tree at a time. A
    determinizing search samples its worlds from its own stream, so at a fixed shard size one
    worker and two must still play the same games."""
    spec = f"search:{exported['pt']}:2:determinize"
    logs: List[Dict[int, Dict[str, Any]]] = []
    for w in (1, 2):
        path = str(tmp_path / f"w{w}.jsonl")
        run_matchups_parallel([spec, "heuristic"], [(0, 1)], games_per_side=2, workers=w,
                              shard_pairs=1, device="cpu", temperature=0.0, log_games=path)
        logs.append(_log(path))
    assert len(logs[0]) == 4
    assert logs[0] == logs[1]


def test_a_file_that_is_not_an_export_is_refused(exported: Dict[str, str], tmp_path) -> None:
    import onnx

    proto = onnx.load(exported["onnx"])
    del proto.metadata_props[:]
    bare = tmp_path / "bare.onnx"
    onnx.save_model(proto, str(bare))
    with pytest.raises(ValueError, match="not a ts-onnx-v1 export"):
        OnnxAgent(str(bare))


# -- parts: one tournament over several machines ---------------------------------------------

def _play_parts(tmp_path, count: int, config: Dict[str, Any]) -> List[str]:
    from tools.lib.parallel_tournament import run_shards, select_part, write_part

    shards = plan_shards([(0, 1)], games_per_side=4, shard_pairs=1)
    paths = []
    for i in range(count):
        played = run_shards(["random", "heuristic"], select_part(shards, i, count), 4, workers=1,
                            device="cpu", capture_games=True)
        path = str(tmp_path / f"part{i}.json")
        write_part(path, config, (i, count), played, 1.0)
        paths.append(path)
    return paths


_CONFIG: Dict[str, Any] = {"models": ["RandomBot", "HeuristicBot"], "matchups": [[0, 1]],
                           "games_per_side": 4, "shard_pairs": 1, "temperature": 0.1}


def test_pooled_parts_equal_one_machine_playing_every_shard(tmp_path) -> None:
    from tools.lib.parallel_tournament import merge_shards, pool_parts, write_games_log

    paths = _play_parts(tmp_path, 3, _CONFIG)
    config, shards, _ = pool_parts(paths)
    assert config == _CONFIG
    whole = run_matchups_parallel(["random", "heuristic"], [(0, 1)], games_per_side=4,
                                  workers=1, shard_pairs=1, device="cpu",
                                  log_games=str(tmp_path / "whole.jsonl"))
    _assert_same_result(merge_shards(shards)[(0, 1)], whole[(0, 1)])
    write_games_log(str(tmp_path / "pooled.jsonl"), shards)
    assert _log(str(tmp_path / "pooled.jsonl")) == _log(str(tmp_path / "whole.jsonl"))


def test_pooling_refuses_a_missing_or_repeated_shard_or_a_different_tournament(tmp_path) -> None:
    from tools.lib.parallel_tournament import pool_parts

    paths = _play_parts(tmp_path, 2, _CONFIG)
    with pytest.raises(ValueError, match="missing"):
        pool_parts(paths[:1])
    with pytest.raises(ValueError, match="more than one part"):
        pool_parts([paths[0], paths[0], paths[1]])
    (tmp_path / "other").mkdir()
    other = _play_parts(tmp_path / "other", 2, {**_CONFIG, "temperature": 1.0})
    with pytest.raises(ValueError, match="different tournament"):
        pool_parts([paths[0], other[1]])
