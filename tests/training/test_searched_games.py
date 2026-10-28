"""P33, the critic on searched games (--searched-games-workers): worker processes play whole games with the
rollout root on both sides and stream back each decision with the game's result from its decider's side;
the trainer keeps them in a replay store and trains the value head on them."""

from __future__ import annotations

import io
import math
import queue
from typing import Any, Optional, cast

import numpy as np
import pytest
import torch

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.search.rollout_root import RolloutConfig
from ai.training.nash_pg import NashPGTrainer
from ai.training.search_workers import WorkerRolloutConfig
from ai.training.searched_games import Rows, SearchedGames, _worker_main
from bindings.ts_env import TsVectorizedEnv


def _net() -> LadderNet:
    torch.manual_seed(0)
    return create_ladder_net("cpu", input_mode="grouped", aggregation="flatten", entity_dim=16,
                             entity_proj_dim=32, card_self_attention=False, cross_attention=False,
                             per_entity_heads=16, head_context=True, head_static=True,
                             head_entities="country", head_center=True, identity_dim=0,
                             drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
                             card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                             card_lookup_identity_dim=0, categorical_value=False).eval()


class _StopAfterFirstResult:
    """The worker's control queue: stop (None) once a finished game has been put on `results`."""

    def __init__(self, results: "queue.Queue[Any]") -> None:
        self.results = results

    def get_nowait(self) -> Optional[bytes]:
        if not self.results.empty():
            return None
        raise queue.Empty


def test_workers_play_whole_searched_games_and_label_each_decision_with_its_result() -> None:
    """The worker loop, driven in this process: it plays until its first game ends, then stops.

    This used to spawn a worker and poll `drain()` every 2 s for up to 300 s of wall clock. The
    loop is the same function either way; run synchronously it is deterministic (fixed seed, one
    env) and ends when the game does, not when a deadline passes.
    """
    results: "queue.Queue[Any]" = queue.Queue()
    buf = io.BytesIO()
    torch.save(_net(), buf)
    cfg = WorkerRolloutConfig.of(RolloutConfig(k=2, worlds=2, horizon=0, rule="argmax"))
    threads = torch.get_num_threads()                   # the worker sets its own; give ours back
    try:
        _worker_main(_StopAfterFirstResult(results), results, buf.getvalue(), cfg, 1, "cpu", 24680)
    finally:
        torch.set_num_threads(threads)

    kind, payload = results.get_nowait()
    assert kind == "ok", payload
    assert results.empty(), "the worker stops at its control queue's None"
    obs, tgt, games = payload
    assert games == 1 and obs.shape[0] == tgt.shape[0] and obs.shape[0] >= 20
    assert set(np.unique(tgt).tolist()) <= {-1.0, 0.0, 1.0}
    assert not np.isnan(obs).any()


def _pool_without_processes() -> SearchedGames:
    """A SearchedGames whose result queue the test fills itself -- `drain`'s side of the pipe."""
    sg = SearchedGames.__new__(SearchedGames)
    sg._results = cast(Any, queue.Queue())   # a thread queue: put() is visible to get_nowait() at once
    sg._control, sg._procs, sg.games = [], [], 0
    return sg


def test_drain_concatenates_what_the_workers_sent_and_raises_on_a_worker_error() -> None:
    sg = _pool_without_processes()
    assert sg.drain() is None
    a = (np.ones((3, 5), np.float32), np.ones(3, np.float32), 1)
    b = (np.zeros((2, 5), np.float32), -np.ones(2, np.float32), 2)
    sg._results.put(("ok", a))
    sg._results.put(("ok", b))
    got = sg.drain()
    assert got is not None
    obs, tgt, games = got
    assert obs.shape == (5, 5) and tgt.tolist() == [1, 1, 1, -1, -1] and games == 3 and sg.games == 3
    assert sg.drain() is None
    sg._results.put(("error", "Traceback: boom"))
    with pytest.raises(RuntimeError, match="boom"):
        sg.drain()


class _FakeGames:
    """Stands in for the worker pool: one batch of rows, then nothing."""

    def __init__(self, width: int) -> None:
        rng = np.random.default_rng(1)
        self.rows: Optional[Rows] = (rng.random((300, width)).astype(np.float32),
                                     np.where(rng.random(300) < 0.5, 1.0, -1.0).astype(np.float32), 3)
        self.games = 0

    def drain(self) -> Optional[Rows]:
        rows, self.rows = self.rows, None
        if rows is not None:
            self.games += rows[2]
        return rows

    def refresh(self, net: object) -> None:
        pass


def test_the_trainer_stores_searched_rows_and_trains_the_value_head_on_them() -> None:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    t = NashPGTrainer(active_net=_net(), env=TsVectorizedEnv(num_envs=4, base_seed=7), num_envs=4,
                      buffer_size=32, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False, device=dev,
                      searched_games_workers=1, searched_value_coef=1.0, searched_value_buffer=256,
                      searched_value_batch=64)
    t._sg = _FakeGames(t.buffer.obs_dim)  # type: ignore[assignment]
    t.collect_rollouts()
    m = t.train_step()
    assert m["searched_rows_added"] == 300 and m["searched_rows"] == 256       # the store wraps at its size
    assert m["searched_games"] == 3 and math.isfinite(m["searched_value_loss"]) and m["searched_value_loss"] > 0
