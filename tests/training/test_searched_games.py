"""P33, the critic on searched games (--searched-games-workers): worker processes play whole games with the
rollout root on both sides and stream back each decision with the game's result from its decider's side;
the trainer keeps them in a replay store and trains the value head on them."""

from __future__ import annotations

import math
import time
from typing import Optional

import numpy as np
import torch

from ai.models.ladder_net import LadderNet, create_ladder_net
from ai.search.rollout_root import RolloutConfig
from ai.training.nash_pg import NashPGTrainer
from ai.training.searched_games import Rows, SearchedGames
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


def test_workers_play_whole_searched_games_and_label_each_decision_with_its_result() -> None:
    sg = SearchedGames(1, _net(), RolloutConfig(k=2, worlds=2, horizon=0, rule="argmax"), "cpu", envs=4)
    got: Optional[Rows] = None
    deadline = time.time() + 300
    try:
        while got is None and time.time() < deadline:
            time.sleep(2)
            got = sg.drain()
    finally:
        sg.close()
    assert got is not None, "no searched game finished in 5 minutes"
    obs, tgt, games = got
    assert games >= 1 and obs.shape[0] == tgt.shape[0] and obs.shape[0] >= 20
    assert set(np.unique(tgt).tolist()) <= {-1.0, 0.0, 1.0}
    assert not np.isnan(obs).any()


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
