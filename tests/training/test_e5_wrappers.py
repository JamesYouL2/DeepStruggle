"""Wrappers that play a published export without training it: search over an ONNX network, a
policy ensemble, and the certain-win / certain-loss safety layer."""
from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from bindings.action_encoder import ActionEncoder
from tools.lib.batch_tournament import BatchMatchRunner
from tools.lib.player_agent import (EnsembleAgent, HeuristicAgent, OnnxAgent, OnnxModule,
                                    load_agent)
from tools.lib.safety import SafetyAgent


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> str:
    """An untrained V2 network exported by tools/export_onnx.py."""
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from tools.export_onnx import export

    run = tmp_path_factory.mktemp("onnx") / "W-01-01_20260101_000000"
    run.mkdir()
    torch.manual_seed(3)
    ckpt = run / "snapshot_1000000steps.pt"
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), ckpt)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))
    out = run / "model.onnx"
    export(str(ckpt), str(out), positions=32)
    return str(out)


def _positions(n: int, seed: int = 11) -> "tuple[np.ndarray, np.ndarray]":
    import random

    from tools.lib.game_step import drain_chance

    rng = random.Random(seed)
    obs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    s = ts.GameState()
    ts.Engine.init_game(s, seed)
    drain_chance(s)
    while not ts.Engine.is_terminal(s) and len(obs) < n:
        p = s.ctx().decision_player
        if p != ts.Player.NONE:
            obs.append(np.asarray(ts.extract_observation(s, p), dtype=np.float32))
            masks.append(np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8))
        ts.Engine.step_flat(s, rng.choice(ActionEncoder.get_legal_indices(s)))
        drain_chance(s)
    return np.stack(obs), np.stack(masks)


def test_onnx_module_is_the_exports_forward(exported: str) -> None:
    agent = OnnxAgent(exported)
    module = agent.as_module()
    obs, masks = _positions(40)
    logits, v_win, v_vp = module(torch.from_numpy(obs), torch.from_numpy(masks))
    np.testing.assert_allclose(logits.numpy(), agent.logits(obs, masks), rtol=0, atol=1e-5)
    assert v_win.shape[0] == v_vp.shape[0] == len(obs)
    assert next(module.parameters()).device.type == "cpu"


def test_search_runs_on_an_export_and_plays_legal_games(exported: str) -> None:
    searcher = load_agent(f"search:{exported}:4:determinize", device="cpu")
    assert searcher.name.endswith("+search4-det")
    res = BatchMatchRunner.play_parallel_matchup(searcher, HeuristicAgent(), games_per_side=1,
                                                 device="cpu", temperature=0.0)
    assert res["total_games"] == 2


def test_an_ensemble_of_one_network_twice_plays_as_that_network(exported: str) -> None:
    kw: Dict[str, Any] = dict(games_per_side=2, device="cpu", temperature=0.0)
    single = BatchMatchRunner.play_parallel_matchup(OnnxAgent(exported), HeuristicAgent(), **kw)
    pair = BatchMatchRunner.play_parallel_matchup(
        EnsembleAgent([OnnxAgent(exported), OnnxAgent(exported)]), HeuristicAgent(), **kw)
    for k in ("a_wins", "b_wins", "draws", "avg_steps", "avg_vp_margin_a"):
        assert pair[k] == single[k], k


def test_ensemble_sampling_only_picks_legal_actions(exported: str) -> None:
    ens = EnsembleAgent([OnnxAgent(exported), OnnxAgent(exported)])
    obs, masks = _positions(120)
    for t in (0.1, 1.0, 5.0):
        picks = ens.act_batch(obs, masks, t, greedy=False)
        assert masks[np.arange(len(picks)), picks].all(), t


class _AlwaysOps:
    """A base agent that plays its card for Ops whenever it can."""

    name = "ops"

    def select_action(self, state: ts.GameState, player: ts.Player, temperature: float = 0.1) -> int:
        mask = np.asarray(ActionEncoder.get_legal_mask(state))
        ops = ActionEncoder.PLAY_MODE_OFFSET + 2
        return ops if mask[ops] else int(np.flatnonzero(mask)[0])


def test_the_safety_layer_refuses_a_certain_loss() -> None:
    """At DEFCON 2 the USSR playing Duck and Cover for Ops fires the US Event, taking DEFCON to
    1 in the USSR's own round: a certain loss. The layer must refuse it while Space is legal."""
    from tests.training.test_doctrine import _to_first_action_round

    state = _to_first_action_round(101)
    me = state.ctx().decision_player
    assert me == ts.Player.USSR
    duck = 4
    state.set_card_location(duck, ts.hand_of(me))
    state.defcon = 2
    state.us_space_track = state.ussr_space_track = 0
    state.set_space_turns_used(me, 0)
    assert ts.Engine.try_step_flat(state, duck - 1)
    assert state.ctx().decision_type == ts.DecisionType.SELECT_PLAY_MODE

    base = _AlwaysOps()
    safe = SafetyAgent(base)
    choice = safe.select_action(state, me)
    assert choice != base.select_action(state, me)
    assert safe.stats["refused_loss"] == 1
