"""A published export (.onnx) as a torch module (tools/lib/player_agent.OnnxModule): the searcher
and the leak probe run on it as on the checkpoint it was exported from, and an export in the
merged-influence view, which the searcher cannot build masks for, is refused."""
from __future__ import annotations

import json
from typing import List, Tuple

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.eval import gumbel_leaks as L
from ai.eval.target_forms import paired_advantage
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import OnnxAgent, OnnxModule, load_agent, load_network


def _export(tmp_path_factory: pytest.TempPathFactory, merged: bool) -> Tuple[str, str]:
    """An untrained V2 network and its tools/export_onnx.py export: (checkpoint, export)."""
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from tools.export_onnx import export

    run = tmp_path_factory.mktemp("onnx") / "W-01-01_20260101_000000"
    run.mkdir()
    torch.manual_seed(3)
    ckpt = run / "snapshot_1000000steps.pt"
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), ckpt)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": merged}))
    out = run / "model.onnx"
    export(str(ckpt), str(out), positions=32)
    return str(ckpt), str(out)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> Tuple[str, str]:
    return _export(tmp_path_factory, merged=False)


def _positions(n: int, seed: int = 11) -> Tuple[np.ndarray, np.ndarray]:
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


def test_the_module_is_the_exports_forward_and_the_checkpoints(exported: Tuple[str, str]) -> None:
    ckpt, onnx = exported
    agent = OnnxAgent(onnx)
    module = load_network(onnx)
    assert isinstance(module, OnnxModule)
    obs, masks = _positions(40)
    o, m = torch.from_numpy(obs), torch.from_numpy(masks)
    logits, v_win, v_vp = module(o, m)
    np.testing.assert_allclose(logits.numpy(), agent.logits(obs, masks), rtol=0, atol=1e-5)
    assert v_win.shape[0] == v_vp.shape[0] == len(obs)
    assert next(module.parameters()).device.type == "cpu"
    with torch.no_grad():
        t_logits, t_win, _ = load_network(ckpt)(o, m)
    legal = masks.astype(bool)
    np.testing.assert_allclose(logits.numpy()[legal], t_logits.numpy()[legal], rtol=0, atol=1e-3)
    np.testing.assert_allclose(v_win.numpy().reshape(-1), t_win.numpy().reshape(-1), rtol=0, atol=1e-4)


def test_search_plays_legal_moves_on_an_export(exported: Tuple[str, str]) -> None:
    searcher = load_agent(f"search:{exported[1]}:8:determinize:all:gumbel_k=4:gumbel_scale=0", device="cpu")
    s = ts.GameState()
    ts.Engine.init_game(s, 5)
    for _ in range(3):
        if ts.Engine.is_terminal(s):
            break
        player = s.ctx().decision_player
        a = searcher.select_action(s, player, 0.0)
        assert ActionEncoder.get_legal_mask(s)[a]
        ts.Engine.step_flat(s, a)


def test_the_leak_probe_on_an_export_reads_as_on_its_checkpoint(exported: Tuple[str, str]) -> None:
    """On the same positions: the same priors within the export's tolerance, the same root choices
    and the same paired-playout verdicts. (Positions are compared, not re-sampled: sampled games
    drift apart once a draw lands within the export's 1e-3 of a probability boundary.)"""
    import copy

    ckpt, onnx = exported
    pt, ox = load_network(ckpt), load_network(onnx)
    positions, _, _ = L.collect_positions(pt, 12, seed=4, games=8)
    twins = [copy.copy(p) for p in positions]
    for t, p in zip(twins, positions):
        t.state = p.state.clone()
    L._fill_network(ox, twins, 0, "cpu")
    for a, b in zip(positions, twins):
        assert a.legal == b.legal
        assert max(abs(a.prior[x] - b.prior[x]) for x in a.legal) < 1e-3
    spec = L.RootSpec(sims=8, k=4)
    picks_pt, _ = L.gumbel_choices(pt, positions, spec, seed=2)
    picks_ox, _ = L.gumbel_choices(ox, twins, spec, seed=2)
    assert sum(x == y for x, y in zip(picks_pt, picks_ox)) >= len(picks_pt) - 1
    items = [(p, p.legal[-1], p.legal[0]) for p in positions[:3]]
    assert paired_advantage(pt, items, 2, 5, batch=16) == paired_advantage(ox, items, 2, 5, batch=16)


def test_a_merged_view_export_is_refused(tmp_path_factory: pytest.TempPathFactory) -> None:
    _ckpt, onnx = _export(tmp_path_factory, merged=True)
    with pytest.raises(ValueError, match="merged-influence"):
        load_network(onnx)
