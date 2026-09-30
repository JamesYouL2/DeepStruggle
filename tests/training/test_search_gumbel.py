"""Gumbel AlphaZero's root in BatchedMCTS (`gumbel_k`): Gumbel-top-k candidates, sequential halving.

The point of it is a small budget: PUCT at the root first visits a move of prior p after ~1/p
simulations, so it cannot overrule a confident network at 64. The halving visits every candidate
in its first phase and keeps the ones whose searched value is best. These tests pin the schedule,
the budget, and the choice; CPU only, on an untrained network where a full search is needed.
"""

from __future__ import annotations

import json
from typing import List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig, _BNode, _Halving
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle


def _model() -> torch.nn.Module:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    torch.manual_seed(0)
    return create_coldwar_net_v2(device="cpu", graph_layers=0).eval()


def _positions(n: int) -> List[ts.GameState]:
    """Decisions with plenty of legal moves, from a few plies into seeded games."""
    out: List[ts.GameState] = []
    for seed in range(8400, 8400 + 20 * n):
        s = ts.GameState()
        ts.Engine.init_game(s, seed)
        settle(s, SettleMode.FORCED)
        for _ in range(seed % 30):
            if ts.Engine.is_terminal(s):
                break
            legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
            ts.Engine.step_flat(s, int(legal[seed % len(legal)]))
            settle(s, SettleMode.FORCED)
        if (not ts.Engine.is_terminal(s)
                and np.asarray(ActionEncoder.get_legal_mask(s)).sum() >= 8):
            out.append(s)
        if len(out) == n:
            break
    return out


def _search(sims: int, **kw) -> List[_BNode]:
    mcts = BatchedMCTS(_model(), device=torch.device("cpu"), config=BatchedMCTSConfig(
        simulations=sims, auto_advance=True, advance_root=False, **kw))
    roots = mcts._search(_positions(4))
    return [r for r in roots if r is not None]


def _root(mover: ts.Player, priors: List[float], q_mover: List[float]) -> _BNode:
    """A root whose children already hold one visit of the given value (mover's side)."""
    sign = 1.0 if mover == ts.Player.US else -1.0
    return _BNode(state=ts.GameState(), mover=int(mover), terminal=False, expanded=True,
                  actions=list(range(len(priors))), priors=priors,
                  n=[1.0] * len(priors), w=[sign * q for q in q_mover])


def test_the_budget_is_spent_exactly_and_every_candidate_is_tried() -> None:
    for root in _search(64, gumbel_k=8):
        assert sum(root.n) == 64
        tried = sum(1 for x in root.n if x > 0)
        assert tried >= min(8, len(root.actions))
        assert root.gumbel_pick is not None and root.n[root.gumbel_pick] > 0


def test_halving_keeps_the_better_valued_candidate_over_a_higher_prior() -> None:
    # Equal Gumbel draws (scale 0): the prior favours move 0, the searched value move 2.
    for mover in (ts.Player.US, ts.Player.USSR):
        root = _root(mover, priors=[0.6, 0.3, 0.1], q_mover=[-0.5, 0.0, 0.9])
        cfg = BatchedMCTSConfig(gumbel_k=3, gumbel_scale=0.0)
        h = _Halving(root, cfg, budget=12, rng=np.random.RandomState(0))
        assert h.pick() == 2


def test_the_schedule_halves_down_to_one_survivor() -> None:
    root = _root(ts.Player.US, priors=[0.125] * 8, q_mover=[0.1 * i for i in range(8)])
    cfg = BatchedMCTSConfig(gumbel_k=8, gumbel_scale=0.0)
    h = _Halving(root, cfg, budget=48, rng=np.random.RandomState(0))
    seen = [h.next_action() for _ in range(48)]
    assert h.phases == 3
    # 48 // (3 phases x 8 candidates) = 2 visits each in phase 1.
    assert sorted(seen[:16]) == sorted(list(range(8)) * 2), "phase 1 visits every candidate"
    # 8 -> 4 -> 2 across the phases; the last halving, 2 -> 1, is the pick itself.
    assert sorted(h.alive) == [6, 7] and h.pick() == 7


def test_scale_zero_with_one_candidate_is_the_network_argmax() -> None:
    for root in _search(16, gumbel_k=1, gumbel_scale=0.0):
        pick = root.gumbel_pick
        assert pick is not None and pick == int(np.argmax(root.priors))
        assert root.n[pick] == 16


def test_gumbel_turns_subtree_reuse_and_root_noise_off() -> None:
    mcts = BatchedMCTS(_model(), device=torch.device("cpu"), config=BatchedMCTSConfig(
        simulations=8, auto_advance=True, advance_root=False, reuse_subtree=True,
        dirichlet_frac=0.5, gumbel_k=4, gumbel_scale=0.0))
    s = _positions(1)
    root = mcts._search(s, keys=[0])[0]
    assert not mcts._trees
    probe = BatchedMCTS(_model(), device=torch.device("cpu"), config=BatchedMCTSConfig(
        simulations=1, auto_advance=True, advance_root=False))._search(s)[0]
    assert root is not None and probe is not None
    assert root.priors == pytest.approx(probe.priors), "no Dirichlet mixed into the priors"


def test_the_search_spec_enables_it_and_names_it(tmp_path) -> None:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    from ai.search.batched_mcts import BatchedMCTSAgent
    from tools.lib.player_agent import load_agent

    run = tmp_path / "T-01-01_20260101_000000"
    run.mkdir()
    ckpt = run / "snapshot_1000000steps.pt"
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), ckpt)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))
    agent = load_agent(f"search:{ckpt}:32:gumbel_k=16", device="cpu")
    assert isinstance(agent, BatchedMCTSAgent)
    assert agent.mcts.cfg.gumbel_k == 16
    assert agent.name.endswith("search32-gumbel_k16")
    state = _positions(1)[0]
    action = agent.select_action(state, None)
    assert np.asarray(ActionEncoder.get_legal_mask(state))[action]
