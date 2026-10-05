"""First-play urgency and the Gumbel root for play.

FPU must mean the same in both trees (the C++ tree is held to the Python one), leave the search
unchanged at 0, and keep the budget on fewer moves as it grows. The Gumbel root must return legal
moves, take the network's top moves deterministically at scale 0 with k = 1 (so one candidate is the
argmax), spend at most its budget, and be settable from the search: spec.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSAgent, BatchedMCTSConfig
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder
from tools.lib.player_agent import load_agent


def _model():
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _states(n: int = 24, advance: int = 60) -> List[ts.GameState]:
    runner = ts.VectorizedBatchRunner(n, 4242)
    for _ in range(advance):
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(m)[0]) if m.any() else 211 for m in masks], True)
    return [runner.get_state(i) for i in range(n) if not ts.Engine.is_terminal(runner.get_state(i))]


@pytest.mark.parametrize("fpu", [0.2, 1.0])
def test_fpu_means_the_same_in_both_trees(fpu: float) -> None:
    states = _states()
    roots = {}
    for backend in ("python", "cpp"):
        cfg = BatchedMCTSConfig(simulations=16, backend=backend, seed=3, fpu_reduction=fpu)
        roots[backend] = BatchedMCTS(_model(), device="cpu", config=cfg,
                                     featurise_capacity=256)._search(states)
    for a, b in zip(roots["python"], roots["cpp"]):
        assert a is not None and b is not None and a.n == b.n


def test_a_larger_fpu_spreads_the_budget_over_fewer_moves() -> None:
    states = _states()
    tried = []
    for fpu in (0.0, 1.0):
        cfg = BatchedMCTSConfig(simulations=32, seed=3, fpu_reduction=fpu)
        roots = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)._search(states)
        tried.append(sum(sum(1 for x in r.n if x > 0) for r in roots if r is not None))
    assert tried[1] < tried[0]


def test_the_gumbel_root_plays_legal_moves_within_its_budget() -> None:
    states = _states()
    cfg = BatchedMCTSConfig(simulations=16, determinize=True, gumbel_k=4, seed=5)
    mcts = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)
    spent: List[int] = []
    picks = mcts.best_actions(states)
    gr = mcts._gumbel
    assert gr is not None
    orig = gr._sub

    def counting(sims: int):
        sub = orig(sims)
        inner = sub._search

        def wrapped(st, keys=None):
            spent.append(sims * len(st))
            return inner(st, keys)
        sub._search = wrapped                           # type: ignore[method-assign]
        return sub
    gr._sub = counting                                  # type: ignore[method-assign]
    picks = mcts.best_actions(states[:1])
    assert sum(spent) <= 16
    for st, a in zip(states, mcts.best_actions(states)):
        assert np.asarray(ActionEncoder.get_legal_mask(st))[a]


def test_one_candidate_without_noise_is_the_networks_top_move() -> None:
    states = _states()
    cfg = BatchedMCTSConfig(simulations=8, gumbel_k=1, gumbel_scale=0.0)
    picks = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256).best_actions(states)
    agent = BatchedMCTSAgent(_model(), device="cpu", config=BatchedMCTSConfig(), featurise_capacity=256)
    assert picks == agent._policy_actions(states)


def test_the_spec_sets_them(tmp_path) -> None:
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    agent = load_agent(f"search:{path}:32:determinize:all:gumbel_k=4:gumbel_scale=0:fpu_reduction=0.3",
                       device="cpu")
    assert isinstance(agent, BatchedMCTSAgent)
    cfg = agent.mcts.cfg
    assert (cfg.gumbel_k, cfg.gumbel_scale, cfg.fpu_reduction) == (4, 0.0, 0.3)
