"""Search knobs for low-budget tuning: prior temperature, the final-move rule, and the search: spec's
`field=value` options.

What must hold: the two backends still agree exactly with the knobs on (the C++ tree is tested
against the Python one throughout); a temperature above 1 flattens the priors search starts from;
"value" picks the best mean value among sufficiently visited moves; and the spec sets any
BatchedMCTSConfig field by name, records it in the entrant's name, and refuses what it cannot set.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSAgent, BatchedMCTSConfig, _BNode, _temper
from tools.lib.player_agent import _search_overrides, load_agent


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


def _entropy(p: List[float]) -> float:
    return -sum(x * math.log(x) for x in p if x > 0)


@pytest.mark.parametrize("knobs", [dict(prior_temp=1.7), dict(root_prior_temp=2.0),
                                   dict(prior_temp=0.8, root_prior_temp=1.5, c_puct=2.5)])
def test_the_backends_agree_with_the_knobs_on(knobs: Dict[str, Any]) -> None:
    states = _states()
    roots = {}
    for backend in ("python", "cpp"):
        cfg = BatchedMCTSConfig(simulations=12, auto_advance=True, backend=backend, seed=9, **knobs)
        mcts = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)
        roots[backend] = mcts._search(states)
    for a, b in zip(roots["python"], roots["cpp"]):
        assert a is not None and b is not None
        assert a.actions == b.actions and a.n == b.n
        np.testing.assert_allclose(a.priors, b.priors, rtol=0, atol=1e-6)


def test_a_temperature_above_one_flattens_the_root_priors() -> None:
    states = _states()
    base = BatchedMCTS(_model(), device="cpu", config=BatchedMCTSConfig(simulations=2),
                       featurise_capacity=256)._search(states)
    for knob in ("prior_temp", "root_prior_temp"):
        cfg = (BatchedMCTSConfig(simulations=2, prior_temp=2.0) if knob == "prior_temp"
               else BatchedMCTSConfig(simulations=2, root_prior_temp=2.0))
        hot = BatchedMCTS(_model(), device="cpu", config=cfg, featurise_capacity=256)._search(states)
        flatter = 0
        for r0, r1 in zip(base, hot):
            assert r0 is not None and r1 is not None
            if len(r0.priors) > 1:
                assert _entropy(r1.priors) >= _entropy(r0.priors) - 1e-9
                flatter += _entropy(r1.priors) > _entropy(r0.priors) + 1e-6
            if knob == "root_prior_temp":
                np.testing.assert_allclose(r1.priors, _temper(r0.priors, 2.0), rtol=0, atol=1e-9)
        assert flatter > 0


def test_the_value_rule_picks_the_best_mean_among_visited_moves() -> None:
    m = BatchedMCTS(_model(), device="cpu",
                    config=BatchedMCTSConfig(final_rule="value", value_min_visits=2))
    # US to move: action 7 has the most visits, 9 the best mean with enough visits, 3 the best
    # mean on a single visit (below the floor).
    root = _BNode(state=ts.GameState(), mover=int(ts.Player.US), terminal=False, actions=[3, 7, 9],
                  priors=[0.2, 0.5, 0.3], n=[1.0, 8.0, 3.0], w=[0.9, 1.6, 1.5], expanded=True,
                  total=12.0)
    assert root.actions[m._choose(root)] == 9
    root.mover = int(ts.Player.USSR)                       # the USSR wants the lowest US value
    assert root.actions[m._choose(root)] == 7
    m.cfg.final_rule = "visits"
    assert root.actions[m._choose(root)] == 7


def test_the_spec_sets_fields_names_them_and_refuses_the_rest(tmp_path) -> None:
    path = tmp_path / "m.pt"
    torch.save(_model().state_dict(), str(path))
    agent = load_agent(f"search:{path}:32:determinize:all:c_puct=2.5:prior_temp=1.5:"
                       f"final_rule=value:placement=4", device="cpu")
    assert isinstance(agent, BatchedMCTSAgent)
    cfg = agent.mcts.cfg
    assert (cfg.simulations, cfg.determinize, cfg.c_puct, cfg.prior_temp, cfg.final_rule,
            cfg.placement_k) == (32, True, 2.5, 1.5, "value", 4)
    assert agent.name == "search32-det-c_puct2.5-prior_temp1.5-final_rulevalue-placement_k4"
    for bad in ("cpuct=1", "c_puct=high", "simulations=5", "determinize=1"):
        with pytest.raises(ValueError):
            _search_overrides([bad])
