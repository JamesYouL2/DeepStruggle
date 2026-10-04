"""Whole-placement search (ai/search/placement_search.py): legal, planned, and honest.

The placement is chosen among complete placements sampled from the policy, compared by searching
the position each leaves; the rest of the winner is played from a plan. What must hold:

* every point played is legal, and the plan's points are actually used;
* placements identical as multisets are one candidate (the order of points gives the same board);
* honest search samples its world from the PLACER's side. After the last point the opponent
  usually moves, and the ordinary searcher would resample from there -- seeing the placer's
  opponent's real hand. So the worlds come from `determinize(..., placer)` and the searches inside
  them never resample.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

import ts_engine as ts
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search import placement_search as P
from ai.search.batched_mcts import BatchedMCTSAgent, BatchedMCTSConfig
from ai.search.pimcts import acting_player
from tools.lib.player_agent import load_agent


def _agent(k: int = 4, determinize: bool = True) -> BatchedMCTSAgent:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    cfg = BatchedMCTSConfig(simulations=2, temperature=0.0, auto_advance=True, advance_root=False,
                            determinize=determinize, node_filter="ops_influence",
                            placement_k=k, placement_samples=8)
    return BatchedMCTSAgent(m, device="cpu", config=cfg, featurise_capacity=256)


def _play(agent: BatchedMCTSAgent, n: int = 12, steps: int = 900, seed: int = 7) -> None:
    """Games that reach action rounds: the first legal action everywhere (an untrained network's
    own play ends them before its first placement), the agent at every influence point."""
    runner = ts.VectorizedBatchRunner(n, seed)
    for _ in range(steps):
        states = [runner.get_state(i) for i in range(n)]
        masks = np.asarray(runner.get_action_masks())
        acts = [int(np.flatnonzero(mk)[0]) if mk.any() else 211 for mk in masks]
        idx = [i for i, st in enumerate(states)
               if not ts.Engine.is_terminal(st) and P.is_influence_point(st)]
        if idx:
            for i, a in zip(idx, agent.select_actions_batch([states[i] for i in idx])):
                acts[i] = int(a)                       # select_actions_batch raises if illegal
        runner.step_flat_all(acts, True)


def test_placements_are_searched_and_their_plans_played() -> None:
    agent = _agent()
    _play(agent)
    pl = agent.placement
    assert pl is not None
    assert pl.placements_searched > 10
    assert pl.planned_count > 0                         # later points came from the plan
    assert getattr(agent, "world_mismatch_count", 0) == 0


def test_placements_in_any_order_are_one_candidate() -> None:
    pl = _agent().placement
    assert pl is not None
    rolls = [([b"a", b"b"], [5, 9]), ([b"c", b"d"], [9, 5]), ([b"e", b"f"], [5, 5]),
             ([b"g", b"h"], [9, 5])]
    cands = pl._candidates(rolls)
    assert [sorted(p) for _k, p in cands] == [[5, 9], [5, 5]]
    assert cands[0][1] == [5, 9]                         # the greedy rollout stays first


def test_worlds_are_sampled_from_the_placers_side_and_never_resampled(monkeypatch) -> None:
    agent = _agent()
    seen: List[tuple] = []
    real = P.determinize

    def spy(state: ts.GameState, me: ts.Player, rng):
        seen.append((int(me), int(acting_player(state))))
        return real(state, me, rng)

    monkeypatch.setattr(P, "determinize", spy)
    _play(agent)
    assert seen, "no placement was searched"
    assert all(me == placer for me, placer in seen)
    pl = agent.placement
    assert pl is not None and pl._subs
    assert all(not sub.cfg.determinize for sub in pl._subs.values())


def test_the_budget_is_at_most_the_point_searchs_total() -> None:
    """simulations x points of the greedy placement, split over the phases and survivors."""
    m = create_coldwar_net_v2("cpu")
    m.eval()
    cfg = BatchedMCTSConfig(simulations=16, temperature=0.0, auto_advance=True, determinize=True,
                            node_filter="ops_influence", placement_k=4, placement_samples=16)
    pl = BatchedMCTSAgent(m, device="cpu", config=cfg, featurise_capacity=256).placement
    assert pl is not None
    spent: List[int] = []
    cands_seen: List[list] = []
    orig_sub, orig_cands = pl._sub, pl._candidates

    def counting(sims: int):
        sub = orig_sub(sims)
        inner = sub._search

        def wrapped(states, keys=None):
            spent.append(sims * len(states))
            return inner(states, keys)
        sub._search = wrapped                            # type: ignore[method-assign]
        return sub

    def capture(rolls):
        out = orig_cands(rolls)
        cands_seen.append(out)
        return out

    pl._sub = counting                                   # type: ignore[method-assign]
    pl._candidates = capture                             # type: ignore[method-assign]
    runner = ts.VectorizedBatchRunner(32, 11)
    for _ in range(300):
        st = [runner.get_state(i) for i in range(32)]
        for s in st:
            if P.is_influence_point(s):
                spent.clear()
                cands_seen.clear()
                pl.choose([s])
                (cands,) = cands_seen
                if len(cands) > 1:
                    assert 0 < sum(spent) <= 16 * len(cands[0][1])
                    return
        masks = np.asarray(runner.get_action_masks())
        runner.step_flat_all([int(np.flatnonzero(mk)[0]) if mk.any() else 211 for mk in masks],
                             True)
    raise AssertionError("no influence placement with more than one candidate in 300 steps")


def test_the_spec_sets_and_names_it(tmp_path) -> None:
    m = create_coldwar_net_v2("cpu")
    path = tmp_path / "m.pt"
    torch.save(m.state_dict(), str(path))
    agent = load_agent(f"search:{path}:16:determinize:ops_influence:placement=6", device="cpu")
    assert isinstance(agent, BatchedMCTSAgent)
    assert agent.mcts.cfg.placement_k == 6 and agent.mcts.cfg.node_filter == "ops_influence"
    assert agent.name.endswith("-place6")
