"""MCTS truncated at half action rounds (`BatchedMCTSConfig.truncate_half_rounds`).

A line is cut once it has crossed N half-round boundaries: the first node past the Nth is valued
by the network and never expanded. These tests walk the finished trees and check the cut is where
it says, that a cut leaf stays a leaf however many simulations reach it, and that 0 leaves the
search unlimited. CPU only, on an untrained network: the tests are about the tree's shape, not
the moves.
"""

from __future__ import annotations

import json
from typing import Iterator, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig, _BNode, _half_round
from bindings.action_encoder import ActionEncoder
from bindings.settle import SettleMode, settle


def _model() -> torch.nn.Module:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    torch.manual_seed(0)
    return create_coldwar_net_v2(device="cpu", graph_layers=0).eval()


def _round_starts(n: int) -> List[ts.GameState]:
    """Positions at the first decision of an action round, where a line soon leaves the round."""
    out: List[ts.GameState] = []
    for seed in range(8300, 8300 + 10 * n):
        s = ts.GameState()
        ts.Engine.init_game(s, seed)
        settle(s, SettleMode.FORCED)
        while not ts.Engine.is_terminal(s) and s.current_phase != ts.Phase.ACTION_ROUND:
            legal = np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(s)))
            ts.Engine.step_flat(s, int(legal[seed % len(legal)]))
            settle(s, SettleMode.FORCED)
        if not ts.Engine.is_terminal(s):
            out.append(s)
        if len(out) == n:
            break
    return out


def _search(horizon: int, sims: int = 48) -> List[_BNode]:
    mcts = BatchedMCTS(_model(), device=torch.device("cpu"), config=BatchedMCTSConfig(
        simulations=sims, determinize=False, auto_advance=True, advance_root=False,
        truncate_half_rounds=horizon))
    roots = mcts._search(_round_starts(3))
    assert all(r is not None for r in roots)
    return [r for r in roots if r is not None]


def _nodes(root: _BNode) -> Iterator[_BNode]:
    stack = [root]
    while stack:
        nd = stack.pop()
        yield nd
        stack.extend(nd.children.values())


@pytest.mark.parametrize("horizon", [1, 2])
def test_no_line_is_expanded_past_the_horizon(horizon: int) -> None:
    cut = 0
    for root in _search(horizon):
        for nd in _nodes(root):
            assert nd.half_rounds <= horizon
            if nd.cutoff:
                cut += 1
                assert nd.half_rounds == horizon and not nd.children
            elif not nd.terminal:
                assert nd.half_rounds < horizon
    assert cut > 0, "the positions must reach the horizon, or this checks nothing"


def test_one_half_round_keeps_every_expanded_node_in_the_root_round() -> None:
    for root in _search(1):
        key = _half_round(root.state)
        for nd in _nodes(root):
            if nd.children:
                assert _half_round(nd.state) == key


def test_a_cut_leaf_stays_a_leaf_however_often_it_is_visited() -> None:
    revisited = 0
    for root in _search(1, sims=96):
        for nd in _nodes(root):
            for action, child in nd.children.items():
                if child.cutoff:
                    visits = nd.n[nd.actions.index(action)]
                    assert not child.children
                    revisited += int(visits > 1)
    assert revisited > 0, "some cut leaf must be visited more than once"


def test_zero_is_unlimited() -> None:
    crossed = 0
    for root in _search(0, sims=96):
        for nd in _nodes(root):
            assert not nd.cutoff and nd.half_rounds == 0
            crossed += int(bool(nd.children) and _half_round(nd.state) != _half_round(root.state))
    assert crossed > 0, "an unlimited search expands past the root's half round"


def test_truncation_turns_subtree_reuse_off() -> None:
    mcts = BatchedMCTS(_model(), device=torch.device("cpu"), config=BatchedMCTSConfig(
        simulations=8, auto_advance=True, advance_root=False, reuse_subtree=True,
        truncate_half_rounds=1))
    states = _round_starts(1)
    mcts._search(states, keys=[0])
    assert not mcts._trees


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    run = tmp_path_factory.mktemp("ckpt") / "T-01-01_20260101_000000"
    run.mkdir()
    torch.manual_seed(7)
    ckpt = run / "snapshot_1000000steps.pt"
    torch.save(create_coldwar_net_v2(torch.device("cpu")).state_dict(), ckpt)
    (run / "metadata.json").write_text(json.dumps({"merged_influence": False}))
    return str(ckpt)


def test_the_search_spec_sets_the_horizon_and_names_it(checkpoint: str) -> None:
    from ai.search.batched_mcts import BatchedMCTSAgent
    from tools.lib.player_agent import load_agent

    agent = load_agent(f"search:{checkpoint}:8::::truncate_half_rounds=1", device="cpu")
    assert isinstance(agent, BatchedMCTSAgent)
    assert agent.mcts.cfg.truncate_half_rounds == 1
    assert agent.name.endswith("search8-trunc1")
    # Positional options still read as before, alongside a named one.
    agent = load_agent(f"search:{checkpoint}:16:determinize:truncate_half_rounds=2", device="cpu")
    assert isinstance(agent, BatchedMCTSAgent)
    assert agent.mcts.cfg.determinize and agent.mcts.cfg.truncate_half_rounds == 2
    assert agent.name.endswith("search16-det-trunc2")


def test_the_search_spec_refuses_what_it_cannot_read(checkpoint: str) -> None:
    from tools.lib.player_agent import load_agent

    for opt in ("truncate_half_round=1", "truncate_half_rounds=one", "determinize=maybe"):
        with pytest.raises(ValueError):
            load_agent(f"search:{checkpoint}:8:{opt}", device="cpu")
