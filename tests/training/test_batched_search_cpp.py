"""The C++ search tree (`ts_engine.BatchedSearch`, backend "cpp") against the Python one.

The Python tree in ai/search/batched_mcts.py is the reference. Given the same network, the same
roots and the same chance seeds, the C++ tree must make every decision the Python one makes: the
same actions at each root, the same visit counts and the same value sums. That pins the PUCT
expression and its order of evaluation, the settle depths, the US-side value convention, the
treatment of terminal and empty-mask leaves, and the per-tree budgets.

Chance is the one deliberate difference. The C++ tree draws each child's seed from a SplitMix64
stream per tree (seeded from the call's seed and the tree's index), so a search is reproducible
whatever the thread count. Here the Python reference is given those same streams, so the
comparison is exact.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig, _BNode, settle

_M64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15


def _splitmix(state: int) -> Tuple[int, int]:
    """(new state, output) -- ts::Prng::next_u64."""
    state = (state + _GOLDEN) & _M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return state, z ^ (z >> 31)


class _PythonWithCppChance(BatchedMCTS):
    """The Python tree, drawing each child's chance seed from the C++ tree's per-tree stream."""

    def _search(self, states: Sequence[ts.GameState], keys: Any = None) -> List[Any]:
        self._seed = self._rng.getrandbits(64)            # where the C++ path draws its seed
        self._streams: Dict[int, int] = {}
        return super()._search(states, keys)

    def _evaluate_batch(self, nodes: Sequence[_BNode]) -> None:
        if self._root_nodes and not self._streams:
            for i, r in enumerate(self._root_nodes):
                s = (self._seed + _GOLDEN * (i + 1)) & _M64
                _s, out = _splitmix(s)
                self._streams[id(r)] = out
        super()._evaluate_batch(nodes)

    def _descend(self, root: _BNode) -> Tuple[List[Tuple[_BNode, int]], _BNode]:
        path: List[Tuple[_BNode, int]] = []
        node = root
        while True:
            if node.terminal or not node.expanded:
                return path, node
            idx = self._select(node)
            path.append((node, idx))
            action = node.actions[idx]
            child = node.children.get(action)
            if child is None:
                nxt = node.state.clone()
                self._streams[id(root)], nxt.rng_state = _splitmix(self._streams[id(root)])
                ts.Engine.step_flat(nxt, action)
                settle(nxt, self.cfg.auto_advance)
                child = self._make_node(nxt)
                node.children[action] = child
                return path, child
            node = child


def _positions(n: int) -> List[ts.GameState]:
    """Positions from random play, spread over the game, with die rolls ahead of most of them."""
    rng = np.random.default_rng(11)
    out: List[ts.GameState] = []
    for i in range(n):
        s = ts.GameState()
        ts.Engine.init_game(s, 900 + i)
        for _ in range(int(rng.integers(5, 90))):
            if ts.Engine.is_terminal(s):
                break
            legal = np.flatnonzero(ts.Engine.get_flat_action_mask(s))
            if len(legal) == 0:
                break
            ts.Engine.step_flat(s, int(rng.choice(legal)))
            ts.Engine.auto_advance_step(s)
        out.append(s)
    return out


@pytest.fixture(scope="module")
def net() -> Any:
    from ai.models.ladder_net import create_ladder_net
    torch.manual_seed(3)
    return create_ladder_net(
        torch.device("cpu"), input_mode="grouped", aggregation="flatten", entity_dim=16,
        entity_proj_dim=64, card_self_attention=False, cross_attention=False, per_entity_heads=16,
        head_context=True, head_static=True, head_entities="country", head_center=True,
        identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
        card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
        categorical_value=False).eval()


@pytest.mark.parametrize("auto_advance", [True, False])
def test_the_cpp_tree_makes_every_decision_the_python_tree_makes(net: Any, auto_advance: bool) -> None:
    roots = _positions(24)
    cfg = dict(simulations=48, c_puct=1.5, temperature=0.0, auto_advance=auto_advance,
               advance_root=False, determinize=False)
    ref = _PythonWithCppChance(net, device=torch.device("cpu"),
                               config=BatchedMCTSConfig(**cfg, backend="python"),
                               featurise_capacity=64)
    cpp = BatchedMCTS(net, device=torch.device("cpu"), config=BatchedMCTSConfig(**cfg, backend="cpp"))
    ref.reseed(77)
    cpp.reseed(77)
    a = ref._search(roots)
    b = cpp._search(roots)
    searched = 0
    for ra, rb in zip(a, b):
        assert ra is not None and rb is not None
        assert ra.terminal == rb.terminal
        if ra.terminal:
            continue
        searched += 1
        assert ra.actions == rb.actions
        assert ra.n == rb.n, "visit counts differ"
        np.testing.assert_allclose(ra.w, rb.w, rtol=0, atol=1e-9)
        np.testing.assert_allclose(ra.priors, rb.priors, rtol=0, atol=1e-12)
        assert ra.value_us == pytest.approx(rb.value_us, abs=1e-7)
        assert sum(rb.n) == 48
    assert searched >= 8


def test_cpp_search_is_reproducible_from_its_seed(net: Any) -> None:
    roots = _positions(6)
    cfg = BatchedMCTSConfig(simulations=32, temperature=0.0, determinize=True, backend="cpp")
    runs = []
    for _ in range(2):
        m = BatchedMCTS(net, device=torch.device("cpu"), config=cfg)
        m.reseed(5)
        runs.append([(r.actions, r.n) for r in m._search(roots) if r is not None])
    assert runs[0] == runs[1]


def test_the_cpp_agent_plays_legal_actions(net: Any) -> None:
    from ai.search.batched_mcts import BatchedMCTSAgent
    roots = [r for r in _positions(10) if not ts.Engine.is_terminal(r)]
    agent = BatchedMCTSAgent(net, device=torch.device("cpu"),
                             config=BatchedMCTSConfig(simulations=16, temperature=0.0,
                                                      determinize=True, backend="cpp"))
    for st, a in zip(roots, agent.select_actions_batch(roots)):
        assert ts.Engine.get_flat_action_mask(st)[a]


def test_more_trees_than_any_one_bucket_holds(net: Any) -> None:
    """Searchers are cached per power-of-two size; a call larger than every cached one builds a
    bigger searcher rather than truncating."""
    roots = _positions(70)
    m = BatchedMCTS(net, device=torch.device("cpu"),
                    config=BatchedMCTSConfig(simulations=4, temperature=0.0, backend="cpp"))
    m._search(roots[:3])
    out = m._search(roots)
    assert len(out) == 70 and sorted(m._cpp) == [64, 128]
    assert all(r is not None and (r.terminal or sum(r.n) == 4) for r in out)


def test_an_unknown_backend_is_refused(net: Any) -> None:
    with pytest.raises(ValueError):
        BatchedMCTS(net, device=torch.device("cpu"), config=BatchedMCTSConfig(backend="rust"))
