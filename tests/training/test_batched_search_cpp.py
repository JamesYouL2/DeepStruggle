"""The C++ search tree (`ts_engine.BatchedSearch`, backend "cpp") against the Python one.

The Python tree in ai/search/batched_mcts.py is the reference, and the C++ tree must reproduce it
bit for bit: the same actions at each root, the same visit counts, the same value sums, and the
same random.Random state afterwards -- the C++ tree draws each new child's chance seed from the
caller's own generator, in the Python tree's order. That pins the PUCT expression and its order of
evaluation, the settle depths, the US-side value convention, the treatment of terminal and
empty-mask leaves, the per-tree budgets, and the order of the random draws, so a game searched with
either tree is the same game.
"""

from __future__ import annotations

import random
from typing import Any, List

import numpy as np
import pytest
import torch

import ts_engine as ts
import ai.search.batched_mcts as bm
from ai.search.batched_mcts import GRAPH_BUCKET, BatchedMCTS, BatchedMCTSConfig, _GraphedNetwork
from ai.search.pimcts import acting_player
from bindings.action_encoder import ActionEncoder


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
@pytest.mark.parametrize("determinize", [False, True])
def test_the_cpp_tree_reproduces_the_python_tree(net: Any, auto_advance: bool, determinize: bool) -> None:
    roots = _positions(24)
    cfg = dict(simulations=48, c_puct=1.5, temperature=0.0, auto_advance=auto_advance,
               advance_root=False, determinize=determinize)
    ref = BatchedMCTS(net, device=torch.device("cpu"), config=BatchedMCTSConfig(**cfg, backend="python"),
                      featurise_capacity=64)
    cpp = BatchedMCTS(net, device=torch.device("cpu"), config=BatchedMCTSConfig(**cfg, backend="cpp"))
    for call in range(2):                                 # and again, from the advanced generators
        ref.reseed(77) if call == 0 else None
        cpp.reseed(77) if call == 0 else None
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
        assert ref._rng.getstate() == cpp._rng.getstate(), "the generators diverged"


def test_the_generator_is_cpythons(net: Any) -> None:
    """The C++ tree's MT19937 draws exactly random.Random.getrandbits(64): a one-tree search that
    creates k children leaves the generator where k getrandbits(64) calls leave it."""
    roots = [r for r in _positions(8) if not ts.Engine.is_terminal(r)][:1]
    m = BatchedMCTS(net, device=torch.device("cpu"),
                    config=BatchedMCTSConfig(simulations=20, temperature=0.0, backend="cpp"))
    m.reseed(3)
    before = m._rng.getstate()
    root = m._search(roots)[0]
    assert root is not None
    assert m._cpp is not None
    created = m._cpp[0].tree_size(0) - 1                   # every node but the root is a new child
    shadow = random.Random()
    shadow.setstate(before)
    for _ in range(created):
        shadow.getrandbits(64)
    assert shadow.getstate() == m._rng.getstate()


def test_cpp_search_is_reproducible_from_its_seed(net: Any) -> None:
    """Also across thread counts: seeds and leaf rows are numbered in tree order."""
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


def test_more_trees_than_the_searcher_holds(net: Any) -> None:
    """One searcher serves every batch up to its capacity; a call larger than it replaces it with
    a bigger one rather than truncating, and a smaller call after that reuses the bigger one."""
    roots = _positions(70)
    m = BatchedMCTS(net, device=torch.device("cpu"),
                    config=BatchedMCTSConfig(simulations=4, temperature=0.0, backend="cpp"))
    m._search(roots[:3])
    assert m._cpp is not None and m._cpp[0].capacity == 64
    out = m._search(roots)
    assert m._cpp is not None and m._cpp[0].capacity == 128
    assert len(out) == 70 and all(r is not None and (r.terminal or sum(r.n) == 4) for r in out)
    big = m._cpp[0]
    m._search(roots[:3])
    assert m._cpp[0] is big


def test_a_larger_searcher_searches_a_batch_as_a_fresh_one_does(net: Any) -> None:
    """Capacity only bounds how many trees a searcher holds: a small batch searched by one that
    has already served a large batch, with larger budgets, comes out as from a fresh searcher."""
    roots = _positions(24)
    cfg = BatchedMCTSConfig(simulations=16, temperature=0.0, determinize=True, backend="cpp")
    used = BatchedMCTS(net, device=torch.device("cpu"), config=cfg)
    used._search(_positions(150), simulations=[40] * 150)
    fresh = BatchedMCTS(net, device=torch.device("cpu"), config=cfg)
    used.reseed(9)
    fresh.reseed(9)
    a, b = used._search(roots), fresh._search(roots)
    assert [(r.actions, r.n) for r in a if r is not None] == [(r.actions, r.n) for r in b if r is not None]
    assert used._rng.getstate() == fresh._rng.getstate()


@pytest.mark.parametrize("determinize", [False, True])
def test_per_tree_budgets_reproduce_the_python_tree(net: Any, determinize: bool) -> None:
    """A budget per root (the Gumbel root's candidates): each tree stops at its own budget, and the
    C++ tree still draws the same seeds as the Python tree, in two groups as in one."""
    for n in (24, 140):
        roots = (_positions(24) * 6)[:n]
        budgets = [(3 * i) % 29 for i in range(n)]                     # 0 included
        cfg = dict(simulations=999, temperature=0.0, determinize=determinize)
        ref = BatchedMCTS(net, device=torch.device("cpu"),
                          config=BatchedMCTSConfig(**cfg, backend="python"), featurise_capacity=256)
        cpp = BatchedMCTS(net, device=torch.device("cpu"), config=BatchedMCTSConfig(**cfg, backend="cpp"))
        ref.reseed(13)
        cpp.reseed(13)
        a = ref._search(roots, simulations=budgets)
        b = cpp._search(roots, simulations=budgets)
        for ra, rb, budget in zip(a, b, budgets):
            assert ra is not None and rb is not None and ra.terminal == rb.terminal
            if not ra.terminal and ra.actions:
                assert ra.actions == rb.actions and ra.n == rb.n
                assert sum(rb.n) == budget
                np.testing.assert_allclose(ra.w, rb.w, rtol=0, atol=1e-9)
        assert ref._rng.getstate() == cpp._rng.getstate()


def test_a_budget_list_must_match_the_roots(net: Any) -> None:
    roots = _positions(4)
    m = BatchedMCTS(net, device=torch.device("cpu"),
                    config=BatchedMCTSConfig(simulations=4, temperature=0.0, backend="cpp"))
    with pytest.raises(ValueError):
        m._search(roots, simulations=[4, 4])


def test_an_unknown_backend_is_refused(net: Any) -> None:
    with pytest.raises(ValueError):
        BatchedMCTS(net, device=torch.device("cpu"), config=BatchedMCTSConfig(backend="rust"))


def test_the_grouping_rule() -> None:
    from ai.search.batched_mcts import _MIN_GROUP, search_groups
    assert search_groups(0) == []
    assert search_groups(1) == [(0, 1)]
    assert search_groups(2 * _MIN_GROUP - 1) == [(0, 2 * _MIN_GROUP - 1)]
    assert search_groups(2 * _MIN_GROUP) == [(0, _MIN_GROUP), (_MIN_GROUP, 2 * _MIN_GROUP)]
    assert search_groups(2 * _MIN_GROUP + 1) == [(0, _MIN_GROUP + 1), (_MIN_GROUP + 1, 2 * _MIN_GROUP + 1)]


def _net_on(device: str) -> Any:
    from ai.models.ladder_net import create_ladder_net
    torch.manual_seed(3)
    return create_ladder_net(
        torch.device(device), input_mode="grouped", aggregation="flatten", entity_dim=16,
        entity_proj_dim=64, card_self_attention=False, cross_attention=False, per_entity_heads=16,
        head_context=True, head_static=True, head_entities="country", head_center=True,
        identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
        card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
        categorical_value=False).eval()


@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")),
])
def test_two_groups_still_reproduce_the_python_tree(device: str, monkeypatch: Any) -> None:
    """From 2 x _MIN_GROUP trees on, the leaves go to the network one half at a time and the C++
    tree pipelines the halves (on CUDA through page-locked buffers and asynchronous copies). The
    Python tree evaluates the same halves, so the two must still agree exactly -- seeds included.
    The threshold is lowered here so 140 trees split; both trees read the one rule."""
    monkeypatch.setattr(bm, "_MIN_GROUP", 64)
    net = _net_on(device)
    roots = (_positions(24) * 6)[:140]
    cfg = dict(simulations=12, temperature=0.0, determinize=True)
    ref = BatchedMCTS(net, device=torch.device(device), config=BatchedMCTSConfig(**cfg, backend="python"),
                      featurise_capacity=256)
    cpp = BatchedMCTS(net, device=torch.device(device), config=BatchedMCTSConfig(**cfg, backend="cpp"))
    for call in range(2):
        if call == 0:
            ref.reseed(21)
            cpp.reseed(21)
        a = ref._search(roots)
        b = cpp._search(roots)
        assert cpp._cpp is not None and cpp._cpp[0].num_groups == 2
        for ra, rb in zip(a, b):
            assert ra is not None and rb is not None and ra.terminal == rb.terminal
            if not ra.terminal:
                assert ra.actions == rb.actions and ra.n == rb.n
                np.testing.assert_allclose(ra.w, rb.w, rtol=0, atol=1e-9)
        assert ref._rng.getstate() == cpp._rng.getstate()


_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _leaf_batch(n: int) -> Any:
    states = (_positions(24) * (n // 24 + 1))[:n]
    obs = torch.from_numpy(np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), 0),
                                                dtype=np.float32) for s in states])).pin_memory()
    masks = torch.from_numpy(np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8)
                                       for s in states])).pin_memory()
    return obs, masks


@_needs_gpu
def test_a_cuda_graph_at_one_batch_size_is_the_eager_network_bit_for_bit() -> None:
    """A graph replays the kernels eager torch launches: at a batch size that needs no padding, the
    probabilities and values are the eager ones exactly. (Padding changes the batch size, which is
    what makes BatchedMCTSConfig.cuda_graphs not bit-identical in general.)"""
    net = _net_on("cuda")
    obs, masks = _leaf_batch(2 * GRAPH_BUCKET)
    graphed = _GraphedNetwork(net, obs.shape[1], torch.device("cuda"))
    probs, values = (t.clone() for t in graphed.run(obs, masks))
    with torch.no_grad():
        logits, v, _ = net.forward(obs.cuda(), masks.cuda())
    assert torch.equal(probs, torch.softmax(logits.float(), dim=-1))
    assert torch.equal(values, v.float().reshape(-1))


@_needs_gpu
def test_padding_rows_do_not_reach_the_real_ones() -> None:
    """Rows are independent: whatever an earlier batch left in a graph's padding rows, the real
    rows come out the same."""
    net = _net_on("cuda")
    obs, masks = _leaf_batch(2 * GRAPH_BUCKET)
    graphed = _GraphedNetwork(net, obs.shape[1], torch.device("cuda"))
    k = 2 * GRAPH_BUCKET - 7
    first = [t.clone() for t in graphed.run(obs[:k], masks[:k])]
    graphed.run(obs[7:], masks[7:])
    second = [t.clone() for t in graphed.run(obs[:k], masks[:k])]
    assert all(torch.equal(a, b) for a, b in zip(first, second))
