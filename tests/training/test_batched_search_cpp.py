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
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig


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
    created = m._cpp[64].tree_size(0) - 1                  # every node but the root is a new child
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


def test_the_grouping_rule() -> None:
    from ai.search.batched_mcts import search_groups
    assert search_groups(0) == []
    assert search_groups(1) == [(0, 1)]
    assert search_groups(127) == [(0, 127)]
    assert search_groups(128) == [(0, 64), (64, 128)]
    assert search_groups(201) == [(0, 101), (101, 201)]


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
def test_two_groups_still_reproduce_the_python_tree(device: str) -> None:
    """From 128 trees on, the leaves go to the network one half at a time and the C++ tree
    pipelines the halves (on CUDA through page-locked buffers and asynchronous copies). The Python
    tree evaluates the same halves, so the two must still agree exactly -- seeds included."""
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
        assert cpp._cpp[256].num_groups == 2
        for ra, rb in zip(a, b):
            assert ra is not None and rb is not None and ra.terminal == rb.terminal
            if not ra.terminal:
                assert ra.actions == rb.actions and ra.n == rb.n
                np.testing.assert_allclose(ra.w, rb.w, rtol=0, atol=1e-9)
        assert ref._rng.getstate() == cpp._rng.getstate()
