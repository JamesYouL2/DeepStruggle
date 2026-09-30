"""The move a search plays when root visit counts tie.

With a small budget ties are the common case: two simulations after the root expansion leave most
positions with two moves at one visit each. Breaking them by list index played whichever move the
engine enumerates first, and a 2-simulation search scored 47.7% against its own network at 512
games a side -- worse than playing the network's move with no search at all. These tests pin the
tie-break (visits, then the mover's mean value, then the prior) and its consequence that a
one-simulation search plays exactly the network's argmax.

CPU only, unlike test_batched_mcts.py, so CI runs them.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch

import ts_engine as ts
from ai.search.batched_mcts import BatchedMCTS, BatchedMCTSConfig, _BNode
from ai.search.pimcts import drain_chance_nodes
from bindings.action_encoder import ActionEncoder


def _root(mover: ts.Player, n: List[float], w: List[float], priors: List[float]) -> _BNode:
    return _BNode(state=ts.GameState(), mover=int(mover), terminal=False,
                  actions=[10 + i for i in range(len(n))], priors=priors, n=n, w=w,
                  expanded=True)


def test_most_visits_wins_regardless_of_value_and_prior():
    r = _root(ts.Player.US, n=[1, 3, 2], w=[1.0, -3.0, 2.0], priors=[0.8, 0.1, 0.1])
    assert BatchedMCTS._most_visited(r) == 1


def test_tied_visits_go_to_the_better_value_for_a_us_mover():
    r = _root(ts.Player.US, n=[1, 1, 0], w=[-0.5, 0.3, 0.0], priors=[0.6, 0.3, 0.1])
    assert BatchedMCTS._most_visited(r) == 1


def test_tied_visits_read_the_value_from_the_ussr_side():
    # w is summed from the US perspective, so the USSR prefers the lower one.
    r = _root(ts.Player.USSR, n=[1, 1, 0], w=[-0.5, 0.3, 0.0], priors=[0.3, 0.6, 0.1])
    assert BatchedMCTS._most_visited(r) == 0


def test_tied_visits_and_values_go_to_the_higher_prior():
    r = _root(ts.Player.US, n=[2, 2, 1], w=[0.4, 0.4, 0.9], priors=[0.2, 0.5, 0.3])
    assert BatchedMCTS._most_visited(r) == 1


def test_no_visits_falls_back_to_the_prior():
    r = _root(ts.Player.US, n=[0, 0, 0], w=[0.0, 0.0, 0.0], priors=[0.2, 0.3, 0.5])
    assert BatchedMCTS._most_visited(r) == 2


def _states(n: int) -> List[ts.GameState]:
    out = []
    for i in range(n):
        s = ts.GameState()
        ts.Engine.init_game(s, 7100 + i)
        for _ in range(20 + 3 * i):
            if ts.Engine.is_terminal(s):
                break
            legal = np.flatnonzero(np.asarray(ts.Engine.get_flat_action_mask(s)))
            if not len(legal):
                break
            ts.Engine.step_flat(s, int(legal[0]))
        drain_chance_nodes(s)
        out.append(s)
    return out


def test_one_simulation_plays_the_network_argmax():
    """The root expansion is the only network call a one-simulation search makes for its
    choice, so it must play the policy's own argmax: with no search, search must be a no-op."""
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    torch.manual_seed(0)
    model = create_coldwar_net_v2(device="cpu", graph_layers=0).eval()
    states = [s for s in _states(8) if not ts.Engine.is_terminal(s)]
    assert states

    mcts = BatchedMCTS(model, device=torch.device("cpu"), config=BatchedMCTSConfig(
        simulations=1, determinize=False, auto_advance=False, advance_root=False))
    picks = mcts.best_actions(states)

    for st, pick in zip(states, picks):
        root = mcts._search([st])[0]
        assert root is not None and root.actions
        expected = root.actions[int(np.argmax(root.priors))]
        assert pick == expected
        assert np.asarray(ActionEncoder.get_legal_mask(st))[pick]


def test_two_simulation_ties_are_broken_by_value_not_by_list_order():
    """Two simulations split one visit each between two children in most positions. The pick
    must be the tied child the mover values more, not the one the engine listed first."""
    from ai.models.coldwar_net_v2 import create_coldwar_net_v2
    torch.manual_seed(0)
    model = create_coldwar_net_v2(device="cpu", graph_layers=0).eval()
    states = [s for s in _states(8) if not ts.Engine.is_terminal(s)]

    def searcher() -> BatchedMCTS:
        return BatchedMCTS(model, device=torch.device("cpu"), config=BatchedMCTSConfig(
            simulations=2, determinize=False, auto_advance=False, advance_root=False))

    picks = searcher().best_actions(states)
    roots = searcher()._search(states)

    order_differs = 0
    for pick, root in zip(picks, roots):
        assert root is not None and root.actions
        top = max(root.n)
        tied = [i for i, v in enumerate(root.n) if v == top]
        sign = 1.0 if root.mover == int(ts.Player.US) else -1.0
        best = max(tied, key=lambda i: (sign * root.w[i] / root.n[i], root.priors[i]))
        assert pick == root.actions[best]
        order_differs += int(best != tied[0])
    # The positions must actually exercise the bug: at least one where first-in-list is wrong.
    assert order_differs > 0
