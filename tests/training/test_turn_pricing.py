"""ai/search/turn_pricing.py's rollout-root switches: the action-round key read from a batch
runner's observation equals `ar_key` of the state, a horizon of 0 reads the critic at once, a
longer horizon spends more rows and stops where it says, and shared dice pair the options."""
from __future__ import annotations

from typing import Any, List

import numpy as np
import torch

import ts_engine as ts
from ai.eval.paired_playouts import ar_key, decider
from ai.models.coldwar_net_v2 import create_coldwar_net_v2
from ai.search.turn_pricing import Candidate, ar_key_from_obs, price
from bindings.action_encoder import ActionEncoder


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    m = create_coldwar_net_v2("cpu")
    m.eval()
    return m


def _runner_positions(n: int = 16, plies: int = 300, seed: int = 99) -> Any:
    """A runner driven by random legal play for `plies` steps."""
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(n, seed)
    for _ in range(plies):
        masks = np.asarray(runner.get_action_masks())
        acts = [int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks]
        runner.step_flat_all(acts, True)
    return runner


def _candidates(k: int = 3, n: int = 8) -> List[Candidate]:
    runner = _runner_positions(n=n, plies=80)
    out = []
    for i in range(n):
        st = runner.get_state(i)
        if ts.Engine.is_terminal(st):
            continue
        legal = [int(a) for a in np.flatnonzero(np.asarray(ActionEncoder.get_legal_mask(st)))]
        if len(legal) < 2:
            continue
        opts = legal[:k]
        out.append(Candidate(game=i, choice=0, state=st, options=opts, probs=[], taken=opts[0]))
    assert out
    return out


def test_the_key_read_from_the_observation_is_the_states_key() -> None:
    # Across many plies of random play, every live row's observation decodes to ar_key(state),
    # from either side's view: the slot indices (global 7, 8, 9) are held here.
    runner = _runner_positions(n=24, plies=0)
    rng = np.random.default_rng(5)
    checked = 0
    for _ in range(400):
        masks = np.asarray(runner.get_action_masks())
        term = np.asarray(runner.get_terminals(), dtype=bool)
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        turns = runner.get_turns()
        dps = runner.get_decision_players()
        for i in np.flatnonzero(~term):
            st = runner.get_state(int(i))
            assert int(dps[i]) == int(decider(st))
            assert ar_key_from_obs(obs[i], int(turns[i]), int(dps[i])) == ar_key(st)
            checked += 1
        acts = [int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks]
        runner.step_flat_all(acts, True)
    assert checked > 1000


def test_horizon_zero_reads_the_critic_at_the_position_after_the_move() -> None:
    model = _model()
    cands = _candidates()
    rows = price(model, torch.device("cpu"), cands, worlds=2, seed=1, horizon=0)
    # At most two network rows per (option, world) -- the batch's forward, and the decider's own
    # view where the other side is to act -- and nothing else: no playout step was taken.
    assert 0 < rows <= 2 * sum(len(c.options) for c in cands) * 2
    for c in cands:
        assert c.values is not None and c.values.shape[0] == len(c.options) and c.values.shape[1] <= 2
        assert np.all(np.abs(c.values) <= 1.0)


def test_a_longer_horizon_spends_more_rows_and_the_turn_end_more_still() -> None:
    model = _model()
    spent = []
    for horizon in (0, 2, None):
        cands = _candidates()
        spent.append(price(model, torch.device("cpu"), cands, worlds=2, seed=1, horizon=horizon))
    assert spent[0] < spent[1] < spent[2]


def test_shared_dice_give_every_option_of_a_world_the_same_stream() -> None:
    # With one option, pricing it twice -- own dice against shared dice -- draws the dice at a
    # different point of the seed's stream, so the values differ; with shared dice the stream is
    # drawn once per world, so two options start from identical states before their moves.
    cands = _candidates(k=2, n=6)
    seen_states = []
    import ai.search.turn_pricing as tp
    original = tp.apply_move

    def spy(state: ts.GameState, action: int) -> ts.GameState:
        seen_states.append(int(state.rng_state))
        return original(state, action)

    tp.apply_move = spy
    try:
        price(_model(), torch.device("cpu"), cands, worlds=3, seed=7, horizon=0, shared_dice=True)
    finally:
        tp.apply_move = original
    # Options of one (candidate, world) are consecutive: their pre-move rng states are equal.
    it = iter(seen_states)
    for c in cands:
        for _w in range(3):
            first = next(it)
            for _ in c.options[1:]:
                assert next(it) == first
