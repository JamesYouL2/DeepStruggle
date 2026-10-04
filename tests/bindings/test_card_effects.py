"""The CARD_EFFECTS observation block (engine/include/ts/card_effects.hpp).

The engine's labeller is a port of ai/training/card_event_targets.py, which stays the reference:
the two must agree exactly on the same state and dice seeds. The block itself is that labeller run
on a redealt copy of the state, so it must not depend on where the opponent's unseen cards really
are -- a block that did would hand the network the opponent's hand.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest

import ts_engine as ts
from ai.training import card_event_targets as C
from bindings.ts_env import TsVectorizedEnv

CE = int(ts.OBS_FEATURE_CARD_EFFECTS)
PER_CARD, N_CARDS = 18, 110
WIDTH = PER_CARD * N_CARDS


def _positions(n: int, seed: int = 7) -> List[Tuple[ts.GameState, ts.Player]]:
    """Card decisions (as the block defines them) from random games, cloned."""
    env = TsVectorizedEnv(num_envs=16, base_seed=seed)
    _obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    out: List[Tuple[ts.GameState, ts.Player]] = []
    for _ in range(4000):
        dp = np.asarray(env.runner.get_decision_players())
        for i in np.flatnonzero(dp != 0):
            st = env.runner.get_state(int(i))
            p = ts.Player(int(dp[i]))
            if ts.card_effects_is_card_decision(st, p) and rng.random() < 0.1:
                out.append((st.clone(), p))
                if len(out) >= n:
                    return out
        acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)])
        _obs, masks, *_ = env.step(acts)
    raise AssertionError("not enough card decisions")


@pytest.fixture(scope="module")
def positions() -> List[Tuple[ts.GameState, ts.Player]]:
    return _positions(80)


def test_the_engine_labeller_matches_the_python_reference(positions) -> None:
    seeds = [11, 22, 33]
    labelled = 0
    for st, mover in positions:
        ref = C.label(st, mover, np.asarray(ts.extract_observation(st, mover)), seeds)
        got = np.asarray(ts.card_effects_label(st, mover, seeds))
        assert got.shape == (N_CARDS, PER_CARD)
        want = np.zeros_like(got)
        for j, c in enumerate(ref["cards"]):
            if c == 0:
                continue
            if ref["m3"][j]:
                want[c - 1, 0] = ref["y3"][j]
            if ref["m2"][j]:
                want[c - 1, 1:6] = ref["y2"][j]
            if ref["m4"][j]:
                want[c - 1, 6:] = ref["y4"][j]
                labelled += 1
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-4)
    assert labelled > 50                                   # the outcomes were actually exercised


def _hidden(st: ts.GameState, me: ts.Player) -> Tuple[List[int], List[int]]:
    opp = ts.CardLocation.HAND_USSR_UNKNOWN if me == ts.Player.US else ts.CardLocation.HAND_US_UNKNOWN
    hand = [c for c in range(1, 111) if st.get_card_location(c) == opp]
    deck = [c for c in range(1, 111) if st.get_card_location(c) == ts.CardLocation.DRAW_DECK]
    return hand, deck


def test_the_redeal_keeps_every_count_and_everything_the_mover_sees(positions) -> None:
    for st, me in positions[:30]:
        w = ts.card_effects_redeal(st, me)
        hand, deck = _hidden(st, me)
        whand, wdeck = _hidden(w, me)
        assert len(whand) == len(hand) and sorted(whand + wdeck) == sorted(hand + deck)
        for c in range(1, 111):
            if c not in hand and c not in deck:
                assert w.get_card_location(c) == st.get_card_location(c)
        assert np.array_equal(np.asarray(ts.extract_observation(w, me)),
                              np.asarray(ts.extract_observation(st, me)))


def test_the_block_does_not_depend_on_the_opponents_real_hand(positions) -> None:
    """Swap cards between the opponent's unseen hand and the deck: same observation, same block."""
    swapped = 0
    for st, me in positions:
        hand, deck = _hidden(st, me)
        if not hand or not deck:
            continue
        other = st.clone()
        opp = st.get_card_location(hand[0])
        for a, b in zip(hand, deck):
            other.set_card_location(a, ts.CardLocation.DRAW_DECK)
            other.set_card_location(b, opp)
        a = np.asarray(ts.extract_observation_features(st, me, CE))
        b = np.asarray(ts.extract_observation_features(other, me, CE))
        np.testing.assert_array_equal(a, b)
        swapped += 1
    assert swapped > 20


def test_the_block_is_zero_away_from_card_decisions_and_deterministic(positions) -> None:
    filled = 0
    for st, me in positions[:30]:
        x = np.asarray(ts.extract_observation_features(st, me, CE))
        assert x.shape == (ts.obs_size_for(CE),) and ts.obs_size_for(CE) == ts.OBS_SIZE + WIDTH
        np.testing.assert_array_equal(x[:ts.OBS_SIZE], np.asarray(ts.extract_observation(st, me)))
        np.testing.assert_array_equal(x, np.asarray(ts.extract_observation_features(st, me, CE)))
        block = x[ts.OBS_SIZE:].reshape(N_CARDS, PER_CARD)
        held = np.asarray(ts.extract_observation(st, me))[ts.OBS_SIZE - 100 - N_CARDS * 14:ts.OBS_SIZE - 100]
        held = held.reshape(N_CARDS, 14)[:, 1] > 0.5
        assert not block[~held].any()                      # only my own cards carry a row
        filled += int(block.any())
        opp = ts.Player.USSR if me == ts.Player.US else ts.Player.US
        assert not np.asarray(ts.extract_observation_features(st, opp, CE))[ts.OBS_SIZE:].any()
    assert filled > 20


def test_the_block_is_the_labeller_on_the_redealt_state_scaled(positions) -> None:
    st, me = positions[0]
    x = np.asarray(ts.extract_observation_features(st, me, CE))[ts.OBS_SIZE:].reshape(N_CARDS, PER_CARD)
    raw = np.asarray(ts.card_effects_label(ts.card_effects_redeal(st, me), me, []))
    # Without seeds the labeller fires no event and gives Ops reach only; that part must agree.
    np.testing.assert_allclose(x[:, 1:6] * C.AUX_SCALE[:5], raw[:, 1:6], rtol=0, atol=1e-5)
    assert raw[:, 1:6].any()


def test_feature_bits_and_widths_are_wired() -> None:
    assert CE == 1 << 2 and ts.OBS_FEATURES_ALL & CE
    assert ts.obs_size_for(CE | int(ts.OBS_FEATURE_OPS_BUDGET)) == ts.OBS_SIZE + 3 + WIDTH
