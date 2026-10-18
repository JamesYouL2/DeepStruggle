"""ai/search/belief.py (P35 3b): the C++ rows are the probe's Python rebuild, the log-likelihoods are
the slow computation's, the tracker deals hands of the right size from the hidden pool and reuses them
until the opponent moves, and a belief root plays a tournament through the game hooks."""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import torch

import ts_engine as ts
from ai.eval.belief_history_probe import collect, world_at
from ai.models.ladder_net import create_ladder_net
from ai.search.belief import (BeliefTracker, Now, UNKNOWN, log_weights, player, record, sample_hands, side)
from ai.search.rollout_root import RolloutConfig, RolloutRootAgent, rollout_spec_config
from bindings.ts_env import model_obs_features
from tests.training.test_ladder_tokens import SHALLOW
from tools.lib.batch_tournament import BatchMatchRunner


def _model() -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **SHALLOW).eval()


def _jobs(model: torch.nn.Module) -> List[Tuple[Now, np.ndarray, list]]:
    samples = collect(model, torch.device("cpu"), 6, 2, 1.0, 11, set(range(6)))
    rng = np.random.default_rng(3)
    jobs = []
    for s in samples:
        now = Now(s.decider, s.locs, s.pool > 0.5, int(s.label.sum()))
        rec = record(s.turn_log, now, True)
        if rec:
            jobs.append((now, sample_hands(now, 8, rng), rec))
        if len(jobs) == 6:
            break
    assert jobs
    return jobs


def test_the_c_rows_are_the_python_rebuild() -> None:
    model = _model()
    for now, hands, rec in _jobs(model)[:3]:
        opp = 3 - now.me
        for d, left in rec[:3]:
            held = np.ascontiguousarray(hands | left.astype(np.uint8)[None, :])
            obs, masks = ts.hypothesis_rows(d.state, player(now.me), held, 0)
            for k in range(len(held)):
                w = world_at(d, opp, held[k] > 0)
                assert np.array_equal(obs[k], ts.extract_observation_features(w, player(opp), 0))
                assert np.array_equal(masks[k], np.asarray(ts.ActionMask.generate_flat_mask(w)))
                assert np.array_equal(np.asarray([int(w.get_card_location(c)) for c in range(1, 111)]),
                                      np.asarray([int(x.get_card_location(c)) for c in range(1, 111)
                                                  for x in [ts.place_hand(d.state, player(now.me), held[k])]]))


def test_log_weights_are_the_slow_computation() -> None:
    model = _model()
    jobs = _jobs(model)
    got = log_weights(model, torch.device("cpu"), jobs)
    feats = model_obs_features(model)
    for (now, hands, rec), lw in zip(jobs, got):
        want = np.zeros(len(hands))
        for d, left in rec:
            for k in range(len(hands)):
                w = world_at(d, 3 - now.me, (hands[k] > 0) | left)
                obs = torch.from_numpy(ts.extract_observation_features(w, player(3 - now.me), feats))[None]
                m = torch.from_numpy(np.asarray(ts.ActionMask.generate_flat_mask(w)))[None]
                with torch.no_grad():
                    lg = model(obs, m)[0].float().masked_fill(m == 0, float("-inf"))
                    v = float(torch.log_softmax(lg, -1)[0, d.action])
                want[k] += v if np.isfinite(v) else -1e9
        assert np.allclose(lw, want, atol=1e-4)


def test_the_tracker_deals_hidden_hands_and_reuses_them_until_the_opponent_moves() -> None:
    model = _model()
    tr = BeliefTracker(model, torch.device("cpu"), particles=32, seed=1)
    runner = ts.VectorizedBatchRunner(2, 77)
    rng = np.random.default_rng(0)
    dealt = 0
    for _ in range(400):
        if all(runner.get_terminals()):
            break
        masks = np.asarray(runner.get_action_masks())
        players = list(runner.get_decision_players())
        states = [runner.get_state(i) for i in range(2)]
        acts = [int(rng.choice(np.flatnonzero(m))) if m.any() else 0 for m in masks]
        live = [i for i in range(2) if players[i] != 0 and not runner.get_terminals()[i]]
        hands = tr.worlds(live, [states[i] for i in live], 4)
        for i, h in zip(live, hands):
            if h is None:
                continue
            now = Now.of(states[i], side(players[i]))
            assert h.shape == (4, 110)
            assert np.all(h.sum(1) == now.hand) and not np.any(h[:, ~now.pool])
            dealt += 1
        computed = tr.stats["positions"]
        again = tr.worlds(live, [states[i] for i in live], 4)
        assert tr.stats["positions"] == computed                     # the second call hits the cache
        assert all((a is None) == (b is None) for a, b in zip(hands, again))
        tr.observe(live, [states[i] for i in live], [acts[i] for i in live], [players[i] for i in live])
        runner.step_flat_all(acts, True)
    assert dealt > 50


def test_the_spec_names_the_belief() -> None:
    _, cfg, label = rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed:belief")
    assert cfg.belief == 2048 and label.endswith("-keyed-belief2048-z2")
    assert rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed:belief512")[1].belief == 512
    assert rollout_spec_config("rollout:m.pt:4:16:4:z2:earlier:own:keyed")[1].belief == 0


def test_a_belief_root_plays_a_tournament_through_the_game_hooks() -> None:
    model = _model()
    belief = RolloutRootAgent(model, "belief", RolloutConfig(k=2, worlds=2, horizon=0, belief=16))
    plain = RolloutRootAgent(model, "plain", RolloutConfig(k=2, worlds=2, horizon=0))
    res = BatchMatchRunner.play_parallel_matchup(belief, plain, games_per_side=1, device="cpu", temperature=0.0)
    assert res["total_games"] == 2
    assert belief.belief is not None
    assert belief.belief.stats["positions"] > 20 and belief.belief.stats["decisions_sum"] > 0
