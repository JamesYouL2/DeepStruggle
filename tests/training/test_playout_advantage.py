"""Paired-branch playout advantages (--playout-adv, ai/training/playout_advantage.py)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast

import numpy as np
import pytest
import torch

import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.training.nash_pg import NashPGTrainer
from ai.training.playout_advantage import (DEFAULT_DECISIONS, PlayoutLabel, PlayoutLabeller, PlayoutRecord,
                                           decision_codes, playout_pg_loss, select_candidates,
                                           split_half_reliability)
from bindings.ts_env import TsVectorizedEnv

_update = cast(Any, NashPGTrainer._playout_update)
_minibatch = cast(Any, NashPGTrainer._playout_minibatch_loss)

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _net() -> Any:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **M2D)
    net.eval()
    return net


def _records(n: int, net: Any) -> List[PlayoutRecord]:
    """Play-mode decisions from random games after turn 1, with the net's candidates."""
    env = TsVectorizedEnv(num_envs=8, base_seed=11)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)
    play_mode = int(ts.DecisionType.SELECT_PLAY_MODE)
    out: List[PlayoutRecord] = []
    for _ in range(4000):
        dp = np.asarray(env.runner.get_decision_players())
        for i in np.flatnonzero(dp != 0):
            st = env.runner.get_state(int(i))
            m = np.asarray(masks)[i] != 0
            if int(st.turn) >= 2 and int(st.ctx().decision_type) == play_mode and m.sum() >= 2 and len(out) < n:
                o = torch.from_numpy(np.asarray(obs)[i].astype(np.float32))
                with torch.no_grad():
                    lg = net(o.unsqueeze(0), torch.from_numpy(m).unsqueeze(0))[0][0].numpy()
                cands, prior = select_candidates(lg, m, 3)
                out.append(PlayoutRecord(st.clone(), o.half(), torch.from_numpy(m), int(dp[i]), cands, prior))
        if len(out) >= n:
            return out
        acts = np.array([int(rng.choice(np.flatnonzero(mm))) for mm in np.asarray(masks)])
        obs, masks, *_ = env.step(acts)
    raise AssertionError("not enough play-mode decisions")


def test_candidates_are_the_likeliest_legal_moves() -> None:
    logits = np.array([5.0, 9.0, 1.0, 3.0, 8.0], dtype=np.float32)
    mask = np.array([True, False, True, True, True])
    cands, prior = select_candidates(logits, mask, 3)
    assert cands.tolist() == [4, 0, 3]                    # 1 is the largest logit but illegal
    assert np.all(np.diff(prior) <= 0) and prior.sum() < 1.0
    assert select_candidates(logits, mask, 10)[0].tolist() == [4, 0, 3, 2]
    assert decision_codes(DEFAULT_DECISIONS) == (int(ts.DecisionType.SELECT_CARD),
                                                 int(ts.DecisionType.SELECT_PLAY_MODE),
                                                 int(ts.DecisionType.CHOOSE_BRANCH))
    with pytest.raises(ValueError):
        decision_codes(["ROLL_DIE"])


def test_the_same_move_twice_pairs_to_exactly_zero_and_the_runs_repeat() -> None:
    net = _net()
    recs = _records(6, net)
    for r in recs:                                        # every candidate is the favourite, twice
        r.cands = np.array([r.cands[0], r.cands[0]], dtype=np.int64)
    before = [r.state.to_save_json() if hasattr(r.state, "to_save_json") else None for r in recs]
    lab = PlayoutLabeller(pairs=4, horizon="turn", max_steps=400, merged=False, obs_features=0)
    mats, stats = lab.label(net, recs, "cpu", seed=7)
    assert stats["playout_capped"] == 0 and stats["playout_games"] == 6 * 2 * 4
    for m in mats:
        assert m.shape == (4, 2) and np.all(np.abs(m) <= 1.0)
        # shared dice, same move: identical games (the critic's batched float arithmetic aside)
        np.testing.assert_allclose(m[:, 0], m[:, 1], rtol=0, atol=1e-5)
    again, _ = lab.label(net, recs, "cpu", seed=7)
    for a, b in zip(mats, again):
        np.testing.assert_array_equal(a, b)
    other, _ = lab.label(net, recs, "cpu", seed=8)
    assert any(not np.array_equal(a, b) for a, b in zip(mats, other))    # the seed reaches the dice
    after = [r.state.to_save_json() if hasattr(r.state, "to_save_json") else None for r in recs]
    assert before == after                                # the recorded states are not touched


def test_a_redeal_is_shared_by_the_candidates_of_a_pair_and_varies_between_pairs() -> None:
    net = _net()
    recs = _records(6, net)
    for r in recs:
        r.cands = np.array([r.cands[0], r.cands[0]], dtype=np.int64)
    before = [r.state.to_save_json() if hasattr(r.state, "to_save_json") else None for r in recs]
    lab = PlayoutLabeller(pairs=4, horizon="turn", max_steps=400, merged=False, obs_features=0, hidden="redeal")
    mats, _ = lab.label(net, recs, "cpu", seed=7)
    for m in mats:
        # same redeal and dice in a pair, same move: identical games
        np.testing.assert_allclose(m[:, 0], m[:, 1], rtol=0, atol=1e-5)
    kept, _ = PlayoutLabeller(4, "turn", 400, False, 0).label(net, recs, "cpu", seed=7)
    # the same dice with the hands redealt: some game goes differently
    assert any(not np.allclose(a, b) for a, b in zip(mats, kept))
    after = [r.state.to_save_json() if hasattr(r.state, "to_save_json") else None for r in recs]
    assert before == after
    with pytest.raises(ValueError):
        PlayoutLabeller(4, "turn", 400, False, 0, hidden="peek")


def test_the_default_decisions_cover_wargames_ending_the_game() -> None:
    assert int(ts.DecisionType.CHOOSE_BRANCH) in decision_codes(DEFAULT_DECISIONS)


def test_game_horizon_scores_results_only() -> None:
    net = _net()
    recs = _records(2, net)
    lab = PlayoutLabeller(pairs=2, horizon="game", max_steps=20000, merged=False, obs_features=0)
    mats, stats = lab.label(net, recs, "cpu", seed=3)
    assert stats["playout_capped"] == 0
    for m in mats:
        assert set(np.unique(m)).issubset({-1.0, 0.0, 1.0})   # every game ended: a result, no critic


def test_a_move_the_state_refuses_raises() -> None:
    net = _net()
    rec = _records(1, net)[0]
    illegal = int(np.flatnonzero(rec.mask.numpy() == 0)[0])
    rec.cands = np.array([rec.cands[0], illegal], dtype=np.int64)
    with pytest.raises(RuntimeError, match="refused"):
        PlayoutLabeller(2, "turn", 400, False, 0).label(net, [rec], "cpu", seed=1)


def test_the_gradient_moves_only_the_candidates_toward_the_better_one() -> None:
    logits = torch.zeros(1, 6, requires_grad=True)
    cands = torch.tensor([[1, 3, 4]])
    valid = torch.tensor([[True, True, True]])
    q = torch.tensor([[0.2, -0.1, -0.1]])
    loss, st = playout_pg_loss(logits, cands, valid, q, min_scale=0.05)
    loss.backward()
    g = logits.grad
    assert g is not None
    assert g[0, 1] < 0 and g[0, 3] > 0 and g[0, 4] > 0      # descent raises the best, lowers the others
    assert torch.all(g[0, [0, 2, 5]] == 0)                 # non-candidates are left alone
    assert abs(float(g[0].sum())) < 1e-6
    assert st["playout_agree"] == 1.0                      # uniform policy: ties go to the first
    # A padded entry contributes nothing.
    logits2 = torch.zeros(1, 6, requires_grad=True)
    loss2, _ = playout_pg_loss(logits2, torch.tensor([[1, 3, 0]]), torch.tensor([[True, True, False]]),
                               torch.tensor([[0.2, -0.1, 5.0]]), min_scale=0.05)
    loss2.backward()
    assert logits2.grad is not None and logits2.grad[0, 0] == 0


def test_split_half_reliability() -> None:
    rng = np.random.default_rng(0)
    signal = [np.tile(rng.normal(size=3), (8, 1)) for _ in range(20)]
    r = split_half_reliability(signal)
    assert r is not None and r > 0.999
    noise = [rng.normal(size=(8, 3)) for _ in range(400)]
    r = split_half_reliability(noise)
    assert r is not None and abs(r) < 0.15
    assert split_half_reliability([np.zeros((1, 3))]) is None


def test_the_update_moves_the_policy_toward_the_playout_best() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **M2D)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = 64
    obs = torch.randn(n, ts.OBS_SIZE).half()
    mask = torch.zeros(n, 220, dtype=torch.bool)
    mask[:, 110:114] = True
    ready = []
    for i in range(n):
        best = i % 4                                       # a different best move per position
        q = np.full(4, -0.2, dtype=np.float32)
        q[best] = 0.2
        ready.append(PlayoutLabel(obs[i], mask[i], np.arange(110, 114, dtype=np.int64), q))

    def fake() -> SimpleNamespace:
        ns = SimpleNamespace(active_net=net, optimizer=opt, max_grad_norm=10.0, playout_coef=1.0,
                             playout_mode="separate", playout_min_batch=n, playout_steps=2, playout_batch=32,
                             playout_candidates=4, playout_min_scale=0.05, _po_ready=ready, _po_labelled=n,
                             _po_stats={})
        ns._playout_minibatch_loss = lambda: _minibatch(ns)
        return ns

    first = _update(fake())
    for _ in range(60):
        last = _update(fake())
    assert last["playout_p_best"] > first["playout_p_best"] + 0.2
    small = fake()
    small._po_ready = ready[:10]
    assert "playout_loss" not in _update(small)


def test_joint_mode_reports_the_terms_the_ppo_minibatches_added_and_resets() -> None:
    ns = SimpleNamespace(playout_mode="joint", _po_ready=[], _po_labelled=7, _po_stats={"playout_recorded": 3.0},
                         _po_joint_acc={"playout_p_best": 1.2, "playout_loss": -0.4}, _po_joint_n=4)
    out = _update(ns)
    assert out["playout_p_best"] == pytest.approx(0.3) and out["playout_loss"] == pytest.approx(-0.1)
    assert out["playout_recorded"] == 3.0 and out["playout_labelled"] == 7.0
    assert ns._po_joint_acc == {} and ns._po_joint_n == 0
    assert "playout_p_best" not in _update(ns)          # nothing added since: nothing reported


def test_fresh_metrics_read_the_policy_at_the_decision_against_the_playout_best() -> None:
    mats = [np.array([[0.0, 0.4, 0.1], [0.0, 0.4, 0.1]]),     # best is candidate 1
            np.array([[0.3, 0.0], [0.3, 0.0]])]               # best is candidate 0, the favourite
    labeller = SimpleNamespace(label=lambda *a, **k: (mats, {}))
    recs = [PlayoutRecord(None, torch.zeros(1).half(), torch.zeros(1, dtype=torch.bool), 1,
                          np.array([5, 6, 7]), np.array([0.5, 0.25, 0.25], dtype=np.float32)),
            PlayoutRecord(None, torch.zeros(1).half(), torch.zeros(1, dtype=torch.bool), -1,
                          np.array([5, 6]), np.array([0.6, 0.2], dtype=np.float32))]
    ns = SimpleNamespace(_po_pending=recs, _po_labeller=labeller, active_net=torch.nn.Linear(1, 1), device="cpu",
                         _po_ready=[], _po_labelled=0, playout_buffer=10, _po_stats={})
    cast(Any, NashPGTrainer._playout_label)(ns)
    st = ns._po_stats
    # p on the best: 0.25 (renormalised over 1.0) and 0.75 (0.6 / 0.8)
    assert st["playout_fresh_p_best"] == pytest.approx((0.25 + 0.75) / 2)
    # gap: 0.4 - (0.5*0 + 0.25*0.4 + 0.25*0.1) = 0.275; 0.3 - 0.75*0.3 = 0.075
    assert st["playout_fresh_gap"] == pytest.approx((0.275 + 0.075) / 2)
    assert st["playout_best_not_favourite"] == 0.5 and len(ns._po_ready) == 2
