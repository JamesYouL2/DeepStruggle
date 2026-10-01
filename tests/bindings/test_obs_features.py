"""The view spec's observation half: optional blocks appended to the base layout (owner, 2026-10-01).

The base layout must stay bit-identical whatever is appended, each block must hold what it says,
a batch mixing two feature sets must give each decider its own, and a model of one set must never
be handed another's floats.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

import ts_engine as ts

from ai.models.ladder_net import OBS_FEATURE_WIDTHS, create_ladder_net
from ai.training.card_event_targets import BREZHNEV, CONTAINMENT, PURGE_US, PURGE_USSR, effective_ops
from bindings.ts_env import TsVectorizedEnv, check_obs_width, model_obs_features

F = int(ts.OBS_FEATURE_OPS_BUDGET)
BASE = int(ts.OBS_SIZE)

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _positions(n: int, seed: int = 3) -> List[Tuple["ts.GameState", "ts.Player"]]:
    env = TsVectorizedEnv(num_envs=16, base_seed=seed)
    _, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    out: List[Tuple["ts.GameState", "ts.Player"]] = []
    for _ in range(4000):
        for i in range(16):
            st = env.runner.get_state(i)
            p = st.ctx().decision_player
            if int(p) != 0:
                out.append((st.clone(), p))
        if len(out) >= n:
            break
        _, masks, *_ = env.step(np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)]))
    return out[:n]


def test_widths_agree_between_engine_and_model() -> None:
    assert ts.obs_size_for(0) == BASE
    for bit, width in OBS_FEATURE_WIDTHS.items():
        assert ts.obs_size_for(bit) == BASE + width
    assert ts.OBS_FEATURES_ALL == sum(OBS_FEATURE_WIDTHS)


def test_unknown_bits_are_refused() -> None:
    st, p = _positions(1)[0]
    with pytest.raises(ValueError):
        ts.obs_size_for(1 << 7)
    with pytest.raises(ValueError):
        ts.extract_observation_features(st, p, 1 << 7)
    with pytest.raises(ValueError):
        ts.VectorizedBatchRunner(2, 1).set_obs_features([1 << 7, 0], [0, 0])


def test_the_base_prefix_is_bit_identical() -> None:
    for st, p in _positions(600):
        base = np.asarray(ts.extract_observation(st, p))
        assert np.array_equal(base, np.asarray(ts.extract_observation_features(st, p, 0)))
        assert np.array_equal(base, np.asarray(ts.extract_observation_features(st, p, F))[:BASE])


def test_ops_budget_is_the_engines_grant_and_modifiers() -> None:
    plays = 0
    for st, p in _positions(3000, seed=8):
        c = st.ctx()
        if int(c.decision_type) != int(ts.DecisionType.SELECT_PLAY_MODE) or int(c.pending_op_card) == 0:
            continue
        card = int(c.pending_op_card)
        for bits in (0, CONTAINMENT, BREZHNEV, PURGE_US, PURGE_USSR, CONTAINMENT | PURGE_USSR):
            x = st.clone()
            x.persistent_effects = (int(x.persistent_effects) & ~(CONTAINMENT | BREZHNEV | PURGE_US | PURGE_USSR)) | bits
            o = np.asarray(ts.extract_observation_features(x, p, F))[BASE:]
            assert o[0] * 5 == pytest.approx(effective_ops(x, card, p, True, True))
            y = x.clone()
            if ts.Engine.try_step_flat(y, 112, True, False) and int(y.ctx().pending_op_card) == card:
                assert o[0] * 5 == pytest.approx(int(y.ctx().pending_ops_value))   # what influence grants
            mod = {ts.Player.US: int(bool(bits & CONTAINMENT)) - int(bool(bits & PURGE_US)),
                   ts.Player.USSR: int(bool(bits & BREZHNEV)) - int(bool(bits & PURGE_USSR))}
            opp = ts.Player.USSR if p == ts.Player.US else ts.Player.US
            assert (o[1], o[2]) == (mod[p], mod[opp])
        plays += 1
    assert plays >= 50
    # away from a play-mode decision the value is 0
    for st, p in _positions(200, seed=9):
        if int(st.ctx().decision_type) != int(ts.DecisionType.SELECT_PLAY_MODE):
            assert np.asarray(ts.extract_observation_features(st, p, F))[BASE] == 0.0


def test_a_mixed_batch_gives_each_decider_its_own_view() -> None:
    r = ts.VectorizedBatchRunner(6, 11)
    us, ussr = [F, 0, 0, F, F, 0], [0, F, 0, F, 0, F]
    r.set_obs_features(us, ussr)
    assert r.obs_width == BASE + OBS_FEATURE_WIDTHS[F]
    obs = np.asarray(r.get_observations())
    for i in range(6):
        st = r.get_state(i)
        p = st.ctx().decision_player if int(st.ctx().decision_player) != 0 else st.phasing_player
        want = np.asarray(ts.extract_observation_features(st, p, us[i] if p == ts.Player.US else ussr[i]))
        assert np.array_equal(obs[i, :len(want)], want) and not obs[i, len(want):].any()
    r.set_obs_features([0] * 6, [0] * 6)
    assert r.obs_width == BASE


def test_the_env_keeps_its_view_across_a_reseeded_reset() -> None:
    env = TsVectorizedEnv(num_envs=4, base_seed=2)
    env.set_obs_features(F, F)
    assert env.observation_size == BASE + OBS_FEATURE_WIDTHS[F]
    obs, _, _ = env.reset_all(base_seed=99)
    assert np.asarray(obs).shape[1] == BASE + OBS_FEATURE_WIDTHS[F]


def test_a_model_names_its_view_and_refuses_another() -> None:
    plain = create_ladder_net("cpu", **M2D)
    feat = create_ladder_net("cpu", **{**M2D, "obs_features": F})
    assert model_obs_features(plain) == 0 and model_obs_features(feat) == F
    assert check_obs_width(plain) == BASE and check_obs_width(feat) == BASE + OBS_FEATURE_WIDTHS[F]
    mask = torch.ones(1, 220, dtype=torch.bool)
    with pytest.raises(ValueError):
        feat(torch.zeros(1, BASE), mask)
    with pytest.raises(ValueError):
        plain(torch.zeros(1, BASE + OBS_FEATURE_WIDTHS[F]), mask)


def test_a_match_between_two_views_plays_out() -> None:
    from tools.lib.batch_tournament import BatchMatchRunner
    from tools.lib.player_agent import NeuralAgent
    torch.manual_seed(0)
    a = NeuralAgent(model=create_ladder_net("cpu", **M2D), name="base", device="cpu")
    b = NeuralAgent(model=create_ladder_net("cpu", **{**M2D, "obs_features": F}), name="ops", device="cpu")
    res = BatchMatchRunner.play_parallel_matchup(a, b, games_per_side=4, batch_chunk_size=8, device="cpu")
    assert res is not None
