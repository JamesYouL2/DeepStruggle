"""P29 bet 2: auxiliary ownership and VP-margin targets (--aux-ownership / --aux-vp-margin)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast

import numpy as np
import torch

import ts_engine as ts

from ai.models.coldwar_net_v2 import create_like
from ai.models.ladder_net import AUX_OWN_COUNTRIES, create_ladder_net, ladder_config_from_state_dict
from ai.training.nash_pg import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv

# The trainer's aux methods run here on stand-in objects holding only the fields they read.
_resolve = cast(Any, NashPGTrainer._aux_resolve)
_update = cast(Any, NashPGTrainer._aux_update)

#: M2d, the current architecture (E6-03-44's ladder_config).
M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=1, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def test_aux_heads_are_absent_by_default_and_recovered_from_the_weights() -> None:
    plain = create_ladder_net("cpu", **M2D)
    assert not plain.aux_heads
    assert not any(k.startswith("aux_") for k in plain.state_dict())
    cfg = ladder_config_from_state_dict(plain.state_dict())
    assert cfg is not None and cfg["aux_heads"] is False

    net = create_ladder_net("cpu", **{**M2D, "aux_heads": True})
    sd = net.state_dict()
    assert net.ladder_config()["aux_heads"] is True
    cfg = ladder_config_from_state_dict(sd)
    assert cfg is not None and cfg["aux_heads"] is True
    twin = create_like(net)
    twin.load_state_dict(sd, strict=True)
    own, vp = net.forward_aux(torch.zeros(3, ts.OBS_SIZE))
    assert own.shape == (3, AUX_OWN_COUNTRIES, 3) and vp.shape == (3, 1)
    # forward() keeps its three-value contract
    assert len(net(torch.zeros(2, ts.OBS_SIZE), torch.ones(2, 220, dtype=torch.bool))) == 3


def _run_until_games_end(env: TsVectorizedEnv, want: int) -> List[Dict[str, Any]]:
    rng = np.random.default_rng(0)
    obs, masks, _ = env.reset_all()
    done: List[Dict[str, Any]] = []
    for _ in range(20000):
        acts = np.array([int(rng.choice(np.flatnonzero(m))) if np.asarray(m).any() else 0
                         for m in np.asarray(masks)])
        obs, masks, _r, _d, info = env.step(acts)
        done.extend(info.get("completed_episodes", []))
        if len(done) >= want:
            break
    return done


def test_final_control_is_recorded_only_when_asked_for() -> None:
    env = TsVectorizedEnv(num_envs=4, base_seed=11)
    eps = _run_until_games_end(env, 2)
    assert eps and all("final_control" not in ep for ep in eps)

    env = TsVectorizedEnv(num_envs=4, base_seed=11)
    env.record_final_control = True
    eps = _run_until_games_end(env, 2)
    assert eps
    for ep in eps:
        ctrl = ep["final_control"]
        assert ctrl.shape == (84,) and ctrl.dtype == np.int8
        assert set(np.unique(ctrl)) <= {-1, 0, 1}


def test_labels_are_in_the_movers_frame() -> None:
    fake = SimpleNamespace(num_envs=2, _aux_pending=[[], []], _aux_ready=[])
    o = torch.zeros(1)
    fake._aux_pending[0] = [(o, 1), (o, -1)]          # a US decision, then a USSR one
    ctrl = np.zeros(84, dtype=np.int8)
    ctrl[0], ctrl[1] = 1, -1                          # country 0 US, country 1 USSR, the rest nobody
    _resolve(fake, [{"env_idx": 0, "final_control": ctrl, "victory_points": 8}])
    assert fake._aux_pending[0] == []
    (_, cls_us, vp_us), (_, cls_ussr, vp_ussr) = fake._aux_ready
    assert list(cls_us[:3]) == [0, 1, 2] and list(cls_ussr[:3]) == [1, 0, 2]   # mine / theirs / neither
    assert vp_us == 0.4 and vp_ussr == -0.4

    # a game whose control was not recorded drops its positions rather than mislabelling them
    fake._aux_pending[1] = [(o, 1)]
    _resolve(fake, [{"env_idx": 1, "victory_points": 3}])
    assert fake._aux_pending[1] == [] and len(fake._aux_ready) == 2


def test_the_aux_step_lowers_its_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**M2D, "aux_heads": True})
    n = 64
    obs = torch.randn(n, ts.OBS_SIZE).half()
    cls = np.random.default_rng(0).integers(0, 3, size=(n, 84)).astype(np.int64)
    vp = [0.5] * n

    def fake() -> SimpleNamespace:
        return SimpleNamespace(
            active_net=net, optimizer=opt, batch_size=32, max_grad_norm=10.0,
            aux_own_coef=1.0, aux_vp_coef=1.0, aux_min_batch=n,
            _aux_pending=[[]], _aux_ready=[(obs[i], cls[i], vp[i]) for i in range(n)])

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    first = _update(fake())
    for _ in range(30):
        last = _update(fake())
    assert first["aux_n"] == n
    assert last["aux_own_loss"] < first["aux_own_loss"] and last["aux_vp_loss"] < first["aux_vp_loss"]
    # below the batch threshold nothing trains
    small = fake()
    small._aux_ready = small._aux_ready[:10]
    assert "aux_n" not in _update(small)
