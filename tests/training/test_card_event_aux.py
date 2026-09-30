"""P30: the card-event auxiliary target (--aux-card-events)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, cast

import numpy as np
import torch

import ts_engine as ts

from ai.models.coldwar_net_v2 import create_like
from ai.models.ladder_net import CARD_AUX_DIM, create_ladder_net, ladder_config_from_state_dict
from ai.training.card_event_targets import (AUX_DIM, AUX_MEAN, AUX_SCALE, CARD_OFFSET, GLOBAL_OFFSET, SCORING,
                                            aux_label, held_cards, label)
from ai.training.nash_pg import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv

_update = cast(Any, NashPGTrainer._card_aux_update)

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def test_head_is_absent_by_default_and_recovered_from_the_weights() -> None:
    assert CARD_AUX_DIM == AUX_DIM
    plain = create_ladder_net("cpu", **M2D)
    assert not plain.card_aux and not any(k.startswith("card_aux_") for k in plain.state_dict())
    net = create_ladder_net("cpu", **{**M2D, "card_aux": True})
    sd = net.state_dict()
    cfg = ladder_config_from_state_dict(sd)
    assert cfg is not None and cfg["card_aux"] is True and net.ladder_config()["card_aux"] is True
    create_like(net).load_state_dict(sd, strict=True)
    assert net.forward_card_aux(torch.zeros(3, ts.OBS_SIZE)).shape == (3, 110, CARD_AUX_DIM)
    assert len(net(torch.zeros(2, ts.OBS_SIZE), torch.ones(2, 220, dtype=torch.bool))) == 3


def _positions(n: int) -> List[tuple]:
    env = TsVectorizedEnv(num_envs=8, base_seed=5)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(0)
    out: List[tuple] = []
    for t in range(4000):
        dp = np.asarray(env.runner.get_decision_players())
        if t % 25 == 0:
            for i in np.flatnonzero(dp != 0):
                # get_state is a view of the live environment; keep a copy, or later steps change it
                st = env.runner.get_state(int(i)).clone()
                if int(st.turn) >= 2:
                    out.append((st, ts.Player(int(dp[i])), np.asarray(obs)[i].astype(np.float32)))
        if len(out) >= n:
            break
        acts = np.array([int(rng.choice(np.flatnonzero(m))) for m in np.asarray(masks)])
        obs, masks, *_ = env.step(acts)
    return out


def test_labels_cover_the_hand_and_the_scoring_vp_follows_the_regional_margin() -> None:
    region_of = {2: 0, 1: 1, 3: 2, 79: 3, 37: 4, 81: 5}
    xs, ys = [], []
    for st, mover, obs in _positions(60):
        lab = label(st, mover, obs, (11, 12))
        held = held_cards(obs)
        assert list(lab["cards"][:len(held)]) == held and not lab["cards"][len(held):].any()
        for j, c in enumerate(lab["cards"]):
            if lab["m4"][j] and int(c) in region_of:
                xs.append(float(obs[GLOBAL_OFFSET + 64 + region_of[int(c)]]) * 20.0)
                ys.append(float(lab["y4"][j, 0]))
        a = aux_label(st, mover, obs, 7)
        if a is not None:
            cards, y, m = a
            assert y.shape == (12, AUX_DIM) and m.shape == (12, AUX_DIM)
            assert not y[~m].any()                    # unlabelled entries carry no target
    assert len(xs) >= 5 and set(region_of) <= set(SCORING)
    assert np.corrcoef(xs, ys)[0, 1] > 0.9           # scoring now pays the margin, from the mover's side


def test_the_update_lowers_its_loss_on_a_fixed_batch() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**M2D, "card_aux": True})
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = 64
    rng = np.random.default_rng(0)
    obs = torch.randn(n, ts.OBS_SIZE).half()
    cards = np.zeros((n, 12), dtype=np.int16)
    cards[:, :5] = rng.integers(1, 111, size=(n, 5))
    y = rng.normal(size=(n, 12, AUX_DIM)).astype(np.float32)
    m = np.zeros((n, 12, AUX_DIM), dtype=bool)
    m[:, :5] = True
    y[~m] = 0.0

    def fake() -> SimpleNamespace:
        return SimpleNamespace(active_net=net, optimizer=opt, batch_size=32, max_grad_norm=10.0,
                               aux_card_coef=1.0, aux_card_min_batch=n, _card_label_seconds=0.0,
                               _card_ready=[(obs[i], cards[i], y[i], m[i]) for i in range(n)])

    first = _update(fake())
    for _ in range(30):
        last = _update(fake())
    assert first["card_aux_n"] == n and last["card_aux_loss"] < first["card_aux_loss"]
    small = fake()
    small._card_ready = small._card_ready[:10]
    assert "card_aux_n" not in _update(small)
    assert np.all(AUX_SCALE > 0) and AUX_MEAN.shape == (AUX_DIM,)
    assert CARD_OFFSET == 84 * 26
