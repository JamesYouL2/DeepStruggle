"""P31 1a (the behaviour floor) and 1b (scenario seeding): the floor samples the mixture it claims and
stores the mixture's log-prob; off, it touches nothing; seeding forces the precursor plays the engine
expects, once per card per game, and stores them as the environment's (learner = 0)."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pytest
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from ai.training.show_and_decide import (EVENT_SLOT, PLAY_MODE_LO, REGION_LO, SCENARIOS,
                                         ScenarioSeeder, apply_floor, event_choice_rows,
                                         floor_eps, floor_rows)
from bindings.action_encoder import ActionEncoder as A
from bindings.ts_env import TsVectorizedEnv

SHALLOW: dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _trainer(buffer: int = 16, envs: int = 8, **kw: Any) -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    model = create_ladder_net(dev, **SHALLOW)
    env = TsVectorizedEnv(num_envs=envs, base_seed=123)
    return NashPGTrainer(active_net=model, env=env, num_envs=envs, buffer_size=buffer, lr=3e-4,
                         eta=0.1, ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


# ------------------------------------------------------------------------------- 1a: the floor

def test_the_floor_covers_play_mode_and_in_event_choices_only() -> None:
    m = torch.zeros(7, 220, dtype=torch.bool)
    m[0, PLAY_MODE_LO:PLAY_MODE_LO + 5] = True            # play mode
    m[1, PLAY_MODE_LO + 2:PLAY_MODE_LO + 5] = True        # the deferred Ops choice: not a play mode
    m[2, REGION_LO:REGION_LO + 6] = True                  # Chernobyl's region
    m[3, A.DEFCON_VALUE_OFFSET:A.DEFCON_VALUE_OFFSET + 3] = True   # a DEFCON value
    m[4, A.BRANCH_OFFSET] = True                          # a one-option branch: nothing to floor
    m[5, A.NODE_OFFSET:A.NODE_OFFSET + 10] = True         # country targets
    m[6, 0:7] = True                                      # card choice
    assert floor_rows(m).tolist() == [True, False, True, True, False, False, False]
    assert event_choice_rows(m).tolist() == [False, False, True, True, False, False, False]


def test_the_floor_samples_the_mixture_and_returns_its_log_prob() -> None:
    g = torch.Generator().manual_seed(0)
    n, eps = 200_000, 0.3
    mask = torch.zeros(n, 220, dtype=torch.bool)
    mask[:, PLAY_MODE_LO:PLAY_MODE_LO + 5] = True
    logits = torch.full((n, 220), -1e9)
    logits[:, PLAY_MODE_LO:PLAY_MODE_LO + 5] = torch.tensor([3.0, 0.0, 1.0, -2.0, 0.5])
    lp = torch.log_softmax(logits, -1)
    pi_a = torch.multinomial(lp.exp(), 1, generator=g).squeeze(1)
    rows = torch.ones(n, dtype=torch.bool)
    rows[: n // 2] = False                                # half the rows off the floor
    acts, out_lp, took = apply_floor(pi_a, lp, mask, rows, eps, generator=g)

    pi = lp[0, PLAY_MODE_LO:PLAY_MODE_LO + 5].exp()
    mu = (1 - eps) * pi + eps / 5
    on = acts[n // 2:] - PLAY_MODE_LO
    freq = torch.bincount(on, minlength=5).double() / on.numel()
    se = (mu.double() * (1 - mu.double()) / on.numel()).sqrt()
    assert bool(((freq - mu.double()).abs() < 5 * se).all()), (freq, mu)
    # off the floor: the pi sample and its log pi, untouched
    assert torch.equal(acts[: n // 2], pi_a[: n // 2])
    assert torch.allclose(out_lp[: n // 2], lp.gather(1, pi_a[:, None]).squeeze(1)[: n // 2])
    assert not bool(took[: n // 2].any())
    # on the floor: log mu of the action actually taken
    assert torch.allclose(out_lp[n // 2:], mu.log()[on], atol=1e-5)
    assert abs(float(took[n // 2:].double().mean()) - eps) < 0.01


def test_the_floor_schedule() -> None:
    assert floor_eps(0.0, 10**9, 0, None, 0) == 0.0
    assert floor_eps(0.03, 99, 100, None, 0) == 0.0
    assert floor_eps(0.03, 100, 100, None, 0) == 0.03
    assert floor_eps(0.03, 150, 100, 200, 100) == 0.03
    assert floor_eps(0.03, 250, 100, 200, 100) == pytest.approx(0.015)
    assert floor_eps(0.03, 400, 100, 200, 100) == 0.0


def test_a_rollout_stores_the_mixture_log_prob_on_floor_rows_and_log_pi_elsewhere() -> None:
    eps = 0.5
    t = _trainer(buffer=64, play_mode_floor=eps)
    m = t.collect_rollouts()
    buf = t.buffer
    obs = buf.obs.reshape(-1, buf.obs.shape[-1]).float()
    masks = buf.masks.reshape(-1, buf.masks.shape[-1])
    acts = buf.actions.reshape(-1).long()
    with torch.no_grad():
        lp_all = torch.log_softmax(t.active_net(obs, masks)[0].float(), -1)
    lp = lp_all.gather(1, acts[:, None]).squeeze(1)
    fr = floor_rows(masks)
    assert int(fr.sum()) > 10, "too few floor rows to test anything"
    n_legal = masks.float().sum(1)
    want = torch.where(fr, torch.log((1 - eps) * lp.exp() + eps / n_legal), lp)
    assert torch.allclose(buf.log_probs.reshape(-1).float(), want, atol=1e-4)
    t.train_step()                                        # and it trains
    assert True


def test_the_floor_metrics_and_validation() -> None:
    t = _trainer(play_mode_floor=0.2)
    t.train_iteration()
    m = t.train_iteration()
    assert m["floor_eps"] == pytest.approx(0.2)
    assert 0.0 < m["floor_row_frac"] <= 1.0
    with pytest.raises(ValueError):
        _trainer(play_mode_floor=1.0)
    with pytest.raises(ValueError, match="seed-scenarios"):
        _trainer(play_mode_floor=0.1, floor_scope="seeded")


def test_off_the_floor_and_seeding_leave_a_rollout_bit_identical() -> None:
    # each trainer runs straight after it is built: _trainer reseeds torch, and a rollout draws
    # from torch's global RNG, so building both first would start the second from another state
    a = _trainer()
    a.collect_rollouts()
    b = _trainer(play_mode_floor=0.0, floor_from=5, floor_anneal_from=7, floor_anneal_steps=3)
    b.collect_rollouts()
    assert torch.equal(a.buffer.actions, b.buffer.actions)
    assert torch.equal(a.buffer.log_probs, b.buffer.log_probs)


# ----------------------------------------------------------------------------- 1b: seeding

def _random_play_with(seeder: ScenarioSeeder, envs: int, steps: int, seed: int) -> List[tuple]:
    """Random legal play with the seeder applied, as the trainer applies it; returns every forced
    (env, action, decision_type) in order."""
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    rng = np.random.default_rng(seed)
    _o, mk, _ = env.reset_all()
    seeder.draw(np.arange(envs), 0)
    out: List[tuple] = []
    for _ in range(steps):
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        acts = torch.tensor([int(rng.choice(np.flatnonzero(r))) for r in mk])
        types = [env.runner.get_state(e).ctx().decision_type for e in range(envs)]
        forced = seeder.apply(acts, np.asarray(mk), dp, env.runner, bool(env._auto_advance))
        for e in np.flatnonzero(forced):
            assert dp[e] == int(ts.Player.US)
            out.append((int(e), int(acts[e]), types[e]))
        _o, mk, _r, dones, _i = env.step(acts.numpy())
        seeder.draw(np.flatnonzero(np.asarray(dones) > 0.5), 0)
    return out


def test_seeding_forces_the_precursor_plays_the_engine_expects() -> None:
    seeder = ScenarioSeeder(["subs", "chernobyl"], 1.0, num_envs=64)
    forced = _random_play_with(seeder, envs=64, steps=1500, seed=7)
    assert seeder.forced["subs"] > 0 and seeder.forced["chernobyl"] > 0
    by_env: dict[int, List[tuple]] = {}
    for e, a, d in forced:
        by_env.setdefault(e, []).append((a, d))
    for seq in by_env.values():
        i = 0
        while i < len(seq):
            a, d = seq[i]
            assert d == ts.DecisionType.SELECT_CARD and a in (SCENARIOS["subs"].card - 1,
                                                               SCENARIOS["chernobyl"].card - 1)
            assert seq[i + 1] == (EVENT_SLOT, ts.DecisionType.SELECT_PLAY_MODE)
            if a == SCENARIOS["chernobyl"].card - 1:
                region, dt = seq[i + 2]
                assert dt == ts.DecisionType.CHOOSE_BRANCH and region >= REGION_LO
                i += 3
            else:
                i += 2


def test_an_unseeded_game_is_never_forced() -> None:
    seeder = ScenarioSeeder(["subs", "chernobyl"], 1.0, num_envs=16, start_step=10**12)
    assert _random_play_with(seeder, envs=16, steps=600, seed=3) == []
    assert seeder.games[0] == 0


def test_seeded_plays_are_stored_as_the_environments() -> None:
    t = _trainer(buffer=64, envs=16, seed_scenarios=["subs", "chernobyl"], seed_frac=1.0)
    learner_zero: List[torch.Tensor] = []
    for _ in range(12):
        t.collect_rollouts()
        z = t.buffer.learner.reshape(-1) < 0.5
        if bool(z.any()):
            learner_zero.append(t.buffer.actions.reshape(-1)[z].long())
        if sum(t.seeder.forced.values()) >= 2 and learner_zero:  # type: ignore[union-attr]
            break
    assert learner_zero, "no forced play in 12 rollouts"
    acts = torch.cat(learner_zero)
    allowed = {SCENARIOS["subs"].card - 1, SCENARIOS["chernobyl"].card - 1, EVENT_SLOT,
               *range(REGION_LO, REGION_LO + 6)}
    # without a pool every other row is the learner's, so learner = 0 is exactly the forced rows
    assert set(acts.tolist()) <= allowed
    t.train_step()


def test_seeding_validation() -> None:
    with pytest.raises(ValueError):
        ScenarioSeeder(["nope"], 0.1, 4)
    with pytest.raises(ValueError):
        _trainer(seed_frac=0.1)
    with pytest.raises(ValueError):
        _trainer(seed_scenarios=["subs"], seed_frac=0.0)
