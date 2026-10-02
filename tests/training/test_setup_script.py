"""--setup-script-frac: a drawn fraction of games is set up by a scripted human opening, and those
placements are trained as the policy's own, credited with the game result (owner, 2026-10-02)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
import ts_engine as ts

from ai.models import create_coldwar_net
from ai.training import NashPGTrainer
from tools.lib.openings import HUMAN_OPENING_MIX, NODE_OFFSET, OPENINGS, play_scripted_setup


def _trainer(shallow: bool = False, **kw: Any) -> NashPGTrainer:
    from bindings.ts_env import TsVectorizedEnv
    torch.manual_seed(0)
    dev = torch.device("cpu")
    if shallow:   # the production trunk: no residual blocks, so no dropout
        from ai.models.ladder_net import create_ladder_net
        model: Any = create_ladder_net(dev, input_mode="grouped", aggregation="flatten", entity_dim=16,
                                       entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                                       per_entity_heads=16, head_context=True, head_static=True,
                                       head_entities="country", head_center=True, identity_dim=0,
                                       drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
                                       card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                                       card_lookup_identity_dim=0, categorical_value=False)
    else:
        model = create_coldwar_net(dev)
    env = TsVectorizedEnv(num_envs=8, base_seed=123)
    # 16 steps from a fresh game: almost all of them are setup placements.
    return NashPGTrainer(active_net=model, env=env, num_envs=8, buffer_size=16, lr=3e-4, eta=0.1,
                         ref_update_freq=500, cuda_graphs=False, device=dev, **kw)


def test_the_human_variants_set_up_the_owners_boards() -> None:
    """Final influence (US/USSR): EG 4, Poland 4, Yugoslavia or Austria 1; WG 4 with Italy 3 +
    Iran 3 or Italy 4 + Iran 2."""
    want = {
        "human": {"Yugoslavia": (0, 1), "Austria": (0, 0), "Italy": (3, 0), "Iran": (3, 0)},
        "human_yugo_it4": {"Yugoslavia": (0, 1), "Austria": (0, 0), "Italy": (4, 0), "Iran": (2, 0)},
        "human_austria_it3": {"Yugoslavia": (0, 0), "Austria": (0, 1), "Italy": (3, 0), "Iran": (3, 0)},
        "human_austria_it4": {"Yugoslavia": (0, 0), "Austria": (0, 1), "Italy": (4, 0), "Iran": (2, 0)},
    }
    assert set(HUMAN_OPENING_MIX) == set(want)
    for name, extra in want.items():
        st = ts.GameState()
        ts.Engine.init_game(st, 11)
        c = play_scripted_setup(st, name).to_dict()["countries"]
        board = {n: (c[n]["us_influence"], c[n]["ussr_influence"]) for n in
                 ("East Germany", "Poland", "West Germany", *extra)}
        assert board == {"East Germany": (0, 4), "Poland": (0, 4), "West Germany": (4, 0), **extra}, name


def test_off_is_bitwise_the_control() -> None:
    torch.manual_seed(1)
    a = _trainer(setup_mc_credit=True)
    a.train_iteration()
    torch.manual_seed(1)
    b = _trainer(setup_mc_credit=True, setup_script_frac=0.0)
    b.train_iteration()
    for pa, pb in zip(a.active_net.parameters(), b.active_net.parameters()):
        assert torch.equal(pa, pb)


def test_scripted_games_play_the_script_and_store_the_policys_own_log_prob() -> None:
    t = _trainer(setup_mc_credit=True, setup_mc_min_batch=10_000, setup_script_frac=1.0,
                 setup_script_openings=("human_austria_it4",))
    t.collect_rollouts()                      # no update yet: the stored log-probs are this network's
    script = {1: [NODE_OFFSET + c for c in OPENINGS["human_austria_it4"]["US"]],
              -1: [NODE_OFFSET + c for c in OPENINGS["human_austria_it4"]["USSR"]]}
    for i in range(8):
        recs = t._setup_pending[i]
        assert len(recs) == 15
        for side in (1, -1):
            assert [int(r[2]) for r in recs if r[5] == side] == script[side]
        # the stored log-prob is the network's own for that action, at that state
        with torch.no_grad():
            obs = torch.stack([r[0] for r in recs])
            masks = torch.stack([r[1] for r in recs])
            acts = torch.stack([r[2] for r in recs]).long()
            lp = torch.log_softmax(t.active_net(obs, masks)[0].float(), -1).gather(1, acts[:, None]).squeeze(1)
        stored = torch.stack([r[3] for r in recs]).float()
        assert torch.allclose(lp, stored, atol=1e-4)


def test_the_draw_is_per_game_and_covers_the_mix() -> None:
    t = _trainer(setup_mc_credit=True, setup_script_frac=0.5, setup_script_openings=HUMAN_OPENING_MIX)
    for _ in range(200):
        t._script_draw(np.arange(8))
    frac = t._script_games[0] / t._script_games.sum()
    assert 0.4 < frac < 0.6
    assert set(t._script_open[t._script_open >= 0].tolist()) <= set(range(len(HUMAN_OPENING_MIX)))


def test_it_refuses_what_it_cannot_do() -> None:
    with pytest.raises(ValueError, match="setup-mc-credit"):
        _trainer(setup_script_frac=0.5, setup_script_openings=HUMAN_OPENING_MIX)
    with pytest.raises(ValueError, match="setup-explore-frac"):
        _trainer(setup_mc_credit=True, setup_explore_frac=0.5, setup_script_frac=0.5,
                 setup_script_openings=HUMAN_OPENING_MIX)
    with pytest.raises(ValueError, match="both sides"):
        _trainer(setup_mc_credit=True, setup_script_frac=0.5, setup_script_openings=("us_e516_43",))


def _half_scripted(opening: str = "human") -> NashPGTrainer:
    """Envs 0-3 scripted, 4-7 the policy's own setup."""
    t = _trainer(shallow=True, setup_mc_credit=True, setup_mc_min_batch=1, setup_script_frac=0.5,
                 setup_script_openings=(opening,))
    t._script_open[:] = [0, 0, 0, 0, -1, -1, -1, -1]
    t.collect_rollouts()
    return t


def _ussr_scripted_lp(t: NashPGTrainer) -> "tuple[torch.Tensor, Any]":
    recs = [r for i in range(4) for r in t._setup_pending[i] if r[5] == -1]
    obs, masks = torch.stack([r[0] for r in recs]), torch.stack([r[1] for r in recs])
    acts = torch.stack([r[2] for r in recs]).long()

    def lp() -> torch.Tensor:
        with torch.no_grad():
            return torch.log_softmax(t.active_net(obs, masks)[0].float(), -1).gather(1, acts[:, None]).squeeze(1)
    return lp(), lp


def test_a_scripted_opening_that_outscores_the_own_one_gains_probability() -> None:
    t = _half_scripted()
    before, lp = _ussr_scripted_lp(t)
    # the scripted games the USSR won, its own games it lost
    t._setup_mc_resolve([{"env_idx": i, "terminal_utility": -1.0 if i < 4 else 1.0} for i in range(8)])
    t._setup_mc_update()
    assert float((lp() - before).mean()) > 0.0


def test_a_uniform_credit_moves_no_scripted_placement() -> None:
    """Every game won by the same side: centred per side the credit is zero everywhere, so a biased
    baseline can no longer push the scripted placements up (E7-11-44 adopted Romania 6 that way)."""
    t = _half_scripted("stupid_romania_australia")
    before, lp = _ussr_scripted_lp(t)
    t._setup_mc_resolve([{"env_idx": i, "terminal_utility": -1.0} for i in range(8)])
    t._setup_ready = [r[:4] + (torch.zeros(()),) + r[5:] for r in t._setup_ready]    # baseline 0
    out = t._setup_mc_update()
    assert out["setup_mc_adv_centre_ussr"] == 1.0
    assert float((lp() - before).abs().max()) < 1e-3


def test_a_stale_rollout_log_prob_does_not_skew_the_scripted_update() -> None:
    """Between a placement and its game's end the policy moves; for a scripted action at p ~ 1e-8
    that drift once put every ratio outside the clip. The update measures from the current policy."""
    t = _trainer(shallow=True, setup_mc_credit=True, setup_mc_min_batch=1, setup_script_frac=1.0,
                 setup_script_openings=("human",))
    t.collect_rollouts()
    t._setup_mc_resolve([{"env_idx": i, "terminal_utility": 1.0} for i in range(8)])
    with torch.no_grad():                         # the policy drifts after the placements were made
        for p in t.active_net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    out = t._setup_mc_update()
    assert out["setup_mc_ratio_dev"] < 1e-4 and out["setup_mc_clip_frac"] == 0.0


def test_the_stupid_control_opening_is_legal() -> None:
    st = ts.GameState()
    ts.Engine.init_game(st, 11)
    c = play_scripted_setup(st, "stupid_romania_australia").to_dict()["countries"]
    assert (c["Romania"]["ussr_influence"], c["West Germany"]["us_influence"], c["Australia"]["us_influence"]) == (6, 7, 6)
