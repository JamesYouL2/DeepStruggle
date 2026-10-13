"""ai/training/forced_openings.py: forced openings play exactly their scripts, as the environment's moves
(learner = 0), and the Marshall variant headlines Marshall Plan on turn 1 when the US holds it.
Deterministic: frac 1.0 forces every game, and the Marshall test puts the card in the US hand."""

import numpy as np
import torch

import ts_engine as ts
from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from bindings.ts_env import TsVectorizedEnv
from tools.lib.openings import MARSHALL_PLAN, NODE_OFFSET, OPENINGS

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)


def _trainer(us: list, ussr: list, envs: int = 2, buffer: int = 32) -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    return NashPGTrainer(active_net=create_ladder_net(dev, **CFG), env=TsVectorizedEnv(num_envs=envs, base_seed=11),
                         num_envs=envs, buffer_size=buffer, lr=3e-4, eta=0.1, ref_update_freq=500, cuda_graphs=False,
                         device=dev, forced_openings_frac=1.0, forced_openings_us=us, forced_openings_ussr=ussr)


def test_both_sides_play_their_scripts_as_environment() -> None:
    t = _trainer(["us_wg4_it3_ir2"], ["ussr_eg1_pl4_au1"])
    t.collect_rollouts()
    want = [NODE_OFFSET + c for c in OPENINGS["ussr_eg1_pl4_au1"]["USSR"]] + \
           [NODE_OFFSET + c for c in OPENINGS["us_wg4_it3_ir2"]["US"]]
    for e in range(2):
        acts = [int(a) for a in t.buffer.actions[:len(want), e]]
        assert acts == want
        assert not bool((t.buffer.learner[:len(want), e] > 0.5).any())     # the environment's moves
    assert t.opening_seeder is not None and t.opening_seeder.forced_moves == 2 * len(want)


def test_the_marshall_opening_headlines_marshall_plan_when_held() -> None:
    t = _trainer(["us_marshall"], [], envs=1)
    runner = t.env.runner
    st = runner.get_state(0)
    us_hand = [c for c in range(1, 111) if ts.in_hand_of(st.get_card_location(c), ts.Player.US)]
    if MARSHALL_PLAN not in us_hand:                     # swap it into the US hand
        loc = st.get_card_location(MARSHALL_PLAN)
        st.set_card_location(MARSHALL_PLAN, st.get_card_location(us_hand[0]))
        st.set_card_location(us_hand[0], loc)
    runner.set_state(0, st)
    runner.refresh_all()
    t._obs_np = np.array(runner.get_observations(), dtype=np.float32)
    t._masks_np = np.array(runner.get_action_masks())
    t.collect_rollouts()
    acts = [int(a) for a in t.buffer.actions[:, 0]]
    script = [NODE_OFFSET + c for c in OPENINGS["us_marshall"]["US"]]
    us_part = acts[6:6 + len(script)]                    # after the USSR's six placements, its own
    assert us_part == script
    assert MARSHALL_PLAN - 1 in acts[6 + len(script):], "Marshall Plan not headlined"
    k = acts.index(MARSHALL_PLAN - 1, 6 + len(script))
    assert float(t.buffer.learner[k, 0]) < 0.5           # the headline is the environment's too
