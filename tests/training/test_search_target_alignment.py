"""P15-X4b: a search target must describe the position it is stored against.

The trainer queues the positions it searches while the rollout runs (`_queue_search_targets`, which
reads the runner's *current* state) and answers them all when the rollout ends
(`_flush_search_targets`), filing each answer into the buffer row it was queued against. The row
holds s_t only while the queueing happens **before** `env.step`; taken afterwards the runner
already holds s_{t+1}, and every target describes the position after the observation it is stored
with.

The targets do not generally become *illegal* when this happens -- consecutive decisions share
most of their legal set -- they become legal and wrong, which is worse, because a mask cannot
catch it. It does not raise or warn; it trains the policy toward noise with a real gradient behind
it, and the only symptom is at the training-curve level: two 20M-step arms collapsed (entropy
1.03 -> 0.36, critic_auc to chance at 0.505 where the step-matched control holds 0.87) before the
cause was found.

So the test is behavioural, through the real `collect_rollouts`: a stand-in teacher answers with
an action derived from the exact observation of the position it was handed, and every stored
target must be the action derived from the observation stored in its own row.
"""

from __future__ import annotations

import hashlib
from typing import Any, List, Tuple

import numpy as np
import torch
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.training import NashPGTrainer
from bindings import TsVectorizedEnv


def _pick(obs: np.ndarray, mask: np.ndarray) -> int:
    """A legal action determined by the exact bytes of an observation, so that two different
    positions almost never give the same answer and the same position always does."""
    legal = np.flatnonzero(mask)
    h = int.from_bytes(hashlib.blake2b(np.ascontiguousarray(obs, dtype=np.float32).tobytes(),
                                       digest_size=8).digest(), "little")
    return int(legal[h % len(legal)])


class _ObservationTeacher:
    """Searches every decision; answers with `_pick` of the position it is handed."""

    def should_search(self, state: Any) -> bool:
        return True

    def run(self, states: List[Any]) -> List[Tuple[List[int], List[float]]]:
        out = []
        for st in states:
            mover = st.ctx().decision_player
            if mover == ts.Player.NONE:
                mover = st.phasing_player
            obs = np.asarray(ts.extract_observation(st, mover), dtype=np.float32)
            mask = np.asarray(ts.get_flat_action_mask(st))
            out.append(([_pick(obs, mask)], [1.0]))
        return out


def _trainer() -> NashPGTrainer:
    torch.manual_seed(0)
    dev = torch.device("cpu")
    net = create_ladder_net(dev, input_mode="grouped", aggregation="flatten", entity_dim=16,
                            entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                            per_entity_heads=16, head_context=True, head_static=True,
                            head_entities="country", head_center=True, identity_dim=0,
                            drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
                            card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
                            card_lookup_identity_dim=0, categorical_value=False)
    env = TsVectorizedEnv(num_envs=8, base_seed=4242)
    t = NashPGTrainer(active_net=net, env=env, num_envs=8, buffer_size=24, lr=3e-4, eta=0.1,
                      ref_update_freq=500, cuda_graphs=False, device=dev, search_ce_coef=0.5,
                      search_sims=4, search_subsample=1.0, search_node_filter="all")
    t._searcher = _ObservationTeacher()  # type: ignore[assignment]
    return t


def test_every_target_describes_the_position_in_its_own_row() -> None:
    t = _trainer()
    for _ in range(2):                                    # the second rollout overwrites the first
        t.collect_rollouts()
        b = t.buffer
        flagged = torch.nonzero(b.has_search > 0.5).tolist()
        assert len(flagged) > 50, "the fixture searched almost nothing; it is not testing anything"
        for step, env in flagged:
            obs = b.obs[step, env].cpu().numpy()
            mask = b.masks[step, env].cpu().numpy()
            target = int(torch.argmax(b.search_pi[step, env]))
            assert float(b.search_pi[step, env].sum()) == 1.0
            assert target == _pick(obs, mask), (
                f"the target in row (step {step}, env {env}) answers a different position than "
                f"the observation stored there")


def test_the_check_can_tell_neighbouring_positions_apart() -> None:
    """Without this the test above could pass with a teacher that answers anything: the answer for
    a row's position must differ from the answer for the next row's position, or a target filed
    one step late would match too."""
    t = _trainer()
    t.collect_rollouts()
    b = t.buffer
    differ = same = 0
    for step in range(b.obs.shape[0] - 1):
        for env in range(b.obs.shape[1]):
            if float(b.dones[step, env]) > 0.5:
                continue
            here = _pick(b.obs[step, env].cpu().numpy(), b.masks[step, env].cpu().numpy())
            later = _pick(b.obs[step + 1, env].cpu().numpy(), b.masks[step, env].cpu().numpy())
            differ += here != later
            same += here == later
    assert differ > 4 * same, (differ, same)
