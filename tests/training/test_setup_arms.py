"""P4 arms. A (--setup-block-lambda): lambda = 1 between consecutive setup placements, exactly the
normal recursion elsewhere; off is bitwise the old update, and eager equals graphed. B
(--inject-setup-only): the injector keeps only setup rows and trains the policy term alone."""

from __future__ import annotations

from typing import Any, NamedTuple

import pytest
import torch

from ai.training.rollout_buffer import RolloutBuffer, setup_phase_slot

T, N = 24, 16


class _Last(NamedTuple):
    last_v_win: torch.Tensor
    last_v_vp: torch.Tensor
    last_dones: torch.Tensor
    last_players: torch.Tensor


def _fill(b: RolloutBuffer, setup_len: int = 6, seed: int = 0) -> _Last:
    g = torch.Generator().manual_seed(seed)
    b.players.copy_(torch.where(torch.rand(T, N, generator=g) < 0.5, 1, -1).to(torch.int8))
    b.dones.zero_()
    b.turns.fill_(1)
    b.rewards.zero_()
    b.values_win.copy_(torch.rand(T, N, generator=g) * 2 - 1)
    b.vps.zero_()
    b.obs.zero_()
    b.obs[:, :, setup_phase_slot()] = 2.0 / 6.0            # action round
    b.obs[:setup_len, :, setup_phase_slot()] = 0.0         # the first steps are setup
    return _Last(torch.rand(N, generator=g), torch.zeros(N), torch.zeros(N, dtype=torch.bool),
                 torch.ones(N, dtype=torch.int8))


def _raw_adv(on: bool, lam: float = 0.9) -> torch.Tensor:
    b = RolloutBuffer(T, N, device="cpu")
    b.setup_block_lambda = on
    last = _fill(b)
    key = (1.0, lam, False, True, 4, False, on)
    b._gae_backward(key, last.last_v_win, last.last_v_vp, last.last_players)
    return b.advantages.clone()


def test_off_leaves_the_recursion_unchanged_and_on_changes_only_the_setup_block() -> None:
    off, on = _raw_adv(False), _raw_adv(True)
    assert torch.equal(off[6:], on[6:])                  # after the block: identical
    assert not torch.equal(off[:6], on[:6])              # inside: lambda 1


def test_inside_the_block_the_advantage_telescopes_to_the_first_real_position() -> None:
    lam = 0.9
    b = RolloutBuffer(T, N, device="cpu")
    b.setup_block_lambda = True
    last = _fill(b)
    b._gae_backward((1.0, lam, False, True, 4, False, True), last.last_v_win, last.last_v_vp,
                    last.last_players)
    a = b.advantages
    # step 5 is the block's last placement: ordinary lambda from there on
    # step t < 5: A_t = delta_t + sign_t * A_{t+1} (lambda = 1)
    v, p = b.values_win, b.players.float()
    for t in range(5):
        sign = p[t] * p[t + 1]
        delta = sign * v[t + 1] - v[t]
        assert torch.allclose(a[t], delta + sign * a[t + 1], atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_graphed_equals_eager_with_the_setup_block() -> None:
    eager = RolloutBuffer(T, N, device="cuda")
    graphed = RolloutBuffer(T, N, device="cuda")
    for b in (eager, graphed):
        b.setup_block_lambda = True
    graphed.graph_gae = True
    for seed in range(3):
        for b in (eager, graphed):
            last = _fill(b, seed=seed)
            b.compute_gae(last.last_v_win.cuda(), last.last_v_vp.cuda(), last.last_dones.cuda(),
                          last.last_players.cuda())
        assert torch.equal(eager.advantages, graphed.advantages)


def test_per_player_gae_is_refused() -> None:
    b = RolloutBuffer(T, N, device="cpu")
    b.setup_block_lambda = True
    last = _fill(b)
    with pytest.raises(ValueError):
        b.compute_gae(*last, per_player_gae=True)


def test_the_setup_only_injector_keeps_setup_rows_and_the_policy_term(tmp_path: Any) -> None:
    import glob
    from ai.models.ladder_net import create_ladder_net
    from ai.training.generic_trainer import _HumanInjector
    ds = sorted(glob.glob("/workspace/data/datasets/human_corpus_e4"))
    if not ds:
        pytest.fail("the human corpus dataset is missing: tools/build_human_dataset.py")
    net = create_ladder_net("cpu", input_mode="grouped", aggregation="flatten", entity_dim=16,
                            entity_proj_dim=64, card_self_attention=False, cross_attention=False,
                            per_entity_heads=16, head_context=True, head_static=True,
                            head_entities="country", identity_dim=0, card_lookup=False,
                            card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
                            drop_static=True, hidden_dim=64, num_res_blocks=1, num_attn_heads=4,
                            categorical_value=False, head_center=True)
    inj = _HumanInjector(ds[0], net, torch.device("cpu"), every=1, weight=1.0, batch_size=64,
                         setup_only=True)
    phase = inj._obs[inj.train_idx][:, setup_phase_slot()]
    assert len(inj.train_idx) > 1000 and float(abs(phase).max()) == 0.0
    assert inj.maybe_step(0) > 0.0
