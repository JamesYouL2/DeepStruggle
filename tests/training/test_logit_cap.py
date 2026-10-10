"""--ladder-logit-cap: no legal move's logit falls more than C below the top legal move.

Uncapped, the policy saturates (a coup led influence by 28 nats; P(coup) = 1 in float32), and policy
gradient, scaled by 1 - P, cannot move it. The cap is monotonic, so greedy play is unchanged."""

from __future__ import annotations

from typing import Any, Dict

import torch

from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.training.generic_trainer import _load_allowing_added_heads

M2D: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=32,
    card_self_attention=False, cross_attention=False, per_entity_heads=8, head_context=True,
    head_static=True, head_entities="country", identity_dim=0, card_lookup=False,
    card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0, drop_static=True,
    hidden_dim=32, num_res_blocks=1, num_attn_heads=4, categorical_value=False, head_center=True)


def _nets(cap: float) -> tuple:
    torch.manual_seed(0)
    plain = create_ladder_net("cpu", **M2D).eval()
    capped = create_ladder_net("cpu", **M2D, logit_cap=cap).eval()
    _load_allowing_added_heads(capped, plain.state_dict())         # same weights, cap added
    return plain, capped


def test_the_cap_bounds_deficits_and_keeps_the_greedy_move() -> None:
    plain, capped = _nets(7.0)
    g = torch.Generator().manual_seed(1)
    obs = torch.randn(16, int(plain.TOTAL_OBS_SIZE), generator=g)
    mask = torch.rand(16, 220, generator=g) > 0.5
    with torch.no_grad():
        raw = plain(obs, mask)[0]
        cap = capped(obs, mask)[0]
    raw = raw * 50.0                                                   # make the gaps large
    with torch.no_grad():
        cap_big = capped._cap_logits(raw, mask).masked_fill(~mask, -1e9)
    assert torch.equal(raw.masked_fill(~mask, -1e9).argmax(1), cap_big.argmax(1))
    top = cap_big.masked_fill(~mask, float("-inf")).amax(1, keepdim=True)
    deficit = (top - cap_big).masked_fill(~mask, 0.0)
    assert float(deficit.max()) <= 7.0 + 1e-4
    assert torch.equal(cap.argmax(1), plain(obs, mask)[0].argmax(1))


def test_the_cap_is_recorded_in_the_weights_and_off_by_default() -> None:
    plain, capped = _nets(5.0)
    assert "logit_cap" in capped.state_dict() and "logit_cap" not in plain.state_dict()
    cfg = ladder_config_from_state_dict(capped.state_dict())
    assert cfg is not None and cfg["logit_cap"] == 5.0                # a checkpoint names its cap
    plain_cfg = ladder_config_from_state_dict(plain.state_dict())
    assert plain_cfg is not None and "logit_cap" not in plain_cfg     # uncapped configs unchanged
    x = torch.zeros(2, 220)
    assert torch.equal(plain._cap_logits(x, None), x)


def test_a_straight_through_cap_plays_the_same_and_passes_the_raw_gradient() -> None:
    """--ladder-logit-cap-grad straight: forward identical to the tanh cap; the gradient on a
    saturated alternative is the softmax's (~p), not attenuated by the tanh's derivative."""
    torch.manual_seed(0)
    nets = [create_ladder_net("cpu", **M2D, logit_cap=7.0, logit_cap_grad=g).eval() for g in ("tanh", "straight")]
    z = torch.tensor([[34.0, 6.0, -2.6]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    grads, outs = [], []
    for net in nets:
        zz = z.clone().requires_grad_(True)
        out = net._cap_logits(zz, mask)
        torch.log_softmax(out, -1)[0, 0].backward()
        assert zz.grad is not None
        grads.append(zz.grad[0, 1].item())
        outs.append(out.detach())
    assert torch.allclose(outs[0], outs[1])
    p_inf = float(torch.softmax(outs[0], -1)[0, 1])
    assert abs(grads[0]) < 1e-5 and abs(grads[1] + p_inf) < 1e-6
    assert ladder_config_from_state_dict(nets[1].state_dict()) is not None
    assert nets[1].ladder_config()["logit_cap_grad"] == "straight"


def test_a_leaky_cap_plays_the_same_and_passes_a_share_of_the_raw_gradient() -> None:
    torch.manual_seed(0)
    tanh, leaky = (create_ladder_net("cpu", **M2D, logit_cap=7.0, logit_cap_leak=l).eval() for l in (0.0, 0.1))
    z = torch.tensor([[34.0, 6.0, -2.6]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    out = []
    for net in (tanh, leaky):
        zz = z.clone().requires_grad_(True)
        o = net._cap_logits(zz, mask)
        torch.log_softmax(o, -1)[0, 0].backward()
        assert zz.grad is not None
        out.append((o.detach(), zz.grad[0, 1].item()))
    assert torch.allclose(out[0][0], out[1][0])
    p_inf = float(torch.softmax(out[0][0], -1)[0, 1])
    assert abs(out[1][1] - (out[0][1] - 0.1 * p_inf)) < 1e-6        # tanh's own plus 0.1 x raw
    assert leaky.ladder_config()["logit_cap_leak"] == 0.1


def test_an_upward_leak_raises_saturated_moves_and_never_pushes_them_down() -> None:
    """--ladder-logit-cap-leak-up: the raw gradient only where descent would raise a logit."""
    torch.manual_seed(0)
    tanh, up = (create_ladder_net("cpu", **M2D, logit_cap=7.0, logit_cap_leak_up=l).eval() for l in (0.0, 1.0))
    z = torch.tensor([[34.0, 6.0, -2.6]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    for sign in (1.0, -1.0):          # loss = -sign * log P(coup): sign 1 rewards the coup, -1 punishes it
        g = []
        for net in (tanh, up):
            zz = z.clone().requires_grad_(True)
            (-sign * torch.log_softmax(net._cap_logits(zz, mask), -1)[0, 0]).backward()
            assert zz.grad is not None
            g.append(zz.grad[0, 1].item())
        if sign > 0:                  # rewarding the coup would lower influence: blocked, tanh's only
            assert abs(g[1] - g[0]) < 1e-9
        else:                         # punishing the coup raises influence: the full softmax gradient
            assert g[1] < g[0] - 5e-4
    assert up.ladder_config()["logit_cap_leak_up"] == 1.0
