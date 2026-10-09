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
