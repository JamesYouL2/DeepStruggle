"""The trunk norm (`--ladder-trunk-norm`): a LayerNorm on the residual blocks' output, so the heads read a bounded
input however large the stream grows; a module only when on, recorded in the config and recovered from the weights,
so every earlier checkpoint is unchanged."""

import pytest
import torch

from ai.models.coldwar_net_v2 import create_like
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=2, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True, res_dropout=0.0)


def _trunk(net: torch.nn.Module, obs: torch.Tensor) -> torch.Tensor:
    seen: dict[str, torch.Tensor] = {}
    head = net.play_mode_head_net
    assert isinstance(head, torch.nn.Module)
    hook = head.register_forward_hook(lambda mod, inp, out: seen.update(x=inp[0]))
    with torch.no_grad():
        net(obs, torch.ones(obs.shape[0], 220, dtype=torch.bool))
    hook.remove()
    return seen["x"][:, :-110]                                 # the trunk output, without the active-card bits


def test_the_heads_read_a_normalised_trunk_however_large_the_stream_grows() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**CFG, "trunk_norm": True}).eval()
    for block in net.res_blocks:                               # a stream grown 100x, as A22's did
        assert isinstance(block, torch.nn.Module)
        for p in block.parameters():
            p.data.mul_(100.0)
    h = _trunk(net, torch.randn(16, 3824))
    assert torch.allclose(h.mean(-1), torch.zeros(16), atol=1e-4)
    assert torch.allclose(h.pow(2).mean(-1).sqrt(), torch.ones(16), atol=1e-2)


def test_off_by_default_and_recorded_only_when_on() -> None:
    off = create_ladder_net("cpu", **CFG)
    assert off.trunk_norm is None and "trunk_norm" not in off.ladder_config()
    assert not any(k.startswith("trunk_norm.") for k in off.state_dict())
    assert "trunk_norm" not in (ladder_config_from_state_dict(off.state_dict()) or {})
    on = create_ladder_net("cpu", **{**CFG, "trunk_norm": True})
    assert on.ladder_config()["trunk_norm"] is True
    assert (ladder_config_from_state_dict(on.state_dict()) or {}).get("trunk_norm") is True
    assert create_like(on).trunk_norm is not None


def test_the_same_seed_builds_the_same_weights_with_or_without_it() -> None:
    torch.manual_seed(3)
    off = create_ladder_net("cpu", **CFG).state_dict()
    torch.manual_seed(3)
    on = create_ladder_net("cpu", **{**CFG, "trunk_norm": True}).state_dict()
    assert all(torch.equal(v, on[k]) for k, v in off.items())


def test_refused_without_residual_blocks() -> None:
    with pytest.raises(ValueError, match="trunk_norm"):
        create_ladder_net("cpu", **{**CFG, "num_res_blocks": 0, "trunk_norm": True})
