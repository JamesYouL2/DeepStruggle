"""Post-LN residual blocks (`--ladder-res-post-norm`): a LayerNorm after every residual add keeps the stream bounded
however large the blocks' weights grow; a module only when on, recorded in the config and recovered from the
weights, and drawing no random numbers, so every earlier checkpoint and seeded init is unchanged."""

import pytest
import torch

from ai.models.coldwar_net_v2 import ResBlock, create_like
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=3, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True, res_dropout=0.0)


def _stream_rms(net: torch.nn.Module) -> float:
    torch.manual_seed(2)
    extract = getattr(net, "extract_features")
    with torch.no_grad():
        return float(extract(torch.randn(32, 3824)).pow(2).mean().sqrt())


def _grow(net: torch.nn.Module) -> None:
    for block in net.get_submodule("res_blocks").children():
        assert isinstance(block, ResBlock)
        for lin in (block.fc1, block.fc2):
            lin.weight.data.mul_(100.0)


def test_the_stream_stays_bounded_however_large_the_blocks_grow() -> None:
    torch.manual_seed(0)
    post = create_ladder_net("cpu", **{**CFG, "res_post_norm": True}).eval()
    torch.manual_seed(0)
    pre = create_ladder_net("cpu", **CFG).eval()
    _grow(post)
    _grow(pre)
    assert _stream_rms(post) < 1.5                  # renormalised at every block
    assert _stream_rms(pre) > 50.0                  # the pre-LN sum grows with the blocks' weights


def test_off_by_default_recorded_only_when_on_and_recovered_from_the_weights() -> None:
    off = create_ladder_net("cpu", **CFG)
    assert not any(".ln_post." in k for k in off.state_dict())
    assert "res_post_norm" not in off.ladder_config()
    assert "res_post_norm" not in (ladder_config_from_state_dict(off.state_dict()) or {})
    on = create_ladder_net("cpu", **{**CFG, "res_post_norm": True})
    assert on.ladder_config()["res_post_norm"] is True
    assert (ladder_config_from_state_dict(on.state_dict()) or {}).get("res_post_norm") is True
    blocks = [b for b in create_like(on).res_blocks if isinstance(b, ResBlock)]
    assert len(blocks) == 3 and all(b.ln_post is not None for b in blocks)


def test_the_same_seed_builds_the_same_weights_with_or_without_it() -> None:
    torch.manual_seed(3)
    off = create_ladder_net("cpu", **CFG).state_dict()
    torch.manual_seed(3)
    on = create_ladder_net("cpu", **{**CFG, "res_post_norm": True}).state_dict()
    assert all(torch.equal(v, on[k]) for k, v in off.items())


def test_refused_without_residual_blocks() -> None:
    with pytest.raises(ValueError, match="res_post_norm"):
        create_ladder_net("cpu", **{**CFG, "num_res_blocks": 0, "res_post_norm": True})
