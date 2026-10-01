"""P30 C1: the card/country token path beside the grouped trunk."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch

from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict

SHALLOW: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


@pytest.mark.parametrize("layers,dim", [(0, 0), (1, 32), (2, 16)])
def test_the_config_is_recovered_from_the_weights(layers: int, dim: int) -> None:
    m = create_ladder_net("cpu", **SHALLOW, token_layers=layers, token_dim=dim)
    assert ladder_config_from_state_dict(m.state_dict()) == m.ladder_config()
    assert m.ladder_config()["token_layers"] == layers


def test_a_base_network_has_no_token_weights() -> None:
    m = create_ladder_net("cpu", **SHALLOW)
    assert not any(k.startswith("tok_") for k in m.state_dict())
    assert m.ladder_config()["token_dim"] == 0


def test_one_entity_reaches_another_through_attention() -> None:
    """Country 0's row changes country 1's and card 5's tokens -- which only attention can do --
    and those tokens are what their per-entity heads read."""
    torch.manual_seed(0)
    m = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=32).eval()
    x = torch.rand(1, m.TOTAL_OBS_SIZE)
    y = x.clone()
    y[0, 5] += 1.0                                 # a slot of country 0's row
    board = slice(0, m.BOARD_SIZE)
    cards = slice(m.CARD_OFFSET, m.CARD_OFFSET + m.CARD_SIZE)
    glob = slice(m.GLOBAL_OFFSET, m.GLOBAL_OFFSET + m.GLOBAL_SIZE)
    with torch.no_grad():
        tx = m._token_path(x[:, board], x[:, cards], x[:, glob], 1)
        ty = m._token_path(y[:, board], y[:, cards], y[:, glob], 1)
        assert not torch.allclose(tx[0, 1 + 1], ty[0, 1 + 1])          # country 1
        assert not torch.allclose(tx[0, 85 + 5], ty[0, 85 + 5])        # card 5
        _h, _a, tokens = m._encode(x)
    assert tokens is not None
    assert tokens[0].shape[-1] == 32 and torch.allclose(tokens[0][0, 1], tx[0, 2])


def test_tokens_need_the_grouped_trunk() -> None:
    with pytest.raises(ValueError):
        create_ladder_net("cpu", **{**SHALLOW, "input_mode": "flat", "per_entity_heads": 0,
                                    "head_entities": "both", "head_center": False},
                          token_layers=1, token_dim=16)
    with pytest.raises(ValueError):
        create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=0)


def test_the_cli_records_the_axis_only_when_on() -> None:
    from ai.training.train import _ladder_config, build_parser
    base = ["--arch", "ladder", "--ladder-input-mode", "grouped", "--ladder-aggregation", "flatten",
            "--ladder-entity-dim", "16", "--ladder-entity-proj-dim", "256", "--ladder-hidden-dim", "480",
            "--ladder-res-blocks", "0", "--drop-static", "--per-entity-heads", "64",
            "--ladder-head-context", "--ladder-head-static", "--ladder-head-entities", "country",
            "--ladder-head-center"]
    off = _ladder_config(build_parser().parse_args(base))
    on = _ladder_config(build_parser().parse_args(base + ["--ladder-token-layers", "1"]))
    assert off is not None and on is not None
    assert "token_layers" not in off
    assert on["token_layers"] == 1 and on["token_dim"] == 128
