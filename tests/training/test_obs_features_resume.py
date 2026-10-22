"""Weights are loaded only into a model built for the view they were trained on.

The feature set is a buffer, so loading copies the saved value over the model's while the model goes
on building the view it was built with. Since E8 the default view is headline_reveal, so continuing an
E7 run without naming its view (`--obs-features` with no names) would otherwise train E7 weights on
E8's view -- and with two sets of one width (ops_budget against ops_budget + headline_reveal) nothing
else would notice.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from ai.training.generic_trainer import check_state_obs_features
from bindings.ts_env import model_obs_features

SMALL: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)
OPS = int(ts.OBS_FEATURE_OPS_BUDGET)
REVEAL = int(ts.OBS_FEATURE_HEADLINE_REVEAL)


def _net(features: int) -> torch.nn.Module:
    torch.manual_seed(0)
    return create_ladder_net("cpu", **{**SMALL, "obs_features": features})


def test_an_e7_state_is_refused_by_a_model_built_with_the_e8_default() -> None:
    with pytest.raises(ValueError, match="--obs-features"):
        check_state_obs_features(_net(REVEAL), _net(0).state_dict(), "the resume state")


def test_an_equal_width_set_is_refused_too() -> None:
    """ops_budget and ops_budget + headline_reveal are one width: only the check tells them apart."""
    built, saved = _net(OPS | REVEAL), _net(OPS)
    built.load_state_dict(saved.state_dict())          # loads without complaint...
    assert model_obs_features(built) == OPS | REVEAL   # ...and the model still builds the other view
    with pytest.raises(ValueError, match="ops_budget"):
        check_state_obs_features(_net(OPS | REVEAL), saved.state_dict(), "the resume state")


def test_weights_of_the_same_view_load() -> None:
    for features in (0, OPS, REVEAL, OPS | REVEAL):
        check_state_obs_features(_net(features), _net(features).state_dict(), "the resume state")
