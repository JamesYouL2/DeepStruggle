"""`inspect_checkpoint` must report a LadderNet as a LadderNet.

A rung also carries `fusion_in.*`, so without the LadderNet check first every current checkpoint
is misreported as a ColdWarNet (V1) -- and a cross-attention rung's `lad_cross_attn.*` contains the
substring `cross_attn`, so it would match the V2 test before reaching the V1 one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from ai.models.ladder_net import create_ladder_net
from tools.lib.checkpoint_utils import inspect_checkpoint

GROUPED: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=0, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)

# The same rung shape with the cross-attention block, whose `lad_cross_attn.*` keys contain the
# substring `cross_attn` and would otherwise match the V2 test.
CROSS_ATTENTION: Dict[str, Any] = {
    **GROUPED,
    "input_mode": "entity", "aggregation": "pool", "entity_dim": 16,
    "drop_static": False, "cross_attention": True,
}


def _save_checkpoint(config: Dict[str, Any], tmp_path: Path) -> str:
    """Builds a small LadderNet from `config` and saves its state dict under `tmp_path`."""
    net = create_ladder_net("cpu", **config)
    path: Path = tmp_path / "checkpoint.pt"
    torch.save(net.state_dict(), str(path))
    return str(path)


def test_a_grouped_ladder_is_reported_as_a_laddernet(tmp_path: Path) -> None:
    checkpoint: str = _save_checkpoint(GROUPED, tmp_path)
    info: Dict[str, Any] = inspect_checkpoint(checkpoint)
    assert info["arch"] == "LadderNet (grouped, hidden 64)"


def test_a_cross_attention_ladder_is_not_mistaken_for_v2(tmp_path: Path) -> None:
    checkpoint: str = _save_checkpoint(CROSS_ATTENTION, tmp_path)
    sd: Dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert any(k.startswith("lad_cross_attn.") for k in sd)
    info: Dict[str, Any] = inspect_checkpoint(checkpoint)
    assert info["arch"] == "LadderNet (entity, hidden 64)"
