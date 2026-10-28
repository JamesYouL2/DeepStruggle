"""A network checkpoint loads through the architecture-detecting loader, in its own view.

NeuralBot (deleted) once picked V1 or V2 by weight name itself, so a LadderNet checkpoint -- every
E7 model -- was rebuilt as a V1 ColdWarNet and refused by load_state_dict, and
`tools/play_match.py` could not play a match between two current checkpoints. Networks now play
through `NeuralAgent.from_checkpoint`; these pin the same two properties on it: a ladder
checkpoint comes back bit for bit, and a checkpoint with an observation feature block is played
in that feature set rather than refused or misread. Built from freshly initialised networks saved
to a temp file, since no checkpoint is committed.
"""

from __future__ import annotations

import pathlib
from typing import Any, Dict

import numpy as np
import torch
import ts_engine as ts

from ai.models.ladder_net import create_ladder_net
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import observe
from tools.lib.player_agent import NeuralAgent

#: A small shallow-trunk rung (the E7 recipe's structure at toy width).
CFG: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=256,
    card_self_attention=False, cross_attention=False, per_entity_heads=64, head_context=True,
    head_static=True, head_entities="country", head_center=True, identity_dim=0, drop_static=True,
    hidden_dim=64, num_res_blocks=0, num_attn_heads=4, card_lookup=False, card_lookup_heads=0,
    card_lookup_dim=0, card_lookup_identity_dim=0, categorical_value=False)


def _decision_state() -> ts.GameState:
    st = ts.GameState()
    ts.Engine.init_game(st, 4242)
    ts.Engine.auto_advance_step(st)
    assert st.ctx().decision_player != ts.Player.NONE
    return st


def test_a_ladder_checkpoint_loads_bit_for_bit(tmp_path: pathlib.Path) -> None:
    net = create_ladder_net("cpu", **CFG)
    path = tmp_path / "snapshot_1000steps.pt"
    torch.save(net.state_dict(), path)
    agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
    assert type(agent.model).__name__ == "LadderNet"
    for name, t in net.state_dict().items():
        assert torch.equal(t, agent.model.state_dict()[name]), name


def test_a_feature_checkpoint_is_played_in_its_own_feature_set(tmp_path: pathlib.Path) -> None:
    bits = int(ts.OBS_FEATURE_OPS_BUDGET)
    net = create_ladder_net("cpu", **{**CFG, "obs_features": bits})
    path = tmp_path / "snapshot_1000steps.pt"
    torch.save(net.state_dict(), path)
    agent = NeuralAgent.from_checkpoint(str(path), device="cpu")
    assert agent.obs_features == bits
    st = _decision_state()
    player = st.ctx().decision_player
    assert observe(st, player, agent.obs_features).shape[-1] == ts.obs_size_for(bits) > ts.OBS_SIZE
    a = agent.select_action(st, player, temperature=0.0)
    assert np.asarray(ActionEncoder.get_legal_mask(st))[a]
