"""The residual blocks' dropout (`--ladder-res-dropout`): at 0 a network with residual blocks computes the same policy
in train mode -- the PPO update's -- as in eval mode -- the rollouts' --, which the historical 0.05 does not; and the
setting is recorded in the config only off its default, so every earlier config is unchanged."""

import torch

from ai.models.coldwar_net_v2 import ResBlock, create_like
from ai.models.ladder_net import create_ladder_net

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=2, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False)


def _logits(net: torch.nn.Module, train: bool) -> torch.Tensor:
    torch.manual_seed(1)
    obs = torch.randn(32, 3824)
    masks = torch.ones(32, 220, dtype=torch.bool)
    net.train(train)
    with torch.no_grad():
        return net(obs, masks)[0]


def test_no_dropout_makes_the_update_see_the_policy_that_played() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **{**CFG, "res_dropout": 0.0})
    assert torch.equal(_logits(net, True), _logits(net, False))
    torch.manual_seed(0)
    old = create_ladder_net("cpu", **CFG)                     # the historical 0.05
    assert not torch.equal(_logits(old, True), _logits(old, False))


def test_the_setting_is_recorded_only_off_its_default_and_copies_keep_it() -> None:
    net = create_ladder_net("cpu", **{**CFG, "res_dropout": 0.0})
    assert net.ladder_config()["res_dropout"] == 0.0
    assert "res_dropout" not in create_ladder_net("cpu", **CFG).ladder_config()
    copy = create_like(net)
    blocks = [b for b in copy.res_blocks if isinstance(b, ResBlock)]
    assert len(blocks) == 2 and all(b.dropout.p == 0.0 for b in blocks)
