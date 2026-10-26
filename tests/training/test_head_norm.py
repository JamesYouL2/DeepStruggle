"""LayerNorms in the heads that have none (`--ladder-head-norm`): off by default and then the forward is the old one;
on, the per-entity, branch and play-mode heads are invariant to the scale of their first layer (what lets their units
drift dead); the norms are registered last, recorded in the config and recovered from the weights, so a state written
without them resumes into them with its optimiser moments mapped."""

import torch

from ai.models.coldwar_net_v2 import create_like
from ai.models.ladder_net import create_ladder_net, ladder_config_from_state_dict
from ai.training.generic_trainer import _load_allowing_added_heads, _optimizer_state_with_added

CFG = dict(input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
           card_self_attention=False, cross_attention=False, per_entity_heads=16,
           head_context=True, head_static=True, head_entities="country", head_center=True,
           identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0, num_attn_heads=4,
           card_lookup=False, card_lookup_heads=0, card_lookup_dim=0, card_lookup_identity_dim=0,
           categorical_value=False, branch_head=True, play_mode_head=True)


def _logits(net: torch.nn.Module, seed: int = 4) -> torch.Tensor:
    torch.manual_seed(seed)
    obs, masks = torch.randn(8, 3824), torch.ones(8, 220, dtype=torch.bool)
    with torch.no_grad():
        return net(obs, masks)[0].float()


def _nonzero_last_layers(net: torch.nn.Module) -> None:
    for name in ("branch_head_net", "play_mode_head_net"):         # zero-initialised: give them an output to compare
        last = net.get_submodule(f"{name}.2")
        assert isinstance(last, torch.nn.Linear)
        torch.nn.init.normal_(last.weight, std=0.1)


def test_off_by_default_and_the_forward_is_unchanged() -> None:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **CFG)
    assert getattr(net, "head_norms") is None and "head_norm" not in net.ladder_config()
    assert not any(k.startswith("head_norms.") for k in net.state_dict())
    _nonzero_last_layers(net)
    x = torch.randn(4, 64 + 110)
    for name in ("branch_head_net", "play_mode_head_net"):
        head = net.get_submodule(name)
        assert torch.allclose(getattr(net, "_mlp_head")(name, x), head(x))


def test_the_heads_are_invariant_to_their_first_layers_scale() -> None:
    torch.manual_seed(1)
    net = create_ladder_net("cpu", **{**CFG, "head_norm": True}).eval()
    _nonzero_last_layers(net)
    before = _logits(net)
    for name in ("pe_country", "branch_head_net", "play_mode_head_net"):
        first = net.get_submodule(f"{name}.0")
        assert isinstance(first, torch.nn.Linear)
        first.weight.data.mul_(100.0)
        first.bias.data.mul_(100.0)
    assert torch.allclose(_logits(net), before, atol=1e-3)


def test_registered_last_recorded_recovered_and_resumable() -> None:
    torch.manual_seed(2)
    on = create_ladder_net("cpu", **{**CFG, "head_norm": True})
    names = [n for n, _ in on.named_parameters()]
    norm_names = [n for n in names if n.startswith("head_norms.")]
    assert norm_names and names[-len(norm_names):] == norm_names
    assert on.ladder_config()["head_norm"] is True
    assert (ladder_config_from_state_dict(on.state_dict()) or {}).get("head_norm") is True
    assert getattr(create_like(on), "head_norms") is not None
    # A state written without the norms (R48's plateau) resumes into a network with them.
    torch.manual_seed(2)
    old = create_ladder_net("cpu", **CFG)
    opt_old = torch.optim.AdamW(old.parameters(), lr=1e-3)
    added = _load_allowing_added_heads(on, old.state_dict())
    assert set(added) == {k for k in on.state_dict() if k.startswith("head_norms.")}
    mapped = _optimizer_state_with_added(on, added, opt_old.state_dict())
    torch.optim.AdamW(on.parameters(), lr=1e-3).load_state_dict(mapped)
