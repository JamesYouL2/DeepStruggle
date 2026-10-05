"""The per-entity heads add the trunk context's term once per sample instead of concatenating the
context onto every entity. That is an exact rewrite, Linear(cat[x, ctx]) = x Wx^T + ctx Wc^T + b,
so it must give the concatenated form's logits and gradients from the same weights -- to float64
rounding, across every head layout that reads a context -- and change nothing where there is none."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.models.ladder_net import LadderNet, create_ladder_net
from bindings.action_encoder import ActionEncoder as A

BASE: Dict[str, Any] = dict(
    input_mode="grouped", aggregation="flatten", entity_dim=16, entity_proj_dim=64,
    card_self_attention=False, cross_attention=False, per_entity_heads=16, head_static=True,
    identity_dim=0, card_lookup=False, card_lookup_heads=0, card_lookup_dim=0,
    card_lookup_identity_dim=0, drop_static=True, hidden_dim=64, num_res_blocks=0,
    num_attn_heads=4, categorical_value=False)


def _concatenated_policy_logits(net: LadderNet, h: torch.Tensor,
                                tokens: Tuple[torch.Tensor, ...]) -> torch.Tensor:
    """The form the heads had before: the context expanded and concatenated onto every entity."""
    base = net.policy_head(h)
    h_board, board_nodes, h_cards, card_nodes = tokens
    parts_country, parts_card = [h_board, board_nodes], [h_cards, card_nodes]
    if net.head_context:
        assert net.pe_trunk is not None
        ctx = net.pe_trunk(h).unsqueeze(1)
        parts_country.append(ctx.expand(-1, 84, -1))
        parts_card.append(ctx.expand(-1, 110, -1))

    def out(head: nn.Module | None, x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        if head is None:
            return like * 0.0
        assert isinstance(head, nn.Sequential)
        f = head[:-1](x)
        if net.head_center:
            f = f - f.mean(dim=1, keepdim=True)
        return head[-1](f).squeeze(-1)

    card = out(net.pe_card, torch.cat(parts_card, -1), base[:, :A.PLAY_MODE_OFFSET])
    country = out(net.pe_country, torch.cat(parts_country, -1), base[:, A.NODE_OFFSET:A.BRANCH_OFFSET])
    return base + torch.cat([card, base[:, A.PLAY_MODE_OFFSET:A.NODE_OFFSET] * 0.0, country,
                             base[:, A.BRANCH_OFFSET:] * 0.0], dim=-1)


def _net(**kw: Any) -> LadderNet:
    torch.manual_seed(0)
    net = create_ladder_net("cpu", **BASE, **kw).double()
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():                       # the output layers start at zero; make them live
        for p in net.parameters():
            p.add_(0.3 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return net


@pytest.mark.parametrize("entities", ["country", "card", "both"])
@pytest.mark.parametrize("context", [True, False])
@pytest.mark.parametrize("center", [True, False])
def test_the_shared_context_head_is_the_concatenated_head(entities: str, context: bool,
                                                          center: bool) -> None:
    import ts_engine as ts
    net = _net(head_entities=entities, head_context=context, head_center=center)
    obs = torch.randn(16, ts.OBS_SIZE, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    h, _attn, tokens = net._encode(obs)
    assert tokens is not None
    got = net._policy_logits(h, tokens)
    want = _concatenated_policy_logits(net, h, tokens)
    assert torch.allclose(got, want, rtol=0, atol=1e-10)

    # The same gradient into every parameter, through the same loss.
    w = torch.randn(got.shape, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    params = [p for p in net.parameters() if p.requires_grad]
    g_got = torch.autograd.grad((got * w).sum(), params, retain_graph=True, allow_unused=True)
    g_want = torch.autograd.grad((want * w).sum(), params, allow_unused=True)
    for a, b in zip(g_got, g_want):
        assert (a is None) == (b is None)
        if a is not None and b is not None:
            assert torch.allclose(a, b, rtol=0, atol=1e-9)


def test_the_full_forward_matches_the_concatenated_head() -> None:
    """Through `forward`, as training and the rollout call it: same log-probabilities."""
    import ts_engine as ts
    net = _net(head_entities="country", head_context=True, head_center=True)
    net.eval()
    obs = torch.randn(8, ts.OBS_SIZE, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    mask = torch.ones(8, 220, dtype=torch.bool)
    with torch.no_grad():
        got = F.log_softmax(net(obs, mask)[0], -1)
        h, _attn, tokens = net._encode(obs)
        assert tokens is not None
        want = F.log_softmax(_concatenated_policy_logits(net, h, tokens), -1)
    assert torch.allclose(got, want, rtol=0, atol=1e-10)
