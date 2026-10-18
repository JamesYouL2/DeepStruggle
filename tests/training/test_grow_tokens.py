"""--resume-grow-tokens: token layers grown onto a trained network at identity. The grown network must
compute exactly what the saved one did, its zero weights must take gradient (so the path can learn), and
Adam's grown state must load and step."""

from __future__ import annotations

from typing import Any, Dict

import torch

import ts_engine as ts
from ai.models.grow import grow_tokens_optimizer_state, grow_tokens_state_dict
from ai.models.ladder_net import create_ladder_net
from tests.training.test_ladder_tokens import SHALLOW


def _randomise(net: torch.nn.Module) -> None:
    """Every parameter nonzero, the zero-initialised head outputs included, so a misplaced column shows."""
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        for p in net.parameters():
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)


def _pair() -> Dict[str, Any]:
    torch.manual_seed(0)
    old = create_ladder_net("cpu", **SHALLOW)
    _randomise(old)
    new = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=16)
    grown = grow_tokens_state_dict(old.state_dict(), new.state_dict(), 16, int(getattr(new, "identity_dim", 0)))
    new.load_state_dict(grown, strict=True)
    obs = torch.randn(32, ts.OBS_SIZE) * 0.5
    return {"old": old.eval(), "new": new.eval(), "obs": obs}


def test_the_grown_network_computes_what_the_saved_one_did() -> None:
    p = _pair()
    with torch.no_grad():
        a = p["old"](p["obs"], None)
        b = p["new"](p["obs"], None)
    for x, y in zip(a, b):
        assert torch.allclose(x, y, atol=1e-5), float((x - y).abs().max())


def test_the_zero_weights_take_gradient() -> None:
    p = _pair()
    net = p["new"].train()
    logits, v, _ = net(p["obs"], None)
    (logits.square().mean() + v.square().mean()).backward()
    fusion = net.fusion_in[0].weight
    assert float(fusion.grad[:, -16:].abs().sum()) > 0


def test_adams_grown_state_loads_and_steps() -> None:
    torch.manual_seed(1)
    old = create_ladder_net("cpu", **SHALLOW)
    opt = torch.optim.AdamW(old.parameters(), lr=1e-3)
    obs = torch.randn(16, ts.OBS_SIZE) * 0.5
    old(obs, None)[0].square().mean().backward()
    opt.step()
    new = create_ladder_net("cpu", **SHALLOW, token_layers=1, token_dim=16)
    names = [n for n, _ in new.named_parameters()]
    new.load_state_dict(grow_tokens_state_dict(old.state_dict(), new.state_dict(), 16, 0), strict=True)
    opt2 = torch.optim.AdamW(new.parameters(), lr=1e-3)
    opt2.load_state_dict(grow_tokens_optimizer_state(opt.state_dict(), names, 16, 0))
    new(obs, None)[0].square().mean().backward()
    opt2.step()
    st = opt2.state_dict()["state"]
    fusion_idx = names.index("fusion_in.0.weight")
    assert st[fusion_idx]["exp_avg"].shape == new.fusion_in[0].weight.shape
