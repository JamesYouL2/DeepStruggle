"""Growing a trained LadderNet in place: the grouped input projections made wider without changing what it plays.

`--resume-widen` (owner, 2026-10-13): the board and card projections (`lad_board`, `lad_card`: the whole
board and the whole card block each into `entity_proj_dim` numbers through one linear layer) are where
the trunk loses state (research/findings/training/trunk_depth_and_state.md), and a run resumed into a
model whose projections are k times wider continues from its state instead of restarting.

Each new unit is a copy of an old one (new unit i is old unit i mod p), and the fusion layer reads each
copy with 1/k of the old unit's weight. A LayerNorm over a vector tiled k times has the same mean and
variance, so every copy carries exactly the old unit's activation and the fusion layer's input sum is
unchanged: the widened network computes the same function. Copies with identical weights would get
identical gradients and never separate, so each copy's incoming weights get a perturbation that sums to
zero over its copies -- the output then moves only at second order.

Adam's moments follow the gradients: a copy's incoming weights and LayerNorm parameters see 1/k of the old
unit's gradient (its outgoing weight is 1/k), so their first moment is scaled by 1/k and the second by
1/k^2, which keeps each element's step the size it was; a fusion column sees the old column's gradient
and keeps its moments.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import torch

#: The grouped projections, in the order the fusion layer reads them (board, then cards).
PROJECTIONS = ("lad_board", "lad_card")
FUSION = "fusion_in.0.weight"


def widen_factor(saved: Mapping[str, torch.Tensor], target: Mapping[str, torch.Tensor]) -> int:
    """How many times wider the target's projections are than the saved ones; 1 when equal."""
    old = int(saved["lad_board.0.weight"].shape[0])
    new = int(target["lad_board.0.weight"].shape[0])
    if new % old or int(target["lad_card.0.weight"].shape[0]) != new or int(saved["lad_card.0.weight"].shape[0]) != old:
        raise ValueError(f"the target's projections ({new}) are not an integer multiple of the saved ones ({old}), "
                         "or the board and card projections differ in width")
    return new // old


def _tile_rows(t: torch.Tensor, k: int) -> torch.Tensor:
    return t.repeat((k,) + (1,) * (t.dim() - 1))


def _fusion_cols(w: torch.Tensor, p: int, k: int, scale: float) -> torch.Tensor:
    """Fusion weight columns: the board and card blocks tiled k times (times `scale`), the rest as they were."""
    board, card, rest = w[:, :p], w[:, p:2 * p], w[:, 2 * p:]
    return torch.cat([board.repeat(1, k) * scale, card.repeat(1, k) * scale, rest], dim=1)


def widen_state_dict(sd: Mapping[str, torch.Tensor], k: int, noise: float = 1e-3,
                     generator: Optional[torch.Generator] = None) -> Dict[str, torch.Tensor]:
    """The state dict of the same network with its grouped projections k times wider.

    `noise` is the perturbation's size relative to each projection weight's own standard deviation; the
    perturbation of a unit's k copies sums to zero. The same `generator` state gives the same perturbation,
    so a network and an identical reference widen identically.
    """
    if k < 1:
        raise ValueError("the widening factor must be >= 1")
    out = {name: t.clone() for name, t in sd.items()}
    if k == 1:
        return out
    p = int(sd["lad_board.0.weight"].shape[0])
    for proj in PROJECTIONS:
        w = sd[f"{proj}.0.weight"]
        tiled = _tile_rows(w, k)
        if noise > 0:
            eps = torch.randn((k,) + tuple(w.shape), generator=generator, dtype=torch.float32).to(w.device, w.dtype)
            eps = (eps - eps.mean(dim=0, keepdim=True)) * (noise * float(w.float().std()))
            tiled = tiled + eps.reshape(tiled.shape)
        out[f"{proj}.0.weight"] = tiled
        for leaf in (f"{proj}.0.bias", f"{proj}.1.weight", f"{proj}.1.bias"):
            out[leaf] = _tile_rows(sd[leaf], k)
    out[FUSION] = _fusion_cols(sd[FUSION], p, k, 1.0 / k)
    return out


def widen_optimizer_state(saved: Dict[str, Any], names: Sequence[str], p: int, k: int) -> Dict[str, Any]:
    """An Adam state dict for the widened network: `names` are the model's parameter names in
    `named_parameters()` order (the order the saved state indexes), `p` the saved projection width."""
    row_params = {f"{proj}.{leaf}" for proj in PROJECTIONS for leaf in ("0.weight", "0.bias", "1.weight", "1.bias")}
    state: Dict[Any, Any] = {}
    for idx, st in saved["state"].items():
        name = names[int(idx)]
        new_st = dict(st)
        for key, power in (("exp_avg", 1), ("exp_avg_sq", 2)):
            if key not in st:
                continue
            t = st[key]
            if name in row_params:
                new_st[key] = _tile_rows(t, k) / float(k ** power)
            elif name == FUSION:
                new_st[key] = _fusion_cols(t, p, k, 1.0)
        state[idx] = new_st
    return {"state": state, "param_groups": [dict(g) for g in saved["param_groups"]]}

