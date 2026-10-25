"""Did Cell A's region attention learn placement? (P33, Cell A = architecture A14.)

Two reads on the searched-leak bank's placement positions (decision type POINT_NODE):

1. **The layer's contribution** (a network with `region_attn` only): the policy's country logits with the
   layer on against the layer bypassed -- mean |change| of the legal logits, relative to their spread, the
   KL between the two policies, and how often the top choice changes.
2. **The better country's place under each network's policy**, on the bank's labelled placement pairs
   (the player's move against a move the playouts found better): its rank among the legal moves, its
   probability against the played move's, and how often it is the policy's top choice -- split by whether
   the better country is in the played country's region (the owner's suspicion: "the wrong country in the
   right region").

    PYTHONPATH=.:build/release python -m ai.eval.placement_probe --bank bank.jsonl.gz \\
        --playouts confirm.jsonl.gz --screen screen.jsonl.gz --checkpoint a14_swa.pt control_swa.pt
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

import ts_engine as ts
from ai.eval.leak_critic import LeakPair, load_pairs, load_screen_pairs
from ai.models.ladder_net import region_bounds
from bindings.action_encoder import ActionEncoder
from bindings.ts_env import model_obs_features
from tools.lib.player_agent import NeuralAgent


@dataclass(frozen=True)
class _Placement:
    """How the model reads one placement pair."""
    #: Rank of the better country among the model's choices (1 = top).
    rank: float
    p_better: float
    p_played: float
    #: 1.0 when the better country is the model's top choice.
    better_is_top: float


def _region(country: int, bounds: Sequence[tuple[int, int]]) -> int:
    return next(r for r, (lo, hi) in enumerate(bounds) if lo <= country < hi)


def _logits(model: Any, device: torch.device, pairs: Sequence[LeakPair], bypass: bool = False) -> np.ndarray:
    """The policy's masked logits at each pair's position, the region layer bypassed if asked."""
    feats = model_obs_features(model)
    obs = np.stack([np.asarray(ts.extract_observation_features(p.state, p.mover, feats)) for p in pairs])
    masks = np.stack([np.asarray(ts.get_flat_action_mask(p.state, False)) for p in pairs])
    layer = getattr(model, "region_attn", None)
    saved = None
    if bypass and layer is not None:
        saved = layer.forward
        layer.forward = lambda x: x                      # type: ignore[method-assign]
    try:
        out = []
        with torch.no_grad():
            for lo in range(0, len(pairs), 1024):
                o = torch.from_numpy(obs[lo:lo + 1024].astype(np.float32)).to(device)
                m = torch.from_numpy(masks[lo:lo + 1024]).to(device)
                out.append(model(o, m)[0].float().cpu().numpy())
    finally:
        if saved is not None and layer is not None:
            layer.forward = saved                        # type: ignore[method-assign]
    lg = np.concatenate(out)
    return np.where(masks > 0, lg, -np.inf)


def contribution(model: Any, device: torch.device, pairs: Sequence[LeakPair]) -> str:
    on = _logits(model, device, pairs)
    off = _logits(model, device, pairs, bypass=True)
    legal = np.isfinite(on)
    d, spread, kl, flips = [], [], [], 0
    for i in range(len(pairs)):
        a, b = on[i][legal[i]], off[i][legal[i]]
        d.append(float(np.abs(a - b).mean()))
        spread.append(float(a.std()) if a.size > 1 else 0.0)
        pa = np.exp(a - a.max()); pa /= pa.sum()
        pb = np.exp(b - b.max()); pb /= pb.sum()
        kl.append(float((pa * (np.log(pa + 1e-12) - np.log(pb + 1e-12))).sum()))
        flips += int(np.argmax(a) != np.argmax(b))
    return (f"   region layer on vs bypassed, {len(pairs)} placement positions: mean |logit change| "
            f"{np.mean(d):.3f} (legal logits' spread {np.mean(spread):.2f}), KL {np.mean(kl):.4f}, "
            f"top choice changes at {100 * flips / max(1, len(pairs)):.1f}%")


def placement(model: Any, device: torch.device, pairs: Sequence[LeakPair], bounds: Sequence[tuple[int, int]]) -> str:
    lg = _logits(model, device, pairs)
    rows: Dict[str, List[_Placement]] = {"all": [], "same region": [], "other region": []}
    for i, p in enumerate(pairs):
        z = lg[i]
        order = np.argsort(-z)
        rank = int(np.flatnonzero(order == p.best)[0]) + 1
        pr = np.exp(z - z[np.isfinite(z)].max()); pr[~np.isfinite(z)] = 0.0; pr /= pr.sum()
        cb, cp = p.best - ActionEncoder.NODE_OFFSET, p.played - ActionEncoder.NODE_OFFSET
        same = 0 <= cb < 84 and 0 <= cp < 84 and _region(cb, bounds) == _region(cp, bounds)
        rec = _Placement(float(rank), float(pr[p.best]), float(pr[p.played]), float(order[0] == p.best))
        rows["all"].append(rec)
        rows["same region" if same else "other region"].append(rec)
    out = []
    for k, v in rows.items():
        if not v:
            continue
        mean_rank = np.asarray([r.rank for r in v]).mean()
        p_better = np.asarray([r.p_better for r in v]).mean()
        p_played = np.asarray([r.p_played for r in v]).mean()
        top = np.asarray([r.better_is_top for r in v]).mean()
        out.append(f"   {k:13s} n={len(v):4d}  rank of the better country {mean_rank:5.2f}  "
                   f"P(better) {p_better:.3f}  P(played) {p_played:.3f}  better is top {100 * top:5.1f}%")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--playouts", required=True)
    ap.add_argument("--screen", required=True)
    ap.add_argument("--min-z", type=float, default=2.0)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bounds = region_bounds()
    confirmed = [p for p in load_pairs(a.bank, a.playouts) if p.decision_type == "POINT_NODE"]
    screened = [p for p in load_screen_pairs(a.bank, a.screen, a.min_z)
                if p.decision_type == "POINT_NODE" and p.label > 0]
    print(f"placement pairs: {len(confirmed)} confirmed leaks, {len(screened)} screened pairs with a better move")
    for ck in a.checkpoint:
        model = NeuralAgent.from_checkpoint(ck, device=str(dev)).model.eval()
        print(f"== {ck}")
        if getattr(model, "region_attn", None) is not None:
            print(contribution(model, dev, confirmed + screened))
        print("  confirmed leaks:")
        print(placement(model, dev, confirmed, bounds))
        print("  screened pairs (the alternative better by the playouts):")
        print(placement(model, dev, screened, bounds))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
