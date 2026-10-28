"""P36c follow-up: is *whether* the rollout root departs from the network predictable from what the network sees?

The students did not learn the root's corrections on held-out positions (P36c), but the network's own second
choice is the root's move at half of its departures, so most of the question is *when* to depart. This probe
asks that alone: a small classifier on the frozen network's trunk features (what it represents) and on the raw
observation (what it is shown), trained on the training games' positions to predict whether the root departs
there, scored by held-out AUC. Well above 0.5: there are patterns, and the move-level fits failed for another
reason; near 0.5: whether to depart is not visible in the observation at this scale.

    PYTHONPATH=.:build/release python -m ai.eval.departure_probe --dataset targets.npz --worlds 128 --checkpoint swa.pt
    PYTHONPATH=.:build/release python -m ai.eval.departure_probe --dataset targets.npz --target h4 --checkpoint swa.pt

`--target` reads any teacher of a `tools/distill_targets.py` set by name (the teacher ladder); `--worlds W` is
P36c's teacher at W worlds.
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.eval.belief_probe import _auc
from ai.training.policy_fit import masks_of
from ai.training.value_fit import Codes
from tools.lib.engine_fingerprint import check_raw_states
from ai.eval.loading import load_for_eval


def fit_classifier(x_tr: torch.Tensor, y_tr: torch.Tensor, x_va: torch.Tensor, y_va: torch.Tensor,
                   epochs: int, seed: int) -> nn.Module:
    """One hidden layer, early-stopped on the validation games' loss."""
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(x_tr.shape[1], 256), nn.GELU(), nn.Linear(256, 1)).to(x_tr.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, state = float("inf"), None
    for _ in range(epochs):
        perm = torch.randperm(x_tr.shape[0], device=x_tr.device)
        for lo in range(0, perm.numel(), 2048):
            i = perm[lo:lo + 2048]
            loss = F.binary_cross_entropy_with_logits(net(x_tr[i]).squeeze(-1), y_tr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = F.binary_cross_entropy_with_logits(net(x_va).squeeze(-1), y_va).item()
        if v < best:
            best, state = v, {k: t.clone() for k, t in net.state_dict().items()}
    if state is not None:
        net.load_state_dict(state)
    return net.eval()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--target", default=None, help="a teacher's name in the set (overrides --worlds)")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_for_eval(a.checkpoint, str(dev), merged_ok=False).model
    check_raw_states(a.dataset)
    z = np.load(a.dataset)
    held = z["heldout"].astype(bool)
    name = a.target or str(a.worlds)
    dep = (z[f"acts_{name}"][:, 0].astype(np.int64) != z["net_argmax"].astype(np.int64)).astype(np.float32)
    codes = Codes(int(getattr(model, "obs_feature_bits", 0)), dev)   # the network's own observation
    x_codes = codes.encode(z["states"])
    masks = masks_of(z["states"])
    feats: Dict[str, np.ndarray] = {}
    trunk, raw = [], []
    for lo in range(0, x_codes.shape[0], 8192):
        obs = codes.decode(torch.from_numpy(x_codes[lo:lo + 8192]).to(dev))
        with torch.no_grad():
            h = getattr(model, "_encode")(obs)[0].float()
            lg = model(obs, torch.from_numpy(masks[lo:lo + 8192]).to(dev))[0].float()
            lg = lg.masked_fill(torch.from_numpy(masks[lo:lo + 8192]).to(dev) == 0, float("-inf"))
            top2 = torch.topk(lg, 2, dim=-1).values
            gap = (top2[:, 0] - top2[:, 1]).clamp(max=50.0)[:, None]
        trunk.append(torch.cat([h, gap], dim=1).cpu())
        raw.append(obs.cpu())
    feats["trunk features + the network's top-2 logit gap"] = torch.cat(trunk).numpy()
    feats["raw observation"] = torch.cat(raw).numpy()
    feats["the network's top-2 logit gap alone"] = feats["trunk features + the network's top-2 logit gap"][:, -1:]
    rng = np.random.default_rng(a.seed)
    tr_games = np.unique(z["game"][~held])
    va_games = set(rng.choice(tr_games, size=max(1, tr_games.size // 9), replace=False).tolist())
    va = (~held) & np.isin(z["game"], list(va_games))
    tr = (~held) & ~va
    print(f"{int(tr.sum()):,} train / {int(va.sum()):,} validation / {int(held.sum()):,} held-out positions; the root "
          f"departs at {100 * dep[held].mean():.1f}% of held-out positions (teacher {name})")
    for name, x in feats.items():
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        xt = torch.from_numpy((x - mu) / sd).float().to(dev)
        yt = torch.from_numpy(dep).to(dev)
        net = fit_classifier(xt[torch.from_numpy(tr).to(dev)], yt[torch.from_numpy(tr).to(dev)],
                             xt[torch.from_numpy(va).to(dev)], yt[torch.from_numpy(va).to(dev)], a.epochs, a.seed)
        with torch.no_grad():
            p_ho = torch.sigmoid(net(xt[torch.from_numpy(held).to(dev)]).squeeze(-1)).cpu().numpy()
            p_tr = torch.sigmoid(net(xt[torch.from_numpy(tr).to(dev)]).squeeze(-1)).cpu().numpy()
        print(f"  {name:48s} held-out AUC {_auc(p_ho, dep[held]):.3f}  (training AUC {_auc(p_tr, dep[tr]):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
