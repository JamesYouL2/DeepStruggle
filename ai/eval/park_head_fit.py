"""A network with its value head refitted at the turn-end park, everything else frozen (P33).

The owner's test of whether the pre-deal critic needs training at all: if a value head fitted on a
frozen trunk gives the pre-deal search (`rollout:...:predeal:<leaf checkpoint>`) what R50's training
does, a head fit is enough. Greedy self-play games of the network are played and every turn-end park
recorded from both sides with the game's result (`park_head_probe.collect`); a copy of the network's
`val_win_head` -- starting from its own weights -- is fitted to predict each side's win there (BCE on
(v + 1) / 2), early-stopped on held-out games, and the whole network is saved with the refitted head, so
the rollout root loads it as its leaf model.

    PYTHONPATH=.:build/release python -m ai.eval.park_head_fit --checkpoint swa.pt --output swa+parkhead.pt
"""

from __future__ import annotations

import argparse
import copy
import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from ai.eval.park_head_probe import collect
from tools.lib.player_agent import NeuralAgent


def _bce(v: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    p = ((v + 1) / 2).clamp(1e-6, 1 - 1e-6)
    return -(y * p.log() + (1 - y) * (1 - p).log()).mean()


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--games", type=int, default=8192)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7171)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
    obs, _side, gid, y_all = collect(model, dev, a.games, a.envs, a.seed)
    keep = y_all >= 0
    obs, gid = obs[keep], gid[keep]
    y = torch.from_numpy(y_all[keep]).float()
    with torch.no_grad():
        h = torch.cat([model.extract_features(obs[i:i + 4096].to(dev)).float().cpu()  # type: ignore[operator]
                       for i in range(0, obs.shape[0], 4096)])
    head = copy.deepcopy(getattr(model, "val_win_head")).cpu().float().train()
    test, val = (gid % 5) == 0, (gid % 5) == 1
    tr = torch.from_numpy(np.flatnonzero(~(test | val)))
    va, te = torch.from_numpy(np.flatnonzero(val)), torch.from_numpy(np.flatnonzero(test))
    with torch.no_grad():
        base_te = float(_bce(head(h[te]).squeeze(-1), y[te]))
    opt = torch.optim.Adam(head.parameters(), lr=a.lr)
    best, best_state, bad = math.inf, copy.deepcopy(head.state_dict()), 0
    torch.manual_seed(a.seed)
    for _ in range(60):
        perm = tr[torch.randperm(tr.numel())]
        for lo in range(0, perm.numel(), 512):
            b = perm[lo:lo + 512]
            loss = _bce(head(h[b]).squeeze(-1), y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(_bce(head(h[va]).squeeze(-1), y[va]))
        if v < best - 1e-5:
            best, best_state, bad = v, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= 4:
                break
    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        fit_te = float(_bce(head(h[te]).squeeze(-1), y[te]))
    print(f"{int(keep.sum())} park rows from {a.games} greedy games; test log loss at the park: "
          f"the network's own head {base_te:.4f}, refitted head {fit_te:.4f}")
    state = {k: t.detach().cpu() for k, t in model.state_dict().items()}
    for k, t in head.state_dict().items():
        state[f"val_win_head.{k}"] = t.detach().cpu().to(state[f"val_win_head.{k}"].dtype)
    torch.save(state, a.output)
    print(f"saved {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
