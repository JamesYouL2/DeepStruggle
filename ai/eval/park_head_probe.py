"""Is the main value head the limit at the turn-end park, or the trunk? (P33, the pre-deal critic.)

Greedy self-play games are played with the network; every turn-end park (after a turn's last action,
before the deal: `VectorizedBatchRunner.set_capture_predeal`) is recorded from both sides with the game's
result. On the network's frozen trunk (`extract_features`) a separate head is fitted to predict that side's
win at the park -- free, and as a correction to the main head's own logit -- with early stopping on
held-out games, and scored against the main head on other held-out games.

    PYTHONPATH=.:build/release python -m ai.eval.park_head_probe --checkpoint swa.pt [--games 2048]
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ts_engine as ts
from bindings.ts_env import model_obs_features
from ai.eval.loading import load_for_eval


def collect(model: nn.Module, device: torch.device, games: int, envs: int, seed: int
            ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    """(obs, side, game id, y) for every turn-end park of `games` greedy self-play games, each recorded
    from both sides; y is 1 where that side won the game, 0 where it lost, -1 for a draw."""
    feats = model_obs_features(model)
    obs: List[np.ndarray] = []
    side: List[int] = []
    gid: List[int] = []
    result: Dict[int, float] = {}
    played = 0
    batch = 0
    while played < games:
        n = min(envs, games - played)
        runner = ts.VectorizedBatchRunner(n, seed + batch * 7919)
        if feats:
            runner.set_obs_features([feats] * n, [feats] * n)
        runner.set_capture_predeal(True)
        done = np.zeros(n, dtype=bool)
        for _ in range(5000):
            done |= np.asarray(runner.get_terminals(), dtype=bool)
            if done.all():
                break
            o = torch.from_numpy(np.asarray(runner.get_observations(), dtype=np.float32)).to(device)
            m = torch.from_numpy(np.asarray(runner.get_action_masks())).to(device)
            with torch.no_grad():
                picks = model(o, m)[0].argmax(-1).cpu().numpy()
            runner.step_flat_all([0 if done[i] else int(picks[i]) for i in range(n)], True)
            flags, pus, pussr = (np.array(x) for x in runner.get_predeal())
            for i in np.flatnonzero(flags):
                for s, rows in ((1, pus), (-1, pussr)):
                    obs.append(rows[i].copy())
                    side.append(s)
                    gid.append(played + int(i))
        utils = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
        for i in range(n):
            if done[i]:
                result[played + i] = float(utils[i])
        played += n
        batch += 1
    g = np.asarray(gid)
    s = np.asarray(side)
    u = np.asarray([result.get(int(x), 0.0) for x in g])
    y = np.where(u == 0.0, -1.0, (u * s > 0).astype(np.float64))
    return torch.from_numpy(np.stack(obs)), s, g, y


def _features(model: nn.Module, device: torch.device, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(trunk vectors, main head's v_win) of `obs`, in chunks."""
    hs, vs = [], []
    with torch.no_grad():
        for lo in range(0, obs.shape[0], 4096):
            o = obs[lo:lo + 4096].to(device)
            hs.append(model.extract_features(o).float().cpu())  # type: ignore[operator]
            vs.append(model(o, None)[1].float().squeeze(-1).cpu())
    return torch.cat(hs), torch.cat(vs)


def _ll(p: torch.Tensor, y: torch.Tensor) -> float:
    q = p.clamp(1e-6, 1 - 1e-6)
    return float(-(y * q.log() + (1 - y) * (1 - q).log()).mean())


def fit_head(h: torch.Tensor, y: torch.Tensor, offset: torch.Tensor, train: np.ndarray, val: np.ndarray,
             seed: int, hidden: int = 256, max_epochs: int = 60, patience: int = 4) -> nn.Module:
    """A one-hidden-layer head on the frozen trunk, its logit added to `offset`; early-stopped on `val`."""
    torch.manual_seed(seed)
    out = nn.Linear(hidden, 1)
    nn.init.zeros_(out.weight)                     # starts at the offset: the main head's own prediction
    nn.init.zeros_(out.bias)
    head = nn.Sequential(nn.Linear(h.shape[1], hidden), nn.GELU(), out)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    tr, va = torch.from_numpy(train), torch.from_numpy(val)
    best, best_state, bad = math.inf, None, 0
    for _ in range(max_epochs):
        perm = tr[torch.randperm(tr.numel())]
        for lo in range(0, perm.numel(), 512):
            b = perm[lo:lo + 512]
            loss = F.binary_cross_entropy_with_logits(head(h[b]).squeeze(-1) + offset[b], y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = float(F.binary_cross_entropy_with_logits(head(h[va]).squeeze(-1) + offset[va], y[va]))
        if v < best - 1e-5:
            best, best_state, bad = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    return head


def report(name: str, model: nn.Module, device: torch.device, games: int, envs: int, seed: int) -> str:
    obs, side, gid, y_all = collect(model, device, games, envs, seed)
    keep = y_all >= 0                                                  # drawn games dropped
    h, v = _features(model, device, obs)
    h, v = h[keep], v[keep]
    y = torch.from_numpy(y_all[keep]).float()
    g = gid[keep]
    test, val = (g % 5) == 0, (g % 5) == 1
    train = ~(test | val)
    p_main = ((v + 1) / 2).clamp(1e-4, 1 - 1e-4)
    main_logit = torch.log(p_main / (1 - p_main))
    zero = torch.zeros_like(main_logit)
    free = fit_head(h, y, zero, np.flatnonzero(train), np.flatnonzero(val), seed)
    corr = fit_head(h, y, main_logit, np.flatnonzero(train), np.flatnonzero(val), seed + 1)
    te = torch.from_numpy(np.flatnonzero(test))
    with torch.no_grad():
        p_free = torch.sigmoid(free(h[te]).squeeze(-1))
        p_corr = torch.sigmoid(corr(h[te]).squeeze(-1) + main_logit[te])
    yt = y[te]
    se = float(np.sqrt(np.var(-(yt * p_main[te].log() + (1 - yt) * (1 - p_main[te]).log()).numpy()) / max(1, te.numel())))
    lines = [f"{name}: {int(keep.sum())} park rows from {games} greedy games ({int(train.sum())} train / "
             f"{int(val.sum())} val / {te.numel()} test, split by game)",
             f"   test log loss: main head {_ll(p_main[te], yt):.4f} (± {se:.4f}),  separate head on the trunk "
             f"{_ll(p_free, yt):.4f},  main head + a fitted correction {_ll(p_corr, yt):.4f}",
             f"   test Brier:    main head {float(((p_main[te] - yt) ** 2).mean()):.4f},  separate head "
             f"{float(((p_free - yt) ** 2).mean()):.4f},  main + correction {float(((p_corr - yt) ** 2).mean()):.4f}"]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", nargs="+", required=True)
    ap.add_argument("--games", type=int, default=2048)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--seed", type=int, default=6161)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for ck in a.checkpoint:
        model = load_for_eval(ck, str(dev), merged_ok=False).model
        print(report(ck, model, dev, a.games, a.envs, a.seed), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
