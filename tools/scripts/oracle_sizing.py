#!/usr/bin/env python3
"""How much would an oracle critic know that the public critic cannot? (P5 sizing, offline.)

An oracle critic sees the opponent's hand during training (not the deck order, and not future
rolls: those are chance, not hidden information). Its value can only reduce the variance of the
policy's advantages by as much as the opponent's hand predicts the result beyond what the public
information already does. This measures that, before anything is built.

From a checkpoint's own self-play at temperature 1 (the training distribution), a sample of
decisions is recorded. Each record holds the network's trunk representation of the mover's
observation (`extract_features`), the opponent's hand (110 bits, from
`VectorizedBatchRunner.get_opponent_hands`), the phase, the turn, the checkpoint's own v_win, and
the game's result from the mover's side. Two heads of identical shape are then fitted on the same
records, one on the trunk representation alone and one with the opponent's hand appended. Both are
scored on held-out games, never held-out positions, and split by phase and turn. The checkpoint's
own critic is the reference.

    PYTHONPATH=.:build/release python tools/scripts/oracle_sizing.py --checkpoint <snap.pt> \\
        --games 4000 --record-prob 0.1 --output-md out.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _root not in sys.path:
    sys.path.insert(0, _root)

from bindings.ts_env import TsVectorizedEnv  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402

_GLOBAL = 84 * 26 + 110 * 14
_PHASE, _TURN = _GLOBAL + 9, _GLOBAL + 6
PHASES = {0: "setup", 1: "headline", 2: "action round"}


@torch.no_grad()
def collect(model: nn.Module, dev: torch.device, games: int, envs: int, record_prob: float,
            seed: int) -> Dict[str, np.ndarray]:
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    obs, masks, _ = env.reset_all()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    game_id = np.arange(envs, dtype=np.int64)
    next_game = envs
    pending: List[List[Dict[str, Any]]] = [[] for _ in range(envs)]
    out: List[Dict[str, Any]] = []
    done_games = 0
    while done_games < games:
        o_np = np.asarray(obs)
        o = torch.from_numpy(o_np).float().to(dev)
        m = torch.from_numpy(np.asarray(masks)).to(dev)
        logits, v, _ = model(o, m)
        acts = torch.multinomial(torch.softmax(logits.float(), -1), 1, generator=gen).squeeze(-1)
        dp = np.asarray(env.runner.get_decision_players(), dtype=np.int8)
        pick = np.flatnonzero((rng.random(envs) < record_prob) & (dp != 0))
        if len(pick):
            idx = torch.from_numpy(pick).to(dev)
            extract = getattr(model, "extract_features")
            h = extract(o.index_select(0, idx)).float().cpu().numpy().astype(np.float16)
            opp = np.asarray(env.runner.get_opponent_hands(dp.tolist()), dtype=np.float32).reshape(envs, 110)
            vv = v.float().reshape(-1).cpu().numpy()
            for j, i in enumerate(pick):
                pending[i].append({"h": h[j], "opp": opp[i].astype(np.uint8), "v": float(vv[i]),
                                   "mover": int(dp[i]), "game": int(game_id[i]),
                                   "phase": int(round(o_np[i, _PHASE] * 6)), "turn": int(round(o_np[i, _TURN] * 10))})
        obs, masks, _r, _d, info = env.step(acts.cpu().numpy())
        for ep in info.get("completed_episodes", []):
            i = int(ep["env_idx"])
            u = float(ep["terminal_utility"])                  # +1 US win, -1 USSR win, 0 draw
            for rec in pending[i]:
                rec["y"] = (u * rec["mover"] + 1.0) / 2.0      # mover's result: 1 win, 0.5 draw, 0 loss
                out.append(rec)
            pending[i] = []
            game_id[i] = next_game
            next_game += 1
            done_games += 1
    return {"h": np.stack([r["h"] for r in out]), "opp": np.stack([r["opp"] for r in out]),
            "v": np.array([r["v"] for r in out], np.float32), "y": np.array([r["y"] for r in out], np.float32),
            "game": np.array([r["game"] for r in out]), "phase": np.array([r["phase"] for r in out]),
            "turn": np.array([r["turn"] for r in out])}


def fit(x: np.ndarray, y: np.ndarray, games: np.ndarray, dev: torch.device, linear: bool = False,
        max_epochs: int = 40, seed: int = 0) -> nn.Module:
    """A head fitted with early stopping on held-out GAMES. The result is one label per game, so
    the effective sample is the number of games, and a wide head memorises them otherwise."""
    torch.manual_seed(seed)
    head: nn.Module = (nn.Linear(x.shape[1], 1) if linear else
                       nn.Sequential(nn.Linear(x.shape[1], 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1)))
    head = head.to(dev)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3 if linear else 1e-2)
    ug = np.unique(games)
    val_games = set(np.random.default_rng(seed + 1).choice(ug, size=max(1, len(ug) // 10), replace=False).tolist())
    va = np.array([g in val_games for g in games])
    xt, yt = torch.from_numpy(x[~va]).float().to(dev), torch.from_numpy(y[~va]).float().to(dev)
    xv, yv = torch.from_numpy(x[va]).float().to(dev), torch.from_numpy(y[va]).float().to(dev)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        head.train()
        perm = torch.randperm(len(yt), device=dev)
        for s in range(0, len(yt), 4096):
            b = perm[s:s + 4096]
            loss = F.binary_cross_entropy_with_logits(head(xt[b]).squeeze(-1), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            vl = float(F.binary_cross_entropy_with_logits(head(xv).squeeze(-1), yv))
        if vl < best - 1e-5:
            best, bad = vl, 0
            best_state = {k: t.detach().clone() for k, t in head.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                break
    if best_state is not None:
        head.load_state_dict(best_state)
    head.eval()
    return head


@torch.no_grad()
def predict(head: nn.Module, x: np.ndarray, dev: torch.device) -> np.ndarray:
    xt = torch.from_numpy(x).float().to(dev)
    return torch.cat([torch.sigmoid(head(xt[s:s + 65536]).squeeze(-1)) for s in range(0, len(x), 65536)]).cpu().numpy()


def scores(p: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    ll = float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())
    base = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
    ll0 = float(-(y * np.log(base) + (1 - y) * np.log(1 - base)).mean())
    brier = float(((p - y) ** 2).mean())
    return {"logloss": ll, "brier": brier, "r2": 1.0 - brier / float(((base - y) ** 2).mean()), "ll_skill": 1.0 - ll / ll0}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--record-prob", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--output-json", default=None)
    a = ap.parse_args(argv)
    dev = torch.device(a.device if torch.cuda.is_available() else "cpu")
    agent = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev))
    model = agent.model.eval()
    d = collect(model, dev, a.games, a.envs, a.record_prob, a.seed)
    games = np.unique(d["game"])
    rng = np.random.default_rng(a.seed)
    test_games = set(rng.choice(games, size=len(games) // 5, replace=False).tolist())
    te = np.array([g in test_games for g in d["game"]])
    tr = ~te
    # Both heads also get the checkpoint's own v_win, so neither starts below the critic it is
    # measured against; the only difference between them is the opponent's hand.
    h = np.concatenate([d["h"].astype(np.float32), d["v"][:, None]], axis=1)
    x_pub = h
    x_ora = np.concatenate([h, d["opp"].astype(np.float32)], axis=1)
    g_tr = d["game"][tr]
    p_pub = predict(fit(x_pub[tr], d["y"][tr], g_tr, dev), x_pub[te], dev)
    p_ora = predict(fit(x_ora[tr], d["y"][tr], g_tr, dev), x_ora[te], dev)
    # Low capacity: the critic's own logit, alone or with the 110 hand bits, fitted linearly --
    # whether the opponent's hand adds anything to the critic's own estimate.
    vl = np.log(np.clip((d["v"] + 1) / 2, 1e-4, 1 - 1e-4) / np.clip((1 - d["v"]) / 2, 1e-4, 1)).astype(np.float32)[:, None]
    l_pub = predict(fit(vl[tr], d["y"][tr], g_tr, dev, linear=True), vl[te], dev)
    x_lo = np.concatenate([vl, d["opp"].astype(np.float32)], axis=1)
    l_ora = predict(fit(x_lo[tr], d["y"][tr], g_tr, dev, linear=True), x_lo[te], dev)
    p_own = (d["v"][te] + 1.0) / 2.0
    y = d["y"][te]
    rows: List[Dict[str, Any]] = []

    def row(label: str, sel: np.ndarray) -> None:
        if sel.sum() < 200:
            return
        rows.append({"slice": label, "n": int(sel.sum()), "own": scores(p_own[sel], y[sel]),
                     "public": scores(p_pub[sel], y[sel]), "oracle": scores(p_ora[sel], y[sel]),
                     "lin_public": scores(l_pub[sel], y[sel]), "lin_oracle": scores(l_ora[sel], y[sel])})

    row("all", np.ones(len(y), dtype=bool))
    for ph, name in PHASES.items():
        row(name, d["phase"][te] == ph)
    t = d["turn"][te]
    for lo, hi in ((1, 1), (2, 3), (4, 5), (6, 7), (8, 10)):
        row(f"turn {lo}–{hi}" if lo != hi else f"turn {lo}", (t >= lo) & (t <= hi))
    lines = [f"{agent.name}: {len(d['y'])} decisions from {len(games)} self-play games at temperature 1 "
             f"(record prob {a.record_prob}); {int(te.sum())} held-out decisions from {len(test_games)} games.",
             "",
             "Skill = 1 − Brier / Brier of the base rate (the share of the result's variance explained).",
             "",
             "| slice | n | checkpoint's critic | MLP public | MLP + opp hand | Δ | linear on critic | linear + opp hand | Δ |",
             "|:---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['slice']} | {r['n']} | {r['own']['r2']:.3f} | {r['public']['r2']:.3f} | "
                     f"{r['oracle']['r2']:.3f} | **{r['oracle']['r2'] - r['public']['r2']:+.3f}** | "
                     f"{r['lin_public']['r2']:.3f} | {r['lin_oracle']['r2']:.3f} | "
                     f"**{r['lin_oracle']['r2'] - r['lin_public']['r2']:+.3f}** |")
    print("\n".join(lines))
    if a.output_md:
        open(a.output_md, "w").write("\n".join(lines) + "\n")
    if a.output_json:
        json.dump(rows, open(a.output_json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
