#!/usr/bin/env python3
"""Should --playout-adv redeal the hidden cards per pair? The label noise, split into hand and dice.

`ai.training.playout_advantage` labels a decision by playing each candidate from P paired copies of
the state. With the hidden cards kept as dealt (`hidden="true"`), all P pairs share one opponent hand,
so that hand's share of the noise never averages out; redealing per pair (`hidden="redeal"`) averages
it, but replaces the real belief over the unseen cards with a uniform one.

From the model's own self-play this samples decisions of the playout kinds, and plays every
candidate from H redealt hands x D dice (a nested design: the candidates of one cell share hand and
dice) and from the true hand x D dice. Per candidate advantage (value minus the mean over the
candidates):

* sigma2_dice -- variance between dice within a hand; sigma2_hand -- variance between hands of the
  hand means, less the dice share;
* signal -- variance of the advantage itself, the noise of its H x D estimate removed;
* the noise of one label at P pairs: true hand sigma2_hand + sigma2_dice/P, redeal
  (sigma2_hand + sigma2_dice)/P, and each label's reliability signal / (signal + noise);
* bias -- the true-hand label minus the redeal mean, averaged over decisions for the favourite
  candidate (z-score): a uniform redeal that misjudges the real belief shows up here.

Usage:
  PYTHONPATH=.:build/release .venv/bin/python tools/scripts/playout_hidden_variance.py \\
      --checkpoint <snapshot.pt> --decisions 256 --output-md <report.md>

Sharded (the fork ran it on CI; the workflow is not on main): each part runs with its own --seed and
--dump; then --merge <dumps...> tabulates them all.
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import ts_engine as ts

from ai.search.dmcts import determinize
from ai.training.playout_advantage import (DEFAULT_DECISIONS, PlayoutLabeller, PlayoutRecord, decision_codes,
                                           select_candidates)

KIND_NAMES = {int(getattr(ts.DecisionType, n)): n for n in DEFAULT_DECISIONS}


def collect(model: Any, n: int, k: int, frac: float, seed: int, envs: int, merged: bool, feats: int
            ) -> List[PlayoutRecord]:
    """`n` decisions of the default playout kinds from the model's sampled self-play."""
    from bindings.ts_env import TsVectorizedEnv
    codes = frozenset(decision_codes(DEFAULT_DECISIONS))
    rng = np.random.default_rng(seed)
    env = TsVectorizedEnv(num_envs=envs, base_seed=seed)
    env.set_merged_influence(merged, merged)
    env.set_obs_features(feats, feats)
    obs, masks, _ = env.reset_all()
    out: List[PlayoutRecord] = []
    # Branch decisions are rare; take every one of them so the kind is represented.
    branch = int(ts.DecisionType.CHOOSE_BRANCH)
    for _ in range(200_000):
        o = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        m = torch.from_numpy(np.asarray(masks))
        with torch.no_grad():
            logits = model(o, m)[0].float()
        acts = torch.multinomial(torch.softmax(logits, -1), 1).squeeze(-1).numpy()
        dp = np.asarray(env.runner.get_decision_players())
        mk = np.asarray(masks) != 0
        for i in np.flatnonzero(dp != 0):
            st = env.runner.get_state(int(i))
            kind = int(st.ctx().decision_type)
            if st.current_phase == ts.Phase.SETUP or kind not in codes or mk[i].sum() < 2:
                continue
            if kind != branch and rng.random() >= frac:
                continue
            cands, prior = select_candidates(logits[i].numpy(), mk[i], k)
            out.append(PlayoutRecord(st.clone(), o[i].half(), m[i], int(dp[i]), cands, prior))
            if len(out) >= n:
                return out
        obs, masks, *_ = env.step(acts)
    raise RuntimeError(f"collected only {len(out)} decisions")


def advantages(mat: np.ndarray) -> np.ndarray:
    """(pairs, candidates) values -> each candidate's value minus the mean over the candidates."""
    return mat - mat.mean(axis=1, keepdims=True)


def measure(model: Any, recs: Sequence[PlayoutRecord], hands: int, dice: int, merged: bool, feats: int,
            seed: int, chunk: int) -> List[Dict[str, Any]]:
    lab = PlayoutLabeller(dice, "turn", 400, merged, feats, hidden="true")
    shuffle = random.Random(seed)
    rows: List[Dict[str, Any]] = []
    for s in range(0, len(recs), chunk):
        part = recs[s:s + chunk]
        flat: List[PlayoutRecord] = []
        for r in part:
            flat.append(r)                                                   # the true hand
            for _ in range(hands):
                w = determinize(r.state, ts.Player(r.decider), shuffle)
                flat.append(PlayoutRecord(w, r.obs, r.mask, r.decider, r.cands, r.prior))
        mats, _ = lab.label(model, flat, torch.device("cpu"), seed + s)
        for j, r in enumerate(part):
            block = mats[j * (hands + 1):(j + 1) * (hands + 1)]
            true_a = advantages(block[0])                                    # (D, C)
            red = np.stack([advantages(m) for m in block[1:]])               # (H, D, C)
            within = red.var(axis=1, ddof=1).mean()                          # sigma2_dice
            hand_means = red.mean(axis=1)                                    # (H, C)
            between = hand_means.var(axis=0, ddof=1).mean() - within / dice  # sigma2_hand
            grand = hand_means.mean(axis=0)                                  # (C,)
            noise_grand = hand_means.var(axis=0, ddof=1) / hands             # per candidate
            rows.append({
                "kind": KIND_NAMES.get(int(r.state.ctx().decision_type), "?"),
                "dice": float(within), "hand": float(between),
                "dice_true": float(true_a.var(axis=0, ddof=1).mean()),
                "signal": float((grand ** 2 - noise_grand).mean()),
                "bias_fav": float(true_a.mean(axis=0)[0] - grand[0]),
                "best_agree": bool(int(np.argmax(true_a.mean(axis=0))) == int(np.argmax(grand))),
            })
        print(f"  {min(s + chunk, len(recs))}/{len(recs)} decisions", flush=True)
    return rows


def report(rows: List[Dict[str, Any]], hands: int, dice: int, pairs_list: Sequence[int], ckpt: str) -> str:
    def block(rs: List[Dict[str, Any]], label: str) -> List[str]:
        d = float(np.mean([r["dice"] for r in rs]))
        h = float(np.mean([r["hand"] for r in rs]))
        dt = float(np.mean([r["dice_true"] for r in rs]))
        sig = float(np.mean([r["signal"] for r in rs]))
        b = np.array([r["bias_fav"] for r in rs])
        z = b.mean() / (b.std(ddof=1) / np.sqrt(len(b))) if len(b) > 1 else float("nan")
        out = [f"### {label} ({len(rs)} decisions)", "",
               f"* advantage variance, x1e-3: dice {1e3 * d:.2f} (true hand {1e3 * dt:.2f}), "
               f"hand {1e3 * h:.2f}, signal {1e3 * sig:.2f}; the hand's share of one pair's noise "
               f"{100 * h / max(h + d, 1e-12):.0f}%",
               f"* favourite's true-hand label minus the redeal mean: {b.mean():+.4f} (z {z:+.1f}); "
               f"same best candidate {100 * np.mean([r['best_agree'] for r in rs]):.0f}%", "",
               "| pairs | noise, true hand (x1e-3) | noise, redeal (x1e-3) | reliability, true hand | reliability, redeal |",
               "|---:|---:|---:|---:|---:|"]
        for p in pairs_list:
            nt, nr = h + d / p, (h + d) / p
            out.append(f"| {p} | {1e3 * nt:.2f} | {1e3 * nr:.2f} | {sig / (sig + nt):.2f} | {sig / (sig + nr):.2f} |")
        return out + [""]

    lines = [f"# Playout labels: hidden-hand against dice noise", "",
             f"Checkpoint `{ckpt}`; {hands} redealt hands x {dice} dice per decision, and the true hand x "
             f"{dice} dice; turn horizon, the model on both sides; advantages in win-value units (+-1).", ""]
    lines += block(rows, "all decisions")
    for kind in sorted({r["kind"] for r in rows}):
        rs = [r for r in rows if r["kind"] == kind]
        if len(rs) >= 8:
            lines += block(rs, kind)
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from bindings.ts_env import model_obs_features
    from tools.lib.action_view import checkpoint_merged_influence
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="required unless --merge")
    ap.add_argument("--decisions", type=int, default=256)
    ap.add_argument("--candidates", type=int, default=4)
    ap.add_argument("--hands", type=int, default=8)
    ap.add_argument("--dice", type=int, default=8)
    ap.add_argument("--sample-frac", type=float, default=0.01)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=32, help="decisions per labelling batch")
    ap.add_argument("--seed", type=int, default=71_000)
    ap.add_argument("--output-md", default=None)
    ap.add_argument("--dump", default=None, help="write the per-decision rows here (JSON), for --merge")
    ap.add_argument("--merge", nargs="+", default=None, help="tabulate these dumps instead of measuring")
    a = ap.parse_args(argv)
    if a.merge:
        rows: List[Dict[str, Any]] = []
        meta: Dict[str, Any] = {}
        for f in a.merge:
            d = json.load(open(f))
            rows += d["rows"]
            meta = d
        md = report(rows, int(meta["hands"]), int(meta["dice"]), (1, 8, 16, 32), str(meta["checkpoint"]))
        print(md)
        if a.output_md:
            open(a.output_md, "w").write(md + "\n")
        return 0
    if not a.checkpoint:
        ap.error("--checkpoint is required unless --merge")
    torch.set_grad_enabled(False)
    model = NeuralAgent.from_checkpoint(a.checkpoint, device="cpu").model
    model.eval()
    merged = checkpoint_merged_influence(a.checkpoint)
    feats = model_obs_features(model)
    recs = collect(model, a.decisions, a.candidates, a.sample_frac, a.seed, a.envs, merged, feats)
    rows = measure(model, recs, a.hands, a.dice, merged, feats, a.seed, a.chunk)
    if a.dump:
        json.dump({"checkpoint": a.checkpoint, "hands": a.hands, "dice": a.dice, "seed": a.seed, "rows": rows},
                  open(a.dump, "w"))
    md = report(rows, a.hands, a.dice, (1, 8, 16, 32), a.checkpoint)
    print(md)
    if a.output_md:
        open(a.output_md, "w").write(md + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
