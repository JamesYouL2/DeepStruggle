#!/usr/bin/env python3
"""Can the network's own trunk rank sibling moves better than its critic does, given better labels?

The search bank found search evaluation-limited, and its data put the bare critic's ranking of two
sibling moves at 61% right against 2,048-pair playouts (55% at card choice; 256-simulation search
over 64 worlds, 74%). Backgammon programs improved by training their evaluation on rollout-labelled
positions. Before training anything, this asks whether that is possible here:

    label  fresh self-play of the model (seeds disjoint from the bank's); at a sampled share of its
           decisions with two or more legal moves, the network's top --k moves played out in paired
           raw-network continuations (ai/eval/paired_playouts.compare: one redeal and one set of
           dice per pair, shared by every move). Each row: the position, the moves, each move's mean
           score for the mover, and each move's paired difference from the network's own move.
    fit    a value head on the FROZEN trunk features h of each child state (the network's own
           representation, unchanged), trained on those labels -- an MLP, and the critic's own head
           fine-tuned from its weights -- and tested on the bank's positions against their
           2,048-pair playouts: how often each picks the better of two siblings the playouts clearly
           separate, beside the untouched critic.

If a head on frozen h ranks siblings well above the critic, the information is in the trunk and the
critic's training signal is what limits it -- a value-target problem. If it does not, the trunk does
not carry it -- a representation problem.

    PYTHONPATH=.:build/release .venv/bin/python tools/value_probe.py label --model m.pt --games 50 \\
        --rate 0.025 --k 3 --pairs 256 --part 1/20 --out label-1.jsonl.gz
    ... fit --model m.pt --labels label-*.jsonl.gz --bank bank.jsonl.gz \\
        --bank-playouts full/playouts.jsonl.gz --out probe_report.md
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


def _read(paths: Sequence[str]) -> Iterator[Dict[str, Any]]:
    for p in paths:
        with gzip.open(p, "rt", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)


def _write(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def position_seed(rid: str, seed: int) -> int:
    """As tools/scripts/bank_playouts.position_seed."""
    return (int(rid, 16) ^ (seed * 0x9E3779B97F4A7C15)) & 0x7FFF_FFFF_FFFF_FFFF


# --- label --------------------------------------------------------------------------------------

def label(model: str, games: int, rate: float, k: int, pairs: int, part: str, seed: int,
          temperature: float, out: str) -> int:
    import ts_engine as ts
    from ai.eval.paired_playouts import compare, paired_diff
    from bindings.action_encoder import ActionEncoder
    from tools.lib.corpus_driver import load_policy, position_token, selfplay

    pk, pn = (int(x) for x in part.split("/"))
    logits_fn, feats = load_policy(model)
    rng = random.Random(seed * 7919 + pk)

    class Sampler:
        def __init__(self) -> None:
            self.positions: List[Tuple[str, Dict[str, Any]]] = []

        def observe(self, st: Any, action: int) -> None:
            mask = np.asarray(ActionEncoder.get_legal_mask(st))
            if int(mask.sum()) < 2 or rng.random() >= rate:
                return
            c = st.ctx()
            mover = c.decision_player if c.decision_player != ts.Player.NONE else st.phasing_player
            self.positions.append((position_token(st), {
                "decision_type": str(c.decision_type).split(".")[-1], "turn": int(st.turn),
                "side": "US" if mover == ts.Player.US else "USSR"}))

    t0 = time.time()
    base = 5_000_000 + 100_000 * pk + 1_000 * seed       # the bank's games were seeds 700,000+
    trackers = selfplay(logits_fn, feats, games, base, min(games, 50), temperature, Sampler)
    sampled = [p for t in trackers for p in t.positions]
    print(f"{len(sampled)} positions from {games} games in {time.time() - t0:.0f}s", file=sys.stderr)

    from tools.lib.corpus_driver import state_from_token
    states = [state_from_token(tok) for tok, _ in sampled]
    from ai.search.pimcts import acting_player
    obs = np.stack([np.asarray(ts.extract_observation_features(s, acting_player(s), feats), dtype=np.float32)
                    for s in states])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
    lg = np.where(masks.astype(bool), np.asarray(logits_fn(obs, masks)), -np.inf)
    rows, positions = [], []
    for (tok, info), s, z in zip(sampled, states, lg):
        top = [int(a) for a in np.argsort(-z)[:k] if np.isfinite(z[a])]
        rid = hashlib.sha256(tok.encode()).hexdigest()[:16]
        rows.append({"id": rid, "pos": tok, "raw": top[0], "moves": top, **info})
        positions.append((s, top))

    def act(o: np.ndarray, m: np.ndarray) -> np.ndarray:
        return np.asarray(logits_fn(o, m)).argmax(axis=1).astype(np.int32)

    t1 = time.time()
    scores = compare(positions, act, pairs, [position_seed(r["id"], seed) for r in rows])
    for r, sc in zip(rows, scores):
        r["pairs"] = pairs
        r["score"] = {str(a): round(sum(v) / len(v), 5) for a, v in sc.items()}
        r["diff_vs_raw"] = {str(a): [round(x, 5) for x in paired_diff(v, sc[r["raw"]])]
                            for a, v in sc.items() if a != r["raw"]}
    _write(out, rows)
    with open(out + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"stage": "label", "model": os.path.basename(model), "model_sha256": _sha256(model),
                   "games": games, "rate": rate, "k": k, "pairs": pairs, "part": part, "seed": seed,
                   "temperature": temperature, "game_seed_base": base, "positions": len(rows),
                   "selfplay_s": round(t1 - t0, 1), "playout_s": round(time.time() - t1, 1)}, f, indent=1)
    print(f"{len(rows)} positions labelled, {pairs} pairs, playouts {time.time() - t1:.0f}s", file=sys.stderr)
    return 0


# --- fit ----------------------------------------------------------------------------------------

def child_features(model: Any, feats: int, pos: str, moves: Sequence[int], worlds: int,
                   seed: int) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Per move: the trunk features h of its child in `worlds` redealt worlds (the mover's unseen
    cards and the dice drawn afresh, as the paired playouts draw them), and the sign that turns a
    value for the child's acting player into one for the mover."""
    import torch
    import ts_engine as ts
    from ai.eval.paired_playouts import apply_move, decider, pair_start
    from ai.search.pimcts import acting_player
    from tools.lib.corpus_driver import state_from_token

    st = state_from_token(pos)
    mover = decider(st)
    rows, keys, signs = [], [], []
    for w in range(worlds):
        base = pair_start(st, w, seed)
        for a in moves:
            ch = apply_move(base, int(a))
            if ts.Engine.is_terminal(ch):
                continue
            who = acting_player(ch)
            rows.append(np.asarray(ts.extract_observation_features(ch, who, feats), dtype=np.float32))
            keys.append(int(a))
            signs.append(1.0 if who == mover else -1.0)
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if not rows:
        return out
    with torch.no_grad():
        h = model.extract_features(torch.from_numpy(np.stack(rows))).float().numpy()
    for a in set(keys):
        idx = [i for i, x in enumerate(keys) if x == a]
        out[a] = (h[idx], np.asarray([signs[i] for i in idx], dtype=np.float32))
    return out


def sibling_accuracy(pred: Dict[str, Dict[int, float]], truth: Dict[str, Dict[int, Tuple[float, float]]],
                     raw: Dict[str, int], min_z: float = 2.0, min_gap: float = 0.01) -> Tuple[float, int]:
    """Share of (move, raw) sibling pairs the playouts clearly separate (|diff| > min_z SE and >
    min_gap) on which `pred` ranks the two the same way."""
    hit = n = 0
    for pid, diffs in truth.items():
        for a, (m, se) in diffs.items():
            if abs(m) <= min_z * se or abs(m) <= min_gap:
                continue
            if pid not in pred or a not in pred[pid] or raw[pid] not in pred[pid]:
                continue
            n += 1
            hit += int((pred[pid][a] - pred[pid][raw[pid]] > 0) == (m > 0))
    return (hit / n if n else float("nan")), n


def fit(model_path: str, labels: Sequence[str], bank: str, bank_playouts: str, worlds: int,
        test_worlds: int, epochs: int, out: str, seed: int = 0, test_limit: int = 0) -> int:
    import torch
    import torch.nn as nn
    from tools.lib.player_agent import NeuralAgent
    from bindings.ts_env import model_obs_features

    torch.manual_seed(seed)
    net = NeuralAgent.from_checkpoint(model_path, device="cpu").model.eval()
    feats = int(model_obs_features(net))
    head = getattr(net, "val_win_head")

    # Training set: every labelled child, each world its own example with the position's label.
    lab = list(_read(labels))
    rng = random.Random(seed)
    rng.shuffle(lab)
    t0 = time.time()
    X, Y, S, G = [], [], [], []           # features, mover-value target, sign, group (position)
    for gi, r in enumerate(lab):
        cf = child_features(net, feats, r["pos"], r["moves"], worlds, position_seed(r["id"], 11))
        for a, (h, sg) in cf.items():
            y = 2.0 * float(r["score"][str(a)]) - 1.0          # mover's value in [-1, 1]
            for hi, si in zip(h, sg):
                X.append(hi)
                Y.append(y)
                S.append(si)
                G.append((gi, a))
    X_t = torch.from_numpy(np.stack(X))
    Y_t = torch.tensor(Y)
    S_t = torch.tensor(S)
    print(f"{len(lab)} labelled positions, {len(X)} child examples, features in {time.time() - t0:.0f}s",
          file=sys.stderr)

    # Test set: the bank's positions, every move its 2,048-pair playouts scored.
    bank_rows = {str(r["id"]): r for r in _read([bank])}
    truth: Dict[str, Dict[int, Tuple[float, float]]] = {}
    test_moves: Dict[str, List[int]] = {}
    raw: Dict[str, int] = {}
    for p in _read([bank_playouts]):
        pid = str(p["id"])
        if pid not in bank_rows:
            continue
        sc = {int(a): v for a, v in p["scores"].items()}
        r0 = int(bank_rows[pid]["raw"])
        if r0 not in sc:
            continue
        raw[pid] = r0
        test_moves[pid] = sorted(sc)
        d: Dict[int, Tuple[float, float]] = {}
        for a, v in sc.items():
            if a == r0:
                continue
            diff = [x - y for x, y in zip(v, sc[r0])]
            m = sum(diff) / len(diff)
            se = math.sqrt(sum((x - m) ** 2 for x in diff) / (len(diff) - 1) / len(diff))
            d[a] = (m, se)
        truth[pid] = d
        if test_limit and len(truth) >= test_limit:
            break
    t1 = time.time()
    test_feats = {pid: child_features(net, feats, bank_rows[pid]["pos"], test_moves[pid], test_worlds,
                                      position_seed(pid, 13)) for pid in truth}
    print(f"{len(truth)} test positions, features in {time.time() - t1:.0f}s", file=sys.stderr)

    def predict(fn: Any) -> Dict[str, Dict[int, float]]:
        out_: Dict[str, Dict[int, float]] = {}
        with torch.no_grad():
            for pid, cf in test_feats.items():
                out_[pid] = {a: float((fn(torch.from_numpy(h)).reshape(-1) * torch.from_numpy(sg)).mean())
                             for a, (h, sg) in cf.items()}
        return out_

    results: Dict[str, Any] = {}
    acc, n = sibling_accuracy(predict(head), truth, raw)
    results["critic (untouched)"] = {"accuracy": acc, "pairs": n}

    def train(module: nn.Module, name: str, lr: float) -> None:
        opt = torch.optim.Adam(module.parameters(), lr=lr)
        idx = np.arange(len(X))
        for ep in range(epochs):
            np.random.default_rng(seed + ep).shuffle(idx)
            for lo in range(0, len(idx), 1024):
                b = torch.from_numpy(idx[lo:lo + 1024])
                pred = module(X_t[b]).reshape(-1) * S_t[b]     # the mover's value
                loss = ((pred - Y_t[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            a_, n_ = sibling_accuracy(predict(module), truth, raw)
            print(f"  {name} epoch {ep + 1}: mse {float(loss):.4f}  test sibling accuracy {a_:.1%} ({n_})",
                  file=sys.stderr, flush=True)
        a_, n_ = sibling_accuracy(predict(module), truth, raw)
        results[name] = {"accuracy": a_, "pairs": n_}

    import copy
    tuned = copy.deepcopy(head)
    for p_ in tuned.parameters():
        p_.requires_grad_(True)
    train(tuned, "critic head fine-tuned on rollout labels", 3e-4)
    width = X_t.shape[1]
    mlp = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1), nn.Tanh())
    train(mlp, "fresh MLP head on frozen h", 1e-3)

    # Learning curve for the fresh head: does accuracy still rise with labels?
    curve = {}
    for frac in (0.25, 0.5):
        keep = {g for g in range(int(len(lab) * frac))}
        sel = [i for i, (gi, _a) in enumerate(G) if gi in keep]
        Xs, Ys, Ss = X_t[sel], Y_t[sel], S_t[sel]
        m2 = nn.Sequential(nn.Linear(width, 256), nn.GELU(), nn.Linear(256, 1), nn.Tanh())
        opt = torch.optim.Adam(m2.parameters(), lr=1e-3)
        idx = np.arange(len(sel))
        for ep in range(epochs):
            np.random.default_rng(seed + ep).shuffle(idx)
            for lo in range(0, len(idx), 1024):
                b = torch.from_numpy(idx[lo:lo + 1024])
                loss = ((m2(Xs[b]).reshape(-1) * Ss[b] - Ys[b]) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        curve[f"{int(frac * len(lab))} positions"] = sibling_accuracy(predict(m2), truth, raw)[0]
    curve[f"{len(lab)} positions"] = results["fresh MLP head on frozen h"]["accuracy"]
    results["learning curve (fresh MLP)"] = curve

    lines = ["# Value probe: can the frozen trunk rank sibling moves better than the critic?", "",
             f"* model `{os.path.basename(model_path)}`; {len(lab)} labelled training positions "
             f"({len(X)} child examples, {worlds} worlds each); test: {len(truth)} search-bank positions "
             f"against their 2,048-pair playouts ({test_worlds} worlds per child).",
             "* accuracy = share of sibling pairs the playouts separate by > 2 SE and > 1 point on "
             "which the head ranks the two the same way.", "",
             "| value | sibling accuracy | pairs |", "|:---|---:|---:|"]
    for k_, v in results.items():
        if "accuracy" in v:
            lines.append(f"| {k_} | {v['accuracy']:.1%} | {v['pairs']} |")
    lines += ["", "Learning curve: " + ", ".join(f"{k_} {v:.1%}" for k_, v in curve.items()), ""]
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.splitext(out)[0] + ".json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    torch.save({"mlp": mlp.state_dict(), "tuned_head": tuned.state_dict()},
               os.path.splitext(out)[0] + ".heads.pt")
    print("\n".join(lines))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                 allow_abbrev=False)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lab = sub.add_parser("label")
    lab.add_argument("--model", required=True)
    lab.add_argument("--games", type=int, default=50)
    lab.add_argument("--rate", type=float, default=0.025)
    lab.add_argument("--k", type=int, default=3)
    lab.add_argument("--pairs", type=int, default=256)
    lab.add_argument("--part", default="1/1")
    lab.add_argument("--seed", type=int, default=0)
    lab.add_argument("--temperature", type=float, default=0.1)
    lab.add_argument("--out", required=True)
    ft = sub.add_parser("fit")
    ft.add_argument("--model", required=True)
    ft.add_argument("--labels", nargs="+", required=True)
    ft.add_argument("--bank", required=True)
    ft.add_argument("--bank-playouts", required=True)
    ft.add_argument("--worlds", type=int, default=4)
    ft.add_argument("--test-worlds", type=int, default=16)
    ft.add_argument("--epochs", type=int, default=8)
    ft.add_argument("--test-limit", type=int, default=0, help="first N bank positions only (smoke runs)")
    ft.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "label":
        return label(a.model, a.games, a.rate, a.k, a.pairs, a.part, a.seed, a.temperature, a.out)
    return fit(a.model, a.labels, a.bank, a.bank_playouts, a.worlds, a.test_worlds, a.epochs, a.out,
               test_limit=a.test_limit)


if __name__ == "__main__":
    sys.exit(main())
