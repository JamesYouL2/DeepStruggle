#!/usr/bin/env python3
"""Why the rollout root's corrections do not transfer: readings on a `tools/distill_targets.py` set.

P36c fitted students to the root's improved policy and found its departures from the network do not carry to
held-out positions. These readings separate the explanations, each on one target set and any of its teachers
(the ladder: the corrected root at several horizons, the network's own critic one step on, another network's
raw policy):

* `agree` -- every teacher's departures and self-agreement, and how often two teachers' departures agree: does
  the critic alone (horizon 0) already point where the deep root does (the critic carries the correction), or
  do the corrections appear only with lookahead?
* `knn` -- local consistency in the network's own representation: at a held-out departure, do the training
  positions nearest in trunk-feature space (or in the observation) prefer the teacher's move too, where it is
  legal there? A teacher whose corrections are a smooth function of what the network represents gives its
  neighbours its move; one whose corrections are position-specific does not. Also the vote of the neighbours'
  teacher moves as a predictor (beside the student's held-out agreement), and the same vote for the network's
  own argmax (the representation's sanity check).
* `track` -- checkpoints against fixed targets: each one's argmax on a teacher's departures (takes the teacher's
  move, keeps the base network's) and where the teacher agrees (leaves it) -- an RL run's snapshots read for
  drift toward the root's corrections without touching the run.

`ai/eval/departure_probe.py --target NAME` is the frozen-trunk probe (is *whether* the teacher departs visible).

    PYTHONPATH=.:build/release python tools/distill_probe.py agree --dataset targets.npz
    PYTHONPATH=.:build/release python tools/distill_probe.py knn --dataset targets.npz --checkpoint swa.pt --target h4
    PYTHONPATH=.:build/release python tools/distill_probe.py track --dataset targets.npz --target h4 \\
        --checkpoints run/snap_400M.pt run/snap_800M.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ai.training.policy_fit import masks_of  # noqa: E402
from ai.training.value_fit import Codes  # noqa: E402
import ts_engine  # noqa: E402, F401 -- the build require_fresh() checks
from tools.lib.engine_fingerprint import check_raw_states, require_fresh  # noqa: E402
from tools.lib.player_agent import NeuralAgent  # noqa: E402


def load(path: str) -> Dict[str, np.ndarray]:
    check_raw_states(path)
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def teachers_of(d: Dict[str, np.ndarray]) -> List[str]:
    return sorted((k[5:] for k in d if k.startswith("acts_")), key=lambda n: (not n.isdigit(), n.zfill(8)))


def _pct(x: float) -> str:
    return "" if x != x else f"{100 * x:.1f}%"


def _mean(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else float("nan")


def agree_table(d: Dict[str, np.ndarray], names: Sequence[str]) -> str:
    """Markdown: per teacher its departures and self-agreement; per pair the agreement on the row teacher's departures."""
    net = d["net_argmax"].astype(np.int64)
    held = d["heldout"].astype(bool)
    top = {n: d[f"acts_{n}"][:, 0].astype(np.int64) for n in names}
    lines = ["| teacher | departs (target's top) | departs (pick) | held-out self-agreement: departures / all / picks |",
             "|:---|---:|---:|---:|"]
    for n in names:
        dep = top[n] != net
        pick = d[f"pick_{n}"].astype(np.int64)
        top2 = d[f"acts2_{n}"][:, 0].astype(np.int64)
        pick2 = d[f"pick2_{n}"].astype(np.int64)
        hd = held & dep
        hp = held & (pick != net)
        lines.append(f"| {n} | {_pct(_mean(dep))} | {_pct(_mean(pick != net))} | {_pct(_mean(top2[hd] == top[n][hd]))} / "
                     f"{_pct(_mean(top2[held] == top[n][held]))} / {_pct(_mean(pick2[hp] == pick[hp]))} |")
    lines += ["", "On the row teacher's departures (all positions): the column teacher's top move is the same (and in "
              "brackets, the column teacher departs at all).", "",
              "| on departures of | " + " | ".join(names) + " |", "|:---|" + "---:|" * len(names)]
    for a in names:
        dep = top[a] != net
        cells = [f"{_pct(_mean(top[b][dep] == top[a][dep]))} ({_pct(_mean(top[b][dep] != net[dep]))})" for b in names]
        lines.append(f"| {a} ({int(dep.sum()):,}) | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def features(model: Any, states: np.ndarray, masks: np.ndarray, space: str, dev: torch.device) -> np.ndarray:
    """Each position's trunk features (`trunk`) or observation (`obs`), standardised and L2-normalised.

    Built in place in one preallocated array: the observation space is n x 3,824 floats (3 GB at 200,000
    positions), so a concatenate and whole-array temporaries would hold several copies at once."""
    codes = Codes(int(getattr(model, "obs_feature_bits", 0)), dev)
    x = codes.encode(states)
    n, step = x.shape[0], 8192
    f: Optional[np.ndarray] = None
    for lo in range(0, n, step):
        obs = codes.decode(torch.from_numpy(x[lo:lo + step]).to(dev))
        with torch.no_grad():
            part = getattr(model, "_encode")(obs)[0].float() if space == "trunk" else obs.float()
        if f is None:
            f = np.empty((n, part.shape[1]), dtype=np.float32)
        f[lo:lo + step] = part.cpu().numpy()
    if f is None:
        return np.zeros((0, 1), np.float32)
    total, sq = np.zeros(f.shape[1]), np.zeros(f.shape[1])
    for lo in range(0, n, step):
        c = f[lo:lo + step].astype(np.float64)
        total += c.sum(0)
        sq += (c * c).sum(0)
    mu = total / n
    sd = np.sqrt(np.maximum(sq / n - mu * mu, 0.0))
    mu32, inv32 = mu.astype(np.float32), (1.0 / (sd + 1e-6)).astype(np.float32)
    for lo in range(0, n, step):
        c = f[lo:lo + step]
        c -= mu32
        c *= inv32
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-9
    return f


def neighbours(f: np.ndarray, query: np.ndarray, pool: np.ndarray, k: int, dev: torch.device) -> np.ndarray:
    """(len(query), k) indices into `pool` of each query row's nearest pool rows by cosine.

    Scored against every row with the rows outside the pool masked, so `f` is never copied (f[pool] would be a
    second n x width array)."""
    fa = torch.from_numpy(f).to(dev)
    outside = torch.ones(f.shape[0], dtype=torch.bool, device=dev)
    outside[torch.from_numpy(pool).to(dev)] = False
    out: List[np.ndarray] = []
    for lo in range(0, query.size, 1024):
        q = fa[torch.from_numpy(query[lo:lo + 1024]).to(dev)]
        s = (q @ fa.T).masked_fill(outside, float("-inf"))
        out.append(torch.topk(s, min(k, pool.size), dim=1).indices.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, k), np.int64)


def knn_readings(top: np.ndarray, net: np.ndarray, masks: np.ndarray, query: np.ndarray, nb: np.ndarray) -> Dict[str, float]:
    """At the query rows where the teacher departs: (1) among neighbours where the teacher's move is legal, the
    share whose teacher plays it, against the share whose network does; (2) the neighbours' teacher-move vote
    (legal at the query, ties to the network's move) as a prediction of the teacher's move; (3) the same vote
    of the neighbours' network moves for the query's network move (all query rows)."""
    dep = top[query] != net[query]
    same_t, same_n, legal_n, vote_hit, net_hit = 0, 0, 0, 0, 0
    for r, (i, js) in enumerate(zip(query, nb)):
        legal = masks[i].astype(bool)
        votes = np.bincount(net[js], minlength=masks.shape[1])[:masks.shape[1]] * legal
        net_hit += int(int(np.argmax(votes)) == int(net[i]) if votes.any() else False)
        if not dep[r]:
            continue
        a = int(top[i])
        ok = masks[js, a].astype(bool)
        legal_n += int(ok.sum())
        same_t += int((top[js][ok] == a).sum())
        same_n += int((net[js][ok] == a).sum())
        tv = np.bincount(top[js], minlength=masks.shape[1])[:masks.shape[1]] * legal
        tv = tv.astype(np.float64)
        tv[int(net[i])] += 0.5                      # ties go to the network's own move
        vote_hit += int(int(np.argmax(tv)) == a)
    n_dep = int(dep.sum())
    return {"departures": float(n_dep), "neighbours_legal": float(legal_n),
            "neighbour_teacher_plays_it": same_t / max(1, legal_n), "neighbour_network_plays_it": same_n / max(1, legal_n),
            "vote_predicts_teacher_on_departures": vote_hit / max(1, n_dep),
            "vote_predicts_network_move": net_hit / max(1, query.size)}


def track_rows(top: np.ndarray, base: np.ndarray, ck: np.ndarray, rows: np.ndarray) -> Dict[str, float]:
    dep = rows[top[rows] != base[rows]]
    agree = rows[top[rows] == base[rows]]
    return {"takes_teacher_on_departures": _mean(ck[dep] == top[dep]), "keeps_base_on_departures": _mean(ck[dep] == base[dep]),
            "leaves_agreed_move": _mean(ck[agree] != top[agree]), "agrees_with_teacher": _mean(ck[rows] == top[rows]),
            "agrees_with_base": _mean(ck[rows] == base[rows])}


def argmaxes(model: Any, states: np.ndarray, masks: np.ndarray, dev: torch.device) -> np.ndarray:
    codes = Codes(int(getattr(model, "obs_feature_bits", 0)), dev)
    x = codes.encode(states)
    out: List[np.ndarray] = []
    for lo in range(0, x.shape[0], 8192):
        m = torch.from_numpy(masks[lo:lo + 8192]).to(dev)
        with torch.no_grad():
            lg = model(codes.decode(torch.from_numpy(x[lo:lo + 8192]).to(dev)), m)[0].float()
        out.append(lg.masked_fill(m == 0, float("-inf")).argmax(-1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("agree", help="teachers' departures, self-agreement and pairwise agreement")
    g.add_argument("--dataset", required=True)
    g.add_argument("--teachers", nargs="*", default=None, help="default: every teacher in the set")
    k = sub.add_parser("knn", help="local consistency of a teacher's departures in the network's representation")
    k.add_argument("--dataset", required=True)
    k.add_argument("--checkpoint", required=True)
    k.add_argument("--target", required=True)
    k.add_argument("--k", type=int, default=32)
    k.add_argument("--space", nargs="+", choices=["trunk", "obs"], default=["trunk", "obs"])
    t = sub.add_parser("track", help="checkpoints' argmax against a teacher's fixed targets")
    t.add_argument("--dataset", required=True)
    t.add_argument("--target", required=True)
    t.add_argument("--checkpoints", nargs="+", required=True)
    t.add_argument("--rows", choices=["all", "heldout"], default="all")
    for p in (g, k, t):
        p.add_argument("--device", default=None)
        p.add_argument("--json", default=None, help="also write the readings here")
    a = ap.parse_args(argv)
    require_fresh()
    d = load(a.dataset)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    result: Dict[str, Any] = {"dataset": a.dataset, "command": a.cmd}
    if a.cmd == "agree":
        names = a.teachers or teachers_of(d)
        table = agree_table(d, names)
        print(table)
        result["table"] = table
    elif a.cmd == "knn":
        model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model.eval()
        masks = masks_of(d["states"])
        held = d["heldout"].astype(bool)
        top = d[f"acts_{a.target}"][:, 0].astype(np.int64)
        net = d["net_argmax"].astype(np.int64)
        query, pool = np.flatnonzero(held), np.flatnonzero(~held)
        print(f"teacher {a.target}: {query.size:,} held-out positions, {int((top[query] != net[query]).sum()):,} "
              f"departures; {pool.size:,} training positions; k {a.k}")
        for space in a.space:
            f = features(model, d["states"], masks, space, dev)
            r = knn_readings(top, net, masks, query, neighbours(f, query, pool, a.k, dev))
            result[space] = r
            print(f"  {space:5s}: where the teacher's move is legal at a neighbour, the teacher plays it "
                  f"{_pct(r['neighbour_teacher_plays_it'])}, the network {_pct(r['neighbour_network_plays_it'])}; the "
                  f"neighbours' vote finds the teacher's move at {_pct(r['vote_predicts_teacher_on_departures'])} of "
                  f"departures; the network's own move at {_pct(r['vote_predicts_network_move'])} of positions")
    else:
        masks = masks_of(d["states"])
        top = d[f"acts_{a.target}"][:, 0].astype(np.int64)
        base = d["net_argmax"].astype(np.int64)
        rows = np.flatnonzero(d["heldout"].astype(bool)) if a.rows == "heldout" else np.arange(base.size)
        print(f"| checkpoint | takes {a.target}'s move on its departures | keeps the base's | leaves an agreed move | "
              f"agrees with {a.target} | agrees with the base |")
        print("|:---|---:|---:|---:|---:|---:|")
        for c in a.checkpoints:
            model = NeuralAgent.from_checkpoint(c, device=str(dev)).model.eval()
            r = track_rows(top, base, argmaxes(model, d["states"], masks, dev), rows)
            result[c] = r
            print(f"| {os.path.basename(c)} | {_pct(r['takes_teacher_on_departures'])} | {_pct(r['keeps_base_on_departures'])} "
                  f"| {_pct(r['leaves_agreed_move'])} | {_pct(r['agrees_with_teacher'])} | {_pct(r['agrees_with_base'])} |",
                  flush=True)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
