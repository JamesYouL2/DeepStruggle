#!/usr/bin/env python3
"""How many keyed worlds a sibling label needs: the split-half reliability of the centred candidate
means at 16, 32, 64, ... worlds, read from one sibling slice labelled at the most (P34 revision 2,
step 4: D3 "uses enough worlds for a ceiling of at least 0.85").

D2 measured 0.41 at 16 keyed worlds, and the plan extrapolates by Spearman-Brown (64 worlds ~0.74,
128 ~0.85). This reads the curve directly. For each world count W it keeps worlds 0..W-1 and splits
them into even and odd worlds. In each half it takes each candidate's mean result over the worlds
that every candidate finished, centred on the decision's mean. The two halves' correlation r, over
every (decision, candidate), is stepped up to the full W as 2r / (1 + r). The SEs are a bootstrap
over decisions. Read on the whole slice, on entry decisions (card choices and headlines), and on
D3's contested population: entry decisions whose mean result over all candidates and worlds lies in
[--lo, --hi].

    python tools/scripts/sibling_reliability.py --dir <labelled dir> --json rel.json > rel.md
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any, Dict, List

import numpy as np

import ts_engine as ts


def load(d: pathlib.Path) -> Dict[str, np.ndarray]:
    """Every siblings_*.npz part of a label stage, joined."""
    files = sorted(d.glob("siblings_*_of_*.npz"))
    if not files:
        raise SystemExit(f"{d}: no siblings_*_of_*.npz")
    parts = [dict(np.load(f)) for f in files]
    return {k: np.concatenate([p[k] for p in parts]) for k in ("id", "meta", "cols")}


def decisions(s: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """`y` (decisions x k x worlds) the mover's result, `ok` (decisions x worlds) the worlds every
    candidate finished, and `id` each decision's position."""
    meta = s["meta"]
    k = int(meta[:, 0].max()) + 1
    w = int(meta[:, 1].max()) + 1
    u, inv = np.unique(s["id"], return_inverse=True)
    y = np.zeros((u.size, k, w), np.float64)
    present = np.zeros((u.size, k, w), bool)
    failed = np.zeros((u.size, k, w), bool)
    y[inv, meta[:, 0], meta[:, 1]] = s["cols"][:, 0]
    present[inv, meta[:, 0], meta[:, 1]] = True
    failed[inv, meta[:, 0], meta[:, 1]] = meta[:, 2] == 1
    ok = (present & ~failed).all(axis=1)
    return {"id": u, "y": y, "ok": ok}


def half_means(y: np.ndarray, ok: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Centred candidate means over the worlds in `mask` (decisions x k); NaN where a decision has none."""
    use = ok & mask[None, :]
    n = use.sum(1)
    m = (y * use[:, None, :]).sum(2) / np.maximum(n, 1)[:, None]
    m = m - m.mean(1, keepdims=True)
    m[n == 0] = np.nan
    return m


def reliability(y: np.ndarray, ok: np.ndarray, worlds: int) -> Dict[str, float]:
    """Split-half reliability at `worlds`, stepped up, and how often the halves agree on the best
    candidate and on each pair's order."""
    w = np.arange(y.shape[2])
    a = half_means(y, ok, (w < worlds) & (w % 2 == 0))
    b = half_means(y, ok, (w < worlds) & (w % 2 == 1))
    keep = np.isfinite(a).all(1) & np.isfinite(b).all(1)
    a, b = a[keep], b[keep]
    r = float(np.corrcoef(a.ravel(), b.ravel())[0, 1]) if a.size > 2 else float("nan")
    da = a[:, :, None] - a[:, None, :]
    db = b[:, :, None] - b[:, None, :]
    both = (da != 0) & (db != 0)
    sign = float(((np.sign(da) == np.sign(db)) & both).sum() / max(1, both.sum()))
    top = float((a.argmax(1) == b.argmax(1)).mean()) if a.size else float("nan")
    return {"r_half": r, "reliability": 2 * r / (1 + r), "pair_sign": sign, "top_agree": top,
            "decisions": int(keep.sum())}


def spearman_brown(rel: float, factor: float) -> float:
    return factor * rel / (1 + (factor - 1) * rel)


def read(y: np.ndarray, ok: np.ndarray, counts: List[int], boots: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    out: Dict[str, Any] = {"decisions": int(y.shape[0])}
    for wc in counts:
        point = reliability(y, ok, wc)
        bs = [reliability(y[i], ok[i], wc)["reliability"]
              for i in (rng.integers(0, y.shape[0], y.shape[0]) for _ in range(boots))]
        point["se"] = float(np.nanstd(bs))
        out[str(wc)] = point
    return out


def within_world_corr(y: np.ndarray, ok: np.ndarray) -> float:
    """The candidates' mean pairwise correlation within a world, net of each candidate's own mean --
    D2 measured 0.39 under keyed chance (0.22 unkeyed)."""
    k = y.shape[1]
    resid = y - np.where(ok[:, None, :], y, 0).sum(2, keepdims=True) / np.maximum(ok.sum(1), 1)[:, None, None]
    rs = []
    for i in range(k):
        for j in range(i + 1, k):
            xi, xj = resid[:, i][ok], resid[:, j][ok]
            rs.append(float(np.corrcoef(xi, xj)[0, 1]))
    return float(np.mean(rs))


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="a value_labels.py directory with positions and siblings")
    ap.add_argument("--worlds", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--lo", type=float, default=0.25, help="contested: the mean result's lower bound")
    ap.add_argument("--hi", type=float, default=0.75, help="contested: the mean result's upper bound")
    ap.add_argument("--boots", type=int, default=200)
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    d = pathlib.Path(a.dir)
    dec = decisions(load(d))
    y, ok = dec["y"], dec["ok"]
    have = y.shape[2]
    counts = [c for c in a.worlds if c <= have]

    pos = [dict(np.load(f)) for f in sorted(d.glob("positions_*.npz"))]
    pid = np.concatenate([p["id"] for p in pos])
    state = np.concatenate([p["state"] for p in pos])
    row = {int(i): r for r, i in enumerate(pid.tolist())}
    entry = np.asarray([ts.state_from_raw_bytes(state[row[int(i)]].tobytes()).ctx().decision_type
                        == ts.DecisionType.SELECT_CARD for i in dec["id"]])
    mean = (y * ok[:, None, :]).sum((1, 2)) / np.maximum(ok.sum(1) * y.shape[1], 1)
    contested = entry & (mean >= a.lo) & (mean <= a.hi)

    slices = {"all": np.ones(y.shape[0], bool), "entry": entry, "contested": contested}
    res: Dict[str, Any] = {"worlds": have, "candidates": int(y.shape[1]), "lo": a.lo, "hi": a.hi,
                           "within_world_corr": within_world_corr(y, ok)}
    for name, m in slices.items():
        res[name] = read(y[m], ok[m], counts, a.boots, a.seed)

    print(f"{y.shape[0]:,} sibling decisions x {y.shape[1]} candidates x {have} worlds; "
          f"within-world candidate correlation {res['within_world_corr']:.2f}\n")
    base = counts[0] if counts else 0
    for name in slices:
        r = res[name]
        print(f"**{name}** ({r['decisions']:,} decisions)\n")
        print("| worlds | reliability | ± | Spearman-Brown from " + str(base) + " | ceiling √rel | halves: pair order agrees | halves: best agrees |")
        print("|--:|--:|--:|--:|--:|--:|--:|")
        for wc in counts:
            p = r[str(wc)]
            sb = spearman_brown(r[str(base)]["reliability"], wc / base)
            print(f"| {wc} | {p['reliability']:.3f} | {p['se']:.3f} | {sb:.3f} | {np.sqrt(max(p['reliability'], 0)):.3f} | "
                  f"{p['pair_sign']:.3f} | {p['top_agree']:.3f} |")
        print()
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
