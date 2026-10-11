"""The setup matrix: every US setup against every USSR setup, the rest of the game played by one
player, and the matrix game the results define, solved.

A setup is the only decision of the game made with nothing on the board but the deal, and the E7
networks make it the same way on every deal (France 6, Italy 2, Iran 1 for the US; Poland 3,
Hungary 2, Austria 1 for the USSR, whatever the hand), where human players choose among a few
other setups (tools/lib/openings.py lists both). The matrix asks which of them wins when the rest
of the game is played by the network, greedy or searched:

* **Cells.** A cell is one US setup against one USSR setup. Its value is the US score (a draw is
  half), over every deal the tournament plays.
* **Pairing.** Entrants are the player with both setups forced (`opening:<us>+<ussr>:<player>`).
  A tournament pairing of A = (US a, USSR d) and B = (US c, USSR b) plays cell (a, b) with A as US
  and cell (c, d) with B as US, so half as many pairings as cells cover the matrix
  (`tools/tournament.py --matchups`). Game pair k of every pairing deals from the same seed, and
  the deal comes before the setup, so every cell is played on the same deals and every
  comparison between two cells is paired by deal.
* **The solve.** The cells are a zero-sum matrix game, US rows against USSR columns; its
  equilibrium (support enumeration, exact for these sizes) says which setups a player who knows
  the continuation would mix, and the bootstrap over deals how firmly.

What it cannot say: the value of a setup *to this continuation*. The network learned from its own
setups, so a setup it never trained from is played by a policy that has not seen its positions;
a human setup that wins under the network's play is strong evidence, one that loses is not
evidence against it until a searched continuation agrees.

    python tools/setup_matrix.py plan --base models/base.pt --out plan.json
    python tools/tournament.py --models $(python tools/setup_matrix.py args --plan plan.json models) \\
        --matchups $(python tools/setup_matrix.py args --plan plan.json matchups) \\
        --games-per-side 5000 --temperature 0 --log-games games.jsonl --output-json t.json
    python tools/setup_matrix.py report --plan plan.json --games games.jsonl --out matrix.md
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import os
import sys
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np
import numpy.typing as npt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

#: The default rows and columns: the E7 networks' setup first (the reference every other is
#: compared with), then the three most common human setups of each side.
DEFAULT_US: Tuple[str, ...] = ("us_e7", "us_wg4_it4", "us_wg3_fr3_it2", "us_wg4_it3_ir2")
DEFAULT_USSR: Tuple[str, ...] = ("ussr_e7", "ussr_pol4_aus", "ussr_pol4_yug", "ussr_pol5")

_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rules", "map.json")


class Entrant(TypedDict):
    label: str
    us: int       # index into the plan's US setups
    ussr: int     # index into the plan's USSR setups
    spec: str


class Cell(TypedDict):
    us: int
    ussr: int
    us_label: str     # the entrant that plays this cell as the US
    ussr_label: str   # the entrant that plays it as the USSR


class Comparison(TypedDict):
    diff: float     # mean paired difference, as a fraction of a game
    se: float
    deals: int


class Equilibrium(TypedDict):
    us: List[float]
    ussr: List[float]
    value: float
    value_ci95: List[float]
    us_support_freq: List[float]     # share of bootstrap resamples with the setup's weight > 1%
    ussr_support_freq: List[float]


class JointTest(TypedDict):
    t2: float
    contrasts: int
    p: float


class Plan(TypedDict):
    us: List[str]
    ussr: List[str]
    player: str
    entrants: List[Entrant]
    matchups: List[List[int]]   # 0-based entrant index pairs, i < j
    cells: List[Cell]


class MatrixData(TypedDict):
    plan: Plan
    matrix: List[List[float]]     # US score, rows US setups, columns USSR setups
    games: List[List[int]]
    deals: int
    equilibrium: Equilibrium
    comparisons: Dict[str, Comparison]
    joint_test: JointTest
    margins: Dict[str, Comparison]      # each setup averaged over the other side's, minus the reference


# --- plan -----------------------------------------------------------------------------------

def make_plan(us: Sequence[str], ussr: Sequence[str], player: str) -> Plan:
    """Entrants and pairings that play every cell exactly once per seat-pair (twice for the last
    cell of an odd matrix, whose pairing has the same setups on both sides)."""
    from tools.lib.openings import opening_scripts

    for name in us:
        if set(opening_scripts(name)) != {"US"}:
            raise ValueError(f"{name!r} is not a US-only opening")
    for name in ussr:
        if set(opening_scripts(name)) != {"USSR"}:
            raise ValueError(f"{name!r} is not a USSR-only opening")
    if len(set(us)) != len(us) or len(set(ussr)) != len(ussr):
        raise ValueError("a setup is listed twice")

    entrants: List[Entrant] = []
    index: Dict[str, int] = {}

    def entrant(a: int, b: int, suffix: str = "") -> int:
        label = f"U{a}S{b}{suffix}"
        if label not in index:
            index[label] = len(entrants)
            entrants.append({"label": label, "us": a, "ussr": b,
                             "spec": f"name:{label}:opening:{us[a]}+{ussr[b]}:{player}"})
        return index[label]

    cells_rc = [(i, j) for i in range(len(us)) for j in range(len(ussr))]
    matchups: List[List[int]] = []
    cells: List[Cell] = []
    for k in range(0, len(cells_rc), 2):
        (i, j) = cells_rc[k]
        if k + 1 < len(cells_rc):
            (c, d) = cells_rc[k + 1]
            # A = (US i, USSR d), B = (US c, USSR j): A as US plays (i, j), B as US plays (c, d).
            a, b = entrant(i, d), entrant(c, j)
            cells.append({"us": i, "ussr": j, "us_label": entrants[a]["label"], "ussr_label": entrants[b]["label"]})
            cells.append({"us": c, "ussr": d, "us_label": entrants[b]["label"], "ussr_label": entrants[a]["label"]})
        else:
            a, b = entrant(i, j), entrant(i, j, "b")
            cells.append({"us": i, "ussr": j, "us_label": entrants[a]["label"], "ussr_label": entrants[b]["label"]})
            cells.append({"us": i, "ussr": j, "us_label": entrants[b]["label"], "ussr_label": entrants[a]["label"]})
        matchups.append(sorted((a, b)))
    return {"us": list(us), "ussr": list(ussr), "player": player, "entrants": entrants,
            "matchups": matchups, "cells": cells}


# --- the matrix game ------------------------------------------------------------------------

FloatArray = npt.NDArray[np.float64]


def solve_zero_sum(a: FloatArray, eps: float = 1e-9) -> Tuple[FloatArray, FloatArray, float]:
    """An equilibrium of the zero-sum game where the row player gets `a[i, j]` and maximises:
    (row mix, column mix, value). Support enumeration over equal-size supports, which finds an
    equilibrium of every nondegenerate game; noisy measured payoffs are nondegenerate."""
    m, n = a.shape
    for k in range(1, min(m, n) + 1):
        for rows in itertools.combinations(range(m), k):
            for cols in itertools.combinations(range(n), k):
                sub = a[np.ix_(rows, cols)]
                # x on rows: x^T sub = v for every chosen column, sum x = 1.
                lhs = np.zeros((k + 1, k + 1))
                lhs[:k, :k], lhs[:k, k], lhs[k, :k] = sub.T, -1.0, 1.0
                rhs = np.zeros(k + 1)
                rhs[k] = 1.0
                lhs_y = np.zeros((k + 1, k + 1))
                lhs_y[:k, :k], lhs_y[:k, k], lhs_y[k, :k] = sub, -1.0, 1.0
                try:
                    sx = np.linalg.solve(lhs, rhs)
                    sy = np.linalg.solve(lhs_y, rhs)
                except np.linalg.LinAlgError:
                    continue
                if (sx[:k] < -eps).any() or (sy[:k] < -eps).any():
                    continue
                x, y = np.zeros(m), np.zeros(n)
                x[list(rows)], y[list(cols)] = sx[:k], sy[:k]
                v = float(sx[k])
                if (x @ a >= v - 1e-7).all() and (a @ y <= v + 1e-7).all():
                    return np.clip(x, 0, None), np.clip(y, 0, None), v
    raise ValueError("no equilibrium found by support enumeration (a degenerate game)")


# --- report ---------------------------------------------------------------------------------

def _score(winner: str) -> float:
    return 1.0 if winner == "US" else 0.5 if winner == "DRAW" else 0.0


def cell_scores(plan: Plan, games: Sequence[Mapping[str, object]]) -> Dict[Tuple[int, int], Dict[int, List[float]]]:
    """US score of every game, by cell and by deal seed. Refuses a game no cell accounts for."""
    by_seat = {(c["us_label"], c["ussr_label"]): (c["us"], c["ussr"]) for c in plan["cells"]}
    out: Dict[Tuple[int, int], Dict[int, List[float]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for g in games:
        key = (str(g["us_agent"]), str(g["ussr_agent"]))
        if key not in by_seat:
            raise ValueError(f"game {g.get('game_index')} ({key}) is not a cell of this plan")
        out[by_seat[key]][int(str(g["seed"]))].append(_score(str(g["winner"])))
    return out


def matrix_from(scores: Dict[Tuple[int, int], Dict[int, List[float]]], n_us: int, n_ussr: int,
                seeds: Optional[Sequence[int]] = None) -> Tuple[FloatArray, npt.NDArray[np.int64]]:
    """Mean US score and game count of every cell, over `seeds` (with repeats, for a bootstrap) or
    every deal."""
    mean, count = np.full((n_us, n_ussr), np.nan), np.zeros((n_us, n_ussr), dtype=np.int64)
    for (i, j), per_seed in scores.items():
        use = per_seed.keys() if seeds is None else seeds
        vals = [v for s in use for v in per_seed.get(s, [])]
        if vals:
            mean[i, j], count[i, j] = float(np.mean(vals)), len(vals)
    return mean, count


def paired(scores: Dict[Tuple[int, int], Dict[int, List[float]]], a: Tuple[int, int],
           b: Tuple[int, int]) -> Tuple[float, float, int]:
    """Mean of (cell a − cell b) over the deals both played, its standard error, and the deal count."""
    sa, sb = scores.get(a, {}), scores.get(b, {})
    common = sorted(set(sa) & set(sb))
    if len(common) < 2:
        return math.nan, math.nan, len(common)
    d = np.array([np.mean(sa[s]) - np.mean(sb[s]) for s in common])
    return float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d))), len(common)


def per_deal(scores: Dict[Tuple[int, int], Dict[int, List[float]]], n_us: int, n_ussr: int) -> FloatArray:
    """deals x cells: each cell's US score on each deal that every cell played (an odd matrix's
    last cell, played twice a deal, contributes its mean)."""
    cells = [(i, j) for i in range(n_us) for j in range(n_ussr)]
    common = set.intersection(*(set(scores.get(c, {})) for c in cells))
    return np.array([[float(np.mean(scores[c][s])) for c in cells] for s in sorted(common)])


def joint_test(m: FloatArray) -> Tuple[float, int, float]:
    """Do the cells differ at all? Hotelling's T² that every cell's mean equals the first's, from
    the deals x cells matrix (blocked by deal); returns (T², contrasts, p). The p is T² against
    chi-square with that many degrees of freedom by the Wilson-Hilferty approximation, which at
    thousands of deals matches the exact F test to the second decimal."""
    k = m.shape[1] - 1
    d = m[:, 1:] - m[:, :1]
    dm = d.mean(axis=0)
    t2 = float(len(d) * dm @ np.linalg.solve(np.cov(d, rowvar=False).reshape(k, k), dm))
    z = ((t2 / k) ** (1 / 3) - (1 - 2 / (9 * k))) / math.sqrt(2 / (9 * k))
    return t2, k, 0.5 * math.erfc(z / math.sqrt(2))


def contrast(m: FloatArray, weights: FloatArray) -> Tuple[float, float]:
    """A weighted sum of cells (weights shaped like the matrix), per deal: its mean and standard error."""
    x = m @ weights.reshape(-1)
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def describe(opening: str) -> str:
    """A one-sided opening as the influence it adds, in country names ("West Germany 4, Italy 4, Iran 1")."""
    from tools.lib.openings import opening_scripts

    names = {int(c["id"]): str(c["name"]) for c in json.load(open(_MAP_PATH))["countries"]}
    (script,) = opening_scripts(opening).values()
    counts = collections.Counter(script)      # most points first, ties in placement order
    return ", ".join(f"{names.get(c, str(c))} {n}" for c, n in sorted(counts.items(), key=lambda x: (-x[1], script.index(x[0]))))


def _pp(x: float, se: Optional[float] = None, signed: bool = False) -> str:
    if math.isnan(x):
        return "—"
    nd = 2 if se is not None and not math.isnan(se) and 100 * se < 0.5 else 1   # two decimals once they mean something
    s = f"{100 * x:+.{nd}f}" if signed else f"{100 * x:.{nd}f}"
    return s + (f" ± {100 * se:.{nd}f}" if se is not None and not math.isnan(se) else "")


def report(plan: Plan, games: Sequence[Mapping[str, object]], bootstrap: int = 1000, seed: int = 0) -> Tuple[str, MatrixData]:
    """The matrix, the paired comparisons with the networks' setups, and the solved game."""
    us, ussr = plan["us"], plan["ussr"]
    scores = cell_scores(plan, games)
    mean, count = matrix_from(scores, len(us), len(ussr))
    if np.isnan(mean).any():
        raise ValueError("a cell has no games")
    x, y, v = solve_zero_sum(mean)

    # The bootstrap resamples deals, which keeps every cell's games of one deal together.
    all_seeds = sorted({s for per in scores.values() for s in per})
    md = per_deal(scores, len(us), len(ussr))
    rng = np.random.default_rng(seed)
    bx, by, bv = np.zeros((bootstrap, len(us))), np.zeros((bootstrap, len(ussr))), np.zeros(bootstrap)
    for b in range(bootstrap):
        mb = md[rng.integers(0, len(md), size=len(md))].mean(axis=0).reshape(len(us), len(ussr))
        bx[b], by[b], bv[b] = solve_zero_sum(mb)

    lines = [f"# Setup matrix: {len(us)} US × {len(ussr)} USSR setups",
             "",
             f"Player for both sides after the setup: `{plan['player']}`. {int(count.sum()):,} games, "
             f"{len(all_seeds):,} deals; every cell is played on the same deals. Scores are the US's "
             "(a draw is half), in percent; ± is one standard error.",
             "",
             "## Setups", "",
             "| | US setup (influence added) | | USSR setup (influence added) |",
             "|:---|:---|:---|:---|"]
    for k in range(max(len(us), len(ussr))):
        u = f"U{k} | `{us[k]}`: {describe(us[k])}" if k < len(us) else " | "
        s = f"S{k} | `{ussr[k]}`: {describe(ussr[k])}" if k < len(ussr) else " | "
        lines.append(f"| {u} | {s} |")

    t2, dof, p_joint = joint_test(md)
    margins: Dict[str, Comparison] = {}
    for i in range(1, len(us)):
        w = np.zeros((len(us), len(ussr)))
        w[i, :], w[0, :] = 1 / len(ussr), -1 / len(ussr)
        dd, se = contrast(md, w)
        margins[f"U{i}-U0"] = {"diff": dd, "se": se, "deals": len(md)}
    for j in range(1, len(ussr)):
        w = np.zeros((len(us), len(ussr)))
        w[:, 0], w[:, j] = 1 / len(us), -1 / len(us)          # USSR points: US(S0) - US(Sj)
        dd, se = contrast(md, w)
        margins[f"S{j}-S0"] = {"diff": dd, "se": se, "deals": len(md)}

    lines += ["", "## Do the setups matter at all?", "",
              f"* **Joint test** that all {len(us) * len(ussr)} cells are equal, blocked by deal: "
              f"Hotelling T² = {t2:.1f} on {dof} contrasts, p = {p_joint:.3f}. Read the single "
              "comparisons below against it: with this many of them, a few at 2 SE are expected by chance.",
              "* **Each setup averaged over the other side's setups**, against the networks' own (paired by deal):"]
    lines += [f"  * {k} ({'US' if k.startswith('U') else 'USSR'} points): {_pp(c['diff'], c['se'], signed=True)}"
              for k, c in margins.items()]

    lines += ["", "## The matrix (US score)", "",
              "| US \\ USSR | " + " | ".join(f"S{j}" for j in range(len(ussr))) + " | mean |",
              "|:---|" + "---:|" * (len(ussr) + 1)]
    for i in range(len(us)):
        row = [_pp(mean[i, j], math.sqrt(mean[i, j] * (1 - mean[i, j]) / count[i, j])) for j in range(len(ussr))]
        lines.append(f"| U{i} | " + " | ".join(row) + f" | {_pp(float(mean[i].mean()))} |")
    lines.append("| mean | " + " | ".join(_pp(float(mean[:, j].mean())) for j in range(len(ussr))) + f" | {_pp(float(mean.mean()))} |")

    lines += ["", "## Against the networks' setups (paired by deal)", "",
              "Each US setup minus U0 against the same USSR setup, in points of the US score:", "",
              "| US setup | " + " | ".join(f"vs S{j}" for j in range(len(ussr))) + " |",
              "|:---|" + "---:|" * len(ussr)]
    comparisons: Dict[str, Comparison] = {}
    for i in range(1, len(us)):
        cells_txt = []
        for j in range(len(ussr)):
            d, se, n = paired(scores, (i, j), (0, j))
            comparisons[f"U{i}-U0@S{j}"] = {"diff": d, "se": se, "deals": n}
            cells_txt.append(_pp(d, se, signed=True))
        lines.append(f"| U{i} − U0 | " + " | ".join(cells_txt) + " |")
    lines += ["", "Each USSR setup minus S0 against the same US setup, in points of the *USSR* score:", "",
              "| USSR setup | " + " | ".join(f"vs U{i}" for i in range(len(us))) + " |",
              "|:---|" + "---:|" * len(us)]
    for j in range(1, len(ussr)):
        cells_txt = []
        for i in range(len(us)):
            d, se, n = paired(scores, (i, 0), (i, j))    # US(i, S0) − US(i, Sj) = USSR gain from Sj
            comparisons[f"S{j}-S0@U{i}"] = {"diff": d, "se": se, "deals": n}
            cells_txt.append(_pp(d, se, signed=True))
        lines.append(f"| S{j} − S0 | " + " | ".join(cells_txt) + " |")

    def mix(w: FloatArray, prefix: str, bw: FloatArray) -> str:
        return ", ".join(f"{prefix}{k} {100 * w[k]:.0f}% (in the support of {100 * float((bw[:, k] > 0.01).mean()):.0f}% of resamples)"
                         for k in range(len(w)) if w[k] > 1e-6 or (bw[:, k] > 0.01).mean() > 0.05)

    lo, hi = np.percentile(bv, [2.5, 97.5])
    lines += ["", "## The solved game", "",
              f"* **Value:** US {_pp(v)} [95% {_pp(float(lo))}, {_pp(float(hi))}] with both sides mixing optimally.",
              f"* **US mix:** {mix(x, 'U', bx)}.",
              f"* **USSR mix:** {mix(y, 'S', by)}.",
              f"* The networks' own pairing (U0 vs S0) scores {_pp(float(mean[0, 0]))} for the US.",
              "",
              f"Bootstrap: {bootstrap} resamples of the deals; a setup is \"in the support\" of a resample "
              "when its equilibrium weight there is above 1%."]
    data: MatrixData = {
        "plan": plan, "matrix": mean.tolist(), "games": count.tolist(), "deals": len(all_seeds),
        "equilibrium": {"us": x.tolist(), "ussr": y.tolist(), "value": v,
                        "value_ci95": [float(lo), float(hi)],
                        "us_support_freq": (bx > 0.01).mean(axis=0).tolist(),
                        "ussr_support_freq": (by > 0.01).mean(axis=0).tolist()},
        "comparisons": comparisons,
        "joint_test": {"t2": t2, "contrasts": dof, "p": p_joint},
        "margins": margins,
    }
    return "\n".join(lines) + "\n", data


# --- CLI ------------------------------------------------------------------------------------

def _load_plan(path: str) -> Plan:
    with open(path) as f:
        plan: Plan = json.load(f)
    return plan


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="entrants and pairings that cover the matrix")
    p.add_argument("--us", nargs="+", default=list(DEFAULT_US), help="US-only openings (tools/lib/openings.py); the first is the reference")
    p.add_argument("--ussr", nargs="+", default=list(DEFAULT_USSR), help="USSR-only openings; the first is the reference")
    p.add_argument("--player", default="{BASE}", help="load_agent spec of the player after the setup; {BASE} is --base")
    p.add_argument("--base", required=True, help="the checkpoint {BASE} stands for")
    p.add_argument("--out", required=True)
    a = sub.add_parser("args", help="print the plan's --models or --matchups values for tools/tournament.py")
    a.add_argument("--plan", required=True)
    a.add_argument("what", choices=("models", "matchups"))
    r = sub.add_parser("report", help="the matrix and its solution from a tournament's game log")
    r.add_argument("--plan", required=True)
    r.add_argument("--games", required=True, help="tools/tournament.py --log-games output")
    r.add_argument("--bootstrap", type=int, default=1000)
    r.add_argument("--out", required=True, help="Markdown report")
    r.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        plan = make_plan(args.us, args.ussr, args.player.replace("{BASE}", args.base))
        with open(args.out, "w") as f:
            json.dump(plan, f, indent=1)
        print(f"{len(plan['cells'])} cells, {len(plan['entrants'])} entrants, {len(plan['matchups'])} pairings -> {args.out}")
        return 0
    if args.cmd == "args":
        plan = _load_plan(args.plan)
        if args.what == "models":
            print(" ".join(e["spec"] for e in plan["entrants"]))
        else:
            print(" ".join(f"{i + 1}-{j + 1}" for i, j in plan["matchups"]))
        return 0
    plan = _load_plan(args.plan)
    with open(args.games) as f:
        games = [json.loads(line) for line in f if line.strip()]
    text, data = report(plan, games, bootstrap=args.bootstrap)
    with open(args.out, "w") as f:
        f.write(text)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(data, f, indent=1)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
