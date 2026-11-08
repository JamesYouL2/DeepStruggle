#!/usr/bin/env python3
"""What would P42's until-control macro have been worth to a frozen network?

P42 asks for a macro slot so a takeover is one choice instead of a product of small probabilities.
Before spending a training run on the view, this prices the takeovers a base network passes up, in
its own positions and with its own play afterwards. No network is trained or grown, and the engine
is not changed: the macro view is already there, it is just not one the network decides in.

    # 1. the opportunities: where a run of influence points starts and a macro is legal (CI, in parts)
    python tools/scripts/takeover_pricing.py collect --model j1.pt --games 400 --part 1/4 \\
        --sample 500 --out-dir out/part-1

    # 2. a price for every move: raw's line against each macro on shared redeals and dice
    #    (search_bank.yml stage=playouts, release asset bank.jsonl.gz = the pooled rows)
    # 3. search_bank.py confirm-select, then the escalated rows again on fresh pairs

    # 4. the report
    python tools/scripts/takeover_pricing.py report --rows rows.jsonl.gz --summary summary-*.json \\
        --screen screen.jsonl.gz --confirm-input confirm_input.jsonl.gz --confirm confirm.jsonl.gz --out report.md

An opportunity is a *run start*, the first point of a run, so a run is counted once and the macro's
choice is the whole run's. It counts if at least one macro raises the mover's own influence (an event
removing influence has macro slots too, and they are not takeovers). A macro the network's own greedy
line already plays is dropped: pricing it would compare a line with itself, and the rate is reported.

Two readings come out. Per macro, the **screen's lead** over raw is unselected, so it says what a kind of
takeover is worth on average when offered. Per opportunity, the gain from **playing the screen's best move
where it clears the bar**, read only on the fresh pairs (a row that does not clear it counts zero), is
what a chooser could have collected: it is unbiased, unlike the screen's maximum, and a lower bound on
the best macro's worth. A clear gain supports P42. No gain does not rule it out: greedy raw play after
the macro may not follow it up, and the critic-free playout says nothing of what a network trained in
the view would learn to do with one.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import load_policy, position_token, rules_json, selfplay, state_from_token
from tools.search_bank import decision_card, mover_of, read_rows

SCHEMA = 1
MACRO = ActionEncoder.MACRO_NODE_OFFSET
NODE = ActionEncoder.NODE_OFFSET
COUNTRIES = ActionEncoder.FLAT_ACTION_SIZE - MACRO
#: a different part plays different games
PART_SEED = 1_000_003

LogitsFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def influence(st: ts.GameState, side: ts.Player, country: int) -> int:
    c = st.get_country(country)
    return int(c.us_influence if side == ts.Player.US else c.ussr_influence)


def opportunity_kind(st: ts.GameState) -> Optional[str]:
    """What the run of points at this `POINT_NODE` is for; None for what is no influence placement (a
    coup or realignment target is a `POINT_NODE` too)."""
    ctx = st.ctx()
    if st.current_phase == ts.Phase.SETUP:
        return "setup"
    if int(ctx.resolving_card):
        return "event"
    if int(ctx.pending_op_card) and ctx.op_mode == ts.OpMode.INFLUENCE:
        return "ops"
    return None


def same_run(prev: Optional[Tuple[Any, ...]], key: Tuple[Any, ...]) -> bool:
    """`key` continues the run `prev` left off: the same decider, phase, round and card, and the
    point counter still counting down. A counter that does not fall is a new allowance: the setup's
    US 7 then US 2 are two runs."""
    return prev is not None and prev[:-1] == key[:-1] and key[-1] < prev[-1]


def run_key(st: ts.GameState) -> Tuple[Any, ...]:
    ctx = st.ctx()
    return (int(ctx.decision_player), str(st.current_phase), int(st.turn), int(st.action_round),
            decision_card(st), int(ctx.remaining_steps))


def macro_slots(st: ts.GameState) -> List[int]:
    """The macro slots legal here."""
    mask = np.asarray(ts.Engine.get_flat_action_mask(st, False, True))
    return [int(a) for a in np.flatnonzero(mask[MACRO:]) + MACRO]


def expansion(st: ts.GameState, country: int) -> List[ts.GameState]:
    """The positions a macro on `country` passes through, `st` first, the last one past its final
    point: its base steps one at a time, for the network's probability of each. The stopping rule is
    `ActionMask::apply_macro_expansion`'s (a point the engine refuses ends it too: the last Op cannot
    pay 2 into a country the opponent controls), and a test holds the two to the same position."""
    side = st.ctx().decision_player
    chain = [st]
    cur = st
    while True:
        nxt = cur.clone()
        if not ts.Engine.try_step_flat(nxt, NODE + country):
            return chain
        chain.append(nxt)
        ctx = nxt.ctx()
        if (ts.Engine.is_terminal(nxt) or ctx.decision_type != ts.DecisionType.POINT_NODE
                or ctx.decision_player != side or ts.Scoring.get_country_control(nxt, country) == side):
            return chain
        cur = nxt


def takeover_macros(st: ts.GameState) -> List[Dict[str, Any]]:
    """The legal macros that raise the mover's own influence, with what each does."""
    side = st.ctx().decision_player
    out: List[Dict[str, Any]] = []
    for slot in macro_slots(st):
        c = slot - MACRO
        after = st.clone()
        ts.Engine.step_flat(after, slot, False, False, True)
        placed = influence(after, side, c) - influence(st, side, c)
        if placed > 0:
            out.append({"slot": slot, "country": c, "placed": placed,
                        "control": ts.Scoring.get_country_control(after, c) == side})
    return out


class RunStarts:
    """A tracker (tools/lib/corpus_driver.py): keeps the position at each opportunity of one game."""

    def __init__(self) -> None:
        self.positions: List[Tuple[str, str]] = []
        self.run_starts = 0
        self.prev: Optional[Tuple[Any, ...]] = None

    def observe(self, st: ts.GameState, action: int) -> None:
        if st.ctx().decision_type != ts.DecisionType.POINT_NODE:
            self.prev = None
            return
        key = run_key(st)
        start = not same_run(self.prev, key)
        self.prev = key
        if not start:
            return
        self.run_starts += 1
        kind = opportunity_kind(st)
        if kind is not None and takeover_macros(st):
            self.positions.append((position_token(st), kind))


def softmax_probs(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The network's masked distribution, per row (logits may already hold -inf at illegal slots)."""
    z = np.where(mask > 0, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def policy_at(states: Sequence[ts.GameState], fn: LogitsFn, features: int) -> Tuple[np.ndarray, np.ndarray]:
    """The network's logits and legal mask at each position, in the base view it decides in."""
    obs = np.stack([np.asarray(ts.extract_observation_features(s, mover_of(s), features), dtype=np.float32)
                    for s in states])
    masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
    return np.asarray(fn(obs, masks), dtype=np.float64), masks


def greedy_line(st: ts.GameState, fn: LogitsFn, features: int) -> Tuple[int, Dict[int, int]]:
    """Raw's greedy move here, and the points its greedy line puts in each country over the run."""
    first = -1
    points: Dict[int, int] = defaultdict(int)
    cur = st.clone()
    prev = run_key(cur)
    while True:
        logits, _ = policy_at([cur], fn, features)
        a = int(logits[0].argmax())
        if first < 0:
            first = a
        if not NODE <= a < NODE + COUNTRIES:
            return first, dict(points)
        points[a - NODE] += 1
        ts.Engine.step_flat(cur, a)
        if ts.Engine.is_terminal(cur) or cur.ctx().decision_type != ts.DecisionType.POINT_NODE:
            return first, dict(points)
        key = run_key(cur)
        if not same_run(prev, key):
            return first, dict(points)
        prev = key


def line_probabilities(st: ts.GameState, macros: Sequence[Dict[str, Any]], fn: LogitsFn, features: int) -> List[float]:
    """Per macro, the product of the network's probabilities of its base steps: how often raw's
    policy, sampling, would play exactly this takeover point by point."""
    chains = [expansion(st, int(m["country"]))[:-1] for m in macros]
    out = [1.0] * len(macros)
    depth = 0
    while True:
        live = [i for i, ch in enumerate(chains) if depth < len(ch)]
        if not live:
            return out
        logits, masks = policy_at([chains[i][depth] for i in live], fn, features)
        probs = softmax_probs(logits, masks)
        for row, i in enumerate(live):
            out[i] *= float(probs[row, NODE + int(macros[i]["country"])])
        depth += 1


def priced_macros(macros: List[Dict[str, Any]], top: int, spread: int, seed: str) -> List[Dict[str, Any]]:
    """The macros a row prices: raw's `top` likeliest lines, and `spread` drawn at random from the rest,
    since a takeover at the policy's floor is what P42 is for and the top alone would never show one.
    The draw is seeded from the row, so it does not depend on the part. `picked` says which is which."""
    ranked = sorted(macros, key=lambda m: -float(m["p_line"]))
    head, rest = ranked[:top], ranked[top:]
    rng = np.random.default_rng(int(seed[:12], 16))
    drawn = [rest[int(i)] for i in sorted(rng.choice(len(rest), size=min(spread, len(rest)), replace=False))]
    return [dict(m, picked="top") for m in head] + [dict(m, picked="spread") for m in drawn]


def build_row(st: ts.GameState, token: str, kind: str, game: str, fn: LogitsFn, features: int, top: int,
              spread: int) -> Tuple[Optional[Dict[str, Any]], Dict[str, int]]:
    """A bank row for one opportunity (None if every macro is already in raw's line), and its counts.
    `search_bank.py playouts` reads `id`, `pos` and `pmoves`; `confirm-select` also reads `raw`."""
    macros = takeover_macros(st)
    raw, line = greedy_line(st, fn, features)
    fresh = [m for m in macros if line.get(int(m["country"]), 0) < int(m["placed"])]
    counts = {"macros": len(macros), "in_line": len(macros) - len(fresh), "unpriced": 0}
    if not fresh:
        return None, counts
    for m, p in zip(fresh, line_probabilities(st, fresh, fn, features)):
        m["p_line"] = p
    rid = hashlib.sha256(token.encode()).hexdigest()[:16]
    priced = priced_macros(fresh, top, spread, rid)
    counts["unpriced"] = len(fresh) - len(priced)
    return {
        "schema": SCHEMA, "id": rid, "pos": token, "raw": raw, "pmoves": [raw] + [int(m["slot"]) for m in priced],
        "macros": priced, "offered": len(fresh), "kind": kind, "game": game,
        "side": "US" if mover_of(st) == ts.Player.US else "USSR", "turn": int(st.turn), "ar": int(st.action_round),
        "card": decision_card(st), "points": int(st.ctx().remaining_steps),
        "line": {str(c): n for c, n in line.items()},
    }, counts


def collect(a: argparse.Namespace) -> int:
    k, n = (int(x) for x in a.part.split("/"))
    fn, features = load_policy(a.model)
    games = a.games // n + (1 if k <= a.games % n else 0)
    t0 = time.time()
    trackers = selfplay(fn, features, games, a.seed + PART_SEED * k, a.batch, a.temperature, RunStarts)
    print(f"{games} games, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    found = [(tok, kind, f"{k}:{g}") for g, tr in enumerate(trackers) for tok, kind in tr.positions]
    rng = np.random.default_rng(a.seed + PART_SEED * k + 1)
    picked = [found[int(i)] for i in sorted(rng.choice(len(found), size=min(a.sample, len(found)), replace=False))]
    totals: Counter[str] = Counter()
    os.makedirs(a.out_dir, exist_ok=True)
    with gzip.open(os.path.join(a.out_dir, "rows.jsonl.gz"), "wt") as f:
        for i, (tok, kind, game) in enumerate(picked):
            try:
                st = state_from_token(tok)
            except (RuntimeError, ValueError) as e:
                # A position the save format will not load (AR9 Space Walk) has no price; it is counted.
                print(f"unloadable: {e}", file=sys.stderr, flush=True)
                totals["unloadable"] += 1
                continue
            row, counts = build_row(st, tok, kind, game, fn, features, a.top_macros, a.spread_macros)
            totals.update(counts)
            totals["sampled"] += 1
            if row is None:
                totals["all_in_line"] += 1
            else:
                totals["rows"] += 1
                totals[f"rows_{kind}"] += 1
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
            if (i + 1) % 100 == 0:
                print(f"{i + 1}/{len(picked)} rows, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    summary = {
        "schema": SCHEMA, "model": os.path.basename(a.model), "part": a.part, "seed": a.seed,
        "temperature": a.temperature, "games": games, "top_macros": a.top_macros, "spread_macros": a.spread_macros,
        "run_starts": sum(t.run_starts for t in trackers), "opportunities": len(found),
        "opportunities_by_kind": dict(Counter(kind for _, kind, _ in found)), "wall_s": round(time.time() - t0, 1),
        **totals,
    }
    with open(os.path.join(a.out_dir, "summary.json"), "w") as sf:
        json.dump(summary, sf, indent=1)
    print(json.dumps(summary), flush=True)
    return 0


def period(turn: int) -> str:
    if turn <= 0:
        return "setup"
    return "early (1-3)" if turn <= 3 else "mid (4-7)" if turn <= 7 else "late (8-10)"


def clustered(values: Sequence[Tuple[str, float]]) -> Tuple[float, float]:
    """The mean of (cluster, value) pairs and its standard error, clustered: positions of one game
    are not independent."""
    if not values:
        return 0.0, 0.0
    mean = sum(v for _, v in values) / len(values)
    by: Dict[str, List[float]] = defaultdict(list)
    for g, v in values:
        by[g].append(v)
    if len(by) < 2:
        return mean, 0.0
    resid = [sum(vs) - mean * len(vs) for vs in by.values()]
    return mean, math.sqrt(sum(r * r for r in resid) * len(by) / (len(by) - 1)) / len(values)


def sparse_mean(values: Sequence[Tuple[str, float]], n: int, clusters: int) -> Tuple[float, float]:
    """The mean over `n` observations in `clusters` clusters when only (cluster, value) pairs of the
    nonzero ones are listed, and its standard error: the variance of the clusters' totals, the
    clusters with nothing listed included at zero."""
    if n == 0:
        return 0.0, 0.0
    total = sum(v for _, v in values)
    if clusters < 2:
        return total / n, 0.0
    by: Dict[str, float] = defaultdict(float)
    for g, v in values:
        by[g] += v
    each = total / clusters
    spread = sum((t - each) ** 2 for t in by.values()) + (clusters - len(by)) * each ** 2
    return total / n, math.sqrt(spread * clusters / (clusters - 1)) / n


def p_bucket(p: float) -> str:
    return "< 0.1%" if p < 1e-3 else "0.1-1%" if p < 1e-2 else "1-10%" if p < 0.1 else ">= 10%"


def row_leads(row: Dict[str, Any], scores: Dict[str, List[float]]) -> List[Tuple[Dict[str, Any], float]]:
    """Each priced macro with its mean lead over raw on these pairs (win probability)."""
    base = np.asarray(scores[str(int(row["raw"]))], dtype=float)
    out = []
    for m in row["macros"]:
        got = scores.get(str(int(m["slot"])))
        if got is not None:
            out.append((m, float((np.asarray(got, dtype=float) - base).mean())))
    return out


GROUPS: Dict[str, Callable[[Dict[str, Any], Dict[str, Any], Dict[int, Dict[str, Any]]], str]] = {
    "side": lambda r, m, c: str(r["side"]),
    "kind": lambda r, m, c: str(r["kind"]),
    "side and kind": lambda r, m, c: f"{r['side']} {r['kind']}",
    "period": lambda r, m, c: period(int(r["turn"])),
    "region": lambda r, m, c: str(c[int(m["country"])]["region"]),
    "battleground": lambda r, m, c: "battleground" if c[int(m["country"])].get("battleground") else "not",
    "reaches control": lambda r, m, c: "control" if m["control"] else "points run out",
    "points placed": lambda r, m, c: str(min(int(m["placed"]), 5)) + ("+" if int(m["placed"]) >= 5 else ""),
    "raw's probability of the line": lambda r, m, c: p_bucket(float(m["p_line"])),
    "how picked": lambda r, m, c: str(m["picked"]),
}


def analyse(rows: Sequence[Dict[str, Any]], summaries: Sequence[Dict[str, Any]], screen: Dict[str, Dict[str, Any]],
            selected: Dict[str, Dict[str, Any]], confirm: Dict[str, Dict[str, Any]],
            countries: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """The whole report's numbers. `selected` is confirm-select's output, `confirm` the fresh pairs."""
    games = sum(int(s["games"]) for s in summaries)
    opps = sum(int(s["opportunities"]) for s in summaries)
    sampled = sum(int(s.get("sampled", 0)) for s in summaries)
    per_game = opps / games if games else 0.0

    # Unselected: every macro's screen lead.
    leads: List[Tuple[Dict[str, Any], Dict[str, Any], float]] = []
    for r in rows:
        got = screen.get(str(r["id"]))
        if got is None:
            raise ValueError(f"row {r['id']} has no screen playouts")
        leads += [(r, m, lead) for m, lead in row_leads(r, got["scores"])]
    unselected: Dict[str, List[Dict[str, Any]]] = {}
    for name, key in GROUPS.items():
        bucket: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        for r, m, lead in leads:
            bucket[key(r, m, countries)].append((str(r["game"]), lead))
        unselected[name] = [{"group": g, "n": len(v), "lead": clustered(v)[0], "se": clustered(v)[1]}
                            for g, v in sorted(bucket.items())]
    per_country: Dict[int, List[Tuple[str, float]]] = defaultdict(list)
    for r, m, lead in leads:
        per_country[int(m["country"])].append((str(r["game"]), lead))
    country_rows = [{"country": countries[c]["name"], "n": len(v), "lead": clustered(v)[0], "se": clustered(v)[1],
                     "p_line": float(np.median([float(m["p_line"]) for _, m, _ in leads if int(m["country"]) == c]))}
                    for c, v in per_country.items()]

    # Selected: the fresh pairs of each escalated row; every other sampled opportunity counts zero.
    gains: List[Tuple[Dict[str, Any], Dict[str, Any], float, float]] = []
    for r in rows:
        s = selected.get(str(r["id"]))
        if s is None:
            raise ValueError(f"row {r['id']} did not go through confirm-select")
        if not s.get("escalated"):
            continue
        got = confirm.get(str(r["id"]))
        if got is None:
            raise ValueError(f"escalated row {r['id']} has no confirm playouts")
        best = int(s["best"])
        d = np.asarray(got["scores"][str(best)], dtype=float) - np.asarray(got["scores"][str(int(r["raw"]))], dtype=float)
        macro = next((m for m in r["macros"] if int(m["slot"]) == best), None)
        if macro is None:
            raise ValueError(f"row {r['id']}: best {best} is no priced macro")
        gains.append((r, macro, float(d.mean()), float(d.std(ddof=1) / math.sqrt(len(d)))))
    mean, se = sparse_mean([(str(r["game"]), g) for r, _, g, _ in gains], sampled, games)
    by_group: Dict[str, List[Dict[str, Any]]] = {}
    for name, key in GROUPS.items():
        bucket = defaultdict(list)
        for r, m, g, _ in gains:
            bucket[key(r, m, countries)].append((str(r["game"]), g))
        by_group[name] = [{"group": g, "n": len(v), "points_a_game": 100 * per_game * sum(x for _, x in v) / sampled}
                          for g, v in sorted(bucket.items())] if sampled else []
    tally: Dict[str, List[float]] = defaultdict(list)
    for _, m, g, _ in gains:
        tally[countries[int(m["country"])]["name"]].append(g)
    by_country = sorted(({"country": c, "n": len(v), "points_a_game": 100 * per_game * sum(v) / sampled}
                         for c, v in tally.items()), key=lambda x: -float(x["points_a_game"])) if sampled else []
    tail = sorted(((r, m, g, e) for r, m, g, e in gains if e > 0 and g / e >= 2), key=lambda t: -t[2])
    return {
        "games": games, "run_starts": sum(int(s.get("run_starts", 0)) for s in summaries), "opportunities": opps,
        "opportunities_a_game": per_game, "by_kind": dict(sum((Counter(s.get("opportunities_by_kind", {})) for s in summaries), Counter())),
        "sampled": sampled, "rows": len(rows), "all_in_line": sum(int(s.get("all_in_line", 0)) for s in summaries),
        "unloadable": sum(int(s.get("unloadable", 0)) for s in summaries),
        "macros": sum(int(s.get("macros", 0)) for s in summaries), "in_line": sum(int(s.get("in_line", 0)) for s in summaries),
        "unpriced": sum(int(s.get("unpriced", 0)) for s in summaries),
        "unselected": unselected, "countries": sorted(country_rows, key=lambda x: -float(x["n"])),
        "escalated": len(gains), "gain": mean, "gain_se": se, "gain_points_a_game": 100 * per_game * mean,
        "gain_points_a_game_se": 100 * per_game * se, "gain_by_group": by_group, "gain_by_country": by_country,
        "positive_fresh": sum(1 for _, _, g, _ in gains if g > 0), "tail_count": len(tail),
        "tail": [{"id": r["id"], "side": r["side"], "turn": r["turn"], "ar": r["ar"], "kind": r["kind"], "card": r["card"],
                  "country": countries[int(m["country"])]["name"], "placed": m["placed"], "control": m["control"],
                  "p_line": m["p_line"], "lead": g, "se": e, "pos": r["pos"]} for r, m, g, e in tail],
    }


def _table(head: Sequence[str], body: Iterable[Sequence[str]]) -> List[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---:|" * len(head)] + ["| " + " | ".join(row) + " |" for row in body]


def render(x: Dict[str, Any], top: int) -> str:
    out = ["# What the until-control macro would price at, on a frozen network", ""]
    out += [f"{x['games']} self-play games; {x['run_starts']} runs of points, {x['opportunities']} with a macro that raises the "
            f"mover's influence ({x['opportunities_a_game']:.1f} a game; {x['by_kind']}). {x['sampled']} opportunities sampled "
            f"uniformly; in {x['all_in_line']} of them raw's own line already plays every macro, and {x['unloadable']} would not load.",
            f"Macros offered: {x['macros']}; already in raw's line: {x['in_line']} "
            f"({100 * x['in_line'] / max(1, x['macros']):.1f}%); left unpriced by the cap on macros a row: {x['unpriced']} "
            "(the top lines by raw's probability and a random draw from the rest are priced; the *spread* group below is that draw).", ""]
    out += ["## The gain a chooser collects", "",
            f"**{x['gain_points_a_game']:+.2f} ± {x['gain_points_a_game_se']:.2f} points of win probability a game** "
            f"({100 * x['gain']:+.3f} ± {100 * x['gain_se']:.3f} an opportunity): the screen's best macro played wherever it "
            "cleared z >= 1.5 and +1 point, read on fresh pairs; every other opportunity counts zero; standard error clustered "
            f"by game. {x['escalated']} rows escalated, {x['positive_fresh']} of them positive on the fresh pairs, "
            f"{x['tail_count']} at z >= 2 ({0.023 * x['escalated']:.1f} expected at z >= 2 if every lead were zero).", "",
            "This is a lower bound on the best macro's worth and it plays raw greedily afterwards, so it is not what a network "
            "trained in the view would do with the macro.", ""]
    for name, rows in x["gain_by_group"].items():
        out += [f"### Gain by {name}", ""] + _table([name, "rows", "points a game"], (
            (str(r["group"]), str(r["n"]), f"{r['points_a_game']:+.2f}") for r in rows)) + [""]
    out += ["### Gain by country", ""] + _table(["country", "rows", "points a game"], (
        (str(r["country"]), str(r["n"]), f"{r['points_a_game']:+.2f}") for r in x["gain_by_country"][:top])) + [""]
    out += ["## What a macro is worth when offered", "",
            "Each priced macro's lead over raw's move on the screen's pairs, in points of win probability. Nothing here is "
            "selected, so a negative mean is what most takeovers really cost, not noise around a maximum.", ""]
    for name, rows in x["unselected"].items():
        out += [f"### By {name}", ""] + _table([name, "macros", "lead", "se"], (
            (str(r["group"]), str(r["n"]), f"{100 * r['lead']:+.2f}", f"{100 * r['se']:.2f}") for r in rows)) + [""]
    out += ["### By country (most often offered first)", ""] + _table(
        ["country", "macros", "lead", "se", "median p(line)"],
        ((str(r["country"]), str(r["n"]), f"{100 * r['lead']:+.2f}", f"{100 * r['se']:.2f}", f"{100 * r['p_line']:.2f}%")
         for r in x["countries"])) + [""]
    out += [f"## Largest confirmed leads (z >= 2 on the fresh pairs, top {top})", ""] + _table(
        ["side", "turn", "AR", "kind", "card", "macro", "points", "control", "p(line)", "lead", "se"],
        ((str(t["side"]), str(t["turn"]), str(t["ar"]), str(t["kind"]), str(t["card"]), str(t["country"]), str(t["placed"]),
          "yes" if t["control"] else "no", f"{100 * t['p_line']:.2f}%", f"{100 * t['lead']:+.1f}", f"{100 * t['se']:.1f}")
         for t in x["tail"][:top])) + [""]
    return "\n".join(out)


def report(a: argparse.Namespace) -> int:
    countries = {int(c["id"]): c for c in rules_json("map.json")["countries"]}
    summaries = []
    for p in a.summary:
        with open(p) as f:
            summaries.append(json.load(f))
    rows = read_rows(a.rows)
    x = analyse(rows, summaries, {str(r["id"]): r for r in read_rows(a.screen)},
                {str(r["id"]): r for r in read_rows(a.confirm_input)},
                {str(r["id"]): r for r in read_rows(a.confirm)}, countries)
    with open(a.out, "w") as f:
        f.write(render(x, a.top))
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(x, f)
    print(f"gain {x['gain_points_a_game']:+.2f} +- {x['gain_points_a_game_se']:.2f} points a game; {a.out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="self-play the network and build this part's bank rows")
    c.add_argument("--model", required=True, help="the checkpoint, .pt or .onnx")
    c.add_argument("--games", type=int, default=400, help="games over all parts")
    c.add_argument("--part", default="1/1", help="k/N: this part's share of the games")
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--temperature", type=float, default=0.1)
    c.add_argument("--batch", type=int, default=32)
    c.add_argument("--sample", type=int, default=1000, help="opportunities this part prices")
    c.add_argument("--top-macros", type=int, default=8, help="macros a row: raw's likeliest lines")
    c.add_argument("--spread-macros", type=int, default=4, help="macros a row: drawn at random from the rest")
    c.add_argument("--out-dir", required=True)
    r = sub.add_parser("report", help="the report, from the pooled rows and their playouts")
    r.add_argument("--rows", nargs="+", required=True)
    r.add_argument("--summary", nargs="+", required=True)
    r.add_argument("--screen", nargs="+", required=True, help="search_bank.py playouts over the rows")
    r.add_argument("--confirm-input", nargs="+", required=True, help="search_bank.py confirm-select's output")
    r.add_argument("--confirm", nargs="+", required=True, help="search_bank.py playouts over the escalated rows")
    r.add_argument("--top", type=int, default=30)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return collect(a) if a.cmd == "collect" else report(a)


if __name__ == "__main__":
    sys.exit(main())
