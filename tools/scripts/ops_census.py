#!/usr/bin/env python3
"""Where Operations go, country by country: humans against bots, in their own games and on the same
positions.

`placement_census.py` compares where each side's Ops influence lands in the humans' games and in a
bot's self-play. Those are different games: a bot that never lets the Middle East become contested
will place there less even if it would choose the same targets in the same spot. This tool adds the
comparison on the same positions, and the other two uses of Ops:

* **records** -- every target chosen with Operations in an action round: a point of Influence, a
  coup, a realignment roll (not an event's placement, coup or realignment, and not the setup), with
  the mover, turn, round, mode, country and the scoring cards the mover held;
* `human` -- the ts-replayer corpus's records, each with its position (`pos=` token);
* `selfplay` -- a checkpoint's greedy self-play, the same records without positions;
* `onhuman` -- the checkpoint's policy at every human record's position: its probability for each
  country, and its own pick;
* `report` -- per side and mode, each country's share of the humans' targets against each bot's
  share in its own games and on the humans' positions, the same-position difference with a standard
  error clustered by game (the points of one play, and the plays of one game, are not independent),
  and how often the bot's pick is the human's.

    export PYTHONPATH=.:build/release
    python tools/scripts/ops_census.py human --out human_ops.jsonl.gz
    python tools/scripts/ops_census.py selfplay --checkpoint soup.pt --games 4096 --out soup_ops.json.gz
    python tools/scripts/ops_census.py onhuman --human human_ops.jsonl.gz --checkpoint soup.pt \\
        --out soup_onhuman.json.gz
    python tools/scripts/ops_census.py report --human human_ops.jsonl.gz --selfplay soup=soup_ops.json.gz \\
        --onhuman soup=soup_onhuman.json.gz --out ops_by_country.md
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import (corpus_map, feed_corpus_game, load_policy, position_token,
                                     selfplay, state_from_token)
from tools.scripts.placement_census import COUNTRY, ERAS, REGIONS, SCORING_CARDS, era_of, region

US, USSR = 1, -1
MODES = {ts.OpMode.INFLUENCE: "influence", ts.OpMode.COUP: "coup", ts.OpMode.REALIGN: "realign"}
MODE_NAMES = ("influence", "coup", "realign")
SIDE_NAMES = {US: "US", USSR: "USSR"}
#: The country tables' periods. The Early War's positions follow the humans' openings, which the
#: bots do not play, so its same-position gaps are kept apart from the rest of the game's.
PERIODS = (("Early War, turns 1-3", (1, 3)), ("Mid and Late War, turns 4-10", (4, 10)))


def mover(st: ts.GameState) -> ts.Player:
    ctx = st.ctx()
    return ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player


def ops_target(st: ts.GameState, a: int) -> Optional[Tuple[str, int]]:
    """(mode, country) when this decision is a target chosen with Operations in an action round."""
    ctx = st.ctx()
    if (st.current_phase != ts.Phase.ACTION_ROUND or ctx.decision_type != ts.DecisionType.POINT_NODE
            or int(ctx.resolving_card) != 0 or ctx.op_mode not in MODES):
        return None
    cid = int(ts.decode_flat_action(st, a).primary_id)
    if cid not in COUNTRY:
        return None
    return MODES[ctx.op_mode], cid


def held_mask(st: ts.GameState, who: ts.Player) -> int:
    return sum(1 << i for i, (card, _) in enumerate(SCORING_CARDS)
               if ts.in_hand_of(st.get_card_location(card), who))


class OpsTracker:
    """Every Ops target of one game. With `positions`, each record keeps its position's token."""

    def __init__(self, positions: bool = False) -> None:
        self.positions = positions
        self.records: List[Dict[str, Any]] = []

    def observe(self, st: ts.GameState, a: int) -> None:
        t = ops_target(st, a)
        if t is None:
            return
        who = mover(st)
        rec: Dict[str, Any] = {"side": int(who), "turn": int(st.turn), "ar": int(st.action_round),
                               "mode": t[0], "c": t[1], "held": held_mask(st, who)}
        if self.positions:
            rec["pos"] = position_token(st)
        self.records.append(rec)


def _human_game(path: str) -> Tuple[str, List[Dict[str, Any]], str]:
    t = OpsTracker(positions=True)
    status = feed_corpus_game(path, t)
    return os.path.basename(path), (t.records if status != "skipped" else []), status


def _write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> None:
    with gzip.open(path, "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def human(out: str, workers: int, limit: int) -> int:
    rows: List[Dict[str, Any]] = []
    status: Dict[str, int] = collections.Counter()
    t0 = time.time()
    for game, recs, st in sorted(corpus_map(_human_game, workers, limit), key=lambda x: x[0]):
        status[st] += 1
        rows += [dict(r, g=game) for r in recs]
    _write_jsonl(out, rows)
    meta = {"games": dict(status), "records": dict(collections.Counter(r["mode"] for r in rows)),
            "seconds": round(time.time() - t0, 1)}
    json.dump(meta, open(out + ".meta.json", "w"))
    print(json.dumps(meta), file=sys.stderr)
    return 0


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_selfplay(checkpoint: str, games: int, seed: int, batch: int, out: str) -> int:
    fn, features = load_policy(checkpoint)
    t0 = time.time()
    trackers = selfplay(fn, features, games, seed, batch, 0.0, OpsTracker)
    rows = [dict(r, g=i) for i, t in enumerate(trackers) for r in t.records]
    with gzip.open(out, "wt") as f:
        json.dump({"checkpoint": os.path.basename(checkpoint), "sha256": _sha256(checkpoint), "games": games,
                   "seed": seed, "seconds": round(time.time() - t0, 1), "records": rows}, f)
    print(f"{len(rows):,} records from {games:,} games in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def onhuman(human_path: str, checkpoint: str, out: str, chunk: int) -> int:
    """The checkpoint's policy at each human record's position: {country: probability} over the legal
    targets (those above 1e-4), and its argmax country."""
    rows = _read_jsonl(human_path)
    fn, features = load_policy(checkpoint)
    res: List[Dict[str, Any]] = []
    t0 = time.time()
    for lo in range(0, len(rows), chunk):
        states = [state_from_token(r["pos"]) for r in rows[lo:lo + chunk]]
        obs = np.stack([np.asarray(ts.extract_observation_features(s, mover(s), features), dtype=np.float32)
                        for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        logits = np.asarray(fn(obs, masks), dtype=np.float64)
        for st, lg, m in zip(states, logits, masks):
            legal = np.flatnonzero(m)
            z = np.exp(lg[legal] - lg[legal].max())
            p = z / z.sum()
            dist: Dict[int, float] = collections.defaultdict(float)
            for a, pa in zip(legal, p):
                cid = int(ts.decode_flat_action(st, int(a)).primary_id)
                dist[cid if cid in COUNTRY else -1] += float(pa)
            best = int(ts.decode_flat_action(st, int(legal[int(np.argmax(p))])).primary_id)
            res.append({"p": {str(c): round(v, 5) for c, v in dist.items() if v >= 1e-4}, "pick": best})
        print(f"  {len(res):,}/{len(rows):,} in {time.time() - t0:.0f}s", file=sys.stderr)
    with gzip.open(out, "wt") as f:
        json.dump({"checkpoint": os.path.basename(checkpoint), "sha256": _sha256(checkpoint),
                   "human": os.path.basename(human_path), "rows": res}, f)
    return 0


# --- the report -------------------------------------------------------------------------------

def clustered(x: np.ndarray, groups: Sequence[Any]) -> Tuple[float, float]:
    """Mean of x and its standard error clustered by group."""
    n = len(x)
    if n == 0:
        return float("nan"), float("nan")
    m = float(x.mean())
    sums: Dict[Any, float] = collections.defaultdict(float)
    counts: Dict[Any, int] = collections.Counter()
    for v, g in zip(x, groups):
        sums[g] += float(v)
        counts[g] += 1
    k = len(sums)
    if k < 2:
        return m, float("nan")
    resid = np.array([sums[g] - counts[g] * m for g in sums])
    return m, float(math.sqrt(k / (k - 1) * float((resid ** 2).sum())) / n)


def share_of(recs: Sequence[Dict[str, Any]], cid: int) -> Tuple[float, float]:
    x = np.array([1.0 if r["c"] == cid else 0.0 for r in recs])
    return clustered(x, [r["g"] for r in recs])


def _pct(x: float, digits: int = 1) -> str:
    return "—" if x != x else f"{100 * x:.{digits}f}"


def country_table(side: int, mode: str, hum: List[Dict[str, Any]], turns: Tuple[int, int],
                  own: Dict[str, List[Dict[str, Any]]], on: Dict[str, List[Dict[str, Any]]],
                  top: int, min_share: float) -> List[str]:
    """One side and mode over `turns` (first, last): the countries ranked by the largest
    same-position gap."""
    lo, hi = turns
    sel = [i for i, r in enumerate(hum) if r["side"] == side and r["mode"] == mode and lo <= r["turn"] <= hi]
    if not sel:
        return ["(no human records)"]
    games = [hum[i]["g"] for i in sel]
    own_sel = {k: [r for r in v if r["side"] == side and r["mode"] == mode and lo <= r["turn"] <= hi]
               for k, v in own.items()}
    rows = []
    for cid in COUNTRY:
        h = np.array([1.0 if hum[i]["c"] == cid else 0.0 for i in sel])
        hs = float(h.mean())
        cells = {}
        for k, o in on.items():
            p = np.array([float(o[i]["p"].get(str(cid), 0.0)) for i in sel])
            d, se = clustered(h - p, games)
            pick = float(np.mean([1.0 if o[i]["pick"] == cid else 0.0 for i in sel]))
            cells[k] = (float(p.mean()), pick, d, se)
        owns = {k: (share_of(v, cid)[0] if v else float("nan")) for k, v in own_sel.items()}
        biggest = max([hs] + [c[0] for c in cells.values()] + [x for x in owns.values() if x == x])
        if biggest < min_share:
            continue
        key = min((abs(c[2]) for c in cells.values()), default=0.0)
        rows.append((-key, int(cid), hs, owns, cells))
    rows.sort()
    names = list(on)
    own_names = list(own)
    head = ("| country | region | humans | " + " | ".join(f"{k} own games" for k in own_names) + " | "
            + " | ".join(f"{k} same positions (pick)" for k in names) + " | "
            + " | ".join(f"humans − {k}, same positions" for k in names) + " |")
    out = [head, "|:---|:---|---:|" + "---:|" * (len(own_names) + 2 * len(names))]
    for _, cid, hs, owns, cells in rows[:top]:
        out.append(f"| {COUNTRY[cid]['name']} | {region(cid)} | {_pct(hs)} | "
                   + " | ".join(_pct(owns[k]) for k in own_names) + " | "
                   + " | ".join(f"{_pct(cells[k][0])} ({_pct(cells[k][1])})" for k in names) + " | "
                   + " | ".join(f"**{100 * cells[k][2]:+.1f} ± {100 * cells[k][3]:.1f}**"
                                if abs(cells[k][2]) >= 2 * cells[k][3] else
                                f"{100 * cells[k][2]:+.1f} ± {100 * cells[k][3]:.1f}" for k in names) + " |")
    return out


def region_table(side: int, mode: str, hum: List[Dict[str, Any]], own: Dict[str, List[Dict[str, Any]]],
                 on: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    sel = [i for i, r in enumerate(hum) if r["side"] == side and r["mode"] == mode]
    names, own_names = list(on), list(own)
    out = ["| region | era | humans | " + " | ".join(f"{k} own games" for k in own_names) + " | "
           + " | ".join(f"{k} same positions" for k in names) + " | "
           + " | ".join(f"humans − {k}, same positions" for k in names) + " | human points |",
           "|:---|:---|---:|" + "---:|" * (len(own_names) + 2 * len(names) + 1)]
    for reg in REGIONS:
        for era, lo, hi in ERAS:
            idx = [i for i in sel if lo <= hum[i]["turn"] <= hi]
            if not idx:
                continue
            h = np.array([1.0 if region(hum[i]["c"]) == reg else 0.0 for i in idx])
            g = [hum[i]["g"] for i in idx]
            owns = []
            for k in own_names:
                v = [r for r in own[k] if r["side"] == side and r["mode"] == mode and lo <= r["turn"] <= hi]
                owns.append(_pct(float(np.mean([region(r["c"]) == reg for r in v])) if v else float("nan"), 0))
            sp, dd = [], []
            for k in names:
                p = np.array([sum(v for c, v in on[k][i]["p"].items() if c != "-1" and region(int(c)) == reg)
                              for i in idx])
                d, se = clustered(h - p, g)
                sp.append(_pct(float(p.mean()), 0))
                dd.append(f"**{100 * d:+.0f} ± {100 * se:.0f}**" if abs(d) >= 2 * se else f"{100 * d:+.0f} ± {100 * se:.0f}")
            out.append(f"| {reg} | {era} | {_pct(float(h.mean()), 0)} | " + " | ".join(owns) + " | "
                       + " | ".join(sp) + " | " + " | ".join(dd) + f" | {len(idx):,} |")
    return out


def agreement_lines(hum: List[Dict[str, Any]], on: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    out = ["| side | mode | era | human targets | " + " | ".join(f"{k}: pick = human's" for k in on)
           + " | " + " | ".join(f"{k}: p(human's)" for k in on) + " |",
           "|:---|:---|:---|---:|" + "---:|" * (2 * len(on))]
    for side in (US, USSR):
        for mode in MODE_NAMES:
            for era, lo, hi in ERAS:
                idx = [i for i, r in enumerate(hum) if r["side"] == side and r["mode"] == mode and lo <= r["turn"] <= hi]
                if len(idx) < 30:
                    continue
                g = [hum[i]["g"] for i in idx]
                agree, ph = [], []
                for k, o in on.items():
                    a, se = clustered(np.array([1.0 if o[i]["pick"] == hum[i]["c"] else 0.0 for i in idx]), g)
                    agree.append(f"{100 * a:.0f}% ± {100 * se:.0f}")
                    ph.append(f"{100 * float(np.mean([o[i]['p'].get(str(hum[i]['c']), 0.0) for i in idx])):.0f}%")
                out.append(f"| {SIDE_NAMES[side]} | {mode} | {era} | {len(idx):,} | " + " | ".join(agree)
                           + " | " + " | ".join(ph) + " |")
    return out


def report(human_path: str, selfplays: Sequence[str], onhumans: Sequence[str], top: int, min_share: float,
           out: str) -> int:
    hum = _read_jsonl(human_path)
    meta = json.load(open(human_path + ".meta.json")) if os.path.exists(human_path + ".meta.json") else {}
    own: Dict[str, List[Dict[str, Any]]] = {}
    own_meta: Dict[str, Dict[str, Any]] = {}
    for spec in selfplays:
        name, _, path = spec.partition("=")
        d = json.load(gzip.open(path, "rt"))
        own[name] = d["records"]
        own_meta[name] = {k: d[k] for k in ("checkpoint", "sha256", "games", "seed", "seconds")}
    on: Dict[str, List[Dict[str, Any]]] = {}
    on_meta: Dict[str, str] = {}
    for spec in onhumans:
        name, _, path = spec.partition("=")
        d = json.load(gzip.open(path, "rt"))
        if len(d["rows"]) != len(hum):
            raise SystemExit(f"{path}: {len(d['rows'])} rows for {len(hum)} human records")
        on[name] = d["rows"]
        on_meta[name] = f"{d['checkpoint']} ({d['sha256'][:12]}…)"
    games_h = len({r["g"] for r in hum})
    lines = ["# Where Operations go, humans against bots, country by country", "",
             f"Humans: the ts-replayer corpus, {games_h} games with Ops targets "
             f"({json.dumps(meta.get('games', {}))}); "
             + ", ".join(f"{sum(r['mode'] == m for r in hum):,} {m}" for m in MODE_NAMES) + ". "
             + "Bots' own games: " + "; ".join(f"**{k}** {v['games']:,} greedy self-play games of {v['checkpoint']} "
                                               f"({v['sha256'][:12]}…)" for k, v in own_meta.items()) + ". "
             + "Same positions: " + "; ".join(f"**{k}** {v}" for k, v in on_meta.items()) + ", asked at every "
             "human target's position.", "",
             "A share is the side's targets in that mode that went to the country. *Same positions* is the bot's "
             "probability for the country, averaged over the humans' decisions (its own pick in brackets). The "
             "gap is humans − bot on those decisions, ± one standard error clustered by game; bold where it is "
             "2+ SE.", ""]
    for mode in MODE_NAMES:
        for side in (US, USSR):
            lines += [f"## {SIDE_NAMES[side]} {mode}", "", "### Regions by era", ""]
            lines += region_table(side, mode, hum, own, on) + [""]
            for label, turns in PERIODS:
                lines += [f"### Countries, {label} (the {top} largest same-position gaps; countries with "
                          f"{100 * min_share:.0f}%+ somewhere)", ""]
                lines += country_table(side, mode, hum, turns, own, on, top, min_share) + [""]
    lines += ["## How often the bot picks the human's target", ""] + agreement_lines(hum, on) + [""]
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("human")
    h.add_argument("--out", required=True)
    h.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    h.add_argument("--limit", type=int, default=0)
    s = sub.add_parser("selfplay")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--games", type=int, default=4096)
    s.add_argument("--seed", type=int, default=55_000)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--out", required=True)
    o = sub.add_parser("onhuman")
    o.add_argument("--human", required=True)
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--chunk", type=int, default=2048)
    o.add_argument("--out", required=True)
    r = sub.add_parser("report")
    r.add_argument("--human", required=True)
    r.add_argument("--selfplay", nargs="*", default=[], metavar="NAME=PATH")
    r.add_argument("--onhuman", nargs="*", default=[], metavar="NAME=PATH")
    r.add_argument("--top", type=int, default=15)
    r.add_argument("--min-share", type=float, default=0.01)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "human":
        return human(a.out, a.workers, a.limit)
    if a.cmd == "selfplay":
        return run_selfplay(a.checkpoint, a.games, a.seed, a.batch, a.out)
    if a.cmd == "onhuman":
        return onhuman(a.human, a.checkpoint, a.out, a.chunk)
    return report(a.human, a.selfplay, a.onhuman, a.top, a.min_share, a.out)


if __name__ == "__main__":
    raise SystemExit(main())
