#!/usr/bin/env python3
"""Where Ops influence goes, humans against bots: by region, era, and the region's scoring card.

One record per point of Influence placed with Operations in an action round -- not an event's
placement (Decolonization, Marshall Plan ...) and not the setup -- with the mover, the turn, the
country, and whether the mover held a scoring card for that country's region at the time (Asia
Scoring covers Southeast Asia too). The same counting runs over a policy's self-play and over the
human ts-replayer corpus, through `event_play_census`'s drivers, so the rows line up.

Three tables per comparison:

* **regions by era** -- each region's share of the Ops influence placed, Early (turns 1-3), Mid (4-7)
  and Late War (8-10), and the battleground share;
* **holding the region's scoring card** -- a region's share when the mover holds its scoring card
  against when it does not: how strongly placement follows the card;
* **countries** -- per side, the countries whose share differs most between the humans and every bot
  in the same direction.

The points of one play are not independent (four Ops into one region are one decision), so no
standard error is printed; read small differences as noise.

    PYTHONPATH=.:build/release python tools/scripts/placement_census.py --human-corpus --dump human_pl.json
    PYTHONPATH=.:build/release python tools/scripts/placement_census.py --checkpoint swa.onnx --dump swa_pl.json
    PYTHONPATH=. python tools/scripts/placement_census.py --compare human_pl.json SWA=swa_pl.json ...
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ts_engine as ts

from tools.scripts.event_play_census import corpus_map, feed_corpus_game, load_policy, selfplay

US, USSR = 1, -1
_MAP = json.load(open("rules/map.json"))["countries"]
COUNTRY = {int(c["id"]): c for c in _MAP}
REGIONS = ("Europe", "Asia", "Middle East", "Africa", "Central America", "South America")
#: the scoring cards, in the order of the bits of a record's `held` mask, and what each scores
SCORING_CARDS = ((2, "Europe"), (1, "Asia"), (3, "Middle East"), (79, "Africa"), (37, "Central America"),
                 (81, "South America"), (38, "Southeast Asia"))
ERAS = (("Early War", 1, 3), ("Mid War", 4, 7), ("Late War", 8, 10))
#: one record: side, turn, country id, and a mask of the scoring cards the mover held (SCORING_CARDS)
Rec = Tuple[int, int, int, int]


def region(cid: int) -> str:
    return str(COUNTRY[cid]["region"])


def in_area(cid: int, area: str) -> bool:
    """Whether a scoring card for `area` scores this country (Southeast Asia is part of Asia)."""
    if area == "Southeast Asia":
        return "Southeast Asia" in (COUNTRY[cid].get("subregions") or [])
    return region(cid) == area


class PlacementTracker:
    def __init__(self) -> None:
        self.records: List[Rec] = []

    def observe(self, st: ts.GameState, a: int) -> None:
        ctx = st.ctx()
        if (st.current_phase != ts.Phase.ACTION_ROUND or ctx.decision_type != ts.DecisionType.POINT_NODE
                or ctx.op_mode != ts.OpMode.INFLUENCE or int(ctx.resolving_card) != 0):
            return
        cid = int(ts.decode_flat_action(st, a).primary_id)
        if cid not in COUNTRY:
            return
        mover = ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player
        held = sum(1 << i for i, (card, _) in enumerate(SCORING_CARDS)
                   if ts.in_hand_of(st.get_card_location(card), mover))
        self.records.append((int(mover), int(st.turn), cid, held))


def _human_game(path: str) -> Tuple[List[Rec], str]:
    t = PlacementTracker()
    status = feed_corpus_game(path, t)
    return (t.records if status != "skipped" else []), status


def era_of(turn: int) -> str:
    return next(name for name, lo, hi in ERAS if lo <= turn <= hi)


def share(recs: Sequence[Rec], pred: Any) -> Tuple[float, int]:
    n = len(recs)
    return (sum(1 for r in recs if pred(r)) / n if n else float("nan")), n


def pct(x: float) -> str:
    return "—" if x != x else f"{100 * x:.0f}%"


def compare(human: List[Rec], bots: Dict[str, List[Rec]], top: int) -> str:
    names = list(bots)
    sets = {"humans": human, **bots}
    cols = list(sets)
    out = ["# Ops influence placement, humans against bots", "",
           "Points of Influence placed with Operations in action rounds (not events, not the setup). "
           + ", ".join(f"**{k}** {len(v):,} points" for k, v in sets.items()) + ".", "",
           "## Regions by era", ""]
    for era, lo, hi in ERAS:
        sub = {k: [r for r in v if lo <= r[1] <= hi] for k, v in sets.items()}
        out += [f"### {era} (turns {lo}-{hi})", "",
                "| region | " + " | ".join(f"{k} ({len(sub[k]):,})" for k in cols) + " | humans − bots |",
                "|:---|" + "---:|" * len(cols) + "---:|"]
        for reg in REGIONS + ("battlegrounds",):
            if reg == "battlegrounds":
                pred = (lambda r: bool(COUNTRY[r[2]]["battleground"]))
            else:
                pred = (lambda r, g=reg: region(r[2]) == g)
            vals = {k: share(sub[k], pred)[0] for k in cols}
            bm = [vals[n] for n in names if vals[n] == vals[n]]
            d = vals["humans"] - sum(bm) / len(bm) if bm else float("nan")
            label = "**battleground share**" if reg == "battlegrounds" else reg
            out.append(f"| {label} | " + " | ".join(pct(vals[k]) for k in cols)
                       + f" | {'—' if d != d else f'{100 * d:+.0f}'} |")
        out.append("")

    out += ["## Holding the region's scoring card", "",
            "The share of the Ops influence placed that goes into an area while the mover holds that area's "
            "scoring card, and while it does not (all eras; Southeast Asia is the subregion its own card scores, "
            "and is part of Asia for Asia Scoring).", "",
            "| area | " + " | ".join(f"{k}: holding / not" for k in cols) + " |",
            "|:---|" + "---:|" * len(cols)]
    for bit, (_, area) in enumerate(SCORING_CARDS):
        cells = []
        for k in cols:
            holding = [r for r in sets[k] if r[3] >> bit & 1]
            not_holding = [r for r in sets[k] if not r[3] >> bit & 1]
            into = (lambda r, g=area: in_area(r[2], g))
            (ha, hn), (na, _) = share(holding, into), share(not_holding, into)
            cells.append(f"{pct(ha)} / {pct(na)} ({hn:,})")
        out.append(f"| {area} | " + " | ".join(cells) + " |")
    out += ["", "(n) is the number of points placed while holding the card.", ""]

    out += ["## Countries", "",
            f"Per side, the countries whose share of that side's Ops influence differs most between the humans and "
            f"the bots, every bot on the same side of the humans (top {top}).", ""]
    for side, sname in ((US, "US"), (USSR, "USSR")):
        sub = {k: [r for r in v if r[0] == side] for k, v in sets.items()}
        cnt = {k: collections.Counter(r[2] for r in v) for k, v in sub.items()}
        rows = []
        for cid in COUNTRY:
            vals = {k: (cnt[k][cid] / len(sub[k]) if sub[k] else float("nan")) for k in cols}
            diffs = [vals["humans"] - vals[n] for n in names]
            if not diffs or not (all(d > 0 for d in diffs) or all(d < 0 for d in diffs)):
                continue
            rows.append((-abs(min(diffs, key=abs)), COUNTRY[cid]["name"], vals, min(diffs, key=abs)))
        rows.sort()
        out += [f"### {sname}", "", "| country | region | " + " | ".join(cols) + " | smallest humans − bot gap |",
                "|:---|:---|" + "---:|" * len(cols) + "---:|"]
        for _, name, vals, d in rows[:top]:
            cid = next(c for c in COUNTRY if COUNTRY[c]["name"] == name)
            out.append(f"| {name} | {region(cid)} | " + " | ".join(f"{100 * vals[k]:.1f}%" for k in cols)
                       + f" | {100 * d:+.1f} pp |")
        out.append("")
    return "\n".join(out)


def load_records(path: str) -> List[Rec]:
    return [(int(r[0]), int(r[1]), int(r[2]), int(r[3])) for r in json.load(open(path))["records"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="a .pt or .onnx policy: count its self-play")
    ap.add_argument("--human-corpus", action="store_true", help="count the ts-replayer corpus")
    ap.add_argument("--compare", nargs="+", default=None, metavar="DUMP",
                    help="HUMAN_DUMP NAME=BOT_DUMP ...: tabulate instead of counting")
    ap.add_argument("--games", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=55_000)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--corpus-limit", type=int, default=0)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--dump", default=None)
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    if a.compare:
        human = load_records(a.compare[0])
        bots = {}
        for spec in a.compare[1:]:
            name, _, path = spec.partition("=")
            bots[name] = load_records(path)
        md = compare(human, bots, a.top)
        print(md)
        if a.output_md:
            open(a.output_md, "w").write(md + "\n")
        return 0
    if a.human_corpus:
        recs: List[Rec] = []
        status: Dict[str, int] = collections.Counter()
        for rs, st in corpus_map(_human_game, a.workers, a.corpus_limit):
            status[st] += 1
            recs += rs
        games = status.get("complete", 0) + status.get("partial", 0)
        print(f"corpus: {dict(status)}", file=sys.stderr)
    elif a.checkpoint:
        fn, features = load_policy(a.checkpoint)
        recs = [r for t in selfplay(fn, features, a.games, a.seed, a.batch, 0.0, PlacementTracker) for r in t.records]
        games = a.games
    else:
        ap.error("give --checkpoint, --human-corpus or --compare")
    print(f"{len(recs):,} points from {games:,} games", file=sys.stderr)
    if a.dump:
        json.dump({"games": games, "records": recs}, open(a.dump, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
