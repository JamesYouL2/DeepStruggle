#!/usr/bin/env python3
"""Humans against bots, card by card: how each card in hand gets used.

Reads holdings dumps from `event_play_census.py` -- one from `--human-corpus`, one or more from
self-play -- and lays them side by side. A census rate is a property of one checkpoint unless other
checkpoints agree (`research/log/event_census_E7-02-44_E7-08-43.md`: about one card in five moves
10 points between two seeds of one recipe), so a gap is **robust** only when every bot sits on the
same side of the humans, each by at least `--min-gap` points and more than 3 standard errors.

Sections:

* **Events** -- owner holdings (neutral cards: either side), evented = headlined or played for the
  event in a round, the census's own measure; robust gaps first.
* **What humans do almost always** -- cards whose commonest use among the humans (headline, event in a
  round, Ops, space) reaches `--auto` of the holdings, with the bots' share of that same use.
* **Neutral cards by side** -- the evented share when the US holds the card and when the USSR does.
* **The opponent's cards** -- what the holder does with a card whose event belongs to the other side:
  headline it, play it for Ops, space it, or keep it.

    PYTHONPATH=. python tools/scripts/event_census_compare.py --human human.json \\
        --bot SWA=swa.json E7-20-44=e720.json soup=soup.json --output-md compare.md
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from typing import Dict, List, Optional, Sequence, Tuple

US, USSR = 1, -1
USES = ("headline", "event", "ops", "space", "kept")
USE_NAMES = {"headline": "headline", "event": "event in a round", "ops": "Ops", "space": "space", "kept": "kept"}
#: one dumped holding: card, side, outcome, legal (the event was playable at some point)
Rec = Tuple[int, int, str, bool]


def load(path: str) -> Tuple[List[Rec], int]:
    d = json.load(open(path))
    recs = [(int(r[0]), int(r[1]), str(r[2]), bool(r[3]) if len(r) > 3 else True) for r in d["holdings"]]
    return recs, int(d["games"])


def cards() -> Dict[int, Dict]:
    return {int(c["id"]): c for c in json.load(open("rules/cards.json"))}


def side_of(c: Dict) -> str:
    return str(c.get("side", "neutral")).lower()


class Counts:
    """Outcome counts per (card, holder side), split by whether the event was ever playable."""

    def __init__(self, recs: Sequence[Rec]) -> None:
        self.legal: Dict[Tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
        self.all: Dict[Tuple[int, int], collections.Counter] = collections.defaultdict(collections.Counter)
        for card, side, outcome, legal in recs:
            # Dumps from before the split say "ops/space"; they cannot answer the Ops/space questions.
            self.all[(card, side)][outcome] += 1
            if legal:
                self.legal[(card, side)][outcome] += 1

    def get(self, keys: Sequence[Tuple[int, int]], legal_only: bool = True) -> collections.Counter:
        src = self.legal if legal_only else self.all
        out: collections.Counter = collections.Counter()
        for k in keys:
            out.update(src.get(k, collections.Counter()))
        return out


def owner_keys(cid: int, c: Dict) -> List[Tuple[int, int]]:
    s = side_of(c)
    return [(cid, US)] if s == "us" else [(cid, USSR)] if s == "ussr" else [(cid, US), (cid, USSR)]


def rate(cnt: collections.Counter, uses: Sequence[str]) -> Tuple[float, float, int]:
    """Share of the holdings that ended in one of `uses`, its standard error, and n."""
    n = sum(cnt.values())
    if n == 0:
        return math.nan, math.nan, 0
    p = sum(cnt[u] for u in uses) / n
    return p, math.sqrt(max(p * (1 - p), 1.0 / n) / n), n


def gap_verdict(h: Tuple[float, float, int], bots: Sequence[Tuple[float, float, int]],
                min_gap: float, min_n: int) -> Optional[int]:
    """+1 when the humans are above every bot, -1 when below every bot, each by `min_gap` and more
    than 3 standard errors; None otherwise."""
    if h[2] < min_n or any(b[2] < min_n for b in bots):
        return None
    signs = set()
    for b in bots:
        d = h[0] - b[0]
        if abs(d) < min_gap or abs(d) <= 3 * math.hypot(h[1], b[1]):
            return None
        signs.add(1 if d > 0 else -1)
    return signs.pop() if len(signs) == 1 else None


def pct(p: float) -> str:
    return "—" if math.isnan(p) else f"{100 * p:.0f}%"


def cell(r: Tuple[float, float, int]) -> str:
    return "—" if r[2] == 0 else f"{100 * r[0]:.0f}% ({r[2]:,})"


def report(human: Tuple[List[Rec], int], bots: Dict[str, Tuple[List[Rec], int]], min_gap: float,
           min_n: int, auto: float) -> str:
    info = cards()
    H = Counts(human[0])
    B = {name: Counts(recs) for name, (recs, _) in bots.items()}
    names = list(bots)
    out = [f"# Humans against bots, card by card", "",
           f"Humans: {human[1]:,} games of the ts-replayer corpus. Bots (greedy self-play): "
           + ", ".join(f"**{n}** {g:,} games" for n, (_, g) in bots.items()) + ".", "",
           f"A gap is **robust** (marked ▲ humans higher, ▼ humans lower) when every bot is on the same "
           f"side of the humans by at least {100 * min_gap:.0f} points and more than 3 standard errors, "
           f"with at least {min_n} holdings in every column.", ""]

    # ---- events
    rows = []
    for cid, c in info.items():
        if cid == 6:                                          # The China Card: no event
            continue
        keys = owner_keys(cid, c)
        h = rate(H.get(keys), ("headline", "event"))
        bs = [rate(B[n].get(keys), ("headline", "event")) for n in names]
        if h[2] == 0:
            continue
        v = gap_verdict(h, bs, min_gap, min_n)
        mean_b = sum(b[0] for b in bs if b[2]) / max(1, sum(1 for b in bs if b[2]))
        rows.append((v is None, -abs(h[0] - mean_b), c["name"], side_of(c), h, bs, v, mean_b))
    rows.sort(key=lambda r: (r[0], r[1]))
    robust = sum(1 for r in rows if r[6] is not None)
    out += ["## Events", "",
            "Owner holdings (neutral cards: either side) in which the event could be played; evented = "
            "headlined or played for the event in a round. Robust gaps first, largest first, then the rest "
            f"by the gap to the bots' mean. {robust} of {len(rows)} cards differ robustly.", "",
            "| card | side | humans | " + " | ".join(names) + " | humans − bots |",
            "|:---|:---|---:|" + "---:|" * len(names) + "---:|"]
    for _, _, name, side, h, bs, v, mb in rows:
        mark = "" if v is None else (" ▲" if v > 0 else " ▼")
        out.append(f"| {name} | {side} | {cell(h)} | " + " | ".join(cell(b) for b in bs)
                   + f" | {100 * (h[0] - mb):+.0f}{mark} |")

    # ---- what humans do almost always
    out += ["", f"## What humans do almost always", "",
            f"Cards whose commonest use among the humans reaches {100 * auto:.0f}% of the owner's holdings "
            f"(at least {min_n}), with each bot's share of that same use. ▼ marks a bot below the humans by "
            f"{100 * min_gap:.0f}+ points and 3 standard errors.", "",
            "| card | side | humans' use | humans | " + " | ".join(names) + " |",
            "|:---|:---|:---|---:|" + "---:|" * len(names)]
    auto_rows = []
    for cid, c in info.items():
        if cid == 6:
            continue
        keys = owner_keys(cid, c)
        hc = H.get(keys)
        n = sum(hc.values())
        if n < min_n:
            continue
        use = max(("headline", "event", "ops", "space"), key=lambda u: hc[u])
        h = rate(hc, (use,))
        if h[0] < auto:
            continue
        cells = []
        for nm in names:
            b = rate(B[nm].get(keys), (use,))
            low = b[2] >= min_n and h[0] - b[0] >= min_gap and h[0] - b[0] > 3 * math.hypot(h[1], b[1])
            cells.append(cell(b) + (" ▼" if low else ""))
        auto_rows.append((-h[0], c["name"], side_of(c), USE_NAMES[use], cell(h), cells))
    auto_rows.sort()
    for _, name, side, use, hcell, cells in auto_rows:
        out.append(f"| {name} | {side} | {use} | {hcell} | " + " | ".join(cells) + " |")

    # ---- neutral cards by side
    out += ["", "## Neutral cards by side", "",
            "Evented share when each side holds the card. ▲/▼ as above, per side.", "",
            "| card | humans US | " + " | ".join(f"{n} US" for n in names)
            + " | humans USSR | " + " | ".join(f"{n} USSR" for n in names) + " |",
            "|:---|" + "---:|" * (2 * len(names) + 2)]
    for cid, c in info.items():
        if side_of(c) != "neutral" or cid == 6:
            continue
        parts = []
        for side in (US, USSR):
            h = rate(H.get([(cid, side)]), ("headline", "event"))
            bs = [rate(B[n].get([(cid, side)]), ("headline", "event")) for n in names]
            v = gap_verdict(h, bs, min_gap, min_n)
            mark = "" if v is None else (" ▲" if v > 0 else " ▼")
            parts += [cell(h) + mark] + [cell(b) for b in bs]
        if any(p != "—" for p in parts):
            out.append(f"| {c['name']} | " + " | ".join(parts) + " |")

    # ---- the opponent's cards
    out += ["", "## The opponent's cards", "",
            "Every holding of a card whose event belongs to the other side: the holder headlines it (firing "
            "the opponent's event), plays it for Ops (the event fires too, before or after), spaces it "
            "(the event does not fire), or keeps it. Humans' split, then each bot's space share; cards with "
            f"at least {min_n} human holdings, by the humans' space share. ▲/▼: the humans' space share "
            "against every bot.", "",
            "| card | owner | humans (n) | headline | Ops | space | kept | " + " | ".join(f"{n} space" for n in names) + " |",
            "|:---|:---|---:|---:|---:|---:|---:|" + "---:|" * len(names)]
    opp_rows = []
    for cid, c in info.items():
        s = side_of(c)
        if s not in ("us", "ussr"):
            continue
        holder = USSR if s == "us" else US
        hc = H.get([(cid, holder)], legal_only=False)
        n = sum(hc.values())
        if n < min_n:
            continue
        sp = rate(hc, ("space",))
        bs = [rate(B[nm].get([(cid, holder)], legal_only=False), ("space",)) for nm in names]
        legacy = any(B[nm].get([(cid, holder)], legal_only=False)["ops/space"] for nm in names)
        v = None if legacy else gap_verdict(sp, bs, min_gap, min_n)
        mark = "" if v is None else (" ▲" if v > 0 else " ▼")
        opp_rows.append((-sp[0], c["name"], s, n, [pct(hc[u] / n) for u in ("headline", "ops", "space", "kept")],
                         mark, ["—" if legacy else cell(b) for b in bs]))
    opp_rows.sort()
    for _, name, s, n, split, mark, bcells in opp_rows:
        out.append(f"| {name} | {s} | {n:,} | {split[0]} | {split[1]} | {split[2]}{mark} | {split[3]} | "
                   + " | ".join(bcells) + " |")
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--human", required=True, help="dump from event_play_census.py --human-corpus")
    ap.add_argument("--bot", nargs="+", required=True, help="NAME=dump from event_play_census.py self-play")
    ap.add_argument("--min-gap", type=float, default=0.10)
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--auto", type=float, default=0.85, help="the share that counts as 'almost always'")
    ap.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    bots = {}
    for spec in a.bot:
        name, _, path = spec.partition("=")
        bots[name] = load(path)
    md = report(load(a.human), bots, a.min_gap, a.min_n, a.auto)
    print(md)
    if a.output_md:
        open(a.output_md, "w").write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
