"""Turn 1 as the network plays it: headlines, the USSR's first action round, the US's reply.

Reads the rows of `tools/turn1_pricing.py run` (every turn-1 decision with its options, the
network's pick and the position before it). Two uses of the same rows:

* **Census** (any number of worlds; `--worlds 2` makes 10,000 games cheap): how often each
  headline is chosen when held; the USSR's AR1 card, mode and target, and when it is *not* the
  Iran coup, what it is and what in the hands or headlines goes with that; the US's AR1 placement
  as the whole spread (the board after AR1 minus the board before it), its countries, and what
  shifts them.
* **Priced** (rows with >= 8 worlds): each option's critic value at the turn's end minus the
  network's pick's, averaged over the decisions where the option was legal -- for headline cards,
  the USSR's AR1 coup targets and the US's AR1 first points. A fixed option compared with the pick
  needs no split-half correction; picking the best of them afterwards would.

    python tools/turn1_census.py --rows rows*.jsonl.gz --out census.md [--json census.json]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import re
import sys
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, TypedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import ts_engine as ts  # noqa: E402

from tools.turn1_pricing import Row, state_from_token  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
_NAME = re.compile(r"\((.*)\)\s*$")
IRAN = 25


def option_label(name: str) -> str:
    """'SelectCard #25 (Containment)' -> 'Containment'; 'Resolution: OPS_COUP / OpMode: COUP' -> 'COUP'."""
    m = _NAME.search(name)
    if m:
        return m.group(1)
    if "OpMode:" in name:
        return name.rsplit("OpMode:", 1)[1].strip()
    if "Resolution:" in name:
        return name.split("Resolution:", 1)[1].strip()
    return name


class AR1(TypedDict):
    card: str
    mode: str                 # EVENT / INFLUENCE / COUP / REALIGN / SPACE, or SCORING / FORCED (no choice)
    ops: str                  # what the round's Operations did: COUP / INFLUENCE / REALIGN, "" if none --
                              # an event played first (an opponent's card) still has its Ops after it
    coup: str                 # the coup target, "" if no coup
    target: str               # coup target, first influence point or first realignment, "" if none
    spread: Dict[str, int]    # the side's influence added by the round, by country


class Game(TypedDict):
    seed: int
    opening: str
    us_headline: str
    ussr_headline: str
    us_hand: List[str]        # at the US headline
    ussr_hand: List[str]
    ussr_ar1: Optional[AR1]
    us_ar1: Optional[AR1]


def _hand(st: ts.GameState, side: ts.Player, names: Mapping[int, str]) -> List[str]:
    return sorted(names[c] for c in range(1, 111) if ts.in_hand_of(st.get_card_location(c), side))


def _influence(st: ts.GameState, side: ts.Player) -> np.ndarray:
    return np.array([int(st.get_country(c).us_influence if side == ts.Player.US else st.get_country(c).ussr_influence)
                     for c in range(84)])


def games_from_rows(rows: Sequence[Row]) -> List[Game]:
    """One record per game; a game missing a row this needs keeps None there."""
    cards = {int(c["id"]): str(c["name"]) for c in json.load(open(os.path.join(_ROOT, "rules", "cards.json")))}
    countries = {int(c["id"]): str(c["name"]) for c in json.load(open(os.path.join(_ROOT, "rules", "map.json")))["countries"]}
    by_game: Dict[Tuple[str, int], List[Row]] = collections.defaultdict(list)
    for r in rows:
        by_game[(r["opening"], r["seed"])].append(r)
    out: List[Game] = []
    for (opening, seed), rs in sorted(by_game.items()):
        rs.sort(key=lambda r: r["choice"])

        def first(side: str, ar: int, kind: str, phase: str = "ACTION_ROUND", after: int = -1) -> Optional[Row]:
            for r in rs:
                if r["side"] == side and r["ar"] == ar and r["kind"] == kind and r["phase"] == phase and r["choice"] > after:
                    return r
            return None

        uh, sh = first("US", 0, "headline", "HEADLINE"), first("USSR", 0, "headline", "HEADLINE")
        if uh is None or sh is None:
            continue
        st0 = state_from_token(uh["pos"])

        def ar1(side: str, nxt: Optional[Row]) -> Optional[AR1]:
            c = first(side, 1, "card to play")
            if c is None or nxt is None:
                return None
            mode_r = first(side, 1, "event or Ops", after=c["choice"])
            # a mode row belongs to this round only if it comes before the next round's first row
            if mode_r is not None and mode_r["choice"] > nxt["choice"]:
                mode_r = None
            card = option_label(c["names"][c["taken"]])
            # no event-or-Ops decision: a scoring card, or a card with only one way to play it
            mode = option_label(mode_r["names"][mode_r["taken"]]) if mode_r else ("SCORING" if "Scoring" in card else "FORCED")
            inside = [r for r in rs if c["choice"] < r["choice"] < nxt["choice"] and r["side"] == side]
            ops = mode if mode in ("COUP", "INFLUENCE", "REALIGN") else next(
                (option_label(r["names"][r["taken"]]) for r in inside if r["kind"] == "Ops mode"), "")
            coup = next((option_label(r["names"][r["taken"]]) for r in inside if r["kind"] == "coup target"), "")
            tgt = next((option_label(r["names"][r["taken"]]) for r in inside
                        if r["kind"] in ("coup target", "influence (first point)", "realignment (first roll)")), "")
            who = ts.Player.US if side == "US" else ts.Player.USSR
            before, after = _influence(state_from_token(c["pos"]), who), _influence(state_from_token(nxt["pos"]), who)
            spread = {countries[int(k)]: int(after[k] - before[k]) for k in np.flatnonzero(after != before)}
            return {"card": card, "mode": mode, "ops": ops, "coup": coup, "target": tgt, "spread": spread}

        us_c, ussr2 = first("US", 1, "card to play"), first("USSR", 2, "card to play")
        out.append({"seed": seed, "opening": opening,
                    "us_headline": option_label(uh["names"][uh["taken"]]),
                    "ussr_headline": option_label(sh["names"][sh["taken"]]),
                    "us_hand": _hand(st0, ts.Player.US, cards), "ussr_hand": _hand(st0, ts.Player.USSR, cards),
                    "ussr_ar1": ar1("USSR", us_c), "us_ar1": ar1("US", ussr2)})
    return out


# --- census ---------------------------------------------------------------------------------

def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.1f}%" if n else "—"


def _se_pct(k: int, n: int) -> float:
    p = k / n if n else 0.0
    return 100 * math.sqrt(p * (1 - p) / n) if n else math.nan


def headline_table(games: Sequence[Game], side: str, top: int) -> List[str]:
    hl = "us_headline" if side == "US" else "ussr_headline"
    hand = "us_hand" if side == "US" else "ussr_hand"
    held, chosen = collections.Counter(), collections.Counter()
    for g in games:
        for c in g[hand]:
            held[c] += 1
        chosen[g[hl]] += 1
    n = len(games)
    lines = [f"| card | headlined (of {n:,} games) | held | headlined when held |", "|:---|---:|---:|---:|"]
    for c, k in chosen.most_common(top):
        lines.append(f"| {c} | {_pct(k, n)} | {_pct(held[c], n)} | {_pct(k, held[c])} |")
    return lines


def _action(a: AR1) -> str:
    then = f", then {a['ops']}" if a["ops"] and a["ops"] != a["mode"] else ""
    return f"{a['card']} → {a['mode']}{then}" + (f" {a['target']}" if a["target"] else "")


def _is_iran_coup(a: Optional[AR1]) -> bool:
    return a is not None and a["coup"] == "Iran"


def lift_table(games: Sequence[Game], outcome: Callable[[Game], bool],
               conditions: Sequence[Tuple[str, Callable[[Game], bool]]], top: int, min_n: int = 100) -> List[str]:
    """The conditions that move P(outcome) most, by |z| of the difference with the rest."""
    base_k = sum(outcome(g) for g in games)
    n = len(games)
    rows = []
    for label, cond in conditions:
        inside = [g for g in games if cond(g)]
        if len(inside) < min_n or n - len(inside) < min_n:
            continue
        k = sum(outcome(g) for g in inside)
        p1, p0 = k / len(inside), (base_k - k) / (n - len(inside))
        se = math.sqrt(p1 * (1 - p1) / len(inside) + p0 * (1 - p0) / (n - len(inside))) or 1e-9
        rows.append((abs(p1 - p0) / se, label, len(inside), p1, p0, se))
    rows.sort(reverse=True)
    lines = ["| condition | games | rate with | rate without | z |", "|:---|---:|---:|---:|---:|"]
    for z, label, m, p1, p0, se in rows[:top]:
        lines.append(f"| {label} | {m:,} | {100 * p1:.1f}% | {100 * p0:.1f}% | {(p1 - p0) / se:+.1f} |")
    return lines


def _hand_of(g: Game, side: str) -> List[str]:
    return g["us_hand"] if side == "US" else g["ussr_hand"]


def _headline_of(g: Game, side: str) -> str:
    return g["us_headline"] if side == "US" else g["ussr_headline"]


def _holds(side: str, card: str) -> Callable[[Game], bool]:
    return lambda g: card in _hand_of(g, side)


def _headlines(side: str, card: str) -> Callable[[Game], bool]:
    return lambda g: _headline_of(g, side) == card


def _ussr_plays(card: str) -> Callable[[Game], bool]:
    return lambda g: g["ussr_ar1"] is not None and g["ussr_ar1"]["card"] == card


def _ussr_action(action: str) -> Callable[[Game], bool]:
    return lambda g: g["ussr_ar1"] is not None and _action(g["ussr_ar1"]) == action


def _us_places_in(country: str) -> Callable[[Game], bool]:
    return lambda g: g["us_ar1"] is not None and g["us_ar1"]["spread"].get(country, 0) > 0


def _conditions(games: Sequence[Game], who: str) -> List[Tuple[str, Callable[[Game], bool]]]:
    """Card-in-hand of `who`, and each side's headline, as conditions."""
    out = [(f"{who} holds {c}", _holds(who, c)) for c in sorted({c for g in games for c in _hand_of(g, who)})]
    for side in ("US", "USSR"):
        out += [(f"{side} headlines {c}", _headlines(side, c)) for c in sorted({_headline_of(g, side) for g in games})]
    return out


def spread_label(s: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(s.items(), key=lambda kv: (-kv[1], kv[0])) if v > 0) or "(none)"


def census(games: Sequence[Game], top: int) -> List[str]:
    n = len(games)
    lines = [f"{n:,} games.", "", "## Headlines", "", "### USSR", ""] + headline_table(games, "USSR", top)
    lines += ["", "### US", ""] + headline_table(games, "US", top)

    ussr = [g for g in games if g["ussr_ar1"] is not None]
    modes = collections.Counter((a["mode"], a["ops"] if a["ops"] != a["mode"] else "", a["coup"])
                                for a in (g["ussr_ar1"] for g in ussr) if a is not None)
    iran = sum(_is_iran_coup(g["ussr_ar1"]) for g in ussr)
    lines += ["", "## The USSR's AR1", "",
              f"Coup in Iran: **{_pct(iran, len(ussr))}** ± {_se_pct(iran, len(ussr)):.1f} of {len(ussr):,} games.", "",
              "### By how the card is played", "",
              "*Then* is what the Operations did after an event played first (an opponent's card).", "",
              "| played as | then | coup target | share |", "|:---|:---|:---|---:|"]
    for (m, o, t), k in modes.most_common(top):
        lines.append(f"| {m} | {o} | {t} | {_pct(k, len(ussr))} |")
    not_iran = [g for g in ussr if not _is_iran_coup(g["ussr_ar1"])]
    lines += ["", f"### When it is not the Iran coup ({len(not_iran):,} games): what instead", "",
              "| action (card → mode, coup target or first point) | share of these |", "|:---|---:|"]
    for a, k in collections.Counter(_action(g["ussr_ar1"]) for g in not_iran if g["ussr_ar1"]).most_common(top):
        lines.append(f"| {a} | {_pct(k, len(not_iran))} |")
    infl = [g for g in not_iran if g["ussr_ar1"] and g["ussr_ar1"]["ops"] == "INFLUENCE"]
    if infl:
        lines += ["", "Its influence spreads, when it places:", "", "| spread | share |", "|:---|---:|"]
        for s, k in collections.Counter(spread_label(g["ussr_ar1"]["spread"]) for g in infl if g["ussr_ar1"]).most_common(top):
            lines.append(f"| {s} | {_pct(k, len(infl))} |")
    lines += ["", "### What goes with not couping Iran", "",
              "The rate of *not* couping Iran with and without each condition (the USSR's hand, either "
              "headline), largest |z| first. A headline is known when the USSR acts; the US hand is not.", ""]
    lines += lift_table(ussr, lambda g: not _is_iran_coup(g["ussr_ar1"]), _conditions(ussr, "USSR"), top)
    ussr_cards = sorted({a["card"] for a in (g["ussr_ar1"] for g in ussr) if a is not None})
    lines += ["", "By the card the USSR plays:", ""]
    lines += lift_table(ussr, lambda g: not _is_iran_coup(g["ussr_ar1"]),
                        [(f"USSR plays {c}", _ussr_plays(c)) for c in ussr_cards], top)

    us = [g for g in games if g["us_ar1"] is not None]
    lines += ["", "## The US's AR1", "", "### By how the card is played", "",
              "| played as | then | coup target | share |", "|:---|:---|:---|---:|"]
    for (m, o, t), k in collections.Counter((a["mode"], a["ops"] if a["ops"] != a["mode"] else "", a["coup"])
                                            for a in (g["us_ar1"] for g in us) if a is not None).most_common(top):
        lines.append(f"| {m} | {o} | {t} | {_pct(k, len(us))} |")
    placed = [g for g in us if g["us_ar1"] and g["us_ar1"]["ops"] == "INFLUENCE"]
    lines += ["", f"### Influence: whole spreads ({len(placed):,} games)", "", "| spread | share |", "|:---|---:|"]
    for s, k in collections.Counter(spread_label(g["us_ar1"]["spread"]) for g in placed if g["us_ar1"]).most_common(top):
        lines.append(f"| {s} | {_pct(k, len(placed))} |")
    per_country = collections.Counter(c for g in placed if g["us_ar1"] for c, v in g["us_ar1"]["spread"].items() if v > 0)
    lines += ["", "### Influence: how often each country gets a point", "", "| country | games with ≥1 point |", "|:---|---:|"]
    for c, k in per_country.most_common(2 * top):
        lines.append(f"| {c} | {_pct(k, len(placed))} |")
    for c, _ in per_country.most_common(4):
        lines += ["", f"### What moves a point into {c}", ""]
        common = collections.Counter(_action(a) for a in (g["ussr_ar1"] for g in placed) if a is not None).most_common(12)
        conds = _conditions(placed, "US") + [(f"USSR AR1 {a}", _ussr_action(a)) for a, _ in common]
        lines += lift_table(placed, _us_places_in(c), conds, 8)
    return lines


# --- priced ---------------------------------------------------------------------------------

def priced_options(rows: Sequence[Row], pick: Callable[[Row], bool], min_worlds: int = 8) -> List[Tuple[str, int, float, float]]:
    """(option, decisions, mean value(option) - value(taken) in points of the decider's score, se)."""
    from tools.turn1_pricing import _values
    acc: Dict[str, List[float]] = collections.defaultdict(list)
    for r in rows:
        if not pick(r):
            continue
        v = _values(r)
        if v.shape[1] < min_worlds:
            continue
        t = v[r["taken"]].mean()
        for i, name in enumerate(r["names"]):
            acc[option_label(name)].append(50.0 * float(v[i].mean() - t))     # [-1, 1] -> points of a game
    out = []
    for k, xs in acc.items():
        a = np.array(xs)
        out.append((k, len(a), float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else math.nan))
    return sorted(out, key=lambda x: -x[2])


def priced_section(rows: Sequence[Row], top: int) -> List[str]:
    lines = ["## Priced: each option against the network's pick", "",
             "Critic value at the end of turn 1, option minus the network's choice, in points of the "
             "decider's score, averaged over the decisions where the option was legal (0 = as good as "
             "the pick; the pick itself is 0 by construction).", ""]
    groups: List[Tuple[str, Callable[[Row], bool]]] = [
        ("USSR headline", lambda r: r["side"] == "USSR" and r["kind"] == "headline"),
        ("US headline", lambda r: r["side"] == "US" and r["kind"] == "headline"),
        ("USSR AR1 coup target", lambda r: r["side"] == "USSR" and r["ar"] == 1 and r["kind"] == "coup target"),
        ("USSR AR1 card", lambda r: r["side"] == "USSR" and r["ar"] == 1 and r["kind"] == "card to play"),
        ("US AR1 first influence point", lambda r: r["side"] == "US" and r["ar"] == 1 and r["kind"] == "influence (first point)"),
    ]
    for label, pick in groups:
        opts = priced_options(rows, pick)
        if not opts:
            continue
        lines += [f"### {label}", "", "| option | decisions | vs the pick |", "|:---|---:|---:|"]
        for name, k, m, se in opts[:top]:
            lines.append(f"| {name} | {k} | {m:+.1f} ± {se:.1f} |")
        lines.append("")
    return lines


def read_rows(paths: Iterable[str]) -> List[Row]:
    rows: List[Row] = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", nargs="+", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json", default=None, help="the per-game records")
    args = ap.parse_args(argv)
    rows = read_rows(args.rows)
    games = games_from_rows(rows)
    lines = [f"# Turn 1 census{': ' + args.label if args.label else ''}", ""] + census(games, args.top)
    lines += [""] + priced_section(rows, args.top)
    with open(args.out, "w") as f:
        f.write("\n".join(lines) + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(games, f)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
