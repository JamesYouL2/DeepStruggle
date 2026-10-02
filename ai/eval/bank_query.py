"""Ask the position bank (`position_bank.py`) a rule, with no new playouts.

A rule maps a bank record to the index of the candidate it prescribes, or None where it does not
apply or agrees with the model (candidate 0). Its effect is pooled over the records it applies to:
per record, the mean over pairs of (its candidate − candidate 0), then mean and standard error over
records. Each rule also names a dose (a feature to bin by) and the card it concerns, for the
generality checks: side, era, and the effect with the most frequent card left out.

`RULES` holds the rule families of `rule_oracle.py` written as queries, plus two hand-timing rules
over every card in `position_bank.HAND_TIMING_CARDS` -- the owner's: hold such a card for the last
action round with exactly one other card in hand; and do not open a round with it while rounds
remain -- reported card by card as well as pooled.
"""
from __future__ import annotations

import gzip
import json
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ai.eval.doctrine_census import cards
from ai.eval.position_bank import scores, timing_ids

MODE_BASE = 110
EVENT, SPACE, COUP = MODE_BASE, MODE_BASE + 1, MODE_BASE + 3
SCORING = {1, 2, 3, 37, 38, 79, 81}
CHINA = 6
Record = Dict[str, Any]


@dataclass
class Rule:
    name: str
    pick: Callable[[Record], Optional[int]]
    dose: Callable[[Record], Optional[float]] = lambda r: None
    bins: Tuple[Tuple[float, float, str], ...] = ()
    card: Callable[[Record], int] = lambda r: int(r["features"].get("card", 0))
    by_card: bool = False


def _first(r: Record) -> int:
    return int(r["candidates"][0]["prefix"][0])


def _find(r: Record, action: int) -> Optional[int]:
    for j, c in enumerate(r["candidates"]):
        if j and len(c["prefix"]) == 1 and int(c["prefix"][0]) == action:
            return j
    return None


def _likeliest(r: Record, ok: Callable[[int], bool]) -> Optional[int]:
    best, bp = None, -1.0
    for j, c in enumerate(r["candidates"]):
        a = int(c["prefix"][0])
        if j and len(c["prefix"]) == 1 and a < 110 and ok(a + 1) and c["prior"] > bp:
            best, bp = j, c["prior"]
    return best


def _hand_card(r: Record, cid: int) -> Dict[str, Any]:
    return r["features"].get("hand_cards", {}).get(str(cid), {})


def _card_choice(r: Record) -> bool:
    return r["kind"] == "card"


def _mode(r: Record) -> bool:
    return r["kind"] == "mode"


def _r_decisive(r: Record) -> Optional[int]:
    for j, c in enumerate(r["candidates"]):
        if c["name"].startswith("decisive:"):
            return j
    return None


def _r_event_vp(r: Record) -> Optional[int]:
    f = r["features"]
    if _mode(r) and f.get("owner") in ("own", "neutral") and "event_gain" in f and _first(r) != EVENT:
        return _find(r, EVENT)
    return None


def _r_space(r: Record) -> Optional[int]:
    f = r["features"]
    if _mode(r) and f.get("owner") == "opp" and f.get("event_cost") is not None and _first(r) != SPACE:
        return _find(r, SPACE)
    return None


def _r_delay(r: Record) -> Optional[int]:
    f = r["features"]
    a = _first(r)
    if not _card_choice(r) or a >= 110 or f["rounds_left"] < 1:
        return None
    if _hand_card(r, a + 1).get("owner") != "opp":
        return None
    return _likeliest(r, lambda c: _hand_card(r, c).get("owner") != "opp" and c != CHINA)


def _r_score_now(r: Record) -> Optional[int]:
    if not _card_choice(r):
        return None
    vals = [(c, _hand_card(r, c).get("score")) for c in r["features"].get("scoring_held", [])]
    good = [(v, c) for c, v in vals if v is not None and v >= 1 and _first(r) != c - 1]
    if not good:
        return None
    return _find(r, max(good)[1] - 1)


def _r_hold_score(r: Record) -> Optional[int]:
    a = _first(r)
    if not _card_choice(r) or a >= 110 or a + 1 not in SCORING or r["features"]["rounds_left"] < 2:
        return None
    v = _hand_card(r, a + 1).get("score")
    if v is None or v > -1:
        return None
    return _likeliest(r, lambda c: c not in SCORING and c != CHINA)


def _r_greedy(r: Record) -> Optional[int]:
    for j, c in enumerate(r["candidates"]):
        if c["name"].startswith("critic's favourite"):
            return j
    return None


def _r_milops(r: Record) -> Optional[int]:
    f = r["features"]
    if _mode(r) and f["rounds_left"] <= 1 and f["short"] >= 2 and _first(r) != COUP:
        return _find(r, COUP)
    return None


def _timing_held(r: Record) -> List[int]:
    return [c for c in r["features"].get("timing_held", []) if c != CHINA]


def _r_timing_last(r: Record) -> Optional[int]:
    """Last action round, the hand is one timing card plus exactly one other: play the timing card."""
    f = r["features"]
    if not _card_choice(r) or f["rounds_left"] != 0 or f["hand"] != 2:
        return None
    for c in _timing_held(r):
        if _first(r) != c - 1:
            j = _find(r, c - 1)
            if j is not None:
                return j
    return None


def _timing_card_last(r: Record) -> int:
    held = _timing_held(r)
    return held[0] if held else 0


def _r_timing_delay(r: Record) -> Optional[int]:
    """A round opens with a timing card while rounds remain: play the likeliest other card instead."""
    a = _first(r)
    if not _card_choice(r) or a >= 110 or r["features"]["rounds_left"] < 1 or a + 1 not in timing_ids():
        return None
    return _likeliest(r, lambda c: c not in timing_ids() and c != CHINA)


RULES: Tuple[Rule, ...] = (
    Rule("decisive", _r_decisive),
    Rule("event_vp", _r_event_vp, lambda r: r["features"].get("event_gain"),
         ((-99, 0, "≤ 0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+"))),
    Rule("space_harmful", _r_space, lambda r: r["features"].get("event_cost"),
         ((-99, 0, "≤ 0"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+"))),
    Rule("delay_harmful", _r_delay, lambda r: _hand_card(r, _first(r) + 1).get("cost"),
         ((-99, 0, "≤ 0"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+")), card=lambda r: _first(r) + 1),
    Rule("score_now", _r_score_now,
         lambda r: max([v for c in r["features"].get("scoring_held", []) for v in [_hand_card(r, c).get("score")]
                        if v is not None] or [0]),
         ((1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, 99, "5+")),
         card=lambda r: max(r["features"].get("scoring_held", [0]))),
    Rule("hold_score", _r_hold_score, lambda r: _hand_card(r, _first(r) + 1).get("score"),
         ((-1, -1, "−1"), (-2, -2, "−2"), (-4, -3, "−3 to −4"), (-99, -5, "≤ −5")), card=lambda r: _first(r) + 1),
    Rule("greedy_choice", _r_greedy),
    Rule("milops_coup", _r_milops, lambda r: r["features"]["short"], ((2, 2, "2"), (3, 3, "3"), (4, 99, "4+"))),
    Rule("timing_last_with_one", _r_timing_last, card=_timing_card_last, by_card=True),
    Rule("timing_delay", _r_timing_delay, card=lambda r: _first(r) + 1, by_card=True),
)


def load(paths: Iterable[str]) -> List[Record]:
    out: List[Record] = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            out += json.load(f)["records"]
    return out


def effect(r: Record, j: int) -> float:
    a, b = scores(r, j), scores(r, 0)
    n = min(len(a), len(b))
    return float((a[:n] - b[:n]).mean())


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def _fmt(xs: Sequence[float]) -> str:
    if len(xs) < 2:
        return f"— ({len(xs)})"
    m, se = _mean_se(xs)
    return f"{100 * m:+.1f} ± {100 * se:.1f} ({len(xs)})"


def ask(records: Sequence[Record], rule: Rule) -> List[Tuple[Record, int, float]]:
    """(record, prescribed candidate, effect) for every record the rule applies to."""
    out = []
    for r in records:
        j = rule.pick(r)
        if j is not None and j != 0:
            out.append((r, j, effect(r, j)))
    return out


def report(records: Sequence[Record], rules: Sequence[Rule] = RULES, meta: Optional[Dict[str, Any]] = None
           ) -> Tuple[str, Dict[str, Any]]:
    info = cards()
    name = lambda c: str(info[c]["name"]) if c in info else "—"  # noqa: E731
    era = lambda t: "Early" if t <= 3 else ("Mid" if t <= 7 else "Late")  # noqa: E731
    kinds: Dict[str, int] = {}
    for r in records:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    out = [f"# Bank query — {(meta or {}).get('model', '?')}", "",
           f"{len(records)} banked decisions ({', '.join(f'{k} {v}' for k, v in sorted(kinds.items(), key=lambda x: -x[1]))}), "
           f"{records[0]['pairs'] if records else '?'} paired playouts per candidate. rule − model in win-rate points, "
           "± one standard error over decisions.", "",
           "| rule | decisions | rule − model | US | USSR | Early | Mid | Late | without its top card |",
           "|:---|---:|---:|---:|---:|---:|---:|---:|:---|"]
    summary: Dict[str, Any] = {"meta": meta or {}, "n": len(records), "rules": {}}
    detail: List[str] = []
    for rule in rules:
        hits = ask(records, rule)
        if not hits:
            out.append(f"| {rule.name} | 0 | — | — | — | — | — | — | — |")
            continue
        d = [e for _, _, e in hits]
        counts: Dict[int, int] = {}
        for r, _, _ in hits:
            counts[rule.card(r)] = counts.get(rule.card(r), 0) + 1
        top = max(counts, key=lambda c: counts[c])
        rest = [e for r, _, e in hits if rule.card(r) != top]
        m, se = _mean_se(d)
        sub = lambda f: _fmt([e for r, _, e in hits if f(r)])  # noqa: E731
        out.append(f"| {rule.name} | {len(hits)} | **{100 * m:+.1f} ± {100 * se:.1f}** | "
                   f"{sub(lambda r: r['features']['side'] == 'US')} | {sub(lambda r: r['features']['side'] == 'USSR')} | "
                   f"{sub(lambda r: era(r['features']['turn']) == 'Early')} | {sub(lambda r: era(r['features']['turn']) == 'Mid')} | "
                   f"{sub(lambda r: era(r['features']['turn']) == 'Late')} | {_fmt(rest)} without {name(top)} |")
        summary["rules"][rule.name] = {"n": len(hits), "effect": (m, se), "top_card": name(top)}
        if rule.bins:
            detail += [f"## `{rule.name}` by dose", "", "| dose | decisions | rule − model |", "|:---|---:|---:|"]
            dosed = [(rule.dose(r), e) for r, _, e in hits]
            for lo, hi, lab in rule.bins:
                b = [e for x, e in dosed if x is not None and lo <= x <= hi]
                detail.append(f"| {lab} | {len(b)} | {_fmt(b)} |")
            detail.append("")
        if rule.by_card:
            detail += [f"## `{rule.name}` card by card", "", "| card | decisions | rule − model |", "|:---|---:|---:|"]
            for c in sorted(counts, key=lambda c: -counts[c]):
                detail.append(f"| {name(c)} | {counts[c]} | {_fmt([e for r, _, e in hits if rule.card(r) == c])} |")
            detail.append("")
    out.append("")
    return "\n".join(out + detail) + "\n", summary
