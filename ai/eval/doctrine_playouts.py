"""What breaking a doctrine rule costs the model: paired playouts at the census's own positions.

The doctrine census (ai/eval/doctrine_census.py) counts how often the model follows a strong
player's card rules; it cannot say whether breaking one costs anything. This probe takes the plays
where an "event this card" rule applied and the model chose another mode, and plays both moves to
the end of the game from paired copies (ai/eval/target_forms.paired_advantage: pair j redeals the
cards the mover cannot see and fixes the dice identically in both branches), the model playing
both sides greedily.

* **advantage** -- the event's result minus the model's choice, from the mover's side (+0.01 = one
  point of win probability), averaged over a rule's positions; its SE is across positions, so it
  carries both the pair noise and the spread between positions;
* **per game** -- breaks per game x advantage: what the habit costs over a game, the census's
  frequency times this probe's price.

What it cannot say: the judge is the model's own continuation, so an event whose worth lies in a
follow-up the model does not find reads as worse than it is; and the positions are the model's own
self-play, not a human's.
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np

import ts_engine as ts
from ai.eval import doctrine_census as D
from ai.eval.target_forms import Position, paired_advantage

DEFAULT_RULES = (
    "US always events Grain Sales to Soviets",
    "USSR always events Aldrich Ames Remix",
    "USSR events Che well over half the time (target: >50%)",
    "Terrorism is evented when behind or after Iranian Hostage Crisis",
)
EVENT = D.MODE_BASE + D.MODES.index("event")


def _rule_table(names: Sequence[str]) -> Dict[str, Any]:
    table = {name: (applies, complies) for name, applies, complies in D.rules()}
    out = {}
    for n in names:
        if n not in table:
            raise ValueError(f"no census rule {n!r}; rules: {sorted(table)}")
        applies, complies = table[n]
        out[n] = (applies, complies)
    return out


def collect_breaks(act: D.PolicyFn, games: int, seed: int, names: Sequence[str], envs: int = 64,
                   max_per_rule: int = 0, obs_features: int = 0) -> Dict[str, Any]:
    """The plays of `games` greedy self-play games where a rule applied, the model broke it, and
    eventing complies -- with their positions. At most `max_per_rule` per rule (0 = all), the
    earliest in play order."""
    table = _rule_table(names)

    def broken(r: Dict[str, Any]) -> List[str]:
        return [n for n, (ap, co) in table.items()
                if ap(r) and not co(r) and co({**r, "mode": "event"})]

    data = D.collect(act, games, seed, envs=envs, obs_features=obs_features,
                     keep_state=lambda r: bool(broken(r)))
    per_rule: Dict[str, int] = defaultdict(int)
    applied: Dict[str, int] = defaultdict(int)
    items: List[Dict[str, Any]] = []
    for r in data["plays"]:
        for n, (ap, _co) in table.items():
            if ap(r):
                applied[n] += 1
        st = r.pop("_state", None)
        if st is None:
            continue
        for n in broken(r):
            per_rule[n] += 1
            if max_per_rule and per_rule[n] > max_per_rule:
                continue
            items.append({"rule": n, "state": st, "play": r})
    return {"games": data["games"], "items": items, "broken": dict(per_rule), "applied": dict(applied)}


def measure(model, items: Sequence[Dict[str, Any]], pairs: int, seed: int, batch: int = 2048
            ) -> List[Dict[str, Any]]:
    """One row per item: the event against the model's own mode, in paired playouts."""
    positions = []
    trials = []
    for it in items:
        st: ts.GameState = it["state"]
        p = it["play"]
        mover = int(st.ctx().decision_player)
        positions.append(Position(state=st, mover=mover, segment="play_mode", turn=int(p["turn"]),
                                  legal=[], prior={}, logits={}, value_mover=0.0))
        trials.append((positions[-1], EVENT, D.MODE_BASE + D.MODES.index(p["mode"])))
    verdicts = paired_advantage(model, trials, pairs=pairs, seed=seed, batch=batch)
    out = []
    for it, (mean, se) in zip(items, verdicts):
        p = it["play"]
        out.append({"rule": it["rule"], "turn": p["turn"], "ar": p["ar"], "side": p["side"],
                    "chosen": p["mode"], "vp": p["vp"], "defcon": p["defcon"],
                    "adv": mean, "se": se})
    return out


def report(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any], names: Sequence[str]) -> str:
    games = int(meta["games"])
    lines = ["# Doctrine rules by paired playouts: what breaking each costs", "",
             f"{games} greedy self-play games of the census; at each play where the rule applied and "
             "the model chose another mode, the event against the model's choice, played to the end "
             f"by the model from {meta['pairs']} paired copies (same redeal, same dice). Advantage = "
             "the event's result minus the choice's, mover's side, in win probability (+0.010 = one "
             "point); SE across positions. Per game = breaks per game x advantage.", "",
             "| rule | applied / game | broken | measured | event's advantage | event better (> 2 SE) "
             "| choice better (< -2 SE) | **per game** |",
             "|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for n in names:
        rs = [r for r in rows if r["rule"] == n]
        applied = meta["applied"].get(n, 0)
        broken = meta["broken"].get(n, 0)
        if not rs:
            lines.append(f"| {n} | {applied / games:.3f} | {broken} | 0 | — | — | — | — |")
            continue
        a = np.array([r["adv"] for r in rs])
        se = float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else float("nan")
        conf = sum(1 for r in rs if r["se"] > 0 and r["adv"] > 2 * r["se"])
        ref = sum(1 for r in rs if r["se"] > 0 and r["adv"] < -2 * r["se"])
        lines.append(f"| {n} | {applied / games:.3f} | {broken} | {len(rs)} | {a.mean():+.3f} ± {se:.3f} "
                     f"| {conf} | {ref} | **{broken / games * a.mean():+.4f}** |")
    lines += ["", "## By turn: the event's advantage", "",
              "| rule | turns 1-3 | turns 4-7 | turns 8-10 |", "|:---|---:|---:|---:|"]
    for n in names:
        cells = []
        for lo, hi in ((1, 3), (4, 7), (8, 10)):
            a = np.array([r["adv"] for r in rows if r["rule"] == n and lo <= r["turn"] <= hi])
            cells.append(f"{a.mean():+.3f} ± {a.std(ddof=1) / math.sqrt(len(a)):.3f} ({len(a)})"
                         if len(a) > 1 else "—")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")
    lines += ["", "## By the mode the model chose instead", "",
              "| rule | chosen | positions | event's advantage |", "|:---|:---|---:|---:|"]
    for n in names:
        for mode in D.MODES:
            a = np.array([r["adv"] for r in rows if r["rule"] == n and r["chosen"] == mode])
            if len(a) > 1:
                lines.append(f"| {n} | {mode} | {len(a)} | {a.mean():+.3f} ± "
                             f"{a.std(ddof=1) / math.sqrt(len(a)):.3f} |")
    lines += ["", "The judge is the model's own greedy continuation: an event whose value depends on "
              "a follow-up the model does not find reads as worse than it is."]
    return "\n".join(lines) + "\n"


def dump(rows: Sequence[Dict[str, Any]], meta: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"meta": meta}) + "\n")
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def merge(paths: Sequence[str]) -> "tuple[List[Dict[str, Any]], Dict[str, Any]]":
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"games": 0, "applied": defaultdict(int), "broken": defaultdict(int)}
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            m = json.loads(fh.readline())["meta"]
            meta["games"] += m["games"]
            meta["pairs"] = m["pairs"]
            for k in ("applied", "broken"):
                for n, v in m[k].items():
                    meta[k][n] += v
            rows += [json.loads(line) for line in fh]
    return rows, meta

