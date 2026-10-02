"""Card-specific event rules: the cards strong humans event far more often than the model, each
played out at the model's own play-mode decisions, and a condition fitted per card.

**Which cards.** `TARGETS` holds, per side, the 18 own or neutral cards with the most
human-versus-model disagreements (gap in event rate × decisions) over the 23,317 play-mode
decisions of the ts-replayer corpus, the model's greedy choice taken on the same positions
(human-disagreement scan, run 36958653207). The rates are kept beside each card.

**The bank.** `position_bank.collect_targets` keeps only play-mode decisions on those cards by
their side in the model's greedy self-play, each with every non-suicide mode played out
(`position_bank.build`), and counts the model's own mode at every play-mode decision it passes.

**The rule per card.** For each card, a record's effect is the event's pairs minus the model's own
mode's (zero where the model already events). "Always event" is their mean. A conditional rule,
"event when feature ≥ t (or ≤ t), else the model's move", is chosen on one half of the records
and scored on the other, both ways round, so the reported gain is out of sample; the rule printed
is the one chosen on all records.

**Headlines are special and are not covered here.** A headline has no Ops alternative: the choice
is *which* event to fire, before the opponent's, with both resolved by Ops value. So an action-round
event rate says nothing about headlines, and they are probed as card choices instead (the choice
oracle's headline scenarios, `human_disagree`'s "headline" kind). Against the corpus the model
agrees on 55% of 4,407 headlines and the humans' picks play out no better (−0.8 ± 0.5 all turns,
−1.7 ± 0.7 turn 1; run 36958653207, 36963933063); the largest gaps in pick rate are a point or two.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _decider
from ai.eval.ops_block import controlled, influence
from ai.eval.position_bank import scores
from ai.eval.rule_oracle import event_gain
from bindings.action_encoder import ActionEncoder

EVENT = 110
MODE_NAMES = {110: "event", 111: "space", 112: "influence", 113: "coup", 114: "realign"}
Record = Dict[str, Any]

#: side -> (card, human event rate, model event rate on the same positions, decisions)
TARGETS: Dict[str, Tuple[Tuple[str, float, float, int], ...]] = {
    "US": (("Special Relationship", 0.21, 0.01, 252), ("John Paul II Elected Pope", 0.57, 0.06, 94),
           ("SALT Negotiations", 0.44, 0.00, 105), ("Bear Trap", 0.48, 0.00, 84),
           ("Our Man in Tehran", 0.39, 0.00, 97), ("“One Small Step…”", 0.25, 0.00, 133),
           ("Five Year Plan", 0.14, 0.00, 218), ("Puppet Governments", 0.84, 0.54, 97),
           ("Missile Envy", 0.18, 0.01, 160), ("Nuclear Subs", 0.26, 0.00, 104),
           ("How I Learned to Stop Worrying", 0.21, 0.00, 130), ("Kitchen Debates", 0.20, 0.01, 119),
           ("Arms Race", 0.19, 0.00, 124), ("Grain Sales to Soviets", 0.89, 0.04, 27),
           ("Marshall Plan", 0.25, 0.00, 89), ("Terrorism", 0.54, 0.00, 41),
           ("Alliance for Progress", 0.33, 0.18, 130), ("Star Wars", 0.49, 0.00, 35)),
    "USSR": (("Che", 0.57, 0.08, 124), ("Arms Race", 0.41, 0.01, 134), ("SALT Negotiations", 0.44, 0.01, 104),
             ("OPEC", 0.65, 0.33, 138), ("Missile Envy", 0.23, 0.00, 197), ("Muslim Revolution", 0.38, 0.00, 94),
             ("Arab-Israeli War", 0.17, 0.03, 236), ("“One Small Step…”", 0.24, 0.00, 127),
             ("Blockade", 0.43, 0.22, 143), ("Terrorism", 0.80, 0.05, 40), ("Romanian Abdication", 0.19, 0.00, 148),
             ("ABM Treaty", 0.77, 0.36, 64), ("Vietnam Revolts", 0.35, 0.07, 94), ("Red Scare/Purge", 0.24, 0.04, 107),
             ("Wargames", 0.34, 0.00, 50), ("De-Stalinization", 0.97, 0.86, 132), ("Nasser", 0.56, 0.36, 70),
             ("Aldrich Ames Remix", 1.00, 0.00, 12)),
}

#: The features a condition may use: name -> reader over a record's features.
SPLITS = {
    "turn": lambda f: f["turn"], "action round": lambda f: f["ar"], "rounds left": lambda f: f["rounds_left"],
    "DEFCON": lambda f: f["defcon"], "VP lead": lambda f: f["lead"], "hand": lambda f: f["hand"],
    "surplus cards": lambda f: f["surplus"], "space lead": lambda f: f["space"] - f["space_opp"],
    "Military Ops short": lambda f: f["short"], "opponent short": lambda f: f["short_opp"],
    "event VP now": lambda f: f.get("event_gain"),
}


def target_ids(by_name: Dict[str, int]) -> Set[Tuple[str, int]]:
    return {(side, by_name[c]) for side, rows in TARGETS.items() for c, *_ in rows}


def _event_index(r: Record) -> Optional[int]:
    for j, c in enumerate(r["candidates"]):
        if len(c["prefix"]) == 1 and int(c["prefix"][0]) == EVENT:
            return j
    return None


def effects(r: Record) -> Optional[float]:
    """Event minus the model's mode, paired over this record's pairs; None if the event was not a candidate."""
    j = _event_index(r)
    if j is None:
        return None
    if j == 0:
        return 0.0
    a, b = scores(r, j), scores(r, 0)
    n = min(len(a), len(b))
    return float((a[:n] - b[:n]).mean())


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


Cond = Tuple[str, str, float]                      # (feature, ">=" or "<=", threshold)


def _applies(f: Dict[str, Any], cond: Cond) -> bool:
    x = SPLITS[cond[0]](f)
    if x is None:
        return False
    return x >= cond[2] if cond[1] == ">=" else x <= cond[2]


def _rule_effects(rows: Sequence[Tuple[Dict[str, Any], float]], cond: Cond) -> List[float]:
    return [e if _applies(f, cond) else 0.0 for f, e in rows]


def best_condition(rows: Sequence[Tuple[Dict[str, Any], float]], min_hits: int = 8) -> Optional[Tuple[Cond, float]]:
    """The condition with the largest mean rule effect over `rows` that applies at least `min_hits` times."""
    best: Optional[Tuple[Cond, float]] = None
    for name, read in SPLITS.items():
        vals = sorted({read(f) for f, _ in rows if read(f) is not None})
        if len(vals) < 2:
            continue
        if len(vals) > 12:
            vals = sorted(set(np.quantile(vals, np.linspace(0.05, 0.95, 12)).round(1).tolist()))
        for t in vals:
            for op in (">=", "<="):
                cond = (name, op, float(t))
                hits = sum(_applies(f, cond) for f, _ in rows)
                if hits < min_hits or hits == len(rows):
                    continue
                m = float(np.mean(_rule_effects(rows, cond)))
                if best is None or m > best[1]:
                    best = (cond, m)
    return best


def cross_validated(rows: Sequence[Tuple[Dict[str, Any], float]]) -> Tuple[List[float], List[Optional[Cond]]]:
    """Fit on one half, score on the other, both ways: the held-out per-record effects and the two conditions."""
    halves = [list(rows[0::2]), list(rows[1::2])]
    out: List[float] = []
    conds: List[Optional[Cond]] = []
    for k in (0, 1):
        fit = best_condition(halves[k])
        conds.append(fit[0] if fit else None)
        test = halves[1 - k]
        out += _rule_effects(test, fit[0]) if fit else [0.0] * len(test)
    return out, conds


def _fmt(m: float, se: float) -> str:
    return f"{100 * m:+.1f} ± {100 * se:.1f}" if not math.isnan(se) else (f"{100 * m:+.1f}" if not math.isnan(m) else "—")


def _cond_str(c: Optional[Cond]) -> str:
    if c is None:
        return "—"
    t = int(c[2]) if float(c[2]).is_integer() else c[2]
    return f"{c[0]} {'≥' if c[1] == '>=' else '≤'} {t}"


def report(records: Sequence[Record], names: Dict[int, str], counts: Dict[str, Dict[str, int]],
           meta: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """`counts`: "side|card id" -> the model's own mode counts over all of its self-play play-mode decisions."""
    out = [f"# Card event rules — {(meta or {}).get('model', '?')}", "",
           f"{len(records)} play-mode decisions on the cards strong humans event far more than the model, every "
           f"non-suicide mode played out ({records[0]['pairs'] if records else '?'} paired playouts each). Effects are "
           "the event minus the model's own mode, in win-rate points, ± one standard error over decisions. The "
           "conditional rule is fitted on half the decisions and scored on the other half, both ways round.", "",
           "| side | card | humans event | model events (corpus positions) | model events (self-play) | decisions | "
           "always event | best rule (fitted on all) | its held-out gain | rule applies |",
           "|:---|:---|---:|---:|---:|---:|---:|:---|---:|---:|"]
    summary: Dict[str, Any] = {"meta": meta or {}, "cards": {}}
    by_name = {v: k for k, v in names.items()}
    for side, rows in TARGETS.items():
        for card, human, model_h, _ in rows:
            cid = by_name.get(card)
            recs = [r for r in records if r["features"]["side"] == side and r["features"].get("card") == cid]
            pairs = [(r["features"], e) for r in recs for e in [effects(r)] if e is not None]
            c = counts.get(f"{side}|{cid}", {})
            tot = sum(c.values())
            sp = f"{100 * c.get('event', 0) / tot:.0f}% ({tot})" if tot else "—"
            if len(pairs) < 4:
                out.append(f"| {side} | {card} | {100 * human:.0f}% | {100 * model_h:.0f}% | {sp} | {len(pairs)} | — | — | — | — |")
                continue
            always = _mean_se([e for _, e in pairs])
            held, _ = cross_validated(pairs)
            fit = best_condition(pairs)
            cond = fit[0] if fit else None
            share = (sum(_applies(f, cond) for f, _ in pairs) / len(pairs)) if cond else 0.0
            hm = _mean_se(held)
            out.append(f"| {side} | {card} | {100 * human:.0f}% | {100 * model_h:.0f}% | {sp} | {len(pairs)} | "
                       f"{_fmt(*always)} | {_cond_str(cond)} | {_fmt(*hm)} | {100 * share:.0f}% |")
            summary["cards"][f"{side} {card}"] = {"n": len(pairs), "always": always, "rule": cond, "held_out": hm,
                                                  "share": share, "human": human, "model_corpus": model_h}
    out += ["", "## The model's own modes on these cards (self-play)", "",
            "| side | card | event | space | influence | coup | realign |", "|:---|:---|---:|---:|---:|---:|---:|"]
    for side, rows in TARGETS.items():
        for card, *_ in rows:
            c = counts.get(f"{side}|{by_name.get(card)}", {})
            tot = sum(c.values()) or 1
            out.append(f"| {side} | {card} | " + " | ".join(f"{100 * c.get(m, 0) / tot:.0f}%" for m in MODE_NAMES.values()) + " |")
    return "\n".join(out) + "\n", summary


@dataclass(frozen=True)
class Forced:
    """A forced condition: the (side, card) play-mode decisions kept only where `keep(state, act)`
    holds, reported in groups by `group(features)`."""
    side: str
    card: str
    what: str
    keep: Callable[[ts.GameState, PolicyFn], bool]
    group: Callable[[Dict[str, Any]], str]


def _scores_at_least(vp: float) -> Callable[[ts.GameState, PolicyFn], bool]:
    def keep(st: ts.GameState, act: PolicyFn) -> bool:
        legal = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
        return bool(legal[EVENT]) and event_gain(st, _decider(st), act) >= vp
    return keep


def _space_at(spots: Tuple[Tuple[int, int], ...]) -> Callable[[ts.GameState, PolicyFn], bool]:
    """The mover's and the opponent's space boxes are one of `spots`. One box behind at 1 vs 2 or
    3 vs 4, One Small Step jumps the mover two boxes, past the opponent into the next VP box."""
    def keep(st: ts.GameState, act: PolicyFn) -> bool:
        us = _decider(st) == ts.Player.US
        me, opp = (int(st.us_space_track), int(st.ussr_space_track))[:: 1 if us else -1]
        return (me, opp) in spots
    return keep


def _south_korea_us(st: ts.GameState, act: PolicyFn) -> bool:
    """Soviets Shoot Down KAL-007 with South Korea US-controlled (the event's influence or realignment
    then applies), at DEFCON 3 or more, where its DEFCON drop is not suicide."""
    legal = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
    sk = int(ts.MapData.get_country_by_name("South Korea"))
    return bool(legal[EVENT]) and int(st.defcon) >= 3 and bool(controlled(influence(st), 0)[sk])


def _reformer_in_play(st: ts.GameState, act: PolicyFn) -> bool:
    legal = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
    return bool(legal[EVENT]) and st.has_flag(int(ts.EffectBits.THE_REFORMER_PLAYED))


def _by_defcon(f: Dict[str, Any]) -> str:
    return f"DEFCON {f['defcon']}"


def _by_vp(f: Dict[str, Any]) -> str:
    return f"scores {int(f.get('event_gain') or 0)} VP"


def _by_mil_ops(f: Dict[str, Any]) -> str:
    return f"Military Ops short {min(int(f['short']), 2)}{'+' if f['short'] >= 2 else ''}, {'last round' if f['rounds_left'] == 0 else 'rounds left'}"


def _by_space(f: Dict[str, Any]) -> str:
    return f"space {f['space']} vs {f['space_opp']}"


#: Named sets of forced conditions (the owner's).
FORCED_SETS: Dict[str, Tuple[Forced, ...]] = {
    "vp": (Forced("US", "Special Relationship",
                  "NATO in effect and the UK US-controlled: 2 influence in Western Europe and 2 VP", _scores_at_least(2), _by_vp),
           Forced("USSR", "OPEC", "5 or more VP", _scores_at_least(5), _by_vp),
           Forced("US", "Alliance for Progress", "5 or more VP", _scores_at_least(5), _by_vp)),
    # VP or space now, each where the owner says the event is right.
    "nextvp": (Forced("US", "Captured Nazi Scientist", "into a VP box (the event scores VP)", _scores_at_least(1), _by_space),
               Forced("USSR", "Captured Nazi Scientist", "into a VP box (the event scores VP)", _scores_at_least(1), _by_space),
               Forced("US", "Soviets Shoot Down KAL-007", "South Korea US-controlled, DEFCON 3+", _south_korea_us, _by_defcon),
               Forced("USSR", "Glasnost", "The Reformer in play", _reformer_in_play, _by_defcon)),
    # A 1-Op card whose event scores 2 VP, against the 1-Op coup the model usually plays for Military Ops.
    "kd": (Forced("US", "Kitchen Debates", "the event scores (more battlegrounds than the USSR): 2 VP", _scores_at_least(2),
                  _by_mil_ops),),
    # Two sets, since 3 vs 4 is about a tenth as common as 1 vs 2 and would never fill its share of a joint cap.
    "oss12": (Forced("US", "“One Small Step…”", "space 1 vs 2", _space_at(((1, 2),)), _by_space),
              Forced("USSR", "“One Small Step…”", "space 1 vs 2", _space_at(((1, 2),)), _by_space)),
    "oss34": (Forced("US", "“One Small Step…”", "space 3 vs 4", _space_at(((3, 4),)), _by_space),
              Forced("USSR", "“One Small Step…”", "space 3 vs 4", _space_at(((3, 4),)), _by_space)),
}


def forced_keep(which: str, by_name: Dict[str, int], act: PolicyFn
                ) -> Dict[Tuple[str, int], Callable[[ts.GameState], bool]]:
    return {(f.side, by_name[f.card]): (lambda st, k=f.keep: k(st, act)) for f in FORCED_SETS[which]}


def forced_report(records: Sequence[Record], names: Dict[int, str], which: str = "vp",
                  meta: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
    """Each forced condition: every mode against the model's own, overall, in close games, and by group."""
    by_name = {v: k for k, v in names.items()}
    out = [f"# Forced event conditions ({which}) — {(meta or {}).get('model', '?')}", "",
           "Positions from the model's greedy self-play where the condition holds naturally, every non-suicide "
           f"mode played out ({records[0]['pairs'] if records else '?'} paired playouts each). Each mode minus the "
           "model's own choice, in win-rate points, ± one standard error over positions; \"close\" keeps "
           "positions the model's own move wins 25-75%.", ""]
    summary: Dict[str, Any] = {"meta": meta or {}, "forced": {}}
    for fc in FORCED_SETS[which]:
        cid = by_name.get(fc.card)
        recs = [r for r in records if r["features"]["side"] == fc.side and r["features"].get("card") == cid]
        out += [f"## {fc.side} {fc.card}: {fc.what}", ""]
        if not recs:
            out += ["No positions.", ""]
            continue
        model_event = sum(int(r["candidates"][0]["prefix"][0]) == EVENT for r in recs) / len(recs)
        ev = [(r, e, float(np.mean(scores(r, 0)))) for r in recs for e in [effects(r)] if e is not None]
        allm = _mean_se([e for _, e, _ in ev])
        close = _mean_se([e for _, e, b in ev if 0.25 <= b <= 0.75])
        n_close = sum(0.25 <= b <= 0.75 for _, _, b in ev)
        out += [f"{len(recs)} positions; the model events {100 * model_event:.0f}% of them.", "",
                "| | positions | model events | event − model |", "|:---|---:|---:|---:|",
                f"| all | {len(ev)} | {100 * model_event:.0f}% | **{_fmt(*allm)}** |",
                f"| close | {n_close} | | {_fmt(*close)} |"]
        for g in sorted({fc.group(r["features"]) for r, _, _ in ev}):
            sub = [(r, e) for r, e, _ in ev if fc.group(r["features"]) == g]
            me = sum(int(r["candidates"][0]["prefix"][0]) == EVENT for r, _ in sub) / len(sub)
            out.append(f"| {g} | {len(sub)} | {100 * me:.0f}% | {_fmt(*_mean_se([e for _, e in sub]))} |")
        out += ["", "| mode | positions | mode − model |", "|:---|---:|---:|"]
        for a, m in MODE_NAMES.items():
            d = []
            for r in recs:
                js = [j for j, c in enumerate(r["candidates"]) if j and c["prefix"] == [a]]
                if js:
                    x, y = scores(r, js[0]), scores(r, 0)
                    n = min(len(x), len(y))
                    d.append(float((x[:n] - y[:n]).mean()))
            if d:
                out.append(f"| {m} | {len(d)} | {_fmt(*_mean_se(d))} |")
        out.append("")
        summary["forced"][f"{fc.side} {fc.card}"] = {"n": len(recs), "model_event": model_event, "all": allm, "close": close}
    return "\n".join(out) + "\n", summary
