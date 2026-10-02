"""Does the model place for scoring? Battlegrounds whose region's scoring card is still live.

The fork owner's question (a strong player's, 2026-10-02): an influence play that could take a
battleground in a region whose scoring card has **not** been discarded -- so the region can still
score this reshuffle -- and the model places elsewhere. Why, and is it right by its own playouts?

The model sees every card's location (the card block of the observation), so it can know a
scoring card is in the discard; whether that changes where it places is the first measure.

**Live.** A region is live when one of its scoring cards is in the draw deck or a hand (the
mover's own, or the opponent's); dead when they are all discarded or removed; and not yet in the
game when the card has not entered (Southeast Asia Scoring before the Mid War). Thailand is live
through Asia Scoring or Southeast Asia Scoring.

**Takeable.** A battleground the mover does not control and can take with this play: placing
points there first reaches control before the play's Ops run out (`ops_block.force_take`).

Two outputs, both from influence plays in the model's greedy self-play:

* **sensitivity:** at every influence play with a takeable battleground, how often the model
  takes one -- split by whether that battleground's region is live (with its scoring card in the
  mover's hand, or elsewhere), dead, or not yet in the game. The same rate across those groups
  would mean the card's location does not reach the decision;
* **the counterfactual:** where a live battleground was takeable and the model put no point into
  any live battleground, each takeable live battleground is a branch -- take it first, then the
  model places the rest -- against the model's own play, by paired playouts (the unseen cards
  redealt per pair, the same dice in every branch, no suicide in the rest of the round). What the
  model did instead is recorded by where its points went.

As with every playout verdict here, the model plays every continuation.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _decider, _pair_start, apply_prefix
from ai.eval.doctrine_census import _map
from ai.eval.ops_block import (NODE_OFFSET, access, controlled, country_table, force_take, influence,
                               is_block_start, play_block)
from ai.eval.playout_audit import play_safe, split_half_regret
from ai.eval.reply_probe import _ar_key, position_link
from bindings.action_encoder import ActionEncoder

#: Scoring card id -> the region (or subregion) it scores.
SCORING = {1: "Asia", 2: "Europe", 3: "Middle East", 37: "Central America", 38: "Southeast Asia",
           79: "Africa", 81: "South America"}
GROUPS = ("live, card in mover's hand", "live, card elsewhere", "dead (discarded)", "not yet in the game")
_HANDS = {ts.Player.US: (ts.CardLocation.HAND_US_KNOWN, ts.CardLocation.HAND_US_UNKNOWN),
          ts.Player.USSR: (ts.CardLocation.HAND_USSR_KNOWN, ts.CardLocation.HAND_USSR_UNKNOWN)}
_LIVE = {ts.CardLocation.DRAW_DECK, *_HANDS[ts.Player.US], *_HANDS[ts.Player.USSR]}


def scoring_cards_for() -> List[Tuple[int, ...]]:
    """Per country, the scoring cards that score it."""
    out = []
    for row in _map():
        cs = [c for c, reg in SCORING.items() if reg == row["region"] or reg in row.get("subregions", [])]
        out.append(tuple(cs))
    return out


def status(st: ts.GameState, mover: ts.Player, country: int) -> str:
    """The country's region for scoring: one of GROUPS."""
    locs = [st.get_card_location(c) for c in scoring_cards_for()[country]]
    if any(loc in _HANDS[mover] for loc in locs):
        return GROUPS[0]
    if any(loc in _LIVE for loc in locs):
        return GROUPS[1]
    if any(loc != ts.CardLocation.UNAVAILABLE for loc in locs):
        return GROUPS[2]
    return GROUPS[3]


def greedy_chooser(act: PolicyFn) -> Callable[[ts.GameState], int]:
    def choose(st: ts.GameState) -> int:
        obs = np.asarray(ts.extract_observation(st, _decider(st)), dtype=np.float32)[None]
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)[None]
        return int(act(obs, mask)[0])
    return choose


def takeable(start: ts.GameState, choose: Callable[[ts.GameState], int]) -> Dict[int, Tuple[int, ...]]:
    """{battleground: the force_take action sequence} for every battleground this play can take."""
    _, bg, _ = country_table()
    side = 0 if _decider(start) == ts.Player.US else 1
    inf = influence(start)
    cand = np.flatnonzero(bg & ~controlled(inf, side) & access(inf, side))
    out = {}
    for c in cand:
        alloc = force_take(start, int(c), choose)
        if alloc is not None:
            out[int(c)] = tuple(alloc.actions)
    return out


def examine(start: ts.GameState, choose: Callable[[ts.GameState], int]) -> Optional[Dict[str, Any]]:
    """One influence play: what was takeable, by group, and what the model's own play did."""
    _, bg, names = country_table()
    mover = _decider(start)
    side = 0 if mover == ts.Player.US else 1
    takes = takeable(start, choose)
    if not takes:
        return None
    pol = play_block(start, choose)
    inf0 = influence(start)
    placed = np.clip(pol.inf_end[side] - inf0[side], 0, None)
    took = controlled(pol.inf_end, side) & ~controlled(inf0, side)
    groups = {c: status(start, mover, c) for c in takes}
    live = [c for c, g in groups.items() if g in GROUPS[:2]]
    ctl0 = controlled(inf0, side)
    elsewhere = {
        "own-controlled (defence)": int(placed[ctl0].sum()),
        "other battlegrounds": int(placed[bg & ~ctl0 & ~np.isin(np.arange(len(bg)), live)].sum()),
        "non-battlegrounds": int(placed[~bg & ~ctl0].sum()),
    }
    return {"start": start, "mover": mover, "policy_actions": pol.actions, "takes": takes,
            "groups": {names[c]: g for c, g in groups.items()},
            "took": {names[c]: bool(took[c]) for c in takes},
            "live": live, "live_points": int(placed[live].sum()) if live else 0,
            "ops": int(start.ctx().pending_ops_value), "elsewhere": elsewhere}


def collect(act: PolicyFn, n_games: int, seed: int, envs: int = 32, max_spots: int = 10_000,
            max_steps: int = 3_000_000) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every influence play with a takeable battleground in `n_games` greedy self-play games.
    Returns (sensitivity rows, counterfactual spots): a spot is a play where a live battleground
    was takeable and the model put no point into any. Spots are drawn uniformly from finished
    games, at most `max_spots`, so a cap does not tilt them toward the early turns."""
    choose = greedy_chooser(act)
    _, _, names = country_table()
    runner = ts.VectorizedBatchRunner(envs, seed * 17 + 3)
    runner.refresh_all()
    sens: List[Dict[str, Any]] = []
    pending: Dict[int, List[Tuple[Dict[str, Any], List[Dict[str, Any]]]]] = {}
    game = list(range(envs))
    done: List[int] = []
    started, finished = envs, 0
    for _ in range(max_steps):
        if finished >= n_games:
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        for i in range(envs):
            if not masks[i][NODE_OFFSET:NODE_OFFSET + 84].any():
                continue
            st = runner.get_state(i)
            if not is_block_start(st):
                continue
            ex = examine(st.clone(), choose)
            if ex is None:
                continue
            row: Dict[str, Any] = {"side": "US" if ex["mover"] == ts.Player.US else "USSR", "turn": int(st.turn),
                   "ops": ex["ops"], "by_group": {}}
            for nm, g in ex["groups"].items():
                d = row["by_group"].setdefault(g, [0, 0])
                d[0] += 1
                d[1] += int(ex["took"][nm])
            pending.setdefault(game[i], []).append((row, [ex] if ex["live"] and ex["live_points"] == 0 else []))
        runner.step_flat_all([int(x) for x in act(obs, masks)], auto_advance=True)
        ends = np.flatnonzero(np.array(runner.get_terminals())).tolist()
        for i in ends:
            finished += 1
            done.append(game[i])
            runner.reset_game(int(i), seed * 1_000_003 + started)
            game[i] = started
            started += 1
        if ends:
            runner.refresh_all()
    spots: List[Dict[str, Any]] = []
    for g in done:
        for row, ex in pending.get(g, []):
            sens.append(row)
            spots += ex
    if len(spots) > max_spots:
        pick = np.random.default_rng(seed).choice(len(spots), max_spots, replace=False)
        spots = [spots[j] for j in sorted(pick)]
    return sens, spots


def play(act: PolicyFn, spots: Sequence[Dict[str, Any]], pairs: int, seed: int, max_live: int = 3,
         chunk: int = 1536) -> List[Dict[str, Any]]:
    """Per spot: the model's play against taking each live battleground first (up to `max_live`)."""
    _, _, names = country_table()
    plan = []
    for ex in spots:
        live = sorted(ex["live"], key=lambda c: len(ex["takes"][c]))[:max_live]
        branches = {"policy": ()} | {f"take {names[c]}": ex["takes"][c] for c in live}
        plan.append((ex, branches))
    rows: List[Dict[str, Any]] = []
    lo = 0
    while lo < len(plan):
        group, size = [], 0
        while lo < len(plan) and (not group or size + len(plan[lo][1]) * pairs <= chunk):
            group.append(plan[lo])
            size += len(plan[lo][1]) * pairs
            lo += 1
        starts, movers, keys = [], [], []
        for gi, (ex, branches) in enumerate(group):
            st = ex["start"]
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo * 7 + gi, "resample")
                for pre in branches.values():
                    starts.append(apply_prefix(base, list(pre)) if pre else base.clone())
                    movers.append(ex["mover"])
                    keys.append(_ar_key(st))
        flat = play_safe(starts, movers, keys, act, seed + lo)
        at = 0
        for ex, branches in group:
            nb = len(branches)
            sc = np.array(flat[at:at + nb * pairs]).reshape(pairs, nb)
            at += nb * pairs
            st = ex["start"]
            names_b = list(branches)
            mine = ex["mover"]
            row = {"side": "US" if mine == ts.Player.US else "USSR", "turn": int(st.turn), "ar": int(st.action_round),
                   "defcon": int(st.defcon), "ops": ex["ops"],
                   "vp": int(st.victory_points) * (1 if mine == ts.Player.US else -1),
                   "groups": ex["groups"], "elsewhere": ex["elsewhere"], "pairs": pairs,
                   "policy_moves": [ActionEncoder.get_action_name(st, int(a)) for a in ex["policy_actions"][:1]],
                   "scores": {nm: float(sc[:, j].mean()) for j, nm in enumerate(names_b)},
                   "diff_se": {nm: float((sc[:, j] - sc[:, 0]).std(ddof=1) / math.sqrt(pairs))
                               for j, nm in enumerate(names_b) if j},
                   "regret": split_half_regret(sc, 0), "link": position_link(st)}
            rows.append(row)
    return rows


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def report(sens: Sequence[Dict[str, Any]], rows: Sequence[Dict[str, Any]], meta: Dict[str, Any],
           top: int = 20) -> Tuple[str, Dict[str, Any]]:
    out = [f"# Live scoring battlegrounds — {meta.get('model', '?')}", "",
           f"{len(sens)} influence plays with a takeable battleground, from greedy self-play; "
           f"{len(rows)} where a live battleground was takeable and the model put nothing into one.", ""]
    summary: Dict[str, Any] = {"meta": meta, "plays": len(sens), "spots": len(rows)}
    out += ["## Does a scoring card's location change where it places?", "",
            "Per takeable battleground: how often the model's play takes it, by whether its region can "
            "still score this reshuffle. Equal rates would mean the location does not reach the choice.", "",
            "| region's scoring card | takeable battlegrounds | taken | US | USSR |", "|:---|---:|---:|---:|---:|"]
    tab: Dict[str, Any] = {}
    for g in GROUPS:
        n = sum(r["by_group"].get(g, [0, 0])[0] for r in sens)
        k = sum(r["by_group"].get(g, [0, 0])[1] for r in sens)
        per = []
        for side in ("US", "USSR"):
            ns = sum(r["by_group"].get(g, [0, 0])[0] for r in sens if r["side"] == side)
            ks = sum(r["by_group"].get(g, [0, 0])[1] for r in sens if r["side"] == side)
            per.append(f"{100 * ks / ns:.0f}%" if ns else "—")
        out.append(f"| {g} | {n} | {100 * k / n:.0f}% | {per[0]} | {per[1]} |" if n else f"| {g} | 0 | — | — | — |")
        tab[g] = (n, k)
    summary["sensitivity"] = tab
    if len(rows) >= 2:
        best = [max(v for k, v in r["scores"].items() if k != "policy") - r["scores"]["policy"] for r in rows]
        firsts = []
        for r in rows:
            alts = [k for k in r["scores"] if k != "policy"]
            firsts.append(r["scores"][alts[0]] - r["scores"]["policy"])
        m1, s1 = _mean_se(firsts)
        mr, sr = _mean_se([r["regret"] for r in rows])
        out += ["", "## Is placing elsewhere right? (the model's own playouts)", "",
                "Take the live battleground first (the cheapest to take when there are several), the model "
                "placing the rest, against the model's own play. Split-half: the best of the live battlegrounds "
                "picked on half the pairs and scored on the other half.", "",
                "| | spots | take − model's play |", "|:---|---:|---:|",
                f"| take the cheapest live battleground | {len(rows)} | {100 * m1:+.1f} ± {100 * s1:.1f} |",
                f"| take the best live battleground (split-half) | {len(rows)} | {100 * mr:+.1f} ± {100 * sr:.1f} |", ""]
        summary["take_first"] = (m1, s1)
        summary["take_best_split_half"] = (mr, sr)
        out += ["### By group, side and era", "", "| | spots | take cheapest − model's |", "|:---|---:|---:|"]
        def first_alt(r: Dict[str, Any]) -> float:
            alts = [k for k in r["scores"] if k != "policy"]
            return r["scores"][alts[0]] - r["scores"]["policy"]
        def group_of(r: Dict[str, Any]) -> str:
            alts = [k for k in r["scores"] if k != "policy"]
            return r["groups"].get(alts[0][len("take "):], "?")
        splits = [(f"card in mover's hand", lambda r: group_of(r) == GROUPS[0]),
                  ("card elsewhere", lambda r: group_of(r) == GROUPS[1]),
                  ("US", lambda r: r["side"] == "US"), ("USSR", lambda r: r["side"] == "USSR"),
                  ("Early War", lambda r: r["turn"] <= 3), ("Mid War", lambda r: 4 <= r["turn"] <= 7),
                  ("Late War", lambda r: r["turn"] >= 8)]
        for label, f in splits:
            sel = [first_alt(r) for r in rows if f(r)]
            if len(sel) >= 2:
                m, s = _mean_se(sel)
                out.append(f"| {label} | {len(sel)} | {100 * m:+.1f} ± {100 * s:.1f} |")
        tot = {k: sum(r["elsewhere"][k] for r in rows) for k in rows[0]["elsewhere"]}
        allp = sum(tot.values()) or 1
        out += ["", "### Where its points went instead", "",
                ", ".join(f"{k}: {100 * v / allp:.0f}%" for k, v in tot.items()) + ".", ""]
        summary["elsewhere"] = tot
        worst = sorted(rows, key=lambda r: -r["regret"])[:top]
        out += [f"## The {top} largest regrets", "",
                "| side | turn | DEFCON | Ops | model's first point | best live take | regret | |",
                "|:---|---:|---:|---:|:---|:---|---:|:---|"]
        for r in worst:
            b = max((k for k in r["scores"] if k != "policy"), key=lambda k: r["scores"][k])
            out.append(f"| {r['side']} | {r['turn']}.{r['ar']} | {r['defcon']} | {r['ops']} | "
                       f"{'; '.join(r['policy_moves'])} | {b} ({r['groups'].get(b[5:], '?')}) | "
                       f"{100 * r['regret']:+.0f} | [position]({r['link']}) |")
    return "\n".join(out) + "\n", summary
