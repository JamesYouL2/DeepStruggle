"""Where does each side put its influence, region by region, and is that right?

The fork owner's question (2026-10-02): does the model over-invest in some regions (Asia, by the
VP ledger) and under-invest in others -- for every region and both sides, not just a few. Two
measures, from influence Ops plays (not an event's placements):

* **Allocation:** each side's share of influence points by region (Europe, Asia, Middle East,
  Africa, Central America, South America), the model's greedy self-play against the human
  ts-replayer corpus, split by era and by the region's scoring card: in the mover's hand, live
  elsewhere (deck or the opponent's hand), discarded, or not yet in the game.
* **Is it right:** at sampled influence plays, the model's own play against the same play kept
  inside region R -- the model still chooses where in R, so each branch is its best play there --
  one branch per region the play can reach at its first point, then the model plays on (the
  no-suicide guard of `playout_audit.play_safe`). Pair k redeals the mover's unseen cards and uses
  the same dice in every branch. Pooled by side × region, then by the region's scoring status and
  era: a positive "region − model" says the model under-invests there; negative, that it is right
  to look elsewhere.

As always here, a branch is valued by how the model follows it up.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import ts_engine as ts

from ai.eval.branch_oracle import PolicyFn, _decider, _pair_start
from ai.eval.doctrine_census import _map
from ai.eval.ops_block import NODE_OFFSET, N_COUNTRIES, is_block_start
from ai.eval.playout_audit import play_safe
from ai.eval.reply_probe import _ar_key, position_link
from bindings.action_encoder import ActionEncoder

REGIONS = ("Europe", "Asia", "Middle East", "Africa", "Central America", "South America")
#: The region's own scoring card (Southeast Asia Scoring scores a subregion of Asia and is left out).
REGION_CARD = {"Europe": 2, "Asia": 1, "Middle East": 3, "Africa": 79, "Central America": 37, "South America": 81}
STATUS = ("in mover's hand", "live elsewhere", "discarded", "not yet in the game")
ERAS = ((1, 3, "Early War"), (4, 7, "Mid War"), (8, 10, "Late War"))
_HANDS = {ts.Player.US: (ts.CardLocation.HAND_US_KNOWN, ts.CardLocation.HAND_US_UNKNOWN),
          ts.Player.USSR: (ts.CardLocation.HAND_USSR_KNOWN, ts.CardLocation.HAND_USSR_UNKNOWN)}
_LIVE = {ts.CardLocation.DRAW_DECK, *_HANDS[ts.Player.US], *_HANDS[ts.Player.USSR]}


def region_of() -> List[str]:
    return [str(r["region"]) for r in _map()]


def region_mask(region: str) -> np.ndarray:
    """Bool mask over flat actions: the point nodes of the countries in `region`."""
    m = np.zeros(int(ActionEncoder.FLAT_ACTION_SIZE), dtype=bool)
    for c, r in enumerate(region_of()):
        if r == region:
            m[NODE_OFFSET + c] = True
    return m


def status(st: ts.GameState, mover: ts.Player, region: str) -> str:
    loc = st.get_card_location(REGION_CARD[region])
    if loc in _HANDS[mover]:
        return STATUS[0]
    if loc in _LIVE:
        return STATUS[1]
    if loc == ts.CardLocation.UNAVAILABLE:
        return STATUS[3]
    return STATUS[2]


def era(turn: int) -> str:
    return next(nm for lo, hi, nm in ERAS if lo <= max(1, min(turn, 10)) <= hi)


def _placement(st: ts.GameState, action: int) -> Optional[Tuple[str, str, str, str]]:
    """(side, region, era, every region's scoring status as one digit each in REGIONS order) of an
    influence-Ops point, else None. All six statuses, so a region's share can be read under its own
    card's status whichever region the point went to."""
    ctx = st.ctx()
    if not (NODE_OFFSET <= action < NODE_OFFSET + N_COUNTRIES):
        return None
    if ctx.decision_type != ts.DecisionType.POINT_NODE or ctx.op_mode != ts.OpMode.INFLUENCE \
            or int(ctx.resolving_card) != 0 or st.current_phase != ts.Phase.ACTION_ROUND:
        return None
    mover = _decider(st)
    reg = region_of()[action - NODE_OFFSET]
    code = "".join(str(STATUS.index(status(st, mover, r))) for r in REGIONS)
    return ("US" if mover == ts.Player.US else "USSR", reg, era(int(st.turn)), code)


def _tally(counts: Dict[str, int], key: Tuple[str, str, str, str]) -> None:
    k = "|".join(key)
    counts[k] = counts.get(k, 0) + 1


def collect(act: PolicyFn, n: int, seed: int, envs: int = 32, accept: float = 0.06, per_game: int = 6,
            max_steps: int = 3_000_000) -> Tuple[List[ts.GameState], Dict[str, int], int]:
    """Greedy self-play until ~n/(per_game·0.6) games finish: every influence point tallied, and up
    to `n` influence-play starts drawn uniformly from the finished games."""
    rng = np.random.default_rng(seed)
    runner = ts.VectorizedBatchRunner(envs, seed * 41 + 23)
    runner.refresh_all()
    counts: Dict[str, int] = {}
    pending_counts: Dict[int, Dict[str, int]] = {}
    pool: Dict[int, List[ts.GameState]] = {}
    game = list(range(envs))
    taken = [0] * envs
    done: List[int] = []
    started, finished = envs, 0
    want = max(1, int(math.ceil(n / (per_game * 0.6))))
    for _ in range(max_steps):
        if finished >= want:
            break
        obs = np.asarray(runner.get_observations())
        masks = np.asarray(runner.get_action_masks())
        acts = act(obs, masks)
        for i in range(envs):
            if not masks[i][NODE_OFFSET:NODE_OFFSET + N_COUNTRIES].any():
                continue
            st = runner.get_state(i)
            key = _placement(st, int(acts[i]))
            if key is not None:
                _tally(pending_counts.setdefault(game[i], {}), key)
            if taken[i] < per_game and is_block_start(st) and rng.random() < accept:
                pool.setdefault(game[i], []).append(st.clone())
                taken[i] += 1
        runner.step_flat_all([int(x) for x in acts], auto_advance=True)
        ends = np.flatnonzero(np.array(runner.get_terminals())).tolist()
        for i in ends:
            done.append(game[i])
            finished += 1
            runner.reset_game(int(i), seed * 1_000_003 + started)
            game[i] = started
            started += 1
            taken[i] = 0
        if ends:
            runner.refresh_all()
    for g in done:
        for k, v in pending_counts.get(g, {}).items():
            counts[k] = counts.get(k, 0) + v
    starts = [s for g in done for s in pool.get(g, [])]
    if len(starts) > n:
        starts = [starts[j] for j in sorted(rng.choice(len(starts), n, replace=False))]
    return starts, counts, len(done)


def human_counts(paths: Sequence[Any]) -> Tuple[Dict[str, int], int]:
    """The same tally over the human corpus's influence-Ops points."""
    from ai.eval.human_disagree import load_game
    from tools.lib.ts_replayer_convert import convert_game

    counts: Dict[str, int] = {}
    games = 0
    for path in paths:
        local: Dict[str, int] = {}

        def on_decision(state: ts.GameState, mover: ts.Player, entry: Any, chosen: int) -> None:
            key = _placement(state, int(chosen))
            if key is not None:
                _tally(local, key)

        conv = convert_game(load_game(path), on_decision=on_decision)
        if conv.skipped:
            continue
        games += 1
        for k, v in local.items():
            counts[k] = counts.get(k, 0) + v
    return counts, games


def play(act: PolicyFn, starts: Sequence[ts.GameState], pairs: int, seed: int, chunk: int = 1536
         ) -> List[Dict[str, Any]]:
    """Per start: the model's play and one region-restricted branch per reachable region."""
    masks = {r: region_mask(r) for r in REGIONS}
    plan = []
    for st in starts:
        legal = np.asarray(ActionEncoder.get_legal_mask(st)).astype(bool)
        regs = [r for r in REGIONS if (legal & masks[r]).any()]
        if regs:
            plan.append((st, ["model"] + regs))
    rows: List[Dict[str, Any]] = []
    lo = 0
    while lo < len(plan):
        group, size = [], 0
        while lo < len(plan) and (not group or size + len(plan[lo][1]) * pairs <= chunk):
            group.append(plan[lo])
            size += len(plan[lo][1]) * pairs
            lo += 1
        games, movers, keys, restrict = [], [], [], []
        for gi, (st, branches) in enumerate(group):
            mover = _decider(st)
            card = int(st.ctx().pending_op_card)
            for k in range(pairs):
                base = _pair_start(st, k, seed * 100_003 + lo * 7 + gi, "resample")
                for b in branches:
                    games.append(base.clone())
                    movers.append(mover)
                    keys.append(_ar_key(st))
                    restrict.append(None if b == "model" else (card, int(mover), masks[b]))
        res = play_safe(games, movers, keys, act, seed + lo, restrict=restrict)
        at = 0
        for st, branches in group:
            nb = len(branches)
            sc = np.array(res[at:at + nb * pairs]).reshape(pairs, nb)
            at += nb * pairs
            mover = _decider(st)
            row: Dict[str, Any] = {"side": "US" if mover == ts.Player.US else "USSR", "turn": int(st.turn),
                                   "era": era(int(st.turn)), "ops": int(st.ctx().pending_ops_value), "pairs": pairs,
                                   "model": float(sc[:, 0].mean()), "regions": {}}
            for j, b in enumerate(branches[1:], start=1):
                d = sc[:, j] - sc[:, 0]
                row["regions"][b] = {"diff": float(d.mean()), "status": status(st, mover, b)}
            rows.append(row)
    return rows


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    if len(a) < 2:
        return (float(a.mean()) if len(a) else float("nan")), float("nan")
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a)))


def _share_table(bot: Dict[str, int], hum: Dict[str, int], side: str, by: Optional[int], values: Sequence[str]
                 ) -> List[str]:
    """Rows: region; columns: model and human shares of `side`'s influence points (within each
    value of field `by`: 2 era, 3 scoring status; None: overall)."""
    def shares(counts: Dict[str, int], val: Optional[str]) -> Dict[str, float]:
        field = by if by is not None else 0
        sel = {k: v for k, v in counts.items()
               if k.split("|")[0] == side and (val is None or k.split("|")[field] == val)}
        tot = sum(sel.values()) or 1
        return {r: sum(v for k, v in sel.items() if k.split("|")[1] == r) / tot for r in REGIONS}
    vals: Sequence[Optional[str]] = list(values) if by is not None else [None]
    head = "| region | " + " | ".join(f"model {v or ''}".strip() + f" | human {v or ''}".strip() for v in vals) + " |"
    out = [head, "|:---|" + "---:|" * (2 * len(vals))]
    sb = {v: shares(bot, v) for v in vals}
    sh = {v: shares(hum, v) for v in vals}
    for r in REGIONS:
        out.append(f"| {r} | " + " | ".join(f"{100 * sb[v][r]:.0f}% | {100 * sh[v][r]:.0f}%" for v in vals) + " |")
    return out


def report(bot_counts: Dict[str, int], hum_counts: Dict[str, int], rows: Sequence[Dict[str, Any]],
           meta: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    out = [f"# Region weight — {meta.get('model', '?')}", "",
           f"Influence-Ops points: {sum(bot_counts.values())} in {meta.get('bot_games', '?')} self-play games, "
           f"{sum(hum_counts.values())} in {meta.get('human_games', '?')} human games. Playouts: {len(rows)} influence "
           f"plays, {meta.get('pairs', '?')} pairs per branch.", ""]
    summary: Dict[str, Any] = {"meta": meta, "cells": {}}
    for side in ("US", "USSR"):
        out += [f"## {side}: where the influence goes", ""] + _share_table(bot_counts, hum_counts, side, None, []) + [""]
        out += [f"### {side}, by era", ""] + _share_table(bot_counts, hum_counts, side, 2, [e for _, _, e in ERAS]) + [""]
        out += [f"### {side}, by the region's own scoring card", "",
                "A region's share of the side's points, among all the points placed while that region's scoring "
                "card had this status.", "",
                "| region | " + " | ".join(f"model, {s_} | human, {s_}" for s_ in STATUS) + " |",
                "|:---|" + "---:|" * (2 * len(STATUS))]
        for ri, r in enumerate(REGIONS):
            cells = []
            for si in range(len(STATUS)):
                for counts in (bot_counts, hum_counts):
                    sel = [(k.split("|"), v) for k, v in counts.items()
                           if k.split("|")[0] == side and k.split("|")[3][ri] == str(si)]
                    tot = sum(v for _, v in sel)
                    mine = sum(v for k, v in sel if k[1] == r)
                    cells.append(f"{100 * mine / tot:.0f}% ({tot})" if tot >= 20 else "—")
            out.append(f"| {r} | " + " | ".join(cells) + " |")
        out.append("")

        out += [f"## {side}: is it right? Region − model's own play, pooled", "",
                "| region | plays | region − model | by status: " + " / ".join(STATUS) + " | by era: Early / Mid / Late |",
                "|:---|---:|---:|:---|:---|"]
        for r in REGIONS:
            d = [row["regions"][r]["diff"] for row in rows if row["side"] == side and r in row["regions"]]
            if len(d) < 2:
                out.append(f"| {r} | {len(d)} | — | — | — |")
                continue
            m, se = _mean_se(d)
            st_cells = []
            for s_ in STATUS:
                ds = [row["regions"][r]["diff"] for row in rows
                      if row["side"] == side and r in row["regions"] and row["regions"][r]["status"] == s_]
                if len(ds) >= 5:
                    ms, ses = _mean_se(ds)
                    st_cells.append(f"{100 * ms:+.1f}±{100 * ses:.1f} ({len(ds)})")
                else:
                    st_cells.append("—")
            era_cells = []
            for _, _, e in ERAS:
                de = [row["regions"][r]["diff"] for row in rows
                      if row["side"] == side and r in row["regions"] and row["era"] == e]
                if len(de) >= 5:
                    me, see = _mean_se(de)
                    era_cells.append(f"{100 * me:+.1f}±{100 * see:.1f}")
                else:
                    era_cells.append("—")
            out.append(f"| {r} | {len(d)} | **{100 * m:+.1f} ± {100 * se:.1f}** | {' / '.join(st_cells)} | "
                       f"{' / '.join(era_cells)} |")
            summary["cells"][f"{side}|{r}"] = {"n": len(d), "diff": (m, se)}
        out.append("")
    out += ["Reading the playout table: each branch keeps the whole influence play inside the region (the model "
            "choosing where), so a negative number is the cost of being forced there, not proof the region is "
            "worthless; a positive one says the model's own play leaves value in that region.", ""]
    return "\n".join(out) + "\n", summary
