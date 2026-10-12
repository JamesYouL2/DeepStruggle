#!/usr/bin/env python3
"""Control breaks: how often a side spends Ops to break the opponent's control of a country, whether
the break lasts, and what it costs -- a checkpoint against the humans of the ts-replayer corpus.

A point of Ops influence placed in a country the opponent controls costs 2 Ops. A *break play* is
one card's Ops influence with at least one such point; its *countries* are those it entered under
the opponent's control. Each ends the action round **flipped** (the mover controls it), **broken**
(nobody does) or **held** (the opponent still does), and a flipped or broken country is **retaken**
when the opponent controls it again by the time the mover's next action round starts.

* `human` -- every Ops influence point and break play of the corpus's games, each point with its
  position (`pos=` token);
* `selfplay` -- the same from a checkpoint's greedy self-play; `--bank` also writes one row per
  break play, at its first point into the opponent's control, as the input of `playouts`;
* `onhuman` -- the checkpoint at each human point's position: its pick, and its probability of a
  target the opponent controls and of Cuba;
* `playouts` -- per bank row, paired playouts of the checkpoint's own play against the same card's
  Ops kept out of every country the opponent controls (`NoBreak` steers the rest of the play), the
  checkpoint greedy on both sides afterwards (`--continue-with` searches instead);
* `report` -- the tables.

    export PYTHONPATH=.:build/release
    python tools/scripts/control_census.py human --out human.json.gz
    python tools/scripts/control_census.py selfplay --checkpoint m.pt --games 4096 --out sp.json.gz \\
        --bank-games 400 --bank bank.jsonl.gz
    python tools/scripts/control_census.py onhuman --human human.json.gz --checkpoint m.pt --out onh.json.gz
    python tools/scripts/control_census.py playouts --model m.pt --bank bank.jsonl.gz --part 1/1 \\
        --out playouts.jsonl.gz --pairs 512
    python tools/scripts/control_census.py report --human human.json.gz --selfplay sp.json.gz \\
        --onhuman onh.json.gz --bank bank.jsonl.gz --playouts playouts.jsonl.gz --out report.md
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
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from ai.eval.paired_playouts import apply_move, ar_key, decider, pair_start, play_safe
from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import (corpus_map, feed_corpus_game, load_policy, position_token, selfplay,
                                     state_from_token)
from tools.lib.jsonl import read_jsonl, sha256_file, write_jsonl
from tools.lib.player_agent import load_agent
from tools.lib.position_bank import position_seed
from tools.scripts.placement_census import COUNTRY, SCORING_CARDS, in_area

SCHEMA = 1
NODE = int(ActionEncoder.NODE_OFFSET)
CUBA, FIDEL = 71, 8
OPEN, CHIP, BREAK = 0, 1, 2       # a point's class: elsewhere; into control it does not break; breaking it
SIDE = {int(ts.Player.US): "US", int(ts.Player.USSR): "USSR"}
ERAS = (("T1-3", 1, 3), ("T4-7", 4, 7), ("T8-10", 8, 10))


def player(side: int) -> ts.Player:
    return ts.Player.US if side == int(ts.Player.US) else ts.Player.USSR


def opponent(who: ts.Player) -> ts.Player:
    return ts.Player.USSR if who == ts.Player.US else ts.Player.US


def ops_influence(st: ts.GameState) -> bool:
    """An Ops point of influence in an action round (not an event's placement, not the setup)."""
    ctx = st.ctx()
    return (st.current_phase == ts.Phase.ACTION_ROUND and ctx.decision_type == ts.DecisionType.POINT_NODE
            and int(ctx.resolving_card) == 0 and ctx.op_mode == ts.OpMode.INFLUENCE)


def controls(own: int, other: int, stability: int) -> bool:
    return own >= stability and own - other >= stability


def point_class(st: ts.GameState, who: ts.Player, cid: int) -> int:
    """OPEN, or, when the opponent controls `cid`, BREAK if one more point of `who`'s ends that
    control and CHIP if it does not."""
    if ts.Scoring.get_country_control(st, cid) != opponent(who):
        return OPEN
    c = st.get_country(cid)
    mine, theirs = (c.us_influence, c.ussr_influence) if who == ts.Player.US else (c.ussr_influence, c.us_influence)
    return CHIP if controls(theirs, mine + 1, int(COUNTRY[cid]["stability"])) else BREAK


def held_mask(st: ts.GameState, who: ts.Player) -> int:
    return sum(1 << i for i, (card, _) in enumerate(SCORING_CARDS)
               if ts.in_hand_of(st.get_card_location(card), who))


def era(turn: int) -> str:
    return next(name for name, lo, hi in ERAS if lo <= turn <= hi)


def row_id(pos: str) -> str:
    return hashlib.sha1(pos.encode()).hexdigest()[:16]


def fidel_where(st: ts.GameState) -> str:
    """Where Fidel is, as the US can know it: gone (removed), US hand, discard, ongoing, or unseen
    (the deck or the USSR's hand)."""
    loc = st.get_card_location(FIDEL)
    if loc == ts.CardLocation.REMOVED_FROM_GAME:
        return "gone"
    if loc in (ts.CardLocation.HAND_US_KNOWN, ts.CardLocation.HAND_US_UNKNOWN):
        return "US hand"
    if loc == ts.CardLocation.DISCARD_PILE:
        return "discard"
    return "unseen"


class BreakTracker:
    """Every Ops influence point of one game, and its break plays followed to the mover's next
    action round and the end of the turn; and the US's *Cuba plays*, a card's Ops with a point in
    Cuba while Fidel is still to come, followed to Fidel's event. `positions` keeps each point's
    position; `bank` keeps each play's position and move at its first point into the opponent's
    control (or into Cuba)."""

    def __init__(self, positions: bool = False, bank: bool = False) -> None:
        self.positions, self.bank = positions, bank
        self.points: List[Dict[str, Any]] = []
        self.plays: List[Dict[str, Any]] = []
        self.cuba: List[Dict[str, Any]] = []
        self._play: Optional[Dict[str, Any]] = None
        self._key: Optional[Tuple[int, int, int, int]] = None
        self._pending: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
        self._cuba_key: Optional[Tuple[int, int, int, int]] = None
        self._us_in_cuba = 0
        self._fidel_gone = False

    def observe(self, st: ts.GameState, a: int) -> None:
        self._follow(st)
        self._follow_fidel(st)
        if not ops_influence(st):
            return
        who = decider(st)
        cid = int(ts.decode_flat_action(st, a).primary_id)
        if cid not in COUNTRY:
            return
        k = point_class(st, who, cid)
        rec: Dict[str, Any] = {"s": int(who), "t": int(st.turn), "ar": int(st.action_round), "c": cid, "k": k}
        if cid == CUBA:
            rec["fidel"] = st.get_card_location(FIDEL) != ts.CardLocation.REMOVED_FROM_GAME
        if who == ts.Player.US:
            rec["fw"] = fidel_where(st)
        if self.positions:
            rec["pos"] = position_token(st)
        self.points.append(rec)
        key = ar_key(st)
        if cid == CUBA and who == ts.Player.US and rec["fidel"]:
            if self._cuba_key != key:
                cp: Dict[str, Any] = {"s": int(who), "t": int(st.turn), "ar": int(st.action_round),
                                      "card": int(st.ctx().pending_op_card), "fidel_at": fidel_where(st),
                                      "us_before": int(st.get_country(CUBA).us_influence), "pts": 0,
                                      "fired": None, "us_when_fired": None}
                if self.bank:
                    cp["pos"], cp["raw"] = position_token(st), int(a)
                self.cuba.append(cp)
                self._cuba_key = key
            self.cuba[-1]["pts"] += 1
        play = self._play if self._key == key else None
        if k != OPEN:
            if play is None:
                new_play: Dict[str, Any] = {
                    "s": int(who), "t": int(st.turn), "ar": int(st.action_round),
                    "card": int(st.ctx().pending_op_card), "held": held_mask(st, who),
                    "vp": int(st.victory_points), "defcon": int(st.defcon), "countries": {}}
                if self.bank:
                    new_play["pos"], new_play["raw"] = position_token(st), int(a)
                self._play, self._key = new_play, key
                self.plays.append(new_play)
                play = new_play
            e = play["countries"].setdefault(cid, {"c": cid, "pts": 0, "ops": 0, "end": None,
                                                   "retaken": None, "turn_end": None})
            e["pts"] += 1
            e["ops"] += 2
        elif play is not None and cid in play["countries"]:
            play["countries"][cid]["pts"] += 1
            play["countries"][cid]["ops"] += 1

    def _follow(self, st: ts.GameState) -> None:
        if self._play is not None and ar_key(st) != self._key:
            who = player(self._play["s"])
            for e in self._play["countries"].values():
                ctl = ts.Scoring.get_country_control(st, int(e["c"]))
                e["end"] = "flipped" if ctl == who else ("held" if ctl == opponent(who) else "broken")
                if e["end"] != "held":
                    self._pending.append((e, self._play))
            self._play, self._key = None, None
        still = []
        for e, play in self._pending:
            who = player(play["s"])
            theirs = ts.Scoring.get_country_control(st, int(e["c"])) == opponent(who)
            if (e["retaken"] is None and st.current_phase == ts.Phase.ACTION_ROUND and st.phasing_player == who
                    and (int(st.turn), int(st.action_round)) != (play["t"], play["ar"])):
                e["retaken"] = theirs
            if e["turn_end"] is None and int(st.turn) > play["t"]:
                e["turn_end"] = theirs
            if e["retaken"] is None or e["turn_end"] is None:
                still.append((e, play))
        self._pending = still

    def _follow_fidel(self, st: ts.GameState) -> None:
        """Fidel's event, the first time Fidel is seen removed: the turn, and the US's influence in
        Cuba at the decision before, for every Cuba play still waiting for it."""
        if not self._fidel_gone and st.get_card_location(FIDEL) == ts.CardLocation.REMOVED_FROM_GAME:
            self._fidel_gone = True
            for cp in self.cuba:
                if cp["fired"] is None:
                    cp["fired"], cp["us_when_fired"] = int(st.turn), self._us_in_cuba
        self._us_in_cuba = int(st.get_country(CUBA).us_influence)

    def result(self) -> Dict[str, Any]:
        plays = [dict(p, countries=list(p["countries"].values())) for p in self.plays]
        return {"points": self.points, "plays": plays, "cuba": self.cuba}


# --- the census stages ------------------------------------------------------------------------

def _human_game(path: str) -> Tuple[str, Dict[str, Any], str]:
    t = BreakTracker(positions=True)
    status = feed_corpus_game(path, t)
    empty: Dict[str, Any] = {"points": [], "plays": [], "cuba": []}
    return os.path.basename(path), (t.result() if status != "skipped" else empty), status


def _dump(path: str, obj: Dict[str, Any]) -> None:
    with gzip.open(path, "wt") as f:
        json.dump(obj, f, separators=(",", ":"))


def _load(path: str) -> Dict[str, Any]:
    with gzip.open(path, "rt") as f:
        return json.load(f)


def human(out: str, workers: int, limit: int) -> int:
    points: List[Dict[str, Any]] = []
    plays: List[Dict[str, Any]] = []
    cuba: List[Dict[str, Any]] = []
    status: Dict[str, int] = collections.Counter()
    t0 = time.time()
    for game, res, st in sorted(corpus_map(_human_game, workers, limit), key=lambda x: x[0]):
        status[st] += 1
        points += [dict(r, g=game) for r in res["points"]]
        plays += [dict(p, g=game) for p in res["plays"]]
        cuba += [dict(p, g=game, complete=st == "complete") for p in res["cuba"]]
    _dump(out, {"schema": SCHEMA, "source": "human", "games": dict(status), "seconds": round(time.time() - t0, 1),
                "points": points, "plays": plays, "cuba": cuba})
    print(f"{len(points):,} points, {len(plays):,} break plays, {dict(status)}", file=sys.stderr)
    return 0


def _bank_row(p: Dict[str, Any], kind: str, g: int, pos: str, raw: int) -> Dict[str, Any]:
    return dict(p, schema=SCHEMA, kind=kind, id=row_id(pos), pos=pos, raw=raw, g=g, side=SIDE[int(p["s"])])


def _write_bank(path: str, rows: List[Dict[str, Any]], checkpoint: str, games: int, seed: int) -> None:
    write_jsonl(path, rows)
    json.dump({"schema": SCHEMA, "checkpoint": os.path.basename(checkpoint), "sha256": sha256_file(checkpoint),
               "games": games, "seed": seed, "rows": len(rows)}, open(path + ".meta.json", "w"))


def run_selfplay(checkpoint: str, games: int, seed: int, batch: int, out: str, bank: str, bank_games: int,
                 cuba_bank: str = "", cuba_games: int = 0) -> int:
    fn, features = load_policy(checkpoint)
    t0 = time.time()
    keep = bool(bank or cuba_bank)
    trackers = selfplay(fn, features, games, seed, batch, 0.0, lambda: BreakTracker(bank=keep))
    points: List[Dict[str, Any]] = []
    plays: List[Dict[str, Any]] = []
    cuba: List[Dict[str, Any]] = []
    rows: List[Dict[str, Any]] = []
    crows: List[Dict[str, Any]] = []
    for g, t in enumerate(trackers):
        res = t.result()
        points += [dict(r, g=g) for r in res["points"]]
        for p in res["plays"]:
            pos, raw = p.pop("pos", None), p.pop("raw", None)
            plays.append(dict(p, g=g))
            if bank and g < bank_games and pos is not None:
                rows.append(_bank_row(p, "break", g, pos, int(raw)))
        for p in res["cuba"]:
            pos, raw = p.pop("pos", None), p.pop("raw", None)
            cuba.append(dict(p, g=g, complete=True))
            if cuba_bank and g < cuba_games and pos is not None:
                crows.append(_bank_row(p, "cuba", g, pos, int(raw)))
    _dump(out, {"schema": SCHEMA, "source": "selfplay", "checkpoint": os.path.basename(checkpoint),
                "sha256": sha256_file(checkpoint), "games": games, "seed": seed, "seconds": round(time.time() - t0, 1),
                "points": points, "plays": plays, "cuba": cuba})
    if bank:
        _write_bank(bank, rows, checkpoint, bank_games, seed)
    if cuba_bank:
        _write_bank(cuba_bank, crows, checkpoint, cuba_games, seed)
    print(f"{len(points):,} points, {len(plays):,} break plays, {len(cuba):,} Cuba plays from {games:,} games "
          f"({len(rows):,} + {len(crows):,} bank rows) in {time.time() - t0:.0f}s", file=sys.stderr)
    return 0


def opponent_controlled(st: ts.GameState, who: ts.Player) -> List[int]:
    opp = opponent(who)
    return [cid for cid in COUNTRY if ts.Scoring.get_country_control(st, cid) == opp]


def onhuman(human_path: str, checkpoint: str, out: str, chunk: int) -> int:
    """The checkpoint at each human point's position: its argmax target and that target's class, and
    its probability of a target the opponent controls and of Cuba."""
    pts = _load(human_path)["points"]
    fn, features = load_policy(checkpoint)
    res: List[Dict[str, Any]] = []
    t0 = time.time()
    for lo in range(0, len(pts), chunk):
        states = [state_from_token(r["pos"]) for r in pts[lo:lo + chunk]]
        obs = np.stack([np.asarray(ts.extract_observation_features(s, decider(s), features), dtype=np.float32)
                        for s in states])
        masks = np.stack([np.asarray(ActionEncoder.get_legal_mask(s), dtype=np.uint8) for s in states])
        logits = np.asarray(fn(obs, masks), dtype=np.float64)
        for st, lg, m in zip(states, logits, masks):
            legal = np.flatnonzero(m)
            z = np.exp(lg[legal] - lg[legal].max())
            p = z / z.sum()
            theirs = {NODE + c for c in opponent_controlled(st, decider(st))}
            pick = int(legal[int(np.argmax(p))])
            cid = int(ts.decode_flat_action(st, pick).primary_id)
            res.append({"pick": cid, "k": point_class(st, decider(st), cid) if cid in COUNTRY else OPEN,
                        "p_opp": round(float(sum(pa for a, pa in zip(legal, p) if int(a) in theirs)), 5),
                        "p_cuba": round(float(sum(pa for a, pa in zip(legal, p) if int(a) == NODE + CUBA)), 5)})
        print(f"  {len(res):,}/{len(pts):,} in {time.time() - t0:.0f}s", file=sys.stderr)
    _dump(out, {"schema": SCHEMA, "checkpoint": os.path.basename(checkpoint), "sha256": sha256_file(checkpoint),
                "human": os.path.basename(human_path), "rows": res})
    return 0


# --- the playouts -----------------------------------------------------------------------------

#: a steered row: its mover, the action round it starts in, and the countries to avoid (None: every
#: country the opponent controls at that point)
Steer = Tuple[ts.Player, Tuple[int, int, int, int], Optional[FrozenSet[int]]]


def avoid_of(row: Dict[str, Any]) -> Optional[FrozenSet[int]]:
    """What a bank row's steered line keeps out of: opponent control for a break play, Cuba for a
    Cuba play."""
    return frozenset({CUBA}) if row.get("kind", "break") == "cuba" else None


class NoBreak:
    """Steers chosen playout rows: until the action round they start in ends, the mover's Ops
    influence points avoid every country the opponent controls (or the row's own set), unless
    nothing else is legal. `changed[i]` counts the decisions where it narrowed row i's mask,
    `forced[i]` those it could not."""

    def __init__(self, rows: Dict[int, Steer]) -> None:
        self.rows = dict(rows)
        self.changed: Dict[int, int] = collections.Counter()
        self.forced: Dict[int, int] = collections.Counter()

    def wants(self, i: int) -> bool:
        return i in self.rows

    def steer(self, i: int, st: ts.GameState, mask: np.ndarray) -> np.ndarray:
        who, key, avoid = self.rows[i]
        if ar_key(st) != key:
            del self.rows[i]
            return mask
        if not ops_influence(st) or decider(st) != who:
            return mask
        bad = [NODE + c for c in (opponent_controlled(st, who) if avoid is None else sorted(avoid)) if mask[NODE + c]]
        if not bad:
            return mask
        if len(bad) == int(np.count_nonzero(mask)):
            self.forced[i] += 1
            return mask
        out = mask.copy()
        out[bad] = 0
        self.changed[i] += 1
        return out

    def seen(self, i: int, st: ts.GameState, action: int) -> None:
        return None


def nobreak_compare(rows: Sequence[Dict[str, Any]], act: Any, pairs: int, seed: int,
                    select: Any = None, chunk: int = 1536) -> List[Dict[str, Any]]:
    """Per row, the mover's score in each pair for its own move (`raw`) and for the same card's Ops
    kept out of opponent control, or out of Cuba for a Cuba row (`steered`); both lines of a pair
    share the redeal and the dice."""
    out: List[Dict[str, Any]] = []
    per = 2 * pairs
    step = max(1, chunk // per)
    for lo in range(0, len(rows), step):
        batch = rows[lo:lo + step]
        starts: List[ts.GameState] = []
        movers: List[ts.Player] = []
        keys: List[Tuple[int, int, int, int]] = []
        steered: Dict[int, Steer] = {}
        for r in batch:
            st = state_from_token(r["pos"])
            who, key = decider(st), ar_key(st)
            s0 = position_seed(str(r["id"]), seed)
            for k in range(pairs):
                base = pair_start(st, k, s0)
                starts.append(apply_move(base, int(r["raw"])))
                steered[len(starts)] = (who, key, avoid_of(r))
                starts.append(base)
                movers += [who, who]
                keys += [key, key]
        watch = NoBreak(steered)
        scores = play_safe(starts, movers, keys, act, position_seed(str(batch[0]["id"]), seed),
                           select=select, watch=watch)
        for j, r in enumerate(batch):
            i0 = j * per
            out.append({"schema": SCHEMA, "id": r["id"], "pairs": pairs,
                        "scores": {"raw": scores[i0:i0 + per:2], "steered": scores[i0 + 1:i0 + per:2]},
                        "changed": sum(watch.changed[i0 + 2 * k + 1] for k in range(pairs)) / pairs,
                        "forced": sum(watch.forced[i0 + 2 * k + 1] for k in range(pairs)) / pairs})
    return out


def playouts(a: argparse.Namespace) -> int:
    k, n = (int(x) for x in a.part.split("/"))
    rows = [r for i, r in enumerate(sorted(read_jsonl([a.bank]), key=lambda r: r["id"])) if i % n == k - 1]
    logits_fn, _ = load_policy(a.model)

    def act(obs: Any, masks: Any) -> Any:
        return np.asarray(logits_fn(obs, masks)).argmax(axis=1).astype(np.int32)

    searcher: Any = None
    if a.continue_with:
        torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))))
        agent: Any = load_agent(f"gumbel:{a.model}:{a.continue_with}", device="cpu")
        agent.reseed(a.seed + 100_003 * k)
        searcher = agent
    t0 = time.time()
    with gzip.open(a.out, "wt") as f:
        for lo in range(0, len(rows), a.batch):
            got = nobreak_compare(rows[lo:lo + a.batch], act, a.pairs, a.seed,
                                  select=searcher.select_actions_batch if searcher is not None else None)
            for r in got:
                f.write(json.dumps(dict(r, continue_with=a.continue_with or "raw"), separators=(",", ":")) + "\n")
            print(f"{lo + len(got)}/{len(rows)} rows, {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
    return 0


# --- the report -------------------------------------------------------------------------------

def ratio_se(num: Dict[Any, float], den: Dict[Any, float]) -> Tuple[float, float]:
    """sum(num) / sum(den), with a standard error clustered by the keys (games)."""
    keys = list(den)
    x = np.array([num.get(g, 0.0) for g in keys])
    d = np.array([den[g] for g in keys])
    if d.sum() == 0:
        return math.nan, math.nan
    r = float(x.sum() / d.sum())
    g = len(keys)
    if g < 2:
        return r, math.nan
    return r, float(math.sqrt(g / (g - 1) * ((x - r * d) ** 2).sum()) / d.sum())


def clustered_mean(vals: Sequence[float], groups: Sequence[Any]) -> Tuple[float, float]:
    by: Dict[Any, List[float]] = collections.defaultdict(list)
    for v, g in zip(vals, groups):
        by[g].append(v)
    return ratio_se({g: sum(v) for g, v in by.items()}, {g: float(len(v)) for g, v in by.items()})


def pct(m: float, se: float) -> str:
    return "–" if math.isnan(m) else (f"{100 * m:.1f}" if math.isnan(se) else f"{100 * m:.1f} ± {100 * se:.1f}")


def share_table(rows: List[Tuple[str, Dict[str, Any]]], pred: Any, onh: Optional[Dict[str, Any]],
                human: Dict[str, Any], title: str, cuba: bool = False) -> List[str]:
    cols = [name for name, _ in rows]
    head = f"| side | turns | {' | '.join(cols)} |"
    if onh is not None:
        head += " same positions: human | same positions: model's pick | model − human |"
    out = [f"### {title}", "", head, "|" + ":---|" * 2 + "---:|" * (len(cols) + (3 if onh else 0))]
    hpts = human["points"]
    for s in (int(ts.Player.US), int(ts.Player.USSR)):
        if cuba and s != int(ts.Player.US):
            continue
        for name, lo, hi in ERAS + (("all", 1, 10),):
            cells = []
            for _, src in rows:
                num: Dict[Any, float] = collections.defaultdict(float)
                den: Dict[Any, float] = collections.defaultdict(float)
                for r in src["points"]:
                    if r["s"] == s and lo <= r["t"] <= hi:
                        den[r["g"]] += 1
                        num[r["g"]] += pred(r)
                cells.append(pct(*ratio_se(num, den)))
            line = f"| {SIDE[s]} | {name} | {' | '.join(cells)} |"
            if onh is not None:
                idx = [i for i, r in enumerate(hpts) if r["s"] == s and lo <= r["t"] <= hi]
                hv = [float(pred(hpts[i])) for i in idx]
                mv = [float(pred({"k": onh["rows"][i]["k"], "c": onh["rows"][i]["pick"]})) for i in idx]
                gs = [hpts[i]["g"] for i in idx]
                line += (f" {pct(*clustered_mean(hv, gs))} | {pct(*clustered_mean(mv, gs))} |"
                         f" {pct(*clustered_mean([m - h for m, h in zip(mv, hv)], gs))} |")
            out.append(line)
    return out + [""]


def play_table(rows: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    out = ["| | " + " | ".join(n for n, _ in rows) + " |", "|:---|" + "---:|" * len(rows)]

    def stat(src: Dict[str, Any], s: int, f: Any, base: Any) -> str:
        num: Dict[Any, float] = collections.defaultdict(float)
        den: Dict[Any, float] = collections.defaultdict(float)
        for p in src["plays"]:
            if p["s"] != s:
                continue
            for e in p["countries"]:
                if base(e):
                    den[p["g"]] += 1
                    num[p["g"]] += f(e)
        return pct(*ratio_se(num, den))

    for s in (int(ts.Player.US), int(ts.Player.USSR)):
        side = SIDE[s]
        out.append(f"| **{side}** countries entered under the opponent's control | "
                   + " | ".join(str(sum(len(p["countries"]) for p in src["plays"] if p["s"] == s)) for _, src in rows)
                   + " |")
        for label, f, base in (
                ("flipped (mover controls after the round)", lambda e: e["end"] == "flipped", lambda e: e["end"] is not None),
                ("broken (nobody controls)", lambda e: e["end"] == "broken", lambda e: e["end"] is not None),
                ("held (still the opponent's)", lambda e: e["end"] == "held", lambda e: e["end"] is not None),
                ("of flipped or broken: retaken by the mover's next round", lambda e: bool(e["retaken"]),
                 lambda e: e["end"] in ("flipped", "broken") and e["retaken"] is not None),
                ("of flipped or broken: the opponent's at the turn's end", lambda e: bool(e["turn_end"]),
                 lambda e: e["end"] in ("flipped", "broken") and e["turn_end"] is not None)):
            out.append(f"| {side}: {label} | " + " | ".join(stat(src, s, f, base) for _, src in rows) + " |")
        ops = [np.mean([e["ops"] for p in src["plays"] if p["s"] == s for e in p["countries"]] or [math.nan])
               for _, src in rows]
        out.append(f"| {side}: Ops spent per country entered | " + " | ".join(f"{o:.2f}" for o in ops) + " |")
    return out + [""]


def country_table(rows: List[Tuple[str, Dict[str, Any]]], top: int) -> List[str]:
    out = ["| side | country | " + " | ".join(f"{n}: entered per 1,000 points / retaken" for n, _ in rows) + " |",
           "|:---|:---|" + "---:|" * len(rows)]
    for s in (int(ts.Player.US), int(ts.Player.USSR)):
        stats = []
        for _, src in rows:
            n_pts = sum(1 for r in src["points"] if r["s"] == s)
            ent: Dict[int, int] = collections.Counter()
            ret: Dict[int, List[int]] = collections.defaultdict(list)
            for p in src["plays"]:
                if p["s"] == s:
                    for e in p["countries"]:
                        ent[e["c"]] += 1
                        if e["end"] in ("flipped", "broken") and e["retaken"] is not None:
                            ret[e["c"]].append(int(e["retaken"]))
            stats.append((n_pts, ent, ret))
        order = sorted({c for _, ent, _ in stats for c in ent}, key=lambda c: -max(1000 * st[1][c] / max(1, st[0])
                                                                                     for st in stats))[:top]
        for c in order:
            cells = []
            for n_pts, ent, ret in stats:
                r = ret[c]
                cells.append(f"{1000 * ent[c] / max(1, n_pts):.1f} / " + (f"{100 * sum(r) / len(r):.0f}% of {len(r)}" if r else "–"))
            out.append(f"| {SIDE[s]} | {COUNTRY[c]['name']} | " + " | ".join(cells) + " |")
    return out + [""]


def steered_scores(p: Dict[str, Any]) -> List[float]:
    """The steered line's scores (`nobreak` in the first run's output)."""
    return p["scores"]["steered"] if "steered" in p["scores"] else p["scores"]["nobreak"]


def holds_its_scoring(r: Dict[str, Any]) -> bool:
    c = int(r["b"]["countries"][0]["c"])
    return any(r["b"]["held"] >> i & 1 and in_area(c, area) for i, (_, area) in enumerate(SCORING_CARDS))


def playout_tables(bank: List[Dict[str, Any]], pl: List[Dict[str, Any]], games: int) -> List[str]:
    by = {r["id"]: r for r in pl}
    rows: List[Dict[str, Any]] = []
    for b in bank:
        p = by.get(b["id"])
        if p is None:
            continue
        d = [x - y for x, y in zip(steered_scores(p), p["scores"]["raw"])]
        m = sum(d) / len(d)
        v = sum((t - m) ** 2 for t in d) / (len(d) - 1) / len(d)
        retaken = any(e["retaken"] for e in b["countries"] if e["end"] in ("flipped", "broken"))
        lasted = any(e["end"] in ("flipped", "broken") and e["retaken"] is False for e in b["countries"])
        outcome = "retaken" if retaken else ("lasted" if lasted else
                                             ("held" if all(e["end"] == "held" for e in b["countries"]) else "unknown"))
        rows.append({"b": b, "gain": m, "var": v, "outcome": outcome, "changed": p["changed"]})
    if not rows:
        return []
    out = [f"Paired playouts of {len(rows):,} break plays from {games:,} games, {pl[0]['pairs']} pairs each, "
           f"continuation `{pl[0].get('continue_with', 'raw')}`. Gain = keeping the card's Ops out of opponent "
           "control minus the model's own play, in points of win probability: per play, and summed per game "
           "(both sides). The ± is over plays (positions), which includes the playout noise.", "",
           "| group | plays | per game | mean gain a play | per game, points | share of plays where no-break is better at 2 SE |",
           "|:---|---:|---:|---:|---:|---:|"]

    def line(label: str, sel: List[Dict[str, Any]]) -> str:
        if not sel:
            return f"| {label} | 0 | | | | |"
        g = np.array([r["gain"] for r in sel])
        mean = float(g.mean())
        se = float(g.std(ddof=1) / math.sqrt(len(g))) if len(g) > 1 else math.nan
        tot = 100 * g.sum() / games
        tot_se = 100 * (g.std(ddof=1) * math.sqrt(len(g)) / games) if len(g) > 1 else math.nan
        sig = sum(1 for r in sel if r["gain"] > 2 * math.sqrt(r["var"])) / len(sel)
        return (f"| {label} | {len(sel)} | {len(sel) / games:.2f} | {100 * mean:+.2f} ± {100 * se:.2f} |"
                f" {tot:+.2f} ± {tot_se:.2f} | {100 * sig:.0f}% |")

    out.append(line("all", rows))
    for s in ("US", "USSR"):
        out.append(line(s, [r for r in rows if r["b"]["side"] == s]))
    for name, lo, hi in ERAS:
        out.append(line(name, [r for r in rows if lo <= r["b"]["t"] <= hi]))
    for o in ("retaken", "lasted", "held", "unknown"):
        out.append(line(f"after the round: {o}", [r for r in rows if r["outcome"] == o]))
    for label, f in (("held the scoring card of the first country's region", holds_its_scoring),
                     ("did not", lambda r: not holds_its_scoring(r))):
        out.append(line(label, [r for r in rows if f(r)]))
    out += ["", "By the first country entered (10+ plays):", "",
            "| side | country | plays | mean gain a play | per game, points |", "|:---|:---|---:|---:|---:|"]
    groups: Dict[Tuple[str, int], List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        groups[(r["b"]["side"], int(r["b"]["countries"][0]["c"]))].append(r)
    for (s, c), sel in sorted(groups.items(), key=lambda kv: -sum(r["gain"] for r in kv[1])):
        if len(sel) >= 10:
            g = np.array([r["gain"] for r in sel])
            out.append(f"| {s} | {COUNTRY[c]['name']} | {len(sel)} | {100 * g.mean():+.2f} ± "
                       f"{100 * g.std(ddof=1) / math.sqrt(len(g)):.2f} | {100 * g.sum() / games:+.2f} |")
    out += ["", "The plays where keeping out of opponent control gains most:", "",
            "| side | turn/AR | countries (end, retaken) | gain ± SE | id |", "|:---|:---|:---|---:|:---|"]
    for r in sorted(rows, key=lambda r: -r["gain"])[:15]:
        b = r["b"]
        cs = ", ".join(f"{COUNTRY[e['c']]['name']} ({e['end']}, {e['retaken']})" for e in b["countries"])
        out.append(f"| {b['side']} | T{b['t']} AR{b['ar']} | {cs} | {100 * r['gain']:+.1f} ± "
                   f"{100 * math.sqrt(r['var']):.1f} | [{b['id']}](https://mihaild.github.io/DeepStruggle/?pos={b['pos']}) |")
    return out + [""]


def fidel_table(srcs: List[Tuple[str, Dict[str, Any]]], onh: Optional[Dict[str, Any]],
                hum: Dict[str, Any]) -> List[str]:
    """US points into Cuba, turns 4-10, by where Fidel is (as the US can know it)."""
    places = ("unseen", "discard", "US hand", "gone")
    out = ["### US points into Cuba, turns 4-10, by where Fidel is", "",
           "| Fidel | " + " | ".join(n for n, _ in srcs)
           + (" | same positions: human | model's pick | model − human |" if onh else " |"),
           "|:---|" + "---:|" * (len(srcs) + (3 if onh else 0))]
    for w in places:
        cells = []
        for _, src in srcs:
            num: Dict[Any, float] = collections.defaultdict(float)
            den: Dict[Any, float] = collections.defaultdict(float)
            for r in src["points"]:
                if r["s"] == int(ts.Player.US) and r["t"] >= 4 and r.get("fw") == w:
                    den[r["g"]] += 1
                    num[r["g"]] += r["c"] == CUBA
            cells.append(pct(*ratio_se(num, den)) + f" ({int(sum(den.values())):,})")
        line = f"| {w} | " + " | ".join(cells) + " |"
        if onh is not None:
            idx = [i for i, r in enumerate(hum["points"]) if r["s"] == int(ts.Player.US) and r["t"] >= 4
                   and r.get("fw") == w]
            gs = [hum["points"][i]["g"] for i in idx]
            hv = [float(hum["points"][i]["c"] == CUBA) for i in idx]
            mv = [float(onh["rows"][i]["pick"] == CUBA) for i in idx]
            line += (f" {pct(*clustered_mean(hv, gs))} | {pct(*clustered_mean(mv, gs))} |"
                     f" {pct(*clustered_mean([m - h for m, h in zip(mv, hv)], gs))} |")
        out.append(line)
    return out + [""]


def cuba_play_lines(srcs: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
    out = []
    for name, src in srcs:
        cp = [p for p in src.get("cuba", []) if p.get("complete")]
        if not cp:
            continue
        fired = [p for p in cp if p["fired"] is not None]
        wiped = [p for p in fired if (p["us_when_fired"] or 0) > 0]
        by = collections.Counter(p["fidel_at"] for p in cp)
        out.append(f"* {name} (complete games): {len(cp):,} Cuba plays ({', '.join(f'{k} {v}' for k, v in by.most_common())}); "
                   f"Fidel fired later in {100 * len(fired) / len(cp):.0f}%, with US influence still in Cuba in "
                   f"{100 * len(wiped) / len(cp):.0f}% (mean {np.mean([p['us_when_fired'] for p in wiped] or [0]):.1f} "
                   "influence removed)")
    return out


def cuba_playout_lines(bank: List[Dict[str, Any]], pl: List[Dict[str, Any]], games: int) -> List[str]:
    by = {r["id"]: r for r in pl}
    rows = [(b, [x - y for x, y in zip(steered_scores(by[b["id"]]), by[b["id"]]["scores"]["raw"])])
            for b in bank if b["id"] in by]
    if not rows:
        return []
    out = [f"Paired playouts of {len(rows):,} US Cuba plays from {games:,} games, {by[rows[0][0]['id']]['pairs']} pairs "
           "each: the same card's Ops kept out of Cuba minus the model's own play, points of win probability.", "",
           "| group | plays | per game | mean gain a play | per game, points |", "|:---|---:|---:|---:|---:|"]

    def line(label: str, sel: List[Tuple[Dict[str, Any], List[float]]]) -> str:
        if not sel:
            return f"| {label} | 0 | | | |"
        g = np.array([sum(d) / len(d) for _, d in sel])
        se = g.std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else math.nan
        return (f"| {label} | {len(g)} | {len(g) / games:.2f} | {100 * g.mean():+.2f} ± {100 * se:.2f} |"
                f" {100 * g.sum() / games:+.2f} ± {100 * se * len(g) / games:.2f} |")

    out.append(line("all", rows))
    for w in ("unseen", "discard", "US hand"):
        out.append(line(f"Fidel {w}", [r for r in rows if r[0]["fidel_at"] == w]))
    for name, lo, hi in ERAS:
        out.append(line(name, [r for r in rows if lo <= r[0]["t"] <= hi]))
    out.append(line("Fidel fired later in the game", [r for r in rows if r[0]["fired"] is not None]))
    out.append(line("Fidel never fired", [r for r in rows if r[0]["fired"] is None]))
    return out + [""]


def report(a: argparse.Namespace) -> int:
    hum = _load(a.human)
    srcs: List[Tuple[str, Dict[str, Any]]] = [("humans", hum)]
    for spec in a.selfplay:
        name, path = spec.split("=", 1)
        srcs.append((f"{name} own games", _load(path)))
    onh = _load(a.onhuman.split("=", 1)[1]) if a.onhuman else None
    if onh is not None and len(onh["rows"]) != len(hum["points"]):
        raise ValueError(f"{a.onhuman}: {len(onh['rows'])} rows for {len(hum['points'])} human points")
    md = ["# Control breaks: humans against the model", "",
          f"Humans: {hum['games']} games of the ts-replayer corpus. "
          + "; ".join(f"{n}: {s.get('games', '?')} greedy self-play games of `{s.get('checkpoint', '?')}`"
                      for n, s in srcs[1:]), "",
          "Shares are of that side's Ops influence points (action rounds, not events), ± a standard error "
          "clustered by game. *Same positions* asks the model at each human point's position.", ""]
    md += share_table(srcs, lambda r: r["k"] != OPEN, onh, hum,
                      "Points placed into a country the opponent controls (2 Ops each)")
    md += share_table(srcs, lambda r: r["k"] == BREAK, onh, hum, "... of which the point that breaks control")
    md += share_table(srcs, lambda r: r["c"] == CUBA, onh, hum, "US points into Cuba", cuba=True)
    md += ["### US points into Cuba while Fidel is still to come (not yet removed)", ""]
    for name, src in srcs:
        cu = [r for r in src["points"] if r["s"] == int(ts.Player.US) and r["c"] == CUBA]
        n_pts = sum(1 for r in src["points"] if r["s"] == int(ts.Player.US))
        pend = sum(1 for r in cu if r.get("fidel"))
        md.append(f"* {name}: {len(cu):,} of {n_pts:,} US points into Cuba ({100 * len(cu) / max(1, n_pts):.1f}%); "
                  f"{pend:,} of them with Fidel to come ({100 * pend / max(1, n_pts):.2f}% of US points)")
    md += [""] + fidel_table(srcs, onh, hum)
    md += ["### US Cuba plays while Fidel is to come: does Fidel fire, and wipe the US out", ""] + cuba_play_lines(srcs)
    md += ["", "### Break plays: what happens to the countries entered", ""] + play_table(srcs)
    md += ["### By country", ""] + country_table(srcs, a.top)
    pl = list(read_jsonl(a.playouts)) if a.playouts else []
    for path in a.bank:
        bank = list(read_jsonl([path]))
        games = int(json.load(open(path + ".meta.json"))["games"])
        if bank and bank[0].get("kind", "break") == "cuba":
            md += ["## What a Cuba placement with Fidel to come costs", ""] + cuba_playout_lines(bank, pl, games)
        else:
            md += ["## What a break costs", ""] + playout_tables(bank, pl, games)
    open(a.out, "w").write("\n".join(md) + "\n")
    print(f"wrote {a.out}", file=sys.stderr)
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
    s.add_argument("--seed", type=int, default=71_000)
    s.add_argument("--batch", type=int, default=512)
    s.add_argument("--out", required=True)
    s.add_argument("--bank", default="", help="also write the break plays of the first --bank-games games here")
    s.add_argument("--bank-games", type=int, default=400)
    s.add_argument("--cuba-bank", default="", help="also write the US's Cuba plays with Fidel to come here")
    s.add_argument("--cuba-games", type=int, default=2000)
    o = sub.add_parser("onhuman")
    o.add_argument("--human", required=True)
    o.add_argument("--checkpoint", required=True)
    o.add_argument("--chunk", type=int, default=2048)
    o.add_argument("--out", required=True)
    p = sub.add_parser("playouts")
    p.add_argument("--model", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--part", default="1/1")
    p.add_argument("--out", required=True)
    p.add_argument("--pairs", type=int, default=512)
    p.add_argument("--seed", type=int, default=29)
    p.add_argument("--batch", type=int, default=16, help="bank rows per progress line")
    p.add_argument("--continue-with", default="", help="sims:k -- a Gumbel continuation instead of the greedy model")
    r = sub.add_parser("report")
    r.add_argument("--human", required=True)
    r.add_argument("--selfplay", nargs="*", default=[], metavar="NAME=PATH")
    r.add_argument("--onhuman", default="", metavar="NAME=PATH")
    r.add_argument("--bank", nargs="*", default=[], help="playout banks (break or Cuba), each with its .meta.json")
    r.add_argument("--playouts", nargs="*", default=[])
    r.add_argument("--top", type=int, default=12)
    r.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "human":
        return human(a.out, a.workers, a.limit)
    if a.cmd == "selfplay":
        return run_selfplay(a.checkpoint, a.games, a.seed, a.batch, a.out, a.bank, a.bank_games,
                            a.cuba_bank, a.cuba_games)
    if a.cmd == "onhuman":
        return onhuman(a.human, a.checkpoint, a.out, a.chunk)
    if a.cmd == "playouts":
        return playouts(a)
    return report(a)


if __name__ == "__main__":
    sys.exit(main())
