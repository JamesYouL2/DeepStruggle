"""Turn 1, priced option by option: the headline and every decision of the action rounds.

Greedy self-play games of one network, from its own setup or a forced opening, stopped at the end
of turn 1. At every turn-1 decision with a small option set, and at the first point of every
influence placement and realignment, **every legal option** is played out over `worlds` worlds to
the end of the turn and valued by the network's critic there, from the decider's side
(`ai/search/turn_pricing.py`, the measurement of P32 B4'):

* **A world** redeals the cards the decider cannot see, the same for every option of the world;
  each option has its own dice. Differences between options are paired by world.
* **Kinds.** headline; card to play; event or Operations; Operations mode; coup target;
  realignment (first roll); influence (first point of a placement, the rest placed by the
  network); an event's placement (first point); other event choices.
* **Honest regret.** Picking the best of several noisy options overstates its lead, so the report
  chooses the best option on half the worlds and measures it on the other half (both ways, averaged):
  the expected gain of switching to the priced best, in points of the decider's score.
* **Confirmation.** The clearest departures (the priced best ahead of the network's move by the
  most standard errors, at most a few per kind and card) are replayed to the *end of the game* with
  `ai.eval.paired_playouts.compare`: both moves from the same redeals and dice, the network greedy
  on both sides, no critic.

A verdict values a move by how this network plays on from it, so a move whose point lies in a
follow-up the network does not find reads worse than it is.

    python tools/turn1_pricing.py run --model <ckpt.pt> --games 200 --worlds 64 --out rows.jsonl.gz
    python tools/turn1_pricing.py bank --rows rows*.jsonl.gz --top 60 --out bank.jsonl
    python tools/turn1_pricing.py confirm --bank bank.jsonl --model <ckpt.pt> --pairs 512 --out conf.jsonl
    python tools/turn1_pricing.py report --rows rows*.jsonl.gz --confirm conf*.jsonl --out report.md
"""
from __future__ import annotations

import argparse
import base64
import collections
import gzip
import json
import math
import os
import random
import sys
import time
import zlib
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import ts_engine as ts  # noqa: E402

from ai.eval.paired_playouts import compare, decider, paired_diff  # noqa: E402
from ai.search.turn_pricing import Candidate, forward, price  # noqa: E402
from bindings.action_encoder import ActionEncoder  # noqa: E402
from bindings.ts_env import model_obs_features  # noqa: E402
from tools.lib.game_loop import GameLoop, SettlePolicy  # noqa: E402
from tools.lib.openings import acting_side, scripted_setup_index  # noqa: E402
from tools.lib.player_agent import NeuralAgent, load_agent  # noqa: E402

WORKBENCH = "https://mihaild.github.io/DeepStruggle/"
KINDS: Tuple[str, ...] = ("headline", "card to play", "event or Ops", "Ops mode", "coup target",
                          "realignment (first roll)", "influence (first point)",
                          "event placement (first point)", "other event choice")


class Row(TypedDict):
    id: str
    seed: int
    choice: int                 # decisions of the game before this one
    opening: str
    side: str
    turn: int
    ar: int
    phase: str
    kind: str
    card: int
    card_name: str
    names: List[str]
    options: List[int]
    probs: List[float]
    taken: int                  # index into options
    values: List[List[Optional[float]]]   # options x worlds, the critic's turn-end value in [-1, 1]
    pos: str


class BankRow(Row):
    bank_best: int              # index into options: the priced best that confirmation replays


class Confirmed(TypedDict):
    id: str
    taken: str
    best: str
    pairs: int
    diff: float                 # full-game score, best minus taken, in [0, 1] units
    se: float


# --- recording ------------------------------------------------------------------------------

def position_token(st: ts.GameState) -> str:
    return base64.urlsafe_b64encode(zlib.compress(st.to_save_json().encode())).decode().rstrip("=")


def state_from_token(token: str) -> ts.GameState:
    raw = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
    return ts.state_from_save_json(zlib.decompress(raw).decode("utf-8"))


def kind_of(st: ts.GameState) -> str:
    ctx = st.ctx()
    dt = int(ctx.decision_type)
    if dt == int(ts.DecisionType.SELECT_CARD):
        return "headline" if st.current_phase == ts.Phase.HEADLINE else "card to play"
    if dt == int(ts.DecisionType.SELECT_PLAY_MODE):
        return "event or Ops"
    if dt == int(ts.DecisionType.SELECT_OP_MODE):
        return "Ops mode"
    if dt == int(ts.DecisionType.POINT_NODE):
        if int(ctx.resolving_card or 0):
            return "event placement (first point)"
        if ctx.op_mode == ts.OpMode.COUP:
            return "coup target"
        if ctx.op_mode == ts.OpMode.REALIGN:
            return "realignment (first roll)"
        return "influence (first point)"
    return "other event choice"


def _point_key(st: ts.GameState) -> Tuple[int, ...]:
    ctx = st.ctx()
    return (int(st.turn), int(st.action_round), int(st.current_phase), int(decider(st)),
            int(ctx.pending_op_card), int(ctx.resolving_card or 0), int(ctx.op_mode))


class _Recorder:
    """A greedy source for GameLoop: forces the opening's setup, records every move, and keeps
    the turn-1 decisions that are priced."""

    def __init__(self, model: Any, device: torch.device, opening: str, max_options: int,
                 point_cap: int) -> None:
        self.model, self.device, self.opening = model, device, opening
        self.max_options, self.point_cap = max_options, point_cap
        self.feats = model_obs_features(model)
        self.cursor = {"US": 0, "USSR": 0}
        self.choices: List[int] = []
        self.cands: List[Tuple[Candidate, str]] = []
        self.prev_point: Optional[Tuple[int, ...]] = None

    def _eligible(self, st: ts.GameState, n_legal: int) -> Optional[str]:
        if st.current_phase == ts.Phase.SETUP or int(st.turn) != 1 or n_legal < 2:
            return None
        kind = kind_of(st)
        if int(st.ctx().decision_type) == int(ts.DecisionType.POINT_NODE):
            key = _point_key(st)
            first = key != self.prev_point
            self.prev_point = key
            if kind != "coup target" and not first:
                return None
            return kind if n_legal <= self.point_cap else None
        self.prev_point = None
        return kind if n_legal <= self.max_options else None

    def choose(self, st: ts.GameState, /) -> Optional[ts.MicroAction]:
        if self.opening:
            idx = scripted_setup_index(st, acting_side(st), self.opening, self.cursor)
            if idx is not None:
                self.choices.append(idx)
                return ts.ActionMask.decode_flat_action(st, idx)
        mover = decider(st)
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
        obs = np.asarray(ts.extract_observation_features(st, mover, self.feats), dtype=np.float32)
        lg, _ = forward(self.model, obs[None], mask[None], self.device)
        legal = np.flatnonzero(mask)
        act = int(legal[np.argmax(lg[0][legal])])
        kind = self._eligible(st, len(legal))
        if kind is not None:
            z = lg[0][legal] - lg[0][legal].max()
            p = np.exp(z) / np.exp(z).sum()
            self.cands.append((Candidate(game=0, choice=len(self.choices), state=st.clone(),
                                         options=[int(a) for a in legal], probs=[float(x) for x in p],
                                         taken=act), kind))
        self.choices.append(act)
        return ts.ActionMask.decode_flat_action(st, act)


def _card_in_play(st: ts.GameState) -> int:
    ctx = st.ctx()
    return int(ctx.resolving_card or 0) or int(ctx.pending_op_card or 0)


def price_game(model: Any, device: torch.device, seed: int, opening: str, worlds: int,
               max_options: int, point_cap: int) -> List[Row]:
    """One game's turn-1 decisions, every option priced."""
    st = ts.GameState()
    ts.Engine.init_game(st, int(seed))
    rec = _Recorder(model, device, opening, max_options, point_cap)
    GameLoop(st, {ts.Player.US: rec, ts.Player.USSR: rec}, settle=SettlePolicy.RECORD_FORCED,
             stop=lambda s: int(s.turn) > 1).run()
    cands = [c for c, _ in rec.cands]
    if not cands:
        return []
    price(model, device, cands, worlds, seed=(int(seed) * 7919 + 17) % (1 << 30))
    rows: List[Row] = []
    for c, kind in rec.cands:
        s = c.state
        card = _card_in_play(s)
        vals = c.values if c.values is not None else np.full((len(c.options), 0), np.nan)
        rows.append({
            "id": f"{seed}-{c.choice}", "seed": int(seed), "choice": c.choice, "opening": opening,
            "side": "US" if decider(s) == ts.Player.US else "USSR", "turn": int(s.turn),
            "ar": int(s.action_round), "phase": str(s.current_phase).replace("Phase.", ""),
            "kind": kind, "card": card,
            "card_name": str(ts.CardData.get_card_info(card)["name"]) if card else "",
            "names": [ActionEncoder.get_action_name(s, a) for a in c.options],
            "options": c.options, "probs": [round(p, 5) for p in c.probs],
            "taken": c.options.index(c.taken),
            "values": [[None if math.isnan(x) else round(float(x), 4) for x in r] for r in vals],
            "pos": position_token(s),
        })
    return rows


# --- statistics -----------------------------------------------------------------------------

def _values(r: Row) -> np.ndarray:
    v = np.array([[np.nan if x is None else x for x in row] for row in r["values"]], dtype=float)
    keep = ~np.isnan(v).any(axis=0)          # a world where some option was illegal is dropped
    return v[:, keep]


class Priced(TypedDict):
    best: int
    diff: float          # mean(best - taken) over all worlds, in [-1, 1] value units
    se: float
    honest: float        # split-half regret, value units
    worlds: int


def priced(r: Row) -> Optional[Priced]:
    v = _values(r)
    if v.shape[1] < 8:
        return None
    t = r["taken"]
    means = v.mean(axis=1)
    b = int(np.argmax(means))
    d = v[b] - v[t]
    se = float(d.std(ddof=1) / math.sqrt(len(d)))
    half = v.shape[1] // 2
    a, c = v[:, :half], v[:, half:2 * half]
    honest = 0.5 * ((c[int(np.argmax(a.mean(axis=1)))] - c[t]).mean()
                    + (a[int(np.argmax(c.mean(axis=1)))] - a[t]).mean())
    return {"best": b, "diff": float(d.mean()), "se": se, "honest": float(honest), "worlds": int(v.shape[1])}


def read_rows(paths: Iterable[str]) -> List[Row]:
    rows: List[Row] = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    return rows


def select_bank(rows: Sequence[Row], top: int, z_min: float, per_kind_card: int) -> List[BankRow]:
    """The clearest departures: the priced best ahead of the network's move by z_min+ SE, most
    standard errors first, at most `per_kind_card` per (kind, card)."""
    scored = []
    for r in rows:
        p = priced(r)
        if p and p["best"] != r["taken"] and p["se"] > 0 and p["diff"] / p["se"] >= z_min:
            scored.append((p["diff"] / p["se"], r, p))
    scored.sort(key=lambda x: -x[0])
    out: List[BankRow] = []
    seen: Dict[Tuple[str, int], int] = collections.Counter()
    for _z, r, p in scored:
        k = (r["kind"], r["card"])
        if seen[k] >= per_kind_card:
            continue
        seen[k] += 1
        b: BankRow = {**r, "bank_best": p["best"]}
        out.append(b)
        if len(out) >= top:
            break
    return out


# --- report ---------------------------------------------------------------------------------

def _pts(x: float, se: Optional[float] = None) -> str:
    """A value difference ([-1, 1] units) in points of the decider's score (half a value unit)."""
    if math.isnan(x):
        return "—"
    return f"{50 * x:+.2f}" + (f" ± {50 * se:.2f}" if se is not None and not math.isnan(se) else "")


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    if not xs:
        return math.nan, math.nan
    a = np.asarray(xs, dtype=float)
    return float(a.mean()), (float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else math.nan)


def report(rows: Sequence[Row], confirmed: Sequence[Confirmed], bank: Sequence[BankRow],
           label: str, z_min: float) -> str:
    games = len({r["seed"] for r in rows})
    openings = sorted({r["opening"] or "the network's own" for r in rows})
    pr = [(r, priced(r)) for r in rows]
    pr = [(r, p) for r, p in pr if p is not None]
    worlds = int(np.median([p["worlds"] for _, p in pr])) if pr else 0
    lines = [f"# Turn 1, priced option by option: `{label}`", "",
             f"{games} greedy self-play games from {', '.join(openings)}; {len(pr):,} turn-1 decisions, "
             f"each option played over ~{worlds} worlds to the end of the turn and valued by the critic "
             "there. Gains are in points of the decider's score (a whole game is 100); ± is one standard error.",
             "",
             "* **lead of the priced best**: its mean lead over the network's move on all worlds -- an upper "
             "bound, since the best of several noisy options is picked on the same worlds it is measured on;",
             "* **honest gain**: the best option chosen on half the worlds, measured on the other half -- what "
             "switching to the priced best is expected to be worth; where the options are close it can fall below "
             "zero (the half picks a worse option), so it is a lower bound on what a perfect pick would gain;",
             f"* **clear**: the priced best ahead of the network's move by {z_min:g}+ SE over all worlds.", "",
             "## By kind of decision", "",
             "| kind | side | decisions | options (median) | network's move is the priced best | clear | lead of the priced best (upper bound) | honest gain per decision | honest gain per game |",
             "|:---|:---|---:|---:|---:|---:|---:|---:|---:|"]
    for kind in KINDS:
        for side in ("US", "USSR"):
            sel = [(r, p) for r, p in pr if r["kind"] == kind and r["side"] == side]
            if not sel:
                continue
            n = len(sel)
            top = sum(1 for r, p in sel if p["best"] == r["taken"])
            clear = sum(1 for r, p in sel if p["best"] != r["taken"] and p["se"] > 0 and p["diff"] / p["se"] >= z_min)
            m, se = _mean_se([p["honest"] for _, p in sel])
            up, up_se = _mean_se([p["diff"] for _, p in sel])
            per_game = m * n / max(1, games)
            lines.append(f"| {kind} | {side} | {n:,} | {int(np.median([len(r['options']) for r, _ in sel]))} | "
                         f"{100 * top / n:.0f}% | {100 * clear / n:.1f}% | {_pts(up, up_se)} | {_pts(m, se)} | "
                         f"{_pts(per_game, se * n / max(1, games))} |")
    tot_m, tot_se = _mean_se([p["honest"] for _, p in pr])
    lines += ["", f"All kinds together: honest gain **{_pts(tot_m * len(pr) / max(1, games))}** points per game "
              f"(both sides), {_pts(tot_m, tot_se)} per decision.", ""]

    # Headlines by card: how often the network headlines each card it holds, and its priced value.
    hl = [(r, p) for r, p in pr if r["kind"] == "headline"]
    if hl:
        offered: Dict[Tuple[str, str], int] = collections.Counter()
        chosen: Dict[Tuple[str, str], int] = collections.Counter()
        best: Dict[Tuple[str, str], int] = collections.Counter()
        lead: Dict[Tuple[str, str], List[float]] = collections.defaultdict(list)
        for r, p in hl:
            v = _values(r)
            means = v.mean(axis=1)
            for oi, name in enumerate(r["names"]):
                k = (r["side"], name)
                offered[k] += 1
                chosen[k] += oi == r["taken"]
                best[k] += oi == p["best"]
                lead[k].append(float(means[oi] - means[r["taken"]]))
        lines += ["## Headlines by card", "",
                  "Every card offered as a headline: how often the network headlines it, how often it is the "
                  "priced best, and its mean value against the network's chosen headline (all worlds).", "",
                  "| side | card | offered | headlined | priced best | vs the chosen headline |",
                  "|:---|:---|---:|---:|---:|---:|"]
        order = sorted(offered, key=lambda k: (k[0], -best[k], -offered[k]))
        for k in order:
            if offered[k] < 5:
                continue
            m, se = _mean_se(lead[k])
            lines.append(f"| {k[0]} | {k[1]} | {offered[k]} | {100 * chosen[k] / offered[k]:.0f}% | "
                         f"{100 * best[k] / offered[k]:.0f}% | {_pts(m, se)} |")
        lines.append("")

    if bank:
        conf = {c["id"]: c for c in confirmed}
        lines += ["## The clearest departures, replayed to the end of the game", "",
                  "Each: the network's move against the priced best, both played from the same redeals and dice "
                  "to the end of the game by the network (no critic). Turn-end gain is the pricing's.", "",
                  "| # | side, AR | kind (card) | network | priced best | turn-end gain | full-game gain | position |",
                  "|---:|:---|:---|:---|:---|---:|---:|:---|"]
        held = []
        for i, b in enumerate(bank, 1):
            p = priced(b)
            c = conf.get(b["id"])
            full = f"{100 * c['diff']:+.1f} ± {100 * c['se']:.1f}" if c else "—"
            if c:
                held.append(c["diff"])
            card = f" ({b['card_name']})" if b["card_name"] else ""
            lines.append(f"| {i} | {b['side']}, {'headline' if b['phase'] == 'HEADLINE' else 'AR ' + str(b['ar'])} | "
                         f"{b['kind']}{card} | {b['names'][b['taken']]} | {b['names'][b['bank_best']]} | "
                         f"{_pts(p['diff'], p['se']) if p else '—'} | {full} | [open]({WORKBENCH}?pos={b['pos']}) |")
        if held:
            m, se = _mean_se(held)
            pos = sum(1 for c in confirmed if c["se"] > 0 and c["diff"] / c["se"] >= 2)
            neg = sum(1 for c in confirmed if c["se"] > 0 and c["diff"] / c["se"] <= -2)
            lines += ["", f"Full game, over the {len(held)} replayed: mean **{100 * m:+.2f} ± {100 * se:.2f}** points; "
                      f"{pos} hold at 2+ SE, {neg} reverse at 2+ SE."]
        lines.append("")
    return "\n".join(lines) + "\n"


# --- CLI ------------------------------------------------------------------------------------

def _load_network(path: str, device: torch.device) -> Any:
    agent = load_agent(path, device=str(device))
    if not isinstance(agent, NeuralAgent):
        raise SystemExit(f"{path} is not a network")
    agent.model.eval()
    return agent.model


def _part(spec: Optional[str]) -> Tuple[int, int]:
    if not spec:
        return 0, 1
    i, n = (int(x) for x in spec.split("/"))
    if not 1 <= i <= n:
        raise SystemExit(f"--part {spec}: I must be in 1..N")
    return i - 1, n


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play games and price their turn-1 decisions")
    r.add_argument("--model", required=True)
    r.add_argument("--games", type=int, default=200)
    r.add_argument("--seed", type=int, default=120_000)
    r.add_argument("--opening", default="", help="a forced opening (tools/lib/openings.py; 'us_x+ussr_y'); empty: the network's own")
    r.add_argument("--worlds", type=int, default=64)
    r.add_argument("--max-options", type=int, default=12)
    r.add_argument("--point-cap", type=int, default=60, help="largest option set priced at a first point")
    r.add_argument("--part", default=None, metavar="I/N", help="every N-th game, starting at the I-th")
    r.add_argument("--out", required=True)
    b = sub.add_parser("bank", help="the clearest departures, for confirmation")
    b.add_argument("--rows", nargs="+", required=True)
    b.add_argument("--top", type=int, default=60)
    b.add_argument("--z-min", type=float, default=3.0)
    b.add_argument("--per-kind-card", type=int, default=3)
    b.add_argument("--out", required=True)
    c = sub.add_parser("confirm", help="replay bank rows to the end of the game")
    c.add_argument("--bank", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--pairs", type=int, default=512)
    c.add_argument("--part", default=None, metavar="I/N")
    c.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--rows", nargs="+", required=True)
    p.add_argument("--bank", default=None)
    p.add_argument("--confirm", nargs="*", default=[])
    p.add_argument("--label", default="")
    p.add_argument("--z-min", type=float, default=3.0)
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.cmd == "run":
        model = _load_network(args.model, device)
        k, n = _part(args.part)
        seeds = [args.seed + g for g in range(args.games) if g % n == k]
        t0 = time.time()
        with gzip.open(args.out, "wt") as f:
            for i, seed in enumerate(seeds):
                for row in price_game(model, device, seed, args.opening, args.worlds, args.max_options, args.point_cap):
                    f.write(json.dumps(row) + "\n")
                print(f"  game {i + 1}/{len(seeds)} (seed {seed}) {time.time() - t0:.0f}s", flush=True)
        return 0
    if args.cmd == "bank":
        chosen = select_bank(read_rows(args.rows), args.top, args.z_min, args.per_kind_card)
        with open(args.out, "w") as f:
            for row in chosen:
                f.write(json.dumps(row) + "\n")
        print(f"{len(chosen)} departures -> {args.out}")
        return 0
    if args.cmd == "confirm":
        model = _load_network(args.model, device)
        with open(args.bank) as f:
            to_confirm: List[BankRow] = [json.loads(line) for line in f if line.strip()]
        k, n = _part(args.part)
        mine = [row for i, row in enumerate(to_confirm) if i % n == k]

        def act(obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
            lg, _ = forward(model, obs, masks, device)
            lg = np.where(masks.astype(bool), lg, -np.inf)
            return lg.argmax(axis=1)

        out: List[Confirmed] = []
        for row in mine:
            st = state_from_token(row["pos"])
            a_t, a_b = row["options"][row["taken"]], row["options"][row["bank_best"]]
            seed = int(zlib.crc32(row["id"].encode()))
            (res,) = compare([(st, [a_t, a_b])], act, args.pairs, [seed])
            d, se = paired_diff(res[a_b], res[a_t])
            out.append({"id": row["id"], "taken": row["names"][row["taken"]], "best": row["names"][row["bank_best"]],
                        "pairs": args.pairs, "diff": d, "se": se})
            print(f"  {row['id']}: {100 * d:+.1f} ± {100 * se:.1f}", flush=True)
        with open(args.out, "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        return 0
    rows = read_rows(args.rows)
    bank: List[BankRow] = []
    if args.bank:
        with open(args.bank) as f:
            bank = [json.loads(line) for line in f if line.strip()]
    confirmed: List[Confirmed] = []
    for pth in args.confirm:
        with open(pth) as f:
            confirmed += [json.loads(line) for line in f if line.strip()]
    text = report(rows, confirmed, bank, args.label or "network", args.z_min)
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
