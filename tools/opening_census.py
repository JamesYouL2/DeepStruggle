"""The first turns as the network plays them from a forced opening: every card and play-mode
choice, and the board at the start of every action round, for each game.

Greedy self-play of one network (`tools.lib.game_loop.GameLoop`), the setup scripted by an
opening (`tools/lib/openings.py`), stopped when turn `--turns` ends. A record per game:

* `plays`: each card choice (headline or action round) and each event-or-Ops / Ops-mode choice,
  with turn, action round, side, the card being resolved (UN Intervention, for a card picked
  inside its event) and the option taken;
* `snaps`: at the first decision of each action round (and once after the last turn), the US and
  USSR influence of every country, the card locations, VP, DEFCON and both space tracks.

Games of the same seed deal the same cards under every opening, so two runs on one seed range
compare deal by deal.

    python tools/opening_census.py run --model <ckpt.pt> --opening us_r34+ussr_r34 --games 5000 \\
        --turns 3 --part 1/20 --out part.jsonl.gz
    python tools/opening_census.py report --games a.jsonl.gz --games b.jsonl.gz --out report.md
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import os
import sys
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, TypedDict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import ts_engine as ts  # noqa: E402

from ai.eval.paired_playouts import decider  # noqa: E402
from ai.search.turn_pricing import forward  # noqa: E402
from bindings.action_encoder import ActionEncoder  # noqa: E402
from bindings.ts_env import model_obs_features  # noqa: E402
from tools.lib.game_loop import GameLoop, SettlePolicy  # noqa: E402
from tools.lib.openings import acting_side, scripted_setup_index  # noqa: E402
from tools.turn1_census import option_label  # noqa: E402
from tools.turn1_pricing import _load_network, _part, kind_of  # noqa: E402

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RECORDED_KINDS = ("headline", "card to play", "event or Ops", "Ops mode")


class Play(TypedDict):
    turn: int
    ar: int
    side: str
    kind: str
    resolving: str        # the card whose event is asking (UN Intervention), "" otherwise
    card: str             # the card in play for a mode choice, "" for a card choice
    taken: str


class Snap(TypedDict):
    turn: int
    ar: int
    phase: str
    us: List[int]         # influence by country id 0..83
    ussr: List[int]
    loc: List[int]        # int(CardLocation) of cards 1..110
    vp: int
    defcon: int
    space: List[int]      # US, USSR


class GameRecord(TypedDict):
    seed: int
    opening: str
    turns: int
    plays: List[Play]
    snaps: List[Snap]


def snap(st: ts.GameState) -> Snap:
    return {"turn": int(st.turn), "ar": int(st.action_round), "phase": str(st.current_phase).replace("Phase.", ""),
            "us": [int(st.get_country(c).us_influence) for c in range(84)],
            "ussr": [int(st.get_country(c).ussr_influence) for c in range(84)],
            "loc": [int(st.get_card_location(c)) for c in range(1, 111)],
            "vp": int(st.victory_points), "defcon": int(st.defcon),
            "space": [int(st.us_space_track), int(st.ussr_space_track)]}


def _card_name(card: int) -> str:
    return str(ts.CardData.get_card_info(card)["name"]) if card else ""


class _Source:
    """Greedy, with the opening's setup forced; records plays and a snapshot per action round."""

    def __init__(self, model: Any, device: torch.device, opening: str) -> None:
        self.model, self.device, self.opening = model, device, opening
        self.feats = model_obs_features(model)
        self.cursor = {"US": 0, "USSR": 0}
        self.plays: List[Play] = []
        self.snaps: List[Snap] = []
        self.key: Optional[Tuple[int, int, str]] = None

    def choose(self, st: ts.GameState, /) -> Optional[ts.MicroAction]:
        if self.opening:
            idx = scripted_setup_index(st, acting_side(st), self.opening, self.cursor)
            if idx is not None:
                return ts.ActionMask.decode_flat_action(st, idx)
        if st.current_phase != ts.Phase.SETUP:
            key = (int(st.turn), int(st.action_round), str(st.current_phase))
            if key != self.key:
                self.key = key
                self.snaps.append(snap(st))
        mover = decider(st)
        mask = np.asarray(ActionEncoder.get_legal_mask(st), dtype=np.uint8)
        obs = np.asarray(ts.extract_observation_features(st, mover, self.feats), dtype=np.float32)
        lg, _ = forward(self.model, obs[None], mask[None], self.device)
        legal = np.flatnonzero(mask)
        act = int(legal[np.argmax(lg[0][legal])])
        if st.current_phase != ts.Phase.SETUP:
            kind = kind_of(st)
            if kind in RECORDED_KINDS:
                ctx = st.ctx()
                resolving = int(ctx.resolving_card or 0)
                self.plays.append({
                    "turn": int(st.turn), "ar": int(st.action_round),
                    "side": "US" if mover == ts.Player.US else "USSR", "kind": kind,
                    "resolving": _card_name(resolving) if kind in ("headline", "card to play") else "",
                    "card": _card_name(int(ctx.pending_op_card or 0) or resolving) if kind not in ("headline", "card to play") else "",
                    "taken": option_label(ActionEncoder.get_action_name(st, act))})
        return ts.ActionMask.decode_flat_action(st, act)


def play_game(model: Any, device: torch.device, seed: int, opening: str, turns: int) -> GameRecord:
    st = ts.GameState()
    ts.Engine.init_game(st, int(seed))
    src = _Source(model, device, opening)
    loop = GameLoop(st, {ts.Player.US: src, ts.Player.USSR: src}, settle=SettlePolicy.RECORD_FORCED,
                    stop=lambda s: int(s.turn) > turns)
    loop.run()
    src.snaps.append(snap(loop.state))
    return {"seed": int(seed), "opening": opening, "turns": turns, "plays": src.plays, "snaps": src.snaps}


def read_games(paths: Iterable[str]) -> List[GameRecord]:
    out: List[GameRecord] = []
    for p in paths:
        with gzip.open(p, "rt") as f:
            out += [json.loads(line) for line in f if line.strip()]
    return out


# --- report ---------------------------------------------------------------------------------

def _map() -> Tuple[Dict[int, Dict[str, Any]], Dict[str, int]]:
    countries = {int(c["id"]): c for c in json.load(open(os.path.join(_ROOT, "rules", "map.json")))["countries"]}
    return countries, {str(c["name"]): i for i, c in countries.items()}


def _pct(k: float, n: int) -> str:
    return f"{100 * k / n:.1f}%" if n else "—"


def turn_end_snaps(g: GameRecord) -> Dict[int, Snap]:
    """The board as each turn ends: the first snapshot of the next turn (the last one is the
    snapshot taken after the loop stopped)."""
    out: Dict[int, Snap] = {}
    for s in g["snaps"]:
        if s["turn"] >= 2 and (s["turn"] - 1) not in out:
            out[s["turn"] - 1] = s
    return out


def europe_section(games: Sequence[GameRecord], label: str) -> List[str]:
    countries, ids = _map()
    eg, pl = ids["East Germany"], ids["Poland"]
    turns = max(g["turns"] for g in games)
    lines = [f"### {label} ({len(games):,} games)", "",
             "USSR influence at the end of each turn (mean), its margin over control (USSR − US − "
             "stability; ≥ 0 is control, every point above 0 is protection a coup or East European "
             "Unrest has to remove), and the USSR influence added there during the turn (mean).", "",
             "| turn | East Germany | margin | added | Poland | margin | added | both margins ≥ 2 |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for t in range(1, turns + 1):
        rows = []
        for g in games:
            ends = turn_end_snaps(g)
            if t not in ends:
                continue
            e = ends[t]
            start = next((s for s in g["snaps"] if s["turn"] == t), None)
            prev = ends.get(t - 1, start)
            if prev is None:
                continue
            rows.append([e["ussr"][eg], e["ussr"][eg] - e["us"][eg] - int(countries[eg]["stability"]),
                         max(0, e["ussr"][eg] - prev["ussr"][eg]),
                         e["ussr"][pl], e["ussr"][pl] - e["us"][pl] - int(countries[pl]["stability"]),
                         max(0, e["ussr"][pl] - prev["ussr"][pl])])
        if not rows:
            continue
        a = np.array(rows, dtype=float)
        both2 = float(((a[:, 1] >= 2) & (a[:, 4] >= 2)).mean())
        lines.append(f"| {t} | {a[:, 0].mean():.2f} | {a[:, 1].mean():+.2f} | {a[:, 2].mean():.2f} | "
                     f"{a[:, 3].mean():.2f} | {a[:, 4].mean():+.2f} | {a[:, 5].mean():.2f} | {100 * both2:.1f}% |")
    lost = collections.Counter()
    for g in games:
        for t, e in turn_end_snaps(g).items():
            for c, nm in ((eg, "East Germany"), (pl, "Poland")):
                if e["ussr"][c] - e["us"][c] < int(countries[c]["stability"]):
                    lost[(t, nm)] += 1
    lines += ["", "USSR does not control it at the end of the turn: " + "; ".join(
        f"turn {t} {nm} {_pct(k, len(games))}" for (t, nm), k in sorted(lost.items()))]
    return lines


def card_section(games: Sequence[GameRecord], card: str, side: str) -> List[str]:
    """When `side` gets `card` and when and how it plays it, over the turns recorded."""
    how: collections.Counter = collections.Counter()
    when: collections.Counter = collections.Counter()
    n = 0
    for g in games:
        ps = g["plays"]
        chosen = [i for i, p in enumerate(ps) if p["side"] == side and p["kind"] in ("headline", "card to play") and p["taken"] == card]
        if not chosen:
            continue
        n += 1
        i = chosen[0]
        p = ps[i]
        if p["kind"] == "headline":
            h = "headline"
        elif p["resolving"] == "UN Intervention":
            h = "Ops via UN Intervention (no event)"
        else:
            m = next((q for q in ps[i + 1:i + 4] if q["side"] == side and q["kind"] == "event or Ops"), None)
            h = m["taken"] if m else "no choice"
        how[h] += 1
        when[(p["turn"], p["ar"])] += 1
    lines = [f"**{side} plays {card}** ({n:,} games where it does, within the turns recorded):", "",
             "| how | share |", "|:---|---:|"]
    lines += [f"| {h} | {_pct(k, n)} |" for h, k in how.most_common()]
    lines += ["", "When (turn, AR; AR 0 = headline): " + ", ".join(f"T{t} AR{a} {_pct(k, n)}" for (t, a), k in sorted(when.items()) if k / max(n, 1) >= 0.02)]
    return lines


USSR_HAND = (4, 5)          # int(CardLocation.HAND_USSR_UNKNOWN), HAND_USSR_KNOWN


def _fires(ps: Sequence[Play], i: int, side: str) -> bool:
    """Whether the card chosen at plays[i] is played so that its (opponent's) event fires: not to
    Space, not inside UN Intervention, not a headline."""
    p = ps[i]
    if p["kind"] == "headline" or p["resolving"] == "UN Intervention":
        return False
    m = next((q for q in ps[i + 1:i + 4] if q["side"] == side and q["kind"] == "event or Ops"), None)
    return m is None or m["taken"] != "SPACE"


def order_section(games: Sequence[GameRecord], card: str, side: str = "USSR", min_n: int = 60) -> List[str]:
    """In turns where `side` fires `card`'s event while holding card X at the turn's start: the share
    where X was played earlier that turn. A card the side waits for stands out from the rest."""
    names = {c: _card_name(c) for c in range(1, 111)}
    before: collections.Counter = collections.Counter()
    seen: collections.Counter = collections.Counter()
    for g in games:
        ps = g["plays"]
        starts = {}
        for sn in g["snaps"]:
            starts.setdefault(sn["turn"], sn)
        for i, p in enumerate(ps):
            if not (p["side"] == side and p["kind"] == "card to play" and p["taken"] == card and _fires(ps, i, side)):
                continue
            sn = starts.get(p["turn"])
            if sn is None:
                continue
            played = [q["taken"] for q in ps[:i] if q["side"] == side and q["turn"] == p["turn"] and q["kind"] in ("headline", "card to play")]
            for c in range(1, 111):
                if names[c] == card or sn["loc"][c - 1] not in USSR_HAND:
                    continue
                seen[names[c]] += 1
                before[names[c]] += names[c] in played
    tot_b, tot_n = sum(before.values()), sum(seen.values())
    rows = sorted(((before[c] / n, c, n) for c, n in seen.items() if n >= min_n), reverse=True)
    lines = [f"**{side} fires {card}'s event while holding card X at the turn's start: X already played that turn** "
             f"(every card held: {_pct(tot_b, tot_n)} of {tot_n:,})", "",
             "| card held | turns | played before it |", "|:---|---:|---:|"]
    lines += [f"| {c} | {n:,} | {100 * b:.0f}% |" for b, c, n in rows]
    return lines


def report(groups: Sequence[Tuple[str, Sequence[GameRecord]]]) -> List[str]:
    lines = ["# Opening census", "", "## East Germany and Poland", ""]
    for label, games in groups:
        lines += europe_section(games, label) + [""]
    lines += ["## East European Unrest and Independent Reds", ""]
    for label, games in groups:
        lines += [f"### {label}", ""]
        for side in ("USSR", "US"):
            for card in ("East European Unrest", "Independent Reds"):
                lines += card_section(games, card, side) + [""]
        for card in ("East European Unrest", "Independent Reds"):
            lines += order_section(games, card) + [""]
    return lines


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="play games and record them")
    r.add_argument("--model", required=True)
    r.add_argument("--opening", required=True, help="a two-sided opening, 'us_x+ussr_y'")
    r.add_argument("--games", type=int, default=1000)
    r.add_argument("--seed", type=int, default=200_000)
    r.add_argument("--turns", type=int, default=3)
    r.add_argument("--part", default=None, metavar="I/N")
    r.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--games", nargs="+", action="append", required=True,
                   help="one group of game files per flag; the group is labelled by its opening")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "run":
        device = torch.device("cpu")
        model = _load_network(args.model, device)
        k, n = _part(args.part)
        seeds = [args.seed + g for g in range(args.games) if g % n == k]
        t0 = time.time()
        with gzip.open(args.out, "wt") as f:
            for i, seed in enumerate(seeds):
                f.write(json.dumps(play_game(model, device, seed, args.opening, args.turns)) + "\n")
                if (i + 1) % 25 == 0 or i + 1 == len(seeds):
                    print(f"  game {i + 1}/{len(seeds)} {time.time() - t0:.0f}s", flush=True)
        return 0
    groups: List[Tuple[str, Sequence[GameRecord]]] = []
    for paths in args.games:
        games = read_games(paths)
        groups.append((games[0]["opening"] if games else "(empty)", games))
    text = "\n".join(report(groups)) + "\n"
    with open(args.out, "w") as f:
        f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
