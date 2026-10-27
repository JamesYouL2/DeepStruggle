#!/usr/bin/env python3
"""P35b(a): how diverse is the policy's strategy, next to its searched player's and the humans'?

One record per game, per side, from three sources fed through one tracker
(`tools.lib.corpus_driver`): the network's greedy self-play, a leaderboard player's self-play (a
network under its search) and the ts-replayer human corpus.

* **setup** -- the opening placement;
* **AR1** -- what the side does with its first action round: coup <country>, influence in
  <the set of regions>, realign <country>, space, event, or the scoring card it plays;
* **investment by turn 3** -- the side's own influence added from the end of setup to the start of
  turn 4, per region (a country's gain counted only when positive, so a coup the opponent suffers
  is not "investment"), and the region it put most of it into;
* **coups, realignments and space** -- Ops coups (not an event's free coup) by target, turns 1-3
  and the whole game; realignment rolls; space-race plays.

For each categorical feature, the report gives the entropy (and the effective number of
strategies, 2^H) at every source's own sample and *rarefied to the human sample* -- a plug-in
entropy falls with the sample, so a 16k-game self-play is only compared with 265 human games at
265 -- the distance to the humans' distribution (TVD), and the share of human games whose choice
the source plays in under 1% of its games. Numeric features get mean and spread.

    # greedy self-play, whole games (a shard of them: --shard i/N)
    PYTHONPATH=.:build/release python tools/scripts/strategy_diversity.py play \\
        --checkpoint <net.pt> --games 16384 --out greedy.jsonl
    # a leaderboard player under its search, turns 1-3 only (later turns take first-legal moves)
    PYTHONPATH=.:build/release python tools/scripts/strategy_diversity.py play \\
        --player '<player id>' --games 2048 --max-turn 3 --batch 64 --out searched.jsonl
    # the human corpus
    PYTHONPATH=.:build/release python tools/scripts/strategy_diversity.py humans --out humans.jsonl
    # the comparison
    python tools/scripts/strategy_diversity.py report greedy=greedy.jsonl searched=searched.jsonl \\
        humans=humans.jsonl --reference humans --output-md diversity.md
"""

from __future__ import annotations

import argparse
import collections
import glob
import math
import os
import sys
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import torch
import ts_engine as ts

from bindings.action_encoder import ActionEncoder
from tools.lib.corpus_driver import (corpus_map, feed_corpus_game, load_policy, read_corpus_game, require_e4_view,
                                     rules_json, selfplay)
from tools.lib.data_root import data_path
from tools.lib.jsonl import read_jsonl, sha256_file, write_jsonl
from tools.lib.leaderboard import load as load_leaderboard, player_spec
from tools.lib.player_agent import load_agent
from tools.lib.player_spec import Policy, load_spec

NODE = ActionEncoder.NODE_OFFSET
PLAY_MODE = ActionEncoder.PLAY_MODE_OFFSET
SPACE_MODE = 1
#: The engine's `Region` order.
REGIONS = ["Europe", "Asia", "Middle East", "Africa", "Central America", "South America"]
_COUNTRIES = rules_json("map.json")["countries"]
COUNTRY_REGION = {int(c["id"]): REGIONS.index(c["region"]) for c in _COUNTRIES}
COUNTRY_NAME = {int(c["id"]): str(c["name"]) for c in _COUNTRIES}
CARD_NAME = {int(c["id"]): str(c["name"]) for c in rules_json("cards.json")}
BATTLEGROUND = {int(c["id"]) for c in _COUNTRIES if c["battleground"]}
SIDES = ("US", "USSR")
OP_COUP, OP_REALIGN, OP_INFLUENCE = int(ts.OpMode.COUP), int(ts.OpMode.REALIGN), int(ts.OpMode.INFLUENCE)
#: Turns 1-3: the opening, before the Mid War deck changes what is possible.
OPENING_TURNS = 3
#: A choice the source plays in fewer than this share of its games counts as one it does not play.
RARE = 0.01
RECORD_VERSION = 1


def _side(player: Any) -> str:
    return "US" if player == ts.Player.US else "USSR"


def _influence(st: ts.GameState) -> Dict[str, List[int]]:
    us, ussr = [0] * 84, [0] * 84
    for c in COUNTRY_REGION:
        country = st.get_country(c)
        us[c], ussr[c] = int(country.us_influence), int(country.ussr_influence)
    return {"US": us, "USSR": ussr}


class GameTrace:
    """One game's strategy features, from its decisions in order (`observe(state, action)`, before
    the action is applied). Decisions after `max_turn` are not recorded; the first of turn 4 still
    gives the investment snapshot."""

    def __init__(self, max_turn: int = 99) -> None:
        self.max_turn = max_turn
        self.setup: Dict[str, Dict[int, int]] = {s: collections.Counter() for s in SIDES}
        self.start: Optional[Dict[str, List[int]]] = None
        self.turn4: Optional[Dict[str, List[int]]] = None
        #: per side, its first action round of turn 1: play modes and Ops targets
        self.ar1: Dict[str, Dict[str, List[int]]] = {s: {"card": [], "mode": [], "coup": [], "realign": [],
                                                         "inf": []} for s in SIDES}
        self.coups: List[List[Any]] = []        # [turn, side, country]
        self.realigns: List[List[Any]] = []     # [turn, side, country]
        self.space: List[List[Any]] = []        # [turn, side]
        self.last_turn = 0

    def observe(self, st: ts.GameState, a: int) -> None:
        turn = int(st.turn)
        phase = st.current_phase
        if phase != ts.Phase.SETUP and self.start is None:
            self.start = _influence(st)
        if turn > OPENING_TURNS and self.turn4 is None:
            self.turn4 = _influence(st)
        if turn > self.max_turn:
            return
        self.last_turn = max(self.last_turn, turn)
        ctx = st.ctx()
        side = _side(ctx.decision_player)
        dt = ctx.decision_type
        own = int(ctx.resolving_card) == 0
        if phase == ts.Phase.SETUP:
            if dt == ts.DecisionType.POINT_NODE and own:
                self.setup[side][a - NODE] += 1
            return
        if phase != ts.Phase.ACTION_ROUND:
            return
        first = turn == 1 and int(st.action_round) == 1
        if dt == ts.DecisionType.SELECT_CARD and own and first:
            self.ar1[side]["card"].append(a + 1)
        elif dt == ts.DecisionType.SELECT_PLAY_MODE:
            mode = a - PLAY_MODE
            if mode == SPACE_MODE:
                self.space.append([turn, side])
            if first:
                self.ar1[side]["mode"].append(mode)
        elif dt == ts.DecisionType.POINT_NODE and own and NODE <= a < NODE + 84:
            cid, op = a - NODE, int(ctx.op_mode)
            if op == OP_COUP:
                self.coups.append([turn, side, cid])
                if first:
                    self.ar1[side]["coup"].append(cid)
            elif op == OP_REALIGN:
                self.realigns.append([turn, side, cid])
                if first:
                    self.ar1[side]["realign"].append(cid)
            elif op == OP_INFLUENCE and first:
                self.ar1[side]["inf"].append(cid)

    def _ar1_token(self, side: str) -> Optional[str]:
        d = self.ar1[side]
        if d["coup"]:
            return f"coup {COUNTRY_NAME[d['coup'][0]]}"
        if d["inf"]:
            return "influence " + "+".join(_ranked_regions(d["inf"]))
        if d["realign"]:
            return f"realign {COUNTRY_NAME[d['realign'][0]]}"
        if SPACE_MODE in d["mode"]:
            return "space"
        if d["mode"]:
            return "event"
        if d["card"]:   # a scoring card: no mode to choose
            return CARD_NAME[d["card"][0]]
        return None

    def record(self, game: int, ended: bool) -> Dict[str, Any]:
        gain: Optional[Dict[str, List[int]]] = None
        if self.start is not None and self.turn4 is not None:
            gain = {}
            for s in SIDES:
                g = [0] * len(REGIONS)
                for c, r in COUNTRY_REGION.items():
                    g[r] += max(0, self.turn4[s][c] - self.start[s][c])
                gain[s] = g
        return {"game": game, "version": RECORD_VERSION, "ended": ended, "last_turn": self.last_turn,
                "max_turn": self.max_turn, "opening_complete": self.turn4 is not None,
                "setup": {s: {COUNTRY_NAME[c]: n for c, n in sorted(self.setup[s].items())} for s in SIDES},
                "ar1": {s: self._ar1_token(s) for s in SIDES}, "gain": gain,
                "coups": self.coups, "realigns": self.realigns, "space": self.space}


def _ranked_regions(countries: Sequence[int]) -> List[str]:
    """The regions the countries lie in, in the engine's order -- a set, so one plan placed in
    another order is the same choice."""
    return [REGIONS[r] for r in sorted({COUNTRY_REGION[c] for c in countries})]


# --------------------------------------------------------------------------------------------
# Sources
# --------------------------------------------------------------------------------------------

def _network(pid: str) -> Dict[str, Any]:
    board = load_leaderboard()
    if pid not in board.players:
        raise SystemExit(f"{pid!r} is not a registered player")
    pl = board.players[pid]
    if pl["network"] is None:
        raise SystemExit(f"{pid!r} is a bot, not a network")
    return dict(board.networks[str(pl["network"])], name=str(pl["network"]))


def fetch_network(pid: str, hf_repo: str) -> str:
    """Download a player's registered network from Hugging Face to where the registry expects it."""
    net = _network(pid)
    path = data_path(net["file"])
    if os.path.exists(path) and sha256_file(path) == net["sha256"]:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    url = f"https://huggingface.co/{hf_repo}/resolve/main/{net['hf']}"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, path)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(10)
    if sha256_file(path) != net["sha256"]:
        raise SystemExit(f"{url}: not the file registered for {net['name']} (sha256 differs)")
    return path


def resolve_player(pid: str, greedy: bool = False) -> Tuple[str, Optional[str]]:
    """A leaderboard player's network file (checked against its registered sha256) and the
    `load_agent` spec that plays it -- None for a greedy policy, or with `greedy`, its network
    played greedily."""
    board = load_leaderboard()
    net = _network(pid)
    path = data_path(net["file"])
    if sha256_file(path) != net["sha256"]:
        raise SystemExit(f"{path} is not the file registered for {net['name']} (sha256 differs)")
    spec = player_spec(board, pid)
    if greedy or (isinstance(spec, Policy) and spec.temperature <= 0.05):
        return path, None
    return path, load_spec(spec, path)


def _cutoff_states_fn(fn: Callable[[List[ts.GameState]], List[int]],
                      max_turn: int) -> Callable[[List[ts.GameState]], List[int]]:
    """The searcher on positions up to `max_turn`; past it, each game's first legal move -- the
    game only has to reach its end, nothing after the cutoff is recorded."""
    def pick(states: List[ts.GameState]) -> List[int]:
        out = [0] * len(states)
        search = [i for i, st in enumerate(states) if int(st.turn) <= max_turn]
        for i, st in enumerate(states):
            if i not in search:
                out[i] = int(np.argmax(ActionEncoder.get_legal_mask(st)))
        if search:
            for i, act in zip(search, fn([states[i] for i in search])):
                out[i] = int(act)
        return out
    return pick


def _shard_batches(games: int, batch: int, shard: Tuple[int, int]) -> List[int]:
    i, n = shard
    return [b0 for k, b0 in enumerate(range(0, games, batch)) if k % n == i]


def play_source(checkpoint: Optional[str], player: Optional[str], greedy: bool, games: int, seed: int, batch: int,
                max_turn: int, shard: Tuple[int, int]) -> List[Dict[str, Any]]:
    """Self-play records of this shard's batches. Batch b0 is seeded `seed + b0` (and a searcher is
    reseeded with it), so shards of one run are the games one process would play."""
    agent_spec: Optional[str] = None
    if player is not None:
        checkpoint, agent_spec = resolve_player(player, greedy)
    assert checkpoint is not None
    logits_fn, features = None, 0
    states_fn = None
    agent: Any = None
    if agent_spec is None:
        if max_turn < 99:
            raise SystemExit("--max-turn is for a searched player; greedy self-play is cheap to the end")
        logits_fn, features = load_policy(checkpoint)
    else:
        agent = cast(Any, load_agent(agent_spec, device="cuda" if torch.cuda.is_available() else "cpu"))
        require_e4_view(agent_spec, agent)
        states_fn = _cutoff_states_fn(agent.select_actions_batch, max_turn)
    out: List[Dict[str, Any]] = []
    for b0 in _shard_batches(games, batch, shard):
        n = min(batch, games - b0)
        if agent is not None and hasattr(agent, "reseed"):
            agent.reseed(seed + b0)
        traces = selfplay(logits_fn, features, n, seed + b0, n, 0.0, lambda: GameTrace(max_turn),
                          states_fn=states_fn)
        out += [t.record(b0 + i, ended=max_turn >= 99) for i, t in enumerate(traces)]
        print(f"  batch {b0:,}-{b0 + n - 1:,} done", file=sys.stderr, flush=True)
    return out


def _human_game(path: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """One corpus game's record and its status. The turn a record stops in never reaches the
    tracker (`untrusted_turn`), so every turn recorded is complete."""
    replay_id = int(read_corpus_game(path).get("replay_id", 0))
    trace = GameTrace()
    status = feed_corpus_game(path, trace)
    if status == "skipped":
        return None, status
    rec = trace.record(replay_id, ended=status == "complete")
    return rec, status


def human_records(workers: int, limit: int) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    status: Dict[str, int] = collections.Counter()
    out: List[Dict[str, Any]] = []
    for rec, st in corpus_map(_human_game, workers, limit):
        status[st] += 1
        if rec is not None:
            out.append(rec)
    out.sort(key=lambda r: int(r["game"]))
    return out, dict(status)


# --------------------------------------------------------------------------------------------
# Features and metrics
# --------------------------------------------------------------------------------------------

def _setup_token(rec: Dict[str, Any], side: str) -> Optional[str]:
    s = rec["setup"][side]
    if not s:
        return None
    return ", ".join(f"{c} {n}" for c, n in sorted(s.items(), key=lambda kv: (-kv[1], kv[0])))


def _top_region(rec: Dict[str, Any], side: str) -> Optional[str]:
    if rec["gain"] is None:
        return None
    g = rec["gain"][side]
    return REGIONS[int(np.argmax(g))] if sum(g) else "none"


def _top2_regions(rec: Dict[str, Any], side: str) -> Optional[str]:
    if rec["gain"] is None:
        return None
    g = rec["gain"][side]
    order = [r for r in sorted(range(len(REGIONS)), key=lambda r: (-g[r], r)) if g[r] > 0][:2]
    return " + ".join(REGIONS[r] for r in sorted(order)) or "none"


def _opening_coups(rec: Dict[str, Any], side: str) -> Optional[str]:
    """The ordered countries the side coups in turns 1-3 ("none": it does not)."""
    if not rec["opening_complete"]:
        return None
    cs = [COUNTRY_NAME[c] for t, s, c in rec["coups"] if s == side and t <= OPENING_TURNS]
    return " > ".join(cs) or "none"


#: name -> (side-free title, token of one game for one side)
CATEGORICAL: Dict[str, Callable[[Dict[str, Any], str], Optional[str]]] = {
    "setup": _setup_token,
    "AR1": lambda r, s: r["ar1"][s],
    "main region, T1-3": _top_region,
    "top two regions (as a set), T1-3": _top2_regions,
    "coup sequence, T1-3": _opening_coups,
}


def _count(rec: Dict[str, Any], key: str, side: str, upto: int) -> int:
    return sum(1 for e in rec[key] if e[1] == side and e[0] <= upto)


def _bg_coups(rec: Dict[str, Any], side: str, upto: int) -> int:
    return sum(1 for t, s, c in rec["coups"] if s == side and t <= upto and c in BATTLEGROUND)


#: name -> (needs the whole game, value of one game for one side)
NUMERIC: Dict[str, Tuple[bool, Callable[[Dict[str, Any], str], float]]] = {
    "coups, T1-3": (False, lambda r, s: _count(r, "coups", s, OPENING_TURNS)),
    "battleground coups, T1-3": (False, lambda r, s: _bg_coups(r, s, OPENING_TURNS)),
    "realignment rolls, T1-3": (False, lambda r, s: _count(r, "realigns", s, OPENING_TURNS)),
    "space plays, T1-3": (False, lambda r, s: _count(r, "space", s, OPENING_TURNS)),
    "influence added, T1-3": (False, lambda r, s: float(sum(r["gain"][s]))),
    "coups, game": (True, lambda r, s: _count(r, "coups", s, 99)),
    "realignment rolls, game": (True, lambda r, s: _count(r, "realigns", s, 99)),
    "space plays, game": (True, lambda r, s: _count(r, "space", s, 99)),
}


def _usable(rec: Dict[str, Any], whole: bool) -> bool:
    return bool(rec["ended"]) if whole else bool(rec["opening_complete"])


def entropy(counts: Sequence[int]) -> float:
    n = float(sum(counts))
    return -sum(k / n * math.log2(k / n) for k in counts if k) if n else 0.0


def rarefied_entropy(tokens: Sequence[str], n: int, reps: int = 200, seed: int = 0) -> float:
    """Mean plug-in entropy of `n` games drawn without replacement -- what a sample the size of
    the reference's would show."""
    if len(tokens) <= n:
        return entropy(list(collections.Counter(tokens).values()))
    rng = np.random.default_rng(seed)
    arr = np.asarray(tokens, dtype=object)
    return float(np.mean([entropy(list(collections.Counter(arr[rng.choice(len(arr), n, replace=False)]).values()))
                          for _ in range(reps)]))


def tvd(p: Dict[str, float], q: Dict[str, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in set(p) | set(q))


def _dist(tokens: Sequence[str]) -> Dict[str, float]:
    n = len(tokens)
    return {k: v / n for k, v in collections.Counter(tokens).items()} if n else {}


def load_records(path: str) -> List[Dict[str, Any]]:
    return list(read_jsonl(sorted(glob.glob(path)) or [path]))


def _tokens(recs: Sequence[Dict[str, Any]], fn: Callable[[Dict[str, Any], str], Optional[str]],
            side: str) -> List[str]:
    return [t for t in (fn(r, side) for r in recs) if t is not None]


def report(sources: Dict[str, List[Dict[str, Any]]], reference: str, top: int = 8) -> str:
    ref = sources[reference]
    lines = ["# Strategy diversity: self-play, searched player and humans (P35b a)", ""]
    lines += ["| source | games | whole games | opening (T1-3) complete |", "|:---|---:|---:|---:|"]
    for name, recs in sources.items():
        lines.append(f"| {name} | {len(recs):,} | {sum(1 for r in recs if r['ended']):,} | "
                     f"{sum(1 for r in recs if r['opening_complete']):,} |")
    lines += ["", f"Entropy in bits; *eff.* = 2^H, the number of equally likely choices it amounts to. "
              f"*H@{reference}* is the entropy of a sample the size of {reference}'s (mean of 200 draws "
              f"without replacement). *{reference} unplayed* is the share of {reference}'s games whose choice "
              f"this source makes in under {RARE:.0%} of its games; *TVD* the total variation distance to "
              f"{reference}.", ""]
    for feat, fn in CATEGORICAL.items():
        for side in SIDES:
            ref_tokens = _tokens(ref, fn, side)
            ref_d = _dist(ref_tokens)
            lines += [f"## {feat} -- {side}", "",
                      f"| source | n | distinct | H | eff. | H@{reference} | eff.@{reference} | top share | "
                      f"TVD | {reference} unplayed |",
                      "|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
            dists: Dict[str, Dict[str, float]] = {}
            for name, recs in sources.items():
                toks = _tokens(recs, fn, side)
                if not toks:
                    continue
                d = _dist(toks)
                dists[name] = d
                h = entropy(list(collections.Counter(toks).values()))
                hr = rarefied_entropy(toks, len(ref_tokens)) if ref_tokens else float("nan")
                unplayed = sum(p for k, p in ref_d.items() if d.get(k, 0.0) < RARE)
                lines.append(f"| {name} | {len(toks):,} | {len(d):,} | {h:.2f} | {2 ** h:.1f} | {hr:.2f} | "
                             f"{2 ** hr:.1f} | {max(d.values()):.1%} | {tvd(d, ref_d):.2f} | {unplayed:.1%} |")
            union: Dict[str, float] = {}
            for d in dists.values():
                for k, p in d.items():
                    union[k] = max(union.get(k, 0.0), p)
            keys = [k for k, _ in sorted(union.items(), key=lambda kv: -kv[1])[:top]]
            lines += ["", "| choice | " + " | ".join(dists) + " |", "|:---|" + "---:|" * len(dists)]
            for k in keys:
                lines.append(f"| {k} | " + " | ".join(f"{d.get(k, 0.0):.1%}" for d in dists.values()) + " |")
            lines.append("")
    lines += ["## Counts per game (mean ± SD across games)", "",
              "Opening counts over games whose turns 1-3 are complete; whole-game counts over games played "
              "to the end (a searched player cut at turn 3 has none).", "",
              "| feature | side | " + " | ".join(sources) + " |", "|:---|:---|" + "---:|" * len(sources)]
    for feat, (whole, fn) in NUMERIC.items():
        for side in SIDES:
            cells = []
            for recs in sources.values():
                v = [float(fn(r, side)) for r in recs if _usable(r, whole) and (whole or r["gain"] is not None)]
                cells.append(f"{np.mean(v):.2f} ± {np.std(v):.2f}" if v else "--")
            lines.append(f"| {feat} | {side} | " + " | ".join(cells) + " |")
    lines += ["", "## Influence added in turns 1-3, share by region (mean ± SD across games)", "",
              "| side | region | " + " | ".join(sources) + " |", "|:---|:---|" + "---:|" * len(sources)]
    for side in SIDES:
        for r, region in enumerate(REGIONS):
            cells = []
            for recs in sources.values():
                v = [g[side][r] / sum(g[side]) for g in (x["gain"] for x in recs if x["gain"] is not None)
                     if sum(g[side])]
                cells.append(f"{np.mean(v):.1%} ± {np.std(v):.1%}" if v else "--")
            lines.append(f"| {side} | {region} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------

def _parse_shard(s: str) -> Tuple[int, int]:
    i, n = (int(x) for x in s.split("/"))
    if not 0 <= i < n:
        raise argparse.ArgumentTypeError(f"--shard {s}: want i/N with 0 <= i < N")
    return i, n


def _write(path: str, recs: Sequence[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_jsonl(path, recs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="self-play records of a network, greedy or under a leaderboard player's search")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="a .pt checkpoint or .onnx export, played greedily")
    src.add_argument("--player", help="a leaderboard player id: its registered network under its search")
    p.add_argument("--greedy", action="store_true", help="with --player, its network played greedily")
    p.add_argument("--games", type=int, default=16384)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--seed", type=int, default=55_000)
    p.add_argument("--max-turn", type=int, default=99,
                   help="record (and search) only turns up to this one; later turns take first-legal moves")
    p.add_argument("--shard", type=_parse_shard, default=(0, 1), help="i/N: play every N-th batch from the i-th")
    p.add_argument("--out", required=True)
    f = sub.add_parser("fetch", help="download a player's registered network from Hugging Face")
    f.add_argument("--player", required=True)
    f.add_argument("--hf-repo", default="mihaild/deepstruggle")
    h = sub.add_parser("humans", help="records of the ts-replayer corpus's games")
    h.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    h.add_argument("--limit", type=int, default=0, help="at most this many games (0: all)")
    h.add_argument("--out", required=True)
    r = sub.add_parser("report", help="compare sources")
    r.add_argument("sources", nargs="+", help="name=path (a path may be a glob, for shards)")
    r.add_argument("--reference", default="humans", help="the source every other is compared with")
    r.add_argument("--output-md", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "play":
        if a.greedy and not a.player:
            ap.error("--greedy goes with --player (a --checkpoint is played greedily already)")
        recs = play_source(a.checkpoint, a.player, a.greedy, a.games, a.seed, a.batch, a.max_turn, a.shard)
        _write(a.out, recs)
        print(f"{len(recs):,} games -> {a.out}", file=sys.stderr)
    elif a.cmd == "fetch":
        print(fetch_network(a.player, a.hf_repo))
    elif a.cmd == "humans":
        recs, status = human_records(a.workers, a.limit)
        _write(a.out, recs)
        print(f"corpus: {status}; {len(recs):,} games -> {a.out}", file=sys.stderr)
    else:
        sources: Dict[str, List[Dict[str, Any]]] = {}
        for s in a.sources:
            name, _, path = s.partition("=")
            sources[name] = load_records(path)
        if a.reference not in sources:
            ap.error(f"--reference {a.reference!r} is not one of the sources")
        md = report(sources, a.reference)
        if a.output_md:
            open(a.output_md, "w").write(md)
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
