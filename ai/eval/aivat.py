"""AIVAT-style evaluation: a tournament's games scored with the luck of the dice and the deal taken out.

AIVAT (Burch, Schmid, Moravčík, Morrill and Bowling, AAAI 2018; research/method/references.md §3)
subtracts, at every chance event of a game, how far the chance moved a value estimate:
v(after the event) - E[v(after the event)]. Each term has expectation zero whatever v is, so the
corrected score keeps the raw score's mean, and its variance falls by as much of the outcome's
variance as v attributes to luck. A good critic removes more luck; a poor one removes less; no
critic biases the result.

A game is replayed from its tournament log (`tools/tournament.py --log-games`: the deal seed and
every action in order, each in its seat's action view). At each step whose resolution draws on the
engine's RNG -- a die, a deal, a random discard; with auto-advance, everything up to the next
decision -- the same action is applied to `clones` copies of the position before it, each with a
fresh RNG state, and their mean value stands for the expectation. The actual draw and the clones'
are drawn alike, so the correction stays zero-mean at any number of clones; the number only sets
how much luck is removed. The actual step is taken by the same function as the clones' (`advance`),
and every replayed game must end where its log says it did, so a clone cannot resolve chance in a
way the tournament did not. The initial deal is corrected the same way, from fresh deal seeds.

Only chance is corrected. AIVAT also corrects at the decisions of a player whose policy is known,
by sum_a pi(a) v(h.a) - v(h.a_taken): for a greedy player that term is zero, and a searcher's
policy is not known, so decisions are left alone -- still unbiased, only less reduced.

The value is the US's, in [-1, 1] like the terminal utility: a network's critic read from the side
about to decide (`ai.eval.leak_critic.mover_values`, the only frame it is trained in), or `vp`,
tanh(VP / 10), a crude one that needs no network. The tournament plays every deal from both seats,
so a matchup's standard errors are over deal pairs, the two games of a seed together.

    tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release python tools/tournament.py \\
        --models A.pt B.pt --games-per-side 1000 --temperature 0 --log-games games.jsonl
    PYTHONPATH=.:build/release python -m ai.eval.aivat --games games.jsonl --value critic.pt [--clones 8]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypedDict

import numpy as np

import ts_engine as ts
from ai.eval.leak_critic import mover_values
from ai.eval.paired_playouts import decider
from tools.lib.game_step import IllegalActionError, drain_chance
from tools.lib.player_agent import NeuralAgent, resolve_device

#: The US's value of each position, in [-1, 1]; a finished game is its terminal utility.
ValueFn = Callable[[Sequence[ts.GameState]], np.ndarray]

_UINT64 = 1 << 64


def vp_value(states: Sequence[ts.GameState]) -> np.ndarray:
    """tanh(VP / 10) from the US's side; a finished game its utility. Needs no network."""
    return np.array([float(ts.Engine.get_terminal_utility(st)) if ts.Engine.is_terminal(st)
                     else math.tanh(int(st.victory_points) / 10.0) for st in states], dtype=np.float64)


def critic_value(model: Any, device: Any) -> ValueFn:
    """`model`'s critic, read from the side about to decide and turned to the US's side."""
    def value(states: Sequence[ts.GameState]) -> np.ndarray:
        return mover_values(model, device, states, [ts.Player.US] * len(states))
    return value


def new_game(seed: int) -> ts.GameState:
    """A game dealt from `seed`, settled at its first decision, as the batch runner deals it."""
    st = ts.GameState()
    ts.Engine.init_game(st, int(seed) % _UINT64)
    drain_chance(st, context="aivat deal")
    return st


def advance(state: ts.GameState, action: int, us_merged: bool, ussr_merged: bool,
            rng_state: Optional[int] = None) -> ts.GameState:
    """The position after `action`, settled to the next decision as the tournament steps
    (auto-advance, chance drained), in the decider's action view; with `rng_state`, the chance it
    meets drawn from that RNG state instead of the position's own."""
    st = state.clone()
    if rng_state is not None:
        st.rng_state = int(rng_state) % _UINT64
    merged = us_merged if decider(state) == ts.Player.US else ussr_merged
    if not ts.Engine.try_step_flat(st, int(action), True, merged):
        raise IllegalActionError(f"aivat replay: the engine refused flat action {action}")
    drain_chance(st, context="aivat replay")
    return st


def is_over(st: ts.GameState) -> bool:
    return bool(ts.Engine.is_terminal(st))


class ReplayMismatch(RuntimeError):
    """A logged game did not replay to its logged end: another engine, or a log this cannot read."""


def check_entry(entry: Dict[str, Any]) -> None:
    for key in ("seed", "actions", "utility", "us_merged", "ussr_merged", "us_agent", "ussr_agent"):
        if key not in entry:
            raise ReplayMismatch(f"game {entry.get('game_index')}: the log has no {key!r}; "
                                 f"replay needs a log from tools/tournament.py --log-games at this commit")
    if not entry.get("auto_advance", True):
        raise ReplayMismatch(f"game {entry.get('game_index')}: played without auto-advance")
    if entry.get("start_state"):
        raise ReplayMismatch(f"game {entry.get('game_index')}: resumed from a supplied position, "
                             f"which the log does not hold")


@dataclass
class ScoredGame:
    """One game: its observed US utility and the sum of its chance corrections, US side."""
    seed: int
    us: str
    ussr: str
    utility: float
    correction: float
    chance_steps: int

    @property
    def corrected(self) -> float:
        return self.utility - self.correction


def _clone_rng(salt: int, entry: Dict[str, Any]) -> random.Random:
    key = f"{salt}/{entry['seed']}/{entry['us_agent']}/{entry['ussr_agent']}/{entry.get('game_index')}"
    return random.Random(int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "little"))


def score_games(entries: Sequence[Dict[str, Any]], value: ValueFn, clones: int = 8, salt: int = 0,
                block: int = 16) -> List[ScoredGame]:
    """Replay every logged game and sum its chance corrections. Games are replayed `block` at a
    time, their positions valued in one call."""
    if clones < 1:
        raise ValueError(f"clones must be at least 1, got {clones}")
    out: List[ScoredGame] = []
    for lo in range(0, len(entries), block):
        states: List[ts.GameState] = []
        # (game in block, index of the actual position, index of its first clone)
        terms: List[Tuple[int, int, int]] = []
        steps = [0] * len(entries[lo:lo + block])
        for j, e in enumerate(entries[lo:lo + block]):
            check_entry(e)
            rng = _clone_rng(salt, e)
            st = new_game(int(e["seed"]))
            terms.append((j, len(states), len(states) + 1))
            states.append(st)
            states.extend(new_game(rng.getrandbits(64)) for _ in range(clones))
            for n, a in enumerate(e["actions"]):
                if is_over(st):
                    raise ReplayMismatch(f"game {e.get('game_index')} (seed {e['seed']}): over after "
                                         f"{n} of its {len(e['actions'])} actions")
                nxt = advance(st, a, bool(e["us_merged"]), bool(e["ussr_merged"]), st.rng_state)
                if nxt.rng_state != st.rng_state:      # the step drew on chance
                    terms.append((j, len(states), len(states) + 1))
                    states.append(nxt)
                    states.extend(advance(st, a, bool(e["us_merged"]), bool(e["ussr_merged"]),
                                          rng.getrandbits(64)) for _ in range(clones))
                    steps[j] += 1
                st = nxt
            want = float(e["utility"])
            got = float(ts.Engine.get_terminal_utility(st)) if is_over(st) else 0.0
            if got != want or ("victory_points" in e and int(st.victory_points) != int(e["victory_points"])):
                raise ReplayMismatch(f"game {e.get('game_index')} (seed {e['seed']}) replays to utility "
                                     f"{got}, VP {int(st.victory_points)}; its log says {want}, VP "
                                     f"{e.get('victory_points')}")
        v = np.asarray(value(states), dtype=np.float64)
        corr = [0.0] * len(steps)
        for j, actual, first in terms:
            corr[j] += float(v[actual] - v[first:first + clones].mean())
        for j, e in enumerate(entries[lo:lo + block]):
            out.append(ScoredGame(int(e["seed"]), str(e["us_agent"]), str(e["ussr_agent"]),
                                  float(e["utility"]), corr[j], steps[j]))
    return out


# ---- the matchup's numbers ----------------------------------------------------------------------


class Estimate(TypedDict):
    score: float            # player a's mean score, a draw half a win
    se: float               # over deal pairs
    elo: Optional[float]    # a's Elo over b; None at a score of 0 or 1
    elo_se: Optional[float]


class MatchupAivat(TypedDict):
    a: str
    b: str
    games: int
    pairs: int
    raw: Estimate
    aivat: Estimate
    #: Var(raw) / Var(aivat): how many times the games the corrected score is worth.
    variance_ratio: Optional[float]
    #: a's mean correction in score units, and its SE: zero up to noise, or the replay is wrong.
    correction: float
    correction_se: float
    chance_steps_per_game: float


def _estimate(pair_means: np.ndarray) -> Estimate:
    p = float(pair_means.mean())
    se = float(pair_means.std(ddof=1) / math.sqrt(len(pair_means))) if len(pair_means) > 1 else float("nan")
    if 0.0 < p < 1.0:
        elo: Optional[float] = 400.0 * math.log10(p / (1.0 - p))
        elo_se: Optional[float] = 400.0 / math.log(10.0) * se / (p * (1.0 - p))
    else:
        elo = elo_se = None
    return {"score": p, "se": se, "elo": elo, "elo_se": elo_se}


def summarise(games: Sequence[ScoredGame]) -> List[MatchupAivat]:
    """Per matchup (a = the first name in sorted order): a's raw and corrected scores, each with
    its SE over deal pairs. A player against itself has nothing to estimate and is left out."""
    by_matchup: Dict[Tuple[str, str], List[ScoredGame]] = {}
    for g in games:
        if g.us != g.ussr:
            by_matchup.setdefault((min(g.us, g.ussr), max(g.us, g.ussr)), []).append(g)
    out: List[MatchupAivat] = []
    for (a, b), gs in sorted(by_matchup.items()):
        pairs: Dict[int, List[Tuple[float, float, float]]] = {}
        for g in gs:
            sign = 1.0 if g.us == a else -1.0               # a's side of a US utility
            pairs.setdefault(g.seed, []).append(((sign * g.utility + 1.0) / 2.0,
                                                 (sign * g.corrected + 1.0) / 2.0,
                                                 sign * g.correction / 2.0))
        cols = np.array([np.mean(np.array(v), axis=0) for v in pairs.values()], dtype=np.float64)
        raw, cor = _estimate(cols[:, 0]), _estimate(cols[:, 1])
        n = len(cols)
        out.append({
            "a": a, "b": b, "games": len(gs), "pairs": n, "raw": raw, "aivat": cor,
            "variance_ratio": (raw["se"] / cor["se"]) ** 2 if cor["se"] > 0 else None,
            "correction": float(cols[:, 2].mean()),
            "correction_se": float(cols[:, 2].std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
            "chance_steps_per_game": float(np.mean([g.chance_steps for g in gs])),
        })
    return out


def _elo(e: Estimate) -> str:
    return "n/a" if e["elo"] is None or e["elo_se"] is None else f"{e['elo']:+.1f} ± {1.96 * e['elo_se']:.1f}"


def markdown(rows: Sequence[MatchupAivat], value_name: str, clones: int) -> str:
    lines = [f"AIVAT over chance nodes, value `{value_name}`, {clones} clones per chance step. "
             f"Scores are a's, ± 1.96 SE over deal pairs.", "",
             "| a | b | pairs | raw score | raw Elo | AIVAT score | AIVAT Elo | variance ratio | "
             "mean correction (z) | chance steps/game |",
             "|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for r in rows:
        z = r["correction"] / r["correction_se"] if r["correction_se"] > 0 else float("nan")
        ratio = "n/a" if r["variance_ratio"] is None else f"{r['variance_ratio']:.2f}×"
        lines.append(f"| {r['a']} | {r['b']} | {r['pairs']} | {r['raw']['score']:.4f} ± "
                     f"{1.96 * r['raw']['se']:.4f} | {_elo(r['raw'])} | {r['aivat']['score']:.4f} ± "
                     f"{1.96 * r['aivat']['se']:.4f} | {_elo(r['aivat'])} | {ratio} | "
                     f"{r['correction']:+.4f} ({z:+.1f}) | {r['chance_steps_per_game']:.1f} |")
    return "\n".join(lines)


def load_games(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", required=True, help="a tools/tournament.py --log-games file")
    ap.add_argument("--value", required=True,
                    help="a checkpoint whose critic is the value, or `vp` (tanh(VP / 10), no network)")
    ap.add_argument("--clones", type=int, default=8, help="fresh chance draws per chance step")
    ap.add_argument("--seed", type=int, default=0, help="salts the clones' chance draws")
    ap.add_argument("--block", type=int, default=16, help="games replayed and valued together")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output-json", default=None)
    a = ap.parse_args(argv)

    if a.value == "vp":
        value: ValueFn = vp_value
    else:
        dev = resolve_device(a.device)
        value = critic_value(NeuralAgent.from_checkpoint(a.value, device=dev).model, dev)
    try:
        rows = summarise(score_games(load_games(a.games), value, a.clones, a.seed, a.block))
    except ReplayMismatch as e:
        print(f"aivat: {e}", file=sys.stderr)
        return 1
    print(markdown(rows, a.value, a.clones))
    if a.output_json:
        with open(a.output_json, "w", encoding="utf-8") as f:
            json.dump({"value": a.value, "clones": a.clones, "seed": a.seed, "matchups": rows}, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
