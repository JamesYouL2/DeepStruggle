"""Does the critic agree with itself across a hand-over between the two sides?

At a hand-over -- the last decision of one side, s_t, followed by the first decision of the other,
s_{t+1} -- each side's critic is read from its own perspective: v_A(s_t) and v_B(s_{t+1}). They see
different things (each knows its own hand), so they need not agree position by position. But each
predicts its own return, so averaged over positions

    E[v_A(s_t) + v_B(s_{t+1})] = E[G_A] + E[G_B],

which is 0 whenever the targets are zero-sum -- no nesting of information is needed for the
average, and it holds within any class of positions both sides can see (turn, phase), though not
within a class defined by one side's hand. The recipe's targets are zero-sum except in the turn of
a blunder (`blunder_aware`: the blunderer's target is -1 and the winner's is pinned to its own
value), so a calibrated critic should show a small negative mean there and ~0 elsewhere.

This plays self-play games, records the mover's value at every decision, and reports the mean of
v_prev + v_next over hand-overs (standard errors clustered by game), split by direction, turn and
phase, and decomposed into each side's bias against the realised result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import ts_engine as ts


@dataclass
class Decision:
    side: int          # +1 US, -1 USSR
    value: float       # v_win from the mover's perspective
    turn: int
    phase: str


@dataclass
class Game:
    decisions: List[Decision] = field(default_factory=list)
    us_util: float = 0.0
    blunder: str = ""              # "", "defcon_self" or "held_scoring"


def _blunder_kind(state: ts.GameState) -> str:
    """How the game ended, as far as the blunder window is concerned (ai/rewards/reward_calculator)."""
    if int(state.defcon) <= 1:
        return "defcon"            # provoked or not -- recorded coarsely; see the report
    if ts.Engine.is_held_scoring_game_over(state):
        return "held_scoring"
    return ""


def play(model: Any, device: Any, num_envs: int = 512, base_seed: int = 4242, temperature: float = 1.0,
         max_iters: int = 4000) -> List[Game]:
    """Self-play games with `model` on both sides, every decision's mover value recorded."""
    import torch

    runner = ts.VectorizedBatchRunner(num_envs, base_seed)
    games = [Game() for _ in range(num_envs)]
    done = np.zeros(num_envs, dtype=bool)
    model.eval()
    for _ in range(max_iters):
        term = np.asarray(runner.get_terminals(), dtype=bool)
        newly = term & ~done
        for i in np.flatnonzero(newly):
            st = runner.get_state(int(i))
            games[i].blunder = _blunder_kind(st)
        done |= term
        if done.all():
            break
        obs = np.asarray(runner.get_observations(), dtype=np.float32)
        masks = np.asarray(runner.get_action_masks())
        players = np.asarray(runner.get_decision_players())
        with torch.no_grad():
            logits, v_win, _ = model(torch.from_numpy(obs).to(device), torch.from_numpy(masks).to(device))
            values = v_win.float().squeeze(-1).cpu().numpy()
            probs = torch.softmax(logits.float() / max(temperature, 1e-6), dim=-1)
            picks = (torch.multinomial(probs, 1).squeeze(-1) if temperature > 0 else logits.argmax(-1)).cpu().numpy()
        for i in np.flatnonzero(~done):
            side = int(players[i])
            if side == 0:
                continue
            st = runner.get_state(int(i))
            games[i].decisions.append(Decision(side, float(values[i]), int(st.turn), str(st.current_phase).split(".")[-1]))
        runner.step_flat_all([int(a) for a in picks], auto_advance=True)
    utils = np.asarray(runner.get_terminal_utilities(), dtype=np.float64)
    for i in range(num_envs):
        games[i].us_util = float(utils[i])
    return [g for i, g in enumerate(games) if done[i]]


def _clustered(values: Sequence[float], clusters: Sequence[int]) -> Tuple[float, float, int]:
    """Mean, its standard error clustered by game, and the count."""
    v = np.asarray(values, dtype=np.float64)
    if v.size == 0:
        return float("nan"), float("nan"), 0
    c = np.asarray(clusters)
    m = float(v.mean())
    sums = np.array([float((v[c == k] - m).sum()) for k in np.unique(c)])
    se = float(np.sqrt((sums ** 2).sum()) / v.size)
    return m, se, int(v.size)


def handovers(games: Sequence[Game]) -> List[Dict[str, Any]]:
    """One row per hand-over: the two values, their sum, and context."""
    rows: List[Dict[str, Any]] = []
    for gi, g in enumerate(games):
        d = g.decisions
        for a, b in zip(d, d[1:]):
            if a.side == b.side:
                continue
            rows.append({"game": gi, "dir": "US->USSR" if a.side == 1 else "USSR->US", "sum": a.value + b.value,
                         "turn": b.turn, "phase": b.phase, "same_turn": a.turn == b.turn,
                         "blunder": g.blunder,
                         # each side's error against what happened, in its own frame
                         "err_a": a.value - g.us_util * a.side, "err_b": b.value - g.us_util * b.side})
    return rows


def report(games: Sequence[Game]) -> str:
    rows = handovers(games)
    out: List[str] = []

    def line(label: str, sel: List[Dict[str, Any]], key: str = "sum") -> None:
        m, se, n = _clustered([r[key] for r in sel], [r["game"] for r in sel])
        out.append(f"  {label:38s} {m:+.4f} ± {se:.4f}   (n={n:,})")

    ends = [g.blunder for g in games]
    out.append(f"{len(games):,} games; endings: DEFCON 1 {ends.count('defcon')/len(games):.1%}, "
               f"held scoring {ends.count('held_scoring')/len(games):.1%}; US wins {np.mean([g.us_util > 0 for g in games]):.1%}")
    out.append("\nMean of v_prev + v_next over hand-overs (0 if calibrated to zero-sum targets):")
    line("all hand-overs", rows)
    for dname in ("US->USSR", "USSR->US"):
        line(dname, [r for r in rows if r["dir"] == dname])
    line("within a turn", [r for r in rows if r["same_turn"]])
    line("across a turn boundary", [r for r in rows if not r["same_turn"]])
    for ph in sorted({r["phase"] for r in rows}):
        line(f"phase {ph}", [r for r in rows if r["phase"] == ph])
    line("games without a blunder-type ending", [r for r in rows if not r["blunder"]])
    line("games ending in DEFCON 1", [r for r in rows if r["blunder"] == "defcon"])
    line("games ending on held scoring", [r for r in rows if r["blunder"] == "held_scoring"])
    # The same mover's own critic along its own consecutive decisions should be a martingale too:
    # E[v(s_{t+1}) - v(s_t)] = 0. A drift here says where in a run of decisions the bias sits.
    same: List[float] = []
    same_g: List[int] = []
    first_after: List[float] = []
    first_g: List[int] = []
    for gi, g in enumerate(games):
        d = g.decisions
        for k, (a, b) in enumerate(zip(d, d[1:])):
            if a.side == b.side:
                same.append(b.value - a.value)
                same_g.append(gi)
        # within each run of one mover: last value minus first value
        start = 0
        for k in range(1, len(d) + 1):
            if k == len(d) or d[k].side != d[start].side:
                if k - 1 > start:
                    first_after.append(d[k - 1].value - d[start].value)
                    first_g.append(gi)
                start = k
    m, se, n = _clustered(same, same_g)
    out.append(f"\nSame mover, consecutive decisions, v_next - v_prev:  {m:+.4f} ± {se:.4f}   (n={n:,})")
    m, se, n = _clustered(first_after, first_g)
    out.append(f"Same mover, last minus first decision of a run:     {m:+.4f} ± {se:.4f}   (n={n:,})")
    out.append("\nBy turn (all hand-overs):")
    for t in sorted({r["turn"] for r in rows}):
        line(f"turn {t}", [r for r in rows if r["turn"] == t])
    out.append("\nWhere a non-zero sum comes from -- each side's bias against the realised result:")
    for dname in ("US->USSR", "USSR->US"):
        sel = [r for r in rows if r["dir"] == dname]
        line(f"{dname}: first mover's v - result", sel, "err_a")
        line(f"{dname}: second mover's v - result", sel, "err_b")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import torch
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--batches", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=4242)
    a = ap.parse_args(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model
    games: List[Game] = []
    for b in range(a.batches):
        games += play(model, dev, num_envs=a.envs, base_seed=a.seed + 1000 * b, temperature=a.temperature)
    print(f"checkpoint {a.checkpoint}, temperature {a.temperature}")
    print(report(games))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
