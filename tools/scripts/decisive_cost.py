#!/usr/bin/env python3
"""How often does a checkpoint miss a forced win in self-play, and what does it cost in Elo?

Plays self-play games (greedy by default -- the setting tournaments rate in), classifies every
decision with `ai.eval.decisive_probe.classify_in_view` from whichever codebase is first on
PYTHONPATH, and records each game's result. With the version-2 classifier (main, PR #6) chances to
win are folded with `ai.eval.safety.fold_win_opportunities` (one per chance, however many decisions
it takes); without it each decision offering a win counts once (version 1).

Cost: a forced win a side missed and then lost the game is a game a player that always takes its
forced wins would have won in that seat. Delta = mean over games and both seats of
P(seat missed a win and lost); Elo = 400 log10((0.5 + Delta) / (0.5 - Delta)) against the
checkpoint itself. First order: it ignores how taking the win earlier would change later play.

    PYTHONPATH=<codebase>:build/release python tools/scripts/decisive_cost.py --checkpoint <pt> --games 1024
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import ts_engine as ts

from ai.eval.decisive_probe import classify_in_view

import importlib

#: version 2 (main, PR #6) folds chances to win; version 1 has no fold and counts per decision.
_fold: Any = getattr(importlib.import_module("ai.eval.safety"), "fold_win_opportunities", None)
VERSION = 2 if _fold is not None else 1


def _opportunities(decisions: List[Tuple[int, int, int, bool]]) -> List[Tuple[int, bool]]:
    """(mover, taken) per chance to win in one game."""
    if _fold is not None:
        return [(decisions[i][0], taken) for i, taken in _fold(decisions)]
    return [(mover, taken) for mover, _n, n_wins, taken in decisions if n_wins]


def play(model: Any, games: int, base_seed: int, temperature: float, batch: int) -> List[Dict[str, Any]]:
    from bindings.ts_env import TsVectorizedEnv, model_obs_features
    device = next(model.parameters()).device
    model.eval()
    out: List[Dict[str, Any]] = []
    for b0 in range(0, games, batch):
        n = min(batch, games - b0)
        env = TsVectorizedEnv(num_envs=n, base_seed=base_seed + b0)
        env.set_obs_features(model_obs_features(model), model_obs_features(model))
        obs, masks, _ = env.reset_all()
        pending: List[List[Tuple[int, int, int, bool]]] = [[] for _ in range(n)]
        counted = [False] * n
        result: List[Optional[float]] = [None] * n
        for _ in range(20_000):
            if all(counted):
                break
            with torch.no_grad():
                logits = model(torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device),
                               torch.from_numpy(np.asarray(masks)).to(device))[0].float()
                if temperature > 0:
                    acts = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(-1)
                else:
                    acts = logits.argmax(-1)
            actions = acts.cpu().numpy()
            for i in range(n):
                if counted[i]:
                    continue
                st = env.runner.get_state(i)
                if ts.Engine.is_terminal(st):
                    continue
                ctx = st.ctx()
                if ctx.decision_type == ts.DecisionType.ROLL_DIE:
                    continue
                player = ctx.decision_player if ctx.decision_player != ts.Player.NONE else st.phasing_player
                kinds = classify_in_view(st, player, False)
                if not kinds:
                    continue
                n_wins = sum(1 for k in kinds.values() if k == "win")
                pending[i].append((int(player), len(kinds), n_wins, kinds.get(int(actions[i])) == "win"))
            obs, masks, _, dones, info = env.step(actions)
            # The env auto-resets a finished game, so its result is read from the episode record
            # the step returns, never from the runner afterwards (that is the next game).
            finished = {int(e["env_idx"]): float(e["terminal_utility"]) for e in info.get("completed_episodes", [])}
            for i, done in enumerate(dones):
                if done and not counted[i]:
                    counted[i] = True
                    result[i] = finished.get(i)
        for i in range(n):
            if result[i] is None:
                raise RuntimeError(f"game {b0 + i}: finished without an episode record")
            out.append({"decisions": len(pending[i]), "opps": _opportunities(pending[i]), "us_util": result[i]})
    return out


def report(rows: Sequence[Dict[str, Any]], version: int = VERSION) -> str:
    us, ussr = int(ts.Player.US), int(ts.Player.USSR)
    n_games = len(rows)
    opps = [(m, t) for r in rows for (m, t) in r["opps"]]
    missed = sum(1 for _, t in opps if not t)
    lines = [f"classifier version {version}; {n_games:,} games, {sum(r['decisions'] for r in rows):,} classified decisions",
             f"chances to win: {len(opps):,} ({len(opps) / n_games:.3f} per game); taken {len(opps) - missed:,}, "
             f"missed {missed:,} -> take rate {1 - missed / max(1, len(opps)):.3f}"]
    cost = 0.0
    for side, name in ((us, "US"), (ussr, "USSR")):
        s_opps = [t for r in rows for (m, t) in r["opps"] if m == side]
        miss_games = [r for r in rows if any(m == side and not t for m, t in r["opps"])]
        lost = [r for r in miss_games if r["us_util"] * (1 if side == us else -1) < 0]
        frac = len(lost) / n_games
        cost += frac / 2.0
        lines.append(f"  {name:4s}: chances {len(s_opps):,}, missed {sum(1 for t in s_opps if not t):,}; "
                     f"games with a missed win {len(miss_games):,}, of which lost {len(lost):,} "
                     f"({frac:.4f} of games)")
    p = 0.5 + cost
    se = math.sqrt(max(cost * (1 - cost), 1e-12) / n_games)
    elo = 400 * math.log10(p / (1 - p))
    lines.append(f"win-rate cost Delta = {cost:.4f} ± {se:.4f} per game -> {elo:+.1f} Elo for always taking forced wins")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    from tools.lib.player_agent import NeuralAgent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--games", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=77_000)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--dump", default=None, help="write the per-game records here (JSON), for --merge")
    ap.add_argument("--merge", nargs="+", default=None, help="report over these dumps instead of playing")
    a = ap.parse_args(argv)
    if a.merge:
        import json
        rows: List[Dict[str, Any]] = []
        versions = set()
        for f in a.merge:
            d = json.load(open(f))
            versions.add(d["version"])
            rows += [{"decisions": r["decisions"], "opps": [tuple(o) for o in r["opps"]], "us_util": r["us_util"]}
                     for r in d["rows"]]
        if len(versions) != 1:
            raise SystemExit(f"dumps mix classifier versions {versions}")
        print(f"merged {len(a.merge)} dumps")
        print(report(rows, versions.pop()))
        return 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralAgent.from_checkpoint(a.checkpoint, device=str(dev)).model
    t0 = time.time()
    rows = play(model, a.games, a.seed, a.temperature, a.batch)
    print(f"checkpoint {a.checkpoint}, temperature {a.temperature}, {time.time() - t0:.0f}s")
    print(report(rows))
    if a.dump:
        import json
        json.dump({"version": VERSION, "checkpoint": a.checkpoint, "seed": a.seed, "rows": rows}, open(a.dump, "w"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
