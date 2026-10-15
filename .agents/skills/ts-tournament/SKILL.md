---
name: ts-tournament
description: >-
  Run tournaments and head-to-head match evaluations between Twilight Struggle neural network models
  and heuristic bots. Activate this skill whenever the user asks to evaluate a model, compare two
  checkpoints, run a tournament, calculate Bradley-Terry Elo ratings, diagnose loss causes, or benchmark agents.
---

# Twilight Struggle AI: Tournament & Evaluation Runbook

Benchmarking models with vectorized parallel simulation. Full flags: `tools/README.md`.

> [!IMPORTANT]
> - Prefix every evaluation with `tools/scripts/check_engine_fresh.sh &&`: a number measured on a
>   stale engine is a number about a different game.
> - Never write ad-hoc evaluation scripts: use `tools/tournament.py` or the leaderboard CLIs.
> - Find checkpoints with `PYTHONPATH=. .venv/bin/python tools/inspect_checkpoints.py` (it detects
>   each architecture from the weights); the recommended models are listed in `research/checkpoints.md`.

---

## 1. Commands

### Head-to-head (two models: per-side results and loss causes)
```bash
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/tournament.py \
  --models data/checkpoints/<run_A>/<snapshot_A>.pt data/checkpoints/<run_B>/<snapshot_B>.pt \
  --games-per-side 500
```
A bot name (`heuristic`, `random`) can stand in for either model.

### Round robin over a run directory (Bradley-Terry Elo)
```bash
PYTHONPATH=.:build/release .venv/bin/python tools/tournament.py \
  --checkpoint-dir data/checkpoints/<run-dir> --games-per-side 500 \
  --anchor-model HeuristicBot --anchor-elo 1500.0 \
  --output-report <run-dir>/tournament_report.md --output-json <run-dir>/tournament_results.json
```
`--workers N` shards the pairs over processes. Elo from one tournament is relative to that field.

### The leaderboard (ratings comparable across runs: one scale per engine epoch)
```bash
PYTHONPATH=. .venv/bin/python tools/leaderboard.py add-network <name> <file.pt> --description "..."
PYTHONPATH=. .venv/bin/python tools/leaderboard.py add-player '<name>[~gumbel(sims=256,k=8)]'
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python \
  tools/leaderboard_play.py --epoch E7 --players '<player-id>' --games-per-side 1000
PYTHONPATH=. .venv/bin/python tools/leaderboard.py show --main-only
```
Records live in `leaderboard/` (append-only); ratings are fitted, never committed.

## 2. Reading the results

- **Per side, always.** Twilight Struggle is asymmetric, so read the US and USSR win rates
  separately — from the JSON (`per_side`, `win_rate_a_as_us`) or the log's Record lines, not the
  markdown's x%/y% pair. A seat's head-to-head rate means little raw: compare it with the control's
  own self-play rate in that seat.
- **Ending reasons** (`bindings.ts_env.ENDING_REASON_KEYS`): `20vp`, `europe_control`,
  `final_scoring`, `defcon1_self` (the player's own move took DEFCON to 1), `defcon1_provoked`
  (forced there by the opponent's card or event), `held_scoring` (a scoring card held at turn end),
  `wargames`.
- How much evidence a difference needs: `research/method/measurement_pitfalls.md`.
