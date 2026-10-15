---
name: ts-match-replay
description: >-
  Simulate individual Twilight Struggle matches between agents, play interactively via CLI terminal,
  generate standardized .tslog.json game replays, and play or watch games in the browser Web Workbench.
  Activate this skill whenever the user asks to play a match, watch a game replay, play interactively
  in the terminal, or play against a model in the web UI.
---

# Twilight Struggle AI: Match Simulator & Replay Guide

Offline matches between any two agents, terminal play, and replays in the Web Workbench. Never
write ad-hoc match scripts: use `tools/play_match.py`.

---

## 1. Matches and replays

```bash
# Two checkpoints (or any agent identifiers, section 3)
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py \
  --us data/checkpoints/<run_A>/<snapshot_A>.pt --ussr data/checkpoints/<run_B>/<snapshot_B>.pt \
  --game-id match_A_vs_B --seed 42

# Human in the terminal against the newest checkpoint
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py --us human --ussr neural --game-id my_game

# With strategic commentary and scoring audits in the replay metadata
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py --us strategic --ussr event_heavy --commentary
```

Each match writes `data/replays/<game-id>.tslog.json` and prints the workbench URL
(`http://localhost:8000/?replay=<file>`). `--trace` records the model's policy and critic at every
step for the workbench's readout; `--opening`, `--temperature` and `--device` are also available.

## 2. Web Workbench

The workbench runs entirely in the browser (engine as WebAssembly, models as ONNX); the local
server only lists this machine's checkpoints and replays.

```bash
tools/scripts/build_web.sh                       # rebuild after any engine change
PYTHONPATH=.:build/release .venv/bin/python -m web.server.main --port 8000
```

Open `http://localhost:8000/`. Choose *Play as US / USSR* (the other side's private information is
hidden and the model plays it) or *Developer*; pick a model in *Model Analysis* (a local checkpoint,
a Hugging Face repo, or a dropped `.onnx`) and set *Auto-play* for its side. Replays load from the
server's list, *Load Replay*, or by dropping a `.tslog.json`. Details: `web/ui/AGENTS.md`.

## 3. Agent identifiers (`--us`, `--ussr`, `--agent` for self-play)

- `heuristic` ([`HeuristicBot`](../../../bot/heuristic_bot.py)), `random` ([`RandomBot`](../../../bot/random_bot.py)),
  `strategic` ([`StrategicBot`](../../../bot/strategic_bot.py)), `event_heavy` ([`EventHeavyBot`](../../../bot/event_heavy_bot.py)),
  `exploratory` ([`ExploratoryBot`](../../../bot/exploratory_bot.py)), `human` ([`HumanBot`](../../../bot/human_bot.py)).
- `heuristic_mcts[:sims]` ([`HeuristicMCTSBot`](../../../bot/heuristic_mcts_bot.py)): search with no checkpoint; it sees both hands.
- `neural`: the newest checkpoint under `data/checkpoints/`; `neural:<path>` or any `.pt` path: that
  checkpoint, its architecture detected from the weights.
