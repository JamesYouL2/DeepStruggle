# Bot Clients & AI Agents Guide (`bot/`)

This directory contains pure bot client implementations, baseline heuristics, and decision policies for **Twilight Struggle**. All bots inherit from [`BaseBot`](base_bot.py).

---

## 1. Overview of Bot Clients

- [`base_bot.py`](base_bot.py): Abstract `BaseBot` class defining the standard interface for Twilight Struggle agents:
  - `select_action(state_dict, legal_actions_dict) -> Optional[dict]`: Dict-based action selection for JSON-driven callers.
  - `select_flat_action(state, player) -> int`: Fast 212-dim flat index selection for vectorized environments.
  - `reset()`: Resets internal memory or search state between games.
- [`random_bot.py`](random_bot.py): `RandomBot` baseline stochastic player making uniform random choices over legal moves.
- [`heuristic_bot.py`](heuristic_bot.py): `HeuristicBot` rule-based expert player prioritizing key battlegrounds, scoring card timing, and ops efficiency.
- [`neural_bot.py`](neural_bot.py): `NeuralBot` deep reinforcement learning player driven by `ColdWarNet` PyTorch checkpoints. The architecture (V1 or V2) is detected from the weights; a checkpoint from a retired architecture is refused outright by `tools.lib.player_agent.reject_retired_architecture` rather than partially loaded.
- [`exploratory_bot.py`](exploratory_bot.py): `ExploratoryBot` agent designed to explore diverse decision paths, Space Race, Realignments, and Coups.
- [`strategic_bot.py`](strategic_bot.py): `StrategicBot` high-level strategic agent prioritizing DEFCON-2 containment, coups, realignments, and Space Race safety with rich strategy/commentary generation.
- [`doctrine/`](doctrine/): `Doctrine`, the struggler project's strategic bot ported onto
  ts_engine (section 3). `DoctrineAgent` is the tournament `PlayerAgent` (`load_agent("doctrine")`);
  `DoctrineBot` the match-loop bot (`play_match.py --us doctrine`).
- [`event_heavy_bot.py`](event_heavy_bot.py): `EventHeavyBot` agent maximizing card event play and event-first timing.
- [`human_bot.py`](human_bot.py): `HumanBot` interactive terminal CLI player prompting the user for numbered choices.

---

## 2. Launching Bots

### Playing Against a Model in the Web Workbench:
The workbench runs in the browser and plays neural models itself (ONNX exports of checkpoints),
so no bot process is involved: pick a model in *Model Analysis* and set *Auto-play* to the side it
should play (see [`web/ui/AGENTS.md`](../web/ui/AGENTS.md)). The network bot client that used to
connect a `BaseBot` to a server-side game (`web/bot_client.py`) was removed with that server.

### Running Offline Matches Between Bots:
Matches can be run offline with full replay generation via [`tools/play_match.py`](../tools/play_match.py):
```bash
# Run a match between HeuristicBot and StrategicBot
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic
```

---

## 3. Doctrine

A port of struggler's `StrategicPlayer` (`src/struggler/bots/strategic/`), aiming at comparable
playing strength rather than decision-for-decision fidelity.

* `evaluator.py` -- struggler's board value, ported term for term with its shipped default
  weights and its fitted per-country weights (`data/fitted_country_weights.json`, renamed to
  ts_engine's country names). `tests/training/test_doctrine.py` pins it to struggler's own
  output on a fixed board, to 1e-12.
* `schedule.py` -- each region's future scoring mass (struggler's `schedule.py` and
  `public_cards.py`), with a simpler deck walk. Public information only.
* `policy.py` -- the search. struggler prices options by hand and runs events in a copy of its
  Python engine; here the C++ engine is the sandbox. Each option is played out on a clone of a
  **determinized** state (the opponent's hidden cards and the deck reshuffled, so nothing hidden
  is read) to the end of the action round, then valued. Card, play-mode, Ops-mode and event-branch
  choices are searched; influence points and targets are chosen greedily, point by point, as
  struggler does; the first die on a line is averaged over its faces, later ones take the middle
  roll.

* The DEFCON whole-hand survival search, reduced: at DEFCON 3 or below each card in hand is tried
  at DEFCON 2, and a card choice that leaves fewer safe plays than rounds to fill is charged the
  game (all of it at DEFCON 2, struggler's measured 0.43 of it at 3).

Measured with `tools/tournament.py` (2026-09-29): 40-0 against HeuristicBotV2 (20 games a side)
and 48-2 against HeuristicMCTS16 (25 a side; it searches the true state, so it sees this bot's
hand). About 20 s a game on one core.

Not yet ported: the one-ply reply look-ahead, the hand-value terms (holding a card, Ask Not,
Missile Envy targets) and the Military Ops discount.

```bash
PYTHONPATH=.:build/release .venv/bin/python tools/play_match.py --us doctrine --ussr heuristic
PYTHONPATH=.:build/release .venv/bin/python tools/tournament.py --models doctrine heuristic_v2 \
  --games-per-side 20 --device cpu
```

---

## 4. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Bot Documentation Synchronized**:
> Whenever adding new bot strategies or modifying bot interfaces, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).
