# Bot Clients & AI Agents Guide (`bot/`)

Bot implementations, baseline heuristics and decision policies for **Twilight Struggle**. All bots
inherit from [`BaseBot`](base_bot.py).

---

## 1. Overview of Bot Clients

- [`base_bot.py`](base_bot.py): the abstract `BaseBot` interface:
  - `select_action(state_dict, legal_actions_dict) -> Optional[dict]`: dict-based selection for JSON-driven callers.
  - `select_flat_action(state, player) -> int`: a flat action index, for vectorized environments.
  - `reset()`: clears internal memory or search state between games.
  - `require_player_decision(d_type)`: raises on a decision type no player is asked
    (`PLAYER_DECISION_TYPES`: SELECT_CARD, SELECT_PLAY_MODE, SELECT_OP_MODE, POINT_NODE,
    CHOOSE_BRANCH). Every dict bot calls it first, and answers an empty `valid_ids` with
    `confirm_done(d_type)` (the engine keeps CONFIRM_DONE legal when nothing else is). A known
    decision a bot's rules have no preference at gets any legal id; an unknown one raises rather
    than being answered with a random or first id. Bots put no dice in an action: the engine
    rolls from its own RNG.
- [`random_bot.py`](random_bot.py): `RandomBot`, uniform over legal moves.
- [`heuristic_bot.py`](heuristic_bot.py): the terminal match's `HeuristicBot`, a fixed rule list
  (battlegrounds, scoring-card timing, ops efficiency). Its JSON influence scoring reads countries
  by their display names. Tournament evaluation uses the separate native-state `HeuristicAgent`
  in `tools/lib/player_agent.py` as its default Elo anchor.
- [`heuristic_mcts_bot.py`](heuristic_mcts_bot.py): `HeuristicMCTSBot` (`heuristic_mcts[:sims]` in `tools/play_match.py`), MCTS over the real engine with `ai.search.heuristic_eval` at the leaf and no checkpoint. 18/20 against HeuristicBot from 16 simulations on (`research/findings/training/heuristic_mcts_baseline.md`). **It searches the true state and sees the opponent's hand**: fine for a benchmark, never a claim about play under real information.
- Networks are not bots here. `NeuralBot` (a dict-driven client that rebuilt the flat mask from
  `valid_ids` by hand and read the base observation, so it could play no featured checkpoint) was
  removed; `tools/play_match.py` plays a checkpoint or ONNX export through
  `tools.lib.player_agent.load_agent` on the engine state, in the network's own observation
  features and action view.
- [`exploratory_bot.py`](exploratory_bot.py): `ExploratoryBot`, explores diverse paths: Space Race, realignments, coups.
- [`strategic_bot.py`](strategic_bot.py): `StrategicBot`, DEFCON-2 containment, coups, realignments and Space Race safety, with strategy commentary.
- [`event_heavy_bot.py`](event_heavy_bot.py): `EventHeavyBot`, maximises event play and event-first timing.
- [`human_bot.py`](human_bot.py): `HumanBot`, interactive terminal player prompting for numbered choices.

---

## 2. Launching Bots

**Against a model in the web workbench** no bot process is involved: the page plays ONNX exports
of checkpoints itself (pick a model in *Model Analysis*, set *Auto-play*; see
[`web/ui/AGENTS.md`](../web/ui/AGENTS.md)).

**Offline matches** with replay generation go through [`tools/play_match.py`](../tools/play_match.py):
```bash
PYTHONPATH=. .venv/bin/python tools/play_match.py --us heuristic --ussr strategic
```

---

## 3. Mandatory Documentation Maintenance Rule for Agents

> [!IMPORTANT]
> **Keep Bot Documentation Synchronized**:
> Whenever adding new bot strategies or modifying bot interfaces, you **MUST** update this file and root [`AGENTS.md`](../AGENTS.md).
