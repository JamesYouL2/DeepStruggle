# P13 — One game driver

**Status: done**, except a forced-deal affordance (below). `tools/lib/game_step.py`,
`tools/lib/game_json.py` and `tools/lib/game_loop.py` cite this file for the design.

## The loop

```
state = init(seed)
while not terminal(state):
    action = source.choose(state, legal_mask(state))   # policy | search | bot | human | replay
    state  = step(state, action)                       # checked, recorded
```

Five loops used to drive a game (`TsVectorizedEnv`, `play_match`, `BatchMatchRunner`, the searcher
and the old server-side web session), each with its own action encoding, settle method and
recording. A mismatch between two of them recorded **1,355 refused actions** in a replay as though
they had happened.

## What was built

1. **`step_checked`** (`game_step.py`): a refused action raises at the point of the mistake.
   **`drain_chance`** is the single chance-node drain.
2. **One conversion boundary** (`game_json.py`): state, action ↔ JSON at the edge only; inside the
   loop the one encoding is the flat index.
3. **`GameLoop`** with an injected action source and an explicit `SettlePolicy`. Both replay
   writers (`tools/play_match.py`, `tools/lib/self_play.py`) build sources and delegate.
4. **Batched paths share the contract, not the loop.** Training and `BatchMatchRunner` keep the C++
   `VectorizedBatchRunner`, which drains chance and refuses illegal actions itself; both now check
   `step_flat_all`'s per-game verdict (0 refused, 1 accepted, 2 terminal), which both had discarded.
   0 refusals in 5,052 batched steps before the check, so it is a guard;
   `tests/training/test_batched_step_is_checked.py` forces one. Cost inside noise (`0 in results`;
   `np.asarray` would cost +29%).
5. **Self-contained saves:** `bindings/state_json.cpp` saves and loads the full state, `rng_state`
   included, defaulting missing fields.

**Open:** a forced-deal affordance (a replay source supplying what a deal produced), so a replay
survives a change to shuffling. Dice already have one: a `ROLL_DIE` action carrying the value.

## The replay format's contract, as its readers implement it

* **A chance node is not a step.** A `ROLL_DIE` with `decision_player == NONE` is reproducible from
  the RNG, so no writer records it and every reader regenerates it; `GameLoop` never offers one to
  an action source (which also stops a bot choosing its own dice).
* **A single-option *player* decision is a step** and is recorded (`SettlePolicy.RECORD_FORCED`).
* **`state_snapshot` is the state after the action and after that drain.**
* **`flat_action_idx` is mandatory**: `_assert_reproduces`
  (`tests/replayer/test_replay_reproduces.py`) skips a replay without it, so omitting it retires
  the check rather than failing it.
* **Training records no replay**, so it keeps the C++ auto-advance (0.76× the wall time of the
  Python drain).
