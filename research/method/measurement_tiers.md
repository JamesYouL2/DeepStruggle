# Measurement tiers — what runs during training, per snapshot, and once per arm

Place a probe by measured cost, not by interest: in the wrong tier it either starves a run of signal
or halves its throughput. **Tier on the artifact as well as the compute** — a probe that costs 0.1%
of a step but emits 880 TensorBoard series (play modes per card × side × mode) is tier 3.

**The budget.** The loop's cost is per vectorized step, multiplied by the number of envs:
`steps_per_sec` counts *env-steps*, so divide by the env count first (getting this wrong once made
every probe look 512× too expensive). Scalar reads and `decode_flat_action` are ~0.1 µs;
`defcon_suicide_cards` walks all 84 countries (~45 µs), so it can run on terminated envs only
(~1% of env-steps), never per env per step.

## Tier 1 — every iteration (under ~1% of a step)

* **Critic discrimination: `critic/auc`, `critic/brier_skill`, `critic/auc_turn3`, `critic/base_rate`**
  (`ai/training/generic_trainer.py`). One state per game per turn (the turn's first decision, US
  perspective), held until the result is known. It catches the transition where one side's
  unanswered strategy makes outcomes predictable and the critic decays toward the base rate
  (E3-17-22, E3-15-22). **`explained_variance` cannot:** with `G = A + V` it rises toward 1.0 exactly
  as advantages collapse (0.996 across E3-17-22's failure). Accuracy is dominated by the base rate
  and moved the wrong way (79.2 → 87.5% while AUC fell 0.865 → 0.728). Keep both: AUC is ranking
  only; **`brier_skill ≤ 0` is the alarm** (no better than a constant). Read it smoothed.
* **DEFCON-1 endings** are logged as `ending_frac_defcon1_self` / `_provoked`. The four-way split
  (own goal, bad bet, forced trap, unforced trap, headlines apart) is `ai/eval/defcon_endings.py`;
  the live in-loop version specified here earlier (`live/defcon1_*`, ~6% of the budget for the
  forced/unforced test) was not built. Never quote an in-loop rate (on-policy, shifting denominator)
  against a fixed-sample probe rate.
* **`eval/win_rate_vs_HeuristicBot`** is a sanity check, not a ranking: every model should win, and a
  value that stops being high means something broke.

## Tier 2 — per snapshot (seconds)

Run by the trainer on its own self-play games, **at a low temperature** (a probe at T = 1.0 once
read 60.3% DEFCON-1 endings against the 31.4% training logged): blunder rates
(`measure_blunders_batched`), the decisive probe (forced-win take rate; decisive choices are ~0.6% of
decisions, so win rate barely sees them), position diagnostics (empty / uncontrolled battlegrounds at
turns 5 and 8, ply distribution). A new probe first reproduces a number the loop already logs.

## Tier 3 — once per arm (minutes)

Trunk recoverability (`state_readout.py`, ridge probes over ~19k positions), the play-mode table per
card and side (`play_modes.py`), the behaviour report (`log/per_checkpoint/`), and the rating itself:
late-snapshot and SWA head-to-heads against the control and fixed references, and the leaderboard
for a candidate ([`running_experiments.md`](running_experiments.md)). Rate against a **fixed core**
plus additions, not all pairs: an all-pairs pool once spent two hours on ~1,400 pairings nobody read.
