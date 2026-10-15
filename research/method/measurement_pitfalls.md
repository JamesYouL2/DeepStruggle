# Before you trust a number — the checklist

The short list of ways an instrument here has lied, and the cheapest check for each. **Living
reference**: an entry is added for a new class of fault and removed only when the fault becomes
structurally impossible. Keep it under ~200 lines. Full histories:
[`../log/measurement_bugs.md`](../log/measurement_bugs.md),
[`../log/variance_and_noise.md`](../log/variance_and_noise.md), probe methodology in
[`../log/P9_architecture.md`](../log/P9_architecture.md).

> **Distrust the measurement before trusting the result.** Many diagnostics here reported confident,
> plausible, wrong numbers. When a result surprises, the cheapest first hypothesis is a broken
> instrument.

## The input the model was given

* **A wrong-width input is misread, not rejected.** With one base layout left (v2.3, 3,824 floats)
  the remaining risk is the optional `obs_features` blocks: a model must be fed its own set. *Check:*
  take the set from the model (`bindings.ts_env.model_obs_features`), and let `check_obs_width` /
  `_assert_width` (`tools/lib/batch_tournament.py`) refuse a mismatch. A probe that builds its own
  environment must do the same (`tests/training/test_probe_observation_layout.py`).
* **A chance node handed to a policy** picks its own dice (sequential and vectorized paths once
  disagreed by >25 points). *Check:* settle through `bindings/settle.py`; the two paths must be
  bit-identical on identical seeds.
* **A checkpoint that cannot load into the tournament path** leaves only the training loop's own
  numbers. If an arm never appears in a tournament report, ask why.

## The sample you actually drew

* **Survivorship in a batched probe:** stopping at `num_episodes < num_envs` samples the fastest
  envs. *Check:* first episode of every env, drained until all finish.
* **A corpus filtered by how games ended:** ts-replayer's finished logs show 1.7% DEFCON-1 endings
  against ITS's 11.7% (n = 44,136). Use ITS for length and endings ([`human_play.md`](human_play.md)).
* **`harvest()` calls `retire_stale()`:** a 1,000-position sample became 91.
* **A held-out split that permutes positions** leaks one game across the split; hold out whole games.
* **A sliding-window monitor is not a trend** (overlapping 40-iteration means turned a 48.5–58.0%
  oscillation into a "climb to 66%"). Bin first.

## The comparison

* **A single snapshot cannot resolve an effect below the run's oscillation** (~30 Elo); rate a block
  of late snapshots or the SWA. **A single cell is not a comparison** (one pairing gave a 64 Elo seed
  effect where sixteen gave 13).
* **Read a seat against the control's own self-play rate in that seat.** Greedy self-play gives the
  US 31.5–46% depending on the checkpoint, so "38% as US" can be level (E5-11-43 against E5-08-43:
  +1.0 as US, +9.5 as USSR). The overall paired rate is unaffected.
* **Per-seat rates come from the JSON, not the markdown table:** the report's pair is USSR first,
  and R39's per-seat figures were once read seat-swapped.
* **An endpoint read from self-play alone** can be satisfied by both sides drifting together (the
  pooling experiment's pre-registered side balance was uncorrelated with strength;
  [`../archive/E3_ladder/findings/pooling.md`](../archive/E3_ladder/findings/pooling.md) §3). Measure
  against an opponent that does not move.
* **One-sidedness is not degeneration.** `critic_base_rate` is `max(p, 1 − p)`: no direction, and a
  fact about the pair. E3-30-28's rose 0.61 → 0.79 while *both* seats improved against a frozen anchor
  (USSR 7.7 → 50.0%, US 4.7 → 23.0%). The instrument is per-seat win rate against a frozen opponent;
  a one-sided phase matters only if it stalls training.
* **A tournament is not reproducible from its configuration** (~1.5 points between identical
  6,000-game runs). **Elo is not comparable across tournaments** (one comparison read 60.50% and
  62.25% in two pools) — use the leaderboard's one scale, or measure the pair directly.
* **A leg's gain carries ±16 Elo** (one 80M leg: +86.9 on one seed, +54.7 on another).
* **`HeuristicBot` overrates weaker arms and saturates past ~120M** (all eight 4 × 4 arms > 89%):
  a wrong ordering, then no ordering that reads like a null.
* **Two points are not a trend; a null is only as wide as its window.** At 100 games a side a point
  carries ±4–5 pp. Quote window means with n, compare three windows, state a null's step range
  (three of nine claims retracted on 2026-09-18 were first-versus-last readings or windows ending
  before the phenomenon).
* **Wall-clock budgets** hand arms different training (contention is asymmetric); **wall-clock
  durations** lie when the host suspends — check `BOOTTIME − MONOTONIC` before trusting one.
* **Same filename, one player twice:** two `snapshot_final`s collided in Bradley-Terry, and
  `snapshot_final.pt` equals the last step snapshot. Compare tensors.
* **`sort -t_ -k2 -n` over underscore-laden paths sorts lexically**, so "late" snapshots were the
  earliest. Extract the step and print the selection.
* **A metric's definition can move under its series** (the blunder taxonomy was redefined
  mid-programme).
* **Temperature is part of the measurement.** Greedy and 0.1 are the same player (50.0% over 300
  games), 0.5 costs 134 Elo, 1.0 costs 374 ([`../archive/E3_ladder/log/P15_temperature_selfplay.md`](../archive/E3_ladder/log/P15_temperature_selfplay.md));
  control it whenever a treatment changes entropy. A broad sampled policy can be strong greedy and
  weak at T = 1 (R34: +7.6 greedy per seat, 36.6% at T = 1).

## The probe itself

* **The answer key must be absent from the observation.** `global_features[64..69]` carry the live
  regional VP differential; board slot 24 is `my_deficit`; a raw-input probe scored 0.886 AUC.
* **Gate every estimator on a known answer:** noiseless column → near-perfect recovery, pure noise →
  exactly the baseline (`tests/training/test_state_readout_probe.py`).
* **Ratios with near-zero denominators** (India's 4% headroom turned two points of noise into −431%):
  aggregate as `sum(acc − base) / sum(headroom)`.
* **Win classifiers:** ask whether the chosen action also wins rather than whether it is in a forced
  set; fold a multi-decision win into one opportunity (`fold_win_opportunities`); exclude headlines;
  every rule needs a test that it labels something (a Wargames rule compared against index `0` for a
  month); measure length in plies (a finished game ends on turn 11, a human log says 10).
* **A benchmark that never played a game** (wrong index space, every `step_flat` refused) reported a
  16% win. Drive with `ActionMask.generate_flat_mask`, assert the step succeeded, check games ended.
* **A representation probe moving the right way is not evidence of strength** (the best trunk-ladder
  arm halved play). **An aggregate can answer nothing** (hand AUC ~0.86 for every arm); build the
  restricted test where the shortcut is unavailable.

## The setup you think you ran

* **A flag can be accepted, recorded and printed and still not describe the run.** `--opponent-pool-size
  12` trained against a pool of **one** because resume globbed the wrong directory and fell through
  to the fresh-run branch; the arm beat its opponent 99.7% and fell to 24% against the anchor, and
  two training-dynamics mechanisms were investigated first
  ([`../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md`](../archive/E3_ladder/log/P15_X4b_collapse_is_pool_starvation.md)).
  A fallback right for one caller is a silent bug for another; check the instruments that describe
  the *setup* (`opp_pool_size`); a replay that fails to reproduce is a measurement.
* **A filename is not provenance.** `snapshot_0s.pt` in a resumed run is the inherited, fully trained
  model (bit-identical to the parent's last snapshot), once reported as "untrained" and used to explain
  a collapse backwards. Read `metadata.json` (`resumed_from`, `warmup_checkpoint`) first.
