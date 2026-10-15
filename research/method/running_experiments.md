# Running an experiment so its result means something

How an arm is budgeted, run and rated. **Rewritten in place**; the history of why is in
[`../log/variance_and_noise.md`](../log/variance_and_noise.md) and
[`../log/measurement_bugs.md`](../log/measurement_bugs.md), the ways an instrument lies in
[`measurement_pitfalls.md`](measurement_pitfalls.md). Launching itself (deriving flags from a
healthy run, `launch_flags.py --diff`, the opponent pool) is `CLAUDE.md`'s invariants 14–15.

## How to rate an arm

* **Measure twice (owner, 2026-09-29):** the late snapshots against the pinned panel, and the SWA of
  the last 80M (`tools/scripts/average_weights.py`, one snapshot per 10M). SWAs are an evaluation
  device; averaging (`--ema-weights`) stays out of training.
* **Accept per seat against the control's own self-play rate in that seat** (owner, amended
  2026-09-29): significantly above the bar in at least one seat and not significantly below it in
  the other. A raw head-to-head seat rate means nothing: the US seat scores ~45–46% in greedy
  self-play.
* **Greedy, 1,000 games a side** for a head to head or a field; read per-seat numbers from the
  JSON (`per_side`, `win_rate_a_as_us`), not the markdown table, whose pair is USSR first.
* **Ratings go on the leaderboard** (`tools/leaderboard.py`, one Elo scale per engine epoch,
  anchored at `E7-A4-R1-S44@4800M = 1500`; `tools/README.md` §2b). Never compare Elo across separate
  tournaments; for a specific comparison measure the pair directly.
* **The noise floor:** two identical 6,000-game tournaments differ by ~1.5 points on a matchup
  (agents sample), more than the binomial SE suggests. A single snapshot also carries the run's own
  oscillation (~30 Elo); rate a block of late snapshots or the SWA.
* **Seeds:** two or three for effects above ~40 Elo; one seed decides nothing below that. From
  scratch, two seeds of one recipe were 55 Elo apart at 1,200M and level at 2,000M (the gap was pace).
* **Judge a recipe at its plateau**, or as a late branch from a plateau
  ([`../findings/training/training_length_and_evaluation.md`](../findings/training/training_length_and_evaluation.md)).
* **Do not rate against `HeuristicBot`:** it overrates weaker arms and has reversed a sign. No live
  training metric ranks arms.

## Budgets and machine

* **Budget by `--train-steps`, never wall clock** — steps/s depends on the policy.
* **At most two training runs at once** (owner). Two shallow arms together give +20% total steps/s;
  three add nothing. Contention is not symmetric, another reason not to budget by time. The host's
  throughput drifts ~±7% over a day.
* **Give every model in a tournament a distinct name**; two `snapshot_final`s once collided in the
  Bradley-Terry fit.

## What a step is: settling

`bindings/settle.py` is the one definition of how far the engine advances past decisions nobody
controls: `NONE`, `CHANCE` (die rolls only — a policy asked for a chance node picks its own dice) or
`FORCED` (chance nodes, single-legal-action decisions, deterministic event targets; the default for
training environments and tournaments, `tools/tournament.py --no-auto-advance` to opt out). A
training "step" is a decision the policy was *asked about*, so the mode changes what a step budget
covers (~4.2% fewer steps per game under `FORCED`). Outcomes are identical per decision stream
(`tests/training/test_auto_advance_outcome_equivalence.py`), but a sampling policy's later draws
shift, so games are statistically, not bit-for-bit, equivalent. The mode is part of the measurement.

## Smaller rules

* **Game length in plies, not turns** (`ai/game_length.py`; 154 = all ten turns). The turn counter
  is coarse and a finished game ends holding turn 11 while a human log says 10.
* **`harvest()` calls `retire_stale()`** (`ai/training/start_pool.py`): take positions out after each
  round or older generations vanish (a 1,000-position sample once shrank to 91).
