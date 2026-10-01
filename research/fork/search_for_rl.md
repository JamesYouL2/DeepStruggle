# Would deeper search make a better RL teacher? (2026-10-01, analysis)

**Short answer: not deep search online. Targeted deep search offline, and a better small-budget
target online, are the two versions worth running.** Nothing here has been run; it assembles what
is measured.

## What is measured

| fact | number | where |
|:---|:---|:---|
| cost of online search distillation | ~1,200 steps/s with 64-sim search on 1 in 8 decisions, against ~45,000 without | [`../log/E4_search_distillation.md`](../log/E4_search_distillation.md) |
| what it bought on E4 | +54 to +262 Elo over a step-matched control, as a level reached within 5M | same |
| what 64-sim search adds over E6 at play time | +12 to +20 Elo | [`gumbel_root.md`](gumbel_root.md) |
| how far the 64-sim target is from the prior | same argmax 98.8%; never different once the prior is above 0.8 | [`determinization_targets.md`](determinization_targets.md) |
| budget at which search leaves Hungary | ~30,000 simulations; value gap ~0.02 even at 65k | [`gumbel_root.md`](gumbel_root.md) |
| search saturation on E3 / E5 | 96 sims (E3); 64 ≈ 128 (E5) | [`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md), [`../log/E5_search_budget_sweep.md`](../log/E5_search_budget_sweep.md) |

## Why not deep search online

* **Cost.** 30,000 simulations is ~470× the 64-sim search that already made training ~40× slower.
* **Saturation.** On every net measured, play strength stops improving by 64–128 simulations; the
  extra depth changes a decision only where values differ by a point or two.
* **Its verdicts are the value head's.** Search backs up the critic. A 0.02 gap at 65k is inside
  the critic's known errors (it mispriced openings by 10–26 points in E5; own-DEFCON losses survive
  search). Deep search would teach those errors with confidence.

## What would be worth running

1. **Offline deep search for the decisions played once a game.** The setup is one decision per
   side per game, locked at p ≈ 1, and small differences there persist for a whole run
   ([`hungary_openings.md`](hungary_openings.md)). A few hundred deals × ~30k simulations, checked
   against paired-deal playouts (not the critic), gives an opening table usable as a forced opening
   or a setup prior. Cheap on CI, and it touches no training code until the table is trusted.
2. **A small-budget target that can leave the prior.** Gumbel MuZero's completed-Q improved policy
   at 16–64 simulations: the target moves toward actions with higher searched value even when
   their prior is low, with a policy-improvement guarantee. The Gumbel root exists
   (`b249e6c`, `feat/mcts-gumbel`); the target does not. First check before any training arm: rerun
   [`determinization_targets.md`](determinization_targets.md)'s probe with that target and see
   whether it departs from the prior where PUCT's does not.
3. **Keep E4-28's guard-rails** if either goes into training: the search CE term collapsed E3 with
   or without a healthy pool, so the tripwires in
   [`../log/E4_search_distillation.md`](../log/E4_search_distillation.md) apply, and
   `launch_flags.py --diff` against a healthy run before launch.
