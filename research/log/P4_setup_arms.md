# P4: setup credit (arm A) and a human setup anchor (arm B), 2026-09-25

**Plan:** [`../plans/P4_setup_macro_action_credit.md`](../plans/P4_setup_macro_action_credit.md).

**Setup:** today's default recipe (E4-08 recipe, TF32, centred per-entity heads), 80M from
scratch, seeds 43 and 44, run as seed pairs.

| arm | runs | change |
|:---|:---|:---|
| control | E4-61-43, E4-61-44 | the same flags, from their 800M runs |
| A | E4-62-43, E4-62-44 | `--setup-block-lambda` |
| B | E4-63-43, E4-63-44 | human setup anchor: `--inject-setup-only`, every iteration, weight 1, 3,060 training setup placements |

`launch_flags.py --diff` against the controls shows only the arm's flags, plus `--train-steps`
(nothing is scheduled on it). All four runs finished without a crash. The only pin was E4-62-43
at 45–50M, one bucket, recovered.

## Setup stability (every 10M snapshot)

`data/reports/p4_setup_stability.{md,json}`: 1,000 openings per snapshot, temperature 0.1. Each
row is the target's rate at 10, 20, …, 80M.

| run | Poland ≥ 3 (USSR) | West Germany ≥ 4 (US) | all four targets |
|:---|:---|:---|:---|
| control E4-61-43 | 0 0 0 0 0 0 0 0 | 0 0 6 0 15 0 57 32 | 0 at every snapshot |
| control E4-61-44 | 98 41 100 100 100 100 100 5 | 100 0 5 0 0 0 3 0 | 96 at 10M, then ≤ 5 |
| A E4-62-43 | 100 at every snapshot | 0 at every snapshot | 0 |
| A E4-62-44 | 36 100 26 100 100 100 100 100 | 0 at every snapshot | 0 |
| B E4-63-43 | 100 at every snapshot | 70 82 60 97 100 100 83 100 | 70 75 60 96 97 100 83 100 |
| B E4-63-44 | 100 at every snapshot | 100 100 23 4 30 4 50 91 | 100 100 5 3 29 4 44 24 |

The human rate for all four targets is 57%.

* **Arm A fixes the USSR's Poland and nothing on the US side.** Poland is 100% at every
  snapshot of E4-62-43, and from 40M on in E4-62-44; the seed-43 control never took it. With
  credit flowing through the 6-placement USSR block, the right USSR opening is found and held.
  **West Germany is 0 in all 16 A snapshots.** With the credit fixed, this self-play meta's
  returns do not favour the human US opening.
* **Arm B holds the full human opening on seed 43**, at 60–100% from 10M, and 100% at 80M. **On
  seed 44 RL pulled West Germany away from ~30M despite the anchor**, down to 4–50%, back to 91%
  at 80M. At this dose the anchor is strong but not absolute, and self-play pushes against West
  Germany.

## Strength

`data/reports/p4_setup_80M.{md,json}`: 25 players, 100 games per side per pair, temperature 0,
HeuristicBot at 1500; E4-08-36@240M rates 2276 in this field.

| run | 40M | 60M | 70M | 80M | mean 60–80M |
|:---|---:|---:|---:|---:|---:|
| control E4-61-43 | 1841 | 1881 | 1995 | 2000 | 1959 |
| control E4-61-44 | 1830 | 1887 | 1877 | 1935 | 1900 |
| A E4-62-43 | 1775 | 1852 | 1939 | 2059 | 1950 |
| A E4-62-44 | 1919 | 1960 | 1957 | 2010 | 1976 |
| B E4-63-43 | 1863 | 1958 | 1974 | 2001 | 1978 |
| B E4-63-44 | 1785 | 1860 | 1981 | 1957 | 1932 |

**Head to head against the same seed's control**, late snapshots (60/70/80M, 9 pairings each):

| arm | seed 43 | seed 44 | mean |
|:---|:---|:---|:---|
| A | 48.8% (61/37), −8 | 59.1% (74/44), +64 | +27 |
| B | 53.0% (55/51), +21 | 52.1% (60/45), +15 | **+18** |

**The owner's question: does a proper setup change strength at this level? Not measurably.**
* Arm B, whose seed 43 plays the full human opening at 100% by 80M, is +21 and +15. Both seeds
  are positive and both are inside the noise. A human-like opening costs nothing and may be
  worth a little, but under ~40 Elo it is not measurable at two seeds.
* Arm A is +27 on average, spread −8 to +64.

## Side effects (goal probes at 40/80M, `data/reports/p4_goal_probes.{md,json}`)

| at 80M | control 43 | control 44 | A 43 | A 44 | B 43 | B 44 |
|:---|---:|---:|---:|---:|---:|---:|
| empty battlegrounds, turn 8 | 3.36 | 2.37 | **1.60** | **1.66** | 1.59 | 3.75 |
| uncontrolled battlegrounds, turn 8 | 6.96 | 5.85 | **2.92** | 5.10 | 4.46 | 7.50 |
| forced wins taken | 67.4% | 68.0% | 71.2% | 61.9% | 63.8% | 64.6% |

**Arm A contests more of the board at 80M on both seeds:** 1.6–1.7 empty battlegrounds at turn 8
against 2.4–3.4. That fits λ = 1 on the setup extending to any consecutive placement block of the
same mover at the opening, and a better USSR opening. It is one budget, and 40M does not show it
cleanly (A 3.33 and 1.60), so it is a lead, not a result. Forced wins are unchanged
everywhere.
