# How far E4's best checkpoints are from the goal, criterion by criterion

The goal ([`../plans/README.md`](../plans/README.md)) is a no-search player at the level of a
mediocre human. It is defined by behaviours, not Elo: a sane setup, contesting battlegrounds, not
losing to one's own DEFCON, and disposing of a card through the exit that card has. This measures
each on the checkpoints that matter now, with `tools/scripts/goal_probes.py`: the same probe
functions training runs at every snapshot, run on finished checkpoints. The human corpus (the
validated 300 files, 264 games with a readable setup) is the yardstick where one exists.

Report: `data/reports/goal_probes_2026-09-23.{md,json}`. Temperature 0.1 unless marked; setup
over 2,000 openings, positions over 256 self-play games, blunders over 128, decisive over 256.

| measurement | human corpus (264 games) | E4-31-03@240M | E4-28-03@200M | E4-08-03-160M.11@240M | E4-08-03@240M | E4-08-03@160M | E4-28-05@105M | E4-03-01@80M |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| setup/USSR Poland >= 3 | 99.2% | 100.0% | 100.0% | 100.0% | 99.8% | 100.0% | 100.0% | 0.0% |
| setup/US West Germany >= 4 | 61.4% | 79.5% | 100.0% | 83.9% | 0.0% | 98.9% | 97.2% | 0.0% |
| setup/US Italy >= 2 | 95.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| setup/US Iran >= 2 | 96.2% | 45.9% | 0.0% | 16.1% | 100.0% | 25.5% | 0.0% | 100.0% |
| **setup/all targets** | **57.2%** | 25.4% | 0.0% | 0.0% | 0.0% | 24.4% | 0.0% | 0.0% |
| empty battlegrounds, turn 8 | — | 0.93 | 0.80 | 2.03 | 2.05 | 1.17 | 1.80 | 2.29 |
| empty battlegrounds, turn 5 | — | 4.75 | 3.19 | 7.08 | 6.58 | 4.47 | 3.92 | 6.06 |
| uncontrolled battlegrounds, turn 8 | — | 3.24 | 3.20 | 5.02 | 4.60 | 3.51 | 4.75 | 4.98 |
| salvageable at turn 6, given reached | — | 70.4% | 67.8% | 68.0% | 60.6% | 57.3% | 67.1% | 66.5% |
| own-DEFCON loss with an alternative (τ 0.1) | — | 1.1% | 0.9% | 2.5% | 4.3% | 2.6% | 0.8% | 2.9% |
| own-DEFCON loss with an alternative (τ 1) | — | 7.6% | 1.3% | 4.6% | 13.7% | 3.6% | 1.6% | 5.4% |
| space exit spent on own/neutral card (τ 0.1) | — | 8.5% | 33.3% | 11.9% | 13.3% | 16.2% | 46.3% | 6.4% |
| **forced wins taken** | — | 58.9% | 60.4% | 69.6% | 58.7% | 66.7% | 59.1% | 61.6% |
| avoidable losses avoided | — | 94.2% | 91.1% | 95.1% | 93.9% | 94.9% | 90.8% | 94.0% |

## Reading, criterion by criterion

* **Setup: Poland is solved, the US opening is not.** Every current checkpoint takes Poland; the
  US half is where they fail, and differently each. The composite (all four targets in one
  opening) is at best **25% against the human 57%**. The failing target is **Iran**, which humans
  take 96% of the time and most checkpoints take 0–46% — not West Germany, which humans take only
  61% of the time. The same lineage swings between snapshots (`E4-08-03`: Iran 25% at 160M, 100%
  at 240M, West Germany 99% then 0%), which is the setup oscillation of
  [`P21_M2d_setup_west_germany.md`](P21_M2d_setup_west_germany.md) seen across targets.
* **Battlegrounds: much better than the goal statement's "~7.5 empty from turn 8"**, which
  described an older lineage. The best checkpoints leave under one battleground empty at turn 8.
  No human yardstick exists for this yet; three or more *uncontrolled* at turn 8 is the larger gap.
* **Own DEFCON: close.** About 1% at the evaluation temperature for the best checkpoints. At τ 1
  the slow-π_ref arm is at 7.6%, so the policy's own distribution still carries the mistake.
* **Card disposal: search distillation made it worse.** Spending the space exit on a card that
  had its own exit runs 33–46% for both search arms against 8.5–16% for the rest. The searcher,
  and the net that distils it, reaches for the space race.
* **Forced wins: the clearest "simple mistake".** Every checkpoint takes an immediately winning
  action only **59–70%** of the times one is available. A mediocre human takes it every time.
  This is the largest single gap between these checkpoints and the goal's "no simple mistakes".

## What it implies

* **E4-31-03@240M (late slow π_ref) is the best or near-best on most criteria**, consistent with
  its tournament result ([`E4_late_dynamics.md`](E4_late_dynamics.md)).
* **More games is not obviously what is missing.** Forced wins are a local, one-decision failure
  that 240M steps have not fixed, and the setup oscillates rather than converging. Both point at
  the training signal for rare or early decisions rather than at scale.

## 2026-09-25: the long runs' best checkpoints against the goal

Report: `data/reports/goal_probes_2026-09-25.{md,json}`, with the same settings as above.
E4-61-44@720M is the strongest model rated (2374, `long_center_800M`). E4-57-44@590M is next
(2349), then E4-08-36@240M (2297).

| measurement | human | E4-61-44@720M | E4-61-44@800M | E4-61-43@800M | E4-57-44@590M | E4-56-43@800M | E4-08-36@240M |
|:---|---:|---:|---:|---:|---:|---:|---:|
| setup/USSR Poland >= 3 | 99.2% | 100.0% | 100.0% | **0.0%** | 100.0% | 100.0% | 100.0% |
| setup/US West Germany >= 4 | 61.4% | 0.0% | 0.0% | 49.1% | 98.7% | 7.1% | 86.9% |
| setup/US Iran >= 2 | 96.2% | 100.0% | 91.5% | 100.0% | 76.4% | 87.5% | 13.1% |
| **setup/all targets** | **57.2%** | 0.0% | 0.0% | 0.0% | **75.1%** | 0.1% | 0.0% |
| empty battlegrounds, turn 8 | — | 2.47 | 2.83 | 3.91 | 2.46 | 2.28 | **1.58** |
| uncontrolled battlegrounds, turn 8 | — | 6.71 | 8.91 | 9.41 | 8.12 | 6.25 | **4.74** |
| own-DEFCON loss with an alternative (τ 0.1) | — | 2.3% | 1.3% | 1.5% | 1.6% | 2.9% | 1.9% |
| space exit on own/neutral card (τ 0.1) | — | 9.9% | 3.2% | 11.4% | 9.9% | 8.8% | 5.7% |
| **forced wins taken** | — | 65.8% | 63.0% | 62.0% | 66.0% | 65.6% | 65.5% |
| avoidable losses avoided | — | 95.3% | 95.6% | 93.1% | 96.2% | 93.7% | 93.5% |

**Elo improved, the goal criteria did not:**
* **Setup still oscillates.** Two snapshots of one run, 80M apart, take West Germany 0% and 0%.
  E4-61-43@800M has *lost Poland* (0%), which every checkpoint measured before had solved.
  E4-57-44@590M happens to hit all four targets 75% of the time, above the human 57%. It is one
  snapshot of an oscillation, not a solved criterion.
* **Battleground coverage got worse with long training.** The 560–800M checkpoints leave 2.3–3.9
  battlegrounds empty and 6.3–9.4 uncontrolled at turn 8. E4-08-36@240M leaves 1.6 and 4.7, and
  E4-31-03@240M left 0.9 and 3.2 on 09-23. The long runs gained Elo while contesting less of the
  board.
* **Forced wins are stuck at 62–66%** in every checkpoint, from 240M to 800M, as they were at
  59–70% on 09-23. It remains the largest simple mistake.
* **DEFCON and card disposal are close to fine** at the evaluation temperature: 1.3–2.9% and
  3–11%.

**What it implies.** Training time and the stability fixes bought Elo, not the named behaviours.
Every one of the goal's criteria either oscillates or is flat under more training, so reaching
the goal needs levers aimed at the signal for those decisions, not more steps.
