# E6-04-43 — the adopted E6-04 recipe replicates on seed 43

**Arm.** E6-04-44's arm exactly, at seed 43: E6-03-43 resumed at 310M to 560M, with game-result
setup credit, the setup entropy floor 0.3, a 10M resume cadence and the league (main exploiter
E6-05-43-g). `launch_flags --diff` against E6-04-44 shows only the seed and the league directory.

**The league.**
* **Five exploiter generations; the fifth was stopped when the main agent finished.**
  * Generation 1 published five snapshots at 59–64%, never reaching the 65% reset bar.
  * Generation 2 published one, at 57%, which was USSR-only (63% as USSR, 50% as US).
  * Generations 3 and 4 published nothing: the first generations on either seed to find no exploit.
* The main agent's win rate against its 16-member pool stayed at about 0.69–0.70.

**Pre-registered.** Against E6-03-43 at the late block, 500–560M, per seat against the pinned panel,
on snapshots and on the 480–560M SWA, by the owner's rule. Openings and goal probes at 560M.

## Result

Round robin, greedy, 1,000 games per seat (`data/reports/e6_04_43_rr.{md,json}`). The panel figures
are E6-04-43 minus E6-03-43; ± is one standard error.

| reading | panel, US | panel, USSR | head to head (US / USSR) | Elo |
|:---|---:|---:|---:|:---|
| snapshots 500–560M | +0.1 ± 0.6 | **+6.9 ± 0.6** | **57.1%** (60.9 / 53.3) | 1543 vs 1495 |
| SWAs 480–560M | −1.1 ± 1.3 | **+7.1 ± 1.3** | **57.5%** (63.6 / 51.4) | 1639 vs 1584 |

**Accepted.** In both readings E6-04-43 is significantly better in one seat and not worse in the
other. Head to head it wins 57% on either reading.

**Openings at 560M** (`data/reports/e6_04_43_560m_setup_oracle_{us,ussr}.txt`):
* **US:** a mixed opening, West Germany 3 / France 3 / Italy 2 / Iran 1 (51%) or West Germany 2 /
  France 3 / Italy 2 / Iran 1 / South Korea 1 (49%). The best alternative, the human opening, is
  +1.6 ± 1.4 better, within the pre-registered 2 points.
* **USSR:** its own opening is best or level: the human opening is +0.2, Poland 3 / Yugoslavia 3
  is −1.4, and Poland 3 / Hungary 3 is −7.2.

**Goal probes at 560M**, E6-04-43 against E6-03-43 (`data/reports/e6_04_43_goal_probes.{md,json}`):

| measurement | E6-04-43 | E6-03-43 |
|:---|---:|---:|
| empty battlegrounds, turn 8 | 1.87 | 2.18 |
| card with its own exit spent on the space race | 10.4% | 13.6% |
| avoidable losses avoided | 92.8% | 91.0% |
| forced wins taken | 63.3% | 64.3% |

## Reading

* **The E6-04 recipe is now adopted on two seeds.**

  | seed | panel, US | panel, USSR | head to head | Elo |
  |:---|---:|---:|---:|---:|
  | 44 | +2.7 | +1.5 | 54.4% | +37 |
  | 43 | +0.1 | +6.9 | 57.1% | +48 |

  The SWAs agree: 57.5% on seed 43 and 58.0% on seed 44.
* **The seat that gains differs by seed.** Mostly US on seed 44, all USSR on seed 43. The recipe's
  gain is real, but not tied to one seat.
* **This is the first replicated adoption on E6.** Every P29 bet is judged against the plain E6-03
  control, which is unaffected.
