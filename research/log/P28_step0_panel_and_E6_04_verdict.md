# P28 step 0 — the E6 panel, and E6-04-44 adopted

Plan: [`../plans/P28_strength_on_E6.md`](../plans/P28_strength_on_E6.md), step 0. Owner, 2026-09-29:
"proceed with 1st step from it to check E6-04 adaptation under new rules".

## The trace and the panel

**E6-03-44's trace** (`data/reports/p28_e6_03_44_trace.{md,json}`): a round robin of every 40M
snapshot to 440M and every 10M snapshot from 480M to 560M, greedy, 200 games per seat.

| step | 40M | 80M | 120M | 160M | 200M | 240M | 280M | 320M | 360M | 400M | 440M |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elo | 1116 | 1241 | 1334 | 1344 | 1420 | 1447 | 1476 | 1550 | 1552 | 1553 | 1582 |

| step | 480M | 490M | 500M | 510M | 520M | 530M | 540M | 550M | 560M |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elo | 1600 | 1597 | 1594 | 1588 | 1582 | 1586 | 1606 | **1617** | 1615 |

* **Plateau.** The run levels off at about 320M (1550), then gains about 35 Elo from 440M to 560M.
* **No end dip.** Unlike every E5 lineage on record, this seed does not dip at the end on E6.

**The panel** is pinned in `data/reports/p28_panel.json`, with each member's sha256:

| role | member | Elo | why |
|:---|:---|---:|:---|
| weak | E6-03-44@80M | 1241 | about 360 below the plateau |
| mid | E6-03-44@240M | 1447 | |
| peak | E6-03-44@550M | 1617 | the trace peak; not the final snapshot, as the plan requires |
| out-of-lineage | `heuristic_mcts:16` | — | |

`heuristic_mcts:16` is at the floor for late snapshots: it loses 97–99% of its games, so it carries
almost no signal in the per-seat reading.

**The late block** is 500, 520, 540 and 560M for both arms, kept apart from the 550M panel peak.

## E6-04-44 against E6-03-44

Neural games were greedy, 1,000 per seat per pairing (`data/reports/p28_step0_rr.{md,json}`).
Games against `heuristic_mcts:16` were 200 per seat (`p28_hmcts_*.md`).

**By the panel rule.** For each seat, the late snapshots' mean result against the panel, E6-04
minus E6-03. The ± values are one standard error.

| seat | E6-04-44 | E6-03-44 | difference (neural panel) | difference (+ `heuristic_mcts:16`) |
|:---|---:|---:|---:|---:|
| **US** | 67.9% | 65.2% | **+2.7 ± 0.6** | +1.8 ± 0.5 |
| **USSR** | 74.6% | 73.0% | **+1.5 ± 0.5** | +1.4 ± 0.4 |

Per member, averaged over the four late snapshots:

| against | E6-04 as US | E6-03 as US | E6-04 as USSR | E6-03 as USSR |
|:---|---:|---:|---:|---:|
| E6-03-44@80M (weak) | 77.8 | **85.7** | 94.0 | 92.9 |
| E6-03-44@240M (mid) | **71.4** | 65.0 | **74.2** | 73.2 |
| E6-03-44@550M (peak) | **54.4** | 44.9 | **55.6** | 53.0 |
| `heuristic_mcts:16` | 97.1 | 98.1 | 99.4 | 98.4 |

**By the self-play-bar rule**, over the 16 head-to-head pairings: 54.4% overall, US +9.6 ± 0.9, USSR
+0.3 ± 0.9. The bars are E6-03-44's own twin self-play at 500/520/540/560M: US 42.7 / 44.0 / 44.2 /
45.3.

Elo, same field: E6-04-44 is 1549 / 1573 / 1562 / 1560 and E6-03-44 is 1520 / 1515 / 1534 / 1551.

## Goal probes and openings at 560M

Probes from `data/reports/p28_goal_probes.{md,json}`:

| measurement | E6-03-44@550M | E6-03-44@560M | **E6-04-44@560M** |
|:---|---:|---:|---:|
| empty battlegrounds, turn 8 | 1.72 | 1.49 | **0.51** |
| empty battlegrounds, turn 5 | 4.23 | 4.24 | **2.56** |
| uncontrolled battlegrounds, turn 8 | 5.14 | 4.58 | **4.36** |
| space exit spent on own/neutral card (τ 0.1) | 7.2% | 5.6% | **1.8%** |
| own-DEFCON loss with an alternative (τ 0.1) | 1.5% | 1.2% | 1.5% |
| forced wins taken | 57.6% | 57.1% | 57.9% |
| avoidable losses avoided | 94.2% | 94.6% | 93.4% |

**Openings** are from [`E6_04_league_seed44.md`](E6_04_league_seed44.md). Both models play the same
opening in each seat:

* **US:** West Germany 4, France 3, Italy 2.
  * The human opening is +2.1 ± 1.4 better for E6-04 and +3.6 ± 1.5 for E6-03.
  * E6-04's critic is closer to calibrated: −4.7 against −11.3.
* **USSR:** Poland 3, Hungary 3, the best of the alternatives for both models.

## Verdict: E6-04-44 adopted

* **Under P28's decision rule,** "beats E6-03 in both seats by the panel rule", both seats are
  significantly positive: US +2.7 ± 0.6 and USSR +1.5 ± 0.5. It is also accepted under the owner's
  amended rule (one seat up, the other neutral) by either measure.
* **The self-play bars misattributed the gain.**
  * Against E6-03's own self-play, all of E6-04's advantage looked like US (+9.6), with the USSR
    seat at zero.
  * Against a fixed panel, E6-04 is better in both seats, with more of the gain in the US seat.
  * This is the mechanism the plan named: the control's USSR is specialised against its own weak
    US, which inflates the USSR bar. E5-21's "US +11.8, USSR −0.8" was very likely the same
    artefact.
* **One exception to the overall edge: E6-04 as US against the weak rung.** E6-04 plays US worse
  than E6-03 against the 80M model (77.8% against 85.7%), and better against every stronger member.
  A league-trained US has given up some of the lines that punish a weak opponent. That seat is a
  panel artefact to watch, not a regression that matters at strength.
* **The behaviour matches the result.**
  * E6-04 leaves a third as many battlegrounds empty at turn 8 (0.51 against 1.5–1.7), and fewer
    at turn 5.
  * It rarely spends a card that has its own exit on the space race (1.8% against 5.6–7.2%).
  * DEFCON discipline and forced-win conversion are unchanged.
* **Consequence (P28):** E6-04-44's recipe is what steps 2–5 build on and control against. That
  recipe is credit + floor + league from 310M, on top of E6-03's.
