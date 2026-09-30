# P29 bet 1 — a 3.5× larger trunk is not better at 560M and is worse at matched wall-clock

Plan: [`../plans/P29_big_bets_from_scratch.md`](../plans/P29_big_bets_from_scratch.md), bet 1.

**Arm.** E6-10-44 is E6-03-44's recipe and seed with `--ladder-hidden-dim 768 --ladder-res-blocks 8`:
11.19M parameters against 3.18M. The residual trunk is 5.1× larger; the encoders and heads are
unchanged. It ran from scratch to 560M, solo. `launch_flags --diff` shows only the two flags.

**Speed.** The median over the run is 54.5k steps/s, against E6-03-44's 82k, so **r = 0.665**. The
matched-wall-clock point is therefore 560M × r ≈ **370M**.

**Pre-registered.** Against E6-03-44, per seat against the panel, on snapshots and on the 80M soup
(the owner's rule):
* at matched games, the late block 500–560M;
* at matched wall-clock, E6-10-44@370M and its 290–370M soup against E6-03-44@560M and its 480–560M
  soup.

Adopt only at matched wall-clock.

## Results

Round robin, greedy, 1,000 games per seat (`data/reports/p29_bet1_rr.{md,json}`). The panel figures
are bet minus control; ± is one standard error.

| comparison | panel, US | panel, USSR | head to head (US / USSR) |
|:---|---:|---:|---:|
| **matched games**, snapshots 500–560M | **−2.4 ± 0.6** | **−10.4 ± 0.6** | 51.3% (58.5 / 44.1) |
| **matched games**, soups 480–560M | **−7.1 ± 1.3** | **−11.9 ± 1.3** | 45.4% (50.5 / 40.3) |
| **matched wall-clock**, @370M against @560M | −9.0 ± 1.3 | −13.1 ± 1.3 | 44.9% (54.0 / 35.7) |
| **matched wall-clock**, soup 290–370M against soup 480–560M | −10.0 ± 1.3 | −19.5 ± 1.3 | 37.6% (43.6 / 31.7) |

**Elo.** Late snapshots 1522 against 1519, level. Soups 1566 against 1597.

**Midway, same step** (`data/reports/p29_bet1_midcheck.{md,json}`):

| step | bet 1 vs control |
|:---|---:|
| 80M | 43.5% |
| 160M | 47.7% |
| 240M | 46.3% |

**Goal probes at 560M** (`data/reports/p29_bet1_goal_probes.{md,json}`):
* **A different US opening:** Iran 2 and West Germany below 4, where the control plays West
  Germany 4.
* **Card disposal is worse:** a card with its own exit spent on the space race 11.5% of the time
  against 1.1%.
* **Fewer forced wins taken:** 53.8% against 58.6%.
* **Fewer empty battlegrounds at turn 5:** 3.2 against 4.5.

## Verdict: bet 1 not adopted

* **At matched games it is no better.** Snapshots are level on Elo and slightly ahead head to head,
  but behind the control against the panel in both seats. The soup is behind on every reading.
  * The panel is E6-03-44's own lineage, which may favour the control a little. Even the
    lineage-free readings (Elo; head to head between soups) show no gain.
* **At matched wall-clock it is clearly worse:** 45% (snapshots) and 38% (soups).
* **By P29's rule, capacity is not what holds the plateau** on this seed and recipe. A 3.5× larger
  net does not level off higher at 560M, and costs a third of the throughput. The capacity
  question closes for M2d-style trunks. M3, card↔card attention, is a different mechanism and
  stays open.
