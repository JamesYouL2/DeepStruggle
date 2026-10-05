# E7 shallow line to 2,000M (2026-10-04): strength every 40M

The owner asked how strength changes up to 2,000M steps, every 40M, after E7-17-44 continued
E7-02-44 from 1,200M to 2,000M with its flags ([`../runs.md`](../runs.md)). The line is one run:
E7-01-44 (0 → 560M), E7-02-44 (560 → 1,200M), E7-17-44 (1,200 → 2,000M), all seed 44, constant
learning rate.

**Method.** One round robin of the 41 snapshots from 400M to 2,000M, the E6-03-44 panel
(80/240/550M) and HeuristicBot, 400 games per side per pairing, temperature 0, anchored at
HeuristicBot = 1500 (`data/reports/e7_trace_400_2000M.{md,json}`). Engine d81e54f0 (bindings
changed for the C++ search; game logic identical to 1d11c2f2). "Panel" is the mean win rate
against the three E6-03-44 snapshots in that seat.

| M | run | Elo | panel, US | panel, USSR |
|---:|:---|---:|---:|---:|
| 400 | E7-01-44 | 2333 | 66.2 | 68.3 |
| 440 | E7-01-44 | 2366 | 68.9 | 69.2 |
| 480 | E7-01-44 | 2366 | 69.7 | 70.3 |
| 520 | E7-01-44 | 2378 | 71.6 | 72.7 |
| 560 | E7-01-44 | 2392 | 72.5 | 71.7 |
| 600 | E7-02-44 | 2409 | 74.2 | 74.2 |
| 640 | E7-02-44 | 2412 | 76.2 | 76.3 |
| 680 | E7-02-44 | 2446 | 76.8 | 76.6 |
| 720 | E7-02-44 | 2446 | 79.2 | 76.1 |
| 760 | E7-02-44 | 2455 | 78.0 | 77.4 |
| 800 | E7-02-44 | 2469 | 79.1 | 78.8 |
| 840 | E7-02-44 | 2480 | 78.5 | 80.2 |
| 880 | E7-02-44 | 2487 | 82.2 | 80.9 |
| 920 | E7-02-44 | 2487 | 81.7 | 79.4 |
| 960 | E7-02-44 | 2491 | 82.6 | 78.9 |
| 1,000 | E7-02-44 | 2508 | 82.8 | 81.0 |
| 1,040 | E7-02-44 | 2497 | 81.4 | 80.5 |
| 1,080 | E7-02-44 | 2504 | 82.3 | 79.7 |
| 1,120 | E7-02-44 | 2512 | 82.8 | 83.2 |
| 1,160 | E7-02-44 | 2513 | 81.0 | 81.1 |
| 1,200 | E7-02-44 | 2519 | 84.6 | 83.0 |
| 1,240 | E7-17-44 | 2511 | 81.7 | 81.9 |
| 1,280 | E7-17-44 | 2528 | 83.8 | 82.0 |
| 1,320 | E7-17-44 | 2525 | 84.0 | 85.0 |
| 1,360 | E7-17-44 | 2530 | 85.4 | 83.2 |
| 1,400 | E7-17-44 | 2532 | 86.1 | 82.1 |
| 1,440 | E7-17-44 | 2534 | 86.8 | 84.0 |
| 1,480 | E7-17-44 | 2525 | 83.4 | 82.5 |
| 1,520 | E7-17-44 | 2532 | 86.3 | 82.6 |
| 1,560 | E7-17-44 | 2543 | 86.5 | 83.0 |
| 1,600 | E7-17-44 | 2549 | 86.1 | 81.2 |
| 1,640 | E7-17-44 | 2550 | 85.2 | 85.7 |
| 1,680 | E7-17-44 | 2550 | 85.3 | 83.0 |
| 1,720 | E7-17-44 | 2529 | 84.7 | 83.6 |
| 1,760 | E7-17-44 | 2548 | 86.3 | 83.3 |
| 1,800 | E7-17-44 | 2539 | 84.4 | 84.4 |
| 1,840 | E7-17-44 | 2560 | 86.2 | 84.0 |
| 1,880 | E7-17-44 | 2562 | 88.2 | 83.8 |
| 1,920 | E7-17-44 | 2564 | 85.9 | 85.2 |
| 1,960 | E7-17-44 | 2566 | 85.2 | 85.0 |
| 2,000 | E7-17-44 | 2561 | 85.9 | 85.8 |

Panel members in the same field: E6-03-44@80M 2067, @240M 2228, @550M 2358.

## Reading

* **The recipe has not saturated at 1B; it slows sharply.** +175 Elo from 400M to 1,000M, about
  +10 from 1,000M to 1,200M -- which is what the earlier "saturates at about 900M–1B" reading was
  based on -- and then **+54 more from 1,200M to 2,000M** (five-snapshot means: 1,040–1,200M 2509,
  1,840–2,000M 2563). Against the panel both seats rise from about 82/81% to 86/85%.
* **The gain is slow and roughly steady, about 5–7 Elo per 100M, flattening from about 1,850M**
  (1,840–2,000M all within 2560–2566). Individual snapshots scatter by ±10–20 Elo, so only the
  averaged trend is readable; the SWA/snapshot comparison by the owner's rule at 2,000M against
  E7-02-44@1,200M is still to be run.
* For the base-model question: at 1,200M the recipe is not at its ceiling. Arms judged at 1,200M
  (P30's protocol) are compared at a point where the control itself is still worth ~50 Elo more
  training, which leaves the comparison fair (same budget) but not "at saturation".

## Against fixed opponents: the 1,000M snapshot, the E6 panel and the soup (2026-10-04)

The owner asked whether the gain past 1,200M shows only against the line's own earlier snapshots or
also against fixed baselines. From the same round robin, each snapshot against E7-02-44@1000M and
against the strongest panel member, E6-03-44@550M (800 games each, ± 1.8); and, in separate
two-model matches (`data/reports/vs_soup/`, 400 games per side, temperature 0), against the best
model, the shallow soup E7-02/03/04/05@1200M:

| M | vs E7-02-44@1000M | vs E6-03-44@550M | vs the soup (US / USSR) |
|---:|---:|---:|---:|
| 1,000 | 50.0 | 70.9 | 36.0 (32.8 / 39.2) |
| 1,040 | 48.6 | 68.6 | 33.0 (34.2 / 31.8) |
| 1,080 | 53.6 | 66.8 | 36.2 (35.8 / 36.8) |
| 1,120 | 47.4 | 70.5 | 33.8 (33.5 / 34.0) |
| 1,160 | 49.6 | 66.2 | 36.1 (38.2 / 34.0) |
| 1,200 | 51.3 | 71.9 | 36.1 (38.2 / 34.0) |
| 1,240 | 50.0 | 71.2 | 38.2 (39.8 / 36.8) |
| 1,280 | 55.2 | 70.0 | 36.6 (34.0 / 39.2) |
| 1,320 | 54.0 | 74.9 | 39.4 (34.2 / 44.5) |
| 1,360 | 55.2 | 76.4 | 41.0 (39.8 / 42.2) |
| 1,400 | 52.9 | 73.6 | 40.0 (40.0 / 40.0) |
| 1,440 | 54.8 | 75.9 | 41.1 (38.2 / 44.0) |
| 1,480 | 53.5 | 73.4 | 36.4 (36.5 / 36.2) |
| 1,520 | 52.1 | 78.2 | 41.5 (44.0 / 39.0) |
| 1,560 | 54.5 | 74.4 | 40.5 (40.5 / 40.5) |
| 1,600 | 58.0 | 72.8 | 39.2 (39.8 / 38.8) |
| 1,640 | 57.0 | 75.5 | 36.1 (37.5 / 34.8) |
| 1,680 | 56.4 | 75.8 | 40.0 (37.8 / 42.2) |
| 1,720 | 51.4 | 76.4 | 38.2 (35.5 / 41.0) |
| 1,760 | 58.7 | 78.0 | 44.8 (44.8 / 44.8) |
| 1,800 | 53.8 | 75.5 | 38.6 (40.0 / 37.2) |
| 1,840 | 58.6 | 74.8 | 44.1 (43.0 / 45.2) |
| 1,880 | 58.1 | 78.1 | 45.8 (46.2 / 45.2) |
| 1,920 | 59.6 | 76.1 | 44.0 (43.5 / 44.5) |
| 1,960 | 58.6 | 76.6 | 45.6 (45.2 / 46.0) |
| 2,000 | 56.0 | 74.9 | 45.6 (43.0 / 48.2) |

**E7-17-44's 1,920–2,000M SWA against the soup: 48.0% ± 1.1** (US 47.1, USSR 49.0; 2,000 games).

* **The gain is real against every fixed opponent.** Means of the 1,000–1,200M and 1,840–2,000M
  snapshots: against E7-02-44@1000M 50.0 → 58.2%, against E6-03-44@550M 69.2 → 76.1%, against the
  soup 35.2 → 45.0% (about +70 Elo), in both seats.
* **One run at 2,000M, averaged over its last 80M, nearly matches the four-branch soup**
  (48.0%, about −14 Elo). At 1,200M the same run's snapshots won only ~35% against it.

## SWAs every 80M against fixed references, per side (2026-10-04)

The owner asked for a table with step 80M: the SWA of each run's last 80M against fixed references,
split by side. Each SWA averages the line's snapshots at 10M spacing over the window (9, or 8 for
0–80M; windows crossing 560M or 1,200M join the consecutive runs). Each cell is the SWA's score
against that reference, 1,000 games per side, temperature 0, two-model matches
(`data/reports/swa_line/`); "panel" is the mean over E6-03-44@80/240/550M. Per side ± 1.5, both
sides ± 1.1.

| SWA of | panel (US / USSR) | E6-03-44@550M (US / USSR) | E7-02-44@1000M (US / USSR) | soup (US / USSR) |
|:---|---:|---:|---:|---:|
| 0–80M | 37.9 (32.9 / 43.0) | 19.6 (16.0 / 23.1) | 11.3 (11.1 / 11.6) | 5.4 (5.1 / 5.7) |
| 80–160M | 56.6 (56.6 / 56.7) | 40.2 (42.9 / 37.6) | 19.6 (18.6 / 20.6) | 10.0 (11.0 / 9.0) |
| 160–240M | 61.9 (61.7 / 62.2) | 41.5 (41.6 / 41.5) | 27.2 (28.5 / 26.0) | 15.2 (15.3 / 15.2) |
| 240–320M | 66.0 (66.1 / 65.9) | 49.8 (53.0 / 46.6) | 28.6 (33.3 / 24.0) | 15.3 (18.4 / 12.3) |
| 320–400M | 71.5 (71.9 / 71.0) | 56.5 (62.1 / 51.0) | 35.1 (39.6 / 30.6) | 21.3 (22.4 / 20.1) |
| 400–480M | 75.4 (75.8 / 74.9) | 60.1 (64.0 / 56.3) | 38.2 (39.3 / 37.2) | 25.1 (26.1 / 24.1) |
| 480–560M | 77.5 (76.8 / 78.1) | 61.9 (63.3 / 60.6) | 38.6 (40.3 / 36.9) | 28.3 (28.9 / 27.7) |
| 560–640M | 79.6 (79.9 / 79.4) | 65.2 (67.9 / 62.5) | 45.5 (51.3 / 39.7) | 29.6 (32.3 / 27.0) |
| 640–720M | 81.8 (81.4 / 82.2) | 69.4 (70.5 / 68.3) | 48.3 (51.5 / 45.1) | 31.9 (33.7 / 30.2) |
| 720–800M | 81.8 (82.0 / 81.7) | 71.0 (73.9 / 68.2) | 49.8 (52.1 / 47.5) | 35.8 (37.9 / 33.7) |
| 800–880M | 85.3 (85.6 / 85.0) | 74.4 (76.0 / 72.7) | 52.4 (56.2 / 48.5) | 37.4 (36.9 / 37.8) |
| 880–960M | 84.2 (84.6 / 83.9) | 73.8 (76.0 / 71.6) | 53.5 (54.9 / 52.1) | 39.6 (39.2 / 40.1) |
| 960–1040M | 84.3 (84.6 / 84.0) | 73.8 (76.4 / 71.3) | 54.8 (55.2 / 54.3) | 39.2 (39.5 / 39.0) |
| 1040–1120M | 85.3 (85.8 / 84.8) | 74.6 (76.1 / 73.2) | 54.6 (52.7 / 56.6) | 42.0 (41.0 / 43.1) |
| 1120–1200M | 85.9 (85.9 / 85.8) | 74.6 (76.4 / 72.9) | 58.6 (58.7 / 58.5) | 40.3 (39.5 / 41.2) |
| 1200–1280M | 86.2 (86.4 / 86.0) | 76.2 (77.0 / 75.4) | 57.9 (59.4 / 56.4) | 43.0 (42.1 / 44.0) |
| 1280–1360M | 86.7 (87.6 / 85.9) | 77.5 (80.5 / 74.5) | 56.4 (57.2 / 55.6) | 44.0 (42.3 / 45.8) |
| 1360–1440M | 87.2 (88.6 / 85.9) | 78.1 (82.1 / 74.2) | 59.1 (60.2 / 58.0) | 42.5 (43.1 / 41.9) |
| 1440–1520M | 87.0 (88.0 / 86.1) | 78.6 (81.8 / 75.5) | 58.9 (59.4 / 58.4) | 45.6 (44.3 / 47.0) |
| 1520–1600M | 87.7 (89.2 / 86.1) | 80.0 (83.8 / 76.1) | 58.5 (60.6 / 56.4) | 47.6 (46.4 / 48.8) |
| 1600–1680M | 87.9 (89.2 / 86.5) | 80.0 (81.2 / 78.7) | 60.9 (62.4 / 59.3) | 46.9 (46.0 / 47.7) |
| 1680–1760M | 87.4 (88.9 / 85.8) | 79.2 (81.8 / 76.6) | 59.7 (62.5 / 57.0) | 46.0 (45.6 / 46.4) |
| 1760–1840M | 88.1 (89.0 / 87.1) | 79.6 (82.0 / 77.2) | 61.1 (63.1 / 59.1) | 49.2 (48.7 / 49.7) |
| 1840–1920M | 88.4 (88.9 / 88.0) | 80.1 (81.6 / 78.6) | 61.3 (61.0 / 61.7) | 48.4 (50.3 / 46.5) |
| 1920–2000M | 87.6 (87.7 / 87.5) | 78.9 (80.3 / 77.5) | 63.1 (62.4 / 63.8) | 48.0 (47.1 / 49.0) |

* Against the soup the SWAs climb from 40% (1,120–1,200M) to 46–49% from 1,520M on; against
  E7-02-44@1000M from 59% to 61–63%; against the fixed E6 references they flatten earlier
  (panel ~86–88% from 1,200M, E6-03-44@550M ~79–80% from 1,520M).
* The sides move together; no seat runs away.

## Past 2,000M: E7-19-44 (2,000 → 2,400M)

E7-17-44 continued from its 2,000M end state with its flags (E7-19-44). Each 80M-window SWA against
the soup and E7-02-44@1000M, 1,000 games per side (per side ± 1.5, both ± 1.1):

| SWA of | E7-02-44@1000M (US / USSR) | soup (US / USSR) |
|:---|---:|---:|
| 2000–2080M | 63.9 (64.9 / 62.8) | 49.8 (49.3 / 50.3) |
| 2080–2160M | 63.3 (63.0 / 63.7) | 48.1 (47.4 / 48.8) |
| 2160–2240M | 62.4 (63.8 / 61.0) | 51.2 (50.1 / 52.4) |
| 2240–2320M | 63.4 (64.2 / 62.6) | 51.0 (50.9 / 51.2) |
| 2320–2400M | 63.9 (65.4 / 62.4) | 50.8 (51.6 / 50.1) |

* **Against E7-02-44@1000M the line is flat from about 1,900M** (61–64% for eight windows).
* **Against the soup it reaches level and edges past it** (51.0, 50.8 in the last two windows).
* **The rule, set before the last two windows** ("saturated if, against both references, the mean
  of the last two windows is within +1.5 points of the mean of the first two"): 1,000M +0.05,
  soup +1.95. Not saturated by the rule, marginally; continued to 2,800M as E7-20-44.

## 2,400 → 2,800M: E7-20-44, and the saturation call

| SWA of | E7-02-44@1000M (US / USSR) | soup (US / USSR) |
|:---|---:|---:|
| 2400–2480M | 64.7 (67.6 / 61.8) | 50.5 (50.9 / 50.1) |
| 2480–2560M | 65.3 (67.7 / 62.9) | 51.2 (52.6 / 49.8) |
| 2560–2640M | 62.9 (65.5 / 60.3) | 49.5 (48.3 / 50.7) |
| 2640–2720M | 62.9 (64.6 / 61.1) | 51.4 (53.4 / 49.3) |
| 2720–2800M | 65.4 (66.1 / 64.7) | 53.0 (52.5 / 53.6) |

* **By the same rule, saturated:** last two windows against the first two, E7-02-44@1000M −0.85,
  soup +1.35 (threshold 1.5).
* **Read as a whole, the line plateaus from about 1,900M against E7-02-44@1000M (62–65% for 13
  windows) and creeps against the soup at about +1 point per 400M** (the five windows of 2,000–2,400M
  average 50.2%, those of 2,400–2,800M 51.1%), reaching 53.0% in the last window.
* The league starts from 2,800M (E7-21-44 + E7-22-44, [`../runs.md`](../runs.md)).

## 2,800 → 4,800M: the plain line past the league point, to a plateau (2026-10-05)

The 2,800M call above was premature. E7-20-44 continued as the league's control to 3,600M
([`E7_league_from_2800M.md`](E7_league_from_2800M.md)) and then, at the owner's request, unchanged
to 4,800M (`E7-20-44_20261005_174535`; from 3,600M the per-entity head adds the trunk context once
per sample, the same function to float64 rounding, commit `0280264`). Same 80M-window SWAs, 1,000
games per side (± 1.1 per window), `data/reports/swa_line_ctl/`:

| SWA of | E7-02-44@1000M (US / USSR) | soup (US / USSR) |
|:---|---:|---:|
| 3600–3680M | 71.2 (76.6 / 65.9) | 55.0 (54.3 / 55.8) |
| 3680–3760M | 69.0 (69.9 / 68.2) | 54.4 (55.9 / 52.9) |
| 3760–3840M | 72.0 (74.8 / 69.3) | 54.9 (59.8 / 50.0) |
| 3840–3920M | 72.4 (75.3 / 69.6) | 55.8 (56.9 / 54.6) |
| 3920–4000M | 70.8 (76.1 / 65.5) | 56.1 (57.8 / 54.5) |
| 4000–4080M | 73.8 (77.5 / 70.0) | 58.1 (60.0 / 56.2) |
| 4080–4160M | 71.1 (70.5 / 71.7) | 58.8 (60.6 / 57.0) |
| 4160–4240M | 71.0 (72.6 / 69.5) | 56.0 (56.5 / 55.5) |
| 4240–4320M | 70.8 (73.3 / 68.3) | 57.6 (61.0 / 54.2) |
| 4320–4400M | 70.3 (71.7 / 69.0) | 54.2 (56.2 / 52.1) |
| 4400–4480M | 69.2 (71.3 / 67.0) | 55.1 (57.6 / 52.7) |
| 4480–4560M | 71.2 (75.4 / 67.0) | 55.5 (56.6 / 54.5) |
| 4560–4640M | 70.8 (71.8 / 69.9) | 55.2 (55.7 / 54.6) |
| 4640–4720M | 73.2 (74.0 / 72.5) | 56.7 (56.7 / 56.8) |
| 4720–4800M | 71.2 (74.4 / 68.0) | 55.6 (55.1 / 56.0) |

**400M-block means** (± ~0.5 each) over the whole line:

| block | vs E7-02-44@1000M | vs soup |
|:---|---:|---:|
| 2,000–2,400M | 63.4 | 50.2 |
| 2,400–2,800M | 64.2 | 51.1 |
| 2,800–3,200M | 66.9 | 52.3 |
| 3,200–3,600M | 68.7 | 54.6 |
| 3,600–4,000M | 71.1 | 55.2 |
| 4,000–4,400M | 71.4 | 56.9 |
| 4,400–4,800M | 71.1 | 55.6 |

**Head to head, the last window's SWA (4,720–4,800M)** against earlier ones of this leg, 1,000 games
per side (± 1.1): 51.8% vs 3,600–3,680M, 49.6% vs 3,920–4,000M, 49.2% vs 4,320–4,400M.

* **The line has plateaued since about 3,700M.** Three consecutive 400M blocks are level against
  E7-02-44@1000M (71.1, 71.4, 71.1) and against the soup (55.2, 56.9, 55.6, i.e. no trend), and head
  to head the 4,800M SWA is level with the 4,000M and 4,400M ones and barely above 3,600M.
* **Before that it rose ~2 points per 400M from 2,400M to 3,800M**, in steps: the 1,900–2,800M band
  that triggered the premature call was one of them. A plateau of 1,200M (three blocks) is the
  longest the line has held; it is the first evidence of saturation that a single step could not
  explain, but the line has resumed after a flat stretch once before.
* The best checkpoints of the pure line are now its 4,000–4,800M SWAs (~71% against E7-02-44@1000M,
  ~56% against the 1,200M soup).
