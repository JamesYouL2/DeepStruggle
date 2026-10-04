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
