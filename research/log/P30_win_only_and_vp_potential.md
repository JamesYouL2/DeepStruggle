# P30 (2026-10-03): two reward simplifications from scratch -- win only, and a small VP potential

**Conclusion.** Neither reward simplification passes the owner's rule against E7-02-44 at 1,200M.
Win only (`--reward-scheme terminal --no-blunder-window`) is 45% head to head and below in both
seats, but the unchanged recipe on seed 43 is just as far below (43%), so the blunder-aware
reward's contribution cannot be told from seed spread. A 0.01 VP potential is level on the SWA and
slightly below on snapshots: no gain.

**Arms** ([`../runs.md`](../runs.md)), both E7-01/02-44's shallow recipe from scratch, seed 44, to
1,200M, two at a time:

* **E7-15-44, win only** -- `--reward-scheme terminal --no-blunder-window`: +1/-1 at the end of the
  game and nothing else.
* **E7-16-44, VP potential** -- the recipe (blunder-aware reward and window) plus
  `--vp-potential 0.01`: each step pays the mover 0.01 x its VP change, and the terminal step takes
  back the VP lead the game ended on, so the shaping sums to zero over a game.

**Control.** E7-01-44 → E7-02-44 (seed 44). **Seed reference.** E7-08-43, the recipe on seed 43.

## Same-field trace (400 games per side per pairing, temperature 0)

Per seat against the panel E6-03-44@80/240/550M, arm minus E7-01/02-44 at the same step; head to
head against it.

| M | arm | US | USSR | head to head | Elo (field) |
|---:|:---|---:|---:|---:|---:|
| 80 | win only E7-15-44 | −3.6 ± 1.9 | −0.6 ± 1.8 | 52.1% ± 1.8 | 1454 |
| 80 | VP potential E7-16-44 | −9.0 ± 1.8 | −6.4 ± 1.8 | 40.9% ± 1.8 | 1405 |
| 80 | *seed 43 E7-08-43* | *−11.8 ± 1.8* | *−2.5 ± 1.8* | *37.2% ± 1.8* | *1421* |
| 160 | win only E7-15-44 | −6.8 ± 1.9 | −0.9 ± 1.9 | 43.1% ± 1.8 | 1455 |
| 160 | VP potential E7-16-44 | −1.4 ± 1.9 | −3.3 ± 1.9 | 47.1% ± 1.8 | 1487 |
| 160 | *seed 43 E7-08-43* | *+0.7 ± 2.0* | *+5.0 ± 1.9* | *53.1% ± 1.8* | *1519* |
| 240 | win only E7-15-44 | −3.7 ± 2.0 | +3.8 ± 1.9 | 51.6% ± 1.8 | 1500 |
| 240 | VP potential E7-16-44 | +0.2 ± 2.0 | +3.3 ± 1.9 | 52.9% ± 1.8 | 1526 |
| 240 | *seed 43 E7-08-43* | *−0.7 ± 2.0* | *+3.3 ± 1.9* | *56.9% ± 1.8* | *1518* |
| 320 | win only E7-15-44 | −1.5 ± 1.9 | −0.4 ± 1.8 | 51.4% ± 1.8 | 1550 |
| 320 | VP potential E7-16-44 | −7.2 ± 1.9 | −7.0 ± 1.8 | 49.0% ± 1.8 | 1518 |
| 320 | *seed 43 E7-08-43* | *−6.0 ± 2.0* | *−1.2 ± 1.8* | *43.5% ± 1.8* | *1525* |
| 400 | win only E7-15-44 | −5.5 ± 1.9 | +1.6 ± 1.8 | 46.4% ± 1.8 | 1542 |
| 400 | VP potential E7-16-44 | −3.4 ± 1.8 | −3.7 ± 1.8 | 46.2% ± 1.8 | 1535 |
| 400 | *seed 43 E7-08-43* | *−1.6 ± 1.9* | *+1.1 ± 1.8* | *47.5% ± 1.8* | *1556* |
| 480 | win only E7-15-44 | −4.7 ± 1.9 | +1.5 ± 1.8 | 49.9% ± 1.8 | 1561 |
| 480 | VP potential E7-16-44 | −4.3 ± 1.8 | +2.2 ± 1.7 | 55.0% ± 1.8 | 1567 |
| 480 | *seed 43 E7-08-43* | *−3.9 ± 1.8* | *+2.8 ± 1.8* | *48.9% ± 1.8* | *1565* |
| 560 | win only E7-15-44 | −4.9 ± 1.8 | −2.2 ± 1.7 | 47.4% ± 1.8 | 1564 |
| 560 | VP potential E7-16-44 | +1.2 ± 1.8 | +3.1 ± 1.7 | 51.4% ± 1.8 | 1584 |
| 560 | *seed 43 E7-08-43* | *−3.8 ± 1.8* | *+3.3 ± 1.7* | *50.2% ± 1.8* | *1570* |
| 640 | win only E7-15-44 | +0.5 ± 1.7 | −2.7 ± 1.7 | 54.2% ± 1.8 | 1594 |
| 640 | VP potential E7-16-44 | +1.9 ± 1.7 | −4.0 ± 1.7 | 53.8% ± 1.8 | 1590 |
| 640 | *seed 43 E7-08-43* | *−7.3 ± 1.8* | *−4.2 ± 1.7* | *49.8% ± 1.8* | *1564* |
| 720 | win only E7-15-44 | −4.6 ± 1.7 | −3.8 ± 1.7 | 49.2% ± 1.8 | 1581 |
| 720 | VP potential E7-16-44 | −0.2 ± 1.7 | +0.4 ± 1.6 | 47.2% ± 1.8 | 1595 |
| 720 | *seed 43 E7-08-43* | *−6.4 ± 1.7* | *+1.7 ± 1.6* | *45.1% ± 1.8* | *1576* |
| 800 | win only E7-15-44 | −0.9 ± 1.6 | −4.0 ± 1.6 | 45.9% ± 1.8 | 1595 |
| 800 | VP potential E7-16-44 | −3.4 ± 1.7 | −3.7 ± 1.6 | 50.1% ± 1.8 | 1595 |
| 800 | *seed 43 E7-08-43* | *−8.1 ± 1.7* | *−2.9 ± 1.6* | *42.2% ± 1.8* | *1579* |
| 880 | win only E7-15-44 | −6.5 ± 1.6 | −8.7 ± 1.6 | 41.6% ± 1.8 | 1574 |
| 880 | VP potential E7-16-44 | −2.4 ± 1.6 | −3.5 ± 1.6 | 46.1% ± 1.8 | 1606 |
| 880 | *seed 43 E7-08-43* | *−6.3 ± 1.6* | *−2.4 ± 1.6* | *42.8% ± 1.8* | *1586* |
| 960 | win only E7-15-44 | −5.2 ± 1.6 | −4.7 ± 1.6 | 41.0% ± 1.8 | 1582 |
| 960 | VP potential E7-16-44 | −0.8 ± 1.6 | −2.1 ± 1.6 | 45.9% ± 1.8 | 1609 |
| 960 | *seed 43 E7-08-43* | *−10.6 ± 1.7* | *−2.6 ± 1.6* | *45.2% ± 1.8* | *1576* |
| 1040 | win only E7-15-44 | −5.6 ± 1.6 | −0.7 ± 1.6 | 44.9% ± 1.8 | 1592 |
| 1040 | VP potential E7-16-44 | −2.3 ± 1.6 | +0.4 ± 1.6 | 49.6% ± 1.8 | 1613 |
| 1040 | *seed 43 E7-08-43* | *−2.9 ± 1.6* | *−2.0 ± 1.6* | *46.1% ± 1.8* | *1598* |
| 1120 | win only E7-15-44 | −4.8 ± 1.6 | −1.9 ± 1.5 | 45.9% ± 1.8 | 1598 |
| 1120 | VP potential E7-16-44 | −0.5 ± 1.5 | −0.4 ± 1.5 | 45.6% ± 1.8 | 1627 |
| 1120 | *seed 43 E7-08-43* | *−5.1 ± 1.6* | *+0.3 ± 1.5* | *44.6% ± 1.8* | *1612* |

E7-01-44 is 1476 at 80M, 1502 at 160M, 1498 at 240M, 1553 at 320M, 1567 at 400M, 1565 at 480M and 1578 at 560M; E7-02-44 is 1583 at 640M, 1607 at 720M, 1623 at 800M, 1639 at 880M, 1636 at 960M, 1627 at 1,040M and 1646 at 1,120M in the same fields (`data/reports/e7_15_16_<M>M.{md,json}`).

**Reading the trace.** Through 560M all four runs stayed within the seed's range of each other
(seed 43 alone moved from 37.2% to 53.1% head to head between 80M and 160M). From 720M on the
seed-44 control E7-02-44 pulled ahead of every other run, the unchanged seed 43 included; the VP
potential stayed level with it per seat within about 1 SE, while win only fell to seed 43's level
(from 880M) and shared its US-seat gap.

## At saturation (1,200M): the owner's rule against E7-02-44

One round robin, 1,000 games per side per pairing, temperature 0 (`data/reports/e7_15_16_rr.{md,json}`).

| comparison | US | USSR | head to head |
|:---|---:|---:|---:|
| **win only E7-15-44**, 1,140–1,200M snapshots | −4.6 ± 0.5 | −3.4 ± 0.5 | 45.4% ± 0.3 |
| **win only E7-15-44**, 1,120–1,200M SWA | −3.7 ± 0.9 | −4.1 ± 0.9 | 44.9% ± 1.1 |
| **VP potential E7-16-44**, 1,140–1,200M snapshots | −1.6 ± 0.5 | −0.9 ± 0.5 | 47.4% ± 0.3 |
| **VP potential E7-16-44**, 1,120–1,200M SWA | +0.0 ± 0.9 | −1.0 ± 0.9 | 47.2% ± 1.1 |
| *seed 43 E7-08-43 (same recipe as the control)*, SWA | *−5.6 ± 0.9* | *−2.9 ± 0.9* | *43.4% ± 1.1* |

SWAs: E7-02-44 1595, E7-16-44 1587, E7-15-44 1560, E7-08-43 1550; the late snapshots of each run
cluster below its SWA in the same order.

## Reading

* **Neither arm passes the owner's rule.** Neither is better than E7-02-44 in any seat.
* **Win only lands where a change of seed lands.** It is below E7-02-44 in both seats and 45% head
  to head, but seed 43 of the unchanged recipe is just as far below (43.4%, SWA 1550 against win
  only's 1560). So this run cannot tell the blunder-aware reward's contribution from seed-to-seed
  spread: dropping it costs at most about what the seed does, possibly nothing. Settling it needs a
  win-only run on seed 43, read against E7-08-43.
* **The VP potential is level with the control on the SWA** (US +0.0, USSR −1.0, within 1 SE) and
  slightly below on snapshots and head to head (47%). No gain; above seed 43 throughout.
* **The control is a strong seed.** From 720M on, E7-02-44 led every other run, the no-change
  seed 43 included, so a one-seed comparison against it reads every arm as slightly worse.
