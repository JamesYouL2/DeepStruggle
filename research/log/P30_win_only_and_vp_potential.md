# P30 (2026-10-03): two reward simplifications from scratch -- win only, and a small VP potential

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

E7-01-44 is 1476 at 80M, 1502 at 160M, 1498 at 240M, 1553 at 320M, 1567 at 400M, 1565 at 480M and 1578 at 560M; E7-02-44 is 1583 at 640M, 1607 at 720M, 1623 at 800M, 1639 at 880M and 1636 at 960M in the same fields (`data/reports/e7_15_16_<M>M.{md,json}`).

At 80M the seed alone moves a run by more than either arm does, so nothing is readable yet.

At 160M win only is behind in the US seat (−6.8, 3.6 SE) and head to head (43.1%), and the VP
potential has closed most of its 80M gap. Both are still within the seed's range (E7-08-43 moved
from 37.2% to 53.1% between 80M and 160M).

At 240M all three are level with the control: every head-to-head within 51.6–56.9%, every seat
difference within 2 SE. The VP potential is at the top of the field (1526).

At 320M win only is level with the control in every column. The VP potential is below it against
the panel in both seats (−7, about 4 SE) but level head to head (49.0%); seed 43 shows a panel gap
of the same size in one seat (US −6.0), so this is still inside the run-to-run range.

At 400M all three runs sit 10–30 Elo under E7-01-44, the arms no further than seed 43 (46.2–47.5%
head to head).

At 480M the four runs are within 6 Elo of each other (1561–1567), with the same per-seat pattern
for all three against E7-01-44 (US about −4, USSR about +2).

At 560M (E7-01-44's last snapshot; the control is E7-02-44 from here) the VP potential is top of
the field (1584) and win only bottom (1564), 20 Elo apart, inside the seed range.

At 640M (control E7-02-44) both arms lead the field by a few Elo (1594, 1590 against 1583), level
within noise.

At 720M E7-02-44 tops the field (1607); the VP potential is level with it per seat, win only
slightly below against the panel (−4.6 / −3.8) but level head to head, and seed 43 lowest.

At 800M E7-02-44 leads all three (1623 against 1595, 1595, 1579), as at 720M: the seed-44
control is pulling ahead of every other run, the no-change seed 43 included.

At 880M win only is below seed 43 for the first time (1574 against 1586; USSR −8.7 against −2.4
against the panel), the VP potential above it (1606). E7-02-44 still leads all three.

At 960M the order of 880M holds: E7-02-44, then the VP potential (level with it per seat within
1 SE), then win only and seed 43 together.
