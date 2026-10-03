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

E7-01-44 is 1476 at 80M and 1502 at 160M in the same fields (`data/reports/e7_15_16_<M>M.{md,json}`).

At 80M the seed alone moves a run by more than either arm does, so nothing is readable yet.

At 160M win only is behind in the US seat (−6.8, 3.6 SE) and head to head (43.1%), and the VP
potential has closed most of its 80M gap. Both are still within the seed's range (E7-08-43 moved
from 37.2% to 53.1% between 80M and 160M).
