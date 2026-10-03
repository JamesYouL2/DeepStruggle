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

E7-01-44@80M is 1476 in the same field (`data/reports/e7_15_16_80M.{md,json}`).

At 80M the seed alone moves a run by more than either arm does, so nothing is readable yet.
