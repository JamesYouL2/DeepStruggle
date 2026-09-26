# E5-01: does the E4 → E5 rules change move training? 2026-09-26

**Arms:** E5-01-43/44, E4-61-43/44's configuration exactly (E4-08 recipe, M2d, λ 0.98, pool 0.3/12, TF32,
centred heads, `--block-lambda off`), trained from scratch to 240M on the E5 engine (`8d05d94`) as a
pair. `launch_flags.py --diff` against E4-61 shows only `--train-steps`. Both finished without a pin,
stall or crash; the pair took 94 minutes.

**What changed in the game:** [`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md),
*The E4 → E5 boundary*. In summary: influence Ops are spent in full; each Ops modifier has its own limit; an owed Event no
longer inherits the Ops play's stop.

**Rating:** `data/reports/e5_01_vs_e4_61.{md,json}`. It is a round robin of both arms at 40, 80, 120, 160, 200, 220
and 240M, plus E4-08-36@240M and the heuristic anchor (1500). There are 100 games per side per pairing at
temperature 0, all played on the E5 engine. The E4 checkpoints therefore play slightly different rules from the
ones they trained on. The handicap is measured and small: E4-61-44 stopped an influence play early 0.06 times
per game, and the observation keeps the value it trained on.

| run | 40M | 80M | 120M | 160M | 200M | 220M | 240M | late mean (200–240M) |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| E5-01-43 | 1850 | 1935 | 2110 | 2206 | 2207 | 2247 | 2208 | 2221 |
| E4-61-43 | 1855 | 1983 | 2080 | 2154 | 2217 | 2217 | 2250 | 2228 |
| E5-01-44 | 1788 | 2015 | 2061 | 2127 | 2164 | 2237 | 2221 | 2207 |
| E4-61-44 | 1859 | 1928 | 2041 | 2111 | 2142 | 2195 | 2251 | 2196 |

E4-08-36@240M = 2269.

**Late mean, E5 minus E4:**
* seed 43: −7;
* seed 44: +12;
* pair mean: **+2**.

**Head to head**, the late snapshots of each arm against the other's (200, 220 and 240M, both seeds, 7,200 games):
* all pairings: E5 wins **51.0% ± 0.6**, or +7 Elo;
* seed 43 against seed 43: 49.8% ± 1.2;
* seed 44 against seed 44: 54.6% ± 1.2.

The standard errors treat games as independent. They are not, because every pairing replays the same deals at
temperature 0.

**Reading:** no measurable effect of the rules change on training. The two seeds differ in sign, and the pair
mean is +2 Elo. The learning curves match from 40M to 240M. So E4 and E5 numbers can be compared, labelled as
cross-engine, and the E4 ladder remains a valid reference for E5 arms.
