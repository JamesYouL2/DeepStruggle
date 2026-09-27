# E5-06: rollout temperature at the policy, from the plateau, 2026-09-26

**Arms:** E5-06-43/44. Each is E4-61-SEED's configuration resumed at 410M on E5 and run to 610M, with
one change: `--rollout-temps 0.8 1.2 0.7 1.1`. `launch_flags.py --diff` against E4-61 shows only that
flag and `--train-steps`. The default bands, 0.15 / 0.50 / 0.10 / 0.35, sharpen every environment
while PPO records temperature-1 log-probabilities. The new bands sample at roughly the policy.

**Prior, stated correctly.** E3 ran the same change as a complete two-by-two: two seeds, each
treatment against its own seed's control, 80M from scratch
([`../archive/E3_ladder/log/P15_rollout_temperature.md`](../archive/E3_ladder/log/P15_rollout_temperature.md)).
* **Both seeds were faster early**, significant from 40M to 60M.
* **At 80M only seed A held** (46.0% against 17.8%). Seed B's treatment plateaued at about 23% while its
  control caught up (+3.0 pp, not significant).
* **The verdict:** it *accelerates*, and whether it raises the ceiling is seed-dependent. The
  recommendation to change the default was withdrawn, pending seeds three and four, which never ran.
* **The mechanism was withdrawn too.** The ending-mix explanation (fewer DEFCON-1 endings) did not
  survive the second control seed.

An earlier version of this entry quoted only seed A's 46.0% against 17.8% as the E3 result. That
overstated it. The change had never been tested on E4.

**Controls:** E4-61-SEED's own 410→610M legs, the same comparison as the P24 league. The known
confounds are the same, each small and measured:
* engine E5 against E4 (+2);
* a resumed run against a continuous one (46.2% against 47.0%);
* 11 own pool members instead of 12 after the resume.

## Result

Ratings from `data/reports/e5_06_temp_{43,44}.{md,json}`: a greedy round robin, 100 games a side,
with the heuristic bot anchored at 1500.

| run | 440M | 480M | 520M | 560M | 590M | 610M | late mean (560–610M) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| E5-06-43 | 2333 | 2346 | 2345 | 2340 | 2365 | 2380 | **2362** |
| E4-61-43, control | 2259 | 2279 | 2283 | 2256 | 2324 | 2287 | 2289 |
| E5-06-44 | 2275 | 2350 | 2364 | 2362 | 2372 | 2366 | **2367** |
| E4-61-44, control | 2273 | 2249 | 2256 | 2230 | 2250 | 2277 | 2253 |

| seed | late mean, arm − control | head to head, late against late | vs E4-08-36@240M, arm / control |
|:---|---:|---:|---:|
| 43 | **+73** | **58.3% ± 1.2** (1,800 games) | 64.8% / 59.5% |
| 44 | **+115** | **61.3% ± 1.1** (1,800 games) | 63.3% / 46.7% |

**Averaged over the pair, +94.**
* The gain is present at the first rated snapshot. At 480M, 70M steps in, a greedy head to head
  against the control's 480M gives 59.2% on seed 43 and 64.0% on seed 44, on both seats.
* The gain holds on both seats throughout.
* Both arms beat the heuristic 99–100%.
* For comparison, the P24 league on the same seeds and controls gave −6 and +65, at twice the GPU
  cost.

## Dynamics

The policy sharpens itself. Entropy falls from about 1.0 to 0.35–0.41 within 70M steps and holds
there, where the controls stay at 1.0–1.2. The win rate against the run's own pool rises from about
0.74 to 0.83–0.86. Under the sharpening bands the rollout did the sharpening and the policy stayed
broad. Sampling at the policy moves the sharpening into the weights, where PPO's gradients and
its log-probabilities agree.

## Reading

**This is the largest single gain on the E4/E5 ladder since the architecture work, and it breaks
the ~400M plateau** that time alone had not moved (the treadmill). Two seeds, both large, both far
outside noise.

**It bears directly on E3's open question.** E3 left open whether the change raises the ceiling or
only speeds up early training. E5-06 starts at a ceiling, the plateau, and holds its gain for 200M
steps on both seeds. That is the persistence E3's seed B lacked, though in a different regime (a
late plateau, not training from scratch), with two seeds again.

Two checks are running:
* **E5-07**, the same resume with the default temperature, tests whether the resume carries any
  of the gain;
* **E5-08**, the change from scratch to 240M, repeats E3's setup at three times its budget, which
  is where E3's seed B lost its gain.

**Proposed:** make `--rollout-temps 0.8 1.2 0.7 1.1` the default. That is the owner's call, since it
changes the recipe for every future arm.

**Open:**
* whether the gain holds from scratch, where E3 saw it (a from-scratch pair to 160M);
* whether other bands do better (a flat 1.0, or bands above 1);
* whether the sharpened endpoint, entropy 0.35, costs anything the probes can see.

## The resume control (E5-07) and the from-scratch test (E5-08), 2026-09-27

### E5-07: resuming is not neutral, and temperature holds against the matched control

**E5-07** is E5-06's resume, E4-61-SEED@410M on E5 to 610M, with the default temperatures.
Round robins `data/reports/e5_07_resume_{43,44}.{md,json}`, greedy, 100 games a side:

| run | 440M | 480M | 520M | 560M | 590M | 610M | late mean |
|:---|---:|---:|---:|---:|---:|---:|---:|
| E5-06-43 | 2350 | 2359 | 2339 | 2346 | 2367 | 2381 | 2364 |
| E5-07-43 | 2272 | 2272 | 2287 | 2320 | 2311 | 2298 | 2310 |
| E4-61-43 | 2270 | 2289 | 2297 | 2267 | 2330 | 2307 | 2301 |
| E5-06-44 | 2314 | 2388 | 2398 | 2398 | 2391 | 2390 | 2393 |
| E5-07-44 | 2309 | 2283 | 2343 | 2308 | 2300 | 2347 | 2318 |
| E4-61-44 | 2303 | 2277 | 2285 | 2275 | 2279 | 2302 | 2286 |

Head to head of the late snapshots (560, 590, 610M), 1,800 games a pair:

| comparison | seed 43 | seed 44 | mean |
|:---|---:|---:|---:|
| resume (and E5) alone: E5-07 vs E4-61 | +9, 51.1% ± 1.2 | **+33, 53.7% ± 1.2** | +21 |
| **temperature alone: E5-06 vs E5-07** | **+55, 54.4%** | **+75, 58.1%** | **+65** |
| both: E5-06 vs E4-61 | +63, 58.4% | +108, 61.2% | +86 |

* **The resume is not neutral on seed 44.** +33 there is about three standard errors. The +73/+115
  quoted above against E4-61 included it, so they overstate the temperature's effect.
* **The temperature effect proper is +55 and +75 against the matched resume control, +65 on
  average.**
* E5-07-44's first leg was aborted at 452M: the stall watchdog read an 8 h host suspend as silence.
  It was continued from its own 450M state as a second leg under the same name. The watchdog now
  counts awake time only.

### E5-08: from scratch, the gain holds to 240M on both seeds

**E5-08** is E5-01-SEED's configuration with `--rollout-temps 0.8 1.2 0.7 1.1`, from scratch to
240M. Round robins `data/reports/e5_08_scratch_{43,44}.{md,json}`:

| run | 40M | 80M | 120M | 160M | 200M | 240M | late mean (200–240M) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| E5-08-43 | 2098 | 2240 | 2289 | 2289 | 2351 | 2400 | **2376** |
| E5-01-43, control | 1858 | 1951 | 2110 | 2205 | 2221 | 2188 | 2205 |
| E5-08-44 | 1886 | 2091 | 2156 | 2207 | 2254 | 2268 | **2261** |
| E5-01-44, control | 1711 | 1932 | 1979 | 2046 | 2104 | 2156 | 2130 |

Head to head at matched steps, arm against control:

| | 40M | 80M | 120M | 160M | 200M | 240M | late (200–240M) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| seed 43 | 85.0% | 81.0% | 69.5% | 66.0% | 70.5% | 74.5% | **73.5% ± 1.6** |
| seed 44 | 76.5% | 75.5% | 79.0% | 75.0% | 63.0% | 62.0% | **68.4% ± 1.6** |

* **The late mean gains +171 on seed 43 and +131 on seed 44.** Both seeds hold past E3's 80M budget
  out to 240M. E3's seed B lost its lead by 80M; neither seed does here.
* Seed 44's lead narrows at the end (62% at 240M). Whether the gap keeps closing is the 800M run's
  question.
* **On training-game blunders:** self-inflicted DEFCON 1 falls from 20–39% of the control's games to
  3–8%, while provoked DEFCON 1 barely moves (10–15%). Games reaching final scoring rise from 2–8%
  to 26–34%. Sharpened sampling made the policy's argmax blunder universal. Sampling at the policy
  lets PPO push that blunder down.

**A cross-field check** (`data/reports/best_2026-09-27.{md,json}`, one field, greedy):
* E5-06-43@610M rates highest, at 2327.
* **E5-08-43@240M, a 240M from-scratch run, rates 2293.** That is above E4-61-44@720M (2268) and
  E4-61-43@800M (2245).

**Next (pre-registered in `runs.md`):** E5-09 (flat 1.0) and E5-10 (flat 1.1) against E5-06 and E5-07
from the same resume. Then the chosen temperature trains from scratch to 800M.

## Which temperature: bands against flat 1.0 against flat 1.1 (E5-09, E5-10), 2026-09-27

**Design:** all four arms from the same resume, E4-61-SEED@410M on E5, to 610M. Round robins
`data/reports/temp_choice_{43,44}.{md,json}`, greedy, 100 games a side:

| arm | seed 43, late mean (560–610M) | seed 44, late mean |
|:---|---:|---:|
| bands 0.8 1.2 0.7 1.1 (E5-06) | 2364 | 2433 |
| flat 1.0 (E5-09) | 2415 | 2416 |
| flat 1.1 (E5-10) | 2417 | 2438 |
| default 0.15 0.50 0.10 0.35 (E5-07) | 2313 | 2335 |

**Head to head of the late snapshots** (1,800 games a seed, 3,600 pooled):

| pairing | seed 43 | seed 44 | pooled |
|:---|---:|---:|---:|
| flat 1.1 vs bands | 58.4% | 51.3% | **54.9% ± 0.8** |
| flat 1.0 vs bands | 59.9% | 44.6% | 52.3% ± 0.8 |
| flat 1.1 vs flat 1.0 | 48.9% | 53.4% | 51.2% ± 0.8 |
| bands vs default | 54.4% | 58.3% | 56.3% |
| flat 1.0 vs default | 63.3% | 60.9% | 62.1% |
| flat 1.1 vs default | 64.6% | 64.9% | 64.8% |

**The pre-registered rule** (`runs.md`, E5-10 row) requires an arm to beat both other temperature arms
by ≥ 2.5 pp pooled and lose on neither seed.
* Flat 1.1 misses on flat 1.0: +1.2.
* Flat 1.0 misses on the bands: +2.3, and it loses to them on seed 44.
* **So the arms are neutral, and flat 1.0 is chosen.**

**Reading:**
* **The large step is leaving the sharpening default.** Every near-policy setting beats it 56–65%.
* **Among the near-policy settings the differences are small.** Flat 1.1 is the pooled leader,
  better than the bands on both seeds and the best against the default. Its lead over flat 1.0 is
  about 1.5 standard errors, and the rule does not act on that.

**Next:** E5-11-43/44, flat 1.0 from scratch to 800M.

## E5-11 at 240M, and per-seat results read against control self-play (2026-09-27)

E5-11 is flat 1.0 from scratch. Greedy head to heads at 240M, 200 games a side. Per-seat effect =
the arm's seat rate − the control's own greedy self-play rate in that seat
([`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)).

| E5-11 at 240M vs | overall | US effect | USSR effect |
|:---|---:|---:|---:|
| E5-01-43 (default) | 79.2% | +38.5 | +20.0 |
| E5-08-43 (bands) | 55.2% | +1.0 | +9.5 |
| E5-01-44 (default) | 76.5% | +37.5 | +15.5 |
| E5-08-44 (bands) | 61.3% | +28.0 | −5.5 |

Self-play US win rates at 240M: E5-01 33.0% / 39.0%, E5-08 37.0% / 31.5%, E5-11 41.0% / 43.0%.

* Against the default, flat 1.0 is stronger on both seats, and most on US.
* Against the bands, the gain sits on a different seat on each seed, so the skew is a seed property.
* Flat 1.0's own games are the most seat-balanced of the three.

**Earlier per-seat readings in this log and in the P24 log quote raw rates.** Those readings are not
seat effects. The overall rates stand.
