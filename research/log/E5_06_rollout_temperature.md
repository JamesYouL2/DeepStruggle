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
