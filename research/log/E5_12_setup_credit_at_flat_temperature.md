# E5-12 — per-side setup credit at flat temperature 1.0: no gain, and it locks the setup, bad or good

**Question.** A1 (`--block-lambda setup-side`, [P4](../plans/P4_setup_macro_action_credit.md)) was
adopted as a default on 2026-09-26 for Poland, at +27/+18 over 80M, inside the noise. It had never
run beyond 80M or at flat rollout temperature 1.0, and E5-11 (flat 1.0, `--block-lambda off`)
already plays Poland in every game. Does A1 add strength or setup stability to the new default
recipe?

**Arms.** E5-12-43/44 is E5-11-SEED's configuration with the setup credit at its default instead of
`off`, from scratch to 560M. `launch_flags.py --diff` against E5-11 prints only `--block-lambda`
(`setup-side` / `off`) and `--train-steps`. Directories are `E5-12-43_20260927_163119` and
`E5-12-44_20260927_163139`. Training was healthy throughout: no stall and no pinned seat. Entropy at
560M was 0.42 / 0.48.

## Strength

A round robin per seed: E5-12 and E5-11 at 240/320/400/480/560M, plus E4-61@560M and
`HeuristicBot`. Greedy play, 100 games per seat per pair. Reports are in
`data/reports/e5_12_rr_{43,44}.{md,json}`.

| E5-12 vs E5-11 at matched steps | seed 43 | seed 44 |
|:---|---:|---:|
| 240M | 44.5% | 43.0% |
| 320M | 36.5% | 49.5% |
| 400M | 41.5% | 44.5% |
| 480M | 40.5% | 53.5% |
| 560M | 40.5% | 53.5% |
| **late block, 480–560 × 480–560 (n = 800)** | **39.6% ± 1.7 (−73)** | **54.2% ± 1.8 (+30)** |

The late seat rates are read against the control's own greedy self-play (E5-11-SEED@560M against
itself, 200 games per seat): US 42.5% on seed 43 and 38.5% on seed 44.

| E5-12@560M vs E5-11@560M | as US, arm − control self-play | as USSR |
|:---|---:|---:|
| seed 43 | 34% (−8) | 47% (−10) |
| seed 44 | 54% (+16) | 53% (−8) |

**Split, mean −22.** Seed 43 is weaker in both seats. Seed 44 is stronger as US and slightly weaker
as USSR. By the adoption rule (a win on both seeds), A1 adds no strength at flat 1.0.

## Setup

The setup probe (`ai/eval/setup_probe.py`, temperature 0.1, 1,000 games) was run at every 40M
snapshot from 400M to 560M (`data/reports/e5_12_setup_series.json`).

| targets met at 400 / 440 / 480 / 520 / 560M | Poland ≥ 3 | West Germany ≥ 4 | Italy ≥ 2 | Iran ≥ 2 |
|:---|:---|:---|:---|:---|
| E5-12-43 | all 100% | all 0% | all 0% | all 100% |
| E5-11-43 | all 100% | 0 0 0 100 85% | all 100% | 100 100 100 0 15% |
| E5-12-44 | all 100% | all 100% | all 100% | all 0% |
| E5-11-44 | all 100% | 0 0 100 100 100% | all 100% | 100 100 97 100 100% |

**A1 does stabilise the setup.** E5-12 opens identically at all five snapshots on both seeds.
E5-11 flips West Germany and Iran on both seeds, though seed 44 has settled on all four targets
since 520M.

**It stabilises whatever it has found.** At 560M, E5-12-43's US opening is **Denmark 7, Iran 2**
in 500 of 500 games. All seven Western Europe points go into Denmark, a non-battleground, and West
Germany, Italy and France are left open. It holds that opening from 400M on. E5-12-44 opens
West Germany 4, France 3, Italy 2, and never takes Iran.

The rest of the goal probes (`data/reports/e5_12_goal_probes.md`) show no change beyond
seed-to-seed spread. Forced wins taken are 58% / 61%, against 66.5% for E5-11-44@560M.

## Reading

* **A1 adds no strength at flat temperature 1.0**: −73 / +30.
* **Its stability is not correctness.** Per-side credit for the whole setup block turns the opening
  into one decision with one reward. At flat 1.0 that decision stops moving, including when it is
  wrong. Denmark 7 on seed 43 coincides with that seed's weak US seat (−8 against self-play).
* E5-11 without A1 reached a correct, stable opening on seed 44 and plays Poland on both seeds.

**Proposed:** return `--block-lambda` to `off` as the default. E5-11, not E5-12, remains the
control for the next arms. That is a recipe decision for the owner.
