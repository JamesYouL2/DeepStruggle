# The P24 league from the saturated E7 line (2026-10-04): 2,800 → 3,200M

The owner asked to start league training once the E7 line clearly saturated; by the rule set in
advance it did at 2,800M ([`E7_line_to_2000M.md`](E7_line_to_2000M.md)).

**Runs** ([`../runs.md`](../runs.md)). `tools/scripts/league.py`:

* **Main, E7-21-44:** resumed from E7-20-44's 2,800M end state with the E7 shallow flags, plus the
  league -- `--league-frac 0.5`, up to four published exploiter snapshots, `--resume-every-steps
  10000000` -- to 3,200M. No setup credit (E6-04-44's arm had it; left out so the league is the one
  factor). `launch_flags --diff` against E7-20-44 shows only the league flags, the resume interval
  and the step target.
* **Main exploiter, E7-22-44-<generation>:** E5-22's recipe -- each 50M-step generation reset to the
  main's newest resume state, trained against a frozen greedy snapshot of it with no KL, no entropy
  bonus and learner-only normalisation; judged greedily, 200 games a side, published at ≥ 55%,
  restarted at ≥ 65%.

League directory `data/league/E7-21-44/` (events in `events.jsonl`).

## The exploiter could barely exploit the main

Each generation's judgements against its frozen target (5 per generation, one per 10M):

| gen | from main @ | judged range | best (US / USSR) |
|---:|---:|---:|---:|
| 1 | 2,810M | 51.4–53.4% | 53.4 (56.5 / 50.2) |
| 2 | 2,860M | 46.5–52.9% | 52.9 (58.5 / 47.2) |
| 3 | 2,910M | 44.1–54.8% | 54.8 (55.5 / 54.0) |
| 4 | 2,960M | 42.4–52.5% | 52.5 (56.8 / 48.2) |
| 5 | 3,030M | 47.1–51.5% | 51.5 (56.0 / 47.0) |
| **6** | 3,080M | 45.9–**55.0%** | **55.0 (61.8 / 48.2) -- published, snapshot 3,110M** |
| 7 | 3,130M | 47.2–51.1% | 51.1 (52.2 / 50.0) |
| 8 | 3,170M | 51.0% (1 judgement) | stopped when the main finished |

On E6, the same exploiter recipe reached ~69% against its first target. Against the 2,800M+ E7 main,
350M exploiter steps produced one snapshot at the 55% bar. Exploiters do somewhat better as US
(52–62%) than as USSR (47–54%).

So **the league pool was empty from 2,800M to 3,110M** -- the main trained as plain self-play -- and
held one member for the last ~90M steps.

## The main's 80M-window SWAs against fixed references

1,000 games per side, temperature 0 (`data/reports/swa_line/`); per side ± 1.5, both ± 1.1.

| SWA of | E7-02-44@1000M (US / USSR) | soup (US / USSR) | league members in the pool |
|:---|---:|---:|:---|
| 2720–2800M (E7-20-44, end of the plain line) | 65.4 (66.1 / 64.7) | 53.0 (52.5 / 53.6) | -- |
| 2800–2880M | 65.1 (67.3 / 62.8) | 51.2 (52.0 / 50.5) | none |
| 2880–2960M | 64.6 (66.4 / 62.8) | 50.4 (49.3 / 51.6) | none |
| 2960–3040M | 67.8 (67.5 / 68.0) | 53.0 (50.8 / 55.2) | none |
| 3040–3120M | 67.5 (69.1 / 66.0) | 50.6 (49.4 / 51.8) | none until 3,110M |
| **3120–3200M** | **71.4 (74.2 / 68.6)** | **56.5 (58.0 / 55.1)** | one (gen 6) |

## Reading

* **Hard to exploit.** At 2,800M+ the main agent resists the E5-22 exploiter recipe at 50M-step
  generations: one publish in eight generations.
* **The line kept creeping while the pool was empty.** 2,960–3,120M sits above the 62–65% band
  against E7-02-44@1000M that held since ~1,900M, so the 2,800M "saturated" call was marginal.
* **The last window is the strongest of the whole line by a clear margin** -- 71.4% against the
  1,000M model (about +4 over the previous two windows) and 56.5% against the soup (about +4.7), up
  in both seats. It is also the only window trained with a league member in the pool, for ~90M
  steps; too short and too confounded with the ongoing creep to attribute to the league yet.
* Continued (owner: "continue it"), unchanged, as E7-21-44 / E7-22-44 from generation 9.
