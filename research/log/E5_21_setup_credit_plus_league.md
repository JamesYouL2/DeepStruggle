# E5-21 — game-result setup credit plus the league, from E5-11-43@310M: the best arm so far

**Question.** E5-16 showed that game-result setup credit repairs the locked opening, but its USSR
lost the punishment of weak US setups (−5.1 per seat). E5-17 showed the league keeps weaknesses
punished and countered, but it cannot repair a setup and was level overall. Do the two together
give both halves?

**Arm.** E5-21-43, driven by `tools/scripts/league.py`:
* **Main agent:** E5-11-43 resumed at 310M (`resume_310050816steps.pt`) to 560M, with
  `--setup-mc-credit --setup-entropy-floor 0.3` and `--resume-every-steps 10000000`, plus the
  league group exactly as in E5-17. `launch_flags --diff` against E5-11 shows only those.
* **Exploiter (E5-22-43-g):** E5-18's recipe, reset to the main agent's newest resume state, at
  most 10M behind its target.

Directory `E5-21-43_20260928_165156`; league events in `data/league/E5-21-43/events.jsonl`.

**Pre-registered** ([`../runs.md`](../runs.md)):
* **Strength:** the owner's per-seat rule against E5-11-43 at 480/520/560M.
* **Setup:** both sides' own openings within 2 points of the best alternative at 560M.

## Dynamics

* **Setup.** The locked Greece-2 setup broke free within about 10M and overshot to 1.14 nats per
  placement (the floor's coefficient was back at 0 by 320M). The game-result credit then brought
  it down: 0.88 (360M), 0.78 (400M), 0.68 (440M), and 0.65–0.68 from 480M to the end.
* **League.** Five generations, each starting level with its target or 10M behind. Only
  generation 2 found an exploit: 56–60% against the main agent at 370M, three snapshots
  published, never reaching the 65% reset. Generations 1, 3 and 4 spent their budgets below 55%,
  and generation 5 was stopped when the main agent finished. The main agent here was harder to
  exploit than E5-17's, where the first four generations each reached a reset.

## Strength

Round robin (`data/reports/e5_21_rr_43.{md,json}`): E5-21, E5-11 and E5-17 at 480/520/560M,
E4-61@560M and `HeuristicBot`, 100 games per seat per pair, greedy. The bars are E5-11's twin
self-play at each step.

| late block, 9 pairings × 100 per seat | E5-21 | E5-17 (league only) |
|:---|---:|---:|
| overall vs E5-11 | **54.6%** | 49.3% |
| as USSR vs bar 58.2% | 57.3% → **−0.8 ± 2.6** | 54.1% → −4.1 |
| as US vs bar 40.7% | 52.4% → **+11.8 ± 2.6** | 45.3% → +4.7 |
| vs E5-17, head to head | 55.1% | — |
| Elo at 480 / 520 / 560M | **2315 / 2340 / 2344** | 2286 / 2288 / 2308 |

E5-11 is at 2293 / 2303 / 2297 at the same steps, so E5-21 leads by +22, +37 and +47, growing.

**By the letter of the rule, not accepted.** The USSR seat is 0.8 below its bar, a third of a
standard error. The US seat is 4.5 standard errors above its bar. Overall it is +4.6 points
(about +32 Elo) against E5-11 and +5.1 against E5-17. It is the first arm that neither loses the
USSR seat nor stays level.

## Setup

Openings at 560M (`data/reports/e5_21_openings_43.txt`) and the opening check, 2,000 paired
deals (`e5_21_setup_oracle_{us,ussr}_43.txt`):

| side | own opening | alternatives − own (playouts) | critic |
|:---|:---|:---|:---|
| US | WG 3, France 3, Italy 2, Iran 1 (430 of 500), no Greece | −1.7, −1.3, **+2.2 ± 1.5** (human, France 1, Iran 1) | −0.4 to −3.2 |
| USSR | Poland 3, Hungary 3 (500 of 500) | +0.5, −0.9 | −12 |

* **US:** the Greece opening is gone and stays gone, unlike E5-17, where the league alone let it
  return. The best alternative is +2.2 ± 1.5, which misses the pre-registered 2 points by a
  fraction of a standard error. The critic is now calibrated on US openings (within 3 points of
  the playouts).
* **USSR:** Poland 3 + Hungary 3; no alternative is better (+0.5, −0.9). This passes.

## Reading

* **The two mechanisms are complementary, as hoped.**
  * Game-result credit repairs the setup: the US opening moved off Greece and stayed off it.
  * The league keeps weak setups punished: the USSR's per-seat deficit shrank from −5.1 (E5-16)
    and −4.1 (E5-17) to −0.8.
  * The result is the strongest arm on this seed, +4.6 points overall against E5-11, with the
    lead growing from 480M to 560M.
* **Seed 43 only.** The per-seat split against E5-11-43 has repeated the same shape across three
  arms. This arm's US gain (+11.8) is too large for that alone, but it needs a second seed.

## Continued 560 → 720M (2026-09-28): the climb stops

E5-21-43 was continued from its 560M end state with the same flags and league. The owner asked
whether it keeps climbing, and whether a stronger US would push the USSR. The exploiter series
restarted as E5-23-43, after a first driver died on a non-conforming run name, so 560–610M ran
with no exploiter training. The three E5-23 generations published nothing.

**Fixed-reference trajectory** (`data/reports/e5_21_trajectory_43_720.{md,json}`), 200 games per
seat per cell, rating anchored at E5-11-43@240M = 2200:

| snapshot | vs E5-11-43@240M, US / USSR | vs E5-21-43@560M, US / USSR | Elo |
|:---|:---|:---|---:|
| E5-21 480M | 56.0 / 62.5 | 38.5 / 51.0 | 2296 |
| E5-21 560M | 61.0 / 70.5 | — | 2320 |
| E5-21 600M | 66.0 / 61.0 | 47.5 / 57.5 | 2315 |
| E5-21 640M | 67.5 / 76.0 | 45.0 / 57.0 | 2327 |
| E5-21 680M | 62.5 / 68.0 | 47.5 / 57.0 | 2320 |
| E5-21 720M | 46.5 / 61.5 | 41.5 / 48.5 | **2267** |
| E5-11 600 / 640 / 680 / 720M | 55.5–65.5 / 64.5–72.0 | 38.0–50.5 / 45.5–50.5 | 2281 / 2286 / 2328 / 2276 |

**Greedy self-play, US win rate** (500 deals, ±2.2): 49.6% (560M), 49.2% (600M), 40.0% (640M),
49.4% (680M), 41.4% (720M).

**Per seat against E5-11-43 at 640/680/720M** (`e5_21_rr_43_720.{md,json}`): overall **50.7%**;
as USSR −0.9 ± 2.6 against the bar; as US +3.6 ± 2.6.

**Reading.**
* **The rating plateaus from 560M** (2315–2327 through 680M) **and dips at 720M** (2267, lower
  in both seats against both references). Setup entropy rose over the continuation, from
  0.65–0.67 to 0.85 at 720M. That may be the setup loosening again, or one weak snapshot.
* **The US did not keep climbing.** Against the 560M model, the US is level (45–47.5% against
  an equal-strength bar of about 47%). Self-play does not tip past 50%: it oscillates between
  about 40% and 49%.
* **The USSR improved a little from 560M to 680M.** Against the 560M model it scores 57–57.5%,
  above the equal-strength bar of about 53%, by about +4. That is the only sign of the stronger
  US pushing the USSR, and it is small and within two standard errors.
* **Against E5-11 the lead has closed.** 54.6% at 480–560M became 50.7% at 640–720M. E5-11's own
  snapshots vary widely (2276–2328), so part of the difference is noise in the control.
* One seed. The clear gain of the combined recipe is the repaired, stable US opening and the
  +32 Elo it bought by 560M. It did not keep compounding past that with this budget.
