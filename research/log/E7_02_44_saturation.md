# E7-02-44 (2026-10-01): the shallow recipe keeps gaining to ~900M, then flattens

E7-01-44 (the shallow trunk, no league, seed 44, on E7) continued with its flags exactly from its
560M end state to 1,200M. Where does it saturate?

**Trace.** One field, 400 games per seat, about ±15 Elo per point; E6-03-44@550M in the field
(`data/reports/e7_02_44_trace_1200.{md,json}`; the midway trace to 880M is `..._trace_880`):

| steps | 80M | 160M | 240M | 320M | 400M | 480M | 560M | 640M | 720M | 800M | 880M | 960M | 1,040M | 1,120M | 1,200M |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Elo | 1237 | 1314 | 1361 | 1419 | 1451 | 1486 | 1505 | 1529 | 1565 | 1583 | 1603 | 1610 | 1616 | 1626 | 1630 |

* **560M is not a plateau.** The 480→560M step (+19) looked like levelling off, but the run then
  gained about +100 more by 880M (+20 to +36 per 80M).
* **It flattens at about 900M.** From 880M to 1,200M it gains +27 in total (+4 to +10 per 80M),
  close to the trace's noise. The recipe saturates at about 900M–1B steps, roughly 3 hours solo.

**At the end, against E7-01-44** (`data/reports/e7_02_44_rr.{md,json}`; the panel E6-03-44@80/240/550M,
greedy, 1,000 games per seat):

| comparison | US | USSR | head to head | Elo |
|:---|---:|---:|---:|:---|
| 1,120–1,200M SWA against 480–560M SWA | +8.7 | +7.9 | **64.4%** | 1694 against 1584 |
| 1,200M snapshot against 560M snapshot | +11.3 | +11.4 | **66.3%** | 1652 against 1533 |

The extra 640M steps are worth about +110 Elo, in both seats equally.

## Reading

* **Every arm judged at 560M was judged before its recipe saturated.** That includes every P28
  and P29 arm and the shallow/deep comparisons. The ordering of recipes at 560M need not hold at
  saturation: the deep line also kept gaining to 760M (E6-06/07/08-44).
* **Open:**
  * where E7-02-44's SWA stands against the current best, the shared-start soup of
    E6-06/07/08-44 at 760M. They have never been in one field.
  * whether the deep recipe also saturates near 900M, and above or below the shallow one.

## Against the current best (2026-10-01)

One field on E7, greedy, 1,000 games per seat, anchored at HeuristicBot = 1500
(`data/reports/e7_long_shallow_vs_best.{md,json}`):

| Elo | model | panel US / USSR |
|---:|:---|:---|
| **2571** | **E7-02-44 SWA 1,120–1,200M** (shallow, no league, one run) | 86.3 / 86.3 |
| 2562 | shared-start soup of E6-06/07/08-44 at 760M (the designated best model soup) | 85.0 / 86.5 |
| 2562 | the same soup made from their 680–760M SWAs | 85.6 / 86.2 |
| 2526 | **E7-02-44@1,200M** (raw snapshot) | 82.7 / 83.6 |
| 2521 | E6-06-44 SWA 680–760M (the designated best SWA) | 82.9 / 81.8 |
| 2510 | E6-08-44 SWA 680–760M | 78.3 / 84.3 |
| 2499 | E6-04-44 SWA 480–560M | 81.1 / 82.0 |
| 2451 | E6-07-44@700M (the designated best raw snapshot) | 73.3 / 77.0 |

Head to head (seat-paired, ±1.1):

| | the soup | the soup of SWAs | E6-06-44 SWA | E6-08-44 SWA | E6-04-44 SWA | E6-07-44@700M |
|:---|---:|---:|---:|---:|---:|---:|
| **E7-02-44 SWA** | **50.8%** | 52.8% | 58.0% | 59.5% | 61.1% | 65.9% |
| **E7-02-44@1,200M** | 43.9% | 44.1% | 51.6% | 52.6% | 51.4% | **60.9%** |

* **E7-02-44's SWA is level with the best model** (50.8% against the soup, 52.8% against the
  soup of SWAs). It is one run of the shallow recipe with no league, against an average of three
  branches of the deep + league line. It beats the designated best SWA 58.0%.
* **E7-02-44@1,200M is the strongest raw snapshot measured.** It beats the designated best raw
  snapshot (E6-07-44@700M) 60.9% and is level with the 760M deep SWAs.
* **The designations are the owner's call.** The candidates are: best raw E7-02-44@1,200M, best
  SWA E7-02-44 1,120–1,200M. For the best overall model, the soup and E7-02-44's SWA are level.
* **Next lever.** A shallow model soup: branch E7-02-44 at its saturation point into a few
  continuations, as E6-06/07/08 branched from E6-04-44. On the deep line that added +42 over its
  best SWA.
