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
