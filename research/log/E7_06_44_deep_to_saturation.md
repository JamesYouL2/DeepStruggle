# E7-06-44 (2026-10-01): the deep trunk stops improving at about 560M; the shallow one gains another ~120

The owner asked whether a deep model had ever been trained to saturation. None had: plain M2d
stopped at 560M, the league line at 760M. E7-06-44 is E6-03-44 (M2d, 4 residual blocks, no
league, seed 44) resumed from its 560M end state with its own flags, to 1,200M on E7. That makes
it the deep counterpart of E7-01-44 → E7-02-44.

**Both lines traced in one field on E7**, 400 games per seat, about ±15 Elo per point
(`data/reports/e7_06_44_trace_1200.{md,json}`; midway `..._trace_880`):

| steps | 80M | 160M | 240M | 320M | 400M | 480M | 560M | 640M | 720M | 800M | 880M | 960M | 1,040M | 1,120M | 1,200M |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| deep (E6-03-44 → E7-06-44) | 1197 | 1291 | 1368 | 1450 | 1464 | 1487 | 1502 | 1528 | 1499 | 1566 | 1500 | 1535 | 1545 | 1520 | 1542 |
| shallow (E7-01-44 → E7-02-44) | 1276 | 1353 | 1399 | 1457 | 1484 | 1512 | 1532 | 1558 | 1591 | 1613 | 1631 | 1638 | 1646 | 1656 | 1660 |
| shallow − deep | +79 | +61 | +31 | +7 | +20 | +25 | +30 | +29 | +92 | +47 | +130 | +103 | +101 | +137 | +118 |

**At the end, by the owner's rule** (`data/reports/e7_06_44_rr.{md,json}`), shallow against deep:

| reading | US | USSR | head to head |
|:---|---:|---:|---:|
| 1,140–1,200M snapshots | **+14.0 ± 0.5** | **+6.1 ± 0.5** | **64.8%** ± 0.3 |
| 1,120–1,200M SWA | **+6.5 ± 0.9** | +0.7 ± 0.9 | **57.2%** ± 1.1 |

## Reading

* **The deep recipe saturates at about 560M** (about 1,500–1,550 in this field). Its snapshots
  then swing by about ±35 without trend. The shallow recipe keeps gaining to about 900M–1B and
  ends about +120 above it.
* **The shallow trunk is the better base at saturation**, more clearly than at 560M. On snapshots
  it is better in both seats; on the SWA it is better as US and level as USSR. It passes the
  owner's rule on both readings.
* **Averaging narrows the gap but does not close it.** The deep SWA gains more over its snapshots
  (the same pattern as before), so the SWA gap is about 57% against about 65% on snapshots.
* **Consequence for P30:** the shallow trunk is confirmed as the base recipe, on seed 44. The
  earlier comparisons at 560M understated the difference.
