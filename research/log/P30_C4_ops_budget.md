# P30 C4 (2026-10-01): the ops budget in the observation -- intermediate readout at 320M

**Arm.** E7-07-44: E7-01-44's shallow recipe from scratch, seed 44, plus `--obs-features ops_budget`
-- 3 floats appended after the base layout: the Ops the card at a play-mode decision grants after
every modifier (/5), and each side's per-card Ops modifier ([`../runs.md`](../runs.md)).
**Control.** E7-01-44 (same recipe, seed 44). **Seed reference.** E7-08-43, the same recipe as the
control on seed 43, compared with E7-01-44 in exactly the same way: what a difference of seed alone
looks like.

Asked for by the owner at 320M ("Lets have an intermediate tournament at 320M"); the arm runs on to
1,200M and is judged at saturation.

## Same-field trace (400 games per side per pairing, temperature 0)

| M | C4 E7-07-44 | control E7-01-44 | seed 43 E7-08-43 | C4 − control | seed 43 − control |
|---:|---:|---:|---:|---:|---:|
| 80 | 1301 | 1400 | 1299 | −98 | −101 |
| 160 | 1446 | 1475 | 1518 | −29 | +43 |
| 240 | 1550 | 1519 | 1555 | +31 | +36 |
| 320 | 1609 | 1603 | 1589 | +6 | −15 |

## Owner's rule (per seat against the panel E6-03-44@80/240/550M, arm minus control; 1,000 games per side)

| comparison | US | USSR | head to head |
|:---|---:|---:|---:|
| **C4 against control**, 290–320M snapshots | +1.0 ± 0.6 | **+4.5 ± 0.6** | 51.4% ± 0.3 |
| **C4 against control**, 240–320M SWA | **+7.5 ± 1.2** | **+4.7 ± 1.1** | 51.8% ± 1.1 |
| *seed 43 against seed 44, same recipe*, 290–320M snapshots | −3.8 ± 0.6 | +3.9 ± 0.6 | 52.9% ± 0.3 |
| *seed 43 against seed 44, same recipe*, 240–320M SWA | +1.8 ± 1.2 | +3.9 ± 1.1 | 61.1% ± 1.1 |

## Reading

* **By the rule as written, C4 passes at 320M:** significantly above the control in the USSR seat on
  snapshots and in both seats on the SWA, and nowhere below it.
* **But a change of seed alone does as much.** The seed-43 control, with no change at all, sits
  +3.9 above seed 44 in the USSR seat and beats it head to head by more than C4 does (52.9% on
  snapshots, 61.1% on the SWAs). The standard errors cover game noise only; the run-to-run spread of
  one recipe is several points per seat at 320M. C4's gain is inside it.
* **So no evidence yet that the ops budget helps, and none that it hurts.** The trace agrees: every
  difference after 80M is within ±45 Elo, for C4 and for the seed alone.
* **What would decide it:** saturation (~1B) for both lines, and C4 on seed 43 compared with
  E7-08-43, so the arm is judged against seed spread rather than game noise.
* Reports: `data/reports/e7_07_44_trace_320.{md,json}`, `data/reports/e7_07_44_rr_320.{md,json}`.

## At saturation (1,200M, 2026-10-02)

Trace (one field, 400 games per side per pairing; the control is E7-01-44 to 560M, E7-02-44 after):

| M | C4 E7-07-44 | control (seed 44) | seed 43 E7-08-43 | C4 − control | seed 43 − control |
|---:|---:|---:|---:|---:|---:|
| 80 | 1171 | 1261 | 1164 | −91 | −98 |
| 240 | 1410 | 1389 | 1398 | +21 | +10 |
| 400 | 1499 | 1481 | 1481 | +18 | +1 |
| 560 | 1507 | 1531 | 1529 | −25 | −2 |
| 720 | 1548 | 1590 | 1552 | −41 | −37 |
| 880 | 1556 | 1627 | 1578 | −71 | −49 |
| 1,040 | 1594 | 1643 | 1608 | −49 | −35 |
| 1,200 | 1624 | 1661 | 1604 | −37 | −56 |

Owner's rule at 1,140–1,200M (1,000 games per side per pairing):

| comparison | US | USSR | head to head |
|:---|---:|---:|---:|
| C4 against the seed-44 control (E7-02-44), snapshots | −1.6 ± 0.5 | **−5.1 ± 0.5** | 44.0% |
| the same, 1,120–1,200M SWA | +1.0 ± 0.9 | −3.5 ± 0.9 | 43.8% |
| *seed 43 against seed 44, unchanged recipe*, snapshots | −6.6 ± 0.5 | −2.8 ± 0.5 | 43.2% |
| *the same, SWA* | −5.6 ± 0.9 | −2.9 ± 0.9 | 43.4% |
| C4 against the seed-43 control, snapshots | +5.1 ± 0.5 | −2.3 ± 0.5 | 53.7% |
| the same, SWA | +6.6 ± 0.9 | −0.6 ± 0.9 | 53.8% |

Elo in the round robin: SWAs E7-02-44 1609, **C4 1582**, E7-08-43 1554; late snapshots ~1567, ~1533, ~1510.

* **By the rule against its paired control, C4 fails** (USSR −5.1 on snapshots, 44% head to head).
* **But the unchanged recipe on seed 43 fails against seed 44 by as much** (43.2%). Seed 44 is a
  strong run: the two controls are 55 Elo apart at saturation, and C4 lies between them (−27 / +28).
* **Reading: no detectable effect of the ops budget at saturation.** With one seed per arm the
  seed-to-seed spread (~55 Elo) is larger than any effect C4 could plausibly have. A decision would
  need C4 on seed 43 (paired with E7-08-43), and probably more seeds per arm for every P30 arm.
* Not pursued further unless the owner wants the seed-43 replicate; reports
  `data/reports/e7_07_44_{trace,rr}_1200.{md,json}`.
