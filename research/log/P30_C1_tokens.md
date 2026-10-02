# P30 C1 (2026-10-02): card and country tokens with attention -- readouts

**Arm.** E7-09-44: E7-01-44's shallow recipe, seed 44, plus `--ladder-token-layers 2 --ladder-token-dim 128`
(195 tokens: 84 countries, 110 cards, one global; each `W·row + b + identity`; two pre-norm
transformer layers, bf16) and `--compile-update default` ([`../runs.md`](../runs.md)). ~10.4k steps/s
compiled (the shallow recipe alone: ~96k). To 400M first.

## Identity vectors grow slowly

Initialised at 0.02 per coordinate. Mean norms, against the projected state part of the token:

| step | card ‖W·row+b‖ / ‖id‖ | country ‖W·row+b‖ / ‖id‖ |
|---:|---:|---:|
| init | 3.73 / 0.23 | 2.44 / 0.23 |
| 20M | 3.21 / 0.64 | 2.17 / 0.57 |
| 40M | 3.31 / 0.88 | 2.30 / 0.76 |

## 80M (owner: "a small tournament on 80M against 80M and 160M snapshots" of the shallow arm)

One field, 1,000 games per side per pairing, with the panel (E6-03-44@80/240/550M):

| model | Elo |
|:---|---:|
| E7-01-44@160M (shallow, seed 44) | 1563 |
| E7-01-44@80M (shallow, seed 44 -- the control) | 1476 |
| E7-07-44@80M (shallow + C4, seed 44) | 1411 |
| E7-08-43@80M (shallow, seed 43) | 1410 |
| **E7-09-44@80M (C1)** | **1404** |

| C1@80M against | US (panel) | USSR (panel) | head to head |
|:---|---:|---:|---:|
| E7-01-44@80M (paired control) | −8.1 ± 1.1 | −6.0 ± 1.1 | 38.7% |
| E7-01-44@160M | | | 29.0% |
| E7-07-44@80M | +1.1 ± 1.0 | −4.5 ± 1.1 | 47.9% |
| E7-08-43@80M | +3.3 ± 1.0 | −2.7 ± 1.1 | 48.9% |
| *reference: E7-07-44@80M against E7-01-44@80M* | −9.2 | −1.5 | 36.2% |
| *reference: E7-08-43@80M against E7-01-44@80M* | −11.4 | −3.3 | 40.0% |

* **C1 is behind its paired control at 80M (38.7%), but E7-01-44@80M is an outlier:** the other two
  shallow runs at 80M (C4 and the seed-43 control) lose to it by as much (36.2%, 40.0%). Against
  those two C1 is level (48–49%).
* **So at 80M C1 is an ordinary shallow-recipe run** -- neither the attention gain nor a cost from
  its slow start is visible. 80M is early (the recipe saturates at ~1B); next look at 160M.
* Reports: `data/reports/e7_09_44_rr_80{,b}.{md,json}`.

## 200M (owner: "check C1@200M strength with matching stages from no-attention shallow trunk")

One field, 1,000 games per side per pairing, panel E6-03-44@80/240/550M (`--batch-chunk-size 250`:
the full-size batch ran out of GPU memory next to C1's own training):

| model | Elo |
|:---|---:|
| **E7-09-44@200M (C1)** | **1563** |
| E7-08-43@200M (shallow, seed 43) | 1533 |
| E7-01-44@240M (shallow, seed 44) | 1511 |
| E7-01-44@200M (shallow, seed 44 -- the paired control) | 1503 |
| E7-07-44@200M (shallow + C4) | 1499 |
| E7-01-44@160M | 1471 |
| E7-09-44@160M (C1) | 1424 |

| C1@200M against | US (panel) | USSR (panel) | head to head |
|:---|---:|---:|---:|
| E7-01-44@200M (paired control) | **+4.6 ± 1.3** | **+8.8 ± 1.2** | **56.0%** |
| E7-07-44@200M | +10.1 ± 1.2 | +6.0 ± 1.2 | 58.2% |
| E7-08-43@200M | +6.1 ± 1.2 | +4.7 ± 1.2 | 57.5% |
| *reference: E7-07-44@200M against E7-01-44@200M* | −5.5 | +2.8 | 50.3% |
| *reference: E7-08-43@200M against E7-01-44@200M* | −1.5 | +4.1 | 53.4% |

* **At 200M C1 is ahead of every shallow run at the same step, in both seats** -- by more than the
  shallow runs differ among themselves (50–53% between them, 56–58% for C1), and above the control's
  own 240M snapshot.
* **But it is one snapshot, and the jump is large:** C1@160M was *below* the control at 160M (1424
  against 1471), so C1 gained ~140 Elo in 40M where the control gained ~30. Single snapshots carry
  real noise; to be read as a lead it needs several snapshots (e.g. 190–210M, or 4 at 240M).
* Reports: `data/reports/e7_09_44_rr_200.{md,json}`.

## 240M, four-snapshot SWAs (owner: "SWA of four C1 snapshots vs SWA of four shallow run snapshots")

Each run's 210/220/230/240M snapshots averaged; one field with the panel, 1,000 games per side:

| model | Elo |
|:---|---:|
| **E7-09-44 (C1), 210–240M SWA** | **1573** |
| E7-08-43 (shallow, seed 43), 210–240M SWA | 1544 |
| E7-07-44 (shallow + C4), 210–240M SWA | 1526 |
| E7-01-44 (shallow, seed 44, paired control), 210–240M SWA | 1519 |

| C1 SWA against | US (panel) | USSR (panel) | head to head |
|:---|---:|---:|---:|
| E7-01-44 SWA (paired control) | **+5.5 ± 1.2** | **+4.6 ± 1.1** | **57.1%** |
| E7-07-44 SWA | +3.8 ± 1.2 | +3.4 ± 1.1 | 55.0% |
| E7-08-43 SWA | +5.0 ± 1.2 | −0.5 ± 1.1 | 58.3% |
| *reference: E7-07-44 against E7-01-44* | +1.7 | +1.2 | 50.8% |
| *reference: E7-08-43 against E7-01-44* | +0.5 | +5.2 | 55.2% |

* **The 200M lead holds on SWAs.** C1 passes the owner's rule against its paired control (both seats
  significantly above) and against the other two runs (above in at least one seat, below in none),
  and is the top model of the field, +29 Elo over the best shallow SWA and +54 over its control.
* **Size against the seed spread:** the shallow SWAs span 25 Elo among themselves (55.2% at most head
  to head); C1 is 55–58% against each of them. A lead at the edge of, or beyond, the seed spread --
  real but not yet a large margin, on one seed of C1.
* Reports: `data/reports/e7_09_44_rr_swa240.{md,json}`.

## 400M: the lead is gone (pre-registered readout)

One field, 1,000 games per side per pairing: the panel, each run's 370/380/390/400M snapshots and
its last-80M SWA (320–400M).

| SWA (320–400M) | Elo |
|:---|---:|
| E7-09-44 (C1) | 1576 |
| E7-07-44 (shallow + C4) | 1572 |
| E7-01-44 (shallow, seed 44, paired control) | 1570 |
| E7-08-43 (shallow, seed 43) | 1554 |

Late snapshots: C1 1495–1503, the shallow runs 1476–1532.

| C1 against | snapshots: US / USSR (panel), head to head | SWA: US / USSR, head to head |
|:---|:---|:---|
| E7-01-44 (paired control) | **−4.6 ± 0.6 / −2.1 ± 0.6**, 49.3% | −2.6 ± 1.1 / −3.2 ± 1.1, 51.2% |
| E7-07-44 | −4.9 / −4.0, 47.7% | −1.2 / −4.6, 49.3% |
| E7-08-43 | −1.6 / −5.1, 49.4% | +0.7 / −6.1, 51.4% |

* **Behind on snapshots, level on the SWA.** On the late snapshots C1 is significantly below its
  paired control in both seats (head to head 49.3%). On the SWA it tops the field by 6 Elo and wins
  head to head 51.2% ± 1.1 (not significant), while scoring 2.6 / 3.2 points lower against the
  panel (~2.5 SE): level within the noise, either way no lead.
* **C1 gains the most from averaging:** SWA over late snapshots +77 Elo, against +50 to +65 for the
  shallow runs -- its snapshots swing more, as the owner expected of the larger model.
* **The 200–240M lead was a faster start, not a higher level.** From 240M to 400M the shallow runs'
  SWAs rose ~45–50 Elo in this field's scale while C1's barely moved: C1 learns faster early and
  then flattens.
* **Seat split:** C1 is weaker as US (40–46% head to head) and stronger as USSR (52–59%).
* **Cost:** at ~10k steps/s against ~96k, C1 reaches the same strength per step at ~9× the compute.
* **Decision rule (registered at launch):** continue to saturation only if ahead at 400M. It is not;
  C1 stops here. C1b (concatenated tokens, identities at full scale) remains the cheaper variant to
  try if the owner wants another attention arm.
* Reports: `data/reports/e7_09_44_rr_400.{md,json}`.
