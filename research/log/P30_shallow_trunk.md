# P30 sanity check (2026-09-30): M2d's residual blocks add nothing -- the shallow net is stronger overall

**E6-12-44** is E6-03-44's recipe and seed exactly with `--ladder-res-blocks 0`: observation →
projections → fusion → heads, 4 linear layers against M2d's 12. It has 1.34M parameters against
3.19M, and ran at a median 96k steps/s against 81k. From scratch to 560M, solo. The owner asked
for it as a sanity check after stage A found that the trained trunk keeps less of the state than
an untrained one ([`P30_stage_A_trunk_probes.md`](P30_stage_A_trunk_probes.md)).

## Against E6-03-44, by the owner's rule

Per seat against the neural panel (E6-03-44@80M, @240M, @550M), arm minus control, ± SE. Greedy,
1,000 games per seat per pairing. Readouts at 80M and 200M were taken mid-run.

| block | US | USSR | head to head | Elo (arm, control) |
|:---|---:|---:|---:|:---|
| 60–80M snapshots | −2.8 ± 0.6 | −5.6 ± 0.6 | 43.3% | 1337–1448 against 1397–1483 |
| 180–200M snapshots | +11.5 ± 0.7 | +10.9 ± 0.7 | 56.5% | 1509–1538 against 1449–1482 |
| **500–560M snapshots** | **+6.6 ± 0.6** | **−1.2 ± 0.5** | **56.6%** (US 58.0, USSR 55.2) | 1543–1563 against 1489–1529 |
| **480–560M SWA** | **+4.5 ± 1.1** | **−2.0 ± 1.0** | **53.8%** (US 57.7, USSR 50.0) | 1613 against 1581 |

Reports: `data/reports/p30_shallow_{80M_,200M_,}rr.{md,json}`.

* **Behind early, ahead from 200M on.** At 80M it trailed by about one snapshot. By 200M it led by
  about 60 Elo in both seats, and at 560M it still wins 56.6% head to head and rates about +35 to
  +50 on snapshots and +32 on the SWA.
* **By seat at the plateau:** clearly better as US (+6.6 snapshots, +4.5 SWA). Slightly worse as USSR
  against the panel (−1.2 ± 0.5 snapshots, −2.0 ± 1.0 SWA). On the acceptance rule's 95% intervals,
  the USSR seat is just short of neutral on snapshots (−2.2 to −0.2) and at the boundary on the
  SWA (−4.0 to 0.0). Head to head the shallow net wins as USSR too (55.2% on snapshots, 50.0% on
  the SWA).
* **At matched wall-clock it is further ahead:** it runs about 18% faster.

## What it holds (stage-A probes at 560M)

`data/reports/p30_shallow_state_probe.{md,json}`, self-play of both 560M nets:

| trunk | control | influence exact | card location | hand recall | critic AUC | fresh MLP value head AUC |
|:---|---:|---:|---:|---:|---:|---:|
| E6-12-44@560M (0 blocks) | 88.4% | 70.0 | 80.1 | 33.4 | 0.769 | 0.774 |
| E6-03-44@560M (4 blocks) | 88.8 | 70.5 | 80.5 | 35.1 | 0.771 | 0.780 |

The two trunks hold the same state and give the same critic. The residual blocks do not decide
what the trunk keeps. With no blocks, the trunk is the fusion layer's output, so the state is lost
before the blocks: in the input projections (the whole 84 × 26 board into 256 through one linear
layer, and the 110 × 14 cards likewise). That is where any representation change (P30 stage C)
has to act.

Goal probes (`data/reports/p30_shallow_goal_probes.{md,json}`) show no new blunder pattern: forced
wins taken 64.2% against 62.1%, avoidable losses avoided 95.2% against 94.8%. The US setup differs
(West Germany 4+ never, Iran 2+ always, against the control's reverse).

## Reading

* **M2d's four residual blocks buy no strength at the plateau** on seed 44. They cost 58% of the
  parameters and 16% of the throughput. They helped only early (to about 100M).
* **Depth is not where the trunk loses the state.** The input projections are.
* **Not adopted yet.** One seed, and the USSR seat is borderline under the acceptance rule. A seed-43
  replicate (E6-12-43 against E6-03-43) is the next check.
