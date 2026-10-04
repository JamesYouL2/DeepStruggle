# P30 (2026-10-04): training with search targets, as a branch -- strength moves from USSR to US

The owner asked for a 400M-step training run with search starting from 800M, after search moved to
C++ ([`search_performance_profile.md`](search_performance_profile.md)) and honest search was measured
at +8 on a single E7 snapshot ([`P30_search_headroom_e7.md`](P30_search_headroom_e7.md)).

**Arm.** E7-18-44: E7-02-44 branched at 820M (`resume_820051968steps.pt`) to 1,220M with its flags,
plus search cross-entropy targets at P15 X4b's settings -- the only ones ever measured positive (+166
Elo at 20M on E3): `--search-ce-coef 0.5 --search-sims 64 --search-node-filter all
--search-subsample 0.125`. An honest (determinized) 64-simulation search answers 1 in 8 decisions,
in one batch at the end of each rollout; rollouts act on the raw policy. About 26k steps/s.

**Control.** The same line without search: E7-02-44 to 1,200M, then E7-17-44. **Branch reference.**
E7-03-44, a branch with new seeds only. Engine d81e54f0.

## Through the run (400 games per side, single snapshots)

E7-18-44 minus the control, per seat against the panel E6-03-44@80/240/550M, and head to head:

| M | US | USSR | head to head |
|---:|---:|---:|---:|
| 900 | −1.2 ± 1.6 | +0.9 ± 1.6 | 51.2% ± 1.8 |
| 980 | +2.0 ± 1.5 | +1.6 ± 1.5 | 46.9% ± 1.8 |
| 1,060 | +4.1 ± 1.5 | −2.4 ± 1.5 | 53.6% ± 1.8 |
| 1,140 | +1.5 ± 1.5 | +1.7 ± 1.5 | 51.2% ± 1.8 |

Policy entropy rose from the control's 0.41 to 0.52 in the first 20M (the search targets' own
entropy was 0.49) and eased back to ~0.45–0.48 by the end -- the opposite direction to the E3
collapse.

## At the end: the owner's rule (1,000 games per side)

`data/reports/e7_18_rr.{md,json}`.

| comparison | US | USSR | head to head |
|:---|---:|---:|---:|
| **E7-18-44**, 1,160–1,220M snapshots | +1.6 ± 0.5 | **−3.0 ± 0.5** | 51.0% ± 0.3 |
| **E7-18-44**, 1,140–1,220M SWA | +2.6 ± 0.9 | **−2.3 ± 0.9** | 50.8% ± 1.1 |
| *E7-03-44 (new seeds only)*, SWA | *+2.0 ± 0.9* | *+0.1 ± 0.9* | *48.0% ± 1.1* |

SWAs: E7-18-44 1600, control 1596, E7-03-44 1595.

## Reading

* **It does not pass the owner's rule.** Worse in the USSR seat on both measures, by 3–6 standard
  errors -- beyond the branch reference, which shows no USSR difference -- and better in the US seat
  by about as much as a change of seeds alone gives (+2.0).
* **Net strength is level:** 51% head to head, SWAs within 4 Elo.
* **So search targets at these settings move strength from the USSR seat to the US seat rather
  than adding it.** One lead to check before another arm: whether the search teaches the two seats
  differently -- per-seat target entropy, search/policy agreement and dropped-visit rates. The
  coefficient (0.5) has never been swept, and 64 simulations on 1 in 8 decisions is a far weaker
  teacher than the 256-simulation, every-decision search measured at +8.
