# A model soup at 4,800M from the plateaued E7 line (2026-10-06)

The owner asked, once the pure E7 line had plateaued from ~3,700M
([`E7_line_to_2000M.md`](E7_line_to_2000M.md)), to "resume it from 4400 with two different seeds for
400M, and check soup".

**Runs** ([`../runs.md`](../runs.md)). Two branches of E7-20-44 from its 4,390M state
(`resume_4390060032steps.pt`, the nearest to 4,400M) to 4,800M, E7-20-44's flags with new
randomness only, as E7-03-44 was for the 1,200M soup:

* **E7-20-44-4390M.45**: `--seed-env/--seed-pool/--seed-sampling 45` (launched as E7-24-44 and renamed:
  a branch takes `<parent>-<steps>M.<seed>`, [`../method/run_nomenclature.md`](../method/run_nomenclature.md));
* **E7-20-44-4390M.46**: the same with 46.

`launch_flags --diff` against E7-20-44 shows only the seed flags. They ran one after the other: side
by side they trained at 34k steps/s each, alone at ~100k.

**Soups** (`tools/scripts/average_weights.py`, uniform):

* raw: E7-20-44@4800M + E7-20-44-4390M.45@4800M + E7-20-44-4390M.46@4800M →
  `data/checkpoints/_soups/soup_E7-20-44+4390M.45+4390M.46_4800M.pt` (sha256 `ef6f6384ea29…`);
* SWA-made: the three runs' 4,720–4,800M SWAs →
  `data/checkpoints/_soups/soup_E7-20-44+4390M.45+4390M.46_swa4720-4800M.pt` (sha256 `e2447821624a…`).

## Results

One field, 13 players, 1,000 games per side per pair, temperature 0
(`data/reports/e7_soup4800_rr.{md,json}`). Elo is this field's own scale (the 1,200M soup at 1572),
not the 2026-10-01 field's.

| player | Elo |
|:---|---:|
| **soup, SWA-made** | **1664** |
| **soup, raw 4,800M** | **1658** |
| E7-20-44-4390M.45 SWA 4,720–4,800M | 1622 |
| E7-20-44 SWA 4,720–4,800M | 1619 |
| E7-20-44-4390M.46 SWA 4,720–4,800M | 1617 |
| E7-20-44-4390M.46@4800M | 1592 |
| E7-20-44-4390M.45@4800M | 1590 |
| E7-20-44@4800M | 1585 |
| old best soup (E7-02/03/04/05@1,200M) | 1572 |
| E7-02-44@1000M | 1457 |
| E6-03-44@550M / 240M / 80M | 1315 / 1185 / 1024 |

The owner's rule, per seat on the E6-03-44 panel (± is SE) and head to head:

| comparison | US Δ | USSR Δ | head to head |
|:---|:---|:---|:---|
| raw soup vs E7-20-44@4800M | **+3.7 ± 0.7** | **+3.9 ± 0.8** | **61.0%** ± 1.1 |
| raw soup vs E7-20-44 SWA 4,720–4,800M | +1.8 ± 0.6 | +1.7 ± 0.7 | **54.8%** |
| SWA-made soup vs E7-20-44 SWA | +2.0 ± 0.6 | +2.5 ± 0.7 | **57.1%** |
| raw soup vs the old best soup (1,200M) | +4.8 ± 0.7 | +2.9 ± 0.7 | **63.5%** |
| branch .45@4800M vs E7-20-44@4800M | +0.9 | −0.4 | 52.2% |
| branch .46@4800M vs E7-20-44@4800M | +1.1 | −0.4 | 52.4% |
| branch SWAs vs E7-20-44's SWA | +0.6 | −0.5 | 51.1% |

* **The soup is the strongest model so far, and passes the owner's rule cleanly.** Better in both
  seats against every reference, 61% head to head against its own ingredient E7-20-44@4800M,
  57% against the line's best SWA, and **63.5% against the previous best soup**.
* **Souping three 410M branches of a plateaued run adds more than averaging over time does.** The
  SWA-made soup beats the line's SWA 57.1%; the SWA beats the raw snapshot by about the same margin
  as the 1,200M lineage showed. The soup gain is on top of the SWA gain.
* **The branches are level with the main line** (52% head to head, within a point per seat): new
  randomness for 410M from a plateau does not move strength, it moves the weights to points that
  average well.
* The SWA-made and raw soups are level (1664 against 1658), as at 1,200M.
* The E6-03-44 panel is near its ceiling for these models (88–95% per seat); the head-to-head and
  the old-soup comparison carry the separation.
