# P28 step 1 — SWA beats every snapshot; search headroom shrinks on the adopted model

Plan: [`../plans/P28_strength_on_E6.md`](../plans/P28_strength_on_E6.md), step 1 (owner, 2026-09-29: "go
ahead with these plans"). The plateau trace was taken in step 0
([`P28_step0_panel_and_E6_04_verdict.md`](P28_step0_panel_and_E6_04_verdict.md)).

## SWA

A uniform average of each run's nine snapshots at 480–560M, one every 10M
(`tools/scripts/average_weights.py`), saved as `<run>/swa_480-560M.pt`. The round robin was greedy, 1,000
games per seat, against the three neural panel members and each run's best and final snapshots
(`data/reports/p28_soups_rr.{md,json}`):

| model | Elo | vs panel, US | vs panel, USSR |
|:---|---:|---:|---:|
| **E6-04-44 SWA** | **1651** | **79.5** | **81.9** |
| E6-04-44@520M (best snapshot) | 1566 | 70.9 | 76.4 |
| E6-04-44@560M | 1546 | 67.9 | 71.9 |
| **E6-03-44 SWA** | **1597** | **73.8** | **79.6** |
| E6-03-44@560M | 1536 | 68.2 | 74.3 |
| E6-03-44@550M (trace peak) | 1526 | — | — |

Each vs-panel cell is ± 0.9 (one standard error). Head to head, 1,000 per seat, ± 1.6:

| SWA | vs its best snapshot | vs its final snapshot |
|:---|---:|---:|
| E6-04-44 | 61.7% (@520M) | 63.9% |
| E6-03-44 | 60.1% (@550M) | 58.5% |

**Every SWA beats every snapshot it was made from, by about 60–85 Elo, in both seats.** The plateau
is therefore partly step-size noise: the late snapshots scatter around a better point than any of
them reaches. By the plan's rule this makes weight averaging the default rated model once step 2b
confirms it in training. **The E6-04-44 SWA is now the strongest model on record.**

## Search headroom on E6

Honest search, `search:<ckpt>:256:determinize` (batched MCTS on one sampled world), against the raw
greedy policy of the same net. 100 games per seat, `--pack-pairs 1`
(`data/reports/p28_search256_*.md`):

| model | search's win rate | as US | as USSR |
|:---|---:|---:|---:|
| E6-04-44@520M (adopted) | **53.5% ± 3.5** | 51.0 | 56.0 |
| E6-03-44@550M (plain) | **61.0% ± 3.5** | 57.0 | 65.0 |

* **Step 6's gate (+5 points) passes on the plain model (+11) and fails on the adopted one (+3.5,
  one standard error).** On E5, E5-11-43@560M gave +7.
* The league-and-credit model leaves search less to find, which fits its better-calibrated critic
  (the setup-oracle critic gap was −4.7 against −11.3).
* Step 6 re-measures this on the best step-4/5 model in any case; on today's adopted model there is
  no teacher.
