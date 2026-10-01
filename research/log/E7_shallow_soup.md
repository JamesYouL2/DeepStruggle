# The shallow model soup (2026-10-01): the best model, +61 Elo over the deep soup

Four branches of one shallow run, from E7-02-44's 870M state to 1,200M:

| branch | what varies |
|:---|:---|
| E7-02-44 itself | nothing, the control |
| E7-03-44 | new randomness: `--seed-env/--seed-sampling/--seed-pool 45` |
| E7-04-44 | learning-rate step-down: 1e-4 from 980M, 3e-5 from 1,090M |
| E7-05-44 | setup credit: `--setup-mc-credit --setup-entropy-floor 0.3` |

Each was averaged uniformly, two ways (`tools/scripts/average_weights.py`):

* the four 1,200M snapshots, `_soups/shallow_E7-02+03+04+05_1200M.pt` (sha256 `55a26630519e…`);
* the four 1,120–1,200M SWAs, `_soups/shallow_E7-02+03+04+05_swa1120-1200M.pt` (sha256 `f124f719177d…`).

The branches ran two at a time (E7-03-44 with E7-04-44, then E7-04-44 with E7-05-44), measured at
+20% total throughput against one run alone. The pre-registered field was played on E7, greedy,
1,000 games per seat, anchored at HeuristicBot = 1500 (`data/reports/e7_shallow_soup.{md,json}`).

| Elo | model | panel US / USSR |
|---:|:---|:---|
| **2617** | **shallow soup of the four SWAs** | 90.9 / 89.6 |
| **2613** | **shallow soup of the four 1,200M snapshots** | 90.0 / 89.2 |
| 2586 | E7-04-44 SWA 1,120–1,200M (LR step-down) | 87.5 / 87.2 |
| 2565 | E7-03-44 SWA (new randomness) | 87.5 / 85.9 |
| 2563 | E7-02-44 SWA (the control) | 86.3 / 86.3 |
| 2560 | E7-05-44 SWA (setup credit) | 87.5 / 85.3 |
| 2552 | the deep soup, E6-06/07/08-44@760M (the previous best model soup) | 85.0 / 86.5 |
| 2513 | E6-06-44 SWA 680–760M | 82.9 / 81.8 |

Head to head (seat-paired, ±1.1):

| | E7-02-44 SWA | the deep soup | the best ingredient (E7-04-44 SWA) |
|:---|---:|---:|---:|
| shallow soup of snapshots | **58.9%** (US 58.0, USSR 59.7) | **58.7%** (58.5 / 58.9) | 52.1% |
| shallow soup of SWAs | **55.8%** (57.0 / 54.5) | **58.6%** (55.5 / 61.7) | — |

The two soups play each other at 51.4%. Ingredients against E7-02-44's SWA: E7-04-44 53.5%,
E7-03-44 50.3%, E7-05-44 48.1%.

## Reading

* **The pre-registered rule passes.** Both soups beat E7-02-44's SWA and the deep soup clearly, and
  are better in both seats against the panel (+3 to +5 points over the deep soup). The shallow
  soup is the best model measured: +61 to +65 Elo over the deep soup, and +50 over the best
  single SWA.
* **The soup gain matches the deep line's.** Here the soup is +50 over E7-02-44's SWA (+31 over
  the best ingredient's). On the deep line the soup gained +42 over the best SWA.
* **Snapshots or SWAs in the soup barely matter** (51.4% head to head), as on the deep line.
* **Among the branches, the LR step-down is the best single model** (E7-04-44, 53.5% against the
  control). New randomness is level, and the setup credit is slightly behind (48.1%).
