# Model soups (2026-09-30): across seeds they break, from a shared start they are the best models yet

The owner pointed out that what P28 called a "soup" is **SWA**: a uniform average of one run's
snapshots along its trajectory. A **model soup** averages separately trained models. Two tests:

* **Cross-seed:** E6-03-43 + E6-03-44 (seeds 43 and 44). Each was trained from scratch from its own
  initialisation (`--seed-init`).
* **Shared start:** E6-06-44 + E6-07-44 + E6-08-44. All three branched from E6-04-44's 560M state
  and trained 200M more differently: constant rate, stepped rate, and EMA snapshots.

Each combination was averaged both raw and as SWAs (`tools/scripts/weight_soup.py`; the files are
in `data/checkpoints/_soups/`). Everything was rated in one field with the ingredients and the
neural panel: greedy, 500 games per seat, 17 models (`data/reports/model_soups.{md,json}`).

| model | Elo | notes |
|:---|---:|:---|
| **shared-start soup of the three 760M snapshots** | **1775** | 69.5% vs E6-06-44@760M, 64.9% vs E6-08-44@760M; panel US 86.2 / USSR 86.4 |
| **shared-start soup of the three 680–760M SWAs** | **1773** | 54.3% vs E6-06-44's SWA, 58.1% vs E6-08-44's; panel US 86.7 / USSR 85.1 |
| E6-06-44 SWA 680–760M (the previous best souped model) | 1730 | panel US 82.4 / USSR 81.1 |
| E6-08-44 SWA 680–760M | 1723 | |
| E6-07-44 SWA 680–760M | 1677 | |
| E6-08-44@760M / E6-07-44@760M / E6-06-44@760M | 1676 / 1650 / 1620 | raw snapshots |
| E6-03-43 SWA / E6-03-44 SWA 480–560M | 1655 / 1653 | |
| E6-03-44@560M / E6-03-43@560M | 1585 / 1584 | |
| **cross-seed soup of the two 480–560M SWAs** | **591** | 0.2% vs either ingredient |
| **cross-seed soup of the two 560M snapshots** | **491** | 0.1–0.3% vs either ingredient |

The two shared-start soups are level head to head: 50.0% (US 44.0, USSR 56.0).

## Reading

* **Across seeds, averaging destroys the network.** The result loses 99.7% or more to both
  ingredients. Two nets trained from different initialisations represent the same features in
  different units, so their average mixes unrelated weights. This is the usual model-soup
  condition (ingredients need a shared start), confirmed here.
* **From a shared start, a model soup beats every ingredient and every SWA.**
  * The three-way soup rates +45 over the best single SWA (E6-06-44's).
  * It beats the ingredient snapshots 65–70%.
  * It improves both seats against the panel by about 4–5 points.
  * The three branches (constant rate, stepped rate, EMA) end in one basin, and their average sits
    at a better point than any of them. That extends the SWA result: the variation between
    branches averages out as the variation along one trajectory does.
* **Whether to soup raw snapshots or SWAs does not matter here.** The two soups are level (1775 /
  1773, 50.0% head to head), so souping the three runs' final snapshots is enough.
* **Consequence for recipes.** Branching a trained state into a few differently-seeded or
  differently-tuned continuations and souping them is a cheap strength lever. On this evidence it
  is worth about +45 Elo over the best SWA at the cost of the extra branches. Per the owner's rule,
  averaging stays an evaluation device and does not enter training.
