# One world against eight: the training searcher's target (2026-10-01)

**Question.** Online search distillation (P15-X4b, E4-28; `ai/training/nash_pg.py`) trains toward
the root visits of a 64-simulation search that samples the hidden cards **once** per search and
spends every simulation in that world. Is that target overdetermined by the world it sampled?

**Answer: no — and the target is almost exactly the network's own prior.** Two one-world targets
agree on the best move 97.2% of the time; the prior's argmax matches the 8-world average 98.8% of
the time, and **never differs once the prior's top move is above 0.8**.

## Setup

* Probe [`../../ai/eval/determinization_targets.py`](../../ai/eval/determinization_targets.py),
  CLI [`../../tools/determinization_probe.py`](../../tools/determinization_probe.py), workflow
  `.github/workflows/determinization_probe.yml` (positions → search over N runners → pool).
* Model **E6-06-44@soup_680-760** (Hugging Face `mihaild/deepstruggle`). 4,000 positions from its
  own self-play at temperature 1 (the rollout temperature), each non-forced decision a candidate
  with probability 1/8 (the training subsample), drawn uniformly from whole games.
* Per position: 8 one-world searches at 64 simulations; 8 at 8 simulations (the equal-budget
  split); one privileged search at 64 (real hidden cards); the prior. Searcher configuration is the
  training searcher's: temperature 0, auto-advance, no root noise, every decision type. Visits on
  actions illegal in the real state are dropped, as `_search_targets` drops them.
* CI run `36797382596`, commit `a22bc49`, engine `5419a582ec07…`. Artifact: `report.md`,
  `report.json`, `rows.jsonl` (every position's raw visit vectors, for re-analysis).

## Result

| group | n | 2 one-world targets agree | one world vs rest, TV | best move varies by world | privileged vs avg, argmax | prior vs avg, argmax |
|:---|---:|---:|---:|---:|---:|---:|
| all | 4000 | 97.2% | 0.034 | 7.0% | 98.0% | 98.8% |
| SELECT_CARD | 1110 | 96.2% | 0.033 | 9.4% | 96.9% | 97.9% |
| POINT_NODE | 1886 | 97.4% | 0.041 | 6.7% | 98.4% | 99.1% |
| SELECT_PLAY_MODE | 844 | 97.9% | 0.021 | 4.9% | 98.3% | 99.2% |

Where search departs from the prior, by how sure the prior is:

| prior's top probability | n | search changes the top move | TV(prior, target) |
|:---|---:|---:|---:|
| < 0.5 | 312 | 8.3% | 0.058 |
| 0.5–0.8 | 958 | 2.4% | 0.044 |
| 0.8–0.95 | 764 | 0.0% | 0.038 |
| ≥ 0.95 | 1,966 | 0.0% | 0.016 |

## Reading

* **Noise is small.** One world's target is within TV 0.034 of the other seven's average; the
  equal-budget 8×8 split is no closer to that average than one world at 64 (TV 0.055).
* **World sensitivity is low and diffuse.** 7% of positions have a world-dependent best move, and
  even there 98% of worlds agree on the modal move. Card choice is the most sensitive type (9.4%);
  no type is a strategy-fusion hotspot.
* **The binding limit is the prior, not the world.** With no root noise, a 64-simulation PUCT
  search never overturns a prior above ~0.8. Distillation then teaches the prior back, and **cannot
  break a locked decision** such as the USSR setup (Hungary at p ≈ 1;
  [`hungary_openings.md`](hungary_openings.md)). Whatever search adds at play time comes from the
  few uncertain decisions.

## Blind spots: root noise on (2026-10-01)

The same probe with Dirichlet root noise in every determinized search (α 1.0), 10,000 positions,
runs `36811501041` (ε 0.25) and `36811506475` (ε 0.5), commit `1817763`. A **candidate** is a
position where search's top move has a prior under 5% and at least 5 of 8 worlds pick it; each is
checked by 256 paired playouts of the model's move against search's, hidden cards redealt per pair.

| ε | search changes the top move | candidates | confirmed |
|---:|---:|---:|---:|
| 0.25 | 1.6% | 1 | 1 |
| 0.5 | 2.5% | 2 | 1 (+ one at 1.5 SE) |

* **Position 1559, the USSR's turn-10 headline** (VP −4, DEFCON 3): the net headlines Missile Envy at
  0.997; search finds South African Unrest (prior 0.003). **+34.0 ± 3.8 points** in the playouts.
  After Missile Envy the USSR loses to its own DEFCON move in ~60% of playouts, almost always at a
  "We Will Bury You" play at DEFCON 2 (the US headlines We Will Bury You in most of them); after
  South African Unrest, never. The chain is not fully traced, and an engine rule error in the
  Missile Envy / We Will Bury You interaction has not been ruled out.
* Position 4656 (USSR, turn 8, Star Wars, prior 0.002): +6.2 ± 4.1, not confirmed.
* **Blind spots exist and are rare** — one confirmed in 10,000 sampled decisions — and the one found
  is decisive and of the known kind (an own-DEFCON loss), set up by a headline.

## What this does not say

* The 8-world sum is the expectation of the one-world target, so this measures **noise**, never
  whether the averaged target is biased; world sensitivity points to where bias could live.
* One model, an SWA, which may be smoother than a raw training snapshot.
* Setup placements (prior ≈ 1) are among the positions and inflate the 0% rows somewhat.
* Not run with a Gumbel or noisy root; the probe takes a searcher config and would answer that in
  ~20 minutes of CI.
