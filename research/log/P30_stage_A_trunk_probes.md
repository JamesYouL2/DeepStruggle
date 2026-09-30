# P30 stage A (2026-09-30): the trunk drops much of the present, but the critic is not short of it

Follow-up to [`P29_bet2_ownership_probes.md`](P29_bet2_ownership_probes.md). There, no trained trunk
encoded more about the *final* board than its own random initialisation. Stage A asks, on frozen
trunks, (1) whether the trunk holds the *present* state and (2), the gate, whether the critic is
limited by what the trunk drops. Tool: `tools/scripts/trunk_state_probe.py`. Every probe
standardises its inputs and stops early on held-out games; all scores are on held-out games.

## 1. The present: reading the current state back from the trunk

Self-play of E6-03-44@560M and E6-07-44@700M, 1,500 games each; 66k fitting and 16k test positions
(`data/reports/p30_a_state_probe.{md,json}`). MLP probes on the frozen hidden vector:

| source | hidden length | control | influence exact | card location | "in my hand" recall / precision |
|:---|---:|---:|---:|---:|---:|
| raw observation (sanity check) | — | 99.8% | 98.4 | 99.2 | 96.6 / 97.6 |
| M2d, untrained (E6-03-44@0) | 29 | 92.1 | 73.2 | 85.2 | 43.8 / 67.0 |
| M2d, E6-03-44@560M | 26 | 89.9 | 72.0 | 80.4 | 34.6 / 67.2 |
| M2d, E6-07-44@700M (best raw) | 36 | 89.6 | 71.9 | 80.0 | 33.5 / 66.8 |
| wide 768 × 8, E6-10-44@560M | 1,752 | 84.2 | 68.2 | 73.1 | **11.5** / 69.1 |
| ColdWarNetV2 (per-country encoder, pooled), E4-03-01@160M | 47 | 88.8 | 71.6 | 79.0 | 33.2 / 72.7 |

* **Trained trunks hold less of the present than untrained ones.** For the best model, the trunk
  tells you which of your own cards are in hand about a third of the time. The wide trunk is
  the worst on every column.
* **The old per-country encoder keeps no more.** ColdWarNetV2 (a shared per-country encoder, then
  pooling) reads the same as M2d, so a per-country encoder alone is not the remedy.
* The per-country policy heads read each country's raw row directly, so this loss hurts them only
  through the 64-wide trunk context. The card head and the critic read the trunk alone.

## 2. The gate: is the critic limited by what the trunk drops?

Predicting the game's winner from the mover's side (draws left out). The critic's `v_win` is
rescaled by a fitted logistic so its calibration is not held against it; AUC needs no
calibration. Self-play of E6-03-44@560M and E6-07-44@700M, **12,000 games each; 457k fitting and
114k test positions** (`data/reports/p30_a_value_concat_big.{md,json}`):

| predictor | AUC | log-loss | turns 8–10 AUC |
|:---|---:|---:|---:|
| MLP on the raw observation, from scratch | 0.787 | 0.556 | 0.863 |
| E6-03-44@560M, its own critic | 0.794 | 0.550 | 0.897 |
| ... fresh MLP head on its frozen trunk | 0.805 | 0.534 | 0.897 |
| ... fresh MLP head on its frozen trunk **+ the raw observation** | 0.802 | 0.537 | 0.891 |
| E6-07-44@700M, its own critic | 0.808 | 0.535 | 0.901 |
| ... fresh MLP head on its frozen trunk | 0.808 | 0.531 | 0.898 |
| ... fresh MLP head on its frozen trunk **+ the raw observation** | 0.803 | 0.537 | 0.892 |

On the 16k-position sample, trained trunks read 0.815 against 0.734 for an untrained one.

* **The critic uses what the trunk has.** A fresh head on the frozen trunk matches the critic
  (0.808 / 0.808 for the best model).
* **The state the trunk drops does not help predict the winner.** Adding the whole raw observation
  to the trunk's vector gives nothing (0.803 against 0.808), at 457k positions as at 66k.
* **Training shapes the trunk for the outcome.** It holds far more value than an untrained one
  (0.815 against 0.734), while holding less of the board. It keeps what predicts the result and
  sheds the rest.
* **Limits.** The raw-observation learner is still climbing with data (0.752 at 66k and 0.787 at
  457k), so a much larger set could find value the trunk misses. The outcome also carries dice
  and hidden cards, so the ceiling is well below 1. This rules out a large, easily recovered
  deficit, not a small one.

## Reading for P30

* **The gate fails for the critic.** Nothing here says a trunk holding more state would give a
  better value estimate. Stage C (per-country tokens) cannot be justified by the critic.
* **What remains is the policy.** Card decisions read only the trunk, and the trunk barely knows
  the hand. That is a policy-side case, and only training can test it.
* **The wide trunk's growth is separate and concrete.** Its hidden vector grows about 38× and it
  holds the least state of any trunk, which bears on stage B (normalisation) and on why bet 1
  lagged.
