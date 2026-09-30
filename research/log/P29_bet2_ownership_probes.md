# P29 bet 2, mid-run (2026-09-30): the ownership target has not reached the trunk

E6-11-44 (bet 2: E6-03-44's recipe plus `--aux-ownership 0.1 --aux-vp-margin 0.1`) was probed at
100M and 200M, at the owner's question: does the ownership head know more than a per-country habit
(Australia is the US's, North Korea the USSR's), and does it call the games where the habit fails?

Baselines on the same positions: **constant**, the country's usual final controller; **current**,
its controller at the position. A **surprise** is a country that did not end with its usual
controller.

## 1. The online head is the per-country prior

`tools/scripts/aux_ownership_probe.py`, 2,000 self-play games at temperature 1.0, ~53k positions
(`data/reports/p29_bet2_ownership_probe{,_100M}.{md,json}`):

| E6-11-44 | head | constant | current | head, surprises | current, surprises |
|:---|---:|---:|---:|---:|---:|
| @100M | 72.1% | 70.9 | 72.7 | 20.6 | 42.9 |
| @200M | 72.8% | 71.1 | 73.0 | 22.2 | 48.1 |
| @200M, turns 8–10 | 74.4% | 70.8 | **86.8** | 31.0 | 69.5 |

* It beats the constant only in contested countries (South Korea 69% against 49%, Italy 61 against
  51, Greece 54 against 43). Elsewhere it is the constant (Japan 84.5 / 84.6, North Korea
  81.9 / 82.0).
* On surprises in near-certain countries it is near 0% (Japan 0.2, Israel 0.3, North Korea 1.0,
  East Germany 0.0). Japan's surprises were already on the board at the position 93% of the time.
* Late in the game it is far below just reading the board, and it barely moved from 100M to 200M.

## 2. No trunk encodes it, trained or not

`tools/scripts/trunk_ownership_probe.py`: each trunk frozen, a fresh head fitted to convergence on
its hidden vector (linear, and an MLP of the aux head's shape; inputs standardised per feature,
early stopping on held-out games), on the same positions: self-play of E6-11-44@200M and
E6-03-44@200M, 1,500 games each, 16k test positions from 702 held-out games
(`data/reports/p29_bet2_trunk_probe.{md,json}`).

| predictor | all | log-loss | turns 8–10 | surprises |
|:---|---:|---:|---:|---:|
| constant | 68.7% | 0.681 | 68.0 | 0 |
| current controller | 73.2 | — | **87.2** | **48.8** |
| **raw observation, MLP from scratch** | **76.8** | **0.552** | 84.5 | 43.5 |
| E6-11-44@200M trunk (aux-trained), MLP probe | 72.8 | 0.620 | 74.7 | 33.6 |
| E6-03-44@200M trunk (control), MLP probe | 72.9 | 0.619 | 74.3 | 32.9 |
| E6-11-44 **untrained initialisation**, MLP probe | 73.1 | 0.614 | 75.7 | 34.0 |
| E6-11-44@200M's own online aux head | 68.9 | 0.710 | 70.8 | 31.3 |

Linear probes read within 0.6 points of the MLP ones.

## 3. The wider trunk loses it with training

The same probe on P29 bet 1's 768 × 8 trunk (E6-10-44), with M2d references, on self-play of
E6-10-44@560M and E6-03-44@560M (`data/reports/p29_trunk_probe_wide.{md,json}`), MLP probe:

| trunk | untrained | @200M | @560M |
|:---|---:|---:|---:|
| M2d, E6-03-44 | 76.4% (surprises 35.1) | 76.1 (34.3) | 76.2 (34.6) |
| wide, E6-10-44 | 76.7 (35.2) | 74.0 (24.4) | **73.3 (23.8)** |

E6-07-44@700M (the best raw model) reads 76.1 (33.4); the raw observation 79.5 (45.0).

* **The wide trunk's hidden vector grows about 38× over training.** Its mean length goes from 49
  at initialisation to 625 at 200M and 1,842 at 560M. M2d's stays at 25–29. The residual
  stream has no normalisation, so nothing bounds it.
* An unstandardised probe fails on the wide trunk (a linear log-loss of 1.19), which is why the
  inputs are now standardised. Even standardised, the wide trunk ends *below* its own
  initialisation, at the constant's level on surprises.

## Reading

* **The aux target has changed nothing measurable in the trunk.** The aux-trained trunk, the
  control's trunk and a randomly initialised one support the same ownership probe
  (72.8 / 72.9 / 73.1%).
* **The information is in the observation, not in the trunk.** A probe on the raw observation
  reaches 76.8–79.5% and 85–87% late in the game, and every trunk loses about 10 points of that
  late. No trained trunk, M2d or wide, at any step, beats its own random initialisation; the wide
  one falls below it.
* **The trained M2d trunk keeps no more of the board than a random projection does.** That is
  consistent with the architecture: the per-country policy decisions come from the per-entity
  heads, which read each country's raw row directly, so nothing asks the pooled vector to carry
  the board.
* **The online head underfits even its own trunk.** It is 3.9 points below a fresh probe on the
  same frozen vector, because one step per ~4k finished positions at weight 0.1 is too little.
* **What this means for bet 2 as run.** A 0.1-weighted target read from the pooled vector is too
  weak to shape the representation. The run's critic and strength readings at 560M will test the
  heads' side effects, not ownership-shaped features.
