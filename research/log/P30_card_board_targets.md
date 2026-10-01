# P30 (2026-09-30): card × board targets -- representable in the grouped trunk, better with attention, not learned by RL

The owner's question: we seem not to need complex card <-> country interaction. Is it (1) not
needed, (2) not representable in the current architecture, or (3) representable but not learned?
This log answers (1) partly and (2)–(3) directly, on engine-labelled targets that depend on a card
*and* the board.

**Targets** (`tools/scripts/card_board_targets.py`). Per card in the mover's hand:

* **T2, ops arithmetic.** With the card's Ops: countries and battlegrounds the mover could bring
  under control, and the best coup chance, overall and among battlegrounds.
* **T3, the event can fire** (`can_trigger_event`). It is true 98.1% of the time, so it carries
  little signal.
* **T4, what the event does.** Fired on a clone for the player it fires for: the change in VP,
  DEFCON, the six regional scoring margins, battlegrounds and influence of each side, from the
  mover's side. Only events that resolve without a further choice; 90 cards.
* **T1** is T4's VP change for the scoring cards.

Dataset: self-play of E6-12-44@560M and E6-03-44@560M, 1,500 games each, 156k positions, 425k
held-card T4 labels (`data/datasets/card_board/e6_12_03_560M.npz`). Scores are on held-out games
(30k positions). **"Within"** is R² after removing each card's mean from target and prediction:
only the part that depends on the board for a fixed card, i.e. the interaction.

Probe: `tools/scripts/card_board_probe.py`; report `data/reports/p30_card_board_probe.{md,json}`.

## 1. Representable? Trained from scratch, supervised

T4 within-card R²:

| architecture | params | 10k positions | 35k | 114k |
|:---|---:|---:|---:|---:|
| shallow (E6-12-44's trunk) | 2.5M | 0.617 | 0.717 | 0.800 |
| M2d (E6-03-44's trunk) | 4.4M | 0.581 | 0.708 | 0.759 |
| **attention** (country and card tokens, 2 layers) | **0.34M** | 0.648 | 0.795 | **0.903** (at the 40-epoch cap, still improving) |
| MLP on the raw observation | 6.5M | 0.607 | 0.693 | 0.737 |

At 114k positions, within-card R² per quantity:

| quantity | shallow | M2d | attention |
|:---|---:|---:|---:|
| regional margin, South America / Europe / Africa | 0.53 / 0.68 / 0.65 | 0.46 / 0.60 / 0.62 | **0.78 / 0.86 / 0.79** |
| battlegrounds controlled, mine / theirs | 0.80 / 0.79 | 0.77 / 0.76 | **0.98 / 0.98** |
| VP, DEFCON, total influence | 0.87–0.95 | 0.85–0.93 | 0.89–0.98 |
| T2 (ops arithmetic, mean) | 0.96 | 0.95 | 0.99 |

* **The grouped trunk can represent these interactions, partly.** Supervised, it reaches 0.80
  within-card on event outcomes and 0.96 on ops arithmetic.
* **Explicit attention represents them clearly better,** with 7× fewer parameters: 0.90, and still
  climbing when the fit stopped. The gap is largest where the target is a sum over a specific
  set of countries: a region's margin, a battleground count. That is the shape of structure
  per-country tokens capture and a flat projection must spread over one layer.
* The residual blocks do not help (M2d is below shallow), and neither does raw width (the MLP is
  lowest).

## 2. Learned? RL-trained trunks against an untrained one

The same head, fitted on the frozen hidden vector, trained on all 114k positions:

| trunk | T1 within | T2 within | **T4 within** |
|:---|---:|---:|---:|
| E6-12-44@560M | 0.857 | 0.798 | 0.664 |
| E6-03-44@560M | 0.835 | 0.782 | 0.660 |
| E6-07-44@700M (best raw) | 0.848 | 0.782 | 0.662 |
| **untrained** (E6-12-44 at step 0) | 0.620 | 0.797 | **0.665** |
| *same architecture trained on the targets (§1)* | *0.886* | *0.962* | *0.800* |

* **RL has not put event outcomes or ops arithmetic into the trunk.** Every trained trunk reads
  exactly like an untrained one on T2 and T4 (0.66 against 0.665). Only the scoring-card VP (T1)
  improved with training.
* **The same architecture holds far more when trained for it** (0.80 and 0.96). So the gap is
  learning, not capacity: the RL-trained trunk is a random projection as far as these targets go.

**Behaviour at card selections.** This is the within-card correlation of a card's relative logit
with its event outcome, both demeaned per card: does the policy prefer a card more when its
event is better on *this* board?

| net | VP | regional margins | battleground balance | influence balance |
|:---|---:|---:|---:|---:|
| E6-12-44@560M | +0.17 | +0.13 | +0.15 | +0.17 |
| E6-03-44@560M | +0.12 | +0.09 | +0.11 | +0.13 |
| E6-07-44@700M | +0.12 | +0.11 | +0.12 | +0.15 |
| untrained | −0.01 | +0.03 | +0.01 | +0.05 |

The policies do respond to the board-dependent value of a card's event, weakly (+0.1 to +0.17).
The shallow net responds most.

## Reading

* **(2) Representable:** in the current architecture, partly. With explicit card↔country
  attention, clearly better and more cheaply. The architecture is a real but not total limit.
* **(3) Learned:** no. What the architecture *could* hold, RL has not put there. The trained
  trunks carry no more event-outcome information than an untrained one, and the policy's use of
  it is weak.
* **(1) Needed:** still open. These targets are what cards *do*, not what they are worth. Whether
  learning them would win games needs an intervention:
  * an auxiliary event-outcome target during RL, which is cheap, and unlike ownership its signal
    is dense and per card;
  * and/or an attention-token arm (P30 stage C).

  Either should be judged by the owner's rule, with these probes as the check that the
  information actually arrived.
* **Limits.**
  * T4 covers only events that resolve without a choice (T5, with choices, is not built).
  * T3 is near-constant.
  * The attention fit stopped at the epoch cap.
  * One dataset from two policies, one seed.

## Addendum (2026-10-01): which token path is affordable (C1 sizing)

The 0.90 came from 2 full self-attention layers at d=128 over 195 tokens. In training that
costs far more than the trunk it would sit beside, so cheaper variants were fitted the same way
(114k positions, 40-epoch cap, `tools/scripts/card_board_probe.py --archs ...`). Cost is one
fwd+bwd+Adam step at batch 4096 in bf16, measured while two training arms shared the GPU, so only
the ratios mean anything:

| token path | params | T4 within-card R² | cost per minibatch (compiled / eager) |
|:---|---:|---:|---:|
| shallow trunk, for reference | 2.5M | 0.800 | ~10 ms |
| full, d=128, 2 layers | 0.34M | **0.903** | 94 / 155 ms |
| full, d=128, 1 layer | — | 0.776 | — / 66 ms |
| full, d=64, 2 layers | 0.10M | 0.716 | 58 / 89 ms |
| full, d=64, 1 layer | — | 0.704 | — / 38 ms |
| full, d=32, 2 layers | 0.03M | 0.612 | 38 / 57 ms |
| full, d=32, 1 layer | — | 0.618 | — / 24 ms |
| cards query countries (one cross block), d=128 | — | 0.757 | — / 30 ms |
| cards query countries, d=64 | — | 0.680 | — / 17 ms |

* **Only the expensive variant beats the trunk.** Both width (d=128) and a second layer are
  needed; every cheaper variant falls below the shallow trunk's 0.80. All fits hit the epoch cap,
  so the small ones may be under-trained, but none was close.
* **Cost.** With 64 minibatches per iteration (4 epochs × 16), d=128 × 2 layers would make an
  iteration roughly 4–5× longer: about 12–25 h to 1,200M solo, against ~3 h for the shallow
  recipe.
