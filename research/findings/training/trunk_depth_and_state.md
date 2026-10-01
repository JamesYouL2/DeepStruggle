# The trunk: depth buys nothing, and what it fails to hold is a learning gap, not a capacity one

Measured 2026-09-30 – 10-01 on E6/E7 (training findings: expected to survive the engine change).

## 1. M2d's residual blocks add no strength

The shallow trunk is `--ladder-res-blocks 0`: observation → grouped projections → fusion → heads,
4 linear layers against M2d's 12, 1.34M parameters against 3.19M, about 16% faster per step.

| seed | 500–560M snapshots, US / USSR against the panel | head to head | 480–560M SWA, head to head |
|:---|:---|---:|---:|
| 44 (E6-12-44 against E6-03-44) | +6.6 / −1.2 | 56.6% | 53.8% |
| 43 (E6-12-43 against E6-03-43) | +2.7 / +9.6 | 59.7% | 56.2% |

* Behind at 80M, ahead from about 160M on both seeds; the same-field traces agree.
* **Averaging gains the deep net more.** Deep SWAs sit 70–100 Elo above their snapshots, shallow
  ones 50–60. So at 560M, with the league, the deep SWA leads (E6-04-44 against E6-13-44: 45.6%
  for the shallow one).
* Log: [`../../log/P30_shallow_trunk.md`](../../log/P30_shallow_trunk.md).

## 2. The trunk holds little of the state, and the loss is at the input projections

Read back from frozen trunks (`tools/scripts/trunk_state_probe.py`, MLP probes on held-out games;
the raw observation reads 99%+ on every target):

| | country control | card location | "in my hand" recall | ownership at game end |
|:---|---:|---:|---:|---:|
| M2d, trained | 90% | 80% | 34% | 76% |
| shallow, trained | 88% | 80% | 33% | — |
| untrained trunk | 92% | 85% | 44% | 76% |
| wide 768 × 8, trained | 84% | 73% | 12% | 73% |

* **Training does not add state to the trunk; it sheds some** while adding outcome information
  (value AUC 0.815 trained against 0.734 untrained).
* **Depth is not where it is lost.** The 0-block trunk holds the same state as the 4-block one, so
  the loss is in the input projections: the whole 84 × 26 board into 256 numbers through one
  linear layer, and likewise the cards.
* **The critic is not short of the dropped state.** A fresh value head on the frozen trunk matches
  the critic (0.808 / 0.808), and adding the raw observation adds nothing (0.803) at 457k
  positions.
* **The wide trunk has an unbounded residual stream.** Its hidden vector grows about 38× over
  training, and it holds the least state of any trunk.
* Logs: [`../../log/P29_bet2_ownership_probes.md`](../../log/P29_bet2_ownership_probes.md),
  [`../../log/P30_stage_A_trunk_probes.md`](../../log/P30_stage_A_trunk_probes.md).

## 3. Card × board interactions: partly representable, better with attention, not learned by RL

Engine-labelled targets per held card: what its event would do on this board, and what its Ops
could reach. Within-card R² counts only the board-dependent part
(`tools/scripts/card_board_targets.py`, `card_board_probe.py`):

| | event outcome, within-card R² |
|:---|---:|
| grouped trunk, trained *on these targets* (114k positions) | 0.80 |
| card and country tokens with 2 attention layers, the same (7× fewer parameters, still improving) | **0.90** |
| the RL-trained trunks, frozen (E6-12-44, E6-03-44, E6-07-44) | 0.66 |
| an untrained trunk, frozen | 0.665 |

* **Representable, partly.** The grouped trunk reaches 0.80 when trained for it. Explicit
  attention reaches 0.90, with the gap largest on sums over a set of countries (regional margins,
  battleground counts).
* **Not learned by RL.** The trained trunks hold no more of it than an untrained one. The policy's
  card choice follows the board-dependent event value only weakly (within-card correlation
  +0.12 to +0.17, against 0 untrained).
* **One simple interaction is learned, and at about the right size.** A held region scoring card
  moves 5–24 points of placement into its region. Paired rollouts find no under- or
  over-placement at the one-point margin; the true interaction is small (about 1–4 points of win
  rate).
* **An auxiliary target at weight 0.1 from the pooled vector does not reach the trunk.** The bet-2
  ownership head ended as the per-country prior.
* Logs: [`../../log/P30_card_country_interaction.md`](../../log/P30_card_country_interaction.md),
  [`../../log/P30_card_board_targets.md`](../../log/P30_card_board_targets.md).

## What this points at for base-model quality

* **Not depth or width after the fusion.** That is settled twice: M2d against shallow, and the
  768 × 8 bet 1.
* **The input representation.** That is where state is lost, and where per-entity structure
  already proved its worth (the country head, +282).
* **The learning signal.** The architecture can hold more than RL puts into it.

[P30](../../plans/P30_base_model_quality.md) turns these into arms.
