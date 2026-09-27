# E5-11-43@560M self-play review: three mistakes checked by playouts, a hand probe, and a realignment census

The owner read four greedy self-play games of E5-11-43@560M (`data/replays/e5_11_43_560M_selfplay_seed{1..4}.tslog.json`,
2026-09-27) and named three US decisions as mistakes. The owner asked two questions: are they
evidence for an architecture that plans with cards (cross-card signals such as "I hold no scoring
card, so The Cambridge Five is safe"), and does the model under-use realignment, perhaps having
learned "never realign" early and never revisited it?

## The three decisions, by counterfactual playouts

Method:

* The replay is re-driven from its seed to the decision; VP and DEFCON match at every step.
* Each alternative is forced: the card, the mode, and, for realignment, every roll that action
  round on the named country.
* Each branch is played to the end by the checkpoint on both sides, greedily, 256–400 times. Pair k
  uses the same dice in every branch.
* The script is `branch_oracle.py`, in the session's scratch space.

| decision | played | alternative | mover's win rate |
|:---|:---|:---|:---|
| seed 4, step 208 (US, T4 AR2). The USSR has just couped Mexico to 5, and holds Central America Scoring | Quagmire to space | Cuban Missile Crisis, realign Mexico | 2.0% → 9.4%, **+7.4 ± 2.1** paired |
| | | Quagmire, realign Mexico | → 8.8%, +6.8 ± 1.9 |
| seed 3, step 120 (US, T2 AR5, hand Nasser + The Cambridge Five, no scoring card) | Nasser | The Cambridge Five first | 32.9% → 41.1%, **+8.2 ± 3.1** |
| seed 4, step 243 (US, T5 AR1, VP −19, the USSR holds Middle East Scoring) | UN Intervention on Brezhnev Doctrine | any of four other cards | 0% in every branch; lost regardless |

At step 208 the policy gave realignment 0.5%. The realignment breaks Mexico in 43% of branches
before the scoring card lands. At step 120 both cards had to be played that turn. Playing The
Cambridge Five later, the policy put 2 of its 3 Ops into Canada, which the US already controlled.

## Does the policy know The Cambridge Five is harmless without a scoring card?

Positions are US card choices on turns 1–3 of the checkpoint's own sampled self-play, with no
scoring card in hand. In each, the US is given both The Cambridge Five and a control, Vietnam
Revolts: a 2-Ops USSR event whose effect does not depend on the US hand. The policy is read twice,
before and after one non-scoring card is swapped for a scoring card (`ai/eval/card_probe.give_card`).

| checkpoint | positions | P(The Cambridge Five), no scoring card | with a scoring card | log-odds vs control, change |
|:---|---:|---:|---:|---:|
| E5-11-43@560M | 576 | 0.455 | 0.063 | −7.1 ± 0.1 |
| E5-11-44@640M | 568 | 0.277 | 0.018 | −6.4 ± 0.1 |

**Yes.** The policy dumps The Cambridge Five while it is safe and holds it when a scoring card
would be revealed. The cross-card conjunction is already computed. The seed-3 error is one of
sequencing against the opponent's next move, not a missing representation.

## How often does it realign?

`tools/scripts/play_mode_census.py`, 1,000 greedy self-play games per checkpoint, against the human
corpus. Figures are the realignment share of Ops plays at nodes where realignment is legal. Reports
are in `data/reports/play_mode_census/`.

| source | all | DEFCON 2 | DEFCON 3+ | P(realign) > 5% at legal nodes |
|:---|---:|---:|---:|---:|
| human corpus (254 games) | 4.4% | 5.6% | 1.6% | — |
| E5-11-43 @ 40 / 80 / 160 / 320 / 560 / 800M | 1.6 / 0.8 / 1.6 / 1.4 / 3.4 / 1.4% | 1.0–5.0% | 0.00–0.08% | 5–10% |
| E5-11-44 @ 40 / 160 / 320 / 640M | 3.3 / 0.7 / 2.0 / 1.1% | 1.0–4.2% | 0.02–0.17% | 4–16% |
| E5-11-43@560M sampled at temperature 1.0 | 3.8% | 5.5% | 0.3% | 7.7% |
| E4-61-43 @ 160 / 560M (old sharpened bands) | 13.3 / 11.6% | 19.3 / 15.6% | 1.0 / 0.8% | 54–61% |
| E5-12 @ 280M | 0.9–2.2% | 1.5–3.4% | 0.00–0.07% | 5–6% |

* **There is no "never realign" collapse.** Nothing trends over training. At every snapshot the
  policy still puts more than 5% on realignment at 4–16% of the positions where it is legal, so at
  training temperature it keeps sampling it.
* **At DEFCON 2 it is near the human rate. At DEFCON 3+ it realigns about 30× less than humans.**
* The old sharpened bands realigned at three times the human rate. Flat 1.0 moved the policy from
  over-realigning to under-realigning.

## Reading

Two of the three named mistakes are real, at +7 to +8 points each. Neither points to a missing
representation. The Cambridge Five conjunction is computed, and realignment is still sampled. What
fails is judging *when* a known move is right in a particular position. Here a one-move search with
playouts found the better move in both genuine cases. Two next steps are cheaper than an
architecture change:

1. **A realignment oracle** in the style of P27, over many positions where realignment is legal and
   the policy gives it little weight. It would show whether under-realignment is systematic,
   especially at DEFCON 3+.
2. **Rating a playout search over the policy's top card-and-mode choices** against the raw policy.
   A large gain would argue for training on search targets.

Card-to-card attention (P21 M3, blocked) stays available.
