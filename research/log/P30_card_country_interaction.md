# P30 (2026-09-30): card <-> country interaction is learned by the policy; the critic shows no missed one

The owner's question: M2d and the shallow trunk have no explicit card <-> country mechanism. Is the
interaction not needed, or needed and just not learned? Tool:
`tools/scripts/card_country_interaction_probe.py`. Positions: self-play of E6-12-44@560M and
E6-03-44@560M at temperature 1.0, 1,500 games each; 172k positions, 73k of them influence
placements (`data/reports/p30_interaction_probe.{md,json}`).

## 1. Used: a region's scoring card in hand moves the placements there

A region's scoring card is swapped into the mover's hand in the observation (in for a random
non-scoring hand card, so the hand size is unchanged). The table shows the policy's shift of mass
onto that region's countries, net of a placebo swap with a random non-scoring card, in
percentage points, ± SE; 13k–35k counterfactuals per region.

| region | E6-12-44@560M (0 blocks) | E6-03-44@560M (M2d) | E6-07-44@700M (best raw) | E4-03-01 (ColdWarNetV2, card↔country attention) | untrained |
|:---|---:|---:|---:|---:|---:|
| Europe | +9.2 | +11.1 | +7.4 | +1.4 | −0.1 |
| Asia | +14.3 | +11.4 | +11.0 | +3.7 | −0.1 |
| Middle East | +13.2 | +12.8 | +6.2 | +3.2 | +0.1 |
| Africa | **+24.0** | +16.0 | +13.8 | +0.3 | −0.0 |
| Central America | +15.6 | +12.6 | +5.3 | +0.0 | −0.1 |
| South America | +16.5 | +10.8 | +6.7 | −0.1 | +0.0 |
| Southeast Asia | +5.8 | +4.2 | +3.2 | +0.6 | +0.1 |

(SE 0.1–0.2 throughout.)

* **Every E6 net conditions placement on the hand, strongly.** Holding a region's scoring card
  moves 5–24 points of placement mass into that region, against a base of 10–40%. The untrained
  net moves nothing, so this is learned.
* **The shallow trunk does it most.** The interaction goes through the trunk context the country
  head reads. The trunk's poor overall hand recall (a third, stage A) is therefore selective: it
  keeps the cards that matter for placement.
* **The old net with explicit card↔country attention does it least** (0–4 points). It is on the
  E4 engine and recipe, so this does not condemn attention, but explicit attention was not what
  produced the behaviour.
* Holding a scoring card also lowers the value in every trained net (−0.04 to −0.07 in `v_win`),
  a main effect: a 0-ops card that must be played.
* Whether the *size* of the shift is right is not measured here. That needs the paired-rollout
  oracle (next).

## 2. Needed by the critic? No sign of a missed interaction

Logistic fits of the winner on held-out games (30%, bootstrap over games). The features: for each
of the six regions, "my hand holds its scoring card" and "the opponent is known to hold it", each
times the observation's live regional scoring margin.

| | held-out log-loss | gain from the interaction terms [95%] |
|:---|---:|---:|
| additive features only, no critic | 0.6269 → 0.6265 | +0.0003 [−0.0001, +0.0008] |
| E6-12-44@560M critic | 0.5473 → 0.5471 | +0.0000 [−0.0003, +0.0003] |
| E6-03-44@560M critic | 0.5626 → 0.5603 | −0.0000 [−0.0003, +0.0002] |
| E6-07-44@700M critic | 0.5447 → 0.5450 | +0.0000 [−0.0003, +0.0003] |

(The first log-loss column moves from the critic alone to the critic plus additive and interaction
terms.)

* The interaction terms predict nothing beyond the critic, and nothing beyond additive features
  even without a critic.
* **Limit:** this is observational. The policies already act on the interaction (they place towards
  held scoring cards, and time the scoring), so its effect on the result is largely absorbed before
  the outcome is recorded. It shows the critic is not *missing* this interaction on the policy's
  own games. It does not show the game would not reward a better one.

## 3. How much the value mixes board and cards

RMS entry of the value's Hessian blocks (Hutchinson, 512 positions × 8 probes); cross / geometric
mean of within-block:

| net | ratio |
|:---|---:|
| E6-12-44@560M (0 blocks) | **0.32** |
| E6-03-44@560M (M2d) | 0.69 |
| E6-07-44@700M | 0.61 |
| E4-03-01 (attention) | 0.61 |
| untrained | 0.67 |

The strongest model's value is the most additive in board and cards. Value-side interaction does
not track strength here.

## Reading

* **The interaction is learned, not absent.** The policy conditions placement on held scoring
  cards in every trained E6 net, most of all in the shallow one, through the trunk's context
  vector. There is no need for an explicit mechanism to get it.
* **The critic is not visibly missing it**, with the observational caveat above.
* **Whether the learned amount is right** is the open question, and only an intervention
  answers it: the P27 paired-rollout oracle with placements forced into the scoring card's
  region, with and without the card in hand.
