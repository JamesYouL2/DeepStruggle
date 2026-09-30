# P30 (2026-09-30): card <-> country interaction is learned by the policy, at about the right size

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

## 4. Is the size right? The paired-rollout oracle at both margins

`tools/scripts/scoring_region_oracle.py`, E6-12-44@560M: starts of influence plays from its own
self-play (20,000). For each region, 100 positions where the mover holds that region's scoring
card and 100 where it does not. Two branches per position, each played to the end 16 times by the
net on both sides, greedy, dice paired. Mover's win-rate change in percentage points, ± SE
(`data/reports/p30_scoring_region_oracle{,_out}.{md,json}`).

| region | **in**: policy placed nothing there, force one point in (holds / not) | **out**: policy placed there, force every point elsewhere (holds / not) | interaction, out (holds − not) |
|:---|---:|---:|---:|
| Europe | +0.3 / +1.6 | −1.7 / +0.3 | −2.0 ± 2.2 |
| Asia | −4.1 / −0.0 | −1.9 / −1.6 | −0.3 ± 2.0 |
| Middle East | −1.0 / −4.4 | −0.7 / −1.2 | +0.5 ± 2.0 |
| Africa | −3.3 / −3.8 | −3.0 / +1.0 | −4.0 ± 2.0 |
| Central America | +1.5 / −1.9 | −2.5 / +0.5 | −3.0 ± 1.7 |
| South America | −2.8 / −2.9 | −2.0 / +0.0 | −2.1 ± 1.7 |
| Southeast Asia | −2.3 / −0.5 | −0.6 / −2.3 | +1.7 ± 2.2 |
| **all** | **−1.7 ± 0.6 / −1.7 ± 0.7** (interaction +0.0 ± 0.9) | **−1.8 ± 0.6 / −0.5 ± 0.5** | **−1.3 ± 0.8** |

* **No under-placement.** Where the net skips a region, forcing a point in costs 1.7 points whether
  or not it holds the scoring card.
* **No over-placement.** Where the net places in a region, taking those points away costs 1.8
  points when it holds the card, and 0.5 when it does not.
* **The interaction is real but small.** Holding the card makes the net's own points in that
  region worth about 1.3 points more (1.6 SE). It is concentrated in the mid-war regions (Africa
  −4.0, Central America −3.0, South America −2.1), which are the regions where the net's
  counterfactual shift is largest (section 1).
* **Limits.** Both sides are played by the same net, so this says the choice is right *within
  its own play*. The interventions are one-point moves. The positions are selected on the net's
  own choice.

## Reading

* **The interaction is learned, not absent.** The policy conditions placement on held scoring
  cards in every trained E6 net, most of all in the shallow one, through the trunk's context
  vector. There is no need for an explicit mechanism to get it.
* **The critic is not visibly missing it**, with the observational caveat above.
* **The learned amount looks right at the one-point margin.** Neither adding a point where the
  net skips a region nor removing its points where it places them improves on the net, with or
  without the card. The true interaction is modest (about 1–4 points per decision, mid-war regions
  first), and the net has learned it about as far as its own play can tell. There is no sign
  here that an explicit card↔country mechanism would find missing strength.
