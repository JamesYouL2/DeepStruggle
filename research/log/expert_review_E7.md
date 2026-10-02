# An expert review of the E7 soup: strengths, leaks, and why training may plateau (2026-10-02)

**Who and why.** James You (fork `JamesYouL2/DeepStruggle`) is a top-20-to-50 Twilight Struggle
player. He played the current best model and judges it at roughly 1,500-1,800 on a human scale:
strong, with a few recurring wrong moves. This log turns his observations into numbers. The
census measures how often the model departs from a strong player's rule. Paired playouts by the
model itself measure what each departure costs.

**Model.** `shallow_E7-02+03+04+05_1200M.onnx`, the `newest` export, on both sides. Engine
fingerprint `5419a582ec07…`.

**Method.**

* Positions come from the model's own greedy self-play, or from the human ts-replayer corpus.
* Each alternative is forced, and the model plays on. For the rest of the action round, a move
  that loses on the spot is never taken while another exists. This guards against untrained
  follow-ups, such as a Star Wars retrieval of a DEFCON card at DEFCON 2.
* Each pair redeals the cards the mover cannot see and uses the same dice in every branch, so
  branches differ only in the decision. ± is one standard error.
* Everything ran on the fork's CI. The run ids are in the last section. The tools are on the fork
  branch [`exp/hungary-openings`](https://github.com/JamesYouL2/DeepStruggle/tree/exp/hungary-openings),
  under `ai/eval/`.

**The standing caveat.** A playout verdict values a move given how *this* model plays afterwards.
So a move whose payoff lies in a follow-up the model does not know reads worse than it is.

## Strengths: placement and board judgment hold up

* **Ops placement.** Playout audit, 24,000 sampled decisions: every legal play mode, and the
  likeliest cards, are each played out. The model's choice is the playout-best on average. Mean
  split-half regret is −1.4 for play mode and −0.1 for card choice, so there is no broad leak.
* **Scoring battlegrounds.** In 48,217 influence plays, the model reads scoring-card locations. It
  takes a takeable battleground 47% of the time when it holds the region's scoring card, 28% when
  the card is live elsewhere, and 11% when it is discarded. Where it skips a live battleground,
  taking it scores −0.5 ± 0.1, so it is right. Its points went 82% to other battlegrounds.
* **Replies.** Determinized search-64 plays the same action round as the net in 91% of 24,000
  rounds. Where they differ, search gains +0.1 ± 0.2, and exposure to the opponent's reply is
  unchanged. A missing reply model is not what holds it back.
* **Against strong humans, on turn 1.** 7,139 turn-1 disagreements with the corpus were played out
  each way. Human − model is −0.0 ± 0.1. The model's own headline and coup choices score better
  than the humans' (−1.7 ± 0.7 and −1.2 ± 0.6). The humans' choices inside events score better
  (+0.6 ± 0.3).
* **The critic is calibrated.** Over 1.9M readings in 4,001 games, P(US wins) from the side to move
  matches the result: fitted temperature 1.02, binned squared error 0.0006. The same holds in each
  era and for each side to move. See §4 for what is not fine.

## The leaks, by cost per game

| leak | how often | cost when it happens | per game |
|:---|:---|:---|---:|
| **Wargames declined at a winning lead** | 429 of 430 times; ~11% of games | 14.3 ± 2.6 pts (it still wins 85.7%) | **~1.5 pts** |
| **OPEC not evented at 5+ VP** | evented 24% | +2.4 ± 0.3 over all such plays | — |
| **Star Wars not evented when ahead in space** | evented 0% | +1.6 ± 0.6 (+2.0 ± 0.7 where it declines) | — |
| **Alliance for Progress not evented at 5+ VP** | evented 22% | +1.1 ± 0.3 | — |
| **Five Year Plan timing** (USSR) | see below | Ops-first over event-first +2.1 ± 0.9 | — |

**Checked and not confirmed: the South Korea setup bonus.** The model puts one or both US bonus
points in South Korea in 23% of games. James reads that as clearly wrong, and on 32 human positions
the human's Italy beat the model's South Korea by +8.1 ± 1.8. But a 4,096-deal bake-off does not
confirm it. It pinned the model's usual WG 3 / France 3 / Italy 2 / Iran 2 against its own setup,
on the same deals (fork runs `36970757999` and `36970763934`):

* **Overall.** +0.0 ± 0.5.
* **On the 927 deals where the model's own setup went outside Europe.** +0.2 ± 2.1.

At E7 strength the placement costs nothing measurable, perhaps because neither side exploits it. WG 4 /
France 2 / Italy 2 / Iran 2, which led an E6 bake-off by +2.1, is −0.6 ± 1.0 here (`36970769645`).

**Wargames.** At DEFCON 2 with a lead over 6, the event wins on the spot. The model plays the card
for Ops instead in 429 of 430 cases (doctrine census, 4,000 games). From those positions it goes on
to win 85.7% (60 positions × 32 pairs). The case for treating this one apart from the general
forced-win question
([`agent_deficiencies_and_decisiveness.md`](agent_deficiencies_and_decisiveness.md): "declining a
forced win is usually free") is that `ai/eval/safety.py` classifies Wargames only at its
`CHOOSE_BRANCH` node. At the play-mode decision the event is "normal", so the decisive probe has
never counted these misses, and the `safe:` wrapper does not take them.

**Event timing and hand management: the action round and the hand.** A strong player holds Five
Year Plan (USSR) and Aldrich Ames Remix (US) for the last action round, with exactly one other card
in hand, so the discard can only take that card. Ideally that card is a bad scoring card.

* **Five Year Plan.** Played in the last round 55% of the time, and with two or more other cards in
  hand 38% of the time.
* **Aldrich Ames.** Played in the last round 79% of the time, with exactly one other card 55% of
  the time.
* **Where the misses fall.** They sit one round early, exactly where the round count changes. In
  turns 1-3, Five Year Plan clusters on round 6, which is right. From turn 4 on, 788 plays fall in
  round 6 against 1,316 in round 7.
* **What the net is not told.** It sees the turn and action round raw, and both hand sizes. It is
  never told rounds left, or whether the opponent replies before the turn ends.

**Inside events.** The model makes its worst choices in spots it rarely reaches:

* **Aldrich Ames' discard.** With a normal US hand, eventing with the playout-best discard is level
  with Ops (−0.8). With the model's own pick it is −5.5. It discards cards the US would have been
  forced to event for the USSR (40 positions).
* **De-Stalinization's removal.** It takes from Syria where humans take from Finland or Romania
  (+3.3 / +4.8).
* **Marshall Plan.** It goes into Sweden where humans choose Canada or West Germany (+2 to +3).
* **The round-8 pass.** Passing the eighth action round is another rarely reached choice.

## The VP ledger: humans get VP from events, the model from the board

Every VP change is credited to the step that made it. That is 4,001 self-play games against 175
complete corpus games, in VP per game:

| source | model US | human US | model USSR | human USSR |
|:---|---:|---:|---:|---:|
| events | 7.8 | **11.1** | 7.3 | **8.9** |
| scoring cards | **15.8** | 14.8 | **17.2** | 16.1 |
| space race | 2.0 | 2.3 | 1.2 | **1.8** |
| Military Ops shortfall gained | 0.7 | 1.2 | 3.5 | 3.8 |

The largest event gaps, human over model, in VP per game:

| event | gap |
|:---|---:|
| OPEC | +0.95 |
| Arms Race | +0.94 |
| Wargames | +0.65 |
| Duck and Cover | +0.50 |
| Alliance for Progress | +0.43 |
| How I Learned to Stop Worrying | +0.34 |
| Special Relationship | +0.32 |

These are the direct-VP events. The populations differ, since humans play humans, so read this as
style rather than a controlled comparison.

**Unresolved.** Forcing the events humans favour, where the model declines, mostly scores
*negative* by the model's own playouts. That holds for John Paul II −1.5, Bear Trap −2.3 and
Missile Envy −2.5 to −3.5. These are exactly the delayed-payoff events, where the standing caveat
bites. The ledger says humans turn events into VP. The playouts can only see the immediate cases.

## The critic: calibrated, but the two seats disagree

Each seat's reading is right on average, but the readings disagree position to position. P(US
wins) moves **1.07 points per decision when the side to move stays the same, and 4.40 points when
it changes** (rms 1.95 against 6.43, 1.9M steps). That is about 4 points of noise between the two
views. It is what makes the workbench curve look swingy. It is also noise that search adds up
across nodes of both seats, which bears on C5's gate in
[`P30_base_model_quality.md`](../plans/P30_base_model_quality.md): "a critic that search can
use". A consistency target, under which the two seats' readings sum to one, would cost no
observation change.

## Why training may plateau where the playouts can still tell

The playouts above see 2-15 point differences at specific decisions. RL credits one sampled move
per decision, from one game's ±1, against that critic noise. The playouts compare all moves under
the same deal and dice. Rare moves, such as Wargames at p ≈ 0.002 or an event the model never plays
in an action round, are almost never sampled, so they never get a gradient. This reads as a
signal-to-noise floor, which fits saturation at about 1B steps.

**A candidate the record has not tried:** policy iteration on paired-playout labels.

* **Label.** Sample decisions, oversampling the weak kinds above. Play out each candidate with the
  current net, using common random numbers.
* **Fine-tune.** Fine-tune the policy toward π′ ∝ π·exp(Q̂/τ), the operator from
  [P3](../archive/E3_ladder/plans/P3_determinized_search_expert_iteration.md), with a KL bound to
  the previous net.
* **Gate.** Gate each round on head-to-head play and the census rates, then relabel with the new
  net.

How it relates to what the record has tried:

* **Search distillation.** It moved this lineage by +48 to +166 Elo
  ([`E4_search_distillation.md`](E4_search_distillation.md) and the X4 logs). The difference here is
  that no critic sits at the leaves, so the headroom does not shrink with the student's critic.
* **The standing caveat.** That a verdict holds only "given how this policy plays afterwards" is
  answered by relabelling each round. Each round's net plays the follow-ups the previous round
  taught.

James would like to run the first round himself. The net is 1.27M parameters, so a CPU is enough
and the labels come from free CI. That needs the trainable weights.

## Reproduction

All runs are on the fork's CI, model and engine as above, and every report carries a provenance
block.

* **Doctrine census.**
  * 4,000 games: `36943954436`.
  * Hand size at discard events: `36953663726`.
* **Choice oracle.**
  * Event and headline questions, 512 positions × 128 pairs: `36941086943`, `36945382157`.
  * The cards humans event most, forced where the model declines: `36964635572`.
* **Playout audit.** 24,000 decisions: `36953614691`; parts redone after a fix in `36955589646` and
  `36958736889`.
* **Reply probe.** Net against search-64: `36951383335`.
* **Live scoring battlegrounds.** `36956833709`.
* **Human disagreements.**
  * All turns: `36958653207`.
  * Turn 1: `36963933063`.
* **VP ledger and calibration.** `36966918630`.
* **US setup bake-off.** Own setup, and the two pinned setups: `36970757999`, `36970763934`,
  `36970769645`.
* **Local measurements**, each with the same paired method: the Wargames cost (60 positions × 32
  pairs), the Aldrich Ames discard (40 positions × 32 pairs), and the US setup distribution (256
  games).
