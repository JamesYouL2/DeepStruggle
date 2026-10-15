# P32 — A teacher that does not flatten the policy, and paired credit at the turn's end

**Status: closed (2026-10-15), superseded by [P33](P33_search_preserved_gains.md).** Branches from
the heads line's plateau (`E7-A8-R1-S44@6400M`), each arm registered with its full result in
[`../runs.md`](../runs.md) (R28–R48, A9–A13).

## The question (2026-10-09)

Search on top of the heads line still paid ~+65 Elo (Gumbel k=8 @256,
[`../log/E7_gumbel_headroom.md`](../log/E7_gumbel_headroom.md)), but search as a teacher had failed
twice the same way (R18, E7-18-44): **entropy 0.30 → ~0.5 within 10M steps**, because PUCT visit
counts at 64 simulations are a broad target and cross-entropy toward them flattens a sharp policy.
Two tracks: **T**, a teacher that does not flatten (Gumbel improved-policy targets, search values
into the critic, surprise-gated CE, an entropy target), judged on the model's own search gap; and
**B4′**, paired counterfactual credit priced to the end of the turn (common chance, a shared redeal
of hidden cards per pair, the critic read at the pre-deal afterstate).

## Outcomes

* **A self-referential teacher loops.** Targets built from the network being trained follow it:
  R29 (k=4 @16, CE 0.5) entropy 0.21 → 0.93 in 13M; R31 (@32, CE 0.1) the same, slower. The
  instrument is KL(target ‖ policy) = `search_ce` − `search_target_entropy`: rising means the loop.
* **A frozen teacher pays.** R32 (the heads soup searched at k=4 @32, CE 0.1): SWA **1645**, the
  strongest no-search model at the time, 59.0% over the plain leg; it absorbed about half of the
  teacher's margin (raw soup 1628 → R32 1645 → teacher 1662). Re-freezing to a student is right only
  if the student *searched* beats the teacher searched at the training setting.
* **The rollout root as teacher** (R39, the R34 soup searched by the rollout root): SWA **1676**,
  but what transfers is the teacher's raw policy (R45 level with R39), and the root's own margin
  does not distil at any tried budget or weight (R40–R42).
* **Search values into the critic** (R35, β 0.5 at 1 in 8): level; no optimism bias.
* **Saturated logits.** No logit cap lets a saturated move learn (A9 tanh: no gradient; A10–A12
  leaks drift the network). **The logit-gap loss** (`--logit-gap-coef 1e-3 --logit-gap 7`, R34) does:
  greedy +7.6 / +2.1 per seat over the plain leg, replicated on seeds 45 and 46, its soup 1644; it
  went into the recipe. With the Cuban Missile Crisis scenario (R33) the USSR learns the defence
  (suicides 87% → 59%); without the scenario it does not, because self-play never reaches the trap
  ([`../log/E7_saturated_policy_and_caps.md`](../log/E7_saturated_policy_and_caps.md)). Anything
  self-play does not reach needs seeding.
* **R32 + R34 (R36)** is R32's gain, not additive.
* **B4′ is parked.** R37 and R38 (4× the pricing) stayed inside branch spread and moved no leak; the
  share of priced decisions with a clear better option sat at 14% throughout, so the term lacks
  weight against the PPO update. A retry is a coefficient cell, not more pricing.
* **CPU checks** ([`../log/P32_cpu_checks.md`](../log/P32_cpu_checks.md)): the human US opening is
  +0.5 ± 1.5 against the heads line's own (no arm), though the critic rates it −12.9; Wargames at a
  lead of 6 is declined correctly; Chernobyl's region choice holds; **the critic sees the
  forced-exit DEFCON trap** at entry (US −0.08, USSR 0.00), as does the attacker's, so no trap arm.
* **A staged-card observation proposal was withdrawn:** Grain Sales' drawn card is in the US hand
  and Star Wars' pick is over a visible discard pile, so the US always chooses with the card in view.
* **The minibatch and width arms** (R46–R48, A13) raise the raw network but not the searched player:
  under the rollout root the R34 soup (1791) stays above R39 (1779) and A13+R48 (1767). That is
  P33's question.
