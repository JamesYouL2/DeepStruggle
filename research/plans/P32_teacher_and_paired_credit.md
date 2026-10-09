# P32 — A teacher that does not flatten the policy, and paired credit at the turn's end

**Status:** proposed (2026-10-09).
**Root:** `E7-A8-R1-S44@6790M` (the heads line's last saved state before its 6,800M plateau,
`E7-A8-R1-S44_20261007_162257/resume_6790…`), the same root as the search-target arm
`E7-A8-R1-S44@6790M+R18`. Controls: the plateau SWA `E7-A8-R1-S44@6720..6800M`, the heads soup
`E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M`, the three fixed references of the plateau rule, and
R18 itself as the failed-teacher baseline.
**Gate:** none for the CPU checks; T1 needs the target export from the Gumbel root; B4′ needs the
1c rewrite below. Arms are branches, two at a time.
**Needs approval:** nothing in T or B4′ touches `engine/` or the observation. Section 4 is an
observation proposal (owner-held) and waits.

## Where the record stands (main, 2026-10-09)

* **The card-conditioned heads are the base.** `E7-A8-R1-S44` kept climbing ~3,000M past the
  plain line's plateau and levelled at 6,800M; one heads run is level with the best soup of three
  plain runs, and the SWA-made heads soup beats that soup 59.4% -- the strongest model
  ([`../log/P31_branch_arms_B1_B2.md`](../log/P31_branch_arms_B1_B2.md)). Souping adds ~+45 Elo
  on top of whatever a line reaches.
* **Search on top still pays about what it did on the plain line:** Gumbel k=8 @256 ~+65 Elo on
  the raw and SWA networks, ~+53 on the soup; PUCT @128 ~+26–35
  ([`../log/E7_gumbel_headroom.md`](../log/E7_gumbel_headroom.md)). The headroom did not shrink
  with the stronger base.
* **Search as a teacher has failed twice the same way.** R18 (CE 0.5 toward PUCT @64 visit
  counts, every node type, 1 in 8) was level after 310M; E7-18-44 was level after 400M. Both times
  **entropy jumped 0.30 → ~0.5 within 10M steps** and the panel's US seat fell. That is the
  diagnosis, not the budget: PUCT visit counts at 64 simulations are a broad distribution by
  construction (exploration spreads visits; at an 84-way placement node they are nearly flat), so
  cross-entropy toward them injects entropy into a 0.30-entropy policy and undoes the sharpness the
  line earned. The teacher measured on top (Gumbel @256) was also never the teacher trained on.
* **The heads' free ride reached more than Wargames.** At 6,800M Chernobyl is evented in 24% of the
  US's plays (2% on the plain line), 87% of them on Europe, and the heads soup *exploits* it: 76%
  of its Late War points go into Europe while it is closed, +26 points over the same games without
  the event, and 14.8% of those games end in a US Europe Control win against 0.4%
  ([`../log/per_checkpoint/`](../log/per_checkpoint/README.md), the heads soup's report). Glasnost
  13% → 30%, Kitchen Debates 5% → 21%, OPEC and Alliance for Progress at 5+ VP evented 90% per
  holding. **Still on the floor:** One Small Step 2%, Special Relationship 4%, Nuclear Subs 1%,
  Summit and Nuclear Test Ban 1% -- the cards with no frequently visited, immediately credited
  sub-decision to bootstrap from.
* **B4 as built cannot do its job.** `ai/training/mode_cf.py` prices every option with *one*
  playout *to the end of the game* on the opponent's *true* hand. A +2-point spot is a 0.04 shift
  in win probability against an outcome sd of ~0.9 (the review needed 64 pairs × 300 positions to
  put ±0.5 on +2.1), the term learns from private information, and the game-end tail is where both
  the variance and the cost (~7 s per iteration at k = 16) come from.
* **The panel is saturated** (93–95% in both seats), so the per-seat rule cannot separate top
  models; only head-to-head and the leaderboard do.

## Track T — fix the teacher

Each arm is a branch from the root, +400M, two at a time, read against the plateau SWA, the heads
soup, the three fixed references and R18. **The new standing instrument is the model's own search
gap** -- Gumbel k=8 @256 against the raw snapshot, 1,000 games a side -- logged per 80M SWA: a
teacher that works shrinks it, and it reads long before strength can.

| arm | change | why |
|:---|:---|:---|
| **T1 Gumbel improved-policy targets** | the CE target is the Gumbel root's improved policy, π′ ∝ π·exp(σ(completed Q)) over its candidates with the untouched tail renormalised, k = 4 at 16–32 evaluations (`ai/search/gumbel_root.py` has the root; the target export is the build) | it is sharp (no entropy injection), defined for every action through completed Q, and the headroom log shows k=4 @16 plays PUCT @128 even -- a stronger teacher at an eighth of the cost, so the arm runs near the plain line's throughput rather than R18's 27k steps/s |
| **T2 search values into the critic** | the root's search value (PUCT or Gumbel, the same searched decisions) mixed into the value target, `(1−β)·λ-return + β·V_search`, β 0.5 first; policy loss unchanged | it cannot flatten the policy; it attacks the critic that bounds search's own leaves and every playout read in B4′; it is how the AlphaZero family's value head learns. Read: critic sign agreement with playouts (the E5-15 instrument), then strength |
| **T3 teach only where the teacher knows better** | the CE term weighted by the root's value improvement over the policy's choice (or KataGo's policy-surprise weight), zero where the search agrees; `POINT_NODE` only when the gap is large | removes the flat-target noise at 84-way nodes without naming a node type |
| **T4 R18 with an entropy target** | R18's recipe plus a dual-variable entropy target held at the plateau's 0.30 | separates "the targets are informative but broad" from "the targets are noise": if T4 gains where R18 did not, the information was there |

**Decision.** Promote an arm iff its search gap shrinks by > 3 SE over the leg *and* it beats the
plateau SWA head to head with the three-reference mean not lower; adopt on the seed-43 replicate
(section 5). If T1–T4 all stay level, the ~+65 Elo is lookahead the network cannot represent at this
size, and the product is the searched soup -- a finding that closes the teacher question.

## Track B4′ — paired counterfactual credit at the turn's end

The owner's design, replacing 1c:

* **Positions.** Decisions with a small option set where every option can be enumerated: play
  mode (≤ 5), event branches (≤ 8), Chernobyl's region (6), UN Intervention's card, Grain Sales'
  keep-or-return, the headline and card choices (~8), coup targets where few are legal.
  Influence placement (84-way) stays out. Targeted, not uniform: decisions where the policy is
  confident (max p > 0.9) *and* some legal option sits below p ≈ 0.02 -- the definition of a
  narrow spot -- plus a small uniform share so the term is not applied only there.
* **Pairing.** m pairs per decision. In pair j every option is played from the same clone with
  the same chance stream (dice and draws in common), and **the cards the decider cannot see --
  the opponent's hand and the deck -- are redealt once per pair, consistently with what it can
  see** (`ai/search/dmcts.determinize`, the honest searcher's). Within a pair the options differ
  only in the move; across pairs the hidden information varies, which is exactly the expectation
  the policy is allowed to learn. Nothing is priced on the opponent's true hand.
* **Horizon and value.** Play to the **end of the current turn** -- the `TURN_CLEANUP` afterstate,
  before the deal -- and read the critic there from the decider's view; the terminal utility if
  the game ends first. At the turn's end almost nothing is private (one held card each; the China
  Card is public), so the critic there is close to a public-state value and the comparison carries
  no hidden-information bias worth naming. On turns 9–10 play to the end of the game instead.
* **The statistics.** A paired difference of turn-end values with common chance and a shared
  redeal has an sd of roughly 0.1–0.2 in win-probability units against ~0.9 for paired game
  outcomes: a 0.04 effect reads at m = 32 (SE ≈ 0.03) where game-end pairs would need thousands.
  The critic's error is systematic, not random -- it does not average away, and it does not add
  noise (its cost is the first caveat below).
* **What feeds back.** Two forms, the second preferred: the all-options term as built, each
  decision's contribution **precision-weighted** by 1/SE² of its paired estimate; or the
  **improved-policy target** π′ ∝ π·exp(β·Q̂) by cross-entropy, applied only where the best option
  clears the taken one by ~3 SE. Both are sharp and speak only with evidence; a hard significance
  gate on an unweighted term is a selection bias (it pushes only toward lucky options) and is not
  used.
* **Cost.** ~200 priced decisions per iteration × 5 options × 16 pairs × ~45 plies ≈ 700k plies,
  under 10 s of batched stepping -- the current k = 16 cost, with usable targets; pricing runs from
  the buffer in its own stream. The clone, the determinizer, the batched playout path from 1c and
  the critic all exist; the build is the pairing, the turn-end stop and the weight.
* **By-product: a continuous leak bank.** Every decision whose taken option is beaten at z > 3 is a
  measured leak, found by the trainer on the model's own distribution; logged by card and decision
  type it is the generic version of the review's position banks, which were never ported.

**Reads.** One Small Step by box pair and side; Special Relationship with NATO; Glasnost with The
Reformer; the Wargames 6-vs-7 line at the branch; Grain Sales' keep-or-return; the leak log's top
entries before and after. Then the search gap and strength as for T.

**What it cannot do.** It inherits the critic's blind spots (a payoff landing after the turn; a
board the critic has never seen) -- mitigated by a small share of decisions played to the game's
end as a calibration check on the turn-end readings, and by T2. It credits single decisions, not
chains: Nuclear Subs still needs the state shown first (P31 1b, the exploitation step forced as
environment), and the Subs coup needs an effect-conditioned coup-target correction before any credit
can land on it -- queued behind T and B4′ as **B6**.

## Checks (CPU, before or alongside the arms)

1. **The US opening on the heads line.** It is locked at France +6, Italy +2, Iran +1 in 98.9% of
   games, West Germany untouched; the human opening is West Germany 4, Italy 3, Iran 2. Run
   `setup_oracle.py` on the heads soup against the human opening and West-Germany-first variants.
   If a sane opening reads +3 or more, the E5-21 recipe (game-result setup credit + the entropy
   floor, +32 Elo there, never tried on this line) is the known fix and becomes an arm.
2. **Wargames at a lead of 6.** The event is taken in ~45% of such spots and ends in a draw (28% of
   all draws). Paired playouts at the branch: if the position's win probability is below 50% the
   draw is right and this is not a leak; if not, B4′ at the branch (two options) is the sharpening
   tool the washout run showed on-policy play cannot supply.
3. **Chernobyl off Europe.** The report shows the exploitation for Europe; the same table for Asia,
   Central and South America (the USSR's own plays fire it there far more often) says whether the
   region choice is right or Europe-only.
4. **The search gap as a leaderboard column** for every main player, and searched players as
   leaderboard anchors above the saturated panel.

## 4. The forced-exit DEFCON trap (owner's observation, 2026-10-10) -- replaces a withdrawn proposal

**Withdrawn:** a staged-card observation proposal stood here. It was wrong. The engine moves
Grain Sales' drawn card **into the US hand, marked known, before the keep-or-return decision**
(`engine/src/events/mid_war.cpp`, `trigger_grain_sales`: the decision is the drawn card's own
`SELECT_PLAY_MODE` with the decline as one option), and Star Wars' retrieval is a `SELECT_CARD`
over a discard pile the observation shows in full (`late_war.cpp`, `trigger_star_wars`). The US
chooses with the card in view both times; `obs_flags::STAGED_CARDS` is a leftover of an older
two-decision shape. The 48 Ortega-suicide losses "inside another event" are therefore choices,
not blindness -- a kept Grain Sales card or a Star Wars pick played into a DEFCON-2 Cuba coup --
and belong to the trap below.

**The trap.** A side holding a DEFCON-suicide card (for the US, Lone Gunman; for either side
any event in `ai/eval/safety.py`'s degrade-with-no-choice list) and then losing a card to an
effect -- Aldrich Ames' discard, Five Year Plan, Terrorism, Missile Envy, a kept Grain Sales
card -- must play its whole hand this turn: it can no longer hold the card over, the space race
takes at most one card a turn (the second attempt only above box 2), UN Intervention is one
exit if held. With more suicide cards than exits the loss is decided several decisions before
AR7, **and the critic does not see it**: "DEFCON 1, forced by the winner's play" is 6.7% of all
games at 6,800M (8.3% of US wins, 5.4% of USSR wins), and P8's measurement stands -- the critic
moves by at most 0.014 at the deciding choice, flat from 5M to 480M
([P8](P8_teach_the_defcon_conjunction.md)).

**Everything needed is in the observation**: own hand count (`globals[71]`), the action round
(`[7]`), the turn, space attempts used this turn (`[59]`), DEFCON, and every held card's identity
and side. So this is a conjunction the trunk does not compute -- hand composition reaches the
policy only through pooled statistics ([`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)) --
and the outcome is credited several decisions late through a critic that has never priced it.
It is also the attacker's skill in mirror: forcing the trap on the opponent is the same
representation.

**4a -- the probe (CPU, first).** A bank from self-play at temperature 1: positions where a side
holds k suicide cards and, counting its remaining action rounds, its space attempts left and a
held UN Intervention, has fewer exits than k (an `ai/eval` rule with no card list beyond
`safety.py`'s). Read the critic's `v_win` there against paired playouts, and the same before and
after the card loss that closed the exits. Also: the opponent's decision that closed them (an
Aldrich Ames discard, a Five Year Plan) -- does the attacker's critic see the gain? Two numbers
decide the next step: the critic's error at trap entry, and how many games end there.

**4b -- the label-driven fix (trainer only).** A **forced-exit auxiliary target**: at every
decision, whether this side will be forced to fire a DEFCON-degrading event before the turn ends
-- the label read from the game's own continuation in the buffer (did it happen), no rule. A
head on the trunk, as P8's 8a (built, `--defcon-coef` on the provoked label) but with a label that
fires at trap *entry* rather than at the suicide. This teaches the conjunction where the
critic's target cannot, and the policy term is untouched. 8a itself is the cheaper first cell:
built, never measured, and its provoked label is a subset of this one.

**4c -- credit at the choices that enter the trap.** B4′ already covers the decisions where the
defender chooses (Aldrich Ames' own discard, Grain Sales' keep-or-return, Missile Envy's pick) and
the attacker's (which card to discard with Aldrich Ames; whether to play Five Year Plan): a paired
turn-end playout sees the forced AR7 inside its horizon, so the trap prices correctly there without
any special case. Read 4c from B4′'s leak log by decision type before building anything else.

**Decision.** 4a's error at entry > 0.1 → 4b as a branch from the root (two cells: 8a as built; the
forced-exit label), read on 4a's bank first and on the "forced by the winner's play" share, then
strength. Error small → the critic knows and the policy does not use it: a card-conditioned head at
the hand-management decisions (the play-mode head already exists; the discard/keep choices may
need the same treatment).

## 5. Replication and the human loop

* **A seed-43 heads run from scratch** (A8, R1, S43), with P30 C2 (the card-event target, the one
  narrow positive) as the single added factor if the owner wants it tested at saturation: the heads
  line's gain is one seed, and a second lineage is what souping across seeds needs.
* **The expert re-review on the heads soup.** The review found the leaks against a human; that is
  where Wargames, OPEC / Alliance and Chernobyl are confirmed repaired, and where the next list
  comes from. Send the soup and the per-checkpoint report.

## Sequencing under the two-arm limit

| slot | arms | note |
|:---|:---|:---|
| now | **T1** ‖ **B4′** | T1 is the cheapest high-ceiling arm; B4′ needs the 1c rewrite (~2 days) -- T4 runs in its slot meanwhile if the rewrite is not ready |
| next | **T2** ‖ **T3** | |
| then | the promoted T arm on seed 43 ‖ **B6** (effect-conditioned coup head + Subs seeding) | |
| CPU throughout | checks 1–4; the search gap per SWA | |
| overnight | the seed-43 heads run from scratch | |

Budget: T arms ~1.5–4 h each at 400M (T1 near plain throughput; T2/T3 at R18's cost); B4′ ~4 h;
the seed-43 run ~9 h to 2,800M and ~20 h to its plateau. Standing rules: two arms at a time, every
arm registered in [`../runs.md`](../runs.md) before launch, `launch_flags.py --diff` before and
after, `run_codes.py --run` on the name, readouts at 40M marks.

## Decision rules, collected

* T1–T4: promote on a shrinking search gap (> 3 SE) *and* head to head above the plateau SWA with the
  three-reference mean not lower; adopt on the seed-43 replicate. All level → the teacher question
  closes; the product is the searched soup.
* B4′: promote iff ≥ 2 of its target spots move by > 3 SE on direct counts with strength not worse;
  the leak log's top entries are reported whether or not it is promoted.
* Checks: 1 → an arm if the opening gap is ≥ +3; 2 → B4′ at the branch if the draw is wrong; 3 → B5
  (the opponent-legality head, built, unrun) if the region choice is Europe-only.
* 4: the owner's.

## Follow-ups

* T1 promoted → the same target as the from-scratch teacher (P30 C5's gate passes on the base
  recipe); a stronger root (k = 8 @64) as the second cell.
* B4′ promoted → extend the option sets (coup targets with many legal targets, top-k placements);
  the leak log replaces the ported banks as P31's step 0.
* Both level → the remaining headroom is in lookahead; publish the searched soup and spend the
  next arms on the trunk (P30 C1).

## Runs

(none yet)
