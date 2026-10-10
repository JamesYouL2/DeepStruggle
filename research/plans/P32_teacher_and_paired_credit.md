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
| **T1 Gumbel improved-policy targets** | the CE target is the Gumbel root's improved policy, π′ ∝ π·exp(σ(completed Q)) over its candidates with the untouched tail renormalised (`ai/search/gumbel_root.py` has the root; the target export is built). **As first specified (k = 4 @16, CE 0.5) it fails** -- see *The T1 loop* below; the live cells are R31 (32 evaluations, CE 0.1) and a **frozen teacher** | the premise "sharp, so no entropy injection" holds only when the search noise is small against the policy's top-two gap (median ~5 nats); at 2–6 evaluations per candidate it is not, and a target built from the network being trained follows the student |
| **T2 search values into the critic** | the root's search value mixed into the value target, `(1−β)·λ-return + β·V_search`, β 0.5 first; policy loss unchanged. **Built** (`--search-value-beta`, `6370f5a`): at searched rows `v_win`'s target becomes (1−β)·λ-return + β·Σπ′·completed Q, blended *after* GAE so the advantages are untouched; the searcher runs with the CE off; `search_value_vs_return` and `_vs_critic` logged. Planned as the next slot after R32's read: frozen soup teacher, k = 4 @32, β 0.5, subsample 1/8, **alone** (cleaner than on top of T1). Caution: with a frozen teacher Σπ′·completed Q is the teacher's value under its own improved policy -- above the student's λ-return early in the leg -- so if `search_value_vs_return` is uniformly positive the searched 1/8 of rows carry an optimism the rest do not; the fix is to blend the *difference* from the student's root value, not the level | it cannot flatten the policy; it attacks the critic that bounds search's own leaves and every playout read in B4′; it is how the AlphaZero family's value head learns. Read: critic sign agreement with playouts (the E5-15 instrument) and the opening familiarity bias (−12.9 against +0.5), then the search gap, then strength |
| **T3 teach only where the teacher knows better** | the CE term weighted by the root's value improvement over the policy's choice (or KataGo's policy-surprise weight), zero where the search agrees; `POINT_NODE` only when the gap is large | removes the flat-target noise at 84-way nodes without naming a node type |
| **T4 R18 with an entropy target** | R18's recipe plus a dual-variable entropy target held at the plateau's 0.30 | separates "the targets are informative but broad" from "the targets are noise": if T4 gains where R18 did not, the information was there |

**A precondition, from the Cuban Missile Crisis probe (ts-main, 2026-10-10;
`research/log/E7_cuban_missile_crisis_probe.md` on `hand-knowledge-tracking`).** The heads
policy is saturated in places: a USSR coup under the US's Cuban Missile Crisis that loses on the
spot leads the alternatives by a median 19–33 nats, P(coup) = 1 in float32. Forcing the combo as
environment in every game (`E7-A8-R1-S44@6400M+R28`, 200M) taught nothing -- PPO's gradient is
scaled by 1 − P = 0 and the alternative is never sampled -- and Gumbel @256 cannot override it
either: σ(completed Q) tops out near 15 at 256 evaluations, so a 28-nat prior wins and the
search walks into the trap 64% of the time. **For T1 this means π′ = π·exp(σ(Q)) equals π
wherever π is saturated, so the CE gradient (π′ − π) is ~0 exactly where the teacher knows
better.** T1 therefore runs on a bounded logit gap. **Not the tanh cap (A9).** As first built,
`--ladder-logit-cap C` put each legal move's deficit to the top legal move through C·tanh(d/C):
greedy play unchanged, addable on resume, sampled entropy 0.28 → 0.41 at C = 7 -- but it restores
exploration and not learning. The gradient reaching a saturated move's raw logit is the softmax
gradient times 1 − tanh²(d/C), about 0.0013 at a 28-nat deficit (1.2e-6 against 9.2e-4 on the
trap's influence), and lowering the top logit moves every capped logit with it, so the softmax
does not change. Measured: the tanh cap with the trap scenario (`E7-A8-R1-S44@6400M+A9-R28`)
showed no change at 50M (154 of 172 suicides) and was stopped at 6,460M. The same attenuation
would hit T1's CE (π′ − π) through the tanh. (Mean-centring was measured destructive too, KL
0.65 at C = 7.)

**The T1 loop (ts-main, 2026-10-10).** T1 as specified -- `E7-A8-R1-S44@6400M+R29`, uncapped,
k = 4 @16, CE 0.5, all nodes 1 in 8 -- injected entropy *worse* than R18: policy entropy 0.21 →
0.93 within 13M, pool win rate 0.658 → 0.643, the CE 70–80% of the gradient. The targets start
sharp (entropy 0.17 against the policy's 0.21), but at 2–6 evaluations per candidate the search
noise -- σ scaled by (50 + n)·0.1 at n ≈ 6 is a few nats -- moves the target's argmax between
similar positions; the CE learns their average, and the targets, built from the network, follow
it (0.53): **a feedback loop**. Stopped. Two cells replace it:

* **R31** -- 32 evaluations, CE 0.1, 200M, against the plain 6,400–6,600M leg (running).
* **A frozen teacher** -- the targets from search on a *held* network (the heads soup), re-frozen
  from the student every ~100M only if the student pulls ahead. Standard distillation practice,
  and the direct break of the loop: the target cannot follow the student. It also makes T3's gate
  well-defined, since the teacher's value no longer moves with the student. Same cost as R31.
  **Built and running as R32 -- and in its first 10M the loop is broken** (the KL series falls
  where R31's rose; see *Runs*). The frozen form is T1 from here; R31 is stopped.

**The loop's instrument.** KL(target ‖ current policy) = `search_ce` − `search_target_entropy`,
both logged per iteration. On the stopped R29 it read 0.19 at the start (0.359 − 0.173) and rose
to ~0.48 (1.01 − 0.53) while the policy flattened: the targets chasing the student. Reported per
snapshot for R31 and the frozen-teacher cell; rising → the loop, flat or falling → a teacher the
student can close on. Entropy alone cannot tell those apart.

**Caps: T depends on no cap.** Every cap gradient other than the plain tanh drifts the network:
straight-through (A10) fast, a symmetric 0.1 leak (A11) slowly, an upward-only leak (A12) too,
even weighted to spare the top move -- a gradient with no effect on play still reaches the shared
trunk, and at an 84-way placement node ~80 saturated moves each leak ~p·A. The plain tanh cap
(A9) is stable (entropy 0.41, KL ~0.01, pool 0.66–0.67 over 70M) but cannot learn a saturated
move. **The saturation remedy is a loss term instead:** `--logit-gap-coef C --logit-gap G`,
C · mean over legal moves of relu(top − z − G)² (~390 per row on the 6,400M policy).
`E7-A8-R1-S44@6400M+R30` (gap 7, coef 1e-4, plus the trap scenario) is stable so far -- entropy
0.21 → 0.54 and levelling, KL ~0.01, pool win rate flat at 0.657, penalty 401 → 36 -- and its trap
probe at 6,450M says whether a saturated move becomes learnable under it. Every T arm still
reports **the share of searched decisions with max p > 1 − 1e-6** before and after.

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

## Checks (CPU) -- done 2026-10-10 by ts-main, `research/log/P32_cpu_checks.md` on `hand-knowledge-tracking`

1. **The US opening on the heads line.** Locked at France +6, Italy +2, Iran +1 in 98.9% of
   games; the human opening is West Germany 4, Italy 3, Iran 2. `setup_oracle.py` on the heads
   soup: the human opening is **+0.5 ± 1.5** against its own -- no arm. But the critic rates the
   human opening **−12.9** against a +0.5 playout: the critic is biased toward its habitual
   opening, a calibration finding carried into T2's reads (search values into the critic should
   move exactly this kind of familiarity bias).
2. **Wargames at a lead of 6.** The draw is right: playing on reads **−0.212 ± 0.045** over 46
   positions. Not a leak; nothing to sharpen.
3. **Chernobyl off Europe.** The region choice holds: forcing Europe is **−0.019 ± 0.008**
   against the chosen region over 220 positions. B5's Chernobyl trigger does not fire.
4. **The search gap as a leaderboard column** for every main player, and searched players as
   leaderboard anchors above the saturated panel -- still to do.

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

**4a done (ts-main, 2026-10-10, `P32_cpu_checks.md`): the critic sees the trap** -- error at entry
−0.08 for the US and 0.00 for the USSR -- so **4b is not triggered**, and the trap accounts for only
29 of 142 DEFCON-1 losses in the bank. The "critic completely unaware" reading was the policy's
behaviour, not the critic's estimate. What remains of §4 is 4c: whether the *choices* that enter the
trap (the attacker's discard and Five Year Plan timing; the defender's keep-or-return) get their
credit, which B4′'s leak log answers by decision type.

**The attacker's side, measured the same day (ts-main, `P32_cpu_checks.md`): the attacker's
critic sees the trap too, so §4 closes.** Over 6,000 games at temperature 1 there were 436
Aldrich Ames discards at DEFCON 2, of which only 8 offered a discard that traps the US; the USSR
took a trapping discard in all 8 (trapping options were 73% of the choices), and its critic
values the trapping discards +0.948 against +0.759 for the others (+0.19). Small sample, same
direction as the defender's side. No §4 fix arm is justified; the entering choices are already
priced, and the forced-exit trap stays only as a bank that B4′'s leak log can confirm against.

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
* Checks, resolved: 1 → no arm (+0.5 ± 1.5; the critic's −12.9 goes to T2's reads); 2 → no leak
  (the draw is right); 3 → B5's trigger does not fire (the region choice holds); 4a → 4b not
  triggered (the critic sees the trap).
* 4: the owner's.

## Follow-ups

* T1 promoted → the same target as the from-scratch teacher (P30 C5's gate passes on the base
  recipe); a stronger root (k = 8 @64) as the second cell.
* B4′ promoted → extend the option sets (coup targets with many legal targets, top-k placements);
  the leak log replaces the ported banks as P31's step 0.
* Both level → the remaining headroom is in lookahead; publish the searched soup and spend the
  next arms on the trunk (P30 C1).

## Runs

Launched and read by ts-main; this ledger mirrors [`../runs.md`](../runs.md).

* `E7-A8-R1-S44@6400M+A9-R28` -- tanh cap + the Cuban Missile Crisis trap scenario; **stopped at
  6,460M**, no change at 50M (154 of 172 suicides): the tanh cap does not let a saturated logit
  learn. Closes A9 for T1.
* `E7-A8-R1-S44@6400M+A10-R28`, `+A10`, `+A11`, `+A12` -- the straight-through, symmetric-leak
  and upward-leak caps: **every cap gradient but the plain tanh drifts the network**; the A10
  trap check never got a clean read. Caps are off the plan for T.
* `E7-A8-R1-S44@6400M+R29` -- T1 as specified (uncapped, k = 4 @16, CE 0.5): **stopped**, entropy
  0.21 → 0.93 in 13M, the target-follows-student loop.
* `E7-A8-R1-S44@6400M+R31` -- T1's second cell, 32 evaluations, CE 0.1, self-referential teacher:
  **stopped** -- KL(target ‖ policy) 0.27 → 0.53 rising, target entropy 0.17 → 0.41, policy entropy
  to 0.71. The loop at half the noise and a fifth of the coefficient.
* `E7-A8-R1-S44@6400M+R30` -- the logit-gap loss (gap 7, coef 1e-4) + the trap scenario: stable;
  **trap probe at 6,450M: USSR suicides 152 of 188 (81%) against 89–91% uncapped**, about 3 SE
  lower -- a saturated move is starting to move under the gap loss, slowly. Running on.
* `E7-A8-R1-S44@6400M+R33` -- the gap penalty at **1e-3** (10× R30) + the trap scenario: **USSR
  suicides 72% at 6,600M against the plain leg's 87%, the first behaviour change in any variant**;
  the coup's lead on the trap positions 9 nats against 19 plain; stable, entropy 0.83. **Done, and
  the trap is learned** (`research/log/E7_saturated_policy_and_caps.md`): USSR suicides at
  6,450–6,800M 84, 87, 72, 65, **59%** against the plain leg's 83–89%. **Not by declining the coup**
  (R33 declines 4 of 214, the plain leg 11 of 158) **but by rebuilding Cuba**: R33 coups and pays
  in 75 of 214 follow-ups against 12 of 158, putting 2+ influence into Cuba between the US's crisis
  headline and the follow-up; the US wins ~66% of the combo games against it against ~82% against
  the plain leg. A bounded logit gap makes a saturated decision learnable, and the network found
  the cheapest defence -- not the one the scenario was written around. The gap loss is the
  saturation remedy.
* `E7-A8-R1-S44@6400M+R34` -- the gap penalty 1e-3 alone, no scenario, 6,400 → 6,800M against the
  plain leg (running; sampled entropy ~0.84): **the cost read** that decides whether the gap loss
  goes into the recipe -- both a greedy field at matched windows and a T = 1 head to head, to tell
  the loss's own cost from the broader rollout policy it induces; if only T = 1 costs, the next
  cell is 3e-4. **Done -- and it splits exactly along that axis, the other way from the worry.**
  Greedy field (1,000 games a side): R34's 6,720–6,800M SWA rates 1546 against the plain leg's
  1508 (heads soup 1550); per seat over the plain leg, US / USSR, +2.2 / −1.2 → +6.1 / +1.8 →
  **+7.6 / +2.1** at 6,480 / 6,640 / 6,800M, head to head 52.2 → 54.4 → **55.5%**. **T = 1 head
  to head, the same pair: 36.6%** (US 38.8, USSR 34.4). The gap penalty is not a cost to the
  argmax -- it improves it, and the gain grows with the window; what breaks is the *sampled*
  policy, because the mass it keeps on bad moves gets played. Reading: bounded gaps keep
  alternatives explored in self-play, a quiet exploration bonus. **The gap loss passes the
  per-seat rule and goes into the recipe**; the product is greedy, so the T = 1 cost is a
  training-data question (weaker rollouts and a weaker pool), not a product one.
* **R32 + R34 from the 6,400M root** (frozen-soup T1 targets + the gap penalty 1e-3) -- the next
  slot, when R32's leg 2 ends at ~6,800M, with R35 (T2) in the other. No interference expected:
  the CE's gradient (π′ − π) is ~0 on deep-tail moves where both teacher and student have no mass,
  so it shapes the top candidates inside the 7-nat band, while the penalty acts only beyond it.
  Read at matched windows against both parents and the plain leg, greedy and T = 1: additivity
  (sum against max of the two gains) is the result; watch the pool win rate and the advantage
  mean, since rollouts and the pool are sampled at T = 1 from a broader policy. Queued as
  `E7-A8-R1-S44@6400M+R36`, launching when R32's leg 2 exits (~07:00).
* **Side result from R34: the gap loss alone does not teach the Cuba trap** -- 142 of 164
  follow-ups still suicide at 6,800M (87%, the plain leg's level). The defence needed the bounded
  gaps *and* the scenario together (R33, 59%). The ~9% first quoted is how often the combo is
  *available* against a scripted US; **the network's own US never plays it, so self-play's natural
  reach of the trap is ~0.** This is P31's division of labour measured directly -- the cap makes a
  saturated decision *learnable*, the seeded state supplies the *samples* -- and the conclusion is
  the strong form: **anything self-play does not reach needs seeding, whatever the exploration
  fix.** The scenario list stays a live instrument for unreached chains, not a scaffold the cap
  retires.
* `E7-A8-R1-S44@6400M+R32` -- **T1 with a frozen teacher** (`--search-teacher-checkpoint`, the
  heads soup as the searcher, Gumbel k = 4 @32, CE 0.1 -- R31's settings, so the pair differs only
  in whose network searches). **First 10M: the loop is broken.** KL(target ‖ policy) 0.81 → 0.39 and
  falling (R31: 0.27 → 0.53 rising); target entropy steady at ~0.22 (R31: 0.17 → 0.41); policy
  entropy 0.22 → 0.46 and levelling (R31: 0.71); pool win rate 0.658 → 0.667 (R31 ~0.66); KL to
  π_ref ~0.01; saturated searched decisions 0.27 → 0.17. Entropy still rises -- search noise varies
  the target between similar positions, so the average is broader -- but it has stopped. Runs to
  6,600M, then the search gap and head to head against the plain leg. **Arranged for the end-of-run
  read:** (1) the teacher's margin -- Gumbel k=8 @256 searched by the frozen soup against R32's raw
  6,600M and against the plain leg's raw 6,600M, beside each network's self-search gap; **the
  re-freeze signal is the teacher's margin over R32 shrinking relative to its margin over the plain
  network**, which sets the cadence P32 guessed at ~100M. (2) If R32 is level or better on strength,
  a k = 4 @64 cell at the same CE, to measure how much of the 0.22 → 0.46 entropy rise is
  target-averaging noise -- the only cost left in the recipe. **Early read, 6,410–6,480M SWA:
  the first teacher result that buys strength.** Against the plain leg's same window 55.0% ± 1.1
  (US 50.6, USSR 59.5), about +35 Elo from 80M steps; against the heads soup as a common opponent
  R32 48.4% (US 44.8, USSR 51.9) against the plain window's 42.6% (38.9, 46.3) -- +5.9 / +5.6 per
  seat. At 6,487M: KL(target ‖ π) 0.33 and falling, policy entropy 0.48, KL to π_ref 0.0065, pool
  win rate 0.681. **The student is already level with or near the soup it learns from**, so the
  teacher's-margin read decides what comes next: the natural continuation is to **re-freeze the
  teacher from the student's own SWA** (not a snapshot -- the SWA is the stable, ~+45 Elo object),
  or from a soup that includes R32, so the teacher stays ahead -- expert iteration with a held
  teacher, the cadence set by the margin read rather than by steps. **6,600M (field of 11, greedy,
  1,000 games a side): the gain is growing.** R32's 6,520–6,600M SWA rates 1555 and the heads soup
  1550; the plain leg's same windows 1508 and 1502. Per seat over the plain leg, mean over seven
  common opponents, US / USSR: +4.6 / +5.7 at 6,480M, **+7.9 / +7.4 at 6,600M**, every opponent
  positive in both windows; head to head 55.0% → 56.7%. One run's SWA now plays level with the
  three-run soup *raw*. R32 continues unchanged to 6,800M (leg 2, same name) to see whether it
  climbs past its teacher. **The search gap at 6,600M** (Gumbel k8@256 on each network against
  the same network raw, 1,000 games a side): **on R32 51.2% ± 1.1** (as US 47.0, as USSR 55.4);
  on the plain leg 62.6% (61.4 / 63.9). A shrink of ~7 SE, past the 3-SE promotion bar: **T1 passes
  the plan's rule on both halves** (gap shrunk, head to head above the plain leg). Caveat: Gumbel's
  Q comes from R32's own critic, which T1 left untrained toward the search, so "the policy absorbed
  its search" and "a sharper prior leaves the old critic little to correct" read the same here;
  the teacher-margin read (the soup searched, against R32 raw and against plain raw) separates
  them -- if R32 absorbed the search, the soup's margin over it shrinks too. Either way the US
  figure (search *below* 50% on R32 as US) says the critic is now what bounds the next search
  gain, which is T2's case, after the re-freeze. **Next slot (R34 frees it at
  ~6,800M): the re-freeze, before T2** -- gated on the teacher's-margin read, not the raw rating:
  the teacher is the soup *searched*, so R32 being level with the raw soup does not mean it has
  caught the teacher. **Correction (ts-main): the teacher actually in use is the soup at k = 4 @32,
  about +25–35 Elo above the raw soup, not the k8@256 figure (~+60), which overstates it**; both
  margins are read. **Decision rule, queued (`r32_refreeze.sh`, right after the search gap):**
  the soup-searched margin over R32's raw 6,600M against its margin over the plain leg's (1,000
  games a side each), then the equal-weight SWA-made soup of R32's 6,520–6,600M SWA with the heads
  soup rated in a five-model greedy field beside R32's SWA, the heads soup, the plain leg's window
  and the A4 soup. **If the margin over R32 has shrunk clearly relative to the plain leg's, or the
  new soup rates above both its parents, the re-freeze takes the slot R34 frees** -- the teacher is
  that soup if it rates above both, otherwise R32's SWA. **If neither, T2 takes the slot.** If the
  loop's second iteration gains little, T2 becomes the lever rather than the follow-up.
  **Reads in, and the literal rule was wrong (ts-main, 2026-10-11).** The soup searched (k8@256)
  beats R32's raw 6,600M 55.6% ± 1.1 against 68.0% over the plain leg's raw -- a margin shrink of
  ~8 SE, which the rule reads as "re-freeze". The R32-SWA + heads-soup soup rates 1522 against its
  parents' 1529 and 1521 (R32 SWA vs the heads soup 50.7%): not above both, so not a teacher, and
  a negative worth keeping -- souping a distilled student with its teacher gains nothing, the two
  are already close in function space. **But re-freezing to R32's SWA would weaken the teacher:**
  the teacher is used *searched*, search adds ~nothing on R32 (51.2% against itself) while the
  soup searched still beats R32 raw 55.6%, so R32-searched ≈ R32 raw ≈ soup raw, and all three sit
  below the soup searched. The margin condition was necessary and not sufficient. **The corrected
  rule: re-freeze to X only if X searched at the training setting (k = 4 @32) beats the current
  teacher searched at the same setting** -- one tournament, and it subsumes the margin read. So:
  R32 continues unchanged on the soup (leg 2, to 6,800M), and **T2 alone takes R34's slot**
  (`E7-A8-R1-S44@6400M+R35`: β 0.5, the soup as searcher at k = 4 @32, no CE; `search_value_bias`
  logged overall and per seat). If T2 repairs the critic, R32 + T2 is the student worth
  re-freezing to. **R35 at 6,480M: level.** SWA 6,410–6,480M against the plain leg's window
  49.2–49.0% (US 49.6, USSR 48.9); against the heads soup 43.0% where the plain window scores 42.6%
  (US −2.8, USSR +3.6) -- R32 was +5.9 / +5.6 at this window. Training healthy: entropy 0.29 level,
  search value vs critic 0.07 and flat, signed bias −0.009 (US +0.001, USSR −0.019): **no
  optimism**, so the level-vs-difference worry does not arise at β 0.5. The critic has not visibly
  moved toward the search value yet; at β 0.5 on 1/8 of rows that may be slow. Reads queued: the
  field at 6,640 and 6,800M, and **the search gap at 6,800M, which is T2's real test** -- a better
  leaf evaluator should make R35's *own* search gain *more* than the plain leg's 62.6%, the
  opposite sign from T1's read -- plus the critic's sign agreement with playouts and the opening
  familiarity bias (−12.9 against +0.5), the two critic defects T2 was aimed at.
* The attacker's side of 4a -- measured: the attacker's critic sees the trap (8 of 8 trapping
  discards taken; +0.948 against +0.759). §4 is closed.
