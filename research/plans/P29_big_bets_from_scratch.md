# P29 — Big bets from scratch: the structural changes that could move the ceiling

**Status:** proposed (2026-09-29). Split out of [P28](P28_strength_on_E6.md) at the owner's
direction: *focus on potentially very large improvements from scratch, rather than squeezing
100 more Elo at the end.* P28 keeps the measurement floor and the tail levers; this file is the
strength programme.
**Gate:** P28 step 0 (the panel and the E6-04 verdict) before any bet is *judged*; nothing gates
launching bet 3.
**Needs approval:** **bet 4** is an observation proposal — owner-held, stated with its cost in
floats, and it waits. Nothing else touches `engine/` or the observation. Bets 1–3 are model,
trainer and action-view changes; bet 5 is engineering plus a trainer hook.

## Why bets, and why from scratch

Every large gain in this project has been structural. Tuning has returned tens of Elo at best.

| change | size | where |
|:---|---:|:---|
| country per-entity head | +282 | [`../log/P21_ladder_status.md`](../log/P21_ladder_status.md) |
| positional board path (grouped projections) | +109 | [`../log/P21_M1_grouped.md`](../log/P21_M1_grouped.md) |
| decision context in the observation | +92 | [`../log/observation_layout.md`](../log/observation_layout.md) |
| rollout temperature at the policy's own | +135 / +170 | [`../log/E5_06_rollout_temperature.md`](../log/E5_06_rollout_temperature.md) |
| the late-E3 bundle over a flat MLP | +447 | [`../log/E4_architecture_ab_result.md`](../log/E4_architecture_ab_result.md) |

And the plateau's measured causes are structural:

* **The critic cannot price a move the policy cannot yet convert.** Half of takeable contested
  battlegrounds are missed, but the rollout oracle finds taking one worth −0.8 ± 0.6 pp under
  the policy's own continuation; the critic's sign agrees with playouts 49–65% of the time
  where they differ by ≥10 points ([`../log/P27_ops_block_stage1.md`](../log/P27_ops_block_stage1.md)).
  A terminal-only signal cannot leave that local optimum.
* **Hand composition is unrepresentable.** No card↔card path exists, so "I hold no scoring
  card, so this event is safe" reaches the policy only through pooled statistics
  ([`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md)).
* **Capacity was never judged where it matters.** M2d is 3.2M parameters, chosen at 80M
  matched steps; every larger or attention-bearing variant lost *at 80M*, a screen that selects
  for fast early learning. No larger net has a clean long run.
* **No plan over a few micro-steps** inside an ops play (P27). The one representation that makes
  an ops play a single decision (E4.1, [P23](P23_merged_influence_E4_1.md)) was parked after
  collapses that all ran under the old sharpening rollout bands.

From-scratch arms on M2d run ~50k steps/s: **560M is ~3h solo**, one evening per seed. That is
what makes this ordering affordable, and why every bet here restarts from zero rather than
resuming a plateaued state whose habits it would inherit.

**Measured on snapshots and on the SWA (owner, 2026-09-29).** Every bet and its control is rated twice: on its late snapshots, and on the SWA of its last 80M. Averaging does not enter training ([P28](P28_strength_on_E6.md), *Measurement rule*).

**One rule for all of it: judged at the plateau, never at 80M.** Each bet is compared with
its control (E6-03-44, or E6-04-44 if P28 step 0 adopts that recipe) at 480/520/560M, pooled
over the late snapshots, against P28's frozen panel, per seat by the panel rule. A bet behind at
80M and ahead at 560M is a win.

## Bet 1 — Architecture at the asymptote

**Change.** Two arms on the ladder flags:
* **M3**, card↔card self-attention over the 110 card tokens — the P21 rung blocked on the
  `board_mode`/`card_mode` split, which is this bet's first engineering item;
* **a wider/deeper M2d**, `--ladder-hidden-dim 768 --ladder-res-blocks 8`.

**Why.** The hand-composition gap is a whole class of decisions the current net can only
approximate (what to hold, what to space, when an event is safe), and the asymptote of anything
above 3.2M parameters is unmeasured.

**Procedure.** From scratch to 560M, seed 44 vs E6-03-44 at matched games *and* matched
wallclock; seed 43 to confirm. Rate at 80M too, but do not read it.

**Decision.** Adopt at matched wallclock. A win at matched games only is recorded as
"capacity-bound, throughput-limited" and routes to P18/P26 before adoption. Behind at 560M on
both seeds at matched games → capacity is not the plateau's cause; the question closes.

**Expected size if right:** the largest of any bet.

## Bet 2 — Auxiliary targets: final ownership and VP margin

**Change.** Two small heads on the trunk, off by default (`--aux-ownership`, `--aux-vp-margin`),
trained from the terminal state the buffer already holds: **who controls each country at game
end** (84-way multi-label; the 29 battlegrounds at minimum) and **final VP margin**. No
observation change; a few thousand parameters.

**Why.** KataGo's lever, aimed squarely at the first cause above: the trunk learns that West
Germany is worth holding *before* the policy can convert it, which is also what search leaves
need in bet 5. It supplies exactly the signal the self-play outcome withholds.

**Procedure.** From scratch to 560M on the step-0 recipe, seed 44 then 43. Decide before
running: loss weights (0.1 / 0.1 first), all countries or battlegrounds only.

**Measure, in order.** Critic sign agreement with playouts (the E5-15 instrument) and P27's
take-rate, then Elo. **Decision:** adopt iff sign agreement rises *and* Elo is not worse on
both seeds. A probe-only gain is logged as a critic result and kept for bet 5.

**Expected size if right:** KataGo-class sample efficiency — plausibly a plateau shift.

## Bet 3 — The merged-influence action view (E4.1), on the fixed recipe

**Change.** P23's view, already implemented: "influence, first point in X" as one decision,
defined as the composition of the two E4 steps it replaces, opt-in per agent and bit-identical
to E4 when off. No engine or observation change.

**Why.** P23 parked it after the US seat collapsed in 6 of 6 from-scratch runs and "the cause
was not found". Every one of those runs used the old rollout bands (0.15 / 0.50 / 0.10 / 0.35;
[`../log/P23_E4_1_ab.md`](../log/P23_E4_1_ab.md)), which flat 1.0 replaced on 2026-09-27 and
which were themselves collapse-prone (50–60% of training games ending at DEFCON 1). The
confound was never named. If right, an ops play becomes a unit the policy can plan over — the
P27 failure in representation form — at ~11% fewer decisions per game.

**Procedure.** From scratch to 560M on the flat-1.0 recipe, seed 44 vs E6-03-44; seed 43 to
confirm. Needs no build, so it launches first.

**Decision.** Adopt iff both seats beat the control by the panel rule. A repeated
non-recovering US collapse closes it pending the audit P23's status asks for (rewards, blunder
windows and GAE around composed actions) — not a third run.

## Bet 4 — The ops budget in the observation (owner-held)

**Proposal.** At an op-choice node the network does not see the Ops it is spending —
`PENDING_OPS_VALUE` is set at commit (`observation.cpp:348`); P23 stage 6 already proposes
bringing it back. **Cost:** a handful of global floats (the pending Ops value and what remains
of it), no per-entity vector. **Precedent:** the decision context, +92, the last observation
change that mattered.

**Stated here so it can be approved or declined before bet 3's arm is read**, since the two
compound: a merged action with a visible budget is what a human sees. **Not run without
approval.** If approved: one factor on top of the bet-3 recipe, seed 44 then 43.

## Bet 5 — Search-driven training from scratch (gated)

**Change.** Search targets throughout training — the AlphaZero-family regime, not a tail.
Two parts: (a) **a compiled searcher** — the tree in C++ over the batched engine, parallel
across the machine's cores (P13's one-driver work is the prerequisite; the Python tree is
~1,900× slower per decision); (b) the trainer hook from P15-X4b — CE toward the search policy
on a subsample of decisions, **search targets only, never search-driven data**, from scratch.

**Why, and why gated.** It paid +150–260 Elo on E3/E4 while the raw policy was weak, and shrank
to +7 pp at 256 sims on E5 because the leaves share the student's critic. So it runs only once
**bet 1 or bet 2 has landed** (a critic search can use) and the headroom is re-measured on the
best new model: honest 256-sim search vs its raw policy, 100 games per seat, **≥ +5 pp** to
proceed.

**Expected size if right:** the largest possible ceiling change, at the largest engineering
cost, in that order.

## Combining

When two bets land on both seeds, the next arm is their combination from scratch. They are
expected to compound: a bigger net with a better-taught critic; a merged view with a visible
ops budget; any of them as a stronger student for bet 5.

## Sequencing under the two-arm limit

| slot | arms | note |
|:---|:---|:---|
| now | **E6-03-43** (control, seed 2; P28 step 0) ‖ **bet 3**, seed 44 | neither needs a build |
| next | **bet 2**, seed 44 ‖ **bet 1** wider M2d, seed 44 | bet 2 is ~a day of model work |
| then | seed-43 replicates of whatever leads | |
| overnight | **bet 1** M3, once the mode split lands; **bet 4** on bet 3 if approved | |
| last | **bet 5** | after its gate |

About **30–40 GPU-hours** through the seed-43 replicates, before bet 5.

## Measure

P28's panel for everything; per seat by the panel rule; the goal probes (forced wins taken,
battlegrounds at turn 8, card disposal, own-DEFCON losses) on every adopted model; the critic
instruments first for bet 2. Seed variance is ~95–100 Elo between seeds, so nothing is adopted
from one seed, and effects under ~40 Elo are not run.

## Follow-ups

* Any bet adopted → P28's polish levers as its tail (LR schedule, EMA, SWA, cross-lineage
  pools).
* Bet 1 adopted → rerun bets 2–3 on the larger net.
* Bet 2 adopted → the ownership head becomes a per-country *value* probe that replaces P27's
  "nothing to train toward".
* Bet 3 adopted → the E4.1 view becomes the default action view; the human-corpus converter's
  merged-stream view (P23 stage 7) follows.
* Bet 5 adopted → search as the evaluation exploiter for approximate exploitability.

## Runs

* **Bet 3, E6-09-44** (2026-09-29): **not adopted.** No collapse under flat 1.0, so P23's collapses were the old bands. But it is ~235 Elo weaker than E6-03-44 in both seats, on snapshots and SWA (head to head 21% / 19%). The first lead is bet 4, since the merged op-choice node does not show the Ops being spent ([`../log/P29_bet3_merged_view.md`](../log/P29_bet3_merged_view.md)).
* **Bet 1, wider M2d, E6-10-44** (2026-09-30): **not adopted.** A 768 × 8 trunk (11.2M parameters, r = 0.665) is level with M2d at matched games on Elo, but behind it against the panel and on SWAs (45.4%), and clearly behind at matched wall-clock (38–45%). Capacity is not what holds the plateau; M3 (card↔card attention) is a separate mechanism and stays open ([`../log/P29_bet1_wider_m2d.md`](../log/P29_bet1_wider_m2d.md)).
* **Bet 2, code (2026-09-30):** `--aux-ownership` / `--aux-vp-margin`, with heads on the trunk and labels from each game's end (all 84 countries, mover's frame), trained on a 10% sample of positions in their own step. A CPU smoke run learned ownership from 29% to about 80% accuracy in 30k steps. Default weights 0.1 / 0.1, as the plan decided. Not yet run.
