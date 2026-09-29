# P28 — Strength on E6: big bets from scratch, not a squeeze at the tail

**Status:** proposed (2026-09-29; restructured the same day at the owner's direction — *focus on
potentially very large improvements from scratch rather than 100 more Elo at the end*).
**Gate:** step 0 (the panel) before any arm is *judged*. Each bet is a from-scratch arm on the
control recipe, judged at the plateau against E6-03-44.
**Needs approval:** bet 4 is an observation proposal (owner-held; stated with its cost in floats
and waits). Nothing else touches `engine/` or the observation. Bets 1–3 are model, trainer and
action-view changes on flags that exist or are small.

## What exists

| run | what | role here |
|:---|:---|:---|
| **E6-03-44** | the control recipe (M2d, flat rollout temperature 1.0, pool 0.3/12, `--block-lambda off`) from scratch to 560M on E6, no league | the matched control for every bet; panel members |
| **E6-04-44** | league + game-result setup credit + setup entropy floor 0.3, from E6-03-44's 310M state | the E5-21 recipe replicated on a new seed and engine; its verdict (step 0) decides which recipe the bets run on |

Both are seed 44. From-scratch arms on M2d run ~50k steps/s, so **560M is ~3h solo** — a big bet
costs one evening per seed. That is what makes this ordering affordable.

## Why big bets, and which

Every large gain in this project has been structural, not a tuning: the country per-entity head
(+282), the positional board path (+109), the decision context in the observation (+92), the
rollout temperature (+135/+170), the late-E3 bundle over an MLP (+447). Tuning levers have
returned tens of Elo at best. And the plateau's measured causes are structural too:

* the critic cannot price a move the policy cannot yet convert (P27's rollout oracle: taking a
  contested battleground is worth −0.8 ± 0.6 pp under the policy's own continuation; sign
  agreement with playouts 49–65%) — a self-play local optimum that a terminal-only signal
  cannot leave;
* hand composition is unrepresentable — there is no card↔card path, so "I hold no scoring card,
  so this event is safe" reaches the policy only through pooled statistics
  ([`../findings/training/forward_pass_trace.md`](../findings/training/forward_pass_trace.md));
* a 3.2M-parameter net whose asymptote was only ever judged at 80M matched steps, a screen that
  selects for fast early learning;
* no plan over a few micro-steps inside an ops play (P27), with the one representation that
  makes an ops play a single decision (E4.1, P23) parked after collapses that — this is the
  confound — all ran under the old sharpening rollout bands, before flat 1.0 was adopted.

**One rule for all of it: judged at the plateau, never at 80M.** An arm is compared with its
control at 480/520/560M, pooled over the late snapshots, against the fixed panel. A bet that is
behind at 80M and ahead at 560M is a win; the 80M screen is kept for throughput questions only.

## Step 0 — the measurement floor (CPU; a day)

A frozen panel, rated greedily, ≥200 games per seat per pairing, pinned by sha256: E6-03-44 at a
weak rung (~80M), a mid rung (~240M) and its **trace peak** in 480–560M (never the final
snapshot; every lineage on record peaks then declines), plus its four late snapshots; E6-04-44's
four late snapshots; `heuristic_mcts:16` as an out-of-lineage style; E6-03-43 when it exists.
A seat's bar is the *panel's* mean result in that seat against the control, not one model's
self-play (the E5 rule mis-read three arms through one lineage's seat profile).

**First verdict:** E6-04-44 vs E6-03-44 at the late block by that rule, plus openings and the
goal probes. Adopted → the bets run on E6-04's recipe and control against it; not adopted → on
E6-03's.

## The bets — each from scratch to 560M, seed 44 first, seed 43 to confirm

### Bet 1 — Architecture at the asymptote: card↔card attention and a wider trunk

The test the ladder never ran. Two arms: **M3** (card↔card self-attention over the 110 card
tokens, the rung blocked on the `board_mode`/`card_mode` split, which is the first engineering
item here) and **a wider/deeper M2d** (`--ladder-hidden-dim 768 --ladder-res-blocks 8`).
Judged at 560M matched games *and* at matched wallclock. **Expected size if right:** the
largest of any bet — the hand-composition gap is a whole class of decisions (what to hold,
what to space, when an event is safe) the current net can only approximate. **Kill:** behind
the control at 560M on both seeds at matched games → capacity is not the plateau's cause.

### Bet 2 — Auxiliary targets: final ownership and VP margin

KataGo's lever, aimed squarely at "the critic cannot price what the policy cannot convert".
Two heads on the trunk, off by default, trained from the terminal state the buffer already
holds: **who controls each country at game end** (84-way multi-label; battlegrounds at
minimum) and **final VP margin**. No observation change. The trunk learns that West Germany is
worth holding *before* the policy can convert it, which is also what search leaves need.
**Read first:** critic sign agreement with playouts and P27's take-rate, then Elo. **Expected
size:** KataGo-class sample efficiency — plausibly a plateau shift, since it supplies exactly
the signal the self-play outcome withholds. Decide before running: loss weights (0.1 / 0.1
first), all countries vs battlegrounds only.

### Bet 3 — The merged-influence action view (E4.1), re-run on the fixed recipe

P23 built "influence, first point in X" as one decision (composed of the two E4 steps, bit-
identical to E4 when off) and parked it after the US seat collapsed in 6 of 6 from-scratch
runs. Every one of those ran under the old rollout bands (0.15 / 0.50 / 0.10 / 0.35), which
flat 1.0 has since replaced and which were themselves collapse-prone. The cause was never
found; the confound was never named. Re-run E4.1 from scratch on the flat-1.0 recipe, seed 44
vs E6-03-44. **Expected size if right:** ~11% fewer decisions per game and an ops play as a
unit the policy can plan over — the P27 failure in representation form. **Kill:** the US seat
collapses again and does not recover → the audit P23's status asks for, not a third run.

### Bet 4 — The ops budget in the observation (owner-held; proposal, then wait)

At an op-choice node the network does not see the Ops it is spending (`PENDING_OPS_VALUE` is
set at commit; `observation.cpp:348`). P23 stage 6 already proposes bringing it back. Cost:
a handful of global floats, no per-entity vector. Precedent: the decision context was +92, the
last observation change that mattered. Stated here so it can be approved or declined before
the bet-3 arm runs, since the two compound. **Not run without approval.**

### Bet 5 — Search-driven training from scratch (engineering; gated)

The regime change: search targets throughout training, not a tail. It paid +150–260 Elo on
E3/E4 while the raw policy was weak and shrank to +7 pp on E5 because the leaves share the
student's critic — so it is gated on **bet 2 or bet 1 landing** (a critic search can use) and
on a re-measured headroom (256-sim honest search on the best new model ≥ +5 pp). Then: move
the tree to C++ over the batched engine (P13's one-driver work is the prerequisite; the Python
tree is ~1,900× slower per decision), and run the distillation arm from scratch — search
targets only, never search-driven data. This is the largest engineering item and the largest
possible ceiling change, in that order.

### Combining

When two bets land on both seeds, the next arm is their combination from scratch; they are
expected to compound (a bigger net with a better-taught critic; a merged view with a visible
ops budget).

## Sequencing under the two-arm limit

| slot | arms | note |
|:---|:---|:---|
| now | **E6-03-43** (control, seed 2) ‖ **bet 3** E4.1 from scratch, seed 44 | bet 3 needs no build; the control seed is needed by every verdict |
| next | **bet 2** aux heads, seed 44 ‖ **bet 1** wider M2d, seed 44 | bet 2 is ~a day of model work; M3 waits for the mode split |
| then | seed-43 replicates of whatever leads | |
| overnight | **bet 1** M3 once unblocked; bet 4 if approved, on top of bet 3 | |
| last | **bet 5** | after the gate |

Step 0 runs on the CPU alongside the first slot. About **30–40 GPU-hours** through the seed-43
replicates, before bet 5.

## Polish — after a bet lands, not before

Kept because they are real and cheap, demoted because they are tail levers: an LR schedule from
the plateau state (constant 3e-4, never scheduled; late snapshots swing ±40 Elo), EMA weights
as the rated model, a weight soup of late snapshots (zero training — worth one afternoon at any
point as a *measurement* of snapshot noise), cross-lineage pools via `--league-dirs`, and
`heuristic_mcts:16` in the pool. Run them on the recipe that wins above, as its final tail.

## Not on this plan, measured small

Hidden information (the opponent's hand adds +0.3 points of explained variance), chance-aware
value targets (~7% of one-step variance), human BC/injection at 300 games, more setup work,
more league seeds (the league keeps setups punished; it did not lift the plateau on either
seed).

## Measure

The panel for everything; per-seat by the panel rule; the goal probes (forced wins taken,
battlegrounds at turn 8, card disposal, own-DEFCON losses) on every adopted model; for bet 2
the critic instruments first. Seed variance is ~95–100 Elo, so nothing is adopted from one
seed, and effects under ~40 Elo are not run.

## Decision rules

* 0: E6-04 adopted iff it beats E6-03 in both seats by the panel rule at the late block.
* 1: adopt at matched wallclock; a matched-games-only win is "capacity-bound, throughput-
  limited" and routes to P18/P26 before adoption.
* 2: adopt iff critic sign agreement rises *and* Elo is not worse on both seeds.
* 3: adopt iff both seats beat the control by the panel rule; a repeated non-recovering US
  collapse closes it pending the P23 audit.
* 4: owner's call; if approved, one-factor on top of the bet-3 recipe.
* 5: run only if the headroom gate passes on the best model from 1–4.

## Follow-ups

* Any bet adopted → the polish levers, as its tail.
* Bet 1 adopted → rerun bets 2–3 on the larger net.
* Bet 2 adopted → the ownership head becomes a per-country value probe that replaces P27's
  "nothing to train toward".
* Bet 5 adopted → search as the evaluation exploiter for approximate exploitability.

## Runs

(none yet)
