# P36 — the issues bank, and whether search distillation is limited by signal or by capacity (2026-10-19)

**Origin.** The running session's P36 proposal (its reply to the owner, 2026-10-19; not a file) grouped
the owner's issue list into causes and proposed an issues bank, expert iteration as the compounding
engine, rule-derived search candidates, planning inputs and a turn-level root, delayed-consequence
value targets, a decided-position tiebreak, and capacity from scratch. This file is the plan as
reviewed and decided by the owner: what stays, what changes, and the test that comes before any
generation of distillation is spent.

**Decisions (owner, 2026-10-19).**

* **Pure search before belief.** The honest keyed root is +180 over the raw network; the belief root is
  +35 over the honest root. The policy's input today determines the honest root's expected choice
  (uniform draws over the unseen cards given the observation), so that teacher is representable by a
  memoryless policy; the belief root's edge is not, until a belief input exists. The belief head
  (P35 3c) goes to the reserve; 3b stays a product gain.
* **Expert iteration is not launched as a generation.** Seven policy-side forms and one value-side form
  (R51) were null or harmful; the proposal did not say what would differ. Before any generation, a
  one-day offline test (below) says whether the limit is the targets or the student.
* **Capacity from scratch starts now** (P35 items 1 and 2), not after richer targets: a bigger student
  is a precondition for distillation, not a follow-up, and the runs are independent of this plan.
* **Dropped:** delayed-consequence value targets (exact endgames, region-scoring auxiliaries) -- P34
  closed the value track; the region swings were in D1's labels and bought nothing. To the reserve.
* **No code by the planning session; the running session builds and runs.**

## The issue groups (from the proposal, kept as the bank's structure)

1. **Options never tried**: coups at DEFCON 2 under Nuclear Subs; Wargames declined; events at the
   wrong moment; the choice inside Aldrich Ames. Self-reinforcing avoidance; the root's top-4 never
   holds a move the policy gives ~0.
2. **Delayed, conditional value**: Europe control; the opponent's possible scoring cards; holding a
   3-Ops card against an unknown Blockade.
3. **No plan for the turn's cards**: high-Ops early and scoring late; Five Year Plan before the last
   round; Five Year Plan with Formosan Resolution and NATO in hand.
4. **Drift in decided positions** (critic below 5% or above 95%): advantages near zero, won
   positions not converted (the trainer's decisive-win take rate ~76%).
5. **Opening quirks**: Yugoslavia 3 then Independent Reds; the setup locked at probability 1.

## `P36a` -- the issues bank (CPU; for the owner's review before it is used)

* **What.** For each group, positions with a verified right answer, built programmatically where the
  owner's examples allow (a Subs coup, a Europe trade, a Blockade hold, the Five Year Plan / NATO /
  Formosan hand, the Yugoslavia + Independent Reds opening) and scored for every checkpoint and
  searched player, the way `ai/eval/banks/blunders_E7.jsonl` is.
* **The verification rule, from P33c's collapse.** A bank's "right answer" is never the network's own
  greedy continuation on a multi-step plan -- that is how the leak bank certified its own judge's
  follow-up. Labels come from the exact solver where the game ends in the turn, the oracle root where
  the hand decides, long keyed paired playouts with the follow-up's bias reported against the exact
  slice, and **the owner's judgment** for the positions that are the point of the bank. Each row records
  which.
* **Deliverable for review.** The candidate position set with its proposed label source per row, before
  any labelling runs. The owner chooses which situations go in.
* **Use.** Every arm in this plan and in P35 is scored on it; self-play Elo cannot see a leak both
  seats share.

## `P36b` -- measure the teacher first (GPU, ~1 h): the pick's reproducibility

* **What.** The honest keyed root (k 4, `earlier`+own, `chance=keyed`) on A13+R48's SWA, 3,000 entry
  decisions from its self-play, searched **three times each** at 16, 64 and 128 keyed worlds; per
  world count: the share of decisions where it departs from the policy, the share of one run's
  departures that recur in another, the share with the same move, and the three-run agreement
  (`ai/eval/rollout_consistency.py` extended to world counts; at 16 shared-stream worlds the record
  has 62–64% / 57–58% / 188 of 432).
* **Why first.** This is the target's signal-to-noise. P34's reliability (0.41 at 16 keyed worlds) was of
  outcome means, the wrong quantity for a policy target; the pick's agreement is the right one and is
  unmeasured above 16 worlds. **The line:** the world count whose three-run agreement on departures
  reaches ~90% is the one the targets are built at; if 128 does not reach it, the test below still runs
  at 128 and reports the agreement as the ceiling.

## `P36c` -- targets and students: signal or capacity (offline, ~a day)

The search's own seed-to-seed agreement on held-out positions is the ceiling for any student's
held-out agreement with it. Three readings per cell separate the causes.

* **Targets.** The keyed root's improved policy (`softmax(logit + σ(completed Q))` over the candidates,
  R32's form) on 200k entry decisions of A13+R48's self-play, at 16, 64 and 128 keyed worlds (the
  128-world root is ~7k network rows a decision; the set is ~30 min on the GPU). Held out by game.
* **Students, trained to convergence** -- not the two epochs at 1e-4 of the earlier offline rounds, which
  underfit by design and never separated these cases: the current network fine-tuned on its policy
  path; the same shape from scratch; a 5–10× network (the deep or token shape of P35). Supervised on the
  policy heads, the value path frozen; under an hour each. Each student an A+R pair.
* **Three numbers per cell:** (1) the training-set fit -- the share of the teacher's departures that are
  the student's argmax on positions it trained on; (2) held-out agreement with the teacher; (3) the
  teacher's own seed-to-seed agreement on the same held-out positions (`P36b`).
* **Reading the cells.**
  - fit rises with world count at a fixed student → **the targets were the limit**;
  - the big student fits the training departures and the current one does not, at convergence →
    **capacity**;
  - no student fits the training set at convergence → the targets are not a consistent function of
    the input (noise, or hidden information) → **no student can help**;
  - every student fits the training set and none generalises → position-specific corrections with no
    rule (the depth log's diagnosis) → only the number of targets helps, the expert-iteration cost
    regime;
  - a student's held-out agreement at the teacher's self-agreement → that student is at the signal
    ceiling; capacity is pointless until the targets improve.
* **Games** only for cells that pass: the student greedy against A13+R48's SWA, and under the keyed
  root (the own-search gain should shrink if the ranking was absorbed), per seat.
* **Then, and only then,** the expert-iteration design: targets at the measured world count, the
  advantage form over the candidates (P35 item 5) rather than a cross-entropy toward the argmax, the
  student size the test chose, as one from-scratch programme.

## The remaining arms (after `P36a`–`P36c`; each scored on the bank)

* **Rule-derived candidates in the root** (a search change): coups where DEFCON allows them and always
  under Nuclear Subs; events that score at once or reach 20 VP; for each card, play now or hold. The
  k = 6 null does not apply (that was the policy's own ranks 5–6, not moves outside its support).
  Cautions: the critic has never valued a post-Subs coup, and the z2 rule may never clear on a
  candidate the critic misprices; the exploration half in training is a separate arm, and the fixes
  reach the network only through a distillation that `P36c` has shown can work.
* **A root over the turn's card order**: search the sequence of the hand's plays across the action
  rounds, not one decision at a time -- the most promising search-approach change in the proposal,
  with no precedent here. Design first; the bank's group-3 rows are its read.
* **Planning inputs.** The observation already carries the turn (`globals[6]`), the action round
  (`globals[7]`) and both hand sizes, so rounds left and surplus cards are a subtraction away; the
  expert review's two-to-three-float convenience block is the owner's call, low prior as a cause.
* **The decided-position tiebreak**: a small VP-margin or safety term where the critic is past 95% or
  below 5% (the plain VP shaping was level in P30); one 400M arm, judged on the bank's group-4 rows
  and the decisive-win take rate.
* **Capacity from scratch**: P35 items 1 and 2, launched now, the GPU being free after A18.

## Sequencing and parallelism

| now | GPU | CPU | owner |
|:---|:---|:---|:---|
| day 1 | `P36b` (1 h), then `P36c`'s targets (30 min), then P35's two from-scratch flagships launched in both slots | `P36a` candidate positions and label sources | reviews `P36a`'s set |
| day 1–2 | `P36c`'s student fits interleaved with the flagships' first windows (each < 1 h) | the turn-level root's design; the rule-candidate build | approves `P36a`; decides on the convenience block |
| after | the cells that pass `P36c` go to games; the tiebreak arm and rule candidates follow, scored on the bank | | |

`P36b` and `P36c` are on the critical path of any distillation; `P36a` is on the critical path of every
read. The flagships are independent of both and take the slots as soon as `P36b`/`P36c`'s GPU hour
is done.

## Decision rules

* `P36b`: targets are built at the smallest world count with ~90% three-run departure agreement;
  below that at 128, the agreement is reported as the ceiling and `P36c` runs anyway.
* `P36c`: the cell readings above decide the cause; a student goes to games only if its training fit
  is above ~90% of the teacher's departures and its held-out agreement is within 5 points of the
  teacher's self-agreement.
* Expert iteration is designed only after `P36c` names a student and a world count; never from the
  proposal's form.
* Every other arm: the per-seat rule raw and under the keyed root, plus the bank's rows for its group.

## Reserve (from this plan)

The belief input (P35 3c) pending pure search; delayed-consequence value targets; the convenience
planning block.

## Runs

(filled in while running; mirrors [`../runs.md`](../runs.md))
