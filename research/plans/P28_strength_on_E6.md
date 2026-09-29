# P28 — Strength on E6: past the ~1M-game plateau

**Status:** in progress (2026-09-29): step 0 done, E6-04-44 adopted. Written from scratch for E6; the E5 record is cited only where
a mechanism was measured there and nothing on E6 contradicts it.
**Gate:** step 0 (the panel and the E6-04 verdict) before any arm is *judged*; steps 1–2 need no
new code; steps 3–5 are trainer/model changes; step 6 is engineering.
**Needs approval:** none of it touches `engine/` or the observation. Step 4 adds output heads
(model only). Step 5 is an architecture arm on the existing ladder flags.

## What exists

| run | what | role here |
|:---|:---|:---|
| **E6-03-44** | the control recipe (E5-11's: M2d, flat rollout temperature 1.0, pool 0.3/12, `--block-lambda off`) from scratch to 560M on the real E6, no league | the matched control for every arm below; the source of E6-04's 310M state; three panel members |
| **E6-04-44** | league (main + exploiter, `--league-frac 0.5`) + game-result setup credit + setup entropy floor 0.3, from E6-03-44's 310M state | the replication of E5-21's recipe on a new seed and engine; whether it is *adopted* is step 0's first verdict |

Both are seed 44. Nothing else on E6 is needed to start.

## The plateau, as measured (E5, expected to transfer; step 0 confirms on E6)

Training levels off at roughly 320–560M steps — about a million games — and the last 240M of an
800M run are 49.9% / 51.5% against its own 480–560M. Three things stand behind it, each with a
lever this plan tests:

1. **A step-size noise floor.** The learning rate is a constant 3e-4 with no schedule, no weight
   averaging and a fixed batch. Adjacent late snapshots swing ±8 pp (±40 Elo) against the same
   opponent, and continuations past the peak *dip*. Nothing has ever been scheduled or averaged.
2. **A critic that cannot price moves the policy cannot yet convert.** Half of takeable contested
   battlegrounds are missed, but the rollout oracle finds taking one worth −0.8 ± 0.6 pp under
   the policy's own continuation ([`../log/P27_ops_block_stage1.md`](../log/P27_ops_block_stage1.md));
   the critic's sign agrees with playouts 49–65% of the time where they differ by ≥10 points.
   The terminal-only signal cannot teach a move whose value needs follow-up competence the
   policy lacks — a self-play local optimum. The critic is also what search leaves and setup
   credit depend on.
3. **Capacity is untested at the asymptote.** M2d is 3.2M parameters, chosen at 80M matched
   steps with throughput as an axis; every larger or attention-bearing variant lost *at 80M*,
   which selects for fast early learning, not for where a run levels off. No larger net has a
   clean long run.

Measured small and **not on this plan**: hidden information (the opponent's hand adds +0.3
points of explained variance), chance-aware value targets (~7% of one-step variance), human
BC/injection at 300 games, exploration temperature (flat 1.0 adopted), further setup work, and
more league seeds — the league keeps weak setups punished but did not lift the plateau on
either seed it ran on.

## One rule that changes for this plan

**Everything is judged at the plateau, never at 80M.** An arm is compared with its control at
480/520/560M (pooled over the late snapshots), against the fixed panel below. The ladder's
80M-matched-steps screen stays for *throughput* questions only.

## Steps

### 0 — The panel and the first verdict (CPU; a day)

A frozen reference panel, rated greedily, ≥200 games per seat per pairing, pinned by sha256:

* **E6-03-44** at three roles chosen on its fixed-reference trace, not by step count: a weak
  end (~80M, ~300–400 Elo below the plateau), a mid rung (~240M), and the **trace peak** in
  480–560M — never the final snapshot, since every lineage on record peaks then declines —
  plus its four late snapshots for the pooled sixteen-pairing comparisons.
* **E6-04-44**'s four late snapshots.
* **`heuristic_mcts:16`** — an out-of-lineage style whose leaf explicitly prices battleground
  control; vary `prior_temperature`, not sims, if a finer dial is wanted.
* **E6-03-43** joins when it exists (step 2 launches it).

**Per-seat rule, revised.** A seat's bar is the *panel's* mean result in that seat against the
arm's control, not one model's self-play — the E5 rule read three unrelated arms with the same
+US/−USSR split because the reference lineage's USSR had specialised against its own weak US.

**First verdict.** E6-04-44 vs E6-03-44 at 480/520/560M by the revised rule, plus openings
(`setup_oracle.py`) and the goal probes (forced wins taken, battlegrounds at turn 8, card
disposal, own-DEFCON losses). Adopted → E6-04 is the recipe steps 2–5 build on and control
against; not adopted → E6-03 is, and the league stays a setup-punisher for later.

### 1 — Zero-training checks (an afternoon)

* **Weight soup.** Average E6-03-44's and E6-04-44's 480–560M snapshots (uniform SWA), rate
  each soup against the panel. A soup that beats its own best snapshot says the plateau is
  partly snapshot noise and makes step 2's EMA the default rated model.
* **Search headroom on E6.** Honest search at 256 sims on the trace peak vs its raw policy,
  100 games per seat. This is the gate for step 6: below +5 pp there is no teacher.
* **The plateau trace itself.** E6-03-44's Elo vs the panel by 40M, so the E6 plateau step
  and peak are numbers, not the E5 ones assumed.

### 2 — Optimizer levers at the plateau (flags; ~1.5h per arm)

Two arms under the two-arm limit, plus the second control seed:

* **2a `--lr-schedule`:** from the recipe's 560M end state, +200M with the learning rate
  stepped 3e-4 → 1e-4 → 3e-5 every 60M (cosine as the alternative cell), against the
  constant-LR continuation at matched steps. Read: late-snapshot spread first (it should
  shrink), panel Elo second.
* **2b `--ema-weights`:** an exponential average of the weights (τ ≈ 10M steps) as the *rated*
  and *pooled* model, the raw weights still training. Free if 1's soup helps.
* **E6-03-43**: the control recipe on seed 43 to 560M — the second panel lineage and the seed
  replicate every adoption below needs.
* 2c, only if 2a moves: buffer 128 → 256 at the plateau (the same mechanism at higher cost).

**Decision.** Adopt a schedule if the late spread halves and panel Elo is not worse; it becomes
the default tail of every long run. Expected size: +20–60 Elo. If nothing moves, the plateau is
not the step size, and steps 3–5 carry the plan.

### 3 — Opponents from outside the lineage (flags; ~3h per arm)

The pool and the exploiters descend from the same weights and cannot punish a weakness they
share. `--league-dirs` already loads other runs' snapshots:

* **3a cross-lineage pool:** seed 44's arm carries seed 43's snapshots in its pool and vice
  versa, at the league fraction.
* **3b `heuristic_mcts:16` in the pool** at 5–10% of games (CPU cost measured first; it is
  1.6 s per game at 96 sims, far less at 16) — an opponent that punishes empty battlegrounds
  by construction.

From the plateau state, +200M each, judged as in step 2. Adopt whichever beats its control in
both seats; both together as a follow-up cell.

### 4 — Auxiliary targets for the critic (model change; a day + ~3.5h per arm)

KataGo's lever, aimed at mechanism 2: two small heads on the trunk, off by default, trained
from the terminal state the buffer already holds — **final control of each country** (84-way
multi-label; the 29 battlegrounds at minimum) and **final VP margin**. No observation change.
The trunk learns that a battleground is worth holding *before* the policy can convert it,
which is also what search leaves need.

From scratch to 560M on the step-0 recipe, seed 44 against E6-03/04-44, then seed 43. Read,
in order: critic sign agreement with playouts (the E5-15 instrument) and the P27 take-rate
before Elo. Decide before running: loss weights (first cell 0.1 / 0.1), whether the ownership
target is all countries or battlegrounds only.

### 5 — Capacity at the asymptote (overnight; ~6h per arm)

The test the ladder never ran: **wider/deeper M2d** (`--ladder-hidden-dim 768
--ladder-res-blocks 8`) and **M3** (card↔card attention, once its board/card split is
unblocked) against M2d, from scratch to **560M matched games** and at matched wallclock, seed
44 then 43. A bigger net that plateaus later and higher makes steps 2–4 multiplicative; one
that does not vindicates 3.2M parameters and closes the capacity question.

### 6 — A compiled searcher, then expert iteration (engineering; after 4–5)

Gated by step 1's headroom re-measured on the best step-4/5 model: search distillation paid
+150–260 Elo on E3/E4 when the raw policy was weak and shrank to +7 pp on E5 because the
leaves share the student's critic. If the gate passes, move the tree to C++ over the batched
engine (P13's one-driver work is the prerequisite; today the Python tree is ~1,900× slower per
decision) and run the distillation arm with search targets only, never search-driven data.

## Sequencing under the two-arm limit

| slot | arms |
|:---|:---|
| now | E6-03-43 (control seed 2) ‖ 2a LR schedule from E6-03-44@560M |
| next | 2a on E6-04-44 (or 2b) ‖ 3a/3b from the plateau |
| after step 4's build | 4 seed 44 ‖ 4 seed 43 |
| overnight | 5, one configuration per night |
| last | 6 |

Step 1 runs on the CPU alongside the first slot. Roughly **35–45 GPU-hours** through step 5.

## Measure

The panel (step 0) for everything; late-snapshot spread as the step-2 instrument; critic sign
agreement and P27 take-rate as the step-4 instruments; the goal probes on every adopted model.
Seed variance is ~95–100 Elo between seeds, so every adoption needs the seed-43 replicate, and
effects under ~40 Elo are not run.

## Decision rules

* 0: E6-04 adopted iff it beats E6-03 in both seats by the panel rule at the late block.
* 2: adopt a schedule iff late spread halves at Elo not worse; EMA adopted iff the soup or the
  EMA beats the best raw snapshot.
* 3: adopt iff both seats improve against the control by the panel rule.
* 4: adopt iff critic sign agreement rises *and* Elo is not worse; a probe-only gain is logged
  as a critic result and kept for step 6.
* 5: adopt a larger net only at matched wallclock; a matched-games-only win is recorded as
  "capacity-bound, throughput-limited" and routes to P18/P26.
* 6: run only if the gate passes on the current best model.

## Follow-ups

* 2 adopted → re-run the 800M question on the scheduled recipe: does the tail keep paying?
* 3b adopted → a stronger out-of-lineage opponent (`heuristic_mcts` with a flatter prior) as
  the dial.
* 4 adopted → the ownership head is the natural target for a per-country *value* probe that
  replaces the P27 oracle's "nothing to train toward".
* 5 adopted → rerun 2–4 on the larger net; they compound.

## Runs

* **Step 0 (2026-09-29): E6-04-44 adopted.** Panel pinned in `data/reports/p28_panel.json` (E6-03-44 @80M / @240M / @550M peak, `heuristic_mcts:16`); late block 500/520/540/560M. By the panel rule E6-04 − E6-03 is US +2.7 ± 0.6, USSR +1.5 ± 0.5 (both seats); the self-play-bar rule had read the same pairings as US +9.6 / USSR +0.3. Empty battlegrounds at turn 8 0.51 vs 1.5–1.7. E6-03-44's trace levels off at ~320M and has no end dip on E6 ([`../log/P28_step0_panel_and_E6_04_verdict.md`](../log/P28_step0_panel_and_E6_04_verdict.md)).
* **Step 1 (2026-09-29).** Uniform soups of 480–560M beat every snapshot they average, by about 60–85 Elo in both seats. The E6-04-44 soup is 1651 against its best snapshot's 1566, and beats that snapshot 61.7% head to head. Weight averaging therefore goes into step 2b (E6-08-44).
  Search headroom at 256 simulations: +3.5 ± 3.5 on E6-04-44@520M (step 6's gate fails on the adopted model) and +11 on E6-03-44@550M ([`../log/P28_step1_soups_and_search.md`](../log/P28_step1_soups_and_search.md)).
  **Correction to the plan:** steps 1–2 did need code. `--lr-schedule`, `--ema-weights` and `weight_soup.py` were added in `27bfc6c`.
* **Step 2a (2026-09-29): LR schedule not adopted.** Stepping to 3e-5 did not shrink the late spread (SD 1.99 against 2.03 points). Raw snapshots at the low rate are +3.4 as USSR against the panel, but soups reverse it: the constant-rate 680–760M soup (1662) beats the scheduled one 57.8%, and beats the 480–560M soup 53.7%. Constant rate plus averaging is the path, and 2b (E6-08-44) tests averaging in training ([`../log/P28_step2a_lr_schedule.md`](../log/P28_step2a_lr_schedule.md)).
* **Step 2b (2026-09-29): EMA beats raw snapshots, but the window must be wide. Not adopted into training:** the owner's rule is to measure every arm on its snapshots and on its 80M soup. τ 10M EMA snapshots beat the matched raw ones by the panel rule (US +2.1 ± 0.6, USSR +7.3 ± 0.6) and halve the spread beyond noise (0.89 against 1.69). They still lose to the 80M uniform soup (41–43%), and a soup of the EMA snapshots is level with a soup of the raw ones (48.0 ± 1.6). The rated model is a soup of the last ~80M, or EMA with τ ≈ 40M ([`../log/P28_step2b_ema_weights.md`](../log/P28_step2b_ema_weights.md)).
