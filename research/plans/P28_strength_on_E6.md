# P28 — Strength on E6: the measurement floor, and the tail levers

**Status:** step 0, step 1 and the first two polish levers done (2026-09-29); see *Runs*. The five structural bets that were this file's centre moved
to [**P29**](P29_big_bets_from_scratch.md) at the owner's direction; P28 keeps what every bet
needs to be judged (step 0), the zero-training checks, and the tail levers that run *after* a
bet lands.
**Gate:** none for step 0 and step 1; the polish levers wait for a P29 adoption.
**Needs approval:** none.

## What exists

| run | what | role here |
|:---|:---|:---|
| **E6-03-44** | the control recipe (M2d, flat rollout temperature 1.0, pool 0.3/12, `--block-lambda off`) from scratch to 560M on E6, no league | the matched control for every P29 bet; panel members |
| **E6-04-44** | league + game-result setup credit + setup entropy floor 0.3, from E6-03-44's 310M state | the E5-21 recipe replicated on a new seed and engine; its verdict decides which recipe the bets run on |

## The plateau, as measured

Training levels off at roughly 320–560M steps — about a million games — and the last 240M of an
800M run are 49.9% / 51.5% against its own 480–560M. Its structural causes and their levers are
P29's subject. What remains here is the part that is not structural: the learning rate is a
constant 3e-4 with no schedule, no weight averaging and a fixed batch; adjacent late snapshots
swing ±8 pp (±40 Elo) against the same opponent, and continuations past the peak dip. That is
a noise floor worth tens of Elo — real, cheap, and **not the priority**.

**One rule, shared with P29: judged at the plateau, never at 80M.** Arms are compared with
their control at 480/520/560M, pooled over the late snapshots, against the fixed panel below.

## Step 0 — the panel and the first verdict (CPU; a day)

A frozen reference panel, rated greedily, ≥200 games per seat per pairing, pinned by sha256:

* **E6-03-44** at three roles chosen on its fixed-reference trace, not by step count: a weak
  end (~80M, ~300–400 Elo below the plateau), a mid rung (~240M), and the **trace peak** in
  480–560M — never the final snapshot, since every lineage on record peaks then declines —
  plus its four late snapshots for the pooled sixteen-pairing comparisons.
* **E6-04-44**'s four late snapshots.
* **`heuristic_mcts:16`** — an out-of-lineage style whose leaf explicitly prices battleground
  control; vary `prior_temperature`, not sims, if a finer dial is wanted.
* **E6-03-43** joins when it exists (it is the first arm in P29's sequencing).

**Per-seat rule, revised.** A seat's bar is the *panel's* mean result in that seat against the
arm's control, not one model's self-play — the E5 rule read three unrelated arms with the same
+US/−USSR split because the reference lineage's USSR had specialised against its own weak US.

**First verdict.** E6-04-44 vs E6-03-44 at 480/520/560M by the revised rule, plus openings
(`setup_oracle.py`) and the goal probes (forced wins taken, battlegrounds at turn 8, card
disposal, own-DEFCON losses). Adopted → P29's bets run on E6-04's recipe and control against
it; not adopted → on E6-03's, and the league stays a setup-punisher for later.

## Step 1 — zero-training checks (an afternoon, CPU)

* **Weight soup.** Average E6-03-44's and E6-04-44's 480–560M snapshots (uniform SWA), rate
  each soup against the panel. A soup that beats its own best snapshot measures how much of
  the plateau is snapshot noise.
* **Search headroom on E6.** Honest search at 256 sims on the trace peak vs its raw policy,
  100 games per seat — the baseline for P29 bet 5's gate.
* **The plateau trace itself.** E6-03-44's Elo vs the panel by 40M, so the E6 plateau step and
  peak are numbers, not the E5 ones assumed.

## Polish — after a P29 bet lands, as its tail

Kept because they are real and cheap, demoted because they are tail levers:

* **an LR schedule from the plateau state** (3e-4 → 1e-4 → 3e-5 every 60M, or cosine over
  200M) vs the constant-LR continuation at matched steps; read the late-snapshot spread first;
* **EMA weights** (τ ≈ 10M steps) as the rated and pooled model;
* **buffer 128 → 256** at the plateau, only if the schedule moves;
* **cross-lineage pools** via `--league-dirs` (seed 44's arm carrying seed 43's snapshots and
  vice versa) and **`heuristic_mcts:16` in the pool** at 5–10% of games.

Decision for each: adopt iff the late spread halves or both seats improve against the control
by the panel rule, at Elo not worse. Expected size: +20–60 Elo combined.

## Not on this plan, measured small

Hidden information (the opponent's hand adds +0.3 points of explained variance), chance-aware
value targets (~7% of one-step variance), human BC/injection at 300 games, more setup work,
more league seeds.

## Measurement rule (owner, 2026-09-29)

Every arm, and every control, is measured twice:

* on its **snapshots**, the late block against the panel;
* on the **uniform soup of its last 80M** (`tools/scripts/weight_soup.py`, one snapshot every 10M).

Both readings are reported. Averaging does not enter training: `--ema-weights` stays off in every
recipe until the owner says otherwise.

## Runs

* **Step 0 (2026-09-29): E6-04-44 adopted.**
  * The panel is pinned in `data/reports/p28_panel.json`: E6-03-44 at 80M, 240M and the 550M peak,
    plus `heuristic_mcts:16`. The late block is 500/520/540/560M.
  * By the panel rule, E6-04 − E6-03 is US +2.7 ± 0.6 and USSR +1.5 ± 0.5: both seats. The
    self-play-bar rule had read the same pairings as US +9.6 / USSR +0.3.
  * Empty battlegrounds at turn 8: 0.51 for E6-04 against 1.5–1.7 for E6-03.
  * E6-03-44's trace levels off at about 320M, and has no end dip on E6.
  * Log: [`../log/P28_step0_panel_and_E6_04_verdict.md`](../log/P28_step0_panel_and_E6_04_verdict.md).
* **Step 0 replicated (2026-09-30):** E6-04-43 against E6-03-43 on seed 43. Panel US +0.1 / USSR +6.9 on snapshots and −1.1 / +7.1 on soups; 57% head to head. E6-04 is adopted on both seeds ([`../log/E6_04_43_seed43_replicate.md`](../log/E6_04_43_seed43_replicate.md)).
* **Step 1 (2026-09-29).**
  * Uniform soups of 480–560M beat every snapshot they average, by about 60–85 Elo in both seats.
    The E6-04-44 soup rates 1651 against its best snapshot's 1566, and beats that snapshot 61.7%
    head to head.
  * Search headroom at 256 simulations: +3.5 ± 3.5 on E6-04-44@520M, so P29 bet 5's gate fails on
    the adopted model; +11 on E6-03-44@550M.
  * Log: [`../log/P28_step1_soups_and_search.md`](../log/P28_step1_soups_and_search.md).
  * These steps needed code after all: `--lr-schedule`, `--ema-weights` and `weight_soup.py`, added in
    `27bfc6c`.
* **Polish, LR schedule (E6-06-44 ‖ E6-07-44, 2026-09-29): not adopted.**
  * Stepping the rate to 3e-5 did not shrink the late spread: SD 1.99 against 2.03 points.
  * Raw snapshots at the lower rate are +3.4 as USSR against the panel, but the soups reverse it.
    The constant-rate 680–760M soup (1662) beats the scheduled one 57.8%, and beats the 480–560M
    soup 53.7%.
  * Log: [`../log/P28_step2a_lr_schedule.md`](../log/P28_step2a_lr_schedule.md).
* **Polish, EMA weights (E6-08-44, 2026-09-29): not adopted into training.**
  * τ 10M EMA snapshots beat the matched raw ones by the panel rule (US +2.1 ± 0.6, USSR +7.3 ± 0.6)
    and halve the spread beyond noise.
  * They still lose to the 80M uniform soup (41–43%), and a soup of the EMA snapshots is level with
    a soup of the raw ones (48.0 ± 1.6).
  * The owner's rule since: measure every arm on its snapshots and on its 80M soup.
  * Log: [`../log/P28_step2b_ema_weights.md`](../log/P28_step2b_ema_weights.md).
