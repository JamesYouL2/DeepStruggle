# P28 — Strength on E6: the measurement floor, and the tail levers

**Status:** proposed (2026-09-29). The five structural bets that were this file's centre moved
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

## Runs

(none yet)
