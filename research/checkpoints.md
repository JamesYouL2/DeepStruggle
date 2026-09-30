# Checkpoint catalogue — what exists on disk, and what it is worth

One row per rated checkpoint, with its measured strength and the field that measured it. The arms
themselves are [`runs.md`](runs.md); this file answers the narrower "which model do I load, and how
good is it".

**Pre-E6 checkpoints archived 2026-09-29** (owner). Everything trained before the real E6 now lives under `/workspace/data/archive/`, not `data/checkpoints/`:

| archive | contents |
|:---|:---|
| `E4_ladder/checkpoints/` | E4 and E4.1, incl. the aborted and lambda-0.99 runs and `E4_1_warmup.pt` |
| `E5_ladder/checkpoints/` | E5, with the E5 league dirs in `E5_ladder/league/` (links rewritten) |
| `E6_intermediate/checkpoints/` | E6-01 and the first E6-02, on the abandoned era-fix-only engine |

The snapshot resolver (`data/logs/p25/resolve.py`) searches the archives, so `E5-21-43@560M` still resolves. `data/checkpoints/` holds only real-E6 runs, which is also all the workbench lists.

**Ladder reset 2026-09-19.** Pre-P17 checkpoints cannot be loaded on this engine at all — their
policy head is 212 wide against the current 220, and `check_checkpoint_layout` refuses them by
width rather than letting them misread. They are in
`/workspace/data/archive/E3_ladder/checkpoints/` with their ratings in
[`archive/E3_ladder/checkpoints.md`](archive/E3_ladder/checkpoints.md), and none of those ratings
transfers.

## How to read a number here

A win rate is only meaningful with its opponent, its game count and its side split. Until E4 has a
frozen anchor, the only cross-engine-comparable opponents are the rule-based bots, so every row
below is against `HeuristicBot` at 50 games a side.

## Best models (owner, 2026-09-29)

Three designations, on the real E6 engine (the model soup added 2026-09-30). They supersede E5-21-43@560M, which is kept below as the
E5-era record. Ratings are from the comprehensive E6 tournament (the soup's from its own field), anchored at HeuristicBot = 1500
([`log/E6_comprehensive_2026-09-29.md`](log/E6_comprehensive_2026-09-29.md)).

| designation | model | file | sha256 | Elo |
|:---|:---|:---|:---|---:|
| **best raw model** (a single training snapshot) | **E6-07-44@700M** | `/workspace/data/checkpoints/E6-07-44_20260929_121304/snapshot_700055552steps.pt` | `34017ae130fe…d9d20a` | 2492 |
| **best SWA model** (uniform average of the last 80M) | **E6-06-44, 680–760M SWA** | `/workspace/data/checkpoints/E6-06-44_20260929_121244/swa_680-760M.pt` | `6a3be70cb503…78b0ea` | **2560** |
| **best model soup** (average of branches of one trained state; 2026-09-30) | **E6-06/07/08-44 at 760M** | `/workspace/data/checkpoints/_soups/shared_start_E6-06+07+08_760M.pt` | `fa21df467d80…bf73e6` | 1775 in its own field, **+45 over the best SWA** there (1730) |

* **Best raw, E6-07-44@700M.**
  * It is the top snapshot of raw weights in the tournament.
  * The EMA snapshots of E6-08-44 rate higher (2503–2517), but they are weight averages, not raw
    snapshots.
  * It was trained at a learning rate of 3e-4 until 620M, 1e-4 until 680M, then 3e-5.
  * Only four late snapshots per arm were rated, so an unrated neighbour (690M, 710M, …) could be
    level with it.
* **Best SWA, E6-06-44 680–760M.**
  * It is the top model in the tournament.
  * It scores 69.4% against E5-21-43@560M and 53.1% against the adopted E6-04-44 SWA.
  * E6-08-44's SWA (2555) is level with it head to head (49.4 / 50.6). E6-06's is preferred for
    its simpler lineage: no averaging in training, per the owner's rule.
* **Best model soup, E6-06/07/08-44 at 760M** ([`log/model_soups_2026-09-30.md`](log/model_soups_2026-09-30.md)).
  * The uniform average of the three 760M snapshots of the continuations of E6-04-44@560M:
    E6-06-44 (constant rate), E6-07-44 (stepped rate) and E6-08-44 (EMA snapshots).
  * It is the strongest model measured. It was rated in a separate field, where it scores 1775
    against 1730 for the best SWA, beats its ingredients 65–70%, and is about 4–5 points better in
    both seats against the panel.
  * Souping the three 680–760M SWAs instead gives a level model (1773, 50.0% head to head).
  * It is an evaluation device like the SWA: no averaging enters training.
* **All three come from seed 44 only.** Each is a bare state dict in the normal (E4) action view, so
  every tool loads them.

### Their lineage, and how to reproduce it

Three phases, all through `tools/train.py`, on E6 (`c1df6f5` + `c248bfb`), guarded by
`tools/scripts/check_engine_fresh.sh`, with the environment
`PYTHONPATH=.:build/release TRITON_CACHE_DIR=.triton_cache`. The shared flags are:

```bash
M2D="--arch ladder --ladder-input-mode grouped --ladder-aggregation flatten --ladder-entity-dim 16 \
  --ladder-entity-proj-dim 256 --ladder-hidden-dim 480 --ladder-res-blocks 4 --drop-static \
  --per-entity-heads 64 --ladder-head-context --ladder-head-static --ladder-head-entities country"
COMMON="--reward-scheme blunder_aware --opponent-frac 0.3 --opponent-self-pool --opponent-pool-size 12 \
  --eval-opponents heuristic random --ladder-head-center --block-lambda off"
```

**1. E6-03-44, the plain recipe from scratch.** It feeds phase 2 through
`resume_310050816steps.pt`.

```bash
.venv/bin/python tools/train.py $M2D $COMMON --rollout-temps 1.0 1.0 1.0 1.0 --seed 44 \
  --train-steps 560000000 --run-name E6-03-44 --description "..."
```

**2. E6-04-44 + E6-05-44-g, the league from 310M to 560M.** This is E5-21's `league.py` command
(below) with `E6-04-44` / `E6-05-44` as the names, `--seed 44`,
`--main-resume <E6-03-44 dir>/resume_310050816steps.pt` and
`--league-dir /workspace/data/league/E6-04-44`. The main agent's extra flags are
`--setup-mc-credit --setup-entropy-floor 0.3 --resume-every-steps 10000000`.

**3. E6-06-44 / E6-07-44, continued from E6-04-44's 560M end state to 760M.** The published
E6-05 exploiters form a static league pool; there is no exploiter training.

```bash
MAIN="--league-dirs /workspace/data/league/E6-04-44 --league-pool-size 4 --league-frac 0.5 \
  --setup-mc-credit --setup-entropy-floor 0.3 --resume-every-steps 10000000"
.venv/bin/python tools/train.py $M2D $COMMON $MAIN --seed 44 --train-steps 760000000 \
  --resume <E6-04-44 dir>/resume_state.pt --run-name E6-06-44 --description "..."
#   E6-07-44: the same plus --lr-schedule step --lr-schedule-every 60000000 --lr-schedule-values 1e-4 3e-5
```

Then:
* **Best raw:** E6-07-44's `snapshot_700…steps.pt`.
* **Best SWA:** `tools/scripts/average_weights.py --snapshots <E6-06-44's 680, 690, …, 760M snapshots>
  --output swa_680-760M.pt`.
* **Best model soup:** E6-08-44 is E6-06-44 plus `--ema-weights 10000000`. Then
  `tools/scripts/average_weights.py --snapshots <E6-06-44, E6-07-44 and E6-08-44's 760M snapshots>
  --output shared_start_E6-06+07+08_760M.pt`.

Check each phase with `tools/scripts/launch_flags.py <original dir> --diff <replicate dir>`. As
with E5-21, a replicate reproduces the recipe, not the weights.

## E6 models (2026-09-29): the comprehensive tournament

Every E6 arm on its late snapshots and on its last-80M SWA, in one field anchored at
HeuristicBot = 1500 ([`log/E6_comprehensive_2026-09-29.md`](log/E6_comprehensive_2026-09-29.md)):

| model | Elo | vs E5-21-43@560M |
|:---|---:|---:|
| **`E6-06-44_*/swa_680-760M.pt`** | **2560** | 69.4% |
| `E6-08-44_*/swa_680-760M.pt` | 2555 | 69.1% |
| `E6-04-44_*/swa_480-560M.pt` | 2541 | 64.8% |
| E6-08-44 snapshots 700–760M (mean) | 2508 | — |
| E6-06-44 snapshots 700–760M (mean) | 2467 | — |
| E5-21-43@560M (E5 best, on E6) | 2441 | — |
| E6-04-44 snapshots 500–560M (mean) | 2434 | — |
| E6-03-44 / E6-03-43 snapshots 500–560M (mean) | 2397 / 2393 | — |


## E5-era best model: E5-21-43@560M (owner, 2026-09-28; superseded 2026-09-29)

`/workspace/data/archive/E5_ladder/checkpoints/E5-21-43_20260928_165156/snapshot_560005120steps.pt` (archived 2026-09-29)
(sha256 `f3213b98e0ba382930a41caadb58ce22c30fae2c14342150a7510cf36775d7c1`, 12.8 MB).

What it is worth ([`log/E5_21_setup_credit_plus_league.md`](log/E5_21_setup_credit_plus_league.md)):

* **Elo 2344**, the highest in its round robin (`data/reports/e5_21_rr_43.md`).
* **Against E5-11-43**, the previous best, over the late block: 54.6% overall.
  * Per seat against E5-11's self-play bars: US +11.8 ± 2.6, USSR −0.8 ± 2.6. That is accepted
    under the owner's per-seat rule as amended on 2026-09-29: above the bar in one seat, neutral in
    the other.
  * Replicated on seed 44 and E6 by E6-04-44: US +10.0 ± 1.1, USSR −0.4 ± 1.1
    ([`log/E6_04_league_seed44.md`](log/E6_04_league_seed44.md)).
* **Past 560M the same run does not improve.** The 720M continuation plateaus and then dips to
  2267.
* **Openings:**
  * US: West Germany 3, France 3, Italy 2, Iran 1.
  * USSR: Poland 3, Hungary 3.
  * Neither side has an alternative more than 2.2 ± 1.5 points better.
* **Self-play is balanced.** Greedy self-play gives US 49.6% over 500 deals.
* **One seed only (43).**
* **Trained on E5**, which had both bugs E6 fixed: the discard came back at turns 4 and 8, and Kitchen Debates spent
  for Operations stayed in the US hand. Its US replays Kitchen Debates in some games
  ([`findings/engine/engine_revisions.md`](findings/engine/engine_revisions.md), *The E5 → E6 boundary*).

### Replication recipe

Two phases, both through `tools/train.py`. The second phase is driven by the P24 league driver,
`tools/scripts/league.py`.

**Code.** Run everything at the merge `1b98bf0` or later. The phases originally ran at `f478ba5`
(E5-11) and `0ceb13e` (E5-21). Between those commits and `1b98bf0`, the only training-path
changes are these, and none of them changes a run that does not ask for it:

* flags that are off by default (setup entropy floor, game-result setup credit, forced
  opening);
* the run-name check in `league.py`;
* a display-only line in `bindings/state_json.cpp`.

Guard both phases with `tools/scripts/check_engine_fresh.sh`, with the environment
`PYTHONPATH=.:build/release TRITON_CACHE_DIR=.triton_cache`.

**Phase 1: E5-11-43 from scratch.** This is the flat-temperature baseline. Only its first 310M
steps are used.

```bash
M2D="--arch ladder --ladder-input-mode grouped --ladder-aggregation flatten --ladder-entity-dim 16 \
  --ladder-entity-proj-dim 256 --ladder-hidden-dim 480 --ladder-res-blocks 4 --drop-static \
  --per-entity-heads 64 --ladder-head-context --ladder-head-static --ladder-head-entities country"
.venv/bin/python tools/train.py $M2D \
  --reward-scheme blunder_aware --opponent-frac 0.3 --opponent-self-pool --opponent-pool-size 12 \
  --eval-opponents heuristic random --ladder-head-center --block-lambda off \
  --train-steps 800000000 --rollout-temps 1.0 1.0 1.0 1.0 --seed 43 --run-name E5-11-43 \
  --description "E5-11-43: ..."
```

Notes on phase 1:

* The step budget is only a stopping point. Nothing is scheduled on it: the learning rate is
  constant, and the curriculum applies only to the `curriculum` reward. A replicate can therefore
  stop once `resume_310050816steps.pt` exists; tagged resume states are written every 50M.
* The original resume state has sha256 `51a31b42…02ea7`.
* The startup banner must read `[opponent pool] self ..., frac=0.3, capacity=12`.

**Phase 2: E5-21-43, from that resume state to 560M.** The main agent adds game-result setup
credit and the setup entropy floor. It trains against a main exploiter (E5-22-43-g) whose
published snapshots join its pool.

```bash
COMMON="$M2D --reward-scheme blunder_aware --eval-opponents heuristic random --ladder-head-center"
.venv/bin/python tools/scripts/league.py \
  --main-name E5-21-43 --main-resume <E5-11-43 dir>/resume_310050816steps.pt \
  --main-steps 560000000 --seed 43 \
  --exploiter-name E5-22-43 --exploiter-reset main-latest --exploiter-start-steps 0 \
  --exploiter-gen-steps 50000000 --exploiter-target frozen \
  --league-dir /workspace/data/league/E5-21-43 --log-dir /workspace/data/logs/league \
  --main-league-frac 0.5 --main-league-size 4 \
  --train-args "$COMMON" \
  --main-args "--setup-mc-credit --setup-entropy-floor 0.3 --resume-every-steps 10000000" \
  --exploiter-args "--opponent-temperature 0 --eta 0 --entropy-coef 0 --adv-norm-learner-only" \
  --publish-win 0.55 --reset-win 0.65 --judge-games 200 \
  --main-description "E5-21-43: ..." --exploiter-description "E5-22-43: ..."
```

What the driver does with these flags:

* It adds the main agent's pool itself: `--opponent-frac 0.3 --opponent-self-pool
  --opponent-pool-size 12 --league-dirs <league-dir> --league-pool-size 4`.
* It resets each exploiter generation to the main agent's newest tagged resume state.
* It judges exploiter snapshots greedily on the CPU, 200 games a side:
  * a snapshot that scores at least 55% is published;
  * one that scores at least 65% is published and starts a new generation.

The main agent and the exploiter together fill both GPU slots.

**Check a replicate against the original:**

* Before launch and after it, run
  `tools/scripts/launch_flags.py <original dir> --diff <replicate dir>` for each phase. It should
  print nothing beyond `--train-steps` in phase 1.
* The phase 2 main agent against E5-11-43 must differ in exactly these flags: `--league-dirs`,
  `--league-frac 0.5`, `--resume-every-steps 10000000`, `--setup-entropy-floor 0.3`,
  `--setup-mc-credit`, `--train-steps 560000000`.

**What a replicate is and is not.**

* **The weights will differ.** GPU training is not bit-reproducible. The league also depends on
  timing: which exploiter snapshots get published, and when, depends on how judging interleaves
  with the main agent's progress.
* **A replicate therefore reruns the recipe, not the checkpoint.** Judge it by the same
  measurements: the per-seat rule against E5-11 at 480/520/560M, and `setup_oracle.py` at 560M.
  The original run's league history is in `data/league/E5-21-43/events.jsonl`. Only generation 2
  published anything: three snapshots at 390–410M.
* **The seed-44 replicate has not been run.** It is on hold until the owner decides.

## E4-era checkpoints (superseded; E4 engine)

| checkpoint | steps | vs HeuristicBot | US / USSR | note |
|:---|---:|---:|:---|:---|
| `E4_1_warmup.pt` | — | 35% | 26 / 44 | BC warmup, 2 epochs on the rebuilt human corpus (254 games, 111,203 samples); 38.65% agreement |
| `E4-02-01 .../snapshot_*` | to 240M | 93–98% | balanced within ~4pp | in flight; see `runs.md` |
| `E4-01-01 .../snapshot_*` | to 184M | peaked 95% @105M | collapsed to 80 / 98 | **aborted, misconfigured** — do not rate |

## Rebuilding a rating

The engine changed, so anything that consumed the old decision stream is void: `(seed, actions)`
datasets truncate silently, and Elo anchors predate the change. `tools/tournament.py` over a run
directory is the way to produce a fresh ladder once E4-02-01 finishes; the human corpus dataset was
already rebuilt against this engine as `/workspace/data/datasets/human_corpus_e4`.
