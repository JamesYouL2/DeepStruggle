# Checkpoint catalogue — what exists on disk, and what it is worth

One row per rated checkpoint, with its measured strength and the field that measured it. The arms
themselves are [`runs.md`](runs.md); this file answers the narrower "which model do I load, and how
good is it".

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

## Best model: E5-21-43@560M (owner, 2026-09-28)

`/workspace/data/checkpoints/E5-21-43_20260928_165156/snapshot_560005120steps.pt`
(sha256 `f3213b98e0ba382930a41caadb58ce22c30fae2c14342150a7510cf36775d7c1`, 12.8 MB).

What it is worth ([`log/E5_21_setup_credit_plus_league.md`](log/E5_21_setup_credit_plus_league.md)):

* **Elo 2344**, the highest in its round robin (`data/reports/e5_21_rr_43.md`).
* **Against E5-11-43**, the previous best, over the late block: 54.6% overall.
  * Per seat against E5-11's self-play bars: US +11.8 ± 2.6, USSR −0.8 ± 2.6. By the letter of
    the owner's per-seat rule it is not accepted, since the USSR seat is a third of a standard
    error short.
* **Past 560M the same run does not improve.** The 720M continuation plateaus and then dips to
  2267.
* **Openings:**
  * US: West Germany 3, France 3, Italy 2, Iran 1.
  * USSR: Poland 3, Hungary 3.
  * Neither side has an alternative more than 2.2 ± 1.5 points better.
* **Self-play is balanced.** Greedy self-play gives US 49.6% over 500 deals.
* **One seed only (43).**

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
