# Checkpoint catalogue — which model to load, and what it is worth

The arms are [`runs.md`](runs.md); this file answers "which model do I load, and how good is it".
Rewritten 2026-10-15; the earlier per-era catalogue (E6 tournament table, the E5-21 replication
recipe, E4-era rows) is in git history (`git show e5e8a7c:research/checkpoints.md`).

## Ratings: the E7 leaderboard

Ratings live in `leaderboard/` (`tools/README.md` §2b): one Elo scale for the E7 engine, anchored at
**`E7-A4-R1-S44@4800M` = 1500**, the main players fitted on their own round robin and every newcomer
against them held fixed (1,000 deals a side, greedy unless the id says otherwise). Each network's file
and sha256 are in `leaderboard/networks.json`; its behaviour report is in
[`log/per_checkpoint/`](log/per_checkpoint/README.md). `tools/leaderboard.py show` at `e5e8a7c`:

| player | Elo | |
|:---|---:|:---|
| `E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M~rollout(k=4,worlds=16,horizon=4,rule=z2)` | 1791 | **strongest player**: the R34 soup under the rollout root |
| `E7-A8-R1-S44@6400M+R39@7120..7200M~rollout(…)` | 1779 | |
| `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M~rollout(…)` | 1767 | |
| `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M` | 1679 | **strongest greedy network** (A13+R48 SWA) |
| `E7-A8-R1-S44@6400M+R39@7120..7200M` | 1676 | |
| `E7-A8-R1-S44@6400M+R34@7200M+R48@9120..9200M` | 1667 | |
| `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M~gumbel(sims=256,k=8,fpu=0.2)` | 1666 | |
| `E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M` | 1650 | main; the best purely trained soup |
| `E7-A8-R1-S44@6400M+R32@6720..6800M` | 1643 | |
| `E7-A8-R1-S44@6400M+R34@7120..7200M` | 1629 | |
| `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M` | 1625 | main; the "heads soup" |
| `E7-A8-R1-S44@6720..6800M` | 1580 | main |
| `E7-A4-R1-S44@4390M+(S44,45,46)@4720..4800M` | 1570 | main |
| `E7-A8-R1-S44@6800M` | 1548 | main |
| `E7-A4-R1-S44@4720..4800M` | 1531 | main |
| `E7-A4-R1-S44@4800M` | 1500 | main, anchor |

Where each comes from: R34 is the logit-gap penalty, R39 the frozen-soup teacher searched by the
rollout root, R48 R34 with a 4× minibatch, A13 the input projections widened in place, R32 a frozen
Gumbel teacher ([`runs.md`](runs.md), [`architectures_and_recipes.md`](architectures_and_recipes.md)).
Search on top: [`log/E7_search_depth_and_value.md`](log/E7_search_depth_and_value.md).

## Designations (owner, 2026-10-06)

The latest formal designation; the leaderboard's main set (above) adds the A8 line's bests and the
R34 soup, and the stronger models above have not been designated.

| designation | label | file | sha256 |
|:---|:---|:---|:---|
| best raw model | `E7-A4-R1-S44@4800M` (E7-20-44@4800M) | `data/checkpoints/E7-20-44_20261005_174535/snapshot_4800053248steps.pt` | `7ca02cc3b432…a8cc164` |
| best SWA | `E7-A4-R1-S44@4720..4800M` | `data/checkpoints/_models/E7-A4-R1-S44@4720..4800M.pt` (= `_swa_line_ctl/E7line_swa_4720-4800M.pt`) | `094a22073974…c072ab` |
| best soup, best overall | `E7-A4-R1-S44@4390M+(S44,45,46)@4720..4800M` (SWA-made; raw twin `…@4800M`, level) | `data/checkpoints/_models/E7-A4-R1-S44@4390M+(S44,45,46)@4720..4800M.pt` | `e2447821624a…0db00487` |

At designation the soup beat the 2026-10-01 best soup 63.5% and passed the per-seat rule against
everything in its field ([`log/E7_soup_4800M.md`](log/E7_soup_4800M.md)).

### Superseded designations

| date | designation | model | file | sha256 |
|:---|:---|:---|:---|:---|
| 2026-10-01 | raw / SWA | E7-02-44@1,200M / its 1,120–1,200M SWA | `E7-02-44_20260930_222159/snapshot_1200029696steps.pt`, `…/swa_1120-1200M.pt` | `75b68238138b…`, `8b205fad425f…` |
| 2026-10-01 | soup | E7-02/03/04/05-44 at 1,200M (`E7-A4-R1-S44@870M+(S44,S45,R8,R9)@1200M`) | `_soups/shallow_E7-02+03+04+05_1200M.pt` | `55a26630519e…` |
| 2026-09-29 | raw / SWA (E6) | E6-07-44@700M / E6-06-44 680–760M SWA | `E6-07-44_20260929_121304/snapshot_700055552steps.pt`, `E6-06-44_20260929_121244/swa_680-760M.pt` | `34017ae130fe…`, `6a3be70cb503…` |
| 2026-09-30 | soup (E6) | E6-06/07/08-44 at 760M | `_soups/shared_start_E6-06+07+08_760M.pt` | `fa21df467d80…` |
| 2026-09-28 | best (E5) | E5-21-43@560M | archived (below) | `f3213b98e0ba…` |

Paths are under `data/checkpoints/`. Evidence: [`log/E7_02_44_saturation.md`](log/E7_02_44_saturation.md),
[`log/E7_shallow_soup.md`](log/E7_shallow_soup.md), [`log/E6_comprehensive_2026-09-29.md`](log/E6_comprehensive_2026-09-29.md),
[`log/model_soups_2026-09-30.md`](log/model_soups_2026-09-30.md),
[`log/E5_21_setup_credit_plus_league.md`](log/E5_21_setup_credit_plus_league.md).

## Reproducing the E7 line

One run of seed 44 under the shallow recipe (A4-R1), through `tools/train.py` on E7, guarded by
`tools/scripts/check_engine_fresh.sh`, with `PYTHONPATH=.:build/release TRITON_CACHE_DIR=.triton_cache`:

```bash
SHALLOW="--arch ladder --ladder-input-mode grouped --ladder-aggregation flatten --ladder-entity-dim 16 \
  --ladder-entity-proj-dim 256 --ladder-hidden-dim 480 --ladder-res-blocks 0 --drop-static \
  --per-entity-heads 64 --ladder-head-context --ladder-head-static --ladder-head-entities country"
COMMON="--reward-scheme blunder_aware --opponent-frac 0.3 --opponent-self-pool --opponent-pool-size 12 \
  --eval-opponents heuristic random --ladder-head-center --block-lambda off --rollout-temps 1.0 1.0 1.0 1.0"
.venv/bin/python tools/train.py $SHALLOW $COMMON --seed 44 --train-steps <budget> \
  --run-name <name> --description "..." [--resume <previous leg's dir>]
# the SWA of a window, and a soup of SWAs
tools/scripts/average_weights.py --snapshots <the window's 10M snapshots> --output <swa>.pt
tools/scripts/average_weights.py --snapshots <the branches' SWAs> --output <soup>.pt
```

* **The A4 lineage** (old names): E7-01-44 (0 → 560M), E7-02-44 (→ 1,200M), E7-17-44 (→ 2,000M),
  E7-19-44 (→ 2,400M), E7-20-44 (→ 4,800M), each leg resumed with unchanged flags; from 3,600M the
  per-entity head adds the trunk context once per sample (`0280264`, identical to float64 rounding).
  The best soup's branches: E7-20-44's `resume_4390060032steps.pt` to 4,800M with
  `--seed-env/--seed-pool/--seed-sampling 45` (and 46).
* **The A8 line** (`E7-A8-R1-S44_20261007_162257`) adds `--ladder-branch-head --ladder-play-mode-head`
  from scratch; its 6,400M state is the branch point of every R2x–R4x arm. For any run, derive the
  command with `tools/scripts/launch_flags.py <its directory>` rather than from this page.

Check a replicate with `tools/scripts/launch_flags.py <original> --diff <replicate>`. GPU training is
not bit-reproducible across hardware, so a replicate reruns the recipe, not the weights.

## Older checkpoints

* **Pre-E6 checkpoints were archived 2026-09-29** out of `data/checkpoints/` into
  `data/archive/{E4_ladder,E5_ladder,E6_intermediate}/checkpoints/`; `data/checkpoints/` holds real-E6
  and E7 runs, which is all the workbench lists.
* **Pre-P17 (E3 and earlier) checkpoints cannot load** on this engine: their policy head is 212 wide
  against 220, and `check_checkpoint_layout` refuses them. Their ratings, none of which transfer, are
  in [`archive/E3_ladder/checkpoints.md`](archive/E3_ladder/checkpoints.md).
