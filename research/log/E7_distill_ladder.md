# The distillation ladder (2026-10-21 to 26): the rollout root's corrections can be reproduced and ranked, but no network learns to play them

**Question.** [`P36c`](../plans/P36_issue_bank_and_distillation_test.md) closed search distillation as
a correction. That test still left three ways out:

1. Its targets came from the root *before* the horizon fix (`aligned`), and no one re-read them on the
   corrected root.
2. A pick or an improved policy throws away the root's evidence: how large each lead is and how sure
   the root is of it. Fitting the per-candidate values, weighted by their noise, might transfer where
   the pick did not.
3. It tried one teacher. A shorter horizon or a different network might teach something the trunk can
   carry.

This record closes all three, offline, on A13+R48's SWA (11,920–12,000M), the network P36c used.

**Answer.**

* **Every teacher's corrections are local.** The positive control, another network's raw policy, is
  the most learnable.
* **On the aligned root, a policy-fit student plays the teacher's move at 18.3%** of held-out
  departures, against the teacher's own 75.2% (P36c: 20.2%).
* **An action-value head learns how the root ranks candidates**: correlation 0.80 against 0.37 for the
  critic alone, and the same within a position. **Its argmax still plays worse than the network's own
  move.** Its realised gain is −0.10 against the root's +1.35 (per decision, ×100, judged on seeds it
  never saw).
* **Noise is not the limit.** Fitting the 4× noisier 32-world leads costs nothing measurable.
* **Data is not the limit either.** Each doubling of the games gains +0.06, and the gain approaches 0
  from below because the head stops departing from the network, not because it starts playing the
  root's moves.

**Offline distillation of the rollout root's corrections is closed in every form tried:** pick,
improved policy, per-candidate values; frozen trunk, trained trunk, or the policy's own logits; 16 to
128 worlds; 1/16 of the data up to all of it.

## Method

All of this ran on the fork (JamesYouL2/DeepStruggle, branch `exp/distill-ladder`), so the run ids
below are on the fork and are an external record. The code is in this change. The CPU work ran on CI
runners and the fits on one rented 4090.

* **Positions.** 200,000 entry decisions from 2,000 greedy games of the network; a tenth of the games
  (`game % 10`) is held out, 19,147 positions. The held-out rows are searched twice, on two seeds.
* **Teachers.** `tools/distill_targets.py --teacher name=spec`:

  | name | spec | what it is |
  |:---|:---|:---|
  | `h4` | `root:4:16:4:z2:earlier:own:keyed:aligned:box4` | the deployed corrected root at 16 worlds |
  | `h4w32` | the same at 32 worlds | |
  | `h4w128` | the same at 128 worlds | |
  | `h2`, `h1` | `h4` at horizons 2 and 1 | |
  | `h0` | `root:16:4:0:argmax:earlier:own:keyed` | the critic alone, read at the mover's next decision |
  | `soup` | `net:` the R34 soup | the positive control: a target that a network of this shape demonstrably represents |

* **Departures.** A departure is a position where the teacher's top move is not the network's argmax.
  The root's *pick*, the move it plays under its z2 rule, departs less often than its top move. A
  *stable* departure is one where both seeds pick the same move and it is not the network's.

**The runs.**

| fork CI run | what it did |
|:---|:---|
| `39613569557` | `distill_ladder.yml`: collected the positions, then searched h0, h1, h2, h4 and soup |
| `39623394928` | `h4w32`, `h4w128` on the same states (`states_from`) |
| `40244320871` | `h4w32`, `h4w128`, `h0` again, at commit `42fbf65`, which also writes each candidate's mean, paired lead over the network's move and standard error, on both seeds |
| `40363516690` | `q_fit_curve.yml`, the data-scaling curve |

* **The fits.** `tools/train.py --mode policy-fit` (X0) and `--mode q-fit`; the codes are R71 to R80
  in [`../architectures_and_recipes.md`](../architectures_and_recipes.md).
  * **Names.** The runs carry `…+A13@12800M+E8`. They read an E7 network on the E8 build, at the
    network's own view (no observation features). That makes X0's architecture A24, not the A19 it
    was launched under. The curve's full-data cell has R72's flags but a different dataset path, the
    same file by sha256, which makes it R80.
* **Reading the Q head.**
  * **`r_dis`** is the correlation with the two seeds' mean lead, disattenuated by that mean's
    split-half reliability.
  * **The gain** is the Q root's realised gain: the judge's two-seed mean lead of the move it plays,
    per decision, ×100, with the network's move scoring 0. It is unbiased because the head never saw
    those seeds.
  * **The bars** are `gain_judge_xseed` (the root's pick on one seed, valued by the other seed) and
    `gain_base` (h0's own pick).
    * *Caveat, added after the runs:* a bar averages only the rows where the judge searched the move
      it plays. h0 at k 16 often picks a move outside h4w128's four candidates, so `gain_base`'s −0.27
      is a mean over fewer rows than the Q root's gain, not a like-for-like bar. q-fit now also reads
      the Q root on the bar's own rows (`gain_<θ>_on_base_rows`). The verdict rests on the head's gain
      against 0 and against `gain_judge_xseed`; that bar's coverage is recorded beside it in each
      fit's `metadata.json`.

## 1. The teacher ladder: every teacher's corrections are local

All teachers on run `39613569557` are at 16 worlds (h0 at 4).

| teacher | departs (its top move) | departs (its pick) | held-out self-agreement on departures | kNN on the trunk: teacher's move at neighbours, teacher − network | neighbours' vote finds the move |
|:---|---:|---:|---:|---:|---:|
| h0 | 21.7% | 5.0% | 91.0% | +3.9 | 27.6% |
| h1 | 30.6% | 22.6% | 70.3% | | |
| h2 | 32.4% | 18.2% | 57.3% | | |
| h4 | 33.5% | 15.8% | 50.9% | +2.0 | 21.3% |
| h4 at 32 worlds | 35.7% | 19.9% | 59.1% | +2.6 | 21.5% |
| h4 at 128 worlds | 42.8% | 29.5% | 75.2% | +3.8 | 22.1% |
| soup (control) | 28.2% | 28.2% | 100% | **+9.0** | **33.2%** |

* **More worlds make the teacher steadier but no more local.** From 16 to 128 worlds, h4's
  self-agreement rises from 51% to 75%, which reproduces P36b's 72.7% ceiling. Its departures still
  carry over to the trunk's nearest neighbours at +2 to +4 points.
* **The soup control carries over at +9.**
* **The longer the horizon, the less the teachers share.** h4 at 128 worlds and h4 at 32 worlds agree
  on 57% of the 128-world departures. h4 and h0 share 18% of h4's departures.
* **What the playouts add over the critic is the least local part.**

## 2. X0: P36c's student on the aligned targets

`…+E8-A24-R71` is R62 (the fine-tune at lr 1e-3, P36c's best student) refitted on `h4w128` of run
`40244320871`. **It plays the teacher's move at 18.3% of held-out departures, against the teacher's
self-agreement of 75.2%**; P36c's unaligned targets gave 20.2%. The horizon bug was not what held P36c
back.

## 3. The action-value head: it ranks the candidates and still cannot act

Held-out: 19,147 positions, 4,337 of them stable `h4w128` departures. The gain is ×100 per decision at
thresholds θ = 0.02 and 0.04: the Q root leaves the network's move only when its predicted lead
exceeds θ.

| player | r_dis | within a position | departs @0.02 / @0.04 | gain @0.02 / @0.04 |
|:---|---:|---:|---:|---:|
| the root itself (`gain_judge_xseed`: one seed's pick, valued by the other) | 1 | 1 | 42.8% | **+1.35** |
| h0, its own z2 pick (`gain_base`) | 0.37 | 0.41 | | −0.27 |
| the starting logits | 0.34 | 0.37 | | |
| R72: head on the frozen trunk, 1/SE² | **0.80** | **0.80** | 15% / 5% | **−0.10 / −0.06** |
| R73: fitted to the h4w32 leads, judged by h4w128 | 0.79 | 0.79 | 15% / 5% | −0.12 / −0.05 |
| R74: trunk trained | 0.80 | 0.80 | 14% / 4% | −0.11 / −0.06 |
| R75: the policy logits × a learned scale | 0.77 | 0.77 | 11% / 3% | −0.03 / 0.00 |

* **The head learns 80% of the reliable ranking signal, within a position too**, where the critic
  alone has 37%. Correlation is the wrong test, though:
  * the departures' margins are small, about +0.03;
  * they sit below the head's error on the top candidates;
  * so where its argmax leaves the network, it picks wrong slightly more often than right.
* **R73 against R72 rules out noise.** Leads four times noisier (32 worlds against 128) give the same
  correlation and the same gain.
* **R74 rules out a frozen trunk as the limit.**
* **R75 is the least bad only because it departs least.**

## 4. Data scaling: the head converges on the network, not on the root

Run `40363516690` fitted R72 on CPU at seed 45, on the first share of the training games kept whole.
The held-out rows are unchanged.

| share | training positions | r_dis | gain @0.02 | gain @0.04 | gain @0.08 | departs @0.04 |
|--:|--:|--:|--:|--:|--:|--:|
| 1/16 (R76) | 10,195 | 0.669 | −0.38 ± 0.03 | −0.20 ± 0.03 | −0.06 ± 0.02 | 13.0% |
| 1/8 (R77) | 20,126 | 0.736 | −0.32 ± 0.03 | −0.17 ± 0.03 | −0.04 ± 0.02 | 10.7% |
| 1/4 (R78) | 39,892 | 0.766 | −0.25 ± 0.03 | −0.12 ± 0.03 | −0.03 ± 0.01 | 8.2% |
| 1/2 (R79) | 79,461 | 0.787 | −0.18 ± 0.02 | −0.09 ± 0.02 | −0.03 ± 0.01 | 6.5% |
| all (R80) | 160,503 | 0.801 | −0.14 ± 0.03 | −0.07 ± 0.02 | −0.03 ± 0.02 | 5.0% |

* **Each doubling gains about +0.06, mostly because the head departs less (13% down to 5%).**
* **`r_dis` saturates near 0.80.**
* **No feasible amount of data reaches the root's +1.35.**
* The seed-44 cells failed on the run-name grammar and were not re-run; GPU R72 is that cell.

## 5. Instrumentation found on the way: the active-search runs logged no search CE

`generic_trainer.train_pipeline` registered the search-CE metrics only when the `--search-ce-coef`
argument was above 0. `--active-search-spec` leaves that argument at 0 and sets the trainer's
coefficient from `--active-search-ce-coef`, so these runs logged none of the CE metrics:

* `E7-A8-R69-S44` (P36 slot 1);
* the gate-only `R70` runs.

As a result, slot 1's 400M and 800M notes could not say how hard the search target pulls. This
change reads the trainer's coefficient instead, and `tests/training/test_search_ce.py` refuses the
argument form. **A run started before this change logs the CE only after it resumes on code that has
the fix.** `tools/distill_probe.py track` reads an active-search run's snapshots offline: where they
take the root's move on its departures, and where they keep the base move.

## What this closes, and what it leaves

**Closed: offline distillation of the rollout root's corrections into this network.** The tests above
cover every form tried:

| form | result | section |
|:---|:---|:---|
| pick or improved policy | P36c and X0 (18–20% against a 75–79% ceiling) | 2 |
| per-candidate values | the action-value head at any trunk setting, ranking at 0.80 and gaining ≤ 0 | 3 |
| fewer noisy targets (more worlds, inverse-variance weights, a 4× noisier control) | flat | 1, 3 |
| more data | +0.06 a doubling toward 0, not toward +1.35 | 4 |
| another teacher horizon | h1 and h2 are no more local than h4 | 1 |

**A pre-registered check for any future proposal of this kind:** a student or head goes to games only
if its realised gain on the judge's held-out seeds (`q_fit.py`'s `gain_*`, or the same reading of a
policy student's argmax) is above 0 at a departure rate of at least 5%. Every fit here failed that
check; it takes minutes on CPU, whereas a match takes hours.

**Left open:**

* **Active search inside a from-scratch run** (P36 slot 1, read at 2,000M). The tandem-effect
  precedent in P36's verdict is exactly the passive-versus-active difference this record cannot test.
* **Whether the root's stable departures are right or a bias the playouts share.** A blind
  human-judge bank was built on the fork, not in this change; its labels are not in.
* **A better critic, which is the channel the root does keep** ([P34](../plans/P34_search_centric_evaluator.md)
  revision 2, step 4).

## Reproduction

**The fork's artifacts cannot be read on main.** Their raw states were written by the build before
ENG-5, and are readable only by the build that wrote them. On main, run the ladder anew; the positions
will differ, because ENG-5 changes the deals.

```bash
# 1. targets: collect, search in 20 shards, merge, agree + knn (CPU, ~4 h at 128 worlds)
gh workflow run distill_ladder.yml --ref main \
  -f teachers='h4w32=root:4:32:4:z2:earlier:own:keyed:aligned:box4 h4w128=root:4:128:4:z2:earlier:own:keyed:aligned:box4 h0=root:16:4:0:argmax:earlier:own:keyed' \
  -f knn_teachers=h4w128
gh run download <run-id> -n distill-ladder-<run-id> -D data/reports/distill_ladder/<run-id>

# 2. the fits (GPU for policy-fit and --q-train-trunk; the frozen-trunk q-fit runs on CPU)
P=E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@12800M+E8   # name its own engine epoch
D=data/reports/distill_ladder/<run-id>/targets.npz
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/train.py \
  --warmup-checkpoint <A13+R48 swa_11920-12000M.pt> --no-tf32 --mode policy-fit --policy-dataset $D \
  --policy-target h4w128 --policy-lr 0.001 --run-name $P-A24-R71 --description "X0"
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/train.py \
  --warmup-checkpoint <same> --no-tf32 --mode q-fit --q-dataset $D --q-target h4w128 \
  --run-name $P-A23-R72 --description "Q head, frozen trunk"
#    R73: --q-target h4w32 --q-judge h4w128;  R74: --q-train-trunk;  R75: --q-head policy

# 3. the data curve (CPU)
gh workflow run q_fit_curve.yml --ref main -f source=<run-id>
```

The local reports are not in the repository (`data/` is git-ignored):

* `data/reports/distill_ladder/<run-id>/` holds `agree.md`, `knn-*.md` and `provenance.md`;
* each fit's `metadata.json` holds `best` and `fixed`, the readings above.
