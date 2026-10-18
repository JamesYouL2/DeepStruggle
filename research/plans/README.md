# Plans — what is queued, and what each closed plan concluded

This directory is the **queue**: one file per step that has not been run yet.
[`../log/`](../log/README.md) is the **record** of what was run. A closed plan's file is deleted
(or, rarely, archived) and leaves one row in the table below with its outcome and a link to the log
that holds the numbers. A deleted plan's text is in git: `git log -- research/plans/<file>`.

## The goal

**Base-model quality** (owner, 2026-10-01): what one training run produces, judged at saturation
on two seeds; SWAs and soups are evaluation devices, not the product. Since 2026-10-15 the
deployed player is the network under the rollout root, so a gain counts when it survives that
search ([P33](P33_search_preserved_gains.md)). The behavioural bar is still a mediocre human: a
sane setup, contested battlegrounds, no self-inflicted DEFCON 1, and a card it must not play
disposed of through that card's own exit.

## Active

| plan | what | status |
|:---|:---|:---|
| **[P35](P35_capacity_belief_dynamics.md)** | **seven checks toward a much stronger base model (owner's priorities, 2026-10-18):** a transformer encoder at scale and the current architecture scaled, both from scratch to their own plateaus (every capacity read so far was pre-plateau); a belief over the hidden state (a probe on logged histories, then belief-weighted worlds for the search, then a belief input as the owner's call); η = 0 and the 5M reference from the plateau (the KL term is 2% of the loss on a three-iteration-old anchor, never ablated on E7); search-computed advantages for the policy gradient; PFSP; strategy conditioning after a diversity read. Branches and probes first (3–4 days, two slots), then the two from-scratch flagships for ~two weeks; the interactions that fix that order are listed | proposed |
| **[P34](P34_search_centric_evaluator.md)** | **the step change (owner, 2026-10-17):** a second network read only at the rollout root's leaves, trained supervised on positions labelled by 32-world playouts (unbiased for the greedy value, ~30× the information of a game outcome), compared across architectures on one dataset, measured under the root, iterated by generation with a reverse curriculum from exact turn-10 values and distilled back into the RL critic. New nomenclature: `D` labelling recipes, datasets `<model>~D1(…)`, evaluators as A+R pairs, `leaf_net=` in the search spec ([`../datasets.md`](../datasets.md)) | generation 1 done: labels (1.1M, exact-slice bias −1.3 points but 16 on contested positions), four fits (fine-tune −30% of the critic's excess error; from-scratch data-limited), the two-network root +10, keyed chance +16 free; **revision 2** (the review's five corrections; next: the leak-pair read, keyed siblings, the corrected refit, then a contested-position set with a stronger follow-up) |
| **[P33](P33_search_preserved_gains.md)** | base-checkpoint gains the rollout root keeps; the critic as the main track, every arm read raw **and** under `rollout(4,16,4,z2)` | diagnostics done ([`../log/P33_diagnostics.md`](../log/P33_diagnostics.md)); R52 proposed |

## Shelved or done, kept for a sizing or a design

| plan | what | why shelved |
|:---|:---|:---|
| [P2](P2_chance_aware_targets.md) | dice expectation + pre-deal bootstrap in the value targets | removes ~7% of one-step error variance (dice 4.3%, deal 2.8%); needs a bindings helper |
| [P5](P5_oracle_critic.md) | oracle critic as a deal-side variance reducer | the opponent's hand adds +0.3 points of explained variance (0.283 → 0.286) |
| [P24](P24_league.md) | AlphaStar-style league without a supervised reset point | run on E5–E7; it helped only paired with setup credit on the deep trunk (E5-21, E6-04), and adds nothing measurable on the shallow trunk or from a plateau; the per-seat design for a next attempt is in the file |
| [P13](P13_one_game_driver.md) | one game driver | **done**: one checked step and one chance drain under every loop; the file keeps the replay contract that code comments cite |
| [reserve](reserve.md) | ideas with a trigger | |

## Closed

| plan | question | outcome |
|:---|:---|:---|
| **P32** ([file](P32_teacher_and_paired_credit.md)) | a search teacher that does not flatten the policy; paired turn-end credit | a **frozen** teacher pays (R32 SWA 1645, R39 1676); the rollout root's margin does not distil (R40–R42); search values into the critic null (R35); B4′ parked (R37/R38); the logit-gap loss (R34) replicated on three seeds and went into the recipe. Superseded by P33 |
| P31 (deleted) | four "show, then let training decide" mechanisms for rare-spot card leaks | B1 play-mode floor and B2 scenario seeding level on strength, leaks unmeasured; forcing taught Wargames only by showing it; **card-conditioned heads (E7-33-44) found it unforced and became the heads line (A8)**; 1c became P32's B4′; 1d never run ([`../log/P31_branch_arms_B1_B2.md`](../log/P31_branch_arms_B1_B2.md)) |
| **P30** ([file](P30_base_model_quality.md)) | base-model quality: input representation, auxiliary targets, LR schedule | shallow trunk adopted; C1 tokens stopped at 400M (level, ~9× compute); C2 card-event target a small positive on a branch, never run from scratch; C4 ops budget no detectable effect; search-trained E7-18-44 fails; C3 never run |
| P29 (deleted) | five structural bets from scratch | bet 1, a 3.5× trunk, not better at 560M and worse at matched wall-clock ([log](../log/P29_bet1_wider_m2d.md)); bet 2, the ownership target never reached the trunk ([log](../log/P29_bet2_ownership_probes.md)); bet 3, the merged-influence view trains cleanly on E6 but plays ~235 Elo weaker ([log](../log/P29_bet3_merged_view.md)); bets 4–5 into P30 |
| P28 (deleted) | the E6 measurement floor and tail levers | panel built, E6-04-44 adopted ([log](../log/P28_step0_panel_and_E6_04_verdict.md)); SWA beats every snapshot ([log](../log/P28_step1_swa_and_search.md)); a stepped LR does not cut the spread ([log](../log/P28_step2a_lr_schedule.md)); EMA beats raw snapshots, but averaging stays out of training ([log](../log/P28_step2b_ema_weights.md)) |
| P27 (deleted) | is "no plan inside an ops play" capacity, learning or data? | half of takeable contested battlegrounds are missed, but the rollout oracle prices taking one at −0.8 ± 0.6 pp under this policy: nothing to train toward ([log](../log/P27_ops_block_stage1.md)) |
| P26 (deleted) | training throughput after the 09-23 pass | TF32 +13–16%, adopted; compile erratic; bf16 drifts; snapshot eval 17% of wall ([log](../log/P26_quick_screen.md)) |
| P23 (deleted) | E4.1: "influence, first point in X" as one decision | US seat collapses in 6 of 6 runs on E4 ([log](../log/P23_E4_1_ab.md)); on E6 trains cleanly but −235 Elo (P29 bet 3). The view survives as `--merged-influence` |
| P22 (deleted) | identity-keyed card lookup; trunk width 256 → 512 | both rejected: −154 / −72 and −129 to −291 at 80M ([log](../log/P22_width_probe_and_card_lookup.md)) |
| **P21** ([file](P21_architecture_ladder.md)) | the architecture ladder, one mechanism at a time from an MLP | **M2d** wins: grouped projections +109, the country per-entity head +282, the card head −7; no identity addition or removal beat it ([log](../log/P21_ladder_status.md)); M3–M5 became P30 C1. The file keeps the `--arch ladder` contract and rung flags the code cites |
| P18 (deleted) | why one training arm cannot fill the GPU | superseded by the CPU pass ([log](../log/training_throughput_cpu.md)) and P26 |
| P11 (deleted) | running the programme on rented GPUs | E3-era programme; the rental scripts are [`../../tools/scripts/cloud/`](../../tools/scripts/cloud/README.md) |
| P8 (deleted) | an auxiliary risk head for provoked DEFCON 1 | premise withdrawn: the critic does see the forced-exit trap at entry (US −0.08, USSR 0.00; [log](../log/P32_cpu_checks.md)); `--defcon-coef` exists, never measured |
| P7 (deleted) | human data | conversion, BC warmup and injection done ([conversion](../log/P7_replayer_conversion.md), [BC](../log/P7_human_bc_warmup.md), [injection](../log/P7_human_injection_and_its_cost.md): a 157–236 Elo handicap); arms 7b–7d never run, kept in [reserve](reserve.md) |
| P4 (deleted) | setup credit through the placement block; a human anchor as contrast | A fixes Poland, not West Germany; B holds the human opening on 1 of 2 seeds; +27 / +18, inside noise ([log](../log/P4_setup_arms.md)); A1 adopted, then reverted at flat temperature ([log](../log/E5_12_setup_credit_at_flat_temperature.md)) |
| P0 (deleted) | instruments to judge later steps by | built piecemeal: `ai/eval/setup_probe.py`, `ai/eval/sequencing.py`, `tools/scripts/goal_probes.py`; the chance decomposition became P2's sizing |
| P25 | collapse robustness | a collapse on the fixed pool is a recoverable delay; no lever adopted ([archived](../archive/E4_ladder/plans/P25_collapse_robustness.md), [log](../log/P25_stress_bench.md#closing-p25-2026-09-24)) |
| P19, P20 | is the late-E3 bundle stronger on E4? a positional board path? | +447 Elo; the positional path is P21's M1 ([archived](../archive/E4_ladder/README.md)) |
| P1–P17 (E3) | the E3 stall programme | [`../archive/E3_ladder/`](../archive/E3_ladder/README.md) |

## How to maintain this directory

1. **Before running a step**, fill in anything its file marks *decide before running*; if the
   design changed, edit the file first so the log can quote it.
2. **While running**, list the run directories under *Runs* so another session can find them.
3. **When a step is done** (adopted, rejected or abandoned with a reason): write the log entry with
   the numbers and the follow-ups; **delete the plan file** and leave a one-line outcome row above
   linking the log. Archive it instead only when its diagnosis outlives its proposal (P20 is the
   example). Queue surviving follow-ups as new files.
4. **New ideas** go in [`reserve.md`](reserve.md) with a trigger, not as a step file: a step file
   means someone intends to run it next.
5. **No results here.** A plan that accumulates results is a log entry not yet written.
6. **One step, one file.** Arms that test two things at once produce unattributable results.
7. **A sweep is one row**, in [`../runs.md`](../runs.md) as well: record it once with its census
   linked; individual rows only for representatives and anomalies.
8. **Labels inside a plan (owner, 2026-10-15).** `P<n>` is a plan. A measurement under it that
   trains nothing is `P<n>a`, `P<n>b`, …; one that trains a model takes a recipe code `R<nn>` or an
   architecture code `A<nn>`, reserved in the plan and registered in
   [`../architectures_and_recipes.md`](../architectures_and_recipes.md) at launch. `E` is the
   engine revision only. A labelling recipe is a `D<nn>` code and a dataset is a model under one,
   `<model>~D<nn>(…)`, ledgered in [`../datasets.md`](../datasets.md) (owner, 2026-10-17).

Budgets, seed counts and error bars are not repeated here: see
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md),
[`../log/variance_and_noise.md`](../log/variance_and_noise.md) and
[`../method/running_experiments.md`](../method/running_experiments.md).

## Step file template

```
# Pn — <name>

**Status:** queued | running | blocked on <what>
**Gate:** <what must be true before this runs>
**Needs approval:** <engine / bindings / observation changes, or "none">

## Goal
## Why (which measured failure, which finding, which paper)
## Change (files, flags, what stays fixed)
## Procedure (arms, seeds, budget, control)
## Measure (probes first, Elo last)
## Decision rule (adopt / kill / extend — decided before running)
## Follow-ups
## Runs (filled in while running)
```
