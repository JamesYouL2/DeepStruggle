# P30 — base-model quality

**Status: closed (2026-10-06).** The candidates below were run or dropped; the work moved to the
rare-spot leaks (P31, deleted; [`../log/P31_branch_arms_B1_B2.md`](../log/P31_branch_arms_B1_B2.md))
and then to search ([P33](P33_search_preserved_gains.md)). Two cells were never run: C2 from
scratch and C3.

**Goal (owner, 2026-10-01):** the quality of the base model, what one training run produces. SWA and
soups read a recipe with less noise; they are not the product, and averaging does not enter training.

**Protocol.** Arm and control from scratch on the same seed, traced every 80M to ~1B steps; read the
late snapshots and the last-80M SWA per seat against the panel and head to head; adopt on the
owner's rule (better in one seat, not worse in the other) at saturation, on two seeds. A gain only
at 560M is "faster", not "better".

## Outcomes

| cell | change | outcome |
|:---|:---|:---|
| base | the shallow trunk, `--ladder-res-blocks 0` (E7-01/02-44) | adopted: beats M2d on two seeds, saturates at ~900M–1B ([`trunk_depth_and_state.md`](../findings/training/trunk_depth_and_state.md), [`E7_06_44_deep_to_saturation.md`](../log/E7_06_44_deep_to_saturation.md)) |
| C1 | card and country tokens with two attention layers (E7-09-44) | stopped at 400M: behind on snapshots, level on the SWA, at ~9× the compute per step ([log](../log/P30_C1_tokens.md)). C1b (concatenated tokens, identities at full scale) never run |
| C2 | the card-event auxiliary target, `--aux-card-events` (E7-13-44, a branch from 820M) | a small positive on the branch (US +3/+4 SE, head to head level); the target does reach the trunk ([branch](../log/P30_branch_arms_c2_play_mode_temp.md), [probe](../log/P30_c2_card_probe_and_branch_soups.md)). **Never run from scratch** |
| C3 | a stepped LR over the last third | never run (the deep line's version lost on its SWA, [P28 step 2a](../log/P28_step2a_lr_schedule.md)) |
| C4 | the ops budget in the observation, `--obs-features ops_budget` (E7-07-44) | no detectable effect at saturation: between the two seeds' controls, which are 55 Elo apart ([log](../log/P30_C4_ops_budget.md)) |
| C5 | search-driven training (E7-18-44, a branch) | fails the rule: USSR −3.0 ± 0.5, US +1.6, 51% head to head ([log](../log/P30_search_training_e7_18.md)); headroom measured in [`P30_search_headroom_e7.md`](../log/P30_search_headroom_e7.md) |
| side arms | play-mode temperature 2 (E7-14-44); win-only reward (E7-15-44); VP potential (E7-16-44) | temperature hurts slightly ([log](../log/P30_branch_arms_c2_play_mode_temp.md)); neither reward change passes ([log](../log/P30_win_only_and_vp_potential.md)) |
