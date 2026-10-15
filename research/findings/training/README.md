# findings/training — what a training choice is worth

Each finding is a **difference measured inside one engine** (matched seeds and budgets). Under
[`../../method/what_survives_an_engine_change.md`](../../method/what_survives_an_engine_change.md)
these outlive the engine they were measured on; absolute ratings do not.

| file | what it settles |
|:---|:---|
| [`training_length_and_evaluation.md`](training_length_and_evaluation.md) | runs keep climbing long after they look saturated (A4 ~3,700M, A8 6,800M, R48 9,200M); SWA and soups are measuring devices; the league adds nothing measurable |
| [`trunk_depth_and_state.md`](trunk_depth_and_state.md) | the shallow trunk (0 residual blocks) beats M2d on two seeds; state is lost at the input projections; card × board interactions are representable but not learned by RL |
| [`side_collapse.md`](side_collapse.md) | the E4 side collapse: reproducible, escapable (5 of 7 recover), free when recovered, detected on entry by `adv_std_raw` / `explained_variance` |
| [`entropy_inflation.md`](entropy_inflation.md) | a second failure mode invisible to those detectors: −330 Elo with entropy rising |
| [`forward_pass_trace.md`](forward_pass_trace.md) | ColdWarNetV2's forward pass (pooling is the bottleneck) and the meaning of every per-entity observation slot |
| [`country_identity_without_graph_conv.md`](country_identity_without_graph_conv.md) | only 30 of 84 countries are distinguishable by static slots; identity is a heavily used per-country bias that encodes no map; graph depth is open |
| [`e4_architecture_discontinuity.md`](e4_architecture_discontinuity.md) | E4 launched with the default network, not late E3's; accepted, so E3 and E4 numbers do not compare |
| [`ref_update_freq_open_ablation.md`](ref_update_freq_open_ablation.md) | 200k is the baseline, not a drift; a slow π_ref loses from scratch on E4 |
| [`heuristic_mcts_baseline.md`](heuristic_mcts_baseline.md) | a checkpoint-free searching baseline that beats `HeuristicBot` 18–20 of 20 |

**E3-era findings** (architecture progression, pooling, seed variance, DEFCON blunders, the value
bootstrap, throughput, which decisions to search) are in
[`../../archive/E3_ladder/findings/`](../../archive/E3_ladder/README.md). Their seed-variance rule
still applies: the spread between same-condition arms is the yardstick an effect must clear
([`../../archive/E3_ladder/findings/seed_variance.md`](../../archive/E3_ladder/findings/seed_variance.md)).

Training results since E5 that have no findings file are summarised by question in
[`../../questions.md`](../../questions.md) and per arm in [`../../runs.md`](../../runs.md).
