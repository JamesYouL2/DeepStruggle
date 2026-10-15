# `ply` numbered headlines by side; the engine resolves them in Ops order (fixed)

**Conclusion.** `ai.game_length.ply` numbered a turn's two headline plies USSR first, but the engine
resolves the higher-Ops headline first, so the index could run backwards (30 → 29). Fixed 2026-09-16:
`ply` orders headlines by `state.headline_stage` (0/1 = first headline ply, 2 = second), and the
argument is **required** when `action_round == 0`, because a defaulted one would return a plausible
wrong number. The ply scale is unchanged (`FULL_GAME_PLIES` 154). **Earlier `terminal_plies` and
decisiveness numbers stand**: over 2,000 random games 14 terminal plies moved by exactly one ply,
0.00026 SD of terminal ply. `ply` feeds only `terminal_plies`, not the observation, reward or actions.

**How it hid.** `test_ply_never_runs_backwards_across_a_real_game` fed per-decision (128-wide)
indices to `step_flat`; every step was refused, the bool return was ignored, and the loop asserted
monotonicity over a state that never moved. Making `step` raise (`51b479c`,
[`engine_change_decision_stream.md`](engine_change_decision_stream.md)) surfaced it; the test now
walks five seeds through headlines.
