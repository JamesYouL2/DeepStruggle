# Method — how to run an experiment and how to trust the number

Read before a run, and again when a result looks too good. **Living reference.**

| file | what it is for |
|:---|:---|
| [`measurement_pitfalls.md`](measurement_pitfalls.md) | **read before trusting a number**: every way an instrument here has lied, with the cheap check |
| [`running_experiments.md`](running_experiments.md) | how an arm is rated (SWA + snapshots, per-seat acceptance, leaderboard), budgeted, and what a step is |
| [`bookkeeping.md`](bookkeeping.md) | how to record an experiment so it can be found: `runs.md`, `questions.md`, the log, findings |
| [`run_nomenclature.md`](run_nomenclature.md) | `E<n>-A<n>-R<n>-S<n>@<step>M+<changes>`, SWAs and soups, the old attempt-number scheme |
| [`measurement_tiers.md`](measurement_tiers.md) | which probe runs every iteration, per snapshot, once per arm |
| [`detecting_collapse.md`](detecting_collapse.md) | which internal metric sees which failure mode; self-relative alarms (`watch_run.py`) |
| [`what_survives_an_engine_change.md`](what_survives_an_engine_change.md) | the assumption that relative results cross an engine revision and absolute ratings do not |
| [`human_play.md`](human_play.md) | the two human corpora (ts-replayer for moves, ITS for length and endings) and the agreement measure |
| [`references.md`](references.md) | the literature behind the plan ordering, mapped to this game |

The evidence for each checklist item is [`../log/measurement_bugs.md`](../log/measurement_bugs.md).
