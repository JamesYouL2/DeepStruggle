# Bookkeeping — how to record an experiment so it can be found again

Written 2026-09-16, when "what did we learn about pooled vs non-pooled?" took one log, two plans,
three git-ignored reports and nine `metadata.json` descriptions to answer, and "pool" named four
different mechanisms. The rule: **an experiment is not recorded until the question it answers has a
row.** Arms are found only by people who already know their names.

## The four places

| | file | discipline |
|:---|:---|:---|
| 1 | [`../runs.md`](../runs.md) | one row per **arm**: what it varied, seeds, budget, directory, link to the writeup |
| 2 | [`../questions.md`](../questions.md) | one row per **question**: the arms that bear on it, and the verdict |
| 3 | `../log/<programme>.md` | the experiment: setup, numbers, prediction, outcome, caveats. Append-only |
| 4 | `../findings/{engine,training}/<topic>.md` | what is true now, cited to 3. Rewritten in place |

Engine or training finding? Ask what would invalidate it: a rules fix, mask change or broken probe
→ **engine**; only a better-powered replication → **training**
([`../findings/README.md`](../findings/README.md)). Ratings live in `leaderboard/`: a log cites a
player's Elo from `tools/leaderboard.py show` at a commit rather than copying a table.

## Every measurement lands in `log/` first

No other document holds the only copy of a number — not `findings/`, `runs.md`, `checkpoints.md`, a
plan, or a commit message. (On 2026-09-16 three tournaments went straight into `findings/` as
verdicts, with no setup, sample size or instrument anywhere.) "Measurement" includes tournaments,
probes that returned nothing, statistics computed from `training_metrics.jsonl`, sensitivity checks
and corpus counts. The entry carries the question, **the instrument and sample size**, the result
with an error bar, the verdict, and what would overturn it. **Test:** from a `findings/` claim, one
link must reach the sample size and the instrument.

A measurement belonging to an existing programme is appended to its file; a new programme gets a new
file and a row in [`../log/README.md`](../log/README.md).

## Before launching

* **Register the prediction** — what the probes and the Elo should do — in the plan or the arm's
  `--description`.
* **Name the endpoint and the instrument that produces it.** The pooling replication pre-registered
  a statistic, never computed it, and reported a different instrument chosen afterwards.
* **If `engine/` changed since the control was trained, measure the decision stream** (both builds,
  fixed seeds, diff action and mask hash;
  [`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md)).
* **Grep before reusing a word** for a new mechanism.
* **Write `--description` for a stranger**: the programme, the matched control, what it is judged on.
  It is often the only record that survives.

## While running and after

* Add the `runs.md` row **at launch**, not at the end; an unlisted directory is how E3-18 and E3-19
  went missing.
* At the end: write the log entry (negative results too); fill the arm's result cell, or write **no
  writeup**; add or update the `questions.md` row (including back to **open**); write a finding only
  when the answer stands without its narrative.

## Artifacts

* **One `data/` tree, the main checkout's** (`/workspace/data`). A worktree's own `data/` is empty and
  deleted with it (1.4 GB of checkpoints were once stranded in one). `tools/lib/data_root.py` resolves
  the shared tree from `git rev-parse --git-common-dir`, `$TS_DATA_ROOT` overrides it, and
  `tools/train.py` uses it. Paths typed on a command line (`--output-dir`) are not routed through
  it: write `/workspace/data/...`. Paths quoted in `research/` are relative to the repository root.
* **`data/` is git-ignored, so copy the numbers that carry a claim into the log** and name the report
  file. Stage checkpoints under readable names before rating them; a report whose entrants are
  `snapshot_final#1..#4` is unusable.

**The test before calling it recorded:** ask "what did we learn about X?" and answer it in one
lookup from [`../questions.md`](../questions.md).
