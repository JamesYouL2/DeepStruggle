# Run nomenclature — the naming scheme

**This file is the scheme and nothing else.** The arms themselves — what each one varied, its
seeds, its budgets, its checkpoint directory and where its result is written up — are in
[`../runs.md`](../runs.md), the registry. They were one file until 2026-09-16, and mixing them
meant that looking up what `E3-20` *is* required reading three pages of prose about what `E3-14`
*measured*.

Every number in this project is only meaningful relative to four things, and three of them have
silently changed under a comparison at least once:

- the **engine**, which decides what game is being played;
- the **observation**, which decides what the network is told;
- the **recipe** — reward, loss, trainer flags — and the **architecture**;
- the **budget** in steps, and the **seed**.

`arm_H2_cont_240to480` says none of them. It took a `git merge-base` against two commit hashes to
establish that H2 predates the Aldrich Ames and Star Wars fixes and so was trained on a different
game from everything in P1 — which makes every "rated against H2" number a cross-engine
comparison. That should have been legible from the name.

## The scheme (since 2026-10-07): engine, architecture, recipe, seed, and the path taken

A run is named by the four things a number depends on, each a labelled field, and by every
branch point on the way to it:

    E7-A4-R1-S44                                   from scratch
    E7-A4-R1-S44@4390M+S45                         its 4,390M state, continued under seed 45
    E7-A4-R1-S44@4390M+A7-R23@4800M+R24            two changes in a row
    E7-A4-R1-S44-2                                 a same-seed replicate of the first

- **`E`** — the engine revision, as below (`E7`; a minor version is `E4.1`).
- **`A`** — the **architecture**: the network's shape, what it reads (its observation features)
  and its policy heads. A code, defined in [`../architectures_and_recipes.md`](../architectures_and_recipes.md).
- **`R`** — the **recipe**: every other training setting -- reward, loss, opponents, schedules,
  forcing. A code in the same file. An auxiliary head trained only as a target (the card-event
  head) is recipe: the policy never reads it.
- **`S`** — the **seed** of the environment, sampling and opponent pool.

**A code is a set of flags, not a description.** `tools/scripts/run_codes.py` reads each run's
`metadata.json`, splits its non-default `tools/train.py` flags into architecture, recipe and
bookkeeping (output, cadence, evals, budget, seeds), and gives each distinct set a code --
recorded with its description in [`../run_codes.json`](../run_codes.json). Two runs with one code
therefore cannot quietly differ in a flag, which is how lineage-critical flags drifted four times
(CLAUDE.md, invariant 15). A and R are numbered across the whole project, not per engine:
`E6-A4-R1-S44` and `E7-A4-R1-S44` are visibly one recipe on two engines. What flags cannot
show -- a code change under the same flags, a league whose exploiter was not training -- is
recorded by hand as an `override` in the JSON and makes a recipe of its own.

**`@<n>M` is always an absolute step on the lineage's own clock**, never a length -- the clock the
snapshots already use, so a typo cannot silently shift every later number. In a run's name every
`@` is a **branch point**: `+` and the fields that change, in E, A, R, S order, naming only what
changed. A **continuation that changes nothing keeps its name** (each leg is a new
`<name>_<timestamp>` directory); a re-run of a from-scratch run takes a replicate index `-2`.

| what happened | name |
|:---|:---|
| plain run, 0 → 2000M | `E7-A4-R1-S47` |
| a head added at 2000M | `E7-A4-R1-S47@2000M+A8` |
| two more seeds from 2600M of that | `E7-A4-R1-S47@2000M+A8@2600M+S45`, `…+S46` |
| a recipe change from 820M | `E7-A4-R1-S44@820M+R14` (E7-13-44) |
| an engine change | `E6-A1-R1-S44@560M+E7` (E7-06-44: the deep E6 run continued on E7) |
| back to the plain recipe | `E7-A4-R1-S44@4390M+A7-R23@4800M+R24@5200M+R1` (E7-32-44) |

### A model: one snapshot, an SWA, or a soup

A *label* names a model, so it ends in a **selector** -- `@4800M` for one snapshot, `@4720..4800M`
for the SWA of every snapshot in that range -- and may put a **group** after the last branch point,
which is a soup of the alternatives:

    E7-A4-R1-S44@4800M                                   one snapshot
    E7-A4-R1-S44@4720..4800M                             the SWA of that range
    E7-A4-R1-S44@4390M+(S44,45,46)@4800M                 the best model (2026-10-06): a soup
    E7-A4-R1-S44@4390M+(S44,45,46)@4720..4800M           the same soup made from SWAs
    E7-A4-R1-S44@870M+(S44,S45,R8,R9)@1200M              the first shallow soup: seeds and recipes
    E7-A2-R2-S5@800M+R3@1200M+(S6,R4-S7)@1500..1600M     two branches of 1200M, each an SWA
    (E7-A4-R1-S44,E7-A4-R10-S44)@1200M                   two runs with no common state

- An item that changes nothing (`S44` on a seed-44 parent) is the **parent's own continuation**;
  write it out, since an empty item is easy to miss. A bare number repeats the previous item's
  letter: `S44,45,46`.
- The selector applies to every item. Groups nest, and a top-level group lists whole runs.
- A soup has **one A and one E**: weights of different architectures cannot be averaged, and a
  soup across engines plays no one game. The parser refuses both.
- A soup of runs with no common state (the top-level group) parses, and
  `shares_a_state()` says it is suspect: independent runs are not aligned, and their average is
  usually broken. Every soup that has worked here branched from one checkpoint.
- **Parentheses, not braces.** Both shells expand `{a,b}` silently into separate words, and `[a]` is
  a filename pattern bash silently rewrites when a file matches; an unquoted `(` is a syntax error
  in bash and zsh -- loud, never wrong. Quote a label with a group on the command line. Directories
  never contain one: a run is a single path, so a soup or SWA is a model file named by its label.
- In a URL query string `+` decodes to a space: encode names there.

The grammar's one parser is `ai/training/run_name.py` (`parse_run_name`, `parse_label`,
`is_run_name`); `tools/train.py --run-name`, the league driver and `tools/lib/checkpoint_id.py`
use it, and a snapshot's label (`E7-A4-R1-S44@4390M+S45@4800M`) parses back to its run.

### Launching a run

1. Derive the flags from a healthy run (`tools/scripts/launch_flags.py`, invariant 15).
2. Find the A and R codes those flags are in [`../architectures_and_recipes.md`](../architectures_and_recipes.md).
   A setting no code has is a new code: launch, then `tools/scripts/run_codes.py --update`
   assigns it and you write its description into `run_codes.json`.
3. Name the run from its root and branch points, and launch with `--run-name`.
4. After launch, `tools/scripts/run_codes.py --run <dir>` checks the name against the flags the run
   recorded and fails if they disagree.

### The old names

Directories launched before 2026-10-07 keep their old names, because research cites them by path;
[`../run_name_map.md`](../run_name_map.md) gives each one's name in this grammar, generated from
its metadata. The current tree (E6 and E7) is mapped; the archived E3–E5 ladders are not. The old
scheme, below, still parses for the lineages that carry it.

## The scheme before 2026-10-07: attempt numbers

A run is named

    <engine>-<attempt>-<seed>-<steps>

for example **`E3-10-21-80M`**: engine revision E3, its tenth configuration, seed 21, 80 million
steps. A continuation keeps everything and changes the budget: `E3-10-21-160M`.

- **engine letter** — bumped whenever `engine/` or `bindings/` changes the decision stream.
  Checkpoints from different engine letters may be *evaluated* together, but the result is a
  cross-engine comparison and must be labelled as one.
- **attempt number** — one row in [`../runs.md`](../runs.md), fixing observation, architecture,
  intervention and recipe. Numbered **from 01 within each engine letter**, so both halves of the
  prefix carry information. It does **not** include the seed.
- **seed** — its own field, the last two digits of the sampling seed (`21` is 20260921). Two
  runs of one configuration differ only here, so `E3-10-21` beside `E3-10-22` is visibly a seed
  pair while `E3-09-21` beside `E3-10-21` is visibly an intervention difference. The seed has
  repeatedly mattered more than the intervention, so it is worth being able to see at a glance
  which kind of difference a comparison is.
- **steps** — the budget the checkpoint was taken at, never omitted. Nothing is comparable
  across budgets.

The registry is the contract. A short name is only meaningful with it, so it lives in the
repository and is updated in the same commit as the run.

## A checkpoint is named by its run, never by its filename

Snapshot filenames collide **by construction**. The step counts a run snapshots at are a
deterministic function of its configuration, so two runs with the same `--snapshot-every-steps`
and batch shape produce byte-identically named files: `E4-01-01` and `E4-02-01` both contain a
`snapshot_150011904steps.pt`. The identity of a checkpoint is its **path**.

`tools/lib/checkpoint_id.py` is the one place that turns a path into a name:

    .../E4-02-01_20260919_040456/snapshot_150011904steps.pt  ->  E4-02-01@150M

Every tool that shows a checkpoint uses it, so a tournament row reads `E4-02-01@150M` rather than
`snapshot_150011904steps#2`. The old behaviour was not a crash -- `tournament.py` disambiguated
by appending `#1` and `#2` -- which is exactly why it went unnoticed for so long: the reports were
complete, and unciteable. A result that cannot be attributed to a run is not a measurement.

`unique_labels` **raises** on a genuine clash rather than suffixing. Two checkpoints that still
collide after being named by their run are the same file entered twice, or two runs sharing a
short name, and both are mistakes worth stopping for.

## The directory carries the name

A short name that lives only in a table is one someone has to look up. The directory is what
every later command quotes, so it is where the name has to be:

    data/checkpoints/E3-12-21_20260912_181622

`tools/train.py --run-name E3-12-21` builds it, validates it against the scheme, and records it
in `metadata.json` as `run_name`. **The steps field is deliberately absent from the directory**:
a lineage's budgets outlive any one directory — `p1_scalar_nofilter` holds 80M, 160M and 240M in
one, and on E4 each continuation leg is a new directory under the *same* short name (below) — so a
steps field in the name is a claim that goes stale the first time the run is continued. Each
snapshot's own filename, or for `snapshot_final.pt` its directory's `metadata.json`, carries its
budget.

Passing `--run-name` together with an `--output-dir` whose basename does not contain it is an
error rather than a preference, because the quiet version of that writes one arm's weights into a
directory named for another.

The directories predating the scheme are left alone: renaming them would break every path in
`research/` that cites one, which is a worse failure than an unhelpful name. The `directory`
column of [`../runs.md`](../runs.md) is what maps them back.

## Engine revisions

| letter | from | what changed in the decision stream |
|:---|:---|:---|
| **E1** | before `9fa5b05` | pre-starred-card fix: a starred card spent for Operations was deleted from the game. Arms A–G and everything older. Not runnable — the observation layouts they used are gone. |
| **E2** | `9fa5b05`, `341088e` | starred-card fix, observation v2.3. Arms H, H2, I. |
| **E3** | `76e7385`, `6381cf4` | the Aldrich Ames discard and the Star Wars pick made mandatory. Everything from P1 onward. |
| **E4** | `7057251`, `0aa3dc0` | P17: Grain Sales flattened to one decision, Missile Envy's starred-card removal fixed, action space 212 → 220. |
| **E5** | `b19b958`, `8d05d94` | influence Ops spent in full; each Ops modifier carries its own limit; an owed Event no longer inherits the Ops stop. |
| **E6** | `c1df6f5` + `c248bfb` | the era transitions at turns 4 and 8 no longer shuffle the discard back into the deck (rule 4.4), and Kitchen Debates spent for Operations is discarded. `c1df6f5` alone was briefly E6 and is abandoned; E6-01 and the first E6-02 ran on it. |
| **E7** | `acc68d6`–`9cd3ed9`, merged as `dec2768` (the 2026-09-28 rules audit) | seven rules fixes: war rolls count the defender's superpower, the region-bonus ladder falls back to an event's modified grant, UN Intervention is an Event only with a companion in hand, We Will Bury You settles on a trapped round, NORAD arms only on a move to DEFCON 2, U-2 Incident reaches UN Intervention through Grain Sales, Summit ignores the scoring-only card effects. Expected **not** to differ measurably from E6: every fix is on a rare situation. |

Each boundary is a *batch* rather than a commit — eight further engine and observation commits
landed with the starred-card fix on 2026-09-10 — and what each one changed, with what it
invalidated, is [`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md).
What a letter bump does and does not invalidate is
[`what_survives_an_engine_change.md`](what_survives_an_engine_change.md).

## Attempt numbers restart with each engine (the old scheme)

An attempt number is unique **within its engine letter**, not across the project. The first
arm on a new engine is `01`.

The first draft of this file numbered them globally, which put the E3 control at `E3-03` with
nothing before it on that engine — and made the letter redundant, since a global `03` already
implies E3. Restarting per engine keeps both halves of the name carrying information.

## A minor version: same game, another way of deciding

`E4.1` (P23) is E4 with an **opt-in action view**: the rules, `GameState` and observation are E4's,
and only how an agent may express one choice differs. A run in that view is named
`E4.1-<attempt>-<seed>`, with attempts numbered from 01 within `E4.1`. Because the game is the
same, **E4 and E4.1 numbers compare directly**, even in one tournament, without the cross-engine
label a letter bump needs. A change to the game itself still bumps the letter.

## When the letter does *not* bump

A commit that touches `engine/` bumps the letter only if it changes the decision stream the
agent sees. That is a claim to be measured, not assumed, and the measurement belongs in the
registry row beside the arm that depends on it. The E3-20/E3-21 pairing is the worked example:
`d124cff` and `09c2498` changed `engine/` between the two runs, and the letter was held at E3
because no `ROLL_DIE` node is ever handed to an agent (0 in 879 single-env and 12,800 vectorized
env-steps) and the runner always forces die 0. See [`../runs.md`](../runs.md), *E3-21*.

P14 is the second worked example and the stronger one: the mask/step collapse and the Missile
Envy forced-play fix (`0d2f499`) were replayed against a build of `b5f2c06` over 1,068 games and
385,812 steps under four policies, comparing outcome, length, chosen action and the legal mask's
hash at every step, with zero divergences. The letter was held at E3 on that measurement. Diffing
the **mask** as well as the action is the part worth copying — an engine change can alter what is
offered without altering what a deterministic chooser picks, and a policy sees the offer. Method:
[`../findings/engine/engine_change_decision_stream.md`](../findings/engine/engine_change_decision_stream.md).

## What the naming buys

The session that produced the first version of the registry rated nine E3 configurations against
**E2-02-480M** and called it "the strongest arm we have". That is still the right reference — it
is the strongest — but `E3-07-21-80M vs E2-02-21-480M` says on its face that the engine differs,
where `p1_scalar_filter_seed20260921 vs arm_H2_cont_240to480` did not.

Three rules follow, and all three were broken at least once before the scheme existed:

1. **Never compare across budgets.** The suffix makes a violation visible.
2. **Label cross-engine comparisons.** An E2 checkpoint evaluated on an E3 engine is playing a
   game it never trained on; the handicap is small for the two mandatory-choice cards but it is
   not zero, and it biases in favour of the E3 arm.
3. **A seed is a field, not a detail.** `E3-07-21` and `E3-07-22` differ only there and landed
   11.8 points apart on anchor win rate while being +3 Elo apart head to head. Giving it its own
   field is what makes that visible without opening the registry.

## Re-running the same seed: append a replicate index (both schemes)

The trailing number is the **seed**, so a re-run of an existing arm is not a new seed and must not
take the next seed number. It appends a replicate index instead:

```
E4-08-01        engine E4, attempt 08, seed 1
E4-08-01-2      the same arm re-run — seed 1, replicate 2
E4-08-01-3      seed 1, replicate 3
E4-08-03        a DIFFERENT seed (3), not a replicate
```

Getting this wrong makes a replicate look like a new seed, which corrupts exactly the statistic a
seed sweep exists to produce: `E4-08-07` and `E4-08-08` were launched as if they were seeds 7 and
8 when both were seed 1 re-runs, and read naively that would have reported a collapse rate of 1
in 7 rather than 1 in 5. Their directories were renamed to `E4-08-01-2` and `E4-08-01-3` on
2026-09-22, and they are recorded under those names in
[`../log/P21_M2d_country_head_collapse.md`](../log/P21_M2d_country_head_collapse.md).

A replicate is worth running when the question is reproducibility — whether a trajectory is
determined by its seeded starting conditions. A new seed is worth running when the question is
variance or rate. They are different experiments and the names should not blur them.

## Continuing an arm keeps its name; branching it adds a suffix (the old scheme)

**A continuation is not a new attempt.** Taking an arm further on its own seed changes only the
budget, so it keeps the short name and gets a new directory, `<short>_<timestamp>`, beside the
old one:

```
E4-08-03_20260919_223012     0 →  80M
E4-08-03_20260921_023139    80 → 160M   resumed from the 80M state, same seed
E4-08-03_20260922_202204   160 → 240M
```

**A snapshot is named by its steps**: `E4-08-03@160M`, never `@final`. A run's last snapshot is
`snapshot_final.pt` on disk, but "final" names a different budget in each directory of a lineage.
`tools/lib/checkpoint_id.py` therefore labels it by the budget recorded in that directory's
`metadata.json`.

**A branch is a resume under a different seed.** It is a different experiment from its parent
from the resume point onward, so it gets its own name: the parent, then `-<steps>M.<new seed>`.
The suffix repeats for a branch of a branch:

```
E4-17-06                seed 6
E4-17-06-50M.11         resumed from seed 6's 50M state under seed 11
E4-17-06-5M.11-90M.07   ...and that branch resumed at 90M under seed 7
```

The separator is `.`, not `_`: the directory is `<short>_<timestamp>`, so an underscore inside the
short name would make the directory ambiguous. `RUN_NAME_RE` in `ai/training/generic_trainer.py`
enforces the whole grammar:

```
E<engine>-<attempt>-<seed>[-<replicate>][-<M>M.<seed>]...
```

Before this rule was applied, continuations and branches were given fresh attempt numbers
(E4-11, 12, 19–22, 25). All of them were renamed into their lineages on 2026-09-22. The numbers
are retired rather than reused, because reports written before the rename still use them.
