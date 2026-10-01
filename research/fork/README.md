# Fork notes (JamesYouL2/DeepStruggle)

Research done on the fork, kept apart from the upstream record so the two never conflict:

* **Only new files, only in this directory.** Nothing here edits an upstream index
  (`../questions.md`, `../runs.md`, `../log/README.md`, `../plans/reserve.md`). When a result here
  is worth upstreaming, it goes up as its own PR that adds the index lines then.
* Links out of this directory point only at files that exist on the branch carrying the note;
  code on another branch is cited by branch and commit, not linked, so `tests/docs` stays green.
* Same conventions as the rest of `research/`: run ids, commits and engine fingerprints for every
  number, the verdict vocabulary of [`../questions.md`](../questions.md), and *what this does not
  say* sections.

| note | question | status |
|:---|:---|:---|
| [`hungary_openings.md`](hungary_openings.md) | Is the USSR's Poland 3 / Hungary 3 opening a mistake, and does a stronger US punish it? | interim — search-64 and 4,096-deal runs pending |
| [`determinization_targets.md`](determinization_targets.md) | Is the training searcher's one-world target overdetermined by the world it sampled? | done, one model |
| [`gumbel_root.md`](gumbel_root.md) | What did the Gumbel root (`feat/mcts-gumbel`) buy at play time? | done, 512 games a run |
| [`search_for_rl.md`](search_for_rl.md) | Would deeper search make a better RL teacher? | analysis, not run |
