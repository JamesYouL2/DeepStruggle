# findings/ — what is true now, split by what can invalidate it

Short, current, cited. **Rewritten in place**: a finding states the conclusion and links to the
append-only [`../log/`](../log/README.md); when a number is superseded it is replaced, and the
narrative stays in the log.

| | | invalidated by |
|:---|:---|:---|
| **[`engine/`](engine/README.md)** | what the simulator does — rules, mask semantics, revision boundaries | a code defect, a rules fix, a probe found lying; its numbers are then gone |
| **[`training/`](training/README.md)** | what a training choice is worth — architecture, length, collapse, evaluation | a better-powered replication, more seeds |

The split rests on the owner's assumption that engine changes make old arms incomparable but (after
the starred-events fix) do not change the *relative* effect of training approaches. What that
licenses: [`../method/what_survives_an_engine_change.md`](../method/what_survives_an_engine_change.md).

Elsewhere: open defects in `BUGS.md`; rules and design in `engine/AGENTS.md`; instruments that
lied in [`../log/measurement_bugs.md`](../log/measurement_bugs.md) (checklist:
[`../method/measurement_pitfalls.md`](../method/measurement_pitfalls.md)); the human corpus in
[`../log/P7_replayer_conversion.md`](../log/P7_replayer_conversion.md) and
[`../method/human_play.md`](../method/human_play.md); E3-era findings in
[`../archive/E3_ladder/findings/`](../archive/E3_ladder/README.md).

**Where the split is not clean:** the observation is neither (an owner-held representation choice
that invalidates checkpoints outright); search results are training questions whose numbers were
twice invalidated by code fixes ([`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md));
seed spread is a training property but tournament noise (~1.5 points between identical runs) is an
apparatus one; the DEFCON-blunder findings rest on the engine's `DEFCON_SUICIDE_PROVOKED` taxonomy.
