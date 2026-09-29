# P27 — close the network's gap to its own search, with CI compute

**Status: proposed (2026-09-29).** Written so a session with no context can pick it up. Everything
here is CPU-bound and runs on GitHub Actions (`.github/workflows/tournament.yml`), not the GPU box;
only step 4 needs a GPU, and step 8 is browser code.

## Where it starts

[`../log/E5_search_budget_sweep.md`](../log/E5_search_budget_sweep.md): the weaker player,
**raw `E5-11-43@560M`**, loses to its own 64–128-simulation determinized search 58/42 (+52 Elo,
one checkpoint, greedy). The goal in [`README.md`](README.md) is a strong *no-search* player, with
search acceptable as a diagnostic and a distillation source. So the ~55 Elo is the target size,
and the questions are which of the cheap wrappers recovers some of it without search, and what the
search knows that the net does not.

Ready on branch `feat/e5-wrappers` (commits `70c1296`, `d737521`, `1bf4631`, `d909d98`), **[PR
#6](https://github.com/JamesYouL2/DeepStruggle/pull/6) opened 2026-09-29, not yet merged**:
`safe:`, `ensemble:`, `temp:` and `search:` specs
(`tools/lib/player_agent.py`, `tools/lib/safety.py`, `tools/README.md` §D), a `temperature` input
on the workflow, and the visit tie-break.

## How to run any of these

```bash
gh workflow run tournament.yml --ref feat/e5-wrappers \
  -f models="newest safe:newest" -f games_per_side=512 -f runners=32 -f shard_pairs=2 -f temperature=0
gh run download <id> -n tournament-<id> -D data/reports/ci      # tournament.md / .json / games.jsonl / provenance.txt
```

* `newest` = newest `.onnx` in `mihaild/deepstruggle`, `hf:<file.onnx>` = that file; both resolve
  inside any spec. **Pin `hf:E5-11-43_560M.onnx`** rather than `newest` once a newer export lands,
  or the comparison silently changes.
* **Always `temperature=0`** for a comparison with a search entrant (default is 0.1).
* **512 games a side puts ±22 Elo on a point.** Do not read anything below that. 1,024 a side for a
  claim about a small effect (`safe:` is one).
* Record the run id, commit and fingerprint from `provenance.txt` in the log entry; a later
  reader cannot re-run a CI job.
* Timing: 16 runners × 200 games of a pure network ≈ 8 min; search-64 at 512 a side ≈ 2h15 on 16
  runners, search-128 ≈ 3h10 on 32. Free standard runners, 20 concurrent jobs, so one large job
  at a time.
* Several dispatches on one branch run independently (no concurrency group), which is how the
  sweep ran seven at once.

## Steps, in order

1. **Open the PR** for `feat/e5-wrappers` and merge, so the workflow inputs exist on `main`.
   (PR #6 is open; the merge is the half that remains.)
2. **Safety layer alone** — `safe:hf:E5-11-43_560M.onnx` against the raw net, T=0, 1,024 a side.
   The goal probes found the networks taking a certain win only 59–70% of the time and losing to
   their own DEFCON choice in 1–8% of games; the sweep still shows ~3% own-DEFCON losses. Ceiling
   if it cured all of them: ~3 points. **Prediction: +5 to +15 Elo; if it is under 5, drop it.**
3. **Ensemble** — `ensemble:hf:<a>.onnx+hf:<b>.onnx` against its best member, for two or three
   E5/E4 exports on the same engine (only same-observation-width exports load; check
   `tools/inspect_checkpoints.py`). Members must differ by seed, not by an engine bump. Cost: n×
   inference, so it is a *product* option, not a distillation one.
4. **Search as a training target (GPU; this is the real prize)** — the E4-28 line
   ([`../log/E4_search_distillation.md`](../log/E4_search_distillation.md)) distilled 64-sim
   visit counts online during RL and gained +60 to +194 Elo over a step-matched control, on
   another network. The sweep says the ceiling for this checkpoint is ~55 Elo at 64 sims and no
   more at 128, so **use 64, not 128 or 256.** Launch with `tools/scripts/launch_flags.py
   <healthy E5 run> --diff <new run>` (`CLAUDE.md` invariants 14–15: opponent pool, and
   `ref_update_freq`). Watch the collapse tripwires in that log.
5. **Where does search win?** Play search-64 against the net on a fixed set of ~2,000 openings and
   bucket the games it wins by ending (see the table in the sweep log) and by the decision that
   swung it (`games.jsonl`). Target: Europe Control as US and provoked-DEFCON escapes as USSR. Any
   pattern here is a probe for `ai/eval/`, and a candidate for a targeted auxiliary loss.
6. **Ladder placement** — `tournament.py --models newest search:newest:64:determinize doctrine
   heuristic_v2` with the Elo anchor, 200 a side, so the sweep's pair-internal Elo can be quoted
   on the ladder's scale.
7. **Doctrine as the third opponent.** Doctrine (`bot/doctrine`, 40-0 against HeuristicV2, ~20 s
   a game per core) was played against `newest` once in a 200-game CI run, whose result is not
   recorded in `research/`. Repeat at 512 a side with search-64 and record it; it is the only
   opponent that reasons about the board with a different method.
8. **A searcher in the page — diagnostic and teacher, not the product.** Independent of steps
   1–7, any time. The GitHub Pages workbench already carries every primitive the searcher needs:
   `WasmEngine.snapshot()/restore()` (the 4 KB trivially-copyable `GameState`),
   `step`/`mask`/`observation`/`decodeFlat`, and `Model.run(obs, masks, rows)` — the batched
   prior+value interface `ai/search/batched_mcts.py` calls. What is missing is a TypeScript port
   of the search loop. **The alignment is fixed before it is written:** the product is the
   no-search network (the goal above), and search in the page exists only as a diagnostic and a
   distillation teacher — (a) a human can play against search on the Pages deployment, and (b)
   the page can show where search disagrees with the raw net, which is step 5's question asked
   interactively. **64 simulations, not more**: the knee of the sweep's curve, and 128 buys
   nothing. The measuring instrument stays the CI wrapper
   `search:hf:E5-11-43_560M.onnx:64:determinize`; the page copy is a port held to it.

   The easy version, deliberately:

   * **Determinization by save-JSON round trip.** `ts_save_json()` / `ts_load_save_json()` carry
     card locations, and `HAND_*_KNOWN` vs `HAND_*_UNKNOWN` marks what we have seen — so
     reshuffling the hidden pool needs no C++ change and no `build_web.sh --engine` rebuild. A
     native export is a later optimisation, not a starting requirement.
   * **One position at a time, N determinized trees** (dmcts's shape: an independent tree per
     sample, root visits summed). N leaves per iteration make one N-row forward through
     `Model.run`, and no virtual loss is needed.
   * **The singleton state is the searcher's.** The WASM engine holds one `g_state` shared with
     the session: snapshot the root, search, restore before returning — and run between frames or
     in a Web Worker (a worker needs no COOP/COEP headers, which Pages does not send; without
     them onnxruntime-web stays single-threaded, which at 64 sims is fine).

   Acceptance: it plays legal moves at 64 sims in the Pages build and agrees with the CI
   wrapper's pick on a fixed set of positions — a disagreement is a port bug until proven to be
   determinization sampling. Its numbers, like every pair-internal search number, do not enter
   [`../checkpoints.md`](../checkpoints.md).

## Traps found so far

* **A search entrant with a low budget can be worse than its prior** — the tie-break bug. Any new
  wrapper around `BatchedMCTS.best_actions` must pass `test_search_tiebreak.py`.
* `search:` is a *searcher's* argmax: `temp:` does nothing to it, so a temperature-0.1 net
  against search is not a same-temperature comparison. Run the whole matchup at `temperature=0`.
* An unpinned `newest` moves under you. Read `provenance.txt` for the file that actually played.
* The mirror is 45% US / 55% USSR for this net. A seat win rate means nothing without the mirror's.
* A CI part that failed is missing from the pool; `provenance.txt` prints "parts received: N". If
  N is below `runners`, rerun; the summary does not fail on it.
* Do not run any of this locally: 6 cores and ~5 GB RAM (`pytest -n auto` was OOM-killed).

## What would settle it

The gap is *closed* if a no-search checkpoint distilled from search-64 beats `E5-11-43@560M` by
≥ 25 Elo at 1,024 a side, T=0, and does not lose more than ~20 Elo to Doctrine. It is *not worth
pursuing* if step 4's arm matches its step-matched control within the same 22 Elo interval.
