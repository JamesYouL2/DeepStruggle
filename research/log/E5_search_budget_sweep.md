# E5-11-43@560M against its own search — the budget sweep (2026-09-29)

**Question.** The published network (`E5-11-43_560M.onnx`, Hugging Face `mihaild/deepstruggle`) is
the weaker player of any pair it forms with itself plus search. How much does honest determinized
MCTS, using that same network as prior and value, add, and where does the curve flatten?

**Answer.** Nothing detectable below ~16 simulations, **+28 Elo at 32, +52 to +54 at 64 and 128**,
and the last doubling is flat. Search never plays worse than the raw network once the tie-break
bug below is fixed. The no-search network is the product ([`../plans/README.md`](../plans/README.md)
"The current goal"), so this is a *distillation ceiling*: about 55 Elo is what a search-derived
target could teach it.

## Setup

* Engine fingerprint `dd7544681737…` (`provenance.txt` of every run). Entrants: the raw ONNX
  export, and `search:<same .onnx>:N:determinize` (`ai/search/batched_mcts.py`, ONNX Runtime, CPU).
* `tools/tournament.py`, **512 games a side** (1,024 per point), `--temperature 0` for every
  network entrant (a search entrant always plays its argmax, so the raw net is the greedy net),
  `--device cpu --workers 0`, spread over 16–32 GitHub Actions runners
  (`.github/workflows/tournament.yml`, `shard_pairs` 1–2).
* Elo is the head-to-head pair only, anchored so the two average 1500. **One standard error is
  about 1.6 points of win rate, ~11 Elo; a 95% interval is ±22 Elo.** Anything under ~20 Elo below
  is inside it.
* `E5-11-43@560M` against itself, both at temperature 0 (run `36552439732`), is the noise floor
  and the seat baseline: **511–511–2**, 45.3% as US / 54.5% as USSR. **The net wins more as USSR
  even against itself**, so compare each seat with its own baseline, not with 50%.

## Result

Search win rate is 1 − the row's raw-net win rate. Runs on commit `d909d98` (tie-break fixed)
where one exists; `1bf4631` otherwise, marked †.

| simulations | raw net won | **search won** | Elo Δ | raw net as US / as USSR | Actions run |
|---:|---:|---:|---:|---:|:---|
| 0 (mirror, `temp:0:`) | 49.9% | — | 0 | 45.3 / 54.5 | 36552439732 |
| 2 | 47.7% | 52.3% | +13.9 | 42.6 / 52.7 | 36576388456 |
| 4 | 47.9% | 52.1% | +11.2 | 41.4 / 54.3 | 36576392867 |
| 8 | 46.4% | 53.6% | +23.4 | 40.8 / 52.0 | 36576396916 |
| 16 † | 47.7% | 52.3% | +14.9 | 43.4 / 52.0 | 36560496705 |
| 32 † | 45.7% | 54.3% | +28.2 | 41.8 / 49.6 | 36560500465 |
| 64 † | 41.8% | **58.2%** | **+53.7** | 34.2 / 49.4 | 36572795128 |
| 128 | 42.2% | **57.8%** | **+51.6** | 39.1 / 45.3 | 36591849067 |

† taken before the tie-break fix. It matters only at tiny budgets (see below); 4 and 8 were
re-run on both commits and moved by 0.2 and 0.4 points, so 16/32/64 are not in doubt on that
account.

Wall time on 16–32 runners: 2 sims ~16 min, 8 sims ~1h15, 64 sims 2h16, 128 sims 3h10. Search cost
grows roughly linearly with the budget, the gain does not, so **64 simulations is the knee**.

### Where the net loses to search (128 sims, 1,024 games)

Losses by the raw net, US seat / USSR seat, against the same seat in the temperature-0 mirror:

| ending | US seat, vs search128 | mirror | USSR seat, vs search128 | mirror |
|:---|---:|---:|---:|---:|
| 20 VP | 47.7% (148) | 49.8% | 42.9% (117) | 41.4% |
| final scoring | 28.4% (88) | 30.5% | 38.8% (106) | 33.2% |
| DEFCON 1, opponent's decision | 11.9% (37) | 10.8% | 12.8% (35) | 21.1% |
| DEFCON 1, own decision | 2.9% (9) | 3.2% | 3.3% (9) | 2.6% |
| Europe Control | 9.0% (28) | 5.7% | 1.1% (3) | 0.4% |

Search does not remove the net's own-DEFCON losses (still ~3%): the searcher shares the value
head that misprices them. It does **win more often on Europe Control as US** (9.0% vs 5.7% of the
net's losses) and turns *fewer* net losses into DEFCON-1-by-opponent as USSR (12.8% vs 21.1%),
i.e. the searcher finds the escape the network does not. Counts are 3–37 per cell; read these as
leads for a probe, not findings.

## The tie-break bug (found by this sweep)

`BatchedMCTS.best_actions` took the first index among tied root visit counts. At 2 simulations a
tie is the normal case, so the searcher played whichever legal move the engine listed first:
**pre-fix, search-2 scored 52.0% for the raw net (search 48.0%, run `36571864834`)** — a search
*worse* than its own prior. Commit `d909d98` breaks ties by the mover's mean value, then the prior,
so **1 simulation plays exactly the net's argmax**, and search-2 became 52.3%. Unit tests:
`tests/training/test_search_tiebreak.py`. Anything measured with `best_actions` at a small budget
before `d909d98` is suspect; nothing else in this file is.

## What this does not say

* One checkpoint, one engine, greedy play. Whether the gap is the same for a temperature-0.1
  net (the CI default before `1bf4631`; the two are the same player by
  [`../archive/E3_ladder/log/P15_temperature_selfplay.md`](../archive/E3_ladder/log/P15_temperature_selfplay.md))
  or for another checkpoint is untested.
* Not run against Doctrine, HeuristicV2 or an anchor: this is pair-internal Elo, not the ladder's
  scale, so **do not add it to `../checkpoints.md`**. To place the 64-sim player on the ladder it
  has to play the anchor field.
* Search here sees a determinized state, never the opponent's hand, so unlike
  `HeuristicMCTS16` it is an honest player.
* Neither seat's gain is separable from noise at 512 games (US +2 to +11 points, USSR +0 to +9
  points at 64–128 sims).
