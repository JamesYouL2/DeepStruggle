# Search headroom with a Gumbel root (2026-10-06): +60 Elo over the network at 64 evaluations, +10 more at 256; k = 4 at 16 matches PUCT at 128; first-play urgency does nothing

**Question.** How much play strength does honest (determinized) search add on top of the
strongest networks, and which root makes the most of a budget?

**Answer.** A noise-free Gumbel root (Danihelka et al., ICLR 2022: the k most probable moves,
narrowed by sequential halving) is worth about **+60 Elo** over the plain network at 64 network
evaluations a decision, and about +10 more at 256. It beats PUCT search of the same network by
about 20 Elo at half PUCT's evaluations (k = 4 at 64 against PUCT at 128), and at an eighth of them
(k = 4 at 16) it plays PUCT at 128 dead even. First-play urgency, 0.2 or 0, makes no difference
to either root. Three networks -- the two latest SWAs of the E7 line and E7-20-44's raw 4,800M
snapshot -- agree on all of it within their noise.

| player (honest search at every decision) | evaluations | Elo vs network, SWA 4,640-4,720M | SWA 4,720-4,800M | snapshot @4,800M | pooled score vs network |
|:---|---:|---:|---:|---:|---:|
| **Gumbel k=8 @256** | 256 | **+61** | **+61** | **+66** | **58.8%** (+61) |
| Gumbel k=4 @64 | 64 | +56 | +53 | +50 | 58.6% (+60) |
| Gumbel k=4 @64, FPU 0 | 64 | +55 | +50 | +52 | 57.4% (+52) |
| Gumbel k=4 @16 | 16 | +25 | +33 | +29 | 55.4% (+37) |
| PUCT @128, FPU 0.2 | 128 | +30 | +29 | +28 | 53.4% (+24) |
| PUCT @128 | 128 | +28 | +25 | +28 | 52.7% (+19) |
| the network, greedy | 1 | 0 | 0 | 0 | — |

Each column is one tournament: 7 entrants, 42,000 games, 1,000 games per side per pair, every
player greedy (temperature 0), Bradley-Terry Elo with the network at 0. The last column pools the
three tournaments' head-to-heads against the network (6,000 games, draws as half; ±0.6 points is
one standard error). The Bradley-Terry column and the head-to-head differ for the weaker searchers
because the fit also uses their games against the stronger ones.

The comparisons that carry the conclusions, pooled over the three networks (6,000 games each,
±0.6):

| | score | Elo |
|:---|---:|---:|
| Gumbel k=8 @256 vs Gumbel k=4 @64 | 51.5% | +10 |
| Gumbel k=4 @64 vs PUCT @128 (FPU 0 / 0.2) | 53.5% / 53.1% | +24 / +21 |
| Gumbel k=4 @16 vs PUCT @128 (FPU 0 / 0.2) | 50.0% / 49.9% | 0 / 0 |
| Gumbel k=4 @64 vs Gumbel k=4 @16 | 53.5% | +24 |
| Gumbel k=4 @64, FPU 0.2 vs FPU 0 | 49.6% | −2 |
| PUCT @128, FPU 0.2 vs FPU 0 | 49.7% | −2 |

Per network, reports in `data/reports/e7_gumbel_headroom_{swa4720,swa4800,snap4800}.{md,json}`.

## Method

* **Networks.** `data/checkpoints/_swa_line_ctl/E7line_swa_4640-4720M.pt` (sha256 `405b6de2…`),
  `E7line_swa_4720-4800M.pt` (`094a2207…`), and E7-20-44's raw
  `snapshot_4800053248steps.pt` (`7ca02cc3…`), the line at its 4,800M end.
* **The Gumbel root** (`ai/search/gumbel_root.py`, `BatchedMCTSConfig.gumbel_k`). At every decision
  it takes the k most probable moves (Gumbel scale 0), splits the budget over ceil(log2 k) halving
  phases, searches every surviving candidate's position with the ordinary batched search -- an
  equal share each -- and keeps the better half by
  logit + sigma(completed Q), with sigma and the completed Q as in mctx (c_visit 50,
  c_scale 0.1). Honest: each phase samples one world from the decider's side and searches the
  candidates' positions in it. Unlike Gumbel MuZero each phase searches a candidate afresh rather
  than growing one tree.
* **The budget is network evaluations**, as PUCT's is: a candidate's share of `per` is one search
  of its position with `per − 1` simulations, at most `simulations` candidates are taken, and a
  phase that cannot give every survivor one evaluation is not run. A Gumbel root at n never
  evaluates more positions than PUCT at n (`tests/training/test_gumbel_root.py`).
* **First-play urgency** (`BatchedMCTSConfig.fpu_reduction`, both trees): an unvisited move is
  valued at its node's value less r, from the mover's side.
* **Entrants**, as agent specs: the checkpoint itself, `search:<ckpt>:128:determinize:all::cpp:0`,
  `search:<ckpt>:128:determinize:all::cpp:0.2`, `gumbel:<ckpt>:16:4`, `gumbel:<ckpt>:64:4`,
  `gumbel:<ckpt>:64:4:0` and `gumbel:<ckpt>:256:8` (Gumbel's FPU defaults to 0.2).
* **The code the games were played with.** The root, its budget and the agent specs are this
  branch's. The search under them was the follow-up branch's (`gumbel-headroom-optimizations`),
  whose C++ tree takes a budget per tree, so a halving phase's candidates go into one search
  rather than one search per distinct share. That changes which leaves share a network batch and
  the order of the chance draws: the same algorithm and, statistically, the same player -- not the
  same games as this branch's code. The first three tournaments ran on engine fingerprint
  `ea4aea7a…`; the last two on that branch's tip, whose later engine changes (the observation's
  speed-up, LTO, the legal-mask cache) change no game.

## Replicate

```bash
C=data/checkpoints/_swa_line_ctl/E7line_swa_4720-4800M.pt
tools/scripts/check_engine_fresh.sh && PYTHONPATH=.:build/release .venv/bin/python tools/tournament.py \
  --models $C search:$C:128:determinize:all::cpp:0 search:$C:128:determinize:all::cpp:0.2 \
    gumbel:$C:16:4 gumbel:$C:64:4 gumbel:$C:64:4:0 gumbel:$C:256:8 \
  --games-per-side 1000 --temperature 0 \
  --anchor-model _swa_line_ctl@E7line_swa_4720-4800M --anchor-elo 0 \
  --output-report gumbel_headroom.md --output-json gumbel_headroom.json
```

## Budgets on the newest networks (2026-10-06): 256 is worth +12 ± 3 Elo over 64

The three budgets again on the newest single checkpoint, E7-20-44-4390M.46 at 4,800M (sha256
`84f05bb4…`), and the soup of E7-20-44 and its two seed branches at 4,800M
(`soup_E7-20-44+4390M.45+4390M.46_4800M.pt`, `ef6f6384…`): the network and Gumbel k=4 @16, k=4 @64
and k=8 @256, 12,000 greedy games each, 1,000 per side per pair (reports
`data/reports/e7_gumbel_budget_{4390M46_4800,soup4800}.{md,json}`). Head-to-head Elo, draws as half, ±1 standard error, with the three networks above:

| network | k=8 @256 vs k=4 @64 | k=4 @64 vs k=4 @16 | k=8 @256 vs network | k=4 @64 vs network | k=4 @16 vs network |
|:---|---:|---:|---:|---:|---:|
| SWA 4,640-4,720M | +8 ± 8 | +41 ± 8 | +56 ± 8 | +71 ± 8 | +30 ± 8 |
| SWA 4,720-4,800M | +8 ± 8 | +23 ± 8 | +68 ± 8 | +55 ± 8 | +41 ± 8 |
| E7-20-44 @4,800M | +15 ± 8 | +9 ± 8 | +60 ± 8 | +55 ± 8 | +41 ± 8 |
| 4390M.46 @4,800M | +13 ± 8 | +24 ± 8 | +76 ± 8 | +78 ± 8 | +39 ± 8 |
| soup @4,800M | +14 ± 8 | +13 ± 8 | +69 ± 8 | +45 ± 8 | +24 ± 8 |
| **pooled, 5 networks** | **+12 ± 3** | **+22 ± 3** | **+66 ± 4** | **+61 ± 4** | **+35 ± 3** |

Every network puts k=8 @256 above k=4 @64, by 8 to 15 Elo: the larger budget is a small, real
gain, at four times the network evaluations. Search is worth
less over the soup at 64 (+45, against +55 to +78 elsewhere) and about the same at 256; one
network, inside two standard errors of the others' spread. Each column is relative to that
network itself -- these say nothing about how the networks compare with each other.

## The fork's first measurement (2026-10-05)

The question and the root came from the DeepStruggle fork, which measured it first on its own CI
(runs `37327142506` and `37318435430`) on `C2_soup+A.pt`, the shallow soup
([`E7_shallow_soup.md`](E7_shallow_soup.md)) averaged with E7-20-44's 2,720-2,800M weights (sha256
`5cbb77ef…`, fork release `e7-20-44-soups`), with its own code -- `search:` entrants and the three
settings as overrides -- not the `gumbel:` spec above. It reported Gumbel k=8 @256 at +79 Elo over
the network, k=4 at 16 *beating* PUCT at 128 (52.1%), k=4 gaining nothing past 64 and k=16 no
better than k=8 at 256.

Remeasured here on three later networks, with the root as merged:

* **The headline is smaller:** +61 against +79. That run sampled the network at temperature 0.1 and
  played the searchers greedy -- a small handicap for the network (greedy and 0.1 were measured the
  same player on an E3 network,
  [`P15_temperature_selfplay.md`](../archive/E3_ladder/log/P15_temperature_selfplay.md)) -- and
  it was a different network.
* **k = 4 at 16 ties PUCT at 128** (50.0%) rather than beating it.
* **256 is worth a little over 64:** +10 Elo for k=8 @256 over k=4 @64, about 2.5 standard errors
  -- the fork's "k = 4 stops gaining at 64" holds for k = 4, but the larger root does gain.
* **FPU 0.2 was part of every fork Gumbel entrant**, and the PUCT comparison's FPU was not recorded;
  here it is separated, and worth nothing either way.
* The fork's budget did not count each candidate search's own root evaluation (k=4 @16 cost 22
  evaluations a decision, PUCT @16 17); here it does.

## What this does not say

* Search as a training target is a separate question: on the fork, fine-tunes toward the visit
  counts or toward this root's choice both lost to their unsearched control.
* k = 16, and k = 8 at 128, were not remeasured.
* Every player was greedy. Whether the gains hold against a sampling opponent was not measured.
