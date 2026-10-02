# Missed forced wins: E7-02-44@1,200M and the shallow soup, old and new classifier (2026-10-02)

Owner: "check for the E7-02-44@1200M and soup self-plays -- how often they miss decisive win
according to old and new classifier, and how much Elo does it cost them."

**Method** (`tools/scripts/decisive_cost.py`). 8,192 greedy self-play games per model (the setting
tournaments rate in), every decision classified by `ai.eval.decisive_probe.classify_in_view`:
version 1 from this branch, version 2 from main (PR #6: a win is any line of the mover's own
choices and dice that wins on every face, and each chance to win counts once however many
decisions it takes). Greedy play makes the games identical under both classifiers. **Cost:** a
side that missed a forced win and then lost the game is a game a player that always takes its forced
wins would have won in that seat; Δ = mean over both seats of P(missed a win and lost), Elo =
400·log10((0.5 + Δ)/(0.5 − Δ)) against the model itself. First order -- it ignores how taking a win
earlier would change later play.

| | chances to win per game | take rate | games where a side missed a win | ... and lost it | Δ win rate | Elo |
|:---|---:|---:|---:|---:|---:|---:|
| E7-02-44, classifier v1 | 0.89 | 0.57 | 26.3% | 1.8% | 0.0089 ± 0.0010 | **+6** |
| E7-02-44, classifier v2 | 0.79 | 0.45 | 24.1% | 3.0% | 0.0150 ± 0.0013 | **+10** |
| soup, classifier v1 | 0.94 | 0.55 | 26.9% | 1.7% | 0.0086 ± 0.0010 | **+6** |
| soup, classifier v2 | 0.83 | 0.45 | 25.4% | 3.2% | 0.0159 ± 0.0014 | **+11** |

Per seat (v2): E7-02-44 US 3,876 chances / 2,068 missed, USSR 2,610 / 1,475; soup US 4,287 / 2,426,
USSR 2,550 / 1,345. The new classifier finds fewer chances on the US side (v1 counted multi-decision
wins several times) and more on the USSR side (coups in the opponent's round, turn-end wins).

* **They miss about half their forced wins** -- a take rate of 0.45 under the new classifier, 0.55
  under the old -- and the soup and the raw snapshot are the same.
* **Most misses cost nothing:** in nearly 9 of 10 games where a side missed a win it won anyway (the
  chance usually recurs, or the game is won another way).
* **The cost is ~10–11 Elo under the new classifier, ~6 under the old** -- the new one counts more
  of the wins that are actually lost, so its cost is about 1.7× larger.
* Raw per-shard dumps: `data/logs/temp/decisive_cost/`.
