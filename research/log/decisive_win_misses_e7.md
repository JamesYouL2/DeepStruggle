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
400·log10((0.5 + Δ)/(0.5 − Δ)) against the model itself. This is exact, not an approximation: taking a
forced win ends the game, and in greedy play the win-taker and the model play the same game up to
the win-taker's first missed chance, where it wins instead. (Exact for classifier v2, whose wins are
certain on every die face; v1 sampled dice, so a few of its "wins" may not be certain.)

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
* The owner pointed out that an earlier "first order" caveat here was wrong (taking the win ends the game).
* Raw per-shard dumps: `data/logs/temp/decisive_cost/`.

## Does the new classifier see every win the old one saw? (owner's question) -- no, ~3%

Per decision, on the identical greedy games (keyed by the decision's index among the game's
non-chance decisions):

| | E7-02-44 | soup |
|:---|---:|---:|
| decisions offering a win, v1 / v2 | 7,315 / 13,938 | 7,716 / 14,644 |
| **v1 win, v2 none** | **226 (3.1% of v1)** | **236 (3.1%)** |
| both see a win, but some v1-winning action is not a v2 win | 11 | 13 |
| v2 win, v1 none | 6,849 | 7,164 |

v2-only: mostly the card choice that starts a winning line (3,827 on E7-02-44), Wargames at the
play-mode decision (682), the China Card, Lone Gunman, Grain Sales, Star Wars -- what PR #6 added.

**The v1-only positions** (19 replayed: E7-02-44, shard 0, first 512-game batch, base seed 77,000;
`tools/scripts/decisive_v1_only_replay.py`), v1 re-run under 12 RNG seeds, and the v1-winning and
other actions played out 200 / 40 times with random moves and dice:

| what it is | count | who is right |
|:---|---:|:---|
| v1's "win" changes with the RNG seed -- a sampled-dice artefact | 9 | v2 |
| a turn-ending position where every action wins (all legal actions v1-wins, or the others also won 200/200) | 6 | neither matters -- no choice to make; v2 counts nothing there either way |
| **a real forced win v2 does not see** | **4** | **v1** |

The four real misses: all USSR to move in action round 7 at DEFCON 2, the game ending before the US
moves again -- one influence placement on turn 4 at VP −19 (game 406, decision 220: the one
v1-winning placement won 200/200, the others ~82%), and three play-mode choices on turn 10 (games
170, 234, 449: 1 winning mode of 5, won 200/200; the others 70–78%). Scaled up, ~1 in 5 of the
v1-only positions, i.e. ~45 per 8,192 games, about 0.7% of v2's chances -- too few to move the cost
estimate, but a gap in the classifier at the end of a turn.
