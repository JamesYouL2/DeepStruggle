# The blunder bank (2026-10-10): 37 positions where the E7 network's own move costs 10+ points of a game, confirmed twice

**Question (owner).** Where is the network clearly making a move that costs a large share of the
game? Keep those positions, so any later network or searcher can be checked against them.

**Answer.** The bank holds 37 positions (`ai/eval/banks/blunders_E7.jsonl`). In each one, a
fresh paired measurement by two judge networks puts the heads soup's greedy move 10+ points of
the mover's score below a fixed alternative, at 3+ standard errors under both. 20 of them cost
20+ points.

* **The census's estimated regret was mostly selection.** The 487 rows the
  [census](E7_blunder_census.md) measured at 10+ points (z ≥ 2, 32 pairs) averaged 21.8 points
  when selected. Re-paid from a fresh seed at 512 pairs, they average 3.8 points (soup judge) and
  2.9 (R32 judge), and 37 survive both. Of the census's 89 random-control rows on the shortlist,
  2 survive.
* **Where the bank's blunders are.** 24 of the 37 are in turn 10. 9 of those are influence
  placements, 6 of them in the last action round before final scoring, as the census found. The
  rest are mostly card choices:
  * headlines (Europe Scoring at turn 3, Middle East Scoring at turn 9, Korean War at turn 3);
  * a scoring card played early or late;
  * the event not taken where it wins: Wargames for the event, Tear Down this Wall and
    Iran-Iraq War played for Ops at turn 10;
  * the China Card's timing, both spent and held at the wrong moment.
* **Who repeats them.** Each player's move on the 37 positions. A move that is neither the
  blunder nor the better move is paid out against the blunder by the soup's continuations, 256
  pairs:

| player | repeats the blunder | finds the better move | other move (its lead on the blunder) | points repeated |
|:---|---:|---:|---:|---:|
| heads soup, greedy | 37 | 0 | 0 | 10.3 |
| R32's SWA, greedy | 25 | 7 | 5 (−0.17) | 7.2 |
| heads soup, Gumbel k=8 @256 | 12 | 18 | 7 (+0.10) | 4.2 |
| R32, Gumbel k=8 @256 | 5 | 23 | 9 (+0.15) | 1.3 |
| heads soup, rollout root (`4:16:4:z2`) | 17 | 18 | 2 (+0.28) | 4.7 |
| R32, rollout root | 9 | 23 | 5 (+0.17) | 1.9 |

* R32 raw avoids a third of the soup's blunders, but its other moves there are worse still (−0.17
  on the blunder). Under search it repeats the fewest.
* **The rollout root repeats more than Gumbel here (17 against 12) though it is the stronger
  player** (+97 Elo over Gumbel@256 on the soup). Its `z2` rule keeps the network's move unless an
  alternative leads by two standard errors over its own 16 worlds.
* **The bank favours Gumbel by construction.** 22 of the 37 better moves are the census's own
  Gumbel@256 pick, because Gumbel nominated the alternatives the census paid out. Compare players
  other than Gumbel with each other; Gumbel's figure is an upper bound.

## Method

Tool `tools/scripts/blunder_bank.py` (`select`, `confirm`, `build`, `score`; `tools/README.md`
§7d). Workflow `bank_playouts.yml` with `stage=confirm`.

* **Shortlist (`select`).** The census's validation rows (CI `37975392781`, relabelled after the How
  I Learned fix): every row whose best measured alternative led greedy by 0.10+ at z ≥ 2 over 32
  pairs. That is 487 rows, 89 of them from the random control, with 2–5 measured moves each. The
  alternative is fixed here, before confirmation. Release `blunder-shortlist-20261010` on the fork,
  sha256 `abb0c6cc…`.
* **Confirmation (`confirm`).** Every measured move of every row paid out again, 512 pairs, from
  seed 7. The census used seeds 0 and 1, so the redeals and dice are fresh. Two judges, one CI run
  each (20 runners):
  * the heads soup `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt` (sha256 `91a43a8b…`): run
    `38086939394`, 137 runner-minutes;
  * R32's SWA `E7-A8-R1-S44@6400M+R32_20261010_005623/swa_6720-6800M.pt` (sha256 `1d4f65f0…`): run
    `38086945497`.

  Both ran on commit `ce34fc4`.
* **Bank (`build`).** Rows whose confirmed cost of greedy against the fixed alternative is 0.10+
  at z ≥ 3 under both judges, ordered by the smaller of the two. Under the soup alone, 56 clear
  the bar; under R32 alone, 41.
* **Score.** Local, `check_engine_fresh` `6c036593…`. Searchers are reseeded per position from
  the row id, so a score does not depend on the order the bank is read in.

The 37 positions with each judge's cost and a workbench link: [bank](E7_blunder_bank/bank.md);
the score table as written: [score](E7_blunder_bank/score.md).

## What this does not say

* **It is not a rate.** The shortlist came from 200 greedy games of one network through a screen
  (forced-loss labels and a Gumbel @32 disagreement) and a 32-pair estimate. Missed blunders
  are not counted, so 37 is a floor on 200 games, not an estimate of a population.
* **The cost is judged by greedy continuations.** A move whose value lies in a follow-up the
  judges do not play is undervalued. Two judges that are near relatives of each other do not
  remove that. A blunder here is confirmed against what these networks would go on to do.
* **The better move is the best one measured, not the best one.** "Other" moves are paid out
  only against the blunder, so they can beat the bank's better move.

## Use

```bash
export PYTHONPATH=.:build/release
python tools/scripts/blunder_bank.py score --bank ai/eval/banks/blunders_E7.jsonl \
    --player new=<ckpt.pt> new-g256=gumbel:<ckpt.pt>:256:8 --judge <soup.pt> --pairs 256 --out score.md
```

A few minutes on a laptop for a greedy player or Gumbel @256. A new network that repeats fewer
of the 37 has not shown it is stronger: its other moves can be worse, as R32's are. Read the
"other" column with the count.
