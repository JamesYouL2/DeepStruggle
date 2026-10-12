# The heads soup's blunder bank (`ai/eval/banks/blunders_E7.jsonl`, 35 positions), scored with the R34 soup

Repeated: the player plays the heads soup's confirmed blunder. Better: the confirmed better move. Other: anything else, paid out by the heads soup's greedy play over 256 pairs against the blunder (seed 11), as in the bank's own score table. Points: the blunders' confirmed cost (the heads-soup judge) summed over the rows the player repeats.

| player | repeated | better | other | points repeated | other vs blunder |
|:---|---:|---:|---:|---:|---:|
| heads soup, greedy | 35 | 0 | 0 | 9.11 | — |
| R34 soup, greedy | 14 | 9 | 12 | 4.42 | +0.04 (12) |
| R34 soup, Gumbel k=8 @256 | 6 | 18 | 11 | 1.13 | +0.10 (11) |
| R34 soup, rollout root (`4:16:4:z2`) | 7 | 13 | 15 | 1.56 | +0.10 (15) |
