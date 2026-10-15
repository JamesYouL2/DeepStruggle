# Where the searched A13+R48 player still loses (2026-10-15): the 12,000M snapshot under the rollout root, judged by full-game playouts

**Question (owner).** The R39 entry ([E7_searched_leak_R39](E7_searched_leak_R39.md)) measured where
R39 under the rollout root `4:32:4:z2` still loses. The same census, on A13+R48.

The network is `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13_20261014_105711/snapshot_12000034816steps.pt`
(sha256 `0cee010a74f1…`), the 12,000M snapshot, not an SWA. The player is that network under the
rollout root `4:32:4:z2`, as for R39.

**Answer.**

* **The searched player still loses at least +96.9 ± 12.9 points a game.** This is a lower bound.
  R39's was +86.6 ± 8.4, so the two intervals overlap and this does not say A13+R48 is worse.
  It is found at 11.3% of the decisions (1,073 of 8,000 bank positions); the rest count as no loss.
* **Card choice is the largest leak and the one that grew.**

  | decision kind | A13+R48 | R39 | share of A13+R48 |
  |:---|---:|---:|---:|
  | action-round card choice | **+50.4 ± 11.1** | +36.6 ± 4.8 | 52% |
  | play mode (event / Ops / space) | +16.9 ± 3.3 | +10.7 ± 2.7 | 17% |
  | influence placement | +16.3 ± 5.1 | +25.7 ± 5.8 | 17% |
  | headline | +7.6 ± 1.1 | +8.7 ± 1.1 | 8% |
  | coup target | +2.8 ± 1.3 | +1.1 ± 1.2 | 3% |
  | Ops mode | +2.7 ± 0.6 | +2.6 ± 0.9 | 3% |
  | branch choice | +0.2 ± 0.1 | +0.3 ± 0.2 | 0% |
  | realignment target | +0.0 ± 0.1 | +0.9 ± 0.4 | 0% |
  | **all** | **+96.9 ± 12.9** | **+86.6 ± 8.4** | |

* Card choice leaks the most in 100% of resamples (R39: 93%). Within it the US leaks +28.4 and the
  USSR +22.0; by era T4–7 +24.9, T8–10 +15.4, T1–3 +10.1.
* Influence placement is lower than R39's by about 9 points, card choice and play mode higher by
  about 14 and 6. None of the three differences is larger than 1.5 combined standard errors, so
  the kinds are not separated from R39's by this census.

The rank, departure and search-gain tables of the R39 entry were not made for this player.

## How it was measured

The method is the R39 entry's, unchanged: every candidate at every bank position is played out to the
end of the game by the network on both sides, in 512 paired continuations (screen), and the best
move is confirmed on 2,048 fresh pairs where its lead over the player's move is at least 1 point and
1.5 SE. Positions that do not escalate count as zero, so the total is a lower bound. Units are
points of win probability a game, both sides' decisions summed, and each ± is a stratified bootstrap
of 2,000 resamples. See the [R39 entry](E7_searched_leak_R39.md) for why the judge is playouts and
for what the number does not say (the continuation is the network's own greedy play, the candidates
are the player's move, the network's top 6 and Gumbel k=8 @256's, and the sum of per-decision gains
is not the strength a perfect fix would add).

What differs from the R39 run:

* **Positions:** 83,293 from 1,500 games, the player departing from the network at 12.6% of them
  (R39: 84,470, 12.9%). That is 444.2 decisions with two or more legal moves a game (R39: 450.5).
  The bank is 8,000 positions in strata, as before, so the strata and weights are this player's.
* **Where it ran:** locally, on one GPU, not on CI. The annotate stage took 1 h 33 min on six
  processes (`annotate --device cuda`); the screen (38,426 moves) took about 4 h 45 min and the
  confirmation 45 min, on five processes each.
* **A resumed stage:** two of the five screen parts stopped on a CUDA error
  (`unspecified launch failure`) partway, and were finished with `playouts --resume`. A row's pairs
  are seeded from its id alone, so a resumed row is the one an uninterrupted run would have played.

## Reproduction

**Code.** Branch `exp/searched-leak-r48a13`, which is the `exp/searched-leak-r39` branch (the tools
of the R39 entry) with two additions: `annotate --device` (default `cpu`, as before) and
`playouts --resume`.

**Data.** Not published. It is under `data/leak_r48a13/` in the shared data tree: `annotate/`
(six parts), `bank.jsonl.gz`, `screen_input/`, `screen/`, `confirm_input/`, `confirm/` and the
`leak.md` copied here.

The stages and their parameters are the R39 entry's; only the player's checkpoint and the
parallelism changed:

```bash
M="data/checkpoints/E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13_20261014_105711/snapshot_12000034816steps.pt"
P="env PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py"
tools/scripts/check_engine_fresh.sh
# 1. annotate (six parts in parallel)
$P annotate --model "$M" --games 1500 --part $k/6 --player rollout:4:32:4:z2 --device cuda --out annotate/part$k.jsonl.gz
# 2-3. select, screen-select
$P select --annotated annotate/part*.jsonl.gz --size 8000 --control 2000 --seed 0 --out bank.jsonl.gz
$P screen-select --bank bank.jsonl.gz --out screen_input/bank.jsonl.gz
# 4. screen playouts (five parts)
$P playouts --model "$M" --bank screen_input/bank.jsonl.gz --part $k/5 --pairs 512 --seed 0 --resume --out screen/playouts_$k.jsonl.gz
# 5-6. confirm-select, confirmation playouts (five parts)
$P confirm-select --screen-input screen_input/bank.jsonl.gz --playouts screen/playouts_*.jsonl.gz --z 1.5 --floor 0.01 --out confirm_input/bank.jsonl.gz
$P playouts --model "$M" --bank confirm_input/bank.jsonl.gz --part $k/5 --pairs 2048 --seed 1 --resume --out confirm/playouts_$k.jsonl.gz
# 7. leak: 444.2 = 83,293 annotated positions x 8 / 1,500 games
$P leak --bank bank.jsonl.gz --playouts-input confirm_input/bank.jsonl.gz --playouts confirm/playouts_*.jsonl.gz \
  --screen screen/playouts_*.jsonl.gz --large 0 --decisions-per-game 444.2 --out leak.md
```

`leak` reads `rules/cards.json` relative to the working directory, so it runs from the repository
root. The full tables, per side, era and decision kind, are in
[`E7_searched_leak_A13_R48/leak.md`](E7_searched_leak_A13_R48/leak.md); there `raw` means the
player's move.
