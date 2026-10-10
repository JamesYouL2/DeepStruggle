# The blunder bank (2026-10-10): 37 positions where the E7 network's own move costs 10+ points of a game, confirmed twice

**Question (owner).** Where is the network clearly making a move that costs a large share of the
game? Keep those positions, so any later network or searcher can be checked against them.

**Answer.** The bank holds 37 positions (`ai/eval/banks/blunders_E7.jsonl`). In each one, a
fresh paired measurement by two judge networks puts the heads soup's greedy move 10+ points of
the mover's score below a fixed alternative, at 3+ standard errors under both. 20 of them cost
20+ points.

* **Are they really mistakes? 20 hold up under every check, 7 probably do, 10 do not.**
  * **6 are proved exactly.** Each is a turn-10 last-round decision, enumerated to the end of the
    game with dice averaged, in nine worlds with the mover's unseen cards redealt. Five hold whatever
    the opponent does, and one against the opponent's best reply.
  * **14 more hold with search on both sides of the continuation** (Gumbel@256, 96 fresh pairs).
  * **7 stay positive at 2+ SE** with search but drop below 3 SE or 10 points.
  * **9 vanish with search** (cost ≈ 0). These were mistakes only given the network's own greedy
    follow-up.
  * **1 is exactly not a mistake.** The US placing in Nigeria rather than India at turn 10 still
    wins in every world with correct play afterward. The playouts lose there because the network,
    and Gumbel@256, then botch the remaining points of the same play. The error at that position is
    real but sits in the follow-up, not in the move recorded.

  Each row's `verdict` in the bank file records which of these it is. See
  [Verification](#verification).
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

## Verification

Three independent checks, all on the 37 rows as built:

* **Exact (`blunder_bank.py prove`).** Where the game ends in the current turn, everything after the
  move is enumerated. The mover takes its best move, each die is averaged over its six faces, and
  the opponent's moves are treated two ways: helping the mover, the best case a blunder is held
  to, and replying against the mover, the worst case the better move is held to. This is done in
  the recorded position and in 8 worlds with the mover's unseen cards redealt
  (`paired_playouts.pair_start`).
  * A row is **proved** if the better move's worst case is never below the blunder's best case and
    is above it in some world.
  * It is **proved against the best reply** if the same holds with the blunder too answered by
    the opponent's best reply.

  Results:
  * 7 rows are solvable, all at turn 10, action round 7: 5 proved, 1 proved against the best
    reply, 1 shown not to be a mistake (Nigeria/India above).
  * 12 rows need a new turn (a new deal), and 18 exceed the 400,000-node budget, mostly turn-10
    rows with the opponent still to move.
  * The redeals matter: in the Poland/Angola row, the blunder wins in 1 world of 9 and the better
    move in all 9.
* **Searched continuations (`confirm --continue-with 256:8`).** The heads soup's Gumbel k=8 @256
  plays both sides of every continuation, 96 pairs, seed 23 (fork CI run `38091099918`, pooled
  locally from its 20 parts). The mean cost of the 37 falls from 27.9 points (greedy soup judge) to
  23.7. 21 hold at 10+ points and z ≥ 3, 7 are positive at z ≥ 2, 9 are not resolved, and none
  reverses.
* **The verdicts combined** (`verdict` in `ai/eval/banks/blunders_E7.jsonl`):

| verdict | rows | what it means |
|:---|---:|:---|
| `proved` | 5 | exact, whatever the opponent does |
| `proved-vs-best-reply` | 1 | exact, against the opponent's best reply |
| `holds-under-search` | 14 | 10+ points at z ≥ 3 under the soup, R32 and the searched continuation |
| `likely` | 7 | positive at z ≥ 2 under the searched continuation |
| `continuation-dependent` | 9 | the cost is gone when the follow-up searches |
| `move-not-a-mistake` | 1 | exact: the move keeps the win; the follow-up loses it |

The searched continuation also scores the Nigeria row at 47 ± 1 points. Paired playouts measure
a move *together with* the continuation policy's play after it, so only the exact check separates
the two. Rows whose cost survives search are robust to the greedy follow-up, not proof that the
move itself is the error.

**What survives.** The 20 rows that hold are of two kinds:
* the last placements and play choices of turn 10, which are arithmetic;
* card choices: game-winning events declined (Wargames at turn 9; Iran-Iraq War at turn 10's last
  round), headlines (Europe Scoring at turn 3, Red Scare/Purge at turn 8, Brush War at turn 10,
  Latin American Debt Crisis at turn 10), a scoring card played in the wrong round (Africa before
  Middle East at turn 6, Kitchen Debates over Africa Scoring at turn 10), and which card to spend
  in a round, opponent cards included.

No mid-game influence placement survives. The one in the bank (Decolonization's point, Haiti
against Cuba, turn 6) is continuation-dependent.

## What this does not say

* **It is not a rate.** The shortlist came from 200 greedy games of one network through a screen
  (forced-loss labels and a Gumbel @32 disagreement) and a 32-pair estimate. Missed blunders
  are not counted, so 37 is a floor on 200 games, not an estimate of a population.
* **The cost is judged by continuations.** A move whose value lies in a follow-up the judges do
  not play is undervalued. Two greedy judges that are near relatives and one searched one reduce
  that but do not remove it. Outside the six exact rows, a blunder here is confirmed against what
  these networks, and their search, would go on to do.
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
