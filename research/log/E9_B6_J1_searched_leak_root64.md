# The B6 soup's and J1's searched leak under the 64-world raced root (2026-11-28): level, and card choice is what is left

**Question.** The B6 soup, `E8.1-A8-R132-S45@11160M+(S45,46,47)@11520..11600M` (sha256
`dfc8e050bc65…`), is the E9 leaderboard's anchor and the first pure self-play network with a census.
Where does it still lose under the search it plays with, the raced 64-world keyed rollout root? The
earlier censuses ([J1 and R48](E8_J1_census_and_leak.md)) ran under the 32-world root on E8, so J1,
`E8-A8-R102-S44@13520..13600M` (sha256 `243ec4b375c5…`), was measured again here under the same
root and engine as the soup.

**Answer.**

* **Under the 64-world raced root the B6 soup loses at least +43.7 ± 5.1 points of win probability a
  game, and J1 +42.0 ± 5.8.** The difference is +1.7 ± 7.8, so the census does not separate the two
  networks. Both are lower bounds.
* **No decision kind separates them either.** The largest difference is play mode, +2.1 ± 2.6.
* **Action-round card choice is now the largest leak in both, about +20 a game, and nothing has moved
  it.** It is +20.8 ± 3.3 on the soup and +20.1 ± 3.1 on J1 here, against +20.9 ± 3.1 for J1 under 32
  worlds and +24.9 ± 3.5 for R48. It is the largest kind in 98% of the soup's bootstrap resamples and
  93% of J1's.
* **J1's fall from the 32-world census, −16.7 ± 9.5, is in influence placement.** Placement went from
  +23.1 ± 6.2 to +11.9 ± 4.5 (−11.2 ± 7.7) and card choice did not move (−0.8 ± 4.4). The root, the
  race and the engine all changed between the two censuses, so the fall is not the worlds' alone. It
  is the size 64 worlds bought on E7 with nothing else changed: −12.2 ± 12.0 on the R34 soup
  ([32](per_checkpoint/E7-A8-R1-S44_6400M+_R34_R34-S45_R34-S46_7120..7200M_rollout_k_4_worlds_32_horizon_4_rule_z2_stop_earlier_leaf_own.md),
  [64](per_checkpoint/E7-A8-R1-S44_6400M+_R34_R34-S45_R34-S46_7120..7200M_rollout_k_4_worlds_64_horizon_4_rule_z2_stop_earlier_leaf_own.md))
  and −27.1 ± 16.6 on A13+R48
  ([32](per_checkpoint/E7-A8-R1-S44_6400M+R34_7200M+R48_9170M+A13_11920..12000M_rollout_k_4_worlds_32_horizon_4_rule_z2_stop_earlier_leaf_own.md),
  [64](per_checkpoint/E7-A8-R1-S44_6400M+R34_7200M+R48_9170M+A13_11920..12000M_rollout_k_4_worlds_64_horizon_4_rule_z2_stop_earlier_leaf_own.md)).

The generated tables are [`leak_B6.md`](E9_B6_J1_searched_leak_root64/leak_B6.md) and
[`leak_J1.md`](E9_B6_J1_searched_leak_root64/leak_J1.md). The kind table and the costliest positions
are also in the searched-leak reports of
[the soup](per_checkpoint/E8.1-A8-R132-S45_11160M+_S45_46_47_11520..11600M_rollout_k_4_worlds_64_horizon_4_rule_z2_stop_earlier_leaf_own_chance_keyed_aligned_true_box4_true_race_16.md)
and
[J1](per_checkpoint/E8-A8-R102-S44_13520..13600M_rollout_k_4_worlds_64_horizon_4_rule_z2_stop_earlier_leaf_own_chance_keyed_aligned_true_box4_true_race_16.md),
which the two networks' behaviour reports now link.

## The census

The player is `rollout:4:64:4:z2:earlier:own:keyed:aligned:box4:race16` for both networks. Each
census uses 1,500 self-play games of the player on the E9 engine and an 8,000-position bank.
Candidates (the player's move, the network's most probable moves and Gumbel k=8 @256's) are played
out at 512 pairs, and a lead of at least 1 point and 1.5 SE is confirmed on 2,048 fresh pairs. Rows
that do not escalate count as zero. Each network's playouts are continued by that network's own
greedy play, so the two columns have different judges.

| decision kind | B6 soup | J1 | difference | J1, 32 worlds (E8) |
|:---|---:|---:|---:|---:|
| action-round card choice | +20.8 ± 3.3 | +20.1 ± 3.1 | +0.7 ± 4.5 | +20.9 ± 3.1 |
| influence placement | +11.0 ± 3.2 | +11.9 ± 4.5 | −0.9 ± 5.6 | +23.1 ± 6.2 |
| play mode | +6.2 ± 2.1 | +4.2 ± 1.6 | +2.1 ± 2.6 | +7.6 ± 2.0 |
| headline | +3.7 ± 0.5 | +4.4 ± 0.7 | −0.7 ± 0.9 | +5.3 ± 0.6 |
| Ops mode | +1.2 ± 0.3 | +0.9 ± 0.2 | +0.3 ± 0.4 | +0.7 ± 0.3 |
| coup target | +0.6 ± 0.4 | +0.6 ± 0.6 | +0.0 ± 0.8 | +1.7 ± 1.4 |
| branch choice | +0.1 ± 0.0 | +0.1 ± 0.0 | +0.0 ± 0.0 | +0.1 ± 0.1 |
| realignment target | +0.1 ± 0.4 | +0.0 ± 0.3 | +0.1 ± 0.5 | −0.8 ± 0.7 |
| **all** | **+43.7 ± 5.1** | **+42.0 ± 5.8** | **+1.7 ± 7.8** | **+58.7 ± 7.5** |

| | B6 soup | J1 | J1, 32 worlds (E8) |
|:---|---:|---:|---:|
| positions escalated | 783 of 8,000 | 902 of 8,000 | 932 of 8,000 |
| share of decisions they stand for | 9.12% | 10.94% | 11.88% |
| decisions with a choice, a game | 441.4 | 433.7 | 435.3 |
| leaks the most, of 2,000 resamples | card 98%, influence 2% | card 93%, influence 7% | influence 60%, card 40% |
| Gumbel k=8 @256 over the player, all rows | −115.6 ± 8.8 | −98.3 ± 9.8 | −94.1 ± 9.2 |

Three things in the sub-tables. Their ± is playout noise alone, smaller than the bootstrap ± above
because it does not resample positions, so none of these is a finding on its own:

* **By side.** The card leak sits on the USSR for the soup (+11.8 against the US's +9.1) and on the US
  for J1 (+11.1 against +9.0). J1's placement leak is mostly the US's (+7.9 against +4.0); the soup's
  is even (+5.1 and +5.9).
* **By era.** The soup's largest cells are T4–7 card choice +8.8 and T8–10 influence +8.0. J1's is
  T8–10 card choice +10.4, and its T4–7 card choice is +4.3.
* **The better card.** The China Card is the better card in 58 of the soup's positions (+3.6 a
  game, its largest) and in 49 of J1's (+0.9); J1's largest is Tear Down this Wall (13 positions,
  +1.6).

**The costliest single rows.** The soup's is the USSR playing Nuclear Test Ban instead of Arms Race at
T4 AR1 (+12.6 ± 0.8 a decision, +1.91 a game once weighted). J1's second row is the USSR playing
Africa Scoring before Tear Down this Wall at T10 AR2, which costs +71.1 ± 1.0 in that one position.
It has not been looked at.

## What the comparison rests on

* **The 32-world column is not a control for the worlds.** It ran on E8 on 2026-11-09 without the
  race; this census ran on E9. The engine's fixes between the two were not measured separately.
* **A level leak is not level strength.** The census counts what a full-playout judge finds against
  each player in its own games. It does not say which player wins a match between them.
* **One part of J1's screen was played twice.** Part 85 of 100 was cancelled at the job's 60-minute
  limit and redone alone (26 minutes); the rows are the redone run's.

## Runs

Fork `JamesYouL2/DeepStruggle`, branch `exp/b6-census` (upstream `f393a39b` plus the CI batching
commit `d7e7e993`; the engine sources are upstream's).

| stage | B6 soup | J1 |
|:---|:---|:---|
| annotate, 60 parts | 45905263225 … 45916441733 (parts 10 and 33 redone: 45908319210, 45920167704) | 46102660868 … 46110345308 |
| screen, 512 pairs, seed 0, 100 parts | 46044009791 … 46076974917 | 46112718204 … 46135107989 (part 85 redone: 46138238751) |
| confirm, 2,048 pairs, seed 1, 30 parts | 46087859534 … 46091558614 | 46140415433 … 46143867641 |
