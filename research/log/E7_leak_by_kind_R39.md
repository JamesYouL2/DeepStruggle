# Where R39 loses, by kind of decision (2026-10-13): the leak halves, and influence placement is now 61% of it

**Question (owner).** R39 is on Hugging Face: remeasure the leak by decision kind on it before
anything else ([the R34 soup's entry](E7_leak_by_kind_R34.md) is the comparison). If placement still
leads, find a searcher that recovers it, and check how much of the rollout root's gain R39 has
absorbed.

The network is `E7-A8-R1-S44@6400M+R39@7120..7200M`, R39's last SWA (leaderboard 1676 raw, 1779
under the rollout root; HF `E7-A8-R1-S44@6400M+R39_20261011_145632/swa_7120-7200M.pt`, sha256
`19d3008d…`).

**Answer.**

* **The measured leak halves: +18.5 ± 8.9 points a game, against the R34 soup's +39.0 ± 8.9.**
* **Influence placement still leaks the most: +11.2 ± 3.3, 61% of the total.** It leads in 88% of
  bootstrap resamples (R34: 85%). Experiment 1 (which searcher recovers placement) goes ahead.
* **Card choice and play mode have fallen to noise:**
  * card choice +1.6 ± 7.2, against +8.9 ± 4.6;
  * play mode −0.4 ± 3.4, against +6.7 ± 3.9.
* **The headline is now second: +4.1 ± 1.1, against +1.3 ± 0.8**, and almost all of it is the US
  (+3.6). The US chooses its headline first, so this is not the hidden-headline artefact the blunder
  census found at USSR headlines.
* **Gumbel@256 now recovers more of the placement leak than before:** +7.6 of the +11.2, against
  +7.9 of +17.8 on R34.
* **Searched R39 is level with the searched soup or slightly ahead, and the rollout root still adds
  +131 Elo on top of R39** ([below](#r39-against-the-rollout-root)).

| decision kind | R39 | share | Gumbel k=8 @256 recovers | R34 soup | older network (`E7line_swa_4720-4800M`) |
|:---|---:|---:|---:|---:|---:|
| influence placement | **+11.2 ± 3.3** | 61% | +7.6 ± 2.1 | +17.8 ± 6.0 | +13.9 ± 5.3 |
| headline | +4.1 ± 1.1 | 22% | +1.6 ± 0.4 | +1.3 ± 0.8 | +4.7 ± 1.0 |
| action-round card choice | +1.6 ± 7.2 | 9% | +3.2 ± 2.0 | +8.9 ± 4.6 | +30.0 ± 4.9 |
| coup target | +1.3 ± 0.6 | 7% | +0.5 ± 0.6 | +0.3 ± 0.3 | |
| Ops mode (influence / coup / realign) | +0.5 ± 0.9 | 3% | +1.0 ± 0.6 | +0.9 ± 0.9 | +2.3 ± 1.3 |
| play mode (event / Ops / space) | −0.4 ± 3.4 | | +4.2 ± 1.2 | +6.7 ± 3.9 | +13.5 ± 5.2 |
| **all** | **+18.5 ± 8.9** | | +18.1 ± 3.2 | +39.0 ± 8.9 | +65.1 ± 8.8 |

**Units, as in the R34 entry.** Points of win probability a game, both sides' decisions summed: what
the network would gain by playing the reference search's best move at the decisions where search
finds a large error (2+ points). Those are 6.7% of R39's decisions (R34: 7.3%). The other 89% show no
leak a playout can detect (−21.8 ± 13.8).

**Errors.** Each ± is a stratified bootstrap: positions resampled within their (bank stratum, playout
class) cell, 2,000 times. It counts the sampling of positions as well as the playout noise. This
table is now part of the `leak` stage (`--resamples`, commit `d6ecde6`). Rerun on the R34 data, it
reproduces that entry's hand-made errors: placement ± 6.0, card choice ± 4.6, the total ± 8.9
against ± 9.3.

**R39 against R34, kind by kind** (independent banks, so the errors add in quadrature):

| | R34 − R39 | |
|:---|---:|:---|
| all | +20.5 ± 12.6 | 1.6 SE |
| placement | +6.6 ± 6.8 | 1 SE |
| card choice | +7.3 ± 8.5 | |
| play mode | +7.1 ± 5.2 | 1.4 SE |
| headline | −2.8 ± 1.4 | 2 SE the other way |

The direction is clear, but no single kind's change is decisive. The leak also measures each
network against the reference search run with that same network, so it is not an absolute scale.
R39's own search finds less to correct.

## Inside the placement leak

* **By era:** turns 1–3 +2.7, turns 4–7 +4.7, turns 8–10 +3.8.
  * Turn 10 alone is +3.2: 30 rows of final-scoring arithmetic. Gumbel@256 already recovers +2.5
    of it. Brazil → Pakistan at T10 AR5 is +100 points in every pair.
* **By side:** US +6.8, USSR +4.4. R34 was the reverse (USSR +10.8, US +7.0).
* **Still concentrated.** The two heaviest rows make 31% of it (R34: 38%):

| row id | turn/AR | side | Ops of | network | best (k=16 @1024) | Gumbel@256 | gain a decision | stands for | a game |
|:---|:---|:---|:---|:---|:---|:---|---:|---:|---:|
| `56cdc794fc713b08` | T5 AR4 | USSR | One Small Step | Nigeria | Thailand | Nigeria | +6.0 ± 1.7 | 68 | +2.15 |
| `28b22f4a2a507c14` | T10 AR5 | US | Cuban Missile Crisis | Brazil | Pakistan | Colombia | +100.0 ± 0.0 | 2 | +1.28 |
| `e0a4185ab4c4cc59` | T2 AR4 | US | Defectors | Pakistan | South Korea | Indonesia | +14.2 ± 2.0 | 13 | +0.98 |
| `1af1889310372c10` | T3 AR3 | USSR | East European Unrest | Lebanon | Angola | Jordan | +3.4 ± 1.0 | 40 | +0.72 |
| `f937f4354a30e6d3` | T5 AR2 | US | OPEC | Algeria | Vietnam | Zaire | +3.5 ± 2.0 | 35 | +0.65 |

* **The heaviest row is again a mid-game USSR placement that the deployed search repeats.** On R34
  it was Arms Race's Ops into Algeria; here One Small Step's into Nigeria, where Thailand was
  better. Read it before believing it: the owner's earlier verdict is that board allocation is the
  network's strength.
* **Africa → Asia is the largest region transition** (+3.2, mostly that row).

## The other kinds

* **Headline +4.1, US +3.6.** The largest rows: Red Scare/Purge where OAS Founded was better (T7,
  +14.7 a decision). Turns 4–7 carry +2.4 of it.
* **Card choice +1.6 ± 7.2.**
  * The best move most often plays The China Card instead (+2.6 over 32 rows), then East European
    Unrest.
  * Turns 8–10 are +3.9 and turns 4–7 +1.0, and the DEFCON 2 rows net out to −0.4. The total sits
    inside its noise.
* **Play mode −0.4.** On these positions the reference search's mode is no better than R39's, while
  Gumbel@256's own move gains +4.2. Its largest raw → best rows: an own card for influence where a
  coup was better (+1.8), De Gaulle and Special Relationship.

All the tables are in [leak.md](E7_leak_by_kind_R39/leak.md).

## R39 against the rollout root

The owner's second question: how much of the rollout root's gain has R39, which trained against
it, absorbed? Round robin, 1,000 games a pairing (500 a side), temperature 0, CI `38368176303`.
The rollout root is k = 4, 16 worlds, horizon 4, rule z2 on both networks.

| row against column | R39 + rollout | soup + rollout (R39's teacher) | R39 raw |
|:---|---:|---:|---:|
| **R39 + rollout** | — | 52.3% (US 58.6, USSR 46.0) | 67.9% (US 74.6, USSR 61.2) |
| **soup + rollout** | 47.7% | — | 64.3% (US 67.6, USSR 61.0) |
| **R39 raw** | 32.1% | 34.9% (US 38.0, USSR 31.8) | — |

* **R39 still trails its teacher by ~110 Elo.** It scores 34.9% against the R34 soup with the
  rollout root (Mihail's 1,000-a-side measurement: 35.1%). The raw soup scores 29.4% against the same
  teacher (`research/runs.md`, R39), so R39 has closed ~35 of the teacher's ~150 Elo.
* **The rollout root on top of R39 is +131 Elo** (67.9%). The search is not exhausted on the new
  network.
* **Searched R39 against the searched soup: 52.3%, +16 Elo, 1.5 SE.** Mihail measured the 6,800M
  SWA at 47.8%; the 7,200M SWA does better. By seat it is 58.6% as US and 46.0% as USSR, the same
  lean as R39 raw. R39 is therefore the base for tuning the rollout root.

## Method

The same pipeline as the R34 entry, unchanged except the model (`tools/search_bank.py`, branch
`exp/leak-r39` = `exp/leak-r34` + main `4383452` + `pr/r32-search`):

1. `annotate`, CI `38367342673`: 1,500 R39 self-play games at temperature 0.1, one decision in 8 with
   two or more legal moves. 85,333 positions, 455.1 such decisions a game.
2. `select`, local: 8,000 positions in 71 strata (6,000 disagree / 2,000 agree). Release
   `search-bank-r39-20261013`, sha256 `c0100330…`.
3. `reference`, CI `38371752726`: Gumbel k=16 @1024 over three seeds, each relevant move valued by
   256-simulation search in 64 shared worlds. The best is chosen on half the worlds, the regret
   measured on the other half.
4. `playout-select`, local: 2,423 positions (1,352 with large regret, 771 of the rest, 300
   controls). Release `playout-input-r39-20261013`, sha256 `769c8363…`.
5. `playouts`, CI `38384371895`: the network's move, the best and Gumbel@256's, 1,024 pairs each,
   R39 greedy on both sides.
6. `leak`, local, `--decisions-per-game 455.11`.

## What this does not say

The R34 entry's caveats all stand:

* **The best move is the reference search's**, so errors the search shares go uncounted.
* **The continuation is the network's own greedy play**, so a move whose value lies in a follow-up
  the network does not find reads worse than it is.
* **The placement figure leans on a few positions.**
