# P33 diagnostics (2026-10-15): the critic is the channel search keeps, and it is blind where the searched player errs

The no-training diagnostics of [P33](../plans/P33_search_preserved_gains.md). Networks: R39's SWA
(`E7-A8-R1-S44@6400M+R39@7120..7200M`), A13+R48's SWA (`…+R48@9170M+A13@11920..12000M`) and the R34
soup (`E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M`). Engine fingerprint b670b258 (the build of
the `hand-knowledge-tracking` worktree, whose `engine/` and `bindings/` sources are identical to this
branch's); the games in play from 4687c0eb (parallel rollout stepping, bit-identical picks).

## P33a — the leaf-critic swap (`rollout:<A>:4:16:4:z2:<B>`)

A two-network root: candidates and greedy playouts from network A, the leaf critic from network B.
Each player 1,000 deals a side against the searched soup (k = 4, 16 worlds, 4 boundaries, z2); the seat
bar is the searched soup's self-play, US 54.1% / USSR 45.9% (both orientations averaged); the two
swapped players also head to head. `data/reports/p33a_*.json`.

| player (playouts / leaf critic) | score | as US | as USSR | US − bar | USSR − bar |
|:---|---:|---:|---:|---:|---:|
| A13+R48 / A13+R48 | 49.8% | 55.3% | 44.2% | +1.2 | −1.7 |
| A13+R48 / soup | 50.4% | 52.0% | 48.9% | −2.1 | +3.0 |
| soup / **A13+R48** | **46.2%** | 48.8% | 43.7% | **−5.3** | **−2.2** |
| R39 / R39 | 51.3% | 57.2% | 45.4% | +3.1 | −0.5 |
| R39 / soup | 51.1% | 55.2% | 47.1% | +1.1 | +1.2 |
| soup / R39 | 47.9% | 52.7% | 43.0% | −1.4 | −2.9 |

| head to head | score | as US | as USSR |
|:---|---:|---:|---:|
| A13+R48 playouts + soup critic vs soup playouts + A13+R48 critic | **58.2%** | 57.2% | 59.2% |
| R39 playouts + soup critic vs soup playouts + R39 critic | 50.3% | 54.0% | 46.7% |

* **A13+R48's critic is its searched weakness.** Its candidates and playouts with the soup's critic are
  level with the searched soup; the soup's with A13+R48's critic lose in both seats, and the two swaps
  head to head are 58.2%, above 50% in both seats. By the plan's rule the leaf moves the rating more
  than the playouts: the critic is the search-preserved channel, and A13+R48's raw gain (+29 over the
  soup) was bought with a worse critic at the leaves.
* **On the 256-wide pair the effect is small:** R39's playouts with the soup's critic are +1 in both
  seats, the soup's with R39's critic −1.4 / −2.9, and the two swaps are level head to head (50.3%).
  The soup's critic is a little better than R39's and clearly better than A13+R48's; the soup's
  searched lead over A13+R48 is mostly A13+R48's critic.
* This sits beside P33c without contradiction: every critic is at chance on the decisive leak pairs,
  but A13+R48's ranks the ordinary leaf differences the root mostly decides on slightly worse.

## P33c — the critic on labelled move pairs (`ai/eval/leak_critic.py`)

The searched-leak bank's data ([`E7_searched_leak_R39.md`](E7_searched_leak_R39.md); release
`searched-leak-r39-data-20261014`, checksums verified): **confirmed** pairs are the 1,122 positions
where a 2,048-pair playout confirmed a move better than the searched player's (R39 + rollout root);
the **baseline** is the screening stage -- every candidate of all 8,000 bank positions against the
player's move at 512 pairs -- kept where the label is signed at |z| ≥ 2 (8,329 pairs). For each pair,
both moves are applied (16 shared dice draws; 8 for the settled read), the mover's action round is
played out greedily by the same network, and the critic is read from the mover's side; the share
of pairs it ranks against the label is reported.

| % of pairs ranked wrong (settled) | R39 | A13+R48 | soup |
|:---|---:|---:|---:|
| baseline, all | 19.1 ± 0.4 | 19.9 ± 0.4 | 19.4 ± 0.4 |
| baseline, card choice | 25.2 ± 0.7 | 25.3 ± 0.7 | 24.8 ± 0.7 |
| baseline, card choice at DEFCON 2 | 27.6 ± 1.4 | 29.4 ± 1.5 | 28.2 ± 1.4 |
| baseline, the alternative better by ≥ 5 points | **55.2 ± 2.0** | **52.2 ± 2.0** | **56.0 ± 2.0** |
| confirmed leaks, all | 48.6 ± 1.5 | 47.4 ± 1.5 | 48.0 ± 1.5 |
| confirmed leaks, card choice at DEFCON 2 | 52.7 ± 3.4 | 51.4 ± 3.4 | 48.6 ± 3.4 |

* **On ordinary pairs the critic is good** (correlation with the playout label +0.79 to +0.83), and on
  play-mode choices very good (10–11% wrong).
* **Where the searched player is wrong, every critic is at chance** (correlation 0.00–0.12), the
  soup's no better than the others. By the plan's rule the leak is the critic's; the soup's searched
  lead does not come from ranking these pairs better.
* **Read right after the move, card choice is mis-measured.** The critic at the decision after a card
  is chosen has not seen the card played and values the hand left behind: 55% wrong on the
  baseline, 63–74% at DEFCON 3. Every number above is the settled read; the raw read is in the
  output files.
* The confirmed set is selected by the search's own failure, so chance there is partly selection;
  the baseline's large-label rows (unconfirmed, same selection only through the bank's strata) say
  the same.

### Does R49's target see what the critic misses? (`leak_critic --horizon 4`)

The same pairs, each move read the way R49's value target reads a position: the rollout root's own
continuation -- worlds sampled from the mover's view, dice shared by the two moves, both sides greedy
through 4 action-round boundaries, the critic at the stop.

| % ranked wrong | A13+R48, 16 worlds | A13+R48, 64 worlds | soup, 16 | R39, 16 |
|:---|---:|---:|---:|---:|
| baseline, all | 16.2 ± 0.4 | 14.6 ± 0.4 | 15.1 ± 0.4 | 12.0 ± 0.4 |
| baseline, the alternative better by ≥ 5 points | 53.9 ± 2.0 | **53.0 ± 2.0** | 58.1 ± 2.0 | 64.4 ± 2.0 |
| confirmed leaks, all | 49.5 ± 1.5 | **47.0 ± 1.5** | 51.0 ± 1.5 | 55.0 ± 1.5 |
| confirmed leaks, card choice at DEFCON 2 | 49.1 ± 3.4 | 49.5 ± 3.4 | 46.4 ± 3.4 | 50.0 ± 3.4 |

* The 4-round continuation ranks ordinary pairs better than the bare critic (12–16% wrong against
  19–20%), but on the decisive pairs it is at chance, and **four times the worlds barely moves it**
  (49.5 → 47.0%): blindness, not noise. It leans slightly the wrong way on average.
* **R49 is skipped:** its target cannot teach what the critic is missing, because the 4-round
  lookahead does not see it either. The difference between the moves shows only beyond that
  horizon -- where the leak bank's labels (512–2,048 paired playouts to the game's end) see it.

### Does a turn-end target see it? (`leak_critic --turn-end`, R52's premise)

Each move played out greedily by both sides until the turn changes, in 64 worlds dealt from the
mover's view (`ts.determinize`), the two moves sharing each world's dice, the critic read at the turn's
end. A13+R48's SWA.

| % ranked wrong | after the action round | 4-boundary continuation | **to the turn's end** |
|:---|---:|---:|---:|
| confirmed leaks, all | 47.4 | 47.0 | **37.8 ± 1.4** (weighted 42.8) |
| confirmed, card choice | 53.1 | — | **35.7 ± 2.1** |
| confirmed, play mode | 39.2 | — | **35.9 ± 3.1** |
| confirmed, card choice at DEFCON 2 | 51.4 | 49.5 | **42.7 ± 3.3** |
| confirmed, influence placement | 44.4 | — | 44.4 ± 3.1 |
| confirmed, label ≥ +5 points | 46.4 | 43.1 | **29.9 ± 3.2** |
| baseline, the alternative better by ≥ 5 points | 52.2 | 53.0 | **37.2 ± 2.0** (weighted 48.7) |

* **R52's premise holds for card choice and play mode:** the correlation with the playout label rises
  from ~0.08 to +0.43 on the confirmed leaks (+0.54 on card choice), and the largest leaks are ranked
  right 70% of the time. What the critic misses is visible by the turn's end, not within 4 boundaries.
* **Influence placement stays at chance** -- the leak's "wrong country in the right region, late"
  resolves at a later scoring or the final count; R53's exact endgame labels are its candidate.
* **Weighted by the population the gain is smaller** (42.8% and 48.7% wrong): it sits in the strata the
  bank over-samples, so the effect on typical play will be smaller than the unweighted figures.

### And in play? The rollout root stopping at the turn's end

The R34 soup under the rollout root with a different stop against the soup under today's root
(`stop=boundaries`, 4 boundaries), both k = 4, 16 worlds, z2; 1,000 deals a side, 8 parts of one
125-pairing batch each, pooled (`data/reports/{turn,later}_soup_*_vs_soup.json`). `turn` stops at the
first position of the next turn (shorter than 4 boundaries late in a turn, longer early); `later` at the
later of the two.

| stop vs today's root | score | as US | as USSR | US − bar | USSR − bar |
|:---|---:|---:|---:|---:|---:|
| `turn` (CUDA graphs on both sides) | **51.3%** | 55.3% | 47.3% | +1.2 | +1.4 |
| `turn`, replicate on other deals (`--base-seed 20000`) | 51.1% | 53.9% | 48.4% | −0.2 | +2.5 |
| `turn`, both (4,000 games) | **51.2%** | 54.6% | 47.9% | +0.5 | +2.0 |
| `earlier` (the turn's end if within 4 boundaries) | **51.9%** | 56.6% | 47.2% | **+2.5** | **+1.3** |
| `earlier`, replicate on other deals (`--base-seed 20000`) | 52.0% | 54.9% | 49.2% | +0.8 | +3.3 |
| `earlier`, both (4,000 games) | **51.95%** | 55.8% | 48.2% | **+1.7** | **+2.3** |
| `later` | 47.8% | 52.3% | 43.2% | −1.8 | −2.7 |
| the searched soup's own self-play bar | — | 54.1% | 45.9% | | |

* **Stopping at the turn's end is level to slightly better**: +1.2 / +1.4 on the first deals, −0.2 /
  +2.5 on the replicate's; pooled over 4,000 games +0.5 / +2.0, about 1.6 SE above level. Not
  significant, never below the bar. The play-time root does not lose by reading the
  critic at the turn boundary instead of 4 boundaries ahead.
* **Stopping at the later of the two is worse** (−1.8 / −2.7, ~2 SE).
* **The two results split by position in the turn.** Where the turn's end lies beyond 4 boundaries
  (early in a turn) `later` and `turn` are the same rule -- both play on to the turn's end. Where it
  comes sooner (late in a turn) `later` is today's root and `turn` stops early, at the boundary. So
  `later` − today = E, the effect of playing on to the turn's end early in a turn: **≈ −2.2 points**;
  and `turn` − today = E + L, with L the effect of stopping at the turn's end late in a turn:
  **L ≈ +3.5 points** (each ± ~1.6, from two 2,000-game matches). Read together: a leaf at the turn
  boundary is worth having when it is nearer than 4 boundaries, not when reaching it costs extra
  plies. The rule that takes only the good half -- the *earlier* of 4 boundaries and the turn's end --
  is predicted at about +L over today's root.
* **The earlier rule, measured** (`stop=earlier`): **51.9%, +2.5 / +1.3** on the first deals, 52.0%
  (+0.8 / +3.3) on the replicate's; pooled over 4,000 games **51.95%, +1.7 / +2.3, ~2.5 SE above level
  (~+14 Elo)**, above the bar in both seats in both runs -- the best of the four stops, against a
  prediction of ~+3.5 points (within its error). Consistent with the split: the leaf at the turn boundary helps when it is
  nearer than 4 boundaries.
* 4 boundaries beat 2 by +35 Elo on the soup ([`E7_search_depth_and_value.md`](E7_search_depth_and_value.md)),
  so a shorter stop alone is not what helps: the leaf's position in the turn is.
* For R52: a turn-end leaf is at least as good as today's in play, at 16 worlds, without training.
  A critic trained toward turn-end values has the noise averaged out, which a single search
  decision cannot do.

## Is the critic self-consistent? (`ai/eval/critic_consistency.py`)

The owner's question after the stop-rule games: playing deeper should only add information, so why does
the root lose when its playouts run on? 1,024 greedy self-play games per network (the R34 soup, A13+R48's
SWA), every decision read from both sides; the deal test on 1,500 turn ends × 16 reseeded deals; the
leaf probe on 512 bank positions. `data/logs/temp/consistency.out`.

**1. The critic disagrees with itself across perspectives.** The trainer reads a decision only from its
decider's side; the rollout root reads every leaf from the root mover's. On the same state:

| | soup | A13+R48 |
|:---|---:|---:|
| mean v_decider + v_other (0 if consistent) | **+0.085** ± 0.001 | **+0.068** ± 0.001 |
| mean \|v_decider + v_other\| | **0.132** | **0.131** |
| log loss of the decider's reading / the other side's | 0.497 / 0.516 | 0.491 / 0.511 |

The other side's reading is optimistic for itself (+0.07–0.09, largest at AR1: +0.11–0.12) and worse
calibrated in every slice; averaging the two is no better than the decider's alone. The E3 measurement
(227 positions, 0.144, [`../archive/E3_ladder/findings/value_bootstrap_perspective.md`](../archive/E3_ladder/findings/value_bootstrap_perspective.md))
was never fixed.

**1b. Which reading carries the offset, and does a constant remove it?** (the owner: averaged over the
hidden cards, the two sides should agree; if they do not, a constant should improve calibration). Each
reading's signed error against the result in the decider's frame, per decider side; then the other
side's reading shifted by the mean sum, fit per bucket and side on even games and scored on odd ones.
`data/logs/temp/offset.out`.

| | soup US | soup USSR | A13+R48 US | A13+R48 USSR |
|:---|---:|---:|---:|---:|
| signed error, decider's reading | −0.022 | +0.024 | −0.057 | +0.048 |
| signed error, other side's reading (negated) | −0.110 | −0.058 | −0.130 | −0.015 |
| log loss, decider's reading | **0.483** | **0.487** | **0.481** | **0.486** |
| log loss, other side's reading + fitted shift | 0.501 | 0.495 | 0.505 | 0.502 |
| log loss, average of both, shift removed | 0.487 | 0.486 | 0.485 | 0.486 |

* **Two offsets, not one.** The other side's reading is optimistic for itself by ~0.08 (0.11–0.13 at AR1).
  And the decider's own reading has a **side offset**: in greedy self-play it underrates the US by 0.02
  (soup) to 0.05 (A13+R48) from both seats -- the US decider is pessimistic, the USSR decider
  optimistic. Pooled over sides the two cancel, which is why P33b's mover-frame bias read under 0.6
  points. A plausible cause: the critic is trained on temperature-1 games, whose seat balance differs
  from greedy play's (the searched soup's own US bar is 54%).
* **A constant does not repair the other side's reading:** shifted, it is still 0.008–0.024 worse in log
  loss than the decider's. Its error is position-specific, not an offset.
* **Averaging the two sides gains nothing** (±0.001 for the USSR, worse for the US): the decider's
  reading alone is the best estimate. Decider-side leaves are the fix; no averaged leaf is worth building.
* The side offset is nearly common-mode in the root -- every candidate of a decision has the same mover,
  and in the mover's frame both deciders' readings lean the same way -- so it matters for absolute values
  (value targets, calibration), not for the root's comparisons.

**2. No drift, but jumps at every change of mover.** The US-frame value's mean change from one decision
to the next is within ±0.01 for every kind of step; its RMS is 0.03 between one mover's decisions and
**0.12–0.13** at hand-overs, round boundaries and turn boundaries alike -- the two perspectives
disagreeing, not new information.

**3. No sharper reading at turn starts.** Log loss falls smoothly through each turn (e.g. turns 4-7:
headline 0.486 → AR6 0.465 on the soup); the headline is not better calibrated than mid-turn.

**4. The deal is nearly a martingale:** the mean value after 16 reseeded deals minus the value before is
+0.004 ± 0.002 (soup) and +0.010 ± 0.002 (A13+R48) -- a small optimism after the deal, under a point of
win probability. The critic's spread across deals of one position is 0.10 SD.

**5. The root's leaves.** The share of leaves whose decider is the root mover, and the mover's reading
against the decider's sign-corrected one where it is not:

| stop | mover decides (soup / A13+R48) | mover − decider reading, mean \|.\| | play vs today's root |
|:---|---:|---:|---:|
| `earlier` | **85.1% / 85.3%** | 0.102 / 0.095 | **51.9%** |
| `boundaries` (today) | 80.6% / 80.7% | 0.101 / 0.094 | 50% |
| `turn` | 49.7% / 49.7% | 0.104 / 0.099 | 51.2% |
| `later` | **45.0% / 45.0%** | 0.103 / 0.098 | **47.8%** |

A `turn` leaf is a headline decision, whose decider is the root mover for one side only. The stop rules'
order in play follows the on-distribution share except `turn` against `boundaries`; at an off-distribution
leaf the reading is off by ~0.10, against the 0.04–0.08 leads the root acts on.

* **Reading:** the critic's main fault as a leaf evaluator is perspective inconsistency, ~0.13 on every
  position, and the rollout root reads 15–55% of its leaves from the side the critic was not trained on.
  Two fixes follow: read each leaf from its decider's side (play time, a bindings option), and train the
  critic on both perspectives (a recipe).

### Reading each leaf from its decider's side (`leaf=decider`)

The fix for probe 5: every leaf read from its decider's side, negated into the mover's frame where that is
the opponent -- always on the critic's training distribution. Against today's root on the R34 soup, the
first deal set, 2,000 games each (`data/reports/dleaf_*_vs_soup.json`):

| stop | leaves read from the mover's side | from the decider's side |
|:---|---:|---:|
| 4 boundaries | 50% (today's root) | 50.5% (+0.7 / +0.4) |
| `earlier` | **51.95%** (4,000 games) | 50.3% (−0.9 / +1.6) |
| `turn` | 51.2% (4,000 games) | **48.4%** (−0.8 / −2.3) |

* **On-distribution is not enough: the decider's reading loses what the mover knows.** Where the opponent
  decides, its side's observation hides the mover's own hand -- the one thing the mover knows for
  certain -- and shows the opponent's sampled one. The loss is largest at turn-end leaves (a headline
  the US chooses first, so half the leaves; the mover's newly dealt hand is thrown away), as the owner
  predicted, and it erases `earlier`'s gain.
* So the off-side reading's bias (~0.08) costs less than the discarded hand.

### Playing on to the mover's own next decision (`leaf=own`, the owner's design)

Past its stop, a playout plays on until the root mover decides, and the critic is read there from the
mover's side: on the training distribution *and* with the mover's own hand. Same matches, same deals:

| stop | mover's side (today) | decider's side | **own decision** |
|:---|---:|---:|---:|
| 4 boundaries | 50% | 50.5% | **52.3%** (55.8 / 48.9: +1.7 / +3.0) |
| `earlier` | 51.95% | 50.3% | **53.3%** (59.9 / 46.8: +5.8 / +0.9) |
| `turn` | 51.2% | 48.4% | **51.7%** (56.0 / 47.4: +1.9 / +1.5) |

* **Best under every stop**, and above the bar in both seats in every cell. On today's 4-boundary stop
  alone it is +2.3 points (~+16 Elo); with `earlier`, +3.3 (~+23 Elo), the best searched player measured.
* Both halves matter: the decider's reading (on-distribution, without the mover's hand) and the
  mover's off-side reading (with the hand, off-distribution) are each worse than reading the mover's own
  decision. Replicates of `earlier`+own and 4 boundaries+own on the second deal set are running.

## P33b — the critic on greedy against sampled games

## P33b — the critic on greedy against sampled games (`ai.eval.critic_boundary --temperatures 0 1`)

2,048 self-play games at each temperature per network; the critic's value at every decision against
the game's result, game-clustered SEs.

| network | log loss, greedy | log loss, T=1 | greedy − T=1 |
|:---|---:|---:|---:|
| R39 SWA | 0.4905 | 0.4982 | −0.008 (−0.7 SE) |
| A13+R48 SWA | 0.4781 | 0.5039 | −0.026 (−2.5 SE) |
| R34 soup | 0.4996 | 0.5272 | −0.028 (−2.6 SE) |

* **The critic is more accurate on greedy games, not less**, in every slice -- DEFCON 2 in turns 8–10
  included (−0.026 to −0.048). Signed bias is under 0.6 points of win probability everywhere.
* **The hypothesis fails; by the plan's rule R50 and R55 are dropped, R49 stays.** The critic's fault
  is resolution at the decisive positions (P33c), not a mismatch between the policy it is trained on
  and the one the search plays.

## P33d — candidate recall (`ai/eval/candidate_recall.py`)

The rollout root at k = 8 (16 worlds, 4 boundaries, z2) on 4,000 bank positions (R39's searched
self-play, population-weighted), the rank under the network of the move it picks.

| decision kind | departs | departures to ranks 5–8 |
|:---|---:|---:|
| all | 13.6% | **28.0%** (unweighted 20.9% ± 1.6) |
| influence placement | 13.6% | 36.7% |
| card choice | 20.7% | 25.2% |
| play mode | 6.8% | 2.7% |
| op mode | 3.4% | 0% |

Above the plan's 25% line overall: channel 1 is open for placement and card choice, so k = 6 at play
time and R51's recall objective qualify by the rule. A caution the rule does not price: with more
candidates more departures can be the seed noise of 16 worlds (a winner's curse); a k = 6 against
k = 4 games check settles the play-time half. The R39 tuning on JamesYouL2's branch found k = 6 at 32
worlds level with k = 4 offline.

## P33f — does the trunk carry a belief about the opponent's hand? (`ai/eval/belief_probe.py`)

A13+R48's SWA, 768 self-play games at T = 1, one decision in three; every card in the hidden pool (the
opponent's unseen hand plus the deck, from the decider's side) labelled by whether the opponent holds
it. Each probe is a one-hidden-layer correction to logit(hand size / pool size), early-stopped on
held-out games; read on another held-out set (19,281 positions, 574,593 pool entries, 15.5% held).

| predictor | log loss | vs the prior | AUC |
|:---|---:|---:|---:|
| the prior: hand size / pool size | 0.3914 | — | 0.714 |
| public probe: card block + globals | 0.3780 | +3.4% | 0.745 |
| the whole observation | 0.3779 | +3.4% | 0.745 |
| the frozen trunk | 0.3778 | +3.5% | 0.745 |

* **The trunk holds exactly what the public card information supports, no more** (mean per-card AUC
  0.729 against the public probe's 0.730); the board adds nothing to it. By the plan's rule a belief is
  not already represented beyond public information; what history could add (the opponent's own
  choices, cards held across turns) is not measurable from this observation and is the owner's
  representation decision.
* Without early stopping the raw-input probes overfit below the prior (0.42–0.44): the first two runs'
  numbers are not usable and are not reported.

## Decisions so far

* **The critic is the search-preserved channel** (P33a): A13+R48's searched deficit is its critic. An
  arm is judged by its critic under the root, not by its raw rating.
* Critic track: **R50 and R55 are dropped** (P33b); **R49 is skipped** -- its own target is blind to the
  leak pairs (the horizon read above).
* The leak is the critic's at the decisive positions (P33c), and invisible within 4 action rounds:
  value targets that see further (R52 to the turn's end, R53 exact endgames, game-end playouts) are
  the remaining route. The turn-end read sees the card-choice and play-mode leaks (37.8% wrong against
  ~47%), not placement: **R52 is the next critic-track arm**, R53 the one for placement.
* Channel 1 is open at 28% (P33d): a k = 6 games check and R51 qualify.
* No belief input follows from P33f; the history question goes to the owner.

Outputs: `data/logs/temp/p33c.out` (raw read), `p33c_settled.out`, `p33b.out`, `p33d.out`, `p33f3.out`.
