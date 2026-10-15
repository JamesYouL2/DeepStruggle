# P33 diagnostics (2026-10-15): the critic is calibrated but blind where the searched player errs

The no-training diagnostics of [P33](../plans/P33_search_preserved_gains.md). Networks: R39's SWA
(`E7-A8-R1-S44@6400M+R39@7120..7200M`), A13+R48's SWA (`…+R48@9170M+A13@11920..12000M`) and the R34
soup (`E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M`). Engine fingerprint b670b258 (the build of
the `hand-knowledge-tracking` worktree, whose `engine/` and `bindings/` sources are identical to this
branch's). P33a's results are added when its games finish.

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

### And in play? The rollout root stopping at the later of 4 boundaries and the turn's end

The R34 soup under the rollout root with `stop=later` (the later of 4 boundaries and the end of the
decision's turn) against the soup under today's root (`stop=boundaries`), both k = 4, 16 worlds, z2;
1,000 deals a side, 8 parts of one 125-pairing batch each, pooled (`data/reports/later_soup_later_vs_soup.json`).

| | score | as US | as USSR |
|:---|---:|---:|---:|
| soup + later root vs soup + today's root | **47.8%** | 52.3% | 43.2% |
| the searched soup's own self-play bar | — | 54.1% | 45.9% |

* **Playing longer is slightly worse in games:** −1.8 and −2.7 points against the bars, about 2 SE
  below level. The turn-end read ranks the leak pairs better (above), but at 16 worlds the longer
  greedy playouts add more noise per world than signal. A longer stop may need more worlds to pay;
  untested.

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

* Critic track: **R50 and R55 are dropped** (P33b); **R49 is skipped** -- its own target is blind to the
  leak pairs (the horizon read above).
* The leak is the critic's at the decisive positions (P33c), and invisible within 4 action rounds:
  value targets that see further (R52 to the turn's end, R53 exact endgames, game-end playouts) are
  the remaining route. The turn-end read sees the card-choice and play-mode leaks (37.8% wrong against
  ~47%), not placement: **R52 is the next critic-track arm**, R53 the one for placement.
* Channel 1 is open at 28% (P33d): a k = 6 games check and R51 qualify.
* No belief input follows from P33f; the history question goes to the owner.

Outputs: `data/logs/temp/p33c.out` (raw read), `p33c_settled.out`, `p33b.out`, `p33d.out`, `p33f3.out`.
