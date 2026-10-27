# How diverse is the line's play? P35b (a), A13+R48 greedy and searched against the human corpus (2026-10-27)

**Question** ([P35 §7](../plans/P35_capacity_belief_dynamics.md)). Before strategy conditioning
(AlphaStar's z, an observation change and the owner's call) is built, measure how narrow the policy
is: its setups, its first action rounds, the regions it invests in by turn 3, and its coup and space
use, by side, against the searched player's and the human corpus's distributions. Part (b), the
exploit margin of a fresh seat-locked exploiter, is RL training on the GPU and is not in this entry.

**Answer.**

* **The setup is the one collapsed dimension.** The network plays one setup per side: US West
  Germany 4, Italy 3, Iran 1, Panama 1 in 99.8% of games, and USSR Poland 3, Yugoslavia 3 in 100%.
  The humans play 4.2 effective setups as US and 3.1 as USSR, and none of the human setups is the
  network's. Search widens it only a little: 2.1 effective US setups, with one Italy point moved to
  France in 22% of games, and 1.2 for the USSR.
* **Everything else is narrower than the humans' play, but not collapsed.**
  * **AR1:** 70–75% of the humans' effective choices (US 9.2 against 12.5, USSR 3.3 against 5.4).
  * **Coup targets in turns 1–3:** 77–83% of the humans' (US 14.5 against 17.5, USSR 8.6 against
    11.1).
  * **The regions invested in by turn 3:** as broad as the humans' or broader (main region US 2.2
    against 2.1, USSR 2.6 against 2.4).
* **The searched player stays inside the network's distribution.** The rollout root `4:32:4:z2` is
  within a TVD of 0.02–0.07 of greedy play on every dimension except the US setup (0.26). It is
  slightly broader everywhere (AR1 US 10.2, USSR 3.7; USSR coup targets 9.5), but its moves are the
  network's top-k by construction. The decision rule's second clause, "a searched-player distribution
  the policy cannot cover", is not met.
* **Part (a) finds no strength cost of low diversity.** The one collapse, the setup, was priced on
  the R34 soup: its own fixed setups were best or tied against the human ones, and no rule that
  varies the setup with the hand beat them out of fold (below). By P35's decision rule, part (a)
  alone does not send the z build to the owner; part (b) is still open.

## Sources

| source | what | games |
|:---|:---|---:|
| greedy | `E7-A8-R1-S44@6400M+R34@7200M+R48@9170M+A13@11920..12000M`, the leaderboard's SWA (sha256 `70786f2a0c61…`), argmax, whole games of self-play | 16,384 |
| searched | the same network under the rollout root, the leaderboard player `…~rollout(k=4,worlds=32,horizon=4,rule=z2,stop=earlier,leaf=own,chance=keyed,aligned=true,box4=true)`; both seats searched through turn 3, then cut | 2,048 |
| humans | the ts-replayer corpus, trusted turns only (`tools/lib/corpus_driver.feed_corpus_game`); 254 with a setup, 230 with turns 1–3 complete, 118 recorded to the end | 265 |

CI run `40567064117` (`.github/workflows/strategy_diversity.yml`) played both self-play sources.
The human records were made locally, because the corpus does not run on GitHub. Records and reports
are in `data/reports/strategy_diversity/40567064117/`, and the full report, with the top eight
choices of every dimension, is [`E7_strategy_diversity_P35b/diversity.md`](E7_strategy_diversity_P35b/diversity.md).
Tool: `tools/scripts/strategy_diversity.py` (`play`, `humans`, `report`).

**How to read the entropies.** Each source is rarefied to the humans' sample size, as the mean
plug-in entropy of 200 draws without replacement. A sample of 16,384 games would otherwise show rare
choices that 254 human games cannot. *eff.* is 2^H. *Human unplayed* is the share of human games
whose choice the source makes in under 1% of its own games.

## Breadth, effective choices at the humans' sample size

| dimension | side | greedy | searched | humans | greedy TVD to humans | human unplayed by greedy |
|:---|:---|---:|---:|---:|---:|---:|
| setup | US | 1.0 | 2.1 | 4.2 | 1.00 | 100% |
| setup | USSR | 1.0 | 1.2 | 3.1 | 1.00 | 100% |
| AR1 | US | 9.2 | 10.2 | 12.5 | 0.33 | 13.0% |
| AR1 | USSR | 3.3 | 3.7 | 5.4 | 0.38 | 16.9% |
| main region, T1–3 | US | 2.2 | 2.3 | 2.1 | 0.18 | 1.3% |
| main region, T1–3 | USSR | 2.6 | 2.6 | 2.4 | 0.21 | 0.9% |
| top two regions, T1–3 | US | 2.6 | 2.8 | 3.1 | 0.17 | 0.9% |
| top two regions, T1–3 | USSR | 3.8 | 3.7 | 3.8 | 0.19 | 2.6% |
| coup targets, T1–3 | US | 14.5 | 15.0 | 17.5 | 0.23 | 15.0% |
| coup targets, T1–3 | USSR | 8.6 | 9.5 | 11.1 | 0.20 | 17.4% |

*AR1* is one token per game and side:
1. the coup target, if the side coups;
2. otherwise the regions its Ops influence went to, as a set;
3. otherwise a realignment target, a space play, an event, or the scoring card played.

*Coup targets* pools every coup in turns 1–3. The sequence of coups was mostly the deal: it had
about 1,100 distinct values.

## Where the distributions differ

* **USSR AR1: Italy, not Iran.** The network coups Italy in 51% of games and Iran in 33%. The humans
  coup Iran in 51% and Italy in 17%. On the R34 soup the Iran coup priced at −1.4 ± 0.2 against the
  network's pick ([below](#earlier-pricing-this-entry-relies-on)). The network never opens with USSR
  influence into Asia, or into Asia and the Middle East, which humans do in 10.6% of games.
* **US AR1: Europe and Asia together.** Influence into Europe and Asia is the network's AR1 in 34% of
  games and the humans' in 11%. The humans' commonest is Asia and the Middle East, 22% against 12%.
* **Europe weighs more by turn 3 on both sides.** Europe takes 32% of the US influence added in
  turns 1–3 (humans 26%) and 29% of the USSR's (humans 22%). The humans put more into Asia, the
  Middle East and Africa. Europe is the main region in 42% of the network's games against 22–25% of
  the humans'.
* **Coup targets.** The network couped targets that humans rarely coup: as US, Syria 8% against 1%
  and Venezuela 7% against 2%; as USSR, Italy 15% against 6%. About 15–17% of human coups go to
  targets the network couped in under 1% of its games.
* **Counts per game, turns 1–3.** These match within about 0.3 a game: coups, battleground coups,
  space plays and influence added. The exception is US realignment rolls, 1.1 against the humans' 0.5.
* **Whole games.** The network coups less (US 7.1 against 10.6 coups a game, USSR 9.9 against 12.2)
  and plays space less (about 4 against 5). Its games are also shorter: 7.9 turns against 9.4 for the
  human games recorded to the end, of which 74% reach turn 10, against 41% of the network's. Per
  turn, the US coups a quarter less (0.86 against 1.11) and both sides play space 10–15% less (US
  0.50 against 0.57, USSR 0.45 against 0.53). USSR coups per turn are level (1.23 against 1.29).

## Earlier pricing this entry relies on

These turn-1 measurements were made on the R34 soup (`E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M`),
the A13+R48 line's parent. Their write-ups are on branch `exp/setup-matrix`, not on main:
`research/log/E7_setup_by_hand.md` and `research/log/E7_turn1_questions.md`.

* **Setups.** The 3 × 3 compared the soup's own setups, the human ones and R32's: 900,000 greedy
  games, paired by deal. In it R34's own setups were best or tied:
  * USSR Yugoslavia 3, Poland 3: +0.72 ± 0.13 over the human Poland 4, East Germany 1, Austria 1;
  * US West Germany 4, France 2, Italy 2, Iran 1: +0.08 ± 0.09 against the human West Germany 3,
    France 3, Italy 2, Iran 1.

  No out-of-fold rule that varies the setup with the hand beat the fixed setup (best +0.03 ± 0.05).
* **The USSR AR1 coup.** The soup coups Italy in 45% of games and Iran in 15%. The Iran coup priced at
  −1.4 ± 0.2 against its pick, so the line's Italy preference is not an unexplored alternative.

Two caveats. A13+R48's US setup (Italy 3, Panama 1) is not R34's. And both prices are values to the
network's own continuation, which never trained on the human setups: the shallow line's critic
misjudged unfamiliar openings by 12–19 points ([E7_shallow_setup_lock](E7_shallow_setup_lock.md)).
"Cheap to the network" is therefore measured; "cheap against a player who exploits it" is part (b)'s
question.

## What this does not say

* **The humans are a population; the network is one agent.** 265 games by many players, of mixed
  strength, include several styles and their mistakes. Lower entropy than that pool is expected of
  any single strong player and is not by itself a cost.
* **Greedy is not the training policy.** Training self-play samples from the policy, which is
  broader. This entry measures the player that is evaluated and that an exploiter would face.
* **Breadth is not exploitability.** Only part (b), a fresh seat-locked exploiter against the line
  scored against the line's own self-play per seat
  ([E7_league_exploiters_by_seat](E7_league_exploiters_by_seat.md): +11.6 points for a US-locked
  exploiter from 6,400M), measures whether the narrowness can be punished. It needs the GPU box.
