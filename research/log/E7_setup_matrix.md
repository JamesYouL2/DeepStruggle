# The setup matrix (2026-10-10): under R32's own play, the setup is worth less than a point

**Question (owner).** A deep solve of turn 1, starting with the setup. The E7 networks make the
setup the same way on every deal, whatever the hand: the US puts France 6, Italy 2 and Iran 1 and
leaves West Germany empty, and the USSR puts Poland 3, Hungary 2 and Austria 1 and leaves East
Germany at 3. The heads soup and R32's SWA do this on all 254 deals of the ts-replayer games that
have a full setup. Humans play other setups. Which of them wins when the rest of the game is
played by the strongest network?

**Answer.**

* **Under R32's greedy play, these setups cannot be told apart.**
  * The 16 cells span 46.2–48.5% for the US, each ± 0.7.
  * A joint test that all 16 are equal, blocked by deal, gives Hotelling T² = 15.2 on 15
    contrasts (p = 0.44).
* **No setup averaged over the other side's setups is 1.5 SE from the networks' own.** In points of
  the side's own score:
  * US setups against the networks' US setup: −0.6, +0.5 and −0.1, each ± 0.5;
  * USSR setups against the networks' USSR setup: +0.4, +0.7 and −0.2, each ± 0.5.
* **The largest single effects are about 2 points and about 2 SE.** That is in line with chance
  across 24 comparisons.
  * Against the networks' US setup, the human USSR setups score +1.6 ± 0.8 more than the
    networks' USSR setup.
  * Against the networks' USSR setup, the human US setups score 1.3 ± 0.8 less than the
    networks' US setup.
  * Combined, that is an interaction of +1.7 ± 0.9: each of the networks' setups looks best against
    the other.
* **The networks play the human setups as well as their own**, although they never trained from
  them. Whatever turn 1 is worth, it is not in this choice: the setup is a near-flat decision for
  this player.
* **The solved game is unstable.** It picks West Germany 3, France 3, Italy 2, Iran 1 against
  Poland 4, East Germany +1, Yugoslavia 1, worth 47.3% [46.4, 48.1] to the US. But across
  bootstrap resamples of the deals, every one of the eight setups is in the equilibrium's support
  at least 12% of the time (those two 85% and 77%). That is a flat matrix, not a recommendation.

## Setup

* **Rows and columns** (`tools/lib/openings.py`; influence added; the share is of the 254 human
  games with a full setup):

| | US | | USSR |
|:---|:---|:---|:---|
| U0 | France 6, Italy 2, Iran 1 (the networks') | S0 | Poland 3, Hungary 2, Austria 1 (the networks') |
| U1 | West Germany 4, Italy 4, Iran 1 (48%) | S1 | Poland 4, East Germany 1, Austria 1 (45%) |
| U2 | West Germany 3, France 3, Italy 2, Iran 1 (31%) | S2 | Poland 4, East Germany 1, Yugoslavia 1 (30%) |
| U3 | West Germany 4, Italy 3, Iran 2 (5%) | S3 | Poland 5, East Germany 1 (24%) |

* **Player.** R32's SWA 6,720–6,800M (sha256 `1d4f65f0…`), greedy, on both sides after the setup.
  It is the strongest network on the E7 leaderboard (1645 against the heads soup's 1628).
* **Design.** `tools/setup_matrix.py`, workflow `setup_matrix.yml` on the fork, run
  `38117828774`, commit `0216a71`; 20 runners.
  * Each entrant has both setups forced (`opening:<us>+<ussr>:<checkpoint>`).
  * A pairing of A = (US a, USSR d) with B = (US c, USSR b) plays cell (a, b) with A as US and
    cell (c, d) with B as US (`tools/tournament.py --matchups`).
  * There are 8 pairings of 5,000 deals each, 80,000 games in all.
  * Game pair k of every pairing deals from the same seed, and the engine deals before the setup.
    So every cell is played on the same 5,000 deals, and every difference between cells is
    paired by deal.
  * A cell's outcomes on a deal correlate with the other cells' at only 0.14: the deal decides
    little once the network plays it out.

## The matrix (US score, %)

| US \ USSR | S0 | S1 | S2 | S3 | mean |
|:---|---:|---:|---:|---:|---:|
| U0 | 48.4 | 46.4 | 46.4 | 47.6 | 47.2 |
| U1 | 47.1 | 46.5 | 46.2 | 46.6 | 46.6 |
| U2 | 47.3 | 47.7 | 47.3 | 48.5 | 47.7 |
| U3 | 46.8 | 47.2 | 46.9 | 47.5 | 47.1 |
| mean | 47.4 | 47.0 | 46.7 | 47.6 | 47.2 |

Each cell ± 0.7; a paired difference between two cells ± 0.9.

## Limitations

* **This is the value of a setup to R32's greedy play.** A setup whose point lies in a follow-up
  R32 does not play is undervalued here. The human setups were not undervalued enough to lose,
  though: they come out level.
* **Four setups a side.** These are the networks' own and the most common human ones. Setups
  nobody plays (West Germany 1, a hand-dependent choice) are not measured.
* **No hand conditioning.** Each setup is scored over all deals. A setup that is right with some
  hands and wrong with others averages out here.
* **Resolution.** A difference of 1 point between two setups is ±0.5 on the margins and ±0.9 per
  cell. Effects smaller than about 1.5 points would need several times the 80,000 games.

## Next

* **Turn 1's value is elsewhere.** The headline and the action rounds can be priced
  option-by-option over paired worlds (`ai/eval/option_pricing.py` to the turn's end, or full-game
  playouts), from both the networks' setup and the human ones.
* **A searched continuation** (`setup_matrix.yml -f player=gumbel:{BASE}:256:8`) on a 2 × 2 of the
  two closest rivals would test whether search opens a gap the greedy network does not see. At
  about 0.36 runner-minutes a Gumbel game, 1,000 deals a cell costs about 1,400 runner-minutes and
  resolves only gaps of 4+ points.

## Replicate

```bash
gh workflow run setup_matrix.yml --ref exp/setup-matrix -f games_per_cell=5000 -f runners=20 -f shard_pairs=50
gh run download <run> -n setup-matrix-<run>      # matrix.md, matrix.json, games.jsonl.gz, plan.json
```
