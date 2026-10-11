# Turn 1 on R32 (2026-10-10/11): the human US setup is worth ~1 point; the action rounds leave ~3 points a side

**Question (owner).** A deep solve of turn 1. It has two parts:

* **The setup.** The E7 networks make the setup the same way on every deal, whatever the hand:
  * the US puts France 6, Italy 2 and Iran 1 and leaves West Germany empty;
  * the USSR puts Poland 3, Hungary 2 and Austria 1 and leaves East Germany at 3.

  The heads soup and R32's SWA do this on all 254 deals of the ts-replayer games that have a full
  setup. Humans play other setups. Which wins when the strongest network plays the rest of the
  game? It should be resolved to 1 point.
* **The headline and action rounds.** How much does the network leave at each kind of turn-1
  decision, priced option by option?

**Answer.**

* **The US setup: West Germany 3, France 3, Italy 2, Iran 2 beats the networks' own by +0.90 ±
  0.15 points.** That is 400,000 games over 100,000 deals, paired by deal, and the lead is
  positive in every block of 10,000 deals.
  * Against the networks' USSR setup the lead is +1.23 ± 0.21.
  * Against the human East Germany 4, Poland 4, Austria 1 it is +0.58 ± 0.21.
* **The USSR setup: East Germany 4, Poland 4, Austria 1 pays only against the human US setup.**
  It gains +0.64 ± 0.20 there and 0.00 ± 0.20 against the networks' US setup (an interaction of
  −0.65 ± 0.29).
  * **The solved 2 × 2** is both human setups in every bootstrap resample, worth 47.8% [47.5,
    48.1] to the US. The networks' own pairing gives 47.2%.
  * Neither network ever trained from these setups. The US gain is measured under play that is
    less at home there (below), so it is if anything an underestimate.
* **The first 4 × 4 at 5,000 deals a cell could not resolve this.** A joint test gave p = 0.44.
  Its apparent lead for the networks' US setup over the human one against the networks' USSR
  setup was those 5,000 deals: the 2 × 2 replays them as identical games (4,998 of 5,000, every
  winner the same) and the other 95,000 deals reverse it.
* **The action rounds leave ~3 points a side per game.** Every option of every turn-1 decision was
  priced to the turn's end over 96 worlds, 20,112 decisions in 500 games. Choosing the best option
  on half the worlds and measuring it on the other half gives these gains per game:

  | | from the networks' setup | from the human setups |
  |:---|---:|---:|
  | US | +3.0 | +6.3 |
  | USSR | +3.1 | +6.5 |

  * **Where:** mostly **which card to play** (US +2.2, USSR +1.9 per game from the networks'
    setup) and **event or Operations** (+0.7, +1.0).
  * **Where not:** headlines (−0.1, −0.3) and coup targets (−0.2, −0.1) are within noise of the
    priced best.
  * From the human setups the same kinds roughly double: card to play +3.3 and +4.3, US headline
    +0.6, US first influence point +0.9.
* **The critic's turn-end verdicts hold to the end of the game.** The 60 clearest departures of
  each run were replayed to the end, paired, with no critic:
  * from the networks' setup: a mean of +7.5 ± 0.6 points (turn-end +7.4); 42 of 60 hold at 2 SE,
    none reverse;
  * from the human setups: +7.2 ± 0.6 (turn-end +8.4); 40 hold, none reverse.

  Turn-end and full-game gains correlate at 0.69 and 0.64.
* **Recurring departures,** from the networks' setup (the full lists are linked below):
  * **Olympic Games** played by the USSR for Operations or a coup rather than as the event, three
    times among the 60 (+10 to +12 points each).
  * **First influence points** are 28 of the 60, with no single direction. 8 move a Southeast Asian
    point elsewhere (Thailand to North Korea +15, Malaysia to Iraq +17), 5 move one into the region
    (France to Thailand +16), and 3 shift it within the region.
  * **Holding Asia or Europe Scoring** a round too long (+8, +14.5).
  * **Blockade.** The US with no influence in West Germany discards a 3-Ops card to it in 26 of 49
    cases, where passing loses nothing. In the two banked cases passing was worth +8.4 and +14.4 to
    the end of the game. The cards discarded were USSR cards (Socialist Governments, Suez Crisis),
    so this may be a deliberate dump; it is for the owner's review.

## The setup: design

* **Setups** (`tools/lib/openings.py`; influence added; the share is of the 254 human games with a
  full setup). The 2 × 2 is U0, U2 against S0, S1.

| | US | | USSR |
|:---|:---|:---|:---|
| U0 | France 6, Italy 2, Iran 1 (the networks') | S0 | Poland 3, Hungary 2, Austria 1 (the networks') |
| U1 | West Germany 4, Italy 4, Iran 1 (48%) | S1 | Poland 4, East Germany 1, Austria 1 (45%) |
| U2 | West Germany 3, France 3, Italy 2, Iran 1 (31%) | S2 | Poland 4, East Germany 1, Yugoslavia 1 (30%) |
| U3 | West Germany 4, Italy 3, Iran 2 (5%) | S3 | Poland 5, East Germany 1 (24%) |

* **Player.** R32's SWA 6,720–6,800M (sha256 `1d4f65f0…`), greedy, both sides after the setup. It
  is the strongest network on the E7 leaderboard (1645 against the heads soup's 1628).
* **Design** (`tools/setup_matrix.py`, workflow `setup_matrix.yml` on the fork):
  * Each entrant has both setups forced (`opening:<us>+<ussr>:<checkpoint>`).
  * A pairing of A = (US a, USSR d) with B = (US c, USSR b) plays cell (a, b) with A as US and
    cell (c, d) with B as US (`tools/tournament.py --matchups`).
  * Game pair k of every pairing deals from the same seed, and the engine deals before the setup,
    so every cell is played on the same deals and every comparison is paired by deal.
* **The runs:**
  * the 4 × 4: run `38117828774` at commit `0216a71`, 5,000 deals a cell, 80,000 games, 60
    runner-minutes;
  * the 2 × 2: run `38121015201` at commit `4cdbfe3`, 100,000 deals a cell, 400,000 games.
* **Precision of the 2 × 2:** a cell difference is ±0.21 points, a margin ±0.15, the interaction
  ±0.29. A cell's outcomes on a deal correlate with the other cells' at only 0.14, and draws are
  1.2%.

### The 2 × 2 (US score, %, each cell ± 0.16)

| US \ USSR | S0 (the networks') | S1 (East Germany 4, Poland 4, Austria 1) |
|:---|---:|---:|
| U0 (the networks') | 47.18 | 47.19 |
| U2 (West Germany 3, France 3, Italy 2, Iran 2) | 48.40 | 47.77 |

The 10 blocks of 10,000 deals give the US margin as +0.7, +1.1, +1.1, +0.7, +1.2, +1.1, +0.3,
+1.0, +1.1, +0.8 (block SE 0.48; χ² 2.8 on 9 degrees of freedom).

### The 4 × 4 (US score, %, each ± 0.7; first 5,000 deals)

| US \ USSR | S0 | S1 | S2 | S3 | mean |
|:---|---:|---:|---:|---:|---:|
| U0 | 48.4 | 46.4 | 46.4 | 47.6 | 47.2 |
| U1 | 47.1 | 46.5 | 46.2 | 46.6 | 46.6 |
| U2 | 47.3 | 47.7 | 47.3 | 48.5 | 47.7 |
| U3 | 46.8 | 47.2 | 46.9 | 47.5 | 47.1 |

A joint test that all 16 are equal, blocked by deal, gives Hotelling T² = 15.2 on 15 contrasts
(p = 0.44). The margins against the networks' setups are all within 1.5 SE (± 0.5).

## Turn 1 priced: design

* **Tool and runs.** `tools/turn1_pricing.py`, workflow `turn1_pricing.yml`, commit `4cdbfe3`:
  * from the networks' setup: run `38121019708`;
  * from the human setups (`us_wg3_fr3_it2+ussr_pol4_aus`): run `38121024513`.

  Each is 500 greedy R32 games (seeds 120,000+), stopped at the end of turn 1. The report step of
  both runs failed on a shell-quoting bug in its provenance line (fixed in `bdc4283`); the reports
  below were built from the runs' own row, bank and confirmation artifacts with the same code.
* **What is priced:** every turn-1 decision with 2–12 options (headline, card to play, event or
  Operations, Operations mode, coup target, an event's other choices), and the first point of every
  influence placement, realignment and event placement (up to 60 options).
  * Every option is played over 96 worlds to the end of the turn by the network, greedy on both
    sides, and valued by its critic from the decider's side there.
  * A world redeals the decider's unseen cards, the same for every option; the dice are each
    option's own.
* **Gains** are in points of the decider's score, half a value unit. Each kind is reported two ways:
  * **upper bound:** the priced best's lead over the network's move on all worlds;
  * **honest:** the best chosen on half the worlds and measured on the other half, averaged both
    ways.
* **Confirmation.** The 60 departures with the largest z (≥ 3, at most 3 per kind and card) were
  replayed to the end of the game by `ai.eval.paired_playouts.compare`: 512 pairs, both moves from
  the same redeal and dice, the network greedy, no critic.

Full reports, with every departure and a workbench link to its position:
[from the networks' setup](E7_setup_matrix/turn1_own_setup.md) ·
[from the human setups](E7_setup_matrix/turn1_human_setups.md).

### Honest gain per game, by kind (points)

| kind | US, networks' setup | USSR, networks' setup | US, human setups | USSR, human setups |
|:---|---:|---:|---:|---:|
| headline | −0.08 ± 0.08 | −0.34 ± 0.07 | +0.63 ± 0.11 | −0.03 ± 0.09 |
| card to play | +2.24 ± 0.18 | +1.88 ± 0.17 | +3.34 ± 0.21 | +4.28 ± 0.22 |
| event or Ops | +0.74 ± 0.15 | +1.00 ± 0.13 | +1.19 ± 0.17 | +1.14 ± 0.15 |
| Ops mode | +0.11 ± 0.07 | +0.10 ± 0.06 | +0.24 ± 0.07 | +0.14 ± 0.06 |
| coup target | −0.16 ± 0.06 | −0.10 ± 0.07 | −0.08 ± 0.06 | −0.19 ± 0.06 |
| influence, first point | +0.12 ± 0.16 | +0.49 ± 0.11 | +0.87 ± 0.17 | +0.49 ± 0.11 |
| event placement, first point | −0.02 ± 0.05 | +0.06 ± 0.07 | +0.12 ± 0.06 | +0.63 ± 0.09 |

The network's move is the priced best at:
* the headline in 60–67% of decisions (8 options);
* card to play 39–46% (4–5);
* event or Ops ~62% (5);
* Ops mode ~76% (3);
* coup target ~74%;
* an influence first point 22–29% (26–40 options).

## Limitations

* **Every value is to R32's own play.** That is greedy for the setup matrix, and greedy after the
  priced move for the pricing. A move whose point lies in a follow-up R32 does not play is
  undervalued.
* **Honest gains are not additive.** Each prices one decision with the network's play after it.
  The per-game sums say how much one-decision pricing finds, not what a player re-priced at every
  turn-1 decision would gain.
* **The critic at the turn's end is unbiased on the departures that were replayed** (ratio 1.01
  and 0.85). It has not been checked on the typical decision, where the gaps are a fraction of a
  point.
* **The setups are scored over all deals.** A hand-dependent choice is measured in
  [E7_setup_by_hand.md](E7_setup_by_hand.md): on R32 it pays +0.3, on the R34 soup nothing.

## Next

* **A setup-mix test in training.** `--setup-script-frac` / `--forced-opening` exist, and the
  measured gain says the US's West Germany setup is the one to teach. Whether training from it
  widens the +0.9 is the owner's call.
* **Price later turns.** Turn 1's decisions leave ~3 points a side. The same tool with a turn
  filter would show whether the card-to-play and event-or-Ops losses persist.
* **Review the Blockade and Olympic Games departures** (owner).

## Replicate

```bash
gh workflow run setup_matrix.yml --ref exp/setup-matrix -f us_setups="us_e7 us_wg3_fr3_it2" \
  -f ussr_setups="ussr_e7 ussr_pol4_aus" -f games_per_cell=100000 -f shard_pairs=250
gh workflow run turn1_pricing.yml --ref exp/setup-matrix -f games=500 -f worlds=96
gh workflow run turn1_pricing.yml --ref exp/setup-matrix -f games=500 -f worlds=96 \
  -f opening=us_wg3_fr3_it2+ussr_pol4_aus
```
