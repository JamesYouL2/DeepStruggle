# The catastrophic-blunder census on the R34 soup (2026-10-12): the turn-10 placement leak is no longer where the points go; 31 confirmed holes, led by DEFCON-2 suicide cards and rare mechanisms

**Question (owner).** The [census](E7_blunder_census.md) and [blunder bank](E7_blunder_bank.md) asked
where the heads soup throws a game away. Ask it again of the current strongest network, the R34
gap-penalty soup `E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M` (E7 leaderboard 1650 ± 4,
against the heads soup's 1625).

**Answer.** It throws games away about as rarely as the heads soup did, but in different places.

* **Engine-proven mistakes: 0.75 per 1,000 decisions** (71 of 94,158). They cost much less than
  the heads soup's: **2.5 game-points over the 200 games, against 4.7.** Only 1 costs a third of a
  game or more, against 7. 57 of the 71 cost nothing in the paired continuations.
* **The heads soup's recurring leak, the US's last influence placement of turn 10, costs nothing
  in R34's own games** (8 such rows, 0.00 points, against 3.66). It is not gone:
  * On the heads soup's banked positions, R34 still plays 4 of the 5 proved last-placement
    blunders, and the fifth move scores the same as the blunder.
  * 6 turn-10 placements are among R34's own confirmed holes.
* **The confirmed holes: 31 positions in 24 of the 200 games, 9.6 game-points**, about 0.05 a
  game, a floor. They come from 50 bank rows: 51 confirmed by both judges at 512 pairs, less
  one that `settle` found won whatever is played. 29 hold with search on both sides of the
  continuation, and 2 more are likely. The largest:
  * **DEFCON-2 suicide cards.** The network spends its safe outlet and is left to play an
    opponent card whose event gives the USSR a coup in the US's own round:
    * spacing Muslim Revolution instead of Ortega Elected in Nicaragua (64 points);
    * discarding Socialist Governments to Blockade, which leaves "Lone Gunman" as the last card
      of the turn (62 points).
  * **Rare mechanisms at the end of a turn.**
    * Passing North Sea Oil's eighth round with a Military Ops shortfall at turn 10 (96 points;
      the pass is an engine-proven loss).
    * Five Year Plan at turn 6, AR7, holding Southeast Asia Scoring: the random discard misses
      the scoring card half the time, and holding it loses (42 points).
  * **Turn-10 arithmetic, 12 of the 31:**
    * the game's last card (Colonial Rear Guards for Nixon, 53 points);
    * East European Unrest for Ops instead of space (21 points);
    * placements (Italy over Argentina 47 points, Italy over Poland 41 points).
  * **Headlines (5).** Search cannot fix these: the better headline is a bet on the opponent's
    simultaneous choice.
* **Search fixes most of them.** On R34's bank, R34 with Gumbel@256 repeats 10 of the 50 (2.8 of
  14.0 points), and the rollout root repeats 15. 4 of Gumbel's 10 are headlines.
* **The holes belong to the network.** The heads soup repeats only 20 of R34's 50, and R34
  repeats 14 of the heads soup's 35 (R32 repeated 24).
* **A measurement flaw, in both censuses: an engine label at a headline reads the opponent's
  committed headline.** The mover cannot see that card. On R34 this puts 1.3 game-points of
  headline rows into the "engine-proven" class. 4 of them are USSR "wins" with We Will Bury You
  or KAL-007 that need the US's own committed Duck and Cover. See
  [Headline labels](#headline-labels). Not fixed here.

## Method

The tools and settings of the heads soup's run, unchanged
([`E7_blunder_census.md`](E7_blunder_census.md), [`E7_blunder_bank.md`](E7_blunder_bank.md);
`tools/README.md` §7d). Engine `b670b258`, the build the E7 leaderboard accepts. Branch
`exp/blunder-census-r34`, commit `904ca11`.

* **Model.** `E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M.pt`, sha256 `f1fb539c6ad9…`.
  The Hugging Face copy is `_models/…`.
* **Stage 1, screen (local, 409 s).** 200 greedy self-play games, seed 0, so the deals are the
  heads soup's. Every decision with two or more legal moves (94,158) was checked two ways:
  `classify_legal_actions`, with the How I Learned fix, and a Gumbel k=4 @32 root.
  * 23,199 candidates: 30 forced losses, 41 missed forced wins, 23,180 Gumbel disagreements.
  * A 5% control of the rest: 3,565.
* **Stage 2, validate (fork CI `38269720407`, 20 runners, 717 runner-minutes).** Gumbel k=8 @256
  over three seeds; the moves paired out over 32 greedy continuations. The 100 most serious
  unresolved rows were escalated to 1,024 simulations (local, 361 s); this moved 2 rows.
* **Bank.**
  * `select`: 795 rows at 0.10+ points, z ≥ 2. Release `blunder-shortlist-r34-20261012`, sha256
    `a08a4986…`.
  * `confirm`: 512 fresh pairs, seed 7, under two judges: the R34 soup (CI `38275187103`) and
    the heads soup (CI `38275196883`).
  * `build`: 0.10+ points at z ≥ 3 under both judges, which leaves 51.
  * Verification of those 51 (release `blunder-bank-r34-51-20261012`, sha256 `ad334456…`):
    * searched continuations, R34's Gumbel k=8 @256 on both sides, 96 pairs, seed 23 (CI
      `38278656854`);
    * `prove` at 2,000,000 nodes, one row per runner (CI `38278670141`); `settle` then dropped 1.
  * The bank is [`ai/eval/banks/blunders_E7_R34.jsonl`](../../ai/eval/banks/blunders_E7_R34.jsonl)
    with each row's `verdict`. The table, with workbench links, is
    [bank](E7_blunder_census_R34/bank.md).
* **Score.** Local, 256 pairs, seed 11. Two tables:
  * [R34's bank](E7_blunder_census_R34/score_R34_bank.md), with R34 as the judge of other moves;
  * [the heads soup's bank](E7_blunder_census_R34/score_E7_bank.md), with the heads soup as the
    judge, as in that bank's own table.

## Results

### The census, against the heads soup's

| figure | heads soup (relabelled) | R34 soup |
|:---|---:|---:|
| decisions screened (200 games) | 98,002 | 94,158 |
| Gumbel@32 disagrees with greedy | 14.0% | 24.6% |
| candidates / control | 13,875 / 4,256 | 23,199 / 3,565 |
| confirmed catastrophic blunders per 1,000 decisions | 22.95 | 21.66 |
| — exact terminal mistakes (engine-proven) | 0.60 | 0.75 |
| — estimated strategic regret (upper bound) | 22.34 | 20.90 |
| control rows clearing z ≥ 2 (the noise floor) | 2.7% | 2.2% |
| candidate rows clearing z ≥ 2 | 3.9% | 4.4% |
| share of confirmed mistakes the cheap screen found | 20.0% | 39.5% |
| share rescued by 256-simulation search | 41.8% | 56.5% |
| validation runner-minutes | 400 | 717 |

R34's Gumbel@32 disagreement rate is 1.8 times the heads soup's. Recipe R34 penalises the gap
between the top logits, which leaves a flatter policy that a small search overturns more often.
So R34's screen sends more positions to validation, and finds more. Read the per-1,000 figures
next to their screens.

### The exact class, by what the paired continuations say it cost

Cost here is the best alternative's score minus greedy's, out of 1. Points are those costs
summed over the 200 games.

| | heads soup: rows (free / small / costly) | points | R34: rows (free / small / costly) | points |
|:---|:---|---:|:---|---:|
| last placement of turn 10 (AR 7+) | 13 (6 / 0 / 7) | 3.66 | 8 (8 / 0 / 0) | 0.00 |
| other turn-10 last-round decisions | 7 (7 / 0 / 0) | 0.00 | 10 (9 / 0 / 1) | 0.91 |
| headlines (see below) | 12 (8 / 4 / 0) | 0.28 | 10 (3 / 6 / 1) | 1.30 |
| everything else | 27 (18 / 9 / 0) | 0.80 | 43 (37 / 6 / 0) | 0.33 |
| **all** | **59 (39 / 13 / 7)** | **4.73** | **71 (57 / 12 / 2)** | **2.53** |

Free is ≤ 0.01, small is 0.01–0.3 and costly is ≥ 0.3. As before, most free rows are forced
losses taken where every alternative also lost, and forced wins taken a move later. Without the
headlines, R34 gives away 1.2 points against the heads soup's 4.5.

### The bank: where R34 is confirmed wrong

The 50 rows have these verdicts. No row was proved exactly:

* 25 need a new turn (a new deal).
* 26 exceed the 2,000,000-node budget. That includes the North Sea Oil pass, whose pass the
  census's own forced probe already labels a loss.

| verdict | rows | points (searched) | what it means |
|:---|---:|---:|:---|
| `holds-under-search` | 29 | 9.32 | 10+ points at z ≥ 3 under both judges and the searched continuation |
| `likely` | 2 | 0.23 | positive at z ≥ 2 under the searched continuation |
| `continuation-dependent` | 19 | 0.58 | the cost is gone when the follow-up searches |

**The 31 that hold or are likely, by kind** (US 16, USSR 15; 12 at turn 10):

* **Card choice (12).** The biggest rows:
  * the North Sea Oil pass, 96 points;
  * the Blockade discard, 62;
  * Colonial Rear Guards over Nixon as the game's last card, 53;
  * Brush War over Grain Sales as the US's Star Wars pick at turn 8, the USSR on 17 VP, 49;
  * Five Year Plan over Southeast Asia Scoring, 42.

  Three play a scoring card early in a round where another card should go first:
  * South America Scoring at turn 8, AR1;
  * Central America Scoring at turn 5, AR3;
  * Africa Scoring at turn 8, AR2.
* **Influence placement (10).** Six are at turn 10 (Italy over Argentina 47, Italy over Poland
  41). Three of them come from one game, seed 144, a late fight over Argentina, so the
  repetition of Argentina is one game rather than a pattern. The others are mid-game: Libya over
  Iran for NATO's placement at turn 7, Egypt over Jordan with ABM Treaty's, Panama over Guatemala
  with We Will Bury You's.
* **Play mode (4).**
  * Muslim Revolution spaced rather than let fire, 64.
  * East European Unrest for Ops in the game's last round rather than spaced, 21.
  * Nuclear Test Ban for Ops rather than as the event, 27.
  * Warsaw Pact for Ops rather than as the event, 7.
* **Headline (5).**
  * Three USSR headlines where a DEFCON-degrading 4-Ops event was better: We Will Bury You at
    turns 5 and 7, KAL-007 at turn 10.
  * Two US headlines where Nuclear Test Ban was better: at turn 5, and at turn 8 on 19 VP.

**The DEFCON-2 suicide cards.** Under the rules (`rules.json`: "if DEFCON drops to 1 during Ops
or during Opponent Event resolution, the phasing player loses"), a card whose event gives the
opponent a coup must not be played in your own round at DEFCON 2. It can only be held over or
spaced, and the turn's single space attempt is often the only way out.

* **Muslim Revolution, US, turn 9, AR3.** The US holds Five Year Plan, Muslim Revolution, The
  Voice of America, Ortega Elected in Nicaragua and Solidarity: 5 cards for 5 rounds.
  * The network spaces Muslim Revolution and uses the turn's only space attempt.
  * Ortega must then be played. Its free coup in Cuba (US 3, USSR 1) ends the game.
  * Spaced: 1 point; Muslim Revolution let fire: 57 points.
* **Socialist Governments, US, turn 4, AR7.** The USSR's Blockade asks for a 3-Ops discard. The
  US holds Socialist Governments and "Lone Gunman".
  * Discarding leaves "Lone Gunman" as the US's AR7 card, and the US lost all 512 pairs.
  * Declining costs nothing, since the US has no influence in West Germany, and it keeps
    Socialist Governments to play: 55 points.

**What search fixes.** The [score](E7_blunder_census_R34/score_R34_bank.md) on R34's own bank:

| player | repeats | finds the better move | other | points repeated |
|:---|---:|---:|---:|---:|
| R34 soup, greedy | 50 | 0 | 0 | 13.99 |
| heads soup, greedy | 20 | 17 | 13 | 5.42 |
| R34 soup, Gumbel k=8 @256 | 10 | 29 | 11 | 2.81 |
| R34 soup, rollout root (`4:16:4:z2`) | 15 | 27 | 8 | 3.46 |

As on the heads soup's bank, the bank favours Gumbel by construction: the census's Gumbel@256
nominated most of its better moves. Gumbel@256 still repeats 10:

* 4 headlines;
* Five Year Plan over Southeast Asia Scoring;
* the Star Wars pick;
* Central America Scoring over OPEC;
* North Sea Oil's Ops for influence rather than a coup;
* Africa Scoring over Flower Power;
* North Korea over Iran with Nuclear Test Ban's Ops.

**R34 on the heads soup's bank** ([score](E7_blunder_census_R34/score_E7_bank.md)):

| player | repeats | finds the better move | other (lead on the blunder) | points repeated |
|:---|---:|---:|---:|---:|
| heads soup, greedy | 35 | 0 | 0 | 9.11 |
| R34 soup, greedy | 14 | 9 | 12 (+0.04) | 4.42 |
| R34 soup, Gumbel k=8 @256 | 6 | 18 | 11 (+0.10) | 1.13 |
| R34 soup, rollout root | 7 | 13 | 15 (+0.10) | 1.56 |

* **R34 greedy avoids 21 of the heads soup's 35 blunders, but its other moves gain little** (+0.04
  on the blunder). R32 avoided 11, with other moves worth −0.10.
* **The five proved turn-10 last placements:** R34 repeats Poland, Latin American Debt Crisis,
  Libya and Angola. Its fifth move, Canada for Ethiopia, scores level with the blunder.
* **Gumbel@256 on R34 repeats 6 of the 35** (heads soup 10, R32 4).

### Headline labels

`classify_legal_actions` labels each legal move on the true state. At a headline the true state
holds the opponent's committed headline (`headline_us_card` or `headline_ussr_card`). The
mover's search (`_search_win`) then resolves both headlines against it.

* **4 of the 6 USSR "wins" among R34's exact rows need the US's committed Duck and Cover
  (card 4).** A 4-Ops DEFCON degrader (We Will Bury You, KAL-007) resolves first, at DEFCON 3.
  Duck and Cover then takes DEFCON to 1 in the US's own event, and the US loses.
* **The other 2 are wins only against the specific US headline committed**: Africa Scoring on
  18 VP and Nuclear Test Ban on 19.
* **The two US "wins" with Nuclear Test Ban at 19 VP read the USSR's committed headline.**
* In 16 worlds per row with the opponent's headline forgotten (`paired_playouts.pair_start` →
  `determinize`, fixed in `b1ca882`), none of the 8 labels holds.

The paired playouts are not affected: each pair redeals the opponent's hand and asks the
opponent's headline again. So these rows' costs are honest measures of R34-as-US walking into
the trap in self-play. Their "engine-proven" label is not. The heads soup's census carries the
same kind of row (12 headline rows, 0.28 points).

A fix would label headline moves on a determinized state. The opponent's choice would then stop
the win search, as it does mid-round. Not made here.

## Reading

* **R34 is stronger without being tidier.**
  * It rates 25 Elo above the heads soup.
  * Its engine-proven mistakes cost about half as much, and the arithmetic leak at turn 10's last
    placement no longer shows in its games.
  * Its confirmed holes are more numerous (31 against 26), and so is their cost (9.6 points
    against 7.1). The wider screen accounts for part of that.

  Both are floors on 200 games, not rates. The pattern agrees with the earlier finding that
  strength is in diffuse evaluation, not in a tail of blunders.
* **The new holes are about what the network must do *later in the turn*.**
  * Keeping a safe outlet for a DEFCON-2 suicide card.
  * Not ending a round holding a scoring card.
  * Using North Sea Oil's extra round to meet Military Ops.
  * Spending the game's last card.

  These are plans over the rest of a turn. Gumbel@256 finds most of them, and the rollout root a
  little fewer.
* **Headlines are the class search cannot reach.** 4 of the 10 rows Gumbel repeats are headlines.
  In 3 of the 5 headline holes the better move is a USSR 4-Ops DEFCON-degrading headline at
  DEFCON 3: We Will Bury You twice, KAL-007 once. In the two of them that are also exact rows
  (turns 7 and 10), the US's committed headline was Duck and Cover. That points at a US-side
  habit worth measuring directly: how often R34 as the US headlines Duck and Cover at DEFCON 3.
  The event census has no DEFCON split, so it was not measured here.
* **The last placement.** R34's own games no longer lose points at the last placement. On the
  heads soup's positions it still repeats 4 of the 5 proved rows, and 6 turn-10 placements hold
  in its own bank. The deterministic last-placement solver proposed in the first census still
  applies, at no training cost.

## What this does not say

* **Not a rate.** One network, greedy self-play, 200 games. The holes are a floor, and the two
  networks' screens differ in width (24.6% against 14.0% disagreement), so their bank sizes do
  not compare as rates. 5 games contribute 2 or more of the 31 rows; one game contributes 4.
* **No exact proof.** `prove` settled one row (won whatever is played) and proved none. The
  `holds-under-search` rows are robust to the greedy follow-up, not proof that the move itself is
  the error.
* **Judges.** R34's Gumbel plays both sides of the searched continuation. A flaw both judges
  share stays in the bank.

## Replicate

```bash
export PYTHONPATH=.:build/release
M='data/checkpoints/hf/E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M.pt'
H='data/checkpoints/hf/E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt'
tools/scripts/check_engine_fresh.sh
.venv/bin/python tools/scripts/blunder_census.py screen --model "$M" --games 200 --seed 0 --out-dir screen/
gh release create blunder-input-<date> screen/candidates.jsonl.gz screen/control.jsonl.gz --prerelease
gh workflow run bank_playouts.yml --ref exp/blunder-census-r34 -f stage=validate \
  -f release=blunder-input-<date> -f model='_models/E7-A8-R1-S44@6400M+(R34,R34-S45,R34-S46)@7120..7200M.pt' -f pairs=32
.venv/bin/python tools/scripts/bank_playouts.py validate --input screen/candidates.jsonl.gz screen/control.jsonl.gz \
  --model "$M" --pairs 32 --seed 1 --search-sims 1024 --escalate-from validation.jsonl.gz --escalate-cap 100 \
  --out escalation.jsonl.gz
.venv/bin/python tools/scripts/blunder_census.py report --screen screen/candidates.jsonl.gz \
  --control screen/control.jsonl.gz --summary screen/summary.json \
  --validation validation.jsonl.gz --escalation escalation.jsonl.gz --out report.md
.venv/bin/python tools/scripts/blunder_bank.py select --validation validation.jsonl.gz --min-gap 0.1 --min-z 2 \
  --out shortlist.jsonl.gz
# confirm once per judge (stage=confirm, pairs=512): the R34 soup and the heads soup
.venv/bin/python tools/scripts/blunder_bank.py build --confirm 'r34=confirm_r34/*.jsonl.gz' \
  'soup=confirm_soup/*.jsonl.gz' --min-gap 0.1 --min-z 3 --out bank.jsonl
# then stage=confirm with -f pairs=96 -f args="--continue-with 256:8 --seed 23", and
# stage=prove with -f runners=51 -f args="--budget 2000000"; then
.venv/bin/python tools/scripts/blunder_bank.py settle --bank bank.jsonl --prove 'prove/*.jsonl.gz' --out bank.jsonl
.venv/bin/python tools/scripts/blunder_bank.py score --bank bank.jsonl --player "r34=$M" "soup=$H" \
  "r34-g256=gumbel:$M:256:8" "r34-rollout=rollout:$M:4:16:4:z2" --judge "$M" --pairs 256 --out score.md
```

The `verdict` field was combined from `exact` (from `prove`) and `searched` (the searched
confirm) by the rule in the verdict table above, as for the heads soup's bank.
