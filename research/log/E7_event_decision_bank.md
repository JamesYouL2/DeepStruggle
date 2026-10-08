# Event-or-not decisions (2026-10-07): the raw network's event choices show no loss that paired playouts can detect, card by card or in total

**Question.** Follows [`E7_search_disagreement_bank.md`](E7_search_disagreement_bank.md). When the
raw network (E7line_swa_4720-4800M) chooses whether to play a card for its event, how much does it
lose? Which cards does it misplay most, and does the loss sit with its own cards, the opponent's
or neutral ones, and with which side?

**Answer.**

* **No loss detectable in total.** These decisions come up 94.6 times a game. Over all of them, a
  move picked by playouts on half the pairs and measured on the other half does *worse* than raw's
  choice: -9.7 ± 4.4 points a game. Raw is better than a 128-pair playout verdict, so the evaluation
  is the limit, not raw.
* **The clear-cut errors are at chance rate.** A clear error is a position where another move
  beats raw's choice by more than 2 SE of the paired difference. The counts match what noise alone
  produces:
  * own cards: 65 errors against 532 positions where raw is clearly right; chance gives ~54
  * neutral cards: 36 errors against 348; chance gives ~27
  * opponent's cards: 82 errors against 228; chance gives ~110 (two alternatives)
* **No card, side or relation stands out.** Every group's loss a game is within about 2 SE of zero
  or negative (raw better than the picker); see the side × relation table below. The largest
  positive card is CIA Created played by the USSR, at +1.09 ± 0.49 a game (2.2 SE, one of 203
  groups). For an opponent's card that is a timing choice, event-first against Ops-first; it is not
  an event-or-not choice.
* **Where raw departs from humans, the playouts side with raw.** Raw almost never events cards
  that humans often do:

  | card | raw AR event rate | human AR event rate | event lead (pts) |
  |:---|---:|---:|---:|
  | Arms Race (USSR) | 1% | 41% | -5.9 |
  | Che | 0% | 58% | -2.9 |
  | Bear Trap | 0% | 49% | -2.3 |
  | Blockade | 2% | 43% | -7.2 |
  | Terrorism (US) | 2% | 53% | -2.8 |
  | Our Man in Tehran | 0% | 40% | -3.0 |

  Under raw's own continuation, the event loses in each of these. Missile Envy (US, 0% against
  55%), Grain Sales (1% against 92%) and Aldrich Ames (0% against 100%, 10 human plays) lean the
  other way, at +1.2 to +1.7 points of regret, but none is 2 SE clear. These are follow-up-dependent
  events, so the playouts are a lower bound for them.
* **The huge negative event leads are DEFCON 2 suicides, and raw already avoids them.** The event
  leads at DEFCON 2 are:
  * Olympic Games: -57 (US), -43 (USSR)
  * KAL-007 (US): -49
  * "We Will Bury You" (USSR): -35
  * Summit: -19 (US), -17 (USSR)

  At DEFCON 3 and above they are about 0. Raw's event rate on all of these is 0%.

**What it implies.** Event-or-not is not where the raw network's error is, at least not at a
resolution of about 0.5 points a decision. The earlier bank's per-category numbers for missed and
wrong events were search-valued (+10 and +17 points a game) but playout-confirmed at only +2.8 and
+1.2. That gap is the same search inflation already measured there. This leaves the recommendation
from [`E7_search_disagreement_bank.md`](E7_search_disagreement_bank.md) unchanged: distill
only on high-gap decisions. Event selection does not need targeting of its own. Any remaining gain
on the human-favoured events (Missile Envy, Grain Sales, Aldrich Ames) would need an evaluator that
plays the follow-up better than raw does, and these playouts cannot show it.

## Method

* **Population.** The 94,451 annotated positions of the search bank: 1,500 raw self-play games,
  one decision in eight sampled. A decision qualifies when all of these hold:
  * it is an action-round SELECT_PLAY_MODE (headlines excluded)
  * EVENT is legal
  * at least one other play is legal

  That leaves 17,738 positions, 94.6 a game.
* **Bank.** Up to 30 positions per (card, side), sampled at random (seed 0), each weighted by its
  group's size over the sample. That gives 5,977 positions in 203 groups.
* **Moves played out.**
  * On an own or neutral card: EVENT against raw's most probable alternative.
  * On an opponent's card: EVENT (event first) against raw's most probable Ops-first mode, plus
    Space when it is legal. Only Space stops the opponent's event; on either of the other two it
    fires.

  Raw's actual choice is always among the moves played out.
* **Playouts.** `ai/eval/paired_playouts.py` runs 256 pairs per move, with the raw network
  continuing for both sides. Within a pair, the mover's unseen cards are redealt identically for
  every move and the dice are shared. Scores are 0, ½ or 1 for the mover.
* **Regret.** The best move is chosen on the even pairs and its lead over raw's move is measured
  on the odd pairs. This is unbiased for "a 128-pair playout picker against raw". It is not the
  oracle regret: a noisy picker can lose to raw, as it does here.
* **Loss a game.** Regret × weight × (8 / 1,500), split by whether the picker's move was an event
  raw skipped (missed event), a non-event raw evented over (wrong event), or neither (other: Space
  or Ops on an opponent's card).
* **Clear errors.** Raw's move is beaten by more than 2 SE of the 256 paired differences, using
  all pairs. Under the null that gives about 2.3% per alternative.
* **Human and bot event rates.** From `tools/scripts/event_play_census.py`: per holding spent in
  an action round, evented / (evented + Ops + Space). The bot rate is from the same model's greedy
  self-play.

## Results

The full report, with every card, is [`E7_event_decision_bank/report.md`](E7_event_decision_bank/report.md).

| side / whose card | n | a game | raw events | best is event | event lead (pts) | regret (pts) | loss a game (pts) |
|:---|---:|---:|---:|---:|---:|---:|---:|
| US / neutral | 593 | 10.7 | 29% | 43% | -6.88 ± 1.08 | +0.07 ± 0.17 | +0.78 ± 1.81 |
| US / opponent | 1,183 | 18.5 | 44% | 58% | +0.50 ± 0.14 | -0.02 ± 0.11 | -0.36 ± 2.01 |
| US / own | 1,188 | 18.0 | 23% | 38% | -4.82 ± 0.47 | -0.17 ± 0.09 | -3.06 ± 1.63 |
| USSR / neutral | 590 | 10.0 | 22% | 41% | -5.75 ± 0.86 | -0.17 ± 0.13 | -1.68 ± 1.29 |
| USSR / opponent | 1,253 | 20.3 | 44% | 61% | +0.41 ± 0.15 | -0.08 ± 0.11 | -1.63 ± 2.14 |
| USSR / own | 1,170 | 17.2 | 23% | 42% | -2.91 ± 0.29 | -0.22 ± 0.11 | -3.71 ± 1.86 |

Two things stand out in this table:

* **"Best is event" is not raw's error rate.** It is often 15-20 points above raw's event rate,
  but most of that is noise in a close decision. The regret column, measured on separate pairs,
  sees nothing.
* **Raw beats the picker significantly on own cards in the early war:** -5.7 ± 1.7 a game. These
  are mostly events raw correctly declines, which a noisy picker sometimes takes.

Opponent's cards in the late war are the one cell positive beyond 2 SE: +2.2 ± 0.7 a game, split
across missed events, wrong events and other. No single card in it reaches 2 SE.

## What this does not say

* **That raw's events are optimal.** The resolution is about 0.5 points a decision. 256 pairs
  with the measured CRN correlation (median 0.09) cannot see smaller per-decision errors, and the
  playout continuation is raw itself. An event whose value needs a follow-up raw does not find
  reads as worse than it is.
* **Anything about headlines.** They were excluded.
* **Anything about which Ops mode or targets.** Only the event-or-not choice was measured. The
  earlier bank's largest category, wrong Ops mode, is outside this bank.

## Replicate

```bash
# on exp/search-bank (606b96f); bank built locally, playouts on CI (run 37719615504, 20 runners)
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py event-select \
  --annotated data/reports/search_bank/annotate/annotate.jsonl.gz --cap 30 --seed 0 \
  --out event_bank.jsonl.gz
gh release create event-bank-20261007 bank.jsonl.gz --prerelease   # event_bank.jsonl.gz renamed
gh workflow run search_bank.yml --ref exp/search-bank -f stage=playouts \
  -f release=event-bank-20261007 -f args="--pairs 256" -f runners=20
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py event-report \
  --bank event_bank.jsonl.gz --playouts playouts.jsonl.gz --games 1500 --sample 8 \
  --human-holdings data/reports/human_census/human_holdings.json \
  --bot-holdings data/reports/human_census/bot_holdings_E7line_swa_4720-4800M.json \
  --out report.md
```
