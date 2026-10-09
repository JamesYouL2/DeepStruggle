# The catastrophic-blunder census on the heads soup (2026-10-09): engine-proven blunders are rare, and the costly ones are the last influence placement of the game

**Question.** Where does the strongest E7 checkpoint throw a game away? The search-disagreement
bank (fork branch `exp/search-bank`, `research/log/E7_search_disagreement_bank.md` there) found
Gumbel's gain spread over ~7% of decisions rather than in a tail; this measures the tail directly,
on the current best network, with the engine's own proof where there is one.

**Answer.** Rarely, and almost all of it in one place.

* **Engine-proven mistakes: 0.60 per 1,000 decisions** (59 of 98,002), and 39 of those 59 cost
  nothing in the paired continuations: forced losses where every alternative also lost, and wins
  taken a turn later instead of at once.
* **The 7 that cost a lot (0.36–0.91 of a game each, 6 of 200 games, 3.66 points in all) are all
  the US's last influence placement of turn 10** -- final-scoring arithmetic. Gumbel k=8 @256
  plays the same losing or non-winning target in 4 of the 7. That is about 1.8 points of the US's
  score.
* **The estimated strategic class is at its noise floor:** 3.9% of candidates clear z ≥ 2 against
  2.7% of the random control, so its 22 per 1,000 is an upper bound made mostly of chance.
* **A classifier bug inflated the first report threefold** (1.88 per 1,000): `ai/eval/safety.py`
  listed How I Learned to Stop Worrying among the DEFCON degraders, so its event and its Ops
  both read "forced loss" at DEFCON 2. Its event is a choice of level 1–5, and the card is neutral,
  so Ops never fire it. Fixed (`333c802`, with tests); 124 of the 184 "exact" rows were this card.

## Method

Tools `tools/scripts/blunder_census.py` (screen, report) and `tools/scripts/bank_playouts.py
validate` (`tools/README.md` §7c); workflow `bank_playouts.yml`, `stage=validate`.

* **Model.** `E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt`, the SWA-made heads soup (sha256
  `91a43a8b089c…`; [`P31_branch_arms_B1_B2.md`](P31_branch_arms_B1_B2.md)).
* **Stage 1, screen (local, 426 s).** 200 greedy self-play games; every decision with two or more
  legal moves (98,002) checked by `classify_legal_actions` (a forced loss taken, a forced win not
  taken) and against a Gumbel k=4 @32 root. 13,875 candidates (forced loss 154, missed forced win
  30, Gumbel disagreement 13,724) plus a 5% random control of the rest (4,256).
* **Stage 2, validate (CI `37975392781`, 20 runners, 400 runner-minutes).** Each row searched by
  Gumbel k=8 @256 over three seeds; greedy's move, the cheap search's, the strong search's and a
  safe alternative paired out over 32 greedy continuations. Verdicts: the engine's exact labels
  first, then confirmed / refuted / unresolved regret at 0.05 points and z ≥ 2. The 100 most
  serious unresolved rows escalated to 1,024 simulations (local, 434 s).
* **Relabelling.** After the fix, every validation and escalation row's exact labels and verdict
  were recomputed from its saved position (`pos=` token) -- the playouts do not depend on the
  labels, so nothing was replayed. 124 rows changed verdict (all How I Learned: 112 unresolved,
  9 refuted, 3 confirmed regret) and one other, an exact forced loss to unresolved (the loss probe
  samples dice). The stage-1 trigger counts above still include the 124.

## Results

| figure | first report | relabelled |
|:---|---:|---:|
| confirmed catastrophic blunders per 1,000 decisions | 24.19 | 22.95 |
| — exact terminal mistakes (engine-proven) | 1.88 | **0.60** |
| — estimated strategic regret (upper bound) | 22.31 | 22.34 |
| confirmation rate of candidates | 4.1% | 3.2% |
| share of confirmed mistakes the cheap screen found | 24.1% | 20.0% |
| share rescued by 256-simulation search | 34.5% | 41.8% |

**The exact class, by what the paired continuations say it cost** (the best alternative's score
minus greedy's, out of 1):

| | nothing (≤ 0.01) | small (0.01–0.3) | costly (≥ 0.3) |
|:---|---:|---:|---:|
| forced loss taken (29) | 24 | 2 | 3 |
| forced win missed (30) | 15 | 11 | 4 |

* The **forced losses that cost nothing** are mostly a scoring card played at action round 6 or 7
  when the only other card would have left it held at the turn end -- also a loss, one choice
  later, which the forced probe does not follow, so it labels the alternative "normal". They are
  already-lost positions, not mistakes.
* The **missed wins that cost nothing** are wins on the spot not taken where the game was won
  anyway (Middle East Scoring instead of Africa Scoring, OPEC instead of Missile Envy).
* **The costly seven**, every one the US at turn 10, action round 7, placing its last influence
  before final scoring:

| game seed | greedy's target | Gumbel @256's | cost |
|---:|:---|:---|---:|
| 95 | Poland (loses) | Poland | 0.91 |
| 17 | South Korea (loses) | Greece | 0.50 |
| 108 | Zaire (draw; a win was on the board) | Zaire | 0.50 |
| 169 | Nigeria (draw) | Nigeria | 0.48 |
| 142 | Angola (draw) | South Korea (wins) | 0.47 |
| 169 | Libya (draw) | Libya | 0.44 |
| 174 | Ethiopia (loses) | Nicaragua | 0.36 |

## Reading

* **The heads soup does not throw games away.** Engine-proven catastrophes come about once in 1,700
  decisions (0.3 a game), two-thirds of them free, and the strategic class cannot be told from the
  control. This agrees with the search bank: what search fixes is diffuse evaluation error, not
  a tail of blunders, so a blunder filter is not where strength is.
* **Final scoring is the one recurring leak, and search does not fix it.** At the US's last
  placement nothing the opponent does comes before the board is scored, so the right target is
  arithmetic. The network gets it wrong in ~3% of games, and a micro-action search with the value
  network at its leaves rarely reaches the scored terminal, so it mostly agrees. A deterministic
  last-placement solver -- enumerate the remaining Ops' targets, score the board exactly -- would
  recover about 1.8 points of the US's score at no training cost, for any network and any
  searcher. Not built yet.
* **What this does not say.** One model, greedy self-play, 200 games: the costly seven are a count
  of 7, and the per-game figure carries that uncertainty. The escalation at 1,024 simulations moved
  nothing. The control's confirmed rows (91) are counted at their weight in the per-1,000 figure,
  which is why it barely moved when the exact class fell by two-thirds.

## Replicate

```bash
export PYTHONPATH=.:build/release
M='data/checkpoints/hf/E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt'
.venv/bin/python tools/scripts/blunder_census.py screen --model "$M" --games 200 --out-dir screen/
gh release create blunder-input-<date> screen/candidates.jsonl.gz screen/control.jsonl.gz --prerelease
gh workflow run bank_playouts.yml --ref exp/blunder-census -f stage=validate \
  -f release=blunder-input-<date> -f model='_models/E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt' -f pairs=32
.venv/bin/python tools/scripts/blunder_census.py report --screen screen/candidates.jsonl.gz \
  --control screen/control.jsonl.gz --summary screen/summary.json \
  --validation validation.jsonl.gz --escalation escalation.jsonl.gz --out report.md
```

Rows validated before `333c802` carry the old labels: recompute them from each row's `pos` with
`classify_legal_actions` and `bank_playouts.verdict_of` (min_z 2, min_gap 0.05) before the report.
