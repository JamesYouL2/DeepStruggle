# P34 D3 measured before it is built (2026-10-26): more keyed worlds make the sibling labels reliable, and a stronger follow-up does not change them

**Question.** [P34 revision 2, step 4](../plans/P34_search_centric_evaluator.md) plans `D3`: relabel the
contested entry decisions with a stronger follow-up, "at enough worlds for a ceiling of at least 0.85".
That world count was an extrapolation by Spearman–Brown from D2's split-half reliability of 0.41 at 16
keyed worlds (64 worlds → ~0.74, 128 → ~0.85). Two things had not been measured:

1. the reliability curve itself, on D3's population;
2. whether a label that reliable would teach anything, either with greedy follow-up or with search
   follow-up.

Both runs were done on CPU runners, in D2's recipe: A13+R48's SWA (11,920–12,000M) as generator and
playout network, keyed chance, the root's leaf at 4 boundaries, the generator's top 4 moves. Each
candidate was played in 128 shared worlds.

**Answer.**

* **The plan's arithmetic holds.** At 128 worlds the reliability is 0.72 on entry decisions and 0.76
  on the contested ones, and Spearman–Brown predicts the curve to within 0.01. 0.85 would need
  about 230–280 worlds.
* **That reliability comes from the easy decisions.** Ranked by candidate spread on worlds the
  measurement does not use, the top 10% of entry decisions reach 0.94 and carry 55% of the signal.
  The bottom half reaches 0.33.
* **A label's pick is worth what play-time search is worth, and no more.** The pick is made on
  some worlds and scored against the generator's argmax on disjoint ones. By the label's own
  evaluator it gains **+0.16 ± 0.09** (×100, per entry decision) at 64 worlds, although it departs
  from the network 60% of the time.
  * At 8–16 worlds, D2's scale, the pick is worse than the network.
  * A game has ~118 entry decisions (positions kept 1 in 8), ~77% of them with four candidates. So
    +0.16 sums to roughly +7 points a side per game. That is the order of the rollout root's whole
    gain at play time, which the [distillation ladder](E7_distill_ladder.md) could not transfer.
  * It is not a correction the critic is missing.
* **A stronger follow-up does not change the ranking.** With the keyed rollout root at the playout's
  entry decisions, the candidate means correlate with greedy follow-up's at **0.98** after
  disattenuation, on disjoint worlds.
  * Scored by search follow-up, the search pick beats the greedy pick by +0.43 ± 0.19.
  * Scored by greedy follow-up, the difference is −0.06 ± 0.19.
* **So D3 is dropped.** Its premise is that a stronger follow-up reveals values greedy play hides,
  and the follow-up does not change the prices. What the labels do resolve is the search's own
  gain, which is already collected at play time. Under step 5, this is the "neither moves" branch:
  direction 1's ceiling is the follow-up's quality, and the remaining strength is search at play
  time.

## Run 1: the reliability curve (greedy follow-up, CI 40373298241)

There are two populations:

* **uniform:** D2's plain sibling shuffle, 886 decisions labelled;
* **entry:** `value_labels select --what entry`, card choices and headlines, 2,321 decisions labelled.

Within the entry set, a decision is **contested** when its mean result over all candidates and worlds
lies in [0.25, 0.75].

At each world count W, the worlds below W are split into even and odd halves. The two halves' centred
candidate means are correlated, and the correlation is stepped up to W. The ± is a bootstrap over
decisions.

| worlds | uniform | entry | contested (1,198) |
|--:|--:|--:|--:|
| 8 | 0.13 ± 0.05 | 0.15 ± 0.03 | 0.16 ± 0.04 |
| 16 | 0.32 ± 0.05 | 0.26 ± 0.03 | 0.27 ± 0.04 |
| 32 | 0.52 ± 0.05 | 0.39 ± 0.04 | 0.42 ± 0.05 |
| 64 | 0.66 ± 0.04 | 0.57 ± 0.04 | 0.61 ± 0.05 |
| 128 | 0.79 ± 0.04 | 0.72 ± 0.03 | 0.76 ± 0.03 |
| *Spearman–Brown from 8 → 128* | *0.70* | *0.73* | *0.76* |

The uniform set's 0.32 at 16 worlds reads about 2 SE below D2's 0.41. That set is a different sample
of 1,000 games' positions.

**Where the reliability comes from.** The worlds are split into thirds. Decisions are ranked by
candidate spread on worlds 0 mod 3, and the reliability is measured between worlds 1 mod 3 and
2 mod 3, so the ranking does not select on the noise it is judged by. Each third has ~43 worlds; the
reliability is then stepped up to 128.

| entry decisions, by spread | n | reliability at 128 | share of the signal |
|:--|--:|--:|--:|
| bottom 50% | 1,167 | 0.33 | 17% |
| 50–90% | 923 | 0.43 | 28% |
| top 10% | 231 | 0.94 | 55% |

In the typical decision the four candidates differ by less than 128 worlds can resolve. The pooled
figure is carried by the decisions with an obviously bad candidate, and the network does not play
those.

**What a label is worth.** The pick is made on the first W even (odd) worlds and scored on all 64 odd
(even) worlds, against candidate 0, the generator's argmax. Both parities are averaged. Units are
win-probability points ×100 per decision.

| pick on | entry: departs | entry: gain | contested: departs | contested: gain |
|--:|--:|--:|--:|--:|
| 8 | 0.49 | −0.29 ± 0.08 | 0.59 | −0.44 ± 0.13 |
| 16 | 0.56 | −0.20 ± 0.08 | 0.65 | −0.37 ± 0.15 |
| 32 | 0.59 | −0.00 ± 0.09 | 0.66 | −0.07 ± 0.13 |
| 64 | 0.60 | +0.16 ± 0.09 | 0.64 | +0.22 ± 0.15 |

This is the label's own evaluator (greedy playouts of the same network) judging its own picks, so it
is the most favourable reading. The trend is about +0.2 per doubling of worlds. Summed over a game's
~90 four-candidate entry decisions, +0.16 is ~+7 points a side. That is the size of search at play
time, and it is what offline distillation failed to carry.

## Run 2: a stronger follow-up (CI 40383481065)

The setup:

* **Decisions:** the first 400 of the entry shuffle; 294 were labelled, all among run 1's.
* **Worlds:** 128 keyed.
* **Follow-up:** `--follow search --search-worlds 8`, the keyed rollout root (k=4, stop=earlier,
  leaf=own, `z2`) at every card choice and headline of the playout's turn.
  * It searched 1.33M decisions and departed from the network's move at 15% of them.
* **Cost:** about 11× greedy; 20 runners took 15–30 minutes.

On the same 294 decisions, scored on 64 disjoint worlds (×100):

| pick on 64 worlds by | scored by greedy | scored by search |
|:--|--:|--:|
| greedy follow-up | −0.17 ± 0.23 | −0.12 ± 0.21 |
| search follow-up | −0.23 ± 0.21 | +0.31 ± 0.24 |
| *search pick − greedy pick (paired)* | *−0.06 ± 0.19* | *+0.43 ± 0.19* |

Other readings:

* Split-half reliability at 128 worlds: 0.79 for greedy and 0.80 for search.
* Correlation of the centred candidate means across the two follow-ups, on disjoint worlds: 0.650,
  against half reliabilities of 0.655 and 0.673. Disattenuated, that is **0.98**.
* The candidates' results correlate at 0.74 across the two follow-ups within a world.
  * That correlation is the shared keyed worlds, which is why every comparison above picks and
    scores on disjoint worlds.

The two follow-ups price the candidates alike. The one cell in favour of search, its own evaluator
scoring its own pick, is 2.3 SE, and the other evaluator does not reproduce it. It matches the
2026-10-18 probe recorded in step 4: a stronger follow-up moves labels by ~0.10 a position with mean
~0.

## Reading

D3 was meant to show a candidate difference that the critic misses and that greedy play hides. Both
halves of that fail here:

* The labels' reliability is carried by the decisions the network already plays correctly.
* What remains, a pick worth +0.16 a decision, is play-time search's own gain, not a hidden value.
* Searching inside the playouts does not change the price of the candidates.

This agrees with the [distillation ladder](E7_distill_ladder.md) from the other direction. The
rollout root's corrections are local, and a value target built from playouts of the same network
reproduces that network's judgement.

Not tested here:

* A follow-up from a *different*, stronger network.
* The turn-10 exact tail (step 6, bindings work).
* Fits. No fit was run: the labels resolve the search's own gain, which the
  [ladder](E7_distill_ladder.md) showed does not transfer, and nothing beyond it.

## Reproduce

```bash
# run 1 (greedy, both populations)
gh workflow run sibling_reliability.yml --ref <branch>
# run 2 (search follow-up, entry decisions only)
gh workflow run sibling_reliability.yml --ref <branch> -f uniform=0 -f entry=400 \
  -f extra="--follow search --search-worlds 8"
gh run download <run-id> -n sibling-reliability-<run-id>
# the tables, from a labelled directory (positions_*.npz + siblings_*_of_*.npz)
PYTHONPATH=.:build/release python tools/scripts/sibling_reliability.py --dir <dir> --json rel.json
```

The spread-tier and two-follow-up tables came from ad hoc reads of the two runs' sibling parts (the
`sibling-reliability-part-*` artifacts), using the script's `decisions`, `half_means` and `pick_gain`.
The fork's runs were at commits 0fdcbd08 and 517da495.
