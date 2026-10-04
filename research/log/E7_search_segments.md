# Where honest search gains on the soup (2026-10-03): ops placement and the late game

**Question.** C5's search filter decides which decisions get the search budget during training.
E7-04-44 and the P15-X4b recipe search card and play-mode choices only. Which parts of the game
does honest search actually improve?

**Answer.** On the shallow soup, 64-simulation honest search wins **54.0%** when it searches
everything. Searching **Ops influence placement** alone gives the largest share (+1.9 points) and
**turns 8-10** alone give +2.7 of the +4.0. Card and play-mode choices together give +1.4; setup,
headlines, coups/realignments and choices inside events give almost nothing.

Code on the fork, branch
[`exp/card-search`](https://github.com/JamesYouL2/DeepStruggle/tree/exp/card-search) (`3219592`):
`BatchedMCTSConfig.node_filter` takes the segments below, and a partition test checks that every
decision falls in exactly one segment. Not proposed for main.

## Setup

* Model: the shallow soup E7-02/03/04/05@1200M (`shallow_E7-02+03+04+05_1200M.pt`), against
  itself with search on the chosen decisions only (`search:<model>:64:determinize:<filter>`).
  Everything outside the filter is played by the network.
* Honest search: 64 simulations, one sampled world per search. The network side plays
  at temperature 0.1, search at its argmax.
* 4,000 games per side (8,000 per row), CI, commits `620495d` / `3219592`, engine `6a5c9041`. One
  standard error ±0.56 points.

**Segments** (`decision_segment`): `setup`; `headline` (choosing a headline); `ar_card` (choosing a
card in an action round); `play_mode` (event, Ops, space, ...); `ops_influence` (placing Ops as
influence); `ops_coup_realign` (coup and realignment targets); `event` (any choice inside an event).
**Eras** by turn: early 1-3, mid 4-7, late 8-10.

## Result

Search's score against the soup:

| searched | score | gain | CI run |
|:---|---:|---:|:---|
| **everything** | **53.99%** | **+3.99** | `37164820062` |
| card + play mode + choices in events (`card_branch`) | 51.39% | +1.39 | `37164816033` |
| setup | 50.18% | +0.18 | `37172289639` |
| headline | 50.28% | +0.28 | `37172294379` |
| card in an action round | 50.91% | +0.91 | `37172299732` |
| play mode | 50.98% | +0.98 | `37172304681` |
| **Ops influence placement** | **51.86%** | **+1.86** | `37172308944` |
| coups and realignments | 50.16% | +0.16 | `37172313329` |
| choices inside events | 50.41% | +0.41 | `37172317565` |
| turns 1-3 | 50.61% | +0.61 | `37172322480` |
| turns 4-7 | 50.92% | +0.92 | `37172327815` |
| **turns 8-10** | **52.70%** | **+2.70** | `37172332440` |

Each gain is ±0.56. The segment gains sum to +4.8 and the eras to +4.2, close to the +4.0 of
searching everything, so the parts add up roughly independently.

## Reading

* **Influence placement is the largest single segment,** as E3's signal analysis found
  ([`../archive/E3_ladder/findings/which_decisions_to_search.md`](../archive/E3_ladder/findings/which_decisions_to_search.md)):
  a card-and-play-mode filter leaves most of the gain unsearched.
* **The late game carries two thirds of the gain.** Late turns have the largest swings (final
  scoring, the end of the deck) and the least time for errors to be repaired.
* Search on the soup gains +4.0 here at 64 simulations, consistent with
  [`P30_search_headroom_e7.md`](P30_search_headroom_e7.md)'s +2.5 at 256 simulations on 2,000 games.

## What this does not say

* One model (the soup), one budget (64). The segments may rank differently on a single model,
  where search gains more.
* These are play-time gains. Training the policy towards search on the segment that gains most did
  not work as a fine-tune ([`E7_search_finetune.md`](E7_search_finetune.md)).
