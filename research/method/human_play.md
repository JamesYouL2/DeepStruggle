# Human play — the baseline the arms are measured against

What the two human corpora contain, which answers which question, the reference numbers, and the
agreement measure. **Rewritten in place.** How the corpus is converted to engine decisions:
[`../log/P7_replayer_conversion.md`](../log/P7_replayer_conversion.md).

## Two corpora: ts-replayer for moves, ITS for length and endings

**ts-replayer** (the training corpus) has 274 distinct logs: **119** reach a terminal state, 146 are
fragments whose recording stops, 9 are empty. Only the 119 give a game length, and `game_ended` must
come from the converter's terminal test (the log's last `defcon` predates the ending: 2 DEFCON-1
endings found instead of 21). Averaging fragments' last turns (6.77) or all 265 usable logs (7.95)
measures when users stopped recording.

**ITS** (`/workspace/data/itsc-games`, a scrape of the ITS Junta results table) has **47,928**
digital games, 44,136 with a rules ending, results only (`endTurn`, `endMode`, no moves). Its
`endTurn` records Final Scoring at turn 11, the engine's own convention.

| | ts-replayer (119) | ITS (44,136) |
|:---|---:|---:|
| went the distance | 67.2% | **29.7%** |
| mean end turn (engine scale) | 9.99 | **8.33** |
| mean ply (of 154) | 142.2 | **~119** (bounded [112, 123]) |
| final scoring | 56.3% | 29.0% |
| 20 VP | 23.5% | 43.1% |
| Wargames | 18.5% | 14.9% |
| DEFCON 1 | 1.7% | **11.7%** |
| held scoring | (in 20 VP) | 1.4% |

**ts-replayer is biased long:** a game that blows up early leaves a fragment, so the finished subset
is not independent of the ending (1.7% against 11.7% DEFCON-1 is not noise). **Use ITS for length
and ending mix; ts-replayer where moves are needed.** The ITS ply is an estimate (a turn pins only a
14–16-ply interval); report it with its interval.

## Side balance in ITS

**49.90% USSR** over 43,685 decided games (1.02% ties). The sides win differently:

| | US wins (21,888) | USSR wins (21,797) |
|:---|---:|---:|
| mean end turn / ply | 8.62 / ~124 | 8.00 / ~115 |
| went the distance | 33.4% | 25.2% |
| 20 VP | 36.2% | **50.8%** |
| final scoring | **32.1%** | 25.0% |
| Wargames | 14.5% | 14.9% |
| DEFCON 1 | **15.6%** | 8.1% |

The USSR wins half its games on the VP track and faster; the US wins 65.9% of DEFCON-1 endings (the
USSR walks into the war more often). The trainer splits length and ending mix by winning side
(`game_won_us/`, `game_won_ussr/`), since a run can match the pooled numbers with both halves wrong.

**The early gap (E1/E2 arms, T 0.1):** arms ran ~99–107 plies against ~119, with DEFCON-1 endings at
43–48% (four times human) and Wargames almost never played (0–0.2% against 14.9%); HeuristicBot,
with its instant-loss safety layer, reached 114.6 plies and no DEFCON-1. Read seed pairs, never one
arm: H and H2 differ by 8.4 plies at equal strength
([`../log/variance_and_noise.md`](../log/variance_and_noise.md)).

## Agreement with human play

Reported per BC epoch, per snapshot and in tournament reports (`ai/eval/agreement.evaluate_dataset`,
for the human corpus directory or a self-play file). It is not accuracy: a multi-point play is one
decision made several times, so consecutive point decisions of one play are scored as a **multiset
of countries**, teacher-forced along the human trajectory, one comparison per point (two points in
one country are two entries). **Coups and realignments are scored strictly** (dice make order
matter; Che's second coup depends on the first) — a blacklist, so a new influence-spreading card
needs no registration.

The ordering artefact was worth **about half a point** (34% of 60,670 decisions sit in multi-point
plays; BC on the human corpus 46.04% ordered → 46.32% unordered; E3 arms +0.55 to +0.75), because
teacher forcing already shows the model the human's earlier points. BC prints three numbers: Strict
Acc (in-batch, moving weights), Agreement (post-epoch, unordered) and ordered (the same pass,
strict), on a 20,000-decision cap stable to ~0.1 point.
