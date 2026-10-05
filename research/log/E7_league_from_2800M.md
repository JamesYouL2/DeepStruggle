# The P24 league from the saturated E7 line (2026-10-04/05): 2,800 → 3,600M, with its control

The owner asked to start league training once the E7 line clearly saturated; by the rule set in
advance it did at 2,800M ([`E7_line_to_2000M.md`](E7_line_to_2000M.md)).

**Runs** ([`../runs.md`](../runs.md)). `tools/scripts/league.py`:

* **Main, E7-21-44:** resumed from E7-20-44's 2,800M end state with the E7 shallow flags, plus the
  league -- `--league-frac 0.5`, up to four published exploiter snapshots, `--resume-every-steps
  10000000` -- to 3,200M. No setup credit (E6-04-44's arm had it; left out so the league is the one
  factor). `launch_flags --diff` against E7-20-44 shows only the league flags, the resume interval
  and the step target.
* **Main exploiter, E7-22-44-<generation>:** E5-22's recipe -- each 50M-step generation reset to the
  main's newest resume state, trained against a frozen greedy snapshot of it with no KL, no entropy
  bonus and learner-only normalisation; judged greedily, 200 games a side, published at ≥ 55%,
  restarted at ≥ 65%.

League directory `data/league/E7-21-44/` (events in `events.jsonl`).

## The exploiter could barely exploit the main

Each generation's judgements against its frozen target (5 per generation, one per 10M):

| gen | from main @ | judged range | best (US / USSR) |
|---:|---:|---:|---:|
| 1 | 2,810M | 51.4–53.4% | 53.4 (56.5 / 50.2) |
| 2 | 2,860M | 46.5–52.9% | 52.9 (58.5 / 47.2) |
| 3 | 2,910M | 44.1–54.8% | 54.8 (55.5 / 54.0) |
| 4 | 2,960M | 42.4–52.5% | 52.5 (56.8 / 48.2) |
| 5 | 3,030M | 47.1–51.5% | 51.5 (56.0 / 47.0) |
| **6** | 3,080M | 45.9–**55.0%** | **55.0 (61.8 / 48.2) -- published, snapshot 3,110M** |
| 7 | 3,130M | 47.2–51.1% | 51.1 (52.2 / 50.0) |
| 8 | 3,170M | 51.0% (1 judgement) | stopped when the main finished |

On E6, the same exploiter recipe reached ~69% against its first target. Against the 2,800M+ E7 main,
350M exploiter steps produced one snapshot at the 55% bar. Exploiters do somewhat better as US
(52–62%) than as USSR (47–54%).

So **the league pool was empty from 2,800M to 3,110M** -- the main trained as plain self-play -- and
held one member for the last ~90M steps.

## The main's 80M-window SWAs against fixed references

1,000 games per side, temperature 0 (`data/reports/swa_line/`); per side ± 1.5, both ± 1.1.

| SWA of | E7-02-44@1000M (US / USSR) | soup (US / USSR) | league members in the pool |
|:---|---:|---:|:---|
| 2720–2800M (E7-20-44, end of the plain line) | 65.4 (66.1 / 64.7) | 53.0 (52.5 / 53.6) | -- |
| 2800–2880M | 65.1 (67.3 / 62.8) | 51.2 (52.0 / 50.5) | none |
| 2880–2960M | 64.6 (66.4 / 62.8) | 50.4 (49.3 / 51.6) | none |
| 2960–3040M | 67.8 (67.5 / 68.0) | 53.0 (50.8 / 55.2) | none |
| 3040–3120M | 67.5 (69.1 / 66.0) | 50.6 (49.4 / 51.8) | none until 3,110M |
| **3120–3200M** | **71.4 (74.2 / 68.6)** | **56.5 (58.0 / 55.1)** | one (gen 6) |

## Reading

* **Hard to exploit.** At 2,800M+ the main agent resists the E5-22 exploiter recipe at 50M-step
  generations: one publish in eight generations.
* **The line kept creeping while the pool was empty.** 2,960–3,120M sits above the 62–65% band
  against E7-02-44@1000M that held since ~1,900M, so the 2,800M "saturated" call was marginal.
* **The last window is the strongest of the whole line by a clear margin** -- 71.4% against the
  1,000M model (about +4 over the previous two windows) and 56.5% against the soup (about +4.7), up
  in both seats. It is also the only window trained with a league member in the pool, for ~90M
  steps; too short and too confounded with the ongoing creep to attribute to the league yet.
* Continued (owner: "continue it"), unchanged, as E7-21-44 / E7-22-44 from generation 9.

## Second leg: 3,200 → 3,600M (continued unchanged, 2026-10-04/05)

E7-21-44 resumed from its 3,200M end state with identical flags (`launch_flags --diff`: only
`--train-steps`), the league directory kept, so the published generation-6 exploiter was in the
pool from the start ("league group now 1 of 13"). Exploiter generations continued from 9
(`league.py --first-generation 9`).

**Exploiter: nothing published.** Eight generations, every judgement below the bar:

| gen | from main @ | judgements against its frozen target |
|---:|---:|:---|
| 9 | 3,210M | 46.1, 49.1, 46.1, 45.5, 46.1 |
| 10 | 3,250M | 51.4, 49.0, 47.5, 52.4, 42.9 |
| 11 | 3,300M | 46.8, 50.4, 44.6, 46.5, 42.9 |
| 12 | 3,370M | 47.1, 44.1, 47.6, 46.8, 46.0 |
| 13 | 3,420M | 47.6, 46.4, 47.4, 49.9, 48.5 |
| 14 | 3,460M | 49.7, 41.4, 48.5, 47.0, 45.6 |
| 15 | 3,530M | 51.1, 47.1, 43.1, 46.9, 47.2 |
| 16 | 3,580M | 46.6 (stopped when the main finished) |

Mostly 43–50%: the exploiter recipe now loses to its own frozen target. The league held one member
for the whole leg.

**Main SWAs** (1,000 games per side; ± 1.5 per side, ± 1.1 both):

| SWA of | E7-02-44@1000M (US / USSR) | soup (US / USSR) |
|:---|---:|---:|
| 3200–3280M | 69.0 (73.0 / 65.0) | 53.6 (53.7 / 53.4) |
| 3280–3360M | 67.8 (71.6 / 64.1) | 53.5 (54.7 / 52.3) |
| 3360–3440M | 68.2 (72.8 / 63.6) | 54.2 (56.8 / 51.6) |
| 3440–3520M | 70.3 (72.4 / 68.3) | 53.6 (55.3 / 51.9) |
| 3520–3600M | 67.2 (70.0 / 64.4) | 54.8 (56.9 / 52.7) |

* **A new, higher plateau.** Since ~3,000M the main holds about 67–70% against E7-02-44@1000M and
  53.5–55% against the soup, against 62–65% and ~50.5% on the plateau of 1,900–2,800M. The
  3,120–3,200M window (71.4 / 56.5) was a high draw at the start of this level.
* **The gain is mostly in the US seat** (70–73% against the 1,000M model, against 63–68% as USSR).
* **Attribution is still open.** The rise began around 2,960M, before the first exploiter was
  published (3,110M), and the league never held more than that one member. Plain continuation or
  the single league member cannot be told apart from this run; a plain continuation of E7-20-44
  from 2,800M over the same steps would be the control -- run below.

## The control: E7-20-44 continued without a league, 2,800 → 3,600M (2026-10-05)

Owner: "Lets try to get a control continuation." E7-20-44 resumed from the same 2,800M end state
with its own flags, only `--train-steps` changed, same name (directory `E7-20-44_20261005_085738`).
Its SWAs are in `data/checkpoints/_swa_line_ctl/`, reports `data/reports/swa_line_ctl/` and
`data/reports/league_vs_control/`.

**Identical until the first publish.** With an empty league pool, the league main and the plain
continuation run the same computation. The control's SWAs for 2,800–2,880M, 2,880–2,960M and
2,960–3,040M match the league's to the last digit (65.1 / 64.6 / 67.8 against E7-02-44@1000M), as do
the training metrics at 3,000M. **The rise to the higher plateau at ~2,960M is therefore plain
self-play.** The runs separate after the generation-6 publish at 3,110M.

**Matched windows against the fixed references** (1,000 games per side; ± 1.1 per row, ± 1.5 for a
difference of two rows):

| SWA window | vs E7-02-44@1000M: league / control (diff) | vs soup: league / control (diff) |
|:---|:---|:---|
| 3040–3120M (league steps only after 3,110M) | 67.5 / 69.1 (−1.6) | 50.6 / 52.3 (−1.7) |
| 3120–3200M | 71.4 / 67.9 (+3.5) | 56.5 / 54.6 (+1.9) |
| 3200–3280M | 69.0 / 67.8 (+1.2) | 53.6 / 51.8 (+1.8) |
| 3280–3360M | 67.8 / 67.7 (+0.1) | 53.5 / 54.2 (−0.7) |
| 3360–3440M | 68.2 / 69.8 (−1.6) | 54.2 / 57.4 (−3.2) |
| 3440–3520M | 70.3 / 69.2 (+1.1) | 53.6 / 56.3 (−2.7) |
| 3520–3600M | 67.2 / 69.2 (−2.0) | 54.8 / 53.2 (+1.6) |
| **mean, 3,120–3,600M** | **69.0 / 68.6 (+0.4)** | **54.4 / 54.6 (−0.2)** |

Control rows per side: 3,120–3,200M 67.9 (68.6 / 67.2) / 54.6 (54.6 / 54.7); 3,200–3,280M 67.8
(70.9 / 64.7) / 51.8 (50.4 / 53.2); 3,280–3,360M 67.7 (70.5 / 64.9) / 54.2 (54.7 / 53.8);
3,360–3,440M 69.8 (72.4 / 67.3) / 57.4 (58.6 / 56.2); 3,440–3,520M 69.2 (71.1 / 67.4) / 56.3
(57.7 / 54.9); 3,520–3,600M 69.2 (72.4 / 65.9) / 53.2 (53.0 / 53.4). The control shows the same US
lean against the 1,000M model, so that asymmetry belongs to the line, not to the league.

**Head to head, league SWA against control SWA at the same window** (1,000 games per side, greedy):

| SWA window | league vs control (league as US / as USSR) |
|:---|:---|
| 3120–3200M | 51.7 (53.2 / 50.2) |
| 3200–3280M | 51.6 (55.3 / 48.0) |
| 3280–3360M | 51.0 (54.7 / 47.3) |
| 3360–3440M | 48.1 (50.8 / 45.4) |
| 3440–3520M | 50.6 (54.1 / 47.0) |
| 3520–3600M | 49.9 (52.0 / 47.8) |
| **mean** | **50.5 (53.4 / 47.6)** |

+0.5 is about 1 SE over 12,000 games, and the edge of the first three windows fades to even by
3,600M. The seat split is the game's US advantage between these lines: the league wins 53.4% as US,
the control 52.4% as US, so net of seat it is the same half point.

**The owner's panel, 3,520–3,600M SWAs, per seat** (1,000 games per side; ± 0.6–1.2 per seat):

| opponent | league (US / USSR) | control (US / USSR) | league − control (US / USSR) |
|:---|:---|:---|:---|
| E6-03-44@80M | 95.9 (95.8 / 96.0) | 96.0 (95.6 / 96.5) | +0.2 / −0.5 |
| E6-03-44@240M | 92.8 (95.0 / 90.6) | 91.9 (92.7 / 91.1) | +2.3 / −0.5 |
| E6-03-44@550M | 84.2 (87.8 / 80.7) | 84.5 (89.6 / 79.5) | −1.8 / +1.2 |

No seat is better against all three panel members, and the per-seat differences change sign
between panel members. By the owner's rule (better in one seat, not worse in the other) the league
is not accepted over the control.

### Reading

* **The league added nothing measurable over plain self-play from 2,800M to 3,600M.** Against the
  fixed references it is +0.4 / −0.2 on the mean of six matched windows, head to head 50.5%, and on
  the panel it is level. The 71.4 / 56.5 window at 3,120–3,200M was the league's best draw, and the
  head to head there (51.7%) is also its best, but neither held.
* **Why it had no chance to matter:** the exploiter recipe published once in 16 generations, so the
  league pool was empty for the first 310M steps and held that one snapshot for the remaining 490M. A main agent that the exploiter cannot beat gets almost no league signal.
* **The plateau moved because of plain self-play.** Between ~2,960M and 3,600M the plain line holds
  67–70% against E7-02-44@1000M and 52–57% against the soup, up from 62–65% / ~50.5% at
  1,900–2,800M. So the 2,800M "saturated" call by the pre-set rule was premature; the line was still
  improving slowly, in steps rather than smoothly.
