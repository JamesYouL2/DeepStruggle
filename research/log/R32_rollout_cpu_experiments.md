# R32 against the searchers, on CPU (2026-10-10): R32 absorbed Gumbel's gain, not the rollout root's

**Question.** R32 (P32 T1: the frozen heads soup's Gumbel k = 4 @32 improved policy as a CE term of
0.1 inside RL, 1 decision in 8, 6,400 → 6,800M) leaves Gumbel k = 8 @256 almost nothing to add:
51.2% on itself ([P32](../plans/P32_teacher_and_paired_credit.md)). Two questions follow:

* **Experiment 1:** does the rollout root, the strongest searcher on the heads soup
  ([E7_search_depth_and_value](E7_search_depth_and_value.md)), still add on R32?
* **Experiment 3:** did R32 take over the soup's search corrections, or only stop Gumbel finding
  new ones?

The rollout horizon and compute sweep (Experiment 2 of the plan) waits on Experiment 1.

**Answer.**

* **The rollout root adds as much on R32 as on the soup:**
  * +128 ± 12 Elo over R32 raw (67.6%), against +149 ± 12 on the soup (a difference of −20 ± 17);
  * +110 ± 12 over Gumbel@256 on R32, against +97 ± 11 on the soup.
* **Gumbel@256 adds nothing on R32:** +1 ± 11 Elo (50.2%), against +57 ± 11 on the soup, a
  difference of −56 ± 16, which is 3.6 SE.
* **R32's edge over the soup is small and does not survive Gumbel:**
  * raw: +13 ± 4.5 Elo (51.9%, 6,000 games);
  * under Gumbel@256 on both: −19 ± 8 (47.3%), so the searched soup is the stronger player;
  * under the rollout root on both: +12 ± 14 (51.7%, 600 games), not resolved.
* **R32 stays close to the soup and changes mainly where the soup's search does.** On 4,474 of
  the soup's own positions, weighted to the population of its decisions:
  * **Gumbel's departures** (where Gumbel@256 leaves the soup's network move, 18.8% of
    decisions): R32 keeps the soup's move at 70.5% and adopts Gumbel's correction at 20.4%.
  * **Gumbel's agreements:** R32 changes only 6.8% of moves.
  * **Control:** a sibling network trained from the same 6,400M root for the same 400M steps
    without the search term (R34) adopts more (26.7%) because it changes everything more (26.6%
    of agreements).

  When R32 leaves the soup's move it lands on Gumbel's pick 69% of the time; R34 does 51%. R32's
  changes are aimed at the search's departures, but it takes a fifth of them.
* **It took over almost none of the rollout root's corrections.** Where the rollout root departs
  and Gumbel keeps the soup's move, R32 plays the rollout root's move 5% of the time (R34 10%).
  R32 was never trained toward them, and the rollout root's gain on R32 is intact.
* **What R32 adopted is not the valuable part.** Adopted and rejected corrections are worth the
  same in paired playouts (+0.29 against +0.31 points per hundred decisions under the soup's
  continuations, +0.40 against +0.27 under R32's, all ± 0.2). Where Gumbel agrees and R32 still
  moves off the soup's choice, its moves average −0.2 to −0.4 (± 0.25), not significant.

**Reading.** Mihail's caveat on R32 was that "the policy absorbed its search" and "a sharper prior
leaves its old critic little to correct" read the same in games. Experiment 3 leans to the
second. R32 took a fifth of the soup's Gumbel@256 corrections (fewer than an unrelated sibling
drifts into), yet Gumbel@256 gains nothing on it. Meanwhile the rollout root, whose leaf is four
action-round boundaries deeper and whose judgement comes mostly from played-out moves rather than
the critic, still gains its full +110 to +130. Search's headroom on R32 is not gone; what is gone
is the part Gumbel's critic-bounded tree could reach. That is T2's case (a better leaf) and the
case for a deeper teacher in T1: the rollout root, or B4''s pricing to the turn's end.

## Setup

* **Checkpoints** (Hugging Face `mihaild/deepstruggle`; sha256 checked on every runner):
  * **soup**: `_models/E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt`, the heads soup and R32's
    teacher, sha256 `91a43a8b089c5889ddd8e46ea24e4c9c0235e8ada70b76b0213b92d1849bcf1b`;
  * **R32**: `E7-A8-R1-S44@6400M+R32_20261010_005623/swa_6720-6800M.pt`, sha256
    `1d4f65f040684610e5ae9bd47ff06bb35b51e25be4233ec16f66c3d61fa725b8`;
  * **R34**, Experiment 3's control only: `E7-A8-R1-S44@6400M+R34_20261010_031009/swa_6720-6800M.pt`,
    sha256 `be81eae0d822f3abb161e8df5501f318a2f0e11e354ae0663acf33f7fa0ba3e1`. It is the logit-gap
    penalty 1e-3 from the same 6,400M root and has no search term.
* **Searchers.**
  * Gumbel k = 8 @256 (`gumbel:<ckpt>:256:8`, first-play urgency 0.2, every node);
  * the rollout root `rollout:<ckpt>:4:16:4:z2`: the network's top 4, 16 shared worlds, 4
    action-round boundaries, the critic at the leaf, and the network's move kept unless another
    leads by 2 SE.
* **Games.** Greedy (temperature 0), both seats, `searcher_tournament.yml` on the fork's CI, 20
  runners, commit `605e72a` (`exp/r32-search`; the same code is `86aa88f` on `pr/r32-search`,
  rebased on upstream `2b6c759`). Scores count a draw as half. SE = √(p(1−p)/n), and Elo CIs are
  95%.
* **The soup's own searcher gains** come from an earlier run, `38042458244` (500 games a side,
  same searchers, same harness), not re-played here. Mihail's port of the rollout root (`264050d`)
  re-played the rollout root against Gumbel@256 on the soup at 61.2% ± 1.5 (+83), against 63.6%
  here.

## Experiment 1: R32 and the searchers

| pairing (first named scores) | score | Elo [95% CI] | games | run |
|:---|---:|---:|---:|:---|
| R32 raw vs soup raw | 51.9% ± 0.6 | +13 [+4, +22] | 6,000 | `38078062379` |
| R32 Gumbel@256 vs R32 raw | 50.2% ± 1.6 | +1 [−20, +23] | 1,000 | `38078057170` |
| R32 rollout root vs R32 raw | 67.6% ± 1.5 | +128 [+105, +151] | 1,000 | `38078057170` |
| R32 rollout root vs R32 Gumbel@256 | 65.3% ± 1.5 | +110 [+88, +133] | 1,000 | `38078057170` |
| R32 Gumbel@256 vs soup Gumbel@256 | 47.3% ± 1.1 | −19 [−34, −4] | 2,000 | `38078068517` |
| R32 rollout root vs soup rollout root | 51.7% ± 2.0 | +12 [−16, +40] | 600 | `38078074630` |
| *soup Gumbel@256 vs soup raw* | *58.1% ± 1.6* | *+57 [+36, +79]* | *1,000* | *`38042458244`* |
| *soup rollout root vs soup raw* | *70.2% ± 1.4* | *+149 [+125, +173]* | *1,000* | *`38042458244`* |
| *soup rollout root vs soup Gumbel@256* | *63.6% ± 1.5* | *+97 [+75, +120]* | *1,000* | *`38042458244`* |

**Search gains, R32 against the soup** (Elo ± 1 SE; the soup's from the separate run):

| searcher over its own network | soup | R32 | R32 − soup |
|:---|---:|---:|---:|
| Gumbel@256 | +57 ± 11 | +1 ± 11 | **−56 ± 16** |
| rollout root | +149 ± 12 | +128 ± 12 | −20 ± 17 |
| rollout root over Gumbel@256 | +97 ± 11 | +110 ± 12 | +13 ± 17 |

Placing everything on the soup's raw network: R32 raw +13. Two paths give R32 under Gumbel +14
(through R32 raw) and +38 (through the soup under Gumbel), consistent at 1.3 SE. R32's rollout
root comes out at +141 or +161 by the same two paths, against the soup's rollout root at +149.
The rollout root on either network is the strongest player measured, ~+150 over the soup.

## Experiment 3: did R32 adopt the soup's search corrections?

**The bank.** 4,474 positions from 1,600 of the heads soup's own greedy self-play games (shards
1–8 of the distillation-targets run `38007091505`), each with Gumbel@256's recorded pick. They
are stratified by whether the pick departs from the network's argmax, with inclusion 0.0831 for
departures and 0.00951 for agreements. Each row is weighted 1/p, so every rate is per decision
of the soup's play. Release `r32-transfer-bank-20261010` on the fork, sha256 `adea9f79…`.

**The run** (`search_reliability.yml`, run `38078080803`):

* every position searched once by soup Gumbel@256, soup rollout root, R32 raw, R32 Gumbel@256 and
  R32 rollout root;
* every distinct pick, and the soup's own move, paid out over 256 paired continuations, once with
  the soup playing the continuations (judge *soup*) and once with R32 (judge *R32*), to keep one
  network from grading itself;
* `tools/search_transfer.py report` classifies each position. The classes are defined by the
  picks alone and the values come from the playouts, so the selection does not bias the values.

**R34's picks** (the control) are R34's greedy moves on the same positions, computed locally.
R34 has no playouts, so it enters the adoption rates only.

Gains are in points of the mover's score per hundred decisions (a hundredth of a percentage
point per decision), ± 1 SE.

| per decision of the soup's play | judge soup | judge R32 |
|:---|---:|---:|
| R32 raw over the soup's move | +0.01 ± 0.02 | +0.01 ± 0.02 |
| soup Gumbel@256 | +0.06 ± 0.03 | +0.06 ± 0.03 |
| R32 Gumbel@256 | +0.08 ± 0.04 | +0.10 ± 0.04 |
| soup rollout root | +0.16 ± 0.04 | +0.15 ± 0.04 |
| R32 rollout root | +0.10 ± 0.04 | +0.15 ± 0.04 |
| R32 Gumbel@256 over R32 raw | +0.07 ± 0.04 | +0.09 ± 0.04 |
| R32 rollout root over R32 raw | +0.09 ± 0.04 | +0.14 ± 0.04 |

**Where the soup's Gumbel@256 departs from the soup's network** (18.8% of decisions; 1,728
positions):

| the network's move there | R32 | R34 (control) | correction's gain where R32 does this (judge soup / R32) |
|:---|---:|---:|---:|
| adopts Gumbel's correction | **20.4% ± 1.6** | 26.7% ± 1.8 | +0.29 ± 0.23 / +0.40 ± 0.23 |
| keeps the soup's move | 70.5% | 47.3% | +0.31 ± 0.18 / +0.27 ± 0.20 |
| another move | 9.1% | 26.0% | (its own: +0.64 ± 0.48 / +0.58 ± 0.48) |

* R32 − R34 adoption: −6.3 ± 2.1 points (paired bootstrap over positions).
* **Where Gumbel agrees** (2,746 positions), the network moves off the soup's choice at 6.8% (R32)
  against 26.6% (R34). R32's changed moves average −0.21 ± 0.26 (judge soup) and −0.37 ± 0.25
  (judge R32). They are confirmed worse at 2 SE at 0.2% of agreements, and worse by 10+ points
  at 0.09% / 0.01%.
* So R32 leaves the soup's move 4.3× as often at departures as at agreements (R34: 2.0×), and its
  departures land on Gumbel's pick 69% of the time (R34: 51%).

| decision type | Gumbel departures | R32 adopts | R34 adopts | correction's gain (judge soup / R32) |
|:---|---:|---:|---:|---:|
| influence (POINT_NODE) | 818 | 20% | 25% | +0.21 / +0.15 |
| card (SELECT_CARD) | 614 | 21% | 29% | +0.26 / +0.30 |
| play mode | 224 | 16% | 24% | +0.71 / +0.79 |
| Ops mode | 68 | 31% | 36% | +0.72 / +0.43 |

**The rollout root's corrections.** The soup's rollout root departs from its network at 13.0% of
decisions. R32 adopts 9.4% of those departures (R34 14.2%). Where the rollout root and Gumbel
disagree (24.6% of decisions; there the rollout root's pick is ahead by +0.39 ± 0.16 / ± 0.18):

| | share of disagreements | R32 plays the rollout root's / Gumbel's move | R34 |
|:---|---:|---:|---:|
| the rollout root keeps the soup's move, Gumbel departs | 57% | 72% / 21% | 49% / 27% |
| Gumbel keeps the soup's move, the rollout root departs | 33% | **5%** / 87% | 10% / 67% |
| both depart | 10% | 12% / 18% | 19% / 28% |

## Compute

| run | what | runner-minutes |
|:---|:---|---:|
| `38078062379` | 6,000 raw games | 61 |
| `38078057170` | 3,000 games: R32 raw, Gumbel@256, rollout root | 936 |
| `38078068517` | 2,000 games, Gumbel@256 on both | 716 |
| `38078074630` | 600 games, rollout root on both | 482 |
| `38078080803` | Experiment 3: search 60, playouts 695 (2 judges × 256 pairs) | 758 |
| R34's picks | local, 4,474 positions | < 1 |

Per decision, batched over 224 positions on a 4-core runner: Gumbel@256 16 ms and 251 network
rows; the rollout root 150 ms and ~945 rows (9× the time, 3.8× the rows). Per game in the
tournaments the rollout root costs about 2.2× Gumbel@256: 482 runner-minutes for 600 games
against 716 for 2,000, each with the same searcher on both sides. The runners are GitHub's
4-core standard machines; the figures include ~1.5 minutes of setup per job.

## Limitations

* **R34 is not a clean control.** It is a sibling from the same root over the same steps, but it
  carries its own intervention (a logit-gap penalty, which may loosen near-ties and so raise its
  drift) and rates below both (1613 against R32 1645 and the soup 1628). The clean control is the
  plain leg's own 6,720–6,800M SWA, which is not published. Read R32's adoption as below
  "what 400M more steps would drift into" and more targeted, not as a measured T1 effect.
* **The teacher measured is Gumbel@256; R32 trained on k = 4 @32.** R32 was never shown k8@256's
  corrections. The departures here are the stronger searcher's, which is the point (R32's search
  gap is measured at k8@256), but adoption of its own training teacher's picks could be higher.
* **Offline judges are coarse and can disagree with games.** Per-decision gains resolve to
  ±0.02–0.04, which cannot see R32 raw's +13 Elo. They put Gumbel@256 on R32 at +0.07 to +0.09
  over R32 raw (2 SE) where 1,000 games say +1 ± 11. Games decide; Experiment 3 is used for
  where the moves differ, not for how much.
* **The soup's searcher gains are a separate run** (same harness, same day, 500 a side), and a
  re-play of the rollout root after Mihail's port gave +83 rather than +97 over Gumbel. The R32 −
  soup differences are ±16–17 Elo, so only the Gumbel gap (3.6 SE) is a finding; the rollout
  root's −20 is not.
* **The positions are the soup's.** R32's own positions could show other departures. The bank is
  population-weighted for the soup's play, not R32's.
* **Rollout root against rollout root** is 600 games; +12 ± 14 does not order the two.

## Next step

**Experiment 2 (the rollout root's horizon and compute on R32) is now worth running.** The
rollout root still gains +128 on R32, and it costs ~2–9× Gumbel@256. Fixed beforehand, on
R32, against Gumbel@256 and the measured player, 500 games a side each:
* horizon 1, 2 and 4 boundaries at 16 worlds;
* 8 and 32 worlds at horizon 4;
* k = 2 and 6;
* runner-minutes per point of score.

That says what a cheap rollout teacher would keep. **The larger question is the teacher in
T1.** Gumbel's targets moved R32 where Gumbel@256 gains and left the rollout root's gain alone.
The next T1 teacher worth testing is the rollout root itself, or B4''s turn-end pricing (R37),
which reaches the same depth. Experiment 3's tool measures either's adoption on the same bank.

## Replicate

```bash
# Experiment 1 (fork CI; the soup is the base model, R32 is model2)
SOUP='_models/E7-A8-R1-S44@6400M+(S44,45,46)@6720..6800M.pt'
R32='E7-A8-R1-S44@6400M+R32_20261010_005623/swa_6720-6800M.pt'
gh workflow run searcher_tournament.yml --ref exp/r32-search -f model="$R32" -f games_per_side=500 \
  -f entrants="name:r32:{BASE} name:r32-g256:gumbel:{BASE}:256:8 name:r32-rollout:rollout:{BASE}:4:16:4:z2"
gh workflow run searcher_tournament.yml --ref exp/r32-search -f model="$SOUP" -f model2="$R32" \
  -f games_per_side=3000 -f entrants="name:soup:{BASE} name:r32:{MODEL2}"
gh workflow run searcher_tournament.yml --ref exp/r32-search -f model="$SOUP" -f model2="$R32" \
  -f games_per_side=1000 -f entrants="name:soup-g256:gumbel:{BASE}:256:8 name:r32-g256:gumbel:{MODEL2}:256:8"
gh workflow run searcher_tournament.yml --ref exp/r32-search -f model="$SOUP" -f model2="$R32" \
  -f games_per_side=300 \
  -f entrants="name:soup-rollout:rollout:{BASE}:4:16:4:z2 name:r32-rollout:rollout:{MODEL2}:4:16:4:z2"

# Experiment 3: the bank from a distillation-targets run, then searchers, two judges, the report
PYTHONPATH=. python tools/search_transfer.py select --targets targets-{1..8}.jsonl.gz --budget 256 \
  --departures 3000 --agreements 1500 --seed 0 --out bank.jsonl.gz
gh release create r32-transfer-bank-20261010 bank.jsonl.gz bank.jsonl.gz.meta.json --prerelease
gh workflow run search_reliability.yml --ref exp/r32-search -f release=r32-transfer-bank-20261010 \
  -f model="$SOUP" -f model2="$R32" -f judges="1 2" -f seeds=1 -f pairs=256 -f runners=20 \
  -f specs="soup_g256=256:8:0.2:all:1 soup_roll=rollout:4:16:4:z2 r32_raw=@2:raw r32_g256=@2:256:8:0.2:all:1 r32_roll=@2:rollout:4:16:4:z2" \
  -f transfer_roles="teacher_search=soup_g256 teacher_rollout=soup_roll student_raw=r32_raw student_search=r32_g256 student_rollout=r32_roll"
gh run download <run> -n search-reliability-<run>    # report/transfer.md, report-j1.md, report-j2.md
```

Artifacts: `searcher-tournament-<run>` for the four tournaments (each with `tournament.json`
and `provenance.txt` naming the sha256s), and `search-reliability-38078080803` (picks,
playouts, both judges' reports and `transfer.json`). The R34 adoption figures are from
`load_policy` greedy picks on the bank's positions, compared with the run's recorded picks.
