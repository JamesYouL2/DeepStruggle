# Where Gumbel search's gain comes from (2026-10-07): ~7% of decisions, moves the raw policy already ranks near the top, and an evaluation too noisy to find them reliably

**Question.** Gumbel root search is worth about +60 Elo over the raw network
([`E7_gumbel_headroom.md`](E7_gumbel_headroom.md)), but every attempt to train that gain into the
policy has lost to its control: visit-count and Gumbel-choice fine-tunes, and Gumbel's improved
policy as a target. Before trying again, find out *what search fixes*:

1. Which positions account for the improvement?
2. Is search fixing **candidate generation** (the right move never gets enough policy mass to be
   considered), **ranking/evaluation** (it is considered, but valued wrongly), or a small
   **catastrophic tail**?

**Answer.**

* **Candidate generation is not the bottleneck (Outcome A).** Where the raw move is wrong, the
  better move is in the raw policy's top 4 for 92-95% of positions and in its top 8 for 98.5-100%.
  Moves outside the top 8 carry about 1% of the measured regret.
* **The gain is concentrated, but not in a catastrophic tail.** About **7% of decisions** (those
  where search values the raw move 2+ points of win probability below the best) hold all the
  playout-measurable gain: **+45 ± 8 points a game** from the best move there, of which Gumbel
  k=8 @256 captures +29 ± 2. The other ~93% of search's departures from the raw move are worth
  nothing or slightly less than nothing. The worst 0.1% of decisions hold only 4.6% of the regret.
* **The limit is evaluation.** When the better move is already a candidate, k=8 @256 picks it
  46% of the time (k=4 @64 35%, k=4 @16 27%). Forcing a missing better move into the candidate set
  gives about the same rate, so exposure is not what is missing.
* **Search's own values overstate its regret about twofold to fivefold.** Summed over a game, they
  credit k=8 @256 with +38 points over the raw network; paired playouts measure +17 ± 16, and the
  tournament +9. Where search sees a gap under 2 points the playouts see none (or a small one the
  other way); above 2 points they confirm about half its size.

**What it implies.** A search target at *every* decision is mostly noise: about 93% of the
departures it would train toward are worth nothing, and that is consistent with every generic
search fine-tune losing to its control. The signal is in a selectable minority of decisions. The
next experiment is **search distillation restricted to decisions where the search-valued gap is
2 points or more** (playout-confirmed where affordable), checked first for moving the policy at
those positions and nowhere else, then in a tournament. Improving the policy's candidates
(representation, architecture) is not where the remaining error is.

## Method

Tool: `tools/search_bank.py` (stages `annotate`, `select`, `reference`, `oracle`, `playouts`,
`report`), CI workflow `.github/workflows/search_bank.yml`, branch `exp/search-bank` on the fork.
The Gumbel root gained two default-off diagnostic hooks (`GumbelRoot.last_stats`, the candidates
with their evaluations, values and drop phases; `choose(candidates=...)`, a forced candidate set);
play is unchanged (`tests/training/test_gumbel_root.py`, `tests/training/test_search_bank.py`).

* **Model.** `E7line_swa_4720-4800M.onnx`, the newest export, rebuilt as a torch checkpoint on
  each runner by `tools/onnx_to_checkpoint.py` (checked to reproduce the export's logits and
  values on 512 positions).
* **Positions (stage 1, CI 37676640252).** 1,500 games of the raw network against itself at
  temperature 0.1 (seeds 700,000 + g); one decision in 8 with two or more legal moves kept:
  **94,451 positions** (503.7 such decisions a game). Each was put to the raw network and to the
  three Gumbel configurations of `E7_gumbel_headroom.md` -- k=4 @16, k=4 @64, k=8 @256 -- with every
  searcher's candidates, evaluations and values recorded.
* **Bank (stage 2).** 8,000 positions: 6,000 where some searcher departs from the raw move and
  2,000 agreement controls, sampled at random within 72 strata (agreement x decision group x era x
  departure size). Each row carries a weight (its stratum's population over its sample), so the
  bank reweights to the decisions raw play actually meets. Published as fork release
  `search-bank-20261007`.
* **Reference (stage 3, CI 37687194319).** Each bank position:
  * solved by Gumbel **k=16 @1024**, three independent seeds (k=16 so the reference can choose
    outside every deployed searcher's candidates); `agreement` is how many of the three agree;
  * every move that matters -- raw's, each searcher's, each reference run's, and the raw top 4 --
    valued by an equal-budget **256-simulation search in each of 64 redealt worlds**, the same
    worlds for every move, so differences are paired world by world; and by the bare critic
    (one simulation) in those worlds.
  * **Regret** of a move = value of the best move minus its value, with the best chosen on the
    even worlds and the regret measured on the odd ones, so picking the largest of several noisy
    values does not inflate it. A verdict is **high confidence** when the best leads the runner-up
    by 2+ standard errors over all worlds *and* the k=16 @1024 reference agrees (2 of 3 seeds or
    more): 2,896 of the 8,000.
* **Oracle (stage 4, CI 37697769282).** The 136 positions (high or medium confidence) whose best
  move lies outside raw's top 4: k=4 @64 (all 136) and k=8 @256 (the 15 outside the top 8) rerun
  with the best move forced in place of their least probable candidate, three seeds each.
* **Playouts (stage 5, CI 37697772335).** 2,300 positions -- all 1,229 with a search-valued raw
  regret of 2+ points, 771 of the other 4,783 departures, 300 of the 1,988 agreement positions --
  each with raw's move, the best move and k=8 @256's move played out 256 times in pairs by the
  raw network on both sides (`ai/eval/paired_playouts.py`: the mover's unseen cards redealt per
  pair, the dice shared). Each row is reweighted by its sampling probability, so per-game figures
  stand for all raw-play decisions.

The full output is [`E7_search_disagreement_bank/report.md`](E7_search_disagreement_bank/report.md).

## Results

### 1. Search's own regret is inflated; the playouts agree with the tournament

| Gumbel k=8 @256 over the raw network, per game | points of win probability |
|:---|---:|
| search-valued (equal-budget values above) | +38 |
| paired playouts | **+17 ± 16** |
| tournament ([`E7_gumbel_headroom.md`](E7_gumbel_headroom.md)) | about +9 |

The search-valued figure is twice the playouts' and four times the tournament's; for the
reference's best move the overstatement is fivefold (+216 search-valued against +44 ± 55 in
playouts). Even agreement-control positions, where every searcher plays the raw move, show a
search-valued regret of 0.27 points a decision. The equal-budget values carry a systematic bias
-- one candidate is the seat disagreement at every hand-over that
[`expert_review_E7.md`](expert_review_E7.md) measured -- so the sizes below are read through the
playouts. Bucketed by the search-valued regret where the reference best differs from raw
(unweighted):

| search says raw loses | positions | search says (points) | playouts say (points) | playouts favour the best move |
|:---|---:|---:|---:|---:|
| < 1 point | 294 | +0.01 | −0.33 ± 0.16 | 38% |
| 1-2 points | 242 | +1.14 | +0.24 ± 0.20 | 50% |
| 2-5 points | 901 | +3.12 | **+1.46 ± 0.12** | 59% |
| 5+ points | 328 | +8.64 | **+4.87 ± 0.19** | 68% |

Below 2 points search's preferences carry no outcome signal; above it they are real, at about half
the size it claims. The playouts value a move by the raw network's continuation, which undervalues
moves whose payoff is in a follow-up the network does not find, so they are a lower bound.

### 2. The gain sits in ~7% of decisions

Population-weighted (the shares sum to 92% rather than 100% because the playout sample's
reweighting is itself an estimate):

| decisions | share | best move over raw, playouts, per game | k=8 @256's move over raw, playouts, per game |
|:---|---:|---:|---:|
| search-valued regret < 2 points | 85.5% | −4.4 ± 49.8 | −12.8 ± 14.3 |
| search-valued regret ≥ 2 points | 6.8% | **+44.6 ± 8.3** | **+28.9 ± 2.0** |

About 34 decisions a game carry the gain. The regret is moderately concentrated -- the worst 1% of
decisions hold 19% of it, the worst 10% hold 68% -- with no catastrophic tail (the worst 0.1% hold
4.6%). By decision, card choice, the headline and the play mode cost about twice as much per
decision as an influence point (raw regret 0.011, 0.011, 0.011 against 0.005), and the worst
individual decisions cluster in turns 9-10: realignment against influence in the last action
round, and card order late in the game (report, *Highest-regret raw decisions*).

### 3. Candidate recall: the better move is already near the top

High-confidence positions where the raw move is not the best (1,321):

| best move in raw's | top 1 | top 2 | top 4 | top 8 | top 16 |
|:---|---:|---:|---:|---:|---:|
| share | 0% | 62.1% | 91.8% | 98.5% | 100% |

Moves outside raw's top 4 carry 6.9% of the raw regret, outside the top 8 1.1%. Where the playouts
confirm a 2+ point regret (205 positions): top 2 49.8%, **top 4 94.6%, top 8 100%**. k=4's
candidate set misses the best move in 8% of errors, k=8's in 1.5%.

### 4. Ranking: search finds the better move only about half the time it is a candidate

| searcher | best move outside its candidates (share of errors) | picks the best move when it is a candidate |
|:---|---:|---:|
| k=4 @16 | 8.2% | 26.8% |
| k=4 @64 | 8.2% | 34.6% |
| k=8 @256 | 1.5% | 46.1% |

**Oracle.** With the best move forced into the candidate set, k=4 @64 picks it 27.5% of the time
over 136 positions and k=8 @256 42.2% over 15 -- about the rate at which each picks it when it is
a candidate anyway. Search that is shown the right move does not recognise it more often: the
limit is evaluation, and more budget helps (27% → 35% → 46%). The bare critic averaged over the
64 worlds ranks the best move first 45% of the time, as often as k=8 @256 -- but those are the
worlds the reference itself was valued in, which flatters the critic.

## Classification against the brief

| outcome | verdict |
|:---|:---|
| A -- candidate generation already excellent | **yes**: 92-95% of errors have the better move in the raw top 4, 98.5-100% in the top 8 |
| B -- important actions frequently missing | no: about 1% of regret lies outside the top 8 |
| C -- errors highly concentrated (catastrophic tail) | partly: the gain is in ~7% of decisions, but medium errors (2-5 points), not a catastrophic tail |
| D -- search cannot recognise the move even when shown | partly: forcing it in changes nothing, and search picks it about half the time -- an evaluation limit, eased by budget |

**Recommendation: search/value work, targeted.** Not candidate or representation work, and not
generic distillation. Distil search only at decisions where its value gap is 2 points or more
(about 7% of decisions), ideally where playouts confirm it; this bank gives both the selection rule
and a held-out validation set. Secondarily, evaluation precision bounds what search itself can
find: the search-valued biases measured here are worth understanding before more budget is spent.

## What this does not say

* The reference best is defined by values the playouts show to be biased; recall and pick rates
  are measured against a noisy label. The high-confidence and playout-confirmed restrictions give
  the same answers.
* The playouts are the raw network's continuation on both sides: a lower bound on any move whose
  payoff needs a follow-up the network does not play.
* The per-game playout estimate for the ~93% of small departures has a ±50 error bar; that they are
  worth nothing rests mainly on the bucket table, which is unweighted.
* One network (the E7 line's newest SWA), one temperature (0.1).

## Replicate

```bash
# stages 1, 3, 4, 5 on CI (fork, branch exp/search-bank); 2 and 6 locally
gh workflow run search_bank.yml --ref exp/search-bank -f stage=annotate -f args="--games 1500"
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py select \
  --annotated annotate.jsonl.gz --size 8000 --control 2000 --out bank.jsonl.gz
gh release create search-bank-<date> bank.jsonl.gz --prerelease
gh workflow run search_bank.yml --ref exp/search-bank -f stage=reference -f release=search-bank-<date>
# oracle / playouts inputs are bank rows with `target` / `pmoves` (+ `pl_incl`), built from the
# reference as described above, each published as its own release's bank.jsonl.gz
gh workflow run search_bank.yml --ref exp/search-bank -f stage=oracle -f release=<oracle-input> -f runners=4
gh workflow run search_bank.yml --ref exp/search-bank -f stage=playouts -f release=<playout-input> \
  -f args="--pairs 256" -f runners=16
PYTHONPATH=.:build/release .venv/bin/python tools/search_bank.py report --bank bank.jsonl.gz \
  --reference reference.jsonl.gz --oracle oracle.jsonl.gz --playouts playouts.jsonl.gz \
  --playouts-input playout_input.jsonl.gz --decisions-per-game 503.7 --out report.md
```
