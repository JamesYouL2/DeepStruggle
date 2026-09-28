# E5-15 — the human opening on a trained model: it helps the US seat, and the USSR only catches up

**Question (owner, 2026-09-28).** Take the current best model. (1) Force the standard human opening
on it (USSR East Germany +1, Poland +4, Yugoslavia +1; US West Germany +4, Italy +3, Iran +2) and
measure its strength. (2) Train it for 80M with every game starting from that opening, and check
whether it comes out stronger.

**Acceptance (owner's rule, set before any result, [`../runs.md`](../runs.md)).** Per seat. Let
E5-11's own greedy self-play give USSR p and US q ≈ 1 − p over the comparison block. E5-15+human
must beat E5-11 on its own opening as USSR above p **and** as US above q, on both seeds.

## Step 1: the opening scripted onto the untrained best

`tools/tournament.py` with an `opening:human:` spec (400 games per seat per pair, greedy;
`data/reports/e5_11_4{3,4}_human_opening.md`):

| | human opening vs own opening, head to head | Elo in a field with E4-61@560M and HeuristicBot |
|:---|---:|---:|
| E5-11-43@560M | 46.1% ± 1.8 | −12 |
| E5-11-44@640M | 50.4% ± 1.8 | +1 |

By seat on seed 43, against the model's own self-play, the human opening helps the US (+2.5) and
hurts the USSR (−7.5). The model plays East Germany 1, Poland 4, Yugoslavia 1 worse than its own
Poland 3, Hungary 3.

## Step 2: 80M of training from the opening

E5-15-SEED is E5-11-SEED resumed from the resume state beside its best snapshot (43: 550M → 630M;
44: 650M → 730M) with `--forced-opening human`. `launch_flags.py --diff` against E5-11 prints only
that flag and `--train-steps`. The resumed opponent pool is E5-11's own, less one member whose
snapshot no longer exists. The control is E5-11's own continuation at the same steps.

Round robins in `data/reports/e5_15_rr_{43,44}.md`, 100 games per seat per pair, greedy. The bar is
E5-11's twin self-play at each late step: identical greedy models play the same game from each
seat, so 200 distinct games per step. Reports are in `data/reports/e5_11_4{3,4}_{step}M_selfplay.md`.

| late block (last 40M × the same steps, 9 pairings × 100 per seat) | seed 43 (590/610/630M) | seed 44 (690/710/730M) |
|:---|---:|---:|
| E5-11 self-play bar, USSR | 53.5% (52.5 / 59.5 / 48.5) | 56.3% (60.5 / 54.0 / 54.5) |
| E5-15+human as USSR vs E5-11 | 52.8% ± 1.7 → **−0.7** | 56.7% ± 1.7 → **+0.4** |
| E5-11 self-play bar, US | 45.5% (46.0 / 39.5 / 51.0) | 42.8% (38.5 / 45.5 / 44.5) |
| E5-15+human as US vs E5-11 | 52.3% ± 1.7 → **+6.8** | 45.1% ± 1.7 → **+2.3** |
| overall | 52.6% | 50.9% |

The difference against a bar has an error of about ± 2.6 points (arm and bar together).

**Verdict: not accepted.** Seed 43 misses the USSR bar (−0.7). Seed 44 clears both bars on point
estimates (+0.4, +2.3), but neither difference is distinguishable from zero.

## Reading

* **The human opening is a US asset for this model.** Before any training it was +2.5 for the US
  on seed 43. After 80M of training from it, the US seat is +6.8 on seed 43 (about 2.6 standard
  errors) and +2.3 on seed 44.
* **For the USSR it was a handicap that training removed but did not reverse.** −7.5 before
  training, −0.7 and +0.4 after. The USSR half of the human opening (Poland 4, East Germany +1,
  Yugoslavia) is no better for this model than its own Poland 3 plus Hungary or Yugoslavia 3.
* **Overall the trained model is level to slightly ahead.** In the Elo fields, E5-15 at 590–630M
  sits at 2332–2355 against E5-11's continuation at 2310–2351 (seed 43). Seed 44's E5-15@730M+human
  dropped to the bottom of the E5 entries (as US 33–51%), which looks like a late oscillation of
  one snapshot, not a trend. Its 690M and 710M entries are level with the controls.
* **What the human opening would buy is a fixed, sane setup at no strength cost.** That removes
  the Greece and Denmark-style locks and the opening wander of E5-13. It does not buy strength by
  the owner's rule. A mixed arm (the human US setup, the USSR's own) would be the natural next
  question if the US gain is wanted.
