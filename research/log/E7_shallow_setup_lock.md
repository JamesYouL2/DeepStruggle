# The shallow line's opening is locked: USSR Poland 6 (2026-10-01)

The owner, playing the shallow soup in the workbench: "the shallow model soup goes very strange
setup". Checked on the soup, its four ingredients, the best raw snapshot and the current runs.

## What they play (greedy; `tools/scripts/setup_placement.py`, and the policy's probabilities)

* **USSR: all six points in Poland, at p = 1.0000 on every placement.** Poland ends at 6, East
  Germany stays at its starting 3 -- exactly control, no buffer (97% of human games finish setup
  with both Poland and East Germany at 4 or more). Identical in the soup, its SWA twin, E7-02/03/04/05-44
  at 1,200M, and E7-02-44@1,200M.
* **US:** Italy 2, West Germany 3–4, France 1–3, then the two bonus points in South Korea or Iran
  (the human US opening is West Germany 4, Italy 3, Iran 2 -- owner; no France, and South Korea is unusual), depending on the hand. The soup is less sharp here
  (West Germany 0.56 against France 0.44 at one step) because its ingredients disagree.
* **Not a soup or export artefact.** The ingredients all play it, and the soup's ONNX export
  (what the workbench runs) matches PyTorch to |Δp| ≤ 7e-6 at every setup decision.

## When it formed

| snapshot | USSR setup |
|:---|:---|
| E7-01-44@10M | Finland 5–6, Yugoslavia |
| E7-01-44@40M | Poland 3, Yugoslavia 3 |
| E7-01-44@80M | Poland 5, Yugoslavia 1 (also E6-12-44@560M, the same recipe on E6) |
| E7-01-44@160M, @320M, and every later shallow snapshot | **Poland 6** |
| E7-08-43@80M (seed 43) | Poland 3, Finland 3 |
| E7-08-43@320M, @480M; E7-07-44 (C4) @320M, @480M | **Poland 6** |
| E6-03-44@550M (deep M2d) | Hungary 3, Poland 3 |

It locks between 80M and 160M in every shallow run, both seeds, with or without C4. The shallow
recipe has no setup credit and no setup entropy floor (the E5/E6 fixes for the E5-11 Greece lock,
[`E5_11_setup_lock_and_critic_views.md`](E5_11_setup_lock_and_critic_views.md)).

## What it costs (`tools/scripts/setup_oracle.py`, 2,000 deals, paired, played out by the checkpoint)

| forced opening − own | soup | E7-02-44@1,200M |
|:---|---:|---:|
| USSR: East Germany 4, Poland 4, Yugoslavia 1 (human) | +2.1 ± 1.5 | −0.4 ± 1.4 |
| USSR: East Germany 4, Poland 4, Austria 1 | −1.4 ± 1.5 | +0.1 ± 1.4 |
| USSR: East Germany 4, Poland 5 | −0.2 ± 1.1 | −1.4 ± 1.1 |
| USSR: East Germany 5, Poland 4 | −2.9 ± 1.3 | −1.2 ± 1.3 |
| US: West Germany 4, Italy 3, France 1, Iran 1 | −0.8 ± 1.5 | +0.7 ± 1.4 |
| US: West Germany 4, Italy 3, Iran 2 (**the human opening**, owner) | −1.1 ± 1.5 | +0.9 ± 1.4 |

* **In its own games the lock costs nothing measurable:** every alternative is within about two
  points, either way. Unlike E5-11's Greece (+5.4), this opening is not a mistake the playouts can see.
* **Two caveats.** The playouts are the checkpoint against itself, and it has only ever played the
  middlegame that follows Poland 6, so they may understate what a familiar alternative is worth; and
  an opponent who targets the bare East Germany (or the US South Korea) is not in its self-play.
* **The critic misjudges every unfamiliar opening:** it rates the human USSR openings 12–19 points
  worse than its own while the playouts say they are level -- it has never been trained on them.
  The same pattern as E5-11.
* **At p = 1 training never samples anything else,** so the recipe cannot discover whether another
  opening is better. If the setup matters for base-model quality, the lever is the one E5/E6 used:
  `--setup-entropy-floor` (and/or `--setup-mc-credit`) on the shallow recipe -- a recipe change,
  so an arm, judged like any other.

## The ingredient trained with the setup flags (E7-05-44) did not unlock the USSR

E7-05-44 is E7-02-44's 870M state continued to 1,200M with `--setup-mc-credit --setup-entropy-floor 0.3`.
Its setup entropy rose from 0.08 at the branch to the floor (~0.30) and held. Openings sampled at
temperature 1, 1,000 deals:

| | USSR | US, most common |
|:---|:---|:---|
| E7-05-44@1,200M (setup flags) | **Poland 6, 100%** | West Germany 4, Italy 3, Iran 1, South Korea 1 (28%); then France-heavy variants, 4–8% each |
| E7-02-44@1,200M (no flags) | Poland 6, 100% | West Germany 4, Italy 2, South Korea 2, France 1 (71%); the same with Iran 1 for one South Korea (26%) |
| the soup | Poland 6, 100% | West Germany 3, France 3, Italy 2, Iran 1 (57%) |

* **The floor's entropy all went to the US.** The floor is on one pooled number, the mean entropy
  over every setup decision of both seats (`ai/training/nash_pg.py`, `entropy_setup`), so a US that
  spreads its nine points satisfies it while the USSR stays at p = 1. The bonus that would push the
  USSR has a vanishing gradient there (the entropy gradient → 0 as p → 1), and the coefficient
  stayed at 0–0.007.
* **With the credit, the US moved towards the human opening** (West Germany 4, Italy 3, and Iran),
  with South Korea still taking one point.
* **It bought no strength:** E7-05-44's SWA is 2560 against E7-02-44's 2571 (48.1% head to head,
  [`E7_shallow_soup.md`](E7_shallow_soup.md)).
* **To test a USSR that is not locked, the floor would have to hold for each seat's setup
  separately** (a symmetric rule, not side-specific training), or the lock be broken before the
  credit can compare anything -- and on a fresh run, before it forms (80–160M).
