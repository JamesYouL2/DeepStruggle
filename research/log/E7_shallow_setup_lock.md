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

## E7-10-44: half the games set up by scripted human openings, trained as its own (2026-10-02)

E7-02-44 branched at 820M to 1,200M with `--setup-mc-credit --setup-script-frac 0.5`: in half the
games, drawn per game, both sides' setup is one of the four human variants (USSR East Germany 4,
Poland 4, Yugoslavia or Austria 1; US West Germany 4 with Italy 3 + Iran 3 or Italy 4 + Iran 2), and
those placements are trained as the policy's own -- the ratio measured from the current policy, the
credit the game result against the critic. (A first launch measured the ratio from the stale rollout
log-prob and drove the scripted openings down; voided, fixed in `544153d`.) Control: E7-02-44's own
820M → 1,200M.

**It adopted the human openings within ~20M steps** (sampled at temperature 1, 1,000 deals):

| | USSR | US |
|:---|:---|:---|
| E7-02-44@1,200M (control) | Poland 6, 100% | WG 4, Italy 2, South Korea 2, France 1 (71%) |
| E7-10-44@840M | Poland 4, EG +1, Austria 1 (73%); Poland 5 variants (22%) | the two human variants 61% |
| E7-10-44@1,200M | Poland 4, EG +1, **Austria 1 (70%)**, **Yugoslavia 1 (26%)** | **WG 4, Italy 3, Iran +2 (67%)**, **WG 4, Italy 4, Iran +1 (24%)** |
| E7-10-44 1,120–1,200M SWA | Yugoslavia 51%, Austria 42% | Italy 4 + Iran 58%, Italy 3 + Iran 33% |

The game-result credit stayed positive throughout (result − baseline +0.03 to +0.16 per update).

**Strength with its own setups is level** (owner's rule, 1,000 games per side per pairing):

| E7-10-44 against E7-02-44 | US | USSR | head to head |
|:---|---:|---:|---:|
| 1,140–1,200M snapshots | +1.1 ± 0.5 | −0.9 ± 0.5 | 51.3% |
| 1,120–1,200M SWA | +1.9 ± 0.8 | −1.4 ± 0.9 | 48.8% |

**With the human opening forced on both sides it plays better:** E7-10-44@1,200M against
E7-02-44@1,200M 53.9% (as US 54.9, as USSR 52.8); SWA against SWA 52.6%.

**The setup oracle on E7-10-44@1,200M: every opening is level in the playouts again** -- the human
variants within ±1.3 pp of its own, and **Poland 6 −0.7 ± 1.4**. But the critic now rates Poland 6
**17.8 points worse** than its own opening: the mirror image of E7-02-44's critic, which rated the
human openings 12–19 points worse than Poland 6.

### Reading

* **The opening is not where strength is.** Against itself the network is level whichever of these
  openings it plays (two checkpoints, both seats, every variant within about two points), and
  switching its habitual opening from Poland 6 to the human one cost and bought nothing at
  saturation (51.3% / 48.8%).
* **What it did buy is breadth:** the network now plays the human opening's middlegames better
  (53.9% when both sides are forced into them), and its own opening is no longer a single line.
* **The critic's verdict on openings is familiarity, not value.** Both times it rated its habitual
  opening ~18 points above alternatives the playouts call level. Any setup credit through the critic
  (the λ-return) inherits that bias, which is why the game-result credit was needed here.
* Reports: `data/reports/e7_10_44_rr{,_human}.{md,json}`.

## E7-11-44, the control: the mechanism adopts a bad opening too (2026-10-02)

The same branch and mechanism, with one deliberately bad scripted opening (USSR Romania 6; US
West Germany 7, Australia +2), 820M → 900M. Sampled at temperature 1:

| | USSR | US |
|:---|:---|:---|
| 830M | Romania 6 50%, Poland 6 18% | WG 4, France 3, Italy 2 (84%) |
| 840M | **Romania 6, 99%** | WG 5, Italy 2, Australia 1–2 (mostly) |
| 860M | Romania 6, 99% | WG 7, Australia 2 (25%) and mixes |
| 900M | **Romania 6, 99.8%** | WG 4, France 3, Italy 2 (95%) -- rejected |

And it is bad (setup oracle, 2,000 paired deals): on E7-02-44@1,200M Romania 6 costs the USSR
**−12.5 ± 1.4** against its own Poland 6, and West Germany 7 / Australia 2 costs the US −12.3 ± 1.5;
on E7-11-44@900M, now playing Romania 6, forcing Poland 6 gains **+10.9 ± 1.5**.

**So E7-10-44's adoption of the human openings is not evidence that they are better.** The cause:
the setup credit (result − critic) averaged **positive** in every update of both runs (+0.03 to
+0.18) -- the learner beats its pool ~71% of the time and the critic cannot see the opponent. A
uniform positive credit hardly moves the policy's own placements (p ≈ 1, gradient ≈ 0) but pushes
every scripted placement up at full strength, good or bad. The fix to test: centre the credit per
side within each setup batch, so a scripted opening rises only where its games beat the average
of that side's placements; then re-run this control (it should be rejected) before the human mix.

## E7-12-44: centred credit -- the US rejects, the USSR still drifts; the design has no signal (2026-10-02)

E7-11-44 re-run with the setup credit centred per side. The US rejected its bad opening throughout
(West Germany 4, France 3, Italy 2 at 95–99%). The USSR still drifted: Poland 6 ~69% at 830–840M,
**Romania 6 82% at 880M**, Romania 5 + one elsewhere at 900M.

**Why: a scripted game scripted both sides.** The USSR's Romania 6 (−12.5 pp) always met the US's
West Germany 7 / Australia 2 (−12.3 pp); the handicaps cancel, the USSR's result in scripted games
matches its normal games, and the credit carries no information about the opening. With no signal,
low-probability scripted placements random-walk -- each noisy update moves them at full strength
(∇log π ≈ 1 at p ≈ 0, against ≈ 0 for the policy's own placements at p ≈ 1), and Adam normalises
the step. The same holds for E7-10-44 (human met human).

## Status (owner, 2026-10-02: stopped)

* **Not concluded.** Neither E7-10-44's adoption of the human openings nor any opening preference
  is evidence: E7-11-44 adopted a −12.5 pp USSR opening through the same mechanism, and with both
  sides scripted the credit cannot tell a good opening from a bad one.
* **What did hold:** with the human opening forced on both sides E7-10-44 plays those positions
  better (53.9%), and strength with its own setup was level -- breadth from playing the positions,
  not from choosing them.
* **If resumed:** script one side per game (the other sets up itself), keep the per-side centring,
  and pass the stupid control (both bad openings rejected) before any real opening is run.
* **Standing findings:** the critic's opinion of an opening is familiarity (±18 pp toward whatever
  it plays); every opening tried except the stupid ones is level in playouts.
