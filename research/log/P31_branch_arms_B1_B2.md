# P31 B1 and B2: the play-mode floor and scenario seeding, from E7-20-44@4,390M (2026-10-06)

[P31](../plans/P31_show_and_decide.md) screens four "show, then let training decide" mechanisms as
branches from a late checkpoint. The owner chose the root E7-20-44@4,390M, whose three no-change
continuations to 4,800M (E7-20-44, E7-20-44-4390M.45 / .46, [`E7_soup_4800M.md`](E7_soup_4800M.md))
are the noise floor. The position banks P31 reads arms on are not ported (owner: not now), so these
two arms are read by the panel per seat, head to head, and the self-inflicted DEFCON-1 rate -- which
cannot see whether the leaks were repaired.

**Runs** ([`../runs.md`](../runs.md)), each `--resume <E7-20-44 dir>/resume_4390060032steps.pt
--train-steps 4800000000`, E7-20-44's flags otherwise (`launch_flags --diff`: only the P31 flag):

* **B2, E7-27-44:** `--seed-scenarios subs chernobyl --seed-frac 0.05`.
* **B1, E7-28-44:** `--play-mode-floor 0.03`. E7-26-44, its first launch, is void (below).

They ran one after the other: side by side two trainers ran at 34k steps/s each, alone ~100k.

## B1's first launch was biased, and why

As first built, the floor stored `log mu` as PPO's old log-prob, `mu = 0.97 pi + 0.03 uniform`. A rare
action the floor drew then sat at ratio `pi / mu ~ 0.25`, far below the clip: a positive advantage
raised it, a negative one was clipped to a constant. The floor could only push rare actions up, and
E7-26-44's entropy rose 0.32 -> 0.37 -> 0.43 -> 0.45 in 95M steps against a flat ~0.326 on the plain
line. Stopped at ~4,485M, void. The floor now keeps `log pi` (ratio and clip on `pi_theta / pi_old`)
and weights each floor sample's surrogate by `pi_old / mu`, detached and at most `1 / (1 - eps)`;
`tests/training/test_show_and_decide.py` shows the replaced design one-sided and this one moving a
rare action down on a negative advantage.

## Training

| 20M windows, entropy | 4,390–4,410M | 4,410–4,430M | 4,430–4,450M |
|:---|---:|---:|---:|
| B1 corrected (E7-28-44) | 0.334 | 0.347 | 0.355 |
| B1 void (E7-26-44) | 0.361 | 0.417 | 0.431 |
| plain line (E7-20-44) | 0.325 | 0.326 | 0.326 |
| B2 (E7-27-44) | 0.326 | 0.336 | 0.337 |

Corrected B1 rose more slowly and held at 0.35–0.40 to the end. The floor covers ~20% of the learner's
decisions (play mode plus the non-country choices inside events) and draws uniformly in 3% of them;
the cost is the **DEFCON tax**: 15–16% of B1's self-play games ended in self-inflicted DEFCON-1,
against 1.6–1.8% on the plain line and on B2, already in the first 40M window. B2 trained like the
plain line throughout (self-play US 48% over 4,390–4,450M against the line's 50%); in 5% of games
Subs and Chernobyl were forced (in the smoke, ~35% and ~20% of seeded games held the card at an
action round).

## The trained checkpoints

**Panel per seat and head to head against the three siblings** (`data/reports/p31_E7-2{7,8}-44_rr.{md,json}`;
1,000 games per side per pair, temperature 0; Δ ± SE):

| arm | US | USSR | head to head |
|:---|:---|:---|:---|
| B1@4800M vs siblings@4800M | +0.4 ± 0.6 | +0.1 ± 0.7 | 50.3% ± 0.6 |
| B1 SWA 4,720–4,800M vs siblings' SWAs | +0.4 ± 0.5 | −1.1 ± 0.6 | 50.0% ± 0.6 |
| B2@4800M vs siblings@4800M | +0.2 ± 0.6 | +0.5 ± 0.7 | 50.6% ± 0.6 |
| B2 SWA vs siblings' SWAs | −0.8 ± 0.5 | +0.8 ± 0.6 | 50.6% ± 0.6 |
| noise floor: .45 / .46 vs E7-20-44@4800M | +0.9 / +1.1 | −0.4 / −0.4 | 52.2% / 52.4% |

**Does the trained checkpoint, playing without the floor, end games by its own DEFCON-1?** Paired
two-model matches against E7-20-44@4800M, 2,000 games a side; own-decision DEFCON-1 per game
(`data/reports/p31_defcon_*`):

| A vs E7-20-44@4800M | T = 1: A / E7-20-44 | T = 0: A / E7-20-44 | A wins (T 1 / 0) |
|:---|:---|:---|:---|
| B1 (E7-28-44) | 0.85% / 0.95% | 0.80% / 0.85% | 49.8% / 51.1% |
| sibling .45 (control) | 0.65% / 0.78% | 0.40% / 0.65% | 51.2% / 51.4% |
| B2 (E7-27-44) | 1.00% / 0.82% | 0.70% / 0.65% | 50.8% / 50.1% |

## Reading

* **Both arms are level with their siblings on strength**, in both seats, as snapshots and as SWAs:
  they pass the "not worse" half of the owner's rule and show nothing on the "better" half -- which
  P31 expected, since each leak is worth 1–2 points a game in both seats.
* **The floor's DEFCON tax stays in training.** Played without the floor, B1's checkpoint ends no more
  games by its own DEFCON-1 than the plain model in the same games (0.85 vs 0.95%, 0.80 vs 0.85%).
  The tax is the uniform draws' own moves, and the floor-drawn suicidal options were pushed down,
  not learned.
* **Whether either arm repaired anything is not measured here.** Wargames at a lead (B1's target) and
  the Subs / Chernobyl chains (B2's) need the banks or a direct count of those plays. B1 cannot reach
  the Subs chain at all: its payoff is a battleground coup at DEFCON 2, and the floor excludes country
  targets by design (owner, 2026-10-06), so a floored OPS_COUP lands where the policy already aims.

## The target cards, counted directly (owner, 2026-10-06)

Self-play, 4,000 games per checkpoint at 4,800M, at temperature 0.1 and 1; per card, the share of its
plays at a play-mode node with the event legal that chose the event (Chernobyl: the US's plays only),
and the share of games ending by Wargames (`card_event_probe.py` in the job scratch; `primary_id` at a
play-mode node is the Resolution, 0 EVENT, 1 SPACE, 2–4 OPS -- `ai/eval/play_modes.py` labels these
wrongly, though its event column is right).

| | ended by Wargames, T 0.1 / 1 | Wargames event US / USSR (T 0.1) | Chernobyl event (US), T 0.1 / 1 | One Small Step US / USSR (T 0.1) | Arms Race US / USSR (T 0.1) |
|:---|:---|:---|:---|:---|:---|
| E7-20-44 | 0.30% / 0.20% | 1.1% / 0.5% | 2.0% / 2.7% | 0.5% / 0.4% | 0.9% / 1.2% |
| E7-20-44-4390M.45 | 0.20% / 0.30% | 0.2% / 0.0% | 1.5% / 1.6% | 0.6% / 0.5% | 0.4% / 1.3% |
| E7-20-44-4390M.46 | 0.42% / 0.60% | 0.8% / 1.1% | 1.3% / 1.8% | 0.2% / 0.9% | 0.6% / 2.0% |
| **B1, E7-28-44** | 0.57% / 0.42% | 0.2% / 2.4% | **0.0% / 0.5%** | 0.1% / 0.4% | 0.2% / 1.5% |
| **B2, E7-27-44** | 0.17% / 0.15% | 0.0% / 0.6% | **0.2% / 0.3%** | 0.2% / 0.3% | 0.2% / 1.1% |

(Plays per cell: Wargames ~800–1,000, One Small Step and Arms Race ~2,200–2,600, Chernobyl ~760–950.)

* **Wargames, One Small Step, Arms Race: no change.** Every rate sits at its 0–2% floor in both arms,
  inside the siblings' spread; B1's Wargames endings (0.57 / 0.42%) are within the siblings'
  0.20–0.60%.
* **Chernobyl moved the wrong way in both arms**, at both temperatures: the siblings play it for the
  event in ~2% of its plays (52 of 2,585 at T 1), B1 in 4 of 806 and B2 in 3 of 886, where ~16–18
  were expected -- far outside chance.
* **The likely cause is the random region, in both arms.** Chernobyl's value is the region it blocks.
  B1's floor also covers the region choice inside the event, and B2's forced plays drew the region
  uniformly (P31's design). A Chernobyl into a random region is mostly wasted, so both arms showed
  training many "Chernobyl event, then nothing gained" games the plain line never plays. Not yet
  verified directly. The fix that keeps P31's idea: force only the event play and leave the region
  to the policy (its own decision, with gradient), and keep the floor off region choices.

## E7-29-44: applicable events forced, trained as the policy's own (owner, 2026-10-06)

`--force-applicable-events wargames arms_race one_small_step --force-event-frac 0.1`: at the learner's
play-mode decision for the card with its event legal and applicable (Wargames: DEFCON 2 and 7+ VP
ahead; Arms Race: ahead in military Ops; One Small Step: behind in space), the event is played in 10%
of cases and credited as the policy's own (`learner = 1`, `log pi(EVENT)`, no weight; owner's choice
over an importance-weighted version, whose expected gradient would not change). From E7-20-44@4,390M
to 4,800M. The learner meets those spots ~3,500 (Wargames), ~15,700 (Arms Race) and ~17,900 (One Small
Step) times per 21M steps, so ~1,600–1,700 forced plays per 21M for each of the two latter.

**Strength: worse in both seats** (`data/reports/p31_E7-29-44_rr.{md,json}`):

| | US | USSR | head to head |
|:---|:---|:---|:---|
| E7-29-44@4800M vs the three siblings | **−2.6 ± 0.6** | **−1.4 ± 0.7** | **43.4% ± 0.6** |
| E7-29-44 SWA vs the siblings' SWAs | **−2.2 ± 0.6** | −1.0 ± 0.6 | **46.2% ± 0.6** |

**The cards, split by the rule's own applicability** (4,000 self-play games, T 0.1 / T 1; siblings ~1%
or less on every cell):

| | applicable | not applicable |
|:---|:---|:---|
| Wargames, USSR | **62.6% / 57.8%** of ~205 | **10.3% / 9.9%** of ~710 |
| Wargames, US | 11.4% / 7.4% of ~190 | 0.2% / 0.0% of ~680 |
| One Small Step, both | **0.0%** of ~950 a side | 0.0% |
| Arms Race, both | **0.0%** of 500–975 | 0.0% |
| Chernobyl (not forced) | **0.0%** of 796 / 888 | |

Games ended by Wargames: 4.4% / 4.0% (siblings 0.2–0.6%). At 4,440M (50M in) the same split read
Wargames applicable US 19.6% / USSR 9.3%, One Small Step 0%, Arms Race 0.4–0.6%.

**Every event was suppressed.** The event share of all the learner's play-mode decisions over training:

| | 4,390–4,490M | 4,500–4,600M | 4,600–4,700M | 4,700–4,800M |
|:---|---:|---:|---:|---:|
| **E7-29-44** | 0.307 | 0.259 | 0.212 | **0.147** |
| E7-20-44 (plain) | 0.316 | 0.303 | 0.309 | 0.304 |
| B1, E7-28-44 | 0.319 | 0.300 | 0.310 | 0.281 |
| B2, E7-27-44 | 0.317 | 0.321 | 0.295 | 0.302 |

### Reading

* **The forced One Small Step and Arms Race events were judged worse than their Ops**, and credited as
  the policy's own, each carried a full-size gradient on the EVENT logit -- `(1 - pi) * A` with
  `pi ~ 0.002`, against `pi * A` for a sample the policy drew itself. ~3,300 such plays per 21M steps,
  mostly with negative advantage, pushed EVENT down. **The play-mode head is shared by every card**
  (EVENT / SPACE / OPS slots, `ai/eval/play_modes.py`'s premise), so the push spread to cards never
  forced: the event share of all play-mode decisions halved (31% -> 15%), Chernobyl went to 0. That
  is the main cost in strength.
* **Wargames was learned, but not the condition.** Forced Wargames at 7+ VP is an instant win, so its
  credit is large and positive; the USSR took it up to ~60% where applicable but also ~10% where it is
  not (a smaller lead, which hands over the game, or DEFCON above 2, which wastes the card). The US
  moved much less (7–11%). The only signal against the non-applicable plays is the policy's own rare
  ones.
* **The same spillover may explain Chernobyl's fall in B1 and B2**, beside the random region: anything
  that pushes EVENT down at some cards drags it at others through the shared head.
* **What this says about the lever.** Crediting a forced action as the policy's own is a much stronger
  push than anything the policy samples, in both directions, and through a shared head it does not
  stay on the card it was aimed at. For a next attempt: Wargames only (the one card whose forced
  outcome is unambiguous), a much smaller fraction, and negative examples at its non-applicable spots
  -- or a card-specific play-mode representation so a push at one card stays there, which is an
  architecture question (P30).

## Does the trunk know, at Wargames' branch, whether ending the game wins? (owner, 2026-10-06)

At DEFCON 2 Wargames' event asks a `CHOOSE_BRANCH`: branch 0 gives the opponent 6 VP and ends the game,
branch 1 passes (`engine/src/events/late_war.cpp`, `trigger_wargames`). Branch states: self-play at T 1,
8,000 games; at every Wargames play-mode decision at DEFCON 2 a clone is stepped with EVENT to reach the
branch. Label: the branch state's clone stepped with branch 0 -- 1 if the game then ends won by the side
deciding. Probe: L2 logistic regression on the trunk output `h` (480 floats), 5-fold CV, held-out AUC
(`wargames_probe.py` in the job scratch).

| checkpoint | branch states | ending wins | trunk LR AUC (acc) | VP lead AUC | policy P(end) AUC |
|:---|---:|---:|---:|---:|---:|
| E7-20-44@4800M (plain) | 2,942 | 24.6% | **0.984** (94.1%) | 1.000 | **0.633** |
| E7-29-44@4800M (forced) | 3,270 | 25.4% | 0.994 (96.5%) | 1.000 | 0.962 |

The policy's P(end) by the decider's VP lead (ending wins exactly at 7+):

| lead | < 5 | 5 | 6 | 7 | 8+ |
|:---|---:|---:|---:|---:|---:|
| E7-20-44 | 0.19 | 0.21 | 0.17 | 0.19 | 0.29 |
| E7-29-44 | 0.20 | 0.70 | 0.86 | 0.92 | 0.98 |

* **The trunk knows; the plain policy at the branch does not use it.** A linear readout of `h`
  predicts the outcome at 0.98 AUC, while the plain policy ends the game ~20% of the time whatever
  the lead. A plain model that reaches this branch throws the game about one time in five when
  behind -- which would make playing Wargames' event look poor to the critic, and the play-mode
  decision decline it even at a winning lead: a likely mechanism of the Wargames leak.
* **Forcing taught "a big lead ends it", not the threshold:** E7-29-44 ends at a lead of 5 or 6 (both
  losses) 70–86% of the time, having seen forced plays only at 7+.
* **Representation is not the bottleneck; the branch head is.** Branch slots 200–207 are shared by every
  card with a branch, so slot 0 means "end the game" for Wargames and something else elsewhere, and a
  head that rarely sees Wargames' branch learns a lead-independent average. The owner's proposal -- a
  branch head reading the trunk context plus a one-hot of which branching card is resolving -- is aimed
  at exactly this; it still needs training signal at the branch, which the floor over branch rows gives.

## E7-30-44: a per-branch head, with Wargames' branch visited (owner, 2026-10-06)

`--ladder-branch-head`: a modifier on the branch block (flat 200..219), `branch_head_net(cat[h, one-hot of
the ACTIVE_NOW card])` (Linear 480+110 -> 128, GELU, Linear 128 -> 20) added to the dense logits there,
zero-initialised, added on resume. Plus `--force-applicable-events wargames_branch --force-event-credit
environment`: Wargames' event at DEFCON 2, **any lead**, in 10% of the learner's plays, learner 0, so the
end-or-pass branch that follows (~21k visits over the arm) is the policy's own decision with both
outcomes on offer. No floor. From E7-20-44@4,390M to 4,800M; ~96k steps/s, as the plain line.

**Wargames' branch, P(end) by the decider's lead** (ending wins at 7+; `wargames_probe.py`, 8,000 games):

| | < 5 | 5 | 6 | 7 | 8+ | AUC | P(end): ending wins / loses |
|:---|---:|---:|---:|---:|---:|---:|:---|
| E7-20-44 (plain) | 0.19 | 0.21 | 0.17 | 0.19 | 0.29 | 0.63 | 0.28 / 0.19 |
| E7-29-44 (forced as own) | 0.20 | 0.70 | 0.86 | 0.92 | 0.98 | 0.96 | 0.97 / 0.26 |
| E7-30-44 @4,440M (50M in) | 0.05 | 0.09 | 0.12 | 0.07 | 0.13 | 0.75 | 0.13 / 0.05 |
| **E7-30-44 @4,800M** | **0.06** | 0.24 | 0.41 | 0.35 | **0.85** | **0.95** | **0.78 / 0.09** |

The trunk's own linear readout stays at 0.99 AUC.

**Strength against the three siblings:** @4800M US −0.7 ± 0.6, USSR −0.6 ± 0.7, head to head 50.3%; SWAs
−0.1 / +0.0, 51.0% (noise floor +0.9…+1.1 / −0.4, 52.2–52.4%).

**Cards** (4,000 games; T 0.1 / T 1; siblings in brackets):

* Wargames played for the event where applicable: US **11.1% / 8.8%** (~1–2%), USSR **4.5% / 4.3%**
  (0–2.5%); where not applicable: US 0.6% / 1.8%, USSR 0.5% / 0.5% (~0.5–1%). Ended by Wargames 1.0% /
  0.7% (0.2–0.6%).
* One Small Step and Arms Race: unchanged, 0.2–1.5% (siblings' spread).
* Chernobyl: 0.7% / 1.3% (1.3–2.7%) -- at the low edge of the siblings; the head also corrects the region
  slots Chernobyl uses, which may be the cause.

### Reading

* **The branch is fixed without a cost.** The policy now ends the game when it wins (0.78) and rarely
  when it loses (0.09), where the plain model ended ~0.2 regardless; strength is level with the
  siblings in both seats, where E7-29-44 lost 45 Elo.
* **The leak behind it has begun to close.** With a sound branch, the play-mode decision takes
  Wargames' event where it applies 5–10x as often as the plain line, and not where it does not -- the
  reverse-curriculum order P31 predicted: the last link learned first, then the precursor.
* **The 7-VP line is still soft.** At leads 6 and 7 P(end) is 0.41 and 0.35 (~70 cases each): "a big
  lead ends it", not the exact threshold, though the trunk carries it at 0.99 AUC. More branch visits or a
  longer run are the obvious next step.
* Learning ran in two phases: at 50M the policy had mostly learned "do not end" (three in four visits
  are at losing leads), and only later to take the 7+ wins.

## E7-31-44: E7-30-44 continued 400M with twice the forcing (owner, 2026-10-06)

E7-30-44's 4,800M end state to 5,200M, its flags with `--force-event-frac 0.2` (Wargames at DEFCON 2, any
lead, environment credit). `launch_flags --diff` against E7-30-44: only that and `--train-steps`.

**Wargames' branch, P(end) by lead** (ending wins at 7+):

| | < 5 | 5 | 6 | 7 | 8+ | AUC | P(end): wins / loses |
|:---|---:|---:|---:|---:|---:|---:|:---|
| E7-20-44 (plain) | 0.19 | 0.21 | 0.17 | 0.19 | 0.29 | 0.63 | 0.28 / 0.19 |
| E7-30-44 @4800M (10%) | 0.06 | 0.24 | 0.41 | 0.35 | 0.85 | 0.95 | 0.78 / 0.09 |
| **E7-31-44 @5200M (20%)** | **0.02** | 0.27 | 0.37 | **0.68** | **0.97** | **0.99** | **0.93 / 0.05** |

**Strength** (`data/reports/p31_E7-31-44_rr.{md,json}`): against E7-30-44@4800M +1.3 ± 0.7 / +0.6 ± 0.8,
head to head 49.5%; SWAs +0.4 / −0.6, 49.6%; against the plain siblings at 4,800M (not step-matched)
+0.6 / +0.1, 49.0%; SWA against the plain line's SWA +0.8 / −0.9, 50.1%. Level throughout.

**Cards** (4,000 games, T 0.1 / T 1):

| Wargames played for the event | applicable | not applicable |
|:---|:---|:---|
| plain line | ~1–2% | ~0.5–1% |
| E7-30-44 | US 11.1 / 8.8%, USSR 4.5 / 4.3% | 0.5–1.8% |
| **E7-31-44** | **US 59.6 / 50.0%, USSR 45.9 / 49.0%** | **US 15.5 / 15.4%, USSR 8.3 / 7.1%** |

Ended by Wargames 5.45% / 5.15% (plain 0.2–0.6%). One Small Step, Arms Race and Chernobyl inside the
siblings' range (0.1–4.2%; Chernobyl 1.0% / 1.4%).

### Reading

* **Twice the forcing sharpened the branch:** the 6 / 7 line is ordered and wider (0.37 / 0.68 against
  E7-30-44's 0.41 / 0.35), 0.93 when ending wins, 0.05 when it loses; still 0.27–0.37 at leads 5–6.
* **The Wargames leak has largely closed:** where a win is on offer the model now plays the event about
  half the time (plain ~1–2%; the review's "429 of 430 declined").
* **It now also plays it where it does not apply, 7–15%** -- DEFCON above 2 (the event does nothing;
  the card's Ops are lost) or a lead under 7 (the branch mostly passes, the card is wasted; at 5–6 it
  still ends, and loses, 27–37% of the time). Likely the forcing condition's doing: forced at DEFCON 2 at
  any lead, the event is mostly seen at losing leads where passing is safe, so playing it looks free.
* **Strength is level**: the wins taken and the cards wasted roughly cancel in self-play, as P31
  expected of a leak worth about a point a game in both seats.

## E7-32-44: the washout check -- E7-31-44 continued with the forcing off (owner, 2026-10-07)

Owner: "If we try to make model train more on its own, now, when it plays wargames, can we expect it to
learn when to play them?" E7-31-44's 5,200M end state to 5,600M, branch head kept, no forcing
(`launch_flags --diff`: only the three forcing flags back to their defaults and `--train-steps`). Read
at 5,400M and 5,600M against the start, with Wargames' not-applicable plays split in two.

**Wargames played for its event** (4,000 games; T 0.1 / T 1):

| | applicable | DEFCON 2, lead < 7 | DEFCON > 2 |
|:---|:---|:---|:---|
| US @5200M (start) | 58.7 / 53.0% | 19.2 / 18.9% | 8.8 / 11.3% (102 / 106) |
| US @5400M | 87.6 / 77.9% | 24.7 / 20.8% | 9.7 / 6.1% (62 / 98) |
| US @5600M | **83.6 / 83.4%** | 12.7 / 14.8% | 12.3 / 7.8% (57 / 51) |
| USSR @5200M (start) | 60.9 / 51.9% | 12.4 / 14.9% | 2.9 / 4.0% (345 / 353) |
| USSR @5400M | 71.1 / 62.2% | 12.0 / 8.4% | 1.8 / 2.2% (166 / 185) |
| USSR @5600M | **84.6 / 79.1%** | 16.2 / 14.1% | 18.4 / 6.9% (76 / 58) |

Games ended by Wargames: 6.1 / 5.4% -> 8.4 / 7.3% -> 10.0 / 10.7%. Plays at DEFCON above 2 fell from
~100 / ~350 to ~55 per side while applicable spots rose (230 / 160 -> 259 / 230 at T 1): the card is
being held for DEFCON 2.

**Wargames' branch, P(end) by lead** (8,000 games):

| | < 5 | 5 | 6 | 7 | 8+ | AUC |
|:---|---:|---:|---:|---:|---:|---:|
| E7-31-44 @5200M (start, re-run) | 0.025 | 0.24 | 0.48 | 0.69 | 0.98 | 0.986 |
| E7-32-44 @5400M | 0.019 | 0.19 | 0.38 | 0.65 | 0.95 | 0.987 |
| E7-32-44 @5600M | 0.030 | 0.35 | 0.57 | 0.75 | 0.98 | 0.981 |

**Strength** (`data/reports/p31_E7-32-44_rr.{md,json}`): @5600M against E7-31-44@5200M −0.2 ± 0.7 / +0.5 ±
0.8, head to head 52.3% ± 1.1; SWAs +0.6 / +1.3, head to head 47.2% ± 1.1; against the plain E7-20-44@4800M
+1.0 / +0.2, 51.4%. Elo in the field: SWAs 1684 (E7-31-44) / 1680, snapshots 1655 / 1644. Level.

### Reading

* **The skill persists and grows on its own.** With no forcing, Wargames where it wins went from ~55% to
  ~83% in both seats, and the card is held for DEFCON 2 more often -- P31's "persisting" case: the spot
  recurs and the move wins, so on-policy training keeps it. **The forcing is needed for acquisition only.**
* **What the model does not learn on its own is the boundary.** The not-applicable plays at DEFCON 2 with
  a lead under 7 stay at 12–25%, and the branch's P(end) at leads 5–6 drifts (0.24 / 0.48 -> 0.19 / 0.38
  -> 0.35 / 0.57) instead of falling: a one-VP threshold reached only through the model's own plays gets
  too thin and noisy a signal to sharpen.
* **Strength stays level**, as through every Wargames arm: the extra wins and the wasted or lost cards
  roughly cancel in self-play.

## E7-33-44: card-conditioned play-mode and branch heads, from scratch (owner, 2026-10-07)

Owner: "add a similar head to 'play mode' (trunk context + one-hot of cards played), and train it from
scratch to 1200M" -- both heads, no forcing. `--ladder-play-mode-head` adds to the five play-mode logits
(110..114) a zero-initialised correction from [trunk output, one-hot of the card being played];
`--ladder-branch-head` does the same for the branch block (200..219) with the resolving card (E7-30-44's
head). E7-02-44's recipe from scratch, seed 44 (`launch_flags --diff` against E7-01-44: only the two
heads and `--train-steps`), read at 1,200M against E7-02-44@1200M.

**Training** was healthy throughout: entropy within 0.02 of E7-02-44's and level with it at the end
(0.380 / 0.381 at 1,120–1,160M). US self-play ran 2–4 points under E7-02-44's in every 40M window from
680M (0.46–0.48 against 0.48–0.52) -- and, as the strength reading shows, that one time it was a real US
deficit. An early pair at 600M (53.9%, +3.5 SE) did not hold.

**Strength** (`data/reports/e7_33_rr.{md,json}`), against E7-02-44 on the E6-03-44 panel:

| | US | USSR | head to head |
|:---|:---|:---|:---|
| 1,160–1,200M snapshots, 3 each | **−3.3 ± 0.6** | −0.9 ± 0.5 | **46.9% ± 0.4** (as US 44.6, as USSR 49.2) |
| 1,120–1,200M SWAs | −2.0 ± 0.9 | −0.6 ± 0.9 | 48.3% ± 1.1 (as US 46.0, as USSR 50.6) |
| branch spread: E7-03-44 SWA | +1.1 ± 0.9 | −0.4 ± 0.9 | 50.3% ± 1.1 |

Elo: SWAs 1613 (E7-02-44) / 1610 (E7-03-44) / 1601; every E7-33-44 snapshot (1546–1554) below every
E7-02-44 snapshot (1562–1575). **Not accepted** by the owner's rule.

**But the deficit is the size of a from-scratch divergence, not clearly the heads'.** E7-23-44 (compile, also
E7-02-44's recipe from scratch at seed 44) lost to E7-02-44 by as much, in the other seat. The three SWAs
in one field (`data/reports/e7_33_vs_23_rr.{md,json}`):

| | US | USSR | head to head |
|:---|:---|:---|:---|
| E7-33-44 vs E7-23-44 | −1.5 | +1.9 | 50.6% ± 1.1 (as US 53.4, as USSR 47.9) |
| E7-33-44 vs E7-02-44 | −2.0 | −0.6 | 48.3% ± 1.1 |
| E7-23-44 vs E7-02-44 | −0.5 | −2.5 | 46.2% ± 1.1 |

Elo there 1680 / 1665 / 1661. The two variants are level with each other and both ~1.5–4 points below
E7-02-44. All three share seed 44, so this is not seed variance: a changed computation (compile's
numerics, two extra heads) sends a from-scratch run down another trajectory, which acts like a fresh
draw.

**The recipe's own seed spread already exists: E7-08-43**, E7-02-44's recipe from scratch at seed 43
(`E7-A4-R1-S43`), which I had overlooked. The four from-scratch SWAs at 1,120–1,200M in one field
(`data/reports/e7_scratch_spread_rr.{md,json}`):

| SWA | Elo | panel US / USSR | against E7-02-44 (seed 44) | against E7-08-43 (seed 43) |
|:---|---:|:---|:---|:---|
| E7-02-44 (plain, seed 44) | 1661 | 86.3 / 86.3 | -- | 56.6% ± 1.1 |
| E7-33-44 (heads) | 1648 | 84.4 / 85.7 | 48.3% | 55.8% (as US 59.6, as USSR 52.0) |
| E7-23-44 (compile) | 1644 | 85.9 / 83.8 | 46.2% | 55.9% |
| E7-08-43 (plain, seed 43) | 1604 | 80.7 / 83.4 | 43.4% | -- |

The plain recipe's two seeds are 57 Elo and 56.6% apart; both variants land between them, nearer
seed 44. Against the control the owner's rule fails, but **within the recipe's seed spread E7-33-44
is level with the plain recipe** -- neither better nor shown worse.

**What the heads did** -- here they clearly work. Card counts (4,000 games, T 0.1 / T 1), share played
for the event with the event legal:

| | E7-33-44 | E7-02-44 |
|:---|:---|:---|
| games ended by Wargames | **4.5 / 5.4%** | 0.25 / 0.23% |
| Wargames US: applicable / DEFCON 2 lead < 7 / DEFCON > 2 | **54 / 53%** · 5.3 / 6.3% · 8.1 / 7.1% | 0.6 / 1.4% · 0.8 / 0.5% · 1.2 / 1.0% |
| Wargames USSR: same | **60 / 62%** · 7.9 / 8.5% · 3.8 / 1.6% | 1.6 / 0.0% · 0.3 / 1.0% · 0.2 / 0.2% |
| Arms Race US applicable / not | 3.3 / 4.1% · 2.8 / 2.3% | 0.7 / 0.8% · 0.6 / 0.7% |
| Arms Race USSR applicable / not | 9.9 / 7.6% · 5.9 / 3.9% | 2.1 / 2.8% · 1.3 / 1.0% |
| One Small Step US / USSR applicable | 0.7 / 0.7% · 0.1 / 0.6% | 0.1 / 0.5% · 1.1 / 1.1% |
| Chernobyl US | 1.7 / 2.5% | 0.6 / 1.5% |

Wargames' branch, P(end) by lead (8,000 games):

| | < 5 | 5 | 6 | 7 | 8+ | policy AUC | trunk LR AUC |
|:---|---:|---:|---:|---:|---:|---:|---:|
| E7-33-44 @1200M | 0.107 | 0.29 | 0.38 | 0.48 | **0.875** | **0.938** | 0.994 |
| E7-02-44 @1200M | 0.198 | 0.32 | 0.43 | 0.53 | 0.640 | 0.812 | 0.989 |

### Reading

* **The play-mode head found Wargames without forcing.** E7-02-44 never plays it for the event (≤1.6%);
  E7-33-44 does where it wins in 54–62% of spots and 5–8% where it does not -- a 7–11x discrimination
  that the forced arms reached only by being shown the move (E7-29..32: 55–88%, with 12–25% misfires).
  The shared play-mode head could not learn a per-card EVENT preference without spilling it onto every
  card (E7-29-44); conditioning on the card removes that, and the plain on-policy gradient then suffices.
* **The branch head sharpens the clear cases, not the threshold.** P(end) rises to 0.875 at leads of 8+
  and falls to 0.107 below 5; at 6 against 7 neither run separates (0.38 / 0.48, 0.43 / 0.53). The trunk
  knows the answer equally well in both (AUC 0.99): the head lets the policy use it.
* **Arms Race moved, but not by its condition** (US 3.3 vs 2.8%), and One Small Step not at all: an
  event the model rarely benefits from gets no signal to learn a condition from.
* **Strength: not accepted against E7-02-44, but inside the plain recipe's seed spread** (between
  E7-02-44 and E7-08-43, level with E7-23-44). The Wargames gain is too rare (≈5% of games) to show
  in strength either way.
