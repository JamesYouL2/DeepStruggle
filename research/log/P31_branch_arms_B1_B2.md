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
