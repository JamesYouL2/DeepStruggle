# P4 — Setup: credit through the placement block, and a human anchor as the contrast

**Status:** running (2026-09-25). Rewritten from the E3-era version, which was queued and never
run, because E4 has since measured what the setup follows. The runs are listed under *Runs*.
**Needs approval:** none. `ai/training/rollout_buffer.py` and the human-data injector only; no
engine or observation change.
**Owner's framing:** the human anchor is not expected to be adopted either way. The question is
whether a *proper* setup changes strength, and which lever can produce one.

## Goal

A sane opening, by the goal's own definition: Poland ≥ 3 as USSR, and West Germany ≥ 4,
Italy ≥ 2, Iran ≥ 2 as US. Humans meet all four in 57% of openings
([`../log/E4_goal_probes.md`](../log/E4_goal_probes.md)). Every current checkpoint either
misses or oscillates. The strongest model rated (E4-61-44@720M, 2374) meets all four in 0%, and
E4-61-43@800M has lost Poland.

## What the setup follows (E4)

The E3 version of this plan assumed a *credit* failure. The setup is 6 (USSR) and 9 (US)
consecutive placements whose value is only realised at the last. With λ = 0.98 the early
placements take their targets from the critic at half-placed boards, which are not real
positions.

E4 measured something stronger ([`../log/P21_M2d_setup_west_germany.md`](../log/P21_M2d_setup_west_germany.md)):
* the US opening is swapped wholesale between snapshots 5M apart, and it is near-certain at each
  one (P(West Germany) ≥ 0.997 or ≤ 0.11);
* the critic's value of the West Germany setup against the alternative flips sign every
  5–25M, by up to 0.6, and holds on 8 of 8 deals, so it is set by the weights, not the hand;
* the policy follows that sign in 12 of 17 snapshots.

So the setup is not starved of credit. It follows a critic whose preference oscillates with the
non-stationary self-play meta. That predicts:
* a pure credit fix (arm A) sharpens the flow toward a target that still moves;
* an anchor that does not move with the meta (arm B) stabilises the setup, at whatever cost or
  gain in strength that has. That cost or gain is the owner's question.

## Arms

Every arm uses today's default recipe: the E4-08 recipe (M2d, λ 0.98, pool 0.3/12), TF32,
centred per-entity heads, snapshots every 10M, pool every 5M. Seeds 43 and 44, 80M from scratch.

| arm | runs | change |
|:---|:---|:---|
| control | **E4-61-43, E4-61-44** (already run, same flags) | — |
| A: credit | E4-62-43, E4-62-44 | `--setup-block-lambda`: GAE λ = 1 between consecutive setup placements, so the setup's advantage telescopes to the first real position (the turn-1 headline). Normal λ everywhere else. Setup is read from the observation's phase slot (SETUP = 0). |
| B: anchor | E4-63-43, E4-63-44 | `--inject-dataset human_corpus_e4 --inject-every 1 --inject-weight 1.0 --inject-setup-only`: one supervised step per iteration on the corpus's 3,060 training setup placements (50 games held out), policy term only, with its own AdamW at 1e-4 as in P7. |

## Measure

1. **The setup probe** (`tools/scripts/goal_probes.py`, 2,000 openings) at 40M and 80M, per arm
   and per seat. It records all four targets and the placement histogram. **Stability** too:
   the same probe over every 10M snapshot, 10–80M. An oscillating setup and a stable wrong one
   are different failures.
2. **Strength:** one field with the arms and the controls at 40, 60 and 80M, plus E4-08-36@240M.
   Per seat, and head to head against the same seed's control over the late snapshots.
3. Battlegrounds and forced wins from the same probe run, as side effects.

## Reading

* **B moves the setup to human-like and holds it:** the anchor works. Its Elo against the control
  answers the owner's question: does a proper setup buy strength at this level?
* **A moves the setup:** credit was at least part of it, and A is adoptable with no human data.
* **A does not move it and B does:** the setup is an anchoring problem, not a credit problem.
  The structural answers are then a league (P24), or a slower-learning setup (a stronger KL to a
  slowly refreshed reference on setup decisions only).
* **Neither moves it:** the dose or the mechanism is wrong. Record the placement histograms,
  since the failure mode says more than the rate.

## Runs

* Controls: `E4-61-43_20260925_143628`, `E4-61-44_20260925_143648`.
* Arm A and arm B: launched 2026-09-25 (see [`../runs.md`](../runs.md)).
