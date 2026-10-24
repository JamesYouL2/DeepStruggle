# Reserve — ideas with a trigger, not a slot

Each entry names the measurement that would promote it to a step file. Nothing here is queued.
Add new ideas here first ([`README.md`](README.md), rule 4). Ideas that were tried are listed at the
end with their result.

## Base-model cells left over from P30
**Trigger:** the owner wants another base-recipe arm. C2 from scratch (the card-event target, a
small positive on a branch, reaches the trunk); C3 (a stepped LR over the last third); C1b (tokens
by concatenation, identities at full scale). [P30](P30_base_model_quality.md).

## Human data as state distribution and critic signal (was P7 7b–7d)
**Trigger:** a measured weakness that lives in positions self-play rarely reaches. 7b: converted
human turn-boundary positions as 5–10% of start states (`--start-pool-frac` machinery), turns ≥ 4
only; 7c: a critic-only loss on human outcomes, policy term off; 7d: the old injection
(`--inject-dataset`) re-run on the new corpus as a discriminator, never adopted. If 7d still costs
Elo, the mechanism is at fault (then a card-level human KL anchor, piKL form); if it helps, the old
corpus was (then advantage-filtered imitation). Background:
[`../log/P7_human_injection_and_its_cost.md`](../log/P7_human_injection_and_its_cost.md).

## Seeding chains self-play never reaches
**Trigger:** a leak on a chain whose precursor the network never plays (Nuclear Subs, the Cuba
trap). The Cuba defence was learned only with the scenario seeded *and* bounded logit gaps (R33);
neither alone. An effect-conditioned coup-target correction for the Subs coup (P32's B6) waits
behind it.

## PSRO-lite: meta-Nash opponent sampling
**Trigger:** a pool or league change is wanted again. Sample pool opponents from the meta-game
Nash of the pool's head-to-head matrix (Lanctot et al. 2017) instead of uniformly; needs a periodic
in-run mini-tournament.

## Optimism / extragradient in the optimizer
**Trigger:** evidence that the π_ref timescale matters but no refresh cadence both damps
oscillation and keeps learning speed. OGDA / optimistic mirror descent are last-iterate-convergent;
deeper surgery than a flag.

## Codified rewind + pool gating (was P25 step 5)
**Trigger:** runs on the production recipe stall (per-seat strength against their own earlier
snapshots flat for 40M+) more often than ~1 in 10. P25 closed without it because collapse on the
fixed pool is a recoverable delay ([`../log/P25_stress_bench.md`](../log/P25_stress_bench.md#closing-p25-2026-09-24));
design in [`../archive/E4_ladder/plans/P25_collapse_robustness.md`](../archive/E4_ladder/plans/P25_collapse_robustness.md)
step 5: resume from the last snapshot that passed with the sampling seed advanced; admit a
snapshot to the pool only if it passes the same check.

## Per-side capacity
**Trigger:** a seat stays behind while the other sharpens inside the shared network. Side-specific
adapters or heads on the shared trunk, or a side-conditioned FiLM layer. An architecture change;
the side must be an input, not a fixed per-side training mode.

## Board handicap curriculum for the weak seat
**Trigger:** one seat cannot hold its level against frozen references from scratch. KataGo-style: a
small *board* handicap at setup (extra influence, never VP: 95.5% of endings sit at |VP| = 20),
annealed to zero as that seat's share rises. An **engine change**; needs approval.

## Voice of America exposure probe
**Trigger:** the agent contests battlegrounds (empty battlegrounds at turn 8 toward the human rate).
VOA (`cid 74`) removes 4 USSR Influence from non-European countries, at most 2 per country, so a
USSR position of 1–2 there can be erased. At each `TURN_CLEANUP`: is VOA unplayed (true state), and
how many non-European countries does the USSR hold at 1–2 (report `min(count, 2)`, split controlled
from uncontrolled: 48 of the 63 have stability ≤ 2). Human yardstick only if VOA reaches play in
30+ corpus games.

## AIVAT-style evaluation
**Trigger:** two arms land within ~20 Elo and the decision matters. Value-based control variates at
die nodes (P2's clone average) and deal nodes (the pre-deal critic); ~20 Elo → ~5 for a 1,000-game
tournament. [`../log/variance_and_noise.md`](../log/variance_and_noise.md),
[`../method/references.md`](../method/references.md) §3.

**Built, chance nodes only** (`ai/eval/aivat.py` over a `tools/tournament.py --log-games` file;
[`../../tools/README.md`](../../tools/README.md) §2E). Decision nodes are not corrected: a greedy
player's term is zero and a searcher's policy is not known. **First measurement (2026-10-24, CPU,
100 deal pairs):** A13 SWA vs the R34 soup, valued by A13's critic, at 48 chance steps per game.
* 4 clones: variance ratio 1.30×.
* 16 clones: 1.51×, raw ±41.6 Elo → ±33.2.
* `vp` (tanh(VP / 10)): 1.00×.
* Mean correction: z −2.3 at 4 clones and −1.4 at 16. These use the same games, so the two are not
  independent checks.

That is ~1.5× the games, not the 16× hoped for above. Most of the outcome's variance is not in the
dice or the deal as this critic reads them. Before relying on it:
* repeat at 1,000 games per side on CI to settle the z;
* try 32 clones, since the ratio still rose from 4 to 16.

## Q-boosting proper (action-value critic, Expected SARSA(λ))
**Trigger:** chance is handled (P2) and the remaining return variance is attributed to policy
sampling. An action-value head over the flat action space; [`../method/references.md`](../method/references.md) §3.

## Categorical value head with ending-type atoms
**Trigger:** DEFCON-1 endings stay mispriced (calibration split by ending type). A plain categorical
head was −28 Elo on E3 ([`../archive/E3_ladder/plans/P1_categorical_value_advantage_filtering.md`](../archive/E3_ladder/plans/P1_categorical_value_advantage_filtering.md)).

## Entropy schedule
**Trigger:** entropy reads as the residual noise floor after the π_ref cadence is settled. Anneal
`--entropy-coef` late in training (Ataraxos damping).

## Tried, with results

* **League / PFSP** — [P24](P24_league.md).
* **Potential shaping** — a small VP potential (E7-16-44) gave no gain
  ([`../log/P30_win_only_and_vp_potential.md`](../log/P30_win_only_and_vp_potential.md)).
* **Setup anchor from human setups** — P4 arm B held the human opening on 1 of 2 seeds
  ([`../log/P4_setup_arms.md`](../log/P4_setup_arms.md)); the owner's openings as forced
  environment are R43 ([`../runs.md`](../runs.md)).
* **Search at play time / as a product** — built: Gumbel, PUCT and the rollout root are leaderboard
  players ([`../checkpoints.md`](../checkpoints.md)).
* **Belief head, oracle critic** — sized and shelved ([P5](P5_oracle_critic.md)).
* **Layout attribution (which v2.2 change carried +92 Elo)** — dead: the retired layouts are gone.
