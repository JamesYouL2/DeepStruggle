# E7 — a saturated policy cannot learn, and a logit cap cannot fix it by its gradient alone

**Question (owner, 2026-10-09):** the heads soup walks into a Cuban Missile Crisis trap
([`E7_cuban_missile_crisis_probe.md`](E7_cuban_missile_crisis_probe.md)): under the US's crisis, after the
US plays Lone Gunman, the USSR coups without the 2 Influence in Cuba to pay and loses on the spot, ~87%
of the time. Forcing the trap in every game (`--seed-scenarios cmc_combo`, R28, 200M) taught the USSR
nothing. Can the policy be made able to learn it? "Try 1 first": a cap on the policy logits.

**The obstacle.** On 60 such positions the coup leads its alternatives by a median 19–33 nats; P(coup) is
1 in float32. Policy gradient on a sampled coup is scaled by 1 − P = 0, and the alternative is never
sampled. Search cannot override it either: Gumbel's sigma(completed Q) adds at most ~15 at 256
evaluations against a 28-nat prior.

All arms below branch from `E7-A8-R1-S44@6400M` with the heads run's own flags; the trap arms add R28's
scenario. Trap readings are the scripted combo US against the snapshot, 2,000 games: USSR coups without
the means to pay, per follow-up (the plain leg: 89% at 6,450M, 83% at 6,500M, 87% at 6,600M).

| arm | what it changes | outcome |
|:---|:---|:---|
| `+A9` / `+A9-R28` | **tanh cap**: each legal move's deficit to the top move bounded to 7 through 7·tanh(d/7), centred on the top move (centring on the mean squashes the good moves together: KL 0.65 from the policy even at c = 7) | stable over 70M (entropy 0.41, KL to π_ref ~0.01, pool win rate 0.66–0.67), but **cannot learn**: 154/172 (90%) at 6,450M. The cap restores exploration (the trap's influence at ~9e-4) but the gradient reaching a saturated move's raw logit is the softmax's times 1 − tanh²(d/7), ~0.0013 at a 28-nat deficit (1.2e-6 against 9.2e-4), and lowering the top logit moves every saturated capped logit with it |
| `+A10` / `+A10-R28` | **straight-through**: the capped logits forward, the raw gradient back | **diverges** in 16M: entropy 0.31 → 0.21 / 0.16, KL to π_ref 0.14–0.19, pool win rate 0.67 → 0.63; and worse at the trap, 114/117 (97%) |
| `+A11` | **symmetric leak**: the tanh's gradient plus 0.1 × the raw one | drifts, slowly: entropy 0.37 → 0.28, KL 0.008 → 0.046, pool 0.658 → 0.638 in 40M |
| `+A12` / `+A12-R28` | **upward-only leak**, the raw gradient only where it raises a logit (unweighted first, which double-pushed rewarded top moves and collapsed entropy to 0.07 -- voided; then weighted by tanh² of the deficit to spare the top move) | drifts: entropy 0.32 → 0.15, KL up to 0.13, pool 0.66 → 0.61 in 20M |
| `+R30` | **a real loss**: 1e-4 × the mean over legal moves of relu(top − logit − 7)² | stable (entropy 0.21 → 0.55, KL ~0.008, pool 0.663; the penalty 401 → 21) but **the trap does not move**: 81% then 89% at 6,450 / 6,500M; the coup still leads by 13.6 nats (13.3 plain) |
| `+R33` | the same at 1e-3 | stable (entropy 0.28 → 0.83 and level, KL ~0.005, pool win rate ~0.65); **the trap is learned**: suicides 84, 87, 72, 65, **59%** at 6,450–6,800M (plain 83–89%). Not by declining the coup (4 of 214, plain 11 of 158) but by **rebuilding Cuba**: the USSR coups and pays in 75 of 214 follow-ups against 12 of 158 -- 2+ Influence back into Cuba before the US's follow-up, so the payment is automatic. The coup's lead on the trap positions halves (22.9 → 9.7 nats against 15.3–22.9 plain) |

## What this says

* **Any gradient that does not match the forward pass drifts the network.** A cap's leak -- straight,
  symmetric or one-sided -- is a signal with no effect on play, and it reaches the shared trunk; at an
  84-way placement node ~80 saturated moves each leak ~p·A, together comparable to the true gradient.
  Only the tanh's own gradient is consistent, and it vanishes exactly where the trap needs it.
* **A loss on the gaps is consistent and stable, and at 1e-3 it makes the trap learnable.** At 1e-4,
  averaged over every row, the trap's rows barely registered; at 1e-3 the gaps fall to ~7-10 nats,
  the alternative's probability rises from ~0 to ~1e-4, and within 400M the network learns a defence
  it never found uncapped. The cost: the sampled policy's entropy 0.28 → 0.83 -- what it does to
  strength is E7-A8-R1-S44@6400M+R34 (the penalty without the scenario): **greedy, a gain** -- the
  6,720–6,800M SWA 1546 against the plain leg's 1508 (heads soup 1550), +7.6 / +2.1 per seat, 55.5%
  head to head; **sampled at T = 1, a loss** (36.6%), the broader policy's mass on bad moves. Bounded
  gaps keep the alternatives in play during training, and the argmax is better for it.
* **The scenario produced a defence nobody wrote down.** It asked "will the USSR stop couping";
  the network answered with an earlier, cheaper move -- rebuild Cuba so the coup pays. The first
  forced scenario to do so, which is what "show the state, let training decide" was meant to allow.
* The same saturation shows wherever the network has decided: 25% of the searched decisions in the
  search-target arms had max p > 1 − 1e-6 at the start (`search_saturated_frac`).

All flags are off by default: `--ladder-logit-cap` with `--ladder-logit-cap-grad`,
`--ladder-logit-cap-leak`, `--ladder-logit-cap-leak-up` (A9–A12 in
[`../architectures_and_recipes.md`](../architectures_and_recipes.md)), and `--logit-gap-coef` /
`--logit-gap` (R30, R33).
