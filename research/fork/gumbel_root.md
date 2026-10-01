# The Gumbel root at play time (2026-09-30)

**What it is.** `BatchedMCTSConfig.gumbel_k = k` replaces PUCT at the root with Danihelka et al.
(ICLR 2022): k candidates by Gumbel-top-k over the log-priors, the budget split by sequential
halving, the best of g + logit + σ(q) played; interior nodes stay PUCT; σ uses mctx's defaults
(c_visit 50, c_scale 0.1). Commit `b249e6c` on branch `feat/mcts-gumbel` (not on the branch
carrying this note; cited, not linked). Spec: `search:<onnx>:64[:determinize]:gumbel_k=K`.

**Why it was built.** PUCT first visits a move of prior p after ~1/p simulations. On the E6
model's USSR opening (Hungary at 99.9% for the first placement), Poland is first tried near 9,400
simulations and overtakes Hungary near 30,000. With k = 16 every candidate is tried within 16
simulations, but the first placement stays Hungary up to 256 (2 of 10 search seeds pick Poland at
c_scale 1.0): at 64 simulations the values are Hungary +0.080, Poland +0.079, and the gap appears
only deep (~0.02 at 65k).

## Result

E6-06-44@soup_680-760 throughout, temperature 0, 256 games a side (512 games), one standard error
≈ 15 Elo. Pair-internal Elo, not the ladder's scale.

| matchup | Elo Δ | run | commit |
|:---|---:|:---|:---|
| honest (determinized) Gumbel k=2, 64 sims vs raw net | +11.5 | `36771294059` | `af5e1ef` |
| honest plain PUCT, 64 sims vs raw net | +20.4 | `36771298459` | `af5e1ef` |
| honest Gumbel k=2 vs honest plain PUCT | +6.1 | `36771301679` | `af5e1ef` |
| **privileged** Gumbel k=2 vs raw net | +158.0 | `36733976167` | `b249e6c` |
| **privileged** Gumbel k=4 vs raw net | +102.6 | `36733990044` | `b249e6c` |
| **privileged** Gumbel k=8 vs raw net | +94.6 | `36734002465` | `b249e6c` |

## Reading

* **Honest Gumbel is level with honest PUCT** (+6 ± 15). Neither adds much to E6-06-44: +12 to
  +20 Elo, against +52 for 64-sim search on the weaker E5-11-43
  ([`../log/E5_search_budget_sweep.md`](../log/E5_search_budget_sweep.md)). The stronger the net,
  the less a 64-simulation search over it adds.
* **Privileged Gumbel gains a lot, which E3's privileged PUCT did not** (+2.5 ± 5.5 pp,
  [`../log/search_cost_and_coverage.md`](../log/search_cost_and_coverage.md) §9). Unverified
  explanation: PUCT rarely leaves the prior ([`determinization_targets.md`](determinization_targets.md)),
  so extra information has no route to the move; Gumbel's forced candidates give it one. A
  privileged-PUCT run on E6 would test it, and is not on record.
* Smaller k is better privileged (k=2 > 4 > 8): more simulations per candidate beats more
  candidates at 64.

## What this does not say

* 512 games a run: the honest differences are all within about one standard error of each other.
* Play-time only. The training target Gumbel MuZero uses — the completed-Q improved policy — is
  not implemented; the root only chooses a move.
* The first `af5e1ef` dispatches (`36767303951`, `36767317803`, `36768140330`) failed on the
  `hf:` pattern rejecting `@` in a file name, which `af5e1ef` fixed; they produced no games.
