# Papers — what the literature did, and what this project took from it

Reading notes, not summaries: what each paper does and which of its choices this project adopted,
rejected or has not tested. The wider bibliography, mapped to this game's shape (weak hidden
information, heavy chance), is [`../method/references.md`](../method/references.md).

| paper | the part this project uses | tested here |
|:---|:---|:---|
| [`paper_nash_policy_gradient.md`](paper_nash_policy_gradient.md) | NashPG: PPO with a KL penalty to a frozen, periodically refreshed π_ref | the training loss; turning the KL off cost 169 Elo (arm I) |
| [`paper_magnetic_mirror_descent.md`](paper_magnetic_mirror_descent.md) | why regularising toward a reference converges where self-play cycles | — |
| [`paper_kl_regularized_search.md`](paper_kl_regularized_search.md) | the one-step improvement `π′(a) ∝ π(a)·exp(Q̂(a)/τ)` | the shape of the Gumbel / turn-credit targets (R29–R32, R37); distillation gains were small |
| [`paper_deepnash_rnad.md`](paper_deepnash_rnad.md) | regularised Nash dynamics at scale with hidden information | — |
| [`paper_student_of_games.md`](paper_student_of_games.md) | search in imperfect information without a public belief state | — |
| [`paper_rebel.md`](paper_rebel.md) | belief-state search, the principled alternative to determinization | not adopted: determinization is legitimate here ([`../method/references.md`](../method/references.md) §2) |
| [`paper_off_belief_learning.md`](paper_off_belief_learning.md) | why naive determinization suffers strategy fusion (`ai/search/dmcts.py` inherits it) | — |
| [`paper_cicero_diplomacy.md`](paper_cicero_diplomacy.md) | anchoring a policy to human play | human injection cost 157–236 Elo ([`../log/P7_human_injection_and_its_cost.md`](../log/P7_human_injection_and_its_cost.md)) |
| [`paper_alphastar.md`](paper_alphastar.md) | league play and opponent pools — the ancestor of `--opponent-self-pool` | the pool is required (`CLAUDE.md` invariant 14); the P24 league added nothing measurable |
| [`paper_suphx_mahjong.md`](paper_suphx_mahjong.md) | oracle-guided training and weaning the privileged critic | oracle critic built (P5), see [`../log/P30_oracle_critic_auc.md`](../log/P30_oracle_critic_auc.md) |
| [`paper_douzero_perfectdou.md`](paper_douzero_perfectdou.md) | perfect-information distillation in a card game | — |

Tests of a paper's claim live in [`../log/`](../log/README.md) and are registered in [`../runs.md`](../runs.md).
