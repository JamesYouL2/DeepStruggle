# P30 (2026-10-03): what search adds on top of the best soup and the best single model

The owner's question, after the E7-15/16-44 tournament: what search adds on top of the best model.
This also re-measures C5's gate, "+5 pp of honest search headroom on the base recipe at saturation"
([`../plans/P30_base_model_quality.md`](../plans/P30_base_model_quality.md)).

**Setup.** Batched MCTS with 256 simulations on the model's own policy and critic
(`search:<ckpt>:256[:determinize]`, `tools/lib/player_agent.py`), against the same model's greedy
policy. 200 games per seat, `--pack-pairs 1`, temperature 0, engine 1d11c2f2.

* **Honest search** (`determinize`): the tree runs on one sampled world consistent with what the
  mover can see. This is the deployable version.
* **Perfect-information search**: the tree runs on the true state, the opponent's hand included --
  an oracle upper bound on what better hidden-information handling could add.

Models: the best model, the shallow soup E7-02/03/04/05@1200M, and the best single snapshot,
E7-02-44@1200M. Reports: `data/reports/soup_search256_{determinize,perfect}.md` and
`data/reports/e7_02_1200M_search256_{determinize,perfect}.md`.

## Search's score against its own greedy policy (draws as half)

| model | search | overall | as US | as USSR |
|:---|:---|---:|---:|---:|
| **soup** | honest | **60.0% ± 2.5** | 64.5 | 55.5 |
| **soup** | perfect information | **61.5% ± 2.4** | 64.0 | 59.0 |
| **E7-02-44@1200M** | honest | **58.0% ± 2.5** | 58.5 | 57.5 |
| **E7-02-44@1200M** | perfect information | **61.1% ± 2.4** | 60.8 | 61.5 |

Per seat ± 3.5.

## Reading

* **Honest search adds about +8 to +10 points (+55 to +70 Elo) on both models.** C5's gate (+5)
  passes on the base recipe at saturation, unlike on the adopted E6 model (+3.5,
  [`P28_step1_swa_and_search.md`](P28_step1_swa_and_search.md)). The soup leaves search as much to
  find as a single snapshot does: averaging smooths the policy but does not do search's work.
* **Perfect information adds almost nothing on top: +1.5 (soup) and +3.1 (single) over honest
  search, each within about 1 standard error of the difference.** What search gains comes from
  lookahead, not from information about the opponent's hand. This agrees with the oracle-critic
  sizing the same day ([`P30_oracle_critic_auc.md`](P30_oracle_critic_auc.md)): the hand adds
  +0.002 AUC to the critic.
* **So the case for C5 (search-driven training) rests on lookahead.** A distillation target from
  honest search is worth about +60 Elo per decision searched, and handling the hidden information
  better (beliefs, oracle critics) would add little on top.
