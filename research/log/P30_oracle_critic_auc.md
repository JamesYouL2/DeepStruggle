# P30 (2026-10-03): can an oracle critic beat the honest one? Sized on the best model -- barely

The owner's question: can a critic trained with the opponent's hand give a better AUC than the
honest one? Measured offline with `tools/scripts/oracle_sizing.py`, the P5 sizing tool
([`../plans/P5_oracle_critic.md`](../plans/P5_oracle_critic.md)), now reporting AUC and with a
`--raw-obs` option.

**Method.** Self-play of the best model, the shallow soup E7-02/03/04/05@1200M, at temperature 1:
about 20,000 games, 2% of decisions recorded (about 200k records), scored on 4,000 held-out *games*.
Two heads of the same shape are fitted on the same records. The public head gets the network's trunk
features plus its own v_win; the oracle head gets the same plus the opponent's hand (110 bits). The
oracle sees no deck order and no future rolls: those are chance, not hidden information. A second
pair is linear on the critic's own logit, with and without the hand. The `--raw-obs` run also gives
both MLP heads the raw 3,824-float observation, in case the trunk dropped something the hand needs
(P30: the trunk loses state at its input projections). Engine 1d11c2f2. Reports:
`data/reports/oracle_sizing_soup_{trunk,raw}.{md,json}`.

## AUC of the game result (wins against losses)

| slice | soup's own critic | MLP public | MLP + opponent's hand | Δ | linear on critic | + hand | Δ |
|:---|---:|---:|---:|---:|---:|---:|---:|
| all, trunk features | 0.8148 | 0.8110 | 0.8130 | **+0.0020** | 0.8148 | 0.8166 | **+0.0017** |
| all, trunk + raw observation | 0.8138 | 0.8096 | 0.8110 | **+0.0015** | 0.8138 | 0.8150 | **+0.0012** |
| setup, trunk | 0.6092 | 0.5974 | 0.6360 | +0.0386 | 0.6092 | 0.6391 | +0.0300 |
| setup, raw | 0.5617 | 0.5700 | 0.5789 | +0.0088 | 0.5617 | 0.5810 | +0.0193 |
| headline, trunk | 0.8287 | 0.8191 | 0.8249 | +0.0058 | 0.8287 | 0.8333 | +0.0046 |
| action round, trunk | 0.8186 | 0.8153 | 0.8165 | +0.0012 | 0.8186 | 0.8197 | +0.0010 |
| turn 1, trunk | 0.6544 | 0.6479 | 0.6556 | +0.0076 | 0.6544 | 0.6604 | +0.0060 |
| turn 8–10, trunk | 0.9117 | 0.9065 | 0.9080 | +0.0015 | 0.9117 | 0.9130 | +0.0014 |

Explained variance (1 − Brier / base-rate Brier), all decisions: critic 0.293; public MLP
0.286 → 0.290 with the hand (+0.003), linear 0.291 → 0.295 (+0.004). The full slice tables are in
the reports.

## Reading

* **Yes, but by almost nothing: +0.0015 to +0.002 AUC overall** (0.815 → 0.817 on the critic's own
  logit), +0.3 points of explained variance. That repeats the E5 sizing (+0.003,
  [`../plans/P5_oracle_critic.md`](../plans/P5_oracle_critic.md) *Sizing*) on a model two engine
  revisions and about 900 Elo later.
* **The raw observation does not change it.** So the small gain is not an artifact of the trunk
  having dropped state: the opponent's hand carries little about the result beyond what the public
  position already does.
* **The only sizeable gain is at setup** (+0.01 to +0.04 AUC), where the critic knows least
  (AUC 0.56–0.61). But setup is 3% of decisions. The setup slice also varies between the two
  collections (critic 0.61 against 0.56), so it is noisy at this sample size.
* **As a variance reducer for training, an oracle critic is not worth building.** Its target would
  differ from the honest critic's by about 0.002 AUC.
* **Limits.** The fitted heads see about 16,000 games' worth of labels and stay below the soup's own
  critic (0.811 against 0.815), so an oracle trained end to end in RL could learn somewhat more.
  But the linear fit on the critic's logit gains the same +0.002 as the MLP, which suggests the
  hand's contribution is small and mostly additive rather than hidden in interactions.
