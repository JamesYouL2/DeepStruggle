# Country identity without graph convolution: a per-country bias for 54 ambiguous countries

**Conclusion (2026-09-19, measured on `E3-30-28`: ColdWarNetV2 with `identity_dim 16,
per_entity_heads 64, graph_layers 0, self_transform`).** The static observation slots identify only
**30 of 84 countries** (46 distinct signatures; Egypt, Iran and Libya are identical, as are
Bulgaria, Czechoslovakia, Hungary and Yugoslavia), and with no graph layer nothing carries the map's
edges. The learned identity rows are the only thing that tells colliding countries apart, and the
trained network **depends on them heavily** — but they encode **no map structure**. Graph depth is
still open, not settled at 0.

## The evidence

* **A per-country bias the shared bias cannot absorb.** `pe_country` is one `Linear` over all 84
  countries, so its bias is one vector; identity turns it into `b + W_id · id_i`. The per-country
  offset is 5.15 in norm against the shared bias's 1.69, and varies across countries **0.80×** as
  much as the whole state-dependent input. Egypt / Iran / Libya get offsets 6–8 apart.
* **Identity is in the cross-attention keys.** Ablating it shifts more than half the card→country
  attention mass (total variation 0.56 zeroed, 0.69 randomised).
* **Ablation, 2,560 real positions:** zeroing country identity moves per-country logits by **1.10×**
  the whole across-country spread and the value by 57% of its SD; norm-matched random rows are as
  bad (1.15×), so the particular learned values matter. Rows trained (moved 32% of their norm) and
  stayed near-orthogonal.
* **No map in them:** a linear probe predicts battleground (R² 0.26) and region (0.19) no better than
  norm-matched random rows (0.31 / 0.22). It is a lookup table, not a map.
* **Naming trap:** `per_entity_heads` is the width of a per-country residual MLP, not attention; the
  attentional read-out `attn_readout` was 0 in these arms.

**What it does not establish:** dependence is not usefulness (an ablation leaves the training
distribution). `E4-03-01` vs `E4-04-01` differed in four flags, so it cannot isolate identity.

**Graph depth is open.** Identity *was* trained with graphs (`E3-15-22` at 2 layers, `E3-16-21` at
1); `graph_layers 0` won one seed and the sign reverses with the seed
([`../../archive/E3_ladder/findings/architecture.md`](../../archive/E3_ladder/findings/architecture.md),
[`../../archive/E3_ladder/findings/seed_variance.md`](../../archive/E3_ladder/findings/seed_variance.md)).
The never-trained cell is `identity_dim 0` with `graph_layers 0`.
