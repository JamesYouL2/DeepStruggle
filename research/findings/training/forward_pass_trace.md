# ColdWarNetV2, exactly: where a country's state goes

**Scope.** The trace of `ai/models/coldwar_net_v2.py::_encode` / `_policy_logits`, the network of
E3 and early E4. Every arm since P21 runs LadderNet (`--arch ladder`, `ai/models/ladder_net.py`;
the A codes are [`../../architectures_and_recipes.md`](../../architectures_and_recipes.md)). The
observation-slot tables at the end apply to both. **Rewrite in place when the forward pass changes.**

## Conclusion

* **One shared transform per entity type.** The board `(B, 84, 26)` ⊕ country identity goes through
  one `Linear` (`board_fc`, `graph_layers 0`: no adjacency at all) or shared graph convolutions, to
  `h_board (B, 84, 64)`; cards likewise to `h_cards (B, 110, 64)`. There is **no card↔card
  attention**: cards meet only through symmetric pooling, so hand composition is representable only
  as pooled statistics.
* **`h_board` reaches the decision four ways:** (1) mean+max pooled into the trunk — position
  destroyed; (2) as keys/values of `cross_attn` (cards over countries, 4 heads), itself pooled over
  cards before the trunk; (3) `attn_readout`, disabled (0) in every arm; (4) `pe_country`, the only
  unpooled route from country *i*'s state to country *i*'s logit.
* **Pooling is the bottleneck.** A probe closes ~89–94% of the country-influence gap from the
  pre-pooling token and ~14% from the 512-float trunk
  ([`../../archive/E3_ladder/findings/architecture.md`](../../archive/E3_ladder/findings/architecture.md)).
  `pe_country` routes around the pool; its correction is **added** to the dense head
  (`logits = policy_head(h) + [pe_card | 0 | pe_country | 0]`, final layer zero-initialised).
  Making it a replacement cost 219 Elo — a 64-float projection became the only path from the trunk.
* **`pe_card` mostly addresses a card *type*.** 110 cards have 46 static signatures (15 unique,
  groups up to 6: The Voice of America / Colonial Rear Guards / Grain Sales are one); a card's dynamic
  slots are only its location and the active marker, so same-location cards in a group get identical
  corrections. Countries collide on static slots too (54 of 84) but differ in rich dynamic slots.
  Identity in the input plus per-entity heads in the output is why the two come as a bundle
  ([`country_identity_without_graph_conv.md`](country_identity_without_graph_conv.md)).
* **Card and country actions are never legal at the same decision** (0.00% of 25,600; card block only
  25.1%, country only 52.9%, neither 22.0%), so no explanation that trades one off against the other
  can be right.
* **Naming traps:** `per_entity_heads` is the width of a residual MLP, not attention; `num_attn_heads`
  (default 4, never varied) governs the real attention; pooling happens at any `graph_layers`.

## The per-entity observation slots (layout v2.3, `engine/src/observation.cpp`)

**26 per country** — eleven constant (3, 4, 10–18), 924 of the 2,184 board floats:

| slot | meaning |
|---:|:---|
| 0, 1 | my / opponent influence ÷ 10 |
| 2 | net realignment modifier ÷ 5 (controlled neighbours, majority, superpower adjacency) |
| 3 | stability ÷ 5 **(constant)** |
| 4 | battleground **(constant)** |
| 5, 6 | controlled by me / by opponent |
| 7 | couping here would take DEFCON to 1 and lose outright |
| 8, 9 | superpower-adjacent to me / to opponent |
| 10–15 | region one-hot **(constant)** |
| 16–18 | in Western Europe / Eastern Europe / Southeast Asia **(constant)** |
| 19, 20 | I / opponent can place influence here |
| 21, 22 | I / opponent can coup here (no adjacency test) |
| 23 | node count ÷ 5 — times acted on in the current op |
| 24, 25 | influence deficit to my / opponent control ÷ 5, capped at 2.0 |

**14 per card** — four constant (8, 10, 11, 12), 440 of the 1,540 card floats:

| slot | meaning |
|---:|:---|
| 0–7 | location one-hot: deck-or-hidden, my hand, known opponent hand, discard, removed, ongoing, peeked, not in game |
| 8 | Ops ÷ 4 **(constant)** |
| 9 | side relative to me: +1 mine, −1 opponent, 0 neutral |
| 10 | era ÷ 2 **(constant)** |
| 11 | one-time **(constant)** |
| 12 | scoring card **(constant)** |
| 13 | active-card marker: 1.0 this decision, 0.6 suspended below it, 0.3 committed next |

These are engineered features (control, coup hazard, legality, deficits are precomputed), so a
shared `Linear(26 → d)` in front of them compresses a designed feature set rather than discovering
structure — why P21's M2 became a per-entity lookup on the raw slots.
