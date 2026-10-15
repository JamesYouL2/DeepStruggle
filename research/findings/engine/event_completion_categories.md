# How completely an event must be executed — three categories, not a boolean

**Conclusion (implemented in P17 §6a).** An event that opens a `POINT_NODE` needs two independent
facts: may the player stop early (`ctx.allow_early_stop`), and is a short or empty target pool news
(`may_fizzle::allowed`, `engine/include/ts/constants.hpp`). Together they give three categories:

| category | early stop | short/empty pool | card text |
|:---|:---|:---|:---|
| **PARTIAL_OK** | legal — declining is a move | legal, quiet | "may", "up to N" |
| **BEST_EFFORT** | **not** legal | legal, quiet | exact count against a pool that can run dry |
| **MUST_COMPLETE** | **not** legal | a defect — `report_anomaly` | exact count against a pool that cannot |

`may_fizzle::allowed` is exactly BEST_EFFORT: **ten cards** — Truman Doctrine, Suez Crisis, East
European Unrest, Muslim Revolution, and the six added in §6a (Socialist Governments, Comecon,
Marshall Plan, The Voice of America, Marine Barracks Bombing, Pershing II Deployed). A card
qualifies only if `remaining_steps` is the literal count (not clamped to the live pool) **and** its
mask predicate depends on the board.

## Why it is recorded

* **The two fixes are coupled.** `action_mask.cpp` reaches the anomaly branch only when
  `allow_early_stop == 0`. The six missing cards had `allow_early_stop = 1`, which shielded them;
  flipping the flag without extending the list in the same change would have turned six silent
  cards into six false anomaly reports.
* **Clamping is the structurally better shape.** Warsaw Pact counts the eligible pool and sets
  `remaining_steps = min(4, count)`, so it can never be short and needs no list entry. The cost is
  per-card code; the cost of a list is that it goes stale, which is how six cards went missing.
* **The early-stop misclassification was wider than the whitelist.** Six more cards (Colonial Rear
  Guards, Decolonization, OAS Founded, Liberation Theology, Ussuri River Skirmish, The Reformer) set
  early stop with no "may" in their text but have pools that cannot run short, so they needed the
  flag flip and no list entry.

Classified over all 110 cards in `rules/cards.json` against the handlers; feeds §6a of
[`../../archive/E3_ladder/plans/P17_action_representation.md`](../../archive/E3_ladder/plans/P17_action_representation.md).
