# The comprehensive tournament on E7 (2026-09-30): the shared-start soup still leads; shallow SWAs sit just under the deep league SWAs

The owner asked for the current best model in one field, with no cross-field inference. 20
models on the E7 engine, greedy, 1,000 games per seat per pairing, anchored at HeuristicBot = 1500
(`data/reports/e7_comprehensive.{md,json}`). E7 plays like E6 for these checkpoints
([`../findings/engine/engine_revisions.md`](../findings/engine/engine_revisions.md), E7-01-44).

| Elo | model | what it is |
|---:|:---|:---|
| **2578** | shared-start soup of the three 680–760M SWAs | model soup, E6-06/07/08-44 |
| **2576** | shared-start soup of the three 760M snapshots | model soup, E6-06/07/08-44 (the designated best model soup) |
| 2534 | E6-06-44 SWA 680–760M | deep + league, continued to 760M (the designated best SWA) |
| 2525 | E6-08-44 SWA 680–760M | the same, with EMA snapshots |
| 2518 | E6-04-44 SWA 480–560M | deep + league, seed 44 |
| 2498 | E6-04-43 SWA 480–560M | deep + league, seed 43 |
| 2496 | E6-12-43 SWA 480–560M | shallow, seed 43 |
| 2496 | E6-13-44 SWA 480–560M | shallow + league, seed 44 |
| 2479 | E6-12-44 SWA 480–560M | shallow, seed 44 |
| 2471 | E6-07-44@700M | the designated best raw snapshot |
| 2468 | E7-01-44 SWA 480–560M | shallow, seed 44, trained on E7 |
| 2449 / 2443 / 2429 / 2415 | E6-12-43 / E6-13-44 / E6-12-44 / E7-01-44 @560M | shallow snapshots |
| 2409 | E6-04-44@560M | deep + league snapshot |
| 2387 / 2247 / 2090 | E6-03-44 @550M / @240M / @80M | the panel |

The raw soup against each of the top models:

| against | soup's win rate |
|:---|---:|
| the soup of SWAs | 48.8% |
| E6-06-44 SWA | 55.2% |
| E6-08-44 SWA | 56.2% |
| E6-04-44 SWA | 59.2% |
| E6-04-43 SWA, E6-12-43 SWA, E6-13-44 SWA | 60.6–60.7% |
| E6-12-44 SWA | 62.5% |
| E6-07-44@700M | 64.4% |
| E7-01-44 SWA | 65.0% |

## Reading

* **The best model is unchanged:** the shared-start model soup. It leads the best SWA by about 42 Elo and
  beats it 55%. Souping the SWAs instead of the snapshots is level (48.8% / 51.2%).
* **Training past 560M still pays on the deep line.** The 760M SWAs (2525–2534) are above every
  560M model.
* **At 560M the shallow SWAs sit just under the deep league SWAs.**
  * Seed 43: E6-12-43, shallow without a league, is level with E6-04-43, deep with a league
    (2496 against 2498).
  * Seed 44: E6-12-44 is 39 below E6-04-44 (2479 against 2518). Adding the league to the shallow
    trunk (E6-13-44, 2496) closes half of that.
* **Among raw 560M snapshots the shallow nets lead** (2415–2449 against E6-04-44@560M's 2409), as in
  every earlier comparison. The deep net gains more from averaging.
* **Open:** nothing shallow has been trained past 560M or souped.
