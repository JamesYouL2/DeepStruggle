# A searching baseline that needs no checkpoint

**Conclusion (2026-09-19, E4 engine).** `HeuristicMCTSBot` (`bot/heuristic_mcts_bot.py`; spec
`heuristic_mcts` or `heuristic_mcts:<sims>`) — MCTS over the true `GameState` with a rules-derived
leaf value (`ai/search/heuristic_eval.py`) and a one-ply lookahead prior — beats `HeuristicBot`
**18–20 of 20** at 16 simulations, symmetric across seats, ~1.6 s a game at 96. With the tree
effectively off (1 simulation) it drops to 9/20 (1/10 as US). 96 and 256 simulations buy nothing:
the sharp one-ply prior (`prior_temperature 0.25`) is already right, so **vary `prior_temperature`,
not `simulations`, for a difficulty dial**. n = 20 per row (SE ~7pp): read only "1 sim ≪ 16".

**The evaluation** (US perspective, weights in `HeuristicWeights`, chosen from the rules, not
fitted): victory points; region standing (`Scoring.evaluate_region`'s `net_delta`) × 1.0 while that
region's scoring card is live and × 0.5 once discarded — the owner's improvement over a fixed region
table; 0.2 per controlled battleground, 0.1 per battleground with access; held scoring cards
penalised on a countdown to turn end. No hand contents (unreadable in the deployable case). Play
rules sit in the prior, not the value: spacing an own/neutral card is a prior penalty reusing
`dominance.space_dominance_outcome`, the predicate `ai/eval/blunders.py` measures.

**Caveats.** It is a perfect-information opponent, so a win rate against it says nothing about play
under the real information set. Built on `PIMCTSAgent` with an optional `leaf_fn`; note that
`PIMCTSAgent.search` drains chance nodes on its root clone, so a caller sitting on an unresolved die
roll gets an action legal only at the next decision — `HeuristicMCTSBot` answers chance nodes
directly.
