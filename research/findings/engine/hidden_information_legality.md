# The legal action *set* can depend on hidden information

**Conclusion (measured 2026-09-16).** The Cambridge Five (card 104) lets the USSR place in a region
named on a scoring card the **US** holds, so its legal targets depend on a hand the USSR cannot see.
A determinized search samples that hand, and different worlds then offer **different legal moves**
at the same node: 1 decision in 7,380 of random play, where the real state had 21 legal actions and
a sampled world 36 (all of Asia, because the sampled US hand held Asia Scoring). That breaks the
assumption every determinized search makes — that visit counts from all worlds sum over one common
set of root children — and a 2,400-game tournament died on it once `step` started raising.

**What the code does.** `BatchedMCTSAgent.select_actions_batch` (`ai/search/batched_mcts.py`) treats
the search as a proposal: a pick illegal in the real state is replaced by the greedy policy over the
real mask and counted in `world_mismatch_count`; a pick still illegal after that raises as a
misrooted tree. Building root children from the true mask instead would remove the substitution; it
is left until a search arm needs it.

**Other cards.** Only The Cambridge Five was found, by a random-play sweep that would miss rarer
cases. Any card whose *targets* are named by a hidden zone belongs here; `rules/cards.json` has not
been searched systematically.
