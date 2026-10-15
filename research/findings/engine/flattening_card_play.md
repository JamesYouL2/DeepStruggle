# Flattening card play, and un-aliasing `INFLUENCE` (landed in P17, engine E4)

**Conclusion.** The old card play was a chain of three shallow decisions — play mode → timing branch
(opponent card) → op mode — averaging **2.1, 2.0 and 2.9** legal actions and taking **35.6% of all
decisions** (400 self-play games, 176,122 decisions, 440.3 per game). Merging them removes 12.6% of
decisions (play mode + op mode) or **17.5%** with the timing branch, without widening the action
space. P17 did it: `SELECT_PLAY_MODE` now carries one `Resolution` (EVENT / SPACE / OPS_INFLUENCE /
OPS_COUP / OPS_REALIGN; `engine/include/ts/types.hpp`), `CHOOSE_TIMING_BRANCH` is retired, and
`SELECT_OP_MODE` survives only for the deferred Ops after an event-first opponent event. The E4
ladder reset paid the cost (every checkpoint and `(seed, actions)` dataset invalidated).

**Why it was worth it.** Those nodes carried almost no choice; a 64-simulation search agreed with
the policy 96.6% at `SELECT_PLAY_MODE` and 95.7% at the timing branch
([`../../archive/E3_ladder/log/P15_X4a_where_the_search_signal_is.md`](../../archive/E3_ladder/log/P15_X4a_where_the_search_signal_is.md)),
and every card play's credit path got shorter. The risk named in advance: the merged mask needs
lookahead (does a coup target exist, does DEFCON allow it) and an error there is an illegal action
accepted silently.

**`INFLUENCE` had two meanings.** At a `SELECT_OP_MODE` node with Ops granted free by Junta or Tear
Down This Wall, index 116 meant "decline the free action", not "place influence". Three engine bugs
came from that overloading (ts-replayer games 141, 146, 105), and a policy had to infer which sense
applied from context. P17 removed the alias: when the free action bars influence, `INFLUENCE` is
simply not legal (`engine/src/action_mask.cpp`).

Full analysis: [`../../archive/E3_ladder/plans/P17_action_representation.md`](../../archive/E3_ladder/plans/P17_action_representation.md).
