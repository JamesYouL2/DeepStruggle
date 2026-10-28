#pragma once

#include <cstdint>

#include "ts/game_state.hpp"
#include "ts/invariant_failed.hpp"

namespace ts {

// A state that should not arise but can be continued from. Unlike invariant_failed this
// returns, because a training run that aborts on one rare board is worse than one that logs it
// and plays on -- provided the log is loud enough that nobody misses it, and complete enough
// to replay the board exactly.
//
// The whole GameState is dumped as hex. It is trivially copyable and under 4 KB, so those
// bytes are the position itself and can be restored with a memcpy; Serializer::to_json is a
// summary and would not be enough to reproduce the decision. Written to stderr unbuffered so
// it survives a training run that is watching stdout, and repeated for every distinct state so
// a loop cannot hide behind a single report -- an identical state seen twice in a row is
// reported once, since a mask may be generated several times for one decision.
void report_anomaly(const char* what, const GameState& state) noexcept;

// How many anomalies this thread has reported so far. A caller that must not continue past
// one -- the human-replay converter, where an anomaly means the log and the engine disagree --
// reads it before and after a step and fails on any increase.
uint64_t anomalies_reported() noexcept;

} // namespace ts
