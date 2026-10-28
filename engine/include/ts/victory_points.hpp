#pragma once

#include <algorithm>
#include <cstdint>

#include "ts/game_state.hpp"
#include "ts/types.hpp"

namespace ts {

// Moves the score by `delta` (US positive), clamped to the track's [-20, 20]. Reaching either
// end is an automatic victory that takes effect at once, so the game ends here -- the one
// place the rule lives, instead of ~40 call sites that each clamped and (sometimes) remembered
// to end the game. Returns true when this award ended it.
//
// A finished game's score does not move again: an award that lands after a 20 VP win (Flower
// Power's penalty on the war that won it) would otherwise "un-win" the score while the phase
// stays GAME_OVER.
inline bool award_vp(GameState& state, int delta) noexcept {
    if (state.current_phase == Phase::GAME_OVER) return false;
    const int vp = std::clamp(static_cast<int>(state.victory_points) + delta, -20, 20);
    state.victory_points = static_cast<int8_t>(vp);
    if (vp >= 20 || vp <= -20) {
        state.current_phase = Phase::GAME_OVER;
        return true;
    }
    return false;
}

} // namespace ts
