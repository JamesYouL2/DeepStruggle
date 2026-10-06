#pragma once
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

namespace ts {

enum class RegionalStatus : uint8_t {
    NONE        = 0,
    PRESENCE    = 1,
    DOMINATION  = 2,
    CONTROL     = 3
};

struct RegionScoreSummary {
    RegionalStatus us_status;
    RegionalStatus ussr_status;
    uint8_t        us_countries;
    uint8_t        ussr_countries;
    uint8_t        us_battlegrounds;
    uint8_t        ussr_battlegrounds;
    uint8_t        us_superpower_adjacent;
    uint8_t        ussr_superpower_adjacent;
    int16_t        us_score;
    int16_t        ussr_score;
    int16_t        net_delta; // us_score - ussr_score
};

class Scoring {
public:
    static Player get_country_control(const GameState& state, uint8_t country_id) noexcept;
    static bool is_controlled_by(const GameState& state, uint8_t country_id, Player p) noexcept;

    // Evaluates a region and returns detailed score breakdown
    static RegionScoreSummary evaluate_region(const GameState& state, Region r, bool is_final_scoring = false) noexcept;

    // The same as evaluate_region(state, r) -- a mid-game scoring -- with each country's control
    // read from `control` (84 entries, get_country_control's values) instead of recomputed. For
    // the observation, which needs control for every country anyway and scores all six regions.
    static RegionScoreSummary evaluate_region_with_control(const GameState& state, Region r,
                                                           const Player* control) noexcept;

    // Whether `p` Dominates or Controls `r` on the board as it stands, with no scoring-only card
    // effect applied (Summit's modifier).
    static bool dominates_or_controls(const GameState& state, Region r, Player p) noexcept;

    // Scores a region, updates state.victory_points (clamped to [-20, 20]), checks instant win
    static void score_region(GameState& state, Region r) noexcept;

    // Scores Southeast Asia (Card #38)
    static void score_southeast_asia(GameState& state) noexcept;

    // Evaluates military operations penalties at end of turn (Phase E)
    static void evaluate_military_ops(GameState& state) noexcept;

    // Conducts Final Scoring (Phase I at end of Turn 10)
    static void execute_final_scoring(GameState& state) noexcept;

    // Computes the strategic useful actions potential Phi(s, p) in [-1.0, 1.0]
    static float compute_useful_actions_potential(const GameState& state, Player p) noexcept;
};

} // namespace ts
