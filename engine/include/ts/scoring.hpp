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
    // The USSR's counts as this scoring counted them: the board's above, less the battleground
    // (its country, and in Asia the enemy-superpower adjacency) Shuttle Diplomacy takes away.
    // Equal to the board's whenever the card does not apply.
    uint8_t        ussr_countries_scored;
    uint8_t        ussr_battlegrounds_scored;
    uint8_t        ussr_superpower_adjacent_scored;
    bool           shuttle_diplomacy_applied;
};

// Every regional scoring on a thread, reported as it happens, for P34's playout labeller
// (bindings/playout_labels.*), which labels a position with each region's swing at its next
// scoring. Inert unless a caller points `scoring_tap` at one, which the engine itself never does:
// the score, the state and every game are the same with or without it. Only the first scoring of
// each region after a reset is kept. Slot = Region (Europe 0 .. South America 5), 6 = Southeast Asia.
struct ScoringTap {
    static constexpr uint8_t kSlots = 7;
    static constexpr uint8_t kSoutheastAsia = 6;
    uint8_t scored = 0;           // bit per slot: scored since the reset
    uint8_t final_scoring = 0;    // 1 once Final Scoring ran
    int8_t  vp_before_final = 0;  // the VP track as Final Scoring began
    int16_t swing[kSlots] = {};   // the slot's first scoring: US points minus USSR points, unclamped
                                  // (a Europe Control win reads +/-20)

    void report(uint8_t slot, int16_t delta) noexcept {
        const uint8_t bit = static_cast<uint8_t>(1u << slot);
        if (scored & bit) return;
        scored |= bit;
        swing[slot] = delta;
    }
};
inline thread_local ScoringTap* scoring_tap = nullptr;

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
