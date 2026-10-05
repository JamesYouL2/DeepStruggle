#include "ts/card_effects.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>

#include "ts/action_mask.hpp"
#include "ts/card_data.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/observation.hpp"
#include "ts/ops.hpp"
#include "ts/prng.hpp"
#include "ts/scoring.hpp"

// A port of ai/training/card_event_targets.py (label, t2, t3_t4, resolve_choices, _score). The
// Python module stays the reference: tests/bindings/test_card_effects.py holds the two equal on
// positions from real games. Where the Python reads a quantity off the observation, this reads
// the same slot of the same observation, so the two cannot disagree about what a slot means.

namespace ts::card_effects {

static_assert(WIDTH == obs_features::CARD_EFFECTS_WIDTH, "the block width is declared in two places");

const float SCALE[N_T2 + N_T4] = {
    0.989f, 9.794f, 2.960f, 0.408f, 0.382f,
    3.996f, 0.350f, 1.349f, 1.007f, 0.896f, 1.080f, 0.823f, 0.565f, 0.630f, 0.508f, 1.592f, 1.674f,
};

namespace {

constexpr uint8_t CHINA = 6;
constexpr size_t BOARD_W = V23_BOARD_FEATURES;
constexpr size_t CARD_W = card_slots::V23_FEATURES;

// `_score`'s weights: a won or lost game, VP, the regional scoring margins, battleground balance,
// influence balance (card_event_targets.SCORE_WEIGHTS).
constexpr float W_END = 100.0f, W_VP = 1.0f, W_MARGIN = 0.5f, W_BG = 1.0f, W_INF = 0.2f;

float sign_of(Player p) noexcept { return p == Player::US ? 1.0f : -1.0f; }

// `_summary`: the regional margins, battlegrounds controlled by each side and total influence of
// each side, from `p`'s side -- computed from the state exactly as the observation computes the
// slots the Python reads (global_features[64..69] x 20, board slots 4/5/6, 0/1 x 10).
struct Summary {
    float margin[6];
    float bg_mine, bg_theirs, inf_mine, inf_theirs;
};

Summary summary(const GameState& s, Player p) noexcept {
    Summary out{};
    const Player opp = get_opponent(p);
    for (size_t r = 0; r < 6; ++r) {
        const auto rs = Scoring::evaluate_region(s, static_cast<Region>(r));
        int net = rs.net_delta;
        if (static_cast<Region>(r) == Region::EUROPE) {   // observation.cpp: Europe Control is a win
            if (rs.us_status == RegionalStatus::CONTROL) net = 20;
            else if (rs.ussr_status == RegionalStatus::CONTROL) net = -20;
        }
        const float mine = (p == Player::US) ? static_cast<float>(net) : -static_cast<float>(net);
        out.margin[r] = std::clamp(mine / 20.0f, -1.0f, 1.0f) * 20.0f;
    }
    for (uint8_t i = 0; i < 84; ++i) {
        const auto& ci = MapData::get_country(i);
        const auto& c = s.countries[i];
        const int mine = (p == Player::US) ? c.us_influence : c.ussr_influence;
        const int theirs = (p == Player::US) ? c.ussr_influence : c.us_influence;
        out.inf_mine += static_cast<float>(mine);
        out.inf_theirs += static_cast<float>(theirs);
        if (ci.battleground) {
            const Player ctrl = Scoring::get_country_control(s, i);
            if (ctrl == p) out.bg_mine += 1.0f;
            else if (ctrl == opp) out.bg_theirs += 1.0f;
        }
    }
    return out;
}

float score(const GameState& s, Player p) noexcept {
    const float sign = sign_of(p);
    if (Engine::is_terminal(s)) {
        const float u = Engine::get_terminal_utility(s) * sign;
        return W_END * (u > 0.0f ? 1.0f : (u < 0.0f ? -1.0f : 0.0f));
    }
    const Summary m = summary(s, p);
    float margins = 0.0f;
    for (float v : m.margin) margins += v;
    return W_VP * sign * static_cast<float>(s.victory_points) + W_MARGIN * margins
         + W_BG * (m.bg_mine - m.bg_theirs) + W_INF * (m.inf_mine - m.inf_theirs);
}

// `_active`: card slot ACTIVE_CARD of `card` in the deciding player's observation. Called only
// with a decision player, where the Python's -1 for none never arises.
float active(const GameState& s, uint8_t card) noexcept {
    return Observation::active_card(s, card);
}

// `resolve_choices`: play out an event that stopped for choices, in place.
bool resolve_choices(GameState& x, uint8_t card) noexcept {
    std::array<uint8_t, FLAT_ACTION_SPACE_SIZE> mask{};
    std::array<uint16_t, FLAT_ACTION_SPACE_SIZE> legal{};
    for (size_t step = 0; step < MAX_EVENT_STEPS; ++step) {
        if (Engine::is_terminal(x)) return true;
        const Player p = x.ctx().decision_player;
        Engine::get_flat_action_mask(x, mask.data(), false);
        size_t n = 0;
        for (size_t a = 0; a < FLAT_ACTION_SPACE_SIZE; ++a)
            if (mask[a]) legal[n++] = static_cast<uint16_t>(a);
        if (n == 0) return false;
        if (p == Player::NONE) {                       // a die roll: the engine's own generator
            bool ok = false;
            for (size_t k = 0; k < n && !ok; ++k) ok = Engine::step_flat(x, legal[k], true, false);
            if (!ok) return false;
            continue;
        }
        if (active(x, card) < 0.99f) return true;
        bool have = false;
        uint16_t best_a = 0;
        float best_v = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            GameState y = x;
            if (!Engine::step_flat(y, legal[k], true, false)) continue;
            const float v = (n > 1) ? score(y, p) : 0.0f;
            if (!have || v > best_v) { have = true; best_a = legal[k]; best_v = v; }
        }
        if (!have || !Engine::step_flat(x, best_a, true, false)) return false;
    }
    return false;
}

// `ops_to_control`: Ops to bring a country under control, one point at a time, two a point while
// the opponent controls it.
int ops_to_control(int mine, int theirs, int stability) noexcept {
    int ops = 0;
    while (mine - theirs < stability) {
        ops += (theirs - mine >= stability) ? 2 : 1;
        ++mine;
    }
    return ops;
}

// `coup_roll_mod`: Latin American Death Squads in Central and South America, then SALT.
int coup_roll_mod(const GameState& s, Player p, uint8_t country) noexcept {
    int mod = 0;
    const Region r = MapData::get_country(country).region;
    if (r == Region::CENTRAL_AMERICA || r == Region::SOUTH_AMERICA) {
        const bool us = (p == Player::US);
        if (s.has_flag(effect_bits::DEATH_SQUADS_US)) mod += us ? 1 : -1;
        else if (s.has_flag(effect_bits::DEATH_SQUADS_USSR)) mod += us ? -1 : 1;
    }
    if (s.has_flag(effect_bits::SALT_ACTIVE)) mod -= 1;
    return mod;
}

// `t2`, reading the mover's observation `ob`.
void ops_reach(const GameState& s, Player mover, const ObservationBufferV23& ob, uint8_t card,
               float* out) noexcept {
    if (CardData::get_card(card).ops == 0) return;
    const bool cmc = s.has_flag(mover == Player::USSR ? effect_bits::CMC_ACTIVE_US
                                                     : effect_bits::CMC_ACTIVE_USSR);
    int reach = 0, reach_bg = 0;
    float best = 0.0f, best_bg = 0.0f;
    bool any_coup = false, any_bg_coup = false;
    for (uint8_t c = 0; c < 84; ++c) {
        const float* b = &ob.board_features[c * BOARD_W];
        const bool can_place = b[19] > 0.5f;
        const bool can_coup = b[21] > 0.5f && !(b[7] > 0.5f) && !cmc;
        if (!can_place && !can_coup) continue;
        const auto& ci = MapData::get_country(c);
        const int ops_c = Operations::get_effective_ops_in(s, card, mover, c);
        if (can_place && !(b[5] > 0.5f)) {
            const int mine = static_cast<int>(std::lround(b[0] * 10.0f));
            const int theirs = static_cast<int>(std::lround(b[1] * 10.0f));
            if (ops_to_control(mine, theirs, ci.stability) <= ops_c) {
                ++reach;
                if (ci.battleground) ++reach_bg;
            }
        }
        if (can_coup) {
            const int need = 2 * static_cast<int>(ci.stability) - ops_c - coup_roll_mod(s, mover, c);
            const float p = static_cast<float>(std::clamp(6 - need, 0, 6)) / 6.0f;
            best = any_coup ? std::max(best, p) : p;
            any_coup = true;
            if (ci.battleground) {
                best_bg = any_bg_coup ? std::max(best_bg, p) : p;
                any_bg_coup = true;
            }
        }
    }
    out[0] = static_cast<float>(Operations::get_effective_ops(s, card, mover));
    out[1] = static_cast<float>(reach);
    out[2] = static_cast<float>(reach_bg);
    out[3] = best;
    out[4] = best_bg;
}

struct Ctx { DecisionType t; Player p; uint8_t op_card; uint8_t steps; };
Ctx ctx_of(const GameState& s) noexcept {
    const auto& c = s.ctx();
    return {c.decision_type, c.decision_player, c.pending_op_card, static_cast<uint8_t>(c.remaining_steps)};
}
bool same(const Ctx& a, const Ctx& b) noexcept {
    return a.t == b.t && a.p == b.p && a.op_card == b.op_card && a.steps == b.steps;
}

// `t3_t4`: whether the event can fire (written to out[0]) and, if it can and finishes, the mean
// outcome over `seeds` (out[T4_AT..]).
void event_outcome(const GameState& s, Player mover, uint8_t card, const Summary& base,
                   const uint64_t* seeds, size_t n_seeds, float* out) noexcept {
    if (card == CHINA) return;
    const Player side = CardData::get_card(card).side;
    const Player fire = (side == Player::NONE) ? mover : side;
    if (!CardHandlers::can_trigger_event(s, card, fire)) return;
    out[0] = 1.0f;
    const float sign = sign_of(mover);
    const Ctx c0 = ctx_of(s);
    float acc[N_T4] = {};
    size_t runs = 0;
    for (size_t k = 0; k < n_seeds; ++k) {
        GameState x = s;
        x.rng_state = seeds[k];
        const int vp0 = x.victory_points, dc0 = x.defcon;
        CardHandlers::trigger_event(x, card, fire, 0);
        const Ctx c1 = ctx_of(x);
        if (!same(c1, c0) && c1.t != DecisionType::NONE && !resolve_choices(x, card)) return;
        const Summary a = summary(x, mover);
        float d[N_T4];
        d[0] = sign * static_cast<float>(x.victory_points - vp0);
        d[1] = static_cast<float>(x.defcon - dc0);
        for (size_t r = 0; r < 6; ++r) d[2 + r] = a.margin[r] - base.margin[r];
        d[8] = a.bg_mine - base.bg_mine;
        d[9] = a.bg_theirs - base.bg_theirs;
        d[10] = a.inf_mine - base.inf_mine;
        d[11] = a.inf_theirs - base.inf_theirs;
        for (size_t q = 0; q < N_T4; ++q) acc[q] += d[q];
        ++runs;
        // Nothing drawn from the generator: every seed would give this same outcome.
        if (x.rng_state == seeds[k]) break;
    }
    for (size_t q = 0; q < N_T4; ++q) out[T4_AT + q] = acc[q] / static_cast<float>(runs);
}

}  // namespace

size_t label(const GameState& state, Player mover, const uint64_t* seeds, size_t n_seeds,
             float* out) noexcept {
    std::memset(out, 0, WIDTH * sizeof(float));
    ObservationBufferV23 ob;
    Observation::extract(state, mover, &ob);
    const Summary base = summary(state, mover);
    size_t held = 0;
    for (uint8_t card = 1; card <= N_CARDS && held < MAX_HAND; ++card) {
        if (!(ob.card_features[(card - 1) * CARD_W + card_slots::MY_HAND] > 0.5f)) continue;
        ++held;
        float* row = out + (card - 1) * PER_CARD;
        ops_reach(state, mover, ob, card, row + T2_AT);
        if (n_seeds > 0) event_outcome(state, mover, card, base, seeds, n_seeds, row);
    }
    return held;
}

bool is_card_decision(const GameState& state, Player perspective) noexcept {
    const auto& c = state.ctx();
    if (perspective == Player::NONE || c.decision_player != perspective) return false;
    if (c.decision_type == DecisionType::SELECT_PLAY_MODE) return true;
    return c.decision_type == DecisionType::SELECT_CARD && c.resolving_card == 0;
}

GameState redeal(const GameState& state, Player me) noexcept {
    GameState out = state;
    if (state.current_phase == Phase::HEADLINE && state.headline_stage == 0) {
        if (me == Player::USSR && state.headline_us_card) out.headline_us_card = 0;
        else if (me == Player::US && state.headline_ussr_card) out.headline_ussr_card = 0;
    }
    const CardLocation opp_unknown = (me == Player::US) ? CardLocation::HAND_USSR_UNKNOWN
                                                        : CardLocation::HAND_US_UNKNOWN;
    std::array<uint8_t, N_CARDS> pool{};
    size_t n = 0, n_hand = 0;
    for (uint8_t c = 1; c <= N_CARDS; ++c) {
        const CardLocation loc = state.card_locations[c];
        if (loc == opp_unknown) { pool[n++] = c; ++n_hand; }
        else if (loc == CardLocation::DRAW_DECK) pool[n++] = c;
    }
    // Fisher-Yates from a generator seeded by the state, so the observation stays a pure
    // function of it; a fixed salt keeps this stream apart from the game's own dice.
    uint64_t g = state.rng_state ^ 0xC4D3EFFEC75A11ULL;
    for (size_t i = n; i > 1; --i) {
        const size_t j = Prng::random_index(g, static_cast<uint32_t>(i));
        std::swap(pool[i - 1], pool[j]);
    }
    for (size_t i = 0; i < n; ++i)
        out.card_locations[pool[i]] = (i < n_hand) ? opp_unknown : CardLocation::DRAW_DECK;
    return out;
}

void write_block(const GameState& state, Player perspective, float* out) noexcept {
    if (!is_card_decision(state, perspective)) {
        std::memset(out, 0, WIDTH * sizeof(float));
        return;
    }
    const GameState world = redeal(state, perspective);
    uint64_t g = state.rng_state ^ 0x5EED5D1CE0F7A11ULL;
    uint64_t seeds[N_BLOCK_SEEDS];
    for (auto& s : seeds) s = Prng::next_u64(g);
    label(world, perspective, seeds, N_BLOCK_SEEDS, out);
    for (size_t card = 0; card < N_CARDS; ++card) {
        float* row = out + card * PER_CARD;
        for (size_t q = 0; q < N_T2 + N_T4; ++q) row[1 + q] /= SCALE[q];
    }
}

} // namespace ts::card_effects
