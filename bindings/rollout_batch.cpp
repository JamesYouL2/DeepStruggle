#include "rollout_batch.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>

#include "ts/action_mask.hpp"
#include "ts/engine.hpp"
#include "ts/observation.hpp"
#include "ts/prng.hpp"

namespace ts::rollout {

namespace {

constexpr size_t A = FLAT_ACTION_SPACE_SIZE;

// Resolve the die rolls a position is sitting on, nothing else (bindings.settle's CHANCE depth).
void drain_chance(GameState& s) noexcept {
    while (!Engine::is_terminal(s) && s.ctx().decision_player == Player::NONE &&
           s.ctx().decision_type == DecisionType::ROLL_DIE) {
        if (!Engine::step(s, MicroAction{DecisionType::ROLL_DIE, 0, 0, 0})) return;
    }
}

// Does `a` end the game against `who` on the spot? (ai/eval/paired_playouts.loses_now)
bool loses_now(const GameState& s, uint16_t a, Player who) noexcept {
    GameState probe = s;
    if (!Engine::step_flat(probe, a)) return false;
    drain_chance(probe);
    if (!Engine::is_terminal(probe)) return false;
    const float u = Engine::get_terminal_utility(probe);
    return who == Player::US ? (u < 0.0f) : (u > 0.0f);
}

bool guarded_decision(const GameState& s) noexcept {
    if (s.defcon > 2) return false;
    const auto& ctx = s.ctx();
    switch (ctx.decision_type) {
        case DecisionType::SELECT_PLAY_MODE:
        case DecisionType::CHOOSE_BRANCH:
            return true;
        case DecisionType::POINT_NODE:
            return ctx.op_mode == OpMode::COUP;
        default:
            return false;
    }
}

float mover_value(const GameState& s, Player mover) noexcept {
    const float u = Engine::get_terminal_utility(s);
    return mover == Player::US ? u : -u;
}

void record_key(Row& r) noexcept {
    r.turn = r.state.turn;
    r.action_round = r.state.action_round;
    r.phasing = r.state.phasing_player;
    r.phase = r.state.current_phase;
}

bool key_changed(const Row& r) noexcept {
    return r.turn != r.state.turn || r.action_round != r.state.action_round ||
           r.phasing != r.state.phasing_player || r.phase != r.state.current_phase;
}

}  // namespace

Player acting_player(const GameState& s) noexcept {
    const Player d = s.ctx().decision_player;
    return d != Player::NONE ? d : s.phasing_player;
}

void determinize(GameState& s, Player me, uint64_t& rng) noexcept {
    if (s.current_phase == Phase::HEADLINE && s.headline_stage == 0) {
        if (me == Player::USSR && s.headline_us_card) s.headline_us_card = 0;
        else if (me == Player::US && s.headline_ussr_card) s.headline_ussr_card = 0;
    }
    const CardLocation opp_unknown =
        me == Player::US ? CardLocation::HAND_USSR_UNKNOWN : CardLocation::HAND_US_UNKNOWN;
    uint8_t pool[110];
    size_t n = 0, n_hand = 0;
    for (uint8_t cid = 1; cid <= 110; ++cid) {
        if (s.card_locations[cid] == opp_unknown) { pool[n++] = cid; ++n_hand; }
    }
    for (uint8_t cid = 1; cid <= 110; ++cid) {
        if (s.card_locations[cid] == CardLocation::DRAW_DECK) pool[n++] = cid;
    }
    if (n == 0) return;
    for (size_t i = n - 1; i > 0; --i) {                       // Fisher-Yates
        const uint32_t j = Prng::random_index(rng, static_cast<uint32_t>(i + 1));
        std::swap(pool[i], pool[j]);
    }
    for (size_t i = 0; i < n; ++i) {
        s.card_locations[pool[i]] = i < n_hand ? opp_unknown : CardLocation::DRAW_DECK;
    }
}

RolloutBatch::RolloutBatch(const Config& cfg, uint64_t seed) noexcept
    : cfg_(cfg), rng_(seed),
      obs_width_(OBS_SIZE_V23 + obs_features::extra_width(cfg.features)) {}

size_t RolloutBatch::live() const noexcept {
    size_t n = 0;
    for (const Row& r : rows_) n += r.done ? 0 : 1;
    return n;
}

size_t RolloutBatch::refused() const noexcept {
    size_t n = 0;
    for (const Row& r : rows_) n += r.refused ? 1 : 0;
    return n;
}

bool RolloutBatch::at_stop(const Row& r) const noexcept {
    const bool boundaries = r.crossed >= cfg_.horizon;
    const bool turn_over = r.state.turn != r.start_turn;
    switch (cfg_.stop) {
        case Stop::kTurnEnd: return turn_over;
        case Stop::kLater: return boundaries && turn_over;
        case Stop::kBoundaries: break;
    }
    return boundaries;
}

void RolloutBatch::settle_and_mark(Row& r) noexcept {
    Engine::auto_advance_step(r.state);
    const Player mover = movers_[r.position];
    if (Engine::is_terminal(r.state)) {
        r.done = r.terminal = true;
        r.value = mover_value(r.state, mover);
        return;
    }
    if (key_changed(r)) {
        ++r.crossed;
        record_key(r);
    }
    if (at_stop(r)) r.done = true;
}

size_t RolloutBatch::add(const GameState& base, const uint16_t* candidates, size_t k, size_t worlds) {
    error_ = "";
    uint8_t base_mask[A];
    ActionMask::generate_flat_mask_212(base, base_mask);
    for (size_t c = 0; c < k; ++c) {
        if (candidates[c] >= A || !base_mask[candidates[c]]) {
            error_ = "a candidate is not legal at the position";
            return 0;
        }
    }
    const uint32_t position = static_cast<uint32_t>(movers_.size());
    const Player mover = acting_player(base);
    movers_.push_back(mover);
    worlds_used_.push_back(0);
    size_t added = 0;
    for (size_t w = 0; w < worlds; ++w) {
        GameState world = base;
        if (!Engine::is_terminal(world)) determinize(world, mover, rng_);
        if (cfg_.shared_dice) world.rng_state = Prng::next_u64(rng_);
        uint8_t wm[A];
        ActionMask::generate_flat_mask_212(world, wm);
        bool all_legal = true;
        for (size_t c = 0; c < k; ++c) all_legal = all_legal && wm[candidates[c]];
        if (!all_legal) continue;
        const uint16_t world_index = static_cast<uint16_t>(worlds_used_[position]++);
        for (size_t c = 0; c < k; ++c) {
            Row r{};
            r.state = world;
            if (!cfg_.shared_dice) r.state.rng_state = Prng::next_u64(rng_);
            r.position = position;
            r.world = world_index;
            r.candidate = static_cast<uint16_t>(c);
            r.start_turn = base.turn;
            r.value = std::numeric_limits<float>::quiet_NaN();
            if (!Engine::step_flat(r.state, candidates[c])) {
                r.refused = r.done = true;
            } else {
                // The move's own settling is not a boundary: a candidate that ends the action
                // round settles into the next one, and the horizon counts from there.
                Engine::auto_advance_step(r.state);
                if (Engine::is_terminal(r.state)) {
                    r.done = r.terminal = true;
                    r.value = mover_value(r.state, mover);
                } else {
                    record_key(r);
                    r.crossed = 0;
                    // A move that ends the turn has already reached the next one: under kTurnEnd
                    // its row is a leaf at once.
                    r.done = at_stop(r);
                }
            }
            rows_.push_back(r);
            ++added;
        }
    }
    return added;
}

size_t RolloutBatch::fill_live(float* obs, uint8_t* masks) {
    live_order_.clear();
    size_t n = 0;
    for (uint32_t i = 0; i < rows_.size(); ++i) {
        Row& r = rows_[i];
        if (r.done) continue;
        extract_observation_features(r.state, acting_player(r.state), cfg_.features, obs + n * obs_width_);
        ActionMask::generate_flat_mask_212(r.state, masks + n * A);
        live_order_.push_back(i);
        ++n;
    }
    return n;
}

uint16_t RolloutBatch::guarded_choice(const GameState& s, const float* logits, const uint8_t* mask,
                                      uint16_t argmax) const noexcept {
    if (!cfg_.guard || !guarded_decision(s)) return argmax;
    const Player who = acting_player(s);
    // The top four legal moves by logit, best first.
    uint16_t top[4];
    size_t n = 0;
    for (size_t pass = 0; pass < 4; ++pass) {
        int best = -1;
        for (size_t a = 0; a < A; ++a) {
            if (!mask[a]) continue;
            bool taken = false;
            for (size_t t = 0; t < n; ++t) taken = taken || top[t] == a;
            if (taken) continue;
            if (best < 0 || logits[a] > logits[best]) best = static_cast<int>(a);
        }
        if (best < 0) break;
        top[n++] = static_cast<uint16_t>(best);
    }
    for (size_t t = 0; t < n; ++t) {
        if (!loses_now(s, top[t], who)) return top[t];
    }
    return argmax;
}

void RolloutBatch::step(const float* logits) {
    uint8_t mask[A];
    for (size_t j = 0; j < live_order_.size(); ++j) {
        Row& r = rows_[live_order_[j]];
        if (r.done) continue;
        const float* lg = logits + j * A;
        ActionMask::generate_flat_mask_212(r.state, mask);
        int best = -1;
        for (size_t a = 0; a < A; ++a) {
            if (mask[a] && (best < 0 || lg[a] > lg[best])) best = static_cast<int>(a);
        }
        if (best < 0) { r.refused = r.done = true; continue; }
        const uint16_t a = guarded_choice(r.state, lg, mask, static_cast<uint16_t>(best));
        if (!Engine::step_flat(r.state, a)) { r.refused = r.done = true; continue; }
        settle_and_mark(r);
    }
    live_order_.clear();
}

size_t RolloutBatch::fill_leaves(float* obs, uint8_t* masks) {
    leaf_order_.clear();
    size_t n = 0;
    for (uint32_t i = 0; i < rows_.size(); ++i) {
        Row& r = rows_[i];
        if (!r.done || r.terminal || r.refused || r.leaf_valued) continue;
        extract_observation_features(r.state, movers_[r.position], cfg_.features, obs + n * obs_width_);
        ActionMask::generate_flat_mask_212(r.state, masks + n * A);
        leaf_order_.push_back(i);
        ++n;
    }
    return n;
}

void RolloutBatch::set_leaf_values(const float* values) {
    for (size_t j = 0; j < leaf_order_.size(); ++j) {
        Row& r = rows_[leaf_order_[j]];
        r.value = values[j];
        r.leaf_valued = true;
    }
    leaf_order_.clear();
}

void RolloutBatch::values(float* out) const noexcept {
    for (size_t i = 0; i < rows_.size(); ++i) {
        const Row& r = rows_[i];
        out[i] = (r.terminal || r.leaf_valued) ? r.value : std::numeric_limits<float>::quiet_NaN();
    }
}

// ---- the root's choice ---------------------------------------------------------------------------

bool parse_rule(const char* text, Rule& out) noexcept {
    if (text == nullptr) return false;
    const std::string s(text);
    auto number = [](const std::string& t, float fallback, float& v) {
        if (t.empty()) { v = fallback; return true; }
        char* end = nullptr;
        const double d = std::strtod(t.c_str(), &end);
        if (end == t.c_str() || *end != '\0' || !(d > 0.0)) return false;
        v = static_cast<float>(d);
        return true;
    };
    if (s == "argmax") { out = Rule{Rule::ARGMAX, 0.0f}; return true; }
    if (s.rfind("kl", 0) == 0) {
        float v; if (!number(s.substr(2), 0.05f, v)) return false;
        out = Rule{Rule::KL, v}; return true;
    }
    if (s.rfind("z", 0) == 0) {
        float v; if (!number(s.substr(1), 2.0f, v)) return false;
        out = Rule{Rule::Z, v}; return true;
    }
    return false;
}

size_t top_candidates(const float* logits, const uint8_t* mask, size_t k, uint16_t* out) noexcept {
    size_t n = 0;
    for (size_t pass = 0; pass < k; ++pass) {
        int best = -1;
        for (size_t a = 0; a < A; ++a) {
            if (!mask[a]) continue;
            bool taken = false;
            for (size_t t = 0; t < n; ++t) taken = taken || out[t] == a;
            if (taken) continue;
            if (best < 0 || logits[a] > logits[best]) best = static_cast<int>(a);
        }
        if (best < 0) break;
        out[n++] = static_cast<uint16_t>(best);
    }
    return n;
}

Decision decide(const RolloutBatch& batch, size_t position, const uint16_t* candidates, size_t k,
                const float* logits, const uint8_t* mask, const Rule& rule) {
    Decision d;
    d.q.assign(k, 0.0f);
    d.lead.assign(k, 0.0f);
    d.lead_se.assign(k, 0.0f);
    d.pick = k ? candidates[0] : 0;
    if (k == 0) return d;
    const size_t worlds = position < batch.positions() ? batch.worlds_used(position) : 0;
    // The table: candidate x world, NaN where a row has no value.
    std::vector<float> table(k * worlds, std::numeric_limits<float>::quiet_NaN());
    for (size_t r = 0; r < batch.rows(); ++r) {
        const Row& row = batch.row(r);
        if (row.position != position || row.candidate >= k || row.world >= worlds) continue;
        if (row.terminal || row.leaf_valued) table[row.candidate * worlds + row.world] = row.value;
    }
    // Worlds every candidate was valued in.
    std::vector<size_t> ok;
    for (size_t w = 0; w < worlds; ++w) {
        bool all = true;
        for (size_t c = 0; c < k; ++c) all = all && !std::isnan(table[c * worlds + w]);
        if (all) ok.push_back(w);
    }
    d.worlds = ok.size();
    if (ok.empty()) return d;
    for (size_t c = 0; c < k; ++c) {
        double sum = 0.0;
        for (size_t w : ok) sum += table[c * worlds + w];
        d.q[c] = static_cast<float>(sum / static_cast<double>(ok.size()));
        double dsum = 0.0;
        for (size_t w : ok) dsum += table[c * worlds + w] - table[w];
        const double mean = dsum / static_cast<double>(ok.size());
        double var = 0.0;
        for (size_t w : ok) {
            const double x = (table[c * worlds + w] - table[w]) - mean;
            var += x * x;
        }
        d.lead[c] = static_cast<float>(mean);
        d.lead_se[c] = ok.size() > 1
            ? static_cast<float>(std::sqrt(var / static_cast<double>(ok.size() - 1) / static_cast<double>(ok.size())))
            : 0.0f;
    }
    size_t best = 0;
    for (size_t c = 1; c < k; ++c) if (d.q[c] > d.q[best]) best = c;
    switch (rule.kind) {
        case Rule::ARGMAX:
            d.pick = candidates[best];
            break;
        case Rule::Z: {
            const float m = d.lead[best], se = d.lead_se[best];
            const bool clear = best != 0 && m > 0.0f && (se == 0.0f || m / se >= rule.param);
            d.pick = clear ? candidates[best] : candidates[0];
            break;
        }
        case Rule::KL: {
            // log pi over the legal moves, from the root's logits.
            float mx = -std::numeric_limits<float>::infinity();
            for (size_t a = 0; a < A; ++a) if (mask[a]) mx = std::max(mx, logits[a]);
            double z = 0.0;
            for (size_t a = 0; a < A; ++a) if (mask[a]) z += std::exp(static_cast<double>(logits[a] - mx));
            const double lz = std::log(z);
            size_t arg = 0;
            double score = -std::numeric_limits<double>::infinity();
            for (size_t c = 0; c < k; ++c) {
                const double logp = static_cast<double>(logits[candidates[c]] - mx) - lz;
                const double s = logp + static_cast<double>(d.q[c]) / rule.param;
                if (s > score) { score = s; arg = c; }
            }
            d.pick = candidates[arg];
            break;
        }
    }
    return d;
}

}  // namespace ts::rollout
