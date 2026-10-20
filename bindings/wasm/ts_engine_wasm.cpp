// The engine for the browser workbench: a flat C API compiled to WebAssembly by Emscripten.
//
// The page drives one game at a time through this, plays models with onnxruntime-web, and keeps
// its own undo stack of raw GameState bytes (the state is trivially copyable). Everything it
// shows about a position comes from the same C++ as the Python bindings -- the display state and
// the save from state_json.cpp, the rules from the engine -- so the browser and the Python stack
// cannot disagree about a position, and a position saved in one opens in the other.
//
// Strings returned here live until the next call that returns one; the JS wrapper copies them.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "belief.hpp"
#include "rollout_batch.hpp"
#include "selftest.hpp"
#include "state_json.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_data.hpp"
#include "ts/engine.hpp"
#include "ts/map_data.hpp"
#include "ts/game_state.hpp"
#include "ts/observation.hpp"
#include "ts/prng.hpp"
#include "ts/state_machine.hpp"

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#define TS_API extern "C" EMSCRIPTEN_KEEPALIVE
#else
#define TS_API extern "C"
#endif

#ifndef TS_ENGINE_FINGERPRINT
#define TS_ENGINE_FINGERPRINT "unknown"
#endif

namespace {

namespace sj = ts::state_json;

ts::GameState g_state{};
// What the last click itself did: the state after its action, or after its last die roll if it
// rolled one -- before the chance nodes that follow without a roll. The turn's cleanup is one: an
// action round that ends the turn stops on it, and resolving it (final VP, the military-ops reset,
// DEFCON, the next deal) is an engine step, and every engine step clears the roll record -- so read
// from the end state, the realignment or coup that ended the turn had no dice in the page's log, and
// the turn's end read as part of it. The page describes the click from here and the rest apart.
// Display only: the game is g_state, stepped exactly as before.
ts::GameState g_settled{};
// Every die the last click rolled, in order, as the display state writes a roll ("die_roll"). One
// click can roll more than once -- a realignment with the opponent's war card, whose event fires in
// the same click and rolls too -- and each engine step keeps only its own roll.
std::vector<sj::Value> g_rolls;
// Every Event's change to VP and DEFCON over the same steps as g_rolls: the second headline commit
// resolves both headlines in one step, and the click's before/after shows only their sum.
std::vector<sj::Value> g_event_tracks;
uint8_t g_mask[ts::FLAT_ACTION_SPACE_SIZE];
float g_obs[ts::OBS_SIZE_V23 + ts::obs_features::extra_width(ts::obs_features::ALL)];
int32_t g_decoded[4];
std::string g_text;
std::string g_error;

const char* keep(std::string s) {
    g_text = std::move(s);
    return g_text.c_str();
}

// Resolve the chance nodes the game is sitting on, as tools.lib.game_step.drain_chance does:
// `forced_die` 0 rolls from the state's RNG, 1..6 forces the face (the workbench's manual die).
// The roll a state's last step made, as the display state writes it.
void keep_roll(const ts::GameState& s) {
    if (s.last_roll.type == ts::RollType::NONE) return;
    const sj::Value display = sj::display_state(s);   // held: find() points into it
    if (const sj::Value* r = display.find("die_roll")) g_rolls.push_back(*r);
}

// The Events a state's last step recorded, as the display state writes them.
void keep_event_tracks(const ts::GameState& s) {
    if (s.last_event_track_count == 0) return;
    const sj::Value display = sj::display_state(s);   // held: find() points into it
    if (const sj::Value* t = display.find("event_tracks")) {
        for (const sj::Value& e : t->items()) g_event_tracks.push_back(e);
    }
}

// `settled`, if given, takes the state after each step that rolled a die (see g_settled), and that
// step's roll joins g_rolls.
bool drain(ts::GameState& s, int forced_die, ts::GameState* settled = nullptr) {
    while (!ts::Engine::is_terminal(s) && s.ctx().decision_player == ts::Player::NONE &&
           s.ctx().decision_type == ts::DecisionType::ROLL_DIE) {
        const ts::MicroAction roll{ts::DecisionType::ROLL_DIE, static_cast<uint8_t>(forced_die), 0, 0};
        if (!ts::Engine::step(s, roll)) return false;
        if (settled && s.last_roll.type != ts::RollType::NONE) {
            *settled = s;
            keep_roll(s);
            keep_event_tracks(s);
        }
    }
    return true;
}

// All or nothing: a refused action, or a refused forced die, leaves the game where it was.
int apply(const ts::MicroAction& action, int forced_die) {
    if (ts::Engine::is_terminal(g_state)) { g_error = "the game is over"; return 0; }
    if (forced_die < 0 || forced_die > 6) { g_error = "forced die must be 0..6"; return 0; }
    const ts::GameState before = g_state;
    if (!ts::Engine::step(g_state, action)) {
        g_state = before;
        g_error = "the engine refused the action";
        return 0;
    }
    g_settled = g_state;
    g_rolls.clear();
    g_event_tracks.clear();
    keep_roll(g_state);
    keep_event_tracks(g_state);
    if (!drain(g_state, forced_die, &g_settled)) {
        g_state = before;
        g_error = "the engine refused the die roll";
        return 0;
    }
    return 1;
}

ts::Player player_from_int(int p) {
    return p > 0 ? ts::Player::US : (p < 0 ? ts::Player::USSR : ts::Player::NONE);
}

}  // namespace

// ---- identity ---------------------------------------------------------------------------------

// tools/lib/engine_fingerprint.py over the sources this build was made from: the page shows it,
// and compares it with the fingerprint a model was exported next to.
TS_API const char* ts_fingerprint() { return TS_ENGINE_FINGERPRINT; }
TS_API int ts_obs_size() { return static_cast<int>(ts::OBS_SIZE_V23); }
TS_API int ts_action_size() { return static_cast<int>(ts::FLAT_ACTION_SPACE_SIZE); }

// The flat action space's layout, so the page never keeps its own copy of these offsets.
TS_API const char* ts_action_layout_json() {
    namespace fs = ts::flat_slots;
    sj::Value v = sj::Value::object();
    v.set("size", static_cast<int>(fs::SIZE));
    sj::Value off = sj::Value::object();
    off.set("card", static_cast<int>(fs::CARD));
    off.set("play_mode", static_cast<int>(fs::RESOLUTION));
    off.set("op_mode", static_cast<int>(fs::OP_MODE));
    off.set("roll_die", static_cast<int>(fs::ROLL_DIE));
    off.set("node", static_cast<int>(fs::NODE));
    off.set("branch", static_cast<int>(fs::BRANCH));
    off.set("defcon_value", static_cast<int>(fs::DEFCON_VALUE));
    off.set("region", static_cast<int>(fs::REGION));
    v.set("offsets", std::move(off));
    v.set("confirm_done_index", static_cast<int>(fs::CONFIRM_DONE));
    v.set("ops_influence_index", static_cast<int>(fs::OPS_INFLUENCE));
    sj::Value dt = sj::Value::object();
    dt.set("SELECT_CARD", static_cast<int>(ts::DecisionType::SELECT_CARD));
    dt.set("SELECT_PLAY_MODE", static_cast<int>(ts::DecisionType::SELECT_PLAY_MODE));
    dt.set("CHOOSE_TIMING_BRANCH", static_cast<int>(ts::DecisionType::CHOOSE_TIMING_BRANCH));
    dt.set("SELECT_OP_MODE", static_cast<int>(ts::DecisionType::SELECT_OP_MODE));
    dt.set("POINT_NODE", static_cast<int>(ts::DecisionType::POINT_NODE));
    dt.set("CHOOSE_BRANCH", static_cast<int>(ts::DecisionType::CHOOSE_BRANCH));
    dt.set("ROLL_DIE", static_cast<int>(ts::DecisionType::ROLL_DIE));
    v.set("decision_types", std::move(dt));
    return keep(v.dump());
}

TS_API const char* ts_last_error() { return g_error.c_str(); }

// The engine's own names, for the action log -- the same text the Python session wrote.
TS_API const char* ts_card_name(int id) {
    return (id >= 1 && id <= 110) ? keep(std::string(ts::CardData::get_card_name(static_cast<uint8_t>(id)))) : "";
}
TS_API int ts_card_ops(int id) {
    return (id >= 1 && id <= 110) ? static_cast<int>(ts::CardData::get_card(static_cast<uint8_t>(id)).ops) : 0;
}
TS_API const char* ts_country_name(int id) {
    return (id >= 0 && id < 84) ? keep(std::string(ts::MapData::get_country_name(static_cast<uint8_t>(id)))) : "";
}

// ---- the position -----------------------------------------------------------------------------

TS_API void ts_new_game(uint32_t seed_lo, uint32_t seed_hi) {
    ts::Engine::init_game(g_state, (static_cast<uint64_t>(seed_hi) << 32) | seed_lo);
    drain(g_state, 0);
}

// Raw GameState bytes, for the page's undo stack: copy out ts_state_size() bytes from
// ts_state_ptr(), and write them back to restore. Trivially copyable by the engine's invariant.
TS_API int ts_state_size() { return static_cast<int>(sizeof(ts::GameState)); }
TS_API uint8_t* ts_state_ptr() { return reinterpret_cast<uint8_t*>(&g_state); }

TS_API const char* ts_display_json() { return keep(sj::display_state(g_state).dump()); }
// The display state of what the last click itself did, before the roll-less chance steps after it
// (g_settled): its die roll, and the board before the turn's end.
TS_API const char* ts_settled_display_json() { return keep(sj::display_state(g_settled).dump()); }
// Every die the last click rolled, in order (g_rolls): a JSON array of display-state rolls.
TS_API const char* ts_click_rolls_json() {
    sj::Value a = sj::Value::array();
    for (const sj::Value& r : g_rolls) a.push(r);
    return keep(a.dump());
}
TS_API const char* ts_click_event_tracks_json() {
    sj::Value a = sj::Value::array();
    for (const sj::Value& t : g_event_tracks) a.push(t);
    return keep(a.dump());
}
TS_API const char* ts_save_json() { return keep(sj::save(g_state).dump()); }

// Open a saved position (a shared link). Returns 1, or 0 with ts_last_error() -- and a save
// that does not survive load -> save unchanged is refused rather than opened approximately.
TS_API int ts_load_save_json(const char* text) {
    sj::Value v;
    std::string error;
    if (!sj::Value::parse(text, v, &error)) { g_error = "not JSON: " + error; return 0; }
    ts::GameState s{};
    if (!sj::load_save(v, s, &error)) { g_error = "not a save: " + error; return 0; }
    if (sj::save(s).dump() != v.dump()) { g_error = "the save does not round-trip through the engine"; return 0; }
    if (!drain(s, 0)) { g_error = "the saved position's die roll was refused"; return 0; }
    g_state = s;
    return 1;
}

TS_API int ts_is_terminal() { return ts::Engine::is_terminal(g_state) ? 1 : 0; }
TS_API float ts_terminal_utility() { return ts::Engine::get_terminal_utility(g_state); }
TS_API const char* ts_ending_reason() { return sj::ending_reason(g_state); }
TS_API int ts_decision_player() { return static_cast<int>(g_state.ctx().decision_player); }
TS_API int ts_decision_type() { return static_cast<int>(g_state.ctx().decision_type); }

// ---- acting -----------------------------------------------------------------------------------

// A click: one MicroAction, then the chance nodes it lands on (forced_die 0 = roll).
TS_API int ts_step(int decision_type, int primary, int secondary, int flags, int forced_die) {
    const ts::MicroAction a{static_cast<ts::DecisionType>(decision_type), static_cast<uint8_t>(primary),
                            static_cast<uint8_t>(secondary), static_cast<uint8_t>(flags)};
    if (a.decision_type != g_state.ctx().decision_type) {
        g_error = "the action is for a different decision than the one open";
        return 0;
    }
    return apply(a, forced_die);
}

// The flat legal mask in either action view (E4, or E4.1's merged influence).
TS_API const uint8_t* ts_mask(int merged) {
    ts::Engine::get_flat_action_mask(g_state, g_mask, merged != 0);
    return g_mask;
}

// The MicroAction a flat index stands for at the open decision: [type, primary, secondary, flags].
TS_API const int32_t* ts_decode_flat(int idx) {
    const ts::MicroAction a = ts::ActionMask::decode_flat_action_212(g_state, static_cast<uint16_t>(idx));
    g_decoded[0] = static_cast<int32_t>(a.decision_type);
    g_decoded[1] = a.primary_id;
    g_decoded[2] = a.secondary_id;
    g_decoded[3] = a.flags;
    return g_decoded;
}

// Whether a flat index is an E4.1 composed action here ("Ops for influence, first point at X").
TS_API int ts_is_merged_influence_action(int idx) {
    return ts::ActionMask::is_merged_influence_action(g_state, static_cast<uint16_t>(idx)) ? 1 : 0;
}

// The observation from one side's perspective (1 = US, -1 = USSR), OBS_SIZE floats.
TS_API const float* ts_observation(int player) {
    ts::ObservationBufferV23 ob;
    ts::Observation::extract(g_state, player_from_int(player), &ob);
    std::memcpy(g_obs, reinterpret_cast<const float*>(&ob), sizeof(ob));
    return g_obs;
}

// The width of the observation a model reading the obs_features bits `features` takes, or 0 (and the
// reason in ts_last_error) for a bit this build does not know.
TS_API int ts_obs_size_for(uint32_t features) {
    if (features & ~ts::obs_features::ALL) { g_error = "unknown obs_features bit"; return 0; }
    return static_cast<int>(ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features));
}

// The observation in a model's own view: the base layout as the bits `features` edit and extend it
// (extract_observation_features, the same call Python makes for that model), ts_obs_size_for(features)
// floats; null for an unknown bit. HEADLINE_REVEAL has no width, so only this tells its view apart.
TS_API const float* ts_observation_features(int player, uint32_t features) {
    if (features & ~ts::obs_features::ALL) { g_error = "unknown obs_features bit"; return nullptr; }
    ts::extract_observation_features(g_state, player_from_int(player), features, g_obs);
    return g_obs;
}

// ---- the workbench's debug tools --------------------------------------------------------------

TS_API int ts_set_country(int cid, int us, int ussr) {
    if (cid < 0 || cid >= 84 || us < 0 || us > 255 || ussr < 0 || ussr > 255) { g_error = "out of range"; return 0; }
    g_state.countries[cid].us_influence = static_cast<uint8_t>(us);
    g_state.countries[cid].ussr_influence = static_cast<uint8_t>(ussr);
    return 1;
}
TS_API int ts_set_defcon(int v) {
    if (v < 1 || v > 5) { g_error = "DEFCON must be 1..5"; return 0; }
    g_state.defcon = static_cast<uint8_t>(v);
    return 1;
}
TS_API int ts_set_vp(int v) {
    if (v < -20 || v > 20) { g_error = "VP must be -20..20"; return 0; }
    g_state.victory_points = static_cast<int8_t>(v);
    return 1;
}

// ---- the rollout root's playouts (bindings/rollout_batch.hpp, the same C++ as Python's) --------
//
// One batch at a time. The page: ts_rollout_reset, ts_rollout_add on the position it holds (its
// candidates, its worlds), then until ts_rollout_all_done: ts_rollout_fill_live -> read the
// observation and mask buffers -> the network -> ts_rollout_step(logits); then
// ts_rollout_fill_leaves -> the critic -> ts_rollout_set_leaf_values; then ts_rollout_values and
// the row tables to average each candidate over its worlds.

namespace {
std::unique_ptr<ts::rollout::RolloutBatch> g_rollout;
std::unique_ptr<ts::belief::Tracker> g_belief;
std::vector<uint8_t> g_belief_hands;    // the worlds' hands ts_rollout_add_belief dealt
std::vector<uint8_t> g_belief_held;     // a chunk's hands for ts_belief_rows
std::vector<float> g_ro_obs;
std::vector<uint8_t> g_ro_masks;
std::vector<float> g_ro_values;
std::vector<uint16_t> g_ro_cands;
}  // namespace

// Redeal the cards `player` cannot see in the position held, from SplitMix64 state `seed`.
TS_API void ts_determinize(int player, uint32_t seed_lo, uint32_t seed_hi) {
    uint64_t rng = (static_cast<uint64_t>(seed_hi) << 32) | seed_lo;
    ts::rollout::determinize(g_state, player_from_int(player), rng);
}

// `stop`: 0 = after `horizon` boundaries, 1 = at the end of the decision's turn, 2 = the later of the two,
// 3 = the earlier of the two, 4 = at the turn-end park before the deal, 5 = the earlier of the boundaries
// and that park (ts::rollout::Stop). `leaf`: 0 = read every leaf from the root mover's side,
// 1 = from the leaf's decider's, negated into the mover's frame, 2 = play on to the mover's next decision
// (ts::rollout::Leaf).
TS_API int ts_rollout_reset(int horizon, int shared_dice, int guard, uint32_t features,
                            uint32_t seed_lo, uint32_t seed_hi, int stop, int leaf, int keyed) {
    if (horizon < 0) { g_error = "horizon must be >= 0"; return 0; }
    if (stop < 0 || stop > 5) {
        g_error = "stop must be 0 (boundaries), 1 (turn), 2 (later), 3 (earlier), 4 (predeal) or 5 (earlier_predeal)";
        return 0;
    }
    if (leaf < 0 || leaf > 2) { g_error = "leaf must be 0 (mover), 1 (decider) or 2 (own)"; return 0; }
    if (features & ~ts::obs_features::ALL) { g_error = "unknown obs_features bit"; return 0; }
    const uint64_t seed = (static_cast<uint64_t>(seed_hi) << 32) | seed_lo;
    g_rollout = std::make_unique<ts::rollout::RolloutBatch>(
        ts::rollout::Config{horizon, shared_dice != 0, guard != 0, features, static_cast<ts::rollout::Stop>(stop),
                            static_cast<ts::rollout::Leaf>(leaf), keyed != 0}, seed);
    return 1;
}

// The position held, with `k` flat candidates at `candidates`, in `worlds` worlds. Rows added.
TS_API int ts_rollout_add(const int32_t* candidates, int k, int worlds) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    g_ro_cands.clear();
    for (int i = 0; i < k; ++i) {
        if (candidates[i] < 0 || candidates[i] >= static_cast<int32_t>(ts::FLAT_ACTION_SPACE_SIZE)) {
            g_error = "a candidate is outside the flat action space";
            return 0;
        }
        g_ro_cands.push_back(static_cast<uint16_t>(candidates[i]));
    }
    const size_t n = g_rollout->add(g_state, g_ro_cands.data(), g_ro_cands.size(),
                                    static_cast<size_t>(worlds < 0 ? 0 : worlds));
    if (n == 0 && g_rollout->last_error()[0]) g_error = g_rollout->last_error();
    return static_cast<int>(n);
}

// ---- the belief over the opponent's hand (bindings/belief.hpp, the same C++ as Python's) -------
//
// One tracker, one game (game 0). The page: ts_belief_reset at a new game; ts_belief_observe before
// every move is applied (the position held is the one the move is made in); at the AI's decision
// ts_belief_prepare, then for each recorded decision and each chunk of hands ts_belief_rows -> the
// network -> the log-probability of ts_belief_action(d) at each of ts_belief_temperature(t) ->
// ts_belief_set_log_likelihoods; then ts_rollout_add_belief deals the root's worlds from it.

TS_API void ts_belief_reset(uint32_t particles, int all_decisions, uint32_t seed_lo, uint32_t seed_hi) {
    g_belief = std::make_unique<ts::belief::Tracker>(particles, all_decisions != 0,
                                                     (static_cast<uint64_t>(seed_hi) << 32) | seed_lo);
}
TS_API int ts_belief_temperatures() { return static_cast<int>(ts::belief::kTemps); }
TS_API float ts_belief_temperature(int t) {
    return (t >= 0 && static_cast<size_t>(t) < ts::belief::kTemps) ? ts::belief::kTemperatures[t] : 0.0f;
}
TS_API int ts_belief_particles() { return g_belief ? static_cast<int>(g_belief->particles()) : 0; }

// The move [type, primary, secondary, flags] about to be applied at the position held: recorded as the
// flat action that decodes to it. 0 when there is no tracker, no decider, or no flat action is the move
// (then nothing is recorded -- a move the network could not have made carries no likelihood).
TS_API int ts_belief_observe(int decision_type, int primary, int secondary, int flags) {
    if (!g_belief) return 0;
    const ts::Player decider = g_state.ctx().decision_player;
    if (decider != ts::Player::US && decider != ts::Player::USSR) return 0;
    uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
    ts::ActionMask::generate_flat_mask_212(g_state, mask);
    for (size_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a) {
        if (!mask[a]) continue;
        const ts::MicroAction m = ts::ActionMask::decode_flat_action_212(g_state, static_cast<uint16_t>(a));
        if (static_cast<int>(m.decision_type) == decision_type && m.primary_id == primary && m.secondary_id == secondary &&
            m.flags == flags) {
            g_belief->observe(0, g_state, decider, static_cast<uint16_t>(a));
            return 1;
        }
    }
    return 0;
}

// The belief for the position held: 1 if it has one (then ts_belief_decisions() rows blocks to
// evaluate, 0 when its weights are reused), 0 where there is nothing to believe.
TS_API int ts_belief_prepare() {
    if (!g_belief) return 0;
    const std::vector<size_t> games{0};
    const std::vector<const ts::GameState*> states{&g_state};
    return g_belief->prepare(games, states)[0] ? 1 : 0;
}
TS_API int ts_belief_decisions() {
    return (g_belief && !g_belief->jobs().empty()) ? static_cast<int>(g_belief->jobs()[0].decisions.size()) : 0;
}
TS_API int ts_belief_action(int d) {
    if (!g_belief || g_belief->jobs().empty() || d < 0 || static_cast<size_t>(d) >= g_belief->jobs()[0].decisions.size())
        return -1;
    return g_belief->decision(0, static_cast<size_t>(d)).action;
}
// Recorded decision d under hands lo .. lo + n - 1: n observation rows (from the opponent's side, in
// feature set `features`) and masks into the rollout buffers (ts_rollout_buf_obs / _masks). Rows written.
TS_API int ts_belief_rows(int d, int lo, int n, uint32_t features) {
    if (!g_belief || g_belief->jobs().empty()) { g_error = "ts_belief_prepare made no job"; return 0; }
    const auto& job = g_belief->jobs()[0];
    const size_t N = g_belief->particles();
    if (d < 0 || static_cast<size_t>(d) >= job.decisions.size() || lo < 0 || n <= 0 || static_cast<size_t>(lo + n) > N) {
        g_error = "no such decision or hands";
        return 0;
    }
    const auto& left = job.lefts[static_cast<size_t>(d)];
    g_belief_held.assign(static_cast<size_t>(n) * 110, 0);
    for (int k = 0; k < n; ++k)
        for (size_t c = 0; c < 110; ++c)
            g_belief_held[static_cast<size_t>(k) * 110 + c] =
                static_cast<uint8_t>(job.hands[static_cast<size_t>(lo + k) * 110 + c] | left[c]);
    const size_t width = ts::OBS_SIZE_V23 + ts::obs_features::extra_width(features);
    g_ro_obs.assign(static_cast<size_t>(n) * width, 0.0f);
    g_ro_masks.assign(static_cast<size_t>(n) * ts::FLAT_ACTION_SPACE_SIZE, 0);
    ts::rollout::hypothesis_rows(g_belief->decision(0, static_cast<size_t>(d)).state, job.me, g_belief_held.data(),
                                 static_cast<size_t>(n), features, g_ro_obs.data(), g_ro_masks.data());
    return n;
}
// lp: temperatures x decisions x particles floats, the log-probability of each recorded decision's move.
TS_API int ts_belief_set_log_likelihoods(const float* lp) {
    if (!g_belief || g_belief->jobs().empty()) { g_error = "ts_belief_prepare made no job"; return 0; }
    g_belief->set_log_likelihoods(lp, g_belief->jobs()[0].decisions.size());
    return 1;
}
// The temperature weights of the position prepared (ts_belief_temperatures() floats into `out`).
TS_API int ts_belief_temperature_weights(float* out) {
    if (!g_belief) return 0;
    const auto w = g_belief->temperature_weights(0);
    std::copy(w.begin(), w.end(), out);
    return 1;
}
// `n` hands resampled from the prepared belief (n x 110 bytes, card c at c - 1); null where it has none.
TS_API const uint8_t* ts_belief_worlds(int n) {
    if (!g_belief || n <= 0) return nullptr;
    g_belief_hands.assign(static_cast<size_t>(n) * 110, 0);
    return g_belief->worlds(0, static_cast<size_t>(n), g_belief_hands.data()) ? g_belief_hands.data() : nullptr;
}

// ts_rollout_add with the worlds' opposing hands resampled from the prepared belief. Rows added; 0 with
// ts_last_error() when the position has no belief (the caller then uses ts_rollout_add).
TS_API int ts_rollout_add_belief(const int32_t* candidates, int k, int worlds) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    if (!g_belief || worlds <= 0) { g_error = "no belief"; return 0; }
    g_belief_hands.assign(static_cast<size_t>(worlds) * 110, 0);
    if (!g_belief->worlds(0, static_cast<size_t>(worlds), g_belief_hands.data())) { g_error = "no belief here"; return 0; }
    g_ro_cands.clear();
    for (int i = 0; i < k; ++i) {
        if (candidates[i] < 0 || candidates[i] >= static_cast<int32_t>(ts::FLAT_ACTION_SPACE_SIZE)) {
            g_error = "a candidate is outside the flat action space";
            return 0;
        }
        g_ro_cands.push_back(static_cast<uint16_t>(candidates[i]));
    }
    const size_t n = g_rollout->add(g_state, g_ro_cands.data(), g_ro_cands.size(), static_cast<size_t>(worlds),
                                    g_belief_hands.data());
    if (n == 0 && g_rollout->last_error()[0]) g_error = g_rollout->last_error();
    return static_cast<int>(n);
}

TS_API int ts_rollout_rows() { return g_rollout ? static_cast<int>(g_rollout->rows()) : 0; }
TS_API int ts_rollout_live() { return g_rollout ? static_cast<int>(g_rollout->live()) : 0; }
TS_API int ts_rollout_all_done() { return !g_rollout || g_rollout->all_done() ? 1 : 0; }
TS_API int ts_rollout_obs_width() { return g_rollout ? static_cast<int>(g_rollout->obs_width()) : 0; }
TS_API int ts_rollout_refused() { return g_rollout ? static_cast<int>(g_rollout->refused()) : 0; }
TS_API int ts_rollout_worlds_used(int position) {
    if (!g_rollout || position < 0 || static_cast<size_t>(position) >= g_rollout->positions()) return 0;
    return static_cast<int>(g_rollout->worlds_used(static_cast<size_t>(position)));
}

// The live rows' observations and masks into the buffers below; returns the count.
TS_API int ts_rollout_fill_live() {
    if (!g_rollout) return 0;
    const size_t n = g_rollout->live();
    g_ro_obs.assign(std::max<size_t>(1, n * g_rollout->obs_width()), 0.0f);
    g_ro_masks.assign(std::max<size_t>(1, n * ts::FLAT_ACTION_SPACE_SIZE), 0);
    return static_cast<int>(g_rollout->fill_live(g_ro_obs.data(), g_ro_masks.data()));
}
TS_API const float* ts_rollout_buf_obs() { return g_ro_obs.data(); }
TS_API const uint8_t* ts_rollout_buf_masks() { return g_ro_masks.data(); }

// `logits`: count x FLAT_ACTION_SPACE_SIZE floats for the rows the last fill_live listed.
TS_API int ts_rollout_step(const float* logits) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    g_rollout->step(logits);
    return 1;
}

// The rows at their horizon and not yet valued, from the mover's side, into the same buffers.
TS_API int ts_rollout_fill_leaves() {
    if (!g_rollout) return 0;
    size_t n = 0;
    for (size_t r = 0; r < g_rollout->rows(); ++r) {
        const auto& row = g_rollout->row(r);
        n += (row.done && !row.terminal && !row.refused && !row.leaf_valued) ? 1 : 0;
    }
    g_ro_obs.assign(std::max<size_t>(1, n * g_rollout->obs_width()), 0.0f);
    g_ro_masks.assign(std::max<size_t>(1, n * ts::FLAT_ACTION_SPACE_SIZE), 0);
    return static_cast<int>(g_rollout->fill_leaves(g_ro_obs.data(), g_ro_masks.data()));
}
TS_API int ts_rollout_set_leaf_values(const float* values) {
    if (!g_rollout) { g_error = "ts_rollout_reset first"; return 0; }
    g_rollout->set_leaf_values(values);
    return 1;
}

// Every row's value from its mover's side (NaN while live or unvalued), rows() floats.
TS_API const float* ts_rollout_values() {
    g_ro_values.assign(std::max<size_t>(1, g_rollout ? g_rollout->rows() : 0), 0.0f);
    if (g_rollout) g_rollout->values(g_ro_values.data());
    return g_ro_values.data();
}
TS_API int ts_rollout_row_position(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? static_cast<int>(g_rollout->row(r).position) : -1;
}
TS_API int ts_rollout_row_world(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? g_rollout->row(r).world : -1;
}
TS_API int ts_rollout_row_candidate(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? g_rollout->row(r).candidate : -1;
}
TS_API int ts_rollout_row_done(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? (g_rollout->row(r).done ? 1 : 0) : -1;
}
TS_API int ts_rollout_row_terminal(int r) {
    return (g_rollout && r >= 0 && static_cast<size_t>(r) < g_rollout->rows()) ? (g_rollout->row(r).terminal ? 1 : 0) : -1;
}

// The `k` most probable legal moves by the root's logits (FLAT_ACTION_SPACE_SIZE floats) under
// `mask`, best first, into `out` (k int32s); returns how many.
TS_API int ts_rollout_candidates(const float* logits, const uint8_t* mask, int k, int32_t* out) {
    if (k <= 0) return 0;
    std::vector<uint16_t> c(static_cast<size_t>(k));
    const size_t n = ts::rollout::top_candidates(logits, mask, static_cast<size_t>(k), c.data());
    for (size_t i = 0; i < n; ++i) out[i] = c[i];
    return static_cast<int>(n);
}

// One position's decision from the valued batch, as JSON: {"pick", "worlds", "q", "lead",
// "lead_se"}; "" with ts_last_error() for an unknown rule or position.
TS_API const char* ts_rollout_decide(int position, const int32_t* candidates, int k, const float* logits,
                                     const uint8_t* mask, const char* rule_text) {
    if (!g_rollout || position < 0 || static_cast<size_t>(position) >= g_rollout->positions()) {
        g_error = "no such position in the rollout batch";
        return "";
    }
    ts::rollout::Rule rule;
    if (!ts::rollout::parse_rule(rule_text, rule)) { g_error = "unknown rule: argmax, z<x> or kl<t>"; return ""; }
    std::vector<uint16_t> c;
    for (int i = 0; i < k; ++i) {
        if (candidates[i] < 0 || candidates[i] >= static_cast<int32_t>(ts::FLAT_ACTION_SPACE_SIZE)) {
            g_error = "a candidate is outside the flat action space";
            return "";
        }
        c.push_back(static_cast<uint16_t>(candidates[i]));
    }
    const ts::rollout::Decision d = ts::rollout::decide(*g_rollout, static_cast<size_t>(position), c.data(), c.size(),
                                                        logits, mask, rule);
    auto list = [](const std::vector<float>& v) {
        std::string s = "[";
        char buf[32];
        for (size_t i = 0; i < v.size(); ++i) {
            std::snprintf(buf, sizeof buf, "%.9g", static_cast<double>(v[i]));
            s += (i ? "," : "");
            s += buf;
        }
        return s + "]";
    };
    std::string out = "{\"pick\":" + std::to_string(d.pick) + ",\"worlds\":" + std::to_string(d.worlds)
        + ",\"q\":" + list(d.q) + ",\"lead\":" + list(d.lead) + ",\"lead_se\":" + list(d.lead_se) + "}";
    return keep(out);
}

// ---- parity -----------------------------------------------------------------------------------

// The shared whole-game digest (bindings/selftest.hpp): the Python module computes the same
// number natively, so tests/web/test_wasm_engine.py can compare the two builds directly.
TS_API uint32_t ts_selftest(int games, uint32_t* steps_out) { return ts::selftest::digest(games, steps_out); }
