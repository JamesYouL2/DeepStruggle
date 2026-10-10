// The rollout root's playouts, written once for both bindings (ai/search/rollout_root.py and the
// browser workbench): the worlds of a decision, their greedy playouts to a horizon, and the leaves
// the critic reads. The network stays outside -- the caller runs it once per ply over the live rows
// and hands the logits back -- so the same C++ drives torch from Python and onnxruntime from the page.
//
// One decision: the mover's `k` candidate moves, each played in `worlds` sampled worlds. A world
// redeals the cards the mover cannot see (`determinize`, the honest searcher's) and, with
// `shared_dice`, draws one dice stream that every candidate of the world then plays under, so two
// candidates' values in a world differ only by the move. A world in which any candidate is illegal
// (The Cambridge Five can change the legal set) is dropped for every candidate. Each row then plays
// greedily -- both sides the network's argmax -- through `horizon` action-round boundaries (a change
// of turn, action round, phasing side or phase; 0 reads the position the move leads to), with the
// DEFCON-2 suicide guard when `guard` is set: at a play mode, an event branch or a coup's target,
// a greedy move that ends the game against its own player on the spot is replaced by the next-best
// of the top four that does not. A finished game is the row's value from the mover's side; a row
// at its horizon is a leaf, whose observation from the mover's side the caller values with the
// critic. Several decisions can share one batch, so a tournament's hundred positions make one
// network call per ply.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "ts/game_state.hpp"

namespace ts::rollout {

// A clone of `s` with the cards `me` cannot see -- the opponent's unknown hand and the draw deck --
// reshuffled between those two places, every count kept and every card `me` has seen left where
// it is; an opponent's chosen but unrevealed headline is forgotten, so the world asks it again
// from the hand it dealt. `rng` is a SplitMix64 state, advanced. The port of
// ai/search/dmcts.determinize; the Python one draws from random.Random, so the two draw different
// worlds from one seed, by design -- this one is what the rollout root uses in both builds.
void determinize(GameState& s, Player me, uint64_t& rng) noexcept;

// The side to act at `s`: the decision's player, else the phasing player (ai/search/pimcts.acting_player).
Player acting_player(const GameState& s) noexcept;

struct Config {
    int horizon = 4;            // action-round boundaries to play through; 0 = none
    bool shared_dice = true;    // one dice stream per world, every candidate
    bool guard = true;          // the DEFCON-2 suicide guard in the playouts
    uint32_t features = 0;      // obs_features bits the network reads (0: the base layout)
};

struct Row {
    GameState state;
    uint32_t position;          // which decision of the batch
    uint16_t world;             // index of the world within the position (dropped worlds excluded)
    uint16_t candidate;         // index into that decision's candidate list
    int32_t crossed;            // action-round boundaries crossed so far
    uint8_t turn, action_round; // the row's current key, with phasing and phase below
    Player phasing;
    Phase phase;
    bool done, terminal, leaf_valued, refused;
    float value;                // from the mover's side, once terminal or valued
};

class RolloutBatch {
public:
    explicit RolloutBatch(const Config& cfg, uint64_t seed) noexcept;

    // Add one decision: `base` at a decision of `mover()`, `k` flat candidate actions, `worlds`
    // worlds. Returns the rows added (k times the worlds kept). Every candidate must be legal at
    // `base`; a caller passing one that is not gets 0 rows and `last_error()`.
    size_t add(const GameState& base, const uint16_t* candidates, size_t k, size_t worlds);

    size_t positions() const noexcept { return movers_.size(); }
    Player mover(size_t position) const noexcept { return movers_[position]; }
    size_t worlds_used(size_t position) const noexcept { return worlds_used_[position]; }
    size_t rows() const noexcept { return rows_.size(); }
    const Row& row(size_t r) const noexcept { return rows_[r]; }
    size_t obs_width() const noexcept { return obs_width_; }
    size_t live() const noexcept;           // rows not yet done
    bool all_done() const noexcept { return live() == 0; }
    size_t listed() const noexcept { return live_order_.size(); }        // rows the last fill_live wrote
    size_t leaves_listed() const noexcept { return leaf_order_.size(); } // rows the last fill_leaves wrote
    size_t refused() const noexcept;        // rows the engine refused a step on (an error)
    const char* last_error() const noexcept { return error_; }

    // Write the live rows' observations (from the row's acting side, in the batch's feature set)
    // and legal masks, row-major, into caller buffers of live() * obs_width() floats and
    // live() * FLAT_ACTION_SPACE_SIZE bytes; returns the count and remembers the order for step().
    size_t fill_live(float* obs, uint8_t* masks);
    // Advance every row fill_live listed by the masked argmax of its logits row
    // (count x FLAT_ACTION_SPACE_SIZE), the guard applied, then settle what the player does not
    // control; count boundaries; mark terminal rows and rows at their horizon done.
    void step(const float* logits);

    // Rows done, not terminal and not yet valued: their observations from the MOVER's side and
    // legal masks into caller buffers; returns the count and remembers the order.
    size_t fill_leaves(float* obs, uint8_t* masks);
    // The critic's values for those rows, in that order, from the mover's side.
    void set_leaf_values(const float* values);

    // Every row's value from the mover's side, NaN while a row is live or an unvalued leaf.
    void values(float* out) const noexcept;

private:
    Config cfg_;
    uint64_t rng_;
    size_t obs_width_;
    std::vector<Row> rows_;
    std::vector<Player> movers_;
    std::vector<size_t> worlds_used_;
    std::vector<uint32_t> live_order_;
    std::vector<uint32_t> leaf_order_;
    const char* error_ = "";

    void settle_and_mark(Row& r) noexcept;
    uint16_t guarded_choice(const GameState& s, const float* logits, const uint8_t* mask,
                            uint16_t argmax) const noexcept;
};

}  // namespace ts::rollout
