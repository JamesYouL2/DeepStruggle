#pragma once
#include <cstddef>
#include <cstdint>
#include "types.hpp"
#include "game_state.hpp"

// What each card in the mover's hand does on the current board: the P30 C2 labels
// (ai/training/card_event_targets.py), computed by the engine so they can be an observation block
// (obs_features::CARD_EFFECTS) rather than only an auxiliary target. The probe in
// research/log/P30_card_board_targets.md found RL never puts these into the trunk -- a trained
// trunk reads them no better than an untrained one -- so the block hands them over as input.
//
// Per card, PER_CARD floats:
//   [0]       the event can fire (CardHandlers::can_trigger_event, for the player it fires for);
//   [1..5]    Ops reach: effective Ops; countries and battlegrounds the Ops could bring under
//             control; the best coup success chance, and the best on a battleground;
//   [6..17]   what the event does here: the change in VP and DEFCON, the six regional scoring
//             margins, battlegrounds controlled by each side, total influence of each side --
//             all from the mover's side. An event that stops for choices is played out with each
//             choice taken greedily by whoever makes it, for a fixed board score.
namespace ts::card_effects {

constexpr size_t N_CARDS = 110;
constexpr size_t N_T2 = 5;
constexpr size_t N_T4 = 12;
constexpr size_t PER_CARD = 1 + N_T2 + N_T4;      // 18
constexpr size_t WIDTH = N_CARDS * PER_CARD;      // 1,980
constexpr size_t T2_AT = 1;
constexpr size_t T4_AT = 1 + N_T2;
constexpr size_t MAX_HAND = 12;                   // as the Python labeller: the first 12 held, by id
constexpr size_t MAX_EVENT_STEPS = 60;
constexpr size_t N_BLOCK_SEEDS = 4;               // dice seeds the observation block averages over

// The labels for every card in `mover`'s hand, unscaled, written to out[WIDTH]; rows of cards not
// held are zero, and so is a held card's outcome when its event cannot fire, has none (the China
// Card) or does not finish within MAX_EVENT_STEPS. Event outcomes are averaged over `seeds` (the
// state's rng_state for each run); an event that draws nothing from the generator is run once.
// `state` is used as given -- this is the labeller, and it sees what it is handed. Returns the
// number of held cards labelled.
size_t label(const GameState& state, Player mover, const uint64_t* seeds, size_t n_seeds,
             float* out) noexcept;

// The decisions the block is filled at: `perspective` choosing the card to headline or to play in
// an action round (SELECT_CARD outside an event) or how to play it (SELECT_PLAY_MODE).
bool is_card_decision(const GameState& state, Player perspective) noexcept;

// `state` with the cards `me` cannot see -- the opponent's unrevealed hand and the draw deck --
// shuffled between them, counts preserved, and an opponent headline chosen but not yet revealed
// forgotten (as ai/search/dmcts.py determinize). Deterministic in the state.
GameState redeal(const GameState& state, Player me) noexcept;

// The observation block, out[WIDTH]: zero unless is_card_decision(state, perspective); otherwise
// label() of redeal(state, perspective) over N_BLOCK_SEEDS seeds drawn from the state, scaled to
// unit order (the can-fire flag as 0/1, the rest divided by SCALE).
void write_block(const GameState& state, Player perspective, float* out) noexcept;

// The fixed scales of the 17 quantities after the flag, T2 then T4: the standard deviations of
// card_event_targets.AUX_SCALE, so a block value of 1 is one typical spread of that quantity.
extern const float SCALE[N_T2 + N_T4];

} // namespace ts::card_effects
