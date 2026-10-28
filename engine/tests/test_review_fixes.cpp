// Regression tests for the 2026-10-28 review fixes: the We Will Bury You round rule and DEFCON-1
// classification, the single VP award, Special Relationship's winning VP, and out-of-range
// actions being refused rather than aliased to a confirm-done.
#include <cstring>

#include "test_framework.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_handlers.hpp"
#include "ts/constants.hpp"
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/victory_points.hpp"

using namespace ts;

namespace {

// An action-round position with every card out of both hands, so a test seats exactly what it
// needs.
GameState empty_hands_action_round(uint64_t seed) {
    GameState state{};
    Engine::init_game(state, seed);
    for (uint8_t i = 1; i <= 110; ++i) {
        if (in_hand_of(state.card_locations[i], Player::US) ||
            in_hand_of(state.card_locations[i], Player::USSR)) {
            state.card_locations[i] = CardLocation::DISCARD_PILE;
        }
    }
    state.current_phase = Phase::ACTION_ROUND;
    state.turn = 4;
    state.action_round = 3;
    state.victory_points = 0;
    return state;
}

} // namespace

// We Will Bury You's 3 VP are owed on the US's next action round unless UN Intervention is
// played as an Event in it. A round the US starts with an empty hand is that round -- nothing,
// UN Intervention included, can be played in it -- so the USSR is paid there, not whenever the
// US next happens to play a card.
TEST(ReviewFixes, WeWillBuryYouPaysOnAnEmptyHandedUsRound) {
    GameState state = empty_hands_action_round(31);
    state.china_card_holder = Player::USSR;
    state.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::USSR);
    state.phasing_player = Player::USSR;
    state.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);

    StateMachine::advance_after_action_round(state);   // the US's round: no cards, passed

    ASSERT_EQ(static_cast<int>(state.victory_points), -3);
    ASSERT_FALSE(state.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
    ASSERT_EQ(static_cast<int>(state.phasing_player), static_cast<int>(Player::USSR));
}

// The same when the US passes a round holding nothing but the China Card.
TEST(ReviewFixes, WeWillBuryYouPaysOnAUsPassHoldingOnlyTheChinaCard) {
    GameState state = empty_hands_action_round(32);
    state.china_card_holder = Player::US;
    state.china_card_playable = 1;
    state.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::USSR);
    state.phasing_player = Player::USSR;
    state.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);

    StateMachine::advance_after_action_round(state);   // the US's round opens on SELECT_CARD
    ASSERT_EQ(static_cast<int>(state.ctx().decision_type), static_cast<int>(DecisionType::SELECT_CARD));
    ASSERT_EQ(static_cast<int>(state.victory_points), 0);

    const MicroAction pass(DecisionType::SELECT_CARD, 0, 0, action_flags::CONFIRM_DONE);
    ASSERT_TRUE(StateMachine::step(state, pass));
    ASSERT_EQ(static_cast<int>(state.victory_points), -3);
    ASSERT_FALSE(state.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
}

// Played normally, nothing changes: the debt is settled when the US plays a card.
TEST(ReviewFixes, WeWillBuryYouStillPaysWhenTheUsPlaysACard) {
    GameState state = empty_hands_action_round(37);
    state.china_card_holder = Player::USSR;
    state.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::USSR);
    state.card_locations[card_ids::FIDEL] = hand_of(Player::US);
    state.card_locations[card_ids::NASSER] = hand_of(Player::US);
    state.phasing_player = Player::USSR;
    state.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);

    StateMachine::advance_after_action_round(state);
    ASSERT_EQ(static_cast<int>(state.victory_points), 0);   // a card to play: not yet
    ASSERT_TRUE(StateMachine::step(state, MicroAction(DecisionType::SELECT_CARD, card_ids::FIDEL, 0, 0)));
    ASSERT_EQ(static_cast<int>(state.victory_points), -3);
    ASSERT_FALSE(state.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
}

// We Will Bury You taking DEFCON to 1 is a DEFCON-1 loss like every other: the phasing player
// loses, and when that is the US firing this USSR card off its Ops the loss was provoked.
TEST(ReviewFixes, WeWillBuryYouDefconOneLossIsClassifiedLikeTheOthers) {
    GameState state = empty_hands_action_round(33);
    state.phasing_player = Player::US;
    state.defcon = 2;
    state.clear_flag(effect_bits::DEFCON_SUICIDE_PROVOKED);

    CardHandlers::trigger_event(state, card_ids::WE_WILL_BURY_YOU, Player::USSR);

    ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
    ASSERT_EQ(static_cast<int>(state.victory_points), -20);
    ASSERT_TRUE(state.has_flag(effect_bits::DEFCON_SUICIDE_PROVOKED));
}

// A finished game's score does not move: Flower Power's 2 VP penalty on the war that just won
// the US the game at 20 would otherwise leave 18 under a GAME_OVER phase.
TEST(ReviewFixes, AwardVpLeavesAFinishedGameAlone) {
    GameState state = empty_hands_action_round(34);
    state.victory_points = 18;
    ASSERT_TRUE(award_vp(state, 2));
    ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
    ASSERT_FALSE(award_vp(state, -2));
    ASSERT_EQ(static_cast<int>(state.victory_points), 20);
}

// Special Relationship's 2 VP can win the game; a won game opens no placement decision.
TEST(ReviewFixes, SpecialRelationshipWinningVpOpensNoDecision) {
    GameState state = empty_hands_action_round(35);
    state.phasing_player = Player::US;
    state.set_flag(effect_bits::NATO_ACTIVE);
    state.countries[countries::UNITED_KINGDOM].us_influence = 5;
    state.countries[countries::UNITED_KINGDOM].ussr_influence = 0;
    state.victory_points = 18;
    state.ctx().decision_type = DecisionType::SELECT_PLAY_MODE;

    CardHandlers::trigger_event(state, card_ids::SPECIAL_RELATIONSHIP, Player::US);

    ASSERT_EQ(static_cast<int>(state.current_phase), static_cast<int>(Phase::GAME_OVER));
    ASSERT_EQ(static_cast<int>(state.victory_points), 20);
    ASSERT_NE(static_cast<int>(state.ctx().decision_type), static_cast<int>(DecisionType::POINT_NODE));
}

// Outside the flat space nothing is legal: not decoded into a confirm-done, not encoded as one.
TEST(ReviewFixes, OutOfRangeActionsAreRefusedNotAliasedToConfirm) {
    GameState state{};
    Engine::init_game(state, 36);
    const GameState before = state;
    ASSERT_FALSE(Engine::step_flat(state, static_cast<uint16_t>(FLAT_ACTION_SPACE_SIZE), false));
    ASSERT_FALSE(Engine::step_flat(state, 5000, false));
    ASSERT_TRUE(std::memcmp(&before, &state, sizeof(GameState)) == 0);

    ASSERT_EQ(static_cast<int>(ActionMask::decode_flat_action_212(state, 5000).decision_type),
              static_cast<int>(DecisionType::NONE));
    ASSERT_EQ(ActionMask::encode_micro_action_212(state, MicroAction(DecisionType::SELECT_CARD, 111, 0, 0)), -1);
    ASSERT_EQ(ActionMask::encode_micro_action_212(state, MicroAction(DecisionType::POINT_NODE, 84, 0, 0)), -1);
    ASSERT_EQ(ActionMask::encode_micro_action_212(state, MicroAction(DecisionType::SELECT_PLAY_MODE, 5, 0, 0)), -1);
    // A real confirm still encodes as one.
    ASSERT_EQ(ActionMask::encode_micro_action_212(
                  state, MicroAction(DecisionType::POINT_NODE, 255, 0, action_flags::CONFIRM_DONE)),
              static_cast<int>(flat_slots::CONFIRM_DONE));
}
