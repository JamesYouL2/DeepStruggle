// Regression tests for the rules audit against the card text, the Deluxe rulebook, the FAQ and
// the struggler engine's rulings table (docs/RULES_SOURCES.md there). One test per defect, each
// written to fail on the engine as it stood before the fix.
#include "test_framework.hpp"
#include "ts/engine.hpp"
#include "ts/state_machine.hpp"
#include "ts/action_mask.hpp"
#include "ts/card_handlers.hpp"
#include "ts/card_data.hpp"
#include "ts/map_data.hpp"
#include "ts/scoring.hpp"
#include "ts/ops.hpp"

using namespace ts;

namespace {

// A US or USSR action round, nothing in either hand. The caller deals what it needs.
GameState action_round_for(Player p) {
    GameState state{};
    Engine::init_game(state, 42);
    for (uint8_t c = 1; c <= 110; ++c) {
        if (is_in_any_hand(state.card_locations[c])) state.card_locations[c] = CardLocation::DRAW_DECK;
    }
    state.current_phase = Phase::ACTION_ROUND;
    state.action_round = 2;
    state.turn = 4;
    state.phasing_player = p;
    state.ctx() = DecisionContext{};
    state.ctx().decision_player = p;
    state.ctx().decision_type = DecisionType::SELECT_CARD;
    return state;
}

// Resolve a war event against `target` with the die forced to `roll`.
void fight(GameState& state, uint8_t card, Player attacker, uint8_t target, uint8_t roll) {
    ASSERT_FALSE(CardHandlers::trigger_event(state, card, attacker));
    ASSERT_FALSE(CardHandlers::handle_event_step(
        state, MicroAction{DecisionType::POINT_NODE, target, 0, 0}));
    ASSERT_TRUE(CardHandlers::handle_event_step(
        state, MicroAction{DecisionType::ROLL_DIE, roll, 0, 0}));
}

} // namespace

// --- War rolls: the defender's superpower is an adjacent controlled country -------------------

TEST(RulesAudit, BrushWarOnMexico_UnitedStatesCountsAgainstTheUSSR) {
    // Mexico's only neighbours are Guatemala and the United States. With Guatemala uncontrolled
    // the one -1 is the superpower's: a 3 fails where it used to succeed, and a 4 succeeds.
    GameState s{};
    s.countries[countries::MEXICO].us_influence = 1;
    fight(s, card_ids::BRUSH_WAR, Player::USSR, countries::MEXICO, 3);
    ASSERT_EQ(s.victory_points, 0);
    ASSERT_EQ(s.countries[countries::MEXICO].us_influence, 1);
    ASSERT_EQ(s.last_roll.mod1, -1);

    GameState t{};
    t.countries[countries::MEXICO].us_influence = 1;
    fight(t, card_ids::BRUSH_WAR, Player::USSR, countries::MEXICO, 4);
    ASSERT_EQ(t.victory_points, -1);
    ASSERT_EQ(t.countries[countries::MEXICO].ussr_influence, 1);
}

TEST(RulesAudit, BrushWarOnAfghanistan_SovietUnionCountsAgainstTheUS) {
    GameState s{};
    s.countries[countries::AFGHANISTAN].ussr_influence = 1;
    fight(s, card_ids::BRUSH_WAR, Player::US, countries::AFGHANISTAN, 3);
    ASSERT_EQ(s.victory_points, 0);
    ASSERT_EQ(s.last_roll.mod1, -1);
}

TEST(RulesAudit, WarRoll_OwnSuperpowerGivesTheAttackerNothing) {
    // Only the defender's superpower counts: the USSR attacking Afghanistan has no penalty.
    GameState s{};
    s.countries[countries::AFGHANISTAN].us_influence = 1;
    fight(s, card_ids::BRUSH_WAR, Player::USSR, countries::AFGHANISTAN, 3);
    ASSERT_EQ(s.last_roll.mod1, 0);
    ASSERT_EQ(s.victory_points, -1);
}

// --- UN Intervention needs an opponent-associated companion to be played as an Event ----------

TEST(RulesAudit, UNIntervention_NotAnEventWithoutACompanion) {
    GameState s = action_round_for(Player::US);
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    s.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::US);   // a US card: no companion
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ASSERT_EQ(s.ctx().decision_type, DecisionType::SELECT_PLAY_MODE);

    uint8_t mask[FLAT_ACTION_SPACE_SIZE];
    ActionMask::generate_flat_mask_212(s, mask);
    ASSERT_EQ(mask[flat_slots::RESOLUTION + static_cast<size_t>(Resolution::EVENT)], 0);
    ASSERT_EQ(mask[flat_slots::OPS_INFLUENCE], 1);

    // With a USSR card in hand it is an Event again.
    GameState t = action_round_for(Player::US);
    t.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    t.card_locations[card_ids::DE_GAULLE] = hand_of(Player::US);
    ASSERT_TRUE(Engine::step(t, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ActionMask::generate_flat_mask_212(t, mask);
    ASSERT_EQ(mask[flat_slots::RESOLUTION + static_cast<size_t>(Resolution::EVENT)], 1);
}

TEST(RulesAudit, UNIntervention_WithoutACompanionCannotDodgeWeWillBuryYou) {
    // The exploit the fizzle allowed: spend the named round on an empty UN Intervention and the
    // 3 VP never came. Playing it for Operations is all that is left, and that pays.
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    s.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::US);
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ASSERT_FALSE(Engine::step(s, MicroAction{DecisionType::SELECT_PLAY_MODE,
                                             static_cast<uint8_t>(Resolution::EVENT), 0, 0}));
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_PLAY_MODE,
                                            static_cast<uint8_t>(Resolution::OPS_INFLUENCE), 0, 0}));
    ASSERT_EQ(s.victory_points, -3);
    ASSERT_FALSE(s.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
}

// --- We Will Bury You on a trapped round -------------------------------------------------------

TEST(RulesAudit, WeWillBuryYou_SettlesOnAQuagmireDiscardRound) {
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::QUAGMIRE_ACTIVE);
    s.set_flag(effect_bits::WE_WILL_BURY_YOU_PENDING);
    s.card_locations[card_ids::DUCK_AND_COVER] = hand_of(Player::US);   // 3 Ops, a legal discard
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_CARD, card_ids::DUCK_AND_COVER, 0, 0}));
    // The round is spent on the discard, so UN Intervention was not played in it: 3 VP now,
    // not in whatever later round the US next plays a card.
    ASSERT_EQ(s.victory_points, -3);
    ASSERT_FALSE(s.has_flag(effect_bits::WE_WILL_BURY_YOU_PENDING));
    ASSERT_EQ(s.ctx().decision_type, DecisionType::ROLL_DIE);
}

// --- NORAD arms on a move to DEFCON 2, not on a setting of 2 at 2 -----------------------------

TEST(RulesAudit, CubanMissileCrisisAtDefcon2_DoesNotArmNorad) {
    GameState s{};
    s.defcon = 2;
    CardHandlers::trigger_event(s, card_ids::CUBAN_MISSILE_CRISIS, Player::USSR);
    ASSERT_EQ(s.defcon, 2);
    ASSERT_EQ(s.defcon_dropped_to_2, 0);

    GameState t{};
    t.defcon = 4;
    CardHandlers::trigger_event(t, card_ids::CUBAN_MISSILE_CRISIS, Player::USSR);
    ASSERT_EQ(t.defcon_dropped_to_2, 1);
}

TEST(RulesAudit, HowILearnedSettingTwoAtTwo_DoesNotArmNorad) {
    GameState s{};
    s.defcon = 2;
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::HOW_I_LEARNED_TO_STOP_WORRYING, Player::US));
    ASSERT_TRUE(CardHandlers::handle_event_step(
        s, MicroAction{DecisionType::CHOOSE_BRANCH, 2, 0, action_flags::DEFCON_VALUE}));
    ASSERT_EQ(s.defcon_dropped_to_2, 0);

    GameState t{};
    t.defcon = 4;
    ASSERT_FALSE(CardHandlers::trigger_event(t, card_ids::HOW_I_LEARNED_TO_STOP_WORRYING, Player::US));
    ASSERT_TRUE(CardHandlers::handle_event_step(
        t, MicroAction{DecisionType::CHOOSE_BRANCH, 2, 0, action_flags::DEFCON_VALUE}));
    ASSERT_EQ(t.defcon_dropped_to_2, 1);
}

// --- U-2 Incident's rider on the Grain Sales route ---------------------------------------------

TEST(RulesAudit, GrainSalesUNIntervention_PaysTheU2Rider) {
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::U2_INCIDENT_ACTIVE);
    s.card_locations[card_ids::DE_GAULLE] = hand_of(Player::USSR);   // the only card Grain Sales can draw
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    s.ctx().decision_player = Player::US;
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US));
    ASSERT_EQ(s.ctx().pending_op_card, card_ids::DE_GAULLE);
    ASSERT_FALSE(CardHandlers::handle_event_step(
        s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ASSERT_EQ(s.victory_points, -1);
    ASSERT_EQ(s.ctx().decision_type, DecisionType::SELECT_OP_MODE);
}

TEST(RulesAudit, GrainSalesUNIntervention_OnAWarCardChargesFlowerPower) {
    GameState s = action_round_for(Player::US);
    s.set_flag(effect_bits::FLOWER_POWER_ACTIVE);
    s.card_locations[card_ids::KOREAN_WAR] = hand_of(Player::USSR);
    s.card_locations[card_ids::UN_INTERVENTION] = hand_of(Player::US);
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::GRAIN_SALES, Player::US));
    ASSERT_FALSE(CardHandlers::handle_event_step(
        s, MicroAction{DecisionType::SELECT_CARD, card_ids::UN_INTERVENTION, 0, 0}));
    ASSERT_EQ(s.victory_points, -2);
}

// --- Era entry does not reshuffle the discard pile ---------------------------------------------

TEST(RulesAudit, MidWarEntry_LeavesTheDiscardPileAlone) {
    GameState s{};
    Engine::init_game(s, 7);
    s.current_phase = Phase::ACTION_ROUND;
    s.turn = 3;
    // Empty both hands -- a scoring card still held would end the game at the turn's end.
    for (uint8_t c = 1; c <= 110; ++c) {
        if (is_in_any_hand(s.card_locations[c])) s.card_locations[c] = CardLocation::DRAW_DECK;
    }
    // Spend a few Early War cards.
    uint8_t discarded = 0;
    for (uint8_t c = 1; c <= 110; ++c) {
        if (s.card_locations[c] == CardLocation::DRAW_DECK && discarded < 5) {
            s.card_locations[c] = CardLocation::DISCARD_PILE;
            discarded++;
        }
    }
    ASSERT_EQ(discarded, 5);
    GameState before = s;

    StateMachine::finish_end_turn(s);   // turn 3 -> 4: Mid War enters, then the deal
    ASSERT_EQ(s.turn, 4);
    for (uint8_t c = 1; c <= 110; ++c) {
        if (before.card_locations[c] == CardLocation::DISCARD_PILE) {
            ASSERT_EQ(s.card_locations[c], CardLocation::DISCARD_PILE);
        }
    }
    // And the Mid War is in play.
    uint8_t mid_in_play = 0;
    for (uint8_t c = 1; c <= 110; ++c) {
        if (CardData::get_card(c).era == WarEra::MID &&
            (s.card_locations[c] == CardLocation::DRAW_DECK || is_in_any_hand(s.card_locations[c]))) {
            mid_in_play++;
        }
    }
    ASSERT_GT(mid_in_play, 0);
}

// --- Summit counts regions without the scoring-only card effects -------------------------------

namespace {

// USSR controls three of the six Middle East battlegrounds (Iraq, Libya, Egypt) and Syria; the
// US controls two battlegrounds (Israel, Iran) and Jordan. USSR Domination, 4 countries to 3 --
// unless Shuttle Diplomacy takes a battleground and a country off its count, which leaves 3 to 3
// and no one dominating.
GameState middle_east_on_a_knife_edge() {
    GameState s{};
    auto ussr = [&](uint8_t c) { s.countries[c].ussr_influence = MapData::get_country(c).stability; };
    auto us = [&](uint8_t c) { s.countries[c].us_influence = MapData::get_country(c).stability; };
    ussr(countries::SYRIA); ussr(countries::IRAQ); ussr(countries::LIBYA); ussr(countries::EGYPT);
    us(countries::ISRAEL); us(countries::IRAN); us(countries::JORDAN);
    return s;
}

} // namespace

TEST(RulesAudit, Summit_IgnoresShuttleDiplomacy) {
    GameState s = middle_east_on_a_knife_edge();
    ASSERT_TRUE(Scoring::dominates_or_controls(s, Region::MIDDLE_EAST, Player::USSR));
    s.set_flag(effect_bits::SHUTTLE_DIPLOMACY_ACTIVE);
    // A scoring would drop the USSR out of Domination...
    ASSERT_EQ(Scoring::evaluate_region(s, Region::MIDDLE_EAST).ussr_status, RegionalStatus::PRESENCE);
    // ...Summit does not.
    ASSERT_TRUE(Scoring::dominates_or_controls(s, Region::MIDDLE_EAST, Player::USSR));

    // And through the event: dice 3 each, the USSR's one region decides it.
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::SUMMIT, Player::US));
    ASSERT_FALSE(CardHandlers::handle_event_step(s, MicroAction{DecisionType::ROLL_DIE, 3, 3, 0}));
    ASSERT_EQ(s.victory_points, -2);
    ASSERT_EQ(s.ctx().decision_player, Player::USSR);
}

TEST(RulesAudit, Summit_IgnoresFormosanResolution) {
    // Asia with Taiwan US-controlled: under Formosan Resolution a scoring counts Taiwan as a
    // battleground. The US holds three countries and two battlegrounds to the USSR's two and
    // one; with Taiwan a battleground that is 3 to 1 either way, so pick a split Taiwan tips.
    GameState s{};
    auto us = [&](uint8_t c) { s.countries[c].us_influence = MapData::get_country(c).stability; };
    auto ussr = [&](uint8_t c) { s.countries[c].ussr_influence = MapData::get_country(c).stability; };
    // US: Taiwan (non-BG), Japan (BG). USSR: North Korea (BG), Afghanistan (non-BG).
    us(countries::TAIWAN); us(countries::JAPAN);
    ussr(countries::NORTH_KOREA); ussr(countries::AFGHANISTAN);
    // US adds a third country so it has more countries: Australia (non-BG).
    us(countries::AUSTRALIA);
    // Without Formosan: US 3 countries, 1 BG; USSR 2 countries, 1 BG -> no US Domination.
    ASSERT_FALSE(Scoring::dominates_or_controls(s, Region::ASIA, Player::US));
    s.set_flag(effect_bits::FORMOSAN_RESOLUTION_ACTIVE);
    // A scoring counts Taiwan: 2 BGs to 1 -> Domination.
    ASSERT_EQ(Scoring::evaluate_region(s, Region::ASIA).us_status, RegionalStatus::DOMINATION);
    // Summit does not.
    ASSERT_FALSE(Scoring::dominates_or_controls(s, Region::ASIA, Player::US));
}

// --- Event Ops and the Vietnam Revolts ladder --------------------------------------------------

TEST(RulesAudit, OlympicBoycottUnderVietnamRevolts_KeepsFourOutsideSoutheastAsia) {
    // The USSR sponsors, the US boycotts: 4 Ops, +1 while they stay in Southeast Asia. The
    // ladder fell back to the Olympic Games' printed 2 on the first placement outside, so the
    // USSR lost two of the four Ops the event grants.
    GameState s = action_round_for(Player::USSR);
    s.defcon = 4;
    s.set_flag(effect_bits::VIETNAM_REVOLTS_ACTIVE);
    s.ctx().decision_player = Player::USSR;
    ASSERT_FALSE(CardHandlers::trigger_event(s, card_ids::OLYMPIC_GAMES, Player::USSR));
    ASSERT_FALSE(CardHandlers::handle_event_step(s, MicroAction{DecisionType::CHOOSE_BRANCH, 1, 0, 0}));
    ASSERT_EQ(s.ctx().pending_ops_value, 5);
    ASSERT_EQ(s.ctx().decision_type, DecisionType::SELECT_OP_MODE);
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::SELECT_OP_MODE,
                                            static_cast<uint8_t>(OpMode::INFLUENCE), 0, 0}));
    // East Germany has USSR Influence from setup.
    ASSERT_TRUE(Engine::step(s, MicroAction{DecisionType::POINT_NODE, countries::EAST_GERMANY, 0, 0}));
    ASSERT_EQ(s.ctx().pending_ops_value, 4);
    ASSERT_EQ(s.ctx().remaining_steps, 3);
}
