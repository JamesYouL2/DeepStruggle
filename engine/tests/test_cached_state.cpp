// CachedState: a GameState that keeps its own legal mask. These tests hold it to the plain engine
// -- the same states, masks and step verdicts, byte for byte -- and pin the two facts its design
// rests on: the mask does not read the RNG or the last-roll fields, and every change invalidates.
#include "test_framework.hpp"

#include <cstring>
#include <vector>

#include "ts/action_mask.hpp"
#include "ts/cached_state.hpp"
#include "ts/engine.hpp"

namespace {

uint64_t next(uint64_t& x) {
    x = x * 6364136223846793005ULL + 1442695040888963407ULL;
    return x >> 17;
}

std::vector<uint16_t> legal_actions(const uint8_t* mask) {
    std::vector<uint16_t> out;
    for (uint16_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a)
        if (mask[a]) out.push_back(a);
    return out;
}

bool same_state(const ts::GameState& a, const ts::GameState& b) {
    return std::memcmp(&a, &b, sizeof(ts::GameState)) == 0;
}

// Positions of whole random games, every decision, die rolls left in place.
std::vector<ts::GameState> random_positions(int games, uint64_t seed) {
    std::vector<ts::GameState> out;
    uint64_t r = seed;
    for (int g = 0; g < games; ++g) {
        ts::GameState s;
        ts::Engine::init_game(s, 5000 + static_cast<uint64_t>(g));
        for (int step = 0; step < 3000 && !ts::Engine::is_terminal(s); ++step) {
            out.push_back(s);
            uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
            ts::Engine::get_flat_action_mask(s, mask);
            const auto legal = legal_actions(mask);
            if (legal.empty()) break;
            if (!ts::Engine::step_flat(s, legal[next(r) % legal.size()], false)) break;
        }
    }
    return out;
}

struct ChecksOn {
    bool before = ts::mask_cache_checks();
    ChecksOn() { ts::set_mask_cache_checks(true); }
    ~ChecksOn() { ts::set_mask_cache_checks(before); }
};

}  // namespace

// StateMachine::step clears the three die-roll fields before it validates, and a search child gets
// a fresh RNG seed after it is copied from its parent: a mask generated before either is used for
// a state that differs from it in exactly those fields. So the mask must not read them.
TEST(CachedState, MaskIgnoresRngAndLastRolls) {
    const auto positions = random_positions(150, 11);
    ASSERT_GT(positions.size(), static_cast<size_t>(15000));
    uint64_t r = 99;
    for (const auto& s : positions) {
        uint8_t base[ts::FLAT_ACTION_SPACE_SIZE];
        ts::Engine::get_flat_action_mask(s, base);
        for (int variant = 0; variant < 3; ++variant) {
            ts::GameState t = s;
            t.rng_state = next(r) ^ (next(r) << 32);
            std::memset(&t.last_roll, static_cast<int>(next(r) & 0xff), sizeof(t.last_roll));
            t.last_die_roll = static_cast<uint8_t>(next(r) % 7);
            t.last_opp_die_roll = static_cast<uint8_t>(next(r) % 7);
            uint8_t other[ts::FLAT_ACTION_SPACE_SIZE];
            ts::Engine::get_flat_action_mask(t, other);
            ASSERT_TRUE(std::memcmp(base, other, sizeof(base)) == 0);
        }
    }
}

// A plain GameState and a CachedState play the same random games side by side -- legal actions,
// refused ones, auto-advance -- and must agree byte for byte at every step, with every cached
// read checked against a fresh mask.
TEST(CachedState, PlaysExactlyAsThePlainEngine) {
    ChecksOn checks;
    uint64_t r = 7;
    size_t steps = 0, refused = 0, cached_after_advance = 0;
    for (int game = 0; game < 100; ++game) {
        ts::GameState plain;
        ts::Engine::init_game(plain, 9000 + static_cast<uint64_t>(game));
        ts::CachedState cached(plain);
        for (int step = 0; step < 3000 && !ts::Engine::is_terminal(plain); ++step) {
            uint8_t mask[ts::FLAT_ACTION_SPACE_SIZE];
            ts::Engine::get_flat_action_mask(plain, mask);
            ASSERT_TRUE(std::memcmp(mask, cached.legal_mask(), sizeof(mask)) == 0);
            const auto legal = legal_actions(mask);
            if (legal.empty()) break;
            // One step in eight offers an action the mask does not: both must refuse it alike.
            uint16_t action = legal[next(r) % legal.size()];
            if (next(r) % 8 == 0) {
                for (uint16_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a)
                    if (!mask[a]) { action = a; break; }
            }
            const bool ok_plain = ts::Engine::step_flat(plain, action, false);
            const bool ok_cached = cached.step_flat(action);
            ASSERT_EQ(ok_plain, ok_cached);
            refused += !ok_plain;
            ASSERT_TRUE(same_state(plain, cached.state()));
            const size_t n_plain = ts::Engine::auto_advance_step(plain);
            const size_t n_cached = cached.auto_advance();
            ASSERT_EQ(n_plain, n_cached);
            ASSERT_TRUE(same_state(plain, cached.state()));
            cached_after_advance += cached.mask_cached();
            ++steps;
        }
    }
    ASSERT_GT(steps, static_cast<size_t>(10000));
    ASSERT_GT(refused, static_cast<size_t>(500));
    ASSERT_GT(cached_after_advance, steps / 2);
}

TEST(CachedState, EveryModificationInvalidates) {
    ChecksOn checks;
    const auto positions = random_positions(5, 3);
    for (size_t i = 0; i < positions.size(); i += 37) {
        ts::CachedState c(positions[i]);
        ASSERT_FALSE(c.mask_cached());
        c.legal_mask();
        ASSERT_TRUE(c.mask_cached());
        c.modify([](ts::GameState& g) { g.defcon = static_cast<uint8_t>(g.defcon == 2 ? 3 : 2); });
        ASSERT_FALSE(c.mask_cached());
        uint8_t fresh[ts::FLAT_ACTION_SPACE_SIZE];
        ts::Engine::get_flat_action_mask(c.state(), fresh);
        ASSERT_TRUE(std::memcmp(fresh, c.legal_mask(), sizeof(fresh)) == 0);
        // A refused step changes the die-roll record, so it invalidates too.
        uint16_t illegal = 0;
        while (illegal < ts::FLAT_ACTION_SPACE_SIZE && fresh[illegal]) ++illegal;
        if (illegal < ts::FLAT_ACTION_SPACE_SIZE) {
            ASSERT_FALSE(c.step_flat(illegal));
            ASSERT_FALSE(c.mask_cached());
        }
    }
}

TEST(CachedState, SetRngKeepsTheCache) {
    ChecksOn checks;
    const auto positions = random_positions(3, 5);
    for (size_t i = 0; i < positions.size(); i += 53) {
        ts::CachedState c(positions[i]);
        c.legal_mask();
        c.set_rng(0x1234567890abcdefULL + i);
        ASSERT_TRUE(c.mask_cached());
        c.legal_mask();   // checked against a fresh mask
        ASSERT_EQ(c.state().rng_state, 0x1234567890abcdefULL + i);
    }
}
