// The tree half of ai/search/batched_mcts.py, in C++: N independent PUCT searches advanced in
// lockstep, one leaf per tree per round, with the network left to Python.
//
// Why: profiled on the real workloads (research/log/search_performance_profile.md), a search
// spent ~90% of its time in Python bookkeeping -- PUCT over ~19 children per node, cloning and
// stepping new children, featurising leaves one at a time -- and ~7% in the network, at 6% GPU
// utilisation. The network still has to run in Python (torch, the model, its weights), so the
// split is: this class does everything between two forward passes, and a round costs Python two
// calls regardless of how many trees there are or how deep they go.
//
//     reset(roots, simulations, seed)
//     k = select_leaves()            # first call: the roots; then one new leaf per tree
//     while k: probs, values = net(obs[:k], masks[:k]); expand_and_backup(probs, values); k = select_leaves()
//
// The semantics are the Python searcher's, decision for decision -- the same PUCT expression in
// the same order, the same settle depths, values kept from the US side, a node with an empty mask
// treated as a draw -- so the Python searcher stays the reference that tests compare against.
// The one deliberate difference is randomness: each tree draws its chance seeds from its own
// SplitMix64 stream, seeded from `seed` and the tree's index, so a search is reproducible from its
// seed whatever the thread count, but does not replay the Python searcher's random.Random draws.
//
// Trees are independent, so descents, featurisation and backups run across the OpenMP team
// (gomp_parallel_for, the pool shared with torch). Leaf rows are then numbered in tree order, so
// the batch handed to the network is the same whatever the thread count.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "ts/action_mask.hpp"
#include "ts/engine.hpp"
#include "ts/game_state.hpp"
#include "ts/micro_action.hpp"
#include "ts/observation.hpp"
#include "ts/prng.hpp"
#include "gomp_parallel.hpp"

namespace ts_search {

struct Edge {
    uint16_t action;
    double prior;
    double n;
    double w;      // sum of backed-up values, US side
    int32_t child; // node index, -1 until first taken
};

struct Node {
    ts::GameState state;
    int8_t mover = 0;          // ts::Player of the side to move, 0 at a terminal
    bool terminal = false;
    bool expanded = false;     // priors filled in by a network evaluation
    double value_us = 0.0;
    double total = 0.0;        // sum of the edges' n
    uint32_t edge_begin = 0;
    uint32_t edge_count = 0;
};

struct Tree {
    std::vector<Node> nodes;   // nodes[0] is the root
    std::vector<Edge> edges;
    std::vector<std::pair<int32_t, int32_t>> path;  // (node, edge) of the pending descent
    uint64_t rng = 0;
    int64_t remaining = 0;
    int32_t leaf = -1;         // node awaiting evaluation, -1 if none
    int32_t row = -1;          // its row in the leaf buffers
};

inline ts::Player acting_player(const ts::GameState& s) {
    const ts::Player d = s.ctx().decision_player;
    return d != ts::Player::NONE ? d : s.phasing_player;
}

class BatchedSearch {
public:
    BatchedSearch(size_t capacity, double c_puct, bool auto_advance, uint32_t obs_features,
                  bool merged_influence)
        : capacity_(capacity), c_puct_(c_puct), auto_advance_(auto_advance),
          features_(obs_features), merged_(merged_influence),
          obs_width_(ts::OBS_SIZE_V23 + ts::obs_features::extra_width(obs_features)),
          obs_(capacity * obs_width_, 0.0f), masks_(capacity * ts::FLAT_ACTION_SPACE_SIZE, 0) {
        if (obs_features & ~ts::obs_features::ALL)
            throw std::invalid_argument("BatchedSearch: unknown observation feature bits");
    }

    size_t capacity() const { return capacity_; }
    size_t obs_width() const { return obs_width_; }
    size_t num_trees() const { return trees_.size(); }
    float* obs_data() { return obs_.data(); }
    uint8_t* mask_data() { return masks_.data(); }

    // Start one search per root. The roots are searched exactly as given (the caller settles and
    // determinizes them). `simulations` is each tree's own budget.
    void reset(const std::vector<ts::GameState>& roots, int64_t simulations, uint64_t seed) {
        if (roots.size() > capacity_)
            throw std::invalid_argument("BatchedSearch: " + std::to_string(roots.size()) +
                                        " roots exceed capacity " + std::to_string(capacity_));
        trees_.resize(roots.size());
        for (size_t i = 0; i < roots.size(); ++i) {
            Tree& t = trees_[i];
            t.nodes.clear();
            t.edges.clear();
            t.path.clear();
            // Each round adds at most one node, so this never reallocates during the search --
            // and the vectors keep their capacity from one reset to the next.
            t.nodes.reserve(static_cast<size_t>(simulations) + 1);
            uint64_t s = seed + 0x9E3779B97F4A7C15ULL * (static_cast<uint64_t>(i) + 1);
            t.rng = ts::Prng::next_u64(s);
            t.nodes.push_back(make_node(roots[i]));
            t.remaining = 0;
            t.leaf = -1;
            t.row = -1;
        }
        simulations_ = simulations;
        roots_phase_ = true;
    }

    // Write the observations and masks of the leaves awaiting evaluation into rows [0, k) and
    // return k. The first call after reset() returns the roots; after that, one new leaf per tree
    // still holding budget. Leaves that are terminal are backed up here without an evaluation, and
    // rounds are repeated until at least one leaf needs the network or every budget is spent.
    // 0 means the search is complete.
    size_t select_leaves() {
        size_t k = 0;
        if (roots_phase_) {
            for (Tree& t : trees_) {
                if (!t.nodes[0].terminal && !t.nodes[0].expanded) {
                    t.leaf = 0;
                    t.path.clear();
                    t.row = static_cast<int32_t>(k++);
                }
            }
            if (k > 0) {
                featurise_rows();
                return k;
            }
            begin_simulations();
        }
        while (true) {
            bool any_budget = false;
            for (const Tree& t : trees_) any_budget = any_budget || t.remaining > 0;
            if (!any_budget) return 0;
            for_each_tree([this](Tree& t) {
                t.leaf = -1;
                if (t.remaining <= 0) return;
                t.remaining -= 1;
                descend(t);
                const Node& leaf = t.nodes[static_cast<size_t>(t.leaf)];
                if (leaf.terminal) {
                    backup(t, leaf.value_us);
                    t.leaf = -1;
                }
            });
            for (Tree& t : trees_)
                if (t.leaf >= 0) t.row = static_cast<int32_t>(k++);
            if (k > 0) {
                featurise_rows();
                return k;
            }
        }
    }

    // Fill in the pending leaves from the network: `probs` is (k, FLAT_ACTION_SPACE_SIZE), the
    // softmax over the masked logits, and `values` is (k,), v_win from each leaf's mover's side.
    void expand_and_backup(const float* probs, const float* values, size_t k) {
        for (const Tree& t : trees_) {
            if (t.leaf >= 0 && (t.row < 0 || static_cast<size_t>(t.row) >= k))
                throw std::invalid_argument("expand_and_backup: fewer rows than pending leaves");
        }
        for_each_tree([&](Tree& t) {
            if (t.leaf < 0) return;
            const size_t r = static_cast<size_t>(t.row);
            expand(t, static_cast<size_t>(t.leaf), probs + r * ts::FLAT_ACTION_SPACE_SIZE,
                   &masks_[r * ts::FLAT_ACTION_SPACE_SIZE], values[r]);
            if (!roots_phase_) backup(t, t.nodes[static_cast<size_t>(t.leaf)].value_us);
            t.leaf = -1;
            t.row = -1;
        });
        if (roots_phase_) begin_simulations();
    }

    // Replace a root's priors (root noise is drawn by the caller, from its own stream).
    void set_root_priors(size_t i, const std::vector<double>& priors) {
        Tree& t = tree(i);
        const Node& root = t.nodes[0];
        if (priors.size() != root.edge_count)
            throw std::invalid_argument("set_root_priors: one prior per root action");
        for (uint32_t e = 0; e < root.edge_count; ++e) t.edges[root.edge_begin + e].prior = priors[e];
    }

    const Node& root(size_t i) { return tree(i).nodes[0]; }
    const Edge& root_edge(size_t i, size_t e) {
        Tree& t = tree(i);
        return t.edges[t.nodes[0].edge_begin + e];
    }
    size_t tree_size(size_t i) { return tree(i).nodes.size(); }

private:
    Tree& tree(size_t i) {
        if (i >= trees_.size()) throw std::out_of_range("BatchedSearch: no tree " + std::to_string(i));
        return trees_[i];
    }

    static Node make_node(const ts::GameState& s) {
        Node nd;
        nd.state = s;
        if (ts::Engine::is_terminal(s)) {
            nd.terminal = true;
            nd.expanded = true;
            nd.value_us = static_cast<double>(ts::Engine::get_terminal_utility(s));
        } else {
            nd.mover = static_cast<int8_t>(acting_player(s));
        }
        return nd;
    }

    void begin_simulations() {
        roots_phase_ = false;
        for (Tree& t : trees_) {
            const Node& r = t.nodes[0];
            t.remaining = (!r.terminal && r.edge_count > 0)
                ? std::max<int64_t>(0, simulations_ - static_cast<int64_t>(r.total)) : 0;
        }
    }

    // ai/search/batched_mcts.py `settle`: FORCED is the engine's own auto-advance, CHANCE drains
    // die rolls only.
    void settle(ts::GameState& s) const {
        if (auto_advance_) {
            ts::Engine::auto_advance_step(s);
            return;
        }
        while (!ts::Engine::is_terminal(s) && s.ctx().decision_player == ts::Player::NONE &&
               s.ctx().decision_type == ts::DecisionType::ROLL_DIE) {
            if (!ts::Engine::step(s, ts::MicroAction{ts::DecisionType::ROLL_DIE, 0, 0, 0})) break;
        }
    }

    // PUCT, read from the mover's side of a US-side value. The expression and its evaluation
    // order are the Python searcher's, so the choice is the same to the last bit.
    int32_t select(const Tree& t, const Node& nd) const {
        const double total = nd.total;
        const double sqrt_total = std::sqrt(total > 1.0 ? total : 1.0);
        const bool us_moves = nd.mover == static_cast<int8_t>(ts::Player::US);
        int32_t best_i = 0;
        double best_v = -1e30;
        for (uint32_t i = 0; i < nd.edge_count; ++i) {
            const Edge& e = t.edges[nd.edge_begin + i];
            double q = e.n > 0 ? e.w / e.n : nd.value_us;
            if (!us_moves) q = -q;
            const double v = q + c_puct_ * e.prior * sqrt_total / (1.0 + e.n);
            if (v > best_v) {
                best_v = v;
                best_i = static_cast<int32_t>(i);
            }
        }
        return best_i;
    }

    void descend(Tree& t) {
        t.path.clear();
        size_t node = 0;
        while (true) {
            const Node& nd = t.nodes[node];
            if (nd.terminal || !nd.expanded) {
                t.leaf = static_cast<int32_t>(node);
                return;
            }
            const int32_t idx = select(t, nd);
            const uint32_t edge = nd.edge_begin + static_cast<uint32_t>(idx);
            t.path.emplace_back(static_cast<int32_t>(node), static_cast<int32_t>(edge));
            const int32_t child = t.edges[edge].child;
            if (child >= 0) {
                node = static_cast<size_t>(child);
                continue;
            }
            ts::GameState next = nd.state;
            next.rng_state = ts::Prng::next_u64(t.rng);
            const uint16_t action = t.edges[edge].action;
            if (!ts::Engine::step_flat(next, action, false, merged_)) {
                throw std::runtime_error("BatchedSearch: the engine refused flat action " +
                                         std::to_string(action) + " taken from its own mask");
            }
            settle(next);
            t.nodes.push_back(make_node(next));
            const int32_t created = static_cast<int32_t>(t.nodes.size() - 1);
            t.edges[edge].child = created;
            t.leaf = created;
            return;
        }
    }

    static void backup(Tree& t, double value_us) {
        for (const auto& [node, edge] : t.path) {
            Edge& e = t.edges[static_cast<size_t>(edge)];
            e.n += 1.0;
            e.w += value_us;
            t.nodes[static_cast<size_t>(node)].total += 1.0;
        }
    }

    void expand(Tree& t, size_t node, const float* probs, const uint8_t* mask, float value) {
        Node& nd = t.nodes[node];
        uint32_t k = 0;
        double total = 0.0;
        for (size_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a) {
            if (mask[a]) {
                ++k;
                total += static_cast<double>(probs[a]);
            }
        }
        if (k == 0) {
            nd.terminal = true;
            nd.value_us = 0.0;
            nd.expanded = true;
            return;
        }
        nd.edge_begin = static_cast<uint32_t>(t.edges.size());
        nd.edge_count = k;
        const bool uniform = !(total > 1e-12);
        for (size_t a = 0; a < ts::FLAT_ACTION_SPACE_SIZE; ++a) {
            if (!mask[a]) continue;
            const double p = uniform ? 1.0 / static_cast<double>(k)
                                     : static_cast<double>(probs[a]) / total;
            t.edges.push_back(Edge{static_cast<uint16_t>(a), p, 0.0, 0.0, -1});
        }
        nd.total = 0.0;
        const double v = static_cast<double>(value);
        nd.value_us = nd.mover == static_cast<int8_t>(ts::Player::US) ? v : -v;
        nd.expanded = true;
    }

    // `body(tree)` for every tree across the OpenMP team. An exception cannot cross the team, so
    // the first one is kept and rethrown here once every thread is done.
    template <class F>
    void for_each_tree(F&& body) {
        std::exception_ptr err;
        std::mutex mu;
        gomp_parallel_for(static_cast<int64_t>(trees_.size()), [&](int64_t i) {
            try {
                body(trees_[static_cast<size_t>(i)]);
            } catch (...) {
                std::lock_guard<std::mutex> lock(mu);
                if (!err) err = std::current_exception();
            }
        });
        if (err) std::rethrow_exception(err);
    }

    void featurise_rows() {
        for_each_tree([this](Tree& t) {
            if (t.leaf < 0) return;
            const Node& nd = t.nodes[static_cast<size_t>(t.leaf)];
            const size_t r = static_cast<size_t>(t.row);
            float* row = &obs_[r * obs_width_];
            const size_t w = ts::extract_observation_features(
                nd.state, acting_player(nd.state), features_, row);
            if (w < obs_width_) std::memset(row + w, 0, (obs_width_ - w) * sizeof(float));
            ts::Engine::get_flat_action_mask(nd.state, &masks_[r * ts::FLAT_ACTION_SPACE_SIZE],
                                             merged_);
        });
    }

    size_t capacity_;
    double c_puct_;
    bool auto_advance_;
    uint32_t features_;
    bool merged_;
    size_t obs_width_;
    std::vector<float> obs_;
    std::vector<uint8_t> masks_;
    std::vector<Tree> trees_;
    int64_t simulations_ = 0;
    bool roots_phase_ = false;
};

}  // namespace ts_search
