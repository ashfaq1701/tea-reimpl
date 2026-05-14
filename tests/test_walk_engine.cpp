// Phase 6.1: WalkEngine — parallel-walk orchestration correctness.
//
// Walks are BACKWARD-IN-TIME: each step picks a t_k strictly less than
// t_{k-1}.  Slot 0 holds the start vertex with t = kSentinelStartTimestamp
// (= INT64_MAX) so the first hop has the full candidate set.
//
// What we verify:
//   • WalkRunStats fields (num_walks, total_steps, dead_at_start) match
//     a manual recount from walk_lens_out.
//   • All emitted walks are temporally valid:
//       - Slot 0 = (start_vertex, kSentinelStartTimestamp).
//       - For step k > 0: (v_k, t_k) is a real out-edge of v_{k-1}
//         with t_k < t_{k-1} (or t_{k-1} == sentinel for k=1).
//   • Walks that die early are sentinel-padded (v == kWalkDeadSentinel).
//   • Determinism: same global_seed → byte-identical walks_out, even
//     across different OpenMP thread counts. (Per-walk RNG seeded from
//     (global_seed, walk_idx) → independent of schedule.)
//   • Dead-start detection: vertices with degree=0 produce walks of len=1.
//   • Both variants (PAT, HPAT) and all four sampler paths exercised.
//   • Node2Vec variant: same invariants plus each accepted step matches
//     a real out-edge of v_{k-1}.
//
// This is the integration test for the engine — sampler correctness lives
// in test_sampler.cpp / test_hpat.cpp / test_node2vec.cpp.

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <unordered_set>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/neighbor_set.hpp"
#include "tea/node2vec_bias.hpp"
#include "tea/pat.hpp"
#include "tea/walk_engine.hpp"

namespace {

// Small but non-trivial graph:
//   v=0..9 each emit a fan of K edges into v=10..19+ with distinct timestamps.
//   v=20 is a dead-end vertex (no outgoing edges). Used to test dead-at-start.
tea::TemporalGraph make_fanout_graph() {
    std::vector<tea::Edge> edges;
    // Vertices 0..9 → each has 20 outgoing edges with timestamps 100..119.
    // To make multi-hop walks possible, every interior vertex also has edges
    // forward (so the graph has a long-running time axis).
    for (int u = 0; u < 30; ++u) {
        for (int j = 0; j < 20; ++j) {
            // edge timestamps spread across (100 + 30*u .. 119 + 30*u)
            int32_t v = (u + j + 1) % 30;
            int64_t t = 100 + static_cast<int64_t>(u) * 30 + j;
            edges.push_back({u, v, t});
        }
    }
    // v=30: dead-end (no outgoing edges)
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/31, /*is_directed=*/true);
    return g;
}

// Verify (v, t) is a valid out-edge of u with t < t_prev (backward walks).
// Returns matching edge index or -1 if no match.  Slow O(degree) — fine for
// tests.
int32_t find_edge(const tea::TemporalGraph& g, int32_t u, int32_t v,
                  int64_t t, int64_t t_prev) {
    const auto tgt = g.targets_of(u);
    const auto ts  = g.timestamps_of(u);
    for (std::size_t i = 0; i < tgt.size(); ++i) {
        if (tgt[i] == v && ts[i] == t && ts[i] < t_prev) {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

void assert_walks_valid(const tea::TemporalGraph& g,
                        const int32_t*  starts,
                        const tea::NodeStep* walks_out,
                        const int32_t*  walk_lens_out,
                        int32_t         num_walks,
                        int32_t         max_walk_len) {
    for (int32_t i = 0; i < num_walks; ++i) {
        const tea::NodeStep* slots = walks_out + static_cast<int64_t>(i) * max_walk_len;
        const int32_t        len   = walk_lens_out[i];

        ASSERT_GE(len, 1);
        ASSERT_LE(len, max_walk_len);
        ASSERT_EQ(slots[0].v, starts[i]) << "walk " << i << " slot 0 mismatch";
        ASSERT_EQ(slots[0].t, tea::kSentinelStartTimestamp);

        int64_t t_prev = tea::kSentinelStartTimestamp;
        int32_t u_prev = starts[i];
        for (int32_t k = 1; k < len; ++k) {
            const int32_t v_next = slots[k].v;
            const int64_t t_next = slots[k].t;
            const int32_t edge_idx = find_edge(g, u_prev, v_next, t_next, t_prev);
            ASSERT_GE(edge_idx, 0) << "walk " << i << " step " << k
                << " has no matching edge from u=" << u_prev
                << " to v=" << v_next << " at t=" << t_next
                << " (t_prev=" << t_prev << ")";
            u_prev = v_next;
            t_prev = t_next;
        }

        // Padding: slots after walk_len must be sentinel.
        for (int32_t k = len; k < max_walk_len; ++k) {
            ASSERT_EQ(slots[k].v, tea::kWalkDeadSentinel)
                << "walk " << i << " slot " << k << " not sentinel-padded";
        }
    }
}

void assert_stats_consistent(const tea::WalkRunStats& stats,
                             const int32_t*           walk_lens_out,
                             int32_t                  num_walks) {
    EXPECT_EQ(stats.num_walks, num_walks);

    int64_t expected_steps = 0;
    int64_t expected_dead  = 0;
    for (int32_t i = 0; i < num_walks; ++i) {
        expected_steps += walk_lens_out[i];
        if (walk_lens_out[i] <= 1) ++expected_dead;
    }
    EXPECT_EQ(stats.total_steps,   expected_steps);
    EXPECT_EQ(stats.dead_at_start, expected_dead);
}

}  // namespace

// -----------------------------------------------------------------------------
// 1. PAT + UniformBias — basic end-to-end + stats correctness
// -----------------------------------------------------------------------------

TEST(WalkEngine, PatUniformBasic) {
    auto g = make_fanout_graph();
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    constexpr int32_t  N = 200;
    constexpr int32_t  L = 12;
    constexpr uint64_t SEED = 0xdeadbeefULL;

    std::vector<int32_t>  starts(N);
    for (int32_t i = 0; i < N; ++i) starts[i] = i % 30;  // visit all real vertices

    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    auto stats = tea::run_walks_pat(g, pat, tea::UniformBias{},
                                    starts.data(), N, L, SEED,
                                    out.data(), lens.data());

    assert_stats_consistent(stats, lens.data(), N);
    assert_walks_valid(g, starts.data(), out.data(), lens.data(), N, L);
}

// -----------------------------------------------------------------------------
// 2. HPAT + LinearBias — sampler-variant coverage
// -----------------------------------------------------------------------------

TEST(WalkEngine, HpatLinearBasic) {
    auto g = make_fanout_graph();
    tea::Hpat<tea::LinearBias> hpat;
    hpat.build(g, tea::LinearBias{});

    constexpr int32_t  N = 200;
    constexpr int32_t  L = 12;
    constexpr uint64_t SEED = 0xc0ffeeULL;

    std::vector<int32_t> starts(N);
    for (int32_t i = 0; i < N; ++i) starts[i] = i % 30;

    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    auto stats = tea::run_walks_hpat(g, hpat, tea::LinearBias{},
                                     starts.data(), N, L, SEED,
                                     out.data(), lens.data());

    assert_stats_consistent(stats, lens.data(), N);
    assert_walks_valid(g, starts.data(), out.data(), lens.data(), N, L);
}

// -----------------------------------------------------------------------------
// 3. Dead-at-start: a degree-0 vertex must produce walks of length 1
// -----------------------------------------------------------------------------

TEST(WalkEngine, DeadAtStartCounted) {
    auto g = make_fanout_graph();
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    // Mix live and dead start vertices. v=30 has no outgoing edges.
    const std::vector<int32_t> starts = {0, 30, 5, 30, 30, 12, 7};
    const int32_t N = static_cast<int32_t>(starts.size());
    constexpr int32_t L = 8;

    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    auto stats = tea::run_walks_pat(g, pat, tea::UniformBias{},
                                    starts.data(), N, L, 0xfeed'face'beefULL,
                                    out.data(), lens.data());

    // Each v=30 walk is dead-at-start (len == 1, only the start slot written).
    int64_t dead_expected = 0;
    for (int32_t s : starts) if (s == 30) ++dead_expected;
    EXPECT_EQ(stats.dead_at_start, dead_expected);

    assert_walks_valid(g, starts.data(), out.data(), lens.data(), N, L);
}

// -----------------------------------------------------------------------------
// 4. Determinism across thread counts (the key WalkEngine invariant)
//    Per-walk RNG seeded from (global_seed, walk_idx) → schedule-independent.
// -----------------------------------------------------------------------------

TEST(WalkEngine, DeterministicAcrossThreadCounts) {
#ifndef _OPENMP
    GTEST_SKIP() << "Determinism-across-threads test needs OpenMP";
#else
    auto g = make_fanout_graph();
    tea::Pat<tea::ExponentialBias> pat;
    tea::ExponentialBias bias;  // proper-TEA default
    pat.build(g, bias);

    constexpr int32_t  N = 500;
    constexpr int32_t  L = 16;
    constexpr uint64_t SEED = 0x1234'5678'9abcULL;

    std::vector<int32_t> starts(N);
    for (int32_t i = 0; i < N; ++i) starts[i] = i % 30;

    auto run = [&](int threads) {
        omp_set_num_threads(threads);
        std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
        std::vector<int32_t>       lens(N);
        tea::run_walks_pat(g, pat, bias, starts.data(), N, L, SEED,
                           out.data(), lens.data());
        return std::pair{std::move(out), std::move(lens)};
    };

    auto [out1, lens1]   = run(1);
    auto [out4, lens4]   = run(4);
    auto [out16, lens16] = run(16);

    ASSERT_EQ(lens1, lens4);
    ASSERT_EQ(lens1, lens16);

    // Compare byte-by-byte (NodeStep is trivially copyable: int32 + int64).
    for (std::size_t i = 0; i < out1.size(); ++i) {
        ASSERT_EQ(out1[i].v, out4[i].v)  << "thread 1 vs 4 mismatch at slot " << i;
        ASSERT_EQ(out1[i].t, out4[i].t)  << "thread 1 vs 4 mismatch at slot " << i;
        ASSERT_EQ(out1[i].v, out16[i].v) << "thread 1 vs 16 mismatch at slot " << i;
        ASSERT_EQ(out1[i].t, out16[i].t) << "thread 1 vs 16 mismatch at slot " << i;
    }
#endif
}

// -----------------------------------------------------------------------------
// 5. PAT vs HPAT produce the SAME walks under the same seed
//    (Both go through the same per-walk RNG; sampler is deterministic given
//    rng state. PAT and HPAT differ in trunk layout but should pick the same
//    edge given the same uniform draws... ACTUALLY they don't, because the
//    internal sampler call patterns differ. So we only test that both produce
//    *valid* walks; same-seed equivalence is NOT a real invariant.)
//
//    Instead: test that re-running with the same seed produces identical
//    walks (reproducibility) within the same variant.
// -----------------------------------------------------------------------------

TEST(WalkEngine, RunReproducibleSameSeed) {
    auto g = make_fanout_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    constexpr int32_t  N = 300;
    constexpr int32_t  L = 10;
    constexpr uint64_t SEED = 42ULL;

    std::vector<int32_t> starts(N);
    for (int32_t i = 0; i < N; ++i) starts[i] = i % 30;

    auto run = [&]() {
        std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
        std::vector<int32_t>       lens(N);
        tea::run_walks_hpat(g, hpat, tea::UniformBias{},
                            starts.data(), N, L, SEED,
                            out.data(), lens.data());
        return std::pair{std::move(out), std::move(lens)};
    };

    auto [a_out, a_lens] = run();
    auto [b_out, b_lens] = run();

    ASSERT_EQ(a_lens, b_lens);
    for (std::size_t i = 0; i < a_out.size(); ++i) {
        ASSERT_EQ(a_out[i].v, b_out[i].v);
        ASSERT_EQ(a_out[i].t, b_out[i].t);
    }
}

// -----------------------------------------------------------------------------
// 6. Different seeds produce different walks (sanity check the seed plumbing)
// -----------------------------------------------------------------------------

TEST(WalkEngine, DifferentSeedDifferentWalks) {
    auto g = make_fanout_graph();
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    constexpr int32_t N = 100;
    constexpr int32_t L = 12;

    std::vector<int32_t> starts(N);
    for (int32_t i = 0; i < N; ++i) starts[i] = i % 30;

    std::vector<tea::NodeStep> out_a(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens_a(N);
    tea::run_walks_pat(g, pat, tea::UniformBias{}, starts.data(), N, L,
                       /*seed=*/1ULL, out_a.data(), lens_a.data());

    std::vector<tea::NodeStep> out_b(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens_b(N);
    tea::run_walks_pat(g, pat, tea::UniformBias{}, starts.data(), N, L,
                       /*seed=*/2ULL, out_b.data(), lens_b.data());

    // Should differ in at least one slot (with N*L=1200 slots, the chance of
    // accidental match under two independent RNG streams is astronomically low).
    bool any_diff = false;
    for (std::size_t i = 0; i < out_a.size(); ++i) {
        if (out_a[i].v != out_b[i].v || out_a[i].t != out_b[i].t) {
            any_diff = true;
            break;
        }
    }
    EXPECT_TRUE(any_diff);
}

// -----------------------------------------------------------------------------
// 7. Node2Vec end-to-end via PAT — walks valid under rejection sampling
// -----------------------------------------------------------------------------

TEST(WalkEngine, PatNode2VecBasic) {
    auto g = make_fanout_graph();

    tea::NeighborSets neighbors;
    neighbors.build(g, /*input_was_directed=*/true);

    tea::Node2VecBias bias;
    bias.p = 1.0;
    bias.q = 2.0;
    bias.timescale_bound = -1;

    tea::Pat<tea::Node2VecBias> pat;
    pat.build(g, bias);

    constexpr int32_t N = 200;
    constexpr int32_t L = 12;

    std::vector<int32_t> starts(N);
    for (int32_t i = 0; i < N; ++i) starts[i] = i % 30;

    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    auto stats = tea::run_walks_pat_node2vec(g, pat, bias, neighbors,
                                              starts.data(), N, L,
                                              0xa1b2'c3d4ULL,
                                              out.data(), lens.data());

    assert_stats_consistent(stats, lens.data(), N);
    assert_walks_valid(g, starts.data(), out.data(), lens.data(), N, L);
}

// -----------------------------------------------------------------------------
// 8. HPAT Node2Vec variant — fourth sample path exercised
// -----------------------------------------------------------------------------

TEST(WalkEngine, HpatNode2VecBasic) {
    auto g = make_fanout_graph();

    tea::NeighborSets neighbors;
    neighbors.build(g, /*input_was_directed=*/true);

    tea::Node2VecBias bias;
    bias.p = 0.5;
    bias.q = 4.0;
    bias.timescale_bound = -1;

    tea::Hpat<tea::Node2VecBias> hpat;
    hpat.build(g, bias);

    constexpr int32_t N = 200;
    constexpr int32_t L = 12;

    std::vector<int32_t> starts(N);
    for (int32_t i = 0; i < N; ++i) starts[i] = i % 30;

    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    auto stats = tea::run_walks_hpat_node2vec(g, hpat, bias, neighbors,
                                               starts.data(), N, L,
                                               0xfade'b00cULL,
                                               out.data(), lens.data());

    assert_stats_consistent(stats, lens.data(), N);
    assert_walks_valid(g, starts.data(), out.data(), lens.data(), N, L);
}

// -----------------------------------------------------------------------------
// 9. make_all_nodes_starts — helper correctness
// -----------------------------------------------------------------------------

TEST(WalkEngine, MakeAllNodesStartsExcludesDeadVertices) {
    auto g = make_fanout_graph();
    auto starts = tea::make_all_nodes_starts(g, /*walks_per_node=*/3);

    // 30 live vertices × 3 walks each = 90. v=30 (degree 0) excluded.
    EXPECT_EQ(starts.size(), 90u);

    // Each live vertex appears exactly 3 times; v=30 never appears.
    std::vector<int> counts(g.num_vertices(), 0);
    for (int32_t v : starts) {
        ASSERT_GE(v, 0);
        ASSERT_LT(v, g.num_vertices());
        counts[v]++;
    }
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        if (g.degree(u) > 0) {
            EXPECT_EQ(counts[u], 3) << "vertex " << u << " count mismatch";
        } else {
            EXPECT_EQ(counts[u], 0) << "dead vertex " << u << " should not appear";
        }
    }
}

// -----------------------------------------------------------------------------
// 10. Empty walk run — no starts → zero stats
// -----------------------------------------------------------------------------

TEST(WalkEngine, EmptyStartsZeroStats) {
    auto g = make_fanout_graph();
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    auto stats = tea::run_walks_pat(g, pat, tea::UniformBias{},
                                    /*starts=*/nullptr, /*num_walks=*/0,
                                    /*max_walk_len=*/8, 0ULL,
                                    /*out=*/nullptr, /*lens=*/nullptr);
    EXPECT_EQ(stats.num_walks,     0);
    EXPECT_EQ(stats.total_steps,   0);
    EXPECT_EQ(stats.dead_at_start, 0);
}
