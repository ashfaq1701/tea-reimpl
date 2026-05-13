// Phase 4.5: temporal-start (TEA paper §3.3 FIRST ad-hoc optimization).
//
// Paper text: "if the temporal information of certain neighbors is earlier
// than all the incoming edges, we can simply discard them."
//
// In our walk model, walks start with t_prev = kSentinelStartTimestamp by
// default, so every outgoing edge of a start vertex is a first-hop
// candidate — no edge is "unreachable".  To enable the §3.3 opt, the walk
// engine has a `use_temporal_start` flag that seeds the first hop's t_prev
// from `graph.min_incoming_time_of(u_start)` instead.  Outgoing edges of u
// with t ≤ that value are then never traversed (the sampler's existing
// candidate-set computation skips them).
//
// What we verify:
//   1. min_incoming_time_of(v) returns the minimum incoming-edge timestamp
//      for v (or kSentinelStartTimestamp for source-only vertices).
//   2. Under use_temporal_start=true, every walk's FIRST hop has t > t_min_in
//      of the start vertex.
//   3. Default (use_temporal_start=false) preserves existing behavior:
//      first-hop t > kSentinelStartTimestamp = every outgoing edge eligible.
//   4. The two modes produce statistically the SAME distribution for
//      vertices whose t_min_in == kSentinelStartTimestamp (no incoming
//      edges → no constraint to apply).

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/pat.hpp"
#include "tea/rng.hpp"
#include "tea/walk_engine.hpp"

namespace {

// Two-component graph:
//   v=0: source-only (no incoming).  Edges to {10, 11, 12, 13} at t=100..103.
//   v=1: has incoming edge from 0 (no — 0 sends to {10..13} only).
//        Actually let's design it differently.
//
// Let's build:
//   0 → 1 @ t=50, 0 → 2 @ t=60, 0 → 3 @ t=80, 0 → 4 @ t=90
//   1 → 5 @ t=10  (older than 1's incoming 50!), 1 → 6 @ t=70, 1 → 7 @ t=80
//   2 → 8 @ t=30  (older than 2's incoming 60), 2 → 9 @ t=120
//   so:
//     vertex 0 has no incoming → t_min_in = sentinel
//     vertex 1's only incoming is t=50 → t_min_in[1] = 50
//     vertex 2's only incoming is t=60 → t_min_in[2] = 60
//   With use_temporal_start, a walk starting at 1 must NOT traverse 1→5 (t=10)
//   because 10 ≤ 50.  Similarly for 2→8 (t=30 ≤ 60).
tea::TemporalGraph make_temporal_start_graph() {
    std::vector<tea::Edge> e = {
        // outgoing from 0 (no incoming → sentinel t_min_in)
        {0, 1, 50}, {0, 2, 60}, {0, 3, 80}, {0, 4, 90},
        // outgoing from 1 (t_min_in[1] = 50)
        {1, 5, 10}, {1, 6, 70}, {1, 7, 80},
        // outgoing from 2 (t_min_in[2] = 60)
        {2, 8, 30}, {2, 9, 120},
    };
    tea::TemporalGraph g;
    g.build(std::move(e), /*num_vertices=*/20, /*is_directed=*/true);
    return g;
}

}  // namespace

// ============================================================================
// 1. min_incoming_time_of correctness.
// ============================================================================
TEST(TemporalStart, MinIncomingTimeIsCorrect) {
    auto g = make_temporal_start_graph();

    // 0 has no incoming edge → sentinel.
    EXPECT_EQ(g.min_incoming_time_of(0), tea::kSentinelStartTimestamp);
    // 1's only incoming is 0→1 @ t=50.
    EXPECT_EQ(g.min_incoming_time_of(1), 50);
    // 2's only incoming is 0→2 @ t=60.
    EXPECT_EQ(g.min_incoming_time_of(2), 60);
    // 3's only incoming is 0→3 @ t=80.
    EXPECT_EQ(g.min_incoming_time_of(3), 80);
    // 4 — incoming 0→4 @ t=90.
    EXPECT_EQ(g.min_incoming_time_of(4), 90);
    // 5..9: leaf nodes with one incoming each.
    EXPECT_EQ(g.min_incoming_time_of(5), 10);
    EXPECT_EQ(g.min_incoming_time_of(6), 70);
    EXPECT_EQ(g.min_incoming_time_of(7), 80);
    EXPECT_EQ(g.min_incoming_time_of(8), 30);
    EXPECT_EQ(g.min_incoming_time_of(9), 120);
    // 10..19: no incoming, no outgoing → sentinel.
    for (int32_t v = 10; v < 20; ++v) {
        EXPECT_EQ(g.min_incoming_time_of(v), tea::kSentinelStartTimestamp);
    }
}

// ============================================================================
// 2. Multi-incoming-edge case: min is taken over all incoming edges.
// ============================================================================
TEST(TemporalStart, MinIncomingPicksTheEarliest) {
    // u=0 → u=2 at three different timestamps + a fourth path 1→2.
    std::vector<tea::Edge> e = {
        {0, 2, 100}, {0, 2, 50}, {0, 2, 200},
        {1, 2, 30},
    };
    tea::TemporalGraph g;
    g.build(std::move(e), /*num_vertices=*/10, /*is_directed=*/true);

    // The earliest of {100, 50, 200, 30} is 30.
    EXPECT_EQ(g.min_incoming_time_of(2), 30);
}

// ============================================================================
// 3. With use_temporal_start=true, walks starting at u never see edges with
//    t ≤ t_min_in(u) on their first hop.
// ============================================================================
TEST(TemporalStart, WalksSkipUnreachableEdgesUnderTemporalStart) {
    auto g = make_temporal_start_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    // Start every walk at vertex 1 (t_min_in=50).  Edge 1→5 has t=10, which
    // is ≤ 50 — so under temporal-start, no walk's first hop should land
    // on v=5.
    constexpr int32_t  N = 5000;
    constexpr int32_t  L = 3;
    std::vector<int32_t> starts(N, 1);
    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    auto stats = tea::run_walks_hpat(g, hpat, tea::UniformBias{},
                                     starts.data(), N, L,
                                     /*seed=*/0xc0deULL,
                                     out.data(), lens.data(),
                                     /*use_temporal_start=*/true);
    (void)stats;

    int hops_to_5 = 0;
    int hops_to_others = 0;
    for (int32_t i = 0; i < N; ++i) {
        if (lens[i] < 2) continue;  // walk died at start (shouldn't happen)
        const int32_t first_hop_v = out[i * L + 1].v;
        if (first_hop_v == 5) ++hops_to_5;
        else                  ++hops_to_others;
    }
    EXPECT_EQ(hops_to_5, 0) << "Walk traversed 1→5 (t=10 ≤ t_min_in=50)";
    EXPECT_GT(hops_to_others, 0);
}

// ============================================================================
// 4. Without use_temporal_start (default), walks DO see those edges — every
//    outgoing edge is a candidate on the first hop.  Sanity check that we
//    didn't accidentally always-enable the optimization.
// ============================================================================
TEST(TemporalStart, DefaultStartSeesAllOutgoingEdges) {
    auto g = make_temporal_start_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    constexpr int32_t  N = 5000;
    constexpr int32_t  L = 3;
    std::vector<int32_t> starts(N, 1);
    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    tea::run_walks_hpat(g, hpat, tea::UniformBias{},
                        starts.data(), N, L,
                        0xfaceULL, out.data(), lens.data()
                        /* use_temporal_start defaults to false */);

    int hops_to_5 = 0;
    for (int32_t i = 0; i < N; ++i) {
        if (lens[i] < 2) continue;
        if (out[i * L + 1].v == 5) ++hops_to_5;
    }
    // With 3 candidates {5, 6, 7} and uniform bias, ~1/3 of 5000 = ~1667.
    // Allow wide margin; just check that 5 is reached at all.
    EXPECT_GT(hops_to_5, 100) << "Default start should reach v=5 a fair fraction of the time";
}

// ============================================================================
// 5. For source-only vertices (no incoming → t_min_in = sentinel), the two
//    modes are identical — temporal-start is a no-op there.
// ============================================================================
TEST(TemporalStart, SourceOnlyVertexUnaffectedByMode) {
    auto g = make_temporal_start_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    // Vertex 0 has no incoming → t_min_in=sentinel; both modes use sentinel.
    EXPECT_EQ(g.min_incoming_time_of(0), tea::kSentinelStartTimestamp);

    constexpr int32_t  N = 5000;
    constexpr int32_t  L = 2;
    std::vector<int32_t> starts(N, 0);
    auto run = [&](bool ts) {
        std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
        std::vector<int32_t>       lens(N);
        tea::run_walks_hpat(g, hpat, tea::UniformBias{},
                            starts.data(), N, L,
                            /*seed=*/0xdeadULL, out.data(), lens.data(), ts);
        return out;
    };
    auto out_off = run(false);
    auto out_on  = run(true);

    // Walks are deterministic given seed + vertex idx.  Source-only vertex
    // means t_min_in == sentinel == default → outputs must be byte-identical.
    for (std::size_t i = 0; i < out_off.size(); ++i) {
        ASSERT_EQ(out_off[i].v, out_on[i].v) << "slot " << i;
        ASSERT_EQ(out_off[i].t, out_on[i].t) << "slot " << i;
    }
}

// ============================================================================
// 6. Distribution under temporal-start is correct: walks starting at v=1 with
//    use_temporal_start=true sample uniformly from the (filtered) candidate
//    set {6, 7} (edge 1→5 at t=10 is excluded since 10 ≤ t_min_in(1)=50).
// ============================================================================
TEST(TemporalStart, DistributionMatchesFilteredCandidateSet) {
    auto g = make_temporal_start_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    // Vertex 1: outgoing edges {5@10, 6@70, 7@80}.
    // t_min_in(1) = 50.  Filtered candidate set on first hop = {6@70, 7@80}.
    // Uniform bias → each picked with probability 0.5.

    constexpr int32_t N = 20000;
    constexpr int32_t L = 2;
    std::vector<int32_t> starts(N, 1);
    std::vector<tea::NodeStep> out(static_cast<std::size_t>(N) * L);
    std::vector<int32_t>       lens(N);

    tea::run_walks_hpat(g, hpat, tea::UniformBias{},
                        starts.data(), N, L,
                        /*seed=*/0xbeefULL, out.data(), lens.data(),
                        /*use_temporal_start=*/true);

    int n5 = 0, n6 = 0, n7 = 0, n_other = 0;
    for (int32_t i = 0; i < N; ++i) {
        if (lens[i] < 2) { ++n_other; continue; }
        switch (out[i * L + 1].v) {
            case 5: ++n5; break;
            case 6: ++n6; break;
            case 7: ++n7; break;
            default: ++n_other; break;
        }
    }
    EXPECT_EQ(n5, 0);                      // filtered out
    EXPECT_EQ(n_other, 0);                 // no dead walks expected
    // Two-cell χ² with expected 50/50: |n6 - N/2| / sqrt(N/4) ~ N(0,1).
    // 4σ tolerance: |Δ| < 4 · sqrt(N/4) = 4·sqrt(5000) ≈ 282.
    EXPECT_LT(std::abs(n6 - N / 2), 350);
    EXPECT_LT(std::abs(n7 - N / 2), 350);
}
