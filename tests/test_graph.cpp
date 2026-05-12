// Phase 1.4: TemporalGraph correctness tests.
//
// Walks are BACKWARD-IN-TIME and the graph stores INBOUND adjacency
// (see graph.hpp): each directed input edge (u, v, t) becomes one entry
// under vertex v with target=u (the source) and ts=t.  Undirected edges
// add entries to both endpoints.  Per-vertex adjacency is sorted by
// timestamp ASCENDING.
//
// What we verify:
//   - in-degree counts match the input
//   - per-vertex adjacency is sorted time-ASCENDING
//   - directed (u, v, t) stores at v's slot with target=u
//   - candidate_set_len(u, t_prev) returns the right prefix length for
//     all the boundary cases
//   - undirected mode duplicates edges in both directions
//   - parallel build is deterministic

#include <algorithm>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include "tea/config.hpp"
#include "tea/graph.hpp"

namespace {

tea::TemporalGraph build_simple() {
    // 4 vertices, 6 directed edges. Timestamps span across vertices.
    //
    //   0 → 1 @ 10, 0 → 2 @ 30, 0 → 3 @ 20
    //   1 → 2 @ 50
    //   2 → 3 @ 40, 2 → 0 @ 25
    //
    // Inbound adjacency (under directed):
    //   v=0 ← {(2 @ 25)}                           in-degree 1
    //   v=1 ← {(0 @ 10)}                           in-degree 1
    //   v=2 ← {(0 @ 30), (1 @ 50)}                 in-degree 2
    //   v=3 ← {(0 @ 20), (2 @ 40)}                 in-degree 2
    std::vector<tea::Edge> edges = {
        {0, 1, 10}, {0, 2, 30}, {0, 3, 20},
        {1, 2, 50},
        {2, 3, 40}, {2, 0, 25},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/4, /*is_directed=*/true);
    return g;
}

}  // namespace

TEST(GraphTest, InDegreesAreCorrect) {
    auto g = build_simple();
    EXPECT_EQ(g.degree(0), 1);  // only 2 → 0 @ 25
    EXPECT_EQ(g.degree(1), 1);  // only 0 → 1 @ 10
    EXPECT_EQ(g.degree(2), 2);  // 0 → 2 @ 30, 1 → 2 @ 50
    EXPECT_EQ(g.degree(3), 2);  // 0 → 3 @ 20, 2 → 3 @ 40
    EXPECT_EQ(g.num_vertices(), 4);
    EXPECT_EQ(g.num_edges(),    6);
}

TEST(GraphTest, InboundEdgesSortedTimeAscending) {
    auto g = build_simple();
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        auto ts = g.timestamps_of(u);
        for (std::size_t i = 1; i < ts.size(); ++i) {
            EXPECT_LE(ts[i - 1], ts[i])
                << "vertex " << u << " not sorted asc at index " << i;
        }
    }
    // v=3 has inbound from {0 @ 20, 2 @ 40} → asc: ts=[20, 40], target=[0, 2]
    auto ts3  = g.timestamps_of(3);
    auto tgt3 = g.targets_of(3);
    ASSERT_EQ(ts3.size(), 2u);
    EXPECT_EQ(ts3[0],  20);  EXPECT_EQ(tgt3[0], 0);
    EXPECT_EQ(ts3[1],  40);  EXPECT_EQ(tgt3[1], 2);
}

TEST(GraphTest, DirectedStoresSourceAsTarget) {
    // For input edge (u, v, t), v's adjacency must contain target=u, ts=t.
    auto g = build_simple();

    // v=2 ← {(0 @ 30), (1 @ 50)}; asc order: ts=[30, 50], target=[0, 1]
    auto ts2  = g.timestamps_of(2);
    auto tgt2 = g.targets_of(2);
    ASSERT_EQ(ts2.size(), 2u);
    EXPECT_EQ(ts2[0],  30);  EXPECT_EQ(tgt2[0], 0);
    EXPECT_EQ(ts2[1],  50);  EXPECT_EQ(tgt2[1], 1);
}

TEST(GraphTest, CandidateSetBoundaryCases) {
    auto g = build_simple();
    // v=3 has inbound ts_asc = {20, 40}; candidate = count of t < t_prev.
    EXPECT_EQ(g.candidate_set_len(3, 15), 0);                          // nothing < 15
    EXPECT_EQ(g.candidate_set_len(3, 20), 0);                          // strictly less
    EXPECT_EQ(g.candidate_set_len(3, 21), 1);                          // only 20
    EXPECT_EQ(g.candidate_set_len(3, 40), 1);                          // 20 only (40 excluded)
    EXPECT_EQ(g.candidate_set_len(3, 41), 2);                          // 20, 40
    EXPECT_EQ(g.candidate_set_len(3, tea::kSentinelStartTimestamp), 2); // sentinel start
}

TEST(GraphTest, DuplicateTimestampsHandled) {
    // Three incoming edges to v=4 at the same timestamp.
    std::vector<tea::Edge> edges = {
        {0, 4, 100}, {1, 4, 100}, {2, 4, 100},
        {3, 4, 200},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/5, /*is_directed=*/true);

    // v=4 has 4 inbound: ts_asc = [100, 100, 100, 200]
    auto ts = g.timestamps_of(4);
    ASSERT_EQ(ts.size(), 4u);
    EXPECT_EQ(ts[0], 100);
    EXPECT_EQ(ts[1], 100);
    EXPECT_EQ(ts[2], 100);
    EXPECT_EQ(ts[3], 200);

    // Boundary: t_prev=100 means "strictly earlier", so nothing < 100.
    EXPECT_EQ(g.candidate_set_len(4, 100), 0);
    // t_prev=101 includes all three 100's.
    EXPECT_EQ(g.candidate_set_len(4, 101), 3);
    // t_prev=201 includes everything.
    EXPECT_EQ(g.candidate_set_len(4, 201), 4);
}

TEST(GraphTest, UndirectedDuplicatesEachEdge) {
    std::vector<tea::Edge> edges = {
        {0, 1, 10}, {0, 2, 20},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/3, /*is_directed=*/false);

    // Undirected: every edge is added to both endpoints' adjacency.
    EXPECT_EQ(g.degree(0), 2);  // {1 @ 10, 2 @ 20}
    EXPECT_EQ(g.degree(1), 1);  // {0 @ 10}
    EXPECT_EQ(g.degree(2), 1);  // {0 @ 20}

    auto ts1 = g.timestamps_of(1);
    auto ts2 = g.timestamps_of(2);
    EXPECT_EQ(ts1[0], 10);
    EXPECT_EQ(ts2[0], 20);
}

TEST(GraphTest, ParallelBuildIsDeterministic) {
    // Build the same input twice; the per-vertex adjacency should be
    // byte-identical even though sorting runs in parallel.
    auto make_input = []() {
        std::vector<tea::Edge> in;
        for (int32_t u = 0; u < 100; ++u) {
            for (int32_t v = 0; v < 50; ++v) {
                in.push_back(tea::Edge{u, v, static_cast<int64_t>(u * 1000 + v)});
            }
        }
        return in;
    };

    tea::TemporalGraph g1; g1.build(make_input(), 100, true);
    tea::TemporalGraph g2; g2.build(make_input(), 100, true);

    for (int32_t u = 0; u < 100; ++u) {
        auto a = g1.targets_of(u);
        auto b = g2.targets_of(u);
        ASSERT_EQ(a.size(), b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            EXPECT_EQ(a[i], b[i]) << "vertex " << u << " idx " << i;
        }
    }
}
