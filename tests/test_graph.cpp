// Phase 1.4: TemporalGraph correctness tests.
//
// Walks are FORWARD-IN-TIME and the graph stores OUTBOUND adjacency
// (see graph.hpp): each directed input edge (u, v, t) becomes one entry
// under vertex u with target=v (the destination) and ts=t.  Undirected
// edges add entries to both endpoints.  Per-vertex adjacency is sorted
// by timestamp DESCENDING.
//
// What we verify:
//   - out-degree counts match the input
//   - per-vertex adjacency is sorted time-DESCENDING
//   - directed (u, v, t) stores at u's slot with target=v
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
    // Outbound adjacency (under directed):
    //   u=0 → {(1 @ 10), (2 @ 30), (3 @ 20)}         out-degree 3
    //   u=1 → {(2 @ 50)}                              out-degree 1
    //   u=2 → {(3 @ 40), (0 @ 25)}                    out-degree 2
    //   u=3 → {}                                      out-degree 0
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

TEST(GraphTest, OutDegreesAreCorrect) {
    auto g = build_simple();
    EXPECT_EQ(g.degree(0), 3);  // 0→1@10, 0→2@30, 0→3@20
    EXPECT_EQ(g.degree(1), 1);  // 1→2@50
    EXPECT_EQ(g.degree(2), 2);  // 2→3@40, 2→0@25
    EXPECT_EQ(g.degree(3), 0);  // no out-edges
    EXPECT_EQ(g.num_vertices(), 4);
    EXPECT_EQ(g.num_edges(),    6);
}

TEST(GraphTest, OutboundEdgesSortedTimeDescending) {
    auto g = build_simple();
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        auto ts = g.timestamps_of(u);
        for (std::size_t i = 1; i < ts.size(); ++i) {
            EXPECT_GE(ts[i - 1], ts[i])
                << "vertex " << u << " not sorted desc at index " << i;
        }
    }
    // u=0 has outbound {(1 @ 10), (3 @ 20), (2 @ 30)} → desc: ts=[30, 20, 10],
    // target=[2, 3, 1].
    auto ts0  = g.timestamps_of(0);
    auto tgt0 = g.targets_of(0);
    ASSERT_EQ(ts0.size(), 3u);
    EXPECT_EQ(ts0[0],  30);  EXPECT_EQ(tgt0[0], 2);
    EXPECT_EQ(ts0[1],  20);  EXPECT_EQ(tgt0[1], 3);
    EXPECT_EQ(ts0[2],  10);  EXPECT_EQ(tgt0[2], 1);
}

TEST(GraphTest, DirectedStoresDestinationAsTarget) {
    // For input edge (u, v, t), u's adjacency must contain target=v, ts=t.
    auto g = build_simple();

    // u=2 → {(3 @ 40), (0 @ 25)}; desc order: ts=[40, 25], target=[3, 0]
    auto ts2  = g.timestamps_of(2);
    auto tgt2 = g.targets_of(2);
    ASSERT_EQ(ts2.size(), 2u);
    EXPECT_EQ(ts2[0],  40);  EXPECT_EQ(tgt2[0], 3);
    EXPECT_EQ(ts2[1],  25);  EXPECT_EQ(tgt2[1], 0);
}

TEST(GraphTest, CandidateSetBoundaryCases) {
    auto g = build_simple();
    // u=0 has outbound ts_desc = {30, 20, 10}; candidate = count of t > t_prev.
    EXPECT_EQ(g.candidate_set_len(0, 35), 0);                          // nothing > 35
    EXPECT_EQ(g.candidate_set_len(0, 30), 0);                          // strictly greater
    EXPECT_EQ(g.candidate_set_len(0, 29), 1);                          // only 30
    EXPECT_EQ(g.candidate_set_len(0, 10), 2);                          // 30, 20 (10 excluded)
    EXPECT_EQ(g.candidate_set_len(0,  9), 3);                          // 30, 20, 10
    EXPECT_EQ(g.candidate_set_len(0, tea::kSentinelStartTimestamp), 3); // sentinel start
}

TEST(GraphTest, DuplicateTimestampsHandled) {
    // Three outgoing edges from u=4 at the same timestamp.
    std::vector<tea::Edge> edges = {
        {4, 0, 100}, {4, 1, 100}, {4, 2, 100},
        {4, 3, 200},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/5, /*is_directed=*/true);

    // u=4 has 4 outbound: ts_desc = [200, 100, 100, 100]
    auto ts = g.timestamps_of(4);
    ASSERT_EQ(ts.size(), 4u);
    EXPECT_EQ(ts[0], 200);
    EXPECT_EQ(ts[1], 100);
    EXPECT_EQ(ts[2], 100);
    EXPECT_EQ(ts[3], 100);

    // Boundary: t_prev=100 means "strictly later", so nothing > 100 except 200.
    EXPECT_EQ(g.candidate_set_len(4, 100), 1);
    // t_prev=99 includes all four entries.
    EXPECT_EQ(g.candidate_set_len(4,  99), 4);
    // t_prev=200 includes nothing (must be strictly greater).
    EXPECT_EQ(g.candidate_set_len(4, 200), 0);
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
