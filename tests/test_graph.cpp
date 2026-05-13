// Phase 1.4: TemporalGraph correctness tests.
//
// What we verify:
//   - degree counts match the input
//   - per-vertex edge list is sorted time-DESCENDING
//   - candidate_set_len(u, t_prev) returns the right prefix length for all
//     the boundary cases (t before earliest / between / after latest / exactly
//     on a timestamp / on a duplicate timestamp)
//   - undirected mode duplicates edges in both directions
//   - parallel build is deterministic (same input → same per-vertex edge order)

#include <algorithm>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include "tea/graph.hpp"

namespace {

tea::TemporalGraph build_simple() {
    // 4 vertices, 6 directed edges. Timestamps span across vertices.
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

TEST(GraphTest, DegreesAreCorrect) {
    auto g = build_simple();
    EXPECT_EQ(g.degree(0), 3);
    EXPECT_EQ(g.degree(1), 1);
    EXPECT_EQ(g.degree(2), 2);
    EXPECT_EQ(g.degree(3), 0);
    EXPECT_EQ(g.num_vertices(), 4);
    EXPECT_EQ(g.num_edges(),    6);
}

TEST(GraphTest, EdgesAreSortedTimeDescending) {
    auto g = build_simple();
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        auto ts = g.timestamps_of(u);
        for (std::size_t i = 1; i < ts.size(); ++i) {
            EXPECT_GE(ts[i - 1], ts[i])
                << "vertex " << u << " not sorted desc at index " << i;
        }
    }
    // u=0 has edges at t={10, 30, 20} → after desc-sort: {30, 20, 10}
    auto ts0 = g.timestamps_of(0);
    ASSERT_EQ(ts0.size(), 3u);
    EXPECT_EQ(ts0[0], 30);
    EXPECT_EQ(ts0[1], 20);
    EXPECT_EQ(ts0[2], 10);
}

TEST(GraphTest, CandidateSetBoundaryCases) {
    auto g = build_simple();
    // u=0: timestamps_desc = {30, 20, 10}
    EXPECT_EQ(g.candidate_set_len(0, 35), 0);  // nothing > 35
    EXPECT_EQ(g.candidate_set_len(0, 30), 0);  // strictly greater
    EXPECT_EQ(g.candidate_set_len(0, 29), 1);  // only 30
    EXPECT_EQ(g.candidate_set_len(0, 20), 1);  // 30 only (20 itself excluded)
    EXPECT_EQ(g.candidate_set_len(0, 10), 2);  // 30, 20
    EXPECT_EQ(g.candidate_set_len(0,  9), 3);  // all three
    EXPECT_EQ(g.candidate_set_len(0, -1), 3);  // sentinel start time
}

TEST(GraphTest, DuplicateTimestampsHandled) {
    // Multiple edges from u=0 at the same timestamp.
    std::vector<tea::Edge> edges = {
        {0, 1, 100}, {0, 2, 100}, {0, 3, 100},
        {0, 4, 200},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/5, /*is_directed=*/true);

    auto ts = g.timestamps_of(0);
    ASSERT_EQ(ts.size(), 4u);
    EXPECT_EQ(ts[0], 200);
    EXPECT_EQ(ts[1], 100);
    EXPECT_EQ(ts[2], 100);
    EXPECT_EQ(ts[3], 100);

    // Boundary: t_prev=100 means "strictly later", so 200 only.
    EXPECT_EQ(g.candidate_set_len(0, 100), 1);
    // t_prev=99 includes all three 100's plus 200.
    EXPECT_EQ(g.candidate_set_len(0,  99), 4);
}

TEST(GraphTest, UndirectedDuplicatesEachEdge) {
    std::vector<tea::Edge> edges = {
        {0, 1, 10}, {0, 2, 20},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/3, /*is_directed=*/false);

    EXPECT_EQ(g.degree(0), 2);
    EXPECT_EQ(g.degree(1), 1);  // back-edge from u=0
    EXPECT_EQ(g.degree(2), 1);

    auto ts1 = g.timestamps_of(1);
    auto ts2 = g.timestamps_of(2);
    EXPECT_EQ(ts1[0], 10);
    EXPECT_EQ(ts2[0], 20);
}

TEST(GraphTest, ParallelBuildIsDeterministic) {
    // Build the same input twice; the per-vertex edge order should be
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
