// Phase 3.1: PAT build correctness — structural invariants.
//
// What we verify (sampler-side correctness is in Phase 3.3, test_sampler.cpp):
//   • trunk_size = ceil(sqrt(D)) clamped to [kPatTrunkSizeMin, D]
//   • num_trunks_of(u) = ceil(D / trunk_size_u) for every vertex
//   • alias_arena entries are in valid range (prob ∈ [0,1], alias < trunk_size)
//   • trunk cumsums are non-negative and monotonically non-decreasing
//   • per-vertex bias params are stored
//   • zero-degree vertices have zero trunks
//   • parallel build produces the same result across runs (deterministic
//     given fixed bias and graph)
//   • PAT works with all three biases (Uniform, Linear, Exponential)
//   • Memory accounting matches reality

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <utility>
#include <vector>

#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/pat.hpp"

namespace {

tea::TemporalGraph build_test_graph() {
    // 5 "main" vertices with varying outbound-degree (the forward-walks
    // adjacency):
    //   u=0:  10 outbound (medium)
    //   u=1: 100 outbound (high; multiple trunks)
    //   u=2:   1 outbound (low; one trunk, padded by min_trunk_size)
    //   u=3:   0 outbound
    //   u=4:   4 outbound (just below min_trunk_size)
    std::vector<tea::Edge> edges;
    for (int i = 0; i < 10;  ++i) edges.push_back({0, 100 + i, 1000 + i});
    for (int i = 0; i < 100; ++i) edges.push_back({1, 200 + i, 2000 + i});
    edges.push_back({2, 300, 3000});
    for (int i = 0; i < 4; ++i) edges.push_back({4, 400 + i, 4000 + i});

    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/500, /*is_directed=*/true);
    return g;
}

}  // namespace

TEST(PatBuild, TrunkSizeHeuristicMatchesPaper) {
    auto g = build_test_graph();
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    // D=10  → ceil(sqrt(10))=4, clamped to min=8 → T=8
    EXPECT_EQ(pat.trunk_size_of(0), 8);
    // D=100 → ceil(sqrt(100))=10, max(10, 8)=10 → T=10
    EXPECT_EQ(pat.trunk_size_of(1), 10);
    // D=1   → ceil(sqrt(1))=1, clamped to min=8, capped at D=1 → T=1
    EXPECT_EQ(pat.trunk_size_of(2), 1);
    // D=0   → T=0
    EXPECT_EQ(pat.trunk_size_of(3), 0);
    // D=4   → ceil(sqrt(4))=2, clamped to min=8, capped at D=4 → T=4
    EXPECT_EQ(pat.trunk_size_of(4), 4);
}

TEST(PatBuild, NumTrunksMatchesCeil) {
    auto g = build_test_graph();
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    EXPECT_EQ(pat.num_trunks_of(0), 2);  // 10 / 8 → 2
    EXPECT_EQ(pat.num_trunks_of(1), 10); // 100 / 10 → 10
    EXPECT_EQ(pat.num_trunks_of(2), 1);  // 1 / 1 → 1
    EXPECT_EQ(pat.num_trunks_of(3), 0);  // empty vertex
    EXPECT_EQ(pat.num_trunks_of(4), 1);  // 4 / 4 → 1
}

TEST(PatBuild, AliasEntriesAreInRange) {
    auto g = build_test_graph();
    tea::Pat<tea::LinearBias> pat;
    pat.build(g, tea::LinearBias{});

    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        const int32_t num_trunks = pat.num_trunks_of(u);
        for (int32_t i = 0; i < num_trunks; ++i) {
            auto av = pat.alias_view_of_trunk(g, u, i);
            for (const auto& e : av.entries()) {
                EXPECT_GE(e.prob, 0.0f);
                EXPECT_LE(e.prob, 1.0f);
                EXPECT_LT(e.alias, av.size());
            }
        }
    }
}

TEST(PatBuild, TrunkCumsumIsMonotoneAndPositive) {
    auto g = build_test_graph();
    tea::Pat<tea::ExponentialBias> pat;
    pat.build(g, tea::ExponentialBias{});

    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        auto cs = pat.trunk_cumsums_of(u);
        if (cs.empty()) continue;
        EXPECT_GT(cs[0], 0.0) << "vertex " << u << " first cumsum should be positive";
        for (std::size_t i = 1; i < cs.size(); ++i) {
            EXPECT_GE(cs[i], cs[i - 1])
                << "vertex " << u << " cumsum not monotone at trunk " << i;
        }
    }
}

TEST(PatBuild, BiasParamsAreStored) {
    auto g = build_test_graph();
    tea::ExponentialBias bias;
    bias.timescale_bound = -1.0;  // proper TEA
    tea::Pat<tea::ExponentialBias> pat;
    pat.build(g, bias);

    // u=0 has 10 outbound edges with ts in 1000..1009. Pivot on t_min = 1000.
    const auto& p0 = pat.bias_params_of(0);
    EXPECT_DOUBLE_EQ(p0.t_pivot, 1000.0);
    EXPECT_DOUBLE_EQ(p0.scale, 1.0);

    // u=1 has 100 outbound edges with ts in 2000..2099. t_min = 2000.
    const auto& p1 = pat.bias_params_of(1);
    EXPECT_DOUBLE_EQ(p1.t_pivot, 2000.0);
    EXPECT_DOUBLE_EQ(p1.scale, 1.0);
}

TEST(PatBuild, EmptyGraphIsHarmless) {
    tea::TemporalGraph g;
    g.build({}, /*num_vertices=*/10, /*is_directed=*/true);
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    EXPECT_EQ(pat.alias_entry_count(),  0);
    EXPECT_EQ(pat.cumsum_entry_count(), 0);
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        EXPECT_EQ(pat.num_trunks_of(u), 0);
        EXPECT_TRUE(pat.trunk_cumsums_of(u).empty());
    }
}

TEST(PatBuild, MemoryBytesAccountsForAllArenas) {
    auto g = build_test_graph();
    tea::Pat<tea::ExponentialBias> pat;
    pat.build(g, tea::ExponentialBias{});

    // 115 edges total → 115 alias entries × 8 = 920 bytes minimum
    EXPECT_GE(pat.memory_bytes(),
              static_cast<int64_t>(pat.alias_entry_count() * sizeof(tea::AliasEntry)));
    // And memory_bytes should be a non-trivial number for our small test graph.
    EXPECT_GT(pat.memory_bytes(), 0);
}

TEST(PatBuild, AliasTablesPerTrunkAreNonEmpty) {
    auto g = build_test_graph();
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});

    // For u=1 (D=100, T=10), there should be 10 trunks each with 10 entries.
    EXPECT_EQ(pat.num_trunks_of(1), 10);
    for (int i = 0; i < 10; ++i) {
        auto av = pat.alias_view_of_trunk(g, 1, i);
        EXPECT_EQ(av.size(), 10u);
    }
    // For u=2 (D=1, T=1), one trunk of 1 entry.
    EXPECT_EQ(pat.num_trunks_of(2), 1);
    EXPECT_EQ(pat.alias_view_of_trunk(g, 2, 0).size(), 1u);
    // For u=4 (D=4, T=4), one trunk of 4 entries.
    EXPECT_EQ(pat.num_trunks_of(4), 1);
    EXPECT_EQ(pat.alias_view_of_trunk(g, 4, 0).size(), 4u);
}

TEST(PatBuild, ParallelBuildIsDeterministic) {
    auto g = build_test_graph();

    tea::Pat<tea::ExponentialBias> pat1;
    pat1.build(g, tea::ExponentialBias{});
    tea::Pat<tea::ExponentialBias> pat2;
    pat2.build(g, tea::ExponentialBias{});

    // Alias entries should be byte-identical across builds.
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        const int32_t num_trunks = pat1.num_trunks_of(u);
        EXPECT_EQ(num_trunks, pat2.num_trunks_of(u));
        for (int32_t i = 0; i < num_trunks; ++i) {
            auto av1 = pat1.alias_view_of_trunk(g, u, i);
            auto av2 = pat2.alias_view_of_trunk(g, u, i);
            ASSERT_EQ(av1.size(), av2.size());
            for (std::size_t j = 0; j < av1.size(); ++j) {
                EXPECT_EQ(av1.entries()[j].alias, av2.entries()[j].alias);
                EXPECT_FLOAT_EQ(av1.entries()[j].prob, av2.entries()[j].prob);
            }
        }
        auto cs1 = pat1.trunk_cumsums_of(u);
        auto cs2 = pat2.trunk_cumsums_of(u);
        ASSERT_EQ(cs1.size(), cs2.size());
        for (std::size_t i = 0; i < cs1.size(); ++i) {
            EXPECT_DOUBLE_EQ(cs1[i], cs2[i]);
        }
    }
}

TEST(PatBuild, ExpBiasTrunkTotalsReflectTimeOrdering) {
    // Under time-DESC storage with Tempest-forward exp pivot (t_min), the
    // LAST trunk (oldest edges) should have the largest weight under
    // ExpBias — exp(t_min − t_min) = 1 lives at position D-1 (desc-list
    // tail), while exp((newer t) − t_min) ≪ 1 lives at position 0.
    auto g = build_test_graph();
    tea::Pat<tea::ExponentialBias> pat;
    pat.build(g, tea::ExponentialBias{});

    auto cs = pat.trunk_cumsums_of(1);  // u=1, 10 trunks of 10 edges each
    ASSERT_GE(cs.size(), 2u);
    // Trunk 0 total = cs[0]; last trunk total = cs[K-1] - cs[K-2].
    const std::size_t K = cs.size();
    const double t_first = cs[0];
    const double t_last  = cs[K - 1] - cs[K - 2];
    EXPECT_GT(t_last, t_first)
        << "oldest trunk (desc-list end) should have higher total weight than the newest";
}
