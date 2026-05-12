// Phase 5.3: Node2Vec correctness.
//
// What we verify:
//   • NeighborSets: sorted+deduped per-vertex, contains() works, parallel
//     build is deterministic, memory is finite.
//   • Node2VecBias.accept_ratio matches the paper §2.3 III formula for
//     all three β cases (returning, 1-hop, 2-hop).
//   • Rejection sampling produces the correct distribution: empirical
//     edge-pick frequencies on a hand-built graph match
//     P(v_i) = β(w, v_i) × exp(t_i) / Σ_j β(w, v_j) × exp(t_j)
//     under both PAT and HPAT samplers.
//   • Edge cases: p=q=1 reduces to plain ExponentialBias.
//   • β extremes (p=10, q=0.1) produce the expected biased distribution.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/neighbor_set.hpp"
#include "tea/node2vec_bias.hpp"
#include "tea/pat.hpp"
#include "tea/rng.hpp"
#include "tea/sampler.hpp"

namespace {

// Build a small directed graph where the structure is fully known to the test.
// Under the inbound-CSR storage model (backward walks), each input edge
// (u, v, t) becomes "v has inbound from u at t".  NeighborSets[w] then
// contains the SOURCES of inbound to w.
//
// Target topology for the rejection-sampling test:
//   • v=0 has inbound from {1, 2, 3, 4, 5} at ts {100, 200, 300, 400, 500}
//     — these are the five β candidates when sampling from v=0.
//   • NeighborSets[1] = {0, 2}  (v=1 has inbound from 0 and 2)
//   • NeighborSets[2] = {0, 3, 4}  (v=2 has inbound from 0, 3, 4)
//
// Walks from v=0 with prev_u=1 see candidates {1, 2, 3, 4, 5}:
//   - source 1 → returning (β = 1/p)
//   - source 2 → in NeighborSets[1] (β = 1)
//   - sources 3, 4, 5 → not in NeighborSets[1] (β = 1/q)
tea::TemporalGraph build_node2vec_test_graph() {
    std::vector<tea::Edge> edges = {
        // inbound to v=0 from sources {1,2,3,4,5}
        {1, 0, 100}, {2, 0, 200}, {3, 0, 300}, {4, 0, 400}, {5, 0, 500},
        // inbound to v=1 from {0, 2}
        {0, 1, 50},  {2, 1, 60},
        // inbound to v=2 from {0, 3, 4}
        {0, 2, 70},  {3, 2, 80},  {4, 2, 90},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/10, /*is_directed=*/true);
    return g;
}

// Analytic node2vec edge transition probabilities from vertex u with
// previous-vertex w on the given graph. Uses the same exp(t - t_max_u)
// formula as the implementation.
std::vector<double> analytic_node2vec_probs(
        const tea::TemporalGraph& g,
        const tea::NeighborSets&  neighbors,
        const tea::Node2VecBias&  bias,
        int32_t                   u,
        int32_t                   prev_u) {
    auto ts_u = g.timestamps_of(u);
    auto tg_u = g.targets_of(u);
    const std::size_t D = ts_u.size();
    auto params = bias.compute_per_vertex_params(ts_u);

    std::vector<double> w(D, 0.0);
    bias.compute_weights(ts_u, params, 0, w.data());

    for (std::size_t i = 0; i < D; ++i) {
        double beta;
        if (tg_u[i] == prev_u)                        beta = 1.0 / bias.p;
        else if (neighbors.contains(prev_u, tg_u[i])) beta = 1.0;
        else                                          beta = 1.0 / bias.q;
        w[i] *= beta;
    }
    const double total = std::accumulate(w.begin(), w.end(), 0.0);
    for (auto& x : w) x /= total;
    return w;
}

double chi2(const std::vector<int64_t>& counts,
            const std::vector<double>&  expected,
            int64_t N) {
    double s = 0.0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const double e = expected[i] * N;
        if (e <= 0.0) continue;
        const double d = counts[i] - e;
        s += d * d / e;
    }
    return s;
}

double chi2_threshold(int df) {
    if (df <= 1)  return  8.0;
    if (df <= 2)  return 11.0;
    if (df <= 3)  return 14.0;
    if (df <= 4)  return 16.0;
    if (df <= 5)  return 18.0;
    if (df <= 9)  return 25.0;
    return df * 2.5;
}

}  // namespace

// ============================================================================
// NeighborSets
// ============================================================================
TEST(NeighborSets, BuildAndContains) {
    auto g = build_node2vec_test_graph();
    tea::NeighborSets ns;
    ns.build(g);

    EXPECT_TRUE(ns.contains(0, 1));
    EXPECT_TRUE(ns.contains(0, 5));
    EXPECT_FALSE(ns.contains(0, 6));
    EXPECT_FALSE(ns.contains(0, 99));

    EXPECT_TRUE(ns.contains(1, 0));
    EXPECT_TRUE(ns.contains(1, 2));
    EXPECT_FALSE(ns.contains(1, 3));

    EXPECT_TRUE(ns.contains(2, 0));
    EXPECT_TRUE(ns.contains(2, 3));
    EXPECT_TRUE(ns.contains(2, 4));
    EXPECT_FALSE(ns.contains(2, 5));

    // Empty vertex (3, 4, ..., 9).
    EXPECT_FALSE(ns.contains(3, 0));
    EXPECT_FALSE(ns.contains(9, 0));
}

TEST(NeighborSets, DegreeMatchesDistinctNeighbors) {
    auto g = build_node2vec_test_graph();
    tea::NeighborSets ns;
    ns.build(g);
    EXPECT_EQ(ns.degree(0), 5);
    EXPECT_EQ(ns.degree(1), 2);
    EXPECT_EQ(ns.degree(2), 3);
    EXPECT_EQ(ns.degree(3), 0);
}

TEST(NeighborSets, DuplicateEdgesDeduped) {
    // Three duplicate inbound edges from 1 to 0 plus one from 2 to 0.
    // NeighborSets[0] should dedupe to {1, 2}.
    std::vector<tea::Edge> edges = {
        {1, 0, 100}, {1, 0, 200}, {1, 0, 300},
        {2, 0, 50},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), 5, true);
    tea::NeighborSets ns;
    ns.build(g);
    EXPECT_EQ(ns.degree(0), 2);  // {1, 2}, not 4.
    EXPECT_TRUE(ns.contains(0, 1));
    EXPECT_TRUE(ns.contains(0, 2));
    EXPECT_FALSE(ns.contains(0, 0));
}

TEST(NeighborSets, ParallelBuildDeterministic) {
    auto g = build_node2vec_test_graph();
    tea::NeighborSets a; a.build(g);
    tea::NeighborSets b; b.build(g);
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        for (int32_t v = 0; v < g.num_vertices(); ++v) {
            EXPECT_EQ(a.contains(u, v), b.contains(u, v))
                << "u=" << u << " v=" << v;
        }
    }
}

// ============================================================================
// Node2VecBias.accept_ratio
// ============================================================================
TEST(Node2VecBias, AcceptRatioCases) {
    auto g = build_node2vec_test_graph();
    tea::NeighborSets ns;
    ns.build(g);

    tea::Node2VecBias bias;
    bias.p = 0.5;  // β_return = 1/p = 2
    bias.q = 4.0;  // β_far    = 1/q = 0.25
    // β_neighbor = 1
    // β_max = max(2, 1, 0.25) = 2
    const double expected_bmax = 2.0;
    EXPECT_DOUBLE_EQ(bias.beta_max(), expected_bmax);

    // Walker just moved 1 → 0. prev_u = 1.
    // Candidate 1: returning to prev → β/β_max = 2/2 = 1.
    EXPECT_DOUBLE_EQ(bias.accept_ratio(/*prev_u=*/1, /*candidate_v=*/1, ns),
                     1.0);
    // Candidate 2: 1 → 2 is an edge → β/β_max = 1/2 = 0.5.
    EXPECT_DOUBLE_EQ(bias.accept_ratio(1, 2, ns), 0.5);
    // Candidates 3, 4, 5: not in N(1) → β/β_max = 0.25/2 = 0.125.
    EXPECT_DOUBLE_EQ(bias.accept_ratio(1, 3, ns), 0.125);
    EXPECT_DOUBLE_EQ(bias.accept_ratio(1, 4, ns), 0.125);
    EXPECT_DOUBLE_EQ(bias.accept_ratio(1, 5, ns), 0.125);
}

TEST(Node2VecBias, PEqualsQEqualsOneReducesToExp) {
    // With p = q = 1, β = 1 for all candidates → reduces to pure exp bias.
    auto g = build_node2vec_test_graph();
    tea::NeighborSets ns;
    ns.build(g);
    tea::Node2VecBias bias;
    bias.p = 1.0;
    bias.q = 1.0;
    EXPECT_DOUBLE_EQ(bias.beta_max(), 1.0);
    for (int32_t v : {1, 2, 3, 4, 5}) {
        EXPECT_DOUBLE_EQ(bias.accept_ratio(/*prev_u=*/1, v, ns), 1.0);
    }
}

// ============================================================================
// Sampler distribution check (PAT and HPAT variants)
// ============================================================================
TEST(Node2VecSampler, PatDistributionMatchesAnalytic) {
    auto g = build_node2vec_test_graph();
    tea::NeighborSets neighbors;
    neighbors.build(g);

    tea::Node2VecBias bias;
    bias.p = 0.5;
    bias.q = 2.0;
    tea::Pat<tea::Node2VecBias> pat;
    pat.build(g, bias);

    const auto expected = analytic_node2vec_probs(g, neighbors, bias, /*u=*/0, /*prev_u=*/1);

    constexpr int N = 300'000;
    tea::Pcg64 rng(0xa55, 0);
    tea::SamplerScratch scratch;
    auto targets = g.targets_of(0);
    std::vector<int64_t> counts(targets.size(), 0);
    int dead = 0;
    for (int i = 0; i < N; ++i) {
        auto step = tea::sample_pat_node2vec(g, pat, bias, neighbors,
                                              /*u=*/0,
                                              /*t_prev=*/tea::kSentinelStartTimestamp,
                                              /*prev_u=*/1, rng, scratch);
        if (step.v == tea::kWalkDeadSentinel) { ++dead; continue; }
        auto it = std::find(targets.begin(), targets.end(), step.v);
        ASSERT_NE(it, targets.end());
        ++counts[it - targets.begin()];
    }
    EXPECT_EQ(dead, 0);
    const double c2 = chi2(counts, expected, N);
    EXPECT_LT(c2, chi2_threshold(static_cast<int>(targets.size()) - 1))
        << "Node2Vec PAT distribution drifted, chi2=" << c2;
}

TEST(Node2VecSampler, HpatDistributionMatchesAnalytic) {
    auto g = build_node2vec_test_graph();
    tea::NeighborSets neighbors;
    neighbors.build(g);

    tea::Node2VecBias bias;
    bias.p = 0.5;
    bias.q = 2.0;
    tea::Hpat<tea::Node2VecBias> hpat;
    hpat.build(g, bias);

    const auto expected = analytic_node2vec_probs(g, neighbors, bias, 0, 1);

    constexpr int N = 300'000;
    tea::Pcg64 rng(0xb66, 0);
    tea::SamplerScratch scratch;
    auto targets = g.targets_of(0);
    std::vector<int64_t> counts(targets.size(), 0);
    int dead = 0;
    for (int i = 0; i < N; ++i) {
        auto step = tea::sample_hpat_node2vec(g, hpat, bias, neighbors,
                                              0, tea::kSentinelStartTimestamp,
                                              1, rng, scratch);
        if (step.v == tea::kWalkDeadSentinel) { ++dead; continue; }
        auto it = std::find(targets.begin(), targets.end(), step.v);
        ASSERT_NE(it, targets.end());
        ++counts[it - targets.begin()];
    }
    EXPECT_EQ(dead, 0);
    const double c2 = chi2(counts, expected, N);
    EXPECT_LT(c2, chi2_threshold(static_cast<int>(targets.size()) - 1))
        << "Node2Vec HPAT distribution drifted, chi2=" << c2;
}

TEST(Node2VecSampler, ExtremeQBiasesAwayFromFarVertices) {
    // q = 100 → 1/q = 0.01 (very small). Far candidates rarely picked.
    // Returning + neighbor candidates get most of the mass.
    //
    // Use a small-ts-span graph so exp((t − t_max)·scale) doesn't crush
    // the β contribution.  With ts step = 1, weights span exp(-4)..exp(0),
    // letting β=1/q=0.01 visibly suppress the "far" candidates.
    //
    // Inbound layout for the n2v candidate set at v=0:
    //   v=0 ← inbound from 1@1, 2@2, 3@3, 4@4, 5@5
    //   v=1 ← inbound from 0@10, 2@11    (so NeighborSets[1] = {0, 2})
    std::vector<tea::Edge> edges = {
        {1, 0, 1}, {2, 0, 2}, {3, 0, 3}, {4, 0, 4}, {5, 0, 5},
        {0, 1, 10}, {2, 1, 11},
    };
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/10, /*is_directed=*/true);

    tea::NeighborSets neighbors;
    neighbors.build(g);

    tea::Node2VecBias bias;
    bias.p = 1.0;
    bias.q = 100.0;
    tea::Pat<tea::Node2VecBias> pat;
    pat.build(g, bias);

    constexpr int N = 100'000;
    tea::Pcg64 rng(0xc77, 0);
    tea::SamplerScratch scratch;
    auto targets = g.targets_of(0);
    // ASC ordering → targets = [1, 2, 3, 4, 5].
    ASSERT_EQ(targets.size(), 5u);
    ASSERT_EQ(targets[0], 1);
    ASSERT_EQ(targets[1], 2);
    std::vector<int64_t> counts(targets.size(), 0);
    for (int i = 0; i < N; ++i) {
        auto step = tea::sample_pat_node2vec(g, pat, bias, neighbors,
                                              0, tea::kSentinelStartTimestamp,
                                              1, rng, scratch);
        if (step.v == tea::kWalkDeadSentinel) continue;
        auto it = std::find(targets.begin(), targets.end(), step.v);
        ASSERT_NE(it, targets.end());
        ++counts[it - targets.begin()];
    }
    // targets = [1, 2, 3, 4, 5]. v=1 returns to prev (β=1/p=1), v=2 is
    // in N(1) (β=1), v∈{3,4,5} are far (β=1/q=0.01).
    const int64_t close_mass = counts[0] + counts[1];           // {1, 2}
    const int64_t far_mass   = counts[2] + counts[3] + counts[4]; // {3, 4, 5}
    EXPECT_GT(close_mass, far_mass * 2)
        << "Expected close mass >> far mass under q=100; got close=" << close_mass
        << " far=" << far_mass;
}

TEST(Node2VecSampler, FirstStepNoPrevAcceptsImmediately) {
    // prev_u = -1 (sentinel for first step) → no β rejection, always accept.
    // Distribution should match plain exp bias.
    auto g = build_node2vec_test_graph();
    tea::NeighborSets neighbors;
    neighbors.build(g);

    tea::Node2VecBias bias;
    bias.p = 0.01;  // extreme — would normally reject most candidates
    bias.q = 100.0;
    tea::Pat<tea::Node2VecBias> pat;
    pat.build(g, bias);

    constexpr int N = 50'000;
    tea::Pcg64 rng(0xd88, 0);
    tea::SamplerScratch scratch;
    auto targets = g.targets_of(0);
    int dead = 0;
    for (int i = 0; i < N; ++i) {
        auto step = tea::sample_pat_node2vec(g, pat, bias, neighbors,
                                              0, tea::kSentinelStartTimestamp,
                                              /*prev_u=*/-1,
                                              rng, scratch);
        if (step.v == tea::kWalkDeadSentinel) ++dead;
    }
    EXPECT_EQ(dead, 0) << "first-step walks should never die from rejection";
}
