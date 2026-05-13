// Phase 3.3: PAT sampler distribution correctness.
//
// The strongest test we can write at this stage. For each bias we:
//   1. Build a small graph by hand.
//   2. Build PAT under that bias.
//   3. For each vertex u and a representative t_prev, sample N times.
//   4. Tally empirical edge-pick frequencies.
//   5. Compare against the analytic P((u, v_i, t_i)) = δ(e_i) / Σ δ(e_j)
//      under the bias's static-weight formula.
//   6. Pass via χ² with df=K-1 and α=0.01 (threshold from tables).
//
// This is the load-bearing test: if the sampler is broken (e.g., trunk-
// boundary off-by-one, alias-table built from wrong weights, ITS picking
// the wrong half), this test catches it. Sampler bugs are silent —
// distribution tests are the ONLY way to find them.
//
// We also verify:
//   • Walk-dies when Γ_t(u) is empty (t_prev ≥ newest edge).
//   • Partial-trunk case is exercised (degree > trunk_size, t_prev in middle).
//   • Full-trunk-only case (Γ_len % trunk_size == 0).
//   • Single-edge case (degree=1, T=1, always returns that one edge).
//   • Determinism with fixed seed.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <utility>
#include <vector>

#include "tea/bias.hpp"
#include "tea/graph.hpp"
#include "tea/pat.hpp"
#include "tea/rng.hpp"
#include "tea/sampler.hpp"

namespace {

// Build a small directed graph with one source vertex u=0, K outgoing edges
// with monotonically decreasing timestamps. Edges go to vertices 100..100+K-1.
// All timestamps fit within the candidate set when t_prev < the oldest ts.
tea::TemporalGraph build_star(int K, int64_t ts_step = 100) {
    std::vector<tea::Edge> edges;
    edges.reserve(K);
    for (int i = 0; i < K; ++i) {
        edges.push_back({0, 100 + i, static_cast<int64_t>((K - i) * ts_step)});
    }
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/200, /*is_directed=*/true);
    return g;
}

// Compute analytic P(e_i) for the bias on this exact slice of u's edges.
template <typename BiasT>
std::vector<double> analytic_probs(const tea::TemporalGraph& g,
                                   const BiasT& bias,
                                   int32_t u,
                                   int64_t partial_start = 0,
                                   int64_t partial_len   = -1) {
    auto ts_u = g.timestamps_of(u);
    const int64_t D = static_cast<int64_t>(ts_u.size());
    if (partial_len < 0) partial_len = D - partial_start;
    auto params = bias.compute_per_vertex_params(ts_u);
    std::vector<double> w(static_cast<std::size_t>(partial_len), 0.0);
    bias.compute_weights(
        tea::span<const int64_t>(ts_u.data() + partial_start,
                                  static_cast<std::size_t>(partial_len)),
        params, static_cast<int32_t>(partial_start), w.data());
    double total = std::accumulate(w.begin(), w.end(), 0.0);
    if (total <= 0.0) {
        // All zero — uniform fallback.
        std::vector<double> p(w.size(), 1.0 / w.size());
        return p;
    }
    for (auto& x : w) x /= total;
    return w;
}

// Chi-square statistic from observed counts vs expected probabilities.
// Returns the chi-square value (caller compares to threshold for df=k-1, α=0.01).
double chi2(const std::vector<int64_t>& counts,
            const std::vector<double>&  expected_probs,
            int64_t N) {
    double s = 0.0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const double exp_count = expected_probs[i] * N;
        if (exp_count <= 0.0) continue;  // skip cells with zero expectation
        const double diff = counts[i] - exp_count;
        s += diff * diff / exp_count;
    }
    return s;
}

// Chi-square critical values at α=0.01 (one-tail).
// df=k-1; for our small K's we hardcode a few.
//   df=1 → 6.63   df=2 → 9.21   df=3 → 11.34  df=4 → 13.28
//   df=5 → 15.09  df=9 → 21.67  df=14 → 29.14 df=19 → 36.19
// Use slightly looser thresholds for safety against rare false alarms.
double chi2_threshold(int df) {
    if (df <= 1)  return  8.0;
    if (df <= 2)  return 11.0;
    if (df <= 3)  return 14.0;
    if (df <= 4)  return 16.0;
    if (df <= 5)  return 18.0;
    if (df <= 9)  return 25.0;
    if (df <= 14) return 33.0;
    if (df <= 19) return 41.0;
    return df * 2.5;  // generous fallback for larger df
}

template <typename BiasT>
void distribution_check(const tea::TemporalGraph& g,
                        const tea::Pat<BiasT>&    pat,
                        const BiasT&              bias,
                        int32_t                   u,
                        int64_t                   t_prev,
                        int64_t                   N_samples,
                        uint64_t                  seed,
                        const std::vector<double>& expected_probs,
                        int64_t                   partial_start) {
    tea::Pcg64 rng(seed, 0xa);
    tea::SamplerScratch scratch;

    auto targets_u = g.targets_of(u);
    const std::size_t K = expected_probs.size();
    std::vector<int64_t> counts(K, 0);
    int64_t dead = 0;

    for (int64_t i = 0; i < N_samples; ++i) {
        auto step = tea::sample_pat(g, pat, bias, u, t_prev, rng, scratch);
        if (step.v == tea::kWalkDeadSentinel) { ++dead; continue; }
        // Find which position in u's edge list was picked.
        // Since vertices in our star are distinct, we can map v → position.
        auto it = std::find(targets_u.begin(), targets_u.end(), step.v);
        ASSERT_NE(it, targets_u.end());
        const int64_t pos = it - targets_u.begin();
        const int64_t local = pos - partial_start;
        ASSERT_GE(local, 0);
        ASSERT_LT(local, static_cast<int64_t>(K));
        ++counts[local];
    }
    EXPECT_EQ(dead, 0);

    const double c2 = chi2(counts, expected_probs, N_samples);
    const double th = chi2_threshold(static_cast<int>(K) - 1);
    EXPECT_LT(c2, th)
        << "chi2=" << c2 << " threshold=" << th << " for bias " << BiasT::name();
}

}  // namespace

// ============================================================================
// UniformBias — every edge equally likely.
// ============================================================================
TEST(SamplerPat, UniformDistribution_FullTrunkOnly) {
    // K = 4 edges, T_u = 8 → 1 trunk holding all 4. No partial trunk.
    auto g = build_star(4);
    tea::Pat<tea::UniformBias> pat;
    tea::UniformBias bias;
    pat.build(g, bias);

    auto p = analytic_probs(g, bias, 0);
    distribution_check(g, pat, bias, /*u=*/0, /*t_prev=*/-1,
                       /*N=*/200000, /*seed=*/0x1337,
                       p, /*partial_start=*/0);
}

TEST(SamplerPat, UniformDistribution_MultipleTrunksAllFull) {
    // K = 16 edges, T_u = 8 → 2 trunks of 8, no partial trunk when Γ_len = 16.
    auto g = build_star(16);
    tea::Pat<tea::UniformBias> pat;
    tea::UniformBias bias;
    pat.build(g, bias);

    auto p = analytic_probs(g, bias, 0);
    distribution_check(g, pat, bias, 0, -1, 200000, 0xa11, p, 0);
}

TEST(SamplerPat, UniformDistribution_WithPartialTrunk) {
    // K = 16 edges, T_u = 8. Set t_prev to expose exactly 12 edges
    //   → Γ_len = 12, num_full = 1, partial_size = 4 (PARTIAL TRUNK)
    auto g = build_star(16, /*ts_step=*/100);
    // ts_desc = [1600, 1500, ..., 100]; want Γ_len = 12 ⇒ t_prev < ts[11] = 500
    //   and t_prev ≥ ts[12] = 400. Pick t_prev = 450.
    tea::Pat<tea::UniformBias> pat;
    tea::UniformBias bias;
    pat.build(g, bias);

    // analytic over only the first 12 edges
    auto p = analytic_probs(g, bias, 0, /*partial_start=*/0, /*partial_len=*/12);
    distribution_check(g, pat, bias, 0, /*t_prev=*/450, 200000, 0xb22, p, 0);
}

// ============================================================================
// LinearBias — δ(e_i) = rank(e_i)
// ============================================================================
TEST(SamplerPat, LinearDistribution_FullTrunkOnly) {
    auto g = build_star(4);
    tea::Pat<tea::LinearBias> pat;
    tea::LinearBias bias;
    pat.build(g, bias);

    auto p = analytic_probs(g, bias, 0);
    distribution_check(g, pat, bias, 0, -1, 200000, 0x123, p, 0);
}

TEST(SamplerPat, LinearDistribution_WithPartialTrunk) {
    // K = 16, T = 8, t_prev = 450 → Γ_len = 12, partial trunk of size 4.
    auto g = build_star(16);
    tea::Pat<tea::LinearBias> pat;
    tea::LinearBias bias;
    pat.build(g, bias);

    // Partial trunk's weights at sample time MUST match what was used at
    // build for positions [0, 12) — verifies the per-vertex bias-params +
    // slice-start-pos contract.
    auto p = analytic_probs(g, bias, 0, 0, 12);
    distribution_check(g, pat, bias, 0, 450, 200000, 0x456, p, 0);
}

// ============================================================================
// ExponentialBias — δ(e_i) = exp((t_i - t_max) × scale)
// ============================================================================
TEST(SamplerPat, ExponentialDistribution_SmallTimeScale) {
    // Use small timestamps so exp() doesn't underflow. K = 4 → no partial trunk.
    // ts step = 1 means weights are exp(0), exp(-1), exp(-2), exp(-3),
    // which give a non-degenerate distribution.
    auto g = build_star(4, /*ts_step=*/1);
    tea::Pat<tea::ExponentialBias> pat;
    tea::ExponentialBias bias;
    pat.build(g, bias);

    auto p = analytic_probs(g, bias, 0);
    // Sanity: newest edge gets the largest probability.
    EXPECT_GT(p[0], p[1]);
    EXPECT_GT(p[1], p[2]);
    EXPECT_GT(p[2], p[3]);
    distribution_check(g, pat, bias, 0, -1, 300000, 0x789, p, 0);
}

TEST(SamplerPat, ExponentialDistribution_WithPartialTrunk) {
    auto g = build_star(16, /*ts_step=*/1);
    tea::Pat<tea::ExponentialBias> pat;
    tea::ExponentialBias bias;
    pat.build(g, bias);

    // Γ_len = 12 ⇒ t_prev < ts[11] = 5 ⇒ t_prev = 4.5; integer ts, use 4.
    auto p = analytic_probs(g, bias, 0, 0, 12);
    distribution_check(g, pat, bias, 0, /*t_prev=*/4, 300000, 0xabc, p, 0);
}

TEST(SamplerPat, ExponentialDistribution_TimescaleBoundCompresses) {
    // With timescale_bound = 5, an exp bias on 4 edges with ts step=10
    // should be much less skewed than without (effective exponents become
    // [0, -5/3, -10/3, -5] instead of [0, -10, -20, -30]).
    auto g = build_star(4, /*ts_step=*/10);
    tea::Pat<tea::ExponentialBias> pat;
    tea::ExponentialBias bias;
    bias.timescale_bound = 5.0;
    pat.build(g, bias);

    auto p = analytic_probs(g, bias, 0);
    distribution_check(g, pat, bias, 0, -1, 300000, 0xdef, p, 0);
}

// ============================================================================
// Sampler edge cases
// ============================================================================
TEST(SamplerPat, WalkDiesWhenCandidateSetEmpty) {
    auto g = build_star(4, 100);  // newest ts = 400, oldest = 100
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});
    tea::Pcg64 rng(0xfeed, 0);
    tea::SamplerScratch scratch;

    // t_prev ≥ newest ts → no candidates → walk dies.
    auto step = tea::sample_pat(g, pat, tea::UniformBias{}, /*u=*/0,
                                /*t_prev=*/500, rng, scratch);
    EXPECT_EQ(step.v, tea::kWalkDeadSentinel);

    step = tea::sample_pat(g, pat, tea::UniformBias{}, 0, 400, rng, scratch);
    EXPECT_EQ(step.v, tea::kWalkDeadSentinel);  // ≥ newest still dies
}

TEST(SamplerPat, SingleEdgeAlwaysReturnsIt) {
    // u=0 has exactly 1 outgoing edge.
    std::vector<tea::Edge> edges = {{0, 42, 100}};
    tea::TemporalGraph g;
    g.build(std::move(edges), 100, true);
    tea::Pat<tea::UniformBias> pat;
    pat.build(g, tea::UniformBias{});
    tea::Pcg64 rng(0x0, 0);
    tea::SamplerScratch scratch;

    for (int i = 0; i < 1000; ++i) {
        auto step = tea::sample_pat(g, pat, tea::UniformBias{}, 0, -1, rng, scratch);
        EXPECT_EQ(step.v, 42);
        EXPECT_EQ(step.t, 100);
    }
}

TEST(SamplerPat, DeterministicWithFixedSeed) {
    auto g = build_star(8, 100);
    tea::Pat<tea::LinearBias> pat;
    pat.build(g, tea::LinearBias{});

    auto run_one = [&](uint64_t seed) {
        tea::Pcg64 rng(seed, 0);
        tea::SamplerScratch scratch;
        std::vector<int32_t> picks;
        for (int i = 0; i < 100; ++i) {
            auto step = tea::sample_pat(g, pat, tea::LinearBias{}, 0, -1, rng, scratch);
            picks.push_back(step.v);
        }
        return picks;
    };

    auto a = run_one(42);
    auto b = run_one(42);
    EXPECT_EQ(a, b);
    auto c = run_one(43);  // different seed
    EXPECT_NE(a, c);
}

TEST(SamplerPat, AcrossManyVerticesAndStartsAllProduceValidTargets) {
    // Build a 100-vertex graph with random connectivity, run many samples,
    // verify every returned target is one of the actual outgoing edges of u
    // at a valid timestamp.
    std::vector<tea::Edge> edges;
    tea::Pcg64 wrng(0xff, 0);
    for (int32_t u = 0; u < 100; ++u) {
        int n_edges = 1 + (wrng.next_below(30));  // [1, 30]
        for (int i = 0; i < n_edges; ++i) {
            int32_t v = static_cast<int32_t>(wrng.next_below(100));
            int64_t t = static_cast<int64_t>(1 + wrng.next_below(10000));
            edges.push_back({u, v, t});
        }
    }
    tea::TemporalGraph g;
    g.build(std::move(edges), 100, true);
    tea::Pat<tea::ExponentialBias> pat;
    pat.build(g, tea::ExponentialBias{});
    tea::Pcg64 rng(0x123, 0);
    tea::SamplerScratch scratch;

    int dead_count = 0;
    int live_count = 0;
    for (int32_t u = 0; u < 100; ++u) {
        for (int trial = 0; trial < 100; ++trial) {
            int64_t t_prev = static_cast<int64_t>(rng.next_below(10000));
            auto step = tea::sample_pat(g, pat, tea::ExponentialBias{}, u, t_prev,
                                        rng, scratch);
            if (step.v == tea::kWalkDeadSentinel) {
                ++dead_count;
                continue;
            }
            ++live_count;
            // Verify (step.v, step.t) corresponds to a real outgoing edge of u
            // with t > t_prev.
            auto targets_u = g.targets_of(u);
            auto ts_u      = g.timestamps_of(u);
            bool found = false;
            for (std::size_t i = 0; i < targets_u.size(); ++i) {
                if (targets_u[i] == step.v && ts_u[i] == step.t) {
                    found = true;
                    EXPECT_GT(ts_u[i], t_prev) << "sampled edge violates t > t_prev";
                    break;
                }
            }
            EXPECT_TRUE(found)
                << "u=" << u << " t_prev=" << t_prev
                << " sampled v=" << step.v << " t=" << step.t << " not in u's edges";
        }
    }
    // Most samples should be live; some die naturally.
    EXPECT_GT(live_count, dead_count);
}
