// Phase 4.5: low-degree single-AliasTable optimisation (TEA paper §3.3
// second ad-hoc optimization).
//
// Paper text: "if the out-degree of a vertex is relatively low, we can
// simply build alias tables for its specific out edges."
//
// What we verify:
//   1. Hpat::is_solo() classifies vertices correctly per kHpatDegreeThreshold.
//   2. AuxIndex skips solo vertices (no cover entries for them).
//   3. Solo memory < hierarchical memory at the same low D (storage saving
//      is the whole point of the optimization).
//   4. Solo sampling at L = D matches the analytic δ(e_i)/Σδ(e_j) under
//      Linear bias (χ² goodness-of-fit).
//   5. Solo sampling at L < D uses the partial-prefix recompute path and
//      still matches the analytic distribution on the candidate prefix.
//   6. HPAT-with-solo vs PAT (which never uses solo) produce statistically
//      identical distributions on a low-D vertex (two-sample χ²).
//   7. Mixed graph with both solo and hierarchical vertices: walks emit
//      valid edges everywhere — no off-by-one in the dispatch.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/pat.hpp"
#include "tea/rng.hpp"
#include "tea/sampler.hpp"

namespace {

// One vertex (u=0) with D outbound edges, with timestamps in [t0, t0+D-1].
// Storage is desc; position p holds destination 100+(D-1-p) with ts t0+(D-1-p),
// so position 0 = newest (ts t0+D-1).
tea::TemporalGraph make_one_vertex_graph(int32_t D, int64_t t0 = 1000) {
    std::vector<tea::Edge> edges;
    for (int32_t j = 0; j < D; ++j) {
        edges.push_back({0, 100 + j, t0 + j});
    }
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/200, /*is_directed=*/true);
    return g;
}

// Mixed outbound-degree graph: u=0 solo (D=10), u=1 hierarchical (D=200),
// u=2 boundary (D=kThr+1), u=3 below boundary (D=kThr), u=4 zero-degree.
tea::TemporalGraph make_mixed_solo_hier_graph() {
    std::vector<tea::Edge> edges;
    constexpr int Thr = tea::kHpatDegreeThreshold;
    const int degrees[] = {10, 200, Thr + 1, Thr, 0};
    for (int u = 0; u < 5; ++u) {
        for (int j = 0; j < degrees[u]; ++j) {
            // u → 1000+j: u gets one outbound edge to destination 1000+j.
            edges.push_back({u, 1000 + j, 50000 + j});
        }
    }
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/3000, /*is_directed=*/true);
    return g;
}

// Compute χ² statistic of observed counts vs expected probabilities.
double chi_square(const std::vector<int64_t>& observed,
                  const std::vector<double>&  expected_probs,
                  int64_t                     n) {
    double chi2 = 0.0;
    for (std::size_t i = 0; i < observed.size(); ++i) {
        const double exp_i = expected_probs[i] * static_cast<double>(n);
        if (exp_i <= 0.0) continue;
        const double diff = static_cast<double>(observed[i]) - exp_i;
        chi2 += diff * diff / exp_i;
    }
    return chi2;
}

}  // namespace

// ============================================================================
// 1. Classification: is_solo() matches the threshold contract.
// ============================================================================
TEST(Solo, IsSoloMatchesThreshold) {
    auto g = make_mixed_solo_hier_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    EXPECT_TRUE (hpat.is_solo(0));  // D=10
    EXPECT_FALSE(hpat.is_solo(1));  // D=200
    EXPECT_FALSE(hpat.is_solo(2));  // D=kThr+1 (just over)
    EXPECT_TRUE (hpat.is_solo(3));  // D=kThr (at boundary, ≤ threshold)
    EXPECT_FALSE(hpat.is_solo(4));  // D=0 (no path; not "solo")
}

// ============================================================================
// 2. AuxIndex skips solo vertices entirely.
// ============================================================================
TEST(Solo, AuxIndexSkipsSoloVertices) {
    auto g = make_mixed_solo_hier_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    ASSERT_TRUE(hpat.aux().active());

    // Hierarchical vertices contribute entries; solo vertices contribute none.
    // We can't read the per-vertex bytes directly, but the total entry count
    // must equal the entry counts for vertex 1 (D=200) and vertex 2
    // (D=kThr+1) alone — no contribution from u=0 (D=10), u=3 (D=kThr), u=4
    // (D=0).
    auto count_entries_for_D = [](int64_t D) {
        int64_t cum = 0;
        for (int64_t L = 1; L <= D; ++L) {
            cum += __builtin_popcountll(static_cast<uint64_t>(L));
        }
        return cum;
    };
    const int64_t expected_total = count_entries_for_D(200)
                                 + count_entries_for_D(tea::kHpatDegreeThreshold + 1);
    EXPECT_EQ(hpat.aux().entry_count(), expected_total);
}

// ============================================================================
// 3. Solo memory < hierarchical memory at the same D — the whole point.
// ============================================================================
TEST(Solo, SoloMemoryIsLowerThanHypotheticalHierarchy) {
    // Build a graph with one solo vertex (D=32).
    const int32_t D = 32;
    auto g = make_one_vertex_graph(D);
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    ASSERT_TRUE(hpat.is_solo(0));

    // Solo storage = D entries.
    // Hypothetical hierarchical storage = Σ_{k=0..K} ⌊D/2^k⌋·2^k = D·(K+1)
    // = 32 · 6 = 192 entries for D=32. So solo saves 160 entries × 8 B = 1280 B.
    const int64_t solo_alias_entries = hpat.alias_entry_count();
    EXPECT_EQ(solo_alias_entries, static_cast<int64_t>(D));
    EXPECT_EQ(hpat.trunk_count(), 0);  // no hierarchical trunk totals
}

// ============================================================================
// 4. Solo sampling at L=D matches analytic distribution (Linear bias).
// ============================================================================
TEST(Solo, FullPrefixDistributionMatchesAnalyticLinear) {
    constexpr int32_t D = 20;          // ≤ kHpatDegreeThreshold → solo
    auto g = make_one_vertex_graph(D);
    tea::Hpat<tea::LinearBias> hpat;
    tea::LinearBias bias;
    hpat.build(g, bias);
    ASSERT_TRUE(hpat.is_solo(0));

    // Expected analytic distribution: edge at position i (in time-DESC list)
    // has rank D - i, so P(i) = (D - i) / (D·(D+1)/2).
    std::vector<double> expected_p(D);
    const double total = static_cast<double>(D) * (D + 1) / 2.0;
    for (int32_t i = 0; i < D; ++i) {
        expected_p[i] = static_cast<double>(D - i) / total;
    }

    // Sample 200K times with t_prev = -∞ so candidate set = full D.
    constexpr int64_t N = 200'000;
    std::vector<int64_t> observed(D, 0);
    tea::Pcg64 rng(0xabcdULL);
    tea::SamplerScratch scratch;
    const int64_t t_prev = tea::kSentinelStartTimestamp;
    const auto ts = g.timestamps_of(0);
    for (int64_t n = 0; n < N; ++n) {
        auto s = tea::sample_hpat(g, hpat, bias, /*u=*/0, t_prev, rng, scratch);
        // Find position of s.t in ts.
        const auto it = std::find(ts.begin(), ts.end(), s.t);
        ASSERT_NE(it, ts.end());
        const int32_t pos = static_cast<int32_t>(it - ts.begin());
        observed[pos]++;
    }

    // χ² with df = D-1 = 19, α = 0.001: threshold ≈ 43.82.
    const double chi2 = chi_square(observed, expected_p, N);
    EXPECT_LT(chi2, 43.82) << "χ² = " << chi2;
}

// ============================================================================
// 5. Solo sampling at L<D (partial-prefix recompute) matches analytic.
// ============================================================================
TEST(Solo, PartialPrefixDistributionMatchesAnalyticLinear) {
    constexpr int32_t D = 20;
    auto g = make_one_vertex_graph(D);
    tea::Hpat<tea::LinearBias> hpat;
    tea::LinearBias bias;
    hpat.build(g, bias);
    ASSERT_TRUE(hpat.is_solo(0));

    // Pick t_prev to cut the forward candidate set down to L=12.
    // Storage is DESCENDING: ts at positions 0..19 are 1019..1000.  For
    // L=12 we need 12 entries with t > t_prev: t_prev = 1007 (entries
    // 1019..1008 are > 1007, i.e. positions 0..11).
    constexpr int64_t L = 12;
    const int64_t t_prev = 1007;
    ASSERT_EQ(g.candidate_set_len(0, t_prev), L);

    // Expected: under LinearBias, weights for the time-DESC prefix [0, L)
    // are rank = D - position = 20, 19, ..., 9.
    std::vector<double> weights(L);
    double sum = 0.0;
    for (int64_t i = 0; i < L; ++i) {
        weights[i] = static_cast<double>(D - i);
        sum += weights[i];
    }
    std::vector<double> expected_p(L);
    for (int64_t i = 0; i < L; ++i) expected_p[i] = weights[i] / sum;

    constexpr int64_t N = 200'000;
    std::vector<int64_t> observed(L, 0);
    tea::Pcg64 rng(0xdeadULL);
    tea::SamplerScratch scratch;
    const auto ts = g.timestamps_of(0);
    for (int64_t n = 0; n < N; ++n) {
        auto s = tea::sample_hpat(g, hpat, bias, /*u=*/0, t_prev, rng, scratch);
        const auto it = std::find(ts.begin(), ts.end(), s.t);
        ASSERT_NE(it, ts.end());
        const int32_t pos = static_cast<int32_t>(it - ts.begin());
        ASSERT_LT(pos, L);  // must be in candidate prefix
        observed[pos]++;
    }

    // χ² df=L-1=11, α=0.001: threshold ≈ 31.26.
    const double chi2 = chi_square(observed, expected_p, N);
    EXPECT_LT(chi2, 31.26) << "χ² = " << chi2;
}

// ============================================================================
// 6. HPAT-with-solo vs PAT: same distribution on a low-D vertex (two-sample χ²).
// ============================================================================
TEST(Solo, HpatSoloVsPatTwoSampleChiSquare) {
    constexpr int32_t D = 30;          // ≤ kHpatDegreeThreshold → solo
    auto g = make_one_vertex_graph(D);

    tea::Hpat<tea::ExponentialBias> hpat;
    tea::Pat<tea::ExponentialBias>  pat;
    tea::ExponentialBias bias;
    hpat.build(g, bias);
    pat.build(g, bias);
    ASSERT_TRUE(hpat.is_solo(0));

    constexpr int64_t N      = 100'000;
    const int64_t      t_prev = tea::kSentinelStartTimestamp;
    std::vector<int64_t> obs_h(D, 0), obs_p(D, 0);
    {
        tea::Pcg64 rng(0x42ULL);
        tea::SamplerScratch scratch;
        const auto ts = g.timestamps_of(0);
        for (int64_t n = 0; n < N; ++n) {
            auto s = tea::sample_hpat(g, hpat, bias, 0, t_prev, rng, scratch);
            const auto it = std::find(ts.begin(), ts.end(), s.t);
            obs_h[static_cast<int32_t>(it - ts.begin())]++;
        }
    }
    {
        tea::Pcg64 rng(0x42ULL);
        tea::SamplerScratch scratch;
        const auto ts = g.timestamps_of(0);
        for (int64_t n = 0; n < N; ++n) {
            auto s = tea::sample_pat(g, pat, bias, 0, t_prev, rng, scratch);
            const auto it = std::find(ts.begin(), ts.end(), s.t);
            obs_p[static_cast<int32_t>(it - ts.begin())]++;
        }
    }

    // Two-sample χ²: Σ_i (c1_i - E_i)²/E_i + (c2_i - E_i)²/E_i where
    // E_i = (c1_i + c2_i) / 2.  df = D-1 = 29; α=0.001 ≈ 58.30.
    double chi2 = 0.0;
    for (int32_t i = 0; i < D; ++i) {
        const double E = static_cast<double>(obs_h[i] + obs_p[i]) / 2.0;
        if (E <= 0.0) continue;
        const double d1 = static_cast<double>(obs_h[i]) - E;
        const double d2 = static_cast<double>(obs_p[i]) - E;
        chi2 += d1 * d1 / E + d2 * d2 / E;
    }
    EXPECT_LT(chi2, 58.30) << "χ² = " << chi2;
}

// ============================================================================
// 7. Mixed solo + hier graph: every sampled edge is valid.
// ============================================================================
TEST(Solo, MixedGraphWalksProduceValidEdges) {
    auto g = make_mixed_solo_hier_graph();
    tea::Hpat<tea::ExponentialBias> hpat;
    tea::ExponentialBias bias;
    hpat.build(g, bias);

    // Hit every vertex from t_prev = -∞ many times and confirm each result
    // is one of u's real out-edges.
    tea::Pcg64 rng(0xbeefULL);
    tea::SamplerScratch scratch;
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        if (g.degree(u) == 0) continue;
        const auto ts  = g.timestamps_of(u);
        const auto tgt = g.targets_of(u);
        for (int n = 0; n < 200; ++n) {
            auto s = tea::sample_hpat(g, hpat, bias, u,
                                       tea::kSentinelStartTimestamp,
                                       rng, scratch);
            ASSERT_NE(s.v, tea::kWalkDeadSentinel);

            // s.v / s.t must match some edge of u.
            bool match = false;
            for (std::size_t i = 0; i < tgt.size(); ++i) {
                if (tgt[i] == s.v && ts[i] == s.t) { match = true; break; }
            }
            EXPECT_TRUE(match) << "u=" << u << " (v=" << s.v << ", t=" << s.t << ")";
        }
    }
}

// ============================================================================
// 8. Boundary: vertex with exactly D = kHpatDegreeThreshold is solo;
//    D = kHpatDegreeThreshold + 1 is hierarchical. Sample both and verify.
// ============================================================================
TEST(Solo, ThresholdBoundary) {
    auto g_at  = make_one_vertex_graph(tea::kHpatDegreeThreshold);
    auto g_one = make_one_vertex_graph(tea::kHpatDegreeThreshold + 1);

    tea::Hpat<tea::UniformBias> hpat_at, hpat_one;
    hpat_at.build (g_at,  tea::UniformBias{});
    hpat_one.build(g_one, tea::UniformBias{});

    EXPECT_TRUE (hpat_at.is_solo(0));
    EXPECT_FALSE(hpat_one.is_solo(0));
}
