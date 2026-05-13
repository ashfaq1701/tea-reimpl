// Phase 4.4: HPAT structural invariants + AuxiliaryIndex cover identity +
// HPAT sampler distribution check + HPAT-vs-PAT cross-validation.
//
// Cover identity is the critical correctness property: for every Γ_len in
// [0, D_u] and every vertex u, the cover trunks produced by binary
// decomposition must EXACTLY partition [0, Γ_len) — no gaps, no overlaps.
// If decomposition has an off-by-one, the sampler would silently sample
// from a wrong distribution.
//
// Cross-validation against PAT is the strongest correctness check: both
// PAT and HPAT must produce the *same* sampling distribution under the
// same bias (because both compute P(e_i) = δ(e_i)/Σ δ exactly). Any
// drift indicates a bug in one or the other.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "tea/bias.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/pat.hpp"
#include "tea/rng.hpp"
#include "tea/sampler.hpp"

namespace {

tea::TemporalGraph build_star(int K, int64_t ts_step = 100) {
    std::vector<tea::Edge> edges;
    edges.reserve(K);
    for (int i = 0; i < K; ++i) {
        edges.push_back({0, 100 + i, static_cast<int64_t>((K - i) * ts_step)});
    }
    tea::TemporalGraph g;
    // num_vertices must cover max(u, v)+1 = max(0, 100+K-1)+1.
    g.build(std::move(edges), /*num_vertices=*/100 + K + 1, /*is_directed=*/true);
    return g;
}

}  // namespace

// ============================================================================
// AuxiliaryIndex: cover identity for ALL L ∈ [0, D]
// ============================================================================
TEST(AuxIndex, CoverPartitionsPrefixExactly) {
    // For every D ∈ [1, 100] and every L ∈ [0, D], verify the cover trunks
    // exactly tile [0, L) — every position in [0, L) appears in exactly one
    // cover trunk, and no position in [L, D) is touched.
    for (int32_t D = 1; D <= 100; ++D) {
        const int32_t K = (D == 0) ? -1
                                   : static_cast<int32_t>(std::floor(std::log2(D)));
        for (int64_t L = 0; L <= D; ++L) {
            tea::TrunkRef cover[64];
            const int32_t m = tea::decompose_to_trunks(L, K, cover);

            std::vector<int> covered(D, 0);
            for (int32_t j = 0; j < m; ++j) {
                const int32_t lvl = cover[j].level;
                const int32_t idx = cover[j].index;
                const int32_t lo  = idx * (1 << lvl);
                const int32_t hi  = lo + (1 << lvl);
                ASSERT_LE(hi, D)
                    << "D=" << D << " L=" << L << " trunk extends past D";
                for (int32_t p = lo; p < hi; ++p) {
                    ASSERT_EQ(covered[p], 0)
                        << "D=" << D << " L=" << L
                        << " pos=" << p << " covered twice";
                    covered[p] = 1;
                }
            }
            int64_t total_covered = 0;
            for (int32_t p = 0; p < D; ++p) total_covered += covered[p];
            EXPECT_EQ(total_covered, L)
                << "D=" << D << " L=" << L
                << " cover size " << total_covered << " ≠ L";

            // Confirm tile is the prefix [0, L)
            for (int32_t p = 0; p < L; ++p) {
                EXPECT_EQ(covered[p], 1)
                    << "D=" << D << " L=" << L << " missed pos=" << p;
            }
            for (int32_t p = L; p < D; ++p) {
                EXPECT_EQ(covered[p], 0)
                    << "D=" << D << " L=" << L << " spurious pos=" << p;
            }
        }
    }
}

TEST(AuxIndex, CoverIsMSBFirstOrder) {
    // The cover should be emitted MSB-first so the levels are decreasing.
    tea::TrunkRef cover[64];
    const int32_t m = tea::decompose_to_trunks(/*L=*/13, /*k_max=*/3, cover);
    // 13 = 1101 in binary → bits {3, 2, 0}
    ASSERT_EQ(m, 3);
    EXPECT_EQ(cover[0].level, 3);
    EXPECT_EQ(cover[1].level, 2);
    EXPECT_EQ(cover[2].level, 0);
    // Trunk indices: pos starts at 0
    //   level 3 set: idx = 0/8 = 0; pos += 8 → 8
    //   level 2 set: idx = 8/4 = 2; pos += 4 → 12
    //   level 0 set: idx = 12/1 = 12
    EXPECT_EQ(cover[0].index, 0);
    EXPECT_EQ(cover[1].index, 2);
    EXPECT_EQ(cover[2].index, 12);
}

// ============================================================================
// HPAT structural invariants
// ============================================================================
TEST(HpatBuild, KMaxIsFloorLog2D) {
    // K_max is only meaningful for vertices on the hierarchical path
    // (D > kHpatDegreeThreshold).  For solo vertices, K_max == -1.
    for (int K : {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 100, 128, 200, 1024}) {
        auto g = build_star(K);
        tea::Hpat<tea::UniformBias> hpat;
        hpat.build(g, tea::UniformBias{});
        if (K <= tea::kHpatDegreeThreshold) {
            EXPECT_EQ(hpat.k_max_of(0), -1) << "solo K=" << K;
            EXPECT_TRUE(hpat.is_solo(0))    << "K=" << K;
        } else {
            const int32_t expected = static_cast<int32_t>(std::floor(std::log2(K)));
            EXPECT_EQ(hpat.k_max_of(0), expected) << "hierarchical K=" << K;
            EXPECT_FALSE(hpat.is_solo(0))         << "K=" << K;
        }
    }
}

TEST(HpatBuild, AliasEntryCountMatchesFormula) {
    // Hierarchical path: D=128 → K=7 → entries = 128*(7+1) = 1024.
    auto g = build_star(128);
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    EXPECT_FALSE(hpat.is_solo(0));
    EXPECT_EQ(hpat.alias_entry_count(), tea::hpat_alias_entries_per_vertex(128));
}

TEST(HpatBuild, TrunkCountMatchesFormula) {
    // Hierarchical path: D=128 → trunks per level 128,64,32,16,8,4,2,1 = 255.
    auto g = build_star(128);
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    EXPECT_FALSE(hpat.is_solo(0));
    EXPECT_EQ(hpat.trunk_count(), tea::hpat_trunks_per_vertex(128));
}

TEST(HpatBuild, SoloPathHasNoTrunkTotalsAndAliasSizeD) {
    // Low-degree path: D=16 → single AliasTable of size 16, no trunk_totals.
    auto g = build_star(16);
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    EXPECT_TRUE(hpat.is_solo(0));
    EXPECT_EQ(hpat.alias_entry_count(), 16);
    EXPECT_EQ(hpat.trunk_count(),       0);
    EXPECT_EQ(hpat.k_max_of(0),         -1);
}

TEST(HpatBuild, AliasEntriesAreInRange) {
    // D=80 puts the vertex on the hierarchical path (above
    // kHpatDegreeThreshold).  ts_step=1 keeps exp(-(D-1)) representable.
    auto g = build_star(80, /*ts_step=*/1);
    tea::Hpat<tea::ExponentialBias> hpat;
    hpat.build(g, tea::ExponentialBias{});
    ASSERT_FALSE(hpat.is_solo(0));

    const int32_t D = 80;
    const int32_t K = hpat.k_max_of(0);
    for (int32_t k = 0; k <= K; ++k) {
        const int32_t num_trunks_k = D >> k;
        for (int32_t i = 0; i < num_trunks_k; ++i) {
            auto av = hpat.alias_view_of_trunk(g, 0, k, i);
            EXPECT_EQ(av.size(), static_cast<std::size_t>(1 << k));
            for (const auto& e : av.entries()) {
                EXPECT_GE(e.prob, 0.0f);
                EXPECT_LE(e.prob, 1.0f);
                EXPECT_LT(e.alias, av.size());
            }
            EXPECT_GT(hpat.trunk_total_of(g, 0, k, i), 0.0);
        }
    }
}

TEST(HpatBuild, EmptyGraphIsHarmless) {
    tea::TemporalGraph g;
    g.build({}, /*num_vertices=*/10, /*is_directed=*/true);
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    EXPECT_EQ(hpat.alias_entry_count(), 0);
    EXPECT_EQ(hpat.trunk_count(),       0);
    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        EXPECT_EQ(hpat.k_max_of(u), -1);
    }
}

TEST(HpatBuild, MultiVertexCoexistence) {
    // Two hierarchical-path vertices with different degrees — both must have
    // correct K, sizes, and trunks; their slices must not bleed into each
    // other.  Degrees chosen above kHpatDegreeThreshold to keep both on the
    // hierarchical (non-solo) path.
    std::vector<tea::Edge> edges;
    for (int i = 0; i < 80;  ++i) edges.push_back({0, 100 + i, 1000 + i});
    for (int i = 0; i < 200; ++i) edges.push_back({1, 1000 + i, 2000 + i});
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/3000, /*is_directed=*/true);

    tea::Hpat<tea::LinearBias> hpat;
    hpat.build(g, tea::LinearBias{});

    EXPECT_FALSE(hpat.is_solo(0));
    EXPECT_FALSE(hpat.is_solo(1));
    EXPECT_EQ(hpat.k_max_of(0), static_cast<int32_t>(std::floor(std::log2(80))));   // 6
    EXPECT_EQ(hpat.k_max_of(1), static_cast<int32_t>(std::floor(std::log2(200))));  // 7

    EXPECT_EQ(hpat.alias_entry_count(),
              tea::hpat_alias_entries_per_vertex(80)
            + tea::hpat_alias_entries_per_vertex(200));
}

TEST(HpatBuild, SoloAndHierCoexist) {
    // One solo (D=10) + one hierarchical (D=200): no cross-bleed.
    std::vector<tea::Edge> edges;
    for (int i = 0; i < 10;  ++i) edges.push_back({0, 100 + i, 1000 + i});
    for (int i = 0; i < 200; ++i) edges.push_back({1, 1000 + i, 2000 + i});
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/3000, /*is_directed=*/true);

    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    EXPECT_TRUE (hpat.is_solo(0));
    EXPECT_FALSE(hpat.is_solo(1));
    EXPECT_EQ(hpat.alias_entry_count(),
              /*solo*/ 10 + tea::hpat_alias_entries_per_vertex(200));
    EXPECT_EQ(hpat.trunk_count(), tea::hpat_trunks_per_vertex(200));
}

// ============================================================================
// HPAT sampler distribution check (mirrors test_sampler.cpp PAT cases)
// ============================================================================
namespace {

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
        std::vector<double> p(w.size(), 1.0 / w.size());
        return p;
    }
    for (auto& x : w) x /= total;
    return w;
}

double chi2(const std::vector<int64_t>& counts,
            const std::vector<double>&  expected_probs,
            int64_t N) {
    double s = 0.0;
    for (std::size_t i = 0; i < counts.size(); ++i) {
        const double exp_count = expected_probs[i] * N;
        if (exp_count <= 0.0) continue;
        const double diff = counts[i] - exp_count;
        s += diff * diff / exp_count;
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
    if (df <= 14) return 33.0;
    if (df <= 19) return 41.0;
    return df * 2.5;
}

template <typename BiasT>
void distribution_check_hpat(const tea::TemporalGraph& g,
                             const tea::Hpat<BiasT>&    hpat,
                             const BiasT&               bias,
                             int32_t                    u,
                             int64_t                    t_prev,
                             int64_t                    N_samples,
                             uint64_t                   seed,
                             const std::vector<double>& expected_probs,
                             int64_t                    partial_start) {
    tea::Pcg64 rng(seed, 0xa);
    tea::SamplerScratch scratch;
    auto targets_u = g.targets_of(u);
    const std::size_t K = expected_probs.size();
    std::vector<int64_t> counts(K, 0);
    int64_t dead = 0;
    for (int64_t i = 0; i < N_samples; ++i) {
        auto step = tea::sample_hpat(g, hpat, bias, u, t_prev, rng, scratch);
        if (step.v == tea::kWalkDeadSentinel) { ++dead; continue; }
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
    EXPECT_LT(c2, th) << "chi2=" << c2 << " threshold=" << th
                       << " bias=" << BiasT::name();
}

}  // namespace

TEST(SamplerHpat, UniformDistribution_FullVertex) {
    auto g = build_star(16);
    tea::Hpat<tea::UniformBias> hpat;
    tea::UniformBias bias;
    hpat.build(g, bias);
    auto p = analytic_probs(g, bias, 0);
    distribution_check_hpat(g, hpat, bias, 0, -1, 200000, 0x1111, p, 0);
}

TEST(SamplerHpat, UniformDistribution_PrefixOfNonPowerLen) {
    // Γ_len = 13 — exercises a cover of 3 trunks (8+4+1)
    auto g = build_star(16, 100);
    tea::Hpat<tea::UniformBias> hpat;
    tea::UniformBias bias;
    hpat.build(g, bias);
    // ts_desc = [1600..100]. Γ_len=13 → t_prev < ts[12] = 400 and ≥ ts[13] = 300
    auto p = analytic_probs(g, bias, 0, 0, 13);
    distribution_check_hpat(g, hpat, bias, 0, /*t_prev=*/350, 300000, 0x2222, p, 0);
}

TEST(SamplerHpat, LinearDistribution_NonPowerLen) {
    auto g = build_star(16);
    tea::Hpat<tea::LinearBias> hpat;
    tea::LinearBias bias;
    hpat.build(g, bias);
    auto p = analytic_probs(g, bias, 0, 0, 11);
    distribution_check_hpat(g, hpat, bias, 0, 550, 300000, 0x3333, p, 0);
}

TEST(SamplerHpat, ExponentialDistribution) {
    auto g = build_star(16, 1);
    tea::Hpat<tea::ExponentialBias> hpat;
    tea::ExponentialBias bias;
    hpat.build(g, bias);
    auto p = analytic_probs(g, bias, 0);
    // Under forward-walk semantics the oldest edge (last in DESC list)
    // carries the highest weight.
    EXPECT_LT(p[0], p[1]);
    distribution_check_hpat(g, hpat, bias, 0, -1, 300000, 0x4444, p, 0);
}

// ============================================================================
// HPAT vs PAT cross-validation — strongest correctness check
// ============================================================================
TEST(HpatVsPat, EmpiricalDistributionsAgree) {
    // Build BOTH on the same graph + bias. Run 200K samples from each.
    // Empirical edge-pick frequencies should agree within Monte Carlo noise.
    auto g = build_star(20);
    tea::ExponentialBias bias;
    tea::Pat<tea::ExponentialBias>  pat;   pat.build(g, bias);
    tea::Hpat<tea::ExponentialBias> hpat;  hpat.build(g, bias);

    constexpr int N = 200000;
    auto run = [&](auto sample_fn) -> std::vector<int64_t> {
        tea::Pcg64 rng(0xfade, 0);
        tea::SamplerScratch scratch;
        std::vector<int64_t> counts(20, 0);
        auto targets_u = g.targets_of(0);
        for (int i = 0; i < N; ++i) {
            auto step = sample_fn(rng, scratch);
            auto it = std::find(targets_u.begin(), targets_u.end(), step.v);
            EXPECT_NE(it, targets_u.end());
            if (it != targets_u.end()) ++counts[it - targets_u.begin()];
        }
        return counts;
    };

    auto pat_counts  = run([&](tea::Pcg64& rng, tea::SamplerScratch& s) {
        return tea::sample_pat(g, pat, bias, 0, -1, rng, s);
    });
    auto hpat_counts = run([&](tea::Pcg64& rng, tea::SamplerScratch& s) {
        return tea::sample_hpat(g, hpat, bias, 0, -1, rng, s);
    });

    // Proper two-sample chi-square test of distribution equality.
    // H0: PAT and HPAT produce the same distribution over edge positions.
    // Test stat: Σ_i [(c1_i - E1_i)² / E1_i + (c2_i - E2_i)² / E2_i]
    // where E_j_i = (c1_i + c2_i) × N_j / (N_1 + N_2).
    // df = K - 1.
    double c2 = 0.0;
    for (std::size_t i = 0; i < pat_counts.size(); ++i) {
        const int64_t c_total = pat_counts[i] + hpat_counts[i];
        if (c_total == 0) continue;
        const double E = static_cast<double>(c_total) * 0.5;  // N1 == N2 == N
        const double d_pat  = pat_counts[i]  - E;
        const double d_hpat = hpat_counts[i] - E;
        c2 += (d_pat  * d_pat)  / E;
        c2 += (d_hpat * d_hpat) / E;
    }
    EXPECT_LT(c2, chi2_threshold(19))
        << "PAT and HPAT empirical distributions disagree, chi2=" << c2;
}

TEST(HpatVsPat, BothAgreeOnPartialPrefix) {
    auto g = build_star(20);
    tea::LinearBias bias;
    tea::Pat<tea::LinearBias>  pat;   pat.build(g, bias);
    tea::Hpat<tea::LinearBias> hpat;  hpat.build(g, bias);

    // Γ_len = 13 by choosing t_prev between ts[12] and ts[13].
    // ts_step=100, ts_desc = [2000, 1900, ..., 100]. ts[12]=800, ts[13]=700.
    constexpr int N = 200000;
    constexpr int64_t T_PREV = 750;

    auto run = [&](auto sample_fn) -> std::vector<int64_t> {
        tea::Pcg64 rng(0xbabe, 0);
        tea::SamplerScratch scratch;
        std::vector<int64_t> counts(13, 0);  // only first 13 of 20 should fire
        auto targets_u = g.targets_of(0);
        int dead = 0;
        for (int i = 0; i < N; ++i) {
            auto step = sample_fn(rng, scratch);
            if (step.v == tea::kWalkDeadSentinel) { ++dead; continue; }
            auto it = std::find(targets_u.begin(), targets_u.end(), step.v);
            EXPECT_NE(it, targets_u.end());
            if (it == targets_u.end()) continue;
            const int64_t pos = it - targets_u.begin();
            EXPECT_LT(pos, 13) << "sampled an edge outside Γ_t(u)!";
            if (pos >= 13) continue;
            ++counts[pos];
        }
        EXPECT_EQ(dead, 0);
        return counts;
    };

    auto pat_counts  = run([&](tea::Pcg64& rng, tea::SamplerScratch& s) {
        return tea::sample_pat(g, pat, bias, 0, T_PREV, rng, s);
    });
    auto hpat_counts = run([&](tea::Pcg64& rng, tea::SamplerScratch& s) {
        return tea::sample_hpat(g, hpat, bias, 0, T_PREV, rng, s);
    });

    double c2 = 0.0;
    for (std::size_t i = 0; i < pat_counts.size(); ++i) {
        const int64_t c_total = pat_counts[i] + hpat_counts[i];
        if (c_total == 0) continue;
        const double E = static_cast<double>(c_total) * 0.5;
        const double d_pat  = pat_counts[i]  - E;
        const double d_hpat = hpat_counts[i] - E;
        c2 += (d_pat  * d_pat)  / E;
        c2 += (d_hpat * d_hpat) / E;
    }
    EXPECT_LT(c2, chi2_threshold(12))
        << "PAT and HPAT disagree on partial-prefix sampling, chi2=" << c2;
}
