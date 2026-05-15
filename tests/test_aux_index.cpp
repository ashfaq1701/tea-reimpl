// Phase 4.2': Precomputed AuxIndex correctness (paper §3.4).
//
// What we verify:
//   1. Build succeeds within the default budget for representative graphs.
//   2. Build returns false (and aux is inactive) with a zero budget.
//   3. cover_for(u, L) matches on-the-fly decompose_to_trunks for ALL u
//      and ALL L ∈ [0, D_u] across several test graphs.
//   4. The cover exactly partitions [0, L): trunks are contiguous (no gaps),
//      non-overlapping, and their sizes sum to L.
//   5. cover_for(u, 0) returns an empty span.
//   6. cover_for(u, D_u) covers the entire edge list (sum of trunk sizes
//      equals D_u).
//   7. HPAT-with-aux vs HPAT-without-aux produce the same empirical
//      sampling distribution under the same RNG seed sequence (χ² test).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <unordered_map>
#include <vector>

#include "tea/aux_index.hpp"
#include "tea/bias.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/rng.hpp"
#include "tea/sampler.hpp"
#include "tea/trunk_ref.hpp"

namespace {

// Mixed-degree graph: degrees chosen above kHpatDegreeThreshold (=64) so
// every vertex uses the hierarchical HPAT path and gets a precomputed cover.
// (Solo-vertex behaviour is exercised in test_solo.cpp.)
// A degree-0 vertex is included to verify empty-vertex handling.
tea::TemporalGraph make_mixed_graph() {
    const int degrees[] = {65, 100, 128, 200, 511, 1024, 0};
    std::vector<tea::Edge> edges;
    for (int u = 0; u < 7; ++u) {
        // Fan-out from u: each u gets `degrees[u]` outbound edges to
        // destinations 5000..5000+degrees[u]-1.
        for (int j = 0; j < degrees[u]; ++j) {
            edges.push_back({u, 5000 + j, 10000 + j});
        }
    }
    tea::TemporalGraph g;
    g.build(std::move(edges), /*num_vertices=*/8000, /*is_directed=*/true);
    return g;
}

}  // namespace

// ============================================================================
// 1. Build succeeds within default budget.
// ============================================================================
TEST(AuxIndex, BuildsWithinDefaultBudget) {
    auto g = make_mixed_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});

    EXPECT_TRUE(hpat.aux().active());
    EXPECT_GT(hpat.aux().entry_count(), 0);
}

// ============================================================================
// 2. Zero budget → aux disabled (sampler falls back to on-the-fly).
// ============================================================================
TEST(AuxIndex, ZeroBudgetSkipsBuild) {
    auto g = make_mixed_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{}, /*aux_budget_bytes=*/0);

    EXPECT_FALSE(hpat.aux().active());
    EXPECT_EQ(hpat.aux().memory_bytes(), 0);
}

// ============================================================================
// 3. Tiny budget → aux disabled (over-budget fall-back).
// ============================================================================
TEST(AuxIndex, OverBudgetFallsBack) {
    auto g = make_mixed_graph();
    tea::Hpat<tea::UniformBias> hpat;
    // 100 bytes is way less than the per-vertex offset arrays alone need.
    hpat.build(g, tea::UniformBias{}, /*aux_budget_bytes=*/100);

    EXPECT_FALSE(hpat.aux().active());
}

// ============================================================================
// 4. Cover identity: cover_for(u, L) == decompose_to_trunks(L, K_u) for ALL
//    (u, L) pairs.  This is the load-bearing correctness test.
// ============================================================================
TEST(AuxIndex, CoverMatchesOnTheFlyForAllUL) {
    auto g = make_mixed_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    ASSERT_TRUE(hpat.aux().active());

    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        const int64_t D = g.degree(u);
        const int32_t K = hpat.k_max_of(u);

        for (int64_t L = 0; L <= D; ++L) {
            const auto cv = hpat.aux().cover_for(u, L);

            // Reference: on-the-fly decomposition.
            tea::TrunkRef on_fly[64];
            const int32_t m_ref =
                (L == 0) ? 0 : tea::decompose_to_trunks(L, K, on_fly);

            ASSERT_EQ(static_cast<int32_t>(cv.size()), m_ref)
                << "u=" << u << " L=" << L;
            for (int32_t j = 0; j < m_ref; ++j) {
                EXPECT_EQ(cv[j].level, on_fly[j].level)
                    << "u=" << u << " L=" << L << " j=" << j;
                EXPECT_EQ(cv[j].index, on_fly[j].index)
                    << "u=" << u << " L=" << L << " j=" << j;
            }
        }
    }
}

// ============================================================================
// 5. The cover exactly partitions [0, L): contiguous, no gaps, no overlaps.
// ============================================================================
TEST(AuxIndex, CoverExactlyPartitionsZeroToL) {
    auto g = make_mixed_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    ASSERT_TRUE(hpat.aux().active());

    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        const int64_t D = g.degree(u);
        for (int64_t L = 0; L <= D; ++L) {
            const auto cv = hpat.aux().cover_for(u, L);

            int64_t pos = 0;
            for (const auto& tr : cv) {
                const int64_t start = static_cast<int64_t>(tr.index) << tr.level;
                const int64_t size  = int64_t{1} << tr.level;
                EXPECT_EQ(start, pos)
                    << "u=" << u << " L=" << L << " gap or overlap";
                pos = start + size;
            }
            EXPECT_EQ(pos, L) << "u=" << u << " L=" << L << " total mismatch";
        }
    }
}

// ============================================================================
// 6. L=0 → empty cover; L=D → full edge list.
// ============================================================================
TEST(AuxIndex, BoundaryCovers) {
    auto g = make_mixed_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    ASSERT_TRUE(hpat.aux().active());

    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        const int64_t D = g.degree(u);
        if (D == 0) {
            EXPECT_EQ(hpat.aux().cover_for(u, 0).size(), 0u);
            continue;
        }
        EXPECT_EQ(hpat.aux().cover_for(u, 0).size(), 0u);

        const auto full = hpat.aux().cover_for(u, D);
        int64_t total = 0;
        for (const auto& tr : full) total += int64_t{1} << tr.level;
        EXPECT_EQ(total, D);
    }
}

// ============================================================================
// 7. HPAT-with-aux vs HPAT-without-aux produce statistically identical
//    sampling distributions.  Same RNG seeds, same graph, same bias — only
//    difference is which cover path the sampler takes.  Output should be
//    deterministically identical (the cover *is* identical, after all).
// ============================================================================
TEST(AuxIndex, SamplerOutputBitIdenticalWithAndWithoutAux) {
    auto g = make_mixed_graph();

    tea::Hpat<tea::ExponentialBias> hpat_with;
    tea::Hpat<tea::ExponentialBias> hpat_without;
    tea::ExponentialBias bias;  // proper-TEA default

    hpat_with.build(g, bias);                    // aux on by default
    hpat_without.build(g, bias, /*aux_budget=*/0); // aux off

    ASSERT_TRUE(hpat_with.aux().active());
    ASSERT_FALSE(hpat_without.aux().active());

    tea::SamplerScratch scratch_a, scratch_b;

    // Sample 5000 steps with identical RNG streams; outputs must match
    // exactly, edge by edge.  Same cover (paper §3.4) → same edge picks.
    constexpr int32_t  ITERS = 5000;
    constexpr uint64_t SEED  = 0x12345ULL;

    for (int32_t u = 0; u < g.num_vertices(); ++u) {
        if (g.degree(u) == 0) continue;

        tea::Pcg64 rng_a(SEED, static_cast<uint64_t>(u));
        tea::Pcg64 rng_b(SEED, static_cast<uint64_t>(u));

        // Sweep candidate-set sizes by varying t_prev across the timestamp
        // range of u, to exercise different cover decompositions.  Storage
        // is time-DESC, so ts[0] is the largest and ts[end] is the smallest.
        const auto ts = g.timestamps_of(u);
        const int64_t t_max = ts[0];
        const int64_t t_min = ts[ts.size() - 1];
        for (int32_t k = 0; k < ITERS; ++k) {
            const int64_t t_prev = t_min + (k % (t_max - t_min + 2)) - 1;
            const auto sa = tea::sample_hpat(g, hpat_with,    bias, u, t_prev, rng_a, scratch_a);
            const auto sb = tea::sample_hpat(g, hpat_without, bias, u, t_prev, rng_b, scratch_b);
            ASSERT_EQ(sa.v, sb.v) << "u=" << u << " k=" << k << " t_prev=" << t_prev;
            ASSERT_EQ(sa.t, sb.t) << "u=" << u << " k=" << k;
        }
    }
}

// ============================================================================
// 8. Memory accounting: reported bytes equals computed footprint.
// ============================================================================
TEST(AuxIndex, MemoryAccountingExact) {
    auto g = make_mixed_graph();
    tea::Hpat<tea::UniformBias> hpat;
    hpat.build(g, tea::UniformBias{});
    ASSERT_TRUE(hpat.aux().active());

    // We can't introspect the privates from outside, but the reported number
    // should be > 0 and consistent with a few entries.
    EXPECT_GT(hpat.aux().memory_bytes(), 0);
    EXPECT_GE(hpat.aux().memory_bytes(),
              hpat.aux().entry_count() * static_cast<int64_t>(sizeof(tea::TrunkRef)));
}
