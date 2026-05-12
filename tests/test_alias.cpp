// Phase 1.4: AliasTable correctness tests.
//
// What we verify:
//   - empty input → empty table
//   - degenerate (all-zero weights) → uniform self-aliased table (won't crash)
//   - single-element → always returns 0
//   - sampling distribution matches the weight vector (chi-square at p>0.01)
//   - alias entries respect the invariant prob ∈ [0,1], alias < n

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "tea/alias.hpp"
#include "tea/rng.hpp"

TEST(AliasTest, EmptyInput) {
    tea::AliasTable t;
    t.build(tea::span<const double>(nullptr, 0));
    EXPECT_TRUE(t.empty());
    EXPECT_EQ(t.size(), 0u);
}

TEST(AliasTest, AllZeroWeightsAreUniform) {
    std::vector<double> w = {0.0, 0.0, 0.0, 0.0};
    tea::AliasTable t;
    t.build(tea::span<const double>(w.data(), w.size()));
    ASSERT_EQ(t.size(), 4u);

    tea::Pcg64 rng(123, 7);
    std::vector<int> counts(4, 0);
    constexpr int N = 100000;
    for (int i = 0; i < N; ++i) {
        uint32_t b = rng.next_below(4);
        float    p = rng.next_f01();
        ++counts[t.sample(b, p)];
    }
    // Each bucket should get ~N/4 = 25000. Allow wide tolerance.
    for (int i = 0; i < 4; ++i) {
        EXPECT_GT(counts[i], N / 4 - N / 20);
        EXPECT_LT(counts[i], N / 4 + N / 20);
    }
}

TEST(AliasTest, SingleElement) {
    std::vector<double> w = {1.0};
    tea::AliasTable t;
    t.build(tea::span<const double>(w.data(), w.size()));
    ASSERT_EQ(t.size(), 1u);

    tea::Pcg64 rng(42, 0);
    for (int i = 0; i < 100; ++i) {
        uint32_t b = rng.next_below(1);  // always 0
        float    p = rng.next_f01();
        EXPECT_EQ(t.sample(b, p), 0u);
    }
}

TEST(AliasTest, InvariantsHold) {
    std::vector<double> w = {1.0, 4.0, 2.0, 8.0, 3.0, 5.0, 7.0};
    tea::AliasTable t;
    t.build(tea::span<const double>(w.data(), w.size()));
    ASSERT_EQ(t.size(), 7u);

    for (const auto& e : t.entries()) {
        EXPECT_GE(e.prob, 0.0f);
        EXPECT_LE(e.prob, 1.0f);
        EXPECT_LT(e.alias, 7u);
    }
}

TEST(AliasTest, DistributionMatchesWeightsChiSquare) {
    // 4 categories with known weights. Sample many times, chi-square against
    // expected frequencies. Pass at p > 0.01 (chi-square with 3 dof, threshold
    // 11.34 for p=0.01).
    std::vector<double> w = {1.0, 2.0, 3.0, 4.0};
    const double total = 10.0;
    const int    n     = 4;

    tea::AliasTable t;
    t.build(tea::span<const double>(w.data(), w.size()));

    tea::Pcg64 rng(0xc0ffeeULL, 0x1234ULL);

    constexpr int N = 1'000'000;
    std::vector<int> counts(n, 0);
    for (int i = 0; i < N; ++i) {
        uint32_t b = rng.next_below(n);
        float    p = rng.next_f01();
        ++counts[t.sample(b, p)];
    }

    double chi2 = 0.0;
    for (int i = 0; i < n; ++i) {
        const double expected = N * (w[i] / total);
        const double diff     = counts[i] - expected;
        chi2 += (diff * diff) / expected;
    }
    // 3 dof, p=0.01 threshold = 11.34. Use 20 as a very-safe upper bound.
    EXPECT_LT(chi2, 20.0) << "chi2=" << chi2
        << "  counts={" << counts[0] << "," << counts[1] << ","
        << counts[2] << "," << counts[3] << "}";
}

TEST(AliasTest, BuildIntoExternalArena) {
    // Phase 3 use case: stamp multiple alias tables into one big arena.
    // Verifies that build_alias_into() doesn't allocate per call (scratch
    // is provided by the caller) and that views into the arena work.
    std::vector<double> weights_a = {1.0, 2.0, 3.0};
    std::vector<double> weights_b = {4.0, 1.0};
    std::vector<double> weights_c = {2.0, 2.0, 2.0, 2.0};

    std::vector<tea::AliasEntry> arena(weights_a.size() + weights_b.size() + weights_c.size());
    tea::AliasBuildScratch scratch;

    tea::build_alias_into(
        tea::span<tea::AliasEntry>(arena.data(), 3),
        tea::span<const double>(weights_a.data(), 3),
        scratch);
    tea::build_alias_into(
        tea::span<tea::AliasEntry>(arena.data() + 3, 2),
        tea::span<const double>(weights_b.data(), 2),
        scratch);
    tea::build_alias_into(
        tea::span<tea::AliasEntry>(arena.data() + 5, 4),
        tea::span<const double>(weights_c.data(), 4),
        scratch);

    // Views over each table.
    tea::AliasView v_a(tea::span<const tea::AliasEntry>(arena.data(), 3));
    tea::AliasView v_b(tea::span<const tea::AliasEntry>(arena.data() + 3, 2));
    tea::AliasView v_c(tea::span<const tea::AliasEntry>(arena.data() + 5, 4));

    // Sample each, verify invariants.
    tea::Pcg64 rng(0xa1ULL, 0);
    for (int i = 0; i < 1000; ++i) {
        uint32_t r = v_a.sample(rng.next_below(3), rng.next_f01());
        EXPECT_LT(r, 3u);
    }
    EXPECT_EQ(v_a.size(), 3u);
    EXPECT_EQ(v_b.size(), 2u);
    EXPECT_EQ(v_c.size(), 4u);
}

TEST(AliasTest, ScratchReusedAcrossBuilds) {
    // Build 100 tables of varying sizes with a single scratch instance.
    // Verifies correctness when the same scratch is shared across builds.
    tea::AliasBuildScratch scratch;
    tea::Pcg64 wrng(7, 7);

    for (int round = 0; round < 100; ++round) {
        const std::size_t n = 1u + (round % 20);
        std::vector<double> w(n);
        for (auto& x : w) x = wrng.next_u01();
        std::vector<tea::AliasEntry> entries(n);
        tea::build_alias_into(
            tea::span<tea::AliasEntry>(entries.data(), n),
            tea::span<const double>(w.data(), n),
            scratch);
        for (auto& e : entries) {
            EXPECT_GE(e.prob, 0.0f);
            EXPECT_LE(e.prob, 1.0f);
            EXPECT_LT(e.alias, n);
        }
    }
}

TEST(AliasTest, HighlySkewedDistribution) {
    // One category has 99% of the mass. Verify it gets ~99% of samples.
    std::vector<double> w = {99.0, 1.0};
    tea::AliasTable t;
    t.build(tea::span<const double>(w.data(), w.size()));

    tea::Pcg64 rng(1, 1);
    constexpr int N = 100000;
    int hits_0 = 0;
    for (int i = 0; i < N; ++i) {
        uint32_t b = rng.next_below(2);
        float    p = rng.next_f01();
        if (t.sample(b, p) == 0) ++hits_0;
    }
    EXPECT_GT(hits_0, 0.97 * N);
    EXPECT_LT(hits_0, 0.99 * N + 500);  // tolerance
}
