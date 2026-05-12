// Phase 0 smoke test: proves the build wiring works (CMake + gtest +
// tea library link) AND that the Phase 0.2 headers compile cleanly on
// their own. Replaced by real unit tests starting in Phase 1.4.

#include <gtest/gtest.h>

#include "tea/edge.hpp"
#include "tea/walk.hpp"
#include "tea/rng.hpp"
#include "tea/config.hpp"

TEST(Phase0_Build, GTestLinksAndRuns) {
    EXPECT_EQ(2 + 2, 4);
}

TEST(Phase0_Edge, IsPackedCorrectly) {
    EXPECT_EQ(sizeof(tea::Edge),     16u);
    EXPECT_EQ(sizeof(tea::NodeStep), 16u);
}

TEST(Phase0_Rng, PerThreadStateIsCacheLineAligned) {
    EXPECT_EQ(alignof(tea::PerThreadRng), tea::kCacheLine);
}

TEST(Phase0_Rng, ProducesNonTrivialOutput) {
    tea::Pcg64 rng(0xdeadbeefULL, 0x5eed5eedULL);
    // Just sanity-check the surface area: distinct draws, in-range floats.
    uint64_t a = rng.next_u64();
    uint64_t b = rng.next_u64();
    EXPECT_NE(a, b);

    double  d = rng.next_u01();
    float   f = rng.next_f01();
    EXPECT_GE(d, 0.0); EXPECT_LT(d, 1.0);
    EXPECT_GE(f, 0.0f); EXPECT_LT(f, 1.0f);

    uint32_t n = rng.next_below(100);
    EXPECT_LT(n, 100u);
}

TEST(Phase0_Config, ConstantsAreSensible) {
    EXPECT_GT(tea::kAuxIndexMaxBytes,       size_t{1});
    EXPECT_GE(tea::kPatTrunkSizeMin,        2);
    EXPECT_GT(tea::kExpBiasRescaleTarget,   0.0);
    EXPECT_GT(tea::kNode2VecMaxRetries,     0);
}
