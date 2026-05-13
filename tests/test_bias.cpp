// Phase 2.2 (post-review): Bias correctness + numerical-stability tests.
//
// Forward-walk semantics (matches Tempest):
//   • Linear: rank = position+1 in DESC list → smallest-t edge wins.
//   • Exponential: exp((t_min − t) × scale) → smallest-t edge wins,
//     no overflow on unix-timestamp data.
//   • timescale_bound > 0 enables Tempest-compat temperature scaling.

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <numeric>
#include <vector>

#include "tea/bias.hpp"
#include "tea/config.hpp"

namespace {

template <typename Bias>
std::vector<double> compute(const Bias& b, const std::vector<int64_t>& ts_desc) {
    std::vector<double> w(ts_desc.size(), 0.0);
    b.compute_weights_full(tea::span<const int64_t>(ts_desc.data(), ts_desc.size()),
                           w.data());
    return w;
}

}  // namespace

// ============================================================================
// UniformBias
// ============================================================================
TEST(UniformBias, AllWeightsAreOne) {
    tea::UniformBias b;
    auto w = compute(b, {300, 200, 100});
    EXPECT_EQ(w.size(), 3u);
    EXPECT_DOUBLE_EQ(w[0], 1.0);
    EXPECT_DOUBLE_EQ(w[1], 1.0);
    EXPECT_DOUBLE_EQ(w[2], 1.0);
}

TEST(UniformBias, EmptyInputIsNoOp) {
    tea::UniformBias b;
    auto w = compute(b, {});
    EXPECT_EQ(w.size(), 0u);
}

TEST(UniformBias, PartialTrunkUseCase) {
    // Sample-time partial-trunk path: we ask the bias to compute weights for
    // a sub-range of the full edge list. Uniform should return 1's regardless.
    tea::UniformBias b;
    std::vector<int64_t> ts_desc = {500, 400, 300, 200, 100};
    auto params = b.compute_per_vertex_params(
        tea::span<const int64_t>(ts_desc.data(), ts_desc.size()));

    // Partial trunk: positions [2, 5) of the full list.
    std::vector<double> w(3, 0.0);
    tea::span<const int64_t> sub(ts_desc.data() + 2, 3);
    b.compute_weights(sub, params, /*slice_start_pos=*/2, w.data());
    EXPECT_DOUBLE_EQ(w[0], 1.0);
    EXPECT_DOUBLE_EQ(w[1], 1.0);
    EXPECT_DOUBLE_EQ(w[2], 1.0);
}

// ============================================================================
// LinearBias — δ(e) = rank, newest = 1, oldest = D  (forward-walk semantics)
// ============================================================================
TEST(LinearBias, RankPattern) {
    tea::LinearBias b;
    auto w = compute(b, {300, 200, 100});  // ts_desc, ts=300 is newest
    EXPECT_DOUBLE_EQ(w[0], 1.0);
    EXPECT_DOUBLE_EQ(w[1], 2.0);
    EXPECT_DOUBLE_EQ(w[2], 3.0);
}

TEST(LinearBias, InvariantUnderAffineTimeShift) {
    // Rank doesn't depend on actual t values, only on ordering.
    tea::LinearBias b;
    auto w_small = compute(b, {3, 2, 1});
    auto w_big   = compute(b, {1700000000LL, 1500000000LL, 1000000000LL});
    EXPECT_EQ(w_small, w_big);
}

TEST(LinearBias, EmptyAndSingleEdge) {
    tea::LinearBias b;
    EXPECT_EQ(compute(b, {}).size(), 0u);
    auto one = compute(b, {42});
    ASSERT_EQ(one.size(), 1u);
    EXPECT_DOUBLE_EQ(one[0], 1.0);
}

TEST(LinearBias, PartialTrunkConsistentWithFullBuild) {
    // The critical correctness property: weights computed for a sub-range
    // at sample time MUST match the weights used at build time for the
    // same positions in the full edge list. Otherwise PAT's prefix-sum
    // gets out of sync with the alias tables.
    tea::LinearBias b;
    std::vector<int64_t> ts_desc = {500, 400, 300, 200, 100};  // D=5
    auto params = b.compute_per_vertex_params(
        tea::span<const int64_t>(ts_desc.data(), ts_desc.size()));

    // Full weights (build time): [1, 2, 3, 4, 5]  (rank = position + 1)
    std::vector<double> w_full(5, 0.0);
    b.compute_weights(tea::span<const int64_t>(ts_desc.data(), ts_desc.size()),
                      params, 0, w_full.data());
    EXPECT_DOUBLE_EQ(w_full[0], 1.0);
    EXPECT_DOUBLE_EQ(w_full[4], 5.0);

    // Sub-range positions [2, 5) — should give [3, 4, 5] matching w_full[2:5].
    std::vector<double> w_sub(3, 0.0);
    tea::span<const int64_t> sub(ts_desc.data() + 2, 3);
    b.compute_weights(sub, params, 2, w_sub.data());
    EXPECT_DOUBLE_EQ(w_sub[0], w_full[2]);
    EXPECT_DOUBLE_EQ(w_sub[1], w_full[3]);
    EXPECT_DOUBLE_EQ(w_sub[2], w_full[4]);
}

// ============================================================================
// ExponentialBias — δ(e) = exp((t_min − t) × scale), softmax peaked at oldest
// ============================================================================
TEST(ExponentialBias, OldestGetsWeightOne) {
    // δ(e_oldest) = exp(t_min − t_min) = exp(0) = 1. Never overflows.
    // Newer edges have weight < 1 (exponent negative).
    tea::ExponentialBias b;  // default timescale_bound = -1 → proper TEA
    auto w = compute(b, {300, 200, 100});
    EXPECT_DOUBLE_EQ(w[2], 1.0);
    EXPECT_LT(w[0], w[1]);
    EXPECT_LT(w[1], w[2]);
}

TEST(ExponentialBias, MonotoneInTimestamp) {
    tea::ExponentialBias b;
    auto w = compute(b, {500, 400, 300, 200, 100});
    for (std::size_t i = 1; i < w.size(); ++i) {
        EXPECT_LT(w[i - 1], w[i]) << "weights should be monotone increasing "
                                     "with position (smallest-t wins)";
    }
}

TEST(ExponentialBias, ExactValuesFor4PointInput) {
    // Hand-check: ts_desc = [10, 7, 4, 1], t_min = 1.
    // weights = [exp(1-10), exp(1-7), exp(1-4), exp(0)]
    //         = [exp(-9),  exp(-6),  exp(-3),  exp(0)]
    tea::ExponentialBias b;
    auto w = compute(b, {10, 7, 4, 1});
    EXPECT_NEAR(w[0], std::exp(-9.0), 1e-12);
    EXPECT_NEAR(w[1], std::exp(-6.0), 1e-12);
    EXPECT_NEAR(w[2], std::exp(-3.0), 1e-12);
    EXPECT_NEAR(w[3], std::exp(0.0),  1e-12);
}

TEST(ExponentialBias, NoOverflowOnUnixSecondTimestamps) {
    // tgbl-comment ts range was [1.13e9, 1.29e9]. Verify no NaN/inf.
    // Pivoting at t_min keeps every exponent ≤ 0 → largest weight is 1 (at
    // the oldest edge), newer edges underflow to 0 for huge spans — OK.
    tea::ExponentialBias b;
    std::vector<int64_t> ts_desc;
    ts_desc.reserve(1000);
    const int64_t t0 = 1'400'000'000LL;
    const int64_t span = 5LL * 365LL * 24LL * 3600LL;
    for (int i = 0; i < 1000; ++i) {
        ts_desc.push_back(t0 + span - i * (span / 999));
    }
    auto w = compute(b, ts_desc);
    EXPECT_DOUBLE_EQ(w[ts_desc.size() - 1], 1.0);  // oldest pinned at exp(0)
    for (double x : w) {
        EXPECT_FALSE(std::isnan(x));
        EXPECT_FALSE(std::isinf(x));
        EXPECT_GE(x, 0.0);
        EXPECT_LE(x, 1.0);
    }
}

TEST(ExponentialBias, NewerEdgesUnderflowToZeroOnLongSpan) {
    // 7-year span in unix seconds → newer-edge weights should underflow.
    // This is correct behavior — exp(-2e8) is genuinely zero in float64.
    tea::ExponentialBias b;
    std::vector<int64_t> ts_desc = {
        1'700'000'000LL,
        1'400'000'000LL,
        1'000'000'000LL,
    };
    auto w = compute(b, ts_desc);
    EXPECT_DOUBLE_EQ(w[0], 0.0);  // exp(-7e8)
    EXPECT_DOUBLE_EQ(w[1], 0.0);  // exp(-4e8)
    EXPECT_DOUBLE_EQ(w[2], 1.0);  // exp(0) — oldest is pinned
}

TEST(ExponentialBias, ConstantTimestampsProduceUniform) {
    tea::ExponentialBias b;
    auto w = compute(b, {500, 500, 500, 500});
    EXPECT_DOUBLE_EQ(w[0], 1.0);
    EXPECT_DOUBLE_EQ(w[1], 1.0);
    EXPECT_DOUBLE_EQ(w[2], 1.0);
    EXPECT_DOUBLE_EQ(w[3], 1.0);
}

TEST(ExponentialBias, TimescaleBoundCompresses) {
    // Tempest-compat mode: timescale_bound = 80 rescales per-vertex span
    // to span 80 in the exponent. ts in [100, 500] becomes exponent in
    // [-80, 0] regardless of the unix scale.  Pivot is t_min=100, so:
    // exponent at position i = (t_min - ts[i]) * scale, where scale=80/400=0.2.
    // ts_desc = [500, 400, 300, 200, 100] →
    // exponents  = [-80, -60, -40, -20, 0].
    tea::ExponentialBias b;
    b.timescale_bound = 80.0;
    auto w = compute(b, {500, 400, 300, 200, 100});  // span = 400
    EXPECT_NEAR(w[0], std::exp(-80.0), 1e-30);
    EXPECT_NEAR(w[1], std::exp(-60.0), 1e-25);
    EXPECT_NEAR(w[2], std::exp(-40.0), 1e-20);
    EXPECT_NEAR(w[3], std::exp(-20.0), 1e-15);
    EXPECT_NEAR(w[4], 1.0,             1e-12);
}

TEST(ExponentialBias, TimescaleBoundChangesDistribution) {
    // Different timescale_bound → different distribution. This is the
    // expected (and documented) behavior — timescale is a temperature knob.
    tea::ExponentialBias b1;  b1.timescale_bound = -1.0;  // default
    tea::ExponentialBias b2;  b2.timescale_bound = 10.0;  // compressed
    std::vector<int64_t> ts = {1'000'000'000LL, 999'999'990LL, 999'999'980LL};
    auto w1 = compute(b1, ts);
    auto w2 = compute(b2, ts);
    EXPECT_NE(w1, w2);
}

TEST(ExponentialBias, PartialTrunkConsistentWithFullBuild) {
    tea::ExponentialBias b;
    std::vector<int64_t> ts_desc = {10, 7, 4, 1};
    auto params = b.compute_per_vertex_params(
        tea::span<const int64_t>(ts_desc.data(), ts_desc.size()));

    std::vector<double> w_full(4, 0.0);
    b.compute_weights(tea::span<const int64_t>(ts_desc.data(), ts_desc.size()),
                      params, 0, w_full.data());

    // Partial trunk at positions [2, 4).
    std::vector<double> w_sub(2, 0.0);
    tea::span<const int64_t> sub(ts_desc.data() + 2, 2);
    b.compute_weights(sub, params, 2, w_sub.data());
    EXPECT_DOUBLE_EQ(w_sub[0], w_full[2]);
    EXPECT_DOUBLE_EQ(w_sub[1], w_full[3]);
}

TEST(ExponentialBias, EmptyAndSingleEdge) {
    tea::ExponentialBias b;
    EXPECT_EQ(compute(b, {}).size(), 0u);
    auto one = compute(b, {1234567890LL});
    ASSERT_EQ(one.size(), 1u);
    EXPECT_DOUBLE_EQ(one[0], 1.0);
}

// ============================================================================
// Compile-time prev-vertex flag
// ============================================================================
TEST(BiasTraits, NeedsPrevVertexFlags) {
    EXPECT_FALSE(tea::bias_needs_prev<tea::UniformBias>());
    EXPECT_FALSE(tea::bias_needs_prev<tea::LinearBias>());
    EXPECT_FALSE(tea::bias_needs_prev<tea::ExponentialBias>());
}
