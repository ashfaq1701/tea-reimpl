// Bias: per-edge static weight computation.
//
// Three concrete biases in this header (Uniform, Linear, Exponential);
// Node2Vec lives in node2vec_bias.hpp (Phase 5) because it needs prev-vertex
// state and per-vertex neighbor sets.
//
// Direction: walks are BACKWARD-IN-TIME (graph.hpp).  Per-vertex edge
// lists are stored ASCENDING by timestamp.  Every bias here preserves the
// invariant "largest t in the candidate set has the largest weight" — for
// backward walks that means the most-recent past edge (closest to t_prev
// going backward) is favoured, which matches Tempest's backward
// convention and the natural "recent context" semantic for representation
// learning.
//
// === Proper-TEA correctness notes ===
//
// • Linear (paper §2.3 I): δ(e) = rank(e), where rank is 1..D with the
//   newest edge getting rank D and the oldest rank 1. The paper allows
//   either rank or t_i directly; we pick rank because it is invariant
//   under affine time transforms (so the picker behaves the same across
//   datasets that differ only in time-unit conventions, e.g. unix-seconds
//   vs nanoseconds). Both choices are within the paper's spec.  On the
//   ASCENDING-time storage layout, "position p" corresponds to rank
//   p + 1 (oldest = position 0 = rank 1; newest = position D-1 = rank D).
//
// • Exponential (paper §2.3 II): δ(e) = exp(t_i).  The normalization
//   makes any additive shift a no-op (softmax is shift-invariant), so we
//   use exp(t_i − t_max_u) which is numerically equivalent and never
//   overflows (largest weight = exp(0) = 1 at the newest edge). Older
//   edges with very large negative exponents underflow to 0 — that is
//   the *correct* behavior under pure exponential bias on unix-timescale
//   data, NOT a bug.
//
//   Scaling is NOT shift-invariant. My earlier "rescale to [0, 80]"
//   approach silently distorted the softmax temperature. We do not
//   rescale by default. The `timescale_bound` knob (paper API parity
//   with Tempest's same-named flag) re-enables a controlled rescale for
//   the cross-comparison runs; it changes the distribution and that is
//   documented explicitly.
//
// === API ===
//
// Every bias exposes:
//   • static constexpr bool        needs_prev_vertex
//   • static constexpr const char* name()
//   • PerVertexParams compute_per_vertex_params(span<const int64_t> ts_asc) const
//   • void compute_weights(span<const int64_t> ts_slice,
//                          const PerVertexParams& params,
//                          int32_t slice_start_pos,
//                          double* out) const
//   • void compute_weights_full(span<const int64_t> ts_asc, double* out) const   // convenience
//
// Why the (params, slice_start_pos) form: PAT/HPAT must recompute weights
// at sample time for the *partial trunk* that straddles the candidate-set
// cutoff (paper §3.2 Case ②). At sample time we know the per-vertex params
// (cached from build) and the slice's position in the full edge list;
// `compute_weights(...)` produces exactly the same weights as were used to
// build the alias tables, so the on-the-fly prefix-sum is consistent with
// the persisted cumsum.

#pragma once

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "tea/walk.hpp"  // for tea::span

namespace tea {

// ============================================================================
// UniformBias: δ(e) = 1.
// ============================================================================
struct UniformBias {
    static constexpr bool        needs_prev_vertex = false;
    static constexpr const char* name() { return "uniform"; }

    struct PerVertexParams { /* empty */ };

    PerVertexParams compute_per_vertex_params(
        span<const int64_t> /*ts_asc*/) const noexcept { return {}; }

    inline void compute_weights(span<const int64_t> ts_slice,
                                const PerVertexParams& /*params*/,
                                int32_t /*slice_start_pos*/,
                                double* out) const noexcept {
        const std::size_t d = ts_slice.size();
        for (std::size_t i = 0; i < d; ++i) out[i] = 1.0;
    }

    inline void compute_weights_full(span<const int64_t> ts_asc,
                                     double* out) const noexcept {
        compute_weights(ts_asc, {}, 0, out);
    }
};

// ============================================================================
// LinearBias: δ(e) = rank(e), newest = D, oldest = 1.
//
// In ts_asc (ASCENDING time order), the edge at position p has
// rank = p + 1 (oldest = position 0 = rank 1; newest = position D-1 = rank D).
// The slice_start_pos argument lets PAT/HPAT compute the rank for a
// sub-range without seeing the full edge list at sample time.
// ============================================================================
struct LinearBias {
    static constexpr bool        needs_prev_vertex = false;
    static constexpr const char* name() { return "linear"; }

    struct PerVertexParams {
        // No degree-dependent term needed for ascending-list rank.
    };

    PerVertexParams compute_per_vertex_params(
        span<const int64_t> /*ts_asc*/) const noexcept {
        return PerVertexParams{};
    }

    inline void compute_weights(span<const int64_t> ts_slice,
                                const PerVertexParams& /*params*/,
                                int32_t slice_start_pos,
                                double* out) const noexcept {
        const std::size_t d = ts_slice.size();
        for (std::size_t i = 0; i < d; ++i) {
            // rank in original time-asc list = position + 1
            out[i] = static_cast<double>(
                slice_start_pos + static_cast<int32_t>(i) + 1);
        }
    }

    inline void compute_weights_full(span<const int64_t> ts_asc,
                                     double* out) const noexcept {
        const auto p = compute_per_vertex_params(ts_asc);
        compute_weights(ts_asc, p, 0, out);
    }
};

// ============================================================================
// ExponentialBias: δ(e) = exp((t_i − t_max_u) × scale).
//
// scale = 1.0       (default, timescale_bound ≤ 0): proper TEA, the softmax
//                   distribution of exp(t_i) without any temperature change.
//                   On unix-timescale data, older edges underflow to 0; that
//                   is correct behavior, not a bug.
//
// scale = timescale_bound / (t_max_u − t_min_u)  (timescale_bound > 0):
//                   Tempest-compatible mode. Compresses the per-vertex time
//                   range to a fixed span before exp(); changes the
//                   distribution's temperature. Use for the Tempest cross-
//                   comparison runs where both engines need the same temp.
// ============================================================================
struct ExponentialBias {
    static constexpr bool        needs_prev_vertex = false;
    static constexpr const char* name() { return "exponential"; }

    // timescale_bound ≤ 0 → proper-TEA mode (no temperature change).
    // timescale_bound > 0 → Tempest-compatible rescale.
    double timescale_bound = -1.0;

    struct PerVertexParams {
        double t_pivot;  // = t_max_u
        double scale;    // = 1.0 (proper TEA) or timescale_bound / span (compat)
    };

    PerVertexParams compute_per_vertex_params(
        span<const int64_t> ts_asc) const noexcept {
        if (ts_asc.empty()) return PerVertexParams{0.0, 0.0};
        // ts is ASCENDING → smallest at [0], largest at [size-1].
        const double t_min = static_cast<double>(ts_asc[0]);
        const double t_max = static_cast<double>(ts_asc[ts_asc.size() - 1]);
        const double span  = t_max - t_min;
        if (span <= 0.0) {
            // All same timestamp → uniform (scale=0 makes every weight exp(0)=1)
            return PerVertexParams{t_max, 0.0};
        }
        const double scale = (timescale_bound > 0.0)
            ? timescale_bound / span
            : 1.0;
        return PerVertexParams{t_max, scale};
    }

    inline void compute_weights(span<const int64_t> ts_slice,
                                const PerVertexParams& params,
                                int32_t /*slice_start_pos*/,
                                double* out) const noexcept {
        const std::size_t d = ts_slice.size();
        for (std::size_t i = 0; i < d; ++i) {
            const double t = static_cast<double>(ts_slice[i]);
            out[i] = std::exp((t - params.t_pivot) * params.scale);
        }
    }

    inline void compute_weights_full(span<const int64_t> ts_asc,
                                     double* out) const noexcept {
        const auto p = compute_per_vertex_params(ts_asc);
        compute_weights(ts_asc, p, 0, out);
    }
};

// ============================================================================
// Compile-time helper for the sampler's "do I need to track prev vertex" check.
// Used by Phase 5 Node2Vec's templated rejection loop.
// ============================================================================
template <typename BiasT>
constexpr bool bias_needs_prev() {
    return BiasT::needs_prev_vertex;
}

}  // namespace tea
