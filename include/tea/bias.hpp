// Bias: per-edge static weight computation.
//
// Three concrete biases in this header (Uniform, Linear, Exponential);
// Node2Vec lives in node2vec_bias.hpp (Phase 5) because it needs prev-vertex
// state and per-vertex neighbor sets.
//
// === Forward-walk bias direction (matches Tempest's forward semantics) ===
//
// Walks go forward (Γ_t(u) = {t_i > t_prev}, out-edges, t strictly
// increasing along the path).  Under both biases the SMALLEST t in the
// candidate set wins — i.e. the closest forward jump from t_prev gets
// the highest probability.  This mirrors Tempest's `exp(max_ts − t)`
// forward weight (node_edge_index.cu) and its
// `pick_random_linear(prioritize_end=false)` forward picker.  The two
// engines now sample from the same distribution.
//
// In time-DESC storage (position 0 = newest, position D−1 = oldest), the
// candidate set is the prefix [0, L); within that prefix, the END (highest
// position = oldest) carries the highest weight.
//
// • Linear: δ(e_i) = rank(e_i) = position_in_desc_list + 1.  Position 0
//   (newest) → rank 1, position D−1 (oldest) → rank D.  Rank is invariant
//   under affine time transforms, so behaviour matches across datasets
//   that differ only in time-unit conventions.
//
// • Exponential: δ(e_i) = exp((t_min_u − t_i) × scale).  At t = t_min_u
//   (oldest edge of u) the weight is exp(0) = 1; at t = t_max_u (newest)
//   it underflows toward exp(−span × scale).  Pivoting at t_min instead
//   of t_max keeps all exponents ≤ 0 → no overflow on unix-timescale
//   data.  Softmax is shift-invariant, so this is the same distribution
//   as Tempest's `exp((max_ts − t) × scale)` once normalised.
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
//   • PerVertexParams compute_per_vertex_params(span<const int64_t> ts_desc) const
//   • void compute_weights(span<const int64_t> ts_slice,
//                          const PerVertexParams& params,
//                          int32_t slice_start_pos,
//                          double* out) const
//   • void compute_weights_full(span<const int64_t> ts_desc, double* out) const   // convenience
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
        span<const int64_t> /*ts_desc*/) const noexcept { return {}; }

    inline void compute_weights(span<const int64_t> ts_slice,
                                const PerVertexParams& /*params*/,
                                int32_t /*slice_start_pos*/,
                                double* out) const noexcept {
        const std::size_t d = ts_slice.size();
        for (std::size_t i = 0; i < d; ++i) out[i] = 1.0;
    }

    inline void compute_weights_full(span<const int64_t> ts_desc,
                                     double* out) const noexcept {
        compute_weights(ts_desc, {}, 0, out);
    }
};

// ============================================================================
// LinearBias: δ(e) = rank(e), oldest = D, newest = 1.
//
// In ts_desc (DESCENDING time order), the edge at position p has
// rank = p + 1.  The slice_start_pos argument lets PAT/HPAT compute the
// rank for a sub-range without seeing the full edge list at sample time.
// Forward walks pick from the prefix [0, L) of the DESC list; the END of
// that prefix (oldest still-alive edge = closest forward jump) carries
// rank L (the highest weight in the prefix).
// ============================================================================
struct LinearBias {
    static constexpr bool        needs_prev_vertex = false;
    static constexpr const char* name() { return "linear"; }

    struct PerVertexParams {
        int32_t degree;  // kept for symmetry with the other biases; not used.
    };

    PerVertexParams compute_per_vertex_params(
        span<const int64_t> ts_desc) const noexcept {
        return PerVertexParams{static_cast<int32_t>(ts_desc.size())};
    }

    inline void compute_weights(span<const int64_t> ts_slice,
                                const PerVertexParams& /*params*/,
                                int32_t slice_start_pos,
                                double* out) const noexcept {
        const std::size_t d = ts_slice.size();
        for (std::size_t i = 0; i < d; ++i) {
            // rank in original time-desc list = position + 1
            out[i] = static_cast<double>(
                slice_start_pos + static_cast<int32_t>(i) + 1);
        }
    }

    inline void compute_weights_full(span<const int64_t> ts_desc,
                                     double* out) const noexcept {
        const auto p = compute_per_vertex_params(ts_desc);
        compute_weights(ts_desc, p, 0, out);
    }
};

// ============================================================================
// ExponentialBias: δ(e) = exp((t_min_u − t_i) × scale).
//
// Per-vertex pivot is t_min_u (oldest edge of u).  Largest weight lands at
// t = t_min_u (exp(0) = 1); newer edges underflow toward exp(−span×scale).
// Pivoting at t_min instead of t_max keeps every exponent ≤ 0 so the
// formula is numerically safe on raw unix timestamps regardless of scale.
// Softmax is shift-invariant, so the resulting distribution is identical
// to Tempest's `exp((max_ts − t) × scale)` once normalised — picking
// t_min as pivot is purely a numerical-stability choice.
//
// scale = 1.0       (default, timescale_bound ≤ 0): proper TEA, the
//                   softmax of exp(−t_i) on the raw timestamps with no
//                   temperature change.  Older-edge underflow is correct
//                   behaviour on unix-timescale data, not a bug.
//
// scale = timescale_bound / (t_max_u − t_min_u)  (timescale_bound > 0):
//                   Tempest-compatible mode.  Compresses the per-vertex
//                   time range to a fixed span before exp(); changes the
//                   distribution's temperature.  Use for cross-comparison
//                   runs where both engines need the same temperature.
// ============================================================================
struct ExponentialBias {
    static constexpr bool        needs_prev_vertex = false;
    static constexpr const char* name() { return "exponential"; }

    // timescale_bound ≤ 0 → proper-TEA mode (no temperature change).
    // timescale_bound > 0 → Tempest-compatible rescale.
    double timescale_bound = -1.0;

    struct PerVertexParams {
        double t_pivot;  // = t_min_u (oldest timestamp of vertex u)
        double scale;    // = 1.0 (proper TEA) or timescale_bound / span (compat)
    };

    PerVertexParams compute_per_vertex_params(
        span<const int64_t> ts_desc) const noexcept {
        if (ts_desc.empty()) return PerVertexParams{0.0, 0.0};
        const double t_max = static_cast<double>(ts_desc[0]);
        const double t_min = static_cast<double>(ts_desc[ts_desc.size() - 1]);
        const double span  = t_max - t_min;
        if (span <= 0.0) {
            // All same timestamp → uniform (scale=0 makes every weight exp(0)=1)
            return PerVertexParams{t_min, 0.0};
        }
        const double scale = (timescale_bound > 0.0)
            ? timescale_bound / span
            : 1.0;
        return PerVertexParams{t_min, scale};
    }

    inline void compute_weights(span<const int64_t> ts_slice,
                                const PerVertexParams& params,
                                int32_t /*slice_start_pos*/,
                                double* out) const noexcept {
        const std::size_t d = ts_slice.size();
        for (std::size_t i = 0; i < d; ++i) {
            const double t = static_cast<double>(ts_slice[i]);
            out[i] = std::exp((params.t_pivot - t) * params.scale);
        }
    }

    inline void compute_weights_full(span<const int64_t> ts_desc,
                                     double* out) const noexcept {
        const auto p = compute_per_vertex_params(ts_desc);
        compute_weights(ts_desc, p, 0, out);
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
