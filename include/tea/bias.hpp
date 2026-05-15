// Bias: per-edge static weight computation.
//
// Three concrete biases in this header (Uniform, Linear, Exponential);
// Node2Vec lives in node2vec_bias.hpp (Phase 5) because it needs prev-vertex
// state and per-vertex neighbor sets.
//
// Direction: walks are FORWARD-IN-TIME (graph.hpp).  Per-vertex edge
// lists are stored DESCENDING by timestamp.
//
// Bias semantics in this branch are FORWARD-FRIENDLY, matching Tempest's
// forward picker convention: within the candidate set Γ_{t_prev}(u) =
// {t > t_prev}, the *most-immediate-next* edge (smallest admissible t —
// closest after t_prev) gets the largest weight, and the furthest-future
// edge gets the smallest.  This is the practical recency-continuity bias
// every representation-learning paper actually wants — and the inverse of
// the TEA paper's literal §2.3 formula (which favours the furthest-future
// jump and produces walks that teleport to the latest timestamp on every
// step).  Tempest's authors silently re-derived the formula, and we
// follow them.
//
// === Correctness notes ===
//
// • Linear: δ(e) = position + 1 on the DESCENDING-time storage layout.
//   Position 0 (newest in vertex history) gets weight 1; position D-1
//   (oldest in vertex history) gets weight D.  Inside the forward
//   candidate prefix [0, L), position L-1 (smallest admissible t = most-
//   immediate-next) carries the largest weight L, and position 0 (largest
//   admissible t = furthest future) carries weight 1.  Same per-edge
//   distribution as Tempest's `pick_random_linear(0, L, prioritize_end=
//   false, r)` for unique-timestamp graphs.
//
// • Exponential: δ(e) = exp((t_min_u − t) · scale).  At t = t_min_u
//   (oldest in vertex history, also the smallest-admissible-t in any
//   forward candidate that includes it), weight = exp(0) = 1.  At t =
//   t_max_u, weight = exp(-(t_max_u − t_min_u) · scale) → underflows to
//   0 on long-span unix-timestamp data, which is the correct paper-style
//   limit of pure exponential bias.  Pivoting on t_min keeps the exponent
//   ≤ 0 so the alias-table build never overflows.  Equivalent per-edge
//   distribution to Tempest's per-group weight `group_size · exp((t_min −
//   t_g) · scale)` after normalisation.
//
//   Scaling is NOT shift-invariant. The `timescale_bound` knob (paper API
//   parity with Tempest's same-named flag) re-enables a controlled rescale
//   for the cross-comparison runs; it changes the distribution and that is
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
// LinearBias: δ(e) = position + 1 on the DESCENDING-time storage.
//
// In ts_desc (DESCENDING time order), position 0 holds the newest edge
// and position D-1 holds the oldest.  This bias assigns weight = p + 1
// at position p, so the OLDEST edge gets weight D and the NEWEST gets
// weight 1.  Within the forward candidate prefix [0, L), the smallest
// admissible t (at position L-1 — the most-immediate-next edge) ends up
// with the largest weight L.
//
// PerVertexParams is empty: weight is a function of position alone, no
// per-vertex preprocessing needed.  The slice_start_pos argument lets
// PAT/HPAT's partial-trunk recompute reconstruct absolute position from
// a sub-range.
// ============================================================================
struct LinearBias {
    static constexpr bool        needs_prev_vertex = false;
    static constexpr const char* name() { return "linear"; }

    struct PerVertexParams { /* empty */ };

    PerVertexParams compute_per_vertex_params(
        span<const int64_t> /*ts_desc*/) const noexcept { return {}; }

    inline void compute_weights(span<const int64_t> ts_slice,
                                const PerVertexParams& /*params*/,
                                int32_t slice_start_pos,
                                double* out) const noexcept {
        const std::size_t d = ts_slice.size();
        for (std::size_t i = 0; i < d; ++i) {
            // weight at desc position p = p + 1
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
// Pivots on t_min_u so the oldest edge in the vertex's history (and the
// most-immediate-next admissible edge in any forward candidate prefix)
// lands at exp(0) = 1.  The exponent is always ≤ 0, so weights live in
// (0, 1] regardless of timescale_bound, and the alias-table build never
// overflows.  Far-future edges underflow to 0 on long-span unix-timestamp
// data — the correct paper-style limit of pure exp on that scale.
//
// scale = 1.0       (default, timescale_bound ≤ 0): proper TEA softmax of
//                   exp(t_i) without temperature change, in the Tempest
//                   forward sign convention.
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
        double t_pivot;  // = t_min_u (smallest ts seen at this vertex)
        double scale;    // = 1.0 (proper TEA) or timescale_bound / span (compat)
    };

    PerVertexParams compute_per_vertex_params(
        span<const int64_t> ts_desc) const noexcept {
        if (ts_desc.empty()) return PerVertexParams{0.0, 0.0};
        // ts is DESCENDING → largest at [0], smallest at [size-1].
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
