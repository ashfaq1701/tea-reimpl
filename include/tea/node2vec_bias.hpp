// Node2VecBias — temporal node2vec per paper §2.3 (III) and §4 Algorithm 2.
//
//   P((u, v_i, t_i)) = β_(w, v_i) × δ(e_i) / Σ_{e_j ∈ Γ_t(u)} δ(e_j)
//
// where δ is the same exp((t_min_u − t) × scale) used by ExponentialBias
// (the "static-weight trick" — the t_cur cancellation makes weights static
// after preprocess), and β depends on the walker's previous vertex w:
//
//   β(w, v_i) = 1/p   if v_i == w           (returning step)
//             = 1     if v_i ∈ N(w)         (one-hop from w)
//             = 1/q   otherwise              (two-hops from w)
//
// TEA samples the proposal from the static-exp PAT/HPAT, then accepts the
// proposal with probability β(w, v) / β_max  (paper Algorithm 2 lines
// 19–22). This is plain rejection sampling on top of the proposal.
//
// β_max = max(1/p, 1, 1/q). Acceptance probability is in [0, 1].

#pragma once

#include <algorithm>
#include <cstdint>

#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/neighbor_set.hpp"
#include "tea/walk.hpp"

namespace tea {

struct Node2VecBias {
    static constexpr bool        needs_prev_vertex = true;
    static constexpr const char* name() { return "temporal_node2vec"; }

    // Node2vec parameters.
    double p = 1.0;
    double q = 1.0;

    // Inherited semantics from ExponentialBias.
    double timescale_bound = -1.0;

    // PerVertexParams is identical to ExponentialBias's — Node2Vec's static
    // weights ARE just the exp-bias weights; the β factor is applied at
    // sample time via the rejection test.
    using PerVertexParams = ExponentialBias::PerVertexParams;

    // β_max for the rejection bound.
    inline double beta_max() const noexcept {
        const double inv_p = 1.0 / p;
        const double inv_q = 1.0 / q;
        return std::max({inv_p, 1.0, inv_q});
    }

    // Forward static-weight computation to ExponentialBias.
    PerVertexParams compute_per_vertex_params(span<const int64_t> ts_desc) const noexcept {
        ExponentialBias e;
        e.timescale_bound = timescale_bound;
        return e.compute_per_vertex_params(ts_desc);
    }

    inline void compute_weights(span<const int64_t>     ts_slice,
                                const PerVertexParams&  params,
                                int32_t                 slice_start_pos,
                                double*                 out) const noexcept {
        ExponentialBias e;
        e.timescale_bound = timescale_bound;
        e.compute_weights(ts_slice, params, slice_start_pos, out);
    }

    inline void compute_weights_full(span<const int64_t> ts_desc,
                                     double*             out) const noexcept {
        ExponentialBias e;
        e.timescale_bound = timescale_bound;
        e.compute_weights_full(ts_desc, out);
    }

    // β(w, v) / β_max — the rejection test's acceptance probability for a
    // proposal landing on `candidate_v` when the walker just came from
    // `prev_u`. Returns a value in (0, 1].
    inline double accept_ratio(int32_t              prev_u,
                               int32_t              candidate_v,
                               const NeighborSets&  neighbors) const noexcept {
        const double bmax = beta_max();
        if (candidate_v == prev_u)      return (1.0 / p) / bmax;
        if (neighbors.contains(prev_u, candidate_v)) return 1.0 / bmax;
        return (1.0 / q) / bmax;
    }
};

}  // namespace tea
