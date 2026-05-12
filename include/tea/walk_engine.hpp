// WalkEngine — orchestrates parallel walk generation.
//
// Templated free functions, one per (sampler-variant × bias-class) pair:
//   run_walks_pat              — PAT,  non-Node2Vec biases (Uniform/Linear/Exp)
//   run_walks_hpat             — HPAT, non-Node2Vec biases
//   run_walks_pat_node2vec     — PAT  + Node2VecBias (with rejection)
//   run_walks_hpat_node2vec    — HPAT + Node2VecBias (with rejection)
//
// All four delegate to a single `detail::run_walks_impl` template that
// encapsulates the per-walk loop. The actual sample() call site differs
// only in the captured closure; the compiler inlines through it.
//
// Design contract (CLAUDE.md):
//   • Zero heap allocations during the walk loop. Caller owns `walks_out`
//     (`num_walks × max_walk_len` NodeSteps) and `walk_lens_out` (`num_walks`).
//   • OpenMP `parallel for schedule(dynamic, 64)` over walks for load balance.
//   • Per-walk RNG seeded from (global_seed, walk_idx) → deterministic
//     regardless of OMP thread distribution. Negligible re-init cost.
//   • Per-thread SamplerScratch reused across walks → no per-walk alloc.
//
// Returns WalkRunStats with timing + step counts for the run banner.

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/edge.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/neighbor_set.hpp"
#include "tea/node2vec_bias.hpp"
#include "tea/pat.hpp"
#include "tea/rng.hpp"
#include "tea/sampler.hpp"

namespace tea {

struct WalkRunStats {
    double  elapsed_sec   = 0.0;
    int64_t num_walks     = 0;
    int64_t total_steps   = 0;  // Σ walk_lens (includes slot 0 = start vertex)
    int64_t dead_at_start = 0;  // walks with len == 1 (no successful hop)

    double walks_per_sec() const noexcept {
        return elapsed_sec > 0 ? static_cast<double>(num_walks) / elapsed_sec : 0.0;
    }
    double steps_per_sec() const noexcept {
        return elapsed_sec > 0 ? static_cast<double>(total_steps) / elapsed_sec : 0.0;
    }
    double avg_walk_len() const noexcept {
        return num_walks > 0
            ? static_cast<double>(total_steps) / static_cast<double>(num_walks)
            : 0.0;
    }
};

namespace detail {

// Core walk loop. SampleFn signature:
//   SampledStep sample(int32_t u, int64_t t_prev, int32_t prev_u,
//                      Pcg64& rng, SamplerScratch& scratch);
//
// Backward walks always seed t_prev with kSentinelStartTimestamp
// (INT64_MAX) so every inbound edge of the start vertex is a first-hop
// candidate (t < INT64_MAX is trivially true for every real timestamp).
template <typename SampleFn>
inline WalkRunStats run_walks_impl(
        const int32_t*  start_vertices,
        int32_t         num_walks,
        int32_t         max_walk_len,
        uint64_t        global_seed,
        NodeStep*       walks_out,
        int32_t*        walk_lens_out,
        SampleFn&&      sample) {
    const auto t_start = std::chrono::steady_clock::now();

    int64_t total_steps_local = 0;

    #pragma omp parallel reduction(+:total_steps_local)
    {
        SamplerScratch scratch;  // per-thread, reused across walks

        #pragma omp for schedule(dynamic, 64)
        for (int32_t i = 0; i < num_walks; ++i) {
            const int32_t u_start = start_vertices[i];
            NodeStep* slots = walks_out + static_cast<int64_t>(i) * max_walk_len;

            // Slot 0 = (start vertex, sentinel timestamp). Always written.
            slots[0] = NodeStep{u_start, kSentinelStartTimestamp};

            // Per-walk Pcg64 seeded from (global_seed, walk_idx). Deterministic
            // regardless of how walks are scheduled across threads.
            Pcg64 rng(global_seed, static_cast<uint64_t>(i));

            int32_t u       = u_start;
            int64_t t_prev  = kSentinelStartTimestamp;
            int32_t prev_u  = -1;
            int32_t walk_len = 1;

            for (int32_t step = 1; step < max_walk_len; ++step) {
                const SampledStep s = sample(u, t_prev, prev_u, rng, scratch);
                if (s.v == kWalkDeadSentinel) break;
                slots[step] = NodeStep{s.v, s.t};
                prev_u  = u;
                u       = s.v;
                t_prev  = s.t;
                ++walk_len;
            }

            // Pad dead tail (caller may rely on sentinel-fill for inspection).
            for (int32_t step = walk_len; step < max_walk_len; ++step) {
                slots[step] = NodeStep{kWalkDeadSentinel, 0};
            }

            walk_lens_out[i]  = walk_len;
            total_steps_local += walk_len;
        }
    }

    const auto t_end = std::chrono::steady_clock::now();

    WalkRunStats stats;
    stats.elapsed_sec = std::chrono::duration<double>(t_end - t_start).count();
    stats.num_walks   = num_walks;
    stats.total_steps = total_steps_local;
    for (int32_t i = 0; i < num_walks; ++i) {
        if (walk_lens_out[i] <= 1) ++stats.dead_at_start;
    }
    return stats;
}

}  // namespace detail

// ----------------------------------------------------------------------------
// Non-Node2Vec variants (UniformBias, LinearBias, ExponentialBias)
// ----------------------------------------------------------------------------

template <typename BiasT>
WalkRunStats run_walks_pat(
        const TemporalGraph& graph,
        const Pat<BiasT>&    pat,
        const BiasT&         bias,
        const int32_t*       start_vertices,
        int32_t              num_walks,
        int32_t              max_walk_len,
        uint64_t             global_seed,
        NodeStep*            walks_out,
        int32_t*             walk_lens_out) {
    return detail::run_walks_impl(
        start_vertices, num_walks, max_walk_len, global_seed,
        walks_out, walk_lens_out,
        [&](int32_t u, int64_t t_prev, int32_t /*prev_u*/,
            Pcg64& rng, SamplerScratch& scratch) {
            return sample_pat(graph, pat, bias, u, t_prev, rng, scratch);
        });
}

template <typename BiasT>
WalkRunStats run_walks_hpat(
        const TemporalGraph& graph,
        const Hpat<BiasT>&   hpat,
        const BiasT&         bias,
        const int32_t*       start_vertices,
        int32_t              num_walks,
        int32_t              max_walk_len,
        uint64_t             global_seed,
        NodeStep*            walks_out,
        int32_t*             walk_lens_out) {
    return detail::run_walks_impl(
        start_vertices, num_walks, max_walk_len, global_seed,
        walks_out, walk_lens_out,
        [&](int32_t u, int64_t t_prev, int32_t /*prev_u*/,
            Pcg64& rng, SamplerScratch& scratch) {
            return sample_hpat(graph, hpat, bias, u, t_prev, rng, scratch);
        });
}

// ----------------------------------------------------------------------------
// Node2Vec variants (rejection loop layered on PAT/HPAT proposals)
// ----------------------------------------------------------------------------

inline WalkRunStats run_walks_pat_node2vec(
        const TemporalGraph&       graph,
        const Pat<Node2VecBias>&   pat,
        const Node2VecBias&        bias,
        const NeighborSets&        neighbors,
        const int32_t*             start_vertices,
        int32_t                    num_walks,
        int32_t                    max_walk_len,
        uint64_t                   global_seed,
        NodeStep*                  walks_out,
        int32_t*                   walk_lens_out) {
    return detail::run_walks_impl(
        start_vertices, num_walks, max_walk_len, global_seed,
        walks_out, walk_lens_out,
        [&](int32_t u, int64_t t_prev, int32_t prev_u,
            Pcg64& rng, SamplerScratch& scratch) {
            return sample_pat_node2vec(graph, pat, bias, neighbors,
                                        u, t_prev, prev_u, rng, scratch);
        });
}

inline WalkRunStats run_walks_hpat_node2vec(
        const TemporalGraph&       graph,
        const Hpat<Node2VecBias>&  hpat,
        const Node2VecBias&        bias,
        const NeighborSets&        neighbors,
        const int32_t*             start_vertices,
        int32_t                    num_walks,
        int32_t                    max_walk_len,
        uint64_t                   global_seed,
        NodeStep*                  walks_out,
        int32_t*                   walk_lens_out) {
    return detail::run_walks_impl(
        start_vertices, num_walks, max_walk_len, global_seed,
        walks_out, walk_lens_out,
        [&](int32_t u, int64_t t_prev, int32_t prev_u,
            Pcg64& rng, SamplerScratch& scratch) {
            return sample_hpat_node2vec(graph, hpat, bias, neighbors,
                                         u, t_prev, prev_u, rng, scratch);
        });
}

// ----------------------------------------------------------------------------
// Helper: build the default "all-nodes, walks_per_node × N" start-vertex list.
// Each node u gets `walks_per_node` walks starting from it.
// ----------------------------------------------------------------------------

inline std::vector<int32_t> make_all_nodes_starts(
        const TemporalGraph& graph,
        int32_t              walks_per_node) {
    const int32_t N = graph.num_vertices();
    // Only include vertices that have a non-empty inbound adjacency
    // (otherwise the backward walk dies immediately and the slot is wasted).
    std::vector<int32_t> starts;
    starts.reserve(static_cast<std::size_t>(N) * walks_per_node);
    for (int32_t u = 0; u < N; ++u) {
        if (graph.degree(u) > 0) {
            for (int32_t k = 0; k < walks_per_node; ++k) starts.push_back(u);
        }
    }
    return starts;
}

}  // namespace tea
