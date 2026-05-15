// PAT (Persistent Alias Table) — paper §3.2.
//
// Per vertex u with degree D_u:
//   • partition u's time-DESC-sorted edge list into trunks of size T_u = √D_u
//     (DESC because walks are forward-in-time; see graph.hpp)
//   • for each trunk, build a Vose alias table
//   • store the inclusive prefix-sum of per-trunk total weights
//
// At sample time (Sampler, Phase 3.2):
//   • compute Γ_len = candidate-set length for the walker's current (u, t_prev)
//   • the first ⌊Γ_len / T_u⌋ trunks of u are fully inside Γ_t(u)
//   • there may be one "partial trunk" straddling the cutoff — its alias table
//     covers more edges than belong to Γ, so we cannot use it directly. We
//     recompute the prefix sum of just the in-Γ prefix of that trunk on the
//     fly (paper §3.2 Case ②).
//
// Storage (the proper-TEA layout — no global per-edge weights arena):
//
//   • alias_arena[E]    — global flat array of AliasEntry, layout shared
//                          with TemporalGraph's CSR (alias_arena[off(u) + i]
//                          is the alias entry for u's i-th time-desc edge).
//                          Total memory: 8 × E bytes (e.g. 2.4 GB for delicious).
//   • cumsum_arena[Σ T_u]  — global flat array of doubles; per-vertex slice is
//                          the INCLUSIVE prefix sum of per-trunk totals.
//                          Total memory: 8 × Σ ⌈D_u/√D_u⌉ ≈ 8 × Σ √D_u bytes.
//                          For coin (avg D≈31, N=700K) ≈ 33 MB.
//   • cumsum_offsets[N+1] — CSR-like offsets into cumsum_arena.
//   • trunk_sizes[N]    — per-vertex trunk size T_u.
//   • bias_params[N]    — per-vertex bias params; reused at sample time for
//                          partial-trunk weight recomputation.
//
// Per-edge weights are NOT stored persistently. They live in thread-local
// scratch during build, and on the stack during partial-trunk sampling.
// This matches paper §3.2 (the persisted state is alias tables + trunk-level
// prefix sum, nothing per edge).

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tea/alias.hpp"
#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/walk.hpp"  // for span

namespace tea {

template <typename BiasT>
class Pat {
public:
    using PerVertexParams = typename BiasT::PerVertexParams;

    Pat() = default;

    // Build the PAT for `graph` under `bias`. Parallel across vertices via
    // OpenMP. Thread-local scratch is reused so per-vertex builds do NO heap
    // allocations.
    void build(const TemporalGraph& graph,
               const BiasT&         bias,
               int32_t              min_trunk_size = kPatTrunkSizeMin) {
        const int32_t N = graph.num_vertices();
        const int64_t E = graph.num_edges();

        // 1. Pick per-vertex trunk size T_u = max(min_trunk_size, ceil(√D)).
        trunk_sizes_.assign(N, 0);
        #pragma omp parallel for schedule(static, 4096)
        for (int32_t u = 0; u < N; ++u) {
            const int32_t D = static_cast<int32_t>(graph.degree(u));
            if (D == 0) { trunk_sizes_[u] = 0; continue; }
            int32_t T = static_cast<int32_t>(
                std::ceil(std::sqrt(static_cast<double>(D))));
            if (T < min_trunk_size) T = min_trunk_size;
            if (T > D)              T = D;
            trunk_sizes_[u] = T;
        }

        // 2. cumsum_offsets[u] = number of trunks BEFORE u in cumsum_arena.
        //    Serial — N is small relative to E and this is a prefix scan.
        cumsum_offsets_.assign(N + 1, 0);
        int64_t total_num_trunks = 0;
        for (int32_t u = 0; u < N; ++u) {
            cumsum_offsets_[u] = total_num_trunks;
            const int32_t D = static_cast<int32_t>(graph.degree(u));
            if (D > 0) {
                const int32_t T = trunk_sizes_[u];
                total_num_trunks += (D + T - 1) / T;  // ceil(D/T)
            }
        }
        cumsum_offsets_[N] = total_num_trunks;

        // 3. Allocate global flat arenas.
        alias_arena_.assign(E, AliasEntry{0, 0.0f});
        cumsum_arena_.assign(total_num_trunks, 0.0);
        bias_params_.assign(N, PerVertexParams{});

        // 4. Per-vertex build, parallel. Thread-local scratch is reused.
        #pragma omp parallel
        {
            std::vector<double> weights;        // sized to max-degree-this-thread
            AliasBuildScratch   alias_scratch;  // reused across trunks

            #pragma omp for schedule(dynamic, 64)
            for (int32_t u = 0; u < N; ++u) {
                const int32_t D = static_cast<int32_t>(graph.degree(u));
                if (D == 0) continue;

                const auto ts = graph.timestamps_of(u);

                // Per-vertex bias params (e.g. t_pivot, scale for ExpBias).
                bias_params_[u] = bias.compute_per_vertex_params(ts);

                // Full-vertex weights into thread-local scratch.
                weights.resize(D);
                bias.compute_weights(ts, bias_params_[u], 0, weights.data());

                const int32_t T = trunk_sizes_[u];
                const int32_t num_trunks = (D + T - 1) / T;
                const int64_t alias_off = graph.offset_of(u);
                const int64_t cumsum_off = cumsum_offsets_[u];

                // For each trunk: stamp alias table + record trunk total.
                double running = 0.0;
                for (int32_t i = 0; i < num_trunks; ++i) {
                    const int32_t trunk_lo = i * T;
                    const int32_t trunk_hi = std::min((i + 1) * T, D);
                    const int32_t trunk_sz = trunk_hi - trunk_lo;

                    span<AliasEntry>   alias_slice(
                        alias_arena_.data() + alias_off + trunk_lo, trunk_sz);
                    span<const double> w_slice(
                        weights.data() + trunk_lo, trunk_sz);

                    build_alias_into(alias_slice, w_slice, alias_scratch);

                    double trunk_total = 0.0;
                    for (int32_t j = 0; j < trunk_sz; ++j) trunk_total += w_slice[j];
                    running += trunk_total;
                    cumsum_arena_[cumsum_off + i] = running;  // inclusive scan
                }
            }
        }
    }

    // ------------------------------------------------------------------------
    // Sample-time accessors (used by Sampler in Phase 3.2).
    // ------------------------------------------------------------------------

    int32_t trunk_size_of(int32_t u) const noexcept { return trunk_sizes_[u]; }

    int32_t num_trunks_of(int32_t u) const noexcept {
        return static_cast<int32_t>(cumsum_offsets_[u + 1] - cumsum_offsets_[u]);
    }

    // Alias view over a specific trunk of vertex u. The trunk's edges occupy
    // positions [trunk_idx * T_u, min((trunk_idx + 1) * T_u, D_u)) in u's
    // time-desc edge list.
    AliasView alias_view_of_trunk(const TemporalGraph& graph,
                                  int32_t u,
                                  int32_t trunk_idx) const noexcept {
        const int32_t T  = trunk_sizes_[u];
        const int32_t D  = static_cast<int32_t>(graph.degree(u));
        const int32_t lo = trunk_idx * T;
        const int32_t hi = std::min((trunk_idx + 1) * T, D);
        return AliasView(span<const AliasEntry>(
            alias_arena_.data() + graph.offset_of(u) + lo,
            static_cast<std::size_t>(hi - lo)));
    }

    // Inclusive prefix-sum over the per-trunk totals of vertex u.
    // cumsums[i] = sum of trunk totals for trunks 0..i (inclusive).
    span<const double> trunk_cumsums_of(int32_t u) const noexcept {
        return span<const double>(
            cumsum_arena_.data() + cumsum_offsets_[u],
            static_cast<std::size_t>(cumsum_offsets_[u + 1]
                                     - cumsum_offsets_[u]));
    }

    // Per-vertex bias params, used by Sampler for partial-trunk weight
    // recomputation (so the on-the-fly prefix sum matches what was used
    // at build time).
    const PerVertexParams& bias_params_of(int32_t u) const noexcept {
        return bias_params_[u];
    }

    // ------------------------------------------------------------------------
    // Memory accounting (for run banners + paper memory tables).
    // ------------------------------------------------------------------------

    int64_t alias_entry_count() const noexcept {
        return static_cast<int64_t>(alias_arena_.size());
    }
    int64_t cumsum_entry_count() const noexcept {
        return static_cast<int64_t>(cumsum_arena_.size());
    }
    int64_t memory_bytes() const noexcept {
        return static_cast<int64_t>(alias_arena_.size()) * sizeof(AliasEntry)
             + static_cast<int64_t>(cumsum_arena_.size()) * sizeof(double)
             + static_cast<int64_t>(cumsum_offsets_.size()) * sizeof(int64_t)
             + static_cast<int64_t>(trunk_sizes_.size())   * sizeof(int32_t)
             + static_cast<int64_t>(bias_params_.size())   * sizeof(PerVertexParams);
    }

private:
    std::vector<AliasEntry>      alias_arena_;
    std::vector<double>          cumsum_arena_;
    std::vector<int64_t>         cumsum_offsets_;
    std::vector<int32_t>         trunk_sizes_;
    std::vector<PerVertexParams> bias_params_;
};

}  // namespace tea
