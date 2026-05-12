// HPAT (Hierarchical Persistent Alias Table) — paper §3.3.
//
// Per vertex u with degree D_u:
//   • K_u = ⌊log₂(D_u)⌋  (= 0 for D_u=1; we treat D_u=0 specially)
//   • At each level k ∈ [0, K_u]:
//       – trunks at this level have size 2^k
//       – number of trunks at level k = ⌊D_u / 2^k⌋
//       – trunk (k, i) covers edges [i·2^k, (i+1)·2^k) of u's time-desc list
//   • For each trunk, store a Vose alias table + the trunk's total weight.
//
// At sample time (Phase 4.3, sample_hpat in sampler.hpp):
//   • Γ_len = candidate_set_len(u, t_prev)
//   • Binary-decompose Γ_len → cover of trunks (≤ K_u + 1 trunks), one per
//     set bit of Γ_len. The trunks exactly partition [0, Γ_len).
//   • Build a small ITS cumsum over cover trunk-totals (≤ 30 entries).
//   • Pick a trunk via ITS; alias-sample inside it.
//   • NO PARTIAL-TRUNK CASE for HPAT — the binary decomposition always
//     covers any Γ_len ∈ [0, D_u] exactly, by design.
//
// Storage (proper-TEA, no global per-edge weights):
//   • alias_arena_[Σ_u Σ_k (⌊D_u/2^k⌋ × 2^k)]  ≈ 8 × E × avg_log_D bytes
//   • trunk_totals_[Σ_u Σ_k ⌊D_u/2^k⌋]          ≤ 16 × E bytes (geometric sum)
//   • alias_offsets_[N+1], totals_offsets_[N+1], K_max_[N]
//   • bias_params_[N]
//
// Within u's alias slice, level k starts at offset:
//   Σ_{k'<k} (⌊D_u / 2^{k'}⌋ × 2^{k'})
// computed on the fly (O(K) ≈ 30 ops) — saves O(N · K) bytes of cached
// per-level offset storage. See CLAUDE.md design contract for why.

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
#include "tea/aux_index.hpp"
#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/trunk_ref.hpp"
#include "tea/walk.hpp"

namespace tea {

// Compute Σ_{k'<k} (⌊D/2^{k'}⌋ × 2^{k'}) — the alias-arena offset for u's level k.
// O(k) inline.
inline int64_t hpat_alias_level_offset(int32_t D, int32_t k) noexcept {
    int64_t off = 0;
    for (int32_t kk = 0; kk < k; ++kk) {
        const int32_t trunks_at_kk  = D >> kk;
        const int64_t entries_at_kk = static_cast<int64_t>(trunks_at_kk) << kk;
        off += entries_at_kk;
    }
    return off;
}

// Compute Σ_{k'<k} ⌊D/2^{k'}⌋ — the totals-arena offset for u's level k.
inline int64_t hpat_totals_level_offset(int32_t D, int32_t k) noexcept {
    int64_t off = 0;
    for (int32_t kk = 0; kk < k; ++kk) {
        off += static_cast<int64_t>(D >> kk);
    }
    return off;
}

// Total alias entries per vertex with degree D.
inline int64_t hpat_alias_entries_per_vertex(int32_t D) noexcept {
    if (D <= 0) return 0;
    const int32_t K = static_cast<int32_t>(std::floor(std::log2(D)));
    int64_t total = 0;
    for (int32_t k = 0; k <= K; ++k) {
        const int32_t trunks_at_k  = D >> k;
        total += static_cast<int64_t>(trunks_at_k) << k;
    }
    return total;
}

// Total trunks per vertex with degree D.
inline int64_t hpat_trunks_per_vertex(int32_t D) noexcept {
    if (D <= 0) return 0;
    const int32_t K = static_cast<int32_t>(std::floor(std::log2(D)));
    int64_t total = 0;
    for (int32_t k = 0; k <= K; ++k) {
        total += static_cast<int64_t>(D >> k);
    }
    return total;
}

template <typename BiasT>
class Hpat {
public:
    using PerVertexParams = typename BiasT::PerVertexParams;

    Hpat() = default;

    // Build HPAT.  Also builds the precomputed AuxiliaryIndex (paper §3.4)
    // by default; pass aux_budget_bytes = 0 to skip it (sampler then falls
    // back to on-the-fly binary decomposition).  If the projected aux memory
    // exceeds aux_budget_bytes, the aux index build is silently skipped and
    // the sampler also falls back — caller can inspect aux().active().
    void build(const TemporalGraph& graph, const BiasT& bias,
               std::size_t aux_budget_bytes = kAuxIndexMaxBytes) {
        const int32_t N = graph.num_vertices();

        // 1. Per-vertex layout decision (paper §3.3 second ad-hoc opt):
        //    D=0            → empty (K_max=-1, is_solo=0)
        //    0<D≤thresh     → SOLO: single AliasTable of size D (K_max=-1, is_solo=1)
        //    D>thresh       → HIERARCHY: K+1 levels of 2^k trunks (K_max=⌊log₂D⌋, is_solo=0)
        K_max_.assign(N, -1);
        is_solo_.assign(N, 0);
        alias_offsets_.assign(N + 1, 0);
        totals_offsets_.assign(N + 1, 0);
        bias_params_.assign(N, PerVertexParams{});

        int64_t alias_arena_size  = 0;
        int64_t totals_arena_size = 0;
        for (int32_t u = 0; u < N; ++u) {
            alias_offsets_[u]  = alias_arena_size;
            totals_offsets_[u] = totals_arena_size;
            const int32_t D = static_cast<int32_t>(graph.degree(u));
            if (D == 0) continue;
            if (D <= kHpatDegreeThreshold) {
                is_solo_[u]       = 1;
                alias_arena_size += D;  // single AliasTable of size D
                // no trunk_totals_ entries for solo vertices
            } else {
                K_max_[u]          = static_cast<int32_t>(std::floor(std::log2(D)));
                alias_arena_size  += hpat_alias_entries_per_vertex(D);
                totals_arena_size += hpat_trunks_per_vertex(D);
            }
        }
        alias_offsets_[N]  = alias_arena_size;
        totals_offsets_[N] = totals_arena_size;

        // 2. Allocate arenas.
        alias_arena_.assign(alias_arena_size,  AliasEntry{0, 0.0f});
        trunk_totals_.assign(totals_arena_size, 0.0);

        // 3. Per-vertex build, parallel. Thread-local scratch reused.
        #pragma omp parallel
        {
            std::vector<double> weights;
            AliasBuildScratch   alias_scratch;

            #pragma omp for schedule(dynamic, 64)
            for (int32_t u = 0; u < N; ++u) {
                const int32_t D = static_cast<int32_t>(graph.degree(u));
                if (D == 0) continue;

                const auto ts = graph.timestamps_of(u);
                bias_params_[u] = bias.compute_per_vertex_params(ts);

                // Compute full-vertex weights into thread-local scratch.
                weights.resize(D);
                bias.compute_weights(ts, bias_params_[u], 0, weights.data());

                const int64_t alias_off_u  = alias_offsets_[u];

                if (is_solo_[u]) {
                    // §3.3 ad-hoc: single AliasTable of size D for u.
                    span<AliasEntry>   alias_slice(
                        alias_arena_.data() + alias_off_u, D);
                    span<const double> w_slice(weights.data(), D);
                    build_alias_into(alias_slice, w_slice, alias_scratch);
                    continue;
                }

                // Hierarchical path.
                const int32_t K            = K_max_[u];
                const int64_t totals_off_u = totals_offsets_[u];
                int64_t       alias_cursor  = 0;
                int64_t       totals_cursor = 0;
                for (int32_t k = 0; k <= K; ++k) {
                    const int32_t trunk_size_k = 1 << k;
                    const int32_t num_trunks_k = D >> k;
                    for (int32_t i = 0; i < num_trunks_k; ++i) {
                        const int32_t edge_lo = i * trunk_size_k;
                        span<AliasEntry>   alias_slice(
                            alias_arena_.data() + alias_off_u + alias_cursor,
                            trunk_size_k);
                        span<const double> w_slice(
                            weights.data() + edge_lo, trunk_size_k);

                        build_alias_into(alias_slice, w_slice, alias_scratch);

                        double total = 0.0;
                        for (int32_t j = 0; j < trunk_size_k; ++j) total += w_slice[j];
                        trunk_totals_[totals_off_u + totals_cursor] = total;

                        alias_cursor  += trunk_size_k;
                        totals_cursor += 1;
                    }
                }
            }
        }

        // 4. Build precomputed AuxiliaryIndex (paper §3.4), if budgeted.
        //    Solo vertices are skipped inside AuxIndex::build (no cover lookup
        //    needed — sampler dispatches them via is_solo() instead).
        if (aux_budget_bytes > 0) {
            aux_index_.build(graph, K_max_.data(), is_solo_.data(),
                             aux_budget_bytes);
        }
    }

    // ------------------------------------------------------------------------
    // Sample-time accessors
    // ------------------------------------------------------------------------

    // Precomputed cover-lookup table.  Empty if build was called with
    // aux_budget_bytes == 0 or projected memory exceeded the budget.
    const AuxIndex& aux() const noexcept { return aux_index_; }

    // K_u = ⌊log₂(D_u)⌋. Returns -1 for empty AND solo vertices.
    int32_t k_max_of(int32_t u) const noexcept { return K_max_[u]; }

    // True iff vertex u uses the §3.3 ad-hoc "single AliasTable" path
    // (degree D_u ∈ [1, kHpatDegreeThreshold]).  Empty (D=0) and hierarchical
    // (D > kHpatDegreeThreshold) vertices return false.
    bool is_solo(int32_t u) const noexcept { return is_solo_[u] != 0; }

    // For solo vertices: the entire size-D AliasTable.  Undefined for non-solo.
    AliasView solo_alias_view(const TemporalGraph& graph,
                              int32_t              u) const noexcept {
        const int32_t D = static_cast<int32_t>(graph.degree(u));
        return AliasView(span<const AliasEntry>(
            alias_arena_.data() + alias_offsets_[u],
            static_cast<std::size_t>(D)));
    }

    AliasView alias_view_of_trunk(const TemporalGraph& graph,
                                  int32_t u,
                                  int32_t k,
                                  int32_t i) const noexcept {
        const int32_t D = static_cast<int32_t>(graph.degree(u));
        const int64_t level_off = hpat_alias_level_offset(D, k);
        const int32_t trunk_size = 1 << k;
        return AliasView(span<const AliasEntry>(
            alias_arena_.data() + alias_offsets_[u] + level_off
                + static_cast<int64_t>(i) * trunk_size,
            static_cast<std::size_t>(trunk_size)));
    }

    double trunk_total_of(const TemporalGraph& graph,
                          int32_t u,
                          int32_t k,
                          int32_t i) const noexcept {
        const int32_t D = static_cast<int32_t>(graph.degree(u));
        const int64_t level_off = hpat_totals_level_offset(D, k);
        return trunk_totals_[totals_offsets_[u] + level_off + i];
    }

    const PerVertexParams& bias_params_of(int32_t u) const noexcept {
        return bias_params_[u];
    }

    // ------------------------------------------------------------------------
    // Memory accounting
    // ------------------------------------------------------------------------

    int64_t alias_entry_count() const noexcept {
        return static_cast<int64_t>(alias_arena_.size());
    }
    int64_t trunk_count() const noexcept {
        return static_cast<int64_t>(trunk_totals_.size());
    }
    int64_t memory_bytes() const noexcept {
        return static_cast<int64_t>(alias_arena_.size())   * sizeof(AliasEntry)
             + static_cast<int64_t>(trunk_totals_.size())  * sizeof(double)
             + static_cast<int64_t>(alias_offsets_.size()) * sizeof(int64_t)
             + static_cast<int64_t>(totals_offsets_.size())* sizeof(int64_t)
             + static_cast<int64_t>(K_max_.size())         * sizeof(int32_t)
             + static_cast<int64_t>(is_solo_.size())       * sizeof(int8_t)
             + static_cast<int64_t>(bias_params_.size())   * sizeof(PerVertexParams)
             + aux_index_.memory_bytes();
    }

private:
    std::vector<AliasEntry>      alias_arena_;
    std::vector<double>          trunk_totals_;
    std::vector<int64_t>         alias_offsets_;     // size N+1
    std::vector<int64_t>         totals_offsets_;    // size N+1
    std::vector<int32_t>         K_max_;             // N; −1 if D=0 or solo
    std::vector<int8_t>          is_solo_;           // N; 1 ⇔ §3.3 single-alias
    std::vector<PerVertexParams> bias_params_;       // size N
    AuxIndex                     aux_index_;         // empty if disabled
};

}  // namespace tea
