// AuxIndex — paper §3.4. Precomputed (u, L) → HPAT trunk cover lookup.
//
// Background:
//   At sample time, HPAT needs the cover of L = |Γ_{t_prev}(u)|. The cover
//   is the binary decomposition L = Σ 2^{k_j} mapped to trunks of vertex u.
//   The on-the-fly version (`decompose_to_trunks` in hpat.hpp) is O(log D).
//   The paper proposes a precomputed table that reduces this to O(1).
//
// Storage (CSR-of-CSR):
//   vertex_L_offset_base_[N+1]  — per-vertex base into L_offset_arena_
//   L_offset_arena_[Σ_u (D_u+1)+1] — per-(u,L) start offset into entries_
//   entries_[Σ_u Σ_{L=1..D_u} popcount(L)] — flat TrunkRef list
//
// Lookup (cover_for(u, L)):
//   base    = vertex_L_offset_base_[u]
//   start_e = L_offset_arena_[base + L]
//   end_e   = L_offset_arena_[base + L + 1]
//   return span<const TrunkRef>(entries_.data() + start_e, end_e - start_e)
//
// Two indirections, both into hot arrays the walk engine touches every step.
// Cover sizes are tiny (popcount(L) ≤ log₂(D_u) + 1).
//
// Memory budget:
//   If the projected total bytes exceeds `memory_budget_bytes`, build() leaves
//   the index empty and returns false. The HPAT sampler then falls back to
//   on-the-fly decomposition. Default budget = kAuxIndexMaxBytes (4 GB).
//
// Construction:
//   Per paper §4.2 "embarrassingly parallel". After a serial pass to compute
//   per-vertex entry counts + base offsets, each vertex u writes independently
//   into its own slice of L_offset_arena_ and entries_. Lock-free.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/trunk_ref.hpp"  // for TrunkRef + decompose_to_trunks
#include "tea/walk.hpp"       // for tea::span

namespace tea {

class AuxIndex {
public:
    AuxIndex() = default;

    // Build per-vertex cover table.
    // Returns true if built; false (and state cleared) if projected memory
    // would exceed `memory_budget_bytes` — callers must then fall back to
    // on-the-fly decomposition.
    //
    // `K_max[u]` is HPAT's per-vertex max level (= ⌊log₂ D_u⌋, or -1 for
    // empty / solo vertices).  `is_solo[u]` flags vertices that the sampler
    // handles via the §3.3 single-AliasTable path and which therefore need
    // no cover entries here — they're skipped to save the O(D) per-vertex
    // L-offset storage.  Pass `is_solo = nullptr` to treat no vertices as solo.
    bool build(const TemporalGraph& graph,
               const int32_t*       K_max,
               const int8_t*        is_solo,
               std::size_t          memory_budget_bytes = kAuxIndexMaxBytes) {
        const int32_t N = graph.num_vertices();

        // --- 1. Per-vertex entry counts + cumulative offsets (serial; N small).
        std::vector<int64_t> entries_start(N + 1, 0);
        vertex_L_offset_base_.assign(N + 1, 0);
        int64_t cum_L = 0;
        int64_t cum_e = 0;
        for (int32_t u = 0; u < N; ++u) {
            vertex_L_offset_base_[u] = cum_L;
            entries_start[u]         = cum_e;
            const int64_t D = graph.degree(u);
            const bool    solo = is_solo && is_solo[u];
            if (solo) continue;  // no L-offsets, no entries for solo vertices
            cum_L += D + 1;      // L ∈ [0, D]
            for (int64_t L = 1; L <= D; ++L) {
                cum_e += __builtin_popcountll(static_cast<uint64_t>(L));
            }
        }
        vertex_L_offset_base_[N] = cum_L;
        entries_start[N]         = cum_e;

        // --- 2. Memory budget check.
        const int64_t bytes_needed =
            static_cast<int64_t>(vertex_L_offset_base_.size()) * sizeof(int64_t)
          + (cum_L + 1)                                         * sizeof(int64_t)
          + cum_e                                                * sizeof(TrunkRef);
        if (static_cast<std::size_t>(bytes_needed) > memory_budget_bytes) {
            std::vector<int64_t>().swap(vertex_L_offset_base_);
            std::vector<int64_t>().swap(L_offset_arena_);
            std::vector<TrunkRef>().swap(entries_);
            active_ = false;
            return false;
        }

        // --- 3. Allocate.
        L_offset_arena_.assign(static_cast<std::size_t>(cum_L + 1), 0);
        entries_.assign(static_cast<std::size_t>(cum_e), TrunkRef{0, 0});

        // --- 4. Parallel per-vertex fill. Each vertex writes only its own
        //         slice of L_offset_arena_ and entries_. Lock-free.
        #pragma omp parallel for schedule(dynamic, 64)
        for (int32_t u = 0; u < N; ++u) {
            const bool solo = is_solo && is_solo[u];
            if (solo) continue;  // skipped vertices have no cover entries

            const int64_t D     = graph.degree(u);
            const int32_t K     = K_max[u];   // -1 if D == 0; loop below skips
            const int64_t L_b   = vertex_L_offset_base_[u];
            int64_t       cur   = entries_start[u];

            for (int64_t L = 0; L <= D; ++L) {
                L_offset_arena_[L_b + L] = cur;
                if (L > 0) {
                    const int32_t cnt = decompose_to_trunks(
                        L, K, &entries_[cur]);
                    cur += cnt;
                }
            }
        }
        // Final sentinel so cover_for(N-1, D_{N-1}) can read [base+L+1].
        L_offset_arena_[cum_L] = cum_e;

        active_ = true;
        return true;
    }

    bool active() const noexcept { return active_; }

    // Returns span over the cover entries for (u, L). For L=0, returns empty
    // span (no trunks needed for the empty candidate set).
    inline span<const TrunkRef> cover_for(int32_t u, int64_t L) const noexcept {
        assert(active_);
        const int64_t base    = vertex_L_offset_base_[u];
        const int64_t start_e = L_offset_arena_[base + L];
        const int64_t end_e   = L_offset_arena_[base + L + 1];
        return span<const TrunkRef>(
            entries_.data() + start_e,
            static_cast<std::size_t>(end_e - start_e));
    }

    int64_t memory_bytes() const noexcept {
        if (!active_) return 0;
        return static_cast<int64_t>(vertex_L_offset_base_.size()) * sizeof(int64_t)
             + static_cast<int64_t>(L_offset_arena_.size())       * sizeof(int64_t)
             + static_cast<int64_t>(entries_.size())              * sizeof(TrunkRef);
    }

    int64_t entry_count() const noexcept {
        return static_cast<int64_t>(entries_.size());
    }

private:
    bool                  active_ = false;
    std::vector<int64_t>  vertex_L_offset_base_;  // size N+1
    std::vector<int64_t>  L_offset_arena_;        // size Σ(D_u+1)+1
    std::vector<TrunkRef> entries_;               // size Σ popcount(L), L>0
};

}  // namespace tea
