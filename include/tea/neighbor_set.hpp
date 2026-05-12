// NeighborSets — per-vertex sorted unique outgoing-neighbor sets, stored in a
// global flat CSR-style arena. Used by Node2VecBias for the β rejection test
// (paper §2.3 III): "is v_i a neighbor of the walker's previous vertex w?".
//
// Design notes:
//
// 1. SoA layout, SoA contract (CLAUDE.md): one flat int32_t neighbors_[]
//    arena across all vertices, plus CSR-style offsets_[N+1]. No nested
//    vector<vector<int32_t>>.
//
// 2. Sorted + deduped per-vertex, so contains() is a O(log D_u) branchless
//    binary search. No hashing overhead. For tgbl-comment (avg D = 44),
//    log₂ D ≈ 5.5 compares per lookup — within ~3× of a hash hit, and we
//    save the abseil dependency at Phase 5 entry.
//
// 3. Memory: ≤ E × 4 bytes globally + 8 × (N+1) for offsets. For
//    tgbl-comment (44M outbound edges), this is ≤ 176 MB.
//
// If profiling shows neighbor lookup is the bottleneck on huge-fanout
// vertices, we'll add a hybrid path that switches to a small hash set
// for D > some threshold. For now, keep the implementation simple.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "tea/graph.hpp"

namespace tea {

class NeighborSets {
public:
    NeighborSets() = default;

    // Build per-vertex sorted+unique outgoing-neighbor sets from `graph`.
    // Parallel across vertices via OpenMP. Thread-local scratch reused.
    void build(const TemporalGraph& graph) {
        const int32_t N = graph.num_vertices();
        offsets_.assign(N + 1, 0);

        // Pass 1: compute per-vertex deduped-neighbor counts (serial).
        // We dedupe by sorting+unique in thread-local scratch.
        std::vector<int32_t> sizes(N, 0);

        #pragma omp parallel
        {
            std::vector<int32_t> scratch;

            #pragma omp for schedule(dynamic, 64)
            for (int32_t u = 0; u < N; ++u) {
                const auto targets = graph.targets_of(u);
                if (targets.empty()) { sizes[u] = 0; continue; }
                scratch.assign(targets.begin(), targets.end());
                std::sort(scratch.begin(), scratch.end());
                const auto new_end = std::unique(scratch.begin(), scratch.end());
                sizes[u] = static_cast<int32_t>(new_end - scratch.begin());
            }
        }

        // Pass 2: prefix-sum to CSR offsets.
        int64_t total = 0;
        for (int32_t u = 0; u < N; ++u) {
            offsets_[u] = total;
            total += sizes[u];
        }
        offsets_[N] = total;

        neighbors_.assign(total, 0);

        // Pass 3: per-vertex fill (parallel). Thread-local scratch reused.
        #pragma omp parallel
        {
            std::vector<int32_t> scratch;

            #pragma omp for schedule(dynamic, 64)
            for (int32_t u = 0; u < N; ++u) {
                const auto targets = graph.targets_of(u);
                const int64_t lo = offsets_[u];
                const int32_t expected = sizes[u];
                if (expected == 0) continue;
                scratch.assign(targets.begin(), targets.end());
                std::sort(scratch.begin(), scratch.end());
                const auto new_end = std::unique(scratch.begin(), scratch.end());
                assert(static_cast<int32_t>(new_end - scratch.begin()) == expected);
                (void)expected;
                std::copy(scratch.begin(), new_end, neighbors_.begin() + lo);
            }
        }
    }

    // O(log D_u) branchless binary search.
    inline bool contains(int32_t u, int32_t v) const noexcept {
        const int64_t lo = offsets_[u];
        const int64_t hi = offsets_[u + 1];
        return std::binary_search(neighbors_.data() + lo,
                                  neighbors_.data() + hi, v);
    }

    int32_t degree(int32_t u) const noexcept {
        return static_cast<int32_t>(offsets_[u + 1] - offsets_[u]);
    }

    int64_t total_neighbor_count() const noexcept {
        return offsets_.empty() ? 0 : offsets_.back();
    }

    int64_t memory_bytes() const noexcept {
        return static_cast<int64_t>(neighbors_.size()) * sizeof(int32_t)
             + static_cast<int64_t>(offsets_.size())   * sizeof(int64_t);
    }

private:
    std::vector<int32_t> neighbors_;   // flat sorted+unique, all vertices
    std::vector<int64_t> offsets_;     // size N+1
};

}  // namespace tea
