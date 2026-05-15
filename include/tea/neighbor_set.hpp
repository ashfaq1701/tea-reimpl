// NeighborSets — per-vertex sorted unique UNDIRECTED-neighbor sets, stored
// in a global flat CSR-style arena.  Used by Node2VecBias for the β rejection
// test (paper §2.3 III): "is v_i a neighbor of the walker's previous vertex
// w in the underlying graph?".
//
// The β test asks an undirected adjacency question.  For an input edge
// (a, b, t) the resulting NeighborSets[a] contains b and NeighborSets[b]
// contains a — regardless of the input graph's directedness.  This matches
// the original node2vec paper's N(w) convention (undirected) and Tempest's
// build_node_adjacency_csr_std, which adds both endpoints for every edge.
//
// Why the input direction matters at build time:
//   • For a graph built with is_directed = false, TemporalGraph::build
//     already mirrored each edge to both endpoints, so graph.targets_of(u)
//     contains the full undirected neighborhood of u.  A single dedupe pass
//     yields the answer.
//   • For a graph built with is_directed = true, TemporalGraph stores only
//     the OUTBOUND side: graph.targets_of(u) is the DESTINATIONS of edges
//     leaving u (out-neighbors only).  We have to recover in-neighbors via
//     a transpose pass over the outbound-CSR ("v→u" exists iff u appears in
//     graph.targets_of(v)").
//
// History: the original implementation called build(graph) without an
// `input_was_directed` flag and only used graph.targets_of(u).  On directed
// inputs that gave a single-direction neighborhood — wrong relative to the
// node2vec convention, and observably different from Tempest's β distribution.
//
// Design notes (unchanged from the original):
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
// 3. Memory: ≤ 2 × E × 4 bytes globally + 8 × (N+1) for offsets (the 2× is
//    because directed inputs store each edge's endpoint pair twice; the
//    dedupe trims this back when the same pair appears multiple times).

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

    // Build per-vertex sorted+unique UNDIRECTED neighbor sets from `graph`.
    // Parallel across vertices via OpenMP. Thread-local scratch reused.
    //
    // `input_was_directed` MUST match the value passed to graph.build().
    // It controls whether we run a transpose pass to recover out-neighbors;
    // see file header for the rationale.
    void build(const TemporalGraph& graph, bool input_was_directed) {
        const int32_t N = graph.num_vertices();
        offsets_.assign(N + 1, 0);

        // -- Optional transpose pass: build per-vertex other-direction-
        //    adjacency as a CSR.  Skipped for undirected inputs
        //    (graph.targets_of(u) already contains both directions in that
        //    case).
        std::vector<int64_t> out_offsets;
        std::vector<int32_t> out_neighbors_arena;
        if (input_was_directed) {
            std::vector<int64_t> out_degree(N, 0);

            #pragma omp parallel for schedule(dynamic, 64)
            for (int32_t v = 0; v < N; ++v) {
                const auto in_tgts = graph.targets_of(v);
                for (std::size_t i = 0; i < in_tgts.size(); ++i) {
                    // in_tgts[i] is the destination of an edge leaving v —
                    // equivalently, an in-edge to in_tgts[i] from v.
                    #pragma omp atomic update
                    ++out_degree[in_tgts[i]];
                }
            }

            // Serial prefix sum (N small relative to E, memory-bound).
            out_offsets.assign(N + 1, 0);
            int64_t total_out = 0;
            for (int32_t u = 0; u < N; ++u) {
                out_offsets[u] = total_out;
                total_out += out_degree[u];
            }
            out_offsets[N] = total_out;
            out_neighbors_arena.assign(static_cast<std::size_t>(total_out), 0);

            // Scatter via atomic-capture cursor — same pattern as graph.cpp.
            std::vector<int64_t> cursor(N, 0);
            #pragma omp parallel for schedule(dynamic, 64)
            for (int32_t v = 0; v < N; ++v) {
                const auto in_tgts = graph.targets_of(v);
                for (std::size_t i = 0; i < in_tgts.size(); ++i) {
                    const int32_t u = in_tgts[i];
                    int64_t slot;
                    #pragma omp atomic capture
                    slot = cursor[u]++;
                    out_neighbors_arena[out_offsets[u] + slot] = v;
                }
            }
        }

        // -- Per-vertex pass 1: gather in + (optional) out into scratch,
        //    sort, dedupe, record size.
        std::vector<int32_t> sizes(N, 0);
        #pragma omp parallel
        {
            std::vector<int32_t> scratch;

            #pragma omp for schedule(dynamic, 64)
            for (int32_t u = 0; u < N; ++u) {
                const auto in_targets = graph.targets_of(u);
                scratch.assign(in_targets.begin(), in_targets.end());
                if (input_was_directed) {
                    const int64_t lo = out_offsets[u];
                    const int64_t hi = out_offsets[u + 1];
                    if (hi > lo) {
                        scratch.insert(scratch.end(),
                                       out_neighbors_arena.begin() + lo,
                                       out_neighbors_arena.begin() + hi);
                    }
                }
                if (scratch.empty()) { sizes[u] = 0; continue; }
                std::sort(scratch.begin(), scratch.end());
                const auto new_end = std::unique(scratch.begin(), scratch.end());
                sizes[u] = static_cast<int32_t>(new_end - scratch.begin());
            }
        }

        // -- Per-vertex pass 2: prefix-sum to global CSR offsets.
        int64_t total = 0;
        for (int32_t u = 0; u < N; ++u) {
            offsets_[u] = total;
            total += sizes[u];
        }
        offsets_[N] = total;
        neighbors_.assign(static_cast<std::size_t>(total), 0);

        // -- Per-vertex pass 3: redo the sort+dedupe and copy into the
        //    final compact arena.  We pay the sort/dedupe a second time
        //    rather than holding the dedupe results across passes — this
        //    keeps the scratch thread-local and avoids a per-vertex heap
        //    allocation in the global path.
        #pragma omp parallel
        {
            std::vector<int32_t> scratch;

            #pragma omp for schedule(dynamic, 64)
            for (int32_t u = 0; u < N; ++u) {
                const int32_t expected = sizes[u];
                if (expected == 0) continue;
                const auto in_targets = graph.targets_of(u);
                scratch.assign(in_targets.begin(), in_targets.end());
                if (input_was_directed) {
                    const int64_t lo = out_offsets[u];
                    const int64_t hi = out_offsets[u + 1];
                    if (hi > lo) {
                        scratch.insert(scratch.end(),
                                       out_neighbors_arena.begin() + lo,
                                       out_neighbors_arena.begin() + hi);
                    }
                }
                std::sort(scratch.begin(), scratch.end());
                const auto new_end = std::unique(scratch.begin(), scratch.end());
                assert(static_cast<int32_t>(new_end - scratch.begin()) == expected);
                (void)expected;
                std::copy(scratch.begin(), new_end,
                          neighbors_.begin() + offsets_[u]);
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
