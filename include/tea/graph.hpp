// TemporalGraph: per-vertex INBOUND adjacency lists sorted by t ASCENDING,
// stored SoA in three flat global arrays with a CSR-style vertex_offsets
// index. Honors the SoA + flat-arena contract from CLAUDE.md.
//
// Construction is parallel across vertices via OpenMP. No nested
// vector<vector<T>> anywhere — that pattern fragments the heap and defeats
// hardware prefetch.
//
// Direction model: walks are BACKWARD-IN-TIME and trace history through
// INCOMING edges (the standard temporal-walk convention shared with
// Tempest, CAW, CTDNE).  For a walker at vertex v at time t_prev, the
// candidate set is
//   Γ_{t_prev}(v) = { (u, v, t) ∈ E : t < t_prev }
// i.e. edges that pointed INTO v with timestamp before t_prev.  Picking
// one moves the walker to its source u.
//
// Storage consequence: for each directed input edge (u, v, t), this class
// stores it under vertex v's slice with "target = u" (the source that
// flowed into v).  For undirected inputs, the edge is added to both
// endpoints' slices, mirroring each as the other's "target."  With ASC
// timestamps the backward candidate Γ_{t_prev}(v) is then the PREFIX
// [0, L) of v's adjacency, keeping PAT/HPAT's prefix-based trunk
// decomposition natural.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tea/edge.hpp"
#include "tea/walk.hpp"  // for tea::span

namespace tea {

class TemporalGraph {
public:
    TemporalGraph() = default;

    // Build the inbound CSR + time-asc-sorted adjacency lists from a raw
    // edge vector.  The edges vector is consumed (moved into temporaries
    // during scatter) and ends up empty.
    //
    // - num_vertices defaults to max(u,v)+1 if not given (dense encoding).
    // - is_directed=true   → each (u,v,t) contributes one entry under v
    //                        with target=u (inbound to v).
    // - is_directed=false  → each contributes two: at u with target=v and
    //                        at v with target=u (both endpoints).
    void build(std::vector<Edge>&& edges,
               int32_t num_vertices = -1,
               bool is_directed = true);

    int32_t   num_vertices() const noexcept { return n_; }
    int64_t   num_edges()    const noexcept { return offsets_.empty() ? 0 : offsets_.back(); }

    // Per-vertex slices into the flat arenas. Returned spans alias internal
    // storage; they remain valid until the next build() or destruction.
    //
    // For vertex v, targets_of(v)[i] is the SOURCE of the i-th inbound
    // edge into v (under directed mode) or the OTHER ENDPOINT of the i-th
    // adjacent edge (under undirected mode).  timestamps_of(v)[i] is the
    // timestamp of that edge.  The two arrays share order and are sorted
    // by timestamp ASCENDING.
    span<const int32_t> targets_of(int32_t u) const noexcept {
        return span<const int32_t>(targets_.data()    + offsets_[u],
                                   offsets_[u + 1] - offsets_[u]);
    }
    span<const int64_t> timestamps_of(int32_t u) const noexcept {
        return span<const int64_t>(timestamps_.data() + offsets_[u],
                                   offsets_[u + 1] - offsets_[u]);
    }
    int64_t degree(int32_t u) const noexcept {
        return offsets_[u + 1] - offsets_[u];
    }

    // Offset of vertex u's adjacency slice in the global flat CSR arenas.
    // Used by PAT/HPAT to compute their own per-vertex slices in alias arenas
    // that share the same offset layout.
    int64_t offset_of(int32_t u) const noexcept {
        return offsets_[u];
    }

    // Computes the backward candidate-set length Γ_{t_prev}(v) = number of
    // adjacency entries of v with t' < t_prev.  Since timestamps_of(v) is
    // sorted ASCENDING, the candidate set is the PREFIX of length L, where
    // L is the count of entries with t' < t_prev.  We binary-search for
    // the first entry >= t_prev.
    int64_t candidate_set_len(int32_t u, int64_t t_prev) const noexcept {
        const auto ts = timestamps_of(u);
        auto it = std::lower_bound(ts.begin(), ts.end(), t_prev);
        return it - ts.begin();
    }

private:
    int32_t              n_ = 0;
    // CSR offsets: offsets_[u] .. offsets_[u+1] is the slice of edges for u.
    std::vector<int64_t> offsets_;
    // Flat arenas. Both length = total stored-entry count (== E for
    // directed graphs, 2E for undirected).
    std::vector<int32_t> targets_;
    std::vector<int64_t> timestamps_;
};

}  // namespace tea
