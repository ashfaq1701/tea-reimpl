// TemporalGraph: per-vertex outgoing-edge lists sorted by t DESCENDING,
// stored SoA in three flat global arrays with a CSR-style vertex_offsets
// index. Honors the SoA + flat-arena contract from CLAUDE.md.
//
// Construction is parallel across vertices via OpenMP. No nested
// vector<vector<T>> anywhere — that pattern fragments the heap and defeats
// hardware prefetch.

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

    // Build the CSR + time-desc-sorted edge lists from a raw edge vector.
    // The edges vector is consumed (moved into temporaries during scatter)
    // and ends up empty.
    //
    // - num_vertices defaults to max(u,v)+1 if not given (dense encoding).
    // - is_directed=true   → each (u,v,t) contributes one out-edge u→v.
    // - is_directed=false  → each contributes two: u→v and v→u.
    void build(std::vector<Edge>&& edges,
               int32_t num_vertices = -1,
               bool is_directed = true);

    int32_t   num_vertices() const noexcept { return n_; }
    int64_t   num_edges()    const noexcept { return offsets_.empty() ? 0 : offsets_.back(); }

    // Per-vertex slices into the flat arenas. Returned spans alias internal
    // storage; they remain valid until the next build() or destruction.
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

    // Offset of vertex u's edge slice in the global flat CSR arenas.
    // Used by PAT/HPAT to compute their own per-vertex slices in alias arenas
    // that share the same offset layout.
    int64_t offset_of(int32_t u) const noexcept {
        return offsets_[u];
    }

    // Computes the candidate-set length Γ_t(u) = number of outgoing edges
    // of u with t' > t_prev. Since timestamps_of(u) is sorted DESCENDING,
    // the candidate set is the PREFIX of length L, where L is the count of
    // entries with t' > t_prev. We binary-search for the first entry <= t_prev.
    int64_t candidate_set_len(int32_t u, int64_t t_prev) const noexcept {
        const auto ts = timestamps_of(u);
        // Find first index i where ts[i] <= t_prev. That's L.
        // ts is sorted DESC, so we want the partition point of the predicate
        // "ts[i] > t_prev" — equivalent to upper_bound on a reverse-sorted
        // range. We do it with std::lower_bound on a custom comparator.
        auto first = ts.begin();
        auto last  = ts.end();
        auto it = std::lower_bound(
            first, last, t_prev,
            [](int64_t ts_val, int64_t threshold) { return ts_val > threshold; });
        return it - first;
    }

    // Paper §3.3 first ad-hoc optimization: minimum timestamp of any
    // incoming edge to u.  For vertices with no incoming edges, returns
    // kSentinelStartTimestamp (INT64_MIN) — i.e., no temporal constraint.
    // A walker reaching u via *any* incoming edge has t_prev ≥ this value,
    // so outgoing edges of u with t ≤ this value are never traversable
    // except possibly on the first hop from a sentinel start.  Use the
    // optional walk-engine flag `use_temporal_start` to seed the first
    // hop's t_prev from this value, which makes those edges fully unused
    // and matches the paper's walk-start convention (Alg 2's start_time).
    int64_t min_incoming_time_of(int32_t u) const noexcept {
        return min_incoming_t_[u];
    }

private:
    int32_t              n_ = 0;
    // CSR offsets: offsets_[u] .. offsets_[u+1] is the slice of edges for u.
    std::vector<int64_t> offsets_;
    // Flat arenas. Both length = total edge count.
    std::vector<int32_t> targets_;
    std::vector<int64_t> timestamps_;
    // Per-vertex min incoming-edge timestamp (size N).  Paper §3.3 opt #1.
    std::vector<int64_t> min_incoming_t_;
};

}  // namespace tea
