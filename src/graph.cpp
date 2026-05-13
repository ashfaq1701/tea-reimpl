// TemporalGraph::build implementation.
//
// Performance notes (post-Phase-1 review):
//   - Degree count is parallelized via OpenMP `atomic update` rather than
//     thread-local histograms because the histogram-then-reduce pattern
//     costs O(T·N) memory which is wasteful for our N≈1M, T≈16 case
//     (16·1M·8B = 128MB just for reductions). The atomic update on int64
//     contends on the same cache line ~1 in 60K times for typical degree
//     distributions; cost is negligible.
//   - Scatter uses OpenMP `atomic capture` for the per-vertex cursor. Same
//     argument as above. Each fetch-add is ~5ns uncontended.
//   - Per-vertex sort uses zip-and-sort (AoS of {ts, target} pairs) instead
//     of permutation sort, eliminating the random-access loads in the
//     comparator on high-degree vertices.
//   - Scratch buffers are reused across vertices within each thread (no
//     per-vertex allocation in the parallel-for hot loop).

#include "tea/graph.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "tea/config.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace tea {

namespace {

// AoS pair for the zip-and-sort. 16 bytes (8 + 4 + 4 pad), naturally aligned.
struct TsTgtPair {
    int64_t ts;
    int32_t target;
    int32_t _pad;  // explicit so the size is stable
};
static_assert(sizeof(TsTgtPair) == 16, "TsTgtPair should be 16 bytes");

}  // namespace

void TemporalGraph::build(std::vector<Edge>&& edges,
                          int32_t num_vertices_arg,
                          bool is_directed) {
    // 1. Determine vertex count (single pass — N is unknown).
    int32_t n = num_vertices_arg;
    if (n < 0) {
        int32_t max_id = -1;
        #pragma omp parallel for reduction(max:max_id)
        for (std::size_t i = 0; i < edges.size(); ++i) {
            const auto& e = edges[i];
            if (e.u > max_id) max_id = e.u;
            if (e.v > max_id) max_id = e.v;
        }
        n = max_id + 1;
        if (n <= 0) n = 0;
    }
    n_ = n;

    if (n == 0) {
        offsets_.assign(1, 0);
        targets_.clear();
        timestamps_.clear();
        min_incoming_t_.clear();
        std::vector<Edge>().swap(edges);
        return;
    }

    // 2. Parallel degree count via atomic update.
    std::vector<int64_t> degree(n, 0);
    const auto E = edges.size();
    #pragma omp parallel for schedule(static, 4096)
    for (std::size_t i = 0; i < E; ++i) {
        const Edge& e = edges[i];
        #pragma omp atomic update
        ++degree[e.u];
        if (!is_directed) {
            #pragma omp atomic update
            ++degree[e.v];
        }
    }

    // 3. Prefix-sum to compute CSR offsets. Serial — N is small (~1M),
    // memory-bound, parallel scan is not worth the complexity here.
    offsets_.assign(n + 1, 0);
    int64_t total = 0;
    for (int32_t u = 0; u < n; ++u) {
        offsets_[u] = total;
        total      += degree[u];
    }
    offsets_[n] = total;

    // 4. Allocate the two flat SoA arenas. assign() zero-fills; could skip,
    // but the cost is one streaming write per int and the prefetcher loves it.
    targets_.assign(total, 0);
    timestamps_.assign(total, 0);

    // 5. Parallel scatter via atomic capture on per-vertex cursor.
    //    The cursor is reused as scratch (we don't need it after this pass).
    std::vector<int64_t> cursor(n, 0);
    #pragma omp parallel for schedule(static, 4096)
    for (std::size_t i = 0; i < E; ++i) {
        const Edge& e = edges[i];
        int64_t slot;
        #pragma omp atomic capture
        slot = cursor[e.u]++;
        const int64_t pos = offsets_[e.u] + slot;
        targets_[pos]    = e.v;
        timestamps_[pos] = e.t;
        if (!is_directed) {
            int64_t slot2;
            #pragma omp atomic capture
            slot2 = cursor[e.v]++;
            const int64_t pos2 = offsets_[e.v] + slot2;
            targets_[pos2]    = e.u;
            timestamps_[pos2] = e.t;
        }
    }

    // 6. Per-vertex time-descending sort via zip-and-sort.
    //    Thread-local scratch reused across vertices (no per-vertex alloc).
    #pragma omp parallel
    {
        std::vector<TsTgtPair> zip;

        #pragma omp for schedule(dynamic, 64)
        for (int32_t u = 0; u < n; ++u) {
            const int64_t lo = offsets_[u];
            const int64_t hi = offsets_[u + 1];
            const int64_t d  = hi - lo;
            if (d <= 1) continue;

            zip.resize(d);
            for (int64_t i = 0; i < d; ++i) {
                zip[i].ts     = timestamps_[lo + i];
                zip[i].target = targets_[lo + i];
            }

            std::sort(zip.data(), zip.data() + d,
                [](const TsTgtPair& a, const TsTgtPair& b) {
                    return a.ts > b.ts;  // DESCENDING
                });

            for (int64_t i = 0; i < d; ++i) {
                timestamps_[lo + i] = zip[i].ts;
                targets_[lo + i]    = zip[i].target;
            }
        }
    }

    // 7. Per-vertex minimum incoming timestamp (paper §3.3 opt #1).
    //    Single-threaded sweep over the CSR: for each (u, v, t), record t
    //    as an incoming-time candidate for v.  E*1ns ≈ tens of ms on real
    //    graphs — atomic-min in parallel adds contention overhead and isn't
    //    a meaningful win over serial here.  Vertices without incoming
    //    edges keep kSentinelStartTimestamp = INT64_MIN.
    min_incoming_t_.assign(n, kSentinelStartTimestamp);
    {
        // First pass: bump every incoming target to a positive sentinel
        // (anything larger than any real ts).  We then track the actual min
        // by lazy initialisation on first observed incoming edge.
        // Simpler: just walk all edges and update a per-vertex tracker.
        std::vector<bool> seen(n, false);
        for (int32_t u = 0; u < n; ++u) {
            const int64_t lo = offsets_[u];
            const int64_t hi = offsets_[u + 1];
            for (int64_t i = lo; i < hi; ++i) {
                const int32_t v = targets_[i];
                const int64_t t = timestamps_[i];
                if (!seen[v] || t < min_incoming_t_[v]) {
                    min_incoming_t_[v] = t;
                    seen[v] = true;
                }
            }
        }
    }

    // 8. Drop the input edges. Releases ~16 bytes × E of memory.
    std::vector<Edge>().swap(edges);
}

}  // namespace tea
