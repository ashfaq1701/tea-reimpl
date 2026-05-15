// TemporalGraph::build implementation — outbound-CSR build for forward
// temporal walks.
//
// For directed graphs: each (u, v, t) input edge produces one stored entry
// under vertex u's adjacency, with target=v (the destination) and ts=t.  A
// forward walker at u with t_prev < t can pick this entry to step to v.
//
// For undirected graphs: each (u, v, t) adds entries to BOTH u's and v's
// adjacencies, each with the OTHER vertex as the stored target.
//
// Performance notes:
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
        std::vector<Edge>().swap(edges);
        return;
    }

    // 2. Parallel degree count via atomic update.
    //    Directed:   each (u, v, t) contributes to u's slot count.
    //    Undirected: each contributes to both u's and v's slot counts.
    std::vector<int64_t> degree(n, 0);
    const auto E = edges.size();
    #pragma omp parallel for schedule(static, 4096)
    for (std::size_t i = 0; i < E; ++i) {
        const Edge& e = edges[i];
        if (is_directed) {
            #pragma omp atomic update
            ++degree[e.u];
        } else {
            #pragma omp atomic update
            ++degree[e.u];
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
    //    Directed:   (u, v, t) → at u's slot: target=v, ts=t.
    //    Undirected: also at v's slot: target=u, ts=t.
    std::vector<int64_t> cursor(n, 0);
    #pragma omp parallel for schedule(static, 4096)
    for (std::size_t i = 0; i < E; ++i) {
        const Edge& e = edges[i];
        // Always store the outbound side for u (directed) or one of the
        // two sides for undirected.
        int64_t slot_u;
        #pragma omp atomic capture
        slot_u = cursor[e.u]++;
        const int64_t pos_u = offsets_[e.u] + slot_u;
        targets_[pos_u]    = e.v;
        timestamps_[pos_u] = e.t;
        if (!is_directed) {
            int64_t slot_v;
            #pragma omp atomic capture
            slot_v = cursor[e.v]++;
            const int64_t pos_v = offsets_[e.v] + slot_v;
            targets_[pos_v]    = e.u;
            timestamps_[pos_v] = e.t;
        }
    }

    // 6. Per-vertex time-DESCENDING sort via zip-and-sort.
    //    Thread-local scratch reused across vertices (no per-vertex alloc).
    //    Descending order makes the forward-walk candidate set
    //    Γ_{t_prev}(u) = {t > t_prev} a contiguous PREFIX [0, L) of the
    //    adjacency list, which matches PAT/HPAT's prefix-based trunk
    //    decomposition and the paper's Figure 5 layout.
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

    // 7. Drop the input edges. Releases ~16 bytes × E of memory.
    std::vector<Edge>().swap(edges);
}

}  // namespace tea
