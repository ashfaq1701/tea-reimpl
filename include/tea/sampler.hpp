// Sampler — per-step picker for a single walker. Paper §3.2 Algorithm 2
// (PAT variant; HPAT variant adds a different inner trunk-finder in Phase 4).
//
// Templated on BiasT so the bias is resolved at compile time — no virtual
// dispatch in the inner loop. Header-only so the compiler can inline through
// the alias / bias / sampler boundaries.
//
// Caller responsibility:
//   • Provide a SamplerScratch per thread (holds the partial-trunk scratch
//     buffers). Scratch is reused across millions of sample() calls — no
//     per-call heap allocations.
//
// Hot-path budget (PAT, typical step):
//   • 1× upper_bound on time-desc timestamps               (candidate-set length)
//   • 1× compute_per_vertex-bias call only at preprocess (already done at build)
//   • 1× upper_bound on trunk cumsums                       (full trunks)  O(log √D)
//   • 1× alias sample inside trunk                          O(1), branchless
//   • Partial trunk path adds 1× weight recompute + scan over ≤√D entries
//
// Algorithm 2 reproduced precisely:
//   given (u, t_prev):
//     Γ_len = candidate_set_len(u, t_prev)
//     if Γ_len == 0 → walk dies
//     T = pat.trunk_size_of(u); num_full = Γ_len / T; partial = Γ_len % T
//     full_total   = (num_full > 0) ? trunk_cs[num_full - 1] : 0
//     partial_total = (partial > 0) ? Σ recomputed weights : 0
//     r ∈ [0, full_total + partial_total)
//     if r < full_total:
//         pick a full trunk via ITS over cumsums; alias-sample inside
//     else:
//         ITS over partial-trunk on-the-fly prefix-sum

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tea/alias.hpp"
#include "tea/bias.hpp"
#include "tea/config.hpp"
#include "tea/graph.hpp"
#include "tea/hpat.hpp"
#include "tea/neighbor_set.hpp"
#include "tea/pat.hpp"
#include "tea/rng.hpp"
#include "tea/walk.hpp"

namespace tea {

// Result of one sample step. v == kWalkDeadSentinel ⇔ walk died.
struct SampledStep {
    int32_t v;
    int64_t t;
};

// Thread-local sampler scratch — partial-trunk weight buffer + cumsum.
// Reused across all sample() calls in a thread; heap-allocated once.
struct SamplerScratch {
    std::vector<double> partial_weights;
    std::vector<double> partial_cumsum;
};

// PAT sampler — Algorithm 2 from the paper, PAT inner-finder variant.
template <typename BiasT>
inline SampledStep sample_pat(const TemporalGraph& graph,
                              const Pat<BiasT>&    pat,
                              const BiasT&         bias,
                              int32_t              u,
                              int64_t              t_prev,
                              Pcg64&               rng,
                              SamplerScratch&      scratch) noexcept {
    // --- 1. Candidate-set length (binary search in time-desc timestamps).
    const auto ts_u   = graph.timestamps_of(u);
    const int64_t G   = graph.candidate_set_len(u, t_prev);
    if (G == 0) return SampledStep{kWalkDeadSentinel, 0};

    const int32_t T            = pat.trunk_size_of(u);
    const int64_t num_full     = G / T;
    const int64_t partial_size = G - num_full * T;  // ∈ [0, T)

    const auto trunk_cs = pat.trunk_cumsums_of(u);

    // --- 2. Full-trunks total weight.
    const double full_total =
        (num_full > 0) ? trunk_cs[num_full - 1] : 0.0;

    // --- 3. Partial-trunk weights (recompute on the fly per paper §3.2
    //         Case ②).  At most one partial trunk per sample call.
    double partial_total = 0.0;
    if (partial_size > 0) {
        const int64_t partial_start = num_full * T;
        const auto partial_ts = ts_u.subspan(
            static_cast<std::size_t>(partial_start),
            static_cast<std::size_t>(partial_size));

        scratch.partial_weights.resize(partial_size);
        bias.compute_weights(partial_ts,
                             pat.bias_params_of(u),
                             static_cast<int32_t>(partial_start),
                             scratch.partial_weights.data());

        scratch.partial_cumsum.resize(partial_size);
        double running = 0.0;
        for (int64_t i = 0; i < partial_size; ++i) {
            running += scratch.partial_weights[i];
            scratch.partial_cumsum[i] = running;
        }
        partial_total = running;
    }

    const double total = full_total + partial_total;
    if (total <= 0.0) {
        // No probability mass — every candidate had weight 0. Walk dies.
        // Only possible if the bias produces all-zero weights (e.g.,
        // exp() underflow on every edge of u's candidate set on a
        // long-span dataset). Returning dead here is safe — the walk
        // simply doesn't extend, which matches the paper's semantics
        // (no edge to pick).
        return SampledStep{kWalkDeadSentinel, 0};
    }

    // --- 4. ITS over (full_trunks ∪ partial_trunk) to pick where.
    const double r = rng.next_u01() * total;

    int64_t chosen_edge_pos;  // position in u's time-desc edge list

    if (r < full_total) {
        // Pick a full trunk via ITS.
        // upper_bound: first cumsum strictly greater than r.
        const double* cs_begin = trunk_cs.data();
        const double* cs_end   = cs_begin + num_full;
        const double* it = std::upper_bound(cs_begin, cs_end, r);
        // it == cs_end can only happen due to FP drift at exactly r ==
        // full_total; clamp to last trunk.
        const int64_t trunk_idx = (it == cs_end)
            ? num_full - 1
            : static_cast<int64_t>(it - cs_begin);

        // Alias-sample inside the chosen full trunk.
        const AliasView av = pat.alias_view_of_trunk(graph, u,
                                                     static_cast<int32_t>(trunk_idx));
        const uint32_t  bucket = rng.next_below(static_cast<uint32_t>(av.size()));
        const float     pf     = rng.next_f01();
        const int32_t   local  = static_cast<int32_t>(av.sample(bucket, pf));

        chosen_edge_pos = trunk_idx * T + local;
    } else {
        // Pick inside the partial trunk via ITS over the on-the-fly cumsum.
        const double r2 = r - full_total;
        const double* pc_begin = scratch.partial_cumsum.data();
        const double* pc_end   = pc_begin + partial_size;
        const double* it = std::upper_bound(pc_begin, pc_end, r2);
        const int64_t local = (it == pc_end)
            ? partial_size - 1
            : static_cast<int64_t>(it - pc_begin);

        chosen_edge_pos = num_full * T + local;
    }

    // --- 5. Map chosen_edge_pos → (v, t) via graph.
    const auto targets_u = graph.targets_of(u);
    assert(chosen_edge_pos >= 0
        && chosen_edge_pos < static_cast<int64_t>(targets_u.size()));
    return SampledStep{targets_u[chosen_edge_pos], ts_u[chosen_edge_pos]};
}

// =============================================================================
// HPAT sampler — paper §3.3, Algorithm 2 with HPAT inner finder.
//
// Cleaner than PAT: HPAT's binary decomposition produces a cover that EXACTLY
// partitions [0, Γ_len) for any Γ_len ∈ [0, D_u]. No partial-trunk case, no
// on-the-fly weight recomputation, no bias_params lookup. The hot path is:
//
//   1. Γ_len = candidate_set_len(u, t_prev)
//   2. decompose_to_trunks(Γ_len, K_u, cover[]) → m trunks (m ≤ K_u+1 ≤ ~30)
//   3. Build small ITS cumsum over cover trunk-totals (stack array of size m)
//   4. ITS pick → choose trunk
//   5. Alias-sample inside it
//   6. Map (trunk_level, trunk_index, local_in_trunk) → edge position
//
// Per-sample work: O(K_u) for cover decomposition + O(K_u) for the small ITS.
// HPAT's theoretical O(log log D) requires the precomputed AuxiliaryIndex
// (skipped here per CLAUDE.md, recoverable later as an opt-in fast path).
// =============================================================================

template <typename BiasT>
inline SampledStep sample_hpat(const TemporalGraph& graph,
                               const Hpat<BiasT>&    hpat,
                               const BiasT&          bias,
                               int32_t               u,
                               int64_t               t_prev,
                               Pcg64&                rng,
                               SamplerScratch&       scratch) noexcept {
    const auto ts_u      = graph.timestamps_of(u);
    const auto targets_u = graph.targets_of(u);
    const int64_t G = graph.candidate_set_len(u, t_prev);
    if (G == 0) return SampledStep{kWalkDeadSentinel, 0};

    // §3.3 ad-hoc: low-degree vertices use a single AliasTable of size D.
    // Sample-time: alias-sample directly when G == D, partial-prefix ITS
    // recompute when G < D (same path the PAT sampler uses for its partial
    // last trunk).
    if (hpat.is_solo(u)) {
        const int64_t D = static_cast<int64_t>(targets_u.size());
        if (G == D) {
            const AliasView av = hpat.solo_alias_view(graph, u);
            const uint32_t bucket = rng.next_below(static_cast<uint32_t>(D));
            const float    pf     = rng.next_f01();
            const int64_t  pos    = static_cast<int64_t>(av.sample(bucket, pf));
            return SampledStep{targets_u[pos], ts_u[pos]};
        }
        // Partial-prefix path — same recompute logic as PAT's Case ②.
        scratch.partial_weights.resize(G);
        bias.compute_weights(ts_u.subspan(0, G),
                             hpat.bias_params_of(u),
                             /*slice_start_pos=*/0,
                             scratch.partial_weights.data());
        scratch.partial_cumsum.resize(G);
        double running = 0.0;
        for (int64_t i = 0; i < G; ++i) {
            running += scratch.partial_weights[i];
            scratch.partial_cumsum[i] = running;
        }
        if (running <= 0.0) return SampledStep{kWalkDeadSentinel, 0};
        const double  r       = rng.next_u01() * running;
        const double* cs_beg  = scratch.partial_cumsum.data();
        const double* cs_end  = cs_beg + G;
        const double* it      = std::upper_bound(cs_beg, cs_end, r);
        const int64_t pos     = (it == cs_end) ? (G - 1)
                                               : static_cast<int64_t>(it - cs_beg);
        return SampledStep{targets_u[pos], ts_u[pos]};
    }

    const int32_t K = hpat.k_max_of(u);
    assert(K >= 0);  // D_u > 0 since G > 0

    // 1. Cover for (u, G).  Precomputed AuxIndex (paper §3.4) if active,
    //    else on-the-fly binary decomposition.  Same cover either way.
    TrunkRef on_fly[64];               // stack fallback
    const TrunkRef* cover;             // points into aux arena or on_fly
    int32_t m;
    if (hpat.aux().active()) {
        const auto cv = hpat.aux().cover_for(u, G);
        cover = cv.data();
        m     = static_cast<int32_t>(cv.size());
    } else {
        m     = decompose_to_trunks(G, K, on_fly);
        cover = on_fly;
    }
    assert(m > 0);

    // 2. Build small ITS cumsum over cover trunk-totals.
    double cumsum[64];
    double total = 0.0;
    for (int32_t j = 0; j < m; ++j) {
        total += hpat.trunk_total_of(graph, u, cover[j].level, cover[j].index);
        cumsum[j] = total;
    }
    if (total <= 0.0) {
        // All-zero weights — walk dies (same semantics as PAT path).
        return SampledStep{kWalkDeadSentinel, 0};
    }

    // 3. ITS pick. Linear scan is faster than binary search for m ≤ ~32.
    const double r = rng.next_u01() * total;
    int32_t j_pick = m - 1;  // default = last (handles FP-drift r == total case)
    for (int32_t j = 0; j < m; ++j) {
        if (r < cumsum[j]) { j_pick = j; break; }
    }
    const TrunkRef& tr = cover[j_pick];

    // 4. Alias-sample inside the chosen trunk.
    const AliasView av = hpat.alias_view_of_trunk(graph, u, tr.level, tr.index);
    const uint32_t bucket = rng.next_below(static_cast<uint32_t>(av.size()));
    const float    pf     = rng.next_f01();
    const int32_t  local  = static_cast<int32_t>(av.sample(bucket, pf));

    // 5. Map (level, index, local_in_trunk) → position in u's edge list → (v, t).
    const int64_t edge_pos =
        (static_cast<int64_t>(tr.index) << tr.level) + local;

    assert(edge_pos >= 0 && edge_pos < static_cast<int64_t>(targets_u.size()));
    return SampledStep{targets_u[edge_pos], ts_u[edge_pos]};
}

// =============================================================================
// Node2Vec rejection-loop samplers (paper §4 Algorithm 2 lines 19–22).
//
// Wraps a PAT or HPAT proposal in a do/until-accepted loop. The proposal
// distribution is the static-exp PAT/HPAT (Node2VecBias forwards static_weight
// to ExponentialBias). The rejection test multiplies in the β factor that
// depends on the walker's previous vertex w.
//
// Retry cap is kNode2VecMaxRetries (config.hpp, default 1024). On overflow,
// accept the last proposal — this is a defensive guard against pathological
// (p, q) configurations; on benign workloads acceptance is hit within a
// handful of retries (since β/β_max ≥ min(1/p, 1, 1/q) / β_max > 0).
// =============================================================================

// Helper: one rejection-loop attempt around an arbitrary proposal function.
// proposal_fn takes (Pcg64&, SamplerScratch&) and returns SampledStep.
template <typename ProposalFn, typename BiasT>
inline SampledStep rejection_loop(
        ProposalFn&&         propose,
        const BiasT&         bias,
        int32_t              prev_u,
        const NeighborSets&  neighbors,
        Pcg64&               rng,
        SamplerScratch&      scratch) noexcept {
    SampledStep last_step{kWalkDeadSentinel, 0};
    for (int32_t retry = 0; retry < kNode2VecMaxRetries; ++retry) {
        SampledStep step = propose(rng, scratch);
        if (step.v == kWalkDeadSentinel) return step;  // walk dies
        last_step = step;
        if (prev_u < 0) return step;  // first step (no prev vertex) — accept.
        const double accept = bias.accept_ratio(prev_u, step.v, neighbors);
        const double u01    = rng.next_u01();
        if (u01 < accept) return step;
    }
    // Retry cap exceeded — accept last proposal (defensive). Should be
    // unreachable on workloads with sane (p, q) settings.
    return last_step;
}

// Node2Vec sampler on PAT.
template <typename BiasT>
inline SampledStep sample_pat_node2vec(
        const TemporalGraph& graph,
        const Pat<BiasT>&    pat,
        const BiasT&         bias,
        const NeighborSets&  neighbors,
        int32_t              u,
        int64_t              t_prev,
        int32_t              prev_u,  // walker's previous vertex; -1 if first step
        Pcg64&               rng,
        SamplerScratch&      scratch) noexcept {
    return rejection_loop(
        [&](Pcg64& r, SamplerScratch& s) {
            return sample_pat(graph, pat, bias, u, t_prev, r, s);
        },
        bias, prev_u, neighbors, rng, scratch);
}

// Node2Vec sampler on HPAT.
template <typename BiasT>
inline SampledStep sample_hpat_node2vec(
        const TemporalGraph& graph,
        const Hpat<BiasT>&   hpat,
        const BiasT&         bias,
        const NeighborSets&  neighbors,
        int32_t              u,
        int64_t              t_prev,
        int32_t              prev_u,
        Pcg64&               rng,
        SamplerScratch&      scratch) noexcept {
    return rejection_loop(
        [&](Pcg64& r, SamplerScratch& s) {
            return sample_hpat(graph, hpat, bias, u, t_prev, r, s);
        },
        bias, prev_u, neighbors, rng, scratch);
}

}  // namespace tea
