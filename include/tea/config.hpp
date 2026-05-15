#pragma once

#include <cstddef>
#include <cstdint>

namespace tea {

// =============================================================================
// Compile-time architectural constants.  These are the contract for the
// design decisions documented in CLAUDE.md §perf-architecture.
// =============================================================================

// --- AuxiliaryIndex storage budget ------------------------------------------
// Two implementations exist (explicit lookup table vs on-the-fly bit walk).
// We pick at runtime: if the explicit table would exceed this many bytes
// across the whole graph, we fall back to on-the-fly.  Default ~4 GB.
constexpr std::size_t kAuxIndexMaxBytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;

// --- PAT trunk size heuristic -----------------------------------------------
// Per the paper §3.2, in-memory choice is trunk_size = ceil(sqrt(D)). We
// clamp to a minimum so very-small-degree vertices don't end up with a
// 1-edge trunk (worse than no trunking).
constexpr int kPatTrunkSizeMin = 8;

// --- WalkEngine output stride -----------------------------------------------
// Each walk gets max_walk_len NodeStep slots in a thread-local flat buffer.
// One walk = walks_buf[walk_id * max_walk_len .. (walk_id+1) * max_walk_len).
// Walks that die early leave trailing slots with v = kWalkDeadSentinel.
constexpr int32_t kWalkDeadSentinel = -1;

// First-step "previous timestamp" sentinel — every edge of the start vertex
// is a candidate (Γ_t(u) = all of u's edges).  We do FORWARD walks: the
// candidate set is {e : t > t_prev}, so the sentinel must be smaller than
// every real timestamp.  INT64_MIN makes "t > sentinel" admit every edge.
constexpr int64_t kSentinelStartTimestamp = INT64_MIN;

// --- Low-degree HPAT threshold (paper §3.3 second ad-hoc optimization) ------
// "If the out-degree of a vertex is relatively low, we can simply build
// alias tables for its specific out edges." Vertices with D ≤ this threshold
// bypass the level-by-level HPAT hierarchy and use a single AliasTable of
// size D, halving the per-vertex alias storage and the build work.
// Sampling on a solo vertex: alias-sample directly when |Γ_t(u)| = D,
// partial-prefix ITS recompute when |Γ_t(u)| < D (same path the PAT sampler
// uses for its partial last trunk).
// Threshold value is not specified by the paper; 64 lets the size-D alias
// table fit in two L1 cache lines on most x86_64.
constexpr int kHpatDegreeThreshold = 64;

// --- Exponential bias time rescaling ----------------------------------------
// Raw unix timestamps in datasets (~1.7e9) make exp(t) overflow. We rescale
// per-vertex timestamps to [0, kExpBiasRescaleTarget] before exp() so the
// magnitude stays in float range.  See CLAUDE.md §7 (numerical caveat).
constexpr double kExpBiasRescaleTarget = 80.0;

// --- Node2Vec rejection-loop retry cap --------------------------------------
// Guard against degenerate (p, q) configurations that produce unbounded
// rejection loops.  Cap at 1024 retries; on overflow, accept the last
// proposal and document the rare event.
constexpr int kNode2VecMaxRetries = 1024;

}  // namespace tea
