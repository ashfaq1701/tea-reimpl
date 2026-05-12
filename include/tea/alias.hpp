// AliasTable: Vose's alias method for O(1) categorical sampling.
//
// Three layers in this header so PAT/HPAT can stamp millions of tables into
// global flat arenas without per-table allocations:
//
//   • AliasEntry           — packed (uint32_t alias, float prob) = 8 bytes
//   • AliasView            — non-owning span<const AliasEntry> with inline sample()
//   • build_alias_into(...) — free function: stamp Vose output into a caller-
//                              provided span<AliasEntry>, using caller-provided
//                              scratch buffers (no per-call alloc)
//   • AliasTable           — convenience wrapper that owns its storage. Used by
//                              tests and any standalone caller; thin shim over
//                              the lower-level pieces.
//
// PAT (Phase 3) and HPAT (Phase 4) use AliasView + build_alias_into directly.

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "tea/walk.hpp"  // for tea::span

namespace tea {

// ---------------------------------------------------------------------------
// AliasEntry: 8 bytes. Half the cache footprint of a naive {int32, double}
// layout (which pads to 16). 24-bit float mantissa is more than enough for
// alias-table thresholds.
// ---------------------------------------------------------------------------
struct AliasEntry {
    uint32_t alias;
    float    prob;
};
static_assert(sizeof(AliasEntry) == 8, "AliasEntry must be 8 bytes packed");

// ---------------------------------------------------------------------------
// Non-owning view. The sample-time API. All PAT/HPAT lookups go through here.
// ---------------------------------------------------------------------------
class AliasView {
public:
    AliasView() = default;
    explicit AliasView(span<const AliasEntry> entries) noexcept
        : entries_(entries) {}

    inline uint32_t sample(uint32_t bucket_idx,
                           float prob_rng) const noexcept {
        assert(bucket_idx < entries_.size());
        const AliasEntry e = entries_[bucket_idx];
        // branchless: 1 compare, 1 cmov on x86
        return (prob_rng < e.prob) ? bucket_idx : e.alias;
    }

    std::size_t size()  const noexcept { return entries_.size(); }
    bool        empty() const noexcept { return entries_.size() == 0; }
    span<const AliasEntry> entries() const noexcept { return entries_; }

private:
    span<const AliasEntry> entries_;
};

// ---------------------------------------------------------------------------
// Reusable scratch buffers for Vose's algorithm. Build PAT/HPAT once per
// thread, reuse across millions of trunk-builds. Without this, each build
// would allocate three vectors → catastrophic on multi-million-edge graphs
// where the PAT/HPAT pass calls into the alias builder once per trunk.
// ---------------------------------------------------------------------------
struct AliasBuildScratch {
    std::vector<uint32_t> under;
    std::vector<uint32_t> over;
    std::vector<double>   scaled;
};

// ---------------------------------------------------------------------------
// Stamp a Vose alias table into the caller-provided `out` span, using the
// caller-provided `scratch`. `out` must already be sized to weights.size().
//
// Preconditions:
//   • out.size() == weights.size()
//   • weights[i] >= 0 for all i
//
// Postconditions:
//   • out[i].prob ∈ [0, 1] for all i
//   • out[i].alias < weights.size()
//   • The resulting table samples each i with frequency weights[i] / Σ weights.
//
// Degenerate handling: if Σ weights == 0 or weights.empty(), the table is
// filled with self-aliases at prob=1 (samples uniformly across all slots).
// ---------------------------------------------------------------------------
inline void build_alias_into(span<AliasEntry> out,
                             span<const double> weights,
                             AliasBuildScratch& scratch) {
    const std::size_t n = weights.size();
    assert(out.size() == n);
    if (n == 0) return;

    // Total weight.
    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i) total += weights[i];

    // Degenerate: zero total → uniform self-alias.
    if (total <= 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = AliasEntry{static_cast<uint32_t>(i), 1.0f};
        }
        return;
    }

    // Reuse scratch (resize is a no-op if already big enough).
    scratch.under.clear();
    scratch.over.clear();
    scratch.scaled.resize(n);
    scratch.under.reserve(n);
    scratch.over.reserve(n);

    const double scale = static_cast<double>(n) / total;
    for (std::size_t i = 0; i < n; ++i) {
        scratch.scaled[i] = weights[i] * scale;
        if (scratch.scaled[i] < 1.0) scratch.under.push_back(static_cast<uint32_t>(i));
        else                          scratch.over.push_back(static_cast<uint32_t>(i));
    }

    while (!scratch.under.empty() && !scratch.over.empty()) {
        const uint32_t i_u = scratch.under.back(); scratch.under.pop_back();
        const uint32_t i_o = scratch.over.back();  scratch.over.pop_back();

        out[i_u].prob  = static_cast<float>(scratch.scaled[i_u]);
        out[i_u].alias = i_o;

        scratch.scaled[i_o] = (scratch.scaled[i_o] + scratch.scaled[i_u]) - 1.0;
        if (scratch.scaled[i_o] < 1.0) scratch.under.push_back(i_o);
        else                            scratch.over.push_back(i_o);
    }
    // Remaining slots: scaled ≈ 1 modulo float drift. Self-alias at prob=1.
    for (uint32_t i : scratch.over) {
        out[i] = AliasEntry{i, 1.0f};
    }
    for (uint32_t i : scratch.under) {
        out[i] = AliasEntry{i, 1.0f};
    }
}

// ---------------------------------------------------------------------------
// Convenience wrapper for standalone use (tests, single-table callers).
// Owns its storage. PAT/HPAT bypass this and use AliasView + build_alias_into.
// ---------------------------------------------------------------------------
class AliasTable {
public:
    AliasTable() = default;

    void build(span<const double> weights) {
        entries_.assign(weights.size(), AliasEntry{0, 0.0f});
        AliasBuildScratch scratch;
        build_alias_into(
            span<AliasEntry>(entries_.data(), entries_.size()),
            weights, scratch);
    }

    AliasView view() const noexcept {
        return AliasView(span<const AliasEntry>(entries_.data(), entries_.size()));
    }

    inline uint32_t sample(uint32_t bucket_idx, float prob_rng) const noexcept {
        return view().sample(bucket_idx, prob_rng);
    }

    std::size_t size()  const noexcept { return entries_.size(); }
    bool        empty() const noexcept { return entries_.empty(); }

    const std::vector<AliasEntry>& entries() const noexcept { return entries_; }

private:
    std::vector<AliasEntry> entries_;
};

}  // namespace tea
