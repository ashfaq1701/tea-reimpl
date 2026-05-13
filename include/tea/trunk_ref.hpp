// TrunkRef + decompose_to_trunks — shared between HPAT and AuxIndex.
//
// Lives in its own header so both hpat.hpp and aux_index.hpp can include it
// without a circular dependency. AuxIndex stores precomputed TrunkRef arrays;
// HPAT's on-the-fly fallback calls decompose_to_trunks directly.

#pragma once

#include <cstdint>

namespace tea {

// A trunk in HPAT: level k (trunk size = 2^k), index i within that level
// (i-th trunk of size 2^k, covering edges [i·2^k, (i+1)·2^k) of u's
// time-DESC edge list).
struct TrunkRef {
    int32_t level;
    int32_t index;
};

// Binary-decompose L into a cover of HPAT trunks of u.  k_max = K_u =
// ⌊log₂ D_u⌋, the highest level used for vertex u.  Each set bit of L
// picks one trunk at that level; the running position advances by 2^k.
//
// Output is written into out[0..count), MSB-first.
// Returns the number of trunks in the cover (= popcount(L) ≤ k_max + 1).
//
// Precondition: L ≤ D_u; k_max ≥ ⌊log₂ L⌋ when L > 0.
inline int32_t decompose_to_trunks(int64_t L,
                                   int32_t k_max,
                                   TrunkRef* out) noexcept {
    int32_t pos   = 0;
    int32_t count = 0;
    for (int32_t k = k_max; k >= 0; --k) {
        if (L & (int64_t{1} << k)) {
            out[count++] = TrunkRef{k, pos >> k};
            pos += (1 << k);
        }
    }
    return count;
}

}  // namespace tea
