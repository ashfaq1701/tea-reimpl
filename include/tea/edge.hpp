#pragma once

#include <cstdint>

namespace tea {

// 16-byte packed temporal edge (u, v, t). Used only at ingestion time —
// the runtime data structures store SoA, not AoS, so this struct does not
// appear in the hot path.
struct Edge {
    int32_t u;
    int32_t v;
    int64_t t;
};

static_assert(sizeof(Edge) == 16, "Edge must be 16 bytes for cache friendliness");

// One step of a walk. 12 bytes packed; we pad walks to a 16-byte stride at
// the buffer level in WalkEngine so SIMD-aligned writes are cheap.
struct NodeStep {
    int32_t v;
    int64_t t;
};

static_assert(sizeof(NodeStep) == 16,
              "NodeStep should be padded to 16 bytes for vectorized writes");

}  // namespace tea
