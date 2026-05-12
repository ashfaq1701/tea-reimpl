#pragma once

#include <cstdint>

namespace tea {

// PCG64 (Permuted Congruential Generator, 64-bit output). Cheap state, good
// statistical quality, fast on every modern CPU. Single 128-bit state, single
// 128-bit increment.
//
// Reference: https://www.pcg-random.org/
// We embed a minimal implementation rather than depending on the pcg-cpp
// library — the inner sample loop needs zero overhead and full inlining.
class Pcg64 {
public:
    using State = __uint128_t;

    Pcg64() noexcept = default;

    explicit Pcg64(uint64_t seed) noexcept {
        seed_with(seed, default_stream());
    }

    Pcg64(uint64_t seed, uint64_t stream) noexcept {
        seed_with(seed, stream);
    }

    void seed_with(uint64_t seed, uint64_t stream) noexcept {
        // Standard PCG seeding: initialize, advance once, mix the seed.
        inc_   = (State{stream} << 1u) | State{1};
        state_ = State{0};
        next_u64();
        state_ += State{seed};
        next_u64();
    }

    // Lemire-style multiplier for the 128-bit state's update.
    static constexpr State MULT = (State{6364136223846793005ULL} << 64) +
                                  State{1442695040888963407ULL};

    inline uint64_t next_u64() noexcept {
        State old   = state_;
        state_      = old * MULT + inc_;
        // PCG-XSL-RR output function for 128->64 reduction.
        uint64_t xorshifted = static_cast<uint64_t>(((old >> 64) ^ old));
        uint64_t rot        = static_cast<uint64_t>(old >> 122);
        return (xorshifted >> rot) | (xorshifted << ((-rot) & 63));
    }

    // Uniform [0, 1) as double. Uses 53 bits of mantissa.
    inline double next_u01() noexcept {
        constexpr double kInv = 1.0 / 9007199254740992.0;  // 1 / 2^53
        return static_cast<double>(next_u64() >> 11) * kInv;
    }

    // Uniform [0, 1) as float, for sampling against AliasTable's packed
    // 32-bit prob field. 24-bit mantissa.
    inline float next_f01() noexcept {
        constexpr float kInv = 1.0f / 16777216.0f;  // 1 / 2^24
        return static_cast<float>(next_u64() >> 40) * kInv;
    }

    // Uniform integer in [0, n). Lemire's rejection-free fast method.
    inline uint32_t next_below(uint32_t n) noexcept {
        __uint128_t m = static_cast<__uint128_t>(next_u64()) *
                        static_cast<__uint128_t>(n);
        return static_cast<uint32_t>(m >> 64);
    }

private:
    static constexpr uint64_t default_stream() noexcept {
        return 0xda3e39cb94b95bdbULL;
    }

    State state_{0};
    State inc_  {1};
};

// Cache-line size. We hard-code 64 — the value on every x86_64 and arm64
// CPU we care about. std::hardware_destructive_interference_size would be
// more portable in principle but GCC warns about its ABI variability
// (-Winterference-size); since this constant appears in alignas() of a type
// exposed by headers, ABI stability matters. 64 is correct and stable.
constexpr std::size_t kCacheLine = 64;

// One-per-thread RNG, padded to a full cache line so per-thread state
// updates never share a line with another thread's state (would cause
// false-sharing storms during parallel walks).
struct alignas(kCacheLine) PerThreadRng {
    Pcg64 rng;
    // tail-pad to a full cache line; the compiler may collapse this to zero
    // if Pcg64 already fills the line.
    char  _pad[kCacheLine - sizeof(Pcg64) % kCacheLine == kCacheLine
                  ? 0
                  : kCacheLine - sizeof(Pcg64) % kCacheLine]{};
};

static_assert(alignof(PerThreadRng) == kCacheLine,
              "PerThreadRng must be cache-line aligned");

}  // namespace tea
