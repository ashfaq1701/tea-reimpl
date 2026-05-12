#pragma once

#include <cstddef>
#include <cstdint>

#include "tea/edge.hpp"

namespace tea {

// Lightweight span<T>. We deliberately do NOT use std::span because that's
// C++20-only and we're on C++17. Implementing it ourselves keeps the build
// portable AND lets us add tea-specific helpers later (e.g., subspan that
// asserts in debug builds).
//
// API matches std::span where it matters (data(), size(), operator[],
// begin(), end()) so a future migration is mechanical.
template <class T>
class span {
public:
    span() noexcept = default;
    span(T* ptr, std::size_t n) noexcept : ptr_(ptr), n_(n) {}

    T*          data()  const noexcept { return ptr_; }
    std::size_t size()  const noexcept { return n_; }
    bool        empty() const noexcept { return n_ == 0; }

    T&  operator[](std::size_t i) const noexcept { return ptr_[i]; }
    T*  begin() const noexcept { return ptr_; }
    T*  end()   const noexcept { return ptr_ + n_; }

    span<T> subspan(std::size_t off, std::size_t count) const noexcept {
        return span<T>(ptr_ + off, count);
    }

private:
    T*          ptr_{nullptr};
    std::size_t n_  {0};
};

// A walk is a span over a thread-local pre-allocated NodeStep buffer
// owned by the WalkEngine. Walk APIs return Walk; the buffer they point
// into outlives the call. No heap allocation per walk.
using Walk = span<const NodeStep>;

}  // namespace tea
