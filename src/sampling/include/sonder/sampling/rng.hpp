#pragma once

#include <array>
#include <cstdint>

namespace sonder::inference::sampling {

/// Portable, seedable pseudo-random generator (xoshiro256** seeded through
/// SplitMix64). Unlike std::mt19937 + std::*_distribution, the output of
/// uniform() is bit-identical on every compiler/standard library, which is
/// what makes fixed-seed sampling reproducible across MSVC, libstdc++ and
/// libc++. Algorithms are public-domain designs by Blackman & Vigna,
/// implemented independently here.
class Rng {
public:
    explicit Rng(std::uint64_t seed = 0) noexcept { reseed(seed); }

    void reseed(std::uint64_t seed) noexcept;

    /// Next raw 64-bit value.
    std::uint64_t next_u64() noexcept;

    /// Uniform double in [0, 1) with 53 bits of precision.
    double uniform() noexcept;

    /// Seed from a non-deterministic source (std::random_device + clock).
    static std::uint64_t entropy_seed();

private:
    std::array<std::uint64_t, 4> s_{};
};

}  // namespace sonder::inference::sampling
