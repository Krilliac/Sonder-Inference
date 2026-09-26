#include "sonder/sampling/rng.hpp"

#include <chrono>
#include <random>

namespace sonder::inference::sampling {

namespace {
std::uint64_t splitmix64(std::uint64_t& x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = x;
    z = (z ^ (z >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27U)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31U);
}

constexpr std::uint64_t rotl(std::uint64_t x, unsigned k) noexcept {
    return (x << k) | (x >> (64U - k));
}
}  // namespace

void Rng::reseed(std::uint64_t seed) noexcept {
    std::uint64_t x = seed;
    for (auto& word : s_) {
        word = splitmix64(x);
    }
}

std::uint64_t Rng::next_u64() noexcept {
    const std::uint64_t result = rotl(s_[1] * 5U, 7U) * 9U;
    const std::uint64_t t = s_[1] << 17U;
    s_[2] ^= s_[0];
    s_[3] ^= s_[1];
    s_[1] ^= s_[2];
    s_[0] ^= s_[3];
    s_[2] ^= t;
    s_[3] = rotl(s_[3], 45U);
    return result;
}

double Rng::uniform() noexcept {
    // Top 53 bits -> [0, 1). Exact and portable.
    return static_cast<double>(next_u64() >> 11U) * 0x1.0p-53;
}

std::uint64_t Rng::entropy_seed() {
    std::random_device rd;
    const auto hi = static_cast<std::uint64_t>(rd());
    const auto lo = static_cast<std::uint64_t>(rd());
    const auto clock = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    return (hi << 32U) ^ lo ^ clock;
}

}  // namespace sonder::inference::sampling
