// Sonder Inference: KV-cache module: fingerprint helpers.
#include "sonder/inference/cache/types.hpp"

namespace sonder::inference::cache {
namespace {
constexpr std::uint64_t kFnvPrime = 0x100000001b3ULL;
}

FingerprintBuilder& FingerprintBuilder::add(std::string_view component) noexcept {
    for (const char c : component) {
        state_ ^= static_cast<unsigned char>(c);
        state_ *= kFnvPrime;
    }
    // Component separator so ("ab","c") != ("a","bc").
    state_ ^= 0xffU;
    state_ *= kFnvPrime;
    return *this;
}

FingerprintBuilder& FingerprintBuilder::add(std::uint64_t component) noexcept {
    for (int i = 0; i < 8; ++i) {
        state_ ^= (component >> (i * 8)) & 0xffU;
        state_ *= kFnvPrime;
    }
    state_ ^= 0xfeU;
    state_ *= kFnvPrime;
    return *this;
}

const char* to_string(PressureLevel level) noexcept {
    switch (level) {
        case PressureLevel::normal: return "normal";
        case PressureLevel::high: return "high";
        case PressureLevel::critical: return "critical";
    }
    return "unknown";
}

}  // namespace sonder::inference::cache
