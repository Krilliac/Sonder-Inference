// Sonder Inference: KV-cache module: shared identifiers and fingerprints.
#pragma once

#include <cstdint>
#include <string_view>

#include "sonder/inference/error.hpp"

namespace sonder::inference::cache {

/// Index of a logical KV block inside a KvCacheManager pool.
using BlockId = std::uint32_t;
/// Caller-assigned identifier of a sequence (request / session branch).
using SequenceId = std::uint64_t;
/// Token id as produced by the tokenizer.
using TokenId = std::int32_t;

inline constexpr BlockId kInvalidBlock = static_cast<BlockId>(-1);

/// Scheduling priority attached to a sequence. Higher value = more important.
/// Values are opaque to the cache; the scheduler maps work classes onto them
/// (see docs/SCHEDULER.md). Eviction policies may use them.
using Priority = std::uint8_t;

/// Compatibility fingerprint for prefix reuse (docs/KV_CACHE.md "Prefix lookup"):
/// callers fold model identity/revision, adapter set, tokenizer/config, rope
/// configuration, cache quantization/format and backend compatibility into it.
/// Two sequences may only share cached prefix blocks when fingerprints match.
struct CacheFingerprint {
    std::uint64_t value = 0;
    friend bool operator==(const CacheFingerprint&, const CacheFingerprint&) = default;
};

/// Builds a fingerprint from arbitrary components (FNV-1a based, order sensitive).
class FingerprintBuilder {
public:
    FingerprintBuilder& add(std::string_view component) noexcept;
    FingerprintBuilder& add(std::uint64_t component) noexcept;
    [[nodiscard]] CacheFingerprint build() const noexcept { return CacheFingerprint{state_}; }

private:
    std::uint64_t state_ = 0xcbf29ce484222325ULL;
};

/// Errors use the core sonder::inference::Status. Cache operations report:
///   ErrorCode::invalid_argument  bad arguments (e.g. truncate beyond length)
///   ErrorCode::not_found         unknown sequence id
///   ErrorCode::invalid_state     sequence id already exists
///   ErrorCode::unavailable       not enough free/evictable blocks (nothing changed)
// (ErrorCode and Status are visible here from the enclosing namespace.)

/// Memory pressure level derived from pinned (referenced) block utilisation.
enum class PressureLevel : std::uint8_t { normal, high, critical };

[[nodiscard]] const char* to_string(PressureLevel level) noexcept;

}  // namespace sonder::inference::cache
