// Internal: platform-specific streaming SHA-256 over an anchored handle for weight
// manifests. Windows implements this in weight_manifest_io_win32.cpp; POSIX
// implements it in weight_manifest_io_posix.cpp.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "common/safe_file.hpp"
#include "sonder/inference/weight_manifest.hpp"

namespace sonder::inference::weights::detail {

inline constexpr std::size_t kMinBuffer = 4096;
inline constexpr std::size_t kMaxBuffer = 64u << 20;

// The handle type and the anchored opens live in src/common/safe_file.hpp (shared with the
// Ollama model store); these names keep the artifact code unchanged.
using sonder::inference::detail::is_link_like;
using sonder::inference::detail::open_anchored_regular;
using sonder::inference::detail::open_single_regular;
using sonder::inference::detail::OpenOutcome;
using sonder::inference::detail::OwnedFileHandle;

inline std::string to_hex(const std::array<std::uint8_t, 32>& d) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint8_t b : d) {
        out.push_back(kDigits[b >> 4]);
        out.push_back(kDigits[b & 15]);
    }
    return out;
}

// Streams the already-open handle through SHA-256 and verifies size/type did
// not change on the handle while hashing.
Result<std::string> hash_open_handle(const OwnedFileHandle& handle, std::uint64_t initial_size,
                                     std::string_view display, const HashOptions& options,
                                     std::uint64_t* bytes_read);

}  // namespace sonder::inference::weights::detail
