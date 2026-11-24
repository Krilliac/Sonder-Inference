// Internal: platform-specific handle-anchored OS file helpers for weight
// manifests. Windows implements this in weight_manifest_io_win32.cpp; POSIX
// implements it in weight_manifest_io_posix.cpp.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "sonder/inference/weight_manifest.hpp"

namespace sonder::inference::weights::detail {

inline constexpr std::size_t kMinBuffer = 4096;
inline constexpr std::size_t kMaxBuffer = 64u << 20;

enum class OpenOutcome { ok, missing, unsafe, io_error };

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

// Move-only RAII wrapper around an OS file handle (HANDLE on Win32, fd on
// POSIX). Implemented in the active platform's TU so cross-platform code never
// includes OS headers.
class OwnedFileHandle {
public:
    OwnedFileHandle() noexcept;
    ~OwnedFileHandle();
    OwnedFileHandle(const OwnedFileHandle&) = delete;
    OwnedFileHandle& operator=(const OwnedFileHandle&) = delete;
    OwnedFileHandle(OwnedFileHandle&& other) noexcept;
    OwnedFileHandle& operator=(OwnedFileHandle&& other) noexcept;

    void reset() noexcept;
    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] std::intptr_t native_value() const noexcept { return handle_; }
    void assign_native(std::intptr_t raw) noexcept;

private:
    std::intptr_t handle_;
};

// Returns true for symlinks and (on Windows) junctions / reparse points.
bool is_link_like(const std::filesystem::path& p, const std::filesystem::file_status& st);

// Opens `rel` under `root` one segment at a time without following symlinks or
// reparse points, verifying containment and regular-file identity on the open
// handle itself.
OpenOutcome open_anchored_regular(const std::filesystem::path& root, std::string_view rel,
                                  OwnedFileHandle& out_handle, std::uint64_t& out_size, std::string& err_msg);

// Opens a single regular file without following symlinks or reparse points.
OpenOutcome open_single_regular(const std::filesystem::path& file, OwnedFileHandle& out_handle,
                                std::uint64_t& out_size, std::string& err_msg);

// Streams the already-open handle through SHA-256 and verifies size/type did
// not change on the handle while hashing.
Result<std::string> hash_open_handle(const OwnedFileHandle& handle, std::uint64_t initial_size,
                                     std::string_view display, const HashOptions& options,
                                     std::uint64_t* bytes_read);

}  // namespace sonder::inference::weights::detail
