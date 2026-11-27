// Windows (Win32) streaming SHA-256 hashing over an anchored handle for weight manifests.
// The anchored opens themselves are in src/common/safe_file_win32.cpp.
#include "weight_manifest_io.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include "server/src/sha256.hpp"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace sonder::inference::weights::detail {

namespace {

HANDLE to_handle(std::intptr_t raw) noexcept { return reinterpret_cast<HANDLE>(raw); }

std::string win32_err_string(DWORD err) { return "win32 error " + std::to_string(err); }

}  // namespace

Result<std::string> hash_open_handle(const OwnedFileHandle& handle, std::uint64_t initial_size,
                                     std::string_view display, const HashOptions& options, std::uint64_t* bytes_read) {
    const HANDLE raw = to_handle(handle.native_value());
    const std::size_t cap = std::clamp(options.buffer_bytes, kMinBuffer, kMaxBuffer);
    std::string buffer(cap, '\0');
    server::detail::Sha256 sha;
    std::uint64_t done = 0;
    while (true) {
        if (options.cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled while hashing '" + printable_path(display) + "'");
        }
        std::size_t filled = 0;
        bool eof = false;
        while (filled < cap) {
            DWORD got = 0;
            const DWORD want = static_cast<DWORD>(std::min<std::size_t>(cap - filled, 0x40000000u));
            if (!::ReadFile(raw, buffer.data() + filled, want, &got, nullptr)) {
                return Status(ErrorCode::io_error,
                              "read error on '" + printable_path(display) + "': " + win32_err_string(::GetLastError()));
            }
            if (got == 0) {
                eof = true;
                break;
            }
            filled += static_cast<std::size_t>(got);
        }
        if (filled > 0) {
            sha.update(std::string_view(buffer.data(), filled));
            done += filled;
            if (options.progress) options.progress(display, done, initial_size);
        }
        if (eof) break;
    }
    BY_HANDLE_FILE_INFORMATION after{};
    if (!::GetFileInformationByHandle(raw, &after)) {
        return Status(ErrorCode::io_error, "cannot verify size after hashing '" + printable_path(display) + "'");
    }
    const std::uint64_t after_size =
        (static_cast<std::uint64_t>(after.nFileSizeHigh) << 32) | static_cast<std::uint64_t>(after.nFileSizeLow);
    if ((after.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 || after_size != done) {
        return Status(ErrorCode::io_error, "'" + printable_path(display) + "' changed while it was being hashed");
    }
    if (bytes_read) *bytes_read = done;
    return to_hex(sha.finish());
}

}  // namespace sonder::inference::weights::detail
