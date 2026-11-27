// POSIX streaming SHA-256 hashing over an anchored handle for weight manifests.
// The anchored opens themselves (openat + O_NOFOLLOW) are in src/common/safe_file_posix.cpp.
#include "weight_manifest_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "server/src/sha256.hpp"

namespace sonder::inference::weights::detail {

namespace {

int to_fd(std::intptr_t raw) noexcept { return static_cast<int>(raw); }

}  // namespace

Result<std::string> hash_open_handle(const OwnedFileHandle& handle, std::uint64_t initial_size,
                                     std::string_view display, const HashOptions& options, std::uint64_t* bytes_read) {
    const int fd = to_fd(handle.native_value());
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
            const ssize_t got = ::read(fd, buffer.data() + filled, cap - filled);
            if (got < 0) {
                if (errno == EINTR) continue;
                return Status(ErrorCode::io_error,
                              "read error on '" + printable_path(display) + "': " + std::strerror(errno));
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
    struct stat after {};
    if (::fstat(fd, &after) != 0 || !S_ISREG(after.st_mode) || after.st_size < 0 ||
        static_cast<std::uint64_t>(after.st_size) != done) {
        return Status(ErrorCode::io_error, "'" + printable_path(display) + "' changed while it was being hashed");
    }
    if (bytes_read) *bytes_read = done;
    return to_hex(sha.finish());
}

}  // namespace sonder::inference::weights::detail
