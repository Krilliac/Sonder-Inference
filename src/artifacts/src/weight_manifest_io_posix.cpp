// POSIX handle-anchored (openat + O_NOFOLLOW) regular-file open, symlink check,
// and streaming SHA-256 hashing for weight manifests.
#include "weight_manifest_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

#include "server/src/sha256.hpp"

namespace sonder::inference::weights::detail {

namespace fs = std::filesystem;

namespace {

int to_fd(std::intptr_t raw) noexcept { return static_cast<int>(raw); }

OwnedFileHandle make_owned(int fd) noexcept {
    OwnedFileHandle out;
    out.assign_native(static_cast<std::intptr_t>(fd));
    return out;
}

}  // namespace

OwnedFileHandle::OwnedFileHandle() noexcept : handle_(-1) {}

OwnedFileHandle::~OwnedFileHandle() { reset(); }

OwnedFileHandle::OwnedFileHandle(OwnedFileHandle&& other) noexcept : handle_(other.handle_) { other.handle_ = -1; }

OwnedFileHandle& OwnedFileHandle::operator=(OwnedFileHandle&& other) noexcept {
    if (this != &other) {
        reset();
        handle_ = other.handle_;
        other.handle_ = -1;
    }
    return *this;
}

void OwnedFileHandle::reset() noexcept {
    const int fd = to_fd(handle_);
    if (fd >= 0) {
        ::close(fd);
        handle_ = -1;
    }
}

bool OwnedFileHandle::valid() const noexcept { return to_fd(handle_) >= 0; }

void OwnedFileHandle::assign_native(std::intptr_t raw) noexcept {
    reset();
    handle_ = raw;
}

bool is_link_like(const fs::path& p, const fs::file_status& st) {
    (void)p;
    return st.type() == fs::file_type::symlink;
}

OpenOutcome open_anchored_regular(const fs::path& root, std::string_view rel, OwnedFileHandle& out_handle,
                                  std::uint64_t& out_size, std::string& err_msg) {
    OwnedFileHandle cur = make_owned(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (!cur.valid()) {
        err_msg = std::strerror(errno);
        return OpenOutcome::io_error;
    }
    std::size_t pos = 0;
    while (pos <= rel.size()) {
        std::size_t next = rel.find('/', pos);
        if (next == std::string_view::npos) next = rel.size();
        const std::string seg(rel.substr(pos, next - pos));
        const bool last = next >= rel.size();

        if (!last) {
            OwnedFileHandle child =
                make_owned(::openat(to_fd(cur.native_value()), seg.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
            if (!child.valid()) {
                const int err = errno;
                if (err == ENOENT) return OpenOutcome::missing;
                if (err == ELOOP || err == ENOTDIR) return OpenOutcome::unsafe;
                err_msg = std::strerror(err);
                return OpenOutcome::io_error;
            }
            cur = std::move(child);
            pos = next + 1;
            continue;
        }

        OwnedFileHandle leaf = make_owned(
            ::openat(to_fd(cur.native_value()), seg.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
        if (!leaf.valid()) {
            const int err = errno;
            if (err == ENOENT) return OpenOutcome::missing;
            if (err == ELOOP || err == EISDIR || err == ENXIO) return OpenOutcome::unsafe;
            err_msg = std::strerror(err);
            return OpenOutcome::io_error;
        }
        struct stat st {};
        if (::fstat(to_fd(leaf.native_value()), &st) != 0) {
            err_msg = std::strerror(errno);
            return OpenOutcome::io_error;
        }
        if (!S_ISREG(st.st_mode)) {
            return OpenOutcome::unsafe;
        }
        out_size = st.st_size < 0 ? 0u : static_cast<std::uint64_t>(st.st_size);
        out_handle = std::move(leaf);
        return OpenOutcome::ok;
    }
    return OpenOutcome::missing;
}

OpenOutcome open_single_regular(const fs::path& file, OwnedFileHandle& out_handle, std::uint64_t& out_size,
                                std::string& err_msg) {
    OwnedFileHandle h = make_owned(::open(file.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
    if (!h.valid()) {
        const int err = errno;
        if (err == ENOENT) {
            err_msg = std::strerror(err);
            return OpenOutcome::missing;
        }
        if (err == ELOOP || err == EISDIR || err == ENXIO) {
            return OpenOutcome::unsafe;
        }
        err_msg = std::strerror(err);
        return OpenOutcome::io_error;
    }
    struct stat st {};
    if (::fstat(to_fd(h.native_value()), &st) != 0) {
        err_msg = std::strerror(errno);
        return OpenOutcome::io_error;
    }
    if (!S_ISREG(st.st_mode)) {
        return OpenOutcome::unsafe;
    }
    out_size = st.st_size < 0 ? 0u : static_cast<std::uint64_t>(st.st_size);
    out_handle = std::move(h);
    return OpenOutcome::ok;
}

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
