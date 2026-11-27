// POSIX handle-anchored (openat + O_NOFOLLOW) regular-file open and symlink checks
// (src/common/safe_file.hpp). Moved here from src/artifacts/src/weight_manifest_io_posix.cpp
// so the weight manifests and the Ollama model store share one implementation.
#include "common/safe_file.hpp"

#include <cerrno>
#include <climits>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace sonder::inference::detail {

namespace fs = std::filesystem;

namespace {

int to_fd(std::intptr_t raw) noexcept { return static_cast<int>(raw); }

OwnedFileHandle make_owned(int fd) noexcept {
    OwnedFileHandle out;
    out.assign_native(static_cast<std::intptr_t>(fd));
    return out;
}

bool unsafe_segment(std::string_view seg) {
    if (seg.empty() || seg == "." || seg == "..") return true;
    return seg.find('\0') != std::string_view::npos;
}

std::int64_t ticks(const struct timespec& ts) {
    return static_cast<std::int64_t>(ts.tv_sec) * 1000000000LL + static_cast<std::int64_t>(ts.tv_nsec);
}

void fill_identity(const struct stat& st, FileIdentity& out) {
    out.volume = static_cast<std::uint64_t>(st.st_dev);
    out.file_id = static_cast<std::uint64_t>(st.st_ino);
    out.size = st.st_size < 0 ? 0u : static_cast<std::uint64_t>(st.st_size);
    out.link_count = static_cast<std::uint64_t>(st.st_nlink);
#if defined(__APPLE__)
    out.write_time = ticks(st.st_mtimespec);
    out.change_time = ticks(st.st_ctimespec);
#else
    out.write_time = ticks(st.st_mtim);
    out.change_time = ticks(st.st_ctim);
#endif
}

// Opens every directory of `rel` but the last segment under `root` without following links.
// On success `dir` is the leaf's parent and `leaf` its name.
OpenOutcome open_parent_dirs(const fs::path& root, std::string_view rel, OwnedFileHandle& dir, std::string& leaf,
                             std::string& err_msg) {
    OwnedFileHandle cur = make_owned(::open(root.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (!cur.valid()) {
        err_msg = std::strerror(errno);
        return OpenOutcome::io_error;
    }
    std::size_t pos = 0;
    while (pos <= rel.size()) {
        std::size_t next = rel.find('/', pos);
        if (next == std::string_view::npos) next = rel.size();
        const std::string_view seg_text = rel.substr(pos, next - pos);
        if (unsafe_segment(seg_text)) return OpenOutcome::unsafe;
        const std::string seg(seg_text);
        if (next >= rel.size()) {
            leaf = seg;
            dir = std::move(cur);
            return OpenOutcome::ok;
        }
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
    }
    return OpenOutcome::missing;
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

OpenOutcome open_anchored_file(const fs::path& root, std::string_view rel, OwnedFileHandle& out_handle,
                               FileIdentity& out_id, std::string& err_msg) {
    OwnedFileHandle dir;
    std::string seg;
    const OpenOutcome parents = open_parent_dirs(root, rel, dir, seg, err_msg);
    if (parents != OpenOutcome::ok) return parents;
    OwnedFileHandle leaf =
        make_owned(::openat(to_fd(dir.native_value()), seg.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
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
    fill_identity(st, out_id);
    out_handle = std::move(leaf);
    return OpenOutcome::ok;
}

OpenOutcome open_anchored_regular(const fs::path& root, std::string_view rel, OwnedFileHandle& out_handle,
                                  std::uint64_t& out_size, std::string& err_msg) {
    FileIdentity id;
    const OpenOutcome outcome = open_anchored_file(root, rel, out_handle, id, err_msg);
    if (outcome == OpenOutcome::ok) out_size = id.size;
    return outcome;
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

bool query_identity(const OwnedFileHandle& handle, FileIdentity& out, std::string& err_msg) {
    struct stat st {};
    if (::fstat(to_fd(handle.native_value()), &st) != 0) {
        err_msg = std::strerror(errno);
        return false;
    }
    fill_identity(st, out);
    return true;
}

std::int64_t read_some(const OwnedFileHandle& handle, char* buf, std::size_t n, std::string& err_msg) {
    const int fd = to_fd(handle.native_value());
    while (true) {
        const ssize_t got = ::read(fd, buf, n);
        if (got >= 0) return static_cast<std::int64_t>(got);
        if (errno == EINTR) continue;
        err_msg = std::strerror(errno);
        return -1;
    }
}

bool read_anchored_leaf_symlink(const fs::path& root, std::string_view rel, std::string& target,
                                FileIdentity& link_id, FileIdentity& target_id, std::string& err_msg) {
    OwnedFileHandle dir;
    std::string seg;
    if (open_parent_dirs(root, rel, dir, seg, err_msg) != OpenOutcome::ok) return false;
    const int dfd = to_fd(dir.native_value());
    struct stat lst {};
    if (::fstatat(dfd, seg.c_str(), &lst, AT_SYMLINK_NOFOLLOW) != 0) {
        err_msg = std::strerror(errno);
        return false;
    }
    if (!S_ISLNK(lst.st_mode)) {
        err_msg = "not a symbolic link";
        return false;
    }
    char buf[PATH_MAX];
    const ssize_t n = ::readlinkat(dfd, seg.c_str(), buf, sizeof(buf));
    if (n < 0 || static_cast<std::size_t>(n) >= sizeof(buf)) {
        err_msg = n < 0 ? std::strerror(errno) : "link target too long";
        return false;
    }
    struct stat tst {};
    if (::fstatat(dfd, seg.c_str(), &tst, 0) != 0) {
        err_msg = std::strerror(errno);
        return false;
    }
    target.assign(buf, static_cast<std::size_t>(n));
    fill_identity(lst, link_id);
    fill_identity(tst, target_id);
    return true;
}

}  // namespace sonder::inference::detail
