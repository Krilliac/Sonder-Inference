// POSIX handle-anchored (openat + O_NOFOLLOW) regular-file opens, directory listing and
// symlink checks (src/common/safe_file.hpp). Shared by the weight manifests and the Ollama
// model store.
#include "common/safe_file.hpp"

#include <atomic>
#include <cerrno>
#include <climits>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <functional>
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

// Splits `rel` into segments; false when any segment is unsafe.
bool split_rel(std::string_view rel, std::vector<std::string>& out) {
    out.clear();
    std::size_t pos = 0;
    while (pos <= rel.size()) {
        std::size_t next = rel.find('/', pos);
        if (next == std::string_view::npos) next = rel.size();
        const std::string_view seg = rel.substr(pos, next - pos);
        if (unsafe_segment(seg)) return false;
        out.emplace_back(seg);
        pos = next + 1;
    }
    return true;
}

std::int64_t ticks(const struct timespec& ts) {
    return static_cast<std::int64_t>(ts.tv_sec) * 1000000000LL + static_cast<std::int64_t>(ts.tv_nsec);
}

std::atomic<bool> g_weak_identity_for_testing{false};  // set_weak_identity_for_testing
thread_local std::uint64_t t_dir_entries_visited = 0;    // dir_entries_visited_on_this_thread

void fill_identity(const struct stat& st, FileIdentity& out) {
    out.volume = static_cast<std::uint64_t>(st.st_dev);
    out.file_id = static_cast<std::uint64_t>(st.st_ino);
    out.file_id_high = 0;
    out.weak_id = g_weak_identity_for_testing.load(std::memory_order_relaxed);
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

// Opens segs[0, count) as directories relative to `anchor` without following links. On ok
// `owned` holds the last directory opened (invalid when count == 0) and `dirfd` is the fd to
// continue from (the anchor when count == 0; never closed here).
OpenOutcome open_dirs(int anchor, const std::vector<std::string>& segs, std::size_t count, OwnedFileHandle& owned,
                      int& dirfd, std::string& err_msg) {
    owned.reset();
    dirfd = anchor;
    for (std::size_t i = 0; i < count; ++i) {
        OwnedFileHandle child =
            make_owned(::openat(dirfd, segs[i].c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
        if (!child.valid()) {
            const int err = errno;
            if (err == ENOENT) return OpenOutcome::missing;
            if (err == ELOOP || err == ENOTDIR) return OpenOutcome::unsafe;
            err_msg = std::strerror(err);
            return OpenOutcome::io_error;
        }
        owned = std::move(child);
        dirfd = to_fd(owned.native_value());
    }
    return OpenOutcome::ok;
}

}  // namespace

void set_weak_identity_for_testing(bool on) noexcept { g_weak_identity_for_testing.store(on, std::memory_order_relaxed); }

std::uint64_t dir_entries_visited_on_this_thread() noexcept { return t_dir_entries_visited; }

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

std::intptr_t OwnedFileHandle::release() noexcept {
    const std::intptr_t raw = handle_;
    handle_ = -1;
    return raw;
}

bool is_link_like(const fs::path& p, const fs::file_status& st) {
    (void)p;
    return st.type() == fs::file_type::symlink;
}

OpenOutcome open_anchor_dir(const fs::path& dir, OwnedFileHandle& out_dir, FileIdentity& out_id, std::string& err_msg) {
    // Follows a link at `dir` (the operator chose it); the descriptor then pins what it reached.
    OwnedFileHandle h = make_owned(::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (!h.valid()) {
        const int err = errno;
        if (err == ENOENT) return OpenOutcome::missing;
        if (err == ENOTDIR) return OpenOutcome::unsafe;
        err_msg = std::strerror(err);
        return OpenOutcome::io_error;
    }
    struct stat st {};
    if (::fstat(to_fd(h.native_value()), &st) != 0) {
        err_msg = std::strerror(errno);
        return OpenOutcome::io_error;
    }
    if (!S_ISDIR(st.st_mode)) return OpenOutcome::unsafe;
    fill_identity(st, out_id);
    out_dir = std::move(h);
    return OpenOutcome::ok;
}

OpenOutcome open_anchored_file(const OwnedFileHandle& anchor, std::string_view rel, ShareMode mode,
                               OwnedFileHandle& out_handle, FileIdentity& out_id, std::string& err_msg, bool* busy) {
    (void)mode;  // POSIX has no mandatory deny-write share mode (safe_file.hpp ShareMode)
    if (busy != nullptr) *busy = false;
    if (!anchor.valid()) {
        err_msg = "store root is not open";
        return OpenOutcome::io_error;
    }
    std::vector<std::string> segs;
    if (!split_rel(rel, segs)) return OpenOutcome::unsafe;
    OwnedFileHandle dir;
    int dfd = -1;
    const OpenOutcome dirs = open_dirs(to_fd(anchor.native_value()), segs, segs.size() - 1, dir, dfd, err_msg);
    if (dirs != OpenOutcome::ok) return dirs;
    OwnedFileHandle leaf =
        make_owned(::openat(dfd, segs.back().c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK));
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
    OwnedFileHandle anchor;
    FileIdentity root_id;
    if (open_anchor_dir(root, anchor, root_id, err_msg) != OpenOutcome::ok) {
        if (err_msg.empty()) err_msg = "cannot open the directory";
        return OpenOutcome::io_error;
    }
    FileIdentity id;
    const OpenOutcome outcome = open_anchored_file(anchor, rel, ShareMode::allow_writers, out_handle, id, err_msg);
    if (outcome == OpenOutcome::ok) out_size = id.size;
    return outcome;
}

OpenOutcome probe_anchored_dir(const OwnedFileHandle& anchor, std::string_view rel, std::string& err_msg) {
    if (!anchor.valid()) {
        err_msg = "store root is not open";
        return OpenOutcome::io_error;
    }
    std::vector<std::string> segs;
    if (!split_rel(rel, segs)) return OpenOutcome::unsafe;
    OwnedFileHandle dir;
    int dfd = -1;
    return open_dirs(to_fd(anchor.native_value()), segs, segs.size(), dir, dfd, err_msg);
}

OpenOutcome visit_anchored_dir(const OwnedFileHandle& anchor, std::string_view rel, std::size_t max_entries,
                               const std::function<bool(DirEntry&)>& visit, bool& truncated, std::string& err_msg) {
    truncated = false;
    if (!anchor.valid()) {
        err_msg = "store root is not open";
        return OpenOutcome::io_error;
    }
    std::vector<std::string> segs;
    if (!split_rel(rel, segs)) return OpenOutcome::unsafe;
    OwnedFileHandle dir;
    int dfd = -1;
    const OpenOutcome opened = open_dirs(to_fd(anchor.native_value()), segs, segs.size(), dir, dfd, err_msg);
    if (opened != OpenOutcome::ok) return opened;
    // `dir` is a private open file description (opened above, never the anchor), so the
    // stream's position is not shared with any other thread. fdopendir owns it from here.
    DIR* d = ::fdopendir(to_fd(dir.native_value()));
    if (d == nullptr) {
        err_msg = std::strerror(errno);
        return OpenOutcome::io_error;
    }
    (void)dir.release();
    struct Closer {
        DIR* stream;
        ~Closer() { ::closedir(stream); }
    } closer{d};
    const int list_fd = dirfd(d);  // unqualified: a macro on some BSD libcs
    std::size_t visited = 0;
    while (true) {
        errno = 0;
        const struct dirent* ent = ::readdir(d);
        if (ent == nullptr) {
            if (errno != 0) {
                err_msg = std::strerror(errno);
                return OpenOutcome::io_error;
            }
            return OpenOutcome::ok;
        }
        const std::string_view name(ent->d_name);
        if (name == "." || name == "..") continue;
        if (visited == max_entries) {  // one entry more than the caller allows: stop reading
            truncated = true;
            return OpenOutcome::ok;
        }
        ++visited;
        ++t_dir_entries_visited;
        DirEntry e;
        e.name.assign(name);
        struct stat st {};
        if (::fstatat(list_fd, ent->d_name, &st, AT_SYMLINK_NOFOLLOW) != 0) {
            e.kind = DirEntryKind::other;  // vanished or unreadable: never followed
        } else if (S_ISLNK(st.st_mode)) {
            e.kind = DirEntryKind::link;
        } else if (S_ISDIR(st.st_mode)) {
            e.kind = DirEntryKind::directory;
        } else if (S_ISREG(st.st_mode)) {
            e.kind = DirEntryKind::file;
        } else {
            e.kind = DirEntryKind::other;
        }
        if (!visit(e)) return OpenOutcome::ok;
    }
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

bool read_anchored_leaf_symlink(const OwnedFileHandle& anchor, std::string_view rel, std::string& target,
                                FileIdentity& link_id, std::string& err_msg) {
    if (!anchor.valid()) {
        err_msg = "store root is not open";
        return false;
    }
    std::vector<std::string> segs;
    if (!split_rel(rel, segs)) {
        err_msg = "unsafe path segment";
        return false;
    }
    OwnedFileHandle dir;
    int dfd = -1;
    if (open_dirs(to_fd(anchor.native_value()), segs, segs.size() - 1, dir, dfd, err_msg) != OpenOutcome::ok) {
        return false;
    }
    const char* leaf = segs.back().c_str();
    struct stat lst {};
    if (::fstatat(dfd, leaf, &lst, AT_SYMLINK_NOFOLLOW) != 0) {
        err_msg = std::strerror(errno);
        return false;
    }
    if (!S_ISLNK(lst.st_mode)) {
        err_msg = "not a symbolic link";
        return false;
    }
    // readlinkat reads the link's own text; nothing it names is resolved, stat'ed or opened.
    char buf[PATH_MAX];
    const ssize_t n = ::readlinkat(dfd, leaf, buf, sizeof(buf));
    if (n < 0 || static_cast<std::size_t>(n) >= sizeof(buf)) {
        err_msg = n < 0 ? std::strerror(errno) : "link target too long";
        return false;
    }
    target.assign(buf, static_cast<std::size_t>(n));
    fill_identity(lst, link_id);
    return true;
}

}  // namespace sonder::inference::detail
