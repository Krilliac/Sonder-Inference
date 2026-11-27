// Internal: handle-anchored, no-follow file opens shared by the weight manifests
// (src/artifacts) and the Ollama model store (src/models). Windows implements this in
// safe_file_win32.cpp (FILE_FLAG_OPEN_REPARSE_POINT on every component, final-path
// containment under the opened root); POSIX in safe_file_posix.cpp (openat + O_NOFOLLOW
// one component at a time).
//
// Ownership: OwnedFileHandle owns one OS handle (HANDLE / fd) and is move-only. Every
// function here is reentrant and holds no global state; a handle may be used from any
// thread, one call at a time (reads advance the shared file position).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace sonder::inference::detail {

enum class OpenOutcome { ok, missing, unsafe, io_error };

// Move-only RAII wrapper around an OS file handle (HANDLE on Win32, fd on POSIX).
// Implemented in the active platform's TU so cross-platform code never includes OS headers.
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

// Identity and metadata of an open file, read from the handle (never from a path).
// Times are platform ticks: 100 ns FILETIME units on Windows, nanoseconds since the
// epoch on POSIX. `change_time` is the metadata/content change time (Windows
// FILE_BASIC_INFO::ChangeTime, POSIX st_ctime), which a plain user cannot set back.
struct FileIdentity {
    std::uint64_t volume = 0;   // Windows volume serial number; POSIX st_dev
    std::uint64_t file_id = 0;  // Windows file index (64-bit); POSIX st_ino
    std::uint64_t size = 0;
    std::uint64_t link_count = 0;
    std::int64_t write_time = 0;
    std::int64_t change_time = 0;

    [[nodiscard]] bool same_file(const FileIdentity& o) const noexcept {
        return volume == o.volume && file_id == o.file_id;
    }
    friend bool operator==(const FileIdentity&, const FileIdentity&) = default;
};

// True for symlinks and (on Windows) junctions / other reparse points.
bool is_link_like(const std::filesystem::path& p, const std::filesystem::file_status& st);

// Opens `rel` (UTF-8, '/'-separated, no empty/'.'/'..' segments) under `root` one segment
// at a time without following symlinks or reparse points, and checks on the open handle
// that every directory is a real directory, the leaf is a regular disk file, and (Windows)
// each component's final path is exactly root/segment. `root` itself may be a link: its
// opened final path is the anchor. A segment that could leave the root lexically
// (empty, ".", "..", and on Windows ':' or '\\') is reported unsafe without opening it.
OpenOutcome open_anchored_regular(const std::filesystem::path& root, std::string_view rel,
                                  OwnedFileHandle& out_handle, std::uint64_t& out_size, std::string& err_msg);

// Same as open_anchored_regular, and also returns the leaf's identity from the handle.
OpenOutcome open_anchored_file(const std::filesystem::path& root, std::string_view rel,
                               OwnedFileHandle& out_handle, FileIdentity& out_id, std::string& err_msg);

// Opens a single regular file without following symlinks or reparse points.
OpenOutcome open_single_regular(const std::filesystem::path& file, OwnedFileHandle& out_handle,
                                std::uint64_t& out_size, std::string& err_msg);

// Fills `out` from the open handle. False (with err_msg) on failure.
bool query_identity(const OwnedFileHandle& handle, FileIdentity& out, std::string& err_msg);

// Reads up to `n` bytes at the handle's current position. Returns the byte count (0 at end
// of file), or -1 with err_msg set. Retries EINTR on POSIX; never reads more than `n`.
std::int64_t read_some(const OwnedFileHandle& handle, char* buf, std::size_t n, std::string& err_msg);

// POSIX only. When the last segment of `rel` under `root` is a symlink (every directory
// before it opened without following links, as above), stores the link's own text in
// `target`, the link's own identity (lstat) in `link_id`, and the identity of the file it
// resolves to (stat, which follows it but opens nothing) in `target_id`. Nothing is read
// through the link. Returns false when the leaf is not a symlink, a directory on the way is
// a link, the link does not resolve, or on Windows (where links are never followed).
bool read_anchored_leaf_symlink(const std::filesystem::path& root, std::string_view rel, std::string& target,
                                FileIdentity& link_id, FileIdentity& target_id, std::string& err_msg);

}  // namespace sonder::inference::detail
