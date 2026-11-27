// Internal: handle-anchored, no-follow file opens shared by the weight manifests
// (src/artifacts) and the Ollama model store (src/models).
//
// Every traversal starts from an open directory handle (an "anchor", open_anchor_dir) and
// opens one path segment at a time relative to the handle it opened last, never following a
// symlink, junction or other reparse point: Windows uses NtCreateFile with
// OBJECT_ATTRIBUTES::RootDirectory and FILE_OPEN_REPARSE_POINT (safe_file_win32.cpp), POSIX
// openat + O_NOFOLLOW (safe_file_posix.cpp). Retargeting or renaming the anchor's path after it
// was opened therefore cannot move later opens to another directory.
//
// Ownership: OwnedFileHandle owns one OS handle (HANDLE / fd) and is move-only. Every
// function here is reentrant and holds no global state. An anchor handle may be shared by
// any number of threads at once (it is only ever used as the starting directory of a
// relative open, never read through); a file handle is used by one thread at a time
// (reads advance its file position).
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

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
    // Gives up ownership: returns the raw value and leaves this object invalid.
    [[nodiscard]] std::intptr_t release() noexcept;

private:
    std::intptr_t handle_;
};

// Identity and metadata of an open file, read from the handle (never from a path).
// Times are platform ticks: 100 ns FILETIME units on Windows, nanoseconds since the
// epoch on POSIX. `change_time` is the metadata/content change time (Windows
// FILE_BASIC_INFO::ChangeTime, POSIX st_ctime). On POSIX only the kernel moves st_ctime
// (a plain user cannot set it back). On Windows it is NOT tamper-evident: anyone holding
// FILE_WRITE_ATTRIBUTES on the file can set ChangeTime (and LastWriteTime) to any value
// with SetFileInformationByHandle(FileBasicInfo), so equal times prove nothing against a
// writer who restores them. The file id is the 128-bit FILE_ID_INFO id on Windows (unique
// on ReFS, where the 64-bit BY_HANDLE_FILE_INFORMATION index is not).
struct FileIdentity {
    std::uint64_t volume = 0;        // Windows FILE_ID_INFO::VolumeSerialNumber; POSIX st_dev
    std::uint64_t file_id = 0;       // low 64 bits of the Windows FILE_ID_128; POSIX st_ino
    std::uint64_t file_id_high = 0;  // high 64 bits of the Windows FILE_ID_128; 0 on POSIX
    std::uint64_t size = 0;
    std::uint64_t link_count = 0;
    std::int64_t write_time = 0;
    std::int64_t change_time = 0;

    [[nodiscard]] bool same_file(const FileIdentity& o) const noexcept {
        return volume == o.volume && file_id == o.file_id && file_id_high == o.file_id_high;
    }
    friend bool operator==(const FileIdentity&, const FileIdentity&) = default;
};

// How other processes may use a regular file while the returned handle is open.
enum class ShareMode {
    // Others may read, write, rename and delete it (the default for every small read).
    allow_writers,
    // Windows: opened without FILE_SHARE_WRITE and FILE_SHARE_DELETE, so no other handle can
    // write, truncate, rename or delete it until this handle is closed; the open itself is
    // refused (io_error, *busy = true) while some other handle already has write or delete
    // access. A writable section mapped before that writer closed its handle is not covered.
    // POSIX has no mandatory equivalent: same as allow_writers.
    deny_writers,
};

// True for symlinks and (on Windows) junctions / other reparse points.
bool is_link_like(const std::filesystem::path& p, const std::filesystem::file_status& st);

// Opens `dir` as an anchor for relative opens. `dir` itself may be (or sit under) a symlink
// or junction: it is followed once, here, and the handle then pins the directory it reached.
// unsafe when the path is not a directory. `out_id` is the directory's identity.
OpenOutcome open_anchor_dir(const std::filesystem::path& dir, OwnedFileHandle& out_dir, FileIdentity& out_id,
                            std::string& err_msg);

// Opens `rel` (UTF-8, '/'-separated, no empty/'.'/'..' segments) relative to `anchor` one
// segment at a time without following symlinks or reparse points, and checks on the open
// handle that every directory is a real directory, the leaf is a regular disk file, and
// (Windows) each component's final path is exactly <anchor's current final path>\segment
// (which also refuses 8.3 aliases). A segment that could leave its directory lexically
// (empty, ".", "..", NUL, and on Windows ':' or '\\') is reported unsafe without opening it.
// With ShareMode::deny_writers and a refused share, returns io_error and sets *busy.
OpenOutcome open_anchored_file(const OwnedFileHandle& anchor, std::string_view rel, ShareMode mode,
                               OwnedFileHandle& out_handle, FileIdentity& out_id, std::string& err_msg,
                               bool* busy = nullptr);

// Path-anchored convenience for callers without a long-lived anchor (src/artifacts): opens
// `root` with open_anchor_dir, then open_anchored_file(allow_writers).
OpenOutcome open_anchored_regular(const std::filesystem::path& root, std::string_view rel,
                                  OwnedFileHandle& out_handle, std::uint64_t& out_size, std::string& err_msg);

// Checks that `rel` relative to `anchor` is a real directory (every segment opened as above).
OpenOutcome probe_anchored_dir(const OwnedFileHandle& anchor, std::string_view rel, std::string& err_msg);

enum class DirEntryKind { directory, file, link, other };

struct DirEntry {
    std::string name;      // UTF-8 (POSIX: the raw bytes)
    bool name_ok = true;   // false: the Windows name is not valid UTF-16 (unpaired surrogate)
    DirEntryKind kind = DirEntryKind::other;
};

// Lists the directory `rel` (non-empty) relative to `anchor` from a handle opened as above
// (Windows: GetFileInformationByHandleEx(FileIdBothDirectoryInfo) on that handle; POSIX:
// fdopendir on a private descriptor). Kinds come from the entry itself, never from following
// it: reparse points (Windows) and symlinks (POSIX, fstatat AT_SYMLINK_NOFOLLOW) are `link`.
// "." and ".." are omitted. On failure `out` is left empty.
OpenOutcome list_anchored_dir(const OwnedFileHandle& anchor, std::string_view rel, std::vector<DirEntry>& out,
                              std::string& err_msg);

// Opens a single regular file without following symlinks or reparse points.
OpenOutcome open_single_regular(const std::filesystem::path& file, OwnedFileHandle& out_handle,
                                std::uint64_t& out_size, std::string& err_msg);

// Fills `out` from the open handle. False (with err_msg) on failure.
bool query_identity(const OwnedFileHandle& handle, FileIdentity& out, std::string& err_msg);

// Reads up to `n` bytes at the handle's current position. Returns the byte count (0 at end
// of file), or -1 with err_msg set. Retries EINTR on POSIX; never reads more than `n`.
std::int64_t read_some(const OwnedFileHandle& handle, char* buf, std::size_t n, std::string& err_msg);

// POSIX only. When the last segment of `rel` relative to `anchor` is a symlink (every
// directory before it opened without following links, as above), stores the link's own text
// in `target` and the link's own identity (fstatat AT_SYMLINK_NOFOLLOW) in `link_id`. The
// target is never resolved, stat'ed or opened. Returns false when the leaf is not a symlink,
// a directory on the way is a link, the text is longer than PATH_MAX, or on Windows (where
// links are never followed).
bool read_anchored_leaf_symlink(const OwnedFileHandle& anchor, std::string_view rel, std::string& target,
                                FileIdentity& link_id, std::string& err_msg);

}  // namespace sonder::inference::detail
