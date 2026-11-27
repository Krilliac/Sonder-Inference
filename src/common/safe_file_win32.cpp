// Windows (Win32 + NT) handle-anchored regular-file opens, directory listing and reparse-point
// checks (src/common/safe_file.hpp). Shared by the weight manifests and the Ollama model store.
//
// Relative opens use NtCreateFile with OBJECT_ATTRIBUTES::RootDirectory (winternl.h, ntdll.lib):
// Win32 has no "open relative to a directory handle". Each name passed to it is a single
// component that was already refused when it holds '\\', ':', NUL or is "."/"..", so the NT
// parser cannot leave the directory handle it starts from; FILE_OPEN_REPARSE_POINT opens a
// reparse point as itself, and the attribute check then refuses it. Directories are opened
// without FILE_DIRECTORY_FILE: Microsoft's NtCreateFile documentation does not list
// FILE_OPEN_REPARSE_POINT among the options compatible with it, so "is a directory" and "is not
// a reparse point" are checked on the attributes of the handle that was actually opened.
#include "common/safe_file.hpp"

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstring>
#include <functional>
#include <string>
#include <utility>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winternl.h>

namespace sonder::inference::detail {

namespace fs = std::filesystem;

namespace {

// ntstatus.h values (shared/ntstatus.h); that header conflicts with windows.h, so the few
// codes mapped here are spelled out.
constexpr NTSTATUS kStatusNoSuchFile = static_cast<NTSTATUS>(0xC000000FUL);
constexpr NTSTATUS kStatusObjectNameInvalid = static_cast<NTSTATUS>(0xC0000033UL);
constexpr NTSTATUS kStatusObjectNameNotFound = static_cast<NTSTATUS>(0xC0000034UL);
constexpr NTSTATUS kStatusObjectPathNotFound = static_cast<NTSTATUS>(0xC000003AUL);
constexpr NTSTATUS kStatusObjectPathSyntaxBad = static_cast<NTSTATUS>(0xC000003BUL);
constexpr NTSTATUS kStatusSharingViolation = static_cast<NTSTATUS>(0xC0000043UL);
constexpr NTSTATUS kStatusDeletePending = static_cast<NTSTATUS>(0xC0000056UL);
constexpr NTSTATUS kStatusFileIsADirectory = static_cast<NTSTATUS>(0xC00000BAUL);
constexpr NTSTATUS kStatusNotADirectory = static_cast<NTSTATUS>(0xC0000103UL);
constexpr NTSTATUS kStatusReparsePointEncountered = static_cast<NTSTATUS>(0xC000050BUL);

constexpr ULONG kShareAll = FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE;
constexpr std::size_t kListBufferBytes = 64u * 1024u;

std::intptr_t to_raw(HANDLE h) noexcept { return reinterpret_cast<std::intptr_t>(h); }
HANDLE to_handle(std::intptr_t raw) noexcept { return reinterpret_cast<HANDLE>(raw); }

std::string win32_err_string(DWORD err) { return "win32 error " + std::to_string(err); }

std::string nt_err_string(NTSTATUS st) { return win32_err_string(::RtlNtStatusToDosError(st)); }

OwnedFileHandle make_owned(HANDLE h) noexcept {
    OwnedFileHandle out;
    out.assign_native(to_raw(h));
    return out;
}

bool utf8_to_wide(std::string_view s, std::wstring& out) {
    out.clear();
    if (s.empty()) return true;
    if (s.size() > 0x7fffffffu) return false;
    const int n = ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return false;
    out.assign(static_cast<std::size_t>(n), L'\0');
    return ::MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), static_cast<int>(s.size()), out.data(), n) == n;
}

bool wide_to_utf8(const std::wstring& w, std::string& out) {
    out.clear();
    if (w.empty()) return true;
    if (w.size() > 0x7fffffffu) return false;
    const int n = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w.data(), static_cast<int>(w.size()), nullptr, 0,
                                        nullptr, nullptr);
    if (n <= 0) return false;
    out.assign(static_cast<std::size_t>(n), '\0');
    return ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, w.data(), static_cast<int>(w.size()), out.data(), n,
                                 nullptr, nullptr) == n;
}

bool get_final_nt_path(HANDLE h, std::wstring& out) {
    DWORD needed = ::GetFinalPathNameByHandleW(h, nullptr, 0, VOLUME_NAME_NT | FILE_NAME_NORMALIZED);
    if (needed == 0) return false;
    std::wstring buf(static_cast<std::size_t>(needed), L'\0');
    DWORD got = ::GetFinalPathNameByHandleW(h, buf.data(), needed, VOLUME_NAME_NT | FILE_NAME_NORMALIZED);
    if (got == 0 || got >= needed) return false;
    buf.resize(static_cast<std::size_t>(got));
    while (buf.size() > 1 && buf.back() == L'\\') {
        buf.pop_back();
    }
    out = std::move(buf);
    return true;
}

bool paths_equal_ordinal_ci(const std::wstring& a, const std::wstring& b) {
    if (a.size() != b.size()) return false;
    return ::CompareStringOrdinal(a.c_str(), static_cast<int>(a.size()), b.c_str(), static_cast<int>(b.size()), TRUE) ==
           CSTR_EQUAL;
}

// A segment that CreateFileW or the NT parser could resolve outside parent\segment.
bool unsafe_segment(std::string_view seg) {
    if (seg.empty() || seg == "." || seg == "..") return true;
    return seg.find_first_of(std::string_view(":\\\0", 3)) != std::string_view::npos;
}

// Splits `rel` into UTF-16 segments; false when any segment is unsafe or not valid UTF-8.
bool split_rel(std::string_view rel, std::vector<std::wstring>& out) {
    out.clear();
    std::size_t pos = 0;
    while (pos <= rel.size()) {
        std::size_t next = rel.find('/', pos);
        if (next == std::string_view::npos) next = rel.size();
        const std::string_view seg = rel.substr(pos, next - pos);
        if (unsafe_segment(seg)) return false;
        std::wstring w;
        if (!utf8_to_wide(seg, w) || w.size() > 0x7fffu / sizeof(wchar_t)) return false;
        out.push_back(std::move(w));
        pos = next + 1;
    }
    return true;
}

std::int64_t filetime_ticks(const FILETIME& ft) {
    return static_cast<std::int64_t>((static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) |
                                     static_cast<std::uint64_t>(ft.dwLowDateTime));
}

std::atomic<bool> g_weak_identity_for_testing{false};  // set_weak_identity_for_testing
thread_local std::uint64_t t_dir_entries_visited = 0;    // dir_entries_visited_on_this_thread

// GetFileInformationByHandleEx(FileIdInfo) errors that mean "this file system does not
// implement the class", as opposed to a transient, permission or device failure.
bool file_id_info_unsupported(DWORD err) {
    return err == ERROR_INVALID_PARAMETER || err == ERROR_INVALID_FUNCTION || err == ERROR_NOT_SUPPORTED;
}

// Identity from the handle: the 128-bit FILE_ID_INFO id and 64-bit volume serial, size, link
// count, last-write and change time. Only when the file system reports FileIdInfo as unsupported
// is the 64-bit BY_HANDLE_FILE_INFORMATION index used (out.weak_id = true); any other FileIdInfo
// failure fails the query (fail closed). `attrs` receives the attributes.
bool fill_identity(HANDLE h, FileIdentity& out, DWORD& attrs, std::string& err_msg) {
    BY_HANDLE_FILE_INFORMATION info{};
    if (!::GetFileInformationByHandle(h, &info)) {
        err_msg = win32_err_string(::GetLastError());
        return false;
    }
    FILE_BASIC_INFO basic{};
    if (!::GetFileInformationByHandleEx(h, FileBasicInfo, &basic, sizeof(basic))) {
        err_msg = win32_err_string(::GetLastError());
        return false;
    }
    FILE_ID_INFO id_info{};
    if (::GetFileInformationByHandleEx(h, FileIdInfo, &id_info, sizeof(id_info))) {
        static_assert(sizeof(id_info.FileId.Identifier) == 16);
        out.volume = id_info.VolumeSerialNumber;
        std::memcpy(&out.file_id, id_info.FileId.Identifier, 8);
        std::memcpy(&out.file_id_high, id_info.FileId.Identifier + 8, 8);
        out.weak_id = false;
    } else {
        const DWORD err = ::GetLastError();
        if (!file_id_info_unsupported(err)) {
            err_msg = "cannot read the file id: " + win32_err_string(err);
            return false;
        }
        out.volume = info.dwVolumeSerialNumber;
        out.file_id =
            (static_cast<std::uint64_t>(info.nFileIndexHigh) << 32) | static_cast<std::uint64_t>(info.nFileIndexLow);
        out.file_id_high = 0;
        out.weak_id = true;
    }
    if (g_weak_identity_for_testing.load(std::memory_order_relaxed)) out.weak_id = true;
    out.size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | static_cast<std::uint64_t>(info.nFileSizeLow);
    out.link_count = info.nNumberOfLinks;
    out.write_time = filetime_ticks(info.ftLastWriteTime);
    out.change_time = static_cast<std::int64_t>(basic.ChangeTime.QuadPart);
    attrs = info.dwFileAttributes;
    return true;
}

// NtCreateFile(FILE_OPEN) of the single component `name` relative to the directory `parent`.
NTSTATUS nt_open(HANDLE parent, const std::wstring& name, ACCESS_MASK access, ULONG share, ULONG options,
                 OwnedFileHandle& out) {
    UNICODE_STRING us;
    us.Length = static_cast<USHORT>(name.size() * sizeof(wchar_t));
    us.MaximumLength = us.Length;
    us.Buffer = const_cast<PWSTR>(name.c_str());
    OBJECT_ATTRIBUTES oa;
    InitializeObjectAttributes(&oa, &us, OBJ_CASE_INSENSITIVE, parent, nullptr);
    IO_STATUS_BLOCK iosb{};
    HANDLE h = nullptr;
    const NTSTATUS st = ::NtCreateFile(&h, access, &oa, &iosb, nullptr, 0, share, FILE_OPEN, options, nullptr, 0);
    if (NT_SUCCESS(st)) out = make_owned(h);
    return st;
}

OpenOutcome map_open_status(NTSTATUS st, bool* busy, std::string& err_msg) {
    if (st == kStatusObjectNameNotFound || st == kStatusObjectPathNotFound || st == kStatusNoSuchFile ||
        st == kStatusDeletePending) {
        return OpenOutcome::missing;
    }
    if (st == kStatusFileIsADirectory || st == kStatusNotADirectory || st == kStatusObjectNameInvalid ||
        st == kStatusObjectPathSyntaxBad || st == kStatusReparsePointEncountered) {
        return OpenOutcome::unsafe;
    }
    if (st == kStatusSharingViolation) {
        if (busy != nullptr) *busy = true;
        err_msg = "the file is open for writing or deletion by another process";
        return OpenOutcome::io_error;
    }
    err_msg = nt_err_string(st);
    return OpenOutcome::io_error;
}

// Opens segs[0, count) as real directories, starting at `anchor`. On ok `dir` is the last one
// (invalid when count == 0: the anchor itself is the parent) and `expected_nt` its final path.
// `list_last` adds FILE_LIST_DIRECTORY to the last directory opened.
OpenOutcome open_dirs(HANDLE anchor, const std::vector<std::wstring>& segs, std::size_t count, bool list_last,
                      OwnedFileHandle& dir, std::wstring& expected_nt, std::string& err_msg) {
    if (!get_final_nt_path(anchor, expected_nt)) {
        err_msg = win32_err_string(::GetLastError());
        return OpenOutcome::io_error;
    }
    dir.reset();
    for (std::size_t i = 0; i < count; ++i) {
        const HANDLE parent = dir.valid() ? to_handle(dir.native_value()) : anchor;
        ACCESS_MASK access = FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE;
        if (list_last && i + 1 == count) access |= FILE_LIST_DIRECTORY;
        OwnedFileHandle child;
        // No FILE_DIRECTORY_FILE (see the file comment): the attribute check below refuses a
        // non-directory and a reparse point on the handle that was actually opened.
        const NTSTATUS st =
            nt_open(parent, segs[i], access, kShareAll, FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT, child);
        if (!NT_SUCCESS(st)) return map_open_status(st, nullptr, err_msg);
        const HANDLE raw = to_handle(child.native_value());
        BY_HANDLE_FILE_INFORMATION info{};
        if (!::GetFileInformationByHandle(raw, &info)) {
            err_msg = win32_err_string(::GetLastError());
            return OpenOutcome::io_error;
        }
        if ((info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
            (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
            return OpenOutcome::unsafe;
        }
        expected_nt.push_back(L'\\');
        expected_nt.append(segs[i]);
        std::wstring actual_nt;
        if (!get_final_nt_path(raw, actual_nt) || !paths_equal_ordinal_ci(actual_nt, expected_nt)) {
            return OpenOutcome::unsafe;
        }
        expected_nt = std::move(actual_nt);  // the on-disk spelling of the components so far
        dir = std::move(child);
    }
    return OpenOutcome::ok;
}

}  // namespace

void set_weak_identity_for_testing(bool on) noexcept { g_weak_identity_for_testing.store(on, std::memory_order_relaxed); }

std::uint64_t dir_entries_visited_on_this_thread() noexcept { return t_dir_entries_visited; }

OwnedFileHandle::OwnedFileHandle() noexcept : handle_(to_raw(INVALID_HANDLE_VALUE)) {}

OwnedFileHandle::~OwnedFileHandle() { reset(); }

OwnedFileHandle::OwnedFileHandle(OwnedFileHandle&& other) noexcept : handle_(other.handle_) {
    other.handle_ = to_raw(INVALID_HANDLE_VALUE);
}

OwnedFileHandle& OwnedFileHandle::operator=(OwnedFileHandle&& other) noexcept {
    if (this != &other) {
        reset();
        handle_ = other.handle_;
        other.handle_ = to_raw(INVALID_HANDLE_VALUE);
    }
    return *this;
}

void OwnedFileHandle::reset() noexcept {
    const HANDLE h = to_handle(handle_);
    if (h != INVALID_HANDLE_VALUE) {
        ::CloseHandle(h);
        handle_ = to_raw(INVALID_HANDLE_VALUE);
    }
}

bool OwnedFileHandle::valid() const noexcept { return to_handle(handle_) != INVALID_HANDLE_VALUE; }

void OwnedFileHandle::assign_native(std::intptr_t raw) noexcept {
    reset();
    handle_ = raw;
}

std::intptr_t OwnedFileHandle::release() noexcept {
    const std::intptr_t raw = handle_;
    handle_ = to_raw(INVALID_HANDLE_VALUE);
    return raw;
}

bool is_link_like(const fs::path& p, const fs::file_status& st) {
    if (st.type() == fs::file_type::symlink) return true;
    // Junctions, OneDrive placeholders and other reparse points can redirect reads.
    const DWORD attrs = ::GetFileAttributesW(p.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
}

OpenOutcome open_anchor_dir(const fs::path& dir, OwnedFileHandle& out_dir, FileIdentity& out_id, std::string& err_msg) {
    // Follows a link at `dir` (the operator chose it); the handle then pins what it reached.
    OwnedFileHandle h = make_owned(::CreateFileW(dir.c_str(), FILE_LIST_DIRECTORY | FILE_TRAVERSE | FILE_READ_ATTRIBUTES | SYNCHRONIZE,
                                                 kShareAll, nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr));
    if (!h.valid()) {
        const DWORD err = ::GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) return OpenOutcome::missing;
        if (err == ERROR_DIRECTORY) return OpenOutcome::unsafe;
        err_msg = win32_err_string(err);
        return OpenOutcome::io_error;
    }
    DWORD attrs = 0;
    FileIdentity id;
    if (!fill_identity(to_handle(h.native_value()), id, attrs, err_msg)) return OpenOutcome::io_error;
    if ((attrs & FILE_ATTRIBUTE_DIRECTORY) == 0) return OpenOutcome::unsafe;
    out_id = id;
    out_dir = std::move(h);
    return OpenOutcome::ok;
}

OpenOutcome open_anchored_file(const OwnedFileHandle& anchor, std::string_view rel, ShareMode mode,
                               OwnedFileHandle& out_handle, FileIdentity& out_id, std::string& err_msg, bool* busy) {
    if (busy != nullptr) *busy = false;
    if (!anchor.valid()) {
        err_msg = "store root is not open";
        return OpenOutcome::io_error;
    }
    std::vector<std::wstring> segs;
    if (!split_rel(rel, segs)) return OpenOutcome::unsafe;
    OwnedFileHandle dir;
    std::wstring expected_nt;
    const HANDLE anchor_raw = to_handle(anchor.native_value());
    const OpenOutcome dirs = open_dirs(anchor_raw, segs, segs.size() - 1, false, dir, expected_nt, err_msg);
    if (dirs != OpenOutcome::ok) return dirs;

    const ULONG share = mode == ShareMode::deny_writers ? FILE_SHARE_READ : kShareAll;
    OwnedFileHandle leaf;
    const NTSTATUS st = nt_open(dir.valid() ? to_handle(dir.native_value()) : anchor_raw, segs.back(), FILE_GENERIC_READ,
                                share,
                                FILE_NON_DIRECTORY_FILE | FILE_OPEN_REPARSE_POINT | FILE_SYNCHRONOUS_IO_NONALERT |
                                    FILE_SEQUENTIAL_ONLY,
                                leaf);
    if (!NT_SUCCESS(st)) return map_open_status(st, busy, err_msg);
    const HANDLE raw = to_handle(leaf.native_value());
    if (::GetFileType(raw) != FILE_TYPE_DISK) return OpenOutcome::unsafe;
    DWORD attrs = 0;
    FileIdentity id;
    if (!fill_identity(raw, id, attrs, err_msg)) return OpenOutcome::io_error;
    if ((attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0 || (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return OpenOutcome::unsafe;
    }
    expected_nt.push_back(L'\\');
    expected_nt.append(segs.back());
    std::wstring leaf_nt;
    if (!get_final_nt_path(raw, leaf_nt) || !paths_equal_ordinal_ci(leaf_nt, expected_nt)) {
        return OpenOutcome::unsafe;
    }
    out_id = id;
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
    std::vector<std::wstring> segs;
    if (!split_rel(rel, segs)) return OpenOutcome::unsafe;
    OwnedFileHandle dir;
    std::wstring expected_nt;
    return open_dirs(to_handle(anchor.native_value()), segs, segs.size(), false, dir, expected_nt, err_msg);
}

OpenOutcome visit_anchored_dir(const OwnedFileHandle& anchor, std::string_view rel, std::size_t max_entries,
                               const std::function<bool(DirEntry&)>& visit, bool& truncated, std::string& err_msg) {
    truncated = false;
    if (!anchor.valid()) {
        err_msg = "store root is not open";
        return OpenOutcome::io_error;
    }
    std::vector<std::wstring> segs;
    if (!split_rel(rel, segs)) return OpenOutcome::unsafe;
    OwnedFileHandle dir;
    std::wstring expected_nt;
    const OpenOutcome opened = open_dirs(to_handle(anchor.native_value()), segs, segs.size(), true, dir, expected_nt, err_msg);
    if (opened != OpenOutcome::ok) return opened;

    // FILE_ID_BOTH_DIR_INFO records are 8-byte aligned inside the buffer; each one is copied out
    // with memcpy (fixed part, then FileNameLength bytes), bounds-checked against the buffer.
    // The buffer belongs to this call, so `visit` may list another directory meanwhile.
    std::vector<std::uint64_t> storage(kListBufferBytes / sizeof(std::uint64_t));
    const auto* base = reinterpret_cast<const unsigned char*>(storage.data());
    constexpr std::size_t kNameOffset = offsetof(FILE_ID_BOTH_DIR_INFO, FileName);
    const HANDLE raw = to_handle(dir.native_value());
    std::size_t visited = 0;
    while (true) {
        if (!::GetFileInformationByHandleEx(raw, FileIdBothDirectoryInfo, storage.data(),
                                            static_cast<DWORD>(kListBufferBytes))) {
            const DWORD err = ::GetLastError();
            if (err == ERROR_NO_MORE_FILES) break;
            err_msg = win32_err_string(err);
            return OpenOutcome::io_error;
        }
        std::size_t off = 0;
        while (true) {
            if (off + kNameOffset > kListBufferBytes) {
                err_msg = "malformed directory listing";
                return OpenOutcome::io_error;
            }
            FILE_ID_BOTH_DIR_INFO info{};
            std::memcpy(&info, base + off, kNameOffset);
            const std::size_t name_bytes = info.FileNameLength;
            if (off + kNameOffset + name_bytes > kListBufferBytes || name_bytes % sizeof(wchar_t) != 0) {
                err_msg = "malformed directory listing";
                return OpenOutcome::io_error;
            }
            std::wstring wname(name_bytes / sizeof(wchar_t), L'\0');
            if (name_bytes != 0) std::memcpy(wname.data(), base + off + kNameOffset, name_bytes);
            if (wname != L"." && wname != L"..") {
                if (visited == max_entries) {  // one entry more than the caller allows: stop reading
                    truncated = true;
                    return OpenOutcome::ok;
                }
                ++visited;
                ++t_dir_entries_visited;
                DirEntry e;
                e.name_ok = wide_to_utf8(wname, e.name);
                if (!e.name_ok) e.name.clear();
                if ((info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
                    e.kind = DirEntryKind::link;
                } else if ((info.FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
                    e.kind = DirEntryKind::directory;
                } else {
                    e.kind = DirEntryKind::file;
                }
                if (!visit(e)) return OpenOutcome::ok;
            }
            if (info.NextEntryOffset == 0) break;
            off += info.NextEntryOffset;
        }
    }
    return OpenOutcome::ok;
}

OpenOutcome open_single_regular(const fs::path& file, OwnedFileHandle& out_handle, std::uint64_t& out_size,
                                std::string& err_msg) {
    OwnedFileHandle h = make_owned(::CreateFileW(
        file.c_str(), GENERIC_READ, kShareAll, nullptr, OPEN_EXISTING,
        FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
    if (!h.valid()) {
        const DWORD err = ::GetLastError();
        if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) {
            err_msg = win32_err_string(err);
            return OpenOutcome::missing;
        }
        if (err == ERROR_ACCESS_DENIED) {
            const DWORD attrs = ::GetFileAttributesW(file.c_str());
            if (attrs != INVALID_FILE_ATTRIBUTES &&
                ((attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0 || (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0)) {
                return OpenOutcome::unsafe;
            }
        }
        err_msg = win32_err_string(err);
        return OpenOutcome::io_error;
    }
    const HANDLE raw = to_handle(h.native_value());
    if (::GetFileType(raw) != FILE_TYPE_DISK) {
        return OpenOutcome::unsafe;
    }
    BY_HANDLE_FILE_INFORMATION info{};
    if (!::GetFileInformationByHandle(raw, &info)) {
        err_msg = win32_err_string(::GetLastError());
        return OpenOutcome::io_error;
    }
    if ((info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
        (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
        return OpenOutcome::unsafe;
    }
    out_size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | static_cast<std::uint64_t>(info.nFileSizeLow);
    out_handle = std::move(h);
    return OpenOutcome::ok;
}

bool query_identity(const OwnedFileHandle& handle, FileIdentity& out, std::string& err_msg) {
    DWORD attrs = 0;
    return fill_identity(to_handle(handle.native_value()), out, attrs, err_msg);
}

std::int64_t read_some(const OwnedFileHandle& handle, char* buf, std::size_t n, std::string& err_msg) {
    const DWORD want = static_cast<DWORD>(std::min<std::size_t>(n, 0x40000000u));
    DWORD got = 0;
    if (!::ReadFile(to_handle(handle.native_value()), buf, want, &got, nullptr)) {
        err_msg = win32_err_string(::GetLastError());
        return -1;
    }
    return static_cast<std::int64_t>(got);
}

bool read_anchored_leaf_symlink(const OwnedFileHandle& anchor, std::string_view rel, std::string& target,
                                FileIdentity& link_id, std::string& err_msg) {
    // Ollama writes manifest copies on Windows (manifest/manifest.go "Avoid symlinks on
    // windows"), and following a link here could open a UNC target. Links are never followed.
    (void)anchor;
    (void)rel;
    (void)target;
    (void)link_id;
    err_msg = "symbolic links are not followed on Windows";
    return false;
}

}  // namespace sonder::inference::detail
