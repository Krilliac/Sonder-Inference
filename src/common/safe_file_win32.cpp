// Windows (Win32) handle-anchored regular-file open and reparse-point checks
// (src/common/safe_file.hpp). Moved here from src/artifacts/src/weight_manifest_io_win32.cpp
// so the weight manifests and the Ollama model store share one implementation.
#include "common/safe_file.hpp"

#include <algorithm>
#include <string>
#include <utility>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace sonder::inference::detail {

namespace fs = std::filesystem;

namespace {

std::intptr_t to_raw(HANDLE h) noexcept { return reinterpret_cast<std::intptr_t>(h); }
HANDLE to_handle(std::intptr_t raw) noexcept { return reinterpret_cast<HANDLE>(raw); }

fs::path from_utf8(std::string_view s) { return fs::path(std::u8string(s.begin(), s.end())); }

std::string win32_err_string(DWORD err) { return "win32 error " + std::to_string(err); }

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

OwnedFileHandle make_owned(HANDLE h) noexcept {
    OwnedFileHandle out;
    out.assign_native(to_raw(h));
    return out;
}

// A segment that operator/ or CreateFileW could resolve outside root/segment.
bool unsafe_segment(std::string_view seg) {
    if (seg.empty() || seg == "." || seg == "..") return true;
    return seg.find_first_of(std::string_view(":\\\0", 3)) != std::string_view::npos;
}

std::int64_t filetime_ticks(const FILETIME& ft) {
    return static_cast<std::int64_t>((static_cast<std::uint64_t>(ft.dwHighDateTime) << 32) |
                                     static_cast<std::uint64_t>(ft.dwLowDateTime));
}

bool fill_identity(HANDLE h, const BY_HANDLE_FILE_INFORMATION& info, FileIdentity& out, std::string& err_msg) {
    FILE_BASIC_INFO basic{};
    if (!::GetFileInformationByHandleEx(h, FileBasicInfo, &basic, sizeof(basic))) {
        err_msg = win32_err_string(::GetLastError());
        return false;
    }
    out.volume = info.dwVolumeSerialNumber;
    out.file_id = (static_cast<std::uint64_t>(info.nFileIndexHigh) << 32) | static_cast<std::uint64_t>(info.nFileIndexLow);
    out.size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | static_cast<std::uint64_t>(info.nFileSizeLow);
    out.link_count = info.nNumberOfLinks;
    out.write_time = filetime_ticks(info.ftLastWriteTime);
    out.change_time = static_cast<std::int64_t>(basic.ChangeTime.QuadPart);
    return true;
}

}  // namespace

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

bool is_link_like(const fs::path& p, const fs::file_status& st) {
    if (st.type() == fs::file_type::symlink) return true;
    // Junctions, OneDrive placeholders and other reparse points can redirect reads.
    const DWORD attrs = ::GetFileAttributesW(p.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
}

OpenOutcome open_anchored_file(const fs::path& root, std::string_view rel, OwnedFileHandle& out_handle,
                               FileIdentity& out_id, std::string& err_msg) {
    OwnedFileHandle root_h = make_owned(::CreateFileW(root.c_str(), FILE_READ_ATTRIBUTES,
                                                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                                      OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr));
    if (!root_h.valid()) {
        err_msg = win32_err_string(::GetLastError());
        return OpenOutcome::io_error;
    }
    std::wstring root_nt;
    if (!get_final_nt_path(to_handle(root_h.native_value()), root_nt)) {
        err_msg = win32_err_string(::GetLastError());
        return OpenOutcome::io_error;
    }

    fs::path cur = root;
    std::wstring expected_nt = root_nt;
    std::size_t pos = 0;
    while (pos <= rel.size()) {
        std::size_t next = rel.find('/', pos);
        if (next == std::string_view::npos) next = rel.size();
        const std::string_view seg_text = rel.substr(pos, next - pos);
        if (unsafe_segment(seg_text)) return OpenOutcome::unsafe;
        const fs::path seg = from_utf8(seg_text);
        cur /= seg;
        expected_nt.push_back(L'\\');
        expected_nt.append(seg.native());
        const bool last = next >= rel.size();

        if (!last) {
            OwnedFileHandle dir_h = make_owned(::CreateFileW(
                cur.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
            if (!dir_h.valid()) {
                const DWORD err = ::GetLastError();
                if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) return OpenOutcome::missing;
                err_msg = win32_err_string(err);
                return OpenOutcome::io_error;
            }
            const HANDLE raw_dir = to_handle(dir_h.native_value());
            BY_HANDLE_FILE_INFORMATION info{};
            if (!::GetFileInformationByHandle(raw_dir, &info)) {
                err_msg = win32_err_string(::GetLastError());
                return OpenOutcome::io_error;
            }
            if ((info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
                (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) == 0) {
                return OpenOutcome::unsafe;
            }
            std::wstring dir_nt;
            if (!get_final_nt_path(raw_dir, dir_nt) || !paths_equal_ordinal_ci(dir_nt, expected_nt)) {
                return OpenOutcome::unsafe;
            }
            pos = next + 1;
            continue;
        }

        // Open the leaf without following reparse points; backup semantics lets
        // us inspect directory/junction handles directly as unsafe instead of
        // misclassifying them as access errors.
        OwnedFileHandle leaf_h = make_owned(::CreateFileW(
            cur.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
            FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_SEQUENTIAL_SCAN, nullptr));
        if (!leaf_h.valid()) {
            const DWORD err = ::GetLastError();
            if (err == ERROR_FILE_NOT_FOUND || err == ERROR_PATH_NOT_FOUND) return OpenOutcome::missing;
            if (err == ERROR_ACCESS_DENIED) {
                const DWORD attrs = ::GetFileAttributesW(cur.c_str());
                if (attrs != INVALID_FILE_ATTRIBUTES &&
                    ((attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0 || (attrs & FILE_ATTRIBUTE_DIRECTORY) != 0)) {
                    return OpenOutcome::unsafe;
                }
            }
            err_msg = win32_err_string(err);
            return OpenOutcome::io_error;
        }
        const HANDLE raw_leaf = to_handle(leaf_h.native_value());
        if (::GetFileType(raw_leaf) != FILE_TYPE_DISK) {
            return OpenOutcome::unsafe;
        }
        BY_HANDLE_FILE_INFORMATION info{};
        if (!::GetFileInformationByHandle(raw_leaf, &info)) {
            err_msg = win32_err_string(::GetLastError());
            return OpenOutcome::io_error;
        }
        if ((info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 ||
            (info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0) {
            return OpenOutcome::unsafe;
        }
        std::wstring leaf_nt;
        if (!get_final_nt_path(raw_leaf, leaf_nt) || !paths_equal_ordinal_ci(leaf_nt, expected_nt)) {
            return OpenOutcome::unsafe;
        }
        if (!fill_identity(raw_leaf, info, out_id, err_msg)) return OpenOutcome::io_error;
        out_handle = std::move(leaf_h);
        return OpenOutcome::ok;
    }
    return OpenOutcome::missing;
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
    OwnedFileHandle h = make_owned(::CreateFileW(
        file.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
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
    const HANDLE raw = to_handle(handle.native_value());
    BY_HANDLE_FILE_INFORMATION info{};
    if (!::GetFileInformationByHandle(raw, &info)) {
        err_msg = win32_err_string(::GetLastError());
        return false;
    }
    return fill_identity(raw, info, out, err_msg);
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

bool read_anchored_leaf_symlink(const fs::path& root, std::string_view rel, std::string& target,
                                FileIdentity& link_id, FileIdentity& target_id, std::string& err_msg) {
    // Ollama writes manifest copies on Windows (manifest/manifest.go "Avoid symlinks on
    // windows"), and following a link here could open a UNC target. Links are never followed.
    (void)root;
    (void)rel;
    (void)target;
    (void)link_id;
    (void)target_id;
    err_msg = "symbolic links are not followed on Windows";
    return false;
}

}  // namespace sonder::inference::detail
