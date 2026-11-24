// Windows (Win32) handle-anchored regular-file open, reparse-point check, and
// streaming SHA-256 hashing for weight manifests.
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

OpenOutcome open_anchored_regular(const fs::path& root, std::string_view rel, OwnedFileHandle& out_handle,
                                  std::uint64_t& out_size, std::string& err_msg) {
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
        const fs::path seg = from_utf8(rel.substr(pos, next - pos));
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
        out_size = (static_cast<std::uint64_t>(info.nFileSizeHigh) << 32) | static_cast<std::uint64_t>(info.nFileSizeLow);
        out_handle = std::move(leaf_h);
        return OpenOutcome::ok;
    }
    return OpenOutcome::missing;
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
