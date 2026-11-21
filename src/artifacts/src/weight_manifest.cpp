#include "sonder/inference/weight_manifest.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <set>

#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"
#include "server/src/sha256.hpp"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace sonder::inference::weights {

namespace fs = std::filesystem;
namespace json = sonder::inference::json;

namespace {

constexpr std::size_t kMinBuffer = 4096;
constexpr std::size_t kMaxBuffer = 64u << 20;

std::string to_utf8(const fs::path& p) {
    const auto u = p.generic_u8string();
    return std::string(u.begin(), u.end());
}

// Converts without throwing: Windows names with unpaired UTF-16 surrogates make generic_u8string throw.
bool try_to_utf8(const fs::path& p, std::string& out) {
    try {
        out = to_utf8(p);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

// For messages only: never throws, falls back to a placeholder.
std::string display_utf8(const fs::path& p) {
    std::string s;
    return try_to_utf8(p, s) ? s : std::string("<name not representable as UTF-8>");
}

// Strict RFC 3629 sequence check at s[i]: returns the sequence length (1-4), or 0 when it is
// ill-formed (no overlongs, no surrogates, nothing above U+10FFFF).
std::size_t utf8_seq_len(std::string_view s, std::size_t i) {
    const std::size_t n = s.size();
    auto cont = [&](std::size_t k) { return k < n && (static_cast<unsigned char>(s[k]) & 0xC0) == 0x80; };
    const auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) return 1;
    if (c >= 0xC2 && c <= 0xDF) return cont(i + 1) ? 2 : 0;
    if (c >= 0xE0 && c <= 0xEF) {
        if (!cont(i + 1) || !cont(i + 2)) return 0;
        const auto c1 = static_cast<unsigned char>(s[i + 1]);
        if (c == 0xE0 && c1 < 0xA0) return 0;   // overlong
        if (c == 0xED && c1 >= 0xA0) return 0;  // surrogate
        return 3;
    }
    if (c >= 0xF0 && c <= 0xF4) {
        if (!cont(i + 1) || !cont(i + 2) || !cont(i + 3)) return 0;
        const auto c1 = static_cast<unsigned char>(s[i + 1]);
        if (c == 0xF0 && c1 < 0x90) return 0;   // overlong
        if (c == 0xF4 && c1 >= 0x90) return 0;  // above U+10FFFF
        return 4;
    }
    return 0;
}

bool is_valid_utf8(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const std::size_t len = utf8_seq_len(s, i);
        if (len == 0) return false;
        i += len;
    }
    return true;
}

fs::path from_utf8(std::string_view s) {
    return fs::path(std::u8string(s.begin(), s.end()));
}

std::string to_hex(const std::array<std::uint8_t, 32>& d) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string out;
    out.reserve(64);
    for (std::uint8_t b : d) {
        out.push_back(kDigits[b >> 4]);
        out.push_back(kDigits[b & 15]);
    }
    return out;
}

bool is_link_like(const fs::path& p, const fs::file_status& st) {
    if (st.type() == fs::file_type::symlink) return true;
#if defined(_WIN32)
    // Junctions, OneDrive placeholders and other reparse points can redirect reads.
    const DWORD attrs = GetFileAttributesW(p.c_str());
    if (attrs != INVALID_FILE_ATTRIBUTES && (attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0) return true;
#else
    (void)p;
#endif
    return false;
}

std::string utc_now() {
    using namespace std::chrono;
    const std::int64_t secs =
        static_cast<std::int64_t>(duration_cast<seconds>(system_clock::now().time_since_epoch()).count());
    std::int64_t days = secs / 86400;
    std::int64_t rem = secs % 86400;
    if (rem < 0) {
        rem += 86400;
        --days;
    }
    // Civil-from-days (proleptic Gregorian).
    const std::int64_t z = days + 719468;
    const std::int64_t era = (z >= 0 ? z : z - 146096) / 146097;
    const std::int64_t doe = z - era * 146097;
    const std::int64_t yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    const std::int64_t doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    const std::int64_t mp = (5 * doy + 2) / 153;
    const std::int64_t d = doy - (153 * mp + 2) / 5 + 1;
    const std::int64_t m = mp < 10 ? mp + 3 : mp - 9;
    const std::int64_t y = yoe + era * 400 + (m <= 2 ? 1 : 0);
    char buf[48];
    std::snprintf(buf, sizeof buf, "%04lld-%02lld-%02lldT%02lld:%02lld:%02lldZ", static_cast<long long>(y),
                  static_cast<long long>(m), static_cast<long long>(d), static_cast<long long>(rem / 3600),
                  static_cast<long long>(rem % 3600 / 60), static_cast<long long>(rem % 60));
    return buf;
}

std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9e3779b97f4a7c15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

enum class Presence { ok, missing, unsafe };

// Checks every component of `rel` below `root` without following links.
Presence check_entry(const fs::path& root, std::string_view rel, fs::path& resolved) {
    resolved = root;
    std::size_t pos = 0;
    while (pos <= rel.size()) {
        std::size_t next = rel.find('/', pos);
        if (next == std::string_view::npos) next = rel.size();
        resolved /= from_utf8(rel.substr(pos, next - pos));
        std::error_code ec;
        const fs::file_status st = fs::symlink_status(resolved, ec);
        if (ec || st.type() == fs::file_type::not_found) return Presence::missing;
        if (is_link_like(resolved, st)) return Presence::unsafe;
        const bool last = next >= rel.size();
        if (last) return st.type() == fs::file_type::regular ? Presence::ok : Presence::unsafe;
        if (st.type() != fs::file_type::directory) return Presence::unsafe;
        pos = next + 1;
    }
    return Presence::missing;
}

Status check_manifest_entries(const WeightManifest& m) {
    std::set<std::string> seen;
    for (const auto& f : m.files) {
        Status s = validate_relative_path(f.path);
        if (!s.ok()) return s;
        if (!is_sha256_hex(f.sha256)) {
            return Status(ErrorCode::invalid_argument,
                          "manifest entry '" + printable_path(f.path) + "': sha256 must be 64 lowercase hex digits");
        }
        if (f.expected_sha256 && !is_sha256_hex(*f.expected_sha256)) {
            return Status(ErrorCode::invalid_argument,
                          "manifest entry '" + printable_path(f.path) + "': expected_sha256 must be 64 lowercase hex digits");
        }
        if (!seen.insert(f.path).second) {
            return Status(ErrorCode::invalid_argument, "manifest lists '" + printable_path(f.path) + "' twice");
        }
    }
    return Status::success();
}

}  // namespace

std::string printable_path(std::string_view s) {
    static constexpr char kHex[] = "0123456789abcdef";
    std::string out;
    out.reserve(s.size());
    auto hex_byte = [&](unsigned char c) {
        out += "\\x";
        out.push_back(kHex[c >> 4]);
        out.push_back(kHex[c & 15]);
    };
    std::size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            if (c < 0x20 || c == 0x7F) {
                hex_byte(c);
            } else if (c == '\\') {
                out += "\\\\";
            } else {
                out.push_back(static_cast<char>(c));
            }
            ++i;
            continue;
        }
        const std::size_t len = utf8_seq_len(s, i);
        if (len == 0) {
            hex_byte(c);
            ++i;
            continue;
        }
        std::uint32_t cp = 0;
        if (len == 2) cp = (c & 0x1Fu) << 6 | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu);
        if (len == 3) {
            cp = (c & 0x0Fu) << 12 | (static_cast<unsigned char>(s[i + 1]) & 0x3Fu) << 6 |
                 (static_cast<unsigned char>(s[i + 2]) & 0x3Fu);
        }
        const bool risky = (cp >= 0x80 && cp <= 0x9F) || cp == 0x2028 || cp == 0x2029 ||
                           (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2066 && cp <= 0x2069);
        if (risky) {
            out += "\\u";
            for (int shift = 12; shift >= 0; shift -= 4) out.push_back(kHex[(cp >> shift) & 15]);
        } else {
            out.append(s.substr(i, len));
        }
        i += len;
    }
    return out;
}

Status validate_relative_path(std::string_view path) {
    auto bad = [&](const char* why) {
        const std::string shown = "\"" + printable_path(path.substr(0, 200)) + "\"";
        return Status(ErrorCode::invalid_argument, std::string("unsafe path ") + shown + ": " + why);
    };
    if (path.empty()) return bad("empty");
    if (path.front() == '/') return bad("absolute");
    if (!is_valid_utf8(path)) return bad("not valid UTF-8");
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        // C1 controls (U+0080-U+009F) are bytes C2 80..C2 9F; a lead byte 0xC2 is never a continuation.
        if (static_cast<unsigned char>(path[i]) == 0xC2 && static_cast<unsigned char>(path[i + 1]) >= 0x80 &&
            static_cast<unsigned char>(path[i + 1]) <= 0x9F) {
            return bad("control character");
        }
    }
    std::size_t pos = 0;
    while (pos <= path.size()) {
        std::size_t next = path.find('/', pos);
        if (next == std::string_view::npos) next = path.size();
        const std::string_view seg = path.substr(pos, next - pos);
        if (seg.empty()) return bad("empty segment");
        if (seg == "." || seg == "..") return bad("'.' or '..' segment");
        if (seg.back() == '.' || seg.back() == ' ') return bad("segment ends with '.' or space");
        for (char c : seg) {
            const auto u = static_cast<unsigned char>(c);
            if (u < 0x20 || u == 0x7f) return bad("control character");
            if (c == '\\') return bad("backslash");
            if (c == ':') return bad("':' (drive letter or stream)");
        }
        pos = next + 1;
    }
    return Status::success();
}

bool is_sha256_hex(std::string_view text) {
    if (text.size() != 64) return false;
    return std::all_of(text.begin(), text.end(),
                       [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
}

Result<std::string> hash_file(const fs::path& file, std::string_view display, const HashOptions& options,
                              std::uint64_t* bytes_read) {
    std::error_code ec;
    const std::uint64_t total = fs::file_size(file, ec);
    if (ec) return Status(ErrorCode::io_error, "cannot stat '" + printable_path(display) + "': " + ec.message());
    std::ifstream in(file, std::ios::binary);
    if (!in) return Status(ErrorCode::io_error, "cannot open '" + printable_path(display) + "'");
    const std::size_t cap = std::clamp(options.buffer_bytes, kMinBuffer, kMaxBuffer);
    std::string buffer(cap, '\0');
    server::detail::Sha256 sha;
    std::uint64_t done = 0;
    while (true) {
        if (options.cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled while hashing '" + printable_path(display) + "'");
        }
        in.read(buffer.data(), static_cast<std::streamsize>(cap));
        const auto got = static_cast<std::size_t>(in.gcount());
        if (got > 0) {
            sha.update(std::string_view(buffer.data(), got));
            done += got;
            if (options.progress) options.progress(display, done, total);
        }
        if (!in) break;
    }
    if (!in.eof()) return Status(ErrorCode::io_error, "read error on '" + printable_path(display) + "'");
    if (bytes_read) *bytes_read = done;
    return to_hex(sha.finish());
}

Result<DirListing> list_directory(const fs::path& dir, const std::vector<fs::path>& ignore) {
    std::error_code ec;
    if (!fs::is_directory(dir, ec)) {
        return Status(ErrorCode::not_found, "not a directory: " + printable_path(display_utf8(dir)));
    }
    fs::recursive_directory_iterator it(dir, fs::directory_options::none, ec);
    if (ec) return Status(ErrorCode::io_error, "cannot list '" + printable_path(display_utf8(dir)) + "': " + ec.message());
    DirListing out;
    const fs::recursive_directory_iterator end;
    for (; it != end; it.increment(ec)) {
        if (ec) return Status(ErrorCode::io_error, "directory walk failed: " + ec.message());
        const fs::path entry = it->path();
        std::error_code e2;
        const fs::file_status st = fs::symlink_status(entry, e2);
        std::string rel;
        if (!try_to_utf8(entry.lexically_relative(dir), rel)) {
            // Name has no UTF-8 form (e.g. unpaired surrogate); it can never be written to a manifest.
            out.unsafe.push_back("<name not representable as UTF-8>");
            if (e2 || is_link_like(entry, st)) it.disable_recursion_pending();
            continue;
        }
        if (e2) return Status(ErrorCode::io_error, "cannot stat '" + printable_path(rel) + "': " + e2.message());
        if (is_link_like(entry, st)) {
            it.disable_recursion_pending();
            out.unsafe.push_back(rel);
            continue;
        }
        if (st.type() == fs::file_type::directory) continue;
        if (st.type() != fs::file_type::regular) {
            out.unsafe.push_back(rel);
            continue;
        }
        bool ignored = false;
        for (const auto& ig : ignore) {
            std::error_code e3;
            if (fs::equivalent(entry, ig, e3)) {
                ignored = true;
                break;
            }
        }
        if (ignored) continue;
        if (validate_relative_path(rel).ok()) {
            out.files.push_back(rel);
        } else {
            out.unsafe.push_back(rel);
        }
    }
    // increment(ec) turns the iterator into end() on failure, so the loop exits without seeing ec.
    if (ec) return Status(ErrorCode::io_error, "directory walk failed: " + ec.message());
    std::sort(out.files.begin(), out.files.end());
    std::sort(out.unsafe.begin(), out.unsafe.end());
    return out;
}

Result<BuildResult> build_manifest(const fs::path& dir, const BuildOptions& options) {
    auto listing = list_directory(dir, options.ignore);
    if (!listing.ok()) return listing.status();
    if (!listing->unsafe.empty()) {
        return Status(ErrorCode::invalid_argument,
                      "refusing directory: unsafe entry (symlink, reparse point, special file or bad name): " +
                          printable_path(listing->unsafe.front()));
    }
    BuildResult result;
    fs::path leaf = dir.filename();
    if (leaf.empty()) leaf = dir.parent_path().filename();
    if (!try_to_utf8(leaf, result.manifest.model_dir) || !is_valid_utf8(result.manifest.model_dir)) {
        return Status(ErrorCode::invalid_argument, "directory name is not representable as UTF-8");
    }
    result.manifest.created_at = utc_now();
    result.manifest.tool_version = version_string();
    for (const std::string& rel : listing->files) {
        const fs::path full = dir / from_utf8(rel);
        std::uint64_t n = 0;
        auto digest = hash_file(full, rel, options.hash, &n);
        if (!digest.ok()) return digest.status();
        std::error_code ec;
        const std::uint64_t now_size = fs::file_size(full, ec);
        if (ec || now_size != n) {
            return Status(ErrorCode::io_error, "'" + printable_path(rel) + "' changed while it was being hashed");
        }
        ManifestFile f;
        f.path = rel;
        f.size = n;
        f.sha256 = std::move(digest).value();
        const auto exp = options.expected.find(rel);
        if (exp != options.expected.end()) {
            f.expected_sha256 = exp->second;
            if (exp->second != f.sha256) result.expected_mismatch.push_back(rel);
        }
        result.manifest.files.push_back(std::move(f));
    }
    for (const auto& entry : options.expected) {
        if (!std::binary_search(listing->files.begin(), listing->files.end(), entry.first)) {
            result.expected_missing.push_back(entry.first);
        }
    }
    return result;
}

std::string manifest_to_json(const WeightManifest& m) {
    json::Array files;
    for (const auto& f : m.files) {
        json::Object o;
        o.set("path", f.path);
        o.set("size", f.size);
        o.set("sha256", f.sha256);
        if (f.expected_sha256) o.set("expected_sha256", *f.expected_sha256);
        files.emplace_back(std::move(o));
    }
    json::Object root;
    root.set("format", kManifestFormat);
    root.set("version", m.version);
    root.set("model_dir", m.model_dir);
    root.set("created_at", m.created_at);
    root.set("tool", "sonder-infer");
    root.set("tool_version", m.tool_version);
    root.set("files", std::move(files));
    return json::Value(std::move(root)).dump() + "\n";
}

Result<WeightManifest> parse_manifest(std::string_view text) {
    auto parsed = json::parse(text);
    if (!parsed.ok()) {
        return Status(ErrorCode::invalid_argument, "manifest is not valid JSON: " + parsed.status().message());
    }
    const json::Value& root = parsed.value();
    auto fail = [](const std::string& why) { return Status(ErrorCode::invalid_argument, "invalid manifest: " + why); };
    if (!root.is_object()) return fail("top level must be an object");
    const json::Value* format = root.find("format");
    if (!format || !format->is_string() || format->as_string() != kManifestFormat) {
        return fail(std::string("format is not ") + kManifestFormat);
    }
    const json::Value* version = root.find("version");
    if (!version || !version->is_integer() || version->as_int(-1) != kManifestVersion) {
        return fail("unsupported version");
    }
    const json::Value* files = root.find("files");
    if (!files || !files->is_array()) return fail("files must be an array");
    WeightManifest m;
    if (const auto* v = root.find("model_dir"); v && v->is_string()) m.model_dir = v->as_string();
    if (const auto* v = root.find("created_at"); v && v->is_string()) m.created_at = v->as_string();
    if (const auto* v = root.find("tool_version"); v && v->is_string()) m.tool_version = v->as_string();
    for (const json::Value& e : files->as_array()) {
        if (!e.is_object()) return fail("file entry must be an object");
        const json::Value* path = e.find("path");
        const json::Value* size = e.find("size");
        const json::Value* sha = e.find("sha256");
        if (!path || !path->is_string()) return fail("file entry without string path");
        if (!size || !size->is_integer() || size->as_int(-1) < 0) {
            return fail("'" + printable_path(path->as_string()) + "': size must be a non-negative integer");
        }
        if (!sha || !sha->is_string()) return fail("'" + printable_path(path->as_string()) + "': sha256 must be a string");
        ManifestFile f;
        f.path = path->as_string();
        f.size = size->as_uint();
        f.sha256 = sha->as_string();
        if (const auto* ex = e.find("expected_sha256"); ex && !ex->is_null()) {
            if (!ex->is_string()) return fail("'" + printable_path(f.path) + "': expected_sha256 must be a string");
            f.expected_sha256 = ex->as_string();
        }
        m.files.push_back(std::move(f));
    }
    Status s = check_manifest_entries(m);
    if (!s.ok()) return s;
    std::sort(m.files.begin(), m.files.end(),
              [](const ManifestFile& a, const ManifestFile& b) { return a.path < b.path; });
    return m;
}

Result<ExpectedHashes> parse_expected_json(std::string_view text) {
    auto parsed = json::parse(text);
    if (!parsed.ok()) {
        return Status(ErrorCode::invalid_argument, "expected-hash file is not valid JSON: " + parsed.status().message());
    }
    const json::Value* list = &parsed.value();
    if (list->is_object()) {
        const json::Value* siblings = list->find("siblings");
        if (!siblings) siblings = list->find("files");
        if (!siblings) {
            return Status(ErrorCode::invalid_argument, "expected-hash object needs a 'siblings' or 'files' array");
        }
        list = siblings;
    }
    if (!list->is_array()) {
        return Status(ErrorCode::invalid_argument, "expected-hash file must hold an array of entries");
    }
    ExpectedHashes out;
    for (const json::Value& e : list->as_array()) {
        if (!e.is_object()) continue;
        if (const auto* type = e.find("type"); type && type->is_string() && type->as_string() != "file") continue;
        const json::Value* path = e.find("path");
        if (!path || !path->is_string()) path = e.find("rfilename");
        if (!path || !path->is_string()) continue;
        std::string sha;
        if (const auto* lfs = e.find("lfs"); lfs && lfs->is_object()) {
            if (const auto* oid = lfs->find("oid"); oid && oid->is_string()) sha = oid->as_string();
            if (sha.empty()) {
                if (const auto* s2 = lfs->find("sha256"); s2 && s2->is_string()) sha = s2->as_string();
            }
        }
        if (sha.empty()) {
            if (const auto* s2 = e.find("sha256"); s2 && s2->is_string()) sha = s2->as_string();
        }
        if (sha.empty()) continue;
        if (sha.rfind("sha256:", 0) == 0) sha.erase(0, 7);
        std::transform(sha.begin(), sha.end(), sha.begin(),
                       [](char c) { return (c >= 'A' && c <= 'F') ? static_cast<char>(c - 'A' + 'a') : c; });
        Status s = validate_relative_path(path->as_string());
        if (!s.ok()) return s;
        if (!is_sha256_hex(sha)) {
            return Status(ErrorCode::invalid_argument,
                          "expected sha256 for '" + printable_path(path->as_string()) + "' is not 64 hex digits");
        }
        out[path->as_string()] = std::move(sha);
    }
    return out;
}

std::vector<std::size_t> select_sample(std::size_t count, std::size_t n, std::uint64_t seed) {
    std::vector<std::size_t> idx(count);
    for (std::size_t i = 0; i < count; ++i) idx[i] = i;
    if (n >= count) return idx;
    std::uint64_t state = seed;
    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = i + static_cast<std::size_t>(splitmix64(state) % (count - i));
        std::swap(idx[i], idx[j]);
    }
    idx.resize(n);
    std::sort(idx.begin(), idx.end());
    return idx;
}

int VerifyReport::exit_code() const noexcept {
    if (!mismatches.empty()) return 2;
    if (!missing.empty() || !extra.empty() || !unsafe.empty()) return 3;
    return 0;
}

Result<VerifyReport> verify_manifest(const fs::path& dir, const WeightManifest& manifest,
                                     const VerifyOptions& options) {
    Status entries = check_manifest_entries(manifest);
    if (!entries.ok()) return entries;
    auto listing = list_directory(dir, options.ignore);
    if (!listing.ok()) return listing.status();

    VerifyReport report;
    report.files_total = manifest.files.size();
    report.seed = options.seed;
    report.unsafe = listing->unsafe;

    std::set<std::string> in_manifest;
    for (const auto& f : manifest.files) in_manifest.insert(f.path);
    for (const auto& rel : listing->files) {
        if (!in_manifest.count(rel)) report.extra.push_back(rel);
    }

    std::vector<bool> present(manifest.files.size(), false);
    for (std::size_t i = 0; i < manifest.files.size(); ++i) {
        const ManifestFile& f = manifest.files[i];
        if (f.expected_sha256 && *f.expected_sha256 != f.sha256) {
            report.mismatches.push_back({f.path, "expected_sha256", *f.expected_sha256, f.sha256});
        }
        fs::path resolved;
        switch (check_entry(dir, f.path, resolved)) {
            case Presence::missing:
                report.missing.push_back(f.path);
                continue;
            case Presence::unsafe:
                if (std::find(report.unsafe.begin(), report.unsafe.end(), f.path) == report.unsafe.end()) {
                    report.unsafe.push_back(f.path);
                }
                continue;
            case Presence::ok:
                break;
        }
        std::error_code ec;
        const std::uint64_t size = fs::file_size(resolved, ec);
        if (ec) return Status(ErrorCode::io_error, "cannot stat '" + printable_path(f.path) + "': " + ec.message());
        if (size != f.size) {
            report.mismatches.push_back({f.path, "size", std::to_string(f.size), std::to_string(size)});
            continue;
        }
        present[i] = true;
    }

    std::vector<std::size_t> to_hash;
    if (options.full) {
        for (std::size_t i = 0; i < present.size(); ++i) to_hash.push_back(i);
    } else if (options.sample) {
        to_hash = select_sample(manifest.files.size(), *options.sample, options.seed);
    }
    for (std::size_t i : to_hash) {
        if (!present[i]) continue;
        const ManifestFile& f = manifest.files[i];
        auto digest = hash_file(dir / from_utf8(f.path), f.path, options.hash);
        if (!digest.ok()) return digest.status();
        report.hashed.push_back(f.path);
        if (digest.value() != f.sha256) report.mismatches.push_back({f.path, "sha256", f.sha256, digest.value()});
    }
    return report;
}

}  // namespace sonder::inference::weights
