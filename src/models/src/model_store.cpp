// Ollama model-store reader (docs/integration/model-store.md). Read-only.
//
// On-disk format sources (github.com/ollama/ollama, checked 2026-11-26 at main 8ae8acb):
//   manifest/paths.go      manifests-v2/<host>/<ns>/<model>/<tag> (':' in host as %3A), legacy
//                          manifests/..., resolution order (v2 first), digest "sha256[:-]<64 hex>",
//                          blobs/sha256-<hex>, public hosts registry.ollama.ai == ollama.com
//   manifest/manifest.go   Manifest JSON, manifest lists (mediaType
//                          application/vnd.ollama.manifest.list.v2+json), runner preference;
//                          v2 names written as relative symlinks to blobs/sha256-<manifest hex>
//                          except on Windows (WriteManifest, openManifestFileLocked)
//   manifest/layer.go      Layer JSON and the media type constants
//   server/images.go       GetModel(): how each layer type is consumed (model/shards, adapter,
//                          projector, template/prompt, system, params, messages, license)
//   llm/model_split.go     split GGUFs: Ollama hard-links the shard blobs into
//                          blobs/.ollama-split-*/ while it runs one (why weights may have links)
//   types/model/name.go    name parsing, defaults and part validation
//   types/model/config.go  ConfigV2 (model_format, remote_host, ...)
//   api/types.go           Options / Runner JSON keys for the params layer
//
// Every file is opened through common/safe_file.hpp relative to the root directory handle
// OllamaStore::open() pinned: one segment at a time, never following a link or reparse
// point, and read from that one handle. Directories are listed from their own handles.
#include "sonder/inference/model_store.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <system_error>
#include <tuple>

#include "common/safe_file.hpp"
#include "common/sha256.hpp"
#include "store_text.hpp"

namespace sonder::inference::models {

namespace fs = std::filesystem;
namespace sf = ::sonder::inference::detail;

// The store root, opened once by OllamaStore::open() and shared (read-only) by copies of the
// store and by every record read through it. `dir` is only ever the starting directory of a
// relative open, so concurrent use from any number of threads is safe.
struct StoreAnchor {
    sf::OwnedFileHandle dir;
    sf::FileIdentity id;
    fs::path path;  // absolute form of the path given to open(); display and record paths only
};

namespace store_text {

namespace {

// Zero-width, bidi, separator and other invisible format code points that can disguise text.
bool invisible_or_control(char32_t cp) {
    return (cp >= 0x80 && cp <= 0x9F) || cp == 0x061C || (cp >= 0x200B && cp <= 0x200F) || cp == 0x2028 ||
           cp == 0x2029 || (cp >= 0x202A && cp <= 0x202E) || (cp >= 0x2060 && cp <= 0x2069) || cp == 0xFEFF ||
           (cp >= 0xFFF9 && cp <= 0xFFFB);
}

}  // namespace

std::string printable(std::string_view s, std::size_t max) {
    static constexpr char kHex[] = "0123456789abcdef";
    const std::size_t n = std::min(s.size(), max);
    std::string out;
    out.reserve(n + 8);
    auto hex_byte = [&](unsigned char c) {
        out += "\\x";
        out.push_back(kHex[c >> 4]);
        out.push_back(kHex[c & 15]);
    };
    std::size_t i = 0;
    while (i < n) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            if (c < 0x20 || c == 0x7f) {
                hex_byte(c);
            } else if (c == '\\') {
                out += "\\\\";
            } else {
                out.push_back(static_cast<char>(c));
            }
            ++i;
            continue;
        }
        // One UTF-8 sequence inside [i, n) (RFC 3629: no overlongs, surrogates or > U+10FFFF).
        std::size_t len = 0;
        char32_t cp = 0;
        if (c >= 0xC2 && c <= 0xDF) {
            len = 2;
            cp = c & 0x1Fu;
        } else if (c >= 0xE0 && c <= 0xEF) {
            len = 3;
            cp = c & 0x0Fu;
        } else if (c >= 0xF0 && c <= 0xF4) {
            len = 4;
            cp = c & 0x07u;
        }
        bool ok = len != 0 && i + len <= n;
        for (std::size_t k = 1; ok && k < len; ++k) {
            const auto b = static_cast<unsigned char>(s[i + k]);
            if ((b & 0xC0u) != 0x80u) {
                ok = false;
            } else {
                cp = (cp << 6) | (b & 0x3Fu);
            }
        }
        if (ok && ((len == 3 && cp < 0x800) || (len == 4 && cp < 0x10000) || (cp >= 0xD800 && cp <= 0xDFFF) ||
                   cp > 0x10FFFF)) {
            ok = false;
        }
        if (!ok) {
            hex_byte(c);  // its continuation bytes, if any, are escaped one by one after it
            ++i;
            continue;
        }
        for (std::size_t k = 0; k < len; ++k) {
            const auto b = static_cast<unsigned char>(s[i + k]);
            if (invisible_or_control(cp)) {
                hex_byte(b);
            } else {
                out.push_back(static_cast<char>(b));
            }
        }
        i += len;
    }
    if (s.size() > max) out += "...";
    return out;
}

}  // namespace store_text

namespace {

constexpr std::uint64_t kMaxManifestBytes = 4ull << 20;
constexpr std::uint64_t kMaxTextLayerBytes = 4ull << 20;
constexpr std::uint64_t kMaxTextBytesPerManifest = 16ull << 20;
constexpr std::size_t kMaxContentLayers = 64;
constexpr std::size_t kMaxListChildren = 64;
constexpr std::size_t kMaxStopEntries = 64;
constexpr std::size_t kMaxStopBytes = 1024;
constexpr std::size_t kMaxWarnings = 64;
constexpr std::size_t kMaxNotes = 256;
constexpr std::size_t kMaxQuoted = 128;
constexpr std::size_t kHashBufferBytes = 1u << 20;
constexpr std::size_t kMaxVerifiedEntries = 4096;
constexpr std::size_t kMaxLiveRecords = 1024;
constexpr std::size_t kMaxHostLen = 350;
constexpr std::size_t kMaxPartLen = 80;
static_assert(kMaxOllamaNameBytes == 8 + kMaxHostLen + 1 + kMaxPartLen + 1 + kMaxPartLen + 1 + kMaxPartLen);

constexpr std::string_view kMediaManifestList = "application/vnd.ollama.manifest.list.v2+json";
constexpr std::string_view kMediaManifest = "application/vnd.docker.distribution.manifest.v2+json";
constexpr std::string_view kMediaTensor = "application/vnd.ollama.image.tensor";

// Manifests are parsed strictly: a repeated key would let a second "layers" override the first.
constexpr json::ParseLimits kManifestLimits{64, 100000, true};
constexpr json::ParseLimits kLayerLimits{64, 100000, false};

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

// Bounded, single-line form of an untrusted string for messages and notes (store_text.hpp).
std::string printable(std::string_view s, std::size_t max = kMaxQuoted) { return store_text::printable(s, max); }

// Converts without throwing: a Windows name with an unpaired UTF-16 surrogate makes
// generic_u8string throw (same approach as src/artifacts/src/weight_manifest.cpp).
bool try_path_utf8(const fs::path& p, std::string& out) {
    try {
        const auto u = p.generic_u8string();
        out.assign(reinterpret_cast<const char*>(u.data()), u.size());
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::string display_path(const fs::path& p) {
    std::string s;
    return try_path_utf8(p, s) ? s : std::string("<path not representable as UTF-8>");
}

fs::path from_utf8(std::string_view s) { return fs::path(std::u8string(s.begin(), s.end())); }

bool is_alnum_us(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

// Windows maps these base names (with any extension, case-insensitively) to devices.
bool is_dos_device_name(std::string_view s) {
    const std::string base = lower(s.substr(0, s.find_first_of(".:")));
    if (base == "con" || base == "prn" || base == "aux" || base == "nul") return true;
    return base.size() == 4 && (base.compare(0, 3, "com") == 0 || base.compare(0, 3, "lpt") == 0) && base[3] >= '0' &&
           base[3] <= '9';
}

enum class Part { host, ns, model, tag };

// types/model/name.go isValidPart / isValidLen, plus no trailing '.' (Win32 strips it, so
// "coder." would open "coder") and no DOS device base name, for every part including the
// host ("nul", "aux:5000", "com1.example" are refused; the host is a directory name too).
bool valid_part(Part kind, std::string_view s) {
    const std::size_t max_len = kind == Part::host ? kMaxHostLen : kMaxPartLen;
    if (s.empty() || s.size() > max_len) return false;
    for (std::size_t i = 0; i < s.size(); ++i) {
        const char c = s[i];
        if (i == 0) {
            if (!is_alnum_us(c)) return false;
            continue;
        }
        if (c == '_' || c == '-') continue;
        if (c == '.') {
            if (kind == Part::ns) return false;
            continue;
        }
        if (c == ':') {
            if (kind != Part::host) return false;
            continue;
        }
        if (!is_alnum_us(c)) return false;
    }
    if (s.back() == '.') return false;
    return !is_dos_device_name(s);
}

bool is_hex64(std::string_view s) {
    if (s.size() != 64) return false;
    return std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; });
}

// A manifest digest: exactly "sha256:<64 hex>" -> lowercase hex. (Ollama also validates
// "sha256-<hex>", but only ever writes ':' in manifests; '-' is the blob file spelling.)
std::optional<std::string> manifest_digest_hex(std::string_view d) {
    if (d.size() != 7 + 64 || d.substr(0, 7) != "sha256:") return std::nullopt;
    const std::string_view hex = d.substr(7);
    if (!is_hex64(hex)) return std::nullopt;
    return lower(hex);
}

std::string short_digest(std::string_view hex) { return "sha256:" + std::string(hex.substr(0, 12)); }

std::string blob_rel(std::string_view hex) { return "blobs/sha256-" + std::string(hex); }

fs::path blob_path(const fs::path& root, std::string_view hex) { return root / "blobs" / ("sha256-" + std::string(hex)); }

std::string canonical_host(std::string_view host) {
    std::string h = lower(host);
    if (h == "ollama.com") return kOllamaDefaultHost;
    return h;
}

bool is_default_host(std::string_view host) { return canonical_host(host) == kOllamaDefaultHost; }

std::string decode_v2_host(std::string host) {
    // manifest/paths.go decodeV2ManifestHost: %3A / %3a -> ':'
    std::string out;
    for (std::size_t i = 0; i < host.size(); ++i) {
        if (host[i] == '%' && i + 2 < host.size() && host[i + 1] == '3' && (host[i + 2] == 'A' || host[i + 2] == 'a')) {
            out.push_back(':');
            i += 2;
        } else {
            out.push_back(host[i]);
        }
    }
    return out;
}

std::string encode_v2_host(std::string_view host) {
    std::string out;
    for (char c : host) {
        if (c == ':') {
            out += "%3A";
        } else {
            out.push_back(c);
        }
    }
    return out;
}

// Drops "user:password@" from a URL-shaped string before it is shown anywhere.
std::string redact_userinfo(std::string_view url) {
    const auto scheme = url.find("://");
    const std::size_t start = scheme == std::string_view::npos ? 0 : scheme + 3;
    const auto end = url.find_first_of("/?#", start);
    const auto at = url.rfind('@', end == std::string_view::npos ? url.size() : end);
    if (at == std::string_view::npos || at < start) return std::string(url);
    return std::string(url.substr(0, start)) + std::string(url.substr(at + 1));
}

// acc += v; on overflow saturates and returns false.
bool add_checked(std::uint64_t& acc, std::uint64_t v) {
    if (v > std::numeric_limits<std::uint64_t>::max() - acc) {
        acc = std::numeric_limits<std::uint64_t>::max();
        return false;
    }
    acc += v;
    return true;
}

FileStamp to_stamp(const sf::FileIdentity& id) {
    FileStamp s;
    s.volume = id.volume;
    s.file_id = id.file_id;
    s.file_id_high = id.file_id_high;
    s.size = id.size;
    s.link_count = id.link_count;
    s.write_time = id.write_time;
    s.change_time = id.change_time;
    return s;
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

// ---------------------------------------------------------------------------
// Bounded reads from an open handle
// ---------------------------------------------------------------------------
enum class ReadResult { ok, too_large, short_read, io_error };

// Reads the whole file from the handle's position, failing once more than `cap` bytes arrive
// (it asks for cap + 1, so growth during the read is seen). `hint` sizes the first allocation.
ReadResult read_capped(const sf::OwnedFileHandle& h, std::uint64_t cap, std::uint64_t hint, std::string& out,
                       std::string& err) {
    out.assign(static_cast<std::size_t>(std::min(hint, cap) + 1), '\0');
    std::size_t filled = 0;
    while (true) {
        if (filled == out.size()) {
            if (out.size() > cap) break;
            out.resize(static_cast<std::size_t>(std::min<std::uint64_t>(cap + 1, out.size() * 2 + 4096)));
        }
        const std::int64_t got = sf::read_some(h, out.data() + filled, out.size() - filled, err);
        if (got < 0) return ReadResult::io_error;
        if (got == 0) break;
        filled += static_cast<std::size_t>(got);
    }
    if (filled > cap) return ReadResult::too_large;
    out.resize(filled);
    return ReadResult::ok;
}

// Reads exactly `expected` bytes; asks for one more so a file that grew is caught.
ReadResult read_exact(const sf::OwnedFileHandle& h, std::uint64_t expected, std::string& out, std::string& err) {
    const ReadResult r = read_capped(h, expected, expected, out, err);
    if (r != ReadResult::ok) return r;
    return out.size() == expected ? ReadResult::ok : ReadResult::short_read;
}

// Streams the handle through SHA-256 with `buf`; false on a read error or cancellation.
bool hash_handle(const sf::OwnedFileHandle& h, std::vector<char>& buf, std::string& hex, std::uint64_t& bytes,
                 std::string& err, const std::function<bool()>& cancelled) {
    ::sonder::inference::detail::Sha256 sha;
    bytes = 0;
    while (true) {
        if (cancelled && cancelled()) {
            err = "hashing cancelled";
            return false;
        }
        const std::int64_t got = sf::read_some(h, buf.data(), buf.size(), err);
        if (got < 0) return false;
        if (got == 0) break;
        sha.update(std::string_view(buf.data(), static_cast<std::size_t>(got)));
        bytes += static_cast<std::uint64_t>(got);
    }
    hex = to_hex(sha.finish());
    return true;
}

Result<GgufHeader> parse_gguf_header(const std::string& head) {
    if (head.size() < 24) return Status(ErrorCode::invalid_argument, "file is shorter than a GGUF header");
    const auto* buf = reinterpret_cast<const unsigned char*>(head.data());
    if (buf[0] != 'G' || buf[1] != 'G' || buf[2] != 'U' || buf[3] != 'F') {
        return Status(ErrorCode::invalid_argument, "missing GGUF magic");
    }
    auto le32 = [&](std::size_t o) {
        return static_cast<std::uint32_t>(buf[o]) | (static_cast<std::uint32_t>(buf[o + 1]) << 8) |
               (static_cast<std::uint32_t>(buf[o + 2]) << 16) | (static_cast<std::uint32_t>(buf[o + 3]) << 24);
    };
    auto le64 = [&](std::size_t o) {
        return static_cast<std::uint64_t>(le32(o)) | (static_cast<std::uint64_t>(le32(o + 4)) << 32);
    };
    GgufHeader h;
    h.version = le32(4);
    if (h.version == 1) return Status(ErrorCode::unsupported, "GGUF version 1 is no longer supported");
    if (h.version != 2 && h.version != 3) {
        return Status(ErrorCode::unsupported, "unsupported GGUF version " + std::to_string(h.version));
    }
    h.tensor_count = le64(8);
    h.kv_count = le64(16);
    return h;
}

// Reads up to the first 24 bytes from the handle.
Result<GgufHeader> header_from_handle(const sf::OwnedFileHandle& h) {
    std::string head(24, '\0');
    std::size_t filled = 0;
    std::string err;
    while (filled < head.size()) {
        const std::int64_t got = sf::read_some(h, head.data() + filled, head.size() - filled, err);
        if (got < 0) return Status(ErrorCode::io_error, "cannot read: " + err);
        if (got == 0) break;
        filled += static_cast<std::size_t>(got);
    }
    head.resize(filled);
    return parse_gguf_header(head);
}

const char* open_failure_text(sf::OpenOutcome o) {
    switch (o) {
        case sf::OpenOutcome::ok: return "opened";
        case sf::OpenOutcome::missing: return "is not in the store";
        case sf::OpenOutcome::unsafe:
            return "is a link, a junction or other reparse point, not a regular file, or under such a directory";
        case sf::OpenOutcome::io_error: return "cannot be opened";
    }
    return "cannot be opened";
}

// ---------------------------------------------------------------------------
// Record bookkeeping
// ---------------------------------------------------------------------------
void warn(ModelRecord& r, std::string text) {
    if (r.warnings.size() < kMaxWarnings) {
        r.warnings.push_back(std::move(text));
    } else if (r.warnings.size() == kMaxWarnings) {
        r.warnings.push_back("further warnings omitted");
    }
}

void mark(ModelRecord& r, UnsupportedReason reason, std::string detail) {
    if (r.status == EntryStatus::unsupported) {
        warn(r, std::move(detail));  // keep the first reason; later ones are context
        return;
    }
    r.status = EntryStatus::unsupported;
    r.reason = reason;
    r.detail = std::move(detail);
}

void note(std::vector<std::string>& notes, std::string text) {
    if (notes.size() < kMaxNotes) {
        notes.push_back(std::move(text));
    } else if (notes.size() == kMaxNotes) {
        notes.push_back("further skipped entries omitted");
    }
}

// ---------------------------------------------------------------------------
// Manifest parsing
// ---------------------------------------------------------------------------
struct LayerInfo {
    std::string media_type;
    std::string hex;  // empty when the digest is absent or malformed
    std::string raw_digest;
    bool has_size = false;
    std::uint64_t size = 0;
    std::string from;
    std::string name;  // tensor name (tensor layers)
};

struct ParsedManifest {
    std::string media_type;
    bool has_config = false;
    LayerInfo config;
    std::vector<LayerInfo> layers;
    std::string runner;
    std::string format;
};

std::string str_field(const json::Value& v, std::string_view key) {
    const auto* f = v.find(key);
    return (f && f->is_string()) ? f->as_string() : std::string();
}

// A non-negative JSON integer (no fraction, no exponent form) up to UINT64_MAX.
bool json_size(const json::Value& v, std::uint64_t& out) {
    if (!v.is_integer()) return false;
    if (const auto i = v.try_as_int()) {
        if (*i < 0) return false;
        out = static_cast<std::uint64_t>(*i);
        return true;
    }
    out = v.as_uint(0);  // integers above INT64_MAX
    return true;
}

bool parse_layer(const json::Value& v, LayerInfo& l, std::string& error) {
    if (!v.is_object()) {
        error = "manifest layer is not an object";
        return false;
    }
    if (const auto* m = v.find("mediaType")) {
        if (!m->is_string()) {
            error = "layer mediaType is not a string";
            return false;
        }
        l.media_type = m->as_string();
    }
    if (const auto* d = v.find("digest")) {
        if (!d->is_string()) {
            error = "layer digest is not a string";
            return false;
        }
        l.raw_digest = d->as_string();
        if (auto hex = manifest_digest_hex(l.raw_digest)) l.hex = *hex;
    }
    if (const auto* s = v.find("size")) {
        if (!json_size(*s, l.size)) {
            error = "layer '" + printable(l.media_type) + "' size is not a non-negative integer";
            return false;
        }
        l.has_size = true;
    }
    if (const auto* f = v.find("from"); f && f->is_string()) l.from = f->as_string();
    if (const auto* n = v.find("name"); n && n->is_string()) l.name = n->as_string();
    return true;
}

bool is_content_layer(std::string_view media_type) {
    return media_type != kMediaTensor && media_type != kMediaManifest && media_type != kMediaManifestList;
}

// A manifest object (top-level or an inline manifest-list child).
bool parse_manifest_object(const json::Value& v, ParsedManifest& out, std::string& error) {
    if (!v.is_object()) {
        error = "manifest is not a JSON object";
        return false;
    }
    out.media_type = str_field(v, "mediaType");
    out.runner = lower(str_field(v, "runner"));
    out.format = lower(str_field(v, "format"));
    if (const auto* cfg = v.find("config"); cfg && !cfg->is_null()) {
        if (!parse_layer(*cfg, out.config, error)) {
            error = "manifest config: " + error;
            return false;
        }
        out.has_config = !out.config.raw_digest.empty();
    }
    // Ollama marshals an empty Go slice as null: cloud stubs on disk carry "layers": null.
    const auto* layers = v.find("layers");
    if (layers != nullptr && !layers->is_null() && !layers->is_array()) {
        error = "manifest 'layers' is not an array";
        return false;
    }
    if (layers == nullptr || layers->is_null()) return true;
    std::size_t content = 0;
    for (const auto& l : layers->as_array()) {
        LayerInfo info;
        if (!parse_layer(l, info, error)) return false;
        if (is_content_layer(info.media_type) && ++content > kMaxContentLayers) {
            error = "manifest has more than " + std::to_string(kMaxContentLayers) + " layers";
            return false;
        }
        out.layers.push_back(std::move(info));
    }
    return true;
}

BlobRole role_for(std::string_view media_type) {
    if (media_type == "application/vnd.ollama.image.model") return BlobRole::model;
    if (media_type == "application/vnd.ollama.image.projector") return BlobRole::projector;
    if (media_type == "application/vnd.ollama.image.adapter") return BlobRole::adapter;
    if (media_type == "application/vnd.ollama.image.draft") return BlobRole::draft;
    if (media_type == "application/vnd.ollama.image.template" || media_type == "application/vnd.ollama.image.prompt") {
        return BlobRole::tmpl;
    }
    if (media_type == "application/vnd.ollama.image.system") return BlobRole::system;
    if (media_type == "application/vnd.ollama.image.params") return BlobRole::params;
    if (media_type == "application/vnd.ollama.image.messages") return BlobRole::messages;
    if (media_type == "application/vnd.ollama.image.license") return BlobRole::license;
    return BlobRole::other;
}

bool is_weight_role(BlobRole r) {
    return r == BlobRole::model || r == BlobRole::projector || r == BlobRole::adapter || r == BlobRole::draft;
}

// Param keys modelled by ModelParams (api/types.go Options + Runner JSON tags).
bool as_int_option(const json::Value& v, std::int64_t& out) {
    if (!v.is_number()) return false;
    if (auto i = v.try_as_int()) {
        out = *i;
        return true;
    }
    // Ollama decodes options as float64 and converts with int(f), which truncates toward zero.
    const double d = v.as_double(0.0);
    if (d != d || d < -9.0e18 || d > 9.0e18) return false;
    out = static_cast<std::int64_t>(d);
    return true;
}

void parse_params(const json::Value& doc, ModelParams& p, ModelRecord& r) {
    if (!doc.is_object()) {
        warn(r, "params layer is not a JSON object; ignored");
        return;
    }
    struct IntKey {
        const char* key;
        std::optional<std::int64_t> ModelParams::*field;
    };
    struct DblKey {
        const char* key;
        std::optional<double> ModelParams::*field;
    };
    static const IntKey kInts[] = {{"num_ctx", &ModelParams::num_ctx},
                                   {"num_batch", &ModelParams::num_batch},
                                   {"num_gpu", &ModelParams::num_gpu},
                                   {"num_thread", &ModelParams::num_thread},
                                   {"num_keep", &ModelParams::num_keep},
                                   {"seed", &ModelParams::seed},
                                   {"num_predict", &ModelParams::num_predict},
                                   {"top_k", &ModelParams::top_k},
                                   {"repeat_last_n", &ModelParams::repeat_last_n}};
    static const DblKey kDbls[] = {{"temperature", &ModelParams::temperature},
                                   {"top_p", &ModelParams::top_p},
                                   {"min_p", &ModelParams::min_p},
                                   {"typical_p", &ModelParams::typical_p},
                                   {"repeat_penalty", &ModelParams::repeat_penalty},
                                   {"presence_penalty", &ModelParams::presence_penalty},
                                   {"frequency_penalty", &ModelParams::frequency_penalty}};
    for (const auto& [key, value] : doc.as_object()) {
        bool handled = false;
        for (const auto& k : kInts) {
            if (key != k.key) continue;
            handled = true;
            std::int64_t n = 0;
            if (value.is_null()) break;
            if (as_int_option(value, n)) {
                p.*(k.field) = n;
            } else {
                warn(r, "params option '" + printable(key) + "' is not an integer; ignored");
            }
            break;
        }
        if (handled) continue;
        for (const auto& k : kDbls) {
            if (key != k.key) continue;
            handled = true;
            if (value.is_null()) break;
            if (value.is_number()) {
                p.*(k.field) = value.as_double(0.0);
            } else {
                warn(r, "params option '" + printable(key) + "' is not a number; ignored");
            }
            break;
        }
        if (handled) continue;
        if (key == "stop") {
            // A later params layer replaces "stop" (Ollama decodes each layer into one map).
            p.stop.clear();
            // False once the entry cap is reached (after warning once).
            auto add = [&](const std::string& s) {
                if (p.stop.size() >= kMaxStopEntries) {
                    warn(r, "params 'stop' has more than " + std::to_string(kMaxStopEntries) + " entries; the rest are ignored");
                    return false;
                }
                if (s.size() > kMaxStopBytes) {
                    warn(r, "params 'stop' entry longer than " + std::to_string(kMaxStopBytes) + " bytes; ignored");
                } else {
                    p.stop.push_back(s);
                }
                return true;
            };
            if (value.is_string()) {
                add(value.as_string());
            } else if (value.is_array()) {
                for (const auto& s : value.as_array()) {
                    if (!s.is_string()) {
                        warn(r, "params 'stop' holds a non-string entry; ignored");
                        continue;
                    }
                    if (!add(s.as_string())) break;
                }
            } else if (!value.is_null()) {
                warn(r, "params option 'stop' is not an array of strings; ignored");
            }
            continue;
        }
        p.extra.set(key, value);
    }
}

void parse_config(const json::Value& doc, ModelConfig& c) {
    if (!doc.is_object()) return;
    c.model_format = lower(str_field(doc, "model_format"));
    c.model_family = str_field(doc, "model_family");
    c.parameter_size = str_field(doc, "model_type");
    c.quantization = str_field(doc, "file_type");
    c.renderer = str_field(doc, "renderer");
    c.parser = str_field(doc, "parser");
    c.requires_version = str_field(doc, "requires");
    c.remote_host = str_field(doc, "remote_host");
    c.remote_model = str_field(doc, "remote_model");
    auto strings = [&](std::string_view key, std::vector<std::string>& dst) {
        if (const auto* a = doc.find(key); a && a->is_array()) {
            for (const auto& s : a->as_array()) {
                if (s.is_string()) dst.push_back(s.as_string());
            }
        }
    };
    strings("model_families", c.model_families);
    strings("capabilities", c.capabilities);
    if (const auto* v = doc.find("context_length")) c.context_length = v->as_int(0);
    if (const auto* v = doc.find("embedding_length")) c.embedding_length = v->as_int(0);
}

// ---------------------------------------------------------------------------
// Reading one record
// ---------------------------------------------------------------------------

// The manifest file as read, or why it could not be.
struct ManifestText {
    sf::OpenOutcome outcome = sf::OpenOutcome::missing;
    UnsupportedReason reason = UnsupportedReason::none;
    std::string error;
    std::string text;
    std::vector<SourceStamp> sources;
};

// The blob a v2 manifest symlink may stand for: its text must be a relative path that,
// resolved lexically from the link's own directory (every directory on the way was opened as
// a real directory, so ".." there means its parent), is exactly blobs/sha256-<64 lowercase
// hex> under the store root. Anything else (absolute, "." or empty segments, climbing above
// the root, another directory, another name) is refused without looking at the target.
std::optional<std::string> manifest_link_blob_hex(std::string_view link_rel, std::string_view text) {
    const auto dir_end = link_rel.rfind('/');
    if (dir_end == std::string_view::npos || text.empty() || text.front() == '/') return std::nullopt;
    std::vector<std::string_view> parts;
    auto split_into = [&](std::string_view path, bool allow_dotdot) {
        std::size_t pos = 0;
        while (pos <= path.size()) {
            std::size_t next = path.find('/', pos);
            if (next == std::string_view::npos) next = path.size();
            const std::string_view seg = path.substr(pos, next - pos);
            if (seg.empty() || seg == ".") return false;
            if (seg == "..") {
                if (!allow_dotdot || parts.empty()) return false;
                parts.pop_back();
            } else {
                parts.push_back(seg);
            }
            pos = next + 1;
        }
        return true;
    };
    if (!split_into(link_rel.substr(0, dir_end), false) || !split_into(text, true)) return std::nullopt;
    if (parts.size() != 2 || parts[0] != "blobs") return std::nullopt;
    const std::string_view name = parts[1];
    if (name.size() != 7 + 64 || name.substr(0, 7) != "sha256-") return std::nullopt;
    const std::string_view hex = name.substr(7);
    const bool lower_hex = std::all_of(hex.begin(), hex.end(), [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
    if (!lower_hex) return std::nullopt;
    return std::string(hex);
}

// Opens `rel` relative to the pinned root without following links. On POSIX a manifests-v2
// name may be a symlink (Ollama writes them as ../../../../blobs/sha256-<hex>,
// manifest.go WriteManifest). Then the link is never followed or stat'ed through: its text
// (readlinkat) must be exactly the relative path of blobs/sha256-<hex>, that blob is opened
// from the root, the link is lstat'ed again (same identity, same text) so a retarget during
// the read is caught, and the blob must hash to <hex> (manifest.go openManifestFileLocked).
// A link under the legacy manifests/ tree is refused.
ManifestText read_manifest(const StoreAnchor& a, const std::string& rel, ManifestLayout layout) {
    ManifestText m;
    sf::OwnedFileHandle h;
    sf::FileIdentity id;
    std::string err;
    std::string expect_hex;
    std::string read_rel = rel;
    m.outcome = sf::open_anchored_file(a.dir, rel, sf::ShareMode::allow_writers, h, id, err);
    if (m.outcome == sf::OpenOutcome::unsafe && layout == ManifestLayout::v2) {
        std::string target, link_err;
        sf::FileIdentity link_id;
        if (sf::read_anchored_leaf_symlink(a.dir, rel, target, link_id, link_err)) {
            const auto hex = manifest_link_blob_hex(rel, target);
            if (!hex) {
                m.reason = UnsupportedReason::unsafe_path;
                m.error = "manifest is a symbolic link whose text is not the relative path of a blob in blobs/";
                return m;
            }
            read_rel = blob_rel(*hex);
            sf::OwnedFileHandle bh;
            sf::FileIdentity bid;
            if (sf::open_anchored_file(a.dir, read_rel, sf::ShareMode::allow_writers, bh, bid, err) != sf::OpenOutcome::ok) {
                m.reason = UnsupportedReason::unsafe_path;
                m.error = "manifest symbolic link names " + short_digest(*hex) + ", which is not a regular file in blobs/";
                return m;
            }
            std::string target_again;
            sf::FileIdentity link_again;
            if (!sf::read_anchored_leaf_symlink(a.dir, rel, target_again, link_again, link_err) || target_again != target ||
                link_again != link_id) {
                m.reason = UnsupportedReason::unsafe_path;
                m.error = "manifest symbolic link changed while it was read";
                return m;
            }
            m.sources.push_back(SourceStamp{rel, to_stamp(link_id), true});
            h = std::move(bh);
            id = bid;
            expect_hex = *hex;
            m.outcome = sf::OpenOutcome::ok;
        }
    }
    switch (m.outcome) {
        case sf::OpenOutcome::ok: break;
        case sf::OpenOutcome::missing: return m;
        case sf::OpenOutcome::unsafe:
            m.reason = UnsupportedReason::unsafe_path;
            m.error = std::string("manifest ") + open_failure_text(m.outcome);
            return m;
        case sf::OpenOutcome::io_error:
            m.reason = UnsupportedReason::manifest_invalid;
            m.error = "manifest cannot be opened: " + printable(err);
            return m;
    }
    if (id.link_count > 1) {
        m.outcome = sf::OpenOutcome::unsafe;
        m.reason = UnsupportedReason::unsafe_path;
        m.error = "manifest has " + std::to_string(id.link_count) + " hard links";
        return m;
    }
    switch (read_capped(h, kMaxManifestBytes, id.size, m.text, err)) {
        case ReadResult::ok:
        case ReadResult::short_read: break;
        case ReadResult::too_large:
            m.reason = UnsupportedReason::manifest_invalid;
            m.error = "manifest is larger than " + std::to_string(kMaxManifestBytes) + " bytes";
            return m;
        case ReadResult::io_error:
            m.reason = UnsupportedReason::manifest_invalid;
            m.error = "cannot read manifest: " + printable(err);
            return m;
    }
    if (!expect_hex.empty() && detail::sha256_hex(m.text) != expect_hex) {
        m.reason = UnsupportedReason::manifest_invalid;
        m.error = "manifest symbolic link target does not match its blob name " + short_digest(expect_hex);
        return m;
    }
    m.sources.push_back(SourceStamp{read_rel, to_stamp(id), false});
    return m;
}

// Per-record reader: opens each blob through the anchored no-follow path, deduplicated by
// digest, and enforces the per-manifest text budget.
class RecordReader {
public:
    RecordReader(const StoreAnchor& a, ModelRecord& r) : a_(a), r_(r) {}

    // A validated small blob (config, template, system, params, messages, list child): a
    // regular file with one link, `declared` bytes (when known), at most `cap`, and hashing
    // to `hex`. Marks the record and returns nullopt otherwise.
    std::optional<std::string> small(const std::string& hex, std::optional<std::uint64_t> declared, std::uint64_t cap,
                                     const std::string& what, BlobRef* ref) {
        auto it = small_.find(hex);
        if (it == small_.end()) it = small_.emplace(hex, read_small(hex, declared, cap, what)).first;
        const SmallResult& res = it->second;
        if (ref != nullptr) {
            ref->disk_size = res.disk_size;
            ref->stamp = res.stamp;
        }
        if (res.reason != UnsupportedReason::none) {
            mark(r_, res.reason, res.detail);
            return std::nullopt;
        }
        if (declared && *declared != res.body.size()) {
            mark(r_, UnsupportedReason::blob_size_mismatch,
                 what + " blob " + short_digest(hex) + " is " + std::to_string(res.body.size()) +
                     " bytes, manifest says " + std::to_string(*declared));
            return std::nullopt;
        }
        return res.body;
    }

    // Opens a weight blob: size from the handle and the GGUF header read from that handle.
    void weight(BlobRef& b, const std::string& hex) {
        auto it = weights_.find(hex);
        if (it == weights_.end()) it = weights_.emplace(hex, check_weight(hex)).first;
        const WeightResult& w = it->second;
        b.disk_size = w.disk_size;
        b.stamp = w.stamp;
        if (w.reason != UnsupportedReason::none) {
            mark(r_, w.reason, std::string(to_string(b.role)) + " blob " + short_digest(hex) + " " + w.detail);
            return;
        }
        if (*w.disk_size != b.declared_size) {
            mark(r_, UnsupportedReason::blob_size_mismatch,
                 std::string(to_string(b.role)) + " blob " + short_digest(hex) + " is " + std::to_string(*w.disk_size) +
                     " bytes on disk, manifest says " + std::to_string(b.declared_size));
            return;
        }
        if (!w.header) {
            mark(r_, UnsupportedReason::invalid_gguf, std::string(to_string(b.role)) + " blob " + short_digest(hex) + ": " +
                                                          w.header_error);
            return;
        }
        if (w.links > 1 && !w.link_warned) {
            it->second.link_warned = true;
            warn(r_, std::string(to_string(b.role)) + " blob " + short_digest(hex) + " has " + std::to_string(w.links) +
                         " hard links (Ollama links split-model shards while it runs one); only a SHA-256 match vouches for it");
        }
        if (b.role == BlobRole::model && b.path == r_.model_path) r_.gguf = *w.header;
    }

    // Presence and identity only (license layers are counted, never read).
    void stat_only(BlobRef& b, const std::string& hex) {
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string err;
        if (sf::open_anchored_file(a_.dir, blob_rel(hex), sf::ShareMode::allow_writers, h, id, err) ==
            sf::OpenOutcome::ok) {
            b.disk_size = id.size;
            b.stamp = to_stamp(id);
        }
    }

private:
    struct SmallResult {
        UnsupportedReason reason = UnsupportedReason::none;
        std::string detail;
        std::string body;
        std::optional<std::uint64_t> disk_size;
        std::optional<FileStamp> stamp;
    };
    struct WeightResult {
        UnsupportedReason reason = UnsupportedReason::none;
        std::string detail;
        std::optional<std::uint64_t> disk_size;
        std::optional<FileStamp> stamp;
        std::optional<GgufHeader> header;
        std::string header_error;
        std::uint64_t links = 1;
        bool link_warned = false;
    };

    SmallResult read_small(const std::string& hex, std::optional<std::uint64_t> declared, std::uint64_t cap,
                           const std::string& what) {
        SmallResult res;
        const std::string label = what + " blob " + short_digest(hex);
        auto fail = [&](UnsupportedReason reason, std::string text) {
            res.reason = reason;
            res.detail = label + " " + text;
            return res;
        };
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string err;
        const sf::OpenOutcome o = sf::open_anchored_file(a_.dir, blob_rel(hex), sf::ShareMode::allow_writers, h, id, err);
        if (o == sf::OpenOutcome::missing) return fail(UnsupportedReason::missing_blob, "is not in blobs/");
        if (o == sf::OpenOutcome::unsafe) return fail(UnsupportedReason::unsafe_path, open_failure_text(o));
        if (o == sf::OpenOutcome::io_error) return fail(UnsupportedReason::missing_blob, "cannot be opened: " + printable(err));
        res.disk_size = id.size;
        res.stamp = to_stamp(id);
        if (id.link_count > 1) {
            return fail(UnsupportedReason::unsafe_path,
                        "has " + std::to_string(id.link_count) + " hard links; small layers must have one");
        }
        const std::uint64_t expect = declared.value_or(id.size);
        if (expect > cap) {
            return fail(UnsupportedReason::manifest_invalid, "is larger than the " + std::to_string(cap) + "-byte cap");
        }
        if (id.size != expect) {
            return fail(UnsupportedReason::blob_size_mismatch, "is " + std::to_string(id.size) +
                                                                   " bytes on disk, manifest says " + std::to_string(expect));
        }
        if (text_bytes_ + expect > kMaxTextBytesPerManifest) {
            return fail(UnsupportedReason::manifest_invalid,
                        "would take this manifest's small layers past " + std::to_string(kMaxTextBytesPerManifest) + " bytes");
        }
        text_bytes_ += expect;
        switch (read_exact(h, expect, res.body, err)) {
            case ReadResult::ok: break;
            case ReadResult::io_error: return fail(UnsupportedReason::missing_blob, "cannot be read: " + printable(err));
            case ReadResult::too_large:
            case ReadResult::short_read:
                res.body.clear();
                return fail(UnsupportedReason::blob_size_mismatch, "changed size while it was read");
        }
        if (detail::sha256_hex(res.body) != hex) {
            res.body.clear();
            return fail(UnsupportedReason::digest_mismatch, "does not hash to its digest");
        }
        return res;
    }

    WeightResult check_weight(const std::string& hex) {
        WeightResult w;
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string err;
        const sf::OpenOutcome o = sf::open_anchored_file(a_.dir, blob_rel(hex), sf::ShareMode::allow_writers, h, id, err);
        if (o != sf::OpenOutcome::ok) {
            w.reason = o == sf::OpenOutcome::unsafe ? UnsupportedReason::unsafe_path : UnsupportedReason::missing_blob;
            w.detail = o == sf::OpenOutcome::missing ? "is not in blobs/" : open_failure_text(o);
            if (o == sf::OpenOutcome::io_error) w.detail += ": " + printable(err);
            return w;
        }
        w.disk_size = id.size;
        w.stamp = to_stamp(id);
        w.links = id.link_count;
        auto header = header_from_handle(h);
        if (header.ok()) {
            w.header = header.value();
        } else {
            w.header_error = header.status().message();
        }
        return w;
    }

    const StoreAnchor& a_;
    ModelRecord& r_;
    std::uint64_t text_bytes_ = 0;
    std::map<std::string, SmallResult> small_;
    std::map<std::string, WeightResult> weights_;
};

// Selects the manifest-list child this engine can use: llamacpp first, then ggml (the same
// order as manifest/manifest.go runnerPreferences on non-Apple platforms, minus mlx).
// Returns false and sets reason/detail when none fits.
bool select_child(const json::Value& list, const StoreAnchor& a, ModelRecord& r, json::Value& child_doc,
                  UnsupportedReason& reason, std::string& detail_text) {
    const auto* children = list.find("manifests");
    if (children == nullptr || !children->is_array()) {
        reason = UnsupportedReason::manifest_invalid;
        detail_text = "manifest list has no manifests array";
        return false;
    }
    if (children->as_array().size() > kMaxListChildren) {
        reason = UnsupportedReason::manifest_invalid;
        detail_text = "manifest list has more than " + std::to_string(kMaxListChildren) + " children";
        return false;
    }
    bool saw_mlx = false;
    // A reference whose blob is not local is skipped like Ollama does (ok=false, next child);
    // the first such failure is reported only if no later child fits.
    UnsupportedReason skipped_reason = UnsupportedReason::none;
    std::string skipped_detail;
    for (const char* runner : {"llamacpp", "ggml"}) {
        for (const auto& c : children->as_array()) {
            if (!c.is_object()) continue;
            const std::string mt = str_field(c, "mediaType");
            if (!mt.empty() && mt != kMediaManifest) continue;
            if (lower(str_field(c, "runner")) != runner) continue;
            const std::string fmt = lower(str_field(c, "format"));
            if (!fmt.empty() && fmt != "gguf") continue;
            // manifest.go isReference: digest set, no config digest, no layers. Anything else
            // with a config or layers is an inline child.
            const auto* cfg = c.find("config");
            const bool has_config_digest = cfg != nullptr && cfg->is_object() && !str_field(*cfg, "digest").empty();
            const auto* inline_layers = c.find("layers");
            const bool has_layers = inline_layers != nullptr && inline_layers->is_array() && !inline_layers->as_array().empty();
            const bool is_reference = !str_field(c, "digest").empty() && !has_config_digest && !has_layers;
            if (!is_reference && (inline_layers != nullptr || has_config_digest)) {  // inline child
                child_doc = c;
                return true;
            }
            const auto hex = manifest_digest_hex(str_field(c, "digest"));
            if (!hex) {
                reason = UnsupportedReason::manifest_invalid;
                detail_text = "manifest list child has a malformed digest";
                return false;
            }
            // Missing: skip to the next child. Anything else that is wrong with it is final.
            {
                sf::OwnedFileHandle probe;
                sf::FileIdentity pid;
                std::string perr;
                if (sf::open_anchored_file(a.dir, blob_rel(*hex), sf::ShareMode::allow_writers, probe, pid, perr) ==
                    sf::OpenOutcome::missing) {
                    if (skipped_reason == UnsupportedReason::none) {
                        skipped_reason = UnsupportedReason::missing_blob;
                        skipped_detail = "manifest list child " + short_digest(*hex) + " is not in blobs/";
                    }
                    continue;
                }
            }
            ModelRecord probe_record;
            RecordReader child_reader(a, probe_record);
            BlobRef ref;
            auto text = child_reader.small(*hex, std::nullopt, kMaxManifestBytes, "manifest list child", &ref);
            if (!text) {
                reason = probe_record.reason == UnsupportedReason::digest_mismatch ? UnsupportedReason::manifest_invalid
                                                                                    : probe_record.reason;
                detail_text = probe_record.detail;
                return false;
            }
            if (ref.stamp) r.sources.push_back(SourceStamp{blob_rel(*hex), *ref.stamp, false});
            auto parsed = json::parse(*text, kManifestLimits);
            if (!parsed.ok()) {
                reason = UnsupportedReason::manifest_invalid;
                detail_text = "manifest list child is not valid JSON: " + parsed.status().message();
                return false;
            }
            child_doc = std::move(parsed).value();
            // The reference carries runner/format that the child blob may omit (manifest.go).
            if (child_doc.is_object()) {
                if (child_doc.find("runner") == nullptr) child_doc.object().set("runner", runner);
            }
            return true;
        }
    }
    if (skipped_reason != UnsupportedReason::none) {
        reason = skipped_reason;
        detail_text = std::move(skipped_detail);
        return false;
    }
    for (const auto& c : children->as_array()) {
        if (c.is_object() && (lower(str_field(c, "runner")) == "mlx" || lower(str_field(c, "format")) == "safetensors")) {
            saw_mlx = true;
        }
    }
    if (saw_mlx) {
        reason = UnsupportedReason::safetensors;
        detail_text = "manifest list only offers safetensors/MLX weights; SI's native engine loads GGUF";
    } else {
        reason = UnsupportedReason::no_compatible_manifest;
        detail_text = "manifest list has no llamacpp or ggml GGUF child";
    }
    return false;
}

ModelRecord build_record(const std::shared_ptr<const StoreAnchor>& anchor, const OllamaName& name, const std::string& rel,
                         ManifestLayout layout, ManifestText&& mt) {
    const StoreAnchor& a = *anchor;
    const fs::path& root = a.path;
    ModelRecord r;
    r.name = name;
    r.store_root = root;
    r.anchor = anchor;
    r.manifest_path = root / from_utf8(rel);
    r.layout = layout;
    r.sources = std::move(mt.sources);
    if (mt.outcome != sf::OpenOutcome::ok || mt.reason != UnsupportedReason::none) {
        mark(r, mt.reason == UnsupportedReason::none ? UnsupportedReason::manifest_invalid : mt.reason,
             mt.error.empty() ? std::string("manifest ") + open_failure_text(mt.outcome) : mt.error);
        return r;
    }
    RecordReader reader(a, r);

    auto doc = json::parse(mt.text, kManifestLimits);
    if (!doc.ok()) {
        mark(r, UnsupportedReason::manifest_invalid, "manifest is not valid JSON: " + doc.status().message());
        return r;
    }
    json::Value manifest_doc = std::move(doc).value();
    if (!manifest_doc.is_object()) {
        mark(r, UnsupportedReason::manifest_invalid, "manifest is not a JSON object");
        return r;
    }
    if (str_field(manifest_doc, "mediaType") == kMediaManifestList) {
        json::Value child;
        UnsupportedReason reason = UnsupportedReason::none;
        std::string why;
        if (!select_child(manifest_doc, a, r, child, reason, why)) {
            mark(r, reason, why);
            return r;
        }
        manifest_doc = std::move(child);
    }

    ParsedManifest pm;
    std::string err;
    if (!parse_manifest_object(manifest_doc, pm, err)) {
        mark(r, UnsupportedReason::manifest_invalid, err);
        return r;
    }
    r.runner = pm.runner;
    r.format = pm.format;

    bool overflow = false;
    // Config blob (model format, cloud stub marker, renderer/parser).
    if (pm.has_config) {
        if (pm.config.hex.empty() || !pm.config.has_size) {
            mark(r, UnsupportedReason::manifest_invalid,
                 pm.config.hex.empty() ? "config layer has a malformed digest '" + printable(pm.config.raw_digest) + "'"
                                       : std::string("config layer has no size"));
            return r;
        }
        BlobRef cfg;
        cfg.role = BlobRole::config;
        cfg.media_type = pm.config.media_type;
        cfg.digest = "sha256:" + pm.config.hex;
        cfg.path = blob_path(root, pm.config.hex);
        cfg.declared_size = pm.config.size;
        if (auto text = reader.small(pm.config.hex, pm.config.size, kMaxTextLayerBytes, "config", &cfg)) {
            auto cj = json::parse(*text, kLayerLimits);
            if (cj.ok()) {
                parse_config(cj.value(), r.config);
            } else {
                warn(r, "config blob is not valid JSON");
            }
        }
        overflow = !add_checked(r.referenced_bytes, cfg.declared_size) || overflow;
        r.blobs.push_back(std::move(cfg));
    }
    if (r.format.empty()) r.format = r.config.model_format;

    bool saw_tensor = false;
    for (const auto& l : pm.layers) {
        if (l.media_type == kMediaTensor) {
            saw_tensor = true;
            continue;
        }
        if (l.media_type == kMediaManifest || l.media_type == kMediaManifestList) {
            continue;  // downgrade anchor (manifest.go WriteLegacyAnchor), not content
        }
        if (l.hex.empty()) {
            mark(r, UnsupportedReason::manifest_invalid,
                 "layer '" + printable(l.media_type) + "' has a malformed digest '" + printable(l.raw_digest) + "'");
            continue;
        }
        const BlobRole role = role_for(l.media_type);
        if (role == BlobRole::other) {
            if (l.media_type == "application/vnd.ollama.image.embed") {
                warn(r, "embed layer is deprecated in Ollama and ignored");
            } else if (l.media_type != "application/vnd.ollama.image.json") {
                warn(r, "unrecognized layer media type '" + printable(l.media_type) + "' ignored");
            }
            continue;
        }
        if (!l.has_size) {
            mark(r, UnsupportedReason::manifest_invalid, "layer '" + printable(l.media_type) + "' has no size");
            continue;
        }
        BlobRef blob;
        blob.role = role;
        blob.media_type = l.media_type;
        blob.digest = "sha256:" + l.hex;
        blob.path = blob_path(root, l.hex);
        blob.declared_size = l.size;
        blob.from = l.from;
        overflow = !add_checked(r.referenced_bytes, blob.declared_size) || overflow;
        if (is_weight_role(role)) overflow = !add_checked(r.weights_bytes, blob.declared_size) || overflow;
        switch (role) {
            case BlobRole::model:
                if (r.model_path.empty()) {
                    r.model_path = blob.path;
                    r.parent = l.from;
                } else {
                    r.shard_paths.push_back(blob.path);
                }
                break;
            case BlobRole::projector:
                r.projector_paths.push_back(blob.path);
                break;
            case BlobRole::adapter:
                r.adapter_paths.push_back(blob.path);
                break;
            case BlobRole::draft:
                r.draft_paths.push_back(blob.path);
                break;
            case BlobRole::tmpl:
                if (auto body = reader.small(l.hex, l.size, kMaxTextLayerBytes, "template", &blob)) {
                    r.template_text = std::move(*body);
                }
                break;
            case BlobRole::system:
                if (auto body = reader.small(l.hex, l.size, kMaxTextLayerBytes, "system", &blob)) {
                    r.system_prompt = std::move(*body);
                }
                break;
            case BlobRole::params:
                if (auto body = reader.small(l.hex, l.size, kMaxTextLayerBytes, "params", &blob); body && !body->empty()) {
                    auto pj = json::parse(*body, kLayerLimits);
                    if (pj.ok()) {
                        parse_params(pj.value(), r.params, r);
                    } else {
                        warn(r, "params layer is not valid JSON; ignored");
                    }
                }
                break;
            case BlobRole::messages:
                if (auto body = reader.small(l.hex, l.size, kMaxTextLayerBytes, "messages", &blob); body && !body->empty()) {
                    if (json::parse(*body, kLayerLimits).ok()) {
                        r.messages_json = std::move(*body);
                    } else {
                        warn(r, "messages layer is not valid JSON; ignored");
                    }
                }
                break;
            case BlobRole::license:
                ++r.license_count;
                reader.stat_only(blob, l.hex);
                break;
            default:
                break;
        }
        r.blobs.push_back(std::move(blob));
    }
    if (overflow) mark(r, UnsupportedReason::manifest_invalid, "declared layer sizes add up to more than 2^64 bytes");

    // Classification, most specific first.
    if (!r.config.remote_host.empty() || !r.config.remote_model.empty()) {
        mark(r, UnsupportedReason::cloud_stub,
             "Ollama Cloud entry (remote model '" + printable(r.config.remote_model) +
                 "'); it has no local weights and is reached over HTTPS, not served from the store");
    }
    if (r.status == EntryStatus::ready &&
        (r.config.model_format == "safetensors" || r.format == "safetensors" || r.runner == "mlx" || saw_tensor)) {
        mark(r, UnsupportedReason::safetensors, "safetensors/MLX weights; SI's native engine loads GGUF only");
    }
    if (r.status == EntryStatus::ready && !r.format.empty() && r.format != "gguf") {
        mark(r, UnsupportedReason::unsupported_format, "weight format '" + printable(r.format) + "' is not GGUF");
    }
    if (r.status == EntryStatus::ready && r.model_path.empty()) {
        mark(r, UnsupportedReason::no_model_layer, "manifest has no image.model layer");
    }
    if (r.status == EntryStatus::ready) {
        for (auto& b : r.blobs) {
            if (!is_weight_role(b.role)) continue;
            reader.weight(b, b.digest.substr(7));
            if (r.status != EntryStatus::ready) break;
        }
    }
    return r;
}

using EntryKind = sf::DirEntryKind;

// Lists `dir_rel` from its own handle, opened relative to the pinned root: each entry's UTF-8
// name and kind (never following an entry). Entries whose names are not representable as
// UTF-8 are noted and skipped. `outcome` (optional) receives the open result.
struct Listed {
    std::string name;
    EntryKind kind;
};

std::vector<Listed> list_rel(const StoreAnchor& a, const std::string& dir_rel, std::vector<std::string>* notes,
                             sf::OpenOutcome* outcome = nullptr) {
    std::vector<Listed> out;
    std::vector<sf::DirEntry> entries;
    std::string err;
    const sf::OpenOutcome o = sf::list_anchored_dir(a.dir, dir_rel, entries, err);
    if (outcome != nullptr) *outcome = o;
    if (o != sf::OpenOutcome::ok) return out;
    out.reserve(entries.size());
    for (auto& e : entries) {
        if (!e.name_ok) {
            if (notes) note(*notes, "skipped an entry in " + printable(dir_rel) + ": its name is not representable as UTF-8");
            continue;
        }
        out.push_back(Listed{std::move(e.name), e.kind});
    }
    return out;
}

// True when the manifest tree `tree` is a real directory under the root (opened without
// following a link); a link, junction or non-directory there is noted and skipped.
bool tree_is_safe(const StoreAnchor& a, const char* tree, std::vector<std::string>* notes) {
    std::string err;
    switch (sf::probe_anchored_dir(a.dir, tree, err)) {
        case sf::OpenOutcome::ok: return true;
        case sf::OpenOutcome::unsafe:
            if (notes) note(*notes, std::string("skipped ") + tree + ": a link, reparse point or not a directory");
            return false;
        case sf::OpenOutcome::missing:
        case sf::OpenOutcome::io_error: return false;
    }
    return false;
}

bool valid_name(const OllamaName& n) {
    return valid_part(Part::host, n.host) && valid_part(Part::ns, n.ns) && valid_part(Part::model, n.model) &&
           valid_part(Part::tag, n.tag);
}

struct Found {
    OllamaName name;
    std::string rel;
    ManifestLayout layout;
};

void walk_tree(const StoreAnchor& a, ManifestLayout layout, std::map<std::string, Found>& out,
               std::vector<std::string>& notes) {
    const char* tree = layout == ManifestLayout::v2 ? "manifests-v2" : "manifests";
    if (!tree_is_safe(a, tree, &notes)) return;
    auto skip_link = [&](const std::string& rel) { note(notes, "skipped " + printable(rel) + ": a link or reparse point"); };
    for (const auto& host_e : list_rel(a, tree, &notes)) {
        const std::string host_rel = std::string(tree) + "/" + host_e.name;
        if (host_e.kind == EntryKind::link) {
            skip_link(host_rel);
            continue;
        }
        if (host_e.kind != EntryKind::directory) continue;
        const std::string host = layout == ManifestLayout::v2 ? decode_v2_host(host_e.name) : host_e.name;
        for (const auto& ns_e : list_rel(a, host_rel, &notes)) {
            const std::string ns_rel = host_rel + "/" + ns_e.name;
            if (ns_e.kind == EntryKind::link) {
                skip_link(ns_rel);
                continue;
            }
            if (ns_e.kind != EntryKind::directory) continue;
            for (const auto& model_e : list_rel(a, ns_rel, &notes)) {
                const std::string model_rel = ns_rel + "/" + model_e.name;
                if (model_e.kind == EntryKind::link) {
                    skip_link(model_rel);
                    continue;
                }
                if (model_e.kind != EntryKind::directory) continue;
                for (const auto& tag_e : list_rel(a, model_rel, &notes)) {
                    const std::string tag_rel = model_rel + "/" + tag_e.name;
                    if (tag_e.kind == EntryKind::directory) continue;
                    if (tag_e.kind == EntryKind::other) {
                        note(notes, "skipped " + printable(tag_rel) + ": not a regular file");
                        continue;
                    }
                    // A link leaf is read (or refused) by read_manifest, the same as resolve().
                    OllamaName n{host, ns_e.name, model_e.name, tag_e.name};
                    if (!valid_name(n)) {
                        note(notes, "skipped " + printable(tag_rel) + ": not a valid model name");
                        continue;
                    }
                    if (is_default_host(n.host)) n.host = kOllamaDefaultHost;
                    const std::string key = n.key();
                    if (out.count(key) != 0) continue;  // manifests-v2 is walked first and wins
                    out.emplace(key, Found{std::move(n), tag_rel, layout});
                }
            }
        }
    }
}

// True when `candidate` lies lexically below `root`. A candidate with a different root-name
// (the drive-relative "c:5000\ns\m\t" that operator/ produces on Windows for a host "c:5000")
// or one that climbs out through ".." makes lexically_relative return an empty path or one
// that starts with "..".
bool under_root(const fs::path& root, const fs::path& candidate) {
    const fs::path rel = candidate.lexically_normal().lexically_relative(root.lexically_normal());
    if (rel.empty() || rel == ".") return false;
    return *rel.begin() != "..";
}

// Case-insensitive lookup of one name by listing only the directories on its path (hosts,
// then the matching namespace/model/tag). It reads nothing besides directory entries, so a
// miss costs a handful of listings instead of a full scan. Layout order and the first-match
// rule are the same as scan(), so both pick the same entry. Links are skipped as in scan().
std::optional<Found> locate_ci(const StoreAnchor& a, const OllamaName& want) {
    const std::string want_host = canonical_host(want.host);
    const std::string want_ns = lower(want.ns), want_model = lower(want.model), want_tag = lower(want.tag);
    for (const ManifestLayout layout : {ManifestLayout::v2, ManifestLayout::legacy}) {
        const char* tree = layout == ManifestLayout::v2 ? "manifests-v2" : "manifests";
        if (!tree_is_safe(a, tree, nullptr)) continue;
        for (const auto& host_e : list_rel(a, tree, nullptr)) {
            if (host_e.kind != EntryKind::directory) continue;
            const std::string host = layout == ManifestLayout::v2 ? decode_v2_host(host_e.name) : host_e.name;
            if (canonical_host(host) != want_host) continue;
            const std::string host_rel = std::string(tree) + "/" + host_e.name;
            for (const auto& ns_e : list_rel(a, host_rel, nullptr)) {
                if (ns_e.kind != EntryKind::directory || lower(ns_e.name) != want_ns) continue;
                const std::string ns_rel = host_rel + "/" + ns_e.name;
                for (const auto& model_e : list_rel(a, ns_rel, nullptr)) {
                    if (model_e.kind != EntryKind::directory || lower(model_e.name) != want_model) continue;
                    const std::string model_rel = ns_rel + "/" + model_e.name;
                    for (const auto& tag_e : list_rel(a, model_rel, nullptr)) {
                        if ((tag_e.kind != EntryKind::file && tag_e.kind != EntryKind::link) ||
                            lower(tag_e.name) != want_tag) {
                            continue;
                        }
                        OllamaName n{host, ns_e.name, model_e.name, tag_e.name};
                        if (!valid_name(n)) continue;
                        if (is_default_host(n.host)) n.host = kOllamaDefaultHost;
                        return Found{std::move(n), model_rel + "/" + tag_e.name, layout};
                    }
                }
            }
        }
    }
    return std::nullopt;
}

}  // namespace

// ---------------------------------------------------------------------------
// Verified-blob cache (first_use), single-flight hashing and the hash-slot limit
// ---------------------------------------------------------------------------
// Shared by copies of one OllamaStore. Every member is guarded by `mutex`; no hash and no
// file I/O ever runs while it is held (lookup() only waits on `cv`, which releases it).
struct OllamaStore::VerifiedCache {
    using Key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::uint64_t, std::int64_t,
                           std::int64_t>;

    static Key key_of(const FileStamp& s) {
        return Key{s.volume, s.file_id, s.file_id_high, s.size, s.link_count, s.write_time, s.change_time};
    }

    // One in-progress hash of one identity. `hex` is nullopt when the hasher failed.
    struct Flight {
        bool done = false;
        std::optional<std::string> hex;
    };

    // Either a known digest for this identity, or `flight` set: the caller is now the one
    // thread hashing this identity and must end the flight with finish().
    struct Lookup {
        std::optional<std::string> hex;
        std::shared_ptr<Flight> flight;
    };

    // Concurrent callers for one identity: the first becomes the hasher, the others wait for
    // its result (and retry, one of them becoming the hasher, if it failed).
    Lookup lookup(const FileStamp& s) {
        const Key k = key_of(s);
        std::unique_lock<std::mutex> lock(mutex);
        while (true) {
            if (const auto it = entries.find(k); it != entries.end()) return Lookup{it->second, nullptr};
            const auto f = inflight.find(k);
            if (f == inflight.end()) {
                auto flight = std::make_shared<Flight>();
                inflight.emplace(k, flight);
                return Lookup{std::nullopt, std::move(flight)};
            }
            const std::shared_ptr<Flight> other = f->second;
            cv.wait(lock, [&] { return other->done; });
            if (other->hex) return Lookup{other->hex, nullptr};
        }
    }

    // Ends a flight from lookup(): a digest is cached and handed to the waiters; nullopt
    // (failure) only wakes them.
    void finish(const FileStamp& s, const std::shared_ptr<Flight>& flight, std::optional<std::string> hex) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            const Key k = key_of(s);
            if (hex) {
                if (entries.size() >= kMaxVerifiedEntries) entries.clear();
                entries[k] = *hex;
            }
            if (const auto it = inflight.find(k); it != inflight.end() && it->second == flight) inflight.erase(it);
            flight->hex = std::move(hex);
            flight->done = true;
        }
        cv.notify_all();
    }

    // At most kMaxConcurrentWeightHashes weight hashes run at once per store.
    void acquire_hash_slot() {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return hashing < kMaxConcurrentWeightHashes; });
        ++hashing;
    }

    void release_hash_slot() {
        {
            std::lock_guard<std::mutex> lock(mutex);
            --hashing;
        }
        cv.notify_all();
    }

    std::mutex mutex;
    std::condition_variable cv;
    std::map<Key, std::string> entries;
    std::map<Key, std::shared_ptr<Flight>> inflight;
    std::size_t hashing = 0;
};

// Owns the open weight handles of a VerifiedWeights (model_store.hpp).
struct VerifiedWeights::Handles {
    std::vector<sf::OwnedFileHandle> owned;
};

// The store's write access to VerifiedWeights' private members.
struct VerifiedWeightsAccess {
    static void clear(VerifiedWeights& w) {
        w.files_.clear();
        if (w.handles_) w.handles_->owned.clear();
        w.verified_ = false;
    }
    static void add(VerifiedWeights& w, const BlobRef& blob, sf::OwnedFileHandle handle) {
        if (!w.handles_) w.handles_ = std::make_unique<VerifiedWeights::Handles>();
        VerifiedWeights::File f;
        f.blob = blob;
        f.native_handle = handle.native_value();
        w.handles_->owned.push_back(std::move(handle));
        w.files_.push_back(std::move(f));
    }
    static void set_verified(VerifiedWeights& w, bool v) { w.verified_ = v; }
};

namespace {

// Holds one of the store's hash slots for its lifetime.
class HashSlot {
public:
    explicit HashSlot(OllamaStore::VerifiedCache* cache) : cache_(cache) {
        if (cache_ != nullptr) cache_->acquire_hash_slot();
    }
    ~HashSlot() {
        if (cache_ != nullptr) cache_->release_hash_slot();
    }
    HashSlot(const HashSlot&) = delete;
    HashSlot& operator=(const HashSlot&) = delete;

private:
    OllamaStore::VerifiedCache* cache_;
};

// Ends a single-flight hash exactly once: complete() publishes the digest; leaving the scope
// without it (an error, an exception from on_weight_hash) wakes the waiters with a failure.
class FlightGuard {
public:
    FlightGuard(OllamaStore::VerifiedCache* cache, const FileStamp& stamp,
                std::shared_ptr<OllamaStore::VerifiedCache::Flight> flight)
        : cache_(cache), stamp_(stamp), flight_(std::move(flight)) {}
    ~FlightGuard() {
        if (flight_ && cache_ != nullptr) cache_->finish(stamp_, flight_, std::nullopt);
    }
    FlightGuard(const FlightGuard&) = delete;
    FlightGuard& operator=(const FlightGuard&) = delete;
    void complete(const std::string& hex) {
        if (flight_ && cache_ != nullptr) cache_->finish(stamp_, flight_, hex);
        flight_.reset();
    }

private:
    OllamaStore::VerifiedCache* cache_;
    FileStamp stamp_;
    std::shared_ptr<OllamaStore::VerifiedCache::Flight> flight_;
};

// Verifies every weight blob of a ready record per the policy. Each blob is opened afresh
// relative to the pinned root, with writers denied (Windows) for as long as the handle is
// open, and its identity must equal the stamp taken while the record was read. Under
// first_use a cache hit is accepted only if the identity read from this same handle after
// the lookup still equals that stamp; concurrent first uses share one hash. With `keep`,
// the handles are moved into it (also under `off`, unhashed) for a loader; on any failure it
// is left empty.
void verify_weights_for_use(const StoreAnchor& a, ModelRecord& r, const StoreOptions& options,
                            OllamaStore::VerifiedCache* cache, bool use_cache, VerifiedWeights* keep) {
    r.weights_verified = false;
    if (keep != nullptr) VerifiedWeightsAccess::clear(*keep);
    if (!r.ready()) return;
    const bool hash = options.verify_weights != WeightVerification::off;
    if (!hash && keep == nullptr) return;
    auto fail = [&](UnsupportedReason reason, std::string text) {
        mark(r, reason, std::move(text));
        if (keep != nullptr) VerifiedWeightsAccess::clear(*keep);
    };
    std::vector<char> buf;
    std::set<std::string> done;
    for (const auto& b : r.blobs) {
        if (!is_weight_role(b.role) || !done.insert(b.digest).second) continue;
        const std::string hex = b.digest.substr(7);
        const std::string label = std::string(to_string(b.role)) + " blob " + short_digest(hex);
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string err;
        bool busy = false;
        const sf::OpenOutcome o =
            sf::open_anchored_file(a.dir, blob_rel(hex), sf::ShareMode::deny_writers, h, id, err, &busy);
        if (o != sf::OpenOutcome::ok) {
            if (busy) {
                fail(UnsupportedReason::blob_busy,
                     label + " is open for writing by another process; it is not hashed or served until that ends");
            } else {
                fail(o == sf::OpenOutcome::unsafe ? UnsupportedReason::unsafe_path : UnsupportedReason::missing_blob,
                     label + " " +
                         (o == sf::OpenOutcome::missing ? std::string("is not in blobs/") : open_failure_text(o)));
            }
            return;
        }
        const FileStamp stamp = to_stamp(id);
        if (!b.stamp || *b.stamp != stamp) {
            fail(UnsupportedReason::blob_size_mismatch, label + " changed while the model was being resolved");
            return;
        }
        if (hash) {
            std::optional<std::string> actual;
            std::shared_ptr<OllamaStore::VerifiedCache::Flight> flight;
            if (use_cache && cache != nullptr) {
                OllamaStore::VerifiedCache::Lookup found = cache->lookup(stamp);
                if (found.hex) {
                    // The cached digest is for `stamp`; it applies to this handle only if the
                    // identity read from it now, after the lookup, is still exactly that.
                    sf::FileIdentity now;
                    if (sf::query_identity(h, now, err) && to_stamp(now) == stamp) actual = std::move(found.hex);
                } else {
                    flight = std::move(found.flight);
                }
            }
            if (!actual) {
                FlightGuard guard(cache, stamp, std::move(flight));
                std::string got;
                std::uint64_t bytes = 0;
                bool read_ok = false;
                {
                    HashSlot slot(cache);
                    if (options.on_weight_hash) options.on_weight_hash(b);
                    if (buf.empty()) buf.resize(kHashBufferBytes);
                    read_ok = hash_handle(h, buf, got, bytes, err, {});
                }
                if (!read_ok) {
                    fail(UnsupportedReason::missing_blob, label + " cannot be read: " + printable(err));
                    return;
                }
                sf::FileIdentity after;
                if (bytes != id.size || !sf::query_identity(h, after, err) || to_stamp(after) != stamp) {
                    fail(UnsupportedReason::blob_size_mismatch, label + " changed while it was hashed");
                    return;
                }
                guard.complete(got);
                actual = std::move(got);
            }
            if (*actual != hex) {
                fail(UnsupportedReason::digest_mismatch, label + " does not hash to its digest");
                return;
            }
        }
        if (keep != nullptr) VerifiedWeightsAccess::add(*keep, b, std::move(h));
    }
    r.weights_verified = hash && r.ready();
    if (keep != nullptr) VerifiedWeightsAccess::set_verified(*keep, r.weights_verified);
}

bool is_needed_role(BlobRole role) {
    return is_weight_role(role) || role == BlobRole::config || role == BlobRole::tmpl || role == BlobRole::system ||
           role == BlobRole::params || role == BlobRole::messages;
}

// True when every file the record was built from still has the identity it had then. A blob
// the record never opened makes a ready record stale when it is one the record needs; for an
// unsupported record (`unopened_must_be_absent`) every such blob must still be absent, since
// one that appeared (a finished download, a repair) can change the outcome.
bool sources_unchanged(const StoreAnchor& a, const ModelRecord& r, bool unopened_must_be_absent) {
    for (const auto& s : r.sources) {
        std::string err;
        if (s.link) {
            std::string target;
            sf::FileIdentity link_id;
            if (!sf::read_anchored_leaf_symlink(a.dir, s.rel, target, link_id, err) || to_stamp(link_id) != s.stamp) {
                return false;
            }
            continue;
        }
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        if (sf::open_anchored_file(a.dir, s.rel, sf::ShareMode::allow_writers, h, id, err) != sf::OpenOutcome::ok ||
            to_stamp(id) != s.stamp) {
            return false;
        }
    }
    for (const auto& b : r.blobs) {
        if (!b.stamp && !unopened_must_be_absent && !is_needed_role(b.role)) continue;
        if (!b.stamp && !unopened_must_be_absent) return false;  // a blob a ready record needs was never opened
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string err;
        const sf::OpenOutcome o =
            sf::open_anchored_file(a.dir, blob_rel(b.digest.substr(7)), sf::ShareMode::allow_writers, h, id, err);
        if (!b.stamp) {
            if (o != sf::OpenOutcome::missing) return false;  // it appeared (or is now refused)
            continue;
        }
        if (o != sf::OpenOutcome::ok || to_stamp(id) != *b.stamp) return false;
    }
    return true;
}

}  // namespace

// ---------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------
std::string OllamaName::full() const { return host + "/" + ns + "/" + model + ":" + tag; }

std::string OllamaName::display() const {
    std::string out;
    if (!is_default_host(host)) {
        out = host + "/" + ns + "/";
    } else if (lower(ns) != kOllamaDefaultNamespace) {
        out = ns + "/";
    }
    return out + model + ":" + tag;
}

std::string OllamaName::key() const {
    return lower(canonical_host(host) + "/" + ns + "/" + model + ":" + tag);
}

Result<OllamaName> parse_ollama_name(std::string_view text) {
    auto bad = [&](const std::string& why) {
        return Status(ErrorCode::invalid_argument, "invalid model name '" + printable(text) + "': " + why);
    };
    if (text.empty()) return bad("empty");
    // No part is copied out of a name that cannot be valid (kMaxOllamaNameBytes).
    if (text.size() > kMaxOllamaNameBytes) {
        return bad("longer than " + std::to_string(kMaxOllamaNameBytes) + " bytes");
    }
    std::string_view s = text;
    OllamaName n;
    // A separator promises a part: "x:" and "/x" are errors, not defaults (name.go cutPromised).
    bool promised_tag = false, promised_ns = false, promised_host = false;
    const auto last_colon = s.rfind(':');
    const auto last_slash = s.rfind('/');
    if (last_colon != std::string_view::npos && (last_slash == std::string_view::npos || last_colon > last_slash)) {
        n.tag = std::string(s.substr(last_colon + 1));
        promised_tag = true;
        s = s.substr(0, last_colon);
    }
    if (const auto slash = s.rfind('/'); slash != std::string_view::npos) {
        n.model = std::string(s.substr(slash + 1));
        s = s.substr(0, slash);
        if (const auto slash2 = s.rfind('/'); slash2 != std::string_view::npos) {
            n.ns = std::string(s.substr(slash2 + 1));
            promised_ns = true;
            s = s.substr(0, slash2);
            if (const auto scheme = s.find("://"); scheme != std::string_view::npos) s = s.substr(scheme + 3);
            n.host = std::string(s);
            promised_host = true;
        } else {
            n.ns = std::string(s);
            promised_ns = true;
        }
    } else {
        n.model = std::string(s);
    }
    if (n.host.empty() && !promised_host) n.host = kOllamaDefaultHost;
    if (n.ns.empty() && !promised_ns) n.ns = kOllamaDefaultNamespace;
    if (n.tag.empty() && !promised_tag) n.tag = kOllamaDefaultTag;
    if (!valid_part(Part::host, n.host)) return bad("bad host part");
    if (!valid_part(Part::ns, n.ns)) return bad("bad namespace part");
    if (!valid_part(Part::model, n.model)) return bad("bad model part");
    if (!valid_part(Part::tag, n.tag)) return bad("bad tag part");
    if (is_default_host(n.host)) n.host = kOllamaDefaultHost;
    return n;
}

const char* to_string(UnsupportedReason reason) noexcept {
    switch (reason) {
        case UnsupportedReason::none: return "none";
        case UnsupportedReason::manifest_invalid: return "manifest_invalid";
        case UnsupportedReason::no_compatible_manifest: return "no_compatible_manifest";
        case UnsupportedReason::cloud_stub: return "cloud_stub";
        case UnsupportedReason::safetensors: return "safetensors";
        case UnsupportedReason::unsupported_format: return "unsupported_format";
        case UnsupportedReason::no_model_layer: return "no_model_layer";
        case UnsupportedReason::missing_blob: return "missing_blob";
        case UnsupportedReason::blob_size_mismatch: return "blob_size_mismatch";
        case UnsupportedReason::invalid_gguf: return "invalid_gguf";
        case UnsupportedReason::unsafe_path: return "unsafe_path";
        case UnsupportedReason::digest_mismatch: return "digest_mismatch";
        case UnsupportedReason::blob_busy: return "blob_busy";
    }
    return "none";
}

const char* to_string(EntryStatus status) noexcept { return status == EntryStatus::ready ? "ready" : "unsupported"; }

const char* to_string(BlobRole role) noexcept {
    switch (role) {
        case BlobRole::config: return "config";
        case BlobRole::model: return "model";
        case BlobRole::projector: return "projector";
        case BlobRole::adapter: return "adapter";
        case BlobRole::draft: return "draft";
        case BlobRole::tmpl: return "template";
        case BlobRole::system: return "system";
        case BlobRole::params: return "params";
        case BlobRole::messages: return "messages";
        case BlobRole::license: return "license";
        case BlobRole::other: return "other";
    }
    return "other";
}

const char* to_string(ManifestLayout layout) noexcept { return layout == ManifestLayout::v2 ? "v2" : "legacy"; }

const char* to_string(WeightVerification policy) noexcept {
    switch (policy) {
        case WeightVerification::always: return "always";
        case WeightVerification::first_use: return "first_use";
        case WeightVerification::off: return "off";
    }
    return "first_use";
}

const char* to_string(BlobVerdict verdict) noexcept {
    switch (verdict) {
        case BlobVerdict::ok: return "ok";
        case BlobVerdict::mismatch: return "mismatch";
        case BlobVerdict::missing: return "missing";
        case BlobVerdict::io_error: return "io_error";
        case BlobVerdict::cancelled: return "cancelled";
        case BlobVerdict::unsafe: return "unsafe";
    }
    return "io_error";
}

bool ModelParams::empty() const {
    return !num_ctx && !num_batch && !num_gpu && !num_thread && !num_keep && !seed && !num_predict && !top_k &&
           !repeat_last_n && !temperature && !top_p && !min_p && !typical_p && !repeat_penalty && !presence_penalty &&
           !frequency_penalty && stop.empty() && extra.empty();
}

// ---------------------------------------------------------------------------
// GGUF header
// ---------------------------------------------------------------------------
Result<GgufHeader> read_gguf_header(const fs::path& file) {
    sf::OwnedFileHandle h;
    std::uint64_t size = 0;
    std::string err;
    switch (sf::open_single_regular(file, h, size, err)) {
        case sf::OpenOutcome::ok: break;
        case sf::OpenOutcome::unsafe: return Status(ErrorCode::invalid_argument, "not a regular file (link or reparse point)");
        case sf::OpenOutcome::missing:
        case sf::OpenOutcome::io_error: return Status(ErrorCode::io_error, "cannot open file");
    }
    return header_from_handle(h);
}

// ---------------------------------------------------------------------------
// OllamaStore
// ---------------------------------------------------------------------------
OllamaStore::OllamaStore(fs::path root, std::shared_ptr<const StoreAnchor> anchor, StoreOptions options)
    : root_(std::move(root)),
      anchor_(std::move(anchor)),
      options_(std::move(options)),
      cache_(std::make_shared<VerifiedCache>()) {}

Result<OllamaStore> OllamaStore::open(fs::path models_dir, StoreOptions options) {
    if (models_dir.empty()) return Status(ErrorCode::not_found, "the models directory is not a directory");
    auto anchor = std::make_shared<StoreAnchor>();
    std::string err;
    // The one path-based open: a link at the root itself is followed here, once.
    if (sf::open_anchor_dir(models_dir, anchor->dir, anchor->id, err) != sf::OpenOutcome::ok) {
        return Status(ErrorCode::not_found, "the models directory is not a directory");
    }
    // A tree that is a link counts as present: the scan then notes and skips it.
    auto present = [&](const char* tree) {
        const sf::OpenOutcome o = sf::probe_anchored_dir(anchor->dir, tree, err);
        return o == sf::OpenOutcome::ok || o == sf::OpenOutcome::unsafe;
    };
    const bool v2 = present("manifests-v2");
    const bool legacy = present("manifests");
    if (!v2 && !legacy) {
        return Status(ErrorCode::invalid_argument,
                      "the directory has neither manifests-v2/ nor manifests/ (not an Ollama models directory)");
    }
    std::error_code ec;
    fs::path root = fs::absolute(models_dir, ec);
    if (ec) root = models_dir;
    anchor->path = root;
    return OllamaStore(std::move(root), std::move(anchor), std::move(options));
}

ScanResult OllamaStore::scan() const {
    std::map<std::string, Found> found;
    ScanResult result;
    walk_tree(*anchor_, ManifestLayout::v2, found, result.notes);
    walk_tree(*anchor_, ManifestLayout::legacy, found, result.notes);
    result.records.reserve(found.size());
    for (auto& [key, f] : found) {
        ModelRecord rec = build_record(anchor_, f.name, f.rel, f.layout, read_manifest(*anchor_, f.rel, f.layout));
        if (options_.verify_weights == WeightVerification::always) {
            verify_weights_for_use(*anchor_, rec, options_, cache_.get(), false, nullptr);
        }
        result.records.push_back(std::move(rec));
    }
    return result;
}

Result<ModelRecord> OllamaStore::resolve(std::string_view name) const { return resolve_impl(name, nullptr); }

Result<LoadableModel> OllamaStore::resolve_for_load(std::string_view name) const {
    VerifiedWeights weights;
    auto rec = resolve_impl(name, &weights);
    if (!rec.ok()) return rec.status();
    LoadableModel out;
    out.record = std::move(rec).value();
    if (out.record.ready()) out.weights = std::move(weights);
    return out;
}

Result<ModelRecord> OllamaStore::resolve_impl(std::string_view name, VerifiedWeights* keep) const {
    auto parsed = parse_ollama_name(name);
    if (!parsed.ok()) return parsed.status();
    const OllamaName n = parsed.value();

    std::vector<std::pair<std::string, ManifestLayout>> candidates;
    const std::string tail = "/" + n.ns + "/" + n.model + "/" + n.tag;
    if (is_default_host(n.host)) {
        candidates.emplace_back("manifests-v2/ollama.com" + tail, ManifestLayout::v2);
        candidates.emplace_back(std::string("manifests-v2/") + kOllamaDefaultHost + tail, ManifestLayout::v2);
        candidates.emplace_back(std::string("manifests/") + kOllamaDefaultHost + tail, ManifestLayout::legacy);
        candidates.emplace_back("manifests/ollama.com" + tail, ManifestLayout::legacy);
    } else {
        candidates.emplace_back("manifests-v2/" + encode_v2_host(n.host) + tail, ManifestLayout::v2);
        // A host with a port has no legacy directory form (':' is not a path character on
        // Windows, where "c:5000" would also be read as a drive-relative path). Such names are
        // served only through the v2 tree (%3A); a legacy directory that exists on another OS
        // is still found by the directory lookup below.
        if (n.host.find(':') == std::string::npos) {
            candidates.emplace_back("manifests/" + n.host + tail, ManifestLayout::legacy);
        }
    }
    const bool use_cache = options_.verify_weights == WeightVerification::first_use;
    auto finish = [&](const OllamaName& rec_name, const std::string& rel, ManifestLayout layout, ManifestText&& mt) {
        ModelRecord rec = build_record(anchor_, rec_name, rel, layout, std::move(mt));
        verify_weights_for_use(*anchor_, rec, options_, cache_.get(), use_cache, keep);
        return rec;
    };
    for (const auto& [rel, layout] : candidates) {
        if (!under_root(root_, root_ / from_utf8(rel))) continue;
        ManifestText mt = read_manifest(*anchor_, rel, layout);
        if (mt.outcome == sf::OpenOutcome::missing && mt.reason == UnsupportedReason::none) continue;
        return finish(n, rel, layout, std::move(mt));
    }
    // Case-insensitive fallback on case-sensitive file systems: list the matching directories
    // only (no manifest, config or GGUF reads except for the one entry that matches).
    if (auto found = locate_ci(*anchor_, n)) {
        ManifestText mt = read_manifest(*anchor_, found->rel, found->layout);
        if (!(mt.outcome == sf::OpenOutcome::missing && mt.reason == UnsupportedReason::none)) {
            return finish(found->name, found->rel, found->layout, std::move(mt));
        }
    }
    return Status(ErrorCode::not_found, "model '" + n.display() + "' is not in the store");
}

bool OllamaStore::revalidate(ModelRecord& record) const { return revalidate_impl(record, nullptr); }

bool OllamaStore::revalidate_impl(ModelRecord& record, VerifiedWeights* keep) const {
    // Only a record read through this very root handle (copies of this store share it).
    if (record.anchor.get() != anchor_.get() || !sources_unchanged(*anchor_, record, false)) return false;
    verify_weights_for_use(*anchor_, record, options_, cache_.get(),
                           options_.verify_weights == WeightVerification::first_use, keep);
    return true;
}

bool OllamaStore::unsupported_still_current(const ModelRecord& record) const {
    if (record.ready() || record.anchor.get() != anchor_.get()) return false;
    if (record.reason == UnsupportedReason::blob_busy) return false;  // ends when the writer closes the file
    return sources_unchanged(*anchor_, record, true);
}

// ---------------------------------------------------------------------------
// VerifiedWeights
// ---------------------------------------------------------------------------
VerifiedWeights::VerifiedWeights() noexcept = default;

VerifiedWeights::~VerifiedWeights() = default;

VerifiedWeights::VerifiedWeights(VerifiedWeights&& other) noexcept
    : files_(std::move(other.files_)), handles_(std::move(other.handles_)), verified_(other.verified_) {
    other.files_.clear();
    other.verified_ = false;
}

VerifiedWeights& VerifiedWeights::operator=(VerifiedWeights&& other) noexcept {
    if (this != &other) {
        files_ = std::move(other.files_);
        handles_ = std::move(other.handles_);
        verified_ = other.verified_;
        other.files_.clear();
        other.verified_ = false;
    }
    return *this;
}

const VerifiedWeights::File* VerifiedWeights::find(std::string_view digest) const noexcept {
    for (const auto& f : files_) {
        if (f.blob.digest == digest) return &f;
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------
bool VerifyReport::ok() const {
    return std::all_of(blobs.begin(), blobs.end(), [](const BlobVerification& b) { return b.verdict == BlobVerdict::ok; });
}

VerifyReport verify_blobs(const ModelRecord& record, const VerifyOptions& options) {
    VerifyReport report;
    // The record's pinned root; a hand-built record without one opens store_root once here.
    std::shared_ptr<const StoreAnchor> root = record.anchor;
    if (!root) {
        auto own = std::make_shared<StoreAnchor>();
        std::string err;
        if (sf::open_anchor_dir(record.store_root, own->dir, own->id, err) == sf::OpenOutcome::ok) {
            own->path = record.store_root;
            root = std::move(own);
        }
    }
    std::vector<char> buf;
    std::set<std::string> seen;
    for (const auto& blob : record.blobs) {
        if (options.weights_only && !is_weight_role(blob.role)) continue;
        if (!seen.insert(blob.digest).second) continue;
        BlobVerification v;
        v.blob = blob;
        if (options.cancel.cancelled()) {
            v.verdict = BlobVerdict::cancelled;
            v.detail = "cancelled before hashing";
            report.blobs.push_back(std::move(v));
            break;
        }
        const auto hex = manifest_digest_hex(blob.digest);
        if (!hex) {
            v.verdict = BlobVerdict::io_error;
            v.detail = "malformed digest";
            report.blobs.push_back(std::move(v));
            continue;
        }
        if (!root) {
            v.verdict = BlobVerdict::io_error;
            v.detail = "the store directory cannot be opened";
            report.blobs.push_back(std::move(v));
            continue;
        }
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string error;
        const sf::OpenOutcome o =
            sf::open_anchored_file(root->dir, blob_rel(*hex), sf::ShareMode::allow_writers, h, id, error);
        if (o != sf::OpenOutcome::ok) {
            v.verdict = o == sf::OpenOutcome::missing ? BlobVerdict::missing
                        : o == sf::OpenOutcome::unsafe ? BlobVerdict::unsafe
                                                       : BlobVerdict::io_error;
            v.detail = o == sf::OpenOutcome::missing ? std::string("blob file is not present")
                                                     : std::string("blob file ") + open_failure_text(o);
            report.blobs.push_back(std::move(v));
            continue;
        }
        if (options.on_blob_start) options.on_blob_start(blob);
        if (buf.empty()) buf.resize(kHashBufferBytes);
        std::string got;
        std::uint64_t bytes = 0;
        const bool done = hash_handle(h, buf, got, bytes, error, [&options] { return options.cancel.cancelled(); });
        if (!done) {
            v.verdict = options.cancel.cancelled() ? BlobVerdict::cancelled : BlobVerdict::io_error;
            v.detail = printable(error);
            const bool stop = v.verdict == BlobVerdict::cancelled;
            report.blobs.push_back(std::move(v));
            if (stop) break;
            continue;
        }
        sf::FileIdentity after;
        if (bytes != id.size || !sf::query_identity(h, after, error) || to_stamp(after) != to_stamp(id)) {
            v.verdict = BlobVerdict::io_error;
            v.detail = "blob changed while it was hashed";
            report.blobs.push_back(std::move(v));
            continue;
        }
        v.actual_sha256 = got;
        if (got == *hex) {
            v.verdict = BlobVerdict::ok;
        } else {
            v.verdict = BlobVerdict::mismatch;
            v.detail = "content hashes to sha256:" + got;
        }
        report.blobs.push_back(std::move(v));
    }
    return report;
}

// ---------------------------------------------------------------------------
// JSON
// ---------------------------------------------------------------------------
namespace {

json::Array path_array(const std::vector<fs::path>& paths) {
    json::Array a;
    for (const auto& p : paths) a.emplace_back(display_path(p));
    return a;
}

json::Array string_array(const std::vector<std::string>& v) {
    json::Array a;
    for (const auto& s : v) a.emplace_back(s);
    return a;
}

json::Object params_json(const ModelParams& p) {
    json::Object o;
    auto put_i = [&](const char* k, const std::optional<std::int64_t>& v) {
        if (v) o.set(k, json::Value(static_cast<long long>(*v)));
    };
    auto put_d = [&](const char* k, const std::optional<double>& v) {
        if (v) o.set(k, json::Value(*v));
    };
    put_i("num_ctx", p.num_ctx);
    put_i("num_batch", p.num_batch);
    put_i("num_gpu", p.num_gpu);
    put_i("num_thread", p.num_thread);
    put_i("num_keep", p.num_keep);
    put_i("seed", p.seed);
    put_i("num_predict", p.num_predict);
    put_i("top_k", p.top_k);
    put_i("repeat_last_n", p.repeat_last_n);
    put_d("temperature", p.temperature);
    put_d("top_p", p.top_p);
    put_d("min_p", p.min_p);
    put_d("typical_p", p.typical_p);
    put_d("repeat_penalty", p.repeat_penalty);
    put_d("presence_penalty", p.presence_penalty);
    put_d("frequency_penalty", p.frequency_penalty);
    if (!p.stop.empty()) o.set("stop", string_array(p.stop));
    for (const auto& [k, v] : p.extra) o.set(k, v);
    return o;
}

}  // namespace

json::Value to_json(const ModelRecord& r) {
    json::Object o;
    o.set("name", r.name.display());
    o.set("full_name", r.name.full());
    o.set("status", to_string(r.status));
    o.set("reason", r.reason == UnsupportedReason::none ? json::Value() : json::Value(to_string(r.reason)));
    o.set("detail", r.detail);
    o.set("manifest", json::Object{{"path", display_path(r.manifest_path)},
                                   {"layout", to_string(r.layout)},
                                   {"runner", r.runner},
                                   {"format", r.format}});
    o.set("parent", r.parent);
    o.set("config", json::Object{{"model_format", r.config.model_format},
                                 {"family", r.config.model_family},
                                 {"families", string_array(r.config.model_families)},
                                 {"parameter_size", r.config.parameter_size},
                                 {"quantization", r.config.quantization},
                                 {"renderer", r.config.renderer},
                                 {"parser", r.config.parser},
                                 {"requires", r.config.requires_version},
                                 {"remote_host", redact_userinfo(r.config.remote_host)},
                                 {"remote_model", r.config.remote_model},
                                 {"capabilities", string_array(r.config.capabilities)},
                                 {"context_length", json::Value(static_cast<long long>(r.config.context_length))},
                                 {"embedding_length", json::Value(static_cast<long long>(r.config.embedding_length))}});
    json::Object files;
    files.set("model", r.model_path.empty() ? json::Value() : json::Value(display_path(r.model_path)));
    files.set("shards", path_array(r.shard_paths));
    files.set("projectors", path_array(r.projector_paths));
    files.set("adapters", path_array(r.adapter_paths));
    files.set("drafts", path_array(r.draft_paths));
    o.set("files", std::move(files));
    if (r.gguf) {
        o.set("gguf", json::Object{{"version", json::Value(static_cast<unsigned>(r.gguf->version))},
                                   {"tensor_count", json::Value(static_cast<unsigned long long>(r.gguf->tensor_count))},
                                   {"kv_count", json::Value(static_cast<unsigned long long>(r.gguf->kv_count))}});
    } else {
        o.set("gguf", json::Value());
    }
    o.set("params", params_json(r.params));
    o.set("template", r.template_text ? json::Value(*r.template_text) : json::Value());
    o.set("system", r.system_prompt ? json::Value(*r.system_prompt) : json::Value());
    o.set("messages", r.messages_json ? json::Value(*r.messages_json) : json::Value());
    o.set("license_count", json::Value(static_cast<unsigned long long>(r.license_count)));
    o.set("weights_bytes", json::Value(static_cast<unsigned long long>(r.weights_bytes)));
    o.set("referenced_bytes", json::Value(static_cast<unsigned long long>(r.referenced_bytes)));
    o.set("weights_verified", r.weights_verified);
    o.set("warnings", string_array(r.warnings));
    return json::Value(std::move(o));
}

json::Value to_json(const ScanResult& scan, const fs::path& store_root) {
    json::Array models;
    for (const auto& r : scan.records) models.push_back(to_json(r));
    return json::Value(json::Object{{"schema", kStoreJsonSchema},
                                    {"source", json::Object{{"kind", "ollama"}, {"dir", display_path(store_root)}}},
                                    {"models", std::move(models)},
                                    {"notes", string_array(scan.notes)}});
}

fs::path default_ollama_models_dir() {
#if defined(_WIN32)
    // _wgetenv: getenv returns the ANSI code page form, which mangles non-ASCII paths.
    if (const wchar_t* env = _wgetenv(L"OLLAMA_MODELS"); env != nullptr && *env != L'\0') return fs::path(env);
    const wchar_t* home = _wgetenv(L"USERPROFILE");
    if (home == nullptr || *home == L'\0') return {};
#else
    if (const char* env = std::getenv("OLLAMA_MODELS"); env != nullptr && *env != '\0') return fs::path(env);
    const char* home = std::getenv("HOME");
    if (home == nullptr || *home == '\0') return {};
#endif
    return fs::path(home) / ".ollama" / "models";
}

// ---------------------------------------------------------------------------
// ModelRegistry
// ---------------------------------------------------------------------------
Status ModelRegistry::add_ollama_dir(fs::path models_dir) {
    auto store = OllamaStore::open(std::move(models_dir), options_);
    if (!store.ok()) return store.status();
    // The same directory: compared by the identity of the pinned root handles, not by path.
    auto registered = [&] {
        for (const auto& s : stores_) {
            if (s.anchor_->id.same_file(store.value().anchor_->id)) return true;
        }
        return false;
    };
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (registered()) return Status::success();
    }
    // The scan (directory walk, config/params reads, GGUF headers) runs without the lock so
    // concurrent resolve()/list() calls are not stalled behind it.
    auto scanned = store.value().scan();
    std::lock_guard<std::mutex> lock(mutex_);
    if (registered()) return Status::success();  // another thread added the same directory meanwhile
    stores_.push_back(store.value());
    for (auto& rec : scanned.records) {
        const std::string key = rec.name.key();
        const bool dup = std::any_of(records_.begin(), records_.end(),
                                     [&](const ModelRecord& e) { return e.name.key() == key; });
        if (!dup) records_.push_back(std::move(rec));
    }
    return Status::success();
}

void ModelRegistry::refresh() {
    std::vector<OllamaStore> stores;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stores = stores_;
    }
    std::vector<ModelRecord> fresh;
    std::set<std::string> seen;
    for (const auto& store : stores) {
        ScanResult scanned = store.scan();  // no lock held: this reads the disk
        for (auto& rec : scanned.records) {
            if (seen.insert(rec.name.key()).second) fresh.push_back(std::move(rec));
        }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    records_ = std::move(fresh);
    live_.clear();
}

std::vector<ModelRecord> ModelRegistry::list() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return records_;
}

Result<ModelRecord> ModelRegistry::resolve(std::string_view name) const { return resolve_impl(name, nullptr); }

Result<LoadableModel> ModelRegistry::resolve_for_load(std::string_view name) const {
    VerifiedWeights weights;
    auto rec = resolve_impl(name, &weights);
    if (!rec.ok()) return rec.status();
    LoadableModel out;
    out.record = std::move(rec).value();
    out.weights = std::move(weights);
    return out;
}

Result<ModelRecord> ModelRegistry::resolve_impl(std::string_view name, VerifiedWeights* keep) const {
    auto parsed = parse_ollama_name(name);
    if (!parsed.ok()) return parsed.status();
    const std::string key = parsed.value().key();
    auto check = [&](const ModelRecord& rec) -> Result<ModelRecord> {
        if (rec.ready()) return rec;
        return Status(ErrorCode::unsupported,
                      "model '" + rec.name.display() + "' cannot be served from the store (" + to_string(rec.reason) +
                          "): " + rec.detail);
    };
    std::optional<ModelRecord> cached;
    bool from_scan = false;
    std::vector<OllamaStore> stores;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& rec : records_) {
            if (rec.name.key() == key) {
                cached = rec;
                from_scan = true;
                break;
            }
        }
        if (!cached) {
            if (const auto it = live_.find(key); it != live_.end()) cached = it->second;
        }
        stores = stores_;
    }
    // Store this result in the caches: a scan entry is replaced (an unsupported one is then
    // re-checked on every lookup, see below), a live entry only kept while it is ready.
    auto remember = [&](const ModelRecord& rec, bool replace_scan) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (replace_scan) {
            for (auto& e : records_) {
                if (e.name.key() == key) {
                    e = rec;
                    return;
                }
            }
        }
        live_.erase(key);
        if (!rec.ready()) return;
        if (live_.size() >= kMaxLiveRecords) live_.clear();
        live_.emplace(key, rec);
    };
    if (cached) {
        for (const auto& store : stores) {
            if (store.anchor_.get() != cached->anchor.get()) continue;
            // Disk reads (and possibly a weight hash) below run without the registry lock.
            if (!cached->ready()) {
                // An unsupported reason is served again only while it still holds: the files it
                // was read from are unchanged and the blobs it could not open are still absent.
                if (store.unsupported_still_current(*cached)) return check(*cached);
                break;  // something changed (a repair, a finished pull): resolve it again below
            }
            ModelRecord rec = *cached;
            if (store.revalidate_impl(rec, keep)) {
                if (!rec.ready()) remember(rec, from_scan);
                return check(rec);
            }
            break;
        }
    }
    // Live lookup without the lock: it reads the disk (manifest and blob headers).
    for (const auto& store : stores) {
        auto live = store.resolve_impl(name, keep);
        if (!live.ok()) continue;
        remember(live.value(), from_scan);
        return check(live.value());
    }
    if (cached) {  // the cached entry is gone from every store
        std::lock_guard<std::mutex> lock(mutex_);
        live_.erase(key);
        records_.erase(std::remove_if(records_.begin(), records_.end(),
                                      [&](const ModelRecord& e) { return e.name.key() == key; }),
                       records_.end());
    }
    return Status(ErrorCode::not_found, "model '" + parsed.value().display() + "' is not in any registered store");
}

}  // namespace sonder::inference::models
