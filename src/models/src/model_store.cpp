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
// Every file is opened through common/safe_file.hpp: one segment at a time from the store
// root, never following a link or reparse point, and read from that one handle.
#include "sonder/inference/model_store.hpp"

#include <algorithm>
#include <array>
#include <cctype>
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

namespace sonder::inference::models {

namespace fs = std::filesystem;
namespace sf = ::sonder::inference::detail;

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

// Bounded, single-line form of an untrusted string for messages and notes: C0/C1 controls,
// DEL and backslash are escaped (\xNN, \\), at most `max` input bytes, "..." when cut.
std::string printable(std::string_view s, std::size_t max = kMaxQuoted) {
    static constexpr char kHex[] = "0123456789abcdef";
    const std::size_t n = std::min(s.size(), max);
    std::string out;
    out.reserve(n + 8);
    auto hex_byte = [&](unsigned char c) {
        out += "\\x";
        out.push_back(kHex[c >> 4]);
        out.push_back(kHex[c & 15]);
    };
    for (std::size_t i = 0; i < n; ++i) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c < 0x20 || c == 0x7f) {
            hex_byte(c);
        } else if (c == '\\') {
            out += "\\\\";
        } else if (c == 0xC2 && i + 1 < n && static_cast<unsigned char>(s[i + 1]) >= 0x80 &&
                   static_cast<unsigned char>(s[i + 1]) <= 0x9F) {
            hex_byte(c);  // a C1 control (U+0080-U+009F)
            hex_byte(static_cast<unsigned char>(s[++i]));
        } else {
            out.push_back(static_cast<char>(c));
        }
    }
    if (s.size() > max) out += "...";
    return out;
}

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
// "coder." would open "coder") and no DOS device base name.
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

// A blob file name: "sha256-<64 hex>" (manifest.go blobFilenamePattern) -> lowercase hex.
std::optional<std::string> blob_name_hex(std::string_view name) {
    if (name.size() != 7 + 64 || name.substr(0, 7) != "sha256-") return std::nullopt;
    const std::string_view hex = name.substr(7);
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

// Opens `rel` under `root` without following links. On POSIX a v2 name may be a symlink
// (Ollama writes them as ../../../../blobs/sha256-<hex>): then the link is never read
// through; blobs/sha256-<hex> is opened from the root instead, must be the very file the
// link resolves to, and must hash to <hex> (manifest.go openManifestFileLocked).
ManifestText read_manifest(const fs::path& root, const std::string& rel) {
    ManifestText m;
    sf::OwnedFileHandle h;
    sf::FileIdentity id;
    std::string err;
    std::string expect_hex;
    std::string read_rel = rel;
    m.outcome = sf::open_anchored_file(root, rel, h, id, err);
    if (m.outcome == sf::OpenOutcome::unsafe) {
        std::string target, link_err;
        sf::FileIdentity link_id, target_id;
        if (sf::read_anchored_leaf_symlink(root, rel, target, link_id, target_id, link_err)) {
            const auto slash = target.find_last_of('/');
            const auto hex = blob_name_hex(slash == std::string::npos ? target : target.substr(slash + 1));
            if (!hex) {
                m.reason = UnsupportedReason::unsafe_path;
                m.error = "manifest is a symbolic link that does not name a blob";
                return m;
            }
            read_rel = blob_rel(*hex);
            sf::OwnedFileHandle bh;
            sf::FileIdentity bid;
            if (sf::open_anchored_file(root, read_rel, bh, bid, err) != sf::OpenOutcome::ok || !bid.same_file(target_id)) {
                m.reason = UnsupportedReason::unsafe_path;
                m.error = "manifest symbolic link does not resolve to " + short_digest(*hex) + " in blobs/";
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
    RecordReader(const fs::path& root, ModelRecord& r) : root_(root), r_(r) {}

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
        if (sf::open_anchored_file(root_, blob_rel(hex), h, id, err) == sf::OpenOutcome::ok) {
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
        const sf::OpenOutcome o = sf::open_anchored_file(root_, blob_rel(hex), h, id, err);
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
        const sf::OpenOutcome o = sf::open_anchored_file(root_, blob_rel(hex), h, id, err);
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

    const fs::path& root_;
    ModelRecord& r_;
    std::uint64_t text_bytes_ = 0;
    std::map<std::string, SmallResult> small_;
    std::map<std::string, WeightResult> weights_;
};

// Selects the manifest-list child this engine can use: llamacpp first, then ggml (the same
// order as manifest/manifest.go runnerPreferences on non-Apple platforms, minus mlx).
// Returns false and sets reason/detail when none fits.
bool select_child(const json::Value& list, const fs::path& root, ModelRecord& r, json::Value& child_doc,
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
                if (sf::open_anchored_file(root, blob_rel(*hex), probe, pid, perr) == sf::OpenOutcome::missing) {
                    if (skipped_reason == UnsupportedReason::none) {
                        skipped_reason = UnsupportedReason::missing_blob;
                        skipped_detail = "manifest list child " + short_digest(*hex) + " is not in blobs/";
                    }
                    continue;
                }
            }
            ModelRecord probe_record;
            RecordReader child_reader(root, probe_record);
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

ModelRecord build_record(const fs::path& root, const OllamaName& name, const std::string& rel, ManifestLayout layout,
                         ManifestText&& mt) {
    ModelRecord r;
    r.name = name;
    r.store_root = root;
    r.manifest_path = root / from_utf8(rel);
    r.layout = layout;
    r.sources = std::move(mt.sources);
    if (mt.outcome != sf::OpenOutcome::ok || mt.reason != UnsupportedReason::none) {
        mark(r, mt.reason == UnsupportedReason::none ? UnsupportedReason::manifest_invalid : mt.reason,
             mt.error.empty() ? std::string("manifest ") + open_failure_text(mt.outcome) : mt.error);
        return r;
    }
    RecordReader reader(root, r);

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
        if (!select_child(manifest_doc, root, r, child, reason, why)) {
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

std::vector<fs::directory_entry> list_dir(const fs::path& dir) {
    std::vector<fs::directory_entry> out;
    std::error_code ec;
    fs::directory_iterator it(dir, fs::directory_options::skip_permission_denied, ec);
    if (ec) return out;
    for (const fs::directory_iterator end{}; it != end; it.increment(ec)) {
        if (ec) break;
        out.push_back(*it);
    }
    return out;
}

enum class EntryKind { directory, file, link, other };

EntryKind kind_of(const fs::directory_entry& e) {
    std::error_code ec;
    const fs::file_status st = e.symlink_status(ec);
    if (ec) return EntryKind::other;
    if (sf::is_link_like(e.path(), st)) return EntryKind::link;
    if (fs::is_directory(st)) return EntryKind::directory;
    if (fs::is_regular_file(st)) return EntryKind::file;
    return EntryKind::other;
}

// Lists `dir_rel` under `root`: each entry's UTF-8 name and kind. Entries whose names are not
// representable as UTF-8 are noted and skipped.
struct Listed {
    std::string name;
    EntryKind kind;
};

std::vector<Listed> list_rel(const fs::path& root, const std::string& dir_rel, std::vector<std::string>* notes) {
    std::vector<Listed> out;
    for (const auto& e : list_dir(root / from_utf8(dir_rel))) {
        std::string name;
        if (!try_path_utf8(e.path().filename(), name)) {
            if (notes) note(*notes, "skipped an entry in " + printable(dir_rel) + ": its name is not representable as UTF-8");
            continue;
        }
        out.push_back(Listed{std::move(name), kind_of(e)});
    }
    return out;
}

bool tree_is_safe(const fs::path& root, const char* tree, std::vector<std::string>* notes) {
    std::error_code ec;
    const fs::path p = root / tree;
    const fs::file_status st = fs::symlink_status(p, ec);
    if (ec || !fs::exists(st)) return false;
    if (sf::is_link_like(p, st)) {
        if (notes) note(*notes, std::string("skipped ") + tree + ": a link or reparse point");
        return false;
    }
    return fs::is_directory(st);
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

void walk_tree(const fs::path& root, ManifestLayout layout, std::map<std::string, Found>& out,
               std::vector<std::string>& notes) {
    const char* tree = layout == ManifestLayout::v2 ? "manifests-v2" : "manifests";
    if (!tree_is_safe(root, tree, &notes)) return;
    auto skip_link = [&](const std::string& rel) { note(notes, "skipped " + printable(rel) + ": a link or reparse point"); };
    for (const auto& host_e : list_rel(root, tree, &notes)) {
        const std::string host_rel = std::string(tree) + "/" + host_e.name;
        if (host_e.kind == EntryKind::link) {
            skip_link(host_rel);
            continue;
        }
        if (host_e.kind != EntryKind::directory) continue;
        const std::string host = layout == ManifestLayout::v2 ? decode_v2_host(host_e.name) : host_e.name;
        for (const auto& ns_e : list_rel(root, host_rel, &notes)) {
            const std::string ns_rel = host_rel + "/" + ns_e.name;
            if (ns_e.kind == EntryKind::link) {
                skip_link(ns_rel);
                continue;
            }
            if (ns_e.kind != EntryKind::directory) continue;
            for (const auto& model_e : list_rel(root, ns_rel, &notes)) {
                const std::string model_rel = ns_rel + "/" + model_e.name;
                if (model_e.kind == EntryKind::link) {
                    skip_link(model_rel);
                    continue;
                }
                if (model_e.kind != EntryKind::directory) continue;
                for (const auto& tag_e : list_rel(root, model_rel, &notes)) {
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
std::optional<Found> locate_ci(const fs::path& root, const OllamaName& want) {
    const std::string want_host = canonical_host(want.host);
    const std::string want_ns = lower(want.ns), want_model = lower(want.model), want_tag = lower(want.tag);
    for (const ManifestLayout layout : {ManifestLayout::v2, ManifestLayout::legacy}) {
        const char* tree = layout == ManifestLayout::v2 ? "manifests-v2" : "manifests";
        if (!tree_is_safe(root, tree, nullptr)) continue;
        for (const auto& host_e : list_rel(root, tree, nullptr)) {
            if (host_e.kind != EntryKind::directory) continue;
            const std::string host = layout == ManifestLayout::v2 ? decode_v2_host(host_e.name) : host_e.name;
            if (canonical_host(host) != want_host) continue;
            const std::string host_rel = std::string(tree) + "/" + host_e.name;
            for (const auto& ns_e : list_rel(root, host_rel, nullptr)) {
                if (ns_e.kind != EntryKind::directory || lower(ns_e.name) != want_ns) continue;
                const std::string ns_rel = host_rel + "/" + ns_e.name;
                for (const auto& model_e : list_rel(root, ns_rel, nullptr)) {
                    if (model_e.kind != EntryKind::directory || lower(model_e.name) != want_model) continue;
                    const std::string model_rel = ns_rel + "/" + model_e.name;
                    for (const auto& tag_e : list_rel(root, model_rel, nullptr)) {
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
// Verified-blob cache (first_use)
// ---------------------------------------------------------------------------
struct OllamaStore::VerifiedCache {
    using Key = std::tuple<std::uint64_t, std::uint64_t, std::uint64_t, std::int64_t, std::int64_t>;

    static Key key_of(const FileStamp& s) { return Key{s.volume, s.file_id, s.size, s.write_time, s.change_time}; }

    // The SHA-256 a file with this identity hashed to, if known.
    std::optional<std::string> find(const FileStamp& s) {
        std::lock_guard<std::mutex> lock(mutex);
        const auto it = entries.find(key_of(s));
        if (it == entries.end()) return std::nullopt;
        return it->second;
    }

    void store(const FileStamp& s, std::string hex) {
        std::lock_guard<std::mutex> lock(mutex);
        if (entries.size() >= kMaxVerifiedEntries) entries.clear();
        entries[key_of(s)] = std::move(hex);
    }

    std::mutex mutex;
    std::map<Key, std::string> entries;
};

namespace {

// Verifies every weight blob of a ready record per the policy, from a fresh anchored handle
// whose identity must equal the stamp taken while the record was read.
void verify_weights_for_use(const fs::path& root, ModelRecord& r, const StoreOptions& options,
                            OllamaStore::VerifiedCache* cache, bool use_cache) {
    r.weights_verified = false;
    if (!r.ready() || options.verify_weights == WeightVerification::off) return;
    std::vector<char> buf;
    std::set<std::string> done;
    for (const auto& b : r.blobs) {
        if (!is_weight_role(b.role) || !done.insert(b.digest).second) continue;
        const std::string hex = b.digest.substr(7);
        const std::string label = std::string(to_string(b.role)) + " blob " + short_digest(hex);
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string err;
        const sf::OpenOutcome o = sf::open_anchored_file(root, blob_rel(hex), h, id, err);
        if (o != sf::OpenOutcome::ok) {
            mark(r, o == sf::OpenOutcome::unsafe ? UnsupportedReason::unsafe_path : UnsupportedReason::missing_blob,
                 label + " " + (o == sf::OpenOutcome::missing ? std::string("is not in blobs/") : open_failure_text(o)));
            return;
        }
        const FileStamp stamp = to_stamp(id);
        if (!b.stamp || *b.stamp != stamp) {
            mark(r, UnsupportedReason::blob_size_mismatch, label + " changed while the model was being resolved");
            return;
        }
        std::optional<std::string> actual;
        if (use_cache && cache != nullptr) actual = cache->find(stamp);
        if (!actual) {
            if (options.on_weight_hash) options.on_weight_hash(b);
            if (buf.empty()) buf.resize(kHashBufferBytes);
            std::string got;
            std::uint64_t bytes = 0;
            if (!hash_handle(h, buf, got, bytes, err, {})) {
                mark(r, UnsupportedReason::missing_blob, label + " cannot be read: " + printable(err));
                return;
            }
            sf::FileIdentity after;
            if (bytes != id.size || !sf::query_identity(h, after, err) || to_stamp(after) != stamp) {
                mark(r, UnsupportedReason::blob_size_mismatch, label + " changed while it was hashed");
                return;
            }
            if (cache != nullptr) cache->store(stamp, got);
            actual = std::move(got);
        }
        if (*actual != hex) {
            mark(r, UnsupportedReason::digest_mismatch, label + " does not hash to its digest");
            return;
        }
    }
    r.weights_verified = r.ready();
}

// True when every file the record was built from still has the identity it had then.
bool sources_unchanged(const ModelRecord& r) {
    for (const auto& s : r.sources) {
        std::string err;
        if (s.link) {
            std::string target;
            sf::FileIdentity link_id, target_id;
            if (!sf::read_anchored_leaf_symlink(r.store_root, s.rel, target, link_id, target_id, err) ||
                to_stamp(link_id) != s.stamp) {
                return false;
            }
            continue;
        }
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        if (sf::open_anchored_file(r.store_root, s.rel, h, id, err) != sf::OpenOutcome::ok || to_stamp(id) != s.stamp) {
            return false;
        }
    }
    for (const auto& b : r.blobs) {
        if (!b.stamp) {
            if (is_weight_role(b.role) || b.role == BlobRole::config || b.role == BlobRole::tmpl ||
                b.role == BlobRole::system || b.role == BlobRole::params || b.role == BlobRole::messages) {
                return false;  // a blob a ready record needs was never opened
            }
            continue;
        }
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string err;
        if (sf::open_anchored_file(r.store_root, blob_rel(b.digest.substr(7)), h, id, err) != sf::OpenOutcome::ok ||
            to_stamp(id) != *b.stamp) {
            return false;
        }
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
OllamaStore::OllamaStore(fs::path root, StoreOptions options)
    : root_(std::move(root)), options_(std::move(options)), cache_(std::make_shared<VerifiedCache>()) {}

Result<OllamaStore> OllamaStore::open(fs::path models_dir, StoreOptions options) {
    std::error_code ec;
    if (models_dir.empty() || !fs::is_directory(models_dir, ec) || ec) {
        return Status(ErrorCode::not_found, "the models directory is not a directory");
    }
    const bool v2 = fs::is_directory(models_dir / "manifests-v2", ec);
    const bool legacy = fs::is_directory(models_dir / "manifests", ec);
    if (!v2 && !legacy) {
        return Status(ErrorCode::invalid_argument,
                      "the directory has neither manifests-v2/ nor manifests/ (not an Ollama models directory)");
    }
    return OllamaStore(fs::absolute(models_dir, ec), std::move(options));
}

ScanResult OllamaStore::scan() const {
    std::map<std::string, Found> found;
    ScanResult result;
    walk_tree(root_, ManifestLayout::v2, found, result.notes);
    walk_tree(root_, ManifestLayout::legacy, found, result.notes);
    result.records.reserve(found.size());
    for (auto& [key, f] : found) {
        ModelRecord rec = build_record(root_, f.name, f.rel, f.layout, read_manifest(root_, f.rel));
        if (options_.verify_weights == WeightVerification::always) {
            verify_weights_for_use(root_, rec, options_, cache_.get(), false);
        }
        result.records.push_back(std::move(rec));
    }
    return result;
}

Result<ModelRecord> OllamaStore::resolve(std::string_view name) const {
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
        ModelRecord rec = build_record(root_, rec_name, rel, layout, std::move(mt));
        verify_weights_for_use(root_, rec, options_, cache_.get(), use_cache);
        return rec;
    };
    for (const auto& [rel, layout] : candidates) {
        if (!under_root(root_, root_ / from_utf8(rel))) continue;
        ManifestText mt = read_manifest(root_, rel);
        if (mt.outcome == sf::OpenOutcome::missing && mt.reason == UnsupportedReason::none) continue;
        return finish(n, rel, layout, std::move(mt));
    }
    // Case-insensitive fallback on case-sensitive file systems: list the matching directories
    // only (no manifest, config or GGUF reads except for the one entry that matches).
    if (auto found = locate_ci(root_, n)) {
        ManifestText mt = read_manifest(root_, found->rel);
        if (!(mt.outcome == sf::OpenOutcome::missing && mt.reason == UnsupportedReason::none)) {
            return finish(found->name, found->rel, found->layout, std::move(mt));
        }
    }
    return Status(ErrorCode::not_found, "model '" + n.display() + "' is not in the store");
}

bool OllamaStore::revalidate(ModelRecord& record) const {
    if (record.store_root != root_ || !sources_unchanged(record)) return false;
    verify_weights_for_use(root_, record, options_, cache_.get(),
                           options_.verify_weights == WeightVerification::first_use);
    return true;
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------
bool VerifyReport::ok() const {
    return std::all_of(blobs.begin(), blobs.end(), [](const BlobVerification& b) { return b.verdict == BlobVerdict::ok; });
}

VerifyReport verify_blobs(const ModelRecord& record, const VerifyOptions& options) {
    VerifyReport report;
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
        sf::OwnedFileHandle h;
        sf::FileIdentity id;
        std::string error;
        const sf::OpenOutcome o = sf::open_anchored_file(record.store_root, blob_rel(*hex), h, id, error);
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
    auto registered = [&] {
        for (const auto& s : stores_) {
            std::error_code ec;
            if (fs::equivalent(s.root(), store.value().root(), ec) && !ec) return true;
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

Result<ModelRecord> ModelRegistry::resolve(std::string_view name) const {
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
    // Store this result in the caches: a scan entry is replaced (so a broken model is not
    // re-resolved on every request), a live entry only kept while it is ready.
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
        if (!cached->ready()) return check(*cached);
        for (const auto& store : stores) {
            if (store.root() != cached->store_root) continue;
            ModelRecord rec = *cached;
            // Revalidation reads the disk (and may hash weights): no lock held.
            if (store.revalidate(rec)) {
                if (!rec.ready()) remember(rec, from_scan);
                return check(rec);
            }
            break;
        }
    }
    // Live lookup without the lock: it reads the disk (manifest and blob headers).
    for (const auto& store : stores) {
        auto live = store.resolve(name);
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
