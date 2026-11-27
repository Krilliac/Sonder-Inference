// Ollama model-store reader (docs/integration/model-store.md). Read-only.
//
// On-disk format sources (github.com/ollama/ollama, checked 2026-11-26 at main 8ae8acb):
//   manifest/paths.go      manifests-v2/<host>/<ns>/<model>/<tag> (':' in host as %3A), legacy
//                          manifests/..., resolution order (v2 first), digest "sha256[:-]<64 hex>",
//                          blobs/sha256-<hex>, public hosts registry.ollama.ai == ollama.com
//   manifest/manifest.go   Manifest JSON, manifest lists (mediaType
//                          application/vnd.ollama.manifest.list.v2+json), runner preference
//   manifest/layer.go      Layer JSON and the media type constants
//   server/images.go       GetModel(): how each layer type is consumed (model/shards, adapter,
//                          projector, template/prompt, system, params, messages, license)
//   types/model/name.go    name parsing, defaults and part validation
//   types/model/config.go  ConfigV2 (model_format, remote_host, ...)
//   api/types.go           Options / Runner JSON keys for the params layer
#include "sonder/inference/model_store.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iterator>
#include <map>
#include <set>
#include <system_error>

#include "common/sha256.hpp"

namespace sonder::inference::models {

namespace fs = std::filesystem;

namespace {

constexpr std::uint64_t kMaxManifestBytes = 4ull << 20;
constexpr std::uint64_t kMaxTextLayerBytes = 4ull << 20;
constexpr std::size_t kMaxHostLen = 350;
constexpr std::size_t kMaxPartLen = 80;

constexpr std::string_view kMediaManifestList = "application/vnd.ollama.manifest.list.v2+json";
constexpr std::string_view kMediaManifest = "application/vnd.docker.distribution.manifest.v2+json";

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

bool is_alnum_us(char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
}

enum class Part { host, ns, model, tag };

// types/model/name.go isValidPart / isValidLen.
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
    return true;
}

bool is_hex64(std::string_view s) {
    if (s.size() != 64) return false;
    return std::all_of(s.begin(), s.end(), [](char c) { return std::isxdigit(static_cast<unsigned char>(c)) != 0; });
}

// "sha256:<hex>" or "sha256-<hex>" (manifest/paths.go digestPattern) -> lowercase hex.
std::optional<std::string> digest_hex(std::string_view d) {
    if (d.size() != 7 + 64) return std::nullopt;
    if (d.substr(0, 6) != "sha256" || (d[6] != ':' && d[6] != '-')) return std::nullopt;
    const std::string_view hex = d.substr(7);
    if (!is_hex64(hex)) return std::nullopt;
    return lower(hex);
}

std::string path_utf8(const fs::path& p) {
    const auto u = p.generic_u8string();
    return std::string(reinterpret_cast<const char*>(u.data()), u.size());
}

std::string short_digest(std::string_view hex) { return "sha256:" + std::string(hex.substr(0, 12)); }

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

bool read_file_capped(const fs::path& path, std::uint64_t cap, std::string& out, std::string& error) {
    std::error_code ec;
    if (!fs::is_regular_file(path, ec) || ec) {
        error = "not a regular file";
        return false;
    }
    const std::uint64_t size = fs::file_size(path, ec);
    if (ec) {
        error = "cannot stat: " + ec.message();
        return false;
    }
    if (size > cap) {
        error = "larger than " + std::to_string(cap) + " bytes";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        error = "cannot open";
        return false;
    }
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

// ---------------------------------------------------------------------------
// Manifest parsing
// ---------------------------------------------------------------------------
struct LayerInfo {
    std::string media_type;
    std::string hex;  // empty when the digest is malformed
    std::string raw_digest;
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

LayerInfo parse_layer(const json::Value& v) {
    LayerInfo l;
    if (!v.is_object()) return l;
    if (const auto* m = v.find("mediaType"); m && m->is_string()) l.media_type = m->as_string();
    if (const auto* d = v.find("digest"); d && d->is_string()) {
        l.raw_digest = d->as_string();
        if (auto hex = digest_hex(l.raw_digest)) l.hex = *hex;
    }
    if (const auto* s = v.find("size")) l.size = s->as_uint(0);
    if (const auto* f = v.find("from"); f && f->is_string()) l.from = f->as_string();
    if (const auto* n = v.find("name"); n && n->is_string()) l.name = n->as_string();
    return l;
}

std::string str_field(const json::Value& v, std::string_view key) {
    const auto* f = v.find(key);
    return (f && f->is_string()) ? f->as_string() : std::string();
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
    if (const auto* cfg = v.find("config"); cfg && cfg->is_object()) {
        out.config = parse_layer(*cfg);
        out.has_config = !out.config.raw_digest.empty();
    }
    // Ollama marshals an empty Go slice as null: cloud stubs on disk carry "layers": null.
    const auto* layers = v.find("layers");
    if (layers != nullptr && !layers->is_null() && !layers->is_array()) {
        error = "manifest 'layers' is not an array";
        return false;
    }
    if (layers == nullptr || layers->is_null()) return true;
    for (const auto& l : layers->as_array()) {
        if (!l.is_object()) {
            error = "manifest layer is not an object";
            return false;
        }
        out.layers.push_back(parse_layer(l));
    }
    return true;
}

fs::path blob_path(const fs::path& root, std::string_view hex) { return root / "blobs" / ("sha256-" + std::string(hex)); }

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

void mark(ModelRecord& r, UnsupportedReason reason, std::string detail) {
    if (r.status == EntryStatus::unsupported) {
        r.warnings.push_back(std::move(detail));  // keep the first reason; later ones are context
        return;
    }
    r.status = EntryStatus::unsupported;
    r.reason = reason;
    r.detail = std::move(detail);
}

BlobRef make_blob(const fs::path& root, BlobRole role, const LayerInfo& l) {
    BlobRef b;
    b.role = role;
    b.media_type = l.media_type;
    b.digest = "sha256:" + l.hex;
    b.path = blob_path(root, l.hex);
    b.declared_size = l.size;
    b.from = l.from;
    std::error_code ec;
    if (fs::is_regular_file(b.path, ec) && !ec) {
        const std::uint64_t sz = fs::file_size(b.path, ec);
        if (!ec) b.disk_size = sz;
    }
    return b;
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

void parse_params(const json::Value& doc, ModelParams& p, std::vector<std::string>& warnings) {
    if (!doc.is_object()) {
        warnings.push_back("params layer is not a JSON object; ignored");
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
                warnings.push_back("params option '" + key + "' is not an integer; ignored");
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
                warnings.push_back("params option '" + key + "' is not a number; ignored");
            }
            break;
        }
        if (handled) continue;
        if (key == "stop") {
            if (value.is_string()) {
                p.stop.push_back(value.as_string());
            } else if (value.is_array()) {
                for (const auto& s : value.as_array()) {
                    if (s.is_string()) {
                        p.stop.push_back(s.as_string());
                    } else {
                        warnings.push_back("params 'stop' holds a non-string entry; ignored");
                    }
                }
            } else if (!value.is_null()) {
                warnings.push_back("params option 'stop' is not an array of strings; ignored");
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

// True when the layer text was read into `out`; false (with a reason recorded on `r`) otherwise.
bool read_text_layer(const BlobRef& blob, std::string& out, ModelRecord& r, const char* what) {
    std::string err;
    if (!blob.disk_size) {
        mark(r, UnsupportedReason::missing_blob, std::string(what) + " blob " + blob.digest + " is not in blobs/");
        return false;
    }
    if (!read_file_capped(blob.path, kMaxTextLayerBytes, out, err)) {
        r.warnings.push_back(std::string(what) + " layer unreadable: " + err);
        out.clear();
        return false;
    }
    return true;
}

// Selects the manifest-list child this engine can use: llamacpp first, then ggml (the same
// order as manifest/manifest.go runnerPreferences on non-Apple platforms, minus mlx).
// Returns false and sets reason/detail when none fits.
bool select_child(const json::Value& list, const fs::path& root, json::Value& child_doc, UnsupportedReason& reason,
                  std::string& detail) {
    const auto* children = list.find("manifests");
    if (children == nullptr || !children->is_array()) {
        reason = UnsupportedReason::manifest_invalid;
        detail = "manifest list has no manifests array";
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
            const auto hex = digest_hex(str_field(c, "digest"));
            if (!hex) {
                reason = UnsupportedReason::manifest_invalid;
                detail = "manifest list child has a malformed digest";
                return false;
            }
            const fs::path p = blob_path(root, *hex);
            std::string text, err;
            if (!read_file_capped(p, kMaxManifestBytes, text, err)) {
                if (skipped_reason == UnsupportedReason::none) {
                    skipped_reason = UnsupportedReason::missing_blob;
                    skipped_detail = "manifest list child " + short_digest(*hex) + " is not readable: " + err;
                }
                continue;
            }
            if (detail::sha256_hex(text) != *hex) {
                reason = UnsupportedReason::manifest_invalid;
                detail = "manifest list child " + short_digest(*hex) + " does not match its digest";
                return false;
            }
            auto parsed = json::parse(text, json::ParseLimits{64, 100000});
            if (!parsed.ok()) {
                reason = UnsupportedReason::manifest_invalid;
                detail = "manifest list child is not valid JSON";
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
        detail = std::move(skipped_detail);
        return false;
    }
    for (const auto& c : children->as_array()) {
        if (c.is_object() && (lower(str_field(c, "runner")) == "mlx" || lower(str_field(c, "format")) == "safetensors")) {
            saw_mlx = true;
        }
    }
    if (saw_mlx) {
        reason = UnsupportedReason::safetensors;
        detail = "manifest list only offers safetensors/MLX weights; SI's native engine loads GGUF";
    } else {
        reason = UnsupportedReason::no_compatible_manifest;
        detail = "manifest list has no llamacpp or ggml GGUF child";
    }
    return false;
}

ModelRecord load_record(const fs::path& root, const OllamaName& name, const fs::path& manifest_path,
                        ManifestLayout layout) {
    ModelRecord r;
    r.name = name;
    r.store_root = root;
    r.manifest_path = manifest_path;
    r.layout = layout;

    std::string text, err;
    if (!read_file_capped(manifest_path, kMaxManifestBytes, text, err)) {
        mark(r, UnsupportedReason::manifest_invalid, "cannot read manifest: " + err);
        return r;
    }
    auto doc = json::parse(text, json::ParseLimits{64, 100000});
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
        std::string detail;
        if (!select_child(manifest_doc, root, child, reason, detail)) {
            mark(r, reason, detail);
            return r;
        }
        manifest_doc = std::move(child);
    }

    ParsedManifest pm;
    if (!parse_manifest_object(manifest_doc, pm, err)) {
        mark(r, UnsupportedReason::manifest_invalid, err);
        return r;
    }
    r.runner = pm.runner;
    r.format = pm.format;

    // Config blob (model format, cloud stub marker, renderer/parser).
    if (pm.has_config) {
        if (pm.config.hex.empty()) {
            r.warnings.push_back("config layer has a malformed digest");
        } else {
            BlobRef cfg = make_blob(root, BlobRole::config, pm.config);
            std::string cfg_text, cfg_err;
            if (cfg.disk_size && read_file_capped(cfg.path, kMaxTextLayerBytes, cfg_text, cfg_err)) {
                auto cj = json::parse(cfg_text, json::ParseLimits{64, 100000});
                if (cj.ok()) {
                    parse_config(cj.value(), r.config);
                } else {
                    r.warnings.push_back("config blob is not valid JSON");
                }
            } else {
                r.warnings.push_back("config blob " + cfg.digest + " is not readable");
            }
            r.referenced_bytes += cfg.declared_size;
            r.blobs.push_back(std::move(cfg));
        }
    }
    if (r.format.empty()) r.format = r.config.model_format;

    bool saw_tensor = false;
    for (const auto& l : pm.layers) {
        if (l.media_type == "application/vnd.ollama.image.tensor") {
            saw_tensor = true;
            continue;
        }
        if (l.media_type == kMediaManifest || l.media_type == kMediaManifestList) {
            continue;  // downgrade anchor (manifest.go WriteLegacyAnchor), not content
        }
        if (l.hex.empty()) {
            mark(r, UnsupportedReason::manifest_invalid,
                 "layer '" + l.media_type + "' has a malformed digest '" + l.raw_digest + "'");
            continue;
        }
        const BlobRole role = role_for(l.media_type);
        if (role == BlobRole::other) {
            if (l.media_type == "application/vnd.ollama.image.embed") {
                r.warnings.push_back("embed layer is deprecated in Ollama and ignored");
            } else if (l.media_type != "application/vnd.ollama.image.json") {
                r.warnings.push_back("unrecognized layer media type '" + l.media_type + "' ignored");
            }
            continue;
        }
        BlobRef blob = make_blob(root, role, l);
        r.referenced_bytes += blob.declared_size;
        if (is_weight_role(role)) r.weights_bytes += blob.declared_size;
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
            case BlobRole::tmpl: {
                std::string body;
                if (read_text_layer(blob, body, r, "template")) r.template_text = std::move(body);
                break;
            }
            case BlobRole::system: {
                std::string body;
                if (read_text_layer(blob, body, r, "system")) r.system_prompt = std::move(body);
                break;
            }
            case BlobRole::params: {
                std::string body;
                if (read_text_layer(blob, body, r, "params") && !body.empty()) {
                    auto pj = json::parse(body, json::ParseLimits{64, 100000});
                    if (pj.ok()) {
                        parse_params(pj.value(), r.params, r.warnings);
                    } else {
                        r.warnings.push_back("params layer is not valid JSON; ignored");
                    }
                }
                break;
            }
            case BlobRole::messages: {
                std::string body;
                if (read_text_layer(blob, body, r, "messages") && !body.empty()) {
                    if (json::parse(body, json::ParseLimits{64, 100000}).ok()) {
                        r.messages_json = std::move(body);
                    } else {
                        r.warnings.push_back("messages layer is not valid JSON; ignored");
                    }
                }
                break;
            }
            case BlobRole::license:
                ++r.license_count;
                break;
            default:
                break;
        }
        r.blobs.push_back(std::move(blob));
    }

    // Classification, most specific first.
    if (!r.config.remote_host.empty() || !r.config.remote_model.empty()) {
        mark(r, UnsupportedReason::cloud_stub,
             "Ollama Cloud entry (remote model '" + r.config.remote_model + "' at '" + r.config.remote_host +
                 "'); it has no local weights and is reached over HTTPS, not served from the store");
    }
    if (r.status == EntryStatus::ready &&
        (r.config.model_format == "safetensors" || r.format == "safetensors" || r.runner == "mlx" || saw_tensor)) {
        mark(r, UnsupportedReason::safetensors,
             "safetensors/MLX weights; SI's native engine loads GGUF only");
    }
    if (r.status == EntryStatus::ready && !r.format.empty() && r.format != "gguf") {
        mark(r, UnsupportedReason::unsupported_format, "weight format '" + r.format + "' is not GGUF");
    }
    if (r.status == EntryStatus::ready && r.model_path.empty()) {
        mark(r, UnsupportedReason::no_model_layer, "manifest has no image.model layer");
    }
    if (r.status == EntryStatus::ready) {
        for (const auto& b : r.blobs) {
            if (!is_weight_role(b.role)) continue;
            if (!b.disk_size) {
                mark(r, UnsupportedReason::missing_blob,
                     std::string(to_string(b.role)) + " blob " + short_digest(b.digest.substr(7)) + " is not in blobs/");
                break;
            }
            if (*b.disk_size != b.declared_size) {
                mark(r, UnsupportedReason::blob_size_mismatch,
                     std::string(to_string(b.role)) + " blob " + short_digest(b.digest.substr(7)) + " is " +
                         std::to_string(*b.disk_size) + " bytes on disk, manifest says " +
                         std::to_string(b.declared_size));
                break;
            }
            auto header = read_gguf_header(b.path);
            if (!header.ok()) {
                mark(r, UnsupportedReason::invalid_gguf,
                     std::string(to_string(b.role)) + " blob " + short_digest(b.digest.substr(7)) + ": " +
                         header.status().message());
                break;
            }
            if (b.role == BlobRole::model && b.path == r.model_path) r.gguf = header.value();
        }
    }
    // Missing text layers are recorded by read_text_layer; a blob that is simply absent for
    // a weight layer is handled above, so a ready record never has a dangling path.
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

bool is_dir(const fs::directory_entry& e) {
    std::error_code ec;
    return e.is_directory(ec) && !ec;
}

bool is_file(const fs::directory_entry& e) {
    std::error_code ec;
    return e.is_regular_file(ec) && !ec;
}

void walk_tree(const fs::path& root, ManifestLayout layout, std::map<std::string, ModelRecord>& out,
               std::vector<std::string>& notes) {
    const fs::path tree = root / (layout == ManifestLayout::v2 ? "manifests-v2" : "manifests");
    for (const auto& host_e : list_dir(tree)) {
        if (!is_dir(host_e)) continue;
        std::string host = path_utf8(host_e.path().filename());
        if (layout == ManifestLayout::v2) host = decode_v2_host(host);
        for (const auto& ns_e : list_dir(host_e.path())) {
            if (!is_dir(ns_e)) continue;
            for (const auto& model_e : list_dir(ns_e.path())) {
                if (!is_dir(model_e)) continue;
                for (const auto& tag_e : list_dir(model_e.path())) {
                    if (!is_file(tag_e)) continue;
                    OllamaName n{host, path_utf8(ns_e.path().filename()), path_utf8(model_e.path().filename()),
                                 path_utf8(tag_e.path().filename())};
                    if (!valid_part(Part::host, n.host) || !valid_part(Part::ns, n.ns) ||
                        !valid_part(Part::model, n.model) || !valid_part(Part::tag, n.tag)) {
                        notes.push_back("skipped " + path_utf8(tag_e.path()) + ": not a valid model name");
                        continue;
                    }
                    if (is_default_host(n.host)) n.host = kOllamaDefaultHost;
                    const std::string key = n.key();
                    if (out.count(key) != 0) continue;  // manifests-v2 is walked first and wins
                    out.emplace(key, load_record(root, n, tag_e.path(), layout));
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

struct Located {
    OllamaName name;
    fs::path manifest;
    ManifestLayout layout = ManifestLayout::v2;
};

// Case-insensitive lookup of one name by listing only the directories on its path (hosts,
// then the matching namespace/model/tag). It reads nothing besides directory entries, so a
// miss costs a handful of listings instead of a full scan. Layout order and the first-match
// rule are the same as scan(), so both pick the same entry.
std::optional<Located> locate_ci(const fs::path& root, const OllamaName& want) {
    const std::string want_host = canonical_host(want.host);
    const std::string want_ns = lower(want.ns), want_model = lower(want.model), want_tag = lower(want.tag);
    for (const ManifestLayout layout : {ManifestLayout::v2, ManifestLayout::legacy}) {
        const fs::path tree = root / (layout == ManifestLayout::v2 ? "manifests-v2" : "manifests");
        for (const auto& host_e : list_dir(tree)) {
            if (!is_dir(host_e)) continue;
            std::string host = path_utf8(host_e.path().filename());
            if (layout == ManifestLayout::v2) host = decode_v2_host(host);
            if (canonical_host(host) != want_host) continue;
            for (const auto& ns_e : list_dir(host_e.path())) {
                if (!is_dir(ns_e) || lower(path_utf8(ns_e.path().filename())) != want_ns) continue;
                for (const auto& model_e : list_dir(ns_e.path())) {
                    if (!is_dir(model_e) || lower(path_utf8(model_e.path().filename())) != want_model) continue;
                    for (const auto& tag_e : list_dir(model_e.path())) {
                        if (!is_file(tag_e) || lower(path_utf8(tag_e.path().filename())) != want_tag) continue;
                        OllamaName n{host, path_utf8(ns_e.path().filename()), path_utf8(model_e.path().filename()),
                                     path_utf8(tag_e.path().filename())};
                        if (!valid_part(Part::host, n.host) || !valid_part(Part::ns, n.ns) ||
                            !valid_part(Part::model, n.model) || !valid_part(Part::tag, n.tag)) {
                            continue;
                        }
                        if (is_default_host(n.host)) n.host = kOllamaDefaultHost;
                        return Located{std::move(n), tag_e.path(), layout};
                    }
                }
            }
        }
    }
    return std::nullopt;
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
        return Status(ErrorCode::invalid_argument, "invalid model name '" + std::string(text) + "': " + why);
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

const char* to_string(BlobVerdict verdict) noexcept {
    switch (verdict) {
        case BlobVerdict::ok: return "ok";
        case BlobVerdict::mismatch: return "mismatch";
        case BlobVerdict::missing: return "missing";
        case BlobVerdict::io_error: return "io_error";
        case BlobVerdict::cancelled: return "cancelled";
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
    std::ifstream in(file, std::ios::binary);
    if (!in) return Status(ErrorCode::io_error, "cannot open file");
    std::array<unsigned char, 24> buf{};
    in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    if (static_cast<std::size_t>(in.gcount()) < buf.size()) {
        return Status(ErrorCode::invalid_argument, "file is shorter than a GGUF header");
    }
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

// ---------------------------------------------------------------------------
// OllamaStore
// ---------------------------------------------------------------------------
Result<OllamaStore> OllamaStore::open(fs::path models_dir) {
    std::error_code ec;
    if (models_dir.empty() || !fs::is_directory(models_dir, ec) || ec) {
        return Status(ErrorCode::not_found, "models directory '" + path_utf8(models_dir) + "' is not a directory");
    }
    const bool v2 = fs::is_directory(models_dir / "manifests-v2", ec);
    const bool legacy = fs::is_directory(models_dir / "manifests", ec);
    if (!v2 && !legacy) {
        return Status(ErrorCode::invalid_argument,
                      "'" + path_utf8(models_dir) + "' has neither manifests-v2/ nor manifests/ (not an Ollama models directory)");
    }
    return OllamaStore(fs::absolute(models_dir, ec));
}

ScanResult OllamaStore::scan() const {
    std::map<std::string, ModelRecord> found;
    ScanResult result;
    walk_tree(root_, ManifestLayout::v2, found, result.notes);
    walk_tree(root_, ManifestLayout::legacy, found, result.notes);
    result.records.reserve(found.size());
    for (auto& [key, rec] : found) result.records.push_back(std::move(rec));
    return result;
}

Result<ModelRecord> OllamaStore::resolve(std::string_view name) const {
    auto parsed = parse_ollama_name(name);
    if (!parsed.ok()) return parsed.status();
    const OllamaName n = parsed.value();

    std::vector<std::pair<fs::path, ManifestLayout>> candidates;
    if (is_default_host(n.host)) {
        candidates.emplace_back(root_ / "manifests-v2" / "ollama.com" / n.ns / n.model / n.tag, ManifestLayout::v2);
        candidates.emplace_back(root_ / "manifests-v2" / kOllamaDefaultHost / n.ns / n.model / n.tag, ManifestLayout::v2);
        candidates.emplace_back(root_ / "manifests" / kOllamaDefaultHost / n.ns / n.model / n.tag, ManifestLayout::legacy);
        candidates.emplace_back(root_ / "manifests" / "ollama.com" / n.ns / n.model / n.tag, ManifestLayout::legacy);
    } else {
        candidates.emplace_back(root_ / "manifests-v2" / encode_v2_host(n.host) / n.ns / n.model / n.tag, ManifestLayout::v2);
        // A host with a port has no legacy directory form (':' is not a path character on
        // Windows, where "c:5000" would also be read as a drive-relative path). Such names are
        // served only through the v2 tree (%3A); a legacy directory that exists on another OS
        // is still found by the directory lookup below.
        if (n.host.find(':') == std::string::npos) {
            candidates.emplace_back(root_ / "manifests" / n.host / n.ns / n.model / n.tag, ManifestLayout::legacy);
        }
    }
    for (const auto& [path, layout] : candidates) {
        if (!under_root(root_, path)) continue;
        std::error_code ec;
        if (fs::is_regular_file(path, ec) && !ec) return load_record(root_, n, path, layout);
    }
    // Case-insensitive fallback on case-sensitive file systems: list the matching directories
    // only (no manifest, config or GGUF reads except for the one entry that matches).
    if (auto found = locate_ci(root_, n)) return load_record(root_, found->name, found->manifest, found->layout);
    return Status(ErrorCode::not_found, "model '" + n.display() + "' is not in '" + path_utf8(root_) + "'");
}

// ---------------------------------------------------------------------------
// Verification
// ---------------------------------------------------------------------------
bool VerifyReport::ok() const {
    return std::all_of(blobs.begin(), blobs.end(), [](const BlobVerification& b) { return b.verdict == BlobVerdict::ok; });
}

VerifyReport verify_blobs(const ModelRecord& record, const VerifyOptions& options) {
    VerifyReport report;
    for (const auto& blob : record.blobs) {
        if (options.weights_only && !is_weight_role(blob.role)) continue;
        BlobVerification v;
        v.blob = blob;
        if (options.cancel.cancelled()) {
            v.verdict = BlobVerdict::cancelled;
            v.detail = "cancelled before hashing";
            report.blobs.push_back(std::move(v));
            break;
        }
        std::error_code ec;
        if (!fs::is_regular_file(blob.path, ec) || ec) {
            v.verdict = BlobVerdict::missing;
            v.detail = "blob file is not present";
            report.blobs.push_back(std::move(v));
            continue;
        }
        if (options.on_blob_start) options.on_blob_start(blob);
        std::string hex, error;
        const bool done = detail::sha256_file_hex(blob.path, hex, error, [&options] { return options.cancel.cancelled(); });
        if (!done) {
            v.verdict = options.cancel.cancelled() ? BlobVerdict::cancelled : BlobVerdict::io_error;
            v.detail = error;
            const bool stop = v.verdict == BlobVerdict::cancelled;
            report.blobs.push_back(std::move(v));
            if (stop) break;
            continue;
        }
        v.actual_sha256 = hex;
        if (blob.digest == "sha256:" + hex) {
            v.verdict = BlobVerdict::ok;
        } else {
            v.verdict = BlobVerdict::mismatch;
            v.detail = "content hashes to sha256:" + hex;
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
    for (const auto& p : paths) a.emplace_back(path_utf8(p));
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
    o.set("manifest", json::Object{{"path", path_utf8(r.manifest_path)},
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
                                 {"remote_host", r.config.remote_host},
                                 {"remote_model", r.config.remote_model},
                                 {"capabilities", string_array(r.config.capabilities)},
                                 {"context_length", json::Value(static_cast<long long>(r.config.context_length))},
                                 {"embedding_length", json::Value(static_cast<long long>(r.config.embedding_length))}});
    json::Object files;
    files.set("model", r.model_path.empty() ? json::Value() : json::Value(path_utf8(r.model_path)));
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
    o.set("warnings", string_array(r.warnings));
    return json::Value(std::move(o));
}

json::Value to_json(const ScanResult& scan, const fs::path& store_root) {
    json::Array models;
    for (const auto& r : scan.records) models.push_back(to_json(r));
    return json::Value(json::Object{{"schema", kStoreJsonSchema},
                                    {"source", json::Object{{"kind", "ollama"}, {"dir", path_utf8(store_root)}}},
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
    auto store = OllamaStore::open(std::move(models_dir));
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
    std::vector<OllamaStore> stores;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& rec : records_) {
            if (rec.name.key() == key) return check(rec);
        }
        stores = stores_;
    }
    // Live lookup without the lock: it reads the disk (manifest and blob headers).
    for (const auto& store : stores) {
        auto live = store.resolve(name);
        if (live.ok()) return check(live.value());
    }
    return Status(ErrorCode::not_found, "model '" + parsed.value().display() + "' is not in any registered store");
}

}  // namespace sonder::inference::models
