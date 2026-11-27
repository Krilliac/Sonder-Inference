#include "identity.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include "sha256.hpp"
#include "sonder/inference/backends.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/launch_profile.hpp"

namespace sonder::inference::server::detail {

namespace {

// MOCK backend descriptors. The mock performs no inference, so its identity
// is a set of digests over fixed strings that describe exactly what it does:
// synthetic, but stable and honest. The server labels it synthetic.
constexpr const char* kMockTokenizer =
    "sonder-inference mock tokenizer v1: whitespace-separated words; word list from src/backends/mock_backend.cpp";
constexpr const char* kGenericChatTemplate =
    "sonder-inference generic chat prompt v1 (format_chat_prompt): '<Role>: <content>\\n\\n' per message, "
    "then 'Assistant:'";

constexpr std::size_t kHashChunkBytes = 65536;
constexpr std::size_t kMaxGgufIdentityCacheEntries = 16;

std::string cpu_hardware() {
    for (const auto& d : enumerate_devices()) {
        if (d.kind == DeviceKind::cpu) {
            return d.id + " " + d.name + " (" + host_platform() + ")";
        }
    }
    return host_platform();
}

std::string format_hardware_summary(const std::vector<GpuDeviceInfo>& gpus) {
    std::string out;
    for (const auto& gpu : gpus) {
        if (!out.empty()) out += "; ";
        if (!gpu.id.empty()) {
            out += gpu.id;
            out += " ";
        }
        if (!gpu.name.empty()) {
            out += gpu.name;
        } else {
            out += "unknown-gpu";
        }
        const std::optional<std::uint64_t> vram_mib = gpu_effective_vram_mib(gpu, false);
        if (vram_mib.has_value() && *vram_mib > 0) {
            out += " (" + std::to_string(*vram_mib) + " MiB)";
        }
    }
    const std::string cpu = cpu_hardware();
    if (!cpu.empty()) {
        if (!out.empty()) out += "; ";
        out += cpu;
    }
    if (out.empty()) {
        return host_platform();
    }
    return out;
}

std::string hardware_summary(const std::vector<GpuDeviceInfo>* gpus) {
    if (gpus != nullptr) {
        return format_hardware_summary(*gpus);
    }
    static std::mutex gpu_mu;
    static bool populated = false;
    static std::vector<GpuDeviceInfo> fallback_gpus;
    std::vector<GpuDeviceInfo> snapshot;
    {
        std::lock_guard<std::mutex> lock(gpu_mu);
        if (!populated) {
            fallback_gpus = enumerate_gpu_devices();
            populated = true;
        }
        snapshot = fallback_gpus;
    }
    return format_hardware_summary(snapshot);
}

bool is_readable_regular_file(const std::string& path) {
    if (path.empty()) return false;
    std::error_code ec;
    const std::filesystem::file_status st = std::filesystem::status(path_from_utf8(path), ec);
    return !ec && std::filesystem::is_regular_file(st);
}

bool is_hex_sha256(std::string_view value) noexcept {
    if (value.size() != 64) return false;
    for (const char ch : value) {
        const bool is_hex = (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        if (!is_hex) return false;
    }
    return true;
}

struct GgufIdentityCacheEntry {
    std::string path;
    std::uint64_t file_size = 0;
    std::filesystem::file_time_type mtime{};
    std::string model_digest;
    std::string tokenizer_digest;
    std::string template_digest;
    std::string quantization;
    std::uint64_t training_context_length = 0;
};

class GgufIdentityCache {
public:
    bool lookup(const std::string& path, std::uint64_t file_size, std::filesystem::file_time_type mtime,
                GgufIdentityCacheEntry& out) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->path == path && it->file_size == file_size && it->mtime == mtime) {
                out = *it;
                if (std::next(it) != entries_.end()) {
                    GgufIdentityCacheEntry promoted = std::move(*it);
                    entries_.erase(it);
                    entries_.push_back(std::move(promoted));
                }
                return true;
            }
        }
        return false;
    }

    void store(GgufIdentityCacheEntry entry) {
        std::lock_guard<std::mutex> lock(mu_);
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->path == entry.path) {
                entries_.erase(it);
                break;
            }
        }
        if (entries_.size() >= kMaxGgufIdentityCacheEntries) {
            entries_.erase(entries_.begin());
        }
        entries_.push_back(std::move(entry));
    }

private:
    std::mutex mu_;
    std::vector<GgufIdentityCacheEntry> entries_;
};

GgufIdentityCache& gguf_identity_cache() {
    static GgufIdentityCache cache;
    return cache;
}

}  // namespace

bool compute_gguf_file_sha256(const std::string& path, std::string& out_hex, std::string& error) {
    out_hex.clear();
    error.clear();
    if (path.empty()) {
        error = "GGUF path is empty";
        return false;
    }
    std::error_code ec;
    const std::filesystem::path fs_path = path_from_utf8(path);
    const std::filesystem::file_status st = std::filesystem::status(fs_path, ec);
    if (ec || !std::filesystem::exists(st)) {
        error = "GGUF file does not exist or cannot be stat'd";
        return false;
    }
    if (!std::filesystem::is_regular_file(st)) {
        error = "GGUF path is not a regular file";
        return false;
    }
    std::ifstream in(fs_path, std::ios::binary);
    if (!in) {
        error = "failed to open GGUF file for hashing";
        return false;
    }
    Sha256 hasher;
    std::vector<char> buf(kHashChunkBytes);
    while (in) {
        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
        const std::streamsize got = in.gcount();
        if (got > 0) {
            hasher.update(std::string_view(buf.data(), static_cast<std::size_t>(got)));
        }
    }
    if (in.bad() || !in.eof()) {
        error = "I/O error while reading GGUF file for hashing";
        return false;
    }
    const auto digest = hasher.finish();
    static constexpr char kHex[] = "0123456789abcdef";
    out_hex.reserve(64);
    for (const std::uint8_t byte : digest) {
        out_hex.push_back(kHex[byte >> 4]);
        out_hex.push_back(kHex[byte & 0x0Fu]);
    }
    return true;
}

IdentityResult gguf_backend_identity(const std::string& backend, const std::string& model_id,
                                     const std::string& gguf_path, std::uint64_t context_tokens,
                                     const std::optional<std::string>& backend_version,
                                     const std::vector<GpuDeviceInfo>* gpus) {
    IdentityResult out;
    if (backend.empty()) {
        out.reason = "backend name is empty";
        return out;
    }
    if (gguf_path.empty()) {
        out.reason = backend + ": GGUF path is empty";
        return out;
    }
    std::error_code status_ec;
    const std::filesystem::path fs_path = path_from_utf8(gguf_path);
    const std::filesystem::file_status st = std::filesystem::status(fs_path, status_ec);
    if (status_ec || !std::filesystem::exists(st)) {
        out.reason = backend + ": GGUF file '" + gguf_path + "' does not exist";
        return out;
    }
    if (!std::filesystem::is_regular_file(st)) {
        out.reason = backend + ": GGUF path '" + gguf_path + "' is not a regular file";
        return out;
    }
    if (!backend_version.has_value() || backend_version->empty()) {
        out.reason = backend + ": backend version cannot be measured";
        return out;
    }
    std::error_code pre_size_ec;
    const std::uint64_t pre_size = static_cast<std::uint64_t>(std::filesystem::file_size(fs_path, pre_size_ec));
    if (pre_size_ec) {
        out.reason = backend + ": failed to read size of GGUF file '" + gguf_path + "'";
        return out;
    }
    std::error_code pre_mtime_ec;
    const std::filesystem::file_time_type pre_mtime = std::filesystem::last_write_time(fs_path, pre_mtime_ec);
    if (pre_mtime_ec) {
        out.reason = backend + ": failed to read modification time of GGUF file '" + gguf_path + "'";
        return out;
    }

    GgufIdentityCacheEntry facts;
    const bool cached = gguf_identity_cache().lookup(gguf_path, pre_size, pre_mtime, facts);
    if (!cached) {
        const Result<GgufModelInfo> parsed = read_gguf_model_info(gguf_path);
        if (!parsed.ok()) {
            out.reason = backend + ": failed to parse GGUF header: " + parsed.status().message();
            return out;
        }
        const GgufModelInfo& info = parsed.value();
        if (info.tokenizer_digest.empty()) {
            out.reason = backend + ": tokenizer metadata missing in GGUF, so tokenizer_digest cannot be measured";
            return out;
        }
        if (info.template_digest.empty()) {
            out.reason = backend + ": tokenizer.chat_template missing in GGUF, so template_digest cannot be measured";
            return out;
        }
        if (info.quantization.empty()) {
            out.reason = backend + ": quantization cannot be measured from GGUF (general.file_type and tensors missing)";
            return out;
        }
        const std::uint64_t resolved_ctx = context_tokens > 0 ? context_tokens : info.training_context_length;
        if (resolved_ctx == 0) {
            out.reason = backend + ": context_tokens cannot be measured (no runtime context or GGUF context_length)";
            return out;
        }
        std::string model_digest;
        std::string hash_error;
        if (!compute_gguf_file_sha256(gguf_path, model_digest, hash_error)) {
            out.reason = backend + ": failed to hash GGUF file: " + hash_error;
            return out;
        }
        std::error_code post_size_ec;
        const std::uint64_t post_size = static_cast<std::uint64_t>(std::filesystem::file_size(fs_path, post_size_ec));
        std::error_code post_mtime_ec;
        const std::filesystem::file_time_type post_mtime = std::filesystem::last_write_time(fs_path, post_mtime_ec);
        if (post_size_ec || post_mtime_ec || post_size != pre_size || post_mtime != pre_mtime) {
            out.reason = backend + ": GGUF file changed while measuring identity";
            return out;
        }
        facts.path = gguf_path;
        facts.file_size = pre_size;
        facts.mtime = pre_mtime;
        facts.model_digest = std::move(model_digest);
        facts.tokenizer_digest = info.tokenizer_digest;
        facts.template_digest = info.template_digest;
        facts.quantization = info.quantization;
        facts.training_context_length = info.training_context_length;
        gguf_identity_cache().store(facts);
    }

    const std::uint64_t effective_context = context_tokens > 0 ? context_tokens : facts.training_context_length;
    if (effective_context == 0) {
        out.reason = backend + ": context_tokens cannot be measured (no runtime context or GGUF context_length)";
        return out;
    }
    if (!is_hex_sha256(facts.model_digest) || !is_hex_sha256(facts.tokenizer_digest) ||
        !is_hex_sha256(facts.template_digest)) {
        out.reason = backend + ": GGUF identity digests are incomplete";
        return out;
    }
    if (facts.quantization.empty()) {
        out.reason = backend + ": quantization cannot be measured from GGUF";
        return out;
    }
    std::string reported_model = model_id;
    if (reported_model.empty()) {
        reported_model = path_to_utf8(fs_path.filename());
    }
    if (reported_model.empty()) {
        reported_model = gguf_path;
    }
    const std::string hw = hardware_summary(gpus);
    if (hw.empty()) {
        out.reason = backend + ": hardware summary cannot be measured";
        return out;
    }
    out.backend_identity = json::Object{
        {"backend", backend},
        {"model", std::move(reported_model)},
        {"model_digest", facts.model_digest},
        {"quantization", facts.quantization},
        {"backend_version", *backend_version},
        {"tokenizer_digest", facts.tokenizer_digest},
        {"template_digest", facts.template_digest},
        {"context_tokens", effective_context},
        {"hardware", hw},
    };
    return out;
}

IdentityResult backend_identity(const Model& model, const std::optional<std::string>& backend_version) {
    return backend_identity(model.backend_name(), model.descriptor().name, &model.descriptor(), backend_version);
}

IdentityResult backend_identity(const std::string& backend, const std::string& model_id,
                                const ModelDescriptor* descriptor, const std::optional<std::string>& backend_version) {
    IdentityResult out;
    if (is_mock_backend(backend)) {
        if (descriptor == nullptr) {
            out.reason = "model '" + model_id +
                         "' is not loaded yet (lazy model residency): its identity is measured once a request "
                         "loads it";
            return out;
        }
        const ModelDescriptor& d = *descriptor;
        if (d.context_length == 0) {
            out.reason = "mock model reports no context length";
            return out;
        }
        out.backend_identity = json::Object{
            {"backend", backend},
            {"model", d.name},
            {"model_digest", sha256_hex("sonder-inference mock model v1: name=" + d.name + " format=" + d.format +
                                        " family=" + d.family + " quantization=" + d.quantization)},
            {"quantization", d.quantization.empty() ? std::string("none") : d.quantization},
            {"backend_version", backend_version.value_or("mock")},
            {"tokenizer_digest", sha256_hex(kMockTokenizer)},
            {"template_digest", sha256_hex(kGenericChatTemplate)},
            {"context_tokens", d.context_length},
            {"hardware", "mock backend on " + cpu_hardware() + "; no inference is performed"}};
        return out;
    }
    if (backend == "ollama") {
        out.reason =
            "ollama: the Ollama API does not expose a measurable tokenizer digest, so a complete backend identity "
            "cannot be reported (API v1 returns null rather than a partial or guessed identity)";
        return out;
    }
    if (backend == "llamacpp") {
        if (descriptor == nullptr) {
            out.reason = "model '" + model_id +
                         "' is not loaded yet (lazy model residency): its identity is measured once a request "
                         "loads it";
            return out;
        }
        const ModelDescriptor& d = *descriptor;
        std::string gguf_path;
        if (is_readable_regular_file(model_id)) {
            gguf_path = model_id;
        } else if (is_readable_regular_file(d.name)) {
            gguf_path = d.name;
        } else {
            const std::string label = !model_id.empty() ? model_id : d.name;
            out.reason = "llamacpp: GGUF file '" + label + "' is not a readable regular file";
            return out;
        }
        const std::string reported_model = !model_id.empty() ? model_id : d.name;
        return gguf_backend_identity(backend, reported_model, gguf_path, d.context_length, backend_version);
    }
    out.reason = backend + ": backend identity is not implemented for this backend";
    return out;
}

}  // namespace sonder::inference::server::detail
