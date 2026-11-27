// Sonder Inference: model store that serves an existing Ollama model library
// in place (docs/integration/model-store.md).
//
// Ollama keeps models as a manifest tree (manifests-v2/<host>/<ns>/<name>/<tag>
// and the legacy manifests/ tree) that names content-addressed blobs
// (blobs/sha256-<hex>). This module reads that layout READ-ONLY: it never
// writes, renames, links or deletes anything in the store, and it never copies
// a blob. A resolved record carries absolute paths into the Ollama blob
// directory so a native backend can mmap the GGUF where it already is.
//
// Cost model: a scan walks the manifest directories and reads the small
// manifest, config, params, template and system files plus the first 24 bytes
// of each GGUF-class blob. It never hashes a model blob. SHA-256 verification
// of multi-GB files is a separate, opt-in call (verify_blobs).
//
// Thread affinity: every free function and OllamaStore method is reentrant and
// holds no global state. ModelRegistry serializes its own state with a mutex;
// records it returns are independent copies.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::models {

inline constexpr const char* kOllamaDefaultHost = "registry.ollama.ai";
inline constexpr const char* kOllamaDefaultNamespace = "library";
inline constexpr const char* kOllamaDefaultTag = "latest";
inline constexpr const char* kStoreJsonSchema = "sonder.inference.model_store/1";

// A model name in Ollama's four-part form. Parsing fills missing parts with the
// same defaults as Ollama's model.ParseName (registry.ollama.ai / library /
// latest). Hosts ollama.com and registry.ollama.ai are the same registry.
struct OllamaName {
    std::string host;
    std::string ns;  // namespace
    std::string model;
    std::string tag;

    // host/ns/model:tag
    [[nodiscard]] std::string full() const;
    // Ollama's DisplayShortest: omits the default host and the "library" namespace.
    [[nodiscard]] std::string display() const;
    // Lowercase identity used for matching: the two public hosts compare equal.
    [[nodiscard]] std::string key() const;
};

// Accepts "model", "model:tag", "ns/model[:tag]", "host/ns/model[:tag]" (an
// "https://" prefix on the host is dropped). Validates each part with Ollama's
// rules (types/model/name.go isValidPart).
Result<OllamaName> parse_ollama_name(std::string_view text);

enum class EntryStatus { ready, unsupported };

enum class UnsupportedReason {
    none,
    manifest_invalid,         // unreadable or malformed manifest
    no_compatible_manifest,   // manifest list without a llamacpp/ggml child
    cloud_stub,               // Ollama Cloud entry: no local weights (config.remote_host)
    safetensors,              // safetensors/MLX weights; SI's native engine loads GGUF only
    unsupported_format,       // a weight format other than GGUF
    no_model_layer,           // no image.model layer and not a cloud stub
    missing_blob,             // a referenced blob is absent from blobs/
    blob_size_mismatch,       // blob size differs from the manifest (partial copy)
    invalid_gguf,             // blob does not start with a supported GGUF header
};
const char* to_string(UnsupportedReason reason) noexcept;
const char* to_string(EntryStatus status) noexcept;

enum class BlobRole { config, model, projector, adapter, draft, tmpl, system, params, messages, license, other };
const char* to_string(BlobRole role) noexcept;

struct BlobRef {
    BlobRole role = BlobRole::other;
    std::string media_type;
    std::string digest;  // "sha256:<64 lowercase hex>"
    std::filesystem::path path;
    std::uint64_t declared_size = 0;           // from the manifest
    std::optional<std::uint64_t> disk_size;    // nullopt when the blob is absent
    std::string from;                          // manifest "from" (parent model), may be empty
};

struct GgufHeader {
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t kv_count = 0;
};

// Reads and checks the first 24 bytes (magic, version 2 or 3, counts). Version 1
// is rejected the way llama.cpp rejects it. Reads nothing else.
Result<GgufHeader> read_gguf_header(const std::filesystem::path& file);

// The generation/runtime options from the params layer (Ollama's api.Options and
// api.Runner fields; types/keys per api/types.go). Unknown keys and the
// options not modelled here (main_gpu, use_mmap, draft_num_predict) stay
// verbatim in `extra`.
struct ModelParams {
    std::optional<std::int64_t> num_ctx, num_batch, num_gpu, num_thread, num_keep, seed, num_predict, top_k,
        repeat_last_n;
    std::optional<double> temperature, top_p, min_p, typical_p, repeat_penalty, presence_penalty, frequency_penalty;
    std::vector<std::string> stop;
    json::Object extra;

    [[nodiscard]] bool empty() const;
};

// The model config blob (Ollama's types/model ConfigV2), the fields SI uses.
struct ModelConfig {
    std::string model_format;  // "gguf", "safetensors", or empty
    std::string model_family;
    std::vector<std::string> model_families;
    std::string parameter_size;  // config "model_type", e.g. "8.2B"
    std::string quantization;    // config "file_type", e.g. "Q4_K_M"
    std::string renderer;        // Ollama-side prompt renderer id (no template layer when set)
    std::string parser;          // Ollama-side output parser id
    std::string requires_version;
    std::string remote_host;     // cloud stubs only
    std::string remote_model;    // cloud stubs only
    std::vector<std::string> capabilities;
    std::int64_t context_length = 0;
    std::int64_t embedding_length = 0;
};

enum class ManifestLayout { v2, legacy };
const char* to_string(ManifestLayout layout) noexcept;

// Everything known about one store entry. A `ready` record has every blob
// present, sized as declared, and starting with a supported GGUF header; an
// `unsupported` record explains why and still carries whatever was parsed.
struct ModelRecord {
    OllamaName name;
    EntryStatus status = EntryStatus::ready;
    UnsupportedReason reason = UnsupportedReason::none;
    std::string detail;  // human-readable reason text (or empty when ready)

    std::filesystem::path store_root;
    std::filesystem::path manifest_path;
    ManifestLayout layout = ManifestLayout::v2;
    std::string runner;  // manifest "runner" (llamacpp | ggml | mlx), empty for legacy manifests
    std::string format;  // manifest "format", empty for legacy manifests
    std::string parent;  // "from" of the model layer (the model this one was built on), if any

    ModelConfig config;

    // Weight files. model_path is the first image.model layer; any further model
    // layers are shard_paths in manifest order. All paths point into <store>/blobs.
    std::filesystem::path model_path;
    std::vector<std::filesystem::path> shard_paths;
    std::vector<std::filesystem::path> projector_paths;
    std::vector<std::filesystem::path> adapter_paths;
    std::vector<std::filesystem::path> draft_paths;
    std::optional<GgufHeader> gguf;  // header of model_path

    ModelParams params;
    std::optional<std::string> template_text;  // image.template (or legacy image.prompt)
    std::optional<std::string> system_prompt;  // image.system
    std::optional<std::string> messages_json;  // image.messages, validated JSON text
    std::size_t license_count = 0;

    std::uint64_t weights_bytes = 0;     // declared size of model + shard + projector + adapter + draft layers
    std::uint64_t referenced_bytes = 0;  // declared size of every layer plus the config
    std::vector<BlobRef> blobs;          // every referenced blob (config first), for verification

    std::vector<std::string> warnings;

    [[nodiscard]] bool ready() const noexcept { return status == EntryStatus::ready; }
};

struct ScanResult {
    std::vector<ModelRecord> records;  // sorted by name key; v2 entries win over legacy duplicates
    std::vector<std::string> notes;    // paths skipped during the walk (invalid names, special files)
};

// One Ollama models directory. Cheap to copy; holds only the root path.
class OllamaStore {
public:
    // Fails with not_found when `models_dir` is not a directory and with
    // invalid_argument when it holds neither manifests-v2/ nor manifests/.
    static Result<OllamaStore> open(std::filesystem::path models_dir);

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }

    // Loads every manifest. A broken entry becomes an `unsupported` record; the scan
    // itself does not fail.
    [[nodiscard]] ScanResult scan() const;

    // Resolves "name[:tag]" the way Ollama's resolveManifestPath does: manifests-v2 first,
    // then the legacy tree (also under the alternate public host). Returns not_found when
    // no manifest exists; an unsupported entry is returned as a record, not an error.
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name) const;

private:
    explicit OllamaStore(std::filesystem::path root) : root_(std::move(root)) {}
    std::filesystem::path root_;
};

// Opt-in content verification: re-hashes blobs and compares with their digest.
enum class BlobVerdict { ok, mismatch, missing, io_error, cancelled };
const char* to_string(BlobVerdict verdict) noexcept;

struct BlobVerification {
    BlobRef blob;
    BlobVerdict verdict = BlobVerdict::ok;
    std::string actual_sha256;  // lowercase hex when hashed
    std::string detail;
};

struct VerifyOptions {
    CancellationToken cancel;
    // Called on the calling thread before each blob is hashed (a model blob can take
    // minutes), so a CLI can show progress.
    std::function<void(const BlobRef&)> on_blob_start;
    // Hash only weight-class blobs (model, projector, adapter, draft); skip the small layers.
    bool weights_only = false;
};

struct VerifyReport {
    std::vector<BlobVerification> blobs;
    [[nodiscard]] bool ok() const;
};

// Streams each blob through SHA-256 (64 KiB buffer, no allocation per blob).
// Stops after the first cancelled blob. Does not modify the record or the store.
VerifyReport verify_blobs(const ModelRecord& record, const VerifyOptions& options = {});

// Stable JSON for a record / a scan (schema kStoreJsonSchema).
json::Value to_json(const ModelRecord& record);
json::Value to_json(const ScanResult& scan, const std::filesystem::path& store_root);

// The directory Ollama itself would use: $OLLAMA_MODELS, else <home>/.ollama/models.
// Empty when neither can be determined. Reading the variable never touches the store.
std::filesystem::path default_ollama_models_dir();

// Name -> record lookup over one or more Ollama stores. This is the hook the native
// llamacpp backend will use to load a model by name (docs/integration/model-store.md):
//
//   ModelRegistry reg; reg.add_ollama_dir(dir);
//   auto rec = reg.resolve("qwen3.5:4b");   // ready records only
//   backend.load(rec.value().model_path, rec.value().projector_paths, rec.value().params ...)
//
// The backend is not wired to it yet. Thread-safe; not copyable.
class ModelRegistry {
public:
    ModelRegistry() = default;
    ModelRegistry(const ModelRegistry&) = delete;
    ModelRegistry& operator=(const ModelRegistry&) = delete;

    // Adds a store (duplicates by path are ignored) and scans it. Earlier stores win
    // when two stores hold the same name.
    Status add_ollama_dir(std::filesystem::path models_dir);

    // Rescans every store (directory walk plus small reads; no hashing).
    void refresh();

    [[nodiscard]] std::vector<ModelRecord> list() const;

    // A ready record for `name`. not_found when unknown; unsupported (with the reason in
    // the message) when the entry exists but cannot be served locally. A miss in the
    // cached scan falls back to resolving directly against each store, so a model pulled
    // after the last refresh() is still found.
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name) const;

private:
    mutable std::mutex mutex_;
    std::vector<OllamaStore> stores_;
    std::vector<ModelRecord> records_;
};

// `sonder-infer models import-ollama` (docs/integration/model-store.md). Returns the
// process exit code: 0 ok, 1 failure (store unreadable, name not found, verification
// failed), 2 usage error.
int models_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace sonder::inference::models
