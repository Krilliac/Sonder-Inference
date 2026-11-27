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
// Every file is opened once, without following symlinks, junctions or other
// reparse points, checked on the open handle to be a regular disk file inside
// the store root, and read from that same handle with a bounded read
// (src/common/safe_file.hpp). Manifests and the small layers (config,
// template, system, params, messages) must not be hard-linked; their declared
// size and SHA-256 are checked on every read. Weight blobs are size- and
// header-checked on every read and fully hashed per StoreOptions (default: on
// first use, cached by file identity). See docs/integration/model-store.md
// "Trust and verification" for the remaining limits (the loader opens weights
// by path).
//
// Cost model: a scan walks the manifest directories, reads and hashes the small
// files, and reads the first 24 bytes of each GGUF-class blob. Under the default
// policy it never hashes a model blob; resolve() does, once per unchanged file.
//
// Thread affinity: every free function and OllamaStore method is reentrant; the
// only shared state is an OllamaStore's verified-blob cache, which is internally
// locked and shared by copies of that store. ModelRegistry serializes its own
// state with a mutex; records it returns are independent copies.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <iosfwd>
#include <map>
#include <memory>
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
// rules (types/model/name.go isValidPart), and additionally rejects a part that
// ends in '.' or whose base name is a Windows device name (CON, PRN, AUX, NUL,
// COM0-9, LPT0-9, with or without an extension), which Win32 would map to a
// different file. The error message quotes at most 128 bytes of `text`, with
// control bytes escaped.
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
    unsafe_path,              // a manifest, directory or blob is a link, junction/reparse point, a
                              // hard link where none is allowed, or not a regular disk file
    digest_mismatch,          // a blob's content does not hash to its digest
};
const char* to_string(UnsupportedReason reason) noexcept;
const char* to_string(EntryStatus status) noexcept;

enum class BlobRole { config, model, projector, adapter, draft, tmpl, system, params, messages, license, other };
const char* to_string(BlobRole role) noexcept;

// Identity of a file as read through its open handle: volume + file id, size,
// link count, last-write and change time (platform ticks; see safe_file.hpp).
// Two stamps are equal only when none of these changed.
struct FileStamp {
    std::uint64_t volume = 0;
    std::uint64_t file_id = 0;
    std::uint64_t size = 0;
    std::uint64_t link_count = 0;
    std::int64_t write_time = 0;
    std::int64_t change_time = 0;
    friend bool operator==(const FileStamp&, const FileStamp&) = default;
};

// A file a record was built from, relative to the store root ('/'-separated).
// `link` marks the manifest symlink itself (POSIX), stamped with its own lstat.
struct SourceStamp {
    std::string rel;
    FileStamp stamp;
    bool link = false;
};

struct BlobRef {
    BlobRole role = BlobRole::other;
    std::string media_type;
    std::string digest;  // "sha256:<64 lowercase hex>"
    std::filesystem::path path;
    std::uint64_t declared_size = 0;           // from the manifest
    std::optional<std::uint64_t> disk_size;    // nullopt when the blob is absent or was not opened
    std::string from;                          // manifest "from" (parent model), may be empty
    std::optional<FileStamp> stamp;            // identity when the blob was opened
};

struct GgufHeader {
    std::uint32_t version = 0;
    std::uint64_t tensor_count = 0;
    std::uint64_t kv_count = 0;
};

// Reads and checks the first 24 bytes (magic, version 2 or 3, counts). Version 1
// is rejected the way llama.cpp rejects it. Reads nothing else. A symlink or
// reparse point at `file` is refused, not followed.
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

// When a weight blob (model, projector, adapter, draft) is fully hashed against its digest.
//   always     on every resolve() and every scan(); nothing is cached.
//   first_use  on resolve() (the loader path); a match or mismatch is cached by the file's
//              identity (volume + file id, size, write and change time), so an unchanged file
//              is hashed once per OllamaStore and a changed one again. scan() does not hash.
//   off        never; only the size and the GGUF header are checked.
// Small layers are always hashed, whatever this says.
enum class WeightVerification { always, first_use, off };
const char* to_string(WeightVerification policy) noexcept;

struct StoreOptions {
    WeightVerification verify_weights = WeightVerification::first_use;
    // Called on the resolving thread just before a weight blob is hashed (a multi-GB file
    // takes minutes), e.g. for a progress line. May be empty.
    std::function<void(const BlobRef&)> on_weight_hash;
};

// Everything known about one store entry. A `ready` record has every blob
// present as a regular file inside the store, sized as declared; every small
// layer matched its digest; every weight starts with a supported GGUF header
// and, when `weights_verified`, matched its digest. An `unsupported` record
// explains why and still carries whatever was parsed. `detail` and `warnings`
// quote untrusted manifest text bounded and control-escaped, and name no
// absolute path.
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
    std::vector<std::filesystem::path> draft_paths;  // every image.draft layer in manifest order: the first is
                                                     // the draft model, later ones its shards (images.go
                                                     // DraftPath / DraftShardPaths)
    std::optional<GgufHeader> gguf;  // header of model_path

    ModelParams params;
    std::optional<std::string> template_text;  // image.template (or legacy image.prompt)
    std::optional<std::string> system_prompt;  // image.system
    std::optional<std::string> messages_json;  // image.messages, validated JSON text
    std::size_t license_count = 0;

    std::uint64_t weights_bytes = 0;     // declared size of model + shard + projector + adapter + draft layers
    std::uint64_t referenced_bytes = 0;  // declared size of every layer plus the config
    std::vector<BlobRef> blobs;          // every referenced blob (config first), for verification
    // True when every weight blob was hashed (or found in the verified cache) and matched
    // its digest while this record was produced; false under WeightVerification::off and
    // for records from scan() under first_use.
    bool weights_verified = false;
    // The manifest (and its symlink, a manifest-list child) this record was read from;
    // blobs carry their own stamps. OllamaStore::revalidate() compares them.
    std::vector<SourceStamp> sources;

    std::vector<std::string> warnings;

    [[nodiscard]] bool ready() const noexcept { return status == EntryStatus::ready; }
};

struct ScanResult {
    std::vector<ModelRecord> records;  // sorted by name key; v2 entries win over legacy duplicates
    std::vector<std::string> notes;    // store-relative paths skipped during the walk (invalid names,
                                       // links, special files), control-escaped; at most 256
};

// One Ollama models directory. Cheap to copy: the root path, the options and a
// shared pointer to the verified-blob cache (copies share it).
class OllamaStore {
public:
    // Fails with not_found when `models_dir` is not a directory and with
    // invalid_argument when it holds neither manifests-v2/ nor manifests/. The
    // messages do not repeat the path (the caller knows it).
    static Result<OllamaStore> open(std::filesystem::path models_dir, StoreOptions options = {});

    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }
    [[nodiscard]] const StoreOptions& options() const noexcept { return options_; }

    // Loads every manifest. A broken entry becomes an `unsupported` record; the scan
    // itself does not fail.
    [[nodiscard]] ScanResult scan() const;

    // Resolves "name[:tag]" the way Ollama's resolveManifestPath does: manifests-v2 first,
    // then the legacy tree (also under the alternate public host). Returns not_found when
    // no manifest exists; an unsupported entry is returned as a record, not an error.
    // Weights are verified per options().verify_weights.
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name) const;

    // Re-checks a record this store produced before it is used again: every file it was
    // read from (sources and blob stamps) must be unchanged, then the weights are verified
    // per options().verify_weights. Returns false when something changed or disappeared
    // (resolve again); on true `record` is current, though it may now be unsupported
    // (digest_mismatch).
    [[nodiscard]] bool revalidate(ModelRecord& record) const;

    struct VerifiedCache;  // defined in model_store.cpp

private:
    OllamaStore(std::filesystem::path root, StoreOptions options);
    std::filesystem::path root_;
    StoreOptions options_;
    std::shared_ptr<VerifiedCache> cache_;
};

// Opt-in content verification: re-hashes blobs and compares with their digest.
// `unsafe`: the blob is a link, a reparse point or not a regular file (not read).
enum class BlobVerdict { ok, mismatch, missing, io_error, cancelled, unsafe };
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

// Streams each distinct blob (by digest) through SHA-256 from a no-follow handle
// anchored at record.store_root (one 1 MiB buffer per call). Stops after the first
// cancelled blob. Does not modify the record or the store.
VerifyReport verify_blobs(const ModelRecord& record, const VerifyOptions& options = {});

// Stable JSON for a record / a scan (schema kStoreJsonSchema).
json::Value to_json(const ModelRecord& record);
json::Value to_json(const ScanResult& scan, const std::filesystem::path& store_root);

// The directory Ollama itself would use: $OLLAMA_MODELS, else <home>/.ollama/models.
// Empty when neither can be determined. Reading the variable never touches the store.
std::filesystem::path default_ollama_models_dir();

// Name -> record lookup over one or more Ollama stores. This is the hook the native
// llamacpp backend will use to load a model by name (docs/integration/model-store.md).
// resolve() never returns a cached record without OllamaStore::revalidate(), and
// verifies weights per the StoreOptions given at construction:
//
//   ModelRegistry reg; reg.add_ollama_dir(dir);
//   auto rec = reg.resolve("qwen3.5:4b");   // ready records only
//   backend.load(rec.value().model_path, rec.value().projector_paths, rec.value().params ...)
//
// The backend is not wired to it yet. Thread-safe; not copyable.
class ModelRegistry {
public:
    ModelRegistry() = default;
    explicit ModelRegistry(StoreOptions options) : options_(std::move(options)) {}
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
    // after the last refresh() is still found; a ready live result is cached (at most
    // 1024, dropped by refresh()) and, like every cached record, revalidated before reuse.
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name) const;

private:
    StoreOptions options_;
    mutable std::mutex mutex_;
    std::vector<OllamaStore> stores_;
    mutable std::vector<ModelRecord> records_;
    mutable std::map<std::string, ModelRecord> live_;
};

// `sonder-infer models import-ollama` (docs/integration/model-store.md). Returns the
// process exit code: 0 ok, 1 failure (store unreadable, name not found, verification
// failed), 2 usage error.
int models_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace sonder::inference::models
