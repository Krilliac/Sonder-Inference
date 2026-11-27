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
// OllamaStore::open() opens the store root once and keeps that directory handle
// (a link at the root itself is followed there, once); every later open and
// directory listing goes relative to it, one segment at a time, without
// following symlinks, junctions or other reparse points. Each file is checked on
// its open handle to be a regular disk file and read from that same handle with
// a bounded read (src/common/safe_file.hpp). Manifests and the small layers
// (config, template, system, params, messages) must not be hard-linked; their
// declared size and SHA-256 are checked on every read. Weight blobs are size-
// and header-checked on every read and fully hashed per StoreOptions (default:
// on first use, cached by file identity). A loader that must load exactly the
// verified bytes takes the open handles from resolve_for_load() (VerifiedWeights)
// and maps those; see docs/integration/model-store.md "Trust and verification".
//
// Cost model: a scan walks the manifest directories, reads and hashes the small
// files, and reads the first 24 bytes of each GGUF-class blob. Under the default
// policy it never hashes a model blob; resolve() does, once per unchanged file.
//
// Thread affinity: every free function and OllamaStore method is reentrant. An
// OllamaStore's shared state (the pinned root handle, used only as the start of
// relative opens, and the verified-blob cache, internally locked) is shared by
// copies of that store. ModelRegistry serializes its own state with a mutex;
// records it returns are independent copies (they share the read-only root).
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
// "https://" + host (350) + "/" + namespace (80) + "/" + model (80) + ":" + tag (80).
inline constexpr std::size_t kMaxOllamaNameBytes = 8 + 350 + 1 + 80 + 1 + 80 + 1 + 80;

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
// COM0-9, LPT0-9, with or without an extension; the host part too), which Win32
// would map to a different file. A name longer than the longest valid one
// (kMaxOllamaNameBytes) is rejected before it is split. The error message quotes
// at most 128 bytes of `text`, with every control, invisible-format (bidi,
// zero-width, line separator) and invalid UTF-8 byte escaped as \xNN.
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
    blob_busy,                // Windows: a weight blob is open for writing by another process, so it
                              // cannot be opened with writes denied to hash or serve it (transient)
};
const char* to_string(UnsupportedReason reason) noexcept;
const char* to_string(EntryStatus status) noexcept;

enum class BlobRole { config, model, projector, adapter, draft, tmpl, system, params, messages, license, other };
const char* to_string(BlobRole role) noexcept;

// Identity of a file as read through its open handle: volume + file id (128-bit
// FILE_ID_INFO on Windows, st_ino on POSIX), size, link count, last-write and
// change time (platform ticks; see safe_file.hpp). Two stamps are equal only when
// none of these changed. On Windows the two times can be set back by anyone who
// may write the file's attributes, so equal stamps are not proof of equal bytes.
struct FileStamp {
    std::uint64_t volume = 0;
    std::uint64_t file_id = 0;
    std::uint64_t file_id_high = 0;  // upper 64 bits of the Windows 128-bit file id; 0 on POSIX
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
//              identity (volume + 128-bit file id, size, link count, write and change time),
//              so an unchanged file is hashed once per OllamaStore and a changed one again.
//              On a cache hit the identity is read again from the same open handle after the
//              lookup; any difference is a miss. Concurrent first uses of one file share a
//              single hash (the others wait for it). scan() does not hash. Between checks the
//              cache trusts metadata: on Windows a writer who restores the size and both
//              times is not seen until the next real hash (use `always` there).
//   off        never; only the size and the GGUF header are checked. Not suitable for a
//              library that untrusted local writers can modify.
// Small layers are always hashed, whatever this says. Every weight hash reads through a
// handle that (Windows) denies writers for as long as it is open, and at most
// kMaxConcurrentWeightHashes run at once per store (copies share the limit).
enum class WeightVerification { always, first_use, off };
const char* to_string(WeightVerification policy) noexcept;

inline constexpr std::size_t kMaxConcurrentWeightHashes = 2;

struct StoreOptions {
    WeightVerification verify_weights = WeightVerification::first_use;
    // Called just before a weight blob is hashed (a multi-GB file takes minutes), e.g. for a
    // progress line, on the thread that does the hash: when several threads need the same
    // file only one hashes it. It may run on two resolving threads at once. May be empty.
    std::function<void(const BlobRef&)> on_weight_hash;
};

struct StoreAnchor;  // the store root opened by OllamaStore::open() (model_store.cpp)

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
    // They name the files; they do not pin them: a loader that opens a path again gets
    // whatever is there by then (OllamaStore::resolve_for_load returns the checked handles).
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
    // "Matched at check time": every weight blob, opened when this record was produced (or
    // revalidated), hashed to its digest through that handle, or the verified cache held a
    // match for the identity read from that handle. It says nothing about the file after the
    // handle was closed; only VerifiedWeights carries the handles. False under
    // WeightVerification::off and for records from scan() under first_use.
    bool weights_verified = false;
    // The manifest (and its symlink, a manifest-list child) this record was read from;
    // blobs carry their own stamps. OllamaStore::revalidate() compares them.
    std::vector<SourceStamp> sources;
    // The store root this record was read through (shared, read-only); verify_blobs() and
    // revalidate() open files relative to it, never through store_root's path again.
    std::shared_ptr<const StoreAnchor> anchor;

    std::vector<std::string> warnings;

    [[nodiscard]] bool ready() const noexcept { return status == EntryStatus::ready; }
};

struct ScanResult {
    std::vector<ModelRecord> records;  // sorted by name key; v2 entries win over legacy duplicates
    std::vector<std::string> notes;    // store-relative paths skipped during the walk (invalid names,
                                       // links, special files), control-escaped; at most 256
};

// The weight files of one resolved model, held open for a loader. Move-only; owns the OS
// handles (HANDLE on Windows, fd on POSIX) and closes them when destroyed.
//
// The loader must map or read exactly these handles, never reopen model_path & co. While
// this object lives, on Windows every handle denies writers (no other process can write,
// truncate, rename or delete the file), so the bytes mapped through it are the bytes that
// were hashed when `verified()` is true. The remaining post-check windows (see
// docs/integration/model-store.md "Trust and verification"):
//   - POSIX has no mandatory deny-write: a process that may write the blob can still change
//     its bytes after the hash, including while pages are being mapped in;
//   - Windows: a writable section mapped before this open (its writer's handle already
//     closed) and raw volume access by an administrator are not blocked;
//   - first_use cache hits: the hash ran at an earlier check; a write between that check
//     and this open that kept the size and restored the timestamps (possible on Windows)
//     is not seen; `always` re-hashes through these very handles.
// File positions are unspecified (the hash read to the end): use mapping or positioned
// reads. Not thread-safe for concurrent mutation; const access from any thread.
class VerifiedWeights {
public:
    struct File {
        BlobRef blob;                    // the first layer that referenced this digest
        std::intptr_t native_handle = 0; // HANDLE / fd owned by this object; do not close it
    };

    VerifiedWeights() noexcept;
    ~VerifiedWeights();
    VerifiedWeights(VerifiedWeights&& other) noexcept;
    VerifiedWeights& operator=(VerifiedWeights&& other) noexcept;
    VerifiedWeights(const VerifiedWeights&) = delete;
    VerifiedWeights& operator=(const VerifiedWeights&) = delete;

    // One entry per distinct weight digest, in manifest order.
    [[nodiscard]] const std::vector<File>& files() const noexcept { return files_; }
    [[nodiscard]] const File* find(std::string_view digest) const noexcept;
    [[nodiscard]] bool empty() const noexcept { return files_.empty(); }
    // True when every file hashed to its digest through its handle (or matched the verified
    // cache for the identity read from that handle); false under WeightVerification::off.
    [[nodiscard]] bool verified() const noexcept { return verified_; }

private:
    friend struct VerifiedWeightsAccess;  // model_store.cpp
    struct Handles;
    std::vector<File> files_;
    std::unique_ptr<Handles> handles_;
    bool verified_ = false;
};

// A record plus the open weight handles it was checked through (resolve_for_load).
struct LoadableModel {
    ModelRecord record;
    VerifiedWeights weights;  // empty unless record.ready()
};

class ModelRegistry;

// One Ollama models directory. Cheap to copy: the root path, the options and shared
// pointers to the pinned root and the verified-blob cache (copies share both).
class OllamaStore {
public:
    // Opens the root directory once and keeps it open; every later open is relative to
    // that handle, so retargeting a link at `models_dir` afterwards does not move the store.
    // Fails with not_found when `models_dir` is not a directory and with
    // invalid_argument when it holds neither manifests-v2/ nor manifests/. The
    // messages do not repeat the path (the caller knows it).
    static Result<OllamaStore> open(std::filesystem::path models_dir, StoreOptions options = {});

    // The absolute path given to open(): for display and record paths only.
    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }
    [[nodiscard]] const StoreOptions& options() const noexcept { return options_; }

    // Loads every manifest. A broken entry becomes an `unsupported` record; the scan
    // itself does not fail.
    [[nodiscard]] ScanResult scan() const;

    // Resolves "name[:tag]" the way Ollama's resolveManifestPath does: manifests-v2 first,
    // then the legacy tree (also under the alternate public host). Returns not_found when
    // no manifest exists; an unsupported entry is returned as a record, not an error.
    // Weights are verified per options().verify_weights; their handles are then closed.
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name) const;

    // resolve(), keeping the weight handles the verification read through (opened with
    // writers denied on Windows, also under WeightVerification::off, where they are not
    // hashed). The loader hook: map `weights`, not the record's paths.
    [[nodiscard]] Result<LoadableModel> resolve_for_load(std::string_view name) const;

    // Re-checks a record this store produced before it is used again: every file it was
    // read from (sources and blob stamps) must be unchanged, then the weights are verified
    // per options().verify_weights. Returns false when something changed or disappeared
    // (resolve again), or when the record came from another store; on true `record` is
    // current, though it may now be unsupported (digest_mismatch, blob_busy).
    [[nodiscard]] bool revalidate(ModelRecord& record) const;

    struct VerifiedCache;  // defined in model_store.cpp

private:
    friend class ModelRegistry;
    OllamaStore(std::filesystem::path root, std::shared_ptr<const StoreAnchor> anchor, StoreOptions options);
    [[nodiscard]] Result<ModelRecord> resolve_impl(std::string_view name, VerifiedWeights* keep) const;
    [[nodiscard]] bool revalidate_impl(ModelRecord& record, VerifiedWeights* keep) const;
    // For a cached unsupported record: true while every file it was read from is unchanged
    // and every blob it could not open is still absent (so its reason still holds).
    [[nodiscard]] bool unsupported_still_current(const ModelRecord& record) const;
    std::filesystem::path root_;
    std::shared_ptr<const StoreAnchor> anchor_;
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
// opened relative to record.anchor (the store root pinned by OllamaStore::open; a
// record without one, built by hand, opens record.store_root once for the call), one
// 1 MiB buffer per call. Stops after the first cancelled blob. Does not modify the
// record or the store.
VerifyReport verify_blobs(const ModelRecord& record, const VerifyOptions& options = {});

// Stable JSON for a record / a scan (schema kStoreJsonSchema).
json::Value to_json(const ModelRecord& record);
json::Value to_json(const ScanResult& scan, const std::filesystem::path& store_root);

// The directory Ollama itself would use: $OLLAMA_MODELS, else <home>/.ollama/models.
// Empty when neither can be determined. Reading the variable never touches the store.
std::filesystem::path default_ollama_models_dir();

// Name -> record lookup over one or more Ollama stores. This is the hook the native
// llamacpp backend will use to load a model by name (docs/integration/model-store.md).
// A cached ready record is never returned without OllamaStore::revalidate(), a cached
// unsupported one only while its reason still holds (see resolve()), and weights are
// verified per the StoreOptions given at construction:
//
//   ModelRegistry reg; reg.add_ollama_dir(dir);
//   auto m = reg.resolve_for_load("qwen3.5:4b");   // ready records only
//   backend.load(m.value().weights /* map these handles */, m.value().record.params ...)
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
    // 1024, dropped by refresh()). A cached ready record is revalidated before reuse
    // (every source and blob stamp, then the weight policy). A cached unsupported record
    // (from the scan, or a scan entry that a later resolve found broken) is returned again
    // only while every file it was read from has the same stamp and every blob it could not
    // open is still absent; otherwise the name is resolved again, so a blob that was
    // repaired, finished downloading or stopped being written is served without refresh().
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name) const;

    // resolve(), plus the open weight handles (OllamaStore::resolve_for_load). Ready only.
    [[nodiscard]] Result<LoadableModel> resolve_for_load(std::string_view name) const;

private:
    [[nodiscard]] Result<ModelRecord> resolve_impl(std::string_view name, VerifiedWeights* keep) const;
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
