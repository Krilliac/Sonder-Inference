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
// on first use, then remembered by file identity; WeightCheck tells a hash made
// in this call from a remembered one). A loader that must load exactly the
// hashed bytes takes the open handles from resolve_for_load() (VerifiedWeights),
// which by default hashes through those very handles, and maps them; see
// docs/integration/model-store.md "Trust and verification".
//
// Cost model: a scan walks the manifest directories, reads and hashes the small
// files, and reads the first 24 bytes of each GGUF-class blob. Under the default
// policy it never hashes a model blob; resolve() does, once per unchanged file.
//
// Thread affinity: every free function and OllamaStore method is reentrant. An
// OllamaStore's shared state (the pinned root handle, used only as the start of
// relative opens, and the verified-blob cache, internally locked) is shared by
// copies of that store. ModelRegistry serializes its own state with a mutex;
// records its resolve() returns are independent copies (they share the read-only root);
// list() hands out the registry's own records as shared read-only pointers.
#pragma once

#include <chrono>
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
inline constexpr const char* kStoreJsonSchema = "sonder.inference.model_store/2";
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
    hash_limit,               // the weights were not hashed to the end: a blob is larger than
                              // StoreOptions::max_weight_file_bytes, the weights of the resolve add up
                              // to more than max_resolve_weight_bytes, or hashing ran past
                              // weight_hash_timeout or was cancelled (StoreOptions::weight_hash_cancel
                              // or ResolveOptions::cancel) (transient: the record is never reused, the
                              // next resolve tries again)
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
    // Windows: the file system does not report FileIdInfo, so `file_id` is the 64-bit index,
    // which can collide; such a file is hashed on every check, never looked up in nor added to
    // the remembered digests (first_use). Copied from the handle's identity; part of ==.
    bool weak_id = false;
    friend bool operator==(const FileStamp&, const FileStamp&) = default;
};

// What a record knows about one file it was built from (or tried to be).
enum class SourceKind {
    file,    // opened: `stamp` is its identity; unchanged while it still opens with that identity
    link,    // (POSIX) the manifest symlink itself, stamped with its own lstat
    absent,  // was not there (a missing manifest-list child); unchanged while still absent
    failed,  // could not be opened or read (refused, I/O error, busy, changed mid-read): the record
             // is never reused, it is always read again
};

// A file a record was built from, relative to the store root ('/'-separated).
struct SourceStamp {
    std::string rel;
    FileStamp stamp;  // set for `file` and `link` only
    SourceKind kind = SourceKind::file;
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
//   always     on every resolve() and every scan(); nothing is remembered.
//   first_use  on resolve(); the digest found (match or mismatch) is remembered per store by
//              the file's identity (volume + 128-bit file id, size, link count, write and
//              change time), so an unchanged file is hashed once per OllamaStore and a changed
//              one again. A later resolve() that finds the remembered digest does not read the
//              bytes: its record says WeightCheck::cached_identity, not hashed_now. On such a hit
//              the identity is read again from the same open handle after the lookup; any
//              difference (or a failed query) fails the record at once. Concurrent first uses
//              of one file share a single hash (the others wait for it). scan() does not hash.
//              Files with a weak (64-bit fallback) id (FileStamp::weak_id) are hashed on
//              every resolve(): their digest is never looked up nor remembered.
//   off        never; only the size and the GGUF header are checked. Not suitable for a
//              library that untrusted local writers can modify.
// Small layers are always hashed, whatever this says. Every weight hash, and every hash
// verify_blobs() makes, reads through a handle that (Windows) denies writers for as long as it
// is open, and at most kMaxConcurrentWeightHashes of them run at once per opened store root
// (copies of the store and its records share the limit).
enum class WeightVerification { always, first_use, off };
const char* to_string(WeightVerification policy) noexcept;

// How the weight bytes of a record (or of one VerifiedWeights file) were checked against their
// digests. Only hashed_now means "these bytes were hashed": read in this call, through a
// handle that denies writers on Windows, and found to match.
enum class WeightCheck {
    unchecked,        // not hashed: WeightVerification::off, a scan() under first_use, or not ready
    cached_identity,  // first_use: a digest remembered for an identical file identity was reused; the
                      // bytes were hashed at an earlier check, not in this call (on Windows a writer
                      // can rewrite a file and restore its times, see Known limits in the doc)
    hashed_now,       // hashed in this call through the open handle and matched
};
const char* to_string(WeightCheck check) noexcept;

inline constexpr std::size_t kMaxConcurrentWeightHashes = 2;

// Bounds on directory enumeration and on what a scan keeps: a store anyone can write to may
// hold any number of entries. Applied while entries are read, before anything is kept; a scan
// that reaches one says so in ScanResult::notes. A name lookup that misses (the
// case-insensitive fallback) reads at most max_scan_entries entries, all its nested
// directories together, and stops at the first match.
struct ScanLimits {
    std::size_t max_dir_entries = 4096;    // entries read from any one directory
    std::size_t max_scan_entries = 65536;  // entries read by one scan() or lookup, all directories
    std::size_t max_models = 1024;         // manifests one scan() reads
    // The small layers (config, params, template, system, messages) one scan() keeps, all
    // records together, counted by declared size and charged before a layer is read (each record
    // is also capped at 16 MiB of small layers). Once the next such layer would pass it, the scan
    // reads no more of them: the records concerned have text_omitted set (the omitted layers were
    // neither read nor checked: only opened, stamped and size-checked) and a note says so.
    // resolve() reads them in full. What is kept from a layer is also capped per field (a config
    // string 1 KiB, a config array 32 entries of 256 bytes, params extras 64 entries and 64 KiB,
    // runner/format/parent 256 bytes), and the registry counts all of it (retained_bytes) against
    // this limit. A ModelRegistry shares one allowance of this size across every store it scans
    // (refresh() carries what is left from store to store) and across the records it caches from
    // resolve(). Not counted, because each is bounded per record by a constant (about 100 KiB
    // together): names, paths, the blob list, source stamps, detail and warnings.
    std::uint64_t max_retained_text_bytes = 64ull << 20;
};

// Defaults for StoreOptions (see there).
inline constexpr std::uint64_t kDefaultMaxWeightFileBytes = 1ull << 40;     // 1 TiB per file
inline constexpr std::uint64_t kDefaultMaxResolveWeightBytes = 2ull << 40;  // 2 TiB per resolve
// 2 TiB at the portable SHA-256's slow end (about 90 minutes per TiB) is 3 hours.
inline constexpr std::chrono::milliseconds kDefaultWeightHashTimeout = std::chrono::hours(3);

struct StoreOptions {
    WeightVerification verify_weights = WeightVerification::first_use;
    // Called before a weight blob is hashed (a multi-GB file takes minutes), e.g. for a
    // progress line, on the thread that wants to hash it. It runs while that thread holds
    // nothing of the store: no hash slot and no single-flight claim, so no other thread ever
    // waits for it. When several threads need the same unchanged file under first_use, each
    // may call it, but only one of them then hashes the file; the others wait for that hash
    // (never for a callback) and reuse its digest. It may wait for other threads that use
    // any store or registry. On the callback's own thread, though, every OllamaStore /
    // ModelRegistry resolve(), resolve_for_load() and ModelRegistry::resolve fails at once
    // with ErrorCode::invalid_state, revalidate() returns false, and scan() hashes nothing
    // (a direct call would hash, and so call back, again). It may run on several threads at
    // once. May be empty. An exception it throws propagates out of the resolve that called it.
    std::function<void(const BlobRef&)> on_weight_hash;
    // Bounds on weight hashing, so one hostile or huge manifest cannot hold a resolve and the
    // store's hash slots for days (kMaxConcurrentWeightHashes of them are shared by every
    // resolve of the store). Whichever is hit, the record fails with UnsupportedReason::hash_limit,
    // as a transient failure (never reused, never remembered as a digest), and any thread waiting
    // for that hash is released to try again.
    //   max_weight_file_bytes     a file larger than this is not hashed (default 1 TiB).
    //   max_resolve_weight_bytes  the declared sizes of the distinct weight blobs of one resolve
    //                             (model, shard, projector, adapter, draft layers) may add up to
    //                             this much (default 2 TiB); more fails before anything is
    //                             opened, hashed or called back (a resolve that would hash).
    //   weight_hash_timeout       the time all hashing of ONE resolve (and one verify_blobs()
    //                             call) may take, counted from when it starts to hash and
    //                             including the wait for a hash slot or for another thread's
    //                             hash (default 3 hours, kDefaultWeightHashTimeout; zero: no time
    //                             limit, not recommended for a service).
    //   weight_hash_cancel        STORE-WIDE and never resettable (the options are copied at
    //                             open()): once cancelled, every later hash on this store fails.
    //                             It is the shutdown switch. To cancel one resolve, pass
    //                             ResolveOptions::cancel to that call.
    std::uint64_t max_weight_file_bytes = kDefaultMaxWeightFileBytes;
    std::uint64_t max_resolve_weight_bytes = kDefaultMaxResolveWeightBytes;
    std::chrono::milliseconds weight_hash_timeout = kDefaultWeightHashTimeout;
    CancellationToken weight_hash_cancel;
    ScanLimits limits;
};

// Per-call options for resolve() and resolve_for_load().
struct ResolveOptions {
    // Cancels this call only: its hashing stops (also while it waits for a hash slot or for
    // another thread's hash of the same file) and the record fails with
    // UnsupportedReason::hash_limit (transient). Nothing else on the store is affected; a later
    // call without a cancelled token hashes normally. Default: never cancelled.
    CancellationToken cancel;
};

struct StoreAnchor;    // the store root opened by OllamaStore::open() (model_store.cpp)
struct RetainBudget;   // what a scan may still keep of the small layers (model_store.cpp)

// Everything known about one store entry. A `ready` record has every blob
// present as a regular file inside the store, sized as declared; every small
// layer matched its digest; every weight starts with a supported GGUF header
// and matched its digest as `weight_check` says. An `unsupported` record
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
    // A scan reached ScanLimits::max_retained_text_bytes before one of this record's small
    // layers (config, params, template, system, messages), or a registry stripped what it was
    // keeping of the record to stay within it: the omitted layers were neither read nor checked
    // (their blobs are only opened, stamped and size-checked; one that cannot be opened makes
    // the record unsupported), so config, params and text fields, and anything classified from
    // them (cloud stub, safetensors config), are provisional or empty. resolve() of the name
    // reads everything; ModelRegistry never serves such a cached record without resolving it
    // again, and OllamaStore::revalidate() returns false for it.
    bool text_omitted = false;
    std::size_t license_count = 0;

    std::uint64_t weights_bytes = 0;     // declared size of model + shard + projector + adapter + draft layers
    std::uint64_t referenced_bytes = 0;  // declared size of every layer plus the config
    std::vector<BlobRef> blobs;          // every referenced blob (config first), for verification
    // How the weights were checked when this record was produced (or revalidated): the weakest
    // of its weight files. It says nothing about a file once its handle was closed; only
    // VerifiedWeights carries the handles.
    WeightCheck weight_check = WeightCheck::unchecked;
    // True only when every weight was hashed in that call and matched (WeightCheck::hashed_now).
    [[nodiscard]] bool weights_verified() const noexcept { return weight_check == WeightCheck::hashed_now; }
    // The manifest (its symlink, manifest-list children, absent ones too) this record was read
    // from, plus any file that failed transiently; blobs carry their own stamps.
    // OllamaStore::revalidate() and the registry's reuse of unsupported records compare them.
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
                                       // links, special files, ScanLimits reached), control-escaped;
                                       // at most 256
};

// The weight files of one resolved model, held open for a loader. Move-only; owns the OS
// handles (HANDLE on Windows, fd on POSIX) and closes them when destroyed.
//
// The loader must map or read exactly these handles, never reopen model_path & co. While
// this object lives, on Windows every handle denies writers (no other process can write,
// truncate, rename or delete the file), so the bytes mapped through it are the bytes that
// were hashed when `verified()` is true (every file WeightCheck::hashed_now: hashed through
// this very handle in the call that returned it). The remaining post-check windows (see
// docs/integration/model-store.md "Trust and verification"):
//   - POSIX has no mandatory deny-write: a process that may write the blob can still change
//     its bytes after the hash, including while pages are being mapped in;
//   - Windows: a writable section mapped before this open (its writer's handle already
//     closed) and raw volume access by an administrator are not blocked;
//   - WeightCheck::cached_identity (only when the caller asked resolve_for_load for
//     first_use): the hash ran at an earlier check; a write between that check and this open
//     that kept the size and restored the timestamps (possible on Windows) is not seen.
// File positions are unspecified (the hash read to the end): use mapping or positioned
// reads. Not thread-safe for concurrent mutation; const access from any thread.
class VerifiedWeights {
public:
    struct File {
        BlobRef blob;                                // the first layer that referenced this digest
        std::intptr_t native_handle = 0;             // HANDLE / fd owned by this object; do not close it
        WeightCheck check = WeightCheck::unchecked;  // how this file's bytes were checked
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
    // The weakest check among the files (unchecked when empty).
    [[nodiscard]] WeightCheck check() const noexcept { return check_; }
    // True only when every file was hashed through its handle in the call that returned this
    // object and matched its digest (check() == WeightCheck::hashed_now). It is a statement
    // about that past hash ("hashed through this handle in this call"), not a guarantee that
    // the bytes cannot change afterwards: POSIX cannot deny writers, and on Windows a writable
    // mapping made before the open is not revoked (see the windows listed above).
    [[nodiscard]] bool verified() const noexcept { return check_ == WeightCheck::hashed_now; }

private:
    friend struct VerifiedWeightsAccess;  // model_store.cpp
    struct Handles;
    std::vector<File> files_;
    std::unique_ptr<Handles> handles_;
    WeightCheck check_ = WeightCheck::unchecked;
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
    // Fails with not_found when `models_dir` does not exist or is not a directory, io_error
    // when it cannot be opened (access denied, an invalid name, a device error; the message
    // carries the OS error), and invalid_argument when it holds neither manifests-v2/ nor
    // manifests/. The messages do not repeat the path (the caller knows it).
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
    // Weights are checked per options().verify_weights (record.weight_check says how); their
    // handles are then closed. The hash limits of StoreOptions apply; `call` can cancel just
    // this resolve.
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name, const ResolveOptions& call = {}) const;

    // resolve(), keeping the weight handles (opened with writers denied on Windows) for a
    // loader. `policy` defaults to `always`: every weight is hashed through the very handle
    // returned, whatever options().verify_weights says, so weights.verified() means these
    // bytes. Pass first_use explicitly to accept a remembered digest (cached_identity) and skip
    // the hash, or off to skip hashing (unchecked). The loader hook: map `weights`, never the
    // record's paths.
    [[nodiscard]] Result<LoadableModel> resolve_for_load(std::string_view name,
                                                         WeightVerification policy = WeightVerification::always,
                                                         const ResolveOptions& call = {}) const;

    // Re-checks a record this store produced before it is used again: every file it was
    // read from (sources and blob stamps) must be unchanged, then the weights are checked
    // per options().verify_weights. Returns false when something changed or disappeared
    // (resolve again), when the record came from another OllamaStore (even one opened on the
    // same directory: records are tied to the root handle that read them), when the record
    // is text_omitted (some of its layers were never read, so it cannot be current), or when
    // called from on_weight_hash; on true `record` is current, though it may now be
    // unsupported (digest_mismatch, blob_busy, hash_limit).
    [[nodiscard]] bool revalidate(ModelRecord& record) const;

    // Diagnostics: weight hashes (and verify_blobs() hashes of this store's records) running
    // right now on this store's root; never more than kMaxConcurrentWeightHashes.
    [[nodiscard]] std::size_t weight_hashes_running() const noexcept;

    struct DigestCache;  // defined in model_store.cpp

private:
    friend class ModelRegistry;
    OllamaStore(std::filesystem::path root, std::shared_ptr<const StoreAnchor> anchor, StoreOptions options);
    // scan() with an allowance shared with other scans (ModelRegistry): the small layers it keeps
    // are charged to `budget`, which it leaves with what is still unspent.
    [[nodiscard]] ScanResult scan_impl(RetainBudget& budget) const;
    [[nodiscard]] Result<ModelRecord> resolve_impl(std::string_view name, WeightVerification policy,
                                                   VerifiedWeights* keep, const CancellationToken& cancel) const;
    [[nodiscard]] bool revalidate_impl(ModelRecord& record, WeightVerification policy, VerifiedWeights* keep,
                                       const CancellationToken& cancel) const;
    // For a cached unsupported record: true while every file it was read from is unchanged,
    // every blob it could not open is still absent and nothing failed transiently (so its
    // reason still holds).
    [[nodiscard]] bool unsupported_still_current(const ModelRecord& record) const;
    std::filesystem::path root_;
    std::shared_ptr<const StoreAnchor> anchor_;
    StoreOptions options_;
    std::shared_ptr<DigestCache> cache_;
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
// 1 MiB buffer per call. Each handle denies writers on Windows (a blob another process
// has open for writing is reported io_error, "open for writing"), and each hash takes one
// of the root's kMaxConcurrentWeightHashes slots (after on_blob_start returns). Stops after
// the first cancelled blob. Does not modify the record or the store. The limits of the
// StoreOptions the record's store was opened with apply: a blob larger than
// max_weight_file_bytes is not hashed (io_error), the whole call may hash for at most
// weight_hash_timeout (io_error for the blob it stops in and for every later one; the wait
// for a hash slot counts), and weight_hash_cancel cancels like VerifyOptions::cancel. A record
// without an anchor uses the defaults.
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
// checked per the StoreOptions given at construction (resolve_for_load: `always` by default):
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

    // Adds a store and scans it. A directory already registered (same root directory
    // identity, whatever path names it) is ignored. Earlier stores win when two stores hold
    // the same name.
    Status add_ollama_dir(std::filesystem::path models_dir);

    // Rescans every store (directory walk plus small reads; no hashing). A store added by
    // another thread while it runs keeps the records its own scan produced.
    void refresh();

    // The scanned records, shared with the registry (read-only, not copied; a refresh() or a
    // resolve() that replaces an entry swaps the registry's pointer, never the record).
    [[nodiscard]] std::vector<std::shared_ptr<const ModelRecord>> list() const;

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
    [[nodiscard]] Result<ModelRecord> resolve(std::string_view name, const ResolveOptions& call = {}) const;

    // resolve(), plus the open weight handles (OllamaStore::resolve_for_load; `policy`
    // defaults to `always`, whatever the registry's StoreOptions say). Ready only.
    [[nodiscard]] Result<LoadableModel> resolve_for_load(std::string_view name,
                                                         WeightVerification policy = WeightVerification::always,
                                                         const ResolveOptions& call = {}) const;

private:
    [[nodiscard]] Result<ModelRecord> resolve_impl(std::string_view name, WeightVerification policy,
                                                   VerifiedWeights* keep, const CancellationToken& cancel) const;
    StoreOptions options_;
    mutable std::mutex mutex_;
    // Bytes of retained small-layer content (config strings and arrays, params, template, system,
    // messages, runner/format/parent) held by records_ and live_, kept within
    // options_.limits.max_retained_text_bytes. A running total, updated wherever a record enters,
    // leaves or is replaced in records_ or live_. Guarded by mutex_.
    mutable std::uint64_t retained_bytes_ = 0;
    std::vector<OllamaStore> stores_;
    mutable std::vector<std::shared_ptr<const ModelRecord>> records_;
    mutable std::map<std::string, std::shared_ptr<const ModelRecord>> live_;
};

// `sonder-infer models import-ollama` (docs/integration/model-store.md). Returns the
// process exit code: 0 ok, 1 failure (store unreadable, name not found, verification
// failed), 2 usage error.
int models_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err);

}  // namespace sonder::inference::models
