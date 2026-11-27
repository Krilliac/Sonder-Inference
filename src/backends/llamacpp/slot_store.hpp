// Sonder Inference - versioned sequence KV slot persistence for llama.cpp.
//
// Saves and restores a single sequence's KV/recurrent state under a fixed-size,
// little-endian versioned header binding:
//   - file magic ("SNDRSLOT") and format version (1),
//   - model identity: exact GGUF file size + SHA-256 of the GGUF header/metadata
//     region [0, gguf_get_data_offset(ctx)) rather than the full weight blob
//     (ggml/include/gguf.h:88, 90, 96),
//   - K and V cache element types (KvCacheType),
//   - context length (n_ctx) and sequence token count,
//   - validated payload byte lengths and a SHA-256 checksum over the payload.
//
// Sequence state capture and restore use llama.cpp v1.0.0's public sequence
// state C APIs (include/llama.h:70, 856-875):
//   - llama_state_seq_get_size(ctx, seq_id)        [include/llama.h:856]
//   - llama_state_seq_get_data(ctx, dst, sz, seq)  [include/llama.h:861]
//   - llama_state_seq_set_data(ctx, src, sz, seq)  [include/llama.h:871]
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

struct llama_context;

namespace sonder::backends::llamacpp {

inline constexpr std::array<std::uint8_t, 8> kSlotFileMagic{'S', 'N', 'D', 'R', 'S', 'L', 'O', 'T'};
inline constexpr std::uint32_t kSlotFormatVersion = 1u;
inline constexpr std::uint32_t kSlotFileHeaderBytes = 128u;
inline constexpr std::uint32_t kMaxSlotContextTokens = 16u * 1024u * 1024u;

inline constexpr std::size_t kDefaultMaxSlotPayloadBytes = kMaxSequenceStateBytes;  // 512 MiB
inline constexpr std::size_t kHardMaxSlotPayloadBytes = std::size_t{1} << 30;       // 1 GiB
inline constexpr std::size_t kDefaultMaxSlotFileBytes = kSlotFileHeaderBytes + kDefaultMaxSlotPayloadBytes;
inline constexpr std::size_t kHardMaxSlotFileBytes = kSlotFileHeaderBytes + kHardMaxSlotPayloadBytes;
inline constexpr std::size_t kDefaultMaxGgufHeaderBytes = std::size_t{128} << 20;   // 128 MiB

enum class SlotStoreError : std::uint8_t {
    kOk = 0,
    kInvalidArgument,
    kInvalidPath,
    kIoError,
    kInvalidGguf,
    kTruncatedFile,
    kFileTooLarge,
    kBadMagic,
    kUnsupportedVersion,
    kInvalidHeader,
    kModelMismatch,
    kKvTypeMismatch,
    kContextLengthMismatch,
    kTokenCountMismatch,
    kChecksumMismatch,
    kBackendError,
};

[[nodiscard]] std::string_view ToString(SlotStoreError code) noexcept;

struct SlotStoreStatus {
    SlotStoreError code = SlotStoreError::kOk;
    std::string message;

    [[nodiscard]] bool ok() const noexcept { return code == SlotStoreError::kOk; }
    static SlotStoreStatus Ok() { return {}; }
    static SlotStoreStatus Error(SlotStoreError c, std::string msg) { return {c, std::move(msg)}; }

    [[nodiscard]] Status ToBackendStatus() const;
};

// Identifies a GGUF model artifact without hashing multi-gigabyte weight tensors:
// `file_size` is the total GGUF byte size on disk, and `header_sha256` is the
// FIPS 180-4 SHA-256 digest over [0, gguf_get_data_offset(ctx)).
struct SlotModelIdentity {
    std::uint64_t file_size = 0;
    std::array<std::uint8_t, 32> header_sha256{};

    [[nodiscard]] bool operator==(const SlotModelIdentity& other) const noexcept = default;
};

// Fixed 128-byte little-endian on-disk header preceding the payload.
// Wire layout:
//   [  0..  8) magic              (8 bytes: "SNDRSLOT")
//   [  8.. 12) format_version     (uint32 LE, 1)
//   [ 12.. 16) header_bytes       (uint32 LE, 128)
//   [ 16.. 24) model_file_size    (uint64 LE, > 0)
//   [ 24.. 56) gguf_header_sha256 (32 bytes)
//   [ 56.. 60) type_k             (uint32 LE wire code)
//   [ 60.. 64) type_v             (uint32 LE wire code)
//   [ 64.. 68) n_ctx              (uint32 LE, 1..kMaxSlotContextTokens)
//   [ 68.. 72) reserved           (uint32 LE, must be 0)
//   [ 72.. 80) token_count        (uint64 LE, 0..n_ctx)
//   [ 80.. 88) token_table_bytes  (uint64 LE, 0 or token_count * sizeof(Token))
//   [ 88.. 96) state_bytes        (uint64 LE, > 0)
//   [ 96..128) payload_sha256     (32 bytes, SHA-256 over [token_table || state])
struct SlotFileHeader {
    std::array<std::uint8_t, 8> magic = kSlotFileMagic;
    std::uint32_t format_version = kSlotFormatVersion;
    std::uint32_t header_bytes = kSlotFileHeaderBytes;
    SlotModelIdentity model_identity{};
    KvCacheType type_k = KvCacheType::kF16;
    KvCacheType type_v = KvCacheType::kF16;
    std::uint32_t n_ctx = 0;
    std::uint32_t reserved = 0;
    std::uint64_t token_count = 0;
    std::uint64_t token_table_bytes = 0;
    std::uint64_t state_bytes = 0;
    std::array<std::uint8_t, 32> payload_sha256{};

    [[nodiscard]] bool operator==(const SlotFileHeader& other) const noexcept = default;
};

struct SlotStoreLimits {
    std::size_t max_payload_bytes = kDefaultMaxSlotPayloadBytes;
    std::size_t max_file_bytes = kDefaultMaxSlotFileBytes;
};

struct SlotSaveMetadata {
    SlotModelIdentity model_identity{};
    KvCacheType type_k = KvCacheType::kF16;
    KvCacheType type_v = KvCacheType::kF16;
    std::uint32_t n_ctx = 0;
    // Used directly when `tokens` is empty; when `tokens` is non-empty,
    // `token_count` must be either 0 or equal to `tokens.size()`.
    std::uint64_t token_count = 0;
    std::vector<Token> tokens{};
};

struct SlotRestoreExpectation {
    SlotModelIdentity model_identity{};
    KvCacheType type_k = KvCacheType::kF16;
    KvCacheType type_v = KvCacheType::kF16;
    std::uint32_t n_ctx = 0;
    std::optional<std::uint64_t> expected_token_count = std::nullopt;
};

struct SlotRestoreResult {
    SlotFileHeader header{};
    std::vector<Token> tokens{};
    std::size_t restored_state_bytes = 0;
};

// Abstraction over sequence KV state reads/writes so unit tests can run with a
// fake buffer (no model weights) while production uses LlamaCppSlotStateBackend.
class SlotStateBackend {
public:
    virtual ~SlotStateBackend() = default;

    [[nodiscard]] virtual SlotStoreStatus GetSequenceStateSize(std::int32_t seq_id,
                                                               std::size_t& out_size) const = 0;
    [[nodiscard]] virtual SlotStoreStatus GetSequenceStateData(std::int32_t seq_id,
                                                               std::span<std::uint8_t> dst,
                                                               std::size_t& out_written) const = 0;
    [[nodiscard]] virtual SlotStoreStatus SetSequenceStateData(std::int32_t seq_id,
                                                               std::span<const std::uint8_t> src) = 0;
};

// Production sequence-state adapter backed by llama.cpp v1.0.0:
// llama_state_seq_get_size / llama_state_seq_get_data / llama_state_seq_set_data
// (include/llama.h:856-875).
class LlamaCppSlotStateBackend final : public SlotStateBackend {
public:
    explicit LlamaCppSlotStateBackend(llama_context* ctx) noexcept : ctx_(ctx) {}

    [[nodiscard]] SlotStoreStatus GetSequenceStateSize(std::int32_t seq_id,
                                                       std::size_t& out_size) const override;
    [[nodiscard]] SlotStoreStatus GetSequenceStateData(std::int32_t seq_id,
                                                       std::span<std::uint8_t> dst,
                                                       std::size_t& out_written) const override;
    [[nodiscard]] SlotStoreStatus SetSequenceStateData(std::int32_t seq_id,
                                                       std::span<const std::uint8_t> src) override;

private:
    llama_context* ctx_ = nullptr;
};

// Computes `SlotModelIdentity` from a GGUF file on disk by parsing only the
// GGUF header/KV/tensor-info region via `gguf_init_from_callback` (bounded by
// `max_header_bytes`) and SHA-256 hashing `[0, gguf_get_data_offset(ctx))`.
[[nodiscard]] SlotStoreStatus ComputeGgufModelIdentity(
    const std::filesystem::path& gguf_path, SlotModelIdentity& out_identity,
    std::size_t max_header_bytes = kDefaultMaxGgufHeaderBytes,
    std::uint64_t* out_hashed_header_bytes = nullptr);

// Validates all header fields and length bounds and encodes `header` into 128 LE bytes.
[[nodiscard]] SlotStoreStatus SerializeSlotHeader(
    const SlotFileHeader& header, std::array<std::uint8_t, kSlotFileHeaderBytes>& out_bytes,
    const SlotStoreLimits& limits = {});

// Validates magic, format version, and every header length field BEFORE any
// payload allocation can occur, populating `out_header` on success.
[[nodiscard]] SlotStoreStatus DeserializeSlotHeader(
    std::span<const std::uint8_t> bytes, SlotFileHeader& out_header,
    const SlotStoreLimits& limits = {});

// Rejects empty paths, embedded NULs, `..` traversal components, directories,
// and symlink/reparse targets.
[[nodiscard]] SlotStoreStatus ValidateSlotFilePath(const std::filesystem::path& path);
[[nodiscard]] SlotStoreStatus ValidateSlotFilePathUtf8(std::string_view utf8_path,
                                                       std::filesystem::path& out_path);

// Confines a single slot filename (no separators, drive prefixes, NULs, or `..`)
// inside `root_dir` for server/endpoint slot persistence hooks.
[[nodiscard]] SlotStoreStatus ResolveSlotFilePath(const std::filesystem::path& root_dir,
                                                  std::string_view slot_filename,
                                                  std::filesystem::path& out_path);

// Captures sequence `seq_id` from `backend` and writes `[header || payload]`
// atomically via an exclusive sibling temp file + flush + atomic rename.
[[nodiscard]] SlotStoreStatus SaveSlotFile(
    const std::filesystem::path& path, std::int32_t seq_id, const SlotSaveMetadata& meta,
    const SlotStateBackend& backend, const SlotStoreLimits& limits = {},
    SlotFileHeader* out_header = nullptr);

// Validates file bounds, header fields, model/KV/context/token expectations,
// and payload SHA-256 checksum BEFORE calling `backend.SetSequenceStateData`.
[[nodiscard]] SlotStoreStatus RestoreSlotFile(
    const std::filesystem::path& path, std::int32_t dest_seq_id,
    const SlotRestoreExpectation& expected, SlotStateBackend& backend,
    SlotRestoreResult* out_result = nullptr, const SlotStoreLimits& limits = {});

// Convenience wrappers over `LlamaCppSlotStateBackend(ctx)`.
[[nodiscard]] SlotStoreStatus SaveLlamaSequenceSlot(
    const std::filesystem::path& path, llama_context* ctx, std::int32_t seq_id,
    const SlotSaveMetadata& meta, const SlotStoreLimits& limits = {},
    SlotFileHeader* out_header = nullptr);

[[nodiscard]] SlotStoreStatus RestoreLlamaSequenceSlot(
    const std::filesystem::path& path, llama_context* ctx, std::int32_t dest_seq_id,
    const SlotRestoreExpectation& expected, SlotRestoreResult* out_result = nullptr,
    const SlotStoreLimits& limits = {});

}  // namespace sonder::backends::llamacpp
