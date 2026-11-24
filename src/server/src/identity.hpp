// Internal: backend identity for GET /v1/sonder/identity (docs/SERVER.md).
//
// The object has exactly the nine keys of Sonder Runtime's BackendIdentity
// (backend, model, model_digest, quantization, backend_version,
// tokenizer_digest, template_digest, context_tokens, hardware). Digests are
// lowercase 64-hex SHA-256 and context_tokens is > 0. When any value cannot
// be measured the identity is null and `reason` says why; nothing is
// fabricated.
#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/model.hpp"

namespace sonder::inference::server::detail {

// UTF-8 std::string <-> std::filesystem::path helpers without deprecated
// u8path() and without narrow ANSI codepage conversion on Windows.
inline std::filesystem::path path_from_utf8(std::string_view utf8_path) {
    std::u8string u;
    u.reserve(utf8_path.size());
    for (const char ch : utf8_path) {
        u.push_back(static_cast<char8_t>(static_cast<unsigned char>(ch)));
    }
    return std::filesystem::path(std::move(u));
}

inline std::string path_to_utf8(const std::filesystem::path& path) {
    const std::u8string u = path.u8string();
    std::string out;
    out.reserve(u.size());
    for (const char8_t ch : u) {
        out.push_back(static_cast<char>(ch));
    }
    return out;
}

struct IdentityResult {
    json::Value backend_identity;  // object, or null
    std::optional<std::string> reason;
};

// Streams `path` in fixed 64 KiB chunks to compute its lowercase 64-hex
// SHA-256 without loading the file into RAM. Rejects empty paths and
// non-regular files cleanly.
bool compute_gguf_file_sha256(const std::string& path, std::string& out_hex, std::string& error);

// Computes a complete 9-key BackendIdentity for a GGUF-backed model from
// `gguf_path` (streaming SHA-256 + single-pass GGUF header inspection,
// cached by `(path, file_size, mtime)`). When `context_tokens` is 0, falls
// back to the GGUF header's `<arch>.context_length`. When `gpus` is non-null,
// uses the caller's cached GPU inventory rather than invoking platform GPU
// discovery concurrently. Returns null with a descriptive `reason` whenever
// any required field (including `backend_version`) cannot be measured.
IdentityResult gguf_backend_identity(const std::string& backend, const std::string& model_id,
                                     const std::string& gguf_path, std::uint64_t context_tokens,
                                     const std::optional<std::string>& backend_version,
                                     const std::vector<GpuDeviceInfo>* gpus = nullptr);

// `backend_version` is the backend's probe() result when it succeeded.
IdentityResult backend_identity(const Model& model, const std::optional<std::string>& backend_version);

// Same, from a backend name and the model descriptor. `descriptor` is null
// when the model has not been loaded yet (lazy residency); backends whose
// identity needs the descriptor then report null with a reason.
IdentityResult backend_identity(const std::string& backend, const std::string& model_id,
                                const ModelDescriptor* descriptor, const std::optional<std::string>& backend_version);

}  // namespace sonder::inference::server::detail
