// Sonder Inference: execution backend interface.
//
// Backends own tensor execution; Sonder owns policy (sessions, scheduling,
// cache, lifecycle, telemetry). The interface is deliberately not shaped
// around a single backend (see docs/BACKENDS.md).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/sampling.hpp"

namespace sonder::inference {

// Capability advertisement (docs/BACKENDS.md). Sonder never assumes a
// capability that a backend does not declare.
enum class Capability : std::uint32_t {
    tokenization = 1u << 0,
    streaming = 1u << 1,
    batched_prefill = 1u << 2,
    continuous_batch_decode = 1u << 3,
    kv_export = 1u << 4,
    kv_import = 1u << 5,
    kv_copy = 1u << 6,
    kv_quantization = 1u << 7,
    prefix_reuse = 1u << 8,
    speculative_decode = 1u << 9,
    lora = 1u << 10,
    layer_telemetry = 1u << 11,
    structured_output = 1u << 12,
    embeddings = 1u << 13,
    deterministic = 1u << 14,  // identical inputs + seed => identical output
    remote_process = 1u << 15, // execution happens in another process/server
    token_logits = 1u << 16,   // exposes per-step logits (open_token_stream); Sonder samples
};

// Vocabulary index as produced by a backend tokenizer.
using TokenId = std::int32_t;

struct BackendCapabilities {
    std::uint32_t bits = 0;
    [[nodiscard]] bool has(Capability c) const noexcept { return (bits & static_cast<std::uint32_t>(c)) != 0; }
    BackendCapabilities& add(Capability c) noexcept {
        bits |= static_cast<std::uint32_t>(c);
        return *this;
    }
    [[nodiscard]] std::vector<std::string> names() const;
};

struct ModelDescriptor {
    std::string name;
    std::string backend;
    std::string format;        // e.g. "gguf", "mock"
    std::string family;        // architecture family when known
    std::string parameter_size;
    std::string quantization;  // e.g. "Q4_K_M"
    std::uint64_t size_bytes = 0;
    std::uint64_t context_length = 0;  // 0 when unknown
    // False when load_model() only fetched metadata and the weights are not
    // resident (e.g. Ollama loads lazily on first request).
    bool resident = true;
};

struct ModelLoadOptions {
    std::string model;
    std::string device_id = "cpu:0";
};

struct GenerateRequest {
    std::string request_id;
    std::string prompt;
    SamplingConfig sampling;
};

struct TokenChunk {
    std::string_view text;
    std::uint64_t index = 0;  // 0-based chunk index within the request
};

enum class StopReason { none, max_tokens, stop_sequence, end_of_sequence, cancelled, callback, error };
const char* to_string(StopReason reason) noexcept;

struct GenerateStats {
    std::uint64_t prompt_tokens = 0;
    std::uint64_t completion_tokens = 0;
    std::uint64_t chunks = 0;              // streamed chunks delivered to the callback
    std::uint64_t load_ns = 0;             // backend-reported, 0 when unknown
    std::uint64_t prompt_eval_ns = 0;      // backend-reported, 0 when unknown
    std::uint64_t eval_ns = 0;             // backend-reported, 0 when unknown
    bool token_counts_from_backend = false;
    StopReason stop_reason = StopReason::none;
};

// Return false to stop generation early (StopReason::callback).
using TokenCallback = std::function<bool(const TokenChunk&)>;

// Token-level decode loop for backends that advertise Capability::token_logits.
// Sonder owns sampling: it asks for the next logits, samples a token with its
// own sampler chain (src/sampling) and commits it with accept().
class TokenStream {
public:
    virtual ~TokenStream() = default;
    [[nodiscard]] virtual std::size_t vocab_size() const = 0;
    // Prompt tokens as the backend tokenized them (after prefill).
    [[nodiscard]] virtual const std::vector<TokenId>& prompt_tokens() const = 0;
    // Logits for the next position (size == vocab_size()). The span stays
    // valid until the next call on this stream.
    virtual Result<std::span<const float>> next_logits(const CancellationToken& cancel) = 0;
    // Commits `token` as the next generated token and returns its text piece.
    virtual Status accept(TokenId token, std::string& piece) = 0;
    // True for end-of-generation tokens (EOS/EOT).
    [[nodiscard]] virtual bool is_end_of_generation(TokenId token) const = 0;
};

class BackendModel {
public:
    virtual ~BackendModel() = default;
    [[nodiscard]] virtual const ModelDescriptor& descriptor() const = 0;
    // Synchronous streaming generation. Implementations must poll `cancel`
    // at least once per produced chunk and while waiting on I/O, and return
    // ErrorCode::cancelled promptly once it trips.
    virtual Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                           const TokenCallback& on_chunk) = 0;
    // Optional (Capability::tokenization): exact prompt tokenization. The engine
    // falls back to an approximate accounting tokenizer when unsupported.
    virtual Result<std::vector<TokenId>> tokenize(std::string_view text) {
        (void)text;
        return Status(ErrorCode::unsupported, "backend does not expose tokenization");
    }
    // Optional (Capability::token_logits): open a token-level decode loop for
    // `request` (prompt prefilled). Sonder samples from the returned logits.
    virtual Result<std::unique_ptr<TokenStream>> open_token_stream(const GenerateRequest& request) {
        (void)request;
        return Status(ErrorCode::unsupported, "backend does not expose token logits");
    }
};

class Backend {
public:
    virtual ~Backend() = default;
    [[nodiscard]] virtual std::string name() const = 0;
    [[nodiscard]] virtual std::string description() const = 0;
    [[nodiscard]] virtual BackendCapabilities capabilities() const = 0;
    // Cheap reachability/health check. Returns a version string on success.
    virtual Result<std::string> probe() = 0;
    virtual Result<std::vector<ModelDescriptor>> list_models() = 0;
    virtual Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) = 0;
};

}  // namespace sonder::inference
