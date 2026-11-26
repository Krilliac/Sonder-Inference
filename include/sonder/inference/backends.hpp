// Sonder Inference: built-in (core) backend factories. Optional backends
// (ollama, llamacpp) live in their module directories; see docs/MODULES.md.
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>

#include "sonder/inference/backend.hpp"

namespace sonder::inference {

// ---------------------------------------------------------------------------
// MOCK BACKEND - deterministic, for tests and harness development ONLY.
// It performs no inference: output is a pure function of (prompt, seed,
// sampling limits). Never use it for quality or performance claims.
// ---------------------------------------------------------------------------
struct MockBackendOptions {
    std::chrono::microseconds token_delay{0};  // artificial per-token latency
    std::int32_t default_completion_tokens = 16; // natural EOS point
    std::int32_t fail_after_tokens = -1;         // >= 0 injects a backend error
    // Advertise Capability::token_logits: the engine then samples with its own
    // sampler chain over a small fixed vocabulary (the mock's word list plus
    // an end-of-sequence token) using deterministic pseudo-logits.
    bool token_logits = false;
    // Test hook (token_logits only): every logit is -infinity, as if the
    // request's policy had banned the whole vocabulary.
    bool ban_all_tokens = false;
    // Opt-in deterministic recurrent-state checkpoint support for hybrid and
    // recurrent tests. Defaults preserve existing mock capabilities and models.
    bool enable_recurrent_checkpoints = false;
    ModelArchitecture default_architecture = ModelArchitecture::attention_only;
    // Upper bound on checkpoint snapshot bytes accepted by mock capture/restore
    // (defaults to kMaxRecurrentCheckpointBytes; tests may lower it to exercise
    // oversized-snapshot rejection without large allocations).
    std::size_t max_checkpoint_bytes = kMaxRecurrentCheckpointBytes;
    // Enable deterministic cross-encoder reranking and pair scoring on MockBackend.
    bool enable_reranking = true;
    // Opt-in Matryoshka-style reduced embedding dimensions for mock tests;
    // false (default) fails closed on dimensions < native width.
    bool allow_embed_dimension_truncation = false;
    // Opt-in process-local execution checkpoint & selective preemption
    // capability advertisements (defaults false to preserve legacy mock caps).
    bool enable_execution_checkpoints = false;
    bool enable_selective_preemption = false;
    // Simulated physical slot pool size for selective preemption tests (0 = unlimited).
    std::size_t physical_slots = 4;
    // Optional tokenizer salt so tests can construct distinct tokenizer fingerprints.
    std::uint64_t tokenizer_salt = 0;
};
inline constexpr const char* kMockBackendName = "mock";
std::shared_ptr<Backend> make_mock_backend(MockBackendOptions options = {});

}  // namespace sonder::inference
