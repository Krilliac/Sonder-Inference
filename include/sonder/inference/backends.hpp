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
};
inline constexpr const char* kMockBackendName = "mock";
std::shared_ptr<Backend> make_mock_backend(MockBackendOptions options = {});

}  // namespace sonder::inference
