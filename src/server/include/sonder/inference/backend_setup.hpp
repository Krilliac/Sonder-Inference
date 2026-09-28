// Sonder Inference: shared backend construction and environment defaults for
// command-line hosts (`sonder-infer serve`, and the other sonder-infer
// commands once they adopt it). Part of the server module (SONDER_HAS_SERVER);
// see docs/SERVER.md.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference {

struct BackendSetup {
    std::string backend;  // "mock", "ollama" or "llamacpp"
    // Ollama base URL; empty uses the backend default (http://127.0.0.1:11434).
    std::string ollama_url;
    // Allow a non-loopback Ollama host. Remote hosts also need https://, which
    // requires a SONDER_WITH_TLS=ON build (not wired into the default build).
    bool ollama_allow_remote = false;
    // llama.cpp: directories scanned for *.gguf files.
    std::vector<std::string> model_dirs;
    // llama.cpp: layers offloaded for gpu:* devices (-1 = all) and context
    // length (0 = model training context); unset keeps the backend defaults.
    std::optional<std::int32_t> llamacpp_gpu_layers;
    std::optional<std::uint32_t> llamacpp_context_length;
    // llama.cpp: tensor placement overrides, "PATTERN=DEVICE" each
    // (see parse_tensor_override). Applied in order; first match wins.
    std::vector<std::string> llamacpp_tensor_overrides;
    // MOCK backend: artificial per-token latency.
    std::chrono::microseconds mock_token_delay{0};
    // MOCK backend: >= 0 injects a backend error after that many tokens
    // (failure-path tests; MockBackendOptions::fail_after_tokens).
    std::int32_t mock_fail_after_tokens = -1;
};

// Values taken from the environment when the matching option is absent:
//   backend    <- SONDER_INFER_BACKEND
//   model      <- SONDER_INFER_MODEL
//   ollama_url <- SONDER_OLLAMA_URL, then OLLAMA_HOST (normalized)
struct BackendEnvDefaults {
    std::optional<std::string> backend;
    std::optional<std::string> model;
    std::optional<std::string> ollama_url;
};

// Reads the defaults through `lookup` (std::getenv when empty). Empty
// variables count as unset.
using EnvLookup = std::function<std::optional<std::string>(const char* name)>;
BackendEnvDefaults backend_env_defaults(const EnvLookup& lookup = {});

// Turns an OLLAMA_HOST value ("host", "host:port", "http://host:port",
// "0.0.0.0") into a base URL. A wildcard bind address maps to 127.0.0.1 and
// a missing port to 11434.
std::string normalize_ollama_host(std::string_view value);

// Splits "PATTERN=DEVICE" at the last '=' (patterns may contain '=').
// Errors: invalid_argument when either side is empty or '=' is missing.
Result<std::pair<std::string, std::string>> parse_tensor_override(std::string_view spec);

// Pattern for MoE expert weights (llama.cpp's --cpu-moe); `--moe-experts cpu`
// adds "<this>=cpu" ahead of any explicit --tensor-override.
inline constexpr const char* kMoeExpertTensorOverridePattern = R"(\.ffn_(up|down|gate)_(ch|)exps)";

// Backends compiled into this build, in a stable order ("mock" first).
std::vector<std::string> available_backend_names();

// True for backends whose output is synthetic (the MOCK backend).
bool is_synthetic_backend(std::string_view name) noexcept;

// Builds the named backend. Errors: invalid_argument for an unknown name,
// unsupported for a backend this build does not include.
Result<std::shared_ptr<Backend>> make_backend(const BackendSetup& setup);

}  // namespace sonder::inference
