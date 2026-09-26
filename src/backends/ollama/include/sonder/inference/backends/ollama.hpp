// Sonder Inference: Ollama compatibility backend (module src/backends/ollama).
// Available when the build defines SONDER_HAS_OLLAMA_BACKEND.
#pragma once

#include <chrono>
#include <memory>
#include <string>

#include "sonder/inference/backend.hpp"

namespace sonder::inference {

// ---------------------------------------------------------------------------
// Ollama compatibility backend (docs/BACKENDS.md, "Backend 0"). Talks to an
// Ollama server over plain HTTP. Loopback only unless allow_remote is set;
// remote workers should sit behind TLS, which this adapter does not speak.
// ---------------------------------------------------------------------------
struct OllamaBackendOptions {
    std::string base_url = "http://127.0.0.1:11434";
    bool allow_remote = false;
    std::chrono::milliseconds connect_timeout{3000};
    std::chrono::milliseconds request_timeout{300000};
    std::string keep_alive;  // forwarded as keep_alive when non-empty, e.g. "5m"
};
inline constexpr const char* kOllamaBackendName = "ollama";
std::shared_ptr<Backend> make_ollama_backend(OllamaBackendOptions options = {});

}  // namespace sonder::inference
