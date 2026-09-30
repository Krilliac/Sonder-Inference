// External llama-server / OpenAI-compatible upstream, without llama.cpp linkage.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sonder/inference/backend.hpp"

namespace sonder::inference {

enum class LlamaServerMode { attach, spawn };

// Same verification semantics as the internal HTTP client and Ollama adapter.
struct LlamaServerTlsOptions {
    std::string ca_bundle_path;
    std::string pinned_sha256;
    std::string pinned_cert_path;
    bool insecure_skip_verify = false;
    std::string server_name;
    std::chrono::milliseconds handshake_timeout{10000};
};

struct LlamaServerBackendOptions {
    LlamaServerMode mode = LlamaServerMode::attach;
    std::string base_url = "http://127.0.0.1:8080";
    // Spawn uses argv directly, without a shell. The supervisor appends its
    // own host and port. User host/port overrides are rejected.
    std::string executable;
    std::vector<std::string> args;
    bool allow_remote = false;
    std::chrono::milliseconds connect_timeout{3000};
    std::chrono::milliseconds request_timeout{300000};
    std::chrono::milliseconds startup_timeout{60000};
    std::chrono::milliseconds poll_interval{50};
    std::chrono::milliseconds shutdown_timeout{2000};
    std::chrono::milliseconds restart_backoff{100};
    std::chrono::milliseconds max_restart_backoff{2000};
    std::size_t max_restarts = 3;
    LlamaServerTlsOptions tls;
    // llama.cpp /completion by default. Set false for generic OpenAI servers
    // providing /v1/completions. Chat always uses /v1/chat/completions.
    bool native_completion = true;
    std::string grammar; // optional upstream GBNF grammar; never synthesized
};

// Slot snapshots are upstream-owned files, not portable Sonder KV blocks.
// Configure llama-server's --slot-save-path to enable this native API.
class LlamaServerBackend : public Backend {
  public:
    virtual Status save_slot(std::uint32_t slot, const std::string &filename) = 0;
    virtual Status restore_slot(std::uint32_t slot, const std::string &filename) = 0;
};

inline constexpr const char *kLlamaServerBackendName = "llamaserver";
std::shared_ptr<LlamaServerBackend> make_llamaserver_backend(LlamaServerBackendOptions options = {});

} // namespace sonder::inference
