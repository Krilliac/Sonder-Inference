// External llama-server / OpenAI-compatible upstream, without llama.cpp linkage.
#pragma once

#include "sonder/inference/backends/llamaserver_residency.hpp"

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

// VRAM-spill guard for a spawned child (docs/integration/vram-spill.md).
// On Windows a llama-server run with `--fit off` that overflows VRAM keeps
// running from shared system memory, 2-15x slower, and nvidia-smi does not
// show it. The guard reads the child's PDH `GPU Process Memory` counters
// after readiness and every `sample_interval`; other platforms report
// "unsupported" and are never refused or refitted.
enum class LlamaServerSpillPolicy {
    warn,     // report only (default; startup behaviour is unchanged)
    refuse,   // fail startup with the measured numbers
    auto_fit  // relaunch with a smaller --ctx-size until not spilled (bounded)
};

struct LlamaServerSpillGuardOptions {
    bool enabled = true;
    LlamaServerSpillPolicy policy = LlamaServerSpillPolicy::warn;
    // Spilled when shared usage > baseline_bytes + threshold_bytes. Measured on
    // an RTX 5070 Ti: 148-198 MiB shared when clean, 292 MiB-2.1 GB spilled.
    std::uint64_t threshold_bytes = 256ull * 1024 * 1024;
    std::uint64_t baseline_bytes = 0;
    // Added to the baseline per 1,024 tokens of --ctx-size (0 = fixed). Clean
    // shared usage grows ~1 MiB/1k ctx, ~2 MiB/1k ctx with MTP speculation.
    std::uint64_t baseline_bytes_per_1k_ctx = 0;
    std::chrono::milliseconds sample_interval{5000};
    // auto_fit: ctx -> align_down(ctx * fit_step_factor, fit_step_align),
    // at least one alignment step smaller, never below fit_min_ctx, and at
    // most fit_max_attempts relaunches. Requires --ctx-size/-c in args.
    double fit_step_factor = 0.85;
    std::uint64_t fit_step_align = 1024;
    std::uint64_t fit_min_ctx = 8192;
    std::size_t fit_max_attempts = 4;
    LlamaServerResidencyGuardOptions residency;
};

struct LlamaServerDiagnosticsOptions {
    // Child log scanned for performance warnings (FlashAttention K/V f16
    // conversion, ignored nextn/MTP tensors, CPU fallback). Appended as
    // "--log-file <path>" unless args already contain --log-file, which is
    // then read instead. Empty and no --log-file: no log diagnostics.
    std::string log_file;
    // Warn when FlashAttention may run with mismatched K/V cache types.
    bool kv_pairing_check = true;
};

// Optional prefix replay. Template kwargs MUST match real requests, including
// thinking settings; changing them can invalidate the prefix from token zero.
struct LlamaServerWarmupOptions {
    std::string messages_file;  // empty disables warm-up
    json::Object chat_template_kwargs;
    bool all_slots = true;
    std::vector<std::uint32_t> slots;
    bool on_restart = true;
    // UTF-8 bytes, including JSON syntax; bounded before parsing/allocation.
    std::size_t max_prefix_chars = 262144;
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
    // Served per-slot context, used when the upstream has no GET /props (a
    // generic OpenAI server). /props n_ctx wins when present. 0 = unknown.
    // Requests may set num_ctx up to it (a no-op); larger values are refused.
    std::uint64_t context_length = 0;
    // Native mode (llama.cpp) always sends "cache_prompt": true, and with
    // slot_affinity pins ChatRequest::session_key to one llama-server slot
    // ("id_slot", slot count from /props total_slots). Generic OpenAI mode
    // (native_completion = false) sends neither llama.cpp-only field.
    bool slot_affinity = true;
    // Spawn mode only. Reported through Backend::runtime_status().
    LlamaServerSpillGuardOptions spill_guard;
    LlamaServerDiagnosticsOptions diagnostics;
    LlamaServerWarmupOptions warmup;
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
