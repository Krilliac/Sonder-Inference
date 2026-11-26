// External llama-server / OpenAI-compatible upstream, without llama.cpp linkage.
#pragma once

#include "sonder/inference/backends/llamaserver_residency.hpp"

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
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
    // Integrated UMA GPUs (e.g. Ryzen APUs) legitimately allocate GPU buffers
    // in shared/GTT system memory. When true, the spill guard does not flag
    // shared usage as a VRAM spill and propagates to residency unless
    // `residency_uma_mode_explicit` is set.
    bool uma_mode = false;
    // Optional tri-state override ("off", "on", "auto", "true", "false").
    // When non-empty, supersedes `uma_mode`, resolving "auto" via
    // enumerate_gpu_devices() and any --device/-dev hint in args.
    std::string uma_mode_policy;
    // True when `residency.uma_mode` was set independently of `uma_mode`.
    bool residency_uma_mode_explicit = false;
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

struct LlamaServerStallGuardOptions {
    bool enabled = true;
    std::chrono::seconds stall_seconds{90};
    std::string policy = "warn";  // warn or restart (spawn only); "cancel" with generic_progress
    // Generic OpenAI attach (native_completion = false) only. Off by default so
    // existing configs behave as before. When true, every active streaming
    // request must make content-bearing progress (a delta with content or
    // reasoning text; empty keepalive deltas and SSE comments do not count)
    // at least every stall_seconds, measured from the last such delta or from
    // the request being sent. policy "warn" records a runtime warning and
    // keeps waiting; policy "cancel" ends the request with ErrorCode::protocol_error.
    // Keep stall_seconds above the upstream's queue wait (colibri: 300 s).
    bool generic_progress = false;
};

// Host RAM admission and floor guard for a spawned child (spawn mode only;
// src/backends/llamaserver/host_memory.hpp). Defaults are inert: with
// estimated_resident_bytes == 0 no pre-spawn check runs, and the floor guard
// is off. serve fills the estimates from the active launch profile.
struct LlamaServerHostMemoryOptions {
    // VramEstimate::host_resident_bytes / host_private_bytes (0 = no estimate).
    std::uint64_t estimated_resident_bytes = 0;
    std::uint64_t estimated_private_bytes = 0;
    // The estimate assumes everything fits the GPU (llama-server --fit with no
    // VRAM budget known): the supervisor raises a warning.
    bool estimate_lower_bound = false;
    // Kept free on top of the estimate for the OS and other processes.
    std::uint64_t reserve_bytes = 2ull * 1024 * 1024 * 1024;
    // Start anyway when the pre-spawn check would refuse (a warning is raised).
    bool override_refusal = false;
    struct Floor {
        bool enabled = false;
        std::string policy = "warn";  // warn, stop_admission or restart_smaller_ctx
        std::uint64_t floor_bytes = 0;
        std::uint64_t resume_bytes = 0;  // 0 = floor + floor / 4
        std::size_t consecutive_samples = 2;
        std::chrono::milliseconds sample_interval{5000};
        std::chrono::milliseconds restart_cooldown{60000};
        std::size_t max_restarts = 1;
    } floor;
};

// Time- and load-aware adaptive speculation controller (opt-in; native
// llama-server mode only). Mirrors AdaptiveSpeculationConfig with
// `enabled = false` by default so existing behavior stays unchanged unless
// explicitly opted in.
struct LlamaServerAdaptiveSpeculationOptions {
    bool enabled = false;
    std::uint32_t min_draft_n = 0;
    std::uint32_t initial_draft_n = 3;
    std::uint32_t max_draft_n = 6;
    std::size_t window_size = 16;
    std::size_t min_samples = 4;
    double pause_acceptance_threshold = 0.25;
    double step_down_acceptance_threshold = 0.45;
    double step_up_acceptance_threshold = 0.72;
    std::uint32_t cooldown_requests = 8;
    std::uint32_t probe_draft_n = 1;
    std::uint64_t min_draft_tokens_per_sample = 4;
    // Measured-cost and bounded-LRU policy controls.
    std::size_t max_entries = 128;
    std::size_t min_baseline_samples = 2;
    double cost_step_down_ratio = 1.05;
    double cost_pause_ratio = 1.12;
    double cost_probe_recover_ratio = 0.98;
    std::uint32_t cost_step_cooldown = 4;
    std::uint32_t baseline_probe_interval = 0;
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
    // Additive child-only environment overrides; empty inherits unchanged, except inherited AIP_*
    // variables, which are removed; AIP_* overrides are refused.
    std::vector<std::pair<std::string, std::string>> environment;
    LlamaServerWarmupOptions warmup;
    LlamaServerStallGuardOptions stall_guard;
    // Spawn mode only.
    LlamaServerHostMemoryOptions host_memory;

    // ---- Slow OpenAI-compatible upstreams (attach mode; docs/integration/colibri.md) ----
    // Name of an environment variable holding a bearer token. The token is
    // read from the process environment (never from config), sent as
    // "Authorization: Bearer <token>" on every completion, discovery and /props request (the native-attach monitor keeps the value read at construction), and never
    // logged or reported (runtime status shows only the variable name and
    // "[redacted]"). Attach mode only; combine with HTTPS or loopback.
    // Not supported together with warmup. Empty = no Authorization header.
    std::string api_key_env;
    // Stream idle timeout: maximum silence between received bytes once the
    // response head has arrived. 0 = disabled (historical behaviour). Must
    // exceed keepalive_gap, otherwise a healthy upstream that only sends its
    // periodic empty keepalive delta would be cut off. Required when
    // unlimited_request_timeout is set.
    std::chrono::milliseconds idle_timeout{0};
    // First-content timeout: from sending the request until the first delta
    // carrying content or reasoning text. Empty keepalive deltas do not count.
    // 0 = disabled. Required when unlimited_request_timeout is set.
    std::chrono::milliseconds first_content_timeout{0};
    // True removes request_timeout as a cap on streaming completions (attach
    // only; discovery and /props requests keep request_timeout). The slow
    // upstream may legitimately stream for hours at 0.2 tok/s. Needs
    // first_content_timeout and idle_timeout.
    bool unlimited_request_timeout = false;
    // The upstream's longest gap between keepalive bytes (colibri: 10 s).
    // idle_timeout is validated to be strictly larger.
    std::chrono::milliseconds keepalive_gap{10000};
    // False declares that the upstream cannot honour a grammar; a non-empty
    // `grammar` is then refused when the backend is built.
    bool upstream_grammar = true;
    // Opt-in workload-aware adaptive speculation policy (native mode only).
    LlamaServerAdaptiveSpeculationOptions adaptive_speculation;
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
