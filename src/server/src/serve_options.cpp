#include "serve_options.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <locale>
#include <sstream>

#if !defined(_WIN32)
#  include <sys/stat.h>
#endif

#include "server/src/env_policy.hpp"
#include "cli/cli_spec.hpp"
#include "request_path.hpp"
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/telemetry.hpp"

namespace sonder::inference::server::detail {

const char* const kServeUsage = R"(sonder-infer serve - local HTTP API and live telemetry (docs/SERVER.md)

Usage:
  sonder-infer serve --backend mock|remote-mock|ollama|ollama_cloud|llamacpp|llamaserver [--model ID]... [options]

Server:
  --host HOST             listen address (default 127.0.0.1; non-loopback needs --token-file)
  --port N                listen port (default 11437; 0 = ephemeral)
  --token-file PATH       require "Authorization: Bearer <token>" (token read from PATH)
  --cors-origin ORIGIN    allow a browser origin (repeatable, exact match)
  --no-default-cors       drop the default Observatory origins
  --max-connections N     concurrent connections (default 64)
  --max-body-bytes N      request body limit (default 4194304)
  --shutdown-grace-ms N   time in-flight requests get on shutdown (default 5000)
  --stream-keepalive-ms N SSE `: keep-alive` heartbeat interval during long
                          prefill waits in ms (0..600000, default 15000; 0 = off)
  --disable-response-store  disable in-memory /v1/responses history storage
                          (rejects previous_response_id chaining)
  --response-history-ttl S  TTL in seconds for /v1/responses history entries
                          (0..86400, default 3600; 0 disables storage)
  --ready-file PATH       write {"url","pid","instance_id","api_version"} once listening
                          (models may still be loading: poll /v1/sonder/health for
                          200); removed again when the server exits
  --log-format text|json  access log and diagnostics on stderr (default text)

Backend:
  --backend NAME          mock / remote-mock (synthetic, tests only), ollama,
                          ollama_cloud (Ollama's hosted models over HTTPS), llamacpp,
                          llamaserver (external llama-server or OpenAI upstream)
                          (env SONDER_INFER_BACKEND). remote-mock launches a private
                          worker subprocess using this sonder-infer executable.
  --model ID              model to serve (repeatable; the first is the default and
                          answers to "default"; env SONDER_INFER_MODEL). The mock
                          backend serves "mock" when no model is given.
  --ollama-url URL        Ollama base URL (env SONDER_OLLAMA_URL, then OLLAMA_HOST)
  --ollama-allow-remote   allow a non-loopback Ollama host (needs https, i.e. a
                          SONDER_WITH_TLS=ON build)
  --ollama-cloud-config PATH
                          ollama_cloud JSON (base_url, api_key_env or api_key_file,
                          models allowlist, timeouts); required by --backend
                          ollama_cloud, whose --model defaults to the allowlist.
                          Needs a SONDER_WITH_TLS=ON build
  --model-dir DIR         llama.cpp: directory with *.gguf files (repeatable)
  --device ID             device to load models on (llama.cpp: gpu:0 offloads to
                          the GPU; default cpu)
  --lazy-models           register models at startup and load each on its first
                          request (default: load every model before ready)
  --model-idle-ttl S      unload a model no request has used for S seconds; the
                          next request loads it again (default 0 = never)
  --max-resident-models N keep at most N models loaded, evicting the least recently
                          used idle one (default 0 = no cap; in-flight requests
                          always keep their model)
  --max-resident-vram-mib N
                          byte budget for resident model VRAM in MiB (default 0 =
                          unlimited; evicts unpinned models by reload cost/byte)
  --max-resident-ram-mib N
                          byte budget for resident model host RAM in MiB (default
                          0 = unlimited; evicts unpinned models by reload cost/byte)
  --gpu-layers N          llama.cpp: layers offloaded to the GPU (-1 = all, default)
  --context-length N      llama.cpp: context length (0 = model training context)
  --moe-experts WHERE     llama.cpp: cpu keeps MoE expert weights in system RAM
                          while attention stays on the GPU; gpu (default) leaves
                          placement to --gpu-layers
  --tensor-override P=D   llama.cpp: place tensors matching regex P on device D
                          (cpu, or a llama.cpp device such as Vulkan0); repeatable
  --batch-size N          llama.cpp: max prompt tokens per decode call (default 512)
  --ubatch-size N         llama.cpp: physical micro-batch, <= --batch-size
                          (default: llama.cpp's 512, capped at --batch-size)
  --cache-type-k T        llama.cpp: K cache type auto (default: q8_0 when model
                          head dims % 32 == 0 and flash-attn != off, else f16),
                          f16, f32, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl,
                          f8_e4m3, q2_0, pq2_0/pq2_0m
  --cache-type-v T        llama.cpp: V cache type, same values and auto default;
                          an explicit quantized V cache needs flash attention
  --allow-q4-k-cache      llama.cpp: allow q4_0 for --cache-type-k on models with
                          K-cache Hadamard rotation
  --flash-attn MODE       llama.cpp: auto (default), on or off
  --swa-kv-mode MODE      llama.cpp: sliding-window KV policy full (default) or
                          compact_if_qualified (guarded by min_retained_pos)
  --threads N             llama.cpp: CPU thread count (1..256)
  --no-mmap               llama.cpp: disable GGUF memory-mapping (direct read)
  --no-kv-unified         llama.cpp: disable unified KV buffer across slots
  --no-prefix-cache       llama.cpp: disable prompt KV prefix reuse across turns
  --startup-smoke-test    llama.cpp: run a post-load greedy smoke test (default off)
  --slots N               llama.cpp: cached KV sequence slots (1..16, default 1)
  --spec-type TYPE        llama.cpp: in-process speculative decoding
                          (none|auto|ngram|ngram-mod|suffix|draft-mtp, default none)
  --spec-draft-max N      llama.cpp: max draft tokens per speculative step (0..64)
  --moe-cache-mib N       llama.cpp: dynamic SLRU MoE expert VRAM cache in MiB
                          (0..262144, default 0 = off)
  --moe-cache-cpu-share F unused on llamacpp (rejected; use llamaserver profile/args)
  --cache-ram-mib N       llama.cpp: bounded L2 host-RAM prefix state cache in MiB
                          (0..65536, default 0 = off)
  --mmproj PATH           llama.cpp: multimodal projector GGUF path
                          (--batch-size through --mmproj: --backend
                          llamacpp only; llama-server takes them in its spawn args)
  --mock-delay-ms N       mock backend per-token delay
  --llamaserver-config PATH  JSON config (mode, URL/executable, args, TLS and timeouts)
  --llamaserver-url URL     attach upstream URL (default http://127.0.0.1:8080)
  --llamaserver-executable PATH  spawn executable
  --llamaserver-arg ARG     repeatable spawn argument (passed verbatim)
  --llamaserver-mode MODE   attach or spawn
  --warmup-file PATH         llamaserver prefix messages JSON (requires --backend llamaserver)

Launch profiles (docs/integration/launch-profiles.md):
  --profiles PATH         JSON file of typed per-model launch profiles
  --profile NAME          serve that profile (repeatable; the first is the default);
                          selects the backend and models and sets serving options
  --all-profiles          serve every profile in --profiles sharing the primary
                          profile's backend (or --backend if specified)
  --vram-budget-mib N     VRAM budget for the fit check; overrides the profile's
                          vram_budget_mib (default: the profile's value, else
                          device memory - 1536 MiB)
  --allow-overcommit      start even when the estimate exceeds the budget

Telemetry (Observatory envelope v1; live at /v1/telemetry/sse and /v1/telemetry/ndjson):
  --telemetry-level L     off|metrics|standard|deep (default standard)
  --telemetry-buffer N    retained events for resume (default 8192)
  --telemetry PATH        also write JSONL to PATH ('-' for stderr)
  --capture-text          include generated text in token events (requires --token-file);
                          same as --text-capture on
  --text-capture P        telemetry text capture policy: off (default), on (generated
                          text in token events) or full (on, plus prompt text in
                          request.started); env SONDER_INFER_TEXT_CAPTURE; on/full
                          require --token-file; declared in discovery as text_capture;
                          a request opts out with the X-Sonder-No-Capture: 1 header

Exit status: 0 after a clean shutdown (SIGINT/SIGTERM; a second signal exits
at once), 1 on a runtime error (e.g. port in use), 2 on a usage error.
)";

namespace {

enum class Kind { value, repeat, flag };

// llama.cpp context options of the direct llamacpp backend (batch, micro-batch,
// KV cache types, flash attention). serve refuses them for any other backend,
// which would ignore them, and next to --profile, whose typed fields set them.
constexpr const char* kLlamaCppContextFlags[] = {"batch-size", "ubatch-size", "cache-type-k", "cache-type-v",
                                                 "flash-attn", "swa-kv-mode"};
// Tensor placement options of the direct llamacpp backend
// (BackendSetup::llamacpp_tensor_overrides). A llamaserver launch profile
// refuses them, because the spawned llama-server never sees them; a llamacpp
// profile takes them from the command line. (--device stays allowed: every
// backend reports it as the model's device_id in telemetry and health.)
constexpr const char* kLlamaCppPlacementFlags[] = {"moe-experts", "tensor-override"};

const std::vector<std::pair<std::string, Kind>>& spec() {
    static const std::vector<std::pair<std::string, Kind>> kSpec = {
        {"host", Kind::value},                   {"port", Kind::value},
        {"backend", Kind::value},                {"model", Kind::repeat},
        {"ollama-url", Kind::value},             {"ollama-allow-remote", Kind::flag},
        {"ollama-cloud-config", Kind::value},
        {"model-dir", Kind::repeat},             {"token-file", Kind::value},
        {"cors-origin", Kind::repeat},           {"no-default-cors", Kind::flag},
        {"telemetry-level", Kind::value},        {"telemetry-buffer", Kind::value},
        {"telemetry", Kind::value},              {"capture-text", Kind::flag},
        {"text-capture", Kind::value},
        {"max-connections", Kind::value},        {"max-body-bytes", Kind::value},
        {"shutdown-grace-ms", Kind::value},      {"stream-keepalive-ms", Kind::value},
        {"disable-response-store", Kind::flag},  {"response-history-ttl", Kind::value},
        {"ready-file", Kind::value},
        {"mock-delay-ms", Kind::value},          {"log-format", Kind::value},
        {"llamaserver-config", Kind::value},     {"llamaserver-url", Kind::value},
        {"llamaserver-executable", Kind::value}, {"llamaserver-arg", Kind::repeat},
        {"llamaserver-mode", Kind::value},       {"warmup-file", Kind::value},
        {"gpu-layers", Kind::value},             {"context-length", Kind::value},
        {"device", Kind::value},                 {"lazy-models", Kind::flag},
        {"model-idle-ttl", Kind::value},         {"max-resident-models", Kind::value},
        {"max-resident-vram-mib", Kind::value},  {"max-resident-ram-mib", Kind::value},
        {"moe-experts", Kind::value},            {"tensor-override", Kind::repeat},
        {"batch-size", Kind::value},             {"ubatch-size", Kind::value},
        {"cache-type-k", Kind::value},           {"cache-type-v", Kind::value},
        {"allow-q4-k-cache", Kind::flag},
        {"flash-attn", Kind::value},             {"swa-kv-mode", Kind::value},
        {"threads", Kind::value},
        {"no-mmap", Kind::flag},                 {"no-kv-unified", Kind::flag},
        {"no-prefix-cache", Kind::flag},         {"startup-smoke-test", Kind::flag},
        {"slots", Kind::value},
        {"spec-type", Kind::value},              {"spec-draft-max", Kind::value},
        {"moe-cache-mib", Kind::value},          {"moe-cache-cpu-share", Kind::value},
        {"cache-ram-mib", Kind::value},          {"mmproj", Kind::value},
        {"profiles", Kind::value},
        {"profile", Kind::repeat},               {"all-profiles", Kind::flag},
        {"vram-budget-mib", Kind::value},        {"allow-overcommit", Kind::flag},
    };
    static const std::vector<std::pair<std::string, Kind>> kAll = [] {
        auto all = kSpec;
        for (const char* flag : kRequestPathFlags) all.emplace_back(flag, Kind::value);
        return all;
    }();
    return kAll;
}

bool parse_uint(const std::string& text, std::uint64_t max, std::uint64_t& out) {
    if (text.empty() || text.size() > 19 || text.find_first_not_of("0123456789") != std::string::npos) {
        return false;
    }
    std::uint64_t n = 0;
    for (const char c : text) {
        n = n * 10 + static_cast<std::uint64_t>(c - '0');
    }
    if (n > max) {
        return false;
    }
    out = n;
    return true;
}

std::string trim(std::string s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = s.find_last_not_of(" \t\r\n");
    return s.substr(first, last - first + 1);
}

}  // namespace

bool parse_serve_args(const std::vector<std::string>& args, ParsedServeArgs& out, std::string& error) {
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string arg = args[i];
        if (arg == "--help" || arg == "-h" || arg == "help") {
            out.help = true;
            continue;
        }
        if (arg.rfind("--", 0) != 0 || arg.size() == 2) {
            error = "unexpected argument: " + arg;
            return false;
        }
        std::string key = arg.substr(2);
        std::optional<std::string> inline_value;
        if (const auto eq = key.find('='); eq != std::string::npos) {
            inline_value = key.substr(eq + 1);
            key.resize(eq);
        }
        const auto it = std::find_if(spec().begin(), spec().end(), [&](const auto& s) { return s.first == key; });
        if (it == spec().end()) {
            // Same wording and did-you-mean suggestion as the other
            // sonder-infer commands (src/cli/cli_spec.hpp).
            std::vector<std::string_view> names;
            for (const auto& entry : spec()) {
                names.emplace_back(entry.first);
            }
            error = sonder::cli::unknown_option_message(key, names);
            return false;
        }
        if (it->second == Kind::flag) {
            if (inline_value) {
                error = "--" + key + " takes no value";
                return false;
            }
            out.flags.insert(key);
            continue;
        }
        std::string value;
        if (inline_value) {
            value = *inline_value;
        } else {
            if (i + 1 >= args.size()) {
                error = "missing value for --" + key;
                return false;
            }
            value = args[++i];
        }
        if (it->second == Kind::value && out.get(key)) {
            error = "--" + key + " given more than once";
            return false;
        }
        out.values.emplace_back(key, std::move(value));
    }
    return true;
}

int build_serve_options(const ParsedServeArgs& a, const std::shared_ptr<LockedLog>& log, ServeCliConfig& out) {
    const auto usage_error = [&](const std::string& message) {
        log->line("error: " + message + " (see `sonder-infer serve --help`)");
        return 2;
    };

    ServerOptions& o = out.options;
    const BackendEnvDefaults env = backend_env_defaults();
    o.host = a.get("host").value_or("127.0.0.1");
    std::uint64_t n = 0;
    if (auto v = a.get("port")) {
        if (!parse_uint(*v, 65535, n)) return usage_error("--port must be an integer from 0 to 65535");
        o.port = static_cast<std::uint16_t>(n);
    }
    // Launch profiles: one or more typed profiles naming the backend and served models.
    std::vector<LaunchProfile> profiles;
    std::vector<const LaunchProfile*> active_profiles;
    const std::vector<std::string> requested_profiles = a.all("profile");
    const bool all_profiles = a.flags.count("all-profiles") != 0;
    if (auto path = a.get("profiles")) {
        auto loaded = load_launch_profiles(*path);
        if (!loaded.ok()) return usage_error(loaded.status().message());
        profiles = std::move(loaded).value();
    }
    if (all_profiles && !requested_profiles.empty()) {
        return usage_error("--all-profiles cannot be combined with --profile");
    }
    if (!requested_profiles.empty()) {
        if (profiles.empty()) return usage_error("--profile needs --profiles FILE");
        std::set<std::string> seen;
        for (const std::string& name : requested_profiles) {
            if (!seen.insert(name).second) {
                return usage_error("duplicate --profile '" + name + "'");
            }
            const LaunchProfile* found = find_launch_profile(profiles, name);
            if (!found) return usage_error("no launch profile named '" + name + "' in --profiles");
            active_profiles.push_back(found);
        }
    } else if (all_profiles) {
        if (profiles.empty()) return usage_error("--all-profiles needs --profiles FILE");
        const std::string target_backend = a.get("backend").value_or(profiles.front().backend);
        for (const auto& candidate : profiles) {
            if (candidate.backend == target_backend) {
                active_profiles.push_back(&candidate);
            } else {
                out.profile_notes.push_back("--all-profiles: skipping launch profile '" + candidate.name +
                                            "' (backend " + candidate.backend + "; serving backend " + target_backend +
                                            ")");
            }
        }
        if (active_profiles.empty()) {
            return usage_error("--backend " + target_backend +
                               " conflicts with --all-profiles (no profile in --profiles uses backend " +
                               target_backend + ")");
        }
    } else if (!profiles.empty()) {
        return usage_error("--profiles needs --profile NAME or --all-profiles (the profile to serve)");
    } else if (a.get("vram-budget-mib") || a.flags.count("allow-overcommit") != 0) {
        return usage_error("--vram-budget-mib and --allow-overcommit apply to --profile / --all-profiles only");
    }
    const LaunchProfile* primary = active_profiles.empty() ? nullptr : active_profiles.front();
    const bool multi_profile = active_profiles.size() > 1;
    if (primary) {
        for (std::size_t i = 1; i < active_profiles.size(); ++i) {
            const LaunchProfile* secondary = active_profiles[i];
            if (secondary->backend != primary->backend) {
                return usage_error("cannot mix incompatible backends across --profile: '" + secondary->name +
                                   "' uses " + secondary->backend + " while '" + primary->name + "' uses " +
                                   primary->backend + " (serve each backend in a separate instance)");
            }
        }
        if (auto b = a.get("backend"); b && *b != primary->backend) {
            return usage_error("--backend " + *b + " conflicts with launch profile '" + primary->name + "' (backend " +
                               primary->backend + ")");
        }
        for (const auto& m : a.all("model")) {
            const bool is_active_name = std::any_of(active_profiles.begin(), active_profiles.end(),
                                                    [&](const LaunchProfile* p) { return p->name == m; });
            if (!is_active_name) {
                if (!multi_profile) {
                    return usage_error("--model " + m + " conflicts with launch profile '" + primary->name +
                                       "' (it serves the model as " + primary->name + ")");
                }
                return usage_error("--model " + m +
                                   " conflicts with active launch profiles (models are selected by --profile)");
            }
        }
        for (const char* flag : {"gpu-layers", "context-length", "llamaserver-arg", "moe-cache-mib",
                                 "moe-cache-cpu-share", "cache-ram-mib", "mmproj", "slots", "threads",
                                 "spec-type", "spec-draft-max"}) {
            if (a.get(flag)) {
                return usage_error(std::string("--") + flag + " conflicts with --profile (set the profile's typed fields)");
            }
        }
        if (a.flags.count("no-kv-unified") != 0) {
            return usage_error("--no-kv-unified conflicts with --profile (set the profile's typed fields)");
        }
        if (a.flags.count("startup-smoke-test") != 0) {
            return usage_error("--startup-smoke-test conflicts with --profile (set the profile's typed fields)");
        }
        if (a.get("llamaserver-url") && (!multi_profile || primary->backend != kLaunchProfileBackendLlamaServer)) {
            return usage_error("--llamaserver-url conflicts with --profile (set the profile's typed fields)");
        }
        for (const char* flag : kLlamaCppContextFlags) {
            if (a.get(flag)) {
                return usage_error(std::string("--") + flag + " conflicts with --profile (set the profile's typed fields)");
            }
        }
        // Only the direct llamacpp backend reads these; a llamaserver profile
        // would start without them and say nothing.
        if (primary->backend == kLaunchProfileBackendLlamaServer) {
            for (const char* flag : kLlamaCppPlacementFlags) {
                if (a.get(flag)) {
                    return usage_error(std::string("option --") + flag + " applies only to --backend llamacpp; for "
                                       "llamaserver launch profile '" + primary->name +
                                       "' put llama-server's own flag (-ot, --cpu-moe) in extra_args");
                }
            }
        }
    }
    o.backend.backend = primary ? primary->backend : a.get("backend").value_or(env.backend.value_or(""));
    if (o.backend.backend.empty()) {
        return usage_error("--backend is required (mock, remote-mock, ollama, ollama_cloud, llamacpp or llamaserver; or set SONDER_INFER_BACKEND)");
    }
    if (!active_profiles.empty()) {
        o.models.clear();
        o.models.reserve(active_profiles.size());
        for (const LaunchProfile* p : active_profiles) {
            o.models.push_back(p->name);
        }
    } else {
        o.models = a.all("model");
    }
    o.device = a.get("device").value_or("");
    if (o.models.empty() && env.model) {
        o.models.push_back(*env.model);
    }
    o.backend.ollama_url = a.get("ollama-url").value_or(env.ollama_url.value_or(""));
    o.backend.ollama_allow_remote = a.flags.count("ollama-allow-remote") != 0;
    if (auto path = a.get("llamaserver-config")) {
        if (auto st = load_llamaserver_config(*path, o.backend); !st.ok()) return usage_error(st.message());
    }
    if (auto path = a.get("ollama-cloud-config")) {
        if (o.backend.backend != "ollama_cloud") return usage_error("--ollama-cloud-config requires --backend ollama_cloud");
        if (auto st = load_ollama_cloud_config(*path, o.backend); !st.ok()) return usage_error(st.message());
        if (o.models.empty()) o.models = o.backend.ollama_cloud_models;
        // Security-relevant, like --ollama-allow-remote in the other commands.
        log->line("sonder-infer serve: warning: --backend ollama_cloud sends prompts to a remote provider (" +
                  (o.backend.ollama_cloud_base_url.empty() ? std::string("https://ollama.com")
                                                           : o.backend.ollama_cloud_base_url) +
                  "); X-Sonder-Local-Only requests are refused");
    } else if (o.backend.backend == "ollama_cloud") {
        return usage_error("--backend ollama_cloud requires --ollama-cloud-config PATH");
    }
    if (auto v = a.get("llamaserver-mode")) o.backend.llamaserver_mode = *v;
    if (auto v = a.get("llamaserver-url")) o.backend.llamaserver_url = *v;
    if (auto v = a.get("llamaserver-executable")) o.backend.llamaserver_executable = *v;
    if (!a.all("llamaserver-arg").empty()) o.backend.llamaserver_args = a.all("llamaserver-arg");
    if (primary && primary->backend == kLaunchProfileBackendLlamaServer) {
        // A profile's child is the one that serves its model, and in shared
        // spawn mode several profiles share it; the RPC client never runs
        // there (docs/integration/rpc-worker.md). Run the worker as its own
        // serve instance without --profile.
        if (o.backend.llamaserver_rpc_worker.enabled || !o.backend.llamaserver_rpc_worker.servers.empty()) {
            return usage_error("launch profile '" + primary->name +
                               "': llamaserver config 'rpc_worker' cannot be combined with --profile or "
                               "--all-profiles; run the RPC worker as a separate `sonder-infer serve` instance");
        }
        const auto check_profile_env = [&]() -> std::string {
            for (const auto& [env_key, env_val] : o.backend.llamaserver_environment) {
                (void)env_val;
                if (!llamaserver::is_allowed_profile_env_override(env_key)) {
                    return "launch profile '" + primary->name + "': llamaserver config env override '" + env_key +
                           "' is not permitted with --profile (only GPU/runtime tuning overrides such as "
                           "CUDA_*, GGML_*, VK_*, ROCR_*, ROCM_*, HSA_*, HIP_*, AMD_*, ROCBLAS_*, HIPBLASLT_*, "
                           "RADV_*, MESA_*, GPU_MAX_ALLOC_PERCENT, DRI_PRIME, OMP_* are allowed)";
                }
            }
            return {};
        };
        if (!multi_profile) {
            if (o.backend.llamaserver_mode.empty()) o.backend.llamaserver_mode = "spawn";
            if (o.backend.llamaserver_mode != "spawn") {
                return usage_error("launch profile '" + primary->name + "' needs llamaserver spawn mode (attach cannot "
                                   "apply its settings)");
            }
            if (!o.backend.llamaserver_args.empty()) {
                return usage_error("launch profile '" + primary->name + "': move the llamaserver config 'args' into the "
                                   "profile's typed fields or extra_args");
            }
            if (const std::string env_err = check_profile_env(); !env_err.empty()) return usage_error(env_err);
            auto argv = llamaserver_arguments(*primary);
            if (!argv.ok()) return usage_error(argv.status().message());
            o.backend.llamaserver_args = std::move(argv).value();
        } else {
            if (o.backend.llamaserver_mode.empty()) {
                o.backend.llamaserver_mode = !o.backend.llamaserver_url.empty() ? "attach" : "spawn";
            }
            if (o.backend.llamaserver_mode == "attach") {
                if (!o.backend.llamaserver_args.empty()) {
                    return usage_error("launch profiles in llamaserver attach mode cannot use llamaserver config 'args'");
                }
                if (const std::string env_err = check_profile_env(); !env_err.empty()) return usage_error(env_err);
            } else if (o.backend.llamaserver_mode == "spawn") {
                if (!o.backend.llamaserver_url.empty()) {
                    return usage_error(
                        std::string(a.get("llamaserver-url") ? "--llamaserver-url" : "llamaserver config 'base_url'") +
                        " conflicts with --profile in llamaserver spawn mode (use --llamaserver-mode attach to target "
                        "an upstream router)");
                }
                for (std::size_t i = 1; i < active_profiles.size(); ++i) {
                    const LaunchProfile* secondary = active_profiles[i];
                    if (secondary->model != primary->model) {
                        return usage_error(
                            "llamaserver spawn mode launches a single child process for one model and cannot spawn "
                            "multiple distinct GGUF models ('" + primary->name + "' uses '" + primary->model + "', '" +
                            secondary->name + "' uses '" + secondary->model +
                            "'); use --llamaserver-mode attach against an upstream router (with --llamaserver-url) "
                            "or run separate `sonder-infer serve` instances");
                    }
                }
                if (!o.backend.llamaserver_args.empty()) {
                    return usage_error("launch profile '" + primary->name + "': move the llamaserver config 'args' into "
                                       "the profile's typed fields or extra_args");
                }
                if (const std::string env_err = check_profile_env(); !env_err.empty()) return usage_error(env_err);
                auto argv = llamaserver_arguments(*primary);
                if (!argv.ok()) return usage_error(argv.status().message());
                for (std::size_t i = 1; i < active_profiles.size(); ++i) {
                    LaunchProfile normalized = *active_profiles[i];
                    normalized.name = primary->name;
                    auto secondary_argv = llamaserver_arguments(normalized);
                    if (!secondary_argv.ok()) return usage_error(secondary_argv.status().message());
                    if (secondary_argv.value() != argv.value()) {
                        return usage_error(
                            "llamaserver spawn mode launches one child process and cannot apply conflicting launch "
                            "settings for '" + active_profiles[i]->name + "' vs '" + primary->name +
                            "'; align launch fields (only name and sampling may differ in shared-model spawn mode), "
                            "use --llamaserver-mode attach, or run separate `sonder-infer serve` instances");
                    }
                }
                o.backend.llamaserver_args = std::move(argv).value();
            }
        }
    }
    if (auto v = a.get("warmup-file")) {
        if (o.backend.backend != "llamaserver")
            return usage_error("--warmup-file requires --backend llamaserver");
        o.backend.llamaserver_warmup_messages_file = *v;
    }
    if (o.backend.backend == "llamaserver") {
        if (o.backend.llamaserver_mode == "spawn") {
            if (o.backend.llamaserver_executable.empty()) {
                return usage_error("--llamaserver-executable is required in spawn mode");
            }
            for (const auto& arg : o.backend.llamaserver_args) {
                if (arg == "--host" || arg.rfind("--host=", 0) == 0 || arg == "--port" || arg.rfind("--port=", 0) == 0) {
                    return usage_error("llamaserver spawn args must not override --host or --port");
                }
            }
        } else if (!o.backend.llamaserver_mode.empty() && o.backend.llamaserver_mode != "attach") {
            return usage_error("--llamaserver-mode must be attach or spawn");
        }
    }
    o.backend.model_dirs = a.all("model-dir");
    if (auto v = a.get("gpu-layers")) {
        if (*v == "-1") {
            o.backend.llamacpp_gpu_layers = -1;
        } else if (!parse_uint(*v, 100000, n)) {
            return usage_error("--gpu-layers must be -1 or a non-negative integer");
        } else {
            o.backend.llamacpp_gpu_layers = static_cast<std::int32_t>(n);
        }
    }
    if (auto v = a.get("context-length")) {
        if (!parse_uint(*v, 1u << 22, n)) return usage_error("--context-length must be an integer from 0 to 4194304");
        o.backend.llamacpp_context_length = static_cast<std::uint32_t>(n);
    }
    if (auto v = a.get("moe-experts")) {
        if (*v == "cpu") {
            o.backend.llamacpp_tensor_overrides.push_back(std::string(kMoeExpertTensorOverridePattern) + "=cpu");
        } else if (*v != "gpu") {
            return usage_error("--moe-experts must be cpu or gpu");
        }
    }
    for (const std::string& spec_item : a.all("tensor-override")) {
        if (auto parsed = parse_tensor_override(spec_item); !parsed) return usage_error(parsed.status().message());
        o.backend.llamacpp_tensor_overrides.push_back(spec_item);
    }
    o.lazy_models = a.flags.count("lazy-models") != 0;
    if (auto v = a.get("model-idle-ttl")) {
        if (!parse_uint(*v, 30u * 24u * 3600u, n)) {
            return usage_error("--model-idle-ttl must be a number of seconds from 0 to 2592000");
        }
        o.model_idle_ttl = std::chrono::seconds(n);
    }
    if (auto v = a.get("max-resident-models")) {
        if (!parse_uint(*v, 4096, n)) return usage_error("--max-resident-models must be an integer from 0 to 4096");
        o.max_resident_models = static_cast<std::size_t>(n);
    }
    if (auto v = a.get("max-resident-vram-mib")) {
        if (!parse_uint(*v, 16777216, n)) {
            return usage_error("--max-resident-vram-mib must be an integer from 0 to 16777216");
        }
        o.max_resident_vram_bytes = n << 20;
    }
    if (auto v = a.get("max-resident-ram-mib")) {
        if (!parse_uint(*v, 16777216, n)) {
            return usage_error("--max-resident-ram-mib must be an integer from 0 to 16777216");
        }
        o.max_resident_ram_bytes = n << 20;
    }
    if (Status st = parse_request_path_flags([&a](const std::string& k) { return a.get(k); }, o); !st.ok()) {
        return usage_error(st.message());
    }
    // Only the direct llamacpp backend reads these. Any other backend would
    // run without them and say nothing; llama-server takes them in its spawn
    // arguments (the --llamaserver-config "args", or a launch profile).
    if (a.get("moe-cache-cpu-share")) {
        return usage_error("--moe-cache-cpu-share is not supported on serve CLI (no execution consumer in llamacpp); "
                           "set moe_cache_cpu_share on a llamaserver launch profile or JSON args");
    }
    if (o.backend.backend != "llamacpp") {
        for (const char* flag : {"gpu-layers", "context-length", "moe-experts", "tensor-override",
                                 "batch-size", "ubatch-size", "cache-type-k", "cache-type-v", "flash-attn",
                                 "swa-kv-mode", "threads", "slots", "spec-type", "spec-draft-max",
                                 "moe-cache-mib", "cache-ram-mib", "mmproj"}) {
            if (a.get(flag)) {
                return usage_error(std::string("option --") + flag +
                                   " applies only to --backend llamacpp; for llamaserver set it in the JSON args");
            }
        }
        for (const char* flag :
             {"no-mmap", "no-kv-unified", "no-prefix-cache", "startup-smoke-test", "allow-q4-k-cache"}) {
            if (a.flags.count(flag) != 0) {
                return usage_error(std::string("option --") + flag +
                                   " applies only to --backend llamacpp; for llamaserver set it in the JSON args");
            }
        }
    }
    if (auto v = a.get("batch-size")) {
        if (!parse_uint(*v, 1u << 22, n) || n == 0) return usage_error("--batch-size must be from 1 to 4194304");
        o.backend.llamacpp_batch_size = static_cast<std::uint32_t>(n);
    }
    if (auto v = a.get("ubatch-size")) {
        if (!parse_uint(*v, 1u << 22, n) || n == 0) return usage_error("--ubatch-size must be from 1 to 4194304");
        o.backend.llamacpp_ubatch_size = static_cast<std::uint32_t>(n);
    }
    // Values are checked by validate_options() (validate_llamacpp_options).
    if (auto v = a.get("cache-type-k")) o.backend.llamacpp_kv_cache_type_k = *v;
    if (auto v = a.get("cache-type-v")) o.backend.llamacpp_kv_cache_type_v = *v;
    if (a.flags.count("allow-q4-k-cache") != 0) o.backend.llamacpp_kv_k_q4_allowed = true;
    if (auto v = a.get("flash-attn")) o.backend.llamacpp_flash_attention = *v;
    if (auto v = a.get("swa-kv-mode")) {
        SwaKvMode parsed_swa = SwaKvMode::Full;
        if (!parse_swa_kv_mode(*v, parsed_swa)) {
            return usage_error("--swa-kv-mode must be full or compact_if_qualified");
        }
        o.backend.llamacpp_swa_kv_mode = parsed_swa;
    }
    if (auto v = a.get("threads")) {
        if (!parse_uint(*v, 256, n) || n == 0) return usage_error("--threads must be an integer from 1 to 256");
        o.backend.llamacpp_threads = static_cast<std::int32_t>(n);
    }
    if (a.flags.count("no-mmap") != 0) {
        o.backend.llamacpp_use_mmap = false;
    }
    if (a.flags.count("no-kv-unified") != 0) {
        o.backend.llamacpp_kv_unified = false;
    }
    if (a.flags.count("no-prefix-cache") != 0) {
        o.backend.llamacpp_prefix_cache = false;
    }
    if (a.flags.count("startup-smoke-test") != 0) {
        o.backend.llamacpp_startup_smoke_test = true;
    }
    if (auto v = a.get("slots")) {
        if (!parse_uint(*v, 16, n) || n == 0) return usage_error("--slots must be an integer from 1 to 16");
        o.backend.llamacpp_slots = static_cast<std::uint32_t>(n);
    }
    if (auto v = a.get("spec-type")) {
        if (*v != "none" && *v != "auto" && *v != "ngram" && *v != "ngram-mod" && *v != "suffix" &&
            *v != "draft-mtp") {
            return usage_error("--spec-type must be none, auto, ngram, ngram-mod, suffix or draft-mtp");
        }
        o.backend.llamacpp_spec_type = *v;
    }
    if (auto v = a.get("spec-draft-max")) {
        if (!parse_uint(*v, 64, n)) return usage_error("--spec-draft-max must be an integer from 0 to 64");
        o.backend.llamacpp_spec_draft_max = static_cast<std::int32_t>(n);
    }
    if (auto v = a.get("moe-cache-mib")) {
        if (!parse_uint(*v, 262144, n)) return usage_error("--moe-cache-mib must be an integer from 0 to 262144");
        o.backend.llamacpp_moe_cache_mib = static_cast<std::uint32_t>(n);
    }
    if (auto v = a.get("cache-ram-mib")) {
        if (!parse_uint(*v, 65536, n)) return usage_error("--cache-ram-mib must be an integer from 0 to 65536");
        o.backend.llamacpp_cache_ram_mib = static_cast<std::uint32_t>(n);
    }
    if (auto v = a.get("mmproj")) {
        if (v->empty() || v->find('\0') != std::string::npos || v->find('\r') != std::string::npos ||
            v->find('\n') != std::string::npos) {
            return usage_error("--mmproj must be a non-empty path without NUL or newline characters");
        }
        o.backend.llamacpp_mmproj_path = *v;
    }
    if (primary && primary->backend == kLaunchProfileBackendLlamaCpp) {
        const BackendSetup base_backend = o.backend;
        const std::string base_device = o.device;
        if (Status st = apply_llamacpp_profile(*primary, o.backend, o.device); !st.ok()) {
            return usage_error(st.message());
        }
        std::string owner_ctx = o.backend.llamacpp_context_length ? primary->name : std::string{};
        std::string owner_ngl = o.backend.llamacpp_gpu_layers ? primary->name : std::string{};
        std::string owner_batch = o.backend.llamacpp_batch_size ? primary->name : std::string{};
        std::string owner_ubatch = o.backend.llamacpp_ubatch_size ? primary->name : std::string{};
        std::string owner_k = o.backend.llamacpp_kv_cache_type_k ? primary->name : std::string{};
        std::string owner_v = o.backend.llamacpp_kv_cache_type_v ? primary->name : std::string{};
        std::string owner_fa = o.backend.llamacpp_flash_attention ? primary->name : std::string{};
        std::string owner_swa = o.backend.llamacpp_swa_kv_mode ? primary->name : std::string{};
        std::string owner_smoke = o.backend.llamacpp_startup_smoke_test ? primary->name : std::string{};
        std::string owner_slots = o.backend.llamacpp_slots ? primary->name : std::string{};
        std::string owner_kv_unified = o.backend.llamacpp_kv_unified ? primary->name : std::string{};
        std::string owner_cache_ram = o.backend.llamacpp_cache_ram_mib ? primary->name : std::string{};
        std::string owner_threads = o.backend.llamacpp_threads ? primary->name : std::string{};
        std::string owner_spec_type = o.backend.llamacpp_spec_type ? primary->name : std::string{};
        std::string owner_spec_max = o.backend.llamacpp_spec_draft_max ? primary->name : std::string{};
        std::string owner_mmproj = o.backend.llamacpp_mmproj_path ? primary->name : std::string{};
        const auto int_text = [](auto v) { return std::to_string(v); };
        const auto bool_text = [](bool b) -> std::string { return b ? "true" : "false"; };
        const auto str_text = [](const std::string& s) { return s; };
        const auto swa_text = [](SwaKvMode m) { return std::string(to_string(m)); };
        for (std::size_t i = 1; i < active_profiles.size(); ++i) {
            const LaunchProfile& secondary = *active_profiles[i];
            BackendSetup candidate_backend = base_backend;
            std::string candidate_device = base_device;
            if (Status st = apply_llamacpp_profile(secondary, candidate_backend, candidate_device); !st.ok()) {
                return usage_error(st.message());
            }
            const auto merge_opt = [&](const char* field_name, auto& shared_opt, const auto& next_opt,
                                       std::string& owner, auto to_text) -> std::string {
                if (!next_opt) {
                    if (shared_opt) {
                        out.profile_notes.push_back("launch profile '" + secondary.name +
                                                    "' inherits shared llamacpp option '" + field_name + "' = " +
                                                    to_text(*shared_opt) + " from '" + owner + "'");
                    }
                    return {};
                }
                if (shared_opt) {
                    if (*shared_opt != *next_opt) {
                        return "launch profile '" + secondary.name + "' conflicts with '" + owner +
                               "' on shared llamacpp option '" + field_name + "' (" + to_text(*next_opt) + " vs " +
                               to_text(*shared_opt) + "); align the profiles or run separate serve instances";
                    }
                    return {};
                }
                shared_opt = *next_opt;
                owner = secondary.name;
                std::string earlier;
                for (std::size_t k = 0; k < i; ++k) {
                    if (k > 0) earlier += ", ";
                    earlier += "'" + active_profiles[k]->name + "'";
                }
                out.profile_notes.push_back("launch profile '" + secondary.name + "' sets shared llamacpp option '" +
                                            field_name + "' = " + to_text(*next_opt) + " (unset on " + earlier + ")");
                return {};
            };
            if (const std::string err = merge_opt("ctx_size", o.backend.llamacpp_context_length,
                                                  candidate_backend.llamacpp_context_length, owner_ctx, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("n_gpu_layers", o.backend.llamacpp_gpu_layers,
                                                  candidate_backend.llamacpp_gpu_layers, owner_ngl, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("batch_size", o.backend.llamacpp_batch_size,
                                                  candidate_backend.llamacpp_batch_size, owner_batch, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("ubatch_size", o.backend.llamacpp_ubatch_size,
                                                  candidate_backend.llamacpp_ubatch_size, owner_ubatch, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("cache_type_k", o.backend.llamacpp_kv_cache_type_k,
                                                  candidate_backend.llamacpp_kv_cache_type_k, owner_k, str_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("cache_type_v", o.backend.llamacpp_kv_cache_type_v,
                                                  candidate_backend.llamacpp_kv_cache_type_v, owner_v, str_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("flash_attn", o.backend.llamacpp_flash_attention,
                                                  candidate_backend.llamacpp_flash_attention, owner_fa, str_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("swa_kv_mode", o.backend.llamacpp_swa_kv_mode,
                                                  candidate_backend.llamacpp_swa_kv_mode, owner_swa, swa_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("startup_smoke_test", o.backend.llamacpp_startup_smoke_test,
                                                  candidate_backend.llamacpp_startup_smoke_test, owner_smoke,
                                                  bool_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("parallel", o.backend.llamacpp_slots,
                                                  candidate_backend.llamacpp_slots, owner_slots, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("kv_unified", o.backend.llamacpp_kv_unified,
                                                  candidate_backend.llamacpp_kv_unified, owner_kv_unified, bool_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("cache_ram_mib", o.backend.llamacpp_cache_ram_mib,
                                                  candidate_backend.llamacpp_cache_ram_mib, owner_cache_ram, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("threads", o.backend.llamacpp_threads,
                                                  candidate_backend.llamacpp_threads, owner_threads, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("speculative.types", o.backend.llamacpp_spec_type,
                                                  candidate_backend.llamacpp_spec_type, owner_spec_type, str_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("speculative.draft_n_max", o.backend.llamacpp_spec_draft_max,
                                                  candidate_backend.llamacpp_spec_draft_max, owner_spec_max, int_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (const std::string err = merge_opt("mmproj", o.backend.llamacpp_mmproj_path,
                                                  candidate_backend.llamacpp_mmproj_path, owner_mmproj, str_text);
                !err.empty()) {
                return usage_error(err);
            }
            if (o.device.empty() && !candidate_device.empty()) {
                o.device = candidate_device;
            }
        }
        if (multi_profile) {
            const std::uint32_t effective_batch = o.backend.llamacpp_batch_size.value_or(512u);
            if (o.backend.llamacpp_ubatch_size && *o.backend.llamacpp_ubatch_size > effective_batch) {
                return usage_error("combined llamacpp launch profiles have ubatch_size (" +
                                   std::to_string(*o.backend.llamacpp_ubatch_size) + ") > batch_size (" +
                                   std::to_string(effective_batch) + ")");
            }
            if (o.backend.llamacpp_kv_cache_type_v &&
                is_quantized_kv_cache_type(*o.backend.llamacpp_kv_cache_type_v) &&
                o.backend.llamacpp_flash_attention && *o.backend.llamacpp_flash_attention == "off") {
                return usage_error("combined llamacpp launch profiles set quantized cache_type_v ('" +
                                   *o.backend.llamacpp_kv_cache_type_v + "') with flash_attn 'off'");
            }
            if (o.backend.llamacpp_spec_type && *o.backend.llamacpp_spec_type == "none" &&
                o.backend.llamacpp_spec_draft_max) {
                return usage_error(
                    "combined llamacpp launch profiles set speculative.draft_n_max alongside speculative type 'none'");
            }
            if (o.backend.llamacpp_gpu_layers && *o.backend.llamacpp_gpu_layers == 0) {
                for (const LaunchProfile* ap : active_profiles) {
                    if (ap != nullptr && !ap->devices.empty()) {
                        return usage_error("combined llamacpp launch profiles set 'devices' on '" + ap->name +
                                           "' alongside 'n_gpu_layers' 0 (CPU only)");
                    }
                }
            }
        }
        std::string resolved = "llamacpp resolved settings:";
        const auto add_resolved = [&](const char* k, const std::string& v) {
            resolved += " ";
            resolved += k;
            resolved += "=";
            resolved += v;
        };
        if (o.backend.llamacpp_context_length) add_resolved("ctx_size", std::to_string(*o.backend.llamacpp_context_length));
        if (o.backend.llamacpp_gpu_layers) add_resolved("n_gpu_layers", std::to_string(*o.backend.llamacpp_gpu_layers));
        if (o.backend.llamacpp_batch_size) add_resolved("batch_size", std::to_string(*o.backend.llamacpp_batch_size));
        if (o.backend.llamacpp_ubatch_size) add_resolved("ubatch_size", std::to_string(*o.backend.llamacpp_ubatch_size));
        if (o.backend.llamacpp_kv_cache_type_k) add_resolved("cache_type_k", *o.backend.llamacpp_kv_cache_type_k);
        if (o.backend.llamacpp_kv_cache_type_v) add_resolved("cache_type_v", *o.backend.llamacpp_kv_cache_type_v);
        if (o.backend.llamacpp_flash_attention) add_resolved("flash_attn", *o.backend.llamacpp_flash_attention);
        if (o.backend.llamacpp_slots) add_resolved("slots", std::to_string(*o.backend.llamacpp_slots));
        if (o.backend.llamacpp_kv_unified) add_resolved("kv_unified", *o.backend.llamacpp_kv_unified ? "true" : "false");
        if (o.backend.llamacpp_startup_smoke_test) {
            add_resolved("startup_smoke_test", *o.backend.llamacpp_startup_smoke_test ? "true" : "false");
        }
        if (o.backend.llamacpp_cache_ram_mib) add_resolved("cache_ram_mib", std::to_string(*o.backend.llamacpp_cache_ram_mib));
        if (o.backend.llamacpp_threads) add_resolved("threads", std::to_string(*o.backend.llamacpp_threads));
        if (o.backend.llamacpp_spec_type) add_resolved("spec_type", *o.backend.llamacpp_spec_type);
        if (o.backend.llamacpp_spec_draft_max) add_resolved("spec_draft_max", std::to_string(*o.backend.llamacpp_spec_draft_max));
        if (o.backend.llamacpp_mmproj_path) add_resolved("mmproj", *o.backend.llamacpp_mmproj_path);
        if (!o.device.empty()) add_resolved("device", o.device);
        out.profile_notes.push_back(std::move(resolved));
    }
    std::optional<std::uint64_t> budget_flag;
    if (auto v = a.get("vram-budget-mib")) {
        if (!parse_uint(*v, 1u << 24, n) || n == 0) return usage_error("--vram-budget-mib must be from 1 to 16777216");
        budget_flag = n;
    }
    if (!active_profiles.empty()) {
        std::string refusal;
        o.profile_catalog = profile_catalog(profiles, active_profiles, o, budget_flag,
                                            a.flags.count("allow-overcommit") != 0, out.profile_notes, refusal);
        if (!refusal.empty()) return usage_error(refusal);
    }
    if (auto v = a.get("mock-delay-ms")) {
        if (!parse_uint(*v, 60000, n)) return usage_error("--mock-delay-ms must be an integer from 0 to 60000");
        o.backend.mock_token_delay = std::chrono::milliseconds(n);
    }
    if (auto path = a.get("token-file")) {
        std::ifstream in(*path, std::ios::binary);
        if (!in) {
            return usage_error("cannot read --token-file " + *path);
        }
        std::ostringstream ss;
        ss << in.rdbuf();
        o.token = trim(ss.str());
        if (o.token.empty() || o.token.find_first_of(" \t\r\n") != std::string::npos) {
            return usage_error("--token-file must contain one non-empty token");
        }
#if !defined(_WIN32)
        struct stat st {};
        if (::stat(path->c_str(), &st) == 0 && (st.st_mode & 077) != 0) {
            log->line("sonder-infer serve: warning: --token-file " + *path +
                      " is readable by group or others; restrict it with chmod 600");
        }
#endif
    }
    o.cors_origins = a.all("cors-origin");
    o.default_cors = a.flags.count("no-default-cors") == 0;
    if (auto v = a.get("telemetry-level")) {
        const auto level = parse_telemetry_level(*v);
        if (!level) return usage_error("--telemetry-level must be off, metrics, standard or deep");
        o.telemetry_level = *level;
    }
    if (auto v = a.get("telemetry-buffer")) {
        if (!parse_uint(*v, 1u << 20, n) || n == 0) return usage_error("--telemetry-buffer must be from 1 to 1048576");
        o.telemetry_buffer = static_cast<std::size_t>(n);
    }
    o.capture_text = a.flags.count("capture-text") != 0;
    {
        // Precedence: --text-capture, then SONDER_INFER_TEXT_CAPTURE, then off.
        // A typo is an error: it must never silently enable or disable capture.
        std::optional<std::string> raw = a.get("text-capture");
        const char* from = "--text-capture";
        if (!raw) {
            if (const char* env_value = std::getenv("SONDER_INFER_TEXT_CAPTURE");
                env_value != nullptr && *env_value != '\0') {
                raw = std::string(env_value);
                from = "SONDER_INFER_TEXT_CAPTURE";
            }
        }
        if (raw) {
            const auto parsed = parse_text_capture_policy(*raw);
            if (!parsed) {
                return usage_error(std::string(from) + " must be off, on or full");
            }
            o.text_capture = *parsed;
        }
        if (o.capture_text && a.get("text-capture") && o.text_capture == TextCapturePolicy::off) {
            return usage_error("--capture-text conflicts with --text-capture off");
        }
    }
    if (auto v = a.get("max-connections")) {
        if (!parse_uint(*v, 100000, n) || n == 0) return usage_error("--max-connections must be from 1 to 100000");
        o.max_connections = static_cast<std::size_t>(n);
    }
    if (auto v = a.get("max-body-bytes")) {
        if (!parse_uint(*v, 1ull << 30, n) || n == 0) {
            return usage_error("--max-body-bytes must be from 1 to 1073741824");
        }
        o.max_body_bytes = static_cast<std::size_t>(n);
    }
    if (auto v = a.get("shutdown-grace-ms")) {
        if (!parse_uint(*v, 600000, n)) return usage_error("--shutdown-grace-ms must be from 0 to 600000");
        o.shutdown_grace = std::chrono::milliseconds(n);
    }
    if (auto v = a.get("stream-keepalive-ms")) {
        if (!parse_uint(*v, 600000, n)) return usage_error("--stream-keepalive-ms must be from 0 to 600000");
        o.stream_keepalive_interval = std::chrono::milliseconds(n);
    }
    o.disable_response_store = a.flags.count("disable-response-store") != 0;
    if (auto v = a.get("response-history-ttl")) {
        if (!parse_uint(*v, 24u * 3600u, n)) {
            return usage_error("--response-history-ttl must be an integer from 0 to 86400");
        }
        o.response_history_ttl = std::chrono::seconds(n);
    }
    if (auto v = a.get("log-format")) {
        if (*v != "text" && *v != "json") return usage_error("--log-format must be text or json");
        o.log_format = *v == "json" ? LogFormat::json : LogFormat::text;
    }
    if (auto path = a.get("telemetry")) {
        if (*path == "-") {
            o.extra_sinks.push_back(std::make_shared<LockedLogSink>(log));
        } else {
            Status st;
            auto sink = make_jsonl_file_sink(*path, false, &st);
            if (!sink) {
                log->line("error: " + st.message());
                return 1;
            }
            o.extra_sinks.push_back(std::shared_ptr<TelemetrySink>(std::move(sink)));
        }
    }
    if (Status st = validate_options(o); !st.ok()) {
        return usage_error(st.message());
    }
    out.ready_path = a.get("ready-file").value_or("");
    return 0;
}

}  // namespace sonder::inference::server::detail
