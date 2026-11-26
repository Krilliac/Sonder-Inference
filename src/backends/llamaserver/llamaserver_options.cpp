#include "llamaserver_options.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>

#include "net/http_client.hpp"

namespace sonder::inference::llamaserver {

Status parse_uma_mode_json(const json::Value &value, std::string &out) {
    if (value.is_bool()) {
        out = value.as_bool() ? "on" : "off";
        return {};
    }
    if (value.is_string()) {
        if (const auto normalized = sonder::inference::normalize_uma_mode(value.as_string())) {
            out.assign(*normalized);
            return {};
        }
    }
    return Status(ErrorCode::invalid_argument,
                  "llamaserver: uma_mode must be boolean or \"auto\", \"on\", \"off\", \"true\", \"false\"");
}

SpillGuardOptions to_spill_guard_options(const LlamaServerSpillGuardOptions &o,
                                         const std::vector<std::string> &args,
                                         const std::vector<GpuDeviceInfo> &gpus) {
    SpillGuardOptions g;
    g.enabled = o.enabled;
    // An out-of-range enum value stays invalid and is rejected by validate_spill_guard.
    g.policy = static_cast<SpillPolicy>(-1);
    switch (o.policy) {
    case LlamaServerSpillPolicy::warn:
        g.policy = SpillPolicy::warn;
        break;
    case LlamaServerSpillPolicy::refuse:
        g.policy = SpillPolicy::refuse;
        break;
    case LlamaServerSpillPolicy::auto_fit:
        g.policy = SpillPolicy::auto_fit;
        break;
    }
    g.threshold_bytes = o.threshold_bytes;
    g.baseline_bytes = o.baseline_bytes;
    g.baseline_bytes_per_1k_ctx = o.baseline_bytes_per_1k_ctx;
    g.sample_interval = o.sample_interval;
    g.step_factor = o.fit_step_factor;
    g.step_align = o.fit_step_align;
    g.min_ctx = o.fit_min_ctx;
    g.max_attempts = o.fit_max_attempts;
    g.residency = o.residency;
    if (!o.uma_mode_policy.empty()) {
        const auto selection = resolve_uma_selection(o.uma_mode_policy, gpus, extract_device_hint(args));
        g.uma_mode = selection != UmaSelectionMode::discrete_only;
        g.hybrid_uma_mode = selection == UmaSelectionMode::hybrid;
    } else {
        g.uma_mode = o.uma_mode;
        g.hybrid_uma_mode = false;
    }
    if (!o.residency_uma_mode_explicit && g.uma_mode) {
        g.residency.uma_mode = true;
    }
    return g;
}

SpillGuardOptions to_spill_guard_options(const LlamaServerSpillGuardOptions &o,
                                         const std::vector<std::string> &args) {
    if (o.uma_mode_policy == "auto" && (o.enabled || o.residency.enabled)) {
        const auto gpus = enumerate_gpu_devices();
        return to_spill_guard_options(o, args, gpus);
    }
    return to_spill_guard_options(o, args, {});
}

SpillGuardOptions to_spill_guard_options(const LlamaServerSpillGuardOptions &o) {
    return to_spill_guard_options(o, {});
}

bool to_host_memory_options(const LlamaServerHostMemoryOptions &o, HostMemoryOptions &out) {
    out.admission.estimated_resident_bytes = o.estimated_resident_bytes;
    out.admission.estimated_commit_bytes = o.estimated_private_bytes;
    out.admission.estimate_lower_bound = o.estimate_lower_bound;
    out.admission.reserve_bytes = o.reserve_bytes;
    out.admission.override_refusal = o.override_refusal;
    out.floor.enabled = o.floor.enabled;
    out.floor.floor_bytes = o.floor.floor_bytes;
    out.floor.resume_bytes = o.floor.resume_bytes;
    out.floor.consecutive_samples = o.floor.consecutive_samples;
    out.floor.sample_interval = o.floor.sample_interval;
    out.floor.restart_cooldown = o.floor.restart_cooldown;
    out.floor.max_restarts = o.floor.max_restarts;
    if (o.floor.policy == "warn")
        out.floor.policy = HostFloorPolicy::warn;
    else if (o.floor.policy == "stop_admission")
        out.floor.policy = HostFloorPolicy::stop_admission;
    else if (o.floor.policy == "restart_smaller_ctx")
        out.floor.policy = HostFloorPolicy::restart_smaller_ctx;
    else
        return false;
    return true;
}

AdaptiveSpeculationConfig to_adaptive_speculation_config(const LlamaServerAdaptiveSpeculationOptions &o) noexcept {
    AdaptiveSpeculationConfig cfg;
    cfg.enabled = o.enabled;
    cfg.min_draft_n = o.min_draft_n;
    cfg.initial_draft_n = o.initial_draft_n;
    cfg.max_draft_n = o.max_draft_n;
    cfg.window_size = o.window_size;
    cfg.min_samples = o.min_samples;
    cfg.pause_acceptance_threshold = o.pause_acceptance_threshold;
    cfg.step_down_acceptance_threshold = o.step_down_acceptance_threshold;
    cfg.step_up_acceptance_threshold = o.step_up_acceptance_threshold;
    cfg.cooldown_requests = o.cooldown_requests;
    cfg.probe_draft_n = o.probe_draft_n;
    cfg.min_draft_tokens_per_sample = o.min_draft_tokens_per_sample;
    cfg.max_entries = o.max_entries;
    cfg.min_baseline_samples = o.min_baseline_samples;
    cfg.cost_step_down_ratio = o.cost_step_down_ratio;
    cfg.cost_pause_ratio = o.cost_pause_ratio;
    cfg.cost_probe_recover_ratio = o.cost_probe_recover_ratio;
    cfg.cost_step_cooldown = o.cost_step_cooldown;
    cfg.baseline_probe_interval = o.baseline_probe_interval;
    return cfg;
}

Result<std::string> read_api_key(const std::string &name) {
    const bool name_ok = !name.empty() && name.size() <= 128 &&
                         !std::isdigit(static_cast<unsigned char>(name.front())) &&
                         std::all_of(name.begin(), name.end(), [](unsigned char c) {
                             return std::isalnum(c) != 0 || c == '_';
                         });
    if (!name_ok)
        return Status(ErrorCode::invalid_argument, "llamaserver: api_key_env is not a valid variable name");
    std::string value;
#if defined(_WIN32)
    char *buffer = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&buffer, &length, name.c_str()) == 0 && buffer != nullptr) {
        value.assign(buffer);
        std::free(buffer);
    }
#else
    if (const char *raw = std::getenv(name.c_str()))
        value.assign(raw);
#endif
    if (value.empty())
        return Status(ErrorCode::invalid_argument, "llamaserver: environment variable " + name + " is not set");
    if (value.size() > 4096 ||
        std::any_of(value.begin(), value.end(), [](unsigned char c) { return c <= 0x20 || c >= 0x7f; }))
        return Status(ErrorCode::invalid_argument,
                      "llamaserver: environment variable " + name + " does not hold a valid bearer token");
    return value;
}

Status validate_backend_options(const LlamaServerBackendOptions &options) {
    const auto duration_ok = [](std::chrono::milliseconds ms) {
        return ms.count() > 0 && ms <= std::chrono::hours(24);
    };
    for (const auto value : {options.connect_timeout, options.request_timeout, options.startup_timeout,
                             options.poll_interval, options.shutdown_timeout, options.restart_backoff,
                             options.max_restart_backoff, options.tls.handshake_timeout}) {
        if (!duration_ok(value))
            return {ErrorCode::invalid_argument, "llamaserver: timeouts must be in (0, 24h]"};
    }
    if (options.max_restarts > 1000 || options.restart_backoff > options.max_restart_backoff) {
        return {ErrorCode::invalid_argument, "llamaserver: invalid restart policy"};
    }
    if (options.context_length > (std::uint64_t{1} << 32))
        return {ErrorCode::invalid_argument, "llamaserver: context_length must be at most 4294967296"};
    const bool generic_progress = options.stall_guard.generic_progress;
    if (options.stall_guard.stall_seconds < std::chrono::seconds(1) ||
        options.stall_guard.stall_seconds > std::chrono::hours(24) ||
        (options.stall_guard.policy != "warn" && options.stall_guard.policy != "restart" &&
         !(generic_progress && options.stall_guard.policy == "cancel")))
        return {ErrorCode::invalid_argument, "llamaserver: invalid stall_guard options"};
    if (generic_progress && (options.mode != LlamaServerMode::attach || options.native_completion))
        return {ErrorCode::invalid_argument,
                "llamaserver: stall_guard.generic_progress needs attach mode with native_completion=false"};
    if (generic_progress && !options.stall_guard.enabled)
        return {ErrorCode::invalid_argument, "llamaserver: stall_guard.generic_progress needs stall_guard.enabled"};
    if (generic_progress && options.stall_guard.policy == "restart")
        return {ErrorCode::invalid_argument, "llamaserver: stall_guard.generic_progress policy is warn or cancel"};
    if (!generic_progress && options.stall_guard.policy == "cancel")
        return {ErrorCode::invalid_argument, "llamaserver: stall_guard policy cancel needs generic_progress"};
    if (!options.upstream_grammar && !options.grammar.empty())
        return {ErrorCode::invalid_argument,
                "llamaserver: grammar is configured but the upstream declares no grammar support"};
    if (options.adaptive_speculation.enabled) {
        if (!options.native_completion) {
            return {ErrorCode::invalid_argument,
                    "llamaserver: adaptive_speculation requires native_completion=true"};
        }
        if (auto st = validate_adaptive_speculation_config(
                to_adaptive_speculation_config(options.adaptive_speculation));
            !st.ok()) {
            return st;
        }
    }
    const auto slow_enabled = options.idle_timeout.count() != 0 || options.first_content_timeout.count() != 0 ||
                              options.unlimited_request_timeout || !options.api_key_env.empty();
    if (slow_enabled && options.mode != LlamaServerMode::attach)
        return {ErrorCode::invalid_argument,
                "llamaserver: api_key_env, idle_timeout, first_content_timeout and unlimited_request_timeout "
                "are attach-mode options"};
    for (const auto value : {options.idle_timeout, options.first_content_timeout}) {
        if (value.count() < 0 || value > std::chrono::hours(24))
            return {ErrorCode::invalid_argument, "llamaserver: idle/first-content timeouts must be in [0, 24h]"};
    }
    if (options.keepalive_gap.count() <= 0 || options.keepalive_gap > std::chrono::hours(1))
        return {ErrorCode::invalid_argument, "llamaserver: keepalive_gap must be in (0, 1h]"};
    // The upstream sends an empty keepalive delta every keepalive_gap. An idle
    // bound at or below that gap would cut off a healthy, merely slow stream.
    if (options.idle_timeout.count() != 0 && options.idle_timeout <= options.keepalive_gap)
        return {ErrorCode::invalid_argument,
                "llamaserver: idle_timeout must exceed keepalive_gap (the upstream's keepalive interval, 10 s by "
                "default), otherwise healthy keepalive-only periods would time out"};
    if (options.unlimited_request_timeout && options.first_content_timeout.count() == 0)
        return {ErrorCode::invalid_argument,
                "llamaserver: unlimited_request_timeout needs first_content_timeout (a request that is never "
                "accepted would otherwise wait forever)"};
    if (options.unlimited_request_timeout && options.idle_timeout.count() == 0)
        return {ErrorCode::invalid_argument,
                "llamaserver: unlimited_request_timeout needs idle_timeout (a stream that goes silent after its "
                "first content would otherwise never end)"};
    if (!options.api_key_env.empty()) {
        if (!options.warmup.messages_file.empty())
            return {ErrorCode::invalid_argument, "llamaserver: api_key_env is not supported together with warmup"};
        if (auto key = read_api_key(options.api_key_env); !key.ok())
            return key.status();
    }
    if (options.mode == LlamaServerMode::spawn) {
        if (options.executable.empty() || options.executable.find('\0') != std::string::npos) {
            return {ErrorCode::invalid_argument, "llamaserver: spawn needs an executable path"};
        }
        if (options.diagnostics.log_file.find('\0') != std::string::npos)
            return {ErrorCode::invalid_argument, "llamaserver: log file path contains NUL"};
        if (!options.spill_guard.uma_mode_policy.empty() && !is_valid_uma_mode(options.spill_guard.uma_mode_policy))
            return {ErrorCode::invalid_argument,
                    "llamaserver: spill_guard.uma_mode_policy must be on, off, auto, true or false"};
        if (options.spill_guard.enabled) {
            if (auto st = validate_spill_guard(to_spill_guard_options(options.spill_guard, options.args, {})); !st.ok())
                return st;
            if (options.spill_guard.policy == LlamaServerSpillPolicy::auto_fit && !find_context_size(options.args))
                return {ErrorCode::invalid_argument,
                        "llamaserver: spill_guard.policy=auto_fit needs an explicit --ctx-size/-c argument"};
        }
        {
            HostMemoryOptions host;
            if (!to_host_memory_options(options.host_memory, host))
                return {ErrorCode::invalid_argument,
                        "llamaserver: host_memory.floor.policy must be warn, stop_admission or restart_smaller_ctx"};
            if (auto st = validate_host_memory_options(host); !st.ok())
                return st;
        }
        if (auto st = validate_process_arguments(options.args); !st.ok())
            return st;
        // Checked here as well as at spawn, so an AIP_* or malformed override is
        // refused when the backend is built rather than when the child is first spawned.
        return validate_process_environment(options.environment);
    }
    if (options.mode != LlamaServerMode::attach)
        return {ErrorCode::invalid_argument, "llamaserver: invalid mode"};
    auto url = net::parse_url(options.base_url);
    if (!url.ok())
        return url.status();
    if (!net::is_loopback_host(url->host) && (!options.allow_remote || url->scheme != "https")) {
        return {ErrorCode::invalid_argument, "llamaserver: remote upstream requires allow_remote and HTTPS"};
    }
    return {};
}

}  // namespace sonder::inference::llamaserver
