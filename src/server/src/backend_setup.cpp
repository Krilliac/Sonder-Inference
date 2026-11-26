#include "sonder/inference/backend_setup.hpp"
#include "llamaserver_config_extensions.hpp"
#include "config_text.hpp"

#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>

#include "sonder/inference/backends.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/json.hpp"
#include "warmup_config.hpp"
#include "residency_config.hpp"
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
#include "sonder/inference/backends/llamaserver.hpp"
#endif
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
#include "sonder/inference/backends/llamacpp.hpp"
#endif

namespace sonder::inference {

namespace {
std::optional<std::string> non_empty(std::optional<std::string> v) {
    if (v && v->empty()) {
        return std::nullopt;
    }
    return v;
}

#if defined(SONDER_HAS_LLAMACPP_BACKEND)
Result<LlamaCppBackendOptions> llamacpp_options(const BackendSetup& setup) {
    LlamaCppBackendOptions lo;
    lo.model_dirs = setup.model_dirs;
    if (setup.llamacpp_gpu_layers) lo.gpu_layers = *setup.llamacpp_gpu_layers;
    if (setup.llamacpp_context_length) lo.context_length = *setup.llamacpp_context_length;
    if (setup.llamacpp_batch_size) lo.batch_size = *setup.llamacpp_batch_size;
    if (setup.llamacpp_ubatch_size) lo.ubatch_size = *setup.llamacpp_ubatch_size;
    if (setup.llamacpp_kv_cache_type_k) lo.kv_cache_type_k = *setup.llamacpp_kv_cache_type_k;
    if (setup.llamacpp_kv_cache_type_v) lo.kv_cache_type_v = *setup.llamacpp_kv_cache_type_v;
    if (setup.llamacpp_flash_attention) lo.flash_attention = *setup.llamacpp_flash_attention;
    if (setup.llamacpp_threads) lo.threads = *setup.llamacpp_threads;
    if (setup.llamacpp_use_mmap) lo.use_mmap = *setup.llamacpp_use_mmap;
    if (setup.llamacpp_kv_unified) lo.kv_unified = *setup.llamacpp_kv_unified;
    if (setup.llamacpp_prefix_cache) lo.prefix_cache = *setup.llamacpp_prefix_cache;
    if (setup.llamacpp_slots) lo.slots = *setup.llamacpp_slots;
    if (setup.llamacpp_spec_type) lo.spec_type = *setup.llamacpp_spec_type;
    if (setup.llamacpp_spec_draft_max) lo.spec_draft_max = *setup.llamacpp_spec_draft_max;
    if (setup.llamacpp_moe_cache_cpu_share.has_value()) {
        return Status(ErrorCode::invalid_argument,
                      "'llamacpp_moe_cache_cpu_share' is not supported by the llamacpp backend "
                      "(no execution consumer); use llamaserver");
    }
    // Guarded C++20 projection for extended llamacpp backend fields so server
    // compiles against both current and Track 1-upgraded LlamaCppBackendOptions.
    const auto wire_extended = [&setup](auto& dst) {
        (void)setup;
        if constexpr (requires { dst.moe_cache_mib; }) {
            if (setup.llamacpp_moe_cache_mib) dst.moe_cache_mib = *setup.llamacpp_moe_cache_mib;
        }
        if constexpr (requires { dst.cache_ram_mib; }) {
            if (setup.llamacpp_cache_ram_mib) dst.cache_ram_mib = *setup.llamacpp_cache_ram_mib;
        }
        if constexpr (requires { dst.mmproj_path; }) {
            if (setup.llamacpp_mmproj_path) dst.mmproj_path = *setup.llamacpp_mmproj_path;
        }
    };
    wire_extended(lo);
    for (const std::string& spec : setup.llamacpp_tensor_overrides) {
        auto parsed = parse_tensor_override(spec);
        if (!parsed) return parsed.status();
        lo.tensor_overrides.push_back(std::move(parsed).value());
    }
    if (Status st = validate_llamacpp_options(lo); !st.ok()) return st;
    return lo;
}
#endif
}  // namespace

BackendEnvDefaults backend_env_defaults(const EnvLookup& lookup) {
    const auto get = [&lookup](const char* name) -> std::optional<std::string> {
        if (lookup) {
            return non_empty(lookup(name));
        }
        const char* v = std::getenv(name);
        return v == nullptr ? std::nullopt : non_empty(std::string(v));
    };
    BackendEnvDefaults d;
    d.backend = get("SONDER_INFER_BACKEND");
    d.model = get("SONDER_INFER_MODEL");
    d.ollama_url = get("SONDER_OLLAMA_URL");
    if (!d.ollama_url) {
        if (auto host = get("OLLAMA_HOST")) {
            d.ollama_url = normalize_ollama_host(*host);
        }
    }
    return d;
}

std::string normalize_ollama_host(std::string_view value) {
    std::string scheme = "http";
    std::string_view rest = value;
    if (const auto p = rest.find("://"); p != std::string_view::npos) {
        scheme = std::string(rest.substr(0, p));
        rest = rest.substr(p + 3);
    }
    std::string path;
    if (const auto slash = rest.find('/'); slash != std::string_view::npos) {
        path = std::string(rest.substr(slash));
        rest = rest.substr(0, slash);
    }
    std::string host;
    std::string port;
    if (!rest.empty() && rest.front() == '[') {
        const auto close = rest.find(']');
        host = std::string(rest.substr(0, close == std::string_view::npos ? rest.size() : close + 1));
        if (close != std::string_view::npos && close + 1 < rest.size() && rest[close + 1] == ':') {
            port = std::string(rest.substr(close + 2));
        }
    } else {
        const auto colon = rest.rfind(':');
        host = std::string(rest.substr(0, colon));
        if (colon != std::string_view::npos) {
            port = std::string(rest.substr(colon + 1));
        }
    }
    if (host.empty() || host == "0.0.0.0" || host == "[::]") {
        host = "127.0.0.1";
    }
    if (port.empty()) {
        port = scheme == "https" ? "443" : "11434";
    }
    while (path.size() > 1 && path.back() == '/') {
        path.pop_back();
    }
    if (path == "/") {
        path.clear();
    }
    return scheme + "://" + host + ":" + port + path;
}

Result<std::pair<std::string, std::string>> parse_tensor_override(std::string_view spec) {
    const std::size_t eq = spec.rfind('=');
    if (eq == std::string_view::npos || eq == 0 || eq + 1 >= spec.size()) {
        return Status(ErrorCode::invalid_argument,
                      "tensor override must be PATTERN=DEVICE (e.g. ffn_.*_exps=cpu): " + std::string(spec));
    }
    return std::make_pair(std::string(spec.substr(0, eq)), std::string(spec.substr(eq + 1)));
}

namespace {

Status parse_host_memory_config(const json::Value& value, BackendSetup::LlamaServerHostMemoryConfig& out) {
    const auto bad = [](const std::string& key, const char* range) {
        return Status(ErrorCode::invalid_argument, "llamaserver config: host_memory." + key + " must be " + range);
    };
    if (!value.is_object()) return Status(ErrorCode::invalid_argument, "llamaserver config: 'host_memory' must be an object");
    static constexpr std::string_view keys[] = {"reserve_mib", "override_refusal", "floor"};
    for (const auto& member : value.as_object()) {
        if (std::find(std::begin(keys), std::end(keys), member.first) == std::end(keys))
            return Status(ErrorCode::invalid_argument, "llamaserver config: unknown host_memory field '" +
                                                           detail::bounded_for_message(member.first) + "'");
    }
    if (const auto* v = value.find("reserve_mib")) {
        if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() > 16777216) return bad("reserve_mib", "an integer in [0, 16777216]");
        out.reserve_mib = v->as_uint();
    }
    if (const auto* v = value.find("override_refusal")) {
        if (!v->is_bool()) return bad("override_refusal", "boolean");
        out.override_refusal = v->as_bool();
    }
    const auto* floor = value.find("floor");
    if (!floor) return {};
    if (!floor->is_object()) return Status(ErrorCode::invalid_argument, "llamaserver config: 'host_memory.floor' must be an object");
    static constexpr std::string_view floor_keys[] = {"enabled", "policy", "floor_mib", "resume_mib", "consecutive_samples",
                                                      "sample_interval_ms", "restart_cooldown_ms", "max_restarts"};
    for (const auto& member : floor->as_object()) {
        if (std::find(std::begin(floor_keys), std::end(floor_keys), member.first) == std::end(floor_keys))
            return Status(ErrorCode::invalid_argument, "llamaserver config: unknown host_memory.floor field '" +
                                                           detail::bounded_for_message(member.first) + "'");
    }
    const auto uint_in = [&](const char* key, std::uint64_t lo, std::uint64_t hi, std::uint64_t& dest, const char* range) -> Status {
        if (const auto* v = floor->find(key)) {
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() < lo || v->as_uint() > hi) return bad(std::string("floor.") + key, range);
            dest = v->as_uint();
        }
        return {};
    };
    if (const auto* v = floor->find("enabled")) {
        if (!v->is_bool()) return bad("floor.enabled", "boolean");
        out.floor_enabled = v->as_bool();
    }
    if (const auto* v = floor->find("policy")) {
        if (!v->is_string() || (v->as_string() != "warn" && v->as_string() != "stop_admission" && v->as_string() != "restart_smaller_ctx"))
            return bad("floor.policy", "warn, stop_admission or restart_smaller_ctx");
        out.floor_policy = v->as_string();
    }
    if (auto st = uint_in("floor_mib", 0, 16777216, out.floor_mib, "an integer in [0, 16777216]"); !st.ok()) return st;
    if (auto st = uint_in("resume_mib", 0, 16777216, out.resume_mib, "an integer in [0, 16777216]"); !st.ok()) return st;
    if (auto st = uint_in("consecutive_samples", 1, 100, out.consecutive_samples, "an integer in [1, 100]"); !st.ok()) return st;
    if (auto st = uint_in("sample_interval_ms", 10, 3600000, out.sample_interval_ms, "an integer in [10, 3600000]"); !st.ok()) return st;
    if (auto st = uint_in("restart_cooldown_ms", 0, 86400000, out.restart_cooldown_ms, "an integer in [0, 86400000]"); !st.ok()) return st;
    if (auto st = uint_in("max_restarts", 0, 1000, out.max_restarts, "an integer in [0, 1000]"); !st.ok()) return st;
    if (out.floor_enabled && out.floor_mib == 0) return bad("floor.floor_mib", "above 0 when the floor guard is enabled");
    if (out.resume_mib != 0 && out.resume_mib < out.floor_mib) return bad("floor.resume_mib", "0 or at least floor_mib");
    return {};
}

}  // namespace

Status load_llamaserver_config(const std::string& path, BackendSetup& destination) {
    BackendSetup setup = destination;
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status(ErrorCode::not_found, "cannot read llamaserver config " + path);
    std::string text(1024 * 1024 + 1, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (in.bad()) return Status(ErrorCode::io_error, "cannot read llamaserver config");
    if (in.gcount() > 1024 * 1024)
        return Status(ErrorCode::invalid_argument, "llamaserver config exceeds 1 MiB");
    text.resize(static_cast<std::size_t>(in.gcount()));
    auto parsed = json::parse(text);
    if (!parsed.ok()) return Status(ErrorCode::invalid_argument, "llamaserver config: " + parsed.status().message());
    if (!parsed.value().is_object()) return Status(ErrorCode::invalid_argument, "llamaserver config must be a JSON object");
    const auto& object = parsed.value().as_object();
    const auto unknown = [&](std::string_view key) {
        static constexpr std::string_view known[] = {"mode", "base_url", "executable", "args", "allow_remote",
            "native_completion", "grammar", "connect_timeout_ms", "request_timeout_ms", "startup_timeout_ms",
            "poll_interval_ms", "shutdown_timeout_ms", "restart_backoff_ms", "max_restart_backoff_ms",
            "max_restarts", "tls", "spill_guard", "log_file", "kv_pairing_check", "context_length",
            "slot_affinity", "env", "results", "warmup", "stall_guard", "host_memory",
            "api_key_env", "idle_timeout_ms", "first_content_timeout_ms", "keepalive_gap_ms",
            "unlimited_request_timeout", "upstream_grammar", "adaptive_speculation"};
        for (auto k : known) if (k == key) return false;
        return true;
    };
    for (const auto& member : object) {
        if (unknown(member.first)) return Status(ErrorCode::invalid_argument, "llamaserver config: unknown field '" + detail::bounded_for_message(member.first) + "'");
    }
    if (auto st = load_llamaserver_extensions(object, setup.llamaserver_environment); !st.ok()) return st;
    const auto string_field = [&](const char* key, std::string& out) -> Status {
        if (const auto* v = object.find(key)) {
            if (!v->is_string()) return Status(ErrorCode::invalid_argument, std::string("llamaserver config: '") + key + "' must be a string");
            out = v->as_string();
        }
        return {};
    };
    const auto bool_field = [&](const char* key, bool& out) -> Status {
        if (const auto* v = object.find(key)) {
            if (!v->is_bool()) return Status(ErrorCode::invalid_argument, std::string("llamaserver config: '") + key + "' must be boolean");
            out = v->as_bool();
        }
        return {};
    };
    const auto uint_field = [&](const char* key, std::uint64_t& out) -> Status {
        if (const auto* v = object.find(key)) {
            const std::string_view k(key);
            const bool restarts = k == "max_restarts";
            const bool slow_bound = k == "idle_timeout_ms" || k == "first_content_timeout_ms";
            const std::uint64_t max = restarts ? 1000 : slow_bound ? 86400000 : 3600000;
            const bool zero_ok = restarts || slow_bound;
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() > max || (!zero_ok && v->as_uint() == 0))
                return Status(ErrorCode::invalid_argument, std::string("llamaserver config: '") + key + "' is out of range");
            out = v->as_uint();
        }
        return {};
    };
    for (const auto& [key, out] : {std::pair{"mode", &setup.llamaserver_mode}, std::pair{"base_url", &setup.llamaserver_url},
                                    std::pair{"executable", &setup.llamaserver_executable}, std::pair{"grammar", &setup.llamaserver_grammar}}) {
        if (auto st = string_field(key, *out); !st.ok()) return st;
    }
    if (object.contains("mode") && (setup.llamaserver_mode.empty() ||
        (setup.llamaserver_mode != "attach" && setup.llamaserver_mode != "spawn")))
        return Status(ErrorCode::invalid_argument, "llamaserver config: mode must be attach or spawn");
    if (auto st = bool_field("allow_remote", setup.llamaserver_allow_remote); !st.ok()) return st;
    if (auto st = bool_field("native_completion", setup.llamaserver_native_completion); !st.ok()) return st;
    if (auto st = string_field("api_key_env", setup.llamaserver_api_key_env); !st.ok()) return st;
    if (setup.llamaserver_api_key_env.find('\0') != std::string::npos)
        return Status(ErrorCode::invalid_argument, "llamaserver config: api_key_env must not contain NUL");
    if (auto st = bool_field("unlimited_request_timeout", setup.llamaserver_unlimited_request_timeout); !st.ok()) return st;
    if (auto st = bool_field("upstream_grammar", setup.llamaserver_upstream_grammar); !st.ok()) return st;
    for (const char* key : {"connect_timeout_ms", "request_timeout_ms", "startup_timeout_ms", "poll_interval_ms",
                            "shutdown_timeout_ms", "restart_backoff_ms", "max_restart_backoff_ms", "max_restarts",
                            "idle_timeout_ms", "first_content_timeout_ms", "keepalive_gap_ms"}) {
        std::uint64_t* out = key == std::string_view("connect_timeout_ms") ? &setup.llamaserver_connect_timeout_ms :
            key == std::string_view("request_timeout_ms") ? &setup.llamaserver_request_timeout_ms :
            key == std::string_view("startup_timeout_ms") ? &setup.llamaserver_startup_timeout_ms :
            key == std::string_view("poll_interval_ms") ? &setup.llamaserver_poll_interval_ms :
            key == std::string_view("shutdown_timeout_ms") ? &setup.llamaserver_shutdown_timeout_ms :
            key == std::string_view("restart_backoff_ms") ? &setup.llamaserver_restart_backoff_ms :
            key == std::string_view("max_restart_backoff_ms") ? &setup.llamaserver_max_restart_backoff_ms :
            key == std::string_view("idle_timeout_ms") ? &setup.llamaserver_idle_timeout_ms :
            key == std::string_view("first_content_timeout_ms") ? &setup.llamaserver_first_content_timeout_ms :
            key == std::string_view("keepalive_gap_ms") ? &setup.llamaserver_keepalive_gap_ms : &setup.llamaserver_max_restarts;
        if (auto st = uint_field(key, *out); !st.ok()) return st;
    }
    if (const auto* args = object.find("args")) {
        if (!args->is_array()) return Status(ErrorCode::invalid_argument, "llamaserver config: 'args' must be an array of strings");
        setup.llamaserver_args.clear();
        for (const auto& item : args->as_array()) {
            if (!item.is_string()) return Status(ErrorCode::invalid_argument, "llamaserver config: 'args' entries must be strings");
            if (item.as_string().find('\0') != std::string::npos) return Status(ErrorCode::invalid_argument, "llamaserver config: args must not contain NUL");
            setup.llamaserver_args.push_back(item.as_string());
        }
    }
    if (setup.llamaserver_executable.find('\0') != std::string::npos)
        return Status(ErrorCode::invalid_argument, "llamaserver config: executable must not contain NUL");
    if (setup.llamaserver_restart_backoff_ms > setup.llamaserver_max_restart_backoff_ms)
        return Status(ErrorCode::invalid_argument, "llamaserver config: restart_backoff_ms exceeds max_restart_backoff_ms");
    if (const auto* tls = object.find("tls")) {
        if (!tls->is_object()) return Status(ErrorCode::invalid_argument, "llamaserver config: 'tls' must be an object");
        for (const auto& member : tls->as_object()) {
            if (member.first != "ca_bundle_path" && member.first != "pinned_sha256" && member.first != "pinned_cert_path" && member.first != "insecure_skip_verify" && member.first != "server_name" && member.first != "handshake_timeout_ms")
                return Status(ErrorCode::invalid_argument, "llamaserver config: unknown tls field '" + detail::bounded_for_message(member.first) + "'");
        }
        if (const auto* v = tls->find("ca_bundle_path")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.ca_bundle_path must be a string"); setup.llamaserver_ca_bundle_path = v->as_string(); }
        if (const auto* v = tls->find("pinned_sha256")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.pinned_sha256 must be a string"); setup.llamaserver_pinned_sha256 = v->as_string(); }
        if (const auto* v = tls->find("pinned_cert_path")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.pinned_cert_path must be a string"); setup.llamaserver_pinned_cert_path = v->as_string(); }
        if (const auto* v = tls->find("server_name")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.server_name must be a string"); setup.llamaserver_server_name = v->as_string(); }
        if (const auto* v = tls->find("insecure_skip_verify")) { if (!v->is_bool()) return Status(ErrorCode::invalid_argument, "tls.insecure_skip_verify must be boolean"); setup.llamaserver_insecure_skip_verify = v->as_bool(); }
        if (const auto* v = tls->find("handshake_timeout_ms")) { if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() > 3600000 || v->as_uint() == 0) return Status(ErrorCode::invalid_argument, "tls.handshake_timeout_ms must be an integer in [1, 3600000]"); setup.llamaserver_handshake_timeout_ms = v->as_uint(); }
    }
    if (auto st = string_field("log_file", setup.llamaserver_log_file); !st.ok()) return st;
    if (setup.llamaserver_log_file.find('\0') != std::string::npos)
        return Status(ErrorCode::invalid_argument, "llamaserver config: log_file must not contain NUL");
    if (auto st = bool_field("kv_pairing_check", setup.llamaserver_kv_pairing_check); !st.ok()) return st;
    if (auto st = bool_field("slot_affinity", setup.llamaserver_slot_affinity); !st.ok()) return st;
    if (const auto* v = object.find("context_length")) {
        if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() > 4294967296ull)
            return Status(ErrorCode::invalid_argument, "llamaserver config: 'context_length' must be an integer in [0, 4294967296]");
        setup.llamaserver_context_length = v->as_uint();
    }
    if (const auto* guard = object.find("spill_guard")) {
        if (!guard->is_object()) return Status(ErrorCode::invalid_argument, "llamaserver config: 'spill_guard' must be an object");
        static constexpr std::string_view guard_keys[] = {"enabled", "policy", "threshold_mib", "baseline_mib", "baseline_per_1k_ctx_mib",
            "sample_interval_ms", "fit_step_factor", "fit_step_align", "fit_min_ctx", "fit_max_attempts", "residency", "uma_mode"};
        for (const auto& member : guard->as_object()) {
            if (std::find(std::begin(guard_keys), std::end(guard_keys), member.first) == std::end(guard_keys))
                return Status(ErrorCode::invalid_argument, "llamaserver config: unknown spill_guard field '" + detail::bounded_for_message(member.first) + "'");
        }
        const auto bad = [](const char* key, const char* range) {
            return Status(ErrorCode::invalid_argument, std::string("llamaserver config: spill_guard.") + key + " must be " + range);
        };
        if (const auto* v = guard->find("enabled")) { if (!v->is_bool()) return bad("enabled", "boolean"); setup.llamaserver_spill_guard = v->as_bool(); }
        if (const auto* v = guard->find("policy")) {
            if (!v->is_string() || (v->as_string() != "warn" && v->as_string() != "refuse" && v->as_string() != "auto_fit"))
                return bad("policy", "warn, refuse or auto_fit");
            setup.llamaserver_spill_policy = v->as_string();
        }
        const auto uint_in = [&](const char* key, std::uint64_t lo, std::uint64_t hi, std::uint64_t& out, const char* range) -> Status {
            if (const auto* v = guard->find(key)) {
                if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() < lo || v->as_uint() > hi) return bad(key, range);
                out = v->as_uint();
            }
            return {};
        };
        if (auto st = uint_in("threshold_mib", 1, 1048576, setup.llamaserver_spill_threshold_mib, "an integer in [1, 1048576]"); !st.ok()) return st;
        if (auto st = uint_in("baseline_mib", 0, 1048576, setup.llamaserver_spill_baseline_mib, "an integer in [0, 1048576]"); !st.ok()) return st;
        if (auto st = uint_in("baseline_per_1k_ctx_mib", 0, 1024, setup.llamaserver_spill_baseline_per_1k_ctx_mib, "an integer in [0, 1024]"); !st.ok()) return st;
        if (auto st = uint_in("sample_interval_ms", 100, 3600000, setup.llamaserver_spill_sample_interval_ms, "an integer in [100, 3600000]"); !st.ok()) return st;
        if (auto st = uint_in("fit_step_align", 1, 1048576, setup.llamaserver_fit_step_align, "an integer in [1, 1048576]"); !st.ok()) return st;
        if (auto st = uint_in("fit_min_ctx", 1, 4294967296ull, setup.llamaserver_fit_min_ctx, "an integer in [1, 4294967296]"); !st.ok()) return st;
        if (auto st = uint_in("fit_max_attempts", 1, 32, setup.llamaserver_fit_max_attempts, "an integer in [1, 32]"); !st.ok()) return st;
        if (const auto* v = guard->find("fit_step_factor")) {
            if (!v->is_number() || !(v->as_double() > 0.0 && v->as_double() < 1.0)) return bad("fit_step_factor", "a number in (0, 1)");
            setup.llamaserver_fit_step_factor = v->as_double();
        }
        if (const auto* v = guard->find("uma_mode")) {
            if (auto st = detail::parse_uma_mode_value(*v, "spill_guard.uma_mode", setup.llamaserver_spill_uma_mode); !st.ok())
                return st;
        }
        if (const auto *v = guard->find("residency")) {
            if (auto st = detail::parse_residency_config(*v, setup); !st.ok()) return st;
        }
    }
    if (const auto* guard = object.find("stall_guard")) {
        if (!guard->is_object())
            return Status(ErrorCode::invalid_argument, "llamaserver config: 'stall_guard' must be an object");
        static constexpr std::string_view guard_keys[] = {"enabled", "stall_seconds", "policy", "generic_progress"};
        for (const auto& member : guard->as_object()) {
            if (std::find(std::begin(guard_keys), std::end(guard_keys), member.first) == std::end(guard_keys))
                return Status(ErrorCode::invalid_argument,
                              "llamaserver config: unknown stall_guard field '" + detail::bounded_for_message(member.first) + "'");
        }
        const auto bad = [](const char* key, const char* range) {
            return Status(ErrorCode::invalid_argument,
                          std::string("llamaserver config: stall_guard.") + key + " must be " + range);
        };
        if (const auto* v = guard->find("enabled")) {
            if (!v->is_bool())
                return bad("enabled", "boolean");
            setup.llamaserver_stall_guard = v->as_bool();
        }
        if (const auto* v = guard->find("stall_seconds")) {
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() < 1 || v->as_uint() > 86400)
                return bad("stall_seconds", "an integer in [1, 86400]");
            setup.llamaserver_stall_seconds = v->as_uint();
        }
        if (const auto* v = guard->find("policy")) {
            if (!v->is_string() || (v->as_string() != "warn" && v->as_string() != "restart" && v->as_string() != "cancel"))
                return bad("policy", "warn, restart or cancel");
            setup.llamaserver_stall_policy = v->as_string();
        }
        if (const auto* v = guard->find("generic_progress")) {
            if (!v->is_bool())
                return bad("generic_progress", "boolean");
            setup.llamaserver_stall_generic_progress = v->as_bool();
        }
    }
    if (const auto* host = object.find("host_memory")) {
        if (auto st = parse_host_memory_config(*host, setup.llamaserver_host_memory); !st.ok()) return st;
    }
    if (const auto* warmup = object.find("warmup")) {
        if (auto st = detail::parse_warmup_config(*warmup, setup); !st.ok()) return st;
    }
    if (const auto* spec = object.find("adaptive_speculation")) {
        if (auto st = detail::parse_adaptive_speculation_config(*spec, setup.llamaserver_adaptive_speculation); !st.ok())
            return st;
    }
    setup.llamaserver_config = path;
    destination = std::move(setup);
    return {};
}

std::vector<std::string> available_backend_names() {
    std::vector<std::string> names{kMockBackendName};
#if defined(SONDER_HAS_OLLAMA_BACKEND)
    names.emplace_back(kOllamaBackendName);
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    names.emplace_back(kLlamaCppBackendName);
#endif
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
    names.emplace_back(kLlamaServerBackendName);
#endif
    return names;
}

bool is_synthetic_backend(std::string_view name) noexcept { return name == kMockBackendName; }

Result<std::shared_ptr<Backend>> make_backend(const BackendSetup& setup) {
    if (setup.backend == kMockBackendName) {
        MockBackendOptions mo;
        mo.token_delay = setup.mock_token_delay;
        mo.fail_after_tokens = setup.mock_fail_after_tokens;
        return make_mock_backend(mo);
    }
    if (setup.backend == "ollama") {
#if defined(SONDER_HAS_OLLAMA_BACKEND)
        OllamaBackendOptions oo;
        if (!setup.ollama_url.empty()) {
            oo.base_url = setup.ollama_url;
        }
        oo.allow_remote = setup.ollama_allow_remote;
        return make_ollama_backend(oo);
#else
        return Status(ErrorCode::unsupported, "this build does not include the ollama backend (src/backends/ollama)");
#endif
    }
    if (setup.backend == "llamacpp") {
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
        auto lo = llamacpp_options(setup);
        if (!lo) return lo.status();
        return make_llamacpp_backend(std::move(lo).value());
#else
        return Status(ErrorCode::unsupported,
                      "this build does not include the llamacpp backend (configure with SONDER_WITH_LLAMA_CPP=ON)");
#endif
    }
    if (setup.backend == "llamaserver") {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        LlamaServerBackendOptions lo;
        lo.environment = setup.llamaserver_environment;
        if (!setup.llamaserver_mode.empty()) {
            if (setup.llamaserver_mode == "attach") lo.mode = LlamaServerMode::attach;
            else if (setup.llamaserver_mode == "spawn") lo.mode = LlamaServerMode::spawn;
            else return Status(ErrorCode::invalid_argument, "llamaserver mode must be attach or spawn");
        }
        if (!setup.llamaserver_url.empty()) lo.base_url = setup.llamaserver_url;
        lo.executable = setup.llamaserver_executable; lo.args = setup.llamaserver_args;
        lo.allow_remote = setup.llamaserver_allow_remote; lo.native_completion = setup.llamaserver_native_completion; lo.grammar = setup.llamaserver_grammar;
        lo.connect_timeout = std::chrono::milliseconds(setup.llamaserver_connect_timeout_ms);
        lo.request_timeout = std::chrono::milliseconds(setup.llamaserver_request_timeout_ms);
        lo.startup_timeout = std::chrono::milliseconds(setup.llamaserver_startup_timeout_ms);
        lo.poll_interval = std::chrono::milliseconds(setup.llamaserver_poll_interval_ms);
        lo.shutdown_timeout = std::chrono::milliseconds(setup.llamaserver_shutdown_timeout_ms);
        lo.restart_backoff = std::chrono::milliseconds(setup.llamaserver_restart_backoff_ms);
        lo.max_restart_backoff = std::chrono::milliseconds(setup.llamaserver_max_restart_backoff_ms);
        lo.max_restarts = static_cast<std::size_t>(setup.llamaserver_max_restarts);
        lo.tls.ca_bundle_path = setup.llamaserver_ca_bundle_path; lo.tls.pinned_sha256 = setup.llamaserver_pinned_sha256;
        lo.tls.pinned_cert_path = setup.llamaserver_pinned_cert_path; lo.tls.insecure_skip_verify = setup.llamaserver_insecure_skip_verify;
        lo.tls.server_name = setup.llamaserver_server_name; lo.tls.handshake_timeout = std::chrono::milliseconds(setup.llamaserver_handshake_timeout_ms);
        lo.spill_guard.enabled = setup.llamaserver_spill_guard;
        if (setup.llamaserver_spill_policy == "warn") lo.spill_guard.policy = LlamaServerSpillPolicy::warn;
        else if (setup.llamaserver_spill_policy == "refuse") lo.spill_guard.policy = LlamaServerSpillPolicy::refuse;
        else if (setup.llamaserver_spill_policy == "auto_fit") lo.spill_guard.policy = LlamaServerSpillPolicy::auto_fit;
        else return Status(ErrorCode::invalid_argument, "llamaserver spill policy must be warn, refuse or auto_fit");
        if (setup.llamaserver_spill_threshold_mib > 1048576 || setup.llamaserver_spill_baseline_mib > 1048576)
            return Status(ErrorCode::invalid_argument, "llamaserver spill threshold/baseline must be at most 1048576 MiB");
        lo.spill_guard.threshold_bytes = setup.llamaserver_spill_threshold_mib * 1024ull * 1024ull;
        lo.spill_guard.baseline_bytes = setup.llamaserver_spill_baseline_mib * 1024ull * 1024ull;
        if (setup.llamaserver_spill_baseline_per_1k_ctx_mib > 1024)
            return Status(ErrorCode::invalid_argument, "llamaserver spill baseline growth must be at most 1024 MiB per 1k ctx");
        lo.spill_guard.baseline_bytes_per_1k_ctx = setup.llamaserver_spill_baseline_per_1k_ctx_mib * 1024ull * 1024ull;
        lo.spill_guard.sample_interval = std::chrono::milliseconds(setup.llamaserver_spill_sample_interval_ms);
        lo.spill_guard.fit_step_factor = setup.llamaserver_fit_step_factor;
        lo.spill_guard.fit_step_align = setup.llamaserver_fit_step_align;
        lo.spill_guard.fit_min_ctx = setup.llamaserver_fit_min_ctx;
        lo.spill_guard.fit_max_attempts = static_cast<std::size_t>(setup.llamaserver_fit_max_attempts);
        if (!detail::is_valid_uma_mode(setup.llamaserver_spill_uma_mode))
            return Status(ErrorCode::invalid_argument,
                          "llamaserver spill_guard.uma_mode must be on, off, auto, true or false");
        if (!setup.llamaserver_residency.uma_mode.empty() &&
            !detail::is_valid_uma_mode(setup.llamaserver_residency.uma_mode))
            return Status(ErrorCode::invalid_argument,
                          "llamaserver residency uma_mode must be empty or on, off, auto, true or false");
        const std::string_view device_hint = detail::extract_device_hint(setup.llamaserver_args);
        const bool is_spawn = setup.llamaserver_mode == "spawn";
        const bool guard_active = setup.llamaserver_spill_guard || setup.llamaserver_residency.enabled;
        const bool needs_live_gpus =
            is_spawn && guard_active &&
            (setup.llamaserver_spill_uma_mode == "auto" || setup.llamaserver_residency.uma_mode == "auto");
        const std::vector<GpuDeviceInfo> live_gpus =
            needs_live_gpus ? enumerate_gpu_devices() : std::vector<GpuDeviceInfo>{};
        lo.spill_guard.uma_mode =
            detail::resolve_uma_mode(setup.llamaserver_spill_uma_mode, live_gpus, device_hint);
        if (is_spawn && setup.llamaserver_spill_uma_mode == "auto") {
            lo.spill_guard.uma_mode_policy = "auto";
        }
        lo.spill_guard.residency_uma_mode_explicit = !setup.llamaserver_residency.uma_mode.empty();
        if (auto st = detail::apply_residency_config(setup, lo.spill_guard.residency, live_gpus); !st.ok()) return st;
        lo.stall_guard.enabled = setup.llamaserver_stall_guard;
        lo.stall_guard.stall_seconds = std::chrono::seconds(setup.llamaserver_stall_seconds);
        lo.stall_guard.policy = setup.llamaserver_stall_policy;
        {
            const auto& h = setup.llamaserver_host_memory;
            lo.host_memory.estimated_resident_bytes = h.estimated_resident_bytes;
            lo.host_memory.estimated_private_bytes = h.estimated_private_bytes;
            lo.host_memory.estimate_lower_bound = h.estimate_lower_bound;
            lo.host_memory.reserve_bytes = h.reserve_mib * 1024ull * 1024ull;
            lo.host_memory.override_refusal = h.override_refusal;
            lo.host_memory.floor.enabled = h.floor_enabled;
            lo.host_memory.floor.policy = h.floor_policy;
            lo.host_memory.floor.floor_bytes = h.floor_mib * 1024ull * 1024ull;
            lo.host_memory.floor.resume_bytes = h.resume_mib * 1024ull * 1024ull;
            lo.host_memory.floor.consecutive_samples = static_cast<std::size_t>(h.consecutive_samples);
            lo.host_memory.floor.sample_interval = std::chrono::milliseconds(h.sample_interval_ms);
            lo.host_memory.floor.restart_cooldown = std::chrono::milliseconds(h.restart_cooldown_ms);
            lo.host_memory.floor.max_restarts = static_cast<std::size_t>(h.max_restarts);
        }
        lo.stall_guard.generic_progress = setup.llamaserver_stall_generic_progress;
        lo.api_key_env = setup.llamaserver_api_key_env;
        lo.idle_timeout = std::chrono::milliseconds(setup.llamaserver_idle_timeout_ms);
        lo.first_content_timeout = std::chrono::milliseconds(setup.llamaserver_first_content_timeout_ms);
        lo.keepalive_gap = std::chrono::milliseconds(setup.llamaserver_keepalive_gap_ms);
        lo.unlimited_request_timeout = setup.llamaserver_unlimited_request_timeout;
        lo.upstream_grammar = setup.llamaserver_upstream_grammar;
        lo.diagnostics.log_file = setup.llamaserver_log_file;
        lo.diagnostics.kv_pairing_check = setup.llamaserver_kv_pairing_check;
        lo.context_length = setup.llamaserver_context_length;
        lo.slot_affinity = setup.llamaserver_slot_affinity;
        lo.warmup.messages_file = setup.llamaserver_warmup_messages_file;
        lo.warmup.chat_template_kwargs = setup.llamaserver_warmup_chat_template_kwargs;
        lo.warmup.all_slots = setup.llamaserver_warmup_all_slots;
        lo.warmup.slots = setup.llamaserver_warmup_slots;
        lo.warmup.on_restart = setup.llamaserver_warmup_on_restart;
        lo.warmup.max_prefix_chars = setup.llamaserver_warmup_max_prefix_chars;
        {
            const auto& s = setup.llamaserver_adaptive_speculation;
            lo.adaptive_speculation.enabled = s.enabled;
            lo.adaptive_speculation.min_draft_n = s.min_draft_n;
            lo.adaptive_speculation.initial_draft_n = s.initial_draft_n;
            lo.adaptive_speculation.max_draft_n = s.max_draft_n;
            lo.adaptive_speculation.window_size = s.window_size;
            lo.adaptive_speculation.min_samples = s.min_samples;
            lo.adaptive_speculation.pause_acceptance_threshold = s.pause_acceptance_threshold;
            lo.adaptive_speculation.step_down_acceptance_threshold = s.step_down_acceptance_threshold;
            lo.adaptive_speculation.step_up_acceptance_threshold = s.step_up_acceptance_threshold;
            lo.adaptive_speculation.cooldown_requests = s.cooldown_requests;
            lo.adaptive_speculation.probe_draft_n = s.probe_draft_n;
            lo.adaptive_speculation.min_draft_tokens_per_sample = s.min_draft_tokens_per_sample;
        }
        std::shared_ptr<Backend> backend = make_llamaserver_backend(std::move(lo));
        return backend;
#else
        return Status(ErrorCode::unsupported, "this build does not include the llamaserver backend");
#endif
    }
    return Status(ErrorCode::invalid_argument,
                  "unknown backend '" + detail::bounded_for_message(setup.backend) + "' (expected mock, ollama, llamacpp or llamaserver)");
}

Status validate_backend_setup(const BackendSetup& setup) {
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    if (setup.backend == kLlamaCppBackendName) {
        auto lo = llamacpp_options(setup);
        return lo ? Status::success() : lo.status();
    }
#else
    (void)setup;
#endif
    return Status::success();
}

}  // namespace sonder::inference
