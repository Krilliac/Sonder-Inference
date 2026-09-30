#include "sonder/inference/backend_setup.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "sonder/inference/backends.hpp"
#include "sonder/inference/json.hpp"
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
            "max_restarts", "tls"};
        for (auto k : known) if (k == key) return false;
        return true;
    };
    for (const auto& member : object) {
        if (unknown(member.first)) return Status(ErrorCode::invalid_argument, "llamaserver config: unknown field '" + member.first + "'");
    }
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
            const bool restarts = std::string_view(key) == "max_restarts";
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() > (restarts ? 1000 : 3600000) || (!restarts && v->as_uint() == 0))
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
    for (const char* key : {"connect_timeout_ms", "request_timeout_ms", "startup_timeout_ms", "poll_interval_ms",
                            "shutdown_timeout_ms", "restart_backoff_ms", "max_restart_backoff_ms", "max_restarts"}) {
        std::uint64_t* out = key == std::string_view("connect_timeout_ms") ? &setup.llamaserver_connect_timeout_ms :
            key == std::string_view("request_timeout_ms") ? &setup.llamaserver_request_timeout_ms :
            key == std::string_view("startup_timeout_ms") ? &setup.llamaserver_startup_timeout_ms :
            key == std::string_view("poll_interval_ms") ? &setup.llamaserver_poll_interval_ms :
            key == std::string_view("shutdown_timeout_ms") ? &setup.llamaserver_shutdown_timeout_ms :
            key == std::string_view("restart_backoff_ms") ? &setup.llamaserver_restart_backoff_ms :
            key == std::string_view("max_restart_backoff_ms") ? &setup.llamaserver_max_restart_backoff_ms : &setup.llamaserver_max_restarts;
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
                return Status(ErrorCode::invalid_argument, "llamaserver config: unknown tls field '" + member.first + "'");
        }
        if (const auto* v = tls->find("ca_bundle_path")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.ca_bundle_path must be a string"); setup.llamaserver_ca_bundle_path = v->as_string(); }
        if (const auto* v = tls->find("pinned_sha256")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.pinned_sha256 must be a string"); setup.llamaserver_pinned_sha256 = v->as_string(); }
        if (const auto* v = tls->find("pinned_cert_path")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.pinned_cert_path must be a string"); setup.llamaserver_pinned_cert_path = v->as_string(); }
        if (const auto* v = tls->find("server_name")) { if (!v->is_string()) return Status(ErrorCode::invalid_argument, "tls.server_name must be a string"); setup.llamaserver_server_name = v->as_string(); }
        if (const auto* v = tls->find("insecure_skip_verify")) { if (!v->is_bool()) return Status(ErrorCode::invalid_argument, "tls.insecure_skip_verify must be boolean"); setup.llamaserver_insecure_skip_verify = v->as_bool(); }
        if (const auto* v = tls->find("handshake_timeout_ms")) { if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() > 3600000 || v->as_uint() == 0) return Status(ErrorCode::invalid_argument, "tls.handshake_timeout_ms must be an integer in [1, 3600000]"); setup.llamaserver_handshake_timeout_ms = v->as_uint(); }
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
        LlamaCppBackendOptions lo;
        lo.model_dirs = setup.model_dirs;
        if (setup.llamacpp_gpu_layers) lo.gpu_layers = *setup.llamacpp_gpu_layers;
        if (setup.llamacpp_context_length) lo.context_length = *setup.llamacpp_context_length;
        for (const std::string& spec : setup.llamacpp_tensor_overrides) {
            auto parsed = parse_tensor_override(spec);
            if (!parsed) return parsed.status();
            lo.tensor_overrides.push_back(std::move(parsed).value());
        }
        return make_llamacpp_backend(lo);
#else
        return Status(ErrorCode::unsupported,
                      "this build does not include the llamacpp backend (configure with SONDER_WITH_LLAMA_CPP=ON)");
#endif
    }
    if (setup.backend == "llamaserver") {
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        LlamaServerBackendOptions lo;
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
        std::shared_ptr<Backend> backend = make_llamaserver_backend(std::move(lo));
        return backend;
#else
        return Status(ErrorCode::unsupported, "this build does not include the llamaserver backend");
#endif
    }
    return Status(ErrorCode::invalid_argument,
                  "unknown backend '" + setup.backend + "' (expected mock, ollama, llamacpp or llamaserver)");
}

}  // namespace sonder::inference
