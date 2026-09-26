#include "sonder/inference/backend_setup.hpp"

#include <cstdlib>

#include "sonder/inference/backends.hpp"
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

std::vector<std::string> available_backend_names() {
    std::vector<std::string> names{kMockBackendName};
#if defined(SONDER_HAS_OLLAMA_BACKEND)
    names.emplace_back(kOllamaBackendName);
#endif
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    names.emplace_back(kLlamaCppBackendName);
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
        return make_llamacpp_backend(lo);
#else
        return Status(ErrorCode::unsupported,
                      "this build does not include the llamacpp backend (configure with SONDER_WITH_LLAMA_CPP=ON)");
#endif
    }
    return Status(ErrorCode::invalid_argument,
                  "unknown backend '" + setup.backend + "' (expected mock, ollama or llamacpp)");
}

}  // namespace sonder::inference
