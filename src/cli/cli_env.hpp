// sonder-infer CLI helpers: environment defaults for --backend, --model and
// --ollama-url (docs/CLI.md "Environment"). Flags always win over these.
//
//   SONDER_INFER_BACKEND   default --backend
//   SONDER_INFER_MODEL     default --model
//   SONDER_OLLAMA_URL      default --ollama-url
//   OLLAMA_HOST            used when SONDER_OLLAMA_URL is unset; "host",
//                          "host:port" or a URL (Ollama's own convention)
//
// Builds with the server module (SONDER_HAS_SERVER) use the shared
// implementation in sonder/inference/backend_setup.hpp, so `sonder-infer
// serve` and the other commands read the environment identically. Builds
// without the module use the equivalent local implementation below;
// tests/test_cli_spec.cpp checks that both agree.
#pragma once

#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#endif

namespace sonder::cli {

struct EnvDefaults {
    std::optional<std::string> backend;
    std::optional<std::string> model;
    std::optional<std::string> ollama_url;
};

// Reads one variable; std::getenv when empty. Empty values count as unset.
using EnvLookup = std::function<std::optional<std::string>(const char* name)>;

namespace detail {

inline std::optional<std::string> env_value(const EnvLookup& lookup, const char* name) {
    std::optional<std::string> v;
    if (lookup) {
        v = lookup(name);
    } else if (const char* raw = std::getenv(name); raw != nullptr) {
        v = std::string(raw);
    }
    if (v && v->empty()) v.reset();
    return v;
}

// OLLAMA_HOST value -> base URL: default scheme http, default port 11434
// (443 for https), wildcard bind addresses (0.0.0.0, [::]) and an empty host
// map to 127.0.0.1, trailing slashes are dropped.
inline std::string normalize_ollama_host_local(std::string_view value) {
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
    if (host.empty() || host == "0.0.0.0" || host == "[::]") host = "127.0.0.1";
    if (port.empty()) port = scheme == "https" ? "443" : "11434";
    while (path.size() > 1 && path.back() == '/') path.pop_back();
    if (path == "/") path.clear();
    return scheme + "://" + host + ":" + port + path;
}

inline EnvDefaults read_env_defaults_local(const EnvLookup& lookup) {
    EnvDefaults d;
    d.backend = env_value(lookup, "SONDER_INFER_BACKEND");
    d.model = env_value(lookup, "SONDER_INFER_MODEL");
    d.ollama_url = env_value(lookup, "SONDER_OLLAMA_URL");
    if (!d.ollama_url) {
        if (auto host = env_value(lookup, "OLLAMA_HOST")) d.ollama_url = normalize_ollama_host_local(*host);
    }
    return d;
}

}  // namespace detail

inline std::string normalize_ollama_host(std::string_view value) {
#if defined(SONDER_HAS_SERVER)
    return sonder::inference::normalize_ollama_host(value);
#else
    return detail::normalize_ollama_host_local(value);
#endif
}

inline EnvDefaults read_env_defaults(const EnvLookup& lookup = {}) {
#if defined(SONDER_HAS_SERVER)
    const auto shared = sonder::inference::backend_env_defaults(lookup);
    return EnvDefaults{shared.backend, shared.model, shared.ollama_url};
#else
    return detail::read_env_defaults_local(lookup);
#endif
}

// --ollama-allow-remote policy: prompts never travel over plain HTTP to a
// remote host. True when `url` is an http:// URL (scheme case-insensitive)
// whose host is not loopback; https:// and loopback hosts return false.
// Host extraction and the loopback rule mirror the Ollama client
// (net::parse_url() and net::is_loopback_host() in src/net/http_client.cpp):
// "localhost", "::1" and any host starting with "127.". URLs the client
// rejects anyway (no scheme, no host, an unterminated IPv6 literal) return
// false so the client's own error is reported.
// tests/test_cli_spec.cpp checks this agrees with the client's functions.
inline bool is_plain_http_remote(std::string_view url) {
    const auto scheme_end = url.find("://");
    if (scheme_end == std::string_view::npos || scheme_end != 4) return false;
    for (std::size_t i = 0; i < 4; ++i) {
        const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(url[i])));
        if (c != "http"[i]) return false;
    }
    std::string_view authority = url.substr(scheme_end + 3);
    authority = authority.substr(0, authority.find('/'));
    std::string_view host;
    if (!authority.empty() && authority.front() == '[') {
        const auto close = authority.find(']');
        if (close == std::string_view::npos) return false;  // the client rejects it
        host = authority.substr(1, close - 1);
    } else {
        host = authority.substr(0, authority.rfind(':'));
    }
    if (host.empty()) return false;  // the client rejects it ("URL has no host")
    const bool loopback = host == "localhost" || host == "::1" || host.rfind("127.", 0) == 0;
    return !loopback;
}

// Colored REPL labels: only on a terminal, never when NO_COLOR is set to a
// non-empty value (https://no-color.org) or TERM is "dumb".
inline bool color_enabled(bool stream_is_terminal, const EnvLookup& lookup = {}) {
    if (!stream_is_terminal) return false;
    if (detail::env_value(lookup, "NO_COLOR")) return false;
    const auto term = detail::env_value(lookup, "TERM");
    return !(term && *term == "dumb");
}

}  // namespace sonder::cli
