// Containment tests for the llamaserver attach and spawn boundaries.
//
// A hostile or confused caller (model-written configuration such as a tune
// output or a shared launch profile, or a request relayed by Sonder-runtime)
// must not be able to make Sonder attach to a non-local backend, or start a
// child llama-server with unsafe arguments or environment. Every table row
// names one attack shape and the outcome it must have.
//
// Classification used in the test names:
//   BUG        a documented boundary that does not hold (none found).
//   gap        no document claims the boundary either way, but a caller
//              could plausibly cause harm; the current behaviour is pinned so
//              a fix has to flip the row on purpose.
//   by-design  the docs explicitly accept the behaviour.
//
// No llama-server is started, no socket is opened and nothing is downloaded.
// The attach and spawn decisions are read through save_slot() on a generic
// mode backend (native_completion = false): it returns the validation status
// computed at construction when the options were refused, and otherwise
// unsupported("slot API requires native mode") before any I/O. The child
// environment is read from ProcessOptions::prepare(), the function that builds
// the block handed to CreateProcessW / posix_spawnp.
//
// llama-server behaviour cited below is from llama.cpp 7fe450e, the revision
// launch_profile.hpp cites: option and environment names from common/arg.cpp
// add_opt(...).set_env(...); the parser in common_params_parse_ex (common/
// arg.cpp:762-875: environment first, then argv, exact case-sensitive lookup
// after '_' -> '-' for "--" arguments, no --flag=value form, no @file response
// files); common_params_apply_system_config (common/arg.cpp:717-760: reads
// %PROGRAMDATA%\llama.cpp\config.ini and %APPDATA%\llama.cpp\config.ini, or
// /etc/llama.cpp and $XDG_CONFIG_HOME/~/.config on POSIX); and gcp_params in
// tools/server/server-http.cpp:83-125 (AIP_MODE=PREDICTION makes
// AIP_HTTP_PORT replace the --port value; Sonder refuses AIP_* overrides and
// clears an inherited AIP_MODE, so that path is closed).
#include <doctest/doctest.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "../process.hpp"
#include "../process_options.hpp"
#include "net/http_client.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/json.hpp"
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/launch_profile.hpp"
#endif

using namespace sonder::inference;
namespace ls = sonder::inference::llamaserver;

namespace {

enum class Decision { accepted, refused };

const char *to_text(Decision d) { return d == Decision::accepted ? "accepted" : "refused"; }

constexpr Decision A = Decision::accepted;
constexpr Decision R = Decision::refused;

constexpr std::string_view kNoIoMarker = "slot API requires native mode";

// save_slot() on a generic-mode backend: the validation status when refused,
// otherwise the no-I/O "slot API requires native mode" refusal.
Decision decision_of(const Status &status) {
    CHECK_FALSE(status.ok());
    if (status.code() == ErrorCode::unsupported && status.message().find(kNoIoMarker) != std::string::npos)
        return Decision::accepted;
    return Decision::refused;
}

LlamaServerBackendOptions quiet_options() {
    LlamaServerBackendOptions o;
    o.native_completion = false; // no attached monitor and no request at construction
    o.spill_guard.enabled = false;
    o.stall_guard.enabled = false;
    o.diagnostics.kv_pairing_check = false;
    return o;
}

Status attach_status(const std::string &url, bool allow_remote = false) {
    auto o = quiet_options();
    o.mode = LlamaServerMode::attach;
    o.base_url = url;
    o.allow_remote = allow_remote;
    return make_llamaserver_backend(std::move(o))->save_slot(0, "containment.bin");
}

Status spawn_status(std::vector<std::string> args, std::vector<std::pair<std::string, std::string>> env = {}) {
    auto o = quiet_options();
    o.mode = LlamaServerMode::spawn;
    o.executable = "containment-never-started-llama-server.exe";
    o.args = std::move(args);
    o.environment = std::move(env);
    return make_llamaserver_backend(std::move(o))->save_slot(0, "containment.bin");
}

// The same two steps validate_options() runs for attach mode
// (llamaserver_backend.cpp: parse_url, then is_loopback_host).
Decision url_decision(const std::string &url) {
    auto parsed = net::parse_url(url);
    if (!parsed.ok())
        return Decision::refused;
    return net::is_loopback_host(parsed.value().host) ? Decision::accepted : Decision::refused;
}

// ASCII -> the Unicode fullwidth form (U+FF01..U+FF5E) as UTF-8. IDNA/UTS-46
// maps these back to ASCII, so a resolver that normalizes them would read the
// ASCII name.
std::string fullwidth(std::string_view ascii) {
    std::string out;
    for (const char c : ascii) {
        const auto u = static_cast<unsigned char>(c);
        if (u < 0x21 || u > 0x7e) {
            out.push_back(c);
            continue;
        }
        const unsigned cp = 0xff00u + (u - 0x20u);
        out.push_back(static_cast<char>(0xe0u | (cp >> 12)));
        out.push_back(static_cast<char>(0x80u | ((cp >> 6) & 0x3fu)));
        out.push_back(static_cast<char>(0x80u | (cp & 0x3fu)));
    }
    return out;
}

struct AttachRow {
    std::string name;
    std::string url;
    Decision expected;
};

// Loopback policy (common/loopback.hpp, docs/SERVER.md --host): "localhost"
// exactly, ::1, or a strict dotted-quad IPv4 literal in 127.0.0.0/8. Rows
// marked "true loopback, refused" are spellings the OS would route to this
// machine that the strict policy refuses anyway; refusing them is the safe
// direction. A non-loopback destination marked accepted would be a BUG.
std::vector<AttachRow> attach_rows() {
    return {
        // IPv4
        {"dotted quad 127.0.0.1 with port", "http://127.0.0.1:8080", A},
        {"dotted quad without port", "http://127.0.0.1", A},
        {"dotted quad at the top of 127/8", "http://127.255.255.254:8080", A},
        {"short form 127.1 (true loopback, refused)", "http://127.1:8080", R},
        {"hex octet 0x7f.1 (true loopback, refused)", "http://0x7f.1:8080", R},
        {"hex dotted quad 0x7f.0.0.1 (true loopback, refused)", "http://0x7f.0.0.1:8080", R},
        {"decimal integer 2130706433 (true loopback, refused)", "http://2130706433:8080", R},
        {"octal 0177.0.0.1 (true loopback, refused)", "http://0177.0.0.1:8080", R},
        {"zero padded 127.000.000.001 (true loopback, refused)", "http://127.000.000.001:8080", R},
        {"wildcard 0.0.0.0", "http://0.0.0.0:8080", R},
        {"private 10.77.0.2", "http://10.77.0.2:8080", R},
        {"just outside 127/8: 128.0.0.1", "http://128.0.0.1:8080", R},
        {"octet overflow 127.0.0.256", "http://127.0.0.256:8080", R},
        {"five octets 127.0.0.1.2", "http://127.0.0.1.2:8080", R},
        {"trailing dot 127.0.0.1.", "http://127.0.0.1.:8080", R},
        // IPv6
        {"bracketed [::1] with port", "http://[::1]:8080", A},
        {"bracketed [::1] without port", "http://[::1]", A},
        {"unbracketed ::1:8080 parses as host ::1 port 8080", "http://::1:8080", A},
        {"unbracketed ::1 without port parses as host ':'", "http://::1", R},
        {"IPv4-mapped [::ffff:127.0.0.1] (true loopback, refused)", "http://[::ffff:127.0.0.1]:8080", R},
        {"IPv4-mapped hex [::ffff:7f00:1] (true loopback, refused)", "http://[::ffff:7f00:1]:8080", R},
        {"uncompressed [0:0:0:0:0:0:0:1] (true loopback, refused)", "http://[0:0:0:0:0:0:0:1]:8080", R},
        {"zone id [::1%25lo]", "http://[::1%25lo]:8080", R},
        {"raw zone id [::1%lo]", "http://[::1%lo]:8080", R},
        {"unspecified [::]", "http://[::]:8080", R},
        {"text after the bracket [::1]evil", "http://[::1]evil:8080", R},
        {"userinfo after the bracket [::1]:80@evil.com", "http://[::1]:80@evil.com", R},
        {"link-local [fe80::1]", "http://[fe80::1]:8080", R},
        // Names
        {"localhost", "http://localhost", A},
        {"localhost with port", "http://localhost:8080", A},
        {"upper-case LOCALHOST", "http://LOCALHOST:8080", R},
        {"upper-case trailing dot LOCALHOST.", "http://LOCALHOST.:8080", R},
        {"trailing dot localhost.", "http://localhost.:8080", R},
        {"suffix localhost.evil.com", "http://localhost.evil.com:8080", R},
        {"prefix evil.localhost", "http://evil.localhost:8080", R},
        {"wildcard DNS 127.0.0.1.nip.io", "http://127.0.0.1.nip.io:8080", R},
        {"fragment then userinfo evil.com#@127.0.0.1", "http://evil.com#@127.0.0.1/", R},
        {"fragment after the address 127.0.0.1#@evil.com", "http://127.0.0.1#@evil.com/", R},
        {"query after the address 127.0.0.1?@evil.com", "http://127.0.0.1?@evil.com/", R},
        {"userinfo user@evil.com", "http://user@evil.com:8080", R},
        {"loopback as userinfo 127.0.0.1@evil.com", "http://127.0.0.1@evil.com:8080", R},
        {"loopback as userinfo localhost@evil.com", "http://localhost@evil.com", R},
        {"userinfo naming loopback as the host evil.com@127.0.0.1 (refused)", "http://evil.com@127.0.0.1:8080", R},
        {"port then userinfo 127.0.0.1:80@evil.com", "http://127.0.0.1:80@evil.com", R},
        {"userinfo with port evil.com:80@127.0.0.1", "http://evil.com:80@127.0.0.1", R},
        {"backslash 127.0.0.1\\@evil.com", "http://127.0.0.1\\@evil.com", R},
        {"percent-encoded slash 127.0.0.1%2f@evil.com", "http://127.0.0.1%2f@evil.com", R},
        {"space inside the host 127.0.0.1 .evil.com", "http://127.0.0.1 .evil.com:8080", R},
        {"trailing tab localhost<TAB>", "http://localhost\t:8080", R},
        // IDNA and look-alikes
        {"fullwidth digits and dots 127.0.0.1", "http://" + fullwidth("127.0.0.1") + ":8080", R},
        {"fullwidth digits with ASCII dots", "http://" + fullwidth("127") + ".0.0.1:8080", R},
        {"ideographic full stops 127\xe3\x80\x82" "0\xe3\x80\x82" "0\xe3\x80\x82" "1",
         "http://127\xe3\x80\x82" "0\xe3\x80\x82" "0\xe3\x80\x82" "1:8080", R},
        {"fullwidth localhost", "http://" + fullwidth("localhost") + ":8080", R},
        {"Cyrillic o in localhost", "http://l\xd0\xbe" "calhost:8080", R},
        {"punycode label xn--localhost-", "http://xn--localhost-:8080", R},
        // Ports
        {"empty port 127.0.0.1: uses the default port", "http://127.0.0.1:", A},
        {"port zero", "http://127.0.0.1:0", R},
        {"port 65536", "http://127.0.0.1:65536", R},
        {"negative port", "http://127.0.0.1:-1", R},
        {"two ports 127.0.0.1:8080:9090", "http://127.0.0.1:8080:9090", R},
        {"non-numeric port", "http://localhost:http", R},
        // Scheme and emptiness
        {"empty string", "", R},
        {"scheme only", "http://", R},
        {"port only", "http://:8080", R},
        {"no scheme", "127.0.0.1:8080", R},
        {"leading space before the scheme", " http://127.0.0.1:8080", R},
        {"ftp scheme", "ftp://127.0.0.1:8080", R},
        {"file scheme", "file:///etc/passwd", R},
        {"upper-case scheme HTTP", "HTTP://127.0.0.1:8080", A},
#if defined(SONDER_HAS_TLS)
        {"https to loopback (TLS build)", "https://127.0.0.1:8443", A},
#else
        {"https to loopback (build without TLS refuses https)", "https://127.0.0.1:8443", R},
#endif
        {"https to a remote host without allow_remote", "https://evil.example", R},
    };
}

#if defined(SONDER_HAS_SERVER)
class TempFile {
  public:
    explicit TempFile(const std::string &text) {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("sonder-containment-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 "-" + std::to_string(++counter) + ".json");
        std::ofstream out(path_, std::ios::binary);
        out << text;
    }
    ~TempFile() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    TempFile(const TempFile &) = delete;
    TempFile &operator=(const TempFile &) = delete;
    std::string path() const { return path_.string(); }

  private:
    std::filesystem::path path_;
};

std::string json_string(const std::string &s) { return json::Value(s).dump(); }

// The serve path: --llamaserver-config FILE -> load_llamaserver_config ->
// make_backend. Returns the load error, or the backend's save_slot status.
Status config_status(const std::string &config_json, BackendSetup *loaded = nullptr) {
    TempFile file(config_json);
    BackendSetup setup;
    setup.backend = "llamaserver";
    if (auto st = load_llamaserver_config(file.path(), setup); !st.ok())
        return st;
    if (loaded)
        *loaded = setup;
    auto backend = make_backend(setup);
    if (!backend.ok())
        return backend.status();
    auto server = std::dynamic_pointer_cast<LlamaServerBackend>(backend.value());
    REQUIRE(server);
    return server->save_slot(0, "containment.bin");
}

std::string attach_config(const std::string &url) {
    return R"({"mode":"attach","native_completion":false,"spill_guard":{"enabled":false},"base_url":)" +
           json_string(url) + "}";
}

std::string spawn_config(const std::string &env_object) {
    return R"({"mode":"spawn","executable":"containment-never-started-llama-server.exe","args":["-m","m.gguf"],)"
           R"("native_completion":false,"spill_guard":{"enabled":false},"env":)" +
           env_object + "}";
}

Status profile_status(const std::vector<std::string> &extra_args) {
    std::string items;
    for (const auto &arg : extra_args)
        items += (items.empty() ? "" : ",") + json_string(arg);
    auto doc = json::parse(R"({"profiles":[{"name":"p","backend":"llamaserver","model":"m.gguf","extra_args":[)" +
                           items + "]}]}");
    REQUIRE(doc.ok());
    auto parsed = parse_launch_profiles(doc.value());
    return parsed.ok() ? Status::success() : parsed.status();
}

Status profile_struct_status(std::vector<std::string> extra_args) {
    LaunchProfile p;
    p.name = "p";
    p.backend = kLaunchProfileBackendLlamaServer;
    p.model = "m.gguf";
    p.extra_args = std::move(extra_args);
    return validate_launch_profile(p);
}
#endif

// ----------------------------------------------------- child environment

// One "KEY=VALUE" entry per element, from the block prepare() built.
std::vector<std::string> prepared_entries(const ls::ProcessOptions &options) {
    std::vector<std::string> out;
#if defined(_WIN32)
    std::string current;
    for (const wchar_t c : options.environment) {
        if (c == L'\0') {
            if (!current.empty())
                out.push_back(current);
            current.clear();
        } else {
            // The test only uses ASCII names and values; anything else is
            // replaced so it cannot match by accident.
            current.push_back(c < 0x80 ? static_cast<char>(c) : '?');
        }
    }
#else
    out = options.environment;
#endif
    return out;
}

bool same_name(std::string_view a, std::string_view b) {
#if defined(_WIN32)
    // Windows environment names are case-insensitive.
    return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
               return std::toupper(static_cast<unsigned char>(x)) == std::toupper(static_cast<unsigned char>(y));
           });
#else
    return a == b;
#endif
}

// Values of every entry whose name matches `name`.
std::vector<std::string> values_named(const std::vector<std::string> &entries, std::string_view name) {
    std::vector<std::string> out;
    for (const auto &entry : entries) {
        const auto split = entry.find('=', 1); // Windows "=C:" entries keep their leading '='
        if (split != std::string::npos && same_name(std::string_view(entry).substr(0, split), name))
            out.push_back(entry.substr(split + 1));
    }
    return out;
}

// Sets (or removes) a variable in this process for the test's lifetime, in
// the OS block GetEnvironmentStringsW / environ read, and restores it.
class ScopedParentEnv {
  public:
    ScopedParentEnv(std::string name, std::optional<std::string> value) : name_(std::move(name)) {
        if (const char *old = std::getenv(name_.c_str()))
            previous_ = old;
        REQUIRE(apply(value));
    }
    ~ScopedParentEnv() { (void)apply(previous_); }
    ScopedParentEnv(const ScopedParentEnv &) = delete;
    ScopedParentEnv &operator=(const ScopedParentEnv &) = delete;

  private:
    bool apply(const std::optional<std::string> &value) const {
#if defined(_WIN32)
        // _putenv_s updates the CRT copy and the Win32 block; "" removes.
        return _putenv_s(name_.c_str(), value ? value->c_str() : "") == 0 &&
               (value ? SetEnvironmentVariableA(name_.c_str(), value->c_str()) != 0
                      : (SetEnvironmentVariableA(name_.c_str(), nullptr), true));
#else
        return value ? setenv(name_.c_str(), value->c_str(), 1) == 0 : unsetenv(name_.c_str()) == 0;
#endif
    }
    std::string name_;
    std::optional<std::string> previous_;
};

struct EnvRow {
    std::string name;
    std::string key;
    // The llama-server flag a launch profile refuses for the same setting, or
    // nullptr when there is no flag (or the flag is not profile-refused).
    const char *profile_refused_flag;
};

// Environment names llama-server reads (llama.cpp 7fe450e) that select a
// setting Sonder refuses as an argument somewhere, plus spelling variants.
std::vector<EnvRow> child_env_rows() {
    return {
        // Credentials and TLS (kForbiddenFlags)
        {"LLAMA_API_KEY is --api-key", "LLAMA_API_KEY", "--api-key"},
        {"LLAMA_ARG_API_KEY_FILE is --api-key-file", "LLAMA_ARG_API_KEY_FILE", "--api-key-file"},
        {"LLAMA_ARG_SSL_KEY_FILE is --ssl-key-file", "LLAMA_ARG_SSL_KEY_FILE", "--ssl-key-file"},
        {"LLAMA_ARG_SSL_CERT_FILE is --ssl-cert-file", "LLAMA_ARG_SSL_CERT_FILE", "--ssl-cert-file"},
        {"HF_TOKEN is --hf-token", "HF_TOKEN", "-hft"},
        // Downloads (kForbiddenFlags)
        {"LLAMA_ARG_MODEL_URL is -mu", "LLAMA_ARG_MODEL_URL", "-mu"},
        {"LLAMA_ARG_HF_REPO is -hf", "LLAMA_ARG_HF_REPO", "-hf"},
        {"LLAMA_ARG_HF_FILE is -hff", "LLAMA_ARG_HF_FILE", "-hff"},
        {"LLAMA_ARG_DOCKER_REPO is -dr", "LLAMA_ARG_DOCKER_REPO", "-dr"},
        {"LLAMA_ARG_MMPROJ_URL is -mmu", "LLAMA_ARG_MMPROJ_URL", "-mmu"},
        {"LLAMA_ARG_SPEC_DRAFT_HF_REPO is -hfd", "LLAMA_ARG_SPEC_DRAFT_HF_REPO", "-hfd"},
        {"LLAMA_ARG_MODELS_DIR is --models-dir", "LLAMA_ARG_MODELS_DIR", "--models-dir"},
        {"LLAMA_ARG_MODELS_PRESET is --models-preset", "LLAMA_ARG_MODELS_PRESET", "--models-preset"},
        // Agent tools, MCP, file serving, disk writes, remote RPC, programs
        {"LLAMA_ARG_TOOLS is --tools", "LLAMA_ARG_TOOLS", "--tools"},
        {"LLAMA_ARG_TOOLS_RUNTIME is --tools-runtime", "LLAMA_ARG_TOOLS_RUNTIME", "--tools-runtime"},
        {"LLAMA_ARG_MCP_SERVERS_CONFIG is --mcp-servers-config", "LLAMA_ARG_MCP_SERVERS_CONFIG",
         "--mcp-servers-config"},
        {"LLAMA_ARG_MCP_SERVERS_JSON is --mcp-servers-json", "LLAMA_ARG_MCP_SERVERS_JSON", "--mcp-servers-json"},
        {"LLAMA_ARG_AGENT is --agent", "LLAMA_ARG_AGENT", "--agent"},
        {"LLAMA_ARG_UI_MCP_PROXY is --ui-mcp-proxy", "LLAMA_ARG_UI_MCP_PROXY", "--ui-mcp-proxy"},
        {"LLAMA_ARG_STATIC_PATH is --path", "LLAMA_ARG_STATIC_PATH", "--path"},
        {"LLAMA_ARG_LOG_FILE is --log-file", "LLAMA_ARG_LOG_FILE", "--log-file"},
        {"LLAMA_ARG_RPC is --rpc", "LLAMA_ARG_RPC", "--rpc"},
        {"LLAMA_ARG_VIDEO_FFMPEG_DIR is --video-ffmpeg-dir", "LLAMA_ARG_VIDEO_FFMPEG_DIR", "--video-ffmpeg-dir"},
        // Listen address: the supervisor's appended --host/--port are argv,
        // which llama-server applies after the environment, so these lose.
        {"LLAMA_ARG_HOST (overridden by the appended --host)", "LLAMA_ARG_HOST", nullptr},
        {"LLAMA_ARG_PORT (overridden by the appended --port)", "LLAMA_ARG_PORT", nullptr},
        // Not read by llama-server 7fe450e (it reads LLAMA_API_KEY), but the
        // name callers expect.
        {"LLAMA_ARG_API_KEY (name not read by 7fe450e)", "LLAMA_ARG_API_KEY", nullptr},
        // Non-LLAMA_ARG names that change what the child does
        // AIP_MODE / AIP_HTTP_PORT (Vertex AI port override) are refused: see
        // "AIP_MODE never reaches the child" below.
        {"APPDATA chooses the user config.ini (Windows)", "APPDATA", nullptr},
        {"PROGRAMDATA chooses the system config.ini (Windows)", "PROGRAMDATA", nullptr},
        {"XDG_CONFIG_HOME chooses the user config.ini (POSIX)", "XDG_CONFIG_HOME", nullptr},
        {"HOME chooses the user config.ini (POSIX)", "HOME", nullptr},
        {"MODEL_ENDPOINT chooses the download host", "MODEL_ENDPOINT", nullptr},
        {"HF_ENDPOINT chooses the download host", "HF_ENDPOINT", nullptr},
        {"LLAMA_CACHE chooses where downloads are written", "LLAMA_CACHE", nullptr},
        // Spelling variants
        {"lower-case llama_api_key (same variable on Windows)", "llama_api_key", nullptr},
        {"mixed-case Llama_Arg_Hf_Repo (same variable on Windows)", "Llama_Arg_Hf_Repo", nullptr},
        {"trailing space LLAMA_ARG_HF_REPO<SP> (a different variable)", "LLAMA_ARG_HF_REPO ", nullptr},
        {"leading space <SP>LLAMA_ARG_HF_REPO (a different variable)", " LLAMA_ARG_HF_REPO", nullptr},
        {"control character LLAMA_ARG_HF_REPO\\x01 (a different variable)", "LLAMA_ARG_HF_REPO\x01", nullptr},
    };
}

} // namespace

// ================================================================ attach

TEST_CASE("containment: attach accepts only loopback base URLs through the backend entry point") {
    const auto rows = attach_rows();
    REQUIRE(rows.size() == 71u);
    std::size_t accepted = 0;
    for (const auto &row : rows) {
        CAPTURE(row.name);
        CAPTURE(row.url);
        const Decision backend = decision_of(attach_status(row.url));
        const Decision direct = url_decision(row.url);
        INFO("expected " << to_text(row.expected) << ", backend " << to_text(backend) << ", parse_url+is_loopback_host "
                         << to_text(direct));
        CHECK(backend == row.expected);
        CHECK(direct == row.expected);
        accepted += backend == Decision::accepted;
    }
    // Both outcomes occur, so the no-I/O probe really distinguishes them.
#if defined(SONDER_HAS_TLS)
    CHECK(accepted == 11u);
#else
    CHECK(accepted == 10u);
#endif
}

TEST_CASE("containment: attach refuses plain HTTP to a remote host even with allow_remote (by-design https only)") {
    // docs/integration/llama-server.md "Security and configuration limits":
    // remote hosts need allow_remote and HTTPS.
    for (const std::string url : {"http://10.77.0.2:8080", "http://evil.example", "http://127.0.0.1.nip.io:8080",
                                  "http://localhost.evil.com", "http://user@evil.com"}) {
        CAPTURE(url);
        CHECK(decision_of(attach_status(url, true)) == Decision::refused);
    }
#if defined(SONDER_HAS_TLS)
    CHECK(decision_of(attach_status("https://evil.example", true)) == Decision::accepted);
#else
    CHECK(decision_of(attach_status("https://evil.example", true)) == Decision::refused);
#endif
    CHECK(decision_of(attach_status("https://evil.example", false)) == Decision::refused);
}

#if defined(SONDER_HAS_SERVER)
TEST_CASE("containment: attach accepts only loopback base URLs through the llamaserver config file") {
    const auto rows = attach_rows();
    REQUIRE(rows.size() == 71u);
    for (const auto &row : rows) {
        CAPTURE(row.name);
        CAPTURE(row.url);
        // An empty base_url keeps the backend default http://127.0.0.1:8080
        // (make_backend only copies a non-empty value).
        const Decision expected = row.url.empty() ? Decision::accepted : row.expected;
        CHECK(decision_of(config_status(attach_config(row.url))) == expected);
    }
    // allow_remote in the file still needs HTTPS.
    CHECK(decision_of(config_status(R"({"mode":"attach","native_completion":false,"allow_remote":true,)"
                                    R"("base_url":"http://10.77.0.2:8080"})")) == Decision::refused);
}
#endif

// ========================================================= spawn arguments

namespace {
struct ArgRow {
    std::string name;
    std::vector<std::string> args;
    Decision expected;
};
} // namespace

TEST_CASE("containment: spawn arguments refuse every llama-server spelling of host, port and the terminator") {
    using V = std::vector<std::string>;
    const std::vector<ArgRow> rows = {
        {"--host value", V{"--host", "0.0.0.0"}, R},
        {"--host=value", V{"--host=0.0.0.0"}, R},
        {"--host with an empty value", V{"--host", ""}, R},
        {"--host_ prefix", V{"--host_", "0.0.0.0"}, R},
        {"--port value", V{"--port", "9000"}, R},
        {"--port=value", V{"--port=9000"}, R},
        {"--port_ prefix", V{"--port_", "9000"}, R},
        {"argument-list terminator --", V{"--"}, R},
        {"--host after allowed flags", V{"-m", "m.gguf", "--no-warmup", "--host", "::"}, R},
        {"--host as the value of another flag", V{"--alias", "--host"}, R},
        {"NUL then --host in one argument", V{std::string("--model\0--host", 14)}, R},
        {"NUL at the end", V{std::string("--no-warmup\0", 12)}, R},
        {"NUL then --port in a value", V{"--alias", std::string("x\0--port", 8)}, R},
        // llama-server never reads these as --host/--port: the parser looks the
        // argument up exactly, only rewriting '_' to '-' after a leading "--",
        // and exits on an unknown argument. They are not smuggling vectors.
        {"leading space ' --host' (unknown to llama-server)", V{" --host", "0.0.0.0"}, A},
        {"trailing tab '--host<TAB>' (unknown to llama-server)", V{"--host\t", "0.0.0.0"}, A},
        {"quoted '\"--host\"' (one literal argv element)", V{"\"--host\"", "0.0.0.0"}, A},
        {"upper-case --HOST (lookup is case-sensitive)", V{"--HOST", "0.0.0.0"}, A},
        {"single dash -host (not a llama-server flag)", V{"-host", "0.0.0.0"}, A},
        {"fullwidth hyphens (not a llama-server flag)", V{fullwidth("--") + "host", "0.0.0.0"}, A},
        {"response file @args.txt (llama-server has none)", V{"@args.txt"}, A},
        {"host flag inside one argument with spaces", V{"x --host 0.0.0.0"}, A},
    };
    REQUIRE(rows.size() == 21u);
    for (const auto &row : rows) {
        CAPTURE(row.name);
        const Decision direct = ls::validate_process_arguments(row.args).ok() ? Decision::accepted : Decision::refused;
        const Decision backend = decision_of(spawn_status(row.args));
        INFO("expected " << to_text(row.expected) << ", validate_process_arguments " << to_text(direct)
                         << ", spawn backend " << to_text(backend));
        CHECK(direct == row.expected);
        CHECK(backend == row.expected);
    }
}

#if defined(SONDER_HAS_SERVER)
namespace {
// Every alias llama-server 7fe450e has for an option in kForbiddenFlags
// (common/arg.cpp add_opt names).
struct FlagGroup {
    std::string what;
    std::vector<std::string> aliases;
};
const std::vector<FlagGroup> &forbidden_groups() {
    static const std::vector<FlagGroup> groups = {
        {"listen host", {"--host"}},
        {"listen port", {"--port"}},
        {"API key", {"--api-key"}},
        {"API key file", {"--api-key-file"}},
        {"TLS key", {"--ssl-key-file"}},
        {"TLS certificate", {"--ssl-cert-file"}},
        {"model URL download", {"-mu", "--model-url"}},
        {"Hugging Face repo download", {"-hf", "-hfr", "--hf-repo"}},
        {"Hugging Face file download", {"-hff", "--hf-file"}},
        {"Hugging Face token", {"-hft", "--hf-token"}},
        {"Docker repo download", {"-dr", "--docker-repo"}},
        {"mmproj URL download", {"-mmu", "--mmproj-url"}},
        {"draft model Hugging Face download", {"--spec-draft-hf", "-hfd", "-hfrd", "--hf-repo-draft"}},
        {"router models directory", {"--models-dir"}},
        {"router model presets", {"--models-preset"}},
    };
    return groups;
}

// The spellings llama-server reads as `alias`: as written, the refused
// --flag=value form, and for "--" flags every '_' variant (the parser turns
// each '_' into '-' before the lookup).
std::vector<std::string> spellings(const std::string &alias) {
    std::vector<std::string> out{alias, alias + "=x"};
    if (alias.rfind("--", 0) != 0)
        return out;
    std::string all = alias;
    std::replace(all.begin() + 2, all.end(), '-', '_');
    if (all != alias) {
        out.push_back(all);
        out.push_back(all + "=x");
    }
    for (std::size_t i = 2; i < alias.size(); ++i) {
        if (alias[i] != '-')
            continue;
        std::string one = alias;
        one[i] = '_';
        if (one != all)
            out.push_back(one);
    }
    return out;
}
} // namespace

TEST_CASE("containment: launch profiles refuse every llama-server spelling of the forbidden flags") {
    std::size_t rows = 0;
    for (const auto &group : forbidden_groups()) {
        for (const auto &alias : group.aliases) {
            for (const auto &spelling : spellings(alias)) {
                CAPTURE(group.what);
                CAPTURE(spelling);
                const Status st = profile_status({"--no-warmup", spelling, "value"});
                CHECK(st.code() == ErrorCode::invalid_argument);
                CHECK(st.message().find("must not contain") != std::string::npos);
                ++rows;
            }
        }
    }
    CHECK(rows == 88u);
    // Smuggling through NUL is refused before any name check.
    for (const std::string &smuggled : {std::string("--no-warmup\0--api-key", 21), std::string("x\0-hf", 5),
                                       std::string("\0--ssl-key-file", 15)}) {
        CAPTURE(smuggled.size());
        CHECK(profile_struct_status({smuggled, "value"}).code() == ErrorCode::invalid_argument);
    }
}

TEST_CASE("containment: launch-profile spellings llama-server would not read as a forbidden flag stay allowed") {
    // Each is an unknown argument to llama-server 7fe450e (exact lookup after
    // '_' -> '-' for "--" arguments; no response files), so the child exits
    // instead of reading a key or downloading. Pinned so a change is noticed.
    const std::vector<std::pair<std::string, std::string>> rows = {
        {"upper-case --API-KEY", "--API-KEY"},
        {"mixed-case --Api-Key", "--Api-Key"},
        {"upper-case -HF", "-HF"},
        {"leading space ' --api-key'", " --api-key"},
        {"leading tab '<TAB>-hf'", "\t-hf"},
        {"quoted '\"--api-key\"'", "\"--api-key\""},
        {"response file @args.txt", "@args.txt"},
        {"fullwidth hyphens before api-key", fullwidth("--") + "api-key"},
        {"single-dash underscore -h_f (single-dash names are not rewritten)", "-h_f"},
        {"single-dash -ssl-key-file (not a llama-server flag)", "-ssl-key-file"},
    };
    for (const auto &[name, arg] : rows) {
        CAPTURE(name);
        CHECK(profile_status({arg, "value"}).ok());
    }
}

TEST_CASE("containment: by-design operator spawn args are filtered only for host, port and the terminator") {
    // Three lists decide spawn arguments:
    //  1. process.cpp is_forbidden (validate_process_arguments): "--",
    //     --host[=], --port[=], --host_*, --port_*. Used by
    //     LlamaServerBackend validate_options (spawn), Supervisor::start and
    //     sonder-infer tune.
    //  2. serve_main.cpp spawn check: --host[=], --port[=] (no "--"); list 1
    //     still runs when serve builds the backend.
    //  3. launch_profile.cpp kForbiddenFlags (+ unsafe-feature, disk-write,
    //     RPC, program and model-preset lists), used only by
    //     validate_launch_profile for a profile's extra_args.
    // docs/integration/launch-profiles.md: "--llamaserver-arg without a
    // profile is an operator flag and is not filtered this way", and with
    // --profile serve refuses the config file's `args`. So a credential or
    // download flag in operator args (--llamaserver-arg, config `args`) is
    // accepted on purpose; this pins that split.
    std::size_t both = 0, profile_only = 0;
    for (const auto &group : forbidden_groups()) {
        for (const auto &alias : group.aliases) {
            CAPTURE(alias);
            CHECK(profile_status({alias, "value"}).code() == ErrorCode::invalid_argument);
            const bool listen = alias == "--host" || alias == "--port";
            const Decision expected = listen ? Decision::refused : Decision::accepted;
            CHECK((ls::validate_process_arguments({alias, "value"}).ok() ? Decision::accepted : Decision::refused) ==
                  expected);
            CHECK(decision_of(spawn_status({alias, "value"})) == expected);
            CHECK(decision_of(config_status(
                      R"({"mode":"spawn","executable":"containment-never-started-llama-server.exe","args":[)" +
                      json_string(alias) +
                      R"(,"value"],"native_completion":false,"spill_guard":{"enabled":false}})")) == expected);
            (listen ? both : profile_only) += 1;
        }
    }
    CHECK(both == 2u);
    CHECK(profile_only == 23u);
    // "--" is refused by lists 1 and 3.
    CHECK(profile_status({"--"}).code() == ErrorCode::invalid_argument);
    CHECK_FALSE(ls::validate_process_arguments({"--"}).ok());
}
#endif

// ======================================================= child environment

TEST_CASE("containment: child environment validation keeps refusing malformed entries") {
    using Env = std::vector<std::pair<std::string, std::string>>;
    const std::vector<std::pair<std::string, Env>> rows = {
        {"empty name", Env{{"", "x"}}},
        {"'=' in the name", Env{{"LLAMA_ARG_HF_REPO=x", "y"}}},
        {"drive-style name =C:", Env{{"=C:", "C:\\"}}},
        {"NUL in the name", Env{{std::string("LLAMA_API_KEY\0X", 15), "x"}}},
        {"NUL in the value", Env{{"SONDER_CONTAINMENT", std::string("a\0LLAMA_API_KEY=k", 17)}}},
        {"more than 128 entries", Env(129, {"SONDER_CONTAINMENT", "x"})},
        {"more than 32760 bytes", Env{{"SONDER_CONTAINMENT", std::string(32760, 'x')}}},
    };
    for (const auto &[name, env] : rows) {
        CAPTURE(name);
        CHECK(ls::validate_process_environment(env).code() == ErrorCode::invalid_argument);
        ls::ProcessSpec spec;
        spec.environment = env;
        ls::ProcessOptions options;
        CHECK(options.prepare(spec).code() == ErrorCode::invalid_argument);
    }
#if defined(SONDER_HAS_SERVER)
    for (const std::string env : {R"({"":"x"})", R"({"A=B":"x"})", R"({"=C:":"x"})", R"({"K":1})"}) {
        CAPTURE(env);
        CHECK(config_status(spawn_config(env)).code() == ErrorCode::invalid_argument);
    }
#endif
}

TEST_CASE("containment gap: child environment overrides equivalent to refused llama-server flags are accepted") {
    // GAP: validate_process_environment (process_options.cpp:12) and the
    // config loader's "env" object (llamaserver_config_extensions.hpp) check
    // only the shape of each entry, never the name. llama-server reads each
    // name below as the setting in the row name, so an "env" object in a
    // (model-written) llamaserver config can set what the argument checks
    // refuse. No document restricts env names; llama-server.md calls env
    // "child-only string environment overrides". With --profile, serve refuses
    // the config's `args` but still applies its `env`.
    const auto rows = child_env_rows();
    REQUIRE(rows.size() == 38u);
#if defined(SONDER_HAS_SERVER)
    std::size_t profile_refused = 0;
#endif
    for (const auto &row : rows) {
        CAPTURE(row.name);
        const std::string value = "containment-value";
        const std::vector<std::pair<std::string, std::string>> env{{row.key, value}};
        CHECK(ls::validate_process_environment(env).ok());
        CHECK(decision_of(spawn_status({"-m", "m.gguf"}, env)) == Decision::accepted);

        // The entry reaches the block prepare() builds for the child.
        ls::ProcessSpec spec;
        spec.environment = env;
        ls::ProcessOptions options;
        REQUIRE(options.prepare(spec).ok());
        const auto values = values_named(prepared_entries(options), row.key);
        CHECK(values == std::vector<std::string>{value});

#if defined(SONDER_HAS_SERVER)
        if (row.key.find('\x01') == std::string::npos) {
            BackendSetup loaded;
            const std::string object = "{" + json_string(row.key) + ":" + json_string(value) + "}";
            CHECK(decision_of(config_status(spawn_config(object), &loaded)) == Decision::accepted);
            CHECK(loaded.llamaserver_environment == env);
        }
        if (row.profile_refused_flag) {
            // The same setting as an argument is refused in a launch profile.
            CHECK(profile_status({row.profile_refused_flag, "value"}).code() == ErrorCode::invalid_argument);
            ++profile_refused;
        }
#endif
    }
#if defined(SONDER_HAS_SERVER)
    CHECK(profile_refused == 23u);
#endif
}

TEST_CASE("containment: AIP_MODE never reaches the child and AIP_* overrides are refused") {
    // llama-server 7fe450e replaces the --port value with AIP_HTTP_PORT when
    // AIP_MODE=PREDICTION (server-http.cpp:117-125; an empty or unset AIP_MODE
    // leaves it disabled). The supervisor owns the child's port, so AIP_*
    // overrides are refused and an inherited AIP_MODE is removed from the
    // child's block.
    CHECK(decision_of(spawn_status({"-m", "m.gguf", "--port", "8080"})) == Decision::refused);
    const std::vector<std::pair<std::string, std::string>> env{{"AIP_MODE", "PREDICTION"}, {"AIP_HTTP_PORT", "8080"}};
    CHECK(ls::validate_process_environment(env).code() == ErrorCode::invalid_argument);
    CHECK(ls::validate_process_environment({{"AIP_HTTP_PORT", "8080"}}).code() == ErrorCode::invalid_argument);
    CHECK(ls::validate_process_environment({{"AIP_HEALTH_ROUTE", "/x"}}).code() == ErrorCode::invalid_argument);
#if defined(_WIN32)
    CHECK(ls::validate_process_environment({{"aip_mode", "PREDICTION"}}).code() == ErrorCode::invalid_argument);
#endif
    CHECK(ls::validate_process_environment({{"AIP", "x"}, {"XAIP_MODE", "x"}}).ok());
    CHECK(decision_of(spawn_status({"-m", "m.gguf"}, env)) == Decision::refused);
#if defined(SONDER_HAS_SERVER)
    CHECK(decision_of(config_status(spawn_config(R"({"AIP_MODE":"PREDICTION","AIP_HTTP_PORT":"8080"})"))) ==
          Decision::refused);
#endif
    // Inherited: a Sonder process running in a Vertex AI container has
    // AIP_MODE=PREDICTION and AIP_HTTP_PORT set.
    ScopedParentEnv mode("AIP_MODE", std::string("PREDICTION"));
    ScopedParentEnv port("AIP_HTTP_PORT", std::string("8080"));
    {
        ls::ProcessSpec spec;
        spec.environment = {{"SONDER_CONTAINMENT_PROBE", "1"}};
        ls::ProcessOptions options;
        REQUIRE(options.prepare(spec).ok());
        const auto entries = prepared_entries(options);
        CHECK(values_named(entries, "SONDER_CONTAINMENT_PROBE") == std::vector<std::string>{"1"});
        CHECK(values_named(entries, "AIP_MODE").empty());
    }
    {
        // No overrides: the block must still be explicit so AIP_MODE is dropped.
        ls::ProcessSpec spec;
        ls::ProcessOptions options;
        REQUIRE(options.prepare(spec).ok());
        const auto entries = prepared_entries(options);
        CHECK_FALSE(entries.empty());
        CHECK(values_named(entries, "AIP_MODE").empty());
    }
}

TEST_CASE("containment gap: inherited LLAMA_ARG and credential variables reach the spawned child") {
    // GAP: the parent's whole environment is the child's. With no overrides,
    // prepare() leaves the block empty and the launcher passes nullptr /
    // environ (process.cpp CreateProcessW / posix_spawnp), i.e. full
    // inheritance; with overrides, prepare() merges them over a copy of the
    // parent's block and removes nothing. Inheritance itself is documented
    // ("empty inherits unchanged", llamaserver.hpp; launch-profiles.md on
    // LLAMA_ARG_SWA_FULL), but no document considers inherited equivalents of
    // the refused flags: LLAMA_API_KEY makes the child demand a key Sonder
    // never sends, and LLAMA_ARG_HF_REPO makes it download a model although
    // "profiles never download".
    ScopedParentEnv key("LLAMA_API_KEY", std::string("containment-parent-secret"));
    ScopedParentEnv repo("LLAMA_ARG_HF_REPO", std::string("containment/parent-repo"));
    ScopedParentEnv tools("LLAMA_ARG_TOOLS", std::string("all"));
    ScopedParentEnv config_home(
#if defined(_WIN32)
        "APPDATA",
#else
        "XDG_CONFIG_HOME",
#endif
        std::string("containment-config-home"));

    {
        ls::ProcessSpec spec;
        ls::ProcessOptions inherit_all;
        REQUIRE(inherit_all.prepare(spec).ok());
        // Empty block -> the launcher hands the child the parent's own block.
        CHECK(inherit_all.environment.empty());
    }
    {
        ls::ProcessSpec spec;
        spec.environment = {{"SONDER_CONTAINMENT_PROBE", "1"}};
        ls::ProcessOptions merged;
        REQUIRE(merged.prepare(spec).ok());
        const auto entries = prepared_entries(merged);
        CHECK(values_named(entries, "SONDER_CONTAINMENT_PROBE") == std::vector<std::string>{"1"});
        CHECK(values_named(entries, "LLAMA_API_KEY") == std::vector<std::string>{"containment-parent-secret"});
        CHECK(values_named(entries, "LLAMA_ARG_HF_REPO") == std::vector<std::string>{"containment/parent-repo"});
        CHECK(values_named(entries, "LLAMA_ARG_TOOLS") == std::vector<std::string>{"all"});
#if defined(_WIN32)
        CHECK(values_named(entries, "APPDATA") == std::vector<std::string>{"containment-config-home"});
#else
        CHECK(values_named(entries, "XDG_CONFIG_HOME") == std::vector<std::string>{"containment-config-home"});
#endif
    }
    {
        // An override can blank an inherited variable but cannot remove it:
        // the child still sees LLAMA_API_KEY set (to "").
        ls::ProcessSpec spec;
        spec.environment = {{"LLAMA_API_KEY", ""}};
        ls::ProcessOptions blanked;
        REQUIRE(blanked.prepare(spec).ok());
        CHECK(values_named(prepared_entries(blanked), "LLAMA_API_KEY") == std::vector<std::string>{""});
    }
    {
        // Windows names are case-insensitive: prepare() merges an override
        // spelled llama_api_key into the inherited LLAMA_API_KEY (one entry).
        // POSIX names are case-sensitive, so the child sees both.
        ls::ProcessSpec spec;
        spec.environment = {{"llama_api_key", "override"}};
        ls::ProcessOptions cased;
        REQUIRE(cased.prepare(spec).ok());
        const auto values = values_named(prepared_entries(cased), "LLAMA_API_KEY");
#if defined(_WIN32)
        CHECK(values == std::vector<std::string>{"override"});
#else
        CHECK(values == std::vector<std::string>{"containment-parent-secret"});
        CHECK(values_named(prepared_entries(cased), "llama_api_key") == std::vector<std::string>{"override"});
#endif
    }
}
