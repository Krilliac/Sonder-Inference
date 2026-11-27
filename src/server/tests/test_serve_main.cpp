// serve_main(): --help, usage errors (exit 2), --ready-file with an ephemeral
// port, the startup banner and access log, a port in use (exit 1), and
// graceful shutdown through the hook (exit 0, engine.stopped delivered,
// telemetry streams closed).
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "server_test_support.hpp"
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "src/serve_options.hpp"

using namespace server_test;
namespace json = sonder::inference::json;
namespace fs = std::filesystem;

namespace {

fs::path temp_path(const std::string& name) {
    return fs::temp_directory_path() / (si::make_id("sonder-serve-test") + "-" + name);
}

std::string read_file(const fs::path& p) {
    std::ifstream in(p, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

// Reads the ready file the server has just renamed into place. On Windows the
// rename keeps a handle with DELETE access on the file for a moment after the
// new name is visible (for milliseconds when the renaming thread is preempted
// right there, as on a loaded CI machine), and std::ifstream shares only read
// and write, so an open in that window fails with a sharing violation (errno
// EACCES) although the file is complete. A failed open is retried for up to
// `patience`; whatever is read once the file opens is returned unchanged, so a
// truncated or malformed file still fails the caller's parse.
std::string read_ready_file(const fs::path& p, std::chrono::milliseconds patience = std::chrono::milliseconds(5000)) {
    const auto deadline = std::chrono::steady_clock::now() + patience;
    for (;;) {
        std::ifstream in(p, std::ios::binary);
        if (in) {
            std::ostringstream ss;
            ss << in.rdbuf();
            return ss.str();
        }
        if (std::chrono::steady_clock::now() >= deadline) {
            return {};
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

// Joins the serve_main thread if a failed REQUIRE unwinds the test first. A
// joinable std::thread destroyed by unwinding calls std::terminate, which
// turns one failed assertion into a crash of the whole test binary.
struct JoinRunner {
    std::thread& thread;
    ~JoinRunner() {
        if (thread.joinable()) {
            srv::request_shutdown();
            thread.join();
        }
    }
};

int run(const std::vector<std::string>& args, std::string& out_text, std::string& err_text) {
    std::ostringstream out;
    std::ostringstream err;
    const int rc = srv::serve_main(args, out, err);
    out_text = out.str();
    err_text = err.str();
    return rc;
}

bool connection_refused(std::uint16_t port) {
    return !det::connect_tcp("127.0.0.1", port, std::chrono::milliseconds(500)).ok();
}

std::string minimal_gguf_header() {
    std::string body;
    const auto raw = [&body](const void* ptr, std::size_t n) { body.append(static_cast<const char*>(ptr), n); };
    const auto u32 = [&](std::uint32_t v) { raw(&v, 4); };
    const auto u64 = [&](std::uint64_t v) { raw(&v, 8); };
    const auto str = [&](const std::string& s) {
        u64(s.size());
        body.append(s);
    };
    const auto kv_u32 = [&](const std::string& key, std::uint32_t v) {
        str(key);
        u32(4u);
        u32(v);
    };
    const auto kv_str = [&](const std::string& key, const std::string& v) {
        str(key);
        u32(8u);
        str(v);
    };
    kv_str("general.architecture", "llama");
    kv_u32("llama.block_count", 4u);
    kv_u32("llama.embedding_length", 128u);
    kv_u32("llama.attention.head_count", 8u);
    kv_u32("llama.attention.head_count_kv", 2u);
    std::string head = "GGUF";
    const auto put = [&head](const void* ptr, std::size_t n) { head.append(static_cast<const char*>(ptr), n); };
    const std::uint32_t version = 3u;
    const std::uint64_t tensors = 0u;
    const std::uint64_t kvs = 5u;
    put(&version, 4);
    put(&tensors, 8);
    put(&kvs, 8);
    return head + body;
}

int build_cli(const std::vector<std::string>& args, det::ServeCliConfig& cfg, std::string& err_text) {
    det::ParsedServeArgs parsed;
    std::string parse_err;
    REQUIRE_MESSAGE(det::parse_serve_args(args, parsed, parse_err), parse_err);
    std::ostringstream err;
    auto log = std::make_shared<det::LockedLog>(err);
    const int rc = det::build_serve_options(parsed, log, cfg);
    log->close();
    err_text = err.str();
    return rc;
}

}  // namespace

TEST_CASE("serve_main: --help prints usage and exits 0") {
    std::string out;
    std::string err;
    CHECK(run({"--help"}, out, err) == 0);
    CHECK(out.find("sonder-infer serve") != std::string::npos);
    CHECK(out.find("--ready-file") != std::string::npos);
    CHECK(out.find("--cors-origin") != std::string::npos);
    CHECK(out.find("--backend-capacity") != std::string::npos);
    CHECK(out.find("--priority-admission auto|on|off") != std::string::npos);
    CHECK(out.find("--warmup-file") != std::string::npos);
}

TEST_CASE("llamaserver config: parses and validates warmup settings") {
    const auto config = temp_path("warmup.json");
    std::ofstream(config) << R"({
      "warmup": {
        "messages_file": "prefix.json",
        "chat_template_kwargs": {"enable_thinking": false, "reasoning_effort": "low"},
        "slots": [2, 7], "on_restart": false, "max_prefix_chars": 4096
      }
    })";
    sonder::inference::BackendSetup setup;
    REQUIRE(sonder::inference::load_llamaserver_config(config.string(), setup).ok());
    CHECK(setup.llamaserver_warmup_messages_file == "prefix.json");
    CHECK_FALSE(setup.llamaserver_warmup_all_slots);
    REQUIRE(setup.llamaserver_warmup_slots.size() == 2);
    CHECK(setup.llamaserver_warmup_slots[0] == 2);
    CHECK(setup.llamaserver_warmup_slots[1] == 7);
    CHECK_FALSE(setup.llamaserver_warmup_on_restart);
    CHECK(setup.llamaserver_warmup_max_prefix_chars == 4096);
    REQUIRE(setup.llamaserver_warmup_chat_template_kwargs.find("enable_thinking") != nullptr);
    CHECK_FALSE(setup.llamaserver_warmup_chat_template_kwargs.find("enable_thinking")->as_bool(true));
    std::error_code ec;
    fs::remove(config, ec);
}

TEST_CASE("llamaserver config: rejects malformed warmup settings") {
    const std::vector<std::string> documents = {
        R"({"warmup": {"slots": [1, 1]}})",
        R"({"warmup": {"slots": [1024]}})",
        R"({"warmup": {"slots": "some"}})",
        R"({"warmup": {"max_prefix_chars": 0}})",
        R"({"warmup": {"unknown": true}})"};
    for (std::size_t i = 0; i < documents.size(); ++i) {
        const auto config = temp_path("bad-warmup-" + std::to_string(i) + ".json");
        std::ofstream(config) << documents[i];
        sonder::inference::BackendSetup setup;
        CHECK_FALSE(sonder::inference::load_llamaserver_config(config.string(), setup).ok());
        std::error_code ec;
        fs::remove(config, ec);
    }
}

TEST_CASE("llamaserver config: slow-upstream attach fields are loaded") {
    const auto config = temp_path("slow-upstream.json");
    std::ofstream(config) << R"({
      "mode": "attach", "base_url": "http://127.0.0.1:8080", "native_completion": false,
      "api_key_env": "COLI_API_KEY", "keepalive_gap_ms": 10000, "idle_timeout_ms": 45000,
      "first_content_timeout_ms": 600000, "unlimited_request_timeout": true, "upstream_grammar": false,
      "stall_guard": {"generic_progress": true, "stall_seconds": 900, "policy": "cancel"}
    })";
    sonder::inference::BackendSetup setup;
    const auto st = sonder::inference::load_llamaserver_config(config.string(), setup);
    std::error_code ec;
    fs::remove(config, ec);
    REQUIRE_MESSAGE(st.ok(), st.message());
    CHECK(setup.llamaserver_api_key_env == "COLI_API_KEY");
    CHECK(setup.llamaserver_keepalive_gap_ms == 10000);
    CHECK(setup.llamaserver_idle_timeout_ms == 45000);
    CHECK(setup.llamaserver_first_content_timeout_ms == 600000);
    CHECK(setup.llamaserver_unlimited_request_timeout);
    CHECK_FALSE(setup.llamaserver_upstream_grammar);
    CHECK(setup.llamaserver_stall_generic_progress);
    CHECK(setup.llamaserver_stall_seconds == 900);
    CHECK(setup.llamaserver_stall_policy == "cancel");
    // Defaults stay off for configs that do not mention the new keys.
    sonder::inference::BackendSetup defaults;
    CHECK(defaults.llamaserver_idle_timeout_ms == 0);
    CHECK(defaults.llamaserver_first_content_timeout_ms == 0);
    CHECK_FALSE(defaults.llamaserver_unlimited_request_timeout);
    CHECK_FALSE(defaults.llamaserver_stall_generic_progress);
    CHECK(defaults.llamaserver_api_key_env.empty());
}

TEST_CASE("llamaserver config: rejects malformed slow-upstream fields") {
    const std::vector<std::string> documents = {
        R"({"api_key_env": 5})", R"({"idle_timeout_ms": 86400001})", R"({"idle_timeout_ms": "x"})",
        R"({"first_content_timeout_ms": -1})", R"({"first_content_timeout_ms": 86400001})",
        R"({"keepalive_gap_ms": 0})", R"({"keepalive_gap_ms": 3600001})",
        R"({"unlimited_request_timeout": 1})", R"({"upstream_grammar": "no"})",
        R"({"stall_guard": {"generic_progress": "yes"}})", R"({"stall_guard": {"policy": "kill"}})"};
    for (std::size_t i = 0; i < documents.size(); ++i) {
        const auto config = temp_path("bad-slow-" + std::to_string(i) + ".json");
        std::ofstream(config) << documents[i];
        sonder::inference::BackendSetup setup;
        INFO(documents[i]);
        CHECK_FALSE(sonder::inference::load_llamaserver_config(config.string(), setup).ok());
        std::error_code ec;
        fs::remove(config, ec);
    }
    // Positive control: a valid document loads, so the rejections above are not
    // caused by something unrelated in the harness.
    const auto config = temp_path("good-slow.json");
    std::ofstream(config) << R"({"idle_timeout_ms": 45000})";
    sonder::inference::BackendSetup setup;
    const auto st = sonder::inference::load_llamaserver_config(config.string(), setup);
    std::error_code ec;
    fs::remove(config, ec);
    CHECK_MESSAGE(st.ok(), st.message());
}

TEST_CASE("llamaserver config: idle and first-content timeouts accept 0 and up to 24 h") {
    const auto load = [](const std::string& document, sonder::inference::BackendSetup& setup) {
        const auto config = temp_path("slow-range.json");
        std::ofstream(config) << document;
        const auto st = sonder::inference::load_llamaserver_config(config.string(), setup);
        std::error_code ec;
        fs::remove(config, ec);
        return st;
    };
    {
        sonder::inference::BackendSetup setup;
        setup.llamaserver_idle_timeout_ms = 7;
        setup.llamaserver_first_content_timeout_ms = 7;
        const auto st = load(R"({"idle_timeout_ms": 0, "first_content_timeout_ms": 0})", setup);
        REQUIRE_MESSAGE(st.ok(), st.message());
        CHECK(setup.llamaserver_idle_timeout_ms == 0);
        CHECK(setup.llamaserver_first_content_timeout_ms == 0);
    }
    for (const std::uint64_t value : {std::uint64_t{7200000}, std::uint64_t{86400000}}) {
        sonder::inference::BackendSetup setup;
        const auto st = load("{\"first_content_timeout_ms\": " + std::to_string(value) + "}", setup);
        REQUIRE_MESSAGE(st.ok(), st.message());
        CHECK(setup.llamaserver_first_content_timeout_ms == value);
    }
}

TEST_CASE("llamaserver config: unknown-key errors bound the echoed key") {
    const std::string longkey(990, 'k');
    const std::vector<std::pair<std::string, std::string>> documents = {
        {"top-level", "{\"" + longkey + "\":1}"},
        {"tls", "{\"tls\":{\"" + longkey + "\":1}}"},
        {"spill_guard", "{\"spill_guard\":{\"" + longkey + "\":1}}"},
        {"stall_guard", "{\"stall_guard\":{\"" + longkey + "\":1}}"}};
    for (const auto& doc : documents) {
        const auto config = temp_path("long-key-" + doc.first + ".json");
        std::ofstream(config) << doc.second;
        sonder::inference::BackendSetup setup;
        const auto st = sonder::inference::load_llamaserver_config(config.string(), setup);
        std::error_code ec;
        fs::remove(config, ec);
        INFO(doc.first);
        REQUIRE_FALSE(st.ok());
        CHECK(st.message().size() < 128);
        CHECK(st.message().find(std::string(64, 'k') + "...'") != std::string::npos);
        CHECK(st.message().find(std::string(65, 'k')) == std::string::npos);
    }
}

TEST_CASE("serve_main: --warmup-file is llamaserver-only") {
    std::string out;
    std::string err;
    CHECK(run({"--backend", "mock", "--warmup-file", "prefix.json"}, out, err) == 2);
    CHECK(err.find("requires --backend llamaserver") != std::string::npos);
}

TEST_CASE("serve_main: usage errors exit 2") {
    std::string out;
    std::string err;
    const auto token = temp_path("token");
    {
        std::ofstream(token) << "tok\n";
    }
    for (const auto& args : std::vector<std::vector<std::string>>{
             {"--backend", "mock", "--bogus"},
             {"--backend", "mock", "stray"},
             {"--backend", "mock", "--port", "70000"},
             {"--backend", "mock", "--port", "12x"},
             {"--backend", "mock", "--port"},
             {"--backend", "mock", "--host", "0.0.0.0"},
             {"--backend", "mock", "--capture-text"},
             {"--backend", "mock", "--telemetry-level", "loud"},
             {"--backend", "mock", "--max-connections", "0"},
             {"--backend", "mock", "--log-format", "xml"},
             {"--backend", "mock", "--no-default-cors=1"},
             {"--backend", "mock", "--host", "a", "--host", "b"},
             {"--backend", "mock", "--backend-capacity", "not-a-number"},
             {"--backend", "mock", "--priority-admission", "sometimes"},
             {"--backend", "mock", "--token-file", (temp_path("missing")).string()},
             {"--backend", "ollama"},
             {"--backend", "mock", "--model", "default"},
             // Unknown backends are usage errors, caught before anything binds.
             {"--backend", "vllm", "--model", "x", "--port", "0"},
             {"--backend", "mock", "--cors-origin", "http://127.0.0.1:4173/", "--port", "0"},
             {"--backend", "mock", "--model-idle-ttl", "-1"},
             {"--backend", "mock", "--model-idle-ttl", "2592001"},
             {"--backend", "mock", "--max-resident-models", "x"},
             {"--backend", "mock", "--lazy-models=1"},
         }) {
        CAPTURE(args.back());
        CHECK(run(args, out, err) == 2);
        CHECK(err.find("error:") != std::string::npos);
    }
    {
        const auto ready = temp_path("never-ready.json");
        CHECK(run({"--backend", "vllm", "--model", "x", "--port", "0", "--ready-file", ready.string()}, out, err) ==
              2);
        CHECK(err.find("unknown or unavailable backend 'vllm'") != std::string::npos);
        CHECK_FALSE(fs::exists(ready));
    }
    std::error_code ec;
    fs::remove(token, ec);
}

TEST_CASE("serve_main: --profile enforces GPU/runtime env allowlist on --llamaserver-config") {
    const auto profiles = temp_path("profiles-env.json");
    const auto bad_cfg = temp_path("cfg-bad-env.json");
    const auto arbitrary_cfg = temp_path("cfg-arbitrary-env.json");
    {
        std::ofstream(profiles) << R"({"profiles":[{"name":"p1","backend":"llamaserver","model":"m.gguf"}]})";
        std::ofstream(bad_cfg) << R"({"mode":"spawn","executable":"C:/llama-server.exe","env":{"LLAMA_ARG_HF_REPO":"evil/repo"}})";
        std::ofstream(arbitrary_cfg) << R"({"mode":"spawn","executable":"C:/llama-server.exe","env":{"ARBITRARY_VAR":"1"}})";
    }
    std::string out, err;
    CHECK(run({"--profiles", profiles.string(), "--profile", "p1", "--llamaserver-config", bad_cfg.string()}, out, err) == 2);
    CHECK(err.find("refused") != std::string::npos);

    out.clear();
    err.clear();
    CHECK(run({"--profiles", profiles.string(), "--profile", "p1", "--llamaserver-config", arbitrary_cfg.string()}, out, err) == 2);
    CHECK(err.find("not permitted with --profile") != std::string::npos);

    std::error_code ec;
    fs::remove(profiles, ec);
    fs::remove(bad_cfg, ec);
    fs::remove(arbitrary_cfg, ec);
}

TEST_CASE("serve_main: rpc_worker in --llamaserver-config is refused with launch profiles") {
    // A profile's child serves its model, and in shared spawn mode several
    // profiles share it; the RPC client never runs there
    // (docs/integration/rpc-worker.md).
    const auto profiles = temp_path("profiles-rpc.json");
    const auto worker_cfg = temp_path("cfg-rpc-worker.json");
    const auto half_cfg = temp_path("cfg-rpc-servers-only.json");
    {
        std::ofstream(profiles) << R"({"profiles":[{"name":"p1","backend":"llamaserver","model":"m.gguf"},)"
                                   R"({"name":"p2","backend":"llamaserver","model":"m.gguf"}]})";
        std::ofstream(worker_cfg) << R"({"mode":"spawn","executable":"C:/llama-server.exe",)"
                                     R"("rpc_worker":{"enabled":true,"servers":["10.0.0.2:50052"]}})";
        std::ofstream(half_cfg) << R"({"mode":"spawn","executable":"C:/llama-server.exe",)"
                                   R"("rpc_worker":{"servers":["10.0.0.2:50052"]}})";
    }
    for (const std::vector<std::string>& selection :
         {std::vector<std::string>{"--profile", "p1"}, std::vector<std::string>{"--profile", "p1", "--profile", "p2"},
          std::vector<std::string>{"--all-profiles"}}) {
        for (const auto& cfg : {worker_cfg, half_cfg}) {
            CAPTURE(selection.size());
            CAPTURE(cfg.string());
            std::vector<std::string> args{"--profiles", profiles.string()};
            args.insert(args.end(), selection.begin(), selection.end());
            args.insert(args.end(), {"--llamaserver-config", cfg.string()});
            std::string out, err;
            CHECK(run(args, out, err) == 2);
            CHECK(err.find("'rpc_worker' cannot be combined with --profile") != std::string::npos);
        }
    }
    std::error_code ec;
    fs::remove(profiles, ec);
    fs::remove(worker_cfg, ec);
    fs::remove(half_cfg, ec);
}

TEST_CASE("serve_main: llama.cpp KV cache, flash attention and batch flags") {
    std::string out;
    std::string err;
    CHECK(run({"--help"}, out, err) == 0);
    for (const char* flag : {"--batch-size", "--ubatch-size", "--cache-type-k", "--cache-type-v", "--flash-attn"}) {
        CAPTURE(flag);
        CHECK(out.find(flag) != std::string::npos);
    }
    // Numeric flags are range-checked while parsing.
    for (const auto& args : std::vector<std::vector<std::string>>{
             {"--backend", "llamacpp", "--model", "x", "--batch-size", "0"},
             {"--backend", "llamacpp", "--model", "x", "--batch-size", "12x"},
             {"--backend", "llamacpp", "--model", "x", "--ubatch-size", "0"},
             {"--backend", "llamacpp", "--model", "x", "--ubatch-size", "5000000"},
         }) {
        CAPTURE(args.back());
        CHECK(run(args, out, err) == 2);
        CHECK(err.find("error:") != std::string::npos);
    }
    // Bad values and combinations are usage errors before anything binds.
    for (const auto& args : std::vector<std::vector<std::string>>{
             {"--backend", "llamacpp", "--model", "x", "--port", "0", "--cache-type-v", "q4_0", "--flash-attn", "off"},
             {"--backend", "llamacpp", "--model", "x", "--port", "0", "--cache-type-k", "q4_k"},
             {"--backend", "llamacpp", "--model", "x", "--port", "0", "--flash-attn", "maybe"},
             {"--backend", "llamacpp", "--model", "x", "--port", "0", "--batch-size", "256", "--ubatch-size", "512"},
         }) {
        CAPTURE(args.back());
        CHECK(run(args, out, err) == 2);
        CHECK(err.find("error:") != std::string::npos);
    }
#if defined(SONDER_HAS_LLAMACPP_BACKEND)
    CHECK(run({"--backend", "llamacpp", "--model", "x", "--port", "0", "--cache-type-v", "q8_0", "--flash-attn",
               "off"},
              out, err) == 2);
    CHECK(err.find("requires flash attention") != std::string::npos);
    CHECK(run({"--backend", "llamacpp", "--model", "x", "--port", "0", "--cache-type-k", "q4_k"}, out, err) == 2);
    CHECK(err.find("unknown K cache type 'q4_k'") != std::string::npos);
    CHECK(run({"--backend", "llamacpp", "--model", "x", "--port", "0", "--batch-size", "256", "--ubatch-size", "512"},
              out, err) == 2);
    CHECK(err.find("n_ubatch (512) must be <= n_batch (256)") != std::string::npos);
#endif
}

TEST_CASE("serve_main: llama.cpp context flags are refused for every other backend") {
    std::string out;
    std::string err;
    const std::vector<std::pair<std::string, std::string>> flags{{"--gpu-layers", "1"},
                                                                 {"--context-length", "1024"},
                                                                 {"--moe-experts", "cpu"},
                                                                 {"--tensor-override", "weight=cpu"},
                                                                 {"--batch-size", "1024"},
                                                                 {"--ubatch-size", "256"},
                                                                 {"--cache-type-k", "q8_0"},
                                                                 {"--cache-type-v", "q8_0"},
                                                                 {"--flash-attn", "on"}};
    // `--max-connections 0` is a later usage error: should the rule go
    // missing, serve still exits before it binds anything and the test fails
    // on the message instead of serving.
    for (const std::string backend : {"llamaserver", "mock", "ollama"}) {
        for (const auto& [flag, value] : flags) {
            CAPTURE(backend);
            CAPTURE(flag);
            CHECK(run({"--backend", backend, "--model", "x", flag, value, "--max-connections", "0"}, out, err) == 2);
            CHECK(err.find("error: option " + flag +
                           " applies only to --backend llamacpp; for llamaserver set it in the JSON args") !=
                  std::string::npos);
        }
    }
    // The llamacpp backend still takes them: the run gets past the flag and
    // stops at the later usage error.
    for (const auto& [flag, value] : flags) {
        CAPTURE(flag);
        CHECK(run({"--backend", "llamacpp", "--model", "x", flag, value, "--max-connections", "0"}, out, err) == 2);
        CHECK(err.find("applies only to --backend llamacpp") == std::string::npos);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
    }
}

TEST_CASE("serve_main: direct llamacpp performance CLI flags parse, validate and reject other backends") {
    std::string out;
    std::string err;
    CHECK(run({"--help"}, out, err) == 0);
    for (const char* flag : {"--threads", "--no-mmap", "--no-kv-unified", "--no-prefix-cache", "--slots",
                             "--spec-type", "--spec-draft-max"}) {
        CAPTURE(flag);
        CHECK(out.find(flag) != std::string::npos);
    }

    // Valid flags populate BackendSetup for llamacpp.
    det::ServeCliConfig cfg;
    const int rc = build_cli({"--backend", "llamacpp", "--model", "x", "--threads", "12", "--no-mmap",
                              "--no-kv-unified", "--no-prefix-cache", "--slots", "4", "--spec-type", "ngram-mod",
                              "--spec-draft-max", "8", "--max-connections", "0"},
                             cfg, err);
    CHECK(rc == 2);
    CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
    CHECK(cfg.options.backend.llamacpp_threads == std::optional<std::int32_t>(12));
    CHECK(cfg.options.backend.llamacpp_use_mmap == std::optional<bool>(false));
    CHECK(cfg.options.backend.llamacpp_kv_unified == std::optional<bool>(false));
    CHECK(cfg.options.backend.llamacpp_prefix_cache == std::optional<bool>(false));
    CHECK(cfg.options.backend.llamacpp_slots == std::optional<std::uint32_t>(4u));
    CHECK(cfg.options.backend.llamacpp_spec_type == std::optional<std::string>("ngram-mod"));
    CHECK(cfg.options.backend.llamacpp_spec_draft_max == std::optional<std::int32_t>(8));

    // Out-of-range / invalid values are usage errors.
    for (const auto& args : std::vector<std::vector<std::string>>{
             {"--backend", "llamacpp", "--model", "x", "--threads", "0"},
             {"--backend", "llamacpp", "--model", "x", "--threads", "257"},
             {"--backend", "llamacpp", "--model", "x", "--slots", "0"},
             {"--backend", "llamacpp", "--model", "x", "--slots", "17"},
             {"--backend", "llamacpp", "--model", "x", "--spec-type", "bogus"},
             {"--backend", "llamacpp", "--model", "x", "--spec-draft-max", "65"},
             {"--backend", "llamacpp", "--model", "x", "--no-mmap=1"},
             {"--backend", "llamacpp", "--model", "x", "--no-kv-unified=1"},
             {"--backend", "llamacpp", "--model", "x", "--no-prefix-cache=1"},
         }) {
        CAPTURE(args.back());
        CHECK(run(args, out, err) == 2);
        CHECK(err.find("error:") != std::string::npos);
    }

    // Refused on non-llamacpp backends.
    for (const auto& args : std::vector<std::vector<std::string>>{
             {"--backend", "mock", "--threads", "4", "--max-connections", "0"},
             {"--backend", "mock", "--no-mmap", "--max-connections", "0"},
             {"--backend", "mock", "--no-kv-unified", "--max-connections", "0"},
             {"--backend", "mock", "--no-prefix-cache", "--max-connections", "0"},
             {"--backend", "mock", "--slots", "2", "--max-connections", "0"},
             {"--backend", "mock", "--spec-type", "auto", "--max-connections", "0"},
             {"--backend", "mock", "--spec-draft-max", "4", "--max-connections", "0"},
         }) {
        CHECK(run(args, out, err) == 2);
        CHECK(err.find("applies only to --backend llamacpp") != std::string::npos);
    }
}

TEST_CASE("serve_main: ready file, banner, access log and graceful shutdown via the hook") {
    const auto ready = temp_path("ready.json");
    const auto events = temp_path("events.jsonl");
    std::ostringstream out;
    std::string err_text;  // written by the runner before it signals completion
    std::promise<int> done;
    auto finished = done.get_future();
    std::thread runner([&] {
        std::ostringstream local_err;
        const int rc = srv::serve_main({"--backend", "mock", "--model", "mock:tiny", "--port", "0", "--ready-file",
                                        ready.string(), "--telemetry", events.string(), "--shutdown-grace-ms", "1000"},
                                       out, local_err);
        err_text = local_err.str();
        done.set_value(rc);
    });
    JoinRunner join_runner{runner};
    REQUIRE(eventually([&] { return fs::exists(ready); }, std::chrono::milliseconds(10000)));
    auto doc = json::parse(read_ready_file(ready));
    REQUIRE(doc.ok());
    const std::string url = doc.value().find("url")->as_string();
    CHECK(url.rfind("http://127.0.0.1:", 0) == 0);
    CHECK(doc.value().find("pid")->as_int() > 0);
    CHECK(doc.value().find("instance_id")->as_string().rfind("tel-", 0) == 0);
    CHECK(doc.value().find("api_version")->as_int() == 1);
    const auto port = static_cast<std::uint16_t>(std::stoi(url.substr(url.rfind(':') + 1)));
    CHECK(port != 0);
    CHECK(port != srv::kDefaultPort);
    REQUIRE(eventually([&] { return get(port, "/v1/sonder/health").status == 200; }));

    // A second server on the same port fails with exit 1 and a hint.
    {
        std::string o2;
        std::string e2;
        CHECK(run({"--backend", "mock", "--port", std::to_string(port)}, o2, e2) == 1);
        CHECK(e2.find("--port 0") != std::string::npos);
    }

    // An open stream must see engine.stopped and then be closed by the server.
    Conn stream(port);
    stream.send(build_request("GET", "/v1/telemetry/sse?since=now", port));
    std::string acc;
    REQUIRE(stream.read_until(acc, "retry: 2000"));
    REQUIRE(post(port, "/v1/chat/completions", chat_body(R"(,"max_tokens":2)")).status == 200);

    srv::request_shutdown();
    REQUIRE(finished.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
    CHECK(finished.get() == 0);
    runner.join();
    acc += stream.read_all(std::chrono::milliseconds(5000));
    CHECK(stream.closed());
    CHECK(acc.find("\"engine.stopped\"") != std::string::npos);
    CHECK(connection_refused(port));

    const std::string& text = err_text;
    CHECK(text.find("listening on " + url) != std::string::npos);
    CHECK(text.find("MOCK BACKEND - synthetic output, not a quality or performance signal") != std::string::npos);
    CHECK(text.find("/v1/telemetry/sse") != std::string::npos);
    CHECK(text.find(" POST /v1/chat/completions 200 ") != std::string::npos);
    CHECK(text.find(" GET /v1/sonder/health 200 ") != std::string::npos);
    // Never bodies or header values.
    CHECK(text.find("hello sonder") == std::string::npos);
    CHECK(read_file(events).find("\"engine.stopped\"") != std::string::npos);
    // The listening line comes before models load; "ready" after.
    const auto loading = text.find("loading models on mock: mock:tiny (default)");
    CHECK(loading != std::string::npos);
    CHECK(text.find("ready on " + url) != std::string::npos);
    CHECK(loading < text.find("ready on " + url));
    // Discovery must never point at a stopped server.
    CHECK_FALSE(fs::exists(ready));
    std::error_code ec;
    fs::remove(events, ec);
}

TEST_CASE("serve_main: a failed start removes the ready file and exits 1") {
    // The mock backend refuses ids that do not start with "mock", after the
    // socket is listening and the ready file is written.
    const auto ready = temp_path("ready-fail.json");
    std::string out;
    std::string err;
    const int rc = run({"--backend", "mock", "--model", "mock:tiny", "--model", "not-a-mock-model", "--port", "0",
                        "--ready-file", ready.string()},
                       out, err);
    CHECK(err.find("cannot load model 'not-a-mock-model'") != std::string::npos);
    CHECK(rc == 1);
    CHECK(err.find("listening on http://127.0.0.1:") != std::string::npos);
    CHECK_FALSE(fs::exists(ready));
}

TEST_CASE("serve_main: a wildcard bind names the bind address and the local URL") {
    const auto ready = temp_path("ready-any.json");
    const auto token = temp_path("token-any");
    {
        std::ofstream(token) << "tok-any\n";
    }
    fs::permissions(token, fs::perms::owner_read | fs::perms::owner_write);
    std::ostringstream out;
    std::string err_text;
    std::promise<int> done;
    auto finished = done.get_future();
    std::thread runner([&] {
        std::ostringstream local_err;
        const int rc = srv::serve_main({"--backend", "mock", "--host", "0.0.0.0", "--port", "0", "--token-file",
                                        token.string(), "--ready-file", ready.string()},
                                       out, local_err);
        err_text = local_err.str();
        done.set_value(rc);
    });
    JoinRunner join_runner{runner};
    REQUIRE(eventually([&] { return fs::exists(ready); }, std::chrono::milliseconds(10000)));
    auto doc = json::parse(read_ready_file(ready));
    REQUIRE(doc.ok());
    const std::string url = doc.value().find("url")->as_string();
    const std::string port = url.substr(url.rfind(':') + 1);
    srv::request_shutdown();
    REQUIRE(finished.wait_for(std::chrono::seconds(20)) == std::future_status::ready);
    CHECK(finished.get() == 0);
    runner.join();
    CHECK(err_text.find("listening on 0.0.0.0:" + port + " (all interfaces; local URL " + url + ")") !=
          std::string::npos);
    CHECK_FALSE(fs::exists(ready));
    std::error_code ec;
    fs::remove(token, ec);
}

#if defined(_WIN32)
TEST_CASE("serve_main tests: a ready file held open for delete is read once it is released") {
    const auto path = temp_path("held-for-delete.json");
    {
        std::ofstream(path, std::ios::binary) << "{\"ok\":true}\n";
    }
    // What a rename leaves on the file for a moment: a handle with DELETE
    // access that shares everything.
    HANDLE held = ::CreateFileW(path.c_str(), DELETE, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                                OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    REQUIRE(held != INVALID_HANDLE_VALUE);
    {
        // The hazard itself: a plain std::ifstream cannot open the file now.
        std::ifstream in(path, std::ios::binary);
        CHECK_FALSE(in.is_open());
    }
    std::thread release([held] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        ::CloseHandle(held);
    });
    const std::string text = read_ready_file(path);
    release.join();
    CHECK(text == "{\"ok\":true}\n");
    std::error_code ec;
    fs::remove(path, ec);
}
#endif

TEST_CASE("serve_main: repeatable --profile and --all-profiles parsing and validation") {
    std::string out;
    std::string err;
    CHECK(run({"--help"}, out, err) == 0);
    CHECK(out.find("--all-profiles") != std::string::npos);
    CHECK(out.find("repeatable") != std::string::npos);

    {
        det::ParsedServeArgs parsed;
        std::string parse_err;
        REQUIRE(det::parse_serve_args({"--profiles", "p.json", "--profile", "alpha", "--profile", "beta"}, parsed,
                                      parse_err));
        CHECK(parsed.all("profile") == std::vector<std::string>{"alpha", "beta"});
    }
    {
        det::ParsedServeArgs parsed;
        std::string parse_err;
        REQUIRE(det::parse_serve_args({"--profiles", "p.json", "--all-profiles"}, parsed, parse_err));
        CHECK(parsed.flags.count("all-profiles") == 1u);
    }

    const auto model = temp_path("multi-parse.gguf");
    {
        std::ofstream(model, std::ios::binary) << minimal_gguf_header();
    }
    const std::string m = model.generic_string();
    const auto profiles = temp_path("multi-parse-profiles.json");
    {
        std::ofstream(profiles) << R"({"profiles":[
            {"name":"direct-a","backend":"llamacpp","model":")" << m << R"(","ctx_size":512,"n_gpu_layers":-1},
            {"name":"srv-b","backend":"llamaserver","model":")" << m << R"(","ctx_size":512,"n_gpu_layers":-1,"fit":false},
            {"name":"direct-c","backend":"llamacpp","model":")" << m << R"(","ctx_size":512,"n_gpu_layers":-1}
        ]})";
    }
    const std::string pf = profiles.string();

    CHECK(run({"--profiles", pf, "--profile", "direct-a", "--profile", "direct-a"}, out, err) == 2);
    CHECK(err.find("duplicate --profile 'direct-a'") != std::string::npos);

    CHECK(run({"--profiles", pf, "--all-profiles", "--profile", "direct-a"}, out, err) == 2);
    CHECK(err.find("--all-profiles cannot be combined with --profile") != std::string::npos);

    CHECK(run({"--all-profiles"}, out, err) == 2);
    CHECK(err.find("--all-profiles needs --profiles") != std::string::npos);

    CHECK(run({"--profiles", pf, "--profile", "direct-a", "--profile", "srv-b"}, out, err) == 2);
    CHECK(err.find("cannot mix incompatible backends") != std::string::npos);

    CHECK(run({"--profiles", pf, "--all-profiles", "--backend", "mock"}, out, err) == 2);
    CHECK(err.find("conflicts with --all-profiles") != std::string::npos);

    // --all-profiles selects all profiles sharing the primary profile's backend
    // (llamacpp: direct-a and direct-c) and skips incompatible backends cleanly.
    det::ServeCliConfig cfg;
    const int rc =
        build_cli({"--profiles", pf, "--all-profiles", "--vram-budget-mib", "10000", "--max-connections", "0"}, cfg,
                  err);
    CHECK(rc == 2);
    CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
    CHECK(cfg.options.backend.backend == "llamacpp");
    CHECK(cfg.options.models == std::vector<std::string>{"direct-a", "direct-c"});
    REQUIRE(cfg.options.profile_bindings.size() == 2u);
    CHECK(cfg.options.profile_bindings[0].model == "direct-a");
    CHECK(cfg.options.profile_bindings[1].model == "direct-c");
    REQUIRE(cfg.options.profile_catalog.size() == 3u);
    CHECK(cfg.options.profile_catalog[0].find("served")->as_bool());
    CHECK_FALSE(cfg.options.profile_catalog[1].find("served")->as_bool());
    CHECK(cfg.options.profile_catalog[2].find("served")->as_bool());
    bool skipped_noted = false;
    for (const auto& note : cfg.profile_notes) {
        if (note.find("skipping launch profile 'srv-b'") != std::string::npos) skipped_noted = true;
    }
    CHECK(skipped_noted);

    std::error_code ec;
    fs::remove(model, ec);
    fs::remove(profiles, ec);
}

TEST_CASE("serve_main: multi-profile llamaserver attach mode binds profiles while spawn rejects distinct models") {
    const auto m1_path = temp_path("bonsai-27b.gguf");
    const auto m2_path = temp_path("bonsai-9b.gguf");
    {
        const std::string gguf = minimal_gguf_header();
        std::ofstream(m1_path, std::ios::binary) << gguf;
        std::ofstream(m2_path, std::ios::binary) << gguf;
    }
    const std::string m1 = m1_path.generic_string();
    const std::string m2 = m2_path.generic_string();
    const auto profiles = temp_path("bonsai-profiles.json");
    {
        std::ofstream(profiles) << R"({"profiles":[
            {"name":"bonsai-27b","backend":"llamaserver","model":")" << m1
                                << R"(","ctx_size":512,"n_gpu_layers":-1,"fit":false,"sampling":{"temperature":0.3}},
            {"name":"bonsai-9b","backend":"llamaserver","model":")" << m2
                                << R"(","ctx_size":512,"n_gpu_layers":-1,"fit":false,"sampling":{"temperature":0.8}},
            {"name":"bonsai-27b-creative","backend":"llamaserver","model":")" << m1
                                << R"(","ctx_size":512,"n_gpu_layers":-1,"fit":false,"sampling":{"temperature":1.1}},
            {"name":"bonsai-27b-wide","backend":"llamaserver","model":")" << m1
                                << R"(","ctx_size":1024,"n_gpu_layers":-1,"fit":false}
        ]})";
    }
    const std::string pf = profiles.string();

    const auto loaded_profiles = si::load_launch_profiles(pf);
    REQUIRE_MESSAGE(loaded_profiles.ok(), loaded_profiles.status().message());
    const auto parsed_gguf = si::read_gguf_model_info(m1);
    REQUIRE_MESSAGE(parsed_gguf.ok(), parsed_gguf.status().message());
    const std::uint64_t single_mib = si::estimate_vram(parsed_gguf.value(), loaded_profiles.value()[0]).total_mib();
    const std::uint64_t tight_budget = single_mib + 50u;
    const std::uint64_t fit_budget = 2u * single_mib + 50u;
    REQUIRE(tight_budget < 2u * single_mib);

    std::string out;
    std::string err;

    // Spawn mode with two distinct GGUFs refuses with actionable guidance.
    CHECK(run({"--profiles", pf, "--profile", "bonsai-27b", "--profile", "bonsai-9b", "--llamaserver-executable",
               "llama-server-not-started"},
              out, err) == 2);
    CHECK(err.find("cannot spawn multiple distinct GGUF models") != std::string::npos);
    CHECK(err.find("--llamaserver-mode attach") != std::string::npos);
    CHECK(err.find("separate") != std::string::npos);

    // Explicit spawn mode also refuses --llamaserver-url.
    CHECK(run({"--profiles", pf, "--profile", "bonsai-27b", "--profile", "bonsai-9b", "--llamaserver-mode", "spawn",
               "--llamaserver-url", "http://127.0.0.1:8080"},
              out, err) == 2);
    CHECK(err.find("--llamaserver-url conflicts with --profile in llamaserver spawn mode") != std::string::npos);

    // Shared-model spawn mode with conflicting launch settings refuses cleanly.
    CHECK(run({"--profiles", pf, "--profile", "bonsai-27b", "--profile", "bonsai-27b-wide", "--llamaserver-executable",
               "llama-server-not-started"},
              out, err) == 2);
    CHECK(err.find("conflicting launch settings") != std::string::npos);

    // Shared-model spawn mode with matching launch settings (only sampling differs)
    // binds both aliases to the single spawned child without double-counting VRAM.
    {
        det::ServeCliConfig cfg_spawn;
        const int rc = build_cli({"--profiles", pf, "--profile", "bonsai-27b", "--profile", "bonsai-27b-creative",
                                  "--llamaserver-executable", "llama-server-not-started", "--vram-budget-mib",
                                  std::to_string(tight_budget), "--max-connections", "0"},
                                 cfg_spawn, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
        CHECK(cfg_spawn.options.backend.llamaserver_mode == "spawn");
        CHECK(cfg_spawn.options.models == std::vector<std::string>{"bonsai-27b", "bonsai-27b-creative"});
        REQUIRE(cfg_spawn.options.profile_bindings.size() == 2u);
        CHECK(cfg_spawn.options.profile_bindings[0].load_name == "bonsai-27b");
        CHECK(cfg_spawn.options.profile_bindings[1].load_name == "bonsai-27b");
        CHECK(*cfg_spawn.options.profile_bindings[0].sampling_defaults.temperature == doctest::Approx(0.3));
        CHECK(*cfg_spawn.options.profile_bindings[1].sampling_defaults.temperature == doctest::Approx(1.1));
    }

    // Attach mode against an upstream router binds multiple distinct profiles cleanly.
    {
        det::ServeCliConfig cfg_attach;
        const int rc = build_cli({"--profiles", pf, "--profile", "bonsai-27b", "--profile", "bonsai-9b",
                                  "--llamaserver-mode", "attach", "--llamaserver-url", "http://127.0.0.1:8080",
                                  "--vram-budget-mib", std::to_string(fit_budget), "--max-connections", "0"},
                                 cfg_attach, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
        CHECK(cfg_attach.options.backend.llamaserver_mode == "attach");
        CHECK(cfg_attach.options.backend.llamaserver_url == "http://127.0.0.1:8080");
        CHECK(cfg_attach.options.models == std::vector<std::string>{"bonsai-27b", "bonsai-9b"});
        REQUIRE(cfg_attach.options.profile_bindings.size() == 2u);
        CHECK(cfg_attach.options.profile_bindings[0].load_name == "bonsai-27b");
        CHECK(cfg_attach.options.profile_bindings[1].load_name == "bonsai-9b");
        CHECK(*cfg_attach.options.profile_bindings[0].sampling_defaults.temperature == doctest::Approx(0.3));
        CHECK(*cfg_attach.options.profile_bindings[1].sampling_defaults.temperature == doctest::Approx(0.8));
        REQUIRE(cfg_attach.options.profile_catalog.size() == 4u);
        CHECK(cfg_attach.options.profile_catalog[0].find("served")->as_bool());
        CHECK(cfg_attach.options.profile_catalog[1].find("served")->as_bool());
        CHECK_FALSE(cfg_attach.options.profile_catalog[2].find("served")->as_bool());
        CHECK_FALSE(cfg_attach.options.profile_catalog[3].find("served")->as_bool());
    }

    // Passing --llamaserver-url with multiple profiles infers attach mode.
    {
        det::ServeCliConfig cfg_inferred;
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        const int rc = build_cli({"--profiles", pf, "--profile", "bonsai-27b", "--profile", "bonsai-9b",
                                  "--llamaserver-url", "http://127.0.0.1:8080", "--vram-budget-mib",
                                  std::to_string(fit_budget)},
                                 cfg_inferred, err);
        CHECK_MESSAGE(rc == 0, err);
#else
        const int rc = build_cli({"--profiles", pf, "--profile", "bonsai-27b", "--profile", "bonsai-9b",
                                  "--llamaserver-url", "http://127.0.0.1:8080", "--vram-budget-mib",
                                  std::to_string(fit_budget), "--max-connections", "0"},
                                 cfg_inferred, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
#endif
        CHECK(cfg_inferred.options.backend.llamaserver_mode == "attach");
        CHECK(cfg_inferred.options.models == std::vector<std::string>{"bonsai-27b", "bonsai-9b"});
    }

    std::error_code ec;
    fs::remove(m1_path, ec);
    fs::remove(m2_path, ec);
    fs::remove(profiles, ec);
}

TEST_CASE("serve_main: multi-profile llamacpp validates shared options and checks aggregate VRAM") {
    const auto a_path = temp_path("direct-a.gguf");
    const auto b_path = temp_path("direct-b.gguf");
    {
        const std::string gguf = minimal_gguf_header();
        std::ofstream(a_path, std::ios::binary) << gguf;
        std::ofstream(b_path, std::ios::binary) << gguf;
    }
    const std::string a_str = a_path.generic_string();
    const std::string b_str = b_path.generic_string();
    const auto profiles = temp_path("direct-multi-profiles.json");
    {
        std::ofstream(profiles) << R"({"profiles":[
            {"name":"alpha","backend":"llamacpp","model":")" << a_str
                                << R"(","ctx_size":512,"n_gpu_layers":-1,"cache_type_k":"q8_0","sampling":{"temperature":0.2}},
            {"name":"beta","backend":"llamacpp","model":")" << b_str
                                << R"(","ctx_size":512,"n_gpu_layers":-1,"cache_type_k":"q8_0","sampling":{"temperature":0.9}},
            {"name":"conflict","backend":"llamacpp","model":")" << b_str
                                << R"(","ctx_size":1024,"n_gpu_layers":-1},
            {"name":"unsupported","backend":"llamacpp","model":")" << b_str
                                << R"(","threads_batch":8}
        ]})";
    }
    const std::string pf = profiles.string();

    const auto loaded_profiles = si::load_launch_profiles(pf);
    REQUIRE_MESSAGE(loaded_profiles.ok(), loaded_profiles.status().message());
    const auto parsed_gguf = si::read_gguf_model_info(a_str);
    REQUIRE_MESSAGE(parsed_gguf.ok(), parsed_gguf.status().message());
    const std::uint64_t single_mib = si::estimate_vram(parsed_gguf.value(), loaded_profiles.value()[0]).total_mib();
    const std::uint64_t tight_budget = single_mib + 50u;
    const std::uint64_t fit_budget = 2u * single_mib + 50u;
    REQUIRE(tight_budget < 2u * single_mib);

    std::string out;
    std::string err;

    // Conflicting shared llamacpp option is refused clearly.
    CHECK(run({"--profiles", pf, "--profile", "alpha", "--profile", "conflict", "--vram-budget-mib",
               std::to_string(fit_budget)},
              out, err) == 2);
    CHECK(err.find("conflicts with 'alpha' on shared llamacpp option 'ctx_size'") != std::string::npos);

    // Unsupported llamacpp field on secondary profile is refused cleanly.
    CHECK(run({"--profiles", pf, "--profile", "alpha", "--profile", "unsupported", "--vram-budget-mib",
               std::to_string(fit_budget)},
              out, err) == 2);
    CHECK(err.find("'threads_batch' is not supported by the llamacpp backend") != std::string::npos);

    // Eager multi-profile overcommit refuses with both profile names, sum vs budget, and suggestions.
    CHECK(run({"--profiles", pf, "--profile", "alpha", "--profile", "beta", "--vram-budget-mib",
               std::to_string(tight_budget)},
              out, err) == 2);
    CHECK(err.find("aggregate VRAM") != std::string::npos);
    CHECK(err.find("alpha") != std::string::npos);
    CHECK(err.find("beta") != std::string::npos);
    CHECK(err.find(std::to_string(2u * single_mib) + " MiB vs budget " + std::to_string(tight_budget) + " MiB") !=
          std::string::npos);
    CHECK(err.find("--lazy-models --max-resident-models 1") != std::string::npos);
    CHECK(err.find("--allow-overcommit") != std::string::npos);

    // --lazy-models --max-resident-models 1 keeps peak resident to 1 model and passes at tight_budget.
    {
        det::ServeCliConfig cfg_lazy;
        const int rc = build_cli({"--profiles", pf, "--profile", "alpha", "--profile", "beta", "--vram-budget-mib",
                                  std::to_string(tight_budget), "--lazy-models", "--max-resident-models", "1",
                                  "--max-connections", "0"},
                                 cfg_lazy, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
        CHECK(cfg_lazy.options.models == std::vector<std::string>{"alpha", "beta"});
        REQUIRE(cfg_lazy.options.profile_bindings.size() == 2u);
        CHECK(cfg_lazy.options.profile_bindings[0].load_name == a_str);
        CHECK(cfg_lazy.options.profile_bindings[1].load_name == b_str);
    }

    // --allow-overcommit allows eager start even when aggregate sum exceeds budget.
    {
        det::ServeCliConfig cfg_over;
        const int rc = build_cli({"--profiles", pf, "--profile", "alpha", "--profile", "beta", "--vram-budget-mib",
                                  std::to_string(tight_budget), "--allow-overcommit", "--max-connections", "0"},
                                 cfg_over, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
    }

    // Budget >= sum passes eagerly and marks both active profiles served in the catalog.
    {
        det::ServeCliConfig cfg_fit;
        const int rc = build_cli({"--profiles", pf, "--profile", "alpha", "--profile", "beta", "--vram-budget-mib",
                                  std::to_string(fit_budget), "--max-connections", "0"},
                                 cfg_fit, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
        CHECK(cfg_fit.options.models == std::vector<std::string>{"alpha", "beta"});
        REQUIRE(cfg_fit.options.profile_catalog.size() == 4u);
        CHECK(cfg_fit.options.profile_catalog[0].find("served")->as_bool());
        CHECK(cfg_fit.options.profile_catalog[1].find("served")->as_bool());
        CHECK_FALSE(cfg_fit.options.profile_catalog[2].find("served")->as_bool());
        CHECK_FALSE(cfg_fit.options.profile_catalog[3].find("served")->as_bool());
    }

    std::error_code ec;
    fs::remove(a_path, ec);
    fs::remove(b_path, ec);
    fs::remove(profiles, ec);
}

TEST_CASE("serve_main: multi-profile llamacpp applies merged shared options to VRAM estimate and metadata") {
    // 32 layers, 16 KV heads, head_dim 128, training ctx 4096:
    //   unmerged (ctx 4096, f16 K/V) ~ 1.3 GiB vs merged (ctx 32768, q8_0 K) ~ 6.5 GiB.
    const auto make_large_kv_gguf = []() {
        std::string body;
        const auto raw = [&body](const void* ptr, std::size_t n) { body.append(static_cast<const char*>(ptr), n); };
        const auto u32 = [&](std::uint32_t v) { raw(&v, 4); };
        const auto u64 = [&](std::uint64_t v) { raw(&v, 8); };
        const auto str = [&](const std::string& s) {
            u64(s.size());
            body.append(s);
        };
        const auto kv_u32 = [&](const std::string& key, std::uint32_t v) {
            str(key);
            u32(4u);
            u32(v);
        };
        const auto kv_str = [&](const std::string& key, const std::string& v) {
            str(key);
            u32(8u);
            str(v);
        };
        kv_str("general.architecture", "llama");
        kv_u32("llama.block_count", 32u);
        kv_u32("llama.embedding_length", 2048u);
        kv_u32("llama.context_length", 4096u);
        kv_u32("llama.attention.head_count", 16u);
        kv_u32("llama.attention.head_count_kv", 16u);
        std::string head = "GGUF";
        const auto put = [&head](const void* ptr, std::size_t n) { head.append(static_cast<const char*>(ptr), n); };
        const std::uint32_t version = 3u;
        const std::uint64_t tensors = 0u;
        const std::uint64_t kvs = 6u;
        put(&version, 4);
        put(&tensors, 8);
        put(&kvs, 8);
        return head + body;
    };

    const auto a_path = temp_path("merged-a.gguf");
    const auto b_path = temp_path("merged-b.gguf");
    {
        const std::string gguf = make_large_kv_gguf();
        std::ofstream(a_path, std::ios::binary) << gguf;
        std::ofstream(b_path, std::ios::binary) << gguf;
    }
    const std::string a_str = a_path.generic_string();
    const std::string b_str = b_path.generic_string();
    const auto profiles = temp_path("merged-llamacpp-profiles.json");
    {
        std::ofstream(profiles) << R"({"profiles":[
            {"name":"tuned","backend":"llamacpp","model":")" << a_str
                                << R"(","ctx_size":32768,"n_gpu_layers":-1,"cache_type_k":"q8_0"},
            {"name":"minimal","backend":"llamacpp","model":")" << b_str << R"("},
            {"name":"minimal2","backend":"llamacpp","model":")" << b_str << R"("},
            {"name":"cpu-only","backend":"llamacpp","model":")" << a_str << R"(","n_gpu_layers":0}
        ]})";
    }
    const std::string pf = profiles.string();

    const auto loaded_profiles = si::load_launch_profiles(pf);
    REQUIRE_MESSAGE(loaded_profiles.ok(), loaded_profiles.status().message());
    const auto parsed_gguf = si::read_gguf_model_info(a_str);
    REQUIRE_MESSAGE(parsed_gguf.ok(), parsed_gguf.status().message());
    const std::uint64_t merged_mib = si::estimate_vram(parsed_gguf.value(), loaded_profiles.value()[0]).total_mib();
    const std::uint64_t unmerged_mib = si::estimate_vram(parsed_gguf.value(), loaded_profiles.value()[1]).total_mib();
    REQUIRE(merged_mib > unmerged_mib + 2000u);
    // Midpoint budget: comfortably above (tuned + unmerged minimal), strictly below (tuned + merged minimal).
    const std::uint64_t mid_budget = merged_mib + unmerged_mib + 500u;
    const std::uint64_t fit_budget = 2u * merged_mib + 500u;
    REQUIRE(mid_budget < 2u * merged_mib);

    std::string out;
    std::string err;

    // Primary 'tuned' sets ctx_size/cache_type_k/n_gpu_layers; secondary 'minimal' leaves them unset.
    // Aggregate VRAM MUST count 'minimal' at merged_mib and refuse mid_budget.
    CHECK(run({"--profiles", pf, "--profile", "tuned", "--profile", "minimal", "--vram-budget-mib",
               std::to_string(mid_budget)},
              out, err) == 2);
    CHECK(err.find("aggregate VRAM") != std::string::npos);
    CHECK(err.find(std::to_string(2u * merged_mib) + " MiB vs budget " + std::to_string(mid_budget) + " MiB") !=
          std::string::npos);

    // At fit_budget, both binding metadata (/v1/models sonder.profile) and catalog metadata
    // (sonder.profiles) for 'minimal' reflect the merged effective options and VRAM estimate.
    {
        det::ServeCliConfig cfg;
        const int rc = build_cli({"--profiles", pf, "--profile", "tuned", "--profile", "minimal", "--vram-budget-mib",
                                  std::to_string(fit_budget), "--max-connections", "0"},
                                 cfg, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
        REQUIRE(cfg.options.profile_bindings.size() == 2u);
        const auto& b_meta = cfg.options.profile_bindings[1].metadata;
        REQUIRE(b_meta.find("context_length") != nullptr);
        CHECK(b_meta.find("context_length")->as_int() == 32768);
        REQUIRE(b_meta.find("cache_type_k") != nullptr);
        CHECK(b_meta.find("cache_type_k")->as_string() == "q8_0");
        REQUIRE(b_meta.find("estimated_vram_mib") != nullptr);
        CHECK(b_meta.find("estimated_vram_mib")->as_uint() == merged_mib);

        REQUIRE(cfg.options.profile_catalog.size() == 4u);
        const auto& cat_b = cfg.options.profile_catalog[1];
        CHECK(cat_b.find("served")->as_bool());
        CHECK(cat_b.find("context_length")->as_int() == 32768);
        CHECK(cat_b.find("cache_type_k")->as_string() == "q8_0");
        CHECK(cat_b.find("estimated_vram_mib")->as_uint() == merged_mib);

        bool noted_inherit_ctx = false;
        bool noted_inherit_k = false;
        for (const auto& note : cfg.profile_notes) {
            if (note.find("'minimal' inherits shared llamacpp option 'ctx_size' = 32768 from 'tuned'") !=
                std::string::npos) {
                noted_inherit_ctx = true;
            }
            if (note.find("'minimal' inherits shared llamacpp option 'cache_type_k' = q8_0 from 'tuned'") !=
                std::string::npos) {
                noted_inherit_k = true;
            }
        }
        CHECK(noted_inherit_ctx);
        CHECK(noted_inherit_k);
    }

    // Reverse order with 3 profiles: 'minimal' and 'minimal2' omit options, third profile 'tuned' sets them.
    // All three reflect merged effective options, and diagnostics accurately name both earlier profiles.
    {
        det::ServeCliConfig cfg;
        const std::uint64_t triple_budget = 3u * merged_mib + 500u;
        const int rc = build_cli({"--profiles", pf, "--profile", "minimal", "--profile", "minimal2", "--profile",
                                  "tuned", "--vram-budget-mib", std::to_string(triple_budget), "--max-connections",
                                  "0"},
                                 cfg, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
        REQUIRE(cfg.options.profile_bindings.size() == 3u);
        for (std::size_t i = 0; i < 3u; ++i) {
            CHECK(cfg.options.profile_bindings[i].metadata.find("context_length")->as_int() == 32768);
            CHECK(cfg.options.profile_bindings[i].metadata.find("cache_type_k")->as_string() == "q8_0");
            CHECK(cfg.options.profile_bindings[i].metadata.find("estimated_vram_mib")->as_uint() == merged_mib);
        }
        bool noted_both_earlier = false;
        for (const auto& note : cfg.profile_notes) {
            if (note.find("'tuned' sets shared llamacpp option 'ctx_size' = 32768 (unset on 'minimal', 'minimal2')") !=
                std::string::npos) {
                noted_both_earlier = true;
            }
        }
        CHECK(noted_both_earlier);
    }

    // CPU-only propagation: primary sets n_gpu_layers=0, secondary omits it; even on --device gpu:0
    // with a 1 MiB budget, effective n_gpu_layers=0 skips GPU offload for both profiles.
    {
        det::ServeCliConfig cfg_cpu;
        const int rc = build_cli({"--profiles", pf, "--profile", "cpu-only", "--profile", "minimal", "--device",
                                  "gpu:0", "--vram-budget-mib", "1", "--max-connections", "0"},
                                 cfg_cpu, err);
        CHECK(rc == 2);
        CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
        REQUIRE(cfg_cpu.options.profile_bindings.size() == 2u);
        CHECK(cfg_cpu.options.profile_bindings[0].metadata.find("estimated_vram_mib")->is_null());
        CHECK(cfg_cpu.options.profile_bindings[1].metadata.find("estimated_vram_mib")->is_null());
    }

    // Consistency checks: --vram-budget-mib without profiles mentions --all-profiles, and
    // llamaserver config base_url conflicts with multi-profile spawn mode.
    CHECK(run({"--backend", "mock", "--vram-budget-mib", "100"}, out, err) == 2);
    CHECK(err.find("--all-profiles") != std::string::npos);

    const auto ls_profiles = temp_path("ls-spawn-url-profiles.json");
    const auto ls_cfg = temp_path("ls-spawn-url-cfg.json");
    {
        std::ofstream(ls_profiles) << R"({"profiles":[
            {"name":"s1","backend":"llamaserver","model":")" << a_str << R"("},
            {"name":"s2","backend":"llamaserver","model":")" << a_str << R"("}
        ]})";
        std::ofstream(ls_cfg) << R"({"mode":"spawn","base_url":"http://127.0.0.1:8080","executable":"llama-server"})";
    }
    CHECK(run({"--profiles", ls_profiles.string(), "--profile", "s1", "--profile", "s2", "--llamaserver-config",
               ls_cfg.string()},
              out, err) == 2);
    CHECK(err.find("base_url") != std::string::npos);
    CHECK(err.find("conflicts with --profile in llamaserver spawn mode") != std::string::npos);

    std::error_code ec;
    fs::remove(a_path, ec);
    fs::remove(b_path, ec);
    fs::remove(profiles, ec);
    fs::remove(ls_profiles, ec);
    fs::remove(ls_cfg, ec);
}

TEST_CASE("serve_main: --stream-keepalive-ms parses, validates bounds, and appears in help") {
    std::string out;
    std::string err;
    CHECK(run({"--help"}, out, err) == 0);
    CHECK(out.find("--stream-keepalive-ms") != std::string::npos);

    {
        det::ServeCliConfig cfg;
        const int rc = build_cli({"--backend", "mock", "--stream-keepalive-ms", "0"}, cfg, err);
        CHECK_MESSAGE(rc == 0, err);
        CHECK(cfg.options.stream_keepalive_interval == std::chrono::milliseconds(0));
    }
    {
        det::ServeCliConfig cfg;
        const int rc = build_cli({"--backend", "mock", "--stream-keepalive-ms", "600000"}, cfg, err);
        CHECK_MESSAGE(rc == 0, err);
        CHECK(cfg.options.stream_keepalive_interval == std::chrono::milliseconds(600000));
    }
    for (const char* bad_val : {"-1", "600001", "abc", ""}) {
        CAPTURE(bad_val);
        CHECK(run({"--backend", "mock", "--stream-keepalive-ms", bad_val}, out, err) == 2);
        CHECK(err.find("--stream-keepalive-ms") != std::string::npos);
    }

    srv::ServerOptions opts;
    opts.backend.backend = "mock";
    opts.stream_keepalive_interval = std::chrono::milliseconds(-5);
    const si::Status st = srv::validate_options(opts);
    CHECK_FALSE(st.ok());
    CHECK(st.message().find("--stream-keepalive-ms") != std::string::npos);
}

TEST_CASE("serve_main: multi-profile llamacpp merges and conflicts on direct parity fields cleanly") {
    const auto g_path = temp_path("parity-direct.gguf");
    {
        std::string head = "GGUF";
        const std::uint32_t version = 3u;
        const std::uint64_t tensors = 0u;
        const std::uint64_t kvs = 1u;
        head.append(reinterpret_cast<const char*>(&version), 4);
        head.append(reinterpret_cast<const char*>(&tensors), 8);
        head.append(reinterpret_cast<const char*>(&kvs), 8);
        const std::string key = "general.architecture";
        const std::uint64_t kl = key.size();
        const std::uint32_t ty = 8u;
        const std::string val = "llama";
        const std::uint64_t vl = val.size();
        head.append(reinterpret_cast<const char*>(&kl), 8);
        head.append(key);
        head.append(reinterpret_cast<const char*>(&ty), 4);
        head.append(reinterpret_cast<const char*>(&vl), 8);
        head.append(val);
        std::ofstream(g_path, std::ios::binary) << head;
    }
    const std::string g_str = g_path.generic_string();
    const auto profiles = temp_path("parity-direct-profiles.json");
    {
        std::ofstream(profiles) << R"({"profiles":[
            {"name":"rich","backend":"llamacpp","model":")" << g_str << R"(",
             "n_gpu_layers":0,"parallel":3,"kv_unified":false,"cache_ram_mib":4096,
             "threads":8,"mmproj":"mm.gguf","speculative":{"types":["draft-mtp"],"draft_n_max":4}},
            {"name":"plain","backend":"llamacpp","model":")" << g_str << R"(","n_gpu_layers":0},
            {"name":"conflict-slots","backend":"llamacpp","model":")" << g_str << R"(","n_gpu_layers":0,"parallel":5},
            {"name":"conflict-spec","backend":"llamacpp","model":")" << g_str << R"(","n_gpu_layers":0,"speculative":{"types":["suffix"]}}
        ]})";
    }
    const std::string pf = profiles.string();

    std::string out;
    std::string err;
    CHECK(run({"--profiles", pf, "--profile", "rich", "--profile", "conflict-slots"}, out, err) == 2);
    CHECK(err.find("conflicts with 'rich' on shared llamacpp option 'parallel'") != std::string::npos);

    out.clear();
    err.clear();
    CHECK(run({"--profiles", pf, "--profile", "rich", "--profile", "conflict-spec"}, out, err) == 2);
    CHECK(err.find("conflicts with 'rich' on shared llamacpp option 'speculative.types'") != std::string::npos);

    det::ServeCliConfig cfg;
    const int rc = build_cli({"--profiles", pf, "--profile", "rich", "--profile", "plain", "--max-connections", "0"},
                             cfg, err);
    CHECK(rc == 2);
    CHECK(err.find("--max-connections must be from 1 to 100000") != std::string::npos);
    CHECK(cfg.options.backend.llamacpp_slots == 3u);
    CHECK(cfg.options.backend.llamacpp_kv_unified == std::optional<bool>(false));
    CHECK(cfg.options.backend.llamacpp_cache_ram_mib == 4096u);
    CHECK(cfg.options.backend.llamacpp_threads == 8);
    CHECK(cfg.options.backend.llamacpp_mmproj_path == std::optional<std::string>("mm.gguf"));
    CHECK(cfg.options.backend.llamacpp_spec_type == std::optional<std::string>("draft-mtp"));
    CHECK(cfg.options.backend.llamacpp_spec_draft_max == std::optional<std::int32_t>(4));

    REQUIRE(cfg.options.profile_bindings.size() == 2u);
    const auto& plain_meta = cfg.options.profile_bindings[1].metadata;
    CHECK(plain_meta.find("parallel")->as_uint() == 3u);
    CHECK(plain_meta.find("kv_unified")->as_bool() == false);
    CHECK(plain_meta.find("cache_ram_mib")->as_uint() == 4096u);
    CHECK(plain_meta.find("threads")->as_uint() == 8u);
    REQUIRE(plain_meta.find("speculative_types") != nullptr);
    CHECK(plain_meta.find("speculative_types")->as_array()[0].as_string() == "draft-mtp");
    CHECK(plain_meta.find("speculative_draft_n_max")->as_uint() == 4u);

    bool has_resolved_note = false;
    for (const auto& note : cfg.profile_notes) {
        if (note.find("llamacpp resolved settings:") != std::string::npos &&
            note.find("slots=3") != std::string::npos && note.find("spec_type=draft-mtp") != std::string::npos) {
            has_resolved_note = true;
        }
    }
    CHECK(has_resolved_note);

    std::error_code ec;
    fs::remove(g_path, ec);
    fs::remove(profiles, ec);
}

TEST_CASE("serve_main: ollama_cloud config is strict, never takes an inline key, and seeds the model list") {
    const auto key_file = temp_path("cloud.key");
    std::ofstream(key_file, std::ios::binary) << "sk-FAKE-serve-test-key\n";
    const auto config = temp_path("cloud.json");
    {
        std::ofstream out(config, std::ios::binary);
        out << R"({"base_url": "https://ollama.com", "api_key_file": )" << json::Value(key_file.string()).dump()
            << R"(, "models": ["gpt-oss:120b", "qwen3-coder:480b"], "request_timeout_ms": 120000,
                 "idle_timeout_ms": 0})";
    }
    si::BackendSetup setup;
    REQUIRE(si::load_ollama_cloud_config(config.string(), setup).ok());
    CHECK(setup.ollama_cloud_base_url == "https://ollama.com");
    CHECK(setup.ollama_cloud_api_key_file == key_file.string());
    CHECK(setup.ollama_cloud_api_key_env.empty());
    REQUIRE(setup.ollama_cloud_models.size() == 2u);
    CHECK(setup.ollama_cloud_request_timeout_ms == 120000u);
    CHECK(setup.ollama_cloud_idle_timeout_ms == 0u);
    CHECK(setup.ollama_cloud_connect_timeout_ms == 10000u);

    det::ServeCliConfig cfg;
    std::string err;
    CHECK(build_cli({"--backend", "ollama_cloud", "--ollama-cloud-config", config.string()}, cfg, err) == 0);
    CHECK(cfg.options.backend.backend == "ollama_cloud");
    CHECK(err.find("remote provider (https://ollama.com)") != std::string::npos);
    CHECK(err.find("sk-FAKE") == std::string::npos);
    REQUIRE(cfg.options.models.size() == 2u);
    CHECK(cfg.options.models[0] == "gpt-oss:120b");
    det::ServeCliConfig only_one;
    CHECK(build_cli({"--backend", "ollama_cloud", "--ollama-cloud-config", config.string(), "--model",
                     "qwen3-coder:480b"},
                    only_one, err) == 0);
    CHECK(only_one.options.models == std::vector<std::string>{"qwen3-coder:480b"});
    det::ServeCliConfig missing;
    CHECK(build_cli({"--backend", "ollama_cloud", "--model", "gpt-oss:120b"}, missing, err) == 2);
    CHECK(err.find("--ollama-cloud-config") != std::string::npos);
    det::ServeCliConfig wrong_backend;
    CHECK(build_cli({"--backend", "mock", "--ollama-cloud-config", config.string()}, wrong_backend, err) == 2);
#if defined(SONDER_HAS_OLLAMA_CLOUD_BACKEND)
    auto made = si::make_backend(cfg.options.backend);
    REQUIRE_MESSAGE(made.ok(), made.status().to_string());
    CHECK(made.value()->name() == "ollama_cloud");
    CHECK(made.value()->description().find("sk-FAKE") == std::string::npos);
#endif

    const std::string inline_key = "sk-FAKE-inline-do-not-echo";
    const std::vector<std::string> bad_documents = {
        R"({"api_key": ")" + inline_key + R"(", "models": ["m"]})",
        R"({"api_key_env": "OLLAMA_API_KEY", "api_key_file": "k", "models": ["m"]})",
        R"({"models": ["m"]})",
        R"({"api_key_env": "OLLAMA_API_KEY", "models": []})",
        R"({"api_key_env": "OLLAMA_API_KEY", "models": ["m"], "insecure_skip_verify": true})",
        R"({"api_key_env": "OLLAMA_API_KEY", "models": ["m"], "connect_timeout_ms": 0})",
        R"({"api_key_env": "OLLAMA_API_KEY", "models": [")" + inline_key + "\"]",  // malformed JSON
        // base_url is checked before anything logs it; messages never quote it.
        R"({"base_url": "https://)" + inline_key + R"(@ollama.com", "api_key_env": "OLLAMA_API_KEY", "models": ["m"]})",
        R"({"base_url": "https://user:)" + inline_key + R"(@ollama.com/x", "api_key_env": "OLLAMA_API_KEY", "models": ["m"]})",
        R"({"base_url": "http://ollama.com/)" + inline_key + R"(", "api_key_env": "OLLAMA_API_KEY", "models": ["m"]})",
        R"({"base_url": "https://ollama.com/?t=)" + inline_key + R"(", "api_key_env": "OLLAMA_API_KEY", "models": ["m"]})",
        R"({"base_url": "https://ollama.com/ )" + inline_key + R"(", "api_key_env": "OLLAMA_API_KEY", "models": ["m"]})",
    };
    for (std::size_t i = 0; i < bad_documents.size(); ++i) {
        CAPTURE(i);
        const auto bad = temp_path("bad-cloud-" + std::to_string(i) + ".json");
        std::ofstream(bad, std::ios::binary) << bad_documents[i];
        si::BackendSetup s;
        const auto st = si::load_ollama_cloud_config(bad.string(), s);
        CHECK_FALSE(st.ok());
        CHECK(st.message().find(inline_key) == std::string::npos);
        // Through the CLI as well: the remote-provider warning must not run first.
        det::ServeCliConfig rejected;
        std::string cli_err;
        CHECK(build_cli({"--backend", "ollama_cloud", "--ollama-cloud-config", bad.string()}, rejected, cli_err) == 2);
        CHECK(cli_err.find(inline_key) == std::string::npos);
        std::error_code ec;
        fs::remove(bad, ec);
    }
    std::error_code ec;
    fs::remove(config, ec);
    fs::remove(key_file, ec);
}

TEST_CASE("serve_main: ollama_cloud base_url off the Ollama Cloud origin needs allow_custom_endpoint") {
    const std::string marker = "attacker-marker";
    const auto write = [](const std::string& name, const std::string& text) {
        const auto p = temp_path(name);
        std::ofstream(p, std::ios::binary) << text;
        return p;
    };
    const std::string tail = R"(, "api_key_env": "OLLAMA_API_KEY", "models": ["gpt-oss:120b"]})";
    // Canonical origin spellings load without the opt-in.
    for (const std::string& url : {std::string("https://ollama.com"), std::string("https://OLLAMA.com:443/"),
                                  std::string("https://ollama.com/")}) {
        CAPTURE(url);
        const auto p = write("cloud-canonical.json", R"({"base_url": ")" + url + "\"" + tail);
        si::BackendSetup s;
        const auto st = si::load_ollama_cloud_config(p.string(), s);
        CHECK_MESSAGE(st.ok(), st.to_string());
        CHECK_FALSE(s.ollama_cloud_allow_custom_endpoint);
        std::error_code ec;
        fs::remove(p, ec);
    }
    const std::vector<std::string> foreign = {"https://" + marker + ".example", "https://ollama.com:8443",
                                              "https://ollama.com." + marker, "https://api.ollama.com"};
    for (const std::string& url : foreign) {
        CAPTURE(url);
        const auto refused = write("cloud-custom.json", R"({"base_url": ")" + url + "\"" + tail);
        si::BackendSetup s;
        const auto st = si::load_ollama_cloud_config(refused.string(), s);
        REQUIRE_FALSE(st.ok());
        CHECK(st.message().find("allow_custom_endpoint") != std::string::npos);
        CHECK(st.message().find(marker) == std::string::npos);
        CHECK(st.message().find("8443") == std::string::npos);
        // The CLI fails before the remote-provider warning prints the URL.
        det::ServeCliConfig cfg;
        std::string err;
        CHECK(build_cli({"--backend", "ollama_cloud", "--ollama-cloud-config", refused.string()}, cfg, err) == 2);
        CHECK(err.find(marker) == std::string::npos);

        const auto allowed =
            write("cloud-custom-ok.json", R"({"base_url": ")" + url + R"(", "allow_custom_endpoint": true)" + tail);
        si::BackendSetup a;
        const auto ok = si::load_ollama_cloud_config(allowed.string(), a);
        CHECK_MESSAGE(ok.ok(), ok.to_string());
        CHECK(a.ollama_cloud_allow_custom_endpoint);
#if defined(SONDER_HAS_OLLAMA_CLOUD_BACKEND)
        // make_backend forwards the opt-in; without it the backend refuses too.
        a.backend = "ollama_cloud";
        auto made = si::make_backend(a);
        if (!made.ok()) {
            // Only the key lookup may fail here (OLLAMA_API_KEY unset), never the endpoint.
            CHECK(made.status().message().find("allow_custom_endpoint") == std::string::npos);
        }
        a.ollama_cloud_allow_custom_endpoint = false;
        auto refused_backend = si::make_backend(a);
        REQUIRE_FALSE(refused_backend.ok());
        CHECK(refused_backend.status().message().find("allow_custom_endpoint") != std::string::npos);
        CHECK(refused_backend.status().message().find(marker) == std::string::npos);
#endif
        std::error_code ec;
        fs::remove(refused, ec);
        fs::remove(allowed, ec);
    }
    // Strict types and bounds for the new fields.
    for (const std::string& extra :
         {std::string(R"("allow_custom_endpoint": "yes")"), std::string(R"("allow_custom_endpoint": 1)"),
          std::string(R"("max_response_bytes": 0)"), std::string(R"("max_response_bytes": 67108865)"),
          std::string(R"("max_response_bytes": "1")")}) {
        CAPTURE(extra);
        const auto p = write("cloud-bad-field.json", "{" + extra + tail);
        si::BackendSetup s;
        CHECK_FALSE(si::load_ollama_cloud_config(p.string(), s).ok());
        std::error_code ec;
        fs::remove(p, ec);
    }
    const auto lower = write("cloud-cap.json", R"({"max_response_bytes": 1048576)" + tail);
    si::BackendSetup s;
    REQUIRE(si::load_ollama_cloud_config(lower.string(), s).ok());
    CHECK(s.ollama_cloud_max_response_bytes == 1048576u);
    si::BackendSetup d;
    CHECK(d.ollama_cloud_max_response_bytes == (std::uint64_t{64} << 20));
    std::error_code ec;
    fs::remove(lower, ec);
}
