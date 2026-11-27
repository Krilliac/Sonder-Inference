// Isolated llama.cpp RPC worker (docs/integration/rpc-worker.md).
//
// The RPC client must only run in the supervised llama-server child that this
// backend owns, never through operator args and never with an unvalidated
// peer list. No llama-server is started, no socket is opened: backend
// decisions are read through save_slot() on a generic-mode backend, which
// returns the construction-time validation status when the options were
// refused and otherwise unsupported("slot API requires native mode") before
// any I/O (the same probe test_containment.cpp uses). Supervisor behaviour is
// driven through the fake launcher.

#include <doctest/doctest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "../process.hpp"
#include "../rpc_worker.hpp"
#include "../supervisor.hpp"
#include "fake_process.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#include "sonder/inference/json.hpp"
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#endif

using namespace sonder::inference;
namespace ls = sonder::inference::llamaserver;

namespace {

constexpr std::string_view kNoIoMarker = "slot API requires native mode";

bool accepted(const Status &status) {
    return status.code() == ErrorCode::unsupported && status.message().find(kNoIoMarker) != std::string::npos;
}

LlamaServerBackendOptions quiet_spawn() {
    LlamaServerBackendOptions o;
    o.mode = LlamaServerMode::spawn;
    o.executable = "rpc-worker-never-started-llama-server.exe";
    o.args = {"-m", "split.gguf", "-ngl", "99"};
    o.native_completion = false;  // no attached monitor and no request at construction
    o.spill_guard.enabled = false;
    o.stall_guard.enabled = false;
    o.diagnostics.kv_pairing_check = false;
    return o;
}

LlamaServerRpcWorkerOptions worker(std::vector<std::string> servers, bool allow_public = false) {
    LlamaServerRpcWorkerOptions w;
    w.enabled = true;
    w.servers = std::move(servers);
    w.allow_public_addresses = allow_public;
    return w;
}

Status backend_status(LlamaServerBackendOptions o) {
    return make_llamaserver_backend(std::move(o))->save_slot(0, "rpc-worker.bin");
}

} // namespace

TEST_CASE("rpc worker: every llama-server spelling of --rpc is recognised, neighbours are not") {
    for (const std::string_view arg : {"--rpc", "--rpc=10.0.0.2:50052", "--rpc=", "--rpc=a,b"}) {
        CAPTURE(arg);
        CHECK(ls::is_rpc_flag(arg));
    }
    for (const std::string_view arg : {"-rpc", "--rpcx", "--rpc-server", "--RPC", "rpc", "--", "--r_c", "", "-",
                                       "10.0.0.2:50052", "--rp", "--no-rpc"}) {
        CAPTURE(arg);
        CHECK_FALSE(ls::is_rpc_flag(arg));
    }
    CHECK(ls::refuse_operator_rpc_arguments({"-m", "x.gguf"}).ok());
    const Status refused = ls::refuse_operator_rpc_arguments({"-m", "x.gguf", "--rpc", "10.0.0.2:50052"});
    CHECK(refused.code() == ErrorCode::invalid_argument);
    CHECK(refused.message().find("rpc_worker") != std::string::npos);
    // The refusal never echoes the peer list back.
    CHECK(refused.message().find("10.0.0.2") == std::string::npos);
}

TEST_CASE("rpc worker: server literals accept only pinned local IPv4:PORT unless public is allowed") {
    for (const std::string_view local :
         {"127.0.0.1:50052", "10.0.0.2:1", "10.255.255.255:65535", "172.16.0.1:50052", "172.31.255.255:50052",
          "192.168.50.7:50052", "169.254.10.20:50052", "100.64.0.1:50052", "100.127.255.254:50052"}) {
        CAPTURE(local);
        CHECK(ls::validate_rpc_server(local, false).ok());
        CHECK(ls::validate_rpc_server(local, true).ok());
    }
    for (const std::string_view routed : {"8.8.8.8:50052", "172.15.255.255:50052", "172.32.0.1:50052",
                                          "100.63.255.255:50052", "100.128.0.1:50052", "192.0.2.10:50052",
                                          "1.2.3.4:50052", "223.255.255.255:50052", "192.169.0.1:50052"}) {
        CAPTURE(routed);
        const Status st = ls::validate_rpc_server(routed, false);
        CHECK(st.code() == ErrorCode::invalid_argument);
        CHECK(st.message().find("allow_public_addresses") != std::string::npos);
        CHECK(ls::validate_rpc_server(routed, true).ok());
    }
    const std::vector<std::string> never = {
        "0.0.0.0:50052", "0.1.2.3:50052", "224.0.0.1:50052", "239.1.1.1:50052", "240.0.0.1:50052",
        "255.255.255.255:50052", "", ":50052", "10.0.0.2", "10.0.0.2:", "10.0.0.2:0", "10.0.0.2:65536",
        "10.0.0.2:99999", "10.0.0.2:050052", "10.0.0.2:05", "10.0.0.2:+5", "10.0.0.2:-5", "10.0.0.2:5x",
        "010.0.0.2:50052", "10.0.0.256:1", "10.0.0:1", "10.0.0.2.3:1", "10..0.2:1", ".10.0.0.2:1",
        "node1:50052", "localhost:50052", "[::1]:50052", "::1:50052", "fe80::1:50052",
        // Injection into the comma-separated --rpc value or the argv entry.
        "10.0.0.2:50052,10.0.0.3:50052", "10.0.0.2:50052 ", " 10.0.0.2:50052", "10.0.0.2 :50052",
        "10.0.0.2:50052\n", "10.0.0.2:50052\r", "10.0.0.2:50052;x", "10.0.0.2:50052/x",
        std::string("10.0.0.2:5\0", 11), std::string("10.0.0.2\0:5", 11),
        "\xef\xbc\x91" "0.0.0.2:50052",  // fullwidth digit one
        "0x0a.0.0.2:50052", "10.0.0.2:0x50",
    };
    for (const std::string &bad : never) {
        CAPTURE(bad);
        CHECK(ls::validate_rpc_server(bad, false).code() == ErrorCode::invalid_argument);
        CHECK(ls::validate_rpc_server(bad, true).code() == ErrorCode::invalid_argument);
    }
    // A hostile literal is echoed bounded and printable only.
    const std::string huge(100000, 'A');
    const Status st = ls::validate_rpc_server(huge, false);
    CHECK(st.code() == ErrorCode::invalid_argument);
    CHECK(st.message().size() < 256);
    const Status ctl = ls::validate_rpc_server("10.0.0.2:5\x1b[2J", false);
    CHECK(ctl.message().find('\x1b') == std::string::npos);
}

TEST_CASE("rpc worker: the block needs spawn mode, a bounded unique server list, and nothing when disabled") {
    CHECK(ls::validate_rpc_worker({}, LlamaServerMode::spawn).ok());
    CHECK(ls::validate_rpc_worker({}, LlamaServerMode::attach).ok());
    CHECK(ls::validate_rpc_worker(worker({"10.0.0.2:50052"}), LlamaServerMode::spawn).ok());

    const Status attach = ls::validate_rpc_worker(worker({"10.0.0.2:50052"}), LlamaServerMode::attach);
    CHECK(attach.code() == ErrorCode::invalid_argument);
    CHECK(attach.message().find("spawn") != std::string::npos);

    CHECK(ls::validate_rpc_worker(worker({}), LlamaServerMode::spawn).code() == ErrorCode::invalid_argument);

    LlamaServerRpcWorkerOptions half;  // servers without enabled is refused, not ignored
    half.servers = {"10.0.0.2:50052"};
    CHECK(ls::validate_rpc_worker(half, LlamaServerMode::spawn).code() == ErrorCode::invalid_argument);

    CHECK(ls::validate_rpc_worker(worker({"10.0.0.2:50052", "10.0.0.3:50052", "10.0.0.2:50052"}),
                                  LlamaServerMode::spawn)
              .code() == ErrorCode::invalid_argument);

    std::vector<std::string> many;
    for (std::size_t i = 0; i < LlamaServerRpcWorkerOptions::kMaxServers; ++i)
        many.push_back("10.0.0." + std::to_string(i + 1) + ":50052");
    CHECK(ls::validate_rpc_worker(worker(many), LlamaServerMode::spawn).ok());
    many.push_back("10.0.1.1:50052");
    CHECK(ls::validate_rpc_worker(worker(many), LlamaServerMode::spawn).code() == ErrorCode::invalid_argument);

    // One bad entry anywhere fails the whole list.
    CHECK(ls::validate_rpc_worker(worker({"10.0.0.2:50052", "node1:50052"}), LlamaServerMode::spawn).code() ==
          ErrorCode::invalid_argument);
    CHECK(ls::validate_rpc_worker(worker({"10.0.0.2:50052", "8.8.8.8:50052"}), LlamaServerMode::spawn).code() ==
          ErrorCode::invalid_argument);
    CHECK(ls::validate_rpc_worker(worker({"10.0.0.2:50052", "8.8.8.8:50052"}, true), LlamaServerMode::spawn).ok());
}

TEST_CASE("rpc worker: the supervisor argv puts the validated --rpc first and leaves operator args intact") {
    const std::vector<std::string> args = {"-m", "split.gguf", "--device", "RPC0,CUDA0"};
    CHECK(ls::rpc_worker_arguments({}, args) == args);
    const auto argv = ls::rpc_worker_arguments(worker({"10.0.0.2:50052", "192.168.50.7:50053"}), args);
    const std::vector<std::string> expected = {"--rpc", "10.0.0.2:50052,192.168.50.7:50053", "-m", "split.gguf",
                                               "--device", "RPC0,CUDA0"};
    CHECK(argv == expected);
    // Exactly one --rpc reaches the child.
    std::size_t rpc_flags = 0;
    for (const auto &a : argv)
        rpc_flags += ls::is_rpc_flag(a) ? 1u : 0u;
    CHECK(rpc_flags == 1u);
}

TEST_CASE("rpc worker: spawn backends refuse operator --rpc with or without rpc_worker") {
    for (const std::vector<std::string> &extra :
         {std::vector<std::string>{"--rpc", "10.0.0.2:50052"}, std::vector<std::string>{"--rpc=10.0.0.2:50052"}}) {
        auto plain = quiet_spawn();
        plain.args.insert(plain.args.end(), extra.begin(), extra.end());
        const Status st = backend_status(plain);
        CHECK(st.code() == ErrorCode::invalid_argument);
        CHECK(st.message().find("rpc_worker") != std::string::npos);

        auto both = quiet_spawn();
        both.rpc_worker = worker({"10.0.0.3:50052"});
        both.args.insert(both.args.end(), extra.begin(), extra.end());
        CHECK(backend_status(both).code() == ErrorCode::invalid_argument);
    }
    // The validated worker block itself is accepted.
    auto ok = quiet_spawn();
    ok.rpc_worker = worker({"10.0.0.2:50052", "192.168.50.7:50052"});
    CHECK(accepted(backend_status(ok)));
    // And refused when it names a host or sits on an attach backend.
    auto named = quiet_spawn();
    named.rpc_worker = worker({"node1:50052"});
    CHECK(backend_status(named).code() == ErrorCode::invalid_argument);
    auto attach = quiet_spawn();
    attach.mode = LlamaServerMode::attach;
    attach.base_url = "http://127.0.0.1:8080";
    attach.rpc_worker = worker({"10.0.0.2:50052"});
    CHECK(backend_status(attach).code() == ErrorCode::invalid_argument);
    // An attach backend only describes an upstream it does not run, so its
    // args are not filtered for --rpc (unchanged behaviour).
    auto described = quiet_spawn();
    described.mode = LlamaServerMode::attach;
    described.base_url = "http://127.0.0.1:8080";
    described.args = {"--rpc", "10.0.0.2:50052"};
    CHECK(accepted(backend_status(described)));
}

TEST_CASE("rpc worker: LLAMA_ARG_RPC cannot be smuggled through the child environment") {
    for (const char *key : {"LLAMA_ARG_RPC", "llama_arg_rpc", "Llama_Arg_Rpc"}) {
        CAPTURE(key);
        CHECK(ls::validate_process_environment({{key, "10.0.0.2:50052"}}).code() == ErrorCode::invalid_argument);
        auto o = quiet_spawn();
        o.environment = {{key, "10.0.0.2:50052"}};
        CHECK(backend_status(o).code() == ErrorCode::invalid_argument);
    }
}

TEST_CASE("rpc worker: runtime status names the isolated child and its pinned servers without starting it") {
    auto o = quiet_spawn();
    o.rpc_worker = worker({"10.0.0.2:50052", "192.168.50.7:50052"});
    auto backend = make_llamaserver_backend(std::move(o));
    const auto status = backend->runtime_status();
    REQUIRE(status.has_value());
    REQUIRE(status->diagnostics.has_value());
    const auto *rpc = status->diagnostics->find("rpc_worker");
    REQUIRE(rpc != nullptr);
    REQUIRE(rpc->is_object());
    CHECK(rpc->as_object().find("isolated_child")->as_bool());
    const auto &servers = rpc->as_object().find("servers")->as_array();
    REQUIRE(servers.size() == 2u);
    CHECK(servers[0].as_string() == "10.0.0.2:50052");
    CHECK(servers[1].as_string() == "192.168.50.7:50052");

    auto plain = make_llamaserver_backend(quiet_spawn());
    const auto plain_status = plain->runtime_status();
    if (plain_status && plain_status->diagnostics)
        CHECK(plain_status->diagnostics->find("rpc_worker") == nullptr);
}

TEST_CASE("rpc worker: a worker that dies (an aborted RPC handshake) is relaunched with backoff and the same argv") {
    using namespace sonder_test;
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.arguments = ls::rpc_worker_arguments(worker({"10.0.0.2:50052"}), {"-m", "split.gguf"});
    options.max_restarts = 2;
    options.spill_guard.enabled = false;
    options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    ls::Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    REQUIRE(supervisor.start().ok());
    child_at(state, 0)->alive.store(false);  // GGML_ABORT in the child: the parent only sees an exit
    REQUIRE(wait_for_starts(state, 2));
    child_at(state, 1)->alive.store(false);
    REQUIRE(wait_for_starts(state, 3));
    child_at(state, 2)->alive.store(false);
    for (int i = 0; i < 1000 && child_at(state, 2)->stops.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    // The restart budget is bounded: a peer that keeps killing the worker
    // ends in a failed backend, not a hot respawn loop.
    CHECK_FALSE(supervisor.start().ok());
    supervisor.stop();
    std::lock_guard lock(state->mutex);
    REQUIRE(state->specs.size() == 3u);
    CHECK(state->starts[1] - state->starts[0] >= options.restart_initial_backoff);
    CHECK(state->starts[2] - state->starts[1] >= options.restart_max_backoff);
    for (const auto &spec : state->specs) {
        REQUIRE(spec.arguments.size() >= 2u);
        CHECK(spec.arguments[0] == "--rpc");
        CHECK(spec.arguments[1] == "10.0.0.2:50052");
        std::size_t rpc_flags = 0;
        for (const auto &a : spec.arguments)
            rpc_flags += ls::is_rpc_flag(a) ? 1u : 0u;
        CHECK(rpc_flags == 1u);
    }
}

#if defined(SONDER_HAS_SERVER)
namespace {

class TempConfig {
  public:
    explicit TempConfig(const std::string &text) {
        static int counter = 0;
        path_ = std::filesystem::temp_directory_path() /
                ("sonder-rpc-worker-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 "-" + std::to_string(++counter) + ".json");
        std::ofstream out(path_, std::ios::binary);
        out << text;
    }
    ~TempConfig() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    TempConfig(const TempConfig &) = delete;
    TempConfig &operator=(const TempConfig &) = delete;
    [[nodiscard]] std::string path() const { return path_.string(); }

  private:
    std::filesystem::path path_;
};

std::string spawn_config(const std::string &tail) {
    return R"({"mode":"spawn","executable":"rpc-worker-never-started-llama-server.exe","args":["-m","split.gguf"],)"
           R"("native_completion":false,"spill_guard":{"enabled":false},"stall_guard":{"enabled":false})" +
           tail + "}";
}

// The serve path: --llamaserver-config FILE -> load_llamaserver_config ->
// make_backend. Returns the load error or the backend's save_slot status.
Status config_status(const std::string &json_text, BackendSetup *loaded = nullptr) {
    TempConfig file(json_text);
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
    return server->save_slot(0, "rpc-worker.bin");
}

} // namespace

TEST_CASE("rpc worker: the llamaserver config file switch is strict and reaches the backend") {
    BackendSetup loaded;
    CHECK(accepted(config_status(
        spawn_config(R"(,"rpc_worker":{"enabled":true,"servers":["10.0.0.2:50052","192.168.50.7:50052"]})"), &loaded)));
    CHECK(loaded.llamaserver_rpc_worker.enabled);
    CHECK(loaded.llamaserver_rpc_worker.servers == std::vector<std::string>{"10.0.0.2:50052", "192.168.50.7:50052"});
    CHECK_FALSE(loaded.llamaserver_rpc_worker.allow_public_addresses);

    // Absent block: off, and existing configs are unchanged.
    BackendSetup plain;
    CHECK(accepted(config_status(spawn_config(""), &plain)));
    CHECK_FALSE(plain.llamaserver_rpc_worker.enabled);
    CHECK(plain.llamaserver_rpc_worker.servers.empty());

    // Shape errors are refused while loading.
    for (const std::string block : {
             R"("rpc_worker":true)",
             R"("rpc_worker":{"enabled":"yes"})",
             R"("rpc_worker":{"enabled":true,"servers":"10.0.0.2:50052"})",
             R"("rpc_worker":{"enabled":true,"servers":[50052]})",
             R"("rpc_worker":{"enabled":true,"servers":[""]})",
             R"("rpc_worker":{"enabled":true,"servers":["10.0.0.2:50052                "]})",
             R"("rpc_worker":{"enabled":true,"servers":["10.0.0.2:50052"],"allow_public_addresses":1})",
             R"("rpc_worker":{"enabled":true,"servers":["10.0.0.2:50052"],"token":"x"})",
             R"("rpc_worker":{"enabled":true,"servers":["1.0.0.1:1","1.0.0.2:1","1.0.0.3:1","1.0.0.4:1","1.0.0.5:1",)"
             R"("1.0.0.6:1","1.0.0.7:1","1.0.0.8:1","1.0.0.9:1","1.0.0.10:1","1.0.0.11:1","1.0.0.12:1","1.0.0.13:1",)"
             R"("1.0.0.14:1","1.0.0.15:1","1.0.0.16:1","1.0.0.17:1"]})",
         }) {
        CAPTURE(block);
        BackendSetup untouched;
        const Status st = config_status(spawn_config("," + block), &untouched);
        CHECK(st.code() == ErrorCode::invalid_argument);
        CHECK(st.message().find("rpc_worker") != std::string::npos);
    }

    // Policy errors are refused by the backend before any child starts.
    for (const std::string block : {
             R"("rpc_worker":{"enabled":true,"servers":["node1:50052"]})",
             R"("rpc_worker":{"enabled":true,"servers":["8.8.8.8:50052"]})",
             R"("rpc_worker":{"enabled":true,"servers":[]})",
             R"("rpc_worker":{"enabled":false,"servers":["10.0.0.2:50052"]})",
             R"("rpc_worker":{"enabled":true,"servers":["10.0.0.2:50052","10.0.0.2:50052"]})",
         }) {
        CAPTURE(block);
        CHECK(config_status(spawn_config("," + block)).code() == ErrorCode::invalid_argument);
    }
    CHECK(accepted(config_status(
        spawn_config(R"(,"rpc_worker":{"enabled":true,"servers":["8.8.8.8:50052"],"allow_public_addresses":true})"))));

    // Operator --rpc in the config file's args is refused: use rpc_worker.
    const Status args_rpc = config_status(
        R"({"mode":"spawn","executable":"rpc-worker-never-started-llama-server.exe",)"
        R"("args":["-m","split.gguf","--rpc","10.0.0.2:50052"],"native_completion":false,)"
        R"("spill_guard":{"enabled":false},"stall_guard":{"enabled":false}})");
    CHECK(args_rpc.code() == ErrorCode::invalid_argument);
    CHECK(args_rpc.message().find("rpc_worker") != std::string::npos);

    // An attach backend cannot host the worker.
    CHECK(config_status(R"({"mode":"attach","base_url":"http://127.0.0.1:8080","native_completion":false,)"
                        R"("spill_guard":{"enabled":false},"rpc_worker":{"enabled":true,"servers":["10.0.0.2:50052"]}})")
              .code() == ErrorCode::invalid_argument);
}
#endif
