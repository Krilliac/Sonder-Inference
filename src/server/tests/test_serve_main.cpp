// serve_main(): --help, usage errors (exit 2), --ready-file with an ephemeral
// port, the startup banner and access log, a port in use (exit 1), and
// graceful shutdown through the hook (exit 0, engine.stopped delivered,
// telemetry streams closed).
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <future>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"

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
    REQUIRE(eventually([&] { return fs::exists(ready); }, std::chrono::milliseconds(10000)));
    auto doc = json::parse(read_file(ready));
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
    REQUIRE(eventually([&] { return fs::exists(ready); }, std::chrono::milliseconds(10000)));
    auto doc = json::parse(read_file(ready));
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
