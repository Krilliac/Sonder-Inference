// HTTP behaviour of `sonder-infer serve` against an in-process server on an
// ephemeral loopback port (MOCK backend): routes and response shapes, common
// headers, auth, the Host check, CORS, request limits and the connection cap.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__linux__)
#  include <arpa/inet.h>
#  include <dirent.h>
#  include <fcntl.h>
#  include <netinet/in.h>
#  include <sys/resource.h>
#  include <sys/syscall.h>
#  include <unistd.h>
#  include <sys/time.h>

#  include <cstdlib>
#  include <map>
#  include <optional>
#endif

#include "server_test_support.hpp"
#include "src/health_features.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

void check_common_headers(const Reply& r) {
    CHECK(r.header("x-sonder-inference-api") == "1");
    CHECK(r.header("cache-control") == "no-store");
    CHECK(r.header("connection") == "close");
    CHECK_FALSE(r.header("x-sonder-request-id").empty());
}

void check_error(const Reply& r, int status, const std::string& type, const std::string& code) {
    CHECK(r.status == status);
    check_common_headers(r);
    const json::Value body = r.json();
    const json::Value* err = body.find("error");
    REQUIRE_MESSAGE(err != nullptr, r.body);
    CHECK(err->find("type")->as_string() == type);
    CHECK(err->find("code")->as_string() == code);
    CHECK_FALSE(err->find("message")->as_string().empty());
    CHECK(err->find("param") != nullptr);
    CHECK(body.find("sonder")->find("api_version")->as_int() == 1);
}

}  // namespace

TEST_CASE("health: ready shape with backends, models and telemetry") {
    Fixture f;
    const Reply r = get(f.port, "/v1/sonder/health");
    REQUIRE(r.status == 200);
    CHECK(r.complete);
    check_common_headers(r);
    CHECK(r.header("content-type") == "application/json");
    const json::Value h = r.json();
    CHECK(h.find("status")->as_string() == "ready");
    CHECK(h.find("api_version")->as_int() == 1);
    CHECK(h.find("abi_version")->as_int() == 1);
    CHECK_FALSE(h.find("version")->as_string().empty());
    CHECK_FALSE(h.find("commit")->as_string().empty());
    CHECK(h.find("instance_id")->as_string() == f.server->instance_id());
    CHECK(h.find("instance_id")->as_string().rfind("tel-", 0) == 0);
    CHECK_FALSE(h.find("node_id")->as_string().empty());
    CHECK(h.find("uptime_s")->as_double() >= 0.0);
    CHECK(h.find("synthetic")->as_bool());
    CHECK_FALSE(h.find("auth_required")->as_bool());
    const auto& backends = h.find("backends")->as_array();
    REQUIRE(backends.size() == 1);
    CHECK(backends[0].find("name")->as_string() == "mock");
    CHECK(backends[0].find("available")->as_bool());
    CHECK(backends[0].find("capabilities")->is_array());
    const auto& models = h.find("models")->as_array();
    REQUIRE(models.size() == 1);
    CHECK(models[0].find("id")->as_string() == "mock:tiny");
    CHECK(models[0].find("backend")->as_string() == "mock");
    CHECK(models[0].find("default")->as_bool());
    const json::Value* t = h.find("telemetry");
    REQUIRE(t != nullptr);
    CHECK(t->find("level")->as_string() == "standard");
    for (const char* key : {"subscribers", "retained", "capacity", "emitted", "dropped", "subscriber_dropped_events"}) {
        CHECK_MESSAGE(t->find(key) != nullptr, key);
    }
    CHECK(t->find("capacity")->as_int() == 8192);
    CHECK(t->find("emitted")->as_int() > 0);
}

TEST_CASE("models: OpenAI list with Sonder extension") {
    auto o = Fixture::defaults();
    o.models = {"mock:tiny", "mock:other"};
    Fixture f(o);
    const Reply r = get(f.port, "/v1/models");
    REQUIRE(r.status == 200);
    const json::Value doc = r.json();
    CHECK(doc.find("object")->as_string() == "list");
    const auto& data = doc.find("data")->as_array();
    REQUIRE(data.size() == 2);
    CHECK(data[0].find("id")->as_string() == "mock:tiny");
    CHECK(data[0].find("object")->as_string() == "model");
    CHECK(data[0].find("owned_by")->as_string() == "sonder-inference");
    CHECK(data[0].find("sonder")->find("backend")->as_string() == "mock");
    CHECK(data[0].find("sonder")->find("default")->as_bool());
    CHECK(data[0].find("sonder")->find("synthetic")->as_bool());
    CHECK_FALSE(data[1].find("sonder")->find("default")->as_bool());
}

TEST_CASE("identity: default alias, explicit model and unknown model") {
    Fixture f;
    const Reply r = get(f.port, "/v1/sonder/identity");
    REQUIRE(r.status == 200);
    const json::Value doc = r.json();
    CHECK(doc.find("schema")->as_string() == "sonder.inference.identity/1");
    CHECK(doc.find("model")->as_string() == "mock:tiny");
    CHECK(doc.find("synthetic")->as_bool());
    CHECK(doc.find("reason")->is_null());
    const json::Value* id = doc.find("backend_identity");
    REQUIRE(id != nullptr);
    REQUIRE(id->is_object());
    CHECK(id->as_object().size() == 9);
    CHECK(id->find("backend")->as_string() == "mock");
    CHECK(id->find("model_digest")->as_string().size() == 64);

    const Reply named = get(f.port, "/v1/sonder/identity?model=mock%3Atiny");
    CHECK(named.status == 200);
    CHECK(named.json().find("backend_identity")->dump() == id->dump());
    check_error(get(f.port, "/v1/sonder/identity?model=nope"), 404, "not_found_error", "model_not_found");
}

TEST_CASE("routing: unknown path, wrong method and embeddings") {
    Fixture f;
    check_error(get(f.port, "/v1/nothing"), 404, "not_found_error", "not_found");
    const Reply wrong = post(f.port, "/v1/models", "{}");
    check_error(wrong, 405, "invalid_request_error", "method_not_allowed");
    CHECK(wrong.header("allow") == "GET, OPTIONS");
    const Reply emb = post(f.port, "/v1/embeddings", R"({"input":"x"})");
    REQUIRE(emb.status == 200);
    CHECK(emb.json().find("object")->as_string() == "list");
}

TEST_CASE("auth: a configured token is required on every route except preflight") {
    auto o = Fixture::defaults();
    o.token = "s3cret-token";
    Fixture f(o);
    const Reply missing = get(f.port, "/v1/sonder/health");
    check_error(missing, 401, "authentication_error", "unauthorized");
    CHECK(missing.header("www-authenticate") == "Bearer");
    check_error(get(f.port, "/v1/models", {{"Authorization", "Bearer wrong-token!"}}), 401, "authentication_error",
                "unauthorized");
    check_error(get(f.port, "/v1/models", {{"Authorization", "Basic s3cret-token"}}), 401, "authentication_error",
                "unauthorized");
    check_error(post(f.port, "/v1/chat/completions", chat_body()), 401, "authentication_error", "unauthorized");
    const Reply ok = get(f.port, "/v1/sonder/health", {{"Authorization", "bearer s3cret-token"}});
    CHECK(ok.status == 200);
    CHECK(ok.json().find("auth_required")->as_bool());
    const Reply preflight = roundtrip(f.port, build_request("OPTIONS", "/v1/chat/completions", f.port,
                                                           {{"Origin", "http://127.0.0.1:4173"},
                                                            {"Access-Control-Request-Method", "POST"}}));
    CHECK(preflight.status == 204);
    // With a token, the default origins may also use POST routes.
    CHECK(preflight.header("access-control-allow-origin") == "http://127.0.0.1:4173");
}

TEST_CASE("host check: loopback binds reject foreign Host headers (DNS rebinding)") {
    Fixture f;
    const auto with_host = [&](const std::string& host) {
        return roundtrip(f.port, build_request("GET", "/v1/sonder/health", f.port, {{"Host", host}}, std::nullopt,
                                              /*host_header=*/false));
    };
    check_error(with_host("evil.example"), 403, "permission_error", "forbidden_host");
    check_error(with_host("evil.example:" + std::to_string(f.port)), 403, "permission_error", "forbidden_host");
    check_error(with_host("0.0.0.0:" + std::to_string(f.port)), 403, "permission_error", "forbidden_host");
    check_error(roundtrip(f.port, build_request("GET", "/v1/sonder/health", f.port, {}, std::nullopt, false)), 403,
                "permission_error", "forbidden_host");
    CHECK(with_host("localhost:" + std::to_string(f.port)).status == 200);
    CHECK(with_host("[::1]:" + std::to_string(f.port)).status == 200);
    CHECK(with_host("127.0.0.1").status == 200);
}

TEST_CASE("cors: preflight, allowed and blocked origins") {
    auto o = Fixture::defaults();
    o.cors_origins = {"http://127.0.0.1:9999"};
    Fixture f(o);
    const Reply pre = roundtrip(f.port, build_request("OPTIONS", "/v1/telemetry/sse", f.port,
                                                     {{"Origin", "http://localhost:5173"},
                                                      {"Access-Control-Request-Method", "GET"},
                                                      {"Access-Control-Request-Headers", "last-event-id"}}));
    CHECK(pre.status == 204);
    check_common_headers(pre);
    CHECK(pre.header("access-control-allow-origin") == "http://localhost:5173");
    CHECK(pre.header("vary") == "Origin");
    CHECK(pre.header("access-control-allow-methods") == "GET, POST, OPTIONS");
    CHECK(pre.header("access-control-allow-headers") ==
          "Accept, Authorization, Cache-Control, Content-Type, Last-Event-ID, X-Sonder-Run-Id, "
          "X-Sonder-Parent-Request-Id, X-Sonder-Agent-Id, X-Sonder-Task-Id, X-Sonder-Workload, X-Sonder-Priority, "
          "X-Sonder-Deadline-Ms, X-Sonder-Reasoning-Budget, X-Sonder-No-Capture, X-Sonder-Local-Only");
    CHECK(pre.header("access-control-max-age") == "600");

    // Every default origin works on read-only GET routes.
    for (const char* origin : {"http://127.0.0.1:5173", "http://localhost:5173", "http://127.0.0.1:4173",
                               "http://localhost:4173", "tauri://localhost", "http://tauri.localhost"}) {
        const Reply r = get(f.port, "/v1/sonder/health", {{"Origin", origin}});
        CHECK(r.status == 200);
        CHECK(r.header("access-control-allow-origin") == origin);
        CHECK(r.header("vary") == "Origin");
    }
    // Default origins do not reach POST routes without a token...
    check_error(post(f.port, "/v1/chat/completions", chat_body(), {{"Origin", "http://127.0.0.1:4173"}}), 403,
                "permission_error", "forbidden_origin");
    const Reply pre_post = roundtrip(f.port, build_request("OPTIONS", "/v1/chat/completions", f.port,
                                                          {{"Origin", "http://127.0.0.1:4173"},
                                                           {"Access-Control-Request-Method", "POST"}}));
    CHECK(pre_post.status == 403);
    CHECK_FALSE(pre_post.has_header("access-control-allow-origin"));
    // ...but an explicit --cors-origin does.
    const Reply explicit_post =
        post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":2)"), {{"Origin", "http://127.0.0.1:9999"}});
    CHECK(explicit_post.status == 200);
    CHECK(explicit_post.header("access-control-allow-origin") == "http://127.0.0.1:9999");
    // Unknown origin: 403 and no CORS headers. No Origin: unaffected.
    const Reply blocked = get(f.port, "/v1/sonder/health", {{"Origin", "http://evil.example"}});
    check_error(blocked, 403, "permission_error", "forbidden_origin");
    CHECK_FALSE(blocked.has_header("access-control-allow-origin"));
    const Reply plain = get(f.port, "/v1/sonder/health");
    CHECK(plain.status == 200);
    CHECK_FALSE(plain.has_header("access-control-allow-origin"));
}

TEST_CASE("cors: --no-default-cors drops the defaults") {
    auto o = Fixture::defaults();
    o.default_cors = false;
    Fixture f(o);
    check_error(get(f.port, "/v1/sonder/health", {{"Origin", "http://127.0.0.1:5173"}}), 403, "permission_error",
                "forbidden_origin");
}

TEST_CASE("limits: 411, 413, 408 and 431") {
    auto o = Fixture::defaults();
    o.max_body_bytes = 64;
    o.read_timeout = std::chrono::milliseconds(300);
    Fixture f(o);
    // 411: POST without Content-Length, and chunked bodies.
    check_error(roundtrip(f.port, "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n"), 411,
                "invalid_request_error", "length_required");
    check_error(roundtrip(f.port,
                         "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n"
                         "0\r\n\r\n"),
                411, "invalid_request_error", "length_required");
    // 413: declared body over --max-body-bytes (rejected before reading it).
    check_error(post(f.port, "/v1/chat/completions", std::string(65, ' ')), 413, "invalid_request_error",
                "payload_too_large");
    // 408: headers that never finish (slowloris) and a body that never arrives.
    {
        Conn c(f.port);
        c.send("GET /v1/sonder/health HTTP/1.1\r\nHost: 127.0.0.1\r\n");
        const Reply r = parse_reply(c.read_all(std::chrono::milliseconds(5000)));
        check_error(r, 408, "invalid_request_error", "request_timeout");
    }
    {
        Conn c(f.port);
        c.send("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 40\r\n\r\n{\"mess");
        const Reply r = parse_reply(c.read_all(std::chrono::milliseconds(5000)));
        check_error(r, 408, "invalid_request_error", "request_timeout");
    }
    // 431: too many headers, and a head over 16 KiB.
    std::string many = "GET /v1/sonder/health HTTP/1.1\r\nHost: 127.0.0.1\r\n";
    for (int i = 0; i < 70; ++i) many += "X-H" + std::to_string(i) + ": v\r\n";
    check_error(roundtrip(f.port, many + "\r\n"), 431, "invalid_request_error", "request_header_fields_too_large");
    check_error(roundtrip(f.port, "GET /v1/sonder/health HTTP/1.1\r\nHost: 127.0.0.1\r\nX-Big: " +
                                     std::string(17 * 1024, 'a') + "\r\n\r\n"),
                431, "invalid_request_error", "request_header_fields_too_large");
    // 400: malformed request line.
    check_error(roundtrip(f.port, "NOT A REQUEST\r\n\r\n"), 400, "invalid_request_error", "malformed_request");
    // The server is still healthy afterwards.
    CHECK(get(f.port, "/v1/sonder/health").status == 200);
}

TEST_CASE("capacity: connections over --max-connections get 503 overloaded") {
    auto o = Fixture::defaults();
    o.max_connections = 1;
    Fixture f(o);
    // Hold the only slot with a telemetry stream.
    Conn stream(f.port);
    stream.send(build_request("GET", "/v1/telemetry/sse?since=now", f.port));
    std::string acc;
    REQUIRE(stream.read_until(acc, "retry: 2000"));
    const Reply over = get(f.port, "/v1/sonder/health");
    check_error(over, 503, "service_unavailable", "overloaded");
    CHECK(over.header("retry-after") == "1");
    stream.close();
    CHECK(eventually([&] { return get(f.port, "/v1/sonder/health").status == 200; }));
}

TEST_CASE("capacity: a client that sends its request late still reads the 503") {
    // The accept thread answers before the request arrives. On Windows a
    // non-blocking connect often completes one scheduler tick (~15.6 ms)
    // late, so the request lands after the reply; if the server has already
    // closed by then, the request draws an RST and Windows discards the 503
    // still unread in the client's receive buffer.
    auto o = Fixture::defaults();
    o.max_connections = 1;
    Fixture f(o);
    Conn stream(f.port);
    stream.send(build_request("GET", "/v1/telemetry/sse?since=now", f.port));
    std::string acc;
    REQUIRE(stream.read_until(acc, "retry: 2000"));
    Conn late(f.port);
    std::this_thread::sleep_for(std::chrono::milliseconds(40));
    late.send(build_request("GET", "/v1/sonder/health", f.port));
    const Reply over = parse_reply(late.read_all(std::chrono::milliseconds(5000)));
    check_error(over, 503, "service_unavailable", "overloaded");
    stream.close();
}

TEST_CASE("startup: health reports starting and requests get 503 not_ready before models load") {
    auto o = Fixture::defaults();
    srv::Server server(o);
    Reply health;
    Reply chat;
    Reply models;
    auto st = server.start([&] {
        health = get(server.port(), "/v1/sonder/health");
        chat = post(server.port(), "/v1/chat/completions", chat_body());
        models = get(server.port(), "/v1/models");
    });
    REQUIRE(st.ok());
    CHECK(health.status == 503);
    CHECK(health.json().find("status")->as_string() == "starting");
    check_error(chat, 503, "service_unavailable", "not_ready");
    check_error(models, 503, "service_unavailable", "not_ready");
    CHECK(get(server.port(), "/v1/sonder/health").status == 200);
    server.stop();
}

TEST_CASE("bind: a port in use is reported as unavailable") {
    Fixture f;
    auto o = Fixture::defaults();
    o.port = f.port;
    srv::Server second(o);
    const auto st = second.start();
    CHECK(st.code() == si::ErrorCode::unavailable);
    CHECK(st.message().find("in use") != std::string::npos);
}

TEST_CASE("host check: duplicate Host headers are a 400 in either order") {
    Fixture f;
    const std::string port = std::to_string(f.port);
    for (const auto& pair : std::vector<std::pair<std::string, std::string>>{
             {"127.0.0.1:" + port, "evil.example"}, {"evil.example", "127.0.0.1:" + port}}) {
        const Reply r = roundtrip(f.port, build_request("GET", "/v1/sonder/health", f.port,
                                                        {{"Host", pair.first}, {"Host", pair.second}},
                                                        std::nullopt, /*host_header=*/false));
        check_error(r, 400, "invalid_request_error", "malformed_request");
    }
}

TEST_CASE("lingering close: a client that sends its whole body first still reads 413 and 401") {
    // Python urllib (Runtime's transport) writes the full request before it
    // reads: without a lingering close the early reply is lost to a reset.
    auto o = Fixture::defaults();
    o.max_body_bytes = 100000;
    o.token = "secret-token";
    Fixture f(o);
    const std::string big(2 * 1024 * 1024, 'x');
    const auto send_all_then_read = [&](const std::vector<std::pair<std::string, std::string>>& headers) {
        Conn c(f.port);
        c.send(build_request("POST", "/v1/chat/completions", f.port, headers, big));  // REQUIREs a full write
        Reply r = parse_reply(c.read_all());
        return r;
    };
    for (int i = 0; i < 3; ++i) {
        check_error(send_all_then_read({{"Authorization", "Bearer secret-token"}}), 413, "invalid_request_error",
                    "payload_too_large");
        check_error(send_all_then_read({}), 401, "authentication_error", "unauthorized");
    }
    CHECK(get(f.port, "/v1/sonder/health", {{"Authorization", "Bearer secret-token"}}).status == 200);
}

TEST_CASE("expect: 100-continue is answered before the body is read, other expectations get 417") {
    Fixture f;
    const std::string body = chat_body(R"(,"max_tokens":2)");
    {
        Conn c(f.port);
        c.send("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
               "Connection: close\r\nExpect: 100-continue\r\nContent-Length: " +
               std::to_string(body.size()) + "\r\n\r\n");
        std::string acc;
        // curl waits 1 s for this before sending the body anyway.
        const auto t0 = std::chrono::steady_clock::now();
        REQUIRE(c.read_until(acc, "\r\n\r\n", std::chrono::milliseconds(900)));
        CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::milliseconds(900));
        CHECK(acc == "HTTP/1.1 100 Continue\r\n\r\n");
        c.send(body);
        const Reply r = parse_reply(c.read_all());
        CHECK(r.status == 200);
        CHECK(r.json().find("object")->as_string() == "chat.completion");
    }
    {
        // A request that fails a pre-body check gets its error, never 100.
        auto o = Fixture::defaults();
        o.max_body_bytes = 16;
        Fixture small(o);
        Conn c(small.port);
        c.send("POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\nExpect: 100-continue\r\n"
               "Content-Length: 1000\r\n\r\n");
        const std::string got = c.read_all(std::chrono::milliseconds(5000));
        CHECK(got.find("100 Continue") == std::string::npos);
        CHECK(parse_reply(got).status == 413);
    }
    check_error(post(f.port, "/v1/chat/completions", body, {{"Expect", "fancy-feature"}}), 417,
                "invalid_request_error", "expectation_failed");
    check_error(get(f.port, "/v1/sonder/health", {{"Expect", "fancy-feature"}}), 417, "invalid_request_error",
                "expectation_failed");
    // HTTP/1.0 requests have Expect ignored (RFC 9110 section 10.1.1).
    const Reply old = roundtrip(f.port, "GET /v1/sonder/health HTTP/1.0\r\nHost: 127.0.0.1\r\nExpect: x\r\n\r\n");
    CHECK(old.status == 200);
}

#if defined(__linux__)
namespace {

std::size_t open_descriptors() {
    std::size_t n = 0;
    if (DIR* d = opendir("/proc/self/fd")) {
        while (readdir(d) != nullptr) ++n;
        closedir(d);
    }
    return n;  // includes ".", ".." and the directory's own descriptor
}

double cpu_seconds() {
    rusage ru{};
    getrusage(RUSAGE_SELF, &ru);
    return static_cast<double>(ru.ru_utime.tv_sec + ru.ru_stime.tv_sec) +
           static_cast<double>(ru.ru_utime.tv_usec + ru.ru_stime.tv_usec) / 1e6;
}

// Every other thread's context-switch count, or nothing if any of them is not
// blocked in the kernel (State S or D in /proc/self/task/<tid>/status).
std::optional<std::map<long, long long>> blocked_thread_switches(long self) {
    std::map<long, long long> switches;
    DIR* d = opendir("/proc/self/task");
    if (d == nullptr) return std::nullopt;
    bool all_blocked = true;
    while (dirent* e = readdir(d)) {
        if (e->d_name[0] == '.') continue;
        const long tid = std::atol(e->d_name);
        if (tid == self) continue;
        std::ifstream status(std::string("/proc/self/task/") + e->d_name + "/status");
        std::string line;
        char state = '?';
        long long count = 0;
        while (std::getline(status, line)) {
            if (line.rfind("State:", 0) == 0) {
                const auto pos = line.find_first_not_of(" \t", 6);
                state = pos == std::string::npos ? '?' : line[pos];
            } else if (line.rfind("voluntary_ctxt_switches:", 0) == 0 ||
                       line.rfind("nonvoluntary_ctxt_switches:", 0) == 0) {
                count += std::atoll(line.c_str() + line.find(':') + 1);
            }
        }
        if (state != 'S' && state != 'D') {
            all_blocked = false;
            break;
        }
        switches[tid] = count;
    }
    closedir(d);
    if (!all_blocked) return std::nullopt;
    return switches;
}

// Waits (bounded) until every other thread of the process is parked: blocked
// in the kernel with an unchanged context-switch count over several samples.
// clang's UBSan vptr check on std::thread's state object runs in the new
// thread's start-up code; if a server or engine thread first runs inside the
// descriptor-exhaustion window, that check cannot create its pipe() and
// reports a false "invalid vptr". One blocked snapshot is not enough: under
// ASan a new thread also waits (state S) for its creator before its start-up
// code, so the counts must stay still too. Returns false if the bound expired.
bool wait_for_other_threads_parked(std::chrono::milliseconds limit = std::chrono::milliseconds(2000)) {
    constexpr int kStableSamples = 3;
    const long self = static_cast<long>(::syscall(SYS_gettid));
    const auto deadline = std::chrono::steady_clock::now() + limit;
    std::optional<std::map<long, long long>> previous;
    int stable = 0;
    for (;;) {
        auto current = blocked_thread_switches(self);
        stable = (current && previous && *current == *previous) ? stable + 1 : 0;
        if (stable + 1 >= kStableSamples) return true;
        if (std::chrono::steady_clock::now() >= deadline) return false;
        previous = std::move(current);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
}

struct NoFileLimit {
    rlimit saved{};
    bool active = false;
    explicit NoFileLimit(rlim_t soft) {
        if (getrlimit(RLIMIT_NOFILE, &saved) != 0) return;
        rlimit lowered = saved;
        lowered.rlim_cur = std::min<rlim_t>(soft, saved.rlim_max);
        active = setrlimit(RLIMIT_NOFILE, &lowered) == 0;
    }
    ~NoFileLimit() {
        if (active) setrlimit(RLIMIT_NOFILE, &saved);
    }
};

}  // namespace

TEST_CASE("descriptor exhaustion: the accept loop answers 503 and does not spin") {
    auto o = Fixture::defaults();
    o.max_connections = 1000;
    o.read_timeout = std::chrono::milliseconds(30000);  // idle connections keep their descriptors
    std::vector<std::string> log_lines;
    std::mutex log_mu;
    o.log = [&](const std::string& line) {
        std::lock_guard<std::mutex> lock(log_mu);
        log_lines.push_back(line);
    };
    Fixture f(o);
    std::vector<det::Socket> clients;
    std::vector<int> fillers;
    std::size_t filled = 0;
    int connected = 0;
    int answered = 0;
    double idle_cpu = 0.0;
    {
        // Client sockets are created first, so connecting them later needs no
        // descriptor: the connections then wait in the backlog while the
        // server's accept() fails with EMFILE.
        for (int i = 0; i < 3; ++i) {
            det::Socket c(::socket(AF_INET, SOCK_STREAM, 0));
            REQUIRE(c.valid());
            clients.push_back(std::move(c));
        }
        // The telemetry writer can look blocked (state S) while it is briefly
        // waiting mid-batch on the start-up events; drain it first so it is
        // idle on its queue rather than still working when the window opens.
        f.server->engine()->telemetry().flush();
        // Not a failure: an expired bound only loses the protection against
        // the UBSan false positive. Reported before the window, so a run that
        // then aborts inside it still says why.
        WARN_MESSAGE(wait_for_other_threads_parked(),
                     "background threads were not all parked within 2 s before the descriptor window");
        NoFileLimit limit(static_cast<rlim_t>(open_descriptors() + 64));
        REQUIRE(limit.active);
        // No doctest assertions while the process is out of descriptors:
        // UBSan's vptr check needs a pipe() and would report false errors.
        // Results are recorded here and checked once descriptors are back.
        for (;;) {
            const int fd = ::open("/dev/null", O_RDONLY | O_CLOEXEC);
            if (fd < 0) break;
            fillers.push_back(fd);
        }
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(f.port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        for (auto& c : clients) {
            if (::connect(c.get(), reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) == 0) ++connected;
        }
        // Each is answered 503 through the reserve descriptor instead of
        // sitting in the backlog while the accept loop spins.
        for (auto& c : clients) {
            std::string got;
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            char buf[4096];
            while (got.find("\r\n\r\n") == std::string::npos && std::chrono::steady_clock::now() < deadline) {
                if (det::wait_socket(c.get(), false, std::chrono::milliseconds(50), nullptr) != det::WaitResult::ready) {
                    continue;
                }
                const long long n = det::recv_some(c.get(), buf, sizeof(buf));
                if (n <= 0 && n != -2) break;
                if (n > 0) got.append(buf, static_cast<std::size_t>(n));
            }
            if (got.rfind("HTTP/1.1 503 ", 0) == 0) ++answered;
        }
        // With the process at its limit, the server must stay idle.
        const double cpu0 = cpu_seconds();
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        idle_cpu = cpu_seconds() - cpu0;
        clients.clear();
        filled = fillers.size();
        for (const int fd : fillers) ::close(fd);
    }
    CHECK(filled >= 8);
    CHECK(connected == 3);
    CHECK(answered == 3);
    CHECK_MESSAGE(idle_cpu < 0.3, "CPU seconds used while idle at the descriptor limit: " << idle_cpu);
    // Descriptors are back: the server recovers.
    CHECK(eventually([&] { return get(f.port, "/v1/sonder/health").status == 200; }));
    std::lock_guard<std::mutex> lock(log_mu);
    bool warned = false;
    for (const auto& l : log_lines) warned = warned || l.find("out of file descriptors") != std::string::npos;
    CHECK(warned);
}
#endif

TEST_CASE("serve: models keep the default device unless --device is given") {
    {
        Fixture f;  // no device set
        auto loaded = f.of_type("model.load.completed");
        REQUIRE_FALSE(loaded.empty());
        const auto* device = loaded.front().find("device_id");
        REQUIRE(device != nullptr);
        CHECK(device->as_string() == "cpu:0");
    }
    {
        auto o = Fixture::defaults();
        o.device = "cpu:0";
        Fixture f(std::move(o));
        auto loaded = f.of_type("model.load.completed");
        REQUIRE_FALSE(loaded.empty());
        CHECK(loaded.front().find("device_id")->as_string() == "cpu:0");
    }
}

TEST_CASE("health: gpu_inventory_json serializes discrete NVIDIA, AMD RX XT, Ryzen APU UMA and unknown readings") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    // Headless / 0-GPU host serializes to an empty JSON array.
    const json::Array empty = det::gpu_inventory_json({});
    CHECK(empty.empty());

    si::GpuDeviceInfo rtx;
    rtx.id = "gpu:0";
    rtx.name = "NVIDIA GeForce RTX 5070 Ti";
    rtx.vendor = si::GpuVendor::nvidia;
    rtx.vendor_id = 0x10DEu;
    rtx.device_id = 0x2C05u;
    rtx.dedicated_total = si::MemoryReading::known(16303ull * kMiB);
    rtx.dedicated_free = si::MemoryReading::known(14800ull * kMiB);
    rtx.shared_total = si::MemoryReading::known(16ull * kGiB);
    rtx.is_integrated_uma = false;
    rtx.backend_hint = "cuda";
    rtx.source = "dxgi";

    si::GpuDeviceInfo rx_xt;
    rx_xt.id = "gpu:1";
    rx_xt.name = "AMD Radeon RX 7900 XT";
    rx_xt.vendor = si::GpuVendor::amd;
    rx_xt.vendor_id = 0x1002u;
    rx_xt.device_id = 0x744Cu;
    rx_xt.dedicated_total = si::MemoryReading::known(20ull * kGiB);
    rx_xt.dedicated_free = si::MemoryReading::known(18ull * kGiB);
    rx_xt.shared_total = si::MemoryReading::known(16ull * kGiB);
    rx_xt.is_integrated_uma = false;
    rx_xt.backend_hint = "rocm";
    rx_xt.source = "amdgpu-sysfs";

    si::GpuDeviceInfo ryzen_apu;
    ryzen_apu.id = "gpu:2";
    ryzen_apu.name = "AMD Radeon 780M Graphics";
    ryzen_apu.vendor = si::GpuVendor::amd;
    ryzen_apu.vendor_id = 0x1002u;
    ryzen_apu.device_id = 0x15BFu;
    ryzen_apu.dedicated_total = si::MemoryReading::known(512ull * kMiB);
    ryzen_apu.dedicated_free = si::MemoryReading::known(384ull * kMiB + 14ull * kGiB);
    ryzen_apu.shared_total = si::MemoryReading::known(16ull * kGiB);
    ryzen_apu.is_integrated_uma = true;
    ryzen_apu.backend_hint = "rocm";
    ryzen_apu.source = "amdgpu-sysfs";

    si::GpuDeviceInfo ryzen_apu_no_free = ryzen_apu;
    ryzen_apu_no_free.id = "gpu:3";
    ryzen_apu_no_free.dedicated_free = si::MemoryReading::unknown();

    si::GpuDeviceInfo unknown_mem;
    unknown_mem.id = "gpu:4";
    unknown_mem.name = "Unknown Adapter";
    unknown_mem.vendor = si::GpuVendor::unknown;
    unknown_mem.vendor_id = 0u;
    unknown_mem.device_id = 0u;
    unknown_mem.dedicated_total = si::MemoryReading::unknown();
    unknown_mem.dedicated_free = si::MemoryReading::unknown();
    unknown_mem.shared_total = si::MemoryReading::unknown();
    unknown_mem.is_integrated_uma = false;
    unknown_mem.backend_hint = "";
    unknown_mem.source = "dxgi";

    si::GpuDeviceInfo zero_free;
    zero_free.id = "gpu:5";
    zero_free.name = "NVIDIA GeForce RTX 4070";
    zero_free.vendor = si::GpuVendor::nvidia;
    zero_free.vendor_id = 0x10DEu;
    zero_free.device_id = 0x2786u;
    zero_free.dedicated_total = si::MemoryReading::known(12ull * kGiB);
    zero_free.dedicated_free = si::MemoryReading::known(0u);
    zero_free.shared_total = si::MemoryReading::unknown();
    zero_free.is_integrated_uma = false;
    zero_free.backend_hint = "cuda";
    zero_free.source = "nvml";

    const json::Array serialized =
        det::gpu_inventory_json({rtx, rx_xt, ryzen_apu, ryzen_apu_no_free, unknown_mem, zero_free});
    const auto parsed = json::parse(json::Value(serialized).dump());
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().to_string());
    REQUIRE(parsed.value().is_array());
    const auto& arr = parsed.value().as_array();
    REQUIRE(arr.size() == 6u);

    // 0: Discrete NVIDIA RTX
    const auto& j_rtx = arr[0];
    REQUIRE(j_rtx.is_object());
    CHECK(j_rtx.as_object().size() == 13u);
    CHECK(j_rtx.find("id")->as_string() == "gpu:0");
    CHECK(j_rtx.find("name")->as_string() == "NVIDIA GeForce RTX 5070 Ti");
    CHECK(j_rtx.find("vendor")->as_string() == "nvidia");
    CHECK(j_rtx.find("vendor_id")->as_uint() == 0x10DEu);
    CHECK(j_rtx.find("device_id")->as_uint() == 0x2C05u);
    CHECK(j_rtx.find("dedicated_total_mib")->as_uint() == 16303ull);
    CHECK(j_rtx.find("dedicated_free_mib")->as_uint() == 14800ull);
    CHECK(j_rtx.find("shared_total_mib")->as_uint() == 16384ull);
    CHECK(j_rtx.find("effective_vram_mib")->as_uint() == 16303ull);
    CHECK(j_rtx.find("effective_free_vram_mib")->as_uint() == 14800ull);
    CHECK_FALSE(j_rtx.find("is_integrated_uma")->as_bool());
    CHECK(j_rtx.find("backend_hint")->as_string() == "cuda");
    CHECK(j_rtx.find("source")->as_string() == "dxgi");

    // 1: Discrete AMD RX XT
    const auto& j_rx = arr[1];
    CHECK(j_rx.find("id")->as_string() == "gpu:1");
    CHECK(j_rx.find("name")->as_string() == "AMD Radeon RX 7900 XT");
    CHECK(j_rx.find("vendor")->as_string() == "amd");
    CHECK(j_rx.find("vendor_id")->as_uint() == 0x1002u);
    CHECK(j_rx.find("device_id")->as_uint() == 0x744Cu);
    CHECK(j_rx.find("dedicated_total_mib")->as_uint() == 20480ull);
    CHECK(j_rx.find("dedicated_free_mib")->as_uint() == 18432ull);
    CHECK(j_rx.find("shared_total_mib")->as_uint() == 16384ull);
    CHECK(j_rx.find("effective_vram_mib")->as_uint() == 20480ull);
    CHECK(j_rx.find("effective_free_vram_mib")->as_uint() == 18432ull);
    CHECK_FALSE(j_rx.find("is_integrated_uma")->as_bool());
    CHECK(j_rx.find("backend_hint")->as_string() == "rocm");
    CHECK(j_rx.find("source")->as_string() == "amdgpu-sysfs");

    // 2: Integrated AMD Ryzen 780M APU UMA (dedicated carveout + GTT pool)
    const auto& j_apu = arr[2];
    CHECK(j_apu.find("id")->as_string() == "gpu:2");
    CHECK(j_apu.find("name")->as_string() == "AMD Radeon 780M Graphics");
    CHECK(j_apu.find("vendor")->as_string() == "amd");
    CHECK(j_apu.find("vendor_id")->as_uint() == 0x1002u);
    CHECK(j_apu.find("device_id")->as_uint() == 0x15BFu);
    CHECK(j_apu.find("dedicated_total_mib")->as_uint() == 512ull);
    CHECK(j_apu.find("dedicated_free_mib")->as_uint() == 14720ull);
    CHECK(j_apu.find("shared_total_mib")->as_uint() == 16384ull);
    CHECK(j_apu.find("effective_vram_mib")->as_uint() == 16896ull);
    CHECK(j_apu.find("effective_free_vram_mib")->as_uint() == 14720ull);
    CHECK(j_apu.find("is_integrated_uma")->as_bool());
    CHECK(j_apu.find("backend_hint")->as_string() == "rocm");
    CHECK(j_apu.find("source")->as_string() == "amdgpu-sysfs");

    // 3: Integrated UMA with unknown free reading falls back to full UMA pool for effective_free_vram_mib
    const auto& j_apu_no_free = arr[3];
    CHECK(j_apu_no_free.find("dedicated_free_mib")->is_null());
    CHECK(j_apu_no_free.find("effective_vram_mib")->as_uint() == 16896ull);
    CHECK(j_apu_no_free.find("effective_free_vram_mib")->as_uint() == 16896ull);

    // 4: Unknown memory readings serialize strictly as JSON null
    const auto& j_unknown = arr[4];
    CHECK(j_unknown.find("vendor")->as_string() == "unknown");
    CHECK(j_unknown.find("vendor_id")->as_uint() == 0u);
    CHECK(j_unknown.find("device_id")->as_uint() == 0u);
    CHECK(j_unknown.find("dedicated_total_mib")->is_null());
    CHECK(j_unknown.find("dedicated_free_mib")->is_null());
    CHECK(j_unknown.find("shared_total_mib")->is_null());
    CHECK(j_unknown.find("effective_vram_mib")->is_null());
    CHECK(j_unknown.find("effective_free_vram_mib")->is_null());
    CHECK_FALSE(j_unknown.find("is_integrated_uma")->as_bool());
    CHECK(j_unknown.find("backend_hint")->as_string().empty());
    CHECK(j_unknown.find("source")->as_string() == "dxgi");

    // 5: Known-zero free VRAM stays integer 0 (distinct from unknown null)
    const auto& j_zero = arr[5];
    CHECK(j_zero.find("dedicated_total_mib")->as_uint() == 12288ull);
    REQUIRE(j_zero.find("dedicated_free_mib")->is_integer());
    CHECK(j_zero.find("dedicated_free_mib")->as_uint() == 0ull);
    CHECK(j_zero.find("shared_total_mib")->is_null());
    CHECK(j_zero.find("effective_vram_mib")->as_uint() == 12288ull);
    REQUIRE(j_zero.find("effective_free_vram_mib")->is_integer());
    CHECK(j_zero.find("effective_free_vram_mib")->as_uint() == 0ull);
}

TEST_CASE("health: additive gpus inventory array is present and well-formed on any host") {
    Fixture f;
    const Reply r = get(f.port, "/v1/sonder/health");
    REQUIRE(r.status == 200);
    const json::Value h = r.json();

    const json::Value* gpus = h.find("gpus");
    REQUIRE(gpus != nullptr);
    REQUIRE(gpus->is_array());
    for (const auto& entry : gpus->as_array()) {
        REQUIRE(entry.is_object());
        const auto* id = entry.find("id");
        const auto* name = entry.find("name");
        const auto* vendor = entry.find("vendor");
        const auto* vendor_id = entry.find("vendor_id");
        const auto* device_id = entry.find("device_id");
        const auto* ded_total = entry.find("dedicated_total_mib");
        const auto* ded_free = entry.find("dedicated_free_mib");
        const auto* shared_total = entry.find("shared_total_mib");
        const auto* eff_total = entry.find("effective_vram_mib");
        const auto* eff_free = entry.find("effective_free_vram_mib");
        const auto* uma = entry.find("is_integrated_uma");
        const auto* hint = entry.find("backend_hint");
        const auto* source = entry.find("source");
        REQUIRE(id != nullptr);
        REQUIRE(name != nullptr);
        REQUIRE(vendor != nullptr);
        REQUIRE(vendor_id != nullptr);
        REQUIRE(device_id != nullptr);
        REQUIRE(ded_total != nullptr);
        REQUIRE(ded_free != nullptr);
        REQUIRE(shared_total != nullptr);
        REQUIRE(eff_total != nullptr);
        REQUIRE(eff_free != nullptr);
        REQUIRE(uma != nullptr);
        REQUIRE(hint != nullptr);
        REQUIRE(source != nullptr);
        CHECK(id->is_string());
        CHECK_FALSE(id->as_string().empty());
        CHECK(name->is_string());
        CHECK(vendor->is_string());
        CHECK_FALSE(vendor->as_string().empty());
        CHECK(vendor_id->is_integer());
        CHECK(device_id->is_integer());
        CHECK((ded_total->is_null() || ded_total->is_integer()));
        CHECK((ded_free->is_null() || ded_free->is_integer()));
        CHECK((shared_total->is_null() || shared_total->is_integer()));
        CHECK((eff_total->is_null() || eff_total->is_integer()));
        CHECK((eff_free->is_null() || eff_free->is_integer()));
        CHECK(uma->is_bool());
        CHECK(hint->is_string());
        CHECK(source->is_string());
        CHECK_FALSE(source->as_string().empty());
    }

    // Adding `gpus` to /v1/sonder/health must not alter `sonder` metadata on /v1/health or /v1/models.
    const json::Value* sonder_meta = h.find("sonder");
    REQUIRE(sonder_meta != nullptr);
    CHECK(sonder_meta->find("features")->is_array());
    CHECK(sonder_meta->find("features")->as_array().empty());
    CHECK(sonder_meta->find("pins") != nullptr);

    const Reply models_reply = get(f.port, "/v1/models");
    REQUIRE(models_reply.status == 200);
    const auto models_doc = models_reply.json();
    CHECK(models_doc.find("sonder")->as_object().size() == 3u);
}

namespace {

struct TempModelDir {
    std::filesystem::path path;
    explicit TempModelDir(const char* prefix) : path(std::filesystem::temp_directory_path() / si::make_id(prefix)) {
        std::filesystem::create_directories(path);
    }
    ~TempModelDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    TempModelDir(const TempModelDir&) = delete;
    TempModelDir& operator=(const TempModelDir&) = delete;
};

class SyntheticGgufWriter {
public:
    void kv_str(const std::string& key, const std::string& value) {
        str(key);
        u32(8u);  // GGUF_TYPE_STRING
        str(value);
        ++kv_count_;
    }
    void kv_u32(const std::string& key, std::uint32_t value) {
        str(key);
        u32(4u);  // GGUF_TYPE_UINT32
        u32(value);
        ++kv_count_;
    }
    void tensor(const std::string& name, const std::vector<std::uint64_t>& dims, std::uint32_t type) {
        str(name);
        u32(static_cast<std::uint32_t>(dims.size()));
        for (const std::uint64_t d : dims) u64(d);
        u32(type);
        u64(0u);
        ++tensor_count_;
    }
    [[nodiscard]] std::string finish() const {
        std::string out = "GGUF";
        append_u32(out, 3u);
        append_u64(out, tensor_count_);
        append_u64(out, kv_count_);
        out += bytes_;
        return out;
    }

private:
    static void append_u32(std::string& dst, std::uint32_t v) {
        for (int i = 0; i < 4; ++i) dst.push_back(static_cast<char>((v >> (i * 8)) & 0xFFu));
    }
    static void append_u64(std::string& dst, std::uint64_t v) {
        for (int i = 0; i < 8; ++i) dst.push_back(static_cast<char>((v >> (i * 8)) & 0xFFu));
    }
    void u32(std::uint32_t v) { append_u32(bytes_, v); }
    void u64(std::uint64_t v) { append_u64(bytes_, v); }
    void str(const std::string& s) {
        u64(static_cast<std::uint64_t>(s.size()));
        bytes_.append(s);
    }

    std::string bytes_;
    std::uint64_t kv_count_ = 0;
    std::uint64_t tensor_count_ = 0;
};

std::string make_synthetic_gguf_bytes() {
    SyntheticGgufWriter w;
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.block_count", 4u);
    w.kv_u32("llama.context_length", 8192u);
    w.kv_u32("llama.embedding_length", 512u);
    w.kv_u32("llama.vocab_size", 32000u);
    w.kv_u32("llama.attention.head_count", 8u);
    w.kv_u32("llama.attention.head_count_kv", 4u);
    w.tensor("token_embd.weight", {512u, 32000u}, 2u);
    for (std::uint32_t b = 0; b < 4u; ++b) {
        w.tensor("blk." + std::to_string(b) + ".attn_q.weight", {512u, 512u}, 2u);
        w.tensor("blk." + std::to_string(b) + ".ffn_up.weight", {512u, 1024u}, 2u);
    }
    w.tensor("output.weight", {512u, 32000u}, 2u);
    return w.finish();
}

void write_binary_file(const std::filesystem::path& path, const std::string& content) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE_MESSAGE(out.good(), "failed to create test fixture file");
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    REQUIRE(out.good());
}

}  // namespace

TEST_CASE("node: capability inventory returns schema, hardware, backends, models and honors auth") {
    {
        auto o = Fixture::defaults();
        o.profile_catalog.emplace_back(json::Object{{"name", "mock:tiny"}, {"backend", "mock"}, {"served", true}});
        Fixture f(o);

        const Reply r = get(f.port, "/v1/sonder/node");
        REQUIRE(r.status == 200);
        CHECK(r.complete);
        check_common_headers(r);
        CHECK(r.header("content-type") == "application/json");
        const json::Value doc = r.json();

        CHECK(doc.find("schema")->as_string() == "sonder.inference.node/1");
        CHECK_FALSE(doc.find("node_id")->as_string().empty());
        CHECK(doc.find("instance_id")->as_string() == f.server->instance_id());
        CHECK_FALSE(doc.find("platform")->as_string().empty());
        CHECK_FALSE(doc.find("version")->as_string().empty());
        CHECK_FALSE(doc.find("commit")->as_string().empty());
        CHECK(doc.find("abi_version")->as_int() == 1);
        CHECK(doc.find("api_version")->as_int() == 1);
        CHECK(doc.find("synthetic")->as_bool());

        const json::Value* host_mem = doc.find("host_memory");
        REQUIRE(host_mem != nullptr);
        REQUIRE(host_mem->is_object());
        REQUIRE(host_mem->find("total_physical_bytes") != nullptr);
        REQUIRE(host_mem->find("available_physical_bytes") != nullptr);
        REQUIRE(host_mem->find("available_commit_bytes") != nullptr);
        REQUIRE(host_mem->find("source") != nullptr);
        CHECK((host_mem->find("total_physical_bytes")->is_null() ||
               host_mem->find("total_physical_bytes")->is_integer()));
        CHECK((host_mem->find("available_physical_bytes")->is_null() ||
               host_mem->find("available_physical_bytes")->is_integer()));
        CHECK((host_mem->find("available_commit_bytes")->is_null() ||
               host_mem->find("available_commit_bytes")->is_integer()));
        CHECK((host_mem->find("source")->is_null() || host_mem->find("source")->is_string()));

        const json::Value* devices = doc.find("devices");
        REQUIRE(devices != nullptr);
        REQUIRE(devices->is_array());
        REQUIRE_FALSE(devices->as_array().empty());
        const auto& cpu = devices->as_array().front();
        CHECK(cpu.find("id")->is_string());
        CHECK(cpu.find("kind")->as_string() == "cpu");
        CHECK(cpu.find("name")->is_string());
        CHECK(cpu.find("logical_cores")->is_integer());
        CHECK(cpu.find("total_memory_bytes")->is_integer());
        CHECK(cpu.find("available_memory_bytes")->is_integer());

        const json::Value* gpus = doc.find("gpus");
        REQUIRE(gpus != nullptr);
        CHECK(gpus->is_array());

        const json::Value* combined_vram = doc.find("combined_gpu_vram_mib");
        REQUIRE(combined_vram != nullptr);
        CHECK(combined_vram->is_integer());

        const json::Value* gpu_pool = doc.find("gpu_pool");
        REQUIRE(gpu_pool != nullptr);
        REQUIRE(gpu_pool->is_object());
        REQUIRE(gpu_pool->find("discrete_vram_mib") != nullptr);
        REQUIRE(gpu_pool->find("integrated_uma_vram_mib") != nullptr);
        REQUIRE(gpu_pool->find("combined_gpu_vram_mib") != nullptr);
        REQUIRE(gpu_pool->find("discrete_free_vram_mib") != nullptr);
        REQUIRE(gpu_pool->find("integrated_uma_free_vram_mib") != nullptr);
        REQUIRE(gpu_pool->find("combined_free_gpu_vram_mib") != nullptr);
        REQUIRE(gpu_pool->find("discrete_gpu_count") != nullptr);
        REQUIRE(gpu_pool->find("integrated_uma_gpu_count") != nullptr);
        REQUIRE(gpu_pool->find("hybrid_vulkan_pool_available") != nullptr);
        REQUIRE(gpu_pool->find("recommended_devices") != nullptr);
        REQUIRE(gpu_pool->find("llama_arg_device") != nullptr);
        CHECK(gpu_pool->find("combined_gpu_vram_mib")->as_uint() == combined_vram->as_uint());
        CHECK(gpu_pool->find("hybrid_vulkan_pool_available")->is_bool());
        CHECK(gpu_pool->find("recommended_devices")->is_array());
        CHECK((gpu_pool->find("llama_arg_device")->is_null() || gpu_pool->find("llama_arg_device")->is_string()));

        const json::Value* backends = doc.find("backends");
        REQUIRE(backends != nullptr);
        REQUIRE(backends->is_array());
        REQUIRE(backends->as_array().size() == 1u);
        CHECK(backends->as_array()[0].find("name")->as_string() == "mock");
        CHECK(backends->as_array()[0].find("available")->as_bool());
        CHECK(backends->as_array()[0].find("capabilities")->is_array());

        const json::Value* models = doc.find("models");
        REQUIRE(models != nullptr);
        REQUIRE(models->is_array());
        REQUIRE(models->as_array().size() == 1u);
        CHECK(models->as_array()[0].find("id")->as_string() == "mock:tiny");
        CHECK(models->as_array()[0].find("backend")->as_string() == "mock");
        CHECK(models->as_array()[0].find("state")->as_string() == "resident");
        CHECK(models->as_array()[0].find("resident")->as_bool());
        CHECK(models->as_array()[0].find("default")->as_bool());

        const json::Value* profiles = doc.find("profiles");
        REQUIRE(profiles != nullptr);
        REQUIRE(profiles->is_array());
        REQUIRE(profiles->as_array().size() == 1u);
        CHECK(profiles->as_array()[0].find("name")->as_string() == "mock:tiny");

        const json::Value* caps = doc.find("capabilities");
        REQUIRE(caps != nullptr);
        REQUIRE(caps->is_object());
        CHECK(caps->find("ternary_gguf")->as_bool());
        CHECK(caps->find("placement_planner")->as_bool());
        CHECK(caps->find("coresidency_planner")->as_bool());
        CHECK(caps->find("multi_gpu")->as_bool());
        CHECK(caps->find("hybrid_gpu_spread")->as_bool());
        CHECK(caps->find("uma_telemetry")->as_bool());
#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
        CHECK(caps->find("adaptive_speculation")->as_bool());
#else
        CHECK_FALSE(caps->find("adaptive_speculation")->as_bool());
#endif

        CHECK(doc.find("sonder")->find("api_version")->as_int() == 1);

        // Wrong HTTP method returns 405 with accurate Allow header.
        const Reply wrong = post(f.port, "/v1/sonder/node", "{}");
        check_error(wrong, 405, "invalid_request_error", "method_not_allowed");
        CHECK(wrong.header("allow") == "GET, OPTIONS");
    }
    {
        // Bearer auth enforcement and no token/path leakage.
        TempModelDir dir("sonder-node-dir");
        auto o = Fixture::defaults();
        o.token = "node-secret-token";
        o.backend.model_dirs = {dir.path.string()};
        Fixture f(o);

        const Reply missing = get(f.port, "/v1/sonder/node");
        check_error(missing, 401, "authentication_error", "unauthorized");
        CHECK(missing.header("www-authenticate") == "Bearer");

        check_error(get(f.port, "/v1/sonder/node", {{"Authorization", "Bearer wrong-token"}}), 401,
                    "authentication_error", "unauthorized");

        const Reply ok = get(f.port, "/v1/sonder/node", {{"Authorization", "Bearer node-secret-token"}});
        REQUIRE(ok.status == 200);
        CHECK(ok.json().find("schema")->as_string() == "sonder.inference.node/1");
        CHECK(ok.json().find("profiles")->as_array().empty());
        CHECK(ok.body.find("node-secret-token") == std::string::npos);
        CHECK(ok.body.find(dir.path.string()) == std::string::npos);
        CHECK(ok.body.find(dir.path.filename().string()) == std::string::npos);
    }
}

TEST_CASE("discovery: links advertise node and plan endpoints") {
    Fixture f;
    const Reply r = get(f.port, "/.well-known/sonder-telemetry");
    REQUIRE(r.status == 200);
    const json::Value doc = r.json();
    const json::Value* links = doc.find("links");
    REQUIRE(links != nullptr);
    REQUIRE(links->is_object());
    CHECK(links->find("health")->as_string() == "/v1/sonder/health");
    CHECK(links->find("identity")->as_string() == "/v1/sonder/identity");
    CHECK(links->find("models")->as_string() == "/v1/models");
    CHECK(links->find("node")->as_string() == "/v1/sonder/node");
    CHECK(links->find("plan")->as_string() == "/v1/sonder/plan");
}

TEST_CASE("plan: rejects bad methods, oversized bodies, malformed JSON, unknown keys, path traversal and missing targets") {
    TempModelDir model_dir("sonder-plan-reject");
    TempModelDir outside_dir("sonder-plan-outside");
    write_binary_file(model_dir.path / "notes.txt", "not a model");
    write_binary_file(outside_dir.path / "outside.gguf", make_synthetic_gguf_bytes());
    write_binary_file(model_dir.path / "corrupt.gguf", "not-a-valid-gguf-header");

    std::error_code sym_ec;
    std::filesystem::create_symlink(outside_dir.path / "outside.gguf", model_dir.path / "escape.gguf", sym_ec);
    const bool symlink_created = !sym_ec;

    auto o = Fixture::defaults();
    o.backend.model_dirs = {model_dir.path.string()};
    Fixture f(o);

    // Wrong method returns 405 with accurate Allow header.
    const Reply wrong_method = get(f.port, "/v1/sonder/plan");
    check_error(wrong_method, 405, "invalid_request_error", "method_not_allowed");
    CHECK(wrong_method.header("allow") == "POST, OPTIONS");

    // Body limit for /v1/sonder/plan is capped at 64 KiB even when max_body_bytes is 4 MiB.
    const std::string oversized(65u * 1024u, ' ');
    check_error(post(f.port, "/v1/sonder/plan", oversized), 413, "invalid_request_error", "payload_too_large");

    // Malformed JSON and non-object payloads.
    check_error(post(f.port, "/v1/sonder/plan", "{"), 400, "invalid_request_error", "invalid_request");
    check_error(post(f.port, "/v1/sonder/plan", "[]"), 400, "invalid_request_error", "invalid_request");
    check_error(post(f.port, "/v1/sonder/plan", "42"), 400, "invalid_request_error", "invalid_request");
    check_error(post(f.port, "/v1/sonder/plan", "{}"), 400, "invalid_request_error", "invalid_request");
    check_error(post(f.port, "/v1/sonder/plan", R"({"profile":"a","model":"b.gguf"})"), 400, "invalid_request_error",
                "invalid_request");

    // Unknown keys rejected strictly.
    check_error(post(f.port, "/v1/sonder/plan", R"({"model":"x.gguf","model_path":"/etc/passwd"})"), 400,
                "invalid_request_error", "invalid_request");
    check_error(post(f.port, "/v1/sonder/plan", R"({"model":"x.gguf","bogus":1})"), 400, "invalid_request_error",
                "invalid_request");

    // Path traversal and invalid target names rejected with 400.
    for (const char* bad_body : {
             R"({"model":"../secret.gguf"})",
             R"({"model":"..\\secret.gguf"})",
             R"({"model":"C:\\foo.gguf"})",
             R"({"model":"/etc/passwd"})",
             R"({"model":"sub/model.gguf"})",
             R"({"model":"model.gguf:stream"})",
             R"({"model":"bad\u0000name.gguf"})",
             R"({"model":""})",
             R"({"profile":"../secret"})",
             R"({"profile":"/etc/passwd"})",
             R"({"model":123})",
             R"({"profile":true})",
         }) {
        CAPTURE(bad_body);
        check_error(post(f.port, "/v1/sonder/plan", bad_body), 400, "invalid_request_error", "invalid_request");
    }

    // Strict type and range checks on optional tuning overrides.
    for (const char* bad_override : {
             R"({"model":"missing.gguf","ctx_size":64})",
             R"({"model":"missing.gguf","ctx_size":2000000})",
             R"({"model":"missing.gguf","ctx_size":"4096"})",
             R"({"model":"missing.gguf","ctx_size":4096.5})",
             R"({"model":"missing.gguf","min_ctx_size":100})",
             R"({"model":"missing.gguf","parallel":0})",
             R"({"model":"missing.gguf","parallel":257})",
             R"({"model":"missing.gguf","vram_headroom_mib":-1})",
             R"({"model":"missing.gguf","vram_headroom_mib":70000})",
             R"({"model":"missing.gguf","prefer_free_vram":"true"})",
             R"({"model":"missing.gguf","allow_quantized_kv":1})",
             R"({"model":"missing.gguf","allow_speculative":null})",
             R"({"model":"missing.gguf","allow_moe_cpu_offload":"no"})",
             R"({"model":"missing.gguf","allow_hybrid_gpu_spread":"yes"})",
             R"({"model":"missing.gguf","device_hint":7})",
         }) {
        CAPTURE(bad_override);
        check_error(post(f.port, "/v1/sonder/plan", bad_override), 400, "invalid_request_error", "invalid_request");
    }
    const std::string long_hint = R"({"model":"missing.gguf","device_hint":")" + std::string(129, 'a') + R"("})";
    check_error(post(f.port, "/v1/sonder/plan", long_hint), 400, "invalid_request_error", "invalid_request");

    // Missing model, missing profile, and non-GGUF file return 404 without leaking host paths.
    const Reply missing_model = post(f.port, "/v1/sonder/plan", R"({"model":"missing.gguf"})");
    check_error(missing_model, 404, "not_found_error", "not_found");
    CHECK(missing_model.body.find(model_dir.path.filename().string()) == std::string::npos);

    const Reply missing_profile = post(f.port, "/v1/sonder/plan", R"({"profile":"missing-profile"})");
    check_error(missing_profile, 404, "not_found_error", "not_found");
    CHECK(missing_profile.body.find(model_dir.path.filename().string()) == std::string::npos);

    const Reply non_gguf = post(f.port, "/v1/sonder/plan", R"({"model":"notes.txt"})");
    check_error(non_gguf, 404, "not_found_error", "not_found");

    // Malformed GGUF inside model_dir returns 400 with host path redacted.
    const Reply corrupt = post(f.port, "/v1/sonder/plan", R"({"model":"corrupt.gguf"})");
    check_error(corrupt, 400, "invalid_request_error", "invalid_request");
    CHECK(corrupt.body.find(model_dir.path.filename().string()) == std::string::npos);

    // Symlink escaping model_dir must be refused.
    if (symlink_created) {
        const Reply escaped = post(f.port, "/v1/sonder/plan", R"({"model":"escape.gguf"})");
        check_error(escaped, 404, "not_found_error", "not_found");
        CHECK(escaped.body.find(outside_dir.path.filename().string()) == std::string::npos);
        CHECK(escaped.body.find(model_dir.path.filename().string()) == std::string::npos);
    }

    // Auth gating on POST /v1/sonder/plan when a bearer token is configured.
    auto auth_opts = Fixture::defaults();
    auth_opts.token = "plan-secret-token";
    Fixture auth_f(auth_opts);
    const Reply unauth = post(auth_f.port, "/v1/sonder/plan", R"({"model":"missing.gguf"})");
    check_error(unauth, 401, "authentication_error", "unauthorized");
    CHECK(unauth.header("www-authenticate") == "Bearer");
}

TEST_CASE("plan: plans a synthetic GGUF in model_dirs and via bound profile with redacted basename-only paths") {
    TempModelDir model_dir("sonder-plan-models");
    TempModelDir bound_dir("sonder-plan-bound");
    const auto gguf_path = model_dir.path / "synthetic-bonsai.gguf";
    const auto bound_gguf_path = bound_dir.path / "bound-model.gguf";
    const auto bound_mmproj_path = bound_dir.path / "bound-mmproj.gguf";
    const std::string gguf_bytes = make_synthetic_gguf_bytes();
    write_binary_file(gguf_path, gguf_bytes);
    write_binary_file(bound_gguf_path, gguf_bytes);
    write_binary_file(bound_mmproj_path, std::string(4096, '\0'));

    auto o = Fixture::defaults();
    o.models = {"mock:tiny", "bound-prof"};
    o.backend.model_dirs = {model_dir.path.string()};
    srv::ModelProfileBinding binding;
    binding.model = "bound-prof";
    binding.load_name = "mock:bound";
    o.profile_bindings.push_back(std::move(binding));
    o.backend.llamaserver_args = {
        "--model", bound_gguf_path.string(), "--alias", "bound-prof", "--mmproj", bound_mmproj_path.string()};
    Fixture f(o);

    // 1. Plan by clean basename inside model_dirs with explicit tuning overrides.
    const Reply r = post(
        f.port, "/v1/sonder/plan",
        R"({"model":"synthetic-bonsai.gguf","ctx_size":2048,"min_ctx_size":1024,"parallel":2,"vram_headroom_mib":512,"prefer_free_vram":true,"allow_quantized_kv":false,"allow_speculative":false,"allow_moe_cpu_offload":true,"allow_hybrid_gpu_spread":true})");
    REQUIRE(r.status == 200);
    check_common_headers(r);
    CHECK(r.header("content-type") == "application/json");
    const json::Value doc = r.json();

    CHECK(doc.find("schema")->as_string() == "sonder.inference.plan/1");
    CHECK(doc.find("sonder")->find("api_version")->as_int() == 1);
    REQUIRE(doc.find("combined_gpu_vram_mib") != nullptr);
    CHECK(doc.find("combined_gpu_vram_mib")->is_integer());
    REQUIRE(doc.find("fits_in_vram") != nullptr);
    CHECK(doc.find("fits_in_vram")->is_bool());
    REQUIRE(doc.find("fits_in_host_ram") != nullptr);
    CHECK(doc.find("fits_in_host_ram")->is_bool());
    REQUIRE(doc.find("zero_cpu_spill") != nullptr);
    CHECK(doc.find("zero_cpu_spill")->is_bool());
    REQUIRE(doc.find("recommended_devices") != nullptr);
    CHECK(doc.find("recommended_devices")->is_array());
    const json::Value* spread = doc.find("multi_gpu_spread");
    REQUIRE(spread != nullptr);
    REQUIRE(spread->is_object());
    REQUIRE(spread->find("active") != nullptr);
    CHECK(spread->find("active")->is_bool());
    REQUIRE(spread->find("hybrid_discrete_igpu") != nullptr);
    CHECK(spread->find("hybrid_discrete_igpu")->is_bool());
    REQUIRE(spread->find("zero_cpu_spill") != nullptr);
    CHECK(spread->find("zero_cpu_spill")->is_bool());
    REQUIRE(spread->find("recommended_devices") != nullptr);
    CHECK(spread->find("recommended_devices")->is_array());
    REQUIRE(spread->find("per_device_split") != nullptr);
    CHECK(spread->find("per_device_split")->is_array());

    const json::Value* prof = doc.find("profile");
    REQUIRE(prof != nullptr);
    REQUIRE(prof->is_object());
    CHECK(prof->find("name")->as_string() == "synthetic-bonsai");
    CHECK(prof->find("model")->as_string() == "synthetic-bonsai.gguf");
    CHECK(prof->find("ctx_size")->as_uint() == 2048u);
    CHECK(prof->find("parallel")->as_uint() == 2u);
    CHECK(prof->find("cache_type_k")->as_string() == "f16");
    CHECK(prof->find("cache_type_v")->as_string() == "f16");
    CHECK(prof->find("speculative") == nullptr);

    const json::Value* est = doc.find("estimate");
    REQUIRE(est != nullptr);
    REQUIRE(est->is_object());
    CHECK(est->find("total_bytes")->is_integer());
    CHECK(est->find("host_resident_bytes")->is_integer());
    CHECK((doc.find("fits_in_vram")->as_bool() ? est->find("total_bytes")->as_uint() > 0u
                                               : est->find("host_resident_bytes")->as_uint() > 0u));

    // Full host filesystem directory path must never appear anywhere in the HTTP body.
    CHECK(r.body.find(model_dir.path.string()) == std::string::npos);
    CHECK(r.body.find(model_dir.path.filename().string()) == std::string::npos);

    // 2. Deterministic CPU-only planning via device_hint="cpu" works identically on GPU and headless hosts.
    const Reply cpu_r =
        post(f.port, "/v1/sonder/plan", R"({"model":"synthetic-bonsai.gguf","device_hint":"cpu","ctx_size":1024})");
    REQUIRE(cpu_r.status == 200);
    const json::Value cpu_doc = cpu_r.json();
    CHECK(cpu_doc.find("recommended_backend_hint")->as_string() == "cpu");
    CHECK_FALSE(cpu_doc.find("fits_in_vram")->as_bool());
    CHECK(cpu_doc.find("profile")->find("model")->as_string() == "synthetic-bonsai.gguf");
    CHECK(cpu_doc.find("profile")->find("n_gpu_layers")->as_int() == 0);
    CHECK(cpu_doc.find("estimate")->find("host_resident_bytes")->as_uint() > 0u);

    // 3. Plan by bound profile resolves operator-configured model + mmproj and redacts both to basenames.
    const Reply prof_r = post(f.port, "/v1/sonder/plan", R"({"profile":"bound-prof","device_hint":"cpu"})");
    REQUIRE(prof_r.status == 200);
    const json::Value prof_doc = prof_r.json();
    CHECK(prof_doc.find("schema")->as_string() == "sonder.inference.plan/1");
    CHECK(prof_doc.find("profile")->find("name")->as_string() == "bound-prof");
    CHECK(prof_doc.find("profile")->find("model")->as_string() == "bound-model.gguf");
    REQUIRE(prof_doc.find("profile")->find("mmproj") != nullptr);
    CHECK(prof_doc.find("profile")->find("mmproj")->as_string() == "bound-mmproj.gguf");
    CHECK(prof_r.body.find(bound_dir.path.string()) == std::string::npos);
    CHECK(prof_r.body.find(bound_dir.path.filename().string()) == std::string::npos);

    // 4. Bound profile whose --mmproj is missing or non-regular omits mmproj rather than emitting a 0-byte phantom.
    auto o_missing_mm = Fixture::defaults();
    o_missing_mm.models = {"mock:tiny", "bound-nomm"};
    srv::ModelProfileBinding binding_nomm;
    binding_nomm.model = "bound-nomm";
    binding_nomm.load_name = "mock:bound-nomm";
    o_missing_mm.profile_bindings.push_back(std::move(binding_nomm));
    o_missing_mm.backend.llamaserver_args = {
        "--model", bound_gguf_path.string(), "--alias", "bound-nomm", "--mmproj",
        (bound_dir.path / "does-not-exist-mmproj.gguf").string()};
    Fixture f_nomm(o_missing_mm);
    const Reply nomm_r = post(f_nomm.port, "/v1/sonder/plan", R"({"profile":"bound-nomm","device_hint":"cpu"})");
    REQUIRE(nomm_r.status == 200);
    const json::Value nomm_doc = nomm_r.json();
    CHECK(nomm_doc.find("profile")->find("model")->as_string() == "bound-model.gguf");
    CHECK(nomm_doc.find("profile")->find("mmproj") == nullptr);
    CHECK(nomm_r.body.find(bound_dir.path.filename().string()) == std::string::npos);
}
