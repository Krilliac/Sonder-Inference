// HTTP behaviour of `sonder-infer serve` against an in-process server on an
// ephemeral loopback port (MOCK backend): routes and response shapes, common
// headers, auth, the Host check, CORS, request limits and the connection cap.
#include <doctest/doctest.h>

#include <algorithm>
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
#  include <fstream>
#  include <map>
#  include <optional>
#endif

#include "server_test_support.hpp"

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
    check_error(post(f.port, "/v1/embeddings", R"({"input":"x"})"), 501, "not_implemented_error", "not_implemented");
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
          "X-Sonder-Parent-Request-Id, X-Sonder-Agent-Id, X-Sonder-Task-Id, X-Sonder-Workload, X-Sonder-Priority");
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
               "Expect: 100-continue\r\nContent-Length: " +
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
