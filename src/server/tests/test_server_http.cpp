// HTTP behaviour of `sonder-infer serve` against an in-process server on an
// ephemeral loopback port (MOCK backend): routes and response shapes, common
// headers, auth, the Host check, CORS, request limits and the connection cap.
#include <doctest/doctest.h>

#include <string>
#include <thread>

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
