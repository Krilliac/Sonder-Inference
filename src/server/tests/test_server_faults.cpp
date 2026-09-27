// Resource faults in request handling must fail one request, never the
// process: a thread the OS refuses to start is answered with 503.
#include <doctest/doctest.h>

#include <string>

#include "server_test_support.hpp"
#include "src/test_hooks.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

TEST_CASE("chat: a refused disconnect-watcher thread is 503 overloaded, and the server keeps serving") {
    Fixture f;
    det::fail_watcher_spawns_for_test().store(1);
    const Reply refused = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)"));
    det::fail_watcher_spawns_for_test().store(0);
    REQUIRE_MESSAGE(refused.status == 503, refused.body);
    CHECK(refused.header("retry-after") == "1");
    const json::Value doc = refused.json();
    CHECK(doc.find("error")->find("code")->as_string() == "overloaded");
    // Nothing ran for the refused request.
    CHECK(f.of_type("request.started").empty());

    // The process survived and the refused request left nothing in flight.
    const Reply ok = post(f.port, "/v1/chat/completions", chat_body(R"(,"max_tokens":4)"));
    CHECK(ok.status == 200);
    const Reply health = get(f.port, "/v1/sonder/health");
    CHECK(health.status == 200);
}
