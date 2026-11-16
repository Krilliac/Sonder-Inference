#include <doctest/doctest.h>

#include <string>
#include <vector>

#include "../src/warmup_config.hpp"

using namespace sonder::inference;
namespace json = sonder::inference::json;

namespace {
json::Value parse(const std::string &text) {
    const auto value = json::parse(text);
    REQUIRE(value.ok());
    return value.value();
}
}  // namespace

TEST_CASE("warmup config unknown-field error bounds the echoed key") {
    BackendSetup setup;
    const auto st = detail::parse_warmup_config(parse("{\"" + std::string(990, 'k') + "\":1}"), setup);
    REQUIRE_FALSE(st.ok());
    CHECK(st.message().size() < 1024);
    CHECK(st.message().find(std::string(64, 'k') + "...") != std::string::npos);
    CHECK(st.message().find(std::string(65, 'k')) == std::string::npos);
    const auto shortkey = detail::parse_warmup_config(parse(R"({"zzz":1})"), setup);
    CHECK(shortkey.message().find("'zzz'") != std::string::npos);
}

TEST_CASE("warmup config rejection leaves the setup unchanged") {
    BackendSetup setup;
    setup.llamaserver_warmup_messages_file = "keep.json";
    setup.llamaserver_warmup_on_restart = !setup.llamaserver_warmup_on_restart;
    setup.llamaserver_warmup_all_slots = false;
    setup.llamaserver_warmup_slots = {4, 5};
    const auto max_prefix = setup.llamaserver_warmup_max_prefix_chars;
    const auto on_restart = setup.llamaserver_warmup_on_restart;

    // Earlier fields are valid, a later one is not: nothing may be applied.
    const char *const invalid[] = {
        R"({"messages_file":"new.json","on_restart":"nope"})",
        R"({"messages_file":"new.json","chat_template_kwargs":{"a":1},"max_prefix_chars":0})",
        R"({"messages_file":"a\u0000b"})",
        R"({"slots":[1,2,1],"messages_file":"new.json"})",
        R"({"slots":"all","max_prefix_chars":99999999})"};
    for (const char *text : invalid) {
        CAPTURE(text);
        CHECK_FALSE(detail::parse_warmup_config(parse(text), setup).ok());
        CHECK(setup.llamaserver_warmup_messages_file == "keep.json");
        CHECK(setup.llamaserver_warmup_chat_template_kwargs.empty());
        CHECK(setup.llamaserver_warmup_on_restart == on_restart);
        CHECK(setup.llamaserver_warmup_max_prefix_chars == max_prefix);
        CHECK_FALSE(setup.llamaserver_warmup_all_slots);
        CHECK(setup.llamaserver_warmup_slots == std::vector<std::uint32_t>{4, 5});
    }
}

TEST_CASE("warmup config accepts a full valid document") {
    BackendSetup setup;
    REQUIRE(detail::parse_warmup_config(
                parse(R"({"messages_file":"w.json","chat_template_kwargs":{"a":1},"slots":[0,3],
                          "on_restart":false,"max_prefix_chars":100})"),
                setup)
                .ok());
    CHECK(setup.llamaserver_warmup_messages_file == "w.json");
    CHECK(setup.llamaserver_warmup_chat_template_kwargs.size() == 1);
    CHECK_FALSE(setup.llamaserver_warmup_on_restart);
    CHECK(setup.llamaserver_warmup_max_prefix_chars == 100);
    CHECK(setup.llamaserver_warmup_slots == std::vector<std::uint32_t>{0, 3});
}
