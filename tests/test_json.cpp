#include <doctest/doctest.h>

#include <limits>

#include "sonder/inference/json.hpp"

namespace json = sonder::inference::json;

TEST_SUITE("json") {
TEST_CASE("round-trips nested values and preserves key order") {
    json::Object o{{"b", 1}, {"a", json::Array{true, nullptr, 2.5, "x"}}, {"c", json::Object{{"k", "v"}}}};
    const std::string text = json::Value(o).dump();
    CHECK(text == R"({"b":1,"a":[true,null,2.5,"x"],"c":{"k":"v"}})");
    auto parsed = json::parse(text);
    REQUIRE(parsed.ok());
    CHECK(parsed.value().dump() == text);
}

TEST_CASE("distinguishes integers from doubles") {
    auto v = json::parse(R"({"i":9007199254740993,"d":1.0,"e":1e3,"n":-4})");
    REQUIRE(v.ok());
    CHECK(v.value().find("i")->is_integer());
    CHECK(v.value().find("i")->as_int() == 9007199254740993LL);
    CHECK(v.value().find("d")->type() == json::Type::number);
    CHECK(v.value().find("e")->as_double() == doctest::Approx(1000.0));
    CHECK(v.value().find("n")->as_int() == -4);
    CHECK(json::Value(3.0).dump() == "3.0");
}

TEST_CASE("escapes and unescapes strings") {
    const std::string raw = "quote\" back\\ nl\n tab\t ctl\x01 utf8 \xC3\xA9";
    const std::string text = json::Value(raw).dump();
    CHECK(text == "\"quote\\\" back\\\\ nl\\n tab\\t ctl\\u0001 utf8 \xC3\xA9\"");
    auto back = json::parse(text);
    REQUIRE(back.ok());
    CHECK(back.value().as_string() == raw);
    auto surrogate = json::parse(R"("\ud83d\ude00")");
    REQUIRE(surrogate.ok());
    CHECK(surrogate.value().as_string() == "\xF0\x9F\x98\x80");
}

TEST_CASE("rejects malformed documents") {
    CHECK_FALSE(json::parse("").ok());
    CHECK_FALSE(json::parse("{").ok());
    CHECK_FALSE(json::parse("[1,]").ok());
    CHECK_FALSE(json::parse("{\"a\" 1}").ok());
    CHECK_FALSE(json::parse("01").ok());
    CHECK_FALSE(json::parse("true false").ok());
    CHECK_FALSE(json::parse("\"unterminated").ok());
    CHECK(json::parse("nul").status().code() == sonder::inference::ErrorCode::protocol_error);
    std::string deep(200, '[');
    deep += std::string(200, ']');
    CHECK_FALSE(json::parse(deep).ok());
}

TEST_CASE("non-finite doubles serialize as null") {
    CHECK(json::Value(std::numeric_limits<double>::infinity()).dump() == "null");
}
}
