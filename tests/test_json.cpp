#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

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

TEST_CASE("unsigned 64-bit values above INT64_MAX are exact, never negative") {
    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
    const json::Value v(kMax);
    CHECK(v.is_integer());
    CHECK(v.dump() == "18446744073709551615");
    CHECK(json::Value(std::uint64_t{9223372036854775808ull}).dump() == "9223372036854775808");
    CHECK(json::Value(std::optional<std::uint64_t>(kMax)).dump() == "18446744073709551615");
    // Values that fit int64 keep the signed representation.
    CHECK(json::Value(std::uint64_t{42}).dump() == "42");
    CHECK(json::Value(std::uint64_t{42}).as_int() == 42);
    // Not representable as int64: the fallback, not a wrapped value.
    CHECK(v.as_int(-7) == -7);
    CHECK(v.as_uint() == kMax);
    CHECK(v.as_double() == doctest::Approx(1.8446744073709552e19));
    // Parsing keeps it an exact integer, so it round-trips.
    auto parsed = json::parse("{\"seed\":18446744073709551615}");
    REQUIRE(parsed.ok());
    const json::Value* seed = parsed.value().find("seed");
    REQUIRE(seed != nullptr);
    CHECK(seed->is_integer());
    CHECK(seed->as_uint() == kMax);
    CHECK(parsed.value().dump() == "{\"seed\":18446744073709551615}");
    CHECK(json::Value(std::int64_t{-5}).as_uint(9) == 9);
}

TEST_CASE("floating JSON numbers convert throughout the uint64 range") {
    auto parsed = json::parse("1.82e19");
    REQUIRE(parsed.ok());
    CHECK(parsed.value().type() == json::Type::number);
    CHECK(parsed.value().as_uint(7) == 18200000000000000000ull);

    constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
    const double upper_exclusive = static_cast<double>(kMax);
    CHECK(json::Value(std::nextafter(upper_exclusive, 0.0)).as_uint(7) == kMax - 2047);
    CHECK(json::Value(upper_exclusive).as_uint(7) == 7);
    CHECK(json::Value(-1.0).as_uint(7) == 7);
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

// Regression tests for docs/integration/hardening.md B1, B4, B5 and the
// out-of-range number observation. Repro inputs also live in fuzz/corpus/json.

TEST_CASE("large objects parse in linear time (B1)") {
    // 200k distinct keys, ~2.2 MiB. The old Object::set-per-member parser
    // needed ~45 s for this at -O2; linear parsing takes well
    // under a second even under sanitizers.
    constexpr int kKeys = 200000;
    std::string text = "{";
    for (int i = 0; i < kKeys; ++i) {
        if (i) text += ',';
        text += "\"k" + std::to_string(i) + "\":" + std::to_string(i);
    }
    text += '}';
    const auto t0 = std::chrono::steady_clock::now();
    auto parsed = json::parse(text);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE(parsed.ok());
    CHECK(parsed.value().as_object().size() == static_cast<std::size_t>(kKeys));
    CHECK(parsed.value().find("k0")->as_int() == 0);
    CHECK(parsed.value().find("k199999")->as_int() == 199999);
    CHECK(seconds < 5.0);
}

TEST_CASE("duplicate keys: first position, last value, small and large objects (B1)") {
    auto small = json::parse(R"({"a":1,"b":2,"a":3})");
    REQUIRE(small.ok());
    CHECK(small.value().dump() == R"({"a":3,"b":2})");

    std::string text = "{\"dup\":0";
    for (int i = 0; i < 40; ++i) text += ",\"k" + std::to_string(i) + "\":" + std::to_string(i);
    text += ",\"dup\":99,\"k5\":-5}";
    auto large = json::parse(text);
    REQUIRE(large.ok());
    const auto& obj = large.value().as_object();
    CHECK(obj.size() == 41u);
    CHECK(obj.begin()->first == "dup");
    CHECK(obj.begin()->second.as_int() == 99);
    CHECK(large.value().find("k5")->as_int() == -5);

    std::vector<json::Member> members;
    for (int i = 0; i < 20; ++i) members.emplace_back("m" + std::to_string(i % 10), i);
    json::Object built = json::Object::from_members(std::move(members));
    CHECK(built.size() == 10u);
    CHECK(built.find("m3")->as_int() == 13);
}

TEST_CASE("unpaired high surrogate does not swallow the next escape (B4)") {
    auto v = json::parse(R"("\uD800\u0041")");
    REQUIRE(v.ok());
    CHECK(v.value().as_string() == "\xEF\xBF\xBD" "A");
    // A second high surrogate that does pair up is still decoded.
    auto w = json::parse(R"("\uD800\uD83D\uDE00")");
    REQUIRE(w.ok());
    CHECK(w.value().as_string() == "\xEF\xBF\xBD\xF0\x9F\x98\x80");
    auto lone_low = json::parse(R"("x\uDC00y")");
    REQUIRE(lone_low.ok());
    CHECK(lone_low.value().as_string() == "x\xEF\xBF\xBDy");
    auto trailing = json::parse(R"("\uD800")");
    REQUIRE(trailing.ok());
    CHECK(trailing.value().as_string() == "\xEF\xBF\xBD");
}

TEST_CASE("serialized strings are always valid UTF-8 (B5)") {
    // Token pieces can split a multi-byte character (first 2 bytes of U+20AC).
    CHECK(json::Value(std::string("\xE2\x82")).dump() == "\"\xEF\xBF\xBD\xEF\xBF\xBD\"");
    CHECK(json::Value(std::string("a\xFFz")).dump() == "\"a\xEF\xBF\xBDz\"");
    // Overlong, surrogate and > U+10FFFF encodings are ill-formed too.
    CHECK(json::Value(std::string("\xC0\xAF")).dump() == "\"\xEF\xBF\xBD\xEF\xBF\xBD\"");
    CHECK(json::Value(std::string("\xED\xA0\x80")).dump() == "\"\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\"");
    CHECK(json::Value(std::string("\xF4\x90\x80\x80")).dump() ==
          "\"\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\xEF\xBF\xBD\"");
    // Well-formed text is untouched, including 4-byte sequences.
    const std::string ok = "\xE2\x82\xAC \xC3\xA9 \xF0\x9F\x98\x80";
    CHECK(json::Value(ok).dump() == "\"" + ok + "\"");
    // Keys are sanitized as well; the result re-parses and is a fixed point.
    json::Object o{{std::string("k\x80"), std::string("v\xE2\x82")}};
    const std::string once = json::Value(o).dump();
    auto back = json::parse(once);
    REQUIRE(back.ok());
    CHECK(back.value().dump() == once);
}

TEST_CASE("raw ill-formed UTF-8 in input parses to U+FFFD (B5)") {
    auto v = json::parse(std::string("\"a\xE2\x82z\""));
    REQUIRE(v.ok());
    CHECK(v.value().as_string() == "a\xEF\xBF\xBD\xEF\xBF\xBDz");
    // Two keys that differ only in invalid bytes become the same key at parse
    // time, so dump() of the result is a fixed point (found by sonder_fuzz_json).
    auto o = json::parse(std::string("{\"k\xD1\":1,\"k\xA7\":2}"));
    REQUIRE(o.ok());
    CHECK(o.value().as_object().size() == 1u);
    const std::string once = o.value().dump();
    CHECK(json::parse(once).value().dump() == once);
    const std::string ok = "\"\xE2\x82\xAC\xF0\x9F\x98\x80\"";
    CHECK(json::parse(ok).value().dump() == ok);
}

TEST_CASE("numbers outside double range are rejected") {
    CHECK_FALSE(json::parse("1e999").ok());
    CHECK_FALSE(json::parse("[-1e400]").ok());
    CHECK(json::parse("1e308").ok());
    auto tiny = json::parse("1e-400");  // underflow to 0 is fine
    REQUIRE(tiny.ok());
    CHECK(tiny.value().as_double() == 0.0);
    CHECK(json::parse("18446744073709551616").ok());  // > int64 falls back to double
}
}
