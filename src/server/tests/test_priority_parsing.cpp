#include <doctest/doctest.h>

#include <string>
#include <map>
#include <limits>

#include "src/openai.hpp"
#include "src/http.hpp"
#include "src/request_path.hpp"

namespace det = sonder::inference::server::detail;
namespace si = sonder::inference;

TEST_CASE("priority and deadline body hints are parsed strictly") {
    const auto body = R"({"messages":[{"role":"user","content":"hi"}],"priority":"background","deadline_ms":1000})";
    auto parsed = det::parse_chat_request(body);
    REQUIRE(std::holds_alternative<det::ChatJob>(parsed));
    const auto& job = std::get<det::ChatJob>(parsed);
    REQUIRE(job.priority_class);
    CHECK(*job.priority_class == si::RequestPriority::background);
    CHECK(job.deadline_ms == std::optional<std::uint64_t>(1000));

    for (const auto value : {"urgent", "1", ""}) {
        auto result = det::parse_chat_request(std::string(R"({"messages":[{"role":"user","content":"hi"}],"priority":")") + value + "\"}");
        REQUIRE(std::holds_alternative<det::ApiError>(result));
        CHECK(std::get<det::ApiError>(result).status == 400);
    }
    for (const auto value : {"0", "-1", "1.5", "\"100\"", "null", "true", "2147483648", "18446744073709551616"}) {
        auto result = det::parse_chat_request(std::string(R"({"messages":[{"role":"user","content":"hi"}],"deadline_ms":)") + value + "}");
        REQUIRE(std::holds_alternative<det::ApiError>(result));
        CHECK(std::get<det::ApiError>(result).status == 400);
    }
}

TEST_CASE("priority hints reject explicit null and default unset stays chat-only") {
    const auto parse = det::parse_chat_request;
    const std::string prefix = R"({"messages":[{"role":"user","content":"hi"}])";
    auto absent = parse(prefix + "}");
    REQUIRE(std::holds_alternative<det::ChatJob>(absent));
    CHECK_FALSE(std::get<det::ChatJob>(absent).priority_class);
    CHECK_FALSE(std::get<det::ChatJob>(absent).deadline_ms);
    for (const auto* hint : {R"(,"priority":null})", R"(,"priority":1})", R"(,"priority":true})",
                             R"(,"priority":"Interactive"})", R"(,"deadline_ms":null})"}) {
        CHECK(std::holds_alternative<det::ApiError>(parse(prefix + hint)));
    }
    auto max = parse(prefix + R"(,"deadline_ms":2147483647})");
    REQUIRE(std::holds_alternative<det::ChatJob>(max));
    CHECK(std::get<det::ChatJob>(max).deadline_ms == 2147483647u);
}

TEST_CASE("priority serve flags are unlimited by default and accept bounded unsigned counts") {
    const auto parse = [](const std::map<std::string, std::string>& values,
                          si::server::ServerOptions& options) {
        return det::parse_request_path_flags([&](const std::string& key) -> std::optional<std::string> {
            const auto it = values.find(key);
            return it == values.end() ? std::nullopt : std::optional<std::string>(it->second);
        }, options);
    };
    si::server::ServerOptions options;
    REQUIRE(parse({}, options).ok());
    CHECK(options.max_concurrent_subagent == 0);
    CHECK(options.max_concurrent_background == 0);
    CHECK(options.max_queue_per_class == 0);
    CHECK(options.backend_capacity == 0);
    CHECK(options.priority_admission == si::server::PriorityAdmissionPolicy::automatic);
    CHECK_FALSE(det::priority_admission_enabled(options));
    REQUIRE(parse({{"max-concurrent-subagent", "3"}, {"max-concurrent-background", "1"},
                   {"max-queue-per-class", "12"}}, options).ok());
    CHECK(options.max_concurrent_subagent == 3);
    CHECK(options.max_concurrent_background == 1);
    CHECK(options.max_queue_per_class == 12);
    for (const auto* flag : {"max-concurrent-subagent", "max-concurrent-background", "max-queue-per-class",
                            "backend-capacity"}) {
        for (const auto* value : {"", "-1", "+1", "1.5", "no", "18446744073709551616"}) {
            CHECK_FALSE(parse({{flag, value}}, options).ok());
        }
        CHECK(parse({{flag, "0"}}, options).ok());
    }
    REQUIRE(parse({{"backend-capacity", "2"}}, options).ok());
    CHECK(options.backend_capacity == 2);
    for (const auto* mode : {"auto", "on", "off"}) {
        REQUIRE(parse({{"priority-admission", mode}}, options).ok());
        const auto expected = std::string(mode) == "auto" ? si::server::PriorityAdmissionPolicy::automatic
                            : std::string(mode) == "on" ? si::server::PriorityAdmissionPolicy::on
                                                         : si::server::PriorityAdmissionPolicy::off;
        CHECK(options.priority_admission == expected);
    }
    for (const auto* invalid : {"", "automatic", "ON", "true", "1"}) {
        CHECK_FALSE(parse({{"priority-admission", invalid}}, options).ok());
    }
}

TEST_CASE("priority admission auto is opt-in through class or queue caps only") {
    using Policy = si::server::PriorityAdmissionPolicy;
    for (const auto mode : {Policy::automatic, Policy::on, Policy::off}) {
        for (const std::size_t capacity : {std::size_t{0}, std::size_t{2}}) {
            for (int caps = 0; caps < 8; ++caps) {
                si::server::ServerOptions options;
                options.priority_admission = mode;
                options.backend_capacity = capacity;
                options.max_concurrent_subagent = (caps & 1) != 0 ? 1u : 0u;
                options.max_concurrent_background = (caps & 2) != 0 ? 1u : 0u;
                options.max_queue_per_class = (caps & 4) != 0 ? 1u : 0u;
                const bool expected = mode == Policy::on || (mode == Policy::automatic && caps != 0);
                CHECK(det::priority_admission_enabled(options) == expected);
            }
        }
    }
}

TEST_CASE("priority and deadline headers are validated independently") {
    det::RequestHead head;
    head.headers = {{"x-sonder-priority", "interactive"}, {"x-sonder-deadline-ms", "25"}};
    auto parsed = det::parse_correlation(head);
    REQUIRE(std::holds_alternative<det::Correlation>(parsed));
    const auto& c = std::get<det::Correlation>(parsed);
    CHECK(c.priority_class == std::optional<si::RequestPriority>(si::RequestPriority::interactive));
    CHECK_FALSE(c.numeric_priority);
    CHECK(c.deadline_ms == std::optional<std::uint64_t>(25));

    head.headers[0].second = "7";
    auto numeric = det::parse_correlation(head);
    REQUIRE(std::holds_alternative<det::Correlation>(numeric));
    CHECK(std::get<det::Correlation>(numeric).priority == 7);
    CHECK(std::get<det::Correlation>(numeric).numeric_priority);
    CHECK(std::get<det::Correlation>(numeric).priority_class == si::RequestPriority::interactive);
    for (const auto* value : {"-16", "0", "+16"}) {
        head.headers[0].second = value;
        auto legacy = det::parse_correlation(head);
        REQUIRE(std::holds_alternative<det::Correlation>(legacy));
        CHECK(std::get<det::Correlation>(legacy).numeric_priority);
        CHECK(std::get<det::Correlation>(legacy).priority_class == si::RequestPriority::interactive);
    }
    for (const auto* value : {"17", "-17", "99", "1.5", ""}) {
        head.headers[0].second = value;
        auto invalid = det::parse_correlation(head);
        REQUIRE(std::holds_alternative<det::ApiError>(invalid));
        CHECK(std::get<det::ApiError>(invalid).status == 400);
        CHECK(std::get<det::ApiError>(invalid).code == "invalid_correlation_header");
    }
    head.headers[0].second = "interactive";
    head.headers[1].second = "0";
    CHECK(std::get<det::ApiError>(det::parse_correlation(head)).status == 400);
    head.headers = {{"x-sonder-priority", "interactive"}, {"x-sonder-priority", "background"}};
    CHECK(std::get<det::ApiError>(det::parse_correlation(head)).status == 400);
    head.headers = {{"x-sonder-deadline-ms", "10"}, {"x-sonder-deadline-ms", "20"}};
    CHECK(std::get<det::ApiError>(det::parse_correlation(head)).status == 400);
}
