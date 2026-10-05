#include <doctest/doctest.h>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "sonder/inference/json.hpp"
#include "sonder/inference/telemetry.hpp"
#include "sonder_inference.h"

namespace {
namespace si = sonder::inference;
struct ParentFixture {
    std::filesystem::path path = std::filesystem::temp_directory_path() / (si::make_id("parent-test") + ".jsonl");
    std::unique_ptr<sonder_engine, decltype(&sonder_engine_destroy)> engine{nullptr, sonder_engine_destroy};
    std::unique_ptr<sonder_model, decltype(&sonder_model_release)> model{nullptr, sonder_model_release};
    std::unique_ptr<sonder_session, decltype(&sonder_session_destroy)> session{nullptr, sonder_session_destroy};
    ParentFixture() {
        const auto filename = path.string();
        sonder_engine_options eo;
        sonder_engine_options_init(&eo);
        eo.telemetry_level = SONDER_TELEMETRY_DEEP;
        eo.telemetry_jsonl_path = filename.c_str();
        sonder_engine* e = nullptr;
        REQUIRE(sonder_engine_create(&eo, &e) == SONDER_OK);
        engine.reset(e);
        REQUIRE(sonder_engine_register_mock_backend(e) == SONDER_OK);
        sonder_model* m = nullptr;
        REQUIRE(sonder_model_load(e, "mock", "mock:tiny", &m) == SONDER_OK);
        model.reset(m);
        sonder_sampling_config sampling;
        sonder_sampling_config_init(&sampling);
        sampling.max_tokens = 5;
        const sonder_session_metadata metadata{sizeof(sonder_session_metadata), "parent-session:1",
                                              "parent-run:1", "parent-agent:1", "parent-task:1"};
        sonder_session* s = nullptr;
        REQUIRE(sonder_session_create_with_metadata(e, m, &sampling, &metadata, &s) == SONDER_OK);
        session.reset(s);
    }
    std::vector<si::json::Value> events() {
        session.reset();
        model.reset();
        engine.reset();  // drain the actual telemetry writer
        std::ifstream input(path);
        std::vector<si::json::Value> events;
        for (std::string line; std::getline(input, line);) {
            CHECK(line.find("parent-private-prompt-canary") == std::string::npos);
            auto parsed = si::json::parse(line);
            REQUIRE(parsed.ok());
            events.push_back(std::move(parsed).value());
        }
        return events;
    }
    ~ParentFixture() {
        session.reset();
        model.reset();
        engine.reset();
        std::error_code ignored;
        std::filesystem::remove(path, ignored);
    }
};
int count_token(void* user, const char*, size_t) {
    ++*static_cast<int*>(user);
    return 0;
}
int cancel_token(void* user, const char*, size_t) {
    CHECK(sonder_session_cancel(static_cast<sonder_session*>(user)) == SONDER_OK);
    return 0;
}
}  // namespace

TEST_SUITE("c_abi_request_metadata") {
TEST_CASE("parent lineage is request local across generate chat cancellation and defaults") {
    ParentFixture f;
    char parent[] = "parent-turn:generate";
    sonder_request_metadata metadata{sizeof(sonder_request_metadata), parent};
    REQUIRE(sonder_session_generate_with_metadata(f.session.get(), "parent-private-prompt-canary", &metadata,
                                                 nullptr, nullptr, nullptr) == SONDER_OK);
    parent[0] = 'X';  // after return: queued telemetry owns its copy
    metadata.parent_request_id = "parent-turn:chat";
    const sonder_chat_message message{sizeof(sonder_chat_message), "user", "parent-private-prompt-canary"};
    const sonder_chat_message* messages[] = {&message};
    REQUIRE(sonder_session_chat_with_metadata(f.session.get(), messages, 1, &metadata,
                                             nullptr, nullptr, nullptr) == SONDER_OK);
    metadata.parent_request_id = "parent-turn:cancel";
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE(sonder_session_generate_with_metadata(f.session.get(), "parent-private-prompt-canary", &metadata,
                                                 cancel_token, f.session.get(), &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_CANCELLED);
    REQUIRE(sonder_session_generate(f.session.get(), "parent-private-prompt-canary", nullptr, nullptr, nullptr) == SONDER_OK);
    REQUIRE(sonder_session_chat(f.session.get(), messages, 1, nullptr, nullptr, nullptr) == SONDER_OK);
    metadata.parent_request_id = nullptr;
    REQUIRE(sonder_session_generate_with_metadata(f.session.get(), "parent-private-prompt-canary", &metadata,
                                                 nullptr, nullptr, nullptr) == SONDER_OK);
    REQUIRE(sonder_session_chat_with_metadata(f.session.get(), messages, 1, nullptr,
                                             nullptr, nullptr, nullptr) == SONDER_OK);
    const std::vector<std::string> parents{"parent-turn:generate", "parent-turn:chat", "parent-turn:cancel", "", "", "", ""};
    std::vector<std::string> ids;
    unsigned completed = 0, cancelled = 0;
    for (const auto& event : f.events()) {
        if (event.find("session_id")->as_string() != "parent-session:1") {
            continue;
        }
        CHECK(event.find("run_id")->as_string() == "parent-run:1");
        CHECK(event.find("agent_id")->as_string() == "parent-agent:1");
        CHECK(event.find("task_id")->as_string() == "parent-task:1");
        const auto& type = event.find("event_type")->as_string();
        const auto* attrs = event.find("attributes");
        if (type == "request.queued") {
            ids.push_back(event.find("request_id")->as_string());
        }
        if (type.rfind("request.", 0) == 0) {
            const auto id = event.find("request_id")->as_string();
            std::size_t index = 0;
            while (index < ids.size() && ids[index] != id) {
                ++index;
            }
            REQUIRE(index < ids.size());
            REQUIRE(index < parents.size());
            const auto* actual = attrs->find("parent_request_id");
            if (parents[index].empty()) {
                CHECK(actual == nullptr);
            } else {
                REQUIRE(actual != nullptr);
                CHECK(actual->as_string() == parents[index]);
            }
        } else {
            CHECK(attrs->find("parent_request_id") == nullptr);
        }
        completed += type == "request.completed";
        cancelled += type == "request.cancelled";
    }
    CHECK(ids.size() == 7);
    CHECK(completed == 6);
    CHECK(cancelled == 1);
    for (std::size_t i = 0; i < ids.size(); ++i) {
        CHECK(ids[i].rfind("req-", 0) == 0);
        for (std::size_t j = i + 1; j < ids.size(); ++j) {
            CHECK(ids[i] != ids[j]);
        }
    }
}

TEST_CASE("parent metadata rejects incomplete prefixes and invalid IDs before submission") {
    ParentFixture f;
    sonder_request_metadata metadata{offsetof(sonder_request_metadata, parent_request_id), nullptr};
    int callbacks = 0;
    const sonder_chat_message message{sizeof(sonder_chat_message), "user", "hi"};
    const sonder_chat_message* messages[] = {&message};
    auto generate = [&] { return sonder_session_generate_with_metadata(f.session.get(), "hi", &metadata,
                                                                      count_token, &callbacks, nullptr); };
    auto chat = [&] { return sonder_session_chat_with_metadata(f.session.get(), messages, 1, &metadata,
                                                              count_token, &callbacks, nullptr); };
    CHECK(generate() == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(chat() == SONDER_ERROR_INVALID_ARGUMENT);
    metadata.struct_size = sizeof(metadata);
    const std::vector<std::string> invalid{"", std::string(129, 'a'), "private/id", "private\nvalue", "h\xC3\xA9llo"};
    for (const auto& value : invalid) {
        metadata.parent_request_id = value.c_str();
        for (bool use_chat : {false, true}) {
            CHECK((use_chat ? chat() : generate()) == SONDER_ERROR_INVALID_ARGUMENT);
            if (!value.empty()) {
                CHECK(std::string(sonder_last_error_message()).find(value) == std::string::npos);
            }
        }
    }
    CHECK(callbacks == 0);
    metadata.parent_request_id = nullptr;
    REQUIRE(generate() == SONDER_OK);
    CHECK(callbacks == 5);
    unsigned queued = 0;
    for (const auto& event : f.events()) {
        queued += event.find("event_type")->as_string() == "request.queued";
    }
    CHECK(queued == 1);
}

TEST_CASE("parent metadata accepts exact identifier boundary and extended records") {
    ParentFixture f;
    const std::string longest(SONDER_MAX_CORRELATION_ID_BYTES, 'a');
    struct Extended {
        sonder_request_metadata prefix;
        const char* ignored;
    } metadata{{sizeof(Extended), longest.c_str()}, "ignored/private/value"};
    REQUIRE(sonder_session_generate_with_metadata(f.session.get(), "hi", &metadata.prefix,
                                                 nullptr, nullptr, nullptr) == SONDER_OK);
    const sonder_chat_message message{sizeof(sonder_chat_message), "user", "hi"};
    const sonder_chat_message* messages[] = {&message};
    REQUIRE(sonder_session_chat_with_metadata(f.session.get(), messages, 1, &metadata.prefix,
                                             nullptr, nullptr, nullptr) == SONDER_OK);
    unsigned lifecycle = 0;
    for (const auto& event : f.events()) {
        if (event.find("event_type")->as_string().rfind("request.", 0) == 0) {
            REQUIRE(event.find("attributes")->find("parent_request_id") != nullptr);
            CHECK(event.find("attributes")->find("parent_request_id")->as_string() == longest);
            ++lifecycle;
        }
    }
    CHECK(lifecycle == 6);
}
}
