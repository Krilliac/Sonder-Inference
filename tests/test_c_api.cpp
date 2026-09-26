#include <doctest/doctest.h>

#include <string>

#include "sonder_inference.h"

TEST_SUITE("c_abi") {
TEST_CASE("version and status strings") {
    CHECK(sonder_abi_version() == SONDER_ABI_VERSION);
    CHECK(std::string(sonder_version_string()).size() > 0);
    CHECK(std::string(sonder_status_string(SONDER_ERROR_CANCELLED)) == "cancelled");
}

TEST_CASE("sampling validation through the C ABI") {
    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_OK);
    cfg.top_p = 0.0f;
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(std::string(sonder_last_error_message()).find("top_p") != std::string::npos);
    cfg.struct_size = 4;
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(sonder_sampling_config_validate(nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
}

namespace {
struct Count {
    int chunks = 0;
    std::string text;
    int stop_at = -1;
};
int on_token(void* user, const char* text, size_t len) {
    auto* c = static_cast<Count*>(user);
    c->text.append(text, len);
    ++c->chunks;
    return (c->stop_at > 0 && c->chunks >= c->stop_at) ? 1 : 0;
}
}  // namespace

TEST_CASE("mock generation end to end") {
    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_OFF;
    sonder_engine* engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &engine) == SONDER_OK);
    CHECK(sonder_engine_device_count(engine) >= 1);
    REQUIRE(sonder_engine_register_mock_backend(engine) == SONDER_OK);
    CHECK(sonder_engine_register_mock_backend(engine) == SONDER_ERROR_INVALID_STATE);

    sonder_model* missing = nullptr;
    CHECK(sonder_model_load(engine, "mock", "not-a-mock", &missing) == SONDER_ERROR_NOT_FOUND);
    CHECK(missing == nullptr);

    sonder_model* model = nullptr;
    REQUIRE(sonder_model_load(engine, "mock", "mock:tiny", &model) == SONDER_OK);
    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    cfg.temperature = 0.0f;
    cfg.max_tokens = 5;
    sonder_session* session = nullptr;
    REQUIRE(sonder_session_create(engine, model, &cfg, &session) == SONDER_OK);
    sonder_model_release(model);  // session keeps the model alive

    Count count;
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE(sonder_session_generate(session, "hello from C", on_token, &count, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    CHECK(stats.completion_tokens == 5);
    CHECK(stats.prompt_tokens == 3);
    CHECK(count.chunks == 5);
    CHECK_FALSE(count.text.empty());

    Count early;
    early.stop_at = 2;
    REQUIRE(sonder_session_generate(session, "stop early", on_token, &early, &stats) == SONDER_OK);
    CHECK(early.chunks == 2);

    CHECK(sonder_session_cancel(session) == SONDER_OK);  // idle cancel is a no-op
    sonder_session_destroy(session);
    sonder_engine_destroy(engine);
}

TEST_CASE("null arguments are rejected") {
    CHECK(sonder_engine_create(nullptr, nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(sonder_session_cancel(nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(sonder_model_load(nullptr, "mock", "mock", nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    sonder_engine_destroy(nullptr);
    sonder_session_destroy(nullptr);
    sonder_model_release(nullptr);
}
}
