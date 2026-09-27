#include <doctest/doctest.h>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
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

TEST_CASE("extended sampling fields through the C ABI") {
    sonder_sampling_config cfg;
    std::memset(&cfg, 0xAB, sizeof(cfg));  // init must overwrite every field
    sonder_sampling_config_init(&cfg);
    CHECK(cfg.struct_size == sizeof(sonder_sampling_config));
    CHECK(cfg.typical_p == 1.0f);
    CHECK(cfg.presence_penalty == 0.0f);
    CHECK(cfg.frequency_penalty == 0.0f);
    CHECK(cfg.repeat_last_n == 64);
    CHECK(cfg.num_ctx == 0);
    CHECK(cfg.logit_bias == nullptr);
    CHECK(cfg.logit_bias_count == 0u);
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_OK);

    const sonder_logit_bias biases[] = {{5, 2.5f}, {9, -std::numeric_limits<float>::infinity()}};
    cfg.logit_bias = biases;
    cfg.logit_bias_count = 2;
    cfg.typical_p = 0.9f;
    cfg.presence_penalty = 0.5f;
    cfg.frequency_penalty = 0.25f;
    cfg.num_ctx = 4096;
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_OK);

    cfg.typical_p = 0.0f;
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(std::string(sonder_last_error_message()).find("typical_p") != std::string::npos);
    cfg.typical_p = 1.0f;
    cfg.num_ctx = -1;
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(std::string(sonder_last_error_message()).find("num_ctx") != std::string::npos);
    cfg.num_ctx = 0;
    cfg.logit_bias = nullptr;  // count > 0 with a null pointer
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(std::string(sonder_last_error_message()).find("logit_bias") != std::string::npos);
    const sonder_logit_bias dup[] = {{1, 1.0f}, {1, 2.0f}};
    cfg.logit_bias = dup;
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("callers built against the original sampling layout still work") {
    // Mirror of the first sonder_sampling_config layout (fields up to max_tokens).
    struct LegacySamplingConfig {
        uint32_t struct_size;
        float temperature;
        float top_p;
        int32_t top_k;
        float min_p;
        float repeat_penalty;
        int32_t has_seed;
        uint64_t seed;
        int32_t max_tokens;
    };
    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    // Garbage past the legacy size must be ignored when struct_size is legacy.
    cfg.typical_p = -7.0f;
    cfg.num_ctx = -7;
    cfg.logit_bias = nullptr;
    cfg.logit_bias_count = 99;
    cfg.struct_size = sizeof(LegacySamplingConfig);
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_OK);
    cfg.struct_size = sizeof(LegacySamplingConfig) - 1;
    CHECK(sonder_sampling_config_validate(&cfg) == SONDER_ERROR_INVALID_ARGUMENT);
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

    // Extended fields are accepted by session creation, and invalid ones rejected.
    sonder_model* model2 = nullptr;
    REQUIRE(sonder_model_load(engine, "mock", "mock:tiny", &model2) == SONDER_OK);
    const sonder_logit_bias biases[] = {{3, 1.0f}};
    cfg.logit_bias = biases;
    cfg.logit_bias_count = 1;
    cfg.presence_penalty = 0.5f;
    cfg.num_ctx = 2048;
    sonder_session* session2 = nullptr;
    REQUIRE(sonder_session_create(engine, model2, &cfg, &session2) == SONDER_OK);
    sonder_session_destroy(session2);
    cfg.frequency_penalty = 3.0f;
    session2 = nullptr;
    CHECK(sonder_session_create(engine, model2, &cfg, &session2) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(session2 == nullptr);
    sonder_model_release(model2);
    sonder_engine_destroy(engine);
}

TEST_CASE("destroying the engine before its sessions is safe (sessions keep the engine alive)") {
    const std::string path = "c_abi_engine_first.jsonl";
    std::remove(path.c_str());
    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_METRICS;
    eo.telemetry_jsonl_path = path.c_str();
    sonder_engine* engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &engine) == SONDER_OK);
    REQUIRE(sonder_engine_register_mock_backend(engine) == SONDER_OK);
    sonder_model* model = nullptr;
    REQUIRE(sonder_model_load(engine, "mock", "mock:tiny", &model) == SONDER_OK);
    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    cfg.temperature = 0.0f;
    cfg.max_tokens = 4;
    sonder_session* session = nullptr;
    REQUIRE(sonder_session_create(engine, model, &cfg, &session) == SONDER_OK);
    sonder_model_release(model);

    // The order the header never forbade: engine first, then its session.
    sonder_engine_destroy(engine);
    Count count;
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    CHECK(sonder_session_generate(session, "still usable", on_token, &count, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    CHECK(count.chunks == 4);
    sonder_session_destroy(session);

    // The session closed on a live engine, and the engine stopped after it.
    std::ifstream in(path);
    std::stringstream buf;
    buf << in.rdbuf();
    const std::string log = buf.str();
    const auto closed = log.find("\"event_type\":\"session.closed\"");
    const auto stopped = log.find("\"event_type\":\"engine.stopped\"");
    CHECK(closed != std::string::npos);
    CHECK(stopped != std::string::npos);
    CHECK(closed < stopped);
    in.close();
    std::remove(path.c_str());
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
