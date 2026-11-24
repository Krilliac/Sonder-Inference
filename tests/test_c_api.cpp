#include <doctest/doctest.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

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

TEST_CASE("init on an original-layout allocation never writes past it") {
    // A caller built against the original header allocates only the legacy
    // layout and calls the exported initializer (not the header macro).
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
    constexpr std::size_t kLegacy = sizeof(LegacySamplingConfig);
    static_assert(kLegacy < sizeof(sonder_sampling_config));
    alignas(sonder_sampling_config) unsigned char buf[sizeof(sonder_sampling_config) + 16];
    std::memset(buf, 0xA5, sizeof(buf));
    (sonder_sampling_config_init)(reinterpret_cast<sonder_sampling_config*>(buf));
    for (std::size_t i = kLegacy; i < sizeof(buf); ++i) {
        CAPTURE(i);
        REQUIRE(buf[i] == 0xA5);  // adjacent memory untouched
    }
    LegacySamplingConfig legacy;
    std::memcpy(&legacy, buf, kLegacy);
    CHECK(legacy.struct_size == kLegacy);
    CHECK(legacy.max_tokens > 0);
    CHECK(legacy.has_seed == 0);
    // The legacy-sized config is usable as is.
    CHECK(sonder_sampling_config_validate(reinterpret_cast<const sonder_sampling_config*>(buf)) == SONDER_OK);

    // The sized initializer writes exactly what the caller owns.
    std::memset(buf, 0xA5, sizeof(buf));
    sonder_sampling_config_init_sized(reinterpret_cast<sonder_sampling_config*>(buf), kLegacy);
    for (std::size_t i = kLegacy; i < sizeof(buf); ++i) {
        CAPTURE(i);
        REQUIRE(buf[i] == 0xA5);
    }
    std::memset(buf, 0xA5, sizeof(buf));
    sonder_sampling_config_init_sized(reinterpret_cast<sonder_sampling_config*>(buf), kLegacy - 1);
    CHECK(buf[0] == 0xA5);  // too small to hold even the original layout: untouched
    // Through the header (current callers) the whole current struct is set.
    sonder_sampling_config cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    sonder_sampling_config_init(&cfg);
    CHECK(cfg.struct_size == sizeof(sonder_sampling_config));
    CHECK(cfg.typical_p == doctest::Approx(1.0f));
    CHECK(cfg.repeat_last_n == 64);
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

TEST_CASE("C ABI session metadata copies correlation ids and retains handles") {
    const std::string path = "c_abi_session_metadata.jsonl";
    std::remove(path.c_str());
    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_DEEP;
    eo.capture_text = 0;
    eo.telemetry_jsonl_path = path.c_str();
    sonder_engine* engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &engine) == SONDER_OK);
    REQUIRE(sonder_engine_register_mock_backend(engine) == SONDER_OK);
    sonder_model* model = nullptr;
    REQUIRE(sonder_model_load(engine, "mock", "mock:tiny", &model) == SONDER_OK);
    char sid[] = "sdk-session:copied";
    char rid[] = "sdk-run:copied";
    char aid[] = "sdk-agent:copied";
    char tid[] = "sdk-task:copied";
    struct Extended {
        sonder_session_metadata metadata;
        uint64_t future_field;
    } options{{sizeof(Extended), sid, rid, aid, tid}, 42};
    sonder_sampling_config sampling;
    sonder_sampling_config_init(&sampling);
    sampling.max_tokens = 4;
    sonder_session* session = nullptr;
    REQUIRE(sonder_session_create_with_metadata(engine, model, &sampling, &options.metadata, &session) == SONDER_OK);
    sid[0] = rid[0] = aid[0] = tid[0] = 'X';  // copied, not borrowed
    sonder_model_release(model);
    sonder_engine_destroy(engine);  // metadata sessions preserve the old lifetime contract
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE(sonder_session_generate(session, "private-metadata-canary", nullptr, nullptr, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    const sonder_chat_message message{sizeof(sonder_chat_message), "user", "private-metadata-canary"};
    const sonder_chat_message* messages[] = {&message};
    REQUIRE(sonder_session_chat(session, messages, 1, nullptr, nullptr, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    sonder_session_destroy(session);
    std::ifstream input(path);
    std::string line;
    unsigned session_events = 0;
    while (std::getline(input, line)) {
        CHECK(line.find("private-metadata-canary") == std::string::npos);
        if (line.find("\"session_id\":\"sdk-session:copied\"") != std::string::npos) {
            ++session_events;
            CHECK(line.find("\"run_id\":\"sdk-run:copied\"") != std::string::npos);
            CHECK(line.find("\"agent_id\":\"sdk-agent:copied\"") != std::string::npos);
            CHECK(line.find("\"task_id\":\"sdk-task:copied\"") != std::string::npos);
        }
    }
    CHECK(session_events > 10);
    input.close();
    std::remove(path.c_str());
}

TEST_CASE("C ABI session metadata validates size and bounded identifiers without echo") {
    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_OFF;
    sonder_engine* engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &engine) == SONDER_OK);
    REQUIRE(sonder_engine_register_mock_backend(engine) == SONDER_OK);
    sonder_model* model = nullptr;
    REQUIRE(sonder_model_load(engine, "mock", "mock:tiny", &model) == SONDER_OK);
    sonder_session_metadata metadata{sizeof(sonder_session_metadata), nullptr, nullptr, nullptr, nullptr};
    sonder_session* session = nullptr;
    CHECK(sonder_session_create_with_metadata(nullptr, model, nullptr, &metadata, &session) == SONDER_ERROR_INVALID_ARGUMENT);
    metadata.struct_size = offsetof(sonder_session_metadata, task_id);
    CHECK(sonder_session_create_with_metadata(engine, model, nullptr, &metadata, &session) == SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(session == nullptr);
    metadata.struct_size = sizeof(metadata);
    const std::string invalid[] = {"", std::string(129, 'a'), "private/id", "line\nbreak", "h\xC3\xA9llo"};
    for (const auto& value : invalid) {
        metadata.run_id = value.c_str();
        CHECK(sonder_session_create_with_metadata(engine, model, nullptr, &metadata, &session) == SONDER_ERROR_INVALID_ARGUMENT);
        CHECK(session == nullptr);
        if (!value.empty()) {
            CHECK(std::string(sonder_last_error_message()).find(value) == std::string::npos);
        }
    }
    const std::string longest(128, 'a');
    metadata.run_id = longest.c_str();
    REQUIRE(sonder_session_create_with_metadata(engine, model, nullptr, &metadata, &session) == SONDER_OK);
    sonder_session_destroy(session);
    metadata.run_id = nullptr;
    REQUIRE(sonder_session_create_with_metadata(engine, model, nullptr, &metadata, &session) == SONDER_OK);
    sonder_session_destroy(session);
    REQUIRE(sonder_session_create_with_metadata(engine, model, nullptr, nullptr, &session) == SONDER_OK);
    sonder_session_destroy(session);
    sonder_model_release(model);
    sonder_engine_destroy(engine);
}

TEST_CASE("C ABI contains a C++ token callback exception and permits session reuse") {
    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_OFF;
    sonder_engine* raw_engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &raw_engine) == SONDER_OK);
    const std::unique_ptr<sonder_engine, decltype(&sonder_engine_destroy)> engine(raw_engine, &sonder_engine_destroy);
    REQUIRE(sonder_engine_register_mock_backend(engine.get()) == SONDER_OK);
    sonder_model* raw_model = nullptr;
    REQUIRE(sonder_model_load(engine.get(), "mock", "mock:tiny", &raw_model) == SONDER_OK);
    const std::unique_ptr<sonder_model, decltype(&sonder_model_release)> model(raw_model, &sonder_model_release);
    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    cfg.temperature = 0.0f;
    cfg.max_tokens = 4;
    sonder_session* raw_session = nullptr;
    REQUIRE(sonder_session_create(engine.get(), model.get(), &cfg, &raw_session) == SONDER_OK);
    const std::unique_ptr<sonder_session, decltype(&sonder_session_destroy)> session(raw_session, &sonder_session_destroy);
    int calls = 0;
    const auto throwing = [](void* user, const char*, size_t) -> int {
        ++*static_cast<int*>(user);
        throw std::runtime_error("C++ callback failure");
    };
    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    sonder_status status = SONDER_OK;
    CHECK_NOTHROW(status = sonder_session_generate(session.get(), "callback exception", throwing, &calls, &stats));
    CHECK(status == SONDER_ERROR_INTERNAL);
    CHECK(calls == 1);
    Count count;
    REQUIRE(sonder_session_generate(session.get(), "immediate reuse", on_token, &count, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    CHECK(count.chunks == 4);
    CHECK(calls == 1);
}

namespace {
struct TelemetryCapture {
    mutable std::mutex mutex;
    std::vector<std::string> lines;

    std::vector<std::string> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        return lines;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex);
        return lines.size();
    }
};

void capture_telemetry_line(void* user_data, const char* json_line, size_t length) {
    auto* cap = static_cast<TelemetryCapture*>(user_data);
    std::lock_guard<std::mutex> lock(cap->mutex);
    cap->lines.emplace_back(json_line, length);
}

bool contains_event(const std::vector<std::string>& lines, const char* needle) {
    for (const auto& line : lines) {
        if (line.find(needle) != std::string::npos) {
            return true;
        }
    }
    return false;
}
}  // namespace

TEST_CASE("GPU inventory C ABI enumerates devices and validates arguments") {
    static_assert(std::is_standard_layout_v<sonder_gpu_info>);
    static_assert(std::is_trivial_v<sonder_gpu_info>);
    static_assert(std::is_trivially_copyable_v<sonder_gpu_info>);

    CHECK(sonder_gpu_device_info(0, nullptr) == SONDER_ERROR_INVALID_ARGUMENT);
    sonder_gpu_info undersized{};
    undersized.struct_size = 0;
    CHECK(sonder_gpu_device_info(0, &undersized) == SONDER_ERROR_INVALID_ARGUMENT);
    undersized.struct_size = static_cast<uint32_t>(sizeof(sonder_gpu_info) - 1);
    CHECK(sonder_gpu_device_info(0, &undersized) == SONDER_ERROR_INVALID_ARGUMENT);

    const size_t count = sonder_gpu_device_count();
    for (size_t i = 0; i < count; ++i) {
        sonder_gpu_info info;
        std::memset(&info, 0xA5, sizeof(info));
        info.struct_size = sizeof(sonder_gpu_info);
        REQUIRE(sonder_gpu_device_info(i, &info) == SONDER_OK);
        CHECK(info.struct_size == sizeof(sonder_gpu_info));
        CHECK(std::memchr(info.id, '\0', sizeof(info.id)) != nullptr);
        CHECK(std::memchr(info.name, '\0', sizeof(info.name)) != nullptr);
        CHECK(std::memchr(info.vendor, '\0', sizeof(info.vendor)) != nullptr);
        CHECK(std::memchr(info.backend_hint, '\0', sizeof(info.backend_hint)) != nullptr);
        CHECK(std::memchr(info.source, '\0', sizeof(info.source)) != nullptr);
        CHECK(info.id[0] != '\0');
        CHECK(info.vendor[0] != '\0');
        CHECK(info.source[0] != '\0');
        CHECK((info.is_integrated_uma == 0 || info.is_integrated_uma == 1));
        if (!info.has_dedicated_total) CHECK(info.dedicated_total_bytes == 0);
        if (!info.has_dedicated_free) CHECK(info.dedicated_free_bytes == 0);
        if (!info.has_shared_total) CHECK(info.shared_total_bytes == 0);
        if (!info.has_effective_vram) CHECK(info.effective_vram_mib == 0);
        if (!info.has_effective_free_vram) CHECK(info.effective_free_vram_mib == 0);
        // Implicit padding bytes between device_id and dedicated_total_bytes must be zeroed.
        const auto* raw = reinterpret_cast<const unsigned char*>(&info);
        const std::size_t pad_begin = offsetof(sonder_gpu_info, device_id) + sizeof(info.device_id);
        const std::size_t pad_end = offsetof(sonder_gpu_info, dedicated_total_bytes);
        for (std::size_t p = pad_begin; p < pad_end; ++p) {
            CHECK(raw[p] == 0u);
        }
    }

    sonder_gpu_info oob{};
    oob.struct_size = sizeof(sonder_gpu_info);
    CHECK(sonder_gpu_device_info(count, &oob) == SONDER_ERROR_NOT_FOUND);
    CHECK(sonder_gpu_device_info(count + 64, &oob) == SONDER_ERROR_NOT_FOUND);

    struct ExtendedGpuInfo {
        sonder_gpu_info base;
        uint64_t guard;
    } extended{};
    extended.base.struct_size = sizeof(ExtendedGpuInfo);
    extended.guard = 0x123456789ABCDEF0ull;
    if (count > 0) {
        REQUIRE(sonder_gpu_device_info(0, &extended.base) == SONDER_OK);
        CHECK(extended.guard == 0x123456789ABCDEF0ull);
    } else {
        CHECK(sonder_gpu_device_info(0, &extended.base) == SONDER_ERROR_NOT_FOUND);
    }
}

TEST_CASE("request metadata v1 backwards compatibility and scheduling validation") {
    struct LegacyRequestMetadata {
        uint32_t struct_size;
        const char* parent_request_id;
    };
    static_assert(std::is_standard_layout_v<sonder_request_metadata>);
    static_assert(std::is_trivial_v<sonder_request_metadata>);
    static_assert(std::is_trivially_copyable_v<sonder_request_metadata>);
    static_assert(sizeof(LegacyRequestMetadata) < sizeof(sonder_request_metadata));
    static_assert(SONDER_WORKLOAD_DEFAULT == 0);
    static_assert(SONDER_WORKLOAD_INTERACTIVE_USER == 1);
    static_assert(SONDER_WORKLOAD_MAINTENANCE == 7);

    TelemetryCapture legacy_cap;
    TelemetryCapture sched_cap;

    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_METRICS;
    sonder_engine* raw_engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &raw_engine) == SONDER_OK);
    const std::unique_ptr<sonder_engine, decltype(&sonder_engine_destroy)> engine(raw_engine, &sonder_engine_destroy);

    REQUIRE(sonder_engine_set_telemetry_callback(engine.get(), capture_telemetry_line, &legacy_cap) == SONDER_OK);
    REQUIRE(sonder_engine_register_mock_backend(engine.get()) == SONDER_OK);

    sonder_model* raw_model = nullptr;
    REQUIRE(sonder_model_load(engine.get(), "mock", "mock:tiny", &raw_model) == SONDER_OK);
    const std::unique_ptr<sonder_model, decltype(&sonder_model_release)> model(raw_model, &sonder_model_release);

    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    cfg.temperature = 0.0f;
    cfg.max_tokens = 3;
    sonder_session* raw_session = nullptr;
    REQUIRE(sonder_session_create(engine.get(), model.get(), &cfg, &raw_session) == SONDER_OK);
    const std::unique_ptr<sonder_session, decltype(&sonder_session_destroy)> session(raw_session,
                                                                                     &sonder_session_destroy);

    // Legacy v1 struct_size: invalid values past v1 must be ignored.
    sonder_request_metadata legacy{};
    legacy.struct_size = sizeof(LegacyRequestMetadata);
    legacy.parent_request_id = "legacy-v1:parent";
    legacy.workload_class = 999;
    legacy.priority = -999;
    const sonder_chat_message msg{sizeof(sonder_chat_message), "user", "hi"};
    const sonder_chat_message* messages[] = {&msg};
    REQUIRE(sonder_session_generate_with_metadata(session.get(), "legacy gen", &legacy, nullptr, nullptr, nullptr) ==
            SONDER_OK);
    REQUIRE(sonder_session_chat_with_metadata(session.get(), messages, 1, &legacy, nullptr, nullptr, nullptr) ==
            SONDER_OK);

    // Zero-initialized full struct selects SONDER_WORKLOAD_DEFAULT (0) and priority 0 (no escalation).
    sonder_request_metadata zeroed;
    std::memset(&zeroed, 0, sizeof(zeroed));
    zeroed.struct_size = sizeof(sonder_request_metadata);
    zeroed.parent_request_id = "zeroed:parent";
    REQUIRE(sonder_session_generate_with_metadata(session.get(), "zeroed gen", &zeroed, nullptr, nullptr, nullptr) ==
            SONDER_OK);

    // Flush legacy + zeroed events into legacy_cap and verify no priority_class was injected.
    REQUIRE(sonder_engine_set_telemetry_callback(engine.get(), capture_telemetry_line, &sched_cap) == SONDER_OK);
    const auto legacy_lines = legacy_cap.snapshot();
    CHECK(contains_event(legacy_lines, "\"parent_request_id\":\"legacy-v1:parent\""));
    CHECK(contains_event(legacy_lines, "\"parent_request_id\":\"zeroed:parent\""));
    CHECK_FALSE(contains_event(legacy_lines, "\"priority_class\""));

    // Undersized struct_size is rejected before submission.
    sonder_request_metadata bad_size{};
    bad_size.struct_size = static_cast<uint32_t>(sizeof(LegacyRequestMetadata) - 1);
    Count count;
    CHECK(sonder_session_generate_with_metadata(session.get(), "bad size", &bad_size, on_token, &count, nullptr) ==
          SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(sonder_session_chat_with_metadata(session.get(), messages, 1, &bad_size, on_token, &count, nullptr) ==
          SONDER_ERROR_INVALID_ARGUMENT);
    CHECK(count.chunks == 0);

    // Valid extended scheduling metadata (all boundaries and classes).
    sonder_request_metadata ext{};
    ext.struct_size = sizeof(sonder_request_metadata);
    ext.parent_request_id = "sched-parent:1";
    ext.workload_class = SONDER_WORKLOAD_BACKGROUND_INDEXING;
    ext.priority = 0;
    REQUIRE(sonder_session_generate_with_metadata(session.get(), "bg req", &ext, nullptr, nullptr, nullptr) ==
            SONDER_OK);

    ext.workload_class = SONDER_WORKLOAD_IMPLEMENTATION_WORKER;
    ext.priority = 0;
    REQUIRE(sonder_session_generate_with_metadata(session.get(), "subagent req", &ext, nullptr, nullptr, nullptr) ==
            SONDER_OK);

    ext.workload_class = SONDER_WORKLOAD_INTERACTIVE_USER;
    ext.priority = 16;
    REQUIRE(sonder_session_chat_with_metadata(session.get(), messages, 1, &ext, nullptr, nullptr, nullptr) ==
            SONDER_OK);

    ext.workload_class = SONDER_WORKLOAD_MAINTENANCE;
    ext.priority = -16;
    REQUIRE(sonder_session_generate_with_metadata(session.get(), "maint req", &ext, nullptr, nullptr, nullptr) ==
            SONDER_OK);

    ext.workload_class = SONDER_WORKLOAD_DEFAULT;
    ext.priority = 0;
    REQUIRE(sonder_session_generate_with_metadata(session.get(), "default req", &ext, nullptr, nullptr, nullptr) ==
            SONDER_OK);

    REQUIRE(sonder_engine_set_telemetry_callback(engine.get(), nullptr, nullptr) == SONDER_OK);
    const auto sched_lines = sched_cap.snapshot();
    CHECK(contains_event(sched_lines, "\"priority_class\":\"background\""));
    CHECK(contains_event(sched_lines, "\"priority_class\":\"subagent\""));
    CHECK(contains_event(sched_lines, "\"priority_class\":\"interactive\""));

    // Out-of-range workload_class and priority are rejected before submission.
    for (const int32_t bad_workload : {-2, -1, 8, 42}) {
        ext.workload_class = bad_workload;
        ext.priority = 0;
        CHECK(sonder_session_generate_with_metadata(session.get(), "bad wl", &ext, on_token, &count, nullptr) ==
              SONDER_ERROR_INVALID_ARGUMENT);
        CHECK(std::string(sonder_last_error_message()).find("workload_class") != std::string::npos);
        CHECK(sonder_session_chat_with_metadata(session.get(), messages, 1, &ext, on_token, &count, nullptr) ==
              SONDER_ERROR_INVALID_ARGUMENT);
    }
    ext.workload_class = SONDER_WORKLOAD_DEFAULT;
    for (const int32_t bad_priority : {-17, 17, 100}) {
        ext.priority = bad_priority;
        CHECK(sonder_session_generate_with_metadata(session.get(), "bad prio", &ext, on_token, &count, nullptr) ==
              SONDER_ERROR_INVALID_ARGUMENT);
        CHECK(std::string(sonder_last_error_message()).find("priority") != std::string::npos);
        CHECK(sonder_session_chat_with_metadata(session.get(), messages, 1, &ext, on_token, &count, nullptr) ==
              SONDER_ERROR_INVALID_ARGUMENT);
    }
    CHECK(count.chunks == 0);
}

TEST_CASE("telemetry callback receives observatory events and stops after clearing") {
    CHECK(sonder_engine_set_telemetry_callback(nullptr, capture_telemetry_line, nullptr) ==
          SONDER_ERROR_INVALID_ARGUMENT);

    TelemetryCapture cap1;
    TelemetryCapture cap2;

    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_STANDARD;
    eo.telemetry_jsonl_path = nullptr;
    sonder_engine* raw_engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &raw_engine) == SONDER_OK);
    std::unique_ptr<sonder_engine, decltype(&sonder_engine_destroy)> engine(raw_engine, &sonder_engine_destroy);

    // Clearing before any callback is registered is a valid no-op.
    CHECK(sonder_engine_set_telemetry_callback(engine.get(), nullptr, nullptr) == SONDER_OK);

    REQUIRE(sonder_engine_set_telemetry_callback(engine.get(), capture_telemetry_line, &cap1) == SONDER_OK);
    REQUIRE(sonder_engine_register_mock_backend(engine.get()) == SONDER_OK);

    sonder_model* raw_model = nullptr;
    REQUIRE(sonder_model_load(engine.get(), "mock", "mock:tiny", &raw_model) == SONDER_OK);
    std::unique_ptr<sonder_model, decltype(&sonder_model_release)> model(raw_model, &sonder_model_release);

    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    cfg.temperature = 0.0f;
    cfg.max_tokens = 4;
    sonder_session* raw_session = nullptr;
    REQUIRE(sonder_session_create(engine.get(), model.get(), &cfg, &raw_session) == SONDER_OK);
    std::unique_ptr<sonder_session, decltype(&sonder_session_destroy)> session(raw_session, &sonder_session_destroy);

    sonder_generation_stats stats{};
    stats.struct_size = sizeof(stats);
    REQUIRE(sonder_session_generate(session.get(), "first callback generation", nullptr, nullptr, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);

    // Replacing callback flushes cap1 and switches delivery to cap2.
    REQUIRE(sonder_engine_set_telemetry_callback(engine.get(), capture_telemetry_line, &cap2) == SONDER_OK);
    const auto lines1 = cap1.snapshot();
    REQUIRE_FALSE(lines1.empty());
    for (const auto& line : lines1) {
        CHECK(line.find("\"schema\":\"sonder.observatory.event/1\"") != std::string::npos);
        CHECK(line.find('\n') == std::string::npos);
    }
    CHECK(contains_event(lines1, "\"event_type\":\"request.queued\""));
    CHECK(contains_event(lines1, "\"event_type\":\"request.started\""));
    CHECK(contains_event(lines1, "\"event_type\":\"inference.token.generated\""));
    CHECK(contains_event(lines1, "\"event_type\":\"request.completed\""));
    const std::size_t cap1_after_switch = cap1.size();

    REQUIRE(sonder_session_generate(session.get(), "second callback generation", nullptr, nullptr, &stats) ==
            SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);

    // Clearing with NULL flushes cap2 and detaches the callback sink.
    REQUIRE(sonder_engine_set_telemetry_callback(engine.get(), nullptr, nullptr) == SONDER_OK);
    const auto lines2 = cap2.snapshot();
    REQUIRE_FALSE(lines2.empty());
    CHECK(contains_event(lines2, "\"schema\":\"sonder.observatory.event/1\""));
    CHECK(contains_event(lines2, "\"event_type\":\"request.completed\""));
    const std::size_t cap2_after_clear = cap2.size();

    // Subsequent generation and full teardown drain must not invoke either detached callback.
    REQUIRE(sonder_session_generate(session.get(), "after clear generation", nullptr, nullptr, &stats) == SONDER_OK);
    CHECK(stats.outcome == SONDER_OUTCOME_COMPLETED);
    session.reset();
    model.reset();
    engine.reset();
    CHECK(cap1.size() == cap1_after_switch);
    CHECK(cap2.size() == cap2_after_clear);
}

TEST_CASE("telemetry callback contains exceptions and detaches on engine destroy") {
    sonder_engine_options eo;
    sonder_engine_options_init(&eo);
    eo.telemetry_level = SONDER_TELEMETRY_METRICS;
    sonder_engine* engine = nullptr;
    REQUIRE(sonder_engine_create(&eo, &engine) == SONDER_OK);
    REQUIRE(sonder_engine_register_mock_backend(engine) == SONDER_OK);

    sonder_model* model = nullptr;
    REQUIRE(sonder_model_load(engine, "mock", "mock:tiny", &model) == SONDER_OK);
    sonder_sampling_config cfg;
    sonder_sampling_config_init(&cfg);
    cfg.temperature = 0.0f;
    cfg.max_tokens = 3;
    sonder_session* session = nullptr;
    REQUIRE(sonder_session_create(engine, model, &cfg, &session) == SONDER_OK);
    sonder_model_release(model);

    std::atomic<int> throwing_calls{0};
    const auto throwing_cb = [](void* user, const char*, size_t) {
        static_cast<std::atomic<int>*>(user)->fetch_add(1, std::memory_order_relaxed);
        throw std::runtime_error("C++ telemetry callback exception");
    };
    REQUIRE(sonder_engine_set_telemetry_callback(engine, throwing_cb, &throwing_calls) == SONDER_OK);
    REQUIRE(sonder_session_generate(session, "throwing telemetry", nullptr, nullptr, nullptr) == SONDER_OK);

    // Switch to a clean counter callback; sink must still be alive after throwing_cb.
    std::atomic<int> clean_calls{0};
    const auto clean_cb = [](void* user, const char*, size_t) {
        static_cast<std::atomic<int>*>(user)->fetch_add(1, std::memory_order_relaxed);
    };
    REQUIRE(sonder_engine_set_telemetry_callback(engine, clean_cb, &clean_calls) == SONDER_OK);
    CHECK(throwing_calls.load(std::memory_order_relaxed) > 0);
    REQUIRE(sonder_session_generate(session, "before engine destroy", nullptr, nullptr, nullptr) == SONDER_OK);

    // Destroy engine BEFORE session: callback must flush and detach cleanly so
    // later session activity on the kept-alive engine never invokes a dangling callback.
    sonder_engine_destroy(engine);
    const int calls_at_destroy = clean_calls.load(std::memory_order_relaxed);
    CHECK(calls_at_destroy > 0);

    REQUIRE(sonder_session_generate(session, "after engine destroy", nullptr, nullptr, nullptr) == SONDER_OK);
    sonder_session_destroy(session);
    CHECK(clean_calls.load(std::memory_order_relaxed) == calls_at_destroy);
}

}
