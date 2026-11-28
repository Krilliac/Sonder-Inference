#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <stdexcept>
#include <set>
#include <thread>

#include "test_helpers.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

namespace {
template <class Marker>
[[noreturn]] void throw_callback_marker(Marker marker, const void*& address) {
    try {
        throw marker;
    } catch (const Marker& caught) {
        address = &caught;
        throw;  // Rethrow the actual live exception object, without capture/copy.
    }
}
}  // namespace

TEST_SUITE("session") {
TEST_CASE("telemetry records a 64-bit seed above INT64_MAX exactly") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    REQUIRE(h.model);
    auto sampling = SamplingConfig::greedy(4);
    sampling.seed = 18446744073709551615ull;
    auto session = h.session(sampling);
    REQUIRE(session);
    REQUIRE(session->generate("seeded").ok());
    h.engine->telemetry().flush();
    int seen = 0;
    for (const auto& line : h.sink->lines()) {
        if (line.find("\"event_type\":\"session.created\"") != std::string::npos ||
            line.find("\"event_type\":\"request.started\"") != std::string::npos) {
            CHECK(line.find("\"seed\":18446744073709551615") != std::string::npos);
            CHECK(line.find("\"seed\":-") == std::string::npos);
            ++seen;
        }
    }
    CHECK(seen == 2);
}

TEST_CASE("lifecycle: idle -> running -> idle -> closed") {
    sonder_test::Harness h;
    REQUIRE(h.model);
    auto session = h.session();
    REQUIRE(session);
    CHECK(session->state() == SessionState::idle);
    CHECK(session->last_outcome() == RequestOutcome::none);
    CHECK_FALSE(session->id().empty());

    SessionState during = SessionState::idle;
    std::string streamed;
    auto r = session->generate("hello there", [&](const TokenChunk& c) {
        during = session->state();
        streamed.append(c.text);
        return true;
    });
    REQUIRE(r.ok());
    CHECK(during == SessionState::running);
    CHECK(session->state() == SessionState::idle);
    CHECK(r.value().outcome == RequestOutcome::completed);
    CHECK(r.value().text == streamed);
    CHECK(r.value().ttft_ms >= 0.0);
    CHECK(r.value().total_ms >= r.value().ttft_ms);
    CHECK(session->last_outcome() == RequestOutcome::completed);

    auto r2 = session->generate("second request");
    REQUIRE(r2.ok());
    CHECK(r2.value().request_id != r.value().request_id);
    CHECK(session->requests_started() == 2);

    session->close();
    CHECK(session->state() == SessionState::closed);
    session->close();  // idempotent
    CHECK(session->generate("closed").status().code() == ErrorCode::invalid_state);
}

TEST_CASE("rejects a concurrent request on the same session") {
    MockBackendOptions slow;
    slow.token_delay = 20ms;
    slow.default_completion_tokens = 20;
    sonder_test::Harness h(slow);
    auto session = h.session(SamplingConfig::greedy(20));
    REQUIRE(session);
    std::atomic<bool> started{false};
    std::thread worker([&] {
        (void)session->generate("long", [&](const TokenChunk&) {
            started = true;
            return true;
        });
    });
    while (!started) {
        std::this_thread::sleep_for(1ms);
    }
    CHECK(session->generate("overlap").status().code() == ErrorCode::invalid_state);
    session->cancel();
    worker.join();
    CHECK(session->state() == SessionState::idle);
}

TEST_CASE("backend failure marks the request failed but keeps the session usable") {
    MockBackendOptions failing;
    failing.fail_after_tokens = 3;
    sonder_test::Harness h(failing);
    auto session = h.session();
    REQUIRE(session);
    auto r = session->generate("will fail");
    CHECK(r.status().code() == ErrorCode::backend_error);
    CHECK(session->last_outcome() == RequestOutcome::failed);
    CHECK(session->state() == SessionState::idle);
    auto failed = h.events_of("request.failed");
    REQUIRE(failed.size() == 1);
    CHECK(failed[0].find("attributes")->find("error_code")->as_string() == "backend_error");
}

TEST_CASE("emits correlated lifecycle events in order") {
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(4));
    REQUIRE(session);
    auto r = session->generate("trace me");
    REQUIRE(r.ok());
    const std::string rid = r.value().request_id;

    std::vector<std::string> types;
    std::vector<std::string> all_types;
    for (const auto& e : h.events()) {
        const json::Value* req = e.find("request_id");
        if (req && req->as_string() == rid) {
            const std::string type = e.find("event_type")->as_string();
            all_types.push_back(type);
            // Scheduler/KV/prefill events are covered by test_engine_runtime.cpp.
            if (type.rfind("request.", 0) == 0 || type.rfind("inference.decode.", 0) == 0 ||
                type == "inference.token.generated") {
                types.push_back(type);
            }
            CHECK(e.find("session_id")->as_string() == session->id());
            CHECK(e.find("model_instance_id")->as_string() == h.model->instance_id());
            CHECK(e.find("device_id")->as_string() == "cpu:0");
        }
    }
    const std::vector<std::string> expected = {
        "request.queued",           "request.started",          "inference.decode.started",
        "inference.token.generated", "inference.token.generated", "inference.token.generated",
        "inference.token.generated", "inference.decode.completed", "request.completed"};
    CHECK(types == expected);
    if (h.engine->scheduling_active()) {
        auto pos = [&](const std::string& t) {
            return std::find(all_types.begin(), all_types.end(), t) - all_types.begin();
        };
        const auto n = static_cast<std::ptrdiff_t>(all_types.size());
        REQUIRE(pos("scheduler.enqueued") < n);
        CHECK(pos("scheduler.enqueued") < pos("scheduler.admitted"));
        CHECK(pos("scheduler.admitted") < pos("scheduler.prefill.completed"));
        CHECK(pos("scheduler.prefill.completed") < pos("inference.decode.started"));
        CHECK(pos("kv.freed") < pos("request.completed"));
    }

    auto done = h.events_of("request.completed");
    REQUIRE(done.size() == 1);
    const auto& a = *done[0].find("attributes");
    CHECK(a.find("completion_tokens")->as_int() == 4);
    CHECK(a.find("prompt_tokens")->as_int() == 2);
    CHECK(a.find("stop_reason")->as_string() == "max_tokens");
    CHECK(a.find("total_ms")->is_number());
    CHECK(a.find("ttft_ms")->is_number());
}

TEST_CASE("model handles outlive unload while sessions hold them") {
    sonder_test::Harness h;
    auto session = h.session();
    REQUIRE(session);
    const std::string id = h.model->instance_id();
    CHECK(h.engine->unload_model(id).ok());
    CHECK(h.engine->unload_model(id).code() == ErrorCode::not_found);
    CHECK(h.engine->loaded_models().empty());
    auto r = session->generate("still works");
    REQUIRE(r.ok());
    CHECK(r.value().outcome == RequestOutcome::completed);
    CHECK(h.events_of("model.unload").size() == 1);
}

TEST_CASE("engine rejects unknown backends and duplicate registration") {
    sonder_test::Harness h;
    ModelLoadOptions lo;
    lo.model = "x";
    CHECK(h.engine->load_model("nope", lo).status().code() == ErrorCode::not_found);
    CHECK(h.engine->register_backend(make_mock_backend()).code() == ErrorCode::invalid_state);
    CHECK(h.engine->register_backend(nullptr).code() == ErrorCode::invalid_argument);
    CHECK(h.engine->create_session(nullptr).status().code() == ErrorCode::invalid_argument);
    CHECK(h.events_of("model.load.completed").size() == 1);
}
TEST_CASE("throwing token callbacks preserve exceptions and allow immediate reuse") {
    for (const bool chat : {false, true}) {
        for (const int throw_at : {1, 3}) {
            for (const bool nonstandard : {false, true}) {
                CAPTURE(chat);
                CAPTURE(throw_at);
                CAPTURE(nonstandard);
                sonder_test::Harness h({}, TelemetryLevel::standard, false);
                auto session = h.session(SamplingConfig::greedy(5));
                REQUIRE(session);
                const std::string secret = "callback-exception-private-canary";
                const void* thrown = nullptr;
                RequestOptions options;
                options.request_id = "callback-failed-request";
                options.parent_request_id = "callback-parent";
                int calls = 0;
                bool overlap_rejected = false;
                std::string delivered;
                const TokenCallback callback = [&](const TokenChunk& chunk) {
                    if (calls == 0) {
                        overlap_rejected = session->generate("nested request").status().code() == ErrorCode::invalid_state
                                           && session->requests_started() == 1;
                    }
                    delivered.append(chunk.text);
                    if (++calls == throw_at) {
                        if (nonstandard) throw_callback_marker(73, thrown);
                        throw_callback_marker(std::runtime_error(secret), thrown);
                    }
                    return true;
                };
                bool caught = false;
                try {
                    if (chat) (void)session->chat({{"user", "first request"}}, callback, std::nullopt, options);
                    else (void)session->generate("first request", callback, std::nullopt, options);
                } catch (const std::runtime_error& error) {
                    caught = true;
                    CHECK_FALSE(nonstandard);
                    CHECK(static_cast<const void*>(&error) == thrown);
                    CHECK(std::string(error.what()) == secret);
                } catch (const int& marker) {
                    caught = true;
                    CHECK(nonstandard);
                    CHECK(static_cast<const void*>(&marker) == thrown);
                    CHECK(marker == 73);
                }
                CHECK(caught);
                CHECK(overlap_rejected);
                CHECK(calls == throw_at);
                CHECK_FALSE(delivered.empty());  // Already delivered effects survive.
                CHECK(session->state() == SessionState::idle);
                CHECK(session->last_outcome() == RequestOutcome::failed);
                CHECK_FALSE(session->last_scheduler_rejected());
                CHECK(session->requests_started() == 1);
                const auto failed = h.events_of("request.failed");
                REQUIRE(failed.size() == 1);
                CHECK(failed[0].find("request_id")->as_string() == "callback-failed-request");
                const auto* attrs = failed[0].find("attributes");
                REQUIRE(attrs);
                CHECK(attrs->find("parent_request_id")->as_string() == "callback-parent");
                CHECK(attrs->find("error_code")->as_string() == "internal");
                CHECK(attrs->find("chunks")->as_int() == throw_at);
                CHECK(h.events_of("request.completed").empty());
                CHECK(h.events_of("request.cancelled").empty());
                for (const auto& line : h.sink->lines()) CHECK(line.find(secret) == std::string::npos);
                const auto retained = delivered;
                auto reused = session->generate("fresh request");
                REQUIRE(reused.ok());
                CHECK(reused->outcome == RequestOutcome::completed);
                CHECK(session->requests_started() == 2);
                CHECK(calls == throw_at);
                CHECK(delivered == retained);
            }
        }
    }
}

TEST_CASE("cancel or close before a callback exception preserves terminal state") {
    for (const bool close : {false, true}) {
        CAPTURE(close);
        sonder_test::Harness h;
        auto session = h.session(SamplingConfig::greedy(4));
        REQUIRE(session);
        const void* thrown = nullptr;
        int calls = 0;
        bool caught = false;
        try {
            (void)session->generate("close or cancel", [&](const TokenChunk&) -> bool {
                ++calls;
                if (close) session->close();
                else session->cancel();
                throw_callback_marker(std::runtime_error("private-close-canary"), thrown);
            });
        } catch (const std::runtime_error& error) {
            caught = true;
            CHECK(static_cast<const void*>(&error) == thrown);
            CHECK(std::string(error.what()) == "private-close-canary");
        }
        CHECK(caught);
        CHECK(calls == 1);
        CHECK(session->last_outcome() == RequestOutcome::failed);
        CHECK_FALSE(session->last_scheduler_rejected());
        if (close) {
            CHECK(session->state() == SessionState::closed);
            CHECK(session->generate("must stay closed").status().code() == ErrorCode::invalid_state);
            CHECK(session->requests_started() == 1);
        } else {
            CHECK(session->state() == SessionState::idle);
            session->cancel();  // No stale cancellation source affects reuse.
            const auto reused = session->generate("new cancellation epoch");
            REQUIRE(reused.ok());
            CHECK(reused->outcome == RequestOutcome::completed);
            CHECK(session->requests_started() == 2);
        }
    }
}

TEST_CASE("callback exception cleanup also works without scheduling") {
    EngineOptions options;
    options.scheduling.enabled = false;
    options.telemetry.level = TelemetryLevel::off;
    Engine engine(options);
    REQUIRE(engine.register_backend(make_mock_backend()).ok());
    ModelLoadOptions load;
    load.model = "mock:tiny";
    auto model = engine.load_model(kMockBackendName, load);
    REQUIRE(model.ok());
    SessionOptions session_options;
    session_options.sampling = SamplingConfig::greedy(5);
    auto created = engine.create_session(model.value(), session_options);
    REQUIRE(created.ok());
    auto session = created.value();
    REQUIRE_FALSE(engine.scheduling_active());
    for (const int throw_at : {1, 3}) {
        CAPTURE(throw_at);
        const void* thrown = nullptr;
        int calls = 0;
        bool caught = false;
        try {
            (void)session->generate("unscheduled", [&](const TokenChunk&) {
                if (++calls == throw_at) throw_callback_marker(91, thrown);
                return true;
            });
        } catch (const int& marker) {
            caught = true;
            CHECK(static_cast<const void*>(&marker) == thrown);
            CHECK(marker == 91);
        }
        CHECK(caught);
        CHECK(calls == throw_at);
        CHECK(session->state() == SessionState::idle);
        CHECK(session->last_outcome() == RequestOutcome::failed);
        auto reused = session->generate("new unscheduled request");
        REQUIRE(reused.ok());
        CHECK(reused->outcome == RequestOutcome::completed);
        CHECK_FALSE(reused->scheduling.scheduled);
        CHECK(calls == throw_at);
    }
}

TEST_CASE("telemetry redacts raw backend failure messages from request.failed and model.load.failed (M-15)") {
    const std::string load_secret = "CANARY_LOAD_SECRET_481516";
    const std::string gen_secret = "CANARY_GEN_SECRET_234200";

    class SecretErrorModel final : public BackendModel {
    public:
        explicit SecretErrorModel(std::string secret) : secret_(std::move(secret)) {
            desc_.name = "secret-model";
            desc_.backend = "secret-backend";
        }
        const ModelDescriptor& descriptor() const override { return desc_; }
        Result<GenerateStats> generate(const GenerateRequest&, const CancellationToken&,
                                       const TokenCallback&) override {
            return Status(ErrorCode::backend_error, "upstream failed: " + secret_);
        }
    private:
        ModelDescriptor desc_;
        std::string secret_;
    };

    class SecretErrorBackend final : public Backend {
    public:
        SecretErrorBackend(std::string load_err, std::string gen_err)
            : load_err_(std::move(load_err)), gen_err_(std::move(gen_err)) {}
        std::string name() const override { return "secret-backend"; }
        std::string description() const override { return "test backend"; }
        BackendCapabilities capabilities() const override { return {}; }
        Result<std::string> probe() override { return std::string("1"); }
        Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }
        Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& opts) override {
            if (opts.model == "fail-load") {
                return Status(ErrorCode::backend_error, "loader saw " + load_err_);
            }
            return std::shared_ptr<BackendModel>(std::make_shared<SecretErrorModel>(gen_err_));
        }
    private:
        std::string load_err_;
        std::string gen_err_;
    };

    EngineOptions options;
    options.telemetry.level = TelemetryLevel::metrics;
    auto sink = std::make_shared<MemoryTelemetrySink>();
    Engine engine(options);
    engine.telemetry().add_sink(sink);
    REQUIRE(engine.register_backend(std::make_shared<SecretErrorBackend>(load_secret, gen_secret)).ok());

    ModelLoadOptions bad_load;
    bad_load.model = "fail-load";
    auto failed_load = engine.load_model("secret-backend", bad_load);
    REQUIRE_FALSE(failed_load.ok());
    CHECK(failed_load.status().message().find(load_secret) != std::string::npos);

    ModelLoadOptions ok_load;
    ok_load.model = "secret-model";
    auto model = engine.load_model("secret-backend", ok_load);
    REQUIRE(model.ok());
    auto session = engine.create_session(model.value());
    REQUIRE(session.ok());
    auto gen = session.value()->generate("hi");
    REQUIRE_FALSE(gen.ok());
    CHECK(gen.status().message().find(gen_secret) != std::string::npos);

    engine.telemetry().flush();
    bool saw_load_failed = false;
    bool saw_req_failed = false;
    for (const auto& line : sink->lines()) {
        CHECK(line.find(load_secret) == std::string::npos);
        CHECK(line.find(gen_secret) == std::string::npos);
        if (line.find("\"model.load.failed\"") != std::string::npos) saw_load_failed = true;
        if (line.find("\"request.failed\"") != std::string::npos) saw_req_failed = true;
    }
    CHECK(saw_load_failed);
    CHECK(saw_req_failed);
}

TEST_CASE("session enforces cumulative response byte cap (M-16)") {
    sonder_test::Harness h;
    auto session = h.session(SamplingConfig::greedy(16));
    REQUIRE(session);
    RequestOptions ro;
    ro.max_response_bytes = 8;
    std::size_t received = 0;
    auto r = session->generate(
        "long response",
        [&](const TokenChunk& c) {
            received += c.text.size() + c.reasoning.size();
            return true;
        },
        std::nullopt, ro);
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::protocol_error);
    CHECK(received <= ro.max_response_bytes);
    CHECK(session->last_outcome() == RequestOutcome::failed);
    CHECK(session->state() == SessionState::idle);
}

TEST_CASE("embed emits request.queued, request.started and request.completed with parent_request_id and admission") {
    sonder_test::Harness h({}, TelemetryLevel::metrics);
    REQUIRE(h.model);
    SessionOptions so;
    so.run_id = "embed-run-1";
    so.agent_id = "agent-embed";
    so.task_id = "task-embed";
    so.workload = WorkloadClass::owner_orchestrator;
    so.priority = 3;
    auto created = h.engine->create_session(h.model, so);
    REQUIRE(created.ok());
    auto session = created.value();

    RequestOptions opts;
    opts.request_id = "embed-req-1";
    opts.parent_request_id = "turn-E1";
    opts.priority_class = RequestPriority::subagent;
    opts.admission_queue_ms = 4.5;
    auto r = session->embed({"hello", "world"}, opts);
    REQUIRE(r.ok());
    CHECK(r.value().embeddings.size() == 2u);
    CHECK(r.value().prompt_tokens > 0u);
    CHECK(session->state() == SessionState::idle);
    CHECK(session->last_outcome() == RequestOutcome::completed);

    std::vector<std::string> lifecycle;
    for (const auto& e : h.events()) {
        const json::Value* rid = e.find("request_id");
        if (!rid || rid->as_string() != "embed-req-1") continue;
        const std::string type = e.find("event_type")->as_string();
        if (type.rfind("request.", 0) != 0) continue;
        lifecycle.push_back(type);
        CHECK(e.find("session_id")->as_string() == session->id());
        CHECK(e.find("model_instance_id")->as_string() == h.model->instance_id());
        CHECK(e.find("device_id")->as_string() == "cpu:0");
        CHECK(e.find("run_id")->as_string() == "embed-run-1");
        CHECK(e.find("agent_id")->as_string() == "agent-embed");
        CHECK(e.find("task_id")->as_string() == "task-embed");
        const auto* attrs = e.find("attributes");
        REQUIRE(attrs != nullptr);
        CHECK(attrs->find("kind")->as_string() == "embed");
        REQUIRE(attrs->find("parent_request_id") != nullptr);
        CHECK(attrs->find("parent_request_id")->as_string() == "turn-E1");
        REQUIRE(attrs->find("priority_class") != nullptr);
        CHECK(attrs->find("priority_class")->as_string() == "subagent");
        const auto* adm = attrs->find("admission");
        REQUIRE(adm != nullptr);
        CHECK(adm->find("priority")->as_string() == "subagent");
        CHECK(adm->find("queue_ms")->as_double() == doctest::Approx(4.5));
        if (type == "request.queued") {
            CHECK(attrs->find("inputs")->as_uint() == 2u);
            CHECK(attrs->find("workload")->as_string() == "owner_orchestrator");
            CHECK(attrs->find("priority")->as_int() == 3);
        } else if (type == "request.started") {
            CHECK(attrs->find("inputs")->as_uint() == 2u);
            CHECK_FALSE(attrs->find("scheduled")->as_bool());
        } else if (type == "request.completed") {
            CHECK(attrs->find("outcome")->as_string() == "completed");
            CHECK(attrs->find("inputs")->as_uint() == 2u);
            CHECK(attrs->find("prompt_tokens")->as_uint() == r.value().prompt_tokens);
            CHECK(attrs->find("total_ms")->as_double() >= 0.0);
        }
    }
    CHECK(lifecycle == std::vector<std::string>{"request.queued", "request.started", "request.completed"});

    // Without optional metadata, parent_request_id and priority_class are omitted.
    RequestOptions plain;
    plain.request_id = "embed-req-plain";
    REQUIRE(session->embed({"plain"}, plain).ok());
    std::vector<std::string> plain_lifecycle;
    for (const auto& e : h.events()) {
        const json::Value* rid = e.find("request_id");
        if (!rid || rid->as_string() != "embed-req-plain") continue;
        const std::string type = e.find("event_type")->as_string();
        if (type.rfind("request.", 0) != 0) continue;
        plain_lifecycle.push_back(type);
        const auto* attrs = e.find("attributes");
        REQUIRE(attrs != nullptr);
        CHECK(attrs->find("parent_request_id") == nullptr);
        CHECK(attrs->find("priority_class") == nullptr);
    }
    CHECK(plain_lifecycle == std::vector<std::string>{"request.queued", "request.started", "request.completed"});
}

TEST_CASE("embed emits request.started and propagates parent_request_id on failure, timeout, cancel, admission and exception") {
    enum class Mode { ok, fail_backend, throw_runtime };
    struct EmbedFaultModel final : BackendModel {
        Mode mode = Mode::ok;
        std::string secret;
        ModelDescriptor desc;

        EmbedFaultModel() {
            desc.name = "fault-embed";
            desc.backend = "fault-embed-backend";
            desc.context_length = 4096;
        }

        const ModelDescriptor& descriptor() const override { return desc; }
        Result<GenerateStats> generate(const GenerateRequest&, const CancellationToken&,
                                       const TokenCallback&) override {
            return Status(ErrorCode::unsupported, "generate unused");
        }
        Result<EmbedResult> embed(const EmbedRequest& req, const CancellationToken&) override {
            if (mode == Mode::fail_backend) {
                return Status(ErrorCode::backend_error, "embed failed: " + secret);
            }
            if (mode == Mode::throw_runtime) {
                throw std::runtime_error("embed threw: " + secret);
            }
            EmbedResult out;
            out.embeddings.assign(req.inputs.size(), {0.25f, 0.75f});
            out.prompt_tokens = static_cast<std::uint64_t>(req.inputs.size()) * 2u;
            return out;
        }
    };
    struct EmbedFaultBackend final : Backend {
        std::shared_ptr<EmbedFaultModel> model = std::make_shared<EmbedFaultModel>();
        std::string name() const override { return "fault-embed-backend"; }
        std::string description() const override { return "test embed fault backend"; }
        BackendCapabilities capabilities() const override {
            BackendCapabilities caps;
            caps.add(Capability::embeddings);
            return caps;
        }
        Result<std::string> probe() override { return std::string("1"); }
        Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{model->desc}; }
        Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions&) override {
            return std::shared_ptr<BackendModel>(model);
        }
    };
    struct StubAdmission final : RequestAdmission {
        Status wait_status = Status::success();
        double measured_queue_ms = 0.0;
        bool ready() const override { return wait_status.ok(); }
        std::uint64_t order() const noexcept override { return 0; }
        bool try_acquire() override { return wait_status.ok(); }
        Status wait(const CancellationToken&) override { return wait_status; }
        double queue_ms() const override { return measured_queue_ms; }
    };

    EngineOptions eo;
    eo.scheduling.enabled = false;
    eo.telemetry.level = TelemetryLevel::metrics;
    auto sink = std::make_shared<MemoryTelemetrySink>();
    Engine engine(eo);
    engine.telemetry().add_sink(sink);
    auto backend = std::make_shared<EmbedFaultBackend>();
    REQUIRE(engine.register_backend(backend).ok());
    auto loaded = engine.load_model("fault-embed-backend", ModelLoadOptions{"fault-embed", "cpu:0"});
    REQUIRE(loaded.ok());
    auto created = engine.create_session(loaded.value());
    REQUIRE(created.ok());
    auto session = created.value();

    const auto lifecycle_for = [&](const std::string& req_id, const std::string& expected_parent) {
        engine.telemetry().flush();
        std::vector<std::string> types;
        std::vector<json::Value> events;
        for (const auto& line : sink->lines()) {
            auto parsed = json::parse(line);
            REQUIRE(parsed.ok());
            const auto* rid = parsed->find("request_id");
            if (!rid || rid->as_string() != req_id) continue;
            const std::string type = parsed->find("event_type")->as_string();
            if (type.rfind("request.", 0) != 0) continue;
            const auto* attrs = parsed->find("attributes");
            REQUIRE(attrs != nullptr);
            CHECK(attrs->find("kind")->as_string() == "embed");
            REQUIRE(attrs->find("parent_request_id") != nullptr);
            CHECK(attrs->find("parent_request_id")->as_string() == expected_parent);
            types.push_back(type);
            events.push_back(std::move(parsed).value());
        }
        return std::make_pair(std::move(types), std::move(events));
    };

    // 1. Backend failure (with M-15 redaction of raw upstream message).
    const std::string fail_secret = "CANARY_EMBED_FAIL_SECRET_777";
    backend->model->mode = Mode::fail_backend;
    backend->model->secret = fail_secret;
    {
        RequestOptions ro;
        ro.request_id = "embed-fail";
        ro.parent_request_id = "turn-fail";
        auto res = session->embed({"a"}, ro);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::backend_error);
        const auto [types, events] = lifecycle_for("embed-fail", "turn-fail");
        CHECK(types == std::vector<std::string>{"request.queued", "request.started", "request.failed"});
        REQUIRE(events.size() == 3u);
        const auto* failed_attrs = events[2].find("attributes");
        CHECK(failed_attrs->find("outcome")->as_string() == "failed");
        CHECK(failed_attrs->find("error_code")->as_string() == "backend_error");
        CHECK(failed_attrs->find("error")->as_string() == "backend_error");
        CHECK(failed_attrs->find("total_ms")->as_double() >= 0.0);
    }

    // 2. Deadline exceeded before backend start.
    backend->model->mode = Mode::ok;
    {
        RequestOptions ro;
        ro.request_id = "embed-timeout";
        ro.parent_request_id = "turn-timeout";
        ro.deadline = std::chrono::steady_clock::now() - 1ms;
        auto res = session->embed({"a"}, ro);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::timeout);
        const auto [types, events] = lifecycle_for("embed-timeout", "turn-timeout");
        CHECK(types == std::vector<std::string>{"request.queued", "request.started", "request.failed"});
        REQUIRE(events.size() == 3u);
        CHECK(events[2].find("attributes")->find("error_code")->as_string() == "timeout");
    }

    // 3. Cancelled at on_backend_start boundary.
    {
        RequestOptions ro;
        ro.request_id = "embed-cancel";
        ro.parent_request_id = "turn-cancel";
        ro.on_backend_start = [&] { session->cancel(); };
        auto res = session->embed({"a"}, ro);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::cancelled);
        const auto [types, events] = lifecycle_for("embed-cancel", "turn-cancel");
        CHECK(types == std::vector<std::string>{"request.queued", "request.started", "request.cancelled"});
        REQUIRE(events.size() == 3u);
        const auto* cancel_attrs = events[2].find("attributes");
        CHECK(cancel_attrs->find("outcome")->as_string() == "cancelled");
        REQUIRE(cancel_attrs->find("cancel_latency_ms") != nullptr);
        CHECK(cancel_attrs->find("cancel_latency_ms")->as_double() >= 0.0);
    }

    // 4. Admission wait failure reports measured admission queue_ms and error_code.
    {
        auto adm = std::make_shared<StubAdmission>();
        adm->wait_status = Status(ErrorCode::timeout, "admission expired");
        adm->measured_queue_ms = 17.25;
        RequestOptions ro;
        ro.request_id = "embed-admit-fail";
        ro.parent_request_id = "turn-admit";
        ro.priority_class = RequestPriority::background;
        ro.admission = adm;
        auto res = session->embed({"a"}, ro);
        REQUIRE_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::timeout);
        const auto [types, events] = lifecycle_for("embed-admit-fail", "turn-admit");
        CHECK(types == std::vector<std::string>{"request.queued", "request.started", "request.failed"});
        REQUIRE(events.size() == 3u);
        const auto* failed_attrs = events[2].find("attributes");
        CHECK(failed_attrs->find("error_code")->as_string() == "timeout");
        REQUIRE(failed_attrs->find("admission") != nullptr);
        CHECK(failed_attrs->find("admission")->find("priority")->as_string() == "background");
        CHECK(failed_attrs->find("admission")->find("queue_ms")->as_double() == doctest::Approx(17.25));
    }

    // 5. Exception recovery emits request.failed with parent_request_id, redacts message, and leaves session idle.
    const std::string throw_secret = "CANARY_EMBED_THROW_SECRET_999";
    backend->model->mode = Mode::throw_runtime;
    backend->model->secret = throw_secret;
    {
        RequestOptions ro;
        ro.request_id = "embed-throw";
        ro.parent_request_id = "turn-throw";
        ro.priority_class = RequestPriority::interactive;
        ro.admission_queue_ms = 2.5;
        CHECK_THROWS_AS((void)session->embed({"a"}, ro), std::runtime_error);
        CHECK(session->state() == SessionState::idle);
        CHECK(session->last_outcome() == RequestOutcome::failed);
        const auto [types, events] = lifecycle_for("embed-throw", "turn-throw");
        CHECK(types == std::vector<std::string>{"request.queued", "request.started", "request.failed"});
        REQUIRE(events.size() == 3u);
        const auto* failed_attrs = events[2].find("attributes");
        CHECK(failed_attrs->find("outcome")->as_string() == "failed");
        CHECK(failed_attrs->find("error_code")->as_string() == "internal");
        REQUIRE(failed_attrs->find("admission") != nullptr);
        CHECK(failed_attrs->find("admission")->find("queue_ms")->as_double() == doctest::Approx(2.5));
    }

    engine.telemetry().flush();
    for (const auto& line : sink->lines()) {
        CHECK(line.find(fail_secret) == std::string::npos);
        CHECK(line.find(throw_secret) == std::string::npos);
    }

    // Session stays reusable after all fault modes.
    backend->model->mode = Mode::ok;
    auto ok = session->embed({"recovered"});
    REQUIRE(ok.ok());
    CHECK(ok.value().embeddings.size() == 1u);
}

TEST_CASE("session rerank and score lifecycle, sorting, index preservation, cancellation, and exception recovery") {
    // 1. Default mock backend: deterministic rerank + score parity and telemetry.
    {
        sonder_test::Harness h({}, TelemetryLevel::metrics);
        REQUIRE(h.model);
        auto session = h.session();
        REQUIRE(session);

        RerankRequest rreq;
        rreq.query = "sonder kv prefix cache";
        rreq.documents = {"unrelated text", "sonder kv prefix cache reuse", "prefix notes"};
        rreq.return_documents = true;
        RequestOptions ro;
        ro.request_id = "req-rerank-1";
        ro.parent_request_id = "turn-r1";
        auto rr = session->rerank(rreq, ro);
        REQUIRE_MESSAGE(rr.ok(), rr.status().message());
        CHECK(rr->model == "mock:tiny");
        CHECK(rr->prompt_tokens > 0u);
        REQUIRE(rr->results.size() == 3u);
        CHECK(rr->results[0].index == 1u);
        REQUIRE(rr->results[0].document.has_value());
        CHECK(*rr->results[0].document == "sonder kv prefix cache reuse");
        CHECK(rr->results[2].index == 0u);
        CHECK(rr->results[0].relevance_score > rr->results[1].relevance_score);
        CHECK(rr->results[1].relevance_score > rr->results[2].relevance_score);

        // top_n truncation keeps top 2 highest-scoring original indices.
        rreq.top_n = 2;
        rreq.return_documents = false;
        auto rr2 = session->rerank(rreq);
        REQUIRE(rr2.ok());
        REQUIRE(rr2->results.size() == 2u);
        CHECK(rr2->results[0].index == 1u);
        CHECK_FALSE(rr2->results[0].document.has_value());

        // Score preserves exact original pair order (not sorted) with identical scores.
        ScoreRequest sreq;
        sreq.pairs = {
            {"sonder kv prefix cache", "unrelated text"},
            {"sonder kv prefix cache", "sonder kv prefix cache reuse"},
            {"sonder kv prefix cache", "prefix notes"},
        };
        RequestOptions sro;
        sro.request_id = "req-score-1";
        auto sr = session->score(sreq, sro);
        REQUIRE_MESSAGE(sr.ok(), sr.status().message());
        CHECK(sr->model == "mock:tiny");
        REQUIRE(sr->scores.size() == 3u);
        CHECK(sr->scores[0].index == 0u);
        CHECK(sr->scores[1].index == 1u);
        CHECK(sr->scores[2].index == 2u);
        CHECK(sr->scores[1].score == doctest::Approx(rr->results[0].relevance_score));
        CHECK(sr->scores[0].score == doctest::Approx(rr->results[2].relevance_score));

        // Validation errors rejected before backend execution.
        RerankRequest bad_r;
        bad_r.query = "";
        bad_r.documents = {"a"};
        CHECK(session->rerank(bad_r).status().code() == ErrorCode::invalid_argument);
        bad_r.query = "q";
        bad_r.documents.clear();
        CHECK(session->rerank(bad_r).status().code() == ErrorCode::invalid_argument);
        bad_r.documents = {"a"};
        bad_r.top_n = 0;
        CHECK(session->rerank(bad_r).status().code() == ErrorCode::invalid_argument);

        ScoreRequest bad_s;
        CHECK(session->score(bad_s).status().code() == ErrorCode::invalid_argument);

        // EmbedRequest fail-closed dimensions validation on default mock (native 16).
        EmbedRequest ereq;
        ereq.inputs = {"alpha", "beta"};
        ereq.dimensions = 16;
        auto eok = session->embed(ereq);
        REQUIRE(eok.ok());
        CHECK(eok->dimensions == 16u);
        CHECK(eok->model == "mock:tiny");

        ereq.dimensions = 8;
        CHECK(session->embed(ereq).status().code() == ErrorCode::invalid_argument);
        ereq.dimensions = 32;
        CHECK(session->embed(ereq).status().code() == ErrorCode::invalid_argument);
        ereq.dimensions.reset();
        ereq.pooling_type = EmbedPoolingType::rank;
        CHECK(session->embed(ereq).status().code() == ErrorCode::invalid_argument);

        // Verify telemetry recorded kind="rerank" and kind="score".
        h.engine->telemetry().flush();
        bool saw_rerank = false;
        bool saw_score = false;
        for (const auto& ev : h.events_of("request.completed")) {
            const auto* attrs = ev.find("attributes");
            if (!attrs) continue;
            if (const auto* k = attrs->find("kind"); k && k->is_string()) {
                if (k->as_string() == "rerank") saw_rerank = true;
                if (k->as_string() == "score") saw_score = true;
            }
        }
        CHECK(saw_rerank);
        CHECK(saw_score);
    }

    // 2. Mock backend with enable_reranking=false returns ErrorCode::unsupported.
    {
        MockBackendOptions mo;
        mo.enable_reranking = false;
        sonder_test::Harness h(mo);
        auto session = h.session();
        REQUIRE(session);
        CHECK_FALSE(h.engine->find_backend("mock")->capabilities().has(Capability::reranking));
        RerankRequest rreq;
        rreq.query = "q";
        rreq.documents = {"d"};
        CHECK(session->rerank(rreq).status().code() == ErrorCode::unsupported);
        ScoreRequest sreq;
        sreq.pairs = {{"q", "d"}};
        CHECK(session->score(sreq).status().code() == ErrorCode::unsupported);
    }

    // 3. Cancellation and exception recovery leave Session idle and reusable.
    {
        struct FaultRerankBackend final : Backend {
            struct Model final : BackendModel {
                ModelDescriptor desc{"fault:rerank", "fault-rerank", "mock", "mock", "0", "none", 0, 4096, true};
                std::atomic<bool> should_throw{false};
                std::atomic<bool> hold{false};
                std::atomic<bool> entered{false};
                const ModelDescriptor& descriptor() const override { return desc; }
                Result<GenerateStats> generate(const GenerateRequest&, const CancellationToken&,
                                               const TokenCallback&) override {
                    return GenerateStats{};
                }
                Result<RerankResult> rerank(const RerankRequest&, const CancellationToken& cancel) override {
                    entered.store(true);
                    if (should_throw.load()) throw std::runtime_error("CANARY_RERANK_EXCEPTION");
                    while (hold.load() && !cancel.cancelled()) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }
                    if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled");
                    RerankResult out;
                    out.results.push_back(RerankItem{0, 1.0, std::nullopt});
                    return out;
                }
                Result<ScoreResult> score(const ScoreRequest&, const CancellationToken&) override {
                    if (should_throw.load()) throw std::runtime_error("CANARY_SCORE_EXCEPTION");
                    ScoreResult out;
                    out.scores.push_back(ScoreItem{0, 1.0});
                    return out;
                }
            };
            std::shared_ptr<Model> model = std::make_shared<Model>();
            std::string name() const override { return "fault-rerank"; }
            std::string description() const override { return "fault test"; }
            BackendCapabilities capabilities() const override {
                return BackendCapabilities{}.add(Capability::reranking);
            }
            Result<std::string> probe() override { return std::string("fault-1"); }
            Result<std::vector<ModelDescriptor>> list_models() override {
                return std::vector<ModelDescriptor>{model->descriptor()};
            }
            Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions&) override {
                return std::static_pointer_cast<BackendModel>(model);
            }
        };

        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling.enabled = false;
        Engine engine(eo);
        auto be = std::make_shared<FaultRerankBackend>();
        REQUIRE(engine.register_backend(be).ok());
        auto loaded = engine.load_model("fault-rerank", {"fault:rerank", "cpu:0"});
        REQUIRE(loaded.ok());
        auto sess = engine.create_session(loaded.value());
        REQUIRE(sess.ok());

        // Exception in rerank & score recovers SessionState::idle and never leaks raw exception text to telemetry.
        be->model->should_throw.store(true);
        RerankRequest rreq;
        rreq.query = "q";
        rreq.documents = {"d"};
        CHECK_THROWS_AS((void)sess.value()->rerank(rreq), std::runtime_error);
        CHECK(sess.value()->state() == SessionState::idle);
        ScoreRequest sreq;
        sreq.pairs = {{"q", "d"}};
        CHECK_THROWS_AS((void)sess.value()->score(sreq), std::runtime_error);
        CHECK(sess.value()->state() == SessionState::idle);

        // Cooperative cancellation mid-rerank.
        be->model->should_throw.store(false);
        be->model->hold.store(true);
        be->model->entered.store(false);
        Result<RerankResult> cancel_res = Status(ErrorCode::internal, "unset");
        std::thread worker([&] { cancel_res = sess.value()->rerank(rreq); });
        const auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (!be->model->entered.load() && std::chrono::steady_clock::now() < dl) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        sess.value()->cancel();
        worker.join();
        CHECK(cancel_res.status().code() == ErrorCode::cancelled);
        CHECK(sess.value()->state() == SessionState::idle);

        // Usable again afterwards.
        be->model->hold.store(false);
        CHECK(sess.value()->rerank(rreq).ok());
        CHECK(sess.value()->score(sreq).ok());

        engine.telemetry().flush();
        for (const auto& line : sink->lines()) {
            CHECK(line.find("CANARY_RERANK_EXCEPTION") == std::string::npos);
            CHECK(line.find("CANARY_SCORE_EXCEPTION") == std::string::npos);
        }
    }
}

TEST_CASE("session request lifecycle at TelemetryLevel::metrics carries model, requested_model, and kind") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry.level = TelemetryLevel::metrics;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    eo.device_sample_interval = std::chrono::milliseconds(0);
    Engine engine(eo);
    REQUIRE(engine.register_backend(make_mock_backend()).ok());
    auto loaded = engine.load_model(kMockBackendName, {"mock:tiny", "cpu:0"});
    REQUIRE(loaded.ok());
    auto sess = engine.create_session(loaded.value(), SessionOptions{.sampling = SamplingConfig::greedy(3)});
    REQUIRE(sess.ok());

    const auto gen = sess.value()->generate("hello metrics");
    REQUIRE(gen.ok());
    const auto chat_res = sess.value()->chat({{"user", "hello chat metrics"}});
    REQUIRE(chat_res.ok());
    const auto emb = sess.value()->embed({"hello embed"});
    REQUIRE(emb.ok());
    RerankRequest rreq;
    rreq.query = "q";
    rreq.documents = {"d1"};
    const auto rrk = sess.value()->rerank(rreq);
    REQUIRE(rrk.ok());
    ScoreRequest sreq;
    sreq.pairs = {{"q", "d1"}};
    const auto scr = sess.value()->score(sreq);
    REQUIRE(scr.ok());

    engine.telemetry().flush();
    std::vector<json::Value> envs;
    for (const auto& line : sink->lines()) {
        auto parsed = json::parse(line);
        REQUIRE(parsed.ok());
        CHECK(parsed.value().find("event_type")->as_string() != "inference.token.generated");
        envs.push_back(parsed.value());
    }

    auto check_req = [&](const std::string& rid, const std::string& expected_kind) {
        CAPTURE(rid);
        std::vector<std::string> req_types;
        for (const auto& e : envs) {
            const json::Value* erid = e.find("request_id");
            if (erid == nullptr || !erid->is_string() || erid->as_string() != rid) continue;
            const std::string type = e.find("event_type")->as_string();
            if (type.rfind("request.", 0) != 0) continue;
            req_types.push_back(type);
            const json::Value* attrs = e.find("attributes");
            REQUIRE(attrs != nullptr);
            REQUIRE(attrs->find("kind") != nullptr);
            CHECK(attrs->find("kind")->as_string() == expected_kind);
            REQUIRE(attrs->find("model") != nullptr);
            CHECK(attrs->find("model")->as_string() == "mock:tiny");
            REQUIRE(attrs->find("requested_model") != nullptr);
            CHECK(attrs->find("requested_model")->as_string() == "mock:tiny");
        }
        CHECK(req_types == std::vector<std::string>{"request.queued", "request.started", "request.completed"});
    };

    check_req(gen->request_id, "generate");
    check_req(chat_res->request_id, "chat");
    // Also verify embed, rerank, score request.* events carry model and requested_model.
    int non_gen_completed = 0;
    for (const auto& e : envs) {
        const std::string type = e.find("event_type")->as_string();
        if (type != "request.queued" && type != "request.started" && type != "request.completed") continue;
        const json::Value* attrs = e.find("attributes");
        REQUIRE(attrs != nullptr);
        const std::string k = attrs->find("kind") ? attrs->find("kind")->as_string() : std::string{};
        if (k == "embed" || k == "rerank" || k == "score") {
            REQUIRE(attrs->find("model") != nullptr);
            CHECK(attrs->find("model")->as_string() == "mock:tiny");
            REQUIRE(attrs->find("requested_model") != nullptr);
            CHECK(attrs->find("requested_model")->as_string() == "mock:tiny");
            if (type == "request.completed") ++non_gen_completed;
        }
    }
    CHECK(non_gen_completed == 3);
}

}
