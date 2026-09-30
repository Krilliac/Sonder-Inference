// SchedulerMode: per-chunk gating for in-process backends, admission and
// accounting only for remote-process backends (docs/integration/engine-wiring.md).
#include <doctest/doctest.h>

#include <chrono>
#include <memory>
#include <optional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "engine/request_runtime.hpp"
#include "sonder/inference.hpp"

using namespace sonder::inference;

namespace {

std::string words(int n, const std::string& prefix) {
    std::string s;
    for (int i = 0; i < n; ++i) {
        if (i) s += ' ';
        s += prefix + std::to_string(i);
    }
    return s;
}

// Stands in for llamaserver/ollama: another process owns the KV cache and
// batching, so it declares Capability::remote_process.
struct RemoteState {
    std::mutex mu;
    ChatRequest last_chat;
    int chats = 0;
};

class RemoteModel final : public BackendModel {
public:
    RemoteModel(std::shared_ptr<RemoteState> state, std::uint64_t context_length, int chunks)
        : state_(std::move(state)), chunks_(chunks) {
        descriptor_.name = "remote-model";
        descriptor_.backend = "remote";
        descriptor_.format = "test";
        descriptor_.context_length = context_length;
    }
    const ModelDescriptor& descriptor() const override { return descriptor_; }
    bool has_native_chat() const override { return true; }
    Result<GenerateStats> generate(const GenerateRequest&, const CancellationToken& cancel,
                                  const TokenCallback& on_chunk) override {
        return stream(cancel, on_chunk);
    }
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            state_->last_chat = request;
            ++state_->chats;
        }
        return stream(cancel, on_chunk);
    }

private:
    Result<GenerateStats> stream(const CancellationToken& cancel, const TokenCallback& on_chunk) {
        GenerateStats stats;
        for (int i = 0; i < chunks_; ++i) {
            if (cancel.cancelled()) {
                return Status(ErrorCode::cancelled, "cancelled");
            }
            const std::string piece = " w" + std::to_string(i);
            if (on_chunk && !on_chunk(TokenChunk{piece, stats.chunks})) {
                stats.stop_reason = StopReason::callback;
                return stats;
            }
            ++stats.chunks;
            ++stats.completion_tokens;
        }
        stats.stop_reason = StopReason::end_of_sequence;
        return stats;
    }

    std::shared_ptr<RemoteState> state_;
    ModelDescriptor descriptor_;
    int chunks_;
};

class RemoteBackend final : public Backend {
public:
    RemoteBackend(std::shared_ptr<RemoteState> state, std::uint64_t context_length, int chunks)
        : state_(std::move(state)), context_length_(context_length), chunks_(chunks) {}
    std::string name() const override { return "remote"; }
    std::string description() const override { return "test remote-process backend"; }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::streaming).add(Capability::remote_process);
        return c;
    }
    Result<std::string> probe() override { return std::string("test"); }
    Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions&) override {
        return std::shared_ptr<BackendModel>(std::make_shared<RemoteModel>(state_, context_length_, chunks_));
    }

private:
    std::shared_ptr<RemoteState> state_;
    std::uint64_t context_length_;
    int chunks_;
};

struct RemoteRig {
    std::shared_ptr<MemoryTelemetrySink> sink = std::make_shared<MemoryTelemetrySink>();
    std::shared_ptr<RemoteState> state = std::make_shared<RemoteState>();
    std::unique_ptr<Engine> engine;
    std::shared_ptr<Model> model;

    RemoteRig(SchedulingOptions sched, std::uint64_t context_length, int chunks) {
        EngineOptions eo;
        eo.telemetry.level = TelemetryLevel::standard;
        eo.telemetry.queue_capacity = 65536;  // count every scheduler step
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling = sched;
        eo.sample_devices_on_start = false;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        engine = std::make_unique<Engine>(std::move(eo));
        REQUIRE(engine->register_backend(std::make_shared<RemoteBackend>(state, context_length, chunks)).ok());
        auto loaded = engine->load_model("remote", ModelLoadOptions{"remote-model", "cpu:0"});
        REQUIRE(loaded.ok());
        model = loaded.value();
    }

    std::shared_ptr<Session> session() {
        SessionOptions so;
        so.sampling = SamplingConfig::greedy(4096);
        auto s = engine->create_session(model, so);
        REQUIRE(s.ok());
        return s.value();
    }

    std::vector<json::Value> events_of(const std::string& type) {
        engine->telemetry().flush();
        std::vector<json::Value> out;
        for (const auto& line : sink->lines()) {
            auto v = json::parse(line);
            if (v.ok() && v.value().find("event_type")->as_string() == type) {
                out.push_back(v.value());
            }
        }
        return out;
    }

    std::string started_mode() {
        const auto started = events_of("request.started");
        REQUIRE(started.size() == 1);
        const auto* mode = started[0].find("attributes")->find("scheduler_mode");
        return mode ? mode->as_string() : std::string("(none)");
    }
};

}  // namespace

TEST_CASE("remote-process backends stream without per-chunk scheduler grants by default") {
    constexpr int kChunks = 200;
    RemoteRig rig({}, 0, kChunks);
    REQUIRE(rig.engine->scheduling_active());
    auto r = rig.session()->chat({{"user", "hello there"}});
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    CHECK(r.value().stats.chunks == kChunks);
    CHECK(r.value().scheduling.scheduled);  // admitted and accounted
    CHECK(rig.started_mode() == "account");
    // One logical prefill step (admission), no decode step per chunk.
    CHECK(rig.events_of("scheduler.batch.formed").size() <= 2);
    CHECK(rig.engine->request_runtime()->tracked_requests() == 0);
    CHECK(rig.engine->kv_usage().pinned_blocks == 0);

    // The same backend under --scheduler gate: one grant (step) per chunk.
    SchedulingOptions gate;
    gate.mode = SchedulerMode::gate;
    RemoteRig gated(gate, 0, kChunks);
    auto g = gated.session()->chat({{"user", "hello there"}});
    REQUIRE_MESSAGE(g.ok(), g.status().to_string());
    CHECK(g.value().stats.chunks == kChunks);
    CHECK(gated.started_mode() == "gate");
    CHECK(gated.events_of("scheduler.batch.formed").size() >= static_cast<std::size_t>(kChunks));
}

TEST_CASE("a prompt beyond the logical KV pool is admitted on a remote backend whose context allows it") {
    const std::string prompt = words(70000, "t");  // > 65,536 pool tokens
    constexpr std::uint64_t kServedCtx = 262144;

    RemoteRig rig({}, kServedCtx, 3);
    auto r = rig.session()->generate(prompt);
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    CHECK(r.value().outcome == RequestOutcome::completed);
    CHECK_FALSE(r.value().scheduling.scheduled);
    CHECK(rig.events_of("scheduler.bypassed").size() == 1);
    CHECK(rig.events_of("scheduler.rejected").empty());

    // A pool sized for it (--kv-pool-tokens 131072) admits and accounts it.
    SchedulingOptions big;
    big.kv_num_blocks = 131072 / 16;
    RemoteRig sized(big, kServedCtx, 3);
    auto s = sized.session()->generate(prompt);
    REQUIRE_MESSAGE(s.ok(), s.status().to_string());
    CHECK(s.value().scheduling.scheduled);
    CHECK(s.value().scheduling.accounted_prompt_tokens == 70001);
    CHECK(sized.events_of("scheduler.bypassed").empty());

    // Explicit per-token gating keeps the old refusal.
    SchedulingOptions gate;
    gate.mode = SchedulerMode::gate;
    RemoteRig gated(gate, kServedCtx, 3);
    auto g = gated.session()->generate(prompt);
    REQUIRE_FALSE(g.ok());
    CHECK(g.status().code() == ErrorCode::invalid_argument);
    CHECK(gated.events_of("scheduler.rejected").size() == 1);
}

TEST_CASE("in-process backends keep per-token gating under the automatic mode") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    Engine engine(std::move(eo));
    REQUIRE(engine.register_backend(make_mock_backend()).ok());
    auto model = engine.load_model(kMockBackendName, ModelLoadOptions{"mock:tiny", "cpu:0"});
    REQUIRE(model.ok());
    SessionOptions so;
    so.sampling = SamplingConfig::greedy(6);
    auto session = engine.create_session(model.value(), so);
    REQUIRE(session.ok());
    auto r = session.value()->generate("hello world");
    REQUIRE(r.ok());
    CHECK(r.value().scheduling.scheduled);
    engine.telemetry().flush();
    bool seen = false;
    for (const auto& line : sink->lines()) {
        auto v = json::parse(line);
        REQUIRE(v.ok());
        if (v.value().find("event_type")->as_string() == "request.started") {
            CHECK(v.value().find("attributes")->find("scheduler_mode")->as_string() == "gate");
            seen = true;
        }
    }
    CHECK(seen);
}

TEST_CASE("chat request options reach a native-chat backend") {
    RemoteRig rig({}, 0, 2);
    RequestOptions ro;
    ro.session_key = "run=r1;agent=a1";
    ro.thinking.enable_thinking = false;
    ro.thinking.reasoning_effort = "low";
    auto r = rig.session()->chat({{"user", "hi"}}, {}, std::nullopt, ro);
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    std::lock_guard<std::mutex> lock(rig.state->mu);
    CHECK(rig.state->chats == 1);
    CHECK(rig.state->last_chat.session_key == "run=r1;agent=a1");
    REQUIRE(rig.state->last_chat.thinking.enable_thinking.has_value());
    CHECK_FALSE(*rig.state->last_chat.thinking.enable_thinking);
    CHECK(rig.state->last_chat.thinking.reasoning_effort == std::optional<std::string>("low"));
}
