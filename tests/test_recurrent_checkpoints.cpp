// Tests for backend recurrent-state checkpoints, bounded runtime blob store,
// and atomic prefill-boundary Session capture/restore wiring across hybrid and
// recurrent architectures.
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <vector>

#include "engine/request_runtime.hpp"
#include "sonder/inference.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

static_assert(kMaxRecurrentCheckpointBytes == 512ULL * 1024ULL * 1024ULL);

namespace {

// Minimal backend model that inherits BackendModel's default checkpoint methods.
class PlainBackendModel final : public BackendModel {
public:
    PlainBackendModel() {
        desc_.name = "plain:tiny";
        desc_.backend = "plain";
        desc_.format = "test";
        desc_.architecture = ModelArchitecture::hybrid;
    }

    const ModelDescriptor& descriptor() const override { return desc_; }

    Result<GenerateStats> generate(const GenerateRequest&, const CancellationToken&,
                                   const TokenCallback&) override {
        GenerateStats stats;
        stats.stop_reason = StopReason::end_of_sequence;
        return stats;
    }

private:
    ModelDescriptor desc_;
};

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

std::string make_words(int count, const std::string& prefix) {
    std::string out;
    for (int i = 0; i < count; ++i) {
        if (i > 0) {
            out += ' ';
        }
        out += prefix + std::to_string(i);
    }
    return out;
}

template <typename Pred>
bool wait_for(Pred pred, std::chrono::milliseconds timeout = 5000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) {
            return false;
        }
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

std::vector<json::Value> events_of(Engine& engine, const std::shared_ptr<MemoryTelemetrySink>& sink,
                                   const std::string& type) {
    engine.telemetry().flush();
    std::vector<json::Value> out;
    for (const auto& line : sink->lines()) {
        auto parsed = json::parse(line);
        if (parsed.ok()) {
            const auto* ev_type = parsed.value().find("event_type");
            if (ev_type != nullptr && ev_type->as_string() == type) {
                out.push_back(parsed.value());
            }
        }
    }
    return out;
}

std::string attr_str(const json::Value& ev, const char* key) {
    const auto* attrs = ev.find("attributes");
    const auto* val = attrs != nullptr ? attrs->find(key) : nullptr;
    return val != nullptr ? val->as_string() : std::string();
}

std::int64_t attr_int(const json::Value& ev, const char* key) {
    const auto* attrs = ev.find("attributes");
    const auto* val = attrs != nullptr ? attrs->find(key) : nullptr;
    return val != nullptr ? val->as_int(-1) : -1;
}

// Spy wrapper over MockBackend to inspect atomic checkpoint exchange activity.
struct CheckpointSpyState {
    std::atomic<int> capture_calls{0};
    std::atomic<int> restore_calls{0};
    std::mutex mutex;
    std::vector<std::uint64_t> captured_positions;
    std::vector<std::vector<std::uint8_t>> captured_blobs;
    std::vector<std::vector<std::uint8_t>> restored_blobs;
};

class SpyCheckpointModel final : public BackendModel {
public:
    SpyCheckpointModel(std::shared_ptr<BackendModel> inner, std::shared_ptr<CheckpointSpyState> spy,
                       bool native_chat = false)
        : inner_(std::move(inner)), spy_(std::move(spy)), native_chat_(native_chat) {}

    const ModelDescriptor& descriptor() const override { return inner_->descriptor(); }

    bool has_native_chat() const override { return native_chat_; }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        if (request.checkpoint == nullptr) {
            return inner_->generate(request, cancel, on_chunk);
        }
        GenerateRequest wrapped = request;
        RecurrentCheckpointExchange* orig = request.checkpoint;
        RecurrentCheckpointExchange proxy;
        proxy.restore_bytes = orig->restore_bytes;
        proxy.restore_token_position = orig->restore_token_position;
        proxy.capture_at_token_position = orig->capture_at_token_position;
        proxy.on_restored = [this, orig] {
            ++spy_->restore_calls;
            {
                std::lock_guard<std::mutex> lock(spy_->mutex);
                spy_->restored_blobs.emplace_back(orig->restore_bytes.begin(), orig->restore_bytes.end());
            }
            orig->restored = true;
            if (orig->on_restored) {
                orig->on_restored();
            }
        };
        proxy.on_captured = [this, orig](std::uint64_t pos, std::vector<std::uint8_t> bytes) {
            ++spy_->capture_calls;
            {
                std::lock_guard<std::mutex> lock(spy_->mutex);
                spy_->captured_positions.push_back(pos);
                spy_->captured_blobs.push_back(bytes);
            }
            orig->captured = true;
            if (orig->on_captured) {
                orig->on_captured(pos, std::move(bytes));
            } else {
                orig->captured_bytes = std::move(bytes);
            }
        };
        wrapped.checkpoint = &proxy;
        auto res = inner_->generate(wrapped, cancel, on_chunk);
        orig->restored = proxy.restored;
        orig->restore_error = proxy.restore_error;
        orig->captured = proxy.captured;
        orig->capture_error = proxy.capture_error;
        return res;
    }

    Result<std::vector<TokenId>> tokenize(std::string_view text) override { return inner_->tokenize(text); }

    Result<std::unique_ptr<TokenStream>> open_token_stream(const GenerateRequest& request) override {
        return inner_->open_token_stream(request);
    }

    Result<RecurrentStateSnapshot> capture_recurrent_state(std::uint64_t token_position) override {
        return inner_->capture_recurrent_state(token_position);
    }

    Status restore_recurrent_state(std::span<const std::uint8_t> bytes) override {
        return inner_->restore_recurrent_state(bytes);
    }

private:
    std::shared_ptr<BackendModel> inner_;
    std::shared_ptr<CheckpointSpyState> spy_;
    bool native_chat_ = false;
};

class SpyCheckpointBackend final : public Backend {
public:
    SpyCheckpointBackend(std::shared_ptr<Backend> inner, std::shared_ptr<CheckpointSpyState> spy,
                         bool native_chat = false)
        : inner_(std::move(inner)), spy_(std::move(spy)), native_chat_(native_chat) {}

    std::string name() const override { return "spy-mock"; }
    std::string description() const override { return "test spy backend for recurrent checkpoints"; }
    BackendCapabilities capabilities() const override { return inner_->capabilities(); }
    Result<std::string> probe() override { return inner_->probe(); }
    Result<std::vector<ModelDescriptor>> list_models() override { return inner_->list_models(); }

    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        auto loaded = inner_->load_model(options);
        if (!loaded.ok()) {
            return loaded.status();
        }
        return std::shared_ptr<BackendModel>(
            std::make_shared<SpyCheckpointModel>(std::move(loaded).value(), spy_, native_chat_));
    }

private:
    std::shared_ptr<Backend> inner_;
    std::shared_ptr<CheckpointSpyState> spy_;
    bool native_chat_ = false;
};

#endif  // SONDER_HAS_KV_CACHE && SONDER_HAS_SCHEDULER

}  // namespace

TEST_SUITE("recurrent_checkpoints") {

TEST_CASE("default BackendModel returns unsupported and capability name maps") {
    PlainBackendModel plain;
    const auto cap_res = plain.capture_recurrent_state(16);
    CHECK_FALSE(cap_res.ok());
    CHECK(cap_res.status().code() == ErrorCode::unsupported);

    const std::vector<std::uint8_t> dummy(64, 0xAB);
    const Status rst_res = plain.restore_recurrent_state(dummy);
    CHECK_FALSE(rst_res.ok());
    CHECK(rst_res.code() == ErrorCode::unsupported);

    BackendCapabilities caps;
    caps.add(Capability::recurrent_checkpoint);
    CHECK(caps.has(Capability::recurrent_checkpoint));
    CHECK(static_cast<std::uint32_t>(Capability::recurrent_checkpoint) == (1u << 17));
    const auto names = caps.names();
    REQUIRE(names.size() == 1u);
    CHECK(names[0] == "recurrent_checkpoint");
    CHECK(kMaxRecurrentCheckpointBytes == (std::size_t{512} << 20));
}

TEST_CASE("mock backend preserves default capability list when recurrent checkpoints disabled") {
    auto plain_mock = make_mock_backend();
    const auto caps = plain_mock->capabilities();
    CHECK_FALSE(caps.has(Capability::recurrent_checkpoint));
    const std::vector<std::string> expected_names{
        "tokenization", "streaming", "embeddings", "deterministic", "tool_calling", "multimodal", "logprobs",
        "reranking"};
    CHECK(caps.names() == expected_names);

    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto loaded = plain_mock->load_model(lo);
    REQUIRE(loaded.ok());
    CHECK(loaded.value()->descriptor().architecture == ModelArchitecture::attention_only);
    CHECK(loaded.value()->capture_recurrent_state(8).status().code() == ErrorCode::unsupported);
    const std::vector<std::uint8_t> sample(64, 0x11);
    CHECK(loaded.value()->restore_recurrent_state(sample).code() == ErrorCode::unsupported);

    MockBackendOptions hybrid_opts;
    hybrid_opts.enable_recurrent_checkpoints = true;
    hybrid_opts.default_architecture = ModelArchitecture::hybrid;
    auto hybrid_mock = make_mock_backend(hybrid_opts);
    const auto hybrid_caps = hybrid_mock->capabilities();
    CHECK(hybrid_caps.has(Capability::recurrent_checkpoint));
    const auto hybrid_names = hybrid_caps.names();
    CHECK(std::find(hybrid_names.begin(), hybrid_names.end(), "recurrent_checkpoint") != hybrid_names.end());

    auto hybrid_model = hybrid_mock->load_model(lo);
    REQUIRE(hybrid_model.ok());
    CHECK(hybrid_model.value()->descriptor().architecture == ModelArchitecture::hybrid);

    auto snap1 = hybrid_model.value()->capture_recurrent_state(16);
    auto snap2 = hybrid_model.value()->capture_recurrent_state(16);
    REQUIRE(snap1.ok());
    REQUIRE(snap2.ok());
    CHECK(snap1.value().token_position == 16u);
    CHECK_FALSE(snap1.value().bytes.empty());
    CHECK(snap1.value().bytes == snap2.value().bytes);
    CHECK(hybrid_model.value()->restore_recurrent_state(snap1.value().bytes).ok());

    auto tampered = snap1.value().bytes;
    tampered[0] ^= 0xFFu;
    CHECK(hybrid_model.value()->restore_recurrent_state(tampered).code() == ErrorCode::invalid_argument);
}

TEST_CASE("oversized and empty recurrent checkpoint snapshots are rejected") {
    // Use a bounded MockBackendOptions::max_checkpoint_bytes cap so the test
    // exercises oversized-snapshot rejection with tiny buffers (no 512 MiB alloc).
    MockBackendOptions opts;
    opts.enable_recurrent_checkpoints = true;
    opts.default_architecture = ModelArchitecture::hybrid;
    opts.max_checkpoint_bytes = 64;
    auto backend = make_mock_backend(opts);
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto loaded = backend->load_model(lo);
    REQUIRE(loaded.ok());
    auto& model = *loaded.value();

    CHECK(model.capture_recurrent_state(0).status().code() == ErrorCode::invalid_argument);
    CHECK(model.restore_recurrent_state({}).code() == ErrorCode::invalid_argument);

    const std::vector<std::uint8_t> oversized(128, 0x5A);
    const Status rst_over = model.restore_recurrent_state(oversized);
    CHECK(rst_over.code() == ErrorCode::invalid_argument);
    CHECK(rst_over.message().find("max_checkpoint_bytes") != std::string::npos);

    // A cap smaller than the 64-byte mock snapshot rejects capture too.
    MockBackendOptions tiny_cap = opts;
    tiny_cap.max_checkpoint_bytes = 32;
    auto tiny_backend = make_mock_backend(tiny_cap);
    auto tiny_model = tiny_backend->load_model(lo);
    REQUIRE(tiny_model.ok());
    CHECK(tiny_model.value()->capture_recurrent_state(8).status().code() == ErrorCode::invalid_argument);

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)
    TelemetryOptions to;
    to.level = TelemetryLevel::off;
    TelemetryBus bus(to);
    TelemetryContext ctx;
    ctx.session_id = "test-oversized";
    ctx.run_id = "test-oversized";

    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    sched.kv_num_blocks = 16;
    sched.mode = SchedulerMode::account;
    detail::CheckpointStoreOptions cp_opts;
    cp_opts.max_bytes = 256;
    cp_opts.max_checkpoints = 16;

    auto runtime = detail::make_request_runtime(sched, bus, ctx, cp_opts);
    REQUIRE(runtime != nullptr);

    detail::RuntimeRequestSpec spec;
    spec.context = ctx;
    spec.prompt_tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    spec.exact_tokens = true;
    spec.max_new_tokens = 2;
    spec.fingerprint = 42;
    spec.architecture = ModelArchitecture::hybrid;
    spec.gated = false;
    spec.retain_for_checkpoint = true;

    auto sub = runtime->submit(std::move(spec));
    REQUIRE(sub.ok());
    CancellationToken cancel;
    REQUIRE(runtime->acquire_token(sub.value().id, cancel).ok());

    CHECK(runtime->register_checkpoint(sub.value().id, 8, {}).code() == ErrorCode::invalid_argument);
    CHECK(runtime->register_checkpoint(sub.value().id, 0, std::vector<std::uint8_t>(64, 1)).code() ==
          ErrorCode::invalid_argument);
    CHECK(runtime->register_checkpoint(sub.value().id, 8, std::vector<std::uint8_t>(300, 1)).code() ==
          ErrorCode::invalid_argument);
    CHECK(runtime->checkpoint_store_usage().entries == 0u);
    CHECK(runtime->checkpoint_store_usage().total_bytes == 0u);

    (void)runtime->finish(sub.value().id, RequestOutcome::completed);
    CHECK(wait_for([&] { return runtime->tracked_requests() == 0u; }));
    runtime.reset();
    bus.shutdown();
#endif
}

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

TEST_CASE("mock hybrid end-to-end capture restores on matching prefix and skips divergent prefix") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry.level = TelemetryLevel::standard;
    eo.telemetry_sinks.push_back(sink);
    eo.scheduling.kv_block_size_tokens = 4;
    eo.scheduling.kv_num_blocks = 64;
    eo.scheduling.prefix_caching = true;
    eo.sample_devices_on_start = false;
    eo.device_sample_interval = std::chrono::milliseconds{0};

    Engine engine(std::move(eo));
    MockBackendOptions mock_opts;
    mock_opts.enable_recurrent_checkpoints = true;
    mock_opts.default_architecture = ModelArchitecture::hybrid;
    mock_opts.default_completion_tokens = 8;
    REQUIRE(engine.register_backend(make_mock_backend(mock_opts)).ok());

    ModelLoadOptions lo;
    lo.model = "mock:hybrid";
    auto model = engine.load_model(kMockBackendName, lo);
    REQUIRE(model.ok());
    CHECK(model.value()->descriptor().architecture == ModelArchitecture::hybrid);

    SessionOptions so;
    so.sampling = SamplingConfig::greedy(4);
    auto session = engine.create_session(model.value(), so);
    REQUIRE(session.ok());

    // 15 words + leading BOS = 16 exact tokens (4 full blocks of 4).
    const std::string shared_prefix = make_words(15, "hybrid");
    auto first = session.value()->generate(shared_prefix);
    REQUIRE(first.ok());
    CHECK(first.value().outcome == RequestOutcome::completed);
    CHECK(first.value().scheduling.scheduled);
    CHECK(first.value().scheduling.accounted_prompt_tokens == 16u);
    CHECK(first.value().scheduling.reused_prompt_tokens == 0u);

    auto saved_after_first = events_of(engine, sink, "kv.checkpoint.saved");
    REQUIRE(saved_after_first.size() == 1u);
    CHECK(attr_int(saved_after_first[0], "token_position") == 16);
    CHECK(attr_int(saved_after_first[0], "bytes") == 64);
    CHECK(attr_str(saved_after_first[0], "architecture") == "hybrid");
    CHECK(events_of(engine, sink, "kv.checkpoint.restored").empty());
    REQUIRE(engine.request_runtime() != nullptr);
    CHECK(engine.request_runtime()->checkpoint_store_usage().entries == 1u);
    CHECK(engine.request_runtime()->checkpoint_store_usage().total_bytes == 64u);

    // Extended prompt sharing the exact 16-token prefix restores checkpoint@16
    // and captures a new checkpoint at block-aligned boundary 20.
    const std::string extended = shared_prefix + " alpha beta gamma delta";
    auto second = session.value()->generate(extended);
    REQUIRE(second.ok());
    CHECK(second.value().outcome == RequestOutcome::completed);
    CHECK(second.value().scheduling.accounted_prompt_tokens == 20u);
    CHECK(second.value().scheduling.reused_prompt_tokens == 16u);

    auto restored_after_second = events_of(engine, sink, "kv.checkpoint.restored");
    REQUIRE(restored_after_second.size() == 1u);
    CHECK(attr_int(restored_after_second[0], "token_position") == 16);
    CHECK(attr_int(restored_after_second[0], "bytes") == 64);
    CHECK(attr_str(restored_after_second[0], "architecture") == "hybrid");

    // Exact repeat of shared_prefix restores checkpoint@16, preserves output parity,
    // and avoids redundant duplicate save at position 16.
    auto repeat = session.value()->generate(shared_prefix);
    REQUIRE(repeat.ok());
    CHECK(repeat.value().text == first.value().text);
    CHECK(repeat.value().scheduling.reused_prompt_tokens == 16u);
    CHECK(events_of(engine, sink, "kv.checkpoint.restored").size() == 2u);
    CHECK(events_of(engine, sink, "kv.checkpoint.saved").size() == 2u);

    // Divergent prefix must NOT restore any saved checkpoint.
    const std::string divergent = make_words(15, "divergent");
    auto div_res = session.value()->generate(divergent);
    REQUIRE(div_res.ok());
    CHECK(div_res.value().scheduling.reused_prompt_tokens == 0u);
    CHECK(events_of(engine, sink, "kv.checkpoint.restored").size() == 2u);

    // Shorter prefix (8 tokens < saved checkpoint at 16) must NOT restore checkpoint@16.
    const std::string shorter = make_words(7, "hybrid");
    auto short_res = session.value()->generate(shorter);
    REQUIRE(short_res.ok());
    CHECK(short_res.value().scheduling.reused_prompt_tokens == 0u);
    CHECK(events_of(engine, sink, "kv.checkpoint.restored").size() == 2u);

    CHECK(engine.kv_usage().sequences == 0u);
    CHECK(engine.kv_usage().pinned_blocks == 0u);
    CHECK(wait_for([&] { return engine.request_runtime()->tracked_requests() == 0u; }));
}

TEST_CASE("non-block-aligned prompt captures at aligned boundary and restores for extended prefix") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry.level = TelemetryLevel::metrics;
    eo.telemetry_sinks.push_back(sink);
    eo.scheduling.kv_block_size_tokens = 4;
    eo.scheduling.kv_num_blocks = 64;
    eo.scheduling.prefix_caching = true;
    eo.sample_devices_on_start = false;
    eo.device_sample_interval = std::chrono::milliseconds{0};

    Engine engine(std::move(eo));
    auto spy = std::make_shared<CheckpointSpyState>();
    MockBackendOptions mock_opts;
    mock_opts.enable_recurrent_checkpoints = true;
    mock_opts.default_architecture = ModelArchitecture::hybrid;
    REQUIRE(engine.register_backend(std::make_shared<SpyCheckpointBackend>(make_mock_backend(mock_opts), spy)).ok());

    ModelLoadOptions lo;
    lo.model = "mock:hybrid-aligned";
    auto model = engine.load_model("spy-mock", lo);
    REQUIRE(model.ok());

    SessionOptions so;
    so.sampling = SamplingConfig::greedy(2);
    auto session = engine.create_session(model.value(), so);
    REQUIRE(session.ok());

    // 9 words + leading BOS = 10 tokens (not a multiple of block_size=4).
    // Capture must occur at block-aligned boundary 8 so every block is full and
    // survives finish().
    const std::string prompt_10 = make_words(9, "align");
    auto r1 = session.value()->generate(prompt_10);
    REQUIRE(r1.ok());
    CHECK(r1.value().scheduling.accounted_prompt_tokens == 10u);
    CHECK(r1.value().scheduling.reused_prompt_tokens == 0u);
    CHECK(spy->capture_calls.load() == 1);
    CHECK(spy->restore_calls.load() == 0);
    REQUIRE(engine.request_runtime() != nullptr);
    CHECK(engine.request_runtime()->checkpoint_store_usage().entries == 1u);
    CHECK(engine.request_runtime()->checkpoint_store_usage().evictions == 0u);
    auto saved_1 = events_of(engine, sink, "kv.checkpoint.saved");
    REQUIRE(saved_1.size() == 1u);
    CHECK(attr_int(saved_1[0], "token_position") == 8);

    // Second prompt: 11 tokens sharing the exact first 8 tokens (BOS + align0..align6)
    // followed by 3 different tail tokens. Must restore the exact checkpoint from 8
    // and skip redundant re-capture at 8.
    const std::string prompt_11 = make_words(7, "align") + " tailA tailB tailC";
    auto r2 = session.value()->generate(prompt_11);
    REQUIRE(r2.ok());
    CHECK(r2.value().scheduling.accounted_prompt_tokens == 11u);
    CHECK(r2.value().scheduling.reused_prompt_tokens == 8u);
    CHECK(spy->restore_calls.load() == 1);
    CHECK(spy->capture_calls.load() == 1);
    {
        std::lock_guard<std::mutex> lock(spy->mutex);
        REQUIRE(spy->captured_positions.size() == 1u);
        CHECK(spy->captured_positions[0] == 8u);
        REQUIRE(spy->restored_blobs.size() == 1u);
        CHECK(spy->restored_blobs[0] == spy->captured_blobs[0]);
    }
    auto restored_events = events_of(engine, sink, "kv.checkpoint.restored");
    REQUIRE(restored_events.size() == 1u);
    CHECK(attr_int(restored_events[0], "token_position") == 8);

    // Third prompt: 13 tokens sharing the first 8 tokens -> restores 8 and
    // captures a new aligned checkpoint at 12.
    const std::string prompt_13 = make_words(7, "align") + " b0 b1 b2 b3 b4";
    auto r3 = session.value()->generate(prompt_13);
    REQUIRE(r3.ok());
    CHECK(r3.value().scheduling.accounted_prompt_tokens == 13u);
    CHECK(r3.value().scheduling.reused_prompt_tokens == 8u);
    CHECK(spy->restore_calls.load() == 2);
    CHECK(spy->capture_calls.load() == 2);
    {
        std::lock_guard<std::mutex> lock(spy->mutex);
        REQUIRE(spy->captured_positions.size() == 2u);
        CHECK(spy->captured_positions[1] == 12u);
    }

    CHECK(engine.kv_usage().sequences == 0u);
    CHECK(engine.kv_usage().pinned_blocks == 0u);
    CHECK(wait_for([&] { return engine.request_runtime()->tracked_requests() == 0u; }));
}

TEST_CASE("recurrent architecture restores exact captured bytes and releases account retention before decode") {
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions eo;
    eo.telemetry.level = TelemetryLevel::metrics;
    eo.telemetry_sinks.push_back(sink);
    eo.scheduling.mode = SchedulerMode::account;
    eo.scheduling.kv_block_size_tokens = 4;
    eo.scheduling.kv_num_blocks = 64;
    eo.scheduling.prefix_caching = true;
    eo.sample_devices_on_start = false;
    eo.device_sample_interval = std::chrono::milliseconds{0};

    Engine engine(std::move(eo));
    auto spy = std::make_shared<CheckpointSpyState>();
    MockBackendOptions mock_opts;
    mock_opts.enable_recurrent_checkpoints = true;
    mock_opts.default_architecture = ModelArchitecture::recurrent;
    auto spy_backend = std::make_shared<SpyCheckpointBackend>(make_mock_backend(mock_opts), spy);
    REQUIRE(engine.register_backend(spy_backend).ok());

    ModelLoadOptions lo;
    lo.model = "mock:mamba";
    auto model = engine.load_model("spy-mock", lo);
    REQUIRE(model.ok());
    CHECK(model.value()->descriptor().architecture == ModelArchitecture::recurrent);

    SessionOptions so;
    so.sampling = SamplingConfig::greedy(4);
    auto session = engine.create_session(model.value(), so);
    REQUIRE(session.ok());

    // In account mode, once prefill capture registers, the retained sequence is
    // released immediately before decode emits its first chunk (zero blocks pinned).
    bool saw_chunk_with_released_sequence = false;
    const std::string prefix = make_words(15, "ssm");
    auto r1 = session.value()->generate(prefix, [&](const TokenChunk&) {
        const auto usage = engine.kv_usage();
        saw_chunk_with_released_sequence = saw_chunk_with_released_sequence ||
                                           (usage.sequences == 0u && usage.pinned_blocks == 0u);
        return true;
    });
    REQUIRE(r1.ok());
    CHECK(saw_chunk_with_released_sequence);
    CHECK(spy->capture_calls.load() == 1);
    CHECK(spy->restore_calls.load() == 0);

    auto r2 = session.value()->generate(prefix + " tail1 tail2 tail3 tail4");
    REQUIRE(r2.ok());
    CHECK(r2.value().scheduling.reused_prompt_tokens == 16u);
    CHECK(spy->restore_calls.load() == 1);
    CHECK(spy->capture_calls.load() == 2);
    {
        std::lock_guard<std::mutex> lock(spy->mutex);
        REQUIRE(spy->captured_blobs.size() == 2u);
        REQUIRE(spy->restored_blobs.size() == 1u);
        CHECK(spy->captured_positions[0] == 16u);
        CHECK(spy->captured_positions[1] == 20u);
        CHECK(spy->restored_blobs[0] == spy->captured_blobs[0]);
    }

    auto r3 = session.value()->generate(make_words(15, "unrelated"));
    REQUIRE(r3.ok());
    CHECK(r3.value().scheduling.reused_prompt_tokens == 0u);
    CHECK(spy->restore_calls.load() == 1);

    auto restored_events = events_of(engine, sink, "kv.checkpoint.restored");
    REQUIRE(restored_events.size() == 1u);
    CHECK(attr_str(restored_events[0], "architecture") == "recurrent");
    CHECK(engine.kv_usage().sequences == 0u);
    CHECK(engine.kv_usage().pinned_blocks == 0u);
    CHECK(wait_for([&] { return engine.request_runtime()->tracked_requests() == 0u; }));
}

TEST_CASE("bounded checkpoint store evicts LRU and block-invalidated blobs without leaks") {
    // Part A: strict byte budget eviction on RequestRuntime directly.
    {
        TelemetryOptions to;
        to.level = TelemetryLevel::off;
        TelemetryBus bus(to);
        TelemetryContext ctx;
        ctx.session_id = "test-bounded";
        ctx.run_id = "test-bounded";

        SchedulingOptions sched;
        sched.kv_block_size_tokens = 4;
        sched.kv_num_blocks = 64;
        sched.prefix_caching = true;
        sched.mode = SchedulerMode::account;
        detail::CheckpointStoreOptions cp_opts;
        cp_opts.max_bytes = 256;  // Fits at most 2 x 100-byte checkpoints.
        cp_opts.max_checkpoints = 16;

        auto runtime = detail::make_request_runtime(sched, bus, ctx, cp_opts);
        REQUIRE(runtime != nullptr);

        auto make_prompt = [](TokenId base) {
            std::vector<TokenId> toks(8);
            for (std::size_t i = 0; i < toks.size(); ++i) {
                toks[i] = base + static_cast<TokenId>(i);
            }
            return toks;
        };

        CancellationToken cancel;
        for (int i = 0; i < 4; ++i) {
            detail::RuntimeRequestSpec spec;
            spec.context = ctx;
            spec.prompt_tokens = make_prompt(100 * (i + 1));
            spec.exact_tokens = true;
            spec.max_new_tokens = 2;
            spec.fingerprint = 99;
            spec.architecture = ModelArchitecture::hybrid;
            spec.gated = false;
            spec.retain_for_checkpoint = true;

            auto sub = runtime->submit(std::move(spec));
            REQUIRE(sub.ok());
            REQUIRE(runtime->acquire_token(sub.value().id, cancel).ok());

            std::vector<std::uint8_t> blob(100, static_cast<std::uint8_t>(i + 1));
            REQUIRE(runtime->register_checkpoint(sub.value().id, 8, std::move(blob)).ok());
            (void)runtime->finish(sub.value().id, RequestOutcome::completed);

            const auto usage = runtime->checkpoint_store_usage();
            CHECK(usage.total_bytes <= 256u);
            CHECK(usage.entries <= 2u);
        }

        const auto after_fill = runtime->checkpoint_store_usage();
        CHECK(after_fill.entries == 2u);
        CHECK(after_fill.total_bytes == 200u);
        CHECK(after_fill.evictions == 2u);

        // Prompt 0 was LRU-evicted and must not restore.
        {
            detail::RuntimeRequestSpec probe;
            probe.context = ctx;
            probe.prompt_tokens = make_prompt(100);
            probe.exact_tokens = true;
            probe.max_new_tokens = 2;
            probe.fingerprint = 99;
            probe.architecture = ModelArchitecture::hybrid;
            probe.gated = false;
            auto sub = runtime->submit(std::move(probe));
            REQUIRE(sub.ok());
            REQUIRE(runtime->acquire_token(sub.value().id, cancel).ok());
            CHECK_FALSE(runtime->lookup_restored_checkpoint(sub.value().id).has_value());
            (void)runtime->finish(sub.value().id, RequestOutcome::completed);
        }

        // Prompt 3 is still retained and restores its exact 100-byte blob.
        {
            detail::RuntimeRequestSpec probe;
            probe.context = ctx;
            probe.prompt_tokens = make_prompt(400);
            probe.exact_tokens = true;
            probe.max_new_tokens = 2;
            probe.fingerprint = 99;
            probe.architecture = ModelArchitecture::hybrid;
            probe.gated = false;
            auto sub = runtime->submit(std::move(probe));
            REQUIRE(sub.ok());
            REQUIRE(runtime->acquire_token(sub.value().id, cancel).ok());
            const auto restored = runtime->lookup_restored_checkpoint(sub.value().id);
            REQUIRE(restored.has_value());
            REQUIRE(restored->bytes != nullptr);
            CHECK(restored->handle.token_position == 8u);
            CHECK(*restored->bytes == std::vector<std::uint8_t>(100, 4));
            (void)runtime->finish(sub.value().id, RequestOutcome::completed);
        }

        CHECK(wait_for([&] { return runtime->tracked_requests() == 0u; }));
        runtime.reset();
        bus.shutdown();
    }

    // Part B: KV block pool eviction cleans up dependent checkpoint blobs via Engine/Session.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry.level = TelemetryLevel::metrics;
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling.kv_block_size_tokens = 4;
        eo.scheduling.kv_num_blocks = 6;  // Small block pool forces block eviction.
        eo.scheduling.admission_watermark_blocks = 0;
        eo.scheduling.prefix_caching = true;
        eo.sample_devices_on_start = false;
        eo.device_sample_interval = std::chrono::milliseconds{0};

        Engine engine(std::move(eo));
        MockBackendOptions mock_opts;
        mock_opts.enable_recurrent_checkpoints = true;
        mock_opts.default_architecture = ModelArchitecture::hybrid;
        REQUIRE(engine.register_backend(make_mock_backend(mock_opts)).ok());

        ModelLoadOptions lo;
        lo.model = "mock:smallpool";
        auto model = engine.load_model(kMockBackendName, lo);
        REQUIRE(model.ok());

        SessionOptions so;
        so.sampling = SamplingConfig::greedy(2);
        auto session = engine.create_session(model.value(), so);
        REQUIRE(session.ok());

        // 11 words + BOS = 12 tokens (3 full blocks).
        const std::string prompt_a = make_words(11, "evictA");
        REQUIRE(session.value()->generate(prompt_a).ok());
        REQUIRE(engine.request_runtime()->checkpoint_store_usage().entries == 1u);

        // 15 words + BOS = 16 tokens (4 full blocks) + 2 decode tokens -> evicts prompt_a's blocks.
        const std::string prompt_b = make_words(15, "evictB");
        REQUIRE(session.value()->generate(prompt_b).ok());

        const auto usage = engine.request_runtime()->checkpoint_store_usage();
        CHECK(usage.evictions >= 1u);
        CHECK(usage.entries == 1u);
        CHECK(usage.total_bytes == 64u);
        CHECK_FALSE(events_of(engine, sink, "kv.checkpoint.evicted").empty());

        // Re-running prompt_a after its blocks were evicted must not restore stale state.
        auto retry_a = session.value()->generate(prompt_a);
        REQUIRE(retry_a.ok());
        CHECK(retry_a.value().scheduling.reused_prompt_tokens == 0u);
        CHECK(events_of(engine, sink, "kv.checkpoint.restored").empty());
        CHECK(engine.kv_usage().sequences == 0u);
        CHECK(engine.kv_usage().pinned_blocks == 0u);
        CHECK(wait_for([&] { return engine.request_runtime()->tracked_requests() == 0u; }));
    }
}

#endif  // SONDER_HAS_KV_CACHE && SONDER_HAS_SCHEDULER

TEST_CASE("disabled scheduling, native chat, and capture failures leave clean state") {
    // Scheduling disabled: no-op checkpoint path.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry.level = TelemetryLevel::metrics;
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling.enabled = false;
        eo.sample_devices_on_start = false;
        eo.device_sample_interval = std::chrono::milliseconds{0};

        Engine engine(std::move(eo));
        MockBackendOptions mock_opts;
        mock_opts.enable_recurrent_checkpoints = true;
        mock_opts.default_architecture = ModelArchitecture::hybrid;
        REQUIRE(engine.register_backend(make_mock_backend(mock_opts)).ok());

        ModelLoadOptions lo;
        lo.model = "mock:hybrid";
        auto model = engine.load_model(kMockBackendName, lo);
        REQUIRE(model.ok());
        CHECK(engine.request_runtime() == nullptr);

        SessionOptions so;
        so.sampling = SamplingConfig::greedy(4);
        auto session = engine.create_session(model.value(), so);
        REQUIRE(session.ok());
        auto res = session.value()->generate("a b c d e f g");
        REQUIRE(res.ok());
        CHECK_FALSE(res.value().scheduling.scheduled);
        engine.telemetry().flush();
        for (const auto& line : sink->lines()) {
            CHECK(line.find("kv.checkpoint.") == std::string::npos);
        }
    }

#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)
    // Capture failure (snapshot exceeds backend cap): generation still succeeds,
    // kv.checkpoint.save_failed is emitted at deep level, and no blob or block leaks.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry.level = TelemetryLevel::deep;
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling.kv_block_size_tokens = 4;
        eo.scheduling.kv_num_blocks = 32;
        eo.sample_devices_on_start = false;
        eo.device_sample_interval = std::chrono::milliseconds{0};

        Engine engine(std::move(eo));
        MockBackendOptions mock_opts;
        mock_opts.enable_recurrent_checkpoints = true;
        mock_opts.default_architecture = ModelArchitecture::hybrid;
        mock_opts.max_checkpoint_bytes = 32;  // Smaller than 64-byte mock snapshot.
        REQUIRE(engine.register_backend(make_mock_backend(mock_opts)).ok());

        ModelLoadOptions lo;
        lo.model = "mock:capfail";
        auto model = engine.load_model(kMockBackendName, lo);
        REQUIRE(model.ok());

        SessionOptions so;
        so.sampling = SamplingConfig::greedy(4);
        auto session = engine.create_session(model.value(), so);
        REQUIRE(session.ok());
        auto res = session.value()->generate(make_words(15, "capfail"));
        REQUIRE(res.ok());
        CHECK(events_of(engine, sink, "kv.checkpoint.saved").empty());
        auto failed_events = events_of(engine, sink, "kv.checkpoint.save_failed");
        REQUIRE(failed_events.size() == 1u);
        CHECK(attr_str(failed_events[0], "stage") == "capture");
        CHECK(attr_str(failed_events[0], "error_code") == "invalid_argument");
        CHECK(engine.request_runtime()->checkpoint_store_usage().entries == 0u);
        CHECK(engine.request_runtime()->checkpoint_store_usage().total_bytes == 0u);
        CHECK(engine.kv_usage().sequences == 0u);
        CHECK(engine.kv_usage().pinned_blocks == 0u);
        CHECK(wait_for([&] { return engine.request_runtime()->tracked_requests() == 0u; }));
    }

    // Native chat uses approximate prompt tokens (not exact) and must gate off checkpoints.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry.level = TelemetryLevel::metrics;
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling.kv_block_size_tokens = 4;
        eo.scheduling.kv_num_blocks = 32;
        eo.sample_devices_on_start = false;
        eo.device_sample_interval = std::chrono::milliseconds{0};

        Engine engine(std::move(eo));
        auto spy = std::make_shared<CheckpointSpyState>();
        MockBackendOptions mock_opts;
        mock_opts.enable_recurrent_checkpoints = true;
        mock_opts.default_architecture = ModelArchitecture::hybrid;
        REQUIRE(engine.register_backend(
                          std::make_shared<SpyCheckpointBackend>(make_mock_backend(mock_opts), spy, true))
                    .ok());

        ModelLoadOptions lo;
        lo.model = "mock:nativechat";
        auto model = engine.load_model("spy-mock", lo);
        REQUIRE(model.ok());

        SessionOptions so;
        so.sampling = SamplingConfig::greedy(4);
        auto session = engine.create_session(model.value(), so);
        REQUIRE(session.ok());
        const std::vector<ChatMessage> msgs{{"user", make_words(15, "chat")}};
        REQUIRE(session.value()->chat(msgs).ok());
        auto second = session.value()->chat(msgs);
        REQUIRE(second.ok());
        CHECK(second.value().scheduling.reused_prompt_tokens == 0u);
        CHECK(spy->capture_calls.load() == 0);
        CHECK(spy->restore_calls.load() == 0);
        CHECK(events_of(engine, sink, "kv.checkpoint.saved").empty());
        CHECK(engine.request_runtime()->checkpoint_store_usage().entries == 0u);
        CHECK(engine.kv_usage().sequences == 0u);
        CHECK(engine.kv_usage().pinned_blocks == 0u);
        CHECK(wait_for([&] { return engine.request_runtime()->tracked_requests() == 0u; }));
    }

    // Decode failure still releases all KV blocks cleanly without leaks.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        EngineOptions eo;
        eo.telemetry.level = TelemetryLevel::metrics;
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling.kv_block_size_tokens = 4;
        eo.scheduling.kv_num_blocks = 32;
        eo.sample_devices_on_start = false;
        eo.device_sample_interval = std::chrono::milliseconds{0};

        Engine engine(std::move(eo));
        MockBackendOptions mock_opts;
        mock_opts.fail_after_tokens = 1;
        mock_opts.enable_recurrent_checkpoints = true;
        mock_opts.default_architecture = ModelArchitecture::hybrid;
        REQUIRE(engine.register_backend(make_mock_backend(mock_opts)).ok());

        ModelLoadOptions lo;
        lo.model = "mock:failing";
        auto model = engine.load_model(kMockBackendName, lo);
        REQUIRE(model.ok());

        SessionOptions so;
        so.sampling = SamplingConfig::greedy(4);
        auto session = engine.create_session(model.value(), so);
        REQUIRE(session.ok());
        auto res = session.value()->generate(make_words(15, "fail"));
        CHECK_FALSE(res.ok());
        CHECK(res.status().code() == ErrorCode::backend_error);
        CHECK(engine.kv_usage().sequences == 0u);
        CHECK(engine.kv_usage().pinned_blocks == 0u);
        CHECK(wait_for([&] { return engine.request_runtime()->tracked_requests() == 0u; }));
    }
#endif
}

}  // TEST_SUITE("recurrent_checkpoints")
