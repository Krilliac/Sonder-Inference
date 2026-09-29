// End-to-end tests for the engine wiring: scheduler + KV cache + sampler
// behind Engine/Session, driven by the mock backend (no network, no model).
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "sonder/inference.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

namespace {

struct Rig {
    std::shared_ptr<MemoryTelemetrySink> sink = std::make_shared<MemoryTelemetrySink>();
    std::unique_ptr<Engine> engine;
    std::shared_ptr<Model> model;

    Rig(MockBackendOptions mock, SchedulingOptions sched, TelemetryLevel level = TelemetryLevel::standard) {
        EngineOptions eo;
        eo.telemetry.level = level;
        eo.telemetry_sinks.push_back(sink);
        eo.scheduling = sched;
        engine = std::make_unique<Engine>(std::move(eo));
        (void)engine->register_backend(make_mock_backend(mock));
        ModelLoadOptions lo;
        lo.model = "mock:tiny";
        auto loaded = engine->load_model(kMockBackendName, lo);
        if (loaded.ok()) {
            model = loaded.value();
        }
    }

    std::shared_ptr<Session> session(SamplingConfig sampling = SamplingConfig::greedy(32),
                                     WorkloadClass workload = WorkloadClass::implementation_worker) {
        SessionOptions so;
        so.sampling = sampling;
        so.workload = workload;
        auto s = engine->create_session(model, so);
        return s.ok() ? s.value() : nullptr;
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
};

// Test-only descriptor override: preserves the mock's execution behaviour while
// exercising the same model -> Session -> scheduler path as native backends.
class ArchitectureModel final : public BackendModel {
public:
    ArchitectureModel(std::shared_ptr<Model> wrapped, ModelArchitecture architecture, bool native_chat)
        : wrapped_(std::move(wrapped)), descriptor_(wrapped_->descriptor()), native_chat_(native_chat) {
        descriptor_.architecture = architecture;
    }
    const ModelDescriptor& descriptor() const override { return descriptor_; }
    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                  const TokenCallback& callback) override {
        return wrapped_->backend_model().generate(request, cancel, callback);
    }
    Result<std::vector<TokenId>> tokenize(std::string_view text) override {
        return wrapped_->backend_model().tokenize(text);
    }
    Result<std::unique_ptr<TokenStream>> open_token_stream(const GenerateRequest& request) override {
        return wrapped_->backend_model().open_token_stream(request);
    }
    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                              const TokenCallback& callback) override {
        return wrapped_->backend_model().chat(request, cancel, callback);
    }
    bool has_native_chat() const override { return native_chat_; }

private:
    std::shared_ptr<Model> wrapped_;
    ModelDescriptor descriptor_;
    bool native_chat_;
};

[[maybe_unused]] std::string attr_str(const json::Value& e, const char* key) {
    const auto* a = e.find("attributes");
    const auto* v = a ? a->find(key) : nullptr;
    return v ? v->as_string() : std::string();
}

[[maybe_unused]] std::int64_t attr_int(const json::Value& e, const char* key) {
    const auto* a = e.find("attributes");
    const auto* v = a ? a->find(key) : nullptr;
    return v ? v->as_int() : -1;
}

[[maybe_unused]] std::string words(int n, const std::string& prefix) {
    std::string out;
    for (int i = 0; i < n; ++i) {
        if (i) out += ' ';
        out += prefix + std::to_string(i);
    }
    return out;
}

template <typename Pred>
[[maybe_unused]] bool wait_for(Pred pred, std::chrono::milliseconds timeout = 5000ms) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (!pred()) {
        if (std::chrono::steady_clock::now() > deadline) return false;
        std::this_thread::sleep_for(1ms);
    }
    return true;
}

[[maybe_unused]] SamplingConfig stochastic(std::uint64_t seed, std::int32_t max_tokens = 20) {
    SamplingConfig s;
    s.temperature = 0.9f;
    s.top_k = 0;
    s.top_p = 1.0f;
    s.min_p = 0.0f;
    s.repeat_penalty = 1.0f;
    s.max_tokens = max_tokens;
    s.seed = seed;
    return s;
}

}  // namespace

TEST_SUITE("engine_runtime") {
#if defined(SONDER_HAS_KV_CACHE) && defined(SONDER_HAS_SCHEDULER)

TEST_CASE("scheduling is active by default and can be disabled") {
    Rig on({}, {});
    REQUIRE(on.model);
    CHECK(on.engine->scheduling_active());
    auto u = on.engine->kv_usage();
    CHECK(u.active);
    CHECK(u.total_blocks == 4096);
    CHECK(u.block_size_tokens == 16);
    CHECK(on.events_of("scheduler.configured").size() == 1);

    SchedulingOptions off;
    off.enabled = false;
    Rig r2({}, off);
    CHECK_FALSE(r2.engine->scheduling_active());
    CHECK_FALSE(r2.engine->kv_usage().active);
    auto res = r2.session()->generate("hello");
    REQUIRE(res.ok());
    CHECK_FALSE(res.value().scheduling.scheduled);
}

TEST_CASE("architecture reaches prefix accounting across generation and both chat routes") {
    // 3 architectures x 2 execution modes x 3 entry routes. No backend can
    // register physical state checkpoints yet, so recurrent reuse must be zero.
    for (const auto architecture : {ModelArchitecture::attention_only, ModelArchitecture::hybrid,
                                   ModelArchitecture::recurrent}) {
        for (const bool token_logits : {false, true}) {
            for (const int route : {0, 1, 2}) {  // generate, generic chat, native chat
                CAPTURE(architecture);
                CAPTURE(token_logits);
                CAPTURE(route);
                MockBackendOptions mock;
                mock.token_logits = token_logits;
                SchedulingOptions sched;
                sched.kv_block_size_tokens = 4;
                sched.prefill_chunk_tokens = 4;
                Rig rig(mock, sched);
                REQUIRE(rig.model);
                auto impl = std::make_shared<ArchitectureModel>(rig.model, architecture, route == 2);
                rig.model = std::make_shared<Model>("architecture-fixture", kMockBackendName, "cpu:0", impl);
                const std::string prompt = words(15, "architecture");
                const std::vector<ChatMessage> messages{{"user", prompt}};
                auto run = [&] {
                    auto session = rig.session(SamplingConfig::greedy(2));
                    if (route == 0) return session->generate(prompt);
                    return session->chat(messages);
                };
                const auto first = run();
                const auto second = run();
                REQUIRE(first.ok());
                REQUIRE(second.ok());
                CHECK(first->text == second->text);
                CHECK(second->scheduling.scheduled);
                CHECK(second->scheduling.accounted_prompt_tokens == first->scheduling.accounted_prompt_tokens);
                if (architecture == ModelArchitecture::attention_only) {
                    CHECK(second->scheduling.reused_prompt_tokens ==
                          (second->scheduling.accounted_prompt_tokens / 4) * 4);
                } else {
                    CHECK(first->scheduling.reused_prompt_tokens == 0);
                    CHECK(second->scheduling.reused_prompt_tokens == 0);
                    CHECK(rig.events_of("kv.reused").empty());
                }
                CHECK(rig.engine->kv_usage().sequences == 0);
                CHECK(rig.engine->kv_usage().pinned_blocks == 0);
            }
        }
    }
}

TEST_CASE("concurrent sessions are batched and produce the unscheduled output") {
    MockBackendOptions mock;
    mock.token_delay = 500us;
    mock.default_completion_tokens = 12;
    SchedulingOptions off;
    off.enabled = false;
    Rig ref(mock, off);
    Rig rig(mock, {});
    REQUIRE(ref.model);
    REQUIRE(rig.model);

    constexpr int kSessions = 6;
    std::vector<std::string> prompts;
    std::vector<std::string> expected;
    for (int i = 0; i < kSessions; ++i) {
        prompts.push_back("agent " + std::to_string(i) + " plans the next step");
        auto r = ref.session()->generate(prompts.back());
        REQUIRE(r.ok());
        expected.push_back(r.value().text);
    }

    std::vector<std::optional<Result<GenerationResult>>> results(kSessions);
    std::vector<std::shared_ptr<Session>> sessions;
    for (int i = 0; i < kSessions; ++i) sessions.push_back(rig.session());
    std::vector<std::thread> threads;
    for (int i = 0; i < kSessions; ++i) {
        threads.emplace_back([&, i] { results[i].emplace(sessions[i]->generate(prompts[i])); });
    }
    for (auto& t : threads) t.join();

    for (int i = 0; i < kSessions; ++i) {
        REQUIRE(results[i]->ok());
        const auto& g = results[i]->value();
        CHECK(g.outcome == RequestOutcome::completed);
        CHECK(g.text == expected[i]);
        CHECK(g.scheduling.scheduled);
        CHECK(g.scheduling.accounted_prompt_tokens > 0);
    }
    CHECK(rig.events_of("scheduler.enqueued").size() == kSessions);
    CHECK(rig.events_of("scheduler.admitted").size() >= kSessions);
    std::int64_t max_batch = 0;
    for (const auto& e : rig.events_of("scheduler.batch.formed")) {
        max_batch = std::max(max_batch, attr_int(e, "sequences"));
    }
    CHECK(max_batch >= 2);
    auto u = rig.engine->kv_usage();
    CHECK(u.pinned_blocks == 0);
    CHECK(u.sequences == 0);
}

TEST_CASE("cache pressure preempts and recomputes without changing output") {
    MockBackendOptions mock;
    mock.token_delay = 1ms;
    mock.default_completion_tokens = 24;
    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    sched.kv_num_blocks = 12;
    sched.admission_watermark_blocks = 0;
    sched.max_requeue_count = 1000;
    SchedulingOptions off;
    off.enabled = false;
    Rig ref(mock, off);
    Rig rig(mock, sched);
    REQUIRE(rig.model);

    // 7 words + BOS = 8 prompt tokens (2 blocks); 24 new tokens -> 8 blocks each.
    constexpr int kSessions = 3;
    std::vector<std::string> prompts;
    std::vector<std::string> expected;
    for (int i = 0; i < kSessions; ++i) {
        prompts.push_back(words(7, "p" + std::to_string(i) + "w"));
        auto r = ref.session(SamplingConfig::greedy(24))->generate(prompts.back());
        REQUIRE(r.ok());
        expected.push_back(r.value().text);
    }
    std::vector<std::optional<Result<GenerationResult>>> results(kSessions);
    std::vector<std::shared_ptr<Session>> sessions;
    for (int i = 0; i < kSessions; ++i) sessions.push_back(rig.session(SamplingConfig::greedy(24)));
    std::vector<std::thread> threads;
    for (int i = 0; i < kSessions; ++i) {
        threads.emplace_back([&, i] { results[i].emplace(sessions[i]->generate(prompts[i])); });
    }
    for (auto& t : threads) t.join();

    std::uint32_t preemptions = 0;
    for (int i = 0; i < kSessions; ++i) {
        REQUIRE(results[i]->ok());
        CHECK(results[i]->value().outcome == RequestOutcome::completed);
        CHECK(results[i]->value().text == expected[i]);
        preemptions += results[i]->value().scheduling.preemptions;
    }
    CHECK(preemptions > 0);
    auto preempted = rig.events_of("scheduler.preempted");
    REQUIRE_FALSE(preempted.empty());
    CHECK(attr_str(preempted.front(), "reason") == "kv_pressure");
    CHECK(rig.events_of("kv.pressure").size() >= 1);
    auto u = rig.engine->kv_usage();
    CHECK(u.pinned_blocks == 0);
    CHECK(u.sequences == 0);
}

TEST_CASE("requeue limit turns repeated preemption into an unavailable error") {
    MockBackendOptions mock;
    mock.token_delay = 1ms;
    mock.default_completion_tokens = 24;
    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    sched.kv_num_blocks = 12;
    sched.admission_watermark_blocks = 0;
    sched.max_requeue_count = 0;
    Rig rig(mock, sched);
    REQUIRE(rig.model);
    constexpr int kSessions = 3;
    std::vector<std::optional<Result<GenerationResult>>> results(kSessions);
    std::vector<std::shared_ptr<Session>> sessions;
    for (int i = 0; i < kSessions; ++i) sessions.push_back(rig.session(SamplingConfig::greedy(24)));
    std::vector<std::thread> threads;
    for (int i = 0; i < kSessions; ++i) {
        threads.emplace_back(
            [&, i] { results[i].emplace(sessions[i]->generate(words(7, "q" + std::to_string(i) + "w"))); });
    }
    for (auto& t : threads) t.join();
    int completed = 0;
    int unavailable = 0;
    for (auto& r : results) {
        if (r->ok() && r->value().outcome == RequestOutcome::completed) ++completed;
        if (!r->ok() && r->status().code() == ErrorCode::unavailable) ++unavailable;
    }
    CHECK(completed >= 1);
    CHECK(unavailable >= 1);
    CHECK(completed + unavailable == kSessions);
    CHECK(rig.engine->kv_usage().pinned_blocks == 0);
    CHECK(rig.engine->kv_usage().sequences == 0);
}

TEST_CASE("prompt larger than the KV pool is rejected up front") {
    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    sched.kv_num_blocks = 4;
    Rig rig({}, sched);
    auto r = rig.session()->generate(words(40, "big"));
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    CHECK(rig.events_of("scheduler.rejected").size() == 1);
    CHECK(rig.engine->kv_usage().sequences == 0);
}

TEST_CASE("an exact prompt that fills num_ctx is rejected before generation") {
    for (bool token_logits : {false, true}) {
        MockBackendOptions mock;
        mock.token_logits = token_logits;
        Rig rig(mock, {});
        REQUIRE(rig.model);
        auto sampling = SamplingConfig::greedy(8);
        sampling.num_ctx = 4;
        // The mock tokenizer adds one BOS token: these prompts count as 4
        // and 11 tokens, respectively.
        for (const auto& prompt : {words(3, "ctx"), words(10, "ctx")}) {
            auto result = rig.session(sampling)->generate(prompt);
            REQUIRE_FALSE(result.ok());
            CHECK(result.status().code() == ErrorCode::invalid_argument);
        }
        CHECK(rig.events_of("scheduler.rejected").size() == 2);
        CHECK(rig.engine->kv_usage().sequences == 0);
    }
}

#if defined(SONDER_HAS_SAMPLER_CHAIN)
TEST_CASE("seeded sampling through the sampler chain is deterministic") {
    MockBackendOptions mock;
    mock.token_logits = true;
    mock.default_completion_tokens = 20;
    Rig rig(mock, {});
    REQUIRE(rig.model);
    const std::string prompt = "write a haiku about schedulers";

    auto a = rig.session(stochastic(1234))->generate(prompt);
    REQUIRE(a.ok());
    CHECK(a.value().scheduling.sonder_sampled);
    CHECK_FALSE(a.value().text.empty());

    // Same seed, run concurrently: identical output.
    std::optional<Result<GenerationResult>> b;
    std::optional<Result<GenerationResult>> c;
    auto sb = rig.session(stochastic(1234));
    auto sc = rig.session(stochastic(1234));
    std::thread tb([&] { b.emplace(sb->generate(prompt)); });
    std::thread tc([&] { c.emplace(sc->generate(prompt)); });
    tb.join();
    tc.join();
    REQUIRE(b->ok());
    REQUIRE(c->ok());
    CHECK(b->value().text == a.value().text);
    CHECK(c->value().text == a.value().text);

    // Scheduling does not influence sampling.
    SchedulingOptions off;
    off.enabled = false;
    Rig plain(mock, off);
    auto d = plain.session(stochastic(1234))->generate(prompt);
    REQUIRE(d.ok());
    CHECK(d.value().text == a.value().text);
    CHECK(d.value().scheduling.sonder_sampled);

    // Different seeds diverge.
    bool diverged = false;
    for (std::uint64_t seed = 1; seed <= 6 && !diverged; ++seed) {
        auto e = rig.session(stochastic(seed))->generate(prompt);
        REQUIRE(e.ok());
        diverged = e.value().text != a.value().text;
    }
    CHECK(diverged);

    // Greedy ignores the seed.
    auto g1 = SamplingConfig::greedy(20);
    auto g2 = g1;
    g1.seed = 1;
    g2.seed = 2;
    auto r1 = rig.session(g1)->generate(prompt);
    auto r2 = rig.session(g2)->generate(prompt);
    REQUIRE(r1.ok());
    REQUIRE(r2.ok());
    CHECK(r1.value().text == r2.value().text);

    auto configured = rig.events_of("sampling.configured");
    REQUIRE_FALSE(configured.empty());
    CHECK(attr_str(configured.front(), "sampler") == "sonder");
}

TEST_CASE("a policy that excludes every token maps NoViableCandidates to invalid_argument") {
    MockBackendOptions mock;
    mock.token_logits = true;
    mock.ban_all_tokens = true;  // stands in for per-request bans (logit_bias lands in feat/sampling-config)
    Rig rig(mock, {});
    auto s = SamplingConfig::greedy(8);
    auto r = rig.session(s)->generate("anything");
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    auto failed = rig.events_of("sampling.failed");
    REQUIRE(failed.size() == 1);
    CHECK(attr_str(failed.front(), "status") == "no_viable_candidates");
    CHECK(attr_str(failed.front(), "error_code") == "invalid_argument");
    CHECK(rig.engine->kv_usage().sequences == 0);
}
#endif

TEST_CASE("cancelling a running request frees its cache blocks") {
    MockBackendOptions mock;
    mock.token_delay = 3ms;
    mock.default_completion_tokens = 400;
    Rig rig(mock, {});
    auto session = rig.session(SamplingConfig::greedy(400));
    std::atomic<int> chunks{0};
    std::optional<Result<GenerationResult>> result;
    std::thread t([&] {
        result.emplace(session->generate(words(20, "ctx"), [&](const TokenChunk&) {
            ++chunks;
            return true;
        }));
    });
    REQUIRE(wait_for([&] { return chunks.load() >= 3; }));
    auto during = rig.engine->kv_usage();
    CHECK(during.pinned_blocks > 0);
    CHECK(during.sequences == 1);
    session->cancel();
    t.join();
    REQUIRE(result->ok());
    CHECK(result->value().outcome == RequestOutcome::cancelled);
    auto after = rig.engine->kv_usage();
    CHECK(after.pinned_blocks == 0);
    CHECK(after.sequences == 0);
    bool freed_cancelled = false;
    for (const auto& e : rig.events_of("kv.freed")) {
        freed_cancelled |= attr_str(e, "reason") == "cancelled";
    }
    CHECK(freed_cancelled);
}

TEST_CASE("cancelling a queued request releases it without admission") {
    MockBackendOptions mock;
    mock.token_delay = 10ms;
    mock.default_completion_tokens = 8;
    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    sched.kv_num_blocks = 4;
    sched.admission_watermark_blocks = 0;
    Rig rig(mock, sched);
    auto a = rig.session(SamplingConfig::greedy(8));
    auto b = rig.session(SamplingConfig::greedy(4));
    std::atomic<int> a_chunks{0};
    std::atomic<int> b_chunks{0};
    std::optional<Result<GenerationResult>> ra;
    std::optional<Result<GenerationResult>> rb;
    // A: 8 prompt tokens (2 blocks). B: 12 prompt tokens (3 blocks) -> cannot be admitted while A runs.
    std::thread ta([&] {
        ra.emplace(a->generate(words(7, "a"), [&](const TokenChunk&) {
            ++a_chunks;
            return true;
        }));
    });
    REQUIRE(wait_for([&] { return a_chunks.load() >= 1; }));
    std::thread tb([&] {
        rb.emplace(b->generate(words(11, "b"), [&](const TokenChunk&) {
            ++b_chunks;
            return true;
        }));
    });
    REQUIRE(wait_for([&] { return rig.events_of("scheduler.enqueued").size() == 2; }));
    std::this_thread::sleep_for(20ms);
    CHECK(b_chunks.load() == 0);
    b->cancel();
    tb.join();
    REQUIRE(rb->ok());
    CHECK(rb->value().outcome == RequestOutcome::cancelled);
    CHECK(rb->value().text.empty());
    ta.join();
    REQUIRE(ra->ok());
    CHECK(ra->value().outcome == RequestOutcome::completed);
    CHECK(rig.engine->kv_usage().sequences == 0);
    CHECK(rig.engine->kv_usage().pinned_blocks == 0);
}

TEST_CASE("a repeated prompt reuses cached prefix blocks") {
    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    Rig rig({}, sched);
    const std::string prompt = words(16, "shared");  // 17 tokens -> 4 full blocks
    auto first = rig.session()->generate(prompt);
    REQUIRE(first.ok());
    CHECK(first.value().scheduling.reused_prompt_tokens == 0);
    auto second = rig.session()->generate(prompt);
    REQUIRE(second.ok());
    CHECK(second.value().scheduling.reused_prompt_tokens >= 12);
    CHECK(second.value().text == first.value().text);
    CHECK_FALSE(rig.events_of("kv.reused").empty());
    CHECK(rig.engine->kv_usage().prefix_hit_blocks > 0);
}

TEST_CASE("concurrent requests with a common prefix share blocks") {
    MockBackendOptions mock;
    mock.token_delay = 3ms;
    mock.default_completion_tokens = 200;
    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    Rig rig(mock, sched);
    const std::string prompt = words(16, "system");
    auto a = rig.session(SamplingConfig::greedy(200));
    auto b = rig.session(SamplingConfig::greedy(200));
    std::atomic<int> ca{0};
    std::atomic<int> cb{0};
    std::thread ta([&] { (void)a->generate(prompt, [&](const TokenChunk&) { return ++ca, true; }); });
    REQUIRE(wait_for([&] { return ca.load() >= 1; }));
    std::thread tb([&] { (void)b->generate(prompt + " extra", [&](const TokenChunk&) { return ++cb, true; }); });
    REQUIRE(wait_for([&] { return cb.load() >= 1; }));
    CHECK(rig.engine->kv_usage().shared_blocks >= 1);
    a->cancel();
    b->cancel();
    ta.join();
    tb.join();
    CHECK(rig.engine->kv_usage().pinned_blocks == 0);
}

TEST_CASE("num_ctx narrows the scheduling context and new sampling fields are reported") {
    Rig rig({}, {});
    auto s = SamplingConfig::greedy(4);
    s.num_ctx = 512;
    s.typical_p = 0.95f;
    s.presence_penalty = 0.5f;
    auto r = rig.session(s)->generate("hello");
    REQUIRE(r.ok());
    auto enq = rig.events_of("scheduler.enqueued");
    REQUIRE(enq.size() == 1);
    CHECK(attr_int(enq[0], "context_limit") == 512);
    auto created = rig.events_of("session.created");
    REQUIRE(created.size() == 1);
    const auto* sampling = created[0].find("attributes")->find("sampling");
    REQUIRE(sampling);
    CHECK(sampling->find("num_ctx")->as_int() == 512);
    CHECK(sampling->find("typical_p")->as_double() == doctest::Approx(0.95));
    CHECK(sampling->find("presence_penalty")->as_double() == doctest::Approx(0.5));
    CHECK(sampling->find("frequency_penalty")->as_double() == doctest::Approx(0.0));
    CHECK(sampling->find("repeat_last_n")->as_int() == 64);
    CHECK(sampling->find("logit_bias_count")->as_int() == 0);
}

TEST_CASE("workload class is reported and higher classes are scheduled") {
    Rig rig({}, {});
    auto s = rig.session(SamplingConfig::greedy(4), WorkloadClass::interactive_user);
    REQUIRE(s);
    auto r = s->generate("hi");
    REQUIRE(r.ok());
    auto created = rig.events_of("session.created");
    REQUIRE_FALSE(created.empty());
    CHECK(attr_str(created.back(), "workload") == "interactive_user");
}

// --- Stalled requests must not freeze the lock-step scheduler -------------

// A streaming backend (no token logits, like Ollama) whose model "thinks"
// before its first visible chunk: it emits nothing for `think` and then
// streams `tokens` chunks. The thinking phase is exactly what a reasoning
// model, a cold weight load or Ollama queueing looks like to the engine.
class ThinkingModel final : public BackendModel {
public:
    ThinkingModel(std::chrono::milliseconds think, int tokens) : think_(think), tokens_(tokens) {
        desc_.name = "thinker";
        desc_.backend = "thinker";
        desc_.format = "test";
    }
    const ModelDescriptor& descriptor() const override { return desc_; }
    Result<GenerateStats> generate(const GenerateRequest&, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        const auto until = std::chrono::steady_clock::now() + think_;
        while (std::chrono::steady_clock::now() < until) {
            if (cancel.cancelled()) return Status(ErrorCode::cancelled, "cancelled");
            std::this_thread::sleep_for(1ms);
        }
        GenerateStats st;
        for (int i = 0; i < tokens_; ++i) {
            const std::string text = " t" + std::to_string(i);
            if (!on_chunk(TokenChunk{text, st.chunks++})) {
                st.stop_reason = StopReason::callback;
                return st;
            }
            ++st.completion_tokens;
        }
        st.stop_reason = StopReason::end_of_sequence;
        return st;
    }

private:
    ModelDescriptor desc_;
    std::chrono::milliseconds think_;
    int tokens_;
};

class ThinkingBackend final : public Backend {
public:
    explicit ThinkingBackend(std::chrono::milliseconds think) : think_(think) {}
    std::string name() const override { return "thinker"; }
    std::string description() const override { return "test backend that thinks before streaming"; }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::streaming);
        return c;
    }
    Result<std::string> probe() override { return std::string("test"); }
    Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions&) override {
        return std::shared_ptr<BackendModel>(std::make_shared<ThinkingModel>(think_, 4));
    }

private:
    std::chrono::milliseconds think_;
};

TEST_CASE("a backend thinking before its first chunk does not freeze other requests") {
    MockBackendOptions mock;
    mock.token_delay = 1ms;
    mock.default_completion_tokens = 12;
    Rig rig(mock, {});
    REQUIRE(rig.model);
    REQUIRE(rig.engine->register_backend(std::make_shared<ThinkingBackend>(3000ms)).ok());
    ModelLoadOptions lo;
    lo.model = "thinker";
    auto thinker = rig.engine->load_model("thinker", lo);
    REQUIRE(thinker.ok());
    SessionOptions so;
    so.sampling = SamplingConfig::greedy(32);
    auto slow = rig.engine->create_session(thinker.value(), so);
    REQUIRE(slow.ok());

    std::optional<Result<GenerationResult>> slow_result;
    std::thread tb([&] { slow_result.emplace(slow.value()->generate("think hard")); });
    // The thinker holds a grant it cannot use until its first chunk.
    REQUIRE(wait_for([&] { return !rig.events_of("scheduler.prefill.completed").empty(); }));

    const auto t0 = std::chrono::steady_clock::now();
    auto fast = rig.session()->generate("quick question");
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    REQUIRE(fast.ok());
    CHECK(fast.value().outcome == RequestOutcome::completed);
    CHECK(fast.value().stats.completion_tokens == 12);
    // Before the fix the fast request waited for the whole thinking phase.
    CHECK(elapsed < 2000ms);
    tb.join();
    REQUIRE(slow_result->ok());
    CHECK(slow_result->value().outcome == RequestOutcome::completed);
    CHECK(slow_result->value().text == " t0 t1 t2 t3");
    CHECK_FALSE(rig.events_of("scheduler.stalled").empty());
    auto u = rig.engine->kv_usage();
    CHECK(u.pinned_blocks == 0);
    CHECK(u.sequences == 0);
}

TEST_CASE("a session blocked in its chunk callback (slow client) does not freeze other requests") {
    MockBackendOptions mock;
    mock.token_delay = 1ms;
    mock.default_completion_tokens = 12;
    Rig rig(mock, {});
    REQUIRE(rig.model);

    std::atomic<bool> blocked{false};
    std::optional<Result<GenerationResult>> slow_result;
    std::thread tb([&] {
        int chunks = 0;
        slow_result.emplace(rig.session()->generate("slow reader", [&](const TokenChunk&) {
            if (++chunks == 2) {
                blocked.store(true);
                std::this_thread::sleep_for(3000ms);  // a client that stopped reading
            }
            return true;
        }));
    });
    REQUIRE(wait_for([&] { return blocked.load(); }));

    const auto t0 = std::chrono::steady_clock::now();
    auto fast = rig.session()->generate("fast reader");
    const auto elapsed = std::chrono::steady_clock::now() - t0;
    REQUIRE(fast.ok());
    CHECK(fast.value().outcome == RequestOutcome::completed);
    CHECK(fast.value().stats.completion_tokens == 12);
    CHECK(elapsed < 2000ms);
    tb.join();
    REQUIRE(slow_result->ok());
    CHECK(slow_result->value().outcome == RequestOutcome::completed);
    CHECK(slow_result->value().stats.completion_tokens == 12);
    auto u = rig.engine->kv_usage();
    CHECK(u.pinned_blocks == 0);
    CHECK(u.sequences == 0);
}

// --- The context/KV budget caps generation ---------------------------------

TEST_CASE("num_ctx caps generation length on the sampler-chain path") {
    MockBackendOptions mock;
    mock.token_logits = true;
    mock.default_completion_tokens = 1000;  // no natural stop within the test
    Rig rig(mock, {});
    REQUIRE(rig.model);
    auto s = SamplingConfig::greedy(64);
    s.num_ctx = 12;
    auto r = rig.session(s)->generate("one two three four five six seven");
    REQUIRE(r.ok());
    const auto& g = r.value();
    REQUIRE(g.scheduling.exact_prompt_tokens);
    const auto prompt = g.scheduling.accounted_prompt_tokens;
    REQUIRE(prompt < 12);
    CHECK(g.stats.completion_tokens <= 12 - prompt);
    CHECK(g.stats.completion_tokens > 0);
    CHECK(g.stats.stop_reason == StopReason::max_tokens);
}

TEST_CASE("num_ctx caps generation length on the backend-streamed path") {
    MockBackendOptions mock;
    mock.default_completion_tokens = 1000;
    Rig rig(mock, {});
    REQUIRE(rig.model);
    auto s = SamplingConfig::greedy(64);
    s.num_ctx = 12;
    auto r = rig.session(s)->generate("one two three four five six seven");
    REQUIRE(r.ok());
    const auto& g = r.value();
    REQUIRE(g.scheduling.exact_prompt_tokens);
    const auto prompt = g.scheduling.accounted_prompt_tokens;
    REQUIRE(prompt < 12);
    CHECK(g.stats.completion_tokens <= 12 - prompt);
    CHECK(g.stats.completion_tokens > 0);
}

TEST_CASE("the KV pool size caps generation length") {
    MockBackendOptions mock;
    mock.token_logits = true;
    mock.default_completion_tokens = 1000;
    SchedulingOptions sched;
    sched.kv_block_size_tokens = 4;
    sched.kv_num_blocks = 6;  // 24 tokens
    sched.admission_watermark_blocks = 0;
    Rig rig(mock, sched);
    REQUIRE(rig.model);
    auto r = rig.session(SamplingConfig::greedy(200))->generate("one two three four five");
    REQUIRE(r.ok());
    const auto& g = r.value();
    const auto prompt = g.scheduling.accounted_prompt_tokens;
    REQUIRE(prompt < 24);
    CHECK(g.stats.completion_tokens <= 24 - prompt);
    CHECK(g.stats.completion_tokens > 0);
}

#else

TEST_CASE("scheduling is inactive without the cache and scheduler modules") {
    Rig rig({}, {});
    CHECK_FALSE(rig.engine->scheduling_active());
    CHECK_FALSE(rig.engine->kv_usage().active);
    auto r = rig.session()->generate("hello");
    REQUIRE(r.ok());
    CHECK_FALSE(r.value().scheduling.scheduled);
}

#endif
}
