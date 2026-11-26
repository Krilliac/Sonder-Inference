// Tests for Package 4 / Items #8 & #9:
//   - Caller-visible ExecutionCheckpointHandle / CheckpointStore with explicit
//     Branch vs ExactResume modes
//   - Session::fork, rewind, export_checkpoint, resume_from_checkpoint, release_checkpoint
//   - Strict same-process / same-model-build-tokenizer compatibility enforcement
//   - Bounded checkpoint memory cap, LRU eviction, TTL expiry, and cleanup on cancel/close
//   - Backend-owned selective preemption safe points & non-blocking gate separation
//     verifying physical slot reclamation and uninterrupted peer progress on a shared worker.
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "engine/request_runtime.hpp"
#include "sonder/inference.hpp"

using namespace sonder::inference;
using namespace std::chrono_literals;

namespace {

EngineOptions make_test_engine_options(std::uint32_t block_size = 4, std::uint32_t num_blocks = 64) {
    EngineOptions opts;
    opts.sample_devices_on_start = false;
    opts.device_sample_interval = 0ms;
    opts.sample_gpu_devices = false;
    opts.scheduling.enabled = true;
    opts.scheduling.kv_block_size_tokens = block_size;
    opts.scheduling.kv_num_blocks = num_blocks;
    opts.scheduling.prefix_caching = true;
    return opts;
}

// Deterministic stateful grammar automaton used to prove ExactResume snapshots
// and restores grammar matcher state, while Branch resets it cleanly.
struct ParityGrammarAutomaton {
    std::uint8_t step = 0;
    std::uint32_t reset_calls = 0;
    std::uint32_t restore_calls = 0;

    GrammarMatcherHooks hooks() {
        GrammarMatcherHooks h;
        h.mask_logits = [this](std::span<float> logits) {
            if (logits.size() <= 2) return;
            // Alternate parity allowed on each step, always keeping EOS (last token) unmasked.
            const std::size_t eos = logits.size() - 1;
            const std::size_t want_parity = static_cast<std::size_t>(step & 1u);
            for (std::size_t i = 0; i < eos; ++i) {
                if ((i & 1u) != want_parity) {
                    logits[i] = -std::numeric_limits<float>::infinity();
                }
            }
        };
        h.accept_token = [this](TokenId tok) {
            // Mix accepted token into automaton phase so divergence would be immediately visible.
            step = static_cast<std::uint8_t>((step + 1u + static_cast<std::uint8_t>(tok & 3)) & 0x0Fu);
        };
        h.reset = [this]() {
            step = 0;
            ++reset_calls;
        };
        h.snapshot_state = [this]() -> std::vector<std::uint8_t> {
            return std::vector<std::uint8_t>{0x47u, 0x4Du, step};
        };
        h.restore_state = [this](std::span<const std::uint8_t> bytes) -> bool {
            if (bytes.size() != 3u || bytes[0] != 0x47u || bytes[1] != 0x4Du) {
                return false;
            }
            step = bytes[2];
            ++restore_calls;
            return true;
        };
        return h;
    }
};

// ---------------------------------------------------------------------------
// Single-worker-thread multiplexing mock backend modeling LlamaCppBackend's
// shared BatchWorkerLoop:
//   - Steps all active slots in lockstep on ONE shared worker thread
//   - Polls each slot's non-blocking on_safe_point hook every decode step
//   - When one slot suspends, immediately captures its spill blob (if yield_spill),
//     reclaims THAT physical slot, and keeps stepping all peer slots on the very
//     next iteration without ever blocking or stalling peers.
// ---------------------------------------------------------------------------
class SharedWorkerModel final : public BackendModel {
public:
    explicit SharedWorkerModel(std::string model_name, std::size_t max_slots = 2)
        : max_slots_(max_slots) {
        descriptor_.name = std::move(model_name);
        descriptor_.backend = "shared-batch-mock";
        descriptor_.format = "test";
        descriptor_.context_length = 4096;
        worker_ = std::thread([this]() { worker_loop(); });
    }

    ~SharedWorkerModel() override {
        {
            std::lock_guard<std::mutex> lock(mu_);
            stop_worker_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    const ModelDescriptor& descriptor() const override { return descriptor_; }

    Result<std::vector<TokenId>> tokenize(std::string_view text) override {
        std::vector<TokenId> out;
        std::size_t i = 0;
        while (i < text.size()) {
            while (i < text.size() && text[i] == ' ') ++i;
            const std::size_t start = i;
            while (i < text.size() && text[i] != ' ') ++i;
            if (i > start) {
                std::uint32_t h = 2166136261u;
                for (std::size_t k = start; k < i; ++k) {
                    h ^= static_cast<std::uint8_t>(text[k]);
                    h *= 16777619u;
                }
                out.push_back(static_cast<TokenId>((h % 64u) + 1u));
            }
        }
        if (out.empty()) out.push_back(1);
        return out;
    }

    std::size_t active_physical_slots() const override {
        return active_slots_.load(std::memory_order_acquire);
    }

    std::uint64_t reclaimed_physical_slots() const override {
        return reclaimed_slots_.load(std::memory_order_acquire);
    }

    std::uint64_t peer_steps_after_reclaim() const {
        return peer_steps_after_reclaim_.load(std::memory_order_acquire);
    }

    std::size_t peak_active_slots() const {
        return peak_active_slots_.load(std::memory_order_acquire);
    }

    Result<std::vector<std::uint8_t>> capture_execution_state(std::uint64_t token_position,
                                                              std::size_t max_bytes) override {
        if (max_bytes < 16u) {
            return Status(ErrorCode::invalid_argument, "max_bytes too small");
        }
        std::vector<std::uint8_t> blob(16, 0);
        std::memcpy(blob.data(), "SBW1", 4);
        std::memcpy(blob.data() + 8, &token_position, sizeof(token_position));
        return blob;
    }

    Status restore_execution_state(std::span<const std::uint8_t> bytes, std::uint64_t token_position) override {
        if (bytes.size() < 16u || std::memcmp(bytes.data(), "SBW1", 4) != 0) {
            return Status(ErrorCode::invalid_argument, "invalid shared-worker blob");
        }
        std::uint64_t stored_pos = 0;
        std::memcpy(&stored_pos, bytes.data() + 8, sizeof(stored_pos));
        if (token_position != 0 && stored_pos != token_position) {
            return Status(ErrorCode::invalid_argument, "token_position mismatch");
        }
        return Status::success();
    }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        auto task = std::make_shared<SlotTask>();
        task->request = &request;
        task->cancel = &cancel;
        task->on_chunk = &on_chunk;
        task->prompt_tokens = tokenize(request.prompt).value();
        task->base_seed = fnv1a_sv(request.prompt) ^ request.sampling.seed.value_or(0);
        task->max_steps = std::max<std::int32_t>(0, request.sampling.max_tokens);
        if (request.execution_resume != nullptr) {
            const auto& er = *request.execution_resume;
            if (!er.backend_state_blob.empty()) {
                if (er.backend_state_blob.size() != 16u || std::memcmp(er.backend_state_blob.data(), "SBW1", 4) != 0) {
                    return Status(ErrorCode::invalid_argument, "corrupted shared-worker spill blob");
                }
                std::uint64_t saved_step = 0;
                std::memcpy(&saved_step, er.backend_state_blob.data() + 8, sizeof(saved_step));
                task->step_index = saved_step;
                task->restored_from_spill = true;
            } else {
                task->step_index = er.backend_step_index;
            }
            task->generated_tokens = er.generated_token_ids;
        }

        auto fut = task->promise.get_future();
        {
            std::lock_guard<std::mutex> lock(mu_);
            pending_.push(std::move(task));
        }
        cv_.notify_one();
        return fut.get();
    }

private:
    struct SlotTask {
        const GenerateRequest* request = nullptr;
        const CancellationToken* cancel = nullptr;
        const TokenCallback* on_chunk = nullptr;
        std::vector<TokenId> prompt_tokens;
        std::vector<TokenId> generated_tokens;
        std::uint64_t base_seed = 0;
        std::uint64_t step_index = 0;
        std::int32_t produced_this_call = 0;
        std::int32_t max_steps = 0;
        bool restored_from_spill = false;
        GenerateStats stats;
        std::promise<Result<GenerateStats>> promise;
    };

    static std::uint64_t fnv1a_sv(std::string_view s) noexcept {
        std::uint64_t h = 14695981039346656037ull;
        for (unsigned char c : s) {
            h ^= c;
            h *= 1099511628211ull;
        }
        return h;
    }

    static std::string word_for(std::uint64_t seed, std::uint64_t step) {
        static constexpr const char* kWords[] = {
            "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
            "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
        };
        const std::uint64_t idx = (seed ^ (step * 0x9E3779B97F4A7C15ull)) % 16u;
        return std::string(step > 0 ? " " : "") + kWords[idx] + std::to_string(step);
    }

    void worker_loop() {
        std::vector<std::shared_ptr<SlotTask>> active;
        while (true) {
            {
                std::unique_lock<std::mutex> lock(mu_);
                cv_.wait(lock, [&]() { return stop_worker_ || !pending_.empty() || !active.empty(); });
                if (stop_worker_ && pending_.empty() && active.empty()) {
                    return;
                }
                while (active.size() < max_slots_ && !pending_.empty()) {
                    auto t = std::move(pending_.front());
                    pending_.pop();
                    t->stats.prompt_tokens = t->prompt_tokens.size();
                    t->stats.token_counts_from_backend = true;
                    if (t->restored_from_spill) {
                        t->stats.cached_tokens = t->prompt_tokens.size() + t->step_index;
                    }
                    active.push_back(std::move(t));
                    const std::size_t now_active = active.size();
                    active_slots_.store(now_active, std::memory_order_release);
                    std::size_t prev_peak = peak_active_slots_.load(std::memory_order_relaxed);
                    while (now_active > prev_peak &&
                           !peak_active_slots_.compare_exchange_weak(prev_peak, now_active)) {
                    }
                }
            }

            // Lockstep batch iteration across all active slots on this SINGLE worker thread.
            for (std::size_t i = 0; i < active.size();) {
                auto& s = *active[i];
                if (s.cancel->cancelled()) {
                    s.promise.set_value(Status(ErrorCode::cancelled, "cancelled"));
                    active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                    active_slots_.store(active.size(), std::memory_order_release);
                    continue;
                }
                if (s.produced_this_call >= s.max_steps) {
                    s.stats.stop_reason = StopReason::max_tokens;
                    s.promise.set_value(s.stats);
                    active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                    active_slots_.store(active.size(), std::memory_order_release);
                    continue;
                }

                // Non-blocking safe-point poll BEFORE stepping this slot.
                if (s.request->on_safe_point) {
                    SafePointContext sp;
                    sp.prompt_tokens = s.prompt_tokens.size();
                    sp.prefill_completed_tokens = s.prompt_tokens.size();
                    sp.generated_tokens = s.step_index;
                    sp.kv_tokens = s.prompt_tokens.size() + s.step_index;
                    sp.active_peers = active.size() > 1u ? active.size() - 1u : 0u;
                    sp.in_prefill = false;
                    const SuspendDecision dec = s.request->on_safe_point(sp);
                    if (dec.action != SuspendAction::continue_running) {
                        if (s.request->suspended_state != nullptr) {
                            auto& ss = *s.request->suspended_state;
                            ss.action = dec.action;
                            ss.token_position = s.prompt_tokens.size() + s.step_index;
                            ss.generated_tokens = s.step_index;
                            ss.backend_step_index = s.step_index;
                            ss.tokens = s.prompt_tokens;
                            ss.tokens.insert(ss.tokens.end(), s.generated_tokens.begin(), s.generated_tokens.end());
                            ss.physical_slot_reclaimed = true;
                            ss.backend_state_blob.clear();
                            if (dec.action == SuspendAction::yield_spill && dec.max_spill_bytes >= 16u) {
                                ss.backend_state_blob.assign(16u, 0);
                                std::memcpy(ss.backend_state_blob.data(), "SBW1", 4);
                                std::memcpy(ss.backend_state_blob.data() + 8, &s.step_index, sizeof(s.step_index));
                            }
                        }
                        s.stats.stop_reason = StopReason::suspended;
                        s.promise.set_value(s.stats);
                        active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                        active_slots_.store(active.size(), std::memory_order_release);
                        reclaimed_slots_.fetch_add(1, std::memory_order_acq_rel);
                        continue;
                    }
                }

                // Produce one deterministic token for this slot.
                const std::string piece = word_for(s.base_seed, s.step_index);
                const TokenId tok = static_cast<TokenId>((s.base_seed + s.step_index) % 1024u + 1u);
                s.generated_tokens.push_back(tok);
                ++s.step_index;
                ++s.produced_this_call;
                ++s.stats.completion_tokens;
                if (reclaimed_slots_.load(std::memory_order_acquire) > 0) {
                    peer_steps_after_reclaim_.fetch_add(1, std::memory_order_acq_rel);
                }
                TokenChunk chunk{piece, s.stats.chunks++};
                if (*s.on_chunk && !(*s.on_chunk)(chunk)) {
                    s.stats.stop_reason = StopReason::callback;
                    s.promise.set_value(s.stats);
                    active.erase(active.begin() + static_cast<std::ptrdiff_t>(i));
                    active_slots_.store(active.size(), std::memory_order_release);
                    continue;
                }
                ++i;
            }
            if (!active.empty()) {
                std::this_thread::sleep_for(2ms);
            }
        }
    }

    ModelDescriptor descriptor_;
    std::size_t max_slots_ = 2;
    std::mutex mu_;
    std::condition_variable cv_;
    bool stop_worker_ = false;
    std::queue<std::shared_ptr<SlotTask>> pending_;
    std::atomic<std::size_t> active_slots_{0};
    std::atomic<std::size_t> peak_active_slots_{0};
    std::atomic<std::uint64_t> reclaimed_slots_{0};
    std::atomic<std::uint64_t> peer_steps_after_reclaim_{0};
    std::thread worker_;
};

class SharedWorkerBackend final : public Backend {
public:
    explicit SharedWorkerBackend(std::size_t slots = 2) : slots_(slots) {}

    std::string name() const override { return "shared-batch-mock"; }
    std::string description() const override { return "shared-worker continuous-batch mock"; }
    std::size_t max_concurrent_requests() const override { return slots_; }

    BackendCapabilities capabilities() const override {
        BackendCapabilities caps;
        caps.add(Capability::tokenization)
            .add(Capability::streaming)
            .add(Capability::deterministic)
            .add(Capability::continuous_batch_decode)
            .add(Capability::execution_checkpoint)
            .add(Capability::selective_preemption);
        return caps;
    }

    Result<std::string> probe() override { return std::string("1.0"); }

    Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }

    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        auto m = std::make_shared<SharedWorkerModel>(options.model, slots_);
        last_model_ = m;
        return std::shared_ptr<BackendModel>(std::move(m));
    }

    std::shared_ptr<SharedWorkerModel> last_model() const { return last_model_; }

private:
    std::size_t slots_ = 2;
    std::shared_ptr<SharedWorkerModel> last_model_;
};

}  // namespace

TEST_CASE("execution lifecycle: Branch fork isolates logical KV and diverges cleanly without mutating parent or sibling") {
    Engine engine(make_test_engine_options(/*block_size=*/4, /*num_blocks=*/64));
    MockBackendOptions mo;
    mo.token_logits = true;
    mo.enable_execution_checkpoints = true;
    mo.enable_selective_preemption = true;
    REQUIRE(engine.register_backend(make_mock_backend(mo)).ok());

    auto model = engine.load_model("mock", ModelLoadOptions{"mock:tiny"});
    REQUIRE(model.ok());

    SessionOptions so;
    so.sampling.temperature = 0.0f;
    so.sampling.max_tokens = 4;
    auto parent_res = engine.create_session(model.value(), so);
    REQUIRE(parent_res.ok());
    auto parent = parent_res.value();

    // Warm parent with an 8-word shared prefix + 4 generated tokens.
    const std::string shared_prefix = "alpha bravo charlie delta echo foxtrot golf hotel";
    const std::size_t prompt_tok_count = model.value()->backend_model().tokenize(shared_prefix).value().size();
    const std::size_t expected_warm_tokens = prompt_tok_count + 4u;
    auto warm = parent->generate(shared_prefix);
    REQUIRE(warm.ok());
    CHECK(warm.value().outcome == RequestOutcome::completed);
    // Ordinary generate() does not pin persistent logical KV blocks until fork/rewind/checkpoint.
    CHECK(parent->logical_kv_sequence_id() == 0u);
    CHECK(engine.kv_usage().sequences == 0u);

    const std::string parent_prompt_before = parent->context_prompt();
    const std::vector<TokenId> parent_tokens_before = parent->context_tokens();
    REQUIRE(parent_tokens_before.size() == expected_warm_tokens);

    // Fork two isolated branches from the warmed parent context.
    auto fork_a = parent->fork();
    auto fork_b = parent->fork();
    REQUIRE(fork_a.ok());
    REQUIRE(fork_b.ok());
    auto child_a = fork_a.value();
    auto child_b = fork_b.value();

    CHECK(parent->logical_kv_sequence_id() != 0u);
    CHECK(child_a->logical_kv_sequence_id() != 0u);
    CHECK(child_b->logical_kv_sequence_id() != 0u);
    CHECK(child_a->logical_kv_sequence_id() != parent->logical_kv_sequence_id());
    CHECK(child_b->logical_kv_sequence_id() != parent->logical_kv_sequence_id());
    CHECK(child_a->logical_kv_sequence_id() != child_b->logical_kv_sequence_id());

    // Logical KV blocks are genuinely shared via KvCacheManager::fork_sequence.
    REQUIRE(engine.request_runtime() != nullptr);
    auto info_a = engine.request_runtime()->inspect_logical_sequence(child_a->logical_kv_sequence_id());
    REQUIRE(info_a.has_value());
    CHECK(info_a->num_tokens == expected_warm_tokens);
    CHECK(info_a->shared_block_count >= 3u);

    // Diverge branch A and branch B with distinct suffixes.
    auto gen_a = child_a->generate("india juliet kilo lima");
    auto gen_b = child_b->generate("mike november oscar papa");
    REQUIRE(gen_a.ok());
    REQUIRE(gen_b.ok());
    CHECK(gen_a.value().text != gen_b.value().text);
    CHECK(child_a->context_prompt() != child_b->context_prompt());
    CHECK(child_a->context_prompt().rfind(parent_prompt_before, 0) == 0u);
    CHECK(child_b->context_prompt().rfind(parent_prompt_before, 0) == 0u);

    // Neither branch mutated the parent or the sibling.
    CHECK(parent->context_prompt() == parent_prompt_before);
    CHECK(parent->context_tokens() == parent_tokens_before);

    // Fork at explicit prefix token offset (6 tokens) and rewind parent to offset 4.
    ForkOptions prefix_fork_opts;
    prefix_fork_opts.prefix_token_offset = 6u;
    auto fork_prefix = parent->fork(prefix_fork_opts);
    REQUIRE(fork_prefix.ok());
    auto child_prefix = fork_prefix.value();
    CHECK(child_prefix->context_token_count() == 6u);
    CHECK(child_prefix->context_tokens() ==
          std::vector<TokenId>(parent_tokens_before.begin(), parent_tokens_before.begin() + 6));
    auto prefix_info = engine.request_runtime()->inspect_logical_sequence(child_prefix->logical_kv_sequence_id());
    REQUIRE(prefix_info.has_value());
    CHECK(prefix_info->num_tokens == 6u);

    // Out-of-bounds fork/rewind fails cleanly without corrupting state.
    ForkOptions bad_fork;
    bad_fork.prefix_token_offset = 999u;
    CHECK(parent->fork(bad_fork).status().code() == ErrorCode::invalid_argument);
    CHECK(parent->rewind(999u).code() == ErrorCode::invalid_argument);

    REQUIRE(parent->rewind(4u).ok());
    CHECK(parent->context_token_count() == 4u);
    CHECK(parent->context_tokens() ==
          std::vector<TokenId>(parent_tokens_before.begin(), parent_tokens_before.begin() + 4));
    auto rewound_info = engine.request_runtime()->inspect_logical_sequence(parent->logical_kv_sequence_id());
    REQUIRE(rewound_info.has_value());
    CHECK(rewound_info->num_tokens == 4u);

    // Closing every session reclaims all logical KV sequences cleanly.
    child_a->close();
    child_b->close();
    child_prefix->close();
    parent->close();
    CHECK(engine.kv_usage().sequences == 0u);
    CHECK(engine.kv_usage().pinned_blocks == 0u);

    // Hybrid/recurrent models cannot arbitrarily rewind KV past a non-checkpointed offset.
    {
        Engine e_hyb(make_test_engine_options(4, 64));
        MockBackendOptions hmo;
        hmo.token_logits = true;
        hmo.enable_recurrent_checkpoints = true;
        hmo.enable_execution_checkpoints = true;
        hmo.default_architecture = ModelArchitecture::hybrid;
        REQUIRE(e_hyb.register_backend(make_mock_backend(hmo)).ok());
        auto hm = e_hyb.load_model("mock", ModelLoadOptions{"mock:hybrid"}).value();
        auto hs = e_hyb.create_session(hm).value();
        SamplingConfig hsc;
        hsc.max_tokens = 4;
        REQUIRE(hs->generate("one two three four five six seven eight", {}, hsc).ok());
        // Offset 5 is not block/checkpoint-aligned on a hybrid architecture -> rejected cleanly.
        Status rew = hs->rewind(5u);
        CHECK_FALSE(rew.ok());
        hs->close();
    }
}

TEST_CASE("execution lifecycle: Branch checkpoint resets sampling and grammar state and supports multi-branch fanout") {
    Engine engine(make_test_engine_options(4, 64));
    MockBackendOptions mo;
    mo.token_logits = true;
    mo.enable_execution_checkpoints = true;
    mo.enable_selective_preemption = true;
    REQUIRE(engine.register_backend(make_mock_backend(mo)).ok());
    auto model = engine.load_model("mock", ModelLoadOptions{"mock:tiny"});
    REQUIRE(model.ok());

    auto s_parent = engine.create_session(model.value()).value();
    SamplingConfig warm_cfg;
    warm_cfg.temperature = 0.0f;
    warm_cfg.max_tokens = 4;
    REQUIRE(s_parent->generate("one two three four five six", {}, warm_cfg).ok());

    CheckpointExportOptions exp;
    exp.mode = CheckpointMode::Branch;
    exp.token_offset = 6u;
    auto handle_res = s_parent->export_checkpoint(exp);
    REQUIRE(handle_res.ok());
    const ExecutionCheckpointHandle handle = handle_res.value();
    CHECK(handle.valid());
    CHECK(handle.mode == CheckpointMode::Branch);
    CHECK(handle.same_process_only);
    CHECK(handle.token_position == 6u);
    CHECK(handle.delivered_offset_bytes == 0u);
    CHECK(handle.delivered_chunks == 0u);
    CHECK(engine.checkpoint_store().stats().entries == 1u);

    // Two different sessions branch from the same Branch handle with fresh sampling & grammar reset.
    ParityGrammarAutomaton gram;
    gram.step = 9;  // deliberately non-zero before Branch resume

    auto s_branch1 = engine.create_session(model.value()).value();
    ResumeOptions r1;
    r1.branch_prompt_suffix = "north path";
    r1.sampling_override = SamplingConfig{};
    r1.sampling_override->temperature = 0.9f;
    r1.sampling_override->seed = 101u;
    r1.sampling_override->max_tokens = 6;
    r1.request_options.grammar_hooks = gram.hooks();
    auto res1 = s_branch1->resume_from_checkpoint(handle, {}, r1);
    REQUIRE(res1.ok());
    CHECK(gram.reset_calls == 1u);

    auto s_branch2 = engine.create_session(model.value()).value();
    ResumeOptions r2;
    r2.branch_prompt_suffix = "south path";
    r2.sampling_override = SamplingConfig{};
    r2.sampling_override->temperature = 0.9f;
    r2.sampling_override->seed = 202u;
    r2.sampling_override->max_tokens = 6;
    r2.request_options.grammar_hooks = gram.hooks();
    auto res2 = s_branch2->resume_from_checkpoint(handle, {}, r2);
    REQUIRE(res2.ok());
    CHECK(gram.reset_calls == 2u);
    CHECK(res1.value().text != res2.value().text);

    CHECK(s_parent->release_checkpoint(handle));
    CHECK_FALSE(s_parent->release_checkpoint(handle));
    CHECK(engine.checkpoint_store().stats().entries == 0u);
    CHECK(engine.checkpoint_store().stats().total_bytes == 0u);

    s_branch1->close();
    s_branch2->close();
    s_parent->close();
    CHECK(engine.kv_usage().sequences == 0u);
}

TEST_CASE("execution lifecycle: ExactResume round-trip preserves RNG, penalties, grammar matcher, stop buffer, and offsets") {
    Engine engine(make_test_engine_options(4, 128));
    MockBackendOptions mo;
    mo.token_logits = true;
    mo.default_completion_tokens = 24;
    mo.enable_execution_checkpoints = true;
    mo.enable_selective_preemption = true;
    REQUIRE(engine.register_backend(make_mock_backend(mo)).ok());
    auto model = engine.load_model("mock", ModelLoadOptions{"mock:tiny"});
    REQUIRE(model.ok());

    const std::string prompt = "exact resume verification prompt alpha bravo charlie";
    SamplingConfig base_cfg;
    base_cfg.temperature = 0.85f;
    base_cfg.top_p = 0.92f;
    base_cfg.repeat_penalty = 1.25f;
    base_cfg.repeat_last_n = 16;
    base_cfg.seed = 424242u;
    base_cfg.max_tokens = 12;

    // 1. Uninterrupted baseline without stop sequence to observe exact token pieces.
    std::vector<std::string> raw_pieces;
    {
        ParityGrammarAutomaton base_gram;
        auto s = engine.create_session(model.value()).value();
        RequestOptions ro;
        ro.grammar = "parity-test-grammar";
        ro.grammar_hooks = base_gram.hooks();
        auto r = s->generate(
            prompt,
            [&](const TokenChunk& c) {
                raw_pieces.emplace_back(c.text);
                return true;
            },
            base_cfg, ro);
        REQUIRE(r.ok());
        REQUIRE(raw_pieces.size() >= 8u);
    }

    // Craft TWO stop-sequence scenarios that force non-empty StopSequenceMatcher::pending()
    // at suspend token 4:
    //   (a) False-alarm prefix: starts with the last 2 bytes of piece 3 + "ZZZ_NEVER_MATCH"
    //       so piece 3 leaves pending bytes buffered at token 4, then releases on resume!
    //   (b) True cross-boundary stop: starts with last 2 bytes of piece 3 + first 3 bytes of piece 4
    //       so suspending after token 4 holds the partial match, and resuming immediately triggers stop!
    REQUIRE(raw_pieces[3].size() >= 2u);
    REQUIRE(raw_pieces[4].size() >= 3u);
    const std::string partial_prefix = raw_pieces[3].substr(raw_pieces[3].size() - 2u);
    const std::string false_alarm_stop = partial_prefix + "ZZZ_NEVER_MATCH";
    const std::string cross_boundary_stop = partial_prefix + raw_pieces[4].substr(0, 3u);

    auto run_exact_resume_case = [&](const std::string& stop_seq, SuspendAction suspend_act, bool expect_stop_match) {
        SamplingConfig cfg = base_cfg;
        cfg.stop = {stop_seq};

        // Uninterrupted baseline with this exact stop sequence & grammar.
        ParityGrammarAutomaton base_gram;
        std::vector<std::uint64_t> base_indices;
        auto s_base = engine.create_session(model.value()).value();
        RequestOptions base_ro;
        base_ro.grammar = "parity-test-grammar";
        base_ro.grammar_hooks = base_gram.hooks();
        auto base_res = s_base->generate(
            prompt,
            [&](const TokenChunk& c) {
                base_indices.push_back(c.index);
                return true;
            },
            cfg, base_ro);
        REQUIRE(base_res.ok());

        // Interrupted run: suspend cleanly right after token 4 (while stop_filter_pending is non-empty!).
        ParityGrammarAutomaton run_gram;
        std::vector<std::uint64_t> split_indices;
        auto s_run = engine.create_session(model.value()).value();
        RequestOptions run_ro;
        run_ro.grammar = "parity-test-grammar";
        run_ro.grammar_hooks = run_gram.hooks();
        run_ro.on_safe_point = [&](const SafePointContext& sp) -> SuspendDecision {
            if (!sp.in_prefill && sp.generated_tokens == 4u) {
                return SuspendDecision{suspend_act, "test_boundary_suspend"};
            }
            return SuspendDecision{};
        };
        auto part1 = s_run->generate(
            prompt,
            [&](const TokenChunk& c) {
                split_indices.push_back(c.index);
                return true;
            },
            cfg, run_ro);
        REQUIRE(part1.ok());
        CHECK(part1.value().outcome == RequestOutcome::completed);
        CHECK(part1.value().stats.stop_reason == StopReason::suspended);
        CHECK(part1.value().scheduling.suspended);
        CHECK(part1.value().scheduling.physical_slot_reclaimed);
        CHECK(part1.value().scheduling.suspend_action == suspend_act);
        REQUIRE(part1.value().suspended_checkpoint.has_value());

        const ExecutionCheckpointHandle ckpt = *part1.value().suspended_checkpoint;
        auto stored_rec = engine.checkpoint_store().get(ckpt.checkpoint_id);
        REQUIRE(stored_rec.ok());
        CHECK(stored_rec.value().has_sampler_rng_state);
        CHECK_FALSE(stored_rec.value().sampler_penalty_history.empty());
        CHECK_FALSE(stored_rec.value().grammar_matcher_state.empty());
        CHECK_FALSE(stored_rec.value().stop_filter_pending.empty());
        if (suspend_act == SuspendAction::yield_spill) {
            CHECK_FALSE(stored_rec.value().backend_state_blob.empty());
        } else {
            CHECK(stored_rec.value().backend_state_blob.empty());
        }

        // Resume into a brand-new session with a fresh ParityGrammarAutomaton instance
        // (proving state comes from the checkpoint, not shared in-memory objects!).
        ParityGrammarAutomaton resume_gram;
        resume_gram.step = 0x0Eu;  // intentionally wrong until restore_state overwrites it
        auto s_resumed = engine.create_session(model.value()).value();
        ResumeOptions res_opts;
        res_opts.consume_checkpoint = true;
        res_opts.request_options.grammar_hooks = resume_gram.hooks();
        auto part2 = s_resumed->resume_from_checkpoint(
            ckpt,
            [&](const TokenChunk& c) {
                split_indices.push_back(c.index);
                return true;
            },
            res_opts);
        REQUIRE(part2.ok());
        CHECK(resume_gram.restore_calls == 1u);
        if (suspend_act == SuspendAction::yield_spill) {
            CHECK(part2.value().stats.cached_tokens.has_value());
        }

        // Byte-for-byte text parity, exact stop reason, exact delivered offset, contiguous chunk indices!
        const std::string combined = part1.value().text + part2.value().text;
        CHECK(combined == base_res.value().text);
        CHECK(part2.value().delivered_offset_bytes == base_res.value().text.size());
        CHECK(part2.value().delivered_chunks == base_res.value().delivered_chunks);
        CHECK(split_indices == base_indices);
        if (expect_stop_match) {
            CHECK(part2.value().stats.stop_reason == StopReason::stop_sequence);
            CHECK(part2.value().stats.matched_stop == stop_seq);
        } else {
            CHECK(part2.value().stats.stop_reason == base_res.value().stats.stop_reason);
        }
        // Consumed checkpoint is gone.
        CHECK_FALSE(engine.checkpoint_store().get(ckpt.checkpoint_id).ok());
    };

    run_exact_resume_case(false_alarm_stop, SuspendAction::yield_spill, /*expect_stop_match=*/false);
    run_exact_resume_case(false_alarm_stop, SuspendAction::yield_recompute, /*expect_stop_match=*/false);
    run_exact_resume_case(cross_boundary_stop, SuspendAction::yield_spill, /*expect_stop_match=*/true);

    // Also verify manual ExactResume export after StopReason::callback with pending_draft_tokens.
    {
        auto s_cb = engine.create_session(model.value()).value();
        int chunks = 0;
        auto cb_res = s_cb->generate(
            prompt,
            [&](const TokenChunk&) { return ++chunks < 3; },
            base_cfg);
        REQUIRE(cb_res.ok());
        CHECK(cb_res.value().stats.stop_reason == StopReason::callback);

        CheckpointExportOptions exp;
        exp.mode = CheckpointMode::ExactResume;
        exp.pending_draft_tokens = {7, 8, 9};
        auto h = s_cb->export_checkpoint(exp);
        REQUIRE(h.ok());
        auto rec = engine.checkpoint_store().get(h.value().checkpoint_id);
        REQUIRE(rec.ok());
        CHECK(rec.value().pending_draft_tokens == std::vector<TokenId>{7, 8, 9});
        CHECK(s_cb->release_checkpoint(h.value()));
    }

    // Non-token_logits MockModel ExactResume round-trip (spill + recompute).
    {
        Engine engine_plain(make_test_engine_options(4, 64));
        MockBackendOptions plain_mo;
        plain_mo.token_logits = false;
        plain_mo.default_completion_tokens = 12;
        plain_mo.enable_execution_checkpoints = true;
        plain_mo.enable_selective_preemption = true;
        REQUIRE(engine_plain.register_backend(make_mock_backend(plain_mo)).ok());
        auto m_plain = engine_plain.load_model("mock", ModelLoadOptions{"mock:tiny"}).value();

        SamplingConfig sc;
        sc.temperature = 0.7f;
        sc.seed = 99u;
        sc.max_tokens = 10;

        auto s_b = engine_plain.create_session(m_plain).value();
        auto baseline = s_b->generate("plain backend exact resume test", {}, sc);
        REQUIRE(baseline.ok());

        for (SuspendAction act : {SuspendAction::yield_spill, SuspendAction::yield_recompute}) {
            auto s1 = engine_plain.create_session(m_plain).value();
            RequestOptions ro;
            ro.on_safe_point = [&](const SafePointContext& sp) -> SuspendDecision {
                if (!sp.in_prefill && sp.generated_tokens == 4u) {
                    return SuspendDecision{act, "plain_suspend"};
                }
                return SuspendDecision{};
            };
            auto p1 = s1->generate("plain backend exact resume test", {}, sc, ro);
            REQUIRE(p1.ok());
            REQUIRE(p1.value().stats.stop_reason == StopReason::suspended);
            REQUIRE(p1.value().suspended_checkpoint.has_value());

            ResumeOptions r;
            r.consume_checkpoint = true;
            auto p2 = s1->resume_from_checkpoint(*p1.value().suspended_checkpoint, {}, r);
            REQUIRE(p2.ok());
            CHECK(p1.value().text + p2.value().text == baseline.value().text);
            CHECK(p2.value().delivered_offset_bytes == baseline.value().text.size());
        }
    }
}

TEST_CASE("execution lifecycle: ExactResume enforces same-process and model-build-tokenizer compatibility fingerprint") {
    Engine engine(make_test_engine_options(4, 64));
    MockBackendOptions mo1;
    mo1.token_logits = true;
    mo1.enable_execution_checkpoints = true;
    mo1.tokenizer_salt = 0x1111u;
    REQUIRE(engine.register_backend(make_mock_backend(mo1)).ok());

    auto model_a = engine.load_model("mock", ModelLoadOptions{"mock:tiny"}).value();
    auto model_b = engine.load_model("mock", ModelLoadOptions{"mock:small"}).value();

    // Second engine with different tokenizer salt.
    Engine engine_alt_tok(make_test_engine_options(4, 64));
    MockBackendOptions mo2 = mo1;
    mo2.tokenizer_salt = 0x2222u;
    REQUIRE(engine_alt_tok.register_backend(make_mock_backend(mo2)).ok());
    auto model_alt_tok = engine_alt_tok.load_model("mock", ModelLoadOptions{"mock:tiny"}).value();

    auto sess_a = engine.create_session(model_a).value();
    auto sess_b = engine.create_session(model_b).value();
    auto sess_alt_tok = engine_alt_tok.create_session(model_alt_tok).value();

    // Fingerprint sanity: distinct tokenizer salt changes tokenizer_hash & digest.
    const auto fp_a = sess_a->compatibility_fingerprint();
    const auto fp_alt = sess_alt_tok->compatibility_fingerprint();
    CHECK(fp_a.tokenizer_hash != fp_alt.tokenizer_hash);
    CHECK_FALSE(fp_a.compatible_with(fp_alt));
    CHECK(fp_a.mismatch_reason(fp_alt).find("tokenizer_hash") != std::string::npos);

    SamplingConfig sc;
    sc.max_tokens = 8;
    RequestOptions ro;
    ro.on_safe_point = [](const SafePointContext& sp) -> SuspendDecision {
        return (!sp.in_prefill && sp.generated_tokens == 3u)
                   ? SuspendDecision{SuspendAction::yield_spill, "fp_test"}
                   : SuspendDecision{};
    };
    auto paused = sess_a->generate("fingerprint isolation check", {}, sc, ro);
    REQUIRE(paused.ok());
    REQUIRE(paused.value().suspended_checkpoint.has_value());
    const ExecutionCheckpointHandle valid_handle = *paused.value().suspended_checkpoint;

    // 1. Different model name on same engine -> rejected.
    auto bad_model = sess_b->resume_from_checkpoint(valid_handle);
    CHECK_FALSE(bad_model.ok());
    CHECK(bad_model.status().code() == ErrorCode::invalid_argument);
    CHECK(bad_model.status().message().find("model_name") != std::string::npos);

    // 2. Different tokenizer fingerprint -> rejected.
    auto bad_tok = sess_alt_tok->resume_from_checkpoint(valid_handle);
    CHECK_FALSE(bad_tok.ok());
    CHECK(bad_tok.status().code() == ErrorCode::invalid_argument);

    // 3. Mutated build_commit -> rejected.
    ExecutionCheckpointHandle bad_build = valid_handle;
    bad_build.compatibility.build_commit = "00000000-incompatible-build";
    auto bad_build_res = sess_a->resume_from_checkpoint(bad_build);
    CHECK_FALSE(bad_build_res.ok());
    CHECK(bad_build_res.status().code() == ErrorCode::invalid_argument);
    CHECK(bad_build_res.status().message().find("build_commit") != std::string::npos);

    // 4. Mutated process_epoch -> rejected (strict same-process scope).
    ExecutionCheckpointHandle bad_proc = valid_handle;
    bad_proc.compatibility.process_epoch += 1u;
    auto bad_proc_res = sess_a->resume_from_checkpoint(bad_proc);
    CHECK_FALSE(bad_proc_res.ok());
    CHECK(bad_proc_res.status().code() == ErrorCode::invalid_argument);
    CHECK(bad_proc_res.status().message().find("process_epoch") != std::string::npos);

    // 5. Claiming cross-process portability (`same_process_only = false`) -> rejected.
    ExecutionCheckpointHandle bad_scope = valid_handle;
    bad_scope.same_process_only = false;
    auto bad_scope_res = sess_a->resume_from_checkpoint(bad_scope);
    CHECK_FALSE(bad_scope_res.ok());
    CHECK(bad_scope_res.status().code() == ErrorCode::invalid_argument);

    // 6. ExactResume rejects branch_prompt_suffix, sampling_override, and rewind token_offset.
    ResumeOptions with_suffix;
    with_suffix.branch_prompt_suffix = "cannot branch exact resume";
    CHECK(sess_a->resume_from_checkpoint(valid_handle, {}, with_suffix).status().code() == ErrorCode::invalid_argument);

    ResumeOptions with_sampling;
    with_sampling.sampling_override = SamplingConfig{};
    CHECK(sess_a->resume_from_checkpoint(valid_handle, {}, with_sampling).status().code() ==
          ErrorCode::invalid_argument);

    CheckpointExportOptions bad_exact_rewind;
    bad_exact_rewind.mode = CheckpointMode::ExactResume;
    bad_exact_rewind.token_offset = 1u;
    CHECK(sess_a->export_checkpoint(bad_exact_rewind).status().code() == ErrorCode::invalid_argument);

    CHECK(sess_a->release_checkpoint(valid_handle));
}

TEST_CASE("execution lifecycle: CheckpointStore bounded memory cap, LRU eviction, TTL expiry, and cancellation cleanup") {
    // 1. Deterministic CheckpointStore capacity, LRU, oversized rejection, and TTL expiry.
    auto make_rec = [](std::string owner, std::size_t blob_bytes, std::uint64_t kv_seq) {
        ExecutionCheckpointRecord rec;
        rec.handle.owner_session_id = std::move(owner);
        rec.handle.mode = CheckpointMode::ExactResume;
        rec.logical_kv_sequence_id = kv_seq;
        rec.backend_state_blob.assign(blob_bytes, 0xABu);
        return rec;
    };

    const std::size_t sample_bytes = make_rec("sess-1", 400, 101).accounted_bytes();
    auto fake_now = std::chrono::steady_clock::now();
    ExecutionCheckpointStoreOptions store_opts;
    store_opts.max_bytes = sample_bytes * 3u + 512u;
    store_opts.max_entries = 3u;
    store_opts.default_ttl = 500ms;

    CheckpointStore store(store_opts, [&]() { return fake_now; });
    std::vector<std::uint64_t> evicted_kv_seqs;
    store.set_evict_callback([&](const ExecutionCheckpointRecord& rec) {
        if (rec.logical_kv_sequence_id != 0) {
            evicted_kv_seqs.push_back(rec.logical_kv_sequence_id);
        }
    });

    auto h1 = store.put(make_rec("sess-1", 400, 101));
    auto h2 = store.put(make_rec("sess-1", 400, 102));
    auto h3 = store.put(make_rec("sess-2", 400, 103));
    REQUIRE(h1.ok());
    REQUIRE(h2.ok());
    REQUIRE(h3.ok());
    CHECK(store.stats().entries == 3u);

    // Touch h1 so h2 becomes LRU.
    REQUIRE(store.get(h1.value().checkpoint_id).ok());

    // Oversized single checkpoint fails without evicting existing entries.
    auto too_big = store.put(make_rec("sess-3", store_opts.max_bytes + 1024u, 999));
    CHECK_FALSE(too_big.ok());
    CHECK(too_big.status().code() == ErrorCode::invalid_argument);
    CHECK(store.stats().entries == 3u);
    CHECK(store.stats().evictions_capacity == 0u);

    // Inserting h4 triggers LRU capacity eviction of h2 (kv_seq 102).
    auto h4 = store.put(make_rec("sess-2", 400, 104));
    REQUIRE(h4.ok());
    CHECK(store.stats().entries == 3u);
    CHECK(store.stats().evictions_capacity == 1u);
    CHECK_FALSE(store.get(h2.value().checkpoint_id).ok());
    CHECK(store.get(h1.value().checkpoint_id).ok());
    REQUIRE(evicted_kv_seqs.size() == 1u);
    CHECK(evicted_kv_seqs[0] == 102u);

    // Advance clock past TTL -> sweep_expired reclaims all 3 expired entries & logical KV.
    fake_now += 600ms;
    CHECK(store.sweep_expired() == 3u);
    CHECK(store.get(h1.value().checkpoint_id).status().code() == ErrorCode::not_found);
    CHECK(store.stats().entries == 0u);
    CHECK(store.stats().total_bytes == 0u);
    CHECK(store.stats().evictions_expired == 3u);
    CHECK(evicted_kv_seqs.size() == 4u);

    // 2. Engine-wired TTL expiry, release on Session::close(), and cleanup on Session::cancel().
    EngineOptions eo = make_test_engine_options(4, 64);
    eo.checkpoints.max_bytes = 1u << 20;
    eo.checkpoints.default_ttl = 40ms;
    Engine engine(eo);
    MockBackendOptions mo;
    mo.token_logits = true;
    mo.token_delay = 5ms;
    mo.default_completion_tokens = 20;
    mo.enable_execution_checkpoints = true;
    REQUIRE(engine.register_backend(make_mock_backend(mo)).ok());
    auto model = engine.load_model("mock", ModelLoadOptions{"mock:tiny"}).value();

    // Short TTL expires cleanly on resume attempt.
    {
        auto s = engine.create_session(model).value();
        SamplingConfig sc;
        sc.max_tokens = 4;
        REQUIRE(s->generate("ttl expiry test prompt", {}, sc).ok());
        CheckpointExportOptions exp;
        exp.mode = CheckpointMode::Branch;
        exp.ttl = 25ms;
        auto h = s->export_checkpoint(exp);
        REQUIRE(h.ok());
        std::this_thread::sleep_for(60ms);
        auto expired_res = s->resume_from_checkpoint(h.value());
        CHECK_FALSE(expired_res.ok());
        CHECK(expired_res.status().code() == ErrorCode::not_found);
        s->close();
    }

    // Session::close() automatically releases all checkpoints owned by that session.
    {
        auto s = engine.create_session(model).value();
        SamplingConfig sc;
        sc.max_tokens = 4;
        REQUIRE(s->generate("close cleanup test prompt", {}, sc).ok());
        CheckpointExportOptions exp;
        exp.mode = CheckpointMode::Branch;
        exp.ttl = 10000ms;
        auto h1_sess = s->export_checkpoint(exp);
        auto h2_sess = s->export_checkpoint(exp);
        REQUIRE(h1_sess.ok());
        REQUIRE(h2_sess.ok());
        CHECK(engine.checkpoint_store().stats().entries == 2u);
        s->close();
        CHECK(engine.checkpoint_store().stats().entries == 0u);
        CHECK(engine.checkpoint_store().stats().total_bytes == 0u);
        CHECK(engine.kv_usage().sequences == 0u);
        CHECK(engine.kv_usage().pinned_blocks == 0u);
    }

    // Cancellation mid-stream leaves zero leaked checkpoints or stale ExactResume continuation.
    {
        auto s = engine.create_session(model).value();
        SamplingConfig sc;
        sc.max_tokens = 16;
        int seen = 0;
        auto cancelled = s->generate(
            "cancel cleanup verification prompt",
            [&](const TokenChunk&) {
                if (++seen == 2) {
                    s->cancel();
                }
                return true;
            },
            sc);
        REQUIRE(cancelled.ok());
        CHECK(cancelled.value().outcome == RequestOutcome::cancelled);
        CHECK_FALSE(cancelled.value().suspended_checkpoint.has_value());
        CHECK(engine.checkpoint_store().stats().entries == 0u);
        CheckpointExportOptions exact_exp;
        exact_exp.mode = CheckpointMode::ExactResume;
        CHECK(s->export_checkpoint(exact_exp).status().code() == ErrorCode::invalid_state);
        s->close();
        CHECK(engine.kv_usage().sequences == 0u);
    }
}

TEST_CASE("execution lifecycle: shared-worker selective preemption parks victim without stalling peer and resumes cleanly") {
    EngineOptions eo = make_test_engine_options(4, 128);
    eo.scheduling.enable_spill_preemption = true;
    Engine engine(eo);
    auto backend = std::make_shared<SharedWorkerBackend>(/*slots=*/2);
    REQUIRE(engine.register_backend(backend).ok());
    auto model = engine.load_model("shared-batch-mock", ModelLoadOptions{"shared:test"}).value();
    auto raw_model = backend->last_model();
    REQUIRE(raw_model != nullptr);

    SamplingConfig sc;
    sc.max_tokens = 12;
    sc.seed = 77u;

    const std::string prompt_a = "background worker long context alpha bravo";
    const std::string prompt_b = "interactive user latency sensitive request";

    // Uninterrupted baselines for A and B.
    const auto base_a = engine.create_session(model).value()->generate(prompt_a, {}, sc).value();
    const auto base_b = engine.create_session(model).value()->generate(prompt_b, {}, sc).value();
    REQUIRE(base_a.outcome == RequestOutcome::completed);
    REQUIRE(base_b.outcome == RequestOutcome::completed);

    SessionOptions opts_a;
    opts_a.workload = WorkloadClass::background_indexing;
    opts_a.sampling = sc;
    SessionOptions opts_b;
    opts_b.workload = WorkloadClass::interactive_user;
    opts_b.sampling = sc;

    auto sess_a = engine.create_session(model, opts_a).value();
    auto sess_b = engine.create_session(model, opts_b).value();

    std::atomic<bool> b_entered{false};
    std::atomic<bool> a_parked{false};
    std::atomic<bool> b_saw_slot_reclaimed{false};

    RequestOptions ro_a;
    ro_a.allow_preempt_spill = true;
    ro_a.on_safe_point = [&](const SafePointContext& sp) -> SuspendDecision {
        // Park A selectively once both A and B are sharing the single worker thread
        // and A has generated at least 4 tokens.
        if (sp.active_peers >= 1u && sp.generated_tokens >= 4u) {
            return SuspendDecision{SuspendAction::yield_spill, "selective_peer_yield"};
        }
        return SuspendDecision{};
    };

    RequestOptions ro_b;
    ro_b.on_safe_point = [&](const SafePointContext&) -> SuspendDecision {
        b_entered.store(true, std::memory_order_release);
        if (a_parked.load(std::memory_order_acquire) && raw_model->active_physical_slots() == 1u) {
            b_saw_slot_reclaimed.store(true, std::memory_order_release);
        }
        return SuspendDecision{};
    };

    Result<GenerationResult> res_a = Status(ErrorCode::internal, "not run");
    Result<GenerationResult> res_b = Status(ErrorCode::internal, "not run");

    std::thread th_a([&]() {
        res_a = sess_a->generate(prompt_a, {}, sc, ro_a);
        a_parked.store(true, std::memory_order_release);
    });
    std::thread th_b([&]() {
        res_b = sess_b->generate(prompt_b, {}, sc, ro_b);
    });

    th_a.join();
    th_b.join();

    REQUIRE(res_a.ok());
    REQUIRE(res_b.ok());

    // 1. Victim A suspended cleanly, reclaimed its physical slot, and auto-checkpointed.
    CHECK(res_a.value().outcome == RequestOutcome::completed);
    CHECK(res_a.value().stats.stop_reason == StopReason::suspended);
    CHECK(res_a.value().scheduling.suspended);
    CHECK(res_a.value().scheduling.physical_slot_reclaimed);
    CHECK(res_a.value().scheduling.suspend_action == SuspendAction::yield_spill);
    REQUIRE(res_a.value().suspended_checkpoint.has_value());
    CHECK(raw_model->peak_active_slots() == 2u);
    CHECK(raw_model->reclaimed_physical_slots() >= 1u);
    CHECK(raw_model->active_physical_slots() == 0u);

    // 2. Peer B finished ALL 12 tokens uninterrupted on the shared worker thread!
    CHECK(res_b.value().outcome == RequestOutcome::completed);
    CHECK(res_b.value().stats.stop_reason == StopReason::max_tokens);
    CHECK_FALSE(res_b.value().scheduling.suspended);
    CHECK(res_b.value().text == base_b.text);
    CHECK(raw_model->peer_steps_after_reclaim() >= 1u);

    // 3. Victim A resumes from its ExactResume checkpoint and matches baseline byte-for-byte!
    ResumeOptions resume_a;
    resume_a.consume_checkpoint = true;
    auto res_a_part2 = sess_a->resume_from_checkpoint(*res_a.value().suspended_checkpoint, {}, resume_a);
    REQUIRE(res_a_part2.ok());
    CHECK(res_a_part2.value().outcome == RequestOutcome::completed);
    CHECK(res_a_part2.value().stats.stop_reason == StopReason::max_tokens);
    CHECK(res_a_part2.value().stats.cached_tokens.has_value());
    CHECK(res_a.value().text + res_a_part2.value().text == base_a.text);
    CHECK(res_a_part2.value().delivered_offset_bytes == base_a.text.size());
    CHECK(res_a_part2.value().delivered_chunks == base_a.delivered_chunks);

    // 4. Asynchronous caller-triggered Session::request_suspend() parks cleanly.
    {
        Engine e_async(make_test_engine_options(4, 64));
        MockBackendOptions mo;
        mo.token_logits = true;
        mo.token_delay = 5ms;
        mo.default_completion_tokens = 20;
        mo.enable_execution_checkpoints = true;
        mo.enable_selective_preemption = true;
        REQUIRE(e_async.register_backend(make_mock_backend(mo)).ok());
        auto m_async = e_async.load_model("mock", ModelLoadOptions{"mock:tiny"}).value();
        auto s_async = e_async.create_session(m_async).value();

        SamplingConfig sc_async;
        sc_async.temperature = 0.0f;
        sc_async.max_tokens = 12;
        const auto base_async = e_async.create_session(m_async).value()->generate("async suspend test", {}, sc_async).value();

        int seen_chunks = 0;
        auto suspended = s_async->generate(
            "async suspend test",
            [&](const TokenChunk&) {
                if (++seen_chunks == 3) {
                    s_async->request_suspend(SuspendAction::yield_spill);
                }
                return true;
            },
            sc_async);
        REQUIRE(suspended.ok());
        CHECK(suspended.value().stats.stop_reason == StopReason::suspended);
        CHECK(suspended.value().scheduling.physical_slot_reclaimed);
        REQUIRE(suspended.value().suspended_checkpoint.has_value());

        ResumeOptions ra;
        ra.consume_checkpoint = true;
        auto resumed = s_async->resume_from_checkpoint(*suspended.value().suspended_checkpoint, {}, ra);
        REQUIRE(resumed.ok());
        CHECK(suspended.value().text + resumed.value().text == base_async.text);
    }
}
