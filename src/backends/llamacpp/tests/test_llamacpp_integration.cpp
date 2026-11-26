// Real-model integration and qualification test suite. Runs only when
// SONDER_TEST_GGUF names a GGUF file; otherwise exits with the CTest skip code
// (77) so CTest reports it as skipped rather than passed. Never commit weights.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/engine.hpp"

using namespace sonder::backends::llamacpp;
namespace si = sonder::inference;

namespace {

struct RunOutput {
    GenerateResult result;
    std::vector<Token> tokens;
    std::string text;
};

struct ScopedThreadJoiner {
    std::vector<std::thread>& threads;
    ~ScopedThreadJoiner() {
        for (std::thread& t : threads) {
            if (t.joinable()) t.join();
        }
    }
};

const char* ModelPath() { return std::getenv("SONDER_TEST_GGUF"); }

bool EqualsAsciiIgnoreCase(std::string_view a, std::string_view b) {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<unsigned char>(a[i]);
        const auto cb = static_cast<unsigned char>(b[i]);
        if (std::tolower(ca) != std::tolower(cb)) return false;
    }
    return true;
}

bool ContainsAsciiIgnoreCase(std::string_view haystack, std::string_view needle) {
    if (needle.empty() || needle.size() > haystack.size()) return false;
    for (std::size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        if (EqualsAsciiIgnoreCase(haystack.substr(i, needle.size()), needle)) return true;
    }
    return false;
}

bool IsEnvTruthy(std::string_view v) {
    return v == "1" || EqualsAsciiIgnoreCase(v, "true") || EqualsAsciiIgnoreCase(v, "yes") ||
           EqualsAsciiIgnoreCase(v, "on");
}

bool IsEnvFalsy(std::string_view v) {
    return v == "0" || EqualsAsciiIgnoreCase(v, "false") || EqualsAsciiIgnoreCase(v, "no") ||
           EqualsAsciiIgnoreCase(v, "off");
}

LoadOptions MakeLoadOptions(const char* model_path, std::uint32_t n_ctx = 512, std::uint32_t n_batch = 64) {
    LoadOptions opts;
    opts.model_path = model_path != nullptr ? model_path : "";
    opts.n_ctx = n_ctx;
    opts.n_batch = n_batch;
    if (const char* t = std::getenv("SONDER_TEST_THREADS"); t != nullptr && t[0] != '\0') {
        opts.n_threads = std::atoi(t);
    }
    if (const char* g = std::getenv("SONDER_TEST_GPU_LAYERS"); g != nullptr && g[0] != '\0') {
        opts.n_gpu_layers = std::atoi(g);
    }
    return opts;
}

RunOutput RunRequest(LlamaCppBackend& be, const GenerateRequest& req, const CancelPredicate& cancel = {},
                     int stop_after = -1) {
    RunOutput out;
    out.result = be.Generate(
        req,
        [&](const TokenEvent& ev) {
            if (ev.token >= 0) out.tokens.push_back(ev.token);
            out.text += ev.text;
            return stop_after < 0 || static_cast<int>(out.tokens.size()) < stop_after;
        },
        cancel);
    return out;
}

RunOutput Run(LlamaCppBackend& be, const std::vector<Token>& prompt, int max_new,
              const CancelPredicate& cancel = {}, int stop_after = -1) {
    GenerateRequest req;
    req.prompt = prompt;
    req.max_new_tokens = max_new;
    return RunRequest(be, req, cancel, stop_after);
}

std::vector<Token> BuildMinLengthPrompt(const LlamaCppBackend& be, std::string_view header, std::string_view unit,
                                        std::size_t min_tokens) {
    std::string text(header);
    std::vector<Token> tokens;
    for (int round = 1; round <= 256; ++round) {
        text += "Section " + std::to_string(round) + ": ";
        text += unit;
        text += "\n";
        tokens.clear();
        if (!be.Tokenize(text, true, tokens).ok()) return {};
        if (tokens.size() >= min_tokens) return tokens;
    }
    return tokens;
}

std::string PickStopSubstring(const std::string& baseline_text) {
    if (baseline_text.size() >= 8u) {
        const std::size_t mid = baseline_text.size() / 2u;
        const std::string candidate = baseline_text.substr(mid, 3u);
        if (!candidate.empty()) return candidate;
    }
    if (baseline_text.find('.') != std::string::npos) return ".";
    if (baseline_text.size() >= 2u) return baseline_text.substr(baseline_text.size() / 2u, 1u);
    return ".";
}

}  // namespace

TEST_CASE("llamacpp integration: wrapper prefill, streaming decode, cancellation") {
    const char* path = ModelPath();

    LlamaCppBackend be;
    int prefill_events = 0;
    int token_events = 0;
    int done_events = 0;
    be.SetTelemetrySink([&](const TelemetryEvent& e) {
        if (e.kind == TelemetryKind::kPrefillDone) ++prefill_events;
        if (e.kind == TelemetryKind::kTokenDecoded) ++token_events;
        if (e.kind == TelemetryKind::kGenerationDone) ++done_events;
    });

    LoadOptions opts;
    opts.model_path = path;
    opts.n_ctx = 512;
    opts.n_batch = 64;  // force multiple prefill chunks for longer prompts
    if (const char* t = std::getenv("SONDER_TEST_THREADS")) opts.n_threads = std::atoi(t);
    const auto load_start = std::chrono::steady_clock::now();
    const Status s = be.Load(opts);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    if (!s.ok()) std::fprintf(stderr, "load failed: %s\n", s.message.c_str());
    REQUIRE(s.ok());
    const ModelInfo info = be.GetModelInfo();
    std::printf("[llamacpp_integration] model: %s | %.1f MiB | %llu params | vocab %d | ctx %u | load %.1f ms\n",
                info.description.c_str(), static_cast<double>(info.size_bytes) / (1024.0 * 1024.0),
                static_cast<unsigned long long>(info.n_params), info.n_vocab, info.n_ctx, load_ms);

    std::vector<Token> prompt;
    REQUIRE(be.Tokenize("Once upon a time, there was a little", true, prompt).ok());
    REQUIRE(!prompt.empty());

    // 1. Greedy streaming decode.
    const int kMax = 64;
    RunOutput a = Run(be, prompt, kMax);
    CHECK(a.result.status.ok());
    CHECK(a.result.generated_tokens > 0);
    CHECK(a.result.prompt_tokens == static_cast<int>(prompt.size()));
    CHECK((a.result.stop_reason == StopReason::kMaxTokens ||
           a.result.stop_reason == StopReason::kEndOfGeneration));
    CHECK(static_cast<int>(a.tokens.size()) == a.result.generated_tokens);
    CHECK(prefill_events == 1);
    CHECK(token_events == a.result.generated_tokens);
    CHECK(done_events == 1);
    std::printf("[llamacpp_integration] prompt %d tok, generated %d tok (%s)\n", a.result.prompt_tokens,
                a.result.generated_tokens, std::string(ToString(a.result.stop_reason)).c_str());
    std::printf("[llamacpp_integration] prefill %.2f ms (%.1f tok/s), decode %.2f ms (%.1f tok/s)\n",
                a.result.prefill_ms, a.result.prefill_tokens_per_second(), a.result.decode_ms,
                a.result.decode_tokens_per_second());
    std::printf("[llamacpp_integration] text: %s\n", a.text.c_str());

    // 2. Greedy is deterministic across runs (context reset between calls).
    RunOutput b = Run(be, prompt, kMax);
    CHECK(b.tokens == a.tokens);

    // 3. Cancellation predicate is honoured between tokens.
    int polls = 0;
    RunOutput c = Run(be, prompt, kMax, [&] { return ++polls > 4; });
    CHECK(c.result.status.ok());
    CHECK(c.result.stop_reason == StopReason::kCancelled);
    CHECK(c.result.generated_tokens < kMax);

    // 4. Returning false from the token callback stops generation.
    RunOutput d = Run(be, prompt, kMax, {}, 3);
    CHECK((d.result.stop_reason == StopReason::kCancelled ||
           d.result.stop_reason == StopReason::kEndOfGeneration));
    CHECK(d.result.generated_tokens <= 3);
    if (d.result.stop_reason == StopReason::kCancelled) {
        CHECK(d.tokens == std::vector<Token>(a.tokens.begin(), a.tokens.begin() + 3));
    }

    // 5. max_new_tokens = 0 prefills only.
    RunOutput e = Run(be, prompt, 0);
    CHECK(e.result.stop_reason == StopReason::kMaxTokens);
    CHECK(e.result.generated_tokens == 0);

    // 6. Prompt larger than the context window is rejected up front.
    GenerateRequest big;
    big.prompt.assign(info.n_ctx + 1, prompt.back());
    CHECK(be.Generate(big, {}).status.code == ErrorCode::kContextOverflow);

    // 7. Sampled decode with a fixed seed is reproducible.
    GenerateRequest sreq;
    sreq.prompt = prompt;
    sreq.max_new_tokens = 16;
    sreq.sampling.temperature = 0.8F;
    sreq.sampling.seed = 1234;
    std::vector<Token> s1;
    std::vector<Token> s2;
    auto collect = [](std::vector<Token>& into) {
        return [&into](const TokenEvent& ev) {
            if (ev.token >= 0) into.push_back(ev.token);
            return true;
        };
    };
    be.Generate(sreq, collect(s1));
    be.Generate(sreq, collect(s2));
    CHECK(!s1.empty());
    CHECK(s1 == s2);

    be.Unload();
    CHECK(!be.IsLoaded());
}

TEST_CASE("llamacpp integration: KV cache types and flash attention load and decode") {
    struct Config {
        KvCacheType k;
        KvCacheType v;
        FlashAttention fa;
    };
    const Config configs[] = {
        {KvCacheType::kF16, KvCacheType::kF16, FlashAttention::kAuto},     // the default
        {KvCacheType::kF16, KvCacheType::kF16, FlashAttention::kDisabled},
        {KvCacheType::kQ8_0, KvCacheType::kQ8_0, FlashAttention::kAuto},   // llama.cpp enables FA
        {KvCacheType::kQ8_0, KvCacheType::kF16, FlashAttention::kDisabled},  // quantized K alone
        {KvCacheType::kQ4_0, KvCacheType::kQ4_0, FlashAttention::kEnabled},
    };
    for (const Config& c : configs) {
        CAPTURE(ToString(c.k));
        CAPTURE(ToString(c.v));
        CAPTURE(ToString(c.fa));
        LlamaCppBackend be;
        LoadOptions opts;
        opts.model_path = ModelPath();
        opts.n_ctx = 512;
        opts.n_batch = 64;
        opts.n_ubatch = 32;  // explicit micro-batch below the batch size
        opts.type_k = c.k;
        opts.type_v = c.v;
        opts.flash_attn = c.fa;
        if (const char* t = std::getenv("SONDER_TEST_THREADS")) opts.n_threads = std::atoi(t);
        const Status s = be.Load(opts);
        if (!s.ok()) std::fprintf(stderr, "load failed: %s\n", s.message.c_str());
        REQUIRE(s.ok());
        std::vector<Token> prompt;
        REQUIRE(be.Tokenize("Once upon a time, there was a little", true, prompt).ok());
        RunOutput out = Run(be, prompt, 16);
        CHECK(out.result.status.ok());
        CHECK(out.result.generated_tokens > 0);
        std::printf("[llamacpp_integration] kv k=%s v=%s fa=%s: %d tok, decode %.1f tok/s: %s\n",
                    std::string(ToString(c.k)).c_str(), std::string(ToString(c.v)).c_str(),
                    std::string(ToString(c.fa)).c_str(), out.result.generated_tokens,
                    out.result.decode_tokens_per_second(), out.text.c_str());
    }

    // A quantized V cache with flash attention off is refused before loading.
    LlamaCppBackend be;
    LoadOptions bad;
    bad.model_path = ModelPath();
    bad.type_v = KvCacheType::kQ8_0;
    bad.flash_attn = FlashAttention::kDisabled;
    CHECK(be.Load(bad).code == ErrorCode::kInvalidArgument);
    CHECK(!be.IsLoaded());
}

TEST_CASE("llamacpp integration: core Backend interface end to end") {
    si::LlamaCppBackendOptions opts;
    opts.context_length = 512;
    auto backend = si::make_llamacpp_backend(opts);
    si::ModelLoadOptions lo;
    lo.model = ModelPath();
    auto loaded = backend->load_model(lo);
    if (!loaded.ok()) std::fprintf(stderr, "load_model: %s\n", loaded.status().message().c_str());
    REQUIRE(loaded.ok());
    std::shared_ptr<si::BackendModel> model = loaded.value();
    CHECK(model->descriptor().format == "gguf");
    CHECK(model->descriptor().context_length == 512u);

    si::GenerateRequest req;
    req.request_id = "it-1";
    req.prompt = "Once upon a time";
    req.sampling = si::SamplingConfig::greedy(24, 7);
    std::string text;
    std::uint64_t chunks = 0;
    si::CancellationSource never;
    auto stats = model->generate(req, never.token(), [&](const si::TokenChunk& c) {
        CHECK(c.index == chunks);
        ++chunks;
        text.append(c.text);
        return true;
    });
    REQUIRE(stats.ok());
    CHECK(stats->token_counts_from_backend);
    CHECK(stats->completion_tokens > 0);
    CHECK(stats->chunks == chunks);
    CHECK(stats->prompt_eval_ns > 0);
    CHECK((stats->stop_reason == si::StopReason::max_tokens ||
           stats->stop_reason == si::StopReason::end_of_sequence));
    CHECK_FALSE(text.empty());

    // Stop sequence: stop at the first period; the stop text is not emitted.
    req.sampling.max_tokens = 64;
    req.sampling.stop = {"."};
    std::string stopped;
    auto s2 = model->generate(req, never.token(), [&](const si::TokenChunk& c) {
        stopped.append(c.text);
        return true;
    });
    REQUIRE(s2.ok());
    if (s2->stop_reason == si::StopReason::stop_sequence) {
        CHECK(stopped.find('.') == std::string::npos);
        CHECK(text.rfind(stopped, 0) == 0u);  // same greedy prefix
    }

    // Callback returning false -> StopReason::callback.
    req.sampling.stop.clear();
    auto s3 = model->generate(req, never.token(), [](const si::TokenChunk&) { return false; });
    REQUIRE(s3.ok());
    CHECK(s3->stop_reason == si::StopReason::callback);
    CHECK(s3->chunks == 1u);

    // Pre-cancelled token -> ErrorCode::cancelled.
    si::CancellationSource src;
    src.cancel();
    auto s4 = model->generate(req, src.token(), {});
    CHECK(s4.status().code() == si::ErrorCode::cancelled);

    // Cancellation mid-stream from the callback's thread.
    si::CancellationSource mid;
    int seen = 0;
    auto s5 = model->generate(req, mid.token(), [&](const si::TokenChunk&) {
        if (++seen == 2) mid.cancel();
        return true;
    });
    CHECK(s5.status().code() == si::ErrorCode::cancelled);
    CHECK(seen == 2);
}

TEST_CASE("llamacpp integration: Engine + Session with Observatory telemetry") {
    auto sink = std::make_shared<si::MemoryTelemetrySink>();
    si::EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    si::Engine engine(eo);
    si::LlamaCppBackendOptions bo;
    bo.context_length = 512;
    REQUIRE(engine.register_backend(si::make_llamacpp_backend(bo)).ok());

    si::ModelLoadOptions lo;
    lo.model = ModelPath();
    auto model = engine.load_model(si::kLlamaCppBackendName, lo);
    REQUIRE(model.ok());
    si::SessionOptions so;
    so.sampling = si::SamplingConfig::greedy(16, 1);
    auto session = engine.create_session(model.value(), so);
    REQUIRE(session.ok());

    auto r = session.value()->generate("Once upon a time");
    REQUIRE(r.ok());
    CHECK(r->outcome == si::RequestOutcome::completed);
    CHECK(r->stats.completion_tokens > 0);
    CHECK(r->ttft_ms >= 0.0);
    CHECK_FALSE(r->text.empty());
    std::printf("[llamacpp_integration] session: %llu tok in %.2f ms, ttft %.2f ms\n",
                static_cast<unsigned long long>(r->stats.completion_tokens), r->total_ms, r->ttft_ms);

    // Cancel from inside the chunk callback: partial result, outcome cancelled.
    si::Session& sess = *session.value();
    auto c = sess.generate("Once upon a time", [&](const si::TokenChunk& chunk) {
        if (chunk.index == 1u) sess.cancel();
        return true;
    });
    REQUIRE(c.ok());
    CHECK(c->outcome == si::RequestOutcome::cancelled);

    CHECK(engine.unload_model(model.value()->instance_id()).ok());
    engine.telemetry().flush();
    CHECK_FALSE(sink->lines().empty());
}

TEST_CASE("llamacpp integration: ngram-mod and suffix speculation qualification") {
    constexpr std::uint32_t kCtx = 1024;
    constexpr std::uint32_t kBatch = 128;
    constexpr int kMaxNew = 48;

    // 1. Baseline ("none") greedy reference on a structured, repeatable prompt.
    LlamaCppBackend base_be;
    LoadOptions base_opts = MakeLoadOptions(ModelPath(), kCtx, kBatch);
    base_opts.prefix_cache = false;
    base_opts.spec_type = "none";
    REQUIRE(base_be.Load(base_opts).ok());
    const ModelInfo base_info = base_be.GetModelInfo();

    const std::string prompt_text =
        "Repeat and continue this structured configuration block verbatim:\n"
        "item_01: status=ready, worker=alpha, retries=3, mode=strict\n"
        "item_02: status=ready, worker=beta, retries=3, mode=strict\n"
        "item_03: status=ready, worker=gamma, retries=3, mode=strict\n"
        "item_04: status=ready, worker=delta, retries=3, mode=strict\n"
        "item_05: status=ready, worker=epsilon, retries=3, mode=strict\n"
        "item_06: status=";
    std::vector<Token> prompt;
    REQUIRE(base_be.Tokenize(prompt_text, true, prompt).ok());
    REQUIRE(prompt.size() >= 32u);

    const RunOutput baseline = Run(base_be, prompt, kMaxNew);
    REQUIRE(baseline.result.status.ok());
    REQUIRE(baseline.result.generated_tokens > 0);
    CHECK(static_cast<int>(baseline.tokens.size()) == baseline.result.generated_tokens);
    CHECK(baseline.result.draft_tokens == 0);
    CHECK(baseline.result.draft_accepted_tokens == 0);
    CHECK((baseline.result.stop_reason == StopReason::kMaxTokens ||
           baseline.result.stop_reason == StopReason::kEndOfGeneration));
    for (Token tok : baseline.tokens) {
        CHECK_FALSE(base_be.IsEndOfGeneration(tok));
    }
    base_be.Unload();

    const char* modes[] = {"ngram-mod", "suffix"};
    for (const char* mode : modes) {
        CAPTURE(mode);
        LlamaCppBackend spec_be;
        LoadOptions spec_opts = MakeLoadOptions(ModelPath(), kCtx, kBatch);
        spec_opts.prefix_cache = false;
        spec_opts.spec_type = mode;
        spec_opts.spec_draft_max = 8;
        REQUIRE(spec_be.Load(spec_opts).ok());

        // Pass 1 (cold speculation state): greedy output must match baseline.
        const RunOutput pass1 = Run(spec_be, prompt, kMaxNew);
        REQUIRE(pass1.result.status.ok());
        CHECK(pass1.result.stop_reason == baseline.result.stop_reason);
        CHECK(static_cast<int>(pass1.tokens.size()) == pass1.result.generated_tokens);
        CHECK(pass1.result.draft_tokens >= 0);
        CHECK(pass1.result.draft_accepted_tokens >= 0);
        CHECK(pass1.result.draft_accepted_tokens <= pass1.result.draft_tokens);
        CHECK(pass1.tokens == baseline.tokens);
        CHECK(pass1.text == baseline.text);
        for (Token tok : pass1.tokens) {
            CHECK_FALSE(spec_be.IsEndOfGeneration(tok));
        }

        // Pass 2 (warmed cross-request session ring / persistent ngram-mod table):
        // prefix_cache is disabled so full decode runs, verifying proposed vs
        // committed token accounting and exact greedy equivalence.
        const RunOutput pass2 = Run(spec_be, prompt, kMaxNew);
        REQUIRE(pass2.result.status.ok());
        CHECK(pass2.result.stop_reason == baseline.result.stop_reason);
        CHECK(static_cast<int>(pass2.tokens.size()) == pass2.result.generated_tokens);
        CHECK(pass2.result.draft_tokens >= 0);
        CHECK(pass2.result.draft_accepted_tokens >= 0);
        CHECK(pass2.result.draft_accepted_tokens <= pass2.result.draft_tokens);
        CHECK(pass2.result.draft_accepted_tokens <= pass2.result.generated_tokens);
        CHECK(pass2.tokens == baseline.tokens);
        CHECK(pass2.text == baseline.text);
        if (baseline.result.generated_tokens >= 12 &&
            base_info.architecture != si::ModelArchitecture::recurrent) {
            CHECK(pass2.result.draft_tokens > 0);
            CHECK(pass2.result.draft_accepted_tokens > 0);
        }
        std::printf("[llamacpp_integration] spec=%s: committed=%d proposed=%d accepted=%d stop=%s\n", mode,
                    pass2.result.generated_tokens, pass2.result.draft_tokens, pass2.result.draft_accepted_tokens,
                    std::string(ToString(pass2.result.stop_reason)).c_str());

        // Callback cancellation mid-speculative-verification preserves exact prefix.
        if (baseline.tokens.size() >= 4u) {
            const RunOutput cb_cancel = Run(spec_be, prompt, kMaxNew, {}, /*stop_after=*/3);
            CHECK(cb_cancel.result.status.ok());
            CHECK(cb_cancel.result.stop_reason == StopReason::kCancelled);
            CHECK(cb_cancel.result.generated_tokens == 3);
            CHECK(cb_cancel.tokens == std::vector<Token>(baseline.tokens.begin(), baseline.tokens.begin() + 3));
        }

        // CancelPredicate cancellation terminates cleanly without corrupting state.
        int polls = 0;
        const RunOutput pred_cancel = Run(spec_be, prompt, kMaxNew, [&] { return ++polls > 2; });
        CHECK(pred_cancel.result.status.ok());
        CHECK(pred_cancel.result.stop_reason == StopReason::kCancelled);
        CHECK(pred_cancel.result.generated_tokens < kMaxNew);

        // Post-cancellation follow-up verifies rejected speculative KV rollback left
        // the context completely clean.
        const RunOutput recovered = Run(spec_be, prompt, kMaxNew);
        REQUIRE(recovered.result.status.ok());
        CHECK(recovered.tokens == baseline.tokens);
        spec_be.Unload();

        // Core BackendModel stop-string & CancellationToken qualification under speculation.
        si::LlamaCppBackendOptions core_opts;
        core_opts.context_length = kCtx;
        core_opts.batch_size = kBatch;
        core_opts.prefix_cache = false;
        core_opts.spec_type = mode;
        core_opts.spec_draft_max = 8;
        if (const char* t = std::getenv("SONDER_TEST_THREADS"); t != nullptr && t[0] != '\0') {
            core_opts.threads = std::atoi(t);
        }
        auto backend = si::make_llamacpp_backend(core_opts);
        si::ModelLoadOptions lo;
        lo.model = ModelPath();
        auto loaded = backend->load_model(lo);
        REQUIRE(loaded.ok());
        std::shared_ptr<si::BackendModel> model = loaded.value();

        si::GenerateRequest req;
        req.request_id = std::string("spec-warm-") + mode;
        req.prompt = prompt_text;
        req.sampling = si::SamplingConfig::greedy(static_cast<std::uint32_t>(kMaxNew), 7);
        si::CancellationSource never;
        std::string full_text;
        auto warm_stats = model->generate(req, never.token(), [&](const si::TokenChunk& c) {
            full_text.append(c.text);
            return true;
        });
        REQUIRE(warm_stats.ok());
        CHECK(warm_stats->completion_tokens > 0u);
        if (warm_stats->draft_tokens.has_value()) {
            CHECK(warm_stats->draft_accepted_tokens.has_value());
            if (warm_stats->draft_accepted_tokens.has_value()) {
                CHECK(*warm_stats->draft_accepted_tokens <= *warm_stats->draft_tokens);
            }
        }

        const std::string stop_str = PickStopSubstring(full_text);
        req.request_id = std::string("spec-stop-") + mode;
        req.sampling.stop = {stop_str};
        std::string stopped_text;
        auto stop_stats = model->generate(req, never.token(), [&](const si::TokenChunk& c) {
            stopped_text.append(c.text);
            return true;
        });
        REQUIRE(stop_stats.ok());
        if (full_text.find(stop_str) != std::string::npos) {
            CHECK(stop_stats->stop_reason == si::StopReason::stop_sequence);
            CHECK(stopped_text.find(stop_str) == std::string::npos);
            CHECK(full_text.rfind(stopped_text, 0) == 0u);
        }

        // Core CancellationSource mid-stream under warmed speculation.
        req.request_id = std::string("spec-cancel-") + mode;
        req.sampling.stop.clear();
        si::CancellationSource mid;
        int seen = 0;
        auto cancel_stats = model->generate(req, mid.token(), [&](const si::TokenChunk&) {
            if (++seen == 2) mid.cancel();
            return true;
        });
        if (full_text.size() >= 2u && warm_stats->chunks >= 2u) {
            CHECK(cancel_stats.status().code() == si::ErrorCode::cancelled);
            CHECK(seen == 2);
        }
    }
}

TEST_CASE("llamacpp integration: optional draft-mtp qualification") {
    const char* mtp_env = std::getenv("SONDER_TEST_GGUF_MTP");
    const std::string_view mtp_val = mtp_env != nullptr ? std::string_view(mtp_env) : std::string_view{};

    LoadOptions probe_opts = MakeLoadOptions(ModelPath(), 512, 64);
    probe_opts.spec_type = "draft-mtp";
    const ModelParamsSummary mtp_params = LlamaCppBackend::ModelParamsFor(probe_opts);

    if (!mtp_val.empty() && IsEnvFalsy(mtp_val)) {
        std::printf("[llamacpp_integration] SKIP (draft-mtp): disabled by SONDER_TEST_GGUF_MTP=%s\n", mtp_env);
        return;
    }

#if !defined(SONDER_LLAMACPP_HAS_MTP)
    CHECK_FALSE(mtp_params.load_mtp);
    // Keep metadata helpers exercised cleanly even when MTP is compiled out.
    (void)IsEnvTruthy(mtp_val);
    (void)ContainsAsciiIgnoreCase(probe_opts.model_path, "mtp");
    std::printf("[llamacpp_integration] SKIP (draft-mtp): build was compiled without SONDER_LLAMACPP_HAS_MTP\n");
    return;
#else
    CHECK(mtp_params.load_mtp);

    std::string mtp_model_path;
    if (!mtp_val.empty()) {
        mtp_model_path = IsEnvTruthy(mtp_val) ? std::string(ModelPath()) : std::string(mtp_val);
    } else {
        LlamaCppBackend vocab_probe;
        LoadOptions vopts = MakeLoadOptions(ModelPath(), 512, 64);
        vopts.vocab_only = true;
        REQUIRE(vocab_probe.Load(vopts).ok());
        const ModelInfo minfo = vocab_probe.GetModelInfo();
        vocab_probe.Unload();
        if (ContainsAsciiIgnoreCase(minfo.description, "mtp") ||
            ContainsAsciiIgnoreCase(vopts.model_path, "mtp")) {
            mtp_model_path = ModelPath();
        }
    }

    if (mtp_model_path.empty()) {
        std::printf(
            "[llamacpp_integration] SKIP (draft-mtp): SONDER_TEST_GGUF_MTP not set and model metadata does not "
            "indicate MTP support\n");
        return;
    }

    constexpr std::uint32_t kCtx = 1024;
    constexpr std::uint32_t kBatch = 128;
    constexpr int kMaxNew = 32;

    LlamaCppBackend base_be;
    LoadOptions base_opts = MakeLoadOptions(mtp_model_path.c_str(), kCtx, kBatch);
    base_opts.prefix_cache = false;
    base_opts.spec_type = "none";
    REQUIRE(base_be.Load(base_opts).ok());
    std::vector<Token> prompt;
    REQUIRE(base_be.Tokenize("Once upon a time, in a quiet laboratory, a small engineer wrote", true, prompt).ok());
    const RunOutput base_out = Run(base_be, prompt, kMaxNew);
    REQUIRE(base_out.result.status.ok());
    REQUIRE(base_out.result.generated_tokens > 0);
    base_be.Unload();

    LlamaCppBackend mtp_be;
    LoadOptions mtp_opts = MakeLoadOptions(mtp_model_path.c_str(), kCtx, kBatch);
    mtp_opts.prefix_cache = false;
    mtp_opts.spec_type = "draft-mtp";
    mtp_opts.spec_draft_max = 3;
    REQUIRE(mtp_be.Load(mtp_opts).ok());

    const RunOutput mtp_out = Run(mtp_be, prompt, kMaxNew);
    REQUIRE(mtp_out.result.status.ok());
    CHECK(static_cast<int>(mtp_out.tokens.size()) == mtp_out.result.generated_tokens);
    CHECK(mtp_out.result.draft_accepted_tokens >= 0);
    CHECK(mtp_out.result.draft_accepted_tokens <= mtp_out.result.draft_tokens);
    CHECK(mtp_out.tokens == base_out.tokens);
    if (base_out.result.generated_tokens >= 4) {
        CHECK(mtp_out.result.draft_tokens > 0);
    }
    std::printf("[llamacpp_integration] spec=draft-mtp: committed=%d proposed=%d accepted=%d\n",
                mtp_out.result.generated_tokens, mtp_out.result.draft_tokens, mtp_out.result.draft_accepted_tokens);

    if (base_out.tokens.size() >= 4u) {
        const RunOutput cancelled = Run(mtp_be, prompt, kMaxNew, {}, /*stop_after=*/3);
        CHECK(cancelled.result.status.ok());
        CHECK(cancelled.result.stop_reason == StopReason::kCancelled);
        CHECK(cancelled.tokens == std::vector<Token>(base_out.tokens.begin(), base_out.tokens.begin() + 3));
        const RunOutput after = Run(mtp_be, prompt, kMaxNew);
        REQUIRE(after.result.status.ok());
        CHECK(after.tokens == base_out.tokens);
    }
    mtp_be.Unload();
#endif
}

TEST_CASE("llamacpp integration: multi-slot concurrent batching, staggered arrivals, and long trailing request") {
    constexpr std::uint32_t kCtx = 2048;
    constexpr std::uint32_t kBatch = 128;
    constexpr std::uint32_t kSlots = 4;
    constexpr int kPeerTokens = 16;
    constexpr int kShortTokens = 6;
    constexpr int kTrailingTokens = 36;

    const std::string prompt_texts[3] = {
        "Alpha worker report: describe the mountain weather at sunrise in a clear sentence.",
        "Beta worker report: explain why unit tests catch regressions before shipping code.",
        "Gamma worker report: list three steps for inspecting a cache miss in a server.",
    };

    // Single-slot reference baselines for exactness and zero cross-slot contamination.
    LlamaCppBackend ref_be;
    LoadOptions ref_opts = MakeLoadOptions(ModelPath(), kCtx, kBatch);
    ref_opts.n_slots = 1;
    ref_opts.prefix_cache = false;
    ref_opts.spec_type = "none";
    REQUIRE(ref_be.Load(ref_opts).ok());

    std::vector<Token> prompts[3];
    RunOutput ref_peer[3];
    for (int i = 0; i < 3; ++i) {
        REQUIRE(ref_be.Tokenize(prompt_texts[i], true, prompts[i]).ok());
        REQUIRE(!prompts[i].empty());
        ref_peer[i] = Run(ref_be, prompts[i], kPeerTokens);
        REQUIRE(ref_peer[i].result.status.ok());
        REQUIRE(ref_peer[i].result.generated_tokens > 0);
    }
    const RunOutput ref_trailing = Run(ref_be, prompts[2], kTrailingTokens);
    REQUIRE(ref_trailing.result.status.ok());
    REQUIRE(ref_trailing.result.generated_tokens > 0);
    ref_be.Unload();

    // Multi-slot backend with continuous batching + suffix speculation enabled so
    // single-request -> concurrent batching -> long trailing request transitions are exercised.
    LlamaCppBackend be;
    LoadOptions opts = MakeLoadOptions(ModelPath(), kCtx, kBatch);
    opts.n_slots = kSlots;
    opts.kv_unified = true;
    opts.prefix_cache = true;
    opts.spec_type = "suffix";
    opts.spec_draft_max = 8;
    REQUIRE(be.Load(opts).ok());
    CHECK(be.SlotCount() == kSlots);

    // Subcase A: Simultaneous sibling workers launched together.
    {
        std::mutex start_mu;
        std::condition_variable start_cv;
        int ready = 0;
        bool go = false;
        std::vector<RunOutput> outputs(3);
        std::vector<std::thread> workers;
        ScopedThreadJoiner joiner{workers};
        workers.reserve(3);

        for (std::size_t i = 0; i < 3u; ++i) {
            workers.emplace_back([&, i] {
                {
                    std::unique_lock<std::mutex> lock(start_mu);
                    ++ready;
                    start_cv.notify_all();
                    start_cv.wait(lock, [&] { return go; });
                }
                GenerateRequest req;
                req.prompt = prompts[i];
                req.max_new_tokens = kPeerTokens;
                req.session_key = "sim-worker-" + std::to_string(i);
                outputs[i] = RunRequest(be, req);
            });
        }
        {
            std::unique_lock<std::mutex> lock(start_mu);
            start_cv.wait(lock, [&] { return ready == 3; });
            go = true;
        }
        start_cv.notify_all();
        for (std::thread& t : workers) t.join();
        workers.clear();

        for (std::size_t i = 0; i < 3u; ++i) {
            CAPTURE(i);
            REQUIRE(outputs[i].result.status.ok());
            CHECK(static_cast<int>(outputs[i].tokens.size()) == outputs[i].result.generated_tokens);
            CHECK(outputs[i].result.prompt_tokens + outputs[i].result.cached_tokens ==
                  static_cast<int>(prompts[i].size()));
            CHECK(outputs[i].tokens == ref_peer[i].tokens);
        }
    }

    // Subcase B: Staggered arrivals (worker B and C arrive after worker A has emitted tokens).
    {
        std::mutex stag_mu;
        std::condition_variable stag_cv;
        bool a_started = false;
        bool a_done = false;
        RunOutput out_a;
        RunOutput out_b;
        RunOutput out_c;
        std::chrono::steady_clock::time_point a_finish_tp{};
        std::chrono::steady_clock::time_point b_submit_tp{};

        std::vector<std::thread> workers;
        ScopedThreadJoiner joiner{workers};
        workers.reserve(3);

        workers.emplace_back([&] {
            GenerateRequest req;
            req.prompt = prompts[2];
            req.max_new_tokens = kTrailingTokens;
            req.session_key = "stag-a";
            out_a.result = be.Generate(req, [&](const TokenEvent& ev) {
                if (ev.token >= 0) out_a.tokens.push_back(ev.token);
                out_a.text += ev.text;
                if (out_a.tokens.size() == 2u) {
                    {
                        std::lock_guard<std::mutex> lock(stag_mu);
                        a_started = true;
                    }
                    stag_cv.notify_all();
                }
                return true;
            });
            a_finish_tp = std::chrono::steady_clock::now();
            {
                std::lock_guard<std::mutex> lock(stag_mu);
                a_done = true;
            }
            stag_cv.notify_all();
        });

        {
            std::unique_lock<std::mutex> lock(stag_mu);
            stag_cv.wait(lock, [&] { return a_started || a_done; });
        }
        b_submit_tp = std::chrono::steady_clock::now();

        workers.emplace_back([&] {
            GenerateRequest req;
            req.prompt = prompts[0];
            req.max_new_tokens = kPeerTokens;
            req.session_key = "stag-b";
            out_b = RunRequest(be, req);
        });
        workers.emplace_back([&] {
            GenerateRequest req;
            req.prompt = prompts[1];
            req.max_new_tokens = kPeerTokens;
            req.session_key = "stag-c";
            out_c = RunRequest(be, req);
        });

        for (std::thread& t : workers) t.join();
        workers.clear();

        REQUIRE(out_a.result.status.ok());
        REQUIRE(out_b.result.status.ok());
        REQUIRE(out_c.result.status.ok());
        CHECK(out_a.tokens == ref_trailing.tokens);
        CHECK(out_b.tokens == ref_peer[0].tokens);
        CHECK(out_c.tokens == ref_peer[1].tokens);
        if (ref_trailing.tokens.size() >= 8u) {
            CHECK(b_submit_tp <= a_finish_tp);
        }
    }

    // Subcase C: Long trailing request continuing after short sibling peers finish.
    {
        std::mutex trail_mu;
        std::condition_variable trail_cv;
        int ready = 0;
        bool go = false;
        RunOutput peer0;
        RunOutput peer1;
        RunOutput trailing;
        std::chrono::steady_clock::time_point peer0_done{};
        std::chrono::steady_clock::time_point peer1_done{};
        std::chrono::steady_clock::time_point trailing_done{};

        std::vector<std::thread> workers;
        ScopedThreadJoiner joiner{workers};
        workers.reserve(3);

        auto wait_gate = [&] {
            std::unique_lock<std::mutex> lock(trail_mu);
            ++ready;
            trail_cv.notify_all();
            trail_cv.wait(lock, [&] { return go; });
        };

        workers.emplace_back([&] {
            wait_gate();
            GenerateRequest req;
            req.prompt = prompts[0];
            req.max_new_tokens = kShortTokens;
            req.session_key = "trail-peer-0";
            peer0 = RunRequest(be, req);
            peer0_done = std::chrono::steady_clock::now();
        });
        workers.emplace_back([&] {
            wait_gate();
            GenerateRequest req;
            req.prompt = prompts[1];
            req.max_new_tokens = kShortTokens;
            req.session_key = "trail-peer-1";
            peer1 = RunRequest(be, req);
            peer1_done = std::chrono::steady_clock::now();
        });
        workers.emplace_back([&] {
            wait_gate();
            GenerateRequest req;
            req.prompt = prompts[2];
            req.max_new_tokens = kTrailingTokens;
            req.session_key = "trail-long";
            trailing = RunRequest(be, req);
            trailing_done = std::chrono::steady_clock::now();
        });

        {
            std::unique_lock<std::mutex> lock(trail_mu);
            trail_cv.wait(lock, [&] { return ready == 3; });
            go = true;
        }
        trail_cv.notify_all();
        for (std::thread& t : workers) t.join();
        workers.clear();

        REQUIRE(peer0.result.status.ok());
        REQUIRE(peer1.result.status.ok());
        REQUIRE(trailing.result.status.ok());
        CHECK(peer0.result.generated_tokens <= kShortTokens);
        CHECK(peer1.result.generated_tokens <= kShortTokens);
        CHECK(trailing.tokens == ref_trailing.tokens);
        const auto last_peer_done = std::max(peer0_done, peer1_done);
        if (ref_trailing.tokens.size() >= static_cast<std::size_t>(kShortTokens + 8)) {
            CHECK(last_peer_done <= trailing_done);
        }
        const double solo_tail_ms =
            std::chrono::duration<double, std::milli>(trailing_done - last_peer_done).count();
        std::printf("[llamacpp_integration] multi-slot trailing: peer0=%d tok, peer1=%d tok, trailing=%d tok, "
                    "solo_tail=%.2f ms\n",
                    peer0.result.generated_tokens, peer1.result.generated_tokens, trailing.result.generated_tokens,
                    solo_tail_ms);
    }

    // Subcase D: Cancelling one concurrent slot does not corrupt its active sibling.
    if (ref_peer[0].tokens.size() >= 4u) {
        std::mutex cancel_mu;
        std::condition_variable cancel_cv;
        int ready = 0;
        bool go = false;
        RunOutput cancelled_out;
        RunOutput sibling_out;
        std::vector<std::thread> workers;
        ScopedThreadJoiner joiner{workers};
        workers.reserve(2);

        auto wait_gate = [&] {
            std::unique_lock<std::mutex> lock(cancel_mu);
            ++ready;
            cancel_cv.notify_all();
            cancel_cv.wait(lock, [&] { return go; });
        };

        workers.emplace_back([&] {
            wait_gate();
            GenerateRequest req;
            req.prompt = prompts[0];
            req.max_new_tokens = kPeerTokens;
            req.session_key = "cancel-slot-0";
            cancelled_out = RunRequest(be, req, {}, /*stop_after=*/2);
        });
        workers.emplace_back([&] {
            wait_gate();
            GenerateRequest req;
            req.prompt = prompts[1];
            req.max_new_tokens = kPeerTokens;
            req.session_key = "cancel-slot-1";
            sibling_out = RunRequest(be, req);
        });

        {
            std::unique_lock<std::mutex> lock(cancel_mu);
            cancel_cv.wait(lock, [&] { return ready == 2; });
            go = true;
        }
        cancel_cv.notify_all();
        for (std::thread& t : workers) t.join();
        workers.clear();

        CHECK(cancelled_out.result.status.ok());
        CHECK(cancelled_out.result.stop_reason == StopReason::kCancelled);
        CHECK(cancelled_out.result.generated_tokens <= 2);
        REQUIRE(sibling_out.result.status.ok());
        CHECK(sibling_out.tokens == ref_peer[1].tokens);
    }

    be.Unload();
}

TEST_CASE("llamacpp integration: L2 host-RAM state cache snapshot, restore, and divergence invalidation") {
    constexpr std::uint32_t kCtx = 1024;
    constexpr std::uint32_t kBatch = 128;
    constexpr std::size_t kMinL2Tokens = 256;
    constexpr std::size_t kMinL2Gain = 128;
    constexpr int kGenTokens = 16;

    // 1. Build deterministic prompts (>= 300 tokens for prefix_a so it crosses
    //    kMinL2SnapshotTokens=256 and kMinL2RestoreGainTokens=128 cleanly).
    LlamaCppBackend ref_be;
    LoadOptions ref_opts = MakeLoadOptions(ModelPath(), kCtx, kBatch);
    ref_opts.n_slots = 1;
    ref_opts.prefix_cache = false;
    ref_opts.cache_ram_mib = 0;
    REQUIRE(ref_be.Load(ref_opts).ok());
    const ModelInfo info = ref_be.GetModelInfo();

    const std::vector<Token> prefix_a = BuildMinLengthPrompt(
        ref_be, "Archive handbook for Cobalt subsystem.\n",
        "module cache ledger verifies prefix ownership, host snapshots, and rollback safety across turns.", 304u);
    REQUIRE(prefix_a.size() >= 304u);

    std::vector<Token> tail_a;
    REQUIRE(ref_be.Tokenize("\nQuestion: summarize the Cobalt subsystem in one sentence.\nAnswer:", false, tail_a)
                .ok());
    REQUIRE(!tail_a.empty());

    std::vector<Token> prompt_a_cont = prefix_a;
    prompt_a_cont.insert(prompt_a_cont.end(), tail_a.begin(), tail_a.end());
    REQUIRE(prompt_a_cont.size() + static_cast<std::size_t>(kGenTokens) < static_cast<std::size_t>(kCtx));

    // Completely unrelated prompt B to evict VRAM slot 0 and force SnapshotSlotToL2.
    const std::vector<Token> prompt_b = BuildMinLengthPrompt(
        ref_be, "Unrelated marine weather bulletin for northern harbor.\n",
        "harbor wind gusts shifting northeast with tide height rising past the pier gauge.", 64u);
    REQUIRE(prompt_b.size() >= 64u);
    CHECK(LongestCommonPrefix(prefix_a, prompt_b) <= 2u);

    // Early-diverged prompt (diverges immediately after BOS).
    const std::vector<Token> prompt_div_early = BuildMinLengthPrompt(
        ref_be, "Different culinary recipe for sourdough bread starter.\n",
        "mix rye flour and warm water in a glass jar and rest overnight.", 64u);
    REQUIRE(prompt_div_early.size() >= 64u);
    CHECK(LongestCommonPrefix(prefix_a, prompt_div_early) <= 2u);

    // Mid-prefix-diverged prompt: shares first 180 tokens of prefix_a, then diverges sharply.
    constexpr std::size_t kSharedSplit = 180u;
    REQUIRE(prefix_a.size() > kSharedSplit + 64u);
    std::vector<Token> mid_div_tail;
    REQUIRE(ref_be.Tokenize(
                      "\nDIVERGED BRANCH: ignore earlier Cobalt notes and describe an alpine glacier expedition "
                      "with compass bearings, ice axes, base camp rations, and radio schedules.\nAnswer:",
                      false, mid_div_tail)
                .ok());
    REQUIRE(!mid_div_tail.empty());
    std::vector<Token> prompt_div_mid(prefix_a.begin(), prefix_a.begin() + static_cast<std::ptrdiff_t>(kSharedSplit));
    prompt_div_mid.insert(prompt_div_mid.end(), mid_div_tail.begin(), mid_div_tail.end());
    const std::size_t actual_mid_lcp = LongestCommonPrefix(prefix_a, prompt_div_mid);
    REQUIRE(actual_mid_lcp >= kSharedSplit);
    REQUIRE(actual_mid_lcp < prefix_a.size());

    // Cold no-cache greedy reference continuations.
    const RunOutput ref_a_cont = Run(ref_be, prompt_a_cont, kGenTokens);
    REQUIRE(ref_a_cont.result.status.ok());
    REQUIRE(ref_a_cont.result.generated_tokens > 0);

    const RunOutput ref_div_early = Run(ref_be, prompt_div_early, kGenTokens);
    REQUIRE(ref_div_early.result.status.ok());
    REQUIRE(ref_div_early.result.generated_tokens > 0);

    const RunOutput ref_div_mid = Run(ref_be, prompt_div_mid, kGenTokens);
    REQUIRE(ref_div_mid.result.status.ok());
    REQUIRE(ref_div_mid.result.generated_tokens > 0);
    ref_be.Unload();

    // 2. Load L2-enabled single-slot backend (n_slots=1 forces VRAM eviction on slot switch).
    LlamaCppBackend l2_be;
    LoadOptions l2_opts = MakeLoadOptions(ModelPath(), kCtx, kBatch);
    l2_opts.n_slots = 1;
    l2_opts.prefix_cache = true;
    l2_opts.cache_ram_mib = 512;
    REQUIRE(l2_be.Load(l2_opts).ok());

    // Step 1: Cold prefill of prefix_a (max_new_tokens=0 leaves exact prefix_a state in slot 0).
    GenerateRequest cold_req;
    cold_req.prompt = prefix_a;
    cold_req.max_new_tokens = 0;
    cold_req.session_key = "l2-session-a";
    const RunOutput cold_a = RunRequest(l2_be, cold_req);
    REQUIRE(cold_a.result.status.ok());
    CHECK(cold_a.result.cached_tokens == 0);
    CHECK(cold_a.result.prompt_tokens == static_cast<int>(prefix_a.size()));

    // Step 2: Evict VRAM slot 0 with unrelated prompt_b -> triggers SnapshotSlotToL2 for prefix_a.
    GenerateRequest evict_req;
    evict_req.prompt = prompt_b;
    evict_req.max_new_tokens = 4;
    evict_req.session_key = "l2-session-b";
    const RunOutput evict_b = RunRequest(l2_be, evict_req);
    REQUIRE(evict_b.result.status.ok());
    CHECK(evict_b.result.cached_tokens <= 2);

    // Step 3: Return with prompt_a_cont -> must restore prefix_a from L2 host RAM and match exact continuation.
    GenerateRequest restore_req;
    restore_req.prompt = prompt_a_cont;
    restore_req.max_new_tokens = kGenTokens;
    restore_req.session_key = "l2-session-a";
    const RunOutput restored_a = RunRequest(l2_be, restore_req);
    REQUIRE(restored_a.result.status.ok());
    CHECK(restored_a.result.cached_tokens >= static_cast<int>(kMinL2Tokens));
    CHECK(restored_a.result.prompt_tokens + restored_a.result.cached_tokens ==
          static_cast<int>(prompt_a_cont.size()));
    CHECK(restored_a.result.prompt_tokens <= static_cast<int>(tail_a.size()) + 1);
    CHECK(restored_a.tokens == ref_a_cont.tokens);
    std::printf("[llamacpp_integration] L2 restore: cached=%d evaluated=%d/%zu generated=%d\n",
                restored_a.result.cached_tokens, restored_a.result.prompt_tokens, prompt_a_cont.size(),
                restored_a.result.generated_tokens);

    // Step 4: Early divergence invalidation -> must not reuse cached tokens and must match cold reference.
    GenerateRequest early_req;
    early_req.prompt = prompt_div_early;
    early_req.max_new_tokens = kGenTokens;
    early_req.session_key = "l2-session-early";
    const RunOutput div_early = RunRequest(l2_be, early_req);
    REQUIRE(div_early.result.status.ok());
    CHECK(div_early.result.cached_tokens <= 2);
    CHECK(div_early.result.prompt_tokens + div_early.result.cached_tokens ==
          static_cast<int>(prompt_div_early.size()));
    CHECK(div_early.tokens == ref_div_early.tokens);

    // Step 5: Mid-prefix divergence after VRAM eviction -> never reuse past divergence point,
    // and continuation must match cold baseline identically.
    const RunOutput evict_b2 = RunRequest(l2_be, evict_req);
    REQUIRE(evict_b2.result.status.ok());

    GenerateRequest mid_req;
    mid_req.prompt = prompt_div_mid;
    mid_req.max_new_tokens = kGenTokens;
    mid_req.session_key = "l2-session-mid";
    const RunOutput div_mid = RunRequest(l2_be, mid_req);
    REQUIRE(div_mid.result.status.ok());
    CHECK(div_mid.result.cached_tokens <= static_cast<int>(actual_mid_lcp));
    CHECK(div_mid.result.prompt_tokens + div_mid.result.cached_tokens == static_cast<int>(prompt_div_mid.size()));
    if (info.architecture == si::ModelArchitecture::attention_only && actual_mid_lcp >= kMinL2Gain + 2u) {
        CHECK(div_mid.result.cached_tokens >= static_cast<int>(kMinL2Gain));
    }
    CHECK(div_mid.tokens == ref_div_mid.tokens);
    std::printf("[llamacpp_integration] L2 mid-div: shared=%zu cached=%d evaluated=%d/%zu\n", actual_mid_lcp,
                div_mid.result.cached_tokens, div_mid.result.prompt_tokens, prompt_div_mid.size());

    l2_be.Unload();
}

int main(int argc, char** argv) {
    const char* path = ModelPath();
    if (path == nullptr || path[0] == '\0') {
        std::printf("[llamacpp_integration] SKIP: SONDER_TEST_GGUF not set\n");
        return 77;  // CTest SKIP_RETURN_CODE
    }
    doctest::Context context;
    context.applyCommandLine(argc, argv);
    return context.run();
}
