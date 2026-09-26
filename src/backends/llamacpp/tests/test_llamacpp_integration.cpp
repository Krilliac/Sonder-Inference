// Real-model integration test. Runs only when SONDER_TEST_GGUF names a GGUF
// file; otherwise exits with the CTest skip code (77) so CTest reports it as
// skipped rather than passed. Never commit weights.
#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
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

RunOutput Run(LlamaCppBackend& be, const std::vector<Token>& prompt, int max_new,
              const CancelPredicate& cancel = {}, int stop_after = -1) {
    RunOutput out;
    GenerateRequest req;
    req.prompt = prompt;
    req.max_new_tokens = max_new;
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

const char* ModelPath() { return std::getenv("SONDER_TEST_GGUF"); }

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
