// Core-interface adapter tests (no model weights needed).
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common/diag_log.hpp"
#include "llamacpp_internal.hpp"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"

using namespace sonder::inference;
using llamacpp_detail::ApplyNoThinkScaffolding;
using llamacpp_detail::StopSequenceFilter;
using llamacpp_detail::ThinkingSplitResult;
using llamacpp_detail::ThinkingStreamSplitter;
using llamacpp_detail::ToLlamaSampling;

TEST_CASE("llamacpp adapter: stop filter passes text through without stops") {
    StopSequenceFilter f({});
    CHECK(f.Push("hello ") == "hello ");
    CHECK(f.Push("world") == "world");
    CHECK(f.Flush().empty());
    CHECK_FALSE(f.matched());
}

TEST_CASE("llamacpp adapter: stop filter holds partial matches and cuts on a hit") {
    StopSequenceFilter f({"\nUser:", "###"});
    std::string out;
    out += f.Push("Answer is 4.");
    out += f.Push("\nUs");  // could be the start of "\nUser:"
    CHECK(out == "Answer is 4.");
    out += f.Push("er: next");
    CHECK(f.matched());
    CHECK(out == "Answer is 4.");
    CHECK(f.Push("more").empty());
    CHECK(f.Flush().empty());
}

TEST_CASE("llamacpp adapter: stop filter releases held text that diverges") {
    StopSequenceFilter f({"###"});
    std::string out = f.Push("a#");
    CHECK(out == "a");
    out += f.Push("#b");
    CHECK(out == "a##b");
    CHECK_FALSE(f.matched());
    out += f.Push("##");
    CHECK(out == "a##b");
    CHECK(f.Flush() == "##");
}

TEST_CASE("llamacpp adapter: sampling config mapping") {
    SamplingConfig greedy = SamplingConfig::greedy(32, 42);
    auto p = ToLlamaSampling(greedy);
    CHECK(p.temperature == 0.0f);
    CHECK(p.top_k == 0);
    CHECK(p.top_p == 1.0f);
    CHECK(p.repeat_penalty == 1.0f);
    CHECK(p.seed == 42u);

    SamplingConfig c;
    c.seed.reset();
    CHECK(ToLlamaSampling(c).seed == 0xFFFFFFFFu);  // llama.cpp random seed
    c.seed = 0xFFFFFFFFull;                           // must not collide with "random"
    CHECK(ToLlamaSampling(c).seed != 0xFFFFFFFFu);
    c.seed = 0x0000000100000002ull;
    CHECK(ToLlamaSampling(c).seed == 3u);
}

TEST_CASE("llamacpp adapter: new sampling fields are mapped and validated") {
    SamplingConfig c;
    auto d = ToLlamaSampling(c);
    CHECK(d.repeat_last_n == 64);
    CHECK(d.presence_penalty == 0.0f);
    CHECK(d.frequency_penalty == 0.0f);
    CHECK(d.typical_p == 1.0f);
    CHECK(d.logit_bias.empty());

    c.repeat_last_n = -1;
    c.presence_penalty = 0.5f;
    c.frequency_penalty = -0.25f;
    c.typical_p = 0.9f;
    c.logit_bias = {{7, 2.5f}, {9, -std::numeric_limits<float>::infinity()}};
    auto p = ToLlamaSampling(c);
    CHECK(p.repeat_last_n == -1);
    CHECK(p.presence_penalty == 0.5f);
    CHECK(p.frequency_penalty == -0.25f);
    CHECK(p.typical_p == doctest::Approx(0.9f));
    REQUIRE(p.logit_bias.size() == 2u);
    CHECK(p.logit_bias[0].first == 7);
    CHECK(p.logit_bias[0].second == 2.5f);
    CHECK(p.logit_bias[1].second == -std::numeric_limits<float>::infinity());
    CHECK(sonder::backends::llamacpp::Validate(p).ok());

    p.repeat_last_n = -2;
    CHECK_FALSE(sonder::backends::llamacpp::Validate(p).ok());
    p.repeat_last_n = 64;
    p.typical_p = 0.0f;
    CHECK_FALSE(sonder::backends::llamacpp::Validate(p).ok());
    p.typical_p = 1.0f;
    p.logit_bias = {{1, std::numeric_limits<float>::quiet_NaN()}};
    CHECK_FALSE(sonder::backends::llamacpp::Validate(p).ok());
}

TEST_CASE("llamacpp adapter: backend identity, capabilities, probe") {
    auto be = make_llamacpp_backend();
    REQUIRE(be != nullptr);
    CHECK(be->name() == kLlamaCppBackendName);
    CHECK(be->description().find("b11195") != std::string::npos);
    const auto caps = be->capabilities();
    CHECK(caps.has(Capability::tokenization));
    CHECK(caps.has(Capability::streaming));
    CHECK(caps.has(Capability::batched_prefill));
    CHECK(caps.has(Capability::prefix_reuse));
    CHECK(caps.has(Capability::speculative_decode));
    CHECK(caps.has(Capability::structured_output));
    CHECK(caps.has(Capability::embeddings));
    CHECK(caps.has(Capability::deterministic));
    CHECK(caps.has(Capability::recurrent_checkpoint));
    const auto cap_names = caps.names();
    CHECK(std::find(cap_names.begin(), cap_names.end(), "recurrent_checkpoint") != cap_names.end());
    CHECK_FALSE(caps.has(Capability::remote_process));
    CHECK_FALSE(caps.has(Capability::kv_export));
    auto probe = be->probe();
    REQUIRE(probe.ok());
    CHECK(probe.value().find("llama.cpp b11195") != std::string::npos);
    CHECK(probe.value().find("cpu") != std::string::npos);
}

TEST_CASE("llamacpp adapter: list_models scans model_dirs for gguf files") {
    LlamaCppBackendOptions opts;
    opts.model_dirs = {SONDER_LLAMACPP_VOCAB_DIR, "does/not/exist"};
    auto be = make_llamacpp_backend(opts);
    auto models = be->list_models();
    REQUIRE(models.ok());
    bool found = false;
    for (const auto& m : models.value()) {
        CHECK(m.format == "gguf");
        CHECK(m.backend == kLlamaCppBackendName);
        found = found || m.name == "ggml-vocab-llama-spm.gguf";
    }
    CHECK(found);
}

TEST_CASE("llamacpp adapter: load_model error mapping") {
    auto be = make_llamacpp_backend();
    ModelLoadOptions o;
    CHECK(be->load_model(o).status().code() == ErrorCode::invalid_argument);
    o.model = "missing-model.gguf";
    CHECK(be->load_model(o).status().code() == ErrorCode::not_found);
    o.model = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    o.device_id = "npu:0";
    CHECK(be->load_model(o).status().code() == ErrorCode::unsupported);
}

TEST_CASE("llamacpp adapter: context options default to today's behaviour and validate") {
    const LlamaCppBackendOptions d;
    CHECK(d.batch_size == 512u);
    CHECK(d.ubatch_size == 0u);
    CHECK(d.kv_cache_type_k == "f16");
    CHECK(d.kv_cache_type_v == "f16");
    CHECK(d.flash_attention == "auto");
    CHECK(validate_llamacpp_options(d).ok());

    LlamaCppBackendOptions o;
    o.kv_cache_type_k = "q8_0";
    o.kv_cache_type_v = "q8_0";
    CHECK(validate_llamacpp_options(o).ok());  // flash attention auto
    o.flash_attention = "on";
    CHECK(validate_llamacpp_options(o).ok());
    o.flash_attention = "off";
    Status s = validate_llamacpp_options(o);
    CHECK(s.code() == ErrorCode::invalid_argument);
    CHECK(s.message().find("flash attention") != std::string::npos);
    o.kv_cache_type_v = "f16";  // quantized K alone is fine without it
    CHECK(validate_llamacpp_options(o).ok());

    o = {};
    o.kv_cache_type_k = "q4_k";
    s = validate_llamacpp_options(o);
    CHECK(s.code() == ErrorCode::invalid_argument);
    CHECK(s.message().find("K cache type 'q4_k'") != std::string::npos);
    o = {};
    o.kv_cache_type_v = "";
    CHECK(validate_llamacpp_options(o).message().find("V cache type") != std::string::npos);
    o = {};
    o.flash_attention = "maybe";
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o = {};
    o.batch_size = 256;
    o.ubatch_size = 512;
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o.ubatch_size = 128;
    CHECK(validate_llamacpp_options(o).ok());
}

TEST_CASE("llamacpp adapter: load_model rejects invalid context options") {
    LlamaCppBackendOptions opts;
    opts.kv_cache_type_v = "q4_0";
    opts.flash_attention = "off";
    auto be = make_llamacpp_backend(opts);
    ModelLoadOptions o;
    o.model = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    const auto r = be->load_model(o);
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    CHECK(r.status().message().find("flash attention") != std::string::npos);
}

TEST_CASE("llamacpp adapter: ThinkingStreamSplitter splits <think> blocks across arbitrary chunk boundaries") {
    // Plain text passes through without buffering artifacts.
    {
        ThinkingStreamSplitter s;
        const ThinkingSplitResult r1 = s.Push("Hello world");
        const ThinkingSplitResult r2 = s.Flush();
        CHECK(r1.text + r2.text == "Hello world");
        CHECK((r1.reasoning + r2.reasoning).empty());
        CHECK_FALSE(s.in_thinking());
    }

    // Tag boundaries split across tiny chunks.
    {
        ThinkingStreamSplitter s;
        std::string text;
        std::string reasoning;
        for (const std::string_view piece : {"<thi", "nk>step 1 ", "and 2</th", "ink>Final ", "answer."}) {
            const ThinkingSplitResult r = s.Push(piece);
            text += r.text;
            reasoning += r.reasoning;
        }
        const ThinkingSplitResult tail = s.Flush();
        text += tail.text;
        reasoning += tail.reasoning;
        CHECK(reasoning == "step 1 and 2");
        CHECK(text == "Final answer.");
        CHECK_FALSE(s.in_thinking());
    }

    // False partial tag prefixes release cleanly without dropping characters.
    {
        ThinkingStreamSplitter s;
        std::string text;
        std::string reasoning;
        for (const std::string_view piece : {"1 < 2 <th", "at <think>a </thi", "x b</think> done"}) {
            const ThinkingSplitResult r = s.Push(piece);
            text += r.text;
            reasoning += r.reasoning;
        }
        const ThinkingSplitResult tail = s.Flush();
        text += tail.text;
        reasoning += tail.reasoning;
        CHECK(text == "1 < 2 <that  done");
        CHECK(reasoning == "a </thix b");
    }

    // Unclosed <think> flushes remaining text to reasoning.
    {
        ThinkingStreamSplitter s;
        const ThinkingSplitResult r1 = s.Push("<think>still thinking</thi");
        CHECK(s.in_thinking());
        const ThinkingSplitResult r2 = s.Flush();
        CHECK(r1.reasoning + r2.reasoning == "still thinking</thi");
        CHECK((r1.text + r2.text).empty());
    }
}

TEST_CASE("llamacpp adapter: ApplyNoThinkScaffolding injects empty think block only on ChatML assistant cue") {
    // Appends when thinking is disabled on ChatML/Qwen3 assistant cue.
    std::vector<ChatPromptSegment> segs{
        {"<|im_start|>user\n", true},
        {"Hi", false},
        {"<|im_end|>\n<|im_start|>assistant\n", true},
    };
    CHECK(ApplyNoThinkScaffolding(segs, /*enable_thinking=*/false));
    REQUIRE(segs.size() == 4u);
    CHECK(segs.back().parse_special);
    CHECK(segs.back().text == "<think>\n\n</think>\n\n");

    // Idempotent: does not append again once <think> is already present.
    CHECK_FALSE(ApplyNoThinkScaffolding(segs, /*enable_thinking=*/false));

    // Does not append when thinking is enabled.
    std::vector<ChatPromptSegment> segs_on{
        {"<|im_start|>assistant\n", true},
    };
    CHECK_FALSE(ApplyNoThinkScaffolding(segs_on, /*enable_thinking=*/true));

    // Does not append on non-ChatML templates or untrusted content segments.
    std::vector<ChatPromptSegment> llama3{
        {"<|start_header_id|>assistant<|end_header_id|>\n\n", true},
    };
    CHECK_FALSE(ApplyNoThinkScaffolding(llama3, /*enable_thinking=*/false));

    std::vector<ChatPromptSegment> untrusted{
        {"<|im_start|>assistant\n", false},
    };
    CHECK_FALSE(ApplyNoThinkScaffolding(untrusted, /*enable_thinking=*/false));
}

TEST_CASE("llamacpp adapter: new tuning, slot, and speculation options validate") {
    LlamaCppBackendOptions o;
    CHECK(o.use_mmap);
    CHECK(o.kv_unified);
    CHECK(o.prefix_cache);
    CHECK(o.slots == 1u);
    CHECK(o.spec_type == "none");
    CHECK(o.spec_draft_max == 0);
    CHECK(validate_llamacpp_options(o).ok());

    o.use_mmap = false;
    o.kv_unified = false;
    o.prefix_cache = true;
    o.slots = 4;
    o.spec_type = "auto";
    o.spec_draft_max = 8;
    CHECK(validate_llamacpp_options(o).ok());

    o.slots = 0;
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o.slots = 17;
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o.slots = 2;
    o.spec_type = "invalid-spec";
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o.spec_type = "draft-mtp";
    o.spec_draft_max = -1;
    CHECK(validate_llamacpp_options(o).code() == ErrorCode::invalid_argument);
    o.spec_draft_max = 3;
    CHECK(validate_llamacpp_options(o).ok());
}

TEST_CASE("llamacpp adapter: ResolveGpuDevicePin routes gpu:N to the matching ggml GPU") {
    namespace lc = sonder::backends::llamacpp;
    const std::vector<lc::DeviceInfo> devs{
        {"CPU", "Ryzen", lc::DeviceKind::kCpu, 0, 0},
        {"Vulkan0", "NVIDIA RTX 5070 Ti", lc::DeviceKind::kGpu, 16000, 15000},
        {"Vulkan1", "AMD Radeon Graphics", lc::DeviceKind::kIntegratedGpu, 512, 512},
    };
    std::vector<std::string> pinned;
    std::int32_t main_gpu = 1;
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:1", devs, pinned, main_gpu).ok());
    REQUIRE(pinned.size() == 1u);
    CHECK(pinned[0] == "Vulkan1");
    CHECK(main_gpu == 0);

    pinned.clear();
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:0", devs, pinned, main_gpu).ok());
    REQUIRE(pinned.size() == 1u);
    CHECK(pinned[0] == "Vulkan0");

    // Out-of-range GPU index fails explicitly instead of silently falling back to GPU 0.
    pinned.clear();
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:2", devs, pinned, main_gpu).code() == ErrorCode::not_found);
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:abc", devs, pinned, main_gpu).code() ==
          ErrorCode::invalid_argument);

    // Explicit caller devices list is preserved untouched.
    pinned = {"CUDA0", "CUDA1"};
    main_gpu = 1;
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:0", devs, pinned, main_gpu).ok());
    CHECK(pinned.size() == 2u);
    CHECK(main_gpu == 1);

    // Explicit multi-GPU tensor_split is preserved untouched (never overwritten by single-GPU pin).
    pinned.clear();
    main_gpu = 1;
    const std::vector<float> split{1.0f, 1.0f};
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:0", devs, pinned, main_gpu, split).ok());
    CHECK(pinned.empty());
    CHECK(main_gpu == 1);

    // Cross-vendor OS vs CUDA-only GGML inventory: OS gpu:0 = AMD iGPU, OS gpu:1 = NVIDIA RTX 5070 Ti.
    // CUDA-only llama.cpp exposes only CUDA0 -> OS gpu:1 resolves to CUDA0, while OS gpu:0 fails closed!
    const std::vector<lc::DeviceInfo> cuda_only{
        {"CPU", "Ryzen", lc::DeviceKind::kCpu, 0, 0},
        {"CUDA0", "NVIDIA GeForce RTX 5070 Ti", lc::DeviceKind::kGpu, 16000, 15000},
    };
    GpuDeviceInfo os_amd;
    os_amd.id = "gpu:0";
    os_amd.name = "AMD Radeon(TM) Graphics";
    os_amd.vendor = GpuVendor::amd;
    GpuDeviceInfo os_nv;
    os_nv.id = "gpu:1";
    os_nv.name = "NVIDIA GeForce RTX 5070 Ti";
    os_nv.vendor = GpuVendor::nvidia;
    const std::vector<GpuDeviceInfo> os_gpus{os_amd, os_nv};

    pinned.clear();
    main_gpu = 0;
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:1", cuda_only, pinned, main_gpu, {}, os_gpus).ok());
    REQUIRE(pinned.size() == 1u);
    CHECK(pinned[0] == "CUDA0");

    pinned.clear();
    CHECK(llamacpp_detail::ResolveGpuDevicePin("gpu:0", cuda_only, pinned, main_gpu, {}, os_gpus).code() ==
          ErrorCode::not_found);
}

TEST_CASE("llamacpp adapter: PopulateGenerateStats separates total, cached, and uncached prompt tokens") {
    namespace lc = sonder::backends::llamacpp;

    // 1. Cold prompt (0 cached tokens).
    {
        lc::GenerateResult r;
        r.prompt_tokens = 128;
        r.cached_tokens = 0;
        r.generated_tokens = 32;
        r.prefill_ms = 20.0;
        r.decode_ms = 400.0;
        GenerateStats stats;
        stats.chunks = 32;
        llamacpp_detail::PopulateGenerateStats(r, /*restore_tokens=*/0, /*load_ns=*/12345u, stats);
        CHECK(stats.token_counts_from_backend);
        CHECK(stats.load_ns == 12345u);
        CHECK(stats.chunks == 32u);
        CHECK(stats.prompt_tokens == 128u);
        REQUIRE(stats.cached_tokens.has_value());
        CHECK(*stats.cached_tokens == 0u);
        CHECK(stats.completion_tokens == 32u);
        CHECK(stats.prompt_eval_ns == 20000000u);
        CHECK(stats.eval_ns == 400000000u);
        REQUIRE(stats.predicted_tokens_per_second.has_value());
        CHECK(*stats.predicted_tokens_per_second == doctest::Approx(80.0));
        CHECK_FALSE(stats.draft_tokens.has_value());
        CHECK_FALSE(stats.draft_accepted_tokens.has_value());
        REQUIRE(stats.backend_timings.has_value());
        CHECK(stats.backend_timings->prompt_n == std::optional<std::uint64_t>(128u));
        CHECK(stats.backend_timings->cache_n == std::optional<std::uint64_t>(0u));
        CHECK(stats.backend_timings->prompt_ms == std::optional<double>(20.0));
        CHECK(stats.backend_timings->predicted_n == std::optional<std::uint64_t>(32u));
        CHECK(stats.backend_timings->predicted_ms == std::optional<double>(400.0));
        CHECK_FALSE(stats.backend_timings->draft_n.has_value());
        CHECK_FALSE(stats.backend_timings->draft_n_accepted.has_value());
    }

    // 2. 99% KV prefix cache hit: prompt_tokens is total (1000), cached_tokens is 990,
    //    and backend_timings.prompt_n is strictly uncached evaluated tokens (10).
    {
        lc::GenerateResult r;
        r.prompt_tokens = 10;
        r.cached_tokens = 990;
        r.generated_tokens = 50;
        r.prefill_ms = 5.0;
        r.decode_ms = 500.0;
        GenerateStats stats;
        llamacpp_detail::PopulateGenerateStats(r, 0, 0, stats);
        CHECK(stats.prompt_tokens == 1000u);
        REQUIRE(stats.cached_tokens.has_value());
        CHECK(*stats.cached_tokens == 990u);
        REQUIRE(stats.backend_timings.has_value());
        CHECK(stats.backend_timings->prompt_n == std::optional<std::uint64_t>(10u));
        CHECK(stats.backend_timings->cache_n == std::optional<std::uint64_t>(990u));
        CHECK(stats.prompt_tokens == *stats.backend_timings->prompt_n + *stats.backend_timings->cache_n);
    }

    // 3. 100% cache hit (0 newly evaluated tokens).
    {
        lc::GenerateResult r;
        r.prompt_tokens = 0;
        r.cached_tokens = 512;
        r.generated_tokens = 8;
        r.prefill_ms = 0.25;
        r.decode_ms = 80.0;
        GenerateStats stats;
        llamacpp_detail::PopulateGenerateStats(r, 0, 0, stats);
        CHECK(stats.prompt_tokens == 512u);
        CHECK(stats.cached_tokens == std::optional<std::uint64_t>(512u));
        REQUIRE(stats.backend_timings.has_value());
        CHECK(stats.backend_timings->prompt_n == std::optional<std::uint64_t>(0u));
        CHECK(stats.backend_timings->cache_n == std::optional<std::uint64_t>(512u));
    }

    // 4. Recurrent checkpoint restore attributes restore_tokens to cache_n / cached_tokens.
    {
        lc::GenerateResult r;
        r.state_restored = true;
        r.prompt_tokens = 44;
        r.cached_tokens = 0;
        r.generated_tokens = 16;
        r.prefill_ms = 10.0;
        r.decode_ms = 160.0;
        GenerateStats stats;
        llamacpp_detail::PopulateGenerateStats(r, /*restore_tokens=*/256, 0, stats);
        CHECK(stats.prompt_tokens == 300u);
        CHECK(stats.cached_tokens == std::optional<std::uint64_t>(256u));
        REQUIRE(stats.backend_timings.has_value());
        CHECK(stats.backend_timings->prompt_n == std::optional<std::uint64_t>(44u));
        CHECK(stats.backend_timings->cache_n == std::optional<std::uint64_t>(256u));
    }

    // 5. Speculative decoding active populates draft counts and clamps accepted <= drafted.
    {
        lc::GenerateResult r;
        r.prompt_tokens = 20;
        r.generated_tokens = 30;
        r.draft_tokens = 24;
        r.draft_accepted_tokens = 18;
        GenerateStats stats;
        llamacpp_detail::PopulateGenerateStats(r, 0, 0, stats);
        CHECK(stats.draft_tokens == std::optional<std::uint64_t>(24u));
        CHECK(stats.draft_accepted_tokens == std::optional<std::uint64_t>(18u));
        REQUIRE(stats.backend_timings.has_value());
        CHECK(stats.backend_timings->draft_n == std::optional<std::uint64_t>(24u));
        CHECK(stats.backend_timings->draft_n_accepted == std::optional<std::uint64_t>(18u));
    }

    // 6. Negative counts, accepted > drafted, and NaN/Inf durations are clamped safely.
    {
        lc::GenerateResult r;
        r.prompt_tokens = -7;
        r.cached_tokens = -3;
        r.generated_tokens = -1;
        r.draft_tokens = 5;
        r.draft_accepted_tokens = 99;
        r.prefill_ms = std::numeric_limits<double>::quiet_NaN();
        r.decode_ms = std::numeric_limits<double>::infinity();
        GenerateStats stats;
        llamacpp_detail::PopulateGenerateStats(r, 0, 0, stats);
        CHECK(stats.prompt_tokens == 0u);
        CHECK(stats.cached_tokens == std::optional<std::uint64_t>(0u));
        CHECK(stats.completion_tokens == 0u);
        CHECK(stats.prompt_eval_ns == 0u);
        CHECK(stats.eval_ns == 0u);
        CHECK_FALSE(stats.predicted_tokens_per_second.has_value());
        CHECK(stats.draft_tokens == std::optional<std::uint64_t>(5u));
        CHECK(stats.draft_accepted_tokens == std::optional<std::uint64_t>(5u));
        REQUIRE(stats.backend_timings.has_value());
        CHECK(stats.backend_timings->prompt_n == std::optional<std::uint64_t>(0u));
        CHECK(stats.backend_timings->cache_n == std::optional<std::uint64_t>(0u));
        CHECK(stats.backend_timings->draft_n == std::optional<std::uint64_t>(5u));
        CHECK(stats.backend_timings->draft_n_accepted == std::optional<std::uint64_t>(5u));
    }
}

TEST_CASE("llamacpp adapter: SummarizeGgmlDeviceMemory and HasIntegratedUmaTarget classify discrete and UMA GPUs") {
    namespace lc = sonder::backends::llamacpp;
    constexpr std::uint64_t kGiB = 1024ull * 1024ull * 1024ull;

    // CPU-only devices report unsupported rather than fabricating zeroes.
    const std::vector<lc::DeviceInfo> cpu_only{
        {"CPU", "Ryzen 9", lc::DeviceKind::kCpu, 0, 0},
    };
    const auto cpu_reading = llamacpp_detail::SummarizeGgmlDeviceMemory(cpu_only, {});
    CHECK(cpu_reading.probe == "none");
    CHECK(cpu_reading.status == "unsupported");
    CHECK_FALSE(llamacpp_detail::HasIntegratedUmaTarget(cpu_only, {}));

    // Hybrid Node1 layout: discrete RX 9060 XT (Vulkan0) + integrated Radeon 780M (Vulkan1).
    const std::vector<lc::DeviceInfo> hybrid{
        {"CPU", "Ryzen 7", lc::DeviceKind::kCpu, 0, 0},
        {"Vulkan0", "AMD Radeon RX 9060 XT", lc::DeviceKind::kGpu, 4 * kGiB, 16 * kGiB},
        {"Vulkan1", "AMD Radeon 780M Graphics", lc::DeviceKind::kIntegratedGpu, 22 * kGiB, 32 * kGiB},
    };
    const auto all_reading = llamacpp_detail::SummarizeGgmlDeviceMemory(hybrid, {});
    CHECK(all_reading.probe == "ggml");
    CHECK(all_reading.status == "ok");
    CHECK(all_reading.dedicated_bytes == 12 * kGiB);
    CHECK(all_reading.shared_bytes == 10 * kGiB);
    CHECK(llamacpp_detail::HasIntegratedUmaTarget(hybrid, {}));
    CHECK(llamacpp_detail::HasIntegratedUmaTarget(hybrid, {"Vulkan0", "Vulkan1"}));

    // Discrete-only configured device excludes integrated GPU.
    const auto discrete_only = llamacpp_detail::SummarizeGgmlDeviceMemory(hybrid, {"vulkan0"});
    CHECK(discrete_only.probe == "ggml");
    CHECK(discrete_only.status == "ok");
    CHECK(discrete_only.dedicated_bytes == 12 * kGiB);
    CHECK(discrete_only.shared_bytes == 0u);
    CHECK_FALSE(llamacpp_detail::HasIntegratedUmaTarget(hybrid, {"vulkan0"}));

    // Anomalous free > total device is skipped cleanly without underflow.
    const std::vector<lc::DeviceInfo> anomalous{
        {"Vulkan0", "Anomalous GPU", lc::DeviceKind::kGpu, 20 * kGiB, 16 * kGiB},
    };
    const auto anom_reading = llamacpp_detail::SummarizeGgmlDeviceMemory(anomalous, {});
    CHECK(anom_reading.status == "unsupported");
}

TEST_CASE("llamacpp adapter: runtime_status populates gpu_memory, spill guard, UMA suppression, and engine telemetry") {
    constexpr std::uint64_t kMiB = 1024ull * 1024ull;
    constexpr std::uint64_t kGiB = 1024ull * kMiB;

    // 1. Default production backend returns non-nullopt runtime_status safely across threads.
    {
        LlamaCppBackendOptions opts;
        opts.context_length = 8192;
        auto be = make_llamacpp_backend(opts);
        const auto st = be->runtime_status();
        REQUIRE(st.has_value());
        CHECK((st->gpu_memory.status == "ok" || st->gpu_memory.status == "unsupported" ||
               st->gpu_memory.status == "error"));
        CHECK(st->context.policy == "warn");
        CHECK(st->context.configured_ctx == std::optional<std::uint64_t>(8192u));
        CHECK_FALSE(st->context.fitted_ctx.has_value());
        CHECK(st->context.outcome == "not_needed");

        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) {
            threads.emplace_back([&be] {
                for (int j = 0; j < 8; ++j) {
                    const auto s = be->runtime_status();
                    CHECK(s.has_value());
                }
            });
        }
        for (auto& t : threads) t.join();
    }

    // 2. Discrete GPU spill detection, peak_shared_bytes high-water mark, error/recovery, and throttle.
    {
        std::atomic<int> call{0};
        LlamaCppBackendOptions opts;
        opts.context_length = 32768;
        auto be = llamacpp_detail::MakeLlamaCppBackendForTest(
            opts,
            [&call]() -> llamacpp_detail::ProcessGpuMemoryReading {
                const int n = call.fetch_add(1);
                if (n == 0) return {"pdh", "ok", "", 12 * kGiB, 128 * kMiB};
                if (n == 1) return {"pdh", "ok", "", 15 * kGiB, 400 * kMiB};
                if (n == 2) return {"pdh", "error", "transient pdh failure", 0, 0};
                return {"pdh", "ok", "", 14 * kGiB, 100 * kMiB};
            },
            std::chrono::milliseconds{0}, /*force_uma_mode=*/false, /*force_gpu_offload=*/true);

        const auto s1 = be->runtime_status();
        REQUIRE(s1.has_value());
        CHECK(s1->gpu_memory.probe == "pdh");
        CHECK(s1->gpu_memory.status == "ok");
        CHECK(s1->gpu_memory.dedicated_bytes == 12 * kGiB);
        CHECK(s1->gpu_memory.shared_bytes == 128 * kMiB);
        CHECK(s1->gpu_memory.peak_shared_bytes == 128 * kMiB);
        CHECK(s1->gpu_memory.samples == 1u);
        CHECK_FALSE(s1->gpu_memory.spilled);
        CHECK(s1->warnings.empty());

        const auto s2 = be->runtime_status();
        REQUIRE(s2.has_value());
        CHECK(s2->gpu_memory.dedicated_bytes == 15 * kGiB);
        CHECK(s2->gpu_memory.shared_bytes == 400 * kMiB);
        CHECK(s2->gpu_memory.peak_shared_bytes == 400 * kMiB);
        CHECK(s2->gpu_memory.samples == 2u);
        CHECK(s2->gpu_memory.spilled);
        REQUIRE(s2->warnings.size() == 1u);
        CHECK(s2->warnings[0].code == "vram_spill");

        // Error reading preserves sample count and peak_shared_bytes without claiming spill.
        const auto s3 = be->runtime_status();
        REQUIRE(s3.has_value());
        CHECK(s3->gpu_memory.status == "error");
        CHECK(s3->gpu_memory.error == "transient pdh failure");
        CHECK(s3->gpu_memory.samples == 2u);
        CHECK(s3->gpu_memory.peak_shared_bytes == 400 * kMiB);
        CHECK_FALSE(s3->gpu_memory.spilled);

        // Recovery increments samples while retaining peak_shared_bytes high-water mark.
        const auto s4 = be->runtime_status();
        REQUIRE(s4.has_value());
        CHECK(s4->gpu_memory.status == "ok");
        CHECK(s4->gpu_memory.error.empty());
        CHECK(s4->gpu_memory.samples == 3u);
        CHECK(s4->gpu_memory.shared_bytes == 100 * kMiB);
        CHECK(s4->gpu_memory.peak_shared_bytes == 400 * kMiB);
        CHECK_FALSE(s4->gpu_memory.spilled);
    }

    // 3. Throttling keeps rapid health calls O(1) cached without re-invoking the probe.
    {
        std::atomic<int> probes{0};
        auto be = llamacpp_detail::MakeLlamaCppBackendForTest(
            {},
            [&probes]() -> llamacpp_detail::ProcessGpuMemoryReading {
                ++probes;
                return {"pdh", "ok", "", 4 * kGiB, 64 * kMiB};
            },
            std::chrono::milliseconds{60000}, false, true);
        const auto a = be->runtime_status();
        const auto b = be->runtime_status();
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        CHECK(probes.load() == 1);
        CHECK(a->gpu_memory.samples == 1u);
        CHECK(b->gpu_memory.samples == 1u);
    }

    // 4. Integrated UMA / hybrid mode suppresses false VRAM spill alerts on high shared usage.
    {
        auto be = llamacpp_detail::MakeLlamaCppBackendForTest(
            {},
            []() -> llamacpp_detail::ProcessGpuMemoryReading {
                return {"pdh", "ok", "", 15 * kGiB, 25 * kGiB};
            },
            std::chrono::milliseconds{0}, /*force_uma_mode=*/true, /*force_gpu_offload=*/true);
        const auto st = be->runtime_status();
        REQUIRE(st.has_value());
        CHECK(st->gpu_memory.status == "ok");
        CHECK(st->gpu_memory.dedicated_bytes == 15 * kGiB);
        CHECK(st->gpu_memory.shared_bytes == 25 * kGiB);
        CHECK_FALSE(st->gpu_memory.spilled);
        CHECK(st->warnings.empty());
    }

    // 5. End-to-end Engine periodic sampler emits backend.gpu_memory.sample for direct llamacpp.
    {
        auto sink = std::make_shared<MemoryTelemetrySink>();
        std::atomic<std::uint64_t> step{0};
        auto be = llamacpp_detail::MakeLlamaCppBackendForTest(
            {},
            [&step]() -> llamacpp_detail::ProcessGpuMemoryReading {
                const std::uint64_t k = step.fetch_add(1);
                return {"pdh", "ok", "", (10 + k) * kGiB, 320 * kMiB};
            },
            std::chrono::milliseconds{0}, /*force_uma_mode=*/false, /*force_gpu_offload=*/true);

        EngineOptions eo;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = false;
        eo.device_sample_interval = std::chrono::milliseconds{15};

        std::vector<json::Value> gpu_samples;
        std::vector<json::Value> warnings;
        {
            Engine engine(eo);
            REQUIRE(engine.register_backend(be).ok());
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (std::chrono::steady_clock::now() < deadline) {
                engine.telemetry().flush();
                gpu_samples.clear();
                warnings.clear();
                for (const auto& line : sink->lines()) {
                    const auto parsed = json::parse(line);
                    if (!parsed.ok()) continue;
                    const auto type = parsed->find("event_type")->as_string();
                    if (type == "backend.gpu_memory.sample") gpu_samples.push_back(parsed.value());
                    if (type == "backend.warning") warnings.push_back(parsed.value());
                }
                if (gpu_samples.size() >= 2 && !warnings.empty()) break;
                std::this_thread::sleep_for(std::chrono::milliseconds{10});
            }
        }
        REQUIRE(gpu_samples.size() >= 2u);
        const auto* first_attrs = gpu_samples[0].find("attributes");
        REQUIRE(first_attrs != nullptr);
        CHECK(first_attrs->find("backend")->as_string() == kLlamaCppBackendName);
        CHECK(first_attrs->find("probe")->as_string() == "pdh");
        CHECK(first_attrs->find("status")->as_string() == "ok");
        CHECK(first_attrs->find("dedicated_bytes")->as_uint() >= 10 * kGiB);
        CHECK(first_attrs->find("shared_bytes")->as_uint() == 320 * kMiB);
        CHECK(first_attrs->find("spilled")->as_bool());
        CHECK(first_attrs->find("samples")->as_uint() == 1u);
        CHECK(gpu_samples[1].find("attributes")->find("samples")->as_uint() >
              first_attrs->find("samples")->as_uint());
        REQUIRE(warnings.size() == 1u);
        CHECK(warnings[0].find("attributes")->find("backend")->as_string() == kLlamaCppBackendName);
        CHECK(warnings[0].find("attributes")->find("code")->as_string() == "vram_spill");
    }
}

namespace {
struct DiagCapture {
    std::mutex mu;
    std::condition_variable cv;
    std::vector<std::string> lines;
};
}  // namespace

TEST_CASE("llamacpp adapter: llama.cpp log lines go through the non-blocking diagnostic queue") {
    auto cap = std::make_shared<DiagCapture>();
    sonder::inference::diag::set_sink([cap](const std::string& line) {
        std::lock_guard<std::mutex> lock(cap->mu);
        cap->lines.push_back(line);
        cap->cv.notify_all();
    });
    auto backend = make_llamacpp_backend();  // installs the diagnostic-queue sink
    REQUIRE(backend);
    sonder::backends::llamacpp::RouteLog(sonder::backends::llamacpp::LogLevel::kError, "ggml exploded\n");
    std::unique_lock<std::mutex> lock(cap->mu);
    const bool got = cap->cv.wait_for(lock, std::chrono::seconds(5), [&cap] { return !cap->lines.empty(); });
    REQUIRE_MESSAGE(got, "the llama.cpp log line never reached the diagnostic queue");
    CHECK(cap->lines.front() == "ggml exploded\n");
    lock.unlock();
    sonder::inference::diag::set_sink(nullptr);
    sonder::backends::llamacpp::SetLogSink(nullptr);
}
