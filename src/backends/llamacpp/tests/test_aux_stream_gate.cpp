// NAT-12 unit, hostile-input, concurrency, and synthetic-GGUF integration tests:
//   - Low-priority vs high-priority CUDA stream resolution and ggml-cuda TLS hints
//   - Primary-context CudaPriorityStream and TensorRT-RTX user_compute_stream options
//   - DecodeAwareAuxGate Deficit Round Robin (DRR) inter-step slicing (zero overlap,
//     >=95% main decode duty preservation, prompt cancellation, domain isolation)
//   - Dynamic <=512-token embedding/reranking sub-slicing and 16 kHz STT slice planning
//   - End-to-end synthetic GGUF co-resident main decode + Embed + ScorePairs + aux STT
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "embed_batch_planner.hpp"
#include "ggml.h"
#include "gguf.h"
#include "sonder/backends/llamacpp/aux_stream_gate.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder_cuda_prio_state.hpp"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

namespace {

class TempGgufDir {
public:
    TempGgufDir() {
        static std::atomic<std::uint64_t> seq{0};
        const auto ticks = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        dir_ = std::filesystem::temp_directory_path() /
               ("sonder-test-nat12-" + std::to_string(ticks) + "-" +
                std::to_string(seq.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(dir_);
    }

    ~TempGgufDir() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    [[nodiscard]] std::string file_utf8(const std::string& name) const {
        const std::u8string u = (dir_ / name).u8string();
        return std::string(u.begin(), u.end());
    }

private:
    std::filesystem::path dir_;
};

// Writes a tiny 1-layer F32 Llama GGUF (~30 KiB) whose weights produce a deterministic
// non-EOS greedy token cycle (1..kVocab-1) and valid dense embeddings.
std::string WriteTinyLlamaGguf(const TempGgufDir& dir, const std::string& filename,
                               std::uint32_t n_ctx = 512u) {
    struct GgufDeleter {
        void operator()(gguf_context* c) const noexcept {
            if (c != nullptr) gguf_free(c);
        }
    };
    struct GgmlDeleter {
        void operator()(ggml_context* c) const noexcept {
            if (c != nullptr) ggml_free(c);
        }
    };

    constexpr std::int64_t kEmbd = 64;
    constexpr std::int64_t kHead = 1;
    constexpr std::int64_t kHeadKv = 1;
    constexpr std::int64_t kFf = 64;
    constexpr std::int64_t kVocab = 8;

    std::unique_ptr<gguf_context, GgufDeleter> ctx(gguf_init_empty());
    REQUIRE(ctx != nullptr);
    gguf_set_val_str(ctx.get(), "general.architecture", "llama");
    gguf_set_val_str(ctx.get(), "general.name", "nat12-tiny-llama");
    gguf_set_val_u32(ctx.get(), "llama.context_length", n_ctx);
    gguf_set_val_u32(ctx.get(), "llama.embedding_length", static_cast<std::uint32_t>(kEmbd));
    gguf_set_val_u32(ctx.get(), "llama.block_count", 1u);
    gguf_set_val_u32(ctx.get(), "llama.feed_forward_length", static_cast<std::uint32_t>(kFf));
    gguf_set_val_u32(ctx.get(), "llama.attention.head_count", static_cast<std::uint32_t>(kHead));
    gguf_set_val_u32(ctx.get(), "llama.attention.head_count_kv", static_cast<std::uint32_t>(kHeadKv));
    gguf_set_val_u32(ctx.get(), "llama.rope.dimension_count", static_cast<std::uint32_t>(kEmbd));
    gguf_set_val_f32(ctx.get(), "llama.attention.layer_norm_rms_epsilon", 1e-5f);

    gguf_set_val_str(ctx.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(ctx.get(), "tokenizer.ggml.pre", "default");
    const char* tokens[kVocab] = {"<|endoftext|>", "a", "b", "c", "d", "e", "f", "\xC4\xA0"};
    gguf_set_arr_str(ctx.get(), "tokenizer.ggml.tokens", tokens, kVocab);
    std::int32_t token_types[kVocab] = {3, 1, 1, 1, 1, 1, 1, 1};
    gguf_set_arr_data(ctx.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types, kVocab);
    const char* merges[1] = {"a b"};
    gguf_set_arr_str(ctx.get(), "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.bos_token_id", 0u);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.eos_token_id", 0u);

    ggml_init_params gparams{std::size_t{4} << 20, nullptr, /*no_alloc=*/false};
    std::unique_ptr<ggml_context, GgmlDeleter> gctx(ggml_init(gparams));
    REQUIRE(gctx != nullptr);

    auto add_1d = [&](const char* tname, std::int64_t n0) -> ggml_tensor* {
        ggml_tensor* t = ggml_new_tensor_1d(gctx.get(), GGML_TYPE_F32, n0);
        REQUIRE(t != nullptr);
        ggml_set_name(t, tname);
        ggml_set_zero(t);
        gguf_add_tensor(ctx.get(), t);
        return t;
    };
    auto add_2d = [&](const char* tname, std::int64_t n0, std::int64_t n1) -> ggml_tensor* {
        ggml_tensor* t = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_F32, n0, n1);
        REQUIRE(t != nullptr);
        ggml_set_name(t, tname);
        ggml_set_zero(t);
        gguf_add_tensor(ctx.get(), t);
        return t;
    };

    ggml_tensor* tok_embd = add_2d("token_embd.weight", kEmbd, kVocab);
    ggml_tensor* out_norm = add_1d("output_norm.weight", kEmbd);
    ggml_tensor* out_w = add_2d("output.weight", kEmbd, kVocab);

    ggml_tensor* attn_norm = add_1d("blk.0.attn_norm.weight", kEmbd);
    add_2d("blk.0.attn_q.weight", kEmbd, kEmbd);
    add_2d("blk.0.attn_k.weight", kEmbd, kEmbd);
    add_2d("blk.0.attn_v.weight", kEmbd, kEmbd);
    add_2d("blk.0.attn_output.weight", kEmbd, kEmbd);
    ggml_tensor* ffn_norm = add_1d("blk.0.ffn_norm.weight", kEmbd);
    add_2d("blk.0.ffn_gate.weight", kEmbd, kFf);
    add_2d("blk.0.ffn_down.weight", kFf, kEmbd);
    add_2d("blk.0.ffn_up.weight", kEmbd, kFf);

    auto* emb_f = static_cast<float*>(tok_embd->data);
    auto* norm_f = static_cast<float*>(out_norm->data);
    auto* anorm_f = static_cast<float*>(attn_norm->data);
    auto* fnorm_f = static_cast<float*>(ffn_norm->data);
    auto* out_f = static_cast<float*>(out_w->data);
    for (std::int64_t i = 0; i < kEmbd; ++i) {
        norm_f[i] = 1.0f;
        anorm_f[i] = 1.0f;
        fnorm_f[i] = 1.0f;
    }
    for (std::int64_t t = 0; t < kVocab; ++t) {
        emb_f[t * kEmbd + t] = 1.0f;
        const std::int64_t next_tok = (t % (kVocab - 1)) + 1;
        out_f[next_tok * kEmbd + t] = 4.0f;
    }

    const std::string out_path = dir.file_utf8(filename);
    REQUIRE(gguf_write_to_file(ctx.get(), out_path.c_str(), /*only_meta=*/false));
    return out_path;
}

// Writes a tiny 1-layer F32 Qwen3 reranker GGUF (`qwen3.pooling_type = 4`, `cls.output.weight`).
std::string WriteTinyQwen3RerankerGguf(const TempGgufDir& dir, const std::string& filename,
                                       std::uint32_t n_ctx = 512u) {
    struct GgufDeleter {
        void operator()(gguf_context* c) const noexcept {
            if (c != nullptr) gguf_free(c);
        }
    };
    struct GgmlDeleter {
        void operator()(ggml_context* c) const noexcept {
            if (c != nullptr) ggml_free(c);
        }
    };

    constexpr std::int64_t kEmbd = 64;
    constexpr std::int64_t kHead = 1;
    constexpr std::int64_t kHeadKv = 1;
    constexpr std::int64_t kFf = 64;
    constexpr std::int64_t kVocab = 8;

    std::unique_ptr<gguf_context, GgufDeleter> ctx(gguf_init_empty());
    REQUIRE(ctx != nullptr);
    gguf_set_val_str(ctx.get(), "general.architecture", "qwen3");
    gguf_set_val_str(ctx.get(), "general.name", "nat12-tiny-qwen3-reranker");
    gguf_set_val_u32(ctx.get(), "qwen3.context_length", n_ctx);
    gguf_set_val_u32(ctx.get(), "qwen3.embedding_length", static_cast<std::uint32_t>(kEmbd));
    gguf_set_val_u32(ctx.get(), "qwen3.block_count", 1u);
    gguf_set_val_u32(ctx.get(), "qwen3.feed_forward_length", static_cast<std::uint32_t>(kFf));
    gguf_set_val_u32(ctx.get(), "qwen3.attention.head_count", static_cast<std::uint32_t>(kHead));
    gguf_set_val_u32(ctx.get(), "qwen3.attention.head_count_kv", static_cast<std::uint32_t>(kHeadKv));
    gguf_set_val_u32(ctx.get(), "qwen3.attention.key_length", static_cast<std::uint32_t>(kEmbd));
    gguf_set_val_u32(ctx.get(), "qwen3.attention.value_length", static_cast<std::uint32_t>(kEmbd));
    gguf_set_val_u32(ctx.get(), "qwen3.rope.dimension_count", static_cast<std::uint32_t>(kEmbd));
    gguf_set_val_f32(ctx.get(), "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    gguf_set_val_u32(ctx.get(), "qwen3.pooling_type", 4u);  // LLAMA_POOLING_TYPE_RANK
    gguf_set_val_str(ctx.get(), "tokenizer.chat_template.rerank", "Query:{query}\nDoc:{document}");

    gguf_set_val_str(ctx.get(), "tokenizer.ggml.model", "gpt2");
    gguf_set_val_str(ctx.get(), "tokenizer.ggml.pre", "default");
    const char* tokens[kVocab] = {"<|endoftext|>", "a", "b", "c", "Q", "D", ":", "\n"};
    gguf_set_arr_str(ctx.get(), "tokenizer.ggml.tokens", tokens, kVocab);
    std::int32_t token_types[kVocab] = {3, 1, 1, 1, 1, 1, 1, 1};
    gguf_set_arr_data(ctx.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types, kVocab);
    const char* merges[1] = {"a b"};
    gguf_set_arr_str(ctx.get(), "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.bos_token_id", 0u);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.eos_token_id", 0u);

    ggml_init_params gparams{std::size_t{4} << 20, nullptr, /*no_alloc=*/false};
    std::unique_ptr<ggml_context, GgmlDeleter> gctx(ggml_init(gparams));
    REQUIRE(gctx != nullptr);

    auto add_1d = [&](const char* tname, std::int64_t n0) -> ggml_tensor* {
        ggml_tensor* t = ggml_new_tensor_1d(gctx.get(), GGML_TYPE_F32, n0);
        REQUIRE(t != nullptr);
        ggml_set_name(t, tname);
        ggml_set_zero(t);
        gguf_add_tensor(ctx.get(), t);
        return t;
    };
    auto add_2d = [&](const char* tname, std::int64_t n0, std::int64_t n1) -> ggml_tensor* {
        ggml_tensor* t = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_F32, n0, n1);
        REQUIRE(t != nullptr);
        ggml_set_name(t, tname);
        ggml_set_zero(t);
        gguf_add_tensor(ctx.get(), t);
        return t;
    };

    ggml_tensor* tok_embd = add_2d("token_embd.weight", kEmbd, kVocab);
    ggml_tensor* out_norm = add_1d("output_norm.weight", kEmbd);
    add_2d("output.weight", kEmbd, kVocab);
    ggml_tensor* cls_out = add_2d("cls.output.weight", kEmbd, 1);

    ggml_tensor* attn_norm = add_1d("blk.0.attn_norm.weight", kEmbd);
    add_2d("blk.0.attn_q.weight", kEmbd, kEmbd);
    add_2d("blk.0.attn_k.weight", kEmbd, kEmbd);
    add_2d("blk.0.attn_v.weight", kEmbd, kEmbd);
    add_2d("blk.0.attn_output.weight", kEmbd, kEmbd);
    ggml_tensor* q_norm = add_1d("blk.0.attn_q_norm.weight", kEmbd);
    ggml_tensor* k_norm = add_1d("blk.0.attn_k_norm.weight", kEmbd);
    ggml_tensor* ffn_norm = add_1d("blk.0.ffn_norm.weight", kEmbd);
    add_2d("blk.0.ffn_gate.weight", kEmbd, kFf);
    add_2d("blk.0.ffn_down.weight", kFf, kEmbd);
    add_2d("blk.0.ffn_up.weight", kEmbd, kFf);

    auto* emb_f = static_cast<float*>(tok_embd->data);
    auto* norm_f = static_cast<float*>(out_norm->data);
    auto* cls_f = static_cast<float*>(cls_out->data);
    for (std::int64_t i = 0; i < kEmbd; ++i) {
        norm_f[i] = 1.0f;
        static_cast<float*>(attn_norm->data)[i] = 1.0f;
        static_cast<float*>(q_norm->data)[i] = 1.0f;
        static_cast<float*>(k_norm->data)[i] = 1.0f;
        static_cast<float*>(ffn_norm->data)[i] = 1.0f;
        cls_f[i] = 0.25f * static_cast<float>(i + 1);
    }
    for (std::int64_t t = 0; t < kVocab; ++t) {
        emb_f[t * kEmbd + (t % kEmbd)] = 1.0f + 0.1f * static_cast<float>(t);
    }

    const std::string out_path = dir.file_utf8(filename);
    REQUIRE(gguf_write_to_file(ctx.get(), out_path.c_str(), /*only_meta=*/false));
    return out_path;
}

// Fake CUDA Driver API state for deterministic, GPU-free primary-context stream testing.
struct MockCudaState {
    int device_count = 2;
    int least_priority = 0;
    int greatest_priority = -5;
    int retain_calls = 0;
    int release_calls = 0;
    int push_calls = 0;
    int pop_calls = 0;
    int create_calls = 0;
    int destroy_calls = 0;
    int sync_calls = 0;
    unsigned int last_flags = 0;
    int last_created_priority = 999;
    bool fail_create = false;
    bool fail_sync = false;
    std::uintptr_t next_handle = 0x100000u;
};

MockCudaState* g_mock_cuda = nullptr;

CudaDriverApi MakeMockCudaApi() {
    CudaDriverApi api;
    api.init = [](unsigned int) -> int { return g_mock_cuda != nullptr ? 0 : 1; };
    api.device_get_count = [](int* count) -> int {
        if (count == nullptr || g_mock_cuda == nullptr) return 1;
        *count = g_mock_cuda->device_count;
        return 0;
    };
    api.device_get = [](int* dev, int ord) -> int {
        if (dev == nullptr || g_mock_cuda == nullptr || ord < 0 || ord >= g_mock_cuda->device_count) {
            return 101;
        }
        *dev = 1000 + ord;
        return 0;
    };
    api.primary_ctx_retain = [](void** pctx, int dev) -> int {
        if (pctx == nullptr || g_mock_cuda == nullptr) return 1;
        ++g_mock_cuda->retain_calls;
        *pctx = reinterpret_cast<void*>(static_cast<std::uintptr_t>(dev));
        return 0;
    };
    api.primary_ctx_release = [](int) -> int {
        if (g_mock_cuda != nullptr) ++g_mock_cuda->release_calls;
        return 0;
    };
    api.ctx_push_current = [](void*) -> int {
        if (g_mock_cuda != nullptr) ++g_mock_cuda->push_calls;
        return 0;
    };
    api.ctx_pop_current = [](void** pctx) -> int {
        if (g_mock_cuda != nullptr) ++g_mock_cuda->pop_calls;
        if (pctx != nullptr) *pctx = nullptr;
        return 0;
    };
    api.ctx_get_stream_priority_range = [](int* least, int* greatest) -> int {
        if (g_mock_cuda == nullptr) return 1;
        if (least != nullptr) *least = g_mock_cuda->least_priority;
        if (greatest != nullptr) *greatest = g_mock_cuda->greatest_priority;
        return 0;
    };
    api.stream_create_with_priority = [](void** ph_stream, unsigned int flags, int prio) -> int {
        if (ph_stream == nullptr || g_mock_cuda == nullptr) return 1;
        if (g_mock_cuda->fail_create) return 2;
        ++g_mock_cuda->create_calls;
        g_mock_cuda->last_flags = flags;
        g_mock_cuda->last_created_priority = prio;
        const std::uintptr_t h = g_mock_cuda->next_handle;
        g_mock_cuda->next_handle += 0x10u;
        *ph_stream = reinterpret_cast<void*>(h);
        return 0;
    };
    api.stream_get_priority = [](void*, int* prio) -> int {
        if (prio == nullptr || g_mock_cuda == nullptr) return 1;
        *prio = g_mock_cuda->last_created_priority;
        return 0;
    };
    api.stream_synchronize = [](void* h) -> int {
        if (h == nullptr || g_mock_cuda == nullptr) return 1;
        ++g_mock_cuda->sync_calls;
        return g_mock_cuda->fail_sync ? 700 : 0;
    };
    api.stream_destroy = [](void* h) -> int {
        if (h == nullptr || g_mock_cuda == nullptr) return 1;
        ++g_mock_cuda->destroy_calls;
        return 0;
    };
    return api;
}

}  // namespace

TEST_CASE("NAT-12: stream priority parsing, role resolution, and CUDA priority convention") {
    CHECK(ToString(StreamPriority::kDefault) == "default");
    CHECK(ToString(StreamPriority::kHigh) == "high");
    CHECK(ToString(StreamPriority::kLow) == "low");

    StreamPriority sp = StreamPriority::kLow;
    CHECK(ParseStreamPriority("high", sp));
    CHECK(sp == StreamPriority::kHigh);
    CHECK(ParseStreamPriority("LOW", sp));
    CHECK(sp == StreamPriority::kLow);
    CHECK(ParseStreamPriority("default", sp));
    CHECK(sp == StreamPriority::kDefault);
    CHECK(ParseStreamPriority("Normal", sp));
    CHECK(sp == StreamPriority::kDefault);

    sp = StreamPriority::kHigh;
    CHECK_FALSE(ParseStreamPriority("", sp));
    CHECK_FALSE(ParseStreamPriority("realtime", sp));
    CHECK_FALSE(ParseStreamPriority(std::string_view("high\0bad", 8), sp));
    CHECK(sp == StreamPriority::kHigh);

    CHECK(ToString(WorkloadRole::kMainDecode) == "main_decode");
    CHECK(ToString(WorkloadRole::kAuxiliary) == "auxiliary");

    WorkloadRole role = WorkloadRole::kAuxiliary;
    CHECK(ParseWorkloadRole("main_decode", role));
    CHECK(role == WorkloadRole::kMainDecode);
    CHECK(ParseWorkloadRole("AUX", role));
    CHECK(role == WorkloadRole::kAuxiliary);
    CHECK(ParseWorkloadRole("Auxiliary", role));
    CHECK(role == WorkloadRole::kAuxiliary);
    CHECK(ParseWorkloadRole("main", role));
    CHECK(role == WorkloadRole::kMainDecode);
    CHECK_FALSE(ParseWorkloadRole("", role));
    CHECK_FALSE(ParseWorkloadRole("sidecar", role));
    CHECK_FALSE(ParseWorkloadRole(std::string_view("aux\0x", 5), role));

    // CUDA convention: numerically smaller values have higher hardware scheduling priority.
    CHECK(ResolveCudaNumericStreamPriority(0, -5, StreamPriority::kHigh) == -5);
    CHECK(ResolveCudaNumericStreamPriority(0, -5, StreamPriority::kLow) == 0);
    CHECK(ResolveCudaNumericStreamPriority(0, -5, StreamPriority::kDefault) == 0);

    CHECK(ResolveCudaNumericStreamPriority(2, -3, StreamPriority::kHigh) == -3);
    CHECK(ResolveCudaNumericStreamPriority(2, -3, StreamPriority::kLow) == 2);
    CHECK(ResolveCudaNumericStreamPriority(2, -3, StreamPriority::kDefault) == 0);
    CHECK(ResolveCudaNumericStreamPriority(-1, -4, StreamPriority::kDefault) == -1);

    // Degenerate (unsupported priority range) and inverted ranges never invert or UB.
    CHECK(ResolveCudaNumericStreamPriority(0, 0, StreamPriority::kHigh) == 0);
    CHECK(ResolveCudaNumericStreamPriority(0, 0, StreamPriority::kLow) == 0);
    CHECK(ResolveCudaNumericStreamPriority(-5, 0, StreamPriority::kHigh) == -5);
    CHECK(ResolveCudaNumericStreamPriority(-5, 0, StreamPriority::kLow) == 0);

    CHECK(EffectiveMainStreamPriority(WorkloadRole::kMainDecode, StreamPriority::kHigh) == StreamPriority::kHigh);
    CHECK(EffectiveMainStreamPriority(WorkloadRole::kMainDecode, StreamPriority::kDefault) == StreamPriority::kDefault);
    CHECK(EffectiveMainStreamPriority(WorkloadRole::kAuxiliary, StreamPriority::kHigh) == StreamPriority::kLow);
    CHECK(EffectiveMainStreamPriority(WorkloadRole::kAuxiliary, StreamPriority::kDefault) == StreamPriority::kLow);

    // Nested RAII thread-local priority hints restore exact prior state and update counters.
    ResetGgmlCudaStreamPriorityStatsForTest();
    CHECK(sonder_ggml_cuda_stream_priority_hint() == SONDER_CUDA_STREAM_PRIO_DEFAULT);
    {
        ScopedGgmlCudaStreamPriority outer(StreamPriority::kHigh);
        CHECK(sonder_ggml_cuda_stream_priority_hint() == SONDER_CUDA_STREAM_PRIO_HIGH);
        {
            ScopedGgmlCudaStreamPriority inner(StreamPriority::kLow);
            CHECK(sonder_ggml_cuda_stream_priority_hint() == SONDER_CUDA_STREAM_PRIO_LOW);
        }
        CHECK(sonder_ggml_cuda_stream_priority_hint() == SONDER_CUDA_STREAM_PRIO_HIGH);
    }
    CHECK(sonder_ggml_cuda_stream_priority_hint() == SONDER_CUDA_STREAM_PRIO_DEFAULT);
    const GgmlCudaStreamPriorityStats st = GetGgmlCudaStreamPriorityStats();
    CHECK(st.high_priority_scopes == 1u);
    CHECK(st.low_priority_scopes == 1u);
}

TEST_CASE("NAT-12: primary-context CudaPriorityStream and TensorRT-RTX user_compute_stream") {
    MockCudaState mock;
    g_mock_cuda = &mock;
    const CudaDriverApi api = MakeMockCudaApi();
    REQUIRE(api.complete());

    {
        CudaPriorityStream low_stream;
        const Status s = CudaPriorityStream::Create(0, StreamPriority::kLow, low_stream, &api);
        REQUIRE(s.ok());
        CHECK(low_stream.valid());
        CHECK(low_stream.device_ordinal() == 0);
        CHECK(low_stream.requested_priority() == StreamPriority::kLow);
        CHECK(low_stream.least_priority() == 0);
        CHECK(low_stream.greatest_priority() == -5);
        CHECK(low_stream.priority_range_supported());
        CHECK(low_stream.numeric_priority() == 0);
        CHECK(mock.last_flags == CudaDriverApi::kStreamNonBlocking);
        CHECK(mock.retain_calls == 1);
        CHECK(mock.push_calls == 1);
        CHECK(mock.pop_calls == 1);
        CHECK(low_stream.user_compute_stream() != nullptr);

        const auto expected_ptr = reinterpret_cast<std::uintptr_t>(low_stream.user_compute_stream());
        CHECK(low_stream.FormatUserComputeStreamOption() == std::to_string(expected_ptr));

        // TensorRT-RTX EP options default enable_cuda_graph to "0" on low-priority aux streams,
        // and allow explicit opt-in ("1") + runtime cache path.
        const TrtRtxProviderOptions opts_default = low_stream.BuildTrtRtxProviderOptions();
        REQUIRE(opts_default.Find("user_compute_stream").has_value());
        CHECK(*opts_default.Find("user_compute_stream") == std::to_string(expected_ptr));
        CHECK(*opts_default.Find("device_id") == "0");
        CHECK(*opts_default.Find("enable_cuda_graph") == "0");
        CHECK_FALSE(opts_default.Find("nv_runtime_cache_path").has_value());

        const TrtRtxProviderOptions opts_graph =
            low_stream.BuildTrtRtxProviderOptions(/*enable_cuda_graph=*/true, "D:/cache/trt");
        CHECK(*opts_graph.Find("enable_cuda_graph") == "1");
        REQUIRE(opts_graph.Find("nv_runtime_cache_path").has_value());
        CHECK(*opts_graph.Find("nv_runtime_cache_path") == "D:/cache/trt");

        REQUIRE(low_stream.Synchronize().ok());
        CHECK(mock.sync_calls == 1);

        // Move semantics transfer ownership without double-free or premature release.
        CudaPriorityStream moved = std::move(low_stream);
        CHECK_FALSE(low_stream.valid());
        CHECK(moved.valid());
        CHECK(mock.destroy_calls == 0);
        CHECK(mock.release_calls == 0);

        CudaPriorityStream high_stream;
        REQUIRE(CudaPriorityStream::Create(1, StreamPriority::kHigh, high_stream, &api).ok());
        CHECK(high_stream.numeric_priority() == -5);
    }
    // Both RAII streams destroyed and both retained primary contexts released cleanly.
    CHECK(mock.destroy_calls == 2);
    CHECK(mock.release_calls == 2);
    CHECK(mock.push_calls == mock.pop_calls);

    // Hostile / error cases fail closed with zero context or handle leaks.
    CudaPriorityStream bad;
    CHECK(CudaPriorityStream::Create(-1, StreamPriority::kLow, bad, &api).code == ErrorCode::kInvalidArgument);
    CHECK(CudaPriorityStream::Create(99, StreamPriority::kLow, bad, &api).code == ErrorCode::kInvalidArgument);

    mock.fail_create = true;
    const int retains_before = mock.retain_calls;
    const int releases_before = mock.release_calls;
    CHECK_FALSE(CudaPriorityStream::Create(0, StreamPriority::kLow, bad, &api).ok());
    CHECK_FALSE(bad.valid());
    CHECK(mock.retain_calls == retains_before + 1);
    CHECK(mock.release_calls == releases_before + 1);
    CHECK(mock.push_calls == mock.pop_calls);

    CudaDriverApi incomplete_api = api;
    incomplete_api.stream_synchronize = nullptr;
    CHECK_FALSE(incomplete_api.complete());
    CHECK(CudaPriorityStream::Create(0, StreamPriority::kLow, bad, &incomplete_api).code ==
          ErrorCode::kUnsupported);

    g_mock_cuda = nullptr;
}

TEST_CASE("NAT-12: AuxGateConfig validation and execution domain isolation") {
    AuxGateConfig cfg;
    CHECK(ValidateAuxGateConfig(cfg).ok());

    AuxGateConfig bad = cfg;
    bad.max_aux_duty_cycle = 0.0;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.max_aux_duty_cycle = -0.05;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.max_aux_duty_cycle = 0.75;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.max_aux_duty_cycle = std::numeric_limits<double>::quiet_NaN();
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.max_aux_duty_cycle = std::numeric_limits<double>::infinity();
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);

    bad = cfg;
    bad.active_slice_max_tokens = 0;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.active_slice_max_tokens = kMaxAuxActiveSliceTokens + 1u;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);

    bad = cfg;
    bad.active_slice_max_sequences = 0;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.active_slice_max_sequences = kMaxEmbedSequences + 1u;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);

    bad = cfg;
    bad.max_decode_hold_ms = 0;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.max_decode_hold_ms = kMaxAuxGateTimeoutMs + 1u;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);

    bad = cfg;
    bad.handoff_pickup_timeout_ms = 0;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.handoff_pickup_timeout_ms = kMaxAuxGateTimeoutMs + 1u;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);

    bad = cfg;
    bad.max_starvation_wait_ms = 0u;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);
    bad.max_starvation_wait_ms = kMaxAuxGateTimeoutMs + 1u;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);

    bad = cfg;
    bad.cancel_poll_interval_ms = 0;
    CHECK(ValidateAuxGateConfig(bad).code == ErrorCode::kInvalidArgument);

    // EffectiveMainStreamPriority honors configured aux_stream_priority for auxiliary models.
    CHECK(EffectiveMainStreamPriority(WorkloadRole::kAuxiliary, StreamPriority::kHigh, StreamPriority::kDefault) ==
          StreamPriority::kDefault);
    CHECK(EffectiveMainStreamPriority(WorkloadRole::kAuxiliary, StreamPriority::kHigh, StreamPriority::kLow) ==
          StreamPriority::kLow);

    // Core LlamaCppBackendOptions validation maps and rejects hostile NAT-12 knobs.
    sonder::inference::LlamaCppBackendOptions bo;
    CHECK(sonder::inference::validate_llamacpp_options(bo).ok());
    bo.workload_role = "auxiliary";
    bo.main_stream_priority = "low";
    bo.aux_stream_priority = "low";
    CHECK(sonder::inference::validate_llamacpp_options(bo).ok());

    bo.workload_role = "invalid_role";
    CHECK(sonder::inference::validate_llamacpp_options(bo).code() == sonder::inference::ErrorCode::invalid_argument);
    bo.workload_role = "main_decode";
    bo.aux_stream_priority = "ultra";
    CHECK(sonder::inference::validate_llamacpp_options(bo).code() == sonder::inference::ErrorCode::invalid_argument);
    bo.aux_stream_priority = "low";
    bo.max_aux_duty_cycle = 0.99;
    CHECK(sonder::inference::validate_llamacpp_options(bo).code() == sonder::inference::ErrorCode::invalid_argument);

    // Execution domain resolution canonicalizes vendor ordinals ("CUDA0", "Vulkan0" -> "gpu:0")
    // so backends configured via main_gpu or explicit devices coordinate on the same GPU.
    CHECK(ResolveAuxExecutionDomain(0, 0, {"CUDA0"}) == "cpu");
    CHECK(ResolveAuxExecutionDomain(-1, 0, {}) == "gpu:0");
    CHECK(ResolveAuxExecutionDomain(32, 1, {}) == "gpu:1");
    CHECK(ResolveAuxExecutionDomain(-1, 0, {"CUDA0"}) == "gpu:0");
    CHECK(ResolveAuxExecutionDomain(-1, 0, {"Vulkan0"}) == "gpu:0");
    CHECK(ResolveAuxExecutionDomain(-1, 0, {"Vulkan1"}) == "gpu:1");
    CHECK(ResolveAuxExecutionDomain(-1, 0, {"CUDA0", "CUDA1"}) == "gpu:0+1");

    // Active decode on gpu:0 never stalls CPU-offloaded auxiliary slices, and CPU-only
    // sessions bypass gating by default unless gate_cpu_domain is explicitly enabled.
    DecodeAwareAuxGate gate;
    auto cpu_default_sess = gate.EnterDecodeSession("cpu");
    CHECK_FALSE(cpu_default_sess.active());
    CHECK_FALSE(gate.IsDecodeActive("cpu"));
    auto gpu_sess = gate.EnterDecodeSession("gpu:0");
    auto gpu_step = gate.EnterDecodeStep("gpu:0");
    CHECK(gate.IsDecodeActive("gpu:0"));
    CHECK(gate.IsDecodeStepInFlight("gpu:0"));
    CHECK_FALSE(gate.IsDecodeActive("cpu"));
    {
        auto cpu_slice = gate.AcquireAuxSlice("cpu");
        CHECK(cpu_slice.status() == AuxAcquireStatus::kAcquiredIdle);
    }
    CHECK(gate.GetStats("cpu").aux_slices_idle_fastpath == 1u);
}

TEST_CASE("NAT-12: DecodeAwareAuxGate DRR accounting enforces zero overlap and configured 5 percent duty cap") {
    MockCudaState mock;
    g_mock_cuda = &mock;
    const CudaDriverApi api = MakeMockCudaApi();
    CudaPriorityStream aux_stream;
    REQUIRE(CudaPriorityStream::Create(0, StreamPriority::kLow, aux_stream, &api).ok());

    // Deterministic atomic clock eliminates OS sleep-granularity flakiness on DRR duty accounting,
    // while real threads still contend concurrently to verify strict mutual exclusion.
    std::atomic<std::int64_t> fake_clock_ns{1'000'000'000LL};
    AuxGateConfig cfg;
    cfg.enabled = true;
    cfg.max_aux_duty_cycle = 0.05;  // Configured 5% DRR duty target
    cfg.max_decode_hold_ms = 200;
    cfg.handoff_pickup_timeout_ms = 25;
    cfg.max_starvation_wait_ms = 5000;
    DecodeAwareAuxGate gate(cfg, [&] { return fake_clock_ns.load(std::memory_order_relaxed); });

    std::atomic<bool> in_decode_step{false};
    std::atomic<bool> in_aux_slice{false};
    std::atomic<std::size_t> observed_overlaps{0};
    std::atomic<bool> decode_started{false};
    std::atomic<bool> stop_aux{false};

    std::size_t total_aux_slices_while_decoding = 0;
    std::thread aux_thread([&] {
        while (!decode_started.load(std::memory_order_acquire)) {
            std::this_thread::yield();
        }
        while (!stop_aux.load(std::memory_order_acquire)) {
            AuxSliceRunSummary sum;
            const Status st = RunGatedAuxSlices(
                gate, "gpu:0", 1u,
                [&](std::size_t, bool decode_active) -> Status {
                    in_aux_slice.store(true, std::memory_order_seq_cst);
                    if (in_decode_step.load(std::memory_order_seq_cst)) {
                        observed_overlaps.fetch_add(1, std::memory_order_relaxed);
                    }
                    fake_clock_ns.fetch_add(2'000'000LL, std::memory_order_relaxed);  // 2.0 ms slice
                    std::this_thread::yield();
                    if (in_decode_step.load(std::memory_order_seq_cst)) {
                        observed_overlaps.fetch_add(1, std::memory_order_relaxed);
                    }
                    in_aux_slice.store(false, std::memory_order_seq_cst);
                    if (decode_active) {
                        ++total_aux_slices_while_decoding;
                    }
                    return Status::Ok();
                },
                [&] { return stop_aux.load(std::memory_order_acquire); },
                StreamPriority::kLow, &aux_stream, &sum);
            if (!st.ok()) break;
        }
    });

    {
        auto session = gate.EnterDecodeSession("gpu:0");
        for (int step = 0; step < 40; ++step) {
            {
                auto step_scope = gate.EnterDecodeStep("gpu:0");
                in_decode_step.store(true, std::memory_order_seq_cst);
                if (in_aux_slice.load(std::memory_order_seq_cst)) {
                    observed_overlaps.fetch_add(1, std::memory_order_relaxed);
                }
                decode_started.store(true, std::memory_order_release);
                fake_clock_ns.fetch_add(4'000'000LL, std::memory_order_relaxed);  // 4.0 ms step
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                if (in_aux_slice.load(std::memory_order_seq_cst)) {
                    observed_overlaps.fetch_add(1, std::memory_order_relaxed);
                }
                in_decode_step.store(false, std::memory_order_seq_cst);
                step_scope.MarkSynchronized();
            }
        }
        stop_aux.store(true, std::memory_order_release);
    }
    aux_thread.join();

    const AuxGateStats stats = gate.GetStats("gpu:0");
    CHECK(observed_overlaps.load() == 0u);
    CHECK(stats.aux_slices_overlapped_decode == 0u);
    CHECK(stats.decode_steps_total == 40u);
    CHECK(stats.decode_steps_synced_before_release == 40u);
    CHECK(stats.aux_slices_gated_between_steps >= 2u);
    CHECK(stats.aux_slices_synced_before_release == stats.aux_slices_total);
    CHECK(mock.sync_calls >= static_cast<int>(stats.aux_slices_total));

    const double total_gpu_busy_ns =
        static_cast<double>(stats.decode_busy_ns) + static_cast<double>(stats.aux_busy_ns);
    REQUIRE(total_gpu_busy_ns > 0.0);
    const double decode_share = static_cast<double>(stats.decode_busy_ns) / total_gpu_busy_ns;
    CAPTURE(stats.decode_busy_ns);
    CAPTURE(stats.aux_busy_ns);
    CAPTURE(decode_share);
    CHECK(decode_share >= 0.945);

    // Verify prompt cooperative cancellation while blocked on an active decode step.
    {
        auto session = gate.EnterDecodeSession("gpu:0");
        auto held_step = gate.EnterDecodeStep("gpu:0");
        std::atomic<bool> cancel_now{false};
        std::thread canceller([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(6));
            cancel_now.store(true, std::memory_order_release);
        });
        const auto t0 = std::chrono::steady_clock::now();
        auto cancelled_slice = gate.AcquireAuxSlice("gpu:0", [&] {
            return cancel_now.load(std::memory_order_acquire);
        });
        const auto cancel_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
        canceller.join();
        CHECK(cancelled_slice.status() == AuxAcquireStatus::kCancelled);
        CHECK_FALSE(cancelled_slice.acquired());
        CHECK(cancel_ms < 200);
        CHECK(gate.GetStats("gpu:0").waiting_aux_slices == 0u);
    }

    g_mock_cuda = nullptr;
}

TEST_CASE("NAT-12: SubslicePlannedBatch and PlanSttAudioSlices cap active slices at <=512 tokens") {
    // 1. SubslicePlannedBatch splits an idle-planned 2048-token batch into <=512-token slices
    // with contiguous 0..k-1 seq_ids and exact original_index preservation.
    PlannedEmbedBatch big;
    for (std::size_t i = 0; i < 8u; ++i) {
        big.items.push_back(EmbedBatchItem{100u + i, 240u, static_cast<std::int32_t>(i)});
        big.total_tokens += 240u;
    }
    const auto subs = SubslicePlannedBatch(big, /*max_slice_tokens=*/512u, /*max_slice_sequences=*/4u);
    REQUIRE(subs.size() == 4u);
    std::vector<std::size_t> seen_indices;
    for (const auto& sb : subs) {
        CHECK(sb.total_tokens <= 512u);
        CHECK(sb.items.size() == 2u);
        CHECK(sb.items[0].seq_id == 0);
        CHECK(sb.items[1].seq_id == 1);
        seen_indices.push_back(sb.items[0].original_index);
        seen_indices.push_back(sb.items[1].original_index);
    }
    CHECK(seen_indices == std::vector<std::size_t>{100, 101, 102, 103, 104, 105, 106, 107});

    // Single oversized fallback item (>512 tokens) stays intact as an isolated 1-item slice.
    PlannedEmbedBatch over;
    over.oversized_fallback = true;
    over.total_tokens = 900u;
    over.items.push_back(EmbedBatchItem{7u, 900u, 0});
    const auto over_subs = SubslicePlannedBatch(over, 512u, 4u);
    REQUIRE(over_subs.size() == 1u);
    CHECK(over_subs[0].oversized_fallback);
    CHECK(over_subs[0].total_tokens == 900u);
    CHECK(over_subs[0].items[0].original_index == 7u);

    // 2. PlanSttAudioSlices: 60s of 16 kHz PCM = 960,000 samples => 1,500 encoder tokens
    // (160 hop * 4x subsampling = 640 samples per encoder token).
    SttSlicePlanConfig stt_idle;
    stt_idle.sample_rate_hz = 16000;
    stt_idle.frame_hop_samples = 160;
    stt_idle.encoder_downsample = 4;
    stt_idle.idle_slice_max_tokens = 2048;
    stt_idle.active_slice_max_tokens = 512;
    stt_idle.decode_active = false;

    const SttSlicePlan idle_plan = PlanSttAudioSlices(960000u, stt_idle);
    REQUIRE(idle_plan.ok);
    CHECK(idle_plan.total_encoder_tokens == 1500u);
    REQUIRE(idle_plan.slices.size() == 1u);
    CHECK(idle_plan.slices[0].encoder_tokens == 1500u);
    CHECK(idle_plan.slices[0].sample_offset == 0u);
    CHECK(idle_plan.slices[0].sample_count == 960000u);

    // Active decode caps each STT slice at <= 512 encoder tokens
    // (new tokens: 512 + 500 + 488 = 1500; total tokens w/ 12-token overlap: 512, 512, 500).
    SttSlicePlanConfig stt_active = stt_idle;
    stt_active.decode_active = true;
    stt_active.overlap_tokens = 12;
    const SttSlicePlan active_plan = PlanSttAudioSlices(960000u, stt_active);
    REQUIRE(active_plan.ok);
    CHECK(active_plan.total_encoder_tokens == 1500u);
    CHECK(active_plan.max_slice_tokens_used <= 512u);
    REQUIRE(active_plan.slices.size() == 3u);
    CHECK(active_plan.slices[0].overlap_prefix_tokens == 0u);
    CHECK(active_plan.slices[0].encoder_tokens == 512u);
    CHECK(active_plan.slices[1].encoder_tokens == 512u);
    CHECK(active_plan.slices[2].encoder_tokens == 500u);
    for (std::size_t i = 1; i < active_plan.slices.size(); ++i) {
        CHECK(active_plan.slices[i].overlap_prefix_tokens == 12u);
        CHECK(active_plan.slices[i].encoder_tokens <= 512u);
    }

    // Hostile STT inputs fail closed without division-by-zero or uint64 overflow.
    CHECK_FALSE(PlanSttAudioSlices(0u, stt_active).ok);
    SttSlicePlanConfig bad_stt = stt_active;
    bad_stt.frame_hop_samples = 0;
    CHECK_FALSE(PlanSttAudioSlices(16000u, bad_stt).ok);
    bad_stt = stt_active;
    bad_stt.encoder_downsample = 0;
    CHECK_FALSE(PlanSttAudioSlices(16000u, bad_stt).ok);
    bad_stt = stt_active;
    bad_stt.overlap_tokens = 512;
    CHECK_FALSE(PlanSttAudioSlices(16000u, bad_stt).ok);
    CHECK_FALSE(PlanSttAudioSlices(std::numeric_limits<std::uint64_t>::max(), stt_active).ok);
}

TEST_CASE("NAT-12: end-to-end synthetic GGUF co-resident main decode plus gated Embed ScorePairs and aux STT") {
    TempGgufDir dir;
    const std::string main_gguf = WriteTinyLlamaGguf(dir, "main-llm.gguf", /*n_ctx=*/512u);
    const std::string embed_gguf = WriteTinyLlamaGguf(dir, "aux-embed.gguf", /*n_ctx=*/512u);
    const std::string rerank_gguf = WriteTinyQwen3RerankerGguf(dir, "aux-rerank.gguf", /*n_ctx=*/512u);

    ResetGgmlCudaStreamPriorityStatsForTest();

    auto shared_gate = std::make_shared<DecodeAwareAuxGate>();
    AuxGateConfig gate_cfg;
    gate_cfg.enabled = true;
    gate_cfg.gate_cpu_domain = true;          // Explicitly enable gating on CPU domain for synthetic fixture
    gate_cfg.max_aux_duty_cycle = 0.05;
    gate_cfg.active_slice_max_tokens = 64u;   // Cap active slices to <= 64 tokens for small 512-ctx fixture
    gate_cfg.active_slice_max_sequences = 2u; // <= 2 sequences per slice during decode
    REQUIRE(shared_gate->Reconfigure(gate_cfg).ok());

    LlamaCppBackend main_be;
    LoadOptions lo_main;
    lo_main.model_path = main_gguf;
    lo_main.n_ctx = 512u;
    lo_main.n_batch = 128u;
    lo_main.n_slots = 2u;  // Exercises continuous-batching BatchWorkerLoop decode gating!
    lo_main.workload_role = WorkloadRole::kMainDecode;
    lo_main.main_stream_priority = StreamPriority::kHigh;
    lo_main.aux_gate = gate_cfg;
    lo_main.shared_aux_gate = shared_gate;
    REQUIRE(main_be.Load(lo_main).ok());
    CHECK(main_be.GetAuxGate() == shared_gate);
    CHECK(main_be.GetAuxExecutionDomain() == "cpu");

    LlamaCppBackend embed_be;
    LoadOptions lo_emb;
    lo_emb.model_path = embed_gguf;
    lo_emb.n_ctx = 512u;
    lo_emb.n_batch = 256u;
    lo_emb.embed_batch_tokens = 256u;
    lo_emb.embed_max_sequences = 8u;
    lo_emb.workload_role = WorkloadRole::kAuxiliary;
    lo_emb.aux_stream_priority = StreamPriority::kLow;
    lo_emb.aux_gate = gate_cfg;
    lo_emb.shared_aux_gate = shared_gate;
    REQUIRE(embed_be.Load(lo_emb).ok());

    LlamaCppBackend rerank_be;
    LoadOptions lo_rerank = lo_emb;
    lo_rerank.model_path = rerank_gguf;
    REQUIRE(rerank_be.Load(lo_rerank).ok());
    REQUIRE(rerank_be.SupportsReranking());

    // Build 6 inputs of 24 tokens each = 144 tokens total:
    //   - When main LLM is IDLE: fits in 1 batch (144 <= 256 tokens, 6 <= 8 seqs).
    //   - When main LLM is ACTIVE: sliced to <= 64 tokens and <= 2 seqs => 3 slices of 2 seqs (48 tokens each)!
    std::vector<std::vector<Token>> emb_inputs(6, std::vector<Token>(24, 1));
    for (std::size_t i = 0; i < emb_inputs.size(); ++i) {
        emb_inputs[i][0] = static_cast<Token>((i % 6u) + 1u);
    }

    EmbedOutput idle_emb;
    REQUIRE(embed_be.Embed(emb_inputs, /*normalize=*/true, idle_emb).ok());
    REQUIRE(idle_emb.embeddings.size() == 6u);
    CHECK(idle_emb.batches_executed == 1u);
    CHECK(idle_emb.gated_slices == 0u);

    std::vector<std::vector<Token>> score_pairs(4, std::vector<Token>(20, 2));
    for (std::size_t i = 0; i < score_pairs.size(); ++i) {
        score_pairs[i][0] = static_cast<Token>((i % 6u) + 1u);
    }
    ScoreOutput idle_scores;
    REQUIRE(rerank_be.ScorePairs(score_pairs, idle_scores).ok());
    REQUIRE(idle_scores.scores.size() == 4u);
    CHECK(idle_scores.batches_executed == 1u);
    CHECK(idle_scores.gated_slices == 0u);

    // Run continuous multi-slot main_be.Generate() until all auxiliary workloads complete
    // so gated slice assertions never depend on fixed wall-clock sleep durations.
    std::atomic<bool> main_running{false};
    std::atomic<bool> aux_done{false};
    std::int64_t main_generated_total = 0;
    Status main_last_status = Status::Ok();

    std::thread main_thread([&] {
        auto keep_session_open = shared_gate->EnterDecodeSession("cpu");
        GenerateRequest greq;
        greq.prompt = {1, 2, 3, 4};
        greq.max_new_tokens = 32;
        greq.sampling.temperature = 0.0f;
        do {
            const GenerateResult r = main_be.Generate(greq, [&](const TokenEvent&) {
                main_running.store(true, std::memory_order_release);
                std::this_thread::yield();
                return true;
            });
            main_last_status = r.status;
            main_generated_total += r.generated_tokens;
            if (!r.status.ok()) break;
        } while (!aux_done.load(std::memory_order_acquire) || main_generated_total < 24);
    });

    while (!main_running.load(std::memory_order_acquire)) {
        std::this_thread::yield();
    }

    EmbedOutput gated_emb;
    REQUIRE(embed_be.Embed(emb_inputs, /*normalize=*/true, gated_emb).ok());
    REQUIRE(gated_emb.embeddings.size() == 6u);
    CHECK(gated_emb.batches_executed == 3u);
    CHECK(gated_emb.gated_slices >= 1u);

    // Numerical parity: sliced embeddings match idle full-batch embeddings within FP32 tolerance.
    for (std::size_t i = 0; i < 6u; ++i) {
        REQUIRE(gated_emb.embeddings[i].size() == idle_emb.embeddings[i].size());
        for (std::size_t d = 0; d < gated_emb.embeddings[i].size(); ++d) {
            CHECK(std::fabs(gated_emb.embeddings[i][d] - idle_emb.embeddings[i][d]) < 1e-5f);
        }
    }

    // ScorePairs on rerank_be also slices to <= 2 sequences / <= 64 tokens under active decode
    // and preserves exact numerical parity with idle full-batch scores.
    ScoreOutput gated_scores;
    REQUIRE(rerank_be.ScorePairs(score_pairs, gated_scores).ok());
    REQUIRE(gated_scores.scores.size() == 4u);
    CHECK(gated_scores.batches_executed == 2u);
    CHECK(gated_scores.gated_slices >= 1u);
    for (std::size_t i = 0; i < 4u; ++i) {
        CHECK(std::fabs(gated_scores.scores[i] - idle_scores.scores[i]) < 1e-5);
    }

    // Auxiliary-role Generate runs on low-priority streams and slices prefill + decode
    // through AuxSliceScope without registering as main decode.
    GenerateRequest stt_req;
    stt_req.prompt = std::vector<Token>(100, 1);  // 100 tokens > 64 active_slice_max_tokens => 2 prefill slices!
    stt_req.max_new_tokens = 3;
    stt_req.sampling.temperature = 0.0f;
    const GenerateResult stt_res = embed_be.Generate(stt_req, {});
    REQUIRE(stt_res.status.ok());
    CHECK(stt_res.generated_tokens == 3);

    aux_done.store(true, std::memory_order_release);
    main_thread.join();

    REQUIRE(main_last_status.ok());
    CHECK(main_generated_total >= 24);

    const AuxGateStats gstats = shared_gate->GetStats("cpu");
    CHECK(gstats.aux_slices_overlapped_decode == 0u);
    CHECK(gstats.decode_steps_total >= 24u);
    CHECK(gstats.aux_slices_gated_between_steps >= 1u);

    const GgmlCudaStreamPriorityStats pstats = GetGgmlCudaStreamPriorityStats();
    CHECK(pstats.high_priority_scopes >= 24u);
    CHECK(pstats.low_priority_scopes >= 6u);

    // Also verify LlamaCppCoreBackend shares one DecodeAwareAuxGate across co-resident BackendModels.
    sonder::inference::LlamaCppBackendOptions core_opts;
    core_opts.context_length = 256u;
    core_opts.aux_active_slice_tokens = 64u;
    core_opts.aux_active_max_sequences = 2u;
    auto core_be = sonder::inference::make_llamacpp_backend(core_opts);
    REQUIRE(core_be != nullptr);
    sonder::inference::ModelLoadOptions mlo1{main_gguf, "cpu:0"};
    sonder::inference::ModelLoadOptions mlo2{embed_gguf, "cpu:0"};
    auto m1 = core_be->load_model(mlo1);
    auto m2 = core_be->load_model(mlo2);
    REQUIRE(m1.ok());
    REQUIRE(m2.ok());
    sonder::inference::EmbedRequest core_ereq;
    core_ereq.inputs = {"a b c", "d e f", "a c e"};
    sonder::inference::CancellationToken cancel;
    auto er = m2.value()->embed(core_ereq, cancel);
    REQUIRE(er.ok());
    CHECK(er.value().embeddings.size() == 3u);
}

TEST_CASE("NAT-12: llama_synchronize executes inside DecodeStepScope and AuxSliceScope before gate release") {
    TempGgufDir dir;
    const std::string main_gguf = WriteTinyLlamaGguf(dir, "sync-main.gguf", /*n_ctx=*/512u);
    const std::string rerank_gguf = WriteTinyQwen3RerankerGguf(dir, "sync-rerank.gguf", /*n_ctx=*/512u);

    auto shared_gate = std::make_shared<DecodeAwareAuxGate>();
    AuxGateConfig gate_cfg;
    gate_cfg.enabled = true;
    gate_cfg.gate_cpu_domain = true;
    gate_cfg.max_aux_duty_cycle = 0.20;
    gate_cfg.active_slice_max_tokens = 64u;
    gate_cfg.active_slice_max_sequences = 2u;
    REQUIRE(shared_gate->Reconfigure(gate_cfg).ok());

    struct SyncRecorder {
        std::mutex mu;
        std::size_t decode_syncs = 0;
        std::size_t decode_syncs_while_in_flight = 0;
        std::size_t aux_syncs = 0;
        std::size_t aux_syncs_while_in_flight = 0;
    } rec;

    struct ClearObserverGuard {
        ~ClearObserverGuard() { SetAuxContextSyncObserverForTest({}); }
    } clear_obs;

    SetAuxContextSyncObserverForTest([&](const AuxContextSyncObservation& obs) {
        if (obs.domain != "cpu") return;
        std::lock_guard<std::mutex> lock(rec.mu);
        if (obs.is_aux_slice) {
            ++rec.aux_syncs;
            if (obs.aux_slice_in_flight_at_sync) {
                ++rec.aux_syncs_while_in_flight;
            }
        } else {
            ++rec.decode_syncs;
            if (obs.decode_step_in_flight_at_sync) {
                ++rec.decode_syncs_while_in_flight;
            }
        }
    });

    // 1. Single-slot (n_slots=1 -> GenerateSingleLocked) AND multi-slot (n_slots=2 -> BatchWorkerLoop)
    //    must synchronize `ctx` BEFORE releasing DecodeStepScope.
    for (const std::uint32_t slots : {1u, 2u}) {
        LlamaCppBackend main_be;
        LoadOptions lo;
        lo.model_path = main_gguf;
        lo.n_ctx = 512u;
        lo.n_batch = 64u;
        lo.n_slots = slots;
        lo.workload_role = WorkloadRole::kMainDecode;
        lo.aux_gate = gate_cfg;
        lo.shared_aux_gate = shared_gate;
        REQUIRE(main_be.Load(lo).ok());

        GenerateRequest req;
        req.prompt = {1, 2, 3, 4, 5};
        req.max_new_tokens = 4;
        req.sampling.temperature = 0.0f;
        const GenerateResult res = main_be.Generate(req, {});
        REQUIRE(res.status.ok());
        CHECK(res.generated_tokens == 4);
    }

    // 2. Embed, ScorePairs, and auxiliary-role Generate must synchronize `ectx`/`rctx`/`ctx`
    //    BEFORE releasing AuxSliceScope.
    LlamaCppBackend aux_embed_be;
    LoadOptions lo_aux;
    lo_aux.model_path = main_gguf;
    lo_aux.n_ctx = 512u;
    lo_aux.n_batch = 64u;
    lo_aux.workload_role = WorkloadRole::kAuxiliary;
    lo_aux.aux_gate = gate_cfg;
    lo_aux.shared_aux_gate = shared_gate;
    REQUIRE(aux_embed_be.Load(lo_aux).ok());

    EmbedOutput emb_out;
    REQUIRE(aux_embed_be.Embed({{1, 2, 3}, {4, 5, 6}}, /*normalize=*/true, emb_out).ok());

    GenerateRequest aux_gen_req;
    aux_gen_req.prompt = {1, 2, 3, 4};
    aux_gen_req.max_new_tokens = 2;
    aux_gen_req.sampling.temperature = 0.0f;
    const GenerateResult aux_gen_res = aux_embed_be.Generate(aux_gen_req, {});
    REQUIRE(aux_gen_res.status.ok());

    LlamaCppBackend aux_rerank_be;
    LoadOptions lo_rr = lo_aux;
    lo_rr.model_path = rerank_gguf;
    REQUIRE(aux_rerank_be.Load(lo_rr).ok());
    ScoreOutput score_out;
    REQUIRE(aux_rerank_be.ScorePairs({{1, 2, 3}, {2, 3, 4}}, score_out).ok());

    const AuxGateStats st = shared_gate->GetStats("cpu");
    REQUIRE(st.decode_steps_total >= 8u);
    CHECK(st.decode_steps_synced_before_release == st.decode_steps_total);
    REQUIRE(st.aux_slices_total >= 4u);
    CHECK(st.aux_slices_synced_before_release == st.aux_slices_total);

    {
        std::lock_guard<std::mutex> lock(rec.mu);
        CHECK(rec.decode_syncs == st.decode_steps_total);
        CHECK(rec.decode_syncs_while_in_flight == rec.decode_syncs);
        CHECK(rec.aux_syncs == st.aux_slices_total);
        CHECK(rec.aux_syncs_while_in_flight == rec.aux_syncs);
    }
}
