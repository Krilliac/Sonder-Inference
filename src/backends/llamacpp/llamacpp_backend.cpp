// Sonder Inference - llama.cpp backend implementation.
#include "sonder/backends/llamacpp/llamacpp_backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <regex>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "batch_scheduler.hpp"
#include "radix_cache.hpp"
#include "suffix_matcher.hpp"

#include "ggml-backend.h"
#include "llama.h"

#if defined(SONDER_LLAMACPP_HAS_MTP)
#include "llama-ext.h"
#endif

#if defined(SONDER_LLAMACPP_HAS_MTMD)
#include "mtmd.h"
#include "mtmd-helper.h"
#endif

#ifndef SONDER_LLAMACPP_TAG_STRING
#define SONDER_LLAMACPP_TAG_STRING "unknown"
#endif

namespace sonder::backends::llamacpp {

static_assert(static_cast<int>(LogLevel::kNone) == GGML_LOG_LEVEL_NONE);
static_assert(static_cast<int>(LogLevel::kDebug) == GGML_LOG_LEVEL_DEBUG);
static_assert(static_cast<int>(LogLevel::kInfo) == GGML_LOG_LEVEL_INFO);
static_assert(static_cast<int>(LogLevel::kWarn) == GGML_LOG_LEVEL_WARN);
static_assert(static_cast<int>(LogLevel::kError) == GGML_LOG_LEVEL_ERROR);
static_assert(static_cast<int>(LogLevel::kCont) == GGML_LOG_LEVEL_CONT);

namespace {

std::atomic<LogSink> g_log_sink{nullptr};

}  // namespace

void SetLogSink(LogSink sink) noexcept { g_log_sink.store(sink, std::memory_order_release); }

// llama.cpp logs verbosely to stderr by default. Keep warnings and errors;
// set SONDER_LLAMACPP_VERBOSE=1 to see everything.
void RouteLog(LogLevel level, const char* text) {
    static const bool verbose = [] {
        const char* v = std::getenv("SONDER_LLAMACPP_VERBOSE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (verbose || level == LogLevel::kWarn || level == LogLevel::kError) {
        if (const LogSink sink = g_log_sink.load(std::memory_order_acquire)) {
            sink(text);
        } else {
            std::fputs(text, stderr);
        }
    }
}

namespace {

using Clock = std::chrono::steady_clock;

std::string ReadModelMetadataValue(const llama_model* model, const char* key) {
    std::size_t capacity = 256;
    for (int attempt = 0; attempt != 3; ++attempt) {
        std::string value(capacity, '\0');
        const int32_t n = llama_model_meta_val_str(model, key, value.data(), value.size());
        if (n < 0) return {};
        if (static_cast<std::size_t>(n) < value.size()) {
            value.resize(static_cast<std::size_t>(n));
            return value;
        }
        capacity = static_cast<std::size_t>(n) + 1;
    }
    return {};
}

sonder::inference::ModelArchitecture ReadModelArchitecture(const llama_model* model) {
    sonder::inference::ModelMetadata metadata;
    const std::string architecture = ReadModelMetadataValue(model, "general.architecture");
    if (!architecture.empty()) metadata.emplace_back("general.architecture", architecture);
    const int32_t count = llama_model_meta_count(model);
    if (count <= 0) return sonder::inference::classify_model_architecture(metadata);
    for (int32_t i = 0; i < count; ++i) {
        std::size_t capacity = 256;
        std::string key;
        for (int attempt = 0; attempt != 3; ++attempt) {
            key.assign(capacity, '\0');
            const int32_t n = llama_model_meta_key_by_index(model, i, key.data(), key.size());
            if (n < 0) {
                key.clear();
                break;
            }
            if (static_cast<std::size_t>(n) < key.size()) {
                key.resize(static_cast<std::size_t>(n));
                break;
            }
            capacity = static_cast<std::size_t>(n) + 1;
        }
        if (key.empty() || key == "general.architecture") continue;
        const std::string lower_key = sonder::inference::detail::lower_copy(key);
        const bool relevant = lower_key.find("deltanet") != std::string::npos ||
                              lower_key.find(".ssm_") != std::string::npos ||
                              lower_key.find(".ssm.") != std::string::npos ||
                              lower_key.find("_ssm") != std::string::npos ||
                              lower_key.find("recurrent") != std::string::npos ||
                              lower_key.find("attention") != std::string::npos ||
                              lower_key.find(".attn") != std::string::npos;
        if (relevant) {
            const std::string value = ReadModelMetadataValue(model, key.c_str());
            metadata.emplace_back(std::move(key), value);
        }
    }
    return sonder::inference::classify_model_architecture(metadata);
}

double MsSince(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// Forwards llama.cpp log lines to RouteLog (filter and sink).
void LogCallback(ggml_log_level level, const char* text, void* /*user*/) {
    RouteLog(static_cast<LogLevel>(level), text);
}

void EnsureBackendInitialized() {
    static std::once_flag once;
    std::call_once(once, [] {
        llama_log_set(&LogCallback, nullptr);
        llama_backend_init();
    });
}

DeviceKind MapDeviceKind(enum ggml_backend_dev_type type) {
    switch (type) {
        case GGML_BACKEND_DEVICE_TYPE_CPU: return DeviceKind::kCpu;
        case GGML_BACKEND_DEVICE_TYPE_GPU: return DeviceKind::kGpu;
        case GGML_BACKEND_DEVICE_TYPE_IGPU: return DeviceKind::kIntegratedGpu;
        case GGML_BACKEND_DEVICE_TYPE_ACCEL: return DeviceKind::kAccelerator;
        default: return DeviceKind::kOther;
    }
}

ggml_type ToGgmlType(KvCacheType type) {
    switch (type) {
        case KvCacheType::kF16: return GGML_TYPE_F16;
        case KvCacheType::kF32: return GGML_TYPE_F32;
        case KvCacheType::kBF16: return GGML_TYPE_BF16;
        case KvCacheType::kQ8_0: return GGML_TYPE_Q8_0;
        case KvCacheType::kQ5_1: return GGML_TYPE_Q5_1;
        case KvCacheType::kQ5_0: return GGML_TYPE_Q5_0;
        case KvCacheType::kQ4_1: return GGML_TYPE_Q4_1;
        case KvCacheType::kQ4_0: return GGML_TYPE_Q4_0;
        case KvCacheType::kIQ4_NL: return GGML_TYPE_IQ4_NL;
        case KvCacheType::kF8_E4M3:
#if defined(SONDER_LLAMACPP_HAS_GGML_F8_E4M3)
            return GGML_TYPE_F8_E4M3;
#else
            return GGML_TYPE_F16;  // unreachable after ValidateContextOptions
#endif
        case KvCacheType::kQ2_0:
#if defined(SONDER_LLAMACPP_HAS_GGML_Q2_0)
            return GGML_TYPE_Q2_0;
#else
            return GGML_TYPE_F16;  // unreachable after ValidateContextOptions
#endif
        case KvCacheType::kPQ2_0:
#if defined(SONDER_LLAMACPP_HAS_GGML_PQ2_0)
            return GGML_TYPE_PQ2_0;
#else
            return GGML_TYPE_F16;  // unreachable after ValidateContextOptions
#endif
    }
    return GGML_TYPE_F16;  // unreachable after ValidateContextOptions
}

llama_flash_attn_type ToLlamaFlashAttn(FlashAttention mode) {
    switch (mode) {
        case FlashAttention::kAuto: return LLAMA_FLASH_ATTN_TYPE_AUTO;
        case FlashAttention::kDisabled: return LLAMA_FLASH_ATTN_TYPE_DISABLED;
        case FlashAttention::kEnabled: return LLAMA_FLASH_ATTN_TYPE_ENABLED;
    }
    return LLAMA_FLASH_ATTN_TYPE_AUTO;
}

llama_split_mode ToLlamaSplitMode(SplitMode mode) {
    switch (mode) {
        case SplitMode::kNone: return LLAMA_SPLIT_MODE_NONE;
        case SplitMode::kLayer: return LLAMA_SPLIT_MODE_LAYER;
        case SplitMode::kRow: return LLAMA_SPLIT_MODE_ROW;
    }
    return LLAMA_SPLIT_MODE_LAYER;  // unreachable after ValidatePlacementOptions
}

std::string_view LlamaSplitModeName(llama_split_mode mode) noexcept {
    switch (mode) {
        case LLAMA_SPLIT_MODE_NONE: return "none";
        case LLAMA_SPLIT_MODE_LAYER: return "layer";
        case LLAMA_SPLIT_MODE_ROW: return "row";
        case LLAMA_SPLIT_MODE_TENSOR: return "tensor";
    }
    return "unknown";
}

inline void BatchClear(llama_batch& b) noexcept { b.n_tokens = 0; }

inline void BatchAdd(llama_batch& b, llama_token id, llama_pos pos, llama_seq_id seq_id, bool logits) noexcept {
    const int32_t i = b.n_tokens;
    b.token[i] = id;
    b.pos[i] = pos;
    b.n_seq_id[i] = 1;
    b.seq_id[i][0] = seq_id;
    b.logits[i] = logits ? 1 : 0;
    b.n_tokens = i + 1;
}

// The single place that turns LoadOptions into llama_model_params; Load() and
// ModelParamsFor() both use it. Buffers owned by the caller must outlive the
// llama_model_load_from_file() call (which copies devices and tensor_split).
llama_model_params MakeModelParams(const LoadOptions& options,
                                   const std::vector<void*>& resolved_devices,
                                   std::vector<ggml_backend_dev_t>& device_buf,
                                   std::vector<float>& tensor_split_buf) {
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = options.n_gpu_layers;
    mp.vocab_only = options.vocab_only;
    mp.main_gpu = options.main_gpu;
    mp.split_mode = ToLlamaSplitMode(options.split_mode);
    // llama.cpp b11195 merges mmap and mlock into llama_model_params::load_mode.
    // Default (use_mmap=true, use_mlock=false) preserves LLAMA_LOAD_MODE_AUTO.
    if (options.use_mlock) {
        mp.load_mode = options.use_mmap ? LLAMA_LOAD_MODE_MMAP_MLOCK : LLAMA_LOAD_MODE_MLOCK;
    } else if (!options.use_mmap) {
        mp.load_mode = LLAMA_LOAD_MODE_NONE;
    }
#if defined(SONDER_LLAMACPP_HAS_MTP)
    SpeculativeType spec = SpeculativeType::kNone;
    if (ParseSpeculativeType(options.spec_type, spec) &&
        (spec == SpeculativeType::kDraftMtp || spec == SpeculativeType::kAuto)) {
        mp.load_mtp = true;
    }
#endif
    device_buf.clear();
    if (!resolved_devices.empty()) {
        device_buf.reserve(resolved_devices.size() + 1);
        for (void* dev : resolved_devices) {
            device_buf.push_back(static_cast<ggml_backend_dev_t>(dev));
        }
        device_buf.push_back(nullptr);
        mp.devices = device_buf.data();
    }
    tensor_split_buf.clear();
    if (!options.tensor_split.empty()) {
        // llama_model copies exactly llama_max_devices() floats from tensor_split.
        const std::size_t slot_count = std::max<std::size_t>(llama_max_devices(), options.tensor_split.size());
        tensor_split_buf.assign(slot_count, 0.0F);
        std::copy(options.tensor_split.begin(), options.tensor_split.end(), tensor_split_buf.begin());
        mp.tensor_split = tensor_split_buf.data();
    }
    return mp;
}

ModelParamsSummary SummarizeModelParams(const llama_model_params& mp) {
    ModelParamsSummary s;
    s.n_gpu_layers = mp.n_gpu_layers;
    s.main_gpu = mp.main_gpu;
    s.split_mode = std::string(LlamaSplitModeName(mp.split_mode));
    const char* lm = llama_load_mode_name(mp.load_mode);
    s.load_mode = lm != nullptr ? std::string(lm) : std::string("unknown");
    s.vocab_only = mp.vocab_only;
    s.has_tensor_split = mp.tensor_split != nullptr;
    s.has_devices = mp.devices != nullptr;
#if defined(SONDER_LLAMACPP_HAS_MTP)
    s.load_mtp = mp.load_mtp;
#else
    s.load_mtp = false;
#endif
    return s;
}

// The single place that turns LoadOptions into llama_context_params; Load()
// and ContextParamsFor() both use it. The abort callback is set by Load().
llama_context_params MakeContextParams(const LoadOptions& options) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = options.n_ctx;
    cp.n_batch = options.n_batch;
    // 0 keeps llama.cpp's default micro-batch, capped at n_batch (the
    // behaviour before n_ubatch was configurable).
    cp.n_ubatch = options.n_ubatch == 0 ? std::min(cp.n_ubatch, options.n_batch) : options.n_ubatch;
    cp.n_seq_max = std::clamp<uint32_t>(options.n_slots, 1, kMaxSlots);
    if (options.n_threads > 0) {
        cp.n_threads = options.n_threads;
        cp.n_threads_batch = options.n_threads;
    } else {
        const unsigned hw = std::thread::hardware_concurrency();
        if (hw > 0) {
            const auto batch_threads = static_cast<int32_t>(std::clamp<unsigned>(hw / 2u, 4u, 16u));
            if (batch_threads > cp.n_threads_batch) {
                cp.n_threads_batch = batch_threads;
            }
        }
    }
    cp.type_k = ToGgmlType(options.type_k);
    cp.type_v = ToGgmlType(options.type_v);
    cp.flash_attn_type = ToLlamaFlashAttn(options.flash_attn);
    cp.offload_kqv = true;
    cp.no_perf = true;
#if defined(SONDER_LLAMACPP_HAS_KV_UNIFIED)
    cp.kv_unified = options.kv_unified;
#endif
#if defined(SONDER_LLAMACPP_HAS_OP_OFFLOAD)
    cp.op_offload = true;
#endif
#if defined(SONDER_LLAMACPP_HAS_MOE_CACHE_SIZE)
    if (options.moe_cache_mib > 0) {
        const std::uint64_t bytes64 = static_cast<std::uint64_t>(options.moe_cache_mib) << 20;
        cp.moe_cache_size = static_cast<size_t>(std::min<std::uint64_t>(bytes64, std::numeric_limits<size_t>::max()));
    }
#endif
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ) || defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
    uint32_t planned_draft_max = 0;
    SpeculativeType planned_spec = SpeculativeType::kNone;
    {
        SpeculativeType spec = SpeculativeType::kNone;
        if (ParseSpeculativeType(options.spec_type, spec) && spec != SpeculativeType::kNone) {
            planned_spec = spec;
            if (options.spec_draft_max > 0) {
                planned_draft_max = static_cast<uint32_t>(options.spec_draft_max);
            } else if (spec == SpeculativeType::kDraftMtp) {
                planned_draft_max = static_cast<uint32_t>(kDefaultMtpDraftMax);
            } else if (spec == SpeculativeType::kSuffix) {
                planned_draft_max = static_cast<uint32_t>(kDefaultSuffixDraftMax);
            } else if (spec == SpeculativeType::kNgramMod || spec == SpeculativeType::kAuto) {
                planned_draft_max = static_cast<uint32_t>(kDefaultNgramModDraftMax);
            } else {
                planned_draft_max = static_cast<uint32_t>(kDefaultNgramDraftMax);
            }
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ)
            // Only allocate per-step GPU recurrent rollback slots (n_rs_seq) for
            // short-horizon draft-model speculation (draft-mtp / auto). Long-horizon
            // n-gram / suffix speculation uses zero GPU rollback slots (saving
            // ~1.6 GiB VRAM on 27B hybrid models) and relies on on-demand
            // PARTIAL_ONLY checkpoints.
            uint32_t rs_need = 0;
            if (spec == SpeculativeType::kDraftMtp) {
                rs_need = planned_draft_max;
            } else if (spec == SpeculativeType::kAuto) {
                rs_need = options.spec_draft_max > 0
                              ? std::min<uint32_t>(static_cast<uint32_t>(options.spec_draft_max), 8u)
                              : static_cast<uint32_t>(kDefaultMtpDraftMax);
            }
            const uint32_t n_batch_eff =
                options.n_ctx > 0 ? std::min(options.n_batch, options.n_ctx) : options.n_batch;
            const uint32_t n_ubatch_eff = cp.n_ubatch == 0 ? n_batch_eff : std::min(n_batch_eff, cp.n_ubatch);
            if (rs_need > 0 && n_ubatch_eff > rs_need + 1u) {
                cp.n_rs_seq = rs_need;
            }
#endif
        }
    }
    (void)planned_spec;
#endif
#if defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
    {
        // Hybrid draft-mtp + suffix escalation can burst up to 16 draft tokens
        // in one verification pass even when MTP's own k=3.
        uint32_t max_burst = planned_draft_max;
        if (planned_spec == SpeculativeType::kDraftMtp || planned_spec == SpeculativeType::kAuto) {
            max_burst = std::max<uint32_t>(max_burst, static_cast<uint32_t>(SuffixMatcher::kHybridMtpMaxBurstDraft));
        }
        const uint32_t per_seq = std::min<uint32_t>(cp.n_batch, std::max<uint32_t>(1u, max_burst + 1u));
        cp.n_outputs_max_per_seq = per_seq;
        cp.n_outputs_max = std::min<uint32_t>(cp.n_batch, cp.n_seq_max * per_seq);
    }
#endif
    return cp;
}

ContextParamsSummary Summarize(const llama_context_params& cp) {
    ContextParamsSummary s;
    s.n_ctx = cp.n_ctx;
    s.n_batch = cp.n_batch;
    s.n_ubatch = cp.n_ubatch;
    s.n_seq_max = cp.n_seq_max;
    s.type_k = ggml_type_name(cp.type_k);
    s.type_v = ggml_type_name(cp.type_v);
    s.flash_attn = llama_flash_attn_type_name(cp.flash_attn_type);
    s.offload_kqv = cp.offload_kqv;
    s.no_perf = cp.no_perf;
#if defined(SONDER_LLAMACPP_HAS_KV_UNIFIED)
    s.kv_unified = cp.kv_unified;
#endif
#if defined(SONDER_LLAMACPP_HAS_OP_OFFLOAD)
    s.op_offload = cp.op_offload;
#endif
#if defined(SONDER_LLAMACPP_HAS_MOE_CACHE_SIZE)
    s.moe_cache_size = static_cast<std::uint64_t>(cp.moe_cache_size);
#endif
    return s;
}

struct SamplerDeleter {
    void operator()(llama_sampler* s) const noexcept { llama_sampler_free(s); }
};
using SamplerPtr = std::unique_ptr<llama_sampler, SamplerDeleter>;

// Order follows llama.cpp's common sampler: logit bias, penalties, then selector
// (greedy or top-k/typical/top-p/min-p/temp/dist). An optional GBNF grammar is
// returned separately so Generate() can apply fast grammar-based rejection
// sampling (checking only the 1 sampled token first and only masking the full
// vocabulary on the rare rejection case).
Status BuildSampler(const SamplingParams& p, const llama_vocab* vocab, int32_t n_vocab, int32_t n_ctx,
                    SamplerPtr& out, SamplerPtr& grammar_out) {
    out.reset();
    grammar_out.reset();
    SamplerPtr chain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
    if (!chain) {
        return Status::Error(ErrorCode::kDecodeFailed, "failed to initialize llama sampler chain");
    }
    if (!p.logit_bias.empty()) {
        std::vector<llama_logit_bias> bias;
        bias.reserve(p.logit_bias.size());
        for (const auto& [token, value] : p.logit_bias) bias.push_back(llama_logit_bias{token, value});
        llama_sampler_chain_add(chain.get(), llama_sampler_init_logit_bias(n_vocab, static_cast<int32_t>(bias.size()),
                                                                           bias.data()));
    }
    const int32_t last_n = p.repeat_last_n < 0 ? n_ctx : p.repeat_last_n;
    const bool penalties = p.repeat_penalty != 1.0F || p.presence_penalty != 0.0F || p.frequency_penalty != 0.0F;
    if (penalties && last_n > 0) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_penalties(n_vocab, last_n, p.repeat_penalty,
                                                                          p.frequency_penalty, p.presence_penalty));
    }
    if (!p.grammar.empty()) {
        llama_sampler* gram = llama_sampler_init_grammar(vocab, p.grammar.c_str(), "root");
        if (gram == nullptr) {
            return Status::Error(ErrorCode::kInvalidArgument, "invalid GBNF grammar");
        }
        grammar_out.reset(gram);
    }
    if (p.temperature <= 0.0F) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        out = std::move(chain);
        return Status::Ok();
    }
    if (p.top_k > 0) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(p.top_k));
    if (p.typical_p < 1.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_typical(p.typical_p, 1));
    if (p.top_p < 1.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(p.top_p, 1));
    if (p.min_p > 0.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(p.min_p, 1));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(p.temperature));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(p.seed));
    out = std::move(chain);
    return Status::Ok();
}

}  // namespace

struct LlamaCppBackend::Impl {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;
    LoadOptions options{};
    TelemetrySink telemetry;
    // Consulted by llama.cpp's abort callback during long llama_decode calls.
    const CancelPredicate* active_cancel = nullptr;
    std::function<bool()> batch_abort_fn;
    std::atomic<bool> abort_requested{false};
    sonder::inference::ModelArchitecture architecture = sonder::inference::ModelArchitecture::attention_only;

    // Reusable decode batch allocated once at Load() time (zero per-step allocs).
    llama_batch batch{};
    int32_t batch_cap = 0;
    bool batch_allocated = false;
    std::vector<llama_token_data> grammar_candidates;

    // Multi-slot KV prefix cache metadata (1..kMaxSlots).
    struct PartialCheckpoint {
        std::size_t n_tokens = 0;
        std::uint64_t last_used = 0;
        std::vector<std::uint8_t> tgt_state;
        std::vector<float> pending_h;

        std::size_t ByteSize() const noexcept {
            return tgt_state.size() + pending_h.size() * sizeof(float);
        }
    };
    static constexpr std::size_t kMaxCheckpointsPerSlot = 3;
    static constexpr std::size_t kMaxPartialCheckpointBytes = std::size_t{256} << 20;
    static constexpr std::size_t kMaxTotalCheckpointBytes = std::size_t{512} << 20;
    static constexpr std::size_t kMinL2SnapshotTokens = 256;
    static constexpr std::size_t kMinL2RestoreGainTokens = 128;

    std::vector<PrefixSlotMeta> slots;
    std::vector<std::uint8_t> slot_busy;
    std::vector<bool> slot_mtp_synced;
    std::vector<std::vector<PartialCheckpoint>> slot_checkpoints;
    std::size_t total_checkpoint_bytes = 0;
    std::uint64_t checkpoint_clock = 0;
    std::vector<std::uint8_t> spec_ckpt_buf;
    std::size_t spec_ckpt_size = 0;
    std::uint64_t slot_clock = 0;

    // Bounded L2 host-RAM Radix prefix cache (`cache_ram_mib`).
    RadixPrefixCache radix_cache;

    // Isolated lazy embedding context so Embed() never disturbs live KV prefix
    // caches or MTP/backend-sampler graph reservations on `ctx`.
    llama_context* embed_ctx = nullptr;
    llama_batch embed_batch{};
    int32_t embed_batch_cap = 0;
    bool embed_batch_allocated = false;

    // Optional libmtmd multimodal projector context (`mmproj_path`).
#if defined(SONDER_LLAMACPP_HAS_MTMD)
    mtmd_context* mtmd_ctx = nullptr;
#endif

    // Resolved in-process speculative decoding state.
    SpeculativeType active_spec = SpeculativeType::kNone;
    std::int32_t active_draft_max = 0;
    NgramMatcher ngram_matcher{false};
    std::unique_ptr<NgramModTable> ngram_mod;
    std::unique_ptr<SuffixMatcher> suffix_matcher;
    std::vector<Token> draft_scratch;
    std::vector<Token> accepted_scratch;
    std::vector<TokenLogprobInfo> accepted_logprobs_scratch;
    std::vector<Token> ngram_history;

    // Single-owner continuous batching worker thread state (active when n_slots > 1).
    struct PendingRequest {
        const GenerateRequest* req = nullptr;
        const TokenCallback* on_token = nullptr;
        const CancelPredicate* should_cancel = nullptr;
        bool exclusive = false;

        std::mutex done_mu;
        std::condition_variable done_cv;
        bool done = false;
        GenerateResult result{};
    };

    std::atomic<LlamaCppBackend*> owner{nullptr};
    std::mutex queue_mu;
    std::condition_variable worker_cv;
    std::deque<PendingRequest*> pending_queue;
    std::atomic<std::size_t> pending_count{0};
    std::atomic<bool> worker_stop{false};
    std::atomic<bool> batch_worker_running{false};
    std::thread batch_worker;

#if defined(SONDER_LLAMACPP_HAS_MTP)
    // Native PrismML / mainline NextN (MTP) draft context state.
    llama_context* ctx_dft = nullptr;
    llama_batch dft_batch{};
    int32_t dft_batch_cap = 0;
    bool dft_batch_allocated = false;
    int32_t mtp_n_embd = 0;
    int32_t mtp_n_layers = 1;
    bool mtp_mem_shared = false;
    bool mtp_chain_heads = false;
    std::vector<SamplerPtr> mtp_samplers;
    std::vector<llama_sampler*> mtp_backend_chains;
    std::vector<std::vector<float>> mtp_pending_h;
    std::vector<float> mtp_chain_h;
    struct MtpDeferred {
        std::vector<Token> tokens;
        std::vector<llama_pos> pos;
        std::vector<float> h;
        bool pending = false;

        void Clear() noexcept {
            tokens.clear();
            pos.clear();
            h.clear();
            pending = false;
        }
    };
    std::vector<MtpDeferred> mtp_deferred;
#endif

    ~Impl() { Release(); }

    static uint32_t EffectiveCtxSeq(const llama_context* c) noexcept {
        if (c == nullptr) return 0;
#if defined(SONDER_LLAMACPP_HAS_N_CTX_SEQ)
        const uint32_t per_seq = llama_n_ctx_seq(c);
        if (per_seq > 0) return per_seq;
#endif
        return llama_n_ctx(c);
    }

    void ClearSlotCheckpoints(std::size_t slot_idx) noexcept {
        if (slot_idx >= slot_checkpoints.size()) return;
        for (const auto& c : slot_checkpoints[slot_idx]) {
            const std::size_t b = c.ByteSize();
            total_checkpoint_bytes = (total_checkpoint_bytes >= b) ? (total_checkpoint_bytes - b) : 0;
        }
        slot_checkpoints[slot_idx].clear();
    }

    void PruneSlotCheckpointsAfter(std::size_t slot_idx, std::size_t max_tokens) noexcept {
        if (slot_idx >= slot_checkpoints.size()) return;
        auto& vec = slot_checkpoints[slot_idx];
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [&](const PartialCheckpoint& c) {
                                     if (c.n_tokens > max_tokens) {
                                         const std::size_t b = c.ByteSize();
                                         total_checkpoint_bytes =
                                             (total_checkpoint_bytes >= b) ? (total_checkpoint_bytes - b) : 0;
                                         return true;
                                     }
                                     return false;
                                 }),
                  vec.end());
    }

    bool EvictOldestCheckpoint() noexcept {
        std::size_t best_slot = slot_checkpoints.size();
        std::size_t best_idx = 0;
        std::uint64_t oldest = UINT64_MAX;
        for (std::size_t s = 0; s < slot_checkpoints.size(); ++s) {
            const auto& vec = slot_checkpoints[s];
            for (std::size_t i = 0; i < vec.size(); ++i) {
                if (best_slot == slot_checkpoints.size() || vec[i].last_used < oldest) {
                    best_slot = s;
                    best_idx = i;
                    oldest = vec[i].last_used;
                }
            }
        }
        if (best_slot >= slot_checkpoints.size()) return false;
        auto& vec = slot_checkpoints[best_slot];
        const std::size_t b = vec[best_idx].ByteSize();
        total_checkpoint_bytes = (total_checkpoint_bytes >= b) ? (total_checkpoint_bytes - b) : 0;
        vec.erase(vec.begin() + static_cast<std::ptrdiff_t>(best_idx));
        return true;
    }

    void ResetAllSlots() noexcept {
        for (std::size_t i = 0; i < slots.size(); ++i) {
            slots[i].tokens.clear();
            slots[i].session_key.clear();
            slots[i].last_used = 0;
            if (i < slot_mtp_synced.size()) slot_mtp_synced[i] = false;
        }
        for (auto& ckpts : slot_checkpoints) {
            ckpts.clear();
        }
        total_checkpoint_bytes = 0;
        checkpoint_clock = 0;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr) {
            if (!mtp_mem_shared) {
                llama_memory_clear(llama_get_memory(ctx_dft), true);
            }
            for (auto& h : mtp_pending_h) std::fill(h.begin(), h.end(), 0.0F);
            for (auto& d : mtp_deferred) d.Clear();
        }
#endif
    }

    void StopBatchWorker() noexcept {
        if (!batch_worker_running.load(std::memory_order_acquire)) return;
        {
            std::lock_guard<std::mutex> lock(queue_mu);
            worker_stop.store(true, std::memory_order_release);
        }
        worker_cv.notify_all();
        if (batch_worker.joinable()) {
            batch_worker.join();
        }
        batch_worker_running.store(false, std::memory_order_release);
        worker_stop.store(false, std::memory_order_relaxed);
        pending_count.store(0, std::memory_order_relaxed);
    }

    void StartBatchWorker();
    void BatchWorkerLoop();
    GenerateResult GenerateSingleLocked(const GenerateRequest& request, const TokenCallback& on_token,
                                        const CancelPredicate& should_cancel);

    void ReleaseMtp() noexcept {
#if defined(SONDER_LLAMACPP_HAS_MTP)
        for (std::size_t i = 0; i < mtp_backend_chains.size(); ++i) {
            if (mtp_backend_chains[i] == nullptr) continue;
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
            if (ctx_dft != nullptr) {
                llama_set_sampler(ctx_dft, static_cast<llama_seq_id>(i), nullptr);
            }
#endif
            llama_sampler_free(mtp_backend_chains[i]);
            mtp_backend_chains[i] = nullptr;
        }
        mtp_backend_chains.clear();
        mtp_samplers.clear();
        if (dft_batch_allocated) {
            if (dft_batch.token != nullptr) {
                std::free(dft_batch.token);
                dft_batch.token = nullptr;
            }
            llama_batch_free(dft_batch);
            dft_batch = {};
            dft_batch_cap = 0;
            dft_batch_allocated = false;
        }
        if (ctx_dft != nullptr) {
            llama_free(ctx_dft);
            ctx_dft = nullptr;
        }
        mtp_n_embd = 0;
        mtp_n_layers = 1;
        mtp_mem_shared = false;
        mtp_chain_heads = false;
        mtp_pending_h.clear();
        mtp_chain_h.clear();
        mtp_deferred.clear();
#endif
    }

    void Release() noexcept {
        StopBatchWorker();
        ReleaseMtp();
#if defined(SONDER_LLAMACPP_HAS_MTMD)
        if (mtmd_ctx != nullptr) {
            mtmd_free(mtmd_ctx);
            mtmd_ctx = nullptr;
        }
#endif
        if (embed_batch_allocated) {
            llama_batch_free(embed_batch);
            embed_batch = {};
            embed_batch_cap = 0;
            embed_batch_allocated = false;
        }
        if (embed_ctx != nullptr) {
            llama_free(embed_ctx);
            embed_ctx = nullptr;
        }
        if (batch_allocated) {
            llama_batch_free(batch);
            batch = {};
            batch_cap = 0;
            batch_allocated = false;
        }
        if (ctx != nullptr) {
            llama_free(ctx);
            ctx = nullptr;
        }
        if (model != nullptr) {
            llama_model_free(model);
            model = nullptr;
        }
        vocab = nullptr;
        slots.clear();
        slot_busy.clear();
        slot_mtp_synced.clear();
        slot_checkpoints.clear();
        radix_cache.Clear();
        total_checkpoint_bytes = 0;
        checkpoint_clock = 0;
        spec_ckpt_buf.clear();
        spec_ckpt_size = 0;
        slot_clock = 0;
        active_spec = SpeculativeType::kNone;
        active_draft_max = 0;
        ngram_mod.reset();
        suffix_matcher.reset();
    }

#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
    static llama_state_seq_flags PartialCheckpointFlags() noexcept {
#if defined(LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY)
        return LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY;
#elif defined(LLAMA_STATE_SEQ_FLAGS_SWA_ONLY)
        return LLAMA_STATE_SEQ_FLAGS_SWA_ONLY;
#else
        return 0;
#endif
    }

    void CapturePartialCheckpoint(std::size_t slot_idx, llama_seq_id slot_id, std::size_t n_tokens) {
        if (ctx == nullptr || architecture != sonder::inference::ModelArchitecture::hybrid || n_tokens == 0) return;
        if (slot_idx >= slot_checkpoints.size()) return;
        const llama_state_seq_flags flags = PartialCheckpointFlags();
        if (flags == 0) return;
        const std::size_t sz = llama_state_seq_get_size_ext(ctx, slot_id, flags);
        if (sz == 0 || sz > kMaxPartialCheckpointBytes) return;
        std::size_t pending_bytes = 0;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr && slot_idx < mtp_pending_h.size()) {
            pending_bytes = mtp_pending_h[slot_idx].size() * sizeof(float);
        }
#endif
        if (sz > kMaxTotalCheckpointBytes || pending_bytes > kMaxTotalCheckpointBytes - sz) return;
        const std::size_t projected_bytes = sz + pending_bytes;

        // Prune superseded checkpoints at >= n_tokens and enforce per-slot and
        // model-wide byte budgets BEFORE allocating the new state buffer.
        auto& vec = slot_checkpoints[slot_idx];
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [&](const PartialCheckpoint& c) {
                                     if (c.n_tokens >= n_tokens) {
                                         const std::size_t b = c.ByteSize();
                                         total_checkpoint_bytes =
                                             (total_checkpoint_bytes >= b) ? (total_checkpoint_bytes - b) : 0;
                                         return true;
                                     }
                                     return false;
                                 }),
                  vec.end());
        while (vec.size() >= kMaxCheckpointsPerSlot) {
            const std::size_t b = vec.front().ByteSize();
            total_checkpoint_bytes = (total_checkpoint_bytes >= b) ? (total_checkpoint_bytes - b) : 0;
            vec.erase(vec.begin());
        }
        while (total_checkpoint_bytes > kMaxTotalCheckpointBytes - projected_bytes) {
            if (!EvictOldestCheckpoint()) return;
        }

        PartialCheckpoint ckpt;
        ckpt.n_tokens = n_tokens;
        ckpt.last_used = ++checkpoint_clock;
        ckpt.tgt_state.resize(sz);
        const std::size_t written = llama_state_seq_get_data_ext(ctx, ckpt.tgt_state.data(), sz, slot_id, flags);
        if (written == 0 || written > sz) return;
        ckpt.tgt_state.resize(written);
        ckpt.tgt_state.shrink_to_fit();
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr && slot_idx < mtp_pending_h.size()) {
            ckpt.pending_h = mtp_pending_h[slot_idx];
        }
#endif
        const std::size_t actual_bytes = ckpt.ByteSize();
        if (actual_bytes > kMaxTotalCheckpointBytes) return;
        while (total_checkpoint_bytes > kMaxTotalCheckpointBytes - actual_bytes) {
            if (!EvictOldestCheckpoint()) return;
        }
        total_checkpoint_bytes += actual_bytes;
        slot_checkpoints[slot_idx].push_back(std::move(ckpt));
    }

    bool RestorePartialCheckpoint(std::size_t slot_idx, llama_seq_id slot_id, std::size_t max_lcp,
                                  std::size_t& restored_tokens, bool& restored_pending_h) {
        restored_tokens = 0;
        restored_pending_h = false;
        if (ctx == nullptr || slot_idx >= slot_checkpoints.size()) return false;
        const llama_state_seq_flags flags = PartialCheckpointFlags();
        if (flags == 0) return false;
        auto& vec = slot_checkpoints[slot_idx];
        PartialCheckpoint* best = nullptr;
        for (auto& c : vec) {
            if (c.n_tokens > 0 && c.n_tokens <= max_lcp) {
                if (best == nullptr || c.n_tokens > best->n_tokens) {
                    best = &c;
                }
            }
        }
        if (best == nullptr) return false;
        const std::size_t read =
            llama_state_seq_set_data_ext(ctx, best->tgt_state.data(), best->tgt_state.size(), slot_id, flags);
        if (read == 0) return false;
        llama_memory_t mem = llama_get_memory(ctx);
        if (!llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(best->n_tokens), -1)) {
            return false;
        }
        best->last_used = ++checkpoint_clock;
        restored_tokens = best->n_tokens;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr && slot_idx < mtp_pending_h.size() &&
            best->pending_h.size() == mtp_pending_h[slot_idx].size()) {
            mtp_pending_h[slot_idx] = best->pending_h;
            restored_pending_h = true;
        }
        if (slot_idx < mtp_deferred.size()) {
            mtp_deferred[slot_idx].Clear();
        }
#endif
        PruneSlotCheckpointsAfter(slot_idx, restored_tokens);
        return true;
    }
#endif

    void Emit(TelemetryKind kind, std::int32_t tokens, double ms, std::string detail = {}) const {
        if (telemetry) telemetry(TelemetryEvent{kind, tokens, ms, std::move(detail)});
    }

    static bool AbortCallback(void* data) {
        auto* self = static_cast<Impl*>(data);
        if (self->abort_requested.load(std::memory_order_relaxed)) return true;
        if (self->active_cancel != nullptr && *self->active_cancel && (*self->active_cancel)()) {
            self->abort_requested.store(true, std::memory_order_relaxed);
            return true;
        }
        if (self->batch_abort_fn && self->batch_abort_fn()) {
            self->abort_requested.store(true, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    // Snapshots an evicted VRAM slot's sequence state into the bounded L2
    // host-RAM Radix prefix cache (strictly enforcing LRU eviction BEFORE
    // allocating the state buffer).
    void SnapshotSlotToL2(std::size_t slot_idx, llama_seq_id slot_id) {
        if (!radix_cache.enabled() || ctx == nullptr || slot_idx >= slots.size()) return;
        const auto& stoks = slots[slot_idx].tokens;
        if (stoks.size() < kMinL2SnapshotTokens) return;

        llama_memory_t mem = llama_get_memory(ctx);
        if (mem == nullptr) return;
        const int64_t kv_len = llama_memory_seq_pos_max(mem, slot_id) + 1;
        if (kv_len != static_cast<int64_t>(stoks.size())) return;

        const std::size_t sz = llama_state_seq_get_size(ctx, slot_id);
        if (sz == 0) return;

        std::size_t dft_sz = 0;
        std::size_t pending_bytes = 0;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        bool capture_mtp = false;
        if (ctx_dft != nullptr && slot_idx < slot_mtp_synced.size() && slot_mtp_synced[slot_idx]) {
            if (!mtp_mem_shared && slot_idx < mtp_deferred.size() && mtp_deferred[slot_idx].pending) {
                if (!MtpFlushDeferred(slot_id)) {
                    slot_mtp_synced[slot_idx] = false;
                }
            }
            if (slot_mtp_synced[slot_idx]) {
                if (!mtp_mem_shared) {
                    const int64_t dft_len = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id) + 1;
                    if (dft_len == kv_len) {
                        dft_sz = llama_state_seq_get_size(ctx_dft, slot_id);
                        capture_mtp = (dft_sz > 0);
                    }
                } else {
                    capture_mtp = true;
                }
                if (capture_mtp && slot_idx < mtp_pending_h.size()) {
                    pending_bytes = mtp_pending_h[slot_idx].size() * sizeof(float);
                }
            }
        }
#endif
        if (sz > radix_cache.max_bytes() || dft_sz > radix_cache.max_bytes() - sz ||
            pending_bytes > radix_cache.max_bytes() - sz - dft_sz) {
            return;
        }
        const std::size_t total_need = sz + dft_sz + pending_bytes;
        if (!radix_cache.ReserveBeforeAlloc(total_need, stoks)) {
            return;
        }

        RadixCacheEntry entry;
        entry.state_blob.resize(sz);
        const std::size_t written = llama_state_seq_get_data(ctx, entry.state_blob.data(), sz, slot_id);
        if (written == 0 || written > sz) return;
        entry.state_blob.resize(written);
        entry.state_blob.shrink_to_fit();

#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (capture_mtp) {
            if (dft_sz > 0 && ctx_dft != nullptr) {
                entry.dft_state_blob.resize(dft_sz);
                const std::size_t dw =
                    llama_state_seq_get_data(ctx_dft, entry.dft_state_blob.data(), dft_sz, slot_id);
                if (dw > 0 && dw <= dft_sz) {
                    entry.dft_state_blob.resize(dw);
                    entry.dft_state_blob.shrink_to_fit();
                } else {
                    entry.dft_state_blob.clear();
                    capture_mtp = false;
                }
            }
            if (capture_mtp && slot_idx < mtp_pending_h.size()) {
                entry.mtp_pending_h = mtp_pending_h[slot_idx];
            }
        }
#endif
        entry.session_key = slots[slot_idx].session_key;
        (void)radix_cache.Insert(stoks, std::move(entry));
    }

    // Attempts to restore a longer matching prefix from the L2 Radix host-RAM
    // cache directly into `slot_id`. Callers must snapshot any evicted VRAM
    // slot BEFORE calling this so the returned Radix entry pointer stays valid.
    // Sets `out_wiped = true` if VRAM KV for `slot_id` was cleared prior to a
    // failed restore so callers reset `lcp = 0` and slot metadata cleanly.
    bool TryRestoreSlotFromL2(std::size_t slot_idx, llama_seq_id slot_id, std::span<const Token> prompt,
                              std::size_t min_beat_lcp, std::size_t& out_lcp, bool& out_mtp_synced,
                              bool& out_wiped) {
        out_lcp = 0;
        out_mtp_synced = false;
        out_wiped = false;
        if (!radix_cache.enabled() || ctx == nullptr || prompt.size() <= 1) return false;
        const std::size_t max_reuse = prompt.size() - 1;
        const bool allow_trim = (architecture == sonder::inference::ModelArchitecture::attention_only);
        const PinnedRadixLookup hit = radix_cache.LookupAndPin(prompt, max_reuse, allow_trim);
        const RadixCacheEntry* entry = hit.entry();
        if (!hit.found() || entry == nullptr || hit.matched_tokens() <= min_beat_lcp ||
            (hit.matched_tokens() - min_beat_lcp) < kMinL2RestoreGainTokens) {
            return false;
        }

        llama_memory_t mem = llama_get_memory(ctx);
        if (mem == nullptr) return false;
        out_wiped = true;
        llama_memory_seq_rm(mem, slot_id, -1, -1);
        ClearSlotCheckpoints(slot_idx);

        const std::size_t read = llama_state_seq_set_data(ctx, entry->state_blob.data(),
                                                          entry->state_blob.size(), slot_id);
        if (read == 0) {
            llama_memory_seq_rm(mem, slot_id, -1, -1);
            return false;
        }
        if (hit.state_tokens() > hit.matched_tokens()) {
            if (!llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(hit.matched_tokens()), -1)) {
                llama_memory_seq_rm(mem, slot_id, -1, -1);
                return false;
            }
        }
        const int64_t restored_len = llama_memory_seq_pos_max(mem, slot_id) + 1;
        if (restored_len != static_cast<int64_t>(hit.matched_tokens())) {
            llama_memory_seq_rm(mem, slot_id, -1, -1);
            return false;
        }

#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr && slot_idx < mtp_pending_h.size()) {
            mtp_deferred[slot_idx].Clear();
            std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
            if (mtp_mem_shared) {
                if (hit.state_tokens() == hit.matched_tokens() &&
                    entry->mtp_pending_h.size() == mtp_pending_h[slot_idx].size()) {
                    mtp_pending_h[slot_idx] = entry->mtp_pending_h;
                }
                out_mtp_synced = true;
            } else if (!entry->dft_state_blob.empty()) {
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                llama_memory_seq_rm(mem_dft, slot_id, -1, -1);
                const std::size_t dr = llama_state_seq_set_data(ctx_dft, entry->dft_state_blob.data(),
                                                                entry->dft_state_blob.size(), slot_id);
                bool dft_ok = (dr > 0);
                if (dft_ok && hit.state_tokens() > hit.matched_tokens()) {
                    dft_ok = llama_memory_seq_rm(mem_dft, slot_id, static_cast<llama_pos>(hit.matched_tokens()), -1);
                }
                if (dft_ok && llama_memory_seq_pos_max(mem_dft, slot_id) + 1 == restored_len) {
                    if (hit.state_tokens() == hit.matched_tokens() &&
                        entry->mtp_pending_h.size() == mtp_pending_h[slot_idx].size()) {
                        mtp_pending_h[slot_idx] = entry->mtp_pending_h;
                    }
                    out_mtp_synced = true;
                } else {
                    llama_memory_seq_rm(mem_dft, slot_id, -1, -1);
                    out_mtp_synced = false;
                }
            } else {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                out_mtp_synced = false;
            }
            if (slot_idx < slot_mtp_synced.size()) {
                slot_mtp_synced[slot_idx] = out_mtp_synced;
            }
        }
#endif
        out_lcp = hit.matched_tokens();
        return true;
    }

    // Samples one token at `batch_idx` using GPU backend sampling when active,
    // fast grammar-based rejection sampling when a GBNF grammar is active, and
    // optionally computes exact log-softmax logprob + top-K from raw logits.
    Token SampleTokenAt(int32_t batch_idx, const SamplingParams& sp, llama_sampler* sampler,
                        llama_sampler* grammar_sampler, bool gpu_sampled, int32_t n_vocab,
                        TokenLogprobInfo* out_logprob) {
        if (out_logprob != nullptr) *out_logprob = {};
        Token tok = LLAMA_TOKEN_NULL;
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
        if (gpu_sampled && !sp.compute_logprobs && grammar_sampler == nullptr) {
            tok = llama_get_sampled_token_ith(ctx, batch_idx);
            if (tok != LLAMA_TOKEN_NULL) {
                llama_sampler_accept(sampler, tok);
                return tok;
            }
        }
#else
        (void)gpu_sampled;
#endif
        if (grammar_sampler == nullptr) {
            tok = llama_sampler_sample(sampler, ctx, batch_idx);
            if (tok != LLAMA_TOKEN_NULL && sp.compute_logprobs && out_logprob != nullptr) {
                if (const float* logits = llama_get_logits_ith(ctx, batch_idx); logits != nullptr) {
                    *out_logprob = ComputeTokenLogprobs(
                        std::span<const float>(logits, static_cast<std::size_t>(n_vocab)), tok, sp.top_logprobs);
                }
            }
            return tok;
        }

        const float* logits = llama_get_logits_ith(ctx, batch_idx);
        if (logits == nullptr) {
            return LLAMA_TOKEN_NULL;
        }
        const bool pure_greedy =
            sp.temperature <= 0.0F && sp.logit_bias.empty() && sp.repeat_penalty == 1.0F &&
            sp.presence_penalty == 0.0F && sp.frequency_penalty == 0.0F;

        // 1. Sample candidate token without mutating sampler penalty history.
        if (pure_greedy) {
            int32_t best = -1;
            float best_val = -std::numeric_limits<float>::infinity();
            for (int32_t v = 0; v < n_vocab; ++v) {
                const float val = logits[v];
                if (std::isfinite(val) && (best < 0 || val > best_val)) {
                    best_val = val;
                    best = v;
                }
            }
            if (best < 0) return LLAMA_TOKEN_NULL;
            tok = best;
        } else {
            auto& cand = grammar_candidates;
            cand.resize(static_cast<std::size_t>(n_vocab));
            for (int32_t v = 0; v < n_vocab; ++v) {
                cand[static_cast<std::size_t>(v)] = llama_token_data{v, logits[v], 0.0F};
            }
            llama_token_data_array cur_p{cand.data(), cand.size(), -1, false};
            llama_sampler_apply(sampler, &cur_p);
            if (cur_p.selected < 0 || static_cast<std::size_t>(cur_p.selected) >= cur_p.size) {
                return LLAMA_TOKEN_NULL;
            }
            tok = cur_p.data[cur_p.selected].id;
        }

        // 2. Test ONLY the 1 candidate token against the GBNF grammar.
        llama_token_data single{tok, 1.0F, 0.0F};
        llama_token_data_array single_arr{&single, 1, -1, false};
        llama_sampler_apply(grammar_sampler, &single_arr);
        if (std::isfinite(single.logit)) {
            llama_sampler_accept(grammar_sampler, tok);
            llama_sampler_accept(sampler, tok);
            if (sp.compute_logprobs && out_logprob != nullptr) {
                *out_logprob = ComputeTokenLogprobs(
                    std::span<const float>(logits, static_cast<std::size_t>(n_vocab)), tok, sp.top_logprobs);
            }
            return tok;
        }

        // 3. Rare rejection fallback: mask full vocabulary with GBNF, then resample.
        auto& cand = grammar_candidates;
        cand.resize(static_cast<std::size_t>(n_vocab));
        for (int32_t v = 0; v < n_vocab; ++v) {
            cand[static_cast<std::size_t>(v)] = llama_token_data{v, logits[v], 0.0F};
        }
        llama_token_data_array cur_p{cand.data(), cand.size(), -1, false};
        llama_sampler_apply(grammar_sampler, &cur_p);
        llama_sampler_apply(sampler, &cur_p);
        if (cur_p.selected < 0 || static_cast<std::size_t>(cur_p.selected) >= cur_p.size ||
            !std::isfinite(cur_p.data[cur_p.selected].logit)) {
            return LLAMA_TOKEN_NULL;
        }
        tok = cur_p.data[cur_p.selected].id;
        llama_sampler_accept(grammar_sampler, tok);
        llama_sampler_accept(sampler, tok);
        if (sp.compute_logprobs && out_logprob != nullptr) {
            *out_logprob = ComputeTokenLogprobs(
                std::span<const float>(logits, static_cast<std::size_t>(n_vocab)), tok, sp.top_logprobs);
        }
        return tok;
    }

#if defined(SONDER_LLAMACPP_HAS_MTMD)
    Status EvaluateMultimodalPrompt(const GenerateRequest& request, llama_seq_id slot_id, int32_t n_batch,
                                    int64_t n_ctx, int64_t& out_cur_pos, int32_t& out_prompt_tokens) {
        out_cur_pos = 0;
        out_prompt_tokens = 0;
        if (mtmd_ctx == nullptr) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "media_blobs provided but no mmproj multimodal projector is loaded (unsupported)");
        }
        std::vector<mtmd_bitmap*> bitmaps;
        bitmaps.reserve(request.media_blobs.size());
        struct BitmapGuard {
            std::vector<mtmd_bitmap*>& list;
            ~BitmapGuard() {
                for (mtmd_bitmap* b : list) {
                    if (b != nullptr) mtmd_bitmap_free(b);
                }
            }
        } bmp_guard{bitmaps};

        for (std::size_t i = 0; i < request.media_blobs.size(); ++i) {
            const auto& blob = request.media_blobs[i];
#if defined(SONDER_LLAMACPP_MTMD_WRAPPER_API)
            const mtmd_helper_init_opt hopt = mtmd_helper_init_opt_default();
            mtmd_helper_bitmap_wrapper wrapped = mtmd_helper_bitmap_init_from_buf(
                mtmd_ctx, blob.data(), blob.size(), /*placeholder=*/false, hopt);
            if (wrapped.video_ctx != nullptr) {
                mtmd_helper_video_free(wrapped.video_ctx);
            }
            mtmd_bitmap* bmp = wrapped.bitmap;
#else
            mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_buf(mtmd_ctx, blob.data(), blob.size());
#endif
            if (bmp == nullptr) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "failed to decode media blob at index " + std::to_string(i));
            }
            bitmaps.push_back(bmp);
        }

        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        if (chunks == nullptr) {
            return Status::Error(ErrorCode::kDecodeFailed, "mtmd_input_chunks_init failed");
        }
        struct ChunksGuard {
            mtmd_input_chunks* ptr;
            ~ChunksGuard() {
                if (ptr != nullptr) mtmd_input_chunks_free(ptr);
            }
        } chunks_guard{chunks};

        int32_t tok_rc = 0;
#if defined(SONDER_LLAMACPP_HAS_MTMD_PARTS)
        if (!request.media_prompt_parts.empty()) {
            // G0028-safe structured multimodal tokenization: trusted chat-template
            // scaffolding uses parse_special=true, untrusted message content uses
            // parse_special=false, and bitmaps are passed directly without marker
            // string substitution (zero special-token or marker injection).
            std::vector<mtmd_input_text> text_storage;
            std::vector<mtmd_input_part> part_storage;
            text_storage.reserve(request.media_prompt_parts.size());
            part_storage.reserve(request.media_prompt_parts.size());
            for (const auto& seg : request.media_prompt_parts) {
                if (seg.media_index.has_value()) {
                    if (*seg.media_index >= bitmaps.size()) {
                        return Status::Error(ErrorCode::kInvalidArgument,
                                             "multimodal media_index out of range");
                    }
                    mtmd_input_part p{};
                    p.text = nullptr;
                    p.bitmap = bitmaps[*seg.media_index];
                    part_storage.push_back(p);
                } else if (!seg.text.empty()) {
                    mtmd_input_text t{};
                    t.text = seg.text.c_str();
#if defined(SONDER_LLAMACPP_MTMD_TEXT_LEN)
                    t.text_len = seg.text.size();
#endif
                    t.add_special = false;
                    t.parse_special = seg.parse_special;
                    text_storage.push_back(t);
                    mtmd_input_part p{};
                    p.text = &text_storage.back();
                    p.bitmap = nullptr;
                    part_storage.push_back(p);
                }
            }
            std::vector<const mtmd_input_part*> part_ptrs;
            part_ptrs.reserve(part_storage.size());
            for (const auto& p : part_storage) part_ptrs.push_back(&p);
            tok_rc = mtmd_tokenize_from_parts(mtmd_ctx, chunks, part_ptrs.data(), part_ptrs.size(),
                                              /*add_special=*/true);
        } else
#endif
        {
            std::string prompt_text = request.media_prompt_text;
            if (prompt_text.empty()) {
                if (LlamaCppBackend* ow = owner.load(std::memory_order_acquire); ow != nullptr) {
                    if (Status s = ow->Detokenize(request.prompt, prompt_text, /*remove_special=*/false,
                                                  /*unparse_special=*/true);
                        !s.ok()) {
                        return s;
                    }
                }
            }
            const char* raw_marker = mtmd_get_marker(mtmd_ctx);
            const std::string marker =
                (raw_marker != nullptr && raw_marker[0] != '\0') ? std::string(raw_marker) : std::string(mtmd_default_marker());
            std::size_t marker_count = 0;
            if (!marker.empty()) {
                std::size_t search_pos = 0;
                while ((search_pos = prompt_text.find(marker, search_pos)) != std::string::npos) {
                    ++marker_count;
                    search_pos += marker.size();
                }
            }
            if (marker_count > bitmaps.size()) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "prompt contains more media markers (" + std::to_string(marker_count) +
                                         ") than media_blobs (" + std::to_string(bitmaps.size()) + ")");
            }
            if (marker_count < bitmaps.size()) {
                std::string prefix_markers;
                for (std::size_t m = marker_count; m < bitmaps.size(); ++m) {
                    prefix_markers += marker;
                    prefix_markers.push_back('\n');
                }
                prompt_text.insert(0, prefix_markers);
            }

            mtmd_input_text text_inp{};
            text_inp.text = prompt_text.c_str();
#if defined(SONDER_LLAMACPP_MTMD_TEXT_LEN)
            text_inp.text_len = prompt_text.size();
#endif
            text_inp.add_special = true;
            text_inp.parse_special = true;

            std::vector<const mtmd_bitmap*> const_bmps(bitmaps.begin(), bitmaps.end());
            tok_rc = mtmd_tokenize(mtmd_ctx, chunks, &text_inp, const_bmps.data(), const_bmps.size());
        }
        if (tok_rc != 0) {
            return Status::Error(ErrorCode::kDecodeFailed,
                                 "mtmd_tokenize failed (rc=" + std::to_string(tok_rc) + ")");
        }

        const llama_pos total_pos = mtmd_helper_get_n_pos(chunks);
        if (static_cast<int64_t>(total_pos) > n_ctx) {
            return Status::Error(ErrorCode::kContextOverflow,
                                 "multimodal prompt exceeds context window");
        }

        llama_memory_t mem = llama_get_memory(ctx);
        if (mem != nullptr) {
            llama_memory_seq_rm(mem, slot_id, -1, -1);
        }
        llama_pos new_n_past = 0;
        const int32_t eval_rc = mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, /*n_past=*/0, slot_id, n_batch,
                                                        /*logits_last=*/true, &new_n_past);
        if (eval_rc != 0) {
            if (mem != nullptr) {
                llama_memory_seq_rm(mem, slot_id, -1, -1);
            }
            return Status::Error(ErrorCode::kDecodeFailed,
                                 "mtmd_helper_eval_chunks failed (rc=" + std::to_string(eval_rc) + ")");
        }
        out_cur_pos = static_cast<int64_t>(new_n_past);
        out_prompt_tokens = static_cast<int32_t>(
            std::min<std::size_t>(mtmd_helper_get_n_tokens(chunks), static_cast<std::size_t>(INT32_MAX)));
        return Status::Ok();
    }
#endif

#if defined(SONDER_LLAMACPP_HAS_MTP)
    bool InitMtp(const LoadOptions& opts, int32_t n_nextn) {
        if (ctx == nullptr || model == nullptr || n_nextn <= 0) return false;
        llama_context_params cp_dft = MakeContextParams(opts);
        cp_dft.ctx_type = LLAMA_CONTEXT_TYPE_MTP;
        cp_dft.n_ctx = llama_n_ctx(ctx);
        cp_dft.abort_callback = &Impl::AbortCallback;
        cp_dft.abort_callback_data = this;
        // Keep cp_dft.n_rs_seq from MakeContextParams(opts) so recurrent/hybrid
        // draft layers can roll back partial speculative tails without silent drift.
#if defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
        cp_dft.n_outputs_max_per_seq = 1;
        cp_dft.n_outputs_max = std::max<uint32_t>(1u, cp_dft.n_seq_max);
#endif
#if defined(SONDER_LLAMACPP_HAS_CTX_OTHER)
        cp_dft.ctx_other = ctx;
#endif
        ctx_dft = llama_init_from_model(model, cp_dft);
        if (ctx_dft == nullptr) return false;

        mtp_n_embd = llama_model_n_embd_out(model);
        if (mtp_n_embd <= 0) {
            ReleaseMtp();
            return false;
        }
        mtp_n_layers = std::max<int32_t>(1, n_nextn);

        const auto n_b = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_batch(ctx_dft)));
        dft_batch = llama_batch_init(n_b, mtp_n_embd, 1);
        // llama_batch_init allocates only embd when embd > 0; MTP needs both token and embd.
        dft_batch.token = static_cast<llama_token*>(std::malloc(sizeof(llama_token) * static_cast<std::size_t>(n_b)));
        if (dft_batch.token == nullptr) {
            llama_batch_free(dft_batch);
            dft_batch = {};
            ReleaseMtp();
            return false;
        }
        dft_batch_cap = n_b;
        dft_batch_allocated = true;

        llama_set_embeddings_nextn(ctx, true, /*masked=*/false);
        llama_set_embeddings_nextn(ctx_dft, true, /*masked=*/true);

#if defined(SONDER_LLAMACPP_HAS_CTX_OTHER)
        mtp_mem_shared = (llama_get_ctx_other(ctx_dft) == ctx);
#else
        mtp_mem_shared = false;
#endif
        mtp_chain_heads = (mtp_n_layers > 1) && !mtp_mem_shared;

        const std::size_t n_slots = slots.size();
        mtp_samplers.resize(n_slots);
        mtp_backend_chains.assign(n_slots, nullptr);
        for (std::size_t i = 0; i < n_slots; ++i) {
            SamplerPtr s(llama_sampler_chain_init(llama_sampler_chain_default_params()));
            llama_sampler_chain_add(s.get(), llama_sampler_init_greedy());
            mtp_samplers[i] = std::move(s);
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
            llama_sampler* bchain = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(bchain, llama_sampler_init_greedy());
            if (!llama_set_sampler(ctx_dft, static_cast<llama_seq_id>(i), bchain)) {
                llama_sampler_free(bchain);
                bchain = nullptr;
            }
            mtp_backend_chains[i] = bchain;
#endif
        }

        mtp_pending_h.assign(n_slots, std::vector<float>(static_cast<std::size_t>(mtp_n_embd), 0.0F));
        mtp_deferred.assign(n_slots, {});
        if (mtp_chain_heads) {
            mtp_chain_h.reserve(static_cast<std::size_t>(mtp_n_layers + 1) * static_cast<std::size_t>(mtp_n_embd));
        }
        return true;
    }

    bool MtpFlushDeferred(llama_seq_id slot_id, int32_t max_rows = -1) {
        const auto sidx = static_cast<std::size_t>(slot_id);
        if (ctx_dft == nullptr || sidx >= mtp_deferred.size()) return true;
        auto& d = mtp_deferred[sidx];
        if (!d.pending || d.tokens.empty() || mtp_mem_shared) {
            d.Clear();
            return true;
        }
        const int32_t total = static_cast<int32_t>(d.tokens.size());
        const int32_t n_flush = (max_rows > 0 && max_rows < total) ? max_rows : total;
        const std::size_t row_bytes = static_cast<std::size_t>(mtp_n_embd) * sizeof(float);
        for (int32_t off = 0; off < n_flush; off += dft_batch_cap) {
            const int32_t step_n = std::min<int32_t>(n_flush - off, dft_batch_cap);
            BatchClear(dft_batch);
            for (int32_t m = 0; m < step_n; ++m) {
                const auto idx = static_cast<std::size_t>(off + m);
                BatchAdd(dft_batch, d.tokens[idx], d.pos[idx], slot_id, /*logits=*/false);
            }
            std::memcpy(dft_batch.embd,
                        d.h.data() + static_cast<std::size_t>(off) * static_cast<std::size_t>(mtp_n_embd),
                        row_bytes * static_cast<std::size_t>(step_n));
            if (llama_decode(ctx_dft, dft_batch) != 0) {
                d.Clear();
                return false;
            }
        }
        d.Clear();
        return true;
    }

    // Eagerly feeds a prefill chunk into `ctx_dft` and captures only the final
    // token's nextn hidden state (avoids copying all 512 rows on every prefill ubatch).
    bool MtpProcessPrefill(const llama_batch& batch_in, llama_seq_id slot_id) {
        if (ctx_dft == nullptr || batch_in.n_tokens <= 0) return true;
        if (batch_in.token == nullptr || batch_in.embd != nullptr) return true;
        const auto sidx = static_cast<std::size_t>(slot_id);
        if (sidx >= slots.size()) return false;

        const int32_t n_tokens = batch_in.n_tokens;
        if (n_tokens > dft_batch_cap) return false;
        const std::size_t row_bytes = static_cast<std::size_t>(mtp_n_embd) * sizeof(float);

        if (!mtp_mem_shared) {
            auto& d = mtp_deferred[sidx];
            if (d.pending) {
                if (!d.pos.empty() && d.pos.back() + 1 == batch_in.pos[0]) {
                    if (!MtpFlushDeferred(slot_id)) return false;
                } else {
                    d.Clear();
                }
            }

            BatchClear(dft_batch);
            for (int32_t k = 0; k < n_tokens; ++k) {
                BatchAdd(dft_batch, batch_in.token[k], batch_in.pos[k], slot_id, /*logits=*/false);
            }
            std::memcpy(dft_batch.embd, mtp_pending_h[sidx].data(), row_bytes);
            if (n_tokens > 1) {
                const float* h_tgt = llama_get_embeddings_nextn(ctx);
                if (h_tgt == nullptr) return false;
                std::memcpy(dft_batch.embd + static_cast<std::size_t>(mtp_n_embd), h_tgt,
                            row_bytes * static_cast<std::size_t>(n_tokens - 1));
            }

            llama_memory_t mem_dft = llama_get_memory(ctx_dft);
            const int32_t n_heads = mtp_chain_heads ? mtp_n_layers : 1;
            bool ok = true;
            for (int32_t head = 0; head < n_heads; ++head) {
                if (mtp_chain_heads) {
                    llama_memory_seq_rm(mem_dft, slot_id, batch_in.pos[0], -1);
#if defined(SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET)
                    llama_set_nextn_layer_offset(ctx_dft, head);
#endif
                }
                if (llama_decode(ctx_dft, dft_batch) != 0) {
                    ok = false;
                    break;
                }
            }
#if defined(SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET)
            if (mtp_chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0);
            }
#endif
            if (!ok) return false;
        }

        const float* h_last = llama_get_embeddings_nextn_ith(ctx, n_tokens - 1);
        if (h_last == nullptr) return false;
        std::memcpy(mtp_pending_h[sidx].data(), h_last, row_bytes);
        return true;
    }

    // Commits ONLY the `1 + n_accepted_drafts` accepted tokens of a verification
    // batch to `ctx_dft`. On single-head non-shared MTP (e.g. Qwen3.5 / Bonsai-2),
    // stashes the accepted rows so the very next `MtpDraft()` fuses catch-up +
    // first draft into ONE `llama_decode(ctx_dft)` call (saving 1 draft decode
    // and 1 KV rollback per verification step).
    bool MtpCommitVerified(const llama_batch& batch_in, llama_seq_id slot_id, int32_t n_accepted_drafts) {
        if (ctx_dft == nullptr || batch_in.n_tokens <= 0) return true;
        if (batch_in.token == nullptr || batch_in.embd != nullptr) return true;
        const auto sidx = static_cast<std::size_t>(slot_id);
        if (sidx >= slots.size()) return false;

        const int32_t n_valid = std::clamp<int32_t>(1 + n_accepted_drafts, 1, batch_in.n_tokens);
        const std::size_t row_bytes = static_cast<std::size_t>(mtp_n_embd) * sizeof(float);
        const float* h_new_pending = llama_get_embeddings_nextn_ith(ctx, n_valid - 1);
        if (h_new_pending == nullptr) return false;

        if (!mtp_mem_shared && !mtp_chain_heads) {
            auto& d = mtp_deferred[sidx];
            if (d.pending && (d.pos.empty() || d.pos.back() + 1 != batch_in.pos[0])) {
                d.Clear();
            }
            if (d.pending && static_cast<int32_t>(d.tokens.size()) + n_valid >= dft_batch_cap) {
                if (!MtpFlushDeferred(slot_id)) return false;
            }
            const std::size_t old_rows = d.tokens.size();
            d.tokens.resize(old_rows + static_cast<std::size_t>(n_valid));
            d.pos.resize(old_rows + static_cast<std::size_t>(n_valid));
            d.h.resize((old_rows + static_cast<std::size_t>(n_valid)) * static_cast<std::size_t>(mtp_n_embd));
            for (int32_t m = 0; m < n_valid; ++m) {
                const float* h_row =
                    (m == 0) ? mtp_pending_h[sidx].data() : llama_get_embeddings_nextn_ith(ctx, m - 1);
                if (h_row == nullptr) {
                    d.Clear();
                    return false;
                }
                const std::size_t dst_row = old_rows + static_cast<std::size_t>(m);
                d.tokens[dst_row] = batch_in.token[m];
                d.pos[dst_row] = batch_in.pos[m];
                std::memcpy(d.h.data() + dst_row * static_cast<std::size_t>(mtp_n_embd), h_row, row_bytes);
            }
            d.pending = true;
            std::memcpy(mtp_pending_h[sidx].data(), h_new_pending, row_bytes);
            return true;
        }

        if (!mtp_mem_shared) {
            if (n_valid > dft_batch_cap) return false;
            BatchClear(dft_batch);
            for (int32_t m = 0; m < n_valid; ++m) {
                BatchAdd(dft_batch, batch_in.token[m], batch_in.pos[m], slot_id, /*logits=*/false);
            }
            std::memcpy(dft_batch.embd, mtp_pending_h[sidx].data(), row_bytes);
            if (n_valid > 1) {
                const float* h_tgt = llama_get_embeddings_nextn(ctx);
                if (h_tgt == nullptr) return false;
                std::memcpy(dft_batch.embd + static_cast<std::size_t>(mtp_n_embd), h_tgt,
                            row_bytes * static_cast<std::size_t>(n_valid - 1));
            }
            llama_memory_t mem_dft = llama_get_memory(ctx_dft);
            const int32_t n_heads = mtp_chain_heads ? mtp_n_layers : 1;
            bool ok = true;
            for (int32_t head = 0; head < n_heads; ++head) {
                if (mtp_chain_heads) {
                    llama_memory_seq_rm(mem_dft, slot_id, batch_in.pos[0], -1);
#if defined(SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET)
                    llama_set_nextn_layer_offset(ctx_dft, head);
#endif
                }
                if (llama_decode(ctx_dft, dft_batch) != 0) {
                    ok = false;
                    break;
                }
            }
#if defined(SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET)
            if (mtp_chain_heads) {
                llama_set_nextn_layer_offset(ctx_dft, 0);
            }
#endif
            if (!ok) return false;
        }

        std::memcpy(mtp_pending_h[sidx].data(), h_new_pending, row_bytes);
        return true;
    }

    // Drafts up to `max_draft` tokens from `ctx_dft` starting at `(id_last, n_past)`.
    // When deferred catch-up rows lead directly into `n_past`, fuses them into
    // step 0's `llama_decode(ctx_dft)` so catch-up + first draft cost a single pass.
    // Returns false if draft KV synchronization was lost (caller must disable MTP
    // on this slot until the slot is reset).
    bool MtpDraft(llama_seq_id slot_id, Token id_last, int64_t n_past, int32_t max_draft,
                  std::vector<Token>& draft_out) {
        draft_out.clear();
        if (ctx_dft == nullptr || max_draft <= 0) return true;
        const auto sidx = static_cast<std::size_t>(slot_id);
        if (sidx >= slots.size()) return false;

        const int32_t limit = mtp_chain_heads ? std::min(max_draft, mtp_n_layers) : max_draft;
        const std::size_t row_bytes = static_cast<std::size_t>(mtp_n_embd) * sizeof(float);

        bool fused_catchup = false;
        auto& d = mtp_deferred[sidx];
        if (d.pending && !mtp_mem_shared) {
            const bool leads_in = !d.pos.empty() && (d.pos.back() + 1 == static_cast<llama_pos>(n_past)) &&
                                  (static_cast<int32_t>(d.tokens.size()) + 1 <= dft_batch_cap);
            if (!leads_in) {
                if (!MtpFlushDeferred(slot_id)) {
                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                    d.Clear();
                    return false;
                }
            }
        }

        BatchClear(dft_batch);
        if (d.pending && !mtp_mem_shared) {
            const auto n_c = static_cast<int32_t>(d.tokens.size());
            for (int32_t m = 0; m < n_c; ++m) {
                const auto idx = static_cast<std::size_t>(m);
                BatchAdd(dft_batch, d.tokens[idx], d.pos[idx], slot_id, /*logits=*/false);
            }
            std::memcpy(dft_batch.embd, d.h.data(), row_bytes * static_cast<std::size_t>(n_c));
            fused_catchup = true;
        }
        const int32_t anchor_idx = dft_batch.n_tokens;
        BatchAdd(dft_batch, id_last, static_cast<llama_pos>(n_past), slot_id, /*logits=*/true);
        std::memcpy(dft_batch.embd + static_cast<std::size_t>(anchor_idx) * static_cast<std::size_t>(mtp_n_embd),
                    mtp_pending_h[sidx].data(), row_bytes);
        int32_t i_last = anchor_idx;

        if (mtp_chain_heads) {
            mtp_chain_h.assign(mtp_pending_h[sidx].begin(), mtp_pending_h[sidx].end());
        }

        llama_memory_t mem_dft = llama_get_memory(ctx_dft);
        llama_pos trim_from = static_cast<llama_pos>(n_past);
        for (int32_t step = 0; step < limit; ++step) {
            if (mtp_chain_heads) {
                llama_memory_seq_rm(mem_dft, slot_id, static_cast<llama_pos>(n_past), -1);
#if defined(SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET)
                llama_set_nextn_layer_offset(ctx_dft, step);
#endif
            }
            if (llama_decode(ctx_dft, dft_batch) != 0) {
                if (step == 0 && fused_catchup && !d.pos.empty()) {
                    trim_from = d.pos.front();
                }
                break;
            }
            if (step == 0 && fused_catchup) {
                d.Clear();
            }

            Token id = LLAMA_TOKEN_NULL;
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
            if (mtp_backend_chains[sidx] != nullptr) {
                id = llama_get_sampled_token_ith(ctx_dft, i_last);
            }
#endif
            if (id == LLAMA_TOKEN_NULL) {
                id = llama_sampler_sample(mtp_samplers[sidx].get(), ctx_dft, i_last);
            }
            draft_out.push_back(id);
            if (llama_vocab_is_eog(vocab, id)) break;
            if (static_cast<int32_t>(draft_out.size()) >= limit) break;

            const float* h_row = llama_get_embeddings_nextn_ith(ctx_dft, i_last);
            if (h_row == nullptr) break;

            BatchClear(dft_batch);
            if (mtp_chain_heads) {
                mtp_chain_h.insert(mtp_chain_h.end(), h_row, h_row + mtp_n_embd);
                const int32_t n_rows = static_cast<int32_t>(draft_out.size()) + 1;
                if (n_rows > dft_batch_cap) break;
                for (int32_t t = 0; t < n_rows; ++t) {
                    const Token tok = (t == 0) ? id_last : draft_out[static_cast<std::size_t>(t - 1)];
                    BatchAdd(dft_batch, tok, static_cast<llama_pos>(n_past + t), slot_id, t == n_rows - 1);
                    std::memcpy(dft_batch.embd + static_cast<std::size_t>(t) * static_cast<std::size_t>(mtp_n_embd),
                                mtp_chain_h.data() + static_cast<std::size_t>(t) * static_cast<std::size_t>(mtp_n_embd),
                                row_bytes);
                }
                i_last = dft_batch.n_tokens - 1;
            } else if (mtp_mem_shared) {
                BatchAdd(dft_batch, id, static_cast<llama_pos>(n_past), slot_id, /*logits=*/true);
                std::memcpy(dft_batch.embd, h_row, row_bytes);
                i_last = 0;
            } else {
                BatchAdd(dft_batch, id, static_cast<llama_pos>(n_past + step + 1), slot_id, /*logits=*/true);
                std::memcpy(dft_batch.embd, h_row, row_bytes);
                i_last = 0;
            }
        }

#if defined(SONDER_LLAMACPP_HAS_NEXTN_LAYER_OFFSET)
        if (mtp_chain_heads) {
            llama_set_nextn_layer_offset(ctx_dft, 0);
        }
#endif
        if (!mtp_mem_shared) {
            if (!llama_memory_seq_rm(mem_dft, slot_id, trim_from, -1)) {
                llama_memory_seq_rm(mem_dft, slot_id, -1, -1);
                d.Clear();
                draft_out.clear();
                return false;
            }
        }
        return true;
    }
#endif
};

std::string_view LlamaCppBackend::UpstreamVersion() noexcept { return SONDER_LLAMACPP_TAG_STRING; }

std::vector<DeviceInfo> LlamaCppBackend::EnumerateDevices() {
    EnsureBackendInitialized();
    std::vector<DeviceInfo> out;
    const std::size_t count = ggml_backend_dev_count();
    out.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        DeviceInfo info;
        if (const char* n = ggml_backend_dev_name(dev)) info.name = n;
        if (const char* d = ggml_backend_dev_description(dev)) info.description = d;
        info.kind = MapDeviceKind(ggml_backend_dev_type(dev));
        std::size_t free_b = 0;
        std::size_t total_b = 0;
        ggml_backend_dev_memory(dev, &free_b, &total_b);
        info.memory_free_bytes = free_b;
        info.memory_total_bytes = total_b;
        out.push_back(std::move(info));
    }
    return out;
}

std::string LlamaCppBackend::SystemInfo() {
    EnsureBackendInitialized();
    const char* s = llama_print_system_info();
    return s != nullptr ? std::string(s) : std::string();
}

ModelParamsSummary LlamaCppBackend::ModelParamsFor(const LoadOptions& options) {
    std::vector<void*> resolved_devices;
    (void)ResolveDevices(options.devices, &resolved_devices);
    std::vector<ggml_backend_dev_t> device_buf;
    std::vector<float> tensor_split_buf;
    return SummarizeModelParams(MakeModelParams(options, resolved_devices, device_buf, tensor_split_buf));
}

ModelParamsSummary LlamaCppBackend::UpstreamModelDefaults() {
    return SummarizeModelParams(llama_model_default_params());
}

ContextParamsSummary LlamaCppBackend::ContextParamsFor(const LoadOptions& options) {
    return Summarize(MakeContextParams(options));
}

ContextParamsSummary LlamaCppBackend::UpstreamContextDefaults() {
    return Summarize(llama_context_default_params());
}

LlamaCppBackend::LlamaCppBackend() : impl_(std::make_unique<Impl>()) {
    impl_->owner.store(this, std::memory_order_release);
}
LlamaCppBackend::~LlamaCppBackend() = default;
LlamaCppBackend::LlamaCppBackend(LlamaCppBackend&& other) noexcept : impl_(std::move(other.impl_)) {
    if (impl_) impl_->owner.store(this, std::memory_order_release);
}
LlamaCppBackend& LlamaCppBackend::operator=(LlamaCppBackend&& other) noexcept {
    if (this != &other) {
        impl_ = std::move(other.impl_);
        if (impl_) impl_->owner.store(this, std::memory_order_release);
    }
    return *this;
}

Status LlamaCppBackend::ResolveDevices(const std::vector<std::string>& device_names, std::vector<void*>* resolved) {
    EnsureBackendInitialized();
    if (resolved != nullptr) resolved->clear();
    if (device_names.size() > llama_max_devices()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "too many devices (llama.cpp accepts " + std::to_string(llama_max_devices()) + ")");
    }
    std::vector<void*> out;
    out.reserve(device_names.size());
    for (const std::string& name : device_names) {
        if (name.empty()) {
            return Status::Error(ErrorCode::kInvalidArgument, "device name in devices list is empty");
        }
        if (name.find('\0') != std::string::npos) {
            return Status::Error(ErrorCode::kInvalidArgument, "device name in devices list contains NUL");
        }
        ggml_backend_dev_t dev = ggml_backend_dev_by_name(name.c_str());
        if (dev == nullptr) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "device '" + name + "' is not available in this build");
        }
        out.push_back(dev);
    }
    if (resolved != nullptr) *resolved = std::move(out);
    return Status::Ok();
}

Status LlamaCppBackend::ResolveTensorOverrides(const std::vector<TensorOverride>& overrides,
                                               std::vector<void*>* buffer_types) {
    EnsureBackendInitialized();
    if (buffer_types != nullptr) buffer_types->clear();
    if (overrides.size() > llama_max_tensor_buft_overrides()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "too many tensor overrides (llama.cpp accepts " +
                                 std::to_string(llama_max_tensor_buft_overrides()) + ")");
    }
    for (const TensorOverride& o : overrides) {
        if (o.pattern.empty()) {
            return Status::Error(ErrorCode::kInvalidArgument, "tensor override pattern is empty");
        }
        try {
            (void)std::regex(o.pattern);
        } catch (const std::regex_error&) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "tensor override pattern is not a valid regex: " + o.pattern);
        }
        ggml_backend_dev_t dev = nullptr;
        std::string lower = o.device;
        std::transform(lower.begin(), lower.end(), lower.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (lower == "cpu") {
            dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        } else if (!o.device.empty()) {
            dev = ggml_backend_dev_by_name(o.device.c_str());
        }
        ggml_backend_buffer_type_t buft = dev != nullptr ? ggml_backend_dev_buffer_type(dev) : nullptr;
        if (buft == nullptr) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "tensor override device '" + o.device + "' is not available in this build");
        }
        if (buffer_types != nullptr) buffer_types->push_back(buft);
    }
    return Status::Ok();
}

Status LlamaCppBackend::Load(const LoadOptions& options) {
    if (options.model_path.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "model_path is empty");
    }
    // Validate placement and context options before touching the file so a bad
    // flag fails fast.
    if (Status s = ValidatePlacementOptions(options); !s.ok()) {
        return s;
    }
    std::vector<void*> resolved_devices;
    if (Status s = ResolveDevices(options.devices, &resolved_devices); !s.ok()) {
        return s;
    }
    std::vector<void*> override_buffers;
    if (Status s = ResolveTensorOverrides(options.tensor_overrides, &override_buffers); !s.ok()) {
        return s;
    }
    if (Status s = ValidateContextOptions(options); !s.ok()) {
        return s;
    }
    if (!options.tensor_split.empty() && options.tensor_split.size() > llama_max_devices()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "tensor_split exceeds llama_max_devices (" + std::to_string(llama_max_devices()) + ")");
    }
    std::error_code ec;
    const std::filesystem::path fs_path(std::u8string(options.model_path.begin(), options.model_path.end()));
    if (!std::filesystem::is_regular_file(fs_path, ec)) {
        return Status::Error(ErrorCode::kFileNotFound, "model file not found: " + options.model_path);
    }
    if (!options.mmproj_path.empty() && !options.vocab_only) {
#if !defined(SONDER_LLAMACPP_HAS_MTMD)
        return Status::Error(ErrorCode::kInvalidArgument,
                             "mmproj_path requires a build with SONDER_WITH_MTMD=ON (libmtmd unsupported)");
#else
        const std::filesystem::path mm_path(std::u8string(options.mmproj_path.begin(), options.mmproj_path.end()));
        if (!std::filesystem::is_regular_file(mm_path, ec)) {
            return Status::Error(ErrorCode::kFileNotFound, "mmproj file not found: " + options.mmproj_path);
        }
#endif
    }
    if (!options.vocab_only && options.n_batch == 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "n_batch must be > 0");
    }

    EnsureBackendInitialized();
    Unload();

    const auto start = Clock::now();
    std::vector<ggml_backend_dev_t> device_buf;
    std::vector<float> tensor_split_buf;
    llama_model_params mp = MakeModelParams(options, resolved_devices, device_buf, tensor_split_buf);
    // Null-terminated list; patterns point into `options`, which outlives the load.
    std::vector<llama_model_tensor_buft_override> overrides;
    if (!options.tensor_overrides.empty()) {
        overrides.reserve(options.tensor_overrides.size() + 1);
        for (std::size_t i = 0; i < options.tensor_overrides.size(); ++i) {
            overrides.push_back({options.tensor_overrides[i].pattern.c_str(),
                                 static_cast<ggml_backend_buffer_type_t>(override_buffers[i])});
        }
        overrides.push_back({nullptr, nullptr});
        mp.tensor_buft_overrides = overrides.data();
    }

    llama_model* model = llama_model_load_from_file(options.model_path.c_str(), mp);
    if (model == nullptr) {
        return Status::Error(ErrorCode::kLoadFailed, "llama.cpp failed to load: " + options.model_path);
    }
    impl_->model = model;
    impl_->vocab = llama_model_get_vocab(model);
    impl_->options = options;
    impl_->architecture = ReadModelArchitecture(model);

    if (!options.vocab_only) {
        llama_context_params cp = MakeContextParams(options);
        cp.abort_callback = &Impl::AbortCallback;
        cp.abort_callback_data = impl_.get();
        impl_->ctx = llama_init_from_model(model, cp);
        if (impl_->ctx == nullptr) {
            impl_->Release();
            return Status::Error(ErrorCode::kLoadFailed, "llama.cpp failed to create context");
        }

        // Pre-allocate reusable decode batch sized to the context's n_batch.
        const auto cap = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_batch(impl_->ctx)));
        impl_->batch = llama_batch_init(cap, /*embd=*/0, /*n_seq_max=*/1);
        impl_->batch_cap = cap;
        impl_->batch_allocated = true;

        // Initialize per-slot KV prefix cache metadata & L2 host-RAM Radix cache.
        const std::uint32_t n_slots = std::clamp<uint32_t>(options.n_slots, 1, kMaxSlots);
        impl_->slots.resize(n_slots);
        impl_->slot_busy.assign(n_slots, 0u);
        impl_->slot_mtp_synced.assign(n_slots, false);
        impl_->slot_checkpoints.assign(n_slots, {});
        for (std::uint32_t i = 0; i < n_slots; ++i) {
            impl_->slots[i].id = static_cast<std::int32_t>(i);
            impl_->slots[i].tokens.clear();
            impl_->slots[i].session_key.clear();
            impl_->slots[i].last_used = 0;
        }
        impl_->slot_clock = 0;
        const std::uint64_t l2_bytes64 = static_cast<std::uint64_t>(options.cache_ram_mib) << 20;
        impl_->radix_cache.SetMaxBytes(
            static_cast<std::size_t>(std::min<std::uint64_t>(l2_bytes64, std::numeric_limits<std::size_t>::max())));

#if defined(SONDER_LLAMACPP_HAS_MTMD)
        if (!options.mmproj_path.empty()) {
            mtmd_context_params mp_mtmd = mtmd_context_params_default();
            mp_mtmd.use_gpu = (options.n_gpu_layers != 0);
            mp_mtmd.print_timings = false;
            if (options.n_threads > 0) {
                mp_mtmd.n_threads = options.n_threads;
            }
            mp_mtmd.flash_attn_type = ToLlamaFlashAttn(options.flash_attn);
            impl_->mtmd_ctx = mtmd_init_from_file(options.mmproj_path.c_str(), model, mp_mtmd);
            if (impl_->mtmd_ctx == nullptr) {
                impl_->Release();
                return Status::Error(ErrorCode::kLoadFailed,
                                     "libmtmd failed to load mmproj: " + options.mmproj_path);
            }
        }
#endif

        // Resolve speculative decoding strategy.
        SpeculativeType requested_spec = SpeculativeType::kNone;
        (void)ParseSpeculativeType(options.spec_type, requested_spec);
        impl_->active_spec = SpeculativeType::kNone;

        if (requested_spec == SpeculativeType::kDraftMtp || requested_spec == SpeculativeType::kAuto) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
            const int32_t n_nextn = llama_model_n_layer_nextn(model);
            if (n_nextn > 0) {
                if (impl_->InitMtp(options, n_nextn)) {
                    impl_->active_spec = SpeculativeType::kDraftMtp;
                } else if (requested_spec == SpeculativeType::kDraftMtp) {
                    impl_->Release();
                    return Status::Error(ErrorCode::kLoadFailed, "llama.cpp failed to create MTP draft context");
                }
            }
#endif
            if (impl_->active_spec == SpeculativeType::kNone && requested_spec == SpeculativeType::kAuto) {
                impl_->active_spec = SpeculativeType::kNgramMod;
            }
        } else {
            impl_->active_spec = requested_spec;
        }

        if (impl_->active_spec == SpeculativeType::kDraftMtp) {
            impl_->active_draft_max = options.spec_draft_max > 0 ? options.spec_draft_max : kDefaultMtpDraftMax;
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (impl_->mtp_chain_heads) {
                impl_->active_draft_max = std::min(impl_->active_draft_max, impl_->mtp_n_layers);
            }
#endif
            // Hybrid draft-mtp + suffix escalation maintains a bounded zero-VRAM
            // SuffixMatcher alongside MTP to burst 6..16 tokens on verbatim matches.
            impl_->suffix_matcher = std::make_unique<SuffixMatcher>();
        } else if (impl_->active_spec == SpeculativeType::kSuffix) {
            impl_->active_draft_max = options.spec_draft_max > 0 ? options.spec_draft_max : kDefaultSuffixDraftMax;
            impl_->suffix_matcher = std::make_unique<SuffixMatcher>();
        } else if (impl_->active_spec == SpeculativeType::kNgramMod) {
            impl_->active_draft_max = options.spec_draft_max > 0 ? options.spec_draft_max : kDefaultNgramModDraftMax;
            const std::size_t min_draft =
                options.spec_draft_max > 0
                    ? std::min<std::size_t>(NgramModTable::kDefaultMinDraft,
                                            static_cast<std::size_t>(impl_->active_draft_max))
                    : NgramModTable::kDefaultMinDraft;
            impl_->ngram_mod = std::make_unique<NgramModTable>(NgramModTable::kDefaultMatchLen,
                                                               NgramModTable::kDefaultTableSize, min_draft,
                                                               static_cast<std::size_t>(impl_->active_draft_max));
        } else if (impl_->active_spec == SpeculativeType::kNgram) {
            impl_->active_draft_max = options.spec_draft_max > 0 ? options.spec_draft_max : kDefaultNgramDraftMax;
            impl_->ngram_matcher = NgramMatcher(/*exact_scan_fallback=*/true);
        } else {
            impl_->active_draft_max = 0;
        }

        if (n_slots > 1) {
            impl_->StartBatchWorker();
        }
    }

    impl_->Emit(TelemetryKind::kModelLoaded, 0, MsSince(start), options.model_path);
    return Status::Ok();
}

void LlamaCppBackend::Unload() noexcept {
    if (!impl_ || impl_->model == nullptr) return;
    impl_->Release();
    try {
        impl_->Emit(TelemetryKind::kModelUnloaded, 0, 0.0, impl_->options.model_path);
    } catch (...) {
        // Telemetry sinks must not break unload.
    }
}

bool LlamaCppBackend::IsLoaded() const noexcept { return impl_ && impl_->model != nullptr; }

bool LlamaCppBackend::HasMmproj() const noexcept {
#if defined(SONDER_LLAMACPP_HAS_MTMD)
    return impl_ && impl_->mtmd_ctx != nullptr;
#else
    return false;
#endif
}

std::uint32_t LlamaCppBackend::SlotCount() const noexcept {
    return impl_ ? static_cast<std::uint32_t>(impl_->slots.size()) : 0u;
}

ModelInfo LlamaCppBackend::GetModelInfo() const {
    ModelInfo info;
    if (!IsLoaded()) return info;
    char buf[256] = {};
    if (llama_model_desc(impl_->model, buf, sizeof(buf)) > 0) info.description = buf;
    info.size_bytes = llama_model_size(impl_->model);
    info.n_params = llama_model_n_params(impl_->model);
    info.n_vocab = llama_vocab_n_tokens(impl_->vocab);
    info.n_ctx = Impl::EffectiveCtxSeq(impl_->ctx);
    info.architecture = impl_->architecture;
    return info;
}

Status LlamaCppBackend::Tokenize(std::string_view text, bool add_special, std::vector<Token>& out,
                                 bool parse_special) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (text.size() > static_cast<std::size_t>(INT32_MAX)) {
        return Status::Error(ErrorCode::kInvalidArgument, "text too long");
    }
    const auto len = static_cast<int32_t>(text.size());
    // Upper bound: one token per byte plus BOS/EOS.
    out.resize(static_cast<std::size_t>(len) + 2);
    int32_t n = llama_tokenize(impl_->vocab, text.data(), len, out.data(),
                               static_cast<int32_t>(out.size()), add_special, parse_special);
    if (n < 0 && n != INT32_MIN) {
        out.resize(static_cast<std::size_t>(-n));
        n = llama_tokenize(impl_->vocab, text.data(), len, out.data(),
                           static_cast<int32_t>(out.size()), add_special, parse_special);
    }
    if (n < 0) {
        out.clear();
        return Status::Error(ErrorCode::kTokenizeFailed, "llama_tokenize failed");
    }
    out.resize(static_cast<std::size_t>(n));
    return Status::Ok();
}

Status LlamaCppBackend::TokenToPiece(Token token, std::string& out) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (token < 0 || token >= llama_vocab_n_tokens(impl_->vocab)) {
        return Status::Error(ErrorCode::kInvalidArgument, "token id out of range");
    }
    char stack_buf[64];
    int32_t n = llama_token_to_piece(impl_->vocab, token, stack_buf, sizeof(stack_buf), 0, /*special=*/false);
    if (n >= 0) {
        out.assign(stack_buf, static_cast<std::size_t>(n));
        return Status::Ok();
    }
    out.resize(static_cast<std::size_t>(-n));
    n = llama_token_to_piece(impl_->vocab, token, out.data(), static_cast<int32_t>(out.size()), 0, false);
    if (n < 0) {
        out.clear();
        return Status::Error(ErrorCode::kTokenizeFailed, "llama_token_to_piece failed");
    }
    out.resize(static_cast<std::size_t>(n));
    return Status::Ok();
}

Status LlamaCppBackend::Detokenize(std::span<const Token> tokens, std::string& out, bool remove_special,
                                   bool unparse_special) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (tokens.empty()) return Status::Ok();
    if (tokens.size() > static_cast<std::size_t>(INT32_MAX)) {
        return Status::Error(ErrorCode::kInvalidArgument, "token span too long");
    }
    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    for (Token t : tokens) {
        if (t < 0 || t >= n_vocab) {
            return Status::Error(ErrorCode::kInvalidArgument, "token id out of range");
        }
    }
    const auto n_tok = static_cast<int32_t>(tokens.size());
    out.resize(std::max<std::size_t>(64, tokens.size() * 8));
    int32_t n = llama_detokenize(impl_->vocab, tokens.data(), n_tok, out.data(),
                                 static_cast<int32_t>(out.size()), remove_special, unparse_special);
    if (n < 0 && n != INT32_MIN) {
        out.resize(static_cast<std::size_t>(-n));
        n = llama_detokenize(impl_->vocab, tokens.data(), n_tok, out.data(),
                             static_cast<int32_t>(out.size()), remove_special, unparse_special);
    }
    if (n < 0) {
        out.clear();
        return Status::Error(ErrorCode::kTokenizeFailed, "llama_detokenize failed");
    }
    out.resize(static_cast<std::size_t>(n));
    return Status::Ok();
}

bool LlamaCppBackend::IsEndOfGeneration(Token token) const {
    return IsLoaded() && llama_vocab_is_eog(impl_->vocab, token);
}

std::string LlamaCppBackend::ChatTemplate() const {
    if (!IsLoaded()) return {};
    const char* tmpl = llama_model_chat_template(impl_->model, /*name=*/nullptr);
    return tmpl != nullptr ? std::string(tmpl) : std::string();
}

Status LlamaCppBackend::ApplyChatTemplate(const std::vector<ChatTurn>& messages, bool add_assistant,
                                          std::string& out) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    const std::string tmpl = ChatTemplate();
    if (tmpl.empty()) return Status::Error(ErrorCode::kInvalidArgument, "model has no chat template");
    return FormatChat(tmpl, messages, add_assistant, out);
}

Status LlamaCppBackend::FormatChat(std::string_view chat_template, const std::vector<ChatTurn>& messages,
                                   bool add_assistant, std::string& out) {
    out.clear();
    if (chat_template.empty()) return Status::Error(ErrorCode::kInvalidArgument, "empty chat template");
    if (messages.empty()) return Status::Error(ErrorCode::kInvalidArgument, "no chat messages");
    const std::string tmpl(chat_template);
    std::vector<llama_chat_message> chat;
    chat.reserve(messages.size());
    std::size_t total = 0;
    for (const auto& m : messages) {
        chat.push_back(llama_chat_message{m.role.c_str(), m.content.c_str()});
        total += m.role.size() + m.content.size();
    }
    if (total > static_cast<std::size_t>(INT32_MAX / 4)) {
        return Status::Error(ErrorCode::kInvalidArgument, "chat too long");
    }
    std::vector<char> buf(total * 2 + 256);
    int32_t n = llama_chat_apply_template(tmpl.c_str(), chat.data(), chat.size(), add_assistant, buf.data(),
                                          static_cast<int32_t>(buf.size()));
    if (n > static_cast<int32_t>(buf.size())) {
        buf.resize(static_cast<std::size_t>(n));
        n = llama_chat_apply_template(tmpl.c_str(), chat.data(), chat.size(), add_assistant, buf.data(),
                                      static_cast<int32_t>(buf.size()));
    }
    if (n < 0) return Status::Error(ErrorCode::kInvalidArgument, "chat template not supported by llama.cpp");
    out.assign(buf.data(), static_cast<std::size_t>(n));
    return Status::Ok();
}

void LlamaCppBackend::SetTelemetrySink(TelemetrySink sink) { impl_->telemetry = std::move(sink); }

Status LlamaCppBackend::CaptureSequenceState(std::vector<std::uint8_t>& out_bytes, std::size_t max_bytes) {
    out_bytes.clear();
    if (max_bytes == 0) return Status::Error(ErrorCode::kInvalidArgument, "max_bytes must be > 0");
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return Status::Error(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");

    const std::size_t state_size = llama_state_seq_get_size(impl_->ctx, 0);
    if (state_size == 0) {
        return Status::Error(ErrorCode::kDecodeFailed, "llama_state_seq_get_size returned 0");
    }
    if (state_size > max_bytes) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "sequence state size (" + std::to_string(state_size) + " bytes) exceeds max_bytes (" +
                                 std::to_string(max_bytes) + ")");
    }
    out_bytes.resize(state_size);
    const std::size_t written = llama_state_seq_get_data(impl_->ctx, out_bytes.data(), out_bytes.size(), 0);
    if (written == 0 || written > state_size) {
        out_bytes.clear();
        return Status::Error(ErrorCode::kDecodeFailed,
                             "llama_state_seq_get_data failed (wrote " + std::to_string(written) + " of " +
                                 std::to_string(state_size) + " bytes)");
    }
    out_bytes.resize(written);
    return Status::Ok();
}

Status LlamaCppBackend::RestoreSequenceState(std::span<const std::uint8_t> bytes, std::size_t max_bytes) {
    if (bytes.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "sequence state buffer is empty");
    }
    if (max_bytes == 0 || bytes.size() > max_bytes) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "sequence state size (" + std::to_string(bytes.size()) + " bytes) exceeds max_bytes (" +
                                 std::to_string(max_bytes) + ")");
    }
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return Status::Error(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");
    const std::size_t read = llama_state_seq_set_data(impl_->ctx, bytes.data(), bytes.size(), 0);
    if (read == 0 || read != bytes.size()) {
        return Status::Error(ErrorCode::kDecodeFailed,
                             "llama_state_seq_set_data failed (read " + std::to_string(read) + " of " +
                                 std::to_string(bytes.size()) + " bytes)");
    }
    if (!impl_->slots.empty()) {
        impl_->slots[0].tokens.clear();
        impl_->slots[0].session_key.clear();
        impl_->slots[0].last_used = 0;
    }
    return Status::Ok();
}

Status LlamaCppBackend::Embed(const std::vector<std::vector<Token>>& inputs, bool normalize, EmbedOutput& out,
                              const CancelPredicate& should_cancel) {
    out = {};
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return Status::Error(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");
    if (inputs.empty()) return Status::Error(ErrorCode::kInvalidArgument, "embedding inputs are empty");

    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
#if defined(SONDER_LLAMACPP_HAS_N_EMBD_OUT)
    int32_t n_embd = llama_model_n_embd_out(impl_->model);
    if (n_embd <= 0) {
        n_embd = llama_model_n_embd(impl_->model);
    }
#else
    const int32_t n_embd = llama_model_n_embd(impl_->model);
#endif
    if (n_embd <= 0) {
        return Status::Error(ErrorCode::kDecodeFailed, "model reports invalid embedding dimension");
    }

    const uint32_t base_ctx = std::max<uint32_t>(256u, Impl::EffectiveCtxSeq(impl_->ctx));
    std::size_t max_input_len = 0;
    for (const auto& seq : inputs) {
        if (seq.empty()) return Status::Error(ErrorCode::kInvalidArgument, "embedding input is empty");
        if (seq.size() > static_cast<std::size_t>(base_ctx)) {
            return Status::Error(ErrorCode::kContextOverflow, "embedding input exceeds embedding context window");
        }
        for (Token t : seq) {
            if (t < 0 || t >= n_vocab) {
                return Status::Error(ErrorCode::kInvalidArgument, "embedding token out of range");
            }
        }
        max_input_len = std::max(max_input_len, seq.size());
    }

    // Lazily allocate an isolated lightweight embedding context sized to demand
    // (using the model's configured KV cache type & flash attention) so Embed()
    // never disturbs live KV prefix caches or MTP/backend-sampler graphs and
    // minimizes extra VRAM footprint.
    const uint32_t aligned_need =
        static_cast<uint32_t>(((max_input_len + 255u) / 256u) * 256u);
    const uint32_t want_ctx = std::min<uint32_t>(base_ctx, std::max<uint32_t>(256u, aligned_need));
    if (impl_->embed_ctx != nullptr && llama_n_ctx(impl_->embed_ctx) < want_ctx) {
        if (impl_->embed_batch_allocated) {
            llama_batch_free(impl_->embed_batch);
            impl_->embed_batch = {};
            impl_->embed_batch_cap = 0;
            impl_->embed_batch_allocated = false;
        }
        llama_free(impl_->embed_ctx);
        impl_->embed_ctx = nullptr;
    }
    if (impl_->embed_ctx == nullptr) {
        llama_context_params cp_emb = llama_context_default_params();
        cp_emb.n_ctx = want_ctx;
        // Keep n_batch == n_ubatch == want_ctx so inputs up to want_ctx are
        // evaluated in a single pass (llama_decode/encode clears embd_seq per call).
        cp_emb.n_batch = want_ctx;
        cp_emb.n_ubatch = want_ctx;
        cp_emb.n_seq_max = 1;
        cp_emb.embeddings = true;
        cp_emb.pooling_type =
            llama_model_has_encoder(impl_->model) ? LLAMA_POOLING_TYPE_UNSPECIFIED : LLAMA_POOLING_TYPE_MEAN;
        cp_emb.type_k = ToGgmlType(impl_->options.type_k);
        cp_emb.type_v = ToGgmlType(impl_->options.type_v);
        cp_emb.flash_attn_type = ToLlamaFlashAttn(impl_->options.flash_attn);
        cp_emb.offload_kqv = true;
        cp_emb.no_perf = true;
        if (impl_->options.n_threads > 0) {
            cp_emb.n_threads = impl_->options.n_threads;
            cp_emb.n_threads_batch = impl_->options.n_threads;
        }
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ)
        cp_emb.n_rs_seq = 0;
#endif
#if defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
        cp_emb.n_outputs_max = cp_emb.n_batch;
        cp_emb.n_outputs_max_per_seq = cp_emb.n_batch;
#endif
        cp_emb.abort_callback = &Impl::AbortCallback;
        cp_emb.abort_callback_data = impl_.get();
        impl_->embed_ctx = llama_init_from_model(impl_->model, cp_emb);
        if (impl_->embed_ctx == nullptr) {
            return Status::Error(ErrorCode::kDecodeFailed, "failed to create embedding context");
        }
        const auto cap = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_batch(impl_->embed_ctx)));
        impl_->embed_batch = llama_batch_init(cap, /*embd=*/0, /*n_seq_max=*/1);
        impl_->embed_batch_cap = cap;
        impl_->embed_batch_allocated = true;
    }

    llama_context* ectx = impl_->embed_ctx;

    impl_->abort_requested.store(false, std::memory_order_relaxed);
    impl_->active_cancel = &should_cancel;
    struct EmbedGuard {
        Impl* impl;
        ~EmbedGuard() { impl->active_cancel = nullptr; }
    } guard{impl_.get()};

    auto cancelled = [&] {
        return impl_->abort_requested.load(std::memory_order_relaxed) || (should_cancel && should_cancel());
    };

    const enum llama_pooling_type pooling = llama_pooling_type(ectx);
    if (pooling == LLAMA_POOLING_TYPE_RANK) {
        return Status::Error(ErrorCode::kDecodeFailed,
                             "reranking/classification pooling is not supported for dense embeddings");
    }
    const bool encoder_only = llama_model_has_encoder(impl_->model) && !llama_model_has_decoder(impl_->model);
    const auto n_batch = static_cast<std::size_t>(std::max<int32_t>(1, impl_->embed_batch_cap));
    const auto start = Clock::now();
    out.embeddings.reserve(inputs.size());
    std::int64_t total_prompt_tokens = 0;

    for (const auto& seq : inputs) {
        if (cancelled()) {
            return Status::Error(ErrorCode::kDecodeFailed, "embedding cancelled");
        }
        if (llama_memory_t emem = llama_get_memory(ectx); emem != nullptr) {
            llama_memory_clear(emem, true);
        }
        const std::size_t n_tok = seq.size();
        if (n_tok > n_batch) {
            return Status::Error(ErrorCode::kContextOverflow,
                                 "embedding input exceeds embedding batch capacity");
        }
        BatchClear(impl_->embed_batch);
        for (std::size_t i = 0; i < n_tok; ++i) {
            const bool want_out = (pooling != LLAMA_POOLING_TYPE_NONE) || (i + 1 == n_tok);
            BatchAdd(impl_->embed_batch, seq[i], static_cast<llama_pos>(i), /*seq_id=*/0, want_out);
        }
        const int32_t rc = encoder_only ? llama_encode(ectx, impl_->embed_batch)
                                        : llama_decode(ectx, impl_->embed_batch);
        if (rc != 0) {
            return Status::Error(ErrorCode::kDecodeFailed,
                                 "llama embedding pass failed (rc=" + std::to_string(rc) + ")");
        }

        const float* raw = nullptr;
        if (pooling != LLAMA_POOLING_TYPE_NONE) {
            raw = llama_get_embeddings_seq(ectx, 0);
        }
        if (raw == nullptr) {
            raw = llama_get_embeddings_ith(ectx, -1);
        }
        if (raw == nullptr) {
            raw = llama_get_embeddings(ectx);
        }
        if (raw == nullptr) {
            return Status::Error(ErrorCode::kDecodeFailed, "llama.cpp returned null embeddings");
        }

        std::vector<float> vec(raw, raw + static_cast<std::size_t>(n_embd));
        if (normalize) {
            double sq_sum = 0.0;
            for (float v : vec) sq_sum += static_cast<double>(v) * static_cast<double>(v);
            const double norm = std::sqrt(sq_sum);
            if (norm > 0.0) {
                const auto inv = static_cast<float>(1.0 / norm);
                for (float& v : vec) v *= inv;
            }
        }
        out.embeddings.push_back(std::move(vec));
        total_prompt_tokens += static_cast<std::int64_t>(n_tok);
    }

    out.prompt_tokens = static_cast<std::int32_t>(
        std::min<std::int64_t>(total_prompt_tokens, static_cast<std::int64_t>(INT32_MAX)));
    out.eval_ms = MsSince(start);
    return Status::Ok();
}

void LlamaCppBackend::Impl::StartBatchWorker() {
    if (batch_worker_running.load(std::memory_order_acquire)) return;
    worker_stop.store(false, std::memory_order_relaxed);
    batch_worker_running.store(true, std::memory_order_release);
    batch_worker = std::thread([this] { BatchWorkerLoop(); });
}

GenerateResult LlamaCppBackend::Generate(const GenerateRequest& request, const TokenCallback& on_token,
                                         const CancelPredicate& should_cancel) {
    GenerateResult result;
    auto fail = [&](ErrorCode code, std::string msg) {
        result.status = Status::Error(code, std::move(msg));
        result.stop_reason = StopReason::kError;
        return result;
    };

    if (!IsLoaded()) return fail(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return fail(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");
    if (request.prompt.empty()) return fail(ErrorCode::kInvalidArgument, "prompt is empty");
    if (request.max_new_tokens < 0) return fail(ErrorCode::kInvalidArgument, "max_new_tokens < 0");
    if (Status s = Validate(request.sampling); !s.ok()) return fail(s.code, s.message);
    if (Status s = ValidateMediaBlobs(request.media_blobs); !s.ok()) return fail(s.code, s.message);
    if (!request.media_blobs.empty() && !HasMmproj()) {
        return fail(ErrorCode::kInvalidArgument,
                    "media_blobs provided but no mmproj multimodal projector is loaded (unsupported)");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    for (Token t : request.prompt) {
        if (t < 0 || t >= n_vocab) return fail(ErrorCode::kInvalidArgument, "prompt token out of range");
    }
    for (const auto& entry : request.sampling.logit_bias) {
        if (entry.first < 0 || entry.first >= n_vocab) {
            return fail(ErrorCode::kInvalidArgument, "logit_bias token out of range");
        }
    }

    // When n_slots > 1, dispatch through the single-owner continuous-batching
    // worker loop so concurrent callers share the llama_context in lockstep.
    if (impl_->batch_worker_running.load(std::memory_order_acquire)) {
        const bool has_restore = !request.restore_state.empty() && request.restore_tokens > 0 &&
                                 request.restore_tokens < request.prompt.size();
        const bool has_capture = request.capture_at_tokens > 0 && request.capture_at_tokens <= request.prompt.size();
        const bool exclusive =
            has_restore || has_capture || !request.media_blobs.empty() || !request.reset_context;

        Impl::PendingRequest task;
        task.req = &request;
        task.on_token = &on_token;
        task.should_cancel = &should_cancel;
        task.exclusive = exclusive;
        {
            std::lock_guard<std::mutex> lock(impl_->queue_mu);
            if (impl_->worker_stop.load(std::memory_order_acquire)) {
                return fail(ErrorCode::kNotLoaded, "backend is unloading");
            }
            impl_->pending_queue.push_back(&task);
            impl_->pending_count.fetch_add(1, std::memory_order_release);
        }
        impl_->worker_cv.notify_one();

        std::unique_lock<std::mutex> dlock(task.done_mu);
        task.done_cv.wait(dlock, [&] { return task.done; });
        return std::move(task.result);
    }

    return impl_->GenerateSingleLocked(request, on_token, should_cancel);
}

GenerateResult LlamaCppBackend::Impl::GenerateSingleLocked(const GenerateRequest& request,
                                                           const TokenCallback& on_token,
                                                           const CancelPredicate& should_cancel) {
    GenerateResult result;
    auto fail = [&](ErrorCode code, std::string msg) {
        result.status = Status::Error(code, std::move(msg));
        result.stop_reason = StopReason::kError;
        return result;
    };

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t mem = llama_get_memory(ctx);
    const auto n_ctx = static_cast<int64_t>(EffectiveCtxSeq(ctx));
    const auto n_prompt = static_cast<int64_t>(request.prompt.size());

    // Build and validate the sampler chain (and separate GBNF grammar for fast
    // grammar-based rejection sampling) upfront so an invalid grammar fails fast
    // without wasting prefill compute.
    SamplerPtr sampler;
    SamplerPtr grammar_sampler;
    if (Status s = BuildSampler(request.sampling, vocab, n_vocab, static_cast<int32_t>(n_ctx), sampler,
                                grammar_sampler);
        !s.ok()) {
        return fail(s.code, s.message);
    }

    abort_requested.store(false, std::memory_order_relaxed);
    active_cancel = &should_cancel;
    batch_abort_fn = nullptr;
    struct ClearCancel {
        Impl* impl;
        ~ClearCancel() {
            impl->active_cancel = nullptr;
            impl->batch_abort_fn = nullptr;
        }
    } clear_cancel{this};

    auto cancelled = [&] {
        return abort_requested.load(std::memory_order_relaxed) || (should_cancel && should_cancel());
    };

    // ---- Slot selection, recurrent checkpoint restore, L2 Radix cache, and prefix cache reuse ----
    std::size_t slot_idx = 0;
    llama_seq_id slot_id = 0;
    int64_t prefill_start_offset = 0;
    bool restored_from_l2 = false;
    bool l2_mtp_synced = false;
    const bool has_media = !request.media_blobs.empty();
    const bool has_restore = !has_media && !request.restore_state.empty() && request.restore_tokens > 0 &&
                             request.restore_tokens < request.prompt.size();
    const bool has_capture = !has_media && request.capture_at_tokens > 0 &&
                             request.capture_at_tokens <= request.prompt.size();

    if (has_media) {
        llama_memory_seq_rm(mem, 0, -1, -1);
        ClearSlotCheckpoints(0);
        if (!slots.empty()) {
            slots[0].tokens.clear();
            slots[0].session_key = request.session_key;
            slots[0].last_used = ++slot_clock;
            if (!slot_mtp_synced.empty()) slot_mtp_synced[0] = false;
        }
    } else if (has_restore) {
        llama_memory_clear(mem, true);
        ResetAllSlots();
        Status rst = owner.load(std::memory_order_acquire)->RestoreSequenceState(request.restore_state,
                                                                                 request.max_capture_bytes);
        if (rst.ok()) {
            const int64_t restored_pos = llama_memory_seq_pos_max(mem, 0) + 1;
            if (restored_pos == static_cast<int64_t>(request.restore_tokens)) {
                result.state_restored = true;
                prefill_start_offset = static_cast<int64_t>(request.restore_tokens);
                if (!slots.empty()) {
                    slots[0].tokens.assign(request.prompt.begin(),
                                           request.prompt.begin() + prefill_start_offset);
                    slots[0].session_key = request.session_key;
                    slots[0].last_used = ++slot_clock;
                }
                if (request.on_state_restored) {
                    request.on_state_restored();
                }
            } else {
                llama_memory_clear(mem, true);
                ResetAllSlots();
                result.restore_status = Status::Error(
                    ErrorCode::kDecodeFailed, "restored sequence position (" + std::to_string(restored_pos) +
                                                  ") does not match expected token_position (" +
                                                  std::to_string(request.restore_tokens) + ")");
            }
        } else {
            llama_memory_clear(mem, true);
            ResetAllSlots();
            result.restore_status = std::move(rst);
        }
    } else if (options.prefix_cache && request.reset_context && !has_capture && !slots.empty()) {
        const SlotSelection sel = SelectPrefixSlot(slots, request.prompt, request.session_key);
        slot_idx = sel.slot_index;
        slot_id = static_cast<llama_seq_id>(slots[slot_idx].id);
        std::size_t lcp = sel.matched_prefix;

        const int64_t kv_len_raw = llama_memory_seq_pos_max(mem, slot_id) + 1;
        const std::size_t kv_len = kv_len_raw > 0 ? static_cast<std::size_t>(kv_len_raw) : 0;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        const std::size_t prev_slot_len = slots[slot_idx].tokens.size();
#endif
        lcp = std::min(lcp, kv_len);

        // Snapshot an evicted/diverged VRAM slot into the L2 Radix host-RAM
        // cache BEFORE overwriting or looking up L2 (prevents dangling pointers).
        if (radix_cache.enabled() && slots[slot_idx].tokens.size() >= kMinL2SnapshotTokens &&
            lcp < slots[slot_idx].tokens.size()) {
            SnapshotSlotToL2(slot_idx, slot_id);
        }

        // Check L2 Radix cache if VRAM did not already cover the prefix by at least
        // `kMinL2RestoreGainTokens` (avoids unnecessary LRU touches / stats churn).
        std::size_t l2_lcp = 0;
        bool l2_wiped = false;
        if (radix_cache.enabled() && request.prompt.size() > 1 && request.prompt.size() - 1 > lcp &&
            (request.prompt.size() - 1 - lcp) >= kMinL2RestoreGainTokens) {
            if (TryRestoreSlotFromL2(slot_idx, slot_id, request.prompt, lcp, l2_lcp, l2_mtp_synced, l2_wiped)) {
                lcp = l2_lcp;
                restored_from_l2 = true;
                slots[slot_idx].tokens.assign(request.prompt.begin(),
                                              request.prompt.begin() + static_cast<std::ptrdiff_t>(lcp));
            } else if (l2_wiped) {
                lcp = 0;
                slots[slot_idx].tokens.clear();
                if (slot_idx < slot_mtp_synced.size()) slot_mtp_synced[slot_idx] = false;
            }
        }

        if (!restored_from_l2) {
            if (lcp == 0) {
                llama_memory_seq_rm(mem, slot_id, -1, -1);
                ClearSlotCheckpoints(slot_idx);
                if (slot_idx < slot_mtp_synced.size()) slot_mtp_synced[slot_idx] = true;
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (ctx_dft != nullptr) {
                    if (!mtp_mem_shared) {
                        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                    }
                    std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                    mtp_deferred[slot_idx].Clear();
                }
#endif
            } else {
                bool restored_pending_h = false;
                if (lcp < kv_len) {
                    if (llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(lcp), -1)) {
                        PruneSlotCheckpointsAfter(slot_idx, lcp);
                    } else {
                        bool restored_from_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                        std::size_t ckpt_pos = 0;
                        if (RestorePartialCheckpoint(slot_idx, slot_id, lcp, ckpt_pos, restored_pending_h)) {
                            lcp = ckpt_pos;
                            restored_from_ckpt = true;
                        }
#endif
                        if (!restored_from_ckpt) {
                            llama_memory_seq_rm(mem, slot_id, -1, -1);
                            ClearSlotCheckpoints(slot_idx);
                            lcp = 0;
                        }
                    }
                }
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (ctx_dft != nullptr) {
                    if (lcp == 0) {
                        if (!mtp_mem_shared) {
                            llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                        }
                        std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                        mtp_deferred[slot_idx].Clear();
                        if (slot_idx < slot_mtp_synced.size()) slot_mtp_synced[slot_idx] = true;
                    } else {
                        if (!mtp_mem_shared) {
                            auto& d = mtp_deferred[slot_idx];
                            if (d.pending) {
                                const int64_t dft_len0 =
                                    llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id) + 1;
                                if (static_cast<int64_t>(lcp) > dft_len0 && !d.pos.empty() &&
                                    d.pos.front() == static_cast<llama_pos>(dft_len0) &&
                                    static_cast<int64_t>(d.pos.back()) + 1 >= static_cast<int64_t>(lcp)) {
                                    (void)MtpFlushDeferred(slot_id,
                                                           static_cast<int32_t>(static_cast<int64_t>(lcp) - dft_len0));
                                } else {
                                    d.Clear();
                                }
                            }
                            const int64_t dft_len = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id) + 1;
                            if (dft_len < static_cast<int64_t>(lcp)) {
                                llama_memory_seq_rm(mem, slot_id, -1, -1);
                                llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                                ClearSlotCheckpoints(slot_idx);
                                std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                                lcp = 0;
                            } else if (dft_len > static_cast<int64_t>(lcp)) {
                                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id,
                                                         static_cast<llama_pos>(lcp), -1)) {
                                    llama_memory_seq_rm(mem, slot_id, -1, -1);
                                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                                    ClearSlotCheckpoints(slot_idx);
                                    std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                                    lcp = 0;
                                }
                            }
                        }
                        if (lcp > 0 && lcp != prev_slot_len && !restored_pending_h) {
                            std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                        }
                        if (slot_idx < slot_mtp_synced.size()) slot_mtp_synced[slot_idx] = true;
                    }
                }
#else
                (void)restored_pending_h;
#endif
            }
        }

        prefill_start_offset = static_cast<int64_t>(lcp);
        result.cached_tokens = static_cast<std::int32_t>(
            std::min<std::size_t>(lcp, static_cast<std::size_t>(INT32_MAX)));
        slots[slot_idx].tokens.resize(lcp);
        slots[slot_idx].session_key = request.session_key;
        slots[slot_idx].last_used = ++slot_clock;
    } else if (request.reset_context) {
        llama_memory_clear(mem, true);
        ResetAllSlots();
        if (!slots.empty()) {
            slots[0].session_key = request.session_key;
            slots[0].last_used = ++slot_clock;
            if (!slot_mtp_synced.empty()) slot_mtp_synced[0] = true;
        }
    }

    const int64_t n_past = llama_memory_seq_pos_max(mem, slot_id) + 1;
    const int64_t remaining_prompt = n_prompt - prefill_start_offset;
    if (!has_media && n_past + remaining_prompt > n_ctx) {
        return fail(ErrorCode::kContextOverflow, "prompt does not fit in the context window");
    }

#if defined(SONDER_LLAMACPP_HAS_MTP)
    bool use_mtp = (active_spec == SpeculativeType::kDraftMtp) && (ctx_dft != nullptr) && !result.state_restored &&
                   !has_media && (!restored_from_l2 || l2_mtp_synced);
#else
    bool use_mtp = false;
    (void)l2_mtp_synced;
#endif
    bool use_hybrid_suffix = use_mtp && (suffix_matcher != nullptr);
    bool use_suffix = !use_mtp && (active_spec == SpeculativeType::kSuffix) && (suffix_matcher != nullptr) &&
                      active_draft_max > 0 && !has_media;
    bool use_ngram_mod = !use_mtp && !use_suffix && (active_spec == SpeculativeType::kNgramMod) &&
                         (ngram_mod != nullptr) && active_draft_max > 0 && !has_media;
    bool use_ngram_short =
        !use_mtp && !use_suffix && (active_spec == SpeculativeType::kNgram) && active_draft_max > 0 && !has_media;
    bool use_ngram = use_ngram_mod || use_ngram_short;
    if (architecture == sonder::inference::ModelArchitecture::recurrent) {
        use_mtp = false;
        use_hybrid_suffix = false;
        use_suffix = false;
        use_ngram_mod = false;
        use_ngram_short = false;
        use_ngram = false;
    }
    uint32_t rs_cap = 0;
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ)
    if (architecture == sonder::inference::ModelArchitecture::hybrid) {
        rs_cap = llama_n_rs_seq(ctx);
    }
#endif
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
    const bool can_hybrid_spec_ckpt =
        (architecture == sonder::inference::ModelArchitecture::hybrid) && (PartialCheckpointFlags() != 0);
#else
    const bool can_hybrid_spec_ckpt = false;
#endif
    int32_t effective_draft_max = active_draft_max;
    if (architecture == sonder::inference::ModelArchitecture::hybrid) {
        if (use_mtp) {
            if (rs_cap == 0) {
                use_mtp = false;
                use_hybrid_suffix = false;
                effective_draft_max = 0;
            } else {
                effective_draft_max = std::min<int32_t>(effective_draft_max, static_cast<int32_t>(rs_cap));
            }
        } else if (use_suffix || use_ngram) {
            if (rs_cap == 0 && !can_hybrid_spec_ckpt) {
                use_suffix = false;
                use_ngram = false;
                use_ngram_mod = false;
                use_ngram_short = false;
                effective_draft_max = 0;
            }
        }
    }

    // Bind GPU backend sampling when supported. Disabled when a grammar OR
    // per-token logprobs are active so raw logits are readable on CPU.
    [[maybe_unused]] bool target_gpu_sampled = false;
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
    if (!grammar_sampler && !request.sampling.compute_logprobs) {
        target_gpu_sampled = llama_set_sampler(ctx, slot_id, sampler.get());
    }
    struct ClearTargetSampler {
        llama_context* c = nullptr;
        llama_seq_id sid = 0;
        bool active = false;
        ~ClearTargetSampler() {
            if (active && c != nullptr) {
                llama_set_sampler(c, sid, nullptr);
            }
        }
    } clear_target_sampler{ctx, slot_id, target_gpu_sampled};
#endif

    auto sample_target = [&](int32_t idx, TokenLogprobInfo* out_lp = nullptr) -> Token {
        return SampleTokenAt(idx, request.sampling, sampler.get(), grammar_sampler.get(), target_gpu_sampled, n_vocab,
                             out_lp);
    };

    bool slot_valid = true;
    auto invalidate_slot = [&]() {
        slot_valid = false;
        llama_memory_seq_rm(mem, slot_id, -1, -1);
        if (slot_idx < slots.size()) {
            slots[slot_idx].tokens.clear();
            slots[slot_idx].session_key.clear();
            slots[slot_idx].last_used = 0;
        }
        if (slot_idx < slot_mtp_synced.size()) {
            slot_mtp_synced[slot_idx] = false;
        }
        ClearSlotCheckpoints(slot_idx);
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr) {
            if (!mtp_mem_shared) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
            }
            if (slot_idx < mtp_pending_h.size()) {
                std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
            }
            if (slot_idx < mtp_deferred.size()) {
                mtp_deferred[slot_idx].Clear();
            }
        }
#endif
    };

    // ---- Prefill ----
    const auto n_batch = static_cast<int64_t>(std::max<int32_t>(1, batch_cap));
    const auto n_ubatch = static_cast<int64_t>(std::max<uint32_t>(1, llama_n_ubatch(ctx)));
    const int64_t prefill_chunk = use_mtp ? std::min(n_batch, n_ubatch) : n_batch;
    const bool want_capture = !has_media && request.capture_at_tokens > static_cast<std::size_t>(prefill_start_offset) &&
                              request.capture_at_tokens <= request.prompt.size();
    const int64_t capture_offset = want_capture ? static_cast<int64_t>(request.capture_at_tokens) : -1;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
    const bool want_hybrid_prefix_ckpt =
        options.prefix_cache && !has_capture && !has_media && !result.state_restored &&
        architecture == sonder::inference::ModelArchitecture::hybrid && n_prompt >= 512 &&
        PartialCheckpointFlags() != 0;
#else
    const bool want_hybrid_prefix_ckpt = false;
#endif
    const int64_t hybrid_ckpt_offset =
        (want_hybrid_prefix_ckpt && (n_prompt - 16) > prefill_start_offset) ? (n_prompt - 16) : -1;

    int64_t cur_pos = n_past;
    const auto prefill_start = Clock::now();
    if (has_media) {
#if defined(SONDER_LLAMACPP_HAS_MTMD)
        int32_t mm_prompt_tokens = 0;
        if (Status s = EvaluateMultimodalPrompt(request, slot_id, static_cast<int32_t>(n_batch), n_ctx, cur_pos,
                                                mm_prompt_tokens);
            !s.ok()) {
            invalidate_slot();
            if (cancelled()) {
                result.stop_reason = StopReason::kCancelled;
                result.prefill_ms = MsSince(prefill_start);
                return result;
            }
            return fail(s.code, s.message);
        }
        result.prompt_tokens = mm_prompt_tokens;
        slot_valid = false;  // Multimodal image KV is not text-token prefix cached.
#else
        return fail(ErrorCode::kInvalidArgument,
                    "media_blobs provided but this build has no libmtmd support");
#endif
    } else {
        for (int64_t off = prefill_start_offset; off < n_prompt;) {
            if (cancelled()) {
                result.stop_reason = StopReason::kCancelled;
                result.prefill_ms = MsSince(prefill_start);
                return result;
            }
            int64_t step_end = std::min(n_prompt, off + prefill_chunk);
            if (capture_offset > off && step_end > capture_offset) {
                step_end = capture_offset;
            }
            if (hybrid_ckpt_offset > off && step_end > hybrid_ckpt_offset) {
                step_end = hybrid_ckpt_offset;
            }
            const auto n = static_cast<int32_t>(step_end - off);
            BatchClear(batch);
            for (int32_t j = 0; j < n; ++j) {
                const bool is_last_prompt_tok = (off + j + 1 == n_prompt);
                BatchAdd(batch, request.prompt[static_cast<std::size_t>(off + j)],
                         static_cast<llama_pos>(cur_pos + j), slot_id, is_last_prompt_tok);
            }
            const int32_t rc = llama_decode(ctx, batch);
            if (rc == 2) {
                if (!llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(cur_pos), -1)) {
                    invalidate_slot();
                }
                result.stop_reason = StopReason::kCancelled;
                result.prefill_ms = MsSince(prefill_start);
                return result;
            }
            if (rc != 0) {
                if (!llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(cur_pos), -1)) {
                    invalidate_slot();
                }
                return fail(ErrorCode::kDecodeFailed,
                            "llama_decode failed during prefill (rc=" + std::to_string(rc) + ")");
            }
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp && !MtpProcessPrefill(batch, slot_id)) {
                invalidate_slot();
                if (cancelled()) {
                    result.stop_reason = StopReason::kCancelled;
                    result.prefill_ms = MsSince(prefill_start);
                    return result;
                }
                return fail(ErrorCode::kDecodeFailed, "MTP draft context prefill sync failed");
            }
#endif
            if (slot_idx < slots.size()) {
                auto& stoks = slots[slot_idx].tokens;
                stoks.insert(stoks.end(), request.prompt.begin() + off, request.prompt.begin() + step_end);
            }
            result.prompt_tokens += n;
            cur_pos += n;
            off = step_end;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
            if (hybrid_ckpt_offset == off && slot_valid) {
                CapturePartialCheckpoint(slot_idx, slot_id, static_cast<std::size_t>(cur_pos));
            }
#endif
            if (capture_offset == off) {
                std::vector<std::uint8_t> captured;
                result.capture_status =
                    owner.load(std::memory_order_acquire)->CaptureSequenceState(captured, request.max_capture_bytes);
                if (result.capture_status.ok()) {
                    result.state_captured = true;
                    if (request.on_state_captured) {
                        request.on_state_captured(std::move(captured));
                    }
                }
            }
        }
    }
    result.prefill_ms = MsSince(prefill_start);
    Emit(TelemetryKind::kPrefillDone, result.prompt_tokens, result.prefill_ms);

    // ---- Streaming decode (with optional in-process speculation) ----
    Utf8StreamBuffer utf8;
    int64_t pos = cur_pos;
    std::size_t ngram_indexed = 0;
    const auto decode_start = Clock::now();
    auto finish = [&](StopReason reason) {
        result.stop_reason = reason;
        result.decode_ms = MsSince(decode_start);
        if (use_ngram_mod && ngram_mod != nullptr && reason != StopReason::kError) {
            ngram_indexed = ngram_mod->IndexRange(ngram_history, ngram_indexed, ngram_history.size());
        }
        if ((use_suffix || use_hybrid_suffix) && suffix_matcher != nullptr && reason != StopReason::kError) {
            suffix_matcher->EndRequest();
        }
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
        if (slot_valid && want_hybrid_prefix_ckpt && reason != StopReason::kError && reason != StopReason::kCancelled &&
            slot_idx < slots.size() && !slots[slot_idx].tokens.empty()) {
            bool mtp_ok = true;
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp && !mtp_mem_shared && slot_idx < mtp_deferred.size() && mtp_deferred[slot_idx].pending) {
                mtp_ok = MtpFlushDeferred(slot_id);
                if (!mtp_ok) {
                    invalidate_slot();
                }
            }
#endif
            if (mtp_ok && slot_valid) {
                CapturePartialCheckpoint(slot_idx, slot_id, slots[slot_idx].tokens.size());
            }
        }
#endif
        std::string tail = utf8.Flush();
        if (!tail.empty() && on_token && reason != StopReason::kError) {
            on_token(TokenEvent{-1, std::move(tail), result.generated_tokens, false, 0.0F, {}});
        }
        Emit(TelemetryKind::kGenerationDone, result.generated_tokens, result.decode_ms,
             std::string(ToString(reason)));
        return result;
    };

    if (request.max_new_tokens == 0) return finish(StopReason::kMaxTokens);
    if (cancelled()) return finish(StopReason::kCancelled);

    auto emit_token = [&](Token tok, const TokenLogprobInfo& lp, bool& keep_going) -> Status {
        std::string piece;
        if (Status s = owner.load(std::memory_order_acquire)->TokenToPiece(tok, piece); !s.ok()) return s;
        TokenEvent ev;
        ev.token = tok;
        ev.text = utf8.Push(piece);
        ev.index = result.generated_tokens;
        if (lp.valid) {
            ev.has_logprob = true;
            ev.logprob = lp.logprob;
            ev.top_logprobs = lp.top_logprobs;
        }
        ++result.generated_tokens;
        keep_going = on_token ? on_token(ev) : true;
        return Status::Ok();
    };

    // Sample first generated token from the prefill's final logits.
    const auto first_tok_start = Clock::now();
    TokenLogprobInfo first_lp;
    Token cur_tok = sample_target(-1, request.sampling.compute_logprobs ? &first_lp : nullptr);
    if (cur_tok == LLAMA_TOKEN_NULL) {
        result.status = Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
        return finish(StopReason::kError);
    }
    if (llama_vocab_is_eog(vocab, cur_tok)) return finish(StopReason::kEndOfGeneration);
    {
        bool keep_going = true;
        if (Status s = emit_token(cur_tok, first_lp, keep_going); !s.ok()) {
            result.status = s;
            return finish(StopReason::kError);
        }
        if (!keep_going) return finish(StopReason::kCancelled);
        Emit(TelemetryKind::kTokenDecoded, 1, MsSince(first_tok_start));
        if (result.generated_tokens >= request.max_new_tokens) return finish(StopReason::kMaxTokens);
        if (pos >= n_ctx) return finish(StopReason::kContextFull);
    }

    // Initialize suffix / n-gram index over prompt tokens.
    AdaptiveDraftController adaptive_ctrl(effective_draft_max);
    if ((use_suffix || use_hybrid_suffix) && suffix_matcher != nullptr) {
        suffix_matcher->BeginRequest(request.prompt);
    }
    if (use_ngram_mod && ngram_mod != nullptr) {
        ngram_history.assign(request.prompt.begin(), request.prompt.end());
        ngram_mod->BeginRequest(ngram_history);
        ngram_indexed =
            ngram_history.size() > ngram_mod->n_match() ? ngram_history.size() - ngram_mod->n_match() : 0;
    } else if (use_ngram_short) {
        ngram_matcher.Reset();
        ngram_history.assign(request.prompt.begin(), request.prompt.end());
        ngram_matcher.IndexRange(ngram_history, 0, ngram_history.size());
        ngram_indexed = ngram_history.size();
    }

    auto& draft_tokens = draft_scratch;
    auto& accepted_tokens = accepted_scratch;
    auto& accepted_lps = accepted_logprobs_scratch;

    while (true) {
        if (cancelled()) return finish(StopReason::kCancelled);

        const auto step_start = Clock::now();
        const int64_t rem_gen =
            static_cast<int64_t>(request.max_new_tokens) - static_cast<int64_t>(result.generated_tokens);
        const int64_t max_k = std::min<int64_t>({rem_gen - 1, n_ctx - pos - 1, n_batch - 1, n_ubatch - 1});

        draft_tokens.clear();
        bool step_used_suffix = false;
        if (max_k > 0) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp) {
                // Hybrid draft-mtp + suffix escalation: burst 6..16 tokens on
                // high-confidence verbatim matches >= 6 tokens.
                if (use_hybrid_suffix && suffix_matcher != nullptr &&
                    (architecture != sonder::inference::ModelArchitecture::hybrid || can_hybrid_spec_ckpt ||
                     rs_cap >= SuffixMatcher::kHybridMtpMinBurstDraft)) {
                    if (suffix_matcher->paused()) {
                        suffix_matcher->NoteSkippedStep();
                    } else {
                        const std::size_t hybrid_cap =
                            (architecture == sonder::inference::ModelArchitecture::hybrid && !can_hybrid_spec_ckpt)
                                ? std::min<std::size_t>(static_cast<std::size_t>(max_k), static_cast<std::size_t>(rs_cap))
                                : static_cast<std::size_t>(max_k);
                        SuffixMatchResult burst = suffix_matcher->TryHybridMtpBurstForCurrent(cur_tok, hybrid_cap);
                        if (!burst.draft.empty()) {
                            draft_tokens = std::move(burst.draft);
                            for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                                if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                                    draft_tokens.resize(j + 1);
                                    break;
                                }
                            }
                            step_used_suffix = !draft_tokens.empty();
                        }
                    }
                }
                if (!step_used_suffix) {
                    const auto k_lim = static_cast<int32_t>(std::min<int64_t>(max_k, effective_draft_max));
                    if (!MtpDraft(slot_id, cur_tok, pos, k_lim, draft_tokens)) {
                        if (cancelled()) return finish(StopReason::kCancelled);
                        use_mtp = false;
                        use_hybrid_suffix = false;
                        slot_valid = false;
                        if (slot_idx < slots.size()) {
                            slots[slot_idx].tokens.clear();
                            slots[slot_idx].session_key.clear();
                        }
                        if (slot_idx < slot_mtp_synced.size()) slot_mtp_synced[slot_idx] = false;
                        ClearSlotCheckpoints(slot_idx);
                        draft_tokens.clear();
                    }
                }
            } else
#endif
            if (use_suffix && suffix_matcher != nullptr) {
                if (suffix_matcher->paused()) {
                    suffix_matcher->NoteSkippedStep();
                } else {
                    const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, effective_draft_max));
                    SuffixMatchResult m = suffix_matcher->DraftForCurrent(cur_tok, k_lim);
                    if (!m.draft.empty()) {
                        draft_tokens = std::move(m.draft);
                        for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                            if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                                draft_tokens.resize(j + 1);
                                break;
                            }
                        }
                        step_used_suffix = !draft_tokens.empty();
                    }
                }
            } else if (use_ngram_mod && ngram_mod != nullptr) {
                const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, effective_draft_max));
                ngram_mod->Draft(ngram_history, cur_tok, k_lim, draft_tokens);
                for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                    if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                        draft_tokens.resize(j + 1);
                        break;
                    }
                }
            } else if (use_ngram_short) {
                const int32_t ctrl_len = adaptive_ctrl.NextDraftLength();
                if (ctrl_len > 0) {
                    ngram_history.push_back(cur_tok);
                    const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, ctrl_len));
                    ngram_matcher.Draft(ngram_history, k_lim, draft_tokens);
                    ngram_history.pop_back();
                    for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                        if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                            draft_tokens.resize(j + 1);
                            break;
                        }
                    }
                } else {
                    adaptive_ctrl.NoteSkippedStep();
                }
            }
        }

        [[maybe_unused]] bool have_spec_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
        if (!draft_tokens.empty() && architecture == sonder::inference::ModelArchitecture::hybrid &&
            draft_tokens.size() > static_cast<std::size_t>(rs_cap)) {
            const llama_state_seq_flags flags = PartialCheckpointFlags();
            const std::size_t sz = flags != 0 ? llama_state_seq_get_size_ext(ctx, slot_id, flags) : 0;
            if (sz > 0 && sz <= kMaxPartialCheckpointBytes) {
                if (spec_ckpt_buf.size() < sz) {
                    spec_ckpt_buf.resize(sz);
                }
                const std::size_t written =
                    llama_state_seq_get_data_ext(ctx, spec_ckpt_buf.data(), sz, slot_id, flags);
                if (written > 0 && written <= sz) {
                    spec_ckpt_size = written;
                    have_spec_ckpt = true;
                }
            }
            if (!have_spec_ckpt) {
                draft_tokens.clear();
                step_used_suffix = false;
            }
        }
#endif

        BatchClear(batch);
        BatchAdd(batch, cur_tok, static_cast<llama_pos>(pos), slot_id, /*logits=*/true);
        for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
            BatchAdd(batch, draft_tokens[j], static_cast<llama_pos>(pos + 1 + static_cast<int64_t>(j)), slot_id,
                     /*logits=*/true);
        }

        const int32_t rc = llama_decode(ctx, batch);
        if (rc != 0) {
            bool rolled_back = llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1);
            if (!rolled_back) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                if (have_spec_ckpt) {
                    const llama_state_seq_flags flags = PartialCheckpointFlags();
                    const std::size_t read =
                        llama_state_seq_set_data_ext(ctx, spec_ckpt_buf.data(), spec_ckpt_size, slot_id, flags);
                    rolled_back = (read > 0) && llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1);
                }
#endif
            }
            if (!rolled_back) {
                invalidate_slot();
            }
            if (rc == 2) return finish(StopReason::kCancelled);
            if (rc == 1) return finish(StopReason::kContextFull);
            result.status =
                Status::Error(ErrorCode::kDecodeFailed, "llama_decode failed (rc=" + std::to_string(rc) + ")");
            return finish(StopReason::kError);
        }

        // Fast path: non-speculative single-token step.
        if (draft_tokens.empty()) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp && !MtpCommitVerified(batch, slot_id, 0)) {
                invalidate_slot();
                result.status = Status::Error(ErrorCode::kDecodeFailed, "MTP draft context sync failed");
                return finish(StopReason::kError);
            }
#endif
            if (slot_valid && slot_idx < slots.size()) {
                slots[slot_idx].tokens.push_back(cur_tok);
            }
            if ((use_suffix || use_hybrid_suffix) && suffix_matcher != nullptr) {
                suffix_matcher->CommitToken(cur_tok);
            }
            if (use_ngram_mod && ngram_mod != nullptr) {
                ngram_history.push_back(cur_tok);
                ngram_indexed = ngram_mod->IndexRange(ngram_history, ngram_indexed, ngram_history.size());
            } else if (use_ngram_short) {
                ngram_history.push_back(cur_tok);
                ngram_matcher.IndexRange(ngram_history, ngram_indexed, ngram_history.size());
                ngram_indexed = ngram_history.size();
            }
            ++pos;

            TokenLogprobInfo step_lp;
            const Token next_tok = sample_target(0, request.sampling.compute_logprobs ? &step_lp : nullptr);
            if (next_tok == LLAMA_TOKEN_NULL) {
                result.status =
                    Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
                return finish(StopReason::kError);
            }
            if (llama_vocab_is_eog(vocab, next_tok)) return finish(StopReason::kEndOfGeneration);

            bool keep_going = true;
            if (Status s = emit_token(next_tok, step_lp, keep_going); !s.ok()) {
                result.status = s;
                return finish(StopReason::kError);
            }
            if (!keep_going) return finish(StopReason::kCancelled);
            Emit(TelemetryKind::kTokenDecoded, 1, MsSince(step_start));
            if (result.generated_tokens >= request.max_new_tokens) {
                if (use_ngram_mod) ngram_history.push_back(next_tok);
                if ((use_suffix || use_hybrid_suffix) && suffix_matcher != nullptr) {
                    suffix_matcher->CommitToken(next_tok);
                }
                return finish(StopReason::kMaxTokens);
            }
            if (pos >= n_ctx) return finish(StopReason::kContextFull);
            cur_tok = next_tok;
            continue;
        }

        // Speculative verification: sample from indices 0..k and accept matched prefix + 1 bonus token.
        const std::size_t k = draft_tokens.size();
        const Token step_base_tok = cur_tok;
        result.draft_tokens += static_cast<std::int32_t>(k);
        accepted_tokens.clear();
        accepted_lps.clear();
        std::size_t n_accepted = 0;
        bool sample_deadlock = false;
        for (std::size_t j = 0; j < k; ++j) {
            TokenLogprobInfo lp;
            const Token sampled =
                sample_target(static_cast<int32_t>(j), request.sampling.compute_logprobs ? &lp : nullptr);
            if (sampled == LLAMA_TOKEN_NULL) {
                sample_deadlock = true;
                break;
            }
            accepted_tokens.push_back(sampled);
            accepted_lps.push_back(std::move(lp));
            if (sampled == draft_tokens[j]) {
                ++n_accepted;
            } else {
                break;
            }
        }
        if (!sample_deadlock && n_accepted == k) {
            TokenLogprobInfo bonus_lp;
            const Token bonus =
                sample_target(static_cast<int32_t>(k), request.sampling.compute_logprobs ? &bonus_lp : nullptr);
            if (bonus == LLAMA_TOKEN_NULL) {
                sample_deadlock = true;
            } else {
                accepted_tokens.push_back(bonus);
                accepted_lps.push_back(std::move(bonus_lp));
            }
        }
        if (sample_deadlock) {
            invalidate_slot();
            result.status =
                Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
            return finish(StopReason::kError);
        }
        result.draft_accepted_tokens += static_cast<std::int32_t>(n_accepted);
        if (step_used_suffix && suffix_matcher != nullptr) {
            suffix_matcher->Observe(k, n_accepted);
        } else if (use_ngram_mod && ngram_mod != nullptr) {
            if (ngram_mod->Observe(k, n_accepted)) {
                ngram_indexed = 0;
            }
        } else if (use_ngram_short) {
            adaptive_ctrl.Observe(static_cast<std::int32_t>(k), static_cast<std::int32_t>(n_accepted));
        }

        if (slot_valid && slot_idx < slots.size()) {
            slots[slot_idx].tokens.push_back(step_base_tok);
        }
        if ((use_suffix || use_hybrid_suffix) && suffix_matcher != nullptr) {
            suffix_matcher->CommitToken(step_base_tok);
        }
        if (use_ngram) {
            ngram_history.push_back(step_base_tok);
        }
        int64_t new_pos = pos + 1;
        std::size_t committed_drafts = 0;

        auto rollback_rejected = [&](int64_t keep_pos, std::size_t keep_drafts) -> bool {
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp && !MtpCommitVerified(batch, slot_id, static_cast<int32_t>(keep_drafts))) {
                invalidate_slot();
                return false;
            }
#endif
            const int64_t batch_end_pos = pos + 1 + static_cast<int64_t>(k);
            if (keep_pos < batch_end_pos) {
                if (!llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(keep_pos), -1)) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                    if (have_spec_ckpt) {
                        const llama_state_seq_flags flags = PartialCheckpointFlags();
                        const std::size_t read = llama_state_seq_set_data_ext(ctx, spec_ckpt_buf.data(), spec_ckpt_size,
                                                                              slot_id, flags);
                        if (read == 0 || !llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1)) {
                            invalidate_slot();
                            return false;
                        }
                        BatchClear(batch);
                        BatchAdd(batch, step_base_tok, static_cast<llama_pos>(pos), slot_id, /*logits=*/false);
                        for (std::size_t j = 0; j < keep_drafts; ++j) {
                            BatchAdd(batch, draft_tokens[j],
                                     static_cast<llama_pos>(pos + 1 + static_cast<int64_t>(j)), slot_id,
                                     /*logits=*/false);
                        }
                        if (llama_decode(ctx, batch) != 0) {
                            invalidate_slot();
                            return false;
                        }
                        return true;
                    }
#else
                    (void)step_base_tok;
                    (void)keep_drafts;
#endif
                    invalidate_slot();
                    return false;
                }
            }
            return true;
        };

        for (std::size_t m = 0; m < accepted_tokens.size(); ++m) {
            const Token tok = accepted_tokens[m];
            const bool is_bonus = (m == n_accepted);
            if (llama_vocab_is_eog(vocab, tok)) {
                (void)rollback_rejected(new_pos, committed_drafts);
                if (m > 0) {
                    Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m), MsSince(step_start));
                }
                return finish(StopReason::kEndOfGeneration);
            }
            bool keep_going = true;
            if (Status s = emit_token(tok, accepted_lps[m], keep_going); !s.ok()) {
                (void)rollback_rejected(new_pos, committed_drafts);
                result.status = s;
                return finish(StopReason::kError);
            }
            if (!is_bonus) {
                if (slot_valid && slot_idx < slots.size()) {
                    slots[slot_idx].tokens.push_back(tok);
                }
                if ((use_suffix || use_hybrid_suffix) && suffix_matcher != nullptr) {
                    suffix_matcher->CommitToken(tok);
                }
                if (use_ngram) {
                    ngram_history.push_back(tok);
                }
                ++new_pos;
                ++committed_drafts;
            }
            if (!keep_going) {
                (void)rollback_rejected(new_pos, committed_drafts);
                return finish(StopReason::kCancelled);
            }
            if (result.generated_tokens >= request.max_new_tokens) {
                if (is_bonus && use_ngram_mod) ngram_history.push_back(tok);
                if (is_bonus && (use_suffix || use_hybrid_suffix) && suffix_matcher != nullptr) {
                    suffix_matcher->CommitToken(tok);
                }
                (void)rollback_rejected(new_pos, committed_drafts);
                Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m + 1), MsSince(step_start));
                return finish(StopReason::kMaxTokens);
            }
            if (is_bonus) {
                cur_tok = tok;
            }
        }

        if (!rollback_rejected(new_pos, committed_drafts)) {
            if (cancelled()) {
                return finish(StopReason::kCancelled);
            }
            result.status = Status::Error(ErrorCode::kDecodeFailed, "failed to trim rejected speculative KV tail");
            return finish(StopReason::kError);
        }
        if (use_ngram_mod && ngram_mod != nullptr) {
            ngram_indexed = ngram_mod->IndexRange(ngram_history, ngram_indexed, ngram_history.size());
        } else if (use_ngram_short) {
            ngram_matcher.IndexRange(ngram_history, ngram_indexed, ngram_history.size());
            ngram_indexed = ngram_history.size();
        }
        Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(accepted_tokens.size()), MsSince(step_start));
        pos = new_pos;
        if (pos >= n_ctx) return finish(StopReason::kContextFull);
    }
}

void LlamaCppBackend::Impl::BatchWorkerLoop() {
    struct ActiveBatchSlot {
        PendingRequest* task = nullptr;
        std::size_t slot_idx = 0;
        llama_seq_id slot_id = 0;
        BatchSlotPhase phase = BatchSlotPhase::kPrefill;
        std::size_t prefill_offset = 0;
        int64_t pos = 0;
        Token cur_tok = LLAMA_TOKEN_NULL;
        SamplerPtr sampler;
        SamplerPtr grammar_sampler;
        Utf8StreamBuffer utf8;
        Clock::time_point prefill_start{};
        Clock::time_point decode_start{};
        bool prefill_done = false;
        bool slot_valid = true;
        bool mtp_enabled = false;
        bool hybrid_suffix_enabled = false;
        bool suffix_enabled = false;
        bool ngram_mod_enabled = false;
        bool ngram_short_enabled = false;
        bool spec_initialized = false;
        bool target_gpu_sampled = false;
        bool want_hybrid_prefix_ckpt = false;
        int64_t hybrid_ckpt_offset = -1;
        int32_t effective_draft_max = 0;
        AdaptiveDraftController adaptive_ctrl{0};
        std::size_t ngram_indexed = 0;
        int32_t batch_logit_idx = -1;
        GenerateResult result{};

        [[nodiscard]] bool IsCancelled() const {
            return task != nullptr && task->should_cancel != nullptr && *task->should_cancel &&
                   (*task->should_cancel)();
        }
    };

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t mem = llama_get_memory(ctx);
    const auto n_ctx = static_cast<int64_t>(EffectiveCtxSeq(ctx));
    const auto n_batch = static_cast<int32_t>(std::max<int32_t>(1, batch_cap));
    const auto n_ubatch = static_cast<int32_t>(std::max<uint32_t>(1u, llama_n_ubatch(ctx)));
    const int32_t step_token_budget = std::min<int32_t>(n_batch, n_ubatch);
    const int32_t step_output_budget = std::max<int32_t>(1, static_cast<int32_t>(slots.size()));

    uint32_t rs_cap = 0;
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ)
    if (architecture == sonder::inference::ModelArchitecture::hybrid) {
        rs_cap = llama_n_rs_seq(ctx);
    }
#endif
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
    const bool can_hybrid_spec_ckpt =
        (architecture == sonder::inference::ModelArchitecture::hybrid) && (PartialCheckpointFlags() != 0);
#else
    const bool can_hybrid_spec_ckpt = false;
#endif

    std::vector<std::unique_ptr<ActiveBatchSlot>> active;
    std::size_t rr_cursor = 0;

    auto complete_task = [](PendingRequest* task, GenerateResult&& res) {
        if (task == nullptr) return;
        // Notify while holding `done_mu` so the waiting caller cannot wake,
        // return, and destroy its stack-allocated `PendingRequest` before
        // `done_cv.notify_one()` finishes.
        std::lock_guard<std::mutex> dlock(task->done_mu);
        task->result = std::move(res);
        task->done = true;
        task->done_cv.notify_one();
    };

    auto detach_gpu_sampler = [&](ActiveBatchSlot& s) {
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
        if (s.target_gpu_sampled) {
            llama_set_sampler(ctx, s.slot_id, nullptr);
            s.target_gpu_sampled = false;
        }
#else
        (void)s;
#endif
    };

    auto invalidate_slot = [&](ActiveBatchSlot& s) {
        s.slot_valid = false;
        s.mtp_enabled = false;
        s.hybrid_suffix_enabled = false;
        llama_memory_seq_rm(mem, s.slot_id, -1, -1);
        if (s.slot_idx < slots.size()) {
            slots[s.slot_idx].tokens.clear();
            slots[s.slot_idx].session_key.clear();
            slots[s.slot_idx].last_used = 0;
        }
        if (s.slot_idx < slot_mtp_synced.size()) {
            slot_mtp_synced[s.slot_idx] = false;
        }
        ClearSlotCheckpoints(s.slot_idx);
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr) {
            if (!mtp_mem_shared) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), s.slot_id, -1, -1);
            }
            if (s.slot_idx < mtp_pending_h.size()) {
                std::fill(mtp_pending_h[s.slot_idx].begin(), mtp_pending_h[s.slot_idx].end(), 0.0F);
            }
            if (s.slot_idx < mtp_deferred.size()) {
                mtp_deferred[s.slot_idx].Clear();
            }
        }
#endif
    };

    auto emit_slot_token = [&](ActiveBatchSlot& s, Token tok, const TokenLogprobInfo& lp, bool& keep_going) -> Status {
        std::string piece;
        if (Status st = owner.load(std::memory_order_acquire)->TokenToPiece(tok, piece); !st.ok()) return st;
        TokenEvent ev;
        ev.token = tok;
        ev.text = s.utf8.Push(piece);
        ev.index = s.result.generated_tokens;
        if (lp.valid) {
            ev.has_logprob = true;
            ev.logprob = lp.logprob;
            ev.top_logprobs = lp.top_logprobs;
        }
        ++s.result.generated_tokens;
        keep_going = (s.task->on_token != nullptr && *s.task->on_token) ? (*s.task->on_token)(ev) : true;
        return Status::Ok();
    };

    auto finish_slot = [&](ActiveBatchSlot& s, StopReason reason) {
        detach_gpu_sampler(s);
        s.result.stop_reason = reason;
        if (s.prefill_done) {
            s.result.decode_ms = MsSince(s.decode_start);
        } else {
            s.result.prefill_ms = MsSince(s.prefill_start);
        }
        if (reason != StopReason::kError) {
            if (s.ngram_mod_enabled && s.spec_initialized && ngram_mod != nullptr) {
                s.ngram_indexed = ngram_mod->IndexRange(ngram_history, s.ngram_indexed, ngram_history.size());
            }
            if ((s.suffix_enabled || s.hybrid_suffix_enabled) && s.spec_initialized && suffix_matcher != nullptr) {
                suffix_matcher->EndRequest();
            } else if (suffix_matcher != nullptr && s.slot_valid && s.slot_idx < slots.size() &&
                       !slots[s.slot_idx].tokens.empty()) {
                suffix_matcher->IngestSessionSequence(slots[s.slot_idx].tokens);
            }
        }
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
        if (s.slot_valid && s.want_hybrid_prefix_ckpt && reason != StopReason::kError &&
            reason != StopReason::kCancelled && s.slot_idx < slots.size() && !slots[s.slot_idx].tokens.empty()) {
            bool mtp_ok = true;
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (s.mtp_enabled && !mtp_mem_shared && s.slot_idx < mtp_deferred.size() &&
                mtp_deferred[s.slot_idx].pending) {
                mtp_ok = MtpFlushDeferred(s.slot_id);
                if (!mtp_ok) {
                    invalidate_slot(s);
                }
            }
#endif
            if (mtp_ok && s.slot_valid) {
                CapturePartialCheckpoint(s.slot_idx, s.slot_id, slots[s.slot_idx].tokens.size());
            }
        }
#endif
        std::string tail = s.utf8.Flush();
        if (!tail.empty() && s.task->on_token != nullptr && *s.task->on_token && reason != StopReason::kError) {
            (*s.task->on_token)(TokenEvent{-1, std::move(tail), s.result.generated_tokens, false, 0.0F, {}});
        }
        Emit(TelemetryKind::kGenerationDone, s.result.generated_tokens, s.result.decode_ms,
             std::string(ToString(reason)));
        if (s.slot_idx < slot_busy.size()) {
            slot_busy[s.slot_idx] = 0u;
        }
        complete_task(s.task, std::move(s.result));
        s.task = nullptr;
    };

    auto admit_request = [&](PendingRequest* task, const SlotSelection& sel,
                             bool allow_l2_snapshot) -> std::unique_ptr<ActiveBatchSlot> {
        auto s = std::make_unique<ActiveBatchSlot>();
        s->task = task;
        s->slot_idx = sel.slot_index;
        s->slot_id = static_cast<llama_seq_id>(slots[s->slot_idx].id);
        s->prefill_start = Clock::now();
        slot_busy[s->slot_idx] = 1u;

        const GenerateRequest& req = *task->req;
        if (Status st = BuildSampler(req.sampling, vocab, n_vocab, static_cast<int32_t>(n_ctx), s->sampler,
                                     s->grammar_sampler);
            !st.ok()) {
            s->result.status = std::move(st);
            finish_slot(*s, StopReason::kError);
            return nullptr;
        }
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
        // Attach the backend GPU sampler ONCE for the entire lifetime of this
        // request (matching llama-server's per-seq_id lifecycle). Never detach
        // mid-request: llama_sampler_chain_backend_init marks chain stages with
        // is_backend=true, so detaching and calling CPU llama_sampler_sample()
        // later would skip stages and trip GGML_ASSERT(cur_p.selected >= 0).
        if (!s->grammar_sampler && !req.sampling.compute_logprobs) {
            s->target_gpu_sampled = llama_set_sampler(ctx, s->slot_id, s->sampler.get());
        }
#endif

        std::size_t lcp = options.prefix_cache ? sel.matched_prefix : 0u;
        const int64_t kv_len_raw = llama_memory_seq_pos_max(mem, s->slot_id) + 1;
        const std::size_t kv_len = kv_len_raw > 0 ? static_cast<std::size_t>(kv_len_raw) : 0;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        const std::size_t prev_slot_len = slots[s->slot_idx].tokens.size();
#endif
        lcp = std::min(lcp, kv_len);

        if (allow_l2_snapshot && options.prefix_cache && radix_cache.enabled() &&
            slots[s->slot_idx].tokens.size() >= kMinL2SnapshotTokens && lcp < slots[s->slot_idx].tokens.size()) {
            SnapshotSlotToL2(s->slot_idx, s->slot_id);
        }

        // Zero-copy VRAM-to-VRAM shared prefix clone on attention-only unified
        // KV caches: if another slot (active or idle) already holds a materially
        // longer matching prefix in VRAM, share its [0, best_lcp) cells via
        // llama_memory_seq_cp instead of paying PCIe L2 restore or re-prefill.
        bool cloned_from_vram = false;
        bool l2_mtp_synced = false;
        if (options.prefix_cache && architecture == sonder::inference::ModelArchitecture::attention_only &&
            req.prompt.size() > 1 && lcp + 32 < req.prompt.size()) {
            const std::size_t max_reuse = req.prompt.size() - 1;
            std::size_t best_src_idx = slots.size();
            std::size_t best_src_lcp = lcp;
            for (std::size_t j = 0; j < slots.size(); ++j) {
                if (j == s->slot_idx || slots[j].tokens.empty()) continue;
                const std::size_t cand = std::min(LongestCommonPrefix(slots[j].tokens, req.prompt), max_reuse);
                if (cand <= best_src_lcp) continue;
                const auto src_sid = static_cast<llama_seq_id>(slots[j].id);
                const int64_t src_kv = llama_memory_seq_pos_max(mem, src_sid) + 1;
                if (src_kv >= static_cast<int64_t>(cand)) {
                    best_src_idx = j;
                    best_src_lcp = cand;
                }
            }
            if (best_src_idx < slots.size() && best_src_lcp >= lcp + 32u) {
                const auto src_sid = static_cast<llama_seq_id>(slots[best_src_idx].id);
                llama_memory_seq_rm(mem, s->slot_id, -1, -1);
                ClearSlotCheckpoints(s->slot_idx);
                llama_memory_seq_cp(mem, src_sid, s->slot_id, 0, static_cast<llama_pos>(best_src_lcp));
                if (llama_memory_seq_pos_max(mem, s->slot_id) + 1 == static_cast<int64_t>(best_src_lcp)) {
                    lcp = best_src_lcp;
                    cloned_from_vram = true;
                    slots[s->slot_idx].tokens.assign(req.prompt.begin(),
                                                     req.prompt.begin() + static_cast<std::ptrdiff_t>(lcp));
#if defined(SONDER_LLAMACPP_HAS_MTP)
                    if (ctx_dft != nullptr && s->slot_idx < mtp_pending_h.size()) {
                        mtp_deferred[s->slot_idx].Clear();
                        std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                        if (mtp_mem_shared) {
                            l2_mtp_synced = true;
                        } else {
                            llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                            llama_memory_seq_rm(mem_dft, s->slot_id, -1, -1);
                            const bool src_dft_ready =
                                best_src_idx < slot_mtp_synced.size() && slot_mtp_synced[best_src_idx] &&
                                (best_src_idx >= mtp_deferred.size() || !mtp_deferred[best_src_idx].pending) &&
                                (llama_memory_seq_pos_max(mem_dft, src_sid) + 1 >= static_cast<int64_t>(lcp));
                            if (src_dft_ready) {
                                llama_memory_seq_cp(mem_dft, src_sid, s->slot_id, 0, static_cast<llama_pos>(lcp));
                                l2_mtp_synced =
                                    (llama_memory_seq_pos_max(mem_dft, s->slot_id) + 1 == static_cast<int64_t>(lcp));
                                if (!l2_mtp_synced) {
                                    llama_memory_seq_rm(mem_dft, s->slot_id, -1, -1);
                                }
                            } else {
                                l2_mtp_synced = false;
                            }
                        }
                        if (s->slot_idx < slot_mtp_synced.size()) {
                            slot_mtp_synced[s->slot_idx] = l2_mtp_synced;
                        }
                    }
#endif
                } else {
                    llama_memory_seq_rm(mem, s->slot_id, -1, -1);
                    lcp = 0;
                    slots[s->slot_idx].tokens.clear();
                    if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = false;
                }
            }
        }

        bool restored_from_l2 = cloned_from_vram;
        std::size_t l2_lcp = 0;
        bool l2_wiped = false;
        if (!restored_from_l2 && options.prefix_cache && radix_cache.enabled() && req.prompt.size() > 1 &&
            req.prompt.size() - 1 > lcp && (req.prompt.size() - 1 - lcp) >= kMinL2RestoreGainTokens) {
            if (TryRestoreSlotFromL2(s->slot_idx, s->slot_id, req.prompt, lcp, l2_lcp, l2_mtp_synced, l2_wiped)) {
                lcp = l2_lcp;
                restored_from_l2 = true;
                slots[s->slot_idx].tokens.assign(req.prompt.begin(),
                                                 req.prompt.begin() + static_cast<std::ptrdiff_t>(lcp));
            } else if (l2_wiped) {
                lcp = 0;
                slots[s->slot_idx].tokens.clear();
                if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = false;
            }
        }

        if (!restored_from_l2) {
            if (lcp == 0) {
                llama_memory_seq_rm(mem, s->slot_id, -1, -1);
                ClearSlotCheckpoints(s->slot_idx);
                if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = true;
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (ctx_dft != nullptr) {
                    if (!mtp_mem_shared) {
                        llama_memory_seq_rm(llama_get_memory(ctx_dft), s->slot_id, -1, -1);
                    }
                    std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                    mtp_deferred[s->slot_idx].Clear();
                }
#endif
            } else {
                bool restored_pending_h = false;
                if (lcp < kv_len) {
                    if (llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(lcp), -1)) {
                        PruneSlotCheckpointsAfter(s->slot_idx, lcp);
                    } else {
                        bool restored_from_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                        std::size_t ckpt_pos = 0;
                        if (RestorePartialCheckpoint(s->slot_idx, s->slot_id, lcp, ckpt_pos, restored_pending_h)) {
                            lcp = ckpt_pos;
                            restored_from_ckpt = true;
                        }
#endif
                        if (!restored_from_ckpt) {
                            llama_memory_seq_rm(mem, s->slot_id, -1, -1);
                            ClearSlotCheckpoints(s->slot_idx);
                            lcp = 0;
                        }
                    }
                }
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (ctx_dft != nullptr) {
                    if (lcp == 0) {
                        if (!mtp_mem_shared) {
                            llama_memory_seq_rm(llama_get_memory(ctx_dft), s->slot_id, -1, -1);
                        }
                        std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                        mtp_deferred[s->slot_idx].Clear();
                        if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = true;
                    } else {
                        const bool was_synced =
                            s->slot_idx < slot_mtp_synced.size() ? slot_mtp_synced[s->slot_idx] : false;
                        if (!was_synced) {
                            if (!mtp_mem_shared) {
                                llama_memory_seq_rm(llama_get_memory(ctx_dft), s->slot_id, -1, -1);
                            }
                            std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                            mtp_deferred[s->slot_idx].Clear();
                        } else if (!mtp_mem_shared) {
                            auto& d = mtp_deferred[s->slot_idx];
                            if (d.pending) {
                                const int64_t dft_len0 =
                                    llama_memory_seq_pos_max(llama_get_memory(ctx_dft), s->slot_id) + 1;
                                if (static_cast<int64_t>(lcp) > dft_len0 && !d.pos.empty() &&
                                    d.pos.front() == static_cast<llama_pos>(dft_len0) &&
                                    static_cast<int64_t>(d.pos.back()) + 1 >= static_cast<int64_t>(lcp)) {
                                    (void)MtpFlushDeferred(s->slot_id,
                                                           static_cast<int32_t>(static_cast<int64_t>(lcp) - dft_len0));
                                } else {
                                    d.Clear();
                                }
                            }
                            const int64_t dft_len = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), s->slot_id) + 1;
                            if (dft_len < static_cast<int64_t>(lcp) ||
                                (dft_len > static_cast<int64_t>(lcp) &&
                                 !llama_memory_seq_rm(llama_get_memory(ctx_dft), s->slot_id,
                                                      static_cast<llama_pos>(lcp), -1))) {
                                llama_memory_seq_rm(llama_get_memory(ctx_dft), s->slot_id, -1, -1);
                                std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                                if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = false;
                            }
                        }
                        if (lcp > 0 && lcp != prev_slot_len && !restored_pending_h) {
                            std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                        }
                    }
                }
#else
                (void)restored_pending_h;
#endif
            }
        }

        const int64_t n_past = llama_memory_seq_pos_max(mem, s->slot_id) + 1;
        const auto n_prompt = static_cast<int64_t>(req.prompt.size());
        const int64_t rem_prompt = n_prompt - static_cast<int64_t>(lcp);
        if (n_past + rem_prompt > n_ctx) {
            invalidate_slot(*s);
            s->result.status = Status::Error(ErrorCode::kContextOverflow, "prompt does not fit in the context window");
            finish_slot(*s, StopReason::kError);
            return nullptr;
        }

        s->prefill_offset = lcp;
        s->pos = n_past;
        s->result.cached_tokens = static_cast<std::int32_t>(
            std::min<std::size_t>(lcp, static_cast<std::size_t>(INT32_MAX)));
        slots[s->slot_idx].tokens.resize(lcp);
        slots[s->slot_idx].session_key = req.session_key;
        slots[s->slot_idx].last_used = ++slot_clock;

#if defined(SONDER_LLAMACPP_HAS_MTP)
        const bool mtp_ok = (active_spec == SpeculativeType::kDraftMtp) && (ctx_dft != nullptr) &&
                            (!restored_from_l2 || l2_mtp_synced) &&
                            (s->slot_idx < slot_mtp_synced.size() && slot_mtp_synced[s->slot_idx]);
        s->mtp_enabled = mtp_ok;
#else
        (void)l2_mtp_synced;
        s->mtp_enabled = false;
#endif
        s->hybrid_suffix_enabled = s->mtp_enabled && (suffix_matcher != nullptr);
        s->suffix_enabled = !s->mtp_enabled && (active_spec == SpeculativeType::kSuffix) && (suffix_matcher != nullptr) &&
                            active_draft_max > 0;
        s->ngram_mod_enabled = !s->mtp_enabled && !s->suffix_enabled && (active_spec == SpeculativeType::kNgramMod) &&
                               (ngram_mod != nullptr) && active_draft_max > 0;
        s->ngram_short_enabled = !s->mtp_enabled && !s->suffix_enabled && (active_spec == SpeculativeType::kNgram) &&
                                 active_draft_max > 0;
        s->effective_draft_max = active_draft_max;
        if (architecture == sonder::inference::ModelArchitecture::recurrent) {
            s->mtp_enabled = false;
            s->hybrid_suffix_enabled = false;
            s->suffix_enabled = false;
            s->ngram_mod_enabled = false;
            s->ngram_short_enabled = false;
            s->effective_draft_max = 0;
        } else if (architecture == sonder::inference::ModelArchitecture::hybrid) {
            if (s->mtp_enabled) {
                if (rs_cap == 0) {
                    s->mtp_enabled = false;
                    s->hybrid_suffix_enabled = false;
                    s->effective_draft_max = 0;
                } else {
                    s->effective_draft_max = std::min<int32_t>(s->effective_draft_max, static_cast<int32_t>(rs_cap));
                }
            } else if ((s->suffix_enabled || s->ngram_mod_enabled || s->ngram_short_enabled) && rs_cap == 0 &&
                       !can_hybrid_spec_ckpt) {
                s->suffix_enabled = false;
                s->ngram_mod_enabled = false;
                s->ngram_short_enabled = false;
                s->effective_draft_max = 0;
            }
        }
        s->adaptive_ctrl.Reset(s->effective_draft_max);
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
        s->want_hybrid_prefix_ckpt = options.prefix_cache &&
                                     architecture == sonder::inference::ModelArchitecture::hybrid && n_prompt >= 512 &&
                                     PartialCheckpointFlags() != 0;
#else
        s->want_hybrid_prefix_ckpt = false;
#endif
        s->hybrid_ckpt_offset =
            (s->want_hybrid_prefix_ckpt && (n_prompt - 16) > static_cast<int64_t>(lcp)) ? (n_prompt - 16) : -1;
        return s;
    };

    // Evicts the least-recently-used idle (`slot_busy[i] == 0`) slot that still
    // holds cached KV tokens so unified KV pools (`kv_unified=true`) reclaim
    // space before failing with rc == 1 (`kContextFull`).
    auto evict_one_idle_slot = [&]() -> bool {
        std::size_t victim = slots.size();
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (i < slot_busy.size() && slot_busy[i] != 0u) continue;
            const auto sid = static_cast<llama_seq_id>(slots[i].id);
            const int64_t kv_max = llama_memory_seq_pos_max(mem, sid);
            if (kv_max < 0 && slots[i].tokens.empty()) continue;
            if (victim == slots.size() || slots[i].last_used < oldest) {
                victim = i;
                oldest = slots[i].last_used;
            }
        }
        if (victim == slots.size()) return false;
        const auto sid = static_cast<llama_seq_id>(slots[victim].id);
        if (options.prefix_cache && radix_cache.enabled() && slots[victim].tokens.size() >= kMinL2SnapshotTokens) {
            SnapshotSlotToL2(victim, sid);
        }
        llama_memory_seq_rm(mem, sid, -1, -1);
        slots[victim].tokens.clear();
        slots[victim].session_key.clear();
        slots[victim].last_used = 0;
        if (victim < slot_mtp_synced.size()) slot_mtp_synced[victim] = false;
        ClearSlotCheckpoints(victim);
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr) {
            if (!mtp_mem_shared) {
                llama_memory_seq_rm(llama_get_memory(ctx_dft), sid, -1, -1);
            }
            if (victim < mtp_pending_h.size()) {
                std::fill(mtp_pending_h[victim].begin(), mtp_pending_h[victim].end(), 0.0F);
            }
            if (victim < mtp_deferred.size()) mtp_deferred[victim].Clear();
        }
#endif
        return true;
    };

    struct SampledStepOutcome {
        std::size_t slot_index = 0;
        Token tok = LLAMA_TOKEN_NULL;
        TokenLogprobInfo lp;
        bool has_sample = false;
    };
    std::vector<BatchSlotPlanView> views;
    std::vector<SampledStepOutcome> outcomes;
    views.reserve(slots.size());
    outcomes.reserve(slots.size());

    PendingRequest* inflight_exclusive = nullptr;
    PendingRequest* inflight_admit = nullptr;
    std::size_t inflight_admit_slot = slots.size();

    try {
    while (true) {
        // 0. Drain all active slots cleanly if StopBatchWorker() requested shutdown.
        if (worker_stop.load(std::memory_order_acquire)) {
            for (auto& sp : active) {
                if (sp && sp->task != nullptr) {
                    finish_slot(*sp, StopReason::kCancelled);
                }
            }
            active.clear();
            break;
        }

        // 1. Sleep cleanly when no active requests are in flight.
        if (active.empty()) {
            std::unique_lock<std::mutex> qlock(queue_mu);
            worker_cv.wait(qlock, [&] { return worker_stop.load(std::memory_order_acquire) || !pending_queue.empty(); });
            if (worker_stop.load(std::memory_order_acquire)) {
                break;
            }
        }

        // 2. Admit pending requests into free slots (zero lock overhead when queue is empty).
        if (pending_count.load(std::memory_order_acquire) > 0) {
            while (true) {
                PendingRequest* exclusive_task = nullptr;
                PendingRequest* admitted_task = nullptr;
                SlotSelection sel{};
                {
                    std::lock_guard<std::mutex> qlock(queue_mu);
                    if (pending_queue.empty()) break;
                    PendingRequest* front = pending_queue.front();
                    if (front->exclusive) {
                        if (!active.empty()) break;  // drain active batch before exclusive request
                        pending_queue.pop_front();
                        pending_count.fetch_sub(1, std::memory_order_release);
                        exclusive_task = front;
                    } else {
                        const auto chosen =
                            SelectAvailablePrefixSlot(slots, slot_busy, front->req->prompt, front->req->session_key);
                        if (!chosen.has_value()) break;  // all n_slots busy
                        pending_queue.pop_front();
                        pending_count.fetch_sub(1, std::memory_order_release);
                        admitted_task = front;
                        sel = *chosen;
                        slot_busy[sel.slot_index] = 1u;
                    }
                }
                if (exclusive_task != nullptr) {
                    inflight_exclusive = exclusive_task;
                    GenerateResult res = GenerateSingleLocked(*exclusive_task->req, *exclusive_task->on_token,
                                                              *exclusive_task->should_cancel);
                    inflight_exclusive = nullptr;
                    complete_task(exclusive_task, std::move(res));
                    continue;
                }
                if (admitted_task != nullptr) {
                    inflight_admit = admitted_task;
                    inflight_admit_slot = sel.slot_index;
                    const bool allow_l2_snapshot =
                        active.empty() && pending_count.load(std::memory_order_acquire) == 0;
                    if (auto slot_obj = admit_request(admitted_task, sel, allow_l2_snapshot)) {
                        active.push_back(std::move(slot_obj));
                    }
                    inflight_admit = nullptr;
                    inflight_admit_slot = slots.size();
                }
            }
        }

        // 3. Evict cancelled slots before running the next batch step.
        for (auto it = active.begin(); it != active.end();) {
            if ((*it)->IsCancelled()) {
                finish_slot(**it, StopReason::kCancelled);
                it = active.erase(it);
            } else {
                ++it;
            }
        }
        if (active.empty()) {
            if (worker_stop.load(std::memory_order_acquire) && pending_count.load(std::memory_order_acquire) == 0) {
                break;
            }
            continue;
        }

        // 4. Case A: Zero-overhead single-request fast path when exactly 1
        // request is active and no other request is waiting in the queue.
        if (CanUseSingleRequestFastPath(active.size(), pending_count.load(std::memory_order_acquire), false)) {
            ActiveBatchSlot& s = *active[0];
            const GenerateRequest& req = *s.task->req;

            abort_requested.store(false, std::memory_order_relaxed);
            active_cancel = nullptr;
            batch_abort_fn = [&]() -> bool {
                return worker_stop.load(std::memory_order_relaxed) || s.IsCancelled();
            };

            if (s.phase == BatchSlotPhase::kPrefill) {
                const auto n_prompt = static_cast<int64_t>(req.prompt.size());
                const auto off = static_cast<int64_t>(s.prefill_offset);
                int64_t step_end = std::min<int64_t>(n_prompt, off + static_cast<int64_t>(step_token_budget));
                if (s.hybrid_ckpt_offset > off && step_end > s.hybrid_ckpt_offset) {
                    step_end = s.hybrid_ckpt_offset;
                }
                const auto n = static_cast<int32_t>(step_end - off);
                BatchClear(batch);
                for (int32_t j = 0; j < n; ++j) {
                    const bool is_last = (off + j + 1 == n_prompt);
                    BatchAdd(batch, req.prompt[static_cast<std::size_t>(off + j)],
                             static_cast<llama_pos>(s.pos + j), s.slot_id, is_last);
                }
                int32_t rc = llama_decode(ctx, batch);
                while (rc == 1 && evict_one_idle_slot()) {
                    abort_requested.store(false, std::memory_order_relaxed);
                    rc = llama_decode(ctx, batch);
                }
                batch_abort_fn = nullptr;
                active_cancel = nullptr;
                if (rc != 0) {
                    if (!llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(s.pos), -1)) {
                        invalidate_slot(s);
                    }
                    if (rc == 2) {
                        finish_slot(s, StopReason::kCancelled);
                    } else if (rc == 1) {
                        finish_slot(s, StopReason::kContextFull);
                    } else {
                        s.result.status = Status::Error(
                            ErrorCode::kDecodeFailed, "llama_decode failed during prefill (rc=" + std::to_string(rc) + ")");
                        finish_slot(s, StopReason::kError);
                    }
                    active.clear();
                    continue;
                }
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (s.mtp_enabled && !MtpProcessPrefill(batch, s.slot_id)) {
                    invalidate_slot(s);
                    if (s.IsCancelled()) {
                        finish_slot(s, StopReason::kCancelled);
                    } else {
                        s.result.status = Status::Error(ErrorCode::kDecodeFailed, "MTP draft context prefill sync failed");
                        finish_slot(s, StopReason::kError);
                    }
                    active.clear();
                    continue;
                }
#endif
                if (s.slot_idx < slots.size()) {
                    auto& stoks = slots[s.slot_idx].tokens;
                    stoks.insert(stoks.end(), req.prompt.begin() + off, req.prompt.begin() + step_end);
                }
                s.result.prompt_tokens += n;
                s.pos += n;
                s.prefill_offset = static_cast<std::size_t>(step_end);
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                if (s.hybrid_ckpt_offset == step_end && s.slot_valid) {
                    CapturePartialCheckpoint(s.slot_idx, s.slot_id, static_cast<std::size_t>(s.pos));
                }
#endif
                if (step_end == n_prompt) {
                    s.prefill_done = true;
                    s.result.prefill_ms = MsSince(s.prefill_start);
                    Emit(TelemetryKind::kPrefillDone, s.result.prompt_tokens, s.result.prefill_ms);
                    s.decode_start = Clock::now();
                    if (req.max_new_tokens == 0) {
                        finish_slot(s, StopReason::kMaxTokens);
                        active.clear();
                        continue;
                    }
                    const auto first_start = Clock::now();
                    TokenLogprobInfo first_lp;
                    const Token first_tok = SampleTokenAt(
                        batch.n_tokens - 1, req.sampling, s.sampler.get(), s.grammar_sampler.get(),
                        s.target_gpu_sampled, n_vocab, req.sampling.compute_logprobs ? &first_lp : nullptr);
                    if (first_tok == LLAMA_TOKEN_NULL) {
                        s.result.status =
                            Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
                        finish_slot(s, StopReason::kError);
                        active.clear();
                        continue;
                    }
                    if (llama_vocab_is_eog(vocab, first_tok)) {
                        finish_slot(s, StopReason::kEndOfGeneration);
                        active.clear();
                        continue;
                    }
                    bool keep_going = true;
                    if (Status st = emit_slot_token(s, first_tok, first_lp, keep_going); !st.ok()) {
                        s.result.status = std::move(st);
                        finish_slot(s, StopReason::kError);
                        active.clear();
                        continue;
                    }
                    if (!keep_going) {
                        finish_slot(s, StopReason::kCancelled);
                        active.clear();
                        continue;
                    }
                    Emit(TelemetryKind::kTokenDecoded, 1, MsSince(first_start));
                    if (s.result.generated_tokens >= req.max_new_tokens) {
                        finish_slot(s, StopReason::kMaxTokens);
                        active.clear();
                        continue;
                    }
                    if (s.pos >= n_ctx) {
                        finish_slot(s, StopReason::kContextFull);
                        active.clear();
                        continue;
                    }
                    s.cur_tok = first_tok;
                    s.phase = BatchSlotPhase::kDecode;
                }
                continue;
            }

            // Single-request fast-path decode step (with full speculation!).
            if (!s.spec_initialized) {
                s.spec_initialized = true;
                if ((s.suffix_enabled || s.hybrid_suffix_enabled) && suffix_matcher != nullptr) {
                    suffix_matcher->BeginRequest(req.prompt);
                }
                if (s.ngram_mod_enabled && ngram_mod != nullptr) {
                    ngram_history.assign(req.prompt.begin(), req.prompt.end());
                    ngram_mod->BeginRequest(ngram_history);
                    s.ngram_indexed = ngram_history.size() > ngram_mod->n_match()
                                          ? ngram_history.size() - ngram_mod->n_match()
                                          : 0;
                } else if (s.ngram_short_enabled) {
                    ngram_matcher.Reset();
                    ngram_history.assign(req.prompt.begin(), req.prompt.end());
                    ngram_matcher.IndexRange(ngram_history, 0, ngram_history.size());
                    s.ngram_indexed = ngram_history.size();
                }
            }

            const auto step_start = Clock::now();
            const int64_t rem_gen =
                static_cast<int64_t>(req.max_new_tokens) - static_cast<int64_t>(s.result.generated_tokens);
            const int64_t max_k = std::min<int64_t>(
                {rem_gen - 1, n_ctx - s.pos - 1, static_cast<int64_t>(step_token_budget) - 1});

            auto& draft_tokens = draft_scratch;
            auto& accepted_tokens = accepted_scratch;
            auto& accepted_lps = accepted_logprobs_scratch;
            draft_tokens.clear();
            bool step_used_suffix = false;

            if (max_k > 0) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (s.mtp_enabled) {
                    if (s.hybrid_suffix_enabled && suffix_matcher != nullptr &&
                        (architecture != sonder::inference::ModelArchitecture::hybrid || can_hybrid_spec_ckpt ||
                         rs_cap >= SuffixMatcher::kHybridMtpMinBurstDraft)) {
                        if (suffix_matcher->paused()) {
                            suffix_matcher->NoteSkippedStep();
                        } else {
                            const std::size_t hybrid_cap =
                                (architecture == sonder::inference::ModelArchitecture::hybrid && !can_hybrid_spec_ckpt)
                                    ? std::min<std::size_t>(static_cast<std::size_t>(max_k), static_cast<std::size_t>(rs_cap))
                                    : static_cast<std::size_t>(max_k);
                            SuffixMatchResult burst = suffix_matcher->TryHybridMtpBurstForCurrent(s.cur_tok, hybrid_cap);
                            if (!burst.draft.empty()) {
                                draft_tokens = std::move(burst.draft);
                                for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                                    if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                                        draft_tokens.resize(j + 1);
                                        break;
                                    }
                                }
                                step_used_suffix = !draft_tokens.empty();
                            }
                        }
                    }
                    if (!step_used_suffix) {
                        const auto k_lim = static_cast<int32_t>(std::min<int64_t>(max_k, s.effective_draft_max));
                        if (!MtpDraft(s.slot_id, s.cur_tok, s.pos, k_lim, draft_tokens)) {
                            active_cancel = nullptr;
                            if (s.IsCancelled()) {
                                finish_slot(s, StopReason::kCancelled);
                                active.clear();
                                continue;
                            }
                            s.mtp_enabled = false;
                            s.hybrid_suffix_enabled = false;
                            s.slot_valid = false;
                            if (s.slot_idx < slots.size()) {
                                slots[s.slot_idx].tokens.clear();
                                slots[s.slot_idx].session_key.clear();
                            }
                            if (s.slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s.slot_idx] = false;
                            ClearSlotCheckpoints(s.slot_idx);
                            draft_tokens.clear();
                        }
                    }
                } else
#endif
                if (s.suffix_enabled && suffix_matcher != nullptr) {
                    if (suffix_matcher->paused()) {
                        suffix_matcher->NoteSkippedStep();
                    } else {
                        const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, s.effective_draft_max));
                        SuffixMatchResult m = suffix_matcher->DraftForCurrent(s.cur_tok, k_lim);
                        if (!m.draft.empty()) {
                            draft_tokens = std::move(m.draft);
                            for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                                if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                                    draft_tokens.resize(j + 1);
                                    break;
                                }
                            }
                            step_used_suffix = !draft_tokens.empty();
                        }
                    }
                } else if (s.ngram_mod_enabled && ngram_mod != nullptr) {
                    const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, s.effective_draft_max));
                    ngram_mod->Draft(ngram_history, s.cur_tok, k_lim, draft_tokens);
                    for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                        if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                            draft_tokens.resize(j + 1);
                            break;
                        }
                    }
                } else if (s.ngram_short_enabled) {
                    const int32_t ctrl_len = s.adaptive_ctrl.NextDraftLength();
                    if (ctrl_len > 0) {
                        ngram_history.push_back(s.cur_tok);
                        const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, ctrl_len));
                        ngram_matcher.Draft(ngram_history, k_lim, draft_tokens);
                        ngram_history.pop_back();
                        for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                            if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                                draft_tokens.resize(j + 1);
                                break;
                            }
                        }
                    } else {
                        s.adaptive_ctrl.NoteSkippedStep();
                    }
                }
            }

            [[maybe_unused]] bool have_spec_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
            if (!draft_tokens.empty() && architecture == sonder::inference::ModelArchitecture::hybrid &&
                draft_tokens.size() > static_cast<std::size_t>(rs_cap)) {
                const llama_state_seq_flags flags = PartialCheckpointFlags();
                const std::size_t sz = flags != 0 ? llama_state_seq_get_size_ext(ctx, s.slot_id, flags) : 0;
                if (sz > 0 && sz <= kMaxPartialCheckpointBytes) {
                    if (spec_ckpt_buf.size() < sz) spec_ckpt_buf.resize(sz);
                    const std::size_t written =
                        llama_state_seq_get_data_ext(ctx, spec_ckpt_buf.data(), sz, s.slot_id, flags);
                    if (written > 0 && written <= sz) {
                        spec_ckpt_size = written;
                        have_spec_ckpt = true;
                    }
                }
                if (!have_spec_ckpt) {
                    draft_tokens.clear();
                    step_used_suffix = false;
                }
            }
#endif

            BatchClear(batch);
            BatchAdd(batch, s.cur_tok, static_cast<llama_pos>(s.pos), s.slot_id, /*logits=*/true);
            for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                BatchAdd(batch, draft_tokens[j], static_cast<llama_pos>(s.pos + 1 + static_cast<int64_t>(j)), s.slot_id,
                         /*logits=*/true);
            }

            int32_t rc = llama_decode(ctx, batch);
            while (rc == 1 && evict_one_idle_slot()) {
                abort_requested.store(false, std::memory_order_relaxed);
                rc = llama_decode(ctx, batch);
            }
            batch_abort_fn = nullptr;
            active_cancel = nullptr;
            if (rc != 0) {
                bool rolled_back = llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(s.pos), -1);
                if (!rolled_back) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                    if (have_spec_ckpt) {
                        const llama_state_seq_flags flags = PartialCheckpointFlags();
                        const std::size_t read = llama_state_seq_set_data_ext(ctx, spec_ckpt_buf.data(), spec_ckpt_size,
                                                                              s.slot_id, flags);
                        rolled_back = (read > 0) && llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(s.pos), -1);
                    }
#endif
                }
                if (!rolled_back) invalidate_slot(s);
                if (rc == 2) {
                    finish_slot(s, StopReason::kCancelled);
                } else if (rc == 1) {
                    finish_slot(s, StopReason::kContextFull);
                } else {
                    s.result.status =
                        Status::Error(ErrorCode::kDecodeFailed, "llama_decode failed (rc=" + std::to_string(rc) + ")");
                    finish_slot(s, StopReason::kError);
                }
                active.clear();
                continue;
            }

            if (draft_tokens.empty()) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (s.mtp_enabled && !MtpCommitVerified(batch, s.slot_id, 0)) {
                    invalidate_slot(s);
                    s.result.status = Status::Error(ErrorCode::kDecodeFailed, "MTP draft context sync failed");
                    finish_slot(s, StopReason::kError);
                    active.clear();
                    continue;
                }
#endif
                if (s.slot_valid && s.slot_idx < slots.size()) slots[s.slot_idx].tokens.push_back(s.cur_tok);
                if ((s.suffix_enabled || s.hybrid_suffix_enabled) && suffix_matcher != nullptr) {
                    suffix_matcher->CommitToken(s.cur_tok);
                }
                if (s.ngram_mod_enabled && ngram_mod != nullptr) {
                    ngram_history.push_back(s.cur_tok);
                    s.ngram_indexed = ngram_mod->IndexRange(ngram_history, s.ngram_indexed, ngram_history.size());
                } else if (s.ngram_short_enabled) {
                    ngram_history.push_back(s.cur_tok);
                    ngram_matcher.IndexRange(ngram_history, s.ngram_indexed, ngram_history.size());
                    s.ngram_indexed = ngram_history.size();
                }
                ++s.pos;

                TokenLogprobInfo step_lp;
                const Token next_tok = SampleTokenAt(0, req.sampling, s.sampler.get(), s.grammar_sampler.get(),
                                                     s.target_gpu_sampled, n_vocab,
                                                     req.sampling.compute_logprobs ? &step_lp : nullptr);
                if (next_tok == LLAMA_TOKEN_NULL) {
                    s.result.status =
                        Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
                    finish_slot(s, StopReason::kError);
                    active.clear();
                    continue;
                }
                if (llama_vocab_is_eog(vocab, next_tok)) {
                    finish_slot(s, StopReason::kEndOfGeneration);
                    active.clear();
                    continue;
                }
                bool keep_going = true;
                if (Status st = emit_slot_token(s, next_tok, step_lp, keep_going); !st.ok()) {
                    s.result.status = std::move(st);
                    finish_slot(s, StopReason::kError);
                    active.clear();
                    continue;
                }
                if (!keep_going) {
                    finish_slot(s, StopReason::kCancelled);
                    active.clear();
                    continue;
                }
                Emit(TelemetryKind::kTokenDecoded, 1, MsSince(step_start));
                if (s.result.generated_tokens >= req.max_new_tokens) {
                    if (s.ngram_mod_enabled) ngram_history.push_back(next_tok);
                    if ((s.suffix_enabled || s.hybrid_suffix_enabled) && suffix_matcher != nullptr) {
                        suffix_matcher->CommitToken(next_tok);
                    }
                    finish_slot(s, StopReason::kMaxTokens);
                    active.clear();
                    continue;
                }
                if (s.pos >= n_ctx) {
                    finish_slot(s, StopReason::kContextFull);
                    active.clear();
                    continue;
                }
                s.cur_tok = next_tok;
                continue;
            }

            // Speculative verification on single-request fast path.
            const std::size_t k = draft_tokens.size();
            const Token step_base_tok = s.cur_tok;
            s.result.draft_tokens += static_cast<std::int32_t>(k);
            accepted_tokens.clear();
            accepted_lps.clear();
            std::size_t n_accepted = 0;
            bool sample_deadlock = false;
            for (std::size_t j = 0; j < k; ++j) {
                TokenLogprobInfo lp;
                const Token sampled = SampleTokenAt(static_cast<int32_t>(j), req.sampling, s.sampler.get(),
                                                    s.grammar_sampler.get(), s.target_gpu_sampled, n_vocab,
                                                    req.sampling.compute_logprobs ? &lp : nullptr);
                if (sampled == LLAMA_TOKEN_NULL) {
                    sample_deadlock = true;
                    break;
                }
                accepted_tokens.push_back(sampled);
                accepted_lps.push_back(std::move(lp));
                if (sampled == draft_tokens[j]) {
                    ++n_accepted;
                } else {
                    break;
                }
            }
            if (!sample_deadlock && n_accepted == k) {
                TokenLogprobInfo bonus_lp;
                const Token bonus = SampleTokenAt(static_cast<int32_t>(k), req.sampling, s.sampler.get(),
                                                  s.grammar_sampler.get(), s.target_gpu_sampled, n_vocab,
                                                  req.sampling.compute_logprobs ? &bonus_lp : nullptr);
                if (bonus == LLAMA_TOKEN_NULL) {
                    sample_deadlock = true;
                } else {
                    accepted_tokens.push_back(bonus);
                    accepted_lps.push_back(std::move(bonus_lp));
                }
            }
            if (sample_deadlock) {
                invalidate_slot(s);
                s.result.status =
                    Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
                finish_slot(s, StopReason::kError);
                active.clear();
                continue;
            }
            s.result.draft_accepted_tokens += static_cast<std::int32_t>(n_accepted);
            if (step_used_suffix && suffix_matcher != nullptr) {
                suffix_matcher->Observe(k, n_accepted);
            } else if (s.ngram_mod_enabled && ngram_mod != nullptr) {
                if (ngram_mod->Observe(k, n_accepted)) s.ngram_indexed = 0;
            } else if (s.ngram_short_enabled) {
                s.adaptive_ctrl.Observe(static_cast<std::int32_t>(k), static_cast<std::int32_t>(n_accepted));
            }

            if (s.slot_valid && s.slot_idx < slots.size()) slots[s.slot_idx].tokens.push_back(step_base_tok);
            if ((s.suffix_enabled || s.hybrid_suffix_enabled) && suffix_matcher != nullptr) {
                suffix_matcher->CommitToken(step_base_tok);
            }
            if (s.ngram_mod_enabled || s.ngram_short_enabled) ngram_history.push_back(step_base_tok);

            int64_t new_pos = s.pos + 1;
            std::size_t committed_drafts = 0;
            auto rollback_rejected = [&](int64_t keep_pos, std::size_t keep_drafts) -> bool {
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (s.mtp_enabled && !MtpCommitVerified(batch, s.slot_id, static_cast<int32_t>(keep_drafts))) {
                    invalidate_slot(s);
                    return false;
                }
#endif
                const int64_t batch_end_pos = s.pos + 1 + static_cast<int64_t>(k);
                if (keep_pos < batch_end_pos) {
                    if (!llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(keep_pos), -1)) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                        if (have_spec_ckpt) {
                            const llama_state_seq_flags flags = PartialCheckpointFlags();
                            const std::size_t read = llama_state_seq_set_data_ext(ctx, spec_ckpt_buf.data(),
                                                                                  spec_ckpt_size, s.slot_id, flags);
                            if (read == 0 || !llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(s.pos), -1)) {
                                invalidate_slot(s);
                                return false;
                            }
                            BatchClear(batch);
                            BatchAdd(batch, step_base_tok, static_cast<llama_pos>(s.pos), s.slot_id, /*logits=*/false);
                            for (std::size_t j = 0; j < keep_drafts; ++j) {
                                BatchAdd(batch, draft_tokens[j],
                                         static_cast<llama_pos>(s.pos + 1 + static_cast<int64_t>(j)), s.slot_id,
                                         /*logits=*/false);
                            }
                            if (llama_decode(ctx, batch) != 0) {
                                invalidate_slot(s);
                                return false;
                            }
                            return true;
                        }
#else
                        (void)step_base_tok;
                        (void)keep_drafts;
#endif
                        invalidate_slot(s);
                        return false;
                    }
                }
                return true;
            };

            bool slot_finished = false;
            for (std::size_t m = 0; m < accepted_tokens.size(); ++m) {
                const Token tok = accepted_tokens[m];
                const bool is_bonus = (m == n_accepted);
                if (llama_vocab_is_eog(vocab, tok)) {
                    (void)rollback_rejected(new_pos, committed_drafts);
                    if (m > 0) {
                        Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m), MsSince(step_start));
                    }
                    finish_slot(s, StopReason::kEndOfGeneration);
                    slot_finished = true;
                    break;
                }
                bool keep_going = true;
                if (Status st = emit_slot_token(s, tok, accepted_lps[m], keep_going); !st.ok()) {
                    (void)rollback_rejected(new_pos, committed_drafts);
                    s.result.status = std::move(st);
                    finish_slot(s, StopReason::kError);
                    slot_finished = true;
                    break;
                }
                if (!is_bonus) {
                    if (s.slot_valid && s.slot_idx < slots.size()) slots[s.slot_idx].tokens.push_back(tok);
                    if ((s.suffix_enabled || s.hybrid_suffix_enabled) && suffix_matcher != nullptr) {
                        suffix_matcher->CommitToken(tok);
                    }
                    if (s.ngram_mod_enabled || s.ngram_short_enabled) ngram_history.push_back(tok);
                    ++new_pos;
                    ++committed_drafts;
                }
                if (!keep_going) {
                    (void)rollback_rejected(new_pos, committed_drafts);
                    finish_slot(s, StopReason::kCancelled);
                    slot_finished = true;
                    break;
                }
                if (s.result.generated_tokens >= req.max_new_tokens) {
                    if (is_bonus && s.ngram_mod_enabled) ngram_history.push_back(tok);
                    if (is_bonus && (s.suffix_enabled || s.hybrid_suffix_enabled) && suffix_matcher != nullptr) {
                        suffix_matcher->CommitToken(tok);
                    }
                    (void)rollback_rejected(new_pos, committed_drafts);
                    Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m + 1), MsSince(step_start));
                    finish_slot(s, StopReason::kMaxTokens);
                    slot_finished = true;
                    break;
                }
                if (is_bonus) {
                    s.cur_tok = tok;
                }
            }
            if (slot_finished) {
                active.clear();
                continue;
            }
            if (!rollback_rejected(new_pos, committed_drafts)) {
                if (s.IsCancelled()) {
                    finish_slot(s, StopReason::kCancelled);
                } else {
                    s.result.status =
                        Status::Error(ErrorCode::kDecodeFailed, "failed to trim rejected speculative KV tail");
                    finish_slot(s, StopReason::kError);
                }
                active.clear();
                continue;
            }
            if (s.ngram_mod_enabled && ngram_mod != nullptr) {
                s.ngram_indexed = ngram_mod->IndexRange(ngram_history, s.ngram_indexed, ngram_history.size());
            } else if (s.ngram_short_enabled) {
                ngram_matcher.IndexRange(ngram_history, s.ngram_indexed, ngram_history.size());
                s.ngram_indexed = ngram_history.size();
            }
            Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(accepted_tokens.size()), MsSince(step_start));
            s.pos = new_pos;
            if (s.pos >= n_ctx) {
                finish_slot(s, StopReason::kContextFull);
                active.clear();
            }
            continue;
        }

        // 5. Case B: True Continuous Batching + Chunked Prefill across multiple
        // concurrent slots in lockstep (1 decode token per decoding slot +
        // chunked prefill <= n_ubatch packed into ONE llama_decode call!).
        // Keep per-seq_id GPU backend samplers attached (never detach mid-request).
        for (auto& sp : active) {
            if (sp->spec_initialized && (sp->suffix_enabled || sp->hybrid_suffix_enabled) &&
                suffix_matcher != nullptr) {
                suffix_matcher->ClearActive();
            }
            if (sp->mtp_enabled) {
                sp->mtp_enabled = false;
                sp->hybrid_suffix_enabled = false;
                if (sp->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[sp->slot_idx] = false;
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (sp->slot_idx < mtp_deferred.size()) mtp_deferred[sp->slot_idx].Clear();
#endif
            }
            sp->suffix_enabled = false;
            sp->ngram_mod_enabled = false;
            sp->ngram_short_enabled = false;
            sp->batch_logit_idx = -1;
        }

        views.clear();
        std::size_t decoding_count = 0;
        for (const auto& sp : active) {
            if (!sp->IsCancelled() && sp->phase == BatchSlotPhase::kDecode) {
                ++decoding_count;
            }
            views.push_back(BatchSlotPlanView{
                sp->slot_idx,
                static_cast<std::int32_t>(sp->slot_id),
                sp->phase,
                sp->task->req->prompt.size(),
                sp->prefill_offset,
                sp->pos,
                sp->IsCancelled(),
            });
        }

        // Prioritize low inter-token latency when any slot is decoding (cap
        // mixed prefill chunks to <= 256 tokens per step), while letting pure
        // prefill waves utilize the full `n_batch` window in one GPU pass.
        const int32_t case_b_budget =
            decoding_count > 0
                ? std::min<int32_t>(step_token_budget, static_cast<int32_t>(decoding_count) + 256)
                : n_batch;
        PlannedBatchStep plan = PlanContinuousBatchStep(views, case_b_budget, step_output_budget, rr_cursor);
        if (plan.items.empty()) {
            continue;
        }

        auto find_active_by_slot = [&](std::size_t slot_index) -> ActiveBatchSlot* {
            for (auto& sp : active) {
                if (sp->slot_idx == slot_index) return sp.get();
            }
            return nullptr;
        };

        auto pack_plan = [&](const PlannedBatchStep& p) {
            BatchClear(batch);
            for (auto& sp : active) sp->batch_logit_idx = -1;
            for (const PlannedBatchItem& item : p.items) {
                ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                if (s == nullptr) continue;
                if (item.is_prefill) {
                    const auto& prompt = s->task->req->prompt;
                    for (int32_t j = 0; j < item.token_count; ++j) {
                        const bool want_logits = item.request_logits && (j + 1 == item.token_count);
                        BatchAdd(batch, prompt[item.token_offset + static_cast<std::size_t>(j)],
                                 static_cast<llama_pos>(item.pos_start + j), s->slot_id, want_logits);
                        if (want_logits) s->batch_logit_idx = batch.n_tokens - 1;
                    }
                } else {
                    BatchAdd(batch, s->cur_tok, static_cast<llama_pos>(item.pos_start), s->slot_id, /*logits=*/true);
                    s->batch_logit_idx = batch.n_tokens - 1;
                }
            }
        };

        pack_plan(plan);

        abort_requested.store(false, std::memory_order_relaxed);
        active_cancel = nullptr;
        batch_abort_fn = [&]() -> bool {
            if (worker_stop.load(std::memory_order_relaxed)) return true;
            for (const auto& sp : active) {
                if (!sp->IsCancelled()) return false;
            }
            return true;
        };

        const auto step_start = Clock::now();
        int32_t rc = llama_decode(ctx, batch);

        // Under unified KV (`kv_unified=true`), idle prefix-cached slots share
        // the same n_ctx pool; evict LRU idle slots before retrying on rc==1.
        while (rc == 1 && evict_one_idle_slot()) {
            abort_requested.store(false, std::memory_order_relaxed);
            rc = llama_decode(ctx, batch);
        }

        // If a mixed prefill+decode batch still returned rc==1 (no KV slot for
        // full batch), retry decode-only tokens so active decoders complete and free KV.
        if (rc == 1 && plan.prefill_slots > 0 && plan.decode_slots > 0) {
            PlannedBatchStep decode_only;
            for (const PlannedBatchItem& item : plan.items) {
                if (!item.is_prefill) {
                    decode_only.items.push_back(item);
                    decode_only.total_tokens += item.token_count;
                    decode_only.total_outputs += 1;
                    decode_only.decode_slots += 1;
                }
            }
            plan = std::move(decode_only);
            pack_plan(plan);
            abort_requested.store(false, std::memory_order_relaxed);
            rc = llama_decode(ctx, batch);
        }
        batch_abort_fn = nullptr;

        if (rc != 0) {
            const bool stopping = worker_stop.load(std::memory_order_acquire);
            for (const PlannedBatchItem& item : plan.items) {
                ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                if (s == nullptr) continue;
                if (!llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(s->pos), -1)) {
                    invalidate_slot(*s);
                }
                if (rc == 2 && (s->IsCancelled() || stopping)) {
                    finish_slot(*s, StopReason::kCancelled);
                } else if (rc == 1) {
                    finish_slot(*s, StopReason::kContextFull);
                } else if (rc != 2) {
                    s->result.status =
                        Status::Error(ErrorCode::kDecodeFailed, "llama_decode failed (rc=" + std::to_string(rc) + ")");
                    finish_slot(*s, StopReason::kError);
                }
            }
            active.erase(std::remove_if(active.begin(), active.end(),
                                        [](const auto& sp) { return sp->task == nullptr; }),
                         active.end());
            continue;
        }

        // Extract every slot's sampled token + optional logprobs from this step's
        // logits BEFORE invoking user callbacks or mutating context state.
        // Keep `outcomes` strictly 1:1 aligned with `plan.items` so indexing
        // `outcomes[idx]` can never go out of bounds.
        outcomes.clear();
        for (const PlannedBatchItem& item : plan.items) {
            SampledStepOutcome out;
            out.slot_index = item.slot_index;
            ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
            if (s != nullptr && s->task != nullptr && s->batch_logit_idx >= 0) {
                out.has_sample = true;
                out.tok = SampleTokenAt(s->batch_logit_idx, s->task->req->sampling, s->sampler.get(),
                                        s->grammar_sampler.get(), s->target_gpu_sampled, n_vocab,
                                        s->task->req->sampling.compute_logprobs ? &out.lp : nullptr);
            }
            outcomes.push_back(std::move(out));
        }

        // Advance each planned slot and stream tokens to callbacks.
        const double step_ms = MsSince(step_start);
        for (std::size_t idx = 0; idx < plan.items.size(); ++idx) {
            const PlannedBatchItem& item = plan.items[idx];
            ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
            if (s == nullptr || s->task == nullptr) continue;
            const GenerateRequest& req = *s->task->req;

            if (item.is_prefill) {
                if (s->slot_valid && s->slot_idx < slots.size()) {
                    auto& stoks = slots[s->slot_idx].tokens;
                    stoks.insert(stoks.end(),
                                 req.prompt.begin() + static_cast<std::ptrdiff_t>(item.token_offset),
                                 req.prompt.begin() +
                                     static_cast<std::ptrdiff_t>(item.token_offset + static_cast<std::size_t>(item.token_count)));
                }
                s->prefill_offset += static_cast<std::size_t>(item.token_count);
                s->pos += item.token_count;
                s->result.prompt_tokens += item.token_count;

                if (s->prefill_offset >= req.prompt.size()) {
                    s->prefill_done = true;
                    s->result.prefill_ms = MsSince(s->prefill_start);
                    Emit(TelemetryKind::kPrefillDone, s->result.prompt_tokens, s->result.prefill_ms);
                    s->decode_start = Clock::now();
                    if (req.max_new_tokens == 0) {
                        finish_slot(*s, StopReason::kMaxTokens);
                        continue;
                    }
                    if (!outcomes[idx].has_sample || outcomes[idx].tok == LLAMA_TOKEN_NULL) {
                        s->result.status = Status::Error(ErrorCode::kDecodeFailed,
                                                         "sampler produced no valid token (grammar deadlock)");
                        finish_slot(*s, StopReason::kError);
                        continue;
                    }
                    const Token first_tok = outcomes[idx].tok;
                    if (llama_vocab_is_eog(vocab, first_tok)) {
                        finish_slot(*s, StopReason::kEndOfGeneration);
                        continue;
                    }
                    bool keep_going = true;
                    if (Status st = emit_slot_token(*s, first_tok, outcomes[idx].lp, keep_going); !st.ok()) {
                        s->result.status = std::move(st);
                        finish_slot(*s, StopReason::kError);
                        continue;
                    }
                    if (!keep_going || s->IsCancelled()) {
                        finish_slot(*s, StopReason::kCancelled);
                        continue;
                    }
                    Emit(TelemetryKind::kTokenDecoded, 1, step_ms);
                    if (s->result.generated_tokens >= req.max_new_tokens) {
                        finish_slot(*s, StopReason::kMaxTokens);
                        continue;
                    }
                    if (s->pos >= n_ctx) {
                        finish_slot(*s, StopReason::kContextFull);
                        continue;
                    }
                    s->cur_tok = first_tok;
                    s->phase = BatchSlotPhase::kDecode;
                }
            } else {
                if (s->slot_valid && s->slot_idx < slots.size()) {
                    slots[s->slot_idx].tokens.push_back(s->cur_tok);
                }
                s->pos += 1;

                if (!outcomes[idx].has_sample || outcomes[idx].tok == LLAMA_TOKEN_NULL) {
                    s->result.status =
                        Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
                    finish_slot(*s, StopReason::kError);
                    continue;
                }
                const Token next_tok = outcomes[idx].tok;
                if (llama_vocab_is_eog(vocab, next_tok)) {
                    finish_slot(*s, StopReason::kEndOfGeneration);
                    continue;
                }
                bool keep_going = true;
                if (Status st = emit_slot_token(*s, next_tok, outcomes[idx].lp, keep_going); !st.ok()) {
                    s->result.status = std::move(st);
                    finish_slot(*s, StopReason::kError);
                    continue;
                }
                if (!keep_going || s->IsCancelled()) {
                    finish_slot(*s, StopReason::kCancelled);
                    continue;
                }
                Emit(TelemetryKind::kTokenDecoded, 1, step_ms);
                if (s->result.generated_tokens >= req.max_new_tokens) {
                    finish_slot(*s, StopReason::kMaxTokens);
                    continue;
                }
                if (s->pos >= n_ctx) {
                    finish_slot(*s, StopReason::kContextFull);
                    continue;
                }
                s->cur_tok = next_tok;
            }
        }

        active.erase(std::remove_if(active.begin(), active.end(),
                                    [](const auto& sp) { return sp->task == nullptr; }),
                     active.end());
    }
    } catch (const std::exception& ex) {
        batch_abort_fn = nullptr;
        active_cancel = nullptr;
        const std::string err_msg = std::string("batch worker error: ") + ex.what();
        if (inflight_exclusive != nullptr) {
            GenerateResult res;
            res.status = Status::Error(ErrorCode::kDecodeFailed, err_msg);
            res.stop_reason = StopReason::kError;
            complete_task(inflight_exclusive, std::move(res));
            inflight_exclusive = nullptr;
        }
        if (inflight_admit != nullptr) {
            if (inflight_admit_slot < slot_busy.size()) slot_busy[inflight_admit_slot] = 0u;
            GenerateResult res;
            res.status = Status::Error(ErrorCode::kDecodeFailed, err_msg);
            res.stop_reason = StopReason::kError;
            complete_task(inflight_admit, std::move(res));
            inflight_admit = nullptr;
        }
        for (auto& sp : active) {
            if (sp && sp->task != nullptr) {
                sp->result.status = Status::Error(ErrorCode::kDecodeFailed, err_msg);
                finish_slot(*sp, StopReason::kError);
            }
        }
        active.clear();
        std::fill(slot_busy.begin(), slot_busy.end(), std::uint8_t{0});
    } catch (...) {
        batch_abort_fn = nullptr;
        active_cancel = nullptr;
        if (inflight_exclusive != nullptr) {
            GenerateResult res;
            res.status = Status::Error(ErrorCode::kDecodeFailed, "batch worker internal exception");
            res.stop_reason = StopReason::kError;
            complete_task(inflight_exclusive, std::move(res));
            inflight_exclusive = nullptr;
        }
        if (inflight_admit != nullptr) {
            if (inflight_admit_slot < slot_busy.size()) slot_busy[inflight_admit_slot] = 0u;
            GenerateResult res;
            res.status = Status::Error(ErrorCode::kDecodeFailed, "batch worker internal exception");
            res.stop_reason = StopReason::kError;
            complete_task(inflight_admit, std::move(res));
            inflight_admit = nullptr;
        }
        for (auto& sp : active) {
            if (sp && sp->task != nullptr) {
                sp->result.status = Status::Error(ErrorCode::kDecodeFailed, "batch worker internal exception");
                finish_slot(*sp, StopReason::kError);
            }
        }
        active.clear();
        std::fill(slot_busy.begin(), slot_busy.end(), std::uint8_t{0});
    }

    // Mark stopped under `queue_mu` BEFORE draining so no new caller can enqueue
    // onto an exited worker thread after an abnormal termination.
    std::deque<PendingRequest*> leftover;
    {
        std::lock_guard<std::mutex> qlock(queue_mu);
        worker_stop.store(true, std::memory_order_release);
        leftover.swap(pending_queue);
        pending_count.store(0, std::memory_order_relaxed);
    }
    for (PendingRequest* task : leftover) {
        GenerateResult res;
        res.status = Status::Error(ErrorCode::kNotLoaded, "backend unloaded before request started");
        res.stop_reason = StopReason::kError;
        complete_task(task, std::move(res));
    }
}

}  // namespace sonder::backends::llamacpp
