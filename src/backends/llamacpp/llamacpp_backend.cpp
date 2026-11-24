// Sonder Inference - llama.cpp backend implementation.
#include "sonder/backends/llamacpp/llamacpp_backend.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <regex>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "ggml-backend.h"
#include "llama.h"

#if defined(SONDER_LLAMACPP_HAS_MTP)
#include "llama-ext.h"
#endif

#ifndef SONDER_LLAMACPP_TAG_STRING
#define SONDER_LLAMACPP_TAG_STRING "unknown"
#endif

namespace sonder::backends::llamacpp {
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

// llama.cpp logs verbosely to stderr by default. Keep warnings and errors;
// set SONDER_LLAMACPP_VERBOSE=1 to see everything.
void LogCallback(ggml_log_level level, const char* text, void* /*user*/) {
    static const bool verbose = [] {
        const char* v = std::getenv("SONDER_LLAMACPP_VERBOSE");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (verbose || level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR) {
        std::fputs(text, stderr);
    }
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
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ) || defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
    uint32_t planned_draft_max = 0;
    {
        SpeculativeType spec = SpeculativeType::kNone;
        if (ParseSpeculativeType(options.spec_type, spec) && spec != SpeculativeType::kNone) {
            if (options.spec_draft_max > 0) {
                planned_draft_max = static_cast<uint32_t>(options.spec_draft_max);
            } else if (spec == SpeculativeType::kDraftMtp) {
                planned_draft_max = static_cast<uint32_t>(kDefaultMtpDraftMax);
            } else if (spec == SpeculativeType::kNgramMod || spec == SpeculativeType::kAuto) {
                planned_draft_max = static_cast<uint32_t>(kDefaultNgramModDraftMax);
            } else {
                planned_draft_max = static_cast<uint32_t>(kDefaultNgramDraftMax);
            }
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ)
            // Only allocate per-step GPU recurrent rollback slots (n_rs_seq) for
            // short-horizon draft-model speculation (draft-mtp / auto). Long-horizon
            // n-gram speculation uses zero GPU rollback slots (saving ~1.6 GiB VRAM
            // on 27B hybrid models) and relies on on-demand PARTIAL_ONLY checkpoints.
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
#endif
#if defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
    {
        const uint32_t per_seq = std::max<uint32_t>(1u, planned_draft_max + 1u);
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
        std::vector<std::uint8_t> tgt_state;
        std::vector<float> pending_h;
    };
    static constexpr std::size_t kMaxCheckpointsPerSlot = 3;
    static constexpr std::size_t kMaxPartialCheckpointBytes = std::size_t{256} << 20;

    std::vector<PrefixSlotMeta> slots;
    std::vector<std::vector<PartialCheckpoint>> slot_checkpoints;
    std::vector<std::uint8_t> spec_ckpt_buf;
    std::size_t spec_ckpt_size = 0;
    std::uint64_t slot_clock = 0;

    // Isolated lazy embedding context so Embed() never disturbs live KV prefix
    // caches or MTP/backend-sampler graph reservations on `ctx`.
    llama_context* embed_ctx = nullptr;
    llama_batch embed_batch{};
    int32_t embed_batch_cap = 0;
    bool embed_batch_allocated = false;

    // Resolved in-process speculative decoding state.
    SpeculativeType active_spec = SpeculativeType::kNone;
    std::int32_t active_draft_max = 0;
    NgramMatcher ngram_matcher{false};
    std::unique_ptr<NgramModTable> ngram_mod;
    std::vector<Token> draft_scratch;
    std::vector<Token> accepted_scratch;
    std::vector<Token> ngram_history;

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

    void ResetAllSlots() noexcept {
        for (std::size_t i = 0; i < slots.size(); ++i) {
            slots[i].tokens.clear();
            slots[i].session_key.clear();
            slots[i].last_used = 0;
        }
        for (auto& ckpts : slot_checkpoints) {
            ckpts.clear();
        }
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
        ReleaseMtp();
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
        slot_checkpoints.clear();
        spec_ckpt_buf.clear();
        spec_ckpt_size = 0;
        slot_clock = 0;
        active_spec = SpeculativeType::kNone;
        active_draft_max = 0;
        ngram_mod.reset();
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
        PartialCheckpoint ckpt;
        ckpt.n_tokens = n_tokens;
        ckpt.tgt_state.resize(sz);
        const std::size_t written = llama_state_seq_get_data_ext(ctx, ckpt.tgt_state.data(), sz, slot_id, flags);
        if (written == 0) return;
        ckpt.tgt_state.resize(written);
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr && slot_idx < mtp_pending_h.size()) {
            ckpt.pending_h = mtp_pending_h[slot_idx];
        }
#endif
        auto& vec = slot_checkpoints[slot_idx];
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [n_tokens](const PartialCheckpoint& c) { return c.n_tokens >= n_tokens; }),
                  vec.end());
        vec.push_back(std::move(ckpt));
        if (vec.size() > kMaxCheckpointsPerSlot) {
            vec.erase(vec.begin());
        }
    }

    bool RestorePartialCheckpoint(std::size_t slot_idx, llama_seq_id slot_id, std::size_t max_lcp,
                                  std::size_t& restored_tokens, bool& restored_pending_h) {
        restored_tokens = 0;
        restored_pending_h = false;
        if (ctx == nullptr || slot_idx >= slot_checkpoints.size()) return false;
        const llama_state_seq_flags flags = PartialCheckpointFlags();
        if (flags == 0) return false;
        auto& vec = slot_checkpoints[slot_idx];
        const PartialCheckpoint* best = nullptr;
        for (const auto& c : vec) {
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
        const std::size_t keep_pos = restored_tokens;
        vec.erase(std::remove_if(vec.begin(), vec.end(),
                                 [keep_pos](const PartialCheckpoint& c) { return c.n_tokens > keep_pos; }),
                  vec.end());
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
        return false;
    }

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
    void MtpDraft(llama_seq_id slot_id, Token id_last, int64_t n_past, int32_t max_draft,
                  std::vector<Token>& draft_out) {
        draft_out.clear();
        if (ctx_dft == nullptr || max_draft <= 0) return;
        const auto sidx = static_cast<std::size_t>(slot_id);
        if (sidx >= slots.size()) return;

        const int32_t limit = mtp_chain_heads ? std::min(max_draft, mtp_n_layers) : max_draft;
        const std::size_t row_bytes = static_cast<std::size_t>(mtp_n_embd) * sizeof(float);

        bool fused_catchup = false;
        auto& d = mtp_deferred[sidx];
        if (d.pending && !mtp_mem_shared) {
            const bool leads_in = !d.pos.empty() && (d.pos.back() + 1 == static_cast<llama_pos>(n_past)) &&
                                  (static_cast<int32_t>(d.tokens.size()) + 1 <= dft_batch_cap);
            if (!leads_in) {
                if (!MtpFlushDeferred(slot_id)) return;
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
            }
        }
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

LlamaCppBackend::LlamaCppBackend() : impl_(std::make_unique<Impl>()) {}
LlamaCppBackend::~LlamaCppBackend() = default;
LlamaCppBackend::LlamaCppBackend(LlamaCppBackend&&) noexcept = default;
LlamaCppBackend& LlamaCppBackend::operator=(LlamaCppBackend&&) noexcept = default;

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

        // Initialize per-slot KV prefix cache metadata.
        const std::uint32_t n_slots = std::clamp<uint32_t>(options.n_slots, 1, kMaxSlots);
        impl_->slots.resize(n_slots);
        impl_->slot_checkpoints.assign(n_slots, {});
        for (std::uint32_t i = 0; i < n_slots; ++i) {
            impl_->slots[i].id = static_cast<std::int32_t>(i);
            impl_->slots[i].tokens.clear();
            impl_->slots[i].session_key.clear();
            impl_->slots[i].last_used = 0;
        }
        impl_->slot_clock = 0;

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

ModelInfo LlamaCppBackend::GetModelInfo() const {
    ModelInfo info;
    if (!IsLoaded()) return info;
    char buf[256] = {};
    if (llama_model_desc(impl_->model, buf, sizeof(buf)) > 0) info.description = buf;
    info.size_bytes = llama_model_size(impl_->model);
    info.n_params = llama_model_n_params(impl_->model);
    info.n_vocab = llama_vocab_n_tokens(impl_->vocab);
    info.n_ctx = impl_->ctx != nullptr ? llama_n_ctx(impl_->ctx) : 0;
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
    const int32_t n_embd = llama_model_n_embd(impl_->model);
    if (n_embd <= 0) {
        return Status::Error(ErrorCode::kDecodeFailed, "model reports invalid embedding dimension");
    }

    const uint32_t base_ctx = std::max<uint32_t>(256u, llama_n_ctx(impl_->ctx));
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

    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    for (Token t : request.prompt) {
        if (t < 0 || t >= n_vocab) return fail(ErrorCode::kInvalidArgument, "prompt token out of range");
    }
    for (const auto& entry : request.sampling.logit_bias) {
        if (entry.first < 0 || entry.first >= n_vocab) {
            return fail(ErrorCode::kInvalidArgument, "logit_bias token out of range");
        }
    }

    llama_context* ctx = impl_->ctx;
    llama_memory_t mem = llama_get_memory(ctx);
    const auto n_ctx = static_cast<int64_t>(llama_n_ctx(ctx));
    const auto n_prompt = static_cast<int64_t>(request.prompt.size());

    // Build and validate the sampler chain (and separate GBNF grammar for fast
    // grammar-based rejection sampling) upfront so an invalid grammar fails fast
    // without wasting prefill compute.
    SamplerPtr sampler;
    SamplerPtr grammar_sampler;
    if (Status s = BuildSampler(request.sampling, impl_->vocab, n_vocab, static_cast<int32_t>(n_ctx), sampler,
                                grammar_sampler);
        !s.ok()) {
        return fail(s.code, s.message);
    }

    // ---- Slot selection, recurrent checkpoint restore, and prefix cache reuse ----
    std::size_t slot_idx = 0;
    llama_seq_id slot_id = 0;
    int64_t prefill_start_offset = 0;
    const bool has_restore = !request.restore_state.empty() && request.restore_tokens > 0 &&
                             request.restore_tokens < request.prompt.size();
    const bool has_capture = request.capture_at_tokens > 0 && request.capture_at_tokens <= request.prompt.size();

    if (has_restore) {
        // Restore atomically into a freshly cleared sequence 0, then prefill
        // only the suffix [restore_tokens .. prompt.size()) so logits are fresh
        // before sampling begins.
        llama_memory_clear(mem, true);
        impl_->ResetAllSlots();
        Status rst = RestoreSequenceState(request.restore_state, request.max_capture_bytes);
        if (rst.ok()) {
            const int64_t restored_pos = llama_memory_seq_pos_max(mem, 0) + 1;
            if (restored_pos == static_cast<int64_t>(request.restore_tokens)) {
                result.state_restored = true;
                prefill_start_offset = static_cast<int64_t>(request.restore_tokens);
                if (!impl_->slots.empty()) {
                    impl_->slots[0].tokens.assign(request.prompt.begin(),
                                                  request.prompt.begin() + prefill_start_offset);
                    impl_->slots[0].session_key = request.session_key;
                    impl_->slots[0].last_used = ++impl_->slot_clock;
                }
                if (request.on_state_restored) {
                    request.on_state_restored();
                }
            } else {
                llama_memory_clear(mem, true);
                impl_->ResetAllSlots();
                result.restore_status = Status::Error(
                    ErrorCode::kDecodeFailed, "restored sequence position (" + std::to_string(restored_pos) +
                                                  ") does not match expected token_position (" +
                                                  std::to_string(request.restore_tokens) + ")");
            }
        } else {
            llama_memory_clear(mem, true);
            impl_->ResetAllSlots();
            result.restore_status = std::move(rst);
        }
    } else if (impl_->options.prefix_cache && request.reset_context && !has_capture && !impl_->slots.empty()) {
        const SlotSelection sel = SelectPrefixSlot(impl_->slots, request.prompt, request.session_key);
        slot_idx = sel.slot_index;
        slot_id = static_cast<llama_seq_id>(impl_->slots[slot_idx].id);
        std::size_t lcp = sel.matched_prefix;

        const int64_t kv_len_raw = llama_memory_seq_pos_max(mem, slot_id) + 1;
        const std::size_t kv_len = kv_len_raw > 0 ? static_cast<std::size_t>(kv_len_raw) : 0;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        const std::size_t prev_slot_len = impl_->slots[slot_idx].tokens.size();
#endif
        lcp = std::min(lcp, kv_len);

        if (lcp == 0) {
            llama_memory_seq_rm(mem, slot_id, -1, -1);
            if (slot_idx < impl_->slot_checkpoints.size()) {
                impl_->slot_checkpoints[slot_idx].clear();
            }
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (impl_->ctx_dft != nullptr) {
                if (!impl_->mtp_mem_shared) {
                    llama_memory_seq_rm(llama_get_memory(impl_->ctx_dft), slot_id, -1, -1);
                }
                std::fill(impl_->mtp_pending_h[slot_idx].begin(), impl_->mtp_pending_h[slot_idx].end(), 0.0F);
                impl_->mtp_deferred[slot_idx].Clear();
            }
#endif
        } else {
            bool restored_pending_h = false;
            if (lcp < kv_len) {
                if (!llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(lcp), -1)) {
                    // Model memory does not support arbitrary partial suffix removal
                    // (e.g. hybrid recurrent/SSM beyond n_rs_seq): attempt fast partial
                    // checkpoint restore before falling back to a full slot clear.
                    bool restored_from_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                    std::size_t ckpt_pos = 0;
                    if (impl_->RestorePartialCheckpoint(slot_idx, slot_id, lcp, ckpt_pos, restored_pending_h)) {
                        lcp = ckpt_pos;
                        restored_from_ckpt = true;
                    }
#endif
                    if (!restored_from_ckpt) {
                        llama_memory_seq_rm(mem, slot_id, -1, -1);
                        if (slot_idx < impl_->slot_checkpoints.size()) {
                            impl_->slot_checkpoints[slot_idx].clear();
                        }
                        lcp = 0;
                    }
                }
            }
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (impl_->ctx_dft != nullptr) {
                if (lcp == 0) {
                    if (!impl_->mtp_mem_shared) {
                        llama_memory_seq_rm(llama_get_memory(impl_->ctx_dft), slot_id, -1, -1);
                    }
                    std::fill(impl_->mtp_pending_h[slot_idx].begin(), impl_->mtp_pending_h[slot_idx].end(), 0.0F);
                    impl_->mtp_deferred[slot_idx].Clear();
                } else {
                    if (!impl_->mtp_mem_shared) {
                        auto& d = impl_->mtp_deferred[slot_idx];
                        if (d.pending) {
                            const int64_t dft_len0 =
                                llama_memory_seq_pos_max(llama_get_memory(impl_->ctx_dft), slot_id) + 1;
                            if (static_cast<int64_t>(lcp) > dft_len0 && !d.pos.empty() &&
                                d.pos.front() == static_cast<llama_pos>(dft_len0) &&
                                static_cast<int64_t>(d.pos.back()) + 1 >= static_cast<int64_t>(lcp)) {
                                (void)impl_->MtpFlushDeferred(slot_id,
                                                              static_cast<int32_t>(static_cast<int64_t>(lcp) - dft_len0));
                            } else {
                                d.Clear();
                            }
                        }
                        const int64_t dft_len = llama_memory_seq_pos_max(llama_get_memory(impl_->ctx_dft), slot_id) + 1;
                        if (dft_len < static_cast<int64_t>(lcp)) {
                            // Draft KV is behind target KV; clear both to stay synchronized.
                            llama_memory_seq_rm(mem, slot_id, -1, -1);
                            llama_memory_seq_rm(llama_get_memory(impl_->ctx_dft), slot_id, -1, -1);
                            if (slot_idx < impl_->slot_checkpoints.size()) {
                                impl_->slot_checkpoints[slot_idx].clear();
                            }
                            std::fill(impl_->mtp_pending_h[slot_idx].begin(), impl_->mtp_pending_h[slot_idx].end(),
                                      0.0F);
                            lcp = 0;
                        } else if (dft_len > static_cast<int64_t>(lcp)) {
                            if (!llama_memory_seq_rm(llama_get_memory(impl_->ctx_dft), slot_id,
                                                     static_cast<llama_pos>(lcp), -1)) {
                                llama_memory_seq_rm(mem, slot_id, -1, -1);
                                llama_memory_seq_rm(llama_get_memory(impl_->ctx_dft), slot_id, -1, -1);
                                if (slot_idx < impl_->slot_checkpoints.size()) {
                                    impl_->slot_checkpoints[slot_idx].clear();
                                }
                                std::fill(impl_->mtp_pending_h[slot_idx].begin(), impl_->mtp_pending_h[slot_idx].end(),
                                          0.0F);
                                lcp = 0;
                            }
                        }
                    }
                    if (lcp > 0 && lcp != prev_slot_len && !restored_pending_h) {
                        // Trimmed into the middle of a prior sequence; zero carryover for
                        // the single boundary position (all subsequent prefill tokens get exact h_tgt).
                        std::fill(impl_->mtp_pending_h[slot_idx].begin(), impl_->mtp_pending_h[slot_idx].end(), 0.0F);
                    }
                }
            }
#else
            (void)restored_pending_h;
#endif
        }

        prefill_start_offset = static_cast<int64_t>(lcp);
        result.cached_tokens = static_cast<std::int32_t>(lcp);
        impl_->slots[slot_idx].tokens.resize(lcp);
        impl_->slots[slot_idx].session_key = request.session_key;
        impl_->slots[slot_idx].last_used = ++impl_->slot_clock;
    } else if (request.reset_context) {
        llama_memory_clear(mem, true);
        impl_->ResetAllSlots();
        if (!impl_->slots.empty()) {
            impl_->slots[0].session_key = request.session_key;
            impl_->slots[0].last_used = ++impl_->slot_clock;
        }
    }

    const int64_t n_past = llama_memory_seq_pos_max(mem, slot_id) + 1;  // 0 when empty
    const int64_t remaining_prompt = n_prompt - prefill_start_offset;
    if (n_past + remaining_prompt > n_ctx) {
        return fail(ErrorCode::kContextOverflow, "prompt does not fit in the context window");
    }

    // Wire cancellation into llama.cpp's abort callback for the duration of
    // this call so a long prefill batch can also be interrupted.
    impl_->abort_requested.store(false, std::memory_order_relaxed);
    impl_->active_cancel = &should_cancel;
    struct ClearCancel {
        Impl* impl;
        ~ClearCancel() { impl->active_cancel = nullptr; }
    } clear_cancel{impl_.get()};

    auto cancelled = [&] {
        return impl_->abort_requested.load(std::memory_order_relaxed) || (should_cancel && should_cancel());
    };

#if defined(SONDER_LLAMACPP_HAS_MTP)
    bool use_mtp = (impl_->active_spec == SpeculativeType::kDraftMtp) && (impl_->ctx_dft != nullptr) &&
                   !result.state_restored;
#else
    bool use_mtp = false;
#endif
    bool use_ngram_mod =
        !use_mtp && (impl_->active_spec == SpeculativeType::kNgramMod) && (impl_->ngram_mod != nullptr) &&
        impl_->active_draft_max > 0;
    bool use_ngram_short = !use_mtp && (impl_->active_spec == SpeculativeType::kNgram) && impl_->active_draft_max > 0;
    bool use_ngram = use_ngram_mod || use_ngram_short;
    if (impl_->architecture == sonder::inference::ModelArchitecture::recurrent) {
        use_mtp = false;
        use_ngram_mod = false;
        use_ngram_short = false;
        use_ngram = false;
    }
    uint32_t rs_cap = 0;
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ)
    if (impl_->architecture == sonder::inference::ModelArchitecture::hybrid) {
        rs_cap = llama_n_rs_seq(ctx);
    }
#endif
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
    const bool can_hybrid_spec_ckpt =
        (impl_->architecture == sonder::inference::ModelArchitecture::hybrid) &&
        (Impl::PartialCheckpointFlags() != 0);
#else
    const bool can_hybrid_spec_ckpt = false;
#endif
    int32_t effective_draft_max = impl_->active_draft_max;
    if (impl_->architecture == sonder::inference::ModelArchitecture::hybrid) {
        if (use_mtp) {
            if (rs_cap == 0) {
                use_mtp = false;
                effective_draft_max = 0;
            } else {
                effective_draft_max = std::min<int32_t>(effective_draft_max, static_cast<int32_t>(rs_cap));
            }
        } else if (use_ngram) {
            if (rs_cap == 0 && !can_hybrid_spec_ckpt) {
                use_ngram = false;
                use_ngram_mod = false;
                use_ngram_short = false;
                effective_draft_max = 0;
            }
        }
    }

    // Bind GPU backend sampling when supported (avoids copying full vocab logits
    // across PCIe on every decode & verification step). Disabled when a grammar
    // is active so CPU resampling can read raw logits on grammar rejection.
    [[maybe_unused]] bool target_gpu_sampled = false;
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
    if (!grammar_sampler) {
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
    const bool pure_greedy =
        request.sampling.temperature <= 0.0F && request.sampling.logit_bias.empty() &&
        request.sampling.repeat_penalty == 1.0F && request.sampling.presence_penalty == 0.0F &&
        request.sampling.frequency_penalty == 0.0F;

    auto sample_target = [&](int32_t idx) -> Token {
        Token tok = LLAMA_TOKEN_NULL;
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
        if (target_gpu_sampled) {
            tok = llama_get_sampled_token_ith(ctx, idx);
            if (tok != LLAMA_TOKEN_NULL) {
                llama_sampler_accept(sampler.get(), tok);
                return tok;
            }
        }
#endif
        if (!grammar_sampler) {
            return llama_sampler_sample(sampler.get(), ctx, idx);
        }

        // Fast grammar-based rejection sampling:
        // 1. Sample candidate token WITHOUT mutating sampler penalty history and
        //    without running GBNF across the full 248K vocab.
        const float* logits = llama_get_logits_ith(ctx, idx);
        if (logits == nullptr) {
            return LLAMA_TOKEN_NULL;
        }
        if (pure_greedy) {
            int32_t best = 0;
            float best_val = logits[0];
            for (int32_t v = 1; v < n_vocab; ++v) {
                if (logits[v] > best_val) {
                    best_val = logits[v];
                    best = v;
                }
            }
            tok = best;
        } else {
            auto& cand = impl_->grammar_candidates;
            cand.resize(static_cast<std::size_t>(n_vocab));
            for (int32_t v = 0; v < n_vocab; ++v) {
                cand[static_cast<std::size_t>(v)] = llama_token_data{v, logits[v], 0.0F};
            }
            llama_token_data_array cur_p{cand.data(), cand.size(), -1, false};
            llama_sampler_apply(sampler.get(), &cur_p);
            if (cur_p.selected < 0 || static_cast<std::size_t>(cur_p.selected) >= cur_p.size) {
                return LLAMA_TOKEN_NULL;
            }
            tok = cur_p.data[cur_p.selected].id;
        }

        // 2. Test ONLY the 1 candidate token against the GBNF grammar.
        llama_token_data single{tok, 1.0F, 0.0F};
        llama_token_data_array single_arr{&single, 1, -1, false};
        llama_sampler_apply(grammar_sampler.get(), &single_arr);
        if (std::isfinite(single.logit)) {
            llama_sampler_accept(grammar_sampler.get(), tok);
            llama_sampler_accept(sampler.get(), tok);
            return tok;
        }

        // 3. Rare rejection fallback: mask full vocabulary with GBNF, then resample.
        auto& cand = impl_->grammar_candidates;
        cand.resize(static_cast<std::size_t>(n_vocab));
        for (int32_t v = 0; v < n_vocab; ++v) {
            cand[static_cast<std::size_t>(v)] = llama_token_data{v, logits[v], 0.0F};
        }
        llama_token_data_array cur_p{cand.data(), cand.size(), -1, false};
        llama_sampler_apply(grammar_sampler.get(), &cur_p);
        llama_sampler_apply(sampler.get(), &cur_p);
        if (cur_p.selected < 0 || static_cast<std::size_t>(cur_p.selected) >= cur_p.size ||
            !std::isfinite(cur_p.data[cur_p.selected].logit)) {
            // Grammar deadlocked (no finite candidate survived); fail cleanly
            // rather than triggering GGML_ABORT / throwing inside llama_grammar_accept.
            return LLAMA_TOKEN_NULL;
        }
        tok = cur_p.data[cur_p.selected].id;
        llama_sampler_accept(grammar_sampler.get(), tok);
        llama_sampler_accept(sampler.get(), tok);
        return tok;
    };

    bool slot_valid = true;
    auto invalidate_slot = [&]() {
        slot_valid = false;
        llama_memory_seq_rm(mem, slot_id, -1, -1);
        if (slot_idx < impl_->slots.size()) {
            impl_->slots[slot_idx].tokens.clear();
            impl_->slots[slot_idx].session_key.clear();
            impl_->slots[slot_idx].last_used = 0;
        }
        if (slot_idx < impl_->slot_checkpoints.size()) {
            impl_->slot_checkpoints[slot_idx].clear();
        }
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (impl_->ctx_dft != nullptr) {
            if (!impl_->mtp_mem_shared) {
                llama_memory_seq_rm(llama_get_memory(impl_->ctx_dft), slot_id, -1, -1);
            }
            if (slot_idx < impl_->mtp_pending_h.size()) {
                std::fill(impl_->mtp_pending_h[slot_idx].begin(), impl_->mtp_pending_h[slot_idx].end(), 0.0F);
            }
            if (slot_idx < impl_->mtp_deferred.size()) {
                impl_->mtp_deferred[slot_idx].Clear();
            }
        }
#endif
    };

    // ---- Prefill in n_batch chunks (split at n_ubatch when MTP is active so
    // MtpProcess receives exact nextn hidden states for every micro-batch) ----
    const auto n_batch = static_cast<int64_t>(std::max<int32_t>(1, impl_->batch_cap));
    const auto n_ubatch = static_cast<int64_t>(std::max<uint32_t>(1, llama_n_ubatch(ctx)));
    const int64_t prefill_chunk = use_mtp ? std::min(n_batch, n_ubatch) : n_batch;
    const bool want_capture = request.capture_at_tokens > static_cast<std::size_t>(prefill_start_offset) &&
                              request.capture_at_tokens <= request.prompt.size();
    const int64_t capture_offset = want_capture ? static_cast<int64_t>(request.capture_at_tokens) : -1;
    // On hybrid (attention + recurrent/SSM) models with prefix caching enabled,
    // capture a lightweight PARTIAL_ONLY recurrent checkpoint 16 tokens before
    // the end of long prefills (>= 512 tokens, prior to trailing assistant cue /
    // `<think>` scaffolding) so multi-turn continuations restore hybrid state
    // without recomputing the shared prefix — while short prompts pay zero
    // checkpoint overhead.
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
    const bool want_hybrid_prefix_ckpt =
        impl_->options.prefix_cache && !has_capture && !result.state_restored &&
        impl_->architecture == sonder::inference::ModelArchitecture::hybrid && n_prompt >= 512 &&
        Impl::PartialCheckpointFlags() != 0;
#else
    const bool want_hybrid_prefix_ckpt = false;
#endif
    const int64_t hybrid_ckpt_offset =
        (want_hybrid_prefix_ckpt && (n_prompt - 16) > prefill_start_offset) ? (n_prompt - 16) : -1;
    int64_t cur_pos = n_past;
    const auto prefill_start = Clock::now();
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
        BatchClear(impl_->batch);
        for (int32_t j = 0; j < n; ++j) {
            const bool is_last_prompt_tok = (off + j + 1 == n_prompt);
            BatchAdd(impl_->batch, request.prompt[static_cast<std::size_t>(off + j)],
                     static_cast<llama_pos>(cur_pos + j), slot_id, is_last_prompt_tok);
        }
        const int32_t rc = llama_decode(ctx, impl_->batch);
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
            return fail(ErrorCode::kDecodeFailed, "llama_decode failed during prefill (rc=" + std::to_string(rc) + ")");
        }
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (use_mtp && !impl_->MtpProcessPrefill(impl_->batch, slot_id)) {
            invalidate_slot();
            return fail(ErrorCode::kDecodeFailed, "MTP draft context prefill sync failed");
        }
#endif
        if (slot_idx < impl_->slots.size()) {
            auto& stoks = impl_->slots[slot_idx].tokens;
            stoks.insert(stoks.end(), request.prompt.begin() + off, request.prompt.begin() + step_end);
        }
        result.prompt_tokens += n;
        cur_pos += n;
        off = step_end;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
        if (hybrid_ckpt_offset == off && slot_valid) {
            impl_->CapturePartialCheckpoint(slot_idx, slot_id, static_cast<std::size_t>(cur_pos));
        }
#endif
        if (capture_offset == off) {
            std::vector<std::uint8_t> captured;
            result.capture_status = CaptureSequenceState(captured, request.max_capture_bytes);
            if (result.capture_status.ok()) {
                result.state_captured = true;
                if (request.on_state_captured) {
                    request.on_state_captured(std::move(captured));
                }
            }
        }
    }
    result.prefill_ms = MsSince(prefill_start);
    impl_->Emit(TelemetryKind::kPrefillDone, result.prompt_tokens, result.prefill_ms);

    // ---- Streaming decode (with optional in-process speculation) ----
    Utf8StreamBuffer utf8;
    int64_t pos = cur_pos;
    std::size_t ngram_indexed = 0;
    const auto decode_start = Clock::now();
    auto finish = [&](StopReason reason) {
        result.stop_reason = reason;
        result.decode_ms = MsSince(decode_start);
        if (use_ngram_mod && impl_->ngram_mod != nullptr && reason != StopReason::kError) {
            ngram_indexed = impl_->ngram_mod->IndexRange(impl_->ngram_history, ngram_indexed, impl_->ngram_history.size());
        }
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
        if (slot_valid && want_hybrid_prefix_ckpt && reason != StopReason::kError && reason != StopReason::kCancelled &&
            slot_idx < impl_->slots.size() && !impl_->slots[slot_idx].tokens.empty()) {
            bool mtp_ok = true;
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp && !impl_->mtp_mem_shared && slot_idx < impl_->mtp_deferred.size() &&
                impl_->mtp_deferred[slot_idx].pending) {
                mtp_ok = impl_->MtpFlushDeferred(slot_id);
                if (!mtp_ok) {
                    invalidate_slot();
                }
            }
#endif
            if (mtp_ok && slot_valid) {
                impl_->CapturePartialCheckpoint(slot_idx, slot_id, impl_->slots[slot_idx].tokens.size());
            }
        }
#endif
        std::string tail = utf8.Flush();
        if (!tail.empty() && on_token && reason != StopReason::kError) {
            on_token(TokenEvent{-1, std::move(tail), result.generated_tokens});
        }
        impl_->Emit(TelemetryKind::kGenerationDone, result.generated_tokens, result.decode_ms,
                    std::string(ToString(reason)));
        return result;
    };

    if (request.max_new_tokens == 0) return finish(StopReason::kMaxTokens);
    if (cancelled()) return finish(StopReason::kCancelled);

    auto emit_token = [&](Token tok, bool& keep_going) -> Status {
        std::string piece;
        if (Status s = TokenToPiece(tok, piece); !s.ok()) return s;
        TokenEvent ev{tok, utf8.Push(piece), result.generated_tokens};
        ++result.generated_tokens;
        keep_going = on_token ? on_token(ev) : true;
        return Status::Ok();
    };

    // Sample first generated token from the prefill's final logits.
    const auto first_tok_start = Clock::now();
    Token cur_tok = sample_target(-1);
    if (cur_tok == LLAMA_TOKEN_NULL) {
        result.status = Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
        return finish(StopReason::kError);
    }
    if (llama_vocab_is_eog(impl_->vocab, cur_tok)) return finish(StopReason::kEndOfGeneration);
    {
        bool keep_going = true;
        if (Status s = emit_token(cur_tok, keep_going); !s.ok()) {
            result.status = s;
            return finish(StopReason::kError);
        }
        if (!keep_going) return finish(StopReason::kCancelled);
        impl_->Emit(TelemetryKind::kTokenDecoded, 1, MsSince(first_tok_start));
        if (result.generated_tokens >= request.max_new_tokens) return finish(StopReason::kMaxTokens);
        if (pos >= n_ctx) return finish(StopReason::kContextFull);
    }

    // Initialize n-gram index over prompt tokens when n-gram speculation is active.
    AdaptiveDraftController adaptive_ctrl(effective_draft_max);
    if (use_ngram_mod && impl_->ngram_mod != nullptr) {
        impl_->ngram_history.assign(request.prompt.begin(), request.prompt.end());
        impl_->ngram_mod->BeginRequest(impl_->ngram_history);
        ngram_indexed = impl_->ngram_history.size() > impl_->ngram_mod->n_match()
                            ? impl_->ngram_history.size() - impl_->ngram_mod->n_match()
                            : 0;
    } else if (use_ngram_short) {
        impl_->ngram_matcher.Reset();
        impl_->ngram_history.assign(request.prompt.begin(), request.prompt.end());
        impl_->ngram_matcher.IndexRange(impl_->ngram_history, 0, impl_->ngram_history.size());
        ngram_indexed = impl_->ngram_history.size();
    }

    auto& draft_tokens = impl_->draft_scratch;
    auto& accepted_tokens = impl_->accepted_scratch;

    while (true) {
        if (cancelled()) return finish(StopReason::kCancelled);

        const auto step_start = Clock::now();
        const int64_t rem_gen =
            static_cast<int64_t>(request.max_new_tokens) - static_cast<int64_t>(result.generated_tokens);
        const int64_t max_k = std::min<int64_t>({rem_gen - 1, n_ctx - pos - 1, n_batch - 1, n_ubatch - 1});

        draft_tokens.clear();
        if (max_k > 0) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp) {
                const auto k_lim = static_cast<int32_t>(std::min<int64_t>(max_k, effective_draft_max));
                impl_->MtpDraft(slot_id, cur_tok, pos, k_lim, draft_tokens);
            } else
#endif
            if (use_ngram_mod && impl_->ngram_mod != nullptr) {
                const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, effective_draft_max));
                impl_->ngram_mod->Draft(impl_->ngram_history, cur_tok, k_lim, draft_tokens);
                for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                    if (llama_vocab_is_eog(impl_->vocab, draft_tokens[j])) {
                        draft_tokens.resize(j + 1);
                        break;
                    }
                }
            } else if (use_ngram_short) {
                const int32_t ctrl_len = adaptive_ctrl.NextDraftLength();
                if (ctrl_len > 0) {
                    // Append cur_tok to history for lookup, then index cur_tok after lookup.
                    impl_->ngram_history.push_back(cur_tok);
                    const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, ctrl_len));
                    impl_->ngram_matcher.Draft(impl_->ngram_history, k_lim, draft_tokens);
                    impl_->ngram_history.pop_back();
                    // Never draft past an EOG token.
                    for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                        if (llama_vocab_is_eog(impl_->vocab, draft_tokens[j])) {
                            draft_tokens.resize(j + 1);
                            break;
                        }
                    }
                } else {
                    adaptive_ctrl.NoteSkippedStep();
                }
            }
        }

        // On hybrid models whose recurrent rollback window `rs_cap` is smaller than
        // the proposed draft (e.g. ngram-mod with n_rs_seq == 0), take a lightweight
        // PARTIAL_ONLY checkpoint before verification so partial acceptance rolls
        // back cleanly without reserving per-step GPU rollback slots.
        [[maybe_unused]] bool have_spec_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
        if (!draft_tokens.empty() && impl_->architecture == sonder::inference::ModelArchitecture::hybrid &&
            draft_tokens.size() > static_cast<std::size_t>(rs_cap)) {
            const llama_state_seq_flags flags = Impl::PartialCheckpointFlags();
            const std::size_t sz = flags != 0 ? llama_state_seq_get_size_ext(ctx, slot_id, flags) : 0;
            if (sz > 0 && sz <= Impl::kMaxPartialCheckpointBytes) {
                if (impl_->spec_ckpt_buf.size() < sz) {
                    impl_->spec_ckpt_buf.resize(sz);
                }
                const std::size_t written =
                    llama_state_seq_get_data_ext(ctx, impl_->spec_ckpt_buf.data(), sz, slot_id, flags);
                if (written > 0) {
                    impl_->spec_ckpt_size = written;
                    have_spec_ckpt = true;
                }
            }
            if (!have_spec_ckpt) {
                draft_tokens.clear();
            }
        }
#endif

        // Build ONE verification batch: [cur_tok, draft_0, ..., draft_{k-1}].
        BatchClear(impl_->batch);
        BatchAdd(impl_->batch, cur_tok, static_cast<llama_pos>(pos), slot_id, /*logits=*/true);
        for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
            BatchAdd(impl_->batch, draft_tokens[j], static_cast<llama_pos>(pos + 1 + static_cast<int64_t>(j)), slot_id,
                     /*logits=*/true);
        }

        const int32_t rc = llama_decode(ctx, impl_->batch);
        if (rc != 0) {
            bool rolled_back = llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1);
            if (!rolled_back) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                if (have_spec_ckpt) {
                    const llama_state_seq_flags flags = Impl::PartialCheckpointFlags();
                    const std::size_t read = llama_state_seq_set_data_ext(ctx, impl_->spec_ckpt_buf.data(),
                                                                          impl_->spec_ckpt_size, slot_id, flags);
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
            if (use_mtp && !impl_->MtpCommitVerified(impl_->batch, slot_id, 0)) {
                invalidate_slot();
                result.status = Status::Error(ErrorCode::kDecodeFailed, "MTP draft context sync failed");
                return finish(StopReason::kError);
            }
#endif
            if (slot_idx < impl_->slots.size()) {
                impl_->slots[slot_idx].tokens.push_back(cur_tok);
            }
            if (use_ngram_mod && impl_->ngram_mod != nullptr) {
                impl_->ngram_history.push_back(cur_tok);
                ngram_indexed =
                    impl_->ngram_mod->IndexRange(impl_->ngram_history, ngram_indexed, impl_->ngram_history.size());
            } else if (use_ngram_short) {
                impl_->ngram_history.push_back(cur_tok);
                impl_->ngram_matcher.IndexRange(impl_->ngram_history, ngram_indexed, impl_->ngram_history.size());
                ngram_indexed = impl_->ngram_history.size();
            }
            ++pos;

            const Token next_tok = sample_target(0);
            if (next_tok == LLAMA_TOKEN_NULL) {
                result.status =
                    Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
                return finish(StopReason::kError);
            }
            if (llama_vocab_is_eog(impl_->vocab, next_tok)) return finish(StopReason::kEndOfGeneration);

            bool keep_going = true;
            if (Status s = emit_token(next_tok, keep_going); !s.ok()) {
                result.status = s;
                return finish(StopReason::kError);
            }
            if (!keep_going) return finish(StopReason::kCancelled);
            impl_->Emit(TelemetryKind::kTokenDecoded, 1, MsSince(step_start));
            if (result.generated_tokens >= request.max_new_tokens) {
                if (use_ngram_mod) impl_->ngram_history.push_back(next_tok);
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
        std::size_t n_accepted = 0;
        bool sample_deadlock = false;
        for (std::size_t j = 0; j < k; ++j) {
            const Token sampled = sample_target(static_cast<int32_t>(j));
            if (sampled == LLAMA_TOKEN_NULL) {
                sample_deadlock = true;
                break;
            }
            accepted_tokens.push_back(sampled);
            if (sampled == draft_tokens[j]) {
                ++n_accepted;
            } else {
                break;
            }
        }
        if (!sample_deadlock && n_accepted == k) {
            const Token bonus = sample_target(static_cast<int32_t>(k));
            if (bonus == LLAMA_TOKEN_NULL) {
                sample_deadlock = true;
            } else {
                accepted_tokens.push_back(bonus);
            }
        }
        if (sample_deadlock) {
            invalidate_slot();
            result.status =
                Status::Error(ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
            return finish(StopReason::kError);
        }
        result.draft_accepted_tokens += static_cast<std::int32_t>(n_accepted);
        if (use_ngram_mod && impl_->ngram_mod != nullptr) {
            if (impl_->ngram_mod->Observe(k, n_accepted)) {
                ngram_indexed = 0;
            }
        } else if (use_ngram_short) {
            adaptive_ctrl.Observe(static_cast<std::int32_t>(k), static_cast<std::int32_t>(n_accepted));
        }

        // Commit `cur_tok` (at `pos`), then emit each verified token + bonus token.
        if (slot_idx < impl_->slots.size()) {
            impl_->slots[slot_idx].tokens.push_back(step_base_tok);
        }
        if (use_ngram) {
            impl_->ngram_history.push_back(step_base_tok);
        }
        int64_t new_pos = pos + 1;
        std::size_t committed_drafts = 0;

        auto rollback_rejected = [&](int64_t keep_pos, std::size_t keep_drafts) -> bool {
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp && !impl_->MtpCommitVerified(impl_->batch, slot_id, static_cast<int32_t>(keep_drafts))) {
                invalidate_slot();
                return false;
            }
#endif
            const int64_t batch_end_pos = pos + 1 + static_cast<int64_t>(k);
            if (keep_pos < batch_end_pos) {
                if (!llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(keep_pos), -1)) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                    if (have_spec_ckpt) {
                        const llama_state_seq_flags flags = Impl::PartialCheckpointFlags();
                        const std::size_t read = llama_state_seq_set_data_ext(ctx, impl_->spec_ckpt_buf.data(),
                                                                              impl_->spec_ckpt_size, slot_id, flags);
                        if (read == 0 || !llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1)) {
                            invalidate_slot();
                            return false;
                        }
                        // Replay only the 1 + keep_drafts accepted tokens (logits=false since
                        // all accepted + bonus tokens have already been sampled).
                        BatchClear(impl_->batch);
                        BatchAdd(impl_->batch, step_base_tok, static_cast<llama_pos>(pos), slot_id, /*logits=*/false);
                        for (std::size_t j = 0; j < keep_drafts; ++j) {
                            BatchAdd(impl_->batch, draft_tokens[j],
                                     static_cast<llama_pos>(pos + 1 + static_cast<int64_t>(j)), slot_id,
                                     /*logits=*/false);
                        }
                        if (llama_decode(ctx, impl_->batch) != 0) {
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
            if (llama_vocab_is_eog(impl_->vocab, tok)) {
                (void)rollback_rejected(new_pos, committed_drafts);
                if (m > 0) {
                    impl_->Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m), MsSince(step_start));
                }
                return finish(StopReason::kEndOfGeneration);
            }
            bool keep_going = true;
            if (Status s = emit_token(tok, keep_going); !s.ok()) {
                (void)rollback_rejected(new_pos, committed_drafts);
                result.status = s;
                return finish(StopReason::kError);
            }
            if (!is_bonus) {
                if (slot_idx < impl_->slots.size()) {
                    impl_->slots[slot_idx].tokens.push_back(tok);
                }
                if (use_ngram) {
                    impl_->ngram_history.push_back(tok);
                }
                ++new_pos;
                ++committed_drafts;
            }
            if (!keep_going) {
                (void)rollback_rejected(new_pos, committed_drafts);
                return finish(StopReason::kCancelled);
            }
            if (result.generated_tokens >= request.max_new_tokens) {
                if (is_bonus && use_ngram_mod) impl_->ngram_history.push_back(tok);
                (void)rollback_rejected(new_pos, committed_drafts);
                impl_->Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m + 1), MsSince(step_start));
                return finish(StopReason::kMaxTokens);
            }
            if (is_bonus) {
                cur_tok = tok;
            }
        }

        if (!rollback_rejected(new_pos, committed_drafts)) {
            result.status = Status::Error(ErrorCode::kDecodeFailed, "failed to trim rejected speculative KV tail");
            return finish(StopReason::kError);
        }
        if (use_ngram_mod && impl_->ngram_mod != nullptr) {
            ngram_indexed =
                impl_->ngram_mod->IndexRange(impl_->ngram_history, ngram_indexed, impl_->ngram_history.size());
        } else if (use_ngram_short) {
            impl_->ngram_matcher.IndexRange(impl_->ngram_history, ngram_indexed, impl_->ngram_history.size());
            ngram_indexed = impl_->ngram_history.size();
        }
        impl_->Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(accepted_tokens.size()),
                    MsSince(step_start));
        pos = new_pos;
        if (pos >= n_ctx) return finish(StopReason::kContextFull);
    }
}

}  // namespace sonder::backends::llamacpp
