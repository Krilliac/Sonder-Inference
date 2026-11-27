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
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <random>
#include <regex>
#include <span>
#include <string>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "batch_scheduler.hpp"
#include "grammar_retry.hpp"
#include "embed_batch_planner.hpp"
#include "gguf_compat.hpp"
#include "prefix_coalescer.hpp"
#include "radix_cache.hpp"
#include "request_guards.hpp"
#include "speculation_transition.hpp"
#include "suffix_matcher.hpp"

#include "ggml-backend.h"
#include "llama.h"

// Older trees (b11195) declare the NextN/MTP calls only in the private src/llama-ext.h;
// from v1.0.0 they are public in llama.h and llama-ext.h is not needed (CMake detects which).
#if defined(SONDER_LLAMACPP_HAS_MTP) && defined(SONDER_LLAMACPP_MTP_VIA_EXT)
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

struct GgmlDeviceLineRegistry {
    static constexpr std::size_t kMaxEntries = 64;
    static constexpr std::size_t kMaxPendingBytes = 2048;

    std::mutex mutex;
    std::string pending;
    std::vector<sonder::inference::GgmlDeviceTelemetry> entries;

    void Feed(LogLevel level, const char* text) {
        if (text == nullptr || text[0] == '\0') return;
        std::lock_guard<std::mutex> lock(mutex);
        if (level != LogLevel::kCont) {
            pending.clear();
        }
        const std::string_view chunk(text);
        std::size_t pos = 0;
        while (pos < chunk.size()) {
            const std::size_t nl = chunk.find('\n', pos);
            const std::string_view part =
                nl == std::string_view::npos ? chunk.substr(pos) : chunk.substr(pos, nl - pos);
            if (pending.size() < kMaxPendingBytes) {
                const std::size_t take = std::min(part.size(), kMaxPendingBytes - pending.size());
                pending.append(part.data(), take);
            }
            if (nl == std::string_view::npos) {
                break;
            }
            if (auto parsed = sonder::inference::parse_ggml_device_line(pending)) {
                sonder::inference::register_ggml_device_telemetry(*parsed);
                bool updated = false;
                for (auto& existing : entries) {
                    if (existing.backend == parsed->backend && existing.index == parsed->index &&
                        existing.device_name == parsed->device_name) {
                        existing = std::move(*parsed);
                        updated = true;
                        break;
                    }
                }
                if (!updated && entries.size() < kMaxEntries) {
                    entries.push_back(std::move(*parsed));
                }
            }
            pending.clear();
            pos = nl + 1;
        }
    }

    std::vector<sonder::inference::GgmlDeviceTelemetry> Snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        return entries;
    }

    void Clear() noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        pending.clear();
        entries.clear();
    }
};

GgmlDeviceLineRegistry& DeviceLineRegistry() {
    static GgmlDeviceLineRegistry registry;
    return registry;
}

}  // namespace

void SetLogSink(LogSink sink) noexcept { g_log_sink.store(sink, std::memory_order_release); }

// llama.cpp logs verbosely to stderr by default. Keep warnings and errors;
// set SONDER_LLAMACPP_VERBOSE=1 to see everything.
void RouteLog(LogLevel level, const char* text) {
    if (text != nullptr) {
        DeviceLineRegistry().Feed(level, text);
    }
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
llama_context_params MakeContextParams(const LoadOptions& options, bool swa_compact_qualified = true) {
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
#if defined(SONDER_LLAMACPP_HAS_SWA_FULL)
    cp.swa_full = !(options.swa_kv_mode == sonder::inference::SwaKvMode::CompactIfQualified && swa_compact_qualified);
#else
    (void)swa_compact_qualified;
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
#if defined(SONDER_LLAMACPP_HAS_SWA_FULL)
    s.swa_full = cp.swa_full;
#else
    s.swa_full = true;
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

// llama.cpp's grammar code reports failure by throwing: when a lazy grammar's trigger matches, it
// replays the text buffered from the match point into the grammar stack and throws
// std::runtime_error (from llama_grammar_accept_token in llama-grammar.cpp) when that text does not
// fit the grammar, and any allocation can throw std::bad_alloc. The exception unwinds through
// llama_sampler_accept / llama_sampler_apply, which are declared extern "C". Under MSVC /EHsc the
// compiler assumes extern "C" functions never throw and may drop the try/catch around the call, so
// the exception then reaches a noexcept frame and calls std::terminate. This target is therefore
// compiled with /EHs (see CMakeLists.txt), which keeps the catch below alive.
//
// Accepts `tok` into a grammar sampler without letting llama.cpp's exceptions escape. Returns
// false on failure; the grammar state is then undefined and the request must be failed.
bool TryAcceptGrammar(llama_sampler* grammar, llama_token tok) noexcept {
    try {
        llama_sampler_accept(grammar, tok);
        return true;
    } catch (...) {
        return false;
    }
}

// Applies the grammar mask to `cur_p` without letting exceptions escape. Returns false on failure,
// in which case `cur_p` must not be used. Same exception-handling rationale as TryAcceptGrammar.
bool TryApplyGrammar(llama_sampler* grammar, llama_token_data_array* cur_p) noexcept {
    try {
        llama_sampler_apply(grammar, cur_p);
        return true;
    } catch (...) {
        return false;
    }
}

// Accepts one grammar_prefill token. llama_grammar_accept_impl aborts the process (GGML_ABORT, not
// catchable) when handed an end-of-generation token while the grammar stack is non-empty, so the
// token is first applied to a single-candidate array exactly as SampleTokenAt does and accepted
// only if the grammar leaves its logit finite. Out-of-range ids and EOG tokens the grammar masks
// are rejected, which makes the caller rebuild the grammar without prefill.
bool TryAcceptPrefillToken(llama_sampler* grammar, llama_token tok, int32_t n_vocab) noexcept {
    if (tok < 0 || tok >= n_vocab) return false;
    llama_token_data single{tok, 1.0F, 0.0F};
    llama_token_data_array single_arr{&single, 1, -1, false};
    if (!TryApplyGrammar(grammar, &single_arr)) return false;
    if (!std::isfinite(single.logit)) return false;
    return TryAcceptGrammar(grammar, tok);
}

bool TryAcceptSampler(llama_sampler* sampler, llama_token tok) noexcept {
    try {
        llama_sampler_accept(sampler, tok);
        return true;
    } catch (...) {
        return false;
    }
}

bool TryApplySampler(llama_sampler* sampler, llama_token_data_array* cur_p) noexcept {
    try {
        llama_sampler_apply(sampler, cur_p);
        return true;
    } catch (...) {
        return false;
    }
}

llama_token TrySampleToken(llama_sampler* sampler, llama_context* ctx, int32_t idx) noexcept {
    try {
        return llama_sampler_sample(sampler, ctx, idx);
    } catch (...) {
        return LLAMA_TOKEN_NULL;
    }
}

#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
bool TrySetSampler(llama_context* ctx, llama_seq_id seq_id, llama_sampler* sampler) noexcept {
    try {
        return llama_set_sampler(ctx, seq_id, sampler);
    } catch (...) {
        return false;
    }
}
#endif

int32_t TryDecode(llama_context* ctx, llama_batch batch, Status* err_out = nullptr) noexcept {
    try {
        return llama_decode(ctx, batch);
    } catch (const std::exception& ex) {
        if (err_out != nullptr) *err_out = ExceptionToStatus(ErrorCode::kDecodeFailed, "llama_decode", &ex);
        return -99;
    } catch (...) {
        if (err_out != nullptr) *err_out = ExceptionToStatus(ErrorCode::kDecodeFailed, "llama_decode", nullptr);
        return -99;
    }
}

int32_t TryEncode(llama_context* ctx, llama_batch batch, Status* err_out = nullptr) noexcept {
    try {
        return llama_encode(ctx, batch);
    } catch (const std::exception& ex) {
        if (err_out != nullptr) *err_out = ExceptionToStatus(ErrorCode::kDecodeFailed, "llama_encode", &ex);
        return -99;
    } catch (...) {
        if (err_out != nullptr) *err_out = ExceptionToStatus(ErrorCode::kDecodeFailed, "llama_encode", nullptr);
        return -99;
    }
}

std::size_t TryStateSeqGetData(llama_context* ctx, std::uint8_t* dst, std::size_t size,
                               llama_seq_id seq_id) noexcept {
    try {
        return llama_state_seq_get_data(ctx, dst, size, seq_id);
    } catch (...) {
        return 0;
    }
}

std::size_t TryStateSeqSetData(llama_context* ctx, const std::uint8_t* src, std::size_t size,
                               llama_seq_id seq_id) noexcept {
    try {
        return llama_state_seq_set_data(ctx, src, size, seq_id);
    } catch (...) {
        return 0;
    }
}

#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
std::size_t TryStateSeqGetDataExt(llama_context* ctx, std::uint8_t* dst, std::size_t size, llama_seq_id seq_id,
                                  llama_state_seq_flags flags) noexcept {
    try {
        return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, flags);
    } catch (...) {
        return 0;
    }
}

std::size_t TryStateSeqSetDataExt(llama_context* ctx, const std::uint8_t* src, std::size_t size, llama_seq_id seq_id,
                                  llama_state_seq_flags flags) noexcept {
    try {
        return llama_state_seq_set_data_ext(ctx, src, size, seq_id, flags);
    } catch (...) {
        return 0;
    }
}
#endif

// Order follows llama.cpp's common sampler: logit bias, penalties, then selector
// (greedy or top-k/typical/top-p/min-p/temp/dist). An optional GBNF grammar is
// returned separately so Generate() can apply fast grammar-based rejection
// sampling (checking only the 1 sampled token first and only masking the full
// vocabulary on the rare rejection case).
Status BuildSampler(const SamplingParams& p, const llama_vocab* vocab, int32_t n_vocab, int32_t n_ctx,
                    SamplerPtr& out, SamplerPtr& grammar_out, bool already_validated = false) {
    out.reset();
    grammar_out.reset();
    if (!already_validated) {
        if (Status s = ValidateSamplingRequestGuards(p, n_vocab); !s.ok()) return s;
    }
    Status status = RunGuarded(
        ErrorCode::kDecodeFailed, "sampler_init",
        [&]() -> Status {
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
    if (!p.grammar_prefill.empty() && (p.grammar.empty() || p.grammar_lazy)) {
        return Status::Error(ErrorCode::kInvalidArgument, "grammar_prefill requires a non-lazy grammar");
    }
    if (p.grammar_lazy && (p.grammar.empty() || (p.trigger_patterns.empty() && p.trigger_tokens.empty()))) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "lazy grammar requires a grammar and at least one trigger pattern or token");
    }
    // Every preserved token must map to exactly one id. This is a deliberate divergence from
    // llama-server, which silently skips preserved texts that do not tokenize to one id.
    for (const std::string& text : p.preserved_tokens) {
        llama_token id = 0;
        const int32_t n = text.size() > static_cast<std::size_t>(INT32_MAX)
                              ? 0
                              : llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), &id, 1,
                                               /*add_special=*/false, /*parse_special=*/true);
        if (n != 1) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "preserved token does not map to exactly one token id");
        }
    }
    if (!p.grammar.empty()) {
        llama_sampler* gram = nullptr;
        if (p.grammar_lazy) {
            std::vector<const char*> patterns;
            patterns.reserve(p.trigger_patterns.size());
            for (const std::string& pat : p.trigger_patterns) patterns.push_back(pat.c_str());
            // llama.cpp compiles trigger regexes with std::regex, which throws on a bad pattern.
            try {
                gram = llama_sampler_init_grammar_lazy_patterns(
                    vocab, p.grammar.c_str(), "root", patterns.data(), patterns.size(), p.trigger_tokens.data(),
                    p.trigger_tokens.size());
            } catch (const std::bad_alloc&) {
                throw;
            } catch (...) {
                gram = nullptr;
            }
        } else {
            try {
                gram = llama_sampler_init_grammar(vocab, p.grammar.c_str(), "root");
            } catch (const std::bad_alloc&) {
                throw;
            } catch (...) {
                gram = nullptr;
            }
        }
        if (gram == nullptr) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 p.grammar_lazy ? "invalid lazy GBNF grammar or trigger pattern"
                                                : "invalid GBNF grammar");
        }
        grammar_out.reset(gram);
        if (!p.grammar_prefill.empty()) {
            // Pre-accept the prefill tokens; on a rejected token rebuild the grammar once without
            // prefill (a failed accept leaves the grammar with no stacks, so it cannot be reused).
            const auto outcome = sonder::backends::llamacpp::PrefillGrammarWithRetry(
                std::span<const std::int32_t>(p.grammar_prefill.data(), p.grammar_prefill.size()),
                [&](std::int32_t t) { return TryAcceptPrefillToken(grammar_out.get(), t, n_vocab); },
                [&]() {
                    try {
                        grammar_out.reset(llama_sampler_init_grammar(vocab, p.grammar.c_str(), "root"));
                    } catch (const std::bad_alloc&) {
                        grammar_out.reset();
                        throw;
                    } catch (...) {
                        grammar_out.reset();
                    }
                    return grammar_out != nullptr;
                },
                [](std::string_view m) {
                    const std::string line = "sonder-llamacpp: " + std::string(m) + "\n";
                    RouteLog(LogLevel::kWarn, line.c_str());
                });
            if (outcome == sonder::backends::llamacpp::GrammarRetryOutcome::kFailed) {
                grammar_out.reset();
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "grammar prefill failed and the grammar could not be rebuilt");
            }
        }
    }
    if (p.temperature <= 0.0F) {
        llama_sampler_chain_add(chain.get(), llama_sampler_init_greedy());
        out = std::move(chain);
        return Status::Ok();
    }
    if (p.top_k > 0) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_k(ClampTopK(p.top_k, n_vocab)));
    if (p.typical_p < 1.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_typical(p.typical_p, 1));
    if (p.top_p < 1.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_top_p(p.top_p, 1));
    if (p.min_p > 0.0F) llama_sampler_chain_add(chain.get(), llama_sampler_init_min_p(p.min_p, 1));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_temp(p.temperature));
    llama_sampler_chain_add(chain.get(), llama_sampler_init_dist(p.seed));
    out = std::move(chain);
    return Status::Ok();
        },
        ErrorCode::kLoadFailed);
    if (!status.ok()) {
        out.reset();
        grammar_out.reset();
    }
    return status;
}

}  // namespace

struct LlamaCppBackend::Impl {
    GgufCompatOverlay compat_overlay;
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
    int32_t ctx_outputs_max = 0;
    int32_t ctx_outputs_max_per_seq = 0;
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
    std::vector<sonder::inference::SwaRollbackGuard> slot_swa_guards;
    std::uint64_t swa_window = 0;
    std::uint64_t swa_retained_slack = 0;
    bool swa_compact_active = false;
    // Target KV position corresponding to the currently captured `mtp_pending_h[slot_idx]`
    // (-1 when not captured or invalidated after drift/reset).
    std::vector<int64_t> slot_mtp_pending_h_pos;
    std::vector<std::vector<PartialCheckpoint>> slot_checkpoints;
    std::size_t total_checkpoint_bytes = 0;
    std::uint64_t checkpoint_clock = 0;
    std::vector<std::uint8_t> spec_ckpt_buf;
    std::size_t spec_ckpt_size = 0;
    std::uint64_t slot_clock = 0;

    // Bounded L2 host-RAM Radix prefix cache (`cache_ram_mib`).
    RadixPrefixCache radix_cache;

    // Cold shared-prefix prefill coalescer (attention-only multi-slot admission)
    // and lock-free atomic telemetry mirrors for cross-thread inspection.
    PrefixCoalescer prefix_coalescer;
    std::atomic<std::uint64_t> stat_coalesced_groups{0};
    std::atomic<std::uint64_t> stat_coalesced_followers{0};
    std::atomic<std::uint64_t> stat_coalesced_tokens_saved{0};
    std::atomic<std::uint64_t> stat_coalesce_fallbacks_timeout{0};
    std::atomic<std::uint64_t> stat_coalesce_fallbacks_producer_failed{0};

    void SyncCoalescerStats() noexcept {
        const PrefixCoalescerStats st = prefix_coalescer.stats();
        stat_coalesced_groups.store(st.coalesced_groups, std::memory_order_relaxed);
        stat_coalesced_followers.store(st.coalesced_followers, std::memory_order_relaxed);
        stat_coalesced_tokens_saved.store(st.coalesced_tokens_saved, std::memory_order_relaxed);
        stat_coalesce_fallbacks_timeout.store(st.coalesce_fallbacks_timeout, std::memory_order_relaxed);
        stat_coalesce_fallbacks_producer_failed.store(st.coalesce_fallbacks_producer_failed,
                                                      std::memory_order_relaxed);
    }

    // Isolated lazy embedding context so Embed() never disturbs live KV prefix
    // caches or MTP/backend-sampler graph reservations on `ctx`.
    llama_context* embed_ctx = nullptr;
    llama_batch embed_batch{};
    int32_t embed_batch_cap = 0;
    uint32_t embed_seq_max = 0;
    enum llama_pooling_type embed_pooling_requested = LLAMA_POOLING_TYPE_UNSPECIFIED;
    bool embed_batch_allocated = false;

    // Isolated lazy RANK-pooling context for cross-encoder reranking/scoring.
    llama_context* rank_ctx = nullptr;
    llama_batch rank_batch{};
    int32_t rank_batch_cap = 0;
    uint32_t rank_seq_max = 0;
    bool rank_batch_allocated = false;

    // Optional libmtmd multimodal projector context (`mmproj_path`).
#if defined(SONDER_LLAMACPP_HAS_MTMD)
    mtmd_context* mtmd_ctx = nullptr;
#endif

    // Resolved in-process speculative decoding state.
    SpeculativeType requested_spec = SpeculativeType::kNone;
    SpeculativeType active_spec = SpeculativeType::kNone;
    std::int32_t active_draft_max = 0;
    SpeculativeVerificationMode requested_spec_verification_mode = SpeculativeVerificationMode::TargetMatch;
    SpeculativeVerificationMode effective_spec_verification_mode = SpeculativeVerificationMode::TargetMatch;
    sonder::inference::SpeculativeVerifier spec_verifier{};
    std::string load_spec_fallback_reason;
    std::string load_draft_identity;
    bool mtp_model_supported = false;
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
        std::int64_t enqueue_time_us = 0;

        std::mutex done_mu;
        std::condition_variable done_cv;
        bool done = false;
        GenerateResult result{};
    };

    Clock::time_point clock_origin = Clock::now();
    StepCostEstimator step_cost_estimator{};
    BatchPlannerConfig batch_planner_config{};

    [[nodiscard]] std::int64_t NowUs() const noexcept {
        const auto us = std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - clock_origin).count();
        return us >= 0 ? (static_cast<std::int64_t>(us) + 1) : std::int64_t{1};
    }

    std::atomic<LlamaCppBackend*> owner{nullptr};
    std::mutex queue_mu;
    std::condition_variable worker_cv;
    std::deque<PendingRequest*> pending_queue;
    std::atomic<std::size_t> pending_count{0};
    std::atomic<std::size_t> pending_exclusive_count{0};
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

    void ResetSwaSlot(std::size_t slot_idx) noexcept {
        if (slot_idx < slot_swa_guards.size()) {
            slot_swa_guards[slot_idx].Reset();
        }
    }

    void AdvanceSwaSlot(std::size_t slot_idx, llama_memory_t mem, llama_seq_id slot_id,
                        std::uint64_t new_seq_len) noexcept {
        if (slot_idx >= slot_swa_guards.size()) return;
        slot_swa_guards[slot_idx].OnAdvance(new_seq_len);
        if (mem != nullptr && swa_compact_active) {
            slot_swa_guards[slot_idx].ObserveBackendPosMin(
                static_cast<std::int64_t>(llama_memory_seq_pos_min(mem, slot_id)));
        }
    }

    bool CheckSwaRollback(std::size_t slot_idx, llama_memory_t mem, llama_seq_id slot_id,
                          std::uint64_t target_pos) noexcept {
        if (slot_idx >= slot_swa_guards.size()) return true;
        if (mem != nullptr && swa_compact_active) {
            slot_swa_guards[slot_idx].ObserveBackendPosMin(
                static_cast<std::int64_t>(llama_memory_seq_pos_min(mem, slot_id)));
        }
        return slot_swa_guards[slot_idx].CheckAndApplyRollback(target_pos);
    }

    bool CanCloneSwaSource(std::size_t src_idx, llama_memory_t mem, llama_seq_id src_sid,
                           std::uint64_t target_pos) noexcept {
        if (!swa_compact_active || src_idx >= slot_swa_guards.size()) return true;
        if (mem != nullptr) {
            slot_swa_guards[src_idx].ObserveBackendPosMin(
                static_cast<std::int64_t>(llama_memory_seq_pos_min(mem, src_sid)));
        }
        return target_pos >= slot_swa_guards[src_idx].min_retained_pos();
    }

    void SyncClonedSwaSlot(std::size_t dst_idx, std::size_t src_idx, llama_memory_t mem, llama_seq_id dst_sid,
                           std::uint64_t cloned_len) noexcept {
        ResetSwaSlot(dst_idx);
        if (swa_compact_active && dst_idx < slot_swa_guards.size() && src_idx < slot_swa_guards.size()) {
            slot_swa_guards[dst_idx].ObserveBackendPosMin(
                static_cast<std::int64_t>(slot_swa_guards[src_idx].oldest_retained_row()));
        }
        AdvanceSwaSlot(dst_idx, mem, dst_sid, cloned_len);
    }

    void ResetAllSlots() noexcept {
        for (std::size_t i = 0; i < slots.size(); ++i) {
            slots[i].tokens.clear();
            slots[i].session_key.clear();
            slots[i].last_used = 0;
            if (i < slot_mtp_synced.size()) slot_mtp_synced[i] = false;
            if (i < slot_swa_guards.size()) slot_swa_guards[i].Reset();
            if (i < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[i] = -1;
        }
        for (auto& ckpts : slot_checkpoints) {
            ckpts.clear();
        }
        prefix_coalescer.ResetActive();
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
        pending_exclusive_count.store(0, std::memory_order_relaxed);
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
                (void)TrySetSampler(ctx_dft, static_cast<llama_seq_id>(i), nullptr);
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
            embed_seq_max = 0;
            embed_pooling_requested = LLAMA_POOLING_TYPE_UNSPECIFIED;
            embed_batch_allocated = false;
        }
        if (embed_ctx != nullptr) {
            llama_free(embed_ctx);
            embed_ctx = nullptr;
        }
        if (rank_batch_allocated) {
            llama_batch_free(rank_batch);
            rank_batch = {};
            rank_batch_cap = 0;
            rank_seq_max = 0;
            rank_batch_allocated = false;
        }
        if (rank_ctx != nullptr) {
            llama_free(rank_ctx);
            rank_ctx = nullptr;
        }
        if (batch_allocated) {
            llama_batch_free(batch);
            batch = {};
            batch_cap = 0;
            ctx_outputs_max = 0;
            ctx_outputs_max_per_seq = 0;
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
        compat_overlay.Reset();
        vocab = nullptr;
        slots.clear();
        slot_busy.clear();
        slot_mtp_synced.clear();
        slot_swa_guards.clear();
        swa_window = 0;
        swa_retained_slack = 0;
        swa_compact_active = false;
        slot_mtp_pending_h_pos.clear();
        slot_checkpoints.clear();
        radix_cache.Clear();
        radix_cache.restore_cost_gate().Reset();
        prefix_coalescer.Reset();
        SyncCoalescerStats();
        total_checkpoint_bytes = 0;
        checkpoint_clock = 0;
        spec_ckpt_buf.clear();
        spec_ckpt_size = 0;
        slot_clock = 0;
        requested_spec = SpeculativeType::kNone;
        active_spec = SpeculativeType::kNone;
        active_draft_max = 0;
        requested_spec_verification_mode = SpeculativeVerificationMode::TargetMatch;
        effective_spec_verification_mode = SpeculativeVerificationMode::TargetMatch;
        spec_verifier.set_mode(SpeculativeVerificationMode::TargetMatch);
        spec_verifier.reset_metrics();
        load_spec_fallback_reason.clear();
        load_draft_identity.clear();
        mtp_model_supported = false;
        ngram_mod.reset();
        suffix_matcher.reset();
        step_cost_estimator.Reset();
        batch_planner_config = BatchPlannerConfig{};
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
        const std::size_t written = TryStateSeqGetDataExt(ctx, ckpt.tgt_state.data(), sz, slot_id, flags);
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
            TryStateSeqSetDataExt(ctx, best->tgt_state.data(), best->tgt_state.size(), slot_id, flags);
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
            if (slot_idx < slot_mtp_pending_h_pos.size()) {
                slot_mtp_pending_h_pos[slot_idx] = static_cast<int64_t>(restored_tokens);
            }
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
        const std::string& session_key = slots[slot_idx].session_key;
        const std::size_t total_need = sz + dft_sz + pending_bytes;
        if (!radix_cache.ReserveBeforeAlloc(total_need, stoks, session_key, !session_key.empty())) {
            return;
        }

        RadixCacheEntry entry;
        entry.state_blob.resize(sz);
        const std::size_t written = TryStateSeqGetData(ctx, entry.state_blob.data(), sz, slot_id);
        if (written == 0 || written > sz) return;
        entry.state_blob.resize(written);
        entry.state_blob.shrink_to_fit();

#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (capture_mtp) {
            if (dft_sz > 0 && ctx_dft != nullptr) {
                entry.dft_state_blob.resize(dft_sz);
                const std::size_t dw =
                    TryStateSeqGetData(ctx_dft, entry.dft_state_blob.data(), dft_sz, slot_id);
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
        entry.session_key = session_key;
        entry.request_soft_lease = !session_key.empty();
        entry.estimated_recompute_ms = radix_cache.restore_cost_gate().PredictRecomputeMs(stoks.size());
        (void)radix_cache.Insert(stoks, std::move(entry));
    }

    // Attempts to restore a longer matching prefix from the L2 Radix host-RAM
    // cache directly into `slot_id`. Callers must snapshot any evicted VRAM
    // slot BEFORE calling this so the returned Radix entry pointer stays valid.
    // Sets `out_wiped = true` if VRAM KV for `slot_id` was cleared prior to a
    // failed restore so callers reset `lcp = 0` and slot metadata cleanly.
    bool TryRestoreSlotFromL2(std::size_t slot_idx, llama_seq_id slot_id, std::span<const Token> prompt,
                              std::string_view session_key, std::size_t min_beat_lcp, std::size_t& out_lcp,
                              bool& out_mtp_synced, bool& out_wiped) {
        out_lcp = 0;
        out_mtp_synced = false;
        out_wiped = false;
        if (!radix_cache.enabled() || ctx == nullptr || prompt.size() <= 1) return false;
        const std::size_t max_reuse = prompt.size() - 1;
        const bool allow_trim = (architecture == sonder::inference::ModelArchitecture::attention_only);
        const PinnedRadixLookup hit = radix_cache.LookupAndPin(prompt, max_reuse, allow_trim, session_key);
        const RadixCacheEntry* entry = hit.entry();
        if (!hit.found() || entry == nullptr || hit.matched_tokens() <= min_beat_lcp ||
            (hit.matched_tokens() - min_beat_lcp) < kMinL2RestoreGainTokens) {
            return false;
        }

        // Measured restore-vs-recompute cost gate: skip host L2 restore BEFORE
        // wiping VRAM whenever predicted restore transfer/deserialization cost
        // meets or exceeds estimated prefill recomputation cost for the net gain.
        const std::size_t gain_tokens = hit.matched_tokens() - min_beat_lcp;
        std::size_t restore_payload_bytes = entry->state_blob.size();
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr && !mtp_mem_shared) {
            restore_payload_bytes += entry->dft_state_blob.size();
        }
#endif
        if (!radix_cache.restore_cost_gate().should_restore(gain_tokens, restore_payload_bytes)) {
            return false;
        }

        llama_memory_t mem = llama_get_memory(ctx);
        if (mem == nullptr) return false;
        const auto restore_start = Clock::now();
        out_wiped = true;
        llama_memory_seq_rm(mem, slot_id, -1, -1);
        ResetSwaSlot(slot_idx);
        ClearSlotCheckpoints(slot_idx);

        const std::size_t read = TryStateSeqSetData(ctx, entry->state_blob.data(),
                                                    entry->state_blob.size(), slot_id);
        if (read == 0) {
            llama_memory_seq_rm(mem, slot_id, -1, -1);
            ResetSwaSlot(slot_idx);
            return false;
        }
        AdvanceSwaSlot(slot_idx, mem, slot_id, static_cast<std::uint64_t>(hit.state_tokens()));
        if (hit.state_tokens() > hit.matched_tokens()) {
            const bool swa_ok = CheckSwaRollback(slot_idx, mem, slot_id, static_cast<std::uint64_t>(hit.matched_tokens()));
            if (!swa_ok || !llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(hit.matched_tokens()), -1)) {
                llama_memory_seq_rm(mem, slot_id, -1, -1);
                ResetSwaSlot(slot_idx);
                return false;
            }
        }
        const int64_t restored_len = llama_memory_seq_pos_max(mem, slot_id) + 1;
        if (restored_len != static_cast<int64_t>(hit.matched_tokens())) {
            llama_memory_seq_rm(mem, slot_id, -1, -1);
            ResetSwaSlot(slot_idx);
            return false;
        }
        // Observe backend pos_min after any trim while preserving the monotonic
        // watermark established from `hit.state_tokens()`.
        AdvanceSwaSlot(slot_idx, mem, slot_id, static_cast<std::uint64_t>(hit.matched_tokens()));

#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (ctx_dft != nullptr && slot_idx < mtp_pending_h.size()) {
            mtp_deferred[slot_idx].Clear();
            std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
            if (slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[slot_idx] = -1;
            if (mtp_mem_shared) {
                if (hit.state_tokens() == hit.matched_tokens() &&
                    entry->mtp_pending_h.size() == mtp_pending_h[slot_idx].size()) {
                    mtp_pending_h[slot_idx] = entry->mtp_pending_h;
                    if (slot_idx < slot_mtp_pending_h_pos.size()) {
                        slot_mtp_pending_h_pos[slot_idx] = static_cast<int64_t>(hit.matched_tokens());
                    }
                }
                out_mtp_synced = true;
            } else if (!entry->dft_state_blob.empty()) {
                llama_memory_t mem_dft = llama_get_memory(ctx_dft);
                llama_memory_seq_rm(mem_dft, slot_id, -1, -1);
                const std::size_t dr = TryStateSeqSetData(ctx_dft, entry->dft_state_blob.data(),
                                                          entry->dft_state_blob.size(), slot_id);
                bool dft_ok = (dr > 0);
                if (dft_ok && hit.state_tokens() > hit.matched_tokens()) {
                    dft_ok = llama_memory_seq_rm(mem_dft, slot_id, static_cast<llama_pos>(hit.matched_tokens()), -1);
                }
                if (dft_ok && llama_memory_seq_pos_max(mem_dft, slot_id) + 1 == restored_len) {
                    if (hit.state_tokens() == hit.matched_tokens() &&
                        entry->mtp_pending_h.size() == mtp_pending_h[slot_idx].size()) {
                        mtp_pending_h[slot_idx] = entry->mtp_pending_h;
                        if (slot_idx < slot_mtp_pending_h_pos.size()) {
                            slot_mtp_pending_h_pos[slot_idx] = static_cast<int64_t>(hit.matched_tokens());
                        }
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
        const double restore_ms = MsSince(restore_start);
        radix_cache.ObserveRestore(hit, restore_payload_bytes, restore_ms);
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
                if (!TryAcceptSampler(sampler, tok)) return LLAMA_TOKEN_NULL;
                return tok;
            }
        }
#else
        (void)gpu_sampled;
#endif
        if (grammar_sampler == nullptr) {
            tok = TrySampleToken(sampler, ctx, batch_idx);
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
            if (!TryApplySampler(sampler, &cur_p)) return LLAMA_TOKEN_NULL;
            if (cur_p.selected < 0 || static_cast<std::size_t>(cur_p.selected) >= cur_p.size) {
                return LLAMA_TOKEN_NULL;
            }
            tok = cur_p.data[cur_p.selected].id;
        }

        // 2. Test ONLY the 1 candidate token against the GBNF grammar.
        llama_token_data single{tok, 1.0F, 0.0F};
        llama_token_data_array single_arr{&single, 1, -1, false};
        if (!TryApplyGrammar(grammar_sampler, &single_arr)) return LLAMA_TOKEN_NULL;
        if (std::isfinite(single.logit)) {
            if (!TryAcceptGrammar(grammar_sampler, tok)) return LLAMA_TOKEN_NULL;
            if (!TryAcceptSampler(sampler, tok)) return LLAMA_TOKEN_NULL;
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
        if (!TryApplyGrammar(grammar_sampler, &cur_p)) return LLAMA_TOKEN_NULL;
        if (!TryApplySampler(sampler, &cur_p)) return LLAMA_TOKEN_NULL;
        if (cur_p.selected < 0 || static_cast<std::size_t>(cur_p.selected) >= cur_p.size ||
            !std::isfinite(cur_p.data[cur_p.selected].logit)) {
            return LLAMA_TOKEN_NULL;
        }
        tok = cur_p.data[cur_p.selected].id;
        if (!TryAcceptGrammar(grammar_sampler, tok)) return LLAMA_TOKEN_NULL;
        if (!TryAcceptSampler(sampler, tok)) return LLAMA_TOKEN_NULL;
        if (sp.compute_logprobs && out_logprob != nullptr) {
            *out_logprob = ComputeTokenLogprobs(
                std::span<const float>(logits, static_cast<std::size_t>(n_vocab)), tok, sp.top_logprobs);
        }
        return tok;
    }

    // Extracts the post-processor target probability distribution `p` at
    // `batch_idx` (applying grammar mask + all non-terminal sampler stages in
    // `sampler`, without invoking the terminal dist/greedy selector). Used
    // only when opt-in `StochasticRejection` activates with real proposal probs.
    bool ExtractPostProcessorTargetProbs(int32_t batch_idx, const SamplingParams& sp, llama_sampler* sampler,
                                         llama_sampler* grammar_sampler, int32_t n_vocab,
                                         std::vector<double>& out_probs) {
        if (ctx == nullptr || sampler == nullptr || n_vocab <= 0) {
            return false;
        }
        const float* logits = llama_get_logits_ith(ctx, batch_idx);
        if (logits == nullptr) {
            return false;
        }
        auto& cand = grammar_candidates;
        cand.resize(static_cast<std::size_t>(n_vocab));
        for (int32_t v = 0; v < n_vocab; ++v) {
            cand[static_cast<std::size_t>(v)] = llama_token_data{v, logits[v], 0.0F};
        }
        llama_token_data_array cur_p{cand.data(), cand.size(), -1, false};
        if (grammar_sampler != nullptr && !TryApplyGrammar(grammar_sampler, &cur_p)) {
            return false;
        }
        const int32_t n_stages = llama_sampler_chain_n(sampler);
        // Apply all processor stages (logit_bias, penalties, top_k, typical_p,
        // top_p, min_p, temp) while skipping the final greedy/dist selector.
        const int32_t n_processors = n_stages > 0 ? (n_stages - 1) : 0;
        for (int32_t s_idx = 0; s_idx < n_processors; ++s_idx) {
            if (llama_sampler* stage = llama_sampler_chain_get(sampler, s_idx); stage != nullptr) {
                if (!TryApplySampler(stage, &cur_p)) return false;
            }
        }
        if (cur_p.size == 0) {
            return false;
        }
        out_probs.assign(static_cast<std::size_t>(n_vocab), 0.0);
        if (sp.temperature <= 0.0F) {
            int32_t best_id = -1;
            float best_val = -std::numeric_limits<float>::infinity();
            for (std::size_t i = 0; i < cur_p.size; ++i) {
                const llama_token_data& td = cur_p.data[i];
                if (td.id < 0 || td.id >= n_vocab || !std::isfinite(td.logit)) continue;
                if (best_id < 0 || td.logit > best_val) {
                    best_val = td.logit;
                    best_id = td.id;
                }
            }
            if (best_id < 0) return false;
            out_probs[static_cast<std::size_t>(best_id)] = 1.0;
            return true;
        }

        double max_logit = -std::numeric_limits<double>::infinity();
        bool any_finite = false;
        for (std::size_t i = 0; i < cur_p.size; ++i) {
            const llama_token_data& td = cur_p.data[i];
            if (td.id < 0 || td.id >= n_vocab || !std::isfinite(td.logit)) continue;
            const double val = static_cast<double>(td.logit);
            if (!any_finite || val > max_logit) {
                max_logit = val;
                any_finite = true;
            }
        }
        if (!any_finite) return false;

        double sum_exp = 0.0;
        for (std::size_t i = 0; i < cur_p.size; ++i) {
            const llama_token_data& td = cur_p.data[i];
            if (td.id < 0 || td.id >= n_vocab || !std::isfinite(td.logit)) continue;
            const double w = std::exp(static_cast<double>(td.logit) - max_logit);
            out_probs[static_cast<std::size_t>(td.id)] = w;
            sum_exp += w;
        }
        if (!(sum_exp > 0.0) || !std::isfinite(sum_exp)) return false;
        const double inv_sum = 1.0 / sum_exp;
        for (std::size_t i = 0; i < cur_p.size; ++i) {
            const llama_token_data& td = cur_p.data[i];
            if (td.id >= 0 && td.id < n_vocab) {
                out_probs[static_cast<std::size_t>(td.id)] *= inv_sum;
            }
        }
        return true;
    }

    // Verifies a speculative draft batch of length `k >= 1` using either the
    // default `TargetMatch` reference path (always used for deterministic
    // suffix/n-gram and greedy MTP proposals) or opt-in `StochasticRejection`
    // (activated only when configured AND `proposal_dists` carries real q_j).
    // Returns false on sampler/grammar deadlock or invalid distribution.
    bool VerifySpeculativeBatch(std::span<const Token> draft_tokens,
                                std::span<const std::vector<double>> proposal_dists, const SamplingParams& sp,
                                llama_sampler* sampler, llama_sampler* grammar_sampler, bool gpu_sampled,
                                int32_t n_vocab, std::mt19937_64& verify_rng, std::vector<Token>& accepted_tokens,
                                std::vector<TokenLogprobInfo>& accepted_lps, std::size_t& n_accepted,
                                GenerateResult& result) {
        const std::size_t k = draft_tokens.size();
        accepted_tokens.clear();
        accepted_lps.clear();
        n_accepted = 0;
        if (!proposal_dists.empty() && proposal_dists.size() != k) {
            return false;
        }

        const SpeculativeVerificationMode step_mode = sonder::inference::resolve_speculative_verification_mode(
            requested_spec_verification_mode, !proposal_dists.empty() && k > 0u);
        result.spec_verifier_mode = step_mode;

        if (step_mode == SpeculativeVerificationMode::StochasticRejection) {
            spec_verifier.set_mode(SpeculativeVerificationMode::StochasticRejection);
            bool grammar_accept_failed = false;
            const sonder::inference::SpeculativeSequenceResult seq = spec_verifier.verify_sequence_lazy(
                draft_tokens, /*has_proposal_dists=*/true,
                [&](std::size_t pos, std::vector<double>& out_p) -> bool {
                    // A failed grammar accept leaves the grammar with no stacks; applying it again would
                    // reach GGML_ASSERT(!stacks.empty()) in llama_grammar_reject_candidates, which aborts
                    // the process and cannot be caught. Stop the sequence instead.
                    if (grammar_accept_failed) {
                        return false;
                    }
                    return ExtractPostProcessorTargetProbs(static_cast<int32_t>(pos), sp, sampler, grammar_sampler,
                                                           n_vocab, out_p);
                },
                [&](std::size_t pos, std::vector<double>& out_q) -> bool {
                    out_q = proposal_dists[pos];
                    return !out_q.empty();
                },
                [&](std::size_t pos, std::int32_t committed_tok) {
                    if (grammar_sampler != nullptr && !TryAcceptGrammar(grammar_sampler, committed_tok)) {
                        grammar_accept_failed = true;
                    }
                    if (!TryAcceptSampler(sampler, committed_tok)) {
                        grammar_accept_failed = true;
                    }
                    TokenLogprobInfo lp;
                    if (sp.compute_logprobs) {
                        if (const float* logits = llama_get_logits_ith(ctx, static_cast<int32_t>(pos));
                            logits != nullptr) {
                            lp = ComputeTokenLogprobs(
                                std::span<const float>(logits, static_cast<std::size_t>(n_vocab)), committed_tok,
                                sp.top_logprobs);
                        }
                    }
                    accepted_lps.push_back(std::move(lp));
                },
                verify_rng);
            if (!seq.ok || grammar_accept_failed) {
                return false;
            }
            accepted_tokens = seq.emitted_tokens;
            n_accepted = seq.accepted_draft_tokens;
            if (seq.used_residual_resample) {
                ++result.spec_residual_resamples;
            }
            result.spec_verify_overhead_us += seq.verify_overhead_us;
            return true;
        }

        // Default reference path: exact target-sample matching (preserves GPU
        // backend sampling and fast 1-token grammar rejection with zero extra
        // distribution-extraction overhead).
        const auto verify_start = Clock::now();
        for (std::size_t j = 0; j < k; ++j) {
            TokenLogprobInfo lp;
            const Token sampled = SampleTokenAt(static_cast<int32_t>(j), sp, sampler, grammar_sampler, gpu_sampled,
                                                n_vocab, sp.compute_logprobs ? &lp : nullptr);
            if (sampled == LLAMA_TOKEN_NULL) {
                return false;
            }
            accepted_tokens.push_back(sampled);
            accepted_lps.push_back(std::move(lp));
            if (sampled == draft_tokens[j]) {
                ++n_accepted;
            } else {
                break;
            }
        }
        bool sampled_bonus = false;
        if (n_accepted == k) {
            TokenLogprobInfo bonus_lp;
            const Token bonus = SampleTokenAt(static_cast<int32_t>(k), sp, sampler, grammar_sampler, gpu_sampled,
                                              n_vocab, sp.compute_logprobs ? &bonus_lp : nullptr);
            if (bonus == LLAMA_TOKEN_NULL) {
                return false;
            }
            accepted_tokens.push_back(bonus);
            accepted_lps.push_back(std::move(bonus_lp));
            sampled_bonus = true;
        }
        const double verify_us =
            std::chrono::duration<double, std::micro>(Clock::now() - verify_start).count();
        const double safe_us = (std::isfinite(verify_us) && verify_us > 0.0) ? verify_us : 0.0;
        result.spec_verify_overhead_us += safe_us;
        spec_verifier.record_external_step(SpeculativeVerificationMode::TargetMatch, k, n_accepted,
                                           n_accepted < k, sampled_bonus, safe_us);
        return true;
    }

#if defined(SONDER_LLAMACPP_HAS_MTMD)
    Status EvaluateMultimodalPrompt(const GenerateRequest& request, llama_seq_id slot_id, int32_t n_batch,
                                    int64_t n_ctx, int64_t& out_cur_pos, int32_t& out_prompt_tokens) {
        out_cur_pos = 0;
        out_prompt_tokens = 0;
        if (Status s = ValidateMultimodalRequest(request, mtmd_ctx != nullptr, true); !s.ok()) {
            return s;
        }
        if (Status s = ValidateSeqId(slot_id, std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(slots.size())));
            !s.ok()) {
            return s;
        }
        return RunGuarded(ErrorCode::kDecodeFailed, "mtmd_eval", [&]() -> Status {
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

        const std::size_t total_tokens = mtmd_helper_get_n_tokens(chunks);
        const llama_pos total_pos = mtmd_helper_get_n_pos(chunks);
        if (Status s = ValidateMultimodalEvalBounds(static_cast<int64_t>(total_tokens),
                                                    static_cast<int64_t>(total_pos), n_ctx, n_batch);
            !s.ok()) {
            return s;
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
            std::min<std::size_t>(total_tokens, static_cast<std::size_t>(INT32_MAX)));
        return Status::Ok();
        });
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
        try {
            ctx_dft = llama_init_from_model(model, cp_dft);
        } catch (...) {
            ctx_dft = nullptr;
        }
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
            try {
                SamplerPtr s(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                llama_sampler_chain_add(s.get(), llama_sampler_init_greedy());
                mtp_samplers[i] = std::move(s);
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
                SamplerPtr bchain(llama_sampler_chain_init(llama_sampler_chain_default_params()));
                llama_sampler_chain_add(bchain.get(), llama_sampler_init_greedy());
                if (TrySetSampler(ctx_dft, static_cast<llama_seq_id>(i), bchain.get())) {
                    mtp_backend_chains[i] = bchain.release();
                }
#endif
            } catch (...) {
                ReleaseMtp();
                return false;
            }
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
            if (TryDecode(ctx_dft, dft_batch) != 0) {
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
                if (TryDecode(ctx_dft, dft_batch) != 0) {
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
        if (sidx < slot_mtp_pending_h_pos.size()) {
            slot_mtp_pending_h_pos[sidx] = static_cast<int64_t>(batch_in.pos[n_tokens - 1]) + 1;
        }
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
            if (sidx < slot_mtp_pending_h_pos.size()) {
                slot_mtp_pending_h_pos[sidx] = static_cast<int64_t>(batch_in.pos[n_valid - 1]) + 1;
            }
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
                if (TryDecode(ctx_dft, dft_batch) != 0) {
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
        if (sidx < slot_mtp_pending_h_pos.size()) {
            slot_mtp_pending_h_pos[sidx] = static_cast<int64_t>(batch_in.pos[n_valid - 1]) + 1;
        }
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
            if (TryDecode(ctx_dft, dft_batch) != 0) {
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
                id = TrySampleToken(mtp_samplers[sidx].get(), ctx_dft, i_last);
                if (id == LLAMA_TOKEN_NULL) break;
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
    const auto captured = DeviceLineRegistry().Snapshot();
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
        // Attach parsed ggml_vulkan device telemetry when available.
        if (info.name.rfind("Vulkan", 0) == 0 && info.name.size() > 6) {
            const std::string_view suffix(info.name.data() + 6, info.name.size() - 6);
            if (const auto idx = sonder::inference::parse_strict_u64(suffix)) {
                for (const auto& line : captured) {
                    if (line.backend == "vulkan" && line.index.has_value() &&
                        static_cast<std::uint64_t>(*line.index) == *idx) {
                        info.telemetry = line;
                        break;
                    }
                }
            }
        }
        out.push_back(std::move(info));
    }
    return out;
}

std::vector<sonder::inference::GgmlDeviceTelemetry> LlamaCppBackend::CapturedGgmlDevices() {
    return DeviceLineRegistry().Snapshot();
}

void LlamaCppBackend::ClearCapturedGgmlDevices() noexcept {
    DeviceLineRegistry().Clear();
    sonder::inference::clear_registered_ggml_device_telemetry();
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

    std::string compat_err;
    llama_model* model = nullptr;
    if (Status s = RunGuarded(ErrorCode::kLoadFailed, "llama_model_load_from_file", [&] {
            model = LoadModelWithGgufCompat(options.model_path, mp, impl_->compat_overlay, &compat_err);
            return Status::Ok();
        });
        !s.ok()) {
        return s;
    }
    if (model == nullptr) {
        return Status::Error(
            ErrorCode::kLoadFailed,
            compat_err.empty() ? ("llama.cpp failed to load: " + options.model_path)
                               : ("llama.cpp failed to load: " + options.model_path + " (" + compat_err + ")"));
    }
    if (impl_->compat_overlay.active() && !impl_->compat_overlay.inspection().detail.empty()) {
        const std::string line = "gguf_compat: " + impl_->compat_overlay.inspection().detail + "\n";
        RouteLog(LogLevel::kInfo, line.c_str());
    }
    impl_->model = model;
    impl_->vocab = llama_model_get_vocab(model);
    impl_->options = options;
    impl_->architecture = ReadModelArchitecture(model);

    if (!options.vocab_only) {
        std::uint64_t model_n_swa = 0;
#if defined(SONDER_LLAMACPP_HAS_MODEL_N_SWA)
        if (const int32_t raw_swa = llama_model_n_swa(model); raw_swa > 0) {
            model_n_swa = static_cast<std::uint64_t>(raw_swa);
        }
#endif
#if defined(SONDER_LLAMACPP_HAS_SWA_FULL)
        const bool swa_qualified = (model_n_swa > 0);
#else
        const bool swa_qualified = false;
#endif
        llama_context_params cp = MakeContextParams(options, swa_qualified);
        cp.abort_callback = &Impl::AbortCallback;
        cp.abort_callback_data = impl_.get();
        if (Status s = RunGuarded(ErrorCode::kLoadFailed, "llama_init_from_model", [&] {
                impl_->ctx = llama_init_from_model(model, cp);
                return Status::Ok();
            });
            !s.ok()) {
            impl_->Release();
            return s;
        }
        if (impl_->ctx == nullptr) {
            impl_->Release();
            return Status::Error(ErrorCode::kLoadFailed, "llama.cpp failed to create context");
        }

        // Pre-allocate reusable decode batch sized to the context's n_batch.
        const auto cap = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_batch(impl_->ctx)));
        impl_->batch = llama_batch_init(cap, /*embd=*/0, /*n_seq_max=*/1);
        impl_->batch_cap = cap;
#if defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
        impl_->ctx_outputs_max =
            cp.n_outputs_max > 0 ? std::min<int32_t>(cap, static_cast<int32_t>(cp.n_outputs_max)) : cap;
        impl_->ctx_outputs_max_per_seq =
            cp.n_outputs_max_per_seq > 0
                ? std::min<int32_t>(impl_->ctx_outputs_max, static_cast<int32_t>(cp.n_outputs_max_per_seq))
                : impl_->ctx_outputs_max;
#else
        impl_->ctx_outputs_max = cap;
        impl_->ctx_outputs_max_per_seq = cap;
#endif
        impl_->batch_allocated = true;

        // Initialize per-slot KV prefix cache metadata & L2 host-RAM Radix cache.
        const std::uint32_t n_slots = std::clamp<uint32_t>(options.n_slots, 1, kMaxSlots);
        impl_->slots.resize(n_slots);
        impl_->slot_busy.assign(n_slots, 0u);
        impl_->slot_mtp_synced.assign(n_slots, false);
        impl_->swa_window = model_n_swa;
        impl_->swa_retained_slack = static_cast<std::uint64_t>(llama_n_ubatch(impl_->ctx));
        impl_->swa_compact_active =
            (options.swa_kv_mode == sonder::inference::SwaKvMode::CompactIfQualified) && swa_qualified;
        impl_->slot_swa_guards.resize(n_slots);
        for (auto& guard : impl_->slot_swa_guards) {
            guard.Configure(options.swa_kv_mode, impl_->swa_window, impl_->swa_retained_slack, swa_qualified);
        }
        impl_->slot_mtp_pending_h_pos.assign(n_slots, -1);
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
        PrefixCoalescerConfig coalesce_cfg;
        coalesce_cfg.enabled = options.prefix_cache && options.prefix_coalesce && (n_slots > 1);
        coalesce_cfg.architecture = impl_->architecture;
        coalesce_cfg.has_valid_checkpoint = false;
        coalesce_cfg.min_coalesce_prefix_tokens = options.min_coalesce_prefix_tokens;
        coalesce_cfg.max_followers_per_producer = options.max_followers_per_producer;
        coalesce_cfg.max_coalesce_wait_ms = options.max_coalesce_wait_ms;
        impl_->prefix_coalescer.Configure(coalesce_cfg);
        impl_->SyncCoalescerStats();

#if defined(SONDER_LLAMACPP_HAS_MTMD)
        if (!options.mmproj_path.empty()) {
            mtmd_context_params mp_mtmd = mtmd_context_params_default();
            mp_mtmd.use_gpu = (options.n_gpu_layers != 0);
            mp_mtmd.print_timings = false;
            if (options.n_threads > 0) {
                mp_mtmd.n_threads = options.n_threads;
            }
            mp_mtmd.flash_attn_type = ToLlamaFlashAttn(options.flash_attn);
            if (Status s = RunGuarded(ErrorCode::kLoadFailed, "mtmd_init_from_file", [&] {
                    impl_->mtmd_ctx = mtmd_init_from_file(options.mmproj_path.c_str(), model, mp_mtmd);
                    return Status::Ok();
                });
                !s.ok()) {
                impl_->Release();
                return s;
            }
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
        impl_->requested_spec = requested_spec;
        impl_->active_spec = SpeculativeType::kNone;
        impl_->load_spec_fallback_reason.clear();
        impl_->load_draft_identity.clear();
        impl_->mtp_model_supported = false;

        if (requested_spec == SpeculativeType::kDraftMtp || requested_spec == SpeculativeType::kAuto) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
            const int32_t n_nextn = llama_model_n_layer_nextn(model);
            if (n_nextn > 0) {
                impl_->mtp_model_supported = true;
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
                impl_->load_spec_fallback_reason = "auto_fallback_ngram_mod";
            } else if (impl_->active_spec == SpeculativeType::kNone && requested_spec == SpeculativeType::kDraftMtp) {
                impl_->load_spec_fallback_reason = "not_supported";
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
            impl_->load_draft_identity = "mtp:native+suffix";
        } else if (impl_->active_spec == SpeculativeType::kSuffix) {
            impl_->active_draft_max = options.spec_draft_max > 0 ? options.spec_draft_max : kDefaultSuffixDraftMax;
            impl_->suffix_matcher = std::make_unique<SuffixMatcher>();
            impl_->load_draft_identity = "suffix";
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
            impl_->load_draft_identity = "ngram-mod";
        } else if (impl_->active_spec == SpeculativeType::kNgram) {
            impl_->active_draft_max = options.spec_draft_max > 0 ? options.spec_draft_max : kDefaultNgramDraftMax;
            impl_->ngram_matcher = NgramMatcher(/*exact_scan_fallback=*/true);
            impl_->load_draft_identity = "ngram";
        } else {
            impl_->active_draft_max = 0;
        }

        impl_->clock_origin = Clock::now();
        impl_->step_cost_estimator.Reset();
        impl_->batch_planner_config = BatchPlannerConfig{};
        impl_->batch_planner_config.default_foreground_itl_ms = options.default_foreground_itl_ms;

        if (n_slots > 1) {
            impl_->StartBatchWorker();
        }
    }

    impl_->requested_spec_verification_mode = options.spec_verification_mode;
    impl_->effective_spec_verification_mode = ResolveSpeculativeVerificationMode(
        options.spec_verification_mode, impl_->active_spec, /*proposer_has_probabilities=*/false);
    impl_->spec_verifier.set_mode(impl_->requested_spec_verification_mode);
    impl_->spec_verifier.reset_metrics();

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

bool LlamaCppBackend::HasMtpBuildSupport() noexcept {
#if defined(SONDER_LLAMACPP_HAS_MTP)
    return true;
#else
    return false;
#endif
}

bool LlamaCppBackend::HasMtmdBuildSupport() noexcept {
#if defined(SONDER_LLAMACPP_HAS_MTMD)
    return true;
#else
    return false;
#endif
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

SpeculativeVerificationMode LlamaCppBackend::RequestedSpecVerificationMode() const noexcept {
    return impl_ ? impl_->requested_spec_verification_mode : SpeculativeVerificationMode::TargetMatch;
}

SpeculativeVerificationMode LlamaCppBackend::EffectiveSpecVerificationMode() const noexcept {
    return impl_ ? impl_->effective_spec_verification_mode : SpeculativeVerificationMode::TargetMatch;
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

SpeculativeLoadInfo LlamaCppBackend::GetSpeculativeInfo() const {
    SpeculativeLoadInfo info;
    info.mtp_build_supported = HasMtpBuildSupport();
    if (!IsLoaded()) return info;
    info.configured_method = std::string(ToString(impl_->requested_spec));
    info.effective_method = std::string(ToString(impl_->active_spec));
    if (impl_->active_spec == SpeculativeType::kDraftMtp && impl_->suffix_matcher != nullptr) {
        info.effective_method = "draft-mtp+suffix";
    }
    info.configured_draft_max =
        impl_->options.spec_draft_max > 0 ? impl_->options.spec_draft_max : impl_->active_draft_max;
    info.effective_draft_max = impl_->active_draft_max;
    info.fallback_reason = impl_->load_spec_fallback_reason;
    info.draft_identity = impl_->load_draft_identity;
    info.mtp_model_supported = impl_->mtp_model_supported;
    info.l2_cache_enabled = impl_->radix_cache.enabled();
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
    return RunGuarded(ErrorCode::kTokenizeFailed, "llama_tokenize", [&]() -> Status {
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
    });
}

Status LlamaCppBackend::ProbeGrammarMask(const SamplingParams& params, std::span<const Token> accepted,
                                         std::vector<std::uint8_t>& allowed) const {
    allowed.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    if (Status v = ValidateTokenIds(accepted, n_vocab, "accepted"); !v.ok()) return v;
    SamplerPtr chain;
    SamplerPtr grammar;
    if (Status s = BuildSampler(params, impl_->vocab, n_vocab, 0, chain, grammar); !s.ok()) return s;
    allowed.assign(static_cast<std::size_t>(n_vocab), 1);
    if (!grammar) return Status::Ok();
    for (const Token t : accepted) {
        if (!TryAcceptPrefillToken(grammar.get(), t, n_vocab)) {
            allowed.clear();
            return Status::Error(ErrorCode::kDecodeFailed, "accepted tokens violate the grammar");
        }
    }
    std::vector<llama_token_data> cand(static_cast<std::size_t>(n_vocab));
    for (int32_t i = 0; i < n_vocab; ++i) cand[static_cast<std::size_t>(i)] = llama_token_data{i, 0.0F, 0.0F};
    llama_token_data_array arr{cand.data(), cand.size(), -1, false};
    if (!TryApplyGrammar(grammar.get(), &arr)) {
        allowed.clear();
        return Status::Error(ErrorCode::kDecodeFailed, "grammar sampler failed to apply");
    }
    for (std::size_t i = 0; i < arr.size; ++i) {
        const auto id = static_cast<std::size_t>(arr.data[i].id);
        if (id < allowed.size()) allowed[id] = arr.data[i].logit == -std::numeric_limits<float>::infinity() ? 0 : 1;
    }
    // Candidates the grammar removed outright (not just set to -inf) are masked too.
    if (arr.size < cand.size()) {
        std::vector<std::uint8_t> kept(allowed.size(), 0);
        for (std::size_t i = 0; i < arr.size; ++i) {
            const auto id = static_cast<std::size_t>(arr.data[i].id);
            if (id < kept.size()) kept[id] = allowed[id];
        }
        allowed.swap(kept);
    }
    return Status::Ok();
}

Status LlamaCppBackend::TokenToPiece(Token token, std::string& out) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    const Token one[1] = {token};
    if (Status s = ValidateTokenIds(one, llama_vocab_n_tokens(impl_->vocab)); !s.ok()) return s;
    return RunGuarded(ErrorCode::kTokenizeFailed, "llama_token_to_piece", [&]() -> Status {
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
    });
}

Status LlamaCppBackend::Detokenize(std::span<const Token> tokens, std::string& out, bool remove_special,
                                   bool unparse_special) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (tokens.empty()) return Status::Ok();
    if (tokens.size() > static_cast<std::size_t>(INT32_MAX)) {
        return Status::Error(ErrorCode::kInvalidArgument, "token span too long");
    }
    if (Status s = ValidateTokenIds(tokens, llama_vocab_n_tokens(impl_->vocab)); !s.ok()) return s;
    const auto n_tok = static_cast<int32_t>(tokens.size());
    return RunGuarded(ErrorCode::kTokenizeFailed, "llama_detokenize", [&]() -> Status {
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
    });
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
    return RunGuarded(
        ErrorCode::kInvalidArgument, "llama_chat_apply_template",
        [&]() -> Status {
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
        },
        ErrorCode::kTokenizeFailed);
}

void LlamaCppBackend::SetTelemetrySink(TelemetrySink sink) { impl_->telemetry = std::move(sink); }

PrefixCoalescerStats LlamaCppBackend::GetPrefixCoalescerStats() const noexcept {
    PrefixCoalescerStats out;
    if (!impl_) return out;
    out.coalesced_groups = impl_->stat_coalesced_groups.load(std::memory_order_relaxed);
    out.coalesced_followers = impl_->stat_coalesced_followers.load(std::memory_order_relaxed);
    out.coalesced_tokens_saved = impl_->stat_coalesced_tokens_saved.load(std::memory_order_relaxed);
    out.coalesce_fallbacks_timeout = impl_->stat_coalesce_fallbacks_timeout.load(std::memory_order_relaxed);
    out.coalesce_fallbacks_producer_failed =
        impl_->stat_coalesce_fallbacks_producer_failed.load(std::memory_order_relaxed);
    return out;
}

Status LlamaCppBackend::CaptureSequenceState(std::vector<std::uint8_t>& out_bytes, std::size_t max_bytes) {
    out_bytes.clear();
    if (max_bytes == 0) return Status::Error(ErrorCode::kInvalidArgument, "max_bytes must be > 0");
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return Status::Error(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");

    return RunGuarded(ErrorCode::kDecodeFailed, "llama_state_seq_get_data", [&]() -> Status {
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
    });
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
    return RunGuarded(ErrorCode::kDecodeFailed, "llama_state_seq_set_data", [&]() -> Status {
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
    });
}

namespace {

enum llama_pooling_type ToLlamaPooling(PoolingType pooling) noexcept {
    switch (pooling) {
        case PoolingType::kUnspecified: return LLAMA_POOLING_TYPE_UNSPECIFIED;
        case PoolingType::kNone: return LLAMA_POOLING_TYPE_NONE;
        case PoolingType::kMean: return LLAMA_POOLING_TYPE_MEAN;
        case PoolingType::kCls: return LLAMA_POOLING_TYPE_CLS;
        case PoolingType::kLast: return LLAMA_POOLING_TYPE_LAST;
        case PoolingType::kRank: return LLAMA_POOLING_TYPE_RANK;
    }
    return LLAMA_POOLING_TYPE_UNSPECIFIED;
}

PoolingType FromLlamaPooling(enum llama_pooling_type pooling) noexcept {
    switch (pooling) {
        case LLAMA_POOLING_TYPE_NONE: return PoolingType::kNone;
        case LLAMA_POOLING_TYPE_MEAN: return PoolingType::kMean;
        case LLAMA_POOLING_TYPE_CLS: return PoolingType::kCls;
        case LLAMA_POOLING_TYPE_LAST: return PoolingType::kLast;
        case LLAMA_POOLING_TYPE_RANK: return PoolingType::kRank;
        case LLAMA_POOLING_TYPE_UNSPECIFIED:
        default: return PoolingType::kUnspecified;
    }
}

std::int32_t QueryModelEmbeddingDim(const llama_model* model) noexcept {
    if (model == nullptr) return 0;
#if defined(SONDER_LLAMACPP_HAS_N_EMBD_OUT)
    int32_t n_embd = llama_model_n_embd_out(model);
    if (n_embd <= 0) {
        n_embd = llama_model_n_embd(model);
    }
    return n_embd;
#else
    return llama_model_n_embd(model);
#endif
}

}  // namespace

bool LlamaCppBackend::SupportsReranking() const noexcept {
    if (!IsLoaded()) return false;
    if (impl_->ctx != nullptr && llama_pooling_type(impl_->ctx) == LLAMA_POOLING_TYPE_RANK) {
        return true;
    }
    if (impl_->model != nullptr) {
        if (llama_model_chat_template(impl_->model, "rerank") != nullptr) {
            return true;
        }
        if (llama_model_cls_label(impl_->model, 0) != nullptr) {
            return true;
        }
    }
    return false;
}

std::int32_t LlamaCppBackend::EmbeddingDimensions() const noexcept {
    if (!IsLoaded()) return 0;
    return std::max<std::int32_t>(0, QueryModelEmbeddingDim(impl_->model));
}

Status LlamaCppBackend::FormatRerankPair(std::string_view query, std::string_view document,
                                         std::vector<Token>& out) const {
    out.clear();
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->vocab == nullptr) return Status::Error(ErrorCode::kNotLoaded, "model vocab is unavailable");

    const char* rerank_prompt = llama_model_chat_template(impl_->model, "rerank");
    if (rerank_prompt != nullptr) {
        const auto segments = SplitRerankTemplate(rerank_prompt, query, document);
        bool first_non_empty = true;
        std::vector<Token> piece;
        for (const auto& seg : segments) {
            if (seg.text.empty()) continue;
            piece.clear();
            const bool add_special = first_non_empty;
            first_non_empty = false;
            if (Status s = Tokenize(seg.text, add_special, piece, seg.parse_special); !s.ok()) {
                out.clear();
                return s;
            }
            out.insert(out.end(), piece.begin(), piece.end());
        }
        if (out.empty()) {
            return Status::Error(ErrorCode::kInvalidArgument, "rerank pair produced no tokens");
        }
        return Status::Ok();
    }

    std::vector<Token> query_tokens;
    std::vector<Token> doc_tokens;
    if (!query.empty()) {
        if (Status s = Tokenize(query, /*add_special=*/false, query_tokens, /*parse_special=*/false); !s.ok()) {
            return s;
        }
    }
    if (!document.empty()) {
        if (Status s = Tokenize(document, /*add_special=*/false, doc_tokens, /*parse_special=*/false); !s.ok()) {
            return s;
        }
    }

    const llama_vocab* vocab = impl_->vocab;
    const llama_token bos_token = llama_vocab_bos(vocab);
    const llama_token sep_token = llama_vocab_sep(vocab);
    llama_token eos_token = llama_vocab_eos(vocab);
    if (eos_token == LLAMA_TOKEN_NULL) {
        eos_token = sep_token;
    }

    out.reserve(query_tokens.size() + doc_tokens.size() + 4u);
    if (llama_vocab_get_add_bos(vocab) && bos_token != LLAMA_TOKEN_NULL) {
        out.push_back(bos_token);
    }
    out.insert(out.end(), query_tokens.begin(), query_tokens.end());
    if (llama_vocab_get_add_eos(vocab) && eos_token != LLAMA_TOKEN_NULL) {
        out.push_back(eos_token);
    }
    if (llama_vocab_get_add_sep(vocab) && sep_token != LLAMA_TOKEN_NULL) {
        out.push_back(sep_token);
    }
    out.insert(out.end(), doc_tokens.begin(), doc_tokens.end());
    if (llama_vocab_get_add_eos(vocab) && eos_token != LLAMA_TOKEN_NULL) {
        out.push_back(eos_token);
    }
    if (out.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "rerank pair produced no tokens");
    }
    return Status::Ok();
}

Status LlamaCppBackend::Embed(const std::vector<std::vector<Token>>& inputs, bool normalize, EmbedOutput& out,
                              const CancelPredicate& should_cancel) {
    EmbedOptions opts;
    opts.normalize = normalize;
    return Embed(inputs, opts, out, should_cancel);
}

Status LlamaCppBackend::Embed(const std::vector<std::vector<Token>>& inputs, const EmbedOptions& options,
                              EmbedOutput& out, const CancelPredicate& should_cancel) {
    out = {};
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return Status::Error(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");
    if (inputs.empty()) return Status::Error(ErrorCode::kInvalidArgument, "embedding inputs are empty");
    if (options.pooling == PoolingType::kRank) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "reranking/classification pooling is not supported for dense embeddings");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    const int32_t n_embd = QueryModelEmbeddingDim(impl_->model);
    if (n_embd <= 0) {
        return Status::Error(ErrorCode::kDecodeFailed, "model reports invalid embedding dimension");
    }

    // Fail closed on dimensions: never silently truncate arbitrary models.
    if (options.dimensions.has_value()) {
        const std::size_t req_dims = *options.dimensions;
        if (req_dims == 0) {
            return Status::Error(ErrorCode::kInvalidArgument, "embedding dimensions must be positive");
        }
        if (req_dims > static_cast<std::size_t>(n_embd)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "requested dimensions (" + std::to_string(req_dims) +
                                     ") exceeds native embedding dimension (" + std::to_string(n_embd) + ")");
        }
        if (req_dims < static_cast<std::size_t>(n_embd)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "model does not support dimensions reduction below native dimension " +
                                     std::to_string(n_embd));
        }
    }

    const uint32_t base_ctx = std::max<uint32_t>(256u, Impl::EffectiveCtxSeq(impl_->ctx));
    std::vector<std::size_t> lengths;
    lengths.reserve(inputs.size());
    for (const auto& seq : inputs) {
        if (seq.empty()) return Status::Error(ErrorCode::kInvalidArgument, "embedding input is empty");
        for (Token t : seq) {
            if (t < 0 || t >= n_vocab) {
                return Status::Error(ErrorCode::kInvalidArgument, "embedding token out of range");
            }
        }
        lengths.push_back(seq.size());
    }

    const uint32_t configured_budget =
        options.max_batch_tokens > 0
            ? options.max_batch_tokens
            : (impl_->options.embed_batch_tokens > 0 ? impl_->options.embed_batch_tokens
                                                     : std::min<uint32_t>(base_ctx, kDefaultEmbedBatchTokens));
    uint32_t configured_seqs =
        options.max_sequences > 0
            ? options.max_sequences
            : (impl_->options.embed_max_sequences > 0 ? impl_->options.embed_max_sequences
                                                      : kDefaultEmbedMaxSequences);
    configured_seqs = std::clamp<uint32_t>(configured_seqs, 1u, kMaxEmbedSequences);
#if !defined(SONDER_LLAMACPP_HAS_KV_UNIFIED)
    // Without kv_unified, llama.cpp partitions n_ctx across n_seq_max; keep single-seq fallback.
    configured_seqs = 1u;
#endif

    EmbedBatchPlanConfig plan_cfg;
    plan_cfg.max_batch_tokens = std::max<std::size_t>(1u, static_cast<std::size_t>(configured_budget));
    plan_cfg.max_sequences_per_batch = static_cast<std::size_t>(configured_seqs);
    plan_cfg.max_single_input_tokens = static_cast<std::size_t>(base_ctx);
    plan_cfg.allow_oversized_single_batch = true;

    const EmbedBatchPlan plan = PlanEmbedBatches(lengths, plan_cfg);
    if (!plan.ok) {
        return Status::Error(plan.context_overflow ? ErrorCode::kContextOverflow : ErrorCode::kInvalidArgument,
                             plan.error);
    }

    // Lazily allocate an isolated multi-sequence embedding context sized to
    // the planned batch budget so Embed() never disturbs live KV prefix caches
    // or MTP/backend-sampler graphs.
    const enum llama_pooling_type requested_pooling =
        options.pooling == PoolingType::kUnspecified
            ? (llama_model_has_encoder(impl_->model) ? LLAMA_POOLING_TYPE_UNSPECIFIED : LLAMA_POOLING_TYPE_MEAN)
            : ToLlamaPooling(options.pooling);
    const uint32_t aligned_need =
        static_cast<uint32_t>(((plan.max_batch_tokens_used + 255u) / 256u) * 256u);
    const uint32_t want_ctx = std::max<uint32_t>(256u, aligned_need);
    const uint32_t want_seqs = static_cast<uint32_t>(std::max<std::size_t>(1u, plan.max_batch_sequences_used));
    const bool need_recreate = impl_->embed_ctx == nullptr ||
                               llama_n_ctx(impl_->embed_ctx) < want_ctx ||
                               impl_->embed_seq_max < want_seqs ||
                               impl_->embed_batch_cap < static_cast<int32_t>(want_ctx) ||
                               impl_->embed_pooling_requested != requested_pooling;
    if (need_recreate && impl_->embed_ctx != nullptr) {
        if (impl_->embed_batch_allocated) {
            llama_batch_free(impl_->embed_batch);
            impl_->embed_batch = {};
            impl_->embed_batch_cap = 0;
            impl_->embed_seq_max = 0;
            impl_->embed_batch_allocated = false;
        }
        llama_free(impl_->embed_ctx);
        impl_->embed_ctx = nullptr;
    }
    if (impl_->embed_ctx == nullptr) {
        llama_context_params cp_emb = llama_context_default_params();
        cp_emb.n_ctx = want_ctx;
        // Keep n_batch == n_ubatch == want_ctx so non-causal encoder models and
        // multi-sequence batches evaluate in a single forward pass.
        cp_emb.n_batch = want_ctx;
        cp_emb.n_ubatch = want_ctx;
        cp_emb.n_seq_max = want_seqs;
        cp_emb.embeddings = true;
        cp_emb.pooling_type = requested_pooling;
        cp_emb.type_k = ToGgmlType(impl_->options.type_k);
        cp_emb.type_v = ToGgmlType(impl_->options.type_v);
        cp_emb.flash_attn_type = ToLlamaFlashAttn(impl_->options.flash_attn);
        cp_emb.offload_kqv = true;
        cp_emb.no_perf = true;
#if defined(SONDER_LLAMACPP_HAS_KV_UNIFIED)
        cp_emb.kv_unified = true;
#endif
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
        if (Status s = RunGuarded(ErrorCode::kDecodeFailed, "llama_init_from_model", [&] {
                impl_->embed_ctx = llama_init_from_model(impl_->model, cp_emb);
                return Status::Ok();
            });
            !s.ok()) {
            return s;
        }
        if (impl_->embed_ctx == nullptr) {
            return Status::Error(ErrorCode::kDecodeFailed, "failed to create embedding context");
        }
        const auto cap = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_batch(impl_->embed_ctx)));
        impl_->embed_batch = llama_batch_init(cap, /*embd=*/0, /*n_seq_max=*/1);
        impl_->embed_batch_cap = cap;
        impl_->embed_seq_max = want_seqs;
        impl_->embed_pooling_requested = requested_pooling;
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
        return Status::Error(ErrorCode::kInvalidArgument,
                             "reranking/classification pooling is not supported for dense embeddings");
    }
    const bool encoder_only = llama_model_has_encoder(impl_->model) && !llama_model_has_decoder(impl_->model);
    const auto n_batch = static_cast<std::size_t>(std::max<int32_t>(1, impl_->embed_batch_cap));
    const auto n_ubatch = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_ubatch(ectx)));
    const auto start = Clock::now();
    std::int64_t total_prompt_tokens = 0;
    Status batch_error = Status::Ok();
    std::size_t batches_executed = 0;

    const EmbedBatchExecStatus exec_status = ExecuteEmbedBatchPlan<std::vector<float>>(
        plan, cancelled,
        [&](const PlannedEmbedBatch& planned_batch, std::vector<std::vector<float>>& batch_out) -> bool {
            if (planned_batch.total_tokens > static_cast<std::size_t>(INT32_MAX)) {
                batch_error = Status::Error(ErrorCode::kContextOverflow, "embedding batch exceeds INT32_MAX");
                return false;
            }
            if (Status s = ValidateBatchBounds(static_cast<int32_t>(planned_batch.total_tokens),
                                               static_cast<int32_t>(n_batch), n_ubatch, /*require_ubatch_fit=*/true);
                !s.ok()) {
                batch_error = std::move(s);
                return false;
            }
            if (llama_memory_t emem = llama_get_memory(ectx); emem != nullptr) {
                llama_memory_clear(emem, true);
            }
            BatchClear(impl_->embed_batch);
            std::vector<int32_t> last_tok_idx;
            last_tok_idx.reserve(planned_batch.items.size());
            for (const EmbedBatchItem& item : planned_batch.items) {
                if (Status s = ValidateSeqId(static_cast<int32_t>(item.seq_id), impl_->embed_seq_max); !s.ok()) {
                    batch_error = std::move(s);
                    return false;
                }
                const auto& seq = inputs[item.original_index];
                const std::size_t n_tok = seq.size();
                for (std::size_t t = 0; t < n_tok; ++t) {
                    const bool want_out = (pooling != LLAMA_POOLING_TYPE_NONE) || (t + 1 == n_tok);
                    BatchAdd(impl_->embed_batch, seq[t], static_cast<llama_pos>(t),
                             static_cast<llama_seq_id>(item.seq_id), want_out);
                }
                last_tok_idx.push_back(impl_->embed_batch.n_tokens - 1);
            }
            const int32_t rc = encoder_only ? TryEncode(ectx, impl_->embed_batch, &batch_error)
                                            : TryDecode(ectx, impl_->embed_batch, &batch_error);
            if (rc != 0) {
                if (!cancelled() && batch_error.ok()) {
                    batch_error = Status::Error(ErrorCode::kDecodeFailed,
                                                "llama embedding pass failed (rc=" + std::to_string(rc) + ")");
                }
                return false;
            }

            batch_out.reserve(planned_batch.items.size());
            for (std::size_t k = 0; k < planned_batch.items.size(); ++k) {
                const EmbedBatchItem& item = planned_batch.items[k];
                const float* raw = nullptr;
                if (pooling != LLAMA_POOLING_TYPE_NONE) {
                    raw = llama_get_embeddings_seq(ectx, static_cast<llama_seq_id>(item.seq_id));
                }
                if (raw == nullptr) {
                    raw = llama_get_embeddings_ith(ectx, last_tok_idx[k]);
                }
                if (raw == nullptr && planned_batch.items.size() == 1u) {
                    raw = llama_get_embeddings(ectx);
                }
                if (raw == nullptr) {
                    batch_error = Status::Error(ErrorCode::kDecodeFailed, "llama.cpp returned null embeddings");
                    return false;
                }
                std::vector<float> vec(raw, raw + static_cast<std::size_t>(n_embd));
                if (options.normalize) {
                    double sq_sum = 0.0;
                    for (float v : vec) sq_sum += static_cast<double>(v) * static_cast<double>(v);
                    const double norm = std::sqrt(sq_sum);
                    if (norm > 0.0) {
                        const auto inv = static_cast<float>(1.0 / norm);
                        for (float& v : vec) v *= inv;
                    }
                }
                batch_out.push_back(std::move(vec));
                total_prompt_tokens += static_cast<std::int64_t>(item.token_count);
            }
            return true;
        },
        out.embeddings, &batches_executed);

    if (exec_status == EmbedBatchExecStatus::kCancelled) {
        out = {};
        return Status::Error(ErrorCode::kDecodeFailed, "embedding cancelled");
    }
    if (exec_status != EmbedBatchExecStatus::kOk) {
        out = {};
        return !batch_error.ok() ? batch_error
                                 : Status::Error(ErrorCode::kDecodeFailed, "embedding batch execution failed");
    }

    out.prompt_tokens = static_cast<std::int32_t>(
        std::min<std::int64_t>(total_prompt_tokens, static_cast<std::int64_t>(INT32_MAX)));
    out.eval_ms = MsSince(start);
    out.dimensions = static_cast<std::size_t>(n_embd);
    out.pooling = FromLlamaPooling(pooling);
    out.batches_executed = batches_executed;
    return Status::Ok();
}

Status LlamaCppBackend::ScorePairs(const std::vector<std::vector<Token>>& pairs, ScoreOutput& out,
                                   const CancelPredicate& should_cancel, std::uint32_t max_batch_tokens,
                                   std::uint32_t max_sequences) {
    out = {};
    if (!IsLoaded()) return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    if (impl_->ctx == nullptr) return Status::Error(ErrorCode::kNotLoaded, "model loaded vocab-only; no context");
    if (pairs.empty()) return Status::Error(ErrorCode::kInvalidArgument, "rerank/score pairs are empty");
    if (!SupportsReranking()) {
        return Status::Error(ErrorCode::kUnsupported,
                             "loaded model does not support reranking (RANK pooling is required)");
    }

    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    const uint32_t base_ctx = std::max<uint32_t>(256u, Impl::EffectiveCtxSeq(impl_->ctx));
    std::vector<std::size_t> lengths;
    lengths.reserve(pairs.size());
    for (const auto& seq : pairs) {
        if (seq.empty()) return Status::Error(ErrorCode::kInvalidArgument, "rerank/score pair is empty");
        for (Token t : seq) {
            if (t < 0 || t >= n_vocab) {
                return Status::Error(ErrorCode::kInvalidArgument, "rerank/score token out of range");
            }
        }
        lengths.push_back(seq.size());
    }

    const uint32_t configured_budget =
        max_batch_tokens > 0
            ? max_batch_tokens
            : (impl_->options.embed_batch_tokens > 0 ? impl_->options.embed_batch_tokens
                                                     : std::min<uint32_t>(base_ctx, kDefaultEmbedBatchTokens));
    uint32_t configured_seqs =
        max_sequences > 0
            ? max_sequences
            : (impl_->options.embed_max_sequences > 0 ? impl_->options.embed_max_sequences
                                                      : kDefaultEmbedMaxSequences);
    configured_seqs = std::clamp<uint32_t>(configured_seqs, 1u, kMaxEmbedSequences);
#if !defined(SONDER_LLAMACPP_HAS_KV_UNIFIED)
    configured_seqs = 1u;
#endif

    EmbedBatchPlanConfig plan_cfg;
    plan_cfg.max_batch_tokens = std::max<std::size_t>(1u, static_cast<std::size_t>(configured_budget));
    plan_cfg.max_sequences_per_batch = static_cast<std::size_t>(configured_seqs);
    plan_cfg.max_single_input_tokens = static_cast<std::size_t>(base_ctx);
    plan_cfg.allow_oversized_single_batch = true;

    const EmbedBatchPlan plan = PlanEmbedBatches(lengths, plan_cfg);
    if (!plan.ok) {
        return Status::Error(plan.context_overflow ? ErrorCode::kContextOverflow : ErrorCode::kInvalidArgument,
                             plan.error);
    }

    const uint32_t aligned_need =
        static_cast<uint32_t>(((plan.max_batch_tokens_used + 255u) / 256u) * 256u);
    const uint32_t want_ctx = std::max<uint32_t>(256u, aligned_need);
    const uint32_t want_seqs = static_cast<uint32_t>(std::max<std::size_t>(1u, plan.max_batch_sequences_used));
    const bool need_recreate = impl_->rank_ctx == nullptr ||
                               llama_n_ctx(impl_->rank_ctx) < want_ctx ||
                               impl_->rank_seq_max < want_seqs ||
                               impl_->rank_batch_cap < static_cast<int32_t>(want_ctx);
    if (need_recreate && impl_->rank_ctx != nullptr) {
        if (impl_->rank_batch_allocated) {
            llama_batch_free(impl_->rank_batch);
            impl_->rank_batch = {};
            impl_->rank_batch_cap = 0;
            impl_->rank_seq_max = 0;
            impl_->rank_batch_allocated = false;
        }
        llama_free(impl_->rank_ctx);
        impl_->rank_ctx = nullptr;
    }
    if (impl_->rank_ctx == nullptr) {
        llama_context_params cp_rank = llama_context_default_params();
        cp_rank.n_ctx = want_ctx;
        cp_rank.n_batch = want_ctx;
        cp_rank.n_ubatch = want_ctx;
        cp_rank.n_seq_max = want_seqs;
        cp_rank.embeddings = true;
        cp_rank.pooling_type = LLAMA_POOLING_TYPE_RANK;
        cp_rank.type_k = ToGgmlType(impl_->options.type_k);
        cp_rank.type_v = ToGgmlType(impl_->options.type_v);
        cp_rank.flash_attn_type = ToLlamaFlashAttn(impl_->options.flash_attn);
        cp_rank.offload_kqv = true;
        cp_rank.no_perf = true;
#if defined(SONDER_LLAMACPP_HAS_KV_UNIFIED)
        cp_rank.kv_unified = true;
#endif
        if (impl_->options.n_threads > 0) {
            cp_rank.n_threads = impl_->options.n_threads;
            cp_rank.n_threads_batch = impl_->options.n_threads;
        }
#if defined(SONDER_LLAMACPP_HAS_N_RS_SEQ)
        cp_rank.n_rs_seq = 0;
#endif
#if defined(SONDER_LLAMACPP_HAS_N_OUTPUTS_MAX)
        cp_rank.n_outputs_max = cp_rank.n_batch;
        cp_rank.n_outputs_max_per_seq = cp_rank.n_batch;
#endif
        cp_rank.abort_callback = &Impl::AbortCallback;
        cp_rank.abort_callback_data = impl_.get();
        if (Status s = RunGuarded(ErrorCode::kDecodeFailed, "llama_init_from_model", [&] {
                impl_->rank_ctx = llama_init_from_model(impl_->model, cp_rank);
                return Status::Ok();
            });
            !s.ok()) {
            return s;
        }
        if (impl_->rank_ctx == nullptr) {
            return Status::Error(ErrorCode::kDecodeFailed, "failed to create reranking context");
        }
        const auto cap = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_batch(impl_->rank_ctx)));
        impl_->rank_batch = llama_batch_init(cap, /*embd=*/0, /*n_seq_max=*/1);
        impl_->rank_batch_cap = cap;
        impl_->rank_seq_max = want_seqs;
        impl_->rank_batch_allocated = true;
    }

    llama_context* rctx = impl_->rank_ctx;
    if (llama_pooling_type(rctx) != LLAMA_POOLING_TYPE_RANK) {
        return Status::Error(ErrorCode::kUnsupported,
                             "reranking context did not activate RANK pooling for loaded model");
    }

    impl_->abort_requested.store(false, std::memory_order_relaxed);
    impl_->active_cancel = &should_cancel;
    struct RankGuard {
        Impl* impl;
        ~RankGuard() { impl->active_cancel = nullptr; }
    } guard{impl_.get()};

    auto cancelled = [&] {
        return impl_->abort_requested.load(std::memory_order_relaxed) || (should_cancel && should_cancel());
    };

    const bool encoder_only = llama_model_has_encoder(impl_->model) && !llama_model_has_decoder(impl_->model);
    const auto n_batch = static_cast<std::size_t>(std::max<int32_t>(1, impl_->rank_batch_cap));
    const auto n_ubatch = static_cast<int32_t>(std::max<uint32_t>(1, llama_n_ubatch(rctx)));
    const auto start = Clock::now();
    std::int64_t total_prompt_tokens = 0;
    Status batch_error = Status::Ok();
    std::size_t batches_executed = 0;

    const EmbedBatchExecStatus exec_status = ExecuteEmbedBatchPlan<double>(
        plan, cancelled,
        [&](const PlannedEmbedBatch& planned_batch, std::vector<double>& batch_out) -> bool {
            if (planned_batch.total_tokens > static_cast<std::size_t>(INT32_MAX)) {
                batch_error = Status::Error(ErrorCode::kContextOverflow, "rerank batch exceeds INT32_MAX");
                return false;
            }
            if (Status s = ValidateBatchBounds(static_cast<int32_t>(planned_batch.total_tokens),
                                               static_cast<int32_t>(n_batch), n_ubatch, /*require_ubatch_fit=*/true);
                !s.ok()) {
                batch_error = std::move(s);
                return false;
            }
            if (llama_memory_t rmem = llama_get_memory(rctx); rmem != nullptr) {
                llama_memory_clear(rmem, true);
            }
            BatchClear(impl_->rank_batch);
            std::vector<int32_t> last_tok_idx;
            last_tok_idx.reserve(planned_batch.items.size());
            for (const EmbedBatchItem& item : planned_batch.items) {
                if (Status s = ValidateSeqId(static_cast<int32_t>(item.seq_id), impl_->rank_seq_max); !s.ok()) {
                    batch_error = std::move(s);
                    return false;
                }
                const auto& seq = pairs[item.original_index];
                for (std::size_t t = 0; t < seq.size(); ++t) {
                    BatchAdd(impl_->rank_batch, seq[t], static_cast<llama_pos>(t),
                             static_cast<llama_seq_id>(item.seq_id), /*logits=*/true);
                }
                last_tok_idx.push_back(impl_->rank_batch.n_tokens - 1);
            }
            const int32_t rc = encoder_only ? TryEncode(rctx, impl_->rank_batch, &batch_error)
                                            : TryDecode(rctx, impl_->rank_batch, &batch_error);
            if (rc != 0) {
                if (!cancelled() && batch_error.ok()) {
                    batch_error = Status::Error(ErrorCode::kDecodeFailed,
                                                "llama reranking pass failed (rc=" + std::to_string(rc) + ")");
                }
                return false;
            }

            batch_out.reserve(planned_batch.items.size());
            for (std::size_t k = 0; k < planned_batch.items.size(); ++k) {
                const EmbedBatchItem& item = planned_batch.items[k];
                const float* raw = llama_get_embeddings_seq(rctx, static_cast<llama_seq_id>(item.seq_id));
                if (raw == nullptr) {
                    raw = llama_get_embeddings_ith(rctx, last_tok_idx[k]);
                }
                if (raw == nullptr) {
                    batch_error = Status::Error(ErrorCode::kDecodeFailed, "llama.cpp returned null rerank score");
                    return false;
                }
                const double score = static_cast<double>(raw[0]);
                if (!std::isfinite(score)) {
                    batch_error = Status::Error(ErrorCode::kDecodeFailed, "llama.cpp returned non-finite rerank score");
                    return false;
                }
                batch_out.push_back(score);
                total_prompt_tokens += static_cast<std::int64_t>(item.token_count);
            }
            return true;
        },
        out.scores, &batches_executed);

    if (exec_status == EmbedBatchExecStatus::kCancelled) {
        out = {};
        return Status::Error(ErrorCode::kDecodeFailed, "reranking cancelled");
    }
    if (exec_status != EmbedBatchExecStatus::kOk) {
        out = {};
        return !batch_error.ok() ? batch_error
                                 : Status::Error(ErrorCode::kDecodeFailed, "rerank batch execution failed");
    }

    out.prompt_tokens = static_cast<std::int32_t>(
        std::min<std::int64_t>(total_prompt_tokens, static_cast<std::int64_t>(INT32_MAX)));
    out.eval_ms = MsSince(start);
    out.batches_executed = batches_executed;
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
    if (request.max_new_tokens < 0) return fail(ErrorCode::kInvalidArgument, "max_new_tokens < 0");
    if (request.target_ttft_ms.has_value() &&
        (!std::isfinite(*request.target_ttft_ms) || *request.target_ttft_ms <= 0.0)) {
        return fail(ErrorCode::kInvalidArgument, "target_ttft_ms must be finite and > 0");
    }
    if (request.target_itl_ms.has_value() &&
        (!std::isfinite(*request.target_itl_ms) || *request.target_itl_ms <= 0.0)) {
        return fail(ErrorCode::kInvalidArgument, "target_itl_ms must be finite and > 0");
    }
    if (Status s = ValidateMultimodalRequest(request, HasMmproj(), HasMtmdBuildSupport()); !s.ok()) {
        return fail(s.code, s.message);
    }

    const int32_t n_vocab = llama_vocab_n_tokens(impl_->vocab);
    if (Status s = ValidatePromptTokens(request.prompt, n_vocab); !s.ok()) return fail(s.code, s.message);
    if (Status s = ValidateSamplingRequestGuards(request.sampling, n_vocab); !s.ok()) {
        return fail(s.code, s.message);
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
        task.enqueue_time_us = impl_->NowUs();
        {
            std::lock_guard<std::mutex> lock(impl_->queue_mu);
            if (impl_->worker_stop.load(std::memory_order_acquire)) {
                return fail(ErrorCode::kNotLoaded, "backend is unloading");
            }
            impl_->pending_queue.push_back(&task);
            impl_->pending_count.fetch_add(1, std::memory_order_release);
            if (exclusive) {
                impl_->pending_exclusive_count.fetch_add(1, std::memory_order_release);
            }
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

    const bool stage_prof = request.enable_stage_profiling;
    result.stage_timings.enabled = stage_prof;
    auto add_stage_ms = [](std::optional<double>& slot, double ms) noexcept {
        if (!std::isfinite(ms) || ms < 0.0) return;
        slot = slot.has_value() ? (*slot + ms) : ms;
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
                                grammar_sampler, /*already_validated=*/true);
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
        const Clock::time_point lookup_start = stage_prof ? Clock::now() : Clock::time_point{};
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
        if (stage_prof) {
            add_stage_ms(result.stage_timings.cache_lookup_ms, MsSince(lookup_start));
        }

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
            const auto l2_start = Clock::now();
            if (TryRestoreSlotFromL2(slot_idx, slot_id, request.prompt, request.session_key, lcp, l2_lcp,
                                     l2_mtp_synced, l2_wiped)) {
                const double l2_ms = MsSince(l2_start);
                lcp = l2_lcp;
                restored_from_l2 = true;
                result.host_l2_restored_tokens =
                    static_cast<std::int32_t>(std::min<std::size_t>(l2_lcp, static_cast<std::size_t>(INT32_MAX)));
                result.host_l2_restore_ms = l2_ms;
                if (stage_prof) {
                    add_stage_ms(result.stage_timings.l2_restore_ms, l2_ms);
                }
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
                ResetSwaSlot(slot_idx);
                ClearSlotCheckpoints(slot_idx);
                if (slot_idx < slot_mtp_synced.size()) slot_mtp_synced[slot_idx] = true;
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (ctx_dft != nullptr) {
                    if (!mtp_mem_shared) {
                        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                    }
                    std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                    if (slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[slot_idx] = -1;
                    mtp_deferred[slot_idx].Clear();
                }
#endif
            } else {
                bool restored_pending_h = false;
                if (lcp < kv_len) {
                    const bool swa_ok = CheckSwaRollback(slot_idx, mem, slot_id, static_cast<std::uint64_t>(lcp));
                    if (swa_ok && llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(lcp), -1)) {
                        PruneSlotCheckpointsAfter(slot_idx, lcp);
                    } else {
                        bool restored_from_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                        std::size_t ckpt_pos = 0;
                        if (!swa_compact_active &&
                            RestorePartialCheckpoint(slot_idx, slot_id, lcp, ckpt_pos, restored_pending_h)) {
                            lcp = ckpt_pos;
                            restored_from_ckpt = true;
                        }
#endif
                        if (!restored_from_ckpt) {
                            llama_memory_seq_rm(mem, slot_id, -1, -1);
                            ResetSwaSlot(slot_idx);
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
                        if (slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[slot_idx] = -1;
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
                                if (slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[slot_idx] = -1;
                                lcp = 0;
                            } else if (dft_len > static_cast<int64_t>(lcp)) {
                                if (!llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id,
                                                         static_cast<llama_pos>(lcp), -1)) {
                                    llama_memory_seq_rm(mem, slot_id, -1, -1);
                                    llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, -1, -1);
                                    ClearSlotCheckpoints(slot_idx);
                                    std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                                    if (slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[slot_idx] = -1;
                                    lcp = 0;
                                }
                            }
                        }
                        if (lcp > 0 && lcp != prev_slot_len && !restored_pending_h) {
                            std::fill(mtp_pending_h[slot_idx].begin(), mtp_pending_h[slot_idx].end(), 0.0F);
                            if (slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[slot_idx] = -1;
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
        const std::int32_t reused_i32 =
            static_cast<std::int32_t>(std::min<std::size_t>(lcp, static_cast<std::size_t>(INT32_MAX)));
        result.cached_tokens = reused_i32;
        if (!restored_from_l2) {
            result.device_prefix_reused_tokens = reused_i32;
        }
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

    if (Status s = ValidateSeqId(slot_id, std::max<std::uint32_t>(1u, static_cast<std::uint32_t>(slots.size())));
        !s.ok()) {
        return fail(s.code, s.message);
    }
    const int64_t n_past = llama_memory_seq_pos_max(mem, slot_id) + 1;
    const int64_t remaining_prompt = n_prompt - prefill_start_offset;
    if (!has_media) {
        if (Status s = ValidateContextPosition(n_past, remaining_prompt, n_ctx); !s.ok()) {
            return fail(s.code, s.message);
        }
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

    // Record configured vs effective speculation strategy and disable/fallback reason.
    result.configured_spec_method = std::string(ToString(requested_spec));
    result.configured_draft_limit = options.spec_draft_max > 0 ? options.spec_draft_max : active_draft_max;
    if (use_mtp) {
        result.effective_spec_method = use_hybrid_suffix ? "draft-mtp+suffix" : "draft-mtp";
        result.effective_draft_limit = effective_draft_max;
        result.draft_identity = use_hybrid_suffix ? "mtp:native+suffix" : "mtp:native";
        result.spec_fallback_reason = load_spec_fallback_reason;
    } else if (use_suffix) {
        result.effective_spec_method = "suffix";
        result.effective_draft_limit = effective_draft_max;
        result.draft_identity = "suffix";
        result.spec_fallback_reason = load_spec_fallback_reason;
    } else if (use_ngram_mod) {
        result.effective_spec_method = "ngram-mod";
        result.effective_draft_limit = effective_draft_max;
        result.draft_identity = "ngram-mod";
        result.spec_fallback_reason = load_spec_fallback_reason;
    } else if (use_ngram_short) {
        result.effective_spec_method = "ngram";
        result.effective_draft_limit = effective_draft_max;
        result.draft_identity = "ngram";
        result.spec_fallback_reason = load_spec_fallback_reason;
    } else {
        result.effective_spec_method = "none";
        result.effective_draft_limit = 0;
        if (active_spec != SpeculativeType::kNone) {
            if (has_media) {
                result.spec_fallback_reason = "media_exclusive";
            } else if (architecture == sonder::inference::ModelArchitecture::recurrent) {
                result.spec_fallback_reason = "recurrent_unsupported";
            } else if (architecture == sonder::inference::ModelArchitecture::hybrid) {
                result.spec_fallback_reason = "hybrid_unsupported";
            } else if (result.state_restored) {
                result.spec_fallback_reason = "state_restored";
            } else {
                result.spec_fallback_reason =
                    load_spec_fallback_reason.empty() ? "not_supported" : load_spec_fallback_reason;
            }
        } else {
            result.spec_fallback_reason = load_spec_fallback_reason;
        }
    }

    // Bind GPU backend sampling when supported. Disabled when a grammar,
    // per-token logprobs, or stochastic rejection verification is active so
    // raw logits remain readable on CPU.
    [[maybe_unused]] bool target_gpu_sampled = false;
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
    if (!grammar_sampler && !request.sampling.compute_logprobs &&
        effective_spec_verification_mode != SpeculativeVerificationMode::StochasticRejection) {
        target_gpu_sampled = TrySetSampler(ctx, slot_id, sampler.get());
    }
    struct ClearTargetSampler {
        llama_context* c = nullptr;
        llama_seq_id sid = 0;
        bool active = false;
        ~ClearTargetSampler() {
            if (active && c != nullptr) {
                (void)TrySetSampler(c, sid, nullptr);
            }
        }
    } clear_target_sampler{ctx, slot_id, target_gpu_sampled};
#endif

    auto sample_target = [&](int32_t idx, TokenLogprobInfo* out_lp = nullptr) -> Token {
        const Clock::time_point t_samp = stage_prof ? Clock::now() : Clock::time_point{};
        const Token tok = SampleTokenAt(idx, request.sampling, sampler.get(), grammar_sampler.get(), target_gpu_sampled,
                                        n_vocab, out_lp);
        if (stage_prof) {
            add_stage_ms(result.stage_timings.sampling_ms, MsSince(t_samp));
        }
        return tok;
    };

    bool slot_valid = true;
    auto invalidate_slot = [&]() {
        slot_valid = false;
        llama_memory_seq_rm(mem, slot_id, -1, -1);
        ResetSwaSlot(slot_idx);
        if (slot_idx < slots.size()) {
            slots[slot_idx].tokens.clear();
            slots[slot_idx].session_key.clear();
            slots[slot_idx].last_used = 0;
        }
        if (slot_idx < slot_mtp_synced.size()) {
            slot_mtp_synced[slot_idx] = false;
        }
        if (slot_idx < slot_mtp_pending_h_pos.size()) {
            slot_mtp_pending_h_pos[slot_idx] = -1;
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
            if (Status s = ValidateBatchBounds(n, batch_cap, static_cast<int32_t>(n_ubatch),
                                               /*require_ubatch_fit=*/false);
                !s.ok()) {
                return fail(s.code, s.message);
            }
            BatchClear(batch);
            for (int32_t j = 0; j < n; ++j) {
                const bool is_last_prompt_tok = (off + j + 1 == n_prompt);
                BatchAdd(batch, request.prompt[static_cast<std::size_t>(off + j)],
                         static_cast<llama_pos>(cur_pos + j), slot_id, is_last_prompt_tok);
            }
            Status decode_err;
            const int32_t rc = TryDecode(ctx, batch, &decode_err);
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
                if (!decode_err.ok()) return fail(decode_err.code, decode_err.message);
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
            AdvanceSwaSlot(slot_idx, mem, slot_id, static_cast<std::uint64_t>(cur_pos));
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
            if (request.on_safe_point && off < n_prompt) {
                SafePointInfo sp;
                sp.slot_index = slot_idx;
                sp.prompt_tokens = request.prompt.size();
                sp.prefill_completed_tokens = static_cast<std::size_t>(off);
                sp.generated_tokens = 0;
                sp.kv_pos = cur_pos;
                sp.active_peers = 0;
                sp.in_prefill = true;
                const SafePointDecision dec = request.on_safe_point(sp);
                if (dec.action != SuspendAction::kContinue) {
                    result.suspended = true;
                    result.suspend_action = dec.action;
                    if (slot_idx < slots.size()) {
                        result.suspended_tokens = slots[slot_idx].tokens;
                    }
                    result.suspended_token_position = result.suspended_tokens.size();
                    if (dec.action == SuspendAction::kYieldSpill) {
                        const std::size_t state_sz = llama_state_seq_get_size(ctx, slot_id);
                        const std::size_t cap = std::min(dec.max_spill_bytes, kMaxSequenceStateBytes);
                        if (state_sz > 0 && state_sz <= cap) {
                            result.suspended_state_blob.resize(state_sz);
                            const std::size_t wrote = TryStateSeqGetData(
                                ctx, result.suspended_state_blob.data(), result.suspended_state_blob.size(), slot_id);
                            if (wrote > 0 && wrote <= state_sz) {
                                result.suspended_state_blob.resize(wrote);
                            } else {
                                result.suspended_state_blob.clear();
                            }
                        }
                    }
                    invalidate_slot();
                    result.physical_slot_reclaimed = (llama_memory_seq_pos_max(mem, slot_id) < 0);
                    result.stop_reason = StopReason::kSuspended;
                    result.prefill_ms = MsSince(prefill_start);
                    Emit(TelemetryKind::kGenerationDone, result.generated_tokens, 0.0,
                         std::string(ToString(StopReason::kSuspended)));
                    return result;
                }
            }
        }
    }
    result.prefill_ms = MsSince(prefill_start);
    if (!has_media && result.prompt_tokens > 0 && result.prefill_ms > 0.0) {
        radix_cache.ObservePrefill(static_cast<std::size_t>(result.prompt_tokens), result.prefill_ms);
    }
    Emit(TelemetryKind::kPrefillDone, result.prompt_tokens, result.prefill_ms);

    // ---- Streaming decode (with optional in-process speculation) ----
    Utf8StreamBuffer utf8;
    int64_t pos = cur_pos;
    std::size_t ngram_indexed = 0;
    const auto decode_start = Clock::now();
    auto finish = [&](StopReason reason) {
        result.stop_reason = reason;
        result.decode_ms = MsSince(decode_start);
        if (stage_prof) {
            result.stage_timings.prefill_ms = result.prefill_ms;
        }
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

    auto poll_single_safe_point = [&](Token pending_tok) -> bool {
        if (!request.on_safe_point) return false;
        SafePointInfo sp;
        sp.slot_index = slot_idx;
        sp.prompt_tokens = request.prompt.size();
        sp.prefill_completed_tokens = request.prompt.size();
        sp.generated_tokens = result.generated_tokens;
        sp.kv_pos = pos;
        sp.active_peers = 0;
        sp.in_prefill = false;
        const SafePointDecision dec = request.on_safe_point(sp);
        if (dec.action == SuspendAction::kContinue) return false;

        result.suspended = true;
        result.suspend_action = dec.action;
        if (slot_idx < slots.size()) {
            result.suspended_tokens = slots[slot_idx].tokens;
        }
        if (pending_tok != LLAMA_TOKEN_NULL) {
            result.suspended_tokens.push_back(pending_tok);
        }
        result.suspended_token_position = result.suspended_tokens.size();
        if (dec.action == SuspendAction::kYieldSpill) {
            const std::size_t state_sz = llama_state_seq_get_size(ctx, slot_id);
            const std::size_t cap = std::min(dec.max_spill_bytes, kMaxSequenceStateBytes);
            if (state_sz > 0 && state_sz <= cap) {
                result.suspended_state_blob.resize(state_sz);
                const std::size_t wrote = TryStateSeqGetData(
                    ctx, result.suspended_state_blob.data(), result.suspended_state_blob.size(), slot_id);
                if (wrote > 0 && wrote <= state_sz) {
                    result.suspended_state_blob.resize(wrote);
                } else {
                    result.suspended_state_blob.clear();
                }
            }
        }
        invalidate_slot();
        result.physical_slot_reclaimed = (llama_memory_seq_pos_max(mem, slot_id) < 0);
        return true;
    };

    if (poll_single_safe_point(cur_tok)) return finish(StopReason::kSuspended);

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
    const std::uint64_t verify_seed =
        request.sampling.seed == 0xFFFFFFFFU ? std::random_device{}() : static_cast<std::uint64_t>(request.sampling.seed);
    std::mt19937_64 verify_rng(verify_seed ^ 0x517CC1B727220A95ULL);

    while (true) {
        if (cancelled()) return finish(StopReason::kCancelled);
        if (poll_single_safe_point(cur_tok)) return finish(StopReason::kSuspended);

        const auto step_start = Clock::now();
        const int64_t rem_gen =
            static_cast<int64_t>(request.max_new_tokens) - static_cast<int64_t>(result.generated_tokens);
        const int64_t max_k = std::min<int64_t>({rem_gen - 1, n_ctx - pos - 1, n_batch - 1, n_ubatch - 1});

        draft_tokens.clear();
        bool step_used_suffix = false;
        const bool spec_enabled_step = use_mtp || use_suffix || use_ngram_mod || use_ngram_short;
        const Clock::time_point draft_start =
            (stage_prof && spec_enabled_step && max_k > 0) ? Clock::now() : Clock::time_point{};
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
                        result.spec_transitioned_during_request = true;
                        result.effective_spec_method = "none";
                        result.effective_draft_limit = 0;
                        result.spec_fallback_reason = "draft_sync_failed";
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
        if (stage_prof && spec_enabled_step && max_k > 0) {
            add_stage_ms(result.stage_timings.draft_ms, MsSince(draft_start));
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
                    TryStateSeqGetDataExt(ctx, spec_ckpt_buf.data(), sz, slot_id, flags);
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

        const Clock::time_point verify_start =
            (stage_prof && !draft_tokens.empty()) ? Clock::now() : Clock::time_point{};
        Status decode_err;
        const int32_t rc = TryDecode(ctx, batch, &decode_err);
        if (stage_prof && !draft_tokens.empty()) {
            add_stage_ms(result.stage_timings.verify_ms, MsSince(verify_start));
        }
        if (rc != 0) {
            bool rolled_back = CheckSwaRollback(slot_idx, mem, slot_id, static_cast<std::uint64_t>(pos)) &&
                               llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1);
            if (!rolled_back) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                if (have_spec_ckpt && !swa_compact_active) {
                    const llama_state_seq_flags flags = PartialCheckpointFlags();
                    const std::size_t read =
                        TryStateSeqSetDataExt(ctx, spec_ckpt_buf.data(), spec_ckpt_size, slot_id, flags);
                    rolled_back = (read > 0) && llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1);
                }
#endif
            }
            if (!rolled_back) {
                invalidate_slot();
            }
            if (rc == 2) return finish(StopReason::kCancelled);
            if (rc == 1) return finish(StopReason::kContextFull);
            result.status = !decode_err.ok()
                                ? std::move(decode_err)
                                : Status::Error(ErrorCode::kDecodeFailed,
                                                "llama_decode failed (rc=" + std::to_string(rc) + ")");
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
            AdvanceSwaSlot(slot_idx, mem, slot_id, static_cast<std::uint64_t>(pos + 1));
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

        // Speculative verification: verify draft tokens [0..k) + bonus token at k.
        // Built-in deterministic/greedy proposers pass empty proposal_dists and
        // stay on the exact TargetMatch reference path; StochasticRejection
        // activates only when opted in AND a stochastic proposer supplies real q.
        const std::size_t k = draft_tokens.size();
        const Token step_base_tok = cur_tok;
        result.draft_tokens += static_cast<std::int32_t>(k);
        std::size_t n_accepted = 0;
        const std::span<const std::vector<double>> no_proposal_dists{};
        const Clock::time_point spec_samp_start = stage_prof ? Clock::now() : Clock::time_point{};
        // No grammar retry here: with the built-in proposers (TargetMatch) a speculative step
        // samples position j from the same logits and sampler state a plain step would use, and a
        // grammar failure (lazy-trigger replay that throws in llama_grammar_accept_impl) is a
        // deterministic function of the token sequence. Restoring a snapshot and re-sampling
        // reproduces the same token and the same failure, so the step fails the request. The real
        // fix is upstream ggml-org/llama.cpp#32934 (transactional trigger activation, unmerged,
        // not in the b11195 pin). The batch worker's per-slot verification behaves the same way.
        const bool sample_deadlock = !VerifySpeculativeBatch(
            draft_tokens, no_proposal_dists, request.sampling, sampler.get(), grammar_sampler.get(),
            target_gpu_sampled, n_vocab, verify_rng, accepted_tokens, accepted_lps, n_accepted, result);
        if (stage_prof) {
            add_stage_ms(result.stage_timings.sampling_ms, MsSince(spec_samp_start));
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
            const Clock::time_point rb_start = stage_prof ? Clock::now() : Clock::time_point{};
            auto finish_rb = [&](bool ok) {
                if (stage_prof) {
                    add_stage_ms(result.stage_timings.rollback_ms, MsSince(rb_start));
                }
                return ok;
            };
#if defined(SONDER_LLAMACPP_HAS_MTP)
            if (use_mtp && !MtpCommitVerified(batch, slot_id, static_cast<int32_t>(keep_drafts))) {
                invalidate_slot();
                return finish_rb(false);
            }
#endif
            const int64_t batch_end_pos = pos + 1 + static_cast<int64_t>(k);
            AdvanceSwaSlot(slot_idx, mem, slot_id, static_cast<std::uint64_t>(batch_end_pos));
            if (keep_pos < batch_end_pos) {
                const bool swa_ok = CheckSwaRollback(slot_idx, mem, slot_id, static_cast<std::uint64_t>(keep_pos));
                if (!swa_ok || !llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(keep_pos), -1)) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                    if (have_spec_ckpt && !swa_compact_active) {
                        const llama_state_seq_flags flags = PartialCheckpointFlags();
                        const std::size_t read = TryStateSeqSetDataExt(ctx, spec_ckpt_buf.data(), spec_ckpt_size,
                                                                       slot_id, flags);
                        if (read == 0 || !llama_memory_seq_rm(mem, slot_id, static_cast<llama_pos>(pos), -1)) {
                            invalidate_slot();
                            return finish_rb(false);
                        }
                        BatchClear(batch);
                        BatchAdd(batch, step_base_tok, static_cast<llama_pos>(pos), slot_id, /*logits=*/false);
                        for (std::size_t j = 0; j < keep_drafts; ++j) {
                            BatchAdd(batch, draft_tokens[j],
                                     static_cast<llama_pos>(pos + 1 + static_cast<int64_t>(j)), slot_id,
                                     /*logits=*/false);
                        }
                        if (TryDecode(ctx, batch) != 0) {
                            invalidate_slot();
                            return finish_rb(false);
                        }
                        return finish_rb(true);
                    }
#else
                    (void)step_base_tok;
                    (void)keep_drafts;
#endif
                    invalidate_slot();
                    return finish_rb(false);
                }
            }
            return finish_rb(true);
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
        std::int64_t enqueue_time_us = 0;
        std::int64_t last_service_time_us = 0;
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
        std::mt19937_64 verify_rng{};
        std::size_t ngram_indexed = 0;
        int32_t batch_logit_idx = -1;
        // Per-request isolated speculation transition state, exact committed
        // token history, local drafters, and proposal/verification buffers.
        SpeculationTransitionState spec_transition;
        std::vector<Token> committed_tokens;
        std::unique_ptr<SuffixMatcher> local_suffix_matcher;
        std::unique_ptr<NgramMatcher> local_ngram_matcher;
        bool local_matcher_synced = false;
        std::vector<Token> draft_tokens;
        std::vector<Token> accepted_tokens;
        std::vector<TokenLogprobInfo> accepted_lps;
        int32_t batch_start_idx = -1;
        std::size_t step_n_accepted = 0;
        bool step_sample_deadlock = false;
        SlotSpecMode concurrent_step_mode = SlotSpecMode::kNone;
        std::uint64_t admit_order = 0;
        GenerateResult result{};

        [[nodiscard]] bool IsCancelled() const {
            return task != nullptr && task->should_cancel != nullptr && *task->should_cancel &&
                   (*task->should_cancel)();
        }
    };

    auto add_stage_ms = [](std::optional<double>& slot, double ms) noexcept {
        if (!std::isfinite(ms) || ms < 0.0) return;
        slot = slot.has_value() ? (*slot + ms) : ms;
    };

    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    llama_memory_t mem = llama_get_memory(ctx);
    const auto n_ctx = static_cast<int64_t>(EffectiveCtxSeq(ctx));
    const auto n_batch = static_cast<int32_t>(std::max<int32_t>(1, batch_cap));
    const auto n_ubatch = static_cast<int32_t>(std::max<uint32_t>(1u, llama_n_ubatch(ctx)));
    const int32_t step_token_budget = std::min<int32_t>(n_batch, n_ubatch);
    const int32_t step_output_budget = std::max<int32_t>(1, static_cast<int32_t>(slots.size()));
    const int32_t concurrent_output_budget =
        std::min<int32_t>(step_token_budget, ctx_outputs_max > 0 ? ctx_outputs_max : step_token_budget);
    const int32_t per_seq_output_budget =
        std::min<int32_t>(concurrent_output_budget,
                          ctx_outputs_max_per_seq > 0 ? ctx_outputs_max_per_seq : concurrent_output_budget);

    SpeculationReactivationPolicy reactivation_policy;
    reactivation_policy.cooldown_steps = options.spec_reactivation_cooldown_steps;

    ConcurrentSpecPolicy concurrent_spec_policy;
    concurrent_spec_policy.enabled = options.spec_batch_enabled;
    concurrent_spec_policy.max_concurrent_slots = std::clamp<std::uint32_t>(options.spec_batch_max_slots, 1u, kMaxSlots);
    concurrent_spec_policy.max_aggregate_draft_tokens =
        options.spec_batch_max_aggregate_draft_tokens > 0 ? options.spec_batch_max_aggregate_draft_tokens : 16;
    ConcurrentSpecCostGate concurrent_cost_gate;
    std::size_t conc_rr_cursor = 0;

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
    const auto coalesce_clock_origin = Clock::now();
    auto coalesce_now_ms = [&]() -> std::uint64_t {
        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - coalesce_clock_origin).count();
        return elapsed > 0 ? static_cast<std::uint64_t>(elapsed) : 0u;
    };

    auto find_active_by_slot = [&](std::size_t slot_index) -> ActiveBatchSlot* {
        for (auto& sp : active) {
            if (sp->slot_idx == slot_index) return sp.get();
        }
        return nullptr;
    };

    // Materializes zero-copy VRAM KV clones for followers released by a
    // producer publishing `[0, clone_prefix_len)` (or a partial chunk on
    // timeout). Falls back safely to independent prefill if the clone fails.
    auto apply_coalesce_releases = [&](const std::vector<CoalesceReleaseAction>& actions) {
        if (actions.empty()) return;
        for (const CoalesceReleaseAction& act : actions) {
            if (act.clone_prefix_len == 0) continue;
            ActiveBatchSlot* follower = find_active_by_slot(act.follower_slot);
            ActiveBatchSlot* producer = find_active_by_slot(act.producer_slot);
            if (follower == nullptr || follower->task == nullptr || follower->IsCancelled()) {
                prefix_coalescer.OnCloneFailed(act.follower_slot,
                                               follower != nullptr ? follower->prefill_offset : 0u,
                                               act.tokens_saved);
                continue;
            }
            const bool can_clone =
                architecture == sonder::inference::ModelArchitecture::attention_only &&
                producer != nullptr && producer->slot_valid &&
                act.clone_prefix_len < follower->task->req->prompt.size() &&
                CanCloneSwaSource(producer->slot_idx, mem, producer->slot_id,
                                  static_cast<std::uint64_t>(act.clone_prefix_len));
            bool cloned_ok = false;
            if (can_clone) {
                const int64_t src_kv = llama_memory_seq_pos_max(mem, producer->slot_id) + 1;
                if (src_kv >= static_cast<int64_t>(act.clone_prefix_len)) {
                    llama_memory_seq_rm(mem, follower->slot_id, -1, -1);
                    ResetSwaSlot(follower->slot_idx);
                    ClearSlotCheckpoints(follower->slot_idx);
                    llama_memory_seq_cp(mem, producer->slot_id, follower->slot_id, 0,
                                        static_cast<llama_pos>(act.clone_prefix_len));
                    if (llama_memory_seq_pos_max(mem, follower->slot_id) + 1 ==
                        static_cast<int64_t>(act.clone_prefix_len)) {
                        cloned_ok = true;
                        SyncClonedSwaSlot(follower->slot_idx, producer->slot_idx, mem, follower->slot_id,
                                          static_cast<std::uint64_t>(act.clone_prefix_len));
                        const auto& fprompt = follower->task->req->prompt;
                        slots[follower->slot_idx].tokens.assign(
                            fprompt.begin(),
                            fprompt.begin() + static_cast<std::ptrdiff_t>(act.clone_prefix_len));
                        slots[follower->slot_idx].last_used = ++slot_clock;
                        follower->committed_tokens.assign(
                            fprompt.begin(),
                            fprompt.begin() + static_cast<std::ptrdiff_t>(act.clone_prefix_len));
                        follower->prefill_offset = act.clone_prefix_len;
                        follower->pos = static_cast<int64_t>(act.clone_prefix_len);
                        follower->result.cached_tokens = static_cast<std::int32_t>(
                            std::min<std::size_t>(act.clone_prefix_len, static_cast<std::size_t>(INT32_MAX)));
                        follower->result.device_prefix_reused_tokens = follower->result.cached_tokens;
                        follower->result.device_cross_slot_reused_tokens = follower->result.cached_tokens;
                        if (follower->slot_idx < slot_mtp_synced.size()) {
                            slot_mtp_synced[follower->slot_idx] = false;
                        }
                        if (follower->slot_idx < slot_mtp_pending_h_pos.size()) {
                            slot_mtp_pending_h_pos[follower->slot_idx] = -1;
                        }
#if defined(SONDER_LLAMACPP_HAS_MTP)
                        if (ctx_dft != nullptr && follower->slot_idx < mtp_pending_h.size()) {
                            if (!mtp_mem_shared) {
                                llama_memory_seq_rm(llama_get_memory(ctx_dft), follower->slot_id, -1, -1);
                            }
                            std::fill(mtp_pending_h[follower->slot_idx].begin(),
                                      mtp_pending_h[follower->slot_idx].end(), 0.0F);
                            if (follower->slot_idx < mtp_deferred.size()) {
                                mtp_deferred[follower->slot_idx].Clear();
                            }
                        }
#endif
                    } else {
                        llama_memory_seq_rm(mem, follower->slot_id, -1, -1);
                        slots[follower->slot_idx].tokens.clear();
                        follower->committed_tokens.clear();
                        follower->prefill_offset = 0;
                        follower->pos = 0;
                        follower->result.cached_tokens = 0;
                        follower->result.device_prefix_reused_tokens = 0;
                        follower->result.device_cross_slot_reused_tokens = 0;
                        if (follower->slot_idx < slot_mtp_synced.size()) {
                            slot_mtp_synced[follower->slot_idx] = false;
                        }
                        if (follower->slot_idx < slot_mtp_pending_h_pos.size()) {
                            slot_mtp_pending_h_pos[follower->slot_idx] = -1;
                        }
                    }
                }
            }
            if (!cloned_ok) {
                prefix_coalescer.OnCloneFailed(follower->slot_idx, follower->prefill_offset, act.tokens_saved);
            }
        }
        SyncCoalescerStats();
    };

    auto sync_slot_flags_from_transition = [](ActiveBatchSlot& s) noexcept {
        const SlotSpecMode m = s.spec_transition.active_mode();
        s.mtp_enabled = (m == SlotSpecMode::kDraftMtp);
        s.hybrid_suffix_enabled = (m == SlotSpecMode::kDraftMtp) && s.spec_transition.active_hybrid_suffix();
        s.suffix_enabled = (m == SlotSpecMode::kSuffix);
        s.ngram_mod_enabled = (m == SlotSpecMode::kNgramMod);
        s.ngram_short_enabled = (m == SlotSpecMode::kNgramShort);
        s.effective_draft_max = s.spec_transition.active_draft_max();
    };

    auto mark_slot_mtp_drifted = [&](ActiveBatchSlot& s) noexcept {
        if (s.slot_idx < slot_mtp_synced.size()) {
            slot_mtp_synced[s.slot_idx] = false;
        }
        if (s.slot_idx < slot_mtp_pending_h_pos.size()) {
            slot_mtp_pending_h_pos[s.slot_idx] = -1;
        }
#if defined(SONDER_LLAMACPP_HAS_MTP)
        if (s.slot_idx < mtp_deferred.size()) {
            mtp_deferred[s.slot_idx].Clear();
        }
#endif
    };

    auto build_mtp_readiness = [&](const ActiveBatchSlot& s) -> MtpReadinessState {
        MtpReadinessState st;
        st.slot_valid = s.slot_valid;
        st.target_pos = s.pos;
        st.slot_mtp_synced = (s.slot_idx < slot_mtp_synced.size()) ? slot_mtp_synced[s.slot_idx] : false;
        st.pending_features_pos =
            (s.slot_idx < slot_mtp_pending_h_pos.size()) ? slot_mtp_pending_h_pos[s.slot_idx] : -1;
        st.pending_features_valid = (st.pending_features_pos >= 0);
#if defined(SONDER_LLAMACPP_HAS_MTP)
        st.draft_ctx_available = (ctx_dft != nullptr);
        st.mem_shared = mtp_mem_shared;
        st.pending_features_size_ok =
            (mtp_n_embd > 0) && (s.slot_idx < mtp_pending_h.size()) &&
            (mtp_pending_h[s.slot_idx].size() == static_cast<std::size_t>(mtp_n_embd));
        if (ctx_dft != nullptr && !mtp_mem_shared) {
            st.draft_pos = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), s.slot_id) + 1;
            if (s.slot_idx < mtp_deferred.size() && mtp_deferred[s.slot_idx].pending) {
                const auto& d = mtp_deferred[s.slot_idx];
                st.has_deferred = true;
                st.deferred_count = d.tokens.size();
                if (!d.pos.empty()) {
                    st.deferred_first_pos = static_cast<std::int64_t>(d.pos.front());
                    st.deferred_last_pos = static_cast<std::int64_t>(d.pos.back());
                }
            }
        } else if (ctx_dft != nullptr && mtp_mem_shared) {
            st.draft_pos = s.pos;
        }
#endif
        return st;
    };

    auto rebuild_slot_history_drafter = [&](ActiveBatchSlot& s, SlotSpecMode mode, bool enable_hybrid_suffix) -> bool {
        if (!s.slot_valid || s.pos <= 0 || s.committed_tokens.size() != static_cast<std::size_t>(s.pos)) {
            return false;
        }
        if (mode == SlotSpecMode::kSuffix || enable_hybrid_suffix) {
            if (!s.local_suffix_matcher) {
                s.local_suffix_matcher = std::make_unique<SuffixMatcher>(
                    BoundedSuffixAutomaton::kMinCapacity, BoundedSuffixAutomaton::kDefaultCapacity);
            }
            s.local_suffix_matcher->BeginRequest(s.committed_tokens);
        }
        if (mode == SlotSpecMode::kNgramMod) {
            if (ngram_mod == nullptr) return false;
            ngram_mod->BeginRequest(s.committed_tokens);
            s.ngram_indexed = s.committed_tokens.size() > ngram_mod->n_match()
                                  ? s.committed_tokens.size() - ngram_mod->n_match()
                                  : 0;
        } else if (mode == SlotSpecMode::kNgramShort) {
            if (!s.local_ngram_matcher) {
                s.local_ngram_matcher = std::make_unique<NgramMatcher>(/*exact_scan_fallback=*/true);
            }
            s.local_ngram_matcher->Reset();
            s.local_ngram_matcher->IndexRange(s.committed_tokens, 0, s.committed_tokens.size());
            s.ngram_indexed = s.committed_tokens.size();
        }
        s.spec_initialized = true;
        s.local_matcher_synced = true;
        s.spec_transition.MarkHistorySynced(true);
        return true;
    };

    auto suspend_slot_for_batch = [&](ActiveBatchSlot& s, SpecSuspendReason reason) {
        if (s.local_suffix_matcher != nullptr) {
            s.local_suffix_matcher->ClearActive();
        }
        s.spec_transition.SuspendForBatch(reason);
        sync_slot_flags_from_transition(s);
        s.spec_initialized = false;
        s.local_matcher_synced = false;
        s.batch_logit_idx = -1;
    };
    std::uint64_t next_admit_order = 1;

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
            (void)TrySetSampler(ctx, s.slot_id, nullptr);
            s.target_gpu_sampled = false;
        }
#else
        (void)s;
#endif
    };

    auto invalidate_slot = [&](ActiveBatchSlot& s) {
        s.slot_valid = false;
        s.spec_transition.MarkSlotInvalidated();
        sync_slot_flags_from_transition(s);
        s.committed_tokens.clear();
        s.local_matcher_synced = false;
        if (s.local_suffix_matcher != nullptr) {
            s.local_suffix_matcher->ClearActive();
        }
        if (prefix_coalescer.IsEnabled()) {
            (void)prefix_coalescer.OnSlotRemoved(s.slot_idx);
            SyncCoalescerStats();
        }
        llama_memory_seq_rm(mem, s.slot_id, -1, -1);
        ResetSwaSlot(s.slot_idx);
        if (s.slot_idx < slots.size()) {
            slots[s.slot_idx].tokens.clear();
            slots[s.slot_idx].session_key.clear();
            slots[s.slot_idx].last_used = 0;
        }
        if (s.slot_idx < slot_mtp_synced.size()) {
            slot_mtp_synced[s.slot_idx] = false;
        }
        if (s.slot_idx < slot_mtp_pending_h_pos.size()) {
            slot_mtp_pending_h_pos[s.slot_idx] = -1;
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
        if (prefix_coalescer.IsEnabled()) {
            (void)prefix_coalescer.OnSlotRemoved(s.slot_idx);
            SyncCoalescerStats();
        }
        detach_gpu_sampler(s);
        s.result.stop_reason = reason;
        if (s.prefill_done) {
            s.result.decode_ms = MsSince(s.decode_start);
        } else {
            s.result.prefill_ms = MsSince(s.prefill_start);
        }
        if (s.result.stage_timings.enabled) {
            s.result.stage_timings.prefill_ms = s.result.prefill_ms;
        }
        if (reason != StopReason::kError) {
            if ((s.ngram_mod_enabled || s.spec_transition.configured_mode() == SlotSpecMode::kNgramMod) &&
                ngram_mod != nullptr && !s.committed_tokens.empty()) {
                s.ngram_indexed =
                    ngram_mod->IndexRange(s.committed_tokens, s.ngram_indexed, s.committed_tokens.size());
            }
            if (suffix_matcher != nullptr) {
                if (s.slot_valid && s.slot_idx < slots.size() && !slots[s.slot_idx].tokens.empty()) {
                    suffix_matcher->IngestSessionSequence(slots[s.slot_idx].tokens);
                } else if (!s.committed_tokens.empty()) {
                    suffix_matcher->IngestSessionSequence(s.committed_tokens);
                }
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

    // Backend-owned safe-point suspension: parks ONE slot, optionally spills its
    // sequence state blob (bounded by max_spill_bytes), actively reclaims THAT
    // slot's physical KV memory, and completes its task without stalling peers.
    auto suspend_slot = [&](ActiveBatchSlot& s, SuspendAction action, std::size_t max_spill_bytes) {
        detach_gpu_sampler(s);
        s.result.suspended = true;
        s.result.suspend_action = action;
        if (s.slot_idx < slots.size()) {
            s.result.suspended_tokens = slots[s.slot_idx].tokens;
        }
        if (s.phase == BatchSlotPhase::kDecode && s.cur_tok != LLAMA_TOKEN_NULL) {
            s.result.suspended_tokens.push_back(s.cur_tok);
        }
        s.result.suspended_token_position = s.result.suspended_tokens.size();
        if (action == SuspendAction::kYieldSpill && s.slot_valid) {
            const std::size_t state_sz = llama_state_seq_get_size(ctx, s.slot_id);
            const std::size_t cap = std::min(max_spill_bytes, kMaxSequenceStateBytes);
            if (state_sz > 0 && state_sz <= cap) {
                s.result.suspended_state_blob.resize(state_sz);
                const std::size_t wrote = TryStateSeqGetData(
                    ctx, s.result.suspended_state_blob.data(), s.result.suspended_state_blob.size(), s.slot_id);
                if (wrote > 0 && wrote <= state_sz) {
                    s.result.suspended_state_blob.resize(wrote);
                } else {
                    s.result.suspended_state_blob.clear();
                }
            }
            if (options.prefix_cache && radix_cache.enabled() && s.slot_idx < slots.size() &&
                slots[s.slot_idx].tokens.size() >= kMinL2SnapshotTokens) {
                SnapshotSlotToL2(s.slot_idx, s.slot_id);
            }
        }
        invalidate_slot(s);
        s.result.physical_slot_reclaimed = (llama_memory_seq_pos_max(mem, s.slot_id) < 0);
        finish_slot(s, StopReason::kSuspended);
    };

    auto poll_slot_safe_point = [&](ActiveBatchSlot& s) -> bool {
        if (s.task == nullptr || s.task->req == nullptr || !s.task->req->on_safe_point) {
            return false;
        }
        SafePointInfo sp;
        sp.slot_index = s.slot_idx;
        sp.prompt_tokens = s.task->req->prompt.size();
        sp.prefill_completed_tokens = s.prefill_offset;
        sp.generated_tokens = s.result.generated_tokens;
        sp.kv_pos = s.pos;
        sp.active_peers = active.size() > 1 ? (active.size() - 1) : 0;
        sp.in_prefill = (s.phase == BatchSlotPhase::kPrefill);
        const SafePointDecision dec = s.task->req->on_safe_point(sp);
        if (dec.action == SuspendAction::kContinue) {
            return false;
        }
        suspend_slot(s, dec.action, dec.max_spill_bytes);
        return true;
    };

    auto admit_request = [&](PendingRequest* task, const SlotSelection& sel,
                             bool allow_l2_snapshot) -> std::unique_ptr<ActiveBatchSlot> {
        auto s = std::make_unique<ActiveBatchSlot>();
        s->task = task;
        s->slot_idx = sel.slot_index;
        s->slot_id = static_cast<llama_seq_id>(slots[s->slot_idx].id);
        s->admit_order = next_admit_order++;
        s->prefill_start = Clock::now();
        s->enqueue_time_us = task->enqueue_time_us;
        s->last_service_time_us = 0;
        slot_busy[s->slot_idx] = 1u;

        const GenerateRequest& req = *task->req;
        const std::uint64_t verify_seed =
            req.sampling.seed == 0xFFFFFFFFU ? std::random_device{}() : static_cast<std::uint64_t>(req.sampling.seed);
        s->verify_rng.seed(verify_seed ^ 0x517CC1B727220A95ULL);
        if (Status st = BuildSampler(req.sampling, vocab, n_vocab, static_cast<int32_t>(n_ctx), s->sampler,
                                     s->grammar_sampler, /*already_validated=*/true);
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
        if (!s->grammar_sampler && !req.sampling.compute_logprobs &&
            effective_spec_verification_mode != SpeculativeVerificationMode::StochasticRejection) {
            s->target_gpu_sampled = TrySetSampler(ctx, s->slot_id, s->sampler.get());
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

        const bool stage_prof = req.enable_stage_profiling;
        s->result.stage_timings.enabled = stage_prof;
        const Clock::time_point lookup_start = stage_prof ? Clock::now() : Clock::time_point{};
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
                if (src_kv >= static_cast<int64_t>(cand) &&
                    CanCloneSwaSource(j, mem, src_sid, static_cast<std::uint64_t>(cand))) {
                    best_src_idx = j;
                    best_src_lcp = cand;
                }
            }
            if (best_src_idx < slots.size() && best_src_lcp >= lcp + 32u) {
                const auto src_sid = static_cast<llama_seq_id>(slots[best_src_idx].id);
                llama_memory_seq_rm(mem, s->slot_id, -1, -1);
                ResetSwaSlot(s->slot_idx);
                ClearSlotCheckpoints(s->slot_idx);
                llama_memory_seq_cp(mem, src_sid, s->slot_id, 0, static_cast<llama_pos>(best_src_lcp));
                if (llama_memory_seq_pos_max(mem, s->slot_id) + 1 == static_cast<int64_t>(best_src_lcp)) {
                    lcp = best_src_lcp;
                    cloned_from_vram = true;
                    SyncClonedSwaSlot(s->slot_idx, best_src_idx, mem, s->slot_id, static_cast<std::uint64_t>(lcp));
                    slots[s->slot_idx].tokens.assign(req.prompt.begin(),
                                                     req.prompt.begin() + static_cast<std::ptrdiff_t>(lcp));
#if defined(SONDER_LLAMACPP_HAS_MTP)
                    if (ctx_dft != nullptr && s->slot_idx < mtp_pending_h.size()) {
                        mtp_deferred[s->slot_idx].Clear();
                        std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                        if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
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
                    if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
                }
            }
        }

        if (stage_prof) {
            add_stage_ms(s->result.stage_timings.cache_lookup_ms, MsSince(lookup_start));
        }

        bool restored_from_l2 = cloned_from_vram;
        bool physically_restored_from_l2 = false;
        std::size_t l2_lcp = 0;
        bool l2_wiped = false;
        if (!restored_from_l2 && options.prefix_cache && radix_cache.enabled() && req.prompt.size() > 1 &&
            req.prompt.size() - 1 > lcp && (req.prompt.size() - 1 - lcp) >= kMinL2RestoreGainTokens) {
            const auto l2_start = Clock::now();
            if (TryRestoreSlotFromL2(s->slot_idx, s->slot_id, req.prompt, req.session_key, lcp, l2_lcp, l2_mtp_synced,
                                     l2_wiped)) {
                const double l2_ms = MsSince(l2_start);
                lcp = l2_lcp;
                restored_from_l2 = true;
                physically_restored_from_l2 = true;
                s->result.host_l2_restored_tokens =
                    static_cast<std::int32_t>(std::min<std::size_t>(l2_lcp, static_cast<std::size_t>(INT32_MAX)));
                s->result.host_l2_restore_ms = l2_ms;
                if (stage_prof) {
                    add_stage_ms(s->result.stage_timings.l2_restore_ms, l2_ms);
                }
                slots[s->slot_idx].tokens.assign(req.prompt.begin(),
                                                 req.prompt.begin() + static_cast<std::ptrdiff_t>(lcp));
            } else if (l2_wiped) {
                lcp = 0;
                slots[s->slot_idx].tokens.clear();
                if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = false;
                if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
            }
        }

        if (!restored_from_l2) {
            if (lcp == 0) {
                llama_memory_seq_rm(mem, s->slot_id, -1, -1);
                ResetSwaSlot(s->slot_idx);
                ClearSlotCheckpoints(s->slot_idx);
                if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = true;
                if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
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
                    const bool swa_ok = CheckSwaRollback(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(lcp));
                    if (swa_ok && llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(lcp), -1)) {
                        PruneSlotCheckpointsAfter(s->slot_idx, lcp);
                    } else {
                        bool restored_from_ckpt = false;
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                        std::size_t ckpt_pos = 0;
                        if (!swa_compact_active &&
                            RestorePartialCheckpoint(s->slot_idx, s->slot_id, lcp, ckpt_pos, restored_pending_h)) {
                            lcp = ckpt_pos;
                            restored_from_ckpt = true;
                        }
#endif
                        if (!restored_from_ckpt) {
                            llama_memory_seq_rm(mem, s->slot_id, -1, -1);
                            ResetSwaSlot(s->slot_idx);
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
                        if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
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
                            if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
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
                                if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
                                if (s->slot_idx < slot_mtp_synced.size()) slot_mtp_synced[s->slot_idx] = false;
                            }
                        }
                        if (lcp > 0 && lcp != prev_slot_len && !restored_pending_h) {
                            std::fill(mtp_pending_h[s->slot_idx].begin(), mtp_pending_h[s->slot_idx].end(), 0.0F);
                            if (s->slot_idx < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[s->slot_idx] = -1;
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
        if (Status st = ValidateContextPosition(n_past, rem_prompt, n_ctx); !st.ok()) {
            invalidate_slot(*s);
            s->result.status = std::move(st);
            finish_slot(*s, StopReason::kError);
            return nullptr;
        }

        s->prefill_offset = lcp;
        s->pos = n_past;
        const std::int32_t reused_i32 =
            static_cast<std::int32_t>(std::min<std::size_t>(lcp, static_cast<std::size_t>(INT32_MAX)));
        s->result.cached_tokens = reused_i32;
        if (!physically_restored_from_l2) {
            s->result.device_prefix_reused_tokens = reused_i32;
            if (cloned_from_vram) {
                s->result.device_cross_slot_reused_tokens = reused_i32;
            }
        }
        slots[s->slot_idx].tokens.resize(lcp);
        slots[s->slot_idx].session_key = req.session_key;
        slots[s->slot_idx].last_used = ++slot_clock;
        s->committed_tokens.assign(slots[s->slot_idx].tokens.begin(), slots[s->slot_idx].tokens.end());

        SlotSpecCapability cap;
        cap.architecture = architecture;
        cap.hybrid_rs_cap = static_cast<std::int32_t>(rs_cap);
        cap.can_hybrid_spec_ckpt = can_hybrid_spec_ckpt;
        cap.fallback_suffix_draft_max =
            options.spec_draft_max > 0 ? options.spec_draft_max : kDefaultSuffixDraftMax;
#if defined(SONDER_LLAMACPP_HAS_MTP)
        const bool mtp_ok = (active_spec == SpeculativeType::kDraftMtp) && (ctx_dft != nullptr) &&
                            (!restored_from_l2 || l2_mtp_synced) &&
                            (s->slot_idx < slot_mtp_synced.size() && slot_mtp_synced[s->slot_idx]);
        if (mtp_ok && active_draft_max > 0) {
            cap.configured_mode = SlotSpecMode::kDraftMtp;
            cap.configured_draft_max = active_draft_max;
            cap.hybrid_suffix_configured = (suffix_matcher != nullptr);
            cap.suffix_fallback_available = (suffix_matcher != nullptr);
        }
#else
        (void)l2_mtp_synced;
#endif
        if (cap.configured_mode == SlotSpecMode::kNone && active_draft_max > 0) {
            if (active_spec == SpeculativeType::kSuffix && suffix_matcher != nullptr) {
                cap.configured_mode = SlotSpecMode::kSuffix;
                cap.configured_draft_max = active_draft_max;
            } else if (active_spec == SpeculativeType::kNgramMod && ngram_mod != nullptr) {
                cap.configured_mode = SlotSpecMode::kNgramMod;
                cap.configured_draft_max = active_draft_max;
            } else if (active_spec == SpeculativeType::kNgram) {
                cap.configured_mode = SlotSpecMode::kNgramShort;
                cap.configured_draft_max = active_draft_max;
            }
        }
        s->spec_transition.Configure(cap);
        sync_slot_flags_from_transition(*s);
        s->adaptive_ctrl.Reset(s->effective_draft_max);
        s->result.configured_spec_method = std::string(ToString(requested_spec));
        s->result.configured_draft_limit = options.spec_draft_max > 0 ? options.spec_draft_max : active_draft_max;
        if (s->mtp_enabled) {
            s->result.effective_spec_method = s->hybrid_suffix_enabled ? "draft-mtp+suffix" : "draft-mtp";
            s->result.effective_draft_limit = s->effective_draft_max;
            s->result.draft_identity = s->hybrid_suffix_enabled ? "mtp:native+suffix" : "mtp:native";
            s->result.spec_fallback_reason = load_spec_fallback_reason;
        } else if (s->suffix_enabled) {
            s->result.effective_spec_method = "suffix";
            s->result.effective_draft_limit = s->effective_draft_max;
            s->result.draft_identity = "suffix";
            s->result.spec_fallback_reason = load_spec_fallback_reason;
        } else if (s->ngram_mod_enabled) {
            s->result.effective_spec_method = "ngram-mod";
            s->result.effective_draft_limit = s->effective_draft_max;
            s->result.draft_identity = "ngram-mod";
            s->result.spec_fallback_reason = load_spec_fallback_reason;
        } else if (s->ngram_short_enabled) {
            s->result.effective_spec_method = "ngram";
            s->result.effective_draft_limit = s->effective_draft_max;
            s->result.draft_identity = "ngram";
            s->result.spec_fallback_reason = load_spec_fallback_reason;
        } else {
            s->result.effective_spec_method = "none";
            s->result.effective_draft_limit = 0;
            if (active_spec != SpeculativeType::kNone) {
                if (architecture == sonder::inference::ModelArchitecture::recurrent) {
                    s->result.spec_fallback_reason = "recurrent_unsupported";
                } else if (architecture == sonder::inference::ModelArchitecture::hybrid) {
                    s->result.spec_fallback_reason = "hybrid_unsupported";
                } else {
                    s->result.spec_fallback_reason =
                        load_spec_fallback_reason.empty() ? "not_supported" : load_spec_fallback_reason;
                }
            } else {
                s->result.spec_fallback_reason = load_spec_fallback_reason;
            }
        }
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
        ResetSwaSlot(victim);
        slots[victim].tokens.clear();
        slots[victim].session_key.clear();
        slots[victim].last_used = 0;
        if (victim < slot_mtp_synced.size()) slot_mtp_synced[victim] = false;
        if (victim < slot_mtp_pending_h_pos.size()) slot_mtp_pending_h_pos[victim] = -1;
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
    std::vector<PendingQueueSlotView> pending_views;
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
            prefix_coalescer.ResetActive();
            SyncCoalescerStats();
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
                    pending_views.clear();
                    pending_views.reserve(pending_queue.size());
                    for (const PendingRequest* p : pending_queue) {
                        PendingQueueSlotView pv;
                        pv.priority = p->req->priority;
                        pv.enqueue_time_us = p->enqueue_time_us;
                        pv.target_ttft_ms = p->req->target_ttft_ms;
                        pv.exclusive = p->exclusive;
                        pending_views.push_back(pv);
                    }
                    const auto pick =
                        SelectNextPendingRequest(pending_views, !active.empty(), NowUs(), batch_planner_config);
                    if (!pick.has_value()) break;
                    PendingRequest* picked = pending_queue[*pick];
                    if (picked->exclusive) {
                        pending_queue.erase(pending_queue.begin() + static_cast<std::ptrdiff_t>(*pick));
                        pending_count.fetch_sub(1, std::memory_order_release);
                        pending_exclusive_count.fetch_sub(1, std::memory_order_release);
                        exclusive_task = picked;
                    } else {
                        const auto chosen = SelectAvailablePrefixSlot(slots, slot_busy, picked->req->prompt,
                                                                      picked->req->session_key);
                        if (!chosen.has_value()) break;  // all n_slots busy
                        pending_queue.erase(pending_queue.begin() + static_cast<std::ptrdiff_t>(*pick));
                        pending_count.fetch_sub(1, std::memory_order_release);
                        admitted_task = picked;
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
                        if (prefix_coalescer.IsEnabled() && slot_obj->phase == BatchSlotPhase::kPrefill &&
                            slot_obj->prefill_offset < admitted_task->req->prompt.size()) {
                            (void)prefix_coalescer.OnAdmit(slot_obj->slot_idx, admitted_task->req->prompt,
                                                           slot_obj->prefill_offset, coalesce_now_ms());
                            SyncCoalescerStats();
                        }
                        active.push_back(std::move(slot_obj));
                    }
                    inflight_admit = nullptr;
                    inflight_admit_slot = slots.size();
                }
            }
        }

        // 3. Evict cancelled or selectively suspended slots before running the next batch step.
        for (auto it = active.begin(); it != active.end();) {
            if ((*it)->IsCancelled()) {
                finish_slot(**it, StopReason::kCancelled);
                it = active.erase(it);
            } else if (poll_slot_safe_point(**it)) {
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
                auto decode_start = Clock::now();
                Status decode_err;
                int32_t rc = TryDecode(ctx, batch, &decode_err);
                while (rc == 1 && evict_one_idle_slot()) {
                    abort_requested.store(false, std::memory_order_relaxed);
                    decode_start = Clock::now();
                    rc = TryDecode(ctx, batch, &decode_err);
                }
                const double prefill_decode_ms = MsSince(decode_start);
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
                        s.result.status = !decode_err.ok()
                                              ? std::move(decode_err)
                                              : Status::Error(ErrorCode::kDecodeFailed,
                                                              "llama_decode failed during prefill (rc=" +
                                                                  std::to_string(rc) + ")");
                        finish_slot(s, StopReason::kError);
                    }
                    active.clear();
                    continue;
                }
                step_cost_estimator.Observe(
                    StepCostObservation{0, n, std::max<std::int64_t>(0, s.pos), prefill_decode_ms});
                s.last_service_time_us = NowUs();
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
                if (!s.mtp_enabled) {
                    mark_slot_mtp_drifted(s);
                }
#endif
                if (s.slot_idx < slots.size()) {
                    auto& stoks = slots[s.slot_idx].tokens;
                    stoks.insert(stoks.end(), req.prompt.begin() + off, req.prompt.begin() + step_end);
                }
                s.committed_tokens.insert(s.committed_tokens.end(), req.prompt.begin() + off,
                                          req.prompt.begin() + step_end);
                s.result.prompt_tokens += n;
                s.pos += n;
                AdvanceSwaSlot(s.slot_idx, mem, s.slot_id, static_cast<std::uint64_t>(s.pos));
                s.prefill_offset = static_cast<std::size_t>(step_end);
                if (prefix_coalescer.IsEnabled() && s.slot_valid) {
                    const auto published = prefix_coalescer.OnProducerProgress(s.slot_idx, s.prefill_offset);
                    apply_coalesce_releases(published);
                }
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                if (s.hybrid_ckpt_offset == step_end && s.slot_valid) {
                    CapturePartialCheckpoint(s.slot_idx, s.slot_id, static_cast<std::size_t>(s.pos));
                }
#endif
                if (step_end == n_prompt) {
                    s.prefill_done = true;
                    s.result.prefill_ms = MsSince(s.prefill_start);
                    if (s.result.prompt_tokens > 0 && s.result.prefill_ms > 0.0) {
                        radix_cache.ObservePrefill(static_cast<std::size_t>(s.result.prompt_tokens),
                                                   s.result.prefill_ms);
                    }
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

            // Single-request fast-path decode step (with safe post-batch
            // reactivation and full per-slot isolated speculation!).
            if (s.spec_transition.suspended_for_batch()) {
                SpeculationReactivationInput rin;
                rin.active_count = active.size();
                rin.pending_count = pending_count.load(std::memory_order_acquire);
                rin.pending_exclusive = pending_exclusive_count.load(std::memory_order_acquire) > 0;
                rin.cancelled = s.IsCancelled();
                rin.in_decode_phase = (s.phase == BatchSlotPhase::kDecode);
                rin.slot_valid = s.slot_valid;
                rin.remaining_max_tokens =
                    static_cast<std::int64_t>(req.max_new_tokens) - static_cast<std::int64_t>(s.result.generated_tokens);
                rin.remaining_ctx_tokens = n_ctx - s.pos;
                rin.target_pos = s.pos;
                rin.committed_history_len = s.committed_tokens.size();
                rin.history_matches_target_pos =
                    s.slot_valid && s.pos > 0 && s.committed_tokens.size() == static_cast<std::size_t>(s.pos);
                rin.mtp_readiness = build_mtp_readiness(s);

                const SpeculationTransitionDecision dec =
                    s.spec_transition.EvaluateSingleRequestStep(rin, reactivation_policy);
                if (dec.should_reactivate) {
                    bool ready_ok = true;
                    if (dec.requires_history_rebuild) {
                        ready_ok = rebuild_slot_history_drafter(s, dec.target_mode, dec.enable_hybrid_suffix);
                    }
                    if (ready_ok) {
                        s.spec_transition.CommitReactivation(dec);
                        sync_slot_flags_from_transition(s);
                        s.adaptive_ctrl.Reset(s.effective_draft_max);
                        s.result.spec_transitioned_during_request = true;
                        s.result.effective_draft_limit = s.effective_draft_max;
                        if (s.mtp_enabled) {
                            s.result.effective_spec_method = s.hybrid_suffix_enabled ? "draft-mtp+suffix" : "draft-mtp";
                            s.result.draft_identity = s.hybrid_suffix_enabled ? "mtp:native+suffix" : "mtp:native";
                            s.result.spec_fallback_reason = load_spec_fallback_reason;
                        } else if (s.suffix_enabled) {
                            s.result.effective_spec_method = "suffix";
                            s.result.draft_identity = "suffix";
                            s.result.spec_fallback_reason =
                                (s.spec_transition.configured_mode() == SlotSpecMode::kDraftMtp)
                                    ? "mtp_drifted_fallback_suffix"
                                    : load_spec_fallback_reason;
                        } else if (s.ngram_mod_enabled) {
                            s.result.effective_spec_method = "ngram-mod";
                            s.result.draft_identity = "ngram-mod";
                            s.result.spec_fallback_reason = load_spec_fallback_reason;
                        } else if (s.ngram_short_enabled) {
                            s.result.effective_spec_method = "ngram";
                            s.result.draft_identity = "ngram";
                            s.result.spec_fallback_reason = load_spec_fallback_reason;
                        }
                    }
                }
            } else if (!s.spec_initialized && s.spec_transition.active_mode() != SlotSpecMode::kNone) {
                const SlotSpecMode cur_mode = s.spec_transition.active_mode();
                if (IsHistoryBasedSpecMode(cur_mode) || s.hybrid_suffix_enabled) {
                    (void)rebuild_slot_history_drafter(s, cur_mode, s.hybrid_suffix_enabled);
                } else {
                    s.spec_initialized = true;
                }
            }

            const auto step_start = Clock::now();
            const int64_t rem_gen =
                static_cast<int64_t>(req.max_new_tokens) - static_cast<int64_t>(s.result.generated_tokens);
            const int64_t max_k = std::min<int64_t>(
                {rem_gen - 1, n_ctx - s.pos - 1, static_cast<int64_t>(step_token_budget) - 1});

            auto& draft_tokens = s.draft_tokens;
            auto& accepted_tokens = s.accepted_tokens;
            auto& accepted_lps = s.accepted_lps;
            draft_tokens.clear();
            bool step_used_suffix = false;
            const BoundedSuffixAutomaton* shared_ring =
                (suffix_matcher != nullptr) ? &suffix_matcher->session_ring() : nullptr;
            const bool slot_stage_prof = s.result.stage_timings.enabled;
            const bool slot_spec_active =
                s.mtp_enabled || s.suffix_enabled || s.ngram_mod_enabled || s.ngram_short_enabled;
            const Clock::time_point draft_start =
                (slot_stage_prof && slot_spec_active && max_k > 0) ? Clock::now() : Clock::time_point{};

            if (max_k > 0) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (s.mtp_enabled) {
                    if (s.hybrid_suffix_enabled && s.local_suffix_matcher != nullptr &&
                        (architecture != sonder::inference::ModelArchitecture::hybrid || can_hybrid_spec_ckpt ||
                         rs_cap >= SuffixMatcher::kHybridMtpMinBurstDraft)) {
                        if (s.local_suffix_matcher->paused()) {
                            s.local_suffix_matcher->NoteSkippedStep();
                        } else {
                            const std::size_t hybrid_cap =
                                (architecture == sonder::inference::ModelArchitecture::hybrid && !can_hybrid_spec_ckpt)
                                    ? std::min<std::size_t>(static_cast<std::size_t>(max_k), static_cast<std::size_t>(rs_cap))
                                    : static_cast<std::size_t>(max_k);
                            SuffixMatchResult burst = s.local_suffix_matcher->TryHybridMtpBurstForCurrent(
                                s.cur_tok, hybrid_cap, shared_ring);
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
                            s.spec_transition.MarkMtpRuntimeFailure();
                            s.spec_transition.MarkSlotInvalidated();
                            sync_slot_flags_from_transition(s);
                            s.result.spec_transitioned_during_request = true;
                            s.result.effective_spec_method = "none";
                            s.result.effective_draft_limit = 0;
                            s.result.spec_fallback_reason = "draft_sync_failed";
                            s.slot_valid = false;
                            s.committed_tokens.clear();
                            if (s.slot_idx < slots.size()) {
                                slots[s.slot_idx].tokens.clear();
                                slots[s.slot_idx].session_key.clear();
                            }
                            mark_slot_mtp_drifted(s);
                            ClearSlotCheckpoints(s.slot_idx);
                            draft_tokens.clear();
                        }
                    }
                } else
#endif
                if (s.suffix_enabled && s.local_suffix_matcher != nullptr) {
                    if (s.local_suffix_matcher->paused()) {
                        s.local_suffix_matcher->NoteSkippedStep();
                    } else {
                        const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, s.effective_draft_max));
                        SuffixMatchResult m = s.local_suffix_matcher->DraftForCurrent(
                            s.cur_tok, k_lim, SuffixMatcher::kDefaultMinMatchLen, shared_ring);
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
                    ngram_mod->Draft(s.committed_tokens, s.cur_tok, k_lim, draft_tokens);
                    for (std::size_t j = 0; j < draft_tokens.size(); ++j) {
                        if (llama_vocab_is_eog(vocab, draft_tokens[j])) {
                            draft_tokens.resize(j + 1);
                            break;
                        }
                    }
                } else if (s.ngram_short_enabled && s.local_ngram_matcher != nullptr) {
                    const int32_t ctrl_len = s.adaptive_ctrl.NextDraftLength();
                    if (ctrl_len > 0) {
                        s.committed_tokens.push_back(s.cur_tok);
                        const auto k_lim = static_cast<std::size_t>(std::min<int64_t>(max_k, ctrl_len));
                        s.local_ngram_matcher->Draft(s.committed_tokens, k_lim, draft_tokens);
                        s.committed_tokens.pop_back();
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
            if (slot_stage_prof && slot_spec_active && max_k > 0) {
                add_stage_ms(s.result.stage_timings.draft_ms, MsSince(draft_start));
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
                        TryStateSeqGetDataExt(ctx, spec_ckpt_buf.data(), sz, s.slot_id, flags);
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

            auto decode_start = Clock::now();
            Status decode_err;
            int32_t rc = TryDecode(ctx, batch, &decode_err);
            while (rc == 1 && evict_one_idle_slot()) {
                abort_requested.store(false, std::memory_order_relaxed);
                decode_start = Clock::now();
                rc = TryDecode(ctx, batch, &decode_err);
            }
            const double decode_call_ms = MsSince(decode_start);
            if (slot_stage_prof && !draft_tokens.empty()) {
                add_stage_ms(s.result.stage_timings.verify_ms, decode_call_ms);
            }
            batch_abort_fn = nullptr;
            active_cancel = nullptr;
            if (rc != 0) {
                bool rolled_back =
                    CheckSwaRollback(s.slot_idx, mem, s.slot_id, static_cast<std::uint64_t>(s.pos)) &&
                    llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(s.pos), -1);
                if (!rolled_back) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                    if (have_spec_ckpt && !swa_compact_active) {
                        const llama_state_seq_flags flags = PartialCheckpointFlags();
                        const std::size_t read = TryStateSeqSetDataExt(ctx, spec_ckpt_buf.data(), spec_ckpt_size,
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
                    s.result.status = !decode_err.ok()
                                          ? std::move(decode_err)
                                          : Status::Error(ErrorCode::kDecodeFailed,
                                                          "llama_decode failed (rc=" + std::to_string(rc) + ")");
                    finish_slot(s, StopReason::kError);
                }
                active.clear();
                continue;
            }
            if (draft_tokens.empty()) {
                step_cost_estimator.Observe(
                    StepCostObservation{1, 0, std::max<std::int64_t>(0, s.pos), decode_call_ms});
            }
            s.last_service_time_us = NowUs();

            if (draft_tokens.empty()) {
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (s.mtp_enabled && !MtpCommitVerified(batch, s.slot_id, 0)) {
                    invalidate_slot(s);
                    s.result.status = Status::Error(ErrorCode::kDecodeFailed, "MTP draft context sync failed");
                    finish_slot(s, StopReason::kError);
                    active.clear();
                    continue;
                }
                if (!s.mtp_enabled) {
                    mark_slot_mtp_drifted(s);
                }
#endif
                if (s.slot_valid && s.slot_idx < slots.size()) slots[s.slot_idx].tokens.push_back(s.cur_tok);
                s.committed_tokens.push_back(s.cur_tok);
                if ((s.suffix_enabled || s.hybrid_suffix_enabled) && s.local_suffix_matcher != nullptr) {
                    s.local_suffix_matcher->CommitToken(s.cur_tok);
                    s.local_suffix_matcher->ObserveBaselineStep(decode_call_ms, 1u);
                }
                if (s.ngram_mod_enabled && ngram_mod != nullptr) {
                    s.ngram_indexed = ngram_mod->IndexRange(s.committed_tokens, s.ngram_indexed, s.committed_tokens.size());
                } else if (s.ngram_short_enabled && s.local_ngram_matcher != nullptr) {
                    s.local_ngram_matcher->IndexRange(s.committed_tokens, s.ngram_indexed, s.committed_tokens.size());
                    s.ngram_indexed = s.committed_tokens.size();
                }
                ++s.pos;
                AdvanceSwaSlot(s.slot_idx, mem, s.slot_id, static_cast<std::uint64_t>(s.pos));

                TokenLogprobInfo step_lp;
                const Clock::time_point samp_start = slot_stage_prof ? Clock::now() : Clock::time_point{};
                const Token next_tok = SampleTokenAt(0, req.sampling, s.sampler.get(), s.grammar_sampler.get(),
                                                     s.target_gpu_sampled, n_vocab,
                                                     req.sampling.compute_logprobs ? &step_lp : nullptr);
                if (slot_stage_prof) {
                    add_stage_ms(s.result.stage_timings.sampling_ms, MsSince(samp_start));
                }
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
                    if (s.ngram_mod_enabled) s.committed_tokens.push_back(next_tok);
                    if ((s.suffix_enabled || s.hybrid_suffix_enabled) && s.local_suffix_matcher != nullptr) {
                        s.local_suffix_matcher->CommitToken(next_tok);
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
            std::size_t n_accepted = 0;
            const std::span<const std::vector<double>> no_proposal_dists{};
            const Clock::time_point spec_samp_start = slot_stage_prof ? Clock::now() : Clock::time_point{};
            const bool sample_deadlock = !VerifySpeculativeBatch(
                draft_tokens, no_proposal_dists, req.sampling, s.sampler.get(), s.grammar_sampler.get(),
                s.target_gpu_sampled, n_vocab, s.verify_rng, accepted_tokens, accepted_lps, n_accepted, s.result);
            if (slot_stage_prof) {
                add_stage_ms(s.result.stage_timings.sampling_ms, MsSince(spec_samp_start));
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
            if (step_used_suffix && s.local_suffix_matcher != nullptr) {
                s.local_suffix_matcher->ObserveWithCost(k, n_accepted, decode_call_ms, accepted_tokens.size());
            } else if (s.ngram_mod_enabled && ngram_mod != nullptr) {
                if (ngram_mod->Observe(k, n_accepted)) s.ngram_indexed = 0;
            } else if (s.ngram_short_enabled) {
                s.adaptive_ctrl.Observe(static_cast<std::int32_t>(k), static_cast<std::int32_t>(n_accepted));
            }

            if (s.slot_valid && s.slot_idx < slots.size()) slots[s.slot_idx].tokens.push_back(step_base_tok);
            s.committed_tokens.push_back(step_base_tok);
            if ((s.suffix_enabled || s.hybrid_suffix_enabled) && s.local_suffix_matcher != nullptr) {
                s.local_suffix_matcher->CommitToken(step_base_tok);
            }

            int64_t new_pos = s.pos + 1;
            std::size_t committed_drafts = 0;
            auto rollback_rejected = [&](int64_t keep_pos, std::size_t keep_drafts) -> bool {
                const Clock::time_point rb_start = slot_stage_prof ? Clock::now() : Clock::time_point{};
                auto finish_rb = [&](bool ok) {
                    if (slot_stage_prof) {
                        add_stage_ms(s.result.stage_timings.rollback_ms, MsSince(rb_start));
                    }
                    return ok;
                };
#if defined(SONDER_LLAMACPP_HAS_MTP)
                if (s.mtp_enabled && !MtpCommitVerified(batch, s.slot_id, static_cast<int32_t>(keep_drafts))) {
                    invalidate_slot(s);
                    return finish_rb(false);
                }
                if (!s.mtp_enabled) {
                    mark_slot_mtp_drifted(s);
                }
#endif
                const SlotKvRollbackRange rb = ComputeSlotRollbackRange(s.slot_id, s.pos, k, keep_drafts);
                AdvanceSwaSlot(s.slot_idx, mem, s.slot_id, static_cast<std::uint64_t>(rb.batch_end_pos));
                if (rb.needs_rollback || keep_pos < rb.batch_end_pos) {
                    const int64_t rm_pos = std::min(keep_pos, rb.rm_from_pos);
                    const bool swa_ok =
                        CheckSwaRollback(s.slot_idx, mem, s.slot_id, static_cast<std::uint64_t>(rm_pos));
                    if (!swa_ok || !llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(rm_pos), -1)) {
#if defined(SONDER_LLAMACPP_HAS_STATE_SEQ_EXT)
                        if (have_spec_ckpt && !swa_compact_active) {
                            const llama_state_seq_flags flags = PartialCheckpointFlags();
                            const std::size_t read = TryStateSeqSetDataExt(ctx, spec_ckpt_buf.data(),
                                                                           spec_ckpt_size, s.slot_id, flags);
                            if (read == 0 || !llama_memory_seq_rm(mem, s.slot_id, static_cast<llama_pos>(s.pos), -1)) {
                                invalidate_slot(s);
                                return finish_rb(false);
                            }
                            BatchClear(batch);
                            BatchAdd(batch, step_base_tok, static_cast<llama_pos>(s.pos), s.slot_id, /*logits=*/false);
                            for (std::size_t j = 0; j < keep_drafts; ++j) {
                                BatchAdd(batch, draft_tokens[j],
                                         static_cast<llama_pos>(s.pos + 1 + static_cast<int64_t>(j)), s.slot_id,
                                         /*logits=*/false);
                            }
                            if (TryDecode(ctx, batch) != 0) {
                                invalidate_slot(s);
                                return finish_rb(false);
                            }
                            return finish_rb(true);
                        }
#else
                        (void)step_base_tok;
                        (void)keep_drafts;
#endif
                        invalidate_slot(s);
                        return finish_rb(false);
                    }
                }
                return finish_rb(true);
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
                    s.committed_tokens.push_back(tok);
                    if ((s.suffix_enabled || s.hybrid_suffix_enabled) && s.local_suffix_matcher != nullptr) {
                        s.local_suffix_matcher->CommitToken(tok);
                    }
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
                    if (is_bonus && s.ngram_mod_enabled) s.committed_tokens.push_back(tok);
                    if (is_bonus && (s.suffix_enabled || s.hybrid_suffix_enabled) && s.local_suffix_matcher != nullptr) {
                        s.local_suffix_matcher->CommitToken(tok);
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
                s.ngram_indexed = ngram_mod->IndexRange(s.committed_tokens, s.ngram_indexed, s.committed_tokens.size());
            } else if (s.ngram_short_enabled && s.local_ngram_matcher != nullptr) {
                s.local_ngram_matcher->IndexRange(s.committed_tokens, s.ngram_indexed, s.committed_tokens.size());
                s.ngram_indexed = s.committed_tokens.size();
            }
            Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(accepted_tokens.size()), MsSince(step_start));
            s.pos = new_pos;
            if (s.pos >= n_ctx) {
                finish_slot(s, StopReason::kContextFull);
                active.clear();
            }
            continue;
        }

        // 5. Case B: Multi-request continuous batching.
        // First attempt opt-in, cost-gated bounded small-batch concurrent linear
        // speculation when pure-decode slots are within threshold and rollback-safe;
        // otherwise immediately fall back to ordinary 1-token continuous batching.
        std::size_t decoding_count = 0;
        std::size_t prefill_count = 0;
        for (const auto& sp : active) {
            if (sp->IsCancelled()) continue;
            if (sp->phase == BatchSlotPhase::kDecode) {
                ++decoding_count;
            } else if (sp->phase == BatchSlotPhase::kPrefill) {
                ++prefill_count;
            }
        }

        bool ran_concurrent_spec = false;
        if (concurrent_spec_policy.enabled && !concurrent_cost_gate.IsRegressed() && prefill_count == 0u &&
            prefix_coalescer.HeldFollowerCount() == 0u &&
            decoding_count >= 2u && decoding_count <= static_cast<std::size_t>(concurrent_spec_policy.max_concurrent_slots) &&
            architecture != sonder::inference::ModelArchitecture::recurrent &&
            !(architecture == sonder::inference::ModelArchitecture::hybrid && rs_cap == 0u) &&
            active_spec != SpeculativeType::kNone) {
            const BoundedSuffixAutomaton* shared_ring =
                (suffix_matcher != nullptr) ? &suffix_matcher->session_ring() : nullptr;

            std::vector<ConcurrentSpecSlotInput> conc_inputs;
            conc_inputs.reserve(active.size());
            for (auto& sp : active) {
                ActiveBatchSlot& s = *sp;
                s.draft_tokens.clear();
                s.concurrent_step_mode = SlotSpecMode::kNone;

                ConcurrentSpecSlotInput in;
                in.slot_index = s.slot_idx;
                in.seq_id = static_cast<std::int32_t>(s.slot_id);
                in.base_pos = s.pos;
                in.is_prefill = (s.phase == BatchSlotPhase::kPrefill);
                in.cancelled = s.IsCancelled();

                if (!in.cancelled && !in.is_prefill && s.slot_valid && s.pos > 0 &&
                    s.committed_tokens.size() == static_cast<std::size_t>(s.pos)) {
                    const GenerateRequest& req = *s.task->req;
                    const SlotSpecCapability& cap = s.spec_transition.capability();
                    SlotSpecMode cmode = cap.configured_mode;
                    std::int32_t req_draft_cap = cap.configured_draft_max;
                    if (cmode == SlotSpecMode::kDraftMtp) {
                        if (cap.suffix_fallback_available && suffix_matcher != nullptr) {
                            cmode = SlotSpecMode::kSuffix;
                            req_draft_cap = cap.fallback_suffix_draft_max;
                        } else {
                            cmode = SlotSpecMode::kNone;
                            req_draft_cap = 0;
                        }
                    }
                    const ArchitectureSpecGuardResult guard = CheckArchitectureSpecGuard(
                        architecture, cmode, req_draft_cap, static_cast<std::int32_t>(rs_cap),
                        can_hybrid_spec_ckpt, /*allow_partial_checkpoint=*/false);

                    const int64_t rem_gen =
                        static_cast<int64_t>(req.max_new_tokens) - static_cast<int64_t>(s.result.generated_tokens);
                    const int64_t rem_ctx = n_ctx - s.pos;
                    const int64_t slot_lim = std::min<int64_t>({
                        rem_gen - 1,
                        rem_ctx - 1,
                        static_cast<int64_t>(guard.effective_draft_max),
                        static_cast<int64_t>(concurrent_spec_policy.max_per_slot_draft_tokens),
                        static_cast<int64_t>(per_seq_output_budget) - 1,
                    });

                    if (guard.allowed && slot_lim > 0) {
                        s.concurrent_step_mode = cmode;
                        if (!s.local_matcher_synced) {
                            (void)rebuild_slot_history_drafter(s, cmode, /*enable_hybrid_suffix=*/false);
                        }
                        if (cmode == SlotSpecMode::kSuffix && s.local_suffix_matcher != nullptr) {
                            if (s.local_suffix_matcher->paused()) {
                                s.local_suffix_matcher->NoteSkippedStep();
                            } else {
                                SuffixMatchResult m = s.local_suffix_matcher->DraftForCurrent(
                                    s.cur_tok, static_cast<std::size_t>(slot_lim),
                                    SuffixMatcher::kDefaultMinMatchLen, shared_ring);
                                if (!m.draft.empty()) {
                                    s.draft_tokens = std::move(m.draft);
                                }
                            }
                        } else if (cmode == SlotSpecMode::kNgramMod && ngram_mod != nullptr) {
                            ngram_mod->Draft(s.committed_tokens, s.cur_tok, static_cast<std::size_t>(slot_lim),
                                             s.draft_tokens);
                        } else if (cmode == SlotSpecMode::kNgramShort && s.local_ngram_matcher != nullptr) {
                            const int32_t ctrl_len = s.adaptive_ctrl.NextDraftLength();
                            if (ctrl_len > 0) {
                                const auto k_lim = static_cast<std::size_t>(
                                    std::min<int64_t>(slot_lim, static_cast<int64_t>(ctrl_len)));
                                s.committed_tokens.push_back(s.cur_tok);
                                s.local_ngram_matcher->Draft(s.committed_tokens, k_lim, s.draft_tokens);
                                s.committed_tokens.pop_back();
                            } else {
                                s.adaptive_ctrl.NoteSkippedStep();
                            }
                        }
                        for (std::size_t j = 0; j < s.draft_tokens.size(); ++j) {
                            if (llama_vocab_is_eog(vocab, s.draft_tokens[j])) {
                                s.draft_tokens.resize(j + 1);
                                break;
                            }
                        }
                        if (!s.draft_tokens.empty()) {
                            in.eligible_for_spec = true;
                            in.max_draft_for_slot = static_cast<std::int32_t>(slot_lim);
                            in.proposed_draft_count = s.draft_tokens.size();
                        }
                    }
                }
                conc_inputs.push_back(in);
            }

            const ConcurrentSpecBatchPlan cplan = PlanConcurrentLinearSpeculation(
                conc_inputs, concurrent_spec_policy, step_token_budget, concurrent_output_budget,
                per_seq_output_budget, architecture, static_cast<std::int32_t>(rs_cap),
                concurrent_cost_gate.IsRegressed(), conc_rr_cursor);

            if (cplan.use_concurrent_spec && cplan.total_draft_tokens > 0) {
                BatchClear(batch);
                for (const ConcurrentSpecSlotPlan& item : cplan.slots) {
                    ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                    if (s == nullptr) continue;
                    s->draft_tokens.resize(static_cast<std::size_t>(item.allocated_draft_tokens));
                    s->batch_start_idx = batch.n_tokens;
                    BatchAdd(batch, s->cur_tok, static_cast<llama_pos>(s->pos), s->slot_id, /*logits=*/true);
                    for (std::int32_t j = 0; j < item.allocated_draft_tokens; ++j) {
                        BatchAdd(batch, s->draft_tokens[static_cast<std::size_t>(j)],
                                 static_cast<llama_pos>(s->pos + 1 + j), s->slot_id, /*logits=*/true);
                    }
                }

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
                auto decode_start = step_start;
                Status decode_err;
                int32_t rc = TryDecode(ctx, batch, &decode_err);
                while (rc == 1 && evict_one_idle_slot()) {
                    abort_requested.store(false, std::memory_order_relaxed);
                    decode_start = Clock::now();
                    rc = TryDecode(ctx, batch, &decode_err);
                }
                const double decode_call_ms = MsSince(decode_start);
                batch_abort_fn = nullptr;

                if (rc == 1) {
                    // KV pressure during concurrent speculation: clean up per-slot
                    // speculative tails and fall through immediately to ordinary
                    // 1-token continuous batching.
                    for (const ConcurrentSpecSlotPlan& item : cplan.slots) {
                        ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                        if (s == nullptr) continue;
                        const bool swa_ok =
                            CheckSwaRollback(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(s->pos));
                        if (!swa_ok || !llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(s->pos), -1)) {
                            invalidate_slot(*s);
                        }
                        s->draft_tokens.clear();
                    }
                } else if (rc != 0) {
                    ran_concurrent_spec = true;
                    const bool stopping = worker_stop.load(std::memory_order_acquire);
                    for (const ConcurrentSpecSlotPlan& item : cplan.slots) {
                        ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                        if (s == nullptr) continue;
                        const bool swa_ok =
                            CheckSwaRollback(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(s->pos));
                        if (!swa_ok || !llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(s->pos), -1)) {
                            invalidate_slot(*s);
                        }
                        if (rc == 2 && (s->IsCancelled() || stopping)) {
                            finish_slot(*s, StopReason::kCancelled);
                        } else if (rc != 2) {
                            s->result.status = !decode_err.ok()
                                                   ? decode_err
                                                   : Status::Error(ErrorCode::kDecodeFailed,
                                                                   "llama_decode failed (rc=" + std::to_string(rc) +
                                                                       ")");
                            finish_slot(*s, StopReason::kError);
                        }
                    }
                    active.erase(std::remove_if(active.begin(), active.end(),
                                                [](const auto& sp) { return sp->task == nullptr; }),
                                 active.end());
                } else {
                    ran_concurrent_spec = true;
                    const double step_ms = MsSince(step_start);
                    const std::int64_t serviced_us = NowUs();

                    // Phase 1: Extract every slot's verified tokens from ONLY its
                    // own disjoint `[batch_start_idx, batch_start_idx + k_i]` span
                    // and its own isolated sampler BEFORE any callbacks or KV writes.
                    for (const ConcurrentSpecSlotPlan& item : cplan.slots) {
                        ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                        if (s == nullptr || s->task == nullptr) continue;
                        const GenerateRequest& req = *s->task->req;
                        const std::size_t k_i = static_cast<std::size_t>(item.allocated_draft_tokens);
                        const auto verify_start = Clock::now();
                        s->accepted_tokens.clear();
                        s->accepted_lps.clear();
                        s->step_n_accepted = 0;
                        s->step_sample_deadlock = false;

                        for (std::size_t j = 0; j < k_i; ++j) {
                            TokenLogprobInfo lp;
                            const Token sampled = SampleTokenAt(
                                s->batch_start_idx + static_cast<int32_t>(j), req.sampling, s->sampler.get(),
                                s->grammar_sampler.get(), s->target_gpu_sampled, n_vocab,
                                req.sampling.compute_logprobs ? &lp : nullptr);
                            if (sampled == LLAMA_TOKEN_NULL) {
                                s->step_sample_deadlock = true;
                                break;
                            }
                            s->accepted_tokens.push_back(sampled);
                            s->accepted_lps.push_back(std::move(lp));
                            if (sampled == s->draft_tokens[j]) {
                                ++s->step_n_accepted;
                            } else {
                                break;
                            }
                        }
                        bool sampled_bonus = false;
                        if (!s->step_sample_deadlock && s->step_n_accepted == k_i) {
                            TokenLogprobInfo bonus_lp;
                            const Token bonus = SampleTokenAt(
                                s->batch_start_idx + static_cast<int32_t>(k_i), req.sampling, s->sampler.get(),
                                s->grammar_sampler.get(), s->target_gpu_sampled, n_vocab,
                                req.sampling.compute_logprobs ? &bonus_lp : nullptr);
                            if (bonus == LLAMA_TOKEN_NULL) {
                                s->step_sample_deadlock = true;
                            } else {
                                s->accepted_tokens.push_back(bonus);
                                s->accepted_lps.push_back(std::move(bonus_lp));
                                sampled_bonus = true;
                            }
                        }
                        if (!s->step_sample_deadlock && k_i > 0u) {
                            const double verify_us =
                                std::chrono::duration<double, std::micro>(Clock::now() - verify_start).count();
                            const double safe_us = (std::isfinite(verify_us) && verify_us > 0.0) ? verify_us : 0.0;
                            s->result.spec_verifier_mode = SpeculativeVerificationMode::TargetMatch;
                            s->result.spec_verify_overhead_us += safe_us;
                            spec_verifier.record_external_step(SpeculativeVerificationMode::TargetMatch, k_i,
                                                               s->step_n_accepted, s->step_n_accepted < k_i,
                                                               sampled_bonus, safe_us);
                        }
                    }

                    // Phase 2: Commit per slot with exact isolated per-sequence
                    // `llama_memory_seq_rm` rollback so one slot's rejection never
                    // affects any sibling slot's KV or sampler history.
                    std::size_t total_committed_step = 0;
                    std::size_t total_drafted_step = 0;
                    std::size_t total_accepted_step = 0;

                    for (const ConcurrentSpecSlotPlan& item : cplan.slots) {
                        ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                        if (s == nullptr || s->task == nullptr) continue;
                        s->last_service_time_us = serviced_us;
                        const GenerateRequest& req = *s->task->req;
                        const std::size_t k_i = static_cast<std::size_t>(item.allocated_draft_tokens);
                        const Token step_base_tok = s->cur_tok;
                        const int64_t batch_end_pos = s->pos + 1 + static_cast<int64_t>(k_i);
                        AdvanceSwaSlot(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(batch_end_pos));

                        mark_slot_mtp_drifted(*s);
                        if (s->spec_transition.configured_mode() == SlotSpecMode::kDraftMtp) {
                            s->spec_transition.SuspendForBatch(SpecSuspendReason::kMultiRequestBatch);
                            sync_slot_flags_from_transition(*s);
                        }

                        if (s->step_sample_deadlock) {
                            const bool swa_ok =
                                CheckSwaRollback(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(s->pos));
                            if (!swa_ok || !llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(s->pos), -1)) {
                                invalidate_slot(*s);
                            }
                            s->result.status = Status::Error(
                                ErrorCode::kDecodeFailed, "sampler produced no valid token (grammar deadlock)");
                            finish_slot(*s, StopReason::kError);
                            continue;
                        }

                        if (k_i > 0) {
                            s->result.draft_tokens += static_cast<std::int32_t>(k_i);
                            s->result.draft_accepted_tokens += static_cast<std::int32_t>(s->step_n_accepted);
                            if (s->concurrent_step_mode == SlotSpecMode::kSuffix) {
                                s->result.effective_spec_method = "suffix";
                                s->result.draft_identity = "suffix";
                            } else if (s->concurrent_step_mode == SlotSpecMode::kNgramMod) {
                                s->result.effective_spec_method = "ngram-mod";
                                s->result.draft_identity = "ngram-mod";
                            } else if (s->concurrent_step_mode == SlotSpecMode::kNgramShort) {
                                s->result.effective_spec_method = "ngram";
                                s->result.draft_identity = "ngram";
                            }
                            s->result.effective_draft_limit =
                                std::max(s->result.effective_draft_limit, static_cast<std::int32_t>(k_i));
                            if (s->spec_transition.configured_mode() == SlotSpecMode::kDraftMtp) {
                                s->result.spec_transitioned_during_request = true;
                                s->result.spec_fallback_reason = "concurrency_batch";
                            }
                            if (s->result.stage_timings.enabled) {
                                add_stage_ms(s->result.stage_timings.verify_ms, decode_call_ms);
                                s->result.stage_timings.stages_may_overlap = true;
                            }
                            total_drafted_step += k_i;
                            total_accepted_step += s->step_n_accepted;
                            if (s->concurrent_step_mode == SlotSpecMode::kSuffix && s->local_suffix_matcher != nullptr) {
                                s->local_suffix_matcher->ObserveWithCost(k_i, s->step_n_accepted, decode_call_ms,
                                                                         s->accepted_tokens.size());
                            } else if (s->concurrent_step_mode == SlotSpecMode::kNgramShort) {
                                s->adaptive_ctrl.Observe(static_cast<std::int32_t>(k_i),
                                                         static_cast<std::int32_t>(s->step_n_accepted));
                            }
                        }

                        if (s->slot_valid && s->slot_idx < slots.size()) {
                            slots[s->slot_idx].tokens.push_back(step_base_tok);
                        }
                        s->committed_tokens.push_back(step_base_tok);
                        if (s->local_suffix_matcher != nullptr && s->local_matcher_synced) {
                            s->local_suffix_matcher->CommitToken(step_base_tok);
                        }

                        int64_t new_pos = s->pos + 1;
                        std::size_t committed_drafts = 0;
                        auto rollback_slot_tail = [&](std::size_t keep_drafts) -> bool {
                            const SlotKvRollbackRange rb =
                                ComputeSlotRollbackRange(s->slot_id, s->pos, k_i, keep_drafts);
                            if (rb.needs_rollback) {
                                const bool swa_ok = CheckSwaRollback(s->slot_idx, mem, s->slot_id,
                                                                     static_cast<std::uint64_t>(rb.rm_from_pos));
                                if (!swa_ok ||
                                    !llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(rb.rm_from_pos), -1)) {
                                    invalidate_slot(*s);
                                    return false;
                                }
                            }
                            return true;
                        };

                        bool slot_finished = false;
                        for (std::size_t m = 0; m < s->accepted_tokens.size(); ++m) {
                            const Token tok = s->accepted_tokens[m];
                            const bool is_bonus = (m == s->step_n_accepted);
                            if (llama_vocab_is_eog(vocab, tok)) {
                                (void)rollback_slot_tail(committed_drafts);
                                if (m > 0) {
                                    Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m), step_ms);
                                    total_committed_step += m;
                                }
                                finish_slot(*s, StopReason::kEndOfGeneration);
                                slot_finished = true;
                                break;
                            }
                            bool keep_going = true;
                            if (Status st = emit_slot_token(*s, tok, s->accepted_lps[m], keep_going); !st.ok()) {
                                (void)rollback_slot_tail(committed_drafts);
                                s->result.status = std::move(st);
                                finish_slot(*s, StopReason::kError);
                                slot_finished = true;
                                break;
                            }
                            if (!is_bonus) {
                                if (s->slot_valid && s->slot_idx < slots.size()) {
                                    slots[s->slot_idx].tokens.push_back(tok);
                                }
                                s->committed_tokens.push_back(tok);
                                if (s->local_suffix_matcher != nullptr && s->local_matcher_synced) {
                                    s->local_suffix_matcher->CommitToken(tok);
                                }
                                ++new_pos;
                                ++committed_drafts;
                            }
                            if (!keep_going || s->IsCancelled()) {
                                (void)rollback_slot_tail(committed_drafts);
                                finish_slot(*s, StopReason::kCancelled);
                                slot_finished = true;
                                break;
                            }
                            if (s->result.generated_tokens >= req.max_new_tokens) {
                                if (is_bonus && s->concurrent_step_mode == SlotSpecMode::kNgramMod) {
                                    s->committed_tokens.push_back(tok);
                                }
                                if (is_bonus && s->local_suffix_matcher != nullptr && s->local_matcher_synced) {
                                    s->local_suffix_matcher->CommitToken(tok);
                                }
                                (void)rollback_slot_tail(committed_drafts);
                                Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(m + 1), step_ms);
                                total_committed_step += m + 1;
                                finish_slot(*s, StopReason::kMaxTokens);
                                slot_finished = true;
                                break;
                            }
                            if (is_bonus) {
                                s->cur_tok = tok;
                            }
                        }
                        if (slot_finished) continue;

                        if (!rollback_slot_tail(committed_drafts)) {
                            s->result.status = Status::Error(
                                ErrorCode::kDecodeFailed, "failed to trim rejected concurrent speculative KV tail");
                            finish_slot(*s, StopReason::kError);
                            continue;
                        }
                        if (s->concurrent_step_mode == SlotSpecMode::kNgramMod && ngram_mod != nullptr) {
                            s->ngram_indexed =
                                ngram_mod->IndexRange(s->committed_tokens, s->ngram_indexed, s->committed_tokens.size());
                        } else if (s->concurrent_step_mode == SlotSpecMode::kNgramShort &&
                                   s->local_ngram_matcher != nullptr) {
                            s->local_ngram_matcher->IndexRange(s->committed_tokens, s->ngram_indexed,
                                                               s->committed_tokens.size());
                            s->ngram_indexed = s->committed_tokens.size();
                        }
                        Emit(TelemetryKind::kTokenDecoded, static_cast<std::int32_t>(s->accepted_tokens.size()), step_ms);
                        total_committed_step += s->accepted_tokens.size();
                        s->pos = new_pos;
                        if (s->pos >= n_ctx) {
                            finish_slot(*s, StopReason::kContextFull);
                        }
                    }

                    (void)concurrent_cost_gate.ObserveSpecStep(step_ms, total_committed_step, total_drafted_step,
                                                               total_accepted_step, concurrent_spec_policy);

                    active.erase(std::remove_if(active.begin(), active.end(),
                                                [](const auto& sp) { return sp->task == nullptr; }),
                                 active.end());
                }
            }
        }

        if (ran_concurrent_spec) {
            continue;
        }

        // Ordinary Case B: True Continuous Batching + Chunked Prefill across
        // concurrent slots in lockstep (1 decode token per decoding slot +
        // chunked prefill <= n_ubatch packed into ONE llama_decode call!).
        // Suspend speculation cleanly WITHOUT permanently zeroing configured
        // capability so trailing requests can safely reactivate after drain.
        const SpecSuspendReason batch_suspend_reason = [&]() noexcept {
            if (pending_exclusive_count.load(std::memory_order_acquire) > 0) {
                return SpecSuspendReason::kExclusivePending;
            }
            if (active.size() == 1u && pending_count.load(std::memory_order_acquire) > 0) {
                return SpecSuspendReason::kPendingRequest;
            }
            if (concurrent_spec_policy.enabled && concurrent_cost_gate.IsRegressed()) {
                return SpecSuspendReason::kVerifyCostRegression;
            }
            return SpecSuspendReason::kMultiRequestBatch;
        }();
        for (auto& sp : active) {
            const bool had_spec_enabled =
                sp->mtp_enabled || sp->suffix_enabled || sp->ngram_mod_enabled || sp->ngram_short_enabled;
            suspend_slot_for_batch(*sp, batch_suspend_reason);
            if (had_spec_enabled) {
                sp->result.spec_transitioned_during_request = true;
                sp->result.effective_spec_method = "none";
                sp->result.effective_draft_limit = 0;
                sp->result.spec_fallback_reason =
                    (batch_suspend_reason == SpecSuspendReason::kVerifyCostRegression)
                        ? "disabled_by_policy"
                        : "concurrency_batch";
            }
        }

        if (prefix_coalescer.IsEnabled()) {
            const auto timed_out = prefix_coalescer.PollTimeouts(coalesce_now_ms());
            apply_coalesce_releases(timed_out);
        }

        const std::int64_t plan_now_us = NowUs();
        auto build_views = [&]() {
            views.clear();
            for (const auto& sp : active) {
                if (prefix_coalescer.IsFollowerHeld(sp->slot_idx)) {
                    continue;
                }
                BatchSlotPlanView view;
                view.slot_index = sp->slot_idx;
                view.seq_id = static_cast<std::int32_t>(sp->slot_id);
                view.phase = sp->phase;
                view.prompt_len = sp->task->req->prompt.size();
                view.prefill_offset = sp->prefill_offset;
                view.pos = sp->pos;
                view.cancelled = sp->IsCancelled();
                view.priority = sp->task->req->priority;
                view.target_ttft_ms = sp->task->req->target_ttft_ms;
                view.target_itl_ms = sp->task->req->target_itl_ms;
                view.enqueue_time_us = sp->enqueue_time_us;
                view.last_service_time_us = sp->last_service_time_us;
                view.kv_context_tokens = std::max<std::int64_t>(0, sp->pos);
                views.push_back(view);
            }
        };
        build_views();
        if (views.empty() && !active.empty() && prefix_coalescer.HeldFollowerCount() > 0) {
            // Defensive anti-deadlock guard (unreachable under coalescer invariants).
            (void)prefix_coalescer.ReleaseAllFollowers();
            SyncCoalescerStats();
            build_views();
        }

        // Adapt mixed decode+prefill chunk size from measured step cost and
        // active foreground ITL targets (protecting decode ITL while aging
        // queued prefills/background work), while letting pure-prefill waves
        // utilize the full `n_batch` window in one GPU pass.
        PlannedBatchStep plan =
            PlanLatencyAwareBatchStep(views, step_token_budget, n_batch, step_output_budget, rr_cursor, plan_now_us,
                                      step_cost_estimator, batch_planner_config);
        if (plan.items.empty()) {
            continue;
        }

        // When a producer has waiting followers and a material non-shared
        // suffix tail in this chunk, align the chunk to the earliest follower
        // publish boundary so followers can clone and join the very next step.
        if (prefix_coalescer.IsEnabled()) {
            const std::size_t min_split_tail = prefix_coalescer.config().min_coalesce_prefix_tokens;
            for (PlannedBatchItem& item : plan.items) {
                if (!item.is_prefill) continue;
                const auto boundary = prefix_coalescer.SuggestedProducerPublishBoundary(item.slot_index);
                if (!boundary.has_value()) continue;
                const std::size_t step_end = item.token_offset + static_cast<std::size_t>(item.token_count);
                if (item.token_offset < *boundary && step_end > *boundary &&
                    (step_end - *boundary) >= min_split_tail) {
                    const auto trimmed = static_cast<std::int32_t>(*boundary - item.token_offset);
                    plan.total_tokens -= (item.token_count - trimmed);
                    item.token_count = trimmed;
                    if (item.request_logits) {
                        item.request_logits = false;
                        plan.total_outputs -= 1;
                    }
                }
            }
        }
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
        auto decode_start = step_start;
        Status decode_err;
        int32_t rc = TryDecode(ctx, batch, &decode_err);

        // Under unified KV (`kv_unified=true`), idle prefix-cached slots share
        // the same n_ctx pool; evict LRU idle slots before retrying on rc==1.
        while (rc == 1 && evict_one_idle_slot()) {
            abort_requested.store(false, std::memory_order_relaxed);
            decode_start = Clock::now();
            rc = TryDecode(ctx, batch, &decode_err);
        }

        // If KV is still full after idle eviction and multiple active slots
        // share the pool, selectively preempt the lowest-priority victim slot
        // (cost-aware Recompute vs SpillToCheckpoint, respecting anti-starvation)
        // so higher-priority peers continue uninterrupted instead of all failing.
        while (rc == 1 && active.size() > 1u) {
            std::vector<PreemptionVictimView> victim_views;
            victim_views.reserve(active.size());
            for (const auto& sp : active) {
                if (!sp || sp->task == nullptr) continue;
                if (prefix_coalescer.IsFollowerHeld(sp->slot_idx)) continue;
                PreemptionVictimView vv;
                vv.slot_index = sp->slot_idx;
                vv.seq_id = static_cast<std::int32_t>(sp->slot_id);
                vv.priority_rank = sp->task->req->priority_rank;
                vv.kv_tokens = sp->pos > 0 ? static_cast<std::size_t>(sp->pos) : 0u;
                vv.admit_order = sp->admit_order;
                vv.requeue_count = sp->task->req->requeue_count;
                vv.cancelled = sp->IsCancelled();
                vv.suspended = false;
                vv.allow_spill = sp->task->req->allow_preempt_spill;
                victim_views.push_back(vv);
            }
            PhysicalPreemptionCostParams cost_params;
            if (radix_cache.enabled() && radix_cache.max_bytes() > radix_cache.used_bytes()) {
                cost_params.available_spill_bytes = radix_cache.max_bytes() - radix_cache.used_bytes();
            }
            const auto choice = SelectSelectivePreemptionVictim(victim_views, cost_params, /*allow_last_slot=*/false);
            if (!choice.has_value()) {
                break;
            }
            ActiveBatchSlot* victim_slot = find_active_by_slot(choice->victim_slot_index);
            if (victim_slot == nullptr || victim_slot->task == nullptr) {
                break;
            }
            const SuspendAction act = choice->policy == PhysicalPreemptionPolicy::kSpillToCheckpoint
                                          ? SuspendAction::kYieldSpill
                                          : SuspendAction::kYieldRecompute;
            suspend_slot(*victim_slot, act, kMaxSequenceStateBytes);
            active.erase(std::remove_if(active.begin(), active.end(),
                                        [](const auto& sp) { return sp->task == nullptr; }),
                         active.end());
            // Remove the victim from the planned batch and retry for surviving peers.
            PlannedBatchStep trimmed;
            for (const PlannedBatchItem& item : plan.items) {
                if (item.slot_index == choice->victim_slot_index) continue;
                trimmed.items.push_back(item);
                trimmed.total_tokens += item.token_count;
                if (item.request_logits) trimmed.total_outputs += 1;
                if (item.is_prefill) {
                    trimmed.prefill_slots += 1;
                } else {
                    trimmed.decode_slots += 1;
                }
            }
            if (trimmed.items.empty()) {
                rc = 0;
                break;
            }
            plan = std::move(trimmed);
            pack_plan(plan);
            abort_requested.store(false, std::memory_order_relaxed);
            decode_start = Clock::now();
            rc = TryDecode(ctx, batch, &decode_err);
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
            decode_start = Clock::now();
            rc = TryDecode(ctx, batch, &decode_err);
        }
        const double decode_call_ms = MsSince(decode_start);
        batch_abort_fn = nullptr;

        if (rc != 0) {
            const bool stopping = worker_stop.load(std::memory_order_acquire);
            for (const PlannedBatchItem& item : plan.items) {
                ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
                if (s == nullptr) continue;
                const bool swa_ok = CheckSwaRollback(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(s->pos));
                if (!swa_ok || !llama_memory_seq_rm(mem, s->slot_id, static_cast<llama_pos>(s->pos), -1)) {
                    invalidate_slot(*s);
                }
                if (rc == 2 && (s->IsCancelled() || stopping)) {
                    finish_slot(*s, StopReason::kCancelled);
                } else if (rc == 1) {
                    finish_slot(*s, StopReason::kContextFull);
                } else if (rc != 2) {
                    s->result.status = !decode_err.ok()
                                           ? decode_err
                                           : Status::Error(ErrorCode::kDecodeFailed,
                                                           "llama_decode failed (rc=" + std::to_string(rc) + ")");
                    finish_slot(*s, StopReason::kError);
                }
            }
            active.erase(std::remove_if(active.begin(), active.end(),
                                        [](const auto& sp) { return sp->task == nullptr; }),
                         active.end());
            continue;
        }

        {
            std::int32_t exec_decode = 0;
            std::int32_t exec_prefill = 0;
            std::int64_t exec_kv = 0;
            for (const PlannedBatchItem& item : plan.items) {
                if (item.is_prefill) {
                    exec_prefill += item.token_count;
                } else {
                    exec_decode += item.token_count;
                }
                exec_kv = detail::SaturatingAddI64(exec_kv, std::max<std::int64_t>(0, item.pos_start));
            }
            step_cost_estimator.Observe(StepCostObservation{exec_decode, exec_prefill, exec_kv, decode_call_ms});
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
        const std::int64_t serviced_us = NowUs();
        std::size_t plain_decode_committed = 0;
        for (std::size_t idx = 0; idx < plan.items.size(); ++idx) {
            const PlannedBatchItem& item = plan.items[idx];
            ActiveBatchSlot* s = find_active_by_slot(item.slot_index);
            if (s == nullptr || s->task == nullptr) continue;
            s->last_service_time_us = serviced_us;
            const GenerateRequest& req = *s->task->req;

            // Target KV advanced without MTP draft context sync -> mark drifted.
            mark_slot_mtp_drifted(*s);

            if (item.is_prefill) {
                if (s->slot_valid && s->slot_idx < slots.size()) {
                    auto& stoks = slots[s->slot_idx].tokens;
                    stoks.insert(stoks.end(),
                                 req.prompt.begin() + static_cast<std::ptrdiff_t>(item.token_offset),
                                 req.prompt.begin() +
                                     static_cast<std::ptrdiff_t>(item.token_offset + static_cast<std::size_t>(item.token_count)));
                }
                s->committed_tokens.insert(
                    s->committed_tokens.end(),
                    req.prompt.begin() + static_cast<std::ptrdiff_t>(item.token_offset),
                    req.prompt.begin() +
                        static_cast<std::ptrdiff_t>(item.token_offset + static_cast<std::size_t>(item.token_count)));
                s->prefill_offset += static_cast<std::size_t>(item.token_count);
                s->pos += item.token_count;
                AdvanceSwaSlot(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(s->pos));
                s->result.prompt_tokens += item.token_count;

                // Publish newly computed prefix KV to waiting followers BEFORE
                // this producer could finish (max_new_tokens==0 / EOG / error)
                // or release its slot.
                if (prefix_coalescer.IsEnabled() && s->slot_valid) {
                    const auto published = prefix_coalescer.OnProducerProgress(s->slot_idx, s->prefill_offset);
                    apply_coalesce_releases(published);
                }

                if (s->prefill_offset >= req.prompt.size()) {
                    s->prefill_done = true;
                    s->result.prefill_ms = MsSince(s->prefill_start);
                    if (s->result.prompt_tokens > 0 && s->result.prefill_ms > 0.0) {
                        radix_cache.ObservePrefill(static_cast<std::size_t>(s->result.prompt_tokens),
                                                   s->result.prefill_ms);
                    }
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
                    (void)poll_slot_safe_point(*s);
                } else {
                    (void)poll_slot_safe_point(*s);
                }
            } else {
                if (s->slot_valid && s->slot_idx < slots.size()) {
                    slots[s->slot_idx].tokens.push_back(s->cur_tok);
                }
                s->committed_tokens.push_back(s->cur_tok);
                s->pos += 1;
                AdvanceSwaSlot(s->slot_idx, mem, s->slot_id, static_cast<std::uint64_t>(s->pos));

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
                ++plain_decode_committed;
                if (s->result.generated_tokens >= req.max_new_tokens) {
                    finish_slot(*s, StopReason::kMaxTokens);
                    continue;
                }
                if (s->pos >= n_ctx) {
                    finish_slot(*s, StopReason::kContextFull);
                    continue;
                }
                s->cur_tok = next_tok;
                (void)poll_slot_safe_point(*s);
            }
        }

        if (plan.prefill_slots == 0u && plain_decode_committed >= 2u) {
            concurrent_cost_gate.ObservePlainStep(step_ms, plain_decode_committed);
            if (concurrent_cost_gate.IsRegressed()) {
                concurrent_cost_gate.NotePlainStepInCooldown();
            }
        }

        active.erase(std::remove_if(active.begin(), active.end(),
                                    [](const auto& sp) { return sp->task == nullptr; }),
                     active.end());
    }
    } catch (const std::exception& ex) {
        batch_abort_fn = nullptr;
        active_cancel = nullptr;
        const std::string err_msg = FormatBoundedExceptionMessage("batch_worker", &ex);
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
        prefix_coalescer.ResetActive();
        SyncCoalescerStats();
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
        prefix_coalescer.ResetActive();
        SyncCoalescerStats();
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
        pending_exclusive_count.store(0, std::memory_order_relaxed);
    }
    for (PendingRequest* task : leftover) {
        GenerateResult res;
        res.status = Status::Error(ErrorCode::kNotLoaded, "backend unloaded before request started");
        res.stop_reason = StopReason::kError;
        complete_task(task, std::move(res));
    }
}

}  // namespace sonder::backends::llamacpp
