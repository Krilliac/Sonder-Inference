// Pure helpers for the llama.cpp backend (no llama.cpp dependency).
#include "sonder/backends/llamacpp/llamacpp_backend.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace sonder::backends::llamacpp {

std::string_view ToString(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::kOk: return "ok";
        case ErrorCode::kInvalidArgument: return "invalid_argument";
        case ErrorCode::kFileNotFound: return "file_not_found";
        case ErrorCode::kLoadFailed: return "load_failed";
        case ErrorCode::kNotLoaded: return "not_loaded";
        case ErrorCode::kTokenizeFailed: return "tokenize_failed";
        case ErrorCode::kContextOverflow: return "context_overflow";
        case ErrorCode::kDecodeFailed: return "decode_failed";
    }
    return "unknown";
}

std::string_view ToString(StopReason reason) noexcept {
    switch (reason) {
        case StopReason::kNone: return "none";
        case StopReason::kEndOfGeneration: return "end_of_generation";
        case StopReason::kMaxTokens: return "max_tokens";
        case StopReason::kCancelled: return "cancelled";
        case StopReason::kContextFull: return "context_full";
        case StopReason::kError: return "error";
    }
    return "unknown";
}

std::string_view ToString(DeviceKind kind) noexcept {
    switch (kind) {
        case DeviceKind::kCpu: return "cpu";
        case DeviceKind::kGpu: return "gpu";
        case DeviceKind::kIntegratedGpu: return "igpu";
        case DeviceKind::kAccelerator: return "accel";
        case DeviceKind::kOther: return "other";
    }
    return "unknown";
}

Status Validate(const SamplingParams& params) {
    if (!std::isfinite(params.temperature)) {
        return Status::Error(ErrorCode::kInvalidArgument, "temperature must be finite");
    }
    if (!std::isfinite(params.top_p) || params.top_p <= 0.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "top_p must be finite and > 0");
    }
    if (!std::isfinite(params.min_p) || params.min_p > 1.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "min_p must be finite and <= 1");
    }
    if (!std::isfinite(params.repeat_penalty) || params.repeat_penalty <= 0.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "repeat_penalty must be finite and > 0");
    }
    if (params.repeat_last_n < -1) {
        return Status::Error(ErrorCode::kInvalidArgument, "repeat_last_n must be >= -1");
    }
    if (!std::isfinite(params.presence_penalty) || !std::isfinite(params.frequency_penalty)) {
        return Status::Error(ErrorCode::kInvalidArgument, "presence/frequency penalties must be finite");
    }
    if (!std::isfinite(params.typical_p) || params.typical_p <= 0.0F) {
        return Status::Error(ErrorCode::kInvalidArgument, "typical_p must be finite and > 0");
    }
    for (const auto& entry : params.logit_bias) {
        if (std::isnan(entry.second) || entry.second == std::numeric_limits<float>::infinity()) {
            return Status::Error(ErrorCode::kInvalidArgument, "logit_bias values must be finite or -inf");
        }
    }
    return Status::Ok();
}

namespace {

constexpr KvCacheType kAllKvCacheTypes[] = {
    KvCacheType::kF16,    KvCacheType::kF32,  KvCacheType::kBF16, KvCacheType::kQ8_0,
    KvCacheType::kQ5_1,   KvCacheType::kQ5_0, KvCacheType::kQ4_1, KvCacheType::kQ4_0,
    KvCacheType::kIQ4_NL, KvCacheType::kF8_E4M3, KvCacheType::kQ2_0, KvCacheType::kPQ2_0,
};

bool EqualsIgnoreCase(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<unsigned char>(a[i]);
        const auto cb = static_cast<unsigned char>(b[i]);
        const auto la = static_cast<unsigned char>(ca >= 'A' && ca <= 'Z' ? ca + ('a' - 'A') : ca);
        const auto lb = static_cast<unsigned char>(cb >= 'A' && cb <= 'Z' ? cb + ('a' - 'A') : cb);
        if (la != lb) return false;
    }
    return true;
}

std::uint64_t Mix64(std::uint64_t x) noexcept {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

std::size_t Hash3(Token a, Token b, Token c) noexcept {
    std::uint64_t h = Mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)));
    h = Mix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(b)));
    h = Mix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(c)));
    return static_cast<std::size_t>(h);
}

std::size_t Hash4(Token a, Token b, Token c, Token d) noexcept {
    std::uint64_t h = Mix64(static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)));
    h = Mix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(b)));
    h = Mix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(c)));
    h = Mix64(h ^ static_cast<std::uint64_t>(static_cast<std::uint32_t>(d)));
    return static_cast<std::size_t>(h);
}

}  // namespace

std::string_view ToString(KvCacheType type) noexcept {
    switch (type) {
        case KvCacheType::kF16: return "f16";
        case KvCacheType::kF32: return "f32";
        case KvCacheType::kBF16: return "bf16";
        case KvCacheType::kQ8_0: return "q8_0";
        case KvCacheType::kQ5_1: return "q5_1";
        case KvCacheType::kQ5_0: return "q5_0";
        case KvCacheType::kQ4_1: return "q4_1";
        case KvCacheType::kQ4_0: return "q4_0";
        case KvCacheType::kIQ4_NL: return "iq4_nl";
        case KvCacheType::kF8_E4M3: return "f8_e4m3";
        case KvCacheType::kQ2_0: return "q2_0";
        case KvCacheType::kPQ2_0: return "pq2_0";
    }
    return "unknown";
}

std::string_view ToString(FlashAttention mode) noexcept {
    switch (mode) {
        case FlashAttention::kAuto: return "auto";
        case FlashAttention::kDisabled: return "off";
        case FlashAttention::kEnabled: return "on";
    }
    return "unknown";
}

std::string_view ToString(SplitMode mode) noexcept {
    switch (mode) {
        case SplitMode::kNone: return "none";
        case SplitMode::kLayer: return "layer";
        case SplitMode::kRow: return "row";
    }
    return "unknown";
}

std::string_view ToString(SpeculativeType type) noexcept {
    switch (type) {
        case SpeculativeType::kNone: return "none";
        case SpeculativeType::kAuto: return "auto";
        case SpeculativeType::kNgram: return "ngram";
        case SpeculativeType::kNgramMod: return "ngram-mod";
        case SpeculativeType::kDraftMtp: return "draft-mtp";
    }
    return "unknown";
}

bool ParseKvCacheType(std::string_view name, KvCacheType& out) noexcept {
    for (const KvCacheType t : kAllKvCacheTypes) {
        if (EqualsIgnoreCase(name, ToString(t))) {
            out = t;
            return true;
        }
    }
    return false;
}

bool ParseFlashAttention(std::string_view name, FlashAttention& out) noexcept {
    if (EqualsIgnoreCase(name, "auto")) {
        out = FlashAttention::kAuto;
    } else if (EqualsIgnoreCase(name, "on") || EqualsIgnoreCase(name, "enabled")) {
        out = FlashAttention::kEnabled;
    } else if (EqualsIgnoreCase(name, "off") || EqualsIgnoreCase(name, "disabled")) {
        out = FlashAttention::kDisabled;
    } else {
        return false;
    }
    return true;
}

bool ParseSplitMode(std::string_view name, SplitMode& out) noexcept {
    if (EqualsIgnoreCase(name, "none")) {
        out = SplitMode::kNone;
    } else if (EqualsIgnoreCase(name, "layer")) {
        out = SplitMode::kLayer;
    } else if (EqualsIgnoreCase(name, "row")) {
        out = SplitMode::kRow;
    } else {
        return false;
    }
    return true;
}

bool ParseSpeculativeType(std::string_view name, SpeculativeType& out) noexcept {
    if (EqualsIgnoreCase(name, "none")) {
        out = SpeculativeType::kNone;
    } else if (EqualsIgnoreCase(name, "auto")) {
        out = SpeculativeType::kAuto;
    } else if (EqualsIgnoreCase(name, "ngram")) {
        out = SpeculativeType::kNgram;
    } else if (EqualsIgnoreCase(name, "ngram-mod")) {
        out = SpeculativeType::kNgramMod;
    } else if (EqualsIgnoreCase(name, "draft-mtp")) {
        out = SpeculativeType::kDraftMtp;
    } else {
        return false;
    }
    return true;
}

bool IsQuantized(KvCacheType type) noexcept {
    return type != KvCacheType::kF16 && type != KvCacheType::kF32 && type != KvCacheType::kBF16;
}

bool IsKvCacheTypeSupported(KvCacheType type) noexcept {
    switch (type) {
        case KvCacheType::kF16:
        case KvCacheType::kF32:
        case KvCacheType::kBF16:
        case KvCacheType::kQ8_0:
        case KvCacheType::kQ5_1:
        case KvCacheType::kQ5_0:
        case KvCacheType::kQ4_1:
        case KvCacheType::kQ4_0:
        case KvCacheType::kIQ4_NL: return true;
        case KvCacheType::kF8_E4M3:
#if defined(SONDER_LLAMACPP_HAS_GGML_F8_E4M3)
            return true;
#else
            return false;
#endif
        case KvCacheType::kQ2_0:
#if defined(SONDER_LLAMACPP_HAS_GGML_Q2_0)
            return true;
#else
            return false;
#endif
        case KvCacheType::kPQ2_0:
#if defined(SONDER_LLAMACPP_HAS_GGML_PQ2_0)
            return true;
#else
            return false;
#endif
    }
    return false;
}

Status ValidatePlacementOptions(const LoadOptions& options) {
    if (options.main_gpu < 0) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "main_gpu (" + std::to_string(options.main_gpu) + ") must be >= 0");
    }
    if (ToString(options.split_mode) == "unknown") {
        return Status::Error(ErrorCode::kInvalidArgument, "unknown split mode");
    }
    if (options.devices.size() > kMaxTensorSplitDevices) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "too many devices (" + std::to_string(options.devices.size()) + ", maximum is " +
                                 std::to_string(kMaxTensorSplitDevices) + ")");
    }
    for (const std::string& dev : options.devices) {
        if (dev.empty()) {
            return Status::Error(ErrorCode::kInvalidArgument, "device name in devices list is empty");
        }
        if (dev.find('\0') != std::string::npos) {
            return Status::Error(ErrorCode::kInvalidArgument, "device name in devices list contains NUL");
        }
    }
    if (!options.tensor_split.empty()) {
        if (options.tensor_split.size() > kMaxTensorSplitDevices) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "tensor_split has " + std::to_string(options.tensor_split.size()) +
                                     " entries (maximum is " + std::to_string(kMaxTensorSplitDevices) + ")");
        }
        bool any_positive = false;
        for (const float weight : options.tensor_split) {
            if (!std::isfinite(weight) || weight < 0.0F) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "tensor_split values must be finite and >= 0");
            }
            if (weight > 0.0F) any_positive = true;
        }
        if (!any_positive) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "tensor_split must contain at least one positive value");
        }
    }
    return Status::Ok();
}

Status ValidateContextOptions(const LoadOptions& options) {
    if (options.vocab_only) return Status::Ok();
    if (ToString(options.type_k) == "unknown" || ToString(options.type_v) == "unknown") {
        return Status::Error(ErrorCode::kInvalidArgument, "unknown KV cache type");
    }
    if (!IsKvCacheTypeSupported(options.type_k)) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "K cache type '" + std::string(ToString(options.type_k)) +
                                 "' is not supported by this llama.cpp build");
    }
    if (!IsKvCacheTypeSupported(options.type_v)) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "V cache type '" + std::string(ToString(options.type_v)) +
                                 "' is not supported by this llama.cpp build");
    }
    if (ToString(options.flash_attn) == "unknown") {
        return Status::Error(ErrorCode::kInvalidArgument, "unknown flash attention mode");
    }
    if (options.n_slots < 1 || options.n_slots > kMaxSlots) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "n_slots (" + std::to_string(options.n_slots) + ") must be in [1, " +
                                 std::to_string(kMaxSlots) + "]");
    }
    SpeculativeType spec = SpeculativeType::kNone;
    if (!ParseSpeculativeType(options.spec_type, spec)) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "unknown spec_type '" + options.spec_type +
                                 "' (expected none, auto, ngram, ngram-mod or draft-mtp)");
    }
    if (options.spec_draft_max < 0 || options.spec_draft_max > kMaxSpecDraftTokens) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "spec_draft_max (" + std::to_string(options.spec_draft_max) + ") must be in [0, " +
                                 std::to_string(kMaxSpecDraftTokens) + "]");
    }
    // n_batch == 0 keeps its own, older error in Load().
    if (options.n_batch > 0 && options.n_ubatch > options.n_batch) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "n_ubatch (" + std::to_string(options.n_ubatch) + ") must be <= n_batch (" +
                                 std::to_string(options.n_batch) + ")");
    }
    // llama.cpp refuses a quantized V cache without Flash Attention
    // (llama_init_from_model: "quantized V cache requires flash_attn").
    if (IsQuantized(options.type_v) && options.flash_attn == FlashAttention::kDisabled) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "V cache type " + std::string(ToString(options.type_v)) +
                                 " is quantized and requires flash attention (use on or auto)");
    }
    return Status::Ok();
}

double GenerateResult::decode_tokens_per_second() const noexcept {
    return decode_ms > 0.0 ? generated_tokens * 1000.0 / decode_ms : 0.0;
}

double GenerateResult::prefill_tokens_per_second() const noexcept {
    return prefill_ms > 0.0 ? prompt_tokens * 1000.0 / prefill_ms : 0.0;
}

namespace {

// Expected total length of a UTF-8 sequence from its lead byte; 0 if the byte
// is a continuation byte or an invalid lead.
std::size_t SequenceLength(unsigned char lead) noexcept {
    if (lead < 0x80U) return 1;
    if ((lead & 0xE0U) == 0xC0U) return lead >= 0xC2U ? 2 : 0;
    if ((lead & 0xF0U) == 0xE0U) return 3;
    if ((lead & 0xF8U) == 0xF0U) return lead <= 0xF4U ? 4 : 0;
    return 0;
}

}  // namespace

std::size_t CompleteUtf8Prefix(std::string_view s) noexcept {
    // Only the last (up to) 3 bytes can belong to an incomplete sequence.
    const std::size_t n = s.size();
    const std::size_t lookback = n < 4 ? n : 4;
    for (std::size_t back = 1; back <= lookback; ++back) {
        const auto byte = static_cast<unsigned char>(s[n - back]);
        if ((byte & 0xC0U) == 0x80U) continue;  // continuation byte, keep scanning
        const std::size_t need = SequenceLength(byte);
        if (need == 0) return n;  // invalid lead: pass bytes through unchanged
        return need > back ? n - back : n;
    }
    return n;  // only continuation bytes in lookback: malformed, pass through
}

std::string Utf8StreamBuffer::Push(std::string_view bytes) {
    pending_.append(bytes.data(), bytes.size());
    const std::size_t keep = CompleteUtf8Prefix(pending_);
    std::string out = pending_.substr(0, keep);
    pending_.erase(0, keep);
    return out;
}

std::string Utf8StreamBuffer::Flush() {
    std::string out;
    out.swap(pending_);
    return out;
}

std::size_t LongestCommonPrefix(std::span<const Token> a, std::span<const Token> b) noexcept {
    const std::size_t limit = std::min(a.size(), b.size());
    std::size_t n = 0;
    while (n < limit && a[n] == b[n]) ++n;
    return n;
}

SlotSelection SelectPrefixSlot(std::span<const PrefixSlotMeta> slots, std::span<const Token> prompt,
                               std::string_view session_key) noexcept {
    if (slots.empty()) return {};
    const std::size_t max_reuse = prompt.size() > 1 ? prompt.size() - 1 : 0;

    // 1. Exact non-empty session_key match takes precedence.
    if (!session_key.empty()) {
        for (std::size_t i = 0; i < slots.size(); ++i) {
            if (slots[i].session_key == session_key) {
                const std::size_t lcp = std::min(LongestCommonPrefix(slots[i].tokens, prompt), max_reuse);
                return {i, lcp};
            }
        }
    }

    // 2. Longest common prefix (> 0), tie-broken by most recently used.
    bool found_lcp = false;
    std::size_t best_idx = 0;
    std::size_t best_lcp = 0;
    std::uint64_t best_used = 0;
    for (std::size_t i = 0; i < slots.size(); ++i) {
        const std::size_t lcp = std::min(LongestCommonPrefix(slots[i].tokens, prompt), max_reuse);
        if (lcp > best_lcp || (lcp > 0 && lcp == best_lcp && slots[i].last_used > best_used)) {
            found_lcp = true;
            best_idx = i;
            best_lcp = lcp;
            best_used = slots[i].last_used;
        }
    }
    if (found_lcp) return {best_idx, best_lcp};

    // 3. Fallback to least recently used slot.
    std::size_t lru_idx = 0;
    std::uint64_t min_used = slots[0].last_used;
    for (std::size_t i = 1; i < slots.size(); ++i) {
        if (slots[i].last_used < min_used) {
            min_used = slots[i].last_used;
            lru_idx = i;
        }
    }
    return {lru_idx, 0};
}

NgramMatcher::NgramMatcher(bool exact_scan_fallback) : exact_scan_fallback_(exact_scan_fallback) { Reset(); }

void NgramMatcher::Reset() {
    pos4_.assign(kTableSize, -1);
    pos3_.assign(kTableSize, -1);
}

void NgramMatcher::IndexRange(std::span<const Token> history, std::size_t from_index, std::size_t to_index) {
    const std::size_t end = std::min(to_index, history.size());
    for (std::size_t i = from_index; i < end; ++i) {
        if (i >= 2) {
            const std::size_t h3 = Hash3(history[i - 2], history[i - 1], history[i]) & kTableMask;
            pos3_[h3] = static_cast<std::int32_t>(i);
        }
        if (i >= 3) {
            const std::size_t h4 = Hash4(history[i - 3], history[i - 2], history[i - 1], history[i]) & kTableMask;
            pos4_[h4] = static_cast<std::int32_t>(i);
        }
    }
}

void NgramMatcher::Draft(std::span<const Token> history, std::size_t max_draft, std::vector<Token>& out) const {
    out.clear();
    if (max_draft == 0 || history.size() < 3) return;

    const std::size_t n = history.size();
    std::int32_t match_pos = -1;

    // 1. Prefer 4-gram O(1) hash lookup.
    if (n >= 4) {
        const Token t0 = history[n - 4];
        const Token t1 = history[n - 3];
        const Token t2 = history[n - 2];
        const Token t3 = history[n - 1];
        const std::int32_t cand = pos4_[Hash4(t0, t1, t2, t3) & kTableMask];
        if (cand >= 3 && static_cast<std::size_t>(cand) + 1 < n) {
            const auto p = static_cast<std::size_t>(cand);
            if (history[p - 3] == t0 && history[p - 2] == t1 && history[p - 1] == t2 && history[p] == t3) {
                match_pos = cand;
            }
        }
    }

    // 2. Fallback to 3-gram O(1) hash lookup.
    if (match_pos < 0) {
        const Token t1 = history[n - 3];
        const Token t2 = history[n - 2];
        const Token t3 = history[n - 1];
        const std::int32_t cand = pos3_[Hash3(t1, t2, t3) & kTableMask];
        if (cand >= 2 && static_cast<std::size_t>(cand) + 1 < n) {
            const auto p = static_cast<std::size_t>(cand);
            if (history[p - 2] == t1 && history[p - 1] == t2 && history[p] == t3) {
                match_pos = cand;
            }
        }
    }

    // 3. Optional bounded reverse scan on hash collision (used by "ngram").
    if (match_pos < 0 && exact_scan_fallback_) {
        const std::size_t scan_floor = n > kFallbackScanLimit + 1 ? n - 1 - kFallbackScanLimit : 0;
        if (n >= 4) {
            const Token t0 = history[n - 4];
            const Token t1 = history[n - 3];
            const Token t2 = history[n - 2];
            const Token t3 = history[n - 1];
            for (std::size_t p = n - 2; p >= std::max<std::size_t>(3, scan_floor); --p) {
                if (history[p] == t3 && history[p - 1] == t2 && history[p - 2] == t1 && history[p - 3] == t0) {
                    match_pos = static_cast<std::int32_t>(p);
                    break;
                }
                if (p == 3) break;
            }
        }
        if (match_pos < 0) {
            const Token t1 = history[n - 3];
            const Token t2 = history[n - 2];
            const Token t3 = history[n - 1];
            for (std::size_t p = n - 2; p >= std::max<std::size_t>(2, scan_floor); --p) {
                if (history[p] == t3 && history[p - 1] == t2 && history[p - 2] == t1) {
                    match_pos = static_cast<std::int32_t>(p);
                    break;
                }
                if (p == 2) break;
            }
        }
    }

    if (match_pos < 0) return;

    const std::size_t start = static_cast<std::size_t>(match_pos) + 1;
    const std::size_t avail = n - start;
    const std::size_t count = std::min(max_draft, avail);
    out.reserve(count);
    for (std::size_t k = 0; k < count; ++k) {
        out.push_back(history[start + k]);
    }
}

namespace {

std::size_t RoundUpPow2(std::size_t v) noexcept {
    if (v <= 1024) return 1024;
    std::size_t p = 1024;
    while (p < v && p <= (std::size_t{1} << 26)) {
        p <<= 1;
    }
    return p;
}

}  // namespace

NgramModTable::NgramModTable(std::size_t n_match, std::size_t table_size, std::size_t n_min, std::size_t n_max)
    : n_match_(std::max<std::size_t>(1, n_match)) {
    const std::size_t cap = RoundUpPow2(table_size);
    table_mask_ = cap - 1;
    entries_.assign(cap, kEmpty);
    Configure(n_min, n_max);
}

void NgramModTable::Configure(std::size_t n_min, std::size_t n_max) {
    n_max_ = std::clamp<std::size_t>(n_max, 1, static_cast<std::size_t>(kMaxSpecDraftTokens));
    n_min_ = std::min<std::size_t>(n_min, n_max_);
    n_cur_ = n_max_;
}

void NgramModTable::Reset() {
    std::fill(entries_.begin(), entries_.end(), kEmpty);
    used_ = 0;
    low_streak_ = 0;
}

std::size_t NgramModTable::IndexOf(const Token* tokens) const noexcept {
    std::uint64_t h = 0;
    for (std::size_t i = 0; i < n_match_; ++i) {
        h = h * 6364136223846793005ULL + static_cast<std::uint64_t>(static_cast<std::uint32_t>(tokens[i]));
    }
    return static_cast<std::size_t>(h) & table_mask_;
}

void NgramModTable::Add(const Token* tokens) noexcept {
    const std::size_t i = IndexOf(tokens);
    if (entries_[i] == kEmpty) {
        ++used_;
    }
    entries_[i] = tokens[n_match_];
}

Token NgramModTable::Get(const Token* tokens) const noexcept {
    return entries_[IndexOf(tokens)];
}

std::size_t NgramModTable::IndexRange(std::span<const Token> history, std::size_t from_index, std::size_t to_index) {
    if (entries_.empty() || history.size() <= n_match_) return from_index;
    const std::size_t max_start = history.size() - n_match_;
    const std::size_t end = std::min(to_index, max_start);
    for (std::size_t i = from_index; i < end; ++i) {
        Add(history.data() + i);
    }
    return std::max(from_index, end);
}

void NgramModTable::BeginRequest(std::span<const Token> prompt) {
    n_cur_ = n_max_;
    low_streak_ = 0;
    if (entries_.empty()) return;
    // Reset only if occupancy exceeds 25% (preserves cross-request n-grams across turns).
    if (used_ * 4 > entries_.size()) {
        Reset();
    }
    (void)IndexRange(prompt, 0, prompt.size());
}

void NgramModTable::Draft(std::span<const Token> history, Token id_last, std::size_t max_draft,
                          std::vector<Token>& out) const {
    out.clear();
    if (entries_.empty() || max_draft == 0) return;
    if (history.size() + 1 < n_match_) return;

    const std::size_t limit = std::min(max_draft, n_max_);
    const std::size_t min_req = std::min(n_min_, limit);
    scratch_.resize(n_match_ + limit);
    const std::size_t prefix_keep = n_match_ - 1;
    const std::size_t base = history.size() - prefix_keep;
    for (std::size_t i = 0; i < prefix_keep; ++i) {
        scratch_[i] = history[base + i];
    }
    scratch_[prefix_keep] = id_last;

    std::size_t chained = 0;
    for (std::size_t i = 0; i < limit; ++i) {
        const Token tok = Get(scratch_.data() + i);
        if (tok == kEmpty) {
            if (i < min_req) {
                return;
            }
            break;
        }
        scratch_[n_match_ + i] = tok;
        ++chained;
    }
    if (chained < min_req) {
        return;
    }

    const std::size_t emit = (n_cur_ > 0 && chained > n_cur_) ? n_cur_ : chained;
    out.assign(scratch_.begin() + static_cast<std::ptrdiff_t>(n_match_),
               scratch_.begin() + static_cast<std::ptrdiff_t>(n_match_ + emit));
}

void NgramModTable::Observe(std::size_t drafted, std::size_t accepted) {
    if (drafted == 0) return;
    const std::size_t floor_len = std::min<std::size_t>(32, n_max_);
    if (accepted >= drafted) {
        n_cur_ = std::min<std::size_t>(n_cur_ * 2, n_max_);
    } else {
        n_cur_ = std::max<std::size_t>(floor_len, n_cur_ / 2);
    }

    if (accepted * 4 < drafted) {
        ++low_streak_;
        if (low_streak_ >= 5) {
            Reset();
        }
    } else {
        low_streak_ = 0;
    }
}

AdaptiveDraftController::AdaptiveDraftController(std::int32_t max_draft) { Reset(max_draft); }

void AdaptiveDraftController::Reset(std::int32_t max_draft) {
    max_draft_ = std::clamp<std::int32_t>(max_draft, 0, kMaxSpecDraftTokens);
    current_draft_ = max_draft_;
    cooldown_remaining_ = 0;
    cooldown_period_ = 8;
    steps_observed_ = 0;
    total_drafted_ = 0;
    acceptance_ema_ = 0.5;
}

std::int32_t AdaptiveDraftController::NextDraftLength() const noexcept {
    if (max_draft_ <= 0 || cooldown_remaining_ > 0) return 0;
    return std::clamp<std::int32_t>(current_draft_, 1, max_draft_);
}

void AdaptiveDraftController::NoteSkippedStep() noexcept {
    if (cooldown_remaining_ > 0) {
        --cooldown_remaining_;
        if (cooldown_remaining_ == 0) {
            current_draft_ = std::min<std::int32_t>(max_draft_, 2);
        }
    }
}

void AdaptiveDraftController::Observe(std::int32_t drafted, std::int32_t accepted) noexcept {
    if (drafted <= 0 || max_draft_ <= 0) return;
    accepted = std::clamp<std::int32_t>(accepted, 0, drafted);
    const double rate = static_cast<double>(accepted) / static_cast<double>(drafted);
    constexpr double kAlpha = 0.25;
    acceptance_ema_ = (steps_observed_ == 0) ? rate : (1.0 - kAlpha) * acceptance_ema_ + kAlpha * rate;
    ++steps_observed_;
    total_drafted_ += drafted;

    if (accepted == drafted && current_draft_ < max_draft_) {
        current_draft_ = std::min<std::int32_t>(max_draft_, current_draft_ + 2);
        cooldown_period_ = 8;
        acceptance_ema_ = std::max(acceptance_ema_, 0.50);
    } else if (acceptance_ema_ >= 0.50) {
        current_draft_ = std::min<std::int32_t>(max_draft_, current_draft_ + 1);
        cooldown_period_ = 8;
    } else if (accepted > 0 && current_draft_ <= 2 && rate >= 0.50) {
        // Probe after pause succeeded: recover immediately.
        cooldown_period_ = 8;
        acceptance_ema_ = std::max(acceptance_ema_, 0.35);
        current_draft_ = std::min<std::int32_t>(max_draft_, current_draft_ + 2);
    } else if (steps_observed_ >= 3 && total_drafted_ >= 8 && acceptance_ema_ < 0.20) {
        cooldown_remaining_ = cooldown_period_;
        cooldown_period_ = std::min<std::int32_t>(cooldown_period_ * 2, 32);
        current_draft_ = std::min<std::int32_t>(max_draft_, 2);
    } else if (acceptance_ema_ < 0.35 && current_draft_ > 2) {
        current_draft_ = std::max<std::int32_t>(2, current_draft_ / 2);
    }
}

}  // namespace sonder::backends::llamacpp
