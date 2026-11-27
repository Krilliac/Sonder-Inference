// Sonder Inference: KV-cache dtype policy implementation.
#include "kv_policy.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <string_view>

#include "ggml.h"
#include "gguf.h"

namespace sonder::backends::llamacpp {

namespace {

std::string LowerAsciiCopy(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char ch : s) {
        if (ch >= 'A' && ch <= 'Z') {
            out.push_back(static_cast<char>(ch + ('a' - 'A')));
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

bool StartsWithIgnoreCase(std::string_view s, std::string_view prefix) noexcept {
    if (s.size() < prefix.size()) return false;
    for (std::size_t i = 0; i < prefix.size(); ++i) {
        const auto a = static_cast<unsigned char>(s[i]);
        const auto b = static_cast<unsigned char>(prefix[i]);
        const auto la = static_cast<unsigned char>(a >= 'A' && a <= 'Z' ? a + ('a' - 'A') : a);
        const auto lb = static_cast<unsigned char>(b >= 'A' && b <= 'Z' ? b + ('a' - 'A') : b);
        if (la != lb) return false;
    }
    return true;
}

bool EqualsIgnoreCaseAscii(std::string_view a, std::string_view b) noexcept {
    return a.size() == b.size() && StartsWithIgnoreCase(a, b);
}

// Architectures in llama.cpp v1.0.0 whose create_memory() creates no attention
// KV cache (encoder-only, diffusion, or pure recurrent; src/llama-model.cpp:2938-3000).
bool IsNonAttentionKvArchitecture(std::string_view arch_lower) noexcept {
    static constexpr std::string_view kNoAttnKvArchs[] = {
        "bert",
        "jina-bert-v2",
        "jina-bert-v3",
        "nomic-bert",
        "nomic-bert-moe",
        "neo-bert",
        "eurobert",
        "wavtokenizer-dec",
        "modern-bert",
        "gemma-embedding",
        "gemma-embedding2",
        "dream",
        "llada",
        "llada-moe",
        "rnd1",
        "clef",
        "mamba",
        "mamba2",
        "rwkv6",
        "rwkv6qwen2",
        "rwkv7",
        "arwkv7",
        "clip",
    };
    for (const std::string_view candidate : kNoAttnKvArchs) {
        if (arch_lower == candidate) return true;
    }
    return false;
}

bool IsValidUtf8NoNul(std::string_view s) noexcept {
    std::size_t i = 0;
    while (i < s.size()) {
        const auto b0 = static_cast<unsigned char>(s[i]);
        if (b0 == 0u) return false;
        if (b0 < 0x80u) {
            ++i;
            continue;
        }
        std::size_t len = 0;
        std::uint32_t cp = 0;
        std::uint32_t min_cp = 0;
        if ((b0 & 0xE0u) == 0xC0u) {
            len = 2;
            cp = static_cast<std::uint32_t>(b0 & 0x1Fu);
            min_cp = 0x80u;
        } else if ((b0 & 0xF0u) == 0xE0u) {
            len = 3;
            cp = static_cast<std::uint32_t>(b0 & 0x0Fu);
            min_cp = 0x800u;
        } else if ((b0 & 0xF8u) == 0xF0u) {
            len = 4;
            cp = static_cast<std::uint32_t>(b0 & 0x07u);
            min_cp = 0x10000u;
        } else {
            return false;
        }
        if (i + len > s.size()) return false;
        for (std::size_t j = 1; j < len; ++j) {
            const auto bx = static_cast<unsigned char>(s[i + j]);
            if ((bx & 0xC0u) != 0x80u) return false;
            cp = (cp << 6u) | static_cast<std::uint32_t>(bx & 0x3Fu);
        }
        if (cp < min_cp || cp > 0x10FFFFu || (cp >= 0xD800u && cp <= 0xDFFFu)) {
            return false;
        }
        i += len;
    }
    return true;
}

bool IsIntegerGgufType(gguf_type t) noexcept {
    switch (t) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
            return true;
        default:
            return false;
    }
}

KvDeviceKind CombineDeviceKinds(const std::vector<KvDeviceKind>& kinds) noexcept {
    if (kinds.empty()) return KvDeviceKind::kCpu;
    bool all_cuda = true;
    bool any_cpu = false;
    bool any_vulkan = false;
    for (const KvDeviceKind k : kinds) {
        if (k != KvDeviceKind::kCuda) all_cuda = false;
        if (k == KvDeviceKind::kCpu) any_cpu = true;
        if (k == KvDeviceKind::kVulkan) any_vulkan = true;
    }
    if (all_cuda) return KvDeviceKind::kCuda;
    if (any_cpu) return KvDeviceKind::kCpu;
    if (any_vulkan) return KvDeviceKind::kVulkan;
    return KvDeviceKind::kOther;
}

struct GgufContextGuard {
    gguf_context* ctx = nullptr;
    ~GgufContextGuard() {
        if (ctx != nullptr) gguf_free(ctx);
    }
};

std::optional<std::uint32_t> ReadGgufScalarU32(const gguf_context* ctx, const std::string& key,
                                               bool& invalid_type_or_range) noexcept {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return std::nullopt;

    switch (gguf_get_kv_type(ctx, id)) {
        case GGUF_TYPE_UINT8:
            return static_cast<std::uint32_t>(gguf_get_val_u8(ctx, id));
        case GGUF_TYPE_UINT16:
            return static_cast<std::uint32_t>(gguf_get_val_u16(ctx, id));
        case GGUF_TYPE_UINT32:
            return gguf_get_val_u32(ctx, id);
        case GGUF_TYPE_UINT64: {
            const std::uint64_t v = gguf_get_val_u64(ctx, id);
            if (v > std::numeric_limits<std::uint32_t>::max()) {
                invalid_type_or_range = true;
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(v);
        }
        case GGUF_TYPE_INT8: {
            const std::int8_t v = gguf_get_val_i8(ctx, id);
            if (v < 0) {
                invalid_type_or_range = true;
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(v);
        }
        case GGUF_TYPE_INT16: {
            const std::int16_t v = gguf_get_val_i16(ctx, id);
            if (v < 0) {
                invalid_type_or_range = true;
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(v);
        }
        case GGUF_TYPE_INT32: {
            const std::int32_t v = gguf_get_val_i32(ctx, id);
            if (v < 0) {
                invalid_type_or_range = true;
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(v);
        }
        case GGUF_TYPE_INT64: {
            const std::int64_t v = gguf_get_val_i64(ctx, id);
            if (v < 0 || v > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
                invalid_type_or_range = true;
                return std::nullopt;
            }
            return static_cast<std::uint32_t>(v);
        }
        default:
            invalid_type_or_range = true;
            return std::nullopt;
    }
}

// Reads <arch>.attention.head_count, which GGUF models store as either a scalar
// integer or a per-layer integer array (where recurrent layers may carry 0).
void ReadGgufHeadCount(const gguf_context* ctx, const std::string& key, KvModelRotationFacts& facts,
                       bool& invalid_type_or_range) noexcept {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) return;

    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        facts.head_count = ReadGgufScalarU32(ctx, key, invalid_type_or_range);
        if (facts.head_count.has_value() && *facts.head_count == 0u) {
            facts.head_count_layer0_zero = true;
        }
        return;
    }

    const gguf_type elem_type = gguf_get_arr_type(ctx, id);
    if (!IsIntegerGgufType(elem_type)) {
        invalid_type_or_range = true;
        return;
    }
    const std::size_t n = gguf_get_arr_n(ctx, id);
    if (n == 0) {
        invalid_type_or_range = true;
        return;
    }
    const void* raw = gguf_get_arr_data(ctx, id);
    if (raw == nullptr) {
        invalid_type_or_range = true;
        return;
    }

    std::optional<std::uint32_t> uniform_positive;
    bool saw_zero = false;
    for (std::size_t i = 0; i < n; ++i) {
        std::uint64_t val = 0;
        if (elem_type == GGUF_TYPE_UINT8) {
            std::uint8_t v = 0;
            std::memcpy(&v, static_cast<const char*>(raw) + i * sizeof(v), sizeof(v));
            val = v;
        } else if (elem_type == GGUF_TYPE_INT8) {
            std::int8_t v = 0;
            std::memcpy(&v, static_cast<const char*>(raw) + i * sizeof(v), sizeof(v));
            if (v < 0) {
                invalid_type_or_range = true;
                return;
            }
            val = static_cast<std::uint64_t>(v);
        } else if (elem_type == GGUF_TYPE_UINT16) {
            std::uint16_t v = 0;
            std::memcpy(&v, static_cast<const char*>(raw) + i * sizeof(v), sizeof(v));
            val = v;
        } else if (elem_type == GGUF_TYPE_INT16) {
            std::int16_t v = 0;
            std::memcpy(&v, static_cast<const char*>(raw) + i * sizeof(v), sizeof(v));
            if (v < 0) {
                invalid_type_or_range = true;
                return;
            }
            val = static_cast<std::uint64_t>(v);
        } else if (elem_type == GGUF_TYPE_UINT32) {
            std::uint32_t v = 0;
            std::memcpy(&v, static_cast<const char*>(raw) + i * sizeof(v), sizeof(v));
            val = v;
        } else if (elem_type == GGUF_TYPE_INT32) {
            std::int32_t v = 0;
            std::memcpy(&v, static_cast<const char*>(raw) + i * sizeof(v), sizeof(v));
            if (v < 0) {
                invalid_type_or_range = true;
                return;
            }
            val = static_cast<std::uint64_t>(v);
        } else if (elem_type == GGUF_TYPE_UINT64) {
            std::memcpy(&val, static_cast<const char*>(raw) + i * sizeof(val), sizeof(val));
            if (val > std::numeric_limits<std::uint32_t>::max()) {
                invalid_type_or_range = true;
                return;
            }
        } else if (elem_type == GGUF_TYPE_INT64) {
            std::int64_t v = 0;
            std::memcpy(&v, static_cast<const char*>(raw) + i * sizeof(v), sizeof(v));
            if (v < 0 || v > static_cast<std::int64_t>(std::numeric_limits<std::uint32_t>::max())) {
                invalid_type_or_range = true;
                return;
            }
            val = static_cast<std::uint64_t>(v);
        } else {
            invalid_type_or_range = true;
            return;
        }

        if (i == 0 && val == 0) {
            facts.head_count_layer0_zero = true;
        }
        if (val == 0) {
            saw_zero = true;
            continue;
        }
        const auto u32 = static_cast<std::uint32_t>(val);
        if (!uniform_positive.has_value()) {
            uniform_positive = u32;
        } else if (*uniform_positive != u32) {
            facts.head_count_heterogeneous = true;
        }
    }

    if (facts.head_count_layer0_zero) {
        facts.head_count = 0u;
        return;
    }
    if (facts.head_count_heterogeneous) {
        return;
    }
    if (uniform_positive.has_value()) {
        facts.head_count = *uniform_positive;
    } else if (saw_zero) {
        facts.head_count = 0u;
    }
}

sonder::inference::Status ValidateDeviceConstraintForFp8(KvCacheType type_k, KvCacheType type_v,
                                                         KvDeviceKind device_kind) {
    const bool k_fp8 = (type_k == KvCacheType::kF8_E4M3);
    const bool v_fp8 = (type_v == KvCacheType::kF8_E4M3);
    if (!k_fp8 && !v_fp8) {
        return sonder::inference::Status::success();
    }
    if (device_kind == KvDeviceKind::kCuda) {
        return sonder::inference::Status::success();
    }
    const char* which = (k_fp8 && v_fp8) ? "K/V" : (k_fp8 ? "K" : "V");
    return sonder::inference::Status(
        sonder::inference::ErrorCode::invalid_argument,
        std::string(which) + " cache type 'f8_e4m3' is a CUDA-only opt-in choice; refused on device '" +
            std::string(ToString(device_kind)) + "'");
}

}  // namespace

std::string_view ToString(KvDeviceKind kind) noexcept {
    std::string_view out = "unknown";
    switch (kind) {
        case KvDeviceKind::kCpu: out = "cpu"; break;
        case KvDeviceKind::kCuda: out = "cuda"; break;
        case KvDeviceKind::kVulkan: out = "vulkan"; break;
        case KvDeviceKind::kOther: out = "other"; break;
    }
    return out;
}

std::string_view ToString(KvRotationSupport support) noexcept {
    std::string_view out = "unknown";
    switch (support) {
        case KvRotationSupport::kUnsupported: out = "unsupported"; break;
        case KvRotationSupport::kSupported: out = "supported"; break;
        case KvRotationSupport::kUnknown: out = "unknown"; break;
    }
    return out;
}

Status KvCacheTypeToGgmlType(KvCacheType type, int& out_ggml_type) noexcept {
    int mapped = -1;
    const char* unsupported_name = nullptr;
    switch (type) {
        case KvCacheType::kF16: mapped = GGML_TYPE_F16; break;
        case KvCacheType::kF32: mapped = GGML_TYPE_F32; break;
        case KvCacheType::kBF16: mapped = GGML_TYPE_BF16; break;
        case KvCacheType::kQ8_0: mapped = GGML_TYPE_Q8_0; break;
        case KvCacheType::kQ5_1: mapped = GGML_TYPE_Q5_1; break;
        case KvCacheType::kQ5_0: mapped = GGML_TYPE_Q5_0; break;
        case KvCacheType::kQ4_1: mapped = GGML_TYPE_Q4_1; break;
        case KvCacheType::kQ4_0: mapped = GGML_TYPE_Q4_0; break;
        case KvCacheType::kIQ4_NL: mapped = GGML_TYPE_IQ4_NL; break;
        case KvCacheType::kF8_E4M3:
#if defined(SONDER_LLAMACPP_HAS_GGML_F8_E4M3)
            mapped = GGML_TYPE_F8_E4M3;
#else
            unsupported_name = "f8_e4m3";
#endif
            break;
        case KvCacheType::kQ2_0:
#if defined(SONDER_LLAMACPP_HAS_GGML_Q2_0)
            mapped = GGML_TYPE_Q2_0;
#else
            unsupported_name = "q2_0";
#endif
            break;
        case KvCacheType::kPQ2_0:
#if defined(SONDER_LLAMACPP_HAS_GGML_PQ2_0)
            mapped = GGML_TYPE_PQ2_0;
#else
            unsupported_name = "pq2_0";
#endif
            break;
    }

    if (unsupported_name != nullptr) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "KV cache type '" + std::string(unsupported_name) +
                                 "' is not supported by this llama.cpp build");
    }
    if (mapped < 0) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "unknown KV cache type enum value (" +
                                 std::to_string(static_cast<int>(type)) + ")");
    }
    out_ggml_type = mapped;
    return Status::Ok();
}

KvDeviceKind ClassifyKvDeviceName(std::string_view device_name) noexcept {
    if (EqualsIgnoreCaseAscii(device_name, "cpu") || StartsWithIgnoreCase(device_name, "cpu:")) {
        return KvDeviceKind::kCpu;
    }
    if (StartsWithIgnoreCase(device_name, "cuda")) {
        return KvDeviceKind::kCuda;
    }
    if (StartsWithIgnoreCase(device_name, "vulkan")) {
        return KvDeviceKind::kVulkan;
    }
    return KvDeviceKind::kOther;
}

std::optional<KvDeviceKind> InferConfiguredKvDeviceKind(
    std::int32_t gpu_layers, const std::vector<std::string>& devices) noexcept {
    if (gpu_layers == 0) {
        return KvDeviceKind::kCpu;
    }
    if (devices.empty()) {
        return std::nullopt;
    }
    std::vector<KvDeviceKind> kinds;
    kinds.reserve(devices.size());
    for (const std::string& dev : devices) {
        if (dev.empty() || dev.find('\0') != std::string::npos) {
            return std::nullopt;
        }
        kinds.push_back(ClassifyKvDeviceName(dev));
    }
    return CombineDeviceKinds(kinds);
}

KvDeviceKind ResolveLoadKvDeviceKind(
    std::int32_t n_gpu_layers,
    const std::vector<std::string>& resolved_devices,
    const std::vector<DeviceInfo>& enumerated_devices) noexcept {
    if (n_gpu_layers == 0) {
        return KvDeviceKind::kCpu;
    }
    std::vector<KvDeviceKind> kinds;
    if (!resolved_devices.empty()) {
        kinds.reserve(resolved_devices.size());
        for (const std::string& dev : resolved_devices) {
            kinds.push_back(ClassifyKvDeviceName(dev));
        }
        return CombineDeviceKinds(kinds);
    }
    for (const DeviceInfo& d : enumerated_devices) {
        if (d.kind == DeviceKind::kGpu || d.kind == DeviceKind::kIntegratedGpu) {
            kinds.push_back(ClassifyKvDeviceName(d.name));
        }
    }
    return CombineDeviceKinds(kinds);
}

KvRotationSupport DetectKvRotationSupport(const KvModelRotationFacts& facts) noexcept {
    if (!facts.model_attn_rot.has_value()) {
        return KvRotationSupport::kUnknown;
    }
    if (!*facts.model_attn_rot || facts.env_attn_rot_disable) {
        return KvRotationSupport::kUnsupported;
    }
    if (facts.architecture.empty()) {
        return KvRotationSupport::kUnknown;
    }

    const std::string arch = LowerAsciiCopy(facts.architecture);
    if (IsNonAttentionKvArchitecture(arch)) {
        return KvRotationSupport::kUnsupported;
    }

    // In llama.cpp v1.0.0 (src/llama-hparams.cpp:49-55, src/llama-model.cpp:1617-1641),
    // hparams.n_head() returns n_head_arr[0] and gates both `n_embd / n_head()` and
    // `LLM_KV_ATTENTION_KEY_LENGTH` behind `if (hparams.n_head() > 0)`, setting
    // n_embd_head_k_full = 0 when n_head(0) == 0. Only plamo2 (src/models/plamo2.cpp:15)
    // re-reads LLM_KV_ATTENTION_KEY_LENGTH in load_arch_hparams.
    const bool is_plamo2 = (arch == "plamo2");
    if (!is_plamo2) {
        if (facts.head_count_layer0_zero ||
            (facts.head_count.has_value() && *facts.head_count == 0u)) {
            return KvRotationSupport::kUnsupported;
        }
        if (!facts.head_count.has_value() && !facts.head_count_heterogeneous) {
            return KvRotationSupport::kUnknown;
        }
    }

    std::optional<std::uint32_t> head_k_full = facts.key_length;
    if (!head_k_full.has_value()) {
        if (facts.head_count_layer0_zero || facts.head_count_heterogeneous) {
            return KvRotationSupport::kUnsupported;
        }
        if (facts.head_count.has_value() && *facts.head_count == 0u) {
            return KvRotationSupport::kUnsupported;
        }
        if (!facts.embedding_length.has_value() || !facts.head_count.has_value()) {
            return KvRotationSupport::kUnknown;
        }
        if (*facts.embedding_length == 0u || (*facts.embedding_length % *facts.head_count) != 0u) {
            return KvRotationSupport::kUnsupported;
        }
        head_k_full = *facts.embedding_length / *facts.head_count;
    }

    if (*head_k_full == 0u || (*head_k_full % 64u) != 0u) {
        return KvRotationSupport::kUnsupported;
    }
    if (facts.key_length_swa.has_value()) {
        if (*facts.key_length_swa == 0u || (*facts.key_length_swa % 64u) != 0u) {
            return KvRotationSupport::kUnsupported;
        }
    }
    return KvRotationSupport::kSupported;
}

KvRotationSupport ProbeGgufKvRotationSupport(const std::string& model_path) {
    if (model_path.empty() || !IsValidUtf8NoNul(model_path)) {
        return KvRotationSupport::kUnknown;
    }

    const gguf_init_params params{/*no_alloc=*/true, /*ctx=*/nullptr};
    GgufContextGuard guard{gguf_init_from_file(model_path.c_str(), params)};
    if (guard.ctx == nullptr) {
        return KvRotationSupport::kUnknown;
    }

    const int64_t arch_id = gguf_find_key(guard.ctx, "general.architecture");
    if (arch_id < 0 || gguf_get_kv_type(guard.ctx, arch_id) != GGUF_TYPE_STRING) {
        return KvRotationSupport::kUnknown;
    }
    const char* arch_str = gguf_get_val_str(guard.ctx, arch_id);
    if (arch_str == nullptr || arch_str[0] == '\0') {
        return KvRotationSupport::kUnknown;
    }

    KvModelRotationFacts facts;
    facts.architecture = arch_str;
    const char* env_rot = std::getenv("LLAMA_ATTN_ROT_DISABLE");
    facts.env_attn_rot_disable = (env_rot != nullptr && std::atoi(env_rot) != 0);
    // llama.cpp v1.0.0 initializes llama_model::attn_rot = true (src/llama-model.h:783);
    // llama_model_set_attn_rot exists only in internal src/llama-ext.h:124 and is never called.
    facts.model_attn_rot = true;

    bool invalid = false;
    facts.key_length = ReadGgufScalarU32(guard.ctx, facts.architecture + ".attention.key_length", invalid);
    facts.key_length_swa = ReadGgufScalarU32(guard.ctx, facts.architecture + ".attention.key_length_swa", invalid);
    facts.embedding_length = ReadGgufScalarU32(guard.ctx, facts.architecture + ".embedding_length", invalid);
    facts.indexer_head_size = ReadGgufScalarU32(guard.ctx, facts.architecture + ".attention.indexer_head_size", invalid);
    ReadGgufHeadCount(guard.ctx, facts.architecture + ".attention.head_count", facts, invalid);
    if (invalid) {
        return KvRotationSupport::kUnknown;
    }

    return DetectKvRotationSupport(facts);
}

sonder::inference::Status ValidateKvPolicyOptions(
    std::string_view type_k_name,
    std::string_view type_v_name,
    bool kv_k_q4_allowed,
    std::optional<KvDeviceKind> configured_device,
    KvCacheType& out_type_k,
    KvCacheType& out_type_v) {
    using sonder::inference::ErrorCode;
    using sonder::inference::Status;

    KvCacheType parsed_k = kDefaultKvPolicyType;
    if (!ParseKvCacheType(type_k_name, parsed_k)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown K cache type '" + std::string(type_k_name) +
                          "' (expected f16, f32, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl, f8_e4m3, q2_0 or pq2_0)");
    }
    KvCacheType parsed_v = kDefaultKvPolicyType;
    if (!ParseKvCacheType(type_v_name, parsed_v)) {
        return Status(ErrorCode::invalid_argument,
                      "unknown V cache type '" + std::string(type_v_name) +
                          "' (expected f16, f32, bf16, q8_0, q5_1, q5_0, q4_1, q4_0, iq4_nl, f8_e4m3, q2_0 or pq2_0)");
    }

    int ggml_k = -1;
    if (const auto sk = KvCacheTypeToGgmlType(parsed_k, ggml_k); !sk.ok()) {
        return Status(ErrorCode::invalid_argument,
                      "K cache type '" + std::string(ToString(parsed_k)) +
                          "' is not supported by this llama.cpp build");
    }
    int ggml_v = -1;
    if (const auto sv = KvCacheTypeToGgmlType(parsed_v, ggml_v); !sv.ok()) {
        return Status(ErrorCode::invalid_argument,
                      "V cache type '" + std::string(ToString(parsed_v)) +
                          "' is not supported by this llama.cpp build");
    }

    if (configured_device.has_value()) {
        if (Status dev_st = ValidateDeviceConstraintForFp8(parsed_k, parsed_v, *configured_device); !dev_st.ok()) {
            return dev_st;
        }
    }

    if (parsed_k == KvCacheType::kQ4_0 && !kv_k_q4_allowed) {
        return Status(ErrorCode::invalid_argument,
                      "K cache type 'q4_0' requires explicit opt-in (kv_k_q4_allowed = true) and model K rotation");
    }

    out_type_k = parsed_k;
    out_type_v = parsed_v;
    return Status::success();
}

sonder::inference::Status EvaluateKvCachePolicy(const KvPolicyRequest& req) {
    using sonder::inference::ErrorCode;
    using sonder::inference::Status;

    int ggml_k = -1;
    if (const auto sk = KvCacheTypeToGgmlType(req.type_k, ggml_k); !sk.ok()) {
        return Status(ErrorCode::invalid_argument, "K " + sk.message);
    }
    int ggml_v = -1;
    if (const auto sv = KvCacheTypeToGgmlType(req.type_v, ggml_v); !sv.ok()) {
        return Status(ErrorCode::invalid_argument, "V " + sv.message);
    }

    if (Status dev_st = ValidateDeviceConstraintForFp8(req.type_k, req.type_v, req.device_kind); !dev_st.ok()) {
        return dev_st;
    }

    if (req.type_k == KvCacheType::kQ4_0) {
        if (!req.kv_k_q4_allowed) {
            return Status(
                ErrorCode::invalid_argument,
                "K cache type 'q4_0' requires explicit opt-in (kv_k_q4_allowed = true) and model K rotation");
        }
        if (req.rotation == KvRotationSupport::kUnknown) {
            return Status(
                ErrorCode::invalid_argument,
                "K cache type 'q4_0' refused: cannot detect K-cache Hadamard rotation (attn_rot_k) support for model");
        }
        if (req.rotation != KvRotationSupport::kSupported) {
            return Status(
                ErrorCode::invalid_argument,
                "K cache type 'q4_0' requires model K-cache Hadamard rotation (attn_rot_k: head_k % 64 == 0 and rotation enabled)");
        }
    }

    return Status::success();
}

}  // namespace sonder::backends::llamacpp
