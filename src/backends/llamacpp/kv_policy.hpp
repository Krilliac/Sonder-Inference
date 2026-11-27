// Sonder Inference: KV-cache dtype policy for the native llama.cpp backend.
//
// Enforces K/V element-type defaults and safety constraints when options are
// parsed and at model load:
//   - Default is "auto" for both K and V, resolving per model to q8_0 when
//     head dimensions are multiples of 32 and flash_attention != "off", else f16.
//   - q4_0 for K requires explicit opt-in (--allow-q4-k-cache / `kv_k_q4_allowed`)
//     AND model Hadamard rotation (`attn_rot_k` in llama.cpp v1.0.0);
//     undetectable rotation support is refused cleanly.
//   - f8_e4m3 is opt-in and CUDA-only (refused on CPU, Vulkan, and other
//     non-CUDA backends).
//   - Unknown or build-unsupported types fail explicitly with no silent
//     fallback.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::backends::llamacpp {

inline constexpr KvCacheType kDefaultKvPolicyType = KvCacheType::kQ8_0;
inline constexpr std::string_view kDefaultKvPolicyTypeName = "q8_0";

enum class KvDeviceKind : std::uint8_t {
    kCpu = 0,
    kCuda = 1,
    kVulkan = 2,
    kOther = 3,
};

[[nodiscard]] std::string_view ToString(KvDeviceKind kind) noexcept;

enum class KvRotationSupport : std::uint8_t {
    kUnsupported = 0,
    kSupported = 1,
    kUnknown = 2,
};

[[nodiscard]] std::string_view ToString(KvRotationSupport support) noexcept;

// Metadata facts extracted from a GGUF header to evaluate llama.cpp v1.0.0
// K-cache Hadamard rotation eligibility (src/llama-kv-cache.cpp:300-336,
// src/llama-model.cpp:1617-1648) and q8_0 default head-dim compatibility
// (src/llama-context.cpp:4552-4572).
struct KvModelRotationFacts {
    std::string architecture;
    std::optional<std::uint32_t> key_length;
    std::optional<std::uint32_t> value_length;
    std::optional<std::uint32_t> key_length_swa;
    std::optional<std::uint32_t> value_length_swa;
    std::optional<std::uint32_t> embedding_length;
    std::optional<std::uint32_t> head_count;
    bool head_count_layer0_zero = false;
    bool head_count_heterogeneous = false;
    std::optional<std::uint32_t> kv_lora_rank;
    std::optional<bool> model_attn_rot = true;
    bool env_attn_rot_disable = false;
};

struct KvPolicyRequest {
    KvCacheType type_k = kDefaultKvPolicyType;
    KvCacheType type_v = kDefaultKvPolicyType;
    KvDeviceKind device_kind = KvDeviceKind::kCpu;
    KvRotationSupport rotation = KvRotationSupport::kUnsupported;
    bool kv_k_q4_allowed = false;
};

// Validates RFC 3629 UTF-8 with no embedded NUL bytes.
[[nodiscard]] bool IsValidUtf8NoNul(std::string_view s) noexcept;

// Maps KvCacheType onto ggml_type (returned as int so ggml.h stays private to
// sonder_backend_llamacpp). Fails with kInvalidArgument on unknown enum values
// or types not compiled into this ggml build; never silently falls back.
[[nodiscard]] Status KvCacheTypeToGgmlType(KvCacheType type, int& out_ggml_type) noexcept;

// Classifies a single ggml device name (e.g. "CPU", "CUDA0", "Vulkan0").
[[nodiscard]] KvDeviceKind ClassifyKvDeviceName(std::string_view device_name) noexcept;

// Infers target device kind from static options when unambiguous
// (gpu_layers == 0 -> kCpu; non-empty devices -> classified target; else nullopt).
[[nodiscard]] std::optional<KvDeviceKind> InferConfiguredKvDeviceKind(
    std::int32_t gpu_layers, const std::vector<std::string>& devices) noexcept;

// Resolves target device kind at load time from effective n_gpu_layers,
// explicit/pinned devices, and enumerated runtime devices.
[[nodiscard]] KvDeviceKind ResolveLoadKvDeviceKind(
    std::int32_t n_gpu_layers,
    const std::vector<std::string>& resolved_devices,
    const std::vector<DeviceInfo>& enumerated_devices) noexcept;

// Pure detector implementing llama.cpp v1.0.0's attn_rot_k eligibility rules.
[[nodiscard]] KvRotationSupport DetectKvRotationSupport(const KvModelRotationFacts& facts) noexcept;

// Reads GGUF metadata only (gguf_init_from_file with no_alloc=true) plus
// LLAMA_ATTN_ROT_DISABLE to determine K-cache rotation support for `model_path`.
[[nodiscard]] KvRotationSupport ProbeGgufKvRotationSupport(const std::string& model_path);

// Returns true when model head dims (K, V, and optional SWA) are non-zero
// multiples of 32 so default "auto" KV cache types can safely resolve to q8_0.
[[nodiscard]] bool DoesModelQualifyForDefaultQ8Kv(const KvModelRotationFacts& facts) noexcept;
[[nodiscard]] bool DoesGgufQualifyForDefaultQ8Kv(const std::string& model_path);

// Parses K/V type strings ("auto" resolves to f16 when flash_attn == kDisabled,
// else q8_0; unknown or empty names fail with invalid_argument) and validates
// option-parse-time KV policy rules.
[[nodiscard]] sonder::inference::Status ValidateKvPolicyOptions(
    std::string_view type_k_name,
    std::string_view type_v_name,
    bool kv_k_q4_allowed,
    std::optional<KvDeviceKind> configured_device,
    KvCacheType& out_type_k,
    KvCacheType& out_type_v,
    FlashAttention flash_attn = FlashAttention::kAuto);

// Load-time KV policy hook: resolves per-model "auto" K/V types (q8_0 when
// head dims % 32 == 0 and flash_attn != kDisabled, else f16), lazily enumerates
// devices only when f8_e4m3 is active, and enforces q4_0 rotation (refusing
// non-regular or unreadable GGUFs with kUnknown) and f8_e4m3 CUDA rules.
[[nodiscard]] sonder::inference::Status CheckKvPolicyAtLoad(
    LoadOptions& lo,
    std::string_view configured_type_k,
    std::string_view configured_type_v,
    bool kv_k_q4_allowed,
    const std::vector<DeviceInfo>& pre_enumerated_devices,
    const std::string& model_path);

// Full KV-cache dtype policy check (used at load time and directly in tests).
[[nodiscard]] sonder::inference::Status EvaluateKvCachePolicy(const KvPolicyRequest& req);

// Convenience overload mapping bool has_rotation -> kSupported / kUnsupported.
[[nodiscard]] inline sonder::inference::Status EvaluateKvCachePolicy(
    KvCacheType type_k, KvCacheType type_v, KvDeviceKind device_kind,
    bool has_rotation, bool kv_k_q4_allowed) {
    return EvaluateKvCachePolicy(KvPolicyRequest{
        type_k,
        type_v,
        device_kind,
        has_rotation ? KvRotationSupport::kSupported : KvRotationSupport::kUnsupported,
        kv_k_q4_allowed,
    });
}

}  // namespace sonder::backends::llamacpp
