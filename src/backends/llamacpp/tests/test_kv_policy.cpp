// Unit and integration-boundary tests for the native llama.cpp KV-cache dtype policy.
#include "kv_policy.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include <doctest/doctest.h>

namespace {

namespace lc = sonder::backends::llamacpp;
namespace si = sonder::inference;

std::uint64_t CurrentTestProcessId() noexcept {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(_getpid());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

// Minimal GGUF v3 metadata-only writer (zero tensors) for testing load-time
// rotation detection through real gguf_init_from_file without model weights.
class ScopedTempGguf {
public:
    explicit ScopedTempGguf(std::string_view stem) {
        static std::atomic<std::uint64_t> counter{0};
        const std::uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed) + 1;
        path_ = std::filesystem::temp_directory_path() /
                ("sonder_kv_policy_" + std::string(stem) + "_" +
                 std::to_string(CurrentTestProcessId()) + "_" + std::to_string(seq) + ".gguf");
    }

    ~ScopedTempGguf() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }

    ScopedTempGguf(const ScopedTempGguf&) = delete;
    ScopedTempGguf& operator=(const ScopedTempGguf&) = delete;

    struct KvEntry {
        enum class Kind { kStr, kU32, kU32Array, kStrArray, kF32Array };
        std::string key;
        Kind kind = Kind::kU32;
        std::string str_val;
        std::uint32_t u32_val = 0;
        std::vector<std::uint32_t> u32_arr;
        std::vector<std::string> str_arr;
        std::vector<float> f32_arr;
    };

    bool Write(const std::vector<KvEntry>& entries) const {
        std::ofstream out(path_, std::ios::binary | std::ios::trunc);
        if (!out) return false;

        auto write_bytes = [&](const void* data, std::size_t n) {
            out.write(static_cast<const char*>(data), static_cast<std::streamsize>(n));
        };
        auto write_u32 = [&](std::uint32_t v) { write_bytes(&v, sizeof(v)); };
        auto write_i32 = [&](std::int32_t v) { write_bytes(&v, sizeof(v)); };
        auto write_f32 = [&](float v) { write_bytes(&v, sizeof(v)); };
        auto write_u64 = [&](std::uint64_t v) { write_bytes(&v, sizeof(v)); };
        auto write_i64 = [&](std::int64_t v) { write_bytes(&v, sizeof(v)); };
        auto write_str = [&](const std::string& s) {
            write_u64(static_cast<std::uint64_t>(s.size()));
            if (!s.empty()) write_bytes(s.data(), s.size());
        };

        // GGUF v3 header: magic "GGUF", version 3, n_tensors = 0, n_kv = entries.size().
        write_bytes("GGUF", 4);
        write_u32(3u);
        write_i64(0);
        write_i64(static_cast<std::int64_t>(entries.size()));

        constexpr std::int32_t kGgufTypeUint32 = 4;
        constexpr std::int32_t kGgufTypeFloat32 = 6;
        constexpr std::int32_t kGgufTypeString = 8;
        constexpr std::int32_t kGgufTypeArray = 9;

        for (const KvEntry& e : entries) {
            write_str(e.key);
            if (e.kind == KvEntry::Kind::kStr) {
                write_i32(kGgufTypeString);
                write_str(e.str_val);
            } else if (e.kind == KvEntry::Kind::kU32) {
                write_i32(kGgufTypeUint32);
                write_u32(e.u32_val);
            } else if (e.kind == KvEntry::Kind::kU32Array) {
                write_i32(kGgufTypeArray);
                write_i32(kGgufTypeUint32);
                write_u64(static_cast<std::uint64_t>(e.u32_arr.size()));
                for (std::uint32_t item : e.u32_arr) {
                    write_u32(item);
                }
            } else if (e.kind == KvEntry::Kind::kStrArray) {
                write_i32(kGgufTypeArray);
                write_i32(kGgufTypeString);
                write_u64(static_cast<std::uint64_t>(e.str_arr.size()));
                for (const std::string& item : e.str_arr) {
                    write_str(item);
                }
            } else {
                write_i32(kGgufTypeArray);
                write_i32(kGgufTypeFloat32);
                write_u64(static_cast<std::uint64_t>(e.f32_arr.size()));
                for (float item : e.f32_arr) {
                    write_f32(item);
                }
            }
        }
        out.flush();
        return static_cast<bool>(out);
    }

    [[nodiscard]] std::string PathString() const {
        const std::u8string u = path_.u8string();
        return std::string(u.begin(), u.end());
    }

private:
    std::filesystem::path path_;
};

bool ContainsCaseInsensitive(std::string_view haystack, std::string_view needle) {
    if (needle.empty()) return true;
    if (haystack.size() < needle.size()) return false;
    for (std::size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            const auto a = static_cast<unsigned char>(haystack[i + j]);
            const auto b = static_cast<unsigned char>(needle[j]);
            const auto la = static_cast<unsigned char>(a >= 'A' && a <= 'Z' ? a + ('a' - 'A') : a);
            const auto lb = static_cast<unsigned char>(b >= 'A' && b <= 'Z' ? b + ('a' - 'A') : b);
            if (la != lb) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

}  // namespace

TEST_CASE("kv_policy: defaults resolve per model to q8_0 when head dims % 32 == 0 and flash_attn != off, else f16") {
    const lc::KvPolicyRequest req;
    CHECK(req.type_k == lc::KvCacheType::kQ8_0);
    CHECK(req.type_v == lc::KvCacheType::kQ8_0);
    CHECK_FALSE(req.kv_k_q4_allowed);
    CHECK(lc::ToString(lc::kDefaultKvPolicyType) == "q8_0");

    const si::LlamaCppBackendOptions opts;
    CHECK(opts.kv_cache_type_k == "auto");
    CHECK(opts.kv_cache_type_v == "auto");
    CHECK_FALSE(opts.kv_k_q4_allowed);
    CHECK(si::validate_llamacpp_options(opts).ok());

    // Default options with flash_attention = "off" resolve to f16 and validate cleanly,
    // whereas an explicit quantized V cache ("q8_0") with flash_attention = "off" is still refused.
    si::LlamaCppBackendOptions fa_off_default;
    fa_off_default.flash_attention = "off";
    CHECK(si::validate_llamacpp_options(fa_off_default).ok());

    si::LlamaCppBackendOptions fa_off_explicit_q8;
    fa_off_explicit_q8.kv_cache_type_k = "q8_0";
    fa_off_explicit_q8.kv_cache_type_v = "q8_0";
    fa_off_explicit_q8.flash_attention = "off";
    CHECK(si::validate_llamacpp_options(fa_off_explicit_q8).code() == si::ErrorCode::invalid_argument);

    using Entry = ScopedTempGguf::KvEntry;
    // Llama-style head_dim = 128 (divisible by 32) -> auto resolves to q8_0 when flash_attn != off, f16 when off.
    ScopedTempGguf llama128_gguf("auto_llama128");
    REQUIRE(llama128_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
        {"llama.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
        {"llama.attention.head_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
    }));
    CHECK(lc::DoesGgufQualifyForDefaultQ8Kv(llama128_gguf.PathString()));
    {
        lc::LoadOptions lo;
        lo.flash_attn = lc::FlashAttention::kAuto;
        REQUIRE(lc::CheckKvPolicyAtLoad(lo, "auto", "auto", false, {}, llama128_gguf.PathString()).ok());
        CHECK(lo.type_k == lc::KvCacheType::kQ8_0);
        CHECK(lo.type_v == lc::KvCacheType::kQ8_0);

        lc::LoadOptions lo_fa_off;
        lo_fa_off.flash_attn = lc::FlashAttention::kDisabled;
        REQUIRE(lc::CheckKvPolicyAtLoad(lo_fa_off, "auto", "auto", false, {}, llama128_gguf.PathString()).ok());
        CHECK(lo_fa_off.type_k == lc::KvCacheType::kF16);
        CHECK(lo_fa_off.type_v == lc::KvCacheType::kF16);
    }

    // Fallback case 1: Phi-2 head_dim = 80 (2560 / 32 = 80, not a multiple of 32) -> auto resolves to f16,
    // while explicit user-set "q8_0" keeps strict q8_0.
    ScopedTempGguf phi2_head80_gguf("auto_phi2_head80");
    REQUIRE(phi2_head80_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "phi2", 0, {}, {}, {}},
        {"phi2.embedding_length", Entry::Kind::kU32, {}, 2560u, {}, {}, {}},
        {"phi2.attention.head_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
    }));
    CHECK_FALSE(lc::DoesGgufQualifyForDefaultQ8Kv(phi2_head80_gguf.PathString()));
    {
        lc::LoadOptions lo;
        lo.flash_attn = lc::FlashAttention::kAuto;
        REQUIRE(lc::CheckKvPolicyAtLoad(lo, "auto", "auto", false, {}, phi2_head80_gguf.PathString()).ok());
        CHECK(lo.type_k == lc::KvCacheType::kF16);
        CHECK(lo.type_v == lc::KvCacheType::kF16);

        lc::LoadOptions lo_explicit;
        lo_explicit.type_k = lc::KvCacheType::kQ8_0;
        lo_explicit.type_v = lc::KvCacheType::kQ8_0;
        lo_explicit.flash_attn = lc::FlashAttention::kAuto;
        REQUIRE(lc::CheckKvPolicyAtLoad(lo_explicit, "q8_0", "q8_0", false, {}, phi2_head80_gguf.PathString()).ok());
        CHECK(lo_explicit.type_k == lc::KvCacheType::kQ8_0);
        CHECK(lo_explicit.type_v == lc::KvCacheType::kQ8_0);
    }

    // Fallback case 2: end-to-end load_model() with default KV types and flash_attention = "off"
    // gets past option/policy checks into llama.cpp tensor loading (backend_error, not invalid_argument).
    {
        auto be = si::make_llamacpp_backend(fa_off_default);
        si::ModelLoadOptions ml;
        ml.model = llama128_gguf.PathString();
        ml.device_id = "cpu:0";
        CHECK(be->load_model(ml).status().code() == si::ErrorCode::backend_error);
    }

    // Hyphenated MLA architectures (kimi-k3, kimi-k31, glm5-next; src/llama-arch.cpp:170-172)
    // derive n_embd_head_v_full from kv_lora_rank even when value_length is omitted or differs.
    for (const std::string_view arch : {"kimi-k3", "kimi-k31", "glm5-next"}) {
        CAPTURE(arch);
        lc::KvModelRotationFacts ok_mla;
        ok_mla.architecture = std::string(arch);
        ok_mla.embedding_length = 7168u;
        ok_mla.head_count = 64u;  // 7168 / 64 = 112 (not % 32 == 0)
        ok_mla.key_length = 576u;
        ok_mla.kv_lora_rank = 512u;
        CHECK(lc::DoesModelQualifyForDefaultQ8Kv(ok_mla));

        lc::KvModelRotationFacts bad_mla = ok_mla;
        bad_mla.value_length = 128u;
        bad_mla.kv_lora_rank = 528u;  // 528 % 32 == 16 != 0 -> overwritten n_embd_head_v_full fails % 32
        CHECK_FALSE(lc::DoesModelQualifyForDefaultQ8Kv(bad_mla));
    }

    // Fallback case 3: llama.cpp v1.0.0 forces flash attention off for every Grok model
    // (src/llama-context.cpp:4478-4481) and then rejects quantized V (src/llama-context.cpp:4546-4548),
    // so "auto" resolves to f16/f16 even though Grok head_dim = 128 is a multiple of 32.
    {
        lc::KvModelRotationFacts grok_facts;
        grok_facts.architecture = "grok";
        grok_facts.embedding_length = 6144u;
        grok_facts.head_count = 48u;  // 6144 / 48 = 128 (% 32 == 0)
        CHECK_FALSE(lc::DoesModelQualifyForDefaultQ8Kv(grok_facts));

        ScopedTempGguf grok_gguf("auto_grok128");
        REQUIRE(grok_gguf.Write({
            {"general.architecture", Entry::Kind::kStr, "grok", 0, {}, {}, {}},
            {"grok.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
            {"grok.attention.head_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
        }));
        CHECK_FALSE(lc::DoesGgufQualifyForDefaultQ8Kv(grok_gguf.PathString()));

        lc::LoadOptions lo;
        lo.flash_attn = lc::FlashAttention::kAuto;
        REQUIRE(lc::CheckKvPolicyAtLoad(lo, "auto", "auto", false, {}, grok_gguf.PathString()).ok());
        CHECK(lo.type_k == lc::KvCacheType::kF16);
        CHECK(lo.type_v == lc::KvCacheType::kF16);

        lc::LoadOptions lo_fa_on;
        lo_fa_on.flash_attn = lc::FlashAttention::kEnabled;
        REQUIRE(lc::CheckKvPolicyAtLoad(lo_fa_on, "auto", "auto", false, {}, grok_gguf.PathString()).ok());
        CHECK(lo_fa_on.type_k == lc::KvCacheType::kF16);
        CHECK(lo_fa_on.type_v == lc::KvCacheType::kF16);
    }
}

TEST_CASE("kv_policy: table test over all K types x V types x devices x rotation x opt-in") {
    constexpr std::array<lc::KvCacheType, 12> kTypes = {
        lc::KvCacheType::kF16,    lc::KvCacheType::kF32,    lc::KvCacheType::kBF16,
        lc::KvCacheType::kQ8_0,   lc::KvCacheType::kQ5_1,   lc::KvCacheType::kQ5_0,
        lc::KvCacheType::kQ4_1,   lc::KvCacheType::kQ4_0,   lc::KvCacheType::kIQ4_NL,
        lc::KvCacheType::kF8_E4M3, lc::KvCacheType::kQ2_0,  lc::KvCacheType::kPQ2_0,
    };
    constexpr std::array<lc::KvDeviceKind, 3> kDevices = {
        lc::KvDeviceKind::kCpu,
        lc::KvDeviceKind::kCuda,
        lc::KvDeviceKind::kVulkan,
    };
    constexpr std::array<lc::KvRotationSupport, 3> kRotations = {
        lc::KvRotationSupport::kUnsupported,
        lc::KvRotationSupport::kSupported,
        lc::KvRotationSupport::kUnknown,
    };
    constexpr std::array<bool, 2> kOptIns = {false, true};

    std::size_t total_cases = 0;
    std::size_t allowed_cases = 0;
    std::size_t refused_cases = 0;

    for (const lc::KvCacheType k : kTypes) {
        for (const lc::KvCacheType v : kTypes) {
            for (const lc::KvDeviceKind dev : kDevices) {
                for (const lc::KvRotationSupport rot : kRotations) {
                    for (const bool opt_in : kOptIns) {
                        ++total_cases;
                        CAPTURE(lc::ToString(k));
                        CAPTURE(lc::ToString(v));
                        CAPTURE(lc::ToString(dev));
                        CAPTURE(lc::ToString(rot));
                        CAPTURE(opt_in);

                        const bool build_supported =
                            lc::IsKvCacheTypeSupported(k) && lc::IsKvCacheTypeSupported(v);
                        const bool uses_fp8 =
                            (k == lc::KvCacheType::kF8_E4M3) || (v == lc::KvCacheType::kF8_E4M3);
                        const bool fp8_ok = !uses_fp8 || (dev == lc::KvDeviceKind::kCuda);
                        const bool k_q4_ok =
                            (k != lc::KvCacheType::kQ4_0) ||
                            (opt_in && rot == lc::KvRotationSupport::kSupported);
                        const bool expected_ok = build_supported && fp8_ok && k_q4_ok;

                        lc::KvPolicyRequest req;
                        req.type_k = k;
                        req.type_v = v;
                        req.device_kind = dev;
                        req.rotation = rot;
                        req.kv_k_q4_allowed = opt_in;

                        const si::Status st = lc::EvaluateKvCachePolicy(req);
                        CHECK(st.ok() == expected_ok);
                        if (expected_ok) {
                            ++allowed_cases;
                        } else {
                            ++refused_cases;
                            CHECK(st.code() == si::ErrorCode::invalid_argument);
                            CHECK_FALSE(st.message().empty());
                            if (!build_supported) {
                                CHECK(ContainsCaseInsensitive(st.message(), "not supported"));
                            } else if (!fp8_ok) {
                                CHECK(ContainsCaseInsensitive(st.message(), "f8_e4m3"));
                                CHECK(ContainsCaseInsensitive(st.message(), "cuda"));
                                CHECK(ContainsCaseInsensitive(st.message(), lc::ToString(dev)));
                            } else if (!opt_in) {
                                CHECK(ContainsCaseInsensitive(st.message(), "q4_0"));
                                CHECK(ContainsCaseInsensitive(st.message(), "kv_k_q4_allowed"));
                            } else if (rot == lc::KvRotationSupport::kUnknown) {
                                CHECK(ContainsCaseInsensitive(st.message(), "detect"));
                            } else {
                                CHECK(ContainsCaseInsensitive(st.message(), "rotation"));
                            }
                        }

                        // Verify bool rotation convenience overload agrees on kSupported/kUnsupported.
                        if (rot != lc::KvRotationSupport::kUnknown) {
                            const si::Status bool_st = lc::EvaluateKvCachePolicy(
                                k, v, dev, rot == lc::KvRotationSupport::kSupported, opt_in);
                            CHECK(bool_st.ok() == expected_ok);
                        }
                    }
                }
            }
        }
    }

    CHECK(total_cases == 2592u);
    CHECK(allowed_cases > 0u);
    CHECK(refused_cases > 0u);
    CHECK(allowed_cases + refused_cases == total_cases);
}

TEST_CASE("kv_policy: KvCacheType to GGML_TYPE mapping rejects unknown and unsupported values") {
    // Supported types map to distinct ggml_type integers matching ContextParamsFor.
    const lc::KvCacheType core_types[] = {
        lc::KvCacheType::kF16,   lc::KvCacheType::kF32,    lc::KvCacheType::kBF16,
        lc::KvCacheType::kQ8_0,  lc::KvCacheType::kQ5_1,   lc::KvCacheType::kQ5_0,
        lc::KvCacheType::kQ4_1,  lc::KvCacheType::kQ4_0,   lc::KvCacheType::kIQ4_NL,
        lc::KvCacheType::kF8_E4M3,
    };
    std::set<int> seen_ggml_types;
    for (const lc::KvCacheType t : core_types) {
        CAPTURE(lc::ToString(t));
        REQUIRE(lc::IsKvCacheTypeSupported(t));
        int ggml_id = -1;
        const lc::Status st = lc::KvCacheTypeToGgmlType(t, ggml_id);
        REQUIRE(st.ok());
        CHECK(ggml_id >= 0);
        seen_ggml_types.insert(ggml_id);

        lc::LoadOptions lo;
        lo.type_k = t;
        lo.type_v = t;
        const lc::ContextParamsSummary summary = lc::LlamaCppBackend::ContextParamsFor(lo);
        CHECK(summary.type_k == lc::ToString(t));
        CHECK(summary.type_v == lc::ToString(t));
    }
    CHECK(seen_ggml_types.size() == 10u);

    // Out-of-range enum values fail explicitly and never silently map to f16.
    for (const int raw : {-1, 42, 99, 255}) {
        int ggml_id = -999;
        const auto bad = static_cast<lc::KvCacheType>(raw);
        const lc::Status st = lc::KvCacheTypeToGgmlType(bad, ggml_id);
        CHECK(st.code == lc::ErrorCode::kInvalidArgument);
        CHECK(ggml_id == -999);

        lc::LoadOptions bad_lo;
        bad_lo.type_k = bad;
        bad_lo.type_v = bad;
        const lc::ContextParamsSummary bad_summary = lc::LlamaCppBackend::ContextParamsFor(bad_lo);
        CHECK(bad_summary.type_k == "unknown");
        CHECK(bad_summary.type_v == "unknown");

        lc::KvPolicyRequest req;
        req.type_k = bad;
        CHECK(lc::EvaluateKvCachePolicy(req).code() == si::ErrorCode::invalid_argument);

        req = {};
        req.type_v = bad;
        CHECK(lc::EvaluateKvCachePolicy(req).code() == si::ErrorCode::invalid_argument);
    }

#if !defined(SONDER_LLAMACPP_HAS_GGML_PQ2_0)
    {
        int ggml_id = -999;
        const lc::Status st = lc::KvCacheTypeToGgmlType(lc::KvCacheType::kPQ2_0, ggml_id);
        CHECK(st.code == lc::ErrorCode::kInvalidArgument);
        CHECK(ggml_id == -999);
    }
#endif
}

TEST_CASE("kv_policy: option parsing rejects unknown strings, unopted q4_0 K, and non-CUDA f8_e4m3") {
    for (const char* bad_name : {"", "q8", "q4_k", "fp16", "fp8", "f8_e5m2", "int8", "q8_0 "}) {
        CAPTURE(bad_name);
        si::LlamaCppBackendOptions bad_k;
        bad_k.kv_cache_type_k = bad_name;
        const si::Status sk = si::validate_llamacpp_options(bad_k);
        CHECK(sk.code() == si::ErrorCode::invalid_argument);
        CHECK(sk.message().find("K cache type") != std::string::npos);

        si::LlamaCppBackendOptions bad_v;
        bad_v.kv_cache_type_v = bad_name;
        const si::Status sv = si::validate_llamacpp_options(bad_v);
        CHECK(sv.code() == si::ErrorCode::invalid_argument);
        CHECK(sv.message().find("V cache type") != std::string::npos);
    }

    // q4_0 for K is refused at option validation unless kv_k_q4_allowed is true, naming --allow-q4-k-cache.
    {
        si::LlamaCppBackendOptions kq4;
        kq4.kv_cache_type_k = "q4_0";
        kq4.kv_cache_type_v = "q8_0";
        const si::Status refused = si::validate_llamacpp_options(kq4);
        CHECK(refused.code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(refused.message(), "kv_k_q4_allowed"));
        CHECK(ContainsCaseInsensitive(refused.message(), "--allow-q4-k-cache"));

        kq4.kv_k_q4_allowed = true;
        CHECK(si::validate_llamacpp_options(kq4).ok());
    }

    // Asymmetric q8_0 K + q4_0 V remains allowed without kv_k_q4_allowed.
    {
        si::LlamaCppBackendOptions asym;
        asym.kv_cache_type_k = "q8_0";
        asym.kv_cache_type_v = "q4_0";
        CHECK(si::validate_llamacpp_options(asym).ok());
    }

    // f8_e4m3 is accepted at option parse for CUDA, refused when CPU or Vulkan is configured.
    {
        si::LlamaCppBackendOptions fp8_cuda;
        fp8_cuda.kv_cache_type_k = "f8_e4m3";
        fp8_cuda.kv_cache_type_v = "f8_e4m3";
        fp8_cuda.devices = {"CUDA0"};
        fp8_cuda.gpu_layers = -1;
        CHECK(si::validate_llamacpp_options(fp8_cuda).ok());

        si::LlamaCppBackendOptions fp8_vulkan = fp8_cuda;
        fp8_vulkan.devices = {"Vulkan0"};
        const si::Status vk_st = si::validate_llamacpp_options(fp8_vulkan);
        CHECK(vk_st.code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(vk_st.message(), "vulkan"));

        si::LlamaCppBackendOptions fp8_cpu_dev = fp8_cuda;
        fp8_cpu_dev.devices = {"CPU"};
        const si::Status cpu_dev_st = si::validate_llamacpp_options(fp8_cpu_dev);
        CHECK(cpu_dev_st.code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(cpu_dev_st.message(), "cpu"));

        si::LlamaCppBackendOptions fp8_cpu_layers;
        fp8_cpu_layers.kv_cache_type_v = "f8_e4m3";
        fp8_cpu_layers.gpu_layers = 0;
        const si::Status cpu_layers_st = si::validate_llamacpp_options(fp8_cpu_layers);
        CHECK(cpu_layers_st.code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(cpu_layers_st.message(), "cpu"));

        si::LlamaCppBackendOptions fp8_rocm = fp8_cuda;
        fp8_rocm.devices = {"ROCm0"};
        CHECK(si::validate_llamacpp_options(fp8_rocm).code() == si::ErrorCode::invalid_argument);

        // Multi-GPU CUDA is allowed; mixed CUDA+Vulkan or CUDA+CPU is refused; case-insensitive names obey policy.
        si::LlamaCppBackendOptions fp8_multi_cuda = fp8_cuda;
        fp8_multi_cuda.kv_cache_type_k = "F8_E4M3";
        fp8_multi_cuda.devices = {"CUDA0", "CUDA1"};
        CHECK(si::validate_llamacpp_options(fp8_multi_cuda).ok());

        si::LlamaCppBackendOptions fp8_mixed_vk = fp8_multi_cuda;
        fp8_mixed_vk.devices = {"CUDA0", "Vulkan0"};
        CHECK(si::validate_llamacpp_options(fp8_mixed_vk).code() == si::ErrorCode::invalid_argument);

        si::LlamaCppBackendOptions fp8_mixed_cpu = fp8_multi_cuda;
        fp8_mixed_cpu.devices = {"CUDA0", "CPU"};
        CHECK(si::validate_llamacpp_options(fp8_mixed_cpu).code() == si::ErrorCode::invalid_argument);

        si::LlamaCppBackendOptions upper_q4;
        upper_q4.kv_cache_type_k = "Q4_0";
        CHECK(si::validate_llamacpp_options(upper_q4).code() == si::ErrorCode::invalid_argument);
    }
}

TEST_CASE("kv_policy: rotation detector mirrors llama.cpp v1.0.0 attn_rot_k rules") {
    // Standard head_k = 128 (4096 / 32) -> supported.
    {
        lc::KvModelRotationFacts f;
        f.architecture = "llama";
        f.embedding_length = 4096u;
        f.head_count = 32u;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kSupported);
    }

    // Explicit key_length = 128 overrides embedding_length / head_count (e.g. Qwen3/Gemma3).
    {
        lc::KvModelRotationFacts f;
        f.architecture = "qwen3";
        f.embedding_length = 2048u;
        f.head_count = 16u;
        f.key_length = 128u;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kSupported);
    }

    // Head dimension not divisible by 64 (e.g. 80 or 96) -> unsupported.
    for (const std::uint32_t bad_head_k : {48u, 80u, 96u, 112u}) {
        lc::KvModelRotationFacts f;
        f.architecture = "llama";
        f.head_count = 32u;
        f.key_length = bad_head_k;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kUnsupported);
    }

    // SWA head_k not divisible by 64 -> unsupported.
    {
        lc::KvModelRotationFacts f;
        f.architecture = "gemma3";
        f.head_count = 32u;
        f.key_length = 128u;
        f.key_length_swa = 96u;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kUnsupported);

        f.key_length_swa = 256u;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kSupported);
    }

    // LLAMA_ATTN_ROT_DISABLE or model.attn_rot == false -> unsupported.
    {
        lc::KvModelRotationFacts f;
        f.architecture = "llama";
        f.head_count = 32u;
        f.key_length = 128u;
        f.env_attn_rot_disable = true;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kUnsupported);

        f.env_attn_rot_disable = false;
        f.model_attn_rot = false;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kUnsupported);

        f.model_attn_rot = std::nullopt;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kUnknown);
    }

    // Pure recurrent and encoder-only architectures have no attention KV rotation.
    for (const char* arch : {"mamba", "mamba2", "rwkv6", "rwkv7", "bert", "modern-bert", "clip"}) {
        lc::KvModelRotationFacts f;
        f.architecture = arch;
        f.head_count = 32u;
        f.key_length = 128u;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kUnsupported);
    }

    // Heterogeneous per-layer head_count without uniform key_length -> unsupported (n_embd_head_k_all == -1).
    {
        lc::KvModelRotationFacts f;
        f.architecture = "llama4";
        f.embedding_length = 4096u;
        f.head_count_heterogeneous = true;
        CHECK(lc::DetectKvRotationSupport(f) == lc::KvRotationSupport::kUnsupported);
    }

    // Layer-0 head_count == 0 sets n_embd_head_k_full = 0 in llama.cpp v1.0.0 even when key_length is present
    // (except plamo2, which re-reads LLM_KV_ATTENTION_KEY_LENGTH in load_arch_hparams).
    {
        lc::KvModelRotationFacts qwen_l0_zero;
        qwen_l0_zero.architecture = "qwen35";
        qwen_l0_zero.embedding_length = 4096u;
        qwen_l0_zero.head_count = 0u;
        qwen_l0_zero.head_count_layer0_zero = true;
        qwen_l0_zero.key_length = 128u;
        CHECK(lc::DetectKvRotationSupport(qwen_l0_zero) == lc::KvRotationSupport::kUnsupported);

        lc::KvModelRotationFacts missing_hc_with_kl;
        missing_hc_with_kl.architecture = "llama";
        missing_hc_with_kl.key_length = 128u;
        CHECK(lc::DetectKvRotationSupport(missing_hc_with_kl) == lc::KvRotationSupport::kUnknown);

        lc::KvModelRotationFacts plamo2_with_kl;
        plamo2_with_kl.architecture = "plamo2";
        plamo2_with_kl.head_count_layer0_zero = true;
        plamo2_with_kl.key_length = 128u;
        CHECK(lc::DetectKvRotationSupport(plamo2_with_kl) == lc::KvRotationSupport::kSupported);
    }

    // Missing architecture or missing head dimensions -> kUnknown (cannot detect).
    {
        lc::KvModelRotationFacts empty_facts;
        CHECK(lc::DetectKvRotationSupport(empty_facts) == lc::KvRotationSupport::kUnknown);

        lc::KvModelRotationFacts missing_dims;
        missing_dims.architecture = "llama";
        CHECK(lc::DetectKvRotationSupport(missing_dims) == lc::KvRotationSupport::kUnknown);
    }
}

TEST_CASE("kv_policy: load_model enforces rotation on GGUF metadata and f8_e4m3 device rules") {
    using Entry = ScopedTempGguf::KvEntry;

    // 1. Metadata-only GGUF with head_k = 80 (not divisible by 64) -> refused as unsupported rotation.
    ScopedTempGguf no_rot_gguf("head80");
    REQUIRE(no_rot_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
        {"llama.embedding_length", Entry::Kind::kU32, {}, 2560u, {}, {}, {}},
        {"llama.attention.head_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(no_rot_gguf.PathString()) == lc::KvRotationSupport::kUnsupported);

    // 2. Metadata-only GGUF with missing head dimensions -> kUnknown -> refused with detection error.
    ScopedTempGguf unknown_rot_gguf("nodims");
    REQUIRE(unknown_rot_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(unknown_rot_gguf.PathString()) == lc::KvRotationSupport::kUnknown);

    // 3. Metadata-only GGUF with scalar head_count = 32 and head_k = 128 (divisible by 64) -> kSupported.
    ScopedTempGguf rot_ok_gguf("head128");
    REQUIRE(rot_ok_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "qwen35", 0, {}, {}, {}},
        {"qwen35.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
        {"qwen35.attention.head_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
        {"qwen35.attention.key_length", Entry::Kind::kU32, {}, 128u, {}, {}, {}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(rot_ok_gguf.PathString()) == lc::KvRotationSupport::kSupported);

    // 4. Per-layer head_count with head_count[0] == 0 on qwen35 -> kUnsupported (n_head(0) == 0 -> n_embd_head_k_full = 0).
    ScopedTempGguf layer0_zero_gguf("l0zero");
    REQUIRE(layer0_zero_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "qwen35", 0, {}, {}, {}},
        {"qwen35.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
        {"qwen35.attention.head_count", Entry::Kind::kU32Array, {}, 0u, {0u, 32u, 0u, 32u}, {}, {}},
        {"qwen35.attention.key_length", Entry::Kind::kU32, {}, 128u, {}, {}, {}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(layer0_zero_gguf.PathString()) == lc::KvRotationSupport::kUnsupported);

    // 5. Missing <arch>.attention.head_count with <arch>.attention.key_length = 128 on llama -> kUnknown.
    ScopedTempGguf missing_hc_gguf("nohc");
    REQUIRE(missing_hc_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
        {"llama.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
        {"llama.attention.key_length", Entry::Kind::kU32, {}, 128u, {}, {}, {}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(missing_hc_gguf.PathString()) == lc::KvRotationSupport::kUnknown);

    // 6. Malformed non-integer head_count arrays (GGUF_TYPE_STRING and GGUF_TYPE_FLOAT32) must not SIGABRT -> kUnknown.
    ScopedTempGguf str_arr_hc_gguf("strarr_hc");
    REQUIRE(str_arr_hc_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
        {"llama.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
        {"llama.attention.head_count", Entry::Kind::kStrArray, {}, 0u, {}, {"32", "32"}, {}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(str_arr_hc_gguf.PathString()) == lc::KvRotationSupport::kUnknown);

    ScopedTempGguf f32_arr_hc_gguf("f32arr_hc");
    REQUIRE(f32_arr_hc_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
        {"llama.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
        {"llama.attention.head_count", Entry::Kind::kF32Array, {}, 0u, {}, {}, {32.0f, 32.0f}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(f32_arr_hc_gguf.PathString()) == lc::KvRotationSupport::kUnknown);

    // 7. Unused indexer_head_size metadata with non-integer type does not spuriously refuse q4_0 K.
    ScopedTempGguf bad_indexer_gguf("bad_indexer");
    REQUIRE(bad_indexer_gguf.Write({
        {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
        {"llama.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
        {"llama.attention.head_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
        {"llama.attention.indexer_head_size", Entry::Kind::kStr, "not_an_int", 0u, {}, {}, {}},
    }));
    CHECK(lc::ProbeGgufKvRotationSupport(bad_indexer_gguf.PathString()) == lc::KvRotationSupport::kSupported);

    // End-to-end load_model() with kv_cache_type_k = "q4_0" + kv_k_q4_allowed = true:
    //   - no_rot_gguf & layer0_zero_gguf fail with invalid_argument naming rotation;
    //   - unknown_rot_gguf, missing_hc_gguf, str_arr_hc_gguf, and non-regular paths
    //     (missing file or directory) fail with invalid_argument naming detection;
    //   - CheckKvPolicyAtLoad + IsValidUtf8NoNul reject invalid UTF-8 / embedded NUL cleanly;
    //   - rot_ok_gguf passes policy and fails later in llama.cpp tensor load (backend_error).
    {
        si::LlamaCppBackendOptions opts;
        opts.kv_cache_type_k = "q4_0";
        opts.kv_cache_type_v = "q8_0";
        opts.kv_k_q4_allowed = true;
        auto be = si::make_llamacpp_backend(opts);

        si::ModelLoadOptions ml;
        ml.device_id = "cpu:0";

        ml.model = no_rot_gguf.PathString();
        const auto r_no_rot = be->load_model(ml);
        CHECK(r_no_rot.status().code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(r_no_rot.status().message(), "rotation"));

        ml.model = layer0_zero_gguf.PathString();
        const auto r_l0_zero = be->load_model(ml);
        CHECK(r_l0_zero.status().code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(r_l0_zero.status().message(), "rotation"));

        ml.model = unknown_rot_gguf.PathString();
        const auto r_unknown = be->load_model(ml);
        CHECK(r_unknown.status().code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(r_unknown.status().message(), "detect"));

        ml.model = missing_hc_gguf.PathString();
        const auto r_missing_hc = be->load_model(ml);
        CHECK(r_missing_hc.status().code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(r_missing_hc.status().message(), "detect"));

        ml.model = str_arr_hc_gguf.PathString();
        const auto r_str_arr = be->load_model(ml);
        CHECK(r_str_arr.status().code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(r_str_arr.status().message(), "detect"));

        const std::u8string tmp_dir_u8 = std::filesystem::temp_directory_path().u8string();
        const std::string tmp_dir_utf8(tmp_dir_u8.begin(), tmp_dir_u8.end());
        for (const std::string& non_regular : {
                 std::string("nonexistent_q4_rotation_probe.gguf"),
                 tmp_dir_utf8,
             }) {
            CAPTURE(non_regular);
            CHECK(lc::ProbeGgufKvRotationSupport(non_regular) == lc::KvRotationSupport::kUnknown);
            ml.model = non_regular;
            const auto r_non_reg = be->load_model(ml);
            CHECK(r_non_reg.status().code() == si::ErrorCode::invalid_argument);
            CHECK(ContainsCaseInsensitive(r_non_reg.status().message(), "detect"));
        }

        for (const std::string& bad_utf8 : {
                 std::string("\xFF\xFEinvalid_utf8.gguf"),
                 std::string("embedded\0nul.gguf", 17),
             }) {
            CHECK_FALSE(lc::IsValidUtf8NoNul(bad_utf8));
            CHECK(lc::ProbeGgufKvRotationSupport(bad_utf8) == lc::KvRotationSupport::kUnknown);
            lc::LoadOptions lo_q4;
            lo_q4.type_k = lc::KvCacheType::kQ4_0;
            lo_q4.type_v = lc::KvCacheType::kQ8_0;
            const si::Status st_q4 = lc::CheckKvPolicyAtLoad(lo_q4, "q4_0", "q8_0", true, {}, bad_utf8);
            CHECK(st_q4.code() == si::ErrorCode::invalid_argument);
        }

        ml.model = rot_ok_gguf.PathString();
        const auto r_rot_ok = be->load_model(ml);
        CHECK(r_rot_ok.status().code() == si::ErrorCode::backend_error);
    }

    // End-to-end load_model() refuses f8_e4m3 when load request targets CPU (device_id = "cpu:0"),
    // even when backend options left devices unspecified.
    {
        si::LlamaCppBackendOptions fp8_opts;
        fp8_opts.kv_cache_type_k = "f8_e4m3";
        fp8_opts.kv_cache_type_v = "f8_e4m3";
        auto be = si::make_llamacpp_backend(fp8_opts);

        si::ModelLoadOptions ml;
        ml.model = rot_ok_gguf.PathString();
        ml.device_id = "cpu:0";
        const auto r_cpu = be->load_model(ml);
        CHECK(r_cpu.status().code() == si::ErrorCode::invalid_argument);
        CHECK(ContainsCaseInsensitive(r_cpu.status().message(), "cuda"));
        CHECK(ContainsCaseInsensitive(r_cpu.status().message(), "cpu"));

        // Partial offload on CUDA (0 < n_gpu_layers <= block_count; src/llama-model.cpp:1778)
        // leaves KV cache layers on CPU and must refuse f8_e4m3; full offload (-1 or >= block_count + 1) passes.
        ScopedTempGguf blk32_gguf("fp8_blk32");
        REQUIRE(blk32_gguf.Write({
            {"general.architecture", Entry::Kind::kStr, "llama", 0, {}, {}, {}},
            {"llama.block_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
            {"llama.embedding_length", Entry::Kind::kU32, {}, 4096u, {}, {}, {}},
            {"llama.attention.head_count", Entry::Kind::kU32, {}, 32u, {}, {}, {}},
        }));
        CHECK(lc::ResolveLoadKvDeviceKind(16, {"CUDA0"}, {}, 32u) == lc::KvDeviceKind::kCpu);
        CHECK(lc::ResolveLoadKvDeviceKind(32, {"CUDA0"}, {}, 32u) == lc::KvDeviceKind::kCpu);
        CHECK(lc::ResolveLoadKvDeviceKind(33, {"CUDA0"}, {}, 32u) == lc::KvDeviceKind::kCuda);
        CHECK(lc::ResolveLoadKvDeviceKind(-1, {"CUDA0"}, {}, 32u) == lc::KvDeviceKind::kCuda);

        for (const std::int32_t partial_ngl : {1, 16, 32}) {
            CAPTURE(partial_ngl);
            lc::LoadOptions lo_partial;
            lo_partial.type_k = lc::KvCacheType::kF8_E4M3;
            lo_partial.type_v = lc::KvCacheType::kF8_E4M3;
            lo_partial.n_gpu_layers = partial_ngl;
            lo_partial.devices = {"CUDA0"};
            const si::Status st_partial =
                lc::CheckKvPolicyAtLoad(lo_partial, "f8_e4m3", "f8_e4m3", false, {}, blk32_gguf.PathString());
            CHECK(st_partial.code() == si::ErrorCode::invalid_argument);
            CHECK(ContainsCaseInsensitive(st_partial.message(), "cpu"));
        }
        for (const std::int32_t full_ngl : {-1, 33, 64}) {
            CAPTURE(full_ngl);
            lc::LoadOptions lo_full;
            lo_full.type_k = lc::KvCacheType::kF8_E4M3;
            lo_full.type_v = lc::KvCacheType::kF8_E4M3;
            lo_full.n_gpu_layers = full_ngl;
            lo_full.devices = {"CUDA0"};
            CHECK(lc::CheckKvPolicyAtLoad(lo_full, "f8_e4m3", "f8_e4m3", false, {}, blk32_gguf.PathString()).ok());
        }
    }

    // LLAMA_ATTN_ROT_DISABLE=1 disables rotation in ProbeGgufKvRotationSupport and load_model().
    {
        struct ScopedEnvVar {
            std::string name;
            std::optional<std::string> prev;
            ScopedEnvVar(const char* k, const char* v) : name(k) {
                if (const char* old = std::getenv(k)) prev = std::string(old);
                Set(v);
            }
            ~ScopedEnvVar() { Set(prev ? prev->c_str() : nullptr); }
            void Set(const char* v) const {
#if defined(_WIN32)
                _putenv_s(name.c_str(), v != nullptr ? v : "");
#else
                if (v != nullptr) setenv(name.c_str(), v, 1);
                else unsetenv(name.c_str());
#endif
            }
        };

        {
            ScopedEnvVar disable_rot("LLAMA_ATTN_ROT_DISABLE", "1");
            CHECK(lc::ProbeGgufKvRotationSupport(rot_ok_gguf.PathString()) == lc::KvRotationSupport::kUnsupported);

            si::LlamaCppBackendOptions opts;
            opts.kv_cache_type_k = "q4_0";
            opts.kv_cache_type_v = "q8_0";
            opts.kv_k_q4_allowed = true;
            auto be = si::make_llamacpp_backend(opts);
            si::ModelLoadOptions ml;
            ml.model = rot_ok_gguf.PathString();
            ml.device_id = "cpu:0";
            const auto r_disabled = be->load_model(ml);
            CHECK(r_disabled.status().code() == si::ErrorCode::invalid_argument);
            CHECK(ContainsCaseInsensitive(r_disabled.status().message(), "rotation"));
        }

        {
            ScopedEnvVar zero_rot("LLAMA_ATTN_ROT_DISABLE", "0");
            CHECK(lc::ProbeGgufKvRotationSupport(rot_ok_gguf.PathString()) == lc::KvRotationSupport::kSupported);
        }
    }
}
