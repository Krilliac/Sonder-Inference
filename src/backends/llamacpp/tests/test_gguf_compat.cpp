// Unit tests for load-time GGUF metadata compatibility shims (Qwen3.5/3.6 3->4
// rope.dimension_sections normalization).
//
// Builds tiny metadata-only GGUF files in a temp directory using gguf's writer
// API (no tensors, no downloads) and verifies both inspection/overlay behavior
// and real llama.cpp v1.0.0 vocab-only loading without rewriting the source file.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "ggml.h"
#include "gguf.h"
#include "gguf_compat.hpp"
#include "llama.h"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

namespace {

std::string PathToUtf8(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

struct GgufDeleter {
    void operator()(gguf_context* ctx) const noexcept {
        if (ctx != nullptr) gguf_free(ctx);
    }
};
using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;

class TempDir {
public:
    TempDir() {
        static std::atomic<std::uint64_t> seq{0};
        const auto ticks = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        dir_ = std::filesystem::temp_directory_path() /
               ("sonder-test-gguf-compat-" + std::to_string(ticks) + "-" +
                std::to_string(seq.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(dir_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    [[nodiscard]] std::filesystem::path file(const std::string& name) const { return dir_ / name; }

private:
    std::filesystem::path dir_;
};

std::vector<std::uint8_t> ReadAllBytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>{});
}

// Writes a minimal metadata-only GGUF with `general.architecture = arch` and an optional
// `<arch>.rope.dimension_sections` int32 array.
std::filesystem::path WriteMinimalRopeGguf(const TempDir& dir, const std::string& filename,
                                           const std::string& arch,
                                           const std::vector<std::int32_t>* sections_i32 = nullptr,
                                           const std::vector<std::uint32_t>* sections_u32 = nullptr) {
    GgufPtr ctx(gguf_init_empty());
    REQUIRE(ctx != nullptr);
    gguf_set_val_str(ctx.get(), "general.architecture", arch.c_str());
    gguf_set_val_str(ctx.get(), "general.name", "compat-test-model");
    const std::string key = arch + ".rope.dimension_sections";
    if (sections_i32 != nullptr) {
        gguf_set_arr_data(ctx.get(), key.c_str(), GGUF_TYPE_INT32, sections_i32->data(), sections_i32->size());
    } else if (sections_u32 != nullptr) {
        gguf_set_arr_data(ctx.get(), key.c_str(), GGUF_TYPE_UINT32, sections_u32->data(), sections_u32->size());
    }
    const std::filesystem::path out = dir.file(filename);
    REQUIRE(gguf_write_to_file(ctx.get(), PathToUtf8(out).c_str(), /*only_meta=*/true));
    return out;
}

// Writes a tiny 2-layer hybrid Qwen3.5 GGUF (layer 0 = Gated DeltaNet recurrent,
// layer 1 = full attention with M-RoPE; ~300 KB of zeroed F32 tensors) that
// executes llama_model_qwen35::load_arch_hparams(), load_vocab(), load_tensors(),
// llama_init_from_model(), and a full forward pass (`Generate`) end to end.
std::filesystem::path WriteLoadableQwen35Gguf(const TempDir& dir, const std::string& filename,
                                              const std::vector<std::int32_t>& rope_sections) {
    GgufPtr ctx(gguf_init_empty());
    REQUIRE(ctx != nullptr);
    const std::string arch = "qwen35";
    gguf_set_val_str(ctx.get(), "general.architecture", arch.c_str());
    gguf_set_val_str(ctx.get(), "general.name", "qwen35-compat-tiny");

    auto set_u32 = [&](const char* suffix, std::uint32_t v) {
        gguf_set_val_u32(ctx.get(), (arch + "." + suffix).c_str(), v);
    };
    auto set_f32 = [&](const char* suffix, float v) {
        gguf_set_val_f32(ctx.get(), (arch + "." + suffix).c_str(), v);
    };

    constexpr std::int64_t kEmbd = 64;
    constexpr std::int64_t kHead = 1;
    constexpr std::int64_t kHeadKv = 1;
    constexpr std::int64_t kHeadDim = 64;
    constexpr std::int64_t kFf = 64;
    constexpr std::int64_t kVocab = 4;
    constexpr std::int64_t kSsmConv = 4;
    constexpr std::int64_t kSsmState = 16;
    constexpr std::int64_t kSsmGroup = 4;
    constexpr std::int64_t kSsmDtRank = 4;
    constexpr std::int64_t kKeyDim = kSsmState * kSsmGroup;     // 64
    constexpr std::int64_t kValueDim = kSsmState * kSsmDtRank;  // 64
    constexpr std::int64_t kConvDim = kKeyDim * 2 + kValueDim;  // 192

    set_u32("context_length", 128);
    set_u32("embedding_length", static_cast<std::uint32_t>(kEmbd));
    set_u32("block_count", 2);
    set_u32("feed_forward_length", static_cast<std::uint32_t>(kFf));
    set_u32("attention.head_count", static_cast<std::uint32_t>(kHead));
    set_u32("attention.head_count_kv", static_cast<std::uint32_t>(kHeadKv));
    set_u32("attention.key_length", static_cast<std::uint32_t>(kHeadDim));
    set_u32("attention.value_length", static_cast<std::uint32_t>(kHeadDim));
    set_u32("rope.dimension_count", static_cast<std::uint32_t>(kHeadDim));
    set_f32("attention.layer_norm_rms_epsilon", 1e-6f);
    set_u32("full_attention_interval", 2);  // layer 0 = recurrent, layer 1 = full attention
    set_u32("ssm.conv_kernel", static_cast<std::uint32_t>(kSsmConv));
    set_u32("ssm.inner_size", static_cast<std::uint32_t>(kValueDim));
    set_u32("ssm.state_size", static_cast<std::uint32_t>(kSsmState));
    set_u32("ssm.time_step_rank", static_cast<std::uint32_t>(kSsmDtRank));
    set_u32("ssm.group_count", static_cast<std::uint32_t>(kSsmGroup));

    gguf_set_arr_data(ctx.get(), "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, rope_sections.data(),
                      rope_sections.size());

    gguf_set_val_str(ctx.get(), "tokenizer.ggml.model", "gpt2");
    const char* tokens[kVocab] = {"<|endoftext|>", "a", "b", "ab"};
    gguf_set_arr_str(ctx.get(), "tokenizer.ggml.tokens", tokens, kVocab);
    const std::int32_t token_types[kVocab] = {3, 1, 1, 1};
    gguf_set_arr_data(ctx.get(), "tokenizer.ggml.token_type", GGUF_TYPE_INT32, token_types, kVocab);
    const char* merges[1] = {"a b"};
    gguf_set_arr_str(ctx.get(), "tokenizer.ggml.merges", merges, 1);
    gguf_set_val_u32(ctx.get(), "tokenizer.ggml.eos_token_id", 0);

    // Allocate the F32 tensors required by llama_model_qwen35::load_arch_tensors (block_count=2).
    struct GgmlDeleter {
        void operator()(ggml_context* g) const noexcept {
            if (g != nullptr) ggml_free(g);
        }
    };
    ggml_init_params gparams{std::size_t{4} << 20, nullptr, /*no_alloc=*/false};
    std::unique_ptr<ggml_context, GgmlDeleter> gctx(ggml_init(gparams));
    REQUIRE(gctx != nullptr);

    auto add_1d = [&](const char* name, std::int64_t n0) {
        ggml_tensor* t = ggml_new_tensor_1d(gctx.get(), GGML_TYPE_F32, n0);
        REQUIRE(t != nullptr);
        ggml_set_name(t, name);
        ggml_set_zero(t);
        gguf_add_tensor(ctx.get(), t);
    };
    auto add_2d = [&](const char* name, std::int64_t n0, std::int64_t n1) {
        ggml_tensor* t = ggml_new_tensor_2d(gctx.get(), GGML_TYPE_F32, n0, n1);
        REQUIRE(t != nullptr);
        ggml_set_name(t, name);
        ggml_set_zero(t);
        gguf_add_tensor(ctx.get(), t);
    };

    add_2d("token_embd.weight", kEmbd, kVocab);
    add_1d("output_norm.weight", kEmbd);

    // Layer 0: Gated DeltaNet recurrent block.
    add_1d("blk.0.attn_norm.weight", kEmbd);
    add_1d("blk.0.post_attention_norm.weight", kEmbd);
    add_2d("blk.0.attn_qkv.weight", kEmbd, kConvDim);
    add_2d("blk.0.attn_gate.weight", kEmbd, kValueDim);
    add_2d("blk.0.ssm_conv1d.weight", kSsmConv, kConvDim);
    add_1d("blk.0.ssm_dt.bias", kSsmDtRank);
    add_1d("blk.0.ssm_a", kSsmDtRank);
    add_2d("blk.0.ssm_beta.weight", kEmbd, kSsmDtRank);
    add_2d("blk.0.ssm_alpha.weight", kEmbd, kSsmDtRank);
    add_1d("blk.0.ssm_norm.weight", kSsmState);
    add_2d("blk.0.ssm_out.weight", kValueDim, kEmbd);
    add_2d("blk.0.ffn_gate.weight", kEmbd, kFf);
    add_2d("blk.0.ffn_down.weight", kFf, kEmbd);
    add_2d("blk.0.ffn_up.weight", kEmbd, kFf);

    // Layer 1: Full attention block with M-RoPE.
    add_1d("blk.1.attn_norm.weight", kEmbd);
    add_1d("blk.1.post_attention_norm.weight", kEmbd);
    add_2d("blk.1.attn_q.weight", kEmbd, kHeadDim * kHead * 2);
    add_2d("blk.1.attn_k.weight", kEmbd, kHeadDim * kHeadKv);
    add_2d("blk.1.attn_v.weight", kEmbd, kHeadDim * kHeadKv);
    add_2d("blk.1.attn_output.weight", kHeadDim * kHead, kEmbd);
    add_1d("blk.1.attn_q_norm.weight", kHeadDim);
    add_1d("blk.1.attn_k_norm.weight", kHeadDim);
    add_2d("blk.1.ffn_gate.weight", kEmbd, kFf);
    add_2d("blk.1.ffn_down.weight", kFf, kEmbd);
    add_2d("blk.1.ffn_up.weight", kEmbd, kFf);

    const std::filesystem::path out = dir.file(filename);
    REQUIRE(gguf_write_to_file(ctx.get(), PathToUtf8(out).c_str(), /*only_meta=*/false));
    return out;
}

}  // namespace

TEST_CASE("gguf_compat: pure architecture allowlist and 3->4 normalization") {
    CHECK(IsQwen35RopeCompatArch("qwen35"));
    CHECK(IsQwen35RopeCompatArch("qwen35moe"));
    CHECK_FALSE(IsQwen35RopeCompatArch(""));
    CHECK_FALSE(IsQwen35RopeCompatArch("llama"));
    CHECK_FALSE(IsQwen35RopeCompatArch("qwen2"));
    CHECK_FALSE(IsQwen35RopeCompatArch("qwen3"));
    CHECK_FALSE(IsQwen35RopeCompatArch("qwen35x"));
    CHECK_FALSE(IsQwen35RopeCompatArch("gemma3"));

    const auto norm = NormalizeRopeDimensionSections({11, 11, 10});
    CHECK(norm == std::array<std::int32_t, 4>{11, 11, 10, 0});
}

TEST_CASE("gguf_compat: detects 3-element rope.dimension_sections on qwen35 and builds normalized overlay") {
    TempDir dir;
    const std::vector<std::int32_t> three = {11, 11, 10};
    const std::filesystem::path src_path = WriteMinimalRopeGguf(dir, "qwen35-3elem.gguf", "qwen35", &three);
    const std::vector<std::uint8_t> original_bytes = ReadAllBytes(src_path);

    const GgufCompatInspection insp = InspectGgufCompat(PathToUtf8(src_path));
    REQUIRE(insp.needs_shim);
    CHECK(insp.architecture == "qwen35");
    REQUIRE(insp.rope_sections.has_value());
    CHECK(insp.rope_sections->architecture == "qwen35");
    CHECK(insp.rope_sections->key == "qwen35.rope.dimension_sections");
    CHECK(insp.rope_sections->original == std::array<std::int32_t, 3>{11, 11, 10});
    CHECK(insp.rope_sections->normalized == std::array<std::int32_t, 4>{11, 11, 10, 0});
    CHECK_FALSE(insp.rope_sections->is_unsigned);
    CHECK_FALSE(insp.detail.empty());

    std::filesystem::path overlay_path;
    {
        GgufCompatOverlay overlay;
        std::string err;
        REQUIRE(PrepareGgufCompatOverlay(PathToUtf8(src_path), overlay, &err));
        CHECK(err.empty());
        REQUIRE(overlay.active());
        overlay_path = overlay.temp_meta_path();
        CHECK(std::filesystem::exists(overlay_path));
        REQUIRE(overlay.split_paths().size() == 2u);
        CHECK(overlay.split_paths()[0] == PathToUtf8(overlay_path));
        CHECK(overlay.split_paths()[1] == PathToUtf8(src_path));

        // Read back the temporary overlay GGUF and verify the normalized 4-element array.
        gguf_init_params gp{/*no_alloc=*/true, /*ctx=*/nullptr};
        GgufPtr patched(gguf_init_from_file(PathToUtf8(overlay_path).c_str(), gp));
        REQUIRE(patched != nullptr);
        CHECK(gguf_get_n_tensors(patched.get()) == 0);
        const std::int64_t kid = gguf_find_key(patched.get(), "qwen35.rope.dimension_sections");
        REQUIRE(kid >= 0);
        CHECK(gguf_get_kv_type(patched.get(), kid) == GGUF_TYPE_ARRAY);
        CHECK(gguf_get_arr_type(patched.get(), kid) == GGUF_TYPE_INT32);
        REQUIRE(gguf_get_arr_n(patched.get(), kid) == 4u);
        const auto* data = static_cast<const std::int32_t*>(gguf_get_arr_data(patched.get(), kid));
        REQUIRE(data != nullptr);
        CHECK(data[0] == 11);
        CHECK(data[1] == 11);
        CHECK(data[2] == 10);
        CHECK(data[3] == 0);

        // Unrelated metadata was preserved.
        const std::int64_t name_id = gguf_find_key(patched.get(), "general.name");
        REQUIRE(name_id >= 0);
        CHECK(std::string(gguf_get_val_str(patched.get(), name_id)) == "compat-test-model");
    }

    // Temp overlay was removed on destruction; original file on disk was never modified.
    CHECK_FALSE(std::filesystem::exists(overlay_path));
    CHECK(ReadAllBytes(src_path) == original_bytes);
}

TEST_CASE("gguf_compat: detects 3-element UINT32 rope.dimension_sections on qwen35moe and preserves element type") {
    TempDir dir;
    const std::vector<std::uint32_t> three_u32 = {16u, 16u, 16u};
    const std::filesystem::path src_path =
        WriteMinimalRopeGguf(dir, "qwen35moe-3elem-u32.gguf", "qwen35moe", nullptr, &three_u32);

    const GgufCompatInspection insp = InspectGgufCompat(PathToUtf8(src_path));
    REQUIRE(insp.needs_shim);
    CHECK(insp.architecture == "qwen35moe");
    REQUIRE(insp.rope_sections.has_value());
    CHECK(insp.rope_sections->key == "qwen35moe.rope.dimension_sections");
    CHECK(insp.rope_sections->original == std::array<std::int32_t, 3>{16, 16, 16});
    CHECK(insp.rope_sections->normalized == std::array<std::int32_t, 4>{16, 16, 16, 0});
    CHECK(insp.rope_sections->is_unsigned);

    GgufCompatOverlay overlay;
    REQUIRE(PrepareGgufCompatOverlay(PathToUtf8(src_path), overlay));
    REQUIRE(overlay.active());

    gguf_init_params gp{/*no_alloc=*/true, /*ctx=*/nullptr};
    GgufPtr patched(gguf_init_from_file(PathToUtf8(overlay.temp_meta_path()).c_str(), gp));
    REQUIRE(patched != nullptr);
    const std::int64_t kid = gguf_find_key(patched.get(), "qwen35moe.rope.dimension_sections");
    REQUIRE(kid >= 0);
    CHECK(gguf_get_arr_type(patched.get(), kid) == GGUF_TYPE_UINT32);
    REQUIRE(gguf_get_arr_n(patched.get(), kid) == 4u);
    const auto* data = static_cast<const std::uint32_t*>(gguf_get_arr_data(patched.get(), kid));
    REQUIRE(data != nullptr);
    CHECK(data[0] == 16u);
    CHECK(data[1] == 16u);
    CHECK(data[2] == 16u);
    CHECK(data[3] == 0u);
}

TEST_CASE("gguf_compat: leaves 4-element array, missing key, wrong lengths, and other architectures alone") {
    TempDir dir;

    // 1. 4-element array on qwen35 -> untouched.
    const std::vector<std::int32_t> four = {11, 11, 10, 0};
    const auto four_path = WriteMinimalRopeGguf(dir, "qwen35-4elem.gguf", "qwen35", &four);
    {
        const auto insp = InspectGgufCompat(PathToUtf8(four_path));
        CHECK_FALSE(insp.needs_shim);
        CHECK_FALSE(insp.rope_sections.has_value());
        GgufCompatOverlay overlay;
        CHECK(PrepareGgufCompatOverlay(PathToUtf8(four_path), overlay));
        CHECK_FALSE(overlay.active());
    }

    // 2. Missing key on qwen35 -> untouched.
    const auto missing_path = WriteMinimalRopeGguf(dir, "qwen35-missing.gguf", "qwen35", nullptr);
    {
        const auto insp = InspectGgufCompat(PathToUtf8(missing_path));
        CHECK_FALSE(insp.needs_shim);
        GgufCompatOverlay overlay;
        CHECK(PrepareGgufCompatOverlay(PathToUtf8(missing_path), overlay));
        CHECK_FALSE(overlay.active());
    }

    // 3. Other architectures (llama, qwen2, qwen3, gemma3) with 3-element array -> untouched.
    const std::vector<std::int32_t> three = {11, 11, 10};
    for (const char* arch : {"llama", "qwen2", "qwen3", "gemma3"}) {
        CAPTURE(arch);
        const auto other_path = WriteMinimalRopeGguf(dir, std::string(arch) + "-3elem.gguf", arch, &three);
        const auto insp = InspectGgufCompat(PathToUtf8(other_path));
        CHECK_FALSE(insp.needs_shim);
        CHECK_FALSE(insp.rope_sections.has_value());
        GgufCompatOverlay overlay;
        CHECK(PrepareGgufCompatOverlay(PathToUtf8(other_path), overlay));
        CHECK_FALSE(overlay.active());
    }

    // 4. Unexpected array lengths (2 or 5) on qwen35 -> untouched (let llama.cpp reject).
    const std::vector<std::int32_t> two = {11, 11};
    const std::vector<std::int32_t> five = {11, 11, 10, 0, 0};
    CHECK_FALSE(InspectGgufCompat(PathToUtf8(WriteMinimalRopeGguf(dir, "qwen35-2elem.gguf", "qwen35", &two))).needs_shim);
    CHECK_FALSE(InspectGgufCompat(PathToUtf8(WriteMinimalRopeGguf(dir, "qwen35-5elem.gguf", "qwen35", &five))).needs_shim);

    // 5. Hostile element types & out-of-range values: 3-element STRING array (must NOT abort
    //    inside gguf_get_arr_data), 3-element FLOAT32 array, scalar INT32, negative INT32, and
    //    UINT32 > INT32_MAX -> all safely left alone (needs_shim == false).
    {
        GgufPtr ctx(gguf_init_empty());
        REQUIRE(ctx != nullptr);
        gguf_set_val_str(ctx.get(), "general.architecture", "qwen35");
        const char* strs[3] = {"11", "11", "10"};
        gguf_set_arr_str(ctx.get(), "qwen35.rope.dimension_sections", strs, 3);
        const auto p = dir.file("qwen35-str-arr.gguf");
        REQUIRE(gguf_write_to_file(ctx.get(), PathToUtf8(p).c_str(), /*only_meta=*/true));
        CHECK_FALSE(InspectGgufCompat(PathToUtf8(p)).needs_shim);
        GgufCompatOverlay overlay;
        CHECK(PrepareGgufCompatOverlay(PathToUtf8(p), overlay));
        CHECK_FALSE(overlay.active());
    }
    {
        GgufPtr ctx(gguf_init_empty());
        REQUIRE(ctx != nullptr);
        gguf_set_val_str(ctx.get(), "general.architecture", "qwen35");
        const float floats[3] = {11.0f, 11.0f, 10.0f};
        gguf_set_arr_data(ctx.get(), "qwen35.rope.dimension_sections", GGUF_TYPE_FLOAT32, floats, 3);
        const auto p = dir.file("qwen35-f32-arr.gguf");
        REQUIRE(gguf_write_to_file(ctx.get(), PathToUtf8(p).c_str(), /*only_meta=*/true));
        CHECK_FALSE(InspectGgufCompat(PathToUtf8(p)).needs_shim);
    }
    {
        GgufPtr ctx(gguf_init_empty());
        REQUIRE(ctx != nullptr);
        gguf_set_val_str(ctx.get(), "general.architecture", "qwen35");
        gguf_set_val_i32(ctx.get(), "qwen35.rope.dimension_sections", 11);
        const auto p = dir.file("qwen35-scalar-i32.gguf");
        REQUIRE(gguf_write_to_file(ctx.get(), PathToUtf8(p).c_str(), /*only_meta=*/true));
        CHECK_FALSE(InspectGgufCompat(PathToUtf8(p)).needs_shim);
    }
    {
        const std::vector<std::int32_t> neg = {11, -1, 10};
        const auto p = WriteMinimalRopeGguf(dir, "qwen35-neg-i32.gguf", "qwen35", &neg);
        CHECK_FALSE(InspectGgufCompat(PathToUtf8(p)).needs_shim);
        GgufCompatOverlay overlay;
        CHECK(PrepareGgufCompatOverlay(PathToUtf8(p), overlay));
        CHECK_FALSE(overlay.active());
    }
    {
        const std::vector<std::uint32_t> huge = {11u, 0x80000000u, 10u};
        const auto p = WriteMinimalRopeGguf(dir, "qwen35-huge-u32.gguf", "qwen35", nullptr, &huge);
        CHECK_FALSE(InspectGgufCompat(PathToUtf8(p)).needs_shim);
    }

    // 6. Non-existent or non-GGUF file -> graceful no-op.
    CHECK_FALSE(InspectGgufCompat(PathToUtf8(dir.file("does-not-exist.gguf"))).needs_shim);
    {
        const auto junk = dir.file("not-a-gguf.bin");
        std::ofstream out(junk, std::ios::binary);
        out << "not a gguf file";
        out.close();
        CHECK_FALSE(InspectGgufCompat(PathToUtf8(junk)).needs_shim);
    }
}

TEST_CASE("gguf_compat: multi-split GGUF expands all splits and strips split keys from overlay") {
    TempDir dir;
    const std::filesystem::path split0 = dir.file("qwen35-model-00001-of-00002.gguf");
    const std::filesystem::path split1 = dir.file("qwen35-model-00002-of-00002.gguf");

    GgufPtr ctx(gguf_init_empty());
    REQUIRE(ctx != nullptr);
    gguf_set_val_str(ctx.get(), "general.architecture", "qwen35");
    const std::int32_t three[3] = {11, 11, 10};
    gguf_set_arr_data(ctx.get(), "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, three, 3);
    gguf_set_val_u16(ctx.get(), "split.no", 0);
    gguf_set_val_u16(ctx.get(), "split.count", 2);
    gguf_set_val_i32(ctx.get(), "split.tensors.count", 10);
    REQUIRE(gguf_write_to_file(ctx.get(), PathToUtf8(split0).c_str(), /*only_meta=*/true));

    GgufCompatOverlay overlay;
    std::string err;
    REQUIRE(PrepareGgufCompatOverlay(PathToUtf8(split0), overlay, &err));
    REQUIRE(overlay.active());
    REQUIRE(overlay.split_paths().size() == 3u);
    CHECK(overlay.split_paths()[0] == PathToUtf8(overlay.temp_meta_path()));
    CHECK(overlay.split_paths()[1] == PathToUtf8(split0));
    CHECK(overlay.split_paths()[2] == PathToUtf8(split1));

    gguf_init_params gp{/*no_alloc=*/true, /*ctx=*/nullptr};
    GgufPtr patched(gguf_init_from_file(PathToUtf8(overlay.temp_meta_path()).c_str(), gp));
    REQUIRE(patched != nullptr);
    CHECK(gguf_find_key(patched.get(), "split.no") < 0);
    CHECK(gguf_find_key(patched.get(), "split.count") < 0);
    CHECK(gguf_find_key(patched.get(), "split.tensors.count") < 0);

    // split.count == UINT16_MAX leaves no room for the prepended overlay (1 + 65535 > UINT16_MAX);
    // must fail closed before formatting split names or creating a temp file.
    {
        const std::filesystem::path max_split = dir.file("qwen35-max-00001-of-65535.gguf");
        GgufPtr max_ctx(gguf_init_empty());
        REQUIRE(max_ctx != nullptr);
        gguf_set_val_str(max_ctx.get(), "general.architecture", "qwen35");
        gguf_set_arr_data(max_ctx.get(), "qwen35.rope.dimension_sections", GGUF_TYPE_INT32, three, 3);
        gguf_set_val_u16(max_ctx.get(), "split.no", 0);
        gguf_set_val_u16(max_ctx.get(), "split.count", 65535u);
        REQUIRE(gguf_write_to_file(max_ctx.get(), PathToUtf8(max_split).c_str(), /*only_meta=*/true));

        GgufCompatOverlay max_overlay;
        std::string max_err;
        CHECK_FALSE(PrepareGgufCompatOverlay(PathToUtf8(max_split), max_overlay, &max_err));
        CHECK_FALSE(max_overlay.active());
        CHECK_FALSE(max_err.empty());
    }
}

TEST_CASE("gguf_compat: end-to-end llama.cpp v1.0.0 full load succeeds on 3-element qwen35 GGUF") {
    TempDir dir;
    const std::filesystem::path bad_path =
        WriteLoadableQwen35Gguf(dir, "qwen35-ollama-3elem.gguf", {11, 11, 10});
    const std::vector<std::uint8_t> bytes_before = ReadAllBytes(bad_path);

    // Baseline proof: raw llama_model_load_from_file (vocab_only = false) rejects
    // the 3-element rope.dimension_sections array in llama.cpp v1.0.0.
    {
        llama_model_params raw_mp = llama_model_default_params();
        raw_mp.vocab_only = false;
        raw_mp.n_gpu_layers = 0;
        llama_model* raw_model = llama_model_load_from_file(PathToUtf8(bad_path).c_str(), raw_mp);
        CHECK(raw_model == nullptr);
        if (raw_model != nullptr) llama_model_free(raw_model);
    }

    // With the compat shim wired into LlamaCppBackend::Load, full model + context load
    // succeeds for both mmap and direct-read modes, without modifying the file on disk.
    for (const bool use_mmap : {true, false}) {
        CAPTURE(use_mmap);
        LlamaCppBackend be;
        LoadOptions opts;
        opts.model_path = PathToUtf8(bad_path);
        opts.vocab_only = false;
        opts.use_mmap = use_mmap;
        opts.n_ctx = 64;
        opts.n_batch = 32;
        opts.n_gpu_layers = 0;
        const Status st = be.Load(opts);
        INFO("load error: " << st.message);
        REQUIRE(st.ok());
        CHECK(be.IsLoaded());
        CHECK(be.GetModelInfo().n_vocab == 4);
        CHECK(be.GetModelInfo().n_ctx >= 64u);

        // Execute a real 1-token forward pass to prove the normalized 4-element M-RoPE
        // sections and mapped weights from the untouched original GGUF run cleanly.
        GenerateRequest req;
        req.prompt = {1, 2};
        req.max_new_tokens = 1;
        req.sampling.temperature = 0.0F;
        const GenerateResult gen = be.Generate(req, [](const TokenEvent&) { return true; });
        INFO("generate error: " << gen.status.message);
        CHECK(gen.status.ok());
        CHECK(gen.prompt_tokens == 2);

        be.Unload();
        CHECK_FALSE(be.IsLoaded());
    }

    // Original GGUF on disk is byte-for-byte unchanged.
    CHECK(ReadAllBytes(bad_path) == bytes_before);
}

TEST_CASE("VIS-03: default TelemetryOptions keeps native llamacpp probe off to preserve backend sampling") {
    namespace si = sonder::inference;
    TempDir dir;
    const std::filesystem::path gguf_path =
        WriteLoadableQwen35Gguf(dir, "tiny-vis03-default-off.gguf", {11, 11, 10, 0});

    auto sink = std::make_shared<si::MemoryTelemetrySink>();
    si::EngineOptions eo;
    // Default TelemetryOptions: TelemetryLevel::standard, no opt-in native probe.
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    eo.sample_gpu_devices = false;
    eo.device_sample_interval = std::chrono::milliseconds(0);
    eo.scheduling.enabled = false;
    si::Engine engine(std::move(eo));
    si::LlamaCppBackendOptions bo;
    bo.context_length = 64;
    bo.batch_size = 32;
    bo.gpu_layers = 0;
    REQUIRE(engine.register_backend(si::make_llamacpp_backend(bo)).ok());
    auto model = engine.load_model(si::kLlamaCppBackendName, si::ModelLoadOptions{PathToUtf8(gguf_path), "cpu:0"});
    REQUIRE_MESSAGE(model.ok(), model.status().to_string());
    auto sess = engine.create_session(model.value(), {});
    REQUIRE(sess.ok());
    si::SamplingConfig sc = si::SamplingConfig::greedy(3);
    sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0f}, {2, 1.0f}, {3, 0.0f}};
    auto res = sess.value()->generate("a", {}, sc);
    REQUIRE_MESSAGE(res.ok(), res.status().to_string());
    engine.telemetry().flush();

    std::size_t token_events = 0;
    std::size_t candidate_events = 0;
    for (const auto& line : sink->lines()) {
        auto parsed = si::json::parse(line);
        REQUIRE(parsed.ok());
        const auto type = parsed->find("event_type")->as_string();
        if (type == "inference.token.generated") {
            ++token_events;
            const auto& attrs = *parsed->find("attributes");
            CHECK(attrs.find("unit")->as_string() == "chunk");
            CHECK(attrs.find("probability") == nullptr);
        } else if (type == "inference.sampling.candidates") {
            ++candidate_events;
        }
    }
    CHECK(token_events == 3u);
    CHECK(candidate_events == 0u);

    // Direct LlamaCppBackend check: telemetry_top_k == 0 keeps backend sampling active
    // (llama_set_sampler), while telemetry_top_k > 0 disables it and honors backend-side
    // stride and cap so skipped/capped tokens do not build candidate lists.
    LlamaCppBackend be;
    LoadOptions lo;
    lo.model_path = PathToUtf8(gguf_path);
    lo.n_ctx = 64;
    lo.n_batch = 32;
    lo.n_gpu_layers = 0;
    lo.n_slots = 1;
    REQUIRE_MESSAGE(be.Load(lo).ok(), "direct load failed");
    {
        GenerateRequest req;
        req.prompt = {1};
        req.max_new_tokens = 3;
        req.sampling.temperature = 0.0F;
        req.sampling.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0F}, {2, 1.0F}};
        req.sampling.telemetry_top_k = 0;
        const GenerateResult r = be.Generate(req, {});
        REQUIRE_MESSAGE(r.status.ok(), r.status.message);
        CHECK(r.generated_tokens == 3);
#if defined(SONDER_LLAMACPP_HAS_SET_SAMPLER)
        CHECK(r.backend_sampling_active);
#endif
    }
    {
        GenerateRequest req;
        req.prompt = {1};
        req.max_new_tokens = 4;
        req.sampling.temperature = 1.0F;
        req.sampling.top_k = 2;
        req.sampling.top_p = 1.0F;
        req.sampling.min_p = 0.0F;
        req.sampling.repeat_penalty = 1.0F;
        req.sampling.seed = 7;
        req.sampling.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0F}, {2, 1.0F}, {3, 0.0F}};
        req.sampling.telemetry_top_k = 4;
        req.sampling.telemetry_candidates_stride = 2;
        req.sampling.telemetry_candidates_max_events = 1;
        std::vector<std::size_t> cand_counts;
        std::vector<bool> has_probs;
        const GenerateResult r = be.Generate(req, [&](const TokenEvent& ev) {
            if (ev.token >= 0) {
                has_probs.push_back(ev.has_sampled_probability);
                cand_counts.push_back(ev.sampled_candidates.size());
            }
            return true;
        });
        REQUIRE_MESSAGE(r.status.ok(), r.status.message);
        CHECK_FALSE(r.backend_sampling_active);
        REQUIRE(has_probs.size() == 4u);
        for (bool hp : has_probs) CHECK(hp);
        // Token 0 collected candidates; tokens 1..3 skipped candidates at backend (cap=1, stride=2).
        CHECK(cand_counts[0] == 2u);
        CHECK(cand_counts[1] == 0u);
        CHECK(cand_counts[2] == 0u);
        CHECK(cand_counts[3] == 0u);
    }
    be.Unload();
}

TEST_CASE("VIS-03: native llamacpp emits token probabilities and inference.sampling.candidates") {
    namespace si = sonder::inference;
    TempDir dir;
    // Valid 4-element M-RoPE sections so llama.cpp loads directly without compat overlay.
    // All weights are zero -> raw logits at every step are [0, 0, 0, 0] for vocab {"<|endoftext|>", "a", "b", "ab"}.
    const std::filesystem::path gguf_path =
        WriteLoadableQwen35Gguf(dir, "tiny-vis03.gguf", {11, 11, 10, 0});

    auto run_native = [&](si::TelemetryOptions telemetry, const si::SamplingConfig& sampling,
                          std::uint32_t slots = 1u, std::function<void(const si::TokenChunk&)> on_chunk = {}) {
        auto sink = std::make_shared<si::MemoryTelemetrySink>();
        si::EngineOptions eo;
        eo.telemetry = telemetry;
        eo.telemetry_sinks.push_back(sink);
        eo.sample_devices_on_start = false;
        eo.sample_gpu_devices = false;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        eo.scheduling.enabled = false;
        si::Engine engine(std::move(eo));
        si::LlamaCppBackendOptions bo;
        bo.context_length = 64;
        bo.batch_size = 32;
        bo.gpu_layers = 0;
        bo.slots = slots;
        REQUIRE(engine.register_backend(si::make_llamacpp_backend(bo)).ok());
        auto model = engine.load_model(si::kLlamaCppBackendName, si::ModelLoadOptions{PathToUtf8(gguf_path), "cpu:0"});
        REQUIRE_MESSAGE(model.ok(), model.status().to_string());
        auto sess = engine.create_session(model.value(), {});
        REQUIRE(sess.ok());
        auto res = sess.value()->generate(
            "a",
            [&](const si::TokenChunk& c) {
                if (on_chunk) on_chunk(c);
                return true;
            },
            sampling);
        REQUIRE_MESSAGE(res.ok(), res.status().to_string());
        engine.telemetry().flush();
        std::vector<si::json::Value> events;
        for (const auto& line : sink->lines()) {
            auto parsed = si::json::parse(line);
            REQUIRE(parsed.ok());
            events.push_back(std::move(parsed).value());
        }
        return std::pair{std::move(res).value(), std::move(events)};
    };

    auto filter_events = [](const std::vector<si::json::Value>& all, std::string_view type) {
        std::vector<si::json::Value> out;
        for (const auto& e : all) {
            if (e.find("event_type")->as_string() == type) out.push_back(e);
        }
        return out;
    };

    // Ban EOS (0) and bias tokens 1, 2, 3 to 2.0, 1.0, 0.0 -> analytical post-sampling softmax.
    const double z_post = std::exp(2.0) + std::exp(1.0) + std::exp(0.0);
    const double p1 = std::exp(2.0) / z_post;
    const double p2 = std::exp(1.0) / z_post;
    const double p3 = std::exp(0.0) / z_post;

    SUBCASE("opt-in native_token_probabilities emits post_sampling probabilities and candidates without leaking logprobs") {
        si::TelemetryOptions t;
        t.native_token_probabilities = true;
        si::SamplingConfig sc = si::SamplingConfig::greedy(3);
        sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0f}, {2, 1.0f}, {3, 0.0f}};
        unsigned caller_lp_chunks = 0;
        auto [res, events] = run_native(t, sc, 1u, [&](const si::TokenChunk& c) {
            if (c.logprob.has_value() || !c.top_logprobs.empty()) ++caller_lp_chunks;
        });
        CHECK(res.text == "aaa");
        CHECK(res.logprobs.empty());
        CHECK(caller_lp_chunks == 0u);

        const auto tokens = filter_events(events, "inference.token.generated");
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(tokens.size() == 3u);
        REQUIRE(cands.size() == 3u);
        for (std::size_t i = 0; i < 3u; ++i) {
            const auto& ta = *tokens[i].find("attributes");
            const auto& ca = *cands[i].find("attributes");
            CHECK(tokens[i].find("sampling")->find("level")->as_string() == "standard");
            CHECK(cands[i].find("sampling")->find("level")->as_string() == "standard");
            CHECK(ta.find("unit")->as_string() == "token");
            CHECK(ta.find("count")->as_int() == 1);
            CHECK(ta.find("token_id")->as_int() == 1);
            CHECK(ta.find("probability")->as_double() == doctest::Approx(p1).epsilon(1e-4));
            CHECK(ta.find("probability_basis")->as_string() == "post_sampling");
            CHECK(ta.find("text") == nullptr);

            CHECK(ca.find("index")->as_int() == ta.find("index")->as_int());
            CHECK(ca.find("sampled_token_id")->as_int() == 1);
            CHECK(ca.find("probability_basis")->as_string() == "post_sampling");
            const auto& list = ca.find("candidates")->as_array();
            REQUIRE(list.size() == 3u);  // token 0 banned (-inf) -> excluded
            CHECK(list[0].find("token_id")->as_int() == 1);
            CHECK(list[0].find("probability")->as_double() == doctest::Approx(p1).epsilon(1e-4));
            CHECK(list[1].find("token_id")->as_int() == 2);
            CHECK(list[1].find("probability")->as_double() == doctest::Approx(p2).epsilon(1e-4));
            CHECK(list[2].find("token_id")->as_int() == 3);
            CHECK(list[2].find("probability")->as_double() == doctest::Approx(p3).epsilon(1e-4));
            for (const auto& entry : list) {
                CHECK(entry.find("text") == nullptr);
            }
        }
    }

    SUBCASE("TelemetryLevel::deep enables native llamacpp probe automatically") {
        si::TelemetryOptions t;
        t.level = si::TelemetryLevel::deep;
        si::SamplingConfig sc = si::SamplingConfig::greedy(2);
        sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0f}, {2, 1.0f}};
        auto [res, events] = run_native(t, sc, 1u);
        const auto tokens = filter_events(events, "inference.token.generated");
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(tokens.size() == 2u);
        REQUIRE(cands.size() == 2u);
        CHECK(tokens[0].find("attributes")->find("unit")->as_string() == "token");
        CHECK(cands[0].find("attributes")->find("probability_basis")->as_string() == "post_sampling");
    }

    SUBCASE("stochastic sampling with top_k, stride, cap, and capture_text opt-in") {
        si::TelemetryOptions t;
        t.native_token_probabilities = true;
        t.token_probability_top_k = 4;
        t.token_candidates_stride = 2;
        t.token_candidates_max_events = 1;
        t.capture_text = true;
        si::SamplingConfig sc;
        sc.max_tokens = 4;
        sc.temperature = 1.0f;
        sc.top_k = 2;
        sc.top_p = 1.0f;
        sc.min_p = 0.0f;
        sc.repeat_penalty = 1.0f;
        sc.seed = 7;
        sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0f}, {2, 1.0f}, {3, 0.0f}};
        auto [res, events] = run_native(t, sc, 1u);
        const auto tokens = filter_events(events, "inference.token.generated");
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(tokens.size() == 4u);
        REQUIRE(cands.size() == 1u);
        const auto& ca = *cands[0].find("attributes");
        CHECK(ca.find("index")->as_int() == 0);
        CHECK(ca.find("probability_basis")->as_string() == "post_sampling");
        const auto& list = ca.find("candidates")->as_array();
        REQUIRE(list.size() == 2u);
        CHECK(list[0].find("token_id")->as_int() == 1);
        CHECK(list[1].find("token_id")->as_int() == 2);
        REQUIRE(list[0].find("text") != nullptr);
        CHECK(list[0].find("text")->as_string() == "a");
        REQUIRE(list[1].find("text") != nullptr);
        CHECK(list[1].find("text")->as_string() == "b");
    }

    SUBCASE("caller-requested logprobs feed pre_sampling telemetry without altering caller top_logprobs") {
        si::TelemetryOptions t;
        t.token_probability_top_k = 4;
        si::SamplingConfig sc = si::SamplingConfig::greedy(2);
        sc.logprobs = true;
        sc.top_logprobs = 2;
        sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0f}, {2, 1.0f}, {3, 0.0f}};
        auto [res, events] = run_native(t, sc, 1u);
        REQUIRE(res.logprobs.size() == 2u);
        CHECK(res.logprobs[0].top_logprobs.size() == 2u);

        const auto tokens = filter_events(events, "inference.token.generated");
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(tokens.size() == 2u);
        REQUIRE(cands.size() == 2u);
        // Raw logits are [0,0,0,0] -> pre-sampling softmax is 0.25 for all 4 tokens.
        CHECK(tokens[0].find("attributes")->find("probability_basis")->as_string() == "pre_sampling");
        CHECK(tokens[0].find("attributes")->find("probability")->as_double() == doctest::Approx(0.25).epsilon(1e-4));
        CHECK(cands[0].find("attributes")->find("probability_basis")->as_string() == "pre_sampling");
        CHECK(cands[0].find("attributes")->find("candidates")->as_array().size() == 4u);
    }

    SUBCASE("token_probability_top_k 0 and metrics level disable candidates") {
        si::SamplingConfig sc = si::SamplingConfig::greedy(2);
        sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0f}};
        {
            si::TelemetryOptions t;
            t.native_token_probabilities = true;
            t.token_probability_top_k = 0;
            auto [res, events] = run_native(t, sc, 1u);
            const auto tokens = filter_events(events, "inference.token.generated");
            REQUIRE(tokens.size() == 2u);
            for (const auto& ev : tokens) {
                CHECK(ev.find("attributes")->find("unit")->as_string() == "chunk");
                CHECK(ev.find("attributes")->find("probability") == nullptr);
            }
            CHECK(filter_events(events, "inference.sampling.candidates").empty());
        }
        {
            si::TelemetryOptions t;
            t.native_token_probabilities = true;
            t.level = si::TelemetryLevel::metrics;
            auto [res, events] = run_native(t, sc, 1u);
            CHECK(filter_events(events, "inference.token.generated").empty());
            CHECK(filter_events(events, "inference.sampling.candidates").empty());
        }
    }

    SUBCASE("multi-slot batch worker emits post_sampling candidates when opted in") {
        si::TelemetryOptions t;
        t.native_token_probabilities = true;
        si::SamplingConfig sc = si::SamplingConfig::greedy(2);
        sc.logit_bias = {{0, -std::numeric_limits<float>::infinity()}, {1, 2.0f}, {2, 1.0f}, {3, 0.0f}};
        auto [res, events] = run_native(t, sc, /*slots=*/2u);
        const auto cands = filter_events(events, "inference.sampling.candidates");
        REQUIRE(cands.size() == 2u);
        CHECK(cands[0].find("attributes")->find("probability_basis")->as_string() == "post_sampling");
        CHECK(cands[0].find("attributes")->find("candidates")->as_array().size() == 3u);
    }

    SUBCASE("speculative TargetMatch path emits post_sampling probabilities and respects stride/cap") {
        // Repeated token-1 prompt gives ngram matcher an immediate match so TargetMatch
        // verification runs on real speculative draft batches (single-slot and multi-slot).
        for (std::uint32_t n_slots : {1u, 2u}) {
            LlamaCppBackend be;
            LoadOptions lo;
            lo.model_path = PathToUtf8(gguf_path);
            lo.n_ctx = 64;
            lo.n_batch = 32;
            lo.n_gpu_layers = 0;
            lo.n_slots = n_slots;
            lo.spec_type = "ngram";
            lo.spec_draft_max = 4;
            REQUIRE_MESSAGE(be.Load(lo).ok(), "speculative load failed");

            GenerateRequest req;
            req.prompt = {1, 1, 1, 1, 1, 1, 1, 1};
            req.max_new_tokens = 6;
            req.sampling.temperature = 0.0F;
            req.sampling.logit_bias = {
                {0, -std::numeric_limits<float>::infinity()}, {1, 2.0F}, {2, 1.0F}, {3, 0.0F}};
            req.sampling.telemetry_top_k = 3;
            req.sampling.telemetry_candidates_stride = 2;
            req.sampling.telemetry_candidates_max_events = 2;

            std::vector<TokenEvent> events;
            const GenerateResult r = be.Generate(req, [&](const TokenEvent& ev) {
                if (ev.token >= 0) events.push_back(ev);
                return true;
            });
            REQUIRE_MESSAGE(r.status.ok(), r.status.message);
            CHECK(r.draft_tokens > 0);
            CHECK(r.draft_accepted_tokens > 0);
            REQUIRE(events.size() == 6u);
            std::size_t non_empty_cands = 0;
            for (const TokenEvent& ev : events) {
                CHECK(ev.token == 1);
                CHECK(ev.has_sampled_probability);
                CHECK(ev.sampled_probability_post_sampling);
                CHECK(ev.sampled_probability == doctest::Approx(p1).epsilon(1e-4));
                if (!ev.sampled_candidates.empty()) {
                    ++non_empty_cands;
                    REQUIRE(ev.sampled_candidates.size() == 3u);
                    CHECK(ev.sampled_candidates[0].first == 1);
                    CHECK(ev.sampled_candidates[0].second == doctest::Approx(p1).epsilon(1e-4));
                    CHECK(ev.sampled_candidates[1].first == 2);
                    CHECK(ev.sampled_candidates[2].first == 3);
                }
            }
            // Stride 2 over 6 tokens -> ordinals 0, 2, 4 eligible, capped at 2 events (ordinals 0 and 2).
            CHECK(non_empty_cands == 2u);
            CHECK(events[0].sampled_candidates.size() == 3u);
            CHECK(events[1].sampled_candidates.empty());
            CHECK(events[2].sampled_candidates.size() == 3u);
            CHECK(events[3].sampled_candidates.empty());
            CHECK(events[4].sampled_candidates.empty());
            CHECK(events[5].sampled_candidates.empty());
            be.Unload();
        }
    }
}
