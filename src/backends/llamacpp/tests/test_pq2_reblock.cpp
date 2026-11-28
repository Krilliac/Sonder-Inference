// Unit & end-to-end tests for RET-09 lossless PQ2_0 (group 128, type 142) ->
// stock Q2_0 (group 64, type 42) re-blocking, verification, hostile-input
// rejection, and in-process llama.cpp v1.0.0 execution with prism.hadamard FWHT.
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "ggml.h"
#include "pq2_reblock.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

namespace {

std::string PathToUtf8(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

class TempDir {
public:
    TempDir() {
        static std::atomic<std::uint64_t> seq{0};
        const auto ticks = static_cast<std::uint64_t>(
            std::chrono::steady_clock::now().time_since_epoch().count());
        dir_ = std::filesystem::temp_directory_path() /
               ("sonder-test-pq2-reblock-" + std::to_string(ticks) + "-" +
                std::to_string(seq.fetch_add(1, std::memory_order_relaxed)));
        std::filesystem::create_directories(dir_);
    }

    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    [[nodiscard]] const std::filesystem::path& path() const noexcept { return dir_; }
    [[nodiscard]] std::filesystem::path file(const std::string& name) const { return dir_ / name; }

private:
    std::filesystem::path dir_;
};

void WriteBytes(const std::filesystem::path& path, const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    REQUIRE(out.good());
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    REQUIRE(out.good());
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.good());
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>{});
}

// Raw GGUF v3 binary writer capable of emitting private tensor types (e.g. 142 PQ2_0,
// 143 PTQ1_0, 144 PQ1_0, 145 PQ2_0M) that stock ggml's gguf_add_tensor rejects.
class RawGgufBuilder {
public:
    void kv_u32(std::string_view key, std::uint32_t val) {
        begin_kv(key, 4);  // GGUF_TYPE_UINT32
        put_u32(kvs_, val);
    }

    void kv_f32(std::string_view key, float val) {
        begin_kv(key, 6);  // GGUF_TYPE_FLOAT32
        std::uint32_t bits = 0;
        std::memcpy(&bits, &val, sizeof(bits));
        put_u32(kvs_, bits);
    }

    void kv_bool(std::string_view key, bool val) {
        begin_kv(key, 7);  // GGUF_TYPE_BOOL
        kvs_.push_back(val ? 1u : 0u);
    }

    void kv_str(std::string_view key, std::string_view val) {
        begin_kv(key, 8);  // GGUF_TYPE_STRING
        put_str(kvs_, val);
    }

    void kv_i32_arr(std::string_view key, std::span<const std::int32_t> vals) {
        begin_kv(key, 9);   // GGUF_TYPE_ARRAY
        put_u32(kvs_, 5);   // GGUF_TYPE_INT32
        put_u64(kvs_, vals.size());
        for (const std::int32_t v : vals) {
            put_u32(kvs_, static_cast<std::uint32_t>(v));
        }
    }

    void kv_str_arr(std::string_view key, std::span<const std::string> vals) {
        begin_kv(key, 9);   // GGUF_TYPE_ARRAY
        put_u32(kvs_, 8);   // GGUF_TYPE_STRING
        put_u64(kvs_, vals.size());
        for (const std::string& s : vals) {
            put_str(kvs_, s);
        }
    }

    void add_tensor(std::string name, std::vector<std::uint64_t> dims, std::uint32_t type,
                    std::vector<std::uint8_t> data, std::optional<std::uint64_t> forced_offset = std::nullopt) {
        tensors_.push_back(TensorSpec{std::move(name), std::move(dims), type, std::move(data), forced_offset});
    }

    [[nodiscard]] std::vector<std::uint8_t> build(std::size_t alignment = 32) const {
        std::vector<std::uint8_t> out;
        out.insert(out.end(), {'G', 'G', 'U', 'F'});
        put_u32(out, 3);  // version 3
        put_u64(out, tensors_.size());
        put_u64(out, kv_count_);
        out.insert(out.end(), kvs_.begin(), kvs_.end());

        std::uint64_t running_offset = 0;
        std::vector<std::uint64_t> offsets;
        offsets.reserve(tensors_.size());
        for (const auto& t : tensors_) {
            const std::uint64_t aligned = align_up(running_offset, alignment);
            const std::uint64_t use_offset = t.forced_offset.value_or(aligned);
            offsets.push_back(use_offset);
            running_offset = aligned + t.data.size();

            put_str(out, t.name);
            put_u32(out, static_cast<std::uint32_t>(t.dims.size()));
            for (const std::uint64_t d : t.dims) {
                put_u64(out, d);
            }
            put_u32(out, t.type);
            put_u64(out, use_offset);
        }

        const std::size_t data_start = align_up(out.size(), alignment);
        out.resize(data_start, 0);
        for (std::size_t i = 0; i < tensors_.size(); ++i) {
            const std::size_t target = data_start + static_cast<std::size_t>(align_up(
                i == 0 ? 0u : (offsets[i]), alignment));
            if (!tensors_[i].forced_offset.has_value()) {
                if (out.size() < target) out.resize(target, 0);
            }
            out.insert(out.end(), tensors_[i].data.begin(), tensors_[i].data.end());
        }
        return out;
    }

private:
    struct TensorSpec {
        std::string name;
        std::vector<std::uint64_t> dims;
        std::uint32_t type = 0;
        std::vector<std::uint8_t> data;
        std::optional<std::uint64_t> forced_offset;
    };

    static std::uint64_t align_up(std::uint64_t v, std::size_t a) {
        return (v + a - 1) & ~(static_cast<std::uint64_t>(a) - 1);
    }

    static void put_u32(std::vector<std::uint8_t>& b, std::uint32_t v) {
        for (int s = 0; s < 32; s += 8) {
            b.push_back(static_cast<std::uint8_t>((v >> s) & 0xFFu));
        }
    }

    static void put_u64(std::vector<std::uint8_t>& b, std::uint64_t v) {
        for (int s = 0; s < 64; s += 8) {
            b.push_back(static_cast<std::uint8_t>((v >> s) & 0xFFu));
        }
    }

    static void put_str(std::vector<std::uint8_t>& b, std::string_view s) {
        put_u64(b, s.size());
        b.insert(b.end(), s.begin(), s.end());
    }

    void begin_kv(std::string_view key, std::uint32_t val_type) {
        put_str(kvs_, key);
        put_u32(kvs_, val_type);
        ++kv_count_;
    }

    std::uint64_t kv_count_ = 0;
    std::vector<std::uint8_t> kvs_;
    std::vector<TensorSpec> tensors_;
};

std::vector<std::uint8_t> MakeF32Bytes(std::size_t count, float value) {
    std::vector<std::uint8_t> out(count * sizeof(float));
    for (std::size_t i = 0; i < count; ++i) {
        std::memcpy(out.data() + i * sizeof(float), &value, sizeof(float));
    }
    return out;
}

std::vector<std::uint8_t> MakePq2TensorBytes(std::size_t rows, std::size_t cols, std::uint16_t fp16_scale,
                                             std::uint8_t seed_byte) {
    REQUIRE(cols % kPq2BlockElements == 0);
    const std::size_t blocks = rows * (cols / kPq2BlockElements);
    std::vector<std::uint8_t> out(blocks * kPq2BlockBytes);
    for (std::size_t b = 0; b < blocks; ++b) {
        std::uint8_t* dst = out.data() + b * kPq2BlockBytes;
        dst[0] = static_cast<std::uint8_t>(fp16_scale & 0xFFu);
        dst[1] = static_cast<std::uint8_t>((fp16_scale >> 8) & 0xFFu);
        for (std::size_t j = 0; j < 32; ++j) {
            dst[2 + j] = static_cast<std::uint8_t>((seed_byte + b * 13u + j * 29u) & 0xFFu);
        }
    }
    return out;
}

// Builds a synthetic 1-layer Llama GGUF whose linear projections are stored as
// private PQ2_0 (type 142, group 128) with valid `prism.hadamard.*` v1 metadata
// (normalized-sylvester-walsh-hadamard, block_size=128, explicit +/-1 signs).
std::vector<std::uint8_t> BuildSyntheticBonsaiPq2Gguf() {
    constexpr std::uint64_t kEmbd = 128;
    constexpr std::uint64_t kFf = 128;
    constexpr std::uint64_t kVocab = 8;

    RawGgufBuilder g;
    g.kv_str("general.architecture", "llama");
    g.kv_str("general.name", "synthetic-bonsai-pq2");
    g.kv_u32("general.file_type", kPrismFtypeMostlyPq2_0);
    g.kv_u32("llama.context_length", 128);
    g.kv_u32("llama.embedding_length", static_cast<std::uint32_t>(kEmbd));
    g.kv_u32("llama.block_count", 1);
    g.kv_u32("llama.feed_forward_length", static_cast<std::uint32_t>(kFf));
    g.kv_u32("llama.attention.head_count", 2);
    g.kv_u32("llama.attention.head_count_kv", 2);
    g.kv_f32("llama.attention.layer_norm_rms_epsilon", 1e-5f);
    g.kv_u32("llama.rope.dimension_count", 64);

    // Tokenizer metadata (gpt2 BPE, 8 tokens).
    g.kv_str("tokenizer.ggml.model", "gpt2");
    const std::vector<std::string> tokens = {"<eos>", "a", "b", "c", "d", "e", "ab", "cd"};
    g.kv_str_arr("tokenizer.ggml.tokens", tokens);
    const std::vector<std::int32_t> token_types = {3, 1, 1, 1, 1, 1, 1, 1};
    g.kv_i32_arr("tokenizer.ggml.token_type", token_types);
    const std::vector<std::string> merges = {"a b", "c d"};
    g.kv_str_arr("tokenizer.ggml.merges", merges);
    g.kv_u32("tokenizer.ggml.eos_token_id", 0);

    // prism.hadamard v1 metadata matching llama.cpp v1.0.0 load_hparams_hadamard.
    g.kv_u32("prism.hadamard.version", 1);
    g.kv_bool("prism.hadamard.tied_output", false);
    g.kv_u32("prism.hadamard.block_size", 128);
    g.kv_str("prism.hadamard.transform", "normalized-sylvester-walsh-hadamard");
    g.kv_str("prism.hadamard.axis", "input-last-dimension");
    g.kv_str("prism.hadamard.sign_mode", "explicit");
    const std::vector<std::int32_t> sign_widths = {128};
    g.kv_i32_arr("prism.hadamard.sign_widths", sign_widths);
    std::vector<std::int32_t> sign_values(128);
    for (std::size_t i = 0; i < sign_values.size(); ++i) {
        sign_values[i] = (i % 3 == 0) ? -1 : 1;
    }
    g.kv_i32_arr("prism.hadamard.sign_values", sign_values);
    const std::vector<std::string> folded_weights = {
        "blk.0.attn_q.weight",     "blk.0.attn_k.weight",   "blk.0.attn_v.weight",
        "blk.0.attn_output.weight","blk.0.ffn_gate.weight", "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",   "output.weight",
    };
    g.kv_str_arr("prism.hadamard.weight_names", folded_weights);

    // Embeddings & norms as F32; all linear projections as PQ2_0 (type 142).
    // 0x2E66 is ~0.0625 in IEEE-754 FP16.
    constexpr std::uint16_t kScale = 0x2C00u;  // 0.0625f exact in FP16
    g.add_tensor("token_embd.weight", {kEmbd, kVocab}, 0 /*F32*/, MakeF32Bytes(kEmbd * kVocab, 0.05f));
    g.add_tensor("output_norm.weight", {kEmbd}, 0 /*F32*/, MakeF32Bytes(kEmbd, 1.0f));
    g.add_tensor("output.weight", {kEmbd, kVocab}, kGgmlTypePq2_0, MakePq2TensorBytes(kVocab, kEmbd, kScale, 0x91u));

    g.add_tensor("blk.0.attn_norm.weight", {kEmbd}, 0 /*F32*/, MakeF32Bytes(kEmbd, 1.0f));
    g.add_tensor("blk.0.attn_q.weight", {kEmbd, kEmbd}, kGgmlTypePq2_0, MakePq2TensorBytes(kEmbd, kEmbd, kScale, 0x11u));
    g.add_tensor("blk.0.attn_k.weight", {kEmbd, kEmbd}, kGgmlTypePq2_0, MakePq2TensorBytes(kEmbd, kEmbd, kScale, 0x22u));
    g.add_tensor("blk.0.attn_v.weight", {kEmbd, kEmbd}, kGgmlTypePq2_0, MakePq2TensorBytes(kEmbd, kEmbd, kScale, 0x33u));
    g.add_tensor("blk.0.attn_output.weight", {kEmbd, kEmbd}, kGgmlTypePq2_0,
                 MakePq2TensorBytes(kEmbd, kEmbd, kScale, 0x44u));
    g.add_tensor("blk.0.ffn_norm.weight", {kEmbd}, 0 /*F32*/, MakeF32Bytes(kEmbd, 1.0f));
    g.add_tensor("blk.0.ffn_gate.weight", {kEmbd, kFf}, kGgmlTypePq2_0, MakePq2TensorBytes(kFf, kEmbd, kScale, 0x55u));
    g.add_tensor("blk.0.ffn_up.weight", {kEmbd, kFf}, kGgmlTypePq2_0, MakePq2TensorBytes(kFf, kEmbd, kScale, 0x66u));
    g.add_tensor("blk.0.ffn_down.weight", {kFf, kEmbd}, kGgmlTypePq2_0, MakePq2TensorBytes(kEmbd, kFf, kScale, 0x77u));
    return g.build();
}

}  // namespace

TEST_CASE("pq2_reblock: row re-blocking is bit-for-bit lossless against ggml Q2_0 dequantization") {
    const auto* q2_traits = ggml_get_type_traits(GGML_TYPE_Q2_0);
    REQUIRE(q2_traits != nullptr);
    REQUIRE(q2_traits->to_float != nullptr);
    CHECK(q2_traits->blck_size == 64);
    CHECK(q2_traits->type_size == 18u);

    // Cover positive, negative, zero, and subnormal FP16 scales across 4 PQ2_0 blocks (512 elements).
    const std::array<std::uint16_t, 4> scales = {
        0x3C00u,  // +1.0
        0xBA00u,  // -0.75
        0x0000u,  // +0.0
        0x0001u,  // smallest positive subnormal FP16
    };
    constexpr std::size_t kElements = scales.size() * kPq2BlockElements;
    std::vector<std::uint8_t> pq2(scales.size() * kPq2BlockBytes, 0);
    for (std::size_t b = 0; b < scales.size(); ++b) {
        std::uint8_t* blk = pq2.data() + b * kPq2BlockBytes;
        blk[0] = static_cast<std::uint8_t>(scales[b] & 0xFFu);
        blk[1] = static_cast<std::uint8_t>((scales[b] >> 8) & 0xFFu);
        for (std::size_t j = 0; j < 32; ++j) {
            // Exercise every 2-bit code {0, 1, 2, 3} -> {-1, 0, +1, +2}.
            blk[2 + j] = static_cast<std::uint8_t>(((b * 37u + j * 53u) ^ 0xE4u) & 0xFFu);
        }
    }

    std::vector<float> pq2_floats;
    REQUIRE(DequantizePq2Row(pq2, kElements, pq2_floats).ok());
    REQUIRE(pq2_floats.size() == kElements);

    std::vector<std::uint8_t> q2_bytes;
    REQUIRE(ReblockPq2RowToQ2(pq2, kElements, q2_bytes).ok());
    REQUIRE(q2_bytes.size() == (kElements / kQ2BlockElements) * kQ2BlockBytes);

    std::vector<float> ggml_q2_floats(kElements, -999.0f);
    q2_traits->to_float(q2_bytes.data(), ggml_q2_floats.data(), static_cast<std::int64_t>(kElements));

    for (std::size_t i = 0; i < kElements; ++i) {
        CAPTURE(i);
        CHECK(std::memcmp(&pq2_floats[i], &ggml_q2_floats[i], sizeof(float)) == 0);
    }

    // Spot-check exact 2-bit codec semantics on block 0 (scale +1.0): 0xE4 = 0b11_10_01_00 -> -1, 0, +1, +2.
    std::vector<std::uint8_t> one_blk(kPq2BlockBytes, 0x55u);  // 0x55 = 01_01_01_01 (all zeros)
    one_blk[0] = 0x00u;
    one_blk[1] = 0x3Cu;  // +1.0
    one_blk[2] = 0xE4u;  // elements 0..3: 00->-1, 01->0, 10->+1, 11->+2
    one_blk[18] = 0x1Bu; // elements 64..67 (second Q2_0 half): 11->+2, 10->+1, 01->0, 00->-1
    std::vector<float> exact;
    REQUIRE(DequantizePq2Row(one_blk, kPq2BlockElements, exact).ok());
    CHECK(exact[0] == -1.0f);
    CHECK(exact[1] == 0.0f);
    CHECK(exact[2] == 1.0f);
    CHECK(exact[3] == 2.0f);
    CHECK(exact[64] == 2.0f);
    CHECK(exact[65] == 1.0f);
    CHECK(exact[66] == 0.0f);
    CHECK(exact[67] == -1.0f);
}

TEST_CASE("pq2_reblock: row helpers reject bad element counts, mismatched buffers, and NaN/Inf scales") {
    std::vector<float> floats;
    std::vector<std::uint8_t> q2;
    const std::vector<std::uint8_t> valid_blk(kPq2BlockBytes, 0);

    CHECK(DequantizePq2Row(valid_blk, 0, floats).code == ErrorCode::kInvalidArgument);
    CHECK(ReblockPq2RowToQ2(valid_blk, 0, q2).code == ErrorCode::kInvalidArgument);
    CHECK(DequantizePq2Row(valid_blk, 64, floats).code == ErrorCode::kInvalidArgument);
    CHECK(ReblockPq2RowToQ2(valid_blk, 192, q2).code == ErrorCode::kInvalidArgument);

    const std::vector<std::uint8_t> short_blk(kPq2BlockBytes - 1, 0);
    CHECK(DequantizePq2Row(short_blk, 128, floats).code == ErrorCode::kInvalidArgument);
    CHECK(ReblockPq2RowToQ2(short_blk, 128, q2).code == ErrorCode::kInvalidArgument);

    // +Inf (0x7C00) and quiet NaN (0x7E00) FP16 scales must fail closed.
    for (const std::uint16_t bad_scale : {std::uint16_t{0x7C00u}, std::uint16_t{0xFC00u}, std::uint16_t{0x7E00u}}) {
        CAPTURE(bad_scale);
        std::vector<std::uint8_t> nan_blk(kPq2BlockBytes, 0);
        nan_blk[0] = static_cast<std::uint8_t>(bad_scale & 0xFFu);
        nan_blk[1] = static_cast<std::uint8_t>((bad_scale >> 8) & 0xFFu);
        CHECK(DequantizePq2Row(nan_blk, 128, floats).code == ErrorCode::kInvalidArgument);
        CHECK(ReblockPq2RowToQ2(nan_blk, 128, q2).code == ErrorCode::kInvalidArgument);
    }
}

TEST_CASE("pq2_reblock: re-blocked Bonsai PQ2_0 + prism.hadamard GGUF passes parity verification and runs natively on v1.0.0") {
    TempDir tmp;
    const std::filesystem::path pq2_path = tmp.file("bonsai-pq2.gguf");
    const std::filesystem::path q2_path = tmp.file("bonsai-q2-g64.gguf");
    const std::vector<std::uint8_t> pq2_raw = BuildSyntheticBonsaiPq2Gguf();
    WriteBytes(pq2_path, pq2_raw);

    // 1. Unpatched v1.0.0 native backend cleanly refuses raw PQ2_0 (type 142) without crashing.
    {
        LlamaCppBackend raw_be;
        LoadOptions raw_opts;
        raw_opts.model_path = PathToUtf8(pq2_path);
        raw_opts.n_ctx = 64;
        raw_opts.n_batch = 32;
        const Status raw_st = raw_be.Load(raw_opts);
        CHECK_FALSE(raw_st.ok());
        CHECK_FALSE(raw_be.IsLoaded());
    }

    // 2. InspectPq2Gguf accurately sizes PQ2_0 -> Q2_0 conversion (+2 bytes per 128 elements).
    Pq2GgufInspection insp;
    const Status insp_st = InspectPq2Gguf(pq2_path, insp);
    REQUIRE_MESSAGE(insp_st.ok(), insp_st.message);
    CHECK(insp.gguf_version == 3u);
    CHECK(insp.tensor_count == 12u);
    CHECK(insp.pq2_tensor_count == 8u);
    CHECK(insp.q2_tensor_count == 0u);
    CHECK(insp.prism_hadamard);
    CHECK(insp.file_type == std::optional<std::uint32_t>(kPrismFtypeMostlyPq2_0));
    CHECK(insp.pq2_input_bytes > 0u);
    CHECK(insp.q2_output_bytes == (insp.pq2_input_bytes / kPq2BlockBytes) * (2u * kQ2BlockBytes));

    // 3. ReblockPq2ToQ2Gguf converts losslessly without touching the source file.
    Pq2ReblockReport report;
    const Status reb_st = ReblockPq2ToQ2Gguf(pq2_path, q2_path, &report);
    REQUIRE_MESSAGE(reb_st.ok(), reb_st.message);
    CHECK(ReadBytes(pq2_path) == pq2_raw);  // source untouched
    CHECK(report.reblocked_tensors == 8u);
    CHECK(report.copied_tensors == 4u);
    CHECK(report.output_file_bytes == std::filesystem::file_size(q2_path));
    CHECK(report.output_sha256.size() == 64u);

    Pq2GgufInspection out_insp;
    REQUIRE_MESSAGE(InspectPq2Gguf(q2_path, out_insp).ok(), "output GGUF must pass inspection");
    CHECK(out_insp.pq2_tensor_count == 0u);
    CHECK(out_insp.q2_tensor_count == 8u);
    CHECK(out_insp.prism_hadamard);
    CHECK(out_insp.file_type == std::optional<std::uint32_t>(kLlamaFtypeMostlyQ2_0));

    // 4. VerifyPq2ToQ2Gguf confirms bit-exact dequantized parity; tampering with 1 byte fails verification.
    const Status ver_st = VerifyPq2ToQ2Gguf(pq2_path, q2_path);
    REQUIRE_MESSAGE(ver_st.ok(), ver_st.message);

    {
        const std::filesystem::path tampered = tmp.file("tampered.gguf");
        std::vector<std::uint8_t> bytes = ReadBytes(q2_path);
        REQUIRE(bytes.size() > 64u);
        bytes.back() ^= 0x01u;
        WriteBytes(tampered, bytes);
        CHECK_FALSE(VerifyPq2ToQ2Gguf(pq2_path, tampered).ok());
    }

    // 5. Pinned llama.cpp v1.0.0 loads the re-blocked Q2_0 + prism.hadamard GGUF and decodes deterministically.
    LlamaCppBackend be;
    LoadOptions opts;
    opts.model_path = PathToUtf8(q2_path);
    opts.n_ctx = 64;
    opts.n_batch = 32;
    opts.n_threads = 2;
    const Status load_st = be.Load(opts);
    REQUIRE_MESSAGE(load_st.ok(), load_st.message);
    CHECK(be.IsLoaded());

    GenerateRequest req;
    req.prompt = {1, 2, 3};
    req.max_new_tokens = 4;
    req.sampling.temperature = 0.0f;
    req.sampling.seed = 42;
    std::vector<Token> generated;
    const GenerateResult res = be.Generate(req, [&](const TokenEvent& ev) {
        generated.push_back(ev.token);
        return true;
    });
    REQUIRE_MESSAGE(res.status.ok(), res.status.message);
    CHECK(res.prompt_tokens == 3);
    CHECK((res.generated_tokens >= 1 || res.stop_reason == StopReason::kEndOfGeneration));
    be.Unload();
}

TEST_CASE("pq2_reblock: hostile and unsupported GGUFs fail closed without partial outputs") {
    TempDir tmp;
    const std::filesystem::path out_path = tmp.file("should-not-exist.gguf");

    // 1. Missing, empty, directory, and equivalent input==output paths.
    Pq2GgufInspection insp;
    CHECK(InspectPq2Gguf("", insp).code == ErrorCode::kInvalidArgument);
    CHECK(InspectPq2Gguf(tmp.file("missing.gguf"), insp).code == ErrorCode::kFileNotFound);
    CHECK(InspectPq2Gguf(tmp.path(), insp).code == ErrorCode::kFileNotFound);

    const std::filesystem::path pq2_path = tmp.file("valid.gguf");
    WriteBytes(pq2_path, BuildSyntheticBonsaiPq2Gguf());
    CHECK(ReblockPq2ToQ2Gguf(pq2_path, pq2_path).code == ErrorCode::kInvalidArgument);
    CHECK(ReblockPq2ToQ2Gguf(pq2_path, tmp.path() / "." / "valid.gguf").code == ErrorCode::kInvalidArgument);

    // Symlink input and symlink output paths must be refused when the OS allows creating symlinks.
    {
        std::error_code sym_ec;
        const std::filesystem::path sym_in = tmp.file("sym-in.gguf");
        std::filesystem::create_symlink(pq2_path, sym_in, sym_ec);
        if (!sym_ec) {
            CHECK(InspectPq2Gguf(sym_in, insp).code == ErrorCode::kInvalidArgument);
            CHECK(ReblockPq2ToQ2Gguf(sym_in, out_path).code == ErrorCode::kInvalidArgument);
            CHECK_FALSE(std::filesystem::exists(out_path));
        }
        std::error_code sym_out_ec;
        const std::filesystem::path sym_target = tmp.file("sym-target.gguf");
        WriteBytes(sym_target, {0x00u});
        const std::filesystem::path sym_out = tmp.file("sym-out.gguf");
        std::filesystem::create_symlink(sym_target, sym_out, sym_out_ec);
        if (!sym_out_ec) {
            CHECK(ReblockPq2ToQ2Gguf(pq2_path, sym_out).code == ErrorCode::kInvalidArgument);
            CHECK(ReadBytes(sym_target) == std::vector<std::uint8_t>{0x00u});
        }
    }

    // 2. Unsupported private Prism types: PTQ1_0 (143), PQ1_0 (144), PQ2_0M (145).
    for (const std::pair<std::uint32_t, std::size_t> priv :
         {std::pair{kGgmlTypePtq1_0, std::size_t{28}}, std::pair{kGgmlTypePq1_0, std::size_t{10}},
          std::pair{kGgmlTypePq2_0m, std::size_t{36}}}) {
        CAPTURE(priv.first);
        RawGgufBuilder g;
        g.kv_str("general.architecture", "llama");
        g.add_tensor("blk.0.attn_q.weight", {128, 1}, priv.first, std::vector<std::uint8_t>(priv.second, 0));
        const std::filesystem::path p = tmp.file("priv-" + std::to_string(priv.first) + ".gguf");
        WriteBytes(p, g.build());
        const Status st = ReblockPq2ToQ2Gguf(p, out_path);
        CHECK(st.code == ErrorCode::kUnsupported);
        CHECK_FALSE(std::filesystem::exists(out_path));
    }

    // 3. Legacy group-128-stored-as-type-42 layout (34 bytes for 128 elements instead of 2*18=36 bytes).
    {
        RawGgufBuilder g;
        g.kv_str("general.architecture", "llama");
        g.add_tensor("blk.0.attn_q.weight", {128, 1}, kGgmlTypeQ2_0, std::vector<std::uint8_t>(34, 0));
        const std::filesystem::path legacy = tmp.file("legacy-g128-as-42.gguf");
        WriteBytes(legacy, g.build());
        const Status st = InspectPq2Gguf(legacy, insp);
        CHECK_FALSE(st.ok());
    }

    // 4. PQ2_0 tensor whose ne[0] is not a multiple of 128.
    {
        RawGgufBuilder g;
        g.kv_str("general.architecture", "llama");
        g.add_tensor("blk.0.attn_q.weight", {64, 2}, kGgmlTypePq2_0, std::vector<std::uint8_t>(34, 0));
        const std::filesystem::path bad_dim = tmp.file("bad-ne0.gguf");
        WriteBytes(bad_dim, g.build());
        CHECK(InspectPq2Gguf(bad_dim, insp).code == ErrorCode::kInvalidArgument);
    }

    // 5. Corrupt header with huge n_kv = 2^62 must fail fast without attempting a massive allocation.
    {
        std::vector<std::uint8_t> huge_kv = {'G', 'G', 'U', 'F', 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        const std::uint64_t bad_n_kv = std::uint64_t{1} << 62;
        for (int s = 0; s < 64; s += 8) {
            huge_kv.push_back(static_cast<std::uint8_t>((bad_n_kv >> s) & 0xFFu));
        }
        const std::filesystem::path huge_path = tmp.file("huge-nkv.gguf");
        WriteBytes(huge_path, huge_kv);
        CHECK(InspectPq2Gguf(huge_path, insp).code == ErrorCode::kInvalidArgument);
    }

    // 6. Overlapping tensor offsets and truncated tensor payload.
    {
        RawGgufBuilder g;
        g.kv_str("general.architecture", "llama");
        g.add_tensor("blk.0.attn_q.weight", {128, 1}, kGgmlTypePq2_0, MakePq2TensorBytes(1, 128, 0x3C00u, 1u), 0u);
        g.add_tensor("blk.0.attn_k.weight", {128, 1}, kGgmlTypePq2_0, MakePq2TensorBytes(1, 128, 0x3C00u, 2u), 0u);
        const std::filesystem::path overlap = tmp.file("overlap.gguf");
        WriteBytes(overlap, g.build());
        CHECK(InspectPq2Gguf(overlap, insp).code == ErrorCode::kInvalidArgument);

        std::vector<std::uint8_t> trunc = BuildSyntheticBonsaiPq2Gguf();
        trunc.resize(trunc.size() - 10);
        const std::filesystem::path trunc_path = tmp.file("trunc.gguf");
        WriteBytes(trunc_path, trunc);
        CHECK(InspectPq2Gguf(trunc_path, insp).code == ErrorCode::kInvalidArgument);
    }

        // 7. NaN FP16 scale inside a PQ2_0 tensor is rejected during re-blocking and leaves no output file.
    {
        RawGgufBuilder g;
        g.kv_str("general.architecture", "llama");
        g.add_tensor("blk.0.attn_q.weight", {128, 1}, kGgmlTypePq2_0, MakePq2TensorBytes(1, 128, 0x7E00u, 1u));
        const std::filesystem::path nan_gguf = tmp.file("nan-scale.gguf");
        WriteBytes(nan_gguf, g.build());
        CHECK(ReblockPq2ToQ2Gguf(nan_gguf, out_path).code == ErrorCode::kInvalidArgument);
        CHECK_FALSE(std::filesystem::exists(out_path));

        // 8. Non-ASCII UTF-8 paths do not throw std::system_error on Windows, and overwriting an
        //    existing destination is atomic (a failed re-block leaves the existing destination intact).
        const std::filesystem::path utf8_in = tmp.path() / std::filesystem::path(u8"bonsai-\u65E5\u672C\u8A9E-in.gguf");
        const std::filesystem::path utf8_out = tmp.path() / std::filesystem::path(u8"bonsai-\u65E5\u672C\u8A9E-out.gguf");
        WriteBytes(utf8_in, BuildSyntheticBonsaiPq2Gguf());
        WriteBytes(utf8_out, {0xAAu, 0xBBu});
        // Failed conversion onto utf8_out must preserve {0xAA, 0xBB}.
        CHECK(ReblockPq2ToQ2Gguf(nan_gguf, utf8_out).code == ErrorCode::kInvalidArgument);
        CHECK(ReadBytes(utf8_out) == std::vector<std::uint8_t>{0xAAu, 0xBBu});
        // Valid conversion onto existing utf8_out atomically replaces it and passes verification.
        Pq2ReblockReport utf8_rep;
        const Status utf8_st = ReblockPq2ToQ2Gguf(utf8_in, utf8_out, &utf8_rep);
        REQUIRE_MESSAGE(utf8_st.ok(), utf8_st.message);
        REQUIRE_MESSAGE(VerifyPq2ToQ2Gguf(utf8_in, utf8_out).ok(), "non-ASCII re-blocked GGUF must verify");
    }
}
