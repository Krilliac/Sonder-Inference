// Unit tests for GPU-VRAM-budgeted KV capacity estimation and DynamicKvCapacity.
#include <doctest/doctest.h>

#include "sonder/inference/scheduler/kv_capacity.hpp"
#include "sonder/inference/scheduler/scheduler.hpp"

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

using namespace sonder::inference::scheduler;

namespace {

constexpr std::uint64_t kKiB = 1024ull;
constexpr std::uint64_t kMiB = 1024ull * kKiB;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

KvGeometry make_standard_32l_geometry(KvQuantFormat k_fmt, KvQuantFormat v_fmt) noexcept {
    KvGeometry g;
    g.kv_layers = 32;
    g.kv_heads = 8;
    g.head_dim_k = 128;
    g.head_dim_v = 0;  // defaults to head_dim_k
    g.block_size_tokens = 16;
    g.k_format = k_fmt;
    g.v_format = v_fmt;
    return g;
}

RequestSpec make_spec(RequestId id, TokenCount prompt, TokenCount max_new) {
    RequestSpec s;
    s.id = id;
    s.workload = WorkloadClass::ImplementationWorker;
    s.prompt_tokens = prompt;
    s.max_new_tokens = max_new;
    return s;
}

}  // namespace

TEST_CASE("kv_quant_format_parse_and_name_round_trip") {
    struct Case {
        KvQuantFormat fmt;
        const char* canonical;
        const char* mixed_case;
    };
    const Case cases[] = {
        {KvQuantFormat::f32, "f32", "F32"},
        {KvQuantFormat::f16, "f16", "F16"},
        {KvQuantFormat::bf16, "bf16", "Bf16"},
        {KvQuantFormat::f8_e4m3, "f8_e4m3", "F8_E4m3"},
        {KvQuantFormat::q8_0, "q8_0", "Q8_0"},
        {KvQuantFormat::q5_1, "q5_1", "Q5_1"},
        {KvQuantFormat::q5_0, "q5_0", "Q5_0"},
        {KvQuantFormat::q4_0, "q4_0", "Q4_0"},
        {KvQuantFormat::q2_0, "q2_0", "Q2_0"},
        {KvQuantFormat::pq2_0m, "pq2_0m", "Pq2_0M"},
    };

    for (const Case& c : cases) {
        const std::string_view name(kv_quant_format_name(c.fmt));
        CHECK(name == std::string_view(c.canonical));

        const std::optional<KvQuantFormat> parsed_canonical = parse_kv_quant_format(c.canonical);
        REQUIRE(parsed_canonical.has_value());
        CHECK(*parsed_canonical == c.fmt);

        const std::optional<KvQuantFormat> parsed_mixed = parse_kv_quant_format(c.mixed_case);
        REQUIRE(parsed_mixed.has_value());
        CHECK(*parsed_mixed == c.fmt);
    }

    const std::string_view invalid_names[] = {
        "", " ", "f16 ", " f16", "fp16", "fp8", "f8", "q4_1", "q2_k", "pq2_0", "int8", "unknown",
    };
    for (const std::string_view bad : invalid_names) {
        CHECK_FALSE(parse_kv_quant_format(bad).has_value());
    }

    const auto invalid_fmt = static_cast<KvQuantFormat>(255u);
    CHECK(std::string_view(kv_quant_format_name(invalid_fmt)) == std::string_view("unknown"));
    CHECK(kv_quant_bits_per_element(invalid_fmt) == KvBitsPerElement{0u, 1u});
}

TEST_CASE("kv_quant_bits_per_element_exact_ggml_layouts") {
    static_assert(kv_quant_bits_per_element(KvQuantFormat::f32) == KvBitsPerElement{32u, 1u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::f16) == KvBitsPerElement{16u, 1u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::bf16) == KvBitsPerElement{16u, 1u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::f8_e4m3) == KvBitsPerElement{8u, 1u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::q8_0) == KvBitsPerElement{17u, 2u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::q5_1) == KvBitsPerElement{6u, 1u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::q5_0) == KvBitsPerElement{11u, 2u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::q4_0) == KvBitsPerElement{9u, 2u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::q2_0) == KvBitsPerElement{9u, 4u});
    static_assert(kv_quant_bits_per_element(KvQuantFormat::pq2_0m) == KvBitsPerElement{9u, 4u});

    CHECK(kv_quant_bits_per_element(KvQuantFormat::f32) == KvBitsPerElement{32u, 1u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::f16) == KvBitsPerElement{16u, 1u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::bf16) == KvBitsPerElement{16u, 1u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::f8_e4m3) == KvBitsPerElement{8u, 1u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::q8_0) == KvBitsPerElement{17u, 2u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::q5_1) == KvBitsPerElement{6u, 1u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::q5_0) == KvBitsPerElement{11u, 2u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::q4_0) == KvBitsPerElement{9u, 2u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::q2_0) == KvBitsPerElement{9u, 4u});
    CHECK(kv_quant_bits_per_element(KvQuantFormat::pq2_0m) == KvBitsPerElement{9u, 4u});
}

TEST_CASE("estimate_kv_blocks_quant_ratios_and_mixed_k_v") {
    const std::uint64_t budget = 2ull * kGiB;

    const auto est_f32 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::f32, KvQuantFormat::f32));
    const auto est_f16 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::f16, KvQuantFormat::f16));
    const auto est_bf16 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::bf16, KvQuantFormat::bf16));
    const auto est_fp8 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::f8_e4m3, KvQuantFormat::f8_e4m3));
    const auto est_q8 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::q8_0, KvQuantFormat::q8_0));
    const auto est_q5_1 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::q5_1, KvQuantFormat::q5_1));
    const auto est_q5_0 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::q5_0, KvQuantFormat::q5_0));
    const auto est_q4 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::q4_0, KvQuantFormat::q4_0));
    const auto est_q2 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::q2_0, KvQuantFormat::q2_0));
    const auto est_pq2 =
        estimate_kv_blocks_for_vram_budget(budget, make_standard_32l_geometry(KvQuantFormat::pq2_0m, KvQuantFormat::pq2_0m));

    REQUIRE(est_f32.valid);
    REQUIRE(est_f16.valid);
    REQUIRE(est_bf16.valid);
    REQUIRE(est_fp8.valid);
    REQUIRE(est_q8.valid);
    REQUIRE(est_q5_1.valid);
    REQUIRE(est_q5_0.valid);
    REQUIRE(est_q4.valid);
    REQUIRE(est_q2.valid);
    REQUIRE(est_pq2.valid);

    CHECK_EQ(est_f32.bytes_per_token, 262'144ull);
    CHECK_EQ(est_f32.bytes_per_block, 4ull * kMiB);
    CHECK_EQ(est_f32.total_blocks, 512ull);
    CHECK_EQ(est_f32.total_token_capacity, 8192ull);

    CHECK_EQ(est_f16.bytes_per_token, 131'072ull);
    CHECK_EQ(est_f16.bytes_per_block, 2ull * kMiB);
    CHECK_EQ(est_f16.total_blocks, 1024ull);
    CHECK_EQ(est_f16.total_token_capacity, 16'384ull);

    CHECK_EQ(est_bf16.bytes_per_token, est_f16.bytes_per_token);
    CHECK_EQ(est_bf16.bytes_per_block, est_f16.bytes_per_block);
    CHECK_EQ(est_bf16.total_blocks, est_f16.total_blocks);

    // FP8 halves byte footprint and doubles block capacity relative to f16.
    CHECK_EQ(est_fp8.bytes_per_token, 65'536ull);
    CHECK_EQ(est_fp8.bytes_per_block, 1ull * kMiB);
    CHECK_EQ(est_fp8.total_blocks, est_f16.total_blocks * 2ull);
    CHECK_EQ(est_fp8.total_token_capacity, 32'768ull);

    CHECK_EQ(est_q8.bytes_per_token, 69'632ull);
    CHECK_EQ(est_q8.bytes_per_block, 1'114'112ull);
    CHECK_EQ(est_q8.total_blocks, 1927ull);
    CHECK_LE(est_q8.total_blocks * est_q8.bytes_per_block, budget);
    CHECK_GT((est_q8.total_blocks + 1ull) * est_q8.bytes_per_block, budget);

    CHECK_EQ(est_q5_1.bytes_per_token, 49'152ull);
    CHECK_EQ(est_q5_1.bytes_per_block, 786'432ull);
    CHECK_EQ(est_q5_1.total_blocks, 2730ull);

    CHECK_EQ(est_q5_0.bytes_per_token, 45'056ull);
    CHECK_EQ(est_q5_0.bytes_per_block, 720'896ull);
    CHECK_EQ(est_q5_0.total_blocks, 2978ull);

    CHECK_EQ(est_q4.bytes_per_token, 36'864ull);
    CHECK_EQ(est_q4.bytes_per_block, 589'824ull);
    CHECK_EQ(est_q4.total_blocks, 3640ull);
    CHECK_GT(est_q4.total_blocks, est_f16.total_blocks * 3ull);

    // Ternary / Bonsai KV (q2_0 = 18 B/64, pq2_0m = 36 B/128 -> exact 9/64 of f16).
    CHECK_EQ(est_q2.bytes_per_token, 18'432ull);
    CHECK_EQ(est_q2.bytes_per_block, 294'912ull);
    CHECK_EQ(est_q2.total_blocks, 7281ull);
    CHECK_EQ(est_q2.bytes_per_token * 64ull, est_f16.bytes_per_token * 9ull);
    CHECK_GT(est_q2.total_blocks, est_f16.total_blocks * 7ull);

    CHECK_EQ(est_pq2.bytes_per_token, est_q2.bytes_per_token);
    CHECK_EQ(est_pq2.bytes_per_block, est_q2.bytes_per_block);
    CHECK_EQ(est_pq2.total_blocks, est_q2.total_blocks);

    // Asymmetric K/V formats and distinct head_dim_k / head_dim_v.
    KvGeometry asym;
    asym.kv_layers = 32;
    asym.kv_heads = 8;
    asym.head_dim_k = 192;
    asym.head_dim_v = 128;
    asym.block_size_tokens = 16;
    asym.k_format = KvQuantFormat::q8_0;
    asym.v_format = KvQuantFormat::q4_0;
    const auto est_asym = estimate_kv_blocks_for_vram_budget(budget, asym);
    REQUIRE(est_asym.valid);
    CHECK_EQ(est_asym.bytes_per_token, 70'656ull);
    CHECK_EQ(est_asym.bytes_per_block, 1'130'496ull);
    CHECK_EQ(est_asym.total_blocks, budget / 1'130'496ull);
}

TEST_CASE("estimate_kv_blocks_ceiling_division_never_overestimates") {
    KvGeometry tiny_q4;
    tiny_q4.kv_layers = 1;
    tiny_q4.kv_heads = 1;
    tiny_q4.head_dim_k = 1;
    tiny_q4.block_size_tokens = 16;
    tiny_q4.k_format = KvQuantFormat::q4_0;
    tiny_q4.v_format = KvQuantFormat::q4_0;
    // 1 token = 4.5 + 4.5 = 9 bits = 1.125 B -> ceil = 2 B; 16 tokens = 144 bits = 18 B.
    const auto est_17 = estimate_kv_blocks_for_vram_budget(17ull, tiny_q4);
    REQUIRE(est_17.valid);
    CHECK_EQ(est_17.bytes_per_token, 2ull);
    CHECK_EQ(est_17.bytes_per_block, 18ull);
    CHECK_EQ(est_17.total_blocks, 0ull);
    CHECK_EQ(est_17.total_token_capacity, 0ull);

    const auto est_18 = estimate_kv_blocks_for_vram_budget(18ull, tiny_q4);
    REQUIRE(est_18.valid);
    CHECK_EQ(est_18.total_blocks, 1ull);
    CHECK_EQ(est_18.total_token_capacity, 16ull);

    // Non-multiple target_context_tokens rounds blocks per sequence UP.
    const auto est_36_ctx17 = estimate_kv_blocks_for_vram_budget(36ull, tiny_q4, 17ull);
    REQUIRE(est_36_ctx17.valid);
    CHECK_EQ(est_36_ctx17.total_blocks, 2ull);
    CHECK_EQ(est_36_ctx17.max_concurrent_sequences, 1u);

    const auto est_35_ctx17 = estimate_kv_blocks_for_vram_budget(35ull, tiny_q4, 17ull);
    REQUIRE(est_35_ctx17.valid);
    CHECK_EQ(est_35_ctx17.total_blocks, 1ull);
    CHECK_EQ(est_35_ctx17.max_concurrent_sequences, 0u);

    // Fractional-bit q2_0 tiny geometry: 1 token = 4.5 bits -> ceil = 1 B; 16 tokens = 72 bits = 9 B.
    KvGeometry tiny_q2 = tiny_q4;
    tiny_q2.k_format = KvQuantFormat::q2_0;
    tiny_q2.v_format = KvQuantFormat::q2_0;
    const auto est_q2 = estimate_kv_blocks_for_vram_budget(27ull, tiny_q2);
    REQUIRE(est_q2.valid);
    CHECK_EQ(est_q2.bytes_per_token, 1ull);
    CHECK_EQ(est_q2.bytes_per_block, 9ull);
    CHECK_EQ(est_q2.total_blocks, 3ull);

    // Zero budget with valid geometry is valid with 0 blocks.
    const auto est_zero = estimate_kv_blocks_for_vram_budget(0ull, tiny_q4, 16ull);
    REQUIRE(est_zero.valid);
    CHECK_EQ(est_zero.bytes_per_token, 2ull);
    CHECK_EQ(est_zero.bytes_per_block, 18ull);
    CHECK_EQ(est_zero.total_blocks, 0ull);
    CHECK_EQ(est_zero.total_token_capacity, 0ull);
    CHECK_EQ(est_zero.max_concurrent_sequences, 0u);
}

TEST_CASE("estimate_kv_blocks_hybrid_gpu_profiles_and_recurrent_cap") {
    // 64-layer hybrid Qwen3.5 / Bonsai-2 style geometry: 16 attention KV layers,
    // 48 GDN/recurrent layers (~150 MiB per-sequence recurrent state).
    constexpr std::uint64_t kRecurrentBytes = 150ull * kMiB;
    constexpr std::uint64_t kTargetCtx = 8192ull;

    KvGeometry hybrid_rtx;
    hybrid_rtx.kv_layers = 16;
    hybrid_rtx.kv_heads = 4;
    hybrid_rtx.head_dim_k = 256;
    hybrid_rtx.head_dim_v = 256;
    hybrid_rtx.block_size_tokens = 16;
    hybrid_rtx.k_format = KvQuantFormat::f8_e4m3;
    hybrid_rtx.v_format = KvQuantFormat::f8_e4m3;

    // 16 GiB RTX 5070 Ti: 4 GiB KV+state budget after model weights.
    const std::uint64_t rtx_budget = 4ull * kGiB;
    const auto rtx_no_rec = estimate_kv_blocks_for_vram_budget(rtx_budget, hybrid_rtx, kTargetCtx);
    REQUIRE(rtx_no_rec.valid);
    CHECK_EQ(rtx_no_rec.bytes_per_token, 32ull * kKiB);
    CHECK_EQ(rtx_no_rec.bytes_per_block, 512ull * kKiB);
    CHECK_EQ(rtx_no_rec.total_blocks, 8192ull);
    CHECK_EQ(rtx_no_rec.total_token_capacity, 131'072ull);
    CHECK_EQ(rtx_no_rec.max_concurrent_sequences, 16u);

    hybrid_rtx.per_sequence_recurrent_bytes = kRecurrentBytes;
    const auto rtx_with_rec = estimate_kv_blocks_for_vram_budget(rtx_budget, hybrid_rtx, kTargetCtx);
    REQUIRE(rtx_with_rec.valid);
    CHECK_EQ(rtx_with_rec.total_blocks, 8192ull);
    // 256 MiB KV + 150 MiB recurrent = 406 MiB/seq -> floor(4096 / 406) = 10.
    CHECK_EQ(rtx_with_rec.max_concurrent_sequences, 10u);

    // 24 GiB AMD RX XT: 10 GiB KV+state budget using q8_0 KV (no FP8 required).
    KvGeometry hybrid_xt = hybrid_rtx;
    hybrid_xt.k_format = KvQuantFormat::q8_0;
    hybrid_xt.v_format = KvQuantFormat::q8_0;
    const std::uint64_t xt_budget = 10ull * kGiB;
    const auto xt_est = estimate_kv_blocks_for_vram_budget(xt_budget, hybrid_xt, kTargetCtx);
    REQUIRE(xt_est.valid);
    CHECK_EQ(xt_est.bytes_per_token, 34'816ull);
    CHECK_EQ(xt_est.bytes_per_block, 557'056ull);
    // 272 MiB KV + 150 MiB recurrent = 422 MiB/seq -> floor(10240 / 422) = 24.
    CHECK_EQ(xt_est.max_concurrent_sequences, 24u);
    CHECK_GT(xt_est.max_concurrent_sequences, rtx_with_rec.max_concurrent_sequences);

    // 12 GiB AMD Ryzen APU UMA: 1536 MiB KV+state budget using pq2_0m ternary KV.
    KvGeometry hybrid_apu = hybrid_rtx;
    hybrid_apu.k_format = KvQuantFormat::pq2_0m;
    hybrid_apu.v_format = KvQuantFormat::pq2_0m;
    hybrid_apu.per_sequence_recurrent_bytes = 0;
    const std::uint64_t apu_budget = 1536ull * kMiB;
    const auto apu_no_rec = estimate_kv_blocks_for_vram_budget(apu_budget, hybrid_apu, kTargetCtx);
    REQUIRE(apu_no_rec.valid);
    CHECK_EQ(apu_no_rec.bytes_per_token, 9216ull);
    CHECK_EQ(apu_no_rec.bytes_per_block, 147'456ull);
    // 72 MiB KV/seq -> floor(1536 / 72) = 21.
    CHECK_EQ(apu_no_rec.max_concurrent_sequences, 21u);

    hybrid_apu.per_sequence_recurrent_bytes = kRecurrentBytes;
    const auto apu_with_rec = estimate_kv_blocks_for_vram_budget(apu_budget, hybrid_apu, kTargetCtx);
    REQUIRE(apu_with_rec.valid);
    // 72 MiB KV + 150 MiB recurrent = 222 MiB/seq -> floor(1536 / 222) = 6.
    CHECK_EQ(apu_with_rec.max_concurrent_sequences, 6u);

    // target_context_tokens == 0 leaves max_concurrent_sequences == 0.
    const auto no_ctx = estimate_kv_blocks_for_vram_budget(rtx_budget, hybrid_rtx, 0ull);
    REQUIRE(no_ctx.valid);
    CHECK_EQ(no_ctx.max_concurrent_sequences, 0u);

    // Joint KV + recurrent check: 512 MiB KV + 600 MiB recurrent > 1024 MiB budget => 0 sequences.
    KvGeometry tight = hybrid_rtx;
    tight.per_sequence_recurrent_bytes = 600ull * kMiB;
    const auto est_tight = estimate_kv_blocks_for_vram_budget(1ull * kGiB, tight, 16'384ull);
    REQUIRE(est_tight.valid);
    CHECK_EQ(est_tight.max_concurrent_sequences, 0u);
}

TEST_CASE("estimate_kv_blocks_invalid_geometry_and_overflow_resistance") {
    const auto check_invalid = [](const VramKvCapacityEstimate& e) {
        CHECK_FALSE(e.valid);
        CHECK_EQ(e.bytes_per_token, 0ull);
        CHECK_EQ(e.bytes_per_block, 0ull);
        CHECK_EQ(e.total_blocks, 0ull);
        CHECK_EQ(e.total_token_capacity, 0ull);
        CHECK_EQ(e.max_concurrent_sequences, 0u);
    };

    const KvGeometry valid_geom = make_standard_32l_geometry(KvQuantFormat::f16, KvQuantFormat::f16);
    {
        KvGeometry g = valid_geom;
        g.kv_layers = 0;
        check_invalid(estimate_kv_blocks_for_vram_budget(1ull * kGiB, g, 1024ull));
    }
    {
        KvGeometry g = valid_geom;
        g.kv_heads = 0;
        check_invalid(estimate_kv_blocks_for_vram_budget(1ull * kGiB, g, 1024ull));
    }
    {
        KvGeometry g = valid_geom;
        g.head_dim_k = 0;
        check_invalid(estimate_kv_blocks_for_vram_budget(1ull * kGiB, g, 1024ull));
    }
    {
        KvGeometry g = valid_geom;
        g.block_size_tokens = 0;
        check_invalid(estimate_kv_blocks_for_vram_budget(1ull * kGiB, g, 1024ull));
    }
    {
        KvGeometry g = valid_geom;
        g.k_format = static_cast<KvQuantFormat>(255u);
        check_invalid(estimate_kv_blocks_for_vram_budget(1ull * kGiB, g, 1024ull));
    }
    {
        KvGeometry g = valid_geom;
        g.v_format = static_cast<KvQuantFormat>(255u);
        check_invalid(estimate_kv_blocks_for_vram_budget(1ull * kGiB, g, 1024ull));
    }

    constexpr std::uint32_t kMaxU32 = std::numeric_limits<std::uint32_t>::max();
    constexpr std::uint64_t kMaxU64 = std::numeric_limits<std::uint64_t>::max();

    // Overflow in bytes_per_token.
    {
        KvGeometry huge;
        huge.kv_layers = kMaxU32;
        huge.kv_heads = kMaxU32;
        huge.head_dim_k = kMaxU32;
        huge.block_size_tokens = 16;
        check_invalid(estimate_kv_blocks_for_vram_budget(kMaxU64, huge, 4096ull));
    }

    // bytes_per_token fits (2^62), but bytes_per_block (2^66) overflows uint64_t.
    {
        KvGeometry block_ovf;
        block_ovf.kv_layers = 1u << 26;
        block_ovf.kv_heads = 1u << 26;
        block_ovf.head_dim_k = 256u;
        block_ovf.block_size_tokens = 16u;
        block_ovf.k_format = KvQuantFormat::f16;
        block_ovf.v_format = KvQuantFormat::f16;
        check_invalid(estimate_kv_blocks_for_vram_budget(kMaxU64, block_ovf, 16ull));
    }

    // total_blocks * block_size_tokens overflows uint64_t when bytes_per_block < block_size_tokens.
    {
        KvGeometry sub_byte_block;
        sub_byte_block.kv_layers = 1;
        sub_byte_block.kv_heads = 1;
        sub_byte_block.head_dim_k = 1;
        sub_byte_block.block_size_tokens = 16;
        sub_byte_block.k_format = KvQuantFormat::q2_0;
        sub_byte_block.v_format = KvQuantFormat::q2_0;
        check_invalid(estimate_kv_blocks_for_vram_budget(kMaxU64, sub_byte_block, 16ull));
    }

    // UINT64_MAX budget with normal geometry succeeds without overflow and saturates sequences.
    {
        const auto est = estimate_kv_blocks_for_vram_budget(kMaxU64, valid_geom, 4096ull);
        REQUIRE(est.valid);
        CHECK_EQ(est.bytes_per_block, 2ull * kMiB);
        CHECK_EQ(est.total_blocks, kMaxU64 / (2ull * kMiB));
        CHECK_EQ(est.total_token_capacity, est.total_blocks * 16ull);
        CHECK_EQ(est.max_concurrent_sequences, kMaxU32);
    }

    // UINT64_MAX target_context_tokens never wraps blocks_per_seq.
    {
        const auto est = estimate_kv_blocks_for_vram_budget(2ull * kGiB, valid_geom, kMaxU64);
        REQUIRE(est.valid);
        CHECK_EQ(est.total_blocks, 1024ull);
        CHECK_EQ(est.max_concurrent_sequences, 0u);
    }

    // UINT64_MAX per_sequence_recurrent_bytes never wraps total_bytes_per_seq.
    {
        KvGeometry huge_rec = valid_geom;
        huge_rec.per_sequence_recurrent_bytes = kMaxU64;
        const auto est = estimate_kv_blocks_for_vram_budget(kMaxU64, huge_rec, 16ull);
        REQUIRE(est.valid);
        CHECK_EQ(est.max_concurrent_sequences, 0u);
    }
}

TEST_CASE("dynamic_kv_capacity_grow_shrink_drain_and_readmit") {
    DynamicKvCapacity cap(100, 16);
    CHECK_EQ(cap.block_size_tokens(), 16u);
    CHECK_EQ(cap.total_blocks(), 100ull);
    CHECK_EQ(cap.free_blocks(), 100ull);
    CHECK_EQ(cap.used_blocks(), 0ull);
    CHECK_EQ(cap.peak_used_blocks(), 0ull);
    CHECK_FALSE(cap.is_over_budget());
    CHECK_EQ(cap.over_budget_blocks(), 0ull);
    CHECK_EQ(cap.resize_count(), 0ull);

    CHECK(cap.try_reserve(1, 40));
    CHECK(cap.try_reserve(2, 40));
    CHECK(cap.try_reserve(99, 0));
    CHECK_EQ(cap.blocks_held(99), 0ull);
    CHECK_EQ(cap.used_blocks(), 80ull);
    CHECK_EQ(cap.free_blocks(), 20ull);
    CHECK_EQ(cap.peak_used_blocks(), 80ull);

    CHECK_FALSE(cap.try_reserve(3, 21));
    CHECK_FALSE(cap.try_reserve(3, std::numeric_limits<BlockCount>::max()));
    CHECK_EQ(cap.used_blocks(), 80ull);

    // Grow capacity and confirm resize_count increments only on actual changes.
    cap.set_total_blocks(150);
    CHECK_EQ(cap.total_blocks(), 150ull);
    CHECK_EQ(cap.free_blocks(), 70ull);
    CHECK_EQ(cap.resize_count(), 1ull);

    cap.set_total_blocks(150);
    CHECK_EQ(cap.resize_count(), 1ull);

    CHECK(cap.try_reserve(3, 30));
    CHECK_EQ(cap.used_blocks(), 110ull);
    CHECK_EQ(cap.peak_used_blocks(), 110ull);
    CHECK_EQ(cap.free_blocks(), 40ull);

    // Shrink below used_blocks(): drain semantics without underflow.
    cap.set_total_blocks(60);
    CHECK_EQ(cap.resize_count(), 2ull);
    CHECK_EQ(cap.total_blocks(), 60ull);
    CHECK_EQ(cap.used_blocks(), 110ull);
    CHECK_EQ(cap.peak_used_blocks(), 110ull);
    CHECK_EQ(cap.free_blocks(), 0ull);
    CHECK(cap.is_over_budget());
    CHECK_EQ(cap.over_budget_blocks(), 50ull);
    CHECK_EQ(cap.blocks_held(1), 40ull);
    CHECK_EQ(cap.blocks_held(2), 40ull);
    CHECK_EQ(cap.blocks_held(3), 30ull);

    // New non-zero reservations fail while over budget; zero reservation is a no-op.
    CHECK_FALSE(cap.try_reserve(4, 1));
    CHECK_FALSE(cap.try_reserve(1, 1));
    CHECK(cap.try_reserve(4, 0));

    // Partial drain: still over budget.
    cap.release(3);
    cap.release(3);  // idempotent double release
    CHECK_EQ(cap.blocks_held(3), 0ull);
    CHECK_EQ(cap.used_blocks(), 80ull);
    CHECK_EQ(cap.free_blocks(), 0ull);
    CHECK(cap.is_over_budget());
    CHECK_EQ(cap.over_budget_blocks(), 20ull);
    CHECK_FALSE(cap.try_reserve(4, 1));

    // Drain below shrunk total: re-admission resumes cleanly.
    cap.release(2);
    CHECK_EQ(cap.used_blocks(), 40ull);
    CHECK_FALSE(cap.is_over_budget());
    CHECK_EQ(cap.over_budget_blocks(), 0ull);
    CHECK_EQ(cap.free_blocks(), 20ull);
    CHECK(cap.try_reserve(4, 20));
    CHECK_EQ(cap.used_blocks(), 60ull);
    CHECK_EQ(cap.free_blocks(), 0ull);
    CHECK_FALSE(cap.try_reserve(5, 1));

    // Shrink all the way to 0 blocks and drain cleanly.
    cap.set_total_blocks(0);
    CHECK_EQ(cap.free_blocks(), 0ull);
    CHECK(cap.is_over_budget());
    CHECK_EQ(cap.over_budget_blocks(), 60ull);
    cap.release(1);
    cap.release(4);
    CHECK_EQ(cap.used_blocks(), 0ull);
    CHECK_FALSE(cap.is_over_budget());
    CHECK_EQ(cap.over_budget_blocks(), 0ull);
}

TEST_CASE("dynamic_kv_capacity_update_from_vram_budget") {
    DynamicKvCapacity cap(0, 16);

    KvGeometry geom;
    geom.kv_layers = 16;
    geom.kv_heads = 4;
    geom.head_dim_k = 128;
    geom.block_size_tokens = 16;
    geom.k_format = KvQuantFormat::f16;
    geom.v_format = KvQuantFormat::f16;
    // bytes_per_token = 16 * 4 * (128 * 2 + 128 * 2) = 32 KiB; bytes_per_block = 512 KiB.
    // 32 MiB budget -> 64 blocks.
    cap.update_from_vram_budget(32ull * kMiB, geom);
    CHECK_EQ(cap.total_blocks(), 64ull);
    CHECK_EQ(cap.free_blocks(), 64ull);
    CHECK_EQ(cap.resize_count(), 1ull);

    CHECK(cap.try_reserve(1, 50));
    CHECK_EQ(cap.used_blocks(), 50ull);

    // Repeating same budget/geometry is a no-op for resize_count.
    cap.update_from_vram_budget(32ull * kMiB, geom);
    CHECK_EQ(cap.resize_count(), 1ull);

    // Switching KV quantization from f16 to f8_e4m3 doubles capacity in the same 32 MiB budget.
    geom.k_format = KvQuantFormat::f8_e4m3;
    geom.v_format = KvQuantFormat::f8_e4m3;
    cap.update_from_vram_budget(32ull * kMiB, geom);
    CHECK_EQ(cap.total_blocks(), 128ull);
    CHECK_EQ(cap.free_blocks(), 78ull);
    CHECK_EQ(cap.resize_count(), 2ull);

    // Shrinking VRAM budget below held reservations triggers safe drain state.
    cap.update_from_vram_budget(8ull * kMiB, geom);
    CHECK_EQ(cap.total_blocks(), 32ull);
    CHECK_EQ(cap.free_blocks(), 0ull);
    CHECK(cap.is_over_budget());
    CHECK_EQ(cap.over_budget_blocks(), 18ull);
    CHECK_EQ(cap.resize_count(), 3ull);

    // Invalid geometry leaves pool untouched.
    KvGeometry bad = geom;
    bad.kv_layers = 0;
    cap.update_from_vram_budget(64ull * kMiB, bad);
    CHECK_EQ(cap.total_blocks(), 32ull);
    CHECK_EQ(cap.resize_count(), 3ull);

    KvGeometry zero_bs = geom;
    zero_bs.block_size_tokens = 0;
    cap.update_from_vram_budget(64ull * kMiB, zero_bs);
    CHECK_EQ(cap.total_blocks(), 32ull);
    CHECK_EQ(cap.resize_count(), 3ull);

    // Pool aligns geometry block size to pool's own block_size_tokens().
    DynamicKvCapacity cap32(0, 32);
    cap32.update_from_vram_budget(32ull * kMiB, geom);  // geom has block_size_tokens=16, cap32 has 32
    CHECK_EQ(cap32.total_blocks(), 64ull);              // 32 MiB / (16 KiB * 32) = 64 blocks

    // Zero budget with valid geometry shrinks to 0 blocks.
    cap.update_from_vram_budget(0ull, geom);
    CHECK_EQ(cap.total_blocks(), 0ull);
    CHECK_EQ(cap.over_budget_blocks(), 50ull);
    CHECK_EQ(cap.resize_count(), 4ull);
}

TEST_CASE("dynamic_kv_capacity_integrates_with_scheduler_shrink_drain") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;

    SimClock clock;
    DynamicKvCapacity kv(12, 16);
    Scheduler sched(cfg, clock, kv);

    REQUIRE(sched.submit(make_spec(1, 60, 2)).accepted);
    REQUIRE(sched.submit(make_spec(2, 60, 2)).accepted);

    const StepPlan p1 = sched.plan_step();
    CHECK_EQ(p1.admitted.size(), 2u);
    CHECK_EQ(kv.used_blocks(), 8ull);
    CHECK_EQ(kv.free_blocks(), 4ull);
    clock.advance(1000);
    sched.complete_step(p1);

    // Shrink pool below active usage (8 held > 5 total).
    kv.set_total_blocks(5);
    CHECK(kv.is_over_budget());
    CHECK_EQ(kv.over_budget_blocks(), 3ull);
    CHECK_EQ(kv.free_blocks(), 0ull);

    // Request 3 fits within total_blocks (3 <= 5) so submit succeeds, but admission waits.
    REQUIRE(sched.submit(make_spec(3, 32, 2)).accepted);

    const StepPlan p2 = sched.plan_step();
    CHECK(p2.admitted.empty());
    const std::optional<RequestState> st3_waiting = sched.state(3);
    REQUIRE(st3_waiting.has_value());
    CHECK(*st3_waiting == RequestState::WaitingAdmission);
    clock.advance(1000);
    sched.complete_step(p2);

    // Both initial requests finished their 2nd token and released their KV blocks.
    CHECK_EQ(kv.used_blocks(), 0ull);
    CHECK_FALSE(kv.is_over_budget());
    CHECK_EQ(kv.free_blocks(), 5ull);

    const StepPlan p3 = sched.plan_step();
    CHECK(p3.admitted == (std::vector<RequestId>{3u}));
    CHECK_EQ(kv.blocks_held(3), 2ull);
    clock.advance(1000);
    sched.complete_step(p3);
}

TEST_CASE("dynamic_kv_capacity_priority_preemption_accounts_for_over_budget_deficit") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 2;
    cfg.enable_priority_preemption = true;
    cfg.priority_preemption_min_rank_gap = 2;

    SimClock clock;
    DynamicKvCapacity kv(20, 16);
    Scheduler sched(cfg, clock, kv);

    // FixedKvCapacity uses KvCapacity::over_budget_blocks() default (0).
    const FixedKvCapacity fixed(20, 16);
    const KvCapacity& fixed_iface = fixed;
    CHECK_EQ(fixed_iface.over_budget_blocks(), 0ull);

    RequestSpec high_running = make_spec(1, 120, 4);  // 8 blocks; decode stays <= 124 tokens
    high_running.workload = WorkloadClass::InteractiveUser;
    RequestSpec low_victim = make_spec(2, 72, 4);     // 5 blocks; decode stays <= 76 tokens
    low_victim.workload = WorkloadClass::BackgroundIndexing;

    REQUIRE(sched.submit(high_running).accepted);
    REQUIRE(sched.submit(low_victim).accepted);

    const StepPlan p1 = sched.plan_step();
    CHECK_EQ(p1.admitted.size(), 2u);
    CHECK_EQ(kv.used_blocks(), 13ull);
    CHECK_EQ(kv.free_blocks(), 7ull);
    clock.advance(1000);
    sched.complete_step(p1);

    // Shrink below active usage: used=13, total=10 => over_budget_blocks=3, free=0.
    // Request 3 needs 2 blocks + 2 watermark + 3 deficit = 7 blocks. Victim 2 only
    // holds 5 blocks (enough for need+watermark=4, NOT enough for 7), so it must
    // NOT be spuriously preempted.
    kv.set_total_blocks(10);
    CHECK(kv.is_over_budget());
    CHECK_EQ(kv.over_budget_blocks(), 3ull);
    CHECK_EQ(kv.free_blocks(), 0ull);

    RequestSpec high_waiting = make_spec(3, 32, 2);   // 2 blocks prefill, worst-case 3 <= 10
    high_waiting.workload = WorkloadClass::InteractiveUser;
    REQUIRE(sched.submit(high_waiting).accepted);

    const StepPlan p2 = sched.plan_step();
    CHECK(p2.preempted.empty());
    CHECK(p2.admitted.empty());
    CHECK_EQ(sched.stats().preemptions_priority, 0ull);
    const std::optional<RequestState> st2_still_running = sched.state(2);
    REQUIRE(st2_still_running.has_value());
    CHECK(*st2_still_running == RequestState::Decode);
    const std::optional<RequestState> st3_still_waiting = sched.state(3);
    REQUIRE(st3_still_waiting.has_value());
    CHECK(*st3_still_waiting == RequestState::WaitingAdmission);
    clock.advance(1000);
    sched.complete_step(p2);

    // Raise total to 12 (still over budget: used=13, total=12 => deficit=1, free=0).
    // Now target = need(2) + watermark(2) + deficit(1) = 5 == blocks_held(2), so
    // preempting victim 2 clears both the deficit and need+watermark and admits req 3.
    kv.set_total_blocks(12);
    CHECK(kv.is_over_budget());
    CHECK_EQ(kv.over_budget_blocks(), 1ull);
    CHECK_EQ(kv.free_blocks(), 0ull);

    const StepPlan p3 = sched.plan_step();
    REQUIRE_EQ(p3.preempted.size(), 1u);
    CHECK_EQ(p3.preempted[0].id, 2ull);
    CHECK(p3.preempted[0].reason == PreemptionReason::PriorityAdmission);
    CHECK(p3.admitted == (std::vector<RequestId>{3u}));
    CHECK_EQ(sched.stats().preemptions_priority, 1ull);
    CHECK_FALSE(kv.is_over_budget());
    CHECK_EQ(kv.over_budget_blocks(), 0ull);
    CHECK_EQ(kv.blocks_held(3), 2ull);
    CHECK_EQ(kv.free_blocks(), 2ull);
    clock.advance(1000);
    sched.complete_step(p3);
}

TEST_CASE("dynamic_kv_capacity_decode_pressure_avoids_spurious_victim_preemption") {
    SchedulerConfig cfg;
    cfg.admission_watermark_blocks = 0;
    cfg.enable_priority_preemption = false;

    SimClock clock;
    DynamicKvCapacity kv(10, 16);
    Scheduler sched(cfg, clock, kv);

    RequestSpec high = make_spec(1, 60, 4);   // 4 blocks; stays <= 64 tokens (no growth)
    high.workload = WorkloadClass::InteractiveUser;
    RequestSpec mid = make_spec(2, 64, 4);    // 4 blocks; next decode wants 5th block
    mid.workload = WorkloadClass::ImplementationWorker;
    RequestSpec low = make_spec(3, 12, 4);    // 1 block; stays <= 16 tokens (no growth)
    low.workload = WorkloadClass::BackgroundIndexing;

    REQUIRE(sched.submit(high).accepted);
    REQUIRE(sched.submit(mid).accepted);
    REQUIRE(sched.submit(low).accepted);

    const StepPlan p1 = sched.plan_step();
    CHECK_EQ(p1.admitted.size(), 3u);
    CHECK_EQ(kv.used_blocks(), 9ull);
    clock.advance(1000);
    sched.complete_step(p1);

    // Shrink total to 6 (used=9, over_budget=3, free=0).
    // mid needs +1 block plus clearing deficit 3 => 4 blocks to free.
    // Worse victim low only holds 1 block (< 4), so preempting low cannot save mid;
    // mid must self-preempt directly without spuriously evicting low.
    kv.set_total_blocks(6);
    const StepPlan p2 = sched.plan_step();
    REQUIRE_EQ(p2.preempted.size(), 1u);
    CHECK_EQ(p2.preempted[0].id, 2ull);
    CHECK(sched.state(2) == RequestState::Preempted);
    CHECK(sched.state(3) == RequestState::Decode);
    CHECK_EQ(kv.used_blocks(), 5ull);
    CHECK_FALSE(kv.is_over_budget());
    clock.advance(1000);
    sched.complete_step(p2);
}
