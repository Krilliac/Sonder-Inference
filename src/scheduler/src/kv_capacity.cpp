#include "sonder/inference/scheduler/kv_capacity.hpp"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>

namespace sonder::inference::scheduler {

namespace {

bool iequals_ascii(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        char ca = a[i];
        if (ca >= 'A' && ca <= 'Z') {
            ca = static_cast<char>(ca - 'A' + 'a');
        }
        if (ca != b[i]) return false;
    }
    return true;
}

struct Uint128 {
    std::uint64_t hi = 0;
    std::uint64_t lo = 0;
};

Uint128 mul_u64(std::uint64_t a, std::uint64_t b) noexcept {
    constexpr std::uint64_t kLoMask = 0xFFFFFFFFull;
    const std::uint64_t a_lo = a & kLoMask;
    const std::uint64_t a_hi = a >> 32;
    const std::uint64_t b_lo = b & kLoMask;
    const std::uint64_t b_hi = b >> 32;

    const std::uint64_t p0 = a_lo * b_lo;
    const std::uint64_t p1 = a_lo * b_hi;
    const std::uint64_t p2 = a_hi * b_lo;
    const std::uint64_t p3 = a_hi * b_hi;

    const std::uint64_t mid = (p0 >> 32) + (p1 & kLoMask) + (p2 & kLoMask);
    const std::uint64_t lo = (p0 & kLoMask) | (mid << 32);
    const std::uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return Uint128{hi, lo};
}

bool add_u128(Uint128 a, Uint128 b, Uint128& out) noexcept {
    const std::uint64_t lo = a.lo + b.lo;
    const std::uint64_t carry = lo < a.lo ? 1ull : 0ull;
    constexpr std::uint64_t kMaxU64 = std::numeric_limits<std::uint64_t>::max();
    if (a.hi > kMaxU64 - b.hi) return false;
    const std::uint64_t hi_sum = a.hi + b.hi;
    if (hi_sum > kMaxU64 - carry) return false;
    out = Uint128{hi_sum + carry, lo};
    return true;
}

bool mul_u128_u64(Uint128 a, std::uint64_t b, Uint128& out) noexcept {
    const Uint128 lo_prod = mul_u64(a.lo, b);
    const Uint128 hi_prod = mul_u64(a.hi, b);
    if (hi_prod.hi != 0) return false;
    constexpr std::uint64_t kMaxU64 = std::numeric_limits<std::uint64_t>::max();
    if (hi_prod.lo > kMaxU64 - lo_prod.hi) return false;
    out = Uint128{hi_prod.lo + lo_prod.hi, lo_prod.lo};
    return true;
}

bool ceil_div_u128_by_u32_to_u64(Uint128 num, std::uint32_t den, std::uint64_t& out) noexcept {
    if (den == 0) return false;
    const std::uint64_t d = static_cast<std::uint64_t>(den);
    const std::uint32_t words[4] = {
        static_cast<std::uint32_t>(num.lo & 0xFFFFFFFFull),
        static_cast<std::uint32_t>(num.lo >> 32),
        static_cast<std::uint32_t>(num.hi & 0xFFFFFFFFull),
        static_cast<std::uint32_t>(num.hi >> 32),
    };

    std::uint64_t rem = 0;
    std::uint32_t q[4] = {0, 0, 0, 0};
    for (int i = 3; i >= 0; --i) {
        const std::uint64_t cur = (rem << 32) | static_cast<std::uint64_t>(words[i]);
        q[i] = static_cast<std::uint32_t>(cur / d);
        rem = cur % d;
    }

    if (q[3] != 0 || q[2] != 0) return false;
    std::uint64_t quot =
        (static_cast<std::uint64_t>(q[1]) << 32) | static_cast<std::uint64_t>(q[0]);
    if (rem != 0) {
        if (quot == std::numeric_limits<std::uint64_t>::max()) return false;
        ++quot;
    }
    out = quot;
    return true;
}

}  // namespace

std::optional<KvQuantFormat> parse_kv_quant_format(std::string_view name) noexcept {
    if (iequals_ascii(name, "f32")) return KvQuantFormat::f32;
    if (iequals_ascii(name, "f16")) return KvQuantFormat::f16;
    if (iequals_ascii(name, "bf16")) return KvQuantFormat::bf16;
    if (iequals_ascii(name, "f8_e4m3")) return KvQuantFormat::f8_e4m3;
    if (iequals_ascii(name, "q8_0")) return KvQuantFormat::q8_0;
    if (iequals_ascii(name, "q5_1")) return KvQuantFormat::q5_1;
    if (iequals_ascii(name, "q5_0")) return KvQuantFormat::q5_0;
    if (iequals_ascii(name, "q4_0")) return KvQuantFormat::q4_0;
    if (iequals_ascii(name, "q2_0")) return KvQuantFormat::q2_0;
    if (iequals_ascii(name, "pq2_0m")) return KvQuantFormat::pq2_0m;
    return std::nullopt;
}

const char* kv_quant_format_name(KvQuantFormat fmt) noexcept {
    switch (fmt) {
        case KvQuantFormat::f32: return "f32";
        case KvQuantFormat::f16: return "f16";
        case KvQuantFormat::bf16: return "bf16";
        case KvQuantFormat::f8_e4m3: return "f8_e4m3";
        case KvQuantFormat::q8_0: return "q8_0";
        case KvQuantFormat::q5_1: return "q5_1";
        case KvQuantFormat::q5_0: return "q5_0";
        case KvQuantFormat::q4_0: return "q4_0";
        case KvQuantFormat::q2_0: return "q2_0";
        case KvQuantFormat::pq2_0m: return "pq2_0m";
    }
    return "unknown";
}

VramKvCapacityEstimate estimate_kv_blocks_for_vram_budget(
    std::uint64_t kv_budget_bytes,
    const KvGeometry& geometry,
    std::uint64_t target_context_tokens) noexcept {
    if (geometry.kv_layers == 0 || geometry.kv_heads == 0 || geometry.head_dim_k == 0 ||
        geometry.block_size_tokens == 0) {
        return {};
    }

    const KvBitsPerElement k_bpe = kv_quant_bits_per_element(geometry.k_format);
    const KvBitsPerElement v_bpe = kv_quant_bits_per_element(geometry.v_format);
    if (k_bpe.num == 0 || k_bpe.den == 0 || v_bpe.num == 0 || v_bpe.den == 0) {
        return {};
    }

    const std::uint64_t layers = static_cast<std::uint64_t>(geometry.kv_layers);
    const std::uint64_t heads = static_cast<std::uint64_t>(geometry.kv_heads);
    const std::uint64_t dim_k = static_cast<std::uint64_t>(geometry.head_dim_k);
    const std::uint64_t dim_v = static_cast<std::uint64_t>(
        geometry.head_dim_v == 0 ? geometry.head_dim_k : geometry.head_dim_v);
    const std::uint64_t block_tokens = static_cast<std::uint64_t>(geometry.block_size_tokens);

    const std::uint32_t common_den = std::lcm(k_bpe.den, v_bpe.den);
    const std::uint32_t byte_den = 8u * common_den;
    const std::uint64_t k_scale = static_cast<std::uint64_t>(common_den / k_bpe.den);
    const std::uint64_t v_scale = static_cast<std::uint64_t>(common_den / v_bpe.den);

    const std::uint64_t k_bits_scaled =
        dim_k * static_cast<std::uint64_t>(k_bpe.num) * k_scale;
    const std::uint64_t v_bits_scaled =
        dim_v * static_cast<std::uint64_t>(v_bpe.num) * v_scale;
    const std::uint64_t per_head_bits_scaled = k_bits_scaled + v_bits_scaled;

    const std::uint64_t layers_x_heads = layers * heads;
    const Uint128 token_bits_scaled = mul_u64(layers_x_heads, per_head_bits_scaled);

    std::uint64_t bytes_per_token = 0;
    if (!ceil_div_u128_by_u32_to_u64(token_bits_scaled, byte_den, bytes_per_token) ||
        bytes_per_token == 0) {
        return {};
    }

    Uint128 block_bits_scaled{};
    if (!mul_u128_u64(token_bits_scaled, block_tokens, block_bits_scaled)) {
        return {};
    }

    std::uint64_t bytes_per_block = 0;
    if (!ceil_div_u128_by_u32_to_u64(block_bits_scaled, byte_den, bytes_per_block) ||
        bytes_per_block == 0) {
        return {};
    }

    const BlockCount total_blocks = kv_budget_bytes / bytes_per_block;
    if (total_blocks > std::numeric_limits<std::uint64_t>::max() / block_tokens) {
        return {};
    }
    const std::uint64_t total_token_capacity = total_blocks * block_tokens;

    std::uint32_t max_concurrent_sequences = 0;
    if (target_context_tokens > 0) {
        const std::uint64_t blocks_per_seq = 1ull + (target_context_tokens - 1ull) / block_tokens;
        const Uint128 kv_bytes_per_seq = mul_u64(blocks_per_seq, bytes_per_block);
        Uint128 total_bytes_per_seq{};
        if (add_u128(kv_bytes_per_seq, Uint128{0, geometry.per_sequence_recurrent_bytes},
                     total_bytes_per_seq) &&
            total_bytes_per_seq.hi == 0 && total_bytes_per_seq.lo > 0) {
            const std::uint64_t seqs = kv_budget_bytes / total_bytes_per_seq.lo;
            constexpr std::uint64_t kMaxU32 = std::numeric_limits<std::uint32_t>::max();
            max_concurrent_sequences = static_cast<std::uint32_t>(std::min(seqs, kMaxU32));
        }
    }

    VramKvCapacityEstimate est;
    est.bytes_per_token = bytes_per_token;
    est.bytes_per_block = bytes_per_block;
    est.total_blocks = total_blocks;
    est.total_token_capacity = total_token_capacity;
    est.max_concurrent_sequences = max_concurrent_sequences;
    est.valid = true;
    return est;
}

bool FixedKvCapacity::try_reserve(RequestId id, BlockCount blocks) {
    if (blocks == 0) return true;
    if (blocks > total_ - used_) return false;
    BlockCount& entry = held_[id];
    entry += blocks;
    used_ += blocks;
    peak_ = std::max(peak_, used_);
    return true;
}

void FixedKvCapacity::release(RequestId id) {
    auto it = held_.find(id);
    if (it == held_.end()) return;
    used_ -= it->second;
    held_.erase(it);
}

BlockCount FixedKvCapacity::blocks_held(RequestId id) const noexcept {
    auto it = held_.find(id);
    return it == held_.end() ? 0 : it->second;
}

bool DynamicKvCapacity::try_reserve(RequestId id, BlockCount blocks) {
    if (blocks == 0) return true;
    if (blocks > free_blocks()) return false;
    BlockCount& entry = held_[id];
    entry += blocks;
    used_ += blocks;
    peak_ = std::max(peak_, used_);
    return true;
}

void DynamicKvCapacity::release(RequestId id) {
    auto it = held_.find(id);
    if (it == held_.end()) return;
    used_ -= it->second;
    held_.erase(it);
}

BlockCount DynamicKvCapacity::blocks_held(RequestId id) const noexcept {
    auto it = held_.find(id);
    return it == held_.end() ? 0 : it->second;
}

void DynamicKvCapacity::set_total_blocks(BlockCount new_total) noexcept {
    if (new_total == total_) return;
    total_ = new_total;
    ++resize_count_;
}

void DynamicKvCapacity::update_from_vram_budget(std::uint64_t kv_budget_bytes,
                                                const KvGeometry& geometry) noexcept {
    if (geometry.block_size_tokens == 0 || block_size_ == 0) {
        return;
    }
    KvGeometry aligned = geometry;
    aligned.block_size_tokens = block_size_;
    const VramKvCapacityEstimate est =
        estimate_kv_blocks_for_vram_budget(kv_budget_bytes, aligned);
    if (!est.valid) {
        return;
    }
    set_total_blocks(est.total_blocks);
}

}  // namespace sonder::inference::scheduler
