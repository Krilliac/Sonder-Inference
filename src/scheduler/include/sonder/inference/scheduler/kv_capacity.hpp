// Abstract KV-capacity interface used for admission control, plus
// GPU-VRAM-budgeted KV block capacity estimation and dynamic pool sizing.
//
// The scheduler never touches the cache implementation (src/cache/). The
// engine adapts the real KV block manager to this interface; tests use
// FixedKvCapacity or DynamicKvCapacity.
#pragma once

#include "sonder/inference/scheduler/types.hpp"

#include <cstdint>
#include <optional>
#include <string_view>
#include <unordered_map>

namespace sonder::inference::scheduler {

class KvCapacity {
public:
    virtual ~KvCapacity() = default;

    /// Tokens stored per logical KV block.
    [[nodiscard]] virtual TokenCount block_size_tokens() const noexcept = 0;
    /// Total blocks the scheduler may ever use (for "never fits" checks).
    [[nodiscard]] virtual BlockCount total_blocks() const noexcept = 0;
    /// Blocks currently free for new reservations.
    [[nodiscard]] virtual BlockCount free_blocks() const noexcept = 0;
    /// Blocks held above `total_blocks()` when a dynamic pool shrinks below
    /// current usage (0 when within budget or for fixed pools).
    [[nodiscard]] virtual BlockCount over_budget_blocks() const noexcept { return 0; }
    /// Reserve `blocks` additional blocks for `id`. Must be all-or-nothing.
    [[nodiscard]] virtual bool try_reserve(RequestId id, BlockCount blocks) = 0;
    /// Release every block held by `id` (no-op if none).
    virtual void release(RequestId id) = 0;
    /// Blocks currently held by `id`.
    [[nodiscard]] virtual BlockCount blocks_held(RequestId id) const noexcept = 0;
};

/// Blocks needed to hold `tokens` tokens.
[[nodiscard]] constexpr BlockCount blocks_for_tokens(TokenCount tokens,
                                                     TokenCount block_size) noexcept {
    return block_size == 0 ? 0
                           : (static_cast<BlockCount>(tokens) + block_size - 1) / block_size;
}

/// Supported KV-cache element quantization formats (matching GGML / llama.cpp
/// and PrismML KV cache type names).
enum class KvQuantFormat : std::uint8_t {
    f32,
    f16,
    bf16,
    f8_e4m3,
    q8_0,
    q5_1,
    q5_0,
    q4_0,
    q2_0,
    pq2_0m,
};

/// Exact rational bits-per-element (`num / den`) for a KV quantization format.
struct KvBitsPerElement {
    std::uint32_t num = 0;
    std::uint32_t den = 1;

    [[nodiscard]] constexpr bool operator==(const KvBitsPerElement& other) const noexcept {
        return num == other.num && den == other.den;
    }
    [[nodiscard]] constexpr bool operator!=(const KvBitsPerElement& other) const noexcept {
        return !(*this == other);
    }
};

/// Parse canonical KV quantization format names case-insensitively.
[[nodiscard]] std::optional<KvQuantFormat> parse_kv_quant_format(std::string_view name) noexcept;

/// Canonical lowercase name for `fmt` (or `"unknown"` for out-of-range values).
[[nodiscard]] const char* kv_quant_format_name(KvQuantFormat fmt) noexcept;

/// Exact rational bits per element matching GGML block layouts:
///   f32     =  4 B /   1 = 32/1  (32.0)
///   f16     =  2 B /   1 = 16/1  (16.0)
///   bf16    =  2 B /   1 = 16/1  (16.0)
///   f8_e4m3 =  1 B /   1 =  8/1  ( 8.0)
///   q8_0    = 34 B /  32 = 17/2  ( 8.5)
///   q5_1    = 24 B /  32 =  6/1  ( 6.0)
///   q5_0    = 22 B /  32 = 11/2  ( 5.5)
///   q4_0    = 18 B /  32 =  9/2  ( 4.5)
///   q2_0    = 18 B /  64 =  9/4  ( 2.25, GGML_TYPE_Q2_0)
///   pq2_0m  = 36 B / 128 =  9/4  ( 2.25, GGML_TYPE_PQ2_0M)
[[nodiscard]] constexpr KvBitsPerElement kv_quant_bits_per_element(KvQuantFormat fmt) noexcept {
    switch (fmt) {
        case KvQuantFormat::f32: return KvBitsPerElement{32u, 1u};
        case KvQuantFormat::f16: return KvBitsPerElement{16u, 1u};
        case KvQuantFormat::bf16: return KvBitsPerElement{16u, 1u};
        case KvQuantFormat::f8_e4m3: return KvBitsPerElement{8u, 1u};
        case KvQuantFormat::q8_0: return KvBitsPerElement{17u, 2u};
        case KvQuantFormat::q5_1: return KvBitsPerElement{6u, 1u};
        case KvQuantFormat::q5_0: return KvBitsPerElement{11u, 2u};
        case KvQuantFormat::q4_0: return KvBitsPerElement{9u, 2u};
        case KvQuantFormat::q2_0: return KvBitsPerElement{9u, 4u};
        case KvQuantFormat::pq2_0m: return KvBitsPerElement{9u, 4u};
    }
    return KvBitsPerElement{0u, 1u};
}

/// Model KV-cache geometry used to translate a VRAM byte budget into logical
/// KV block capacity.
struct KvGeometry {
    /// Attention KV layers only (e.g. 16 for 64-layer hybrid Qwen3.5/Bonsai-2,
    /// or full layer count for attention-only models).
    std::uint32_t kv_layers = 0;
    std::uint32_t kv_heads = 0;
    std::uint32_t head_dim_k = 0;
    /// Value head dimension; if 0, defaults to `head_dim_k`.
    std::uint32_t head_dim_v = 0;
    TokenCount block_size_tokens = 16;
    KvQuantFormat k_format = KvQuantFormat::f16;
    KvQuantFormat v_format = KvQuantFormat::f16;
    /// Optional non-KV recurrent/Mamba/GDN state bytes required per concurrent
    /// sequence on hybrid architectures.
    std::uint64_t per_sequence_recurrent_bytes = 0;
};

/// Result of estimating KV block capacity from a VRAM byte budget.
struct VramKvCapacityEstimate {
    std::uint64_t bytes_per_token = 0;
    std::uint64_t bytes_per_block = 0;
    BlockCount total_blocks = 0;
    std::uint64_t total_token_capacity = 0;
    /// Maximum concurrent sequences at `target_context_tokens` (0 when
    /// `target_context_tokens == 0` or a single sequence does not fit),
    /// jointly bounded by per-sequence KV blocks and
    /// `KvGeometry::per_sequence_recurrent_bytes`.
    std::uint32_t max_concurrent_sequences = 0;
    bool valid = false;
};

/// Estimate KV block capacity that fits within `kv_budget_bytes`.
///
/// Uses exact overflow-checked 128-bit integer arithmetic and ceiling division
/// for `bytes_per_token` and `bytes_per_block` so block capacity never
/// overestimates fit. Returns `{}` (`valid == false`) on invalid geometry
/// (`kv_layers == 0`, `kv_heads == 0`, `head_dim_k == 0`,
/// `block_size_tokens == 0`, or invalid quant format) or arithmetic overflow.
[[nodiscard]] VramKvCapacityEstimate estimate_kv_blocks_for_vram_budget(
    std::uint64_t kv_budget_bytes,
    const KvGeometry& geometry,
    std::uint64_t target_context_tokens = 0) noexcept;

/// Simple fixed-size pool; reference implementation for tests/simulation.
class FixedKvCapacity final : public KvCapacity {
public:
    FixedKvCapacity(BlockCount total_blocks, TokenCount block_size_tokens) noexcept
        : total_(total_blocks), block_size_(block_size_tokens) {}

    [[nodiscard]] TokenCount block_size_tokens() const noexcept override { return block_size_; }
    [[nodiscard]] BlockCount total_blocks() const noexcept override { return total_; }
    [[nodiscard]] BlockCount free_blocks() const noexcept override { return total_ - used_; }
    [[nodiscard]] bool try_reserve(RequestId id, BlockCount blocks) override;
    void release(RequestId id) override;
    [[nodiscard]] BlockCount blocks_held(RequestId id) const noexcept override;

    [[nodiscard]] BlockCount used_blocks() const noexcept { return used_; }
    [[nodiscard]] BlockCount peak_used_blocks() const noexcept { return peak_; }

private:
    BlockCount total_;
    TokenCount block_size_;
    BlockCount used_ = 0;
    BlockCount peak_ = 0;
    std::unordered_map<RequestId, BlockCount> held_;
};

/// Dynamically resizable KV block pool with safe drain-on-shrink semantics.
///
/// When `total_blocks()` shrinks below `used_blocks()`, held reservations stay
/// intact without underflow: `free_blocks()` returns 0, `is_over_budget()`
/// returns true, `over_budget_blocks()` returns `used_ - total_`, and new
/// non-zero `try_reserve()` calls fail until releases drain usage below
/// `total_blocks()`.
///
/// Thread safety: single-threaded / externally synchronized. Callers must
/// serialize `set_total_blocks()` and `update_from_vram_budget()` with all
/// `Scheduler` operations (for example, between `complete_step()` and the
/// next `plan_step()`).
class DynamicKvCapacity final : public KvCapacity {
public:
    DynamicKvCapacity(BlockCount total_blocks, TokenCount block_size_tokens) noexcept
        : total_(total_blocks), block_size_(block_size_tokens) {}

    [[nodiscard]] TokenCount block_size_tokens() const noexcept override { return block_size_; }
    [[nodiscard]] BlockCount total_blocks() const noexcept override { return total_; }
    [[nodiscard]] BlockCount free_blocks() const noexcept override {
        return total_ > used_ ? total_ - used_ : 0;
    }
    [[nodiscard]] BlockCount over_budget_blocks() const noexcept override {
        return used_ > total_ ? used_ - total_ : 0;
    }
    [[nodiscard]] bool try_reserve(RequestId id, BlockCount blocks) override;
    void release(RequestId id) override;
    [[nodiscard]] BlockCount blocks_held(RequestId id) const noexcept override;

    /// Update total capacity. Increments `resize_count()` when `new_total`
    /// differs from the current `total_blocks()`.
    void set_total_blocks(BlockCount new_total) noexcept;

    /// Recompute total capacity from a VRAM KV budget and geometry (aligned to
    /// `block_size_tokens()`). No-op if geometry is invalid,
    /// `block_size_tokens()` is 0, or estimation overflows.
    void update_from_vram_budget(std::uint64_t kv_budget_bytes,
                                 const KvGeometry& geometry) noexcept;

    [[nodiscard]] BlockCount used_blocks() const noexcept { return used_; }
    [[nodiscard]] BlockCount peak_used_blocks() const noexcept { return peak_; }
    [[nodiscard]] bool is_over_budget() const noexcept { return used_ > total_; }
    [[nodiscard]] std::uint64_t resize_count() const noexcept { return resize_count_; }

private:
    BlockCount total_;
    TokenCount block_size_;
    BlockCount used_ = 0;
    BlockCount peak_ = 0;
    std::uint64_t resize_count_ = 0;
    std::unordered_map<RequestId, BlockCount> held_;
};

}  // namespace sonder::inference::scheduler
