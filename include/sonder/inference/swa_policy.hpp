// Capability-gated compact sliding-window attention (SWA) KV policy and
// fail-closed rollback watermark guard (Item #26).
//
// Compact sliding-window KV (llama.cpp `swa_full = false`) reduces KV memory
// on qualified SWA architectures by retaining only a bounded rolling window of
// rows per SWA layer. Once a sequence advances past the retained window, older
// SWA rows are overwritten. Rewinding, trimming a cached prefix, or rolling
// back rejected speculative tokens to a position whose attention window
// reaches into overwritten rows would silently continue over corrupt/missing
// SWA state.
//
// `SwaKvMode::Full` is the safe default. `SwaKvMode::CompactIfQualified` is
// opt-in and paired with `SwaRollbackGuard`, which tracks the valid-history
// watermark (`min_retained_pos`) monotonically until full recomputation and
// fails closed whenever a target rollback/reuse position falls below it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>

namespace sonder::inference {

enum class SwaKvMode {
    // Default: allocate full-context SWA KV (`swa_full = true`). Arbitrary
    // prefix trims, rewinds, and speculative rollbacks stay valid.
    Full = 0,
    // Opt-in: allocate compact sliding-window KV (`swa_full = false`) only
    // when the model architecture and runtime qualify, with mandatory
    // `min_retained_pos` rollback guarding.
    CompactIfQualified = 1,
};

[[nodiscard]] inline std::string_view to_string(SwaKvMode mode) noexcept {
    switch (mode) {
        case SwaKvMode::Full: return "full";
        case SwaKvMode::CompactIfQualified: return "compact_if_qualified";
    }
    return "unknown";
}

[[nodiscard]] inline bool parse_swa_kv_mode(std::string_view name, SwaKvMode& out) noexcept {
    const auto ieq = [](std::string_view a, std::string_view b) noexcept {
        if (a.size() != b.size()) return false;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const char ca = (a[i] >= 'A' && a[i] <= 'Z') ? static_cast<char>(a[i] - 'A' + 'a') : a[i];
            const char cb = (b[i] >= 'A' && b[i] <= 'Z') ? static_cast<char>(b[i] - 'A' + 'a') : b[i];
            if (ca != cb) return false;
        }
        return true;
    };
    if (ieq(name, "full")) {
        out = SwaKvMode::Full;
        return true;
    }
    if (ieq(name, "compact_if_qualified")) {
        out = SwaKvMode::CompactIfQualified;
        return true;
    }
    return false;
}

// Per-sequence valid-history watermark tracker for compact sliding-window KV.
//
// Let `W = sliding_window` and `C = W + extra_retained_slack` be the physical
// retained ring capacity (in llama.cpp compact SWA, `extra_retained_slack` is
// at most `n_ubatch`, or 0 when conservative). Once the sequence reaches length
// `L` (positions `0..L-1`), rows `< oldest_retained_row` have been overwritten,
// where:
//   oldest_retained_row = max(observed_pos_min, L > C ? L - C : 0)
//
// Decoding a new token at `target_pos` requires every SWA row in
// `[max(0, target_pos - W), target_pos - 1]` to be intact. That holds iff
// `oldest_retained_row == 0` OR `target_pos >= oldest_retained_row + W`.
// Thus `min_retained_pos`:
//   - is `0` while no rows have been overwritten (`oldest_retained_row == 0`)
//   - is `oldest_retained_row + W` once any row has been overwritten
//
// Any rewind, prefix reuse cut, or speculative rejection targeting a position
// `< min_retained_pos` fails closed (returns false and resets the sequence
// watermark for full recomputation).
class SwaRollbackGuard {
public:
    void Configure(SwaKvMode mode, std::uint64_t sliding_window, std::uint64_t extra_retained_slack = 0,
                   bool architecture_qualified = false) noexcept {
        mode_ = mode;
        sliding_window_ = sliding_window;
        extra_retained_slack_ = extra_retained_slack;
        compact_active_ =
            (mode == SwaKvMode::CompactIfQualified) && architecture_qualified && (sliding_window > 0);
        Reset();
    }

    // Clears sequence position and watermark after a full KV clear / reset.
    // Preserves cumulative diagnostic counters.
    void Reset() noexcept {
        seq_len_ = 0;
        oldest_retained_row_ = 0;
        min_retained_pos_ = 0;
    }

    void ResetCounters() noexcept {
        rollback_checks_ = 0;
        rollback_rejections_ = 0;
    }

    // Advances the high-water sequence length (exclusive: positions 0..new_seq_len-1).
    void OnAdvance(std::uint64_t new_seq_len) noexcept {
        if (new_seq_len > seq_len_) {
            seq_len_ = new_seq_len;
            RecomputeWatermark();
        }
    }

    // Incorporates a live backend lower-bound observation (e.g.
    // `llama_memory_seq_pos_min(mem, seq_id)`). Ignored when negative (empty
    // sequence) or when compact SWA is inactive. Monotonic until Reset().
    void ObserveBackendPosMin(std::int64_t pos_min) noexcept {
        if (!compact_active_ || pos_min <= 0) {
            return;
        }
        const auto observed = static_cast<std::uint64_t>(pos_min);
        if (observed > oldest_retained_row_) {
            oldest_retained_row_ = observed;
        }
        RecomputeWatermark();
    }

    // Returns true if rewinding/cutting to `target_pos` (keeping [0, target_pos)
    // and discarding [target_pos, inf)) preserves all SWA rows needed to attend
    // at `target_pos`. Returns false (fail-closed) when `target_pos < min_retained_pos`.
    [[nodiscard]] bool CanRollbackTo(std::uint64_t target_pos) noexcept {
        ++rollback_checks_;
        if (!compact_active_) {
            return true;
        }
        if (target_pos < min_retained_pos_) {
            ++rollback_rejections_;
            return false;
        }
        return true;
    }

    // Checks `CanRollbackTo(target_pos)`. On success, updates `seq_len_` to
    // `min(seq_len_, target_pos)` while keeping `min_retained_pos_` monotonic
    // (already-overwritten rows cannot be recovered). On failure, resets the
    // watermark for full recomputation and returns false.
    [[nodiscard]] bool CheckAndApplyRollback(std::uint64_t target_pos) noexcept {
        if (!CanRollbackTo(target_pos)) {
            Reset();
            return false;
        }
        if (target_pos < seq_len_) {
            seq_len_ = target_pos;
        }
        return true;
    }

    [[nodiscard]] SwaKvMode mode() const noexcept { return mode_; }
    [[nodiscard]] bool compact_active() const noexcept { return compact_active_; }
    [[nodiscard]] std::uint64_t sliding_window() const noexcept { return sliding_window_; }
    [[nodiscard]] std::uint64_t extra_retained_slack() const noexcept { return extra_retained_slack_; }
    [[nodiscard]] std::uint64_t retained_capacity() const noexcept {
        return SaturatingAdd(sliding_window_, extra_retained_slack_);
    }
    [[nodiscard]] std::uint64_t seq_len() const noexcept { return seq_len_; }
    [[nodiscard]] std::uint64_t oldest_retained_row() const noexcept { return oldest_retained_row_; }
    [[nodiscard]] std::uint64_t min_retained_pos() const noexcept { return min_retained_pos_; }
    [[nodiscard]] std::uint64_t rollback_checks() const noexcept { return rollback_checks_; }
    [[nodiscard]] std::uint64_t rollback_rejections() const noexcept { return rollback_rejections_; }

private:
    static std::uint64_t SaturatingAdd(std::uint64_t a, std::uint64_t b) noexcept {
        return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
    }

    void RecomputeWatermark() noexcept {
        if (!compact_active_ || sliding_window_ == 0) {
            oldest_retained_row_ = 0;
            min_retained_pos_ = 0;
            return;
        }
        const std::uint64_t cap = retained_capacity();
        const std::uint64_t theoretical_oldest = (seq_len_ > cap) ? (seq_len_ - cap) : 0;
        if (theoretical_oldest > oldest_retained_row_) {
            oldest_retained_row_ = theoretical_oldest;
        }
        min_retained_pos_ =
            (oldest_retained_row_ == 0) ? 0 : SaturatingAdd(oldest_retained_row_, sliding_window_);
    }

    SwaKvMode mode_ = SwaKvMode::Full;
    bool compact_active_ = false;
    std::uint64_t sliding_window_ = 0;
    std::uint64_t extra_retained_slack_ = 0;
    std::uint64_t seq_len_ = 0;
    std::uint64_t oldest_retained_row_ = 0;
    std::uint64_t min_retained_pos_ = 0;
    std::uint64_t rollback_checks_ = 0;
    std::uint64_t rollback_rejections_ = 0;
};

}  // namespace sonder::inference
