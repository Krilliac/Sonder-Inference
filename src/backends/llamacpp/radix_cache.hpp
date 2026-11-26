// Radix Prefix Tree + Bounded L2 Host-RAM KV/SSM State Cache.
//
// Indexes cached token prefixes in a compressed radix tree mapped to serialized
// llama.cpp sequence state blobs stored in a strictly bounded host-RAM budget
// (`cache_ram_mib`). Enforces value-aware global eviction BEFORE allocation so
// host commit headroom never spikes past the configured cap, supports RAII
// in-flight hard node pinning (`LookupAndPin` / `PinnedRadixLookup`), bounded
// soft session leases with strict per-session fairness caps and automatic TTL
// expiry cleanup, multi-branch shared prefix invalidation (`ErasePrefix`),
// measured restore-vs-recompute gating (`RestoreCostGate`), and thread-safe
// leaf-locked access.
#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"

namespace sonder::backends::llamacpp {

// Monotonic steady-clock timestamp in milliseconds (>= 1 so 0 remains "unset").
[[nodiscard]] inline std::uint64_t SteadyNowMs() noexcept {
    const auto now = std::chrono::steady_clock::now().time_since_epoch();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
    return ms > 0 ? static_cast<std::uint64_t>(ms) : 1u;
}

using RadixClockMsFn = std::function<std::uint64_t()>;

// Configuration for the measured host-L2 restore-vs-recompute cost gate.
struct RestoreCostGateConfig {
    // Conservative cold-start defaults used before live measurements arrive:
    // ~1.5 ms/MiB (~667 MiB/s deserialize/restore) and ~0.25 ms/token (~4000 tok/s prefill).
    double default_ms_per_mib_restore = 1.5;
    double default_ms_per_token_prefill = 0.25;
    // Rolling EWMA smoothing factor in (0, 1].
    double ewma_alpha = 0.25;
    // Optional minimum positive net savings margin (ms). 0.0 rejects whenever
    // predicted T_restore >= predicted T_recompute.
    double min_net_savings_ms = 0.0;
    std::size_t min_sample_tokens = 1;
    std::size_t min_sample_bytes = 1;
};

struct RestoreCostDecision {
    bool should_restore = false;
    double predicted_restore_ms = 0.0;
    double predicted_recompute_ms = 0.0;
    double net_savings_ms = 0.0;
};

struct RestoreCostRates {
    double ms_per_mib_restore = 1.5;
    double ms_per_token_prefill = 0.25;
    std::uint64_t restore_samples = 0;
    std::uint64_t prefill_samples = 0;
};

struct RestoreCostGateStats {
    double ms_per_mib_restore = 1.5;
    double ms_per_token_prefill = 0.25;
    std::uint64_t restore_samples = 0;
    std::uint64_t prefill_samples = 0;
    std::uint64_t restore_accepted = 0;
    std::uint64_t restore_skipped_unprofitable = 0;
};

// Measured restore-vs-recompute cost gate.
// Tracks rolling observed `ms_per_mib_restore` and `ms_per_token_prefill` via
// EWMA (with safe cold-start defaults) and refuses host L2 restores whenever
// predicted `T_restore >= T_recompute`.
class RestoreCostGate {
public:
    static constexpr double kBytesPerMiB = 1024.0 * 1024.0;

    explicit RestoreCostGate(RestoreCostGateConfig config = {})
        : config_(SanitizeConfig(config)),
          ms_per_mib_restore_(config_.default_ms_per_mib_restore),
          ms_per_token_prefill_(config_.default_ms_per_token_prefill) {}

    void SetConfig(RestoreCostGateConfig config) noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        config_ = SanitizeConfig(config);
        if (restore_samples_ == 0) {
            ms_per_mib_restore_ = config_.default_ms_per_mib_restore;
        }
        if (prefill_samples_ == 0) {
            ms_per_token_prefill_ = config_.default_ms_per_token_prefill;
        }
    }

    void Reset() noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        ms_per_mib_restore_ = config_.default_ms_per_mib_restore;
        ms_per_token_prefill_ = config_.default_ms_per_token_prefill;
        restore_samples_ = 0;
        prefill_samples_ = 0;
        restore_accepted_ = 0;
        restore_skipped_unprofitable_ = 0;
    }

    void ObserveRestore(std::size_t payload_bytes, double restore_ms) noexcept {
        if (!std::isfinite(restore_ms) || restore_ms <= 0.0) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (payload_bytes < config_.min_sample_bytes) return;
        const double mib = static_cast<double>(payload_bytes) / kBytesPerMiB;
        if (!(mib > 0.0) || !std::isfinite(mib)) return;
        const double sample = restore_ms / mib;
        if (!std::isfinite(sample) || sample <= 0.0) return;
        if (restore_samples_ == 0) {
            ms_per_mib_restore_ = sample;
        } else {
            ms_per_mib_restore_ =
                config_.ewma_alpha * sample + (1.0 - config_.ewma_alpha) * ms_per_mib_restore_;
        }
        ++restore_samples_;
    }

    void ObservePrefill(std::size_t prefilled_tokens, double prefill_ms) noexcept {
        if (!std::isfinite(prefill_ms) || prefill_ms <= 0.0) return;
        std::lock_guard<std::mutex> lock(mu_);
        if (prefilled_tokens < config_.min_sample_tokens) return;
        const double sample = prefill_ms / static_cast<double>(prefilled_tokens);
        if (!std::isfinite(sample) || sample <= 0.0) return;
        if (prefill_samples_ == 0) {
            ms_per_token_prefill_ = sample;
        } else {
            ms_per_token_prefill_ =
                config_.ewma_alpha * sample + (1.0 - config_.ewma_alpha) * ms_per_token_prefill_;
        }
        ++prefill_samples_;
    }

    [[nodiscard]] RestoreCostDecision Evaluate(std::size_t matched_tokens,
                                               std::size_t payload_bytes) const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return EvaluateLocked(matched_tokens, payload_bytes);
    }

    // Evaluates restore profitability and records telemetry counters
    // (`restore_accepted` vs `restore_skipped_unprofitable`).
    [[nodiscard]] bool should_restore(std::size_t matched_tokens, std::size_t payload_bytes) noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        const RestoreCostDecision d = EvaluateLocked(matched_tokens, payload_bytes);
        if (d.should_restore) {
            ++restore_accepted_;
        } else {
            ++restore_skipped_unprofitable_;
        }
        return d.should_restore;
    }

    [[nodiscard]] double PredictRestoreMs(std::size_t payload_bytes) const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return ms_per_mib_restore_ * (static_cast<double>(payload_bytes) / kBytesPerMiB);
    }

    [[nodiscard]] double PredictRecomputeMs(std::size_t matched_tokens) const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return ms_per_token_prefill_ * static_cast<double>(matched_tokens);
    }

    [[nodiscard]] RestoreCostRates SnapshotRates() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return {ms_per_mib_restore_, ms_per_token_prefill_, restore_samples_, prefill_samples_};
    }

    [[nodiscard]] RestoreCostGateStats stats() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return {ms_per_mib_restore_, ms_per_token_prefill_, restore_samples_,
                prefill_samples_,    restore_accepted_,     restore_skipped_unprofitable_};
    }

    [[nodiscard]] double ms_per_mib_restore() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return ms_per_mib_restore_;
    }
    [[nodiscard]] double ms_per_token_prefill() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return ms_per_token_prefill_;
    }
    [[nodiscard]] std::uint64_t restore_samples() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return restore_samples_;
    }
    [[nodiscard]] std::uint64_t prefill_samples() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return prefill_samples_;
    }
    [[nodiscard]] std::uint64_t restore_accepted() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return restore_accepted_;
    }
    [[nodiscard]] std::uint64_t restore_skipped_unprofitable() const noexcept {
        std::lock_guard<std::mutex> lock(mu_);
        return restore_skipped_unprofitable_;
    }

private:
    static RestoreCostGateConfig SanitizeConfig(RestoreCostGateConfig cfg) noexcept {
        if (!std::isfinite(cfg.default_ms_per_mib_restore) || cfg.default_ms_per_mib_restore <= 0.0) {
            cfg.default_ms_per_mib_restore = 1.5;
        }
        if (!std::isfinite(cfg.default_ms_per_token_prefill) || cfg.default_ms_per_token_prefill <= 0.0) {
            cfg.default_ms_per_token_prefill = 0.25;
        }
        if (!std::isfinite(cfg.ewma_alpha) || cfg.ewma_alpha <= 0.0) {
            cfg.ewma_alpha = 0.25;
        } else if (cfg.ewma_alpha > 1.0) {
            cfg.ewma_alpha = 1.0;
        }
        if (!std::isfinite(cfg.min_net_savings_ms) || cfg.min_net_savings_ms < 0.0) {
            cfg.min_net_savings_ms = 0.0;
        }
        if (cfg.min_sample_tokens == 0) cfg.min_sample_tokens = 1;
        if (cfg.min_sample_bytes == 0) cfg.min_sample_bytes = 1;
        return cfg;
    }

    [[nodiscard]] RestoreCostDecision EvaluateLocked(std::size_t matched_tokens,
                                                     std::size_t payload_bytes) const noexcept {
        RestoreCostDecision d;
        d.predicted_recompute_ms = ms_per_token_prefill_ * static_cast<double>(matched_tokens);
        d.predicted_restore_ms = ms_per_mib_restore_ * (static_cast<double>(payload_bytes) / kBytesPerMiB);
        d.net_savings_ms = d.predicted_recompute_ms - d.predicted_restore_ms;
        if (matched_tokens == 0 || payload_bytes == 0 || !std::isfinite(d.predicted_recompute_ms) ||
            !std::isfinite(d.predicted_restore_ms) || !std::isfinite(d.net_savings_ms)) {
            d.should_restore = false;
            return d;
        }
        if (config_.min_net_savings_ms > 0.0) {
            d.should_restore =
                (d.predicted_restore_ms < d.predicted_recompute_ms) && (d.net_savings_ms >= config_.min_net_savings_ms);
        } else {
            d.should_restore = (d.predicted_restore_ms < d.predicted_recompute_ms);
        }
        return d;
    }

    mutable std::mutex mu_;
    RestoreCostGateConfig config_{};
    double ms_per_mib_restore_ = 1.5;
    double ms_per_token_prefill_ = 0.25;
    std::uint64_t restore_samples_ = 0;
    std::uint64_t prefill_samples_ = 0;
    std::uint64_t restore_accepted_ = 0;
    std::uint64_t restore_skipped_unprofitable_ = 0;
};

// Value-aware physical Radix cache retention policy.
struct RadixCachePolicy {
    // Bounded soft-session lease TTL (ms). Non-empty session keys receive a
    // soft lease capped at `max_soft_lease_ttl_ms` (0 disables soft leases).
    std::uint64_t default_soft_lease_ttl_ms = 30'000;
    std::uint64_t max_soft_lease_ttl_ms = 120'000;

    // Strict per-session fairness cap on soft-leased payload bytes: no single
    // session may hold unexpired soft leases covering more than this fraction
    // of `max_bytes` (or `max_session_soft_lease_bytes` when non-zero).
    double max_session_soft_lease_fraction = 0.50;
    std::size_t max_session_soft_lease_bytes = 0;

    // Optional strict upper bound on total cached payload bytes per session
    // as a fraction of `max_bytes` (1.0 allows a lone session to use idle RAM
    // while still bounding its soft leases via `max_session_soft_lease_fraction`).
    double max_session_cache_fraction = 1.0;

    // Multiplicative value boost applied while an entry's soft lease is unexpired.
    double soft_lease_boost = 4.0;

    // Recency half-life (ms) for decaying `P(reuse_i)` with idle age (0 disables decay).
    double recency_half_life_ms = 60'000.0;

    // Weights for branching importance (`shared_prefix_children`) and observed reuse.
    double branch_importance_weight = 0.50;
    double reuse_count_weight = 0.75;
    double base_reuse_prob = 0.20;

    // Restore-vs-recompute cost gate configuration.
    RestoreCostGateConfig restore_cost_gate{};
};

struct RadixCacheEntry {
    std::vector<std::uint8_t> state_blob;
    std::vector<std::uint8_t> dft_state_blob;
    std::vector<float> mtp_pending_h;
    std::string session_key;

    // Optional per-entry retention hints (all default to automatic policy behavior).
    bool request_soft_lease = true;
    std::uint64_t soft_lease_ttl_ms = 0;  // 0 => policy.default_soft_lease_ttl_ms
    double estimated_recompute_ms = 0.0;  // 0 => predict via RestoreCostGate
    double last_restore_ms = 0.0;         // 0 => predict via RestoreCostGate

    [[nodiscard]] std::size_t ByteSize() const noexcept {
        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        auto sat_add = [](std::size_t a, std::size_t b) noexcept -> std::size_t {
            return a > kMax - b ? kMax : a + b;
        };
        const std::size_t h_bytes =
            mtp_pending_h.size() > kMax / sizeof(float) ? kMax : mtp_pending_h.size() * sizeof(float);
        return sat_add(sat_add(state_blob.size(), dft_state_blob.size()), h_bytes);
    }
};

struct RadixLookupResult {
    // Number of prompt tokens matched (clamped to caller's max_match_tokens).
    std::size_t matched_tokens = 0;
    // Exact token length encoded in `entry->state_blob` (>= matched_tokens).
    // When `state_tokens > matched_tokens`, the caller must trim KV suffix
    // `[matched_tokens, -1)` after restoring `state_blob`.
    std::size_t state_tokens = 0;
    const RadixCacheEntry* entry = nullptr;

    [[nodiscard]] bool found() const noexcept { return entry != nullptr && matched_tokens > 0; }
};

// Read-only snapshot of per-entry retention signals and live value score.
struct RadixEntryMetrics {
    std::size_t payload_bytes = 0;
    std::size_t metadata_overhead_bytes = 0;
    std::size_t retained_bytes = 0;
    std::size_t token_count = 0;
    std::uint64_t reuse_count = 0;
    std::size_t shared_prefix_children = 0;
    std::uint64_t last_access_ms = 0;
    std::string session_id;
    bool soft_lease_active = false;
    std::uint64_t soft_lease_expires_ms = 0;
    double last_restore_ms = 0.0;
    double recompute_ms = 0.0;
    double value_score = 0.0;
    std::size_t pin_count = 0;
};

class RadixPrefixCache {
private:
    struct TransparentStringHash {
        using is_transparent = void;
        [[nodiscard]] std::size_t operator()(std::string_view sv) const noexcept {
            return std::hash<std::string_view>{}(sv);
        }
        [[nodiscard]] std::size_t operator()(const std::string& s) const noexcept {
            return std::hash<std::string_view>{}(s);
        }
    };

    using SessionByteMap =
        std::unordered_map<std::string, std::size_t, TransparentStringHash, std::equal_to<>>;

    struct Node;

    struct EntryRecord {
        RadixCacheEntry payload;
        std::size_t bytes = 0;
        std::size_t metadata_overhead_bytes = 0;
        std::size_t retained_bytes = 0;
        std::size_t token_count = 0;
        std::uint64_t reuse_count = 0;
        std::uint64_t last_access_ms = 0;
        std::string session_id;
        bool has_soft_lease = false;
        std::uint64_t soft_lease_expires_ms = 0;
        double last_restore_ms = 0.0;
        double recompute_ms = 0.0;
        std::size_t pin_count = 0;
        std::uint64_t lru_seq = 0;
        bool retired = false;
    };

    struct Node {
        Node* parent = nullptr;
        std::vector<Token> edge;
        std::size_t depth = 0;
        // Exact count of cached entries strictly below this node in its subtree.
        std::size_t cached_descendants = 0;
        std::unordered_map<Token, std::unique_ptr<Node>> children;
        std::shared_ptr<EntryRecord> record;
        std::list<Node*>::iterator lru_it;

        [[nodiscard]] bool has_entry() const noexcept { return record != nullptr; }
    };

    struct CacheState {
        mutable std::mutex mu;
        std::size_t max_bytes = 0;
        std::size_t used_bytes = 0;
        std::size_t active_metadata_bytes = 0;
        std::size_t retired_metadata_bytes = 0;
        std::size_t entry_count = 0;
        std::size_t active_pinned_count = 0;
        std::size_t active_pinned_bytes = 0;
        std::size_t retired_pinned_count = 0;
        std::size_t retired_pinned_bytes = 0;
        std::size_t active_soft_lease_count = 0;
        std::uint64_t lru_clock = 0;
        std::uint64_t evictions = 0;
        std::uint64_t hits = 0;
        std::uint64_t misses = 0;
        std::uint64_t soft_leases_expired = 0;
        std::uint64_t soft_leases_capped = 0;
        RadixCachePolicy policy{};
        RadixClockMsFn clock_ms;
        RestoreCostGate restore_gate{};
        SessionByteMap session_total_bytes;
        SessionByteMap session_soft_leased_bytes;
        std::unique_ptr<Node> root = std::make_unique<Node>();
        std::list<Node*> lru_list;
        std::vector<Node*> search_scratch;
    };

    [[nodiscard]] static std::size_t SaturatingAdd(std::size_t a, std::size_t b) noexcept {
        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        return a > kMax - b ? kMax : a + b;
    }

    [[nodiscard]] static std::uint64_t SaturatingAddU64(std::uint64_t a, std::uint64_t b) noexcept {
        constexpr std::uint64_t kMax = std::numeric_limits<std::uint64_t>::max();
        return a > kMax - b ? kMax : a + b;
    }

    [[nodiscard]] static std::size_t EstimateMetadataOverheadBytes(std::size_t token_count,
                                                                   std::size_t session_id_len) noexcept {
        constexpr std::size_t kMax = std::numeric_limits<std::size_t>::max();
        const std::size_t base = sizeof(Node) + sizeof(EntryRecord) + sizeof(void*) * 4u;
        const std::size_t tok_bytes =
            token_count > kMax / sizeof(Token) ? kMax : token_count * sizeof(Token);
        return SaturatingAdd(SaturatingAdd(base, tok_bytes), session_id_len);
    }

    [[nodiscard]] static RadixCachePolicy SanitizePolicy(RadixCachePolicy p) noexcept {
        if (p.max_soft_lease_ttl_ms == 0) {
            p.default_soft_lease_ttl_ms = 0;
        } else if (p.default_soft_lease_ttl_ms > p.max_soft_lease_ttl_ms) {
            p.default_soft_lease_ttl_ms = p.max_soft_lease_ttl_ms;
        }
        if (!std::isfinite(p.max_session_soft_lease_fraction)) {
            p.max_session_soft_lease_fraction = 0.50;
        } else {
            p.max_session_soft_lease_fraction = std::clamp(p.max_session_soft_lease_fraction, 0.0, 1.0);
        }
        if (!std::isfinite(p.max_session_cache_fraction) || p.max_session_cache_fraction <= 0.0) {
            p.max_session_cache_fraction = 1.0;
        } else if (p.max_session_cache_fraction > 1.0) {
            p.max_session_cache_fraction = 1.0;
        }
        if (!std::isfinite(p.soft_lease_boost) || p.soft_lease_boost < 1.0) {
            p.soft_lease_boost = 1.0;
        }
        if (!std::isfinite(p.recency_half_life_ms) || p.recency_half_life_ms < 0.0) {
            p.recency_half_life_ms = 60'000.0;
        }
        if (!std::isfinite(p.branch_importance_weight) || p.branch_importance_weight < 0.0) {
            p.branch_importance_weight = 0.50;
        }
        if (!std::isfinite(p.reuse_count_weight) || p.reuse_count_weight < 0.0) {
            p.reuse_count_weight = 0.75;
        }
        if (!std::isfinite(p.base_reuse_prob) || p.base_reuse_prob <= 0.0) {
            p.base_reuse_prob = 0.20;
        }
        return p;
    }

    [[nodiscard]] static std::uint64_t NowMsFromState(const CacheState& st) noexcept {
        if (st.clock_ms) {
            try {
                return st.clock_ms();
            } catch (...) {
                return SteadyNowMs();
            }
        }
        return SteadyNowMs();
    }

    [[nodiscard]] static std::uint64_t EffectiveSoftLeaseTtlMs(const RadixCachePolicy& policy,
                                                               std::uint64_t requested_ttl_ms) noexcept {
        if (policy.max_soft_lease_ttl_ms == 0) return 0;
        const std::uint64_t raw = requested_ttl_ms > 0 ? requested_ttl_ms : policy.default_soft_lease_ttl_ms;
        return std::min(raw, policy.max_soft_lease_ttl_ms);
    }

    [[nodiscard]] static std::size_t FractionCapBytes(std::size_t max_bytes, double fraction) noexcept {
        if (max_bytes == 0 || !std::isfinite(fraction) || fraction <= 0.0) return 0;
        if (fraction >= 1.0) return max_bytes;
        const double raw = std::floor(static_cast<double>(max_bytes) * fraction);
        if (!std::isfinite(raw) || raw <= 0.0) return 0;
        const double clamped = std::min(raw, static_cast<double>(max_bytes));
        return static_cast<std::size_t>(clamped);
    }

    [[nodiscard]] static std::size_t SessionSoftLeaseCapBytesFromState(const CacheState& st) noexcept {
        if (st.max_bytes == 0) return 0;
        std::size_t cap = FractionCapBytes(st.max_bytes, st.policy.max_session_soft_lease_fraction);
        if (st.policy.max_session_soft_lease_bytes > 0) {
            cap = std::min(cap, st.policy.max_session_soft_lease_bytes);
        }
        return std::min(cap, st.max_bytes);
    }

    [[nodiscard]] static std::size_t SessionTotalCapBytesFromState(const CacheState& st) noexcept {
        if (st.max_bytes == 0) return 0;
        return FractionCapBytes(st.max_bytes, st.policy.max_session_cache_fraction);
    }

    [[nodiscard]] static std::size_t GetSessionBytes(const SessionByteMap& map,
                                                     std::string_view session_id) noexcept {
        if (session_id.empty()) return 0;
        const auto it = map.find(session_id);
        return it != map.end() ? it->second : 0;
    }

    static void AddSessionBytes(SessionByteMap& map, const std::string& session_id, std::size_t delta) {
        if (session_id.empty() || delta == 0) return;
        map[session_id] += delta;
    }

    static void SubSessionBytes(SessionByteMap& map, std::string_view session_id, std::size_t delta) noexcept {
        if (session_id.empty() || delta == 0) return;
        const auto it = map.find(session_id);
        if (it == map.end()) return;
        if (it->second <= delta) {
            map.erase(it);
        } else {
            it->second -= delta;
        }
    }

    static void RevokeSoftLeaseFromState(CacheState& st, EntryRecord* rec) noexcept {
        if (rec == nullptr || !rec->has_soft_lease) return;
        rec->has_soft_lease = false;
        rec->soft_lease_expires_ms = 0;
        SubSessionBytes(st.session_soft_leased_bytes, rec->session_id, rec->bytes);
        if (st.active_soft_lease_count > 0) --st.active_soft_lease_count;
    }

    static std::size_t PurgeExpiredSoftLeasesFromState(CacheState& st, std::uint64_t now_ms) noexcept {
        if (st.active_soft_lease_count == 0) return 0;
        std::size_t expired = 0;
        for (Node* n : st.lru_list) {
            if (n == nullptr || !n->has_entry()) continue;
            EntryRecord& rec = *n->record;
            if (rec.has_soft_lease && now_ms >= rec.soft_lease_expires_ms) {
                RevokeSoftLeaseFromState(st, &rec);
                ++st.soft_leases_expired;
                ++expired;
            }
        }
        return expired;
    }

    // Computes V_i = P(reuse_i) * max(0, T_recompute_i - T_restore_i) / max(1, retained_bytes_i),
    // with soft-session-lease boost while unexpired and fairness pressure for over-cap sessions.
    [[nodiscard]] static double ComputeEntryValueScoreFromState(
        const CacheState& st, const Node* node, std::uint64_t now_ms, const RestoreCostRates& rates,
        std::size_t session_lease_cap, std::string_view incoming_session_id,
        std::size_t incoming_bytes) noexcept {
        if (node == nullptr || !node->has_entry()) return 0.0;
        const EntryRecord& rec = *node->record;

        // 1. Reuse probability P(reuse_i) in (0, 1] combining observed reuse,
        //    shared-prefix branching importance, and exponential recency decay.
        const double reuse_term =
            st.policy.reuse_count_weight * std::log1p(static_cast<double>(rec.reuse_count));
        const double branch_term =
            st.policy.branch_importance_weight * std::log1p(static_cast<double>(node->cached_descendants));
        const double demand = std::max(0.0, st.policy.base_reuse_prob + reuse_term + branch_term);
        double p_reuse = demand / (1.0 + demand);
        if (st.policy.recency_half_life_ms > 0.0 && now_ms > rec.last_access_ms) {
            const double age_ms = static_cast<double>(now_ms - rec.last_access_ms);
            p_reuse *= std::exp2(-age_ms / st.policy.recency_half_life_ms);
        }

        // 2. Expected net latency saved: max(0, T_recompute_i - T_restore_i).
        const double t_recompute =
            (rec.recompute_ms > 0.0 && std::isfinite(rec.recompute_ms))
                ? rec.recompute_ms
                : rates.ms_per_token_prefill * static_cast<double>(rec.token_count);
        const double t_restore =
            (rec.last_restore_ms > 0.0 && std::isfinite(rec.last_restore_ms))
                ? rec.last_restore_ms
                : rates.ms_per_mib_restore * (static_cast<double>(rec.bytes) / RestoreCostGate::kBytesPerMiB);
        const double net_saved_ms = std::max(0.0, t_recompute - t_restore);

        // 3. Base value per retained byte (including payload + estimated metadata overhead).
        const double denom = static_cast<double>(std::max<std::size_t>(1u, rec.retained_bytes));
        double score = (p_reuse * net_saved_ms) / denom;
        if (!std::isfinite(score) || score < 0.0) {
            score = 0.0;
        }

        // 4. Soft session lease boost while unexpired.
        const bool lease_active = rec.has_soft_lease && now_ms < rec.soft_lease_expires_ms;
        const bool is_incoming_session = !incoming_session_id.empty() && incoming_session_id == rec.session_id;
        const std::size_t sess_bytes = GetSessionBytes(st.session_total_bytes, rec.session_id);
        const std::size_t projected_sess_bytes =
            is_incoming_session ? SaturatingAdd(sess_bytes, incoming_bytes) : sess_bytes;

        if (lease_active) {
            // If this session is pushing a single oversized entry past its fair
            // share cap, do not let its existing soft leases shield it ahead of peers.
            const bool oversized_same_session =
                is_incoming_session && session_lease_cap > 0 && incoming_bytes > session_lease_cap &&
                projected_sess_bytes > session_lease_cap;
            if (!oversized_same_session && st.policy.soft_lease_boost > 1.0) {
                score *= st.policy.soft_lease_boost;
            }
        }

        // 5. Over-cap fairness pressure: when a session exceeds its fair-share byte
        //    threshold, scale down its excess/unleased entries so it yields to peers.
        if (!rec.session_id.empty() && session_lease_cap > 0 && projected_sess_bytes > session_lease_cap) {
            if (!lease_active || (is_incoming_session && incoming_bytes > session_lease_cap)) {
                score *= static_cast<double>(session_lease_cap) / static_cast<double>(projected_sess_bytes);
            }
        }

        return score;
    }

    // Sheds lowest-value / oldest soft leases belonging to `session_id` until
    // `session_soft_leased_bytes[session_id] + needed_bytes <= cap`.
    static bool MakeRoomForSessionSoftLeaseFromState(CacheState& st, std::string_view session_id,
                                                     std::size_t needed_bytes, std::uint64_t now_ms,
                                                     const Node* exclude_node) noexcept {
        if (session_id.empty()) return false;
        const std::size_t cap = SessionSoftLeaseCapBytesFromState(st);
        if (cap == 0 || needed_bytes > cap) return false;
        const std::size_t limit = cap - needed_bytes;
        std::size_t current = GetSessionBytes(st.session_soft_leased_bytes, session_id);
        if (current <= limit) return true;

        const RestoreCostRates rates = st.restore_gate.SnapshotRates();
        while (current > limit) {
            Node* victim = nullptr;
            double victim_score = 0.0;
            std::uint64_t victim_lru = 0;
            for (Node* n : st.lru_list) {
                if (n == nullptr || n == exclude_node || !n->has_entry()) continue;
                const EntryRecord& rec = *n->record;
                if (!rec.has_soft_lease || rec.session_id != session_id) continue;
                const double cand_score =
                    ComputeEntryValueScoreFromState(st, n, now_ms, rates, cap, {}, 0);
                if (victim == nullptr || cand_score < victim_score ||
                    (cand_score == victim_score && rec.lru_seq < victim_lru)) {
                    victim = n;
                    victim_score = cand_score;
                    victim_lru = rec.lru_seq;
                }
            }
            if (victim == nullptr) {
                return false;
            }
            RevokeSoftLeaseFromState(st, victim->record.get());
            ++st.soft_leases_capped;
            current = GetSessionBytes(st.session_soft_leased_bytes, session_id);
        }
        return true;
    }

    static bool GrantSoftLeaseFromState(CacheState& st, Node* node, std::uint64_t now_ms,
                                        std::uint64_t requested_ttl_ms) {
        if (node == nullptr || !node->has_entry()) return false;
        EntryRecord& rec = *node->record;
        if (rec.session_id.empty()) return false;
        const std::uint64_t ttl_ms = EffectiveSoftLeaseTtlMs(st.policy, requested_ttl_ms);
        if (ttl_ms == 0) {
            RevokeSoftLeaseFromState(st, &rec);
            return false;
        }
        const std::size_t cap = SessionSoftLeaseCapBytesFromState(st);
        if (cap == 0 || rec.bytes > cap) {
            RevokeSoftLeaseFromState(st, &rec);
            ++st.soft_leases_capped;
            return false;
        }
        if (rec.has_soft_lease) {
            // Already leased and accounted; ensure cap holds (e.g. if policy changed) and extend TTL.
            if (GetSessionBytes(st.session_soft_leased_bytes, rec.session_id) > cap) {
                if (!MakeRoomForSessionSoftLeaseFromState(st, rec.session_id, 0, now_ms, node)) {
                    RevokeSoftLeaseFromState(st, &rec);
                    ++st.soft_leases_capped;
                    return false;
                }
            }
            rec.soft_lease_expires_ms = SaturatingAddU64(now_ms, ttl_ms);
            return true;
        }
        if (!MakeRoomForSessionSoftLeaseFromState(st, rec.session_id, rec.bytes, now_ms, node)) {
            ++st.soft_leases_capped;
            return false;
        }
        rec.has_soft_lease = true;
        rec.soft_lease_expires_ms = SaturatingAddU64(now_ms, ttl_ms);
        AddSessionBytes(st.session_soft_leased_bytes, rec.session_id, rec.bytes);
        ++st.active_soft_lease_count;
        return true;
    }

    static void EnforceAllSessionSoftLeaseCapsFromState(CacheState& st, std::uint64_t now_ms) noexcept {
        if (st.active_soft_lease_count == 0) return;
        const std::size_t cap = SessionSoftLeaseCapBytesFromState(st);
        if (cap == 0) {
            for (Node* n : st.lru_list) {
                if (n != nullptr && n->has_entry() && n->record->has_soft_lease) {
                    RevokeSoftLeaseFromState(st, n->record.get());
                    ++st.soft_leases_capped;
                }
            }
            return;
        }
        // Snapshot over-cap session keys first so map erasure during revocation never invalidates iterators.
        std::vector<std::string> over_cap;
        for (const auto& [sid, leased] : st.session_soft_leased_bytes) {
            if (leased > cap) {
                over_cap.push_back(sid);
            }
        }
        for (const std::string& sid : over_cap) {
            (void)MakeRoomForSessionSoftLeaseFromState(st, sid, 0, now_ms, nullptr);
        }
    }

public:
    // Move-only RAII hard pin handle returned by `LookupAndPin`. Keeps the matched
    // entry's payload alive and protected from eviction (and accounted against
    // `used_bytes()`) until destroyed or released.
    class PinnedLookup {
    public:
        PinnedLookup() = default;
        ~PinnedLookup() { Release(); }

        PinnedLookup(const PinnedLookup&) = delete;
        PinnedLookup& operator=(const PinnedLookup&) = delete;

        PinnedLookup(PinnedLookup&& other) noexcept
            : matched_tokens_(std::exchange(other.matched_tokens_, 0)),
              state_tokens_(std::exchange(other.state_tokens_, 0)),
              state_(std::move(other.state_)),
              record_(std::move(other.record_)) {}

        PinnedLookup& operator=(PinnedLookup&& other) noexcept {
            if (this != &other) {
                Release();
                matched_tokens_ = std::exchange(other.matched_tokens_, 0);
                state_tokens_ = std::exchange(other.state_tokens_, 0);
                state_ = std::move(other.state_);
                record_ = std::move(other.record_);
            }
            return *this;
        }

        [[nodiscard]] bool found() const noexcept { return record_ != nullptr && matched_tokens_ > 0; }
        [[nodiscard]] std::size_t matched_tokens() const noexcept { return matched_tokens_; }
        [[nodiscard]] std::size_t state_tokens() const noexcept { return state_tokens_; }
        [[nodiscard]] const RadixCacheEntry* entry() const noexcept {
            return record_ != nullptr ? &record_->payload : nullptr;
        }

        void Release() noexcept {
            if (!record_) {
                matched_tokens_ = 0;
                state_tokens_ = 0;
                state_.reset();
                return;
            }
            std::shared_ptr<EntryRecord> rec = std::move(record_);
            std::shared_ptr<CacheState> st = state_.lock();
            state_.reset();
            matched_tokens_ = 0;
            state_tokens_ = 0;
            std::vector<std::shared_ptr<EntryRecord>> to_free;
            if (st) {
                std::lock_guard<std::mutex> lock(st->mu);
                if (rec->pin_count > 0) {
                    --rec->pin_count;
                    if (rec->pin_count == 0) {
                        if (rec->retired) {
                            st->used_bytes = st->used_bytes >= rec->bytes ? (st->used_bytes - rec->bytes) : 0;
                            st->retired_pinned_bytes = st->retired_pinned_bytes >= rec->bytes
                                                           ? (st->retired_pinned_bytes - rec->bytes)
                                                           : 0;
                            st->retired_metadata_bytes = st->retired_metadata_bytes >= rec->metadata_overhead_bytes
                                                             ? (st->retired_metadata_bytes - rec->metadata_overhead_bytes)
                                                             : 0;
                            if (st->retired_pinned_count > 0) --st->retired_pinned_count;
                        } else {
                            st->active_pinned_bytes = st->active_pinned_bytes >= rec->bytes
                                                          ? (st->active_pinned_bytes - rec->bytes)
                                                          : 0;
                            if (st->active_pinned_count > 0) --st->active_pinned_count;
                            // If SetMaxBytes() shrank the budget while this entry was pinned,
                            // enforce the cap as soon as the last pin releases.
                            try {
                                while (st->used_bytes > st->max_bytes) {
                                    if (!EvictLowestValueFromState(*st, to_free)) break;
                                }
                            } catch (...) {
                                // Destructor / Release() must remain noexcept even under OOM.
                            }
                        }
                    }
                }
            }
            // Destroy `rec` and `to_free` outside `st->mu` so freeing large state blobs never stalls waiters.
        }

    private:
        friend class RadixPrefixCache;

        PinnedLookup(std::size_t matched_tokens, std::size_t state_tokens, std::weak_ptr<CacheState> state,
                     std::shared_ptr<EntryRecord> record) noexcept
            : matched_tokens_(matched_tokens),
              state_tokens_(state_tokens),
              state_(std::move(state)),
              record_(std::move(record)) {}

        std::size_t matched_tokens_ = 0;
        std::size_t state_tokens_ = 0;
        std::weak_ptr<CacheState> state_;
        std::shared_ptr<EntryRecord> record_;
    };

    explicit RadixPrefixCache(std::size_t max_bytes = 0, RadixCachePolicy policy = {},
                              RadixClockMsFn clock_ms = {})
        : state_(std::make_shared<CacheState>()) {
        state_->max_bytes = max_bytes;
        state_->policy = SanitizePolicy(policy);
        state_->clock_ms = std::move(clock_ms);
        state_->restore_gate.SetConfig(state_->policy.restore_cost_gate);
    }

    RadixPrefixCache(const RadixPrefixCache&) = delete;
    RadixPrefixCache& operator=(const RadixPrefixCache&) = delete;
    RadixPrefixCache(RadixPrefixCache&&) noexcept = default;
    RadixPrefixCache& operator=(RadixPrefixCache&&) noexcept = default;

    void SetPolicy(RadixCachePolicy policy) {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->policy = SanitizePolicy(policy);
        state_->restore_gate.SetConfig(state_->policy.restore_cost_gate);
        const std::uint64_t now_ms = NowMsFromState(*state_);
        PurgeExpiredSoftLeasesFromState(*state_, now_ms);
        EnforceAllSessionSoftLeaseCapsFromState(*state_, now_ms);
    }

    [[nodiscard]] RadixCachePolicy policy() const {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->policy;
    }

    void SetClock(RadixClockMsFn clock_ms) {
        std::lock_guard<std::mutex> lock(state_->mu);
        state_->clock_ms = std::move(clock_ms);
    }

    [[nodiscard]] std::uint64_t NowMs() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return NowMsFromState(*state_);
    }

    [[nodiscard]] RestoreCostGate& restore_cost_gate() noexcept { return state_->restore_gate; }
    [[nodiscard]] const RestoreCostGate& restore_cost_gate() const noexcept { return state_->restore_gate; }

    void ObservePrefill(std::size_t prefilled_tokens, double prefill_ms) noexcept {
        state_->restore_gate.ObservePrefill(prefilled_tokens, prefill_ms);
    }

    void ObserveRestore(const PinnedLookup& pin, std::size_t payload_bytes, double restore_ms) noexcept {
        state_->restore_gate.ObserveRestore(payload_bytes, restore_ms);
        if (pin.record_ != nullptr && std::isfinite(restore_ms) && restore_ms > 0.0) {
            std::lock_guard<std::mutex> lock(state_->mu);
            pin.record_->last_restore_ms = restore_ms;
        }
    }

    void ObserveRestore(std::span<const Token> prefix, std::size_t payload_bytes, double restore_ms) noexcept {
        state_->restore_gate.ObserveRestore(payload_bytes, restore_ms);
        if (prefix.empty() || !std::isfinite(restore_ms) || restore_ms <= 0.0) return;
        std::lock_guard<std::mutex> lock(state_->mu);
        if (Node* node = FindExactNodeLocked(prefix); node != nullptr && node->has_entry()) {
            node->record->last_restore_ms = restore_ms;
        }
    }

    std::size_t PurgeExpiredSoftLeases() noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return PurgeExpiredSoftLeasesFromState(*state_, NowMsFromState(*state_));
    }

    void SetMaxBytes(std::size_t max_bytes) {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            state_->max_bytes = max_bytes;
            if (max_bytes == 0) {
                ClearLocked(to_free);
                return;
            }
            const std::uint64_t now_ms = NowMsFromState(*state_);
            PurgeExpiredSoftLeasesFromState(*state_, now_ms);
            EnforceAllSessionSoftLeaseCapsFromState(*state_, now_ms);
            while (state_->used_bytes > state_->max_bytes) {
                if (!EvictLowestValueLocked(to_free)) break;
            }
        }
    }

    void Clear() {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            ClearLocked(to_free);
        }
    }

    [[nodiscard]] bool enabled() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->max_bytes > 0;
    }
    [[nodiscard]] std::size_t max_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->max_bytes;
    }
    [[nodiscard]] std::size_t used_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->used_bytes;
    }
    [[nodiscard]] std::size_t metadata_overhead_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return SaturatingAdd(state_->active_metadata_bytes, state_->retired_metadata_bytes);
    }
    [[nodiscard]] std::size_t retained_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return SaturatingAdd(state_->used_bytes,
                             SaturatingAdd(state_->active_metadata_bytes, state_->retired_metadata_bytes));
    }
    [[nodiscard]] std::size_t entry_count() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->entry_count;
    }
    [[nodiscard]] std::size_t pinned_entries() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->active_pinned_count + state_->retired_pinned_count;
    }
    [[nodiscard]] std::size_t pinned_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->active_pinned_bytes + state_->retired_pinned_bytes;
    }
    [[nodiscard]] std::size_t retired_pinned_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->retired_pinned_bytes;
    }
    [[nodiscard]] std::uint64_t evictions() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->evictions;
    }
    [[nodiscard]] std::uint64_t hits() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->hits;
    }
    [[nodiscard]] std::uint64_t misses() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->misses;
    }
    [[nodiscard]] std::size_t active_soft_leases() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->active_soft_lease_count;
    }
    [[nodiscard]] std::uint64_t soft_leases_expired() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->soft_leases_expired;
    }
    [[nodiscard]] std::uint64_t soft_leases_capped() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return state_->soft_leases_capped;
    }
    [[nodiscard]] std::size_t session_bytes(std::string_view session_id) const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return GetSessionBytes(state_->session_total_bytes, session_id);
    }
    [[nodiscard]] std::size_t session_soft_leased_bytes(std::string_view session_id) const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return GetSessionBytes(state_->session_soft_leased_bytes, session_id);
    }
    [[nodiscard]] std::size_t session_soft_lease_cap_bytes() const noexcept {
        std::lock_guard<std::mutex> lock(state_->mu);
        return SessionSoftLeaseCapBytesFromState(*state_);
    }

    // Strict value-aware global eviction BEFORE caller allocates a state buffer.
    // Returns false with ZERO evictions if `incoming_bytes` cannot fit within
    // `max_bytes_` (including when pinned in-flight entries occupy too much budget).
    bool ReserveBeforeAlloc(std::size_t incoming_bytes, std::span<const Token> overwrite_prefix = {},
                            std::string_view session_id = {}, bool request_soft_lease = true) {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        bool ok = false;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            const std::uint64_t now_ms = NowMsFromState(*state_);
            ok = ReserveBeforeAllocLocked(incoming_bytes, overwrite_prefix, session_id, request_soft_lease, now_ms,
                                          to_free);
        }
        return ok;
    }

    // Inserts or overwrites the state blob for `prefix`. Enforces strict
    // value-aware global eviction before mutating the tree, enforces the
    // per-session soft-lease fairness cap, and never evicts or invalidates
    // hard-pinned in-flight entries (retiring an overwritten pinned entry in
    // RAM until its last pin is released, strictly within `max_bytes_`).
    bool Insert(std::span<const Token> prefix, RadixCacheEntry entry) {
        const std::size_t new_bytes = entry.ByteSize();
        if (prefix.empty() || entry.state_blob.empty() || new_bytes == 0) {
            return false;
        }
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            const std::uint64_t now_ms = NowMsFromState(*state_);
            PurgeExpiredSoftLeasesFromState(*state_, now_ms);
            if (state_->max_bytes == 0 || new_bytes > state_->max_bytes) {
                return false;
            }
            const bool wants_soft_lease = !entry.session_key.empty() && entry.request_soft_lease &&
                                          EffectiveSoftLeaseTtlMs(state_->policy, entry.soft_lease_ttl_ms) > 0;

            // Evict unpinned entries BEFORE EnsureNode so EnsureNode can never be
            // invalidated by PruneIfEmpty during eviction.
            if (!ReserveBeforeAllocLocked(new_bytes, prefix, entry.session_key, wants_soft_lease, now_ms, to_free)) {
                return false;
            }

            Node* node = EnsureNodeLocked(prefix);
            if (node == nullptr) {
                return false;
            }

            if (node->has_entry()) {
                if (node->record->pin_count == 0) {
                    DetachEntryLocked(node, to_free);
                } else {
                    RetireEntryLocked(node);
                }
            }

            auto rec = std::make_shared<EntryRecord>();
            rec->session_id = entry.session_key;
            rec->token_count = prefix.size();
            rec->bytes = new_bytes;
            rec->metadata_overhead_bytes =
                EstimateMetadataOverheadBytes(rec->token_count, rec->session_id.size());
            rec->retained_bytes = SaturatingAdd(rec->bytes, rec->metadata_overhead_bytes);
            rec->last_access_ms = now_ms;
            rec->recompute_ms = (std::isfinite(entry.estimated_recompute_ms) && entry.estimated_recompute_ms > 0.0)
                                    ? entry.estimated_recompute_ms
                                    : 0.0;
            rec->last_restore_ms = (std::isfinite(entry.last_restore_ms) && entry.last_restore_ms > 0.0)
                                       ? entry.last_restore_ms
                                       : 0.0;
            rec->lru_seq = ++state_->lru_clock;
            const std::uint64_t req_ttl = entry.soft_lease_ttl_ms;
            rec->payload = std::move(entry);

            node->record = std::move(rec);
            state_->lru_list.push_back(node);
            node->lru_it = std::prev(state_->lru_list.end());
            state_->used_bytes += new_bytes;
            state_->active_metadata_bytes += node->record->metadata_overhead_bytes;
            ++state_->entry_count;
            for (Node* p = node->parent; p != nullptr; p = p->parent) {
                ++p->cached_descendants;
            }
            if (!node->record->session_id.empty()) {
                AddSessionBytes(state_->session_total_bytes, node->record->session_id, new_bytes);
            }
            if (wants_soft_lease) {
                (void)GrantSoftLeaseFromState(*state_, node, now_ms, req_ttl);
            }
            return true;
        }
    }

    // Finds the longest cached prefix matching `prompt`, capped at
    // `max_match_tokens` (typically `prompt.size() - 1`).
    // When `allow_prefix_trim` is true and a deeper descendant shares a longer
    // prefix than any exact ancestor node, returns that descendant so callers on
    // attention-only models can restore and trim the trailing KV positions.
    // Note: Prefer `LookupAndPin` whenever the entry will be read across concurrent
    // cache mutations or subsequent `ReserveBeforeAlloc`/`Insert` calls.
    [[nodiscard]] RadixLookupResult LookupLongestPrefix(
        std::span<const Token> prompt, std::size_t max_match_tokens = std::numeric_limits<std::size_t>::max(),
        bool allow_prefix_trim = true, std::string_view session_id = {}) {
        RadixLookupResult result;
        std::lock_guard<std::mutex> lock(state_->mu);
        const std::uint64_t now_ms = NowMsFromState(*state_);
        PurgeExpiredSoftLeasesFromState(*state_, now_ms);
        std::size_t matched_len = 0;
        Node* chosen = LookupNodeLocked(prompt, max_match_tokens, allow_prefix_trim, matched_len);
        if (chosen == nullptr || !chosen->has_entry() || matched_len == 0) {
            ++state_->misses;
            return result;
        }
        TouchLocked(chosen, now_ms, session_id);
        ++state_->hits;
        result.matched_tokens = matched_len;
        result.state_tokens = chosen->depth;
        result.entry = &chosen->record->payload;
        return result;
    }

    // Thread-safe & eviction-safe lookup returning an RAII hard pin handle that
    // prevents the matched entry from being evicted or freed while in flight.
    [[nodiscard]] PinnedLookup LookupAndPin(
        std::span<const Token> prompt, std::size_t max_match_tokens = std::numeric_limits<std::size_t>::max(),
        bool allow_prefix_trim = true, std::string_view session_id = {}) {
        std::lock_guard<std::mutex> lock(state_->mu);
        const std::uint64_t now_ms = NowMsFromState(*state_);
        PurgeExpiredSoftLeasesFromState(*state_, now_ms);
        std::size_t matched_len = 0;
        Node* chosen = LookupNodeLocked(prompt, max_match_tokens, allow_prefix_trim, matched_len);
        if (chosen == nullptr || !chosen->has_entry() || matched_len == 0) {
            ++state_->misses;
            return {};
        }
        TouchLocked(chosen, now_ms, session_id);
        ++state_->hits;
        std::shared_ptr<EntryRecord> rec = chosen->record;
        if (rec->pin_count == 0) {
            ++state_->active_pinned_count;
            state_->active_pinned_bytes += rec->bytes;
        }
        ++rec->pin_count;
        return PinnedLookup(matched_len, chosen->depth, state_, std::move(rec));
    }

    // Explicitly refreshes an exact entry's soft lease for its owning session.
    bool RefreshSoftLease(std::span<const Token> prefix, std::string_view session_id,
                          std::uint64_t ttl_ms = 0) {
        if (prefix.empty() || session_id.empty()) return false;
        std::lock_guard<std::mutex> lock(state_->mu);
        const std::uint64_t now_ms = NowMsFromState(*state_);
        PurgeExpiredSoftLeasesFromState(*state_, now_ms);
        Node* node = FindExactNodeLocked(prefix);
        if (node == nullptr || !node->has_entry() || node->record->session_id != session_id) {
            return false;
        }
        return GrantSoftLeaseFromState(*state_, node, now_ms, ttl_ms);
    }

    // Inspects retention metadata and live value score for an exact cached prefix.
    [[nodiscard]] std::optional<RadixEntryMetrics> InspectEntry(std::span<const Token> prefix) const {
        if (prefix.empty()) return std::nullopt;
        std::lock_guard<std::mutex> lock(state_->mu);
        const Node* node = FindExactNodeLocked(prefix);
        if (node == nullptr || !node->has_entry()) return std::nullopt;
        const EntryRecord& rec = *node->record;
        const std::uint64_t now_ms = NowMsFromState(*state_);
        const RestoreCostRates rates = state_->restore_gate.SnapshotRates();
        const std::size_t lease_cap = SessionSoftLeaseCapBytesFromState(*state_);
        RadixEntryMetrics m;
        m.payload_bytes = rec.bytes;
        m.metadata_overhead_bytes = rec.metadata_overhead_bytes;
        m.retained_bytes = rec.retained_bytes;
        m.token_count = rec.token_count;
        m.reuse_count = rec.reuse_count;
        m.shared_prefix_children = node->cached_descendants;
        m.last_access_ms = rec.last_access_ms;
        m.session_id = rec.session_id;
        m.soft_lease_active = rec.has_soft_lease && now_ms < rec.soft_lease_expires_ms;
        m.soft_lease_expires_ms = m.soft_lease_active ? rec.soft_lease_expires_ms : 0;
        m.last_restore_ms = rec.last_restore_ms;
        m.recompute_ms = rec.recompute_ms;
        m.value_score =
            ComputeEntryValueScoreFromState(*state_, node, now_ms, rates, lease_cap, {}, 0);
        m.pin_count = rec.pin_count;
        return m;
    }

    // Removes an exact cached prefix if present. If currently pinned in flight,
    // retires it immediately from lookup visibility while keeping its payload
    // alive and counted in `used_bytes()` until the last pin releases.
    bool Erase(std::span<const Token> prefix) {
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        bool erased = false;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            Node* node = FindExactNodeLocked(prefix);
            if (node == nullptr || !node->has_entry()) {
                return false;
            }
            if (node->record->pin_count == 0) {
                DetachEntryLocked(node, to_free);
            } else {
                RetireEntryLocked(node);
            }
            PruneIfEmptyLocked(node);
            erased = true;
        }
        return erased;
    }

    // Multi-branch shared prefix invalidation: removes every cached entry whose
    // token sequence begins with `prefix` (including when `prefix` falls inside
    // a compressed radix edge). Returns the number of cached entries invalidated.
    // Empty `prefix` is a safe no-op returning 0 (use `Clear()` to wipe all).
    std::size_t ErasePrefix(std::span<const Token> prefix) {
        if (prefix.empty()) return 0;
        std::vector<std::shared_ptr<EntryRecord>> to_free;
        std::size_t removed = 0;
        {
            std::lock_guard<std::mutex> lock(state_->mu);
            Node* cur = state_->root.get();
            std::size_t pos = 0;
            Node* target = nullptr;
            while (pos < prefix.size()) {
                const auto it = cur->children.find(prefix[pos]);
                if (it == cur->children.end()) {
                    return 0;
                }
                Node* child = it->second.get();
                const std::size_t edge_len = child->edge.size();
                const std::size_t rem = prefix.size() - pos;
                const std::size_t cmp_len = std::min(edge_len, rem);
                for (std::size_t k = 0; k < cmp_len; ++k) {
                    if (child->edge[k] != prefix[pos + k]) {
                        return 0;
                    }
                }
                if (rem <= edge_len) {
                    target = child;
                    break;
                }
                pos += edge_len;
                cur = child;
            }
            if (target == nullptr) {
                return 0;
            }

            // Iteratively detach/retire every cached entry in `target`'s subtree.
            auto& stack = state_->search_scratch;
            stack.clear();
            stack.push_back(target);
            while (!stack.empty()) {
                Node* n = stack.back();
                stack.pop_back();
                if (n->has_entry()) {
                    if (n->record->pin_count == 0) {
                        DetachEntryLocked(n, to_free);
                    } else {
                        RetireEntryLocked(n);
                    }
                    ++removed;
                }
                for (const auto& [tok, ch] : n->children) {
                    (void)tok;
                    stack.push_back(ch.get());
                }
            }
            stack.clear();

            Node* parent = target->parent;
            if (parent != nullptr && !target->edge.empty()) {
                const Token key = target->edge.front();
                parent->children.erase(key);
                PruneIfEmptyLocked(parent);
            }
        }
        return removed;
    }

    // Verifies all structural, compression, depth, descendant-branching,
    // metadata/byte-budget, hard-pin, and per-session soft-lease fairness invariants.
    [[nodiscard]] bool CheckInvariants() const {
        std::lock_guard<std::mutex> lock(state_->mu);
        if (!state_->root || state_->root->parent != nullptr || !state_->root->edge.empty() ||
            state_->root->depth != 0 || state_->root->has_entry()) {
            return false;
        }
        std::size_t counted_entries = 0;
        std::size_t counted_bytes = 0;
        std::size_t counted_metadata_bytes = 0;
        std::size_t counted_active_pinned = 0;
        std::size_t counted_active_pinned_bytes = 0;
        std::size_t counted_soft_leases = 0;
        SessionByteMap counted_session_total;
        SessionByteMap counted_session_leased;

        std::vector<const Node*> order;
        order.push_back(state_->root.get());
        for (std::size_t idx = 0; idx < order.size(); ++idx) {
            const Node* n = order[idx];
            if (n != state_->root.get()) {
                if (n->parent == nullptr || n->edge.empty()) return false;
                if (n->depth != n->parent->depth + n->edge.size()) return false;
                if (!n->has_entry() && n->children.size() <= 1) return false;
                if (n->has_entry()) {
                    const EntryRecord& rec = *n->record;
                    const std::size_t expected_meta =
                        EstimateMetadataOverheadBytes(rec.token_count, rec.session_id.size());
                    if (rec.retired || rec.bytes == 0 || rec.bytes != rec.payload.ByteSize()) {
                        return false;
                    }
                    if (rec.token_count != n->depth) return false;
                    if (rec.metadata_overhead_bytes != expected_meta) return false;
                    if (rec.retained_bytes != SaturatingAdd(rec.bytes, expected_meta)) return false;
                    if (rec.has_soft_lease && rec.session_id.empty()) return false;
                    if (*n->lru_it != n) return false;
                    ++counted_entries;
                    counted_bytes += rec.bytes;
                    counted_metadata_bytes += rec.metadata_overhead_bytes;
                    if (rec.pin_count > 0) {
                        ++counted_active_pinned;
                        counted_active_pinned_bytes += rec.bytes;
                    }
                    if (!rec.session_id.empty()) {
                        counted_session_total[rec.session_id] += rec.bytes;
                    }
                    if (rec.has_soft_lease) {
                        ++counted_soft_leases;
                        counted_session_leased[rec.session_id] += rec.bytes;
                    }
                }
            }
            for (const auto& [tok, ch] : n->children) {
                if (!ch || ch->parent != n || ch->edge.empty() || ch->edge.front() != tok) {
                    return false;
                }
                order.push_back(ch.get());
            }
        }

        // Bottom-up verification of `cached_descendants` across every node.
        std::unordered_map<const Node*, std::size_t> subtree_entries;
        subtree_entries.reserve(order.size());
        for (std::size_t idx = order.size(); idx > 0; --idx) {
            const Node* n = order[idx - 1];
            std::size_t desc = 0;
            for (const auto& [tok, ch] : n->children) {
                (void)tok;
                desc += subtree_entries[ch.get()];
            }
            if (n->cached_descendants != desc) return false;
            subtree_entries[n] = desc + (n->has_entry() ? 1u : 0u);
        }

        if (counted_entries != state_->entry_count) return false;
        if (counted_entries != state_->lru_list.size()) return false;
        if (counted_active_pinned != state_->active_pinned_count) return false;
        if (counted_active_pinned_bytes != state_->active_pinned_bytes) return false;
        if (counted_metadata_bytes != state_->active_metadata_bytes) return false;
        if (counted_bytes + state_->retired_pinned_bytes != state_->used_bytes) return false;
        if (counted_soft_leases != state_->active_soft_lease_count) return false;
        if (counted_session_total != state_->session_total_bytes) return false;
        if (counted_session_leased != state_->session_soft_leased_bytes) return false;

        const std::size_t lease_cap = SessionSoftLeaseCapBytesFromState(*state_);
        for (const auto& [sid, leased] : state_->session_soft_leased_bytes) {
            if (sid.empty() || leased == 0 || leased > lease_cap) return false;
        }
        for (const auto& [sid, tot] : state_->session_total_bytes) {
            if (sid.empty() || tot == 0) return false;
        }

        const std::size_t pinned_total = state_->active_pinned_bytes + state_->retired_pinned_bytes;
        if (state_->used_bytes > state_->max_bytes && state_->used_bytes > pinned_total) {
            return false;
        }
        for (const Node* n : state_->lru_list) {
            if (n == nullptr || !n->has_entry()) return false;
        }
        return true;
    }

private:
    void ClearLocked(std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        for (Node* n : state_->lru_list) {
            if (n == nullptr || !n->has_entry()) continue;
            if (n->record->pin_count == 0) {
                to_free.push_back(std::move(n->record));
            } else {
                n->record->retired = true;
                n->record->has_soft_lease = false;
                n->record->soft_lease_expires_ms = 0;
                state_->retired_pinned_bytes += n->record->bytes;
                state_->retired_metadata_bytes += n->record->metadata_overhead_bytes;
                ++state_->retired_pinned_count;
                n->record.reset();
            }
        }
        state_->lru_list.clear();
        state_->root = std::make_unique<Node>();
        state_->entry_count = 0;
        state_->active_pinned_count = 0;
        state_->active_pinned_bytes = 0;
        state_->active_metadata_bytes = 0;
        state_->active_soft_lease_count = 0;
        state_->session_total_bytes.clear();
        state_->session_soft_leased_bytes.clear();
        state_->used_bytes = state_->retired_pinned_bytes;
    }

    void TouchLocked(Node* node, std::uint64_t now_ms, std::string_view session_id) {
        if (node == nullptr || !node->has_entry()) return;
        EntryRecord& rec = *node->record;
        ++rec.reuse_count;
        rec.last_access_ms = now_ms;
        rec.lru_seq = ++state_->lru_clock;
        state_->lru_list.splice(state_->lru_list.end(), state_->lru_list, node->lru_it);
        if (!session_id.empty() && rec.session_id == session_id && rec.payload.request_soft_lease) {
            (void)GrantSoftLeaseFromState(*state_, node, now_ms, rec.payload.soft_lease_ttl_ms);
        }
    }

    static void DetachEntryFromState(CacheState& st, Node* node,
                                     std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        if (node == nullptr || !node->has_entry()) return;
        EntryRecord& rec = *node->record;
        const std::size_t bytes = rec.bytes;
        const std::size_t meta_bytes = rec.metadata_overhead_bytes;
        RevokeSoftLeaseFromState(st, &rec);
        if (!rec.session_id.empty()) {
            SubSessionBytes(st.session_total_bytes, rec.session_id, bytes);
        }
        for (Node* p = node->parent; p != nullptr; p = p->parent) {
            if (p->cached_descendants > 0) --p->cached_descendants;
        }
        st.used_bytes = st.used_bytes >= bytes ? (st.used_bytes - bytes) : 0;
        st.active_metadata_bytes =
            st.active_metadata_bytes >= meta_bytes ? (st.active_metadata_bytes - meta_bytes) : 0;
        st.lru_list.erase(node->lru_it);
        to_free.push_back(std::move(node->record));
        if (st.entry_count > 0) --st.entry_count;
    }

    static void PruneIfEmptyFromState(CacheState& st, Node* node) {
        while (node != nullptr && node != st.root.get()) {
            if (node->has_entry()) {
                return;
            }
            if (node->children.empty()) {
                Node* parent = node->parent;
                if (parent == nullptr || node->edge.empty()) return;
                const Token first_tok = node->edge.front();
                parent->children.erase(first_tok);
                node = parent;
                continue;
            }
            // Compress single-child non-cached internal node back into its child.
            if (node->children.size() == 1) {
                Node* parent = node->parent;
                if (parent == nullptr) return;
                auto child_it = node->children.begin();
                std::unique_ptr<Node> child = std::move(child_it->second);
                std::vector<Token> merged;
                merged.reserve(node->edge.size() + child->edge.size());
                merged.insert(merged.end(), node->edge.begin(), node->edge.end());
                merged.insert(merged.end(), child->edge.begin(), child->edge.end());
                child->edge = std::move(merged);
                child->parent = parent;
                const Token parent_key = child->edge.front();
                parent->children[parent_key] = std::move(child);
                return;
            }
            return;
        }
    }

    // Value-aware global eviction: selects the unpinned entry with lowest
    // V_i = P(reuse_i) * max(0, T_recompute_i - T_restore_i) / max(1, retained_bytes_i)
    // (tie-broken by unleased before leased, then oldest `lru_seq`).
    // When `restrict_to_session` is non-empty, only considers entries owned by that session.
    static bool EvictLowestValueFromState(CacheState& st, std::vector<std::shared_ptr<EntryRecord>>& to_free,
                                          std::string_view incoming_session_id = {},
                                          std::size_t incoming_bytes = 0,
                                          std::string_view restrict_to_session = {}) {
        const std::uint64_t now_ms = NowMsFromState(st);
        PurgeExpiredSoftLeasesFromState(st, now_ms);
        const RestoreCostRates rates = st.restore_gate.SnapshotRates();
        const std::size_t lease_cap = SessionSoftLeaseCapBytesFromState(st);

        Node* victim = nullptr;
        double victim_score = 0.0;
        bool victim_leased = false;
        std::uint64_t victim_lru = 0;

        for (Node* cand : st.lru_list) {
            if (cand == nullptr || !cand->has_entry() || cand->record->pin_count > 0) {
                continue;
            }
            const EntryRecord& rec = *cand->record;
            if (!restrict_to_session.empty() && rec.session_id != restrict_to_session) {
                continue;
            }
            const double cand_score = ComputeEntryValueScoreFromState(
                st, cand, now_ms, rates, lease_cap, incoming_session_id, incoming_bytes);
            const bool cand_leased = rec.has_soft_lease && now_ms < rec.soft_lease_expires_ms;
            const bool better = (victim == nullptr) || (cand_score < victim_score) ||
                                (cand_score == victim_score && cand_leased != victim_leased && !cand_leased) ||
                                (cand_score == victim_score && cand_leased == victim_leased &&
                                 rec.lru_seq < victim_lru);
            if (better) {
                victim = cand;
                victim_score = cand_score;
                victim_leased = cand_leased;
                victim_lru = rec.lru_seq;
            }
        }
        if (victim == nullptr) {
            return false;
        }
        DetachEntryFromState(st, victim, to_free);
        ++st.evictions;
        PruneIfEmptyFromState(st, victim);
        return true;
    }

    void DetachEntryLocked(Node* node, std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        DetachEntryFromState(*state_, node, to_free);
    }

    void RetireEntryLocked(Node* node) noexcept {
        if (node == nullptr || !node->has_entry()) return;
        EntryRecord& rec = *node->record;
        const std::size_t bytes = rec.bytes;
        const std::size_t meta_bytes = rec.metadata_overhead_bytes;
        RevokeSoftLeaseFromState(*state_, &rec);
        if (!rec.session_id.empty()) {
            SubSessionBytes(state_->session_total_bytes, rec.session_id, bytes);
        }
        for (Node* p = node->parent; p != nullptr; p = p->parent) {
            if (p->cached_descendants > 0) --p->cached_descendants;
        }
        rec.retired = true;
        state_->lru_list.erase(node->lru_it);
        if (state_->entry_count > 0) --state_->entry_count;
        state_->active_metadata_bytes = state_->active_metadata_bytes >= meta_bytes
                                            ? (state_->active_metadata_bytes - meta_bytes)
                                            : 0;
        state_->retired_metadata_bytes += meta_bytes;
        state_->active_pinned_bytes =
            state_->active_pinned_bytes >= bytes ? (state_->active_pinned_bytes - bytes) : 0;
        if (state_->active_pinned_count > 0) --state_->active_pinned_count;
        state_->retired_pinned_bytes += bytes;
        ++state_->retired_pinned_count;
        node->record.reset();
    }

    bool ReserveBeforeAllocLocked(std::size_t incoming_bytes, std::span<const Token> overwrite_prefix,
                                  std::string_view incoming_session_id, bool incoming_wants_soft_lease,
                                  std::uint64_t now_ms,
                                  std::vector<std::shared_ptr<EntryRecord>>& to_free) {
        PurgeExpiredSoftLeasesFromState(*state_, now_ms);
        if (state_->max_bytes == 0 || incoming_bytes == 0 || incoming_bytes > state_->max_bytes) {
            return false;
        }
        const std::size_t budget_limit = state_->max_bytes - incoming_bytes;
        // Fail fast BEFORE mutating soft leases or evicting anything if hard-pinned
        // entries alone exceed the remaining global budget.
        const std::size_t pinned_total = state_->active_pinned_bytes + state_->retired_pinned_bytes;
        if (pinned_total > budget_limit) {
            return false;
        }

        // Check optional strict per-session total-byte cap before mutating anything.
        const bool enforce_session_total_cap =
            !incoming_session_id.empty() && state_->policy.max_session_cache_fraction < 1.0;
        const std::size_t session_total_cap =
            enforce_session_total_cap ? SessionTotalCapBytesFromState(*state_) : state_->max_bytes;
        if (enforce_session_total_cap) {
            if (incoming_bytes > session_total_cap) {
                return false;
            }
            // Verify pinned entries of this session alone do not block fitting within session_total_cap.
            std::size_t session_pinned = 0;
            for (const Node* n : state_->lru_list) {
                if (n != nullptr && n->has_entry() && n->record->pin_count > 0 &&
                    n->record->session_id == incoming_session_id) {
                    session_pinned += n->record->bytes;
                }
            }
            if (session_pinned > session_total_cap - incoming_bytes) {
                return false;
            }
        }

        // Release any unpinned existing payload at `overwrite_prefix` FIRST so peak
        // RAM stays strictly <= max_bytes and self-overwrites free their own lease/bytes.
        if (!overwrite_prefix.empty()) {
            if (Node* existing = FindExactNodeLocked(overwrite_prefix);
                existing != nullptr && existing->has_entry() && existing->record->pin_count == 0) {
                DetachEntryLocked(existing, to_free);
                PruneIfEmptyLocked(existing);
            }
        }

        // If this session is requesting a soft lease that would exceed its per-session
        // soft-lease fairness cap, shed its oldest/lowest-value soft leases BEFORE
        // eviction scoring runs so excess same-session entries yield before peers.
        if (incoming_wants_soft_lease && !incoming_session_id.empty()) {
            const std::size_t lease_cap = SessionSoftLeaseCapBytesFromState(*state_);
            if (lease_cap > 0 && incoming_bytes <= lease_cap) {
                (void)MakeRoomForSessionSoftLeaseFromState(*state_, incoming_session_id, incoming_bytes, now_ms,
                                                           nullptr);
            }
        }

        // Enforce optional strict per-session total byte cap by evicting this session's
        // lowest-value unpinned entries first.
        if (enforce_session_total_cap) {
            const std::size_t sess_limit = session_total_cap - incoming_bytes;
            while (GetSessionBytes(state_->session_total_bytes, incoming_session_id) > sess_limit) {
                if (!EvictLowestValueFromState(*state_, to_free, incoming_session_id, incoming_bytes,
                                               incoming_session_id)) {
                    return false;
                }
            }
        }

        while (state_->used_bytes > budget_limit) {
            if (!EvictLowestValueFromState(*state_, to_free, incoming_session_id, incoming_bytes)) {
                return false;
            }
        }
        return true;
    }

    bool EvictLowestValueLocked(std::vector<std::shared_ptr<EntryRecord>>& to_free,
                                std::string_view incoming_session_id = {},
                                std::size_t incoming_bytes = 0) {
        return EvictLowestValueFromState(*state_, to_free, incoming_session_id, incoming_bytes);
    }

    void PruneIfEmptyLocked(Node* node) { PruneIfEmptyFromState(*state_, node); }

    [[nodiscard]] Node* FindExactNodeLocked(std::span<const Token> prefix) const noexcept {
        if (prefix.empty()) return nullptr;
        Node* cur = state_->root.get();
        std::size_t pos = 0;
        while (pos < prefix.size()) {
            const auto it = cur->children.find(prefix[pos]);
            if (it == cur->children.end()) return nullptr;
            Node* child = it->second.get();
            const std::size_t edge_len = child->edge.size();
            if (prefix.size() - pos < edge_len) return nullptr;
            for (std::size_t k = 0; k < edge_len; ++k) {
                if (child->edge[k] != prefix[pos + k]) return nullptr;
            }
            pos += edge_len;
            cur = child;
        }
        return cur;
    }

    Node* EnsureNodeLocked(std::span<const Token> prefix) {
        Node* cur = state_->root.get();
        std::size_t pos = 0;
        while (pos < prefix.size()) {
            const Token key = prefix[pos];
            auto it = cur->children.find(key);
            if (it == cur->children.end()) {
                auto leaf = std::make_unique<Node>();
                leaf->parent = cur;
                leaf->edge.assign(prefix.begin() + static_cast<std::ptrdiff_t>(pos), prefix.end());
                leaf->depth = prefix.size();
                Node* raw = leaf.get();
                cur->children.emplace(key, std::move(leaf));
                return raw;
            }

            Node* child = it->second.get();
            const std::size_t edge_len = child->edge.size();
            const std::size_t rem = prefix.size() - pos;
            const std::size_t cmp_len = std::min(edge_len, rem);
            std::size_t k = 0;
            while (k < cmp_len && child->edge[k] == prefix[pos + k]) {
                ++k;
            }

            if (k == edge_len) {
                pos += edge_len;
                cur = child;
                continue;
            }

            // Split `child` at offset `k`: create internal node for `edge[0..k)`
            // and move `child` beneath it with `edge[k..)`.
            auto split = std::make_unique<Node>();
            split->parent = cur;
            split->edge.assign(child->edge.begin(), child->edge.begin() + static_cast<std::ptrdiff_t>(k));
            split->depth = cur->depth + k;

            std::unique_ptr<Node> existing_child = std::move(it->second);
            split->cached_descendants =
                existing_child->cached_descendants + (existing_child->has_entry() ? 1u : 0u);
            existing_child->edge.erase(existing_child->edge.begin(),
                                       existing_child->edge.begin() + static_cast<std::ptrdiff_t>(k));
            existing_child->parent = split.get();
            const Token existing_key = existing_child->edge.front();
            split->children.emplace(existing_key, std::move(existing_child));

            Node* split_raw = split.get();
            it->second = std::move(split);
            pos += k;
            cur = split_raw;
        }
        return cur;
    }

    [[nodiscard]] Node* LookupNodeLocked(std::span<const Token> prompt, std::size_t max_match_tokens,
                                         bool allow_prefix_trim, std::size_t& out_matched_len) {
        out_matched_len = 0;
        if (state_->max_bytes == 0 || state_->entry_count == 0 || prompt.empty() || max_match_tokens == 0) {
            return nullptr;
        }
        const std::size_t limit = std::min(prompt.size(), max_match_tokens);
        Node* cur = state_->root.get();
        Node* best_exact = nullptr;
        std::size_t pos = 0;

        Node* partial_branch = nullptr;
        std::size_t partial_matched = 0;

        while (pos < limit) {
            const auto it = cur->children.find(prompt[pos]);
            if (it == cur->children.end()) {
                break;
            }
            Node* child = it->second.get();
            const std::size_t edge_len = child->edge.size();
            const std::size_t cmp_limit = std::min(edge_len, limit - pos);
            std::size_t k = 0;
            while (k < cmp_limit && child->edge[k] == prompt[pos + k]) {
                ++k;
            }
            if (k < edge_len) {
                if (k > 0) {
                    partial_branch = child;
                    partial_matched = pos + k;
                }
                break;
            }
            pos += edge_len;
            cur = child;
            if (cur->has_entry()) {
                best_exact = cur;
            }
        }

        // If we stopped cleanly at an internal node `cur` with pos > best_exact->depth,
        // `cur` itself may have descendants that share all `pos` tokens with `prompt`.
        if (partial_branch == nullptr && pos > 0 && (best_exact == nullptr || pos > best_exact->depth)) {
            partial_branch = cur;
            partial_matched = pos;
        }

        Node* chosen = best_exact;
        std::size_t matched_len = best_exact != nullptr ? best_exact->depth : 0;

        if (allow_prefix_trim && partial_branch != nullptr && partial_matched > matched_len) {
            Node* desc = FindShallowestCachedDescendantLocked(partial_branch);
            if (desc != nullptr) {
                chosen = desc;
                matched_len = partial_matched;
            }
        }

        if (chosen == nullptr || !chosen->has_entry() || matched_len == 0) {
            return nullptr;
        }
        out_matched_len = matched_len;
        return chosen;
    }

    // Deterministic, allocation-light search for the best cached descendant:
    // smallest depth first, tie-broken by smallest byte size (cheapest restore),
    // then most recently used (`lru_seq`).
    [[nodiscard]] Node* FindShallowestCachedDescendantLocked(Node* start) {
        if (start == nullptr) return nullptr;
        if (start->has_entry()) return start;

        constexpr std::size_t kStackInlineCap = 64;
        std::array<Node*, kStackInlineCap> inline_stack{};
        std::size_t inline_top = 0;
        auto& overflow = state_->search_scratch;
        overflow.clear();

        auto push_node = [&](Node* p) {
            if (inline_top < kStackInlineCap) {
                inline_stack[inline_top++] = p;
            } else {
                overflow.push_back(p);
            }
        };
        auto pop_node = [&]() -> Node* {
            if (!overflow.empty()) {
                Node* p = overflow.back();
                overflow.pop_back();
                return p;
            }
            return inline_stack[--inline_top];
        };

        auto is_better = [](const Node* cand, const Node* cur_best) noexcept -> bool {
            if (cur_best == nullptr) return true;
            if (cand->depth != cur_best->depth) return cand->depth < cur_best->depth;
            if (cand->record->bytes != cur_best->record->bytes) return cand->record->bytes < cur_best->record->bytes;
            return cand->record->lru_seq > cur_best->record->lru_seq;
        };

        push_node(start);
        Node* best = nullptr;
        while (inline_top > 0 || !overflow.empty()) {
            Node* n = pop_node();
            if (n->has_entry()) {
                if (is_better(n, best)) {
                    best = n;
                }
                continue;
            }
            if (best != nullptr && n->depth >= best->depth) {
                continue;
            }
            for (const auto& [tok, child] : n->children) {
                (void)tok;
                push_node(child.get());
            }
        }
        overflow.clear();
        return best;
    }

    std::shared_ptr<CacheState> state_;
};

using PinnedRadixLookup = RadixPrefixCache::PinnedLookup;

}  // namespace sonder::backends::llamacpp
