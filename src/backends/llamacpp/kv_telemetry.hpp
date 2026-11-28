// Physical KV-cache telemetry for the native llama.cpp backend.
//
// This header has no llama.cpp dependency: it holds the byte maths for each KV
// cache type, the usage-event rate limiter, the event shapes, and a small
// bounded, thread-safe event queue. The backend feeds it observations and the
// core adapter drains it through BackendRuntimeStatus::take_events, which the
// engine sampler turns into telemetry events (docs/TELEMETRY.md, "Native KV
// cache events").
//
// Events (all `scope: "physical"`; the scheduler's logical `kv.*` events are
// a different stream):
//   kv.pool.configured  a context was created
//   kv.usage            cells used / total, rate limited, gauge
//   kv.evicted          cached sequence tokens were dropped
//   kv.shifted          cached sequence positions were shifted
#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace sonder::backends::llamacpp {

// ---------------------------------------------------------------------------
// Event model (JSON-free so the wrapper library does not need the core JSON).
// ---------------------------------------------------------------------------

using KvAttrValue = std::variant<bool, std::int64_t, double, std::string>;

struct KvAttr {
    std::string key;
    KvAttrValue value;
};

struct KvEvent {
    std::string type;
    std::vector<KvAttr> attrs;

    // First attribute with this key, or nullptr.
    [[nodiscard]] const KvAttrValue* Find(std::string_view key) const noexcept;
};

// ---------------------------------------------------------------------------
// Byte maths
// ---------------------------------------------------------------------------

// Elements per block and bytes per block of a ggml type name as printed by
// ggml_type_name() ("f16", "q8_0", ...). False for a type that cannot be a KV
// cache type here.
[[nodiscard]] bool KvTypeLayout(std::string_view ggml_name, std::uint32_t& block_elems,
                                std::uint32_t& block_bytes) noexcept;

// Bytes of one row of `n_elems` elements (ggml_row_size). nullopt for an
// unknown type, or when n_elems is not a multiple of the block size.
[[nodiscard]] std::optional<std::uint64_t> KvRowBytes(std::string_view ggml_name, std::uint64_t n_elems) noexcept;

// Per-layer KV widths of the layers that keep a token-indexed cache.
struct KvShape {
    std::uint32_t n_layer_kv = 0;
    std::uint64_t n_embd_k_gqa = 0;  // n_head_kv * key head size
    std::uint64_t n_embd_v_gqa = 0;  // n_head_kv * value head size
};

// K+V bytes stored for one token across all layers. nullopt when a type is
// unknown or a width is not a multiple of the type's block size.
[[nodiscard]] std::optional<std::uint64_t> KvBytesPerToken(const KvShape& shape, std::string_view type_k,
                                                           std::string_view type_v) noexcept;

// ---------------------------------------------------------------------------
// Event shapes
// ---------------------------------------------------------------------------

struct KvPoolConfig {
    std::uint64_t n_ctx = 0;      // total cells of the context (llama_n_ctx)
    std::uint64_t n_ctx_seq = 0;  // cells one sequence may use (llama_n_ctx_seq)
    std::uint32_t n_seq = 0;
    bool kv_unified = false;
    std::string type_k;
    std::string type_v;
    std::uint64_t bytes_per_token = 0;
    // "estimate" (derived from model metadata), "upper_bound_hybrid" (a hybrid
    // model: counts every layer, only attention layers keep a token cache), or
    // "unavailable" (bytes_per_token is 0 and unknown).
    std::string bytes_per_token_source = "unavailable";
    std::string device;  // "cpu", "gpu" or a pinned device name
};

struct KvUsageSample {
    std::uint64_t cells_used = 0;
    std::uint64_t cells_total = 0;
};

[[nodiscard]] KvEvent MakeKvPoolConfiguredEvent(const KvPoolConfig& cfg);
[[nodiscard]] KvEvent MakeKvUsageEvent(const KvUsageSample& sample, std::uint64_t bytes_per_token);
[[nodiscard]] KvEvent MakeKvEvictedEvent(std::int64_t seq_id, std::uint64_t tokens, std::uint64_t bytes_per_token,
                                         std::string_view reason);
// Positions [p0, p1) of `seq_id` moved by `delta` (a negative delta follows a
// removal of the same range: the llama_memory_seq_rm + llama_memory_seq_add
// pair of a context shift). `tokens_discarded` is the removed range, if any.
[[nodiscard]] KvEvent MakeKvShiftedEvent(std::int64_t seq_id, std::int64_t p0, std::int64_t p1, std::int64_t delta,
                                         std::uint64_t tokens_discarded, std::uint64_t bytes_per_token);

// ---------------------------------------------------------------------------
// Rate limiter: at most one usage event per `min_tokens` tokens or per
// `min_interval_ms`, whichever comes LATER. Both conditions must hold. Time is
// passed in, so tests need no clock.
// ---------------------------------------------------------------------------

class KvUsageLimiter {
public:
    static constexpr std::uint64_t kDefaultMinTokens = 64;
    static constexpr std::uint64_t kDefaultMinIntervalMs = 500;

    KvUsageLimiter(std::uint64_t min_tokens = kDefaultMinTokens,
                   std::uint64_t min_interval_ms = kDefaultMinIntervalMs) noexcept
        : min_tokens_(min_tokens), min_interval_ms_(min_interval_ms) {}

    void AddTokens(std::uint64_t n) noexcept;
    // Cheap check with no side effect.
    [[nodiscard]] bool WouldAllow(std::uint64_t now_ms) const noexcept;
    // True (and the window restarts) when an event may be emitted now. The
    // first call is always allowed.
    [[nodiscard]] bool TryAcquire(std::uint64_t now_ms) noexcept;
    void Reset() noexcept;

private:
    std::uint64_t min_tokens_;
    std::uint64_t min_interval_ms_;
    std::uint64_t tokens_since_ = 0;
    std::uint64_t last_ms_ = 0;
    bool emitted_once_ = false;
};

// ---------------------------------------------------------------------------
// KvTelemetry: limiter + bounded queue. Thread safe. A usage event is a gauge:
// only the newest one is kept in the queue, and an observation the limiter
// suppressed is kept as "latest" and emitted by the next Drain(), so the last
// state of a burst always reaches the consumer.
// ---------------------------------------------------------------------------

struct KvTelemetryOptions {
    std::uint64_t min_tokens = KvUsageLimiter::kDefaultMinTokens;
    std::uint64_t min_interval_ms = KvUsageLimiter::kDefaultMinIntervalMs;
    std::size_t max_queued_events = 64;  // discrete events; usage is coalesced
};

class KvTelemetry {
public:
    explicit KvTelemetry(KvTelemetryOptions options = {});

    // A context was created: queues kv.pool.configured and restarts the usage
    // window. The pool's bytes_per_token is used for later byte figures.
    void Configure(const KvPoolConfig& cfg, std::uint64_t now_ms);
    [[nodiscard]] std::uint64_t BytesPerToken() const;

    void OnTokens(std::uint64_t n);
    // Cheap pre-check so the backend skips reading cache state when the limiter
    // would refuse and no observation is needed. Never blocks on I/O.
    [[nodiscard]] bool WantsUsageSample(std::uint64_t now_ms) const;
    void ObserveUsage(const KvUsageSample& sample, std::uint64_t now_ms);

    void NoteEvicted(std::int64_t seq_id, std::uint64_t tokens, std::string_view reason, std::uint64_t now_ms);
    void NoteShifted(std::int64_t seq_id, std::int64_t p0, std::int64_t p1, std::int64_t delta,
                     std::uint64_t tokens_discarded, std::uint64_t now_ms);

    // Queued discrete events oldest first, then the newest usage observation (queued or
    // suppressed by the limiter), if any. Every event carries `observed_ms`, the caller's
    // monotonic clock at observation, because the engine drains at its sampler interval.
    [[nodiscard]] std::vector<KvEvent> Drain();
    // Discrete events dropped because the queue was full (cumulative).
    [[nodiscard]] std::uint64_t Dropped() const;

private:
    void PushLocked(KvEvent ev);

    mutable std::mutex mutex_;
    KvTelemetryOptions options_;
    KvUsageLimiter limiter_;
    std::uint64_t bytes_per_token_ = 0;
    std::vector<KvEvent> queue_;
    std::optional<KvEvent> queued_usage_;   // newest usage already allowed by the limiter
    std::optional<KvEvent> pending_usage_;  // newest usage the limiter suppressed
    std::uint64_t dropped_ = 0;
};

// ---------------------------------------------------------------------------
// KvTelemetryRegistry: owns one KvTelemetry per resident model instance so
// multiple concurrently loaded models never share bytes_per_token, limiters,
// or usage gauges, and every drained event carries model attribution.
// ---------------------------------------------------------------------------

class KvTelemetryRegistry {
public:
    // Registers `kv` under unique instance `model_id` and human-readable `model`
    // name. Null `kv` is ignored; re-registering `model_id` replaces its entry.
    void Register(std::uint64_t model_id, std::string model, std::shared_ptr<KvTelemetry> kv);

    // Removes `model_id` when the model unloads. Unknown ids are a no-op.
    void Unregister(std::uint64_t model_id) noexcept;

    [[nodiscard]] std::size_t ActiveCount() const;

    // Drains every registered model's KvTelemetry in ascending `model_id` order
    // and appends `model_id` (int64) and, when non-empty, `model` (str) to each
    // emitted event.
    [[nodiscard]] std::vector<KvEvent> DrainAll();

private:
    struct Entry {
        std::uint64_t model_id = 0;
        std::string model;
        std::shared_ptr<KvTelemetry> kv;
    };

    mutable std::mutex mutex_;
    std::vector<Entry> entries_;
};

}  // namespace sonder::backends::llamacpp
