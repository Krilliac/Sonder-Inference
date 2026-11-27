#include "kv_telemetry.hpp"

#include <algorithm>
#include <limits>
#include <utility>

namespace sonder::backends::llamacpp {

namespace {

constexpr std::uint64_t kU64Max = std::numeric_limits<std::uint64_t>::max();

// JSON integers are signed 64-bit; clamp so a huge value never turns negative.
std::int64_t ToI64(std::uint64_t v) noexcept {
    constexpr auto kMax = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    return static_cast<std::int64_t>(std::min(v, kMax));
}

std::uint64_t SatMul(std::uint64_t a, std::uint64_t b) noexcept {
    if (a == 0 || b == 0) return 0;
    return a > kU64Max / b ? kU64Max : a * b;
}

std::uint64_t SatAdd(std::uint64_t a, std::uint64_t b) noexcept { return a > kU64Max - b ? kU64Max : a + b; }

KvAttr Attr(std::string key, KvAttrValue v) { return KvAttr{std::move(key), std::move(v)}; }

void AppendObserved(KvEvent& ev, std::uint64_t now_ms) { ev.attrs.push_back(Attr("observed_ms", ToI64(now_ms))); }

}  // namespace

const KvAttrValue* KvEvent::Find(std::string_view key) const noexcept {
    for (const auto& a : attrs) {
        if (a.key == key) return &a.value;
    }
    return nullptr;
}

bool KvTypeLayout(std::string_view n, std::uint32_t& block_elems, std::uint32_t& block_bytes) noexcept {
    struct Entry {
        std::string_view name;
        std::uint32_t elems;
        std::uint32_t bytes;
    };
    // Sizes of ggml's block structs: f32/f16/bf16 are scalar; the Q types are
    // one fp16 scale (plus fp16 min for _1, plus a 4-byte high-bit mask for q5)
    // and 32 packed values.
    static constexpr Entry kTable[] = {
        {"f32", 1, 4},   {"f16", 1, 2},   {"bf16", 1, 2},   {"q8_0", 32, 34}, {"q4_0", 32, 18},
        {"q4_1", 32, 20}, {"q5_0", 32, 22}, {"q5_1", 32, 24}, {"iq4_nl", 32, 18},
    };
    for (const auto& e : kTable) {
        if (e.name == n) {
            block_elems = e.elems;
            block_bytes = e.bytes;
            return true;
        }
    }
    return false;
}

std::optional<std::uint64_t> KvRowBytes(std::string_view ggml_name, std::uint64_t n_elems) noexcept {
    std::uint32_t be = 0;
    std::uint32_t bb = 0;
    if (!KvTypeLayout(ggml_name, be, bb) || be == 0) return std::nullopt;
    if (n_elems % be != 0) return std::nullopt;
    const std::uint64_t blocks = n_elems / be;
    if (blocks != 0 && blocks > kU64Max / bb) return std::nullopt;
    return blocks * bb;
}

std::optional<std::uint64_t> KvBytesPerToken(const KvShape& shape, std::string_view type_k,
                                             std::string_view type_v) noexcept {
    const auto k = KvRowBytes(type_k, shape.n_embd_k_gqa);
    const auto v = KvRowBytes(type_v, shape.n_embd_v_gqa);
    if (!k || !v) return std::nullopt;
    if (*k > kU64Max - *v) return std::nullopt;
    const std::uint64_t per_layer = *k + *v;
    if (shape.n_layer_kv == 0) return std::uint64_t{0};
    if (per_layer > kU64Max / shape.n_layer_kv) return std::nullopt;
    return per_layer * shape.n_layer_kv;
}

KvEvent MakeKvPoolConfiguredEvent(const KvPoolConfig& cfg) {
    KvEvent ev;
    ev.type = "kv.pool.configured";
    ev.attrs = {Attr("scope", std::string("physical")),
                Attr("n_ctx", ToI64(cfg.n_ctx)),
                Attr("n_ctx_seq", ToI64(cfg.n_ctx_seq)),
                Attr("n_seq", static_cast<std::int64_t>(cfg.n_seq)),
                Attr("kv_unified", cfg.kv_unified),
                Attr("type_k", cfg.type_k),
                Attr("type_v", cfg.type_v),
                Attr("bytes_per_token", ToI64(cfg.bytes_per_token)),
                Attr("bytes_per_token_source", cfg.bytes_per_token_source),
                Attr("total_bytes", ToI64(SatMul(cfg.n_ctx, cfg.bytes_per_token))),
                Attr("device", cfg.device)};
    return ev;
}

KvEvent MakeKvUsageEvent(const KvUsageSample& sample, std::uint64_t bytes_per_token) {
    double occupancy = 0.0;
    if (sample.cells_total != 0) {
        occupancy = std::min(1.0, static_cast<double>(sample.cells_used) / static_cast<double>(sample.cells_total));
    }
    KvEvent ev;
    ev.type = "kv.usage";
    ev.attrs = {Attr("scope", std::string("physical")),
                Attr("cells_used", ToI64(sample.cells_used)),
                Attr("cells_total", ToI64(sample.cells_total)),
                Attr("bytes_used", ToI64(SatMul(sample.cells_used, bytes_per_token))),
                Attr("bytes_total", ToI64(SatMul(sample.cells_total, bytes_per_token))),
                Attr("occupancy", occupancy)};
    return ev;
}

KvEvent MakeKvEvictedEvent(std::int64_t seq_id, std::uint64_t tokens, std::uint64_t bytes_per_token,
                           std::string_view reason) {
    KvEvent ev;
    ev.type = "kv.evicted";
    ev.attrs = {Attr("scope", std::string("physical")),
                Attr("seq_id", seq_id),
                Attr("tokens", ToI64(tokens)),
                Attr("bytes", ToI64(SatMul(tokens, bytes_per_token))),
                Attr("reason", std::string(reason))};
    return ev;
}

KvEvent MakeKvShiftedEvent(std::int64_t seq_id, std::int64_t p0, std::int64_t p1, std::int64_t delta,
                           std::uint64_t tokens_discarded, std::uint64_t bytes_per_token) {
    KvEvent ev;
    ev.type = "kv.shifted";
    ev.attrs = {Attr("scope", std::string("physical")),
                Attr("seq_id", seq_id),
                Attr("p0", p0),
                Attr("p1", p1),
                Attr("delta", delta),
                Attr("tokens_discarded", ToI64(tokens_discarded)),
                Attr("bytes_discarded", ToI64(SatMul(tokens_discarded, bytes_per_token)))};
    return ev;
}

// --- limiter ---------------------------------------------------------------

void KvUsageLimiter::AddTokens(std::uint64_t n) noexcept { tokens_since_ = SatAdd(tokens_since_, n); }

bool KvUsageLimiter::WouldAllow(std::uint64_t now_ms) const noexcept {
    if (!emitted_once_) return true;
    const std::uint64_t elapsed = now_ms > last_ms_ ? now_ms - last_ms_ : 0;
    return tokens_since_ >= min_tokens_ && elapsed >= min_interval_ms_;
}

bool KvUsageLimiter::TryAcquire(std::uint64_t now_ms) noexcept {
    if (!WouldAllow(now_ms)) return false;
    emitted_once_ = true;
    last_ms_ = now_ms;
    tokens_since_ = 0;
    return true;
}

void KvUsageLimiter::Reset() noexcept {
    tokens_since_ = 0;
    last_ms_ = 0;
    emitted_once_ = false;
}

// --- KvTelemetry -----------------------------------------------------------

KvTelemetry::KvTelemetry(KvTelemetryOptions options)
    : options_(options), limiter_(options.min_tokens, options.min_interval_ms) {
    queue_.reserve(std::min<std::size_t>(options_.max_queued_events, 64));
}

void KvTelemetry::PushLocked(KvEvent ev) {
    if (queue_.size() >= options_.max_queued_events) {
        ++dropped_;
        return;
    }
    queue_.push_back(std::move(ev));
}

void KvTelemetry::Configure(const KvPoolConfig& cfg, std::uint64_t now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    bytes_per_token_ = cfg.bytes_per_token;
    limiter_.Reset();
    queued_usage_.reset();
    pending_usage_.reset();
    KvEvent ev = MakeKvPoolConfiguredEvent(cfg);
    AppendObserved(ev, now_ms);
    PushLocked(std::move(ev));
}

std::uint64_t KvTelemetry::BytesPerToken() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return bytes_per_token_;
}

void KvTelemetry::OnTokens(std::uint64_t n) {
    std::lock_guard<std::mutex> lock(mutex_);
    limiter_.AddTokens(n);
}

bool KvTelemetry::WantsUsageSample(std::uint64_t now_ms) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return limiter_.WouldAllow(now_ms);
}

void KvTelemetry::ObserveUsage(const KvUsageSample& sample, std::uint64_t now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    KvEvent ev = MakeKvUsageEvent(sample, bytes_per_token_);
    AppendObserved(ev, now_ms);
    if (limiter_.TryAcquire(now_ms)) {
        queued_usage_ = std::move(ev);
        pending_usage_.reset();
    } else {
        pending_usage_ = std::move(ev);
    }
}

void KvTelemetry::NoteEvicted(std::int64_t seq_id, std::uint64_t tokens, std::string_view reason,
                              std::uint64_t now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    KvEvent ev = MakeKvEvictedEvent(seq_id, tokens, bytes_per_token_, reason);
    AppendObserved(ev, now_ms);
    PushLocked(std::move(ev));
}

void KvTelemetry::NoteShifted(std::int64_t seq_id, std::int64_t p0, std::int64_t p1, std::int64_t delta,
                              std::uint64_t tokens_discarded, std::uint64_t now_ms) {
    std::lock_guard<std::mutex> lock(mutex_);
    KvEvent ev = MakeKvShiftedEvent(seq_id, p0, p1, delta, tokens_discarded, bytes_per_token_);
    AppendObserved(ev, now_ms);
    PushLocked(std::move(ev));
}

std::vector<KvEvent> KvTelemetry::Drain() {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<KvEvent> out = std::move(queue_);
    queue_.clear();
    // The suppressed observation, when there is one, is newer than the queued one.
    if (pending_usage_) {
        out.push_back(std::move(*pending_usage_));
    } else if (queued_usage_) {
        out.push_back(std::move(*queued_usage_));
    }
    pending_usage_.reset();
    queued_usage_.reset();
    return out;
}

std::uint64_t KvTelemetry::Dropped() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_;
}

// --- KvTelemetryRegistry ---------------------------------------------------

void KvTelemetryRegistry::Register(std::uint64_t model_id, std::string model, std::shared_ptr<KvTelemetry> kv) {
    if (!kv) return;
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = std::lower_bound(entries_.begin(), entries_.end(), model_id,
                               [](const Entry& e, std::uint64_t id) noexcept { return e.model_id < id; });
    if (it != entries_.end() && it->model_id == model_id) {
        it->model = std::move(model);
        it->kv = std::move(kv);
    } else {
        entries_.insert(it, Entry{model_id, std::move(model), std::move(kv)});
    }
}

void KvTelemetryRegistry::Unregister(std::uint64_t model_id) noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                  [model_id](const Entry& e) noexcept { return e.model_id == model_id; }),
                   entries_.end());
}

std::size_t KvTelemetryRegistry::ActiveCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return entries_.size();
}

std::vector<KvEvent> KvTelemetryRegistry::DrainAll() {
    std::vector<Entry> snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        snapshot = entries_;
    }
    std::vector<KvEvent> out;
    for (const auto& entry : snapshot) {
        for (auto& ev : entry.kv->Drain()) {
            ev.attrs.push_back(Attr("model_id", ToI64(entry.model_id)));
            if (!entry.model.empty()) {
                ev.attrs.push_back(Attr("model", entry.model));
            }
            out.push_back(std::move(ev));
        }
    }
    return out;
}

}  // namespace sonder::backends::llamacpp
