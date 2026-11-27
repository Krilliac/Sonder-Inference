#include "sonder/inference/session.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

#include "engine/accounting_tokens.hpp"
#include "engine/request_runtime.hpp"
#include "sonder/inference/backends.hpp"
#include "sonder/inference/engine.hpp"

#if defined(SONDER_HAS_SAMPLER_CHAIN)
#include "sonder/sampling/core_bridge.hpp"
#include "sonder/sampling/stop.hpp"
#endif

namespace sonder::inference {

const char* to_string(SessionState state) noexcept {
    switch (state) {
        case SessionState::idle: return "idle";
        case SessionState::running: return "running";
        case SessionState::closed: return "closed";
    }
    return "unknown";
}

const char* to_string(WorkloadClass workload) noexcept {
    switch (workload) {
        case WorkloadClass::interactive_user: return "interactive_user";
        case WorkloadClass::owner_orchestrator: return "owner_orchestrator";
        case WorkloadClass::critic_verification: return "critic_verification";
        case WorkloadClass::implementation_worker: return "implementation_worker";
        case WorkloadClass::research_worker: return "research_worker";
        case WorkloadClass::background_indexing: return "background_indexing";
        case WorkloadClass::maintenance: return "maintenance";
    }
    return "unknown";
}

const char* to_string(RequestOutcome outcome) noexcept {
    switch (outcome) {
        case RequestOutcome::none: return "none";
        case RequestOutcome::completed: return "completed";
        case RequestOutcome::cancelled: return "cancelled";
        case RequestOutcome::failed: return "failed";
    }
    return "unknown";
}

const char* to_string(CheckpointMode mode) noexcept {
    switch (mode) {
        case CheckpointMode::Branch: return "branch";
        case CheckpointMode::ExactResume: return "exact_resume";
    }
    return "unknown";
}

namespace {
// These are backend observations. Keep their names distinct from Sonder's
// logical reused_prompt_tokens, which do not prove physical KV reuse.
void backend_observations(json::Object& attrs, const GenerateStats& stats) {
    if (stats.cached_tokens) attrs.set("backend_cached_tokens", *stats.cached_tokens);
    if (stats.draft_tokens) attrs.set("backend_draft_tokens", *stats.draft_tokens);
    if (stats.draft_accepted_tokens) attrs.set("backend_draft_accepted_tokens", *stats.draft_accepted_tokens);
    if (stats.draft_tokens && *stats.draft_tokens > 0 && stats.draft_accepted_tokens) {
        attrs.set("backend_draft_acceptance_ratio",
                  static_cast<double>(*stats.draft_accepted_tokens) / static_cast<double>(*stats.draft_tokens));
    }
    if (stats.predicted_tokens_per_second) {
        attrs.set("backend_predicted_tokens_per_second", *stats.predicted_tokens_per_second);
    }
}

json::Value sampling_json(const SamplingConfig& s) {
    // With explicit_only, a field the caller did not set was never sent to a
    // model-default backend, so it is recorded as null ("model default"),
    // not as this struct's placeholder value.
    const auto field = [&](SamplingConfig::Field f, json::Value v) {
        return (!s.explicit_only || s.is_explicit(f)) ? std::move(v) : json::Value(nullptr);
    };
    json::Object o{{"temperature", field(SamplingConfig::kTemperature, s.temperature)},
                   {"top_p", field(SamplingConfig::kTopP, s.top_p)},
                   {"top_k", field(SamplingConfig::kTopK, s.top_k)},
                   {"min_p", field(SamplingConfig::kMinP, s.min_p)},
                   {"repeat_penalty", field(SamplingConfig::kRepeatPenalty, s.repeat_penalty)},
                   {"typical_p", s.typical_p},
                   {"repeat_last_n", field(SamplingConfig::kRepeatLastN, s.repeat_last_n)},
                   {"presence_penalty", field(SamplingConfig::kPresencePenalty, s.presence_penalty)},
                   {"frequency_penalty", field(SamplingConfig::kFrequencyPenalty, s.frequency_penalty)},
                   {"explicit_only", s.explicit_only},
                   {"logit_bias_count", static_cast<std::int64_t>(s.logit_bias.size())},
                   {"num_ctx", s.num_ctx},
                   {"seed", s.seed},
                   {"max_tokens", s.max_tokens}};
    return o;
}

// Compatibility fingerprint for prefix sharing: requests may share cached KV
// only for the same backend model (name, format, family, quantization).
std::uint64_t model_fingerprint(const Model& m) {
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const std::string& part) {
        for (const char c : part) {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ull;
        }
        h ^= 0xFFu;
        h *= 1099511628211ull;
    };
    const auto& d = m.descriptor();
    mix(m.backend_name());
    mix(d.name);
    mix(d.format);
    mix(d.family);
    mix(d.quantization);
    return h;
}

double ms_between(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::uint64_t current_process_epoch() noexcept {
    static const std::uint64_t kEpoch = []() noexcept {
        std::uint64_t h = 1469598103934665603ull;
        const std::uint64_t t = monotonic_ns();
        const std::uint64_t addr = reinterpret_cast<std::uintptr_t>(&current_process_epoch);
        auto mix64 = [&](std::uint64_t v) noexcept {
            for (int i = 0; i < 8; ++i) {
                h ^= static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu);
                h *= 1099511628211ull;
            }
        };
        mix64(t);
        mix64(addr);
        mix64(0x517CC1B727220A95ull);
        return h == 0 ? 1ull : h;
    }();
    return kEpoch;
}

std::uint64_t compute_fingerprint_digest(const ModelCompatibilityFingerprint& fp) noexcept {
    std::uint64_t h = 1469598103934665603ull;
    auto mix_u64 = [&](std::uint64_t v) noexcept {
        for (int i = 0; i < 8; ++i) {
            h ^= static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu);
            h *= 1099511628211ull;
        }
    };
    auto mix_str = [&](std::string_view s) noexcept {
        mix_u64(static_cast<std::uint64_t>(s.size()));
        for (const char c : s) {
            h ^= static_cast<unsigned char>(c);
            h *= 1099511628211ull;
        }
        h ^= 0xFFu;
        h *= 1099511628211ull;
    };
    mix_u64(fp.process_epoch);
    mix_str(fp.backend_name);
    mix_str(fp.build_commit);
    mix_str(fp.model_name);
    mix_str(fp.model_format);
    mix_str(fp.model_family);
    mix_str(fp.parameter_size);
    mix_str(fp.quantization);
    mix_u64(fp.model_size_bytes);
    mix_u64(fp.context_length);
    mix_u64(static_cast<std::uint64_t>(fp.architecture));
    mix_u64(fp.tokenizer_hash);
    return h;
}

std::vector<std::string> build_prompt_token_pieces(std::string_view prompt, std::span<const TokenId> tokens) {
    // Matches accounting_tokenize(): leading BOS (empty piece) + 1 piece per word
    // preserving exact original whitespace so prefix slices reconstruct verbatim.
    std::vector<std::string> pieces;
    if (tokens.empty()) {
        return pieces;
    }
    pieces.emplace_back();  // BOS
    std::size_t i = 0;
    while (i < prompt.size()) {
        const std::size_t seg_start = i;
        while (i < prompt.size() && detail::is_space(prompt[i])) ++i;
        const std::size_t word_start = i;
        while (i < prompt.size() && !detail::is_space(prompt[i])) ++i;
        if (i > word_start) {
            pieces.emplace_back(prompt.substr(seg_start, i - seg_start));
        } else if (i > seg_start && !pieces.empty()) {
            pieces.back().append(prompt.substr(seg_start, i - seg_start));
        }
    }
    if (pieces.size() != tokens.size()) {
        pieces.clear();  // Non-word tokenizer (e.g. BPE/SPM); fall back to detokenize.
    }
    return pieces;
}

std::string reconstruct_prefix_text(BackendModel& backend_model, std::span<const TokenId> tokens,
                                    const std::vector<std::string>& pieces, std::size_t offset,
                                    std::string_view full_fallback) {
    if (offset == 0) {
        return {};
    }
    if (offset >= tokens.size() && pieces.size() != tokens.size()) {
        return std::string(full_fallback);
    }
    if (pieces.size() == tokens.size()) {
        std::string out;
        for (std::size_t i = 0; i < offset; ++i) {
            out += pieces[i];
        }
        return out;
    }
    if (auto dt = backend_model.detokenize(tokens.subspan(0, offset)); dt.ok()) {
        return std::move(dt).value();
    }
    return std::string(full_fallback);
}

}  // namespace

bool ModelCompatibilityFingerprint::compatible_with(const ModelCompatibilityFingerprint& other) const noexcept {
    return process_epoch == other.process_epoch && backend_name == other.backend_name &&
           build_commit == other.build_commit && model_name == other.model_name &&
           model_format == other.model_format && model_family == other.model_family &&
           parameter_size == other.parameter_size && quantization == other.quantization &&
           model_size_bytes == other.model_size_bytes && context_length == other.context_length &&
           architecture == other.architecture && tokenizer_hash == other.tokenizer_hash &&
           digest == other.digest;
}

std::string ModelCompatibilityFingerprint::mismatch_reason(const ModelCompatibilityFingerprint& other) const {
    if (process_epoch != other.process_epoch) {
        return "process_epoch mismatch (execution checkpoints are strictly same-process only)";
    }
    if (build_commit != other.build_commit) {
        return "build_commit mismatch ('" + build_commit + "' vs '" + other.build_commit + "')";
    }
    if (backend_name != other.backend_name) {
        return "backend_name mismatch ('" + backend_name + "' vs '" + other.backend_name + "')";
    }
    if (model_name != other.model_name) {
        return "model_name mismatch ('" + model_name + "' vs '" + other.model_name + "')";
    }
    if (model_format != other.model_format) {
        return "model_format mismatch ('" + model_format + "' vs '" + other.model_format + "')";
    }
    if (model_family != other.model_family) {
        return "model_family mismatch ('" + model_family + "' vs '" + other.model_family + "')";
    }
    if (parameter_size != other.parameter_size) {
        return "parameter_size mismatch ('" + parameter_size + "' vs '" + other.parameter_size + "')";
    }
    if (quantization != other.quantization) {
        return "quantization mismatch ('" + quantization + "' vs '" + other.quantization + "')";
    }
    if (model_size_bytes != other.model_size_bytes) {
        return "model_size_bytes mismatch";
    }
    if (context_length != other.context_length) {
        return "context_length mismatch";
    }
    if (architecture != other.architecture) {
        return "model architecture mismatch";
    }
    if (tokenizer_hash != other.tokenizer_hash) {
        return "tokenizer_hash mismatch";
    }
    if (digest != other.digest) {
        return "fingerprint digest mismatch";
    }
    return {};
}

ModelCompatibilityFingerprint compute_model_compatibility_fingerprint(const Model& model) {
    ModelCompatibilityFingerprint fp;
    const auto& d = model.descriptor();
    fp.process_epoch = current_process_epoch();
    fp.backend_name = model.backend_name();
    fp.build_commit = std::string(version_string()) + "@" + std::string(build_commit());
    fp.model_name = d.name;
    fp.model_format = d.format;
    fp.model_family = d.family;
    fp.parameter_size = d.parameter_size;
    fp.quantization = d.quantization;
    fp.model_size_bytes = d.size_bytes;
    fp.context_length = d.context_length;
    fp.architecture = d.architecture;

    std::uint64_t tok_h = 1469598103934665603ull;
    auto mix_tok = [&](std::uint32_t v) noexcept {
        for (int b = 0; b < 4; ++b) {
            tok_h ^= static_cast<std::uint8_t>((v >> (b * 8)) & 0xFFu);
            tok_h *= 1099511628211ull;
        }
    };
    static constexpr std::string_view kProbe = "sonder checkpoint compatibility probe 0123456789";
    if (auto toks = model.backend_model().tokenize(kProbe); toks.ok()) {
        mix_tok(1u);
        mix_tok(static_cast<std::uint32_t>(toks.value().size()));
        for (const TokenId t : toks.value()) {
            mix_tok(static_cast<std::uint32_t>(t));
        }
    } else {
        mix_tok(0u);
        for (const char c : fp.backend_name) {
            tok_h ^= static_cast<unsigned char>(c);
            tok_h *= 1099511628211ull;
        }
        for (const char c : fp.model_name) {
            tok_h ^= static_cast<unsigned char>(c);
            tok_h *= 1099511628211ull;
        }
    }
    fp.tokenizer_hash = tok_h;
    fp.digest = compute_fingerprint_digest(fp);
    return fp;
}

std::size_t ExecutionCheckpointRecord::accounted_bytes() const noexcept {
    std::size_t total = sizeof(ExecutionCheckpointRecord);
    total += handle.checkpoint_id.size();
    total += handle.owner_session_id.size();
    total += prompt.size();
    total += prompt_tokens.size() * sizeof(TokenId);
    for (const auto& p : context_token_pieces) {
        total += sizeof(std::string) + p.size();
    }
    total += generated_tokens.size() * sizeof(TokenId);
    total += delivered_text.size();
    total += sampler_penalty_history.size() * sizeof(TokenId);
    total += grammar.size();
    total += grammar_matcher_state.size();
    total += stop_filter_pending.size();
    total += pending_draft_tokens.size() * sizeof(TokenId);
    total += backend_state_blob.size();
    return total;
}

CheckpointStore::CheckpointStore(ExecutionCheckpointStoreOptions options, ClockFn clock)
    : options_(options), clock_(std::move(clock)) {}

std::chrono::steady_clock::time_point CheckpointStore::now() const {
    return clock_ ? clock_() : std::chrono::steady_clock::now();
}

void CheckpointStore::collect_expired_locked(std::chrono::steady_clock::time_point ts,
                                             std::vector<ExecutionCheckpointRecord>& evicted) {
    for (auto it = lru_.begin(); it != lru_.end();) {
        if (it->record.handle.expired(ts)) {
            total_bytes_ = total_bytes_ >= it->bytes ? (total_bytes_ - it->bytes) : 0;
            index_.erase(it->id);
            evicted.push_back(std::move(it->record));
            it = lru_.erase(it);
            ++evictions_expired_;
        } else {
            ++it;
        }
    }
}

void CheckpointStore::invoke_evicted(const std::vector<ExecutionCheckpointRecord>& evicted) {
    if (evicted.empty()) {
        return;
    }
    EvictCallback cb;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cb = evict_callback_;
    }
    if (!cb) {
        return;
    }
    for (const auto& rec : evicted) {
        cb(rec);
    }
}

Result<ExecutionCheckpointHandle> CheckpointStore::put(ExecutionCheckpointRecord record,
                                                       std::optional<std::chrono::milliseconds> ttl) {
    if (ttl && ttl->count() < 0) {
        return Status(ErrorCode::invalid_argument, "checkpoint TTL must be non-negative");
    }
    if (options_.max_bytes == 0 || options_.max_entries == 0) {
        return Status(ErrorCode::unavailable, "execution checkpoint store is disabled (zero capacity)");
    }
    if (record.handle.checkpoint_id.empty()) {
        record.handle.checkpoint_id = make_id("ckpt");
    }
    record.handle.same_process_only = true;
    const auto ts = now();
    record.handle.created_at = ts;
    const std::chrono::milliseconds effective_ttl = ttl.value_or(options_.default_ttl);
    if (effective_ttl.count() > 0) {
        record.handle.expires_at = ts + effective_ttl;
    } else if (ttl.has_value()) {
        record.handle.expires_at = ts;
    } else {
        record.handle.expires_at = {};
    }

    const std::size_t bytes = record.accounted_bytes();
    record.handle.approx_bytes = bytes;
    if (bytes > options_.max_bytes) {
        return Status(ErrorCode::invalid_argument,
                      "execution checkpoint (" + std::to_string(bytes) +
                          " bytes) exceeds store max_bytes (" + std::to_string(options_.max_bytes) + ")");
    }

    std::vector<ExecutionCheckpointRecord> evicted;
    ExecutionCheckpointHandle handle = record.handle;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        collect_expired_locked(ts, evicted);

        if (auto existing = index_.find(handle.checkpoint_id); existing != index_.end()) {
            const auto lru_it = existing->second;
            total_bytes_ = total_bytes_ >= lru_it->bytes ? (total_bytes_ - lru_it->bytes) : 0;
            evicted.push_back(std::move(lru_it->record));
            lru_.erase(lru_it);
            index_.erase(existing);
        }

        while (!lru_.empty() &&
               (total_bytes_ > options_.max_bytes - bytes || index_.size() >= options_.max_entries)) {
            Entry& victim = lru_.back();
            total_bytes_ = total_bytes_ >= victim.bytes ? (total_bytes_ - victim.bytes) : 0;
            index_.erase(victim.id);
            evicted.push_back(std::move(victim.record));
            lru_.pop_back();
            ++evictions_capacity_;
        }

        lru_.push_front(Entry{handle.checkpoint_id, std::move(record), bytes});
        index_.emplace(handle.checkpoint_id, lru_.begin());
        total_bytes_ += bytes;
    }
    invoke_evicted(evicted);
    return handle;
}

Result<ExecutionCheckpointRecord> CheckpointStore::get(const std::string& checkpoint_id) {
    if (checkpoint_id.empty()) {
        return Status(ErrorCode::invalid_argument, "checkpoint_id is empty");
    }
    std::vector<ExecutionCheckpointRecord> evicted;
    std::optional<ExecutionCheckpointRecord> found;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        collect_expired_locked(now(), evicted);
        auto it = index_.find(checkpoint_id);
        if (it != index_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            found = it->second->record;
        }
    }
    invoke_evicted(evicted);
    if (!found) {
        return Status(ErrorCode::not_found, "execution checkpoint not found or expired: " + checkpoint_id);
    }
    return std::move(*found);
}

bool CheckpointStore::release(const std::string& checkpoint_id) {
    if (checkpoint_id.empty()) {
        return false;
    }
    std::vector<ExecutionCheckpointRecord> evicted;
    bool removed = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = index_.find(checkpoint_id);
        if (it != index_.end()) {
            const auto lru_it = it->second;
            total_bytes_ = total_bytes_ >= lru_it->bytes ? (total_bytes_ - lru_it->bytes) : 0;
            evicted.push_back(std::move(lru_it->record));
            lru_.erase(lru_it);
            index_.erase(it);
            ++releases_;
            removed = true;
        }
    }
    invoke_evicted(evicted);
    return removed;
}

std::size_t CheckpointStore::release_owned_by(std::string_view session_id) {
    if (session_id.empty()) {
        return 0;
    }
    std::vector<ExecutionCheckpointRecord> evicted;
    std::size_t count = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto it = lru_.begin(); it != lru_.end();) {
            if (it->record.handle.owner_session_id == session_id) {
                total_bytes_ = total_bytes_ >= it->bytes ? (total_bytes_ - it->bytes) : 0;
                index_.erase(it->id);
                evicted.push_back(std::move(it->record));
                it = lru_.erase(it);
                ++releases_;
                ++count;
            } else {
                ++it;
            }
        }
    }
    invoke_evicted(evicted);
    return count;
}

std::size_t CheckpointStore::sweep_expired() {
    std::vector<ExecutionCheckpointRecord> evicted;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        collect_expired_locked(now(), evicted);
    }
    const std::size_t n = evicted.size();
    invoke_evicted(evicted);
    return n;
}

void CheckpointStore::clear() {
    std::vector<ExecutionCheckpointRecord> evicted;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        evicted.reserve(lru_.size());
        for (auto& e : lru_) {
            evicted.push_back(std::move(e.record));
        }
        lru_.clear();
        index_.clear();
        total_bytes_ = 0;
    }
    invoke_evicted(evicted);
}

void CheckpointStore::set_evict_callback(EvictCallback cb) {
    std::lock_guard<std::mutex> lock(mutex_);
    evict_callback_ = std::move(cb);
}

ExecutionCheckpointStoreStats CheckpointStore::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    ExecutionCheckpointStoreStats s;
    s.entries = index_.size();
    s.total_bytes = total_bytes_;
    s.max_bytes = options_.max_bytes;
    s.max_entries = options_.max_entries;
    s.evictions_capacity = evictions_capacity_;
    s.evictions_expired = evictions_expired_;
    s.releases = releases_;
    return s;
}

Session::Session(Engine& engine, std::shared_ptr<Model> model, SessionOptions options)
    : engine_(engine), model_(std::move(model)), options_(std::move(options)) {
    engine_.telemetry().emit("session.created", telemetry_context(),
                             json::Object{{"model", model_->descriptor().name},
                                          {"backend", model_->backend_name()},
                                          {"priority", options_.priority},
                                          {"workload", to_string(options_.workload)},
                                          {"text_capture", engine_.telemetry().options().capture_text ? "on" : "off"},
                                          {"sampling", sampling_json(options_.sampling)}},
                             TelemetryLevel::metrics);
}

Session::~Session() { close(); }

SessionState Session::state() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return state_;
}

RequestOutcome Session::last_outcome() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_outcome_;
}

std::uint64_t Session::requests_started() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return requests_started_;
}

TelemetryContext Session::telemetry_context(const std::optional<std::string>& request_id) const {
    TelemetryContext ctx;
    ctx.session_id = options_.session_id;
    // Defaults to the engine id so session/request events group with the
    // engine's own events (Observatory run grouping).
    ctx.run_id = options_.run_id.value_or(engine_.engine_id());
    ctx.request_id = request_id;
    ctx.agent_id = options_.agent_id;
    ctx.task_id = options_.task_id;
    ctx.model_instance_id = model_->instance_id();
    ctx.device_id = model_->device_id();
    ctx.synthetic_work = model_->backend_name() == kMockBackendName;
    return ctx;
}

void Session::cancel() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_cancel_) {
        std::uint64_t expected = 0;
        cancel_requested_ns_.compare_exchange_strong(expected, monotonic_ns());
        active_cancel_->cancel();
    }
}

void Session::request_suspend(SuspendAction action) {
    suspend_requested_.store(static_cast<std::uint8_t>(action), std::memory_order_release);
}

void Session::close() {
    std::uint64_t seq_to_release = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return;
        }
        if (active_cancel_) {
            std::uint64_t expected = 0;
            cancel_requested_ns_.compare_exchange_strong(expected, monotonic_ns());
            active_cancel_->cancel();
        }
        state_ = SessionState::closed;
        seq_to_release = logical_kv_seq_id_;
        logical_kv_seq_id_ = 0;
        last_continuation_.reset();
    }
    if (seq_to_release != 0) {
        if (detail::RequestRuntime* rt = engine_.request_runtime()) {
            rt->release_logical_sequence(seq_to_release);
        }
    }
    (void)engine_.checkpoint_store().release_owned_by(options_.session_id);
    engine_.telemetry().emit("session.closed", telemetry_context(),
                             json::Object{{"requests", requests_started()}}, TelemetryLevel::metrics);
}

std::size_t Session::context_token_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return context_tokens_.size();
}

std::vector<TokenId> Session::context_tokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return context_tokens_;
}

std::string Session::context_prompt() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return context_prompt_;
}

std::uint64_t Session::logical_kv_sequence_id() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return logical_kv_seq_id_;
}

ModelCompatibilityFingerprint Session::compatibility_fingerprint() const {
    return compute_model_compatibility_fingerprint(*model_);
}

Result<std::shared_ptr<Session>> Session::fork(const ForkOptions& options) {
    std::string parent_prompt;
    std::vector<TokenId> parent_tokens;
    std::vector<std::string> parent_pieces;
    bool parent_exact = false;
    std::uint64_t parent_seq_id = 0;
    std::vector<std::uint8_t> parent_recurrent_blob;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "cannot fork a closed session");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "cannot fork while a request is in flight");
        }
        parent_prompt = context_prompt_;
        parent_tokens = context_tokens_;
        parent_pieces = context_token_pieces_;
        parent_exact = exact_context_tokens_;
        parent_seq_id = logical_kv_seq_id_;
        parent_recurrent_blob = last_recurrent_state_blob_;
    }

    const std::size_t target_len = options.prefix_token_offset.value_or(parent_tokens.size());
    if (target_len > parent_tokens.size()) {
        return Status(ErrorCode::invalid_argument,
                      "fork prefix_token_offset (" + std::to_string(target_len) +
                          ") exceeds session context length (" + std::to_string(parent_tokens.size()) + ")");
    }

    SessionOptions child_opts = options.session_options;
    if (options.inherit_session_options) {
        SessionOptions inherited = options_;
        inherited.session_id = options.session_options.session_id;
        if (options.session_options.run_id) inherited.run_id = options.session_options.run_id;
        if (options.session_options.agent_id) inherited.agent_id = options.session_options.agent_id;
        if (options.session_options.task_id) inherited.task_id = options.session_options.task_id;
        child_opts = std::move(inherited);
    }
    if (child_opts.session_id.empty()) {
        child_opts.session_id = make_id("sess");
    }
    if (Status st = validate(child_opts.sampling); !st.ok()) {
        return st;
    }

    std::uint64_t child_seq_id = 0;
    detail::RequestRuntime* runtime = engine_.request_runtime();
    if (runtime != nullptr && !parent_tokens.empty()) {
        const std::int32_t rank = static_cast<std::int32_t>(options_.workload) - options_.priority;
        const std::int32_t cache_prio = std::clamp(6 - rank, 0, 255);
        if (parent_seq_id == 0) {
            auto ensured = runtime->ensure_logical_sequence(
                0, parent_tokens, model_fingerprint(*model_), model_->descriptor().architecture, cache_prio,
                parent_recurrent_blob);
            if (!ensured.ok()) {
                return ensured.status();
            }
            parent_seq_id = ensured.value().sequence_id;
            std::lock_guard<std::mutex> lock(mutex_);
            if (state_ == SessionState::closed) {
                runtime->release_logical_sequence(parent_seq_id);
                return Status(ErrorCode::invalid_state, "parent session closed during fork");
            }
            if (logical_kv_seq_id_ == 0) {
                logical_kv_seq_id_ = parent_seq_id;
            } else {
                runtime->release_logical_sequence(parent_seq_id);
                parent_seq_id = logical_kv_seq_id_;
            }
        }
        const std::int32_t child_rank = static_cast<std::int32_t>(child_opts.workload) - child_opts.priority;
        const std::int32_t child_cache_prio = std::clamp(6 - child_rank, 0, 255);
        std::optional<std::size_t> rewind_opt =
            target_len < parent_tokens.size() ? std::optional<std::size_t>(target_len) : std::nullopt;
        auto forked = runtime->fork_logical_sequence(parent_seq_id, rewind_opt, child_cache_prio);
        if (!forked.ok()) {
            return forked.status();
        }
        child_seq_id = forked.value().sequence_id;
    }

    std::vector<TokenId> child_tokens(parent_tokens.begin(),
                                      parent_tokens.begin() + static_cast<std::ptrdiff_t>(target_len));
    std::vector<std::string> child_pieces;
    if (parent_pieces.size() == parent_tokens.size()) {
        child_pieces.assign(parent_pieces.begin(),
                            parent_pieces.begin() + static_cast<std::ptrdiff_t>(target_len));
    }
    const std::string child_prompt =
        target_len == parent_tokens.size()
            ? parent_prompt
            : reconstruct_prefix_text(model_->backend_model(), parent_tokens, parent_pieces, target_len, parent_prompt);

    auto child = std::make_shared<Session>(engine_, model_, std::move(child_opts));
    {
        std::lock_guard<std::mutex> lock(child->mutex_);
        child->context_prompt_ = child_prompt;
        child->context_tokens_ = std::move(child_tokens);
        child->context_token_pieces_ = std::move(child_pieces);
        child->exact_context_tokens_ = parent_exact;
        child->branched_context_ = true;
        child->logical_kv_seq_id_ = child_seq_id;
        child->last_continuation_.reset();  // Branch starts with fresh sampler/stop/grammar state.
    }

    engine_.telemetry().emit("session.forked", telemetry_context(),
                             json::Object{{"child_session_id", child->id()},
                                          {"mode", to_string(CheckpointMode::Branch)},
                                          {"token_offset", static_cast<std::uint64_t>(target_len)},
                                          {"parent_tokens", static_cast<std::uint64_t>(parent_tokens.size())},
                                          {"logical_kv_sequence_id", child_seq_id}},
                             TelemetryLevel::metrics);
    return child;
}

Status Session::rewind(std::size_t token_offset) {
    std::string cur_prompt;
    std::vector<TokenId> cur_tokens;
    std::vector<std::string> cur_pieces;
    std::uint64_t seq_id = 0;
    std::vector<std::uint8_t> recurrent_blob;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "cannot rewind a closed session");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "cannot rewind while a request is in flight");
        }
        if (token_offset > context_tokens_.size()) {
            return Status(ErrorCode::invalid_argument,
                          "rewind token_offset (" + std::to_string(token_offset) +
                              ") exceeds session context length (" + std::to_string(context_tokens_.size()) + ")");
        }
        cur_prompt = context_prompt_;
        cur_tokens = context_tokens_;
        cur_pieces = context_token_pieces_;
        seq_id = logical_kv_seq_id_;
        recurrent_blob = last_recurrent_state_blob_;
    }

    detail::RequestRuntime* runtime = engine_.request_runtime();
    if (runtime != nullptr && !cur_tokens.empty()) {
        const std::int32_t rank = static_cast<std::int32_t>(options_.workload) - options_.priority;
        const std::int32_t cache_prio = std::clamp(6 - rank, 0, 255);
        if (seq_id == 0) {
            auto ensured = runtime->ensure_logical_sequence(
                0, cur_tokens, model_fingerprint(*model_), model_->descriptor().architecture, cache_prio,
                recurrent_blob);
            if (!ensured.ok()) {
                return ensured.status();
            }
            seq_id = ensured.value().sequence_id;
        }
        auto truncated = runtime->truncate_logical_sequence(seq_id, token_offset);
        if (!truncated.ok()) {
            if (logical_kv_sequence_id() == 0) {
                runtime->release_logical_sequence(seq_id);
            }
            return truncated.status();
        }
    }

    const std::string rewound_prompt =
        reconstruct_prefix_text(model_->backend_model(), cur_tokens, cur_pieces, token_offset, cur_prompt);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            if (seq_id != 0 && runtime != nullptr) {
                runtime->release_logical_sequence(seq_id);
            }
            return Status(ErrorCode::invalid_state, "session closed during rewind");
        }
        context_tokens_.resize(token_offset);
        if (context_token_pieces_.size() >= token_offset) {
            context_token_pieces_.resize(token_offset);
        } else {
            context_token_pieces_.clear();
        }
        context_prompt_ = rewound_prompt;
        logical_kv_seq_id_ = seq_id;
        branched_context_ = true;
        last_continuation_.reset();
    }

    engine_.telemetry().emit("session.rewound", telemetry_context(),
                             json::Object{{"token_offset", static_cast<std::uint64_t>(token_offset)},
                                          {"previous_tokens", static_cast<std::uint64_t>(cur_tokens.size())},
                                          {"logical_kv_sequence_id", seq_id}},
                             TelemetryLevel::metrics);
    return Status::success();
}

Result<ExecutionCheckpointHandle> Session::export_checkpoint(const CheckpointExportOptions& options) {
    std::string cur_prompt;
    std::vector<TokenId> cur_tokens;
    std::vector<std::string> cur_pieces;
    bool cur_exact = false;
    std::uint64_t cur_seq_id = 0;
    std::optional<ExecutionCheckpointRecord> cur_cont;
    std::vector<std::uint8_t> cur_recurrent_blob;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "cannot export checkpoint from a closed session");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "cannot export checkpoint while a request is in flight");
        }
        cur_prompt = context_prompt_;
        cur_tokens = context_tokens_;
        cur_pieces = context_token_pieces_;
        cur_exact = exact_context_tokens_;
        cur_seq_id = logical_kv_seq_id_;
        cur_cont = last_continuation_;
        cur_recurrent_blob = last_recurrent_state_blob_;
    }

    detail::RequestRuntime* runtime = engine_.request_runtime();
    const ModelCompatibilityFingerprint fp = compatibility_fingerprint();

    ExecutionCheckpointRecord rec;
    rec.handle.owner_session_id = options_.session_id;
    rec.handle.mode = options.mode;
    rec.handle.compatibility = fp;
    rec.handle.same_process_only = true;

    if (options.mode == CheckpointMode::Branch) {
        const std::size_t target_len = options.token_offset.value_or(cur_tokens.size());
        if (target_len > cur_tokens.size()) {
            return Status(ErrorCode::invalid_argument,
                          "export_checkpoint token_offset (" + std::to_string(target_len) +
                              ") exceeds session context length (" + std::to_string(cur_tokens.size()) + ")");
        }
        std::uint64_t ckpt_seq_id = 0;
        if (runtime != nullptr && !cur_tokens.empty()) {
            const std::int32_t rank = static_cast<std::int32_t>(options_.workload) - options_.priority;
            const std::int32_t cache_prio = std::clamp(6 - rank, 0, 255);
            if (cur_seq_id == 0) {
                auto ensured = runtime->ensure_logical_sequence(
                    0, cur_tokens, model_fingerprint(*model_), model_->descriptor().architecture, cache_prio,
                    cur_recurrent_blob);
                if (!ensured.ok()) {
                    return ensured.status();
                }
                cur_seq_id = ensured.value().sequence_id;
                std::lock_guard<std::mutex> lock(mutex_);
                if (logical_kv_seq_id_ == 0 && state_ != SessionState::closed) {
                    logical_kv_seq_id_ = cur_seq_id;
                } else {
                    runtime->release_logical_sequence(cur_seq_id);
                    cur_seq_id = logical_kv_seq_id_;
                }
            }
            std::optional<std::size_t> rewind_opt =
                target_len < cur_tokens.size() ? std::optional<std::size_t>(target_len) : std::nullopt;
            auto forked = runtime->fork_logical_sequence(cur_seq_id, rewind_opt, cache_prio);
            if (!forked.ok()) {
                return forked.status();
            }
            ckpt_seq_id = forked.value().sequence_id;
        }
        rec.logical_kv_sequence_id = ckpt_seq_id;
        rec.prompt = target_len == cur_tokens.size()
                         ? cur_prompt
                         : reconstruct_prefix_text(model_->backend_model(), cur_tokens, cur_pieces, target_len,
                                                   cur_prompt);
        rec.prompt_tokens.assign(cur_tokens.begin(), cur_tokens.begin() + static_cast<std::ptrdiff_t>(target_len));
        if (cur_pieces.size() == cur_tokens.size()) {
            rec.context_token_pieces.assign(cur_pieces.begin(),
                                            cur_pieces.begin() + static_cast<std::ptrdiff_t>(target_len));
        }
        rec.exact_tokens = cur_exact;
        rec.handle.token_position = static_cast<std::uint64_t>(target_len);
        // Branch explicitly resets continuation state (fresh sampling, stop, grammar, offset).
        rec.delivered_offset_bytes = 0;
        rec.delivered_chunks = 0;
        rec.delivered_completion_tokens = 0;
        rec.has_sampler_rng_state = false;
    } else {
        if (!cur_cont.has_value()) {
            return Status(ErrorCode::invalid_state,
                          "ExactResume export requires an active/interrupted generation state on this session");
        }
        if (options.token_offset.has_value() && *options.token_offset != cur_cont->handle.token_position) {
            return Status(ErrorCode::invalid_argument,
                          "ExactResume cannot rewind to an arbitrary token_offset; use CheckpointMode::Branch instead");
        }
        rec = *cur_cont;
        rec.handle.checkpoint_id.clear();
        rec.handle.owner_session_id = options_.session_id;
        rec.handle.mode = CheckpointMode::ExactResume;
        rec.handle.compatibility = fp;
        rec.handle.same_process_only = true;
        if (!options.pending_draft_tokens.empty()) {
            rec.pending_draft_tokens = options.pending_draft_tokens;
        }
        if (!options.grammar_matcher_state.empty()) {
            rec.grammar_matcher_state = options.grammar_matcher_state;
        }
        rec.logical_kv_sequence_id = 0;
        if (runtime != nullptr && cur_seq_id != 0) {
            auto forked = runtime->fork_logical_sequence(cur_seq_id);
            if (forked.ok()) {
                rec.logical_kv_sequence_id = forked.value().sequence_id;
            }
        }
    }

    const std::uint64_t ckpt_kv_seq = rec.logical_kv_sequence_id;
    auto stored = engine_.checkpoint_store().put(std::move(rec), options.ttl);
    if (!stored.ok()) {
        if (ckpt_kv_seq != 0 && runtime != nullptr) {
            runtime->release_logical_sequence(ckpt_kv_seq);
        }
        return stored.status();
    }
    engine_.telemetry().emit("session.checkpoint.exported", telemetry_context(),
                             json::Object{{"checkpoint_id", stored.value().checkpoint_id},
                                          {"mode", to_string(options.mode)},
                                          {"token_position", stored.value().token_position},
                                          {"approx_bytes", stored.value().approx_bytes},
                                          {"same_process_only", true}},
                             TelemetryLevel::metrics);
    return stored;
}

Result<GenerationResult> Session::resume_from_checkpoint(const ExecutionCheckpointHandle& handle,
                                                         const TokenCallback& on_chunk,
                                                         const ResumeOptions& resume_options) {
    if (!handle.valid()) {
        return Status(ErrorCode::invalid_argument, "checkpoint handle is empty/invalid");
    }
    if (!handle.same_process_only) {
        return Status(ErrorCode::invalid_argument,
                      "cross-process/cross-host execution checkpoints are not supported");
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "session is closed");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "session already has a request in flight");
        }
    }
    const ModelCompatibilityFingerprint live_fp = compatibility_fingerprint();
    if (!handle.compatibility.compatible_with(live_fp)) {
        return Status(ErrorCode::invalid_argument,
                      "incompatible execution checkpoint handle: " + handle.compatibility.mismatch_reason(live_fp));
    }

    auto fetched = engine_.checkpoint_store().get(handle.checkpoint_id);
    if (!fetched.ok()) {
        return fetched.status();
    }
    ExecutionCheckpointRecord rec = std::move(fetched).value();
    if (!rec.handle.compatibility.compatible_with(live_fp)) {
        return Status(ErrorCode::invalid_argument,
                      "incompatible stored execution checkpoint: " + rec.handle.compatibility.mismatch_reason(live_fp));
    }
    if (rec.handle.mode != handle.mode) {
        return Status(ErrorCode::invalid_argument, "checkpoint handle mode does not match stored checkpoint");
    }

    if (rec.handle.mode == CheckpointMode::Branch) {
        if (resume_options.request_options.grammar_hooks && resume_options.request_options.grammar_hooks->reset) {
            resume_options.request_options.grammar_hooks->reset();
        }
        std::string prompt = rec.prompt;
        if (!resume_options.branch_prompt_suffix.empty()) {
            if (!prompt.empty() && !detail::is_space(prompt.back()) &&
                !detail::is_space(resume_options.branch_prompt_suffix.front())) {
                prompt.push_back(' ');
            }
            prompt += resume_options.branch_prompt_suffix;
        }
        std::uint64_t new_seq_id = 0;
        if (rec.logical_kv_sequence_id != 0) {
            if (detail::RequestRuntime* rt = engine_.request_runtime()) {
                auto forked = rt->fork_logical_sequence(rec.logical_kv_sequence_id);
                if (forked.ok()) {
                    new_seq_id = forked.value().sequence_id;
                }
            }
        }
        std::uint64_t old_seq_id = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            old_seq_id = logical_kv_seq_id_;
            logical_kv_seq_id_ = new_seq_id;
            context_prompt_ = rec.prompt;
            context_tokens_ = rec.prompt_tokens;
            context_token_pieces_ = rec.context_token_pieces;
            exact_context_tokens_ = rec.exact_tokens;
            branched_context_ = false;  // `prompt` already includes the branched prefix.
            last_continuation_.reset();
        }
        if (old_seq_id != 0) {
            if (detail::RequestRuntime* rt = engine_.request_runtime()) {
                rt->release_logical_sequence(old_seq_id);
            }
        }
        if (resume_options.consume_checkpoint) {
            (void)engine_.checkpoint_store().release(handle.checkpoint_id);
        }
        return run_request("generate", prompt, nullptr, on_chunk, resume_options.sampling_override,
                           resume_options.request_options);
    }

    // ExactResume semantics: must continue from exact interrupted boundary.
    if (!resume_options.branch_prompt_suffix.empty()) {
        return Status(ErrorCode::invalid_argument,
                      "ExactResume does not accept branch_prompt_suffix; use CheckpointMode::Branch to branch");
    }
    if (resume_options.sampling_override.has_value()) {
        return Status(ErrorCode::invalid_argument,
                      "ExactResume preserves checkpointed sampling/RNG/stop state; use CheckpointMode::Branch to "
                      "override sampling or max_additional_tokens to extend token budget");
    }
    if (resume_options.max_additional_tokens.has_value() && *resume_options.max_additional_tokens < 0) {
        return Status(ErrorCode::invalid_argument, "max_additional_tokens must be >= 0");
    }
    if (resume_options.consume_checkpoint) {
        (void)engine_.checkpoint_store().release(handle.checkpoint_id);
    }
    RequestOptions effective_ro = resume_options.request_options;
    if (effective_ro.grammar.empty() && !rec.grammar.empty()) {
        effective_ro.grammar = rec.grammar;
    }
    return run_request("generate", rec.prompt, nullptr, on_chunk, rec.sampling, effective_ro, &rec,
                       resume_options.max_additional_tokens);
}

bool Session::release_checkpoint(const ExecutionCheckpointHandle& handle) {
    if (!handle.valid()) {
        return false;
    }
    return engine_.checkpoint_store().release(handle.checkpoint_id);
}

#if defined(SONDER_HAS_SAMPLER_CHAIN)
namespace {

class HooksConstraint final : public sampling::Constraint {
public:
    explicit HooksConstraint(GrammarMatcherHooks hooks) : hooks_(std::move(hooks)) {}

    std::string_view name() const noexcept override { return "grammar_hooks"; }

    void apply(sampling::Candidates& candidates) override {
        if (!hooks_.mask_logits || candidates.data.empty()) {
            return;
        }
        std::size_t max_id = 0;
        for (const auto& item : candidates.data) {
            if (item.id >= 0 && static_cast<std::size_t>(item.id) + 1 > max_id) {
                max_id = static_cast<std::size_t>(item.id) + 1;
            }
        }
        if (max_id == 0) {
            return;
        }
        buf_.assign(max_id, sampling::kNegInf);
        for (const auto& item : candidates.data) {
            if (item.id >= 0) {
                buf_[static_cast<std::size_t>(item.id)] = item.logit;
            }
        }
        hooks_.mask_logits(std::span<float>(buf_));
        for (auto& item : candidates.data) {
            if (item.id >= 0) {
                item.logit = buf_[static_cast<std::size_t>(item.id)];
            }
        }
        candidates.sorted = false;
    }

    void accept(sampling::TokenId token) override {
        if (hooks_.accept_token) {
            hooks_.accept_token(token);
        }
    }

    void reset() override {
        if (hooks_.reset) {
            hooks_.reset();
        }
    }

    std::vector<std::uint8_t> snapshot_state() const override {
        return hooks_.snapshot_state ? hooks_.snapshot_state() : std::vector<std::uint8_t>{};
    }

    bool restore_state(std::span<const std::uint8_t> state) override {
        if (hooks_.restore_state) {
            return hooks_.restore_state(state);
        }
        return state.empty();
    }

private:
    GrammarMatcherHooks hooks_;
    std::vector<float> buf_;
};

// Token-level decode loop with Sonder's sampler chain (Capability::token_logits).
// NoViableCandidates maps to invalid_argument: it only happens when the
// request's own policy (bias/penalties/constraints) excludes every
// token, so retrying the same request cannot succeed. EmptyLogits is a backend
// fault (backend_error).
// Supports both fresh generation and ExactResume continuation (restoring RNG,
// penalty history, grammar matcher state, stop-filter pending buffer, delivered
// offset, and backend stream state) plus non-blocking safe-point suspension.
template <class Gate, class Produced, class Deliver, class PollSafePoint>
Result<GenerateStats> run_sonder_sampling(
    TokenStream& stream, BackendModel& backend_model, const SamplingConfig& config, const CancellationToken& cancel,
    TelemetryBus& bus, const TelemetryContext& ctx, const Gate& gate, const Produced& produced,
    const Deliver& deliver, const PollSafePoint& poll_safe_point, const RequestOptions& request_options,
    const ExecutionCheckpointRecord* exact_resume, std::optional<std::int32_t> max_additional_tokens,
    ExecutionCheckpointRecord* out_continuation, SuspendedExecutionState* out_suspended,
    std::vector<TokenId>& out_generated_tokens, std::vector<std::string>& out_generated_pieces,
    StageTimings* stage_timings = nullptr) {
    namespace smp = sampling;
    GenerateStats stats;
    stats.prompt_tokens = stream.prompt_tokens().size();
    stats.token_counts_from_backend = true;

    std::shared_ptr<smp::Constraint> constraint;
    if (request_options.grammar_hooks) {
        constraint = std::make_shared<HooksConstraint>(*request_options.grammar_hooks);
    }

    auto built = smp::make_chain(config, constraint, stream.vocab_size());
    if (!built.ok()) {
        return built.status();
    }
    smp::SamplerChain& chain = built.value();
    smp::StopSequenceMatcher stop(config.stop);

    std::uint64_t prior_completion_tokens = 0;
    std::int32_t step_limit = config.max_tokens;
    if (exact_resume != nullptr) {
        // Restore backend stream state (spill blob) or replay generated tokens (recompute).
        bool restored_stream = false;
        if (!exact_resume->backend_state_blob.empty()) {
            Status rst = stream.restore_state(exact_resume->backend_state_blob,
                                              exact_resume->backend_state_token_position);
            if (rst.ok()) {
                restored_stream = true;
                stats.cached_tokens = exact_resume->backend_state_token_position;
            } else if (rst.code() != ErrorCode::unsupported) {
                return rst;
            }
        }
        if (!restored_stream) {
            for (const TokenId t : exact_resume->generated_tokens) {
                std::string ignored;
                if (Status st = stream.accept(t, ignored); !st.ok()) {
                    return st;
                }
            }
        }
        smp::SamplerChain::StateSnapshot snap;
        snap.rng_state = exact_resume->sampler_rng_state;
        snap.penalty_history = exact_resume->sampler_penalty_history;
        snap.constraint_state = exact_resume->grammar_matcher_state;
        if (exact_resume->has_sampler_rng_state) {
            if (!chain.restore_state(snap)) {
                return Status(ErrorCode::invalid_argument,
                              "failed to restore sampler/grammar continuation state on ExactResume");
            }
        } else {
            chain.accept_prompt(stream.prompt_tokens());
            for (const TokenId t : exact_resume->generated_tokens) {
                chain.accept(t);
            }
        }
        stop.restore_state(exact_resume->stop_filter_pending, exact_resume->stop_filter_stopped);
        stats.chunks = exact_resume->delivered_chunks;
        prior_completion_tokens = exact_resume->delivered_completion_tokens;
        out_generated_tokens = exact_resume->generated_tokens;
        if (max_additional_tokens.has_value()) {
            step_limit = std::max<std::int32_t>(0, *max_additional_tokens);
        } else {
            const std::uint64_t cap = static_cast<std::uint64_t>(std::max<std::int32_t>(0, config.max_tokens));
            step_limit = cap > prior_completion_tokens
                             ? static_cast<std::int32_t>(std::min<std::uint64_t>(
                                   cap - prior_completion_tokens,
                                   static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())))
                             : 0;
        }
    } else {
        chain.accept_prompt(stream.prompt_tokens());
    }

    {
        json::Array stages;
        for (const auto name : chain.stage_names()) {
            stages.emplace_back(std::string(name));
        }
        bus.emit("sampling.configured", ctx,
                 json::Object{{"sampler", "sonder"},
                              {"selector", config.temperature == 0.0f ? "greedy" : "distribution"},
                              {"stages", std::move(stages)},
                              {"seed", config.seed},
                              {"vocab_size", stream.vocab_size()},
                              {"resumed", exact_resume != nullptr}},
                 TelemetryLevel::metrics);
    }

    auto capture_continuation_state = [&](SuspendAction suspend_act, std::size_t max_spill_bytes,
                                          bool do_suspend) {
        std::vector<std::uint8_t> stream_blob;
        if (!do_suspend || suspend_act == SuspendAction::yield_spill) {
            if (auto cap = stream.capture_state(max_spill_bytes); cap.ok()) {
                stream_blob = std::move(cap).value();
            }
        }
        const bool reclaimed = do_suspend ? stream.reclaim_physical_state() : false;
        const std::uint64_t total_gen = prior_completion_tokens + stats.completion_tokens;
        const std::uint64_t total_pos = static_cast<std::uint64_t>(stream.prompt_tokens().size()) + total_gen;
        if (do_suspend && out_suspended != nullptr) {
            out_suspended->action = suspend_act;
            out_suspended->token_position = total_pos;
            out_suspended->generated_tokens = total_gen;
            out_suspended->tokens = stream.prompt_tokens();
            out_suspended->tokens.insert(out_suspended->tokens.end(), out_generated_tokens.begin(),
                                         out_generated_tokens.end());
            out_suspended->backend_state_blob = stream_blob;
            out_suspended->physical_slot_reclaimed = reclaimed;
        }
        if (out_continuation != nullptr) {
            const auto snap = chain.snapshot_state();
            out_continuation->prompt_tokens = stream.prompt_tokens();
            out_continuation->generated_tokens = out_generated_tokens;
            out_continuation->delivered_chunks = stats.chunks;
            out_continuation->delivered_completion_tokens = total_gen;
            out_continuation->sampling = config;
            out_continuation->sampler_rng_state = snap.rng_state;
            out_continuation->has_sampler_rng_state = true;
            out_continuation->sampler_penalty_history = snap.penalty_history;
            out_continuation->grammar = request_options.grammar;
            out_continuation->grammar_matcher_state = snap.constraint_state;
            out_continuation->stop_filter_pending = stop.pending();
            out_continuation->stop_filter_stopped = stop.stopped();
            out_continuation->backend_state_blob = std::move(stream_blob);
            out_continuation->backend_state_token_position = total_pos;
            out_continuation->handle.token_position = total_pos;
            out_continuation->handle.delivered_chunks = stats.chunks;
            out_continuation->handle.delivered_completion_tokens = total_gen;
        }
    };

    const auto t0 = std::chrono::steady_clock::now();
    // Pre-decode safe point.
    {
        SafePointContext sp;
        sp.prompt_tokens = stream.prompt_tokens().size();
        sp.prefill_completed_tokens = stream.prompt_tokens().size();
        sp.generated_tokens = prior_completion_tokens;
        sp.kv_tokens = stream.prompt_tokens().size() + prior_completion_tokens;
        sp.in_prefill = true;
        const SuspendDecision dec = poll_safe_point(sp);
        if (dec.action != SuspendAction::continue_running) {
            capture_continuation_state(dec.action, dec.max_spill_bytes, /*do_suspend=*/true);
            stats.stop_reason = StopReason::suspended;
            stats.eval_ns = static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
            return stats;
        }
    }

    for (std::int32_t i = 0; i < step_limit; ++i) {
        if (cancel.cancelled()) {
            return Status(ErrorCode::cancelled, "cancelled by caller");
        }
        if (!gate()) {
            stats.stop_reason = StopReason::callback;
            break;
        }
        auto logits = stream.next_logits(cancel);
        if (!logits.ok()) {
            return logits.status();
        }
        ScopedStageTimer sample_timer(stage_timings != nullptr ? &stage_timings->sampling_ms : nullptr);
        const smp::SampleResult sampled = chain.sample(logits.value());
        sample_timer.stop();
        if (!sampled.ok()) {
            const bool no_viable = sampled.status == smp::SampleStatus::NoViableCandidates;
            const ErrorCode code = no_viable ? ErrorCode::invalid_argument : ErrorCode::backend_error;
            bus.emit("sampling.failed", ctx,
                     json::Object{{"status", no_viable ? "no_viable_candidates" : "empty_logits"},
                                  {"error_code", to_string(code)},
                                  {"position", i}},
                     TelemetryLevel::metrics);
            return Status(code, no_viable ? "sampling failed: no viable candidates (bias, penalties or "
                                            "constraints excluded every token)"
                                          : "sampling failed: backend returned empty logits");
        }
        chain.accept(sampled.token);
        produced(sampled.token);
        if (stream.is_end_of_generation(sampled.token) || chain.is_stop_token(sampled.token)) {
            stats.stop_reason = StopReason::end_of_sequence;
            break;
        }
        std::string piece;
        if (Status st = stream.accept(sampled.token, piece); !st.ok()) {
            return st;
        }
        out_generated_tokens.push_back(sampled.token);
        out_generated_pieces.push_back(piece);
        ++stats.completion_tokens;
        const smp::StopCheck check = stop.feed(piece);
        bool stopped_by_callback = false;
        if (!check.emit.empty()) {
            TokenChunk chunk{check.emit, stats.chunks++};
            if (config.logprobs) {
                const auto raw = logits.value();
                double max_l = -std::numeric_limits<double>::infinity();
                for (const float v : raw) {
                    if (std::isfinite(v) && static_cast<double>(v) > max_l) max_l = static_cast<double>(v);
                }
                double sum_exp = 0.0;
                if (std::isfinite(max_l)) {
                    for (const float v : raw) {
                        if (std::isfinite(v)) sum_exp += std::exp(static_cast<double>(v) - max_l);
                    }
                }
                const double lse = (std::isfinite(max_l) && sum_exp > 0.0) ? (max_l + std::log(sum_exp)) : 0.0;
                if (sampled.token >= 0 && static_cast<std::size_t>(sampled.token) < raw.size() &&
                    std::isfinite(raw[static_cast<std::size_t>(sampled.token)])) {
                    chunk.logprob =
                        static_cast<float>(static_cast<double>(raw[static_cast<std::size_t>(sampled.token)]) - lse);
                } else {
                    chunk.logprob = sampled.probability > 0.0f ? std::log(sampled.probability) : -1e9f;
                }
                if (config.top_logprobs > 0 && !raw.empty()) {
                    const std::size_t k =
                        std::min<std::size_t>(static_cast<std::size_t>(config.top_logprobs), raw.size());
                    std::vector<std::size_t> order(raw.size());
                    std::iota(order.begin(), order.end(), 0);
                    std::partial_sort(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(k), order.end(),
                                      [&](std::size_t a, std::size_t b) { return raw[a] > raw[b]; });
                    chunk.top_logprobs.reserve(k);
                    for (std::size_t j = 0; j < k; ++j) {
                        const TokenId tid = static_cast<TokenId>(order[j]);
                        std::string tok_str;
                        if (tid == sampled.token) {
                            tok_str = piece;
                        } else if (auto dt = backend_model.detokenize(std::span<const TokenId>(&tid, 1)); dt.ok()) {
                            tok_str = std::move(dt).value();
                        } else {
                            tok_str = std::to_string(tid);
                        }
                        const float tlp = std::isfinite(raw[order[j]])
                                              ? static_cast<float>(static_cast<double>(raw[order[j]]) - lse)
                                              : -1e9f;
                        chunk.top_logprobs.push_back(TopLogprob{std::move(tok_str), tlp});
                    }
                }
            }
            if (!deliver(chunk, sampled.token, sampled.probability)) {
                stats.stop_reason = StopReason::callback;
                stopped_by_callback = true;
            }
        }
        if (stopped_by_callback) {
            break;
        }
        if (check.stopped) {
            stats.stop_reason = StopReason::stop_sequence;
            if (check.stop_index < config.stop.size()) {
                stats.matched_stop = config.stop[check.stop_index];
            }
            break;
        }
        SafePointContext sp;
        sp.prompt_tokens = stream.prompt_tokens().size();
        sp.prefill_completed_tokens = stream.prompt_tokens().size();
        sp.generated_tokens = prior_completion_tokens + stats.completion_tokens;
        sp.kv_tokens = stream.prompt_tokens().size() + sp.generated_tokens;
        sp.in_prefill = false;
        const SuspendDecision dec = poll_safe_point(sp);
        if (dec.action != SuspendAction::continue_running) {
            capture_continuation_state(dec.action, dec.max_spill_bytes, /*do_suspend=*/true);
            stats.stop_reason = StopReason::suspended;
            break;
        }
    }
    if (stats.stop_reason == StopReason::none) {
        stats.stop_reason = StopReason::max_tokens;
    }
    if (stats.stop_reason == StopReason::max_tokens || stats.stop_reason == StopReason::end_of_sequence) {
        const std::string rest = stop.flush();
        if (!rest.empty()) {
            const TokenChunk chunk{rest, stats.chunks++};
            (void)deliver(chunk, std::nullopt, 0.0f);
        }
    }
    if (stats.stop_reason != StopReason::suspended) {
        capture_continuation_state(SuspendAction::continue_running, kMaxRecurrentCheckpointBytes, /*do_suspend=*/false);
    }
    stats.eval_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count());
    return stats;
}

}  // namespace
#endif

bool Session::last_scheduler_rejected() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return last_scheduler_rejected_;
}

Result<GenerationResult> Session::generate(const std::string& prompt, const TokenCallback& on_chunk,
                                           const std::optional<SamplingConfig>& sampling_override,
                                           const RequestOptions& request_options) {
    return run_request("generate", prompt, nullptr, on_chunk, sampling_override, request_options);
}

Result<GenerationResult> Session::chat(const std::vector<ChatMessage>& messages, const TokenCallback& on_chunk,
                                       const std::optional<SamplingConfig>& sampling_override,
                                       const RequestOptions& request_options) {
    if (Status st = validate_chat_messages(messages); !st.ok()) {
        return st;
    }
    return run_request("chat", format_chat_prompt(messages), &messages, on_chunk, sampling_override,
                       request_options);
}

Result<EmbedResult> Session::embed(const std::vector<std::string>& inputs, const RequestOptions& request_options) {
    EmbedRequest req;
    req.inputs = inputs;
    return embed_request(req, request_options);
}

Result<EmbedResult> Session::embed_request(const EmbedRequest& request, const RequestOptions& request_options) {
    if (request.inputs.empty()) {
        return Status(ErrorCode::invalid_argument, "embed inputs must be non-empty");
    }
    if (request.inputs.size() > kMaxEmbedInputs) {
        return Status(ErrorCode::invalid_argument,
                      "embed inputs exceeds maximum of " + std::to_string(kMaxEmbedInputs) + " items");
    }
    if (request.dimensions.has_value() && *request.dimensions == 0) {
        return Status(ErrorCode::invalid_argument, "embed dimensions must be positive");
    }
    if (request.pooling_type == EmbedPoolingType::rank) {
        return Status(ErrorCode::invalid_argument,
                      "backend does not support pooling_type 'rank' for dense embeddings");
    }
    const std::string req_id = request_options.request_id && !request_options.request_id->empty()
                                   ? *request_options.request_id
                                   : (!request.request_id.empty() ? request.request_id : make_id("req"));
    TelemetryBus& bus = engine_.telemetry();
    const auto ctx = telemetry_context(req_id);
    const std::uint64_t input_count = static_cast<std::uint64_t>(request.inputs.size());
    bool terminal_attempted = false;
    auto t_start = std::chrono::steady_clock::now();
    const auto current_queue_ms = [&]() {
        return request_options.admission ? request_options.admission->queue_ms()
                                         : request_options.admission_queue_ms;
    };
    const auto& parent = request_options.parent_request_id;
    const auto with_parent = [&parent](json::Object attrs) {
        if (parent) {
            attrs.set("parent_request_id", *parent);
        }
        return attrs;
    };
    const auto with_admission = [&](json::Object attrs, double queue_ms = 0.0) {
        if (request_options.priority_class) {
            attrs.set("priority_class", to_string(*request_options.priority_class));
            attrs.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                {"queue_ms", queue_ms}});
        }
        return attrs;
    };
    const auto recover_embed = [&]() noexcept {
        try {
            if (!terminal_attempted) {
                terminal_attempted = true;
                json::Object failed{{"kind", "embed"},
                                    {"outcome", "failed"},
                                    {"inputs", input_count},
                                    {"total_ms", ms_between(t_start, std::chrono::steady_clock::now())},
                                    {"error_code", to_string(ErrorCode::internal)},
                                    {"error", "request aborted by exception"}};
                bus.emit("request.failed", ctx,
                         with_parent(with_admission(std::move(failed), current_queue_ms())),
                         TelemetryLevel::metrics);
            }
        } catch (...) {
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            active_cancel_.reset();
            cancel_requested_ns_.store(0);
            last_outcome_ = RequestOutcome::failed;
            last_scheduler_rejected_ = false;
            if (state_ == SessionState::running) {
                state_ = SessionState::idle;
            }
        } catch (...) {
        }
    };
    struct EmbedFailureGuard {
        const decltype(recover_embed)& recover;
        bool armed = false;
        ~EmbedFailureGuard() noexcept {
            if (armed) recover();
        }
    } failure_guard{recover_embed};

    CancellationToken token;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "session is closed");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "session already has a request in flight");
        }
        active_cancel_.emplace();
        token = active_cancel_->token();
        state_ = SessionState::running;
        cancel_requested_ns_.store(0);
        ++requests_started_;
        failure_guard.armed = true;
    }

    auto finish_state = [&](RequestOutcome outcome) {
        failure_guard.armed = false;
        std::lock_guard<std::mutex> lock(mutex_);
        active_cancel_.reset();
        cancel_requested_ns_.store(0);
        last_outcome_ = outcome;
        last_scheduler_rejected_ = false;
        if (state_ == SessionState::running) {
            state_ = SessionState::idle;
        }
    };

    auto emit_terminal = [&](RequestOutcome outcome, json::Object attrs) {
        const double total_ms = ms_between(t_start, std::chrono::steady_clock::now());
        attrs.set("kind", "embed");
        attrs.set("outcome", to_string(outcome));
        if (!attrs.find("inputs")) {
            attrs.set("inputs", input_count);
        }
        if (!attrs.find("total_ms")) {
            attrs.set("total_ms", total_ms);
        }
        if (outcome == RequestOutcome::cancelled) {
            const std::uint64_t cancel_ns = cancel_requested_ns_.load();
            const std::uint64_t now_ns = monotonic_ns();
            if (cancel_ns != 0 && now_ns >= cancel_ns) {
                attrs.set("cancel_latency_ms", static_cast<double>(now_ns - cancel_ns) / 1e6);
            }
        }
        const char* event_type = outcome == RequestOutcome::completed ? "request.completed"
                                 : outcome == RequestOutcome::cancelled ? "request.cancelled"
                                                                        : "request.failed";
        terminal_attempted = true;
        bus.emit(event_type, ctx, with_parent(with_admission(std::move(attrs), current_queue_ms())),
                 TelemetryLevel::metrics);
        finish_state(outcome);
    };

    t_start = std::chrono::steady_clock::now();
    bus.emit("request.queued", ctx,
             with_parent(with_admission(
                 json::Object{{"kind", "embed"},
                              {"priority", options_.priority},
                              {"workload", to_string(options_.workload)},
                              {"inputs", input_count}},
                 request_options.admission_queue_ms)),
             TelemetryLevel::metrics);
    bus.emit("request.started", ctx,
             with_parent(with_admission(
                 json::Object{{"kind", "embed"},
                              {"inputs", input_count},
                              {"scheduled", false}},
                 request_options.admission_queue_ms)),
             TelemetryLevel::metrics);

    if (request_options.admission) {
        Status st = request_options.admission->wait(token);
        if (!st.ok()) {
            const bool cancelled = token.cancelled() || st.code() == ErrorCode::cancelled;
            json::Object attrs;
            if (!cancelled) {
                attrs.set("error_code", to_string(st.code()));
                attrs.set("error", to_string(st.code()));
            }
            emit_terminal(cancelled ? RequestOutcome::cancelled : RequestOutcome::failed, std::move(attrs));
            return st;
        }
    }

    if (request_options.deadline && std::chrono::steady_clock::now() >= *request_options.deadline) {
        emit_terminal(RequestOutcome::failed,
                      json::Object{{"error_code", to_string(ErrorCode::timeout)},
                                   {"error", to_string(ErrorCode::timeout)}});
        return Status(ErrorCode::timeout, "request deadline exceeded before backend start");
    }
    if (token.cancelled()) {
        emit_terminal(RequestOutcome::cancelled, json::Object{});
        return Status(ErrorCode::cancelled, "cancelled before backend start");
    }
    if (request_options.on_backend_start) {
        request_options.on_backend_start();
    }
    if (token.cancelled()) {
        emit_terminal(RequestOutcome::cancelled, json::Object{});
        return Status(ErrorCode::cancelled, "cancelled before backend start");
    }

    EmbedRequest req = request;
    req.request_id = req_id;
    auto embedded = model_->backend_model().embed(req, token);
    const double total_ms = ms_between(t_start, std::chrono::steady_clock::now());

    if (!embedded.ok()) {
        const bool cancelled = token.cancelled() || embedded.status().code() == ErrorCode::cancelled;
        json::Object attrs{{"total_ms", total_ms}};
        if (!cancelled) {
            // Never emit raw upstream error text into shared Observatory
            // telemetry (M-15): only record the safe structured error_code.
            attrs.set("error_code", to_string(embedded.status().code()));
            attrs.set("error", to_string(embedded.status().code()));
        }
        emit_terminal(cancelled ? RequestOutcome::cancelled : RequestOutcome::failed, std::move(attrs));
        return embedded.status();
    }

    if (embedded.value().model.empty()) {
        embedded.value().model = model_->descriptor().name;
    }
    if (embedded.value().dimensions == 0 && !embedded.value().embeddings.empty()) {
        embedded.value().dimensions = embedded.value().embeddings.front().size();
    }
    if (embedded.value().encoding_format.empty()) {
        embedded.value().encoding_format = req.encoding_format.empty() ? std::string("float") : req.encoding_format;
    }

    emit_terminal(RequestOutcome::completed,
                  json::Object{{"prompt_tokens", embedded.value().prompt_tokens},
                               {"inputs", static_cast<std::uint64_t>(embedded.value().embeddings.size())},
                               {"total_ms", total_ms}});
    return embedded;
}

Result<RerankResult> Session::rerank(const RerankRequest& request, const RequestOptions& request_options) {
    if (Status st = validate_rerank_request(request); !st.ok()) {
        return st;
    }
    const std::string req_id = request_options.request_id && !request_options.request_id->empty()
                                   ? *request_options.request_id
                                   : (!request.request_id.empty() ? request.request_id : make_id("req"));
    TelemetryBus& bus = engine_.telemetry();
    const auto ctx = telemetry_context(req_id);
    const std::uint64_t doc_count = static_cast<std::uint64_t>(request.documents.size());
    bool terminal_attempted = false;
    auto t_start = std::chrono::steady_clock::now();
    const auto current_queue_ms = [&]() {
        return request_options.admission ? request_options.admission->queue_ms()
                                         : request_options.admission_queue_ms;
    };
    const auto& parent = request_options.parent_request_id;
    const auto with_parent = [&parent](json::Object attrs) {
        if (parent) {
            attrs.set("parent_request_id", *parent);
        }
        return attrs;
    };
    const auto with_admission = [&](json::Object attrs, double queue_ms = 0.0) {
        if (request_options.priority_class) {
            attrs.set("priority_class", to_string(*request_options.priority_class));
            attrs.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                {"queue_ms", queue_ms}});
        }
        return attrs;
    };
    const auto recover_rerank = [&]() noexcept {
        try {
            if (!terminal_attempted) {
                terminal_attempted = true;
                json::Object failed{{"kind", "rerank"},
                                    {"outcome", "failed"},
                                    {"documents", doc_count},
                                    {"total_ms", ms_between(t_start, std::chrono::steady_clock::now())},
                                    {"error_code", to_string(ErrorCode::internal)},
                                    {"error", "request aborted by exception"}};
                bus.emit("request.failed", ctx,
                         with_parent(with_admission(std::move(failed), current_queue_ms())),
                         TelemetryLevel::metrics);
            }
        } catch (...) {
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            active_cancel_.reset();
            cancel_requested_ns_.store(0);
            last_outcome_ = RequestOutcome::failed;
            last_scheduler_rejected_ = false;
            if (state_ == SessionState::running) {
                state_ = SessionState::idle;
            }
        } catch (...) {
        }
    };
    struct RerankFailureGuard {
        const decltype(recover_rerank)& recover;
        bool armed = false;
        ~RerankFailureGuard() noexcept {
            if (armed) recover();
        }
    } failure_guard{recover_rerank};

    CancellationToken token;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "session is closed");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "session already has a request in flight");
        }
        active_cancel_.emplace();
        token = active_cancel_->token();
        state_ = SessionState::running;
        cancel_requested_ns_.store(0);
        ++requests_started_;
        failure_guard.armed = true;
    }

    auto finish_state = [&](RequestOutcome outcome) {
        failure_guard.armed = false;
        std::lock_guard<std::mutex> lock(mutex_);
        active_cancel_.reset();
        cancel_requested_ns_.store(0);
        last_outcome_ = outcome;
        last_scheduler_rejected_ = false;
        if (state_ == SessionState::running) {
            state_ = SessionState::idle;
        }
    };

    auto emit_terminal = [&](RequestOutcome outcome, json::Object attrs) {
        const double total_ms = ms_between(t_start, std::chrono::steady_clock::now());
        attrs.set("kind", "rerank");
        attrs.set("outcome", to_string(outcome));
        if (!attrs.find("documents")) {
            attrs.set("documents", doc_count);
        }
        if (!attrs.find("total_ms")) {
            attrs.set("total_ms", total_ms);
        }
        if (outcome == RequestOutcome::cancelled) {
            const std::uint64_t cancel_ns = cancel_requested_ns_.load();
            const std::uint64_t now_ns = monotonic_ns();
            if (cancel_ns != 0 && now_ns >= cancel_ns) {
                attrs.set("cancel_latency_ms", static_cast<double>(now_ns - cancel_ns) / 1e6);
            }
        }
        const char* event_type = outcome == RequestOutcome::completed ? "request.completed"
                                 : outcome == RequestOutcome::cancelled ? "request.cancelled"
                                                                        : "request.failed";
        terminal_attempted = true;
        bus.emit(event_type, ctx, with_parent(with_admission(std::move(attrs), current_queue_ms())),
                 TelemetryLevel::metrics);
        finish_state(outcome);
    };

    t_start = std::chrono::steady_clock::now();
    bus.emit("request.queued", ctx,
             with_parent(with_admission(
                 json::Object{{"kind", "rerank"},
                              {"priority", options_.priority},
                              {"workload", to_string(options_.workload)},
                              {"documents", doc_count}},
                 request_options.admission_queue_ms)),
             TelemetryLevel::metrics);
    bus.emit("request.started", ctx,
             with_parent(with_admission(
                 json::Object{{"kind", "rerank"},
                              {"documents", doc_count},
                              {"scheduled", false}},
                 request_options.admission_queue_ms)),
             TelemetryLevel::metrics);

    if (request_options.admission) {
        Status st = request_options.admission->wait(token);
        if (!st.ok()) {
            const bool cancelled = token.cancelled() || st.code() == ErrorCode::cancelled;
            json::Object attrs;
            if (!cancelled) {
                attrs.set("error_code", to_string(st.code()));
                attrs.set("error", to_string(st.code()));
            }
            emit_terminal(cancelled ? RequestOutcome::cancelled : RequestOutcome::failed, std::move(attrs));
            return st;
        }
    }

    if (request_options.deadline && std::chrono::steady_clock::now() >= *request_options.deadline) {
        emit_terminal(RequestOutcome::failed,
                      json::Object{{"error_code", to_string(ErrorCode::timeout)},
                                   {"error", to_string(ErrorCode::timeout)}});
        return Status(ErrorCode::timeout, "request deadline exceeded before backend start");
    }
    if (token.cancelled()) {
        emit_terminal(RequestOutcome::cancelled, json::Object{});
        return Status(ErrorCode::cancelled, "cancelled before backend start");
    }
    if (request_options.on_backend_start) {
        request_options.on_backend_start();
    }
    if (token.cancelled()) {
        emit_terminal(RequestOutcome::cancelled, json::Object{});
        return Status(ErrorCode::cancelled, "cancelled before backend start");
    }

    RerankRequest req = request;
    req.request_id = req_id;
    auto ranked = model_->backend_model().rerank(req, token);
    const double total_ms = ms_between(t_start, std::chrono::steady_clock::now());

    if (!ranked.ok()) {
        const bool cancelled = token.cancelled() || ranked.status().code() == ErrorCode::cancelled;
        json::Object attrs{{"total_ms", total_ms}};
        if (!cancelled) {
            attrs.set("error_code", to_string(ranked.status().code()));
            attrs.set("error", to_string(ranked.status().code()));
        }
        emit_terminal(cancelled ? RequestOutcome::cancelled : RequestOutcome::failed, std::move(attrs));
        return ranked.status();
    }

    if (ranked.value().model.empty()) {
        ranked.value().model = model_->descriptor().name;
    }

    emit_terminal(RequestOutcome::completed,
                  json::Object{{"prompt_tokens", ranked.value().prompt_tokens},
                               {"documents", doc_count},
                               {"results", static_cast<std::uint64_t>(ranked.value().results.size())},
                               {"total_ms", total_ms}});
    return ranked;
}

Result<ScoreResult> Session::score(const ScoreRequest& request, const RequestOptions& request_options) {
    if (Status st = validate_score_request(request); !st.ok()) {
        return st;
    }
    const std::string req_id = request_options.request_id && !request_options.request_id->empty()
                                   ? *request_options.request_id
                                   : (!request.request_id.empty() ? request.request_id : make_id("req"));
    TelemetryBus& bus = engine_.telemetry();
    const auto ctx = telemetry_context(req_id);
    const std::uint64_t pair_count = static_cast<std::uint64_t>(request.pairs.size());
    bool terminal_attempted = false;
    auto t_start = std::chrono::steady_clock::now();
    const auto current_queue_ms = [&]() {
        return request_options.admission ? request_options.admission->queue_ms()
                                         : request_options.admission_queue_ms;
    };
    const auto& parent = request_options.parent_request_id;
    const auto with_parent = [&parent](json::Object attrs) {
        if (parent) {
            attrs.set("parent_request_id", *parent);
        }
        return attrs;
    };
    const auto with_admission = [&](json::Object attrs, double queue_ms = 0.0) {
        if (request_options.priority_class) {
            attrs.set("priority_class", to_string(*request_options.priority_class));
            attrs.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                {"queue_ms", queue_ms}});
        }
        return attrs;
    };
    const auto recover_score = [&]() noexcept {
        try {
            if (!terminal_attempted) {
                terminal_attempted = true;
                json::Object failed{{"kind", "score"},
                                    {"outcome", "failed"},
                                    {"pairs", pair_count},
                                    {"total_ms", ms_between(t_start, std::chrono::steady_clock::now())},
                                    {"error_code", to_string(ErrorCode::internal)},
                                    {"error", "request aborted by exception"}};
                bus.emit("request.failed", ctx,
                         with_parent(with_admission(std::move(failed), current_queue_ms())),
                         TelemetryLevel::metrics);
            }
        } catch (...) {
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            active_cancel_.reset();
            cancel_requested_ns_.store(0);
            last_outcome_ = RequestOutcome::failed;
            last_scheduler_rejected_ = false;
            if (state_ == SessionState::running) {
                state_ = SessionState::idle;
            }
        } catch (...) {
        }
    };
    struct ScoreFailureGuard {
        const decltype(recover_score)& recover;
        bool armed = false;
        ~ScoreFailureGuard() noexcept {
            if (armed) recover();
        }
    } failure_guard{recover_score};

    CancellationToken token;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "session is closed");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "session already has a request in flight");
        }
        active_cancel_.emplace();
        token = active_cancel_->token();
        state_ = SessionState::running;
        cancel_requested_ns_.store(0);
        ++requests_started_;
        failure_guard.armed = true;
    }

    auto finish_state = [&](RequestOutcome outcome) {
        failure_guard.armed = false;
        std::lock_guard<std::mutex> lock(mutex_);
        active_cancel_.reset();
        cancel_requested_ns_.store(0);
        last_outcome_ = outcome;
        last_scheduler_rejected_ = false;
        if (state_ == SessionState::running) {
            state_ = SessionState::idle;
        }
    };

    auto emit_terminal = [&](RequestOutcome outcome, json::Object attrs) {
        const double total_ms = ms_between(t_start, std::chrono::steady_clock::now());
        attrs.set("kind", "score");
        attrs.set("outcome", to_string(outcome));
        if (!attrs.find("pairs")) {
            attrs.set("pairs", pair_count);
        }
        if (!attrs.find("total_ms")) {
            attrs.set("total_ms", total_ms);
        }
        if (outcome == RequestOutcome::cancelled) {
            const std::uint64_t cancel_ns = cancel_requested_ns_.load();
            const std::uint64_t now_ns = monotonic_ns();
            if (cancel_ns != 0 && now_ns >= cancel_ns) {
                attrs.set("cancel_latency_ms", static_cast<double>(now_ns - cancel_ns) / 1e6);
            }
        }
        const char* event_type = outcome == RequestOutcome::completed ? "request.completed"
                                 : outcome == RequestOutcome::cancelled ? "request.cancelled"
                                                                        : "request.failed";
        terminal_attempted = true;
        bus.emit(event_type, ctx, with_parent(with_admission(std::move(attrs), current_queue_ms())),
                 TelemetryLevel::metrics);
        finish_state(outcome);
    };

    t_start = std::chrono::steady_clock::now();
    bus.emit("request.queued", ctx,
             with_parent(with_admission(
                 json::Object{{"kind", "score"},
                              {"priority", options_.priority},
                              {"workload", to_string(options_.workload)},
                              {"pairs", pair_count}},
                 request_options.admission_queue_ms)),
             TelemetryLevel::metrics);
    bus.emit("request.started", ctx,
             with_parent(with_admission(
                 json::Object{{"kind", "score"},
                              {"pairs", pair_count},
                              {"scheduled", false}},
                 request_options.admission_queue_ms)),
             TelemetryLevel::metrics);

    if (request_options.admission) {
        Status st = request_options.admission->wait(token);
        if (!st.ok()) {
            const bool cancelled = token.cancelled() || st.code() == ErrorCode::cancelled;
            json::Object attrs;
            if (!cancelled) {
                attrs.set("error_code", to_string(st.code()));
                attrs.set("error", to_string(st.code()));
            }
            emit_terminal(cancelled ? RequestOutcome::cancelled : RequestOutcome::failed, std::move(attrs));
            return st;
        }
    }

    if (request_options.deadline && std::chrono::steady_clock::now() >= *request_options.deadline) {
        emit_terminal(RequestOutcome::failed,
                      json::Object{{"error_code", to_string(ErrorCode::timeout)},
                                   {"error", to_string(ErrorCode::timeout)}});
        return Status(ErrorCode::timeout, "request deadline exceeded before backend start");
    }
    if (token.cancelled()) {
        emit_terminal(RequestOutcome::cancelled, json::Object{});
        return Status(ErrorCode::cancelled, "cancelled before backend start");
    }
    if (request_options.on_backend_start) {
        request_options.on_backend_start();
    }
    if (token.cancelled()) {
        emit_terminal(RequestOutcome::cancelled, json::Object{});
        return Status(ErrorCode::cancelled, "cancelled before backend start");
    }

    ScoreRequest req = request;
    req.request_id = req_id;
    auto scored = model_->backend_model().score(req, token);
    const double total_ms = ms_between(t_start, std::chrono::steady_clock::now());

    if (!scored.ok()) {
        const bool cancelled = token.cancelled() || scored.status().code() == ErrorCode::cancelled;
        json::Object attrs{{"total_ms", total_ms}};
        if (!cancelled) {
            attrs.set("error_code", to_string(scored.status().code()));
            attrs.set("error", to_string(scored.status().code()));
        }
        emit_terminal(cancelled ? RequestOutcome::cancelled : RequestOutcome::failed, std::move(attrs));
        return scored.status();
    }

    if (scored.value().model.empty()) {
        scored.value().model = model_->descriptor().name;
    }

    emit_terminal(RequestOutcome::completed,
                  json::Object{{"prompt_tokens", scored.value().prompt_tokens},
                               {"pairs", pair_count},
                               {"total_ms", total_ms}});
    return scored;
}

Result<GenerationResult> Session::run_request(const char* kind, const std::string& prompt,
                                              const std::vector<ChatMessage>* messages, const TokenCallback& on_chunk,
                                              const std::optional<SamplingConfig>& sampling_override,
                                              const RequestOptions& request_options,
                                              const ExecutionCheckpointRecord* exact_resume,
                                              std::optional<std::int32_t> max_additional_tokens) {
    SamplingConfig sampling = sampling_override.value_or(options_.sampling);
    if (auto st = validate(sampling); !st.ok()) {
        return st;
    }
    if (request_options.target_ttft_ms.has_value() &&
        (!std::isfinite(*request_options.target_ttft_ms) || *request_options.target_ttft_ms <= 0.0)) {
        return Status(ErrorCode::invalid_argument, "target_ttft_ms must be finite and > 0");
    }
    if (request_options.target_itl_ms.has_value() &&
        (!std::isfinite(*request_options.target_itl_ms) || *request_options.target_itl_ms <= 0.0)) {
        return Status(ErrorCode::invalid_argument, "target_itl_ms must be finite and > 0");
    }
    const RequestPriority effective_priority = [&]() noexcept -> RequestPriority {
        if (request_options.priority_class.has_value()) {
            return *request_options.priority_class;
        }
        const int effective_rank = static_cast<int>(options_.workload) - options_.priority;
        if (effective_rank <= static_cast<int>(WorkloadClass::owner_orchestrator)) {
            return RequestPriority::interactive;
        }
        if (effective_rank <= static_cast<int>(WorkloadClass::research_worker)) {
            return RequestPriority::subagent;
        }
        return RequestPriority::background;
    }();

    // For forked/rewound sessions, prepend the active branched context prefix
    // on ordinary generate() calls (unless caller already passed the full prefix).
    std::string effective_prompt = prompt;
    std::uint64_t existing_branch_seq_id = 0;
    bool using_branch_prefix = false;
    std::size_t branch_prefix_tokens = 0;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "session is closed");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "session already has a request in flight");
        }
        existing_branch_seq_id = logical_kv_seq_id_;
        if (exact_resume == nullptr && messages == nullptr && branched_context_ && !context_prompt_.empty()) {
            using_branch_prefix = true;
            branch_prefix_tokens = context_tokens_.size();
            if (effective_prompt.empty()) {
                effective_prompt = context_prompt_;
            } else if (effective_prompt.rfind(context_prompt_, 0) != 0) {
                std::string combined = context_prompt_;
                if (!detail::is_space(combined.back()) && !detail::is_space(effective_prompt.front())) {
                    combined.push_back(' ');
                }
                combined += effective_prompt;
                effective_prompt = std::move(combined);
            }
        }
    }

    GenerateRequest request;
    request.request_id = request_options.request_id && !request_options.request_id->empty()
                             ? *request_options.request_id
                             : make_id("req");
    request.prompt = effective_prompt;
    request.sampling = sampling;
    request.session_key = request_options.session_key;
    request.grammar = request_options.grammar;
    request.priority_class = effective_priority;
    request.target_ttft_ms = request_options.target_ttft_ms;
    request.target_itl_ms = request_options.target_itl_ms;
    request.enable_stage_profiling = request_options.enable_stage_profiling;
    request.local_only = request_options.local_only;
    request.priority_rank = static_cast<int>(options_.workload) - options_.priority;
    request.allow_preempt_spill =
        request_options.allow_preempt_spill || engine_.scheduling_options().enable_spill_preemption;

    // Prepare failure recovery before publishing running. No cleanup action
    // can replace an exception from a public C++ callback.
    TelemetryBus& bus = engine_.telemetry();
    const auto ctx = telemetry_context(request.request_id);
    detail::RequestRuntime* runtime = engine_.request_runtime();
    std::optional<std::uint64_t> sched_id;
    bool finish_attempted = false;
    bool terminal_attempted = false;
    GenerationResult result;
    std::uint64_t delivered = 0;
    auto t_start = std::chrono::steady_clock::now();
    const auto recover_request = [&]() noexcept {
        if (sched_id && !finish_attempted) {
            finish_attempted = true;  // finish can itself throw: never retry it.
            try {
                const detail::RuntimeRequestSummary sched = runtime->finish(*sched_id, RequestOutcome::failed);
                result.scheduling.preemptions = sched.preemptions;
                result.scheduling.queue_ms = request_options.admission_queue_ms + sched.queue_ms;
                result.scheduling.reused_prompt_tokens = sched.reused_prompt_tokens;
            } catch (...) {
                // Preserve the original exception. Partial runtime cleanup is
                // not an allocator/backend exception-safety guarantee.
            }
        }
        try {
            if (!terminal_attempted) {
                terminal_attempted = true;
                if (request_options.admission) result.scheduling.queue_ms = request_options.admission->queue_ms();
                json::Object failed{{"outcome", "failed"},
                                    {"stop_reason", to_string(result.stats.stop_reason)},
                                    {"prompt_tokens", result.stats.prompt_tokens},
                                    {"completion_tokens", result.stats.completion_tokens},
                                    {"chunks", delivered},
                                    {"token_counts_from_backend", result.stats.token_counts_from_backend},
                                    {"ttft_ms", result.ttft_ms},
                                    {"total_ms", ms_between(t_start, std::chrono::steady_clock::now())},
                                    {"scheduled", result.scheduling.scheduled},
                                    {"sampler", result.scheduling.sonder_sampled ? "sonder" : "backend"},
                                    {"error_code", to_string(ErrorCode::internal)},
                                    {"error", "request aborted by exception"}};
                if (sched_id) {
                    failed.set("queue_ms", result.scheduling.queue_ms);
                    failed.set("preemptions", result.scheduling.preemptions);
                    failed.set("accounted_prompt_tokens", result.scheduling.accounted_prompt_tokens);
                    failed.set("reused_prompt_tokens", result.scheduling.reused_prompt_tokens);
                }
                if (request_options.parent_request_id) {
                    failed.set("parent_request_id", *request_options.parent_request_id);
                }
                if (request_options.priority_class) {
                    failed.set("priority_class", to_string(*request_options.priority_class));
                    failed.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                         {"queue_ms", result.scheduling.queue_ms}});
                }
                bus.emit("request.failed", ctx, std::move(failed), TelemetryLevel::metrics);
            }
        } catch (...) {
            // Telemetry is best effort and never contains exception text.
        }
        try {
            std::lock_guard<std::mutex> lock(mutex_);
            active_cancel_.reset();
            cancel_requested_ns_.store(0);
            suspend_requested_.store(0, std::memory_order_release);
            last_outcome_ = RequestOutcome::failed;
            last_scheduler_rejected_ = false;
            if (state_ == SessionState::running) state_ = SessionState::idle;
        } catch (...) {
            // A mutex failure must not mask the original exception either.
            // Recovery from synchronization failure is outside this contract.
        }
    };
    struct RequestFailureGuard {
        const decltype(recover_request)& recover;
        bool armed = false;
        ~RequestFailureGuard() noexcept { if (armed) recover(); }
    } failure_guard{recover_request};

    CancellationToken token;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (state_ == SessionState::closed) {
            return Status(ErrorCode::invalid_state, "session is closed");
        }
        if (state_ == SessionState::running) {
            return Status(ErrorCode::invalid_state, "session already has a request in flight");
        }
        active_cancel_.emplace();  // Allocation may throw while still idle.
        token = active_cancel_->token();
        state_ = SessionState::running;
        cancel_requested_ns_.store(0);
        suspend_requested_.store(0, std::memory_order_release);
        ++requests_started_;
        failure_guard.armed = true;
    }

    const bool emit_tokens = bus.enabled(TelemetryLevel::standard);
    const bool capture_text = bus.options().capture_text;
    const bool stage_profiling = request_options.enable_stage_profiling || bus.stage_profiling_enabled();
    request.enable_stage_profiling = stage_profiling;
    StageTimings session_stages;
    session_stages.enabled = stage_profiling;
    BackendModel& backend_model = model_->backend_model();
    const auto backend = engine_.find_backend(model_->backend_name());
    const bool has_media = (messages != nullptr) && has_chat_media(*messages);
    // Chat with a backend-native template bypasses the generic prompt (and
    // therefore Sonder's token-level sampler, which works on that prompt).
    const bool native_chat = messages != nullptr && backend_model.has_native_chat();
    // Backends that advertise tool_calling or multimodal receive the structured
    // ChatRequest via chat() when tools or media are present, even if their
    // plain text chat path uses the generic template (e.g. MockBackend).
    const bool backend_chat_call =
        messages != nullptr &&
        (native_chat ||
         (backend && ((!request_options.tools.empty() && backend->capabilities().has(Capability::tool_calling)) ||
                      (has_media && backend->capabilities().has(Capability::multimodal)))));
    const auto& parent = request_options.parent_request_id;
    const auto with_parent = [&parent](json::Object attrs) {
        if (parent) {
            attrs.set("parent_request_id", *parent);
        }
        return attrs;
    };
    const auto with_admission = [&](json::Object attrs, double queue_ms = 0.0) {
        if (request_options.priority_class) {
            attrs.set("priority_class", to_string(*request_options.priority_class));
            attrs.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                   {"queue_ms", queue_ms}});
        }
        return attrs;
    };

    {
        json::Object queued{{"kind", kind},
                            {"priority", options_.priority},
                            {"workload", to_string(options_.workload)},
                            {"prompt_bytes", effective_prompt.size()}};
        if (messages != nullptr) {
            queued.set("messages", messages->size());
        }
        bus.emit("request.queued", ctx,
                 with_parent(with_admission(std::move(queued), request_options.admission_queue_ms)),
                 TelemetryLevel::metrics);
    }
    t_start = std::chrono::steady_clock::now();
    result.request_id = request.request_id;
    result.scheduling.queue_ms = request_options.admission_queue_ms;
    Status failure;
    bool pre_failed = false;
    bool scheduler_rejected = false;

    // Sampling path: Sonder's sampler chain when the backend exposes logits.
    std::unique_ptr<TokenStream> stream;
    const ModelArchitecture arch = model_->descriptor().architecture;
    const bool checkpoint_candidate =
        runtime != nullptr && !backend_chat_call &&
        (arch == ModelArchitecture::hybrid || arch == ModelArchitecture::recurrent) &&
        backend && backend->capabilities().has(Capability::recurrent_checkpoint);
#if defined(SONDER_HAS_SAMPLER_CHAIN)
    const bool use_token_logits =
        !backend_chat_call && (request.grammar.empty() || request_options.grammar_hooks.has_value()) && backend &&
        backend->capabilities().has(Capability::token_logits);
    const bool defer_stream_open =
        (engine_.scheduling_options().strict_priority_admission || checkpoint_candidate) && use_token_logits;
    if (!defer_stream_open && use_token_logits) {
        ScopedStageTimer prefill_timer(stage_profiling ? &session_stages.prefill_ms : nullptr);
        auto opened = backend_model.open_token_stream(request);
        prefill_timer.stop();
        if (opened.ok()) {
            stream = std::move(opened).value();
        } else {
            failure = opened.status();
            pre_failed = true;
        }
    }
#else
    const bool defer_stream_open = false;
#endif
    result.scheduling.sonder_sampled = stream != nullptr;

    // Scheduling: the request enters the scheduler queue and holds KV blocks.
    // Per-chunk grants pace an in-process backend's decode; a remote process
    // batches on its own, so by default it is only admitted and accounted.
    // Backends with shared continuous-batching workers or selective_preemption
    // use non-blocking gate separation so pausing ONE request yields cleanly at
    // a backend safe point without stalling peer slots on the shared thread.
    const SchedulerMode mode = engine_.scheduling_options().mode;
    const bool remote = backend && backend->capabilities().has(Capability::remote_process);
    const bool gated = mode == SchedulerMode::gate || (mode == SchedulerMode::automatic && !remote);
    const bool nonblocking_gate =
        request_options.nonblocking_gate ||
        (backend && (backend->capabilities().has(Capability::selective_preemption) ||
                     backend->capabilities().has(Capability::continuous_batch_decode)));
    bool use_checkpoints = false;
    std::uint64_t aligned_capture_pos = 0;
    std::vector<TokenId> tracked_prompt_tokens;
    bool tracked_exact_tokens = false;
    if (runtime != nullptr && !pre_failed) {
        detail::RuntimeRequestSpec spec;
        spec.context = ctx;
        spec.workload = options_.workload;
        spec.priority = options_.priority;
        spec.priority_class = request_options.priority_class;
        spec.preserve_numeric_priority = request_options.preserve_numeric_priority;
        spec.admission = request_options.admission;
        if (stream) {
            spec.prompt_tokens = stream->prompt_tokens();
            spec.exact_tokens = true;
        } else if (!native_chat) {
            // Exact only for the prompt the backend actually receives: a native
            // chat template differs from the generic prompt, so native chat is
            // accounted approximately below.
            if (auto tk = backend_model.tokenize(effective_prompt); tk.ok() && !tk.value().empty()) {
                spec.prompt_tokens = std::move(tk).value();
                spec.exact_tokens = true;
            }
        }
        if (spec.prompt_tokens.empty()) {
            spec.prompt_tokens = detail::accounting_tokenize(effective_prompt);
            spec.exact_tokens = false;
        }
        tracked_prompt_tokens = spec.prompt_tokens;
        tracked_exact_tokens = spec.exact_tokens;
        spec.max_new_tokens = static_cast<std::uint32_t>(sampling.max_tokens);
        // A requested num_ctx narrows the model's context window (0 = model default).
        std::uint64_t ctx_limit = model_->descriptor().context_length;
        if (sampling.num_ctx > 0) {
            const auto requested = static_cast<std::uint64_t>(sampling.num_ctx);
            ctx_limit = ctx_limit == 0 ? requested : std::min(ctx_limit, requested);
        }
        spec.context_limit = static_cast<std::uint32_t>(std::min<std::uint64_t>(ctx_limit, 0xFFFFFFFFull));
        const std::uint64_t block_size =
            std::max<std::uint32_t>(1, engine_.scheduling_options().kv_block_size_tokens);
        use_checkpoints = checkpoint_candidate && spec.exact_tokens;
        aligned_capture_pos =
            use_checkpoints ? (static_cast<std::uint64_t>(spec.prompt_tokens.size()) / block_size) * block_size : 0;
        spec.fingerprint = model_fingerprint(*model_);
        spec.architecture = arch;
        spec.gated = gated;
        spec.retain_for_checkpoint = use_checkpoints && aligned_capture_pos > 0;
        result.scheduling.accounted_prompt_tokens = spec.prompt_tokens.size();
        result.scheduling.exact_prompt_tokens = spec.exact_tokens;
        const bool exact = spec.exact_tokens;
        auto submitted = runtime->submit(std::move(spec));
        if (submitted.ok() && submitted.value().bypassed) {
            // Ungated prompt beyond the logical KV pool: runs unscheduled.
            result.scheduling.accounted_prompt_tokens = 0;
            result.scheduling.exact_prompt_tokens = false;
        } else if (submitted.ok()) {
            sched_id = submitted.value().id;
            result.scheduling.scheduled = true;
            // The runtime clamps the budget to the context window (num_ctx)
            // and the KV pool. Generation honours it: past it the scheduler
            // stops gating, so nothing else would stop the request short of
            // max_tokens. Only with exact prompt tokens; an approximate count
            // must not truncate a reply the backend itself would allow.
            const auto budget = static_cast<std::int32_t>(
                std::min<std::uint32_t>(submitted.value().max_new_tokens, 0x7FFFFFFFu));
            if (exact && budget > 0 && budget < sampling.max_tokens) {
                sampling.max_tokens = budget;
                request.sampling.max_tokens = budget;
            }
        } else {
            failure = submitted.status();
            pre_failed = true;
            scheduler_rejected = true;
        }
    } else if (!pre_failed) {
        if (stream) {
            tracked_prompt_tokens = stream->prompt_tokens();
            tracked_exact_tokens = true;
        } else if (!native_chat) {
            if (auto tk = backend_model.tokenize(effective_prompt); tk.ok() && !tk.value().empty()) {
                tracked_prompt_tokens = std::move(tk).value();
                tracked_exact_tokens = true;
            }
        }
        if (tracked_prompt_tokens.empty()) {
            tracked_prompt_tokens = detail::accounting_tokenize(effective_prompt);
            tracked_exact_tokens = false;
        }
    }

    {
        json::Object started{{"kind", kind},
                             {"sampling", sampling_json(sampling)},
                             {"scheduled", sched_id.has_value()},
                             {"sampler", (stream || defer_stream_open) ? "sonder" : "backend"}};
        if (sched_id) {
            started.set("scheduler_mode", gated ? "gate" : "account");
            if (nonblocking_gate) {
                started.set("nonblocking_gate", true);
            }
        }
        if (messages != nullptr) {
            started.set("chat_template", native_chat ? "native" : "generic");
        }
        bus.emit("request.started", ctx,
                 with_parent(with_admission(std::move(started), request_options.admission_queue_ms)),
                 TelemetryLevel::metrics);
    }

    std::chrono::steady_clock::time_point t_first{};
    bool have_first = false;
    std::size_t delivered_bytes = exact_resume != nullptr ? static_cast<std::size_t>(exact_resume->delivered_offset_bytes) : 0;
    const std::uint64_t resume_chunk_base = exact_resume != nullptr ? exact_resume->delivered_chunks : 0;
    std::optional<Status> output_limit_failure;

    // Telemetry + user callback for one visible chunk (not gated).
    const auto deliver = [&](const TokenChunk& chunk, std::optional<TokenId> token_id, float probability) -> bool {
        std::size_t chunk_bytes = chunk.text.size() + chunk.reasoning.size();
        for (const auto& tcd : chunk.tool_call_deltas) {
            chunk_bytes += tcd.id.size() + tcd.name.size() + tcd.arguments_delta.size();
        }
        if (request_options.max_response_bytes > 0 &&
            (chunk_bytes > request_options.max_response_bytes ||
             delivered_bytes > request_options.max_response_bytes - chunk_bytes)) {
            output_limit_failure = Status(ErrorCode::protocol_error, "response output exceeds cumulative byte limit");
            return false;
        }
        delivered_bytes += chunk_bytes;
        const auto now = std::chrono::steady_clock::now();
        if (!have_first) {
            have_first = true;
            t_first = now;
            result.ttft_ms = ms_between(t_start, now);
            bus.emit("inference.decode.started", ctx, json::Object{{"ttft_ms", result.ttft_ms}},
                     TelemetryLevel::metrics);
        }
        if (request_options.retain_text) {
            if (!chunk.text.empty()) {
                result.text.append(chunk.text.data(), chunk.text.size());
            }
            if (!chunk.reasoning.empty()) {
                result.reasoning.append(chunk.reasoning.data(), chunk.reasoning.size());
            }
            if (chunk.logprob.has_value()) {
                result.logprobs.push_back(
                    TokenLogprob{std::string(chunk.text), *chunk.logprob, chunk.top_logprobs});
            }
            for (const auto& tcd : chunk.tool_call_deltas) {
                if (tcd.index < 64u) {
                    if (tcd.index >= result.tool_calls.size()) {
                        result.tool_calls.resize(static_cast<std::size_t>(tcd.index) + 1);
                    }
                    ToolCall& dst = result.tool_calls[tcd.index];
                    if (!tcd.id.empty()) dst.id = tcd.id;
                    if (!tcd.name.empty()) dst.name = tcd.name;
                    dst.arguments_json += tcd.arguments_delta;
                }
            }
        }
        ++delivered;
        if (emit_tokens) {
            json::Object attrs{{"index", chunk.index},
                               {"bytes", chunk.text.size()},
                               {"elapsed_ms", ms_between(t_start, now)}};
            if (token_id) {
                // Sonder-sampled: exactly one token per event.
                attrs.set("unit", "token");
                attrs.set("count", 1);
                attrs.set("token_id", *token_id);
                attrs.set("probability", probability);
            } else {
                // Backend-streamed chunk: token count per chunk is unknown.
                attrs.set("unit", "chunk");
            }
            if (capture_text) {
                attrs.set("text", std::string(chunk.text));
            }
            bus.emit("inference.token.generated", ctx, std::move(attrs), TelemetryLevel::standard);
        }
        if (on_chunk) {
            return on_chunk(chunk);
        }
        return true;
    };

    // Scheduler gate: every generated token waits for a grant (gated), or
    // the request waits once for admission (ungated, before the backend call).
    // When nonblocking_gate is active, admission waits on the caller thread
    // while in-flight per-token callbacks never block the shared worker thread.
    std::optional<Status> gate_failure;
    const auto prepare_backend = [&]() -> Status {
        if (sched_id &&
            (!gated || nonblocking_gate || engine_.scheduling_options().strict_priority_admission || use_checkpoints)) {
            Status st = gated ? runtime->admit_backend(*sched_id, token, request_options.deadline)
                              : runtime->acquire_token(*sched_id, token, request_options.deadline);
            if (!st.ok()) return st;
        } else if (!sched_id && request_options.admission) {
            Status st = request_options.admission->wait(token);
            if (!st.ok()) return st;
        }
        if (request_options.deadline && std::chrono::steady_clock::now() >= *request_options.deadline)
            return Status(ErrorCode::timeout, "request deadline exceeded before backend start");
        if (token.cancelled()) return Status(ErrorCode::cancelled, "cancelled before backend start");
        if (request_options.on_backend_start) request_options.on_backend_start();
        // The host can notice a disconnect at the exact start boundary and
        // cancel here, even if its earlier cancel raced session initialization.
        if (token.cancelled()) return Status(ErrorCode::cancelled, "cancelled before backend start");
        return Status::success();
    };
    const auto gate = [&]() -> bool {
        if (!sched_id || !gated) {
            return true;
        }
        if (nonblocking_gate) {
            const detail::GatePollResult gp = runtime->try_acquire_token_nonblocking(*sched_id, token);
            if (gp.status == detail::GatePollStatus::granted) {
                return true;
            }
            if (gp.status == detail::GatePollStatus::yield_recompute ||
                gp.status == detail::GatePollStatus::yield_spill) {
                // Never block on shared worker; safe-point callback handles selective suspension.
                return true;
            }
            if (gp.status != detail::GatePollStatus::cancelled && !gp.error.ok()) {
                gate_failure = gp.error;
            }
            return false;
        }
        ScopedStageTimer gate_timer(stage_profiling ? &session_stages.callback_gate_wait_ms : nullptr);
        Status st = runtime->acquire_token(*sched_id, token);
        gate_timer.stop();
        if (stage_profiling) {
            session_stages.note_overlap("callback_gate_wait_in_decode");
        }
        if (st.ok()) {
            return true;
        }
        if (st.code() != ErrorCode::cancelled) {
            gate_failure = std::move(st);
        }
        return false;
    };
    const auto produced = [&](TokenId t) {
        if (sched_id && gated) {
            runtime->token_produced(*sched_id, t);
        }
    };

    const char* suspend_reason_seen = nullptr;
    const auto poll_safe_point = [&](const SafePointContext& sp) -> SuspendDecision {
        const auto req_act = static_cast<SuspendAction>(suspend_requested_.load(std::memory_order_acquire));
        if (req_act != SuspendAction::continue_running) {
            suspend_reason_seen = "caller_suspend";
            return SuspendDecision{req_act, "caller_suspend", kMaxRecurrentCheckpointBytes};
        }
        if (request_options.on_safe_point) {
            const SuspendDecision user_dec = request_options.on_safe_point(sp);
            if (user_dec.action != SuspendAction::continue_running) {
                suspend_reason_seen = user_dec.reason ? user_dec.reason : "safe_point_callback";
                return user_dec;
            }
        }
        if (sched_id && gated && nonblocking_gate) {
            const SuspendDecision sched_dec = runtime->poll_suspend_decision(*sched_id, token);
            if (sched_dec.action != SuspendAction::continue_running) {
                suspend_reason_seen = sched_dec.reason ? sched_dec.reason : "scheduler_preempt";
                return sched_dec;
            }
        }
        return SuspendDecision{};
    };

    ExecutionCheckpointRecord captured_continuation;
    bool has_captured_continuation = false;
    SuspendedExecutionState suspended_state;
    std::vector<TokenId> generated_tokens_accum;
    std::vector<std::string> generated_pieces_accum;
    std::vector<std::uint8_t> captured_recurrent_blob;

    Result<GenerateStats> generated = GenerateStats{};
    Status prepared = pre_failed ? failure : prepare_backend();
    if (!prepared.ok()) {
        generated = prepared;
    } else {
        std::optional<detail::RestoredCheckpoint> restored_checkpoint;
        RecurrentCheckpointExchange checkpoint_io;
        bool checkpoint_attached = false;
        bool restored_event_emitted = false;
        bool capture_registered_or_attempted = false;

        if (use_checkpoints && sched_id) {
            restored_checkpoint = runtime->lookup_restored_checkpoint(*sched_id);
            std::uint64_t restored_pos = 0;
            if (restored_checkpoint && restored_checkpoint->bytes && !restored_checkpoint->bytes->empty()) {
                restored_pos = restored_checkpoint->handle.token_position;
                checkpoint_io.restore_bytes = *restored_checkpoint->bytes;
                checkpoint_io.restore_token_position = restored_pos;
                checkpoint_io.on_restored = [&] {
                    if (restored_event_emitted) return;
                    restored_event_emitted = true;
                    bus.emit("kv.checkpoint.restored", ctx,
                             json::Object{{"scheduler_request_id", *sched_id},
                                          {"owner_scheduler_request_id", restored_checkpoint->handle.owner_request_id},
                                          {"token_position", restored_checkpoint->handle.token_position},
                                          {"bytes", restored_checkpoint->bytes->size()},
                                          {"architecture", std::string(to_string(arch))}},
                             TelemetryLevel::metrics);
                };
                checkpoint_attached = true;
            }

            const bool want_capture = aligned_capture_pos > 0 && aligned_capture_pos > restored_pos;
            if (want_capture) {
                checkpoint_io.capture_at_token_position = aligned_capture_pos;
                checkpoint_io.on_captured = [&](std::uint64_t pos, std::vector<std::uint8_t> bytes) {
                    capture_registered_or_attempted = true;
                    const std::size_t snapshot_bytes = bytes.size();
                    if (snapshot_bytes == 0 || snapshot_bytes > kMaxRecurrentCheckpointBytes) {
                        bus.emit("kv.checkpoint.save_failed", ctx,
                                 json::Object{{"scheduler_request_id", *sched_id},
                                              {"token_position", pos},
                                              {"stage", "validate"},
                                              {"architecture", std::string(to_string(arch))},
                                              {"error_code", to_string(ErrorCode::invalid_argument)}},
                                 TelemetryLevel::deep);
                    } else {
                        captured_recurrent_blob = bytes;
                        const Status reg = runtime->register_checkpoint(*sched_id, pos, std::move(bytes));
                        if (reg.ok()) {
                            bus.emit("kv.checkpoint.saved", ctx,
                                     json::Object{{"scheduler_request_id", *sched_id},
                                                  {"token_position", pos},
                                                  {"bytes", snapshot_bytes},
                                                  {"architecture", std::string(to_string(arch))}},
                                     TelemetryLevel::metrics);
                        } else {
                            bus.emit("kv.checkpoint.save_failed", ctx,
                                     json::Object{{"scheduler_request_id", *sched_id},
                                                  {"token_position", pos},
                                                  {"stage", "register"},
                                                  {"architecture", std::string(to_string(arch))},
                                                  {"error_code", to_string(reg.code())}},
                                     TelemetryLevel::deep);
                        }
                    }
                    if (!gated) {
                        runtime->release_checkpoint_retention(*sched_id);
                    }
                };
                checkpoint_attached = true;
            } else if (!gated && aligned_capture_pos > 0) {
                // No new capture needed (reused checkpoint already at aligned boundary):
                // release retention immediately so blocks are not pinned during decode.
                runtime->release_checkpoint_retention(*sched_id);
            }

            if (checkpoint_attached) {
                request.checkpoint = &checkpoint_io;
            }
        }
#if defined(SONDER_HAS_SAMPLER_CHAIN)
        if (defer_stream_open) {
            ScopedStageTimer prefill_timer(stage_profiling ? &session_stages.prefill_ms : nullptr);
            auto opened = backend_model.open_token_stream(request);
            prefill_timer.stop();
            request.checkpoint = nullptr;
            if (checkpoint_attached && sched_id && !gated && !capture_registered_or_attempted) {
                runtime->release_checkpoint_retention(*sched_id);
            }
            if (opened.ok()) {
                stream = std::move(opened).value();
                result.scheduling.sonder_sampled = true;
            } else {
                prepared = opened.status();
            }
        }
#endif
        if (!prepared.ok()) {
            generated = prepared;
        } else if (stream) {
#if defined(SONDER_HAS_SAMPLER_CHAIN)
            generated = run_sonder_sampling(*stream, backend_model, sampling, token, bus, ctx, gate, produced, deliver,
                                            poll_safe_point, request_options, exact_resume, max_additional_tokens,
                                            &captured_continuation, &suspended_state, generated_tokens_accum,
                                            generated_pieces_accum, stage_profiling ? &session_stages : nullptr);
            has_captured_continuation = generated.ok();
#endif
        } else {
            ExecutionResumeState exec_resume;
            if (exact_resume != nullptr) {
                exec_resume.token_position = exact_resume->handle.token_position;
                exec_resume.generated_tokens = exact_resume->delivered_completion_tokens;
                exec_resume.backend_step_index = exact_resume->backend_step_index;
                exec_resume.generated_token_ids = exact_resume->generated_tokens;
                exec_resume.pending_draft_tokens = exact_resume->pending_draft_tokens;
                exec_resume.backend_state_blob = exact_resume->backend_state_blob;
                exec_resume.stop_filter_pending = exact_resume->stop_filter_pending;
                exec_resume.stop_filter_stopped = exact_resume->stop_filter_stopped;
                request.execution_resume = &exec_resume;
                if (max_additional_tokens.has_value()) {
                    request.sampling.max_tokens = std::max<std::int32_t>(0, *max_additional_tokens);
                } else {
                    const std::uint64_t cap = static_cast<std::uint64_t>(std::max<std::int32_t>(0, sampling.max_tokens));
                    request.sampling.max_tokens =
                        cap > exact_resume->delivered_completion_tokens
                            ? static_cast<std::int32_t>(cap - exact_resume->delivered_completion_tokens)
                            : 0;
                }
            }
            request.on_safe_point = poll_safe_point;
            request.suspended_state = &suspended_state;

            const auto on_backend_chunk = [&](const TokenChunk& chunk) -> bool {
                TokenChunk adjusted = chunk;
                adjusted.index = resume_chunk_base + chunk.index;
                const TokenId tok = detail::accounting_chunk_token(chunk.text);
                generated_tokens_accum.push_back(tok);
                generated_pieces_accum.emplace_back(chunk.text);
                if (!gated || !sched_id) return deliver(adjusted, std::nullopt, 0.0f);
                if (nonblocking_gate) {
                    const detail::GatePollResult gp = runtime->try_acquire_token_nonblocking(*sched_id, token);
                    if (gp.status == detail::GatePollStatus::cancelled) {
                        return false;
                    }
                    if (gp.status == detail::GatePollStatus::failed) {
                        gate_failure = gp.error;
                        return false;
                    }
                    produced(tok);
                    return deliver(adjusted, std::nullopt, 0.0f);
                }
                if (!gate()) return false;
                produced(tok);
                return deliver(adjusted, std::nullopt, 0.0f);
            };
            if (backend_chat_call) {
                ChatRequest chat_request;
                chat_request.request_id = request.request_id;
                chat_request.messages = *messages;
                chat_request.sampling = sampling;
                chat_request.session_key = request_options.session_key;
                chat_request.thinking = request_options.thinking;
                chat_request.separate_reasoning = request_options.separate_reasoning;
                chat_request.reasoning_budget_tokens = request_options.reasoning_budget_tokens;
                chat_request.reasoning_budget_message = request_options.reasoning_budget_message;
                chat_request.grammar = request_options.grammar;
                chat_request.response_format_type = request_options.response_format_type;
                chat_request.response_format_schema = request_options.response_format_schema;
                chat_request.tools = request_options.tools;
                chat_request.tool_choice = request_options.tool_choice;
                chat_request.parallel_tool_calls = request_options.parallel_tool_calls;
                chat_request.priority_class = effective_priority;
                chat_request.target_ttft_ms = request_options.target_ttft_ms;
                chat_request.target_itl_ms = request_options.target_itl_ms;
                chat_request.enable_stage_profiling = stage_profiling;
                chat_request.local_only = request.local_only;
                generated = backend_model.chat(chat_request, token, on_backend_chunk);
            } else {
                generated = backend_model.generate(request, token, on_backend_chunk);
            }
            request.execution_resume = nullptr;
            request.suspended_state = nullptr;
            if (generated.ok()) {
                const std::uint64_t prior_gen = exact_resume != nullptr ? exact_resume->delivered_completion_tokens : 0;
                const std::uint64_t total_gen = prior_gen + generated.value().completion_tokens;
                captured_continuation.prompt_tokens = tracked_prompt_tokens;
                if (generated.value().stop_reason == StopReason::suspended && !suspended_state.tokens.empty() &&
                    suspended_state.tokens.size() >= tracked_prompt_tokens.size()) {
                    captured_continuation.generated_tokens.assign(
                        suspended_state.tokens.begin() + static_cast<std::ptrdiff_t>(tracked_prompt_tokens.size()),
                        suspended_state.tokens.end());
                } else {
                    if (exact_resume != nullptr) {
                        captured_continuation.generated_tokens = exact_resume->generated_tokens;
                    }
                    captured_continuation.generated_tokens.insert(captured_continuation.generated_tokens.end(),
                                                                  generated_tokens_accum.begin(),
                                                                  generated_tokens_accum.end());
                }
                captured_continuation.delivered_chunks = resume_chunk_base + delivered;
                captured_continuation.delivered_completion_tokens = total_gen;
                captured_continuation.sampling = sampling;
                captured_continuation.grammar = request_options.grammar;
                captured_continuation.pending_draft_tokens = suspended_state.pending_draft_tokens;
                captured_continuation.backend_state_blob = suspended_state.backend_state_blob;
                captured_continuation.backend_step_index =
                    suspended_state.backend_step_index > 0 ? suspended_state.backend_step_index : total_gen;
                const std::uint64_t tok_pos = suspended_state.token_position > 0
                                                  ? suspended_state.token_position
                                                  : (static_cast<std::uint64_t>(tracked_prompt_tokens.size()) + total_gen);
                captured_continuation.backend_state_token_position = tok_pos;
                captured_continuation.handle.token_position = tok_pos;
                captured_continuation.handle.delivered_chunks = captured_continuation.delivered_chunks;
                captured_continuation.handle.delivered_completion_tokens = total_gen;
                if (!suspended_state.stop_filter_pending.empty() || suspended_state.stop_filter_stopped) {
                    captured_continuation.stop_filter_pending = suspended_state.stop_filter_pending;
                    captured_continuation.stop_filter_stopped = suspended_state.stop_filter_stopped;
                } else if (!sampling.stop.empty()) {
                    captured_continuation.stop_filter_pending =
                        (exact_resume != nullptr ? exact_resume->stop_filter_pending : std::string()) + result.text;
                }
                has_captured_continuation = true;
            }
        }
        request.checkpoint = nullptr;
        if (checkpoint_attached && sched_id) {
            if (checkpoint_io.restored && !restored_event_emitted && restored_checkpoint &&
                restored_checkpoint->bytes) {
                restored_event_emitted = true;
                bus.emit("kv.checkpoint.restored", ctx,
                         json::Object{{"scheduler_request_id", *sched_id},
                                      {"owner_scheduler_request_id", restored_checkpoint->handle.owner_request_id},
                                      {"token_position", restored_checkpoint->handle.token_position},
                                      {"bytes", restored_checkpoint->bytes->size()},
                                      {"architecture", std::string(to_string(arch))}},
                         TelemetryLevel::metrics);
            } else if (checkpoint_io.restore_error && restored_checkpoint) {
                bus.emit("kv.checkpoint.restore_failed", ctx,
                         json::Object{{"scheduler_request_id", *sched_id},
                                      {"owner_scheduler_request_id", restored_checkpoint->handle.owner_request_id},
                                      {"token_position", restored_checkpoint->handle.token_position},
                                      {"architecture", std::string(to_string(arch))},
                                      {"error_code", to_string(checkpoint_io.restore_error->code())}},
                         TelemetryLevel::deep);
            }
            if (checkpoint_io.capture_error) {
                bus.emit("kv.checkpoint.save_failed", ctx,
                         json::Object{{"scheduler_request_id", *sched_id},
                                      {"token_position", checkpoint_io.capture_at_token_position},
                                      {"stage", "capture"},
                                      {"architecture", std::string(to_string(arch))},
                                      {"error_code", to_string(checkpoint_io.capture_error->code())}},
                         TelemetryLevel::deep);
            }
            if (!gated && !capture_registered_or_attempted) {
                runtime->release_checkpoint_retention(*sched_id);
            }
        }
    }
    const auto t_end = std::chrono::steady_clock::now();
    result.total_ms = ms_between(t_start, t_end);
    result.delivered_offset_bytes = static_cast<std::uint64_t>(delivered_bytes);
    result.delivered_chunks = resume_chunk_base + delivered;

    const bool was_cancelled =
        token.cancelled() || (!generated.ok() && generated.status().code() == ErrorCode::cancelled);

    RequestOutcome outcome = RequestOutcome::completed;
    if (output_limit_failure && !token.cancelled()) {
        outcome = RequestOutcome::failed;
        failure = *output_limit_failure;
        if (generated.ok()) {
            result.stats = generated.value();
        }
    } else if (gate_failure && !token.cancelled()) {
        outcome = RequestOutcome::failed;
        failure = *gate_failure;
        if (generated.ok()) {
            result.stats = generated.value();
        }
    } else if (was_cancelled) {
        outcome = RequestOutcome::cancelled;
        if (generated.ok()) {
            result.stats = generated.value();
        }
        result.stats.stop_reason = StopReason::cancelled;
        if (result.stats.chunks == 0) {
            result.stats.chunks = delivered;
        }
        if (!result.stats.token_counts_from_backend && result.stats.completion_tokens == 0) {
            result.stats.completion_tokens = delivered;
        }
    } else if (!generated.ok()) {
        outcome = RequestOutcome::failed;
        failure = generated.status();
    } else {
        result.stats = generated.value();
    }
    result.outcome = outcome;

    const bool was_suspended = (outcome == RequestOutcome::completed && result.stats.stop_reason == StopReason::suspended);
    if (was_suspended) {
        result.scheduling.suspended = true;
        result.scheduling.physical_slot_reclaimed = suspended_state.physical_slot_reclaimed;
        result.scheduling.suspend_action = suspended_state.action;
    }

    if (sched_id) {
        finish_attempted = true;
        const detail::RuntimeRequestSummary sched = runtime->finish(*sched_id, outcome);
        result.scheduling.preemptions = sched.preemptions;
        result.scheduling.queue_ms = request_options.admission_queue_ms + sched.queue_ms;
        result.scheduling.reused_prompt_tokens = sched.reused_prompt_tokens;
    }

    if (request_options.admission) result.scheduling.queue_ms = request_options.admission->queue_ms();

    {
        EffectiveExecutionReport eff = result.stats.effective_execution.value_or(EffectiveExecutionReport{});
        if (eff.target_identity.empty()) {
            eff.target_identity = model_->descriptor().name;
        }
        if (!eff.committed_tokens.has_value() &&
            (result.stats.token_counts_from_backend || result.stats.completion_tokens > 0)) {
            eff.committed_tokens = result.stats.completion_tokens;
        }
        if (!eff.proposed_draft_tokens.has_value() && result.stats.draft_tokens.has_value()) {
            eff.proposed_draft_tokens = result.stats.draft_tokens;
        }
        if (!eff.accepted_draft_tokens.has_value() && result.stats.draft_accepted_tokens.has_value()) {
            eff.accepted_draft_tokens = result.stats.draft_accepted_tokens;
        }
        // Strict 3-tier cache separation: engine KvCacheManager accounting
        // stays exclusively in logical_cache_hit_tokens and is never copied
        // into physical device_prefix_reused_tokens or host_l2_restored_tokens.
        if (result.scheduling.scheduled) {
            eff.cache.logical_cache_hit_tokens = result.scheduling.reused_prompt_tokens;
        }
        if (eff.speculation_observed_active == CapabilityState::unknown) {
            if (eff.proposed_draft_tokens.has_value() && *eff.proposed_draft_tokens > 0u) {
                eff.speculation_observed_active = CapabilityState::yes;
            } else if (backend && !backend->capabilities().has(Capability::speculative_decode)) {
                eff.speculation_observed_active = CapabilityState::no;
                if (eff.configured_spec_method.empty()) {
                    eff.configured_spec_method = "none";
                }
                if (eff.effective_spec_method.empty()) {
                    eff.effective_spec_method = "none";
                }
                if (eff.fallback_reason.empty()) {
                    eff.fallback_reason = std::string(kSpecReasonNotSupported);
                }
            }
        }
        result.effective_execution = bound_effective_execution_report(std::move(eff));
    }

    if (stage_profiling) {
        StageTimings merged = result.stats.stage_timings.value_or(StageTimings{});
        merged.enabled = true;
        if (result.scheduling.scheduled || request_options.admission || request_options.admission_queue_ms > 0.0) {
            merged.queue_wait_ms = std::max(0.0, result.scheduling.queue_ms);
        }
        if (!merged.l2_restore_ms.has_value() && result.effective_execution &&
            result.effective_execution->cache.host_l2_restore_ms.has_value()) {
            merged.l2_restore_ms = result.effective_execution->cache.host_l2_restore_ms;
        }
        if (!merged.prefill_ms.has_value()) {
            if (session_stages.prefill_ms.has_value()) {
                merged.prefill_ms = session_stages.prefill_ms;
            } else if (result.stats.prompt_eval_ns > 0) {
                merged.prefill_ms = static_cast<double>(result.stats.prompt_eval_ns) / 1e6;
            }
        }
        if (session_stages.sampling_ms.has_value()) {
            StageTimings::add_ms(merged.sampling_ms, *session_stages.sampling_ms);
        }
        if (session_stages.callback_gate_wait_ms.has_value()) {
            StageTimings::add_ms(merged.callback_gate_wait_ms, *session_stages.callback_gate_wait_ms);
        }
        for (const auto& note : session_stages.overlap_notes) {
            merged.note_overlap(note);
        }
        if (session_stages.stages_may_overlap) {
            merged.stages_may_overlap = true;
        }
        merged.total_elapsed_ms = std::max(0.0, result.total_ms);
        result.stage_timings = std::move(merged);
    }

    // Update session context tracking, logical KV branch (if already forked),
    // continuation state, and safe cleanup on cancellation.
    if (outcome == RequestOutcome::cancelled) {
        std::lock_guard<std::mutex> lock(mutex_);
        last_continuation_.reset();
    } else if (outcome == RequestOutcome::completed) {
        std::vector<TokenId> full_tokens = tracked_prompt_tokens;
        std::vector<std::string> full_pieces = build_prompt_token_pieces(effective_prompt, tracked_prompt_tokens);
        const bool pieces_aligned = (full_pieces.size() == tracked_prompt_tokens.size());
        if (exact_resume != nullptr && !exact_resume->generated_tokens.empty()) {
            // generated_tokens_accum already includes resumed tokens on token_logits,
            // or only newly generated tokens on non-token_logits.
            if (!result.scheduling.sonder_sampled) {
                full_tokens.insert(full_tokens.end(), exact_resume->generated_tokens.begin(),
                                   exact_resume->generated_tokens.end());
            }
        }
        full_tokens.insert(full_tokens.end(), generated_tokens_accum.begin(), generated_tokens_accum.end());
        if (pieces_aligned && generated_pieces_accum.size() == generated_tokens_accum.size() &&
            full_tokens.size() == tracked_prompt_tokens.size() + generated_pieces_accum.size()) {
            full_pieces.insert(full_pieces.end(), generated_pieces_accum.begin(), generated_pieces_accum.end());
        } else {
            full_pieces.clear();
        }

        const std::string total_delivered_text =
            (exact_resume != nullptr ? exact_resume->delivered_text : std::string()) + result.text;
        std::string full_prompt_text = effective_prompt;
        if (!total_delivered_text.empty()) {
            if (!full_prompt_text.empty() && !detail::is_space(full_prompt_text.back()) &&
                !detail::is_space(total_delivered_text.front())) {
                full_prompt_text.push_back(' ');
            }
            full_prompt_text += total_delivered_text;
        }

        // If this session already owns a forked logical KV sequence, append only
        // the newly added suffix tokens onto it (exercising real COW isolation!).
        std::uint64_t updated_branch_seq = existing_branch_seq_id;
        if (existing_branch_seq_id != 0 && runtime != nullptr && !full_tokens.empty()) {
            const std::size_t existing_len = using_branch_prefix ? branch_prefix_tokens : 0;
            if (using_branch_prefix && full_tokens.size() >= existing_len) {
                auto appended = runtime->append_logical_sequence(
                    existing_branch_seq_id,
                    std::span<const TokenId>(full_tokens.data() + existing_len, full_tokens.size() - existing_len));
                if (!appended.ok()) {
                    // Sequence length diverged; re-materialize cleanly.
                    const std::int32_t rank = static_cast<std::int32_t>(options_.workload) - options_.priority;
                    auto ensured = runtime->ensure_logical_sequence(
                        existing_branch_seq_id, full_tokens, model_fingerprint(*model_), arch,
                        std::clamp(6 - rank, 0, 255), captured_recurrent_blob);
                    if (ensured.ok()) {
                        updated_branch_seq = ensured.value().sequence_id;
                    }
                }
            } else {
                const std::int32_t rank = static_cast<std::int32_t>(options_.workload) - options_.priority;
                auto ensured = runtime->ensure_logical_sequence(
                    existing_branch_seq_id, full_tokens, model_fingerprint(*model_), arch,
                    std::clamp(6 - rank, 0, 255), captured_recurrent_blob);
                if (ensured.ok()) {
                    updated_branch_seq = ensured.value().sequence_id;
                }
            }
        }

        if (has_captured_continuation) {
            captured_continuation.handle.owner_session_id = options_.session_id;
            captured_continuation.handle.mode = CheckpointMode::ExactResume;
            captured_continuation.handle.compatibility = compatibility_fingerprint();
            captured_continuation.handle.same_process_only = true;
            captured_continuation.handle.delivered_offset_bytes = static_cast<std::uint64_t>(delivered_bytes);
            captured_continuation.delivered_offset_bytes = static_cast<std::uint64_t>(delivered_bytes);
            captured_continuation.prompt = effective_prompt;
            captured_continuation.exact_tokens = tracked_exact_tokens;
            captured_continuation.delivered_text =
                (exact_resume != nullptr ? exact_resume->delivered_text : std::string()) + result.text;
            if (!captured_recurrent_blob.empty() && captured_continuation.backend_state_blob.empty()) {
                captured_continuation.backend_state_blob = captured_recurrent_blob;
            }
        }

        if (was_suspended && has_captured_continuation && request_options.auto_checkpoint_on_suspend) {
            ExecutionCheckpointRecord store_rec = captured_continuation;
            store_rec.logical_kv_sequence_id = 0;
            if (updated_branch_seq != 0 && runtime != nullptr) {
                auto forked = runtime->fork_logical_sequence(updated_branch_seq);
                if (forked.ok()) {
                    store_rec.logical_kv_sequence_id = forked.value().sequence_id;
                }
            }
            const std::uint64_t store_kv_seq = store_rec.logical_kv_sequence_id;
            auto stored = engine_.checkpoint_store().put(std::move(store_rec), request_options.suspend_checkpoint_ttl);
            if (stored.ok()) {
                result.suspended_checkpoint = stored.value();
            } else if (store_kv_seq != 0 && runtime != nullptr) {
                runtime->release_logical_sequence(store_kv_seq);
            }
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            context_prompt_ = std::move(full_prompt_text);
            context_tokens_ = std::move(full_tokens);
            context_token_pieces_ = std::move(full_pieces);
            exact_context_tokens_ = tracked_exact_tokens;
            logical_kv_seq_id_ = updated_branch_seq;
            if (!captured_recurrent_blob.empty()) {
                last_recurrent_state_blob_ = std::move(captured_recurrent_blob);
            }
            if (has_captured_continuation) {
                last_continuation_ = std::move(captured_continuation);
            }
        }
    }

    const std::uint64_t cancel_ns = cancel_requested_ns_.load();
    const std::uint64_t now_ns = monotonic_ns();

    json::Object summary{{"outcome", to_string(outcome)},
                         {"stop_reason", to_string(result.stats.stop_reason)},
                         {"prompt_tokens", result.stats.prompt_tokens},
                         {"completion_tokens", result.stats.completion_tokens},
                         {"chunks", delivered},
                         {"token_counts_from_backend", result.stats.token_counts_from_backend},
                         {"ttft_ms", result.ttft_ms},
                         {"total_ms", result.total_ms},
                         {"scheduled", result.scheduling.scheduled},
                         {"sampler", result.scheduling.sonder_sampled ? "sonder" : "backend"}};
    backend_observations(summary, result.stats);
    if (result.effective_execution) {
        summary.set("effective_execution", to_json(*result.effective_execution));
    }
    if (result.stage_timings) {
        summary.set("stage_timings", to_json(*result.stage_timings));
        bus.emit_stage_timings(ctx, *result.stage_timings, model_->backend_name(),
                               result.effective_execution ? std::string_view(result.effective_execution->backend_version)
                                                          : std::string_view{},
                               TelemetryLevel::metrics);
    }
    if (was_suspended) {
        summary.set("suspended", true);
        summary.set("suspend_action", to_string(result.scheduling.suspend_action));
        summary.set("physical_slot_reclaimed", result.scheduling.physical_slot_reclaimed);
        if (suspend_reason_seen != nullptr) {
            summary.set("suspend_reason", suspend_reason_seen);
        }
        if (result.suspended_checkpoint) {
            summary.set("checkpoint_id", result.suspended_checkpoint->checkpoint_id);
        }
    }
    if (request_options.reasoning_budget_tokens) summary.set("reasoning_budget", *request_options.reasoning_budget_tokens);
    if (result.stats.backend_timings) {
        const auto& t = *result.stats.backend_timings;
        if (t.cache_n) summary.set("backend_cache_n", *t.cache_n);
        if (t.prompt_n) summary.set("backend_prompt_n", *t.prompt_n);
        if (t.predicted_n) summary.set("backend_predicted_n", *t.predicted_n);
        if (t.draft_n) summary.set("backend_draft_n", *t.draft_n);
        if (t.draft_n_accepted) summary.set("backend_draft_accepted", *t.draft_n_accepted);
    }
    if (result.scheduling.scheduled) {
        summary.set("queue_ms", result.scheduling.queue_ms);
        summary.set("preemptions", result.scheduling.preemptions);
        summary.set("accounted_prompt_tokens", result.scheduling.accounted_prompt_tokens);
        summary.set("reused_prompt_tokens", result.scheduling.reused_prompt_tokens);
    }
    if (request_options.priority_class) {
        summary.set("priority_class", to_string(*request_options.priority_class));
        summary.set("admission", json::Object{{"priority", to_string(*request_options.priority_class)},
                                                {"queue_ms", result.scheduling.queue_ms}});
    }

    if (outcome == RequestOutcome::completed) {
        if (was_suspended) {
            bus.emit("request.suspended", ctx,
                     json::Object{{"action", to_string(result.scheduling.suspend_action)},
                                  {"reason", suspend_reason_seen ? suspend_reason_seen : "suspended"},
                                  {"physical_slot_reclaimed", result.scheduling.physical_slot_reclaimed},
                                  {"delivered_offset_bytes", result.delivered_offset_bytes},
                                  {"delivered_chunks", result.delivered_chunks},
                                  {"spilled_bytes", suspended_state.backend_state_blob.size()}},
                     TelemetryLevel::metrics);
        }
        json::Object pre{{"prompt_tokens", result.stats.prompt_tokens},
                         {"token_counts_from_backend", result.stats.token_counts_from_backend},
                         {"ttft_ms", result.ttft_ms}};
        if (result.stats.cached_tokens) pre.set("backend_cached_tokens", *result.stats.cached_tokens);
        if (result.stats.prompt_eval_ns > 0) {
            pre.set("backend_prompt_eval_ms", static_cast<double>(result.stats.prompt_eval_ns) / 1e6);
        }
        if (result.scheduling.scheduled) {
            pre.set("reused_prompt_tokens", result.scheduling.reused_prompt_tokens);
        }
        bus.emit("inference.prefill.completed", ctx, std::move(pre), TelemetryLevel::metrics);
        const double decode_ms = have_first ? ms_between(t_first, t_end) : 0.0;
        json::Object dec{{"completion_tokens", result.stats.completion_tokens},
                         {"chunks", delivered},
                         {"decode_wall_ms", decode_ms}};
        backend_observations(dec, result.stats);
        if (result.stats.eval_ns > 0) {
            dec.set("backend_eval_ms", static_cast<double>(result.stats.eval_ns) / 1e6);
            dec.set("backend_tokens_per_sec",
                    static_cast<double>(result.stats.completion_tokens) * 1e9 / static_cast<double>(result.stats.eval_ns));
        }
        if (result.stats.prompt_eval_ns > 0) {
            dec.set("backend_prompt_eval_ms", static_cast<double>(result.stats.prompt_eval_ns) / 1e6);
        }
        bus.emit("inference.decode.completed", ctx, std::move(dec), TelemetryLevel::metrics);
        terminal_attempted = true;
        bus.emit("request.completed", ctx, with_parent(std::move(summary)), TelemetryLevel::metrics);
    } else if (outcome == RequestOutcome::cancelled) {
        if (cancel_ns != 0 && now_ns >= cancel_ns) {
            summary.set("cancel_latency_ms", static_cast<double>(now_ns - cancel_ns) / 1e6);
        }
        terminal_attempted = true;
        bus.emit("request.cancelled", ctx, with_parent(std::move(summary)), TelemetryLevel::metrics);
    } else {
        // Never emit raw upstream error text into shared Observatory telemetry
        // (M-15): only record the safe structured error_code.
        summary.set("error_code", to_string(failure.code()));
        summary.set("error", to_string(failure.code()));
        if (scheduler_rejected) {
            summary.set("scheduler_rejected", true);
        }
        terminal_attempted = true;
        bus.emit("request.failed", ctx, with_parent(std::move(summary)), TelemetryLevel::metrics);
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        active_cancel_.reset();
        cancel_requested_ns_.store(0);
        suspend_requested_.store(0, std::memory_order_release);
        last_outcome_ = outcome;
        last_scheduler_rejected_ = outcome == RequestOutcome::failed && scheduler_rejected;
        if (state_ == SessionState::running) {
            state_ = SessionState::idle;
        }
    }

    failure_guard.armed = false;
    if (outcome == RequestOutcome::failed) {
        return failure;
    }
    return result;
}

}  // namespace sonder::inference
