#include "sonder/inference/scheduler/scheduler.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace sonder::inference::scheduler {

namespace {

constexpr int kStarvingRank = std::numeric_limits<int>::min() / 2;

[[nodiscard]] TimeUs elapsed_us(TimeUs from, TimeUs to) noexcept {
    if (to <= from) return 0;
    constexpr TimeUs kMax = std::numeric_limits<TimeUs>::max();
    if (from < 0 && to > kMax + from) return kMax;
    return to - from;
}

[[nodiscard]] TimeUs saturating_add_time(TimeUs a, TimeUs b) noexcept {
    if (b <= 0) return a;
    constexpr TimeUs kMax = std::numeric_limits<TimeUs>::max();
    return a > kMax - b ? kMax : a + b;
}

[[nodiscard]] BlockCount saturating_add_blocks(BlockCount a, BlockCount b) noexcept {
    constexpr BlockCount kMax = std::numeric_limits<BlockCount>::max();
    return a > kMax - b ? kMax : a + b;
}

[[nodiscard]] bool fits_with_watermark(BlockCount free_blocks, BlockCount need,
                                       BlockCount watermark) noexcept {
    constexpr BlockCount kMax = std::numeric_limits<BlockCount>::max();
    return watermark <= kMax - need && free_blocks >= need + watermark;
}

}  // namespace

Scheduler::Scheduler(SchedulerConfig config, const Clock& clock, KvCapacity& kv)
    : config_(config), clock_(clock), kv_(kv) {
    if (config_.max_step_tokens == 0 || config_.max_step_sequences == 0 ||
        config_.max_running_sequences == 0 || config_.prefill_chunk_tokens == 0 ||
        config_.max_step_prefill_tokens == 0) {
        throw std::invalid_argument("SchedulerConfig budgets must be non-zero");
    }
    if (config_.aging_interval_us <= 0) {
        throw std::invalid_argument("SchedulerConfig::aging_interval_us must be positive");
    }
    if (config_.starvation_threshold_us < 0) {
        throw std::invalid_argument("SchedulerConfig::starvation_threshold_us must be non-negative");
    }
    if (config_.enable_priority_preemption && !config_.strict_priority_admission &&
        config_.priority_preemption_min_rank_gap <= 0) {
        throw std::invalid_argument("SchedulerConfig::priority_preemption_min_rank_gap must be positive");
    }
    if (kv_.block_size_tokens() == 0) {
        throw std::invalid_argument("KvCapacity block size must be non-zero");
    }
}

// ---------------------------------------------------------------- submission

SubmitResult Scheduler::submit(RequestSpec spec) {
    const TimeUs now = clock_.now();
    const auto cls = index_of(spec.workload);
    if (cls >= kWorkloadClassCount) {
        return SubmitResult{false, FailureReason::InvalidRequest};
    }
    counters_.per_class[cls].submitted++;

    auto reject = [&](FailureReason why) {
        counters_.per_class[cls].rejected++;
        return SubmitResult{false, why};
    };

    if (seqs_.count(spec.id) != 0) {
        return reject(FailureReason::DuplicateId);
    }
    if (spec.prompt_tokens == 0 || spec.max_new_tokens == 0 ||
        static_cast<std::uint64_t>(spec.prompt_tokens) + spec.max_new_tokens >
            std::numeric_limits<TokenCount>::max()) {
        return reject(FailureReason::InvalidRequest);
    }
    const BlockCount worst_case =
        blocks_for_tokens(spec.prompt_tokens + spec.max_new_tokens, kv_.block_size_tokens());
    if (worst_case > kv_.total_blocks()) {
        return reject(FailureReason::NeverFits);
    }
    if (!spec.sequence_fingerprint.empty() && !spec.allow_duplicate &&
        active_fingerprints_.count(spec.sequence_fingerprint) != 0) {
        return reject(FailureReason::DuplicateSequence);
    }

    Sequence s;
    s.base_rank = spec.priority_override.value_or(static_cast<int>(spec.workload));
    s.submit_order = next_submit_order_++;
    s.enqueued_at = now;
    s.prefill_target = spec.prompt_tokens;
    s.state = RequestState::WaitingAdmission;
    s.tl.id = spec.id;
    s.tl.workload = spec.workload;
    s.tl.arrival = now;
    s.tl.final_state = RequestState::WaitingAdmission;
    if (!spec.sequence_fingerprint.empty()) {
        active_fingerprints_.insert(spec.sequence_fingerprint);
    }
    const RequestId id = spec.id;
    s.spec = std::move(spec);
    seqs_.emplace(id, std::move(s));
    waiting_.push_back(id);
    return SubmitResult{true, FailureReason::None};
}

bool Scheduler::cancel(RequestId id) {
    auto it = seqs_.find(id);
    if (it == seqs_.end() || is_terminal(it->second.state) || !it->second.spec.cancellable) {
        return false;
    }
    finish(it->second, RequestState::Cancelled, FailureReason::None, clock_.now());
    return true;
}

bool Scheduler::forget(RequestId id) {
    auto it = seqs_.find(id);
    if (it == seqs_.end() || !is_terminal(it->second.state)) {
        return false;
    }
    seqs_.erase(it);
    return true;
}

// ---------------------------------------------------------------- ordering

int Scheduler::running_rank(const Sequence& s) const noexcept { return s.base_rank; }

bool Scheduler::running_better(const Sequence& a, const Sequence& b) const noexcept {
    if (a.base_rank != b.base_rank) return a.base_rank < b.base_rank;
    return a.submit_order < b.submit_order;  // older work wins; newest is the victim
}

bool Scheduler::is_starving(const Sequence& s, TimeUs now) const noexcept {
    return !config_.strict_priority_admission &&
           elapsed_us(s.enqueued_at, now) >= config_.starvation_threshold_us;
}

int Scheduler::waiting_rank(const Sequence& s, TimeUs now) const noexcept {
    if (config_.strict_priority_admission) return s.base_rank;
    if (is_starving(s, now)) return std::min(s.base_rank, kStarvingRank);
    const TimeUs age = elapsed_us(s.enqueued_at, now);
    const std::int64_t boost =
        std::min<TimeUs>(age / config_.aging_interval_us, std::numeric_limits<int>::max() / 4);
    const std::int64_t ranked = static_cast<std::int64_t>(s.base_rank) - boost;
    return static_cast<int>(std::clamp<std::int64_t>(
        ranked, std::numeric_limits<int>::min(), std::numeric_limits<int>::max()));
}

std::vector<RequestId> Scheduler::sorted_waiting(TimeUs now) const {
    std::vector<RequestId> out = waiting_;
    std::stable_sort(out.begin(), out.end(), [&](RequestId a, RequestId b) {
        const Sequence& sa = seqs_.at(a);
        const Sequence& sb = seqs_.at(b);
        const int ra = waiting_rank(sa, now);
        const int rb = waiting_rank(sb, now);
        if (ra != rb) return ra < rb;
        if (config_.strict_priority_admission) {
            const auto oa = sa.spec.admission_order.value_or(sa.submit_order);
            const auto ob = sb.spec.admission_order.value_or(sb.submit_order);
            if (oa != ob) return oa < ob;
            return sa.submit_order < sb.submit_order;
        }
        if (sa.base_rank != sb.base_rank) return sa.base_rank < sb.base_rank;
        if (sa.enqueued_at != sb.enqueued_at) return sa.enqueued_at < sb.enqueued_at;
        return sa.submit_order < sb.submit_order;
    });
    return out;
}

std::vector<RequestId> Scheduler::sorted_running_best_first() const {
    std::vector<RequestId> out = running_;
    std::stable_sort(out.begin(), out.end(), [&](RequestId a, RequestId b) {
        return running_better(seqs_.at(a), seqs_.at(b));
    });
    return out;
}

std::optional<int> Scheduler::effective_rank(RequestId id) const {
    auto it = seqs_.find(id);
    if (it == seqs_.end()) return std::nullopt;
    if (std::find(waiting_.begin(), waiting_.end(), id) == waiting_.end()) return std::nullopt;
    return waiting_rank(it->second, clock_.now());
}

std::vector<RequestId> Scheduler::queue_order() const { return sorted_waiting(clock_.now()); }

PreemptionMode Scheduler::decide_preemption_mode(TokenCount kv_tokens) const noexcept {
    // Stub policy: swapping large contexts is assumed cheaper than recompute.
    // A real cost model (transfer bandwidth vs prefill throughput, tier
    // pressure) belongs here once the cache exposes spill tiers.
    if (config_.enable_swap && kv_tokens >= config_.swap_min_tokens) {
        return PreemptionMode::Swap;
    }
    return PreemptionMode::Recompute;
}

// ---------------------------------------------------------------- helpers

void Scheduler::erase_from(std::vector<RequestId>& v, RequestId id) {
    v.erase(std::remove(v.begin(), v.end(), id), v.end());
}

void Scheduler::add_work(StepPlan& plan, Budget& budget, Sequence& s, Phase phase,
                         TokenCount n) {
    ScheduledWork w;
    w.id = s.spec.id;
    w.phase = phase;
    w.num_tokens = n;
    w.context_offset = s.kv_tokens;
    w.completes_prefill = phase == Phase::Prefill && s.prefill_done + n == s.prefill_target;
    plan.work.push_back(w);
    budget.tokens -= n;
    budget.sequences -= 1;
    if (phase == Phase::Prefill) {
        budget.prefill_tokens -= n;
        plan.prefill_tokens += n;
    } else {
        plan.decode_tokens += n;
    }
}

void Scheduler::remove_work(StepPlan& plan, Budget& budget, RequestId id) {
    auto it = std::find_if(plan.work.begin(), plan.work.end(),
                           [&](const ScheduledWork& w) { return w.id == id; });
    if (it == plan.work.end()) return;
    budget.tokens += it->num_tokens;
    budget.sequences += 1;
    if (it->phase == Phase::Prefill) {
        budget.prefill_tokens += it->num_tokens;
        plan.prefill_tokens -= it->num_tokens;
    } else {
        plan.decode_tokens -= it->num_tokens;
    }
    plan.work.erase(it);
}

void Scheduler::finish(Sequence& s, RequestState terminal, FailureReason why, TimeUs now) {
    const RequestId id = s.spec.id;
    kv_.release(id);
    erase_from(waiting_, id);
    erase_from(running_, id);
    s.state = terminal;
    s.tl.final_state = terminal;
    s.tl.failure = why;
    s.tl.finished = now;
    if (!s.spec.sequence_fingerprint.empty()) {
        // Only drop the fingerprint if no other active request still uses it.
        bool still_used = false;
        for (const auto& [oid, other] : seqs_) {
            if (oid != id && !is_terminal(other.state) &&
                other.spec.sequence_fingerprint == s.spec.sequence_fingerprint) {
                still_used = true;
                break;
            }
        }
        if (!still_used) active_fingerprints_.erase(s.spec.sequence_fingerprint);
    }
    auto& cs = counters_.per_class[index_of(s.spec.workload)];
    switch (terminal) {
        case RequestState::Completed: cs.completed++; break;
        case RequestState::Cancelled: cs.cancelled++; break;
        case RequestState::Failed: cs.failed++; break;
        default: break;
    }
}

void Scheduler::emit_token(Sequence& s, TimeUs now) {
    s.generated++;
    s.tl.generated_tokens = s.generated;
    counters_.per_class[index_of(s.spec.workload)].generated_tokens++;
    if (s.tl.first_token < 0) {
        s.tl.first_token = now;
    } else if (s.last_token_time >= 0) {
        const TimeUs gap = elapsed_us(s.last_token_time, now);
        s.tl.max_inter_token_gap = std::max(s.tl.max_inter_token_gap, gap);
        const auto cls = index_of(s.spec.workload);
        auto& samples = inter_token_samples_[cls];
        const std::size_t window = config_.latency_sample_window;
        if (window == 0) {
            // Statistics disabled: keep nothing.
        } else if (samples.size() < window) {
            samples.push_back(gap);
        } else {
            std::size_t& next = inter_token_next_[cls];
            samples[next % window] = gap;
            next = (next + 1) % window;
        }
    }
    s.last_token_time = now;
}

void Scheduler::preempt(StepPlan& plan, Budget& budget, RequestId victim,
                        PreemptionReason reason, RequestId beneficiary, TimeUs now) {
    Sequence& s = seqs_.at(victim);
    remove_work(plan, budget, victim);

    PreemptionEvent ev;
    ev.id = victim;
    ev.reason = reason;
    ev.kv_tokens = s.kv_tokens;
    ev.beneficiary = beneficiary;
    ev.mode = decide_preemption_mode(s.kv_tokens);

    kv_.release(victim);
    erase_from(running_, victim);

    if (reason == PreemptionReason::KvPressure) {
        counters_.preemptions_kv_pressure++;
    } else {
        counters_.preemptions_priority++;
    }
    counters_.per_class[index_of(s.spec.workload)].preemptions++;
    s.tl.preemptions++;

    if (ev.mode == PreemptionMode::Swap) {
        counters_.preemptions_swap++;
        counters_.swapped_tokens += s.kv_tokens;
        s.swapped = true;  // KV logically retained in a lower tier
    } else {
        counters_.preemptions_recompute++;
        counters_.recomputed_tokens += s.kv_tokens;
        s.swapped = false;
        // Recompute prompt plus everything generated so far; the chunk that
        // completes this prefill yields the next token.
        s.prefill_target = s.spec.prompt_tokens + s.generated;
        s.prefill_done = 0;
        s.kv_tokens = 0;
    }

    const bool exceeded = s.requeues >= config_.max_requeue_count;
    if (s.requeues < std::numeric_limits<std::uint32_t>::max()) {
        s.requeues++;
    }
    if (exceeded) {
        ev.failed = true;
        plan.failed.push_back(victim);
        finish(s, RequestState::Failed, FailureReason::RequeueLimit, now);
    } else {
        s.state = RequestState::Preempted;
        s.enqueued_at = now;
        s.starving_counted = false;
        waiting_.push_back(victim);
    }
    plan.preempted.push_back(ev);
}

// ---------------------------------------------------------------- planning

void Scheduler::schedule_decode(StepPlan& plan, Budget& budget, TimeUs now) {
    const TokenCount bs = kv_.block_size_tokens();
    for (RequestId id : sorted_running_best_first()) {
        auto it = seqs_.find(id);
        if (it == seqs_.end() || it->second.state != RequestState::Decode) continue;
        if (std::find(running_.begin(), running_.end(), id) == running_.end()) continue;
        if (budget.sequences == 0 || budget.tokens == 0) break;
        Sequence& s = it->second;

        bool self_preempted = false;
        for (;;) {
            const BlockCount want = blocks_for_tokens(s.kv_tokens + 1, bs);
            const BlockCount held = kv_.blocks_held(id);
            if (want <= held || kv_.try_reserve(id, want - held)) break;
            // Out of KV: preempt the lowest-priority running sequence (possibly
            // this one). Newest, least urgent work yields first.
            const auto order = sorted_running_best_first();
            const RequestId victim = order.back();
            if (victim != id) {
                // Before evicting lower-priority work, check whether freeing
                // every sequence worse than `id` could even satisfy `want`
                // (e.g. after a DynamicKvCapacity shrink).
                const BlockCount delta = want - held;
                const BlockCount deficit = kv_.over_budget_blocks();
                constexpr BlockCount kMaxBlocks = std::numeric_limits<BlockCount>::max();
                bool can_recover = delta <= kMaxBlocks - deficit && want <= kv_.total_blocks();
                if (can_recover) {
                    const BlockCount target = delta + deficit;
                    BlockCount freeable = kv_.free_blocks();
                    for (auto rit = order.rbegin(); rit != order.rend() && *rit != id; ++rit) {
                        freeable = saturating_add_blocks(freeable, kv_.blocks_held(*rit));
                        if (freeable >= target) break;
                    }
                    can_recover = freeable >= target;
                }
                if (!can_recover) {
                    preempt(plan, budget, id, PreemptionReason::KvPressure, id, now);
                    self_preempted = true;
                    break;
                }
            }
            preempt(plan, budget, victim, PreemptionReason::KvPressure, id, now);
            if (victim == id) {
                self_preempted = true;
                break;
            }
        }
        if (self_preempted || s.state != RequestState::Decode) continue;
        add_work(plan, budget, s, Phase::Decode, 1);
    }
}

void Scheduler::schedule_prefill(StepPlan& plan, Budget& budget) {
    for (RequestId id : sorted_running_best_first()) {
        Sequence& s = seqs_.at(id);
        if (s.state != RequestState::Prefill) continue;
        if (budget.sequences == 0) break;
        const TokenCount remaining = s.prefill_target - s.prefill_done;
        const TokenCount chunk = std::min({config_.prefill_chunk_tokens, remaining,
                                           budget.prefill_tokens, budget.tokens});
        if (chunk == 0) break;
        add_work(plan, budget, s, Phase::Prefill, chunk);
    }
}

bool Scheduler::try_priority_preemption(StepPlan& plan, Budget& budget, const Sequence& cand,
                                        BlockCount need, bool need_slot, bool resumes_decode,
                                        TimeUs now) {
    if (config_.strict_priority_admission || !config_.enable_priority_preemption) return false;
    const std::int64_t cand_rank = static_cast<std::int64_t>(cand.base_rank);
    const std::int64_t min_gap = static_cast<std::int64_t>(config_.priority_preemption_min_rank_gap);
    auto order = sorted_running_best_first();
    std::vector<RequestId> victims;
    const BlockCount deficit = kv_.over_budget_blocks();
    BlockCount freeable = kv_.free_blocks();
    std::uint32_t freed_seqs = 0;
    std::uint64_t freed_tokens = 0;
    std::uint64_t freed_prefill = 0;

    auto satisfied = [&]() noexcept -> bool {
        if (need_slot && running_.size() - victims.size() >= config_.max_running_sequences) {
            return false;
        }
        if (budget.sequences + freed_seqs == 0 ||
            static_cast<std::uint64_t>(budget.tokens) + freed_tokens == 0) {
            return false;
        }
        if (!resumes_decode &&
            static_cast<std::uint64_t>(budget.prefill_tokens) + freed_prefill == 0) {
            return false;
        }
        // If preempting these victims empties running_, the decode watermark is waived.
        const bool empties_running = (victims.size() == order.size());
        const BlockCount eff_watermark =
            empties_running ? BlockCount{0} : config_.admission_watermark_blocks;
        constexpr BlockCount kMaxBlocks = std::numeric_limits<BlockCount>::max();
        if (eff_watermark > kMaxBlocks - need) return false;
        const BlockCount required_free = need + eff_watermark;
        if (required_free > kMaxBlocks - deficit) return false;
        return freeable >= required_free + deficit;
    };

    bool ok = false;
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        const Sequence& v = seqs_.at(*it);
        // A request admitted earlier in this pass (including a starving one) must not be evicted again.
        if (std::find(plan.admitted.begin(), plan.admitted.end(), *it) != plan.admitted.end()) continue;
        if (static_cast<std::int64_t>(running_rank(v)) - cand_rank < min_gap) break;
        victims.push_back(*it);
        freeable = saturating_add_blocks(freeable, kv_.blocks_held(*it));
        const auto wit = std::find_if(plan.work.begin(), plan.work.end(),
                                      [&](const ScheduledWork& w) { return w.id == *it; });
        if (wit != plan.work.end()) {
            freed_seqs += 1;
            freed_tokens += wit->num_tokens;
            if (wit->phase == Phase::Prefill) {
                freed_prefill += wit->num_tokens;
            }
        }
        if (satisfied()) {
            ok = true;
            break;
        }
    }
    if (!ok || victims.empty()) return false;  // would not help; preempt nothing
    for (RequestId v : victims) {
        preempt(plan, budget, v, PreemptionReason::PriorityAdmission, cand.spec.id, now);
    }
    return true;
}

void Scheduler::admit(StepPlan& plan, Budget& budget, TimeUs now) {
    const TokenCount bs = kv_.block_size_tokens();
    bool head_blocked = false;
    std::uint32_t bypassed = 0;

    for (RequestId id : sorted_waiting(now)) {
        auto it = seqs_.find(id);
        if (it == seqs_.end() || is_terminal(it->second.state)) continue;
        Sequence& s = it->second;
        if (s.spec.admission_ready && !s.spec.admission_ready()) continue;

        if (head_blocked) {
            if (config_.strict_priority_admission || bypassed >= config_.admission_lookahead) break;
            ++bypassed;
        }

        const bool resumes_decode = s.swapped && s.prefill_done == s.prefill_target;
        const bool need_slot = running_.size() >= config_.max_running_sequences;
        const bool need_budget = budget.sequences == 0 || budget.tokens == 0 ||
                                 (!resumes_decode && budget.prefill_tokens == 0);

        if (!need_slot && need_budget) break;
        if (need_slot &&
            (config_.strict_priority_admission || !config_.enable_priority_preemption)) {
            break;
        }

        // Reserve the whole (re)prefill context up front; decode growth is
        // reserved block-by-block, protected by the admission watermark.
        const TokenCount ctx = config_.strict_priority_admission
                                   ? static_cast<TokenCount>(s.spec.prompt_tokens + s.spec.max_new_tokens)
                                   : (s.swapped ? std::max(s.kv_tokens, s.prefill_target) : s.prefill_target);
        const BlockCount need = blocks_for_tokens(ctx, bs);
        const BlockCount watermark = config_.strict_priority_admission
                                         ? 0
                                         : (running_.empty() ? 0 : config_.admission_watermark_blocks);
        bool fits = !need_slot && !need_budget &&
                    fits_with_watermark(kv_.free_blocks(), need, watermark);
        if (!fits && try_priority_preemption(plan, budget, s, need, need_slot, resumes_decode, now)) {
            const BlockCount post_watermark =
                config_.strict_priority_admission
                    ? 0
                    : (running_.empty() ? 0 : config_.admission_watermark_blocks);
            const bool post_budget_ok = budget.sequences > 0 && budget.tokens > 0 &&
                                        (resumes_decode || budget.prefill_tokens > 0);
            fits = running_.size() < config_.max_running_sequences && post_budget_ok &&
                   fits_with_watermark(kv_.free_blocks(), need, post_watermark);
        }
        if (fits && kv_.try_reserve(id, need)) {
            if (s.spec.try_admit && !s.spec.try_admit()) {
                kv_.release(id);
                // A class at its concurrency cap must not stall another
                // eligible class. The admission ticket enforces global order.
                continue;
            }
        } else if (fits) {
            fits = false;
        }

        if (!fits) {
            if (running_.size() >= config_.max_running_sequences) {
                break;
            }
            if (!head_blocked) {
                head_blocked = true;
                counters_.admission_kv_blocked++;
                // Never bypass a starving or top-urgency request.
                if (is_starving(s, now) || s.base_rank <= 0) break;
            }
            if (config_.strict_priority_admission || config_.admission_lookahead == 0) break;
            continue;
        }

        if (head_blocked) counters_.lookahead_admissions++;
        erase_from(waiting_, id);
        running_.push_back(id);
        if (s.tl.first_admitted < 0) s.tl.first_admitted = now;
        s.tl.total_queue_time =
            saturating_add_time(s.tl.total_queue_time, elapsed_us(s.enqueued_at, now));
        plan.admitted.push_back(id);

        if (resumes_decode) {
            s.state = RequestState::Decode;  // swap-in handled by the engine
            s.swapped = false;
            continue;  // decodes next step
        }
        s.swapped = false;
        s.state = RequestState::Prefill;
        const TokenCount chunk =
            std::min({config_.prefill_chunk_tokens, s.prefill_target - s.prefill_done,
                      budget.prefill_tokens, budget.tokens});
        if (chunk > 0) add_work(plan, budget, s, Phase::Prefill, chunk);
    }
}

StepPlan Scheduler::plan_step() {
    if (in_flight_) throw std::logic_error("Scheduler::plan_step: previous plan not completed");
    const TimeUs now = clock_.now();

    StepPlan plan;
    plan.step_index = counters_.steps;
    plan.planned_at = now;

    // Starvation telemetry.
    TimeUs oldest = 0;
    for (RequestId id : waiting_) {
        Sequence& s = seqs_.at(id);
        oldest = std::max(oldest, elapsed_us(s.enqueued_at, now));
        if (!s.starving_counted && is_starving(s, now)) {
            s.starving_counted = true;
            counters_.starvation_promotions++;
        }
    }
    counters_.current_oldest_queue_age = oldest;
    counters_.max_starvation_age = std::max(counters_.max_starvation_age, oldest);

    Budget budget;
    budget.tokens = config_.max_step_tokens;
    budget.prefill_tokens = std::min(config_.max_step_prefill_tokens, config_.max_step_tokens);
    budget.sequences = config_.max_step_sequences;

    // Decode first: running sequences keep their inter-token latency and
    // prefill can never starve them (decode capacity is reserved).
    schedule_decode(plan, budget, now);
    // Continue in-progress chunked prefills.
    schedule_prefill(plan, budget);
    // Admit new/preempted work into the remaining budget.
    admit(plan, budget, now);

    in_flight_ = plan.step_index;
    return plan;
}

void Scheduler::complete_step(const StepPlan& plan, const StepOutcome& outcome) {
    if (!in_flight_ || *in_flight_ != plan.step_index) {
        throw std::logic_error("Scheduler::complete_step: plan does not match in-flight step");
    }
    in_flight_.reset();
    const TimeUs now = clock_.now();

    counters_.steps++;
    if (plan.empty()) {
        counters_.empty_steps++;
        return;
    }
    counters_.budget_tokens += config_.max_step_tokens;
    counters_.sequences_scheduled += plan.work.size();

    for (const ScheduledWork& w : plan.work) {
        auto it = seqs_.find(w.id);
        if (it == seqs_.end()) continue;
        Sequence& s = it->second;
        // Cancelled (or otherwise finished) after planning: ignore the result.
        if (s.state != RequestState::Prefill && s.state != RequestState::Decode) continue;
        if (std::find(outcome.stalled.begin(), outcome.stalled.end(), w.id) != outcome.stalled.end()) {
            continue;  // not executed this step; planned again next step
        }

        if (w.phase == Phase::Prefill) {
            counters_.prefill_tokens += w.num_tokens;
            s.prefill_done += w.num_tokens;
            s.kv_tokens += w.num_tokens;
            if (s.prefill_done >= s.prefill_target) {
                s.state = RequestState::Decode;
                emit_token(s, now);
            }
        } else {
            counters_.decode_tokens += w.num_tokens;
            s.kv_tokens += 1;
            emit_token(s, now);
        }

        const bool eos = std::find(outcome.finished_early.begin(), outcome.finished_early.end(),
                                   w.id) != outcome.finished_early.end();
        if (s.generated >= s.spec.max_new_tokens || (eos && s.generated > 0)) {
            finish(s, RequestState::Completed, FailureReason::None, now);
        }
    }
}

// ---------------------------------------------------------------- queries

std::optional<RequestState> Scheduler::state(RequestId id) const {
    auto it = seqs_.find(id);
    if (it == seqs_.end()) return std::nullopt;
    return it->second.state;
}

std::optional<RequestTimeline> Scheduler::timeline(RequestId id) const {
    auto it = seqs_.find(id);
    if (it == seqs_.end()) return std::nullopt;
    RequestTimeline t = it->second.tl;
    t.final_state = it->second.state;
    return t;
}

std::vector<RequestTimeline> Scheduler::timelines() const {
    std::vector<RequestTimeline> out;
    out.reserve(seqs_.size());
    for (const auto& [id, s] : seqs_) {
        RequestTimeline t = s.tl;
        t.final_state = s.state;
        out.push_back(t);
    }
    return out;
}

SchedulerStats Scheduler::stats() const {
    SchedulerStats out = counters_;
    out.now = clock_.now();
    out.active_sequences = running_.size();
    out.queued_sequences = waiting_.size();

    TimeUs oldest = 0;
    for (RequestId id : waiting_) {
        oldest = std::max(oldest, elapsed_us(seqs_.at(id).enqueued_at, out.now));
    }
    out.current_oldest_queue_age = oldest;
    out.max_starvation_age = std::max(out.max_starvation_age, oldest);

    PerClass<std::vector<TimeUs>> queue{}, ttft{}, e2e{};
    for (const auto& [id, s] : seqs_) {
        const auto c = index_of(s.spec.workload);
        if (s.tl.first_admitted >= 0) queue[c].push_back(s.tl.total_queue_time);
        if (s.tl.first_token >= 0) ttft[c].push_back(s.tl.ttft());
        if (s.state == RequestState::Completed) e2e[c].push_back(elapsed_us(s.tl.arrival, s.tl.finished));
    }
    for (std::size_t c = 0; c < kWorkloadClassCount; ++c) {
        auto& cs = out.per_class[c];
        cs.queue_time = LatencySummary::from(queue[c]);
        cs.ttft = LatencySummary::from(ttft[c]);
        cs.end_to_end = LatencySummary::from(e2e[c]);
        cs.inter_token = LatencySummary::from(inter_token_samples_[c]);
    }
    return out;
}

}  // namespace sonder::inference::scheduler
