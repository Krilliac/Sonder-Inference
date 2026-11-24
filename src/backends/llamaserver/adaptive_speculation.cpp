#include "adaptive_speculation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace sonder::inference::llamaserver {
namespace {

constexpr std::array<SpeculationWorkload, kSpeculationWorkloadCount> kAllWorkloads = {
    SpeculationWorkload::general,
    SpeculationWorkload::code_edit,
    SpeculationWorkload::tool_json,
    SpeculationWorkload::prose,
    SpeculationWorkload::reasoning,
};

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
}

std::size_t workload_index(SpeculationWorkload workload) noexcept {
    switch (workload) {
    case SpeculationWorkload::general:
        return 0;
    case SpeculationWorkload::code_edit:
        return 1;
    case SpeculationWorkload::tool_json:
        return 2;
    case SpeculationWorkload::prose:
        return 3;
    case SpeculationWorkload::reasoning:
        return 4;
    }
    return 0;
}

bool valid_probability(double value) noexcept {
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

AdaptiveSpeculationConfig sanitize_config(AdaptiveSpeculationConfig config) {
    if (validate_adaptive_speculation_config(config).ok())
        return config;

    const AdaptiveSpeculationConfig defaults{};
    config.max_draft_n = std::clamp<std::uint32_t>(config.max_draft_n, 1, AdaptiveSpeculationConfig::kMaxDraftN);
    config.min_draft_n = std::min(config.min_draft_n, config.max_draft_n);
    config.initial_draft_n = std::clamp(config.initial_draft_n, config.min_draft_n, config.max_draft_n);
    config.probe_draft_n = std::clamp<std::uint32_t>(config.probe_draft_n, 1, config.max_draft_n);
    config.window_size = std::clamp<std::size_t>(config.window_size, 1, AdaptiveSpeculationConfig::kMaxWindowSize);
    config.min_samples = std::clamp<std::size_t>(config.min_samples, 1, config.window_size);

    if (!valid_probability(config.pause_acceptance_threshold))
        config.pause_acceptance_threshold = defaults.pause_acceptance_threshold;
    if (!valid_probability(config.step_down_acceptance_threshold))
        config.step_down_acceptance_threshold = defaults.step_down_acceptance_threshold;
    if (!valid_probability(config.step_up_acceptance_threshold))
        config.step_up_acceptance_threshold = defaults.step_up_acceptance_threshold;

    if (config.step_down_acceptance_threshold < config.pause_acceptance_threshold)
        config.step_down_acceptance_threshold = config.pause_acceptance_threshold;
    if (config.step_up_acceptance_threshold < config.step_down_acceptance_threshold)
        config.step_up_acceptance_threshold = config.step_down_acceptance_threshold;
    return config;
}

}  // namespace

const char *to_string(SpeculationWorkload workload) noexcept {
    switch (workload) {
    case SpeculationWorkload::general:
        return "general";
    case SpeculationWorkload::code_edit:
        return "code_edit";
    case SpeculationWorkload::tool_json:
        return "tool_json";
    case SpeculationWorkload::prose:
        return "prose";
    case SpeculationWorkload::reasoning:
        return "reasoning";
    }
    return "unknown";
}

std::optional<SpeculationWorkload> parse_speculation_workload(std::string_view text) noexcept {
    if (text == "general")
        return SpeculationWorkload::general;
    if (text == "code_edit")
        return SpeculationWorkload::code_edit;
    if (text == "tool_json")
        return SpeculationWorkload::tool_json;
    if (text == "prose")
        return SpeculationWorkload::prose;
    if (text == "reasoning")
        return SpeculationWorkload::reasoning;
    return std::nullopt;
}

Status validate_adaptive_speculation_config(const AdaptiveSpeculationConfig &config) {
    const auto invalid = [](const char *what) {
        return Status(ErrorCode::invalid_argument, std::string("llamaserver: adaptive speculation: ") + what);
    };
    if (config.max_draft_n == 0 || config.max_draft_n > AdaptiveSpeculationConfig::kMaxDraftN)
        return invalid("max_draft_n must be in [1, 128]");
    if (config.min_draft_n > config.max_draft_n)
        return invalid("min_draft_n must not exceed max_draft_n");
    if (config.initial_draft_n < config.min_draft_n || config.initial_draft_n > config.max_draft_n)
        return invalid("initial_draft_n must be in [min_draft_n, max_draft_n]");
    if (config.probe_draft_n == 0 || config.probe_draft_n > config.max_draft_n)
        return invalid("probe_draft_n must be in [1, max_draft_n]");
    if (config.window_size == 0 || config.window_size > AdaptiveSpeculationConfig::kMaxWindowSize)
        return invalid("window_size must be in [1, 256]");
    if (config.min_samples == 0 || config.min_samples > config.window_size)
        return invalid("min_samples must be in [1, window_size]");
    if (!valid_probability(config.pause_acceptance_threshold))
        return invalid("pause_acceptance_threshold must be finite and in [0, 1]");
    if (!valid_probability(config.step_down_acceptance_threshold))
        return invalid("step_down_acceptance_threshold must be finite and in [0, 1]");
    if (!valid_probability(config.step_up_acceptance_threshold))
        return invalid("step_up_acceptance_threshold must be finite and in [0, 1]");
    if (config.pause_acceptance_threshold > config.step_down_acceptance_threshold)
        return invalid("pause_acceptance_threshold must not exceed step_down_acceptance_threshold");
    if (config.step_down_acceptance_threshold > config.step_up_acceptance_threshold)
        return invalid("step_down_acceptance_threshold must not exceed step_up_acceptance_threshold");
    return Status::success();
}

SpeculationObservation observation_from_timings(const Timings &timings, SpeculationWorkload workload) {
    SpeculationObservation obs;
    obs.workload = workload;

    if (timings.draft_present || timings.draft_tokens != 0) {
        obs.draft_n = timings.draft_tokens;
    } else if (timings.backend_timings && timings.backend_timings->draft_n) {
        obs.draft_n = *timings.backend_timings->draft_n;
    }

    if (timings.draft_accepted_present || timings.draft_accepted_tokens != 0) {
        obs.draft_n_accepted = timings.draft_accepted_tokens;
    } else if (timings.backend_timings && timings.backend_timings->draft_n_accepted) {
        obs.draft_n_accepted = *timings.backend_timings->draft_n_accepted;
    }

    if (obs.draft_n_accepted > obs.draft_n)
        obs.draft_n_accepted = obs.draft_n;

    if (timings.completion_present || timings.completion_tokens != 0) {
        obs.predicted_n = timings.completion_tokens;
    } else if (timings.backend_timings && timings.backend_timings->predicted_n) {
        obs.predicted_n = *timings.backend_timings->predicted_n;
    }

    if ((timings.speed_present || timings.predicted_tokens_per_second > 0.0) &&
        std::isfinite(timings.predicted_tokens_per_second) && timings.predicted_tokens_per_second > 0.0) {
        obs.decode_tps = timings.predicted_tokens_per_second;
    } else if (obs.predicted_n > 0 && timings.eval_ns > 0) {
        const double tps = static_cast<double>(obs.predicted_n) * 1e9 / static_cast<double>(timings.eval_ns);
        if (std::isfinite(tps) && tps > 0.0)
            obs.decode_tps = tps;
    } else if (obs.predicted_n > 0 && timings.backend_timings && timings.backend_timings->predicted_ms &&
               std::isfinite(*timings.backend_timings->predicted_ms) && *timings.backend_timings->predicted_ms > 0.0) {
        const double tps =
            static_cast<double>(obs.predicted_n) * 1000.0 / *timings.backend_timings->predicted_ms;
        if (std::isfinite(tps) && tps > 0.0)
            obs.decode_tps = tps;
    }

    return obs;
}

SpeculationObservation observation_from_timings(const BackendTimings &timings, SpeculationWorkload workload) {
    SpeculationObservation obs;
    obs.workload = workload;
    obs.draft_n = timings.draft_n.value_or(0);
    obs.draft_n_accepted = std::min(timings.draft_n_accepted.value_or(0), obs.draft_n);
    obs.predicted_n = timings.predicted_n.value_or(0);
    if (obs.predicted_n > 0 && timings.predicted_ms && std::isfinite(*timings.predicted_ms) &&
        *timings.predicted_ms > 0.0) {
        const double tps = static_cast<double>(obs.predicted_n) * 1000.0 / *timings.predicted_ms;
        if (std::isfinite(tps) && tps > 0.0)
            obs.decode_tps = tps;
    }
    return obs;
}

AdaptiveSpeculationPolicy::AdaptiveSpeculationPolicy(AdaptiveSpeculationConfig config)
    : config_(sanitize_config(config)) {
    for (auto &bucket : buckets_) {
        bucket.window.resize(config_.window_size);
        reset_bucket_locked(bucket);
    }
}

void AdaptiveSpeculationPolicy::clear_window(BucketState &bucket) noexcept {
    std::fill(bucket.window.begin(), bucket.window.end(), Sample{});
    bucket.head = 0;
    bucket.count = 0;
}

void AdaptiveSpeculationPolicy::push_sample(BucketState &bucket, const Sample &sample) noexcept {
    if (bucket.window.empty())
        return;
    bucket.window[bucket.head] = sample;
    bucket.head = (bucket.head + 1) % bucket.window.size();
    if (bucket.count < bucket.window.size())
        ++bucket.count;
}

AdaptiveSpeculationPolicy::WindowStats AdaptiveSpeculationPolicy::compute_window_stats(
    const BucketState &bucket) noexcept {
    WindowStats stats;
    long double total_draft = 0.0L;
    long double total_accepted = 0.0L;
    long double total_tps = 0.0L;
    std::size_t tps_samples = 0;
    for (std::size_t i = 0; i < bucket.count; ++i) {
        const Sample &sample = bucket.window[i];
        total_draft += static_cast<long double>(sample.draft_n);
        total_accepted += static_cast<long double>(sample.draft_n_accepted);
        stats.draft_tokens = saturating_add(stats.draft_tokens, sample.draft_n);
        stats.accepted_tokens = saturating_add(stats.accepted_tokens, sample.draft_n_accepted);
        if (std::isfinite(sample.decode_tps) && sample.decode_tps > 0.0) {
            total_tps += static_cast<long double>(sample.decode_tps);
            ++tps_samples;
        }
    }
    if (total_draft > 0.0L) {
        const double ratio = static_cast<double>(total_accepted / total_draft);
        stats.rolling_acceptance = std::isfinite(ratio) ? std::clamp(ratio, 0.0, 1.0) : 0.0;
    }
    if (tps_samples > 0) {
        const double mean_tps = static_cast<double>(total_tps / static_cast<long double>(tps_samples));
        if (std::isfinite(mean_tps) && mean_tps > 0.0)
            stats.rolling_decode_tps = mean_tps;
    }
    return stats;
}

void AdaptiveSpeculationPolicy::reset_bucket_locked(BucketState &bucket) const {
    clear_window(bucket);
    bucket.current_draft_n = config_.initial_draft_n;
    bucket.paused = (config_.initial_draft_n == 0);
    bucket.probe_in_flight = false;
    bucket.cooldown_remaining = bucket.paused ? config_.cooldown_requests : 0;
    bucket.total_observations = 0;
    bucket.ignored_observations = 0;
    bucket.clamped_observations = 0;
    bucket.probes_issued = 0;
    bucket.probe_recoveries = 0;
    bucket.last_reason = !config_.enabled ? "disabled" : (bucket.paused ? "initial_paused" : "initial");
}

SpeculationDecision AdaptiveSpeculationPolicy::recommend(SpeculationWorkload workload) {
    std::lock_guard<std::mutex> lock(mutex_);
    BucketState &bucket = buckets_[workload_index(workload)];
    if (!config_.enabled) {
        return SpeculationDecision{
            config_.initial_draft_n,
            config_.initial_draft_n == 0,
            false,
            0.0,
            0,
            "disabled",
        };
    }

    const WindowStats stats = compute_window_stats(bucket);
    if (!bucket.paused) {
        return SpeculationDecision{
            bucket.current_draft_n,
            false,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
        };
    }

    if (config_.cooldown_requests == 0) {
        return SpeculationDecision{
            0,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "paused_no_probe",
        };
    }

    if (bucket.probe_in_flight) {
        return SpeculationDecision{
            0,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "paused_probe_in_flight",
        };
    }

    if (bucket.cooldown_remaining > 0) {
        --bucket.cooldown_remaining;
        return SpeculationDecision{
            0,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            "paused_cooldown",
        };
    }

    bucket.probe_in_flight = true;
    bucket.probes_issued = saturating_add(bucket.probes_issued, 1);
    return SpeculationDecision{
        config_.probe_draft_n,
        false,
        true,
        stats.rolling_acceptance,
        bucket.count,
        "cooldown_probe",
    };
}

SpeculationDecision AdaptiveSpeculationPolicy::observe(const SpeculationObservation &obs) {
    std::lock_guard<std::mutex> lock(mutex_);
    BucketState &bucket = buckets_[workload_index(obs.workload)];
    if (!config_.enabled) {
        return SpeculationDecision{
            config_.initial_draft_n,
            config_.initial_draft_n == 0,
            false,
            0.0,
            0,
            "disabled",
        };
    }

    const std::uint64_t min_tokens = std::max<std::uint64_t>(1, config_.min_draft_tokens_per_sample);
    if (obs.draft_n < min_tokens) {
        bucket.ignored_observations = saturating_add(bucket.ignored_observations, 1);
        // Release an in-flight probe when the probe itself completes below the
        // evaluation floor (including draft_n == 0 on immediate-EOS replies) so
        // the bucket cannot wedge permanently in paused_probe_in_flight, and
        // restart cooldown so short responses do not spin-probe every request.
        // Concurrent non-probe requests (is_probe == false && draft_n == 0)
        // leave the in-flight probe untouched.
        if (bucket.paused && bucket.probe_in_flight && (obs.is_probe || obs.draft_n > 0)) {
            bucket.probe_in_flight = false;
            bucket.cooldown_remaining = config_.cooldown_requests;
        }
        const WindowStats stats = compute_window_stats(bucket);
        return SpeculationDecision{
            bucket.paused ? 0u : bucket.current_draft_n,
            bucket.paused,
            false,
            stats.rolling_acceptance,
            bucket.count,
            obs.draft_n == 0 ? "ignored_no_draft" : "ignored_below_min_draft_tokens",
        };
    }

    std::uint64_t accepted = obs.draft_n_accepted;
    if (accepted > obs.draft_n) {
        accepted = obs.draft_n;
        bucket.clamped_observations = saturating_add(bucket.clamped_observations, 1);
    }
    const double clean_tps = (std::isfinite(obs.decode_tps) && obs.decode_tps > 0.0) ? obs.decode_tps : 0.0;
    const Sample sample{obs.draft_n, accepted, clean_tps};
    bucket.total_observations = saturating_add(bucket.total_observations, 1);

    const std::uint32_t active_floor = std::max<std::uint32_t>(1, config_.min_draft_n);

    // Resolve an in-flight cooldown probe directly against its own acceptance
    // ratio so stale pre-pause history does not block recovery.
    if (bucket.paused && bucket.probe_in_flight) {
        bucket.probe_in_flight = false;
        const double probe_acceptance = std::clamp(
            static_cast<double>(static_cast<long double>(accepted) / static_cast<long double>(obs.draft_n)), 0.0, 1.0);
        if (probe_acceptance >= config_.pause_acceptance_threshold) {
            clear_window(bucket);
            push_sample(bucket, sample);
            bucket.paused = false;
            bucket.cooldown_remaining = 0;
            bucket.current_draft_n = std::clamp(config_.probe_draft_n, active_floor, config_.max_draft_n);
            bucket.probe_recoveries = saturating_add(bucket.probe_recoveries, 1);
            bucket.last_reason = "probe_recovered";
            const WindowStats stats = compute_window_stats(bucket);
            return SpeculationDecision{
                bucket.current_draft_n,
                false,
                false,
                stats.rolling_acceptance,
                bucket.count,
                bucket.last_reason,
            };
        }

        push_sample(bucket, sample);
        bucket.current_draft_n = 0;
        bucket.cooldown_remaining = config_.cooldown_requests;
        bucket.last_reason = "probe_failed_repaused";
        const WindowStats stats = compute_window_stats(bucket);
        return SpeculationDecision{
            0,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
        };
    }

    push_sample(bucket, sample);
    const WindowStats stats = compute_window_stats(bucket);

    if (bucket.paused) {
        if (bucket.count >= config_.min_samples &&
            stats.rolling_acceptance >= config_.step_down_acceptance_threshold) {
            bucket.paused = false;
            bucket.cooldown_remaining = 0;
            bucket.current_draft_n = std::clamp(config_.probe_draft_n, active_floor, config_.max_draft_n);
            bucket.last_reason = "recovered";
            return SpeculationDecision{
                bucket.current_draft_n,
                false,
                false,
                stats.rolling_acceptance,
                bucket.count,
                bucket.last_reason,
            };
        }
        bucket.last_reason = "paused_low_acceptance";
        return SpeculationDecision{
            0,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
        };
    }

    if (bucket.count < config_.min_samples) {
        bucket.last_reason = "warming_up";
        return SpeculationDecision{
            bucket.current_draft_n,
            false,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
        };
    }

    if (config_.min_draft_n == 0 && stats.rolling_acceptance < config_.pause_acceptance_threshold) {
        bucket.paused = true;
        bucket.probe_in_flight = false;
        bucket.current_draft_n = 0;
        bucket.cooldown_remaining = config_.cooldown_requests;
        bucket.last_reason = "paused_low_acceptance";
        return SpeculationDecision{
            0,
            true,
            false,
            stats.rolling_acceptance,
            bucket.count,
            bucket.last_reason,
        };
    }

    if (stats.rolling_acceptance < config_.step_down_acceptance_threshold) {
        if (bucket.current_draft_n > active_floor) {
            --bucket.current_draft_n;
            bucket.last_reason = "step_down";
        } else {
            bucket.current_draft_n = active_floor;
            bucket.last_reason = "at_min_draft";
        }
    } else if (stats.rolling_acceptance >= config_.step_up_acceptance_threshold) {
        if (bucket.current_draft_n < config_.max_draft_n) {
            ++bucket.current_draft_n;
            bucket.last_reason = "step_up";
        } else {
            bucket.current_draft_n = config_.max_draft_n;
            bucket.last_reason = "at_max_draft";
        }
    } else {
        bucket.last_reason = "hold_hysteresis";
    }

    return SpeculationDecision{
        bucket.current_draft_n,
        false,
        false,
        stats.rolling_acceptance,
        bucket.count,
        bucket.last_reason,
    };
}

void AdaptiveSpeculationPolicy::cancel_probe(SpeculationWorkload workload) {
    std::lock_guard<std::mutex> lock(mutex_);
    BucketState &bucket = buckets_[workload_index(workload)];
    if (bucket.paused && bucket.probe_in_flight) {
        bucket.probe_in_flight = false;
        bucket.cooldown_remaining = config_.cooldown_requests;
    }
}

json::Object AdaptiveSpeculationPolicy::status_json() const {
    std::lock_guard<std::mutex> lock(mutex_);

    json::Object config_obj;
    config_obj.set("min_draft_n", static_cast<unsigned long long>(config_.min_draft_n));
    config_obj.set("initial_draft_n", static_cast<unsigned long long>(config_.initial_draft_n));
    config_obj.set("max_draft_n", static_cast<unsigned long long>(config_.max_draft_n));
    config_obj.set("window_size", static_cast<unsigned long long>(config_.window_size));
    config_obj.set("min_samples", static_cast<unsigned long long>(config_.min_samples));
    config_obj.set("pause_acceptance_threshold", config_.pause_acceptance_threshold);
    config_obj.set("step_down_acceptance_threshold", config_.step_down_acceptance_threshold);
    config_obj.set("step_up_acceptance_threshold", config_.step_up_acceptance_threshold);
    config_obj.set("cooldown_requests", static_cast<unsigned long long>(config_.cooldown_requests));
    config_obj.set("probe_draft_n", static_cast<unsigned long long>(config_.probe_draft_n));
    config_obj.set("min_draft_tokens_per_sample",
                   static_cast<unsigned long long>(config_.min_draft_tokens_per_sample));

    json::Object workloads_obj;
    for (const auto workload : kAllWorkloads) {
        const BucketState &bucket = buckets_[workload_index(workload)];
        const WindowStats stats = compute_window_stats(bucket);
        json::Object entry;
        entry.set("draft_n_max", static_cast<unsigned long long>(bucket.paused ? 0u : bucket.current_draft_n));
        entry.set("paused", bucket.paused);
        entry.set("probe_in_flight", bucket.probe_in_flight);
        entry.set("cooldown_remaining", static_cast<unsigned long long>(bucket.cooldown_remaining));
        entry.set("rolling_acceptance", stats.rolling_acceptance);
        entry.set("rolling_decode_tps", stats.rolling_decode_tps);
        entry.set("sample_count", static_cast<unsigned long long>(bucket.count));
        entry.set("window_draft_tokens", static_cast<unsigned long long>(stats.draft_tokens));
        entry.set("window_accepted_tokens", static_cast<unsigned long long>(stats.accepted_tokens));
        entry.set("total_observations", static_cast<unsigned long long>(bucket.total_observations));
        entry.set("ignored_observations", static_cast<unsigned long long>(bucket.ignored_observations));
        entry.set("clamped_observations", static_cast<unsigned long long>(bucket.clamped_observations));
        entry.set("probes_issued", static_cast<unsigned long long>(bucket.probes_issued));
        entry.set("probe_recoveries", static_cast<unsigned long long>(bucket.probe_recoveries));
        entry.set("reason", bucket.last_reason);
        workloads_obj.set(to_string(workload), std::move(entry));
    }

    json::Object root;
    root.set("enabled", config_.enabled);
    root.set("config", std::move(config_obj));
    root.set("workloads", std::move(workloads_obj));
    return root;
}

void AdaptiveSpeculationPolicy::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto &bucket : buckets_)
        reset_bucket_locked(bucket);
}

}  // namespace sonder::inference::llamaserver
