#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "net/http_client.hpp"
#include "sonder/inference/backend_runtime.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::llamaserver {

struct ChildMetricCounters {
    bool valid = true;
    std::optional<double> prompt_tokens_seconds;
    std::optional<double> predicted_tokens_seconds;
    std::optional<std::uint64_t> prompt_tokens_total;
    std::optional<std::uint64_t> tokens_predicted_total;
    std::optional<std::uint64_t> requests_processing;
    std::optional<std::uint64_t> requests_deferred;
    std::optional<double> n_busy_slots_per_decode;
    std::optional<std::uint64_t> n_decode_total;
    std::optional<std::uint64_t> spec_decode_num_draft_tokens_total;
    std::optional<std::uint64_t> spec_decode_accepted_tokens_total;
    std::optional<std::uint64_t> spec_decode_drafts_total;
    std::vector<std::optional<std::uint64_t>> spec_decode_accepted_tokens_per_position;
    std::optional<double> kv_cache_usage_ratio;
};

struct ChildMetricDerived {
    std::optional<double> mean_accepted_len;
    std::vector<std::optional<double>> acceptance_by_position;
    std::optional<double> speedup_est;
};

struct ChildSlot {
    std::optional<std::uint64_t> id;
    std::optional<bool> is_processing;
    std::optional<std::uint64_t> n_ctx;
};

struct ChildSnapshot {
    std::uint16_t port = 0;
    ChildMetricCounters metrics;
    ChildMetricDerived speculation;
    std::vector<ChildSlot> slots;
    std::chrono::steady_clock::time_point sampled_at{};

    [[nodiscard]] json::Object to_json() const;
};

struct ChildMetricsOptions {
    bool enabled = true;
    std::chrono::seconds stall_seconds{90};
    std::string policy = "warn"; // warn or restart
    std::optional<std::uint64_t> n_max;
    double speedup_coefficient = 0.6;
};

struct ChildMetricsHttpResponse {
    int status = 0;
    std::string body;
};

using ChildMetricsPoll =
    std::function<ChildMetricsHttpResponse(const net::HttpRequest &, const CancellationToken &)>;
using ChildMetricsClock = std::function<std::chrono::steady_clock::time_point()>;

struct ChildMetricsSample {
    std::optional<ChildSnapshot> snapshot;
    std::optional<json::Object> stall;
    std::vector<BackendWarning> warnings;
    bool polled = false;
    bool restart_requested = false;
    bool metrics_unavailable = false;
};

// Pure parsers used by the supervisor and by tests. Unknown Prometheus
// metrics are ignored, and missing whitelisted values remain nullopt.
ChildMetricCounters parse_child_metrics(std::string_view text, bool *valid = nullptr);
std::vector<ChildSlot> parse_child_slots(std::string_view text, bool *valid = nullptr);
ChildMetricDerived derive_child_speculation(const ChildMetricCounters &metrics,
                                            const ChildMetricsOptions &options);

class ChildMetricsSampler {
  public:
    explicit ChildMetricsSampler(ChildMetricsOptions options = {}, ChildMetricsPoll poll = {},
                                 ChildMetricsClock clock = {});

    void reset(std::uint16_t port);
    [[nodiscard]] ChildMetricsSample sample(std::uint16_t port, const CancellationToken &cancel = {});
    [[nodiscard]] ChildMetricsSample sample_at(std::uint16_t port, std::chrono::steady_clock::time_point now,
                                               const CancellationToken &cancel = {});
    [[nodiscard]] ChildMetricsSample sample_at(const net::HttpRequest &metrics_endpoint,
                                               std::chrono::steady_clock::time_point now,
                                               const CancellationToken &cancel = {});

  private:
    ChildMetricsHttpResponse get(const net::HttpRequest &request, const CancellationToken &cancel) const;
    ChildMetricsOptions options_;
    ChildMetricsPoll poll_;
    ChildMetricsClock clock_;
    std::uint16_t port_ = 0;
    bool unavailable_reported_ = false;
    bool have_progress_ = false;
    bool stall_reported_ = false;
    bool restart_issued_ = false;
    bool scrape_failure_reported_ = false;
    std::chrono::steady_clock::time_point last_progress_{};
    std::chrono::steady_clock::time_point detected_at_{};
    std::optional<std::uint64_t> previous_prompt_tokens_;
    std::optional<std::uint64_t> previous_predicted_tokens_;
};

// Short integration name used by the supervisor monitor.
using ChildMetrics = ChildMetricsSampler;

} // namespace sonder::inference::llamaserver
