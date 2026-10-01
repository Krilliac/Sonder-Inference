#include "child_metrics.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <limits>

namespace sonder::inference::llamaserver {
namespace {

using Clock = std::chrono::steady_clock;

bool finite(double value) { return std::isfinite(value); }
std::string trim(std::string_view value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string_view::npos)
        return {};
    const auto last = value.find_last_not_of(" \t\r\n");
    return std::string(value.substr(first, last - first + 1));
}

std::optional<std::uint64_t> parse_counter(std::string_view text) {
    std::uint64_t integer = 0;
    const auto direct = std::from_chars(text.data(), text.data() + text.size(), integer);
    if (direct.ec == std::errc{} && direct.ptr == text.data() + text.size())
        return integer;
    // UINT64_MAX rounds to 2^64 with MSVC's long double too. Never cast that
    // exclusive upper bound. Integer text is parsed exactly above.
    double value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !finite(value) ||
        value < 0.0 || value >= 0x1p64 || std::floor(value) != value)
        return std::nullopt;
    return static_cast<std::uint64_t>(value);
}

void assign_scalar(ChildMetricCounters &out, std::string_view name, std::string_view text, bool &valid) {
    double value = 0;
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || !finite(value) ||
        value < 0.0) {
        valid = false;
        return;
    }
    if (name == "llamacpp:prompt_tokens_seconds")
        out.prompt_tokens_seconds = value;
    else if (name == "llamacpp:predicted_tokens_seconds")
        out.predicted_tokens_seconds = value;
    else if (name == "llamacpp:n_busy_slots_per_decode")
        out.n_busy_slots_per_decode = value;
    else if (name == "llamacpp:kv_cache_usage_ratio")
        out.kv_cache_usage_ratio = value;
    else if (name == "llamacpp:prompt_tokens_total")
        out.prompt_tokens_total = parse_counter(text);
    else if (name == "llamacpp:tokens_predicted_total")
        out.tokens_predicted_total = parse_counter(text);
    else if (name == "llamacpp:requests_processing")
        out.requests_processing = parse_counter(text);
    else if (name == "llamacpp:requests_deferred")
        out.requests_deferred = parse_counter(text);
    else if (name == "llamacpp:n_decode_total")
        out.n_decode_total = parse_counter(text);
    else if (name == "llamacpp:spec_decode_num_draft_tokens_total")
        out.spec_decode_num_draft_tokens_total = parse_counter(text);
    else if (name == "llamacpp:spec_decode_accepted_tokens_total")
        out.spec_decode_accepted_tokens_total = parse_counter(text);
    else if (name == "llamacpp:spec_decode_drafts_total")
        out.spec_decode_drafts_total = parse_counter(text);
    if ((name.ends_with("_total") || name == "llamacpp:requests_processing" ||
         name == "llamacpp:requests_deferred") &&
        !parse_counter(text))
        valid = false;
}

std::optional<std::uint64_t> value_uint(const json::Value *value) {
    if (!value || !value->is_integer())
        return std::nullopt;
    if (value->as_int() < 0)
        return std::nullopt;
    return value->as_uint();
}

std::optional<bool> value_bool(const json::Value *value) {
    if (!value)
        return std::nullopt;
    if (value->is_bool())
        return value->as_bool();
    return std::nullopt;
}

json::Value optional_uint(const std::optional<std::uint64_t> &value) {
    return value ? json::Value(*value) : json::Value(nullptr);
}
json::Value optional_double(const std::optional<double> &value) {
    return value ? json::Value(*value) : json::Value(nullptr);
}

void set_metric(json::Object &object, std::string key, const std::optional<std::uint64_t> &value) {
    object.set(std::move(key), optional_uint(value));
}
void set_metric(json::Object &object, std::string key, const std::optional<double> &value) {
    object.set(std::move(key), optional_double(value));
}

} // namespace

ChildMetricCounters parse_child_metrics(std::string_view text, bool *valid_out) {
    ChildMetricCounters result;
    constexpr std::size_t kMaxBody = 1024u * 1024u;
    constexpr std::size_t kMaxLine = 4096u;
    bool valid = text.size() <= kMaxBody;
    bool found = false;
    std::size_t offset = 0;
    while (offset < text.size() && valid) {
        const auto end_line = text.find('\n', offset);
        const auto end = end_line == std::string_view::npos ? text.size() : end_line;
        if (end - offset > kMaxLine) {
            valid = false;
            break;
        }
        std::string_view line = text.substr(offset, end - offset);
        if (!line.empty() && line.back() == '\r')
            line.remove_suffix(1);
        const auto line_start = line.find_first_not_of(" \t");
        if (line_start != std::string_view::npos)
            line.remove_prefix(line_start);
        else
            line = {};
        if (!line.empty() && line.front() != '#') {
            const auto space = line.find_first_of(" \t");
            if (space == std::string_view::npos) {
                valid = false;
                break;
            }
            auto name = trim(line.substr(0, space));
            const auto raw_end = line.find_first_not_of(" \t", space);
            if (name.empty() || raw_end == std::string_view::npos) {
                valid = false;
                break;
            }
            const auto raw_end_token = line.find_first_of(" \t", raw_end);
            const auto raw =
                line.substr(raw_end, raw_end_token == std::string_view::npos ? line.size() - raw_end
                                                                             : raw_end_token - raw_end);
            // Strip the optional label set while retaining only the metric name.
            const auto labels = name.find('{');
            const bool had_labels = labels != std::string::npos;
            std::string position;
            if (labels != std::string::npos) {
                const auto close = name.find('}', labels);
                if (close == std::string::npos) {
                    valid = false;
                    break;
                }
                const auto label_text = name.substr(labels + 1, close - labels - 1);
                const auto marker = label_text == "position=\"0\"" ? 0u : std::string::npos;
                if (marker != std::string::npos)
                    position = "0";
                else if (label_text.size() > 11 && label_text.starts_with("position=\"") &&
                         label_text.back() == '"')
                    position = label_text.substr(10, label_text.size() - 11);
                name.resize(labels);
            }
            const bool known =
                name == "llamacpp:prompt_tokens_seconds" || name == "llamacpp:predicted_tokens_seconds" ||
                name == "llamacpp:prompt_tokens_total" || name == "llamacpp:tokens_predicted_total" ||
                name == "llamacpp:requests_processing" || name == "llamacpp:requests_deferred" ||
                name == "llamacpp:n_busy_slots_per_decode" || name == "llamacpp:n_decode_total" ||
                name == "llamacpp:spec_decode_num_draft_tokens_total" ||
                name == "llamacpp:spec_decode_accepted_tokens_total" ||
                name == "llamacpp:spec_decode_drafts_total" ||
                name == "llamacpp:spec_decode_accepted_tokens_per_pos_total" ||
                name == "llamacpp:kv_cache_usage_ratio";
            if (!known) {
                offset = end_line == std::string_view::npos ? text.size() : end_line + 1;
                continue;
            }
            found = true;
            if (name == "llamacpp:spec_decode_accepted_tokens_per_pos_total" && had_labels &&
                position.empty()) {
                valid = false;
            } else if (name == "llamacpp:spec_decode_accepted_tokens_per_pos_total" && !position.empty()) {
                try {
                    std::size_t index = 0;
                    const auto parsed =
                        std::from_chars(position.data(), position.data() + position.size(), index);
                    if (parsed.ec != std::errc{} || parsed.ptr != position.data() + position.size() ||
                        index >= 256) {
                        valid = false;
                        break;
                    }
                    if (index >= result.spec_decode_accepted_tokens_per_position.size())
                        result.spec_decode_accepted_tokens_per_position.resize(index + 1);
                    result.spec_decode_accepted_tokens_per_position[index] = parse_counter(raw);
                    if (!result.spec_decode_accepted_tokens_per_position[index])
                        valid = false;
                } catch (...) {
                    valid = false;
                }
            } else {
                assign_scalar(result, name, raw, valid);
            }
        }
        offset = end_line == std::string_view::npos ? text.size() : end_line + 1;
    }
    result.valid = valid && found;
    if (valid_out)
        *valid_out = result.valid;
    return result;
}

std::vector<ChildSlot> parse_child_slots(std::string_view text, bool *valid_out) {
    std::vector<ChildSlot> slots;
    bool valid = text.size() <= 1024u * 1024u;
    if (!valid) {
        if (valid_out)
            *valid_out = false;
        return slots;
    }
    const auto parsed = json::parse(text);
    if (!parsed || !parsed.value().is_array())
        valid = false;
    if (!valid) {
        if (valid_out)
            *valid_out = false;
        return slots;
    }
    if (parsed.value().as_array().size() > 1024) {
        if (valid_out)
            *valid_out = false;
        return slots;
    }
    for (const auto &value : parsed.value().as_array()) {
        if (!value.is_object()) {
            valid = false;
            break;
        }
        const auto &object = value.as_object();
        ChildSlot slot;
        slot.id = value_uint(object.find("id"));
        if (!slot.id)
            slot.id = value_uint(object.find("id_slot"));
        slot.is_processing = value_bool(object.find("is_processing"));
        slot.n_ctx = value_uint(object.find("n_ctx"));
        slots.push_back(std::move(slot));
    }
    if (valid_out)
        *valid_out = valid;
    if (!valid)
        slots.clear();
    return slots;
}

ChildMetricDerived derive_child_speculation(const ChildMetricCounters &metrics,
                                            const ChildMetricsOptions &options) {
    ChildMetricDerived derived;
    if (metrics.spec_decode_drafts_total && *metrics.spec_decode_drafts_total > 0 &&
        metrics.spec_decode_accepted_tokens_total) {
        derived.mean_accepted_len = static_cast<double>(*metrics.spec_decode_accepted_tokens_total) /
                                    static_cast<double>(*metrics.spec_decode_drafts_total);
    }
    derived.acceptance_by_position.reserve(metrics.spec_decode_accepted_tokens_per_position.size());
    for (const auto &accepted : metrics.spec_decode_accepted_tokens_per_position) {
        if (accepted && metrics.spec_decode_drafts_total && *metrics.spec_decode_drafts_total > 0)
            derived.acceptance_by_position.push_back(static_cast<double>(*accepted) /
                                                     static_cast<double>(*metrics.spec_decode_drafts_total));
        else
            derived.acceptance_by_position.push_back(std::nullopt);
    }
    if (derived.mean_accepted_len && options.n_max && finite(options.speedup_coefficient) &&
        options.speedup_coefficient >= 0.0) {
        const double denominator = 1.0 + options.speedup_coefficient * static_cast<double>(*options.n_max);
        if (finite(denominator) && denominator > 0.0)
            derived.speedup_est = (1.0 + *derived.mean_accepted_len) / denominator;
    }
    return derived;
}

json::Object ChildSnapshot::to_json() const {
    json::Object root;
    root.set("port", port);
    json::Object metrics_object;
    set_metric(metrics_object, "prompt_tokens_seconds", metrics.prompt_tokens_seconds);
    set_metric(metrics_object, "predicted_tokens_seconds", metrics.predicted_tokens_seconds);
    set_metric(metrics_object, "prompt_tokens_total", metrics.prompt_tokens_total);
    set_metric(metrics_object, "tokens_predicted_total", metrics.tokens_predicted_total);
    set_metric(metrics_object, "requests_processing", metrics.requests_processing);
    set_metric(metrics_object, "requests_deferred", metrics.requests_deferred);
    set_metric(metrics_object, "n_busy_slots_per_decode", metrics.n_busy_slots_per_decode);
    set_metric(metrics_object, "n_decode_total", metrics.n_decode_total);
    set_metric(metrics_object, "spec_decode_num_draft_tokens_total",
               metrics.spec_decode_num_draft_tokens_total);
    set_metric(metrics_object, "spec_decode_accepted_tokens_total",
               metrics.spec_decode_accepted_tokens_total);
    set_metric(metrics_object, "spec_decode_drafts_total", metrics.spec_decode_drafts_total);
    set_metric(metrics_object, "kv_cache_usage_ratio", metrics.kv_cache_usage_ratio);
    json::Array positions;
    for (const auto &value : speculation.acceptance_by_position)
        positions.emplace_back(optional_double(value));
    metrics_object.set("spec_decode_accepted_tokens_per_pos_total", [&] {
        json::Array values;
        for (const auto &value : metrics.spec_decode_accepted_tokens_per_position)
            values.emplace_back(optional_uint(value));
        return json::Value(std::move(values));
    }());
    root.set("metrics", std::move(metrics_object));
    json::Array slot_values;
    for (const auto &slot : slots) {
        json::Object item;
        item.set("id", slot.id ? json::Value(*slot.id) : json::Value(nullptr));
        item.set("is_processing",
                 slot.is_processing ? json::Value(*slot.is_processing) : json::Value(nullptr));
        item.set("n_ctx", slot.n_ctx ? json::Value(*slot.n_ctx) : json::Value(nullptr));
        slot_values.emplace_back(std::move(item));
    }
    root.set("slots", std::move(slot_values));
    json::Object speculation_object;
    speculation_object.set("mean_accepted_len", optional_double(speculation.mean_accepted_len));
    speculation_object.set("acceptance_by_position", std::move(positions));
    speculation_object.set("speedup_est", optional_double(speculation.speedup_est));
    root.set("speculation", std::move(speculation_object));
    root.set(
        "sampled_at",
        static_cast<std::int64_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(sampled_at.time_since_epoch()).count()));
    return root;
}

ChildMetricsSampler::ChildMetricsSampler(ChildMetricsOptions options, ChildMetricsPoll poll,
                                         ChildMetricsClock clock)
    : options_(std::move(options)), poll_(std::move(poll)), clock_(std::move(clock)) {
    if (!poll_)
        poll_ = [](const net::HttpRequest &request, const CancellationToken &cancel) {
            std::string body;
            const auto response = net::http_request_buffered(request, body, cancel, 1024u * 1024u);
            return ChildMetricsHttpResponse{response ? response.value().status : 0, std::move(body)};
        };
    if (!clock_)
        clock_ = [] { return Clock::now(); };
}

void ChildMetricsSampler::reset(std::uint16_t port) {
    port_ = port;
    unavailable_reported_ = false;
    have_progress_ = false;
    stall_reported_ = false;
    restart_issued_ = false;
    scrape_failure_reported_ = false;
    previous_prompt_tokens_.reset();
    previous_predicted_tokens_.reset();
}

ChildMetricsHttpResponse ChildMetricsSampler::get(const net::HttpRequest &request,
                                                  const CancellationToken &cancel) const {
    try {
        return poll_(request, cancel);
    } catch (...) {
        // Observation failure must not fail a healthy supervisor. Exception
        // text from the upstream is deliberately excluded from diagnostics.
        return {};
    }
}

ChildMetricsSample ChildMetricsSampler::sample(std::uint16_t port, const CancellationToken &cancel) {
    return sample_at(port, clock_(), cancel);
}

ChildMetricsSample ChildMetricsSampler::sample_at(std::uint16_t port, Clock::time_point now,
                                                  const CancellationToken &cancel) {
    net::HttpRequest endpoint;
    endpoint.host = "127.0.0.1";
    endpoint.port = port;
    return sample_at(endpoint, now, cancel);
}

ChildMetricsSample ChildMetricsSampler::sample_at(const net::HttpRequest &metrics_endpoint,
                                                  Clock::time_point now, const CancellationToken &cancel) {
    ChildMetricsSample result;
    const std::uint16_t port = metrics_endpoint.port;
    if (port != port_)
        reset(port);
    if (unavailable_reported_ || cancel.cancelled())
        return result;
    result.polled = true;
    net::HttpRequest metrics_request = metrics_endpoint;
    // The supplied endpoint target is the configured upstream base path.
    while (!metrics_request.target.empty() && metrics_request.target.back() == '/')
        metrics_request.target.pop_back();
    metrics_request.target += "/metrics";
    metrics_request.connect_timeout =
        std::min(metrics_request.connect_timeout, std::chrono::milliseconds{250});
    metrics_request.total_timeout = std::min(metrics_request.total_timeout, std::chrono::milliseconds{250});
    metrics_request.tls.handshake_timeout =
        std::min(metrics_request.tls.handshake_timeout, std::chrono::milliseconds{250});
    const auto metrics_response = get(metrics_request, cancel);
    if (metrics_response.status == 404) {
        if (!unavailable_reported_) {
            result.metrics_unavailable = true;
            unavailable_reported_ = true;
            result.warnings.push_back({"metrics_unavailable",
                                       "warning",
                                       "child_metrics",
                                       "child does not expose /metrics or /slots",
                                       {},
                                       1});
        }
        return result;
    }
    if (metrics_response.status < 200 || metrics_response.status >= 300) {
        have_progress_ = false;
        stall_reported_ = false;
        restart_issued_ = false;
        if (!scrape_failure_reported_) {
            result.warnings.push_back(
                {"metrics_scrape_failed", "warning", "child_metrics", "GET /metrics failed", {}, 1});
            scrape_failure_reported_ = true;
        }
        return result;
    }
    bool metrics_valid = false;
    const auto counters = parse_child_metrics(metrics_response.body, &metrics_valid);
    net::HttpRequest slots_request = metrics_request;
    const auto slash = slots_request.target.find_last_of('/');
    slots_request.target =
        slash == std::string::npos ? "/slots" : slots_request.target.substr(0, slash + 1) + "slots";
    const auto slots_response = get(slots_request, cancel);
    if (slots_response.status == 404) {
        if (!unavailable_reported_) {
            result.metrics_unavailable = true;
            unavailable_reported_ = true;
            result.warnings.push_back({"metrics_unavailable",
                                       "warning",
                                       "child_metrics",
                                       "child does not expose /metrics or /slots",
                                       {},
                                       1});
        }
        return result;
    }
    if (slots_response.status < 200 || slots_response.status >= 300) {
        have_progress_ = false;
        stall_reported_ = false;
        restart_issued_ = false;
        if (!scrape_failure_reported_) {
            result.warnings.push_back(
                {"metrics_scrape_failed", "warning", "child_metrics", "GET /slots failed", {}, 1});
            scrape_failure_reported_ = true;
        }
        return result;
    }
    bool slots_valid = false;
    ChildSnapshot snapshot;
    snapshot.port = port;
    snapshot.metrics = counters;
    snapshot.slots = parse_child_slots(slots_response.body, &slots_valid);
    if (!metrics_valid || !slots_valid) {
        have_progress_ = false;
        stall_reported_ = false;
        restart_issued_ = false;
        if (!scrape_failure_reported_) {
            result.warnings.push_back({"metrics_scrape_failed",
                                       "warning",
                                       "child_metrics",
                                       "invalid metrics or slots payload",
                                       {},
                                       1});
            scrape_failure_reported_ = true;
        }
        return result;
    }
    scrape_failure_reported_ = false;
    snapshot.speculation = derive_child_speculation(counters, options_);
    snapshot.sampled_at = now;
    result.snapshot = snapshot;

    const auto processing = counters.requests_processing && *counters.requests_processing > 0;
    const bool has_token_counter = counters.prompt_tokens_total && counters.tokens_predicted_total;
    const bool reset = (counters.prompt_tokens_total && previous_prompt_tokens_ &&
                        *counters.prompt_tokens_total < *previous_prompt_tokens_) ||
                       (counters.tokens_predicted_total && previous_predicted_tokens_ &&
                        *counters.tokens_predicted_total < *previous_predicted_tokens_);
    const bool moved = (counters.prompt_tokens_total && previous_prompt_tokens_ &&
                        *counters.prompt_tokens_total > *previous_prompt_tokens_) ||
                       (counters.tokens_predicted_total && previous_predicted_tokens_ &&
                        *counters.tokens_predicted_total > *previous_predicted_tokens_);
    if (!processing || !has_token_counter) {
        have_progress_ = false;
        stall_reported_ = false;
        restart_issued_ = false;
    } else if (!have_progress_ || moved || reset) {
        have_progress_ = has_token_counter;
        last_progress_ = now;
        stall_reported_ = false;
        restart_issued_ = false;
    } else if (options_.enabled && now - last_progress_ >= options_.stall_seconds) {
        const bool first_stall = !stall_reported_;
        if (first_stall)
            detected_at_ = now;
        stall_reported_ = true;
        const auto since =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - last_progress_).count();
        json::Object stall;
        stall.set(
            "detected_at",
            std::chrono::duration_cast<std::chrono::milliseconds>(detected_at_.time_since_epoch()).count());
        stall.set("since_ms", since);
        result.stall = std::move(stall);
        if (options_.enabled && first_stall) {
            BackendWarning warning{"backend_stalled",
                                   "warning",
                                   "child_metrics",
                                   "child has processing requests but token counters are not moving",
                                   {},
                                   1};
            warning.details = {
                {"processing", std::to_string(*counters.requests_processing)},
                {"deferred", counters.requests_deferred ? std::to_string(*counters.requests_deferred) : ""},
                {"since_ms", std::to_string(since)}};
            result.warnings.push_back(std::move(warning));
        }
        if (options_.enabled && options_.policy == "restart" && !restart_issued_) {
            result.restart_requested = true;
            restart_issued_ = true;
        }
    }
    previous_prompt_tokens_ = counters.prompt_tokens_total;
    previous_predicted_tokens_ = counters.tokens_predicted_total;
    return result;
}

} // namespace sonder::inference::llamaserver
