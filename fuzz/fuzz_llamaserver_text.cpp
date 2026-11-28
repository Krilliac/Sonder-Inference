// libFuzzer target: the text the llama-server backend reads from a supervised
// child: Prometheus /metrics (parse_child_metrics), /slots (parse_child_slots),
// the child log (LogDiagnostics::feed/finish), plus parse_spill_policy and
// map_finish_reason.
//
// Input layout: byte 0 = control (seed for the log chunk-split pattern); the
// rest is the text given to every parser. Seeds use printable control bytes.
//
// Invariants:
//   * nothing crashes, hangs, or trips ASan/UBSan on arbitrary bytes.
//   * metrics: gauges are finite and non-negative whether or not the payload
//     was accepted; the *valid out-parameter agrees with
//     ChildMetricCounters::valid; derived speculation values are finite and
//     non-negative; an invalid payload yields an empty (default) result.
//   * slots: a valid result has at most 1024 entries; an invalid one is empty.
//   * LogDiagnostics: feeding the text in a fuzz-chosen split then finish()
//     gives exactly the same warnings (code, severity, source, message,
//     details, count), line count and offload status as one feed() call;
//     warnings never exceed kMaxWarnings and counts are at least 1.
//   * parse_spill_policy accepts exactly the names to_string() emits.
//   * map_finish_reason matches the documented table.
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "backends/llamaserver/child_metrics.hpp"
#include "backends/llamaserver/spill_guard.hpp"
#include "backends/llamaserver/llamaserver_protocol.hpp"
#include "devices/log_diagnostics.hpp"
#include "fuzz_common.hpp"

namespace ls = sonder::inference::llamaserver;
namespace si = sonder::inference;

namespace {

bool sane(const std::optional<double>& v) { return !v || (std::isfinite(*v) && *v >= 0.0); }

bool metrics_empty(const ls::ChildMetricCounters& m) {
    return !m.prompt_tokens_seconds && !m.predicted_tokens_seconds && !m.prompt_tokens_total &&
           !m.tokens_predicted_total && !m.requests_processing && !m.requests_deferred &&
           !m.n_busy_slots_per_decode && !m.n_decode_total && !m.spec_decode_num_draft_tokens_total &&
           !m.spec_decode_accepted_tokens_total && !m.spec_decode_drafts_total &&
           m.spec_decode_accepted_tokens_per_position.empty() && !m.kv_cache_usage_ratio;
}

void check_metrics(std::string_view text) {
    bool valid_out = true;
    const ls::ChildMetricCounters m = ls::parse_child_metrics(text, &valid_out);
    SONDER_FUZZ_CHECK(valid_out == m.valid);
    SONDER_FUZZ_CHECK(sane(m.prompt_tokens_seconds) && sane(m.predicted_tokens_seconds) &&
                      sane(m.n_busy_slots_per_decode) && sane(m.kv_cache_usage_ratio));
    SONDER_FUZZ_CHECK(m.spec_decode_accepted_tokens_per_position.size() <= 256);
    if (!m.valid) {
        SONDER_FUZZ_CHECK(metrics_empty(m));
    } else {
        SONDER_FUZZ_CHECK(!metrics_empty(m) || text.find("llamacpp:") != std::string_view::npos);
    }
    ls::ChildMetricsOptions options;
    options.n_max = 4;
    const ls::ChildMetricDerived d = ls::derive_child_speculation(m, options);
    SONDER_FUZZ_CHECK(sane(d.mean_accepted_len) && sane(d.speedup_est));
    SONDER_FUZZ_CHECK(d.acceptance_by_position.size() == m.spec_decode_accepted_tokens_per_position.size());
    for (const auto& a : d.acceptance_by_position) SONDER_FUZZ_CHECK(sane(a));
}

void check_slots(std::string_view text) {
    bool valid = true;
    const auto slots = ls::parse_child_slots(text, &valid);
    if (valid) {
        SONDER_FUZZ_CHECK(slots.size() <= 1024);
    } else {
        SONDER_FUZZ_CHECK(slots.empty());
    }
}

struct LogOutcome {
    std::vector<si::BackendWarning> warnings;
    std::uint64_t lines = 0;
    std::string offload;
};

LogOutcome run_log(std::string_view text, std::uint32_t seed, bool whole) {
    ls::LogDiagnostics d;
    if (whole) {
        d.feed(text);
    } else {
        std::uint32_t state = seed * 2654435761u + 1u;
        while (!text.empty()) {
            state = state * 1103515245u + 12345u;
            std::size_t n = 1 + ((state >> 16) % 61u);
            if (n > text.size()) n = text.size();
            d.feed(text.substr(0, n));
            text.remove_prefix(n);
        }
    }
    d.finish();
    return {d.warnings(), d.lines(), d.offload_status()};
}

bool same(const si::BackendWarning& a, const si::BackendWarning& b) {
    return a.code == b.code && a.severity == b.severity && a.source == b.source && a.message == b.message &&
           a.details == b.details && a.count == b.count;
}

void check_log(std::string_view text, std::uint32_t seed) {
    const LogOutcome whole = run_log(text, seed, true);
    const LogOutcome parts = run_log(text, seed, false);
    SONDER_FUZZ_CHECK(whole.warnings.size() == parts.warnings.size());
    for (std::size_t i = 0; i < whole.warnings.size(); ++i) {
        SONDER_FUZZ_CHECK(same(whole.warnings[i], parts.warnings[i]));
        SONDER_FUZZ_CHECK(whole.warnings[i].count >= 1);
    }
    SONDER_FUZZ_CHECK(whole.warnings.size() <= ls::LogDiagnostics::kMaxWarnings);
    SONDER_FUZZ_CHECK(whole.lines == parts.lines);
    SONDER_FUZZ_CHECK(whole.offload == parts.offload);
}

void check_names(std::string_view text) {
    const auto policy = ls::parse_spill_policy(text);
    const bool known = text == "warn" || text == "refuse" || text == "auto_fit";
    SONDER_FUZZ_CHECK(policy.has_value() == known);
    if (policy) SONDER_FUZZ_CHECK(text == ls::to_string(*policy));
    for (const auto p : {ls::SpillPolicy::warn, ls::SpillPolicy::refuse, ls::SpillPolicy::auto_fit}) {
        SONDER_FUZZ_CHECK(ls::parse_spill_policy(ls::to_string(p)) == p);
    }

    const si::StopReason r = ls::map_finish_reason(text);
    si::StopReason want = si::StopReason::none;
    if (text == "length" || text == "limit") want = si::StopReason::max_tokens;
    else if (text == "word") want = si::StopReason::stop_sequence;
    else if (text == "stop" || text == "eos" || text == "end_of_sequence") want = si::StopReason::end_of_sequence;
    SONDER_FUZZ_CHECK(r == want);
    SONDER_FUZZ_CHECK(si::to_string(r) != nullptr);
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    if (size == 0) return 0;
    const std::uint32_t seed = data[0];
    const std::string_view text = sonder_fuzz::as_view(data + 1, size - 1);
    check_metrics(text);
    check_slots(text);
    check_log(text, seed);
    check_names(text);
    check_names(text.substr(0, text.find('\n')));
    return 0;
}
