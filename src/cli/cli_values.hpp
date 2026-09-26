// sonder-infer CLI helpers: checked option values (numbers, correlation ids,
// workload classes, --stats modes), the documented exit codes and the
// request stats line. Header-only; shared by tools/sonder-infer,
// bench/tools/sonder_bench.cpp and tests/test_cli_spec.cpp. Reference:
// docs/CLI.md.
#pragma once

#include <cctype>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/session.hpp"

namespace sonder::cli {

namespace si = sonder::inference;

// Exit codes (docs/CLI.md "Exit codes").
inline constexpr int kExitOk = 0;
inline constexpr int kExitFailure = 1;    // runtime or backend failure
inline constexpr int kExitUsage = 2;      // bad flags or values, unknown backend, unreadable input file
inline constexpr int kExitCancelled = 130;  // request cancelled (Ctrl-C)

// Exit code for a failed operation: cancelled -> 130, otherwise 1.
inline int exit_code_for(const si::Status& status) noexcept {
    if (status.ok()) return kExitOk;
    return status.code() == si::ErrorCode::cancelled ? kExitCancelled : kExitFailure;
}

// Exit code for a finished request.
inline int exit_code_for(si::RequestOutcome outcome) noexcept {
    switch (outcome) {
        case si::RequestOutcome::completed:
            return kExitOk;
        case si::RequestOutcome::cancelled:
            return kExitCancelled;
        case si::RequestOutcome::none:
        case si::RequestOutcome::failed:
            break;
    }
    return kExitFailure;
}

namespace detail {
inline std::string range_text(long long min, long long max) {
    return "an integer from " + std::to_string(min) + " to " + std::to_string(max);
}
inline std::string invalid_value(std::string_view flag, std::string_view text, const std::string& expected) {
    return "invalid value '" + std::string(text) + "' for --" + std::string(flag) + " (expected " + expected + ")";
}

// Plain decimal number syntax: -?(digits[.digits*] | .digits)([eE][+-]?digits)?
// No leading '+', no whitespace, no hex floats, no inf/nan.
inline bool is_decimal_number(std::string_view t) noexcept {
    std::size_t i = 0;
    const auto digits = [&] {
        const std::size_t start = i;
        while (i < t.size() && t[i] >= '0' && t[i] <= '9') ++i;
        return i - start;
    };
    if (i < t.size() && t[i] == '-') ++i;
    std::size_t mantissa = digits();
    if (i < t.size() && t[i] == '.') {
        ++i;
        mantissa += digits();
    }
    if (mantissa == 0) return false;
    if (i < t.size() && (t[i] == 'e' || t[i] == 'E')) {
        ++i;
        if (i < t.size() && (t[i] == '+' || t[i] == '-')) ++i;
        if (digits() == 0) return false;
    }
    return i == t.size();
}
}  // namespace detail

// Parses a decimal integer in [min, max]. The whole text must be consumed;
// no sign prefix other than '-', no whitespace. Errors name the flag and the
// bad value, e.g. "invalid value 'abc' for --runs (expected an integer from
// 1 to 1000)".
template <class Int>
bool parse_integer(std::string_view flag, std::string_view text, Int min, Int max, Int& out, std::string& error) {
    static_assert(std::is_integral_v<Int>, "integral type required");
    using Wide = std::conditional_t<std::is_signed_v<Int>, long long, unsigned long long>;
    Wide value{};
    const char* first = text.data();
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(first, last, value);
    const std::string expected = detail::range_text(static_cast<long long>(min), static_cast<long long>(max));
    if (text.empty() || ec == std::errc::invalid_argument || ptr != last) {
        error = detail::invalid_value(flag, text, expected);
        return false;
    }
    if (ec == std::errc::result_out_of_range || value < static_cast<Wide>(min) || value > static_cast<Wide>(max)) {
        error = "value '" + std::string(text) + "' for --" + std::string(flag) + " is out of range (expected " +
                expected + ")";
        return false;
    }
    out = static_cast<Int>(value);
    return true;
}

// Unsigned 64-bit variant (e.g. --seed), full range.
inline bool parse_u64(std::string_view flag, std::string_view text, std::uint64_t& out, std::string& error) {
    unsigned long long value = 0;
    const char* last = text.data() + text.size();
    const auto [ptr, ec] = std::from_chars(text.data(), last, value);
    if (text.empty() || ec != std::errc() || ptr != last) {
        error = detail::invalid_value(flag, text, "an integer from 0 to 18446744073709551615");
        return false;
    }
    out = static_cast<std::uint64_t>(value);
    return true;
}

// Parses a finite decimal number in [min, max] ("C" locale). Only plain
// decimal syntax is accepted (detail::is_decimal_number): no empty text,
// whitespace, trailing characters, leading '+', hex floats, inf or nan.
inline bool parse_number(std::string_view flag, std::string_view text, double min, double max, double& out,
                         std::string& error) {
    const std::string s(text);
    const auto expected = [&] {
        if (std::isinf(min) && std::isinf(max)) return std::string("a number");
        std::ostringstream os;
        os << "a number from " << min << " to " << max;
        return os.str();
    };
    if (!detail::is_decimal_number(s)) {
        error = detail::invalid_value(flag, text, expected());
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end != s.c_str() + s.size() || !std::isfinite(v) || (errno == ERANGE && std::fabs(v) > 1.0)) {
        error = detail::invalid_value(flag, text, expected());
        return false;
    }
    if (v < min || v > max) {
        error = "value '" + s + "' for --" + std::string(flag) + " is out of range (expected " + expected() + ")";
        return false;
    }
    out = v;
    return true;
}

// Float variant: any finite number that fits a float (range checks are left
// to si::validate(), which names the valid range).
inline bool parse_float(std::string_view flag, std::string_view text, float& out, std::string& error) {
    constexpr double inf = std::numeric_limits<double>::infinity();
    double v = 0.0;
    if (!parse_number(flag, text, -inf, inf, v, error)) return false;
    if (std::fabs(v) > static_cast<double>(std::numeric_limits<float>::max())) {
        error = "value '" + std::string(text) + "' for --" + std::string(flag) + " is out of range (too large)";
        return false;
    }
    out = static_cast<float>(v);
    return true;
}

// Correlation ids (--run-id, --agent-id, --task-id): the same format the
// serve API accepts in X-Sonder-* headers, [A-Za-z0-9._:-]{1,128}.
inline bool is_valid_correlation_id(std::string_view id) noexcept {
    if (id.empty() || id.size() > 128) return false;
    for (const char c : id) {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == ':' || c == '-';
        if (!ok) return false;
    }
    return true;
}

inline bool parse_correlation_id(std::string_view flag, std::string_view text, std::string& out, std::string& error) {
    if (!is_valid_correlation_id(text)) {
        error = detail::invalid_value(flag, text, "1 to 128 characters from A-Z a-z 0-9 . _ : -");
        return false;
    }
    out = std::string(text);
    return true;
}

// The seven scheduler workload classes (docs/SCHEDULER.md), by name.
inline std::optional<si::WorkloadClass> parse_workload(std::string_view text) noexcept {
    for (int i = 0; i <= static_cast<int>(si::WorkloadClass::maintenance); ++i) {
        const auto w = static_cast<si::WorkloadClass>(i);
        if (text == si::to_string(w)) return w;
    }
    return std::nullopt;
}

inline std::string workload_names() {
    std::string out;
    for (int i = 0; i <= static_cast<int>(si::WorkloadClass::maintenance); ++i) {
        out += (i ? ", " : "") + std::string(si::to_string(static_cast<si::WorkloadClass>(i)));
    }
    return out;
}

// --priority range (the scheduler's per-request rank adjustment).
inline constexpr int kMinPriority = -16;
inline constexpr int kMaxPriority = 16;

// --stats text|json|none.
enum class StatsMode { text, json, none };

inline std::optional<StatsMode> parse_stats_mode(std::string_view text) noexcept {
    if (text == "text") return StatsMode::text;
    if (text == "json") return StatsMode::json;
    if (text == "none") return StatsMode::none;
    return std::nullopt;
}

// One request's statistics, printed after generate and chat turns.
struct RequestStats {
    std::string command;  // "generate" or "chat"
    std::string backend;
    std::string model;
    bool synthetic = false;  // the MOCK backend produced it
    std::string request_id;
    std::string outcome;      // completed | cancelled | failed
    std::string stop_reason;  // StopReason name
    std::uint64_t prompt_tokens = 0;
    std::uint64_t completion_tokens = 0;
    bool token_counts_from_backend = false;
    double ttft_ms = -1.0;  // -1 when no chunk arrived
    double total_ms = 0.0;
    // chat only
    std::optional<bool> native_chat;
    std::optional<std::size_t> messages;
    std::optional<std::uint64_t> turn;
};

inline RequestStats make_request_stats(std::string command, const std::string& backend, const std::string& model,
                                       bool synthetic, const si::GenerationResult& r) {
    RequestStats s;
    s.command = std::move(command);
    s.backend = backend;
    s.model = model;
    s.synthetic = synthetic;
    s.request_id = r.request_id;
    s.outcome = si::to_string(r.outcome);
    s.stop_reason = si::to_string(r.stats.stop_reason);
    s.prompt_tokens = r.stats.prompt_tokens;
    s.completion_tokens = r.stats.completion_tokens;
    s.token_counts_from_backend = r.stats.token_counts_from_backend;
    s.ttft_ms = r.ttft_ms;
    s.total_ms = r.total_ms;
    return s;
}

// Human line, e.g.
// "[sonder-infer] chat native=no messages=4 stop=max_tokens prompt_tokens=9 ..."
inline std::string format_stats_text(const RequestStats& s) {
    std::ostringstream os;
    os << "[sonder-infer] ";
    if (s.command == "chat") {
        os << "chat";
        if (s.turn) os << " turn=" << *s.turn;
        if (s.native_chat) os << " native=" << (*s.native_chat ? "yes" : "no");
        if (s.messages) os << " messages=" << *s.messages;
        os << " outcome=" << s.outcome;
    } else {
        os << "outcome=" << s.outcome;
    }
    os << " stop=" << s.stop_reason << " prompt_tokens=" << s.prompt_tokens
       << " completion_tokens=" << s.completion_tokens << " ttft_ms=" << s.ttft_ms << " total_ms=" << s.total_ms;
    if (!s.request_id.empty()) os << " request_id=" << s.request_id;
    return os.str();
}

// One JSON object on one line (docs/CLI.md "Stats line").
inline std::string format_stats_json(const RequestStats& s) {
    si::json::Object o{{"command", s.command},
                       {"backend", s.backend},
                       {"model", s.model},
                       {"synthetic", s.synthetic},
                       {"request_id", s.request_id},
                       {"outcome", s.outcome},
                       {"stop_reason", s.stop_reason},
                       {"prompt_tokens", s.prompt_tokens},
                       {"completion_tokens", s.completion_tokens},
                       {"token_counts_from_backend", s.token_counts_from_backend},
                       {"ttft_ms", s.ttft_ms < 0 ? si::json::Value(nullptr) : si::json::Value(s.ttft_ms)},
                       {"total_ms", s.total_ms}};
    if (s.native_chat) o.set("native_chat", *s.native_chat);
    if (s.messages) o.set("messages", *s.messages);
    if (s.turn) o.set("turn", *s.turn);
    return si::json::Value(std::move(o)).dump();
}

}  // namespace sonder::cli
