#include "reasoning_budget.hpp"

#include <charconv>
#include <limits>
#include <system_error>

namespace sonder::inference::server::detail::reasoning_budget {

std::optional<std::int64_t> parse_integer(std::string_view text) noexcept {
    if (text.empty()) return std::nullopt;
    std::int64_t value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size() || value < -1) return std::nullopt;
    return value;
}

std::optional<ApiError> parse_body(const json::Object& body, ChatJob& job) {
    // Validate both spellings, then prefer the canonical name.
    for (const auto* field : {"thinking_budget_tokens", "reasoning_budget_tokens"}) {
        if (const auto* value = body.find(field)) {
            const auto n = value->as_int(std::numeric_limits<std::int64_t>::min());
            if (!value->is_integer() || n < -1) {
                return make_error(400, "invalid_json", std::string(field) + " must be an integer from -1 to INT64_MAX", field);
            }
            job.reasoning_budget_tokens = n;
        }
    }
    if (const auto* value = body.find("reasoning_budget_message")) {
        if (!value->is_string() || value->as_string().size() > 512) {
            return make_error(400, "invalid_json", "reasoning_budget_message must be a string of at most 512 bytes",
                              "reasoning_budget_message");
        }
        job.reasoning_budget_message = value->as_string();
    }
    return std::nullopt;
}

std::optional<ApiError> parse_header(const RequestHead& head, Correlation& correlation) {
    constexpr auto name = "X-Sonder-Reasoning-Budget";
    std::size_t count = 0;
    for (const auto& field : head.headers) if (field.first == "x-sonder-reasoning-budget") ++count;
    if (count > 1) return make_error(400, "invalid_correlation_header", std::string(name) + " must occur once", name);
    if (const auto* text = head.header("x-sonder-reasoning-budget")) {
        correlation.reasoning_budget_tokens = parse_integer(*text);
        if (!correlation.reasoning_budget_tokens) {
            return make_error(400, "invalid_correlation_header", std::string(name) + " must be an integer from -1 to INT64_MAX", name);
        }
    }
    return std::nullopt;
}

void apply(const ChatJob& job, const Correlation& correlation, const ServerOptions& pins,
           std::string_view backend, RequestOptions& request, std::vector<std::string>& warnings) {
    request.reasoning_budget_tokens = correlation.reasoning_budget_tokens ? correlation.reasoning_budget_tokens
        : job.reasoning_budget_tokens ? job.reasoning_budget_tokens : pins.pin_reasoning_budget_tokens;
    request.reasoning_budget_message = job.reasoning_budget_message ? job.reasoning_budget_message
                                                                  : pins.pin_reasoning_budget_message;
    if (backend != "llamaserver") {
        if (request.reasoning_budget_tokens) warnings.emplace_back("reasoning_budget_tokens is ignored by backend " + std::string(backend));
        if (request.reasoning_budget_message) warnings.emplace_back("reasoning_budget_message is ignored by backend " + std::string(backend));
        request.reasoning_budget_tokens.reset();
        request.reasoning_budget_message.reset();
    }
}

}  // namespace sonder::inference::server::detail::reasoning_budget
