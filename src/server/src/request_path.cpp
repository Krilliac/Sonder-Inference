#include "request_path.hpp"

#include "thinking_pins.hpp"
#include "reasoning_budget.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace sonder::inference::server::detail {

namespace {

constexpr std::uint64_t kMaxKvPoolTokens = std::uint64_t{1} << 24;

bool effort_ok(const std::string& v) {
    if (v.empty() || v.size() > 64) {
        return false;
    }
    for (const char c : v) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' ||
                        c == '_' || c == '-';
        if (!ok) {
            return false;
        }
    }
    return true;
}

Status invalid(std::string message) { return Status(ErrorCode::invalid_argument, std::move(message)); }

std::optional<std::size_t> nonnegative_size(const std::string& text) {
    if (text.empty() || text.size() > 20 || text.find_first_not_of("0123456789") != std::string::npos) {
        return std::nullopt;
    }
    std::size_t value = 0;
    for (const char ch : text) {
        const auto digit = static_cast<std::size_t>(ch - '0');
        if (value > (std::numeric_limits<std::size_t>::max() - digit) / 10) return std::nullopt;
        value = value * 10 + digit;
    }
    return value;
}

}  // namespace

Status parse_request_path_flags(const FlagLookup& get, ServerOptions& o) {
    if (auto v = get("scheduler")) {
        if (*v == "automatic") {
            o.scheduler = SchedulerPolicy::automatic;
        } else if (*v == "gate") {
            o.scheduler = SchedulerPolicy::gate;
        } else if (*v == "account") {
            o.scheduler = SchedulerPolicy::account;
        } else if (*v == "off") {
            o.scheduler = SchedulerPolicy::off;
        } else {
            return invalid("--scheduler must be automatic, gate, account or off");
        }
    }
    if (auto v = get("kv-pool-tokens")) {
        std::uint64_t n = 0;
        const bool digits = !v->empty() && v->size() <= 9 && v->find_first_not_of("0123456789") == std::string::npos;
        if (digits) {
            for (const char c : *v) n = n * 10 + static_cast<std::uint64_t>(c - '0');
        }
        if (!digits || n < 16 || n > kMaxKvPoolTokens) {
            return invalid("--kv-pool-tokens must be an integer from 16 to " + std::to_string(kMaxKvPoolTokens));
        }
        o.kv_pool_tokens = n;
    }
    if (auto v = get("pin-enable-thinking")) {
        if (*v == "on" || *v == "true") {
            o.pin_enable_thinking = true;
        } else if (*v == "off" || *v == "false") {
            o.pin_enable_thinking = false;
        } else {
            return invalid("--pin-enable-thinking must be on or off");
        }
    }
    if (auto v = get("pin-reasoning-effort")) {
        o.pin_reasoning_effort = *v;
    }
    if (auto v = get("pin-mode")) {
        if (*v == "override") {
            o.pin_mode = PinMode::override_request;
        } else if (*v == "default") {
            o.pin_mode = PinMode::default_value;
        } else {
            return invalid("--pin-mode must be override or default");
        }
    }
    if (auto v = get("pin-reasoning-budget")) {
        const auto value = reasoning_budget::parse_integer(*v);
        if (!value) return invalid("--pin-reasoning-budget must be an integer >= -1 within INT64_MAX");
        o.pin_reasoning_budget_tokens = value;
    }
    if (auto v = get("pin-reasoning-budget-message")) {
        o.pin_reasoning_budget_message = *v;
    }
    for (const auto& entry : {std::pair{"max-concurrent-subagent", &o.max_concurrent_subagent},
                              std::pair{"max-concurrent-background", &o.max_concurrent_background},
                              std::pair{"max-queue-per-class", &o.max_queue_per_class}}) {
        if (auto v = get(entry.first)) {
            const auto n = nonnegative_size(*v);
            if (!n) return invalid(std::string("--") + entry.first + " must be a non-negative integer");
            *entry.second = *n;
        }
    }
    if (auto v = get("backend-capacity")) {
        const auto n = nonnegative_size(*v);
        if (!n) return invalid("--backend-capacity must be a non-negative integer");
        o.backend_capacity = *n;
    }
    if (auto v = get("priority-admission")) {
        if (*v == "auto") {
            o.priority_admission = PriorityAdmissionPolicy::automatic;
        } else if (*v == "on") {
            o.priority_admission = PriorityAdmissionPolicy::on;
        } else if (*v == "off") {
            o.priority_admission = PriorityAdmissionPolicy::off;
        } else {
            return invalid("--priority-admission must be auto, on or off");
        }
    }
    return validate_request_path_options(o);
}

Status validate_request_path_options(const ServerOptions& o) {
    if (o.kv_pool_tokens != 0 && (o.kv_pool_tokens < 16 || o.kv_pool_tokens > kMaxKvPoolTokens)) {
        return invalid("--kv-pool-tokens must be an integer from 16 to " + std::to_string(kMaxKvPoolTokens));
    }
    if (o.pin_reasoning_effort && !effort_ok(*o.pin_reasoning_effort)) {
        return invalid("--pin-reasoning-effort must be 1 to 64 characters of [A-Za-z0-9._-]");
    }
    if (o.pin_reasoning_budget_tokens && *o.pin_reasoning_budget_tokens < -1) {
        return invalid("--pin-reasoning-budget must be an integer >= -1");
    }
    if (o.pin_reasoning_budget_message && o.pin_reasoning_budget_message->size() > 512) {
        return invalid("--pin-reasoning-budget-message must be at most 512 bytes");
    }
    return Status::success();
}

void apply_scheduling(const ServerOptions& o, SchedulingOptions& scheduling) {
    switch (o.scheduler) {
        case SchedulerPolicy::automatic: scheduling.mode = SchedulerMode::automatic; break;
        case SchedulerPolicy::gate: scheduling.mode = SchedulerMode::gate; break;
        case SchedulerPolicy::account: scheduling.mode = SchedulerMode::account; break;
        case SchedulerPolicy::off: scheduling.enabled = false; break;
    }
    if (o.kv_pool_tokens != 0) {
        const std::uint64_t block = scheduling.kv_block_size_tokens == 0 ? 1 : scheduling.kv_block_size_tokens;
        scheduling.kv_num_blocks = static_cast<std::uint32_t>((o.kv_pool_tokens + block - 1) / block);
    }
}

bool priority_admission_enabled(const ServerOptions& o) {
    if (o.priority_admission == PriorityAdmissionPolicy::on) return true;
    if (o.priority_admission == PriorityAdmissionPolicy::off) return false;
    return o.max_concurrent_subagent != 0 || o.max_concurrent_background != 0 || o.max_queue_per_class != 0;
}

std::string chat_session_key(const ChatJob& job, const Correlation& correlation) {
    if (job.session_key) {
        return *job.session_key;
    }
    std::string key;
    if (correlation.run_id) {
        key = "run=" + *correlation.run_id;
    }
    if (correlation.agent_id) {
        key += (key.empty() ? "agent=" : ";agent=") + *correlation.agent_id;
    }
    return key;
}

std::vector<std::string> apply_thinking_pins(const ServerOptions& o, ThinkingOptions& thinking) {
    ThinkingOptions pins;
    pins.enable_thinking = o.pin_enable_thinking;
    pins.reasoning_effort = o.pin_reasoning_effort;
    return thinking_pins::apply(o.pin_mode, thinking, pins);
}

}  // namespace sonder::inference::server::detail
