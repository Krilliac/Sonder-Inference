#include "request_path.hpp"

#include <cstdint>
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

const char* on_off(bool v) { return v ? "on" : "off"; }

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
    return validate_request_path_options(o);
}

Status validate_request_path_options(const ServerOptions& o) {
    if (o.kv_pool_tokens != 0 && (o.kv_pool_tokens < 16 || o.kv_pool_tokens > kMaxKvPoolTokens)) {
        return invalid("--kv-pool-tokens must be an integer from 16 to " + std::to_string(kMaxKvPoolTokens));
    }
    if (o.pin_reasoning_effort && !effort_ok(*o.pin_reasoning_effort)) {
        return invalid("--pin-reasoning-effort must be 1 to 64 characters of [A-Za-z0-9._-]");
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
    std::vector<std::string> warnings;
    if (o.pin_enable_thinking) {
        if (thinking.enable_thinking && *thinking.enable_thinking != *o.pin_enable_thinking) {
            warnings.push_back(std::string("enable_thinking=") + on_off(*thinking.enable_thinking) +
                               " was overridden by the server pin (" + on_off(*o.pin_enable_thinking) +
                               ") that keeps the prompt prefix cacheable");
        }
        thinking.enable_thinking = o.pin_enable_thinking;
    }
    if (o.pin_reasoning_effort) {
        if (thinking.reasoning_effort && *thinking.reasoning_effort != *o.pin_reasoning_effort) {
            warnings.push_back("reasoning_effort=" + *thinking.reasoning_effort +
                               " was overridden by the server pin (" + *o.pin_reasoning_effort +
                               ") that keeps the prompt prefix cacheable");
        }
        thinking.reasoning_effort = o.pin_reasoning_effort;
    }
    return warnings;
}

}  // namespace sonder::inference::server::detail
