// Additive tune output fields and strict adaptive speculation config parsing.
// Execution options remain strictly validated by the existing loader; results
// are inert provenance, never launch arguments.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "server/src/env_policy.hpp"
#include "config_text.hpp"
#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference {
inline Status load_llamaserver_extensions(const json::Object &object,
        std::vector<std::pair<std::string, std::string>> &environment) {
    if (const auto *env = object.find("env")) {
        if (!env->is_object() || env->as_object().size() > 128)
            return {ErrorCode::invalid_argument, "llamaserver config: env must be an object with at most 128 entries"};
        std::vector<std::pair<std::string, std::string>> parsed;
        parsed.reserve(env->as_object().size());
        std::size_t bytes = 0;
        for (const auto &[key, value] : env->as_object()) {
            if (!value.is_string() || value.as_string().find('\0') != std::string::npos)
                return {ErrorCode::invalid_argument, "llamaserver config: invalid env entry"};
            bytes += key.size() + value.as_string().size() + 2;
            if (bytes > 32760) return {ErrorCode::invalid_argument, "llamaserver config: env exceeds bounds"};
            parsed.emplace_back(key, value.as_string());
        }
        if (auto st = llamaserver::check_child_environment_overrides(parsed); !st.ok()) {
            return {ErrorCode::invalid_argument, "llamaserver config: " + st.message()};
        }
        environment = std::move(parsed);
    }
    if (const auto *results = object.find("results")) {
        const auto *schema = results->find("schema");
        const auto *mode = object.find("mode");
        if (!results->is_object() || !schema || !schema->is_string() || schema->as_string() != "sonder.inference.tune/1" ||
            !mode || mode->as_string() != "spawn" || !object.contains("executable") || !object.contains("args"))
            return {ErrorCode::invalid_argument, "llamaserver config: results requires a tune/1 spawn configuration"};
    }
    return {};
}

namespace detail {

inline Status parse_adaptive_speculation_config(
    const json::Value &value,
    BackendSetup::LlamaServerAdaptiveSpeculationConfig &out) {
    if (!value.is_object()) {
        return {ErrorCode::invalid_argument, "llamaserver config: 'adaptive_speculation' must be an object"};
    }
    static constexpr std::string_view kKeys[] = {
        "enabled",
        "min_draft_n",
        "initial_draft_n",
        "max_draft_n",
        "window_size",
        "min_samples",
        "pause_acceptance_threshold",
        "step_down_acceptance_threshold",
        "step_up_acceptance_threshold",
        "cooldown_requests",
        "probe_draft_n",
        "min_draft_tokens_per_sample",
    };
    for (const auto &member : value.as_object()) {
        if (std::find(std::begin(kKeys), std::end(kKeys), member.first) == std::end(kKeys)) {
            return {ErrorCode::invalid_argument,
                    "llamaserver config: unknown adaptive_speculation field '" +
                        bounded_for_message(member.first) + "'"};
        }
    }
    const auto bad = [](const char *key, const char *constraint) {
        return Status(ErrorCode::invalid_argument,
                      std::string("llamaserver config: adaptive_speculation.") + key + " must be " + constraint);
    };

    auto parsed = out;
    if (const auto *v = value.find("enabled")) {
        if (!v->is_bool())
            return bad("enabled", "boolean");
        parsed.enabled = v->as_bool();
    }
    const auto uint32_in = [&](const char *key, std::uint32_t lo, std::uint32_t hi, std::uint32_t &dest,
                               const char *range) -> Status {
        if (const auto *v = value.find(key)) {
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() < lo || v->as_uint() > hi)
                return bad(key, range);
            dest = static_cast<std::uint32_t>(v->as_uint());
        }
        return {};
    };
    const auto size_in = [&](const char *key, std::size_t lo, std::size_t hi, std::size_t &dest,
                             const char *range) -> Status {
        if (const auto *v = value.find(key)) {
            if (!v->is_integer() || v->as_int(-1) < 0 || v->as_uint() < lo || v->as_uint() > hi)
                return bad(key, range);
            dest = static_cast<std::size_t>(v->as_uint());
        }
        return {};
    };
    const auto prob_in = [&](const char *key, double &dest) -> Status {
        if (const auto *v = value.find(key)) {
            if (!v->is_number())
                return bad(key, "a finite number in [0, 1]");
            const double d = v->as_double();
            if (!std::isfinite(d) || d < 0.0 || d > 1.0)
                return bad(key, "a finite number in [0, 1]");
            dest = d;
        }
        return {};
    };

    if (auto st = uint32_in("min_draft_n", 0u, 128u, parsed.min_draft_n, "an integer in [0, 128]"); !st.ok())
        return st;
    if (auto st = uint32_in("initial_draft_n", 0u, 128u, parsed.initial_draft_n, "an integer in [0, 128]"); !st.ok())
        return st;
    if (auto st = uint32_in("max_draft_n", 1u, 128u, parsed.max_draft_n, "an integer in [1, 128]"); !st.ok())
        return st;
    if (auto st = size_in("window_size", 1u, 256u, parsed.window_size, "an integer in [1, 256]"); !st.ok())
        return st;
    if (auto st = size_in("min_samples", 1u, 256u, parsed.min_samples, "an integer in [1, 256]"); !st.ok())
        return st;
    if (auto st = prob_in("pause_acceptance_threshold", parsed.pause_acceptance_threshold); !st.ok())
        return st;
    if (auto st = prob_in("step_down_acceptance_threshold", parsed.step_down_acceptance_threshold); !st.ok())
        return st;
    if (auto st = prob_in("step_up_acceptance_threshold", parsed.step_up_acceptance_threshold); !st.ok())
        return st;
    if (auto st = uint32_in("cooldown_requests", 0u, std::numeric_limits<std::uint32_t>::max(),
                            parsed.cooldown_requests, "an integer in [0, 4294967295]");
        !st.ok()) {
        return st;
    }
    if (auto st = uint32_in("probe_draft_n", 1u, 128u, parsed.probe_draft_n, "an integer in [1, 128]"); !st.ok())
        return st;
    if (const auto *v = value.find("min_draft_tokens_per_sample")) {
        if (!v->is_integer() || v->as_double() < 0.0)
            return bad("min_draft_tokens_per_sample", "a non-negative integer");
        parsed.min_draft_tokens_per_sample = v->as_uint();
    }

    if (parsed.min_draft_n > parsed.max_draft_n)
        return bad("min_draft_n", "at most max_draft_n");
    if (parsed.initial_draft_n < parsed.min_draft_n || parsed.initial_draft_n > parsed.max_draft_n)
        return bad("initial_draft_n", "in [min_draft_n, max_draft_n]");
    if (parsed.probe_draft_n == 0 || parsed.probe_draft_n > parsed.max_draft_n)
        return bad("probe_draft_n", "in [1, max_draft_n]");
    if (parsed.min_samples > parsed.window_size)
        return bad("min_samples", "at most window_size");
    if (parsed.pause_acceptance_threshold > parsed.step_down_acceptance_threshold)
        return bad("pause_acceptance_threshold", "at most step_down_acceptance_threshold");
    if (parsed.step_down_acceptance_threshold > parsed.step_up_acceptance_threshold)
        return bad("step_down_acceptance_threshold", "at most step_up_acceptance_threshold");

    out = parsed;
    return {};
}

}  // namespace detail
} // namespace sonder::inference
