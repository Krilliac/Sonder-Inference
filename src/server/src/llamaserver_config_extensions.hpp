// Additive tune output fields. Execution options remain strictly validated by
// the existing loader; results are inert provenance, never launch arguments.
#pragma once
#include "sonder/inference/json.hpp"
#include <string>
#include <utility>
#include <vector>

namespace sonder::inference {
inline Status load_llamaserver_extensions(const json::Object &object,
        std::vector<std::pair<std::string, std::string>> &environment) {
    if (const auto *env = object.find("env")) {
        if (!env->is_object() || env->as_object().size() > 128)
            return {ErrorCode::invalid_argument, "llamaserver config: env must be an object with at most 128 entries"};
        environment.clear();
        std::size_t bytes = 0;
        for (const auto &[key, value] : env->as_object()) {
            if (key.empty() || key.find('=') != std::string::npos || key.find('\0') != std::string::npos ||
                !value.is_string() || value.as_string().find('\0') != std::string::npos)
                return {ErrorCode::invalid_argument, "llamaserver config: invalid env entry"};
            bytes += key.size() + value.as_string().size() + 2;
            if (bytes > 32760) return {ErrorCode::invalid_argument, "llamaserver config: env exceeds bounds"};
            environment.emplace_back(key, value.as_string());
        }
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
} // namespace sonder::inference
