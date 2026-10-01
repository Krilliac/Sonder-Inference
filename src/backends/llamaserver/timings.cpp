#include "timings.hpp"

#include <cmath>
#include <limits>

namespace sonder::inference::llamaserver {
namespace {

const json::Value *timings_object(const json::Value &value, Status &status) {
    if (!value.is_object()) {
        status = Status(ErrorCode::protocol_error, "llamaserver: response must be an object");
        return nullptr;
    }
    const auto *timings = value.find("timings");
    if (!timings || timings->is_null())
        return &value;
    if (!timings->is_object()) {
        status = Status(ErrorCode::protocol_error, "llamaserver: timings must be an object");
        return nullptr;
    }
    return timings;
}

bool read_count(const json::Value *object, const char *key, std::optional<std::uint64_t> &out) {
    const auto *value = object->find(key);
    if (!value)
        return true;
    if (!value->is_integer() || value->as_double() < 0)
        return false;
    out = value->as_uint();
    return true;
}

bool read_ms(const json::Value *object, const char *key, std::optional<double> &out) {
    const auto *value = object->find(key);
    if (!value)
        return true;
    if (!value->is_number() || !std::isfinite(value->as_double()) || value->as_double() < 0)
        return false;
    out = value->as_double();
    return true;
}

} // namespace

Status parse_timings(const json::Value &value, BackendTimings &out) {
    out = {};
    Status status = Status::success();
    const auto *object = timings_object(value, status);
    if (!object)
        return status;
    if (!read_count(object, "prompt_n", out.prompt_n) || !read_count(object, "cache_n", out.cache_n) ||
        !read_count(object, "predicted_n", out.predicted_n) || !read_count(object, "draft_n", out.draft_n) ||
        !read_count(object, "draft_n_accepted", out.draft_n_accepted) ||
        !read_ms(object, "prompt_ms", out.prompt_ms) || !read_ms(object, "predicted_ms", out.predicted_ms)) {
        out = {};
        return Status(ErrorCode::protocol_error, "llamaserver: invalid timing value");
    }
    if (out.draft_n && out.draft_n_accepted && *out.draft_n_accepted > *out.draft_n) {
        out = {};
        return Status(ErrorCode::protocol_error, "llamaserver: inconsistent timing token counts");
    }
    return Status::success();
}

void merge_timings(std::optional<BackendTimings> &target, const std::optional<BackendTimings> &source) {
    if (!source) return;
    if (!target) target.emplace();
    if (source->prompt_n) target->prompt_n = source->prompt_n;
    if (source->cache_n) target->cache_n = source->cache_n;
    if (source->prompt_ms) target->prompt_ms = source->prompt_ms;
    if (source->predicted_n) target->predicted_n = source->predicted_n;
    if (source->predicted_ms) target->predicted_ms = source->predicted_ms;
    if (source->draft_n) target->draft_n = source->draft_n;
    if (source->draft_n_accepted) target->draft_n_accepted = source->draft_n_accepted;
}

} // namespace sonder::inference::llamaserver
