#pragma once

#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::llamaserver {

// Parse the optional llama-server `timings` object from either a streaming
// chunk or a non-streaming response. Missing members are left unset.
Status parse_timings(const json::Value &value, BackendTimings &out);
// Retain earlier measured fields across partial/final streaming frames.
void merge_timings(std::optional<BackendTimings> &target, const std::optional<BackendTimings> &source);

} // namespace sonder::inference::llamaserver
