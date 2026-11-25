// Internal: bounded SSE/NDJSON stream framing, HTTP error mapping, and
// completion/chat frame decoding for the llamaserver backend.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "net/http_client.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference::llamaserver {

// Default cumulative byte cap on streamed tool-call delta payloads
// (id + name + arguments_delta) across a single completion stream (M-16).
inline constexpr std::size_t kMaxToolCallDeltaBytes = 1024u * 1024u;
// Default cumulative byte cap on total streamed text + reasoning + tool-call
// delta bytes across a single completion stream (M-16).
inline constexpr std::size_t kMaxStreamOutputBytes = 32u << 20;

Status protocol_error(std::string message);
Status status_for_http(int code);

// The idle, first-content and stall bounds raised by the HTTP client end the
// request with a protocol_error and get the backend prefix; every other client
// status (including the total timeout) passes through unchanged.
Status upstream_status(const Status &status);

// Executes a prepared POST completion/chat request and decodes its bounded
// SSE or NDJSON stream into `GenerateStats` and `callback` invocations,
// incrementing `progress_events` on frames that carry content, reasoning, or
// tool-call deltas (when `chat && tools_requested`).
Result<GenerateStats> run_completion_stream(net::HttpRequest &request, bool chat, bool native, bool separate_reasoning,
                                            const CancellationToken &cancel, const TokenCallback &callback,
                                            std::uint64_t &progress_events, bool tools_requested = false,
                                            std::size_t max_tool_call_bytes = kMaxToolCallDeltaBytes);

}  // namespace sonder::inference::llamaserver
