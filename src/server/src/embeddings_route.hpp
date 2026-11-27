// Internal: POST /v1/embeddings request-input parsing, limits, wire encoding
// and response shaping (docs/SERVER.md). Pure functions, no engine access.
#pragma once

#include <optional>
#include <span>
#include <string>
#include <variant>

#include "openai.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::server::detail {

// Reads the OpenAI `input` member into job.inputs (strings) or
// job.token_inputs (one token array, or an array of token arrays), enforcing
// `limits`. Exactly one of the two job vectors is filled on success.
// Errors are 400 invalid_request with param "input".
[[nodiscard]] std::optional<ApiError> parse_embedding_input(const json::Value* input, const EmbedRouteLimits& limits,
                                                            EmbedJob& job);

// "float" (default) or "base64"; anything else is 400 unsupported_parameter.
[[nodiscard]] std::variant<std::string, ApiError> parse_embedding_encoding_format(const json::Value& value);

// OpenAI base64 wire form: the float32 vector as little-endian bytes, base64.
[[nodiscard]] std::string encode_embedding_base64(std::span<const float> vector);

// json::parse bounds for the request body, sized so that a request carrying
// the maximum number of token ids still parses.
[[nodiscard]] json::ParseLimits embed_parse_limits(const EmbedRouteLimits& limits) noexcept;

// The 200 response document: `data` (float arrays or base64 strings per
// job.encoding_format), `usage` (prompt_tokens == total_tokens) and the
// `sonder` metadata block.
[[nodiscard]] json::Object build_embeddings_response(const EmbedJob& job, const EmbedResult& result,
                                                     const std::string& model, const std::string& backend,
                                                     const std::string& request_id, bool synthetic);

}  // namespace sonder::inference::server::detail
