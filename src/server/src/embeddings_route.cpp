#include "embeddings_route.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>

#include "sonder/inference/server.hpp"
#include "tool_call_parser.hpp"

namespace sonder::inference::server::detail {

namespace {

ApiError bad_input(std::string message) {
    return make_error(400, "invalid_request", std::move(message), "input");
}

bool is_token_id(const json::Value& v, TokenId& out) {
    if (!v.is_integer()) return false;
    const auto n = v.try_as_int();
    if (!n || *n < 0 || *n > std::numeric_limits<TokenId>::max()) return false;
    out = static_cast<TokenId>(*n);
    return true;
}

}  // namespace

std::optional<ApiError> parse_embedding_input(const json::Value* input, const EmbedRouteLimits& limits, EmbedJob& job) {
    job.inputs.clear();
    job.token_inputs.clear();
    if (input == nullptr || input->is_null()) {
        return bad_input("input is required (string, array of strings, token array, or array of token arrays)");
    }
    const auto too_many = [&] {
        return bad_input("input array exceeds maximum of " + std::to_string(limits.max_inputs) + " items");
    };
    const auto text_too_long = [&](std::size_t index) {
        return bad_input("input[" + std::to_string(index) + "] exceeds maximum length of " +
                         std::to_string(limits.max_input_bytes) + " bytes");
    };
    const auto total_text_too_long = [&] {
        return bad_input("input exceeds maximum total length of " + std::to_string(limits.max_total_bytes) + " bytes");
    };
    const auto tokens_too_long = [&](std::size_t index) {
        return bad_input("input[" + std::to_string(index) + "] exceeds maximum length of " +
                         std::to_string(limits.max_input_tokens) + " tokens");
    };
    const auto total_tokens_too_long = [&] {
        return bad_input("input exceeds maximum total length of " + std::to_string(limits.max_total_tokens) +
                         " tokens");
    };

    if (input->is_string()) {
        const std::string& text = input->as_string();
        if (text.size() > limits.max_input_bytes) return bad_input("input exceeds maximum length of " +
                                                                  std::to_string(limits.max_input_bytes) + " bytes");
        if (text.size() > limits.max_total_bytes) return total_text_too_long();
        job.inputs.push_back(text);
        return std::nullopt;
    }
    if (!input->is_array()) {
        return bad_input("input must be a string, an array of strings, a token array, or an array of token arrays");
    }
    const auto& arr = input->as_array();
    if (arr.empty()) return bad_input("input array must not be empty");

    // A flat array of integers is ONE pre-tokenized input (OpenAI semantics).
    if (arr.front().is_integer()) {
        std::vector<TokenId> tokens;
        if (arr.size() > limits.max_input_tokens) return tokens_too_long(0);
        if (arr.size() > limits.max_total_tokens) return total_tokens_too_long();
        tokens.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            TokenId id = 0;
            if (!is_token_id(arr[i], id)) {
                return bad_input("input[" + std::to_string(i) + "] must be an integer token id between 0 and " +
                                 std::to_string(std::numeric_limits<TokenId>::max()));
            }
            tokens.push_back(id);
        }
        job.token_inputs.push_back(std::move(tokens));
        return std::nullopt;
    }

    if (arr.size() > limits.max_inputs) return too_many();

    if (arr.front().is_string()) {
        std::size_t total = 0;
        job.inputs.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_string()) {
                return bad_input("input[" + std::to_string(i) +
                                 "] must be a string (an array cannot mix strings and token arrays)");
            }
            const std::string& text = arr[i].as_string();
            if (text.size() > limits.max_input_bytes) return text_too_long(i);
            if (text.size() > limits.max_total_bytes - total) return total_text_too_long();
            total += text.size();
            job.inputs.push_back(text);
        }
        return std::nullopt;
    }

    if (arr.front().is_array()) {
        std::size_t total = 0;
        job.token_inputs.reserve(arr.size());
        for (std::size_t i = 0; i < arr.size(); ++i) {
            if (!arr[i].is_array()) {
                return bad_input("input[" + std::to_string(i) +
                                 "] must be a token array (an array cannot mix strings and token arrays)");
            }
            const auto& inner = arr[i].as_array();
            if (inner.empty()) return bad_input("input[" + std::to_string(i) + "] must not be empty");
            if (inner.size() > limits.max_input_tokens) return tokens_too_long(i);
            if (inner.size() > limits.max_total_tokens - total) return total_tokens_too_long();
            total += inner.size();
            std::vector<TokenId> tokens;
            tokens.reserve(inner.size());
            for (std::size_t j = 0; j < inner.size(); ++j) {
                TokenId id = 0;
                if (!is_token_id(inner[j], id)) {
                    return bad_input("input[" + std::to_string(i) + "][" + std::to_string(j) +
                                     "] must be an integer token id between 0 and " +
                                     std::to_string(std::numeric_limits<TokenId>::max()));
                }
                tokens.push_back(id);
            }
            job.token_inputs.push_back(std::move(tokens));
        }
        return std::nullopt;
    }
    return bad_input("input must be a string, an array of strings, a token array, or an array of token arrays");
}

std::variant<std::string, ApiError> parse_embedding_encoding_format(const json::Value& value) {
    if (value.is_string()) {
        const std::string& s = value.as_string();
        if (s == "float" || s == "base64") return s;
    }
    return make_error(400, "unsupported_parameter", "encoding_format must be 'float' or 'base64'",
                      "encoding_format");
}

std::string encode_embedding_base64(std::span<const float> vector) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(vector.size() * 4u);
    for (const float f : vector) {
        std::uint32_t bits = 0;
        static_assert(sizeof(bits) == sizeof(f));
        std::memcpy(&bits, &f, sizeof bits);
        bytes.push_back(static_cast<std::uint8_t>(bits & 0xFFu));
        bytes.push_back(static_cast<std::uint8_t>((bits >> 8) & 0xFFu));
        bytes.push_back(static_cast<std::uint8_t>((bits >> 16) & 0xFFu));
        bytes.push_back(static_cast<std::uint8_t>((bits >> 24) & 0xFFu));
    }
    return encode_base64(bytes);
}

json::ParseLimits embed_parse_limits(const EmbedRouteLimits& limits) noexcept {
    // Value nodes: every token id is one, plus an array node per input and slack for the
    // envelope. Never below the historical 65536 bound.
    constexpr std::size_t kSlack = 4096;
    const auto sat_add = [](std::size_t a, std::size_t b) noexcept {
        return a > std::numeric_limits<std::size_t>::max() - b ? std::numeric_limits<std::size_t>::max() : a + b;
    };
    const std::size_t token_nodes = sat_add(sat_add(limits.max_total_tokens, limits.max_inputs), kSlack);
    return json::ParseLimits{128, std::max<std::size_t>(65536u, token_nodes)};
}

json::Object build_embeddings_response(const EmbedJob& job, const EmbedResult& result, const std::string& model,
                                       const std::string& backend, const std::string& request_id, bool synthetic) {
    // The wire form is the caller's choice; a backend's echo of it never overrides that.
    const bool as_base64 = job.encoding_format == "base64";
    json::Array data;
    data.reserve(result.embeddings.size());
    for (std::size_t i = 0; i < result.embeddings.size(); ++i) {
        json::Value embedding = nullptr;
        if (as_base64) {
            embedding = encode_embedding_base64(result.embeddings[i]);
        } else {
            json::Array vec;
            vec.reserve(result.embeddings[i].size());
            for (const float v : result.embeddings[i]) vec.emplace_back(static_cast<double>(v));
            embedding = std::move(vec);
        }
        data.emplace_back(json::Object{{"object", "embedding"},
                                       {"index", static_cast<std::int64_t>(i)},
                                       {"embedding", std::move(embedding)}});
    }
    const std::size_t effective_dims =
        result.dimensions > 0 ? result.dimensions : (!result.embeddings.empty() ? result.embeddings.front().size() : 0u);
    json::Object meta{{"api_version", kApiVersion},
                      {"request_id", request_id},
                      {"backend", backend},
                      {"model", model},
                      {"synthetic", synthetic},
                      {"dimensions", static_cast<std::uint64_t>(effective_dims)},
                      {"pooling_type", to_string(result.pooling_type)},
                      {"encoding_format", as_base64 ? "base64" : "float"}};
    return json::Object{{"object", "list"},
                        {"data", std::move(data)},
                        {"model", model},
                        {"usage", json::Object{{"prompt_tokens", result.prompt_tokens},
                                               {"total_tokens", result.prompt_tokens}}},
                        {"sonder", std::move(meta)}};
}

}  // namespace sonder::inference::server::detail
