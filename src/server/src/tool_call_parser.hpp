// Internal: OpenAI & Anthropic tool calling parsers, incremental streaming
// tool-call tag parser, tool-call GBNF generator, multimodal base64 data-URI
// validator (zero-SSRF), and logprob JSON formatters. Pure functions.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "openai.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::server::detail {

// Strict multimodal security limits (base64 data URIs only — zero SSRF).
inline constexpr std::size_t kMaxImageDecodedBytes = 16u * 1024u * 1024u;
inline constexpr std::size_t kMaxImagesPerRequest = 8u;
inline constexpr std::size_t kMaxToolsPerRequest = 128u;
inline constexpr std::size_t kMaxToolCallPayloadBytes = 256u * 1024u;

// True for image/png, image/jpeg, image/webp, image/gif.
[[nodiscard]] bool is_supported_image_mime(std::string_view mime) noexcept;

// Strict RFC 4648 base64 decoder (no whitespace/invalid chars, valid padding,
// non-empty, decoded length <= max_decoded_bytes).
[[nodiscard]] std::variant<std::vector<std::uint8_t>, ApiError> decode_strict_base64(
    std::string_view b64, std::size_t max_decoded_bytes, std::string_view param);

[[nodiscard]] std::string encode_base64(std::span<const std::uint8_t> bytes);

// Parses a data:image/<png|jpeg|webp|gif>;base64,... URI. Strictly rejects
// http://, https://, file://, and any non-allowlisted URI with 400 invalid_request.
[[nodiscard]] std::variant<MediaContent, ApiError> parse_data_image_uri(std::string_view url, std::string_view param);

// Parses an Anthropic {"type":"image","source":{"type":"base64",...}} block.
// Strictly rejects source.type != "base64" (zero SSRF).
[[nodiscard]] std::variant<MediaContent, ApiError> parse_anthropic_image_block(const json::Object& block,
                                                                               std::string_view param);

// Parses OpenAI Chat Completions `tools` (or flat `/v1/responses` function tool
// entries when `allow_flat_responses_format` is true).
[[nodiscard]] std::optional<ApiError> parse_openai_tools(const json::Value& value, std::vector<ToolDefinition>& out,
                                                         bool allow_flat_responses_format = false);

// Parses OpenAI `tool_choice` ("auto" | "none" | "required" | {"type":"function",...}).
[[nodiscard]] std::optional<ApiError> parse_openai_tool_choice(const json::Value& value,
                                                               const std::vector<ToolDefinition>& tools,
                                                               ToolChoice& out,
                                                               bool allow_flat_responses_format = false);

// Parses assistant message `tool_calls` array.
[[nodiscard]] std::optional<ApiError> parse_openai_message_tool_calls(const json::Value& value,
                                                                      std::vector<ToolCall>& out,
                                                                      std::string_view param);

// Parses Anthropic `/v1/messages` `tools` and `tool_choice`.
[[nodiscard]] std::optional<ApiError> parse_anthropic_tools(const json::Value& value, std::vector<ToolDefinition>& out);
[[nodiscard]] std::optional<ApiError> parse_anthropic_tool_choice(const json::Value& value,
                                                                  const std::vector<ToolDefinition>& tools,
                                                                  ToolChoice& out);

// Formats tool definitions cleanly as a system instruction block for backends
// without native tool-template passthrough.
[[nodiscard]] std::string format_tools_system_prompt(const std::vector<ToolDefinition>& tools,
                                                     const std::optional<ToolChoice>& choice);

// Builds a GBNF grammar constraining output to `<tool_call>{"name":...,"arguments":...}</tool_call>`
// when tool_choice is `required` or a specific named `function`.
[[nodiscard]] Result<std::string> build_tool_call_gbnf(const std::vector<ToolDefinition>& tools,
                                                       const ToolChoice& choice);

// Incremental streaming parser recognizing `<tool_call>...</tool_call>`
// (Qwen/Bonsai/Hermes) and fenced ```tool_call ... ``` blocks across arbitrary
// chunk boundaries, suppressing raw tool-call markup from visible text.
struct ParsedToolEvent {
    std::uint32_t index = 0;
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct ToolCallStreamOutput {
    std::string visible_text;
    std::vector<ParsedToolEvent> tool_calls;
};

class ToolCallStreamParser {
public:
    explicit ToolCallStreamParser(std::string id_prefix = "call_");

    [[nodiscard]] ToolCallStreamOutput feed(std::string_view chunk);
    [[nodiscard]] ToolCallStreamOutput flush();
    [[nodiscard]] bool has_tool_calls() const noexcept { return next_index_ > 0; }
    [[nodiscard]] const std::vector<ToolCall>& completed_calls() const noexcept { return completed_; }

private:
    enum class State { text, in_xml, in_fence };

    void try_emit_payload(std::string_view payload, std::string_view raw_close, ToolCallStreamOutput& out);
    std::string next_call_id();

    std::string id_prefix_;
    State state_ = State::text;
    std::string buf_;
    std::string raw_open_;
    std::uint32_t next_index_ = 0;
    std::vector<ToolCall> completed_;
};

struct ExtractedToolOutput {
    std::string visible_text;
    std::vector<ToolCall> tool_calls;
};

[[nodiscard]] ExtractedToolOutput extract_tool_calls(std::string_view full_text, std::string id_prefix = "call_");

// Spec-compliant OpenAI logprobs JSON formatters.
[[nodiscard]] json::Object chat_token_logprob_json(const TokenLogprob& lp);
[[nodiscard]] json::Object chat_logprobs_json(const std::vector<TokenLogprob>& logprobs);
[[nodiscard]] json::Object completion_logprobs_json(const std::vector<TokenLogprob>& logprobs,
                                                    std::size_t base_offset = 0);

}  // namespace sonder::inference::server::detail
