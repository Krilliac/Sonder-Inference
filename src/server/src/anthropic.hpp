// Internal: Anthropic Messages API request and response codec.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "openai.hpp"

namespace sonder::inference::server::detail {

std::variant<ChatJob, ApiError> parse_messages_request(std::string_view body);
json::Object anthropic_error_body(const ApiError& error);

json::Object anthropic_message_response(std::string_view id, std::string_view model,
                                        const GenerationResult& result,
                                        const std::vector<std::string>& warnings = {});

// Produces complete SSE frames. The caller writes each returned string as one
// response chunk; no state is shared with the HTTP connection.
class AnthropicStream {
  public:
    AnthropicStream(std::string id, std::string model);
    std::string start();
    std::string token(const TokenChunk& chunk);
    std::string chunk(const TokenChunk& value) { return token(value); }
    std::string finish(const GenerationResult& result, const std::vector<std::string>& warnings = {},
                       json::Object metadata = {});

  private:
    std::string id_;
    std::string model_;
    bool started_ = false;
    bool finished_ = false;
    bool thinking_block_ = false;
    bool text_block_ = false;
    std::uint64_t block_index_ = 0;
};

} // namespace sonder::inference::server::detail
