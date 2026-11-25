// Internal: OpenAI Responses API (POST /v1/responses) request parser,
// bounded LRU response-chain store (mapping previous_response_id directly onto
// session_key for KV prefix cache reuse), and synchronous/SSE execution handler.
#pragma once

#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>
#include <vector>

#include "chat_handler.hpp"
#include "openai.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/sampling.hpp"
#include "sonder/inference/session.hpp"

namespace sonder::inference::server::detail {

// Thread-safe bounded LRU store retaining up to 256 prior /v1/responses turns.
class ResponseHistoryStore {
public:
    static constexpr std::size_t kMaxEntries = 256;

    struct Entry {
        std::string id;
        std::string session_key;
        std::string model;
        std::vector<ChatMessage> messages;
    };

    explicit ResponseHistoryStore(std::size_t capacity = kMaxEntries);

    [[nodiscard]] std::optional<Entry> get(const std::string& id);
    void put(Entry entry);
    [[nodiscard]] std::size_t size() const;

private:
    std::size_t capacity_ = kMaxEntries;
    mutable std::mutex mu_;
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

struct ResponsesJob {
    std::string model = "default";
    std::optional<std::string> instructions;
    std::optional<std::string> previous_response_id;
    std::vector<ChatMessage> input_messages;
    std::vector<ToolDefinition> tools;
    std::optional<ToolChoice> tool_choice;
    SamplingConfig sampling;
    ThinkingOptions thinking;
    bool stream = false;
    bool store = true;
    std::optional<std::string> session_key;
    std::string grammar;
    std::optional<std::string> response_format_type;
    std::optional<std::string> response_format_schema;
    std::optional<RequestPriority> priority_class;
    std::optional<std::uint64_t> deadline_ms;
};

[[nodiscard]] std::variant<ResponsesJob, ApiError> parse_responses_request(std::string_view body);

void execute_responses(ChatExchange& exchange, const ResponsesJob& job, const Correlation& correlation,
                       const std::string& model, const std::string& backend,
                       const std::shared_ptr<Session>& session, ResponseHistoryStore& history,
                       bool backend_native_tools = true);

}  // namespace sonder::inference::server::detail
