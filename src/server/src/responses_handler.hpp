// Internal: OpenAI Responses API (POST /v1/responses) request parser,
// bounded LRU+TTL response-chain store with caller-scope isolation (mapping
// previous_response_id directly onto session_key for KV prefix cache reuse),
// and synchronous/SSE execution handler.
#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
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
#include "http.hpp"
#include "openai.hpp"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/sampling.hpp"
#include "sonder/inference/session.hpp"

namespace sonder::inference::server::detail {

// Thread-safe bounded LRU+TTL store retaining up to 256 prior /v1/responses turns
// with caller-scope isolation so callers cannot read or overwrite each other's
// response chains.
class ResponseHistoryStore {
public:
    static constexpr std::size_t kMaxEntries = 256;
    static constexpr std::size_t kMaxTotalBytes = std::size_t{32} << 20;  // 32 MiB total cap
    static constexpr std::size_t kMaxEntryBytes = std::size_t{4} << 20;   // 4 MiB per entry cap
    static constexpr std::size_t kMaxMessagesPerEntry = 256;
    static constexpr std::chrono::milliseconds kDefaultTtl = std::chrono::hours(1);
    static constexpr std::chrono::milliseconds kMaxTtl = std::chrono::hours(24 * 30);

    using TimePoint = std::chrono::steady_clock::time_point;
    using ClockFn = std::function<TimePoint()>;

    struct Entry {
        std::string id;
        std::string session_key;
        std::string model;
        std::vector<ChatMessage> messages;
        std::size_t approx_bytes = 0;
        std::string caller_scope{};
        TimePoint expires_at{};
    };

    explicit ResponseHistoryStore(std::size_t capacity = kMaxEntries, std::size_t max_bytes = kMaxTotalBytes,
                                  std::chrono::milliseconds ttl = kDefaultTtl, ClockFn clock = {});

    [[nodiscard]] std::optional<Entry> get(const std::string& id, std::string_view caller_scope = {});
    void put(Entry entry);
    [[nodiscard]] std::size_t size() const;
    [[nodiscard]] std::chrono::milliseconds ttl() const noexcept { return ttl_; }

private:
    static std::size_t estimate_entry_bytes(const Entry& e) noexcept;
    [[nodiscard]] TimePoint now() const;
    void evict_back_locked() noexcept;
    void prune_expired_locked(TimePoint now) noexcept;

    std::size_t capacity_ = kMaxEntries;
    std::size_t max_bytes_ = kMaxTotalBytes;
    std::chrono::milliseconds ttl_ = kDefaultTtl;
    ClockFn clock_;
    std::size_t used_bytes_ = 0;
    mutable std::mutex mu_;
    std::list<Entry> lru_;
    std::unordered_map<std::string, std::list<Entry>::iterator> index_;
};

// Derives a non-reversible SHA-256 caller scope key from request authentication
// (Authorization / X-Api-Key) and correlation identity (X-Sonder-Run-Id /
// X-Sonder-Agent-Id) so raw bearer tokens are never retained in memory and
// distinct callers cannot chain each other's previous_response_id.
[[nodiscard]] std::string derive_responses_caller_scope(const RequestHead& head, const Correlation& correlation);

struct ResponsesJob {
    std::string model = "default";
    std::optional<std::string> instructions = std::nullopt;
    std::optional<std::string> previous_response_id = std::nullopt;
    std::vector<ChatMessage> input_messages;
    std::vector<ToolDefinition> tools;
    std::optional<ToolChoice> tool_choice = std::nullopt;
    std::optional<bool> parallel_tool_calls = std::nullopt;
    SamplingConfig sampling;
    ThinkingOptions thinking;
    std::optional<std::int64_t> reasoning_budget_tokens = std::nullopt;
    std::optional<std::string> reasoning_budget_message = std::nullopt;
    bool stream = false;
    bool store = true;
    std::optional<std::string> session_key = std::nullopt;
    std::string grammar;
    std::optional<std::string> response_format_type = std::nullopt;
    std::optional<std::string> response_format_schema = std::nullopt;
    std::optional<json::Value> pending_response_schema = std::nullopt;
    std::shared_ptr<const CompiledGrammar> compiled_grammar;
    std::optional<RequestPriority> priority_class = std::nullopt;
    std::optional<std::uint64_t> deadline_ms = std::nullopt;
};

[[nodiscard]] std::variant<ResponsesJob, ApiError> parse_responses_request(std::string_view body);
[[nodiscard]] std::variant<ResponsesJob, ApiError> parse_responses_request_with_mode(
    std::string_view body, StructuredParseMode schema_mode);

[[nodiscard]] std::optional<ApiError> prepare_responses_grammar(
    ResponsesJob& job, std::string_view tokenizer_identity,
    bool backend_supports_structured_output, bool backend_native_tools,
    std::string_view backend_name, CompiledGrammarCache* cache = nullptr,
    const StructuredGrammarEngine& engine = default_structured_grammar_engine());

void execute_responses(ChatExchange& exchange, const ResponsesJob& job, const Correlation& correlation,
                       const std::string& model, const std::string& backend,
                       const std::shared_ptr<Session>& session, ResponseHistoryStore& history,
                       bool backend_native_tools = true, bool backend_supports_multimodal = true,
                       std::string_view caller_scope = {},
                       CompiledGrammarCache* grammar_cache = nullptr,
                       std::string_view tokenizer_identity = {},
                       bool backend_supports_structured_output = true);

}  // namespace sonder::inference::server::detail
