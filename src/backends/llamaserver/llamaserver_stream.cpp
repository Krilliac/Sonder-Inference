#include "llamaserver_stream.hpp"

#include <functional>
#include <string_view>
#include <utility>

#include "llamaserver_protocol.hpp"
#include "sonder/inference/json.hpp"
#include "timings.hpp"

namespace sonder::inference::llamaserver {

namespace {

// Bounded SSE event/line framing, also accepting native newline-delimited JSON.
// Frame callbacks own JSON validation; there is no unbounded LineSplitter.
class EventDecoder {
  public:
    using Callback = std::function<bool(std::string_view)>;
    explicit EventDecoder(Callback callback) : callback_(std::move(callback)) {}
    bool feed(std::string_view bytes) {
        while (!bytes.empty()) {
            const auto newline = bytes.find('\n');
            const auto part = bytes.substr(0, newline);
            if (line_.size() + part.size() > kLimit)
                return fail();
            line_.append(part);
            if (newline == std::string_view::npos)
                return true;
            bytes.remove_prefix(newline + 1);
            if (!line_.empty() && line_.back() == '\r')
                line_.pop_back();
            if (!line())
                return false;
            line_.clear();
        }
        return true;
    }
    bool finish() {
        if (!line_.empty()) {
            if (line_.back() == '\r')
                line_.pop_back();
            if (!line())
                return false;
            line_.clear();
        }
        return dispatch();
    }
    bool oversized() const { return oversized_; }

  private:
    bool fail() {
        oversized_ = true;
        return false;
    }
    bool dispatch() {
        if (event_.empty())
            return true;
        const bool keep_going = callback_(event_);
        event_.clear();
        return keep_going;
    }
    bool line() {
        if (line_.empty())
            return dispatch();
        if (line_.front() == ':')
            return true;
        if (line_.rfind("data:", 0) == 0) {
            std::string_view data(line_);
            data.remove_prefix(5);
            if (!data.empty() && data.front() == ' ')
                data.remove_prefix(1);
            if (event_.size() + data.size() + 1 > kLimit)
                return fail();
            if (!event_.empty())
                event_.push_back('\n');
            event_.append(data);
            return true;
        }
        if (line_.rfind("event:", 0) == 0 || line_.rfind("id:", 0) == 0 || line_.rfind("retry:", 0) == 0)
            return true;
        // Native completions on some upstream versions use NDJSON.
        if (!event_.empty())
            return callback_("invalid mixed SSE/JSON framing");
        return callback_(line_);
    }
    static constexpr std::size_t kLimit = 1024 * 1024;
    Callback callback_;
    std::string line_;
    std::string event_;
    bool oversized_ = false;
};

}  // namespace

Status protocol_error(std::string message) { return {ErrorCode::protocol_error, "llamaserver: " + message}; }

Status status_for_http(int code) {
    const std::string message = "llamaserver: upstream HTTP " + std::to_string(code);
    // Do not include arbitrary upstream bodies (which can contain prompts or credentials).
    if (code == 400 || code == 422)
        return {ErrorCode::invalid_argument, message};
    if (code == 404)
        return {ErrorCode::not_found, message};
    if (code == 408 || code == 504)
        return {ErrorCode::timeout, message};
    if (code == 503 || code == 429)
        return {ErrorCode::unavailable, message};
    return {ErrorCode::backend_error, message};
}

Status upstream_status(const Status &status) {
    if (status.code() == ErrorCode::protocol_error) {
        const std::string &m = status.message();
        for (const char *prefix : {"idle timeout", "first-content timeout", "stall timeout"})
            if (m.rfind(prefix, 0) == 0)
                return {ErrorCode::protocol_error, "llamaserver: " + m};
    }
    return status;
}

Result<GenerateStats> run_completion_stream(net::HttpRequest &request, bool chat, bool native, bool separate_reasoning,
                                            const CancellationToken &cancel, const TokenCallback &callback,
                                            std::uint64_t &progress_events) {
    int http_status = 0;
    request.on_status = [&](int status) { http_status = status; };
    GenerateStats stats;
    Status error;
    bool stopped = false, terminal = false, done = false;
    bool usage_prompt = false, usage_completion = false, usage_cache = false;
    std::uint64_t observed_chunks = 0;

    const auto count = [&](const json::Value &object, const char *key, std::uint64_t &dest) {
        const auto *value = object.find(key);
        if (!value)
            return false;
        if (!value->is_integer() || value->as_double() < 0) {
            error = protocol_error(std::string("invalid count: ") + key);
            return false;
        }
        dest = value->as_uint();
        return true;
    };

    const auto frame = [&](std::string_view text) {
        if (cancel.cancelled())
            return false;
        if (text == "[DONE]") {
            if (!terminal) {
                error = protocol_error("DONE without a finish reason");
                return false;
            }
            done = true;
            return true;
        }
        if (done) {
            error = protocol_error("data after DONE");
            return false;
        }
        auto parsed = json::parse(text);
        if (!parsed.ok() || !parsed->is_object()) {
            error = protocol_error("malformed completion frame");
            return false;
        }
        const auto &object = parsed.value();
        if (const auto *upstream_error = object.find("error"); upstream_error && !upstream_error->is_null()) {
            error = {ErrorCode::backend_error, "llamaserver: upstream reported a completion error"};
            return false;
        }
        Timings timing;
        error = parse_timings(object, timing);
        if (!error.ok())
            return false;
        merge_timings(stats.backend_timings, timing.backend_timings);
        if (timing.prompt_present && !usage_prompt) {
            stats.prompt_tokens = timing.prompt_tokens + timing.cached_tokens;
            stats.token_counts_from_backend = true;
        }
        if (timing.completion_present && !usage_completion) {
            stats.completion_tokens = timing.completion_tokens;
            stats.token_counts_from_backend = true;
        }
        if (timing.cache_present && !usage_cache)
            stats.cached_tokens = timing.cached_tokens;
        if (timing.draft_present)
            stats.draft_tokens = timing.draft_tokens;
        if (timing.draft_accepted_present)
            stats.draft_accepted_tokens = timing.draft_accepted_tokens;
        if (timing.speed_present)
            stats.predicted_tokens_per_second = timing.predicted_tokens_per_second;
        if (timing.prompt_ns)
            stats.prompt_eval_ns = timing.prompt_ns;
        if (timing.eval_ns)
            stats.eval_ns = timing.eval_ns;

        if (const auto *usage = object.find("usage"); usage && !usage->is_null()) {
            if (!usage->is_object()) {
                error = protocol_error("usage must be an object");
                return false;
            }
            usage_prompt = count(*usage, "prompt_tokens", stats.prompt_tokens) || usage_prompt;
            usage_completion = count(*usage, "completion_tokens", stats.completion_tokens) || usage_completion;
            if (const auto *details = usage->find("prompt_tokens_details")) {
                if (!details->is_object()) {
                    error = protocol_error("prompt_tokens_details must be an object");
                    return false;
                }
                std::uint64_t cached = 0;
                if (count(*details, "cached_tokens", cached)) {
                    stats.cached_tokens = cached;
                    usage_cache = true;
                }
            }
            if (usage_prompt || usage_completion)
                stats.token_counts_from_backend = true;
        }
        std::string piece;
        std::string reasoning;
        std::string reason;
        bool saw_reasoning_text = false;
        if (native) {
            if (const auto *content = object.find("content")) {
                if (!content->is_string()) {
                    error = protocol_error("content must be text");
                    return false;
                }
                piece = content->as_string();
            }
            if (!usage_prompt && count(object, "tokens_evaluated", stats.prompt_tokens))
                stats.token_counts_from_backend = true;
            if (!usage_completion && count(object, "tokens_predicted", stats.completion_tokens))
                stats.token_counts_from_backend = true;
            if (const auto *cached = object.find("tokens_cached"); cached) {
                std::uint64_t value = 0;
                if (count(object, "tokens_cached", value) && !timing.cache_present && !usage_cache)
                    stats.cached_tokens = value;
            }
            const auto *stop = object.find("stop");
            if (stop && stop->is_bool() && stop->as_bool()) {
                terminal = true;
                const auto *type = object.find("stop_type");
                const auto is_true = [&](const char *key) {
                    const auto *v = object.find(key);
                    return v && v->is_bool() && v->as_bool();
                };
                if (is_true("stopped_limit"))
                    reason = "limit";
                else if (is_true("stopped_word"))
                    reason = "word";
                else
                    reason = type && type->is_string() ? type->as_string() : "eos";
            }
        } else if (const auto *choices = object.find("choices")) {
            if (!choices->is_array() || choices->as_array().size() > 1) {
                error = protocol_error("invalid choices");
                return false;
            }
            if (!choices->as_array().empty()) {
                const auto &choice = choices->as_array().front();
                const json::Value *delta = chat ? choice.find("delta") : nullptr;
                const auto *content = chat ? (delta ? delta->find("content") : nullptr) : choice.find("text");
                if (content && !content->is_null()) {
                    if (!content->is_string()) {
                        error = protocol_error("completion content must be text");
                        return false;
                    }
                    piece = content->as_string();
                }
                if (delta && delta->is_object()) {
                    // Reasoning text is progress even when the caller did not
                    // ask for it separately; a thinking model can stream only
                    // reasoning for a long time.
                    const auto *probe = delta->find("reasoning_content");
                    if (!probe)
                        probe = delta->find("reasoning");
                    if (probe && probe->is_string() && !probe->as_string().empty())
                        saw_reasoning_text = true;
                }
                if (separate_reasoning && delta && delta->is_object()) {
                    // llama-server's OpenAI-compatible reasoning extension is
                    // emitted as reasoning_content. Accept reasoning as a
                    // compatibility fallback used by some upstream builds.
                    const auto *value = delta->find("reasoning_content");
                    if (!value)
                        value = delta->find("reasoning");
                    if (value && !value->is_null()) {
                        if (!value->is_string()) {
                            error = protocol_error("reasoning content must be text");
                            return false;
                        }
                        reasoning = value->as_string();
                    }
                }
                if (const auto *finish = choice.find("finish_reason"); finish && !finish->is_null()) {
                    if (!finish->is_string()) {
                        error = protocol_error("invalid finish reason");
                        return false;
                    }
                    reason = finish->as_string();
                    terminal = true;
                }
            }
        }
        if (!error.ok())
            return false;
        if (!reason.empty()) {
            if (reason == "tool_calls" || reason == "function_call") {
                error = {ErrorCode::unsupported,
                         "llamaserver: tool calls are not represented by the text backend API"};
                return false;
            }
            const auto mapped = map_finish_reason(reason);
            if (mapped == StopReason::none) {
                error = protocol_error("unsupported finish reason");
                return false;
            }
            stats.stop_reason = mapped;
        }
        if (!piece.empty() || !reasoning.empty() || saw_reasoning_text)
            ++progress_events;
        if (!piece.empty() || !reasoning.empty()) {
            ++observed_chunks;
            if (callback) {
                const bool keep = callback(TokenChunk{piece, stats.chunks, reasoning});
                ++stats.chunks;
                if (!keep) {
                    stopped = true;
                    return false;
                }
            }
        }
        return true;
    };

    EventDecoder decoder(frame);
    const auto response = net::http_request(
        request,
        [&](std::string_view bytes) {
            if (http_status != 200)
                return false;
            return decoder.feed(bytes);
        },
        cancel);
    if (cancel.cancelled())
        return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
    if (http_status != 0 && http_status != 200)
        return status_for_http(http_status);
    if (!error.ok())
        return error;
    if (decoder.oversized())
        return protocol_error("completion frame exceeds 1 MiB");
    if (stopped) {
        stats.stop_reason = StopReason::callback;
        return stats;
    }
    if (!response.ok())
        return upstream_status(response.status());
    const bool finished = decoder.finish();
    if (cancel.cancelled())
        return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
    if (!error.ok())
        return error;
    if (stopped) {
        stats.stop_reason = StopReason::callback;
        return stats;
    }
    if (!finished || !terminal)
        return protocol_error("stream ended without a valid completion");
    if (!stats.token_counts_from_backend)
        stats.completion_tokens = observed_chunks;
    return stats;
}

}  // namespace sonder::inference::llamaserver
