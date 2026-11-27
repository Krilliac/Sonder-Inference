#include "llamaserver_stream.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

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

// A llama-server token id: an integer in [0, INT32_MAX].
std::optional<TokenId> parse_token_id(const json::Value *value) {
    if (value == nullptr)
        return std::nullopt;
    const auto id = value->try_as_int();
    if (!id || *id < 0 || *id > std::numeric_limits<TokenId>::max())
        return std::nullopt;
    return static_cast<TokenId>(*id);
}

// One probability from a llama-server entry: "prob" when post_sampling
// (post_sampling_probs), else exp("logprob"). Nullopt when absent or not a
// probability.
std::optional<float> parse_probability(const json::Value &entry, bool post_sampling) {
    const auto *value = entry.find(post_sampling ? "prob" : "logprob");
    if (value == nullptr || !value->is_number())
        return std::nullopt;
    const double raw = value->as_double();
    if (!std::isfinite(raw))
        return std::nullopt;
    const double p = post_sampling ? raw : std::exp(raw);
    // Tolerate float rounding just above 1 (a logprob of +1e-7); refuse the rest.
    if (!(p >= 0.0) || p > 1.0 + 1e-6)
        return std::nullopt;
    return static_cast<float>(std::min(p, 1.0));
}

// Parses one llama-server probability entry ({id, token, prob | logprob,
// top_probs | top_logprobs}, tools/server/server-task.cpp
// probs_vector_to_json) into `out`, keeping at most `top_k` alternatives,
// most probable first. False when the sampled token has no usable id or
// probability. `out.candidates` keeps its capacity across calls.
bool parse_sampled_probability(const json::Value &entry, std::size_t top_k, SampledTokenProbability &out) {
    if (!entry.is_object())
        return false;
    const auto token_id = parse_token_id(entry.find("id"));
    if (!token_id)
        return false;
    const bool post_sampling = entry.find("prob") != nullptr;
    const auto probability = parse_probability(entry, post_sampling);
    if (!probability)
        return false;
    out.token_id = *token_id;
    out.probability = *probability;
    out.post_sampling = post_sampling;
    out.candidates.clear();
    const auto *tops = entry.find(post_sampling ? "top_probs" : "top_logprobs");
    if (top_k == 0 || tops == nullptr || !tops->is_array())
        return true;
    for (const auto &top : tops->as_array()) {
        if (!top.is_object())
            continue;
        const auto id = parse_token_id(top.find("id"));
        const auto p = parse_probability(top, post_sampling);
        if (!id || !p)
            continue;
        const auto *text = top.find("token");
        out.candidates.push_back(
            TokenCandidate{*id, text != nullptr && text->is_string() ? text->as_string() : std::string(), *p});
    }
    std::stable_sort(out.candidates.begin(), out.candidates.end(),
                     [](const TokenCandidate &a, const TokenCandidate &b) { return a.probability > b.probability; });
    if (out.candidates.size() > top_k)
        out.candidates.resize(top_k);
    return true;
}

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
                                            std::uint64_t &progress_events, bool tools_requested,
                                            std::size_t max_tool_call_bytes, const StreamProbabilities &probabilities) {
    int http_status = 0;
    request.on_status = [&](int status) { http_status = status; };
    GenerateStats stats;
    Status error;
    bool stopped = false, terminal = false, done = false;
    bool usage_prompt = false, usage_completion = false, usage_cache = false;
    std::optional<std::uint64_t> timing_prompt_n = std::nullopt;
    std::optional<std::uint64_t> timing_cache_n = std::nullopt;
    std::optional<std::uint64_t> native_evaluated_n = std::nullopt;
    std::optional<std::uint64_t> native_cached_n = std::nullopt;
    std::uint64_t observed_chunks = 0;
    std::size_t accumulated_tool_call_bytes = 0;
    std::size_t accumulated_output_bytes = 0;
    const bool allow_tools = chat && tools_requested;
    const bool expose_logprobs = probabilities.expose_logprobs;
    const std::size_t telemetry_top_k = static_cast<std::size_t>(
        std::clamp<std::int32_t>(probabilities.telemetry_top_k, 0, kMaxTelemetryTopK));
    const bool want_sampled = probabilities.telemetry_top_k > 0;
    // Reused for every frame so the candidate vector keeps its capacity.
    SampledTokenProbability sampled_scratch;

    const auto recompute_prompt_tokens = [&]() -> bool {
        if (usage_prompt)
            return true;
        if (!timing_prompt_n.has_value() && !timing_cache_n.has_value() && !native_evaluated_n.has_value() &&
            !native_cached_n.has_value())
            return true;
        const std::uint64_t eval_n = timing_prompt_n.has_value() ? *timing_prompt_n : native_evaluated_n.value_or(0);
        const std::uint64_t cache_n =
            usage_cache ? stats.cached_tokens.value_or(0)
                        : (timing_cache_n.has_value() ? *timing_cache_n : native_cached_n.value_or(0));
        if (cache_n > std::numeric_limits<std::uint64_t>::max() - eval_n) {
            error = protocol_error("inconsistent timing token counts");
            return false;
        }
        stats.prompt_tokens = eval_n + cache_n;
        stats.token_counts_from_backend = true;
        return true;
    };

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
        if (timing.prompt_present)
            timing_prompt_n = timing.prompt_tokens;
        if (timing.cache_present)
            timing_cache_n = timing.cached_tokens;
        if (timing.cache_present && !usage_cache)
            stats.cached_tokens = timing.cached_tokens;
        if (timing.prompt_present || timing.cache_present) {
            if (!recompute_prompt_tokens())
                return false;
        }
        if (timing.completion_present && !usage_completion) {
            stats.completion_tokens = timing.completion_tokens;
            stats.token_counts_from_backend = true;
        }
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
            if (!error.ok())
                return false;
            usage_completion = count(*usage, "completion_tokens", stats.completion_tokens) || usage_completion;
            if (!error.ok())
                return false;
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
                if (!error.ok())
                    return false;
            }
            if (usage_prompt || usage_completion)
                stats.token_counts_from_backend = true;
            if (usage_cache && !usage_prompt && !recompute_prompt_tokens())
                return false;
        }
        std::string piece;
        std::string reasoning;
        std::string reason;
        bool saw_reasoning_text = false;
        std::optional<float> chunk_logprob = std::nullopt;
        std::vector<TopLogprob> chunk_top_logprobs;
        std::vector<ToolCallDelta> chunk_tool_deltas;
        bool have_sampled = false;
        if (native) {
            if (const auto *content = object.find("content")) {
                if (!content->is_string()) {
                    error = protocol_error("content must be text");
                    return false;
                }
                piece = content->as_string();
            }
            if (const auto *probs = object.find("completion_probabilities");
                probs && probs->is_array() && !probs->as_array().empty()) {
                const auto &p0 = probs->as_array().front();
                if (want_sampled)
                    have_sampled = parse_sampled_probability(p0, telemetry_top_k, sampled_scratch);
                if (expose_logprobs && p0.is_object()) {
                    if (const auto *lp = p0.find("logprob"); lp && lp->is_number() && std::isfinite(lp->as_double())) {
                        chunk_logprob = static_cast<float>(lp->as_double());
                    }
                    if (const auto *tops = p0.find("top_logprobs"); tops && tops->is_array()) {
                        for (const auto &entry : tops->as_array()) {
                            if (!entry.is_object()) continue;
                            const auto *tok = entry.find("token");
                            const auto *lp = entry.find("logprob");
                            if (tok && tok->is_string() && lp && lp->is_number() && std::isfinite(lp->as_double())) {
                                chunk_top_logprobs.push_back(
                                    TopLogprob{tok->as_string(), static_cast<float>(lp->as_double())});
                            }
                        }
                    }
                }
            }
            std::uint64_t evaluated = 0;
            const bool has_evaluated = count(object, "tokens_evaluated", evaluated);
            if (!error.ok())
                return false;
            std::uint64_t predicted = 0;
            const bool has_predicted = count(object, "tokens_predicted", predicted);
            if (!error.ok())
                return false;
            std::uint64_t cached = 0;
            const bool has_cached = count(object, "tokens_cached", cached);
            if (!error.ok())
                return false;
            if (has_evaluated)
                native_evaluated_n = evaluated;
            if (has_cached)
                native_cached_n = cached;
            if (has_cached && !timing_cache_n.has_value() && !usage_cache)
                stats.cached_tokens = cached;
            if (has_evaluated || has_cached) {
                if (!recompute_prompt_tokens())
                    return false;
            }
            if (has_predicted && !usage_completion && !timing.completion_present) {
                stats.completion_tokens = predicted;
                stats.token_counts_from_backend = true;
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
                    // Parse native OpenAI delta.tool_calls when explicitly requested.
                    if (allow_tools) {
                        if (const auto *tcs = delta->find("tool_calls"); tcs && tcs->is_array()) {
                            for (const auto &item : tcs->as_array()) {
                                if (!item.is_object()) continue;
                                ToolCallDelta tcd;
                                if (const auto *idx = item.find("index"); idx != nullptr) {
                                    const auto parsed_idx = idx->try_as_int();
                                    if (!parsed_idx || *parsed_idx < 0 || *parsed_idx >= 64) {
                                        continue;
                                    }
                                    tcd.index = static_cast<std::uint32_t>(*parsed_idx);
                                }
                                if (const auto *id = item.find("id"); id && id->is_string()) {
                                    tcd.id = id->as_string();
                                }
                                if (const auto *fn = item.find("function"); fn && fn->is_object()) {
                                    if (const auto *nm = fn->find("name"); nm && nm->is_string()) {
                                        tcd.name = nm->as_string();
                                    }
                                    if (const auto *args = fn->find("arguments"); args && !args->is_null()) {
                                        if (args->is_string()) {
                                            tcd.arguments_delta = args->as_string();
                                        } else if (args->is_object()) {
                                            tcd.arguments_delta = args->dump();
                                        } else {
                                            error = protocol_error("tool call arguments must be a string or object");
                                            return false;
                                        }
                                    }
                                }
                                const std::size_t delta_bytes =
                                    tcd.id.size() + tcd.name.size() + tcd.arguments_delta.size();
                                if (delta_bytes > 0) {
                                    if (max_tool_call_bytes > 0 &&
                                        (delta_bytes > max_tool_call_bytes ||
                                         accumulated_tool_call_bytes > max_tool_call_bytes - delta_bytes)) {
                                        error = protocol_error("tool call deltas exceed cumulative byte limit");
                                        return false;
                                    }
                                    accumulated_tool_call_bytes += delta_bytes;
                                    chunk_tool_deltas.push_back(std::move(tcd));
                                }
                            }
                        }
                    }
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
                if (const auto *lp_obj = choice.find("logprobs"); lp_obj && lp_obj->is_object()) {
                    if (const auto *lp_content = lp_obj->find("content");
                        lp_content && lp_content->is_array() && !lp_content->as_array().empty()) {
                        const auto &entry = lp_content->as_array().front();
                        if (want_sampled)
                            have_sampled = parse_sampled_probability(entry, telemetry_top_k, sampled_scratch);
                        if (expose_logprobs && entry.is_object()) {
                            if (const auto *lp = entry.find("logprob");
                                lp && lp->is_number() && std::isfinite(lp->as_double())) {
                                chunk_logprob = static_cast<float>(lp->as_double());
                            }
                            if (const auto *tops = entry.find("top_logprobs"); tops && tops->is_array()) {
                                for (const auto &t : tops->as_array()) {
                                    if (!t.is_object()) continue;
                                    const auto *tok = t.find("token");
                                    const auto *tlp = t.find("logprob");
                                    if (tok && tok->is_string() && tlp && tlp->is_number() &&
                                        std::isfinite(tlp->as_double())) {
                                        chunk_top_logprobs.push_back(
                                            TopLogprob{tok->as_string(), static_cast<float>(tlp->as_double())});
                                    }
                                }
                            }
                        }
                    } else if (const auto *tlps = lp_obj->find("token_logprobs");
                               expose_logprobs && tlps && tlps->is_array() && !tlps->as_array().empty()) {
                        const auto &v0 = tlps->as_array().front();
                        if (v0.is_number() && std::isfinite(v0.as_double())) {
                            chunk_logprob = static_cast<float>(v0.as_double());
                        }
                        if (const auto *tops = lp_obj->find("top_logprobs");
                            tops && tops->is_array() && !tops->as_array().empty() &&
                            tops->as_array().front().is_object()) {
                            for (const auto &[tok, val] : tops->as_array().front().as_object()) {
                                if (val.is_number() && std::isfinite(val.as_double())) {
                                    chunk_top_logprobs.push_back(
                                        TopLogprob{tok, static_cast<float>(val.as_double())});
                                }
                            }
                        }
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
                if (allow_tools) {
                    stats.stop_reason = StopReason::tool_calls;
                } else {
                    error = {ErrorCode::unsupported,
                             "llamaserver: tool calls are not represented by the text backend API"};
                    return false;
                }
            } else {
                const auto mapped = map_finish_reason(reason);
                if (mapped == StopReason::none) {
                    error = protocol_error("unsupported finish reason");
                    return false;
                }
                stats.stop_reason = mapped;
            }
        }
        std::size_t chunk_output_bytes = piece.size() + reasoning.size();
        for (const auto &tcd : chunk_tool_deltas) {
            chunk_output_bytes += tcd.id.size() + tcd.name.size() + tcd.arguments_delta.size();
        }
        if (chunk_output_bytes > kMaxStreamOutputBytes ||
            accumulated_output_bytes > kMaxStreamOutputBytes - chunk_output_bytes) {
            error = protocol_error("stream output exceeds cumulative byte limit");
            return false;
        }
        accumulated_output_bytes += chunk_output_bytes;
        if (!piece.empty() || !reasoning.empty() || saw_reasoning_text || !chunk_tool_deltas.empty())
            ++progress_events;
        if (!piece.empty() || !reasoning.empty() || !chunk_tool_deltas.empty()) {
            ++observed_chunks;
            if (callback) {
                TokenChunk chunk{piece, stats.chunks, reasoning};
                chunk.logprob = chunk_logprob;
                chunk.top_logprobs = std::move(chunk_top_logprobs);
                chunk.tool_call_deltas = std::move(chunk_tool_deltas);
                chunk.sampled_probability = have_sampled ? &sampled_scratch : nullptr;
                const bool keep = callback(chunk);
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
