#include "llamaserver_protocol.hpp"

#include <cmath>
#include <limits>

namespace sonder::inference::llamaserver {
namespace {
// A flagged field is always sent. Otherwise, unless explicit_only (set by
// parsers that know exactly what the caller supplied), a field is sent only
// when it differs from the local default, so the upstream keeps its own
// model defaults.
bool supplied(const SamplingConfig &s, SamplingConfig::Field field, bool changed) {
    return s.is_explicit(field) || (!s.explicit_only && changed);
}
void put_sampling(json::Object &o, const SamplingConfig &s, bool native) {
    const SamplingConfig d{};
    if (supplied(s, SamplingConfig::kTemperature, s.temperature != d.temperature))
        o.set("temperature", static_cast<double>(s.temperature));
    if (supplied(s, SamplingConfig::kTopP, s.top_p != d.top_p))
        o.set("top_p", static_cast<double>(s.top_p));
    if (supplied(s, SamplingConfig::kTopK, s.top_k != d.top_k))
        o.set("top_k", s.top_k);
    if (supplied(s, SamplingConfig::kMinP, s.min_p != d.min_p))
        o.set("min_p", static_cast<double>(s.min_p));
    if (supplied(s, SamplingConfig::kRepeatPenalty, s.repeat_penalty != d.repeat_penalty))
        o.set("repeat_penalty", static_cast<double>(s.repeat_penalty));
    if (supplied(s, SamplingConfig::kTypicalP, s.typical_p != d.typical_p))
        o.set("typical_p", static_cast<double>(s.typical_p));
    if (supplied(s, SamplingConfig::kRepeatLastN, s.repeat_last_n != d.repeat_last_n))
        o.set("repeat_last_n", s.repeat_last_n);
    if (supplied(s, SamplingConfig::kPresencePenalty, s.presence_penalty != d.presence_penalty))
        o.set("presence_penalty", static_cast<double>(s.presence_penalty));
    if (supplied(s, SamplingConfig::kFrequencyPenalty, s.frequency_penalty != d.frequency_penalty))
        o.set("frequency_penalty", static_cast<double>(s.frequency_penalty));
    if (s.seed && supplied(s, SamplingConfig::kSeed, true))
        o.set("seed", static_cast<unsigned long long>(*s.seed));
    if (supplied(s, SamplingConfig::kMaxTokens, s.max_tokens != d.max_tokens))
        o.set(native ? "n_predict" : "max_tokens", s.max_tokens);
    if (supplied(s, SamplingConfig::kStop, !s.stop.empty())) {
        json::Array a;
        for (const auto &v : s.stop)
            a.emplace_back(v);
        o.set("stop", std::move(a));
    }
    if (supplied(s, SamplingConfig::kLogitBias, !s.logit_bias.empty())) {
        if (native) {
            json::Array bias;
            for (const auto &v : s.logit_bias) {
                bias.emplace_back(
                    json::Array{v.token, std::isinf(v.bias) ? json::Value(false) : json::Value(v.bias)});
            }
            o.set("logit_bias", std::move(bias));
        } else {
            json::Object bias;
            for (const auto &v : s.logit_bias)
                bias.set(std::to_string(v.token), std::isinf(v.bias) ? -100.0 : v.bias);
            o.set("logit_bias", std::move(bias));
        }
    }
}
} // namespace

Status validate_sampling(const SamplingConfig &sampling) {
    if (auto st = validate(sampling); !st.ok())
        return st;
    if (supplied(sampling, SamplingConfig::kNumCtx, sampling.num_ctx != 0)) {
        return Status(ErrorCode::unsupported, "llamaserver: num_ctx must be configured in server arguments");
    }
    return {};
}

json::Value build_completion_body(const std::string &model, const GenerateRequest &request,
                                  bool native_completion, std::string_view grammar) {
    json::Object o;
    o.set("model", model);
    o.set("prompt", request.prompt);
    o.set("stream", true);
    if (!native_completion)
        o.set("stream_options", json::Object{{"include_usage", true}});
    put_sampling(o, request.sampling, native_completion);
    if (!grammar.empty())
        o.set("grammar", std::string(grammar));
    return o;
}

json::Value build_chat_body(const std::string &model, const ChatRequest &request, std::string_view grammar) {
    json::Object o;
    o.set("model", model);
    o.set("stream", true);
    json::Array messages;
    for (const auto &m : request.messages)
        messages.emplace_back(json::Object{{"role", m.role}, {"content", m.content}});
    o.set("messages", std::move(messages));
    o.set("stream_options", json::Object{{"include_usage", true}});
    put_sampling(o, request.sampling, false);
    if (!grammar.empty())
        o.set("grammar", std::string(grammar));
    return o;
}

Status parse_timings(const json::Value &value, Timings &out) {
    out = {};
    const json::Value *t = value.find("timings");
    if (t && !t->is_object() && !t->is_null())
        return Status(ErrorCode::protocol_error, "llamaserver: timings must be an object");
    if (!t || t->is_null())
        t = &value;
    const auto count = [&](const char *key, std::uint64_t &dest, bool &present) {
        const auto *v = t->find(key);
        if (!v)
            return true;
        if (!v->is_integer() || v->as_double() < 0)
            return false;
        dest = v->as_uint();
        present = true;
        out.present = true;
        return true;
    };
    if (!count("prompt_n", out.prompt_tokens, out.prompt_present) ||
        !count(t->find("cache_n") ? "cache_n" : "cached_tokens", out.cached_tokens, out.cache_present) ||
        !count(t->find("predicted_n") ? "predicted_n" : "completion_n", out.completion_tokens,
               out.completion_present) ||
        !count("draft_n", out.draft_tokens, out.draft_present) ||
        !count("draft_n_accepted", out.draft_accepted_tokens, out.draft_accepted_present)) {
        return Status(ErrorCode::protocol_error, "llamaserver: invalid timing token count");
    }
    if (out.cached_tokens > std::numeric_limits<std::uint64_t>::max() - out.prompt_tokens ||
        (out.draft_present && out.draft_accepted_present && out.draft_accepted_tokens > out.draft_tokens)) {
        return Status(ErrorCode::protocol_error, "llamaserver: inconsistent timing token counts");
    }
    if (const auto *speed = t->find("predicted_per_second")) {
        if (!speed->is_number() || !std::isfinite(speed->as_double()) || speed->as_double() < 0) {
            return Status(ErrorCode::protocol_error, "llamaserver: invalid token rate");
        }
        out.speed_present = true;
        out.present = true;
        out.predicted_tokens_per_second = speed->as_double();
    }
    const auto duration = [&](const char *key, std::uint64_t &ns) {
        const auto *v = t->find(key);
        if (!v)
            return true;
        const double ms = v->as_double(-1.0);
        if (!v->is_number() || !std::isfinite(ms) || ms < 0 ||
            ms >= static_cast<double>(std::numeric_limits<std::uint64_t>::max()) / 1e6)
            return false;
        ns = static_cast<std::uint64_t>(ms * 1e6);
        out.present = true;
        return true;
    };
    if (!duration("prompt_ms", out.prompt_ns) || !duration("predicted_ms", out.eval_ns)) {
        return Status(ErrorCode::protocol_error, "llamaserver: invalid timing duration");
    }
    return Status::success();
}

StopReason map_finish_reason(std::string_view reason) {
    if (reason == "length" || reason == "limit")
        return StopReason::max_tokens;
    if (reason == "word")
        return StopReason::stop_sequence;
    if (reason == "stop" || reason == "eos" || reason == "end_of_sequence")
        return StopReason::end_of_sequence;
    return StopReason::none;
}
} // namespace sonder::inference::llamaserver
