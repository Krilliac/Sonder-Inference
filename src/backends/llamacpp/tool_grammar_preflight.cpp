// Tool-schema grammar preflight implementation: canonical JSON hashing,
// bounded thread-safe LRU caching, logarithmic binary-split bisection on
// schema compilation failures, and native llama.cpp tool-call grammar compilation.
#include "tool_grammar_preflight.hpp"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "common/sha256.hpp"
#if defined(SONDER_HAS_SERVER)
#  include "server/src/tool_call_parser.hpp"
#endif

namespace sonder::inference::llamacpp_detail {

namespace {

namespace lc = sonder::backends::llamacpp;

Status MapLlamaCppStatus(const lc::Status& st) {
    if (st.ok()) return Status::success();
    switch (st.code) {
        case lc::ErrorCode::kOk: return Status::success();
        case lc::ErrorCode::kInvalidArgument:
        case lc::ErrorCode::kContextOverflow:
            return Status(ErrorCode::invalid_argument, st.message);
        case lc::ErrorCode::kFileNotFound:
            return Status(ErrorCode::not_found, st.message);
        case lc::ErrorCode::kNotLoaded:
            return Status(ErrorCode::invalid_state, st.message);
        case lc::ErrorCode::kUnsupported:
            return Status(ErrorCode::unsupported, st.message);
        case lc::ErrorCode::kLoadFailed:
        case lc::ErrorCode::kTokenizeFailed:
        case lc::ErrorCode::kDecodeFailed:
            return Status(ErrorCode::backend_error, st.message);
    }
    return Status(ErrorCode::internal, st.message);
}

bool IsSchemaError(ErrorCode code) noexcept {
    return code == ErrorCode::invalid_argument;
}

ToolGrammarPreflightError MakeNonSchemaError(const Status& st) {
    ToolGrammarPreflightError err;
    err.kind = st.code() == ErrorCode::unsupported
                   ? ToolGrammarPreflightErrorKind::unsupported
                   : ToolGrammarPreflightErrorKind::backend_error;
    err.status_code = st.code();
    err.compiler_error = st.message();
    err.message = "tool grammar preflight failed: " + st.message();
    return err;
}

std::string DisplayToolName(const OffendingTool& tool) {
    if (!tool.name.empty()) return tool.name;
    return "<unnamed#" + std::to_string(tool.index) + ">";
}

std::string FormatOffendingMessage(const std::vector<OffendingTool>& offenders, bool truncated) {
    std::string msg = "tool grammar preflight failed for tool(s) [";
    for (std::size_t i = 0; i < offenders.size(); ++i) {
        if (i > 0) msg += ", ";
        msg += DisplayToolName(offenders[i]);
    }
    if (truncated) {
        msg += ", ...";
    }
    msg += "]: ";
    for (std::size_t i = 0; i < offenders.size(); ++i) {
        if (i > 0) msg += "; ";
        msg += DisplayToolName(offenders[i]);
        msg += ": ";
        msg += offenders[i].reason;
    }
    return msg;
}

class LlamaCppToolGrammarCompiler final : public ToolGrammarCompiler {
public:
    explicit LlamaCppToolGrammarCompiler(const lc::LlamaCppBackend& runtime) noexcept : runtime_(runtime) {}

    Status Compile(std::string_view chat_template_id, std::span<const ToolDefinition> tools) const override {
        (void)chat_template_id;
        if (tools.empty()) {
            return Status::success();
        }
#if defined(SONDER_HAS_SERVER)
        std::vector<ToolDefinition> tool_vec(tools.begin(), tools.end());
        ToolChoice choice;
        choice.mode = ToolChoiceMode::required;
        auto gbnf = server::detail::build_tool_call_gbnf(tool_vec, choice);
        if (!gbnf.ok()) {
            return gbnf.status();
        }
        lc::SamplingParams params;
        params.grammar = std::move(gbnf).value();
        std::vector<std::uint8_t> allowed;
        return MapLlamaCppStatus(runtime_.ProbeGrammarMask(params, {}, allowed));
#else
        (void)runtime_;
        return Status(ErrorCode::unsupported,
                      "tool-call GBNF builder is unavailable because SONDER_HAS_SERVER is not enabled");
#endif
    }

private:
    const lc::LlamaCppBackend& runtime_;
};

}  // namespace

std::string_view ToString(ToolGrammarPreflightErrorKind kind) noexcept {
    switch (kind) {
        case ToolGrammarPreflightErrorKind::none: return "none";
        case ToolGrammarPreflightErrorKind::compile_failed: return "compile_failed";
        case ToolGrammarPreflightErrorKind::interaction_failure: return "interaction_failure";
        case ToolGrammarPreflightErrorKind::invalid_input: return "invalid_input";
        case ToolGrammarPreflightErrorKind::unsupported: return "unsupported";
        case ToolGrammarPreflightErrorKind::backend_error: return "backend_error";
    }
    return "unknown";
}

Status ToolGrammarPreflightError::to_status() const {
    switch (kind) {
        case ToolGrammarPreflightErrorKind::none:
            return Status::success();
        case ToolGrammarPreflightErrorKind::compile_failed:
        case ToolGrammarPreflightErrorKind::interaction_failure:
        case ToolGrammarPreflightErrorKind::invalid_input:
            return Status(ErrorCode::invalid_argument, message);
        case ToolGrammarPreflightErrorKind::unsupported:
            return Status(ErrorCode::unsupported, message);
        case ToolGrammarPreflightErrorKind::backend_error:
            return Status(status_code != ErrorCode::ok ? status_code : ErrorCode::backend_error, message);
    }
    return Status(ErrorCode::internal, message);
}

std::shared_ptr<ToolGrammarCompiler> MakeLlamaCppToolGrammarCompiler(const lc::LlamaCppBackend& runtime) {
    return std::make_shared<LlamaCppToolGrammarCompiler>(runtime);
}

json::Value CanonicalizeJsonValue(const json::Value& value) {
    if (value.is_object()) {
        const auto& obj = value.as_object();
        std::vector<std::pair<std::string_view, const json::Value*>> entries;
        entries.reserve(obj.size());
        for (const auto& [k, v] : obj) {
            entries.emplace_back(k, &v);
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        json::Object out;
        for (const auto& [k, v] : entries) {
            out.set(std::string(k), CanonicalizeJsonValue(*v));
        }
        return json::Value(std::move(out));
    }
    if (value.is_array()) {
        const auto& arr = value.as_array();
        json::Array out;
        out.reserve(arr.size());
        for (const auto& elem : arr) {
            out.push_back(CanonicalizeJsonValue(elem));
        }
        return json::Value(std::move(out));
    }
    return value;
}

Result<std::string> CanonicalizeJsonString(std::string_view raw_json) {
    auto parsed = json::parse(raw_json);
    if (!parsed.ok()) {
        return parsed.status();
    }
    return CanonicalizeJsonValue(parsed.value()).dump();
}

std::string CanonicalizeToolsetJson(std::span<const ToolDefinition> tools) {
    json::Array arr;
    arr.reserve(tools.size());
    for (const auto& tool : tools) {
        // Keys inserted in exact lexicographical order:
        // valid JSON schemas use "parameters"; unparseable schemas use out-of-band
        // "parameters_raw" so malformed text can never collide with any valid schema object.
        json::Object entry;
        entry.set("name", tool.name);
        if (tool.parameters_json.empty()) {
            entry.set("parameters", json::Object{{"type", "object"}});
        } else {
            auto parsed = json::parse(tool.parameters_json);
            if (parsed.ok()) {
                entry.set("parameters", CanonicalizeJsonValue(parsed.value()));
            } else {
                entry.set("parameters_raw", tool.parameters_json);
            }
        }
        entry.set("strict", tool.strict.has_value() ? json::Value(*tool.strict) : json::Value(nullptr));
        arr.emplace_back(std::move(entry));
    }
    return json::Value(std::move(arr)).dump();
}

std::string ComputeToolsetPreflightKey(std::string_view chat_template_id, std::span<const ToolDefinition> tools) {
    const std::string canonical_tools = CanonicalizeToolsetJson(tools);
    std::string payload;
    payload.reserve(40u + chat_template_id.size() + canonical_tools.size());
    payload += "tool_grammar_preflight/v1:";
    payload += std::to_string(chat_template_id.size());
    payload.push_back(':');
    payload.append(chat_template_id);
    payload.push_back(':');
    payload += std::to_string(canonical_tools.size());
    payload.push_back(':');
    payload.append(canonical_tools);
    return detail::sha256_hex(payload);
}

ToolGrammarPreflight::ToolGrammarPreflight(std::shared_ptr<ToolGrammarCompiler> compiler,
                                           ToolGrammarPreflightOptions options)
    : compiler_(std::move(compiler)), options_(options) {}

ToolGrammarPreflightResult ToolGrammarPreflight::RegisterToolset(std::string_view chat_template_id,
                                                                 std::span<const ToolDefinition> tools) {
    return Preflight(chat_template_id, tools);
}

ToolGrammarPreflightResult ToolGrammarPreflight::Preflight(std::string_view chat_template_id,
                                                           std::span<const ToolDefinition> tools) {
    ToolGrammarPreflightResult out;
    out.cache_key = ComputeToolsetPreflightKey(chat_template_id, tools);
    if (tools.empty()) {
        out.ok = true;
        return out;
    }
    if (!compiler_) {
        out.ok = false;
        out.error.kind = ToolGrammarPreflightErrorKind::invalid_input;
        out.error.status_code = ErrorCode::invalid_argument;
        out.error.message = "tool grammar preflight compiler is null";
        return out;
    }

    // 1. Bounded LRU lookup under lock.
    {
        std::lock_guard<std::mutex> lock(mu_);
        if (options_.max_cache_entries > 0) {
            if (auto it = index_.find(out.cache_key); it != index_.end()) {
                lru_.splice(lru_.begin(), lru_, it->second);
                ++hits_;
                out.ok = it->second->ok;
                out.cache_hit = true;
                out.error = it->second->error;
                return out;
            }
        }
        ++misses_;
    }

    // 2. Compile whole toolset once outside the cache lock.
    std::uint64_t local_compile_calls = 1;
    const Status whole_st = compiler_->Compile(chat_template_id, tools);
    bool should_cache = false;

    if (whole_st.ok()) {
        out.ok = true;
        should_cache = true;
    } else if (!IsSchemaError(whole_st.code())) {
        // Non-schema errors (unsupported, backend_error, internal, invalid_state,
        // cancelled, timeout, etc.) are never blamed on tools and never cached.
        out.ok = false;
        out.error = MakeNonSchemaError(whole_st);
        should_cache = false;
    } else {
        out.ok = false;
        out.error.status_code = ErrorCode::invalid_argument;
        out.error.compiler_error = whole_st.message();
        bool truncated = false;
        Status non_schema_error = Status::success();
        BisectFailure(chat_template_id, tools, 0, tools.size(), whole_st, out.error.offending_tools, truncated,
                      non_schema_error, local_compile_calls);
        if (!non_schema_error.ok()) {
            out.error = MakeNonSchemaError(non_schema_error);
            should_cache = false;
        } else if (!out.error.offending_tools.empty()) {
            out.error.kind = ToolGrammarPreflightErrorKind::compile_failed;
            out.error.message = FormatOffendingMessage(out.error.offending_tools, truncated);
            should_cache = true;
        } else {
            out.error.kind = ToolGrammarPreflightErrorKind::interaction_failure;
            out.error.message = "tool grammar preflight failed for combined toolset (" +
                                std::to_string(tools.size()) + " tools) with no single offending tool: " +
                                whole_st.message();
            should_cache = true;
        }
    }

    // 3. Record stats and insert deterministic schema outcomes into bounded LRU cache.
    {
        std::lock_guard<std::mutex> lock(mu_);
        compile_calls_ += local_compile_calls;
        if (!out.ok) {
            ++failures_;
        }
        if (should_cache && options_.max_cache_entries > 0) {
            if (auto it = index_.find(out.cache_key); it != index_.end()) {
                it->second->ok = out.ok;
                it->second->error = out.error;
                lru_.splice(lru_.begin(), lru_, it->second);
            } else {
                while (lru_.size() >= options_.max_cache_entries) {
                    index_.erase(lru_.back().key);
                    lru_.pop_back();
                    ++evictions_;
                }
                lru_.push_front(CacheEntry{out.cache_key, out.ok, out.error});
                index_[lru_.front().key] = lru_.begin();
            }
        }
    }

    return out;
}

void ToolGrammarPreflight::BisectFailure(std::string_view chat_template_id,
                                         std::span<const ToolDefinition> all_tools,
                                         std::size_t begin,
                                         std::size_t end,
                                         const Status& known_status,
                                         std::vector<OffendingTool>& offenders,
                                         bool& truncated,
                                         Status& non_schema_error,
                                         std::uint64_t& local_compile_calls) const {
    if (!non_schema_error.ok()) {
        return;
    }
    if (!IsSchemaError(known_status.code())) {
        non_schema_error = known_status;
        return;
    }
    if (options_.max_offending_tools > 0 && offenders.size() >= options_.max_offending_tools) {
        truncated = true;
        return;
    }
    const std::size_t count = end - begin;
    if (count == 0) {
        return;
    }
    if (count == 1) {
        offenders.push_back(OffendingTool{all_tools[begin].name, begin, known_status.message()});
        return;
    }

    const std::size_t mid = begin + count / 2;
    const Status left_st = compiler_->Compile(chat_template_id, all_tools.subspan(begin, mid - begin));
    ++local_compile_calls;
    if (!left_st.ok()) {
        if (!IsSchemaError(left_st.code())) {
            non_schema_error = left_st;
            return;
        }
        BisectFailure(chat_template_id, all_tools, begin, mid, left_st, offenders, truncated, non_schema_error,
                      local_compile_calls);
        if (!non_schema_error.ok() || truncated) {
            return;
        }
    }

    const Status right_st = compiler_->Compile(chat_template_id, all_tools.subspan(mid, end - mid));
    ++local_compile_calls;
    if (!right_st.ok()) {
        if (!IsSchemaError(right_st.code())) {
            non_schema_error = right_st;
            return;
        }
        BisectFailure(chat_template_id, all_tools, mid, end, right_st, offenders, truncated, non_schema_error,
                      local_compile_calls);
    }
}

ToolGrammarPreflightStats ToolGrammarPreflight::Stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    ToolGrammarPreflightStats s;
    s.entries = lru_.size();
    s.max_entries = options_.max_cache_entries;
    s.hits = hits_;
    s.misses = misses_;
    s.evictions = evictions_;
    s.compile_calls = compile_calls_;
    s.failures = failures_;
    return s;
}

void ToolGrammarPreflight::ClearCache() {
    std::lock_guard<std::mutex> lock(mu_);
    lru_.clear();
    index_.clear();
}

}  // namespace sonder::inference::llamacpp_detail
