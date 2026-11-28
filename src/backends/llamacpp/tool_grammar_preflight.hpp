// Tool-schema grammar preflight for the direct llama.cpp backend:
// compiles a toolset's tool-call grammar once on registration or first request,
// bisects compilation failures to identify offending tool(s), and caches
// results in a bounded LRU cache keyed by SHA-256(chat_template_id, canonical toolset JSON).
// Preflight is strictly read-only and never alters the grammar used at generation time.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backend.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::llamacpp_detail {

// Classifies why a tool-schema grammar preflight failed.
enum class ToolGrammarPreflightErrorKind {
    none = 0,
    compile_failed,       // One or more individual tools failed grammar compilation (bisected).
    interaction_failure,  // Whole toolset failed even though partitioned subsets compiled cleanly.
    invalid_input,        // Malformed preflight input or missing compiler.
    unsupported,          // Compiler/backend does not support tool-call grammar compilation.
    backend_error,        // Non-schema runtime/compiler failure unrelated to a specific tool.
};

[[nodiscard]] std::string_view ToString(ToolGrammarPreflightErrorKind kind) noexcept;

struct OffendingTool {
    std::string name;
    std::size_t index = 0;  // 0-based index in the input tool list
    std::string reason;     // compiler error reported for this tool
};

struct ToolGrammarPreflightError {
    ToolGrammarPreflightErrorKind kind = ToolGrammarPreflightErrorKind::none;
    ErrorCode status_code = ErrorCode::ok;
    std::vector<OffendingTool> offending_tools;
    std::string message;
    std::string compiler_error;

    [[nodiscard]] bool has_error() const noexcept {
        return kind != ToolGrammarPreflightErrorKind::none;
    }

    [[nodiscard]] Status to_status() const;
};

struct ToolGrammarPreflightResult {
    bool ok = true;
    bool cache_hit = false;
    std::string cache_key;
    ToolGrammarPreflightError error;

    [[nodiscard]] Status to_status() const {
        return ok ? Status::success() : error.to_status();
    }
};

struct ToolGrammarPreflightOptions {
    // Maximum number of cached (chat_template_id, toolset) preflight results. 0 disables caching.
    std::size_t max_cache_entries = 128;
    // Upper bound on offending tools collected during bisection (0 = unlimited).
    std::size_t max_offending_tools = 16;
};

struct ToolGrammarPreflightStats {
    std::size_t entries = 0;
    std::size_t max_entries = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t evictions = 0;
    std::uint64_t compile_calls = 0;
    std::uint64_t failures = 0;
};

// Abstract tool-call grammar compiler so preflight logic and bisection can be
// unit-tested cleanly without requiring model weights.
class ToolGrammarCompiler {
public:
    virtual ~ToolGrammarCompiler() = default;

    // Compiles the tool-call grammar for `tools` under `chat_template_id`.
    // Must not mutate generation state or alter any request's active grammar.
    [[nodiscard]] virtual Status Compile(std::string_view chat_template_id,
                                         std::span<const ToolDefinition> tools) const = 0;
};

// Returns a compiler backed by the native llama.cpp tool-call grammar path
// (JSON Schema -> GBNF tool-call grammar -> LlamaCppBackend::ProbeGrammarMask).
// The referenced `runtime` must outlive the returned compiler.
[[nodiscard]] std::shared_ptr<ToolGrammarCompiler> MakeLlamaCppToolGrammarCompiler(
    const sonder::backends::llamacpp::LlamaCppBackend& runtime);

// Recursively sorts all JSON object keys lexicographically while preserving array order.
[[nodiscard]] json::Value CanonicalizeJsonValue(const json::Value& value);

// Parses raw JSON and returns its compact key-sorted canonical serialization.
[[nodiscard]] Result<std::string> CanonicalizeJsonString(std::string_view raw_json);

// Produces a deterministic, key-order-independent canonical JSON representation
// of a toolset (including recursively canonicalized tool parameter schemas).
[[nodiscard]] std::string CanonicalizeToolsetJson(std::span<const ToolDefinition> tools);

// Computes a domain-separated, length-prefixed SHA-256 hex digest over
// (chat_template_id, CanonicalizeToolsetJson(tools)).
[[nodiscard]] std::string ComputeToolsetPreflightKey(std::string_view chat_template_id,
                                                     std::span<const ToolDefinition> tools);

// Thread-safe tool-schema grammar preflighter with binary-split bisection on
// compilation failure and a bounded LRU cache.
class ToolGrammarPreflight {
public:
    explicit ToolGrammarPreflight(std::shared_ptr<ToolGrammarCompiler> compiler,
                                  ToolGrammarPreflightOptions options = {});

    // Explicitly registers and preflights a toolset ahead of requests.
    [[nodiscard]] ToolGrammarPreflightResult RegisterToolset(std::string_view chat_template_id,
                                                             std::span<const ToolDefinition> tools);

    // Preflights a toolset on request arrival (returns cached result on hit).
    [[nodiscard]] ToolGrammarPreflightResult Preflight(std::string_view chat_template_id,
                                                       std::span<const ToolDefinition> tools);

    [[nodiscard]] ToolGrammarPreflightStats Stats() const;
    void ClearCache();

private:
    struct CacheEntry {
        std::string key;
        bool ok = true;
        ToolGrammarPreflightError error;
    };

    void BisectFailure(std::string_view chat_template_id,
                       std::span<const ToolDefinition> all_tools,
                       std::size_t begin,
                       std::size_t end,
                       const Status& known_status,
                       std::vector<OffendingTool>& offenders,
                       bool& truncated,
                       Status& non_schema_error,
                       std::uint64_t& local_compile_calls) const;

    std::shared_ptr<ToolGrammarCompiler> compiler_;
    ToolGrammarPreflightOptions options_;

    mutable std::mutex mu_;
    std::list<CacheEntry> lru_;
    std::unordered_map<std::string, std::list<CacheEntry>::iterator> index_;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t compile_calls_ = 0;
    std::uint64_t failures_ = 0;
};

}  // namespace sonder::inference::llamacpp_detail
