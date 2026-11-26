// Internal: converts JSON Schema (draft-07 / 2020-12 subset) and OpenAI
// response_format specs into llama.cpp-compatible GBNF grammars, provides the
// StructuredGrammarEngine + GrammarMatcher abstraction (with honest build/link
// capability reporting for optional engines like SONDER_WITH_LLGUIDANCE), and a
// bounded thread-safe LRU CompiledGrammarCache shared across Chat and Responses.
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

#include "sonder/inference/backend.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::server::detail {

// Strict RFC 8259 JSON object GBNF grammar (root ::= object).
[[nodiscard]] std::string json_object_gbnf();

// Converts a JSON Schema object into a self-contained GBNF grammar whose root
// rule matches JSON values conforming to `schema`.
// Supports: type (object, array, string, integer, number, boolean, null, or
// array of types), properties, required, additionalProperties (false when
// properties are declared; bool or schema when properties are absent/empty),
// items, minItems/maxItems (<= 32), enum, const, anyOf, provably disjoint
// oneOf (disjoint JSON types, disjoint const/enum literals, or tagged-union
// objects with disjoint required discriminator const/enum), basic regex
// pattern subset (^...$, literals, character classes, *, +, ?, {m,n} <= 32),
// exact bounded integer minimum/maximum ([-999999, 999999], plus half-open 0),
// minLength/maxLength (<= 32), and local $ref into $defs / definitions, plus
// JSON Schema annotation metadata.
// Errors: invalid_argument for unsupported/unenforceable validation keywords
// (e.g. overlapping oneOf, allOf, lookaround/backref patterns, floating-point
// or out-of-range numeric bounds, bounds > 32, properties + permissive
// additionalProperties), malformed schemas, or exceeded traversal/size limits.
[[nodiscard]] Result<std::string> json_schema_to_gbnf(const json::Value& schema);

// Pluggable structured grammar compilation backend.
enum class GrammarEngineKind { gbnf, llguidance };
enum class GrammarConstraintKind { json_object, json_schema, regex, structural_tag, tool_call };
enum class StructuredParseMode { compile_eagerly, defer_grammar_compile };

[[nodiscard]] const char* to_string(GrammarEngineKind kind) noexcept;
[[nodiscard]] const char* to_string(GrammarConstraintKind kind) noexcept;

// Canonical key-sorted JSON serialization for deterministic schema normalization.
// Preserves array order while sorting object member keys lexicographically.
[[nodiscard]] std::string canonical_json_dump(const json::Value& value);

// Derives an opaque tokenizer / vocab identity key from served backend & model
// metadata so compiled artifacts remain isolated across distinct models/vocabs.
[[nodiscard]] std::string build_tokenizer_identity_key(
    std::string_view backend_name, std::string_view model_id,
    const ModelDescriptor* descriptor = nullptr,
    std::optional<std::uint64_t> vocab_size = std::nullopt);

struct GrammarCompileOptions {
    bool strict = true;
    bool coerce_unsupported_tool_args = false;
    std::string grammar_format = "gbnf";

    [[nodiscard]] std::string cache_key_fragment() const;
    bool operator==(const GrammarCompileOptions&) const noexcept = default;
};

struct GrammarEngineCapabilities {
    GrammarEngineKind engine = GrammarEngineKind::gbnf;
    std::string engine_name;
    bool compiled_in = false;            // Compile-time engine hook exists in binary
    bool sampler_linked = false;         // Active sampler/matcher implementation is compiled & linked
    bool active = false;                 // compiled_in && sampler_linked
    bool supports_json_object = false;
    bool supports_json_schema = false;
    bool supports_regex = false;         // Standalone unconstrained regex engine support
    bool supports_regex_subset = false;  // Bounded anchored regex subset inside JSON Schema `pattern`
    bool supports_structural_tags = false;
    std::size_t max_schema_depth = 0;
    std::size_t max_unrolled_bounds = 0;
    std::int64_t max_bounded_integer_abs = 0;
    std::size_t max_grammar_bytes = 0;
    std::string detail;
};

struct GrammarCapabilityMetadata {
    GrammarEngineKind engine = GrammarEngineKind::gbnf;
    GrammarConstraintKind constraint_kind = GrammarConstraintKind::json_schema;
    bool enforces_strict_schema = true;
    bool uses_fallback_relaxation = false;
    bool requires_gbnf_sampler = true;
    bool sampler_linked = true;
};

// Immutable compiled grammar artifact shared safely across concurrent requests.
struct CompiledGrammar {
    GrammarEngineKind engine = GrammarEngineKind::gbnf;
    GrammarConstraintKind constraint_kind = GrammarConstraintKind::json_schema;
    std::string normalized_schema_key;
    std::string normalized_schema_hash;
    std::string tokenizer_identity;
    // Llama.cpp GBNF text (populated when requires_gbnf_sampler is true).
    std::string gbnf;
    // Engine-native compiled payload (GBNF text for gbnf engine).
    std::string payload;
    // Canonical JSON schema or constraint spec representation (diagnostics / upstream passthrough).
    std::string raw_schema;
    GrammarCapabilityMetadata capabilities{};
    std::chrono::nanoseconds compile_duration{0};

    [[nodiscard]] std::size_t approx_bytes() const noexcept;
};

// Lowercase SHA-256 hash over normalized constraint key and compilation options.
[[nodiscard]] std::string normalized_schema_hash(GrammarConstraintKind kind, std::string_view normalized_key,
                                                 const GrammarCompileOptions& options = {});

// Per-request mutable matcher/application state bound to one immutable CompiledGrammar.
// Never shared across requests; owns independent progression/reset state and validates
// that the bound compiled artifact is backed by a linked sampler before execution.
class GrammarMatcher {
public:
    explicit GrammarMatcher(std::shared_ptr<const CompiledGrammar> compiled,
                            std::size_t max_bytes = 256u * 1024u);

    GrammarMatcher(const GrammarMatcher&) = delete;
    GrammarMatcher& operator=(const GrammarMatcher&) = delete;
    GrammarMatcher(GrammarMatcher&&) = default;
    GrammarMatcher& operator=(GrammarMatcher&&) = default;

    [[nodiscard]] const std::shared_ptr<const CompiledGrammar>& compiled() const noexcept { return compiled_; }
    [[nodiscard]] const std::string& gbnf() const noexcept { return compiled_->gbnf; }
    [[nodiscard]] bool active() const noexcept { return active_; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }
    [[nodiscard]] bool rejected() const noexcept { return rejected_; }
    [[nodiscard]] std::size_t bytes_fed() const noexcept { return bytes_fed_; }
    [[nodiscard]] std::size_t token_steps() const noexcept { return token_steps_; }
    [[nodiscard]] const std::string& last_error() const noexcept { return last_error_; }

    // Marks this per-request matcher active and returns the grammar payload to bind on RequestOptions.
    // Fails closed if compiled is null or not backed by a linked runtime sampler.
    [[nodiscard]] Result<std::string> bind_for_request();

    // Advances per-request chunk/step progression independently of any other request.
    Status accept_chunk(std::string_view piece);
    void mark_finished() noexcept;
    void reset() noexcept;

private:
    std::shared_ptr<const CompiledGrammar> compiled_;
    std::size_t max_bytes_ = 256u * 1024u;
    bool active_ = false;
    bool finished_ = false;
    bool rejected_ = false;
    std::size_t bytes_fed_ = 0;
    std::size_t token_steps_ = 0;
    std::string last_error_;
};

class StructuredGrammarEngine {
public:
    virtual ~StructuredGrammarEngine() = default;
    [[nodiscard]] virtual GrammarEngineKind kind() const noexcept = 0;
    [[nodiscard]] virtual std::string_view name() const noexcept = 0;
    [[nodiscard]] virtual GrammarEngineCapabilities capabilities() const noexcept = 0;
    [[nodiscard]] virtual Result<CompiledGrammar> compile_json_object() const = 0;
    [[nodiscard]] virtual Result<CompiledGrammar> compile_json_schema(const json::Value& schema) const = 0;
    [[nodiscard]] virtual Result<CompiledGrammar> compile_regex(std::string_view pattern) const = 0;
    [[nodiscard]] virtual Result<CompiledGrammar> compile_structural_tag(std::string_view spec) const = 0;
    [[nodiscard]] virtual Result<std::unique_ptr<GrammarMatcher>> create_matcher(
        std::shared_ptr<const CompiledGrammar> compiled) const = 0;
};

// Returns the built-in GBNF structured grammar engine.
[[nodiscard]] const StructuredGrammarEngine& gbnf_structured_grammar_engine() noexcept;

// Returns the llguidance engine wrapper (reports compiled_in/sampler_linked/active honestly;
// refuses compilation unless an actual llguidance sampler implementation is linked).
[[nodiscard]] const StructuredGrammarEngine& llguidance_structured_grammar_engine() noexcept;

// Returns the structured grammar engine for `kind`.
[[nodiscard]] const StructuredGrammarEngine& structured_grammar_engine(GrammarEngineKind kind) noexcept;

// Returns the process-default structured grammar engine (always an engine whose
// capabilities().active is true: GBNF unless an actual guidance sampler is linked).
[[nodiscard]] const StructuredGrammarEngine& default_structured_grammar_engine() noexcept;

// Binds a fresh per-request GrammarMatcher when `compiled` is non-null, populating `out_grammar`;
// otherwise preserves `explicit_fallback_grammar` with a null matcher.
[[nodiscard]] Result<std::unique_ptr<GrammarMatcher>> bind_request_grammar_matcher(
    const std::shared_ptr<const CompiledGrammar>& compiled,
    std::string& out_grammar,
    std::string_view explicit_fallback_grammar = {});

struct CompiledGrammarCacheKey {
    GrammarEngineKind engine = GrammarEngineKind::gbnf;
    GrammarConstraintKind constraint_kind = GrammarConstraintKind::json_schema;
    std::string tokenizer_identity;
    std::string normalized_schema_key;
    std::string options_key;

    bool operator==(const CompiledGrammarCacheKey&) const noexcept = default;
};

struct CompiledGrammarCacheKeyHash {
    [[nodiscard]] std::size_t operator()(const CompiledGrammarCacheKey& k) const noexcept;
};

struct CompiledGrammarCacheStats {
    std::size_t entries = 0;
    std::size_t used_bytes = 0;
    std::size_t max_entries = 0;
    std::size_t max_bytes = 0;
    std::size_t max_entry_bytes = 0;
    std::uint64_t hits = 0;
    std::uint64_t misses = 0;
    std::uint64_t inserts = 0;
    std::uint64_t evictions = 0;
    std::uint64_t bypassed_oversize = 0;
    std::uint64_t compile_failures = 0;
};

// Bounded, thread-safe LRU cache storing immutable CompiledGrammar artifacts keyed by
// (normalized_schema_key, tokenizer_identity, engine_kind, constraint_kind, options).
class CompiledGrammarCache {
public:
    static constexpr std::size_t kDefaultMaxEntries = 256u;
    static constexpr std::size_t kDefaultMaxTotalBytes = std::size_t{16} << 20;  // 16 MiB
    static constexpr std::size_t kDefaultMaxEntryBytes = std::size_t{1} << 20;   // 1 MiB

    explicit CompiledGrammarCache(std::size_t max_entries = kDefaultMaxEntries,
                                  std::size_t max_bytes = kDefaultMaxTotalBytes,
                                  std::size_t max_entry_bytes = kDefaultMaxEntryBytes);

    [[nodiscard]] std::shared_ptr<const CompiledGrammar> find(const CompiledGrammarCacheKey& key);
    void insert(const CompiledGrammarCacheKey& key, const std::shared_ptr<const CompiledGrammar>& artifact);

    // Looks up `key` or compiles via `compiler()` outside the cache lock on a miss.
    // On hit returns the shared immutable artifact directly; on oversize returns the compiled
    // artifact without evicting existing cache entries.
    [[nodiscard]] Result<std::shared_ptr<const CompiledGrammar>> get_or_compile(
        const CompiledGrammarCacheKey& key,
        const std::function<Result<CompiledGrammar>()>& compiler);

    [[nodiscard]] Result<std::shared_ptr<const CompiledGrammar>> get_or_compile_json_object(
        const StructuredGrammarEngine& engine,
        std::string_view tokenizer_identity = "default",
        const GrammarCompileOptions& options = {});

    [[nodiscard]] Result<std::shared_ptr<const CompiledGrammar>> get_or_compile_schema(
        const StructuredGrammarEngine& engine,
        const json::Value& schema,
        std::string_view tokenizer_identity = "default",
        const GrammarCompileOptions& options = {});

    [[nodiscard]] CompiledGrammarCacheStats stats() const;
    void clear();

private:
    struct Slot {
        CompiledGrammarCacheKey key;
        std::shared_ptr<const CompiledGrammar> artifact;
        std::size_t bytes = 0;
    };

    void evict_until_fit_locked(std::size_t needed_bytes) noexcept;

    std::size_t max_entries_ = kDefaultMaxEntries;
    std::size_t max_bytes_ = kDefaultMaxTotalBytes;
    std::size_t max_entry_bytes_ = kDefaultMaxEntryBytes;
    mutable std::mutex mu_;
    std::list<Slot> lru_;
    std::unordered_map<CompiledGrammarCacheKey, std::list<Slot>::iterator, CompiledGrammarCacheKeyHash> index_;
    std::size_t used_bytes_ = 0;
    std::uint64_t hits_ = 0;
    std::uint64_t misses_ = 0;
    std::uint64_t inserts_ = 0;
    std::uint64_t evictions_ = 0;
    std::uint64_t bypassed_oversize_ = 0;
    std::uint64_t compile_failures_ = 0;
};

}  // namespace sonder::inference::server::detail
