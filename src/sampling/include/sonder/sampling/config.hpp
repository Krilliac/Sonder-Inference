#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/sampling/types.hpp"

namespace sonder::inference::sampling {

/// Transforming / truncating stages whose relative order is configurable.
/// Logit bias and the constraint hook always run first; the final selector
/// (greedy or seeded distribution) always runs last.
enum class StageKind : std::uint8_t {
    Penalties,
    TopK,
    TypicalP,
    TopP,
    MinP,
    Temperature,
};

[[nodiscard]] std::string_view to_string(StageKind kind) noexcept;

/// Parse a stage name ("penalties", "top_k", "typ_p"/"typical_p", "top_p",
/// "min_p", "temperature"/"temp"). Names follow llama.cpp's --samplers.
[[nodiscard]] std::optional<StageKind> parse_stage(std::string_view name) noexcept;

/// llama.cpp's default order restricted to the stages implemented here:
/// penalties -> top_k -> typ_p -> top_p -> min_p -> temperature.
[[nodiscard]] std::vector<StageKind> default_stage_order();

struct LogitBias {
    TokenId token = kInvalidToken;
    float bias = 0.0f;  ///< Added to the logit. -inf bans the token.
};

/// Full sampling policy. Defaults mirror llama.cpp's common defaults so a
/// default-constructed config samples like `llama-cli` with no flags.
/// See src/sampling/README.md for the exact semantics of every field.
struct SamplerConfig {
    // --- selection -------------------------------------------------------
    /// Always pick the highest-logit token after bias/constraint/penalties;
    /// truncation and temperature stages are skipped.
    bool greedy = false;
    /// RNG seed. nullopt = non-deterministic seed chosen at chain build time.
    std::optional<std::uint64_t> seed;

    // --- truncation / shaping -------------------------------------------
    float temperature = 0.8f;   ///< <= 0 behaves like greedy (argmax mask). Must be >= 0.
    std::int32_t top_k = 40;    ///< 0 = disabled. Values > vocab keep everything.
    float top_p = 0.95f;        ///< 1.0 = disabled. Range [0, 1].
    float min_p = 0.05f;        ///< 0.0 = disabled. Range [0, 1].
    float typical_p = 1.0f;     ///< 1.0 = disabled. Range [0, 1].
    std::int32_t min_keep = 0;  ///< Min candidates kept by top_p/min_p/typical_p (0 treated as 1).

    // --- penalties -------------------------------------------------------
    std::int32_t penalty_last_n = 64;  ///< Window of accepted tokens. 0 = disabled, -1 = unbounded.
    float repeat_penalty = 1.0f;       ///< 1.0 = disabled. Must be > 0.
    float frequency_penalty = 0.0f;    ///< 0.0 = disabled.
    float presence_penalty = 0.0f;     ///< 0.0 = disabled.

    // --- biasing ---------------------------------------------------------
    std::vector<LogitBias> logit_bias;

    // --- stopping --------------------------------------------------------
    std::vector<std::string> stop_sequences;  ///< Text-level; see StopSequenceMatcher.
    std::vector<TokenId> stop_tokens;         ///< Token-level (e.g. EOS / EOT ids).

    // --- ordering --------------------------------------------------------
    std::vector<StageKind> stage_order = default_stage_order();
};

struct ConfigError {
    std::string field;
    std::string message;
};

struct ValidationResult {
    std::vector<ConfigError> errors;

    [[nodiscard]] bool ok() const noexcept { return errors.empty(); }
    /// "sampling config invalid: top_p: must be in [0, 1] (got 1.5); ..."
    [[nodiscard]] std::string to_string() const;
};

/// Validate every field. When `vocab_size` is known, token ids (logit bias,
/// stop tokens) are also range-checked. Reports all problems, not just the first.
[[nodiscard]] ValidationResult validate(const SamplerConfig& config,
                                        std::optional<std::size_t> vocab_size = std::nullopt);

}  // namespace sonder::inference::sampling
