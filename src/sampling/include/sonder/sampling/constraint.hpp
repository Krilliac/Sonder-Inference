#pragma once

#include <string_view>
#include <unordered_set>
#include <vector>

#include "sonder/sampling/types.hpp"

namespace sonder::inference::sampling {

/// Grammar / structured-output hook.
///
/// A Constraint masks candidates that would violate some output structure
/// (GBNF grammar, JSON schema, regex, tool-call format, ...). The chain calls
/// apply() before any truncation stage so disallowed tokens never receive
/// probability mass, and accept() after a token is committed so the
/// constraint can advance its internal state.
///
/// Only the interface and a trivial reference implementation exist today; a
/// real grammar engine is future work (see INTEGRATION_NOTES.md).
class Constraint {
public:
    virtual ~Constraint() = default;

    [[nodiscard]] virtual std::string_view name() const noexcept = 0;

    /// Set logit = -inf on every candidate that is not allowed next.
    virtual void apply(Candidates& candidates) = 0;

    /// Advance state after `token` has been committed to the sequence.
    virtual void accept(TokenId token) = 0;

    /// Return to the initial state (new generation).
    virtual void reset() = 0;

    /// True when the structure is complete and generation may stop.
    [[nodiscard]] virtual bool is_complete() const noexcept { return false; }
};

/// Reference constraint: only tokens in a fixed allow-list may be sampled.
/// Stateless; useful for tests and for "choose one of N labels" prompts.
class TokenAllowlistConstraint final : public Constraint {
public:
    explicit TokenAllowlistConstraint(std::vector<TokenId> allowed);

    [[nodiscard]] std::string_view name() const noexcept override { return "allowlist"; }
    void apply(Candidates& candidates) override;
    void accept(TokenId) override {}
    void reset() override {}

private:
    std::unordered_set<TokenId> allowed_;
};

}  // namespace sonder::inference::sampling
