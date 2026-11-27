// Grammar failure recovery policies for the direct llama.cpp backend.
//
// Pure stdlib-only C++20 (no llama.cpp headers) so both retry paths can be unit-tested without
// model weights. The backend supplies the llama.cpp-specific work as callables:
//
//   * speculative step: a grammar accept failure during speculative verification leaves the
//     grammar with no stacks. The step is retried once without speculation from a snapshot of the
//     sampler state taken before verification.
//   * grammar prefill: tokens pre-accepted into a (non-lazy) grammar before generation. If one is
//     rejected the grammar is rebuilt once without the prefill and generation continues.
//
// Thread affinity: the helpers hold no state and run on the caller's thread.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>

namespace sonder::backends::llamacpp {

enum class GrammarRetryOutcome : std::uint8_t {
    kNotNeeded = 0,     // nothing to do (no prefill tokens)
    kSucceeded,         // first attempt worked
    kRetriedSucceeded,  // first attempt failed, the single retry worked
    kFailed,            // retry impossible or failed as well
};

// Runs a speculative verification; on failure with an active grammar, restores the pre-step
// sampler snapshot and runs the same step once without speculation.
//   verify(bool speculative) -> bool   true when a usable result was produced
//   restore()                -> bool   true when the pre-step sampler state is back
//   log(std::string_view)              diagnostics (must not throw)
template <class Verify, class Restore, class Log>
[[nodiscard]] GrammarRetryOutcome VerifyWithNonSpeculativeRetry(bool grammar_active, std::size_t draft_count,
                                                                Verify&& verify, Restore&& restore, Log&& log) {
    if (verify(true)) {
        return GrammarRetryOutcome::kSucceeded;
    }
    if (!grammar_active || draft_count == 0) {
        return GrammarRetryOutcome::kFailed;
    }
    log("grammar rejected a speculative step (accept failure or empty grammar stack); "
        "retrying the step without speculation");
    if (!restore()) {
        log("grammar retry skipped: pre-step sampler snapshot unavailable");
        return GrammarRetryOutcome::kFailed;
    }
    if (verify(false)) {
        log("non-speculative grammar retry succeeded");
        return GrammarRetryOutcome::kRetriedSucceeded;
    }
    log("non-speculative grammar retry failed");
    return GrammarRetryOutcome::kFailed;
}

// Feeds `tokens` to the grammar via accept(token) -> bool. On the first rejected token the
// grammar is rebuilt without prefill via rebuild() -> bool and generation proceeds from there.
template <class Accept, class Rebuild, class Log>
[[nodiscard]] GrammarRetryOutcome PrefillGrammarWithRetry(std::span<const std::int32_t> tokens, Accept&& accept,
                                                          Rebuild&& rebuild, Log&& log) {
    if (tokens.empty()) {
        return GrammarRetryOutcome::kNotNeeded;
    }
    bool ok = true;
    for (const std::int32_t t : tokens) {
        if (!accept(t)) {
            ok = false;
            break;
        }
    }
    if (ok) {
        return GrammarRetryOutcome::kSucceeded;
    }
    log("grammar prefill failed; retrying once with grammar prefill disabled");
    if (!rebuild()) {
        log("grammar rebuild without prefill failed");
        return GrammarRetryOutcome::kFailed;
    }
    return GrammarRetryOutcome::kRetriedSucceeded;
}

}  // namespace sonder::backends::llamacpp
