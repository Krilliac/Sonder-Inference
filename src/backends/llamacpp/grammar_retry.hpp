// Grammar failure recovery policies for the direct llama.cpp backend.
//
// Pure stdlib-only C++20 (no llama.cpp headers) so both retry paths can be unit-tested without
// model weights. The backend supplies the llama.cpp-specific work as callables:
//
//   * There is deliberately no speculative-step retry: a grammar accept failure is a deterministic
//     function of the token sequence, so restoring a snapshot and re-sampling reproduces it. The
//     real fix is upstream ggml-org/llama.cpp#32934 (transactional lazy-trigger activation).
//   * grammar prefill: tokens pre-accepted into a (non-lazy) grammar before generation. If one is
//     rejected the grammar is rebuilt once without the prefill and generation continues.
//
// Thread affinity: the helpers hold no state and run on the caller's thread.
#pragma once

#include <cstdint>
#include <span>
#include <string_view>

namespace sonder::backends::llamacpp {

enum class GrammarRetryOutcome : std::uint8_t {
    kNotNeeded = 0,     // nothing to do (no prefill tokens)
    kSucceeded,         // first attempt worked
    kRetriedSucceeded,  // first attempt failed, the single retry worked
    kFailed,            // the rebuild failed
};

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
