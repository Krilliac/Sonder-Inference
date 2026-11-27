// Grammar prefill retry policy (stdlib-only). The backend wiring is tested in
// src/backends/llamacpp/tests/test_llamacpp_util.cpp against a real grammar sampler.
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "backends/llamacpp/grammar_retry.hpp"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

TEST_CASE("grammar prefill retry: empty prefill is a no-op") {
    int rebuilds = 0;
    const std::vector<std::int32_t> none;
    CHECK(PrefillGrammarWithRetry(
              none, [](std::int32_t) { return true; },
              [&] {
                  ++rebuilds;
                  return true;
              },
              [](std::string_view) {}) == GrammarRetryOutcome::kNotNeeded);
    CHECK(rebuilds == 0);
}

TEST_CASE("grammar prefill retry: all tokens accepted needs no rebuild") {
    int rebuilds = 0;
    std::vector<std::int32_t> seen;
    const std::vector<std::int32_t> toks{5, 6, 7};
    CHECK(PrefillGrammarWithRetry(
              toks,
              [&](std::int32_t t) {
                  seen.push_back(t);
                  return true;
              },
              [&] {
                  ++rebuilds;
                  return true;
              },
              [](std::string_view) {}) == GrammarRetryOutcome::kSucceeded);
    CHECK(seen == toks);
    CHECK(rebuilds == 0);
}

TEST_CASE("grammar prefill retry: rejected token triggers exactly one rebuild with logging") {
    int rebuilds = 0;
    int accepts = 0;
    std::vector<std::string> logs;
    const std::vector<std::int32_t> toks{5, 6, 7};
    const auto r = PrefillGrammarWithRetry(
        toks,
        [&](std::int32_t t) {
            ++accepts;
            return t != 6;
        },
        [&] {
            ++rebuilds;
            return true;
        },
        [&](std::string_view m) { logs.emplace_back(m); });
    CHECK(r == GrammarRetryOutcome::kRetriedSucceeded);
    CHECK(accepts == 2);  // stops at the first rejected token
    CHECK(rebuilds == 1);
    CHECK(logs.size() == 1);
}

TEST_CASE("grammar prefill retry: failed rebuild reports failure") {
    std::vector<std::string> logs;
    const std::vector<std::int32_t> toks{1};
    CHECK(PrefillGrammarWithRetry(
              toks, [](std::int32_t) { return false; }, [] { return false; },
              [&](std::string_view m) { logs.emplace_back(m); }) == GrammarRetryOutcome::kFailed);
    CHECK(logs.size() == 2);
}
