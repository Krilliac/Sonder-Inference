// Grammar failure recovery policies (non-speculative step retry, grammar prefill retry).
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "backends/llamacpp/grammar_retry.hpp"
#include <doctest/doctest.h>

using namespace sonder::backends::llamacpp;

TEST_CASE("grammar retry: speculative step succeeds first time without restoring") {
    int verifies = 0;
    int restores = 0;
    std::vector<std::string> logs;
    const auto r = VerifyWithNonSpeculativeRetry(
        true, 3,
        [&](bool) {
            ++verifies;
            return true;
        },
        [&] {
            ++restores;
            return true;
        },
        [&](std::string_view m) { logs.emplace_back(m); });
    CHECK(r == GrammarRetryOutcome::kSucceeded);
    CHECK(verifies == 1);
    CHECK(restores == 0);
    CHECK(logs.empty());
}

TEST_CASE("grammar retry: speculative failure restores then retries without speculation and logs") {
    std::vector<bool> modes;
    int restores = 0;
    std::vector<std::string> logs;
    const auto r = VerifyWithNonSpeculativeRetry(
        true, 4,
        [&](bool spec) {
            modes.push_back(spec);
            return !spec;
        },
        [&] {
            ++restores;
            return true;
        },
        [&](std::string_view m) { logs.emplace_back(m); });
    CHECK(r == GrammarRetryOutcome::kRetriedSucceeded);
    REQUIRE(modes.size() == 2);
    CHECK(modes[0] == true);
    CHECK(modes[1] == false);
    CHECK(restores == 1);
    CHECK(logs.size() == 2);
}

TEST_CASE("grammar retry: no retry without a grammar or without drafts") {
    int restores = 0;
    int verifies = 0;
    auto verify = [&](bool) {
        ++verifies;
        return false;
    };
    auto restore = [&] {
        ++restores;
        return true;
    };
    auto log = [](std::string_view) {};
    CHECK(VerifyWithNonSpeculativeRetry(false, 4, verify, restore, log) == GrammarRetryOutcome::kFailed);
    CHECK(VerifyWithNonSpeculativeRetry(true, 0, verify, restore, log) == GrammarRetryOutcome::kFailed);
    CHECK(verifies == 2);
    CHECK(restores == 0);
}

TEST_CASE("grammar retry: failed restore or failed retry reports failure, retry runs at most once") {
    int verifies = 0;
    auto log = [](std::string_view) {};
    CHECK(VerifyWithNonSpeculativeRetry(
              true, 2,
              [&](bool) {
                  ++verifies;
                  return false;
              },
              [] { return false; }, log) == GrammarRetryOutcome::kFailed);
    CHECK(verifies == 1);
    verifies = 0;
    CHECK(VerifyWithNonSpeculativeRetry(
              true, 2,
              [&](bool) {
                  ++verifies;
                  return false;
              },
              [] { return true; }, log) == GrammarRetryOutcome::kFailed);
    CHECK(verifies == 2);
}

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
