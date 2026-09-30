#include <doctest/doctest.h>

#include <sstream>

#include "test_helpers.hpp"

using namespace sonder::inference;

namespace {
struct Collected {
    std::string text;
    std::vector<std::uint64_t> indices;
    GenerateStats stats;
    Status status;
};

Collected run(BackendModel& m, const std::string& prompt, SamplingConfig s, int stop_after = -1) {
    GenerateRequest req;
    req.request_id = "req-test";
    req.prompt = prompt;
    req.sampling = s;
    Collected c;
    auto r = m.generate(req, {}, [&](const TokenChunk& ch) {
        c.text.append(ch.text);
        c.indices.push_back(ch.index);
        return stop_after < 0 || static_cast<int>(c.indices.size()) < stop_after;
    });
    if (r.ok()) {
        c.stats = r.value();
    } else {
        c.status = r.status();
    }
    return c;
}

std::shared_ptr<BackendModel> load_mock(MockBackendOptions o = {}) {
    auto b = make_mock_backend(o);
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    return b->load_model(lo).value();
}
}  // namespace

TEST_SUITE("mock_backend") {
TEST_CASE("is clearly labelled and advertises only what it does") {
    auto b = make_mock_backend();
    CHECK(b->name() == "mock");
    CHECK(b->description().find("MOCK") != std::string::npos);
    const auto caps = b->capabilities();
    CHECK(caps.has(Capability::streaming));
    CHECK(caps.has(Capability::deterministic));
    CHECK_FALSE(caps.has(Capability::kv_export));
    CHECK_FALSE(caps.has(Capability::remote_process));
    ModelLoadOptions lo;
    lo.model = "llama3";
    CHECK(b->load_model(lo).status().code() == ErrorCode::not_found);
}

TEST_CASE("streams deterministic output") {
    auto m = load_mock();
    const auto a = run(*m, "hello world", SamplingConfig::greedy(8));
    const auto b = run(*m, "hello world", SamplingConfig::greedy(8));
    REQUIRE(a.status.ok());
    CHECK(a.text == b.text);
    CHECK_FALSE(a.text.empty());
    CHECK(a.stats.prompt_tokens == 2);
    CHECK(a.stats.completion_tokens == 8);
    CHECK(a.stats.chunks == 8);
    for (std::size_t i = 0; i < a.indices.size(); ++i) {
        CHECK(a.indices[i] == i);
    }
    const auto other = run(*m, "different prompt", SamplingConfig::greedy(8));
    CHECK(other.text != a.text);
}

TEST_CASE("greedy ignores the seed, sampling honours it") {
    auto m = load_mock();
    CHECK(run(*m, "p", SamplingConfig::greedy(12, 1)).text == run(*m, "p", SamplingConfig::greedy(12, 2)).text);
    SamplingConfig s1 = SamplingConfig::greedy(12, 1);
    s1.temperature = 0.7f;
    SamplingConfig s2 = s1;
    s2.seed = 2;
    CHECK(run(*m, "p", s1).text == run(*m, "p", s1).text);
    CHECK(run(*m, "p", s1).text != run(*m, "p", s2).text);
}

TEST_CASE("reports max_tokens versus natural end of sequence") {
    MockBackendOptions o;
    o.default_completion_tokens = 10;
    auto m = load_mock(o);
    const auto limited = run(*m, "x", SamplingConfig::greedy(4));
    CHECK(limited.stats.stop_reason == StopReason::max_tokens);
    CHECK(limited.stats.completion_tokens == 4);
    const auto natural = run(*m, "x", SamplingConfig::greedy(100));
    CHECK(natural.stats.stop_reason == StopReason::end_of_sequence);
    CHECK(natural.stats.completion_tokens == 10);
}

TEST_CASE("truncates at stop sequences") {
    auto m = load_mock();
    const auto full = run(*m, "stop test", SamplingConfig::greedy(16));
    std::istringstream words(full.text);
    std::string w0, w1, w2;
    words >> w0 >> w1 >> w2;
    REQUIRE_FALSE(w2.empty());
    SamplingConfig s = SamplingConfig::greedy(16);
    s.stop = {" " + w2};
    const auto cut = run(*m, "stop test", s);
    CHECK(cut.stats.stop_reason == StopReason::stop_sequence);
    CHECK(cut.text.find(" " + w2) == std::string::npos);
    CHECK(full.text.rfind(cut.text, 0) == 0);  // prefix of the full output
}

TEST_CASE("callback can stop the stream") {
    auto m = load_mock();
    const auto c = run(*m, "x", SamplingConfig::greedy(16), 2);
    CHECK(c.stats.stop_reason == StopReason::callback);
    CHECK(c.indices.size() == 2);
}

TEST_CASE("injected failures surface as backend errors") {
    MockBackendOptions o;
    o.fail_after_tokens = 2;
    auto m = load_mock(o);
    const auto c = run(*m, "x", SamplingConfig::greedy(16));
    CHECK(c.status.code() == ErrorCode::backend_error);
    CHECK(c.indices.size() == 2);
}
}
