// sonder-infer argument parsing, chat message files and the interactive chat
// loop (src/cli/cli_args.hpp), exercised with the MOCK backend.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include "cli/cli_args.hpp"
#include "test_helpers.hpp"

using namespace sonder::inference;
namespace cli = sonder::cli;

namespace {

std::optional<cli::Args> parse(std::vector<const char*> argv, std::string& error) {
    argv.insert(argv.begin(), "sonder-infer");
    return cli::parse_args(static_cast<int>(argv.size()), argv.data(), error);
}

}  // namespace

TEST_CASE("cli: parse chat subcommand with values, stops and flags") {
    std::string error;
    auto a = parse({"chat", "--backend", "mock", "--model", "mock:tiny", "--messages", "conv.json", "--system",
                    "be brief", "--stop", "User:", "--stop", "\n\n", "--capture-text", "--max-tokens", "12"},
                   error);
    REQUIRE_MESSAGE(a.has_value(), error);
    CHECK(a->command == "chat");
    CHECK(a->get("backend") == std::optional<std::string>("mock"));
    CHECK(a->get("model") == std::optional<std::string>("mock:tiny"));
    CHECK(a->get("messages") == std::optional<std::string>("conv.json"));
    CHECK(a->get("system") == std::optional<std::string>("be brief"));
    CHECK(a->get("max-tokens") == std::optional<std::string>("12"));
    CHECK_FALSE(a->get("prompt").has_value());
    REQUIRE(a->stops.size() == 2u);
    CHECK(a->stops[0] == "User:");
    CHECK(a->stops[1] == "\n\n");
    CHECK(a->has("capture-text"));
    CHECK_FALSE(a->has("markdown"));
}

TEST_CASE("cli: bench --markdown is a boolean flag anywhere in the list") {
    std::string error;
    auto a = parse({"bench", "--markdown", "--backend", "mock", "--out", "r.json"}, error);
    REQUIRE_MESSAGE(a.has_value(), error);
    CHECK(a->has("markdown"));
    CHECK(a->get("backend") == std::optional<std::string>("mock"));
    CHECK(a->get("out") == std::optional<std::string>("r.json"));

    auto b = parse({"bench", "--backend", "mock", "--markdown"}, error);
    REQUIRE_MESSAGE(b.has_value(), error);
    CHECK(b->has("markdown"));
    CHECK(cli::is_boolean_flag("--markdown"));
    CHECK(cli::is_boolean_flag("--capture-text"));
    CHECK_FALSE(cli::is_boolean_flag("--messages"));
}

TEST_CASE("cli: parse errors") {
    std::string error;
    CHECK_FALSE(parse({}, error).has_value());
    CHECK(error == "missing command");
    CHECK_FALSE(parse({"chat", "--messages"}, error).has_value());
    CHECK(error == "missing value for --messages");
    CHECK_FALSE(parse({"chat", "stray"}, error).has_value());
    CHECK(error == "unexpected argument: stray");
    CHECK_FALSE(parse({"chat", "--"}, error).has_value());
    CHECK(error == "unexpected argument: --");
    // The last value wins for repeated non-repeatable options.
    auto a = parse({"chat", "--model", "a", "--model", "b"}, error);
    REQUIRE(a.has_value());
    CHECK(a->get("model") == std::optional<std::string>("b"));
}

TEST_CASE("cli: chat message files") {
    auto obj = cli::parse_chat_messages(
        R"({"model":"x","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi \"there\""}]})");
    REQUIRE_MESSAGE(obj.ok(), obj.status().to_string());
    REQUIRE(obj->size() == 2u);
    CHECK(obj->at(0).role == "system");
    CHECK(obj->at(1).content == "hi \"there\"");

    auto arr = cli::parse_chat_messages(R"([{"role":"user","content":"one"}])");
    REQUIRE(arr.ok());
    CHECK(arr->size() == 1u);

    CHECK(cli::parse_chat_messages("not json").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"({"msgs":[]})").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"("text")").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"([{"role":"user"}])").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"([{"role":"user","content":7}])").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages("[]").status().code() == ErrorCode::invalid_argument);
    CHECK(cli::parse_chat_messages(R"([{"role":"bard","content":"x"}])").status().code() ==
          ErrorCode::invalid_argument);
    CHECK(cli::load_chat_messages("definitely/not/here.json").status().code() == ErrorCode::not_found);
}

TEST_CASE("cli: system prompt is prepended only when absent") {
    std::vector<ChatMessage> m{{"user", "hi"}};
    cli::apply_system_prompt(m, "");
    CHECK(m.size() == 1u);
    cli::apply_system_prompt(m, "be kind");
    REQUIRE(m.size() == 2u);
    CHECK(m[0].role == "system");
    CHECK(m[0].content == "be kind");
    cli::apply_system_prompt(m, "other");
    CHECK(m.size() == 2u);
    CHECK(m[0].content == "be kind");
}

TEST_CASE("cli: run_chat_turn streams the mock reply") {
    auto backend = make_mock_backend();
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto model = backend->load_model(lo).value();
    std::ostringstream out;
    auto turn = cli::run_chat_turn(*model, {{"user", "hello"}}, SamplingConfig::greedy(5), {}, out);
    REQUIRE_MESSAGE(turn.ok(), turn.status().to_string());
    CHECK_FALSE(turn->text.empty());
    CHECK(out.str() == turn->text);
    CHECK(turn->stats.completion_tokens == 5u);

    std::ostringstream none;
    CHECK(cli::run_chat_turn(*model, {}, SamplingConfig::greedy(5), {}, none).status().code() ==
          ErrorCode::invalid_argument);
    CHECK(none.str().empty());
}

TEST_CASE("cli: interactive chat loop keeps history and handles commands") {
    std::vector<std::vector<ChatMessage>> seen;
    int reply_no = 0;
    auto turn = [&](const std::vector<ChatMessage>& history) -> Result<std::string> {
        seen.push_back(history);
        if (history.back().content == "fail") return Status(ErrorCode::backend_error, "boom");
        return "reply" + std::to_string(++reply_no);
    };
    std::istringstream in("first\n\n   \nsecond\r\nfail\n/reset\nthird\n/exit\nnever\n");
    std::ostringstream out;
    std::ostringstream err;
    cli::ChatReplOptions ro;
    ro.system = "sys";
    const int rc = cli::run_chat_repl(in, out, err, ro, turn);
    CHECK(rc == 1);  // the "fail" turn
    REQUIRE(seen.size() == 4u);
    // Turn 1: system + first.
    REQUIRE(seen[0].size() == 2u);
    CHECK(seen[0][0].role == "system");
    CHECK(seen[0][1].content == "first");
    // Turn 2 sees the first exchange; CR is stripped.
    REQUIRE(seen[1].size() == 4u);
    CHECK(seen[1][2].role == "assistant");
    CHECK(seen[1][2].content == "reply1");
    CHECK(seen[1][3].content == "second");
    // The failed turn is dropped from history, then /reset clears it.
    CHECK(seen[2].size() == 6u);
    REQUIRE(seen[3].size() == 2u);
    CHECK(seen[3][0].content == "sys");
    CHECK(seen[3][1].content == "third");
    CHECK(err.str().find("boom") != std::string::npos);
    CHECK(out.str().find("[history cleared]") != std::string::npos);
    CHECK(out.str().find("> ") != std::string::npos);
}

TEST_CASE("cli: interactive chat loop with the mock backend ends cleanly at EOF") {
    auto backend = make_mock_backend();
    ModelLoadOptions lo;
    lo.model = "mock:tiny";
    auto model = backend->load_model(lo).value();
    std::istringstream in("hello\nagain\n");
    std::ostringstream out;
    std::ostringstream err;
    cli::ChatReplOptions ro;
    ro.user_prompt.clear();
    const int rc = cli::run_chat_repl(in, out, err, ro, [&](const std::vector<ChatMessage>& h) -> Result<std::string> {
        auto t = cli::run_chat_turn(*model, h, SamplingConfig::greedy(3), {}, out);
        if (!t.ok()) return t.status();
        return t->text;
    });
    CHECK(rc == 0);
    CHECK(err.str().empty());
    // Two replies, each followed by a newline.
    const std::string s = out.str();
    CHECK(std::count(s.begin(), s.end(), '\n') == 2);
}

TEST_CASE("cli: --logit-bias parsing") {
    std::vector<TokenLogitBias> out;
    std::string error;
    REQUIRE(cli::parse_logit_bias("42:-100,7:2.5,3:-inf", out, error));
    REQUIRE(out.size() == 3u);
    CHECK(out[0].token == 42);
    CHECK(out[0].bias == doctest::Approx(-100.0f));
    CHECK(out[1].token == 7);
    CHECK(out[1].bias == doctest::Approx(2.5f));
    CHECK(out[2].token == 3);
    CHECK(std::isinf(out[2].bias));
    CHECK(out[2].bias < 0.0f);
    for (const char* bad : {"", "42", ":1", "42:", "x:1", "4x:1", "1:y", "1:2,", "1:2,,3:4"}) {
        CAPTURE(bad);
        CHECK_FALSE(cli::parse_logit_bias(bad, out, error));
        CHECK_FALSE(error.empty());
    }
}
