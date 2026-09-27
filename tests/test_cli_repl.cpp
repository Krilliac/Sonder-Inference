// sonder-infer chat REPL (cli::run_chat_session in src/cli/cli_args.hpp):
// commands, labels, colors, stats, cancelled turns, and chat turns through
// Session::chat with request telemetry, on the MOCK backend.
#include <doctest/doctest.h>

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "cli/cli_args.hpp"
#include "cli/cli_values.hpp"
#include "test_helpers.hpp"

using namespace sonder::inference;
namespace cli = sonder::cli;

namespace {

cli::ChatTurnReport canned_report(const std::string& text, std::uint64_t prompt_tokens, std::uint64_t completion) {
    cli::ChatTurnReport r;
    r.text = text;
    r.stats.command = "chat";
    r.stats.backend = "mock";
    r.stats.model = "mock:tiny";
    r.stats.synthetic = true;
    r.stats.request_id = "req-" + text;
    r.stats.outcome = "completed";
    r.stats.stop_reason = "end_of_sequence";
    r.stats.prompt_tokens = prompt_tokens;
    r.stats.completion_tokens = completion;
    return r;
}

std::size_t count(const std::string& haystack, const std::string& needle) {
    std::size_t n = 0;
    for (auto pos = haystack.find(needle); pos != std::string::npos; pos = haystack.find(needle, pos + 1)) ++n;
    return n;
}

}  // namespace

TEST_CASE("cli repl: /help, /stats, unknown commands and the // escape") {
    std::vector<std::vector<ChatMessage>> seen;
    auto turn = [&](const std::vector<ChatMessage>& history) -> Result<cli::ChatTurnReport> {
        seen.push_back(history);
        return canned_report("r" + std::to_string(seen.size()), 3, 5);
    };
    std::istringstream in("/stats\nhello\n/help\n/bogus arg\n//literal\n/stats\n/quit\nnever\n");
    std::ostringstream out, err;
    cli::ChatReplOptions ro;
    ro.user_prompt.clear();
    CHECK(cli::run_chat_session(in, out, err, ro, turn) == 0);
    REQUIRE(seen.size() == 2u);
    CHECK(seen[0].back().content == "hello");
    CHECK(seen[1].back().content == "/literal");  // "//" sends a leading "/"
    REQUIRE(seen[1].size() == 3u);
    CHECK(seen[1][1].content == "r1");

    const std::string o = out.str();
    CHECK(o.find("turns=0 failed=0 cancelled=0 prompt_tokens=0 completion_tokens=0\nlast: no completed turn yet\n") !=
          std::string::npos);
    CHECK(o.find("/stats         last turn and session totals") != std::string::npos);
    CHECK(o.find("turns=2 failed=0 cancelled=0 prompt_tokens=6 completion_tokens=10\n") != std::string::npos);
    CHECK(o.find("last: [sonder-infer] chat turn=2 outcome=completed") != std::string::npos);
    CHECK(err.str() == "unknown command /bogus (/help lists commands; start with // to send a leading /)\n");
    CHECK(o.find("never") == std::string::npos);
}

TEST_CASE("cli repl: labels, colors and no dangling prompt at EOF") {
    auto turn = [](const std::vector<ChatMessage>&) -> Result<cli::ChatTurnReport> {
        return canned_report("reply", 1, 1);
    };
    {
        std::istringstream in("hi\n");
        std::ostringstream out, err;
        cli::ChatReplOptions ro;
        ro.user_prompt = "you> ";
        ro.assistant_label = "assistant> ";
        CHECK(cli::run_chat_session(in, out, err, ro, turn) == 0);
        // The turn function streams the reply itself; the loop prints the
        // label before it and a newline after it, then a final newline for the
        // prompt left open at EOF.
        CHECK(out.str() == "you> assistant> \nyou> \n");
        CHECK(out.str().find("\x1b[") == std::string::npos);
        CHECK(err.str().empty());
    }
    {
        std::istringstream in("hi\n");
        std::ostringstream out, err;
        cli::ChatReplOptions ro;
        ro.user_prompt = "you> ";
        ro.assistant_label = "assistant> ";
        ro.color = true;
        CHECK(cli::run_chat_session(in, out, err, ro, turn) == 0);
        CHECK(out.str() == "\x1b[1;36myou> \x1b[0m\x1b[1;32massistant> \x1b[0m\n\x1b[1;36myou> \x1b[0m\n");
    }
    {
        // /exit right after a prompt: the typed line already ended it.
        std::istringstream in("/exit\n");
        std::ostringstream out, err;
        cli::ChatReplOptions ro;
        ro.user_prompt = "> ";
        CHECK(cli::run_chat_session(in, out, err, ro, turn) == 0);
        CHECK(out.str() == "> ");
    }
    {
        // Non-interactive: no prompt, output ends with the reply's newline.
        std::istringstream in("a\nb");
        std::ostringstream out, err;
        cli::ChatReplOptions ro;
        ro.user_prompt.clear();
        CHECK(cli::run_chat_session(in, out, err, ro, turn) == 0);
        CHECK(out.str() == "\n\n");
    }
}

TEST_CASE("cli repl: per-turn stats, failed and cancelled turns") {
    int n = 0;
    std::vector<std::size_t> sizes;
    auto turn = [&](const std::vector<ChatMessage>& history) -> Result<cli::ChatTurnReport> {
        sizes.push_back(history.size());
        ++n;
        if (history.back().content == "fail") return Status(ErrorCode::backend_error, "boom");
        auto r = canned_report("t" + std::to_string(n), 2, 3);
        r.cancelled = history.back().content == "cancel";
        return r;
    };
    std::istringstream in("one\nfail\ncancel\ntwo\n/stats\n");
    std::ostringstream out, err;
    cli::ChatReplOptions ro;
    ro.user_prompt.clear();
    ro.system = "sys";
    ro.stats = cli::StatsMode::json;
    CHECK(cli::run_chat_session(in, out, err, ro, turn) == 1);  // one failed turn
    // History: sys+one, sys+one+t1+fail (dropped), sys+one+t1+cancel (dropped), sys+one+t1+two.
    CHECK(sizes == std::vector<std::size_t>{2, 4, 4, 4});
    CHECK(out.str().find("turns=2 failed=1 cancelled=1 prompt_tokens=4 completion_tokens=6") != std::string::npos);

    // err: two JSON stats lines (turn 1 and 2), the error, and the cancel note.
    std::istringstream lines(err.str());
    std::string line;
    std::vector<std::string> json_lines;
    while (std::getline(lines, line)) {
        if (!line.empty() && line.front() == '{') json_lines.push_back(line);
    }
    REQUIRE(json_lines.size() == 2u);
    for (std::size_t i = 0; i < json_lines.size(); ++i) {
        auto j = json::parse(json_lines[i]);
        REQUIRE(j.ok());
        CHECK(j.value().find("turn")->as_int() == static_cast<std::int64_t>(i + 1));
        CHECK(j.value().find("command")->as_string() == "chat");
    }
    CHECK(err.str().find("error: backend_error: boom") != std::string::npos);
    CHECK(err.str().find("[turn cancelled]") != std::string::npos);

    // Text stats.
    std::istringstream in2("x\n");
    std::ostringstream out2, err2;
    ro.stats = cli::StatsMode::text;
    CHECK(cli::run_chat_session(in2, out2, err2, ro, turn) == 0);
    CHECK(err2.str().rfind("[sonder-infer] chat turn=1 outcome=completed", 0) == 0);
}

TEST_CASE("cli repl: turns through Session::chat emit chat request telemetry") {
    sonder_test::Harness h;
    REQUIRE(h.model);
    auto session = h.session(SamplingConfig::greedy(6));
    REQUIRE(session);
    std::ostringstream out, err;
    auto turn = [&](const std::vector<ChatMessage>& history) -> Result<cli::ChatTurnReport> {
        auto res = session->chat(history, [&](const TokenChunk& c) {
            out << c.text;
            return true;
        });
        if (!res.ok()) return res.status();
        cli::ChatTurnReport r;
        r.cancelled = res.value().outcome == RequestOutcome::cancelled;
        r.stats = cli::make_request_stats("chat", "mock", "mock:tiny", true, res.value());
        r.text = res.value().text;
        return r;
    };
    std::istringstream in("hello\n/stats\nagain\n");
    cli::ChatReplOptions ro;
    ro.user_prompt.clear();
    CHECK(cli::run_chat_session(in, out, err, ro, turn) == 0);
    CHECK(err.str().empty());
    CHECK(out.str().find("turns=1 failed=0 cancelled=0 prompt_tokens=") != std::string::npos);

    const auto queued = h.events_of("request.queued");
    REQUIRE(queued.size() == 2u);
    for (const auto& e : queued) {
        CHECK(e.find("attributes")->find("kind")->as_string() == "chat");
    }
    CHECK(h.events_of("request.completed").size() == 2u);
    CHECK(count(out.str(), "\n") >= 2u);
}
