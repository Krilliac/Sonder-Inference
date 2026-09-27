// sonder-infer CLI helpers: argument parsing, chat message files, and the
// interactive chat loop. Header-only so the CLI (tools/sonder-infer) and the
// unit tests (tests/test_cli_args.cpp) share one implementation without a new
// library target. Per-command option specs live in cli_spec.hpp, checked
// values and stats in cli_values.hpp (docs/CLI.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <functional>
#include <istream>
#include <limits>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

#include "cli_spec.hpp"
#include "cli_values.hpp"

namespace sonder::cli {

namespace si = sonder::inference;

// Generic parse_args() only: options that take no value. Everything else
// starting with "--" consumes the next argument as its value. The sonder-infer
// commands use per-command specs instead (parse_command_args()).
inline bool is_boolean_flag(std::string_view key) noexcept {
    return key == "--capture-text" || key == "--markdown" || key == "--json" || key == "--quiet" ||
           key == "--ollama-allow-remote";
}

struct Args {
    std::string command;
    std::map<std::string, std::string> values;  // "--key value" (key without dashes)
    std::vector<std::string> stops;             // repeatable --stop
    std::set<std::string> flags;                // boolean flags (without dashes)
    std::map<std::string, std::vector<std::string>> lists;  // other repeatable options, in order
    bool help = false;                          // -h / --help (parse_command_args only)

    [[nodiscard]] std::optional<std::string> get(const std::string& key) const {
        auto it = values.find(key);
        if (it == values.end()) return std::nullopt;
        return it->second;
    }
    [[nodiscard]] bool has(const std::string& flag) const { return flags.count(flag) != 0; }
    [[nodiscard]] std::vector<std::string> all(const std::string& key) const {
        if (key == "stop") return stops;
        auto it = lists.find(key);
        return it == lists.end() ? std::vector<std::string>{} : it->second;
    }
};

inline std::optional<Args> parse_args(int argc, const char* const* argv, std::string& error) {
    Args a;
    if (argc < 2) {
        error = "missing command";
        return std::nullopt;
    }
    a.command = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key.rfind("--", 0) != 0 || key.size() == 2) {
            error = "unexpected argument: " + key;
            return std::nullopt;
        }
        if (is_boolean_flag(key)) {
            a.flags.insert(key.substr(2));
            continue;
        }
        if (i + 1 >= argc) {
            error = "missing value for " + key;
            return std::nullopt;
        }
        std::string value = argv[++i];
        if (key == "--stop") {
            a.stops.push_back(std::move(value));
        } else {
            a.values[key.substr(2)] = std::move(value);
        }
    }
    return a;
}

// Spec-driven parsing for one command: `args` excludes the program name and
// the command word. Unknown options fail with a did-you-mean suggestion
// (cli_spec.hpp); --stop fills `stops`, other repeatable options `lists`.
// `error` is not prefixed with the command name.
inline std::optional<Args> parse_command_args(const CommandSpec& spec, const std::vector<std::string>& args,
                                              std::string& error) {
    ParsedCommand parsed;
    if (!parse_command(spec, args, parsed, error)) {
        return std::nullopt;
    }
    Args a;
    a.command = std::string(spec.name);
    a.values = std::move(parsed.values);
    a.flags = std::move(parsed.flags);
    a.help = parsed.help;
    for (auto& [key, list] : parsed.lists) {
        if (key == "stop") {
            a.stops = std::move(list);
        } else {
            a.lists[key] = std::move(list);
        }
    }
    return a;
}

// --logit-bias value: comma-separated TOKEN:BIAS pairs, e.g. "42:-100,7:2.5".
// BIAS may be "-inf" to ban a token. Range checks (token >= 0, unique tokens,
// bias in [-100, 100] or -inf) are left to si::validate().
inline bool parse_logit_bias(std::string_view text, std::vector<si::TokenLogitBias>& out, std::string& error) {
    out.clear();
    std::size_t pos = 0;
    while (pos <= text.size()) {
        const std::size_t comma = text.find(',', pos);
        const std::string_view item =
            text.substr(pos, comma == std::string_view::npos ? std::string_view::npos : comma - pos);
        const std::size_t colon = item.find(':');
        if (item.empty() || colon == std::string_view::npos || colon == 0 || colon + 1 == item.size()) {
            error = "invalid --logit-bias entry '" + std::string(item) + "' (expected TOKEN:BIAS)";
            return false;
        }
        const std::string token_text(item.substr(0, colon));
        const std::string bias_text(item.substr(colon + 1));
        si::TokenLogitBias b;
        try {
            std::size_t used = 0;
            const long long token = std::stoll(token_text, &used);
            if (used != token_text.size() || token < std::numeric_limits<std::int32_t>::min() ||
                token > std::numeric_limits<std::int32_t>::max()) {
                throw std::invalid_argument("token");
            }
            b.token = static_cast<std::int32_t>(token);
            if (bias_text == "-inf") {
                b.bias = -std::numeric_limits<float>::infinity();
            } else {
                b.bias = std::stof(bias_text, &used);
                if (used != bias_text.size()) {
                    throw std::invalid_argument("bias");
                }
            }
        } catch (const std::exception&) {
            error = "invalid --logit-bias entry '" + std::string(item) + "' (expected TOKEN:BIAS)";
            return false;
        }
        out.push_back(b);
        if (comma == std::string_view::npos) {
            break;
        }
        pos = comma + 1;
    }
    return true;
}

// Chat message file: either a JSON array of {"role", "content"} objects or an
// object with a "messages" array (the Ollama/OpenAI request shape). Messages
// are validated with si::validate_chat_messages().
inline si::Result<std::vector<si::ChatMessage>> parse_chat_messages(std::string_view json_text) {
    auto doc = si::json::parse(json_text);
    if (!doc.ok()) {
        return si::Status(si::ErrorCode::invalid_argument, "messages: " + doc.status().message());
    }
    const si::json::Value* list = &doc.value();
    if (list->is_object()) {
        list = list->find("messages");
        if (list == nullptr) {
            return si::Status(si::ErrorCode::invalid_argument, "messages: object has no \"messages\" array");
        }
    }
    if (!list->is_array()) {
        return si::Status(si::ErrorCode::invalid_argument, "messages: expected an array of {role, content}");
    }
    std::vector<si::ChatMessage> out;
    for (std::size_t i = 0; i < list->as_array().size(); ++i) {
        const si::json::Value& m = list->as_array()[i];
        const si::json::Value* role = m.find("role");
        const si::json::Value* content = m.find("content");
        if (role == nullptr || !role->is_string() || content == nullptr || !content->is_string()) {
            return si::Status(si::ErrorCode::invalid_argument,
                              "messages: entry " + std::to_string(i) + " needs string \"role\" and \"content\"");
        }
        out.push_back(si::ChatMessage{role->as_string(), content->as_string()});
    }
    if (si::Status st = si::validate_chat_messages(out); !st.ok()) {
        return st;
    }
    return out;
}

inline si::Result<std::vector<si::ChatMessage>> load_chat_messages(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return si::Status(si::ErrorCode::not_found, "messages: cannot open " + path);
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_chat_messages(ss.str());
}

// Prepends a system message unless the conversation already starts with one.
inline void apply_system_prompt(std::vector<si::ChatMessage>& messages, const std::string& system) {
    if (system.empty() || (!messages.empty() && messages.front().role == "system")) return;
    messages.insert(messages.begin(), si::ChatMessage{"system", system});
}

struct ChatTurnResult {
    std::string text;
    si::GenerateStats stats;
};

// Runs one chat turn on `model`, streaming the reply to `out` as it arrives.
inline si::Result<ChatTurnResult> run_chat_turn(si::BackendModel& model, const std::vector<si::ChatMessage>& messages,
                                                const si::SamplingConfig& sampling,
                                                const si::CancellationToken& cancel, std::ostream& out,
                                                const std::string& request_id = "chat") {
    si::ChatRequest req;
    req.request_id = request_id;
    req.messages = messages;
    req.sampling = sampling;
    ChatTurnResult turn;
    auto res = model.chat(req, cancel, [&](const si::TokenChunk& c) {
        turn.text.append(c.text);
        out.write(c.text.data(), static_cast<std::streamsize>(c.text.size()));
        out.flush();
        return true;
    });
    if (!res.ok()) {
        return res.status();
    }
    turn.stats = res.value();
    return turn;
}

// One assistant turn over the full history; returns the reply text.
using ChatTurnFn = std::function<si::Result<std::string>(const std::vector<si::ChatMessage>& history)>;

// One assistant turn with its statistics (run_chat_session()).
struct ChatTurnReport {
    std::string text;
    RequestStats stats;
    bool cancelled = false;  // the turn was cancelled (Ctrl-C); partial text is discarded
};
using ChatReportTurnFn = std::function<si::Result<ChatTurnReport>(const std::vector<si::ChatMessage>& history)>;

struct ChatReplOptions {
    std::string system;              // optional system prompt
    std::string user_prompt = "> ";  // printed before each user line (empty = none)
    std::string assistant_label;     // printed before each reply (empty = none)
    bool color = false;              // ANSI-color the prompt and label
    StatsMode stats = StatsMode::none;  // per-turn stats line on `err`
};

inline constexpr std::string_view kChatReplHelp =
    "commands:\n"
    "  /help          this list\n"
    "  /stats         last turn and session totals\n"
    "  /reset         clear the history (the system prompt stays)\n"
    "  /exit, /quit   end the chat (EOF also ends it)\n"
    "  //text         send \"/text\" as a message\n"
    "Ctrl-C cancels the current turn.\n";

namespace repl_detail {
inline std::string paint(const std::string& text, const char* ansi, bool color) {
    return color ? std::string(ansi) + text + "\x1b[0m" : text;
}
}  // namespace repl_detail

// Interactive chat: each non-empty input line is a user message; the reply is
// produced by `turn`, streamed by it to `out`, and appended to the history.
// Commands: see kChatReplHelp; any other line starting with a single '/' is
// rejected (not sent). A failed turn drops the user message from history; a
// cancelled turn drops it and the partial reply. Ends at EOF or /exit without
// leaving a dangling prompt. Returns 0, or 1 when any turn failed.
inline int run_chat_session(std::istream& in, std::ostream& out, std::ostream& err, const ChatReplOptions& options,
                            const ChatReportTurnFn& turn) {
    std::vector<si::ChatMessage> history;
    apply_system_prompt(history, options.system);
    const std::size_t base = history.size();
    int rc = 0;
    std::uint64_t turns = 0, failed = 0, cancelled = 0, prompt_tokens = 0, completion_tokens = 0;
    std::optional<RequestStats> last;
    std::string line;
    for (;;) {
        bool prompt_pending = false;
        if (!options.user_prompt.empty()) {
            out << repl_detail::paint(options.user_prompt, "\x1b[1;36m", options.color);
            out.flush();
            prompt_pending = true;
        }
        if (!std::getline(in, line)) {
            if (prompt_pending) {
                out << "\n";
                out.flush();
            }
            break;
        }
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        std::string message = line;
        if (line.rfind("//", 0) == 0) {
            message = line.substr(1);
        } else if (line.front() == '/') {
            const std::string command = line.substr(0, line.find_first_of(" \t"));
            if (command == "/exit" || command == "/quit") break;
            if (command == "/reset") {
                history.resize(base);
                out << "[history cleared]\n";
            } else if (command == "/help") {
                out << kChatReplHelp;
            } else if (command == "/stats") {
                out << "turns=" << turns << " failed=" << failed << " cancelled=" << cancelled
                    << " prompt_tokens=" << prompt_tokens << " completion_tokens=" << completion_tokens << "\n";
                if (last) {
                    out << "last: " << format_stats_text(*last) << "\n";
                } else {
                    out << "last: no completed turn yet\n";
                }
            } else {
                err << "unknown command " << command << " (/help lists commands; start with // to send a leading /)\n";
            }
            out.flush();
            continue;
        }
        history.push_back(si::ChatMessage{"user", message});
        if (!options.assistant_label.empty()) {
            out << repl_detail::paint(options.assistant_label, "\x1b[1;32m", options.color);
            out.flush();
        }
        auto reply = turn(history);
        out << "\n";
        out.flush();
        if (!reply.ok()) {
            err << "error: " << reply.status().to_string() << "\n";
            history.pop_back();
            ++failed;
            rc = 1;
            continue;
        }
        if (reply.value().cancelled) {
            err << "[turn cancelled]\n";
            history.pop_back();
            ++cancelled;
            continue;
        }
        ++turns;
        RequestStats st = reply.value().stats;
        st.turn = turns;
        prompt_tokens += st.prompt_tokens;
        completion_tokens += st.completion_tokens;
        if (options.stats == StatsMode::text) {
            err << format_stats_text(st) << "\n";
        } else if (options.stats == StatsMode::json) {
            err << format_stats_json(st) << "\n";
        }
        err.flush();
        last = std::move(st);
        history.push_back(si::ChatMessage{"assistant", std::move(reply.value().text)});
    }
    return rc;
}

// Text-only variant (no stats); same commands and behaviour.
inline int run_chat_repl(std::istream& in, std::ostream& out, std::ostream& err, const ChatReplOptions& options,
                         const ChatTurnFn& turn) {
    return run_chat_session(in, out, err, options,
                            [&turn](const std::vector<si::ChatMessage>& history) -> si::Result<ChatTurnReport> {
                                auto text = turn(history);
                                if (!text.ok()) return text.status();
                                ChatTurnReport report;
                                report.text = std::move(text.value());
                                report.stats.command = "chat";
                                return report;
                            });
}

}  // namespace sonder::cli
