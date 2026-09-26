// sonder-infer CLI helpers: argument parsing, chat message files, and the
// interactive chat loop. Header-only so the CLI (tools/sonder-infer) and the
// unit tests (tests/test_cli_args.cpp) share one implementation without a new
// library target.
#pragma once

#include <cstddef>
#include <fstream>
#include <functional>
#include <istream>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/cancellation.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::cli {

namespace si = sonder::inference;

// Options that take no value. Everything else starting with "--" consumes
// the next argument as its value.
inline bool is_boolean_flag(std::string_view key) noexcept {
    return key == "--capture-text" || key == "--markdown";
}

struct Args {
    std::string command;
    std::map<std::string, std::string> values;  // "--key value" (key without dashes)
    std::vector<std::string> stops;             // repeatable --stop
    std::set<std::string> flags;                // boolean flags (without dashes)

    [[nodiscard]] std::optional<std::string> get(const std::string& key) const {
        auto it = values.find(key);
        if (it == values.end()) return std::nullopt;
        return it->second;
    }
    [[nodiscard]] bool has(const std::string& flag) const { return flags.count(flag) != 0; }
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

struct ChatReplOptions {
    std::string system;           // optional system prompt
    std::string user_prompt = "> ";  // printed before each user line (empty = none)
};

// Interactive chat: each non-empty input line is a user message; the reply is
// produced by `turn` and appended to the history. Commands: /exit, /quit,
// /reset (clear history, keep the system prompt). Ends at EOF. Returns 0, or
// 1 when any turn failed (the failed user message is dropped from history).
inline int run_chat_repl(std::istream& in, std::ostream& out, std::ostream& err, const ChatReplOptions& options,
                         const ChatTurnFn& turn) {
    std::vector<si::ChatMessage> history;
    apply_system_prompt(history, options.system);
    const std::size_t base = history.size();
    int rc = 0;
    std::string line;
    for (;;) {
        if (!options.user_prompt.empty()) {
            out << options.user_prompt;
            out.flush();
        }
        if (!std::getline(in, line)) break;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.find_first_not_of(" \t") == std::string::npos) continue;
        if (line == "/exit" || line == "/quit") break;
        if (line == "/reset") {
            history.resize(base);
            out << "[history cleared]\n";
            continue;
        }
        history.push_back(si::ChatMessage{"user", line});
        auto reply = turn(history);
        out << "\n";
        if (!reply.ok()) {
            err << "error: " << reply.status().to_string() << "\n";
            history.pop_back();
            rc = 1;
            continue;
        }
        history.push_back(si::ChatMessage{"assistant", reply.value()});
    }
    return rc;
}

}  // namespace sonder::cli
