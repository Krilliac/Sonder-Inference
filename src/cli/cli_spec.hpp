// sonder-infer CLI helpers: per-command option specs, did-you-mean
// suggestions, spec-driven parsing and generated help text. Header-only and
// dependent on the standard library only, so the CLI (tools/sonder-infer),
// sonder-bench, the serve module's argument parser and the unit tests
// (tests/test_cli_spec.cpp) share one implementation. Reference: docs/CLI.md.
#pragma once

#include <algorithm>
#include <cstddef>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace sonder::cli {

// How an option consumes the command line.
enum class OptionKind {
    flag,    // boolean: present or absent, takes no value
    value,   // takes one value; giving it twice is a usage error
    repeat,  // takes one value per occurrence; all values are kept in order
};

struct OptionSpec {
    std::string_view name;        // without the leading "--"
    OptionKind kind = OptionKind::value;
    std::string_view value_name;  // e.g. "N", "PATH"; empty for flags
    std::string_view help;        // one line; may contain '\n' for continuation lines
};

struct OptionGroup {
    std::string_view title;
    std::vector<OptionSpec> options;
};

struct CommandSpec {
    std::string_view name;
    std::string_view summary;               // one line for the command list
    std::vector<std::string_view> synopsis;  // usage lines after "<program> "
    std::vector<OptionGroup> groups;
    std::vector<std::string_view> notes;     // free text paragraphs after the options
    std::vector<std::string_view> examples;  // command lines after "<program> "

    [[nodiscard]] const OptionSpec* find(std::string_view option) const {
        for (const auto& g : groups) {
            for (const auto& o : g.options) {
                if (o.name == option) {
                    return &o;
                }
            }
        }
        return nullptr;
    }
    [[nodiscard]] std::vector<std::string_view> option_names() const {
        std::vector<std::string_view> out;
        for (const auto& g : groups) {
            for (const auto& o : g.options) {
                out.push_back(o.name);
            }
        }
        return out;
    }
};

// Optimal string alignment distance (Levenshtein plus adjacent
// transpositions), so "--max-token" is 1 away from "--max-tokens" and
// "--tpo-k" is 1 away from "--top-k".
inline std::size_t edit_distance(std::string_view a, std::string_view b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::vector<std::size_t>> d(n + 1, std::vector<std::size_t>(m + 1, 0));
    for (std::size_t i = 0; i <= n; ++i) d[i][0] = i;
    for (std::size_t j = 0; j <= m; ++j) d[0][j] = j;
    for (std::size_t i = 1; i <= n; ++i) {
        for (std::size_t j = 1; j <= m; ++j) {
            const std::size_t cost = a[i - 1] == b[j - 1] ? 0 : 1;
            d[i][j] = std::min({d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost});
            if (i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]) {
                d[i][j] = std::min(d[i][j], d[i - 2][j - 2] + 1);
            }
        }
    }
    return d[n][m];
}

// The closest candidate within `max_distance` edits, or nothing. A candidate
// must also be closer than the length of `word` (so "x" never suggests "y").
// Ties go to the earliest candidate.
inline std::optional<std::string> suggest(std::string_view word, const std::vector<std::string_view>& candidates,
                                          std::size_t max_distance = 2) {
    std::optional<std::string> best;
    std::size_t best_distance = max_distance + 1;
    for (const auto c : candidates) {
        const std::size_t d = edit_distance(word, c);
        if (d < best_distance && d < word.size()) {
            best_distance = d;
            best = std::string(c);
        }
    }
    return best;
}

// "unknown option --max-token (did you mean --max-tokens?)" for a command
// whose options are `candidates` (names without dashes).
inline std::string unknown_option_message(std::string_view option, const std::vector<std::string_view>& candidates) {
    std::string message = "unknown option --" + std::string(option);
    if (auto s = suggest(option, candidates)) {
        message += " (did you mean --" + *s + "?)";
    }
    return message;
}

// Result of parsing one command's arguments against its spec.
struct ParsedCommand {
    std::map<std::string, std::string> values;               // OptionKind::value
    std::map<std::string, std::vector<std::string>> lists;   // OptionKind::repeat, in order
    std::set<std::string> flags;                             // OptionKind::flag
    bool help = false;                                       // -h / --help seen

    [[nodiscard]] std::optional<std::string> get(const std::string& key) const {
        auto it = values.find(key);
        if (it == values.end()) return std::nullopt;
        return it->second;
    }
    [[nodiscard]] std::vector<std::string> all(const std::string& key) const {
        auto it = lists.find(key);
        return it == lists.end() ? std::vector<std::string>{} : it->second;
    }
    [[nodiscard]] bool has(const std::string& flag) const { return flags.count(flag) != 0; }
};

// Parses `args` (everything after the command word) against `spec`.
// Accepts "--name value", "--name=value", boolean "--flag", and -h/--help
// anywhere (sets `help`; the other arguments are still checked). Errors
// (returned false, `error` set, never prefixed with the command name):
//   unknown option (with a did-you-mean suggestion), positional argument,
//   missing value, a value on a flag, a non-repeatable option given twice.
inline bool parse_command(const CommandSpec& spec, const std::vector<std::string>& args, ParsedCommand& out,
                          std::string& error) {
    out = ParsedCommand{};
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string& arg = args[i];
        if (arg == "-h" || arg == "--help") {
            out.help = true;
            continue;
        }
        if (arg.rfind("--", 0) != 0 || arg.size() == 2) {
            error = "unexpected argument '" + arg + "'";
            return false;
        }
        std::string key = arg.substr(2);
        std::optional<std::string> inline_value;
        if (const auto eq = key.find('='); eq != std::string::npos) {
            inline_value = key.substr(eq + 1);
            key.resize(eq);
        }
        const OptionSpec* opt = spec.find(key);
        if (opt == nullptr) {
            error = unknown_option_message(key, spec.option_names());
            return false;
        }
        if (opt->kind == OptionKind::flag) {
            if (inline_value) {
                error = "--" + key + " takes no value";
                return false;
            }
            out.flags.insert(key);
            continue;
        }
        std::string value;
        if (inline_value) {
            value = std::move(*inline_value);
        } else {
            if (i + 1 >= args.size()) {
                error = "missing value for --" + key;
                return false;
            }
            value = args[++i];
        }
        if (opt->kind == OptionKind::repeat) {
            out.lists[key].push_back(std::move(value));
            continue;
        }
        if (out.values.count(key) != 0) {
            error = "--" + key + " given more than once";
            return false;
        }
        out.values.emplace(key, std::move(value));
    }
    return true;
}

// Command-specific help: usage, grouped options, notes and examples. Usage
// and example lines are printed as "<program> <line>".
inline std::string render_help(std::string_view program, const CommandSpec& spec) {
    std::string out;
    // A spec without a name describes a standalone program (sonder-bench).
    out += std::string(program) + (spec.name.empty() ? "" : " " + std::string(spec.name)) + " - " +
           std::string(spec.summary) + "\n\nUsage:\n";
    for (const auto line : spec.synopsis) {
        out += "  " + std::string(program) + " " + std::string(line) + "\n";
    }
    std::size_t width = 0;
    for (const auto& g : spec.groups) {
        for (const auto& o : g.options) {
            width = std::max(width, 2 + o.name.size() + (o.value_name.empty() ? 0 : 1 + o.value_name.size()));
        }
    }
    width = std::min<std::size_t>(width, 28);
    for (const auto& g : spec.groups) {
        if (g.options.empty()) continue;
        out += "\n" + std::string(g.title) + ":\n";
        for (const auto& o : g.options) {
            std::string left = "--" + std::string(o.name);
            if (!o.value_name.empty()) left += " " + std::string(o.value_name);
            if (o.kind == OptionKind::repeat) left += "...";
            std::string line = "  " + left;
            const std::string pad(width + 4, ' ');
            if (line.size() < pad.size()) {
                line.append(pad.size() - line.size(), ' ');
            } else {
                line += "\n" + pad;
            }
            // Continuation lines of the help text are indented to the column.
            std::string_view help = o.help;
            std::size_t start = 0;
            bool first = true;
            while (start <= help.size()) {
                const std::size_t nl = help.find('\n', start);
                const std::string_view part =
                    help.substr(start, nl == std::string_view::npos ? std::string_view::npos : nl - start);
                line += (first ? "" : "\n" + pad) + std::string(part);
                first = false;
                if (nl == std::string_view::npos) break;
                start = nl + 1;
            }
            out += line + "\n";
        }
    }
    for (const auto note : spec.notes) {
        out += "\n" + std::string(note) + "\n";
    }
    if (!spec.examples.empty()) {
        out += "\nExamples:\n";
        for (const auto ex : spec.examples) {
            out += "  " + std::string(program) + " " + std::string(ex) + "\n";
        }
    }
    return out;
}

// Short command list ("  name  summary") for the overview and usage errors.
inline std::string render_command_list(const std::vector<const CommandSpec*>& commands) {
    std::size_t width = 0;
    for (const auto* c : commands) width = std::max(width, c->name.size());
    std::string out;
    for (const auto* c : commands) {
        out += "  " + std::string(c->name) + std::string(width - c->name.size() + 2, ' ') + std::string(c->summary) +
               "\n";
    }
    return out;
}

}  // namespace sonder::cli
