#include "tune.hpp"
#include "tune_model.hpp"
#include "tune_runner.hpp"
#include "utf8_path.hpp"
#include "cli/cli_args.hpp"
#include "cli/cli_values.hpp"
#include "cli/sonder_infer_commands.hpp"

#include <atomic>
#include <algorithm>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <stdexcept>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace sonder::inference::llamaserver::tune {
namespace {
std::atomic<int> interrupted{0};
static_assert(std::atomic<int>::is_always_lock_free);
extern "C" void tune_interrupt(int) { interrupted.store(1); }
class Signals {
  public:
    Signals() {
        interrupted.store(0);
        old_int_ = std::signal(SIGINT, tune_interrupt);
        old_term_ = std::signal(SIGTERM, tune_interrupt);
    }
    ~Signals() {
        if (old_int_ != SIG_ERR) std::signal(SIGINT, old_int_);
        if (old_term_ != SIG_ERR) std::signal(SIGTERM, old_term_);
    }
  private:
    using Handler = void (*)(int);
    Handler old_int_ = SIG_DFL, old_term_ = SIG_DFL;
};
class Scratch {
  public:
    explicit Scratch(const std::filesystem::path &parent) {
        static std::atomic<unsigned> sequence{0};
        for (int attempt = 0; attempt < 100; ++attempt) {
            auto path = parent / (".sonder-tune-" + std::to_string(Clock::now().time_since_epoch().count()) +
                                  "-" + std::to_string(sequence.fetch_add(1)));
            if (std::filesystem::create_directory(path)) { path_ = std::move(path); return; }
        }
        throw std::runtime_error("cannot create unique calibration directory");
    }
    ~Scratch() { std::error_code ec; std::filesystem::remove_all(path_, ec); }
    std::string file(const char *name) const {
        const auto text = (path_ / name).u8string();
        return std::string(text.begin(), text.end());
    }
  private:
    std::filesystem::path path_;
};
Result<std::string> read_small_file(const std::string &path) {
    std::ifstream in(utf8_path(path), std::ios::binary);
    if (!in) return Status(ErrorCode::not_found, "cannot open grid file");
    std::string text(65537, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(in.gcount()));
    if (in.bad() || text.size() > 65536) return Status(ErrorCode::invalid_argument, "grid exceeds 64 KiB");
    return text;
}
void write_output(const std::string &destination, const std::string &text, const Scratch &scratch) {
    const auto temporary = utf8_path(scratch.file("output.json"));
    const auto target = utf8_path(destination);
    {
        std::ofstream file(temporary, std::ios::binary | std::ios::trunc);
        if (!file) throw std::runtime_error("cannot create calibration output");
        file << text << '\n';
        file.close();
        if (!file) throw std::runtime_error("cannot finish writing calibration output");
    }
    // Scratch is in the output directory: replacement stays on the same volume
    // and a failed write cannot truncate an existing usable configuration.
#if defined(_WIN32)
    if (!MoveFileExW(temporary.c_str(), target.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot replace calibration output");
#else
    std::filesystem::rename(temporary, target);
#endif
}
} // namespace

int tune_main(const std::vector<std::string> &args, std::ostream &out, std::ostream &err) {
    namespace cli = sonder::cli;
    const auto usage_error = [&](const std::string &message) {
        err << "error: tune: " << message << '\n';
        return cli::kExitUsage;
    };
    try {
        std::string error;
        const auto parsed = cli::parse_command_args(cli::tune_command(), args, error);
        // Match the CLI's help-wins convention even alongside malformed options.
        if ((parsed && parsed->help) || std::find(args.begin(), args.end(), "--help") != args.end() ||
            std::find(args.begin(), args.end(), "-h") != args.end()) {
            out << cli::render_help(cli::kProgram, cli::tune_command());
            return cli::kExitOk;
        }
        if (!parsed) return usage_error(error);
        Options options;
        options.executable = parsed->get("executable").value_or("");
        options.model = parsed->get("model").value_or("");
        options.mmproj = parsed->get("mmproj").value_or("");
        options.out = parsed->get("out").value_or("tuned.json");
        options.thorough = parsed->has("thorough");
        if (options.executable.empty() || options.model.empty() || options.out.empty())
            return usage_error("--executable and --model are required; paths must not be empty");
        for (const auto *path : {&options.executable, &options.model, &options.mmproj, &options.out})
            if (path->find('\0') != std::string::npos) return usage_error("paths must not contain NUL");
        if (auto budget = parsed->get("budget-minutes")) {
            double minutes = 0;
            if (!cli::parse_number("budget-minutes", *budget, 0.1, 1440, minutes, error)) return usage_error(error);
            options.budget = std::chrono::milliseconds(static_cast<std::int64_t>(minutes * 60000));
        }
        for (const auto &entry : parsed->all("env")) {
            const auto split = entry.find('=');
            if (split == std::string::npos) return usage_error("--env expects KEY=VALUE");
            options.environment.emplace_back(entry.substr(0, split), entry.substr(split + 1));
        }
        if (auto status = validate_process_environment(options.environment); !status.ok()) return usage_error(status.message());
        if (auto grid_arg = parsed->get("grid")) {
            auto text = *grid_arg;
            const auto first = text.find_first_not_of(" \t\r\n");
            if (first == std::string::npos || text[first] != '{') {
                auto read = read_small_file(!text.empty() && text.front() == '@' ? text.substr(1) : text);
                if (!read.ok()) return usage_error(read.status().message());
                text = std::move(read.value());
            }
            if (text.size() > 65536) return usage_error("grid exceeds 64 KiB");
            auto json = json::parse(text);
            if (!json.ok()) return usage_error(json.status().message());
            auto grid = parse_grid(json.value());
            if (!grid.ok()) return usage_error(grid.status().message());
            options.grid = std::move(grid.value());
        }
        if (parsed->has("dry-run")) {
            out << "# Planned calibration search space\n\n"
                   "Adaptive bisection probes context first; only leaders receive throughput measurements.\n"
                   "MTP >0 rows are conditional on nextn tensors and executable --help; neither is probed here.\n"
                   "Benchmarks also include 65536 and clean-edge minus 4096; every recommendation gets a +4096 safety probe.\n\n"
                   "| K/V | ctx | ubatch | MTP | spec_type | p_min |\n|---|---:|---:|---:|---|---:|\n";
            for (const auto &c : planned_candidates(options.grid, true))
                out << "| " << c.kv.k << '/' << c.kv.v << " | " << c.ctx << " | " << c.ubatch << " | " << c.mtp
                    << " | " << (c.mtp == 0 ? "none" : c.spec_type) << " | " << c.p_min << " |\n";
            return cli::kExitOk;
        }
        // Platform eligibility precedes --help or any model/server process.
        Dependencies dependencies;
        if (!dependencies.counters->supported()) {
            err << "error: tune: live calibration requires Windows GPU Process Memory counters; use --dry-run here\n";
            return cli::kExitFailure;
        }
        Signals signals;
        dependencies.interrupted = [] { return interrupted.load() != 0; };
        const auto deadline = Clock::now() + options.budget;
        std::ifstream model(utf8_path(options.model), std::ios::binary);
        if (!model) return usage_error("cannot open model GGUF");
        auto info = read_model_info(model, [&] { return dependencies.interrupted() || Clock::now() >= deadline; });
        if (!info.ok()) {
            if (dependencies.interrupted()) return cli::kExitCancelled;
            if (Clock::now() >= deadline) { err << "error: tune: budget exhausted inspecting GGUF\n"; return cli::kExitFailure; }
            return usage_error(info.status().message());
        }
        model.close();
        if (!options.mmproj.empty() && !std::filesystem::is_regular_file(utf8_path(options.mmproj)))
            return usage_error("mmproj is not a readable local file");
        for (const auto &destination : {options.out, options.out + ".results.json"}) {
            for (const auto &input : {options.model, options.executable, options.mmproj}) {
                if (input.empty()) continue;
                std::error_code ec;
                if (std::filesystem::equivalent(utf8_path(destination), utf8_path(input), ec) && !ec)
                    return usage_error("output must not replace a model, projector or executable");
            }
        }
        auto parent = utf8_path(options.out).parent_path();
        if (parent.empty()) parent = std::filesystem::current_path();
        Scratch scratch(parent);
        auto help = executable_help(options, deadline, scratch.file("help.log"), dependencies);
        if (!help.ok()) {
            err << "error: tune: " << help.status().message() << '\n';
            return dependencies.interrupted() ? cli::kExitCancelled : cli::kExitFailure;
        }
        const bool mtp = info.value().has_nextn && help_supports_mtp(help.value());
        options.spec_type_supported = help_supports_mtp(help.value());
        err << "[tune] architecture=" << sonder::inference::to_string(info.value().architecture)
            << "; MTP=" << (mtp ? "eligible" : "disabled (needs nextn tensors and --help support)") << '\n';
        auto report = search(options, mtp, deadline, [&](const Candidate &c, Phase phase, Deadline until) {
            err << "[tune] " << (phase == Phase::probe ? "probe " : "bench ") << c.kv.k << '/' << c.kv.v
                << " ctx=" << c.ctx << " ubatch=" << c.ubatch << " mtp=" << c.mtp
                << " spec_type=" << (c.mtp == 0 ? "none" : c.spec_type) << " p_min=" << c.p_min << '\n';
            return run_candidate(options, c, phase, until, scratch.file("candidate.log"), dependencies);
        });
        out << markdown(report);
        const bool has_recommendation = report.fast_default || report.long_context;
        // Never replace an existing usable profile with an empty/failed run.
        const auto destination = has_recommendation ? options.out : options.out + ".results.json";
        const auto document = report_json(options, report).dump();
        if (document.size() >= 1024 * 1024) throw std::runtime_error("calibration output exceeds the config loader's 1 MiB bound");
        write_output(destination, document, scratch);
        err << "[tune] wrote " << destination << '\n';
        if (dependencies.interrupted()) return cli::kExitCancelled;
        return has_recommendation && report.stopped_reason.empty() ? cli::kExitOk : cli::kExitFailure;
    } catch (const std::exception &e) {
        err << "error: tune: " << e.what() << '\n';
        return cli::kExitFailure;
    }
}
} // namespace sonder::inference::llamaserver::tune
