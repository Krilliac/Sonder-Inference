// `sonder-infer weights manifest|verify` (docs/integration/weight-manifest.md).
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iterator>
#include <ostream>
#include <random>

#include "sonder/inference/weight_manifest.hpp"

namespace sonder::inference::weights {

namespace fs = std::filesystem;

namespace {

constexpr int kExitOk = 0;
constexpr int kExitIo = 1;
constexpr int kExitMismatch = 2;
constexpr int kExitMissing = 3;
constexpr int kExitUsage = 4;
constexpr int kExitCancelled = 130;
constexpr std::uint64_t kMaxJsonBytes = 256ull << 20;

const char* kUsage =
    "usage: sonder-infer weights manifest <dir> -o <manifest.json> [--expected <hf-tree.json>] [--quiet]\n"
    "       sonder-infer weights verify <dir> <manifest.json> [--sample N [--seed S]] [--full] [--quiet]\n"
    "\n"
    "manifest  hash every file under <dir> (SHA-256, streaming) and write a versioned JSON manifest.\n"
    "          --expected FILE  a Hugging Face tree-API response saved locally (LFS oids become\n"
    "                           expected sha256 values); a mismatch is reported and exits 2.\n"
    "          -o FILE          an existing FILE is replaced only if it is a weight manifest; links,\n"
    "                           directories and any other file are refused (exit 4).\n"
    "verify    compare <dir> with a manifest. Always checks presence, size, extra files, links and\n"
    "          the manifest's own expected hashes; hashes content only with:\n"
    "          --sample N   hash up to N randomly chosen files (--seed S makes the choice repeatable;\n"
    "                       files already reported missing or size-mismatched are not hashed;\n"
    "                       the seed used is printed)\n"
    "          --full       hash every file\n"
    "Symlinks, reparse points and paths escaping <dir> are refused. Progress goes to stderr.\n"
    "Exit status: 0 ok, 1 I/O failure, 2 mismatch, 3 missing or extra file, 4 usage error or\n"
    "invalid manifest, 130 cancelled. A mismatch outranks a missing or extra file.\n";

int usage_error(std::ostream& err, const std::string& message) {
    err << "error: weights: " << printable_path(message) << "\n" << kUsage;
    return kExitUsage;
}

int status_exit(std::ostream& err, const Status& s) {
    err << "error: weights: " << s.message() << "\n";
    switch (s.code()) {
        case ErrorCode::invalid_argument:
            return kExitUsage;
        case ErrorCode::cancelled:
            return kExitCancelled;
        default:
            return kExitIo;
    }
}

// `shown` is the argument as the user typed it, used (escaped) in messages.
Result<std::string> read_text(const fs::path& path, const std::string& shown) {
    std::error_code ec;
    const std::uint64_t size = fs::file_size(path, ec);
    if (ec) return Status(ErrorCode::io_error, "cannot read '" + printable_path(shown) + "': " + ec.message());
    if (size > kMaxJsonBytes) {
        return Status(ErrorCode::invalid_argument, "'" + printable_path(shown) + "' is too large");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status(ErrorCode::io_error, "cannot open '" + printable_path(shown) + "'");
    return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

bool parse_u64(const std::string& s, std::uint64_t& out) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos) return false;
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (errno != 0 || end != s.c_str() + s.size()) return false;
    out = v;
    return true;
}

// Prints at most one progress line per ~2 s per run, plus a line when a file finishes.
class Progress {
public:
    Progress(std::ostream& err, bool quiet) : err_(err), quiet_(quiet), last_(std::chrono::steady_clock::now()) {}
    ProgressFn fn() {
        return [this](std::string_view path, std::uint64_t done, std::uint64_t total) {
            if (quiet_) return;
            const auto now = std::chrono::steady_clock::now();
            const bool finished = done >= total;
            if (!finished && now - last_ < std::chrono::seconds(2)) return;
            last_ = now;
            err_ << "hashing " << printable_path(path) << ": " << done << "/" << total << " bytes";
            if (total > 0) err_ << " (" << (done * 100 / total) << "%)";
            err_ << "\n";
        };
    }

private:
    std::ostream& err_;
    bool quiet_;
    std::chrono::steady_clock::time_point last_;
};

struct Parsed {
    std::vector<std::string> positional;
    std::optional<std::string> expected;
    std::optional<std::string> out;
    std::optional<std::string> sample;
    std::optional<std::string> seed;
    bool full = false;
    bool quiet = false;
};

// Returns an error message, or empty on success.
std::string parse_flags(const std::vector<std::string>& args, bool is_manifest, Parsed& p) {
    for (std::size_t i = 1; i < args.size(); ++i) {
        std::string name = args[i];
        std::optional<std::string> inline_value;
        if (name.rfind("--", 0) == 0) {
            if (const auto eq = name.find('='); eq != std::string::npos) {
                inline_value = name.substr(eq + 1);
                name.erase(eq);
            }
        }
        auto value = [&](std::optional<std::string>& slot) -> std::string {
            if (slot) return "option " + name + " given twice";
            if (inline_value) {
                slot = *inline_value;
            } else if (i + 1 < args.size()) {
                slot = args[++i];
            } else {
                return "option " + name + " needs a value";
            }
            return {};
        };
        std::string e;
        if (name == "-o" || name == "--out") {
            if (!is_manifest) return "unknown option " + name;
            e = value(p.out);
        } else if (name == "--expected") {
            if (!is_manifest) return "unknown option " + name;
            e = value(p.expected);
        } else if (name == "--sample") {
            if (is_manifest) return "unknown option " + name;
            e = value(p.sample);
        } else if (name == "--seed") {
            if (is_manifest) return "unknown option " + name;
            e = value(p.seed);
        } else if (name == "--full") {
            if (is_manifest) return "unknown option " + name;
            p.full = true;
        } else if (name == "--quiet") {
            p.quiet = true;
        } else if (name.rfind("-", 0) == 0 && name.size() > 1) {
            return "unknown option " + name;
        } else {
            p.positional.push_back(args[i]);
        }
        if (!e.empty()) return e;
    }
    return {};
}

// Refuses to write over anything that is not a regular file holding a weight manifest, so a typo or a
// link cannot destroy a weight shard. Not atomic: a writer racing the tool can still swap the target
// between this check and the open (TOCTOU).
Status check_output_target(const fs::path& out, const std::string& shown) {
    std::error_code ec;
    const fs::file_status st = fs::symlink_status(out, ec);
    if (st.type() == fs::file_type::not_found) return Status::success();
    if (ec) return Status(ErrorCode::io_error, "cannot stat '" + printable_path(shown) + "': " + ec.message());
    if (st.type() != fs::file_type::regular) {
        return Status(ErrorCode::invalid_argument,
                      "refusing to overwrite '" + printable_path(shown) + "': not a regular file");
    }
    auto text = read_text(out, shown);
    if (!text.ok() || !parse_manifest(text.value()).ok()) {
        return Status(ErrorCode::invalid_argument, "refusing to overwrite '" + printable_path(shown) +
                                                       "': it exists and is not a weight manifest (remove it first)");
    }
    return Status::success();
}

int run_manifest(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    Parsed p;
    if (std::string e = parse_flags(args, true, p); !e.empty()) return usage_error(err, e);
    if (p.positional.size() != 1) return usage_error(err, "manifest takes exactly one <dir>");
    if (!p.out) return usage_error(err, "manifest needs -o <file>");
    const fs::path dir = fs::path(p.positional[0]);
    const fs::path out_path = fs::path(*p.out);

    if (Status s = check_output_target(out_path, *p.out); !s.ok()) return status_exit(err, s);

    BuildOptions options;
    options.ignore.push_back(out_path);
    if (p.expected) {
        auto text = read_text(fs::path(*p.expected), *p.expected);
        if (!text.ok()) return status_exit(err, text.status());
        auto expected = parse_expected_json(text.value());
        if (!expected.ok()) return status_exit(err, expected.status());
        options.expected = std::move(expected).value();
    }
    Progress progress(err, p.quiet);
    options.hash.progress = progress.fn();
    auto built = build_manifest(dir, options);
    if (!built.ok()) return status_exit(err, built.status());

    if (Status s = check_output_target(out_path, *p.out); !s.ok()) return status_exit(err, s);
    {
        std::ofstream file(out_path, std::ios::binary | std::ios::trunc);
        if (!file) {
            err << "error: weights: cannot write '" << printable_path(*p.out) << "'\n";
            return kExitIo;
        }
        file << manifest_to_json(built->manifest);
        file.flush();
        if (!file) {
            err << "error: weights: write failed for '" << printable_path(*p.out) << "'\n";
            return kExitIo;
        }
    }
    std::uint64_t bytes = 0;
    for (const auto& f : built->manifest.files) bytes += f.size;
    out << "manifest: " << built->manifest.files.size() << " files, " << bytes << " bytes -> " << printable_path(*p.out) << "\n";
    for (const auto& path : built->expected_mismatch) out << "expected-mismatch: " << printable_path(path) << "\n";
    for (const auto& path : built->expected_missing) out << "missing: " << printable_path(path) << " (listed in --expected, absent from directory)\n";
    if (!built->expected_mismatch.empty()) return kExitMismatch;
    if (!built->expected_missing.empty()) return kExitMissing;
    return kExitOk;
}

int run_verify(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    Parsed p;
    if (std::string e = parse_flags(args, false, p); !e.empty()) return usage_error(err, e);
    if (p.positional.size() != 2) return usage_error(err, "verify takes <dir> <manifest>");
    if (p.full && p.sample) return usage_error(err, "--full and --sample are mutually exclusive");
    if (p.seed && !p.sample) return usage_error(err, "--seed needs --sample");

    VerifyOptions options;
    options.full = p.full;
    if (p.sample) {
        std::uint64_t n = 0;
        if (!parse_u64(*p.sample, n)) return usage_error(err, "--sample needs a non-negative integer");
        options.sample = static_cast<std::size_t>(n);
        std::uint64_t seed = 0;
        if (p.seed) {
            if (!parse_u64(*p.seed, seed)) return usage_error(err, "--seed needs a non-negative integer");
        } else {
            std::random_device rd;
            seed = (static_cast<std::uint64_t>(rd()) << 32) ^ static_cast<std::uint64_t>(rd());
        }
        options.seed = seed;
    }
    const fs::path dir = fs::path(p.positional[0]);
    const fs::path manifest_path = fs::path(p.positional[1]);
    options.ignore.push_back(manifest_path);

    auto text = read_text(manifest_path, p.positional[1]);
    if (!text.ok()) return status_exit(err, text.status());
    auto manifest = parse_manifest(text.value());
    if (!manifest.ok()) return status_exit(err, manifest.status());

    Progress progress(err, p.quiet);
    options.hash.progress = progress.fn();
    auto report = verify_manifest(dir, manifest.value(), options);
    if (!report.ok()) return status_exit(err, report.status());
    const VerifyReport& r = report.value();

    if (options.sample) out << "sample: seed " << r.seed << ", " << r.hashed.size() << " of " << r.files_total << " files hashed\n";
    for (const auto& path : r.hashed) {
        if (options.sample) out << "sampled: " << printable_path(path) << "\n";
    }
    for (const auto& m : r.mismatches) {
        out << "mismatch: " << printable_path(m.path) << " (" << m.kind << " expected " << m.expected << ", actual " << m.actual << ")\n";
    }
    for (const auto& path : r.missing) out << "missing: " << printable_path(path) << "\n";
    for (const auto& path : r.extra) out << "extra: " << printable_path(path) << "\n";
    for (const auto& path : r.unsafe) out << "unsafe: " << printable_path(path) << " (symlink, reparse point or special entry)\n";
    const int rc = r.exit_code();
    if (rc == kExitOk) {
        out << "verify: OK (" << r.files_total << " files checked, " << r.hashed.size() << " hashed";
        if (!options.full && !options.sample) out << "; sizes and names only, pass --sample N or --full to hash";
        out << ")\n";
    } else {
        out << "verify: FAILED\n";
    }
    return rc;
}

int dispatch(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    if (args.empty()) return usage_error(err, "expected 'manifest' or 'verify'");
    const std::string& sub = args.front();
    if (sub == "--help" || sub == "-h" || sub == "help") {
        out << kUsage;
        return kExitOk;
    }
    for (const auto& a : args) {
        if (a == "--help" || a == "-h") {
            out << kUsage;
            return kExitOk;
        }
    }
    if (sub == "manifest") return run_manifest(args, out, err);
    if (sub == "verify") return run_verify(args, out, err);
    return usage_error(err, "unknown subcommand '" + printable_path(sub) + "'");
}

}  // namespace

int weights_main(const std::vector<std::string>& args, std::ostream& out, std::ostream& err) {
    try {
        return dispatch(args, out, err);
    } catch (const std::exception& e) {
        err << "error: weights: " << printable_path(e.what()) << "\n";
        return kExitIo;
    }
}

}  // namespace sonder::inference::weights
