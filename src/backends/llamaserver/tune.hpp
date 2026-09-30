// Bounded calibration policy, separate from process/HTTP/GPU mechanisms.
#pragma once

#include <chrono>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>
#include <vector>

#include "gpu_memory.hpp"
#include "process.hpp"
#include "sonder/inference/json.hpp"

namespace sonder::inference::llamaserver::tune {

using Clock = std::chrono::steady_clock;
using Deadline = Clock::time_point;
inline constexpr std::uint64_t kMargin = 4096;
inline constexpr std::uint64_t kFastContext = 65536;

struct KvPair {
    std::string k;
    std::string v;
    bool operator==(const KvPair &) const = default;
};
struct Grid {
    std::vector<KvPair> kv{{"q4_0", "q4_0"}, {"q8_0", "q8_0"}, {"q8_0", "q4_0"}, {"q5_1", "q5_1"}};
    std::vector<std::uint64_t> ctx;
    std::vector<std::uint32_t> ubatch{512, 1024, 2048};
    std::vector<std::uint32_t> mtp{0, 1, 2, 3};
    std::uint32_t batch = 2048;
    // Spill rule for candidates without speculation.
    SpillGuardOptions spill;
    // Spill rule for MTP candidates (mtp > 0): MTP raises the clean shared
    // line to ~126 MiB + 2 MiB per 1k ctx (+~40 MiB after the first prompt),
    // measured 2026-09-30, so the fixed 0 + 256 MiB rule would call every
    // healthy MTP load near 73k spilled. Defaults: 166 MiB + 2 MiB/1k, +32.
    SpillGuardOptions spill_mtp;
    Grid();
};
struct Options {
    std::string executable;
    std::string model;
    std::string mmproj;
    std::string out = "tuned.json";
    std::vector<std::pair<std::string, std::string>> environment;
    Grid grid;
    std::chrono::milliseconds budget{std::chrono::minutes(20)};
    bool thorough = false;
    bool spec_type_supported = false; // set only after the executable's --help
};
struct Candidate {
    KvPair kv;
    std::uint64_t ctx = 0;
    std::uint32_t ubatch = 512;
    std::uint32_t mtp = 0;
    bool operator==(const Candidate &) const = default;
};
// The spill rule that applies to `candidate` (spill_mtp when it speculates).
const SpillGuardOptions &spill_guard_for(const Grid &grid, const Candidate &candidate);
enum class Phase { probe, benchmark };
enum class Verdict { clean, spill, slow_kernel, error, unsupported, timed_out, cancelled, release_failed };
const char *to_string(Verdict verdict);

struct Measurement {
    Candidate candidate;
    Phase phase = Phase::probe;
    Verdict verdict = Verdict::error;
    std::optional<std::uint64_t> dedicated_bytes;
    std::optional<std::uint64_t> shared_bytes;
    std::optional<double> short_tps;
    std::optional<double> prefill_tps;
    std::optional<double> long_prefill_tps;
    std::uint64_t served_ctx = 0;
    std::uint64_t safety_probe_ctx = 0;
    bool memory_released = false;
    bool slow_kernel = false;
    bool stop_search = false;
    std::string detail;
};
struct Report {
    std::vector<Measurement> results;
    std::optional<std::size_t> fast_default;
    std::optional<std::size_t> long_context;
    bool budget_exhausted = false;
    std::string stopped_reason;
};

// A binary search over sorted context candidates. Only clean/spill are
// monotonic observations; a protocol/kernel failure must abandon this lane.
class Bisection {
  public:
    explicit Bisection(std::vector<std::uint64_t> contexts);
    [[nodiscard]] std::optional<std::uint64_t> next() const;
    void observe(bool clean);
    [[nodiscard]] std::optional<std::uint64_t> edge() const { return edge_; }
  private:
    std::vector<std::uint64_t> contexts_;
    std::size_t lower_ = 0;
    std::size_t upper_ = 0; // exclusive
    std::optional<std::uint64_t> edge_;
};

Status validate_grid(const Grid &grid);
Result<Grid> parse_grid(const json::Value &value);
std::vector<Candidate> planned_candidates(const Grid &grid, bool mtp_available);
std::optional<std::uint64_t> margin_context(std::uint64_t measured_edge);
std::vector<std::string> arguments(const Options &options, const Candidate &candidate);
using Trial = std::function<Measurement(const Candidate &, Phase, Deadline)>;
using Now = std::function<Deadline()>;
Report search(const Options &options, bool mtp_available, Deadline deadline, const Trial &trial,
              const Now &now = [] { return Clock::now(); });
void rank(Report &report);
json::Value report_json(const Options &options, const Report &report);
std::string markdown(const Report &report);
int tune_main(const std::vector<std::string> &args, std::ostream &out, std::ostream &err);

} // namespace sonder::inference::llamaserver::tune
