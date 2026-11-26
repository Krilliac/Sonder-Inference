// Bounded calibration policy, separate from process/HTTP/GPU mechanisms.
#pragma once

#include <chrono>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gpu_memory.hpp"
#include "process.hpp"
#include "tune_model.hpp"
#include "sonder/inference/device.hpp"
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
    // Speculation variants are applied only to speculative rows (mtp > 0).
    // The baseline (mtp=0) is emitted once and never multiplied by these axes.
    // Standalone "ngram-mod" may also run when mtp_available is false.
    std::vector<std::string> spec_type{"draft-mtp", "draft-mtp,ngram-mod"};
    std::vector<double> p_min{0.0, 0.5};
    std::vector<std::string> flash_attn{"on"};
    std::uint32_t batch = 2048;
    struct Profile {
        // Calibration requires one llama-server slot; generated profiles may
        // be edited later, but any such change requires revalidation.
        std::uint32_t parallel = 1;
        bool no_kv_unified = true;
        std::uint32_t cache_ram = 2048;
        std::uint32_t ctx_checkpoints = 8;
        std::uint64_t checkpoint_min_step = 8192;
        bool jinja = true;
        std::string reasoning_format = "deepseek";
        double min_p = 0.0;
        bool metrics = true;
    } profile;
    // Spill rule for candidates without speculation.
    SpillGuardOptions spill;
    // Spill rule for MTP candidates (mtp > 0): MTP raises the clean shared
    // line to ~126 MiB + 2 MiB per 1k ctx (+~40 MiB after the first prompt),
    // measured 2026-09-30, so the fixed 0 + 256 MiB rule would call every
    // healthy MTP load near 73k spilled. Defaults: 166 MiB + 2 MiB/1k, +32.
    SpillGuardOptions spill_mtp;
    Grid();
};
enum class UmaMode { off, on, auto_detect };
const char *to_string(UmaMode mode) noexcept;
std::optional<UmaMode> parse_uma_mode(std::string_view text) noexcept;

struct ArtifactIdentity {
    std::string path;
    std::string sha256;
    std::string architecture;
    std::string tokenizer_model;
    std::optional<std::uint64_t> vocab_size;
    std::optional<std::uint64_t> file_bytes;
    std::optional<std::uint32_t> trained_block_limit;
};
struct ExecutableIdentity {
    std::string path;
    std::string sha256;
    bool supports_spec_type = false;
    bool supports_mtp = false;
    bool supports_draft_model = false;
};

struct Options {
    std::string executable;
    std::string model;
    std::string draft_model;
    std::string target_model_sha256;
    std::string draft_model_sha256;
    std::optional<std::uint32_t> trained_block_limit;
    std::string mmproj;
    std::string out = "tuned.json";
    std::vector<std::pair<std::string, std::string>> environment;
    Grid grid;
    std::chrono::milliseconds budget{std::chrono::minutes(20)};
    bool thorough = false;
    bool spec_type_supported = false;   // set only after the executable's --help
    bool draft_model_supported = false; // set only after the executable's --help
    UmaMode uma_mode = UmaMode::off;
    bool emit_adaptive_speculation = false;
    // Optional UMA host-RAM pool ceiling override in bytes (0 = auto-detect from host RAM and UMA GPU pool).
    std::uint64_t uma_pool_ceiling_bytes = 0;
    std::optional<ArtifactIdentity> target_identity;
    std::optional<ArtifactIdentity> draft_identity;
    std::optional<ExecutableIdentity> executable_identity;
};
struct Candidate {
    KvPair kv;
    std::uint64_t ctx = 0;
    std::uint32_t ubatch = 512;
    std::uint32_t mtp = 0;
    std::string spec_type = "draft-mtp";
    double p_min = 0.0;
    std::string flash_attn = "on";
    bool operator==(const Candidate &other) const noexcept {
        if (kv != other.kv || ctx != other.ctx || ubatch != other.ubatch || mtp != other.mtp ||
            flash_attn != other.flash_attn)
            return false;
        if (mtp == 0) return true;
        if (spec_type != other.spec_type) return false;
        if (spec_type == "ngram-mod") return true;
        return p_min == other.p_min;
    }
};
// True when `candidate` uses multi-token prediction weights/buffers
// (draft-mtp or draft-mtp,ngram-mod), as opposed to baseline, standalone
// ngram-mod, or external/diffusion draft speculation.
inline bool uses_mtp(const Candidate &candidate) noexcept {
    return candidate.mtp != 0 &&
           (candidate.spec_type == "draft-mtp" || candidate.spec_type == "draft-mtp,ngram-mod");
}
// True when `spec_type` selects an external standalone draft model
// (as opposed to embedded target MTP weights or n-gram/suffix self-drafting).
inline bool uses_draft_model(std::string_view spec_type) noexcept {
    return spec_type == "dflash" || spec_type == "draft-dflash" || spec_type == "draft-eagle3" ||
           spec_type == "draft" || spec_type == "draft-simple" || spec_type == "draft-dspark";
}
// True when `candidate` uses draft-probability speculation flags
// (`--spec-draft-n-max` / `--spec-draft-p-min`), i.e. any non-zero draft
// length except standalone `ngram-mod`.
inline bool uses_draft_p_min(const Candidate &candidate) noexcept {
    return candidate.mtp != 0 && candidate.spec_type != "ngram-mod";
}
// The spill rule that applies to `candidate` (spill_mtp when it uses MTP).
const SpillGuardOptions &spill_guard_for(const Grid &grid, const Candidate &candidate);
enum class Phase { probe, benchmark };
enum class Verdict { clean, spill, slow_kernel, error, unsupported, timed_out, cancelled, release_failed };
const char *to_string(Verdict verdict);

struct WorkloadMeasurement {
    std::string name;
    double weight = 0.0;
    double tok_s = 0.0;
    std::optional<std::uint64_t> draft_n;
    std::optional<std::uint64_t> draft_n_accepted;
};

// Returns the fixed .4/.3/.2/.1 weighted decode rate when exactly the four
// natural workload classes are present and valid.  Missing or invalid classes
// make the candidate ineligible for recommendation.
std::optional<double> weighted_decode_tps(const std::vector<WorkloadMeasurement> &workloads);

struct Measurement {
    Candidate candidate;
    Phase phase = Phase::probe;
    Verdict verdict = Verdict::error;
    std::optional<std::uint64_t> dedicated_bytes;
    std::optional<std::uint64_t> shared_bytes;
    std::optional<double> short_tps;
    std::optional<double> prefill_tps;
    std::optional<double> long_prefill_tps;
    std::vector<WorkloadMeasurement> workloads;
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
    std::optional<ArtifactIdentity> target_identity;
    std::optional<ArtifactIdentity> draft_identity;
    std::optional<ExecutableIdentity> executable_identity;
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
Status validate_options_for_grid(const Options &options, bool dry_run = false);
Status validate_draft_pair_compatibility(const ModelInfo &target, const ModelInfo &draft);
struct PreparedArtifacts {
    bool mtp_available = false;
    ModelInfo target_info;
    std::optional<ModelInfo> draft_info;
};
// Reads, hashes, and validates model (and optional draft_model) GGUFs plus
// pair compatibility and trained_block_limit bounds before launching processes.
Status prepare_model_and_draft_artifacts(Options &options, PreparedArtifacts &out,
                                         const std::function<bool()> &cancelled = [] { return false; });
// Checks executable --help capabilities and computes executable provenance
// against the already-inspected target/draft artifacts and configured grid.
Status apply_executable_capabilities(Options &options, PreparedArtifacts &artifacts,
                                     std::string_view executable_help_text,
                                     const std::function<bool()> &cancelled = [] { return false; });
// Pre-grid eligibility and artifact identity verification: runs both steps above.
Result<PreparedArtifacts> prepare_live_artifacts(Options &options, std::string_view executable_help_text,
                                                 const std::function<bool()> &cancelled = [] { return false; });
Result<Grid> parse_grid(const json::Value &value);
Grid grid_for_gpu(const GpuDeviceInfo &gpu, bool is_ternary = false);
bool resolve_tune_uma_mode(UmaMode mode, const std::vector<GpuDeviceInfo> &gpus,
                           const std::vector<std::string> &args = {}) noexcept;
std::optional<std::uint64_t> uma_pool_ceiling_bytes(const HostMemory &host,
                                                    const std::vector<GpuDeviceInfo> &gpus = {}) noexcept;
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
bool help_supports_spec_type(std::string_view help);
int tune_main(const std::vector<std::string> &args, std::ostream &out, std::ostream &err);

} // namespace sonder::inference::llamaserver::tune
