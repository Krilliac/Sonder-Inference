// Sonder Inference: benchmark harness (docs/BENCHMARK_PLAN.md).
//
// Runs a prompt corpus through Engine sessions on any backend (ollama, mock,
// future native backends) and produces a results document
// (schema sonder.inference.bench/1): per-request rows, per-prompt and global
// distributions (TTFT, total latency, decode/prompt tokens/sec, errors),
// agent fan-out batches, and full provenance. render_markdown() turns a
// results document into a human-readable summary.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/device.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/sampling.hpp"

namespace sonder::inference::bench {

inline constexpr std::string_view kResultSchema = "sonder.inference.bench/1";
inline constexpr std::string_view kCorpusSchema = "sonder.inference.corpus/1";

struct Prompt {
    std::string id;
    std::string workload;  // e.g. interactive_chat, coding, long_context, agent_fanout
    std::string prompt;    // full prompt; for agent_fanout the shared prefix
    std::int32_t max_tokens = 64;
    // agent_fanout: full child prompts (shared prefix + task), issued
    // concurrently, one session each. Empty for single-request prompts.
    std::vector<std::string> children{};
    std::size_t shared_prefix_bytes = 0;

    [[nodiscard]] bool is_fanout() const noexcept { return !children.empty(); }
};

// Corpus JSON (schema sonder.inference.corpus/1):
//   { "schema", "name", "description", "version",
//     "fillers": { "<name>": "<text>" },                 // optional
//     "prompts": [ { "id", "workload", "prompt", "max_tokens",
//                    "context": {"filler": "<name>", "repeat": N},  // optional, prepended
//                    "shared_prefix": "...",                          // optional, prepended
//                    "children": ["task", ...] } ] }                  // agent fan-out
struct Corpus {
    std::string name;
    std::string description;
    std::string version;
    std::string fnv1a;  // hex hash of the corpus bytes (provenance)
    std::vector<Prompt> prompts;
};

// Largest corpus JSON accepted (checked before reading or parsing it).
inline constexpr std::size_t kMaxCorpusFileBytes = 64u << 20;

// Bounds for Options::concurrency (and `sonder-bench --concurrency`).
inline constexpr int kMinConcurrency = 1;
inline constexpr int kMaxConcurrency = 64;

Result<Corpus> parse_corpus(std::string_view json_text);
Result<Corpus> load_corpus(const std::string& path);

struct Options {
    int warmup_runs = 1;    // per prompt, excluded from summary
    int measured_runs = 3;  // per prompt
    SamplingConfig sampling = SamplingConfig::greedy();
    std::string label;      // free-form note recorded in the results
    std::string hardware;   // free-form hardware description (GPU/VRAM/CPU/RAM/driver)
    std::vector<std::string> prompt_ids;  // run only these prompt ids (empty = all)
    // Stop issuing new requests after this many seconds (0 = unlimited). The
    // document then carries "truncated_by_budget": true.
    double budget_seconds = 0.0;
    // Session metadata applied to every request (envelope run_id, agent_id,
    // task_id; scheduler workload class and priority). run_id defaults to
    // `label` when unset.
    std::optional<std::string> run_id;
    std::optional<std::string> agent_id;
    std::optional<std::string> task_id;
    WorkloadClass workload = WorkloadClass::implementation_worker;
    int priority = 0;
    // Parallel worker sessions per measured iteration for non-fanout prompts
    // (clamped to [kMinConcurrency, kMaxConcurrency], i.e. 1..64). Each worker
    // uses its own fresh Session so true aggregate decode throughput and
    // per-request TTFT/total latency under load are measured.
    int concurrency = 1;
    // Capture lightweight enumerate_gpu_devices() snapshots before and after
    // the benchmark loop into top-level `gpus_before` and `gpus_after` arrays
    // (with per-GPU `peak_vram_delta_mib` when free VRAM dropped).
    bool capture_gpu_snapshot = true;
};

// Serializes a GPU inventory snapshot into the `gpus_before` / `gpus_after`
// JSON array shape. When `before` is non-null, adds `peak_vram_delta_mib` to
// each GPU whose known free VRAM dropped relative to the matching GPU in
// `before`.
json::Array gpu_snapshot_to_json(const std::vector<GpuDeviceInfo>& gpus,
                                 const std::vector<GpuDeviceInfo>* before = nullptr);

// Parsed command-line arguments for the standalone `sonder-bench` runner.
struct BenchCliArgs {
    bool help = false;
    std::string backend = "ollama";
    std::string ollama_url;
    std::string model;
    std::string corpus_path = "bench/corpus/baseline.json";
    std::string out_dir = "bench/results";
    std::string stem;
    std::string telemetry_path;
    bool list_models = false;
    bool dry_run = false;
    bool require_idle = false;
    std::uint64_t seed = 42;
    Options options;
};

// Generated help text for `sonder-bench`.
std::string render_bench_cli_help();

// Parses `sonder-bench` argv (excluding argv[0]) against its option spec and
// validates numeric ranges (including --concurrency 1..64 and --no-gpu-snapshot).
// Returns true on success (or when -h/--help is present, setting out.help = true);
// returns false and writes a diagnostic into `error` on usage errors.
bool parse_bench_cli_args(const std::vector<std::string>& args, BenchCliArgs& out, std::string& error);

// Runs every prompt through fresh sessions on `model` and returns a results
// document (schema sonder.inference.bench/1). Progress lines go to `progress`
// when non-null. Individual request failures are recorded, not fatal.
json::Object run(Engine& engine, const std::shared_ptr<Model>& model, const Corpus& corpus, const Options& options,
                 std::ostream* progress = nullptr);

// Percentile with linear interpolation; p in [0, 100]. Empty input -> 0.
double percentile(std::vector<double> values, double p);

// Markdown summary of a results document produced by run().
std::string render_markdown(const json::Value& results);

// "<YYYY-MM-DD>-<backend>-<model>-<corpus>" with filesystem-safe characters.
std::string default_result_stem(const json::Value& results);

// Writes <dir>/<stem>.json and <dir>/<stem>.md (creates dir).
Status write_results(const json::Value& results, const std::string& dir, const std::string& stem,
                     std::string* json_path = nullptr, std::string* markdown_path = nullptr);

}  // namespace sonder::inference::bench
