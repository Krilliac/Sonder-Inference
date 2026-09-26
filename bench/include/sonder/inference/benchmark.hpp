// Sonder Inference: benchmark harness (docs/BENCHMARK_PLAN.md).
//
// Runs a prompt corpus through Engine sessions on any backend (ollama, mock,
// future native backends) and produces a results document
// (schema sonder.inference.bench/1): per-request rows, per-prompt and global
// distributions (TTFT, total latency, decode/prompt tokens/sec, errors),
// agent fan-out batches, and full provenance. render_markdown() turns a
// results document into a human-readable summary.
#pragma once

#include <cstdint>
#include <memory>
#include <ostream>
#include <string>
#include <string_view>
#include <vector>

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
};

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
