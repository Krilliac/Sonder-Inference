// Sonder Inference: benchmark harness skeleton (docs/BENCHMARK_PLAN.md).
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
    std::string workload;  // e.g. interactive_chat, coding, long_context
    std::string prompt;
    std::int32_t max_tokens = 64;
};

struct Corpus {
    std::string name;
    std::string description;
    std::vector<Prompt> prompts;
};

Result<Corpus> parse_corpus(std::string_view json_text);
Result<Corpus> load_corpus(const std::string& path);

struct Options {
    int warmup_runs = 1;    // per prompt, excluded from summary
    int measured_runs = 3;  // per prompt
    SamplingConfig sampling = SamplingConfig::greedy();
    std::string label;      // free-form note recorded in the results
};

// Runs every prompt through a fresh session on `model` and returns a results
// document (schema sonder.inference.bench/1). Progress lines go to `progress`
// when non-null. Individual request failures are recorded, not fatal.
json::Object run(Engine& engine, const std::shared_ptr<Model>& model, const Corpus& corpus, const Options& options,
                 std::ostream* progress = nullptr);

// Percentile with linear interpolation; p in [0, 100]. Empty input -> 0.
double percentile(std::vector<double> values, double p);

}  // namespace sonder::inference::bench
