#pragma once
#include "tune.hpp"
#include "sonder/inference/cancellation.hpp"

namespace sonder::inference::llamaserver::tune {
using Exchange = std::function<Result<json::Value>(std::uint16_t, const std::string &, const std::string &,
                                                  const json::Value &, Deadline, const CancellationToken &)>;
struct Dependencies {
    std::function<std::unique_ptr<ProcessLauncher>()> launcher = [] { return make_process_launcher(); };
    std::shared_ptr<GpuCounterSource> counters = make_gpu_counter_source();
    Exchange exchange; // empty selects the existing HTTP client
    std::function<bool()> interrupted = [] { return false; };
    std::chrono::milliseconds sample_interval{200};
    std::chrono::milliseconds release_timeout{10000};
};
// One child at a time, owned by the existing supervisor. All dependencies
// needed for lifecycle/workload tests are injectable; no GPU is needed.
Measurement run_candidate(const Options &options, const Candidate &candidate, Phase phase, Deadline deadline,
                          const std::string &log_path, const Dependencies &dependencies);
Result<std::string> executable_help(const Options &options, Deadline deadline, const std::string &capture_path,
                                    const Dependencies &dependencies);
} // namespace sonder::inference::llamaserver::tune
