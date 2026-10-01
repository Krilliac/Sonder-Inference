#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// Internal include kept separate so supervisor integration can depend on the
// policy without making implementation details part of the public contract.
#include "sonder/inference/backends/llamaserver_residency.hpp"
#include "sonder/inference/error.hpp"

namespace sonder::inference::llamaserver {

Status validate_residency_guard(const LlamaServerResidencyGuardOptions &options);
bool expects_gpu_offload(const std::vector<std::string> &args, bool assume_gpu = false);

class GpuResidencyGuard {
  public:
    explicit GpuResidencyGuard(LlamaServerResidencyGuardOptions options = {}, bool gpu_expected = false);
    void reset();
    void unavailable();
    void observe(std::uint64_t dedicated_bytes);
    [[nodiscard]] bool gpu_offload_missing() const noexcept { return gpu_offload_missing_; }
    [[nodiscard]] bool vram_evicted() const noexcept { return vram_evicted_; }
    [[nodiscard]] std::uint64_t peak_dedicated_bytes() const noexcept { return peak_dedicated_bytes_; }
    [[nodiscard]] std::uint64_t observed_dedicated_bytes() const noexcept { return observed_dedicated_bytes_; }

  private:
    LlamaServerResidencyGuardOptions options_;
    bool gpu_expected_ = false;
    bool gpu_offload_missing_ = false;
    bool vram_evicted_ = false;
    bool event_observed_ = false;
    std::size_t missing_streak_ = 0;
    std::size_t eviction_streak_ = 0;
    std::uint64_t peak_dedicated_bytes_ = 0;
    std::uint64_t observed_dedicated_bytes_ = 0;
};

}  // namespace sonder::inference::llamaserver
