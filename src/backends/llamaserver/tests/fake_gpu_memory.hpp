// Shared injectable counter source for spill-guard and calibration tests.
#pragma once
#include "../gpu_memory.hpp"
#include <atomic>
#include <functional>
#include <utility>

namespace sonder_test {
class FakeCounters final : public sonder::inference::llamaserver::GpuCounterSource {
  public:
    using Fn = std::function<sonder::inference::Result<sonder::inference::llamaserver::GpuProcessCounters>(std::uint32_t)>;
    explicit FakeCounters(Fn fn, bool is_supported = true) : fn_(std::move(fn)), supported_(is_supported) {}
    std::string name() const override { return "fake"; }
    bool supported() const override { return supported_; }
    sonder::inference::Result<sonder::inference::llamaserver::GpuProcessCounters> read(std::uint32_t pid) override {
        reads.fetch_add(1);
        return fn_(pid);
    }
    std::atomic<unsigned> reads{0};
  private:
    Fn fn_;
    bool supported_;
};
} // namespace sonder_test
