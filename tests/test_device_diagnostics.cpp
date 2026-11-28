// Shared helpers must remain usable without the optional llamaserver module.
#include <doctest/doctest.h>

#include "devices/gpu_memory.hpp"
#include "devices/log_diagnostics.hpp"

using namespace sonder::inference::llamaserver;

TEST_CASE("device counters filter process instances and sum dedicated and shared usage") {
    GpuProcessCounters counters;
    counters.dedicated = {{"pid_42_adapter_0", 100}, {"pid_42_adapter_1", 200},
                          {"pid_420_adapter_0", 999}};
    counters.shared = {{"pid_42_adapter_0", 10}, {"pid_42_adapter_1", 20},
                       {"pid_7_adapter_0", 999}};
    const auto sample = summarize_gpu_counters(42, counters);
    CHECK(sample.dedicated_bytes == 300);
    CHECK(sample.shared_bytes == 30);
    CHECK(sample.instances == 2);
}

TEST_CASE("device diagnostics preserve chunk boundaries and bounded warning aggregation") {
    LogDiagnostics parser;
    parser.feed("ggml_vulkan: VK_ERROR_DEVICE_");
    CHECK(parser.warnings().empty());
    parser.feed("LOST\r\nCUDA error: out of memory\nCUDA error: out of memory");
    parser.finish();
    REQUIRE(parser.warnings().size() == 2);
    CHECK(parser.warnings()[0].code == "vulkan_runtime_error");
    CHECK(parser.warnings()[1].code == "gpu_oom");
    CHECK(parser.warnings()[1].count == 2);
    CHECK(parser.lines() == 3);

    parser.reset();
    for (unsigned i = 0; i < 100; ++i) {
        parser.feed("offloaded " + std::to_string(i) + "/200 layers to GPU\n");
    }
    CHECK(parser.warnings().size() == LogDiagnostics::kMaxWarnings);
}
