#include <doctest/doctest.h>

#include "../gpu_residency.hpp"

#include <limits>

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;

namespace {
constexpr std::uint64_t MiB = 1024ull * 1024ull;
constexpr std::uint64_t GiB = 1024ull * MiB;
}

TEST_CASE("GPU layer parsing is explicit, last occurrence wins, and fail closed") {
    CHECK(expects_gpu_offload({"--n-gpu-layers", "4"}));
    CHECK(expects_gpu_offload({"--gpu-layers=all"}));
    CHECK(expects_gpu_offload({"-ngl=999"}));
    CHECK(expects_gpu_offload({"-ngl", "-1"}));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers", "0"}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers=bad"}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers="}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers"}, true));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers", "4", "-ngl", "0"}));
    CHECK_FALSE(expects_gpu_offload({"--n-gpu-layers=", "4"}));
    CHECK_FALSE(expects_gpu_offload({}));
    CHECK(expects_gpu_offload({}, true));
}

TEST_CASE("residency guard validates bounded finite configuration") {
    LlamaServerResidencyGuardOptions o;
    CHECK(validate_residency_guard(o).ok());
    o.consecutive_samples = 0;
    CHECK(validate_residency_guard(o).code() == ErrorCode::invalid_argument);
    o = {};
    o.eviction_fraction = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_fraction = std::numeric_limits<double>::infinity();
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_fraction = 1.01;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_fraction = 0;
    o.eviction_bytes = 0;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.eviction_bytes = 1;
    o.max_eviction_restarts = 33;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.max_eviction_restarts = 1;
    o.consecutive_samples = 1001;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.consecutive_samples = 3;
    o.min_dedicated_bytes = 0;
    CHECK_FALSE(validate_residency_guard(o).ok());
    o.min_dedicated_bytes = 512 * MiB;
    o.eviction_bytes = 1024ull * GiB + 1;
    CHECK_FALSE(validate_residency_guard(o).ok());
}

TEST_CASE("missing GPU offload requires consecutive low samples and latches") {
    LlamaServerResidencyGuardOptions o;
    GpuResidencyGuard g(o, true);
    g.observe(0);
    g.observe(0);
    CHECK_FALSE(g.gpu_offload_missing());
    g.observe(0);
    CHECK(g.gpu_offload_missing());
    CHECK(g.observed_dedicated_bytes() == 0);
    g.observe(2 * GiB);
    CHECK(g.gpu_offload_missing());
    g.reset();
    CHECK_FALSE(g.gpu_offload_missing());
    CHECK(g.peak_dedicated_bytes() == 0);
}

TEST_CASE("healthy samples reset low and eviction streaks") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 2;
    GpuResidencyGuard g(o, true);
    g.observe(0);
    g.observe(2 * GiB);
    g.observe(0);
    CHECK_FALSE(g.gpu_offload_missing());
    g.observe(2 * GiB);
    CHECK_FALSE(g.vram_evicted());
    g.observe(0);
    CHECK_FALSE(g.vram_evicted());
    g.unavailable();
    g.observe(0);
    CHECK_FALSE(g.vram_evicted());
}

TEST_CASE("eviction uses strict fraction or byte thresholds and high water") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 2;
    GpuResidencyGuard g(o, false);
    g.observe(8 * GiB);
    g.observe(6 * GiB);
    CHECK_FALSE(g.vram_evicted());
    g.observe(5 * GiB);
    CHECK_FALSE(g.vram_evicted());
    g.observe(5 * GiB);
    CHECK(g.vram_evicted());
    CHECK(g.peak_dedicated_bytes() == 8 * GiB);
    CHECK(g.observed_dedicated_bytes() == 5 * GiB);
}

TEST_CASE("eviction can independently disable fraction or byte dimensions") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    o.eviction_fraction = 0;
    o.eviction_bytes = 1 * GiB;
    GpuResidencyGuard bytes(o, false);
    bytes.observe(8 * GiB);
    bytes.observe(7 * GiB);
    CHECK_FALSE(bytes.vram_evicted());
    bytes.observe(6 * GiB);
    CHECK(bytes.vram_evicted());

    o.eviction_fraction = 0.25;
    o.eviction_bytes = 0;
    GpuResidencyGuard fraction(o, false);
    fraction.observe(8 * GiB);
    fraction.observe(6 * GiB);
    CHECK_FALSE(fraction.vram_evicted());
    fraction.observe(5 * GiB);
    CHECK(fraction.vram_evicted());
}

TEST_CASE("tiny peaks do not create eviction events under the floor") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, false);
    g.observe(400 * MiB);
    g.observe(0);
    CHECK_FALSE(g.vram_evicted());
    CHECK(g.peak_dedicated_bytes() == 400 * MiB);
}

TEST_CASE("missing and eviction findings may coexist and unavailable preserves latches") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, true);
    g.observe(2 * GiB);
    g.observe(0);
    CHECK(g.gpu_offload_missing());
    CHECK(g.vram_evicted());
    g.unavailable();
    CHECK(g.gpu_offload_missing());
    CHECK(g.vram_evicted());
    CHECK(g.peak_dedicated_bytes() == 2 * GiB);
}

TEST_CASE("a later eviction records its own sample after an earlier missing offload incident") {
    LlamaServerResidencyGuardOptions o;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, true);
    g.observe(0);
    REQUIRE(g.gpu_offload_missing());
    g.observe(8 * GiB);
    g.observe(3 * GiB);
    REQUIRE(g.vram_evicted());
    CHECK(g.observed_dedicated_bytes() == 3 * GiB);
}

TEST_CASE("explicit CPU intent overrides expect_gpu at detector construction") {
    LlamaServerResidencyGuardOptions o;
    o.expect_gpu = true;
    o.consecutive_samples = 1;
    GpuResidencyGuard g(o, expects_gpu_offload({"-ngl", "0"}, o.expect_gpu));
    g.observe(0);
    CHECK_FALSE(g.gpu_offload_missing());
    CHECK_FALSE(g.vram_evicted());
}
