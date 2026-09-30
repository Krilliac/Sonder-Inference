// GPU-free policy and concurrency regressions, using the existing fake launcher.
#include "../supervisor.hpp"
#include "fake_process.hpp"

#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
using namespace sonder_test;

namespace {
class ResidencyCounters final : public GpuCounterSource {
  public:
    std::atomic<std::uint64_t> dedicated{14736 * kMiB};
    std::atomic<bool> error{false};
    std::atomic<bool> empty{false};
    std::atomic<bool> paused{false};
    bool available = true;
    std::string name() const override { return "fake"; }
    bool supported() const override { return available; }
    Result<GpuProcessCounters> read(std::uint32_t pid) override {
        if (paused.load())
            return Status(ErrorCode::io_error, "fake counter paused");
        if (error.load())
            return Status(ErrorCode::io_error, "fake counter failure");
        if (empty.load())
            return GpuProcessCounters{};
        const auto name = "pid_" + std::to_string(pid) + "_adapter";
        return GpuProcessCounters{{{name, dedicated.load()}}, {{name, 0}}};
    }
};

bool until(const std::function<bool()> &predicate) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    do {
        if (predicate())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while (std::chrono::steady_clock::now() < deadline);
    return predicate();
}

struct Fixture {
    std::shared_ptr<ResidencyCounters> counters = std::make_shared<ResidencyCounters>();
    std::shared_ptr<LaunchState> launches = std::make_shared<LaunchState>();
    SupervisorOptions options = base_options();
    Fixture() {
        launches->next_pid = 1000;
        options.arguments = {"--model", "model.gguf", "--n-gpu-layers", "999"};
        options.gpu_counters = counters;
        options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
        options.spill_guard.sample_interval = std::chrono::milliseconds(100);
    }
    std::unique_ptr<Supervisor> supervisor() const {
        return std::make_unique<Supervisor>(options, std::make_unique<FakeLauncher>(launches));
    }
    std::size_t starts() const {
        std::lock_guard lock(launches->mutex);
        return launches->starts.size();
    }
};

bool warned(const BackendRuntimeStatus &status, const char *code) {
    for (const auto &warning : status.warnings)
        if (warning.code == code)
            return true;
    return false;
}
} // namespace

TEST_CASE("residency supervisor warns on CPU fallback only after consecutive ready samples") {
    Fixture f;
    f.counters->dedicated = 0;
    f.counters->paused = true;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    CHECK_FALSE(s->runtime_status().gpu_memory.residency.has_value());
    f.counters->paused = false;
    REQUIRE(until([&] { return warned(s->runtime_status(), "gpu_offload_missing"); }));
    const auto status = s->runtime_status();
    REQUIRE(status.gpu_memory.residency.has_value());
    CHECK(status.gpu_memory.residency->gpu_offload_missing);
    CHECK(status.gpu_memory.residency->action == "warn");
    CHECK_FALSE(status.gpu_memory.spilled);
    CHECK(s->running());
    CHECK(f.starts() == 1);
    CHECK(child_at(f.launches, 0)->stops == 0);
}

TEST_CASE("residency supervisor recognizes no GPU counter instances as CPU fallback") {
    Fixture f;
    f.counters->empty = true;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    REQUIRE(until([&] { return warned(s->runtime_status(), "gpu_offload_missing"); }));
    CHECK(s->runtime_status().gpu_memory.dedicated_bytes == 0);
}

TEST_CASE("residency supervisor refuse stops a CPU child without crash restart") {
    Fixture f;
    f.counters->dedicated = 0;
    f.counters->paused = true;
    f.options.spill_guard.policy = SpillPolicy::refuse;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    f.counters->paused = false;
    REQUIRE(until([&] { return !s->failure().ok(); }));
    CHECK(s->failure().message().find("gpu_offload_missing") != std::string::npos);
    CHECK(s->failure().message().find("refusing to serve") != std::string::npos);
    CHECK_FALSE(s->running());
    CHECK_FALSE(child_at(f.launches, 0)->alive);
    CHECK(f.starts() == 1);
    CHECK_FALSE(s->start().ok());
}

TEST_CASE("residency supervisor refuse can reject the readiness sample when configured for one") {
    Fixture f;
    f.counters->dedicated = 0;
    f.options.spill_guard.policy = SpillPolicy::refuse;
    f.options.spill_guard.residency.consecutive_samples = 1;
    auto s = f.supervisor();
    CHECK_FALSE(s->start().ok());
    CHECK_FALSE(child_at(f.launches, 0)->alive);
    CHECK(s->runtime_status().gpu_memory.residency->action == "refused");
}

TEST_CASE("residency supervisor high dedicated usage preserves legacy JSON shape") {
    Fixture f;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    REQUIRE(until([&] { return s->runtime_status().gpu_memory.samples >= 4; }));
    auto status = s->runtime_status();
    CHECK_FALSE(status.gpu_memory.residency.has_value());
    const json::Value object(to_json(status.gpu_memory));
    CHECK(object.find("residency") == nullptr);
    CHECK_FALSE(warned(status, "gpu_offload_missing"));
    CHECK_FALSE(warned(status, "vram_evicted"));
    CHECK(s->running());
    CHECK(f.starts() == 1);
}

TEST_CASE("residency supervisor honors disabled guards and intentional CPU or unknown GPU intent") {
    Fixture f;
    f.counters->dedicated = 0;
    SUBCASE("explicit CPU overrides assume GPU") {
        f.options.arguments = {"-ngl", "0"};
        f.options.spill_guard.residency.expect_gpu = true;
    }
    SUBCASE("no GPU flags") { f.options.arguments = {"--model", "model.gguf"}; }
    SUBCASE("residency disabled") { f.options.spill_guard.residency.enabled = false; }
    SUBCASE("probe unavailable") { f.counters->available = false; }
    SUBCASE("probe fails") { f.counters->error = true; }
    SUBCASE("unknown PID") { f.launches->next_pid = 0; }
    SUBCASE("all counter checks disabled") { f.options.spill_guard.enabled = false; }
    f.options.spill_guard.policy = SpillPolicy::refuse;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    // Waiting beyond the confirmation interval must not cause a refusal.
    std::this_thread::sleep_for(std::chrono::milliseconds(350));
    CHECK_FALSE(s->runtime_status().gpu_memory.residency.has_value());
    CHECK(s->running());
}

TEST_CASE("residency supervisor detects eviction with shared memory zero and keeps serving by default") {
    Fixture f;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    f.counters->dedicated = 2889 * kMiB;
    REQUIRE(until([&] { return warned(s->runtime_status(), "vram_evicted"); }));
    auto status = s->runtime_status();
    REQUIRE(status.gpu_memory.residency.has_value());
    CHECK(status.gpu_memory.residency->peak_dedicated_bytes == 14736 * kMiB);
    CHECK(status.gpu_memory.residency->observed_dedicated_bytes == 2889 * kMiB);
    CHECK(status.gpu_memory.dedicated_bytes == 2889 * kMiB);
    CHECK_FALSE(status.gpu_memory.spilled);
    CHECK(s->running());
    CHECK(f.starts() == 1);
    const auto message = status.warnings.back().message;
    f.counters->dedicated = 15000 * kMiB;
    REQUIRE(until([&] { return s->runtime_status().gpu_memory.dedicated_bytes == 15000 * kMiB; }));
    // The event numbers/message are stable; changing current usage must not
    // defeat the engine's warning deduplication and emit on every sample.
    CHECK(s->runtime_status().warnings.back().message == message);
    CHECK(s->runtime_status().gpu_memory.residency->peak_dedicated_bytes == 14736 * kMiB);
}

TEST_CASE("residency restart drains leases closes admission and relaunches once without flapping") {
    Fixture f;
    f.options.spill_guard.residency.on_eviction = LlamaServerEvictionPolicy::restart;
    auto s = f.supervisor();
    auto acquired = s->acquire_request();
    REQUIRE(acquired.ok());
    Supervisor::RequestLease lease = std::move(acquired.value());
    REQUIRE(lease.port() != 0);
    f.counters->dedicated = 2889 * kMiB;
    REQUIRE(until([&] {
        const auto r = s->runtime_status().gpu_memory.residency;
        return r && r->action == "waiting_for_idle";
    }));
    CHECK(f.starts() == 1);
    CHECK(child_at(f.launches, 0)->alive);
    CancellationSource cancel;
    std::atomic<bool> done{false};
    Status admission;
    std::thread waiter([&] {
        admission = s->acquire_request(cancel.token()).status();
        done = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK_FALSE(done.load());
    cancel.cancel();
    waiter.join();
    CHECK(admission.code() == ErrorCode::cancelled);
    // RAII movement must transfer ownership, with exactly one release.
    Supervisor::RequestLease moved = std::move(lease);
    CHECK(lease.port() == 0);
    moved = {};
    REQUIRE(wait_for_starts(f.launches, 2));
    REQUIRE(s->start().ok());
    CHECK_FALSE(child_at(f.launches, 0)->alive);
    REQUIRE(until([&] { return s->runtime_status().gpu_memory.samples >= 4; }));
    CHECK(f.starts() == 2);
    CHECK(s->runtime_status().gpu_memory.residency->eviction_restarts == 1);
    CHECK(s->runtime_status().gpu_memory.residency->action == "restarted");
    // Even another peak/drop in the replacement cannot reset the lifetime budget.
    f.counters->dedicated = 14736 * kMiB;
    REQUIRE(until([&] { return s->runtime_status().gpu_memory.dedicated_bytes == 14736 * kMiB; }));
    f.counters->dedicated = 2889 * kMiB;
    REQUIRE(until([&] { return s->runtime_status().gpu_memory.residency->action == "restart_exhausted"; }));
    CHECK(f.starts() == 2);
    CHECK(s->acquire_request().ok());
}

TEST_CASE("residency refusal also drains an active request") {
    Fixture f;
    f.options.spill_guard.policy = SpillPolicy::refuse;
    auto s = f.supervisor();
    auto acquired = s->acquire_request();
    REQUIRE(acquired.ok());
    auto lease = std::move(acquired.value());
    f.counters->dedicated = 0;
    REQUIRE(until([&] {
        const auto r = s->runtime_status().gpu_memory.residency;
        return r && r->action == "waiting_for_idle";
    }));
    CHECK(child_at(f.launches, 0)->alive);
    lease = {};
    REQUIRE(until([&] { return !s->failure().ok(); }));
    CHECK_FALSE(child_at(f.launches, 0)->alive);
}

TEST_CASE("residency lease can safely unwind after supervisor shutdown and destruction") {
    Fixture f;
    auto s = f.supervisor();
    auto acquired = s->acquire_request();
    REQUIRE(acquired.ok());
    auto lease = std::move(acquired.value());
    s.reset();
    CHECK_FALSE(child_at(f.launches, 0)->alive);
    lease = {};  // ASan regression: no call through a freed Supervisor pointer
    CHECK(lease.port() == 0);
}
