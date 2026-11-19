// Host RAM admission and floor guard: policy (injected fake memory and clock)
// and the supervisor integration (fake launcher, fake memory provider).
#include "../host_memory.hpp"
#include "../supervisor.hpp"
#include "fake_process.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <functional>
#include <iterator>
#include <thread>

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
using namespace sonder_test;

namespace {

using Clock = std::chrono::steady_clock;
constexpr std::uint64_t kGiB = 1ull << 30;

HostMemory memory_with(MemoryReading available, MemoryReading commit = {}) {
    HostMemory m;
    m.total_physical = MemoryReading::known(64 * kGiB);
    m.available_physical = available;
    m.available_commit = commit;
    m.source = "fake";
    return m;
}

bool contains(const std::string &text, const std::string &needle) { return text.find(needle) != std::string::npos; }

bool until(const std::function<bool()> &predicate) {
    const auto deadline = Clock::now() + std::chrono::seconds(5);
    do {
        if (predicate())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while (Clock::now() < deadline);
    return predicate();
}

bool warned(const BackendRuntimeStatus &status, const char *code) {
    return std::any_of(status.warnings.begin(), status.warnings.end(),
                       [&](const BackendWarning &warning) { return warning.code == code; });
}

HostMemoryFloorOptions floor_options(HostFloorPolicy policy) {
    HostMemoryFloorOptions f;
    f.enabled = true;
    f.policy = policy;
    f.floor_bytes = 8 * kGiB;
    f.resume_bytes = 12 * kGiB;
    f.consecutive_samples = 2;
    f.sample_interval = std::chrono::milliseconds(1000);
    f.restart_cooldown = std::chrono::milliseconds(60000);
    f.max_restarts = 2;
    return f;
}

} // namespace

TEST_CASE("host admission refuses with the estimate, reserve and available numbers in the message") {
    HostMemoryAdmissionOptions o;
    o.estimated_resident_bytes = 100 * kGiB;
    o.reserve_bytes = 4 * kGiB;
    const auto verdict = check_host_admission(o, memory_with(MemoryReading::known(60 * kGiB)));
    CHECK_FALSE(verdict.admit());
    CHECK(verdict.outcome == HostAdmissionVerdict::Outcome::refused);
    CHECK(verdict.required_bytes == 104 * kGiB);
    CHECK(verdict.available_bytes == 60 * kGiB);
    CHECK(contains(verdict.message, "102400 MiB estimated host RAM"));
    CHECK(contains(verdict.message, "4096 MiB reserve"));
    CHECK(contains(verdict.message, "106496 MiB"));
    CHECK(contains(verdict.message, "only 61440 MiB of physical memory is available"));
    CHECK(contains(verdict.message, "override_refusal"));
}

TEST_CASE("host admission admits when memory suffices and boundaries are exact") {
    HostMemoryAdmissionOptions o;
    o.estimated_resident_bytes = 10 * kGiB;
    o.reserve_bytes = 2 * kGiB;
    CHECK(check_host_admission(o, memory_with(MemoryReading::known(12 * kGiB))).outcome ==
          HostAdmissionVerdict::Outcome::admitted);
    CHECK(check_host_admission(o, memory_with(MemoryReading::known(12 * kGiB - 1))).outcome ==
          HostAdmissionVerdict::Outcome::refused);
}

TEST_CASE("host admission tri-state: known zero refuses, unknown admits with a warning, no estimate skips") {
    HostMemoryAdmissionOptions o;
    o.estimated_resident_bytes = 1 * kGiB;
    o.reserve_bytes = 0;
    const auto zero = check_host_admission(o, memory_with(MemoryReading::known(0)));
    CHECK(zero.outcome == HostAdmissionVerdict::Outcome::refused);
    CHECK(contains(zero.message, "only 0 MiB"));

    HostMemory unknown = memory_with(MemoryReading::unknown());
    unknown.note = "probe failed";
    const auto verdict = check_host_admission(o, unknown);
    CHECK(verdict.outcome == HostAdmissionVerdict::Outcome::admitted_unknown);
    CHECK(verdict.admit());
    CHECK(contains(verdict.message, "unknown"));
    CHECK(contains(verdict.message, "probe failed"));
    CHECK(contains(verdict.message, "1024 MiB"));

    HostMemoryAdmissionOptions none;
    CHECK(check_host_admission(none, memory_with(MemoryReading::known(0))).outcome ==
          HostAdmissionVerdict::Outcome::skipped);
}

TEST_CASE("host admission override admits a refusal and keeps the numbers") {
    HostMemoryAdmissionOptions o;
    o.estimated_resident_bytes = 50 * kGiB;
    o.reserve_bytes = 0;
    o.override_refusal = true;
    const auto verdict = check_host_admission(o, memory_with(MemoryReading::known(10 * kGiB)));
    CHECK(verdict.outcome == HostAdmissionVerdict::Outcome::admitted_override);
    CHECK(verdict.admit());
    CHECK(contains(verdict.message, "51200 MiB"));
    CHECK(contains(verdict.message, "10240 MiB"));
}

TEST_CASE("host admission checks commit separately when it is known") {
    HostMemoryAdmissionOptions o;
    o.estimated_resident_bytes = 40 * kGiB;
    o.estimated_commit_bytes = 20 * kGiB;
    o.reserve_bytes = 2 * kGiB;
    // Plenty of physical memory, too little commit.
    const auto verdict = check_host_admission(o, memory_with(MemoryReading::known(60 * kGiB), MemoryReading::known(10 * kGiB)));
    CHECK(verdict.outcome == HostAdmissionVerdict::Outcome::refused);
    CHECK(contains(verdict.message, "available commit is 10240 MiB"));
    CHECK(contains(verdict.message, "22528 MiB"));
    // Unknown commit does not refuse.
    CHECK(check_host_admission(o, memory_with(MemoryReading::known(60 * kGiB), MemoryReading::unknown())).outcome ==
          HostAdmissionVerdict::Outcome::admitted);
}

TEST_CASE("host memory options are validated") {
    HostMemoryOptions ok;
    CHECK(validate_host_memory_options(ok).ok());
    ok.floor = floor_options(HostFloorPolicy::warn);
    CHECK(validate_host_memory_options(ok).ok());

    auto bad = ok;
    bad.floor.floor_bytes = 0;
    CHECK_FALSE(validate_host_memory_options(bad).ok());
    bad = ok;
    bad.floor.resume_bytes = 4 * kGiB;  // below the floor
    CHECK_FALSE(validate_host_memory_options(bad).ok());
    bad = ok;
    bad.floor.consecutive_samples = 0;
    CHECK_FALSE(validate_host_memory_options(bad).ok());
    bad = ok;
    bad.floor.sample_interval = std::chrono::milliseconds(1);
    CHECK_FALSE(validate_host_memory_options(bad).ok());
    bad = ok;
    bad.admission.estimated_resident_bytes = 1 * kGiB;
    bad.admission.estimated_commit_bytes = 2 * kGiB;
    CHECK_FALSE(validate_host_memory_options(bad).ok());
}

TEST_CASE("floor guard enters pressure only after consecutive low samples and leaves with hysteresis") {
    HostMemoryFloorGuard guard(floor_options(HostFloorPolicy::stop_admission));
    auto now = Clock::time_point{} + std::chrono::seconds(100);
    guard.reset(now);
    CHECK(guard.state() == HostMemoryFloorGuard::State::ok);
    CHECK(guard.admitting());
    CHECK(guard.due(now));

    // One low sample is not enough; a healthy one resets the streak.
    auto u = guard.observe(MemoryReading::known(7 * kGiB), now);
    CHECK_FALSE(u.entered_pressure);
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(9 * kGiB), now);
    CHECK_FALSE(u.entered_pressure);
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(7 * kGiB), now);
    CHECK_FALSE(u.entered_pressure);
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(0), now);  // known zero counts as low
    CHECK(u.entered_pressure);
    CHECK(guard.state() == HostMemoryFloorGuard::State::pressure);
    CHECK_FALSE(guard.admitting());

    // Between floor and resume: still pressure (hysteresis), no streak progress.
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(10 * kGiB), now);
    CHECK_FALSE(u.left_pressure);
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(12 * kGiB), now);
    CHECK_FALSE(u.left_pressure);  // one recovered sample
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(10 * kGiB), now);  // dips again: streak resets
    CHECK_FALSE(u.left_pressure);
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(13 * kGiB), now);
    CHECK_FALSE(u.left_pressure);
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(13 * kGiB), now);
    CHECK(u.left_pressure);
    CHECK(guard.state() == HostMemoryFloorGuard::State::ok);
    CHECK(guard.admitting());
}

TEST_CASE("floor guard warn policy never closes admission") {
    HostMemoryFloorGuard guard(floor_options(HostFloorPolicy::warn));
    auto now = Clock::time_point{} + std::chrono::seconds(5);
    guard.reset(now);
    (void)guard.observe(MemoryReading::known(1 * kGiB), now);
    now += std::chrono::seconds(1);
    const auto u = guard.observe(MemoryReading::known(1 * kGiB), now);
    CHECK(u.entered_pressure);
    CHECK(guard.state() == HostMemoryFloorGuard::State::pressure);
    CHECK(guard.admitting());
    CHECK_FALSE(u.restart_requested);
}

TEST_CASE("floor guard unknown samples change nothing") {
    HostMemoryFloorGuard guard(floor_options(HostFloorPolicy::stop_admission));
    auto now = Clock::time_point{} + std::chrono::seconds(5);
    guard.reset(now);
    (void)guard.observe(MemoryReading::known(1 * kGiB), now);  // low streak 1
    now += std::chrono::seconds(1);
    auto u = guard.observe(MemoryReading::unknown(), now);
    CHECK(u.unknown_sample);
    CHECK_FALSE(u.entered_pressure);
    CHECK(guard.unknown_samples() == 1);
    now += std::chrono::seconds(1);
    u = guard.observe(MemoryReading::known(1 * kGiB), now);  // streak continues: 2
    CHECK(u.entered_pressure);
    // Unknown in pressure does not recover.
    for (int i = 0; i < 5; ++i) {
        now += std::chrono::seconds(1);
        CHECK_FALSE(guard.observe(MemoryReading::unknown(), now).left_pressure);
    }
    CHECK(guard.state() == HostMemoryFloorGuard::State::pressure);
}

TEST_CASE("floor guard schedules samples from the injected clock") {
    auto options = floor_options(HostFloorPolicy::warn);
    options.sample_interval = std::chrono::milliseconds(500);
    HostMemoryFloorGuard guard(options);
    const auto start = Clock::time_point{} + std::chrono::seconds(10);
    guard.reset(start);
    CHECK(guard.due(start));
    (void)guard.observe(MemoryReading::known(20 * kGiB), start);
    CHECK_FALSE(guard.due(start + std::chrono::milliseconds(499)));
    CHECK(guard.due(start + std::chrono::milliseconds(500)));
    HostMemoryFloorGuard disabled{HostMemoryFloorOptions{}};
    CHECK_FALSE(disabled.due(start));
}

TEST_CASE("floor guard restart requests are one-shot, cooled down and capped") {
    auto options = floor_options(HostFloorPolicy::restart_smaller_ctx);
    options.consecutive_samples = 1;
    options.max_restarts = 2;
    options.restart_cooldown = std::chrono::seconds(60);
    HostMemoryFloorGuard guard(options);
    auto now = Clock::time_point{} + std::chrono::seconds(1000);
    guard.reset(now);

    auto u = guard.observe(MemoryReading::known(1 * kGiB), now);
    CHECK(u.entered_pressure);
    CHECK(u.restart_requested);
    CHECK(guard.restart_pending());
    now += std::chrono::seconds(1);
    CHECK_FALSE(guard.observe(MemoryReading::known(1 * kGiB), now).restart_requested);  // already pending

    guard.restart_performed(now);
    CHECK(guard.restarts() == 1);
    CHECK_FALSE(guard.restart_pending());
    guard.reset(now);  // new child: clean state, restart budget kept
    CHECK(guard.restarts() == 1);

    now += std::chrono::seconds(10);
    u = guard.observe(MemoryReading::known(1 * kGiB), now);
    CHECK(u.entered_pressure);
    CHECK_FALSE(u.restart_requested);  // within the 60 s cooldown
    now += std::chrono::seconds(60);
    u = guard.observe(MemoryReading::known(1 * kGiB), now);
    CHECK(u.restart_requested);
    guard.restart_performed(now);
    CHECK(guard.restarts() == 2);
    guard.reset(now);

    now += std::chrono::seconds(120);
    u = guard.observe(MemoryReading::known(1 * kGiB), now);
    CHECK(u.entered_pressure);
    CHECK_FALSE(u.restart_requested);  // max_restarts reached
    CHECK_FALSE(guard.admitting());
}

TEST_CASE("floor guard drops a pending restart when pressure ends first") {
    auto options = floor_options(HostFloorPolicy::restart_smaller_ctx);
    options.consecutive_samples = 1;
    HostMemoryFloorGuard guard(options);
    auto now = Clock::time_point{} + std::chrono::seconds(5);
    guard.reset(now);
    CHECK(guard.observe(MemoryReading::known(1 * kGiB), now).restart_requested);
    now += std::chrono::seconds(1);
    CHECK(guard.observe(MemoryReading::known(20 * kGiB), now).left_pressure);
    CHECK_FALSE(guard.restart_pending());
}

namespace {

struct HostFixture {
    std::shared_ptr<LaunchState> launches = std::make_shared<LaunchState>();
    SupervisorOptions options = base_options();
    std::atomic<std::uint64_t> available{64 * kGiB};
    std::atomic<bool> unknown{false};
    std::atomic<unsigned> reads{0};

    HostFixture() {
        options.arguments = {"--model", "model.gguf", "--ctx-size", "65536"};
        options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
        options.spill_guard.enabled = false;
        options.native_completion = false;
        options.kv_pairing_check = false;
        options.host_memory.provider = [this] {
            reads.fetch_add(1);
            return memory_with(unknown.load() ? MemoryReading::unknown() : MemoryReading::known(available.load()));
        };
    }
    std::unique_ptr<Supervisor> supervisor() const {
        return std::make_unique<Supervisor>(options, std::make_unique<FakeLauncher>(launches));
    }
    std::size_t starts() const {
        std::lock_guard lock(launches->mutex);
        return launches->starts.size();
    }
};

} // namespace

TEST_CASE("supervisor refuses before spawning when the estimate does not fit and names both numbers") {
    HostFixture f;
    f.available = 20 * kGiB;
    f.options.host_memory.admission.estimated_resident_bytes = 90 * kGiB;
    f.options.host_memory.admission.reserve_bytes = 10 * kGiB;
    auto s = f.supervisor();
    const auto started = s->start();
    REQUIRE_FALSE(started.ok());
    CHECK(started.status().code() == ErrorCode::unavailable);
    CHECK(contains(started.status().message(), "102400 MiB"));  // estimate + reserve
    CHECK(contains(started.status().message(), "20480 MiB"));   // available
    CHECK(contains(started.status().message(), "host memory admission refused"));
    CHECK(f.starts() == 0);
    CHECK_FALSE(s->failure().ok());
    CHECK_FALSE(s->start().ok());  // terminal
    CHECK(f.starts() == 0);
}

TEST_CASE("supervisor admits with a warning when the override is set") {
    HostFixture f;
    f.available = 20 * kGiB;
    f.options.host_memory.admission.estimated_resident_bytes = 90 * kGiB;
    f.options.host_memory.admission.reserve_bytes = 0;
    f.options.host_memory.admission.override_refusal = true;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    CHECK(f.starts() == 1);
    CHECK(warned(s->runtime_status(), "host_memory_admission_override"));
}

TEST_CASE("supervisor admits with a documented warning when availability is unknown") {
    HostFixture f;
    f.unknown = true;
    f.options.host_memory.admission.estimated_resident_bytes = 90 * kGiB;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    CHECK(warned(s->runtime_status(), "host_memory_unknown"));
}

TEST_CASE("supervisor without an estimate never reads host memory before spawning") {
    HostFixture f;
    f.available = 0;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    CHECK(f.reads.load() == 0);
}

TEST_CASE("supervisor rejects invalid host_memory options") {
    HostFixture f;
    f.options.host_memory.floor = floor_options(HostFloorPolicy::warn);
    f.options.host_memory.floor.floor_bytes = 0;
    auto s = f.supervisor();
    const auto started = s->start();
    REQUIRE_FALSE(started.ok());
    CHECK(started.status().code() == ErrorCode::invalid_argument);
    CHECK(f.starts() == 0);
}

TEST_CASE("supervisor floor guard warns without refusing under the warn policy") {
    HostFixture f;
    f.options.host_memory.floor = floor_options(HostFloorPolicy::warn);
    f.options.host_memory.floor.sample_interval = std::chrono::milliseconds(10);
    f.options.host_memory.floor.consecutive_samples = 1;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    f.available = 2 * kGiB;
    REQUIRE(until([&] { return warned(s->runtime_status(), "host_memory_low"); }));
    auto lease = s->acquire_request();
    CHECK(lease.ok());  // warn never closes admission
    f.available = 40 * kGiB;
    REQUIRE(until([&] { return !warned(s->runtime_status(), "host_memory_low"); }));
    CHECK(f.starts() == 1);
}

TEST_CASE("supervisor stop_admission closes new requests under the floor and reopens on recovery") {
    HostFixture f;
    f.options.host_memory.floor = floor_options(HostFloorPolicy::stop_admission);
    f.options.host_memory.floor.sample_interval = std::chrono::milliseconds(10);
    f.options.host_memory.floor.consecutive_samples = 1;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    {
        // A request admitted before the pressure is not interrupted by it.
        auto before = s->acquire_request();
        REQUIRE(before.ok());
        f.available = 2 * kGiB;
        REQUIRE(until([&] { return warned(s->runtime_status(), "host_memory_low"); }));
        // Wait for a refusal rather than asserting the first call: robust even
        // if the warning is observed a moment before the flag.
        ErrorCode refused_code = ErrorCode::ok;
        std::string refused_message;
        REQUIRE(until([&] {
            auto refused = s->acquire_request();
            if (refused.ok())
                return false;
            refused_code = refused.status().code();
            refused_message = refused.status().message();
            return true;
        }));
        CHECK(refused_code == ErrorCode::unavailable);
        CHECK(contains(refused_message, "host memory"));
        CHECK(s->running());
        CHECK(child_at(f.launches, 0)->alive.load());
    }
    f.available = 40 * kGiB;
    REQUIRE(until([&] { return s->acquire_request().ok(); }));
    CHECK(f.starts() == 1);
}

TEST_CASE("supervisor restart_smaller_ctx relaunches idle with a smaller context") {
    HostFixture f;
    f.options.host_memory.floor = floor_options(HostFloorPolicy::restart_smaller_ctx);
    f.options.host_memory.floor.sample_interval = std::chrono::milliseconds(10);
    f.options.host_memory.floor.consecutive_samples = 1;
    f.options.host_memory.floor.max_restarts = 1;
    // Admission would refuse the relaunch (the estimate does not shrink with
    // the context); the floor guard's own relaunch must not be re-judged.
    f.options.host_memory.admission.estimated_resident_bytes = 30 * kGiB;
    f.options.host_memory.admission.reserve_bytes = 0;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    REQUIRE(f.starts() == 1);
    f.available = 2 * kGiB;  // below the 8 GiB floor
    REQUIRE(wait_for_starts(f.launches, 2));
    REQUIRE(until([&] { return s->running(); }));
    const auto args = s->launch_arguments();
    const auto it = std::find(args.begin(), args.end(), "--ctx-size");
    REQUIRE(it != args.end());
    REQUIRE(std::next(it) != args.end());
    CHECK(std::stoull(*std::next(it)) < 65536);
    CHECK(child_at(f.launches, 0)->stops >= 1);
    {
        std::lock_guard lock(f.launches->mutex);
        const auto &spec = f.launches->specs.at(1).arguments;
        const auto at = std::find(spec.begin(), spec.end(), "--ctx-size");
        REQUIRE(at != spec.end());
        CHECK(std::stoull(*std::next(at)) < 65536);
    }
    // The restart budget is spent: sustained pressure now only closes admission.
    REQUIRE(until([&] { return !s->acquire_request().ok(); }));
    CHECK(f.starts() == 2);
}

TEST_CASE("supervisor drops a pending restart and reopens admission when memory recovers while a request holds the child") {
    HostFixture f;
    f.options.host_memory.floor = floor_options(HostFloorPolicy::restart_smaller_ctx);
    f.options.host_memory.floor.sample_interval = std::chrono::milliseconds(10);
    f.options.host_memory.floor.consecutive_samples = 1;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    auto held = s->acquire_request();
    REQUIRE(held.ok());
    f.available = 2 * kGiB;
    REQUIRE(until([&] { return warned(s->runtime_status(), "host_memory_low"); }));
    // The restart is pending but cannot run: the lease keeps the child busy.
    REQUIRE(until([&] { return !s->acquire_request().ok(); }));
    CHECK(f.starts() == 1);
    // Memory recovers while the lease is still held: the pending restart must
    // be dropped and admission reopened without ever restarting the child.
    f.available = 40 * kGiB;
    REQUIRE(until([&] { return !warned(s->runtime_status(), "host_memory_low"); }));
    REQUIRE(until([&] { return s->acquire_request().ok(); }));
    CHECK(f.starts() == 1);
    CHECK(child_at(f.launches, 0)->alive.load());
    CHECK(child_at(f.launches, 0)->stops == 0);
}

TEST_CASE("supervisor warns that the host estimate is a lower bound and still admits") {
    HostFixture f;
    f.options.host_memory.admission.estimated_resident_bytes = 4 * kGiB;
    f.options.host_memory.admission.reserve_bytes = 0;
    f.options.host_memory.admission.estimate_lower_bound = true;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    CHECK(f.starts() == 1);
    CHECK(warned(s->runtime_status(), "host_memory_estimate_lower_bound"));
    // The flag defaults off: no warning without it.
    HostFixture g;
    g.options.host_memory.admission.estimated_resident_bytes = 4 * kGiB;
    g.options.host_memory.admission.reserve_bytes = 0;
    auto t = g.supervisor();
    REQUIRE(t->start().ok());
    CHECK_FALSE(warned(t->runtime_status(), "host_memory_estimate_lower_bound"));
}

TEST_CASE("supervisor floor guard says admission stays closed while readings are unknown") {
    HostFixture f;
    f.options.host_memory.floor = floor_options(HostFloorPolicy::stop_admission);
    f.options.host_memory.floor.sample_interval = std::chrono::milliseconds(10);
    f.options.host_memory.floor.consecutive_samples = 1;
    auto s = f.supervisor();
    REQUIRE(s->start().ok());
    f.available = 2 * kGiB;
    REQUIRE(until([&] { return warned(s->runtime_status(), "host_memory_low"); }));
    // The warning is raised before the admission flag flips, so wait for the closed state.
    REQUIRE(until([&] { return !s->acquire_request().ok(); }));
    f.unknown = true;
    const auto closed_warning = [&] {
        for (const auto &w : s->runtime_status().warnings)
            if (w.code == "host_memory_unknown" && contains(w.message, "admission closed"))
                return true;
        return false;
    };
    REQUIRE(until(closed_warning));
    CHECK_FALSE(s->acquire_request().ok());
    // A known low reading clears the unknown warning; unknown again raises it again.
    f.unknown = false;
    REQUIRE(until([&] { return !warned(s->runtime_status(), "host_memory_unknown"); }));
    f.unknown = true;
    REQUIRE(until(closed_warning));
}
