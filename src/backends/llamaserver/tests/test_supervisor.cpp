#include "../process.hpp"
#include "../supervisor.hpp"
#include "fake_server.hpp"
#include <atomic>
#include <condition_variable>
#include <doctest/doctest.h>
#include <mutex>
#include <thread>

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
namespace {
struct ChildState {
    std::atomic<bool> alive{true};
    std::atomic<unsigned> stops{0};
};
struct LaunchState {
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<ChildState>> children;
    std::vector<std::chrono::steady_clock::time_point> starts;
    std::vector<ProcessSpec> specs;
};
class FakeProcess final : public Process {
  public:
    explicit FakeProcess(std::shared_ptr<ChildState> state) : state_(std::move(state)) {}
    bool running() const override { return state_->alive.load(std::memory_order_acquire); }
    void stop(std::chrono::milliseconds) override {
        state_->stops.fetch_add(1);
        state_->alive.store(false);
    }

  private:
    std::shared_ptr<ChildState> state_;
};
class FakeLauncher final : public ProcessLauncher {
  public:
    explicit FakeLauncher(std::shared_ptr<LaunchState> state) : state_(std::move(state)) {}
    Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
        auto child = std::make_shared<ChildState>();
        {
            std::lock_guard lock(state_->mutex);
            state_->children.push_back(child);
            state_->starts.push_back(std::chrono::steady_clock::now());
            state_->specs.push_back(spec);
        }
        state_->changed.notify_all();
        return std::unique_ptr<Process>(new FakeProcess(std::move(child)));
    }

  private:
    std::shared_ptr<LaunchState> state_;
};
bool wait_for_starts(const std::shared_ptr<LaunchState> &state, std::size_t n) {
    std::unique_lock lock(state->mutex);
    return state->changed.wait_for(lock, std::chrono::seconds(3), [&] { return state->starts.size() >= n; });
}
std::shared_ptr<ChildState> child_at(const std::shared_ptr<LaunchState> &state, std::size_t index) {
    std::lock_guard lock(state->mutex);
    return state->children.at(index);
}
SupervisorOptions base_options() {
    SupervisorOptions o;
    o.executable = "fake-llama-server";
    o.arguments = {"--model", "model.gguf", "--spec-type", "draft-mtp"};
    o.readiness_timeout = std::chrono::milliseconds(2000);
    o.health_poll_interval = std::chrono::milliseconds(2);
    o.restart_initial_backoff = std::chrono::milliseconds(5);
    o.restart_max_backoff = std::chrono::milliseconds(8);
    return o;
}
} // namespace

TEST_CASE("supervisor polls readiness false false true and appends loopback binding") {
    auto state = std::make_shared<LaunchState>();
    auto probes = std::make_shared<std::atomic<unsigned>>(0);
    auto o = base_options();
    o.health_check = [probes](std::uint16_t port, std::chrono::milliseconds) {
        CHECK(port != 0);
        return probes->fetch_add(1) >= 2;
    };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    REQUIRE(state->specs.size() == 1);
    const auto &a = state->specs.front().arguments;
    CHECK(a[a.size() - 4] == "--host");
    CHECK(a[a.size() - 3] == "127.0.0.1");
    CHECK(a[a.size() - 2] == "--port");
    s.stop();
}

TEST_CASE("supervisor waits through real HTTP health 503 responses before 200") {
    struct HttpChild final : Process {
        std::shared_ptr<sonder_test::FakeLlamaServer> server;
        explicit HttpChild(std::shared_ptr<sonder_test::FakeLlamaServer> s) : server(std::move(s)) {}
        bool running() const override { return server != nullptr; }
        void stop(std::chrono::milliseconds) override { server.reset(); }
    };
    struct HttpLauncher final : ProcessLauncher {
        std::weak_ptr<sonder_test::FakeLlamaServer> server;
        Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
            auto upstream = std::make_shared<sonder_test::FakeLlamaServer>(spec.port, 2);
            server = upstream;
            return std::unique_ptr<Process>(new HttpChild(std::move(upstream)));
        }
    };
    auto launcher = std::make_unique<HttpLauncher>();
    auto *handle = launcher.get();
    Supervisor supervisor(base_options(), std::move(launcher));
    auto ready = supervisor.start();
    REQUIRE_MESSAGE(ready.ok(), ready.status().to_string());
    auto server = handle->server.lock();
    REQUIRE(server);
    CHECK(server->health_requests() == 3);
    server.reset();
    supervisor.stop();
    CHECK(handle->server.expired());
}

TEST_CASE("supervisor timeout stops child") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.readiness_timeout = std::chrono::milliseconds(15);
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return false; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    CHECK_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::timeout);
    REQUIRE(state->children.size() == 1);
    CHECK(state->children.front()->stops.load() >= 1);
}

TEST_CASE("supervisor crash retries twice, caps backoff, then exhausts") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.max_restarts = 2;
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    child_at(state, 0)->alive.store(false);
    REQUIRE(wait_for_starts(state, 2));
    child_at(state, 1)->alive.store(false);
    REQUIRE(wait_for_starts(state, 3));
    child_at(state, 2)->alive.store(false);
    {
        std::lock_guard lock(state->mutex);
        REQUIRE(state->starts.size() == 3);
        CHECK(state->starts[1] - state->starts[0] >= o.restart_initial_backoff);
        CHECK(state->starts[2] - state->starts[1] >= o.restart_max_backoff);
    }
    // Wait for the final state, not an earlier transient backoff failure.
    for (int i = 0; i < 1000 && child_at(state, 2)->stops.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK_FALSE(s.start().ok());
    CHECK_FALSE(s.start().ok());
    s.stop();
    CHECK(state->starts.size() == 3);
}

TEST_CASE("supervisor stop during readiness prevents later launches") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return false; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    std::thread t([&] { (void)s.start(); });
    const bool launched = wait_for_starts(state, 1);
    s.stop();
    t.join();
    REQUIRE(launched);
    CHECK(state->starts.size() == 1);
    CHECK_FALSE(s.start().ok());
    CHECK(child_at(state, 0)->stops.load() == 1);
}

TEST_CASE("concurrent start calls launch only one child") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    Result<std::uint16_t> a(Status(ErrorCode::internal, "unset")), b(Status(ErrorCode::internal, "unset"));
    std::thread x([&] { a = s.start(); });
    std::thread y([&] { b = s.start(); });
    x.join();
    y.join();
    CHECK(a.ok());
    CHECK(b.ok());
    CHECK(state->starts.size() == 1);
    s.stop();
}

TEST_CASE("startup death fails promptly without treating a healthy port as the child") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.max_restarts = 0;
    options.health_check = [state](std::uint16_t, std::chrono::milliseconds) {
        child_at(state, 0)->alive.store(false);
        return true;
    };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    auto result = supervisor.start();
    CHECK_FALSE(result.ok());
    CHECK(result.status().code() == ErrorCode::unavailable);
    CHECK(child_at(state, 0)->stops.load() == 1);
}

TEST_CASE("shutdown interrupts restart backoff and is terminal") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.restart_initial_backoff = std::chrono::milliseconds(5000);
    options.restart_max_backoff = options.restart_initial_backoff;
    options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    REQUIRE(supervisor.start().ok());
    auto child = child_at(state, 0);
    child->alive.store(false);
    for (int i = 0; i < 1000 && child->stops.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto before = std::chrono::steady_clock::now();
    supervisor.stop();
    CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds(2));
    CHECK(state->starts.size() == 1);
    CHECK_FALSE(supervisor.start().ok());
}

TEST_CASE("cancelled readiness waiter leaves supervisor usable for other requests") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    std::atomic<bool> ready{false};
    options.health_check = [&](std::uint16_t, std::chrono::milliseconds) { return ready.load(); };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    CancellationSource cancel;
    Result<std::uint16_t> result(Status(ErrorCode::internal, "unset"));
    std::thread waiter([&] { result = supervisor.start(cancel.token()); });
    const bool launched = wait_for_starts(state, 1);
    cancel.cancel();
    waiter.join();
    REQUIRE(launched);
    CHECK(result.status().code() == ErrorCode::cancelled);
    ready.store(true);
    CHECK(supervisor.start().ok());
    supervisor.stop();
    CHECK(state->starts.size() == 1);
}

TEST_CASE("argument validation rejects NUL and host or port overrides") {
    CHECK_FALSE(validate_process_arguments({"--"}).ok());
    CHECK_FALSE(validate_process_arguments({"--host=0.0.0.0"}).ok());
    CHECK_FALSE(validate_process_arguments({"--port", "1"}).ok());
    CHECK_FALSE(validate_process_arguments({std::string("--model\0hidden", 13)}).ok());
    CHECK(validate_process_arguments({"-p", "prompt"}).ok());
}
