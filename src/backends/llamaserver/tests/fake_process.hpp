// Fake process launcher for supervisor tests: records every launch (spec and
// time) and hands out children whose liveness the test controls. Shared by
// test_supervisor.cpp and test_runtime_guard.cpp.
#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

#include "../process.hpp"
#include "../supervisor.hpp"

namespace sonder_test {

struct ChildState {
    std::atomic<bool> alive{true};
    std::atomic<unsigned> stops{0};
    std::uint32_t pid = 0;
};

struct LaunchState {
    mutable std::mutex mutex;
    std::condition_variable changed;
    std::vector<std::shared_ptr<ChildState>> children;
    std::vector<std::chrono::steady_clock::time_point> starts;
    std::vector<sonder::inference::llamaserver::ProcessSpec> specs;
    // Process ids handed to children in launch order (1000, 1001, ... when 0).
    std::uint32_t next_pid = 0;
};

class FakeProcess final : public sonder::inference::llamaserver::Process {
  public:
    explicit FakeProcess(std::shared_ptr<ChildState> state) : state_(std::move(state)) {}
    bool running() const override { return state_->alive.load(std::memory_order_acquire); }
    void stop(std::chrono::milliseconds) override {
        state_->stops.fetch_add(1);
        state_->alive.store(false);
    }
    std::uint32_t pid() const override { return state_->pid; }

  private:
    std::shared_ptr<ChildState> state_;
};

class FakeLauncher final : public sonder::inference::llamaserver::ProcessLauncher {
  public:
    explicit FakeLauncher(std::shared_ptr<LaunchState> state) : state_(std::move(state)) {}
    sonder::inference::Result<std::unique_ptr<sonder::inference::llamaserver::Process>>
    start(const sonder::inference::llamaserver::ProcessSpec &spec) override {
        auto child = std::make_shared<ChildState>();
        {
            std::lock_guard lock(state_->mutex);
            child->pid = state_->next_pid ? state_->next_pid++ : 0;
            state_->children.push_back(child);
            state_->starts.push_back(std::chrono::steady_clock::now());
            state_->specs.push_back(spec);
        }
        state_->changed.notify_all();
        return std::unique_ptr<sonder::inference::llamaserver::Process>(new FakeProcess(std::move(child)));
    }

  private:
    std::shared_ptr<LaunchState> state_;
};

inline bool wait_for_starts(const std::shared_ptr<LaunchState> &state, std::size_t n) {
    std::unique_lock lock(state->mutex);
    return state->changed.wait_for(lock, std::chrono::seconds(3), [&] { return state->starts.size() >= n; });
}

inline std::shared_ptr<ChildState> child_at(const std::shared_ptr<LaunchState> &state, std::size_t index) {
    std::lock_guard lock(state->mutex);
    return state->children.at(index);
}

inline sonder::inference::llamaserver::SupervisorOptions base_options() {
    sonder::inference::llamaserver::SupervisorOptions o;
    o.executable = "fake-llama-server";
    o.arguments = {"--model", "model.gguf", "--spec-type", "draft-mtp"};
    o.readiness_timeout = std::chrono::milliseconds(2000);
    o.health_poll_interval = std::chrono::milliseconds(2);
    o.restart_initial_backoff = std::chrono::milliseconds(5);
    o.restart_max_backoff = std::chrono::milliseconds(8);
    return o;
}

} // namespace sonder_test
