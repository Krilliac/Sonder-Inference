// Server lifecycle against a backend whose probe() and load_model() block:
// health and identity never wait on backend I/O (503 "starting" while a model
// loads, cached reachability once ready), and stop() is safe while start()
// runs, including from its on_listening callback.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"
#include "sonder/inference/backends.hpp"

using namespace server_test;

namespace {

// A gate that blocks callers until opened.
class Gate {
public:
    void close() {
        std::lock_guard<std::mutex> lock(mu_);
        open_ = false;
    }
    void open() {
        {
            std::lock_guard<std::mutex> lock(mu_);
            open_ = true;
        }
        cv_.notify_all();
    }
    void pass() {
        std::unique_lock<std::mutex> lock(mu_);
        ++waiting_;
        cv_.notify_all();
        cv_.wait(lock, [this] { return open_; });
        --waiting_;
    }
    bool wait_for_waiter(std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [this] { return waiting_ > 0; });
    }

private:
    std::mutex mu_;
    std::condition_variable cv_;
    bool open_ = true;
    int waiting_ = 0;
};

// The MOCK backend behind gates on load_model() and probe(), standing in for
// a remote backend (Ollama) that accepts connections and never answers.
class BlockingBackend final : public si::Backend {
public:
    std::shared_ptr<Gate> load = std::make_shared<Gate>();
    std::shared_ptr<Gate> probe_gate = std::make_shared<Gate>();
    std::atomic<int> probes{0};

    [[nodiscard]] std::string name() const override { return inner_->name(); }
    [[nodiscard]] std::string description() const override { return "mock behind test gates"; }
    [[nodiscard]] si::BackendCapabilities capabilities() const override { return inner_->capabilities(); }
    si::Result<std::string> probe() override {
        ++probes;
        probe_gate->pass();
        return inner_->probe();
    }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override { return inner_->list_models(); }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions& options) override {
        load->pass();
        return inner_->load_model(options);
    }

private:
    std::shared_ptr<si::Backend> inner_ = si::make_mock_backend();
};

srv::ServerOptions blocking_options(const std::shared_ptr<BlockingBackend>& backend) {
    auto o = Fixture::defaults();
    o.backend.backend.clear();
    o.backend_instance = backend;
    o.shutdown_grace = std::chrono::milliseconds(500);
    return o;
}

// GET with a hard deadline on the whole exchange.
Reply timed_get(std::uint16_t port, const std::string& target, std::chrono::milliseconds& took) {
    const auto t0 = std::chrono::steady_clock::now();
    Conn c(port);
    c.send(build_request("GET", target, port));
    Reply r = parse_reply(c.read_all(std::chrono::milliseconds(3000)));
    took = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0);
    return r;
}

}  // namespace

TEST_CASE("lifecycle: health answers 503 starting while a model load blocks, then 200") {
    auto backend = std::make_shared<BlockingBackend>();
    backend->load->close();
    srv::Server server(blocking_options(backend));
    std::promise<std::uint16_t> listening;
    std::future<si::Status> started = std::async(std::launch::async, [&] {
        return server.start([&] { listening.set_value(server.port()); });
    });
    const std::uint16_t port = listening.get_future().get();
    REQUIRE(backend->load->wait_for_waiter(std::chrono::milliseconds(5000)));

    std::chrono::milliseconds took{};
    const Reply health = timed_get(port, "/v1/sonder/health", took);
    CHECK(health.status == 503);
    CHECK(health.json().find("status")->as_string() == "starting");
    CHECK(took < std::chrono::milliseconds(1000));
    CHECK(timed_get(port, "/v1/sonder/identity", took).status == 503);
    CHECK(took < std::chrono::milliseconds(1000));

    backend->load->open();
    REQUIRE(started.get().ok());
    const Reply ready = timed_get(port, "/v1/sonder/health", took);
    CHECK(ready.status == 200);
    const auto body = ready.json();
    CHECK(body.find("backends")->as_array().at(0).find("available")->as_bool());
    server.stop();
}

TEST_CASE("lifecycle: a hung backend probe never blocks health or identity") {
    auto backend = std::make_shared<BlockingBackend>();
    srv::Server server(blocking_options(backend));
    REQUIRE(server.start().ok());
    const std::uint16_t port = server.port();
    std::chrono::milliseconds took{};
    REQUIRE(timed_get(port, "/v1/sonder/health", took).status == 200);

    // From now on every probe hangs (a backend that accepts and never replies).
    backend->probe_gate->close();
    REQUIRE(backend->probe_gate->wait_for_waiter(std::chrono::milliseconds(5000)));
    std::vector<std::future<std::pair<int, std::chrono::milliseconds>>> calls;
    for (int i = 0; i < 4; ++i) {
        calls.push_back(std::async(std::launch::async, [port, i] {
            std::chrono::milliseconds t{};
            const Reply r = timed_get(port, i % 2 == 0 ? "/v1/sonder/health" : "/v1/sonder/identity", t);
            return std::make_pair(r.status, t);
        }));
    }
    for (auto& c : calls) {
        const auto [status, t] = c.get();
        CHECK(status == 200);
        CHECK(t < std::chrono::milliseconds(1000));
    }
    // stop() does not wait out the hung probe either.
    const auto t0 = std::chrono::steady_clock::now();
    server.stop();
    CHECK(std::chrono::steady_clock::now() - t0 < std::chrono::seconds(5));
    backend->probe_gate->open();  // lets the detached refresher finish
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
}

TEST_CASE("lifecycle: stop() from the on_listening callback cancels start() cleanly") {
    auto o = Fixture::defaults();
    srv::Server server(o);
    const si::Status st = server.start([&] {
        std::thread stopper([&] { server.stop(); });
        stopper.join();
    });
    CHECK(st.code() == si::ErrorCode::cancelled);
    CHECK(server.engine() == nullptr);
    server.stop();  // idempotent
    server.wait();
}

TEST_CASE("lifecycle: stop() while a model load blocks reports draining and cancels the start") {
    auto backend = std::make_shared<BlockingBackend>();
    backend->load->close();
    srv::Server server(blocking_options(backend));
    std::promise<std::uint16_t> listening;
    std::future<si::Status> started = std::async(std::launch::async, [&] {
        return server.start([&] { listening.set_value(server.port()); });
    });
    const std::uint16_t port = listening.get_future().get();
    REQUIRE(backend->load->wait_for_waiter(std::chrono::milliseconds(5000)));
    std::future<void> stopped = std::async(std::launch::async, [&] { server.stop(); });
    // stop() waits for the load in progress; health meanwhile says draining.
    CHECK(eventually([&] {
        std::chrono::milliseconds took{};
        const Reply r = timed_get(port, "/v1/sonder/health", took);
        return r.status == 503 && r.json().find("status")->as_string() == "draining";
    }));
    CHECK(stopped.wait_for(std::chrono::milliseconds(200)) == std::future_status::timeout);
    backend->load->open();
    CHECK(started.get().code() == si::ErrorCode::cancelled);
    REQUIRE(stopped.wait_for(std::chrono::seconds(10)) == std::future_status::ready);
    CHECK(server.engine() == nullptr);
}
