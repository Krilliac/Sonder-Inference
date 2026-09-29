// Model residency (docs/SERVER.md "Model residency"): the defaults keep every
// model loaded from start() to stop(); --lazy-models registers them and loads
// each on first use (concurrent first requests share one load); an idle TTL
// and a max-resident cap evict unpinned models, never one a request holds.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"
#include "sonder/inference/backends.hpp"
#include "src/residency.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

// Blocks callers until opened (open by default).
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

// The MOCK backend, counting load_model() calls, with a gate on them.
class CountingBackend final : public si::Backend {
public:
    explicit CountingBackend(si::MockBackendOptions mock = {}) : inner_(si::make_mock_backend(mock)) {}

    std::shared_ptr<Gate> load_gate = std::make_shared<Gate>();
    std::atomic<int> loads{0};

    [[nodiscard]] std::string name() const override { return inner_->name(); }
    [[nodiscard]] std::string description() const override { return "mock counting loads"; }
    [[nodiscard]] si::BackendCapabilities capabilities() const override { return inner_->capabilities(); }
    si::Result<std::string> probe() override { return inner_->probe(); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override { return inner_->list_models(); }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions& options) override {
        ++loads;
        load_gate->pass();
        return inner_->load_model(options);
    }

private:
    std::shared_ptr<si::Backend> inner_;
};

srv::ServerOptions counting_options(const std::shared_ptr<CountingBackend>& backend) {
    auto o = Fixture::defaults();
    o.backend.backend.clear();
    o.backend_instance = backend;
    return o;
}

const json::Value* health_model(const json::Value& health, const std::string& id) {
    for (const auto& m : health.find("models")->as_array()) {
        if (m.find("id")->as_string() == id) return &m;
    }
    return nullptr;
}

std::string model_state(std::uint16_t port, const std::string& id) {
    const Reply h = get(port, "/v1/sonder/health");
    const json::Value doc = h.json();
    const json::Value* m = health_model(doc, id);
    return m != nullptr ? m->find("state")->as_string() : std::string("missing");
}

std::string chat_for(const std::string& model, int max_tokens = 4) {
    return R"({"model":")" + model + R"(","max_tokens":)" + std::to_string(max_tokens) +
           R"(,"messages":[{"role":"user","content":"hello"}]})";
}

// A model loaded on a real engine (mock backend) for the unit tests.
struct EngineModels {
    std::unique_ptr<si::Engine> engine;
    std::atomic<int> loads{0};
    EngineModels() {
        si::EngineOptions eo;
        eo.sample_devices_on_start = false;
        eo.device_sample_interval = std::chrono::milliseconds(0);
        engine = std::make_unique<si::Engine>(std::move(eo));
        REQUIRE(engine->register_backend(si::make_mock_backend()).ok());
    }
    si::Result<std::shared_ptr<si::Model>> load(const std::string& id) {
        ++loads;
        si::ModelLoadOptions lo;
        lo.model = id;
        return engine->load_model(si::kMockBackendName, lo);
    }
};

}  // namespace

// ------------------------------------------------------------ unit: ModelResidency

TEST_CASE("residency unit: concurrent first acquires share one load") {
    EngineModels em;
    auto gate = std::make_shared<Gate>();
    gate->close();
    auto r = std::make_shared<det::ModelResidency>(
        det::ResidencyConfig{},
        [&](const std::string& id) {
            gate->pass();
            return em.load(id);
        },
        [](det::Eviction&) {});
    REQUIRE(r->add("mock:tiny", "mock", true));
    constexpr int kThreads = 8;
    std::atomic<int> ok{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; ++i) {
        threads.emplace_back([&] {
            auto pin = r->acquire(i % 2 == 0 ? "mock:tiny" : "default");
            if (pin.ok() && pin.value().model()) ++ok;
        });
    }
    REQUIRE(gate->wait_for_waiter(std::chrono::milliseconds(5000)));
    // Every request is pinned (joined the load) before the load finishes.
    REQUIRE(eventually([&] { return r->snapshot()[0].pins == static_cast<std::uint64_t>(kThreads); }));
    CHECK(r->snapshot()[0].state == det::ResidencyState::loading);
    gate->open();
    for (auto& t : threads) t.join();
    CHECK(ok.load() == kThreads);
    CHECK(em.loads.load() == 1);
    const auto s = r->snapshot();
    CHECK(s[0].state == det::ResidencyState::resident);
    CHECK(s[0].pins == 0);
    CHECK(s[0].loads == 1);
}

TEST_CASE("residency unit: a failed load fails every joined request once, the next acquire retries") {
    std::atomic<int> calls{0};
    auto gate = std::make_shared<Gate>();
    gate->close();
    auto r = std::make_shared<det::ModelResidency>(
        det::ResidencyConfig{},
        [&](const std::string&) -> si::Result<std::shared_ptr<si::Model>> {
            ++calls;
            gate->pass();
            return si::Status(si::ErrorCode::unavailable, "backend down");
        },
        [](det::Eviction&) {});
    REQUIRE(r->add("m", "mock", true));
    std::atomic<int> failed{0};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) {
        threads.emplace_back([&] {
            auto pin = r->acquire("m");
            if (!pin.ok() && pin.status().code() == si::ErrorCode::unavailable) ++failed;
        });
    }
    REQUIRE(gate->wait_for_waiter(std::chrono::milliseconds(5000)));
    REQUIRE(eventually([&] { return r->snapshot()[0].pins == 4; }));
    gate->open();
    for (auto& t : threads) t.join();
    CHECK(failed.load() == 4);
    CHECK(calls.load() == 1);
    CHECK(r->snapshot()[0].pins == 0);
    CHECK(r->snapshot()[0].state == det::ResidencyState::unloaded);
    CHECK_FALSE(r->acquire("m").ok());
    CHECK(calls.load() == 2);
    CHECK(r->totals().load_failures == 2);
    CHECK(r->acquire("unknown").status().code() == si::ErrorCode::not_found);
}

TEST_CASE("residency unit: a pinned model is never evicted by the idle TTL") {
    EngineModels em;
    std::vector<std::string> evicted;
    auto r = std::make_shared<det::ModelResidency>(
        det::ResidencyConfig{std::chrono::milliseconds(1), 0}, [&](const std::string& id) { return em.load(id); },
        [&](det::Eviction& ev) { evicted.push_back(ev.id + ":" + ev.reason); });
    REQUIRE(r->add("mock:tiny", "mock", true));
    auto pinned = r->acquire("mock:tiny");
    REQUIRE(pinned.ok());
    const auto far = det::ModelResidency::Clock::now() + std::chrono::hours(1);
    CHECK(r->sweep(far) == 0);
    CHECK(r->snapshot()[0].state == det::ResidencyState::resident);
    CHECK(evicted.empty());
    {
        det::ModelResidency::Pin pin = std::move(pinned).value();
        CHECK(r->snapshot()[0].pins == 1);
    }  // released
    CHECK(r->snapshot()[0].pins == 0);
    CHECK(r->sweep(far) == 1);
    CHECK(r->snapshot()[0].state == det::ResidencyState::unloaded);
    REQUIRE(evicted.size() == 1);
    CHECK(evicted[0] == "mock:tiny:idle_ttl");
    // The next acquire loads it again.
    CHECK(r->acquire("mock:tiny").ok());
    CHECK(em.loads.load() == 2);
    CHECK(r->snapshot()[0].evictions == 1);
}

TEST_CASE("residency unit: defaults never evict") {
    EngineModels em;
    int evictions = 0;
    auto r = std::make_shared<det::ModelResidency>(
        det::ResidencyConfig{}, [&](const std::string& id) { return em.load(id); },
        [&](det::Eviction&) { ++evictions; });
    REQUIRE(r->add("mock:a", "mock", true));
    REQUIRE(r->add("mock:b", "mock", false));
    CHECK(r->acquire("mock:a").ok());
    CHECK(r->acquire("mock:b").ok());
    CHECK(r->sweep(det::ModelResidency::Clock::now() + std::chrono::hours(24 * 365)) == 0);
    r->start_sweeper();  // no thread without a TTL
    r->stop_sweeper();
    CHECK(evictions == 0);
    CHECK(r->totals().resident == 2);
    CHECK(em.loads.load() == 2);
}

TEST_CASE("residency unit: max-resident evicts the least recently used unpinned model; pinned ones stay") {
    EngineModels em;
    std::vector<std::string> evicted;
    auto r = std::make_shared<det::ModelResidency>(
        det::ResidencyConfig{std::chrono::milliseconds(0), 1}, [&](const std::string& id) { return em.load(id); },
        [&](det::Eviction& ev) { evicted.push_back(ev.id + ":" + ev.reason); });
    REQUIRE(r->add("mock:a", "mock", true));
    REQUIRE(r->add("mock:b", "mock", false));
    CHECK(r->acquire("mock:a").ok());
    CHECK(r->acquire("mock:b").ok());  // evicts a
    REQUIRE(evicted.size() == 1);
    CHECK(evicted[0] == "mock:a:max_resident");
    CHECK(r->totals().resident == 1);

    // Soft cap: with b pinned, loading a keeps both until b is released.
    auto pin_b = r->acquire("mock:b");
    REQUIRE(pin_b.ok());
    auto pin_a = r->acquire("mock:a");
    REQUIRE(pin_a.ok());
    CHECK(r->totals().resident == 2);
    CHECK(evicted.size() == 1);
    pin_b.value().release();  // b is now the only unpinned one: evicted
    CHECK(r->totals().resident == 1);
    REQUIRE(evicted.size() == 2);
    CHECK(evicted[1] == "mock:b:max_resident");
    CHECK(r->snapshot()[0].state == det::ResidencyState::resident);
}

TEST_CASE("residency unit: the sweeper thread evicts after the TTL") {
    EngineModels em;
    std::atomic<int> evicted{0};
    auto r = std::make_shared<det::ModelResidency>(
        det::ResidencyConfig{std::chrono::milliseconds(50), 0}, [&](const std::string& id) { return em.load(id); },
        [&](det::Eviction&) { ++evicted; });
    REQUIRE(r->add("mock:tiny", "mock", true));
    r->start_sweeper();
    CHECK(r->acquire("mock:tiny").ok());
    CHECK(eventually([&] { return evicted.load() == 1; }, std::chrono::milliseconds(3000)));
    CHECK(r->snapshot()[0].state == det::ResidencyState::unloaded);
    r->stop_sweeper();
}

// ------------------------------------------------------------ server

TEST_CASE("residency: defaults load every model before ready and never unload them") {
    auto backend = std::make_shared<CountingBackend>();
    auto o = counting_options(backend);
    o.models = {"mock:a", "mock:b"};
    Fixture f(o);
    // Both loaded during start(), before any request.
    CHECK(backend->loads.load() == 2);
    CHECK(f.of_type("model.load.completed").size() == 2);
    const json::Value h = get(f.port, "/v1/sonder/health").json();
    CHECK(h.find("status")->as_string() == "ready");
    const json::Value* res = h.find("residency");
    REQUIRE(res != nullptr);
    CHECK(res->find("mode")->as_string() == "eager");
    CHECK(res->find("idle_ttl_s")->is_null());
    CHECK(res->find("max_resident")->is_null());
    CHECK(res->find("registered")->as_int() == 2);
    CHECK(res->find("resident")->as_int() == 2);
    for (const auto& m : h.find("models")->as_array()) {
        // The pre-residency keys are unchanged.
        CHECK(m.find("backend")->as_string() == "mock");
        CHECK(m.find("default")->is_bool());
        CHECK(m.find("state")->as_string() == "resident");
        CHECK(m.find("resident")->as_bool());
    }
    for (int i = 0; i < 3; ++i) {
        CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:a")).status == 200);
        CHECK(post(f.port, "/v1/chat/completions", chat_for("default")).status == 200);
        CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:b")).status == 200);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    CHECK(backend->loads.load() == 2);
    CHECK(f.of_type("model.unload").empty());
    CHECK(f.of_type("model.evicted").empty());
    CHECK(model_state(f.port, "mock:a") == "resident");
}

TEST_CASE("residency: eager start still fails when a model cannot load") {
    auto o = Fixture::defaults();
    o.models = {"mock:tiny", "not-a-mock-model"};
    srv::Server server(o);
    const si::Status st = server.start();
    CHECK_FALSE(st.ok());
    server.stop();
}

TEST_CASE("residency: lazy models load on first request, not at start") {
    auto backend = std::make_shared<CountingBackend>();
    auto o = counting_options(backend);
    o.models = {"mock:a", "mock:b"};
    o.lazy_models = true;
    Fixture f(o);
    CHECK(backend->loads.load() == 0);
    CHECK(f.of_type("model.load.started").empty());
    const json::Value h = get(f.port, "/v1/sonder/health").json();
    CHECK(h.find("status")->as_string() == "ready");
    CHECK(h.find("residency")->find("mode")->as_string() == "lazy");
    CHECK(h.find("residency")->find("resident")->as_int() == 0);
    const json::Value* a = health_model(h, "mock:a");
    REQUIRE(a != nullptr);
    CHECK(a->find("state")->as_string() == "unloaded");
    CHECK_FALSE(a->find("resident")->as_bool());
    CHECK(a->find("model_instance_id")->is_null());
    // /v1/models lists registered models without loading them.
    const json::Value models = get(f.port, "/v1/models").json();
    CHECK(models.find("data")->as_array().size() == 2);
    // Identity never loads a model either.
    const json::Value ident = get(f.port, "/v1/sonder/identity?model=mock:a").json();
    CHECK(ident.find("backend_identity")->is_null());
    CHECK(backend->loads.load() == 0);

    CHECK(post(f.port, "/v1/chat/completions", chat_for("default")).status == 200);
    CHECK(backend->loads.load() == 1);
    CHECK(f.of_type("model.load.completed").size() == 1);
    CHECK(model_state(f.port, "mock:a") == "resident");
    CHECK(model_state(f.port, "mock:b") == "unloaded");
    CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:a")).status == 200);
    CHECK(backend->loads.load() == 1);
    // Resident now: identity is measured.
    const json::Value ident2 = get(f.port, "/v1/sonder/identity?model=mock:a").json();
    CHECK(ident2.find("backend_identity")->is_object());
}

TEST_CASE("residency: concurrent first requests on a lazy model load it once") {
    auto backend = std::make_shared<CountingBackend>();
    auto o = counting_options(backend);
    o.lazy_models = true;
    Fixture f(o);
    backend->load_gate->close();
    constexpr int kClients = 6;
    std::atomic<int> ok{0};
    std::vector<std::thread> clients;
    for (int i = 0; i < kClients; ++i) {
        clients.emplace_back([&] {
            if (post(f.port, "/v1/chat/completions", chat_for("mock:tiny")).status == 200) ++ok;
        });
    }
    REQUIRE(backend->load_gate->wait_for_waiter(std::chrono::milliseconds(5000)));
    // Every client is waiting on the one load (pinned in health) before it ends.
    const bool all_joined = eventually([&] {
        const json::Value h = get(f.port, "/v1/sonder/health").json();
        const json::Value* m = health_model(h, "mock:tiny");
        return m != nullptr && m->find("in_flight")->as_int() == kClients &&
               m->find("state")->as_string() == "loading";
    });
    backend->load_gate->open();
    for (auto& t : clients) t.join();
    CHECK(all_joined);
    CHECK(ok.load() == kClients);
    CHECK(backend->loads.load() == 1);
    CHECK(f.of_type("model.load.started").size() == 1);
}

TEST_CASE("residency: a lazy model that cannot load fails its requests, not the server") {
    auto o = Fixture::defaults();
    o.models = {"mock:tiny", "not-a-mock-model"};
    o.lazy_models = true;
    Fixture f(o);
    const Reply bad = post(f.port, "/v1/chat/completions", chat_for("not-a-mock-model"));
    CHECK(bad.status == 404);
    CHECK(bad.json().find("error")->find("code")->as_string() == "model_not_found");
    CHECK(f.of_type("model.load.failed").size() == 1);
    CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:tiny")).status == 200);
    const json::Value h = get(f.port, "/v1/sonder/health").json();
    CHECK(h.find("residency")->find("load_failures")->as_int() == 1);
}

TEST_CASE("residency: an idle TTL unloads the model and the next request reloads it") {
    auto backend = std::make_shared<CountingBackend>();
    auto o = counting_options(backend);
    o.model_idle_ttl = std::chrono::milliseconds(150);
    Fixture f(o);
    CHECK(backend->loads.load() == 1);  // eager by default
    const json::Value h = get(f.port, "/v1/sonder/health").json();
    CHECK(h.find("residency")->find("idle_ttl_s")->as_double() == doctest::Approx(0.15));
    REQUIRE(eventually([&] { return model_state(f.port, "mock:tiny") == "unloaded"; }));
    const auto evicted = f.of_type("model.evicted");
    REQUIRE(evicted.size() == 1);
    const json::Value* attrs = evicted[0].find("attributes");
    CHECK(attrs->find("model")->as_string() == "mock:tiny");
    CHECK(attrs->find("backend")->as_string() == "mock");
    CHECK(attrs->find("reason")->as_string() == "idle_ttl");
    CHECK(attrs->find("idle_ms")->as_double() >= 150.0);
    const auto unloads = f.of_type("model.unload");
    REQUIRE(unloads.size() == 1);
    CHECK(unloads[0].find("attributes")->find("outstanding_references")->as_int() == 0);

    CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:tiny")).status == 200);
    CHECK(backend->loads.load() == 2);
    const json::Value h2 = get(f.port, "/v1/sonder/health").json();
    CHECK(health_model(h2, "mock:tiny")->find("evictions")->as_int() >= 1);
    CHECK(h2.find("residency")->find("loads")->as_int() == 2);
}

TEST_CASE("residency: a request in flight pins its model past the idle TTL") {
    si::MockBackendOptions mock;
    mock.token_delay = std::chrono::milliseconds(25);
    mock.default_completion_tokens = 1000;
    auto backend = std::make_shared<CountingBackend>(mock);
    auto o = counting_options(backend);
    o.model_idle_ttl = std::chrono::milliseconds(60);
    o.lazy_models = true;
    Fixture f(o);
    std::atomic<int> status{0};
    // ~40 tokens x 25 ms = ~1 s, far past the 60 ms TTL.
    std::thread client([&] { status = post(f.port, "/v1/chat/completions", chat_for("mock:tiny", 40)).status; });
    REQUIRE(eventually([&] { return model_state(f.port, "mock:tiny") == "resident"; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    CHECK(model_state(f.port, "mock:tiny") == "resident");
    CHECK(f.of_type("model.evicted").empty());
    client.join();
    CHECK(status.load() == 200);
    CHECK(backend->loads.load() == 1);
    // Released: now it idles out.
    CHECK(eventually([&] { return model_state(f.port, "mock:tiny") == "unloaded"; }));
}

TEST_CASE("residency: --max-resident-models caps eager loading and evicts on demand") {
    auto backend = std::make_shared<CountingBackend>();
    auto o = counting_options(backend);
    o.models = {"mock:a", "mock:b"};
    o.max_resident_models = 1;
    Fixture f(o);
    CHECK(backend->loads.load() == 1);
    CHECK(model_state(f.port, "mock:a") == "resident");
    CHECK(model_state(f.port, "mock:b") == "unloaded");
    CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:b")).status == 200);
    CHECK(backend->loads.load() == 2);
    CHECK(model_state(f.port, "mock:a") == "unloaded");
    CHECK(model_state(f.port, "mock:b") == "resident");
    const auto evicted = f.of_type("model.evicted");
    REQUIRE(evicted.size() == 1);
    CHECK(evicted[0].find("attributes")->find("reason")->as_string() == "max_resident");
}

TEST_CASE("residency: stop() with lazy and TTL options shuts down cleanly") {
    auto o = Fixture::defaults();
    o.lazy_models = true;
    o.model_idle_ttl = std::chrono::milliseconds(10000);
    Fixture f(o);
    CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:tiny")).status == 200);
    f.server->stop();
    CHECK(f.of_type("model.evicted").empty());
}

TEST_CASE("residency: validate_options refuses a negative idle TTL") {
    auto o = Fixture::defaults();
    o.model_idle_ttl = std::chrono::milliseconds(-1);
    CHECK(srv::validate_options(o).code() == si::ErrorCode::invalid_argument);
    o.model_idle_ttl = std::chrono::milliseconds(0);
    CHECK(srv::validate_options(o).ok());
}
