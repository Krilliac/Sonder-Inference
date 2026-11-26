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
        threads.emplace_back([&, i] {
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

TEST_CASE("residency unit: a failed load fails every joined request once and the next acquire retries") {
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
    const auto much_later = det::ModelResidency::Clock::now() + std::chrono::hours(1);
    CHECK(r->sweep(much_later) == 0);
    CHECK(r->snapshot()[0].state == det::ResidencyState::resident);
    CHECK(evicted.empty());
    {
        det::ModelResidency::Pin pin = std::move(pinned).value();
        CHECK(r->snapshot()[0].pins == 1);
    }  // released
    CHECK(r->snapshot()[0].pins == 0);
    CHECK(r->sweep(much_later) == 1);
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

TEST_CASE("residency unit: max-resident evicts the least recently used unpinned model and keeps pinned ones") {
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
    // `unloaded` is reported only once the evictor has returned.
    CHECK(eventually([&] { return r->snapshot()[0].state == det::ResidencyState::unloaded; },
                     std::chrono::milliseconds(3000)));
    CHECK(evicted.load() == 1);
    r->stop_sweeper();
}

TEST_CASE("residency unit: an evicted model reads unloaded only after its release and a reload waits for it") {
    EngineModels em;
    Gate evict_gate;
    evict_gate.close();
    std::shared_ptr<det::ModelResidency> r;
    std::atomic<int> seen_state{-1};
    r = std::make_shared<det::ModelResidency>(
        det::ResidencyConfig{std::chrono::milliseconds(1), 0}, [&](const std::string& id) { return em.load(id); },
        [&](det::Eviction&) {
            seen_state = static_cast<int>(r->snapshot()[0].state);
            evict_gate.pass();  // the old handle is still being released
        });
    REQUIRE(r->add("mock:tiny", "mock", true));
    REQUIRE(r->acquire("mock:tiny").ok());  // loaded, then released
    const auto much_later = det::ModelResidency::Clock::now() + std::chrono::hours(1);
    std::thread sweeper([&] { (void)r->sweep(much_later); });
    REQUIRE(evict_gate.wait_for_waiter(std::chrono::milliseconds(3000)));
    CHECK(seen_state.load() == static_cast<int>(det::ResidencyState::unloading));
    CHECK(r->snapshot()[0].state == det::ResidencyState::unloading);
    CHECK(r->totals().unloading == 1);
    CHECK(r->totals().resident == 0);
    std::atomic<bool> reloaded{false};
    std::thread reloader([&] {
        auto pin = r->acquire("mock:tiny");
        reloaded = pin.ok();
    });
    REQUIRE(eventually([&] { return r->snapshot()[0].pins == 1; }));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    // No second copy while the old one is still being released.
    CHECK(em.loads.load() == 1);
    CHECK_FALSE(reloaded.load());
    evict_gate.open();
    sweeper.join();
    reloader.join();
    CHECK(reloaded.load());
    CHECK(em.loads.load() == 2);
    CHECK(r->snapshot()[0].state == det::ResidencyState::resident);
    CHECK(r->totals().unloading == 0);
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

TEST_CASE("residency: lazy models load on first request and not at start") {
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

TEST_CASE("residency: a lazy model that cannot load fails its requests and not the server") {
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
    // `unloaded` is reported only after model.evicted and model.unload were
    // emitted and the old handle released, so no wait is needed below.
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

// ------------------------------------------------------------ Item #18: byte-aware & reload-cost residency

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

det::ModelResourceFootprint make_vram_fp(std::uint64_t weights_mib, std::uint64_t kv_mib = 0,
                                         std::uint64_t workspace_mib = 0, std::uint64_t projector_mib = 0,
                                         std::uint64_t adapter_mib = 0, std::uint64_t ram_weights_mib = 0) {
    det::ModelResourceFootprint fp;
    fp.vram.weights_bytes = weights_mib * kMiB;
    fp.vram.kv_cache_bytes = kv_mib * kMiB;
    fp.vram.workspace_bytes = workspace_mib * kMiB;
    fp.vram.projector_bytes = projector_mib * kMiB;
    fp.vram.adapter_bytes = adapter_mib * kMiB;
    fp.ram.weights_bytes = ram_weights_mib * kMiB;
    return fp;
}

}  // namespace

TEST_CASE("residency unit: ResourceFootprint accounts for weights, KV, workspace, projector, and adapter bytes") {
    det::ResourceFootprint fp;
    CHECK(fp.empty());
    CHECK(fp.total_bytes() == 0u);
    fp.weights_bytes = 2000 * kMiB;
    fp.kv_cache_bytes = 500 * kMiB;
    fp.workspace_bytes = 256 * kMiB;
    fp.projector_bytes = 128 * kMiB;
    fp.adapter_bytes = 64 * kMiB;
    CHECK_FALSE(fp.empty());
    CHECK(fp.total_bytes() == (2000 + 500 + 256 + 128 + 64) * kMiB);
}

TEST_CASE("residency unit: pre-load byte reservation admits or refuses concurrently and rolls back immediately on load error") {
    EngineModels em;
    auto gate = std::make_shared<Gate>();
    gate->close();
    std::atomic<bool> fail_a{true};
    det::ResidencyConfig cfg;
    cfg.max_resident_vram_bytes = 8 * kGiB;

    auto r = std::make_shared<det::ModelResidency>(
        cfg,
        [&](const std::string& id) -> si::Result<std::shared_ptr<si::Model>> {
            if (id == "mock:a") {
                gate->pass();
                if (fail_a.load()) {
                    return si::Status(si::ErrorCode::unavailable, "simulated OOM during cold load");
                }
            }
            return em.load(id);
        },
        [](det::Eviction&) {});

    REQUIRE(r->add("mock:a", "mock", true, make_vram_fp(5120, 512, 256, 128, 128)));  // 6 GiB total VRAM
    REQUIRE(r->add("mock:b", "mock", false, make_vram_fp(3072, 1024)));              // 4 GiB total VRAM

    si::Status status_a;
    std::thread loader_a([&] {
        auto pin = r->acquire("mock:a");
        status_a = pin.ok() ? si::Status::success() : pin.status();
    });
    REQUIRE(gate->wait_for_waiter(std::chrono::milliseconds(5000)));
    REQUIRE(eventually([&] { return r->totals().loading == 1u; }));

    // While mock:a is mid-load, its 6 GiB reservation is held in totals and snapshot.
    const auto mid_totals = r->totals();
    CHECK(mid_totals.reserved_vram_bytes == 6 * kGiB);
    CHECK(mid_totals.resident_vram_bytes == 0u);
    CHECK(r->snapshot()[0].reserved_footprint.vram_bytes() == 6 * kGiB);

    // Concurrent cold load of mock:b (4 GiB) sees 6 + 4 = 10 GiB > 8 GiB budget and is refused cleanly.
    const auto refused = r->acquire("mock:b");
    CHECK_FALSE(refused.ok());
    CHECK(refused.status().code() == si::ErrorCode::unavailable);
    CHECK(r->totals().admission_rejections == 1u);
    CHECK(r->totals().reserved_vram_bytes == 6 * kGiB);  // mock:b leaked zero reservation

    // Let mock:a fail: its 6 GiB reservation must be released immediately.
    gate->open();
    loader_a.join();
    CHECK_FALSE(status_a.ok());
    const auto after_fail = r->totals();
    CHECK(after_fail.loading == 0u);
    CHECK(after_fail.resident == 0u);
    CHECK(after_fail.reserved_vram_bytes == 0u);
    CHECK(after_fail.resident_vram_bytes == 0u);
    CHECK(after_fail.load_failures == 1u);
    CHECK(r->snapshot()[0].reserved_footprint.empty());

    // Now mock:b (4 GiB) admits and loads cleanly.
    auto pin_b = r->acquire("mock:b");
    REQUIRE(pin_b.ok());
    CHECK(r->totals().resident_vram_bytes == 4 * kGiB);
    CHECK(r->totals().reserved_vram_bytes == 0u);
}

TEST_CASE("residency unit: post-load reconciliation updates observed bytes and enforces budget on overflow") {
    EngineModels em;
    std::vector<std::string> evicted;
    det::ResidencyConfig cfg;
    cfg.max_resident_vram_bytes = 8 * kGiB;

    auto r = std::make_shared<det::ModelResidency>(
        cfg, [&](const std::string& id) { return em.load(id); },
        [&](det::Eviction& ev) { evicted.push_back(ev.id + ":" + ev.reason); },
        [](const std::string& id, const std::shared_ptr<si::Model>&,
           const det::ModelResourceFootprint& reserved) -> std::optional<det::ModelResourceFootprint> {
            if (id == "mock:grow") {
                // Pre-load estimated 2 GiB, actual observed allocation is 6 GiB (with projector + adapter).
                return make_vram_fp(4096, 1024, 512, 256, 256);
            }
            return reserved;
        });

    REQUIRE(r->add("mock:base", "mock", true, make_vram_fp(4096)));   // 4 GiB
    REQUIRE(r->add("mock:grow", "mock", false, make_vram_fp(2048)));  // 2 GiB est -> 6 GiB observed

    // Load mock:base (4 GiB) and release pin so it is evictable.
    REQUIRE(r->acquire("mock:base").ok());
    CHECK(r->totals().resident_vram_bytes == 4 * kGiB);

    // Load mock:grow: pre-load 4 + 2 = 6 GiB fits without pre-eviction, then post-load
    // observer reconciles mock:grow to 6 GiB -> 4 + 6 = 10 GiB > 8 GiB -> evicts unpinned mock:base!
    auto pin_grow = r->acquire("mock:grow");
    REQUIRE(pin_grow.ok());
    REQUIRE(evicted.size() == 1u);
    CHECK(evicted[0] == "mock:base:vram_budget");
    CHECK(r->totals().resident == 1u);
    CHECK(r->totals().resident_vram_bytes == 6 * kGiB);
    CHECK(r->totals().reserved_vram_bytes == 0u);

    const auto snap = r->snapshot();
    CHECK(snap[1].observed_footprint.vram.weights_bytes == 4096 * kMiB);
    CHECK(snap[1].observed_footprint.vram.kv_cache_bytes == 1024 * kMiB);
    CHECK(snap[1].observed_footprint.vram.workspace_bytes == 512 * kMiB);
    CHECK(snap[1].observed_footprint.vram.projector_bytes == 256 * kMiB);
    CHECK(snap[1].observed_footprint.vram.adapter_bytes == 256 * kMiB);

    // Explicit live reconcile_footprint() updates resident bytes immediately (e.g. adapter attach).
    CHECK(r->reconcile_footprint("mock:grow", make_vram_fp(4096, 1024, 512, 256, 512)));
    CHECK(r->totals().resident_vram_bytes == (4096 + 1024 + 512 + 256 + 512) * kMiB);
    CHECK_FALSE(r->reconcile_footprint("mock:base", make_vram_fp(1024)));  // not resident
}

TEST_CASE("residency unit: cost-per-byte victim selection keeps three small hot models over one huge cheap-reload model and never evicts pins") {
    EngineModels em;
    std::vector<std::string> evicted_ids;
    std::vector<std::string> evicted_reasons;
    det::ResidencyConfig cfg;
    cfg.max_resident_vram_bytes = 10 * kGiB;

    auto r = std::make_shared<det::ModelResidency>(
        cfg, [&](const std::string& id) { return em.load(id); },
        [&](det::Eviction& ev) {
            evicted_ids.push_back(ev.id);
            evicted_reasons.push_back(ev.reason);
        });

    REQUIRE(r->add("mock:small1", "mock", true, make_vram_fp(1024)));   // 1 GiB
    REQUIRE(r->add("mock:small2", "mock", false, make_vram_fp(1024)));  // 1 GiB
    REQUIRE(r->add("mock:small3", "mock", false, make_vram_fp(1024)));  // 1 GiB
    REQUIRE(r->add("mock:huge", "mock", false, make_vram_fp(6144)));    // 6 GiB
    REQUIRE(r->add("mock:incoming", "mock", false, make_vram_fp(4096))); // 4 GiB

    // Load all three small models + the one huge model (total 9 GiB <= 10 GiB).
    REQUIRE(r->acquire("mock:small1").ok());
    REQUIRE(r->acquire("mock:small2").ok());
    REQUIRE(r->acquire("mock:small3").ok());
    // Make mock:small1..3 hot (higher demand + expensive reload), and load mock:huge LAST
    // so pure LRU would have evicted mock:small1..3 instead of mock:huge!
    for (int i = 0; i < 8; ++i) {
        REQUIRE(r->acquire("mock:small1").ok());
        REQUIRE(r->acquire("mock:small2").ok());
        REQUIRE(r->acquire("mock:small3").ok());
    }
    REQUIRE(r->acquire("mock:huge").ok());
    REQUIRE(r->set_reload_ms("mock:small1", 500.0));
    REQUIRE(r->set_reload_ms("mock:small2", 500.0));
    REQUIRE(r->set_reload_ms("mock:small3", 500.0));
    REQUIRE(r->set_reload_ms("mock:huge", 50.0));
    CHECK(r->totals().resident == 4u);
    CHECK(r->totals().resident_vram_bytes == 9 * kGiB);

    // Confirm eviction cost formula ordering: mock:huge has far lower cost/byte.
    const double cost_small = det::ModelResidency::compute_eviction_cost(9.0, 500.0, 1 * kGiB);
    const double cost_huge = det::ModelResidency::compute_eviction_cost(1.0, 50.0, 6 * kGiB);
    CHECK(cost_huge < cost_small);

    // Loading mock:incoming (4 GiB) needs 3 GiB freed: MUST evict ONLY mock:huge (6 GiB),
    // keeping all three small hot models resident!
    {
        auto pin_in = r->acquire("mock:incoming");
        REQUIRE(pin_in.ok());
    }
    REQUIRE(evicted_ids.size() == 1u);
    CHECK(evicted_ids[0] == "mock:huge");
    CHECK(evicted_reasons[0] == det::kEvictVramBudget);
    CHECK(r->resident_model("mock:small1") != nullptr);
    CHECK(r->resident_model("mock:small2") != nullptr);
    CHECK(r->resident_model("mock:small3") != nullptr);
    CHECK(r->resident_model("mock:incoming") != nullptr);
    CHECK(r->resident_model("mock:huge") == nullptr);
    CHECK(r->totals().resident_vram_bytes == 7 * kGiB);

    // Pins are NEVER evicted: pin mock:small1..3 and mock:incoming (7 GiB pinned total).
    // Reloading mock:huge (6 GiB) needs 7 + 6 = 13 GiB > 10 GiB -> refused without evicting any pin!
    auto p1 = r->acquire("mock:small1");
    auto p2 = r->acquire("mock:small2");
    auto p3 = r->acquire("mock:small3");
    auto pin_in = r->acquire("mock:incoming");
    REQUIRE((p1.ok() && p2.ok() && p3.ok() && pin_in.ok()));
    const auto blocked = r->acquire("mock:huge");
    CHECK_FALSE(blocked.ok());
    CHECK(blocked.status().code() == si::ErrorCode::unavailable);
    CHECK(evicted_ids.size() == 1u);  // zero pinned models evicted
    CHECK(r->resident_model("mock:small1") != nullptr);
    CHECK(r->resident_model("mock:incoming") != nullptr);

    // Release mock:incoming (4 GiB) while mock:small1..3 (3 GiB) stay pinned:
    // now 3 GiB pinned + 6 GiB incoming = 9 GiB <= 10 GiB -> evicts unpinned mock:incoming, keeps pinned mock:small1..3!
    pin_in.value().release();
    auto pin_huge = r->acquire("mock:huge");
    REQUIRE(pin_huge.ok());
    REQUIRE(evicted_ids.size() == 2u);
    CHECK(evicted_ids[1] == "mock:incoming");
    CHECK(r->resident_model("mock:small1") != nullptr);
    CHECK(r->resident_model("mock:small2") != nullptr);
    CHECK(r->resident_model("mock:small3") != nullptr);
}

TEST_CASE("residency unit: reload_ms is measured automatically and RAM budget evicts RAM victims without touching GPU-only models") {
    EngineModels em;
    std::vector<std::string> evicted;
    det::ResidencyConfig cfg;
    cfg.max_resident_ram_bytes = 4 * kGiB;

    auto r = std::make_shared<det::ModelResidency>(
        cfg,
        [&](const std::string& id) {
            if (id == "mock:ram_slow") {
                std::this_thread::sleep_for(std::chrono::milliseconds(15));
            }
            return em.load(id);
        },
        [&](det::Eviction& ev) { evicted.push_back(ev.id + ":" + ev.reason); });

    REQUIRE(r->add("mock:gpu_only", "mock", true, make_vram_fp(4096, 0, 0, 0, 0, 0)));  // 4 GiB VRAM, 0 RAM
    REQUIRE(r->add("mock:ram_slow", "mock", false, make_vram_fp(0, 0, 0, 0, 0, 2048))); // 2 GiB RAM
    REQUIRE(r->add("mock:ram_fast", "mock", false, make_vram_fp(0, 0, 0, 0, 0, 2048))); // 2 GiB RAM
    REQUIRE(r->add("mock:ram_new", "mock", false, make_vram_fp(0, 0, 0, 0, 0, 2048)));  // 2 GiB RAM

    REQUIRE(r->acquire("mock:gpu_only").ok());
    REQUIRE(r->acquire("mock:ram_slow").ok());
    REQUIRE(r->acquire("mock:ram_fast").ok());

    const auto snaps = r->snapshot();
    CHECK(snaps[1].reload_ms >= 10.0);  // measured wall-clock load time

    // Make mock:ram_slow strictly more expensive to reload than mock:ram_fast so mock:ram_fast is evicted first,
    // and mock:gpu_only (0 RAM) is never touched for RAM budget pressure.
    REQUIRE(r->set_reload_ms("mock:ram_slow", 400.0));
    REQUIRE(r->set_reload_ms("mock:ram_fast", 10.0));
    REQUIRE(r->acquire("mock:ram_new").ok());

    REQUIRE(evicted.size() == 1u);
    CHECK(evicted[0] == "mock:ram_fast:ram_budget");
    CHECK(r->resident_model("mock:gpu_only") != nullptr);
    CHECK(r->resident_model("mock:ram_slow") != nullptr);
    CHECK(r->resident_model("mock:ram_new") != nullptr);
    CHECK(r->totals().resident_ram_bytes == 4 * kGiB);
}

TEST_CASE("residency: server enforces max_resident_vram_bytes using profile footprints and exposes byte totals in health") {
    auto backend = std::make_shared<CountingBackend>();
    auto o = counting_options(backend);
    o.models = {"mock:a", "mock:b"};
    o.lazy_models = true;
    o.max_resident_vram_bytes = 5 * kGiB;

    srv::ModelProfileBinding ba;
    ba.model = "mock:a";
    ba.estimated_footprint = make_vram_fp(3072, 512, 512);  // 4 GiB
    srv::ModelProfileBinding bb;
    bb.model = "mock:b";
    bb.estimated_footprint = make_vram_fp(2560, 512, 0);    // 3 GiB
    o.profile_bindings = {ba, bb};

    Fixture f(o);
    CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:a")).status == 200);
    const json::Value h1 = get(f.port, "/v1/sonder/health").json();
    CHECK(h1.find("residency")->find("max_resident_vram_bytes")->as_uint() == 5 * kGiB);
    CHECK(h1.find("residency")->find("resident_vram_bytes")->as_uint() == 4 * kGiB);
    const json::Value* ma = health_model(h1, "mock:a");
    REQUIRE(ma != nullptr);
    CHECK(ma->find("footprint")->find("vram")->find("total_bytes")->as_uint() == 4 * kGiB);

    // Loading mock:b (3 GiB) exceeds 5 GiB with mock:a (4 GiB) -> evicts mock:a with vram_budget.
    CHECK(post(f.port, "/v1/chat/completions", chat_for("mock:b")).status == 200);
    CHECK(model_state(f.port, "mock:a") == "unloaded");
    CHECK(model_state(f.port, "mock:b") == "resident");
    const auto evicted = f.of_type("model.evicted");
    REQUIRE(evicted.size() == 1u);
    const json::Value* attrs = evicted[0].find("attributes");
    CHECK(attrs->find("reason")->as_string() == "vram_budget");
    CHECK(attrs->find("freed_vram_bytes")->as_uint() == 4 * kGiB);
}
