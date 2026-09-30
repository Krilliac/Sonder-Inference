#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>

#include "../prefix_warmup.hpp"
#include "../process.hpp"
#include "../slot_affinity.hpp"
#include "../supervisor.hpp"
#include "fake_process.hpp"
#include "fake_server.hpp"
#include "sonder/inference/backends/llamaserver.hpp"

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;

namespace {

struct PrefixFile {
    std::filesystem::path path;
    explicit PrefixFile(const std::string &text) {
        const auto id = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::current_path() / (".warmup-test-" + std::to_string(id) + ".json");
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << text;
    }
    ~PrefixFile() {
        std::error_code ec;
        std::filesystem::remove(path, ec);
    }
};

bool eventually(const std::function<bool()> &predicate, std::chrono::milliseconds timeout = std::chrono::seconds(3)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (predicate())
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    return predicate();
}

LlamaServerBackendOptions options(const std::string &file) {
    LlamaServerBackendOptions o;
    o.warmup.messages_file = file;
    o.warmup.chat_template_kwargs = json::Object{{"enable_thinking", false}, {"reasoning_effort", "low"}};
    o.warmup.max_prefix_chars = 262144;
    o.connect_timeout = std::chrono::milliseconds(500);
    o.request_timeout = std::chrono::seconds(5);
    o.startup_timeout = std::chrono::seconds(2);
    o.poll_interval = std::chrono::milliseconds(2);
    return o;
}

json::Value parse_body(const std::string &body) {
    auto parsed = json::parse(body);
    REQUIRE_MESSAGE(parsed.ok(), "warm-up body is not JSON: " << body);
    return parsed.value();
}

} // namespace

TEST_SUITE("llamaserver_warmup") {

TEST_CASE("replays each selected slot with the nonstream cache request shape and reports counters") {
    PrefixFile prefix(R"([{"role":"system","content":"stable system"},{"role":"user","content":"tools text"}])");
    sonder_test::FakeLlamaServer server;
    server.set_props(R"({"default_generation_settings":{"n_ctx":8192},"total_slots":2})");

    auto o = options(prefix.path.string());
    PrefixWarmup warmup(o);
    const auto began = std::chrono::steady_clock::now();
    warmup.start(server.url());
    CHECK(std::chrono::steady_clock::now() - began < std::chrono::milliseconds(500));
    REQUIRE(eventually([&] { return server.nonstream_bodies().size() == 2; }));
    REQUIRE(eventually([&] {
        const auto status = warmup.status();
        return status && status->status == "complete";
    }));

    const auto requests = server.nonstream_bodies();
    REQUIRE(requests.size() == 2);
    for (std::size_t i = 0; i < requests.size(); ++i) {
        const auto body = parse_body(requests[i]);
        CHECK(body.find("id_slot")->as_uint() == i);
        CHECK(body.find("cache_prompt")->as_bool());
        CHECK(body.find("max_tokens")->as_uint() == 1);
        CHECK_FALSE(body.find("stream")->as_bool());
        CHECK(body.find("chat_template_kwargs")->find("enable_thinking")->as_bool() == false);
        CHECK(body.find("chat_template_kwargs")->find("reasoning_effort")->as_string() == "low");
        REQUIRE(body.find("messages")->is_array());
        CHECK(body.find("messages")->as_array().size() == 2);
    }
    const auto status = warmup.status();
    REQUIRE(status);
    REQUIRE(status->slots.size() == 2);
    for (const auto &slot : status->slots) {
        CHECK(slot.status == "complete");
        REQUIRE(slot.prompt_tokens);
        CHECK(*slot.prompt_tokens == 12);
        REQUIRE(slot.cache_n);
        CHECK(*slot.cache_n == 9);
        CHECK(slot.milliseconds >= 0.0);
    }
    CHECK(warmup.warmed_slots() == std::vector<std::uint32_t>{0, 1});
    CHECK(warmup.warnings().empty());
}

TEST_CASE("subset selection warms only requested slots") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    sonder_test::FakeLlamaServer server;
    server.set_props(R"({"total_slots":4,"default_generation_settings":{"n_ctx":8192}})");
    auto o = options(prefix.path.string());
    o.warmup.all_slots = false;
    o.warmup.slots = {3, 1};
    PrefixWarmup warmup(o);
    warmup.start(server.url());
    REQUIRE(eventually([&] { return server.nonstream_bodies().size() == 2; }));
    REQUIRE(eventually([&] { return warmup.status()->status == "complete"; }));
    const auto bodies = server.nonstream_bodies();
    CHECK(parse_body(bodies[0]).find("id_slot")->as_uint() == 1);
    CHECK(parse_body(bodies[1]).find("id_slot")->as_uint() == 3);
    CHECK(warmup.warmed_slots() == std::vector<std::uint32_t>{1, 3});
}

TEST_CASE("warmed slots are preferred for new affinity keys") {
    llamaserver::SlotAffinity affinity;
    auto first = affinity.acquire("first", 4, {2, 3});
    REQUIRE(first.slot());
    CHECK(*first.slot() == 2u);
    first.reset();
    auto second = affinity.acquire("second", 4, {2, 3});
    REQUIRE(second.slot());
    CHECK(*second.slot() == 3u);
    second.reset();
    auto ordinary = affinity.acquire("ordinary", 4);
    REQUIRE(ordinary.slot());
    CHECK(*ordinary.slot() == 0u);
}

namespace {
struct SupervisorChild final : Process {
    std::shared_ptr<sonder_test::ChildState> state;
    std::shared_ptr<sonder_test::FakeLlamaServer> server;
    bool running() const override { return state->alive.load(std::memory_order_acquire); }
    void stop(std::chrono::milliseconds) override {
        state->stops.fetch_add(1);
        state->alive.store(false, std::memory_order_release);
        if (server)
            server->stop();
    }
    std::uint32_t pid() const override { return state->pid; }
};

struct HttpLauncher final : ProcessLauncher {
    std::shared_ptr<sonder_test::LaunchState> state = std::make_shared<sonder_test::LaunchState>();
    std::vector<std::shared_ptr<sonder_test::FakeLlamaServer>> servers;
    std::vector<std::shared_ptr<sonder_test::ChildState>> children;
    bool hold_warmup = false;
    mutable std::mutex mutex;

    Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
        auto child = std::make_shared<sonder_test::ChildState>();
        {
            std::lock_guard lock(mutex);
            child->pid = static_cast<std::uint32_t>(1000 + servers.size());
        }
        auto server = std::make_shared<sonder_test::FakeLlamaServer>(spec.port);
        server->set_props(R"({"total_slots":1,"default_generation_settings":{"n_ctx":8192}})");
        server->hold_nonstream(hold_warmup);
        {
            std::lock_guard lock(mutex);
            servers.push_back(server);
            children.push_back(child);
        }
        auto result = std::make_unique<SupervisorChild>();
        result->state = child;
        result->server = server;
        return std::unique_ptr<Process>(std::move(result));
    }
    std::size_t server_count() const {
        std::lock_guard lock(mutex);
        return servers.size();
    }
    std::shared_ptr<sonder_test::FakeLlamaServer> server_at(std::size_t index) const {
        std::lock_guard lock(mutex);
        return servers.at(index);
    }
    std::shared_ptr<sonder_test::ChildState> child_at(std::size_t index) const {
        std::lock_guard lock(mutex);
        return children.at(index);
    }
};

// No PDH/GPU activity: only the first fake child spills. The accepted refit
// and later crash replacement get clean synthetic counters.
class FitCounters final : public GpuCounterSource {
  public:
    std::string name() const override { return "fake"; }
    bool supported() const override { return true; }
    Result<GpuProcessCounters> read(std::uint32_t pid) override {
        const auto name = "pid_" + std::to_string(pid) + "_fake";
        return GpuProcessCounters{{{name, 14000 * kMiB}}, {{name, (pid == 1000 ? 512 : 128) * kMiB}}};
    }
};
} // namespace

TEST_CASE("warm-up runs on the accepted auto_fit child and its crash replacement only") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    auto launcher = std::make_unique<HttpLauncher>();
    auto *observed = launcher.get();
    auto warmup = std::make_shared<PrefixWarmup>(options(prefix.path.string()));
    auto o = sonder_test::base_options();
    o.arguments = {"--ctx-size", "16384"};
    o.spill_guard.policy = SpillPolicy::auto_fit;
    o.spill_guard.step_factor = 0.5;
    o.gpu_counters = std::make_shared<FitCounters>();
    o.warmup = warmup;
    Supervisor supervisor(o, std::move(launcher));
    REQUIRE(supervisor.start().ok());
    REQUIRE(observed->server_count() == 2);
    CHECK(observed->server_at(0)->nonstream_bodies().empty());
    REQUIRE(eventually([&] { return warmup->status()->status == "complete"; }));
    CHECK(observed->server_at(1)->nonstream_bodies().size() == 1);
    CHECK(warmup->status()->generation == 1);
    CHECK(supervisor.runtime_status().context.fitted_ctx == std::uint64_t{8192});
    observed->child_at(1)->alive.store(false, std::memory_order_release);
    REQUIRE(eventually([&] { return observed->server_count() == 3; }));
    REQUIRE(eventually([&] { return warmup->status()->generation == 2 &&
                                    warmup->status()->status == "complete"; }));
    CHECK(observed->server_at(2)->nonstream_bodies().size() == 1);
    supervisor.stop();
}

TEST_CASE("supervisor starts warm-up after readiness and repeats it for a restarted child") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    auto launcher = std::make_unique<HttpLauncher>();
    auto *launcher_ptr = launcher.get();
    auto warmup_options = options(prefix.path.string());
    auto warmup = std::make_shared<PrefixWarmup>(warmup_options);
    auto supervisor_options = sonder_test::base_options();
    supervisor_options.spill_guard.enabled = false;
    supervisor_options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    supervisor_options.warmup = warmup;
    supervisor_options.restart_initial_backoff = std::chrono::milliseconds(2);
    supervisor_options.restart_max_backoff = std::chrono::milliseconds(4);
    Supervisor supervisor(supervisor_options, std::move(launcher));
    REQUIRE(supervisor.start().ok());
    REQUIRE(eventually([&] { return launcher_ptr->server_count() == 1 &&
                                    launcher_ptr->server_at(0)->nonstream_bodies().size() == 1; }));
    launcher_ptr->child_at(0)->alive.store(false, std::memory_order_release);
    REQUIRE(eventually([&] { return launcher_ptr->server_count() == 2; }));
    REQUIRE(eventually([&] { return launcher_ptr->server_at(1)->nonstream_bodies().size() == 1; }));
    REQUIRE(warmup->status());
    CHECK(warmup->status()->generation == 2);
    supervisor.stop();
    CHECK((warmup->status()->status == "cancelled" || warmup->status()->status == "complete"));
}

TEST_CASE("supervisor readiness returns while background warm-up is still blocked") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    auto launcher = std::make_unique<HttpLauncher>();
    launcher->hold_warmup = true;
    auto *launcher_ptr = launcher.get();
    auto warmup = std::make_shared<PrefixWarmup>(options(prefix.path.string()));
    auto supervisor_options = sonder_test::base_options();
    supervisor_options.spill_guard.enabled = false;
    supervisor_options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    supervisor_options.warmup = warmup;
    Supervisor supervisor(supervisor_options, std::move(launcher));
    const auto began = std::chrono::steady_clock::now();
    REQUIRE(supervisor.start().ok());
    CHECK(std::chrono::steady_clock::now() - began < std::chrono::milliseconds(500));
    REQUIRE(eventually([&] { return launcher_ptr->server_count() == 1 &&
                                    launcher_ptr->server_at(0)->wait_for_nonstream(1); }));
    auto upstream = launcher_ptr->server_at(0);
    upstream->set_body("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"
                       "data: [DONE]\n\n");
    LlamaServerBackendOptions request_options;
    request_options.base_url = upstream->url();
    auto request_backend = make_llamaserver_backend(request_options);
    auto request_model = request_backend->load_model({"fake-model", "cpu:0"});
    REQUIRE(request_model.ok());
    ChatRequest request;
    request.messages = {{"user", "live turn"}};
    auto live_turn = std::async(std::launch::async, [&] {
        return request_model.value()->chat(request, {}, {});
    });
    CHECK(live_turn.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    // A supervisor shutdown must cancel the still-held warm-up before child
    // cleanup. The real request has its own independent completion path.
    auto stopped = std::async(std::launch::async, [&] { supervisor.stop(); });
    CHECK(stopped.wait_for(std::chrono::seconds(1)) == std::future_status::ready);
    upstream->hold_nonstream(false); // cleanup even if the assertion failed
    stopped.get();
    if (live_turn.wait_for(std::chrono::seconds(1)) != std::future_status::ready)
        CHECK(live_turn.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    CHECK(live_turn.get().ok());
    supervisor.stop();
}

TEST_CASE("unreported counters remain unknown and invalid slot ids never issue a replay") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    sonder_test::FakeLlamaServer server;
    server.set_props(R"({"total_slots":1})");
    server.set_nonstream_body(R"({"choices":[{"finish_reason":"length"}]})");
    auto o = options(prefix.path.string());
    PrefixWarmup warmup(o);
    warmup.start(server.url());
    REQUIRE(eventually([&] { return warmup.status()->status == "complete"; }));
    const auto slot = warmup.status()->slots.front();
    CHECK_FALSE(slot.prompt_tokens);
    CHECK_FALSE(slot.cache_n);
    CHECK(json::Value(to_json(slot)).find("cache_n")->is_null());
    o.warmup.all_slots = false;
    o.warmup.slots = {1};
    PrefixWarmup invalid(o);
    invalid.start(server.url());
    REQUIRE(eventually([&] { return invalid.status()->status == "error"; }));
    CHECK(server.nonstream_bodies().size() == 1);
    o.native_completion = false;
    PrefixWarmup generic(o);
    generic.start(server.url());
    REQUIRE(eventually([&] { return generic.status()->status == "error"; }));
    CHECK(server.nonstream_bodies().size() == 1);
}

TEST_CASE("attach backend starts one warm-up and ordinary probes or loads do not duplicate it") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    sonder_test::FakeLlamaServer server;
    server.set_props(R"({"total_slots":1,"default_generation_settings":{"n_ctx":8192}})");
    server.hold_nonstream();
    server.set_body("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\n"
                    "data: [DONE]\n\n");
    auto o = options(prefix.path.string());
    o.base_url = server.url();
    auto backend = make_llamaserver_backend(o);
    REQUIRE(eventually([&] { return server.nonstream_bodies().size() == 1; }));
    REQUIRE(backend->probe().ok());
    auto model = backend->load_model(ModelLoadOptions{"fake-model", "cpu:0"});
    REQUIRE(model.ok());
    REQUIRE(backend->load_model(ModelLoadOptions{"fake-model", "cpu:0"}).ok());
    ChatRequest request;
    request.messages = {{"user", "first real turn"}};
    request.session_key = "first";
    auto live = std::async(std::launch::async, [&] { return model.value()->chat(request, {}, {}); });
    const bool finished_while_warming = live.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    CHECK(finished_while_warming);
    CHECK(backend->runtime_status()->warmup->status == "warming");
    server.hold_nonstream(false);
    CHECK(live.get().ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(server.nonstream_bodies().size() == 1);
}

TEST_CASE("warm-up waits for readiness, repeats after restart, and honors on_restart") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    sonder_test::FakeLlamaServer server(0, 2);
    server.set_props(R"({"total_slots":1})");
    auto o = options(prefix.path.string());
    PrefixWarmup warmup(o);
    warmup.start(server.url(), true);
    REQUIRE(eventually([&] { return server.nonstream_bodies().size() == 1; }));
    CHECK(server.health_requests() >= 3);
    REQUIRE(eventually([&] { return warmup.status()->status == "complete"; }));
    warmup.start(server.url());
    REQUIRE(eventually([&] { return server.nonstream_bodies().size() == 2; }));
    CHECK(warmup.status()->generation == 2);

    o.warmup.on_restart = false;
    PrefixWarmup once(o);
    once.start(server.url());
    REQUIRE(eventually([&] { return server.nonstream_bodies().size() == 3; }));
    once.start(server.url());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    CHECK(server.nonstream_bodies().size() == 3);
    REQUIRE(once.status());
    CHECK(once.status()->status == "skipped");
    CHECK(once.status()->generation == 2);
}

TEST_CASE("malformed, missing, and oversized prefixes are warnings and never throw") {
    auto missing = options((std::filesystem::current_path() / ".missing-warmup.json").string());
    PrefixWarmup absent(missing);
    absent.start("http://127.0.0.1:1");
    REQUIRE(eventually([&] { return absent.status()->status == "error"; }));
    CHECK_FALSE(absent.warnings().empty());

    PrefixFile malformed("{}");
    auto bad = options(malformed.path.string());
    PrefixWarmup invalid(bad);
    invalid.start("http://127.0.0.1:1");
    REQUIRE(eventually([&] { return invalid.status()->status == "error"; }));

    PrefixFile oversized("[{}]");
    auto huge = options(oversized.path.string());
    huge.warmup.max_prefix_chars = 2;
    PrefixWarmup too_big(huge);
    too_big.start("http://127.0.0.1:1");
    REQUIRE(eventually([&] { return too_big.status()->status == "error"; }));
}

TEST_CASE("malformed upstream response records one warning without retrying a slot") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    sonder_test::FakeLlamaServer server;
    server.set_props(R"({"total_slots":1})");
    server.set_nonstream_body("{}");
    PrefixWarmup warmup(options(prefix.path.string()));
    warmup.start(server.url());
    REQUIRE(eventually([&] { return warmup.status() && warmup.status()->status == "error"; }));
    CHECK(server.nonstream_bodies().size() == 1);
    REQUIRE(warmup.status());
    REQUIRE(warmup.status()->slots.size() == 1);
    CHECK(warmup.status()->slots.front().status == "error");
    CHECK(warmup.warnings().size() == 1);
}

TEST_CASE("shutdown cancels an in-flight warm-up promptly") {
    PrefixFile prefix(R"([{"role":"system","content":"stable"}])");
    sonder_test::FakeLlamaServer server;
    server.set_props(R"({"total_slots":1})");
    server.hold_nonstream();
    auto o = options(prefix.path.string());
    PrefixWarmup warmup(o);
    warmup.start(server.url());
    REQUIRE(server.wait_for_nonstream(1));
    auto stopper = std::async(std::launch::async, [&] { warmup.stop(); });
    const auto stopped_while_held = stopper.wait_for(std::chrono::seconds(1)) == std::future_status::ready;
    CHECK(stopped_while_held);
    server.hold_nonstream(false);
    if (!stopped_while_held)
        CHECK(stopper.wait_for(std::chrono::seconds(2)) == std::future_status::ready);
    REQUIRE(warmup.status());
    CHECK(warmup.status()->status == "cancelled");
}

TEST_CASE("disabled by default leaves the upstream untouched") {
    sonder_test::FakeLlamaServer server;
    LlamaServerBackendOptions o;
    o.base_url = server.url();
    PrefixWarmup warmup(o);
    warmup.start(server.url());
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    CHECK(server.nonstream_bodies().empty());
    CHECK_FALSE(warmup.status().has_value());
    CHECK(warmup.warmed_slots().empty());
}

} // TEST_SUITE
