#include "../process.hpp"
#include "../child_metrics.hpp"
#include "../supervisor.hpp"
#include "fake_process.hpp"
#include "fake_server.hpp"
#include <atomic>
#include <condition_variable>
#include <doctest/doctest.h>
#include <mutex>
#include <thread>
#include <limits>
#include <filesystem>
#include <fstream>
#include "sonder/inference/engine.hpp"
#include "sonder/inference/backends/llamaserver.hpp"
#if defined(SONDER_HAS_SERVER)
#include "sonder/inference/backend_setup.hpp"
#endif

#if !defined(_WIN32)
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
using namespace sonder_test;

namespace {
constexpr const char *kChildMetrics =
    "# HELP ignored exposition comments\n"
    "llamacpp:prompt_tokens_seconds 1050.5\n"
    "llamacpp:predicted_tokens_seconds 86.8\n"
    "llamacpp:prompt_tokens_total 2500\n"
    "llamacpp:tokens_predicted_total 1500\n"
    "llamacpp:requests_processing 1\n"
    "llamacpp:requests_deferred 2\n"
    "llamacpp:n_busy_slots_per_decode 1.5\n"
    "llamacpp:n_decode_total 1000\n"
    "llamacpp:spec_decode_num_draft_tokens_total 2000\n"
    "llamacpp:spec_decode_accepted_tokens_total 1500\n"
    "llamacpp:spec_decode_drafts_total 1000\n"
    "llamacpp:spec_decode_accepted_tokens_per_pos_total{position=\"0\"} 900\n"
    "llamacpp:spec_decode_accepted_tokens_per_pos_total{position=\"1\"} 600\n"
    "llamacpp:kv_cache_usage_ratio 0.25\n";
constexpr const char *kChildSlots = R"([{"id":0,"is_processing":true,"n_ctx":77824}])";
using MetricsClock = std::chrono::steady_clock;

ChildMetricsPoll frozen_metrics() {
    return [](const net::HttpRequest &request, const CancellationToken &) {
        return ChildMetricsHttpResponse{200, request.target == "/metrics" ? kChildMetrics : kChildSlots};
    };
}
} // namespace

TEST_CASE("child metrics parse every documented counter and cumulative speculation") {
    const auto metrics = parse_child_metrics(kChildMetrics);
    CHECK(metrics.prompt_tokens_seconds == 1050.5);
    CHECK(metrics.predicted_tokens_seconds == 86.8);
    CHECK(metrics.prompt_tokens_total == 2500);
    CHECK(metrics.tokens_predicted_total == 1500);
    CHECK(metrics.requests_processing == 1);
    CHECK(metrics.requests_deferred == 2);
    CHECK(metrics.n_busy_slots_per_decode == 1.5);
    CHECK(metrics.n_decode_total == 1000);
    CHECK(metrics.spec_decode_num_draft_tokens_total == 2000);
    CHECK(metrics.spec_decode_accepted_tokens_total == 1500);
    CHECK(metrics.spec_decode_drafts_total == 1000);
    CHECK(metrics.kv_cache_usage_ratio == 0.25);
    ChildMetricsOptions options;
    options.n_max = 2;
    const auto derived = derive_child_speculation(metrics, options);
    REQUIRE(derived.mean_accepted_len);
    CHECK(*derived.mean_accepted_len == doctest::Approx(1.5));
    REQUIRE(derived.acceptance_by_position.size() == 2);
    CHECK(*derived.acceptance_by_position[0] == doctest::Approx(0.9));
    CHECK(*derived.acceptance_by_position[1] == doctest::Approx(0.6));
    CHECK(*derived.speedup_est == doctest::Approx(2.5 / 2.2));
}

TEST_CASE("child metrics reject invalid numbers and bound adversarial position labels") {
    for (const auto *body : {"llamacpp:prompt_tokens_seconds NaN\n",
        "llamacpp:predicted_tokens_seconds -5\n",
        "llamacpp:prompt_tokens_total 18446744073709551616\n",
        "llamacpp:tokens_predicted_total -1\n",
        "llamacpp:requests_processing 0.5\n",
        "llamacpp:requests_deferred +Inf\n",
        "llamacpp:spec_decode_accepted_tokens_per_pos_total{position=\"18446744073709551615\"} 1\n",
        "llamacpp:spec_decode_accepted_tokens_per_pos_total{position=\"1junk\"} 1\n",
        "llamacpp:spec_decode_accepted_tokens_per_pos_total{notposition=\"0\"} 1\n"}) {
        const auto metrics = parse_child_metrics(body);
        CHECK_FALSE(metrics.valid);
        CHECK_FALSE(metrics.prompt_tokens_seconds);
        CHECK_FALSE(metrics.predicted_tokens_seconds);
        CHECK_FALSE(metrics.prompt_tokens_total);
        CHECK_FALSE(metrics.tokens_predicted_total);
        CHECK_FALSE(metrics.requests_processing);
        CHECK_FALSE(metrics.requests_deferred);
        CHECK(metrics.spec_decode_accepted_tokens_per_position.empty());
    }
}

TEST_CASE("child metrics accept exposition whitespace timestamps and exact integer counters") {
    const auto metrics = parse_child_metrics(
        " \t \r\n"
        "  llamacpp:prompt_tokens_total 18446744073709551615 \r\n"
        "llamacpp:tokens_predicted_total 2.5e3 123456\n"
        "llamacpp:requests_processing 1\t\n"
        "llamacpp:spec_decode_accepted_tokens_per_pos_total{position=\"0\"} 3 \n");
    CHECK(metrics.valid);
    CHECK(metrics.prompt_tokens_total == std::numeric_limits<std::uint64_t>::max());
    CHECK(metrics.tokens_predicted_total == 2500);
    CHECK(metrics.requests_processing == 1);
    REQUIRE(metrics.spec_decode_accepted_tokens_per_position.size() == 1);
    CHECK(metrics.spec_decode_accepted_tokens_per_position[0] == 3);
}

TEST_CASE("child speculation absent evidence remains null") {
    ChildMetricsOptions options;
    options.n_max = 2;
    auto metrics = parse_child_metrics("llamacpp:spec_decode_drafts_total 0\n");
    auto derived = derive_child_speculation(metrics, options);
    CHECK_FALSE(derived.mean_accepted_len);
    CHECK_FALSE(derived.speedup_est);
    metrics = parse_child_metrics(kChildMetrics);
    options.n_max.reset();
    derived = derive_child_speculation(metrics, options);
    CHECK(derived.mean_accepted_len);
    CHECK_FALSE(derived.speedup_est);
}

TEST_CASE("child slots reject bogus unsigned and bool values without echoing prompts") {
    const auto slots = parse_child_slots(
        R"([{"id":0,"is_processing":true,"n_ctx":77824,"prompt":"private"},)"
        R"({"id":-1,"is_processing":"yes","n_ctx":0.5}])");
    REQUIRE(slots.size() == 2);
    CHECK(slots[0].id == 0);
    CHECK(slots[0].is_processing == true);
    CHECK(slots[0].n_ctx == 77824);
    CHECK_FALSE(slots[1].id);
    CHECK_FALSE(slots[1].is_processing);
    CHECK_FALSE(slots[1].n_ctx);
    ChildSnapshot snapshot;
    snapshot.slots = slots;
    CHECK(json::Value(snapshot.to_json()).dump().find("private") == std::string::npos);
}

TEST_CASE("child metrics frozen processing warns once and retains live stall health") {
    ChildMetricsOptions options;
    options.n_max = 2;
    ChildMetricsSampler sampler(options, frozen_metrics());
    const auto now = MetricsClock::now();
    CHECK(sampler.sample_at(1234, now).warnings.empty());
    CHECK(sampler.sample_at(1234, now + std::chrono::seconds(89)).warnings.empty());
    auto sample = sampler.sample_at(1234, now + std::chrono::seconds(90));
    REQUIRE(sample.warnings.size() == 1);
    CHECK(sample.warnings.front().code == "backend_stalled");
    REQUIRE(sample.stall);
    CHECK(sample.stall->find("since_ms")->as_int() == 90000);
    CHECK_FALSE(sample.restart_requested);
    const auto detected = sample.stall->find("detected_at")->as_int();
    sample = sampler.sample_at(1234, now + std::chrono::seconds(95));
    CHECK(sample.warnings.empty());
    REQUIRE(sample.stall);
    CHECK(sample.stall->find("detected_at")->as_int() == detected);
    CHECK(sample.stall->find("since_ms")->as_int() == 95000);
    REQUIRE(sample.snapshot);
    const auto child = sample.snapshot->to_json();
    CHECK(child.find("port")->as_uint() == 1234);
    CHECK(child.find("metrics")->find("requests_processing")->as_uint() == 1);
    CHECK(child.find("slots")->as_array().front().find("n_ctx")->as_uint() == 77824);
    CHECK(child.find("speculation")->find("mean_accepted_len")->as_double() == doctest::Approx(1.5));
    CHECK(child.find("sampled_at")->is_number());
}

TEST_CASE("child moving either counter and idle transitions clear stall evidence") {
    std::string body = kChildMetrics;
    ChildMetricsSampler sampler({}, [&](const net::HttpRequest &request, const CancellationToken &) {
        return ChildMetricsHttpResponse{200, request.target == "/metrics" ? body : kChildSlots};
    });
    const auto now = MetricsClock::now();
    (void)sampler.sample_at(1234, now);
    for (unsigned i = 1; i <= 4; ++i) {
        body = "llamacpp:requests_processing 1\nllamacpp:prompt_tokens_total " + std::to_string(2500 + i) +
               "\nllamacpp:tokens_predicted_total 1500\n";
        const auto sample = sampler.sample_at(1234, now + std::chrono::seconds(80 * i));
        CHECK(sample.warnings.empty());
        CHECK_FALSE(sample.stall);
    }
    body = "llamacpp:requests_processing 1\nllamacpp:prompt_tokens_total 2504\nllamacpp:tokens_predicted_total 1501\n";
    CHECK(sampler.sample_at(1234, now + std::chrono::seconds(400)).warnings.empty());
    REQUIRE(sampler.sample_at(1234, now + std::chrono::seconds(490)).stall);
    body = "llamacpp:requests_processing 0\nllamacpp:prompt_tokens_total 2504\nllamacpp:tokens_predicted_total 1501\n";
    CHECK_FALSE(sampler.sample_at(1234, now + std::chrono::seconds(500)).stall);
}

TEST_CASE("child scrape failures and counter reset break continuous stall evidence") {
    int status = 200;
    std::string body = kChildMetrics;
    ChildMetricsSampler sampler({}, [&](const net::HttpRequest &request, const CancellationToken &) {
        return ChildMetricsHttpResponse{status, request.target == "/metrics" ? body : kChildSlots};
    });
    const auto now = MetricsClock::now();
    (void)sampler.sample_at(1234, now);
    status = 503;
    auto failed = sampler.sample_at(1234, now + std::chrono::seconds(80));
    REQUIRE(failed.warnings.size() == 1);
    CHECK(failed.warnings.front().code == "metrics_scrape_failed");
    CHECK_FALSE(failed.stall);
    status = 200;
    CHECK_FALSE(sampler.sample_at(1234, now + std::chrono::seconds(100)).stall);
    CHECK_FALSE(sampler.sample_at(1234, now + std::chrono::seconds(189)).stall);
    body = "llamacpp:requests_processing 1\nllamacpp:prompt_tokens_total 1\nllamacpp:tokens_predicted_total 1\n";
    CHECK_FALSE(sampler.sample_at(1234, now + std::chrono::seconds(190)).stall);
    CHECK_FALSE(sampler.sample_at(1234, now + std::chrono::seconds(279)).stall);
}

TEST_CASE("child needs both counters for stall detection and disabled guard still scrapes") {
    for (const auto *body : {"llamacpp:requests_processing 1\nllamacpp:prompt_tokens_total 1\n",
                             "llamacpp:requests_processing 1\nllamacpp:tokens_predicted_total 1\n"}) {
        ChildMetricsSampler sampler({}, [body](const net::HttpRequest &request, const CancellationToken &) {
            return ChildMetricsHttpResponse{200, request.target == "/metrics" ? body : kChildSlots};
        });
        const auto now = MetricsClock::now();
        (void)sampler.sample_at(1234, now);
        const auto later = sampler.sample_at(1234, now + std::chrono::seconds(100));
        CHECK_FALSE(later.stall);
        CHECK_FALSE(later.restart_requested);
    }
    ChildMetricsOptions disabled;
    disabled.enabled = false;
    ChildMetricsSampler sampler(disabled, frozen_metrics());
    const auto now = MetricsClock::now();
    REQUIRE(sampler.sample_at(1234, now).snapshot);
    const auto later = sampler.sample_at(1234, now + std::chrono::seconds(100));
    CHECK(later.snapshot);
    CHECK(later.warnings.empty());
    CHECK_FALSE(later.stall);
}

TEST_CASE("child metrics 404 suppresses all later polling until a new child") {
    for (const bool slots_missing : {false, true}) {
        unsigned polls = 0;
        ChildMetricsSampler sampler({}, [&](const net::HttpRequest &request, const CancellationToken &) {
            ++polls;
            return ChildMetricsHttpResponse{slots_missing && request.target == "/metrics" ? 200 : 404, kChildMetrics};
        });
        const auto first = sampler.sample(1234);
        REQUIRE(first.warnings.size() == 1);
        CHECK(first.warnings.front().code == "metrics_unavailable");
        const auto count = polls;
        CHECK(sampler.sample(1234).warnings.empty());
        CHECK(sampler.sample(1234).warnings.empty());
        CHECK(polls == count);
        sampler.reset(1234);
        CHECK(sampler.sample(1234).warnings.size() == 1);
        CHECK(polls > count);
    }
}

TEST_CASE("child restart policy requests one restart per continuous stall episode") {
    ChildMetricsOptions options;
    options.policy = "restart";
    ChildMetricsSampler sampler(options, frozen_metrics());
    const auto now = MetricsClock::now();
    (void)sampler.sample_at(1234, now);
    CHECK(sampler.sample_at(1234, now + std::chrono::seconds(90)).restart_requested);
    CHECK_FALSE(sampler.sample_at(1234, now + std::chrono::seconds(180)).restart_requested);
    sampler.reset(1234);
    CHECK_FALSE(sampler.sample_at(1234, now + std::chrono::seconds(190)).restart_requested);
}

TEST_CASE("child metrics scrape real fake endpoints and preserve cached poll counts on 404") {
    FakeLlamaServer server;
    server.set_metrics(kChildMetrics);
    server.set_slots(kChildSlots);
    auto url = net::parse_url(server.url());
    REQUIRE(url.ok());
    ChildMetricsSampler sampler;
    const auto sample = sampler.sample(url->port);
    REQUIRE(sample.snapshot);
    CHECK(sample.snapshot->metrics.prompt_tokens_total == 2500);
    CHECK(server.metrics_requests() == 1);
    CHECK(server.slots_requests() == 1);
    server.set_metrics("", 404);
    REQUIRE(sampler.sample(url->port).warnings.size() == 1);
    (void)sampler.sample(url->port);
    CHECK(server.metrics_requests() == 2);
    CHECK(server.slots_requests() == 1);
}

TEST_CASE("child attach preserves base path TLS policy and bounded observation budgets") {
    net::HttpRequest endpoint;
    endpoint.host = "private.invalid";
    endpoint.port = 8443;
    endpoint.target = "/inference/";
    endpoint.use_tls = true;
    endpoint.tls.pinned_sha256 = "test-only-pin";
    endpoint.tls.ca_bundle_path = "test-only-ca";
    endpoint.tls.server_name = "private.invalid";
    unsigned calls = 0;
    ChildMetricsSampler sampler({}, [&](const net::HttpRequest &request, const CancellationToken &) {
        ++calls;
        CHECK(request.host == endpoint.host);
        CHECK(request.port == 8443);
        CHECK(request.use_tls);
        CHECK(request.tls.pinned_sha256 == endpoint.tls.pinned_sha256);
        CHECK(request.tls.ca_bundle_path == endpoint.tls.ca_bundle_path);
        CHECK(request.tls.server_name == endpoint.tls.server_name);
        CHECK_FALSE(request.tls.insecure_skip_verify);
        CHECK(request.connect_timeout <= std::chrono::milliseconds(250));
        CHECK(request.total_timeout <= std::chrono::milliseconds(250));
        CHECK(request.tls.handshake_timeout <= std::chrono::milliseconds(250));
        CHECK((request.target == "/inference/metrics" || request.target == "/inference/slots"));
        return ChildMetricsHttpResponse{200, request.target.ends_with("/metrics") ? kChildMetrics : kChildSlots};
    });
    CHECK(sampler.sample_at(endpoint, MetricsClock::now()).snapshot);
    CHECK(calls == 2);
    CancellationSource cancelled;
    cancelled.cancel();
    CHECK_FALSE(sampler.sample_at(endpoint, MetricsClock::now(), cancelled.token()).polled);
    CHECK(calls == 2);
}

TEST_CASE("child malformed payload and thrown transport produce warnings without stall evidence") {
    bool throwing = true;
    ChildMetricsSampler sampler({}, [&](const net::HttpRequest &request, const CancellationToken &) -> ChildMetricsHttpResponse {
        if (throwing) throw std::runtime_error("private upstream body must never escape");
        return {200, request.target == "/metrics" ? kChildMetrics : "not json"};
    });
    const auto first = sampler.sample(1234);
    CHECK(first.polled);
    REQUIRE(first.warnings.size() == 1);
    CHECK(first.warnings.front().code == "metrics_scrape_failed");
    CHECK(first.warnings.front().message.find("private") == std::string::npos);
    throwing = false;
    const auto second = sampler.sample(1234);
    CHECK(second.polled);
    CHECK_FALSE(second.snapshot);
    CHECK_FALSE(second.stall);
    CHECK(second.warnings.empty()); // one failure episode
}

TEST_CASE("supervisor health includes the ready child port before its first metrics poll") {
    // Regression: health said ready throughout the measured five-minute hang
    // but did not identify the upstream port or expose child observations.
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    options.spill_guard.enabled = false;
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    const auto ready = supervisor.start();
    REQUIRE(ready.ok());
    const auto runtime = to_json(supervisor.runtime_status());
    const auto *child = runtime.find("child");
    REQUIRE(child != nullptr);
    REQUIRE(child->find("port") != nullptr);
    CHECK(child->find("port")->as_uint() == ready.value());
    supervisor.stop();
}

TEST_CASE("supervisor stall restart uses crash budget and emits one restart with its reason") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    options.spill_guard.enabled = false;
    options.stall_guard.policy = "restart";
    options.max_restarts = 1;
    std::atomic<std::int64_t> elapsed{0};
    options.metrics_clock = [&] { return MetricsClock::time_point(std::chrono::milliseconds(elapsed.load())); };
    std::atomic<unsigned> polls{0};
    options.metrics_poll = [&](const net::HttpRequest &request, const CancellationToken &) {
        if (request.target == "/metrics")
            ++polls;
        return ChildMetricsHttpResponse{200, request.target == "/metrics" ? kChildMetrics : kChildSlots};
    };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    REQUIRE(supervisor.start().ok());
    const auto wait_polls = [&](unsigned count) {
        const auto end = MetricsClock::now() + std::chrono::seconds(2);
        while (polls.load() < count && MetricsClock::now() < end)
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        return polls.load() >= count;
    };
    REQUIRE(wait_polls(1));
    // Wait for snapshot publication before advancing the injected clock.
    const auto published = [&] {
        const auto state_json = supervisor.child_runtime_status();
        return state_json.find("child") && !state_json.find("child")->find("sampled_at")->is_null();
    };
    for (int i = 0; i < 1000 && !published(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(published());
    elapsed.store(95000);
    REQUIRE(wait_for_starts(state, 2));
    REQUIRE(supervisor.start().ok());
    REQUIRE(wait_polls(3)); // the replacement gets its own baseline
    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    elapsed.store(190000);
    REQUIRE(wait_polls(4));
    for (int i = 0; i < 1000 && child_at(state, 1)->stops.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK_FALSE(supervisor.start().ok());
    supervisor.stop();
    CHECK(state->starts.size() == 2);
    const auto events = supervisor.runtime_events()->drain();
    unsigned restarts = 0;
    unsigned stalls = 0;
    for (const auto &[type, attrs] : events) {
        if (type == "backend.restart") {
            ++restarts;
            CHECK(attrs.find("reason")->as_string() == "backend_stalled");
            CHECK(attrs.find("attempt")->as_uint() == 1);
        }
        if (type == "backend.warning" && attrs.find("kind") &&
            attrs.find("kind")->as_string() == "backend_stalled") {
            ++stalls;
            CHECK(attrs.find("processing")->as_uint() == 1);
            CHECK(attrs.find("deferred")->as_uint() == 2);
        }
    }
    CHECK(restarts == 1);
    CHECK(stalls == 2);
}

TEST_CASE("supervisor generic mode never polls native endpoints") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.native_completion = false;
    options.spill_guard.enabled = false;
    options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    std::atomic<unsigned> polls{0};
    options.metrics_poll = [&](const net::HttpRequest &, const CancellationToken &) {
        ++polls;
        return ChildMetricsHttpResponse{404, {}};
    };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    REQUIRE(supervisor.start().ok());
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
    supervisor.stop();
    CHECK(polls.load() == 0);
    CHECK_FALSE(supervisor.child_runtime_status().contains("child"));
}

TEST_CASE("supervisor attach samples without a process and never restarts external ownership") {
    FakeLlamaServer server;
    server.set_metrics(kChildMetrics);
    server.set_slots(kChildSlots);
    const auto endpoint = net::parse_url(server.url());
    REQUIRE(endpoint.ok());
    SupervisorOptions options;
    net::HttpRequest request;
    request.host = endpoint->host;
    request.port = endpoint->port;
    options.attached_endpoint = request;
    options.spill_guard.enabled = false;
    options.stall_guard.policy = "restart";
    std::atomic<std::int64_t> elapsed{0};
    options.metrics_clock = [&] { return MetricsClock::time_point(std::chrono::milliseconds(elapsed.load())); };
    Supervisor supervisor(options);
    REQUIRE(supervisor.start().ok());
    const auto sampled = [&] {
        const auto state_json = supervisor.child_runtime_status();
        return state_json.find("child") && !state_json.find("child")->find("sampled_at")->is_null();
    };
    for (int i = 0; i < 1000 && !sampled(); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    REQUIRE(sampled());
    elapsed.store(95000);
    for (int i = 0; i < 1000 && !supervisor.child_runtime_status().contains("stall"); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK(supervisor.child_runtime_status().contains("stall"));
    CHECK(supervisor.running());
    supervisor.stop();
    for (const auto &[type, attrs] : supervisor.runtime_events()->drain()) {
        (void)attrs;
        CHECK(type != "backend.restart");
    }
    // Its listener still works after the observer stops.
    ChildMetricsSampler sampler;
    CHECK(sampler.sample(endpoint->port).snapshot);
}

TEST_CASE("native attach backend exposes child health while retaining its existing health probe") {
    FakeLlamaServer server;
    server.set_metrics(kChildMetrics);
    server.set_slots(kChildSlots);
    LlamaServerBackendOptions options;
    options.base_url = server.url();
    options.args = {"--spec-draft-n-max", "2"};
    const auto backend = make_llamaserver_backend(options);
    REQUIRE(backend->probe().ok());
    json::Object runtime;
    const auto end = MetricsClock::now() + std::chrono::seconds(2);
    do {
        const auto status = backend->runtime_status();
        if (status) runtime = to_json(*status);
        const auto *child = runtime.find("child");
        if (child && !child->find("sampled_at")->is_null()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while (MetricsClock::now() < end);
    const auto *child = runtime.find("child");
    REQUIRE(child);
    CHECK(child->find("port")->as_uint() == net::parse_url(server.url())->port);
    CHECK(child->find("metrics")->find("prompt_tokens_total")->as_uint() == 2500);
    CHECK(child->find("speculation")->find("speedup_est")->as_double() == doctest::Approx(2.5 / 2.2));
    REQUIRE(runtime.find("diagnostics"));
    CHECK(runtime.find("diagnostics")->find("offload")->as_string() == "blind");
    CHECK(server.models_requests() == 1); // existing probe uses /v1/models
    CHECK(server.health_requests() == 0);
}

TEST_CASE("supervisor observability queue is bounded and health does not drain events") {
    Supervisor::RuntimeEvents events;
    for (unsigned i = 0; i < 100; ++i)
        events.push("backend.metrics.sample", {{"sample", i}});
    const auto drained = events.drain();
    REQUIRE(drained.size() == 65);
    CHECK(drained.back().first == "backend.metrics.dropped");
    CHECK(drained.back().second.find("dropped_events")->as_uint() == 36);
    CHECK(events.drain().empty());
}

TEST_CASE("engine emits every child poll and health snapshots never consume queued observations") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.spill_guard.enabled = false;
    options.spill_guard.sample_interval = std::chrono::milliseconds(100);
    options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    std::atomic<std::int64_t> elapsed{0};
    options.metrics_clock = [&] { return MetricsClock::time_point(std::chrono::milliseconds(elapsed.load())); };
    options.metrics_poll = frozen_metrics();
    auto supervisor = std::make_shared<Supervisor>(options, std::make_unique<FakeLauncher>(state));
    REQUIRE(supervisor->start().ok());
    const auto wait_sample = [&](std::int64_t timestamp) {
        const auto end = MetricsClock::now() + std::chrono::seconds(2);
        while (MetricsClock::now() < end) {
            const auto runtime = supervisor->child_runtime_status();
            if (const auto *child = runtime.find("child")) {
                const auto *sampled = child->find("sampled_at");
                if (sampled && !sampled->is_null() && sampled->as_int() == timestamp) return true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        return false;
    };
    REQUIRE(wait_sample(0));
    elapsed.store(4999);
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    CHECK(wait_sample(0)); // the shorter GPU cadence must not increase HTTP polling
    elapsed.store(5000);
    REQUIRE(wait_sample(5000));
    elapsed.store(10000);
    REQUIRE(wait_sample(10000));
    for (unsigned i = 0; i < 5; ++i) (void)to_json(supervisor->runtime_status());
    class ObservedBackend final : public Backend {
      public:
        explicit ObservedBackend(std::shared_ptr<Supervisor> s) : supervisor_(std::move(s)) {}
        std::string name() const override { return "fake-child-observer"; }
        std::string description() const override { return "test-only child observations"; }
        BackendCapabilities capabilities() const override { return {}; }
        Result<std::string> probe() override { return std::string("fake"); }
        Result<std::vector<ModelDescriptor>> list_models() override { return std::vector<ModelDescriptor>{}; }
        Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions &) override {
            return Status(ErrorCode::unsupported, "fake observations only");
        }
        std::optional<BackendRuntimeStatus> runtime_status() const override { return supervisor_->runtime_status(); }
      private:
        std::shared_ptr<Supervisor> supervisor_;
    };
    auto sink = std::make_shared<MemoryTelemetrySink>();
    EngineOptions engine_options;
    engine_options.telemetry_sinks.push_back(sink);
    engine_options.device_sample_interval = std::chrono::milliseconds(10);
    engine_options.sample_devices_on_start = false;
    engine_options.scheduling.enabled = false;
    Engine engine(engine_options);
    REQUIRE(engine.register_backend(std::make_shared<ObservedBackend>(supervisor)).ok());
    std::vector<json::Value> samples;
    const auto end = MetricsClock::now() + std::chrono::seconds(2);
    do {
        samples.clear();
        for (const auto &line : sink->lines()) {
            const auto event = json::parse(line);
            if (event.ok() && event->find("event_type")->as_string() == "backend.metrics.sample")
                samples.push_back(event.value());
        }
        if (samples.size() >= 3) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    } while (MetricsClock::now() < end);
    REQUIRE(samples.size() == 3);
    for (std::size_t i = 0; i < samples.size(); ++i) {
        CHECK(samples[i].find("schema")->as_string() == "sonder.observatory.event/1");
        const auto *attrs = samples[i].find("attributes");
        REQUIRE(attrs);
        CHECK(attrs->find("backend")->as_string() == "fake-child-observer");
        CHECK(attrs->find("sampled_at")->as_uint() == i * 5000);
        CHECK(attrs->find("metrics")->find("requests_processing")->as_uint() == 1);
    }
    supervisor->stop();
}

#if defined(SONDER_HAS_SERVER)
TEST_CASE("stall guard config is strict transactional and preserves defaults") {
    BackendSetup setup;
    CHECK(setup.llamaserver_stall_guard);
    CHECK(setup.llamaserver_stall_seconds == 90);
    CHECK(setup.llamaserver_stall_policy == "warn");
    struct TempConfig {
        std::filesystem::path path = std::filesystem::current_path() /
            (".l3-stall-config-" + std::to_string(MetricsClock::now().time_since_epoch().count()) + ".json");
        ~TempConfig() { std::error_code ec; std::filesystem::remove(path, ec); }
        void write(const std::string &text) const { std::ofstream out(path, std::ios::binary); out << text; }
    } config;
    config.write(R"({"stall_guard":{"enabled":false,"stall_seconds":120,"policy":"restart"}})");
    REQUIRE(load_llamaserver_config(config.path.string(), setup).ok());
    CHECK_FALSE(setup.llamaserver_stall_guard);
    CHECK(setup.llamaserver_stall_seconds == 120);
    CHECK(setup.llamaserver_stall_policy == "restart");
    for (const auto *bad : {"[]", "null", "true", R"({"enabled":1})", R"({"stall_seconds":0})",
                            R"({"stall_seconds":86401})", R"({"stall_seconds":1.5})",
                            R"({"stall_seconds":"90"})", R"({"policy":"refuse"})",
                            R"({"policy":false})", R"({"unknown":1})"}) {
        config.write(std::string(R"({"mode":"spawn","stall_guard":)") + bad + "}");
        const auto old = setup;
        CHECK_FALSE(load_llamaserver_config(config.path.string(), setup).ok());
        CHECK(setup.llamaserver_mode == old.llamaserver_mode);
        CHECK(setup.llamaserver_stall_seconds == old.llamaserver_stall_seconds);
        CHECK(setup.llamaserver_stall_policy == old.llamaserver_stall_policy);
        CHECK(setup.llamaserver_stall_guard == old.llamaserver_stall_guard);
    }
}
#endif

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

TEST_CASE("supervisor readiness timeout waits for the monitor to stop a slow-probed child") {
    // Deterministic form of a CI race: the health probe outlives start()'s own
    // deadline, so start() must wait for the monitor to stop the child and
    // publish its failure instead of returning while the child still runs.
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.readiness_timeout = std::chrono::milliseconds(15);
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return false;
    };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    CHECK_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::timeout);
    REQUIRE(state->children.size() == 1);
    CHECK(state->children.front()->stops.load() >= 1);
    CHECK_FALSE(state->children.front()->alive.load());
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

#if !defined(_WIN32)
namespace {
std::string temporary_pid_file() {
    char path[] = "/tmp/sonder-llamaserver-pid-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return {};
    close(fd);
    unlink(path);
    return path;
}

bool read_pid(const std::string &path, pid_t &pid) {
    std::ifstream input(path);
    long value = 0;
    if (!(input >> value) || value <= 0)
        return false;
    pid = static_cast<pid_t>(value);
    return true;
}

bool wait_for_pid_file(const std::string &path, pid_t &pid, std::chrono::milliseconds timeout) {
    const auto until = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < until) {
        if (read_pid(path, pid))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return read_pid(path, pid);
}

struct PidFileCleanup {
    std::vector<std::string> paths;
    ~PidFileCleanup() {
        for (const auto &path : paths)
            unlink(path.c_str());
    }
};

bool process_gone(pid_t pid) {
    if (kill(pid, 0) != 0)
        return errno == ESRCH;

    // A killed, reparented child can briefly remain as a zombie while init
    // (or launchd) reaps it. It no longer owns model/GPU resources, so accept
    // that state as gone while still rejecting a live process.
    char command[64]{};
    std::snprintf(command, sizeof(command), "ps -o state= -p %ld", static_cast<long>(pid));
    FILE *pipe = popen(command, "r");
    if (!pipe)
        return false;
    char state[16]{};
    const bool zombie =
        std::fgets(state, sizeof(state), pipe) != nullptr && (state[0] == 'Z' || state[1] == 'Z');
    (void)pclose(pipe);
    return zombie;
}

bool wait_for_exit(pid_t pid, int &status, std::chrono::milliseconds timeout) {
    const auto until = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid)
            return true;
        if (result < 0 && errno != EINTR && errno != ECHILD)
            return false;
        if (std::chrono::steady_clock::now() >= until)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void helper_spawn_long_running_children(const std::string &first_file, const std::string &second_file,
                                        const std::string &ready_file) {
    std::unique_ptr<Process> first;
    std::unique_ptr<Process> second;
    std::thread launcher([&] {
        ProcessSpec first_spec;
        first_spec.executable = "/bin/sh";
        first_spec.arguments = {"-c", "trap '' TERM; echo $$ > '" + first_file + "'; while :; do :; done"};
        first_spec.port = 1;
        first_spec.shutdown_timeout = std::chrono::seconds(30);
        auto first_result = make_process_launcher()->start(first_spec);
        if (!first_result.ok())
            return;
        first = std::move(first_result.value());

        ProcessSpec second_spec = first_spec;
        second_spec.arguments = {"-c", "trap '' TERM; echo $$ > '" + second_file + "'; while :; do :; done"};
        auto second_result = make_process_launcher()->start(second_spec);
        if (second_result.ok())
            second = std::move(second_result.value());
    });
    launcher.join();
    if (!first || !second)
        _exit(111);
    // Publish readiness only after the launching thread has exited. A
    // thread-bound PDEATHSIG implementation must not pass this regression.
    {
        std::ofstream ready(ready_file);
        ready << getpid() << '\n';
    }
    for (;;) {
        pause();
    }
}
} // namespace

TEST_CASE("POSIX child is gone when launcher parent is SIGKILLed") {
    PidFileCleanup files;
    const auto first_file = temporary_pid_file();
    const auto second_file = temporary_pid_file();
    const auto ready_file = temporary_pid_file();
    REQUIRE_FALSE(first_file.empty());
    REQUIRE_FALSE(second_file.empty());
    REQUIRE_FALSE(ready_file.empty());
    files.paths = {first_file, second_file, ready_file};

    const pid_t helper = fork();
    REQUIRE(helper >= 0);
    if (helper == 0)
        helper_spawn_long_running_children(first_file, second_file, ready_file);

    pid_t first_child = -1;
    pid_t second_child = -1;
    pid_t ready_parent = -1;
    const bool started = wait_for_pid_file(ready_file, ready_parent, std::chrono::seconds(3)) &&
                         wait_for_pid_file(first_file, first_child, std::chrono::seconds(2)) &&
                         wait_for_pid_file(second_file, second_child, std::chrono::seconds(2));
    CHECK(started);
    if (started) {
        CHECK(ready_parent == helper);
        CHECK_FALSE(process_gone(first_child));
        CHECK_FALSE(process_gone(second_child));
        CHECK(kill(helper, SIGKILL) == 0);
        int status = 0;
        CHECK(wait_for_exit(helper, status, std::chrono::seconds(2)));
        CHECK(WIFSIGNALED(status));

        bool first_gone = false;
        bool second_gone = false;
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < until) {
            first_gone = first_gone || process_gone(first_child);
            second_gone = second_gone || process_gone(second_child);
            if (first_gone && second_gone)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(first_gone);
        CHECK(second_gone);
        if (!first_gone) {
            const pid_t group = getpgid(first_child);
            if (group > 0 && group != getpgrp())
                kill(-group, SIGKILL);
            else
                kill(first_child, SIGKILL);
        }
        if (!second_gone) {
            const pid_t group = getpgid(second_child);
            if (group > 0 && group != getpgrp())
                kill(-group, SIGKILL);
            else
                kill(second_child, SIGKILL);
        }
    } else {
        kill(helper, SIGKILL);
        int status = 0;
        (void)wait_for_exit(helper, status, std::chrono::seconds(2));
        // Also clean up partial startup when running against a broken
        // launcher; do not leave a busy fake server behind after a failure.
        if (read_pid(first_file, first_child) && !process_gone(first_child))
            kill(first_child, SIGKILL);
        if (read_pid(second_file, second_child) && !process_gone(second_child))
            kill(second_child, SIGKILL);
    }
}

TEST_CASE("POSIX SIGTERM shutdown returns when child exits") {
    PidFileCleanup files;
    const auto pid_file = temporary_pid_file();
    REQUIRE_FALSE(pid_file.empty());
    files.paths.push_back(pid_file);
    ProcessSpec spec;
    spec.executable = "/bin/sh";
    spec.arguments = {"-c", "trap 'exit 0' TERM; echo $$ > '" + pid_file + "'; while :; do :; done"};
    spec.port = 1;
    spec.shutdown_timeout = std::chrono::seconds(2);
    auto result = make_process_launcher()->start(spec);
    REQUIRE_MESSAGE(result.ok(), result.status().to_string());
    auto process = std::move(result.value());
    pid_t child = -1;
    CHECK(wait_for_pid_file(pid_file, child, std::chrono::seconds(1)));

    const auto started = std::chrono::steady_clock::now();
    process->stop(std::chrono::seconds(2));
    const auto elapsed = std::chrono::steady_clock::now() - started;
    CHECK(elapsed < std::chrono::milliseconds(500));
    CHECK_FALSE(process->running());
    if (child > 0)
        CHECK(process_gone(child));
}
#endif
