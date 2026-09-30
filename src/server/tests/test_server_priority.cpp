#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "server_test_support.hpp"
#include "src/openai.hpp"

using namespace server_test;
namespace si = sonder::inference;
namespace srv = sonder::inference::server;

namespace {

class SlowControl {
public:
    void started(std::string prompt) {
        std::lock_guard<std::mutex> lock(mu_);
        starts_.push_back(std::move(prompt));
        ++active_;
        cv_.notify_all();
    }

    bool wait_for_starts(std::size_t count, std::chrono::milliseconds timeout = std::chrono::milliseconds(3000)) {
        std::unique_lock<std::mutex> lock(mu_);
        return cv_.wait_for(lock, timeout, [&] { return starts_.size() >= count; });
    }

    void finish_one(const std::string& prompt) {
        std::lock_guard<std::mutex> lock(mu_);
        finished_.insert(prompt);
        cv_.notify_all();
    }

    void release_all() {
        std::lock_guard<std::mutex> lock(mu_);
        finish_all_ = true;
        cv_.notify_all();
    }

    bool wait_to_finish(const si::CancellationToken& cancel, const std::string& prompt) {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait_for(lock, std::chrono::milliseconds(20), [&] {
            return finish_all_ || finished_.count(prompt) != 0 || cancel.cancelled();
        });
        if (cancel.cancelled()) return false;
        if (finish_all_ || finished_.erase(prompt) != 0) {
            if (active_ != 0) --active_;
            return true;
        }
        return false;
    }

    std::vector<std::string> starts() const {
        std::lock_guard<std::mutex> lock(mu_);
        return starts_;
    }
    int cancelled() const {
        std::lock_guard<std::mutex> lock(mu_);
        return cancelled_;
    }
    void record_cancelled() {
        std::lock_guard<std::mutex> lock(mu_);
        ++cancelled_;
        cv_.notify_all();
    }

private:
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::vector<std::string> starts_;
    std::size_t active_ = 0;
    std::set<std::string> finished_;
    bool finish_all_ = false;
    int cancelled_ = 0;
};

class SlowModel final : public si::BackendModel {
public:
    SlowModel(si::ModelDescriptor descriptor, std::shared_ptr<SlowControl> control)
        : descriptor_(std::move(descriptor)), control_(std::move(control)) {}

    const si::ModelDescriptor& descriptor() const override { return descriptor_; }

    si::Result<si::GenerateStats> generate(const si::GenerateRequest& request, const si::CancellationToken& cancel,
                                           const si::TokenCallback& on_chunk) override {
        control_->started(request.prompt);
        si::GenerateStats stats;
        stats.prompt_tokens = 1;
        stats.token_counts_from_backend = true;
        if (on_chunk && !on_chunk(si::TokenChunk{"x", 0})) {
            control_->record_cancelled();
            return si::Status(si::ErrorCode::cancelled, "client stopped the response");
        }
        while (!cancel.cancelled()) {
            if (control_->wait_to_finish(cancel, request.prompt)) {
                stats.completion_tokens = 1;
                stats.chunks = 1;
                stats.stop_reason = si::StopReason::end_of_sequence;
                return stats;
            }
        }
        control_->record_cancelled();
        return si::Status(si::ErrorCode::cancelled, "cancelled by caller");
    }

private:
    si::ModelDescriptor descriptor_;
    std::shared_ptr<SlowControl> control_;
};

class SlowBackend final : public si::Backend {
public:
    SlowBackend(std::shared_ptr<SlowControl> control, std::size_t capacity, bool remote = false)
        : control_(std::move(control)), capacity_(capacity), remote_(remote) {}

    std::string name() const override { return "slow-test"; }
    std::string description() const override { return "deterministic slow backend for server priority tests"; }
    si::BackendCapabilities capabilities() const override {
        si::BackendCapabilities c;
        c.add(si::Capability::streaming).add(si::Capability::deterministic);
        if (remote_) c.add(si::Capability::remote_process);
        return c;
    }
    si::Result<std::string> probe() override { return std::string("slow-test-1"); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override {
        return std::vector<si::ModelDescriptor>{descriptor()};
    }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
        return std::shared_ptr<si::BackendModel>(std::make_shared<SlowModel>(descriptor(), control_));
    }
    std::size_t max_concurrent_requests() const override { return capacity_; }

private:
    static si::ModelDescriptor descriptor() {
        si::ModelDescriptor d;
        d.name = "slow:tiny";
        d.backend = "slow-test";
        d.format = "test";
        d.context_length = 4096;
        return d;
    }
    std::shared_ptr<SlowControl> control_;
    std::size_t capacity_;
    bool remote_;
};

struct RunningRequest {
    std::thread thread;
    std::shared_ptr<Reply> reply = std::make_shared<Reply>();
    std::shared_ptr<SlowControl> control;
    std::string prompt;

    RunningRequest() = default;
    RunningRequest(const RunningRequest&) = delete;
    RunningRequest& operator=(const RunningRequest&) = delete;
    RunningRequest(RunningRequest&& other) noexcept
        : thread(std::move(other.thread)), reply(std::move(other.reply)), control(std::move(other.control)), prompt(std::move(other.prompt)) {}
    RunningRequest& operator=(RunningRequest&& other) noexcept {
        if (this != &other) {
            if (thread.joinable()) {
                if (control) {
                    control->release_all();
                }
                thread.join();
            }
            thread = std::move(other.thread);
            reply = std::move(other.reply);
            control = std::move(other.control);
            prompt = std::move(other.prompt);
        }
        return *this;
    }

    ~RunningRequest() {
        if (thread.joinable() && control) {
            // Make teardown safe even when a preceding assertion aborts this
            // test before all request threads were explicitly released.
            control->release_all();
        }
        if (thread.joinable()) thread.join();
    }
};

srv::ServerOptions options_for(const std::shared_ptr<SlowBackend>& backend, srv::SchedulerPolicy scheduler) {
    auto o = Fixture::defaults();
    o.backend.backend.clear();
    o.backend_instance = backend;
    o.models = {"slow:tiny"};
    o.scheduler = scheduler;
    o.shutdown_grace = std::chrono::milliseconds(1000);
    return o;
}

std::string request_body(const std::string& content, bool stream = false, const std::string& extra = {}) {
    return R"({"model":"slow:tiny","messages":[{"role":"user","content":")" + content +
           R"("}],"stream":)" + (stream ? "true" : "false") + extra + "}";
}

RunningRequest launch(std::uint16_t port, const std::shared_ptr<SlowControl>& current_control, const std::string& body,
                      const std::vector<std::pair<std::string, std::string>>& headers = {}) {
    RunningRequest request;
    request.control = current_control;
    const auto parsed = det::parse_chat_request(body);
    REQUIRE(std::holds_alternative<det::ChatJob>(parsed));
    request.prompt = si::format_chat_prompt(std::get<det::ChatJob>(parsed).messages);
    auto reply = request.reply;
    request.thread = std::thread([reply, port, body, headers] { *reply = post(port, "/v1/chat/completions", body, headers); });
    return request;
}

void finish_and_join(SlowControl& control, RunningRequest& request) {
    control.finish_one(request.prompt);
    if (request.thread.joinable()) request.thread.join();
}

TEST_CASE("default admission preserves concurrent backend starts for unknown and known capacity") {
    // Every backend call remains blocked until ALL N calls have started. A
    // capacity-one fallback (or gating by the advertised single slot) fails
    // this barrier, even if all requests would eventually return success.
    constexpr std::size_t count = 3;
    for (const bool remote : {false, true}) {
        for (const std::size_t capacity : {std::size_t{0}, std::size_t{1}}) {
            for (const auto policy : {srv::SchedulerPolicy::automatic, srv::SchedulerPolicy::gate,
                                      srv::SchedulerPolicy::account, srv::SchedulerPolicy::off}) {
                CAPTURE(remote);
                CAPTURE(capacity);
                CAPTURE(static_cast<int>(policy));
                auto control = std::make_shared<SlowControl>();
                auto backend = std::make_shared<SlowBackend>(control, capacity, remote);
                Fixture fixture(options_for(backend, policy));
                CHECK_FALSE(fixture.server->engine()->scheduling_options().strict_priority_admission);
                std::vector<RunningRequest> requests;
                requests.reserve(count);
                for (std::size_t i = 0; i < count; ++i) {
                    requests.push_back(launch(fixture.port, control, request_body("parallel-" + std::to_string(i))));
                }
                const bool concurrent = control->wait_for_starts(count);
                CHECK(concurrent);
                CHECK(control->starts().size() == count);
                const auto health = get(fixture.port, "/v1/sonder/health").json();
                for (const char* cls : {"interactive", "subagent", "background"}) {
                    CHECK(health.find("queued_by_class")->find(cls)->as_uint() == 0);
                }
                control->release_all();
                for (auto& request : requests) {
                    request.thread.join();
                    CHECK(request.reply->status == 200);
                    CHECK(request.reply->json().find("choices")->as_array().at(0)
                              .find("message")->find("content")->as_string() == "x");
                }
            }
        }
    }
}

TEST_CASE("unknown capacity remains ungated when priority admission is on") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 0, true);
    auto options = options_for(backend, srv::SchedulerPolicy::account);
    options.priority_admission = srv::PriorityAdmissionPolicy::on;
    Fixture fixture(std::move(options));
    auto first = launch(fixture.port, control, request_body("first"));
    auto second = launch(fixture.port, control, request_body("second"));
    auto third = launch(fixture.port, control, request_body("third"));
    CHECK(control->wait_for_starts(3));
    control->release_all();
    for (auto* request : {&first, &second, &third}) {
        request->thread.join();
        CHECK(request->reply->status == 200);
    }
}

TEST_CASE("explicit backend capacity gates unknown capacity only with admission enabled") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 0, true);
    auto options = options_for(backend, srv::SchedulerPolicy::account);
    options.backend_capacity = 1;
    options.priority_admission = srv::PriorityAdmissionPolicy::on;
    Fixture fixture(std::move(options));
    auto first = launch(fixture.port, control, request_body("first"));
    REQUIRE(control->wait_for_starts(1));
    auto second = launch(fixture.port, control, request_body("second"));
    REQUIRE(eventually([&] {
        return get(fixture.port, "/v1/sonder/health").json().find("queued_by_class")
                   ->find("interactive")->as_uint() == 1;
    }));
    CHECK(control->starts().size() == 1);
    finish_and_join(*control, first);
    REQUIRE(control->wait_for_starts(2));
    finish_and_join(*control, second);
    CHECK(first.reply->status == 200);
    CHECK(second.reply->status == 200);
}

TEST_CASE("off disables admission caps and auto requires a class or queue cap") {
    for (const auto mode : {srv::PriorityAdmissionPolicy::automatic, srv::PriorityAdmissionPolicy::off}) {
        auto control = std::make_shared<SlowControl>();
        auto backend = std::make_shared<SlowBackend>(control, 1, true);
        auto options = options_for(backend, srv::SchedulerPolicy::account);
        options.priority_admission = mode;
        options.backend_capacity = 1;
        if (mode == srv::PriorityAdmissionPolicy::off) {
            options.max_concurrent_background = 1;
            options.max_queue_per_class = 1;
        }
        Fixture fixture(std::move(options));
        auto first = launch(fixture.port, control, request_body("first"), {{"X-Sonder-Priority", "background"}});
        auto second = launch(fixture.port, control, request_body("second"), {{"X-Sonder-Priority", "background"}});
        auto third = launch(fixture.port, control, request_body("third"), {{"X-Sonder-Priority", "background"}});
        CHECK(control->wait_for_starts(3));
        control->release_all();
        for (auto* request : {&first, &second, &third}) {
            request->thread.join();
            CHECK(request->reply->status == 200);
        }
    }
}

TEST_CASE("priority ordering is strict in gate and account modes") {
    for (const auto policy : {srv::SchedulerPolicy::gate, srv::SchedulerPolicy::account}) {
        auto control = std::make_shared<SlowControl>();
        auto backend = std::make_shared<SlowBackend>(control, 1);
        auto options = options_for(backend, policy);
        options.priority_admission = srv::PriorityAdmissionPolicy::on;
        Fixture fixture(std::move(options));

        auto first = launch(fixture.port, control, request_body("first"), {{"X-Sonder-Priority", "background"}});
        REQUIRE(control->wait_for_starts(1));
        auto queued_background = launch(fixture.port, control, request_body("queued-background"),
                                        {{"X-Sonder-Priority", "background"}});
        REQUIRE(eventually([&] {
            const auto health = get(fixture.port, "/v1/sonder/health").json();
            return health.find("queued_by_class")->find("background")->as_uint() == 1;
        }));
        auto interactive = launch(fixture.port, control, request_body("interactive"),
                                  {{"X-Sonder-Priority", "interactive"}});
        REQUIRE(eventually([&] {
            const auto health = get(fixture.port, "/v1/sonder/health").json();
            return health.find("queued_by_class")->find("interactive")->as_uint() == 1;
        }));

        finish_and_join(*control, first);
        REQUIRE(control->wait_for_starts(2));
        auto starts = control->starts();
        CHECK(starts[1].find("interactive") != std::string::npos);
        finish_and_join(*control, interactive);
        REQUIRE(control->wait_for_starts(3));
        starts = control->starts();
        CHECK(starts[2].find("queued-background") != std::string::npos);
        finish_and_join(*control, queued_background);
        CHECK(first.reply->status == 200);
        CHECK(interactive.reply->status == 200);
        CHECK(queued_background.reply->status == 200);
    }
}

TEST_CASE("subagent and background caps coexist while interactive work advances") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 2);
    auto options = options_for(backend, srv::SchedulerPolicy::account);
    options.max_concurrent_subagent = 1;
    options.max_concurrent_background = 1;
    Fixture fixture(std::move(options));

    auto subagent_1 = launch(fixture.port, control, request_body("subagent-1"), {{"X-Sonder-Priority", "subagent"}});
    REQUIRE(control->wait_for_starts(1));
    auto subagent_2 = launch(fixture.port, control, request_body("subagent-2"), {{"X-Sonder-Priority", "subagent"}});
    auto background_1 = launch(fixture.port, control, request_body("background-1"), {{"X-Sonder-Priority", "background"}});
    REQUIRE(control->wait_for_starts(2));
    auto background_2 = launch(fixture.port, control, request_body("background-2"), {{"X-Sonder-Priority", "background"}});
    auto interactive = launch(fixture.port, control, request_body("interactive"));
    REQUIRE(eventually([&] {
        const auto health = get(fixture.port, "/v1/sonder/health").json();
        const auto* q = health.find("queued_by_class");
        return q->find("subagent")->as_uint() == 1 && q->find("interactive")->as_uint() == 1 &&
               q->find("background")->as_uint() == 1;
    }));
    auto starts = control->starts();
    CHECK(std::count_if(starts.begin(), starts.end(), [](const std::string& s) { return s.find("subagent-2") != std::string::npos; }) == 0);
    finish_and_join(*control, subagent_1);
    REQUIRE(control->wait_for_starts(3));
    starts = control->starts();
    CHECK(starts[2].find("interactive") != std::string::npos);
    finish_and_join(*control, interactive);
    finish_and_join(*control, subagent_2);
    finish_and_join(*control, background_1);
    finish_and_join(*control, background_2);
    CHECK(subagent_2.reply->status == 200);
    CHECK(background_1.reply->status == 200);
    CHECK(background_2.reply->status == 200);
}

TEST_CASE("numeric headers retain interactive class and FIFO admission independent of scheduler rank") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 1);
    auto options = options_for(backend, srv::SchedulerPolicy::account);
    options.priority_admission = srv::PriorityAdmissionPolicy::on;
    Fixture fixture(std::move(options));
    auto first = launch(fixture.port, control, request_body("first"));
    REQUIRE(control->wait_for_starts(1));
    auto background = launch(fixture.port, control, request_body("background"), {{"X-Sonder-Priority", "background"}});
    REQUIRE(eventually([&] {
        return get(fixture.port, "/v1/sonder/health").json().find("queued_by_class")->find("background")->as_uint() == 1;
    }));
    // The first numeric request has a WORSE legacy scheduler rank, yet both
    // numeric headers are interactive admission requests in arrival order.
    auto low_numeric = launch(fixture.port, control, request_body("low-numeric", false, R"(,"priority":"background")"),
                              {{"X-Sonder-Priority", "-16"}, {"X-Sonder-Workload", "maintenance"}});
    REQUIRE(eventually([&] {
        return get(fixture.port, "/v1/sonder/health").json().find("queued_by_class")->find("interactive")->as_uint() == 1;
    }));
    auto high_numeric = launch(fixture.port, control, request_body("high-numeric"), {{"X-Sonder-Priority", "16"}});
    REQUIRE(eventually([&] {
        return get(fixture.port, "/v1/sonder/health").json().find("queued_by_class")->find("interactive")->as_uint() == 2;
    }));
    finish_and_join(*control, first);
    REQUIRE(control->wait_for_starts(2));
    CHECK(control->starts()[1] == low_numeric.prompt);
    finish_and_join(*control, low_numeric);
    REQUIRE(control->wait_for_starts(3));
    CHECK(control->starts()[2] == high_numeric.prompt);
    finish_and_join(*control, high_numeric);
    REQUIRE(control->wait_for_starts(4));
    CHECK(control->starts()[3] == background.prompt);
    finish_and_join(*control, background);
    for (auto* request : {&first, &low_numeric, &high_numeric, &background}) CHECK(request->reply->status == 200);
}

TEST_CASE("per-class queue cap returns 429 with Retry-After") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 1);
    auto options = options_for(backend, srv::SchedulerPolicy::account);
    options.max_queue_per_class = 1;
    Fixture fixture(std::move(options));

    auto first = launch(fixture.port, control, request_body("first"));
    REQUIRE(control->wait_for_starts(1));
    auto queued = launch(fixture.port, control, request_body("queued"), {{"X-Sonder-Priority", "background"}});
    REQUIRE(eventually([&] { return get(fixture.port, "/v1/sonder/health").json().find("queued_by_class")->find("background")->as_uint() == 1; }));
    auto rejected = post(fixture.port, "/v1/chat/completions", request_body("rejected"), {{"X-Sonder-Priority", "background"}});
    CHECK(rejected.status == 429);
    CHECK(rejected.header("retry-after") == "1");
    finish_and_join(*control, first);
    finish_and_join(*control, queued);
}

TEST_CASE("logical KV waiting shares the priority queue and queue cap with backend capacity") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 2);  // one backend slot remains free
    auto options = options_for(backend, srv::SchedulerPolicy::gate);
    options.kv_pool_tokens = 16;  // one block: a single running request holds all logical KV
    options.max_queue_per_class = 1;
    Fixture fixture(std::move(options));
    const auto body = [](const std::string& name) {
        return request_body(name, false, R"(,"max_tokens":4)");
    };
    auto first = launch(fixture.port, control, body("first"));
    REQUIRE(control->wait_for_starts(1));
    auto background = launch(fixture.port, control, body("background"), {{"X-Sonder-Priority", "background"}});
    REQUIRE(eventually([&] {
        const auto health = get(fixture.port, "/v1/sonder/health").json();
        return health.find("queued_by_class")->find("background")->as_uint() == 1;
    }));
    CHECK(post(fixture.port, "/v1/chat/completions", body("full"),
               {{"X-Sonder-Priority", "background"}}).status == 429);
    auto interactive = launch(fixture.port, control, body("interactive"));
    REQUIRE(eventually([&] {
        const auto health = get(fixture.port, "/v1/sonder/health").json();
        return health.find("queued_by_class")->find("interactive")->as_uint() == 1;
    }));
    CHECK(control->starts().size() == 1);
    finish_and_join(*control, first);
    REQUIRE(control->wait_for_starts(2));
    CHECK(control->starts()[1].find("interactive") != std::string::npos);
    finish_and_join(*control, interactive);
    REQUIRE(control->wait_for_starts(3));
    CHECK(control->starts()[2].find("background") != std::string::npos);
    finish_and_join(*control, background);
    CHECK(interactive.reply->status == 200);
    CHECK(background.reply->status == 200);
}

TEST_CASE("header hints win after both body and header validation") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 1);
    auto options = options_for(backend, srv::SchedulerPolicy::account);
    options.priority_admission = srv::PriorityAdmissionPolicy::on;
    Fixture fixture(std::move(options));
    CHECK(post(fixture.port, "/v1/chat/completions", request_body("invalid", false, R"(,"priority":"bad")"),
               {{"X-Sonder-Priority", "interactive"}}).status == 400);
    CHECK(post(fixture.port, "/v1/chat/completions", request_body("bad", false, R"(,"deadline_ms":-1)"),
               {{"X-Sonder-Deadline-Ms", "1000"}}).status == 400);
    auto first = launch(fixture.port, control, request_body("first"));
    REQUIRE(control->wait_for_starts(1));
    auto hinted = launch(fixture.port, control, request_body("hinted", false,
                          R"(,"priority":"background","deadline_ms":1)"),
                          {{"X-Sonder-Priority", "interactive"}, {"X-Sonder-Deadline-Ms", "3000"}});
    REQUIRE(eventually([&] {
        const auto health = get(fixture.port, "/v1/sonder/health").json();
        return health.find("queued_by_class")->find("interactive")->as_uint() == 1 &&
               health.find("queued_by_class")->find("background")->as_uint() == 0;
    }));
    finish_and_join(*control, first);
    REQUIRE(control->wait_for_starts(2));
    finish_and_join(*control, hinted);
    CHECK(hinted.reply->status == 200);
}

TEST_CASE("queued deadlines return 504 without starting the backend") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 1);
    auto options = options_for(backend, srv::SchedulerPolicy::account);
    options.priority_admission = srv::PriorityAdmissionPolicy::on;
    Fixture fixture(std::move(options));
    auto first = launch(fixture.port, control, request_body("first"));
    REQUIRE(control->wait_for_starts(1));
    auto expired = post(fixture.port, "/v1/chat/completions", request_body("expired", false, R"(,"deadline_ms":100)"));
    CHECK(expired.status == 504);
    CHECK(control->starts().size() == 1);
    finish_and_join(*control, first);
}

TEST_CASE("running deadlines cancel nonstream and SSE requests") {
    for (const bool stream : {false, true}) {
        auto control = std::make_shared<SlowControl>();
        auto backend = std::make_shared<SlowBackend>(control, 1);
        Fixture fixture(options_for(backend, srv::SchedulerPolicy::account));
        auto timed = launch(fixture.port, control, request_body("timed", stream, R"(,"deadline_ms":100)"));
        REQUIRE(control->wait_for_starts(1));
        timed.thread.join();
        CHECK(timed.reply->status == 200);
        CHECK(timed.reply->body.find("cancelled") != std::string::npos);
        CHECK(control->cancelled() > 0);
    }
}

TEST_CASE("client disconnect cancels the running fake backend") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 1);
    Fixture fixture(options_for(backend, srv::SchedulerPolicy::account));
    Conn connection(fixture.port);
    connection.send(build_request("POST", "/v1/chat/completions", fixture.port, {}, request_body("disconnect", true)));
    std::string received;
    REQUIRE(connection.read_until(received, "data:"));
    connection.close();
    REQUIRE(control->wait_for_starts(1));
    REQUIRE(eventually([&] { return control->cancelled() > 0; }));
}

TEST_CASE("omitted priority matches explicit interactive and preserves chat response fields") {
    auto control = std::make_shared<SlowControl>();
    auto backend = std::make_shared<SlowBackend>(control, 1);
    Fixture fixture(options_for(backend, srv::SchedulerPolicy::account));
    auto omitted = launch(fixture.port, control, request_body("default"));
    REQUIRE(control->wait_for_starts(1));
    finish_and_join(*control, omitted);
    auto explicit_interactive = launch(fixture.port, control, request_body("explicit"), {{"X-Sonder-Priority", "interactive"}});
    REQUIRE(control->wait_for_starts(2));
    finish_and_join(*control, explicit_interactive);
    CHECK(omitted.reply->status == 200);
    CHECK(explicit_interactive.reply->status == 200);
    const auto omitted_json = omitted.reply->json();
    const auto explicit_json = explicit_interactive.reply->json();
    CHECK(omitted_json.find("object")->as_string() == explicit_json.find("object")->as_string());
    CHECK(omitted_json.find("model")->as_string() == explicit_json.find("model")->as_string());
    CHECK(omitted_json.find("choices")->dump() == explicit_json.find("choices")->dump());
    CHECK(omitted_json.find("object")->as_string() == "chat.completion");
    CHECK(omitted_json.find("model")->as_string() == "slow:tiny");
    const auto& choice = omitted_json.find("choices")->as_array().at(0);
    CHECK(choice.find("message")->find("role")->as_string() == "assistant");
    CHECK(choice.find("message")->find("content")->as_string() == "x");
    CHECK(choice.find("finish_reason")->as_string() == "stop");
    CHECK(omitted_json.find("usage")->find("prompt_tokens")->as_int() == 1);
    CHECK(omitted_json.find("usage")->find("completion_tokens")->as_int() == 1);
    CHECK(omitted_json.find("usage")->find("total_tokens")->as_int() == 2);
    CHECK(omitted_json.find("sonder")->find("api_version")->as_int() ==
          explicit_json.find("sonder")->find("api_version")->as_int());
    // The unrequested text endpoint is not exposed by this priority slice.
    CHECK(post(fixture.port, "/v1/completions", R"({"prompt":"completion"})").status == 404);
}

}  // namespace
