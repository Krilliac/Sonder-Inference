#include <doctest/doctest.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <vector>
#include <functional>

#if defined(SONDER_HAS_SCHEDULER)
#include "sonder/inference/scheduler/scheduler.hpp"
#include "sonder/inference.hpp"
#if defined(SONDER_HAS_KV_CACHE)
#include "engine/request_runtime.hpp"
#endif

using namespace sonder::inference::scheduler;

namespace {
RequestSpec request(RequestId id, WorkloadClass cls) {
    RequestSpec r;
    r.id = id;
    r.workload = cls;
    r.prompt_tokens = 1;
    r.max_new_tokens = 1;
    return r;
}
}  // namespace

TEST_CASE("strict admission is class ordered and FIFO without aging") {
    SchedulerConfig config;
    config.strict_priority_admission = true;
    config.max_running_sequences = 1;
    config.max_step_sequences = 1;
    config.max_step_tokens = 1;
    config.max_step_prefill_tokens = 1;
    config.prefill_chunk_tokens = 1;
    config.admission_watermark_blocks = 0;
    SimClock clock;
    FixedKvCapacity kv(8, 1);
    Scheduler scheduler(config, clock, kv);

    CHECK(scheduler.submit(request(1, WorkloadClass::Maintenance)).accepted);
    CHECK(scheduler.submit(request(2, WorkloadClass::ImplementationWorker)).accepted);
    CHECK(scheduler.submit(request(3, WorkloadClass::ImplementationWorker)).accepted);
    CHECK(scheduler.submit(request(4, WorkloadClass::InteractiveUser)).accepted);
    CHECK(scheduler.queue_order() == std::vector<RequestId>{4, 2, 3, 1});

    clock.advance(config.starvation_threshold_us + config.aging_interval_us * 4);
    CHECK(scheduler.queue_order() == std::vector<RequestId>{4, 2, 3, 1});
}

TEST_CASE("outer admission rejection leaves the request queued without blocking an eligible class") {
    SchedulerConfig config;
    config.strict_priority_admission = true;
    config.max_step_sequences = 2;
    config.max_step_tokens = 2;
    config.max_step_prefill_tokens = 2;
    config.prefill_chunk_tokens = 1;
    config.admission_watermark_blocks = 0;
    SimClock clock;
    FixedKvCapacity kv(8, 1);
    Scheduler scheduler(config, clock, kv);

    bool background_allowed = false;
    auto background = request(1, WorkloadClass::Maintenance);
    background.try_admit = [&] { return background_allowed; };
    auto interactive = request(2, WorkloadClass::InteractiveUser);
    CHECK(scheduler.submit(background).accepted);
    CHECK(scheduler.submit(interactive).accepted);

    const StepPlan first = scheduler.plan_step();
    CHECK(first.admitted == std::vector<RequestId>{2});
    CHECK(first.preempted.empty());
    CHECK(kv.blocks_held(1) == 0);
    scheduler.complete_step(first);

    background_allowed = true;
    const StepPlan second = scheduler.plan_step();
    CHECK(second.admitted == std::vector<RequestId>{1});
    scheduler.complete_step(second);
}

TEST_CASE("strict admission never urgency-preempts running work") {
    SchedulerConfig config;
    config.strict_priority_admission = true;
    config.max_step_sequences = 2;
    config.max_step_tokens = 2;
    config.max_step_prefill_tokens = 2;
    config.prefill_chunk_tokens = 1;
    config.max_running_sequences = 2;
    config.admission_watermark_blocks = 0;
    SimClock clock;
    FixedKvCapacity kv(6, 1);
    Scheduler scheduler(config, clock, kv);

    auto background = request(1, WorkloadClass::Maintenance);
    background.max_new_tokens = 4;
    CHECK(scheduler.submit(background).accepted);
    StepPlan first = scheduler.plan_step();
    REQUIRE(first.admitted == std::vector<RequestId>{1});
    scheduler.complete_step(first);

    CHECK(scheduler.submit(request(2, WorkloadClass::InteractiveUser)).accepted);
    const StepPlan contention = scheduler.plan_step();
    CHECK(contention.preempted.empty());
    CHECK(contention.admitted.empty());
    scheduler.complete_step(contention);
}

TEST_CASE("strict mode is opt-in and default scheduler configuration is unchanged") {
    SchedulerConfig config;
    CHECK_FALSE(config.strict_priority_admission);
    SimClock clock;
    FixedKvCapacity kv(100, 16);
    Scheduler scheduler(config, clock, kv);
    CHECK(scheduler.submit(request(1, WorkloadClass::Maintenance)).accepted);
    clock.advance(config.aging_interval_us + 1);
    CHECK(*scheduler.effective_rank(1) < static_cast<int>(WorkloadClass::Maintenance));
}

#if defined(SONDER_HAS_KV_CACHE)
namespace {
// Opening the host gate deliberately sends no runtime notification. Host
// permits can be released after Session has already called runtime.finish().
class ControlledAdmission final : public sonder::inference::RequestAdmission {
public:
    bool ready() const override {
        std::lock_guard<std::mutex> lock(mutex_);
        checked_ = true;
        cv_.notify_all();
        return allowed_;
    }
    std::uint64_t order() const noexcept override { return 0; }
    bool try_acquire() override {
        std::lock_guard<std::mutex> lock(mutex_);
        acquired_ = allowed_;
        return acquired_;
    }
    sonder::inference::Status wait(const sonder::inference::CancellationToken&) override {
        return sonder::inference::Status(sonder::inference::ErrorCode::internal,
                                         "this test requires scheduler admission");
    }
    double queue_ms() const override { return 0.0; }
    bool wait_until_checked() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return cv_.wait_for(lock, std::chrono::seconds(2), [&] { return checked_; });
    }
    void allow() {
        std::lock_guard<std::mutex> lock(mutex_);
        allowed_ = true;
    }
    bool acquired() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return acquired_;
    }

private:
    mutable std::mutex mutex_;
    mutable std::condition_variable cv_;
    mutable bool checked_ = false;
    bool allowed_ = false;
    bool acquired_ = false;
};
}  // namespace

TEST_CASE("runtime rechecks host admission without another engine event") {
    namespace si = sonder::inference;
    // Empty-plan waits cover gate and account. The lagging-plan variant
    // retains an unrelated gated request that never consumes its first grant.
    for (const bool lagging : {false, true}) {
        for (const bool gated : {false, true}) {
            CAPTURE(lagging);
            CAPTURE(gated);
            si::EngineOptions options;
            options.scheduling.strict_priority_admission = true;
            options.scheduling.step_stall_timeout_ms = 1;
            options.device_sample_interval = std::chrono::milliseconds(0);
            options.telemetry.level = si::TelemetryLevel::off;
            si::Engine engine(options);
            auto* runtime = engine.request_runtime();
            REQUIRE(runtime != nullptr);
            std::optional<std::uint64_t> stalled_id;
            if (lagging) {
                si::detail::RuntimeRequestSpec stalled;
                stalled.prompt_tokens = {1};
                stalled.max_new_tokens = 4;
                const auto submitted = runtime->submit(std::move(stalled));
                REQUIRE(submitted.ok());
                stalled_id = submitted.value().id;
                REQUIRE(runtime->admit_backend(*stalled_id, {},
                    std::chrono::steady_clock::now() + std::chrono::seconds(2)).ok());
            }
            auto ticket = std::make_shared<ControlledAdmission>();
            si::detail::RuntimeRequestSpec spec;
            spec.prompt_tokens = {2};
            spec.gated = gated;
            spec.admission = ticket;
            const auto submitted = runtime->submit(std::move(spec));
            REQUIRE(submitted.ok());
            REQUIRE(ticket->wait_until_checked());
            // ready() ran under the coordinator mutex. Taking that mutex
            // here ensures the blocked plan has reached its wait before we
            // open the host gate, without sleeps or a new runtime event.
            CHECK(runtime->tracked_requests() == (lagging ? 2u : 1u));
            CHECK_FALSE(ticket->acquired());
            ticket->allow();
            const auto admitted = runtime->admit_backend(submitted.value().id, {},
                std::chrono::steady_clock::now() + std::chrono::seconds(2));
            CHECK(admitted.ok());
            CHECK(ticket->acquired());
            (void)runtime->finish(submitted.value().id, si::RequestOutcome::cancelled);
            if (stalled_id) (void)runtime->finish(*stalled_id, si::RequestOutcome::cancelled);
            CHECK(runtime->tracked_requests() == 0);
        }
    }
}

TEST_CASE("an expired strict request never opens a token-logit prefill") {
    namespace si = sonder::inference;
    struct CountingModel final : si::BackendModel {
        si::ModelDescriptor description;
        int opens = 0;
        int generates = 0;
        CountingModel() {
            description.name = "deadline-model";
            description.backend = "deadline-test";
            description.context_length = 128;
        }
        const si::ModelDescriptor& descriptor() const override { return description; }
        si::Result<std::vector<si::TokenId>> tokenize(std::string_view) override {
            return std::vector<si::TokenId>{1};
        }
        si::Result<std::unique_ptr<si::TokenStream>> open_token_stream(const si::GenerateRequest&) override {
            ++opens;
            return si::Status(si::ErrorCode::unsupported, "must not prefill an expired request");
        }
        si::Result<si::GenerateStats> generate(const si::GenerateRequest&, const si::CancellationToken&,
                                             const si::TokenCallback&) override {
            ++generates;
            return si::GenerateStats{};
        }
    };
    struct CountingBackend final : si::Backend {
        std::shared_ptr<CountingModel> model = std::make_shared<CountingModel>();
        std::string name() const override { return "deadline-test"; }
        std::string description() const override { return "no-I/O deadline test backend"; }
        si::BackendCapabilities capabilities() const override {
            si::BackendCapabilities caps;
            caps.add(si::Capability::token_logits).add(si::Capability::tokenization);
            return caps;
        }
        si::Result<std::string> probe() override { return std::string("test"); }
        si::Result<std::vector<si::ModelDescriptor>> list_models() override {
            return std::vector<si::ModelDescriptor>{model->descriptor()};
        }
        si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions&) override {
            return std::static_pointer_cast<si::BackendModel>(model);
        }
    };
    auto backend = std::make_shared<CountingBackend>();
    si::EngineOptions options;
    options.scheduling.strict_priority_admission = true;
    options.device_sample_interval = std::chrono::milliseconds(0);
    options.telemetry.level = si::TelemetryLevel::off;
    si::Engine engine(options);
    REQUIRE(engine.register_backend(backend).ok());
    auto model = engine.load_model(backend->name(), si::ModelLoadOptions{"deadline-model", "cpu:0"});
    REQUIRE(model.ok());
    auto session = engine.create_session(model.value());
    REQUIRE(session.ok());
    si::RequestOptions request_options;
    request_options.deadline = std::chrono::steady_clock::now() - std::chrono::milliseconds(1);
    bool started = false;
    request_options.on_backend_start = [&] { started = true; };
    const auto result = session.value()->generate("hello", {}, si::SamplingConfig::greedy(1), request_options);
    CHECK_FALSE(result.ok());
    CHECK(result.status().code() == si::ErrorCode::timeout);
    CHECK_FALSE(started);
    CHECK(backend->model->opens == 0);
    CHECK(backend->model->generates == 0);
}
#endif

#endif  // SONDER_HAS_SCHEDULER
