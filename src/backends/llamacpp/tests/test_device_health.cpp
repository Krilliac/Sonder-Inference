// Unit and adapter tests for in-process GPU device health detection and fail-fast policy.
#include <doctest/doctest.h>

#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "common/diag_log.hpp"
#include "device_health.hpp"
#include "llamacpp_internal.hpp"
#include "radix_cache.hpp"
#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/json.hpp"

namespace {

namespace lc = sonder::backends::llamacpp;
namespace si = sonder::inference;

bool CheckIsDeviceUnhealthyStatus(const si::Status& st) noexcept {
#if defined(SONDER_HAS_LLAMACPP_DEVICE_HEALTH_API)
    return lc::IsDeviceUnhealthyInferenceStatus(st) && si::is_llamacpp_device_unhealthy_error(st);
#else
    return lc::IsDeviceUnhealthyInferenceStatus(st);
#endif
}

si::Status CallRecoverBackendDevice(si::Backend& backend, std::string_view device) {
#if defined(SONDER_HAS_LLAMACPP_DEVICE_HEALTH_API)
    return si::recover_llamacpp_device(backend, device);
#else
    return lc::RecoverCoreBackendDevice(backend, device);
#endif
}

}  // namespace

TEST_CASE("device_health: classify v1.0.0 llama_decode and llama_process return codes") {
    // Non-fault return codes in llama.cpp v1.0.0:
    //   0 = success, 1 = no KV slot warning, 2 = aborted, >2 = positive warning, -1 = invalid batch.
    CHECK(lc::ClassifyDecodeStatus(0) == lc::DeviceFaultReason::kNone);
    CHECK(lc::ClassifyDecodeStatus(1) == lc::DeviceFaultReason::kNone);
    CHECK(lc::ClassifyDecodeStatus(2) == lc::DeviceFaultReason::kNone);
    CHECK(lc::ClassifyDecodeStatus(3) == lc::DeviceFaultReason::kNone);
    CHECK(lc::ClassifyDecodeStatus(-1) == lc::DeviceFaultReason::kNone);

    // Fatal backend return codes in llama.cpp v1.0.0:
    //  -2 = GGML_STATUS_ALLOC_FAILED (CUDA/Vulkan buffer alloc nullptr -> sched_alloc_graph failure)
    //  -3 and < -3 = GGML_STATUS_FAILED or fatal backend execution failure.
    CHECK(lc::ClassifyDecodeStatus(-2) == lc::DeviceFaultReason::kDecodeAllocFailed);
    CHECK(lc::ClassifyDecodeStatus(-3) == lc::DeviceFaultReason::kDecodeComputeFailed);
    CHECK(lc::ClassifyDecodeStatus(-4) == lc::DeviceFaultReason::kDecodeComputeFailed);
    CHECK(lc::ClassifyDecodeStatus(INT32_MIN) == lc::DeviceFaultReason::kDecodeComputeFailed);

    CHECK(lc::ToString(lc::DeviceFaultReason::kDecodeAllocFailed) == "decode_alloc_failed");
    CHECK(lc::ToString(lc::DeviceFaultReason::kDecodeComputeFailed) == "decode_compute_failed");
}

TEST_CASE("device_health: classify v1.0.0 ggml_status and Vulkan system_error codes") {
    CHECK(lc::ClassifyGgmlStatus(lc::GgmlComputeStatus::kSuccess) == lc::DeviceFaultReason::kNone);
    CHECK(lc::ClassifyGgmlStatus(lc::GgmlComputeStatus::kAborted) == lc::DeviceFaultReason::kNone);
    CHECK(lc::ClassifyGgmlStatus(lc::GgmlComputeStatus::kAllocFailed) == lc::DeviceFaultReason::kGgmlAllocFailed);
    CHECK(lc::ClassifyGgmlStatus(lc::GgmlComputeStatus::kFailed) == lc::DeviceFaultReason::kGgmlComputeFailed);
    CHECK(lc::ClassifyGgmlStatus(-99) == lc::DeviceFaultReason::kGgmlComputeFailed);
    CHECK(lc::ClassifyGgmlStatus(99) == lc::DeviceFaultReason::kGgmlComputeFailed);

    // Vulkan-Hpp vk::DeviceLostError (VK_ERROR_DEVICE_LOST = -4) and
    // vk::OutOfDeviceMemoryError (VK_ERROR_OUT_OF_DEVICE_MEMORY = -2).
    CHECK(lc::ClassifyBackendSystemError(lc::kVkErrorDeviceLost) == lc::DeviceFaultReason::kDeviceLost);
    CHECK(lc::ClassifyBackendSystemError(lc::kVkErrorOutOfDeviceMemory) == lc::DeviceFaultReason::kDeviceOutOfMemory);
    CHECK(lc::ClassifyBackendSystemError(lc::kVkErrorInitializationFailed) ==
          lc::DeviceFaultReason::kBackendDeviceError);
    CHECK(lc::ClassifyBackendSystemError(5) == lc::DeviceFaultReason::kBackendDeviceError);

    CHECK(lc::ToString(lc::DeviceFaultReason::kGgmlAllocFailed) == "ggml_alloc_failed");
    CHECK(lc::ToString(lc::DeviceFaultReason::kGgmlComputeFailed) == "ggml_compute_failed");
    CHECK(lc::ToString(lc::DeviceFaultReason::kDeviceLost) == "device_lost");
    CHECK(lc::ToString(lc::DeviceFaultReason::kDeviceOutOfMemory) == "device_out_of_memory");
    CHECK(lc::ToString(lc::DeviceFaultReason::kBackendDeviceError) == "backend_device_error");
}

TEST_CASE("device_health: state machine transitions on each failing status then fails fast until explicit recovery") {
    struct FaultCase {
        const char* label;
        const char* device;
        int mode;  // 0 = decode rc, 1 = ggml_status, 2 = system_error
        int code;
        lc::DeviceFaultReason expected_reason;
        const char* expected_reason_str;
    };

    const FaultCase cases[] = {
        {"decode alloc failed (-2)", "CUDA0", 0, -2, lc::DeviceFaultReason::kDecodeAllocFailed, "decode_alloc_failed"},
        {"decode compute failed (-3)", "CUDA1", 0, -3, lc::DeviceFaultReason::kDecodeComputeFailed,
         "decode_compute_failed"},
        {"ggml alloc failed (-2)", "Vulkan0", 1, -2, lc::DeviceFaultReason::kGgmlAllocFailed, "ggml_alloc_failed"},
        {"ggml compute failed (-1)", "Vulkan1", 1, -1, lc::DeviceFaultReason::kGgmlComputeFailed,
         "ggml_compute_failed"},
        {"vulkan device lost (-4)", "Vulkan0", 2, lc::kVkErrorDeviceLost, lc::DeviceFaultReason::kDeviceLost,
         "device_lost"},
        {"vulkan device OOM (-2)", "Vulkan1", 2, lc::kVkErrorOutOfDeviceMemory,
         lc::DeviceFaultReason::kDeviceOutOfMemory, "device_out_of_memory"},
    };

    for (const auto& tc : cases) {
        CAPTURE(tc.label);
        lc::DeviceHealthTracker tracker("llamacpp");

        // 1. Starts healthy; non-faulting codes keep device healthy and emit nothing.
        CHECK(tracker.CheckDevice(tc.device).ok());
        CHECK(tracker.GetDeviceState(tc.device).state == lc::DeviceHealthState::kHealthy);
        CHECK_FALSE(tracker.ObserveDecodeStatus(tc.device, 0));
        CHECK_FALSE(tracker.ObserveDecodeStatus(tc.device, 1));
        CHECK_FALSE(tracker.ObserveDecodeStatus(tc.device, 2));
        CHECK_FALSE(tracker.ObserveDecodeStatus(tc.device, -1));
        CHECK_FALSE(tracker.ObserveGgmlStatus(tc.device, 0));
        CHECK_FALSE(tracker.ObserveGgmlStatus(tc.device, 1));
        CHECK(tracker.CheckDevice(tc.device).ok());
        CHECK(tracker.PendingTelemetryEventCount() == 0u);

        // 2. Failing status transitions target device to unhealthy and queues one telemetry event.
        bool faulted = false;
        if (tc.mode == 0) {
            faulted = tracker.ObserveDecodeStatus(tc.device, static_cast<std::int32_t>(tc.code));
        } else if (tc.mode == 1) {
            faulted = tracker.ObserveGgmlStatus(tc.device, tc.code);
        } else {
            faulted = tracker.ObserveSystemError(tc.device, tc.code);
        }
        CHECK(faulted);

        const lc::DeviceHealthEntry st = tracker.GetDeviceState(tc.device);
        CHECK(st.state == lc::DeviceHealthState::kUnhealthy);
        CHECK(st.reason == tc.expected_reason);
        CHECK(st.raw_status == tc.code);
        CHECK(st.fault_count == 1u);

        // Sibling device remains completely healthy.
        CHECK(tracker.CheckDevice("ROCm9").ok());

        // 3. Fail-fast admission rejects both exact and case-differing device names without calling the driver.
        const lc::DeviceHealthError check_err = tracker.CheckDevice(tc.device);
        CHECK_FALSE(check_err.ok());
        CHECK(check_err.is_device_unhealthy());
        CHECK(check_err.code() == "device_unhealthy");
        CHECK(check_err.reason == tc.expected_reason);
        const si::Status inf_st = check_err.ToInferenceStatus();
        CHECK(inf_st.code() == si::ErrorCode::unavailable);
        CHECK(CheckIsDeviceUnhealthyStatus(inf_st));

        int driver_calls = 0;
        std::int32_t guarded_rc = 0;
        const std::vector<std::string> target_devs{tc.device};
        const lc::DeviceHealthError guarded_err = tracker.ExecuteDecodeGuarded(
            target_devs,
            [&]() -> std::int32_t {
                ++driver_calls;
                return 0;
            },
            &guarded_rc);
        CHECK(driver_calls == 0);
        CHECK(guarded_err.is_device_unhealthy());
        CHECK(guarded_rc == lc::kDeviceUnhealthyDecodeSentinel);

        // 4. Recovery is NEVER automatic: observing healthy status codes does not clear unhealthy state.
        CHECK_FALSE(tracker.ObserveDecodeStatus(tc.device, 0));
        CHECK_FALSE(tracker.ObserveGgmlStatus(tc.device, 0));
        CHECK(tracker.CheckDevice(tc.device).is_device_unhealthy());

        // Subsequent fault observations while already unhealthy do not emit duplicate events.
        CHECK(tracker.ObserveDecodeStatus(tc.device, -3));
        auto events = tracker.DrainTelemetryEvents();
        REQUIRE(events.size() == 1u);
        CHECK(events[0].first == "backend.device_unhealthy");
        REQUIRE(events[0].second.find("backend") != nullptr);
        REQUIRE(events[0].second.find("device") != nullptr);
        REQUIRE(events[0].second.find("reason") != nullptr);
        CHECK(events[0].second.find("backend")->as_string() == "llamacpp");
        CHECK(events[0].second.find("device")->as_string() == tc.device);
        CHECK(events[0].second.find("reason")->as_string() == tc.expected_reason_str);

        // 5. Explicit recovery restores healthy state and permits execution again.
        REQUIRE(tracker.RecoverDevice(tc.device).ok());
        CHECK(tracker.CheckDevice(tc.device).ok());
        CHECK(tracker.GetDeviceState(tc.device).state == lc::DeviceHealthState::kHealthy);

        const lc::DeviceHealthError after_recover = tracker.ExecuteDecodeGuarded(
            target_devs,
            [&]() -> std::int32_t {
                ++driver_calls;
                return 0;
            },
            &guarded_rc);
        CHECK(after_recover.ok());
        CHECK(driver_calls == 1);
        CHECK(guarded_rc == 0);

        // A new fault episode after explicit recovery emits a fresh telemetry event.
        CHECK(tracker.ObserveDecodeStatus(tc.device, -2));
        auto second_events = tracker.DrainTelemetryEvents();
        REQUIRE(second_events.size() == 1u);
        CHECK(second_events[0].first == "backend.device_unhealthy");
        CHECK(second_events[0].second.find("reason")->as_string() == "decode_alloc_failed");
    }
}

TEST_CASE("device_health: ExecuteDecodeGuarded catches thrown std::system_error and marks device unhealthy") {
    lc::DeviceHealthTracker tracker("llamacpp");
    const std::vector<std::string> devs{"Vulkan0"};
    int calls = 0;
    std::int32_t out_rc = 0;

    // 1. Host std::bad_alloc must NOT mark GPU devices unhealthy and must propagate.
    CHECK_THROWS_AS(
        tracker.ExecuteDecodeGuarded(
            devs,
            [&]() -> std::int32_t {
                ++calls;
                throw std::bad_alloc{};
            },
            &out_rc),
        std::bad_alloc);
    CHECK(calls == 1);
    CHECK(tracker.CheckDevice("Vulkan0").ok());
    CHECK(tracker.PendingTelemetryEventCount() == 0u);

    // 2. Non-Vulkan error_category (generic_category / system_category) must NOT have its numeric
    //    value read as a Vulkan VkResult (-4 = kDeviceLost, -2 = kDeviceOutOfMemory).
    CHECK_FALSE(lc::IsVulkanErrorCategory(std::generic_category()));
    CHECK_FALSE(lc::IsVulkanErrorCategory(std::system_category()));
    CHECK(lc::IsVulkanErrorCategory(lc::VulkanResultErrorCategory()));
    CHECK(lc::ClassifyBackendSystemError(std::error_code(lc::kVkErrorDeviceLost, std::generic_category())) ==
          lc::DeviceFaultReason::kBackendDeviceError);
    CHECK(lc::ClassifyBackendSystemError(std::error_code(lc::kVkErrorOutOfDeviceMemory, std::system_category())) ==
          lc::DeviceFaultReason::kBackendDeviceError);
    CHECK(lc::ClassifyBackendSystemError(std::error_code(lc::kVkErrorDeviceLost, lc::VulkanResultErrorCategory())) ==
          lc::DeviceFaultReason::kDeviceLost);
    CHECK(lc::ClassifyBackendSystemError(
              std::error_code(lc::kVkErrorOutOfDeviceMemory, lc::VulkanResultErrorCategory())) ==
          lc::DeviceFaultReason::kDeviceOutOfMemory);

    {
        lc::DeviceHealthTracker generic_tracker("llamacpp");
        const lc::DeviceHealthError gen_err = generic_tracker.ExecuteDecodeGuarded(
            devs,
            [&]() -> std::int32_t {
                throw std::system_error(std::error_code(lc::kVkErrorDeviceLost, std::generic_category()),
                                        "generic POSIX error with numeric -4");
            },
            &out_rc);
        CHECK(gen_err.is_device_unhealthy());
        CHECK(gen_err.reason == lc::DeviceFaultReason::kBackendDeviceError);
    }

    // 3. Simulate ggml_backend_vk_graph_compute throwing vk::DeviceLostError (VulkanResultErrorCategory, code -4).
    const lc::DeviceHealthError err = tracker.ExecuteDecodeGuarded(
        devs,
        [&]() -> std::int32_t {
            ++calls;
            throw std::system_error(std::error_code(lc::kVkErrorDeviceLost, lc::VulkanResultErrorCategory()),
                                    "simulated vulkan device lost");
        },
        &out_rc);

    CHECK(calls == 2);
    CHECK(err.is_device_unhealthy());
    CHECK(err.reason == lc::DeviceFaultReason::kDeviceLost);
    CHECK(out_rc == lc::kDeviceUnhealthyDecodeSentinel);

    // Next call fails fast before entering the lambda.
    const lc::DeviceHealthError err2 = tracker.ExecuteDecodeGuarded(
        devs,
        [&]() -> std::int32_t {
            ++calls;
            return 0;
        },
        &out_rc);
    CHECK(calls == 2);
    CHECK(err2.is_device_unhealthy());
}

TEST_CASE("device_health: unbound CPU backend preserves decode exceptions rollback and prefix cache on -2") {
    lc::LlamaCppBackend unbound_be;
    CHECK_FALSE(lc::HasBoundBackendDeviceTracker(&unbound_be));

    // Unbound RunBoundBackendDecode must not swallow exceptions into -1000.
    CHECK_THROWS_AS(
        (void)lc::RunBoundBackendDecode(&unbound_be,
                                        []() -> std::int32_t { throw std::runtime_error("cpu callback threw"); }),
        std::runtime_error);
    CHECK(lc::RunBoundBackendDecode(&unbound_be, []() -> std::int32_t { return -2; }) == -2);

    // CollectInitialGpuDevices / CollectActiveGpuDevices exclude CPU and deduplicate case-insensitively.
    lc::LoadOptions cpu_lo;
    cpu_lo.n_gpu_layers = 0;
    cpu_lo.devices = {"CPU"};
    CHECK(lc::CollectInitialGpuDevices(cpu_lo, "cpu:0").empty());
    CHECK(lc::CollectActiveGpuDevices(cpu_lo, {}, "cpu:0").empty());
    lc::LoadOptions gpu_lo;
    gpu_lo.n_gpu_layers = -1;
    gpu_lo.devices = {"CPU", "Vulkan0", "vulkan0"};
    const auto active_gpus = lc::CollectActiveGpuDevices(gpu_lo, {}, "gpu:0");
    REQUIRE(active_gpus.size() == 1u);
    CHECK(active_gpus[0] == "Vulkan0");

    // Seed a real RadixPrefixCache entry representing an existing session's cached prefix.
    lc::RadixPrefixCache cache(/*max_bytes=*/4096);
    const std::vector<lc::Token> cached_prefix{10, 20, 30, 40};
    lc::RadixCacheEntry entry;
    entry.state_blob.assign(128, 0xAB);
    REQUIRE(cache.Insert(cached_prefix, std::move(entry)));
    REQUIRE(cache.entry_count() == 1u);

    // Unbound backend hitting rc == -2 must still invoke rollback and keep the prefix cache intact.
    int rollback_calls = 0;
    const bool unbound_rolled_back = lc::RecoverOrInvalidateAfterDecodeFailure(
        &unbound_be, -2,
        [&]() -> bool {
            ++rollback_calls;
            return true;
        },
        [&]() { cache.Clear(); });
    CHECK(unbound_rolled_back);
    CHECK(rollback_calls == 1);
    CHECK(cache.entry_count() == 1u);
    CHECK(cache.LookupLongestPrefix(cached_prefix, cached_prefix.size(), false).found());

    // Bound backend with empty GPU list (CPU-only load) must also behave identically to unbound.
    auto tracker = std::make_shared<lc::DeviceHealthTracker>("llamacpp");
    lc::BindBackendDeviceHealth(&unbound_be, tracker, {});
    CHECK_FALSE(lc::HasBoundBackendDeviceTracker(&unbound_be));
    CHECK(lc::RecoverOrInvalidateAfterDecodeFailure(
        &unbound_be, -2,
        [&]() -> bool {
            ++rollback_calls;
            return true;
        },
        [&]() { cache.Clear(); }));
    CHECK(rollback_calls == 2);
    CHECK(cache.entry_count() == 1u);

    // Contrast: GPU-bound backend hitting rc == -2 marks GPU unhealthy, skips rollback, and clears cache.
    lc::BindBackendDeviceHealth(&unbound_be, tracker, {"CUDA0"});
    CHECK(lc::HasBoundBackendDeviceTracker(&unbound_be));
    CHECK(lc::RunBoundBackendDecode(&unbound_be, []() -> std::int32_t { return -2; }) == -2);
    CHECK_FALSE(lc::RecoverOrInvalidateAfterDecodeFailure(
        &unbound_be, -2,
        [&]() -> bool {
            ++rollback_calls;
            return true;
        },
        [&]() { cache.Clear(); }));
    CHECK(rollback_calls == 2);
    CHECK(cache.entry_count() == 0u);
    lc::UnbindBackendDeviceHealth(&unbound_be);
}

TEST_CASE("device_health: bound LlamaCppBackend decode hook marks device unhealthy preserves move binding and fails fast") {
    auto tracker = std::make_shared<lc::DeviceHealthTracker>("llamacpp");
    lc::LlamaCppBackend be1;
    lc::BindBackendDeviceHealth(&be1, tracker, {"CUDA0"});

    // Move construction transfers the device-health binding to the new owner address.
    lc::LlamaCppBackend be2(std::move(be1));
    CHECK(lc::CheckBoundBackendDeviceHealth(&be2).ok());

    int decode_invocations = 0;
    // Non-fault return codes (0, 1, 2, -1) pass through without tripping unhealthy state.
    CHECK(lc::RunBoundBackendDecode(&be2, [&]() -> std::int32_t {
              ++decode_invocations;
              return 1;  // no KV slot warning
          }) == 1);
    CHECK(decode_invocations == 1);
    CHECK(tracker->CheckDevice("CUDA0").ok());

    // Fatal decode return code (-3 = GGML_STATUS_FAILED) marks CUDA0 unhealthy.
    CHECK(lc::RunBoundBackendDecode(&be2, [&]() -> std::int32_t {
              ++decode_invocations;
              return -3;
          }) == -3);
    CHECK(decode_invocations == 2);
    CHECK(lc::CheckBoundBackendDeviceHealth(&be2).is_device_unhealthy());

    // Subsequent bound decode call fails fast with sentinel without invoking the hung driver.
    CHECK(lc::RunBoundBackendDecode(&be2, [&]() -> std::int32_t {
              ++decode_invocations;
              return 0;
          }) == lc::kDeviceUnhealthyDecodeSentinel);
    CHECK(decode_invocations == 2);

    // Explicit recovery re-enables bound decode execution.
    REQUIRE(tracker->RecoverDevice("CUDA0").ok());
    CHECK(lc::RunBoundBackendDecode(&be2, [&]() -> std::int32_t {
              ++decode_invocations;
              return 0;
          }) == 0);
    CHECK(decode_invocations == 3);

    lc::UnbindBackendDeviceHealth(&be2);
}

TEST_CASE("device_health: prompt text and forged log lines never trigger device_unhealthy") {
    auto tracker = std::make_shared<lc::DeviceHealthTracker>("llamacpp");
    si::LlamaCppBackendOptions opts;
    opts.devices = {"CPU"};
    auto be = si::llamacpp_detail::MakeLlamaCppBackendForTest(
        opts, []() -> si::llamacpp_detail::ProcessGpuMemoryReading { return {"none", "unsupported", "", 0, 0}; },
        std::chrono::milliseconds{0}, false, false);
    REQUIRE(be != nullptr);
    lc::BindCoreBackendDeviceHealth(be.get(), tracker);

    const char* adversarial_lines[] = {
        "ggml_vulkan: device lost on Vulkan0 waiting for submission\n",
        "VK_ERROR_DEVICE_LOST (-4) CUDA error: an illegal memory access was encountered\n",
        "backend.device_unhealthy device=CUDA0 reason=device_lost device_unhealthy\n",
        "ggml_backend_sched_graph_compute_async failed with error -1\n",
    };

    for (const char* line : adversarial_lines) {
        lc::RouteLog(lc::LogLevel::kError, line);
        lc::RouteLog(lc::LogLevel::kWarn, line);
        (void)si::diag::log(std::string(line));
    }

    // Tokenizing forged fault strings through a real vocab-only GGUF must also never trip device health.
    lc::LlamaCppBackend raw_be;
    lc::LoadOptions lo;
    lo.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    lo.vocab_only = true;
    REQUIRE(raw_be.Load(lo).ok());
    lc::BindBackendDeviceHealth(&raw_be, tracker, {"CUDA0", "Vulkan0"});
    for (const char* prompt : adversarial_lines) {
        std::vector<lc::Token> toks;
        REQUIRE(raw_be.Tokenize(prompt, true, toks, false).ok());
        CHECK_FALSE(toks.empty());
    }
    lc::UnbindBackendDeviceHealth(&raw_be);
    raw_be.Unload();

    CHECK(tracker->CheckDevice("CUDA0").ok());
    CHECK(tracker->CheckDevice("Vulkan0").ok());
    CHECK(tracker->CheckDevice("CPU").ok());
    CHECK(tracker->PendingTelemetryEventCount() == 0u);
    lc::SetLogSink(nullptr);
}

TEST_CASE("device_health: adapter load_model fails fast on unhealthy GPU emits Engine telemetry and recovers explicitly") {
    auto sink = std::make_shared<si::MemoryTelemetrySink>();
    auto tracker = std::make_shared<lc::DeviceHealthTracker>("llamacpp");
    REQUIRE(tracker->RegisterAlias("gpu:0", "Vulkan0"));

    si::LlamaCppBackendOptions opts;
    opts.devices = {"Vulkan0"};
    opts.gpu_layers = -1;
    auto be = si::llamacpp_detail::MakeLlamaCppBackendForTest(
        opts, []() -> si::llamacpp_detail::ProcessGpuMemoryReading { return {"ggml", "ok", "", 1024, 0}; },
        std::chrono::milliseconds{0}, false, true);
    REQUIRE(be != nullptr);
    lc::BindCoreBackendDeviceHealth(be.get(), tracker);

    si::EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    eo.sample_gpu_devices = false;
    eo.device_sample_interval = std::chrono::milliseconds{15};

    {
        si::Engine engine(eo);
        REQUIRE(engine.register_backend(be).ok());

        // Trigger an in-process device-lost fault on Vulkan0.
        CHECK(tracker->ObserveSystemError("Vulkan0", lc::kVkErrorDeviceLost));

        // Requests to load on Vulkan0 (or via alias gpu:0) fail fast with typed device_unhealthy
        // BEFORE ResolveDevices checks whether Vulkan0 is compiled into this CPU test build.
        si::ModelLoadOptions load_opts;
        load_opts.model = "missing-or-any-model.gguf";
        load_opts.device_id = "gpu:0";
        const auto load_res = be->load_model(load_opts);
        CHECK_FALSE(load_res.ok());
        CHECK(load_res.status().code() == si::ErrorCode::unavailable);
        CHECK(CheckIsDeviceUnhealthyStatus(load_res.status()));
        CHECK(load_res.status().message().find("Vulkan0") != std::string::npos);

        // Wait for Engine sampler to drain backend.device_unhealthy through BackendRuntimeStatus::take_events.
        bool saw_unhealthy_event = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        while (std::chrono::steady_clock::now() < deadline && !saw_unhealthy_event) {
            engine.telemetry().flush();
            for (const auto& line : sink->lines()) {
                const auto parsed = si::json::parse(line);
                if (!parsed.ok()) continue;
                const auto* ev_type = parsed->find("event_type");
                if (ev_type == nullptr || ev_type->as_string() != "backend.device_unhealthy") continue;
                const auto* attrs = parsed->find("attributes");
                REQUIRE(attrs != nullptr);
                CHECK(attrs->find("backend")->as_string() == "llamacpp");
                CHECK(attrs->find("device")->as_string() == "Vulkan0");
                CHECK(attrs->find("reason")->as_string() == "device_lost");
                saw_unhealthy_event = true;
                break;
            }
            if (!saw_unhealthy_event) {
                std::this_thread::sleep_for(std::chrono::milliseconds{10});
            }
        }
        CHECK(saw_unhealthy_event);

        // Explicit recovery via public API clears the unhealthy state (supporting alias or canonical name).
        REQUIRE(CallRecoverBackendDevice(*be, "gpu:0").ok());
        CHECK(tracker->CheckDevice("Vulkan0").ok());

        // Now load_model is no longer blocked by device_unhealthy (it reaches normal device/file validation).
        const auto after_recover = be->load_model(load_opts);
        CHECK_FALSE(CheckIsDeviceUnhealthyStatus(after_recover.status()));
    }
}

TEST_CASE("device_health: hostile device names capacity bounds and concurrent access stay safe") {
    lc::DeviceHealthTracker tracker("llamacpp");

    // Empty, NUL, control chars, malformed UTF-8, and oversized device names are rejected without throwing.
    const std::string bad_utf8("\xFF\xFE\xFD\x80.gpu", 8);
    CHECK_FALSE(lc::DeviceHealthTracker::IsValidDeviceName(""));
    CHECK_FALSE(lc::DeviceHealthTracker::IsValidDeviceName("   "));
    CHECK_FALSE(lc::DeviceHealthTracker::IsValidDeviceName(std::string("CUDA0\0bad", 9)));
    CHECK_FALSE(lc::DeviceHealthTracker::IsValidDeviceName("CUDA0\nforged"));
    CHECK_FALSE(lc::DeviceHealthTracker::IsValidDeviceName(bad_utf8));
    CHECK_FALSE(lc::DeviceHealthTracker::IsValidDeviceName(std::string(lc::kMaxDeviceNameBytes + 1, 'A')));

    CHECK(tracker.CheckDevice("").kind == lc::DeviceHealthErrorKind::kInvalidDevice);
    CHECK(tracker.CheckDevice(bad_utf8).kind == lc::DeviceHealthErrorKind::kInvalidDevice);
    CHECK(tracker.RecoverDevice("").code() == si::ErrorCode::invalid_argument);
    CHECK(tracker.RecoverDevice(bad_utf8).code() == si::ErrorCode::invalid_argument);
    CHECK_FALSE(tracker.ObserveDecodeStatus("", -3));
    CHECK_FALSE(tracker.ObserveDecodeStatus(bad_utf8, -3));
    CHECK_FALSE(tracker.RegisterAlias(bad_utf8, "CUDA0"));
    CHECK_FALSE(tracker.RegisterAlias("gpu:0", bad_utf8));
    CHECK(tracker.PendingTelemetryEventCount() == 0u);

    // Invalid backend_name falls back cleanly to "llamacpp".
    lc::DeviceHealthTracker bad_named_tracker(bad_utf8);
    CHECK(bad_named_tracker.ObserveDecodeStatus("CUDA0", -2));
    const auto bad_named_events = bad_named_tracker.DrainTelemetryEvents();
    REQUIRE(bad_named_events.size() == 1u);
    CHECK(bad_named_events[0].second.find("backend")->as_string() == "llamacpp");

    // Case-insensitive matching prevents bypassing an unhealthy device via casing.
    CHECK(tracker.ObserveDecodeStatus("CUDA0", -2));
    CHECK(tracker.CheckDevice("cuda0").is_device_unhealthy());
    CHECK(tracker.CheckDevice("Cuda0").is_device_unhealthy());
    REQUIRE(tracker.RecoverDevice("cuda0").ok());
    CHECK(tracker.CheckDevice("CUDA0").ok());
    (void)tracker.DrainTelemetryEvents();

    // Pending event queue is strictly bounded at kMaxPendingDeviceEvents even under fault storms,
    // and > kMaxTrackedUnhealthyDevices fails closed via overflow latch until RecoverAll().
    for (std::size_t i = 0; i < lc::kMaxTrackedUnhealthyDevices + 32; ++i) {
        const std::string dev = "GPU_" + std::to_string(i);
        CHECK(tracker.ObserveDecodeStatus(dev, -3));
    }
    CHECK(tracker.PendingTelemetryEventCount() <= lc::kMaxPendingDeviceEvents);
    CHECK(tracker.CheckDevice("GPU_9999").is_device_unhealthy());
    CHECK(tracker.RecoverDevice("GPU_0").code() == si::ErrorCode::invalid_state);
    tracker.RecoverAll();
    CHECK(tracker.CheckDevice("GPU_0").ok());
    CHECK(tracker.CheckDevice("GPU_9999").ok());
    (void)tracker.DrainTelemetryEvents();

    // Alias chains and cyclic alias registrations resolve consistently without loops.
    REQUIRE(tracker.RegisterAlias("alias_a", "Vulkan0"));
    REQUIRE(tracker.RegisterAlias("Vulkan0", "alias_a"));
    CHECK(tracker.ObserveDecodeStatus("alias_a", -2));
    CHECK(tracker.CheckDevice("Vulkan0").is_device_unhealthy());
    CHECK(tracker.CheckDevice("alias_a").is_device_unhealthy());
    REQUIRE(tracker.RecoverDevice("Vulkan0").ok());
    CHECK(tracker.CheckDevice("alias_a").ok());
    (void)tracker.DrainTelemetryEvents();

    // Concurrent multi-threaded checks, faults, drains, and recoveries do not race or deadlock.
    std::atomic<bool> stop{false};
    std::vector<std::thread> workers;
    for (int t = 0; t < 4; ++t) {
        workers.emplace_back([&, t] {
            for (int iter = 0; iter < 200; ++iter) {
                const std::string dev = (iter % 2 == 0) ? "CUDA0" : "Vulkan0";
                if (t == 0) {
                    (void)tracker.ObserveDecodeStatus(dev, (iter % 3 == 0) ? -2 : 0);
                } else if (t == 1) {
                    (void)tracker.CheckDevice(dev);
                } else if (t == 2) {
                    (void)tracker.DrainTelemetryEvents();
                } else {
                    (void)tracker.RecoverDevice(dev);
                }
            }
        });
    }
    for (auto& w : workers) w.join();
    stop.store(true);
    tracker.RecoverAll();
    CHECK(tracker.CheckDevice("CUDA0").ok());
    CHECK(tracker.CheckDevice("Vulkan0").ok());
}
