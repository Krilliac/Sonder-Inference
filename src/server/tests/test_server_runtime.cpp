// Backend runtime status (GPU memory, context fit, warnings) on the health,
// models and telemetry surfaces: present only when the backend reports it,
// and absent (responses unchanged) for every backend that does not.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"

using namespace server_test;
namespace json = sonder::inference::json;

namespace {

// The MOCK backend plus a fixed runtime status, standing in for a spawned
// llama-server (whose status comes from its supervisor).
class RuntimeBackend final : public si::Backend {
public:
    [[nodiscard]] std::string name() const override { return inner_->name(); }
    [[nodiscard]] std::string description() const override { return "mock with runtime status"; }
    [[nodiscard]] si::BackendCapabilities capabilities() const override { return inner_->capabilities(); }
    si::Result<std::string> probe() override { return inner_->probe(); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override { return inner_->list_models(); }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions& options) override {
        return inner_->load_model(options);
    }
    [[nodiscard]] std::optional<si::BackendRuntimeStatus> runtime_status() const override {
        std::lock_guard<std::mutex> lock(mu_);
        return status_;
    }
    void set(si::BackendRuntimeStatus status) {
        std::lock_guard<std::mutex> lock(mu_);
        status_ = std::move(status);
    }

private:
    std::shared_ptr<si::Backend> inner_ = si::make_mock_backend();
    mutable std::mutex mu_;
    si::BackendRuntimeStatus status_;
};

si::BackendRuntimeStatus spilled_status() {
    si::BackendRuntimeStatus s;
    s.gpu_memory.probe = "pdh";
    s.gpu_memory.status = "ok";
    s.gpu_memory.dedicated_bytes = 15266ull << 20;
    s.gpu_memory.shared_bytes = 388ull << 20;
    s.gpu_memory.peak_shared_bytes = 388ull << 20;
    s.gpu_memory.spill_threshold_bytes = 256ull << 20;
    s.gpu_memory.spilled = true;
    s.gpu_memory.samples = 1;
    s.context.policy = "warn";
    s.context.configured_ctx = 100096;
    s.context.fitted_ctx = 100096;
    s.context.outcome = "not_needed";
    s.warnings.push_back(si::BackendWarning{"kv_kernel_f16_fallback", "warning", "log", "converted to f16",
                                            {{"k_type", "q8_0"}, {"v_type", "q5_1"}}, 1});
    return s;
}

std::set<std::string> keys(const json::Value& v) {
    std::set<std::string> out;
    for (const auto& m : v.as_object()) out.insert(m.first);
    return out;
}

}  // namespace

TEST_CASE("runtime: health and models are unchanged for backends without runtime status") {
    Fixture f;  // MOCK backend
    const json::Value h = get(f.port, "/v1/sonder/health").json();
    const auto& backend = h.find("backends")->as_array().at(0);
    const std::set<std::string> known{"name", "available", "capabilities", "version"};
    for (const auto& k : keys(backend)) CHECK_MESSAGE(known.count(k) == 1, k);
    CHECK(backend.find("runtime") == nullptr);
    const json::Value m = get(f.port, "/v1/models").json();
    const auto& ext = *m.find("data")->as_array().at(0).find("sonder");
    CHECK(ext.dump() == R"({"backend":"mock","default":true,"synthetic":true})");
    CHECK(f.of_type("backend.gpu_memory.sample").empty());
    CHECK(f.of_type("backend.warning").empty());
}

TEST_CASE("runtime: health and models add the backend's runtime status") {
    auto backend = std::make_shared<RuntimeBackend>();
    backend->set(spilled_status());
    auto o = Fixture::defaults();
    o.backend.backend.clear();
    o.backend_instance = backend;
    Fixture f(o);

    const json::Value h = get(f.port, "/v1/sonder/health").json();
    const auto& entry = h.find("backends")->as_array().at(0);
    // Existing fields keep their values; runtime is the only addition.
    CHECK(entry.find("name")->as_string() == "mock");
    CHECK(entry.find("available")->is_bool());
    const json::Value* runtime = entry.find("runtime");
    REQUIRE(runtime != nullptr);
    CHECK(runtime->find("gpu_memory")->find("shared_bytes")->as_uint() == (388ull << 20));
    CHECK(runtime->find("gpu_memory")->find("dedicated_bytes")->as_uint() == (15266ull << 20));
    CHECK(runtime->find("gpu_memory")->find("spilled")->as_bool());
    CHECK(runtime->find("context")->find("fitted_ctx")->as_uint() == 100096u);
    const auto& warnings = runtime->find("warnings")->as_array();
    REQUIRE(warnings.size() == 1);
    CHECK(warnings[0].find("code")->as_string() == "kv_kernel_f16_fallback");
    CHECK(warnings[0].find("details")->find("v_type")->as_string() == "q5_1");

    const json::Value m = get(f.port, "/v1/models").json();
    const auto& ext = *m.find("data")->as_array().at(0).find("sonder");
    CHECK(ext.find("backend")->as_string() == "mock");
    CHECK(ext.find("default")->as_bool());
    CHECK(ext.find("synthetic")->as_bool());
    const json::Value* model_runtime = ext.find("runtime");
    REQUIRE(model_runtime != nullptr);
    CHECK(model_runtime->dump() == runtime->dump());
}

TEST_CASE("runtime: the engine sampler emits GPU samples and each warning once") {
    auto sink = std::make_shared<si::MemoryTelemetrySink>();
    si::EngineOptions eo;
    eo.telemetry_sinks.push_back(sink);
    eo.sample_devices_on_start = false;
    eo.device_sample_interval = std::chrono::milliseconds(20);
    auto backend = std::make_shared<RuntimeBackend>();
    backend->set(spilled_status());
    std::vector<json::Value> samples;
    std::vector<json::Value> warnings;
    {
        si::Engine engine(eo);
        REQUIRE(engine.register_backend(backend).ok());
        const auto collect = [&] {
            engine.telemetry().flush();
            samples.clear();
            warnings.clear();
            for (const auto& line : sink->lines()) {
                auto v = json::parse(line);
                if (!v.ok()) continue;
                const auto& type = v.value().find("event_type")->as_string();
                if (type == "backend.gpu_memory.sample") samples.push_back(v.value());
                if (type == "backend.warning") warnings.push_back(v.value());
            }
        };
        REQUIRE(eventually([&] {
            collect();
            return !samples.empty() && !warnings.empty();
        }));
        // Same sample count and warnings: nothing new is emitted.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        collect();
        CHECK(samples.size() == 1);
        CHECK(warnings.size() == 1);
        // A new sample and a new warning are emitted once each.
        auto next = spilled_status();
        next.gpu_memory.samples = 2;
        next.warnings.push_back(si::BackendWarning{"vram_spill", "warning", "gpu_probe", "spilled", {}, 1});
        backend->set(next);
        REQUIRE(eventually([&] {
            collect();
            return samples.size() == 2 && warnings.size() == 2;
        }));
    }
    const auto* attrs = samples.at(0).find("attributes");
    REQUIRE(attrs != nullptr);
    CHECK(attrs->find("backend")->as_string() == "mock");
    CHECK(attrs->find("shared_bytes")->as_uint() == (388ull << 20));
    CHECK(attrs->find("spilled")->as_bool());
    CHECK(attrs->find("fitted_ctx")->as_uint() == 100096u);
    CHECK(attrs->find("fit_outcome")->as_string() == "not_needed");
    const auto* w = warnings.at(1).find("attributes");
    REQUIRE(w != nullptr);
    CHECK(w->find("code")->as_string() == "vram_spill");
    CHECK(w->find("backend")->as_string() == "mock");
}

TEST_CASE("runtime: llamaserver config accepts spill_guard, log_file and kv_pairing_check strictly") {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("sonder-llamaserver-spill-test-" + std::to_string(unique) + ".json");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{path};
    const auto write = [&](const std::string& body) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << body;
    };
    // Absent keys keep the defaults (guard on, warn, 256 MiB, pairing check on).
    write(R"({"mode":"spawn","executable":"llama-server"})");
    si::BackendSetup defaults;
    REQUIRE(si::load_llamaserver_config(path.string(), defaults).ok());
    CHECK(defaults.llamaserver_spill_guard);
    CHECK(defaults.llamaserver_spill_policy == "warn");
    CHECK(defaults.llamaserver_spill_threshold_mib == 256);
    CHECK(defaults.llamaserver_log_file.empty());
    CHECK(defaults.llamaserver_kv_pairing_check);

    write(R"({"mode":"spawn","executable":"llama-server","args":["-c","100096"],"log_file":"ls.log",)"
          R"("kv_pairing_check":false,"spill_guard":{"enabled":true,"policy":"auto_fit","threshold_mib":300,)"
          R"("baseline_mib":10,"sample_interval_ms":2000,"fit_step_factor":0.8,"fit_step_align":512,)"
          R"("fit_min_ctx":16384,"fit_max_attempts":3}})");
    si::BackendSetup setup;
    auto status = si::load_llamaserver_config(path.string(), setup);
    REQUIRE_MESSAGE(status.ok(), status.to_string());
    CHECK(setup.llamaserver_spill_policy == "auto_fit");
    CHECK(setup.llamaserver_spill_threshold_mib == 300);
    CHECK(setup.llamaserver_spill_baseline_mib == 10);
    CHECK(setup.llamaserver_spill_sample_interval_ms == 2000);
    CHECK(setup.llamaserver_fit_step_factor == doctest::Approx(0.8));
    CHECK(setup.llamaserver_fit_step_align == 512);
    CHECK(setup.llamaserver_fit_min_ctx == 16384);
    CHECK(setup.llamaserver_fit_max_attempts == 3);
    CHECK(setup.llamaserver_log_file == "ls.log");
    CHECK_FALSE(setup.llamaserver_kv_pairing_check);

    for (const auto* body : {R"({"spill_guard":{"policy":"fit"}})", R"({"spill_guard":{"bogus":1}})",
                             R"({"spill_guard":[]})", R"({"spill_guard":{"threshold_mib":0}})",
                             R"({"spill_guard":{"fit_step_factor":1}})", R"({"spill_guard":{"fit_max_attempts":33}})",
                             R"({"spill_guard":{"sample_interval_ms":5}})", R"({"spill_guard":{"enabled":"yes"}})",
                             R"({"log_file":7})", R"({"kv_pairing_check":"on"})"}) {
        write(body);
        si::BackendSetup invalid;
        CHECK_MESSAGE(!si::load_llamaserver_config(path.string(), invalid).ok(), body);
    }

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
    // auto_fit without --ctx-size is rejected before anything is launched.
    si::BackendSetup no_ctx;
    no_ctx.backend = "llamaserver";
    no_ctx.llamaserver_mode = "spawn";
    no_ctx.llamaserver_executable = "llama-server-that-does-not-exist";
    no_ctx.llamaserver_spill_policy = "auto_fit";
    auto made = si::make_backend(no_ctx);
    REQUIRE(made.ok());
    CHECK(made.value()->list_models().status().code() == si::ErrorCode::invalid_argument);
    no_ctx.llamaserver_args = {"-c", "8192"};
    auto with_ctx = si::make_backend(no_ctx);
    REQUIRE(with_ctx.ok());
    const auto runtime = with_ctx.value()->runtime_status();
    REQUIRE(runtime.has_value());
    CHECK(runtime->context.policy == "auto_fit");
    CHECK(runtime->context.configured_ctx == 8192u);
#endif
}
