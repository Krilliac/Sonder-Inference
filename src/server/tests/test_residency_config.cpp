#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "../src/residency_config.hpp"

using namespace sonder::inference;
namespace json = sonder::inference::json;

namespace {
constexpr std::uint64_t kMiB = 1024ull * 1024ull;
constexpr std::uint64_t kGiB = 1024ull * kMiB;

json::Value parse(const char *text) {
    const auto value = json::parse(text);
    REQUIRE(value.ok());
    return value.value();
}

GpuDeviceInfo make_rtx_gpu(const char *id = "gpu:0") {
    GpuDeviceInfo gpu;
    gpu.id = id;
    gpu.name = "NVIDIA GeForce RTX 5070 Ti";
    gpu.vendor = GpuVendor::nvidia;
    gpu.vendor_id = 0x10DEu;
    gpu.dedicated_total = MemoryReading::known(16303 * kMiB);
    gpu.dedicated_free = MemoryReading::known(15000 * kMiB);
    gpu.shared_total = MemoryReading::known(16 * kGiB);
    gpu.is_integrated_uma = false;
    gpu.backend_hint = "cuda";
    gpu.source = "dxgi";
    return gpu;
}

GpuDeviceInfo make_ryzen_780m_gpu(const char *id = "gpu:0") {
    GpuDeviceInfo gpu;
    gpu.id = id;
    gpu.name = "AMD Radeon 780M Graphics";
    gpu.vendor = GpuVendor::amd;
    gpu.vendor_id = 0x1002u;
    gpu.dedicated_total = MemoryReading::known(512 * kMiB);
    gpu.dedicated_free = MemoryReading::known(14 * kGiB);
    gpu.shared_total = MemoryReading::known(16 * kGiB);
    gpu.is_integrated_uma = true;
    gpu.backend_hint = "vulkan";
    gpu.source = "dxgi";
    return gpu;
}
}  // namespace

TEST_CASE("residency config parser accepts all fields") {
    BackendSetup setup;
    const auto value = parse(R"({"enabled":false,"expect_gpu":true,"min_dedicated_mib":1,
                                "consecutive_samples":1000,"eviction_fraction":0,
                                "eviction_mib":1,"on_eviction":"restart",
                                "max_eviction_restarts":32})");
    REQUIRE(detail::parse_residency_config(value, setup).ok());
    CHECK_FALSE(setup.llamaserver_residency.enabled);
    CHECK(setup.llamaserver_residency.expect_gpu);
    CHECK(setup.llamaserver_residency.min_dedicated_mib == 1);
    CHECK(setup.llamaserver_residency.consecutive_samples == 1000);
    CHECK(setup.llamaserver_residency.eviction_fraction == 0.0);
    CHECK(setup.llamaserver_residency.eviction_mib == 1);
    CHECK(setup.llamaserver_residency.on_eviction == "restart");
    CHECK(setup.llamaserver_residency.max_eviction_restarts == 32);
}

TEST_CASE("residency config parser rejects malformed and unsafe values") {
    const char *const invalid[] = {
        "[]", R"({"unknown":1})", R"({"enabled":1})", R"({"expect_gpu":"yes"})",
        R"({"min_dedicated_mib":0})", R"({"min_dedicated_mib":1048577})",
        R"({"consecutive_samples":0})", R"({"consecutive_samples":1001})",
        R"({"eviction_fraction":-0.1})", R"({"eviction_fraction":1.1})",
        R"({"eviction_mib":1048577})", R"({"on_eviction":"refuse"})",
        R"({"max_eviction_restarts":33})", R"({"eviction_fraction":0,"eviction_mib":0})"};
    for (const auto text : invalid) {
        BackendSetup setup;
        CHECK_MESSAGE(!detail::parse_residency_config(parse(text), setup).ok(), text);
    }
}

TEST_CASE("residency config unknown-field error bounds the echoed key") {
    BackendSetup setup;
    const auto value = parse((std::string("{\"") + std::string(990, 'k') + "\":1}").c_str());
    const auto st = detail::parse_residency_config(value, setup);
    REQUIRE_FALSE(st.ok());
    CHECK(st.message().size() < 1024);
    CHECK(st.message().find(std::string(64, 'k') + "...") != std::string::npos);
    CHECK(st.message().find(std::string(65, 'k')) == std::string::npos);
    const auto shortkey = detail::parse_residency_config(parse(R"({"zzz":1})"), setup);
    CHECK(shortkey.message().find("'zzz'") != std::string::npos);
    // A multi-byte character straddling the 64 byte limit is dropped whole.
    const auto utf8 = parse((std::string("{\"") + std::string(63, 'k') + "\xC3\xA9" + std::string(10, 'k') + "\":1}").c_str());
    const auto cut = detail::parse_residency_config(utf8, setup);
    CHECK(cut.message().find(std::string(63, 'k') + "...") != std::string::npos);
    CHECK(cut.message().find("\xC3") == std::string::npos);
}

#if defined(SONDER_HAS_LLAMASERVER_BACKEND)
TEST_CASE("residency config mapping converts MiB and preserves policy") {
    BackendSetup setup;
    auto &r = setup.llamaserver_residency;
    r.enabled = false;
    r.expect_gpu = true;
    r.min_dedicated_mib = 512;
    r.consecutive_samples = 7;
    r.eviction_fraction = 0.4;
    r.eviction_mib = 2048;
    r.on_eviction = "restart";
    r.max_eviction_restarts = 3;
    LlamaServerResidencyGuardOptions out;
    REQUIRE(detail::apply_residency_config(setup, out).ok());
    CHECK_FALSE(out.enabled);
    CHECK(out.expect_gpu);
    CHECK(out.min_dedicated_bytes == 512ull * 1024 * 1024);
    CHECK(out.consecutive_samples == 7);
    CHECK(out.eviction_fraction == doctest::Approx(0.4));
    CHECK(out.eviction_bytes == 2048ull * 1024 * 1024);
    CHECK(out.on_eviction == LlamaServerEvictionPolicy::restart);
    CHECK(out.max_eviction_restarts == 3);

    r.min_dedicated_mib = 1048577;
    CHECK_FALSE(detail::apply_residency_config(setup, out).ok());
    r.min_dedicated_mib = 512;
    r.eviction_fraction = 0;
    r.eviction_mib = 0;
    CHECK_FALSE(detail::apply_residency_config(setup, out).ok());
}

TEST_CASE("residency config mapping propagates spill_guard uma_mode and honors overrides") {
    const std::vector<GpuDeviceInfo> rtx_only{make_rtx_gpu("gpu:0")};
    const std::vector<GpuDeviceInfo> apu_only{make_ryzen_780m_gpu("gpu:0")};
    const std::vector<GpuDeviceInfo> mixed{make_rtx_gpu("gpu:0"), make_ryzen_780m_gpu("gpu:1")};
    const std::vector<GpuDeviceInfo> empty_gpus;

    // Default omitted uma_mode keeps false even on an APU-only host (backwards-compatible).
    {
        BackendSetup setup;
        LlamaServerResidencyGuardOptions out;
        REQUIRE(detail::apply_residency_config(setup, out, apu_only).ok());
        CHECK_FALSE(out.uma_mode);
    }

    // spill_guard.uma_mode = "on" propagates to residency when residency.uma_mode is omitted.
    {
        BackendSetup setup;
        setup.llamaserver_spill_uma_mode = "on";
        LlamaServerResidencyGuardOptions out;
        REQUIRE(detail::apply_residency_config(setup, out, rtx_only).ok());
        CHECK(out.uma_mode);
    }

    // Explicit residency.uma_mode = "off" overrides spill_guard.uma_mode = "on".
    {
        BackendSetup setup;
        setup.llamaserver_spill_uma_mode = "on";
        setup.llamaserver_residency.uma_mode = "off";
        LlamaServerResidencyGuardOptions out;
        REQUIRE(detail::apply_residency_config(setup, out, apu_only).ok());
        CHECK_FALSE(out.uma_mode);
    }

    // Explicit residency.uma_mode = "on" overrides spill_guard.uma_mode = "off".
    {
        BackendSetup setup;
        setup.llamaserver_spill_uma_mode = "off";
        setup.llamaserver_residency.uma_mode = "on";
        LlamaServerResidencyGuardOptions out;
        REQUIRE(detail::apply_residency_config(setup, out, rtx_only).ok());
        CHECK(out.uma_mode);
    }

    // spill_guard.uma_mode = "auto" resolves against injected GPUs and --device in args.
    {
        BackendSetup setup;
        setup.llamaserver_spill_uma_mode = "auto";
        LlamaServerResidencyGuardOptions out;

        REQUIRE(detail::apply_residency_config(setup, out, rtx_only).ok());
        CHECK_FALSE(out.uma_mode);

        REQUIRE(detail::apply_residency_config(setup, out, empty_gpus).ok());
        CHECK_FALSE(out.uma_mode);

        REQUIRE(detail::apply_residency_config(setup, out, apu_only).ok());
        CHECK(out.uma_mode);

        REQUIRE(detail::apply_residency_config(setup, out, mixed).ok());
        CHECK_FALSE(out.uma_mode);

        setup.llamaserver_args = {"--device", "Vulkan1"};
        REQUIRE(detail::apply_residency_config(setup, out, mixed).ok());
        CHECK(out.uma_mode);

        setup.llamaserver_args = {"-dev=CUDA0"};
        REQUIRE(detail::apply_residency_config(setup, out, mixed).ok());
        CHECK_FALSE(out.uma_mode);
    }

    // Invalid uma_mode strings on direct BackendSetup are rejected with invalid_argument.
    {
        BackendSetup bad_spill;
        bad_spill.llamaserver_spill_uma_mode = "bogus";
        LlamaServerResidencyGuardOptions out;
        const auto st = detail::apply_residency_config(bad_spill, out, apu_only);
        CHECK(st.code() == ErrorCode::invalid_argument);

        BackendSetup bad_residency;
        bad_residency.llamaserver_residency.uma_mode = "yes";
        const auto st2 = detail::apply_residency_config(bad_residency, out, apu_only);
        CHECK(st2.code() == ErrorCode::invalid_argument);
    }
}
#endif

TEST_CASE("residency config parser accepts uma_mode booleans and strings and rejects invalid values") {
    {
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"uma_mode":true})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode == "on");
    }
    {
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"uma_mode":false})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode == "off");
    }
    {
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"uma_mode":"auto"})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode == "auto");
    }
    {
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"uma_mode":"on"})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode == "on");
    }
    {
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"uma_mode":"off"})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode == "off");
    }
    {
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"uma_mode":"true"})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode == "on");
    }
    {
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"uma_mode":"false"})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode == "off");
    }
    {
        // Omitting uma_mode leaves the field empty so residency inherits spill_guard.uma_mode.
        BackendSetup setup;
        REQUIRE(detail::parse_residency_config(parse(R"({"enabled":true})"), setup).ok());
        CHECK(setup.llamaserver_residency.uma_mode.empty());
    }

    const char *const invalid[] = {
        R"({"uma_mode":""})",
        R"({"uma_mode":"yes"})",
        R"({"uma_mode":"no"})",
        R"({"uma_mode":"AUTO"})",
        R"({"uma_mode":"uma"})",
        R"({"uma_mode":1})",
        R"({"uma_mode":0})",
        R"({"uma_mode":null})",
        R"({"uma_mode":[]})",
        R"({"uma_mode":{}})"};
    for (const auto text : invalid) {
        BackendSetup setup;
        const auto st = detail::parse_residency_config(parse(text), setup);
        CHECK_MESSAGE(st.code() == ErrorCode::invalid_argument, text);
    }
}

TEST_CASE("llamaserver JSON config parses spill_guard uma_mode and residency override strictly") {
    const auto unique = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto path = std::filesystem::temp_directory_path() /
                      ("sonder-residency-uma-test-" + std::to_string(unique) + ".json");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    } cleanup{path};
    const auto write = [&](const std::string &body) {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << body;
    };

    // Omitted uma_mode defaults to "off" (backwards-compatible) with residency inheriting.
    write(R"({"mode":"spawn","executable":"llama-server","spill_guard":{"enabled":true}})");
    {
        BackendSetup setup;
        REQUIRE(load_llamaserver_config(path.string(), setup).ok());
        CHECK(setup.llamaserver_spill_uma_mode == "off");
        CHECK(setup.llamaserver_residency.uma_mode.empty());
    }

    // Boolean true / false and string "auto" at spill_guard level.
    write(R"({"mode":"spawn","executable":"llama-server","spill_guard":{"uma_mode":true}})");
    {
        BackendSetup setup;
        REQUIRE(load_llamaserver_config(path.string(), setup).ok());
        CHECK(setup.llamaserver_spill_uma_mode == "on");
        CHECK(setup.llamaserver_residency.uma_mode.empty());
    }

    write(R"({"mode":"spawn","executable":"llama-server","spill_guard":{"uma_mode":false}})");
    {
        BackendSetup setup;
        REQUIRE(load_llamaserver_config(path.string(), setup).ok());
        CHECK(setup.llamaserver_spill_uma_mode == "off");
    }

    write(R"({"mode":"spawn","executable":"llama-server","spill_guard":{"uma_mode":"auto","residency":{"uma_mode":false}}})");
    {
        BackendSetup setup;
        REQUIRE(load_llamaserver_config(path.string(), setup).ok());
        CHECK(setup.llamaserver_spill_uma_mode == "auto");
        CHECK(setup.llamaserver_residency.uma_mode == "off");
    }

    for (const auto *body : {
             R"({"spill_guard":{"uma_mode":""}})",
             R"({"spill_guard":{"uma_mode":"yes"}})",
             R"({"spill_guard":{"uma_mode":"AUTO"}})",
             R"({"spill_guard":{"uma_mode":1}})",
             R"({"spill_guard":{"uma_mode":null}})",
             R"({"spill_guard":{"residency":{"uma_mode":"bogus"}}})",
             R"({"spill_guard":{"residency":{"uma_mode":0}}})"}) {
        write(body);
        BackendSetup invalid;
        const auto st = load_llamaserver_config(path.string(), invalid);
        CHECK_MESSAGE(st.code() == ErrorCode::invalid_argument, body);
    }
}

TEST_CASE("detail resolve_uma_mode handles RTX, mixed RTX plus iGPU, Ryzen 780M APU, and empty GPU lists") {
    const std::vector<GpuDeviceInfo> rtx_only{make_rtx_gpu("gpu:0")};
    const std::vector<GpuDeviceInfo> apu_only{make_ryzen_780m_gpu("gpu:0")};
    const std::vector<GpuDeviceInfo> rtx_first{make_rtx_gpu("gpu:0"), make_ryzen_780m_gpu("gpu:1")};
    const std::vector<GpuDeviceInfo> igpu_first{make_ryzen_780m_gpu("gpu:0"), make_rtx_gpu("gpu:1")};
    const std::vector<GpuDeviceInfo> empty_gpus;

    // Explicit on/true and off/false ignore GPU inventory.
    CHECK(detail::resolve_uma_mode("on", rtx_only));
    CHECK(detail::resolve_uma_mode("true", empty_gpus));
    CHECK_FALSE(detail::resolve_uma_mode("off", apu_only));
    CHECK_FALSE(detail::resolve_uma_mode("false", apu_only));

    // Discrete RTX only -> false.
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_only));
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_only, "CUDA0"));
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_only, "Vulkan0"));

    // Mixed RTX + Ryzen iGPU: default hint -> false (discrete RTX wins regardless of order).
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_first));
    CHECK_FALSE(detail::resolve_uma_mode("auto", igpu_first));
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_first, "CUDA0"));
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_first, "Vulkan0"));
    CHECK_FALSE(detail::resolve_uma_mode("auto", igpu_first, "Vulkan1"));
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_first, "CUDA0,Vulkan1"));
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_first, "none"));
    CHECK_FALSE(detail::resolve_uma_mode("auto", rtx_first, "CPU"));

    // Mixed RTX + Ryzen iGPU: explicit Vulkan/AMD/id iGPU hint -> true.
    CHECK(detail::resolve_uma_mode("auto", rtx_first, "Vulkan1"));
    CHECK(detail::resolve_uma_mode("auto", rtx_first, "gpu:1"));
    CHECK(detail::resolve_uma_mode("auto", rtx_first, "ROCm0"));
    CHECK(detail::resolve_uma_mode("auto", rtx_first, "amd"));
    CHECK(detail::resolve_uma_mode("auto", igpu_first, "Vulkan0"));
    CHECK(detail::resolve_uma_mode("auto", igpu_first, "gpu:0"));

    // AMD Ryzen 780M APU-only -> true.
    CHECK(detail::resolve_uma_mode("auto", apu_only));
    CHECK(detail::resolve_uma_mode("auto", apu_only, "Vulkan0"));
    CHECK(detail::resolve_uma_mode("auto", apu_only, "ROCm0"));
    CHECK(detail::resolve_uma_mode("auto", apu_only, "gpu:0"));
    CHECK_FALSE(detail::resolve_uma_mode("auto", apu_only, "CUDA0"));

    // Empty GPU list (headless CI) -> false.
    CHECK_FALSE(detail::resolve_uma_mode("auto", empty_gpus));
    CHECK_FALSE(detail::resolve_uma_mode("auto", empty_gpus, "Vulkan0"));
}
