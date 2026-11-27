#include <atomic>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "doctest/doctest.h"
#include "sonder/inference/coopmat_canary.hpp"

using namespace sonder::inference;

namespace {

struct TempDir {
    std::filesystem::path path;
    explicit TempDir(const char* prefix) {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        path = std::filesystem::temp_directory_path() / (std::string(prefix) + "-" + std::to_string(stamp));
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
};

GpuDeviceInfo make_amd_gpu(const char* id, const char* name, std::uint32_t dev_id, bool uma,
                           const char* driver_build, const char* matrix_cores = "KHR_coopmat") {
    GpuDeviceInfo g;
    g.id = id;
    g.name = name;
    g.vendor = GpuVendor::amd;
    g.vendor_id = 0x1002u;
    g.device_id = dev_id;
    g.is_integrated_uma = uma;
    g.driver_name = "AMD proprietary driver";
    g.driver_build = driver_build;
    g.matrix_cores = matrix_cores;
    g.backend_hint = "vulkan";
    g.source = "dxgi";
    return g;
}

}  // namespace

TEST_CASE("NAT-16: eligibility covers AMD Windows discrete 9060 XT and integrated 780M only") {
    const auto rx9060 = make_amd_gpu("gpu:0", "AMD Radeon RX 9060 XT", 0x7590u, false, "26.11.2 (32.0.21045.29003)");
    const auto r780m = make_amd_gpu("gpu:1", "AMD Radeon 780M Graphics", 0x15BFu, true, "26.11.2 (32.0.21045.29003)");
    const auto no_cm = make_amd_gpu("gpu:2", "AMD Old GPU", 0x67DFu, false, "24.1.1", "none");
    const auto unknown_cm = make_amd_gpu("gpu:3", "AMD Radeon 780M Graphics", 0x15BFu, true, "26.11.2", "");

    CHECK(needs_amd_windows_coopmat_canary(rx9060, true));
    CHECK(needs_amd_windows_coopmat_canary(r780m, true));
    CHECK(needs_amd_windows_coopmat_canary(unknown_cm, true));
    CHECK_FALSE(needs_amd_windows_coopmat_canary(no_cm, true));
    CHECK_FALSE(needs_amd_windows_coopmat_canary(rx9060, false));
    CHECK_FALSE(needs_amd_windows_coopmat_canary(r780m, false));

    GpuDeviceInfo nv;
    nv.vendor = GpuVendor::nvidia;
    nv.vendor_id = 0x10DEu;
    nv.device_id = 0x2C05u;
    nv.matrix_cores = "NV_coopmat2";
    CHECK_FALSE(needs_amd_windows_coopmat_canary(nv, true));

    CHECK(std::string(to_string(CoopmatCanaryStatus::not_applicable)) == "not_applicable");
    CHECK(std::string(to_string(CoopmatCanaryStatus::skipped_env_disabled)) == "skipped_env_disabled");
    CHECK(std::string(to_string(CoopmatCanaryStatus::passed)) == "passed");
    CHECK(std::string(to_string(CoopmatCanaryStatus::failed)) == "failed");
    CHECK(std::string(to_string(CoopmatCanaryStatus::error)) == "error");
}

TEST_CASE("NAT-16: cache key is keyed on vendor, device, driver build, and ubatch") {
    CoopmatCanaryKey k1{0x1002u, 0x7590u, "25.10.2 (32.0.21025.1024)", 64u};
    CoopmatCanaryKey k1_dup{0x1002u, 0x7590u, "25.10.2 (32.0.21025.1024)", 64u};
    CoopmatCanaryKey k_new_drv{0x1002u, 0x7590u, "26.11.2 (32.0.21045.29003)", 64u};
    CoopmatCanaryKey k_780m{0x1002u, 0x15BFu, "25.10.2 (32.0.21025.1024)", 64u};
    CoopmatCanaryKey k_ub128{0x1002u, 0x7590u, "25.10.2 (32.0.21025.1024)", 128u};

    const std::string s1 = canonical_coopmat_cache_key(k1);
    REQUIRE_FALSE(s1.empty());
    CHECK(s1 == canonical_coopmat_cache_key(k1_dup));
    CHECK(s1 != canonical_coopmat_cache_key(k_new_drv));
    CHECK(s1 != canonical_coopmat_cache_key(k_780m));
    CHECK(s1 != canonical_coopmat_cache_key(k_ub128));
    CHECK(s1.find("1002") != std::string::npos);
    CHECK(s1.find("7590") != std::string::npos);

    // Control characters and whitespace are sanitized cleanly.
    CoopmatCanaryKey dirty{0x1002u, 0x7590u, "  25.10.2\r\n\x01  ", 64u};
    CHECK(canonical_coopmat_cache_key(dirty) == canonical_coopmat_cache_key({0x1002u, 0x7590u, "25.10.2", 64u}));
}

TEST_CASE("NAT-16: cache persistence round-trips and fails closed on corrupt or hostile inputs") {
    TempDir tmp("sonder-coopmat-cache");
    const std::string cache_file = (tmp.path / "canary.json").string();

    CoopmatCanaryRecord rec;
    rec.key = {0x1002u, 0x7590u, "26.11.2 (32.0.21045.29003)", 64u};
    rec.gpu_id = "gpu:0";
    rec.gpu_name = "AMD Radeon RX 9060 XT";
    rec.is_integrated_uma = false;
    rec.status = CoopmatCanaryStatus::passed;
    rec.disable_coopmat = false;
    rec.elapsed_ms = 42u;
    rec.exit_code = 0u;
    rec.reason = "canary_passed";

    REQUIRE(save_coopmat_canary_cache(cache_file, {rec}));
    const auto loaded = load_coopmat_canary_cache(cache_file);
    REQUIRE(loaded.size() == 1u);
    CHECK(loaded[0].key.vendor_id == 0x1002u);
    CHECK(loaded[0].key.device_id == 0x7590u);
    CHECK(loaded[0].key.driver_build == "26.11.2 (32.0.21045.29003)");
    CHECK(loaded[0].key.ubatch == 64u);
    CHECK(loaded[0].status == CoopmatCanaryStatus::passed);
    CHECK_FALSE(loaded[0].disable_coopmat);

    // Corrupt JSON fails closed to empty.
    {
        std::ofstream out(cache_file, std::ios::binary | std::ios::trunc);
        out << "{not valid json";
    }
    CHECK(load_coopmat_canary_cache(cache_file).empty());

    // Wrong schema fails closed to empty.
    {
        std::ofstream out(cache_file, std::ios::binary | std::ios::trunc);
        out << R"({"schema":"wrong/99","entries":[]})";
    }
    CHECK(load_coopmat_canary_cache(cache_file).empty());

    // Oversized file (> 128 KiB) fails closed to empty.
    {
        std::ofstream out(cache_file, std::ios::binary | std::ios::trunc);
        out << std::string(200000u, ' ');
    }
    CHECK(load_coopmat_canary_cache(cache_file).empty());

    // Path traversal is refused for both load and save.
    CHECK(load_coopmat_canary_cache((tmp.path / ".." / "escape.json").string()).empty());
    CHECK_FALSE(save_coopmat_canary_cache((tmp.path / ".." / "escape.json").string(), {rec}));
    CHECK_FALSE(save_coopmat_canary_cache("", {rec}));
}

TEST_CASE("NAT-16: evaluate_amd_coopmat_canary caches by driver build and triggers fallback on crash or timeout") {
    TempDir tmp("sonder-coopmat-eval");
    const std::string cache_file = (tmp.path / "canary.json").string();

    const auto rx9060 = make_amd_gpu("gpu:0", "AMD Radeon RX 9060 XT", 0x7590u, false, "25.10.2 (32.0.21025.1024)");
    const auto r780m = make_amd_gpu("gpu:1", "AMD Radeon 780M Graphics", 0x15BFu, true, "25.10.2 (32.0.21025.1024)");

    // 1. Both 9060 XT and 780M pass -> disable_coopmat=false, cached on disk.
    int calls = 0;
    CoopmatCanaryOptions opts;
    opts.is_windows = true;
    opts.existing_disable_coopmat_env = std::string{};
    opts.cache_path = cache_file;
    opts.use_default_cache_path = false;
    opts.apply_env_fallback = false;  // keep test process env clean
    opts.child_runner = [&](const CoopmatCanaryKey& key, std::chrono::milliseconds) {
        ++calls;
        CHECK(key.ubatch == 64u);
        return CoopmatCanaryChildOutcome{true, false, false, 0u, 15u, "ok"};
    };

    const auto d1 = evaluate_amd_coopmat_canary({rx9060, r780m}, opts);
    CHECK(d1.status == CoopmatCanaryStatus::passed);
    CHECK_FALSE(d1.disable_coopmat);
    CHECK(calls == 2);
    REQUIRE(d1.records.size() == 2u);
    CHECK_FALSE(d1.records[0].from_cache);
    CHECK_FALSE(d1.records[1].from_cache);

    // 2. Second run with same driver build hits cache (0 child spawns).
    const auto d2 = evaluate_amd_coopmat_canary({rx9060, r780m}, opts);
    CHECK(d2.status == CoopmatCanaryStatus::passed);
    CHECK_FALSE(d2.disable_coopmat);
    CHECK(calls == 2);
    REQUIRE(d2.records.size() == 2u);
    CHECK(d2.records[0].from_cache);
    CHECK(d2.records[1].from_cache);

    // 3. Driver build changes -> cache miss, 780M crashes (#22750) -> automatic fallback.
    auto rx9060_v2 = rx9060;
    rx9060_v2.driver_build = "26.11.2 (32.0.21045.29003)";
    auto r780m_v2 = r780m;
    r780m_v2.driver_build = "26.11.2 (32.0.21045.29003)";
    opts.child_runner = [&](const CoopmatCanaryKey& key, std::chrono::milliseconds) {
        ++calls;
        if (key.device_id == 0x15BFu) {
            return CoopmatCanaryChildOutcome{true, false, false, 0xC0000005u, 8u, "access violation"};
        }
        return CoopmatCanaryChildOutcome{true, false, false, 0u, 12u, "ok"};
    };
    const auto d3 = evaluate_amd_coopmat_canary({rx9060_v2, r780m_v2}, opts);
    CHECK(d3.status == CoopmatCanaryStatus::failed);
    CHECK(d3.disable_coopmat);
    CHECK(calls == 4);
    REQUIRE(d3.records.size() == 2u);
    CHECK(d3.records[0].status == CoopmatCanaryStatus::passed);
    CHECK(d3.records[1].status == CoopmatCanaryStatus::failed);
    CHECK(d3.records[1].exit_code == 0xC0000005u);
    CHECK(d3.records[1].disable_coopmat);

    // 4. Subsequent run on v2 hits cached failure immediately (0 new spawns).
    const auto d4 = evaluate_amd_coopmat_canary({rx9060_v2, r780m_v2}, opts);
    CHECK(d4.status == CoopmatCanaryStatus::failed);
    CHECK(d4.disable_coopmat);
    CHECK(calls == 4);
    CHECK(d4.records[1].from_cache);

    // 5. Timeout (#32447 hang) triggers fallback and is cached as failed.
    auto rx9060_v3 = rx9060;
    rx9060_v3.driver_build = "26.12.1";
    opts.child_runner = [&](const CoopmatCanaryKey&, std::chrono::milliseconds) {
        ++calls;
        return CoopmatCanaryChildOutcome{false, true, false, 0u, 15000u, "timed out"};
    };
    const auto d5 = evaluate_amd_coopmat_canary({rx9060_v3}, opts);
    CHECK(d5.status == CoopmatCanaryStatus::failed);
    CHECK(d5.disable_coopmat);
    CHECK(calls == 5);
    CHECK(d5.records[0].reason.find("timeout") != std::string::npos);

    // 6. Operator-set GGML_VK_DISABLE_COOPMAT=1 skips canary with 0 child spawns.
    auto env_opts = opts;
    env_opts.existing_disable_coopmat_env = std::string("1");
    const auto d6 = evaluate_amd_coopmat_canary({rx9060_v3}, env_opts);
    CHECK(d6.status == CoopmatCanaryStatus::skipped_env_disabled);
    CHECK(d6.disable_coopmat);
    CHECK(calls == 5);

    // 7. Transient spawn error fails safe (disable_coopmat=true) WITHOUT poisoning cache.
    auto rx9060_v4 = rx9060;
    rx9060_v4.driver_build = "27.1.0";
    opts.child_runner = [&](const CoopmatCanaryKey&, std::chrono::milliseconds) {
        ++calls;
        return CoopmatCanaryChildOutcome{false, false, true, 0u, 1u, "exe not found"};
    };
    const auto d7 = evaluate_amd_coopmat_canary({rx9060_v4}, opts);
    CHECK(d7.status == CoopmatCanaryStatus::error);
    CHECK(d7.disable_coopmat);
    CHECK(calls == 6);
    // Re-running with working runner succeeds because error was not cached.
    opts.child_runner = [&](const CoopmatCanaryKey&, std::chrono::milliseconds) {
        ++calls;
        return CoopmatCanaryChildOutcome{true, false, false, 0u, 10u, "ok"};
    };
    const auto d8 = evaluate_amd_coopmat_canary({rx9060_v4}, opts);
    CHECK(d8.status == CoopmatCanaryStatus::passed);
    CHECK_FALSE(d8.disable_coopmat);
    CHECK(calls == 7);

    // 8. Non-Windows host returns not_applicable with 0 spawns.
    auto linux_opts = opts;
    linux_opts.is_windows = false;
    const auto d9 = evaluate_amd_coopmat_canary({rx9060_v4}, linux_opts);
    CHECK(d9.status == CoopmatCanaryStatus::not_applicable);
    CHECK_FALSE(d9.disable_coopmat);
    CHECK(calls == 7);

    // 9. Concurrent cache reads/writes across 4 threads stay consistent.
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (int iter = 0; iter < 5; ++iter) {
                auto g = make_amd_gpu("gpu:0", "AMD Radeon RX 9060 XT", 0x7590u, false,
                                      ("28.0." + std::to_string(t)).c_str());
                CoopmatCanaryOptions topts = opts;
                topts.child_runner = [](const CoopmatCanaryKey&, std::chrono::milliseconds) {
                    return CoopmatCanaryChildOutcome{true, false, false, 0u, 5u, "ok"};
                };
                const auto res = evaluate_amd_coopmat_canary({g}, topts);
                if (res.status != CoopmatCanaryStatus::passed || res.disable_coopmat) {
                    failures.fetch_add(1);
                }
            }
        });
    }
    for (auto& th : threads) th.join();
    CHECK(failures.load() == 0);

    // 10. Process-wide decision recording round-trips cleanly.
    clear_coopmat_canary_decision();
    CHECK_FALSE(last_coopmat_canary_decision().has_value());
    record_coopmat_canary_decision(d3);
    REQUIRE(last_coopmat_canary_decision().has_value());
    CHECK(last_coopmat_canary_decision()->disable_coopmat);
    clear_coopmat_canary_decision();
    CHECK_FALSE(last_coopmat_canary_decision().has_value());
}
