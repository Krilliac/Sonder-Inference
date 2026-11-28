// VRAM-spill guard, KV cache pairing check and llama-server log diagnostics
// (docs/integration/vram-spill.md). GPU counters come from a fake source and
// children from the fake launcher; nothing here needs a GPU or llama-server.
#include "../spill_guard.hpp"
#include "../llamaserver_options.hpp"
#include "../log_diagnostics.hpp"
#include "../supervisor.hpp"
#include "fake_process.hpp"
#include "fake_gpu_memory.hpp"
#include "sonder/inference/backends/llamaserver.hpp"

#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
using namespace sonder_test;

namespace {

constexpr std::uint64_t MiB = 1024ull * 1024ull;

// Exact lines from the 2026-09-29 llama-server logs (Ollama's bundled
// llama-server 0.4.1-dev, Qwen3.8-27B, --cache-type-k q8_0 --cache-type-v q5_1).
const char *const kFaLine =
    "1.29.621.363 W ggml_cuda_flash_attn_ext_vec: no FlashAttention vector kernel compiled for K/V types q8_0-q5_1, "
    "converting K and V to f16 instead (slow). Add \"q8_0-q5_1\" to GGML_CUDA_FA_QUANTS to compile it.";
const char *const kNextnLines[] = {
    "0.01.674.095 W model has unused tensor blk.64.nextn.eh_proj.weight (size = 43008000 bytes) -- ignoring",
    "0.01.674.104 W model has unused tensor blk.64.nextn.enorm.weight (size = 20480 bytes) -- ignoring",
    "0.01.674.112 W model has unused tensor blk.64.nextn.hnorm.weight (size = 20480 bytes) -- ignoring",
    "0.01.674.133 W model has unused tensor blk.64.nextn.shared_head_norm.weight (size = 20480 bytes) -- ignoring",
};
const char *const kQuietLines[] = {
    "0.01.674.000 W model has unused tensor blk.64.attn_norm.weight (size = 20480 bytes) -- ignoring",
    ("0.01.204.049 W srv  llama_server: security: no API key is set and CORS allows all origins (see "
     "https://github.com/ggml-org/llama.cpp/pull/25655)"),
    "1.34.364.127 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 81920, kv_unified = 'false'",
};

std::string instance(std::uint32_t pid, const char *luid = "0x000104A4") {
    return "pid_" + std::to_string(pid) + "_luid_0x00000000_" + luid + "_phys_0";
}

GpuProcessCounters counters(std::uint32_t pid, std::uint64_t dedicated, std::uint64_t shared) {
    GpuProcessCounters c;
    c.dedicated = {{instance(pid), dedicated}, {instance(pid, "0x000118C2"), 0}};
    c.shared = {{instance(pid), shared}, {instance(pid, "0x000118C2"), 0}};
    return c;
}

// Shared usage as a function of the launched child's --ctx-size: children
// above `limit` spill (388 MiB, measured), the others stay clean (182 MiB).
std::shared_ptr<FakeCounters> ctx_dependent_counters(std::shared_ptr<LaunchState> state, std::uint64_t limit) {
    return std::make_shared<FakeCounters>([state, limit](std::uint32_t pid) -> Result<GpuProcessCounters> {
        // Runs on the monitor thread: report problems as errors, not asserts.
        std::lock_guard lock(state->mutex);
        const auto index = static_cast<std::size_t>(pid - 1000);
        if (pid < 1000 || index >= state->specs.size())
            return Status(ErrorCode::internal, "unknown fake pid");
        const auto ctx = find_context_size(state->specs[index].arguments);
        if (!ctx)
            return Status(ErrorCode::internal, "fake child has no --ctx-size");
        return counters(pid, 15266 * MiB, *ctx > limit ? 388 * MiB : 182 * MiB);
    });
}

SupervisorOptions guarded_options(std::shared_ptr<GpuCounterSource> source, SpillPolicy policy) {
    auto o = base_options();
    o.arguments = {"--model", "model.gguf", "--ctx-size", "100096", "--flash-attn", "on"};
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    o.gpu_counters = std::move(source);
    o.spill_guard.policy = policy;
    return o;
}

std::shared_ptr<LaunchState> pid_state() {
    auto state = std::make_shared<LaunchState>();
    state->next_pid = 1000;
    return state;
}

std::vector<std::uint64_t> launched_contexts(const std::shared_ptr<LaunchState> &state) {
    std::lock_guard lock(state->mutex);
    std::vector<std::uint64_t> out;
    for (const auto &spec : state->specs)
        out.push_back(find_context_size(spec.arguments).value_or(0));
    return out;
}

const BackendWarning *find_warning(const BackendRuntimeStatus &status, const std::string &code) {
    for (const auto &w : status.warnings) {
        if (w.code == code)
            return &w;
    }
    return nullptr;
}

std::string detail_of(const BackendWarning &w, const std::string &key) {
    for (const auto &[k, v] : w.details) {
        if (k == key)
            return v;
    }
    return {};
}

std::filesystem::path temp_log(const char *tag) {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    return std::filesystem::temp_directory_path() /
           ("sonder-llamaserver-" + std::string(tag) + "-" + std::to_string(stamp) + ".log");
}

std::string fixture_text(const char *name) {
    const auto relative = std::filesystem::path("src/backends/llamaserver/tests/fixtures") / name;
    std::vector<std::filesystem::path> candidates{
        std::filesystem::path(__FILE__).parent_path() / "fixtures" / name,
        relative,
    };
    auto root = std::filesystem::current_path();
    for (int i = 0; i < 6; ++i) {
        candidates.push_back(root / relative);
        if (root == root.root_path())
            break;
        root = root.parent_path();
    }
    for (const auto &path : candidates) {
        std::ifstream in(path, std::ios::binary);
        if (!in)
            continue;
        std::ostringstream text;
        text << in.rdbuf();
        return text.str();
    }
    REQUIRE_MESSAGE(false, "missing fixture: ", relative.string());
    return {};
}

} // namespace

// ---------------------------------------------------------------- GPU probe

TEST_CASE("gpu probe sums only the child's instances across adapters") {
    GpuProcessCounters c;
    c.dedicated = {{"pid_1234_luid_0x00000000_0x000104A4_phys_0", 15100 * MiB},
                   {"pid_1234_luid_0x00000000_0x000118C2_phys_0", 3 * MiB},
                   {"pid_12345_luid_0x00000000_0x000104A4_phys_0", 999 * MiB},
                   {"pid_123_luid_0x00000000_0x000104A4_phys_0", 777 * MiB},
                   {"pid_1234", 555 * MiB}};
    c.shared = {{"pid_1234_luid_0x00000000_0x000104A4_phys_0", 950 * MiB},
                {"pid_1234_luid_0x00000000_0x000118C2_phys_0", 22 * MiB},
                {"pid_12340_luid_0x00000000_0x000104A4_phys_0", 4096 * MiB}};
    const auto s = summarize_gpu_counters(1234, c);
    CHECK(s.dedicated_bytes == 15103 * MiB);
    CHECK(s.shared_bytes == 972 * MiB);
    CHECK(s.instances == 2);
    CHECK(is_process_instance("pid_7_luid_x", 7));
    CHECK_FALSE(is_process_instance("pid_7", 7));
    CHECK_FALSE(is_process_instance("pid_70_luid_x", 7));
    CHECK_FALSE(is_process_instance("xpid_7_luid", 7));
    CHECK(summarize_gpu_counters(99, c).instances == 0);
}

TEST_CASE("gpu probe saturates instead of wrapping") {
    GpuProcessCounters c;
    const auto max = std::numeric_limits<std::uint64_t>::max();
    c.shared = {{instance(5), max - 1}, {instance(5, "0x1"), 10}};
    CHECK(summarize_gpu_counters(5, c).shared_bytes == max);
}

TEST_CASE("platform gpu counter source reports support honestly") {
    auto source = make_gpu_counter_source();
    REQUIRE(source);
#if defined(_WIN32)
    CHECK(source->name() == "pdh");
    CHECK(source->supported());
    // Pid 0 is never sampled; a live process without a GPU context is an
    // empty sample or a clean error, never a crash.
    CHECK(source->read(0).status().code() == ErrorCode::invalid_argument);
    const auto pid = static_cast<std::uint32_t>(_getpid());
    auto self = source->read(pid);
    REQUIRE_MESSAGE(self.ok(), self.status().to_string());
    const auto sample = summarize_gpu_counters(pid, self.value());
    CHECK(sample.instances <= 64);
#elif defined(__linux__)
    if (source->supported()) {
        const auto name = source->name();
        CHECK((name == "amdgpu-fdinfo" || name == "nvml" || name == "amdgpu-fdinfo+nvml"));
        CHECK(source->read(0).status().code() == ErrorCode::invalid_argument);
        const auto pid = static_cast<std::uint32_t>(getpid());
        auto self = source->read(pid);
        if (self.ok()) {
            const auto sample = summarize_gpu_counters(pid, self.value());
            CHECK(sample.instances <= 64);
        }
    } else {
        CHECK(source->name() == "none");
        CHECK(source->read(1).status().code() == ErrorCode::unsupported);
    }
#else
    CHECK(source->name() == "none");
    CHECK_FALSE(source->supported());
    CHECK(source->read(1).status().code() == ErrorCode::unsupported);
#endif
}

// ----------------------------------------------------------- classification

TEST_CASE("spill classification uses the measured clean and spilled samples") {
    SpillGuardOptions o; // defaults: 256 MiB above a 0 baseline
    CHECK(o.threshold_bytes == 256 * MiB);
    const auto spilled = [&](std::uint64_t shared_mib) {
        GpuMemorySample s;
        s.shared_bytes = shared_mib * MiB;
        return is_spilled(s, o);
    };
    // Clean Q3_K_XL runs (49k-100k ctx): 148-198 MiB shared.
    for (const std::uint64_t clean : {148u, 170u, 182u, 190u, 198u})
        CHECK_FALSE(spilled(clean));
    // Spilled Q3_K_XL runs: 292-388 MiB; IQ4_XS: 0.71, 0.95 and 2.1 GB.
    for (const std::uint64_t bad : {292u, 342u, 348u, 388u, 677u, 906u, 2003u})
        CHECK(spilled(bad));
    CHECK_FALSE(spilled(256)); // strictly above the threshold
    CHECK(spilled(257));
    o.baseline_bytes = 200 * MiB;
    CHECK_FALSE(spilled(456));
    CHECK(spilled(457));
    o.baseline_bytes = std::numeric_limits<std::uint64_t>::max();
    CHECK_FALSE(spilled(1u << 20)); // saturating limit
}

TEST_CASE("spill baseline can grow with context for MTP profiles") {
    // MTP (--spec-type draft-mtp) raises the clean line: measured 2026-09-30
    // on the RTX 5070 Ti at about 126 MiB + 2 MiB per 1k ctx, +40 MiB after
    // the first prompt (about 310 MiB at 73,728), and runs 33+ MiB above that
    // line were already slower. A fixed 0 + 256 MiB limit calls that clean run
    // spilled, which under auto_fit would shrink the context for nothing.
    GpuMemorySample s;
    s.shared_bytes = 310 * MiB;
    SpillGuardOptions fixed; // defaults
    CHECK(is_spilled(s, fixed, 73728u));
    CHECK(effective_baseline(fixed, 73728u) == 0);

    SpillGuardOptions o;
    o.baseline_bytes = 166 * MiB;           // 126 + 40
    o.baseline_bytes_per_1k_ctx = 2 * MiB;  // + 2 MiB per 1,024 tokens
    o.threshold_bytes = 32 * MiB;
    CHECK(effective_baseline(o, 73728u) == (166 + 144) * MiB);
    CHECK_FALSE(is_spilled(s, o, 73728u));  // clean MTP line
    s.shared_bytes = 342 * MiB;
    CHECK_FALSE(is_spilled(s, o, 73728u));  // exactly baseline + threshold
    s.shared_bytes = 343 * MiB;
    CHECK(is_spilled(s, o, 73728u));        // mild spill
    // Unknown context: the fixed baseline applies (no growth term).
    CHECK(effective_baseline(o, std::nullopt) == 166 * MiB);
    CHECK(is_spilled(s, o));
    // The growth term saturates instead of wrapping.
    o.baseline_bytes_per_1k_ctx = 1024 * MiB;
    CHECK(effective_baseline(o, std::numeric_limits<std::uint64_t>::max()) ==
          std::numeric_limits<std::uint64_t>::max());
    CHECK(validate_spill_guard(o).ok());
    o.baseline_bytes_per_1k_ctx = 1024 * MiB + 1;
    CHECK_FALSE(validate_spill_guard(o).ok());
}

TEST_CASE("auto_fit context steps are aligned, strictly decreasing and floored") {
    SpillGuardOptions o;
    CHECK(next_fit_context(100096, o) == 84992u);
    CHECK(next_fit_context(84992, o) == 71680u);
    CHECK(next_fit_context(71680, o) == 60416u);
    // Always at least one alignment step, even with a factor close to 1.
    o.step_factor = 0.999;
    CHECK(next_fit_context(65536, o) == 64512u);
    o = SpillGuardOptions{};
    o.min_ctx = 65536;
    CHECK(next_fit_context(71680, o) == 65536u); // clamped to the floor
    CHECK_FALSE(next_fit_context(65536, o).has_value());
    CHECK_FALSE(next_fit_context(4096, o).has_value());
    // Termination from any start: repeated steps reach the floor quickly.
    o = SpillGuardOptions{};
    std::uint64_t ctx = 1u << 30;
    int steps = 0;
    while (auto next = next_fit_context(ctx, o)) {
        REQUIRE(*next < ctx);
        ctx = *next;
        REQUIRE(++steps < 200);
    }
    CHECK(ctx == o.min_ctx);
}

TEST_CASE("spill guard options are validated") {
    CHECK(validate_spill_guard({}).ok());
    const auto bad = [](auto mutate) {
        SpillGuardOptions o;
        mutate(o);
        return validate_spill_guard(o).code() == ErrorCode::invalid_argument;
    };
    CHECK(bad([](SpillGuardOptions &o) { o.threshold_bytes = 0; }));
    CHECK(bad([](SpillGuardOptions &o) { o.step_factor = 1.0; }));
    CHECK(bad([](SpillGuardOptions &o) { o.step_factor = 0.0; }));
    CHECK(bad([](SpillGuardOptions &o) { o.step_factor = std::numeric_limits<double>::quiet_NaN(); }));
    CHECK(bad([](SpillGuardOptions &o) { o.step_align = 0; }));
    CHECK(bad([](SpillGuardOptions &o) { o.min_ctx = 0; }));
    CHECK(bad([](SpillGuardOptions &o) { o.max_attempts = 0; }));
    CHECK(bad([](SpillGuardOptions &o) { o.max_attempts = 33; }));
    CHECK(bad([](SpillGuardOptions &o) { o.sample_interval = std::chrono::milliseconds(10); }));
    CHECK(bad([](SpillGuardOptions &o) { o.policy = static_cast<SpillPolicy>(7); }));
    CHECK(parse_spill_policy("auto_fit") == SpillPolicy::auto_fit);
    CHECK_FALSE(parse_spill_policy("fit").has_value());
    CHECK(std::string(to_string(SpillPolicy::refuse)) == "refuse");
}

// -------------------------------------------------------------- argv reading

TEST_CASE("context size and log file are read from every llama-server spelling") {
    CHECK_FALSE(find_context_size({"--model", "m.gguf"}).has_value());
    CHECK(find_context_size({"-c", "4096"}) == 4096u);
    CHECK(find_context_size({"--ctx-size", "8192", "-c", "65536"}) == 65536u);
    CHECK(find_context_size({"--ctx-size=32768"}) == 32768u);
    CHECK_FALSE(find_context_size({"--ctx-size", "0"}).has_value());
    CHECK_FALSE(find_context_size({"--ctx-size", "12k"}).has_value());
    CHECK_FALSE(find_context_size({"--ctx-size"}).has_value());
    CHECK_FALSE(find_context_size({"-c=4096"}).has_value()); // short options take a separate value

    const std::vector<std::string> base{"--ctx-size", "100096", "--jinja", "-c", "9", "--ctx-size=5"};
    const auto rewritten = with_context_size(base, 65536);
    CHECK(rewritten == std::vector<std::string>{"--ctx-size", "65536", "--jinja", "-c", "65536", "--ctx-size=65536"});
    CHECK(with_context_size({"--jinja"}, 4096) == std::vector<std::string>{"--jinja", "--ctx-size", "4096"});

    CHECK(find_log_file({"--log-file", "a.log"}) == std::string("a.log"));
    CHECK(find_log_file({"--log-file=b.log"}) == std::string("b.log"));
    CHECK_FALSE(find_log_file({"--log-file="}).has_value());
    CHECK_FALSE(find_log_file({"--log-disable"}).has_value());
}

TEST_CASE("kv pairing check: the benchmark baseline flags warn, matched pairs do not") {
    // The exact cache flags of the 2026-09-29 baseline.
    const std::vector<std::string> baseline{"--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q5_1"};
    auto w = check_kv_cache_pairing(baseline);
    REQUIRE(w.size() == 1);
    CHECK(w[0].code == "kv_type_mismatch");
    CHECK(w[0].severity == "warning");
    CHECK(w[0].source == "config");
    CHECK(detail_of(w[0], "cache_type_k") == "q8_0");
    CHECK(detail_of(w[0], "cache_type_v") == "q5_1");
    CHECK(w[0].message.find("--cache-type-k q8_0 --cache-type-v q8_0") != std::string::npos);

    CHECK(check_kv_cache_pairing({"-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"}).empty());
    CHECK(check_kv_cache_pairing({}).empty()); // f16/f16 defaults
    CHECK(check_kv_cache_pairing({"--flash-attn", "off", "-ctk", "q8_0", "-ctv", "q5_1"}).empty());
    CHECK(check_kv_cache_pairing({"--flash-attn=off", "-ctk", "q8_0", "-ctv", "q4_0"}).empty());
    // Bare -fa (older llama.cpp) means on; flash_attn auto still converts.
    CHECK(check_kv_cache_pairing({"-fa", "-ctk", "q8_0", "-ctv", "q4_0"}).size() == 1);
    CHECK(check_kv_cache_pairing({"-ctk", "Q8_0", "-ctv", "q4_0"}).size() == 1);

    auto q51 = check_kv_cache_pairing({"-fa", "on", "-ctk", "q5_1", "-ctv", "q5_1"});
    REQUIRE(q51.size() == 1);
    CHECK(q51[0].code == "kv_type_no_vector_kernel");
    CHECK(q51[0].message.find("GGML_CUDA_FA_ALL_QUANTS") != std::string::npos);
    CHECK(q51[0].message.find("GGML_CUDA_FA_QUANTS") == std::string::npos);
    auto other = check_kv_cache_pairing({"-fa", "on", "-ctk", "iq4_nl", "-ctv", "iq4_nl"});
    REQUIRE(other.size() == 1);
    CHECK(other[0].code == "kv_type_kernel_unknown");
    CHECK(other[0].severity == "info");
}

TEST_CASE("kv pairing check tailors wording by backend hint and --device argument") {
    const std::vector<std::string> mixed{"--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q5_1"};
    const std::vector<std::string> q51{"--flash-attn", "on", "--cache-type-k", "q5_1", "--cache-type-v", "q5_1"};
    const std::vector<std::string> unknown{"--flash-attn", "on", "--cache-type-k", "iq4_nl", "--cache-type-v", "iq4_nl"};

    for (const auto &args : {mixed, q51, unknown}) {
        const auto vk = check_kv_cache_pairing(args, "vulkan");
        REQUIRE(vk.size() == 1);
        CHECK(detail_of(vk[0], "backend") == "vulkan");
        CHECK(vk[0].message.find("Vulkan") != std::string::npos);
        CHECK(vk[0].message.find("CUDA") == std::string::npos);
        CHECK(vk[0].message.find("GGML_CUDA_FA_ALL_QUANTS") == std::string::npos);

        for (const std::string_view hint : {"rocm", "hip"}) {
            const auto rocm = check_kv_cache_pairing(args, hint);
            REQUIRE(rocm.size() == 1);
            CHECK(detail_of(rocm[0], "backend") == "rocm");
            CHECK(rocm[0].message.find("ROCm/HIP") != std::string::npos);
            CHECK(rocm[0].message.find("CUDA") == std::string::npos);
            CHECK(rocm[0].message.find("GGML_CUDA_FA_ALL_QUANTS") == std::string::npos);
        }
    }

    const auto cuda_q51 = check_kv_cache_pairing(q51, "cuda");
    REQUIRE(cuda_q51.size() == 1);
    CHECK(detail_of(cuda_q51[0], "backend") == "cuda");
    CHECK(cuda_q51[0].message.find("CUDA") != std::string::npos);
    CHECK(cuda_q51[0].message.find("GGML_CUDA_FA_ALL_QUANTS") != std::string::npos);

    // Infer backend from --device / -dev when no explicit hint is passed.
    const auto vk_dev = check_kv_cache_pairing({"--device", "Vulkan0", "-fa", "on", "-ctk", "q8_0", "-ctv", "q4_0"});
    REQUIRE(vk_dev.size() == 1);
    CHECK(detail_of(vk_dev[0], "backend") == "vulkan");
    CHECK(vk_dev[0].message.find("Vulkan") != std::string::npos);
    CHECK(vk_dev[0].message.find("CUDA") == std::string::npos);

    const auto rocm_dev = check_kv_cache_pairing({"--device=ROCm0", "-fa", "on", "-ctk", "q5_1", "-ctv", "q5_1"});
    REQUIRE(rocm_dev.size() == 1);
    CHECK(detail_of(rocm_dev[0], "backend") == "rocm");
    CHECK(rocm_dev[0].message.find("ROCm/HIP") != std::string::npos);
    CHECK(rocm_dev[0].message.find("GGML_CUDA_FA_ALL_QUANTS") == std::string::npos);

    const auto hip_dev = check_kv_cache_pairing({"-dev", "HIP0", "-fa", "on", "-ctk", "q8_0", "-ctv", "q5_1"});
    REQUIRE(hip_dev.size() == 1);
    CHECK(detail_of(hip_dev[0], "backend") == "rocm");

    const auto cuda_dev = check_kv_cache_pairing({"--device", "CUDA0", "-fa", "on", "-ctk", "q5_1", "-ctv", "q5_1"});
    REQUIRE(cuda_dev.size() == 1);
    CHECK(detail_of(cuda_dev[0], "backend") == "cuda");
    CHECK(cuda_dev[0].message.find("GGML_CUDA_FA_ALL_QUANTS") != std::string::npos);

    CHECK(check_kv_cache_pairing({"--device", "Vulkan0", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0"}).empty());
    CHECK(check_kv_cache_pairing({"--device", "ROCm0", "-fa", "off", "-ctk", "q8_0", "-ctv", "q5_1"}).empty());
}

// --------------------------------------------------------------- log parser

TEST_CASE("log parser extracts the FlashAttention f16 conversion from the real log line") {
    LogDiagnostics d;
    d.feed(std::string(kFaLine) + "\n");
    REQUIRE(d.warnings().size() == 1);
    const auto &w = d.warnings()[0];
    CHECK(w.code == "kv_kernel_f16_fallback");
    CHECK(w.source == "log");
    CHECK(w.message.find("GGML_CUDA_FA_ALL_QUANTS") != std::string::npos);
    CHECK(w.message.find("GGML_CUDA_FA_QUANTS") == std::string::npos);
    CHECK(detail_of(w, "k_type") == "q8_0");
    CHECK(detail_of(w, "v_type") == "q5_1");
    d.feed(std::string(kFaLine) + "\r\n"); // repeated per context: folded
    REQUIRE(d.warnings().size() == 1);
    CHECK(d.warnings()[0].count == 2);
}

TEST_CASE("log parser folds ignored nextn (MTP) tensors and skips other lines") {
    LogDiagnostics d;
    for (const auto *line : kQuietLines)
        d.feed(std::string(line) + "\n");
    CHECK(d.warnings().empty());
    for (const auto *line : kNextnLines)
        d.feed(std::string(line) + "\n");
    REQUIRE(d.warnings().size() == 1);
    const auto &w = d.warnings()[0];
    CHECK(w.code == "mtp_tensors_ignored");
    CHECK(w.count == 4);
    CHECK(detail_of(w, "layer") == "blk.64");
    CHECK(detail_of(w, "tensors") == "4");
    CHECK(detail_of(w, "ignored_bytes") == std::to_string(43008000ull + 3 * 20480ull));
    CHECK(d.lines() == 7);
}

TEST_CASE("log parser reports CPU fallback and offload messages") {
    // llama.cpp source strings (not from the 2026-09-29 runs, which offloaded
    // every layer).
    LogDiagnostics d;
    d.feed("load_tensors: offloaded 65/65 layers to GPU\n");
    CHECK(d.warnings().empty());
    d.feed("load_tensors: offloaded 40/65 layers to GPU\n");
    d.feed("tensor blk.0.ffn_down.weight (q4_K) cannot be used with preferred buffer type CUDA_Host, using CPU "
           "instead\n");
    d.feed("tensor blk.1.ffn_down.weight (q4_K) cannot be used with preferred buffer type CUDA_Host, using CPU "
           "instead\n");
    d.feed("warning: no usable GPU found, --gpu-layers option will be ignored\n");
    d.feed("ggml_cuda_init: failed to initialize CUDA: no CUDA-capable device is detected\n");
    REQUIRE(d.warnings().size() == 4);
    CHECK(d.warnings()[0].code == "partial_gpu_offload");
    CHECK(detail_of(d.warnings()[0], "offloaded_layers") == "40");
    CHECK(detail_of(d.warnings()[0], "total_layers") == "65");
    CHECK(d.warnings()[1].code == "cpu_buffer_fallback");
    CHECK(d.warnings()[1].count == 2);
    CHECK(detail_of(d.warnings()[1], "buffer_type") == "CUDA_Host");
    CHECK(d.warnings()[2].code == "no_gpu_device");
    CHECK(d.warnings()[3].code == "gpu_init_failed");
}

TEST_CASE("log parser detects ROCm, Vulkan, multi-vendor OOM and ternary incompatibility") {
    {
        LogDiagnostics d;
        d.feed("ggml_hip_init: failed to initialize HIP: no HIP-capable device detected\n");
        d.feed("ggml_rocm_init: failed to initialize ROCm runtime\n");
        d.feed("HSA_STATUS_ERROR_INVALID_ISA: unsupported gfx1100 target\n");
        d.feed("HIP error: hipErrorNoBinaryForGpu\n");
        REQUIRE(d.warnings().size() == 2);
        CHECK(d.warnings()[0].code == "gpu_init_failed");
        CHECK(d.warnings()[0].count == 2);
        CHECK(detail_of(d.warnings()[0], "backend") == "rocm");
        CHECK(d.warnings()[1].code == "rocm_runtime_error");
        CHECK(d.warnings()[1].count == 2);
        CHECK(detail_of(d.warnings()[1], "backend") == "rocm");
    }
    {
        LogDiagnostics d;
        d.feed("ggml_vulkan: failed to initialize Vulkan\n");
        d.feed("ggml_vulkan: device lost on AMD Radeon RX 9060 XT\n");
        REQUIRE(d.warnings().size() == 2);
        CHECK(d.warnings()[0].code == "vulkan_runtime_error");
        CHECK(detail_of(d.warnings()[0], "backend") == "vulkan");
        CHECK(detail_of(d.warnings()[0], "reason") == "init_failed");
        CHECK(d.warnings()[1].code == "vulkan_runtime_error");
        CHECK(detail_of(d.warnings()[1], "backend") == "vulkan");
        CHECK(detail_of(d.warnings()[1], "reason") == "device_lost");
    }
    {
        LogDiagnostics d;
        d.feed("CUDA error: out of memory\n");
        d.feed("ggml_vulkan: vk::OutOfDeviceMemoryError: Device::allocateMemory\n");
        d.feed("ggml_hip: hipErrorOutOfMemory allocating KV cache\n");
        d.feed("ggml_hip: allocation failed: out of memory\n");
        d.feed("ggml_vulkan: buffer allocation out of memory\n");
        // Non-GPU text mentioning "out of memory" or a word containing "hip" must not trigger.
        d.feed("ordinary text about a relationship running out of memory\n");
        REQUIRE(d.warnings().size() == 3);
        CHECK(d.warnings()[0].code == "gpu_oom");
        CHECK(detail_of(d.warnings()[0], "backend") == "cuda");
        CHECK(d.warnings()[0].count == 1);
        CHECK(d.warnings()[1].code == "gpu_oom");
        CHECK(detail_of(d.warnings()[1], "backend") == "vulkan");
        CHECK(d.warnings()[1].count == 2);
        CHECK(d.warnings()[2].code == "gpu_oom");
        CHECK(detail_of(d.warnings()[2], "backend") == "rocm");
        CHECK(d.warnings()[2].count == 2);
    }
    {
        LogDiagnostics d;
        // Normal PrismML metadata info lines must not trigger a warning.
        d.feed("llama_model_loader: - kv 24: prism.hadamard.block_size u32 = 128\n");
        CHECK(d.warnings().empty());

        d.feed("gguf_init_from_file_impl: unknown tensor type 142\n");
        d.feed("llama_model_load: unsupported ggml type 41\n");
        d.feed("llama_model_load: error: unsupported key prism.hadamard.sign_values\n");
        REQUIRE(d.warnings().size() == 1);
        CHECK(d.warnings()[0].code == "ternary_runtime_incompatible");
        CHECK(d.warnings()[0].count == 3);
        CHECK(d.warnings()[0].message.find("PrismML") != std::string::npos);
        CHECK(d.warnings()[0].message.find("b11556") != std::string::npos);
    }
}

TEST_CASE("log diagnostics readiness reports blind verbosity and final option precedence") {
    LogDiagnostics d;
    d.ready({"--log-verbosity", "4", "-lv", "3"});
    CHECK(d.offload_status() == "blind");
    REQUIRE(d.warnings().size() == 1);
    CHECK(d.warnings()[0].code == "diagnostics_blind");

    d.ready({"--log-verbosity=3", "--log-verbosity", "4"});
    CHECK(d.offload_status() == "pending");
    CHECK(d.warnings().empty());
    d.ready({"-lv", "4", "--log-verbosity", "invalid"});
    CHECK(d.offload_status() == "blind");
    REQUIRE(d.warnings().size() == 1);
    CHECK(d.warnings()[0].code == "diagnostics_blind");
}

TEST_CASE("log diagnostics tracks partial and complete offload") {
    LogDiagnostics d;
    d.ready({"--log-verbosity=4"});
    CHECK(d.offload_status() == "pending");
    d.feed("load_tensors: offloaded 63/66 layers to GPU\n");
    CHECK(d.offload_status() == "partial");
    REQUIRE(d.warnings().size() == 1);
    CHECK(d.warnings()[0].code == "partial_gpu_offload");
    d.feed("load_tensors: offloaded 66/66 layers to GPU\n");
    CHECK(d.offload_status() == "ok");
    CHECK(d.warnings().size() == 1);

    d.reset();
    CHECK(d.offload_status() == "pending");
    CHECK(d.warnings().empty());
}

TEST_CASE("real diagnostics fixtures classify lv3 as blind and lv4 offload as observable") {
    LogDiagnostics blind;
    blind.ready({"--log-verbosity", "3"});
    blind.feed(fixture_text("llama-server.log.snapshot-1144.txt"));
    CHECK(blind.offload_status() == "blind");
    REQUIRE(blind.warnings().size() >= 1);
    CHECK(std::any_of(blind.warnings().begin(), blind.warnings().end(), [](const auto &warning) {
        return warning.code == "diagnostics_blind";
    }));

    LogDiagnostics visible;
    visible.ready({"--log-verbosity", "4"});
    visible.feed(fixture_text("ollama-27b-load-excerpt.txt"));
    CHECK(visible.offload_status() == "ok");
    CHECK(std::none_of(visible.warnings().begin(), visible.warnings().end(), [](const auto &warning) {
        return warning.code == "partial_gpu_offload" || warning.code == "diagnostics_blind";
    }));
    // Keep the captured fixture exact; vary only the offload count to replay
    // the requested 63/66 partial-offload arm with its real log prefix.
    auto partial = fixture_text("ollama-27b-load-excerpt.txt");
    const auto at = partial.find("offloaded 66/66");
    REQUIRE(at != std::string::npos);
    partial.replace(at, std::string("offloaded 66/66").size(), "offloaded 63/66");
    visible.reset();
    visible.feed(partial);
    CHECK(visible.offload_status() == "partial");
    CHECK(std::any_of(visible.warnings().begin(), visible.warnings().end(), [](const auto &warning) {
        return warning.code == "partial_gpu_offload";
    }));
    visible.feed("load_tensors: offloaded 63/66 layers to GPU\n");
    CHECK(visible.offload_status() == "partial");
}

TEST_CASE("log parser frames lines across chunks and bounds line length and warning count") {
    LogDiagnostics d;
    const std::string line = std::string(kFaLine) + "\n";
    for (const char c : line)
        d.feed(std::string_view(&c, 1));
    REQUIRE(d.warnings().size() == 1);

    // An oversized line keeps its head (where the diagnostic text is) and
    // never grows the buffer without bound.
    d.reset();
    d.feed(std::string(kFaLine) + std::string(1u << 20, 'x'));
    d.feed(std::string(1u << 20, 'y') + "\n");
    REQUIRE(d.warnings().size() == 1);
    CHECK(d.lines() == 1);

    d.reset();
    for (int i = 1; i <= 100; ++i)
        d.feed("load_tensors: offloaded " + std::to_string(i) + "/200 layers to GPU\n");
    CHECK(d.warnings().size() == LogDiagnostics::kMaxWarnings);

    d.reset();
    d.feed(kFaLine); // no trailing newline
    CHECK(d.warnings().empty());
    d.finish();
    CHECK(d.warnings().size() == 1);
}

TEST_CASE("log tail reads incrementally and restarts after truncation") {
    const auto path = temp_log("tail");
    {
        std::ofstream out(path, std::ios::binary);
        out << kQuietLines[2] << "\n" << kNextnLines[0] << "\n";
    }
    LogDiagnostics d;
    LogTail tail(path.string());
    REQUIRE(tail.poll(d).ok());
    CHECK(d.lines() == 2);
    REQUIRE(d.warnings().size() == 1);
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out << kFaLine << "\n";
    }
    REQUIRE(tail.poll(d, 16).ok()); // bounded read: only part of the new line
    CHECK(d.lines() == 2);
    for (int i = 0; i < 100 && d.lines() < 3; ++i)
        REQUIRE(tail.poll(d, 16).ok());
    CHECK(d.lines() == 3);
    CHECK(d.warnings().size() == 2);
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out << kNextnLines[1] << "\n";
    }
    REQUIRE(tail.poll(d).ok());
    CHECK(d.lines() == 1); // parser state restarted with the new file
    REQUIRE(d.warnings().size() == 1);
    CHECK(d.warnings()[0].code == "mtp_tensors_ignored");
    std::filesystem::remove(path);
    CHECK(LogTail(path.string()).poll(d).code() == ErrorCode::not_found);
}

// ------------------------------------------------------- supervisor policies

TEST_CASE("warn policy serves a spilled child and reports the numbers") {
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 15100 * MiB, 950 * MiB); });
    Supervisor s(guarded_options(source, SpillPolicy::warn), std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    const auto status = s.runtime_status();
    CHECK(status.gpu_memory.probe == "fake");
    CHECK(status.gpu_memory.status == "ok");
    CHECK(status.gpu_memory.dedicated_bytes == 15100 * MiB);
    CHECK(status.gpu_memory.shared_bytes == 950 * MiB);
    CHECK(status.gpu_memory.spilled);
    CHECK(status.gpu_memory.samples == 1);
    CHECK(status.context.policy == "warn");
    CHECK(status.context.configured_ctx == 100096u);
    CHECK(status.context.fitted_ctx == 100096u);
    CHECK(status.context.outcome == "not_needed");
    const auto *w = find_warning(status, "vram_spill");
    REQUIRE(w);
    CHECK(detail_of(*w, "shared_bytes") == std::to_string(950 * MiB));
    CHECK(w->message.find("950 MiB") != std::string::npos);
    CHECK(launched_contexts(state) == std::vector<std::uint64_t>{100096});
    s.stop();
}

TEST_CASE("refuse policy fails startup with the measured numbers and stops the child") {
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 15100 * MiB, 972 * MiB); });
    Supervisor s(guarded_options(source, SpillPolicy::refuse), std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::unavailable);
    const std::string message = r.status().message();
    CHECK(message.find("972 MiB") != std::string::npos);
    CHECK(message.find("256 MiB") != std::string::npos);
    CHECK(message.find("15100 MiB") != std::string::npos);
    CHECK(message.find("ctx 100096") != std::string::npos);
    CHECK(message.find("refuse") != std::string::npos);
    REQUIRE(launched_contexts(state).size() == 1);
    CHECK(child_at(state, 0)->stops.load() >= 1);
    CHECK(s.runtime_status().context.outcome == "refused");
    CHECK_FALSE(s.start().ok()); // terminal: never relaunched behind the caller
    CHECK(launched_contexts(state).size() == 1);
}

TEST_CASE("refuse policy serves a clean child") {
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 15200 * MiB, 198 * MiB); });
    Supervisor s(guarded_options(source, SpillPolicy::refuse), std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    CHECK_FALSE(s.runtime_status().gpu_memory.spilled);
    CHECK_FALSE(find_warning(s.runtime_status(), "vram_spill"));
    s.stop();
}

TEST_CASE("auto_fit steps the context down until the child no longer spills") {
    auto state = pid_state();
    auto o = guarded_options(ctx_dependent_counters(state, 70000), SpillPolicy::auto_fit);
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    REQUIRE_MESSAGE(r.ok(), r.status().to_string());
    CHECK(launched_contexts(state) == std::vector<std::uint64_t>{100096, 84992, 71680, 60416});
    for (std::size_t i = 0; i < 3; ++i)
        CHECK(child_at(state, i)->stops.load() >= 1);
    CHECK(child_at(state, 3)->alive.load());
    const auto status = s.runtime_status();
    CHECK(status.context.configured_ctx == 100096u);
    CHECK(status.context.fitted_ctx == 60416u);
    CHECK(status.context.fit_attempts == 3);
    CHECK(status.context.outcome == "fitted");
    CHECK_FALSE(status.gpu_memory.spilled);
    CHECK(status.gpu_memory.shared_bytes == 182 * MiB);
    // The fitted context survives a crash restart (no new fitting needed).
    child_at(state, 3)->alive.store(false);
    REQUIRE(wait_for_starts(state, 5));
    CHECK(launched_contexts(state).back() == 60416u);
    s.stop();
}

TEST_CASE("auto_fit stops at the floor and fails instead of looping") {
    auto state = pid_state();
    auto o = guarded_options(ctx_dependent_counters(state, 0), SpillPolicy::auto_fit);
    o.spill_guard.min_ctx = 65536;
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::unavailable);
    CHECK(r.status().message().find("minimum context 65536") != std::string::npos);
    CHECK(launched_contexts(state) == std::vector<std::uint64_t>{100096, 84992, 71680, 65536});
    CHECK(s.runtime_status().context.outcome == "floor_reached");
    CHECK(s.runtime_status().context.fit_attempts == 3);
}

TEST_CASE("auto_fit gives up after max_attempts relaunches") {
    auto state = pid_state();
    auto o = guarded_options(ctx_dependent_counters(state, 0), SpillPolicy::auto_fit);
    o.spill_guard.max_attempts = 2;
    o.spill_guard.min_ctx = 1024;
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().message().find("after 2 context reduction(s)") != std::string::npos);
    CHECK(launched_contexts(state).size() == 3);
    CHECK(s.runtime_status().context.outcome == "exhausted");
    for (std::size_t i = 0; i < 3; ++i)
        CHECK(child_at(state, i)->stops.load() >= 1);
}

TEST_CASE("auto_fit needs an explicit context size") {
    auto state = pid_state();
    auto o = guarded_options(ctx_dependent_counters(state, 0), SpillPolicy::auto_fit);
    o.arguments = {"--model", "model.gguf"};
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    REQUIRE_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::invalid_argument);
    CHECK(state->specs.empty());
}

TEST_CASE("probe errors and unsupported platforms never refuse or refit") {
    auto state = pid_state();
    auto failing = std::make_shared<FakeCounters>(
        [](std::uint32_t) -> Result<GpuProcessCounters> { return Status(ErrorCode::io_error, "PDH failed"); });
    Supervisor a(guarded_options(failing, SpillPolicy::refuse), std::make_unique<FakeLauncher>(state));
    REQUIRE(a.start().ok());
    CHECK(a.runtime_status().gpu_memory.status == "error");
    CHECK(a.runtime_status().gpu_memory.error == "PDH failed");
    a.stop();

    auto unsupported = std::make_shared<FakeCounters>(
        [](std::uint32_t) -> Result<GpuProcessCounters> { return GpuProcessCounters{}; }, false);
    auto state2 = pid_state();
    Supervisor b(guarded_options(unsupported, SpillPolicy::auto_fit), std::make_unique<FakeLauncher>(state2));
    REQUIRE(b.start().ok());
    CHECK(b.runtime_status().gpu_memory.status == "unsupported");
    CHECK(unsupported->reads.load() == 0);
    CHECK(launched_contexts(state2).size() == 1);
    b.stop();
}

TEST_CASE("disabled guard never samples and keeps the launch argv unchanged") {
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 1, 4096 * MiB); });
    auto o = guarded_options(source, SpillPolicy::refuse);
    o.spill_guard.enabled = false;
    const auto argv = o.arguments;
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    CHECK(source->reads.load() == 0);
    CHECK(s.runtime_status().gpu_memory.status == "disabled");
    std::lock_guard lock(state->mutex);
    const auto &launched = state->specs.at(0).arguments;
    CHECK(std::vector<std::string>(launched.begin(), launched.end() - 4) == argv);
}

TEST_CASE("the monitor re-samples periodically after readiness") {
    auto state = pid_state();
    auto calls = std::make_shared<std::atomic<unsigned>>(0);
    auto source = std::make_shared<FakeCounters>([calls](std::uint32_t pid) -> Result<GpuProcessCounters> {
        // Clean at readiness, spilled later (e.g. another app took VRAM).
        return counters(pid, 15000 * MiB, calls->fetch_add(1) == 0 ? 180 * MiB : 700 * MiB);
    });
    auto o = guarded_options(source, SpillPolicy::refuse);
    o.spill_guard.sample_interval = std::chrono::milliseconds(100);
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    CHECK_FALSE(s.runtime_status().gpu_memory.spilled);
    bool spilled = false;
    for (int i = 0; i < 300 && !spilled; ++i) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        spilled = s.runtime_status().gpu_memory.spilled;
    }
    CHECK(spilled);
    const auto status = s.runtime_status();
    CHECK(status.gpu_memory.samples >= 2);
    CHECK(status.gpu_memory.peak_shared_bytes == 700 * MiB);
    // A later spill is reported; the serving child is not killed for it.
    CHECK(s.running());
    CHECK(launched_contexts(state).size() == 1);
    s.stop();
}

TEST_CASE("supervisor appends --log-file and surfaces log warnings after readiness") {
    const auto path = temp_log("supervisor");
    {
        std::ofstream out(path, std::ios::binary);
        out << kNextnLines[0] << "\n" << kFaLine << "\n";
    }
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 15000 * MiB, 180 * MiB); });
    auto o = guarded_options(source, SpillPolicy::warn);
    o.arguments = {"--model", "m.gguf", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q5_1"};
    o.log_file = path.string();
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    // The up-front pairing warning exists before anything is launched.
    REQUIRE(find_warning(s.runtime_status(), "kv_type_mismatch"));
    REQUIRE(s.start().ok());
    {
        std::lock_guard lock(state->mutex);
        const auto &argv = state->specs.at(0).arguments;
        const auto it = std::find(argv.begin(), argv.end(), "--log-file");
        REQUIRE(it != argv.end());
        CHECK(*(it + 1) == path.string());
    }
    const auto status = s.runtime_status();
    const auto *fa = find_warning(status, "kv_kernel_f16_fallback");
    REQUIRE(fa);
    CHECK(detail_of(*fa, "k_type") == "q8_0");
    CHECK(find_warning(status, "mtp_tensors_ignored"));
    CHECK(status.warnings.front().code == "kv_type_mismatch"); // config warnings first
    s.stop();
    std::filesystem::remove(path);

    // An explicit --log-file in args wins and is not duplicated.
    auto state2 = pid_state();
    auto o2 = guarded_options(source, SpillPolicy::warn);
    o2.arguments.insert(o2.arguments.end(), {"--log-file", "explicit.log"});
    o2.log_file = "ignored.log";
    Supervisor s2(o2, std::make_unique<FakeLauncher>(state2));
    const auto argv2 = s2.launch_arguments();
    CHECK(std::count(argv2.begin(), argv2.end(), "--log-file") == 1);
    CHECK(std::find(argv2.begin(), argv2.end(), "ignored.log") == argv2.end());
}

TEST_CASE("log parser handles the real upstream order: fault info first, then the lost line") {
    {
        LogDiagnostics d;
        d.feed("ggml_vulkan: device fault on AMD Radeon RX 9060 XT: page fault at 0x1000\n");
        d.feed("address fault 0: type=2 address=0x1000 precision=0x1000\n");
        d.feed("vendor fault 0: GCVM_L2_PROTECTION_FAULT (code=0x1 data=0x2)\n");
        d.feed("ggml_vulkan: device lost on AMD Radeon RX 9060 XT, likely caused by previous submission (nodes 1 to 2):\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->reason == "device_fault");
        CHECK(d.device_fault()->device == "AMD Radeon RX 9060 XT");
        CHECK(d.device_fault()->description == "page fault at 0x1000");
        CHECK(d.device_fault()->details.size() == 2);
    }
    {
        // No driver description: upstream prints the detail lines with no fault line.
        LogDiagnostics d;
        d.feed("address fault 0: type=2 address=0x1000 precision=0x1000\n");
        d.feed("vendor fault 0: GCVM_L2_PROTECTION_FAULT (code=0x1 data=0x2)\n");
        CHECK_FALSE(d.device_fault().has_value());
        d.feed("ggml_vulkan: device lost on AMD Radeon RX 9060 XT\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->details.size() == 2);
    }
    {
        LogDiagnostics d;
        d.feed("ggml_vulkan: device lost on AMD Radeon RX 9060 XT waiting for submission (nodes 1 to 2):\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->device == "AMD Radeon RX 9060 XT");
    }
}

TEST_CASE("log parser records device faults: Vulkan device lost, VK_EXT_device_fault and fatal CUDA errors") {
    {
        LogDiagnostics d;
        d.feed("ggml_vulkan: Using Vulkan0 (AMD Radeon RX 9060 XT)\n");
        CHECK_FALSE(d.device_fault().has_value());
        d.feed("ggml_vulkan: device lost on AMD Radeon RX 9060 XT\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->backend == "vulkan");
        CHECK(d.device_fault()->reason == "device_lost");
        CHECK(d.device_fault()->device == "AMD Radeon RX 9060 XT");
        // The existing warning is unchanged.
        REQUIRE(d.warnings().size() == 1);
        CHECK(d.warnings()[0].code == "vulkan_runtime_error");
        // VK_EXT_device_fault output arrives after the lost line and enriches it.
        d.feed("ggml_vulkan: device fault on AMD Radeon RX 9060 XT: page fault at 0x1000\n");
        d.feed("  address fault 0: type=2 address=0x1000 precision=0x1000\n");
        d.feed("  vendor fault 0: GCVM_L2_PROTECTION_FAULT (code=0x1 data=0x2)\n");
        d.feed("  unrelated indented line\n");
        const auto &fault = *d.device_fault();
        CHECK(fault.reason == "device_lost"); // first fault wins
        CHECK(fault.device == "AMD Radeon RX 9060 XT");
        CHECK(fault.description == "page fault at 0x1000");
        REQUIRE(fault.details.size() == 2);
        CHECK(fault.details[0].rfind("address fault 0", 0) == 0);
        CHECK(fault.details[1].rfind("vendor fault 0", 0) == 0);
        REQUIRE(d.warnings().size() == 2);
        CHECK(d.warnings()[1].code == "device_fault");
        CHECK(detail_of(d.warnings()[1], "backend") == "vulkan");
        d.reset();
        CHECK_FALSE(d.device_fault().has_value());
    }
    {
        LogDiagnostics d;
        d.feed("ggml_vulkan: device fault on Radeon 780M Graphics: hang\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->reason == "device_fault");
        CHECK(d.device_fault()->device == "Radeon 780M Graphics");
        CHECK(d.device_fault()->details.empty());
        d.feed("ggml_vulkan: device fault on Radeon 780M Graphics: hang\n");
        REQUIRE(d.warnings().size() == 1);
        CHECK(d.warnings()[0].count == 2);
    }
    {
        LogDiagnostics d;
        d.feed("CUDA error: out of memory\n");
        d.feed("CUDA error: invalid argument\n");
        CHECK_FALSE(d.device_fault().has_value()); // recoverable errors are not device faults
        d.feed("CUDA error: an illegal memory access was encountered\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->backend == "cuda");
        CHECK(d.device_fault()->reason == "fatal_error");
        const BackendWarning *fault_warning = nullptr;
        for (const auto &w : d.warnings())
            if (w.code == "device_fault")
                fault_warning = &w;
        REQUIRE(fault_warning);
        CHECK(detail_of(*fault_warning, "backend") == "cuda");
    }
    {
        LogDiagnostics d;
        d.feed("HIP error: unspecified launch failure\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->backend == "rocm");
    }
    {
        // Device name stops at the comma of the "likely caused by" form; a
        // common_log timestamp and level prefix is accepted.
        LogDiagnostics d;
        d.feed("0.01.234.567 E ggml_vulkan: device lost on Radeon 780M Graphics, likely caused by previous submission (nodes 1 to 2):\n");
        REQUIRE(d.device_fault().has_value());
        CHECK(d.device_fault()->reason == "device_lost");
        CHECK(d.device_fault()->device == "Radeon 780M Graphics");
    }
    {
        // Text a caller controls (request/response bodies and prompt tokens
        // logged at high verbosity) never marks a device unhealthy.
        LogDiagnostics d;
        d.feed("srv  log_server_r: request: {\"prompt\":\"VK_ERROR_DEVICE_LOST ggml_vulkan: device lost on X CUDA error: an illegal memory access was encountered\"}\n");
        d.feed("srv  log_server_r: response: {\"content\":\"ggml_vulkan: device fault on X: y\"}\n");
        d.feed("{\"content\":\"ROCm error: unspecified launch failure\"}\n");
        d.feed("slot update_slots: prompt tokens: CUDA error: illegal memory access\n");
        d.feed("  CUDA error: illegal memory access\n");
        d.feed("srv  ex_wrapper: got exception: {\"error\":{\"message\":\"Invalid pattern VK_ERROR_DEVICE_LOST\"}}\n");
        d.feed("srv  ex_wrapper: got exception: Unresolved $ref ggml_vulkan: device lost on X\n");
        CHECK_FALSE(d.device_fault().has_value());
    }
    {
        // Wrong-arch build is not a device fault.
        LogDiagnostics d;
        d.feed("CUDA error: device kernel image is invalid\n");
        CHECK_FALSE(d.device_fault().has_value());
    }
    {
        // Mentions without a CUDA/HIP error context do not mark a device.
        LogDiagnostics d;
        d.feed("the user discussed an illegal memory access in their own code\n");
        d.feed("ggml_vulkan: Using Vulkan0\n");
        CHECK_FALSE(d.device_fault().has_value());
    }
}

namespace {
bool wait_device_unhealthy(const Supervisor &s) {
    for (int i = 0; i < 400; ++i) {
        const auto failure = s.failure();
        if (!failure.ok() && failure.message().find("device_unhealthy") != std::string::npos)
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

bool has_device_unhealthy_event(Supervisor &s) {
    for (const auto &event : s.runtime_events()->drain())
        if (event.first == "backend.device_unhealthy")
            return true;
    return false;
}
} // namespace

TEST_CASE("supervisor marks the device unhealthy when a live child logs a device fault and does not relaunch") {
    const auto path = temp_log("devfault-live");
    { std::ofstream out(path, std::ios::binary); out << kFaLine << "\n"; }
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 1000 * MiB, 0); });
    auto o = guarded_options(source, SpillPolicy::warn);
    o.log_file = path.string();
    o.spill_guard.sample_interval = std::chrono::milliseconds(100);
    auto healthy = std::make_shared<std::atomic<bool>>(true);
    o.health_check = [healthy](std::uint16_t, std::chrono::milliseconds) { return healthy->load(); };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto started = s.start();
    INFO(started.status().message());
    REQUIRE(started.ok());
    healthy->store(false); // second signal: /health stops answering
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out << "ggml_vulkan: device lost on AMD Radeon RX 9060 XT, likely caused by previous submission (nodes 3 to 9):\n"
            << "ggml_vulkan: device fault on AMD Radeon RX 9060 XT: page fault\n"
            << "  vendor fault 0: GCVM_L2_PROTECTION_FAULT (code=0x1 data=0x2)\n";
    }
    REQUIRE(wait_device_unhealthy(s));
    std::this_thread::sleep_for(std::chrono::milliseconds(150)); // a relaunch (5-8 ms backoff) would show by now
    {
        std::lock_guard lock(state->mutex);
        CHECK(state->starts.size() == 1);
    }
    CHECK(child_at(state, 0)->stops.load() >= 1);
    CHECK_FALSE(s.running());
    CHECK_FALSE(s.start().ok()); // nothing is scheduled on the lost device
    CHECK_FALSE(s.acquire_request().ok());
    const auto status = s.runtime_status();
    REQUIRE(status.diagnostics.has_value());
    const auto *device = status.diagnostics->find("device");
    REQUIRE(device);
    CHECK(device->as_object().find("status")->as_string() == "unhealthy");
    CHECK(device->as_object().find("backend")->as_string() == "vulkan");
    CHECK(device->as_object().find("device")->as_string() == "AMD Radeon RX 9060 XT");
    CHECK(device->as_object().find("details")->as_array().size() == 1);
    CHECK(find_warning(status, "device_fault"));
    CHECK(has_device_unhealthy_event(s));
    s.stop();
    std::filesystem::remove(path);
}

TEST_CASE("supervisor marks the device unhealthy when the child dies after logging a device fault") {
    const auto path = temp_log("devfault-exit");
    { std::ofstream out(path, std::ios::binary); out << kFaLine << "\n"; }
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 1000 * MiB, 0); });
    auto o = guarded_options(source, SpillPolicy::warn);
    o.log_file = path.string();
    o.spill_guard.sample_interval = std::chrono::minutes(30); // only the exit path can see the fault
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out << "0.12.345.678 E ggml_vulkan: device lost on AMD Radeon RX 9060 XT\n";
    }
    child_at(state, 0)->alive.store(false); // the child aborts
    REQUIRE(wait_device_unhealthy(s));
    std::this_thread::sleep_for(std::chrono::milliseconds(150));
    {
        std::lock_guard lock(state->mutex);
        CHECK(state->starts.size() == 1);
    }
    CHECK(has_device_unhealthy_event(s));
    s.stop();
    std::filesystem::remove(path);
}

TEST_CASE("a fault line from a child that still answers /health is not terminal") {
    const auto path = temp_log("devfault-unconfirmed");
    { std::ofstream out(path, std::ios::binary); out << kFaLine << "\n"; }
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 1000 * MiB, 0); });
    auto o = guarded_options(source, SpillPolicy::warn);
    o.log_file = path.string();
    o.spill_guard.sample_interval = std::chrono::milliseconds(100);
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out << "ggml_vulkan: device lost on AMD Radeon RX 9060 XT\n";
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(400));
    CHECK(s.running());
    {
        std::lock_guard lock(state->mutex);
        CHECK(state->starts.size() == 1);
    }
    CHECK_FALSE(s.failure().message().find("device_unhealthy") != std::string::npos);
    s.stop();
    std::filesystem::remove(path);
}

TEST_CASE("a fault logged while the model loads marks the device unhealthy when the child dies") {
    const auto path = temp_log("devfault-load");
    { std::ofstream out(path, std::ios::binary); out << kFaLine << "\n"; }
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 1000 * MiB, 0); });
    auto o = guarded_options(source, SpillPolicy::warn);
    o.log_file = path.string();
    auto ready = std::make_shared<std::atomic<bool>>(false);
    o.health_check = [ready](std::uint16_t, std::chrono::milliseconds) { return ready->load(); };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    std::thread starter([&] { (void)s.start(); });
    REQUIRE(wait_for_starts(state, 1));
    {
        std::ofstream out(path, std::ios::binary | std::ios::app);
        out << "ggml_vulkan: device lost on AMD Radeon RX 9060 XT waiting for submission (nodes 1 to 2):\n";
    }
    child_at(state, 0)->alive.store(false); // dies while loading, before /health ever answers
    REQUIRE(wait_device_unhealthy(s));
    s.stop();
    starter.join();
    std::filesystem::remove(path);
}

TEST_CASE("a child that exits without a device fault is still restarted") {
    const auto path = temp_log("nofault-exit");
    { std::ofstream out(path, std::ios::binary); out << kFaLine << "\n"; }
    auto state = pid_state();
    auto source = std::make_shared<FakeCounters>(
        [](std::uint32_t pid) -> Result<GpuProcessCounters> { return counters(pid, 1000 * MiB, 0); });
    auto o = guarded_options(source, SpillPolicy::warn);
    o.log_file = path.string();
    o.spill_guard.sample_interval = std::chrono::minutes(30);
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    child_at(state, 0)->alive.store(false);
    REQUIRE(wait_for_starts(state, 2));
    CHECK_FALSE(has_device_unhealthy_event(s));
    s.stop();
    std::filesystem::remove(path);
}

// ------------------------------------------------------------ backend level

TEST_CASE("backend runtime status: spawn reports and generic attach stays silent") {
    LlamaServerBackendOptions attach;
    attach.native_completion = false;
    CHECK_FALSE(make_llamaserver_backend(attach)->runtime_status().has_value());

    LlamaServerBackendOptions spawn;
    spawn.mode = LlamaServerMode::spawn;
    spawn.executable = "llama-server";
    spawn.args = {"--ctx-size", "81920", "--flash-attn", "on", "--cache-type-k", "q8_0", "--cache-type-v", "q5_1"};
    const auto status = make_llamaserver_backend(spawn)->runtime_status();
    REQUIRE(status.has_value());
    CHECK(status->context.policy == "warn");
    CHECK(status->context.configured_ctx == 81920u);
    CHECK(status->gpu_memory.status != "ok"); // nothing launched yet
    CHECK(status->gpu_memory.spill_threshold_bytes == 256 * MiB);
    REQUIRE(find_warning(*status, "kv_type_mismatch"));

    spawn.diagnostics.kv_pairing_check = false;
    CHECK(make_llamaserver_backend(spawn)->runtime_status()->warnings.empty());

    // Invalid guard settings surface on the first request, before any launch.
    spawn.spill_guard.policy = LlamaServerSpillPolicy::auto_fit;
    spawn.args = {"--jinja"};
    auto backend = make_llamaserver_backend(spawn);
    const auto models = backend->list_models();
    REQUIRE_FALSE(models.ok());
    CHECK(models.status().code() == ErrorCode::invalid_argument);
    spawn.args = {"-c", "4096"};
    spawn.spill_guard.fit_step_factor = 1.5;
    CHECK(make_llamaserver_backend(spawn)->list_models().status().code() == ErrorCode::invalid_argument);
}

TEST_CASE("runtime status JSON shape") {
    BackendRuntimeStatus s;
    s.gpu_memory.probe = "pdh";
    s.gpu_memory.status = "ok";
    s.gpu_memory.dedicated_bytes = 15266 * MiB;
    s.gpu_memory.shared_bytes = 388 * MiB;
    s.gpu_memory.spill_threshold_bytes = 256 * MiB;
    s.gpu_memory.spilled = true;
    s.gpu_memory.samples = 1;
    s.context.policy = "auto_fit";
    s.context.configured_ctx = 100096;
    s.context.fitted_ctx = 84992;
    s.context.fit_attempts = 1;
    s.context.outcome = "fitted";
    s.warnings.push_back(BackendWarning{"kv_kernel_f16_fallback", "warning", "log", "m", {{"k_type", "q8_0"}}, 2});
    const auto v = json::Value(to_json(s));
    CHECK(v.dump() ==
          R"({"gpu_memory":{"probe":"pdh","status":"ok","dedicated_bytes":16007561216,"shared_bytes":406847488,)"
          R"("peak_shared_bytes":0,"shared_baseline_bytes":0,"spill_threshold_bytes":268435456,"spilled":true,)"
          R"("samples":1},"context":{"policy":"auto_fit","configured_ctx":100096,"fitted_ctx":84992,)"
          R"("fit_attempts":1,"outcome":"fitted"},"warnings":[{"code":"kv_kernel_f16_fallback","severity":)"
          R"("warning","source":"log","message":"m","details":{"k_type":"q8_0"},"count":2}]})");
    BackendRuntimeStatus empty;
    const auto e = json::Value(to_json(empty));
    CHECK(e.find("context")->find("configured_ctx")->is_null());
    CHECK(e.find("context")->find("fitted_ctx")->is_null());
    CHECK(e.find("gpu_memory")->find("error") == nullptr);
}

TEST_CASE("hybrid discrete+iGPU Vulkan0,Vulkan1 pool avoids false-positive spill kills while catching true discrete spill") {
    constexpr std::uint64_t GiB = 1024ull * MiB;
    constexpr AdapterLuid kRxLuid{0x00000000u, 0x000104A4u};
    constexpr AdapterLuid kIgpuLuid{0x00000000u, 0x000118C2u};
    const std::vector<AdapterLuidInfo> adapters{
        {kRxLuid, false},
        {kIgpuLuid, true},
    };

    GpuDeviceInfo rx9060xt;
    rx9060xt.id = "gpu:0";
    rx9060xt.name = "AMD Radeon RX 9060 XT";
    rx9060xt.vendor = GpuVendor::amd;
    rx9060xt.vendor_id = 0x1002u;
    rx9060xt.dedicated_total = MemoryReading::known(16 * GiB);
    rx9060xt.shared_total = MemoryReading::known(32 * GiB);
    rx9060xt.is_integrated_uma = false;
    rx9060xt.backend_hint = "vulkan";

    GpuDeviceInfo radeon780m;
    radeon780m.id = "gpu:1";
    radeon780m.name = "AMD Radeon 780M Graphics";
    radeon780m.vendor = GpuVendor::amd;
    radeon780m.vendor_id = 0x1002u;
    radeon780m.dedicated_total = MemoryReading::known(512 * MiB);
    radeon780m.shared_total = MemoryReading::known(32 * GiB);
    radeon780m.is_integrated_uma = true;
    radeon780m.backend_hint = "vulkan";

    const std::vector<GpuDeviceInfo> node1_gpus{rx9060xt, radeon780m};
    const std::vector<std::string> hybrid_args{"--model", "qwen-80b.gguf", "--ctx-size", "32768",
                                               "--device", "Vulkan0,Vulkan1", "-ngl", "999"};

    const auto make_hybrid_counters = [&](std::uint32_t pid, std::uint64_t discrete_shared_bytes,
                                          std::uint64_t uma_shared_bytes) {
        GpuProcessCounters c;
        c.dedicated = {{instance(pid, "0x000104A4"), 15 * GiB}, {instance(pid, "0x000118C2"), 512 * MiB}};
        c.shared = {{instance(pid, "0x000104A4"), discrete_shared_bytes},
                    {instance(pid, "0x000118C2"), uma_shared_bytes}};
        classify_pdh_counters(c, adapters);
        return c;
    };

    // 1. Healthy hybrid spread (~25 GiB intentional UMA shared on 780M, 160 MiB clean on RX 9060 XT):
    //    must serve under BOTH policy=refuse and policy=auto_fit with zero context shrink.
    for (const auto policy : {LlamaServerSpillPolicy::refuse, LlamaServerSpillPolicy::auto_fit}) {
        auto state = pid_state();
        auto source = std::make_shared<FakeCounters>([&](std::uint32_t pid) -> Result<GpuProcessCounters> {
            return make_hybrid_counters(pid, 160 * MiB, 25 * GiB);
        });
        LlamaServerSpillGuardOptions guard_in;
        guard_in.policy = policy;
        guard_in.uma_mode_policy = "auto";

        auto o = base_options();
        o.arguments = hybrid_args;
        o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
        o.gpu_counters = source;
        o.spill_guard = to_spill_guard_options(guard_in, hybrid_args, node1_gpus);
        REQUIRE(o.spill_guard.uma_mode);
        REQUIRE(o.spill_guard.hybrid_uma_mode);

        Supervisor s(o, std::make_unique<FakeLauncher>(state));
        const auto started = s.start();
        REQUIRE_MESSAGE(started.ok(), started.status().to_string());
        const auto status = s.runtime_status();
        CHECK_FALSE(status.gpu_memory.spilled);
        CHECK_FALSE(find_warning(status, "vram_spill"));
        CHECK(status.context.fit_attempts == 0);
        CHECK(status.context.fitted_ctx == 32768u);
        CHECK(launched_contexts(state) == std::vector<std::uint64_t>{32768u});
        s.stop();
    }

    // 2. Genuine discrete VRAM spill inside a hybrid spread (RX 9060 XT 950 MiB shared + 780M 25 GiB UMA):
    //    refuse policy still fails startup, and auto_fit reduces ctx until discrete shared drops clean.
    {
        auto state = pid_state();
        auto source = std::make_shared<FakeCounters>([&](std::uint32_t pid) -> Result<GpuProcessCounters> {
            return make_hybrid_counters(pid, 950 * MiB, 25 * GiB);
        });
        LlamaServerSpillGuardOptions guard_in;
        guard_in.policy = LlamaServerSpillPolicy::refuse;
        guard_in.uma_mode_policy = "auto";

        auto o = base_options();
        o.arguments = hybrid_args;
        o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
        o.gpu_counters = source;
        o.spill_guard = to_spill_guard_options(guard_in, hybrid_args, node1_gpus);

        Supervisor s(o, std::make_unique<FakeLauncher>(state));
        const auto started = s.start();
        REQUIRE_FALSE(started.ok());
        CHECK(started.status().code() == ErrorCode::unavailable);
        CHECK(s.runtime_status().context.outcome == "refused");
        CHECK(s.runtime_status().gpu_memory.spilled);
    }

    {
        auto state = pid_state();
        auto source = std::make_shared<FakeCounters>([state, &make_hybrid_counters](std::uint32_t pid) -> Result<GpuProcessCounters> {
            std::lock_guard lock(state->mutex);
            const auto index = static_cast<std::size_t>(pid - 1000);
            if (pid < 1000 || index >= state->specs.size())
                return Status(ErrorCode::internal, "unknown fake pid");
            const auto ctx = find_context_size(state->specs[index].arguments);
            if (!ctx)
                return Status(ErrorCode::internal, "missing ctx");
            const std::uint64_t discrete_shared = *ctx > 24576u ? 950 * MiB : 160 * MiB;
            return make_hybrid_counters(pid, discrete_shared, 25 * GiB);
        });
        LlamaServerSpillGuardOptions guard_in;
        guard_in.policy = LlamaServerSpillPolicy::auto_fit;
        guard_in.uma_mode_policy = "auto";

        auto o = base_options();
        o.arguments = hybrid_args;
        o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
        o.gpu_counters = source;
        o.spill_guard = to_spill_guard_options(guard_in, hybrid_args, node1_gpus);

        Supervisor s(o, std::make_unique<FakeLauncher>(state));
        const auto started = s.start();
        REQUIRE_MESSAGE(started.ok(), started.status().to_string());
        const auto status = s.runtime_status();
        CHECK_FALSE(status.gpu_memory.spilled);
        CHECK(status.context.outcome == "fitted");
        CHECK(status.context.fit_attempts >= 1);
        CHECK(status.context.fitted_ctx.value_or(0) <= 24576u);
        s.stop();
    }

    // 3. Unclassified aggregate counters in hybrid mode (e.g. Linux combined amdgpu-fdinfo):
    //    must NOT false-positive kill or shrink a healthy hybrid spread.
    {
        auto state = pid_state();
        auto source = std::make_shared<FakeCounters>([](std::uint32_t pid) -> Result<GpuProcessCounters> {
            GpuProcessCounters c;
            c.dedicated = {{"pid_" + std::to_string(pid) + "_drm_fdinfo", 15 * GiB + 512 * MiB}};
            c.shared = {{"pid_" + std::to_string(pid) + "_drm_fdinfo", 25 * GiB + 160 * MiB}};
            return c;
        });
        LlamaServerSpillGuardOptions guard_in;
        guard_in.policy = LlamaServerSpillPolicy::refuse;
        guard_in.uma_mode_policy = "auto";

        auto o = base_options();
        o.arguments = hybrid_args;
        o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
        o.gpu_counters = source;
        o.spill_guard = to_spill_guard_options(guard_in, hybrid_args, node1_gpus);

        Supervisor s(o, std::make_unique<FakeLauncher>(state));
        const auto started = s.start();
        REQUIRE_MESSAGE(started.ok(), started.status().to_string());
        CHECK_FALSE(s.runtime_status().gpu_memory.spilled);
        s.stop();
    }
}
