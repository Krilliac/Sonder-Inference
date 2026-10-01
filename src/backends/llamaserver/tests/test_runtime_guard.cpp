// VRAM-spill guard, KV cache pairing check and llama-server log diagnostics
// (docs/integration/vram-spill.md). GPU counters come from a fake source and
// children from the fake launcher; nothing here needs a GPU or llama-server.
#include "../gpu_memory.hpp"
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
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
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
    auto other = check_kv_cache_pairing({"-fa", "on", "-ctk", "iq4_nl", "-ctv", "iq4_nl"});
    REQUIRE(other.size() == 1);
    CHECK(other[0].code == "kv_type_kernel_unknown");
    CHECK(other[0].severity == "info");
}

// --------------------------------------------------------------- log parser

TEST_CASE("log parser extracts the FlashAttention f16 conversion from the real log line") {
    LogDiagnostics d;
    d.feed(std::string(kFaLine) + "\n");
    REQUIRE(d.warnings().size() == 1);
    const auto &w = d.warnings()[0];
    CHECK(w.code == "kv_kernel_f16_fallback");
    CHECK(w.source == "log");
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
