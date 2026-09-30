// Launch profiles: schema parsing and validation, llama-server argv golden
// tests, direct llamacpp mapping and rejection, the GGUF header reader, the
// VRAM estimate (attention-only and hybrid), sampling defaults, and the
// profile metadata in /v1/models (plus unchanged behaviour without profiles).
#include <doctest/doctest.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "server_test_support.hpp"
#include "sonder/inference/launch_profile.hpp"

using namespace server_test;
namespace json = sonder::inference::json;
using si::LaunchProfile;

namespace {

LaunchProfile parse_single(const std::string& profile_json) {
    auto doc = json::parse("{\"profiles\":[" + profile_json + "]}");
    REQUIRE(doc.ok());
    auto parsed = si::parse_launch_profiles(doc.value());
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    REQUIRE(parsed.value().size() == 1);
    return parsed.value().front();
}

std::string parse_error(const std::string& profile_json) {
    auto doc = json::parse("{\"profiles\":[" + profile_json + "]}");
    REQUIRE(doc.ok());
    auto parsed = si::parse_launch_profiles(doc.value());
    REQUIRE_FALSE(parsed.ok());
    CHECK(parsed.status().code() == si::ErrorCode::invalid_argument);
    return parsed.status().message();
}

bool contains(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

// The owner's measured Qwen3.8-27B UD-Q3_K_XL profile (RTX 5070 Ti 16 GB).
const char* kMeasuredProfile = R"({
    "name": "qwen3.8-27b-q3-100k", "backend": "llamaserver",
    "model": "D:/models/Qwen3.8-27B-UD-Q3_K_XL.gguf",
    "n_gpu_layers": 999, "ctx_size": 100096, "parallel": 1, "kv_unified": false,
    "batch_size": 2048, "ubatch_size": 512, "flash_attn": "on",
    "cache_type_k": "q8_0", "cache_type_v": "q5_1", "fit": false,
    "cache_ram_mib": 4096, "ctx_checkpoints": 4, "checkpoint_min_step": 8192,
    "context_shift": false, "jinja": true, "reasoning_format": "deepseek",
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0}
})";

}  // namespace

// ------------------------------------------------------------------ schema

TEST_CASE("launch profile: the measured profile parses into typed fields") {
    const LaunchProfile p = parse_single(kMeasuredProfile);
    CHECK(p.name == "qwen3.8-27b-q3-100k");
    CHECK(p.backend == "llamaserver");
    CHECK(p.ctx_size == 100096u);
    CHECK(p.n_gpu_layers == 999);
    CHECK(p.parallel == 1u);
    CHECK(p.kv_unified == false);
    CHECK(p.flash_attn == std::string("on"));
    CHECK(p.cache_type_k == std::string("q8_0"));
    CHECK(p.cache_type_v == std::string("q5_1"));
    CHECK(p.fit == false);
    CHECK(p.cache_ram_mib == 4096);
    CHECK(p.ctx_checkpoints == 4u);
    CHECK(p.checkpoint_min_step == 8192u);
    CHECK(p.context_shift == false);
    CHECK(p.jinja == true);
    CHECK(p.reasoning_format == std::string("deepseek"));
    CHECK(*p.sampling.temperature == doctest::Approx(1.0));
    CHECK(p.sampling.top_k == 20);
    CHECK(*p.sampling.min_p == doctest::Approx(0.0));
    CHECK_FALSE(p.sampling.presence_penalty.has_value());
    CHECK_FALSE(p.threads.has_value());
    CHECK(p.mmproj.empty());
}

TEST_CASE("launch profile: flash_attn accepts the llamacpp backend's synonyms") {
    CHECK(parse_single(R"({"name":"a","backend":"llamaserver","model":"m.gguf","flash_attn":"enabled"})").flash_attn ==
          std::string("on"));
    CHECK(parse_single(R"({"name":"a","backend":"llamaserver","model":"m.gguf","flash_attn":"disabled"})").flash_attn ==
          std::string("off"));
    CHECK(parse_single(R"({"name":"a","backend":"llamaserver","model":"m.gguf","flash_attn":"auto"})").flash_attn ==
          std::string("auto"));
}

TEST_CASE("launch profile: schema and range violations are rejected") {
    const std::string base = R"("name":"p","backend":"llamaserver","model":"m.gguf")";
    CHECK(contains(parse_error("{" + base + R"(,"ctx":100})"), "unknown field 'ctx'"));
    CHECK(contains(parse_error(R"({"backend":"llamaserver","model":"m.gguf"})"), "'name' is required"));
    CHECK(contains(parse_error(R"({"name":"p","model":"m.gguf"})"), "'backend' is required"));
    CHECK(contains(parse_error(R"({"name":"p","backend":"ollama","model":"m.gguf"})"), "llamaserver or llamacpp"));
    CHECK(contains(parse_error(R"({"name":"default","backend":"llamaserver","model":"m.gguf"})"), "'name'"));
    CHECK(contains(parse_error(R"({"name":"a b","backend":"llamaserver","model":"m.gguf"})"), "'name'"));
    CHECK(contains(parse_error("{" + base + R"(,"ctx_size":"100"})"), "'ctx_size' must be an integer"));
    CHECK(contains(parse_error("{" + base + R"(,"ctx_size":16777217})"), "'ctx_size'"));
    CHECK(contains(parse_error("{" + base + R"(,"n_gpu_layers":-2})"), "'n_gpu_layers'"));
    CHECK(contains(parse_error("{" + base + R"(,"parallel":0})"), "'parallel'"));
    CHECK(contains(parse_error("{" + base + R"(,"kv_unified":1})"), "'kv_unified' must be true or false"));
    CHECK(contains(parse_error("{" + base + R"(,"flash_attn":"yes"})"), "on, off or auto"));
    CHECK(contains(parse_error("{" + base + R"(,"cache_type_k":"q3_k"})"), "'cache_type_k' must be one of"));
    CHECK(contains(parse_error("{" + base + R"(,"cache_type_v":"q8"})"), "'cache_type_v' must be one of"));
    CHECK(contains(parse_error("{" + base + R"(,"cache_ram_mib":-2})"), "'cache_ram_mib'"));
    CHECK(contains(parse_error("{" + base + R"(,"reasoning_format":"hidden"})"), "'reasoning_format'"));
    CHECK(contains(parse_error("{" + base + R"(,"image_max_tokens":2400})"), "need 'mmproj'"));
    CHECK(contains(parse_error("{" + base + R"(,"mmproj_offload":false})"), "need 'mmproj'"));
    CHECK(contains(parse_error("{" + base + R"(,"speculative":{"types":[]}})"), "must not be empty"));
    CHECK(contains(parse_error("{" + base + R"(,"speculative":{"types":["draft-magic"]}})"), "unknown speculative type"));
    CHECK(contains(parse_error("{" + base + R"(,"speculative":{"types":["none","draft-mtp"]}})"), "cannot be combined"));
    CHECK(contains(parse_error("{" + base + R"(,"speculative":{"types":["draft-mtp"],"n_max":3}})"), "unknown field"));
    CHECK(contains(parse_error("{" + base + R"(,"speculative":{"types":["draft-mtp"],"draft_n_max":0}})"),
                   "'draft_n_max'"));
    CHECK(contains(parse_error("{" + base + R"(,"sampling":{"temperature":11}})"), "sampling:"));
    CHECK(contains(parse_error("{" + base + R"(,"sampling":{"top_p":0}})"), "sampling:"));
    CHECK(contains(parse_error("{" + base + R"(,"sampling":{"typical_p":0.9}})"), "unknown field 'typical_p'"));
    CHECK(contains(parse_error("{" + base + R"(,"vram_budget_mib":0})"), "'vram_budget_mib'"));
}

TEST_CASE("launch profile: a quantized V cache requires flash attention") {
    const std::string base = R"("name":"p","backend":"llamaserver","model":"m.gguf")";
    CHECK(contains(parse_error("{" + base + R"(,"cache_type_v":"q5_1","flash_attn":"off"})"),
                   "requires flash attention"));
    // auto lets llama.cpp enable it; a quantized K cache alone is fine.
    CHECK(parse_single("{" + base + R"(,"cache_type_v":"q5_1","flash_attn":"auto"})").cache_type_v ==
          std::string("q5_1"));
    CHECK(parse_single("{" + base + R"(,"cache_type_v":"q8_0"})").cache_type_v == std::string("q8_0"));
    CHECK(parse_single("{" + base + R"(,"cache_type_k":"q4_0","flash_attn":"off"})").cache_type_k ==
          std::string("q4_0"));
    CHECK(parse_single("{" + base + R"(,"cache_type_v":"f16","flash_attn":"off"})").flash_attn == std::string("off"));
}

TEST_CASE("launch profile: ubatch_size must not exceed batch_size (upstream defaults apply)") {
    CHECK(contains(parse_error(R"({"name":"p","backend":"llamaserver","model":"m","batch_size":256,"ubatch_size":512})"),
                   "must not exceed"));
    // llama-server's default batch is 2048, the llamacpp backend's 512.
    CHECK(parse_single(R"({"name":"p","backend":"llamaserver","model":"m","ubatch_size":1024})").ubatch_size == 1024u);
    CHECK(contains(parse_error(R"({"name":"p","backend":"llamacpp","model":"m","ubatch_size":1024})"),
                   "must not exceed 'batch_size' (512)"));
}

TEST_CASE("launch profile: extra_args refuse host and port and credentials and downloads and typed flags") {
    const auto with = [](const std::string& arg) {
        return R"({"name":"p","backend":"llamaserver","model":"m.gguf","extra_args":[)" + arg + "]}";
    };
    for (const char* forbidden : {R"("--host","0.0.0.0")", R"("--port=9000")", R"("--api-key","k")",
                                  R"("--api-key-file","f")", R"("-hf","org/model")", R"("--model-url","u")",
                                  R"("--")", R"("--host_x")", R"("--models-dir","d")"}) {
        CAPTURE(forbidden);
        CHECK(contains(parse_error(with(forbidden)), "must not contain"));
    }
    for (const auto& [flag, field] : std::vector<std::pair<std::string, std::string>>{
             {R"("-c","4096")", "ctx_size"},
             {R"("--ctx-size=4096")", "ctx_size"},
             {R"("-ngl","99")", "n_gpu_layers"},
             {R"("--cache-type-v","q8_0")", "cache_type_v"},
             {R"("--no-kv-unified")", "kv_unified"},
             {R"("--temp","0.7")", "sampling.temperature"},
             {R"("-m","other.gguf")", "model"},
             {R"("--alias","x")", "name"},
             {R"("--spec-type","draft-mtp")", "speculative.types"}}) {
        CAPTURE(flag);
        const std::string message = parse_error(with(flag));
        CHECK(contains(message, "use the typed field '" + field + "'"));
    }
    const LaunchProfile ok = parse_single(with(R"("--no-warmup","--override-kv","a=int:1")"));
    CHECK(ok.extra_args == std::vector<std::string>{"--no-warmup", "--override-kv", "a=int:1"});
}

TEST_CASE("launch profile: extra_args refuse llama-server agent tools, MCP and file serving") {
    const auto with = [](const std::string& arg) {
        return R"({"name":"p","backend":"llamaserver","model":"m.gguf","extra_args":[)" + arg + "]}";
    };
    for (const char* unsafe :
         {R"("--tools","all")", R"("--tools=exec_shell_command")", R"("--tools-runtime","ssh:host")",
          R"("--mcp-servers-config","m.json")", R"("--mcp-servers-json","{}")", R"("-ag")", R"("--agent")",
          R"("--ui-mcp-proxy")", R"("--webui-mcp-proxy")", R"("--path","C:/")", R"("--media-path","D:/")"}) {
        CAPTURE(unsafe);
        CHECK(contains(parse_error(with(unsafe)), "agent tools, MCP and local file serving"));
    }
    // The disabling spellings stay allowed.
    const LaunchProfile ok =
        parse_single(with(R"("--no-agent","-no-ag","--no-ui-mcp-proxy","--no-webui-mcp-proxy")"));
    CHECK(ok.extra_args.size() == 4u);
}

TEST_CASE("launch profile: file-level validation") {
    const auto error = [](const std::string& text) {
        auto doc = json::parse(text);
        REQUIRE(doc.ok());
        auto parsed = si::parse_launch_profiles(doc.value());
        REQUIRE_FALSE(parsed.ok());
        return parsed.status().message();
    };
    CHECK(contains(error("[]"), "JSON object"));
    CHECK(contains(error(R"({"profiles":[]})"), "non-empty"));
    CHECK(contains(error(R"({"profiles":[],"x":1})"), "unknown field 'x'"));
    CHECK(contains(error(R"({"profiles":[{"name":"a","backend":"llamacpp","model":"m"},)"
                         R"({"name":"a","backend":"llamaserver","model":"n"}]})"),
                   "more than one profile"));
    CHECK(contains(error(R"({"profiles":[5]})"), "#0: must be a JSON object"));
}

TEST_CASE("launch profile: load_launch_profiles reads a file and reports a missing one") {
    const auto path = std::filesystem::temp_directory_path() / (si::make_id("sonder-profiles") + ".json");
    {
        std::ofstream out(path, std::ios::binary);
        out << "{\"profiles\":[" << kMeasuredProfile << "]}";
    }
    auto loaded = si::load_launch_profiles(path.string());
    REQUIRE(loaded.ok());
    CHECK(si::find_launch_profile(loaded.value(), "qwen3.8-27b-q3-100k") != nullptr);
    CHECK(si::find_launch_profile(loaded.value(), "other") == nullptr);
    std::filesystem::remove(path);
    CHECK(si::load_launch_profiles(path.string()).status().code() == si::ErrorCode::not_found);
}

// ------------------------------------------------------------ llama-server argv

TEST_CASE("launch profile argv: the measured 100k profile (golden)") {
    auto argv = si::llamaserver_arguments(parse_single(kMeasuredProfile));
    REQUIRE(argv.ok());
    const std::vector<std::string> expected{
        "--model", "D:/models/Qwen3.8-27B-UD-Q3_K_XL.gguf", "--alias", "qwen3.8-27b-q3-100k",
        "--ctx-size", "100096", "--n-gpu-layers", "999", "--batch-size", "2048", "--ubatch-size", "512",
        "--parallel", "1", "--no-kv-unified", "--flash-attn", "on", "--cache-type-k", "q8_0",
        "--cache-type-v", "q5_1", "--ctx-checkpoints", "4", "--checkpoint-min-step", "8192",
        "--cache-ram", "4096", "--no-context-shift", "--fit", "off", "--jinja", "--reasoning-format", "deepseek"};
    CHECK(argv.value() == expected);
    // Sampling defaults are applied per request, not as server arguments.
    for (const auto& a : argv.value()) {
        CHECK(a.rfind("--temp", 0) != 0);
        CHECK(a.rfind("--top-", 0) != 0);
    }
}

TEST_CASE("launch profile argv: vision and speculative and threads and every toggle (golden)") {
    const LaunchProfile p = parse_single(R"({
        "name": "qwen-iq4xs-vision", "backend": "llamaserver", "model": "m.gguf",
        "mmproj": "mmproj-F16.gguf", "mmproj_offload": false, "image_max_tokens": 2400,
        "n_gpu_layers": -1, "kv_unified": true, "context_shift": true, "fit": true, "jinja": false,
        "speculative": {"types": ["draft-mtp", "ngram-mod"], "draft_n_max": 3, "draft_model": "draft.gguf"},
        "threads": 8, "threads_batch": 12, "extra_args": ["--no-warmup"]
    })");
    auto argv = si::llamaserver_arguments(p);
    REQUIRE(argv.ok());
    const std::vector<std::string> expected{
        "--model", "m.gguf", "--alias", "qwen-iq4xs-vision", "--mmproj", "mmproj-F16.gguf", "--no-mmproj-offload",
        "--image-max-tokens", "2400", "--n-gpu-layers", "all", "--kv-unified", "--context-shift", "--fit", "on",
        "--spec-type", "draft-mtp,ngram-mod", "--spec-draft-n-max", "3", "--spec-draft-model", "draft.gguf",
        "--threads", "8", "--threads-batch", "12", "--no-jinja", "--no-warmup"};
    CHECK(argv.value() == expected);
}

TEST_CASE("launch profile argv: a minimal profile passes only model and alias") {
    auto argv = si::llamaserver_arguments(parse_single(R"({"name":"m1","backend":"llamaserver","model":"a b.gguf"})"));
    REQUIRE(argv.ok());
    CHECK(argv.value() == std::vector<std::string>{"--model", "a b.gguf", "--alias", "m1"});
    auto wrong = si::llamaserver_arguments(parse_single(R"({"name":"m1","backend":"llamacpp","model":"a.gguf"})"));
    CHECK_FALSE(wrong.ok());
}

// ------------------------------------------------------- direct llamacpp backend

TEST_CASE("launch profile llamacpp: context and GPU layers map onto the backend setup") {
    const LaunchProfile p =
        parse_single(R"({"name":"direct","backend":"llamacpp","model":"D:/m.gguf","ctx_size":8192,"n_gpu_layers":-1,
                         "sampling":{"temperature":0.6}})");
    si::BackendSetup setup;
    setup.backend = "llamacpp";
    std::string device;
    REQUIRE(si::apply_llamacpp_profile(p, setup, device).ok());
    CHECK(setup.llamacpp_context_length == 8192u);
    CHECK(setup.llamacpp_gpu_layers == -1);
    CHECK(device == "gpu:0");  // n_gpu_layers is honoured only on a gpu:* device

    std::string explicit_device = "gpu:1";
    si::BackendSetup other;
    REQUIRE(si::apply_llamacpp_profile(p, other, explicit_device).ok());
    CHECK(explicit_device == "gpu:1");

    std::string cpu = "cpu:0";
    si::BackendSetup untouched;
    const si::Status refused = si::apply_llamacpp_profile(p, untouched, cpu);
    CHECK_FALSE(refused.ok());
    CHECK(contains(refused.message(), "needs a gpu:* --device"));
    CHECK_FALSE(untouched.llamacpp_context_length.has_value());  // nothing applied on failure

    const LaunchProfile cpu_only =
        parse_single(R"({"name":"direct","backend":"llamacpp","model":"D:/m.gguf","n_gpu_layers":0})");
    std::string cpu_device = "cpu:0";
    si::BackendSetup s2;
    REQUIRE(si::apply_llamacpp_profile(cpu_only, s2, cpu_device).ok());
    CHECK(s2.llamacpp_gpu_layers == 0);
    CHECK(cpu_device == "cpu:0");
}

TEST_CASE("launch profile llamacpp: fields the direct backend cannot honour are rejected and never dropped") {
    const std::vector<std::pair<std::string, std::string>> cases{
        {R"("mmproj":"p.gguf")", "mmproj"},
        {R"("mmproj":"p.gguf","image_max_tokens":100)", "mmproj"},
        {R"("parallel":2)", "parallel"},
        {R"("kv_unified":false)", "kv_unified"},
        {R"("ctx_checkpoints":4)", "ctx_checkpoints"},
        {R"("checkpoint_min_step":8192)", "checkpoint_min_step"},
        {R"("cache_ram_mib":4096)", "cache_ram_mib"},
        {R"("context_shift":false)", "context_shift"},
        {R"("fit":false)", "fit"},
        {R"("speculative":{"types":["draft-mtp"]})", "speculative"},
        {R"("threads":8)", "threads"},
        {R"("threads_batch":8)", "threads_batch"},
        {R"("jinja":true)", "jinja"},
        {R"("reasoning_format":"deepseek")", "reasoning_format"},
        {R"("extra_args":["--no-warmup"])", "extra_args"},
    };
    for (const auto& [fields, name] : cases) {
        CAPTURE(fields);
        const LaunchProfile p = parse_single(R"({"name":"d","backend":"llamacpp","model":"m.gguf",)" + fields + "}");
        si::BackendSetup setup;
        std::string device;
        const si::Status st = si::apply_llamacpp_profile(p, setup, device);
        REQUIRE_FALSE(st.ok());
        CHECK(st.code() == si::ErrorCode::invalid_argument);
        CHECK(contains(st.message(), "'" + name + "' is not supported by the llamacpp backend; use llamaserver"));
    }
}

TEST_CASE("launch profile llamacpp: KV cache types and flash attention and batch use the backend's option names") {
    const LaunchProfile p = parse_single(
        R"({"name":"d","backend":"llamacpp","model":"m.gguf","batch_size":1024,"ubatch_size":256,
            "cache_type_k":"q8_0","cache_type_v":"q8_0","flash_attn":"on"})");
    si::BackendSetup setup;
    std::string device;
    const si::Status st = si::apply_llamacpp_profile(p, setup, device);
    if constexpr (si::kLlamaCppContextOptionsAvailable) {
        // The llamacpp backend's own fields (kv_cache_type_k/_v,
        // flash_attention, batch/ubatch) carry the values unchanged.
        REQUIRE(st.ok());
    } else {
        REQUIRE_FALSE(st.ok());
        CHECK(contains(st.message(), "'batch_size' is not supported by the llamacpp backend; use llamaserver"));
        for (const char* field : {"ubatch_size", "cache_type_k", "cache_type_v", "flash_attn"}) {
            CAPTURE(field);
            const LaunchProfile one =
                parse_single(std::string(R"({"name":"d","backend":"llamacpp","model":"m.gguf",")") + field +
                             (std::string(field) == "ubatch_size" ? R"(":256})"
                              : std::string(field) == "flash_attn" ? R"(":"on"})"
                                                                    : R"(":"q8_0"})"));
            si::BackendSetup s;
            std::string d;
            const si::Status one_st = si::apply_llamacpp_profile(one, s, d);
            REQUIRE_FALSE(one_st.ok());
            CHECK(contains(one_st.message(), std::string("'") + field + "' is not supported by the llamacpp backend"));
        }
    }
}

// ------------------------------------------------------------- sampling defaults

TEST_CASE("launch profile: sampling defaults fill only fields the request left unset") {
    si::SamplingDefaults d;
    d.temperature = 1.0f;
    d.top_p = 0.95f;
    d.top_k = 20;
    d.min_p = 0.0f;
    si::SamplingConfig s;
    s.explicit_only = true;
    s.temperature = 0.2f;
    s.explicit_fields = si::SamplingConfig::kTemperature;
    si::apply_sampling_defaults(d, s);
    CHECK(s.temperature == doctest::Approx(0.2));  // the caller's value wins
    CHECK(s.top_p == doctest::Approx(0.95));
    CHECK(s.top_k == 20);
    CHECK(s.min_p == doctest::Approx(0.0));
    CHECK(s.is_explicit(si::SamplingConfig::kTopK));
    CHECK(s.is_explicit(si::SamplingConfig::kMinP));  // sent even though it equals a local default
    CHECK_FALSE(s.is_explicit(si::SamplingConfig::kPresencePenalty));
    CHECK_FALSE(s.is_explicit(si::SamplingConfig::kRepeatPenalty));
    CHECK(s.explicit_only);

    si::SamplingConfig untouched;
    si::apply_sampling_defaults(si::SamplingDefaults{}, untouched);
    CHECK(untouched.explicit_fields == 0u);
}

// --------------------------------------------------------------- GGUF reader

namespace {

// Minimal GGUF v3 writer for header-only fixtures.
class GgufWriter {
public:
    void u32(std::uint32_t v) { raw(&v, 4); }
    void u64(std::uint64_t v) { raw(&v, 8); }
    void str(const std::string& s) {
        u64(s.size());
        bytes_.append(s);
    }
    void kv_u32(const std::string& key, std::uint32_t v) {
        str(key);
        u32(4);
        u32(v);
    }
    void kv_str(const std::string& key, const std::string& v) {
        str(key);
        u32(8);
        str(v);
    }
    void kv_i32_array(const std::string& key, const std::vector<std::int32_t>& values) {
        str(key);
        u32(9);
        u32(5);
        u64(values.size());
        for (auto v : values) raw(&v, 4);
    }
    void kv_str_array(const std::string& key, const std::vector<std::string>& values) {
        str(key);
        u32(9);
        u32(8);
        u64(values.size());
        for (const auto& v : values) str(v);
    }
    void tensor(const std::string& name, const std::vector<std::uint64_t>& dims, std::uint32_t type) {
        str(name);
        u32(static_cast<std::uint32_t>(dims.size()));
        for (auto d : dims) u64(d);
        u32(type);
        u64(0);
    }
    std::string finish(std::uint64_t tensors, std::uint64_t kvs) const {
        std::string head = "GGUF";
        const auto put = [&head](const void* p, std::size_t n) { head.append(static_cast<const char*>(p), n); };
        const std::uint32_t version = 3;
        put(&version, 4);
        put(&tensors, 8);
        put(&kvs, 8);
        return head + bytes_;
    }

private:
    void raw(const void* p, std::size_t n) { bytes_.append(static_cast<const char*>(p), n); }
    std::string bytes_;
};

// A four-block hybrid: blocks 1 and 3 carry attention (attn_k tensors),
// blocks 0 and 2 are delta-net layers.
std::string hybrid_fixture() {
    GgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_u32("qwen35.block_count", 4);
    w.kv_u32("qwen35.embedding_length", 64);
    w.kv_u32("qwen35.context_length", 1000);
    w.kv_u32("qwen35.attention.head_count", 8);
    w.kv_u32("qwen35.attention.head_count_kv", 2);
    w.kv_u32("qwen35.attention.key_length", 16);
    w.kv_u32("qwen35.attention.value_length", 32);
    w.kv_u32("qwen35.full_attention_interval", 2);
    w.kv_u32("qwen35.ssm.conv_kernel", 4);
    w.kv_u32("qwen35.ssm.inner_size", 32);
    w.kv_u32("qwen35.ssm.state_size", 8);
    w.kv_u32("qwen35.ssm.group_count", 2);
    w.kv_str_array("tokenizer.ggml.tokens", {"a", "b", "c", "d", "e"});
    w.tensor("token_embd.weight", {64, 5}, 1);        // f16: 640 B
    w.tensor("blk.0.attn_qkv.weight", {64, 64}, 1);   // 8192 B (delta-net, not attention KV)
    w.tensor("blk.1.attn_k.weight", {64, 32}, 0);     // f32: 8192 B
    w.tensor("blk.1.ffn_up.weight", {64, 64}, 8);     // q8_0: 4096/32*34 = 4352 B
    w.tensor("blk.2.ssm_out.weight", {32, 64}, 1);    // 4096 B
    w.tensor("blk.3.attn_k.weight", {64, 32}, 1);     // 4096 B
    w.tensor("output_norm.weight", {64}, 0);          // 256 B
    w.tensor("output.weight", {64, 5}, 1);            // 640 B
    return w.finish(8, 14);
}

si::GgufModelInfo parse_fixture(const std::string& bytes) {
    auto info = si::parse_gguf_model_info(bytes);
    REQUIRE_MESSAGE(info.ok(), info.status().message());
    return info.value();
}

}  // namespace

TEST_CASE("GGUF reader: metadata and tensor sizes and hybrid attention layers") {
    const si::GgufModelInfo m = parse_fixture(hybrid_fixture());
    CHECK(m.architecture == "qwen35");
    CHECK(m.kind == si::ModelArchitecture::hybrid);
    CHECK(m.block_count == 4u);
    CHECK(m.embedding_length == 64u);
    CHECK(m.vocab_size == 5u);
    CHECK(m.key_length == 16u);
    CHECK(m.value_length == 32u);
    CHECK(m.kv_heads == std::vector<std::uint32_t>{0, 2, 0, 2});  // attn_k tensors decide
    CHECK(m.block_bytes == std::vector<std::uint64_t>{8192, 8192 + 4352, 4096, 4096});
    CHECK(m.embedding_bytes == 640u);
    CHECK(m.output_bytes == 256u + 640u);
    // (4 - 1) x (32 + 2 x 2 x 8) + 32 x 8
    CHECK(m.recurrent_state_values == 3u * 64u + 256u);
}

TEST_CASE("GGUF reader: full_attention_interval and per-layer head counts") {
    GgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_u32("qwen35.block_count", 8);
    w.kv_u32("qwen35.embedding_length", 64);
    w.kv_u32("qwen35.attention.head_count", 4);
    w.kv_u32("qwen35.full_attention_interval", 4);
    w.kv_u32("qwen35.ssm.conv_kernel", 4);
    const si::GgufModelInfo by_interval = parse_fixture(w.finish(0, 6));
    CHECK(by_interval.kv_heads == std::vector<std::uint32_t>{0, 0, 0, 4, 0, 0, 0, 4});
    CHECK(by_interval.key_length == 16u);  // embedding / heads

    GgufWriter a;
    a.kv_str("general.architecture", "lfm2");
    a.kv_u32("lfm2.block_count", 4);
    a.kv_u32("lfm2.embedding_length", 32);
    a.kv_i32_array("lfm2.attention.head_count_kv", {0, 8, 0, 4});
    const si::GgufModelInfo per_layer = parse_fixture(a.finish(0, 4));
    CHECK(per_layer.kv_heads == std::vector<std::uint32_t>{0, 8, 0, 4});

    GgufWriter dense;
    dense.kv_str("general.architecture", "llama");
    dense.kv_u32("llama.block_count", 3);
    dense.kv_u32("llama.embedding_length", 128);
    dense.kv_u32("llama.attention.head_count", 8);
    dense.kv_u32("llama.attention.head_count_kv", 2);
    const si::GgufModelInfo d = parse_fixture(dense.finish(0, 5));
    CHECK(d.kind == si::ModelArchitecture::attention_only);
    CHECK(d.kv_heads == std::vector<std::uint32_t>{2, 2, 2});
}

TEST_CASE("GGUF reader: malformed and truncated headers are rejected") {
    CHECK_FALSE(si::parse_gguf_model_info("GGML").ok());
    CHECK_FALSE(si::parse_gguf_model_info("").ok());
    const std::string good = hybrid_fixture();
    for (std::size_t cut : {std::size_t{5}, std::size_t{30}, good.size() / 2, good.size() - 1}) {
        CAPTURE(cut);
        CHECK(si::parse_gguf_model_info(good.substr(0, cut)).status().code() == si::ErrorCode::invalid_argument);
    }
    GgufWriter unknown;
    unknown.kv_str("general.architecture", "llama");
    unknown.kv_u32("llama.block_count", 1);
    unknown.tensor("blk.0.x", {32}, 99);
    CHECK(contains(si::parse_gguf_model_info(unknown.finish(1, 2)).status().message(), "unknown ggml type"));
    GgufWriter out_of_range;
    out_of_range.kv_str("general.architecture", "llama");
    out_of_range.kv_u32("llama.block_count", 1);
    out_of_range.tensor("blk.7.x", {32}, 0);
    CHECK(contains(si::parse_gguf_model_info(out_of_range.finish(1, 2)).status().message(), "unknown block"));
    CHECK(si::read_gguf_model_info("definitely-missing-model.gguf").status().code() == si::ErrorCode::not_found);
}

// ------------------------------------------------------------------ estimate

namespace {

// Qwen3.8-27B UD-Q3_K_XL (the owner's file, 13,146,393,504 bytes), from its
// GGUF header: 65 blocks (the last is the MTP block), attention every 4th
// block (16 + the MTP block), 4 KV heads of 256, vocab 248320.
si::GgufModelInfo qwen38_q3() {
    si::GgufModelInfo m;
    m.architecture = "qwen35";
    m.kind = si::ModelArchitecture::hybrid;
    m.block_count = 65;
    m.nextn_layers = 1;
    m.embedding_length = 5120;
    m.vocab_size = 248320;
    m.key_length = 256;
    m.value_length = 256;
    m.kv_heads.assign(65, 0);
    for (std::uint32_t i = 3; i < 64; i += 4) m.kv_heads[i] = 4;
    m.kv_heads[64] = 4;
    // 11,714,985,984 block bytes in total; the MTP block holds 351,008,768.
    m.block_bytes.assign(65, 0);
    const std::uint64_t regular = 11714985984ull - 351008768ull;
    for (std::uint32_t i = 0; i < 64; ++i) m.block_bytes[i] = regular / 64;
    m.block_bytes[0] += regular % 64;
    m.block_bytes[64] = 351008768ull;
    m.output_bytes = 874086400ull + 20480ull;
    m.embedding_bytes = 546304000ull;
    m.recurrent_state_values = 3ull * (6144 + 2 * 16 * 128) + 6144ull * 128;
    m.metadata = {{"general.architecture", "qwen35"}, {"qwen35.context_length", "262144"}};
    return m;
}

}  // namespace

TEST_CASE("VRAM estimate: hybrid KV counts attention layers only (measured 100k profile)") {
    const si::VramEstimate e = si::estimate_vram(qwen38_q3(), parse_single(kMeasuredProfile));
    CHECK(e.gpu_layers == 64u);        // the MTP block is not loaded without draft-mtp
    CHECK(e.attention_layers == 16u);  // n_kv_heads x head_dim x (bytes k + bytes v) x 16 layers
    // 16 x 4 x 256 x (34/32 + 24/32) = 29,696 bytes per token
    CHECK(e.kv_bytes_per_token == 29696u);
    CHECK(e.context == 100096u);
    CHECK(e.sequences == 1u);
    CHECK(e.kv_bytes == 29696ull * 100096ull);
    CHECK(e.weights_bytes == 11714985984ull - 351008768ull + 874086400ull + 20480ull);
    CHECK(e.recurrent_bytes == 48ull * (3ull * 10240 + 786432) * 4);
    CHECK(e.compute_bytes == 248320ull * 512 * 4 + 4ull * 5120 * 512 * 4 + (256ull << 20));
    CHECK(e.total_bytes == e.weights_bytes + e.kv_bytes + e.recurrent_bytes + e.compute_bytes);
    // Measured on the RTX 5070 Ti: 15,266 MiB dedicated to the llama-server
    // process (bench-kv-fit probe). The estimate is within 2 %.
    const double error = (static_cast<double>(e.total_mib()) - 15266.0) / 15266.0;
    MESSAGE("estimate " << e.total_mib() << " MiB vs 15266 MiB measured: " << error * 100.0 << " %");
    CHECK(std::fabs(error) < 0.02);
}

TEST_CASE("VRAM estimate: parallel sequences and unified KV and MTP and partial offload") {
    const si::GgufModelInfo m = qwen38_q3();
    LaunchProfile p = parse_single(kMeasuredProfile);
    const si::VramEstimate one = si::estimate_vram(m, p);

    // kv_unified false: llama.cpp splits --ctx-size into one stream per
    // sequence of pad(ctx / parallel, 256) cells, so the KV stays about ctx
    // in total (100096 / 2 = 50048 -> 50176). Recurrent state is per sequence.
    p.parallel = 2;
    const si::VramEstimate two = si::estimate_vram(m, p);
    CHECK(two.sequences == 2u);
    CHECK(two.context == 100096u);
    CHECK(two.context_per_sequence == 50176u);
    CHECK(two.kv_bytes == 29696ull * 50176ull * 2);
    CHECK(two.kv_bytes < 2 * one.kv_bytes);  // not ctx x parallel
    CHECK(two.recurrent_bytes == 2 * one.recurrent_bytes);
    p.parallel = 4;  // 100096 / 4 = 25024 -> 25088
    const si::VramEstimate four = si::estimate_vram(m, p);
    CHECK(four.context_per_sequence == 25088u);
    CHECK(four.kv_bytes == 29696ull * 25088ull * 4);
    CHECK(four.recurrent_bytes == 4 * one.recurrent_bytes);
    CHECK(one.context_per_sequence == 100096u);
    p.parallel = 2;

    p.kv_unified = true;  // one shared pool of ctx tokens
    const si::VramEstimate unified = si::estimate_vram(m, p);
    CHECK(unified.kv_bytes == one.kv_bytes);
    CHECK(unified.recurrent_bytes == two.recurrent_bytes);

    LaunchProfile mtp = parse_single(kMeasuredProfile);
    mtp.speculative = si::SpeculativeSettings{{"draft-mtp"}, 3, ""};
    const si::VramEstimate with_mtp = si::estimate_vram(m, mtp);
    CHECK(with_mtp.attention_layers == 17u);
    CHECK(with_mtp.weights_bytes == one.weights_bytes + 351008768ull);

    LaunchProfile partial = parse_single(kMeasuredProfile);
    partial.n_gpu_layers = 20;  // llama.cpp offloads the last 20 blocks, not the output
    const si::VramEstimate p20 = si::estimate_vram(m, partial);
    CHECK(p20.gpu_layers == 19u);  // blocks 45..63 (block 64 is the unloaded MTP block)
    CHECK(p20.attention_layers == 5u);  // 47, 51, 55, 59, 63
    CHECK(p20.weights_bytes == 19 * m.block_bytes[1]);

    LaunchProfile none = parse_single(kMeasuredProfile);
    none.n_gpu_layers = 0;
    const si::VramEstimate cpu = si::estimate_vram(m, none);
    CHECK(cpu.total_bytes == 0u);

    LaunchProfile f16 = parse_single(kMeasuredProfile);
    f16.cache_type_k = "f16";
    f16.cache_type_v = "f16";
    CHECK(si::estimate_vram(m, f16).kv_bytes_per_token == 16u * 4 * 256 * 4);
}

TEST_CASE("VRAM estimate: attention-only models count every layer and ctx 0 uses the training context") {
    si::GgufModelInfo m;
    m.architecture = "llama";
    m.block_count = 4;
    m.embedding_length = 128;
    m.vocab_size = 1000;
    m.key_length = 64;
    m.value_length = 64;
    m.kv_heads = {8, 8, 8, 8};
    m.block_bytes = {100, 100, 100, 100};
    m.output_bytes = 50;
    m.metadata = {{"llama.context_length", "4000"}};
    LaunchProfile p = parse_single(R"({"name":"l","backend":"llamaserver","model":"m.gguf","ctx_size":0})");
    const si::VramEstimate e = si::estimate_vram(m, p);
    CHECK(e.attention_layers == 4u);
    CHECK(e.context == 4096u);  // 4000 padded to 256
    CHECK(e.kv_bytes == 4ull * 8 * (64 * 2 + 64 * 2) * 4096);
    CHECK(e.weights_bytes == 450u);
    CHECK(e.recurrent_bytes == 0u);
    CHECK_FALSE(e.notes.empty());
}

TEST_CASE("VRAM estimate: the GGUF fixture round-trips through the estimator") {
    const si::GgufModelInfo m = parse_fixture(hybrid_fixture());
    const LaunchProfile p =
        parse_single(R"({"name":"f","backend":"llamaserver","model":"m.gguf","ctx_size":512,"cache_type_k":"f16"})");
    const si::VramEstimate e = si::estimate_vram(m, p);
    CHECK(e.attention_layers == 2u);
    CHECK(e.kv_bytes_per_token == 2u * 2 * (16 * 2 + 32 * 2));
    CHECK(e.kv_bytes == e.kv_bytes_per_token * 512);
    CHECK(e.recurrent_bytes == 2u * (3u * 64 + 256) * 4);
}

// Opt-in: SONDER_TEST_GGUF=<path to a GGUF (a header-only copy is enough)>.
TEST_CASE("VRAM estimate: a real GGUF header (opt-in)") {
    const char* path = std::getenv("SONDER_TEST_GGUF");
    if (path == nullptr || *path == '\0') {
        MESSAGE("SONDER_TEST_GGUF not set; skipped");
        return;
    }
    auto info = si::read_gguf_model_info(path);
    REQUIRE_MESSAGE(info.ok(), info.status().message());
    // The measured profile's settings at several context sizes.
    for (const std::uint32_t ctx : {65536u, 73728u, 81920u, 100096u}) {
        LaunchProfile p = parse_single(kMeasuredProfile);
        p.ctx_size = ctx;
        const si::VramEstimate e = si::estimate_vram(info.value(), p);
        MESSAGE(path << ": " << info.value().architecture << " blocks " << info.value().block_count << ", ctx " << ctx
                     << ", attention " << e.attention_layers << ", weights " << (e.weights_bytes >> 20) << " MiB, KV "
                     << (e.kv_bytes >> 20) << " MiB, recurrent " << (e.recurrent_bytes >> 20) << " MiB, compute "
                     << (e.compute_bytes >> 20) << " MiB, total " << e.total_mib() << " MiB");
        CHECK(e.total_bytes > 0u);
    }
}

// --------------------------------------------------------------- /v1/models

namespace {

// Mock backend that records the model names it loads and the sampling each
// request carries (no token_logits, so requests reach generate()).
class RecordingBackend final : public si::Backend {
public:
    struct Log {
        std::mutex mu;
        std::vector<std::string> loads;
        std::vector<si::SamplingConfig> sampling;
    };
    std::shared_ptr<Log> log = std::make_shared<Log>();
    // Set before the server starts; reported through runtime_status().
    std::optional<si::BackendRuntimeStatus> runtime;

    [[nodiscard]] std::string name() const override { return inner_->name(); }
    [[nodiscard]] std::optional<si::BackendRuntimeStatus> runtime_status() const override { return runtime; }
    [[nodiscard]] std::string description() const override { return "mock that records requests"; }
    [[nodiscard]] si::BackendCapabilities capabilities() const override {
        si::BackendCapabilities c;
        c.add(si::Capability::streaming);
        return c;
    }
    si::Result<std::string> probe() override { return inner_->probe(); }
    si::Result<std::vector<si::ModelDescriptor>> list_models() override { return inner_->list_models(); }
    si::Result<std::shared_ptr<si::BackendModel>> load_model(const si::ModelLoadOptions& options) override {
        {
            std::lock_guard<std::mutex> lock(log->mu);
            log->loads.push_back(options.model);
        }
        auto inner = inner_->load_model(options);
        if (!inner.ok()) return inner.status();
        return std::shared_ptr<si::BackendModel>(std::make_shared<Model>(inner.value(), log));
    }

private:
    class Model final : public si::BackendModel {
    public:
        Model(std::shared_ptr<si::BackendModel> inner, std::shared_ptr<Log> log)
            : inner_(std::move(inner)), log_(std::move(log)) {}
        const si::ModelDescriptor& descriptor() const override { return inner_->descriptor(); }
        si::Result<si::GenerateStats> generate(const si::GenerateRequest& request, const si::CancellationToken& cancel,
                                               const si::TokenCallback& on_chunk) override {
            {
                std::lock_guard<std::mutex> lock(log_->mu);
                log_->sampling.push_back(request.sampling);
            }
            return inner_->generate(request, cancel, on_chunk);
        }

    private:
        std::shared_ptr<si::BackendModel> inner_;
        std::shared_ptr<Log> log_;
    };
    std::shared_ptr<si::Backend> inner_ = si::make_mock_backend();
};

std::string model_chat(const std::string& model, const std::string& extra = "") {
    return R"({"model":")" + model + R"(","messages":[{"role":"user","content":"hello"}])" + extra + "}";
}

}  // namespace

TEST_CASE("models: launch profile metadata is additive and sampling defaults reach the backend") {
    auto backend = std::make_shared<RecordingBackend>();
    auto o = Fixture::defaults();
    o.backend.backend.clear();
    o.backend_instance = backend;
    o.models = {"qwen-profile", "mock:plain"};
    LaunchProfile p = parse_single(kMeasuredProfile);
    p.name = "qwen-profile";
    si::VramEstimate estimate;
    estimate.total_bytes = 15000ull << 20;
    srv::ModelProfileBinding binding;
    binding.model = "qwen-profile";
    binding.load_name = "mock:weights.gguf";
    binding.sampling_defaults = p.sampling;
    binding.metadata = si::launch_profile_metadata(p, estimate, 15500);
    o.profile_bindings.push_back(binding);
    json::Object catalog_entry = binding.metadata;
    catalog_entry.set("served", true);
    o.profile_catalog.emplace_back(catalog_entry);
    // "qwen-profile" does not start with "mock": the mock only loads it under
    // the bound load name, which proves the binding is used.
    Fixture f(o);
    {
        std::lock_guard<std::mutex> lock(backend->log->mu);
        CHECK(backend->log->loads == std::vector<std::string>{"mock:weights.gguf", "mock:plain"});
    }

    const Reply r = get(f.port, "/v1/models");
    REQUIRE(r.status == 200);
    const json::Value doc = r.json();
    const auto& data = doc.find("data")->as_array();
    REQUIRE(data.size() == 2);
    CHECK(data[0].find("id")->as_string() == "qwen-profile");
    CHECK(data[0].find("object")->as_string() == "model");
    CHECK(data[0].find("owned_by")->as_string() == "sonder-inference");
    const json::Value* ext = data[0].find("sonder");
    CHECK(ext->find("backend")->as_string() == "mock");
    CHECK(ext->find("default")->as_bool());
    CHECK(ext->find("synthetic")->as_bool());
    const json::Value* prof = ext->find("profile");
    REQUIRE(prof != nullptr);
    CHECK(prof->find("name")->as_string() == "qwen-profile");
    CHECK(prof->find("backend")->as_string() == "llamaserver");
    CHECK(prof->find("context_length")->as_int() == 100096);
    CHECK(prof->find("parallel")->as_int() == 1);
    CHECK(prof->find("cache_type_k")->as_string() == "q8_0");
    CHECK(prof->find("cache_type_v")->as_string() == "q5_1");
    CHECK(prof->find("flash_attn")->as_string() == "on");
    CHECK(prof->find("estimated_vram_mib")->as_int() == 15000);
    CHECK(prof->find("vram_budget_mib")->as_int() == 15500);
    // Sonder's chat route rejects tool definitions: "tools" is an upstream
    // capability only, never an endpoint capability.
    CHECK(prof->find("capabilities")->as_array().empty());  // no speculative
    const auto& upstream = prof->find("upstream_capabilities")->as_array();
    REQUIRE(upstream.size() == 1);
    CHECK(upstream[0].as_string() == "tools");  // no mmproj
    CHECK(prof->find("configured_context_length") == nullptr);  // no runtime context fit
    CHECK(data[1].find("sonder")->find("profile") == nullptr);
    const auto& profiles = doc.find("sonder")->find("profiles")->as_array();
    REQUIRE(profiles.size() == 1);
    CHECK(profiles[0].find("served")->as_bool());
    CHECK(doc.find("sonder")->find("api_version")->as_int() == 1);

    // No sampling fields: the profile's defaults are sent as explicit values.
    REQUIRE(post(f.port, "/v1/chat/completions", model_chat("qwen-profile")).status == 200);
    // The caller's temperature wins over the profile's.
    REQUIRE(post(f.port, "/v1/chat/completions", model_chat("qwen-profile", R"(,"temperature":0.2)")).status ==
            200);
    // A model without a profile is unchanged.
    REQUIRE(post(f.port, "/v1/chat/completions", model_chat("mock:plain")).status == 200);
    std::lock_guard<std::mutex> lock(backend->log->mu);
    REQUIRE(backend->log->sampling.size() == 3);
    const si::SamplingConfig& d = backend->log->sampling[0];
    CHECK(d.temperature == doctest::Approx(1.0));
    CHECK(d.top_k == 20);
    CHECK(d.top_p == doctest::Approx(0.95));
    CHECK(d.min_p == doctest::Approx(0.0));
    CHECK(d.is_explicit(si::SamplingConfig::kTemperature));
    CHECK(d.is_explicit(si::SamplingConfig::kTopK));
    CHECK(d.is_explicit(si::SamplingConfig::kMinP));
    CHECK_FALSE(d.is_explicit(si::SamplingConfig::kPresencePenalty));
    const si::SamplingConfig& c = backend->log->sampling[1];
    CHECK(c.temperature == doctest::Approx(0.2));
    CHECK(c.top_k == 20);
    const si::SamplingConfig& plain = backend->log->sampling[2];
    CHECK_FALSE(plain.is_explicit(si::SamplingConfig::kTemperature));
    CHECK_FALSE(plain.is_explicit(si::SamplingConfig::kTopK));
    CHECK(plain.top_k == si::SamplingConfig{}.top_k);
}

TEST_CASE("launch profile metadata: endpoint vs upstream capabilities") {
    LaunchProfile p = parse_single(R"({"name":"v","backend":"llamaserver","model":"m.gguf","mmproj":"mm.gguf",
        "speculative":{"types":["draft-mtp"]}})");
    json::Object meta = si::launch_profile_metadata(p, std::nullopt, std::nullopt);
    const auto& caps = meta.find("capabilities")->as_array();
    REQUIRE(caps.size() == 1);
    CHECK(caps[0].as_string() == "speculative");
    const auto& upstream = meta.find("upstream_capabilities")->as_array();
    REQUIRE(upstream.size() == 3);
    CHECK(upstream[0].as_string() == "vision");
    CHECK(upstream[1].as_string() == "tools");
    CHECK(upstream[2].as_string() == "speculative");
    CHECK(meta.find("estimated_vram_mib")->is_null());
    p.jinja = false;
    p.speculative.reset();
    meta = si::launch_profile_metadata(p, std::nullopt, std::nullopt);
    CHECK(meta.find("capabilities")->as_array().empty());
    const auto& vision_only = meta.find("upstream_capabilities")->as_array();
    REQUIRE(vision_only.size() == 1);
    CHECK(vision_only[0].as_string() == "vision");
}

namespace {

// A server with one launch-profile model on a backend that reports `rt`.
struct RuntimeProfileFixture {
    std::shared_ptr<RecordingBackend> backend = std::make_shared<RecordingBackend>();
    std::unique_ptr<Fixture> fixture;

    explicit RuntimeProfileFixture(const si::BackendRuntimeStatus& rt) {
        backend->runtime = rt;  // before the server starts
        auto o = Fixture::defaults();
        o.backend.backend.clear();
        o.backend_instance = backend;
        o.models = {"qwen-profile"};
        LaunchProfile p = parse_single(kMeasuredProfile);
        p.name = "qwen-profile";
        si::VramEstimate estimate;
        estimate.total_bytes = 15000ull << 20;
        srv::ModelProfileBinding binding;
        binding.model = "qwen-profile";
        binding.load_name = "mock:weights.gguf";
        binding.metadata = si::launch_profile_metadata(p, estimate, 15500);
        o.profile_bindings.push_back(binding);
        json::Object served_entry = binding.metadata;
        served_entry.set("served", true);
        o.profile_catalog.emplace_back(served_entry);
        json::Object other_entry = binding.metadata;
        other_entry.set("name", "other");
        other_entry.set("served", false);
        o.profile_catalog.emplace_back(other_entry);
        fixture = std::make_unique<Fixture>(o);
    }
};

}  // namespace

TEST_CASE("models: a context reduced at run time (spill guard auto_fit) replaces the profile's context_length") {
    si::BackendRuntimeStatus rt;
    rt.context.policy = "auto_fit";
    rt.context.configured_ctx = 100096;
    rt.context.fitted_ctx = 84992;
    rt.context.fit_attempts = 1;
    rt.context.outcome = "fitted";
    RuntimeProfileFixture f(rt);

    const Reply r = get(f.fixture->port, "/v1/models");
    REQUIRE(r.status == 200);
    const json::Value doc = r.json();
    const auto& data = doc.find("data")->as_array();
    REQUIRE(data.size() == 1);
    const json::Value* ext = data[0].find("sonder");
    const json::Value* prof = ext->find("profile");
    REQUIRE(prof != nullptr);
    CHECK(prof->find("context_length")->as_int() == 84992);
    CHECK(prof->find("configured_context_length")->as_int() == 100096);
    CHECK(prof->find("estimated_vram_mib")->as_int() == 15000);  // for the configured context
    // The runtime object from the backend sits beside the profile, unchanged.
    REQUIRE(ext->find("runtime") != nullptr);
    CHECK(ext->find("runtime")->find("context")->find("fitted_ctx")->as_int() == 84992);
    const auto& profiles = doc.find("sonder")->find("profiles")->as_array();
    REQUIRE(profiles.size() == 2);
    CHECK(profiles[0].find("context_length")->as_int() == 84992);
    CHECK(profiles[0].find("configured_context_length")->as_int() == 100096);
    // Only the served profile runs with the fitted context.
    CHECK(profiles[1].find("context_length")->as_int() == 100096);
    CHECK(profiles[1].find("configured_context_length") == nullptr);
}

TEST_CASE("models: a runtime context equal to the profile's leaves the profile unchanged") {
    si::BackendRuntimeStatus rt;
    rt.context.policy = "warn";
    rt.context.configured_ctx = 100096;
    rt.context.fitted_ctx = 100096;
    rt.context.outcome = "not_needed";
    RuntimeProfileFixture f(rt);
    const Reply r = get(f.fixture->port, "/v1/models");
    REQUIRE(r.status == 200);
    const json::Value doc = r.json();
    const json::Value* prof = doc.find("data")->as_array().at(0).find("sonder")->find("profile");
    REQUIRE(prof != nullptr);
    CHECK(prof->find("context_length")->as_int() == 100096);
    CHECK(prof->find("configured_context_length") == nullptr);
}

TEST_CASE("models: without launch profiles the response has no profile fields") {
    Fixture f;
    const Reply r = get(f.port, "/v1/models");
    REQUIRE(r.status == 200);
    const json::Value doc = r.json();
    const auto& ext = doc.find("data")->as_array().at(0).find("sonder")->as_object();
    CHECK(ext.size() == 3u);  // backend, default, synthetic
    CHECK(ext.find("profile") == nullptr);
    const auto& meta = doc.find("sonder")->as_object();
    CHECK(meta.size() == 1u);  // api_version
    CHECK(meta.find("profiles") == nullptr);
}

TEST_CASE("server options: a profile binding must name a served model") {
    auto o = Fixture::defaults();
    srv::ModelProfileBinding b;
    b.model = "not-served";
    o.profile_bindings.push_back(b);
    CHECK(contains(srv::validate_options(o).message(), "is not a served --model"));
    o.profile_bindings[0].model = "mock:tiny";
    o.profile_bindings.push_back(o.profile_bindings[0]);
    CHECK(contains(srv::validate_options(o).message(), "bound twice"));
}

// ------------------------------------------------------------------ serve CLI

namespace {

std::filesystem::path write_temp(const std::string& name, const std::string& content) {
    const auto path = std::filesystem::temp_directory_path() / (si::make_id("sonder-profile-test") + "-" + name);
    std::ofstream(path, std::ios::binary) << content;
    return path;
}

int serve(const std::vector<std::string>& args, std::string& err_text) {
    std::ostringstream out;
    std::ostringstream err;
    const int rc = srv::serve_main(args, out, err);
    err_text = err.str();
    return rc;
}

}  // namespace

TEST_CASE("serve_main: launch profile usage errors exit 2 before starting anything") {
    // The model is a header-only GGUF fixture; nothing is spawned because
    // every case fails during option handling.
    const auto model = write_temp("model.gguf", hybrid_fixture());
    const std::string m = model.generic_string();
    const auto profiles = write_temp("profiles.json", R"({"profiles":[
        {"name":"big","backend":"llamaserver","model":")" + m + R"(","ctx_size":512,"n_gpu_layers":-1,"fit":false},
        {"name":"direct","backend":"llamacpp","model":")" + m + R"(","parallel":2},
        {"name":"auto","backend":"llamaserver","model":"missing-model.gguf"}]})");
    const std::string pf = profiles.string();
    std::string err;
    CHECK(serve({"--backend", "mock", "--profile", "big"}, err) == 2);
    CHECK(contains(err, "--profile needs --profiles"));
    CHECK(serve({"--profiles", pf}, err) == 2);
    CHECK(contains(err, "--profiles needs --profile"));
    CHECK(serve({"--profiles", pf, "--profile", "nope"}, err) == 2);
    CHECK(contains(err, "no launch profile named 'nope'"));
    CHECK(serve({"--backend", "mock", "--allow-overcommit"}, err) == 2);
    CHECK(serve({"--profiles", pf, "--profile", "big", "--backend", "llamacpp"}, err) == 2);
    CHECK(contains(err, "conflicts with launch profile"));
    CHECK(serve({"--profiles", pf, "--profile", "big", "--model", "x"}, err) == 2);
    CHECK(serve({"--profiles", pf, "--profile", "big", "--context-length", "10"}, err) == 2);
    CHECK(serve({"--profiles", pf, "--profile", "big", "--llamaserver-arg", "--no-warmup"}, err) == 2);
    CHECK(serve({"--profiles", pf, "--profile", "big", "--llamaserver-mode", "attach"}, err) == 2);
    CHECK(contains(err, "needs llamaserver spawn mode"));
    CHECK(serve({"--profiles", pf, "--profile", "direct"}, err) == 2);
    CHECK(contains(err, "'parallel' is not supported by the llamacpp backend; use llamaserver"));
    // Over budget: refused unless --allow-overcommit (the fixture needs the
    // 256 MiB runtime allowance, so a 100 MiB budget cannot fit).
    CHECK(serve({"--profiles", pf, "--profile", "big", "--vram-budget-mib", "100", "--llamaserver-executable",
                 "llama-server-not-started"},
                err) == 2);
    CHECK(contains(err, "does not fit"));
    CHECK(contains(err, "vs budget 100 MiB"));
    CHECK(serve({"--profiles", pf, "--profile", "big", "--vram-budget-mib", "0"}, err) == 2);
    const auto bad = write_temp("bad.json", R"({"profiles":[{"name":"x","backend":"llamaserver"}]})");
    CHECK(serve({"--profiles", bad.string(), "--profile", "x"}, err) == 2);
    CHECK(contains(err, "'model' is required"));
    std::filesystem::remove(model);
    std::filesystem::remove(profiles);
    std::filesystem::remove(bad);
}
