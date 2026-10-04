// Launch profiles: schema parsing and validation, llama-server argv golden
// tests, direct llamacpp mapping and rejection, the GGUF header reader, the
// VRAM estimate (attention-only and hybrid), sampling defaults, and the
// profile metadata in /v1/models (plus unchanged behaviour without profiles).
#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
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

class ScopedSwaFullEnvironment {
public:
    ScopedSwaFullEnvironment() {
        if (const char* value = std::getenv("LLAMA_ARG_SWA_FULL")) previous_ = value;
        set(nullptr);
    }
    ~ScopedSwaFullEnvironment() { (void)change(previous_ ? previous_->c_str() : nullptr); }
    void set(const char* value) { REQUIRE(change(value) == 0); }

private:
    static int change(const char* value) {
#if defined(_WIN32)
        return _putenv_s("LLAMA_ARG_SWA_FULL", value == nullptr ? "" : value);
#else
        return value == nullptr ? unsetenv("LLAMA_ARG_SWA_FULL") : setenv("LLAMA_ARG_SWA_FULL", value, 1);
#endif
    }
    std::optional<std::string> previous_;
};

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
    // Without `parallel`, llama-server's automatic slots unify the KV cache
    // whatever --no-kv-unified says: the field would silently not apply.
    CHECK(contains(parse_error("{" + base + R"(,"kv_unified":false})"), "'kv_unified': false needs 'parallel'"));
    CHECK(parse_single("{" + base + R"(,"kv_unified":false,"parallel":2})").kv_unified == false);
    CHECK(parse_single("{" + base + R"(,"kv_unified":true})").kv_unified == true);
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
    for (const std::string forbidden : {R"("--host","0.0.0.0")", R"("--port=9000")", R"("--api-key","k")",
                                  R"("--api-key-file","f")", R"("-hf","org/model")", R"("--model-url","u")",
                                  R"("--")", R"("--models-dir","d")",
                                  // llama-server reads every '_' of a "--" flag as '-' (common/arg.cpp).
                                  R"("--api_key","k")", R"("--api_key_file","f")", R"("--hf_repo","org/model")",
                                  R"("--hf_token","t")", R"("--model_url","u")", R"("--models_dir","d")",
                                  R"("--models_preset","p.ini")", R"("--spec_draft_hf","org/draft")"}) {
        CAPTURE(forbidden);
        CHECK(contains(parse_error(with(forbidden)), "must not contain"));
    }
    // The refusal names the flag llama-server would have read.
    CHECK(contains(parse_error(with(R"("--api_key","k")")), "--api_key, which llama-server reads as --api-key"));
    for (const auto& [flag, field] : std::vector<std::pair<std::string, std::string>>{
             {R"("-c","4096")", "ctx_size"},
             {R"("--ctx-size=4096")", "ctx_size"},
             {R"("-ngl","99")", "n_gpu_layers"},
             {R"("--cache-type-v","q8_0")", "cache_type_v"},
             {R"("--no-kv-unified")", "kv_unified"},
             {R"("--temp","0.7")", "sampling.temperature"},
             {R"("-m","other.gguf")", "model"},
             {R"("--alias","x")", "name"},
             {R"("--spec-type","draft-mtp")", "speculative.types"},
             // Underscore spellings: llama-server would take the last of
             // `--ctx-size 65536 --ctx_size 262144`, so the fit check and
             // /v1/models would describe a launch that does not happen.
             {R"("--ctx_size","262144")", "ctx_size"},
             {R"("--ctx_size=262144")", "ctx_size"},
             {R"("--n_gpu_layers","99")", "n_gpu_layers"},
             {R"("--cache_type_k","q8_0")", "cache_type_k"},
             {R"("--no_kv_unified")", "kv_unified"},
             {R"("--spec_draft_model","d.gguf")", "speculative.draft_model"},
             {R"("--top_p","0.5")", "sampling.top_p"}}) {
        CAPTURE(flag);
        const std::string message = parse_error(with(flag));
        CHECK(contains(message, "use the typed field '" + field + "'"));
    }
    const LaunchProfile ok = parse_single(with(R"("--no-warmup","--override-kv","a=int:1")"));
    CHECK(ok.extra_args == std::vector<std::string>{"--no-warmup", "--override-kv", "a=int:1"});
    // Allowed flags stay allowed under any spelling, and pass through verbatim.
    CHECK(parse_single(with(R"("--no_warmup")")).extra_args == std::vector<std::string>{"--no_warmup"});
}

TEST_CASE("launch profile: extra_args refuse llama-server agent tools, MCP and file serving") {
    const auto with = [](const std::string& arg) {
        return R"({"name":"p","backend":"llamaserver","model":"m.gguf","extra_args":[)" + arg + "]}";
    };
    for (const std::string unsafe :
         {R"("--tools","all")", R"("--tools=exec_shell_command")", R"("--tools-runtime","ssh:host")",
          R"("--mcp-servers-config","m.json")", R"("--mcp-servers-json","{}")", R"("-ag")", R"("--agent")",
          R"("--ui-mcp-proxy")", R"("--webui-mcp-proxy")", R"("--path","C:/")", R"("--media-path","D:/")",
          // llama-server reads these as the hyphen spellings above; MCP
          // servers in the JSON are started with the child.
          R"("--mcp_servers_json","{}")", R"("--mcp_servers_config","m.json")", R"("--tools_runtime","ssh:host")",
          R"("--ui_mcp_proxy")", R"("--webui_mcp_proxy")", R"("--media_path","D:/")"}) {
        CAPTURE(unsafe);
        CHECK(contains(parse_error(with(unsafe)), "agent tools, MCP and local file serving"));
    }
    CHECK(contains(parse_error(with(R"("--mcp_servers_json","{}")")),
                   "--mcp_servers_json, which llama-server reads as --mcp-servers-json"));
    // The directory llama-server starts ffmpeg/ffprobe from (video input).
    for (const std::string program : {R"("--video-ffmpeg-dir","C:/tools")", R"("--video_ffmpeg_dir","C:/tools")"}) {
        CAPTURE(program);
        CHECK(contains(parse_error(with(program)), "cannot choose programs for llama-server to run"));
    }
    // The disabling spellings stay allowed.
    const LaunchProfile ok =
        parse_single(with(R"("--no-agent","-no-ag","--no-ui-mcp-proxy","--no-webui-mcp-proxy","--no_agent")"));
    CHECK(ok.extra_args.size() == 5u);
}

TEST_CASE("launch profile: extra_args refuse model presets and prompt/log files") {
    const auto with = [](const std::string& arg) {
        return R"({"name":"p","backend":"llamaserver","model":"m.gguf","extra_args":[)" + arg + "]}";
    };
    // Every preset listed by `llama-server --help`; each can download weights.
    for (const std::string preset :
         {R"("--embd-gemma-default")", R"("--fim-qwen-1.5b-default")", R"("--fim-qwen-3b-default")",
          R"("--fim-qwen-7b-default")", R"("--fim-qwen-7b-spec")", R"("--fim-qwen-14b-spec")",
          R"("--fim-qwen-30b-default")", R"("--gpt-oss-20b-default")", R"("--gpt-oss-120b-default")",
          R"("--vision-gemma-4b-default")", R"("--vision-gemma-12b-default")", R"("--spec-default")",
          R"("--some-future-model-default")",
          // Underscore spellings select the same presets in llama-server.
          R"("--gpt_oss_20b_default")", R"("--fim_qwen_7b_spec")", R"("--spec_default")"}) {
        CAPTURE(preset);
        CHECK(contains(parse_error(with(preset)), "built-in model presets"));
    }
    for (const std::string writer : {R"("--log-file","C:/x.log")", R"("--log-file=x.log")",
                               R"("--log-prompts-dir","D:/prompts")", R"("--log-prompts-dir=p")",
                               R"("-lcd","C:/ngram.bin")", R"("--lookup-cache-dynamic","ngram.bin")",
                               R"("--lookup-cache-dynamic=ngram.bin")", R"("--log_prompts_dir","D:/prompts")",
                               R"("--log_file","x.log")", R"("--lookup_cache_dynamic","ngram.bin")"}) {
        CAPTURE(writer);
        CHECK(contains(parse_error(with(writer)), "write logs or prompts to disk"));
    }
    // Flags that merely contain "default"/"spec" elsewhere stay allowed.
    const LaunchProfile ok = parse_single(with(R"("--spec-draft-n-min","2","--log-disable","--no-warmup")"));
    CHECK(ok.extra_args.size() == 4u);
}

TEST_CASE("launch profile: extra_args refuse remote RPC offload") {
    const auto with = [](const std::string& arg) {
        return R"({"name":"p","backend":"llamaserver","model":"m.gguf","extra_args":[)" + arg + "]}";
    };
    // --rpc registers rpc-server devices on other hosts; layers placed there
    // send tensors and every prompt's activations over the network.
    for (const std::string remote : {R"("--rpc","10.0.0.5:50052")", R"("--rpc=192.168.1.2:50052,192.168.1.3:50052")"}) {
        CAPTURE(remote);
        CHECK(contains(parse_error(with(remote)), "extra_args must not contain --rpc"));
        CHECK(contains(parse_error(with(remote)), "remote RPC servers"));
    }
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
            "cache_type_k":"q8_0","cache_type_v":"q5_1","flash_attn":"enabled"})");
    si::BackendSetup setup;
    std::string device;
    REQUIRE(si::apply_llamacpp_profile(p, setup, device).ok());
    // The values land in the BackendSetup fields serve's --batch-size,
    // --ubatch-size, --cache-type-k/-v and --flash-attn set (#45), unchanged
    // except that flash_attn is canonical (enabled -> on).
    CHECK(setup.llamacpp_batch_size == 1024u);
    CHECK(setup.llamacpp_ubatch_size == 256u);
    CHECK(setup.llamacpp_kv_cache_type_k == std::string("q8_0"));
    CHECK(setup.llamacpp_kv_cache_type_v == std::string("q5_1"));
    CHECK(setup.llamacpp_flash_attention == std::string("on"));
    CHECK(device.empty());  // no n_gpu_layers: the device is left alone

    // Fields the profile leaves unset keep the backend defaults (and any
    // value already in the setup).
    const LaunchProfile minimal = parse_single(R"({"name":"d","backend":"llamacpp","model":"m.gguf"})");
    si::BackendSetup untouched;
    untouched.llamacpp_batch_size = 2048u;
    REQUIRE(si::apply_llamacpp_profile(minimal, untouched, device).ok());
    CHECK(untouched.llamacpp_batch_size == 2048u);
    CHECK_FALSE(untouched.llamacpp_ubatch_size.has_value());
    CHECK_FALSE(untouched.llamacpp_kv_cache_type_k.has_value());
    CHECK_FALSE(untouched.llamacpp_kv_cache_type_v.has_value());
    CHECK_FALSE(untouched.llamacpp_flash_attention.has_value());

    // A rejected profile applies nothing, these fields included.
    const LaunchProfile rejected = parse_single(
        R"({"name":"d","backend":"llamacpp","model":"m.gguf","cache_type_k":"q8_0","parallel":2})");
    si::BackendSetup clean;
    CHECK_FALSE(si::apply_llamacpp_profile(rejected, clean, device).ok());
    CHECK_FALSE(clean.llamacpp_kv_cache_type_k.has_value());
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
    // GGUF BOOL array (item type 7, one byte per item): how gguf-py writes a
    // per-layer sliding_window_pattern.
    void kv_bool_array(const std::string& key, const std::vector<bool>& values) {
        str(key);
        u32(9);
        u32(7);
        u64(values.size());
        for (const bool v : values) bytes_.push_back(v ? '\x01' : '\x00');
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

// A Gemma-4-12B-shaped header (gemma4 architecture, no tensors): 48 blocks.
// Every 6th block (5, 11, ..., 47) is full attention with 1 KV head of 512;
// the other 40 use sliding-window attention (window 1024) with 8 KV heads
// of 256. As convert_hf_to_gguf writes it: per-layer head_count_kv, the
// _swa head sizes and a BOOL sliding_window_pattern (true = SWA).
std::string gemma4_fixture() {
    std::vector<std::int32_t> heads(48, 8);
    std::vector<bool> pattern(48, true);
    for (std::size_t i = 5; i < 48; i += 6) {
        heads[i] = 1;
        pattern[i] = false;
    }
    GgufWriter w;
    w.kv_str("general.architecture", "gemma4");
    w.kv_u32("gemma4.block_count", 48);
    w.kv_u32("gemma4.embedding_length", 3840);
    w.kv_u32("gemma4.context_length", 262144);
    w.kv_u32("gemma4.attention.head_count", 16);
    w.kv_i32_array("gemma4.attention.head_count_kv", heads);
    w.kv_u32("gemma4.attention.key_length", 512);
    w.kv_u32("gemma4.attention.value_length", 512);
    w.kv_u32("gemma4.attention.key_length_swa", 256);
    w.kv_u32("gemma4.attention.value_length_swa", 256);
    w.kv_u32("gemma4.attention.sliding_window", 1024);
    w.kv_bool_array("gemma4.attention.sliding_window_pattern", pattern);
    w.kv_str_array("tokenizer.ggml.tokens", {"a", "b", "c", "d"});
    return w.finish(0, 13);
}

// Blocks 0..n-1, true where block i % 6 != 5 (the Gemma 4 fixture's SWA blocks).
std::vector<bool> every_sixth_full(std::size_t n) {
    std::vector<bool> swa(n, true);
    for (std::size_t i = 5; i < n; i += 6) swa[i] = false;
    return swa;
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

namespace {

// "1101" -> {true, true, false, true}
std::vector<bool> bits(std::string_view text) {
    std::vector<bool> out;
    for (const char c : text) out.push_back(c == '1');
    return out;
}

std::string repeat(std::string_view text, std::size_t times) {
    std::string out;
    for (std::size_t i = 0; i < times; ++i) out += text;
    return out;
}

// A dense header for `arch`: `blocks` blocks of 8 KV heads of 128 (embedding
// 1024 over 8 heads; f16 K and V: 4,096 bytes per token and layer), plus
// integer keys under "<arch>.".
si::GgufModelInfo swa_header(const std::string& arch, std::uint32_t blocks,
                             const std::vector<std::pair<std::string, std::uint32_t>>& keys) {
    GgufWriter w;
    w.kv_str("general.architecture", arch);
    w.kv_u32(arch + ".block_count", blocks);
    w.kv_u32(arch + ".embedding_length", 1024);
    w.kv_u32(arch + ".attention.head_count", 8);
    w.kv_u32(arch + ".attention.head_count_kv", 8);
    for (const auto& [key, value] : keys) w.kv_u32(arch + "." + key, value);
    return parse_fixture(w.finish(0, 5 + keys.size()));
}

}  // namespace

TEST_CASE("GGUF reader: a BOOL sliding_window_pattern array and the SWA head sizes") {
    // gemma4's loader reads the per-layer pattern array (true = SWA), the
    // window and key_length_swa/value_length_swa (src/models/gemma4.cpp).
    const si::GgufModelInfo m = parse_fixture(gemma4_fixture());
    CHECK(m.architecture == "gemma4");
    CHECK(m.kind == si::ModelArchitecture::attention_only);
    CHECK(m.swa_layers == every_sixth_full(48));
    CHECK(m.sliding_window == 1024u);
    CHECK(m.key_length == 512u);
    CHECK(m.value_length == 512u);
    CHECK(m.key_length_swa == 256u);
    CHECK(m.value_length_swa == 256u);
    CHECK(m.kv_heads[0] == 8u);
    CHECK(m.kv_heads[5] == 1u);
    CHECK(m.kv_heads[47] == 1u);
    CHECK_FALSE(m.sliding_window_unmodelled);

    // An INT32 pattern array works too (llama.cpp's get_arr takes BOOL, INT32
    // and UINT32); blocks past the array's end stay full attention.
    GgufWriter g;
    g.kv_str("general.architecture", "granite_swa");
    g.kv_u32("granite_swa.block_count", 6);
    g.kv_u32("granite_swa.embedding_length", 512);
    g.kv_u32("granite_swa.attention.head_count", 4);
    g.kv_u32("granite_swa.attention.sliding_window", 512);
    g.kv_i32_array("granite_swa.attention.sliding_window_pattern", {1, 1, 0, 1});
    const si::GgufModelInfo granite = parse_fixture(g.finish(0, 6));
    CHECK(granite.swa_layers == bits("110100"));
    CHECK(granite.sliding_window == 512u);
    // Without the _swa keys, SWA blocks use the full-attention head sizes.
    CHECK(granite.key_length_swa == 128u);
    CHECK(granite.value_length_swa == 128u);
}

TEST_CASE("GGUF reader: SWA layout per architecture and llama.cpp's built-in period when the file has none") {
    // Each loader's load_arch_hparams in llama.cpp 7fe450e (src/models/). Without
    // a per-layer array, load_swa_pattern(ml, period, dense_first) uses the
    // file's scalar sliding_window_pattern, else the built-in period, and
    // set_swa_pattern marks block i sliding-window when i % period < period - 1
    // (the full block closes each period), or i % period != 0 when dense first.
    struct Case {
        std::string arch;
        std::uint32_t blocks;
        std::vector<std::pair<std::string, std::uint32_t>> keys;
        std::string swa;  // expected per block; empty = no SWA blocks
        std::uint64_t window;
    };
    for (const Case& c : std::vector<Case>{
             // gemma3: SWA only with a positive window; period 6.
             {"gemma3", 12, {{"attention.sliding_window", 1024}}, "111110111110", 1024},
             {"gemma3", 12, {{"attention.sliding_window", 0}}, "", 0},
             {"gemma3", 12, {}, "", 0},
             // gpt-oss (openai-moe): period 2.
             {"gpt-oss", 6, {{"attention.sliding_window", 128}}, "101010", 128},
             // cohere2moe: period 4, dense first.
             {"cohere2moe", 8, {{"attention.sliding_window", 4096}}, "01110111", 4096},
             // gemma2: the window defaults to 4096 without the key.
             {"gemma2", 4, {}, "1010", 4096},
             // llama4: chunked attention (3 of 4 layers), window fixed at 8192,
             // unless the file's window is 0.
             {"llama4", 8, {}, "11101110", 8192},
             {"llama4", 8, {{"attention.sliding_window", 0}}, "", 0},
             // smallthinker: dense first, and the loader overwrites the window with 4096.
             {"smallthinker", 8, {{"attention.sliding_window", 2048}}, "01110111", 4096},
             // plamo3: period 8, or the file's scalar pattern.
             {"plamo3", 8, {{"attention.sliding_window", 1024}}, "11111110", 1024},
             {"plamo3", 8, {{"attention.sliding_window", 1024}, {"attention.sliding_window_pattern", 4}}, "11101110", 1024},
             // exaone4: only the 64-layer model has SWA.
             {"exaone4", 30, {{"attention.sliding_window", 4096}}, "", 0},
             {"exaone4", 64, {{"attention.sliding_window", 2048}}, repeat("1110", 16), 2048},
             // phi3: llama.cpp switches Phi SWA off whatever the file says.
             {"phi3", 4, {{"attention.sliding_window", 2047}}, "", 0},
             // gemma4 requires the per-layer array; without it the model does not load.
             {"gemma4", 6, {{"attention.sliding_window", 1024}}, "", 0},
         }) {
        CAPTURE(c.arch);
        CAPTURE(c.blocks);
        CAPTURE(c.keys.size());
        const si::GgufModelInfo m = swa_header(c.arch, c.blocks, c.keys);
        CHECK(m.swa_layers == bits(c.swa));
        CHECK(m.sliding_window == c.window);
        CHECK_FALSE(m.sliding_window_unmodelled);
    }
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

TEST_CASE("GGUF reader: implausible ssm sizes and oversized names are refused without overflow") {
    // 2 x groups x state = 2^63 overflowed a signed 64-bit product before.
    GgufWriter huge;
    huge.kv_str("general.architecture", "mamba2");
    huge.kv_u32("mamba2.block_count", 2);
    huge.kv_u32("mamba2.ssm.conv_kernel", 4);
    huge.kv_u32("mamba2.ssm.inner_size", 4096);
    huge.kv_u32("mamba2.ssm.state_size", 0x80000000u);
    huge.kv_u32("mamba2.ssm.group_count", 0x80000000u);
    const auto h = si::parse_gguf_model_info(huge.finish(0, 6));
    CHECK_FALSE(h.ok());
    CHECK(h.status().code() == si::ErrorCode::invalid_argument);
    CHECK(contains(h.status().message(), "implausible mamba2.ssm.* sizes"));

    // general.architecture is kept and quoted in messages: a short printable name.
    for (const std::string& arch : {std::string(300, 'a'), std::string("llama\x1b[2J")}) {
        CAPTURE(arch.size());
        GgufWriter w;
        w.kv_str("general.architecture", arch);
        w.kv_u32(arch + ".block_count", 1);
        const auto parsed = si::parse_gguf_model_info(w.finish(0, 2));
        CHECK_FALSE(parsed.ok());
        CHECK(contains(parsed.status().message(), "general.architecture must be at most 256 printable"));
    }

    // A key or tensor name quoted in an error (which reaches /v1/models
    // `estimate_error`) is cut to 64 printable characters.
    GgufWriter long_tensor;
    long_tensor.kv_str("general.architecture", "llama");
    long_tensor.kv_u32("llama.block_count", 1);
    long_tensor.tensor(std::string(100000, 'x') + "\n", {32}, 99);
    const std::string tensor_message = si::parse_gguf_model_info(long_tensor.finish(1, 2)).status().message();
    CHECK(contains(tensor_message, "unknown ggml type 99"));
    CHECK(tensor_message.size() < 200);
    CHECK(tensor_message.find('\n') == std::string::npos);
    GgufWriter long_key;
    long_key.kv_str("general.architecture", "llama");
    long_key.str(std::string(100000, 'k'));
    long_key.u32(99);  // no such value type
    const std::string key_message = si::parse_gguf_model_info(long_key.finish(0, 2)).status().message();
    CHECK(contains(key_message, "unsupported value type"));
    CHECK(key_message.size() < 200);
}

TEST_CASE("gguf: client-visible read errors do not carry the model path") {
    const std::string missing = "C:/Users/someone/models/secret-dir/missing.gguf";
    const auto status = si::read_gguf_model_info(missing).status();
    REQUIRE(contains(status.message(), missing));  // the operator log keeps the path
    const std::string redacted = si::redact_model_path(status, missing);
    CHECK_FALSE(contains(redacted, "secret-dir"));
    CHECK(contains(redacted, "<model path>"));

    // A malformed header appends the path too.
    const auto dir = std::filesystem::temp_directory_path() / si::make_id("sonder-gguf-redact");
    std::filesystem::create_directories(dir);
    const std::string bad = (dir / "not-a-model.gguf").string();
    {
        std::ofstream out(bad, std::ios::binary);
        out << "not a gguf";
    }
    const auto bad_status = si::read_gguf_model_info(bad).status();
    REQUIRE(contains(bad_status.message(), bad));
    const std::string bad_redacted = si::redact_model_path(bad_status, bad);
    CHECK_FALSE(contains(bad_redacted, dir.string()));
    CHECK(contains(bad_redacted, "not a GGUF file (<model path>)"));
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
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

TEST_CASE("VRAM estimate: speculative rollback keeps one recurrent snapshot per draft position") {
    const si::GgufModelInfo m = qwen38_q3();
    LaunchProfile p = parse_single(kMeasuredProfile);
    const si::VramEstimate plain = si::estimate_vram(m, p);
    // One snapshot: 48 delta-net layers x 817,152 f32 values = 149.6 MiB.
    CHECK(plain.recurrent_snapshots == 1u);
    CHECK(plain.recurrent_bytes == 48ull * 817152 * 4);
    for (const std::uint32_t n : {1u, 2u, 3u}) {
        CAPTURE(n);
        p.speculative = si::SpeculativeSettings{{"draft-mtp"}, n, ""};
        const si::VramEstimate e = si::estimate_vram(m, p);
        CHECK(e.recurrent_snapshots == 1u + n);
        CHECK(e.recurrent_bytes == (1u + n) * plain.recurrent_bytes);
        CHECK(e.total_bytes == e.weights_bytes + e.kv_bytes + e.recurrent_bytes + e.compute_bytes);
    }
    // Unset draft_n_max: llama.cpp's default of 3.
    p.speculative = si::SpeculativeSettings{{"draft-mtp"}, std::nullopt, ""};
    CHECK(si::estimate_vram(m, p).recurrent_snapshots == 4u);
    // Per sequence: parallel 2 with n_max 2 keeps 2 x 3 states.
    p.speculative = si::SpeculativeSettings{{"draft-mtp", "ngram-mod"}, 2, ""};
    p.parallel = 2;
    CHECK(si::estimate_vram(m, p).recurrent_bytes == 6 * plain.recurrent_bytes);
    p.parallel = 1;
    // N-gram drafts re-decode on a miss instead of rolling back: one state.
    p.speculative = si::SpeculativeSettings{{"ngram-mod"}, 3, ""};
    CHECK(si::estimate_vram(m, p).recurrent_snapshots == 1u);
    CHECK(si::estimate_vram(m, p).recurrent_bytes == plain.recurrent_bytes);
    // llama.cpp clamps the snapshots to none on architectures it cannot roll back.
    si::GgufModelInfo other = m;
    other.architecture = "qwen3next";
    p.speculative = si::SpeculativeSettings{{"draft-mtp"}, 2, ""};
    CHECK(si::estimate_vram(other, p).recurrent_snapshots == 1u);
}

TEST_CASE("VRAM estimate: the live MTP profile against the measured dedicated memory") {
    // stack/llamaserver-switch/llamaserver.default.json as a profile: Q3_K_XL,
    // q4_0/q4_0, draft-mtp n_max 2, parallel 1, ubatch 512, fit off.
    const auto profile = [](std::uint32_t ctx, std::optional<std::uint32_t> n_max) {
        LaunchProfile p = parse_single(R"({
            "name": "qwen3.8-27b-q3-mtp", "backend": "llamaserver", "model": "q3.gguf",
            "n_gpu_layers": 999, "ctx_size": 65536, "parallel": 1, "kv_unified": false,
            "batch_size": 2048, "ubatch_size": 512, "flash_attn": "on",
            "cache_type_k": "q4_0", "cache_type_v": "q4_0", "fit": false})");
        p.ctx_size = ctx;
        if (n_max) p.speculative = si::SpeculativeSettings{{"draft-mtp"}, *n_max, ""};
        return p;
    };
    const si::VramEstimate live = si::estimate_vram(qwen38_q3(), profile(65536u, 2u));
    CHECK(live.gpu_layers == 65u);        // the MTP block loads
    CHECK(live.attention_layers == 17u);  // 16 + the MTP block's attention
    CHECK(live.kv_bytes == 17ull * 1152 * 65536);
    CHECK(live.recurrent_snapshots == 3u);
    CHECK(live.recurrent_bytes == 3ull * 48 * 817152 * 4);
    // Dedicated memory at load on the RTX 5070 Ti (bench-round3, llama-server
    // 161755f29). Each +1 of n_max costs one 150 MiB recurrent snapshot;
    // without that term the MTP rows were 2.2 to 4.2 % low.
    struct Measured {
        std::uint32_t ctx;
        std::optional<std::uint32_t> n_max;
        double mib;
    };
    for (const Measured& point : {Measured{65536u, std::nullopt, 13604.0}, Measured{65536u, 1u, 14478.0},
                                  Measured{65536u, 2u, 14628.0}, Measured{65536u, 3u, 14778.0},
                                  Measured{32768u, 2u, 13762.0}}) {
        const si::VramEstimate e = si::estimate_vram(qwen38_q3(), profile(point.ctx, point.n_max));
        const double error = (static_cast<double>(e.total_mib()) - point.mib) / point.mib;
        MESSAGE("ctx " << point.ctx << ", n_max " << (point.n_max ? static_cast<int>(*point.n_max) : 0) << ": estimate "
                       << e.total_mib() << " MiB vs " << point.mib << " MiB measured: " << error * 100.0 << " %");
        CHECK(std::fabs(error) < 0.02);
    }
}

TEST_CASE("VRAM estimate: a llamaserver profile without parallel counts llama-server's 4 automatic slots") {
    // llama-server's --parallel defaults to -1 (auto): 4 slots with a unified
    // KV cache (server.cpp, llama.cpp 7fe450e; the bundled 161755f29 logs
    // "using n_parallel = 4 and kv_unified = true"). Every slot keeps its own
    // recurrent state (n_seq_max x (1 + n_rs_seq) rows).
    const auto live = [](std::optional<std::uint32_t> parallel) {
        LaunchProfile p = parse_single(R"({
            "name": "qwen3.8-27b-q3-mtp", "backend": "llamaserver", "model": "q3.gguf",
            "n_gpu_layers": 999, "ctx_size": 65536, "batch_size": 2048, "ubatch_size": 512,
            "flash_attn": "on", "cache_type_k": "q4_0", "cache_type_v": "q4_0", "fit": false,
            "speculative": {"types": ["draft-mtp"], "draft_n_max": 2}})");
        p.parallel = parallel;
        return p;
    };
    const si::GgufModelInfo m = qwen38_q3();
    const std::uint64_t snapshot = 48ull * 817152 * 4;  // one recurrent state: 149.6 MiB
    const si::VramEstimate one = si::estimate_vram(m, live(1u));
    const si::VramEstimate unset = si::estimate_vram(m, live(std::nullopt));
    CHECK(one.recurrent_bytes == 1 * 3 * snapshot);    // 1 sequence x (1 + draft_n_max 2)
    CHECK(unset.recurrent_bytes == 4 * 3 * snapshot);  // 4 automatic slots x 3
    // The automatic slots share one unified pool of ctx tokens: same KV.
    CHECK(unset.sequences == 1u);
    CHECK(unset.kv_bytes == one.kv_bytes);
    // +1,347 MiB on the live profile: 15,807 instead of 14,460 MiB on the real
    // header, above the 16 GB card's default budget of 14,767 MiB.
    CHECK(unset.total_bytes - one.total_bytes == 3 * 3 * snapshot);
    // An explicit `parallel` is what runs.
    CHECK(si::estimate_vram(m, live(2u)).recurrent_bytes == 2 * 3 * snapshot);
    // The direct llamacpp backend decodes one sequence.
    const LaunchProfile direct = parse_single(R"({"name":"d","backend":"llamacpp","model":"q3.gguf","ctx_size":65536})");
    CHECK(si::estimate_vram(m, direct).recurrent_bytes == snapshot);
}

TEST_CASE("VRAM estimate: placement flags in extra_args are noted under the name llama-server reads") {
    for (const std::string flag : {"--override-tensor", "--override_tensor", "--n_cpu_moe", "--tensor_split", "--split_mode"}) {
        CAPTURE(flag);
        const LaunchProfile p = parse_single(R"({"name":"p","backend":"llamaserver","model":"m.gguf","ctx_size":4096,)"
                                             R"("parallel":1,"extra_args":[")" +
                                             flag + R"(","x"]})");
        bool noted = false;
        for (const auto& note : si::estimate_vram(qwen38_q3(), p).notes) {
            noted = noted || contains(note, "placement flags in extra_args");
        }
        CHECK(noted);
    }
}

TEST_CASE("VRAM estimate: crafted sizes saturate instead of overflowing") {
    // KV arithmetic runs in double; a cast of a value past 2^64 back to an
    // integer was undefined behaviour.
    si::GgufModelInfo wide;
    wide.architecture = "llama";
    wide.block_count = 2;
    wide.embedding_length = 64;
    wide.key_length = std::uint64_t{1} << 62;
    wide.value_length = std::uint64_t{1} << 62;
    wide.kv_heads = {1u << 31, 1u << 31};
    wide.block_bytes = {1, 1};
    const LaunchProfile p = parse_single(R"({"name":"w","backend":"llamaserver","model":"m.gguf","ctx_size":1048576})");
    const si::VramEstimate e = si::estimate_vram(wide, p);
    CHECK(e.kv_bytes_per_token == std::numeric_limits<std::uint64_t>::max());
    CHECK(e.kv_bytes == std::numeric_limits<std::uint64_t>::max());
    // MiB round up without wrapping to 0 near 2^64.
    si::VramEstimate near_max;
    near_max.total_bytes = std::numeric_limits<std::uint64_t>::max();
    CHECK(near_max.total_mib() == (std::uint64_t{1} << 44));
    near_max.total_bytes = (std::uint64_t{1} << 20) + 1;
    CHECK(near_max.total_mib() == 2u);
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
    // Two delta-net blocks; `parallel` unset: llama-server's 4 automatic slots.
    CHECK(e.recurrent_bytes == 4u * 2u * (3u * 64 + 256) * 4);
}

// ------------------------------------------------- sliding-window attention
//
// llama.cpp keeps sliding-window (SWA) layers in a second KV cache of
//   size_swa = GGML_PAD(min(size_base, n_swa * (unified ? n_seq_max : 1) + n_ubatch), 256)
// cells per stream, where size_base = n_ctx_seq is what full-attention layers
// keep (llama_kv_cache_iswa, src/llama-kv-cache-iswa.cpp:73 in llama.cpp
// 7fe450e; the same in b11195). --swa-full sets size_swa = size_base.

namespace {

LaunchProfile gemma4_profile(const std::string& extra = "") {
    return parse_single(R"({"name":"gemma4-12b","backend":"llamaserver","model":"g4.gguf","ctx_size":262144,)"
                        R"("parallel":1,"kv_unified":true,"n_gpu_layers":-1)" +
                        extra + "}");
}

}  // namespace

TEST_CASE("VRAM estimate: sliding-window layers keep llama.cpp's window-sized cache (Gemma 4 12B shape)") {
    const si::GgufModelInfo m = parse_fixture(gemma4_fixture());
    const si::VramEstimate e = si::estimate_vram(m, gemma4_profile());
    // Full blocks: 8 x 1 head x (512 + 512) values x 2 bytes = 16,384 B/token,
    // over all 262,144 cells: 4,294,967,296 bytes.
    // SWA blocks: 40 x 8 heads x (256 + 256) x 2 = 327,680 B/token, over
    // pad256(min(262144, 1024 x 1 + 512)) = 1,536 cells: 503,316,480 bytes.
    // Every block at full context with the full blocks' head size was
    // 176,093,659,136 bytes (164 GiB) for about 4.5 GiB of real KV.
    CHECK(e.kv_bytes == 4798283776ull);
    CHECK(e.attention_layers == 48u);
    CHECK(e.swa_attention_layers == 40u);
    CHECK(e.kv_bytes_per_token == 16384u);
    CHECK(e.swa_kv_bytes_per_token == 327680u);
    CHECK(e.context_per_sequence == 262144u);
    CHECK(e.swa_context_per_sequence == 1536u);
    CHECK(e.sequences == 1u);
    CHECK(e.kv_bytes == 16384ull * 262144 + 327680ull * 1536);
    CHECK((e.kv_bytes >> 20) == 4576u);  // 4,096 MiB + 480 MiB
    CHECK(e.total_bytes == e.weights_bytes + e.kv_bytes + e.recurrent_bytes + e.compute_bytes);

    // The micro-batch is part of the SWA cache: ubatch 1024 -> 1024 + 1024 =
    // 2,048 cells; batch 256 caps the default ubatch at 256 -> pad256(1,280).
    CHECK(si::estimate_vram(m, gemma4_profile(R"(,"ubatch_size":1024)")).swa_context_per_sequence == 2048u);
    CHECK(si::estimate_vram(m, gemma4_profile(R"(,"batch_size":256)")).swa_context_per_sequence == 1280u);
    CHECK(si::estimate_vram(m, gemma4_profile(R"(,"ubatch_size":300)")).swa_context_per_sequence == 1536u);
    // A context below the window: the SWA cache is as large as the full one.
    LaunchProfile small = gemma4_profile();
    small.ctx_size = 1024;
    const si::VramEstimate s = si::estimate_vram(m, small);
    CHECK(s.swa_context_per_sequence == 1024u);
    CHECK(s.kv_bytes == (16384ull + 327680ull) * 1024);
}

TEST_CASE("VRAM estimate: sliding-window cells per stream follow parallel and kv_unified") {
    const si::GgufModelInfo m = parse_fixture(gemma4_fixture());
    // Not unified, 4 slots: 4 streams of n_ctx_seq = pad(262144 / 4, 256) =
    // 65,536 cells; each stream's SWA cache holds window + ubatch = 1,536.
    LaunchProfile p = gemma4_profile();
    p.parallel = 4;
    p.kv_unified = false;
    const si::VramEstimate split = si::estimate_vram(m, p);
    CHECK(split.sequences == 4u);
    CHECK(split.context_per_sequence == 65536u);
    CHECK(split.swa_context_per_sequence == 1536u);
    CHECK(split.kv_bytes == 16384ull * 65536 * 4 + 327680ull * 1536 * 4);  // 6,308,233,216
    // Unified, 4 slots: one stream whose SWA cache holds window x 4 + ubatch.
    p.kv_unified = true;
    const si::VramEstimate unified = si::estimate_vram(m, p);
    CHECK(unified.sequences == 1u);
    CHECK(unified.swa_context_per_sequence == 4608u);
    CHECK(unified.kv_bytes == 16384ull * 262144 + 327680ull * 4608);  // 5,804,916,736
    // llama-server's automatic default (parallel unset) is 4 unified slots.
    LaunchProfile automatic = gemma4_profile();
    automatic.parallel.reset();
    automatic.kv_unified.reset();
    CHECK(si::estimate_vram(m, automatic).kv_bytes == unified.kv_bytes);
    // Streams smaller than window + ubatch keep their full size.
    p.kv_unified = false;
    p.ctx_size = 4096;  // 4 streams of 1,024 cells
    const si::VramEstimate tiny = si::estimate_vram(m, p);
    CHECK(tiny.context_per_sequence == 1024u);
    CHECK(tiny.swa_context_per_sequence == 1024u);
    CHECK(tiny.kv_bytes == (16384ull + 327680ull) * 1024 * 4);
}

TEST_CASE("VRAM estimate: --swa-full and the llamacpp backend keep full-size sliding-window caches") {
    const si::GgufModelInfo m = parse_fixture(gemma4_fixture());
    // Every SWA block at full context, each with its own head size:
    // (16,384 + 327,680) x 262,144 = 90,194,313,216 bytes (84 GiB).
    const std::uint64_t full_size = (16384ull + 327680ull) * 262144;
    // llama-server reads `--swa_full` as `--swa-full`.
    for (const std::string flag : {"--swa-full", "--swa_full"}) {
        CAPTURE(flag);
        const si::VramEstimate e = si::estimate_vram(m, gemma4_profile(R"(,"extra_args":[")" + flag + R"("])"));
        CHECK(e.kv_bytes == full_size);
        CHECK(e.swa_context_per_sequence == e.context_per_sequence);
        CHECK(e.kv_bytes == (e.kv_bytes_per_token + e.swa_kv_bytes_per_token) * e.context_per_sequence * e.sequences);
        bool noted = false;
        for (const auto& note : e.notes) noted = noted || contains(note, "--swa-full");
        CHECK(noted);
    }
    // The direct backend keeps llama.cpp's library default swa_full = true
    // (llama_context_default_params, b11195).
    const LaunchProfile direct = parse_single(R"({"name":"d","backend":"llamacpp","model":"g4.gguf","ctx_size":262144})");
    const si::VramEstimate d = si::estimate_vram(m, direct);
    CHECK(d.kv_bytes == full_size);
    bool noted = false;
    for (const auto& note : d.notes) noted = noted || contains(note, "llamacpp backend");
    CHECK(noted);
}

TEST_CASE("VRAM estimate: inherited full-SWA setting is supplied explicitly by the host") {
    ScopedSwaFullEnvironment environment;
    const si::GgufModelInfo model = parse_fixture(gemma4_fixture());
    const LaunchProfile profile = gemma4_profile();
    const auto window = si::estimate_vram(model, profile);
    environment.set("1");
    // Embedding estimates do not depend on ambient state until the caller
    // supplies its snapshot. The CLI separately checks the inherited value.
    CHECK(si::estimate_vram(model, profile).kv_bytes == window.kv_bytes);
    si::VramEstimateInputs inputs;
    inputs.inherited_swa_full = true;
    const auto full = si::estimate_vram(model, profile, inputs);
    CHECK(full.swa_context_per_sequence == full.context_per_sequence);
    CHECK(full.kv_bytes == (16384ull + 327680ull) * 262144);
    CHECK(full.kv_bytes > window.kv_bytes);
    bool noted = false;
    for (const auto& note : full.notes) noted = noted || contains(note, "LLAMA_ARG_SWA_FULL");
    CHECK(noted);
}

TEST_CASE("VRAM estimate: with flash attention off every layer's V row is the widest layer's") {
    // A transposed V cache (flash attention off) has n_embd_v_gqa_max() values
    // per row in every layer (llama_kv_cache, [TAG_V_CACHE_VARIABLE]). Gemma
    // 4's widest V row is a SWA block's 8 x 256 = 2,048, so the full blocks'
    // 1 x 512 rows are allocated at 2,048. `auto` and `on` allocate before
    // flash attention is resolved, with each layer's own row.
    const si::GgufModelInfo m = parse_fixture(gemma4_fixture());
    LaunchProfile p = gemma4_profile();
    p.flash_attn = "on";
    const si::VramEstimate on = si::estimate_vram(m, p);
    p.flash_attn = "auto";
    CHECK(si::estimate_vram(m, p).kv_bytes == on.kv_bytes);
    p.flash_attn = "off";
    const si::VramEstimate off = si::estimate_vram(m, p);
    CHECK(off.kv_bytes - on.kv_bytes == 8ull * (2048 - 512) * 2 * 262144);
    CHECK(off.kv_bytes_per_token == 8u * (512 + 2048) * 2);
    CHECK(off.swa_kv_bytes_per_token == on.swa_kv_bytes_per_token);
}

TEST_CASE("VRAM estimate: SWA layers from an architecture's built-in period") {
    // gemma3 GGUFs carry no pattern: llama.cpp's period 6 makes blocks 5 and
    // 11 full attention. 2 x 4,096 B/token x 32,768 + 10 x 4,096 x 1,536.
    const si::GgufModelInfo m = swa_header("gemma3", 12, {{"attention.sliding_window", 1024}});
    const LaunchProfile p =
        parse_single(R"({"name":"g3","backend":"llamaserver","model":"g3.gguf","ctx_size":32768,"parallel":1})");
    const si::VramEstimate e = si::estimate_vram(m, p);
    CHECK(e.kv_bytes == 2ull * 4096 * 32768 + 10ull * 4096 * 1536);
    CHECK(e.swa_attention_layers == 10u);
    // Dense first (cohere2moe): blocks 0 and 4 full, window 4096.
    const si::GgufModelInfo c = swa_header("cohere2moe", 8, {{"attention.sliding_window", 4096}});
    CHECK(si::estimate_vram(c, p).kv_bytes == 2ull * 4096 * 32768 + 6ull * 4096 * 4608);
}

TEST_CASE("VRAM estimate: a window on an architecture whose SWA layout is not modelled counts at full context") {
    // llama.cpp ignores sliding_window for llama; the estimate must not guess.
    const si::GgufModelInfo m = swa_header("llama", 4, {{"attention.sliding_window", 4096}});
    CHECK(m.swa_layers.empty());
    CHECK(m.sliding_window_unmodelled);
    const LaunchProfile p =
        parse_single(R"({"name":"l","backend":"llamaserver","model":"l.gguf","ctx_size":32768,"parallel":1})");
    const si::VramEstimate e = si::estimate_vram(m, p);
    CHECK(e.kv_bytes == 4ull * 4096 * 32768);
    bool noted = false;
    for (const auto& note : e.notes) noted = noted || contains(note, "llama.attention.sliding_window is set");
    CHECK(noted);
}

TEST_CASE("VRAM estimate: models without sliding-window layers are unchanged") {
    // Guards the non-SWA path: these values are what the estimator produced
    // before it knew about sliding windows.
    const si::VramEstimate q = si::estimate_vram(qwen38_q3(), parse_single(kMeasuredProfile));
    CHECK(q.kv_bytes == 29696ull * 100096);
    CHECK(q.kv_bytes_per_token == 29696u);
    CHECK(q.total_mib() == 15437u);
    CHECK(q.swa_attention_layers == 0u);
    CHECK(q.swa_kv_bytes_per_token == 0u);
    CHECK(q.swa_context_per_sequence == 0u);

    // A dense header carrying a BOOL array that the reader used to skip: the
    // "recurrent" key name must not make it hybrid now that BOOL arrays are kept.
    GgufWriter w;
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.block_count", 4);
    w.kv_u32("llama.embedding_length", 1024);
    w.kv_u32("llama.attention.head_count", 8);
    w.kv_u32("llama.attention.head_count_kv", 8);
    w.kv_bool_array("llama.attention.recurrent_layers", {false, false, false, false});
    const si::GgufModelInfo dense = parse_fixture(w.finish(0, 6));
    CHECK(dense.kind == si::ModelArchitecture::attention_only);
    CHECK(dense.kv_heads == std::vector<std::uint32_t>{8, 8, 8, 8});
    CHECK(dense.swa_layers.empty());
    CHECK_FALSE(dense.sliding_window_unmodelled);
    // 4 blocks x 4,096 B/token, whatever the flash attention mode and --swa-full.
    for (const std::string fa : {"on", "auto", "off"}) {
        for (const std::string extra : {"", R"(,"extra_args":["--swa-full"])"}) {
            CAPTURE(fa);
            CAPTURE(extra);
            LaunchProfile p = parse_single(R"({"name":"d","backend":"llamaserver","model":"d.gguf","ctx_size":8192,)"
                                           R"("parallel":2,"kv_unified":false)" +
                                           extra + "}");
            p.flash_attn = fa;
            const si::VramEstimate e = si::estimate_vram(dense, p);
            CHECK(e.kv_bytes == 4ull * 4096 * 4096 * 2);  // 2 streams of 4,096 cells
            CHECK(e.kv_bytes_per_token == 4u * 4096);
            CHECK(e.swa_attention_layers == 0u);
            CHECK(e.swa_context_per_sequence == 0u);
            CHECK(e.notes.empty());
        }
    }
    const LaunchProfile direct = parse_single(R"({"name":"d","backend":"llamacpp","model":"d.gguf","ctx_size":8192})");
    CHECK(si::estimate_vram(dense, direct).kv_bytes == 4ull * 4096 * 8192);

    // The hybrid fixture: flash attention off and --swa-full change nothing.
    const si::GgufModelInfo hybrid = parse_fixture(hybrid_fixture());
    LaunchProfile h = parse_single(R"({"name":"f","backend":"llamaserver","model":"m.gguf","ctx_size":512,"cache_type_k":"f16"})");
    const std::uint64_t base = si::estimate_vram(hybrid, h).kv_bytes;
    CHECK(base == 2u * 2 * (16 * 2 + 32 * 2) * 512);
    h.flash_attn = "off";
    h.extra_args = {"--swa-full"};
    CHECK(si::estimate_vram(hybrid, h).kv_bytes == base);
}

// Opt-in diagnostic, not a check: SONDER_TEST_GGUF=<path to a GGUF (a
// header-only copy is enough)> prints the estimate for a real model. Without
// the variable doctest skips it, so it is neither listed (CTest does not
// register it) nor counted as a pass.
TEST_CASE("VRAM estimate: a real GGUF header (opt-in)" * doctest::skip(std::getenv("SONDER_TEST_GGUF") == nullptr)) {
    const char* path = std::getenv("SONDER_TEST_GGUF");
    if (path == nullptr || *path == '\0') {
        MESSAGE("SONDER_TEST_GGUF is empty; nothing to print");
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
    // `profiles` is added next to the fields every /v1/models response has:
    // api_version and the bound backend's features and pins.
    const json::Value* meta = doc.find("sonder");
    CHECK(meta->find("api_version")->as_int() == 1);
    REQUIRE(meta->find("features") != nullptr);
    CHECK(meta->find("features")->is_array());
    REQUIRE(meta->find("pins") != nullptr);
    CHECK(meta->find("pins")->find("mode")->as_string() == "override");
    CHECK(meta->as_object().size() == 4u);  // api_version, features, pins, profiles

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

namespace {

// Residency state of `id` in /v1/sonder/health ("missing" when absent).
std::string served_model_state(std::uint16_t port, const std::string& id) {
    const json::Value health = get(port, "/v1/sonder/health").json();
    for (const auto& m : health.find("models")->as_array()) {
        if (m.find("id")->as_string() == id) return m.find("state")->as_string();
    }
    return "missing";
}

// One launch-profile model ("qwen-profile", loaded as "mock:weights.gguf")
// with the measured profile's sampling defaults.
srv::ServerOptions profile_model_options(const std::shared_ptr<RecordingBackend>& backend) {
    auto o = Fixture::defaults();
    o.backend.backend.clear();
    o.backend_instance = backend;
    o.models = {"qwen-profile"};
    srv::ModelProfileBinding binding;
    binding.model = "qwen-profile";
    binding.load_name = "mock:weights.gguf";
    binding.sampling_defaults = parse_single(kMeasuredProfile).sampling;
    o.profile_bindings.push_back(binding);
    return o;
}

}  // namespace

TEST_CASE("models: a lazy launch profile model loads under its load name, and reloads so after an idle eviction") {
    auto backend = std::make_shared<RecordingBackend>();
    auto o = profile_model_options(backend);
    o.lazy_models = true;
    o.model_idle_ttl = std::chrono::milliseconds(150);
    Fixture f(o);
    {
        std::lock_guard<std::mutex> lock(backend->log->mu);
        CHECK(backend->log->loads.empty());  // lazy: nothing loads at start
    }
    // "qwen-profile" is not a mock model name: a load under the served id
    // fails (404), so a 200 proves the residency loader used the load name.
    REQUIRE(post(f.port, "/v1/chat/completions", model_chat("qwen-profile")).status == 200);
    REQUIRE(eventually([&] { return served_model_state(f.port, "qwen-profile") == "unloaded"; }));
    REQUIRE(post(f.port, "/v1/chat/completions", model_chat("default")).status == 200);
    {
        std::lock_guard<std::mutex> lock(backend->log->mu);
        CHECK(backend->log->loads == std::vector<std::string>{"mock:weights.gguf", "mock:weights.gguf"});
        // The reloaded model still gets the profile's defaults ("default"
        // resolves to the served id before the binding is looked up).
        REQUIRE(backend->log->sampling.size() == 2);
        CHECK(backend->log->sampling[1].top_k == 20);
        CHECK(backend->log->sampling[1].is_explicit(si::SamplingConfig::kTopK));
    }
    const json::Value health = get(f.port, "/v1/sonder/health").json();
    CHECK(health.find("residency")->find("loads")->as_int() == 2);
    CHECK(health.find("residency")->find("load_failures")->as_int() == 0);
}

TEST_CASE("messages: launch profile sampling defaults reach the backend through /v1/messages too") {
    auto backend = std::make_shared<RecordingBackend>();
    Fixture f(profile_model_options(backend));
    const std::string hello = R"("messages":[{"role":"user","content":"hello"}]})";
    REQUIRE(post(f.port, "/v1/messages", R"({"model":"qwen-profile","max_tokens":8,)" + hello).status == 200);
    // The caller's own value still wins.
    REQUIRE(post(f.port, "/v1/messages", R"({"model":"qwen-profile","max_tokens":8,"temperature":0.3,)" + hello)
                .status == 200);
    std::lock_guard<std::mutex> lock(backend->log->mu);
    REQUIRE(backend->log->sampling.size() == 2);
    const si::SamplingConfig& d = backend->log->sampling[0];
    CHECK(d.temperature == doctest::Approx(1.0));
    CHECK(d.top_p == doctest::Approx(0.95));
    CHECK(d.top_k == 20);
    CHECK(d.min_p == doctest::Approx(0.0));
    CHECK(d.is_explicit(si::SamplingConfig::kTemperature));
    CHECK(d.is_explicit(si::SamplingConfig::kTopP));
    CHECK(d.is_explicit(si::SamplingConfig::kTopK));
    CHECK(d.is_explicit(si::SamplingConfig::kMinP));
    CHECK_FALSE(d.is_explicit(si::SamplingConfig::kPresencePenalty));
    CHECK(d.max_tokens == 8);
    const si::SamplingConfig& c = backend->log->sampling[1];
    CHECK(c.temperature == doctest::Approx(0.3));
    CHECK(c.top_k == 20);
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
    CHECK(meta.size() == 3u);  // api_version, and the backend's features and pins (health_features)
    CHECK(meta.find("api_version") != nullptr);
    CHECK(meta.find("features") != nullptr);
    CHECK(meta.find("pins") != nullptr);
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

TEST_CASE("serve_main: --profile refuses the llama.cpp context flags its typed fields own") {
    const auto model = write_temp("model.gguf", hybrid_fixture());
    const std::string m = model.generic_string();
    const auto profiles = write_temp("profiles.json", R"({"profiles":[
        {"name":"server","backend":"llamaserver","model":")" + m + R"(","ctx_size":512},
        {"name":"direct","backend":"llamacpp","model":")" + m + R"(","ctx_size":512}]})");
    const std::string pf = profiles.string();
    std::string err;
    for (const std::string name : {"server", "direct"}) {
        for (const auto& [flag, value] : std::vector<std::pair<std::string, std::string>>{{"--batch-size", "1024"},
                                                                                          {"--ubatch-size", "256"},
                                                                                          {"--cache-type-k", "q8_0"},
                                                                                          {"--cache-type-v", "q8_0"},
                                                                                          {"--flash-attn", "on"}}) {
            CAPTURE(name);
            CAPTURE(flag);
            // `--max-connections 0` is a later usage error, so a missing
            // check exits with another message instead of serving.
            CHECK(serve({"--profiles", pf, "--profile", name, flag, value, "--max-connections", "0"}, err) == 2);
            CHECK(contains(err, flag + " conflicts with --profile (set the profile's typed fields)"));
        }
    }
    std::filesystem::remove(model);
    std::filesystem::remove(profiles);
}

TEST_CASE("serve_main: --vram-budget-mib overrides the profile's vram_budget_mib") {
    // The fixture needs the 256 MiB runtime allowance: 100 MiB never fits,
    // 100000 MiB always does.
    const auto model = write_temp("model.gguf", hybrid_fixture());
    const std::string m = model.generic_string();
    const std::string fixed = R"(","ctx_size":512,"n_gpu_layers":-1,"fit":false,"vram_budget_mib":)";
    const auto profiles = write_temp("profiles.json", R"({"profiles":[
        {"name":"loose","backend":"llamaserver","model":")" + m + fixed + R"(100000},
        {"name":"tight","backend":"llamaserver","model":")" + m + fixed + R"(100}]})");
    const std::string pf = profiles.string();
    // `--max-connections 0` is a later usage error: a profile that passes the
    // fit check exits there instead of starting llama-server.
    const auto run = [&pf](const std::string& name, const std::optional<std::string>& budget, std::string& err) {
        std::vector<std::string> args{"--profiles", pf, "--profile", name, "--llamaserver-executable",
                                      "llama-server-not-started", "--max-connections", "0"};
        if (budget) {
            args.emplace_back("--vram-budget-mib");
            args.push_back(*budget);
        }
        return serve(args, err);
    };
    std::string err;
    // A budget tightened on the command line applies over the profile's.
    CHECK(run("loose", std::string("100"), err) == 2);
    CHECK(contains(err, "does not fit"));
    CHECK(contains(err, "vs budget 100 MiB"));
    // A looser one too: the profile gets past the fit check.
    CHECK(run("tight", std::string("100000"), err) == 2);
    CHECK_FALSE(contains(err, "does not fit"));
    CHECK(contains(err, "--max-connections must be from 1 to 100000"));
    // Without the flag, each profile's own budget applies.
    CHECK(run("tight", std::nullopt, err) == 2);
    CHECK(contains(err, "vs budget 100 MiB"));
    CHECK(run("loose", std::nullopt, err) == 2);
    CHECK_FALSE(contains(err, "does not fit"));
    CHECK(contains(err, "--max-connections must be from 1 to 100000"));
    std::filesystem::remove(model);
    std::filesystem::remove(profiles);
}

TEST_CASE("serve_main: the fit check sizes sliding-window layers as llama.cpp does") {
    ScopedSwaFullEnvironment environment;
    // The Gemma 4 header at its full 262,144 context needs about 4.7 GiB
    // (KV 4,576 MiB + compute); counting every SWA block at full context made
    // it 164 GiB, and serve refused a model that fits with exit status 2.
    const auto model = write_temp("gemma4.gguf", gemma4_fixture());
    const std::string m = model.generic_string();
    const std::string fixed = R"(","ctx_size":262144,"parallel":1,"kv_unified":true,"n_gpu_layers":-1,"fit":false})";
    const auto profiles = write_temp("profiles.json", R"({"profiles":[{"name":"g4","backend":"llamaserver","model":")" +
                                                          m + fixed + "]}");
    const std::string pf = profiles.string();
    const auto run = [&pf](const std::string& budget, std::string& err) {
        // `--max-connections 0` is a later usage error: a profile that passes
        // the fit check exits there instead of starting llama-server.
        return serve({"--profiles", pf, "--profile", "g4", "--vram-budget-mib", budget, "--llamaserver-executable",
                      "llama-server-not-started", "--max-connections", "0"},
                     err);
    };
    std::string err;
    CHECK(run("6000", err) == 2);
    CHECK_FALSE(contains(err, "does not fit"));
    CHECK(contains(err, "--max-connections must be from 1 to 100000"));
    // Below the estimate it is refused, and the log names both caches.
    CHECK(run("4000", err) == 2);
    CHECK(contains(err, "does not fit"));
    CHECK(contains(err, "KV 4576 = (8 full-attention layers x 262144 tokens + 40 sliding-window layers x 1536 "
                        "tokens) x 1 sequence(s)"));
    // The child inherits LLAMA_ARG_SWA_FULL. A nonempty setting must not
    // let the CLI accept the small-window budget while allocating a full
    // cache. False-looking values conservatively count full size as well.
    for (const char* value : {"1", "0"}) {
        CAPTURE(value);
        environment.set(value);
        CHECK(run("6000", err) == 2);
        CHECK(contains(err, "does not fit"));
        CHECK(contains(err, "LLAMA_ARG_SWA_FULL"));
        CHECK(contains(err, "40 sliding-window layers x 262144 tokens"));
    }
    environment.set("");
    CHECK(run("6000", err) == 2);
    CHECK_FALSE(contains(err, "does not fit"));
    CHECK(contains(err, "--max-connections must be from 1 to 100000"));
    std::filesystem::remove(model);
    std::filesystem::remove(profiles);
}

TEST_CASE("serve_main: a llamaserver profile refuses the llamacpp-only tensor placement flags") {
    const auto model = write_temp("model.gguf", hybrid_fixture());
    const std::string m = model.generic_string();
    const auto profiles = write_temp("profiles.json", R"({"profiles":[
        {"name":"server","backend":"llamaserver","model":")" + m + R"(","ctx_size":512},
        {"name":"direct","backend":"llamacpp","model":")" + m + R"(","ctx_size":512}]})");
    const std::string pf = profiles.string();
    std::string err;
    for (const auto& [flag, value] : std::vector<std::pair<std::string, std::string>>{
             {"--moe-experts", "cpu"}, {"--tensor-override", "ffn_.*_exps=cpu"}}) {
        CAPTURE(flag);
        // `--max-connections 0` is a later usage error, so a missing check
        // exits with another message instead of serving.
        CHECK(serve({"--profiles", pf, "--profile", "server", flag, value, "--llamaserver-executable",
                     "llama-server-not-started", "--max-connections", "0"},
                    err) == 2);
        CHECK(contains(err, "option " + flag + " applies only to --backend llamacpp; for llamaserver launch profile "
                                               "'server' put llama-server's own flag (-ot, --cpu-moe) in extra_args"));
        // The direct backend reads them: a llamacpp profile still takes them.
        CHECK(serve({"--profiles", pf, "--profile", "direct", flag, value, "--max-connections", "0"}, err) == 2);
        CHECK_FALSE(contains(err, "applies only to --backend llamacpp"));
        CHECK(contains(err, "--max-connections must be from 1 to 100000"));
    }
    std::filesystem::remove(model);
    std::filesystem::remove(profiles);
}
