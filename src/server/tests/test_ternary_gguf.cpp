// Ternary / 1-bit / FP4 / FP8 / PrismML GGUF header parsing, new LaunchProfile
// typed fields (multi-GPU, load_mode, no_repack, reasoning), KV cache types,
// and multi-GPU tensor_split VRAM estimation.
#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sonder/inference/json.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/server.hpp"

namespace si = sonder::inference;
namespace srv = sonder::inference::server;
namespace json = sonder::inference::json;

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;

bool contains(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

si::LaunchProfile parse_single(const std::string& profile_json) {
    auto doc = json::parse("{\"profiles\":[" + profile_json + "]}");
    REQUIRE(doc.ok());
    auto parsed = si::parse_launch_profiles(doc.value());
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    REQUIRE(parsed.value().size() == 1u);
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

// Minimal GGUF v3 binary header builder for synthetic ternary/PrismML fixtures.
class TernaryGgufWriter {
public:
    void u32(std::uint32_t v) { raw(&v, 4); }
    void u64(std::uint64_t v) { raw(&v, 8); }
    void str(const std::string& s) {
        u64(s.size());
        bytes_.append(s);
    }
    void kv_str(const std::string& key, const std::string& v) {
        str(key);
        u32(8);
        str(v);
    }
    void kv_u32(const std::string& key, std::uint32_t v) {
        str(key);
        u32(4);
        u32(v);
    }
    void kv_i32(const std::string& key, std::int32_t v) {
        str(key);
        u32(5);
        raw(&v, 4);
    }
    void kv_bool(const std::string& key, bool v) {
        str(key);
        u32(7);
        const std::uint8_t b = v ? 1u : 0u;
        raw(&b, 1);
    }
    void kv_i8_array(const std::string& key, std::size_t count, std::int8_t fill) {
        str(key);
        u32(9);
        u32(1);  // GGUF_TYPE_INT8
        u64(count);
        bytes_.append(count, static_cast<char>(fill));
    }
    void kv_i32_array(const std::string& key, const std::vector<std::int32_t>& values) {
        str(key);
        u32(9);
        u32(5);  // GGUF_TYPE_INT32
        u64(values.size());
        for (const std::int32_t v : values) raw(&v, 4);
    }
    void tensor(const std::string& name, const std::vector<std::uint64_t>& dims, std::uint32_t type) {
        str(name);
        u32(static_cast<std::uint32_t>(dims.size()));
        for (const std::uint64_t d : dims) u64(d);
        u32(type);
        u64(0);
    }
    std::string finish(std::uint64_t tensors, std::uint64_t kvs) const {
        std::string head = "GGUF";
        const std::uint32_t version = 3;
        head.append(reinterpret_cast<const char*>(&version), 4);
        head.append(reinterpret_cast<const char*>(&tensors), 8);
        head.append(reinterpret_cast<const char*>(&kvs), 8);
        return head + bytes_;
    }

private:
    void raw(const void* p, std::size_t n) { bytes_.append(static_cast<const char*>(p), n); }
    std::string bytes_;
};

si::GgufModelInfo multi_gpu_model() {
    si::GgufModelInfo m;
    m.architecture = "llama";
    m.block_count = 4;
    m.embedding_length = 2048;
    m.vocab_size = 32000;
    m.key_length = 128;
    m.value_length = 128;
    m.kv_heads = {8, 8, 8, 8};
    m.block_bytes = {1024 * kMiB, 1024 * kMiB, 1024 * kMiB, 1024 * kMiB};
    m.output_bytes = 512 * kMiB;
    m.embedding_bytes = 256 * kMiB;
    m.metadata = {{"general.architecture", "llama"}, {"llama.context_length", "8192"}};
    return m;
}

}  // namespace

// ------------------------------------------------ ternary / PrismML GGUF types

TEST_CASE("ternary GGUF: all 7 new ggml tensor types compute exact block sizes and classification flags") {
    struct TypeCase {
        std::uint32_t type;
        std::uint64_t block_elems;
        std::uint64_t block_bytes;
        bool ternary;
        bool one_bit;
        bool prism_private;
        bool fp4_or_fp8;
    };
    const std::vector<TypeCase> cases{
        {40, 64, 36, false, false, false, true},   // NVFP4
        {41, 128, 18, false, true, false, false},  // Q1_0 (Bonsai-1)
        {42, 64, 18, true, false, false, false},   // Q2_0 (mainline 2-bit ternary)
        {43, 1, 1, false, false, false, true},     // F8_E4M3
        {142, 128, 34, true, false, true, false},  // PQ2_0 (PrismML Bonsai-2)
        {143, 128, 28, true, false, true, false},  // PTQ1_0 (PrismML Bonsai-2 dense ternary)
        {145, 128, 36, true, false, true, false},  // PQ2_0M (PrismML KV-only)
    };
    for (const TypeCase& c : cases) {
        CAPTURE(c.type);
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.kv_u32("llama.embedding_length", 128);
        w.kv_u32("llama.attention.head_count", 4);
        w.kv_u32("llama.attention.head_count_kv", 4);
        // One exact block and one 16-block matrix.
        w.tensor("blk.0.attn_q.weight", {c.block_elems}, c.type);
        w.tensor("blk.0.ffn_up.weight", {c.block_elems, 16}, c.type);
        const auto parsed = si::parse_gguf_model_info(w.finish(2, 5));
        REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
        const si::GgufModelInfo& info = parsed.value();
        REQUIRE(info.tensors.size() == 2u);
        CHECK(info.tensors[0].type == c.type);
        CHECK(info.tensors[0].bytes == c.block_bytes);
        CHECK(info.tensors[1].bytes == 16u * c.block_bytes);
        CHECK(info.block_bytes[0] == 17u * c.block_bytes);
        CHECK(info.has_ternary_tensors == c.ternary);
        CHECK(info.has_one_bit_tensors == c.one_bit);
        CHECK(info.has_prism_private_tensors == c.prism_private);
        CHECK(info.has_fp4_or_fp8_tensors == c.fp4_or_fp8);
        CHECK_FALSE(info.prism_hadamard);
    }

    // Existing ternary (TQ1_0=34, TQ2_0=35) and FP4 (MXFP4=39) also classify cleanly.
    for (const std::uint32_t t : {34u, 35u}) {
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.tensor("blk.0.attn_q.weight", {256}, t);
        const auto parsed = si::parse_gguf_model_info(w.finish(1, 2));
        REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
        CHECK(parsed->has_ternary_tensors);
        CHECK_FALSE(parsed->has_prism_private_tensors);
        CHECK_FALSE(parsed->has_one_bit_tensors);
        CHECK_FALSE(parsed->has_fp4_or_fp8_tensors);
    }
    {
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.tensor("blk.0.attn_q.weight", {32}, 39);
        const auto parsed = si::parse_gguf_model_info(w.finish(1, 2));
        REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
        CHECK(parsed->has_fp4_or_fp8_tensors);
        CHECK_FALSE(parsed->has_ternary_tensors);
    }
    // Unassigned type IDs remain rejected.
    for (const std::uint32_t unknown : {44u, 144u, 146u}) {
        CAPTURE(unknown);
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.tensor("blk.0.x", {128}, unknown);
        const auto parsed = si::parse_gguf_model_info(w.finish(1, 2));
        CHECK_FALSE(parsed.ok());
        CHECK(contains(parsed.status().message(), "unknown ggml type"));
    }
}

TEST_CASE("ternary GGUF: uint64 tensor size and accumulation overflows are rejected without wraparound") {
    constexpr std::uint64_t kMaxU64 = std::numeric_limits<std::uint64_t>::max();
    // Quantized types with block_bytes < block_elems near UINT64_MAX elements:
    // `(elements + block_elems - 1)` used to wrap around to ~0; exact block
    // division computes the true byte count without wraparound.
    {
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.tensor("blk.0.pq2.weight", {kMaxU64}, 142);  // PQ2_0 (128, 34)
        const auto parsed = si::parse_gguf_model_info(w.finish(1, 2));
        REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
        const std::uint64_t expected_blocks = kMaxU64 / 128u + 1u;
        CHECK(parsed->block_bytes[0] == expected_blocks * 34u);
    }
    {
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.tensor("blk.0.nvfp4.weight", {kMaxU64 - 10u}, 40);  // NVFP4 (64, 36)
        const auto parsed = si::parse_gguf_model_info(w.finish(1, 2));
        REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
        const std::uint64_t expected_blocks = (kMaxU64 - 10u) / 64u + 1u;
        CHECK(parsed->block_bytes[0] == expected_blocks * 36u);
    }

    // Single tensor where blocks * block_bytes overflows uint64_t is rejected.
    for (const auto& [type, elems] : std::vector<std::pair<std::uint32_t, std::uint64_t>>{
             {0u, (kMaxU64 / 4u) + 1u},   // F32 (1, 4): blocks * 4 overflows
             {1u, (kMaxU64 / 2u) + 1u},   // F16 (1, 2): blocks * 2 overflows
             {28u, (kMaxU64 / 8u) + 1u},  // F64 (1, 8): blocks * 8 overflows
         }) {
        CAPTURE(type);
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.tensor("blk.0.huge.weight", {elems}, type);
        const auto parsed = si::parse_gguf_model_info(w.finish(1, 2));
        REQUIRE_FALSE(parsed.ok());
        CHECK(parsed.status().code() == si::ErrorCode::invalid_argument);
        CHECK(contains(parsed.status().message(), "tensor too large"));
    }

    // Accumulation overflow across two valid F8_E4M3 (1 elem = 1 byte) tensors
    // in every bucket: block_bytes, embedding_bytes, output_bytes, other_bytes.
    const std::uint64_t over_half = (kMaxU64 / 2u) + 100u;
    const std::vector<std::pair<std::string, std::string>> bucket_pairs{
        {"blk.0.a.weight", "blk.0.b.weight"},
        {"token_embd.weight", "token_embd.weight"},
        {"output.weight", "output_norm.weight"},
        {"misc.a", "misc.b"},
    };
    for (const auto& [t1, t2] : bucket_pairs) {
        CAPTURE(t1);
        TernaryGgufWriter w;
        w.kv_str("general.architecture", "llama");
        w.kv_u32("llama.block_count", 1);
        w.tensor(t1, {over_half}, 43);  // F8_E4M3: 1 byte per element
        w.tensor(t2, {over_half}, 43);
        const auto parsed = si::parse_gguf_model_info(w.finish(2, 2));
        REQUIRE_FALSE(parsed.ok());
        CHECK(parsed.status().code() == si::ErrorCode::invalid_argument);
        CHECK(contains(parsed.status().message(), "tensor sizes overflow"));
    }
}

TEST_CASE("ternary GGUF: prism.hadamard metadata and large sign_values arrays parse without bloating metadata") {
    TernaryGgufWriter w;
    w.kv_str("general.architecture", "qwen35");
    w.kv_u32("qwen35.block_count", 2);
    w.kv_u32("qwen35.embedding_length", 128);
    w.kv_u32("qwen35.context_length", 8192);
    w.kv_u32("qwen35.attention.head_count", 4);
    w.kv_u32("qwen35.attention.head_count_kv", 2);
    w.kv_str("prism.hadamard.transform", "fwht");
    w.kv_u32("prism.hadamard.block_size", 4096);
    w.kv_bool("prism.hadamard.gdn_v_grouped", true);
    // Large array like Nathan's real Bonsai-2 GGUFs (28,672 elements) plus a
    // smaller <=4096 prism.hadamard array: both must be skipped cleanly.
    w.kv_i8_array("prism.hadamard.sign_values", 28672, -1);
    w.kv_i32_array("prism.hadamard.small_signs", std::vector<std::int32_t>(1024, 1));
    w.tensor("blk.0.ffn_gate.weight", {128, 32}, 142);  // PQ2_0: 32 * 34 = 1088 B
    w.tensor("blk.1.ffn_down.weight", {128, 32}, 143);  // PTQ1_0: 32 * 28 = 896 B
    const auto parsed = si::parse_gguf_model_info(w.finish(2, 11));
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    const si::GgufModelInfo& info = parsed.value();
    CHECK(info.prism_hadamard);
    CHECK(info.prism_hadamard_transform == "fwht");
    CHECK(info.prism_hadamard_block_size == 4096u);
    CHECK(info.has_ternary_tensors);
    CHECK(info.has_prism_private_tensors);
    CHECK_FALSE(info.has_one_bit_tensors);
    CHECK_FALSE(info.has_fp4_or_fp8_tensors);
    CHECK(info.block_bytes == std::vector<std::uint64_t>{1088u, 896u});
    for (const auto& [k, v] : info.metadata) {
        (void)v;
        CHECK(k != "prism.hadamard.sign_values");
        CHECK(k != "prism.hadamard.small_signs");
    }

    // Also verify i32 scalar block_size (1024).
    TernaryGgufWriter w2;
    w2.kv_str("general.architecture", "llama");
    w2.kv_u32("llama.block_count", 1);
    w2.kv_str("prism.hadamard.transform", "walsh");
    w2.kv_i32("prism.hadamard.block_size", 1024);
    const auto parsed2 = si::parse_gguf_model_info(w2.finish(0, 4));
    REQUIRE_MESSAGE(parsed2.ok(), parsed2.status().message());
    CHECK(parsed2->prism_hadamard);
    CHECK(parsed2->prism_hadamard_transform == "walsh");
    CHECK(parsed2->prism_hadamard_block_size == 1024u);

    // Estimator emits a PrismML runtime diagnostic note and propagates flags.
    const si::LaunchProfile p = parse_single(
        R"({"name":"bonsai-2","backend":"llamaserver","model":"b2.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1})");
    const si::VramEstimate est = si::estimate_vram(info, p);
    CHECK(est.prism_hadamard);
    CHECK(est.requires_prism_runtime);
    bool prism_noted = false;
    for (const auto& note : est.notes) {
        prism_noted = prism_noted || (contains(note, "PrismML Hadamard-capable runtime") && contains(note, "PQ2_0/PTQ1_0") &&
                                      contains(note, "fwht") && contains(note, "4096"));
    }
    CHECK(prism_noted);

    const json::Object meta = si::launch_profile_metadata(p, est, 16000);
    REQUIRE(meta.find("prism_hadamard") != nullptr);
    CHECK(meta.find("prism_hadamard")->as_bool());
    REQUIRE(meta.find("requires_prism_runtime") != nullptr);
    CHECK(meta.find("requires_prism_runtime")->as_bool());
}

// ---------------------------------------- LaunchProfile new typed fields & argv

TEST_CASE("launch profile: new multi-GPU, load_mode, no_repack, reasoning fields parse and build argv") {
    const si::LaunchProfile p = parse_single(R"({
        "name": "bonsai-dual-gpu",
        "backend": "llamaserver",
        "model": "D:/models/Bonsai-2-27B.gguf",
        "ctx_size": 32768,
        "n_gpu_layers": -1,
        "device": "CUDA0,CUDA1",
        "main_gpu": 0,
        "split_mode": "layer",
        "tensor_split": [3, 1],
        "load_mode": "dio",
        "no_repack": true,
        "parallel": 1,
        "flash_attn": "on",
        "cache_type_k": "f8_e4m3",
        "cache_type_v": "pq2_0m",
        "reasoning": "on",
        "reasoning_budget": 4096
    })");
    CHECK(p.device == std::string("CUDA0,CUDA1"));
    CHECK(p.main_gpu == 0);
    CHECK(p.split_mode == std::string("layer"));
    REQUIRE(p.tensor_split.size() == 2u);
    CHECK(p.tensor_split[0] == doctest::Approx(3.0));
    CHECK(p.tensor_split[1] == doctest::Approx(1.0));
    CHECK(p.load_mode == std::string("dio"));
    CHECK(p.no_repack == true);
    CHECK(p.cache_type_k == std::string("f8_e4m3"));
    CHECK(p.cache_type_v == std::string("pq2_0m"));
    CHECK(p.reasoning == std::string("on"));
    CHECK(p.reasoning_budget == 4096);

    const auto argv = si::llamaserver_arguments(p);
    REQUIRE_MESSAGE(argv.ok(), argv.status().message());
    const std::vector<std::string> expected{
        "--model",      "D:/models/Bonsai-2-27B.gguf",
        "--alias",      "bonsai-dual-gpu",
        "--ctx-size",   "32768",
        "--n-gpu-layers", "all",
        "--device",     "CUDA0,CUDA1",
        "--main-gpu",   "0",
        "--split-mode", "layer",
        "--tensor-split", "3,1",
        "--load-mode",  "dio",
        "--no-repack",
        "--parallel",   "1",
        "--flash-attn", "on",
        "--cache-type-k", "f8_e4m3",
        "--cache-type-v", "pq2_0m",
        "--reasoning",  "on",
        "--reasoning-budget", "4096",
    };
    CHECK(argv.value() == expected);

    // Fractional tensor_split, no_repack=false, reasoning_budget=-1.
    const si::LaunchProfile p2 = parse_single(R"({
        "name": "rocm-split",
        "backend": "llamaserver",
        "model": "m.gguf",
        "device": "ROCm0",
        "split_mode": "row",
        "tensor_split": [0.6, 0.4],
        "load_mode": "mlock",
        "no_repack": false,
        "reasoning": "auto",
        "reasoning_budget": -1
    })");
    const auto argv2 = si::llamaserver_arguments(p2);
    REQUIRE_MESSAGE(argv2.ok(), argv2.status().message());
    const std::vector<std::string> expected2{
        "--model", "m.gguf", "--alias", "rocm-split", "--device", "ROCm0", "--split-mode", "row",
        "--tensor-split", "0.6,0.4", "--load-mode", "mlock", "--repack", "--reasoning", "auto",
        "--reasoning-budget", "-1"};
    CHECK(argv2.value() == expected2);
}

TEST_CASE("launch profile: new typed field validation, extra_args rejection, and llamacpp refusal") {
    const std::string base = R"("name":"p","backend":"llamaserver","model":"m.gguf")";

    // Schema & range rejections for new typed fields.
    CHECK(contains(parse_error("{" + base + R"(,"device":""})"), "'device'"));
    CHECK(contains(parse_error("{" + base + R"(,"device":"-bad"})"), "'device'"));
    CHECK(contains(parse_error("{" + base + R"(,"device":"CUDA0,"})"), "'device'"));
    CHECK(contains(parse_error("{" + base + R"(,"device":"CUDA0 --host"})"), "'device'"));
    CHECK(contains(parse_error("{" + base + R"(,"main_gpu":-1})"), "'main_gpu'"));
    CHECK(contains(parse_error("{" + base + R"(,"split_mode":"tensor"})"), "'split_mode'"));
    CHECK(contains(parse_error("{" + base + R"(,"tensor_split":[0,1]})"), "'tensor_split'"));
    CHECK(contains(parse_error("{" + base + R"(,"tensor_split":[-1,2]})"), "'tensor_split'"));
    CHECK(contains(parse_error("{" + base + R"(,"tensor_split":"1,1"})"), "'tensor_split'"));
    CHECK(contains(parse_error("{" + base + R"(,"tensor_split":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]})"),
                   "at most 16"));
    CHECK(contains(parse_error("{" + base + R"(,"load_mode":"direct"})"), "'load_mode'"));
    CHECK(contains(parse_error("{" + base + R"(,"no_repack":1})"), "'no_repack'"));
    CHECK(contains(parse_error("{" + base + R"(,"reasoning":"maybe"})"), "'reasoning'"));
    CHECK(contains(parse_error("{" + base + R"(,"reasoning_budget":-2})"), "'reasoning_budget'"));
    // Quantized FP8 / 2-bit V caches still require flash attention.
    for (const std::string kv : {"f8_e4m3", "q2_0", "pq2_0m"}) {
        CAPTURE(kv);
        CHECK(contains(parse_error("{" + base + R"(,"cache_type_v":")" + kv + R"(","flash_attn":"off"})"),
                       "requires flash attention"));
        CHECK(parse_single("{" + base + R"(,"cache_type_v":")" + kv + R"(","flash_attn":"on"})").cache_type_v == kv);
    }

    // Non-finite tensor_split rejected in validate_launch_profile.
    {
        si::LaunchProfile bad_nan = parse_single("{" + base + "}");
        bad_nan.tensor_split = {1.0, std::numeric_limits<double>::quiet_NaN()};
        CHECK_FALSE(si::validate_launch_profile(bad_nan).ok());
        bad_nan.tensor_split = {1.0, std::numeric_limits<double>::infinity()};
        CHECK_FALSE(si::validate_launch_profile(bad_nan).ok());
    }

    // extra_args refuses every short, long, =value, and underscore spelling of the new typed fields.
    const auto with = [](const std::string& arg) {
        return R"({"name":"p","backend":"llamaserver","model":"m.gguf","extra_args":[)" + arg + "]}";
    };
    const std::vector<std::pair<std::string, std::string>> typed_flags{
        {R"("-dev","CUDA0")", "device"},
        {R"("--device","Vulkan0")", "device"},
        {R"("--device=ROCm0")", "device"},
        {R"("-mg","0")", "main_gpu"},
        {R"("--main-gpu","1")", "main_gpu"},
        {R"("--main_gpu","1")", "main_gpu"},
        {R"("-sm","layer")", "split_mode"},
        {R"("--split-mode","row")", "split_mode"},
        {R"("--split_mode","none")", "split_mode"},
        {R"("-ts","1,1")", "tensor_split"},
        {R"("--tensor-split","3,1")", "tensor_split"},
        {R"("--tensor_split=2,1")", "tensor_split"},
        {R"("-lm","dio")", "load_mode"},
        {R"("--load-mode","mmap")", "load_mode"},
        {R"("--load_mode","mlock")", "load_mode"},
        {R"("-nr")", "no_repack"},
        {R"("--no-repack")", "no_repack"},
        {R"("--no_repack")", "no_repack"},
        {R"("--repack")", "no_repack"},
        {R"("-rea","off")", "reasoning"},
        {R"("--reasoning","off")", "reasoning"},
        {R"("--reasoning-budget","0")", "reasoning_budget"},
        {R"("--reasoning_budget","-1")", "reasoning_budget"},
    };
    for (const auto& [flag, field] : typed_flags) {
        CAPTURE(flag);
        CHECK(contains(parse_error(with(flag)), "use the typed field '" + field + "'"));
    }

    // Remote P2P tunnel flags and newly blocked unsafe UI/tool/media flags are refused.
    for (const std::string remote : {R"("--connect")", R"("--connect-code","abc")", R"("--connect_code","abc")"}) {
        CAPTURE(remote);
        CHECK(contains(parse_error(with(remote)), "remote RPC servers"));
    }
    for (const std::string unsafe :
         {R"("--ui-config-file","cfg.json")", R"("--ui_config_file","cfg.json")",
          R"("--webui-config-file","cfg.json")", R"("--media-allow-private-urls")",
          R"("--allow-unauthenticated-tools")", R"("--ui-mcp-proxy-allow-private-urls")"}) {
        CAPTURE(unsafe);
        CHECK(contains(parse_error(with(unsafe)), "agent tools, MCP and local file serving"));
    }

    // CR/LF control characters are rejected in model, mmproj, draft_model, and extra_args flags.
    for (const char ctrl : {'\r', '\n'}) {
        CAPTURE(static_cast<int>(ctrl));
        {
            si::LaunchProfile bad = parse_single("{" + base + "}");
            bad.model = std::string("m") + ctrl + ".gguf";
            CHECK(si::validate_launch_profile(bad).code() == si::ErrorCode::invalid_argument);
        }
        {
            si::LaunchProfile bad = parse_single("{" + base + "}");
            bad.mmproj = std::string("mm") + ctrl + ".gguf";
            CHECK(si::validate_launch_profile(bad).code() == si::ErrorCode::invalid_argument);
        }
        {
            si::LaunchProfile bad = parse_single("{" + base + "}");
            bad.speculative = si::SpeculativeSettings{{"draft-simple"}, 2, std::string("d") + ctrl + ".gguf"};
            CHECK(si::validate_launch_profile(bad).code() == si::ErrorCode::invalid_argument);
        }
        {
            si::LaunchProfile bad = parse_single("{" + base + "}");
            bad.extra_args = {std::string("--no-warmup") + ctrl};
            CHECK(si::validate_launch_profile(bad).code() == si::ErrorCode::invalid_argument);
        }
    }

    // Direct llamacpp backend rejects every new unsupported field without mutating setup/device.
    const std::vector<std::pair<std::string, std::string>> llamacpp_rejected{
        {R"("device":"CUDA0")", "device"},
        {R"("main_gpu":0)", "main_gpu"},
        {R"("split_mode":"layer")", "split_mode"},
        {R"("tensor_split":[1,1])", "tensor_split"},
        {R"("load_mode":"mmap")", "load_mode"},
        {R"("no_repack":true)", "no_repack"},
        {R"("reasoning":"off")", "reasoning"},
        {R"("reasoning_budget":0)", "reasoning_budget"},
        {R"("cache_type_k":"f8_e4m3")", "cache_type_k"},
        {R"("cache_type_v":"pq2_0m")", "cache_type_v"},
        {R"("cache_type_k":"q2_0")", "cache_type_k"},
    };
    for (const auto& [field_json, field_name] : llamacpp_rejected) {
        CAPTURE(field_json);
        const si::LaunchProfile p =
            parse_single(R"({"name":"d","backend":"llamacpp","model":"m.gguf",)" + field_json + "}");
        si::BackendSetup setup;
        std::string dev;
        const si::Status st = si::apply_llamacpp_profile(p, setup, dev);
        REQUIRE_FALSE(st.ok());
        CHECK(st.code() == si::ErrorCode::invalid_argument);
        CHECK(contains(st.message(), "'" + field_name + "'"));
        CHECK(contains(st.message(), "not supported by the llamacpp backend; use llamaserver"));
        CHECK(dev.empty());
    }
}

// ------------------------------- KV cache bytes & multi-GPU tensor_split VRAM

TEST_CASE("VRAM estimate: f8_e4m3, q2_0 and pq2_0m KV cache byte rates") {
    CHECK(si::kv_cache_type_bytes("f8_e4m3") == doctest::Approx(1.0));
    CHECK(si::kv_cache_type_bytes("q2_0") == doctest::Approx(18.0 / 64.0));
    CHECK(si::kv_cache_type_bytes("pq2_0m") == doctest::Approx(36.0 / 128.0));
    CHECK(si::is_quantized_kv_cache_type("f8_e4m3"));
    CHECK(si::is_quantized_kv_cache_type("q2_0"));
    CHECK(si::is_quantized_kv_cache_type("pq2_0m"));

    const si::GgufModelInfo m = multi_gpu_model();
    const si::LaunchProfile p_f16 = parse_single(
        R"({"name":"k","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"cache_type_k":"f16","cache_type_v":"f16"})");
    const si::LaunchProfile p_fp8 = parse_single(
        R"({"name":"k","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"cache_type_k":"f8_e4m3","cache_type_v":"f8_e4m3","flash_attn":"on"})");
    const si::LaunchProfile p_pq2 = parse_single(
        R"({"name":"k","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"cache_type_k":"pq2_0m","cache_type_v":"pq2_0m","flash_attn":"on"})");
    const si::VramEstimate e_f16 = si::estimate_vram(m, p_f16);
    const si::VramEstimate e_fp8 = si::estimate_vram(m, p_fp8);
    const si::VramEstimate e_pq2 = si::estimate_vram(m, p_pq2);
    CHECK(e_fp8.kv_bytes == e_f16.kv_bytes / 2u);
    // pq2_0m is 36/128 = 9/32 bytes/elem vs f16 2 bytes/elem => ratio 9/64.
    CHECK(e_pq2.kv_bytes * 64u == e_f16.kv_bytes * 9u);
}

TEST_CASE("VRAM estimate: multi-GPU tensor_split computes max_gpu_bytes and surfaces per_gpu_max_mib") {
    const si::GgufModelInfo m = multi_gpu_model();

    // Single GPU: max_gpu_bytes stays 0 and metadata omits per_gpu_max_mib.
    const si::LaunchProfile single = parse_single(
        R"({"name":"one","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"fit":false})");
    const si::VramEstimate e_single = si::estimate_vram(m, single);
    CHECK(e_single.max_gpu_bytes == 0u);
    CHECK(si::launch_profile_metadata(single, e_single, 16000).find("per_gpu_max_mib") == nullptr);

    // 2 GPUs equal split [1, 1] with default split_mode (layer).
    const si::LaunchProfile equal = parse_single(
        R"({"name":"two","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"fit":false,"tensor_split":[1,1]})");
    const si::VramEstimate e_equal = si::estimate_vram(m, equal);
    const std::uint64_t splittable = e_equal.weights_bytes + e_equal.kv_bytes + e_equal.recurrent_bytes;
    const std::uint64_t expected_half =
        static_cast<std::uint64_t>(std::ceil(static_cast<double>(splittable) * 0.5)) + e_equal.compute_bytes;
    CHECK(e_equal.total_bytes == e_single.total_bytes);
    CHECK(e_equal.max_gpu_bytes == expected_half);
    CHECK(e_equal.max_gpu_mib() < e_equal.total_mib());
    bool split_noted = false;
    for (const auto& note : e_equal.notes) {
        CHECK_FALSE(contains(note, "ignored"));
        split_noted = split_noted || contains(note, "multi-GPU tensor_split");
    }
    CHECK(split_noted);
    const json::Object meta_equal = si::launch_profile_metadata(equal, e_equal, 4096);
    REQUIRE(meta_equal.find("per_gpu_max_mib") != nullptr);
    CHECK(static_cast<std::uint64_t>(meta_equal.find("per_gpu_max_mib")->as_int()) == e_equal.max_gpu_mib());

    // Asymmetric split [3, 1] with split_mode=layer: heaviest GPU gets 75% of (weights+KV+recurrent) + compute.
    const si::LaunchProfile asym_layer = parse_single(
        R"({"name":"asym-layer","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"fit":false,"split_mode":"layer","tensor_split":[3,1]})");
    const si::VramEstimate e_asym_layer = si::estimate_vram(m, asym_layer);
    const std::uint64_t expected_layer_3q =
        static_cast<std::uint64_t>(std::ceil(static_cast<double>(splittable) * 0.75)) + e_asym_layer.compute_bytes;
    CHECK(e_asym_layer.max_gpu_bytes == expected_layer_3q);
    CHECK(e_asym_layer.max_gpu_bytes > e_equal.max_gpu_bytes);

    // split_mode=row keeps KV cache unsplit on main_gpu (heaviest shard):
    // weights/recurrent split by 0.75, full kv_bytes + compute_bytes stay on main_gpu.
    const si::LaunchProfile asym_row = parse_single(
        R"({"name":"asym-row","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"fit":false,"split_mode":"row","tensor_split":[3,1]})");
    const si::VramEstimate e_asym_row = si::estimate_vram(m, asym_row);
    const std::uint64_t row_splittable = e_asym_row.weights_bytes + e_asym_row.recurrent_bytes;
    const std::uint64_t expected_row_3q =
        static_cast<std::uint64_t>(std::ceil(static_cast<double>(row_splittable) * 0.75)) + e_asym_row.kv_bytes +
        e_asym_row.compute_bytes;
    CHECK(e_asym_row.max_gpu_bytes == expected_row_3q);
    CHECK(e_asym_row.max_gpu_bytes - e_asym_layer.max_gpu_bytes == e_asym_row.kv_bytes / 4u);
    bool row_unsplit_noted = false;
    for (const auto& note : e_asym_row.notes) {
        row_unsplit_noted = row_unsplit_noted || (contains(note, "split_mode=row") && contains(note, "unsplit KV"));
    }
    CHECK(row_unsplit_noted);

    // Equal split [1, 1] with split_mode=row keeps full KV on main_gpu (+50% KV vs layer mode).
    const si::LaunchProfile equal_row = parse_single(
        R"({"name":"eq-row","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"fit":false,"split_mode":"row","tensor_split":[1,1]})");
    const si::VramEstimate e_equal_row = si::estimate_vram(m, equal_row);
    CHECK(e_equal_row.max_gpu_bytes - e_equal.max_gpu_bytes == e_equal_row.kv_bytes / 2u);

    // split_mode=none disables multi-GPU splitting even if tensor_split has >= 2 weights.
    const si::LaunchProfile none = parse_single(
        R"({"name":"none","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"fit":false,"split_mode":"none","tensor_split":[1,1]})");
    const si::VramEstimate e_none = si::estimate_vram(m, none);
    CHECK(e_none.max_gpu_bytes == 0u);
    bool none_noted = false;
    for (const auto& note : e_none.notes) {
        none_noted = none_noted || contains(note, "split_mode=none");
    }
    CHECK(none_noted);
}

TEST_CASE("VRAM estimate & catalog: device=none disables GPU offload and accounts everything on host") {
    const si::GgufModelInfo m = multi_gpu_model();
    for (const std::string dev : {"none", "NONE"}) {
        CAPTURE(dev);
        si::LaunchProfile p = parse_single(
            R"({"name":"cpu-dev","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,)"
            R"("mmproj":"mm.gguf","speculative":{"types":["draft-simple"],"draft_model":"d.gguf"},"device":")" +
            dev + R"("})");
        si::VramEstimateInputs in{};
        in.mmproj_bytes = 64 * kMiB;
        in.draft_model_bytes = 128 * kMiB;
        const si::VramEstimate e = si::estimate_vram(m, p, in);
        CHECK(e.gpu_layers == 0u);
        CHECK(e.weights_bytes == 0u);
        CHECK(e.kv_bytes == 0u);
        CHECK(e.recurrent_bytes == 0u);
        CHECK(e.compute_bytes == 0u);
        CHECK(e.total_bytes == 0u);
        CHECK(e.max_gpu_bytes == 0u);
        CHECK_FALSE(e.host_fit_applies);
        CHECK(e.host_kv_bytes > 0u);
        // All blocks (4096 MiB) + output (512 MiB) + embedding (256 MiB) + mmproj (64 MiB) + draft (128 MiB).
        CHECK(e.host_dense_weight_bytes == (4096u + 512u + 256u + 64u + 128u) * kMiB);
    }
}

TEST_CASE("VRAM estimate: typed load_mode and no_repack control host RAM accounting") {
    si::GgufModelInfo m = multi_gpu_model();
    m.routed_experts = si::RoutedExpertCounts{8, 2};
    m.tensors.clear();
    m.block_bytes.assign(4, 0);
    for (std::uint32_t b = 0; b < 4; ++b) {
        si::GgufTensorInfo t;
        t.name = "blk." + std::to_string(b) + ".ffn_up_exps.weight";
        t.bytes = 1000 * kMiB;
        t.block = b;
        t.expert = true;
        t.type = 12;  // Q4_K (repackable)
        m.block_bytes[b] = t.bytes;
        m.tensors.push_back(std::move(t));
    }

    const si::LaunchProfile default_p = parse_single(
        R"({"name":"h","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":0})");
    const si::LaunchProfile nr_p = parse_single(
        R"({"name":"h","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":0,"no_repack":true})");
    const si::LaunchProfile dio_p = parse_single(
        R"({"name":"h","backend":"llamaserver","model":"m.gguf","ctx_size":4096,"parallel":1,"n_gpu_layers":0,"load_mode":"dio"})");

    const si::VramEstimate e_def = si::estimate_vram(m, default_p);
    const si::VramEstimate e_nr = si::estimate_vram(m, nr_p);
    const si::VramEstimate e_dio = si::estimate_vram(m, dio_p);
    CHECK(e_def.host_repacked_bytes == 4000 * kMiB);
    CHECK(e_nr.host_repacked_bytes == 0u);
    CHECK(e_nr.host_weights_mmapped);
    CHECK_FALSE(e_dio.host_weights_mmapped);
    CHECK(e_dio.host_page_cache_bytes == 0u);
    CHECK(e_dio.host_private_bytes >= 4000 * kMiB);
}

TEST_CASE("serve_profile_catalog: surfaces per_gpu_max_mib and PrismML flags and fits against per-GPU shard") {
    TernaryGgufWriter w;
    w.kv_str("general.architecture", "llama");
    w.kv_u32("llama.block_count", 2);
    w.kv_u32("llama.embedding_length", 128);
    w.kv_u32("llama.context_length", 4096);
    w.kv_u32("llama.attention.head_count", 4);
    w.kv_u32("llama.attention.head_count_kv", 4);
    w.kv_str("prism.hadamard.transform", "fwht");
    w.kv_u32("prism.hadamard.block_size", 4096);
    // Two PQ2_0 tensors: 6,000,000 blocks * 34 B = 204,000,000 B (~194.5 MiB) each.
    w.tensor("blk.0.ffn_up.weight", {128, 6000000}, 142);
    w.tensor("blk.1.ffn_up.weight", {128, 6000000}, 142);

    const auto path = std::filesystem::temp_directory_path() /
                      ("sonder-ternary-catalog-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".gguf");
    {
        std::ofstream out(path, std::ios::binary);
        out << w.finish(2, 8);
    }

    si::LaunchProfile p = parse_single(
        R"({"name":"bonsai-2-dual","backend":"llamaserver","model":"placeholder.gguf",)"
        R"("ctx_size":4096,"parallel":1,"n_gpu_layers":-1,"fit":false,"tensor_split":[1,1]})");
    p.model = path.generic_string();

    const auto parsed = si::read_gguf_model_info(p.model);
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    const si::VramEstimate ref = si::estimate_vram(parsed.value(), p);
    REQUIRE(ref.max_gpu_mib() > 0u);
    REQUIRE(ref.max_gpu_mib() < ref.total_mib());

    // Budget between max_gpu_mib() and total_mib() fits multi-GPU shard cleanly!
    const std::uint64_t per_gpu_budget = ref.max_gpu_mib() + 10u;
    REQUIRE(per_gpu_budget < ref.total_mib());

    srv::ServerOptions opts;
    std::vector<std::string> notes;
    std::string refusal;
    const std::vector<si::LaunchProfile> profiles{p};
    const json::Array catalog =
        srv::profile_catalog(profiles, profiles[0], opts, std::optional<std::uint64_t>(per_gpu_budget), false, notes,
                             refusal);
    CHECK(refusal.empty());
    REQUIRE(catalog.size() == 1u);
    const json::Value& entry = catalog[0];
    REQUIRE(entry.find("per_gpu_max_mib") != nullptr);
    CHECK(static_cast<std::uint64_t>(entry.find("per_gpu_max_mib")->as_int()) == ref.max_gpu_mib());
    REQUIRE(entry.find("prism_hadamard") != nullptr);
    CHECK(entry.find("prism_hadamard")->as_bool());
    REQUIRE(entry.find("requires_prism_runtime") != nullptr);
    CHECK(entry.find("requires_prism_runtime")->as_bool());

    // A per-GPU budget below max_gpu_mib() is refused and reports per-GPU max.
    srv::ServerOptions tight_opts;
    std::vector<std::string> tight_notes;
    std::string tight_refusal;
    (void)srv::profile_catalog(profiles, profiles[0], tight_opts,
                               std::optional<std::uint64_t>(ref.max_gpu_mib() - 10u), false, tight_notes,
                               tight_refusal);
    CHECK(contains(tight_refusal, "does not fit"));
    CHECK(contains(tight_refusal, "per-GPU max"));

    // device="none" skips VRAM fit check (even with a 1 MiB budget) while still
    // computing the active llamaserver host-RAM estimate.
    si::LaunchProfile cpu_dev = p;
    cpu_dev.device = std::string("none");
    cpu_dev.tensor_split.clear();
    const std::vector<si::LaunchProfile> cpu_profiles{cpu_dev};
    srv::ServerOptions cpu_opts;
    std::vector<std::string> cpu_notes;
    std::string cpu_refusal;
    (void)srv::profile_catalog(cpu_profiles, cpu_profiles[0], cpu_opts, std::optional<std::uint64_t>(1u), false,
                               cpu_notes, cpu_refusal);
    CHECK(cpu_refusal.empty());
    CHECK(cpu_opts.backend.llamaserver_host_memory.estimated_resident_bytes > 0u);
    bool skipped_noted = false;
    for (const auto& note : cpu_notes) {
        skipped_noted = skipped_noted || contains(note, "no GPU offload; VRAM fit check skipped");
    }
    CHECK(skipped_noted);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}
