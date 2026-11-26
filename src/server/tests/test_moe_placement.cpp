// --n-cpu-moe / --cpu-moe / -ot / --override-tensor in the VRAM estimate: the
// flag parser, the VRAM/host split of expert tensors, and the host-RAM
// resident estimate (mmapped weights counted as page cache).
#include <doctest/doctest.h>

#include <algorithm>
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

#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/moe_routing_telemetry.hpp"
#include "sonder/inference/server.hpp"

namespace si = sonder::inference;
namespace srv = sonder::inference::server;

namespace {

constexpr std::uint64_t kMiB = 1ull << 20;

bool contains(const std::string& text, const std::string& needle) { return text.find(needle) != std::string::npos; }

// Six blocks; each holds attention (100 MiB), three routed-expert tensors
// (1000 MiB each) and one shared-expert tensor (10 MiB): 3110 MiB, of which
// 3000 MiB are routed experts. token_embd 200 MiB, output 150 MiB. Declared
// experts 2 of 8, so the default mmapped resident fraction is 0.25.
si::GgufModelInfo moe_model() {
    si::GgufModelInfo m;
    m.architecture = "llama";
    m.block_count = 6;
    m.embedding_length = 1024;
    m.vocab_size = 32000;
    m.key_length = 128;
    m.value_length = 128;
    m.kv_heads.assign(6, 2);
    m.metadata = {{"general.architecture", "llama"}, {"llama.context_length", "4096"}};
    m.routed_experts = si::RoutedExpertCounts{8, 2};
    m.block_bytes.assign(6, 0);
    const auto add = [&](const std::string& name, std::uint64_t mib, std::uint32_t block, bool expert) {
        si::GgufTensorInfo t;
        t.name = name;
        t.bytes = mib * kMiB;
        t.block = block;
        t.expert = expert;
        if (block != si::kNoBlock) m.block_bytes[block] += t.bytes;
        m.tensors.push_back(std::move(t));
    };
    for (std::uint32_t b = 0; b < 6; ++b) {
        const std::string p = "blk." + std::to_string(b) + ".";
        add(p + "attn_q.weight", 100, b, false);
        add(p + "ffn_up_exps.weight", 1000, b, true);
        add(p + "ffn_down_exps.weight", 1000, b, true);
        add(p + "ffn_gate_exps.weight", 1000, b, true);
        add(p + "ffn_up_shexp.weight", 10, b, false);
    }
    m.embedding_bytes = 200 * kMiB;
    m.output_bytes = 150 * kMiB;
    return m;
}

si::LaunchProfile profile(std::vector<std::string> extra, std::optional<std::int32_t> ngl = -1) {
    si::LaunchProfile p;
    p.name = "moe";
    p.backend = si::kLaunchProfileBackendLlamaServer;
    p.model = "m.gguf";
    p.ctx_size = 4096;
    p.parallel = 1;
    p.n_gpu_layers = ngl;
    p.extra_args = std::move(extra);
    return p;
}

constexpr std::uint64_t kHostCompute = 256 * kMiB + 32000ull * 512 * 4;  // allowance + logits at ubatch 512

}  // namespace

TEST_CASE("moe placement parser reads every spelling of the flags") {
    const auto r = si::parse_moe_placement({"--n-cpu-moe", "3", "-ncmoe=4", "--n_cpu_moe", "5", "--cpu-moe", "-cmoe",
                                            "--override-tensor", "exps=CPU", "-ot", "a=CUDA0,b=CUDA_Host,c=CPU_Mapped"});
    REQUIRE_MESSAGE(r.ok(), r.status().message());
    const auto& rules = r.value().rules;
    REQUIRE(rules.size() == 9);
    CHECK(rules[0].kind == si::TensorPlacementRule::Kind::first_n_experts);
    CHECK(rules[0].count == 3);
    CHECK(rules[1].count == 4);
    CHECK(rules[2].count == 5);
    CHECK(rules[3].kind == si::TensorPlacementRule::Kind::all_experts);
    CHECK(rules[4].kind == si::TensorPlacementRule::Kind::all_experts);
    CHECK(rules[5].kind == si::TensorPlacementRule::Kind::pattern);
    CHECK(rules[5].pattern == "exps");
    CHECK(rules[5].buffer_type == "CPU");
    CHECK(rules[5].host);
    CHECK_FALSE(rules[6].host);  // CUDA0
    CHECK(rules[7].host);        // CUDA_Host is pinned host memory
    CHECK(rules[8].host);        // CPU_Mapped
    CHECK(r.value().mmap);
    CHECK_FALSE(r.value().mlock);
}

TEST_CASE("moe placement parser splits a pattern at the first equals sign and keeps regex syntax") {
    const auto r = si::parse_moe_placement({"-ot=blk\\.(0|1)\\.ffn_.*_exps\\.=CPU"});
    REQUIRE_MESSAGE(r.ok(), r.status().message());
    REQUIRE(r.value().rules.size() == 1);
    CHECK(r.value().rules[0].pattern == "blk\\.(0|1)\\.ffn_.*_exps\\.");
    CHECK(r.value().rules[0].buffer_type == "CPU");
}

TEST_CASE("moe placement parser reads --load-mode and ignores unrelated arguments") {
    const auto r = si::parse_moe_placement({"--threads", "8", "-lm", "mlock", "-ngl", "3"});
    REQUIRE(r.ok());
    CHECK_FALSE(r.value().mmap);
    CHECK(r.value().mlock);
    CHECK_FALSE(r.value().has_tensor_rules());
    struct Case { const char* value; bool mmap; bool mlock; };
    for (const Case c : {Case{"auto", true, false}, Case{"none", false, false}, Case{"mmap", true, false},
                         Case{"mlock", false, true}, Case{"mmap+mlock", true, true}, Case{"dio", false, false}}) {
        CAPTURE(c.value);
        const auto spaced = si::parse_moe_placement({"--load-mode", c.value});
        REQUIRE(spaced.ok());
        CHECK(spaced.value().mmap == c.mmap);
        CHECK(spaced.value().mlock == c.mlock);
        const auto eq = si::parse_moe_placement({std::string("--load-mode=") + c.value});
        REQUIRE(eq.ok());
        CHECK(eq.value().mmap == c.mmap);
        CHECK(eq.value().mlock == c.mlock);
    }
    // The last flag wins.
    const auto back = si::parse_moe_placement({"-lm", "none", "-lm", "mmap"});
    REQUIRE(back.ok());
    CHECK(back.value().mmap);
    CHECK_FALSE(back.value().mlock);
    // Flags that b11195 does not define are not modelled: the default stays mmap.
    const auto fake = si::parse_moe_placement({"--no-mmap", "--mlock"});
    REQUIRE(fake.ok());
    CHECK(fake.value().mmap);
    CHECK_FALSE(fake.value().mlock);
}

TEST_CASE("moe placement parser refuses unknown --load-mode values and a missing value") {
    for (const char* bad : {"mmapp", "MMAP", "", "mmap,mlock", "direct"}) {
        CAPTURE(bad);
        const auto r = si::parse_moe_placement({"--load-mode", bad});
        CHECK_FALSE(r.ok());
    }
    CHECK_FALSE(si::parse_moe_placement({"-lm"}).ok());
    CHECK_FALSE(si::parse_moe_placement({"--load-mode=bogus"}).ok());
}

TEST_CASE("moe placement parser refuses malformed input") {
    const std::vector<std::vector<std::string>> bad = {
        {"--n-cpu-moe"},                        // missing value
        {"--n-cpu-moe", "x"},                   // not a number
        {"--n-cpu-moe", "-1"},                  // sign
        {"--n-cpu-moe", "+1"},
        {"--n-cpu-moe", "007"},                 // zero padded
        {"--n-cpu-moe", "1.5"},
        {"--n-cpu-moe", "99999999"},            // too many digits
        {"--n-cpu-moe", "4194305"},             // above the cap
        {"--n-cpu-moe", ""},
        {"--cpu-moe=1"},                        // takes no value
        {"-ot"},                                // missing value
        {"-ot", "noequals"},
        {"-ot", "=CPU"},                        // empty pattern
        {"-ot", "a="},                          // empty buffer type
        {"-ot", "a=CP U"},                      // bad buffer type characters
        {"-ot", "a=CPU;rm"},
        {"-ot", "(unclosed=CPU"},               // invalid regex
        {"-ot", "[bad=CPU"},
        {"-ot", "a=CPU,"},                      // empty item
        {"-ot", ",a=CPU"},
        {"-ot", "a=CPU,b"},                     // second item malformed
        {"-ot", std::string(257, 'a') + "=CPU"},  // pattern too long
        {"-ot", "a=" + std::string(65, 'C')},   // buffer type too long
        {"-ot", std::string(5000, 'a')},        // value too long
    };
    for (const auto& args : bad) {
        CAPTURE(args);
        const auto r = si::parse_moe_placement(args);
        CHECK_FALSE(r.ok());
        if (!r.ok()) {
            CHECK(r.status().code() == si::ErrorCode::invalid_argument);
            CHECK(r.status().message().size() < 400);  // excerpts are bounded
        }
    }
    // More rules than the cap.
    std::vector<std::string> many;
    for (int i = 0; i < 300; ++i) many.push_back("--cpu-moe");
    CHECK_FALSE(si::parse_moe_placement(many).ok());
}

TEST_CASE("estimate: --cpu-moe moves every expert from VRAM to host RAM") {
    const auto m = moe_model();
    const si::VramEstimate base = si::estimate_vram(m, profile({}));
    const si::VramEstimate cpu = si::estimate_vram(m, profile({"--cpu-moe"}));
    CHECK(base.weights_bytes == (6 * 3110 + 150) * kMiB);
    CHECK(base.host_expert_bytes == 0);
    CHECK(cpu.weights_bytes == (6 * 110 + 150) * kMiB);
    CHECK(base.weights_bytes - cpu.weights_bytes == 18000 * kMiB);
    CHECK(base.total_bytes - cpu.total_bytes == 18000 * kMiB);
    CHECK(cpu.host_expert_bytes == 18000 * kMiB);
    CHECK(cpu.host_dense_weight_bytes == 200 * kMiB);  // token_embd; the output is offloaded
    CHECK(cpu.host_kv_bytes == 0);
    CHECK(cpu.host_compute_bytes == kHostCompute);
    CHECK(cpu.moe_placement_error.empty());
    // Not a placement note: the flag is modelled.
    for (const auto& note : cpu.notes) CHECK_FALSE(contains(note, "placement flags in extra_args"));
}

TEST_CASE("estimate: mmapped experts are page cache at the active/total fraction, not private memory") {
    const auto m = moe_model();
    const si::VramEstimate e = si::estimate_vram(m, profile({"--cpu-moe"}));
    CHECK(e.host_weights_mmapped);
    CHECK(e.host_expert_resident_fraction == doctest::Approx(0.25));
    CHECK(e.host_page_cache_bytes == 200 * kMiB + 4500 * kMiB);  // dense + 18000 MiB x 2/8
    CHECK(e.host_private_bytes == kHostCompute);                 // no weights, no KV on the host
    CHECK(e.host_resident_bytes == e.host_private_bytes + e.host_page_cache_bytes);
    CHECK(e.host_resident_mib() == (e.host_resident_bytes + kMiB - 1) / kMiB);
    // A caller-stated fraction wins; above 1 clamps; NaN and 0 fall back.
    si::VramEstimateInputs half;
    half.expert_resident_fraction = 0.5;
    CHECK(si::estimate_vram(m, profile({"--cpu-moe"}), half).host_page_cache_bytes == (200 + 9000) * kMiB);
    si::VramEstimateInputs over;
    over.expert_resident_fraction = 7.0;
    CHECK(si::estimate_vram(m, profile({"--cpu-moe"}), over).host_page_cache_bytes == (200 + 18000) * kMiB);
    si::VramEstimateInputs nan;
    nan.expert_resident_fraction = std::numeric_limits<double>::quiet_NaN();
    CHECK(si::estimate_vram(m, profile({"--cpu-moe"}), nan).host_page_cache_bytes == (200 + 4500) * kMiB);
    // Without routed-expert counts every expert byte is counted.
    auto undeclared = m;
    undeclared.routed_experts.reset();
    CHECK(si::estimate_vram(undeclared, profile({"--cpu-moe"})).host_page_cache_bytes == (200 + 18000) * kMiB);
}

TEST_CASE("estimate: load modes none, mlock, mmap+mlock and dio make host weights private memory") {
    const auto m = moe_model();
    for (const char* mode : {"none", "mlock", "mmap+mlock", "dio"}) {
        CAPTURE(mode);
        const si::VramEstimate e = si::estimate_vram(m, profile({"--cpu-moe", "--load-mode", mode}));
        CHECK_FALSE(e.host_weights_mmapped);
        CHECK(e.host_page_cache_bytes == 0);
        CHECK(e.host_expert_resident_fraction == doctest::Approx(1.0));
        CHECK(e.host_private_bytes == kHostCompute + (200 + 18000) * kMiB);
        CHECK(e.host_resident_bytes == e.host_private_bytes);
    }
}

TEST_CASE("estimate: --n-cpu-moe keeps only the first N blocks' experts on the host") {
    const auto m = moe_model();
    const si::VramEstimate base = si::estimate_vram(m, profile({}));
    const si::VramEstimate two = si::estimate_vram(m, profile({"--n-cpu-moe", "2"}));
    CHECK(base.weights_bytes - two.weights_bytes == 6000 * kMiB);  // blocks 0 and 1
    CHECK(two.host_expert_bytes == 6000 * kMiB);
    // N larger than the model is every block; 0 changes nothing.
    CHECK(si::estimate_vram(m, profile({"-ncmoe", "99"})).host_expert_bytes == 18000 * kMiB);
    const si::VramEstimate zero = si::estimate_vram(m, profile({"--n-cpu-moe", "0"}));
    CHECK(zero.weights_bytes == base.weights_bytes);
    bool noted = false;
    for (const auto& note : zero.notes) noted = noted || contains(note, "matched no tensor");
    CHECK(noted);
}

TEST_CASE("estimate: -ot applies the first matching rule, per tensor") {
    const auto m = moe_model();
    const si::VramEstimate base = si::estimate_vram(m, profile({}));
    const si::VramEstimate one_block = si::estimate_vram(m, profile({"-ot", "blk\\.5\\.ffn_up_exps=CPU"}));
    CHECK(base.weights_bytes - one_block.weights_bytes == 1000 * kMiB);
    // The up experts stay on the GPU because the first rule claims them.
    const si::VramEstimate first_wins =
        si::estimate_vram(m, profile({"-ot", "ffn_up_exps=CUDA0,ffn_.*_exps=CPU"}));
    CHECK(base.weights_bytes - first_wins.weights_bytes == 12000 * kMiB);
    CHECK(first_wins.host_expert_bytes == 12000 * kMiB);
    // A pattern can move any tensor, not just experts (dense host weight).
    const si::VramEstimate attn = si::estimate_vram(m, profile({"-ot", "attn_q=CPU"}));
    CHECK(base.weights_bytes - attn.weights_bytes == 600 * kMiB);
    CHECK(attn.host_expert_bytes == 0);
    CHECK(attn.host_dense_weight_bytes == (200 + 600) * kMiB);
    // A pattern naming the GPU leaves the estimate alone.
    CHECK(si::estimate_vram(m, profile({"-ot", "exps=CUDA0"})).weights_bytes == base.weights_bytes);
}

TEST_CASE("estimate: non-offloaded blocks and their KV are host RAM") {
    const auto m = moe_model();
    const si::VramEstimate e = si::estimate_vram(m, profile({}, 4));  // blocks 2..5 on the GPU, 0 and 1 on the host
    CHECK(e.gpu_layers == 4u);
    CHECK(e.kv_bytes == 4ull * 1024 * 4096);
    CHECK(e.host_kv_bytes == 2ull * 1024 * 4096);
    // Blocks 0 and 1 (all tensors), token_embd, and the output tensor (not every block is offloaded).
    CHECK(e.host_expert_bytes == 6000 * kMiB);
    CHECK(e.host_dense_weight_bytes == (2 * 110 + 200 + 150) * kMiB);
    // -ot can put a tensor of a host block on the GPU.
    const si::VramEstimate pulled = si::estimate_vram(m, profile({"-ot", "blk\\.0\\.ffn_up_exps=CUDA0"}, 4));
    CHECK(pulled.weights_bytes - e.weights_bytes == 1000 * kMiB);
    CHECK(e.host_expert_bytes - pulled.host_expert_bytes == 1000 * kMiB);
}

TEST_CASE("estimate: nothing offloaded puts the activations on the host too") {
    const auto m = moe_model();
    const si::VramEstimate e = si::estimate_vram(m, profile({}, 0));
    CHECK(e.gpu_layers == 0u);
    CHECK(e.host_compute_bytes == kHostCompute + 4ull * 1024 * 512 * 4);
    CHECK(e.host_kv_bytes == 6ull * 1024 * 4096);
}

TEST_CASE("estimate: a malformed flag is ignored and noted, as before") {
    const auto m = moe_model();
    const si::VramEstimate base = si::estimate_vram(m, profile({}));
    for (const std::vector<std::string>& args : std::vector<std::vector<std::string>>{
             {"--override-tensor", "x"}, {"--n_cpu_moe", "x"}, {"--cpu-moe", "-ot"}}) {
        CAPTURE(args);
        const si::VramEstimate e = si::estimate_vram(m, profile(args));
        CHECK(e.weights_bytes == base.weights_bytes);
        CHECK_FALSE(e.moe_placement_error.empty());
        bool noted = false;
        for (const auto& note : e.notes) noted = noted || contains(note, "placement flags in extra_args");
        CHECK(noted);
    }
}

TEST_CASE("estimate: without tensor names the flags are noted and not applied") {
    auto m = moe_model();
    m.tensors.clear();
    const si::VramEstimate base = si::estimate_vram(m, profile({}));
    const si::VramEstimate e = si::estimate_vram(m, profile({"--cpu-moe"}));
    CHECK(e.weights_bytes == base.weights_bytes);
    bool noted = false;
    for (const auto& note : e.notes) noted = noted || contains(note, "not applied");
    CHECK(noted);
    auto truncated = moe_model();
    truncated.tensors_truncated = true;
    CHECK(si::estimate_vram(truncated, profile({"--cpu-moe"})).weights_bytes == base.weights_bytes);
}

namespace {

// Minimal GGUF v3 header: architecture, block_count, then the tensor table.
class Gguf {
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
    void tensor(const std::string& name, std::uint64_t dim0, std::uint64_t dim1, std::uint32_t type = 0) {
        str(name);
        u32(2);
        u64(dim0);
        u64(dim1);
        u32(type);  // 0 = f32
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

}  // namespace

TEST_CASE("gguf reader keeps tensor names and marks routed experts") {
    Gguf g;
    g.kv_str("general.architecture", "llama");
    g.kv_u32("llama.block_count", 2);
    g.kv_u32("llama.embedding_length", 16);
    g.tensor("token_embd.weight", 16, 4);
    g.tensor("blk.0.attn_q.weight", 16, 16);
    g.tensor("blk.0.ffn_up_exps.weight", 16, 64);
    g.tensor("blk.1.ffn_gate_up_exps.weight", 16, 64);
    g.tensor("blk.1.ffn_up_shexp.weight", 16, 8);
    g.tensor("blk.1.ffn_up_chexps.weight", 16, 8);
    g.tensor("output.weight", 16, 4);
    const auto parsed = si::parse_gguf_model_info(g.finish(7, 3));
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    const auto& info = parsed.value();
    REQUIRE(info.tensors.size() == 7);
    CHECK_FALSE(info.tensors_truncated);
    CHECK(info.tensors[0].block == si::kNoBlock);
    CHECK_FALSE(info.tensors[0].expert);
    CHECK(info.tensors[1].block == 0);
    CHECK_FALSE(info.tensors[1].expert);
    CHECK(info.tensors[2].expert);
    CHECK(info.tensors[2].bytes == 16ull * 64 * 4);
    CHECK(info.tensors[3].expert);
    CHECK_FALSE(info.tensors[4].expert);  // shared experts are dense
    CHECK(info.tensors[5].expert);        // grove-style chexps
    CHECK(info.tensors[3].block == 1);
    // The block totals stay complete.
    CHECK(info.block_bytes[0] == (16ull * 16 + 16 * 64) * 4);
}

TEST_CASE("gguf reader truncates the stored table for an over-long tensor name but keeps totals") {
    Gguf g;
    g.kv_str("general.architecture", "llama");
    g.kv_u32("llama.block_count", 1);
    g.kv_u32("llama.embedding_length", 16);
    g.tensor("blk.0.attn_q.weight", 16, 16);
    g.tensor("extra." + std::string(300, 'x'), 16, 16);
    const auto parsed = si::parse_gguf_model_info(g.finish(2, 3));
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    CHECK(parsed.value().tensors_truncated);
    CHECK(parsed.value().block_bytes[0] == 16ull * 16 * 4);
    CHECK(parsed.value().other_bytes == 16ull * 16 * 4);
}

namespace {

// moe_model() with its routed experts stored as `type` (a Q4_K model: 12).
si::GgufModelInfo moe_model_of_type(std::uint32_t type) {
    auto m = moe_model();
    for (auto& t : m.tensors) {
        if (t.expert) t.type = type;
    }
    return m;
}

}  // namespace

TEST_CASE("gguf reader keeps the ggml type of each tensor") {
    Gguf g;
    g.kv_str("general.architecture", "llama");
    g.kv_u32("llama.block_count", 1);
    g.kv_u32("llama.embedding_length", 16);
    g.tensor("blk.0.ffn_up_exps.weight", 256, 4, 12);  // Q4_K: 144 bytes per 256 elements
    g.tensor("blk.0.attn_q.weight", 16, 16);
    const auto parsed = si::parse_gguf_model_info(g.finish(2, 3));
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    REQUIRE(parsed.value().tensors.size() == 2);
    CHECK(parsed.value().tensors[0].type == 12);
    CHECK(parsed.value().tensors[0].bytes == 4ull * 144);
    CHECK(parsed.value().tensors[1].type == 0);
}

TEST_CASE("moe placement parser reads -nr and --no-repack") {
    CHECK(si::parse_moe_placement({}).value().repack);
    CHECK(si::parse_moe_placement({"--cpu-moe"}).value().repack);
    CHECK_FALSE(si::parse_moe_placement({"-nr"}).value().repack);
    CHECK_FALSE(si::parse_moe_placement({"--cpu-moe", "--no-repack"}).value().repack);
    CHECK_FALSE(si::parse_moe_placement({"--no-repack=1"}).ok());
}

TEST_CASE("estimate: host experts of a repackable type are private memory at 100%, not page cache at active/total") {
    // b11195 copies CPU-placed Q4_0/Q8_0/Q4_K/IQ4_NL weights (including the
    // MUL_MAT_ID expert matmul) into a repack buffer unless -nr is given. The
    // old active/total fraction (2 of 8) would have admitted this at 4500 MiB.
    for (const std::uint32_t type : {2u, 8u, 12u, 20u}) {
        CAPTURE(type);
        const auto m = moe_model_of_type(type);
        const si::VramEstimate e = si::estimate_vram(m, profile({"--cpu-moe"}));
        CHECK(e.host_expert_bytes == 18000 * kMiB);
        CHECK(e.host_repacked_bytes == 18000 * kMiB);
        CHECK(e.host_page_cache_bytes == 200 * kMiB);  // token_embd (F32 here) stays mmapped
        CHECK(e.host_private_bytes == kHostCompute + 18000 * kMiB);
        CHECK(e.host_resident_bytes == kHostCompute + (200 + 18000) * kMiB);
        CHECK(e.host_resident_bytes > kHostCompute + (200 + 4500) * kMiB);
    }
    // -nr / --no-repack restores the mmapped fraction.
    for (const char* flag : {"-nr", "--no-repack"}) {
        CAPTURE(flag);
        const si::VramEstimate e = si::estimate_vram(moe_model_of_type(12), profile({"--cpu-moe", flag}));
        CHECK(e.host_repacked_bytes == 0);
        CHECK(e.host_page_cache_bytes == (200 + 4500) * kMiB);
        CHECK(e.host_private_bytes == kHostCompute);
    }
    // A type with no repack path (Q5_K = 13) is unchanged.
    const si::VramEstimate q5 = si::estimate_vram(moe_model_of_type(13), profile({"--cpu-moe"}));
    CHECK(q5.host_repacked_bytes == 0);
    CHECK(q5.host_page_cache_bytes == (200 + 4500) * kMiB);
    // Experts that stay on the GPU are not repacked.
    CHECK(si::estimate_vram(moe_model_of_type(12), profile({})).host_repacked_bytes == 0);
    // Without mmap everything is private already: no double count.
    const si::VramEstimate none =
        si::estimate_vram(moe_model_of_type(12), profile({"--cpu-moe", "--load-mode", "none"}));
    CHECK(none.host_private_bytes == kHostCompute + (200 + 18000) * kMiB);
    CHECK(none.host_resident_bytes == none.host_private_bytes);
    // A caller-stated fraction cannot lower repacked bytes.
    si::VramEstimateInputs tiny;
    tiny.expert_resident_fraction = 0.01;
    CHECK(si::estimate_vram(moe_model_of_type(12), profile({"--cpu-moe"}), tiny).host_resident_bytes ==
          kHostCompute + (200 + 18000) * kMiB);
}

TEST_CASE("moe placement parser reads --repack (last of --repack, -nr, --no-repack wins) and -ncffn") {
    CHECK(si::parse_moe_placement({"-nr", "--repack"}).value().repack);
    CHECK_FALSE(si::parse_moe_placement({"--repack", "-nr"}).value().repack);
    CHECK_FALSE(si::parse_moe_placement({"--repack=1"}).ok());
    CHECK(si::parse_moe_placement({}).value().n_cpu_ffn == 0);
    const auto ffn = si::parse_moe_placement({"-ncffn", "3"});
    REQUIRE(ffn.ok());
    CHECK(ffn.value().n_cpu_ffn == 3);
    CHECK_FALSE(ffn.value().has_tensor_rules());
    CHECK(si::parse_moe_placement({"--n-cpu-ffn=2"}).value().n_cpu_ffn == 2);
    CHECK_FALSE(si::parse_moe_placement({"--n-cpu-ffn", "03"}).ok());
    CHECK_FALSE(si::parse_moe_placement({"--n-cpu-ffn"}).ok());
}

TEST_CASE("estimate: fit on and no n_gpu_layers counts what exceeds the VRAM budget on the host") {
    const auto m = moe_model();
    const si::VramEstimate ref = si::estimate_vram(m, profile({}, -1));
    const auto fit_with = [&](std::uint64_t budget, std::vector<std::string> extra = {}) {
        si::VramEstimateInputs in;
        in.fit_vram_budget_bytes = budget;
        return si::estimate_vram(m, profile(std::move(extra), std::nullopt), in);
    };
    // A budget far above the need: the fit path applies but nothing overflows.
    {
        const si::VramEstimate e = fit_with(1ull << 40);
        CHECK(e.host_fit_applies);
        CHECK(e.host_fit_overflow_bytes == 0);
        CHECK(e.host_resident_bytes == ref.host_resident_bytes);
        CHECK_FALSE(e.host_estimate_lower_bound);
    }
    constexpr std::uint64_t kGiB = 1ull << 30;
    {
        const si::VramEstimate e = fit_with(8 * kGiB);
        const std::uint64_t movable = e.weights_bytes + e.kv_bytes + e.recurrent_bytes;
        const std::uint64_t need = e.total_bytes + kGiB;
        REQUIRE(need > 8 * kGiB);
        const std::uint64_t overflow = std::min(movable, need - 8 * kGiB + 3110 * kMiB);
        CHECK(e.host_fit_applies);
        CHECK(e.host_fit_overflow_bytes == overflow);
        CHECK(e.host_fit_overflow_bytes > 0);
        const std::uint64_t overflow_expert = std::min<std::uint64_t>(overflow, 18000 * kMiB);
        const std::uint64_t overflow_dense = overflow - overflow_expert;
        CHECK(e.host_expert_bytes == overflow_expert);
        CHECK(e.host_resident_bytes > ref.host_resident_bytes);
        const auto quarter = static_cast<std::uint64_t>(std::ceil(0.25 * static_cast<double>(overflow_expert)));
        CHECK(e.host_page_cache_bytes - ref.host_page_cache_bytes == quarter + overflow_dense);
        // The VRAM side is unchanged.
        CHECK(e.weights_bytes == ref.weights_bytes);
        CHECK(e.kv_bytes == ref.kv_bytes);
        CHECK(e.recurrent_bytes == ref.recurrent_bytes);
        CHECK(e.compute_bytes == ref.compute_bytes);
        CHECK(e.total_bytes == ref.total_bytes);
        CHECK(e.gpu_layers == ref.gpu_layers);
        bool noted = false;
        for (const auto& n : e.notes) noted = noted || contains(n, "--fit sizes the offload");
        CHECK(noted);
    }
    {
        const si::VramEstimate e = fit_with(0);
        CHECK(e.host_fit_overflow_bytes == e.weights_bytes + e.kv_bytes + e.recurrent_bytes);
        CHECK(e.host_resident_bytes > ref.host_resident_bytes);
    }
    // Repackable experts that overflow are private memory at 100%; -nr keeps them mapped.
    {
        const auto q4 = moe_model_of_type(12);
        const si::VramEstimate q4_ref = si::estimate_vram(q4, profile({}, -1));
        si::VramEstimateInputs in;
        in.fit_vram_budget_bytes = 0;
        const si::VramEstimate e = si::estimate_vram(q4, profile({}, std::nullopt), in);
        CHECK(e.host_expert_bytes == 18000 * kMiB);
        CHECK(e.host_repacked_bytes >= 18000 * kMiB);
        CHECK(e.host_private_bytes - q4_ref.host_private_bytes == e.host_repacked_bytes);
        const si::VramEstimate nr = si::estimate_vram(q4, profile({"-nr"}, std::nullopt), in);
        CHECK(nr.host_repacked_bytes == 0);
        CHECK(nr.host_private_bytes == q4_ref.host_private_bytes);
        CHECK(nr.host_page_cache_bytes > q4_ref.host_page_cache_bytes);
    }
}

TEST_CASE("estimate: fit path without a budget is flagged as a lower bound") {
    const auto m = moe_model();
    const si::VramEstimate ref = si::estimate_vram(m, profile({}, -1));
    const si::VramEstimate e = si::estimate_vram(m, profile({}, std::nullopt));
    CHECK(e.host_fit_applies);
    CHECK(e.host_estimate_lower_bound);
    CHECK(e.host_fit_overflow_bytes == 0);
    CHECK(e.host_resident_bytes == ref.host_resident_bytes);
    // The flag carries the caveat; VramEstimate::notes stay VRAM-only (the
    // dense "no sliding-window layers" test asserts they are empty).
    for (const auto& n : e.notes) CHECK_FALSE(contains(n, "lower bound"));
    CHECK_FALSE(ref.host_estimate_lower_bound);
}

TEST_CASE("estimate: fit path does not apply when upstream fit is off or aborts") {
    const auto m = moe_model();
    const auto applies = [&](const si::LaunchProfile& p) { return si::estimate_vram(m, p).host_fit_applies; };
    CHECK(applies(profile({}, std::nullopt)));
    CHECK(applies(profile({"--n-cpu-moe", "0"}, std::nullopt)));  // adds no override
    si::LaunchProfile off = profile({}, std::nullopt);
    off.fit = false;
    CHECK_FALSE(applies(off));
    CHECK_FALSE(applies(profile({"--cpu-moe"}, std::nullopt)));
    CHECK_FALSE(applies(profile({"--n-cpu-moe", "2"}, std::nullopt)));
    CHECK_FALSE(applies(profile({"-ot", "x=CPU"}, std::nullopt)));
    CHECK_FALSE(applies(profile({"--n-cpu-ffn", "1"}, std::nullopt)));
    si::LaunchProfile direct = profile({}, std::nullopt);
    direct.backend = si::kLaunchProfileBackendLlamaCpp;
    CHECK_FALSE(applies(direct));
    CHECK_FALSE(applies(profile({}, -1)));
    CHECK_FALSE(applies(profile({}, 0)));
}


namespace {

// Writes a small GGUF to the temp directory; removed by the destructor.
class TempGguf {
public:
    TempGguf() {
        Gguf g;
        g.kv_str("general.architecture", "llama");
        g.kv_u32("llama.block_count", 2);
        g.kv_u32("llama.embedding_length", 16);
        g.tensor("token_embd.weight", 16, 4);
        g.tensor("blk.0.attn_q.weight", 16, 16);
        g.tensor("blk.1.attn_q.weight", 16, 16);
        path_ = std::filesystem::temp_directory_path() /
                ("sonder-hostwiring-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                 ".gguf");
        std::ofstream(path_, std::ios::binary) << g.finish(3, 3);
    }
    ~TempGguf() {
        std::error_code ec;
        std::filesystem::remove(path_, ec);
    }
    TempGguf(const TempGguf&) = delete;
    TempGguf& operator=(const TempGguf&) = delete;
    [[nodiscard]] std::string path() const { return path_.generic_string(); }

private:
    std::filesystem::path path_;
};

}  // namespace

TEST_CASE("serve wiring: the active llamaserver profile's host estimate reaches BackendSetup, offloaded or not") {
    TempGguf gguf;
    // `ngl` is the target profile's n_gpu_layers; `other_ngl` the second profile's (default: the same).
    const auto run = [&](std::optional<std::int32_t> ngl, bool target_is_active, srv::ServerOptions& o,
                         std::string& refusal, std::optional<std::int32_t> other_ngl = std::nullopt) {
        si::LaunchProfile p = profile({}, ngl);
        p.model = gguf.path();
        si::LaunchProfile other = p;
        other.name = "other";
        if (other_ngl.has_value()) other.n_gpu_layers = other_ngl;
        const std::vector<si::LaunchProfile> profiles{p, other};
        std::vector<std::string> notes;
        return srv::profile_catalog(profiles, target_is_active ? profiles[0] : profiles[1], o,
                                    std::optional<std::uint64_t>(1u << 20), false, notes, refusal);
    };
    const auto parsed = si::read_gguf_model_info(gguf.path());
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    si::LaunchProfile reference = profile({}, 0);
    reference.model = gguf.path();
    const si::VramEstimate cpu_only = si::estimate_vram(parsed.value(), reference);
    reference.n_gpu_layers = -1;
    const si::VramEstimate offloaded = si::estimate_vram(parsed.value(), reference);
    REQUIRE(cpu_only.host_resident_bytes >= 256 * kMiB);

    // n_gpu_layers == 0: no VRAM fit check, but the host estimate is passed on.
    srv::ServerOptions cpu;
    std::string refusal;
    const auto catalog = run(0, true, cpu, refusal);
    CHECK(refusal.empty());
    CHECK(catalog.size() == 2);
    CHECK(cpu.backend.llamaserver_host_memory.estimated_resident_bytes == cpu_only.host_resident_bytes);
    CHECK(cpu.backend.llamaserver_host_memory.estimated_private_bytes == cpu_only.host_private_bytes);
    CHECK(cpu.backend.llamaserver_host_memory.estimated_resident_bytes > 0);

    // Offloading profile: same wiring, offloaded estimate.
    srv::ServerOptions gpu;
    refusal.clear();
    run(-1, true, gpu, refusal);
    CHECK(refusal.empty());
    CHECK(gpu.backend.llamaserver_host_memory.estimated_resident_bytes == offloaded.host_resident_bytes);
    CHECK(gpu.backend.llamaserver_host_memory.estimated_private_bytes == offloaded.host_private_bytes);

    // Only the active profile feeds the guard: the inactive one offloads and
    // reads fine, the active one is CPU-only; the CPU-only numbers must win.
    srv::ServerOptions inactive;
    refusal.clear();
    run(-1, false, inactive, refusal, 0);
    CHECK(refusal.empty());
    CHECK(inactive.backend.llamaserver_host_memory.estimated_resident_bytes == cpu_only.host_resident_bytes);
    CHECK(inactive.backend.llamaserver_host_memory.estimated_resident_bytes != offloaded.host_resident_bytes);

    // An unreadable model on a CPU-only profile is a note, not a refusal.
    si::LaunchProfile missing = profile({}, 0);
    missing.model = gguf.path() + ".missing";
    srv::ServerOptions none;
    std::vector<std::string> notes;
    refusal.clear();
    const std::vector<si::LaunchProfile> only{missing};
    srv::profile_catalog(only, only[0], none, std::nullopt, false, notes, refusal);
    CHECK(refusal.empty());
    CHECK(none.backend.llamaserver_host_memory.estimated_resident_bytes == 0);
}

TEST_CASE("serve wiring: a fit-path profile feeds the VRAM budget into the host estimate") {
    TempGguf gguf;
    si::LaunchProfile p = profile({}, std::nullopt);
    p.model = gguf.path();
    const auto parsed = si::read_gguf_model_info(gguf.path());
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    si::LaunchProfile all = p;
    all.n_gpu_layers = -1;
    const si::VramEstimate ref = si::estimate_vram(parsed.value(), all);
    si::VramEstimateInputs in;
    in.fit_vram_budget_bytes = 1 * kMiB;
    const si::VramEstimate expected = si::estimate_vram(parsed.value(), p, in);
    REQUIRE(expected.host_fit_applies);
    REQUIRE(expected.host_fit_overflow_bytes > 0);

    const std::vector<si::LaunchProfile> profiles{p};
    {
        srv::ServerOptions o;
        std::vector<std::string> notes;
        std::string refusal;
        srv::profile_catalog(profiles, profiles[0], o, std::optional<std::uint64_t>(1), false, notes, refusal);
        CHECK(refusal.empty());
        const auto& h = o.backend.llamaserver_host_memory;
        CHECK(h.estimated_resident_bytes == expected.host_resident_bytes);
        CHECK(h.estimated_resident_bytes > ref.host_resident_bytes);
        CHECK_FALSE(h.estimate_lower_bound);
    }
    if (!si::detect_device_vram_mib()) {
        srv::ServerOptions o;
        std::vector<std::string> notes;
        std::string refusal;
        srv::profile_catalog(profiles, profiles[0], o, std::nullopt, false, notes, refusal);
        CHECK(o.backend.llamaserver_host_memory.estimate_lower_bound);
    } else {
        MESSAGE("a GPU is detected here: the no-budget lower-bound wiring runs on CI without one");
    }
}

TEST_CASE("moe placement parser reads and validates --moe-cache-mib and --moe-cache-cpu-share") {
    const auto parsed = si::parse_moe_placement(
        {"--n-cpu-moe", "6", "--moe-cache-mib", "4096", "--moe-cache-cpu-share", "0.35"});
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    CHECK(parsed->moe_cache_mib == 4096u);
    CHECK(parsed->moe_cache_cpu_share == doctest::Approx(0.35));

    const auto eq_parsed =
        si::parse_moe_placement({"--moe-cache-mib=0", "--moe-cache-cpu-share=1.0", "--moe_cache_mib=2048"});
    REQUIRE_MESSAGE(eq_parsed.ok(), eq_parsed.status().message());
    CHECK(eq_parsed->moe_cache_mib == 2048u);
    CHECK(eq_parsed->moe_cache_cpu_share == doctest::Approx(1.0));

    const auto zero_share = si::parse_moe_placement({"--moe-cache-cpu-share", "0.0"});
    REQUIRE(zero_share.ok());
    CHECK(zero_share->moe_cache_cpu_share == doctest::Approx(0.0));

    for (const std::vector<std::string>& bad : std::vector<std::vector<std::string>>{
             {"--moe-cache-mib"},
             {"--moe-cache-mib", "-1"},
             {"--moe-cache-mib", "0512"},
             {"--moe-cache-mib", "262145"},
             {"--moe-cache-mib", "abc"},
             {"--moe-cache-cpu-share"},
             {"--moe-cache-cpu-share", "-0.1"},
             {"--moe-cache-cpu-share", "1.01"},
             {"--moe-cache-cpu-share", "nan"},
             {"--moe-cache-cpu-share", "inf"},
             {"--moe-cache-cpu-share", "1e-1"},
         }) {
        CAPTURE(bad);
        const auto res = si::parse_moe_placement(bad);
        CHECK_FALSE(res.ok());
        CHECK(res.status().code() == si::ErrorCode::invalid_argument);
    }
}

TEST_CASE("estimate: moe_cache_mib adds VRAM for CPU-offloaded experts and warns when experts are already in VRAM") {
    const auto m = moe_model();

    // With --cpu-moe and typed moe_cache_mib=2048, VRAM increases by exactly 2048 MiB.
    si::LaunchProfile base_p = profile({"--cpu-moe"}, -1);
    const si::VramEstimate base = si::estimate_vram(m, base_p);
    REQUIRE(base.host_expert_bytes > 0u);
    CHECK(base.moe_cache_bytes == 0u);

    si::LaunchProfile cached_p = base_p;
    cached_p.moe_cache_mib = 2048u;
    cached_p.moe_cache_cpu_share = 0.35;
    const si::VramEstimate cached = si::estimate_vram(m, cached_p);
    CHECK(cached.moe_cache_bytes == 2048ull * kMiB);
    CHECK(cached.total_bytes == base.total_bytes + 2048ull * kMiB);

    // Multi-GPU split includes moe_cache_bytes in the splittable shard calculation.
    si::LaunchProfile split_p = cached_p;
    split_p.split_mode = "layer";
    split_p.tensor_split = {0.5, 0.5};
    const si::VramEstimate split_base = [&] {
        si::LaunchProfile p = base_p;
        p.split_mode = "layer";
        p.tensor_split = {0.5, 0.5};
        return si::estimate_vram(m, p);
    }();
    const si::VramEstimate split_cached = si::estimate_vram(m, split_p);
    CHECK(split_cached.moe_cache_bytes == 2048ull * kMiB);
    CHECK(split_cached.max_gpu_bytes == split_base.max_gpu_bytes + 1024ull * kMiB);

    // Anti-regression diagnostic: setting moe_cache_mib when 0 experts are on host adds a note.
    si::LaunchProfile full_vram_p = profile({}, -1);
    full_vram_p.moe_cache_mib = 1024u;
    const si::VramEstimate wasted = si::estimate_vram(m, full_vram_p);
    CHECK(wasted.host_expert_bytes == 0u);
    CHECK(wasted.moe_cache_bytes == 1024ull * kMiB);
    bool noted_overhead = false;
    for (const auto& n : wasted.notes) {
        if (contains(n, "graph-splitting overhead")) noted_overhead = true;
    }
    CHECK(noted_overhead);
}

TEST_CASE("gguf reader detects and sizes MXFP4 (4.25 bpw), NVFP4 (4.50 bpw), and OCP FP8 (1 B/elem) tensors") {
    Gguf g;
    g.kv_str("general.architecture", "llama");
    g.kv_u32("llama.block_count", 1);
    g.kv_u32("llama.embedding_length", 128);
    g.tensor("blk.0.ffn_gate.weight", 64, 4, 39);  // MXFP4: 256 elems / 32 * 17 = 136 B (4.25 bpw)
    g.tensor("blk.0.ffn_up.weight", 64, 4, 40);    // NVFP4: 256 elems / 64 * 36 = 144 B (4.50 bpw)
    g.tensor("blk.0.ffn_down.weight", 64, 4, 43);  // F8_E4M3: 256 elems * 1 = 256 B (8.00 bpw)

    const auto parsed = si::parse_gguf_model_info(g.finish(3, 3));
    REQUIRE_MESSAGE(parsed.ok(), parsed.status().message());
    const auto& info = parsed.value();
    CHECK(info.has_mxfp4_tensors);
    CHECK(info.has_nvfp4_tensors);
    CHECK(info.has_fp8_tensors);
    CHECK(info.has_fp4_or_fp8_tensors);
    REQUIRE(info.tensors.size() == 3u);
    CHECK(info.tensors[0].bytes == 136u);
    CHECK(info.tensors[1].bytes == 144u);
    CHECK(info.tensors[2].bytes == 256u);
    CHECK(info.block_bytes[0] == 136u + 144u + 256u);
}

TEST_CASE("moe routing telemetry: inactive when all experts reside in VRAM, active when offloaded") {
    const auto m = moe_model();
    const si::VramEstimate full_vram = si::estimate_vram(m, profile({}, -1));
    const si::VramEstimate cpu_offload = si::estimate_vram(m, profile({"--cpu-moe"}, -1));

    const auto inactive_cfg = si::make_moe_routing_telemetry_config(full_vram);
    CHECK_FALSE(inactive_cfg.offload_active);
    si::MoeRoutingTelemetry inactive(inactive_cfg);
    CHECK_FALSE(inactive.offload_active());
    inactive.record_expert_access(0, 3, false, 64 * kMiB, 1.5);
    inactive.record_routing_batch(0, {si::MoeExpertObservation{1, true, 0, 0.0}});
    const auto inactive_snap = inactive.snapshot();
    CHECK_FALSE(inactive_snap.offload_active);
    CHECK(inactive_snap.total_routes == 0u);
    CHECK(inactive_snap.tracked_layers == 0u);
    CHECK(inactive_snap.layers.empty());

    const auto active_cfg = si::make_moe_routing_telemetry_config(cpu_offload, /*top_k=*/3);
    CHECK(active_cfg.offload_active);
    si::MoeRoutingTelemetry active(active_cfg);
    CHECK(active.offload_active());

    // Record hits, misses, transfer bytes, and waits (including negative/NaN wait sanitization).
    active.record_expert_access(2, 5, true, 0, 0.0);
    active.record_expert_access(2, 5, true, 0, 0.0);
    active.record_expert_access(2, 5, false, 16 * kMiB, 2.5);
    active.record_expert_access(2, 1, false, 16 * kMiB, 4.0);
    active.record_expert_access(2, 7, false, 16 * kMiB, std::numeric_limits<double>::quiet_NaN());
    active.record_expert_access(2, 7, true, 0, -5.0);
    active.record_routing_batch(0, {
        si::MoeExpertObservation{2, true, 0, 0.0},
        si::MoeExpertObservation{2, false, 8 * kMiB, 1.25},
        si::MoeExpertObservation{4, true, 0, 0.0},
    });

    const auto snap = active.snapshot();
    CHECK(snap.offload_active);
    CHECK(snap.total_routes == 9u);
    CHECK(snap.expert_hits == 5u);
    CHECK(snap.expert_misses == 4u);
    CHECK(snap.hit_rate() == doctest::Approx(5.0 / 9.0));
    CHECK(snap.transfer_count == 4u);
    CHECK(snap.transfer_bytes == 56 * kMiB);
    CHECK(snap.transfer_wait_ms == doctest::Approx(7.75));
    CHECK(snap.max_transfer_wait_ms == doctest::Approx(4.0));
    REQUIRE(snap.layers.size() == 2u);
    CHECK(snap.layers[0].layer_id == 0u);
    CHECK(snap.layers[1].layer_id == 2u);

    const auto hot_l2 = active.hot_experts_for_layer(2, 2);
    REQUIRE(hot_l2.size() == 2u);
    CHECK(hot_l2[0].expert_id == 5u);
    CHECK(hot_l2[0].activations == 3u);
    CHECK(hot_l2[0].hits == 2u);
    CHECK(hot_l2[0].misses == 1u);
    CHECK(hot_l2[0].overestimate_error == 0u);
    CHECK(hot_l2[0].share == doctest::Approx(0.5));
    CHECK(hot_l2[1].expert_id == 7u);
    CHECK(hot_l2[1].activations == 2u);

    const auto doc = si::moe_routing_telemetry_to_json(snap);
    CHECK(doc.find("schema")->as_string() == "sonder.inference.moe_routing_telemetry/1");
    CHECK(doc.find("expert_hits")->as_uint() == 5u);
    CHECK(doc.find("expert_misses")->as_uint() == 4u);
    CHECK(doc.find("transfer_bytes")->as_uint() == 56 * kMiB);
    CHECK(doc.find("layers")->as_array().size() == 2u);

    active.reset();
    CHECK(active.snapshot().total_routes == 0u);
    CHECK(active.snapshot().tracked_layers == 0u);
}

TEST_CASE("moe routing telemetry: layer and per-layer expert tables stay strictly bounded while retaining hot experts") {
    si::MoeRoutingTelemetryConfig cfg;
    cfg.offload_active = true;
    cfg.max_layers = 4;
    cfg.max_experts_per_layer = 8;
    cfg.top_k = 3;
    si::MoeRoutingTelemetry tracker(cfg);

    // Hammer 20 distinct layers into a 4-layer cap.
    for (std::uint32_t layer = 0; layer < 20; ++layer) {
        tracker.record_expert_access(layer, 0, true);
    }
    {
        const auto snap = tracker.snapshot();
        CHECK(snap.tracked_layers == 4u);
        CHECK(snap.dropped_layer_observations == 16u);
        CHECK(snap.total_routes == 20u);
    }

    // Hammer 200 distinct experts into layer 0's 8-slot Space-Saving table,
    // with experts {7, 42, 99} genuinely hot (50 accesses each).
    for (int round = 0; round < 50; ++round) {
        tracker.record_expert_access(0, 7, true);
        tracker.record_expert_access(0, 42, true);
        tracker.record_expert_access(0, 99, false, 4 * kMiB, 0.5);
    }
    for (std::uint32_t cold = 100; cold < 297; ++cold) {
        tracker.record_expert_access(0, cold, false, 1 * kMiB, 0.1);
    }

    const auto snap = tracker.snapshot();
    REQUIRE(snap.layers.size() == 4u);
    CHECK(snap.layers[0].layer_id == 0u);
    CHECK(snap.layers[0].tracked_experts == 8u);  // strictly bounded at max_experts_per_layer
    CHECK(snap.layers[0].table_evictions > 0u);
    REQUIRE(snap.layers[0].top_experts.size() == 3u);

    // All three genuinely hot experts stay in the top-3 despite 197 cold evictors!
    std::vector<std::uint32_t> top_ids;
    for (const auto& e : snap.layers[0].top_experts) {
        top_ids.push_back(e.expert_id);
        CHECK(e.activations == 50u);
        CHECK(e.overestimate_error == 0u);
    }
    std::sort(top_ids.begin(), top_ids.end());
    CHECK(top_ids == std::vector<std::uint32_t>{7u, 42u, 99u});
}
