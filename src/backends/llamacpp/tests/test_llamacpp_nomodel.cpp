// Backend tests that need llama.cpp linked but no model weights. Tokenization
// uses llama.cpp's bundled vocab-only GGUF fixtures (tokenizer metadata only).
#include <cstdint>
#include <limits>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"
#include <doctest/doctest.h>

namespace sonder::inference::llamacpp_detail {
Result<std::int32_t> ResolveGpuLayers(const LlamaCppBackendOptions& options, std::string_view device_id,
                                      bool has_gpu_backend);
}  // namespace sonder::inference::llamacpp_detail

using namespace sonder::backends::llamacpp;

#ifndef SONDER_LLAMACPP_VOCAB_DIR
#error "SONDER_LLAMACPP_VOCAB_DIR must be defined by the build"
#endif

TEST_CASE("llamacpp wrapper: pinned version and CPU device") {
    CHECK(LlamaCppBackend::UpstreamVersion() == "b11195");
    const auto devices = LlamaCppBackend::EnumerateDevices();
    bool has_cpu = false;
    for (const auto& d : devices) has_cpu = has_cpu || d.kind == DeviceKind::kCpu;
    CHECK(has_cpu);
    CHECK(!LlamaCppBackend::SystemInfo().empty());
}

TEST_CASE("llamacpp wrapper: operations before load fail cleanly") {
    LlamaCppBackend be;
    CHECK(!be.IsLoaded());
    std::vector<Token> toks;
    CHECK(be.Tokenize("hi", true, toks).code == ErrorCode::kNotLoaded);
    std::string piece;
    CHECK(be.TokenToPiece(1, piece).code == ErrorCode::kNotLoaded);
    GenerateRequest req;
    req.prompt = {1, 2, 3};
    const GenerateResult r = be.Generate(req, {});
    CHECK(r.status.code == ErrorCode::kNotLoaded);
    CHECK(r.stop_reason == StopReason::kError);
    be.Unload();  // no-op, must not crash
    CHECK(be.GetModelInfo().n_vocab == 0);
}

TEST_CASE("llamacpp wrapper: load rejects bad paths") {
    LlamaCppBackend be;
    LoadOptions opts;
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    opts.model_path = "this/file/does/not/exist.gguf";
    CHECK(be.Load(opts).code == ErrorCode::kFileNotFound);
    opts.model_path = SONDER_LLAMACPP_VOCAB_DIR;  // a directory, not a file
    CHECK(be.Load(opts).code == ErrorCode::kFileNotFound);
    CHECK(!be.IsLoaded());
}

TEST_CASE("llamacpp wrapper: vocab-only GGUF tokenizes and detokenizes") {
    LlamaCppBackend be;
    int loaded = 0;
    int unloaded = 0;
    be.SetTelemetrySink([&](const TelemetryEvent& e) {
        if (e.kind == TelemetryKind::kModelLoaded) ++loaded;
        if (e.kind == TelemetryKind::kModelUnloaded) ++unloaded;
    });
    LoadOptions opts;
    opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    opts.vocab_only = true;
    const Status s = be.Load(opts);
    REQUIRE(s.ok());
    CHECK(be.IsLoaded());
    CHECK(loaded == 1);
    const ModelInfo info = be.GetModelInfo();
    CHECK(info.n_vocab == 32000);
    CHECK(info.n_ctx == 0u);

    std::vector<Token> with_bos;
    std::vector<Token> no_bos;
    REQUIRE(be.Tokenize("Hello world", true, with_bos).ok());
    REQUIRE(be.Tokenize("Hello world", false, no_bos).ok());
    CHECK(!no_bos.empty());
    CHECK(with_bos.size() == no_bos.size() + 1);

    std::string text;
    for (Token t : no_bos) {
        std::string piece;
        REQUIRE(be.TokenToPiece(t, piece).ok());
        text += piece;
    }
    // SentencePiece adds a leading space to the first word.
    CHECK((text == " Hello world" || text == "Hello world"));

    std::string piece;
    CHECK(be.TokenToPiece(-1, piece).code == ErrorCode::kInvalidArgument);
    CHECK(be.TokenToPiece(info.n_vocab, piece).code == ErrorCode::kInvalidArgument);

    // Multi-byte text round-trips through the UTF-8 stream buffer.
    std::vector<Token> uni;
    REQUIRE(be.Tokenize("caf\xC3\xA9 \xE2\x82\xAC", false, uni).ok());
    Utf8StreamBuffer buf;
    std::string streamed;
    for (Token t : uni) {
        REQUIRE(be.TokenToPiece(t, piece).ok());
        streamed += buf.Push(piece);
    }
    streamed += buf.Flush();
    CHECK(streamed.find("caf\xC3\xA9") != std::string::npos);
    CHECK(streamed.find("\xE2\x82\xAC") != std::string::npos);

    GenerateRequest req;
    req.prompt = no_bos;
    CHECK(be.Generate(req, {}).status.code == ErrorCode::kNotLoaded);  // no context

    be.Unload();
    CHECK(!be.IsLoaded());
    CHECK(unloaded == 1);
}

TEST_CASE("llamacpp wrapper: tensor overrides are validated before any load") {
    CHECK(LlamaCppBackend::ResolveTensorOverrides({}).ok());
    std::vector<void*> buffers;
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{kMoeExpertTensorPattern, "cpu"}}, &buffers).ok());
    CHECK(buffers.size() == 1);
    CHECK(buffers.front() != nullptr);
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"x", "CPU"}}).ok());  // device name is case-insensitive for cpu
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"", "cpu"}}).code == ErrorCode::kInvalidArgument);
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"(unclosed", "cpu"}}).code == ErrorCode::kInvalidArgument);
    CHECK(LlamaCppBackend::ResolveTensorOverrides({{"x", "NoSuchDevice9"}}).code == ErrorCode::kInvalidArgument);

    // Load reports a bad override even when the model path does not exist.
    LlamaCppBackend be;
    LoadOptions opts;
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.tensor_overrides = {{"x", "NoSuchDevice9"}};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
}

TEST_CASE("llamacpp wrapper: the MoE expert pattern matches expert tensors only") {
    const std::regex re(kMoeExpertTensorPattern);
    CHECK(std::regex_search("blk.3.ffn_up_exps.weight", re));
    CHECK(std::regex_search("blk.12.ffn_down_exps.weight", re));
    CHECK(std::regex_search("blk.0.ffn_gate_exps.weight", re));
    CHECK(std::regex_search("blk.7.ffn_up_chexps.weight", re));
    CHECK_FALSE(std::regex_search("blk.3.ffn_up.weight", re));        // dense FFN
    CHECK_FALSE(std::regex_search("blk.3.ffn_up_shexp.weight", re));  // shared expert stays on GPU
    CHECK_FALSE(std::regex_search("blk.3.attn_q.weight", re));
    CHECK_FALSE(std::regex_search("blk.3.ffn_gate_inp.weight", re));  // router
}

TEST_CASE("llamacpp wrapper: default context params are unchanged and match llama.cpp's") {
    const ContextParamsSummary upstream = LlamaCppBackend::UpstreamContextDefaults();
    // Pinned b11195 defaults; if an upgrade moves them, Sonder's defaults move too.
    CHECK(upstream.type_k == "f16");
    CHECK(upstream.type_v == "f16");
    CHECK(upstream.flash_attn == "auto");
    CHECK(upstream.n_ubatch == 512u);

    // Default LoadOptions pass exactly what Load() passed before these options
    // existed: n_ubatch = min(llama.cpp default, n_batch), llama.cpp's KV types
    // and flash attention mode.
    LoadOptions d;
    ContextParamsSummary s = LlamaCppBackend::ContextParamsFor(d);
    CHECK(s.n_ctx == 2048u);
    CHECK(s.n_batch == 512u);
    CHECK(s.n_ubatch == 512u);
    CHECK(s.type_k == upstream.type_k);
    CHECK(s.type_v == upstream.type_v);
    CHECK(s.flash_attn == upstream.flash_attn);

    d.n_batch = 64;  // smaller batch caps the micro-batch, as before
    CHECK(LlamaCppBackend::ContextParamsFor(d).n_ubatch == 64u);
    d.n_batch = 2048;  // larger batch keeps llama.cpp's micro-batch
    CHECK(LlamaCppBackend::ContextParamsFor(d).n_ubatch == 512u);
    d.n_ubatch = 1024;  // explicit micro-batch is passed through
    CHECK(LlamaCppBackend::ContextParamsFor(d).n_ubatch == 1024u);
}

TEST_CASE("llamacpp wrapper: KV cache types and flash attention reach llama_context_params") {
    const KvCacheType all[] = {KvCacheType::kF16,  KvCacheType::kF32,  KvCacheType::kBF16,
                               KvCacheType::kQ8_0, KvCacheType::kQ5_1, KvCacheType::kQ5_0,
                               KvCacheType::kQ4_1, KvCacheType::kQ4_0, KvCacheType::kIQ4_NL};
    for (const KvCacheType t : all) {
        CAPTURE(ToString(t));
        LoadOptions o;
        o.type_k = t;
        o.type_v = t;
        const ContextParamsSummary s = LlamaCppBackend::ContextParamsFor(o);
        // ggml's own type name for the value we pass must be our name for it.
        CHECK(s.type_k == ToString(t));
        CHECK(s.type_v == ToString(t));
    }
    LoadOptions o;
    o.type_k = KvCacheType::kQ8_0;
    o.type_v = KvCacheType::kQ4_0;
    ContextParamsSummary s = LlamaCppBackend::ContextParamsFor(o);
    CHECK(s.type_k == "q8_0");
    CHECK(s.type_v == "q4_0");
    o.flash_attn = FlashAttention::kEnabled;
    CHECK(LlamaCppBackend::ContextParamsFor(o).flash_attn == "enabled");
    o.flash_attn = FlashAttention::kDisabled;
    CHECK(LlamaCppBackend::ContextParamsFor(o).flash_attn == "disabled");
    o.flash_attn = FlashAttention::kAuto;
    CHECK(LlamaCppBackend::ContextParamsFor(o).flash_attn == "auto");
}

TEST_CASE("llamacpp wrapper: Load rejects bad context options before touching the file") {
    LlamaCppBackend be;
    LoadOptions opts;
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.type_v = KvCacheType::kQ8_0;
    opts.flash_attn = FlashAttention::kDisabled;
    const Status s = be.Load(opts);
    CHECK(s.code == ErrorCode::kInvalidArgument);  // not kFileNotFound
    CHECK(s.message.find("flash attention") != std::string::npos);
    opts = {};
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.n_batch = 128;
    opts.n_ubatch = 256;
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    // Valid options still reach the file check.
    opts.n_ubatch = 128;
    opts.type_k = KvCacheType::kQ4_0;
    opts.type_v = KvCacheType::kQ4_0;
    CHECK(be.Load(opts).code == ErrorCode::kFileNotFound);
    CHECK(!be.IsLoaded());
}

TEST_CASE("llamacpp wrapper: ResolveDevices validates ggml devices upfront") {
    std::vector<void*> resolved;
    CHECK(LlamaCppBackend::ResolveDevices({}, &resolved).ok());
    CHECK(resolved.empty());

    // CPU is always registered in ggml (case-insensitive via ggml_backend_dev_by_name).
    CHECK(LlamaCppBackend::ResolveDevices({"CPU"}, &resolved).ok());
    REQUIRE(resolved.size() == 1u);
    CHECK(resolved.front() != nullptr);
    CHECK(LlamaCppBackend::ResolveDevices({"cpu"}).ok());

    // Empty, unknown, or over-limit device entries fail cleanly without partial output.
    CHECK(LlamaCppBackend::ResolveDevices({""}).code == ErrorCode::kInvalidArgument);
    CHECK(LlamaCppBackend::ResolveDevices({"NoSuchDevice9"}).code == ErrorCode::kInvalidArgument);
    CHECK(LlamaCppBackend::ResolveDevices({"CPU", "NoSuchDevice9"}, &resolved).code == ErrorCode::kInvalidArgument);
    CHECK(resolved.empty());
    CHECK(LlamaCppBackend::ResolveDevices(std::vector<std::string>(kMaxTensorSplitDevices + 1, "CPU")).code ==
          ErrorCode::kInvalidArgument);
}

TEST_CASE("llamacpp wrapper: placement options reach llama_model_params") {
    const ModelParamsSummary upstream = LlamaCppBackend::UpstreamModelDefaults();
    CHECK(upstream.split_mode == "layer");
    CHECK(upstream.main_gpu == 0);
    CHECK(upstream.load_mode == "auto");
    CHECK_FALSE(upstream.has_tensor_split);
    CHECK_FALSE(upstream.has_devices);

    LoadOptions d;
    ModelParamsSummary s = LlamaCppBackend::ModelParamsFor(d);
    CHECK(s.n_gpu_layers == 0);
    CHECK(s.main_gpu == 0);
    CHECK(s.split_mode == "layer");
    CHECK(s.load_mode == "auto");
    CHECK_FALSE(s.has_tensor_split);
    CHECK_FALSE(s.has_devices);

    // Split modes map to LLAMA_SPLIT_MODE_NONE / LAYER / ROW.
    d.split_mode = SplitMode::kNone;
    CHECK(LlamaCppBackend::ModelParamsFor(d).split_mode == "none");
    d.split_mode = SplitMode::kRow;
    CHECK(LlamaCppBackend::ModelParamsFor(d).split_mode == "row");
    d.split_mode = SplitMode::kLayer;
    CHECK(LlamaCppBackend::ModelParamsFor(d).split_mode == "layer");

    // main_gpu, tensor_split, and devices reach llama_model_params.
    d.main_gpu = 2;
    d.tensor_split = {0.6F, 0.4F};
    d.devices = {"CPU"};
    s = LlamaCppBackend::ModelParamsFor(d);
    CHECK(s.main_gpu == 2);
    CHECK(s.has_tensor_split);
    CHECK(s.has_devices);

    // use_mmap + use_mlock map cleanly onto llama.cpp b11195's load_mode enum.
    LoadOptions lm;
    lm.use_mmap = true;
    lm.use_mlock = false;
    CHECK(LlamaCppBackend::ModelParamsFor(lm).load_mode == "auto");
    lm.use_mmap = false;
    lm.use_mlock = false;
    CHECK(LlamaCppBackend::ModelParamsFor(lm).load_mode == "none");
    lm.use_mmap = true;
    lm.use_mlock = true;
    CHECK(LlamaCppBackend::ModelParamsFor(lm).load_mode == "mmap+mlock");
    lm.use_mmap = false;
    lm.use_mlock = true;
    CHECK(LlamaCppBackend::ModelParamsFor(lm).load_mode == "mlock");
}

TEST_CASE("llamacpp wrapper: Load rejects bad placement options before touching the file") {
    LlamaCppBackend be;

    // main_gpu < 0 refused before file check.
    LoadOptions opts;
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.main_gpu = -1;
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);

    // Bad tensor_split values refused before file check.
    opts = {};
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.tensor_split = {std::numeric_limits<float>::quiet_NaN()};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    opts.tensor_split = {std::numeric_limits<float>::infinity()};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    opts.tensor_split = {-1.0F, 2.0F};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    opts.tensor_split = {0.0F, 0.0F};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    opts.tensor_split.assign(kMaxTensorSplitDevices + 1, 1.0F);
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);

    // Unknown or empty explicit device refused before file check.
    opts = {};
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.devices = {"NoSuchDevice9"};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);
    opts.devices = {""};
    CHECK(be.Load(opts).code == ErrorCode::kInvalidArgument);

    // Valid placement options reach the file check.
    opts = {};
    opts.model_path = "this/file/does/not/exist.gguf";
    opts.devices = {"CPU"};
    opts.main_gpu = 0;
    opts.split_mode = SplitMode::kRow;
    opts.tensor_split = {0.7F, 0.3F};
    opts.use_mlock = true;
    CHECK(be.Load(opts).code == ErrorCode::kFileNotFound);
    CHECK(!be.IsLoaded());

    // Vocab-only load exercises explicit devices + padded tensor_split end to end.
    opts = {};
    opts.model_path = std::string(SONDER_LLAMACPP_VOCAB_DIR) + "/ggml-vocab-llama-spm.gguf";
    opts.vocab_only = true;
    opts.devices = {"CPU"};
    opts.main_gpu = 0;
    opts.split_mode = SplitMode::kNone;
    opts.tensor_split = {1.0F};
    REQUIRE(be.Load(opts).ok());
    CHECK(be.IsLoaded());
    be.Unload();
}

TEST_CASE("llamacpp adapter: multi-GPU placement options validate and forward") {
    namespace si = sonder::inference;
    const si::LlamaCppBackendOptions d;
    CHECK(d.devices.empty());
    CHECK(d.main_gpu == 0);
    CHECK(d.split_mode == "layer");
    CHECK(d.tensor_split.empty());
    CHECK_FALSE(d.use_mlock);
    CHECK(si::validate_llamacpp_options(d).ok());

    si::LlamaCppBackendOptions o;
    o.devices = {"CUDA0", "CUDA1"};
    o.main_gpu = 1;
    o.split_mode = "ROW";
    o.tensor_split = {0.5F, 0.5F};
    o.use_mlock = true;
    CHECK(si::validate_llamacpp_options(o).ok());

    o.split_mode = "none";
    CHECK(si::validate_llamacpp_options(o).ok());
    o.split_mode = "invalid";
    auto st = si::validate_llamacpp_options(o);
    CHECK(st.code() == si::ErrorCode::invalid_argument);
    CHECK(st.message().find("split mode") != std::string::npos);

    o = {};
    o.main_gpu = -1;
    CHECK(si::validate_llamacpp_options(o).code() == si::ErrorCode::invalid_argument);

    o = {};
    o.tensor_split = {0.0F, 0.0F};
    CHECK(si::validate_llamacpp_options(o).code() == si::ErrorCode::invalid_argument);
    o.tensor_split = {std::numeric_limits<float>::quiet_NaN()};
    CHECK(si::validate_llamacpp_options(o).code() == si::ErrorCode::invalid_argument);
    o.tensor_split = {-0.25F, 1.0F};
    CHECK(si::validate_llamacpp_options(o).code() == si::ErrorCode::invalid_argument);
    o.tensor_split.assign(kMaxTensorSplitDevices + 1, 1.0F);
    CHECK(si::validate_llamacpp_options(o).code() == si::ErrorCode::invalid_argument);

    o = {};
    o.devices = {""};
    CHECK(si::validate_llamacpp_options(o).code() == si::ErrorCode::invalid_argument);

    // Unknown explicit device is forwarded to Load() and rejected before touching the file.
    si::LlamaCppBackendOptions bad_dev;
    bad_dev.devices = {"NoSuchDevice9"};
    auto be = si::make_llamacpp_backend(bad_dev);
    si::ModelLoadOptions ml;
    ml.model = "this/file/does/not/exist.gguf";
    CHECK(be->load_model(ml).status().code() == si::ErrorCode::invalid_argument);
}

TEST_CASE("llamacpp adapter: explicit devices honor gpu_layers when device_id is empty") {
    namespace si = sonder::inference;
    using si::llamacpp_detail::ResolveGpuLayers;

    // Default (no explicit devices) with empty device_id remains CPU-only (0 layers).
    si::LlamaCppBackendOptions defaults;
    defaults.gpu_layers = -1;
    auto r = ResolveGpuLayers(defaults, "", /*has_gpu_backend=*/true);
    REQUIRE(r.ok());
    CHECK(r.value() == 0);

    // Explicit ggml devices (e.g. CUDA0 or Vulkan0) with empty device_id honor gpu_layers.
    si::LlamaCppBackendOptions explicit_cuda;
    explicit_cuda.devices = {"CUDA0"};
    explicit_cuda.gpu_layers = -1;
    r = ResolveGpuLayers(explicit_cuda, "", /*has_gpu_backend=*/false);
    REQUIRE(r.ok());
    CHECK(r.value() == -1);

    si::LlamaCppBackendOptions explicit_vulkan;
    explicit_vulkan.devices = {"Vulkan0"};
    explicit_vulkan.gpu_layers = 28;
    r = ResolveGpuLayers(explicit_vulkan, "", /*has_gpu_backend=*/true);
    REQUIRE(r.ok());
    CHECK(r.value() == 28);

    // Explicit "cpu:*" device_id still forces 0 layers even when options.devices is set.
    r = ResolveGpuLayers(explicit_cuda, "cpu:0", /*has_gpu_backend=*/true);
    REQUIRE(r.ok());
    CHECK(r.value() == 0);

    // Explicit "gpu:*" checks GPU backend availability and honors gpu_layers.
    r = ResolveGpuLayers(defaults, "gpu:0", /*has_gpu_backend=*/true);
    REQUIRE(r.ok());
    CHECK(r.value() == -1);
    CHECK(ResolveGpuLayers(defaults, "gpu:0", /*has_gpu_backend=*/false).status().code() ==
          si::ErrorCode::unsupported);

    // Unsupported device_id prefix still fails with unsupported.
    CHECK(ResolveGpuLayers(explicit_cuda, "npu:0", /*has_gpu_backend=*/true).status().code() ==
          si::ErrorCode::unsupported);

    // End-to-end: empty device_id with a valid explicit ggml device ("CPU") passes
    // device resolution cleanly and reaches the missing-file check.
    si::LlamaCppBackendOptions explicit_cpu;
    explicit_cpu.devices = {"CPU"};
    explicit_cpu.gpu_layers = -1;
    auto be = si::make_llamacpp_backend(explicit_cpu);
    si::ModelLoadOptions ml;
    ml.model = "this/file/does/not/exist.gguf";
    ml.device_id = "";
    CHECK(be->load_model(ml).status().code() == si::ErrorCode::not_found);
}
