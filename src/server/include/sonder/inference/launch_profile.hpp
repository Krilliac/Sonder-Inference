// Sonder Inference: typed per-model launch profiles for `sonder-infer serve`.
//
// A launch profile names one model and every serving setting it needs, as
// typed fields rather than free-form arguments. The same profile drives both
// serving backends:
//
//   llamaserver  the fields become a llama-server argument vector
//                (llamaserver_arguments), checked against the installed
//                llama-server's `--help`; the supervisor still appends its own
//                --host 127.0.0.1 and --port.
//   llamacpp     the fields map onto the direct backend's options
//                (apply_llamacpp_profile); a field that backend cannot honour
//                is rejected with "... is not supported by the llamacpp
//                backend; use llamaserver", never dropped.
//
// Default sampling values apply only to requests that do not set the field
// themselves (apply_sampling_defaults marks them explicit, so the upstream
// receives them exactly like caller-set values).
//
// A VRAM estimate from the GGUF header (read_gguf_model_info, estimate_vram)
// gates `serve` against a budget. Reference: docs/integration/launch-profiles.md.
// Part of the server module (SONDER_HAS_SERVER).
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "sonder/inference/backend_setup.hpp"
#include "sonder/inference/error.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/model_architecture.hpp"
#include "sonder/inference/sampling.hpp"

namespace sonder::inference {

// Sampling values a profile supplies for requests that leave them unset.
struct SamplingDefaults {
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<std::int32_t> top_k;
    std::optional<float> min_p;
    std::optional<float> presence_penalty;
    std::optional<float> repeat_penalty;

    [[nodiscard]] bool empty() const noexcept {
        return !temperature && !top_p && !top_k && !min_p && !presence_penalty && !repeat_penalty;
    }
};

// Speculative decoding (llama-server --spec-type, --spec-draft-n-max,
// --spec-draft-model).
struct SpeculativeSettings {
    std::vector<std::string> types;  // e.g. {"draft-mtp"}; see kSpeculativeTypes
    std::optional<std::uint32_t> draft_n_max;
    std::string draft_model;  // GGUF path; empty = none
};

struct LaunchProfile {
    std::string name;     // served model id; [A-Za-z0-9._:-], not "default"
    std::string backend;  // "llamaserver" or "llamacpp"
    std::string model;    // GGUF path

    // Vision projector (llama-server only).
    std::string mmproj;
    std::optional<bool> mmproj_offload;
    std::optional<std::uint32_t> image_max_tokens;

    std::optional<std::uint32_t> ctx_size;       // 0 = model training context
    std::optional<std::int32_t> n_gpu_layers;    // -1 = all
    std::optional<std::uint32_t> batch_size;
    std::optional<std::uint32_t> ubatch_size;
    std::optional<std::uint32_t> parallel;
    std::optional<bool> kv_unified;
    // KV cache and flash attention: the llamacpp backend's option names and
    // values (LlamaCppBackendOptions::kv_cache_type_k/_v, flash_attention).
    std::optional<std::string> flash_attn;    // on, off or auto
    std::optional<std::string> cache_type_k;  // f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1
    std::optional<std::string> cache_type_v;
    std::optional<std::uint32_t> ctx_checkpoints;
    std::optional<std::uint32_t> checkpoint_min_step;
    std::optional<std::int64_t> cache_ram_mib;  // -1 = no limit, 0 = disabled
    std::optional<bool> context_shift;
    std::optional<bool> fit;
    std::optional<SpeculativeSettings> speculative;
    std::optional<std::int32_t> threads;
    std::optional<std::int32_t> threads_batch;
    std::optional<bool> jinja;
    std::optional<std::string> reasoning_format;  // none, deepseek, deepseek-legacy, auto

    SamplingDefaults sampling;

    // Further llama-server arguments, passed verbatim after the typed ones.
    // Host, port, API-key, download, model-preset, prompt/log-file, agent
    // tool, external-program, remote RPC and argument-list-terminator flags
    // are refused, as are flags that duplicate a typed field above
    // (validate_launch_profile). Every flag is checked under the name
    // llama-server looks it up by (llamaserver_flag_name), so `--api_key` is
    // refused exactly like `--api-key`.
    std::vector<std::string> extra_args;

    // VRAM budget for the fit check. serve's --vram-budget-mib takes
    // precedence; with neither, the detected device total minus
    // kDefaultVramReserveMib.
    std::optional<std::uint64_t> vram_budget_mib;
};

inline constexpr const char* kLaunchProfileBackendLlamaServer = "llamaserver";
inline constexpr const char* kLaunchProfileBackendLlamaCpp = "llamacpp";
inline constexpr std::uint64_t kDefaultVramReserveMib = 1536;  // 1.5 GiB
// llama-server's --parallel defaults to -1 (auto): 4 slots with a unified KV
// cache, set unconditionally (tools/server/server.cpp, llama.cpp 7fe450e; the
// bundled 161755f29 logs "n_parallel is set to auto, using n_parallel = 4 and
// kv_unified = true"). A llamaserver profile without `parallel` runs that.
inline constexpr std::uint32_t kLlamaServerAutoParallel = 4;

// The name llama-server looks a command-line argument up by. llama.cpp's
// parser (common_params_parse_ex, common/arg.cpp at 7fe450e; the bundled
// 161755f29 behaves the same) rewrites every '_' to '-' in an argument that
// starts with "--" before the lookup, so `--api_key` is `--api-key` to
// llama-server. Single-dash flags are looked up as written. Anything from the
// first '=' on is dropped: llama-server has no --flag=value form and refuses
// such an argument, so the name before it is what a check has to judge.
std::string llamaserver_flag_name(std::string_view arg);

// KV cache element types accepted by llama-server --cache-type-k/-v and by
// the llamacpp backend, and their storage in bytes per cached value
// (ggml block layouts). Returns 0 for an unknown name.
double kv_cache_type_bytes(std::string_view type) noexcept;
[[nodiscard]] bool is_quantized_kv_cache_type(std::string_view type) noexcept;

// llama-server --spec-type values (verified against `llama-server --help`).
inline constexpr std::string_view kSpeculativeTypes[] = {
    "none",         "draft-simple", "draft-eagle3", "draft-mtp",     "draft-dflash", "draft-dspark",
    "ngram-simple", "ngram-map-k",  "ngram-map-k4v", "ngram-mod",    "ngram-cache"};

// Strict parsing: unknown keys, wrong types and out-of-range values are
// errors (invalid_argument naming the profile and field). `document` is
// {"profiles": [ {...}, ... ]}. Profile names must be unique.
Result<std::vector<LaunchProfile>> parse_launch_profiles(const json::Value& document);
// Reads and parses a profile file (at most 1 MiB).
Result<std::vector<LaunchProfile>> load_launch_profiles(const std::string& path);
// Cross-field checks shared by both backends (ranges, ubatch <= batch, a
// quantized V cache needs flash attention, mmproj options need mmproj,
// extra_args policy). parse_launch_profiles() runs it on every profile.
Status validate_launch_profile(const LaunchProfile& profile);
const LaunchProfile* find_launch_profile(const std::vector<LaunchProfile>& profiles, std::string_view name);

// The llama-server argument vector for `profile` (backend llamaserver), in a
// fixed order: --model, --alias <name>, the typed fields, then extra_args.
// Sampling defaults are not arguments: Sonder applies them per request.
Result<std::vector<std::string>> llamaserver_arguments(const LaunchProfile& profile);

// Maps a llamacpp profile onto `setup`: context length, GPU layers, batch and
// micro-batch sizes, K and V cache types and flash attention (the
// BackendSetup::llamacpp_* fields `serve`'s --batch-size, --ubatch-size,
// --cache-type-k/-v and --flash-attn set, with the same value names). Nothing
// is applied on error. Returns invalid_argument for any field the direct
// backend cannot honour. `device` is serve's --device: a profile that offloads
// layers sets "gpu:0" when it is empty and refuses a cpu device.
Status apply_llamacpp_profile(const LaunchProfile& profile, BackendSetup& setup, std::string& device);

// Sets every default the request did not set explicitly and marks it
// explicit. Caller-set fields are never changed.
void apply_sampling_defaults(const SamplingDefaults& defaults, SamplingConfig& sampling);

// ------------------------------------------------------------------ VRAM fit

// What the estimator needs from a GGUF header. Tensor sizes come from the
// tensor table; no tensor data is read.
struct GgufModelInfo {
    std::string architecture;  // general.architecture
    ModelArchitecture kind = ModelArchitecture::attention_only;
    std::uint32_t block_count = 0;
    std::uint32_t nextn_layers = 0;  // trailing MTP blocks (<arch>.nextn_predict_layers)
    std::uint64_t embedding_length = 0;
    std::uint64_t vocab_size = 0;
    std::uint64_t key_length = 0;    // per-head K dimension (full-attention blocks)
    std::uint64_t value_length = 0;  // per-head V dimension (full-attention blocks)
    // Per block: KV heads (0 = no attention KV in that block).
    std::vector<std::uint32_t> kv_heads;
    // Sliding-window attention (SWA), resolved as llama.cpp's loader resolves
    // it for the architectures vram_estimate.cpp lists (kSwaArchitectures).
    // Per block: true = the block attends to the last `sliding_window` tokens
    // only, and llama.cpp keeps its KV in a separate, smaller cache. Empty =
    // no SWA blocks: the architecture has none, this file does not enable
    // them, or the architecture is not one whose SWA layout is modelled.
    std::vector<bool> swa_layers;
    std::uint64_t sliding_window = 0;    // window in tokens (llama.cpp n_swa)
    // Per-head K and V dimensions on SWA blocks: <arch>.attention.key_length_swa
    // and value_length_swa; the parser sets key_length/value_length without them.
    std::uint64_t key_length_swa = 0;
    std::uint64_t value_length_swa = 0;
    // <arch>.attention.sliding_window is set, but the architecture's SWA
    // layout is not modelled: every block counts at full context.
    bool sliding_window_unmodelled = false;
    // Per block: tensor bytes of blk.N.*.
    std::vector<std::uint64_t> block_bytes;
    std::uint64_t output_bytes = 0;      // output.weight and output_norm (GPU when all layers offload)
    std::uint64_t embedding_bytes = 0;   // token_embd (kept on the CPU by llama.cpp)
    std::uint64_t other_bytes = 0;       // remaining non-block tensors
    // Recurrent (SSM / delta-net) state per recurrent block and sequence, in
    // f32 values: conv (kernel-1) x channels + state.
    std::uint64_t recurrent_state_values = 0;
    ModelMetadata metadata;  // scalar keys as strings (for classification)
};

// Parses the GGUF header of `path` (versions 2 and 3). Errors: not_found,
// io_error, invalid_argument for a malformed or truncated header, a
// general.architecture longer than 256 bytes or not printable ASCII, or
// implausible recurrent (ssm.*) sizes. Keys and tensor names quoted in an
// error are cut to 64 printable characters (errors reach /v1/models).
Result<GgufModelInfo> read_gguf_model_info(const std::string& path);
Result<GgufModelInfo> parse_gguf_model_info(std::string_view header_bytes);
// `error` (from read_gguf_model_info(path)) with every occurrence of `path`
// replaced by "<model path>", for client-visible metadata such as
// /v1/models `estimate_error`; local filesystem paths stay in operator logs.
std::string redact_model_path(const Status& error, const std::string& path);

struct VramEstimate {
    std::uint64_t weights_bytes = 0;  // offloaded blocks (+ output) + mmproj/draft when offloaded
    // (kv_bytes_per_token x context_per_sequence + swa_kv_bytes_per_token x
    // swa_context_per_sequence) x sequences
    std::uint64_t kv_bytes = 0;
    std::uint64_t recurrent_bytes = 0;  // recurrent state of GPU blocks x recurrent_sequences x recurrent_snapshots
    std::uint64_t compute_bytes = 0;    // compute buffers and runtime allowance
    std::uint64_t total_bytes = 0;
    std::uint32_t gpu_layers = 0;
    std::uint32_t attention_layers = 0;      // attention layers whose KV is on the GPU (full and sliding-window)
    std::uint32_t swa_attention_layers = 0;  // of those, sliding-window layers
    // KV bytes per cached token: over the full-attention layers on the GPU,
    // and over the sliding-window ones.
    std::uint64_t kv_bytes_per_token = 0;
    std::uint64_t swa_kv_bytes_per_token = 0;
    std::uint64_t context = 0;               // total context (--ctx-size, padded to 256)
    std::uint64_t context_per_sequence = 0;  // context when unified; else pad(context / sequences, 256)
    // Tokens per sequence stream in llama.cpp's sliding-window cache:
    // pad(min(context_per_sequence, sliding_window x (unified ? parallel : 1)
    // + ubatch), 256), or context_per_sequence when that cache is full size
    // (--swa-full in extra_args; always on the llamacpp backend). 0 without
    // sliding-window layers on the GPU.
    std::uint64_t swa_context_per_sequence = 0;
    std::uint32_t sequences = 1;  // KV streams: 1 when unified, else parallel
    // Sequences that each keep a recurrent state (llama.cpp's n_seq_max):
    // `parallel`, kLlamaServerAutoParallel for a llamaserver profile without
    // it, 1 on the llamacpp backend.
    std::uint32_t recurrent_sequences = 1;
    // Recurrent states kept per sequence: 1, or 1 + --spec-draft-n-max
    // (default 3) when draft-mtp/eagle3/dflash/dspark roll the state back.
    std::uint32_t recurrent_snapshots = 1;
    std::vector<std::string> notes;  // what the estimate does not cover

    // Rounded up; cannot wrap, even for a total near 2^64.
    [[nodiscard]] std::uint64_t total_mib() const noexcept {
        return (total_bytes >> 20) + ((total_bytes & ((1u << 20) - 1)) != 0 ? 1 : 0);
    }
};

// Extra files the estimate needs sizes of (mmproj, draft model); 0 = absent.
struct VramEstimateInputs {
    std::uint64_t mmproj_bytes = 0;
    std::uint64_t draft_model_bytes = 0;
};

// Estimate for serving `profile` with `model`:
//   weights  blocks on the GPU (the last n_gpu_layers, llama.cpp order),
//            plus output tensors when every block is offloaded; MTP (nextn)
//            blocks only when speculative types include draft-mtp
//   KV       sum over GPU attention layers of n_kv_heads x (key_length x
//            bytes(k) + value_length x bytes(v)) x cells x sequences:
//            kv_unified gives one pool of ctx (sequences 1); otherwise
//            sequences = parallel and ctx_seq = pad(ctx / parallel, 256),
//            as llama.cpp splits --ctx-size across the per-sequence streams.
//            A llamaserver profile without `parallel` gets llama-server's
//            auto default: kLlamaServerAutoParallel slots, unified.
//            Full-attention layers keep ctx_seq cells. Sliding-window layers
//            use key_length_swa/value_length_swa and keep llama.cpp's
//            size_swa = pad(min(ctx_seq, sliding_window x (unified ? slots :
//            1) + ubatch), 256) cells, or ctx_seq with --swa-full in
//            extra_args and on the llamacpp backend (llama.cpp's library
//            default swa_full, which that backend keeps). With flash_attn
//            off, every layer's V row is the widest layer's (llama.cpp pads
//            the transposed V cache)
//   recurrent  for each GPU block without attention in a hybrid or
//            recurrent model: state values x 4 bytes x recurrent_sequences
//            (parallel; kLlamaServerAutoParallel when a llamaserver profile
//            leaves it unset) x recurrent_snapshots (1 + --spec-draft-n-max
//            when draft-mtp, draft-eagle3, draft-dflash or draft-dspark roll
//            the state back on an architecture llama.cpp supports that for)
//   compute  n_vocab x ubatch x 4 (logits) + 4 x n_embd x ubatch x 4
//            + 256 MiB runtime allowance
// ctx 0 or unset uses <arch>.context_length. Hybrid models count only their
// attention layers (per-block head_count_kv, attn_k tensors, or
// full_attention_interval). MLA (DeepSeek-style) layers are counted as
// regular attention, which overestimates them.
VramEstimate estimate_vram(const GgufModelInfo& model, const LaunchProfile& profile,
                           const VramEstimateInputs& inputs = {});

// Largest dedicated video memory of any hardware adapter, in MiB (Windows
// DXGI); nullopt where unknown (other platforms, no adapter).
std::optional<std::uint64_t> detect_device_vram_mib();

// `sonder.profile` metadata for /v1/models (additive): backend, context
// length, parallel, cache types, flash attention, capabilities (usable
// through Sonder's endpoint), upstream_capabilities (supported by the
// upstream server, possibly not yet forwarded by Sonder), estimated VRAM and
// budget. Unknown values are null.
json::Object launch_profile_metadata(const LaunchProfile& profile, const std::optional<VramEstimate>& estimate,
                                     std::optional<std::uint64_t> budget_mib);

}  // namespace sonder::inference
