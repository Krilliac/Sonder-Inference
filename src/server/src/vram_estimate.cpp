// VRAM and host-RAM fit estimation for launch profiles (launch_profile.hpp),
// plus dedicated device VRAM detection. GGUF header parsing lives in
// gguf_header.cpp; MoE tensor placement flag parsing/evaluation lives in
// moe_placement.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/device.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "vram_internal.hpp"

namespace sonder::inference {

namespace {

using detail::kMiB;
using detail::saturating_add;
using detail::saturating_mul;
using detail::saturating_u64;

// Speculative decoding that rolls back recurrent state keeps one state
// snapshot per draft position: llama.cpp allocates max(1, n_seq_max) x
// (1 + n_rs_seq) recurrent rows per layer, with n_rs_seq = --spec-draft-n-max
// for these types (common_params_speculative::need_n_rs_seq), clamped to 0 on
// architectures without recurrent rollback (llm_arch_supports_rs_rollback;
// llama-memory-recurrent.cpp; llama.cpp source at 7fe450e). On the bundled
// llama-server (161755f29) each +1 of --spec-draft-n-max measured +150 MiB
// with Qwen3.8-27B, whose snapshot is 149.6 MiB.
constexpr std::string_view kRollbackSpeculativeTypes[] = {"draft-mtp", "draft-eagle3", "draft-dflash", "draft-dspark"};
constexpr std::string_view kRecurrentRollbackArchitectures[] = {
    "qwen35", "qwen35moe", "qwen4exp", "deepseek4", "nemotron_h", "nemotron_h_moe", "bailingmoe3", "lfm2", "lfm2moe",
    "kimi-k3"};
constexpr std::uint32_t kDefaultDraftNMax = 3;  // llama.cpp common_params_speculative_draft::n_max
// Host workspace outside the weights/KV: output buffers, tokenizer, server.
constexpr std::uint64_t kHostWorkspaceBytes = 256 * kMiB;

// n_rs_seq for `profile` on `model`: the extra recurrent snapshots per sequence.
std::uint32_t recurrent_rollback_snapshots(const GgufModelInfo& model, const LaunchProfile& profile) {
    if (!profile.speculative) return 0;
    const auto& types = profile.speculative->types;
    const bool rolls_back = std::any_of(types.begin(), types.end(), [](const std::string& t) {
        return std::find(std::begin(kRollbackSpeculativeTypes), std::end(kRollbackSpeculativeTypes), t) !=
               std::end(kRollbackSpeculativeTypes);
    });
    const bool supported =
        std::find(std::begin(kRecurrentRollbackArchitectures), std::end(kRecurrentRollbackArchitectures),
                  model.architecture) != std::end(kRecurrentRollbackArchitectures);
    return rolls_back && supported ? profile.speculative->draft_n_max.value_or(kDefaultDraftNMax) : 0;
}

bool is_device_none(const std::optional<std::string>& device) noexcept {
    if (!device || device->size() != 4) return false;
    const auto lower = [](char c) noexcept {
        return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
    };
    return lower((*device)[0]) == 'n' && lower((*device)[1]) == 'o' && lower((*device)[2]) == 'n' &&
           lower((*device)[3]) == 'e';
}

}  // namespace

VramEstimate estimate_vram(const GgufModelInfo& m, const LaunchProfile& p, const VramEstimateInputs& inputs) {
    VramEstimate e;
    const std::uint32_t n_layer = m.block_count;
    const bool mtp =
        p.speculative &&
        std::find(p.speculative->types.begin(), p.speculative->types.end(), "draft-mtp") != p.speculative->types.end();
    // llama.cpp: `--device none` disables GPU offload regardless of n_gpu_layers;
    // otherwise -1/all, or more layers than blocks, also offloads the output.
    const bool device_none = is_device_none(p.device);
    const std::int64_t requested = device_none ? 0 : p.n_gpu_layers.value_or(-1);
    const bool all = !device_none && (requested < 0 || requested > static_cast<std::int64_t>(n_layer));
    const std::uint32_t offloaded = all ? n_layer : static_cast<std::uint32_t>(requested);
    const std::uint32_t first_gpu = n_layer - offloaded;
    const std::uint32_t first_nextn = n_layer - std::min(m.nextn_layers, n_layer);

    const std::uint64_t ctx_raw = p.ctx_size && *p.ctx_size > 0 ? *p.ctx_size : 0;
    std::uint64_t ctx = ctx_raw;
    if (ctx == 0) {
        for (const auto& [k, v] : m.metadata) {
            if (k == m.architecture + ".context_length") {
                try {
                    ctx = std::stoull(v);
                } catch (...) {
                    ctx = 0;
                }
            }
        }
        e.notes.emplace_back("ctx_size unset: the model's training context is assumed");
    }
    ctx = (ctx + 255) / 256 * 256;  // llama.cpp pads the context to 256
    e.context = ctx;
    // Without `parallel`, llama-server runs its automatic default: 4 slots
    // (n_seq_max 4, so 4 recurrent states) with a unified KV cache
    // (kLlamaServerAutoParallel). The direct llamacpp backend decodes one
    // sequence and refuses `parallel`.
    const bool auto_parallel = !p.parallel && p.backend == kLaunchProfileBackendLlamaServer;
    const std::uint32_t slots = p.parallel ? *p.parallel : (auto_parallel ? kLlamaServerAutoParallel : 1u);
    const bool unified = auto_parallel || p.kv_unified.value_or(!p.parallel.has_value());
    e.sequences = unified ? 1 : slots;
    e.recurrent_sequences = slots;
    // --ctx-size is the total KV size. A non-unified cache splits it into one
    // stream per sequence of n_ctx_seq = pad(n_ctx / n_seq_max, 256) cells
    // (llama_context's constructor, llama.cpp b11195), so the pool is
    // n_ctx_seq x parallel, not ctx x parallel. Unified: one pool of ctx.
    std::uint64_t ctx_per_sequence = ctx;
    if (e.sequences > 1) {
        ctx_per_sequence = (ctx / e.sequences + 255) / 256 * 256;
    }
    e.context_per_sequence = ctx_per_sequence;

    const double bk = kv_cache_type_bytes(p.cache_type_k.value_or("f16"));
    const double bv = kv_cache_type_bytes(p.cache_type_v.value_or("f16"));
    // Micro-batch: --ubatch-size, at most --batch-size (llama_context). The
    // defaults are llama-server's 2048/512 and the direct backend's 512/512.
    const std::uint64_t batch = p.batch_size.value_or(p.backend == kLaunchProfileBackendLlamaCpp ? 512u : 2048u);
    const std::uint64_t ubatch = std::min<std::uint64_t>(p.ubatch_size.value_or(512), batch);
    // Recurrent rows per layer: one state per sequence, times (1 + n_rs_seq)
    // snapshots when the speculative types roll recurrent state back.
    e.recurrent_snapshots = 1 + recurrent_rollback_snapshots(m, p);
    const std::uint64_t recurrent_rows = static_cast<std::uint64_t>(slots) * e.recurrent_snapshots;
    const auto sliding = [&m](std::uint32_t i) { return i < m.swa_layers.size() && m.swa_layers[i]; };
    // With flash attention off llama.cpp stores V transposed and gives every
    // layer the widest layer's V row (n_embd_v_gqa_max over all blocks;
    // llama_kv_cache, [TAG_V_CACHE_VARIABLE]). `auto` and `on` allocate the
    // cache before flash attention is resolved, with each layer's own row.
    const bool v_padded = p.flash_attn && (*p.flash_attn == "off" || *p.flash_attn == "disabled");
    double v_row_max = 0.0;
    if (v_padded) {
        for (std::uint32_t i = 0; i < n_layer; ++i) {
            const double v = static_cast<double>(sliding(i) ? m.value_length_swa : m.value_length);
            v_row_max = std::max(v_row_max, static_cast<double>(m.kv_heads[i]) * v);
        }
    }
    // Tensor placement (-ot, --n-cpu-moe, --cpu-moe). The first rule that
    // matches a tensor decides; a tensor no rule matches follows its block
    // (offloaded blocks on the GPU, the rest on the host).
    MoePlacement placement;
    {
        auto parsed = parse_moe_placement(p.extra_args);
        if (parsed.ok()) {
            placement = std::move(parsed.value());
        } else {
            e.moe_placement_error = parsed.status().message();
        }
    }
    if (p.load_mode) {
        if (*p.load_mode == "auto" || *p.load_mode == "mmap") {
            placement.mmap = true;
            placement.mlock = false;
        } else if (*p.load_mode == "none" || *p.load_mode == "dio") {
            placement.mmap = false;
            placement.mlock = false;
        } else if (*p.load_mode == "mlock") {
            placement.mmap = false;
            placement.mlock = true;
        } else if (*p.load_mode == "mmap+mlock") {
            placement.mmap = true;
            placement.mlock = true;
        }
    }
    if (p.no_repack) {
        placement.repack = !*p.no_repack;
    }
    detail::TensorPlacementBreakdown placed =
        detail::evaluate_tensor_placement(m, placement, n_layer, first_gpu, first_nextn, mtp, e.notes);
    const bool fit_path = p.backend == kLaunchProfileBackendLlamaServer && p.fit.value_or(true) && !p.n_gpu_layers &&
                          !device_none && !placed.user_overrides;

    double kv_per_token = 0.0;      // full-attention layers on the GPU
    double swa_kv_per_token = 0.0;  // sliding-window layers on the GPU
    for (std::uint32_t i = first_gpu; i < n_layer; ++i) {
        if (i >= first_nextn && !mtp) continue;  // MTP blocks load only for draft-mtp
        e.weights_bytes += m.block_bytes[i] - std::min(m.block_bytes[i], placed.moved_to_host[i]);
        ++e.gpu_layers;
        if (m.kv_heads[i] > 0) {
            ++e.attention_layers;
            const bool swa = sliding(i);
            const double heads = static_cast<double>(m.kv_heads[i]);
            const double k = static_cast<double>(swa ? m.key_length_swa : m.key_length);
            const double v = static_cast<double>(swa ? m.value_length_swa : m.value_length);
            const double bytes = v_padded ? heads * k * bk + v_row_max * bv : heads * (k * bk + v * bv);
            if (swa) {
                ++e.swa_attention_layers;
                swa_kv_per_token += bytes;
            } else {
                kv_per_token += bytes;
            }
        } else if (m.kind != ModelArchitecture::attention_only) {
            e.recurrent_bytes += m.recurrent_state_values * 4 * recurrent_rows;
        }
    }
    for (std::uint32_t i = 0; i < first_gpu; ++i) {
        if (i >= first_nextn && !mtp) continue;
        e.weights_bytes = saturating_add(e.weights_bytes, placed.moved_to_gpu[i]);
    }
    if (all) e.weights_bytes += m.output_bytes;
    e.kv_bytes_per_token = saturating_u64(kv_per_token);
    e.swa_kv_bytes_per_token = saturating_u64(swa_kv_per_token);
    double kv = kv_per_token * static_cast<double>(ctx_per_sequence) * e.sequences;
    if (e.swa_attention_layers > 0) {
        // Full size with --swa-full (judged under the name llama-server
        // reads), a caller-supplied inherited setting, and on the direct backend, which keeps the llama.cpp
        // library's default swa_full = true (llama_context_default_params,
        // b11195). llama-server's own default is false.
        bool swa_full_flag = false;
        for (const auto& arg : p.extra_args) {
            swa_full_flag = swa_full_flag || llamaserver_flag_name(arg) == "--swa-full";
        }
        const bool direct = p.backend == kLaunchProfileBackendLlamaCpp;
        if (direct || swa_full_flag || inputs.inherited_swa_full) {
            e.swa_context_per_sequence = ctx_per_sequence;
            e.notes.emplace_back(direct
                                     ? "the llamacpp backend allocates full-size sliding-window caches (llama.cpp's "
                                       "default swa_full); sliding-window layers are counted at full context"
                                 : swa_full_flag ? "--swa-full: sliding-window layers are counted at full context"
                                                 : "LLAMA_ARG_SWA_FULL is set: sliding-window layers are counted at "
                                                   "full context");
        } else {
            // size_swa: window x n_seq_max for one unified stream, the window
            // for each stream of a split cache, plus the micro-batch. Below
            // ctx_per_sequence, a multiple of 256, so the padding cannot wrap.
            const std::uint64_t window = saturating_add(saturating_mul(m.sliding_window, unified ? slots : 1u), ubatch);
            const std::uint64_t cells = std::min(ctx_per_sequence, window);
            e.swa_context_per_sequence = (cells + 255) / 256 * 256;
        }
        kv += swa_kv_per_token * static_cast<double>(e.swa_context_per_sequence) * e.sequences;
    }
    e.kv_bytes = saturating_u64(kv);
    if (m.sliding_window_unmodelled && e.attention_layers > 0) {
        e.notes.emplace_back(m.architecture +
                             ".attention.sliding_window is set, but sliding-window caches are modelled only for "
                             "architectures checked against llama.cpp; every layer is counted at full context");
    }

    const bool mmproj_on_gpu = !p.mmproj.empty() && !device_none && p.mmproj_offload.value_or(true);
    if (mmproj_on_gpu) {
        e.weights_bytes += inputs.mmproj_bytes;
        e.notes.emplace_back("vision encoder compute buffers are not included");
    }
    if (!device_none && p.speculative && !p.speculative->draft_model.empty()) {
        e.weights_bytes += inputs.draft_model_bytes;
        e.notes.emplace_back("the draft model's KV cache is not included");
    }
    if (mtp && m.nextn_layers > 0) {
        e.notes.emplace_back("the MTP draft context's compute buffer, which grows with the context, is not included");
    }
    if (offloaded > 0) {
        e.compute_bytes = m.vocab_size * ubatch * 4 + 4 * m.embedding_length * ubatch * 4 + 256 * kMiB;
    }
    for (const auto& arg : p.extra_args) {
        // The name llama-server reads (`--override_tensor` is `--override-tensor`).
        const std::string flag = llamaserver_flag_name(arg);
        // -ot/--cpu-moe/--n-cpu-moe are modelled when they parsed (see
        // MoePlacement); a malformed one stays ignored and noted.
        const bool moe_flag = flag == "-ot" || flag == "--override-tensor" || flag == "-cmoe" || flag == "--cpu-moe" ||
                              flag == "-ncmoe" || flag == "--n-cpu-moe" || flag == "-ncffn" || flag == "--n-cpu-ffn";
        const bool ffn_flag = flag == "-ncffn" || flag == "--n-cpu-ffn";
        if (moe_flag && !ffn_flag && e.moe_placement_error.empty()) continue;
        if (moe_flag || flag == "-ts" || flag == "--tensor-split" || flag == "-sm" || flag == "--split-mode" ||
            flag == "-dev" || flag == "--device") {
            e.notes.emplace_back("placement flags in extra_args (" + flag +
                                 ") are ignored: the estimate assumes every offloaded block is on one GPU");
            break;
        }
    }
    if (m.kind != ModelArchitecture::attention_only && m.recurrent_state_values == 0) {
        e.notes.emplace_back("recurrent state size unknown for this architecture");
    }
    if (auto_parallel && e.recurrent_bytes > 0) {
        e.notes.emplace_back("parallel unset: llama-server's automatic 4 slots keep 4 recurrent states");
    }
    e.total_bytes = e.weights_bytes + e.kv_bytes + e.recurrent_bytes + e.compute_bytes;
    e.prism_hadamard = m.prism_hadamard;
    e.requires_prism_runtime = m.has_prism_private_tensors || m.prism_hadamard;
    if (e.requires_prism_runtime) {
        std::string note = "model requires a PrismML Hadamard-capable runtime (PQ2_0/PTQ1_0";
        if (!m.prism_hadamard_transform.empty()) {
            note += ", transform=" + m.prism_hadamard_transform;
        }
        if (m.prism_hadamard_block_size > 0) {
            note += ", block_size=" + std::to_string(m.prism_hadamard_block_size);
        }
        note += ")";
        e.notes.push_back(std::move(note));
    }
    if (!p.tensor_split.empty() && e.total_bytes > 0) {
        std::size_t positive_gpus = 0;
        double weight_sum = 0.0;
        double weight_max = 0.0;
        for (const double w : p.tensor_split) {
            if (std::isfinite(w) && w > 0.0) {
                ++positive_gpus;
                weight_sum += w;
                weight_max = std::max(weight_max, w);
            }
        }
        const std::string mode = p.split_mode.value_or("layer");
        if (positive_gpus >= 2 && weight_sum > 0.0) {
            if (mode == "row") {
                // llama.cpp --split-mode row splits weights/rows across GPUs
                // but keeps the KV cache unsplit on main_gpu.
                const double max_share = weight_max / weight_sum;
                const std::uint64_t splittable = saturating_add(e.weights_bytes, e.recurrent_bytes);
                const std::uint64_t shard = std::min(
                    splittable, saturating_u64(std::ceil(static_cast<double>(splittable) * max_share)));
                e.max_gpu_bytes = saturating_add(shard, saturating_add(e.kv_bytes, e.compute_bytes));
                e.notes.emplace_back("multi-GPU tensor_split (" + std::to_string(positive_gpus) +
                                     " GPUs, split_mode=row): heaviest GPU shard estimated at " +
                                     std::to_string(e.max_gpu_mib()) +
                                     " MiB (proportional weights/recurrent + unsplit KV + compute)");
            } else if (mode != "none") {
                const double max_share = weight_max / weight_sum;
                const std::uint64_t splittable =
                    saturating_add(e.weights_bytes, saturating_add(e.kv_bytes, e.recurrent_bytes));
                const std::uint64_t shard = std::min(
                    splittable, saturating_u64(std::ceil(static_cast<double>(splittable) * max_share)));
                e.max_gpu_bytes = saturating_add(shard, e.compute_bytes);
                e.notes.emplace_back("multi-GPU tensor_split (" + std::to_string(positive_gpus) +
                                     " GPUs, split_mode=" + mode + "): heaviest GPU shard estimated at " +
                                     std::to_string(e.max_gpu_mib()) +
                                     " MiB (proportional weights/KV/recurrent + compute)");
            } else {
                e.notes.emplace_back(
                    "split_mode=none: tensor_split is not applied; all offloaded layers stay on one GPU");
            }
        }
    }

    // ---- Host RAM resident estimate.
    // Weights: token_embd and the other non-block tensors, the output tensors
    // unless every block is offloaded, host blocks (minus what -ot sends to
    // the GPU) and the tensors of offloaded blocks that -ot/--n-cpu-moe keep
    // on the host. A vision encoder that is not offloaded also lives here.
    std::uint64_t host_weights = saturating_add(m.embedding_bytes, m.other_bytes);
    if (!all) host_weights = saturating_add(host_weights, m.output_bytes);
    for (std::uint32_t i = 0; i < n_layer && i < m.block_bytes.size(); ++i) {
        if (i >= first_nextn && !mtp) continue;
        if (i < first_gpu) {
            host_weights = saturating_add(host_weights, m.block_bytes[i] - std::min(m.block_bytes[i], placed.moved_to_gpu[i]));
        } else {
            host_weights = saturating_add(host_weights, placed.moved_to_host[i]);
        }
    }
    if (!p.mmproj.empty() && !mmproj_on_gpu) {
        host_weights = saturating_add(host_weights, inputs.mmproj_bytes);
    }
    if (device_none && p.speculative && !p.speculative->draft_model.empty()) {
        host_weights = saturating_add(host_weights, inputs.draft_model_bytes);
    }
    std::uint64_t host_expert_total = placed.host_expert_total;
    std::uint64_t host_repacked_expert = placed.host_repacked_expert;
    std::uint64_t host_repacked_dense = placed.host_repacked_dense;
    if (fit_path) {
        // llama-server's --fit picks the offload itself: whatever does not fit
        // the VRAM budget (plus its margin) lands on the host, experts first.
        e.host_fit_applies = true;
        std::uint64_t overflow = 0;
        if (inputs.fit_vram_budget_bytes) {
            const std::uint64_t budget = *inputs.fit_vram_budget_bytes;
            const std::uint64_t need = saturating_add(e.total_bytes, kLlamaServerFitTargetBytes);
            if (need > budget) {
                std::uint64_t granule = 0;
                for (const std::uint64_t b : m.block_bytes) granule = std::max(granule, b);
                const std::uint64_t movable =
                    saturating_add(e.weights_bytes, saturating_add(e.kv_bytes, e.recurrent_bytes));
                overflow = std::min(movable, saturating_add(need - budget, granule));
            }
            const std::uint64_t overflow_expert =
                placed.tensor_table_usable
                    ? std::min(overflow, placed.all_expert_total - std::min(placed.all_expert_total, host_expert_total))
                    : 0;
            const std::uint64_t overflow_dense = overflow - overflow_expert;
            host_weights = saturating_add(host_weights, overflow);
            host_expert_total = saturating_add(host_expert_total, overflow_expert);
            if (placement.repack) {
                host_repacked_expert = saturating_add(
                    host_repacked_expert,
                    std::min(overflow_expert,
                             placed.all_repack_expert - std::min(placed.all_repack_expert, host_repacked_expert)));
                host_repacked_dense = saturating_add(
                    host_repacked_dense,
                    placed.tensor_table_usable
                        ? std::min(overflow_dense,
                                   placed.all_repack_dense - std::min(placed.all_repack_dense, host_repacked_dense))
                        : overflow_dense);
            }
            e.host_fit_overflow_bytes = overflow;
            e.notes.emplace_back("host RAM: llama-server --fit sizes the offload; " + std::to_string(overflow / kMiB) +
                                 " MiB that does not fit the " + std::to_string(budget / kMiB) +
                                 " MiB VRAM budget (plus 1024 MiB fit margin) is counted on the host; the budget is "
                                 "not free VRAM, so less may fit");
        } else {
            // No VRAM note: `notes` are VRAM caveats that callers fold into the
            // VRAM fit summary. The flag carries this and the supervisor warns
            // (host_memory_estimate_lower_bound).
            e.host_estimate_lower_bound = true;
        }
    }
    // Experts the table could not classify (no tensor names) are dense here:
    // the whole host weight is then counted as needed.
    e.host_expert_bytes = std::min(host_expert_total, host_weights);
    e.host_dense_weight_bytes = host_weights - e.host_expert_bytes;
    // Repacked tensors are copied into private memory whole (not mmapped).
    host_repacked_expert = std::min(host_repacked_expert, e.host_expert_bytes);
    host_repacked_dense = std::min(host_repacked_dense, e.host_dense_weight_bytes);
    e.host_repacked_bytes = saturating_add(host_repacked_expert, host_repacked_dense);

    // KV and recurrent state of non-offloaded blocks (the offloaded ones are
    // in kv_bytes/recurrent_bytes above). --no-kv-offload is not modelled.
    double host_kv_per_token = 0.0;
    double host_swa_kv_per_token = 0.0;
    std::uint64_t host_recurrent = 0;
    for (std::uint32_t i = 0; i < first_gpu && i < m.kv_heads.size(); ++i) {
        if (i >= first_nextn && !mtp) continue;
        if (m.kv_heads[i] > 0) {
            const bool swa = sliding(i);
            const double heads = static_cast<double>(m.kv_heads[i]);
            const double k = static_cast<double>(swa ? m.key_length_swa : m.key_length);
            const double v = static_cast<double>(swa ? m.value_length_swa : m.value_length);
            const double bytes = v_padded ? heads * k * bk + v_row_max * bv : heads * (k * bk + v * bv);
            (swa ? host_swa_kv_per_token : host_kv_per_token) += bytes;
        } else if (m.kind != ModelArchitecture::attention_only) {
            host_recurrent = saturating_add(host_recurrent, saturating_mul(m.recurrent_state_values * 4, recurrent_rows));
        }
    }
    double host_kv = host_kv_per_token * static_cast<double>(ctx_per_sequence) * e.sequences;
    if (host_swa_kv_per_token > 0.0) {
        std::uint64_t cells = e.swa_context_per_sequence;
        if (e.swa_attention_layers == 0) {
            bool swa_full_flag = false;
            for (const auto& arg : p.extra_args) {
                swa_full_flag = swa_full_flag || llamaserver_flag_name(arg) == "--swa-full";
            }
            if (p.backend == kLaunchProfileBackendLlamaCpp || swa_full_flag || inputs.inherited_swa_full) {
                cells = ctx_per_sequence;
            } else {
                const std::uint64_t window =
                    saturating_add(saturating_mul(m.sliding_window, unified ? slots : 1u), ubatch);
                cells = (std::min(ctx_per_sequence, window) + 255) / 256 * 256;
            }
        }
        host_kv += host_swa_kv_per_token * static_cast<double>(cells) * e.sequences;
    }
    e.host_kv_bytes = saturating_u64(host_kv);
    e.host_recurrent_bytes = host_recurrent;
    e.host_compute_bytes = saturating_add(kHostWorkspaceBytes, saturating_mul(m.vocab_size, saturating_mul(ubatch, 4)));
    if (offloaded == 0) {
        // Nothing is offloaded: the compute graph's activations are host buffers too.
        e.host_compute_bytes =
            saturating_add(e.host_compute_bytes, saturating_mul(saturating_mul(4 * m.embedding_length, ubatch), 4));
    }

    // Weights are page-cache-backed under mmap (clean file pages the kernel
    // may drop), private memory under --load-mode none|mlock|dio, and pinned (not droppable) with mlock.
    e.host_weights_mmapped = placement.mmap && !placement.mlock;
    double fraction = 1.0;
    if (e.host_weights_mmapped) {
        if (inputs.expert_resident_fraction > 0.0) {  // false for NaN and negatives
            fraction = std::min(1.0, inputs.expert_resident_fraction);
        } else if (m.routed_experts && m.routed_experts->total > 0) {
            fraction = static_cast<double>(m.routed_experts->active) / static_cast<double>(m.routed_experts->total);
        }
    }
    e.host_expert_resident_fraction = fraction;
    // Repacked tensors are private memory at 100%; only the rest is page cache.
    const std::uint64_t expert_mapped = e.host_expert_bytes - host_repacked_expert;
    const std::uint64_t dense_mapped = e.host_dense_weight_bytes - host_repacked_dense;
    const std::uint64_t expert_resident =
        fraction >= 1.0 ? expert_mapped
                        : std::min(expert_mapped, saturating_u64(std::ceil(static_cast<double>(expert_mapped) * fraction)));
    const std::uint64_t anonymous =
        saturating_add(saturating_add(e.host_kv_bytes, e.host_recurrent_bytes), e.host_compute_bytes);
    if (e.host_weights_mmapped) {
        e.host_page_cache_bytes = saturating_add(dense_mapped, expert_resident);
        e.host_private_bytes = saturating_add(anonymous, e.host_repacked_bytes);
    } else {
        e.host_page_cache_bytes = 0;
        e.host_private_bytes =
            saturating_add(anonymous, saturating_add(e.host_dense_weight_bytes, e.host_expert_bytes));
    }
    e.host_resident_bytes = saturating_add(e.host_private_bytes, e.host_page_cache_bytes);
    if (placement.has_tensor_rules() || e.host_expert_bytes > 0) {
        e.notes.emplace_back("host RAM: mmapped weights are counted as page cache (experts at the stated resident "
                             "fraction), not as private memory, except host Q4_0/Q4_K/Q8_0/IQ4_NL tensors, "
                             "which llama.cpp repacks into private memory unless -nr/--no-repack; "
                             "--no-kv-offload is not modelled");
    }
    return e;
}

std::optional<std::uint64_t> detect_device_vram_mib(std::string_view device_hint, bool prefer_free) {
    return select_gpu_vram_budget_mib(enumerate_gpu_devices(), device_hint, prefer_free);
}

std::optional<std::uint64_t> detect_device_vram_mib() {
    return detect_device_vram_mib(std::string_view{}, false);
}

}  // namespace sonder::inference
