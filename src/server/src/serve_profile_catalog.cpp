// Launch-profile catalog wiring and VRAM/host-RAM fit check for `serve`
// (declared in sonder/inference/server.hpp).
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "sonder/inference/json.hpp"
#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server {

namespace {

std::uint64_t file_bytes(const std::string& path) {
    if (path.empty()) return 0;
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    return ec ? 0 : static_cast<std::uint64_t>(size);
}

std::string mib_text(std::uint64_t bytes) { return std::to_string((bytes + (1u << 20) - 1) >> 20); }

}  // namespace

// Builds the `sonder.profiles` catalog, binds the active profile into `o`, and
// runs the VRAM fit check for it. On refusal, `refusal` holds the message.
// For an active llamaserver profile it also feeds the host-RAM estimate into
// o.backend.llamaserver_host_memory, whether or not the profile offloads.
json::Array profile_catalog(const std::vector<LaunchProfile>& profiles, const LaunchProfile& active,
                            ServerOptions& o, std::optional<std::uint64_t> budget_flag, bool allow_overcommit,
                            std::vector<std::string>& notes, std::string& refusal) {
    std::optional<std::uint64_t> detected;
    bool detection_done = false;
    // The operator's --vram-budget-mib wins over a (possibly shared) profile's
    // vram_budget_mib: a budget tightened on the command line always applies.
    const auto budget_for = [&](const LaunchProfile& p) -> std::optional<std::uint64_t> {
        if (budget_flag) return budget_flag;
        if (p.vram_budget_mib) return p.vram_budget_mib;
        if (!detection_done) {
            detected = detect_device_vram_mib();
            detection_done = true;
        }
        if (detected && *detected > kDefaultVramReserveMib) return *detected - kDefaultVramReserveMib;
        return std::nullopt;
    };
    // llamaserver offloads by default (--n-gpu-layers auto); the llamacpp
    // backend only on a gpu:* device.
    const auto offloads = [&](const LaunchProfile& p) {
        if (p.n_gpu_layers && *p.n_gpu_layers == 0) return false;
        if (p.backend == kLaunchProfileBackendLlamaCpp) return &p == &active && o.device.rfind("gpu:", 0) == 0;
        return true;
    };
    json::Array catalog;
    // llama-server inherits this setting. Treat any nonempty value as full
    // cache conservatively instead of guessing an external binary's boolean
    // parser; an explicitly false value may overestimate, never undercount.
    const char* inherited_swa_full = std::getenv("LLAMA_ARG_SWA_FULL");
    const bool swa_full = inherited_swa_full != nullptr && *inherited_swa_full != '\0';
    for (const auto& p : profiles) {
        const bool is_active = &p == &active;
        std::optional<VramEstimate> estimate;
        std::string estimate_error;         // for the operator's log and refusal (includes the path)
        std::string public_estimate_error;  // for /v1/models (path redacted)
        std::optional<std::uint64_t> budget;
        // The active llamaserver profile needs the estimate even with no GPU
        // offload: its host-RAM footprint is the largest then.
        const bool wants_host_estimate = is_active && p.backend == kLaunchProfileBackendLlamaServer;
        if (offloads(p) || wants_host_estimate) {
            if (offloads(p)) budget = budget_for(p);
            auto info = read_gguf_model_info(p.model);  // header only
            if (info.ok()) {
                VramEstimateInputs in{};
                in.mmproj_bytes = file_bytes(p.mmproj);
                in.draft_model_bytes = p.speculative ? file_bytes(p.speculative->draft_model) : 0;
                in.inherited_swa_full = swa_full;
                if (budget) {
                    in.fit_vram_budget_bytes =
                        *budget > (std::numeric_limits<std::uint64_t>::max() >> 20)
                            ? std::numeric_limits<std::uint64_t>::max()
                            : *budget << 20;
                }
                estimate = estimate_vram(info.value(), p, in);
            } else {
                estimate_error = info.status().message();
                public_estimate_error = redact_model_path(info.status(), p.model);
            }
        }
        json::Object meta = launch_profile_metadata(p, offloads(p) ? estimate : std::nullopt, budget);
        if (!public_estimate_error.empty()) meta.set("estimate_error", public_estimate_error);
        meta.set("served", is_active);
        catalog.emplace_back(meta);
        if (!is_active) continue;

        ModelProfileBinding binding;
        binding.model = p.name;
        binding.load_name = p.backend == kLaunchProfileBackendLlamaCpp ? p.model : p.name;
        binding.sampling_defaults = p.sampling;
        binding.metadata = std::move(meta);
        o.profile_bindings.push_back(std::move(binding));
        if (p.backend == kLaunchProfileBackendLlamaServer && estimate) {
            // Feeds the supervisor's pre-spawn host RAM check and floor guard
            // (BackendSetup::llamaserver_host_memory; reserve, override and
            // floor come from the llamaserver config's "host_memory").
            o.backend.llamaserver_host_memory.estimated_resident_bytes = estimate->host_resident_bytes;
            o.backend.llamaserver_host_memory.estimated_private_bytes = estimate->host_private_bytes;
            o.backend.llamaserver_host_memory.estimate_lower_bound = estimate->host_estimate_lower_bound;
        }
        if (p.backend == kLaunchProfileBackendLlamaServer) {
            // The child inherits the environment (only AIP_* is removed), so
            // llama-server reads these; the estimates do not model them.
            for (const char* name : {"LLAMA_ARG_FIT", "LLAMA_ARG_N_GPU_LAYERS", "LLAMA_ARG_REPACK", "LLAMA_ARG_N_CPU_MOE",
                                     "LLAMA_ARG_CPU_MOE", "LLAMA_ARG_N_CPU_FFN", "LLAMA_ARG_OVERRIDE_TENSOR",
                                     "LLAMA_ARG_LOAD_MODE"}) {
                const char* value = std::getenv(name);
                if (value != nullptr && *value != '\0') {
                    notes.push_back("launch profile '" + p.name + "': inherited " + name +
                                    " is not modelled; the VRAM and host RAM estimates may be wrong");
                }
            }
        }
        if (!offloads(p)) {
            notes.push_back("launch profile '" + p.name + "': no GPU offload; VRAM fit check skipped");
            if (wants_host_estimate && !estimate) {
                notes.push_back("launch profile '" + p.name + "': host RAM estimate unavailable (" + estimate_error + ")");
            }
            continue;
        }
        // With --fit on (llama-server's default) and no fixed layer count,
        // llama-server sizes the offload itself; the estimate only informs.
        const bool upstream_fits =
            p.backend == kLaunchProfileBackendLlamaServer && p.fit.value_or(true) && !p.n_gpu_layers;
        if (!estimate) {
            if (!allow_overcommit && !upstream_fits) {
                refusal = "launch profile '" + p.name + "': cannot estimate VRAM (" + estimate_error +
                          "); fix the model path or pass --allow-overcommit";
                return catalog;
            }
            notes.push_back("launch profile '" + p.name + "': VRAM estimate unavailable (" + estimate_error +
                            "); the host RAM pre-spawn check is skipped");
            continue;
        }
        const VramEstimate& e = *estimate;
        std::string recurrent = mib_text(e.recurrent_bytes);
        if (e.recurrent_bytes > 0 && (e.recurrent_sequences > 1 || e.recurrent_snapshots > 1)) {
            recurrent += " = " + std::to_string(e.recurrent_sequences) + " sequence(s) x " +
                         std::to_string(e.recurrent_snapshots) + " snapshot(s)" +
                         (e.recurrent_snapshots > 1 ? " for speculative rollback" : "");
        }
        // Sliding-window layers keep their own, smaller cache: name both.
        std::string kv = mib_text(e.kv_bytes) + " = ";
        if (e.swa_attention_layers > 0) {
            kv += "(" + std::to_string(e.attention_layers - e.swa_attention_layers) + " full-attention layers x " +
                  std::to_string(e.context_per_sequence) + " tokens + " + std::to_string(e.swa_attention_layers) +
                  " sliding-window layers x " + std::to_string(e.swa_context_per_sequence) + " tokens)";
        } else {
            kv += std::to_string(e.attention_layers) + " attention layers x " + std::to_string(e.context_per_sequence) +
                  " tokens";
        }
        std::string summary = "launch profile '" + p.name + "': estimated VRAM " + std::to_string(e.total_mib()) +
                              " MiB (weights " + mib_text(e.weights_bytes) + ", KV " + kv + " x " +
                              std::to_string(e.sequences) + " sequence(s), recurrent " + recurrent + ", compute " +
                              mib_text(e.compute_bytes) + ")";
        if (budget) summary += " vs budget " + std::to_string(*budget) + " MiB";
        for (const auto& note : e.notes) summary += "; " + note;
        if (upstream_fits) {
            summary += " (llama-server --fit sizes the offload)";
        } else if (!budget) {
            if (!allow_overcommit) {
                refusal = summary + "; cannot detect the device's VRAM: set vram_budget_mib in the profile or "
                                    "--vram-budget-mib, or pass --allow-overcommit";
                return catalog;
            }
        } else if (e.total_mib() > *budget) {
            if (!allow_overcommit) {
                refusal = summary + ": does not fit; lower ctx_size or n_gpu_layers, use a smaller KV cache type, "
                                    "raise the budget, or pass --allow-overcommit";
                return catalog;
            }
            summary += " (over budget; --allow-overcommit)";
        }
        notes.push_back(summary);
    }
    return catalog;
}

}  // namespace sonder::inference::server
