// MoE tensor placement flag parsing (-ot/--override-tensor, --cpu-moe,
// --n-cpu-moe, --moe-cache-mib, --moe-cache-cpu-share, --load-mode,
// --repack/--no-repack), per-tensor placement evaluation against GGUF tensor
// tables, and dynamic SLRU expert cache sizing.
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <locale>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/launch_profile.hpp"
#include "sonder/inference/moe_routing_telemetry.hpp"
#include "vram_internal.hpp"

namespace sonder::inference {

namespace {

constexpr std::size_t kMaxPlacementRules = 256;
constexpr std::size_t kMaxOverridePattern = 256;
constexpr std::size_t kMaxOverrideValue = 4096;
constexpr std::size_t kMaxBufferTypeName = 64;
constexpr std::uint64_t kMaxCpuMoeLayers = 1u << 22;
constexpr std::uint64_t kMaxMoeCacheMib = 262144u;
constexpr std::uint64_t kMinMoeCacheMib = 512u;
constexpr std::uint64_t kMoeCacheAlignMib = 64u;

bool parse_plain_decimal_u64(std::string_view value, std::uint64_t max_value, std::uint64_t& out) noexcept {
    if (value.empty() || value.size() > 7 || (value.size() > 1 && value[0] == '0')) return false;
    std::uint64_t n = 0;
    for (const char c : value) {
        if (c < '0' || c > '9') return false;
        n = n * 10 + static_cast<std::uint64_t>(c - '0');
    }
    if (n > max_value) return false;
    out = n;
    return true;
}

bool parse_unit_interval_double(std::string_view value, double& out) {
    if (value.empty() || value.size() > 32) return false;
    for (const char c : value) {
        const bool ok = (c >= '0' && c <= '9') || c == '.';
        if (!ok) return false;
    }
    std::istringstream in{std::string(value)};
    in.imbue(std::locale::classic());
    double parsed = 0.0;
    in >> parsed;
    if (!in || !in.eof() || !std::isfinite(parsed) || parsed < 0.0 || parsed > 1.0) return false;
    out = parsed;
    return true;
}

std::string format_compact_double(double value, int precision = 2) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(precision) << value;
    std::string s = out.str();
    if (s.find('.') != std::string::npos) {
        while (!s.empty() && s.back() == '0') s.pop_back();
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    return s;
}

bool host_buffer_type(std::string_view name) {
    std::string up(name);
    for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return up == "CPU" || up.rfind("CPU_", 0) == 0 || (up.size() >= 5 && up.compare(up.size() - 5, 5, "_HOST") == 0);
}

Status placement_error(const std::string& flag, const std::string& what, std::string_view value = {}) {
    std::string text = "extra_args " + detail::excerpt(flag) + ": " + what;
    if (!value.empty()) text += " (" + detail::excerpt(value) + ")";
    return Status(ErrorCode::invalid_argument, std::move(text));
}

}  // namespace

Result<MoePlacement> parse_moe_placement(const std::vector<std::string>& extra_args) {
    MoePlacement out;
    for (std::size_t i = 0; i < extra_args.size(); ++i) {
        const std::string& arg = extra_args[i];
        if (arg.size() < 2 || arg[0] != '-') continue;
        const std::string flag = llamaserver_flag_name(arg);
        const bool has_equals = arg.find('=') != std::string::npos;
        if (flag == "-lm" || flag == "--load-mode") {
            // b11195 common/arg.cpp: -lm/--load-mode {auto,none,mmap,mlock,mmap+mlock,dio}.
            // There is no --mmap/--no-mmap/--mlock in that build. The last one wins.
            std::string value;
            if (has_equals) {
                value = arg.substr(arg.find('=') + 1);
            } else if (i + 1 < extra_args.size()) {
                value = extra_args[++i];
            } else {
                return placement_error(flag, "needs a value");
            }
            if (value == "auto" || value == "mmap") {
                out.mmap = true;
                out.mlock = false;
            } else if (value == "none" || value == "dio") {
                out.mmap = false;  // private memory (dio bypasses the page cache)
                out.mlock = false;
            } else if (value == "mlock") {
                out.mmap = false;  // read into private memory and pinned
                out.mlock = true;
            } else if (value == "mmap+mlock") {
                out.mmap = true;  // mapped but pinned: not droppable
                out.mlock = true;
            } else {
                return placement_error(flag, "must be one of auto, none, mmap, mlock, mmap+mlock, dio", value);
            }
            continue;
        }
        if (flag == "-nr" || flag == "--no-repack") {
            if (has_equals) return placement_error(flag, "takes no value");
            out.repack = false;
            continue;
        }
        if (flag == "--repack") {
            if (has_equals) return placement_error(flag, "takes no value");
            out.repack = true;
            continue;
        }
        if (flag == "--cpu-moe" || flag == "-cmoe") {
            if (has_equals) return placement_error(flag, "takes no value");
            if (out.rules.size() >= kMaxPlacementRules) return placement_error(flag, "too many placement rules");
            TensorPlacementRule rule;
            rule.kind = TensorPlacementRule::Kind::all_experts;
            rule.host = true;
            out.rules.push_back(std::move(rule));
            continue;
        }
        if (flag == "--moe-cache-mib") {
            std::string value;
            if (has_equals) {
                value = arg.substr(arg.find('=') + 1);
            } else if (i + 1 < extra_args.size()) {
                value = extra_args[++i];
            } else {
                return placement_error(flag, "needs a value");
            }
            std::uint64_t mib = 0;
            if (!parse_plain_decimal_u64(value, kMaxMoeCacheMib, mib)) {
                return placement_error(flag, "must be a plain decimal MiB budget in [0, 262144]", value);
            }
            out.moe_cache_mib = static_cast<std::uint32_t>(mib);
            continue;
        }
        if (flag == "--moe-cache-cpu-share") {
            std::string value;
            if (has_equals) {
                value = arg.substr(arg.find('=') + 1);
            } else if (i + 1 < extra_args.size()) {
                value = extra_args[++i];
            } else {
                return placement_error(flag, "needs a value");
            }
            double share = 0.0;
            if (!parse_unit_interval_double(value, share)) {
                return placement_error(flag, "must be a finite number in [0.0, 1.0]", value);
            }
            out.moe_cache_cpu_share = share;
            continue;
        }
        const bool is_override = flag == "-ot" || flag == "--override-tensor";
        const bool is_ffn = flag == "--n-cpu-ffn" || flag == "-ncffn";
        const bool is_count = flag == "--n-cpu-moe" || flag == "-ncmoe" || is_ffn;
        if (!is_override && !is_count) continue;
        std::string value;
        if (has_equals) {
            value = arg.substr(arg.find('=') + 1);
        } else if (i + 1 < extra_args.size()) {
            value = extra_args[++i];
        } else {
            return placement_error(flag, "needs a value");
        }
        if (value.empty()) return placement_error(flag, "needs a value");
        if (is_count) {
            if (value.size() > 7 || (value.size() > 1 && value[0] == '0')) {
                return placement_error(flag, "must be a plain decimal layer count", value);
            }
            std::uint64_t n = 0;
            for (const char c : value) {
                if (c < '0' || c > '9') return placement_error(flag, "must be a plain decimal layer count", value);
                n = n * 10 + static_cast<std::uint64_t>(c - '0');
            }
            if (n > kMaxCpuMoeLayers) return placement_error(flag, "layer count is implausibly large", value);
            if (is_ffn) {
                out.n_cpu_ffn = static_cast<std::uint32_t>(n);
                continue;
            }
            if (out.rules.size() >= kMaxPlacementRules) return placement_error(flag, "too many placement rules");
            TensorPlacementRule rule;
            rule.kind = TensorPlacementRule::Kind::first_n_experts;
            rule.count = static_cast<std::uint32_t>(n);
            rule.host = true;
            out.rules.push_back(std::move(rule));
            continue;
        }
        if (value.size() > kMaxOverrideValue) return placement_error(flag, "value is too long");
        std::size_t start = 0;
        for (;;) {
            const std::size_t comma = value.find(',', start);
            const std::string item =
                value.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
            const std::size_t eq = item.find('=');
            if (eq == std::string::npos || eq == 0) {
                return placement_error(flag, "each override must be <pattern>=<buffer type>", item);
            }
            TensorPlacementRule rule;
            rule.pattern = item.substr(0, eq);
            rule.buffer_type = item.substr(eq + 1);
            if (rule.pattern.size() > kMaxOverridePattern) return placement_error(flag, "pattern is too long");
            if (rule.buffer_type.empty() || rule.buffer_type.size() > kMaxBufferTypeName ||
                !std::all_of(rule.buffer_type.begin(), rule.buffer_type.end(), [](char c) {
                    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_';
                })) {
                return placement_error(flag, "buffer type must be 1-64 characters from [A-Za-z0-9_]", item);
            }
            try {
                const std::regex check(rule.pattern, std::regex::ECMAScript | std::regex::nosubs);
                (void)check;
            } catch (...) {
                return placement_error(flag, "pattern is not a valid regular expression", item);
            }
            rule.host = host_buffer_type(rule.buffer_type);
            if (out.rules.size() >= kMaxPlacementRules) return placement_error(flag, "too many placement rules");
            out.rules.push_back(std::move(rule));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
    }
    return out;
}

namespace detail {

TensorPlacementBreakdown evaluate_tensor_placement(const GgufModelInfo& m, MoePlacement& placement,
                                                   std::uint32_t n_layer, std::uint32_t first_gpu,
                                                   std::uint32_t first_nextn, bool mtp,
                                                   std::vector<std::string>& notes) {
    TensorPlacementBreakdown out;
    // llama-server --fit sizes the offload when the profile leaves n_gpu_layers
    // unset (b11195 common.h fit_params = true); any user tensor override makes
    // upstream fit abort (fit.cpp), and `--n-cpu-moe 0` adds no override. A
    // placement parse error leaves user_overrides false: the conservative side.
    out.user_overrides =
        placement.n_cpu_ffn > 0 ||
        std::any_of(placement.rules.begin(), placement.rules.end(), [](const TensorPlacementRule& r) {
            return !(r.kind == TensorPlacementRule::Kind::first_n_experts && r.count == 0);
        });
    out.tensor_table_usable = !m.tensors.empty() && !m.tensors_truncated;
    out.moved_to_host.assign(n_layer, 0);
    out.moved_to_gpu.assign(n_layer, 0);
    if (placement.has_tensor_rules() && !out.tensor_table_usable) {
        notes.emplace_back(m.tensors_truncated
                               ? "the GGUF tensor table was too large to keep names for: -ot/--n-cpu-moe/--cpu-moe "
                                 "are not applied"
                               : "no tensor names are available: -ot/--n-cpu-moe/--cpu-moe are not applied");
        placement.rules.clear();
    }
    if (!out.tensor_table_usable) {
        return out;
    }
    bool regex_failed = false;
    std::vector<std::optional<std::regex>> compiled(placement.rules.size());
    for (std::size_t r = 0; r < placement.rules.size(); ++r) {
        if (placement.rules[r].kind != TensorPlacementRule::Kind::pattern) continue;
        try {
            compiled[r].emplace(placement.rules[r].pattern, std::regex::ECMAScript | std::regex::nosubs);
        } catch (...) {
            // parse_moe_placement already compiled it; stay conservative
            // if the standard library disagrees: the rule never matches.
        }
    }
    std::uint64_t matched_experts = 0;
    for (const GgufTensorInfo& t : m.tensors) {
        if (t.block == kNoBlock || t.block >= n_layer) continue;
        if (t.block >= first_nextn && !mtp) continue;
        const bool default_host = t.block < first_gpu;
        bool host = default_host;
        for (std::size_t r = 0; r < placement.rules.size(); ++r) {
            const TensorPlacementRule& rule = placement.rules[r];
            bool hit = false;
            switch (rule.kind) {
                case TensorPlacementRule::Kind::all_experts: hit = t.expert; break;
                case TensorPlacementRule::Kind::first_n_experts: hit = t.expert && t.block < rule.count; break;
                case TensorPlacementRule::Kind::pattern:
                    if (compiled[r].has_value()) {
                        try {
                            hit = std::regex_search(t.name, *compiled[r]);
                        } catch (...) {
                            // e.g. MSVC's regex_error(error_complexity); the rule does not match.
                            hit = false;
                            regex_failed = true;
                        }
                    }
                    break;
            }
            if (!hit) continue;
            host = rule.host;
            break;
        }
        if (t.expert) out.all_expert_total = saturating_add(out.all_expert_total, t.bytes);
        if (placement.repack && repackable_ggml_type(t.type)) {
            (t.expert ? out.all_repack_expert : out.all_repack_dense) =
                saturating_add(t.expert ? out.all_repack_expert : out.all_repack_dense, t.bytes);
        }
        if (host != default_host) {
            auto& bucket = host ? out.moved_to_host : out.moved_to_gpu;
            bucket[t.block] = saturating_add(bucket[t.block], t.bytes);
            if (t.expert && host) matched_experts = saturating_add(matched_experts, t.bytes);
        }
        if (host && t.expert) out.host_expert_total = saturating_add(out.host_expert_total, t.bytes);
        if (host && placement.repack && repackable_ggml_type(t.type)) {
            (t.expert ? out.host_repacked_expert : out.host_repacked_dense) =
                saturating_add(t.expert ? out.host_repacked_expert : out.host_repacked_dense, t.bytes);
        }
    }
    if (regex_failed) {
        notes.emplace_back("an -ot pattern could not be evaluated against a tensor name; that rule is treated as "
                           "not matching");
    }
    if (placement.has_tensor_rules()) {
        if (matched_experts == 0 &&
            std::none_of(out.moved_to_host.begin(), out.moved_to_host.end(), [](std::uint64_t v) { return v != 0; }) &&
            std::none_of(out.moved_to_gpu.begin(), out.moved_to_gpu.end(), [](std::uint64_t v) { return v != 0; })) {
            notes.emplace_back("-ot/--n-cpu-moe/--cpu-moe matched no tensor that changes placement");
        }
    }
    return out;
}

MoeExpertCacheSizing plan_moe_expert_cache(const GgufModelInfo& model, std::uint64_t host_expert_bytes,
                                           std::uint64_t spare_vram_mib,
                                           std::optional<std::uint64_t> available_host_ram_bytes) {
    MoeExpertCacheSizing out;
    out.spare_vram_mib = spare_vram_mib;

    // Anti-regression rule (a) (llama.cpp PR #29887 / #30966 / #30637):
    // NEVER enable dynamic SLRU MoE caching when all experts fit in VRAM
    // (avoids ~10% graph-splitting overhead).
    if (host_expert_bytes == 0) {
        out.reason =
            "MoE model fits 100% in VRAM; left --moe-cache-mib disabled to avoid ~10% graph-splitting overhead";
        return out;
    }

    out.offloaded_expert_mib = (host_expert_bytes >> 20) + ((host_expert_bytes & (kMiB - 1)) != 0 ? 1ull : 0ull);
    double active_ratio = 1.0;
    if (model.routed_experts && model.routed_experts->total > 0 && model.routed_experts->active > 0) {
        active_ratio = std::min(1.0, static_cast<double>(model.routed_experts->active) /
                                         static_cast<double>(model.routed_experts->total));
    }
    out.active_working_set_mib = std::max<std::uint64_t>(
        1ull, saturating_u64(std::ceil(static_cast<double>(out.offloaded_expert_mib) * active_ratio)));

    // Anti-regression rule (b): require >= 512 MiB spare VRAM headroom and a
    // cache large enough to cover the active expert working set so hit rate
    // stays high enough to avoid <75% hit-rate PCIe thrash.
    if (spare_vram_mib < kMinMoeCacheMib) {
        out.reason = "skipped dynamic SLRU MoE cache (--moe-cache-mib): only " + std::to_string(spare_vram_mib) +
                     " MiB spare VRAM remaining (< 512 MiB floor)";
        return out;
    }

    std::uint64_t usable_cache_mib = 0;
    if (spare_vram_mib >= 1024u) {
        usable_cache_mib = spare_vram_mib - 512u;
    } else if (spare_vram_mib >= 640u) {
        usable_cache_mib = spare_vram_mib - 128u;
    } else {
        usable_cache_mib = kMinMoeCacheMib;
    }
    usable_cache_mib = (usable_cache_mib / kMoeCacheAlignMib) * kMoeCacheAlignMib;
    usable_cache_mib = std::min(usable_cache_mib, std::min(out.offloaded_expert_mib, kMaxMoeCacheMib));

    if (usable_cache_mib < kMinMoeCacheMib) {
        out.reason = "skipped dynamic SLRU MoE cache (--moe-cache-mib): allocatable cache (" +
                     std::to_string(usable_cache_mib) + " MiB from " + std::to_string(spare_vram_mib) +
                     " MiB spare VRAM) is below the 512 MiB floor";
        return out;
    }

    if (usable_cache_mib < out.active_working_set_mib) {
        out.reason = "skipped dynamic SLRU MoE cache (--moe-cache-mib): " + std::to_string(usable_cache_mib) +
                     " MiB available cache is below the " + std::to_string(out.active_working_set_mib) +
                     " MiB active CPU-offloaded expert working set (avoids <75% hit-rate PCIe thrash)";
        return out;
    }

    const double coverage = static_cast<double>(usable_cache_mib) / static_cast<double>(out.active_working_set_mib);
    const bool ample_ddr5_host =
        available_host_ram_bytes.has_value() && *available_host_ram_bytes >= 24ull * 1024ull * kMiB;
    double cpu_share = 0.25;
    if (ample_ddr5_host) {
        if (coverage >= 4.0) {
            cpu_share = 0.25;
        } else if (coverage >= 2.0) {
            cpu_share = 0.35;
        } else {
            cpu_share = 0.50;
        }
    }

    out.eligible = true;
    out.recommended_cache_mib = static_cast<std::uint32_t>(usable_cache_mib);
    out.recommended_cpu_share = cpu_share;
    out.reason = "enabled dynamic SLRU MoE expert cache (--moe-cache-mib " + std::to_string(out.recommended_cache_mib) +
                 ", --moe-cache-cpu-share " + format_compact_double(cpu_share, 2) + ") from " +
                 std::to_string(spare_vram_mib) + " MiB spare VRAM (covers " + format_compact_double(coverage, 2) +
                 "x of " + std::to_string(out.active_working_set_mib) +
                 " MiB active CPU-offloaded expert working set" +
                 (ample_ddr5_host ? " with ample DDR5 host bandwidth" : "") + ")";
    return out;
}

}  // namespace detail

// ------------------------------------------------ Bounded MoE routing telemetry

MoeRoutingTelemetry::MoeRoutingTelemetry(MoeRoutingTelemetryConfig config)
    : offload_active_(config.offload_active),
      max_layers_(std::min(config.max_layers, MoeRoutingTelemetryConfig::kMaxTrackedLayers)),
      max_experts_per_layer_(std::min(config.max_experts_per_layer, MoeRoutingTelemetryConfig::kMaxExpertsPerLayer)),
      default_top_k_(config.top_k == 0 ? 1u : std::min(config.top_k, MoeRoutingTelemetryConfig::kMaxExpertsPerLayer)) {}

void MoeRoutingTelemetry::set_offload_active(bool active) noexcept {
    offload_active_.store(active, std::memory_order_release);
}

MoeRoutingTelemetry::LayerTable* MoeRoutingTelemetry::find_or_create_layer_locked(std::uint32_t layer_id) {
    for (auto& layer : layers_) {
        if (layer.layer_id == layer_id) {
            return &layer;
        }
    }
    if (layers_.size() >= max_layers_) {
        ++dropped_layer_observations_;
        return nullptr;
    }
    LayerTable created;
    created.layer_id = layer_id;
    layers_.push_back(std::move(created));
    return &layers_.back();
}

void MoeRoutingTelemetry::record_locked(LayerTable* layer, std::uint32_t expert_id, bool cache_hit,
                                        std::uint64_t transfer_bytes, double wait_ms) {
    const double clean_wait_ms = (std::isfinite(wait_ms) && wait_ms > 0.0) ? wait_ms : 0.0;
    ++total_routes_;
    if (cache_hit) {
        ++expert_hits_;
    } else {
        ++expert_misses_;
    }
    if (transfer_bytes > 0 || clean_wait_ms > 0.0) {
        ++transfer_count_;
        transfer_bytes_ = detail::saturating_add(transfer_bytes_, transfer_bytes);
        transfer_wait_ms_ += clean_wait_ms;
        if (clean_wait_ms > max_transfer_wait_ms_) {
            max_transfer_wait_ms_ = clean_wait_ms;
        }
    }

    if (layer == nullptr) {
        return;
    }

    ++layer->total_routes;
    if (cache_hit) {
        ++layer->hits;
    } else {
        ++layer->misses;
    }
    layer->transfer_bytes = detail::saturating_add(layer->transfer_bytes, transfer_bytes);
    layer->transfer_wait_ms += clean_wait_ms;

    if (max_experts_per_layer_ == 0) {
        return;
    }

    for (auto& slot : layer->slots) {
        if (slot.expert_id == expert_id) {
            ++slot.activations;
            if (cache_hit) {
                ++slot.hits;
            } else {
                ++slot.misses;
            }
            return;
        }
    }

    if (layer->slots.size() < max_experts_per_layer_) {
        ExpertSlot created;
        created.expert_id = expert_id;
        created.activations = 1;
        created.hits = cache_hit ? 1u : 0u;
        created.misses = cache_hit ? 0u : 1u;
        created.overestimate_error = 0;
        layer->slots.push_back(created);
        return;
    }

    // Space-Saving heavy-hitter replacement: evict the minimum-activation slot
    // (tie-broken by highest overestimate_error, then largest expert_id) so
    // memory stays strictly bounded to `max_experts_per_layer_` while true hot
    // experts remain tracked with bounded error.
    std::size_t victim = 0;
    for (std::size_t i = 1; i < layer->slots.size(); ++i) {
        const ExpertSlot& cand = layer->slots[i];
        const ExpertSlot& cur = layer->slots[victim];
        if (cand.activations < cur.activations ||
            (cand.activations == cur.activations && cand.overestimate_error > cur.overestimate_error) ||
            (cand.activations == cur.activations && cand.overestimate_error == cur.overestimate_error &&
             cand.expert_id > cur.expert_id)) {
            victim = i;
        }
    }
    ExpertSlot& replaced = layer->slots[victim];
    const std::uint64_t inherited = replaced.activations;
    replaced.expert_id = expert_id;
    replaced.overestimate_error = inherited;
    replaced.activations = detail::saturating_add(inherited, 1u);
    replaced.hits = cache_hit ? 1u : 0u;
    replaced.misses = cache_hit ? 0u : 1u;
    ++layer->table_evictions;
}

void MoeRoutingTelemetry::record_expert_access(std::uint32_t layer_id, std::uint32_t expert_id, bool cache_hit,
                                               std::uint64_t transfer_bytes, double wait_ms) {
    if (!offload_active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!offload_active()) {
        return;
    }
    LayerTable* layer = find_or_create_layer_locked(layer_id);
    record_locked(layer, expert_id, cache_hit, transfer_bytes, wait_ms);
}

void MoeRoutingTelemetry::record_routing_batch(std::uint32_t layer_id,
                                               const std::vector<MoeExpertObservation>& observations) {
    if (observations.empty() || !offload_active()) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (!offload_active()) {
        return;
    }
    LayerTable* layer = find_or_create_layer_locked(layer_id);
    for (const auto& obs : observations) {
        record_locked(layer, obs.expert_id, obs.cache_hit, obs.transfer_bytes, obs.wait_ms);
    }
    if (layer == nullptr && observations.size() > 1) {
        dropped_layer_observations_ =
            detail::saturating_add(dropped_layer_observations_, static_cast<std::uint64_t>(observations.size() - 1));
    }
}

std::vector<MoeExpertFrequency> MoeRoutingTelemetry::build_top_k_locked(const LayerTable& layer, std::size_t k) const {
    const std::size_t limit = k == 0 ? default_top_k_ : k;
    std::vector<ExpertSlot> sorted = layer.slots;
    std::sort(sorted.begin(), sorted.end(), [](const ExpertSlot& a, const ExpertSlot& b) {
        if (a.activations != b.activations) return a.activations > b.activations;
        return a.expert_id < b.expert_id;
    });
    if (sorted.size() > limit) {
        sorted.resize(limit);
    }
    std::vector<MoeExpertFrequency> out;
    out.reserve(sorted.size());
    const double denom = layer.total_routes > 0 ? static_cast<double>(layer.total_routes) : 1.0;
    for (const auto& slot : sorted) {
        MoeExpertFrequency f;
        f.expert_id = slot.expert_id;
        f.activations = slot.activations;
        f.hits = slot.hits;
        f.misses = slot.misses;
        f.overestimate_error = slot.overestimate_error;
        f.share = std::min(1.0, static_cast<double>(slot.activations) / denom);
        out.push_back(f);
    }
    return out;
}

MoeRoutingSnapshot MoeRoutingTelemetry::snapshot(std::size_t top_k_override) const {
    std::lock_guard<std::mutex> lock(mu_);
    MoeRoutingSnapshot out;
    out.offload_active = offload_active();
    out.total_routes = total_routes_;
    out.expert_hits = expert_hits_;
    out.expert_misses = expert_misses_;
    out.transfer_count = transfer_count_;
    out.transfer_bytes = transfer_bytes_;
    out.transfer_wait_ms = transfer_wait_ms_;
    out.max_transfer_wait_ms = max_transfer_wait_ms_;
    out.dropped_layer_observations = dropped_layer_observations_;
    out.tracked_layers = layers_.size();
    out.max_layers = max_layers_;
    out.max_experts_per_layer = max_experts_per_layer_;

    std::vector<const LayerTable*> ordered;
    ordered.reserve(layers_.size());
    for (const auto& l : layers_) {
        ordered.push_back(&l);
    }
    std::sort(ordered.begin(), ordered.end(),
              [](const LayerTable* a, const LayerTable* b) { return a->layer_id < b->layer_id; });

    out.layers.reserve(ordered.size());
    for (const LayerTable* l : ordered) {
        MoeLayerTelemetry lt;
        lt.layer_id = l->layer_id;
        lt.total_routes = l->total_routes;
        lt.hits = l->hits;
        lt.misses = l->misses;
        lt.transfer_bytes = l->transfer_bytes;
        lt.transfer_wait_ms = l->transfer_wait_ms;
        lt.tracked_experts = l->slots.size();
        lt.table_evictions = l->table_evictions;
        lt.top_experts = build_top_k_locked(*l, top_k_override);
        out.layers.push_back(std::move(lt));
    }
    return out;
}

std::vector<MoeExpertFrequency> MoeRoutingTelemetry::hot_experts_for_layer(std::uint32_t layer_id,
                                                                           std::size_t top_k_override) const {
    std::lock_guard<std::mutex> lock(mu_);
    for (const auto& l : layers_) {
        if (l.layer_id == layer_id) {
            return build_top_k_locked(l, top_k_override);
        }
    }
    return {};
}

void MoeRoutingTelemetry::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    total_routes_ = 0;
    expert_hits_ = 0;
    expert_misses_ = 0;
    transfer_count_ = 0;
    transfer_bytes_ = 0;
    transfer_wait_ms_ = 0.0;
    max_transfer_wait_ms_ = 0.0;
    dropped_layer_observations_ = 0;
    layers_.clear();
}

MoeRoutingTelemetryConfig make_moe_routing_telemetry_config(const VramEstimate& estimate, std::size_t top_k,
                                                            std::size_t max_layers,
                                                            std::size_t max_experts_per_layer) noexcept {
    MoeRoutingTelemetryConfig cfg;
    cfg.offload_active = estimate.host_expert_bytes > 0;
    cfg.max_layers = std::min(max_layers, MoeRoutingTelemetryConfig::kMaxTrackedLayers);
    cfg.max_experts_per_layer = std::min(max_experts_per_layer, MoeRoutingTelemetryConfig::kMaxExpertsPerLayer);
    cfg.top_k = top_k == 0 ? 1u : std::min(top_k, MoeRoutingTelemetryConfig::kMaxExpertsPerLayer);
    return cfg;
}

json::Object moe_routing_telemetry_to_json(const MoeRoutingSnapshot& snapshot) {
    json::Array layers_arr;
    layers_arr.reserve(snapshot.layers.size());
    for (const auto& l : snapshot.layers) {
        json::Array experts_arr;
        experts_arr.reserve(l.top_experts.size());
        for (const auto& e : l.top_experts) {
            experts_arr.emplace_back(json::Object{{"expert_id", static_cast<std::uint64_t>(e.expert_id)},
                                                  {"activations", e.activations},
                                                  {"hits", e.hits},
                                                  {"misses", e.misses},
                                                  {"overestimate_error", e.overestimate_error},
                                                  {"share", e.share}});
        }
        layers_arr.emplace_back(json::Object{{"layer_id", static_cast<std::uint64_t>(l.layer_id)},
                                             {"total_routes", l.total_routes},
                                             {"hits", l.hits},
                                             {"misses", l.misses},
                                             {"transfer_bytes", l.transfer_bytes},
                                             {"transfer_wait_ms", l.transfer_wait_ms},
                                             {"tracked_experts", static_cast<std::uint64_t>(l.tracked_experts)},
                                             {"table_evictions", l.table_evictions},
                                             {"top_experts", std::move(experts_arr)}});
    }

    return json::Object{{"schema", "sonder.inference.moe_routing_telemetry/1"},
                        {"offload_active", snapshot.offload_active},
                        {"total_routes", snapshot.total_routes},
                        {"expert_hits", snapshot.expert_hits},
                        {"expert_misses", snapshot.expert_misses},
                        {"hit_rate", snapshot.hit_rate()},
                        {"transfer_count", snapshot.transfer_count},
                        {"transfer_bytes", snapshot.transfer_bytes},
                        {"transfer_wait_ms", snapshot.transfer_wait_ms},
                        {"max_transfer_wait_ms", snapshot.max_transfer_wait_ms},
                        {"dropped_layer_observations", snapshot.dropped_layer_observations},
                        {"tracked_layers", static_cast<std::uint64_t>(snapshot.tracked_layers)},
                        {"max_layers", static_cast<std::uint64_t>(snapshot.max_layers)},
                        {"max_experts_per_layer", static_cast<std::uint64_t>(snapshot.max_experts_per_layer)},
                        {"layers", std::move(layers_arr)}};
}

}  // namespace sonder::inference
