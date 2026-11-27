#include "log_diagnostics.hpp"

#include <cctype>
#include <limits>
#include <utility>

namespace sonder::inference::llamaserver {
namespace {

std::optional<std::uint64_t> parse_uint(std::string_view text) {
    if (text.empty() || text.size() > 20)
        return std::nullopt;
    std::uint64_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9')
            return std::nullopt;
        const auto digit = static_cast<std::uint64_t>(c - '0');
        if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10)
            return std::nullopt;
        value = value * 10 + digit;
    }
    return value;
}

std::optional<std::uint64_t> last_log_verbosity(const std::vector<std::string> &args) {
    std::optional<std::uint64_t> result;
    for (std::size_t i = 0; i < args.size(); ++i) {
        std::string_view value;
        const auto arg = std::string_view(args[i]);
        if (arg == "-lv" || arg == "--log-verbosity") {
            if (i + 1 >= args.size()) {
                result.reset();
                continue;
            }
            value = args[++i];
        } else if (arg.rfind("--log-verbosity=", 0) == 0) {
            value = arg.substr(std::string_view("--log-verbosity=").size());
        } else if (arg.rfind("-lv=", 0) == 0) {
            value = arg.substr(4);
        } else {
            continue;
        }
        result = parse_uint(value);
    }
    return result;
}

char ascii_lower_char(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

bool is_ascii_alpha(char c) noexcept {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

bool icontains(std::string_view haystack, std::string_view needle) noexcept {
    if (needle.empty())
        return true;
    if (needle.size() > haystack.size())
        return false;
    for (std::size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            if (ascii_lower_char(haystack[i + j]) != ascii_lower_char(needle[j])) {
                match = false;
                break;
            }
        }
        if (match)
            return true;
    }
    return false;
}

// Case-insensitive match bounded by non-letter characters on both sides
// (matches "HIP", "HIP0", "ggml_hip", "HIP:" without matching "chip" or "ship").
bool icontains_alpha_token(std::string_view haystack, std::string_view needle) noexcept {
    if (needle.empty() || needle.size() > haystack.size())
        return false;
    for (std::size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
        if (i > 0 && is_ascii_alpha(haystack[i - 1]))
            continue;
        const std::size_t end = i + needle.size();
        if (end < haystack.size() && is_ascii_alpha(haystack[end]))
            continue;
        bool match = true;
        for (std::size_t j = 0; j < needle.size(); ++j) {
            if (ascii_lower_char(haystack[i + j]) != ascii_lower_char(needle[j])) {
                match = false;
                break;
            }
        }
        if (match)
            return true;
    }
    return false;
}

// Token of [A-Za-z0-9_] characters starting at `pos`.
std::string_view type_token(std::string_view text, std::size_t pos) {
    std::size_t end = pos;
    while (end < text.size() && (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_'))
        ++end;
    return text.substr(pos, end - pos);
}

// Tensor name of [A-Za-z0-9_.] characters starting at `pos`.
std::string_view name_token(std::string_view text, std::size_t pos) {
    std::size_t end = pos;
    while (end < text.size() &&
           (std::isalnum(static_cast<unsigned char>(text[end])) || text[end] == '_' || text[end] == '.'))
        ++end;
    return text.substr(pos, end - pos);
}

} // namespace

void LogDiagnostics::reset() {
    partial_.clear();
    discarding_ = false;
    lines_ = 0;
    warnings_.clear();
    keys_.clear();
    mtp_bytes_ = 0;
    offload_.reset();
    if (ready_ && diagnostics_blind_) {
        add(BackendWarning{"diagnostics_blind", "warning", "config",
                           "llama-server log verbosity is below 4; offload diagnostics cannot be observed", {}, 1},
            "diagnostics_blind");
    }
}

void LogDiagnostics::ready(const std::vector<std::string> &args) {
    ready_ = true;
    const auto verbosity = last_log_verbosity(args);
    diagnostics_blind_ = !verbosity || *verbosity < 4;
    offload_.reset();
    // Readiness is a new child boundary. Keep the parser's existing warning
    // accounting useful to callers while replacing only this readiness fact.
    for (std::size_t i = warnings_.size(); i > 0; --i) {
        if (warnings_[i - 1].code == "diagnostics_blind") {
            warnings_.erase(warnings_.begin() + static_cast<std::ptrdiff_t>(i - 1));
            keys_.erase(keys_.begin() + static_cast<std::ptrdiff_t>(i - 1));
        }
    }
    if (diagnostics_blind_) {
        add(BackendWarning{"diagnostics_blind", "warning", "config",
                           "llama-server log verbosity is below 4; offload diagnostics cannot be observed", {}, 1},
            "diagnostics_blind");
    }
}

std::string LogDiagnostics::offload_status() const {
    if (diagnostics_blind_)
        return "blind";
    if (!offload_)
        return "pending";
    return offload_->first < offload_->second ? "partial" : "ok";
}

void LogDiagnostics::feed(std::string_view bytes) {
    while (!bytes.empty()) {
        const auto newline = bytes.find('\n');
        const auto part = bytes.substr(0, newline);
        if (!discarding_) {
            if (partial_.size() + part.size() > kMaxLineBytes) {
                // Keep the head of an oversized line (the diagnostic text is
                // near the start) and drop the rest up to the newline.
                partial_.append(part.substr(0, kMaxLineBytes - partial_.size()));
                discarding_ = true;
            } else {
                partial_.append(part);
            }
        }
        if (newline == std::string_view::npos)
            return;
        bytes.remove_prefix(newline + 1);
        if (!partial_.empty() && partial_.back() == '\r')
            partial_.pop_back();
        line(partial_);
        partial_.clear();
        discarding_ = false;
    }
}

void LogDiagnostics::finish() {
    if (!partial_.empty()) {
        if (partial_.back() == '\r')
            partial_.pop_back();
        line(partial_);
    }
    partial_.clear();
    discarding_ = false;
}

BackendWarning *LogDiagnostics::find(std::string_view code, std::string_view key) {
    for (std::size_t i = 0; i < warnings_.size(); ++i) {
        if (warnings_[i].code == code && keys_[i] == key)
            return &warnings_[i];
    }
    return nullptr;
}

BackendWarning *LogDiagnostics::add(BackendWarning warning, std::string key) {
    if (warnings_.size() >= kMaxWarnings)
        return nullptr;
    warnings_.push_back(std::move(warning));
    keys_.push_back(std::move(key));
    return &warnings_.back();
}

void LogDiagnostics::line(std::string_view text) {
    ++lines_;
    const auto bump = [&](std::string_view code, const std::string &key, auto make) {
        if (auto *existing = find(code, key)) {
            ++existing->count;
            return existing;
        }
        return add(make(), key);
    };

    static constexpr std::string_view kFaNeedle = "no FlashAttention vector kernel compiled for K/V types ";
    if (const auto at = text.find(kFaNeedle); at != std::string_view::npos) {
        const auto pos = at + kFaNeedle.size();
        const auto k = type_token(text, pos);
        std::string_view v;
        if (pos + k.size() < text.size() && text[pos + k.size()] == '-')
            v = type_token(text, pos + k.size() + 1);
        const std::string ks(k.empty() ? "unknown" : k);
        const std::string vs(v.empty() ? "unknown" : v);
        bump("kv_kernel_f16_fallback", ks + "-" + vs, [&] {
            return BackendWarning{
                "kv_kernel_f16_fallback", "warning", "log",
                "llama.cpp has no FlashAttention vector kernel for K/V cache types " + ks + "-" + vs +
                    " and converts K and V to f16 instead (llama.cpp marks this slow). Use a pair with a kernel "
                    "(e.g. --cache-type-k q8_0 --cache-type-v q8_0) or build llama.cpp with "
                    "GGML_CUDA_FA_ALL_QUANTS=ON",
                {{"k_type", ks}, {"v_type", vs}}, 1};
        });
        return;
    }

    static constexpr std::string_view kUnused = "model has unused tensor ";
    if (const auto at = text.find(kUnused); at != std::string_view::npos) {
        const auto name = name_token(text, at + kUnused.size());
        const auto nextn = name.find(".nextn.");
        if (nextn != std::string_view::npos) {
            const std::string layer(name.substr(0, nextn));
            std::uint64_t size = 0;
            static constexpr std::string_view kSize = "(size = ";
            if (const auto s = text.find(kSize, at); s != std::string_view::npos) {
                const auto digits = text.substr(s + kSize.size());
                const auto end = digits.find(' ');
                if (auto parsed = parse_uint(digits.substr(0, end)))
                    size = *parsed;
            }
            auto *w = bump("mtp_tensors_ignored", layer, [&] {
                return BackendWarning{"mtp_tensors_ignored", "info", "log",
                                      "the model's nextn (multi-token prediction) tensors in " + layer +
                                          " are ignored by this llama-server build: MTP speculative decoding is "
                                          "unavailable and those weights are loaded without being used",
                                      {}, 1};
            });
            if (w) {
                mtp_bytes_ = mtp_bytes_ > std::numeric_limits<std::uint64_t>::max() - size
                                 ? std::numeric_limits<std::uint64_t>::max()
                                 : mtp_bytes_ + size;
                w->details = {{"layer", layer}, {"tensors", std::to_string(w->count)},
                              {"ignored_bytes", std::to_string(mtp_bytes_)}};
            }
            return;
        }
    }

    static constexpr std::string_view kOffloaded = "offloaded ";
    if (const auto at = text.find(kOffloaded); at != std::string_view::npos &&
                                               text.find(" layers to GPU", at) != std::string_view::npos) {
        const auto rest = text.substr(at + kOffloaded.size());
        const auto slash = rest.find('/');
        const auto space = rest.find(' ');
        if (slash == std::string_view::npos || space == std::string_view::npos || slash > space)
            return;
        const auto done = parse_uint(rest.substr(0, slash));
        const auto total = parse_uint(rest.substr(slash + 1, space - slash - 1));
        if (!done || !total || *done > *total)
            return;
        offload_ = std::make_pair(*done, *total);
        if (*done == *total)
            return;
        const auto d = std::to_string(*done);
        const auto t = std::to_string(*total);
        bump("partial_gpu_offload", d + "/" + t, [&] {
            return BackendWarning{"partial_gpu_offload", "warning", "log",
                                  "only " + d + " of " + t +
                                      " layers are offloaded to the GPU; the remaining layers run on the CPU",
                                  {{"offloaded_layers", d}, {"total_layers", t}}, 1};
        });
        return;
    }

    static constexpr std::string_view kCpuInstead = "using CPU instead";
    static constexpr std::string_view kBuft = "cannot be used with preferred buffer type ";
    if (const auto at = text.find(kBuft); at != std::string_view::npos &&
                                          text.find(kCpuInstead, at) != std::string_view::npos) {
        const std::string buft(type_token(text, at + kBuft.size()));
        bump("cpu_buffer_fallback", buft, [&] {
            return BackendWarning{"cpu_buffer_fallback", "warning", "log",
                                  "some tensors cannot use buffer type " + buft + " and were placed on the CPU",
                                  {{"buffer_type", buft}}, 1};
        });
        return;
    }

    if (text.find("no usable GPU found") != std::string_view::npos) {
        bump("no_gpu_device", "", [&] {
            return BackendWarning{"no_gpu_device", "warning", "log",
                                  "llama-server found no usable GPU; the model runs on the CPU", {}, 1};
        });
        return;
    }

    if (text.find("failed to initialize CUDA") != std::string_view::npos) {
        bump("gpu_init_failed", "", [&] {
            return BackendWarning{"gpu_init_failed", "warning", "log",
                                  "CUDA initialization failed; llama-server falls back to the CPU", {}, 1};
        });
        return;
    }

    if (text.find("failed to initialize HIP") != std::string_view::npos ||
        text.find("failed to initialize ROCm") != std::string_view::npos) {
        bump("gpu_init_failed", "rocm", [&] {
            return BackendWarning{"gpu_init_failed", "warning", "log",
                                  "ROCm/HIP initialization failed; llama-server may fall back to the CPU",
                                  {{"backend", "rocm"}}, 1};
        });
        return;
    }

    // Check vendor-qualified GPU out-of-memory before generic runtime error codes
    // so e.g. hipErrorOutOfMemory and OutOfDeviceMemoryError surface as gpu_oom.
    std::string_view oom_vendor;
    if (icontains(text, "cuda error: out of memory")) {
        oom_vendor = "cuda";
    } else if (text.find("OutOfDeviceMemoryError") != std::string_view::npos ||
               text.find("VK_ERROR_OUT_OF_DEVICE_MEMORY") != std::string_view::npos) {
        oom_vendor = "vulkan";
    } else if (text.find("hipErrorOutOfMemory") != std::string_view::npos ||
               text.find("hipErrorMemoryAllocation") != std::string_view::npos) {
        oom_vendor = "rocm";
    } else if (icontains(text, "out of memory")) {
        if (icontains(text, "cuda") || icontains(text, "cublas"))
            oom_vendor = "cuda";
        else if (icontains(text, "rocm") || icontains_alpha_token(text, "hip") || icontains_alpha_token(text, "hsa"))
            oom_vendor = "rocm";
        else if (icontains(text, "vulkan") || icontains(text, "vk_"))
            oom_vendor = "vulkan";
    }
    if (!oom_vendor.empty()) {
        const std::string vendor(oom_vendor);
        bump("gpu_oom", vendor, [&] {
            return BackendWarning{
                "gpu_oom", "warning", "log",
                "GPU out of memory on " + vendor +
                    "; reduce --ctx-size, use smaller KV cache quantization, or offload fewer layers",
                {{"backend", vendor}}, 1};
        });
        return;
    }

    if (text.find("HSA_STATUS_ERROR") != std::string_view::npos ||
        text.find("hipError") != std::string_view::npos) {
        bump("rocm_runtime_error", "rocm", [&] {
            return BackendWarning{
                "rocm_runtime_error", "warning", "log",
                "ROCm/HIP runtime error reported by llama-server; check GPU architecture support and driver health",
                {{"backend", "rocm"}}, 1};
        });
        return;
    }

    if (text.find("failed to initialize Vulkan") != std::string_view::npos) {
        bump("vulkan_runtime_error", "init", [&] {
            return BackendWarning{"vulkan_runtime_error", "warning", "log",
                                  "Vulkan initialization failed; llama-server may fall back to the CPU",
                                  {{"backend", "vulkan"}, {"reason", "init_failed"}}, 1};
        });
        return;
    }

    if (text.find("VK_ERROR_DEVICE_LOST") != std::string_view::npos) {
        bump("vulkan_runtime_error", "device_lost", [&] {
            return BackendWarning{
                "vulkan_runtime_error", "warning", "log",
                "Vulkan device lost (VK_ERROR_DEVICE_LOST); GPU driver reset or compute timeout occurred",
                {{"backend", "vulkan"}, {"reason", "device_lost"}}, 1};
        });
        return;
    }

    const bool hadamard_error =
        icontains(text, "prism.hadamard") &&
        (icontains(text, "error") || icontains(text, "fail") || icontains(text, "unsupported") ||
         icontains(text, "unknown") || icontains(text, "invalid") || icontains(text, "missing") ||
         icontains(text, "incompatible") || icontains(text, "not supported") || icontains(text, "cannot") ||
         icontains(text, "unused"));
    if (icontains(text, "unknown tensor type") || icontains(text, "unsupported ggml type") ||
        icontains(text, "unsupported tensor type") || hadamard_error) {
        bump("ternary_runtime_incompatible", "", [&] {
            return BackendWarning{
                "ternary_runtime_incompatible", "warning", "log",
                "model requires tensor types or PrismML Hadamard support unavailable in this llama-server build; "
                "use a PrismML-capable binary or llama.cpp >= b11556",
                {}, 1};
        });
    }
}

} // namespace sonder::inference::llamaserver
