#include "log_diagnostics.hpp"
#include "utf8_path.hpp"

#include <algorithm>
#include <cctype>
#include <initializer_list>
#include <fstream>
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

// Value of an option spelled as `names` ("-c", "--ctx-size"), as
// "--name value" or "--name=value". Returns every occurrence's position of
// the value so callers can take the last or rewrite all of them.
struct OptionValue {
    std::size_t index;      // argv index holding the value
    std::size_t offset;     // start of the value within that argument
};

std::vector<OptionValue> option_values(const std::vector<std::string> &args,
                                       std::initializer_list<std::string_view> names) {
    std::vector<OptionValue> out;
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        for (const auto name : names) {
            if (arg == name) {
                if (i + 1 < args.size())
                    out.push_back({i + 1, 0});
                break;
            }
            if (arg.size() > name.size() && arg.compare(0, name.size(), name) == 0 && arg[name.size()] == '=' &&
                name.rfind("--", 0) == 0) {
                out.push_back({i, name.size() + 1});
                break;
            }
        }
    }
    return out;
}

std::optional<std::string> last_value(const std::vector<std::string> &args,
                                      std::initializer_list<std::string_view> names) {
    const auto values = option_values(args, names);
    if (values.empty())
        return std::nullopt;
    const auto &v = values.back();
    return args[v.index].substr(v.offset);
}

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    });
    return text;
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

enum class KvBackend { cuda, rocm, vulkan };

struct ResolvedBackend {
    KvBackend kind = KvBackend::cuda;
    std::string_view explicit_label;
};

std::string_view backend_label(KvBackend kind) noexcept {
    switch (kind) {
    case KvBackend::cuda: return "cuda";
    case KvBackend::rocm: return "rocm";
    case KvBackend::vulkan: return "vulkan";
    }
    return "cuda";
}

std::optional<KvBackend> parse_backend_hint(std::string_view hint) {
    const std::string low = lower(std::string(hint));
    if (low == "vulkan" || low == "vk")
        return KvBackend::vulkan;
    if (low == "rocm" || low == "hip")
        return KvBackend::rocm;
    if (low == "cuda" || low == "nvidia")
        return KvBackend::cuda;
    return std::nullopt;
}

ResolvedBackend resolve_kv_backend(const std::vector<std::string> &args, std::string_view backend_hint) {
    if (!backend_hint.empty()) {
        if (const auto parsed = parse_backend_hint(backend_hint))
            return {*parsed, backend_label(*parsed)};
    }
    if (const auto dev = last_value(args, {"-dev", "--device"})) {
        if (icontains(*dev, "vulkan"))
            return {KvBackend::vulkan, "vulkan"};
        if (icontains(*dev, "rocm") || icontains_alpha_token(*dev, "hip"))
            return {KvBackend::rocm, "rocm"};
        if (icontains(*dev, "cuda"))
            return {KvBackend::cuda, "cuda"};
    }
    return {KvBackend::cuda, {}};
}

bool is_default_kernel_type(std::string_view type) {
    return type == "f16" || type == "bf16" || type == "q8_0" || type == "q4_0";
}

} // namespace

std::optional<std::uint64_t> find_context_size(const std::vector<std::string> &args) {
    const auto value = last_value(args, {"-c", "--ctx-size"});
    if (!value)
        return std::nullopt;
    const auto parsed = parse_uint(*value);
    if (!parsed || *parsed == 0)
        return std::nullopt;
    return parsed;
}

std::vector<std::string> with_context_size(std::vector<std::string> args, std::uint64_t ctx) {
    const auto values = option_values(args, {"-c", "--ctx-size"});
    if (values.empty()) {
        args.push_back("--ctx-size");
        args.push_back(std::to_string(ctx));
        return args;
    }
    for (const auto &v : values)
        args[v.index] = args[v.index].substr(0, v.offset) + std::to_string(ctx);
    return args;
}

std::optional<std::string> find_log_file(const std::vector<std::string> &args) {
    auto value = last_value(args, {"--log-file"});
    if (value && value->empty())
        return std::nullopt;
    return value;
}

KvCacheConfig read_kv_cache_config(const std::vector<std::string> &args) {
    KvCacheConfig config;
    if (auto v = last_value(args, {"-ctk", "--cache-type-k"}))
        config.type_k = lower(*v);
    if (auto v = last_value(args, {"-ctv", "--cache-type-v"}))
        config.type_v = lower(*v);
    // -fa/--flash-attn takes on|off|auto in current llama.cpp; older builds
    // used a bare flag meaning "on".
    for (std::size_t i = 0; i < args.size(); ++i) {
        const std::string_view arg = args[i];
        std::optional<std::string> value;
        if (arg == "-fa" || arg == "--flash-attn") {
            if (i + 1 < args.size()) {
                const auto next = lower(args[i + 1]);
                if (next == "on" || next == "off" || next == "auto" || next == "true" || next == "false" ||
                    next == "1" || next == "0" || next == "enabled" || next == "disabled")
                    value = next;
            }
            if (!value)
                value = "on";
        } else if (arg.rfind("--flash-attn=", 0) == 0) {
            value = lower(std::string(arg.substr(13)));
        } else if (arg == "--no-flash-attn" || arg == "-nofa") {
            value = "off";
        }
        if (!value)
            continue;
        if (*value == "true" || *value == "1" || *value == "enabled")
            config.flash_attn = "on";
        else if (*value == "false" || *value == "0" || *value == "disabled")
            config.flash_attn = "off";
        else
            config.flash_attn = *value;
    }
    return config;
}

std::vector<BackendWarning> check_kv_cache_pairing(const std::vector<std::string> &args,
                                                   std::string_view backend_hint) {
    const auto config = read_kv_cache_config(args);
    std::vector<BackendWarning> out;
    if (config.flash_attn == "off")
        return out;
    const auto backend = resolve_kv_backend(args, backend_hint);
    std::vector<std::pair<std::string, std::string>> details{
        {"cache_type_k", config.type_k}, {"cache_type_v", config.type_v}, {"flash_attn", config.flash_attn}};
    if (!backend.explicit_label.empty())
        details.emplace_back("backend", std::string(backend.explicit_label));
    static constexpr const char *kMeasured =
        " (measured 2026-09-29 on an RTX 5070 Ti: q8_0/q5_1 and q5_1/q5_1 both log the conversion; q8_0/q8_0 "
        "does not)";
    if (config.type_k != config.type_v) {
        const std::string suggestion = is_default_kernel_type(config.type_k) ? config.type_k : std::string("q8_0");
        std::string message;
        if (backend.kind == KvBackend::vulkan) {
            message = "cache_type_k=" + config.type_k + " and cache_type_v=" + config.type_v +
                      " differ: llama.cpp's Vulkan FlashAttention may fall back or dequantize mixed K/V cache types "
                      "at run time. Use matching types, e.g. --cache-type-k " +
                      suggestion + " --cache-type-v " + suggestion;
        } else if (backend.kind == KvBackend::rocm) {
            message = "cache_type_k=" + config.type_k + " and cache_type_v=" + config.type_v +
                      " differ: llama.cpp's ROCm/HIP FlashAttention may lack a vector kernel for mixed K/V types and "
                      "convert K and V to f16 at run time. Use matching types, e.g. --cache-type-k " +
                      suggestion + " --cache-type-v " + suggestion;
        } else {
            message = "cache_type_k=" + config.type_k + " and cache_type_v=" + config.type_v +
                      " differ: llama.cpp's CUDA FlashAttention has no default vector kernel for mixed K/V types and "
                      "converts K and V to f16 at run time" +
                      std::string(kMeasured) + ". Use matching types, e.g. --cache-type-k " + suggestion +
                      " --cache-type-v " + suggestion;
        }
        out.push_back(BackendWarning{"kv_type_mismatch", "warning", "config", std::move(message), details, 1});
    } else if (config.type_k == "q5_1") {
        std::string message;
        if (backend.kind == KvBackend::vulkan) {
            message = "cache type q5_1/q5_1 may lack an optimized FlashAttention shader in llama.cpp's Vulkan "
                      "backend and fall back to f16 at run time. Use q8_0/q8_0 (or q4_0/q4_0 to save VRAM)";
        } else if (backend.kind == KvBackend::rocm) {
            message = "cache type q5_1/q5_1 has no default FlashAttention vector kernel in a standard llama.cpp "
                      "ROCm/HIP build and may convert K and V to f16 at run time. Use q8_0/q8_0 (or q4_0/q4_0 to save "
                      "VRAM)";
        } else {
            message = "cache type q5_1/q5_1 has no FlashAttention vector kernel in a default llama.cpp CUDA build, "
                      "so K and V are converted to f16 at run time" +
                      std::string(kMeasured) +
                      ". Use q8_0/q8_0 (or q4_0/q4_0 to save VRAM), or build llama.cpp with GGML_CUDA_FA_ALL_QUANTS=ON";
        }
        out.push_back(BackendWarning{"kv_type_no_vector_kernel", "warning", "config", std::move(message), details, 1});
    } else if (!is_default_kernel_type(config.type_k)) {
        std::string message;
        if (backend.kind == KvBackend::vulkan) {
            message = "cache type " + config.type_k + "/" + config.type_v +
                      " may lack an optimized FlashAttention shader in llama.cpp's Vulkan backend (f16, bf16, q8_0 "
                      "and q4_0 pairs are the common set); check the log for FlashAttention fallback messages";
        } else if (backend.kind == KvBackend::rocm) {
            message = "cache type " + config.type_k + "/" + config.type_v +
                      " may have no FlashAttention vector kernel in a standard llama.cpp ROCm/HIP build (f16, bf16, "
                      "q8_0 and q4_0 pairs are the common set); check the log for 'converting K and V to f16'";
        } else {
            message = "cache type " + config.type_k + "/" + config.type_v +
                      " may have no FlashAttention vector kernel in a default llama.cpp CUDA build (f16, bf16, q8_0 "
                      "and q4_0 pairs are the common set); check the log for 'converting K and V to f16' or build with "
                      "GGML_CUDA_FA_ALL_QUANTS=ON";
        }
        out.push_back(BackendWarning{"kv_type_kernel_unknown", "info", "config", std::move(message), details, 1});
    }
    return out;
}

Status LogTail::poll(LogDiagnostics &sink, std::size_t max_bytes) {
    if (path_.empty() || max_bytes == 0)
        return {};
    std::ifstream in(utf8_ ? utf8_path(path_) : std::filesystem::path(path_), std::ios::binary);
    if (!in)
        return Status(ErrorCode::not_found, "llamaserver: cannot open log file");
    in.seekg(0, std::ios::end);
    const auto end = in.tellg();
    if (end < 0)
        return Status(ErrorCode::io_error, "llamaserver: cannot size log file");
    const auto size = static_cast<std::uint64_t>(end);
    if (size < offset_) {
        // Truncated by a new child: start over with a clean parser state.
        offset_ = 0;
        sink.reset();
    }
    if (size == offset_)
        return {};
    const auto want = static_cast<std::size_t>(std::min<std::uint64_t>(size - offset_, max_bytes));
    std::string buffer(want, '\0');
    in.seekg(static_cast<std::streamoff>(offset_), std::ios::beg);
    in.read(buffer.data(), static_cast<std::streamsize>(want));
    const auto got = in.gcount();
    if (got <= 0)
        return in.bad() ? Status(ErrorCode::io_error, "llamaserver: cannot read log file") : Status{};
    buffer.resize(static_cast<std::size_t>(got));
    offset_ += static_cast<std::uint64_t>(got);
    sink.feed(buffer);
    return {};
}

} // namespace sonder::inference::llamaserver
