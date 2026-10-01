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

std::string lower(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
    });
    return text;
}

bool is_default_kernel_type(std::string_view type) {
    return type == "f16" || type == "bf16" || type == "q8_0" || type == "q4_0";
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

std::vector<BackendWarning> check_kv_cache_pairing(const std::vector<std::string> &args) {
    const auto config = read_kv_cache_config(args);
    std::vector<BackendWarning> out;
    if (config.flash_attn == "off")
        return out;
    const std::vector<std::pair<std::string, std::string>> details{
        {"cache_type_k", config.type_k}, {"cache_type_v", config.type_v}, {"flash_attn", config.flash_attn}};
    static constexpr const char *kMeasured =
        " (measured 2026-09-29 on an RTX 5070 Ti: q8_0/q5_1 and q5_1/q5_1 both log the conversion; q8_0/q8_0 "
        "does not)";
    if (config.type_k != config.type_v) {
        const std::string suggestion = is_default_kernel_type(config.type_k) ? config.type_k : std::string("q8_0");
        out.push_back(BackendWarning{
            "kv_type_mismatch", "warning", "config",
            "cache_type_k=" + config.type_k + " and cache_type_v=" + config.type_v +
                " differ: llama.cpp's CUDA FlashAttention has no default vector kernel for mixed K/V types and "
                "converts K and V to f16 at run time" + kMeasured + ". Use matching types, e.g. --cache-type-k " +
                suggestion + " --cache-type-v " + suggestion,
            details, 1});
    } else if (config.type_k == "q5_1") {
        out.push_back(BackendWarning{
            "kv_type_no_vector_kernel", "warning", "config",
            "cache type q5_1/q5_1 has no FlashAttention vector kernel in a default llama.cpp CUDA build, so K and V "
            "are converted to f16 at run time" + std::string(kMeasured) +
                ". Use q8_0/q8_0 (or q4_0/q4_0 to save VRAM), or build llama.cpp with the pair in GGML_CUDA_FA_QUANTS",
            details, 1});
    } else if (!is_default_kernel_type(config.type_k)) {
        out.push_back(BackendWarning{
            "kv_type_kernel_unknown", "info", "config",
            "cache type " + config.type_k + "/" + config.type_v +
                " may have no FlashAttention vector kernel in a default llama.cpp CUDA build (f16, bf16, q8_0 "
                "and q4_0 pairs are the common set); check the log for 'converting K and V to f16'",
            details, 1});
    }
    return out;
}

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
                    "(e.g. --cache-type-k q8_0 --cache-type-v q8_0) or build llama.cpp with this pair in "
                    "GGML_CUDA_FA_QUANTS",
                {{"k_type", ks}, {"v_type", vs}}, 1};
        });
        return;
    }

    static constexpr std::string_view kUnused = "model has unused tensor ";
    if (const auto at = text.find(kUnused); at != std::string_view::npos) {
        const auto name = name_token(text, at + kUnused.size());
        const auto nextn = name.find(".nextn.");
        if (nextn == std::string_view::npos)
            return;
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
    }
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
