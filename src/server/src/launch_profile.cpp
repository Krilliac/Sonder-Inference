// Launch profiles: schema, validation, llama-server argv, llamacpp mapping,
// sampling defaults and /v1/models metadata. See launch_profile.hpp.
#include "sonder/inference/launch_profile.hpp"
#include "config_text.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <set>
#include <sstream>
#include <utility>

namespace sonder::inference {

namespace {

constexpr std::uint32_t kMaxContext = 1u << 24;  // SamplingConfig::kMaxContextLimit
constexpr std::uint32_t kMaxBatch = 1u << 22;
constexpr std::size_t kMaxProfiles = 64;
constexpr std::size_t kMaxExtraArgs = 256;

Status bad(const std::string& where, const std::string& message) {
    return Status(ErrorCode::invalid_argument, "launch profile " + where + ": " + message);
}

std::string where_of(const LaunchProfile& p) { return p.name.empty() ? std::string("(unnamed)") : "'" + detail::bounded_for_message(p.name) + "'"; }

bool valid_name(std::string_view name) {
    if (name.empty() || name.size() > 128 || name == "default") return false;
    return std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
               c == '-' || c == ':';
    });
}

bool has_nul(std::string_view s) { return s.find('\0') != std::string_view::npos; }

bool has_nul_or_newline(std::string_view s) {
    return has_nul(s) || s.find('\r') != std::string_view::npos || s.find('\n') != std::string_view::npos;
}

std::string canonical_flash_attn(std::string_view v) {
    if (v == "on" || v == "enabled") return "on";
    if (v == "off" || v == "disabled") return "off";
    if (v == "auto") return "auto";
    return {};
}

bool valid_device_spec(std::string_view spec) {
    if (spec.empty() || spec.size() > 256 || has_nul(spec)) return false;
    std::size_t start = 0;
    for (;;) {
        const std::size_t comma = spec.find(',', start);
        const std::string_view item =
            spec.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (item.empty() || item.front() == '-') return false;
        for (const char c : item) {
            const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                            c == '_' || c == ':' || c == '.' || c == '-';
            if (!ok) return false;
        }
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return true;
}

bool valid_load_mode(std::string_view mode) {
    return mode == "auto" || mode == "mmap" || mode == "none" || mode == "mlock" || mode == "mmap+mlock" ||
           mode == "dio";
}

bool is_llamacpp_kv_cache_type(std::string_view type) {
    return type == "f32" || type == "f16" || type == "bf16" || type == "q8_0" || type == "q5_1" || type == "q5_0" ||
           type == "q4_1" || type == "q4_0" || type == "iq4_nl";
}

std::string format_tensor_split(const std::vector<double>& weights) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(6);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        if (i > 0) out << ',';
        out << weights[i];
    }
    return out.str();
}

std::string format_profile_double(double value) {
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(6) << value;
    return out.str();
}

// The flag lists below hold the names llama-server looks flags up by
// (llamaserver_flag_name): hyphens, never underscores. check_extra_args
// compares each extra argument under that name, so `--ctx_size` and
// `--mcp_servers_json` match `--ctx-size` and `--mcp-servers-json`.

// Every spelling of the llama-server flags that the typed fields own, plus
// flags Sonder owns (model, alias). Taken from `llama-server --help`.
constexpr std::pair<std::string_view, std::string_view> kTypedFlags[] = {
    {"-m", "model"}, {"--model", "model"},
    {"-a", "name"}, {"--alias", "name"},
    {"-mm", "mmproj"}, {"--mmproj", "mmproj"},
    {"--mmproj-offload", "mmproj_offload"}, {"--no-mmproj-offload", "mmproj_offload"},
    {"--image-max-tokens", "image_max_tokens"},
    {"-c", "ctx_size"}, {"--ctx-size", "ctx_size"},
    {"-ngl", "n_gpu_layers"}, {"--gpu-layers", "n_gpu_layers"}, {"--n-gpu-layers", "n_gpu_layers"},
    {"-dev", "device"}, {"--device", "device"},
    {"-mg", "main_gpu"}, {"--main-gpu", "main_gpu"},
    {"-sm", "split_mode"}, {"--split-mode", "split_mode"},
    {"-ts", "tensor_split"}, {"--tensor-split", "tensor_split"},
    {"-lm", "load_mode"}, {"--load-mode", "load_mode"},
    {"-nr", "no_repack"}, {"--no-repack", "no_repack"}, {"--repack", "no_repack"},
    {"--moe-cache-mib", "moe_cache_mib"},
    {"--moe-cache-cpu-share", "moe_cache_cpu_share"},
    {"-b", "batch_size"}, {"--batch-size", "batch_size"},
    {"-ub", "ubatch_size"}, {"--ubatch-size", "ubatch_size"},
    {"-np", "parallel"}, {"--parallel", "parallel"},
    {"-kvu", "kv_unified"}, {"--kv-unified", "kv_unified"}, {"-no-kvu", "kv_unified"}, {"--no-kv-unified", "kv_unified"},
    {"-fa", "flash_attn"}, {"--flash-attn", "flash_attn"},
    {"-ctk", "cache_type_k"}, {"--cache-type-k", "cache_type_k"},
    {"-ctv", "cache_type_v"}, {"--cache-type-v", "cache_type_v"},
    {"-ctxcp", "ctx_checkpoints"}, {"--ctx-checkpoints", "ctx_checkpoints"}, {"--swa-checkpoints", "ctx_checkpoints"},
    {"-cms", "checkpoint_min_step"}, {"--checkpoint-min-step", "checkpoint_min_step"},
    {"-cram", "cache_ram_mib"}, {"--cache-ram", "cache_ram_mib"},
    {"--context-shift", "context_shift"}, {"--no-context-shift", "context_shift"},
    {"-fit", "fit"}, {"--fit", "fit"},
    {"--spec-type", "speculative.types"},
    {"--spec-draft-n-max", "speculative.draft_n_max"},
    {"-md", "speculative.draft_model"}, {"--model-draft", "speculative.draft_model"},
    {"--spec-draft-model", "speculative.draft_model"},
    {"-t", "threads"}, {"--threads", "threads"},
    {"-tb", "threads_batch"}, {"--threads-batch", "threads_batch"},
    {"--jinja", "jinja"}, {"--no-jinja", "jinja"},
    {"--reasoning-format", "reasoning_format"},
    {"-rea", "reasoning"}, {"--reasoning", "reasoning"},
    {"--reasoning-budget", "reasoning_budget"},
    {"--temp", "sampling.temperature"}, {"--temperature", "sampling.temperature"},
    {"--top-p", "sampling.top_p"}, {"--top-k", "sampling.top_k"}, {"--min-p", "sampling.min_p"},
    {"--presence-penalty", "sampling.presence_penalty"}, {"--repeat-penalty", "sampling.repeat_penalty"},
};

// Flags a profile may never pass: the supervisor owns the listen address,
// Sonder does not hand the child credentials, and profiles never download.
constexpr std::string_view kForbiddenFlags[] = {
    "--", "--host", "--port", "--api-key", "--api-key-file", "--ssl-key-file", "--ssl-cert-file",
    "-mu", "--model-url", "-hf", "-hfr", "--hf-repo", "-hff", "--hf-file", "-hft", "--hf-token",
    "-hfv", "-hfrv", "--hf-repo-v", "-hffv", "--hf-file-v",
    "-dr", "--docker-repo", "-mmu", "--mmproj-url", "-hfd", "-hfrd", "--hf-repo-draft", "-hffd",
    "--hf-file-draft", "--spec-draft-hf", "--vocoder-url",
    "--models-dir", "--models-preset",
};

// llama-server features that run tools, reach MCP servers or read/serve local
// files (the `--help` of each says "do not enable in untrusted environments"
// or serves a directory). A profile is data that is meant to be shared, so it
// may not switch them on; the `--no-*` spellings stay allowed.
constexpr std::string_view kUnsafeFeatureFlags[] = {
    "--tools", "--tools-runtime", "--allow-unauthenticated-tools",
    "--mcp-servers-config", "--mcp-servers-json", "-ag", "--agent",
    "--ui-mcp-proxy", "--webui-mcp-proxy", "--ui-mcp-proxy-allow-private-urls",
    "--ui-config-file", "--webui-config-file",
    "--path", "--media-path", "--media-allow-private-urls",
};

// llama-server flags that write prompts or logs to an arbitrary local path.
// A shared profile must not be able to record user prompts to disk.
// `-lcd`/`--lookup-cache-dynamic` is an n-gram cache that generation updates,
// so it records prompt/response text too. `--slot-save-path` stays allowed on
// purpose: docs/integration/llama-server.md tells operators to pass it.
constexpr std::string_view kDiskWriteFlags[] = {
    "--log-file", "--log-prompts-dir", "-lcd", "--lookup-cache-dynamic",
};

// llama-server flags that move model work to other machines or expose P2P
// remote tunnels: `--rpc HOST:PORT,...` registers remote rpc-server devices,
// and `--connect`/`--connect-code` tunnel requests off-host. A shared profile
// must not route prompts off this host.
constexpr std::string_view kRemoteOffloadFlags[] = {"--rpc", "--connect", "--connect-code"};

// llama-server flags that choose a program for the child to run:
// `--video-ffmpeg-dir` is the directory it starts ffmpeg and ffprobe from for
// video input (tools/mtmd/mtmd-helper.cpp). A shared profile must not pick
// executables.
constexpr std::string_view kExternalProgramFlags[] = {"--video-ffmpeg-dir"};

template <std::size_t N>
bool listed(const std::string_view (&list)[N], std::string_view flag) {
    return std::find(std::begin(list), std::end(list), flag) != std::end(list);
}

// llama-server's built-in model presets (`--gpt-oss-20b-default`,
// `--fim-qwen-7b-spec`, `--spec-default`, ...). Each one swaps in its own model
// and settings, overriding the typed fields, and most "can download weights
// from the internet". Every `--*-default` / `--*-spec` spelling is refused, so
// presets added upstream later are covered too.
bool is_model_preset_flag(std::string_view flag) {
    const auto ends_with = [&](std::string_view suffix) {
        return flag.size() > suffix.size() && flag.substr(flag.size() - suffix.size()) == suffix;
    };
    return flag.rfind("--", 0) == 0 && (ends_with("-default") || ends_with("-spec"));
}

// Future-proof download-family classifier (G0094): refuses any Hugging Face
// download flag (-hf*, --hf-*, --*-hf, --*-hf-*) or URL download flag
// (--*-url) so newly added upstream download aliases cannot bypass offline
// launch profiles.
bool is_download_flag(std::string_view flag) {
    const auto ends_with = [&](std::string_view suffix) {
        return flag.size() > suffix.size() && flag.substr(flag.size() - suffix.size()) == suffix;
    };
    if (flag.rfind("-hf", 0) == 0) {
        return true;
    }
    if (flag.rfind("--", 0) == 0) {
        if (flag.rfind("--hf-", 0) == 0 || ends_with("-url") || ends_with("-hf") ||
            flag.find("-hf-") != std::string_view::npos) {
            return true;
        }
    }
    return false;
}

Status check_extra_args(const LaunchProfile& p) {
    if (p.extra_args.size() > kMaxExtraArgs) return bad(where_of(p), "extra_args has more than 256 entries");
    for (const auto& arg : p.extra_args) {
        if (has_nul(arg)) return bad(where_of(p), "extra_args must not contain NUL");
        if (arg.size() < 2 || arg[0] != '-') continue;  // a value, not a flag
        if (has_nul_or_newline(arg)) return bad(where_of(p), "extra_args flag must not contain newline characters");
        // Judge the flag llama-server will actually run: it reads
        // `--mcp_servers_json` as `--mcp-servers-json`.
        const std::string flag = llamaserver_flag_name(arg);
        const std::string_view written = std::string_view(arg).substr(0, arg.find('='));
        const std::string refused = "extra_args must not contain " + detail::bounded_for_message(written) +
                                    (written == flag ? std::string() : ", which llama-server reads as " + detail::bounded_for_message(flag));
        if (listed(kForbiddenFlags, flag) || is_download_flag(flag)) {
            return bad(where_of(p), refused +
                                        " (the supervisor owns host and port; credentials and downloads are not "
                                        "passed to llama-server)");
        }
        if (is_model_preset_flag(flag)) {
            return bad(where_of(p), refused +
                                        " (llama-server's built-in model presets replace the profile's model and can "
                                        "download weights; set the typed fields instead)");
        }
        if (listed(kDiskWriteFlags, flag)) {
            return bad(where_of(p), refused + " (a launch profile cannot make llama-server write logs or prompts to disk)");
        }
        if (listed(kRemoteOffloadFlags, flag)) {
            return bad(where_of(p), refused +
                                        " (a launch profile cannot offload model work, and the activations that "
                                        "carry each prompt, to remote RPC servers)");
        }
        if (listed(kUnsafeFeatureFlags, flag)) {
            return bad(where_of(p), refused +
                                        " (llama-server's agent tools, MCP and local file serving cannot be enabled "
                                        "from a launch profile)");
        }
        if (listed(kExternalProgramFlags, flag)) {
            return bad(where_of(p), refused + " (a launch profile cannot choose programs for llama-server to run)");
        }
        for (const auto& [f, field] : kTypedFlags) {
            if (flag == f) return bad(where_of(p), refused + "; use the typed field '" + std::string(field) + "'");
        }
    }
    return {};
}

// ------------------------------------------------------------ JSON reading

class Reader {
public:
    Reader(const json::Object& object, std::string where) : object_(object), where_(std::move(where)) {}

    Status check_keys(std::initializer_list<std::string_view> known) const {
        for (const auto& member : object_) {
            if (std::find(known.begin(), known.end(), member.first) == known.end()) {
                return bad(where_, "unknown field '" + detail::bounded_for_message(member.first) + "'");
            }
        }
        return {};
    }
    Status string(const char* key, std::string& out, bool required = false) const {
        const auto* v = object_.find(key);
        if (!v) return required ? bad(where_, std::string("'") + key + "' is required") : Status{};
        if (!v->is_string() || v->as_string().empty() || has_nul(v->as_string()))
            return bad(where_, std::string("'") + key + "' must be a non-empty string");
        out = v->as_string();
        return {};
    }
    Status string(const char* key, std::optional<std::string>& out) const {
        std::string s;
        if (object_.find(key) == nullptr) return {};
        if (auto st = string(key, s); !st.ok()) return st;
        out = std::move(s);
        return {};
    }
    Status boolean(const char* key, std::optional<bool>& out) const {
        const auto* v = object_.find(key);
        if (!v) return {};
        if (!v->is_bool()) return bad(where_, std::string("'") + key + "' must be true or false");
        out = v->as_bool();
        return {};
    }
    template <class T>
    Status integer(const char* key, std::optional<T>& out, std::int64_t lo, std::int64_t hi) const {
        const auto* v = object_.find(key);
        if (!v) return {};
        if (!v->is_integer() || v->as_int(lo - 1) < lo || v->as_int(lo - 1) > hi) {
            return bad(where_, std::string("'") + key + "' must be an integer from " + std::to_string(lo) + " to " +
                                   std::to_string(hi));
        }
        out = static_cast<T>(v->as_int());
        return {};
    }
    Status number(const char* key, std::optional<float>& out) const {
        const auto* v = object_.find(key);
        if (!v) return {};
        if (!v->is_number() || !std::isfinite(v->as_double())) return bad(where_, std::string("'") + key + "' must be a number");
        out = static_cast<float>(v->as_double());
        return {};
    }
    Status number(const char* key, std::optional<double>& out) const {
        const auto* v = object_.find(key);
        if (!v) return {};
        if (!v->is_number() || !std::isfinite(v->as_double())) return bad(where_, std::string("'") + key + "' must be a number");
        out = v->as_double();
        return {};
    }
    Status strings(const char* key, std::vector<std::string>& out) const {
        const auto* v = object_.find(key);
        if (!v) return {};
        if (!v->is_array()) return bad(where_, std::string("'") + key + "' must be an array of strings");
        out.clear();
        for (const auto& item : v->as_array()) {
            if (!item.is_string()) return bad(where_, std::string("'") + key + "' entries must be strings");
            out.push_back(item.as_string());
        }
        return {};
    }
    Status numbers(const char* key, std::vector<double>& out) const {
        const auto* v = object_.find(key);
        if (!v) return {};
        if (!v->is_array()) return bad(where_, std::string("'") + key + "' must be an array of numbers");
        out.clear();
        for (const auto& item : v->as_array()) {
            if (!item.is_number() || !std::isfinite(item.as_double())) {
                return bad(where_, std::string("'") + key + "' entries must be positive finite numbers");
            }
            out.push_back(item.as_double());
        }
        return {};
    }
    const json::Value* find(const char* key) const { return object_.find(key); }
    const std::string& where() const { return where_; }

private:
    const json::Object& object_;
    std::string where_;
};

#define SONDER_TRY(expr)                   \
    do {                                   \
        if (Status st_ = (expr); !st_.ok()) \
            return st_;                    \
    } while (false)

Result<LaunchProfile> parse_one(const json::Value& value, std::size_t index) {
    const std::string at = "#" + std::to_string(index);
    if (!value.is_object()) return bad(at, "must be a JSON object");
    const auto* name = value.find("name");
    const std::string where = name && name->is_string() ? "'" + name->as_string() + "'" : at;
    Reader r(value.as_object(), where);
    SONDER_TRY(r.check_keys({"name", "backend", "model", "mmproj", "mmproj_offload", "image_max_tokens", "ctx_size",
                             "n_gpu_layers", "device", "main_gpu", "split_mode", "tensor_split", "load_mode",
                             "no_repack", "moe_cache_mib", "moe_cache_cpu_share", "batch_size", "ubatch_size",
                             "parallel", "kv_unified", "swa_kv_mode", "flash_attn", "cache_type_k", "cache_type_v",
                             "ctx_checkpoints", "checkpoint_min_step", "cache_ram_mib", "context_shift", "fit",
                             "speculative", "threads", "threads_batch", "jinja", "reasoning_format", "reasoning",
                             "reasoning_budget", "sampling", "extra_args", "vram_budget_mib"}));
    LaunchProfile p;
    SONDER_TRY(r.string("name", p.name, true));
    SONDER_TRY(r.string("backend", p.backend, true));
    SONDER_TRY(r.string("model", p.model, true));
    SONDER_TRY(r.string("mmproj", p.mmproj));
    SONDER_TRY(r.boolean("mmproj_offload", p.mmproj_offload));
    SONDER_TRY(r.integer("image_max_tokens", p.image_max_tokens, 1, 65536));
    SONDER_TRY(r.integer("ctx_size", p.ctx_size, 0, kMaxContext));
    SONDER_TRY(r.integer("n_gpu_layers", p.n_gpu_layers, -1, 100000));
    SONDER_TRY(r.string("device", p.device));
    SONDER_TRY(r.integer("main_gpu", p.main_gpu, 0, 1024));
    SONDER_TRY(r.string("split_mode", p.split_mode));
    SONDER_TRY(r.numbers("tensor_split", p.tensor_split));
    SONDER_TRY(r.string("load_mode", p.load_mode));
    SONDER_TRY(r.boolean("no_repack", p.no_repack));
    SONDER_TRY(r.integer("moe_cache_mib", p.moe_cache_mib, 0, 262144));
    SONDER_TRY(r.number("moe_cache_cpu_share", p.moe_cache_cpu_share));
    SONDER_TRY(r.integer("batch_size", p.batch_size, 1, kMaxBatch));
    SONDER_TRY(r.integer("ubatch_size", p.ubatch_size, 1, kMaxBatch));
    SONDER_TRY(r.integer("parallel", p.parallel, 1, 256));
    SONDER_TRY(r.boolean("kv_unified", p.kv_unified));
    {
        std::optional<std::string> swa_mode_str;
        SONDER_TRY(r.string("swa_kv_mode", swa_mode_str));
        if (swa_mode_str) {
            SwaKvMode parsed_mode = SwaKvMode::Full;
            if (!parse_swa_kv_mode(*swa_mode_str, parsed_mode)) {
                return bad(where, "'swa_kv_mode' must be full or compact_if_qualified");
            }
            p.swa_kv_mode = parsed_mode;
        }
    }
    SONDER_TRY(r.string("flash_attn", p.flash_attn));
    SONDER_TRY(r.string("cache_type_k", p.cache_type_k));
    SONDER_TRY(r.string("cache_type_v", p.cache_type_v));
    SONDER_TRY(r.integer("ctx_checkpoints", p.ctx_checkpoints, 0, 1024));
    SONDER_TRY(r.integer("checkpoint_min_step", p.checkpoint_min_step, 0, kMaxContext));
    SONDER_TRY(r.integer("cache_ram_mib", p.cache_ram_mib, -1, 1 << 24));
    SONDER_TRY(r.boolean("context_shift", p.context_shift));
    SONDER_TRY(r.boolean("fit", p.fit));
    SONDER_TRY(r.integer("threads", p.threads, -1, 1024));
    SONDER_TRY(r.integer("threads_batch", p.threads_batch, -1, 1024));
    SONDER_TRY(r.boolean("jinja", p.jinja));
    SONDER_TRY(r.string("reasoning_format", p.reasoning_format));
    SONDER_TRY(r.string("reasoning", p.reasoning));
    SONDER_TRY(r.integer("reasoning_budget", p.reasoning_budget, -1, kMaxContext));
    SONDER_TRY(r.strings("extra_args", p.extra_args));
    SONDER_TRY(r.integer("vram_budget_mib", p.vram_budget_mib, 1, 1 << 24));
    if (const auto* spec = r.find("speculative")) {
        if (!spec->is_object()) return bad(where, "'speculative' must be an object");
        Reader s(spec->as_object(), where + " speculative");
        SONDER_TRY(s.check_keys({"types", "draft_n_max", "draft_model"}));
        SpeculativeSettings settings;
        SONDER_TRY(s.strings("types", settings.types));
        SONDER_TRY(s.integer("draft_n_max", settings.draft_n_max, 1, 64));
        SONDER_TRY(s.string("draft_model", settings.draft_model));
        p.speculative = std::move(settings);
    }
    if (const auto* sampling = r.find("sampling")) {
        if (!sampling->is_object()) return bad(where, "'sampling' must be an object");
        Reader s(sampling->as_object(), where + " sampling");
        SONDER_TRY(s.check_keys({"temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty"}));
        SONDER_TRY(s.number("temperature", p.sampling.temperature));
        SONDER_TRY(s.number("top_p", p.sampling.top_p));
        SONDER_TRY(s.integer("top_k", p.sampling.top_k, 0, 100000));
        SONDER_TRY(s.number("min_p", p.sampling.min_p));
        SONDER_TRY(s.number("presence_penalty", p.sampling.presence_penalty));
        SONDER_TRY(s.number("repeat_penalty", p.sampling.repeat_penalty));
    }
    if (p.flash_attn) {
        const std::string canonical = canonical_flash_attn(*p.flash_attn);
        if (canonical.empty()) return bad(where, "'flash_attn' must be on, off or auto");
        p.flash_attn = canonical;
    }
    return p;
}

}  // namespace

std::string llamaserver_flag_name(std::string_view arg) {
    std::string flag(arg.substr(0, arg.find('=')));
    // common_params_parse_ex: `if (arg.compare(0, 2, "--") == 0)
    // std::replace(arg.begin(), arg.end(), '_', '-');` before the lookup.
    if (flag.rfind("--", 0) == 0) std::replace(flag.begin(), flag.end(), '_', '-');
    return flag;
}

double kv_cache_type_bytes(std::string_view type) noexcept {
    if (type == "f32") return 4.0;
    if (type == "f16" || type == "bf16") return 2.0;
    if (type == "f8_e4m3") return 1.0;
    if (type == "q8_0") return 34.0 / 32.0;
    if (type == "q5_1") return 24.0 / 32.0;
    if (type == "q5_0") return 22.0 / 32.0;
    if (type == "q4_1") return 20.0 / 32.0;
    if (type == "q4_0" || type == "iq4_nl") return 18.0 / 32.0;
    if (type == "q2_0") return 18.0 / 64.0;
    if (type == "pq2_0m") return 36.0 / 128.0;
    return 0.0;
}

bool is_quantized_kv_cache_type(std::string_view type) noexcept {
    return kv_cache_type_bytes(type) > 0.0 && type != "f32" && type != "f16" && type != "bf16";
}

Status validate_launch_profile(const LaunchProfile& p) {
    const std::string where = where_of(p);
    if (!valid_name(p.name)) {
        return bad(where, "'name' must be 1-128 characters from [A-Za-z0-9._:-] and not 'default'");
    }
    if (p.backend != kLaunchProfileBackendLlamaServer && p.backend != kLaunchProfileBackendLlamaCpp) {
        return bad(where, "'backend' must be llamaserver or llamacpp");
    }
    if (p.model.empty() || has_nul_or_newline(p.model)) {
        return bad(where, "'model' must be a GGUF path without NUL or newline characters");
    }
    if (has_nul_or_newline(p.mmproj)) return bad(where, "'mmproj' must not contain NUL or newline characters");
    if (p.mmproj.empty() && (p.mmproj_offload || p.image_max_tokens)) {
        return bad(where, "'mmproj_offload' and 'image_max_tokens' need 'mmproj'");
    }
    if (p.ctx_size && *p.ctx_size > kMaxContext) return bad(where, "'ctx_size' is out of range");
    if (p.n_gpu_layers && *p.n_gpu_layers < -1) return bad(where, "'n_gpu_layers' must be -1 or more");
    if (p.device && !valid_device_spec(*p.device)) {
        return bad(where, "'device' must be a comma-separated list of device names (e.g. CUDA0, ROCm0, Vulkan0)");
    }
    if (p.main_gpu && *p.main_gpu < 0) return bad(where, "'main_gpu' must be 0 or more");
    if (p.split_mode && *p.split_mode != "none" && *p.split_mode != "layer" && *p.split_mode != "row") {
        return bad(where, "'split_mode' must be none, layer or row");
    }
    if (p.tensor_split.size() > 16) return bad(where, "'tensor_split' must have at most 16 entries");
    for (const double w : p.tensor_split) {
        if (!std::isfinite(w) || !(w > 0.0)) {
            return bad(where, "'tensor_split' entries must be positive finite numbers");
        }
    }
    if (p.load_mode && !valid_load_mode(*p.load_mode)) {
        return bad(where, "'load_mode' must be one of auto, mmap, none, mlock, mmap+mlock, dio");
    }
    if (p.moe_cache_mib && *p.moe_cache_mib > 262144u) {
        return bad(where, "'moe_cache_mib' must be from 0 to 262144");
    }
    if (p.moe_cache_cpu_share &&
        (!std::isfinite(*p.moe_cache_cpu_share) || *p.moe_cache_cpu_share < 0.0 || *p.moe_cache_cpu_share > 1.0)) {
        return bad(where, "'moe_cache_cpu_share' must be a finite number from 0.0 to 1.0");
    }
    if (p.swa_kv_mode && to_string(*p.swa_kv_mode) == "unknown") {
        return bad(where, "'swa_kv_mode' must be full or compact_if_qualified");
    }
    if (p.flash_attn && canonical_flash_attn(*p.flash_attn) != *p.flash_attn) {
        return bad(where, "'flash_attn' must be on, off or auto");
    }
    for (const auto* t : {&p.cache_type_k, &p.cache_type_v}) {
        if (*t && kv_cache_type_bytes(**t) == 0.0) {
            return bad(where, std::string(t == &p.cache_type_k ? "'cache_type_k'" : "'cache_type_v'") +
                                  " must be one of f32, f16, bf16, f8_e4m3, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1, "
                                  "q2_0, pq2_0m");
        }
    }
    // llama.cpp refuses a quantized V cache without flash attention
    // ("quantized V cache requires flash_attn"); auto lets it switch FA on.
    if (p.cache_type_v && is_quantized_kv_cache_type(*p.cache_type_v) && p.flash_attn && *p.flash_attn == "off") {
        return bad(where, "cache_type_v " + *p.cache_type_v + " is quantized and requires flash attention (on or auto)");
    }
    // Upstream defaults: llama-server batch 2048 / ubatch 512; the llamacpp
    // backend batch 512.
    const std::uint32_t batch =
        p.batch_size.value_or(p.backend == kLaunchProfileBackendLlamaCpp ? 512u : 2048u);
    if (p.ubatch_size && *p.ubatch_size > batch) {
        return bad(where, "'ubatch_size' (" + std::to_string(*p.ubatch_size) + ") must not exceed 'batch_size' (" +
                              std::to_string(batch) + ")");
    }
    if (p.reasoning_format && *p.reasoning_format != "none" && *p.reasoning_format != "deepseek" &&
        *p.reasoning_format != "deepseek-legacy" && *p.reasoning_format != "auto") {
        return bad(where, "'reasoning_format' must be none, deepseek, deepseek-legacy or auto");
    }
    if (p.reasoning && *p.reasoning != "on" && *p.reasoning != "off" && *p.reasoning != "auto") {
        return bad(where, "'reasoning' must be on, off or auto");
    }
    if (p.reasoning_budget && *p.reasoning_budget < -1) {
        return bad(where, "'reasoning_budget' must be -1 or more");
    }
    // Without --parallel, llama-server picks its slots automatically and then
    // unifies the KV cache whatever --no-kv-unified said
    // (kLlamaServerAutoParallel), so the field would silently not apply.
    if (p.backend == kLaunchProfileBackendLlamaServer && p.kv_unified.has_value() && !*p.kv_unified && !p.parallel) {
        return bad(where, "'kv_unified': false needs 'parallel' (with 'parallel' unset, llama-server runs 4 slots and "
                          "always unifies the KV cache)");
    }
    if (p.speculative) {
        if (p.speculative->types.empty()) return bad(where, "'speculative.types' must not be empty");
        std::set<std::string> seen;
        const auto in_list = [](const auto& list, std::string_view v) {
            return std::find(std::begin(list), std::end(list), v) != std::end(list);
        };
        for (const auto& t : p.speculative->types) {
            const bool valid = in_list(kSpeculativeTypes, t) ||
                               (p.backend == kLaunchProfileBackendLlamaCpp && in_list(kLlamaCppSpeculativeTypes, t));
            if (!valid) {
                return bad(where, "unknown speculative type '" + detail::bounded_for_message(t) + "'");
            }
            if (!seen.insert(t).second) return bad(where, "speculative type '" + t + "' is listed twice");
        }
        if (seen.count("none") && seen.size() > 1) return bad(where, "speculative type 'none' cannot be combined");
        if (has_nul_or_newline(p.speculative->draft_model)) {
            return bad(where, "'speculative.draft_model' must not contain NUL or newline characters");
        }
    }
    SamplingConfig check;
    apply_sampling_defaults(p.sampling, check);
    if (Status st = validate(check); !st.ok()) return bad(where, "sampling: " + st.message());
    if (p.vram_budget_mib && *p.vram_budget_mib == 0) return bad(where, "'vram_budget_mib' must be positive");
    return check_extra_args(p);
}

Result<std::vector<LaunchProfile>> parse_launch_profiles(const json::Value& document) {
    if (!document.is_object()) return bad("file", "must be a JSON object with a 'profiles' array");
    for (const auto& member : document.as_object()) {
        if (member.first != "profiles") return bad("file", "unknown field '" + detail::bounded_for_message(member.first) + "'");
    }
    const auto* list = document.find("profiles");
    if (!list || !list->is_array() || list->as_array().empty()) {
        return bad("file", "'profiles' must be a non-empty array");
    }
    if (list->as_array().size() > kMaxProfiles) return bad("file", "at most 64 profiles");
    std::vector<LaunchProfile> out;
    std::set<std::string> names;
    std::size_t index = 0;
    for (const auto& item : list->as_array()) {
        auto parsed = parse_one(item, index++);
        if (!parsed.ok()) return parsed.status();
        if (Status st = validate_launch_profile(parsed.value()); !st.ok()) return st;
        if (!names.insert(parsed.value().name).second) {
            return bad("'" + detail::bounded_for_message(parsed.value().name) + "'", "the name is used by more than one profile");
        }
        out.push_back(std::move(parsed).value());
    }
    return out;
}

Result<std::vector<LaunchProfile>> load_launch_profiles(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return Status(ErrorCode::not_found, "cannot read launch profiles " + path);
    std::string text(1024 * 1024 + 1, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (in.bad()) return Status(ErrorCode::io_error, "cannot read launch profiles " + path);
    if (in.gcount() > 1024 * 1024) return Status(ErrorCode::invalid_argument, "launch profiles file exceeds 1 MiB");
    text.resize(static_cast<std::size_t>(in.gcount()));
    auto parsed = json::parse(text);
    if (!parsed.ok()) return Status(ErrorCode::invalid_argument, "launch profiles " + path + ": " + parsed.status().message());
    return parse_launch_profiles(parsed.value());
}

const LaunchProfile* find_launch_profile(const std::vector<LaunchProfile>& profiles, std::string_view name) {
    for (const auto& p : profiles) {
        if (p.name == name) return &p;
    }
    return nullptr;
}

Result<std::vector<std::string>> llamaserver_arguments(const LaunchProfile& p) {
    if (Status st = validate_launch_profile(p); !st.ok()) return st;
    if (p.backend != kLaunchProfileBackendLlamaServer) {
        return bad(where_of(p), "llama-server arguments need backend llamaserver");
    }
    std::vector<std::string> a{"--model", p.model, "--alias", p.name};
    const auto num = [&a](const char* flag, auto value) {
        a.emplace_back(flag);
        a.push_back(std::to_string(value));
    };
    const auto toggle = [&a](const std::optional<bool>& v, const char* on, const char* off) {
        if (v) a.emplace_back(*v ? on : off);
    };
    if (!p.mmproj.empty()) {
        a.emplace_back("--mmproj");
        a.push_back(p.mmproj);
    }
    toggle(p.mmproj_offload, "--mmproj-offload", "--no-mmproj-offload");
    if (p.image_max_tokens) num("--image-max-tokens", *p.image_max_tokens);
    if (p.ctx_size) num("--ctx-size", *p.ctx_size);
    if (p.n_gpu_layers) {
        a.emplace_back("--n-gpu-layers");
        a.push_back(*p.n_gpu_layers < 0 ? std::string("all") : std::to_string(*p.n_gpu_layers));
    }
    if (p.device) {
        a.emplace_back("--device");
        a.push_back(*p.device);
    }
    if (p.main_gpu) num("--main-gpu", *p.main_gpu);
    if (p.split_mode) {
        a.emplace_back("--split-mode");
        a.push_back(*p.split_mode);
    }
    if (!p.tensor_split.empty()) {
        a.emplace_back("--tensor-split");
        a.push_back(format_tensor_split(p.tensor_split));
    }
    if (p.load_mode) {
        a.emplace_back("--load-mode");
        a.push_back(*p.load_mode);
    }
    toggle(p.no_repack, "--no-repack", "--repack");
    if (p.moe_cache_mib) num("--moe-cache-mib", *p.moe_cache_mib);
    if (p.moe_cache_cpu_share) {
        a.emplace_back("--moe-cache-cpu-share");
        a.push_back(format_profile_double(*p.moe_cache_cpu_share));
    }
    if (p.batch_size) num("--batch-size", *p.batch_size);
    if (p.ubatch_size) num("--ubatch-size", *p.ubatch_size);
    if (p.parallel) num("--parallel", *p.parallel);
    toggle(p.kv_unified, "--kv-unified", "--no-kv-unified");
    if (p.swa_kv_mode && *p.swa_kv_mode == SwaKvMode::Full) {
        const bool already_swa_full = std::any_of(p.extra_args.begin(), p.extra_args.end(), [](const std::string& arg) {
            return llamaserver_flag_name(arg) == "--swa-full";
        });
        if (!already_swa_full) {
            a.emplace_back("--swa-full");
        }
    }
    if (p.flash_attn) {
        a.emplace_back("--flash-attn");
        a.push_back(*p.flash_attn);
    }
    if (p.cache_type_k) {
        a.emplace_back("--cache-type-k");
        a.push_back(*p.cache_type_k);
    }
    if (p.cache_type_v) {
        a.emplace_back("--cache-type-v");
        a.push_back(*p.cache_type_v);
    }
    if (p.ctx_checkpoints) num("--ctx-checkpoints", *p.ctx_checkpoints);
    if (p.checkpoint_min_step) num("--checkpoint-min-step", *p.checkpoint_min_step);
    if (p.cache_ram_mib) num("--cache-ram", *p.cache_ram_mib);
    toggle(p.context_shift, "--context-shift", "--no-context-shift");
    if (p.fit) {
        a.emplace_back("--fit");
        a.emplace_back(*p.fit ? "on" : "off");
    }
    if (p.speculative) {
        std::string types;
        for (const auto& t : p.speculative->types) types += (types.empty() ? "" : ",") + t;
        a.emplace_back("--spec-type");
        a.push_back(types);
        if (p.speculative->draft_n_max) num("--spec-draft-n-max", *p.speculative->draft_n_max);
        if (!p.speculative->draft_model.empty()) {
            a.emplace_back("--spec-draft-model");
            a.push_back(p.speculative->draft_model);
        }
    }
    if (p.threads) num("--threads", *p.threads);
    if (p.threads_batch) num("--threads-batch", *p.threads_batch);
    toggle(p.jinja, "--jinja", "--no-jinja");
    if (p.reasoning_format) {
        a.emplace_back("--reasoning-format");
        a.push_back(*p.reasoning_format);
    }
    if (p.reasoning) {
        a.emplace_back("--reasoning");
        a.push_back(*p.reasoning);
    }
    if (p.reasoning_budget) num("--reasoning-budget", *p.reasoning_budget);
    a.insert(a.end(), p.extra_args.begin(), p.extra_args.end());
    return a;
}

Result<LlamaCppSpeculativeMapping> map_llamacpp_speculative(const SpeculativeSettings& spec,
                                                          std::string_view profile_where) {
    const std::string where = profile_where.empty() ? std::string("(direct)") : std::string(profile_where);
    if (spec.types.empty()) {
        return bad(where, "'speculative.types' must not be empty");
    }
    if (spec.types.size() > 1) {
        return bad(where, "'speculative.types' has multiple entries; the llamacpp backend supports only a single "
                          "speculative type (use llamaserver)");
    }
    if (!spec.draft_model.empty()) {
        return bad(where, "'speculative.draft_model' is not supported by the llamacpp backend "
                          "(direct backend supports in-process speculation only); use llamaserver");
    }
    const std::string& mode = spec.types.front();
    const bool supported =
        std::find(std::begin(kLlamaCppSpeculativeTypes), std::end(kLlamaCppSpeculativeTypes), mode) !=
        std::end(kLlamaCppSpeculativeTypes);
    if (!supported) {
        return bad(where, "'speculative.types' ('" + detail::bounded_for_message(mode) +
                              "') is not supported by the llamacpp backend; use llamaserver");
    }
    if (spec.draft_n_max) {
        if (*spec.draft_n_max < 1 || *spec.draft_n_max > 64) {
            return bad(where, "'speculative.draft_n_max' must be from 1 to 64");
        }
        if (mode == "none") {
            return bad(where, "'speculative.draft_n_max' cannot be set when speculative type is 'none'");
        }
    }
    return LlamaCppSpeculativeMapping{mode, spec.draft_n_max};
}

Status apply_llamacpp_profile(const LaunchProfile& p, BackendSetup& setup, std::string& device) {
    if (Status st = validate_launch_profile(p); !st.ok()) return st;
    if (p.backend != kLaunchProfileBackendLlamaCpp) return bad(where_of(p), "backend is not llamacpp");
    const auto unsupported = [&p](const char* field) {
        return bad(where_of(p), std::string("'") + field + "' is not supported by the llamacpp backend; use llamaserver");
    };
    if (p.mmproj_offload) return unsupported("mmproj_offload");
    if (p.image_max_tokens) return unsupported("image_max_tokens");
    if (p.device) return unsupported("device");
    if (p.main_gpu) return unsupported("main_gpu");
    if (p.split_mode) return unsupported("split_mode");
    if (!p.tensor_split.empty()) return unsupported("tensor_split");
    if (p.load_mode) return unsupported("load_mode");
    if (p.no_repack) return unsupported("no_repack");
    if (p.moe_cache_mib) return unsupported("moe_cache_mib");
    if (p.moe_cache_cpu_share) return unsupported("moe_cache_cpu_share");
    if (p.parallel && (*p.parallel < 1 || *p.parallel > 16)) {
        return bad(where_of(p), "'parallel' (" + std::to_string(*p.parallel) +
                                    ") must be from 1 to 16 on the llamacpp backend; use llamaserver");
    }
    if (p.cache_type_k && !is_llamacpp_kv_cache_type(*p.cache_type_k)) {
        return bad(where_of(p), "'cache_type_k' (" + *p.cache_type_k +
                                    ") is not supported by the llamacpp backend; use llamaserver");
    }
    if (p.cache_type_v && !is_llamacpp_kv_cache_type(*p.cache_type_v)) {
        return bad(where_of(p), "'cache_type_v' (" + *p.cache_type_v +
                                    ") is not supported by the llamacpp backend; use llamaserver");
    }
    if (p.ctx_checkpoints) return unsupported("ctx_checkpoints");
    if (p.checkpoint_min_step) return unsupported("checkpoint_min_step");
    if (p.cache_ram_mib && (*p.cache_ram_mib < 0 || *p.cache_ram_mib > 65536)) {
        return bad(where_of(p), "'cache_ram_mib' (" + std::to_string(*p.cache_ram_mib) +
                                    ") must be from 0 to 65536 on the llamacpp backend; use llamaserver");
    }
    if (p.context_shift) return unsupported("context_shift");
    if (p.fit) return unsupported("fit");
    std::optional<LlamaCppSpeculativeMapping> spec_mapping;
    if (p.speculative) {
        auto mapped = map_llamacpp_speculative(*p.speculative, where_of(p));
        if (!mapped.ok()) return mapped.status();
        spec_mapping = std::move(mapped).value();
    }
    if (p.threads && (*p.threads < 1 || *p.threads > 256)) {
        return bad(where_of(p), "'threads' (" + std::to_string(*p.threads) +
                                    ") must be from 1 to 256 on the llamacpp backend; use llamaserver");
    }
    if (p.threads_batch) return unsupported("threads_batch");
    if (p.jinja) return unsupported("jinja");
    if (p.reasoning_format) return unsupported("reasoning_format");
    if (p.reasoning) return unsupported("reasoning");
    if (p.reasoning_budget) return unsupported("reasoning_budget");
    if (!p.extra_args.empty()) return unsupported("extra_args");
    if (p.ctx_size && *p.ctx_size > (1u << 22)) {
        return bad(where_of(p), "'ctx_size' above 4194304 is not supported by the llamacpp backend; use llamaserver");
    }
    BackendSetup next = setup;
    // Same option names and values as serve's direct llamacpp CLI flags;
    // validate_options() checks them with validate_llamacpp_options() when built.
    if (p.batch_size) next.llamacpp_batch_size = *p.batch_size;
    if (p.ubatch_size) next.llamacpp_ubatch_size = *p.ubatch_size;
    if (p.cache_type_k) next.llamacpp_kv_cache_type_k = *p.cache_type_k;
    if (p.cache_type_v) next.llamacpp_kv_cache_type_v = *p.cache_type_v;
    if (p.flash_attn) next.llamacpp_flash_attention = *p.flash_attn;
    if (p.swa_kv_mode) next.llamacpp_swa_kv_mode = *p.swa_kv_mode;
    if (p.ctx_size) next.llamacpp_context_length = *p.ctx_size;
    if (p.parallel) next.llamacpp_slots = *p.parallel;
    if (p.kv_unified) next.llamacpp_kv_unified = *p.kv_unified;
    if (p.cache_ram_mib) next.llamacpp_cache_ram_mib = static_cast<std::uint32_t>(*p.cache_ram_mib);
    if (p.threads) next.llamacpp_threads = *p.threads;
    if (!p.mmproj.empty()) next.llamacpp_mmproj_path = p.mmproj;
    if (spec_mapping) {
        next.llamacpp_spec_type = spec_mapping->spec_type;
        if (spec_mapping->draft_max) next.llamacpp_spec_draft_max = *spec_mapping->draft_max;
    }
    std::string next_device = device;
    if (p.n_gpu_layers) {
        next.llamacpp_gpu_layers = *p.n_gpu_layers;
        // The backend offloads only for gpu:* devices; a cpu device would
        // silently ignore n_gpu_layers.
        if (*p.n_gpu_layers != 0) {
            if (next_device.empty()) {
                next_device = "gpu:0";
            } else if (next_device.rfind("gpu:", 0) != 0) {
                return bad(where_of(p), "'n_gpu_layers' needs a gpu:* --device (got " + next_device + ")");
            }
        }
    }
    setup = std::move(next);
    device = std::move(next_device);
    return {};
}

void apply_sampling_defaults(const SamplingDefaults& d, SamplingConfig& s) {
    const auto put = [&s](const auto& value, auto& field, SamplingConfig::Field flag) {
        if (value && !s.is_explicit(flag)) {
            field = *value;
            s.explicit_fields |= flag;
        }
    };
    put(d.temperature, s.temperature, SamplingConfig::kTemperature);
    put(d.top_p, s.top_p, SamplingConfig::kTopP);
    put(d.top_k, s.top_k, SamplingConfig::kTopK);
    put(d.min_p, s.min_p, SamplingConfig::kMinP);
    put(d.presence_penalty, s.presence_penalty, SamplingConfig::kPresencePenalty);
    put(d.repeat_penalty, s.repeat_penalty, SamplingConfig::kRepeatPenalty);
}

json::Object launch_profile_metadata(const LaunchProfile& p, const std::optional<VramEstimate>& estimate,
                                     std::optional<std::uint64_t> budget_mib) {
    // `capabilities` lists only what a request to this Sonder endpoint can
    // use. Speculative decoding is transparent to the caller. `vision` and
    // `tools` are what the upstream llama-server supports (an mmproj; jinja
    // templates, on by default), but Sonder's /v1/chat/completions rejects
    // tool definitions and non-string message content with
    // unsupported_parameter, so they are listed as `upstream_capabilities`
    // only. A consumer that reads `capabilities` never sends a request that
    // Sonder refuses.
    json::Array capabilities;
    json::Array upstream_capabilities;
    if (!p.mmproj.empty()) upstream_capabilities.emplace_back("vision");
    if (p.backend == kLaunchProfileBackendLlamaServer && p.jinja.value_or(true)) {
        upstream_capabilities.emplace_back("tools");
    }
    const bool speculative = p.speculative && !(p.speculative->types.size() == 1 && p.speculative->types[0] == "none");
    if (speculative) {
        capabilities.emplace_back("speculative");
        upstream_capabilities.emplace_back("speculative");
    }
    json::Object o{{"name", p.name},
                   {"backend", p.backend},
                   {"context_length", p.ctx_size && *p.ctx_size > 0 ? json::Value(*p.ctx_size) : json::Value()},
                   {"parallel", p.parallel ? json::Value(*p.parallel) : json::Value()},
                   {"cache_type_k", p.cache_type_k.value_or("f16")},
                   {"cache_type_v", p.cache_type_v.value_or("f16")},
                   {"flash_attn", p.flash_attn.value_or("auto")},
                   {"capabilities", std::move(capabilities)},
                   {"upstream_capabilities", std::move(upstream_capabilities)}};
    if (speculative) {
        json::Array types;
        for (const auto& t : p.speculative->types) types.emplace_back(t);
        o.set("speculative_types", std::move(types));
        if (p.speculative->draft_n_max) {
            o.set("speculative_draft_n_max", json::Value(*p.speculative->draft_n_max));
        }
    }
    if (p.kv_unified.has_value()) {
        o.set("kv_unified", json::Value(*p.kv_unified));
    }
    if (p.cache_ram_mib.has_value()) {
        o.set("cache_ram_mib", json::Value(*p.cache_ram_mib));
    }
    if (p.threads.has_value()) {
        o.set("threads", json::Value(*p.threads));
    }
    o.set("estimated_vram_mib", estimate ? json::Value(estimate->total_mib()) : json::Value());
    o.set("vram_budget_mib", budget_mib ? json::Value(*budget_mib) : json::Value());
    if (estimate && estimate->max_gpu_bytes > 0) {
        o.set("per_gpu_max_mib", json::Value(estimate->max_gpu_mib()));
    }
    if (estimate && (estimate->prism_hadamard || estimate->requires_prism_runtime)) {
        o.set("prism_hadamard", json::Value(estimate->prism_hadamard));
        o.set("requires_prism_runtime", json::Value(estimate->requires_prism_runtime));
    }
    if (p.moe_cache_mib) {
        o.set("moe_cache_mib", json::Value(*p.moe_cache_mib));
    }
    if (p.moe_cache_cpu_share) {
        o.set("moe_cache_cpu_share", json::Value(*p.moe_cache_cpu_share));
    }
    return o;
}

}  // namespace sonder::inference
