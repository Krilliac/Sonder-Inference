// Launch profiles: schema, validation, llama-server argv, llamacpp mapping,
// sampling defaults and /v1/models metadata. See launch_profile.hpp.
#include "sonder/inference/launch_profile.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
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

std::string where_of(const LaunchProfile& p) { return p.name.empty() ? std::string("(unnamed)") : "'" + p.name + "'"; }

bool valid_name(std::string_view name) {
    if (name.empty() || name.size() > 128 || name == "default") return false;
    return std::all_of(name.begin(), name.end(), [](char c) {
        return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '.' || c == '_' ||
               c == '-' || c == ':';
    });
}

bool has_nul(std::string_view s) { return s.find('\0') != std::string_view::npos; }

std::string canonical_flash_attn(std::string_view v) {
    if (v == "on" || v == "enabled") return "on";
    if (v == "off" || v == "disabled") return "off";
    if (v == "auto") return "auto";
    return {};
}

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
    {"--temp", "sampling.temperature"}, {"--temperature", "sampling.temperature"},
    {"--top-p", "sampling.top_p"}, {"--top-k", "sampling.top_k"}, {"--min-p", "sampling.min_p"},
    {"--presence-penalty", "sampling.presence_penalty"}, {"--repeat-penalty", "sampling.repeat_penalty"},
};

// Flags a profile may never pass: the supervisor owns the listen address,
// Sonder does not hand the child credentials, and profiles never download.
constexpr std::string_view kForbiddenFlags[] = {
    "--", "--host", "--port", "--api-key", "--api-key-file", "--ssl-key-file", "--ssl-cert-file",
    "-mu", "--model-url", "-hf", "-hfr", "--hf-repo", "-hff", "--hf-file", "-hft", "--hf-token",
    "-dr", "--docker-repo", "-mmu", "--mmproj-url", "-hfd", "-hfrd", "--hf-repo-draft", "--spec-draft-hf",
    "--models-dir", "--models-preset",
};

// llama-server features that run tools, reach MCP servers or read/serve local
// files (the `--help` of each says "do not enable in untrusted environments"
// or serves a directory). A profile is data that is meant to be shared, so it
// may not switch them on; the `--no-*` spellings stay allowed.
constexpr std::string_view kUnsafeFeatureFlags[] = {
    "--tools", "--tools-runtime", "--mcp-servers-config", "--mcp-servers-json", "-ag", "--agent",
    "--ui-mcp-proxy", "--webui-mcp-proxy", "--path", "--media-path",
};

// llama-server flags that write prompts or logs to an arbitrary local path.
// A shared profile must not be able to record user prompts to disk.
// `-lcd`/`--lookup-cache-dynamic` is an n-gram cache that generation updates,
// so it records prompt/response text too. `--slot-save-path` stays allowed on
// purpose: docs/integration/llama-server.md tells operators to pass it.
constexpr std::string_view kDiskWriteFlags[] = {
    "--log-file", "--log-prompts-dir", "-lcd", "--lookup-cache-dynamic",
};

// llama-server flags that move model work to other machines: `--rpc
// HOST:PORT,...` registers remote rpc-server devices, and layers placed on
// them send their tensors and every request's activations over the network.
// A shared profile must not route prompts off this host.
constexpr std::string_view kRemoteOffloadFlags[] = {"--rpc"};

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

Status check_extra_args(const LaunchProfile& p) {
    if (p.extra_args.size() > kMaxExtraArgs) return bad(where_of(p), "extra_args has more than 256 entries");
    for (const auto& arg : p.extra_args) {
        if (has_nul(arg)) return bad(where_of(p), "extra_args must not contain NUL");
        if (arg.size() < 2 || arg[0] != '-') continue;  // a value, not a flag
        const std::string_view flag = std::string_view(arg).substr(0, arg.find('='));
        for (auto f : kForbiddenFlags) {
            if (flag == f || (f.size() > 2 && (flag.rfind(std::string(f) + "_", 0) == 0))) {
                return bad(where_of(p), "extra_args must not contain " + std::string(flag) +
                                            " (the supervisor owns host and port; credentials and downloads are "
                                            "not passed to llama-server)");
            }
        }
        if (is_model_preset_flag(flag)) {
            return bad(where_of(p), "extra_args must not contain " + std::string(flag) +
                                        " (llama-server's built-in model presets replace the profile's model and can "
                                        "download weights; set the typed fields instead)");
        }
        for (auto f : kDiskWriteFlags) {
            if (flag == f) {
                return bad(where_of(p), "extra_args must not contain " + std::string(flag) +
                                            " (a launch profile cannot make llama-server write logs or prompts to "
                                            "disk)");
            }
        }
        for (auto f : kRemoteOffloadFlags) {
            if (flag == f) {
                return bad(where_of(p), "extra_args must not contain " + std::string(flag) +
                                            " (a launch profile cannot offload model work, and the activations "
                                            "that carry each prompt, to remote RPC servers)");
            }
        }
        for (auto f : kUnsafeFeatureFlags) {
            if (flag == f) {
                return bad(where_of(p), "extra_args must not contain " + std::string(flag) +
                                            " (llama-server's agent tools, MCP and local file serving cannot be "
                                            "enabled from a launch profile)");
            }
        }
        for (const auto& [f, field] : kTypedFlags) {
            if (flag == f) {
                return bad(where_of(p), "extra_args must not contain " + std::string(flag) + "; use the typed field '" +
                                            std::string(field) + "'");
            }
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
                return bad(where_, "unknown field '" + member.first + "'");
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
                             "n_gpu_layers", "batch_size", "ubatch_size", "parallel", "kv_unified", "flash_attn",
                             "cache_type_k", "cache_type_v", "ctx_checkpoints", "checkpoint_min_step", "cache_ram_mib",
                             "context_shift", "fit", "speculative", "threads", "threads_batch", "jinja",
                             "reasoning_format", "sampling", "extra_args", "vram_budget_mib"}));
    LaunchProfile p;
    SONDER_TRY(r.string("name", p.name, true));
    SONDER_TRY(r.string("backend", p.backend, true));
    SONDER_TRY(r.string("model", p.model, true));
    SONDER_TRY(r.string("mmproj", p.mmproj));
    SONDER_TRY(r.boolean("mmproj_offload", p.mmproj_offload));
    SONDER_TRY(r.integer("image_max_tokens", p.image_max_tokens, 1, 65536));
    SONDER_TRY(r.integer("ctx_size", p.ctx_size, 0, kMaxContext));
    SONDER_TRY(r.integer("n_gpu_layers", p.n_gpu_layers, -1, 100000));
    SONDER_TRY(r.integer("batch_size", p.batch_size, 1, kMaxBatch));
    SONDER_TRY(r.integer("ubatch_size", p.ubatch_size, 1, kMaxBatch));
    SONDER_TRY(r.integer("parallel", p.parallel, 1, 256));
    SONDER_TRY(r.boolean("kv_unified", p.kv_unified));
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

double kv_cache_type_bytes(std::string_view type) noexcept {
    if (type == "f32") return 4.0;
    if (type == "f16" || type == "bf16") return 2.0;
    if (type == "q8_0") return 34.0 / 32.0;
    if (type == "q5_1") return 24.0 / 32.0;
    if (type == "q5_0") return 22.0 / 32.0;
    if (type == "q4_1") return 20.0 / 32.0;
    if (type == "q4_0" || type == "iq4_nl") return 18.0 / 32.0;
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
    if (p.model.empty() || has_nul(p.model)) return bad(where, "'model' must be a GGUF path");
    if (has_nul(p.mmproj)) return bad(where, "'mmproj' must not contain NUL");
    if (p.mmproj.empty() && (p.mmproj_offload || p.image_max_tokens)) {
        return bad(where, "'mmproj_offload' and 'image_max_tokens' need 'mmproj'");
    }
    if (p.ctx_size && *p.ctx_size > kMaxContext) return bad(where, "'ctx_size' is out of range");
    if (p.n_gpu_layers && *p.n_gpu_layers < -1) return bad(where, "'n_gpu_layers' must be -1 or more");
    if (p.flash_attn && canonical_flash_attn(*p.flash_attn) != *p.flash_attn) {
        return bad(where, "'flash_attn' must be on, off or auto");
    }
    for (const auto* t : {&p.cache_type_k, &p.cache_type_v}) {
        if (*t && kv_cache_type_bytes(**t) == 0.0) {
            return bad(where, std::string(t == &p.cache_type_k ? "'cache_type_k'" : "'cache_type_v'") +
                                  " must be one of f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1");
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
    if (p.speculative) {
        if (p.speculative->types.empty()) return bad(where, "'speculative.types' must not be empty");
        std::set<std::string> seen;
        for (const auto& t : p.speculative->types) {
            if (std::find(std::begin(kSpeculativeTypes), std::end(kSpeculativeTypes), t) == std::end(kSpeculativeTypes)) {
                return bad(where, "unknown speculative type '" + t + "'");
            }
            if (!seen.insert(t).second) return bad(where, "speculative type '" + t + "' is listed twice");
        }
        if (seen.count("none") && seen.size() > 1) return bad(where, "speculative type 'none' cannot be combined");
        if (has_nul(p.speculative->draft_model)) return bad(where, "'speculative.draft_model' must not contain NUL");
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
        if (member.first != "profiles") return bad("file", "unknown field '" + member.first + "'");
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
            return bad("'" + parsed.value().name + "'", "the name is used by more than one profile");
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
    if (p.batch_size) num("--batch-size", *p.batch_size);
    if (p.ubatch_size) num("--ubatch-size", *p.ubatch_size);
    if (p.parallel) num("--parallel", *p.parallel);
    toggle(p.kv_unified, "--kv-unified", "--no-kv-unified");
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
    a.insert(a.end(), p.extra_args.begin(), p.extra_args.end());
    return a;
}

Status apply_llamacpp_profile(const LaunchProfile& p, BackendSetup& setup, std::string& device) {
    if (Status st = validate_launch_profile(p); !st.ok()) return st;
    if (p.backend != kLaunchProfileBackendLlamaCpp) return bad(where_of(p), "backend is not llamacpp");
    const auto unsupported = [&p](const char* field) {
        return bad(where_of(p), std::string("'") + field + "' is not supported by the llamacpp backend; use llamaserver");
    };
    if (!p.mmproj.empty()) return unsupported("mmproj");
    if (p.mmproj_offload) return unsupported("mmproj_offload");
    if (p.image_max_tokens) return unsupported("image_max_tokens");
    if (p.parallel) return unsupported("parallel");
    if (p.kv_unified) return unsupported("kv_unified");
    if (p.ctx_checkpoints) return unsupported("ctx_checkpoints");
    if (p.checkpoint_min_step) return unsupported("checkpoint_min_step");
    if (p.cache_ram_mib) return unsupported("cache_ram_mib");
    if (p.context_shift) return unsupported("context_shift");
    if (p.fit) return unsupported("fit");
    if (p.speculative) return unsupported("speculative");
    if (p.threads) return unsupported("threads");
    if (p.threads_batch) return unsupported("threads_batch");
    if (p.jinja) return unsupported("jinja");
    if (p.reasoning_format) return unsupported("reasoning_format");
    if (!p.extra_args.empty()) return unsupported("extra_args");
    if (p.ctx_size && *p.ctx_size > (1u << 22)) {
        return bad(where_of(p), "'ctx_size' above 4194304 is not supported by the llamacpp backend; use llamaserver");
    }
    BackendSetup next = setup;
    // Same option names and values as serve's --batch-size, --ubatch-size,
    // --cache-type-k/-v and --flash-attn; validate_options() checks them
    // with validate_llamacpp_options() in builds that have the backend.
    if (p.batch_size) next.llamacpp_batch_size = *p.batch_size;
    if (p.ubatch_size) next.llamacpp_ubatch_size = *p.ubatch_size;
    if (p.cache_type_k) next.llamacpp_kv_cache_type_k = *p.cache_type_k;
    if (p.cache_type_v) next.llamacpp_kv_cache_type_v = *p.cache_type_v;
    if (p.flash_attn) next.llamacpp_flash_attention = *p.flash_attn;
    if (p.ctx_size) next.llamacpp_context_length = *p.ctx_size;
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
    }
    o.set("estimated_vram_mib", estimate ? json::Value(estimate->total_mib()) : json::Value());
    o.set("vram_budget_mib", budget_mib ? json::Value(*budget_mib) : json::Value());
    return o;
}

}  // namespace sonder::inference
