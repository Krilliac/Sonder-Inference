// Sonder Inference - LoRA adapter registry and runtime lifecycle manager.
#include "lora_registry.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

#include "llama.h"

namespace sonder::backends::llamacpp {

namespace {

namespace fs = std::filesystem;

bool IsValidUtf8(std::string_view s) noexcept {
    std::size_t i = 0;
    const std::size_t n = s.size();
    while (i < n) {
        const auto c = static_cast<unsigned char>(s[i]);
        if (c <= 0x7Fu) {
            ++i;
            continue;
        }
        if (c >= 0xC2u && c <= 0xDFu) {
            if (i + 1 >= n) return false;
            const auto c1 = static_cast<unsigned char>(s[i + 1]);
            if ((c1 & 0xC0u) != 0x80u) return false;
            i += 2;
            continue;
        }
        if (c >= 0xE0u && c <= 0xEFu) {
            if (i + 2 >= n) return false;
            const auto c1 = static_cast<unsigned char>(s[i + 1]);
            const auto c2 = static_cast<unsigned char>(s[i + 2]);
            if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u) return false;
            if (c == 0xE0u && c1 < 0xA0u) return false;  // overlong
            if (c == 0xEDu && c1 >= 0xA0u) return false;  // UTF-16 surrogate U+D800..U+DFFF
            i += 3;
            continue;
        }
        if (c >= 0xF0u && c <= 0xF4u) {
            if (i + 3 >= n) return false;
            const auto c1 = static_cast<unsigned char>(s[i + 1]);
            const auto c2 = static_cast<unsigned char>(s[i + 2]);
            const auto c3 = static_cast<unsigned char>(s[i + 3]);
            if ((c1 & 0xC0u) != 0x80u || (c2 & 0xC0u) != 0x80u || (c3 & 0xC0u) != 0x80u) return false;
            if (c == 0xF0u && c1 < 0x90u) return false;  // overlong
            if (c == 0xF4u && c1 > 0x8Fu) return false;  // > U+10FFFF
            i += 4;
            continue;
        }
        return false;
    }
    return true;
}

bool TryPathFromUtf8(std::string_view s, fs::path& out) noexcept {
    out.clear();
    if (!IsValidUtf8(s)) return false;
    try {
        out = fs::path(std::u8string(s.begin(), s.end()));
        return true;
    } catch (...) {
        out.clear();
        return false;
    }
}

std::string PathToUtf8(const fs::path& p) {
    try {
        const std::u8string u = p.u8string();
        return std::string(u.begin(), u.end());
    } catch (...) {
        return {};
    }
}

std::string LowerAscii(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char ch : s) {
        if (ch >= 'A' && ch <= 'Z') {
            out.push_back(static_cast<char>(ch - 'A' + 'a'));
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

bool EqualsIgnoreCaseAscii(std::string_view a, std::string_view b) noexcept {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const auto ca = static_cast<unsigned char>(a[i]);
        const auto cb = static_cast<unsigned char>(b[i]);
        const auto la = (ca >= 'A' && ca <= 'Z') ? static_cast<unsigned char>(ca - 'A' + 'a') : ca;
        const auto lb = (cb >= 'A' && cb <= 'Z') ? static_cast<unsigned char>(cb - 'A' + 'a') : cb;
        if (la != lb) return false;
    }
    return true;
}

bool HasEmbeddedNul(std::string_view s) noexcept {
    return s.find('\0') != std::string_view::npos;
}

bool HasLeadingOrTrailingWhitespace(std::string_view s) noexcept {
    if (s.empty()) return false;
    return std::isspace(static_cast<unsigned char>(s.front())) != 0 ||
           std::isspace(static_cast<unsigned char>(s.back())) != 0;
}

// Splits `p` into non-empty lexical components for prefix containment checks.
std::vector<std::string> SplitNormalizedComponents(const fs::path& p) {
    std::vector<std::string> parts;
    for (const auto& comp : p) {
        std::string s = PathToUtf8(comp);
        if (s.empty() || s == ".") continue;
#if defined(_WIN32)
        s = LowerAscii(s);
#endif
        parts.push_back(std::move(s));
    }
    return parts;
}

// Returns true iff `canonical_file` lies strictly inside `canonical_root`
// (component-wise prefix match so sibling prefix collisions like
// `models_evil` vs `models` are rejected).
bool IsStrictlyUnderRoot(const fs::path& canonical_file, const fs::path& canonical_root) {
    const std::vector<std::string> file_parts = SplitNormalizedComponents(canonical_file);
    const std::vector<std::string> root_parts = SplitNormalizedComponents(canonical_root);
    if (root_parts.empty() || file_parts.size() <= root_parts.size()) {
        return false;
    }
    for (std::size_t i = 0; i < root_parts.size(); ++i) {
        if (file_parts[i] != root_parts[i]) return false;
    }
    return true;
}

bool HasDotDotSegment(std::string_view raw_path) noexcept {
    // Check both POSIX '/' and Windows '\' separators explicitly so a backslash
    // traversal string is refused on every platform.
    std::size_t pos = 0;
    while (pos <= raw_path.size()) {
        std::size_t end = raw_path.find_first_of("/\\", pos);
        if (end == std::string_view::npos) end = raw_path.size();
        if (raw_path.substr(pos, end - pos) == "..") {
            return true;
        }
        if (end == raw_path.size()) break;
        pos = end + 1;
    }
    return false;
}

std::string NormalizePathKey(std::string_view canonical_path) {
#if defined(_WIN32)
    return LowerAscii(canonical_path);
#else
    return std::string(canonical_path);
#endif
}

}  // namespace

Status ValidateLoraScale(float scale) {
    if (!std::isfinite(scale)) {
        return Status::Error(ErrorCode::kInvalidArgument, "LoRA adapter scale must be finite");
    }
    if (scale < kMinLoraScale || scale > kMaxLoraScale) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter scale " + std::to_string(scale) + " is outside [" +
                                 std::to_string(kMinLoraScale) + ", " + std::to_string(kMaxLoraScale) + "]");
    }
    return Status::Ok();
}

Status ValidateLoraPathLexical(std::string_view raw_path) {
    if (raw_path.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "LoRA adapter path is empty");
    }
    if (HasEmbeddedNul(raw_path)) {
        return Status::Error(ErrorCode::kInvalidArgument, "LoRA adapter path contains embedded NUL byte");
    }
    if (HasDotDotSegment(raw_path)) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter path must not contain '..' path traversal: " + std::string(raw_path));
    }
    fs::path p;
    if (!TryPathFromUtf8(raw_path, p)) {
        return Status::Error(ErrorCode::kInvalidArgument, "LoRA adapter path is not valid UTF-8");
    }
    for (const auto& part : p) {
        if (part == "..") {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "LoRA adapter path must not contain '..' path traversal: " + std::string(raw_path));
        }
    }
#if defined(_WIN32)
    // Reject drive-relative paths (e.g. "C:adapter.gguf") which depend on per-drive CWD.
    if (p.has_root_name() && !p.is_absolute()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "drive-relative LoRA adapter path is not permitted: " + std::string(raw_path));
    }
    // Reject NTFS alternate data stream syntax (':' anywhere outside drive-letter position 1).
    for (std::size_t i = 0; i < raw_path.size(); ++i) {
        if (raw_path[i] == ':') {
            const bool is_drive_colon =
                (i == 1) && std::isalpha(static_cast<unsigned char>(raw_path[0])) != 0;
            if (!is_drive_colon) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "LoRA adapter path contains invalid ':' character: " + std::string(raw_path));
            }
        }
    }
#endif
    return Status::Ok();
}

Status ValidateLoraAdapterPath(std::string_view raw_path, const std::vector<std::string>& model_roots,
                               std::string& canonical_out) {
    canonical_out.clear();
    if (Status s = ValidateLoraPathLexical(raw_path); !s.ok()) {
        return s;
    }
    if (model_roots.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "no configured model roots available to validate LoRA adapter path");
    }

    // Canonicalize valid configured root directories.
    std::vector<fs::path> canonical_roots;
    canonical_roots.reserve(model_roots.size());
    for (const std::string& root : model_roots) {
        if (root.empty() || HasEmbeddedNul(root)) continue;
        fs::path rp;
        if (!TryPathFromUtf8(root, rp)) continue;
        std::error_code ec;
        if (!fs::is_directory(rp, ec)) continue;
        const fs::path canon_root = fs::canonical(rp, ec);
        if (!ec) {
            canonical_roots.push_back(canon_root);
        }
    }
    if (canonical_roots.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "no existing configured model root directories for LoRA adapter validation");
    }

    fs::path input_path;
    if (!TryPathFromUtf8(raw_path, input_path)) {
        return Status::Error(ErrorCode::kInvalidArgument, "LoRA adapter path is not valid UTF-8");
    }
    std::vector<fs::path> candidates;
    if (input_path.is_absolute()) {
        candidates.push_back(input_path);
    } else {
        candidates.reserve(canonical_roots.size());
        for (const fs::path& root : canonical_roots) {
            candidates.push_back(root / input_path);
        }
    }

    bool saw_existing_non_regular = false;
    bool saw_escaped_regular = false;
    for (const fs::path& cand : candidates) {
        std::error_code ec;
        const fs::file_status st = fs::status(cand, ec);
        if (ec || !fs::exists(st)) {
            continue;
        }
        if (!fs::is_regular_file(st)) {
            saw_existing_non_regular = true;
            continue;
        }
        const fs::path canon_file = fs::canonical(cand, ec);
        if (ec) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "failed to canonicalize LoRA adapter path: " + std::string(raw_path));
        }
        // Re-verify regular file after canonicalization (follows symlinks).
        if (!fs::is_regular_file(canon_file, ec)) {
            saw_existing_non_regular = true;
            continue;
        }
        for (const fs::path& root : canonical_roots) {
            if (IsStrictlyUnderRoot(canon_file, root)) {
                canonical_out = PathToUtf8(canon_file);
                return Status::Ok();
            }
        }
        saw_escaped_regular = true;
    }

    if (saw_escaped_regular) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter path escapes configured model roots: " + std::string(raw_path));
    }
    if (saw_existing_non_regular) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter path is not a regular file: " + std::string(raw_path));
    }
    return Status::Error(ErrorCode::kFileNotFound, "LoRA adapter file not found: " + std::string(raw_path));
}

Status ValidateLoraCompatibility(const LoraModelMetadata& base_meta, const LoraAdapterMetadata& adapter_meta,
                                 std::string_view adapter_label) {
    const std::string suffix = adapter_label.empty() ? std::string() : (" (" + std::string(adapter_label) + ")");

    if (adapter_meta.general_type != "adapter") {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter metadata general.type must be 'adapter', got '" +
                                 adapter_meta.general_type + "'" + suffix);
    }
    if (adapter_meta.adapter_type != "lora") {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter metadata adapter.type must be 'lora', got '" +
                                 adapter_meta.adapter_type + "'" + suffix);
    }
    if (base_meta.architecture.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "loaded base model has empty general.architecture metadata" + suffix);
    }
    if (adapter_meta.architecture.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter has empty general.architecture metadata" + suffix);
    }
    if (adapter_meta.architecture != base_meta.architecture) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA adapter general.architecture '" + adapter_meta.architecture +
                                 "' does not match base model architecture '" + base_meta.architecture + "'" + suffix);
    }
    if (adapter_meta.alpha.has_value()) {
        if (!std::isfinite(*adapter_meta.alpha) || *adapter_meta.alpha < 0.0F) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "LoRA adapter metadata adapter.lora.alpha must be finite and >= 0" + suffix);
        }
    }

    // If the adapter explicitly names its base model and the loaded base model
    // carries identifying metadata, require a match to catch same-architecture
    // mismatches (e.g. Qwen2.5-14B adapter on a Qwen2.5-7B base).
    if (!adapter_meta.base_model_name.empty()) {
        const bool base_has_identity =
            !base_meta.base_model_name.empty() || !base_meta.basename.empty() || !base_meta.name.empty();
        if (base_has_identity) {
            const bool matches =
                (!base_meta.base_model_name.empty() &&
                 EqualsIgnoreCaseAscii(adapter_meta.base_model_name, base_meta.base_model_name)) ||
                (!base_meta.basename.empty() &&
                 EqualsIgnoreCaseAscii(adapter_meta.base_model_name, base_meta.basename)) ||
                (!base_meta.name.empty() &&
                 EqualsIgnoreCaseAscii(adapter_meta.base_model_name, base_meta.name));
            if (!matches) {
                const std::string base_desc = !base_meta.base_model_name.empty()
                                                  ? base_meta.base_model_name
                                                  : (!base_meta.name.empty() ? base_meta.name : base_meta.basename);
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "LoRA adapter base model '" + adapter_meta.base_model_name +
                                         "' does not match loaded base model '" + base_desc + "'" + suffix);
            }
        }
    }
    return Status::Ok();
}

bool BaseModelMatches(std::string_view alias_base_model, std::string_view loaded_model_id) {
    if (alias_base_model.empty() || loaded_model_id.empty()) return false;
    if (!IsValidUtf8(alias_base_model) || !IsValidUtf8(loaded_model_id)) return false;
    if (alias_base_model == loaded_model_id) return true;
#if defined(_WIN32)
    if (EqualsIgnoreCaseAscii(alias_base_model, loaded_model_id)) return true;
#endif

    fs::path alias_path;
    fs::path loaded_path;
    if (!TryPathFromUtf8(alias_base_model, alias_path) || !TryPathFromUtf8(loaded_model_id, loaded_path)) {
        return false;
    }

    std::error_code ec1;
    std::error_code ec2;
    const fs::path c1 = fs::canonical(alias_path, ec1);
    const fs::path c2 = fs::canonical(loaded_path, ec2);
    if (!ec1 && !ec2) {
        return NormalizePathKey(PathToUtf8(c1)) == NormalizePathKey(PathToUtf8(c2));
    }

    // Allow a bare filename in either identifier (e.g. "base-qwen.gguf") to
    // match the filename of a full path, while keeping two distinct directory
    // paths separate.
    if (alias_path.parent_path().empty() || loaded_path.parent_path().empty()) {
        const std::string f1 = PathToUtf8(alias_path.filename());
        const std::string f2 = PathToUtf8(loaded_path.filename());
        if (!f1.empty() && !f2.empty()) {
            return NormalizePathKey(f1) == NormalizePathKey(f2);
        }
    }
    return false;
}

Status ValidateLoraAliasSyntax(const LoraAliasEntry& entry) {
    if (entry.alias.empty() || HasEmbeddedNul(entry.alias) || !IsValidUtf8(entry.alias) ||
        HasLeadingOrTrailingWhitespace(entry.alias)) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA alias name must be non-empty valid UTF-8 without leading/trailing whitespace or NUL");
    }
    if (entry.base_model.empty() || HasEmbeddedNul(entry.base_model) || !IsValidUtf8(entry.base_model)) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA alias '" + entry.alias + "' must specify a non-empty valid UTF-8 base_model");
    }
    if (entry.adapters.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA alias '" + entry.alias + "' must specify at least one adapter");
    }
    if (entry.adapters.size() > kMaxLoraAdaptersPerRequest) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "LoRA alias '" + entry.alias + "' exceeds max adapters (" +
                                 std::to_string(kMaxLoraAdaptersPerRequest) + ")");
    }
    for (const LoraAdapterSpec& spec : entry.adapters) {
        if (Status s = ValidateLoraPathLexical(spec.path); !s.ok()) {
            return s;
        }
        if (Status s = ValidateLoraScale(spec.scale); !s.ok()) {
            return s;
        }
    }
    return Status::Ok();
}

Status ValidateLoraAliasOptions(std::span<const sonder::inference::LlamaCppLoraAliasOption> aliases) {
    if (aliases.empty()) return Status::Ok();
    std::unordered_set<std::string> seen_aliases;
    seen_aliases.reserve(aliases.size());
    for (const auto& opt_alias : aliases) {
        LoraAliasEntry entry;
        entry.alias = opt_alias.alias;
        entry.base_model = opt_alias.base_model;
        entry.adapters.reserve(opt_alias.adapters.size());
        for (const auto& a : opt_alias.adapters) {
            entry.adapters.push_back(LoraAdapterSpec{a.path, a.scale});
        }
        if (Status s = ValidateLoraAliasSyntax(entry); !s.ok()) {
            return s;
        }
        if (!seen_aliases.insert(entry.alias).second) {
            return Status::Error(ErrorCode::kInvalidArgument, "duplicate LoRA alias: " + entry.alias);
        }
    }
    return Status::Ok();
}

Status ResolveConfiguredLoraAlias(std::string_view requested_model,
                                  std::span<const sonder::inference::LlamaCppLoraAliasOption> aliases,
                                  const std::vector<std::string>& lora_roots,
                                  const std::vector<std::string>& model_dirs,
                                  std::optional<LoraAliasEntry>& matched_out,
                                  std::vector<std::string>& effective_roots_out) {
    matched_out.reset();
    effective_roots_out.clear();
    for (const auto& alias_opt : aliases) {
        if (alias_opt.alias != requested_model) continue;
        effective_roots_out = lora_roots.empty() ? model_dirs : lora_roots;
        LoraRegistry temp_reg(effective_roots_out);
        LoraAliasEntry candidate;
        candidate.alias = alias_opt.alias;
        candidate.base_model = alias_opt.base_model;
        candidate.adapters.reserve(alias_opt.adapters.size());
        for (const auto& a : alias_opt.adapters) {
            candidate.adapters.push_back(LoraAdapterSpec{a.path, a.scale});
        }
        if (Status reg_s = temp_reg.RegisterAlias(candidate); !reg_s.ok()) {
            effective_roots_out.clear();
            return reg_s;
        }
        LoraAliasEntry resolved_alias;
        if (Status res_s = temp_reg.ResolveAlias(alias_opt.alias, resolved_alias); !res_s.ok()) {
            effective_roots_out.clear();
            return res_s;
        }
        matched_out = std::move(resolved_alias);
        return Status::Ok();
    }
    return Status::Ok();
}

// ---- LoraRegistry ----

LoraRegistry::LoraRegistry(std::vector<std::string> model_roots) : model_roots_(std::move(model_roots)) {}

void LoraRegistry::SetModelRoots(std::vector<std::string> model_roots) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_roots_ = std::move(model_roots);
}

std::vector<std::string> LoraRegistry::ModelRoots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return model_roots_;
}

Status LoraRegistry::RegisterAlias(const LoraAliasEntry& entry) {
    if (Status s = ValidateLoraAliasSyntax(entry); !s.ok()) {
        return s;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    if (aliases_.find(entry.alias) != aliases_.end()) {
        return Status::Error(ErrorCode::kInvalidArgument, "LoRA alias already registered: " + entry.alias);
    }

    LoraAliasEntry canonical_entry;
    canonical_entry.alias = entry.alias;
    canonical_entry.base_model = entry.base_model;
    canonical_entry.adapters.reserve(entry.adapters.size());

    std::unordered_set<std::string> seen_paths;
    for (const LoraAdapterSpec& spec : entry.adapters) {
        std::string canonical_path;
        if (Status s = ValidateLoraAdapterPath(spec.path, model_roots_, canonical_path); !s.ok()) {
            return s;
        }
        const std::string key = NormalizePathKey(canonical_path);
        if (!seen_paths.insert(key).second) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "duplicate LoRA adapter path in alias '" + entry.alias + "': " + canonical_path);
        }
        canonical_entry.adapters.push_back(LoraAdapterSpec{std::move(canonical_path), spec.scale});
    }

    aliases_.emplace(canonical_entry.alias, std::move(canonical_entry));
    return Status::Ok();
}

bool LoraRegistry::UnregisterAlias(std::string_view alias) {
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = aliases_.find(std::string(alias));
    if (it == aliases_.end()) return false;
    aliases_.erase(it);
    return true;
}

bool LoraRegistry::HasAlias(std::string_view alias) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return aliases_.find(std::string(alias)) != aliases_.end();
}

Status LoraRegistry::ResolveAlias(std::string_view alias, LoraAliasEntry& out) const {
    if (alias.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "LoRA alias name is empty");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    const auto it = aliases_.find(std::string(alias));
    if (it == aliases_.end()) {
        return Status::Error(ErrorCode::kFileNotFound, "unknown LoRA alias: " + std::string(alias));
    }
    out = it->second;
    return Status::Ok();
}

std::vector<LoraAliasEntry> LoraRegistry::ListAliases() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<LoraAliasEntry> out;
    out.reserve(aliases_.size());
    for (const auto& [name, entry] : aliases_) {
        out.push_back(entry);
    }
    std::sort(out.begin(), out.end(),
              [](const LoraAliasEntry& a, const LoraAliasEntry& b) { return a.alias < b.alias; });
    return out;
}

void LoraRegistry::ClearAliases() {
    std::lock_guard<std::mutex> lock(mutex_);
    aliases_.clear();
}

// ---- LoraAdapterManager ----

LoraAdapterManager::LoraAdapterManager(LoraRuntimeOps* ops, std::vector<std::string> model_roots)
    : ops_(ops), model_roots_(std::move(model_roots)) {}

LoraAdapterManager::~LoraAdapterManager() { UnloadAll(); }

void LoraAdapterManager::SetModelRoots(std::vector<std::string> model_roots) {
    std::lock_guard<std::mutex> lock(mutex_);
    model_roots_ = std::move(model_roots);
}

std::vector<std::string> LoraAdapterManager::ModelRoots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return model_roots_;
}

Status LoraAdapterManager::ApplyAdapters(const std::vector<LoraAdapterSpec>& adapters) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ops_ == nullptr) {
        return Status::Error(ErrorCode::kNotLoaded, "LoRA runtime ops not initialized");
    }
    if (adapters.empty()) {
        if (Status s = ops_->SetContextAdapters({}); !s.ok()) {
            return s;
        }
        active_adapters_.clear();
        return Status::Ok();
    }
    if (adapters.size() > kMaxLoraAdaptersPerRequest) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "too many LoRA adapters in request (max " +
                                 std::to_string(kMaxLoraAdaptersPerRequest) + ")");
    }

    // Validate scales, paths, root containment, and uniqueness upfront.
    std::vector<LoraAdapterSpec> canonical_specs;
    canonical_specs.reserve(adapters.size());
    std::unordered_set<std::string> seen_keys;
    for (const LoraAdapterSpec& spec : adapters) {
        if (Status s = ValidateLoraScale(spec.scale); !s.ok()) {
            return s;
        }
        std::string canonical_path;
        if (Status s = ValidateLoraAdapterPath(spec.path, model_roots_, canonical_path); !s.ok()) {
            return s;
        }
        const std::string key = NormalizePathKey(canonical_path);
        if (!seen_keys.insert(key).second) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "duplicate LoRA adapter in request: " + canonical_path);
        }
        canonical_specs.push_back(LoraAdapterSpec{std::move(canonical_path), spec.scale});
    }

    LoraModelMetadata base_meta;
    if (Status s = ops_->ReadBaseModelMetadata(base_meta); !s.ok()) {
        return s;
    }

    std::vector<std::pair<std::string, LoraAdapterHandle>> newly_loaded;
    auto rollback_newly_loaded = [&]() noexcept {
        for (const auto& [path_key, handle] : newly_loaded) {
            if (handle != nullptr) {
                ops_->FreeAdapter(handle);
            }
        }
        newly_loaded.clear();
    };

    std::vector<ActiveLoraBinding> bindings;
    bindings.reserve(canonical_specs.size());

    for (const LoraAdapterSpec& spec : canonical_specs) {
        const std::string key = NormalizePathKey(spec.path);
        LoraAdapterHandle handle = nullptr;
        if (const auto it = loaded_adapters_.find(key); it != loaded_adapters_.end()) {
            handle = it->second;
        } else {
            LoraAdapterMetadata pre_meta;
            if (Status s = ops_->ReadAdapterMetadata(spec.path, pre_meta); !s.ok()) {
                rollback_newly_loaded();
                return s;
            }
            if (Status s = ValidateLoraCompatibility(base_meta, pre_meta, spec.path); !s.ok()) {
                rollback_newly_loaded();
                return s;
            }
            if (Status s = ops_->InitAdapter(spec.path, handle); !s.ok()) {
                rollback_newly_loaded();
                return s;
            }
            if (handle == nullptr) {
                rollback_newly_loaded();
                return Status::Error(ErrorCode::kLoadFailed,
                                     "llama_adapter_lora_init returned null handle for: " + spec.path);
            }
            newly_loaded.emplace_back(key, handle);

            // Defense in depth: re-check metadata exposed by the loaded adapter handle.
            LoraAdapterMetadata post_meta;
            const Status meta_s = ops_->ReadLoadedAdapterMetadata(handle, post_meta);
            if (meta_s.ok()) {
                if (Status compat_s = ValidateLoraCompatibility(base_meta, post_meta, spec.path); !compat_s.ok()) {
                    rollback_newly_loaded();
                    return compat_s;
                }
            } else if (meta_s.code != ErrorCode::kUnsupported) {
                rollback_newly_loaded();
                return meta_s;
            }
        }
        bindings.push_back(ActiveLoraBinding{handle, spec.scale});
    }

    if (Status s = ops_->SetContextAdapters(bindings); !s.ok()) {
        rollback_newly_loaded();
        return s;
    }

    for (auto& [key, handle] : newly_loaded) {
        loaded_adapters_.emplace(std::move(key), handle);
    }
    active_adapters_ = std::move(canonical_specs);
    return Status::Ok();
}

Status LoraAdapterManager::ApplyAlias(const LoraRegistry& registry, std::string_view alias) {
    LoraAliasEntry entry;
    if (Status s = registry.ResolveAlias(alias, entry); !s.ok()) {
        return s;
    }
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (ops_ == nullptr) {
            return Status::Error(ErrorCode::kNotLoaded, "LoRA runtime ops not initialized");
        }
        LoraModelMetadata base_meta;
        if (Status s = ops_->ReadBaseModelMetadata(base_meta); !s.ok()) {
            return s;
        }
        if (!BaseModelMatches(entry.base_model, base_meta.model_id)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "LoRA alias '" + entry.alias + "' targets base model '" + entry.base_model +
                                     "', which does not match loaded model '" + base_meta.model_id + "'");
        }
    }
    return ApplyAdapters(entry.adapters);
}

Status LoraAdapterManager::ClearActive() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ops_ == nullptr) {
        return Status::Error(ErrorCode::kNotLoaded, "LoRA runtime ops not initialized");
    }
    if (Status s = ops_->SetContextAdapters({}); !s.ok()) {
        return s;
    }
    active_adapters_.clear();
    return Status::Ok();
}

Status LoraAdapterManager::UnloadAdapter(std::string_view path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ops_ == nullptr) {
        return Status::Error(ErrorCode::kNotLoaded, "LoRA runtime ops not initialized");
    }
    std::string canonical_path;
    if (Status s = ValidateLoraAdapterPath(path, model_roots_, canonical_path); !s.ok()) {
        // Also allow unloading by an already-cached exact key even if the file
        // was removed from disk after loading.
        const std::string fallback_key = NormalizePathKey(path);
        if (loaded_adapters_.find(fallback_key) == loaded_adapters_.end()) {
            return s;
        }
        canonical_path = std::string(path);
    }

    const std::string key = NormalizePathKey(canonical_path);
    const auto it = loaded_adapters_.find(key);
    if (it == loaded_adapters_.end()) {
        return Status::Error(ErrorCode::kFileNotFound, "LoRA adapter is not loaded: " + canonical_path);
    }

    // Remove from active context bindings BEFORE freeing the adapter handle so
    // llama_context never holds a dangling llama_adapter_lora*.
    bool was_active = false;
    std::vector<LoraAdapterSpec> remaining_active;
    std::vector<ActiveLoraBinding> remaining_bindings;
    for (const LoraAdapterSpec& active : active_adapters_) {
        if (NormalizePathKey(active.path) == key) {
            was_active = true;
            continue;
        }
        const auto hit = loaded_adapters_.find(NormalizePathKey(active.path));
        if (hit != loaded_adapters_.end()) {
            remaining_active.push_back(active);
            remaining_bindings.push_back(ActiveLoraBinding{hit->second, active.scale});
        }
    }
    if (was_active) {
        if (Status s = ops_->SetContextAdapters(remaining_bindings); !s.ok()) {
            return s;
        }
        active_adapters_ = std::move(remaining_active);
    }

    LoraAdapterHandle handle = it->second;
    loaded_adapters_.erase(it);
    if (handle != nullptr) {
        ops_->FreeAdapter(handle);
    }
    return Status::Ok();
}

void LoraAdapterManager::UnloadAll() noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    if (ops_ == nullptr) {
        loaded_adapters_.clear();
        active_adapters_.clear();
        return;
    }
    // Always clear context bindings before freeing any adapter handles.
    if (!active_adapters_.empty() || !loaded_adapters_.empty()) {
        try {
            (void)ops_->SetContextAdapters({});
        } catch (...) {
            // UnloadAll is noexcept; proceed to free handles safely.
        }
    }
    active_adapters_.clear();
    for (const auto& [key, handle] : loaded_adapters_) {
        if (handle != nullptr) {
            ops_->FreeAdapter(handle);
        }
    }
    loaded_adapters_.clear();
}

std::size_t LoraAdapterManager::LoadedCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaded_adapters_.size();
}

bool LoraAdapterManager::IsAdapterLoaded(std::string_view canonical_path) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return loaded_adapters_.find(NormalizePathKey(canonical_path)) != loaded_adapters_.end();
}

std::vector<LoraAdapterSpec> LoraAdapterManager::ActiveAdapters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return active_adapters_;
}

// ---- ScopedLoraApplication ----

ScopedLoraApplication::ScopedLoraApplication(LoraAdapterManager& manager,
                                             const std::vector<LoraAdapterSpec>& adapters, Status& out_status)
    : manager_(&manager), previous_(manager.ActiveAdapters()) {
    out_status = manager_->ApplyAdapters(adapters);
    applied_ = out_status.ok();
}

ScopedLoraApplication::~ScopedLoraApplication() {
    if (applied_ && manager_ != nullptr) {
        try {
            (void)manager_->ApplyAdapters(previous_);
        } catch (...) {
            // Destructor must never throw.
        }
    }
}

// ---- Concrete llama.cpp v1.0.0 LoRA runtime implementation ----

namespace {

struct BackendHookEntry {
    const LlamaCppBackend* owner = nullptr;
    void* impl = nullptr;
    const LlamaCppBackendLoraHooks* hooks = nullptr;
};

struct BackendHookRegistry {
    std::mutex mu;
    std::vector<BackendHookEntry> entries;
};

BackendHookRegistry& GetBackendHookRegistry() noexcept {
    static auto* reg = new BackendHookRegistry();
    return *reg;
}

BackendHookEntry LookupBackendHook(const LlamaCppBackend* owner) noexcept {
    if (owner == nullptr) return {};
    auto& reg = GetBackendHookRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    for (const BackendHookEntry& e : reg.entries) {
        if (e.owner == owner) return e;
    }
    return {};
}

std::string ReadModelMetaStr(const llama_model* model, const char* key) {
    if (model == nullptr || key == nullptr) return {};
    std::size_t capacity = 256;
    for (int attempt = 0; attempt != 3; ++attempt) {
        std::string value(capacity, '\0');
        // llama.cpp v1.0.0 include/llama.h:611
        const int32_t n = llama_model_meta_val_str(model, key, value.data(), value.size());
        if (n < 0) return {};
        if (static_cast<std::size_t>(n) < value.size()) {
            value.resize(static_cast<std::size_t>(n));
            return value;
        }
        capacity = static_cast<std::size_t>(n) + 1;
    }
    return {};
}

std::string ReadAdapterMetaStr(const llama_adapter_lora* adapter, const char* key) {
    if (adapter == nullptr || key == nullptr) return {};
    std::size_t capacity = 256;
    for (int attempt = 0; attempt != 3; ++attempt) {
        std::string value(capacity, '\0');
        // llama.cpp v1.0.0 include/llama.h:701
        const int32_t n = llama_adapter_meta_val_str(adapter, key, value.data(), value.size());
        if (n < 0) return {};
        if (static_cast<std::size_t>(n) < value.size()) {
            value.resize(static_cast<std::size_t>(n));
            return value;
        }
        capacity = static_cast<std::size_t>(n) + 1;
    }
    return {};
}

class LlamaCppLoraRuntimeOps final : public LoraRuntimeOps {
public:
    LlamaCppLoraRuntimeOps(void* impl, const LlamaCppBackendLoraHooks* hooks) noexcept : impl_(impl), hooks_(hooks) {}

    Status ReadBaseModelMetadata(LoraModelMetadata& out) const override {
        auto* model = static_cast<const llama_model*>(hooks_->get_model(impl_));
        if (model == nullptr) {
            return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
        }
        out = {};
        out.model_id = hooks_->get_model_path(impl_);
        out.architecture = ReadModelMetaStr(model, "general.architecture");
        out.name = ReadModelMetaStr(model, "general.name");
        out.basename = ReadModelMetaStr(model, "general.basename");
        out.base_model_name = ReadModelMetaStr(model, "general.base_model.0.name");
        return Status::Ok();
    }

    Status ReadAdapterMetadata(const std::string& canonical_path, LoraAdapterMetadata& out) const override {
        out = {};
        gguf_init_params params{/*no_alloc=*/true, /*ctx=*/nullptr};
        gguf_context* gctx = gguf_init_from_file(canonical_path.c_str(), params);
        if (gctx == nullptr) {
            return Status::Error(ErrorCode::kLoadFailed,
                                 "failed to read GGUF metadata from LoRA adapter: " + canonical_path);
        }
        struct GgufGuard {
            gguf_context* ctx = nullptr;
            ~GgufGuard() noexcept {
                if (ctx != nullptr) gguf_free(ctx);
            }
        } guard{gctx};

        auto get_str = [&](const char* key) -> std::string {
            const int64_t kid = gguf_find_key(gctx, key);
            if (kid < 0 || gguf_get_kv_type(gctx, kid) != GGUF_TYPE_STRING) return {};
            const char* v = gguf_get_val_str(gctx, kid);
            return v != nullptr ? std::string(v) : std::string();
        };

        out.general_type = get_str("general.type");
        out.adapter_type = get_str("adapter.type");
        out.architecture = get_str("general.architecture");
        out.base_model_name = get_str("general.base_model.0.name");

        const int64_t alpha_id = gguf_find_key(gctx, "adapter.lora.alpha");
        if (alpha_id >= 0) {
            if (gguf_get_kv_type(gctx, alpha_id) == GGUF_TYPE_FLOAT32) {
                out.alpha = gguf_get_val_f32(gctx, alpha_id);
            } else {
                // Upstream llama.cpp v1.0.0 reads adapter.lora.alpha with gguf_get_val_f32
                // (src/llama-adapter.cpp:199-201, 221), whose GGML_ASSERT aborts on any
                // non-FLOAT32 type (ggml/src/gguf.cpp:196). Mark invalid so compatibility
                // fails closed before llama_adapter_lora_init.
                out.alpha = std::numeric_limits<float>::quiet_NaN();
            }
        }
        return Status::Ok();
    }

    Status InitAdapter(const std::string& canonical_path, LoraAdapterHandle& out_handle) override {
        out_handle = nullptr;
        auto* model = static_cast<llama_model*>(hooks_->get_model(impl_));
        if (model == nullptr) {
            return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
        }
        // llama.cpp v1.0.0 include/llama.h:685
        llama_adapter_lora* adapter = llama_adapter_lora_init(model, canonical_path.c_str());
        if (adapter == nullptr) {
            return Status::Error(ErrorCode::kLoadFailed, "llama_adapter_lora_init failed for: " + canonical_path);
        }
        out_handle = adapter;
        return Status::Ok();
    }

    Status ReadLoadedAdapterMetadata(LoraAdapterHandle handle, LoraAdapterMetadata& out) const override {
        out = {};
        const auto* adapter = static_cast<const llama_adapter_lora*>(handle);
        if (adapter == nullptr) {
            return Status::Error(ErrorCode::kInvalidArgument, "null LoRA adapter handle");
        }
        out.general_type = ReadAdapterMetaStr(adapter, "general.type");
        out.adapter_type = ReadAdapterMetaStr(adapter, "adapter.type");
        out.architecture = ReadAdapterMetaStr(adapter, "general.architecture");
        out.base_model_name = ReadAdapterMetaStr(adapter, "general.base_model.0.name");
        const std::string alpha_str = ReadAdapterMetaStr(adapter, "adapter.lora.alpha");
        if (!alpha_str.empty()) {
            try {
                std::size_t consumed = 0;
                const float parsed = std::stof(alpha_str, &consumed);
                if (consumed == alpha_str.size()) {
                    out.alpha = parsed;
                } else {
                    out.alpha = std::numeric_limits<float>::quiet_NaN();
                }
            } catch (...) {
                out.alpha = std::numeric_limits<float>::quiet_NaN();
            }
        }
        return Status::Ok();
    }

    void FreeAdapter(LoraAdapterHandle handle) noexcept override {
        if (handle == nullptr) return;
        // llama.cpp v1.0.0 include/llama.h:714
        llama_adapter_lora_free(static_cast<llama_adapter_lora*>(handle));
    }

    Status SetContextAdapters(std::span<const ActiveLoraBinding> bindings) override {
        if (hooks_->get_model(impl_) == nullptr) {
            return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
        }
        auto* ctx = static_cast<llama_context*>(hooks_->get_ctx(impl_));
        if (ctx == nullptr) {
            if (bindings.empty()) return Status::Ok();
            return Status::Error(ErrorCode::kNotLoaded, "model loaded vocab-only; no context for LoRA adapters");
        }

        // llama.cpp v1.0.0 include/llama.h:723 (llama_set_adapters_lora)
        int32_t rc = 0;
        if (bindings.empty()) {
            rc = llama_set_adapters_lora(ctx, nullptr, 0, nullptr);
        } else {
            std::vector<llama_adapter_lora*> ptrs;
            std::vector<float> scales;
            ptrs.reserve(bindings.size());
            scales.reserve(bindings.size());
            for (const ActiveLoraBinding& b : bindings) {
                ptrs.push_back(static_cast<llama_adapter_lora*>(b.handle));
                scales.push_back(b.scale);
            }
            rc = llama_set_adapters_lora(ctx, ptrs.data(), ptrs.size(), scales.data());
        }
        if (rc != 0) {
            return Status::Error(ErrorCode::kDecodeFailed,
                                 "llama_set_adapters_lora failed with code " + std::to_string(rc));
        }

        if (bindings.size() != last_applied_.size() ||
            !std::equal(bindings.begin(), bindings.end(), last_applied_.begin(),
                        [](const ActiveLoraBinding& a, const ActiveLoraBinding& b) noexcept {
                            return a.handle == b.handle && a.scale == b.scale;
                        })) {
            if (hooks_->invalidate_kv != nullptr) {
                hooks_->invalidate_kv(impl_);
            }
            last_applied_.assign(bindings.begin(), bindings.end());
        }
        return Status::Ok();
    }

private:
    void* impl_ = nullptr;
    const LlamaCppBackendLoraHooks* hooks_ = nullptr;
    std::vector<ActiveLoraBinding> last_applied_;
};

LoraAdapterManager* EnsureLoraManager(const BackendHookEntry& entry, const std::vector<std::string>& model_roots) {
    if (entry.impl == nullptr || entry.hooks == nullptr || entry.hooks->get_model(entry.impl) == nullptr) {
        return nullptr;
    }
    auto& ops_slot = entry.hooks->lora_ops_slot(entry.impl);
    auto& mgr_slot = entry.hooks->lora_manager_slot(entry.impl);
    if (!ops_slot) {
        ops_slot = std::make_unique<LlamaCppLoraRuntimeOps>(entry.impl, entry.hooks);
    }
    if (!mgr_slot) {
        mgr_slot = std::make_unique<LoraAdapterManager>(ops_slot.get(), model_roots);
    } else if (!model_roots.empty()) {
        mgr_slot->SetModelRoots(model_roots);
    }
    return mgr_slot.get();
}

}  // namespace

void RegisterLlamaCppBackendForLora(const LlamaCppBackend* owner, void* impl,
                                    const LlamaCppBackendLoraHooks* hooks) {
    if (owner == nullptr || impl == nullptr || hooks == nullptr) return;
    auto& reg = GetBackendHookRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    for (BackendHookEntry& e : reg.entries) {
        if (e.owner == owner) {
            e.impl = impl;
            e.hooks = hooks;
            return;
        }
    }
    reg.entries.push_back(BackendHookEntry{owner, impl, hooks});
}

void TransferLlamaCppBackendForLora(const LlamaCppBackend* from, const LlamaCppBackend* to) noexcept {
    if (from == nullptr || to == nullptr || from == to) return;
    auto& reg = GetBackendHookRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    // Remove any existing slot owned by `to` in-place (zero allocation).
    for (std::size_t i = 0; i < reg.entries.size(); ++i) {
        if (reg.entries[i].owner == to) {
            reg.entries[i] = reg.entries.back();
            reg.entries.pop_back();
            break;
        }
    }
    // Rebind `from`'s existing slot to `to` in-place (zero allocation).
    for (BackendHookEntry& e : reg.entries) {
        if (e.owner == from) {
            e.owner = to;
            return;
        }
    }
}

void UnregisterLlamaCppBackendForLora(const LlamaCppBackend* owner) noexcept {
    if (owner == nullptr) return;
    auto& reg = GetBackendHookRegistry();
    std::lock_guard<std::mutex> lock(reg.mu);
    for (std::size_t i = 0; i < reg.entries.size(); ++i) {
        if (reg.entries[i].owner == owner) {
            reg.entries[i] = reg.entries.back();
            reg.entries.pop_back();
            return;
        }
    }
}

Status ApplyLoraAdapters(LlamaCppBackend& backend, const std::vector<LoraAdapterSpec>& adapters,
                         const std::vector<std::string>& model_roots) {
    const BackendHookEntry entry = LookupBackendHook(&backend);
    LoraAdapterManager* mgr = EnsureLoraManager(entry, model_roots);
    if (mgr == nullptr) {
        return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    }
    return mgr->ApplyAdapters(adapters);
}

Status ApplyLoraAlias(LlamaCppBackend& backend, const LoraRegistry& registry, std::string_view alias) {
    const BackendHookEntry entry = LookupBackendHook(&backend);
    LoraAdapterManager* mgr = EnsureLoraManager(entry, registry.ModelRoots());
    if (mgr == nullptr) {
        return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    }
    return mgr->ApplyAlias(registry, alias);
}

Status ClearLoraAdapters(LlamaCppBackend& backend) {
    const BackendHookEntry entry = LookupBackendHook(&backend);
    if (entry.impl == nullptr || entry.hooks == nullptr || entry.hooks->get_model(entry.impl) == nullptr) {
        return Status::Error(ErrorCode::kNotLoaded, "no model loaded");
    }
    auto& mgr_slot = entry.hooks->lora_manager_slot(entry.impl);
    if (!mgr_slot) return Status::Ok();
    return mgr_slot->ClearActive();
}

void UnloadAllLoraAdapters(LlamaCppBackend& backend) noexcept {
    const BackendHookEntry entry = LookupBackendHook(&backend);
    if (entry.impl != nullptr && entry.hooks != nullptr) {
        auto& mgr_slot = entry.hooks->lora_manager_slot(entry.impl);
        if (mgr_slot) mgr_slot->UnloadAll();
    }
}

std::vector<LoraAdapterSpec> ActiveLoraAdapters(const LlamaCppBackend& backend) {
    const BackendHookEntry entry = LookupBackendHook(&backend);
    if (entry.impl == nullptr || entry.hooks == nullptr) return {};
    auto& mgr_slot = entry.hooks->lora_manager_slot(entry.impl);
    return mgr_slot ? mgr_slot->ActiveAdapters() : std::vector<LoraAdapterSpec>{};
}

}  // namespace sonder::backends::llamacpp
