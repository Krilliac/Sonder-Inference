// Sonder Inference - LoRA adapter registry and runtime lifecycle manager.
//
// Maps aliases to (base_model, adapters + scales), validates adapter paths
// against configured model roots (no ".." escapes, regular files only),
// verifies base-model metadata compatibility, and manages per-alias and
// per-request LoRA application and clean unloading over llama.cpp v1.0.0.
//
// Upstream llama.cpp v1.0.0 (tag v1.0.0, commit b1b770d) C APIs used by the
// concrete LlamaCppBackend runtime implementation in llamacpp_backend.cpp:
//   - llama_adapter_lora_init      (include/llama.h:685)
//   - llama_adapter_meta_val_str   (include/llama.h:701)
//   - llama_adapter_lora_free      (include/llama.h:714)
//   - llama_set_adapters_lora      (include/llama.h:723)
//   - llama_model_meta_val_str     (include/llama.h:611)
#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "sonder/backends/llamacpp/llamacpp_backend.h"
#include "sonder/inference/backends/llamacpp.hpp"

namespace sonder::backends::llamacpp {

inline constexpr std::size_t kMaxLoraAdaptersPerRequest = 16;
inline constexpr float kMinLoraScale = 0.0F;
inline constexpr float kMaxLoraScale = 4.0F;

struct LoraAdapterSpec {
    std::string path;
    float scale = 1.0F;

    [[nodiscard]] bool operator==(const LoraAdapterSpec& other) const noexcept = default;
};

struct LoraAliasEntry {
    std::string alias;
    std::string base_model;
    std::vector<LoraAdapterSpec> adapters;
};

struct LoraModelMetadata {
    std::string model_id;         // Loaded base model path or identifier
    std::string architecture;     // GGUF "general.architecture"
    std::string name;             // GGUF "general.name" (optional)
    std::string basename;         // GGUF "general.basename" (optional)
    std::string base_model_name;  // GGUF "general.base_model.0.name" (optional)
};

struct LoraAdapterMetadata {
    std::string general_type;     // Expected: "adapter"
    std::string adapter_type;     // Expected: "lora"
    std::string architecture;     // Expected: matches base model "general.architecture"
    std::string base_model_name;  // Optional: "general.base_model.0.name"
    std::optional<float> alpha;   // Optional: "adapter.lora.alpha"
};

// Opaque handle representing a loaded llama_adapter_lora*.
using LoraAdapterHandle = void*;

struct ActiveLoraBinding {
    LoraAdapterHandle handle = nullptr;
    float scale = 1.0F;
};

// Abstract runtime seam over llama.cpp v1.0.0 LoRA calls so pure unit tests
// can exercise registry, validation, caching, per-request switching, and
// teardown ordering without real GGUF weights.
class LoraRuntimeOps {
public:
    virtual ~LoraRuntimeOps() = default;

    // Reads metadata for the currently loaded base model.
    virtual Status ReadBaseModelMetadata(LoraModelMetadata& out) const = 0;

    // Reads GGUF metadata from an adapter file prior to tensor allocation.
    virtual Status ReadAdapterMetadata(const std::string& canonical_path, LoraAdapterMetadata& out) const = 0;

    // Initializes a LoRA adapter against the loaded base model (llama_adapter_lora_init).
    virtual Status InitAdapter(const std::string& canonical_path, LoraAdapterHandle& out_handle) = 0;

    // Reads metadata from an initialized adapter handle (llama_adapter_meta_val_str).
    virtual Status ReadLoadedAdapterMetadata(LoraAdapterHandle handle, LoraAdapterMetadata& out) const = 0;

    // Frees an initialized adapter handle (llama_adapter_lora_free).
    virtual void FreeAdapter(LoraAdapterHandle handle) noexcept = 0;

    // Applies (or clears, when empty) active adapters on the context (llama_set_adapters_lora).
    virtual Status SetContextAdapters(std::span<const ActiveLoraBinding> bindings) = 0;
};

// ---- Pure validators (no llama.cpp dependency) ----

// Checks that `scale` is finite and within [kMinLoraScale, kMaxLoraScale].
[[nodiscard]] Status ValidateLoraScale(float scale);

// Checks lexical path rules (non-empty, no embedded NUL, no ".." components)
// without touching the filesystem. Useful for startup option validation.
[[nodiscard]] Status ValidateLoraPathLexical(std::string_view raw_path);

// Validates that `raw_path` has no ".." traversal, exists as a regular file,
// and resolves canonically strictly inside at least one configured root in
// `model_roots`. Writes the canonical UTF-8 path to `canonical_out`.
[[nodiscard]] Status ValidateLoraAdapterPath(std::string_view raw_path, const std::vector<std::string>& model_roots,
                                             std::string& canonical_out);

// Checks that an adapter's GGUF metadata matches the loaded base model:
//   - general.type == "adapter"
//   - adapter.type == "lora"
//   - general.architecture non-empty and equal to base model's general.architecture
//   - if adapter specifies general.base_model.0.name and base model specifies
//     any base name/basename/name, at least one must match (case-insensitively)
//   - if adapter.lora.alpha is present, it must be finite and >= 0
[[nodiscard]] Status ValidateLoraCompatibility(const LoraModelMetadata& base_meta,
                                               const LoraAdapterMetadata& adapter_meta,
                                               std::string_view adapter_label = {});

// Checks whether an alias's configured base_model identifier matches a loaded
// base model's model_id (exact match, canonical path match, or filename match).
[[nodiscard]] bool BaseModelMatches(std::string_view alias_base_model, std::string_view loaded_model_id);

// Validates alias entry syntax and scale/path-traversal rules without disk I/O.
[[nodiscard]] Status ValidateLoraAliasSyntax(const LoraAliasEntry& entry);

// Pure startup validator for LlamaCppBackendOptions::lora_aliases (checks
// syntax, scales, lexical path rules, and duplicate alias names without disk I/O).
[[nodiscard]] Status ValidateLoraAliasOptions(
    std::span<const sonder::inference::LlamaCppLoraAliasOption> aliases);

// Resolves `requested_model` against configured `aliases`. If matched,
// validates adapter paths against `lora_roots` (falling back to `model_dirs`
// when empty) and populates `matched_out` and `effective_roots_out`. When
// `requested_model` is not a configured alias, clears `matched_out` and returns Ok.
[[nodiscard]] Status ResolveConfiguredLoraAlias(
    std::string_view requested_model, std::span<const sonder::inference::LlamaCppLoraAliasOption> aliases,
    const std::vector<std::string>& lora_roots, const std::vector<std::string>& model_dirs,
    std::optional<LoraAliasEntry>& matched_out, std::vector<std::string>& effective_roots_out);

// ---- Registry: alias -> (base_model, adapters + scales) ----

class LoraRegistry {
public:
    explicit LoraRegistry(std::vector<std::string> model_roots = {});

    void SetModelRoots(std::vector<std::string> model_roots);
    [[nodiscard]] std::vector<std::string> ModelRoots() const;

    // Validates alias, base_model, scales, adapter count, duplicate paths, and
    // full filesystem/root containment, then registers the canonicalized entry.
    Status RegisterAlias(const LoraAliasEntry& entry);
    bool UnregisterAlias(std::string_view alias);
    [[nodiscard]] bool HasAlias(std::string_view alias) const;
    Status ResolveAlias(std::string_view alias, LoraAliasEntry& out) const;
    [[nodiscard]] std::vector<LoraAliasEntry> ListAliases() const;
    void ClearAliases();

private:
    mutable std::mutex mutex_;
    std::vector<std::string> model_roots_;
    std::unordered_map<std::string, LoraAliasEntry> aliases_;
};

// ---- Adapter lifecycle manager bound to one loaded model/context ----
//
// Concurrency and context scope:
//   - ApplyAdapters, ApplyAlias, ClearActive, UnloadAdapter, UnloadAll, and
//     ScopedLoraApplication mutate the backend's single shared generation
//     llama_context (`ctx`). They MUST only be called while the continuous-
//     batching worker (`n_slots > 1`) is stopped (e.g., at model load/unload
//     or under exclusive quiescence with no concurrent decode in flight).
//   - Auxiliary isolated contexts (`embed_ctx`, `rank_ctx`, and speculative
//     MTP `ctx_dft`) do not receive LoRA adapters and always run on the base
//     model weights without adapters attached.

class LoraAdapterManager {
public:
    explicit LoraAdapterManager(LoraRuntimeOps* ops, std::vector<std::string> model_roots = {});
    ~LoraAdapterManager();

    LoraAdapterManager(const LoraAdapterManager&) = delete;
    LoraAdapterManager& operator=(const LoraAdapterManager&) = delete;

    void SetModelRoots(std::vector<std::string> model_roots);
    [[nodiscard]] std::vector<std::string> ModelRoots() const;

    // Applies an explicit set of adapters (per-request or per-alias) to the
    // active context. An empty vector clears active adapters. Atomic on failure:
    // previously active adapters stay active and newly initialized handles from
    // a failed call are freed before returning.
    Status ApplyAdapters(const std::vector<LoraAdapterSpec>& adapters);

    // Resolves `alias` from `registry`, verifies its base_model matches the
    // loaded base model, and applies its adapters.
    Status ApplyAlias(const LoraRegistry& registry, std::string_view alias);

    // Clears all active adapters from the context without freeing cached handles.
    Status ClearActive();

    // Unloads a single adapter by path (updating/clearing the context FIRST so
    // the context never holds a dangling handle, then freeing the adapter).
    Status UnloadAdapter(std::string_view path);

    // Clears all active adapters from the context FIRST, then frees every
    // loaded adapter handle cleanly.
    void UnloadAll() noexcept;

    [[nodiscard]] std::size_t LoadedCount() const;
    [[nodiscard]] bool IsAdapterLoaded(std::string_view canonical_path) const;
    [[nodiscard]] std::vector<LoraAdapterSpec> ActiveAdapters() const;

private:
    LoraRuntimeOps* ops_ = nullptr;  // non-owning; must outlive manager
    mutable std::mutex mutex_;
    std::vector<std::string> model_roots_;
    std::unordered_map<std::string, LoraAdapterHandle> loaded_adapters_;
    std::vector<LoraAdapterSpec> active_adapters_;
};

// RAII guard for per-request LoRA overrides: applies `adapters` (or clears
// adapters when empty) and restores the previous active set on destruction.
// Operates on the single shared generation context (`ctx`) and must only be
// used while the batch worker is stopped (no concurrent decode in flight);
// embed/rank/draft contexts run without the adapter.
class ScopedLoraApplication {
public:
    ScopedLoraApplication(LoraAdapterManager& manager, const std::vector<LoraAdapterSpec>& adapters,
                          Status& out_status);
    ~ScopedLoraApplication();

    ScopedLoraApplication(const ScopedLoraApplication&) = delete;
    ScopedLoraApplication& operator=(const ScopedLoraApplication&) = delete;

    [[nodiscard]] bool applied() const noexcept { return applied_; }

private:
    LoraAdapterManager* manager_ = nullptr;
    std::vector<LoraAdapterSpec> previous_;
    bool applied_ = false;
};

// ---- Hooks bridging LlamaCppBackend (implemented in lora_registry.cpp / llamacpp_backend.cpp) ----

struct LlamaCppBackendLoraHooks {
    void* (*get_model)(void* impl) noexcept = nullptr;
    void* (*get_ctx)(void* impl) noexcept = nullptr;
    std::string (*get_model_path)(void* impl) = nullptr;
    void (*invalidate_kv)(void* impl) noexcept = nullptr;
    std::unique_ptr<LoraRuntimeOps>& (*lora_ops_slot)(void* impl) noexcept = nullptr;
    std::unique_ptr<LoraAdapterManager>& (*lora_manager_slot)(void* impl) noexcept = nullptr;
};

// Registers a newly constructed LlamaCppBackend instance for LoRA hooks (called
// from the non-noexcept constructor). Transfer/Unregister mutate the existing
// table slot in place without allocating so move/destroy paths stay noexcept.
void RegisterLlamaCppBackendForLora(const LlamaCppBackend* owner, void* impl,
                                    const LlamaCppBackendLoraHooks* hooks);
void TransferLlamaCppBackendForLora(const LlamaCppBackend* from, const LlamaCppBackend* to) noexcept;
void UnregisterLlamaCppBackendForLora(const LlamaCppBackend* owner) noexcept;

// Context-mutating LoRA hooks act on the backend's single shared generation
// context (`ctx`) and must only be called while the batch worker is stopped.
// Auxiliary embed, rank, and speculative draft contexts run without LoRA adapters.
Status ApplyLoraAdapters(LlamaCppBackend& backend, const std::vector<LoraAdapterSpec>& adapters,
                         const std::vector<std::string>& model_roots);
Status ApplyLoraAlias(LlamaCppBackend& backend, const LoraRegistry& registry, std::string_view alias);
Status ClearLoraAdapters(LlamaCppBackend& backend);
void UnloadAllLoraAdapters(LlamaCppBackend& backend) noexcept;
[[nodiscard]] std::vector<LoraAdapterSpec> ActiveLoraAdapters(const LlamaCppBackend& backend);

}  // namespace sonder::backends::llamacpp
