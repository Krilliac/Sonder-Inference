// Load-time GGUF metadata compatibility shims for llama.cpp v1.0.0.
//
// Background & upstream citations (llama.cpp v1.0.0, commit b1b770d):
// - GGUFs exported by Ollama for Qwen3.5 / Qwen3.6 store
//   `<arch>.rope.dimension_sections` with 3 elements ([t, h, w]).
// - Upstream's converter pads `mrope_section` with trailing zeros to 4 elements
//   `[time, height, width, extra]` before writing `rope.dimension_sections`:
//     * convert_hf_to_gguf.py:17-26 -> conversion/base.py:1968-1974
//     * conversion/qwen.py:718-731 (_Qwen35MRopeMixin: default [11, 11, 10, 0])
// - llama.cpp v1.0.0 expects 4 elements in `hparams.rope_sections`
//   (src/llama-hparams.h:180; zero-init/log in src/llama-model.cpp:1554,2676)
//   via `ml.get_key_or_arr(LLM_KV_ROPE_DIMENSION_SECTIONS, hparams.rope_sections, 4, true)`
//   in src/models/qwen35.cpp:6 and src/models/qwen35moe.cpp:9
//   (arch strings "qwen35" and "qwen35moe" in src/llama-arch.cpp:45-46,373).
// - `llama_model_kv_override` (include/llama.h:276-314) only supports scalar
//   INT/FLOAT/BOOL/STR tags and `llama_model_loader::get_key_or_arr`
//   (src/llama-model-loader.cpp:464-486) throws on a 3-element GGUF_TYPE_ARRAY
//   without consulting `kv_overrides`.
// - When a 3-element `<arch>.rope.dimension_sections` is detected on an affected
//   architecture, we materialize a tiny metadata-only temporary GGUF overlay
//   (0 tensors; original file untouched on disk) with the key padded to 4
//   elements `[s0, s1, s2, 0]`, and load via `llama_model_load_from_splits`
//   with `[overlay, orig_file(s)...]` so `llama_model_loader` uses the overlay's
//   metadata (`n_backbone = 1`, split keys stripped) and maps all weights
//   directly from the original file(s) via the `has_extras` shard path
//   (src/llama-model-loader.cpp:604-758).
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

struct llama_model;
struct llama_model_params;

namespace sonder::backends::llamacpp {

// Describes a detected `<arch>.rope.dimension_sections` 3 -> 4 compatibility fixup.
struct RopeDimensionSectionsFixup {
    std::string architecture;                  // e.g. "qwen35" or "qwen35moe"
    std::string key;                           // e.g. "qwen35.rope.dimension_sections"
    std::array<std::int32_t, 3> original{};
    std::array<std::int32_t, 4> normalized{};  // {original[0], original[1], original[2], 0}
    bool is_unsigned = false;                  // true when source GGUF array element type was GGUF_TYPE_UINT32
};

// Result of inspecting a GGUF file's metadata for load-time compatibility shims.
struct GgufCompatInspection {
    bool needs_shim = false;
    std::string architecture;
    std::optional<RopeDimensionSectionsFixup> rope_sections;
    std::string detail;  // human-readable diagnostic when needs_shim is true
};

// RAII owner for a temporary metadata-only GGUF overlay file created during Load().
// Must outlive the `llama_model*` loaded from it because llama.cpp may keep the
// overlay file memory-mapped until `llama_model_free()`. Removes the temp file on
// Reset() or destruction. Never modifies the original model file.
class GgufCompatOverlay {
public:
    GgufCompatOverlay() = default;
    ~GgufCompatOverlay() noexcept;

    GgufCompatOverlay(const GgufCompatOverlay&) = delete;
    GgufCompatOverlay& operator=(const GgufCompatOverlay&) = delete;
    GgufCompatOverlay(GgufCompatOverlay&& other) noexcept;
    GgufCompatOverlay& operator=(GgufCompatOverlay&& other) noexcept;

    [[nodiscard]] bool active() const noexcept { return !temp_meta_path_.empty(); }
    [[nodiscard]] const std::filesystem::path& temp_meta_path() const noexcept { return temp_meta_path_; }
    [[nodiscard]] const std::vector<std::string>& split_paths() const noexcept { return split_paths_; }
    [[nodiscard]] const GgufCompatInspection& inspection() const noexcept { return inspection_; }

    void Reset() noexcept;

private:
    friend bool PrepareGgufCompatOverlay(const std::string& model_path,
                                         GgufCompatOverlay& out_overlay,
                                         std::string* error_out);

    std::filesystem::path temp_meta_path_;
    std::vector<std::string> split_paths_;
    GgufCompatInspection inspection_;
};

// True for architectures affected by the Ollama 3-element rope.dimension_sections export
// ("qwen35" and "qwen35moe").
[[nodiscard]] bool IsQwen35RopeCompatArch(std::string_view arch) noexcept;

// Pure normalization matching upstream conversion/base.py:1968-1974:
// pads a 3-element [t, h, w] section tuple with a trailing 0 to 4 elements.
[[nodiscard]] constexpr std::array<std::int32_t, 4> NormalizeRopeDimensionSections(
    const std::array<std::int32_t, 3>& sections) noexcept {
    return {sections[0], sections[1], sections[2], 0};
}

// Inspects GGUF metadata only (no tensor read, never modifies the file on disk).
// Returns needs_shim == false when the file cannot be opened as GGUF, uses an
// unaffected architecture, lacks `<arch>.rope.dimension_sections`, or already has
// != 3 elements.
[[nodiscard]] GgufCompatInspection InspectGgufCompat(const std::string& model_path);

// If `model_path` needs a compatibility shim, builds a metadata-only temporary GGUF
// overlay in the system temp directory with normalized metadata and populates
// `out_overlay` with the overlay path and split list ([overlay, model_path, ...]).
// Returns true on success (whether or not a shim was needed; check `out_overlay.active()`).
// Returns false only if a shim was needed but creating the temp overlay failed
// (populates `error_out` if non-null). Never modifies `model_path`.
[[nodiscard]] bool PrepareGgufCompatOverlay(const std::string& model_path,
                                            GgufCompatOverlay& out_overlay,
                                            std::string* error_out = nullptr);

// Loads a llama_model from `model_path`, transparently applying GGUF compatibility
// shims when needed via a temporary metadata-only overlay and `llama_model_load_from_splits`.
// When no shim is needed, calls `llama_model_load_from_file` directly.
// `out_overlay` must stay alive until after `llama_model_free()` is called on the
// returned model (and is automatically reset on failure).
[[nodiscard]] llama_model* LoadModelWithGgufCompat(const std::string& model_path,
                                                   const llama_model_params& params,
                                                   GgufCompatOverlay& out_overlay,
                                                   std::string* error_out = nullptr);

}  // namespace sonder::backends::llamacpp
