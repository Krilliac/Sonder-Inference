// Load-time GGUF metadata compatibility shims for llama.cpp v1.0.0.
//
// See gguf_compat.hpp for the exact upstream llama.cpp v1.0.0 citations.
#include "gguf_compat.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "gguf.h"
#include "llama.h"

namespace sonder::backends::llamacpp {

namespace {

struct GgufDeleter {
    void operator()(gguf_context* ctx) const noexcept {
        if (ctx != nullptr) gguf_free(ctx);
    }
};
using GgufPtr = std::unique_ptr<gguf_context, GgufDeleter>;

std::string PathToUtf8(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

std::uint64_t CurrentPid() noexcept {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(_getpid());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

std::filesystem::path MakeTempOverlayPath() {
    static std::atomic<std::uint64_t> counter{0};
    const std::uint64_t seq = counter.fetch_add(1, std::memory_order_relaxed);
    const auto ticks =
        static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
    const std::string filename = "sonder-gguf-compat-" + std::to_string(CurrentPid()) + "-" +
                                 std::to_string(ticks) + "-" + std::to_string(seq) + ".gguf";
    return std::filesystem::temp_directory_path() / filename;
}

GgufCompatInspection InspectGgufContext(const gguf_context* ctx) {
    GgufCompatInspection result;
    if (ctx == nullptr) return result;

    const std::int64_t arch_kid = gguf_find_key(ctx, "general.architecture");
    if (arch_kid < 0 || gguf_get_kv_type(ctx, arch_kid) != GGUF_TYPE_STRING) {
        return result;
    }
    const char* arch_cstr = gguf_get_val_str(ctx, arch_kid);
    if (arch_cstr == nullptr || arch_cstr[0] == '\0') {
        return result;
    }
    result.architecture = arch_cstr;
    if (!IsQwen35RopeCompatArch(result.architecture)) {
        return result;
    }

    const std::string key = result.architecture + ".rope.dimension_sections";
    const std::int64_t kid = gguf_find_key(ctx, key.c_str());
    if (kid < 0 || gguf_get_kv_type(ctx, kid) != GGUF_TYPE_ARRAY) {
        return result;
    }
    if (gguf_get_arr_n(ctx, kid) != 3u) {
        return result;
    }

    const gguf_type elem_type = gguf_get_arr_type(ctx, kid);
    // Must reject non-integer array element types BEFORE calling gguf_get_arr_data(),
    // because llama.cpp v1.0.0 asserts `get_type() != GGUF_TYPE_STRING` inside
    // gguf_get_arr_data() (ggml/src/gguf.cpp:1090-1094).
    if (elem_type != GGUF_TYPE_INT32 && elem_type != GGUF_TYPE_UINT32) {
        return result;
    }

    const void* raw_data = gguf_get_arr_data(ctx, kid);
    if (raw_data == nullptr) {
        return result;
    }

    RopeDimensionSectionsFixup fixup;
    fixup.architecture = result.architecture;
    fixup.key = key;

    if (elem_type == GGUF_TYPE_INT32) {
        std::array<std::int32_t, 3> vals{};
        std::memcpy(vals.data(), raw_data, sizeof(vals));
        if (vals[0] < 0 || vals[1] < 0 || vals[2] < 0) {
            return result;
        }
        fixup.original = vals;
        fixup.is_unsigned = false;
    } else {
        std::array<std::uint32_t, 3> vals{};
        std::memcpy(vals.data(), raw_data, sizeof(vals));
        constexpr auto kMaxI32 = static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max());
        if (vals[0] > kMaxI32 || vals[1] > kMaxI32 || vals[2] > kMaxI32) {
            return result;
        }
        fixup.original = {static_cast<std::int32_t>(vals[0]), static_cast<std::int32_t>(vals[1]),
                          static_cast<std::int32_t>(vals[2])};
        fixup.is_unsigned = true;
    }

    fixup.normalized = NormalizeRopeDimensionSections(fixup.original);
    result.needs_shim = true;
    result.detail = "normalized " + fixup.key + " [" + std::to_string(fixup.original[0]) + ", " +
                    std::to_string(fixup.original[1]) + ", " + std::to_string(fixup.original[2]) + "] -> [" +
                    std::to_string(fixup.normalized[0]) + ", " + std::to_string(fixup.normalized[1]) + ", " +
                    std::to_string(fixup.normalized[2]) + ", " + std::to_string(fixup.normalized[3]) +
                    "] via metadata-only overlay (original file untouched)";
    result.rope_sections = std::move(fixup);
    return result;
}

}  // namespace

GgufCompatOverlay::~GgufCompatOverlay() noexcept { Reset(); }

GgufCompatOverlay::GgufCompatOverlay(GgufCompatOverlay&& other) noexcept
    : temp_meta_path_(std::move(other.temp_meta_path_)),
      split_paths_(std::move(other.split_paths_)),
      inspection_(std::move(other.inspection_)) {
    other.temp_meta_path_.clear();
    other.split_paths_.clear();
    other.inspection_ = {};
}

GgufCompatOverlay& GgufCompatOverlay::operator=(GgufCompatOverlay&& other) noexcept {
    if (this != &other) {
        Reset();
        temp_meta_path_ = std::move(other.temp_meta_path_);
        split_paths_ = std::move(other.split_paths_);
        inspection_ = std::move(other.inspection_);
        other.temp_meta_path_.clear();
        other.split_paths_.clear();
        other.inspection_ = {};
    }
    return *this;
}

void GgufCompatOverlay::Reset() noexcept {
    if (!temp_meta_path_.empty()) {
        std::error_code ec;
        std::filesystem::remove(temp_meta_path_, ec);
        temp_meta_path_.clear();
    }
    split_paths_.clear();
    inspection_ = {};
}

bool IsQwen35RopeCompatArch(std::string_view arch) noexcept {
    return arch == "qwen35" || arch == "qwen35moe";
}

GgufCompatInspection InspectGgufCompat(const std::string& model_path) {
    if (model_path.empty()) return {};
    gguf_init_params params{/*no_alloc=*/true, /*ctx=*/nullptr};
    GgufPtr src(gguf_init_from_file(model_path.c_str(), params));
    if (!src) return {};
    return InspectGgufContext(src.get());
}

bool PrepareGgufCompatOverlay(const std::string& model_path, GgufCompatOverlay& out_overlay,
                              std::string* error_out) {
    out_overlay.Reset();
    if (error_out != nullptr) error_out->clear();
    if (model_path.empty()) return true;

    gguf_init_params params{/*no_alloc=*/true, /*ctx=*/nullptr};
    GgufPtr src(gguf_init_from_file(model_path.c_str(), params));
    if (!src) return true;

    GgufCompatInspection insp = InspectGgufContext(src.get());
    if (!insp.needs_shim || !insp.rope_sections.has_value()) {
        return true;
    }

    // Resolve original shard paths when the source GGUF is itself a multi-split file.
    std::vector<std::string> orig_splits;
    const std::int64_t split_count_kid = gguf_find_key(src.get(), "split.count");
    if (split_count_kid >= 0 && gguf_get_kv_type(src.get(), split_count_kid) == GGUF_TYPE_UINT16) {
        const std::uint16_t n_split = gguf_get_val_u16(src.get(), split_count_kid);
        if (n_split > 1) {
            const std::int64_t split_no_kid = gguf_find_key(src.get(), "split.no");
            if (split_no_kid < 0 || gguf_get_kv_type(src.get(), split_no_kid) != GGUF_TYPE_UINT16 ||
                gguf_get_val_u16(src.get(), split_no_kid) != 0) {
                // Leave non-first splits alone so llama.cpp emits its standard split error.
                return true;
            }
            // llama_model_loader rejects paths.size() > UINT16_MAX (src/llama-model-loader.cpp:643),
            // and the metadata overlay prepends 1 path (`1 + n_split`). Fail closed before formatting.
            if (n_split >= std::numeric_limits<std::uint16_t>::max()) {
                if (error_out != nullptr) {
                    *error_out = "gguf_compat: split.count (" + std::to_string(n_split) +
                                 ") exceeds maximum supported with metadata overlay";
                }
                return false;
            }
            const std::size_t buf_cap = model_path.size() + 64;
            std::vector<char> prefix_buf(buf_cap, '\0');
            const std::int32_t prefix_len = llama_split_prefix(
                prefix_buf.data(), prefix_buf.size(), model_path.c_str(), /*split_no=*/0, n_split);
            if (prefix_len <= 0) {
                if (error_out != nullptr) {
                    *error_out = "gguf_compat: failed to derive split prefix for multi-split GGUF: " + model_path;
                }
                return false;
            }
            const std::string prefix(prefix_buf.data(), static_cast<std::size_t>(prefix_len));
            orig_splits.reserve(n_split);
            orig_splits.push_back(model_path);
            std::vector<char> split_buf(prefix.size() + 64, '\0');
            for (std::int32_t idx = 1; idx < static_cast<std::int32_t>(n_split); ++idx) {
                const std::int32_t written =
                    llama_split_path(split_buf.data(), split_buf.size(), prefix.c_str(), idx, n_split);
                if (written <= 0) {
                    if (error_out != nullptr) {
                        *error_out = "gguf_compat: failed to format split path #" + std::to_string(idx);
                    }
                    return false;
                }
                orig_splits.emplace_back(split_buf.data(), static_cast<std::size_t>(written));
            }
        }
    }
    if (orig_splits.empty()) {
        orig_splits.push_back(model_path);
    }

    // Materialize a metadata-only GGUF overlay (0 tensors) with normalized metadata.
    GgufPtr patched(gguf_init_empty());
    if (!patched) {
        if (error_out != nullptr) *error_out = "gguf_compat: gguf_init_empty() failed";
        return false;
    }
    gguf_set_kv(patched.get(), src.get());

    // Strip split numbering keys from the overlay so llama_model_loader treats the
    // overlay as n_backbone = 1 and appends the original GGUF file(s) via the
    // `has_extras` shard path (src/llama-model-loader.cpp:604-758).
    gguf_remove_key(patched.get(), "split.no");
    gguf_remove_key(patched.get(), "split.count");
    gguf_remove_key(patched.get(), "split.tensors.count");

    const RopeDimensionSectionsFixup& fixup = *insp.rope_sections;
    if (fixup.is_unsigned) {
        const std::array<std::uint32_t, 4> vals = {
            static_cast<std::uint32_t>(fixup.normalized[0]),
            static_cast<std::uint32_t>(fixup.normalized[1]),
            static_cast<std::uint32_t>(fixup.normalized[2]),
            static_cast<std::uint32_t>(fixup.normalized[3]),
        };
        gguf_set_arr_data(patched.get(), fixup.key.c_str(), GGUF_TYPE_UINT32, vals.data(), vals.size());
    } else {
        gguf_set_arr_data(patched.get(), fixup.key.c_str(), GGUF_TYPE_INT32, fixup.normalized.data(),
                          fixup.normalized.size());
    }

    std::filesystem::path temp_path;
    try {
        temp_path = MakeTempOverlayPath();
    } catch (const std::exception& ex) {
        if (error_out != nullptr) {
            *error_out = std::string("gguf_compat: failed to resolve temp directory: ") + ex.what();
        }
        return false;
    }

    const std::string temp_utf8 = PathToUtf8(temp_path);
    if (!gguf_write_to_file(patched.get(), temp_utf8.c_str(), /*only_meta=*/true)) {
        std::error_code ec;
        std::filesystem::remove(temp_path, ec);
        if (error_out != nullptr) {
            *error_out = "gguf_compat: failed to write temporary metadata overlay: " + temp_utf8;
        }
        return false;
    }

    out_overlay.temp_meta_path_ = std::move(temp_path);
    out_overlay.split_paths_.reserve(1 + orig_splits.size());
    out_overlay.split_paths_.push_back(temp_utf8);
    for (auto& s : orig_splits) {
        out_overlay.split_paths_.push_back(std::move(s));
    }
    out_overlay.inspection_ = std::move(insp);
    return true;
}

llama_model* LoadModelWithGgufCompat(const std::string& model_path, const llama_model_params& params,
                                     GgufCompatOverlay& out_overlay, std::string* error_out) {
    if (!PrepareGgufCompatOverlay(model_path, out_overlay, error_out)) {
        return nullptr;
    }
    if (!out_overlay.active()) {
        return llama_model_load_from_file(model_path.c_str(), params);
    }

    std::vector<const char*> c_paths;
    c_paths.reserve(out_overlay.split_paths().size());
    for (const auto& p : out_overlay.split_paths()) {
        c_paths.push_back(p.c_str());
    }
    llama_model* model = llama_model_load_from_splits(c_paths.data(), c_paths.size(), params);
    if (model == nullptr) {
        out_overlay.Reset();
    }
    return model;
}

}  // namespace sonder::backends::llamacpp
