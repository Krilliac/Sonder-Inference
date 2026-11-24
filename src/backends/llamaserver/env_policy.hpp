// Shared child-environment containment policy for spawned llama-server processes.
// Header-only so process_options.cpp, llamaserver_config_extensions.hpp,
// serve_main.cpp and fuzz_backend_config.cpp enforce identical rules.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "sonder/inference/error.hpp"

namespace sonder::inference::llamaserver {

namespace detail {

inline bool ascii_upper_into(std::string_view key, std::string& out) {
    out.clear();
    out.reserve(key.size());
    for (const char ch : key) {
        const auto u = static_cast<unsigned char>(ch);
        if (u <= 0x20 || u >= 0x7F || ch == '=') {
            return false;
        }
        out.push_back((ch >= 'a' && ch <= 'z') ? static_cast<char>(ch - 'a' + 'A') : ch);
    }
    return !out.empty();
}

inline bool ascii_upper_into(std::wstring_view key, std::string& out) {
    out.clear();
    out.reserve(key.size());
    for (const wchar_t ch : key) {
        if (ch <= 0x20 || ch >= 0x7F || ch == L'=') {
            return false;
        }
        const char c = static_cast<char>(ch);
        out.push_back((c >= 'a' && c <= 'z') ? static_cast<char>(c - 'a' + 'A') : c);
    }
    return !out.empty();
}

inline bool starts_with(std::string_view s, std::string_view prefix) noexcept {
    return s.size() >= prefix.size() && s.substr(0, prefix.size()) == prefix;
}

inline bool ends_with(std::string_view s, std::string_view suffix) noexcept {
    return s.size() >= suffix.size() && s.substr(s.size() - suffix.size()) == suffix;
}

inline bool is_credential_suffix(std::string_view upper) noexcept {
    static constexpr std::string_view kSensitiveSuffixes[] = {
        "_TOKEN",      "_API_KEY",     "_SECRET",     "_PASSWORD", "_SECRET_KEY", "_APIKEY",
        "_ACCESS_KEY", "_PRIVATE_KEY", "_AUTH",       "_AUTH_KEY", "_PASSWD",     "_PASS",
        "_CREDENTIAL", "_CREDENTIALS", "_BEARER",     "_PROXY",    "_CERT_FILE",  "_CA_BUNDLE",
    };
    for (const std::string_view suffix : kSensitiveSuffixes) {
        if (ends_with(upper, suffix)) {
            return true;
        }
    }
    return false;
}

// Explicit denylist of GPU runtime DLL/.so loaders, layer/driver injection hooks,
// and file-write log path variables. Checked before any prefix allowlist so
// broad vendor prefixes (GGML_*, CUDA_*, HSA_*, VK_*, MESA_*, AMD_*, ROCBLAS_*,
// HIPBLASLT_*) can never load arbitrary libraries or write arbitrary files.
inline bool is_gpu_loader_or_log_hook_upper(std::string_view upper) noexcept {
    if (starts_with(upper, "ROCBLAS_LOG_")) {
        return true;
    }
    static constexpr std::string_view kBlockedHooks[] = {
        "GGML_BACKEND_PATH",
        "CUDA_INJECTION64_PATH",
        "CUDA_INJECTION32_PATH",
        "HSA_TOOLS_LIB",
        "VK_LAYER_PATH",
        "VK_ADD_LAYER_PATH",
        "VK_INSTANCE_LAYERS",
        "VK_LOADER_LAYERS_ENABLE",
        "VK_DRIVER_FILES",
        "VK_ADD_DRIVER_FILES",
        "MESA_LOADER_DRIVER_OVERRIDE",
        "AMD_LOG_LEVEL_FILE",
        "HIPBLASLT_LOG_FILE",
    };
    for (const std::string_view hook : kBlockedHooks) {
        if (upper == hook) {
            return true;
        }
    }
    return false;
}

inline bool is_gpu_runtime_prefix(std::string_view upper) noexcept {
    return starts_with(upper, "CUDA_") || starts_with(upper, "CUDNN_") || starts_with(upper, "GGML_") ||
           starts_with(upper, "VK_") || starts_with(upper, "ROCR_") || starts_with(upper, "ROCM_") ||
           starts_with(upper, "HSA_") || starts_with(upper, "HIP_") || starts_with(upper, "AMD_") ||
           starts_with(upper, "ROCBLAS_") || starts_with(upper, "HIPBLASLT_") || starts_with(upper, "RADV_") ||
           starts_with(upper, "MESA_") || starts_with(upper, "NVIDIA_") || starts_with(upper, "SYCL_") ||
           starts_with(upper, "ZE_") || starts_with(upper, "METAL_") || starts_with(upper, "MTL_");
}

inline bool is_gpu_runtime_exact(std::string_view upper) noexcept {
    static constexpr std::string_view kGpuRuntimeExact[] = {
        "GPU_MAX_ALLOC_PERCENT",
        "GPU_MAX_HEAP_SIZE",
        "GPU_SINGLE_ALLOC_PERCENT",
        "GPU_FORCE_64BIT_PTR",
        "GPU_USE_SYNC_OBJECTS",
        "DRI_PRIME",
    };
    for (const std::string_view exact : kGpuRuntimeExact) {
        if (upper == exact) {
            return true;
        }
    }
    return false;
}

inline bool is_forbidden_override_upper(std::string_view upper) noexcept {
    if (starts_with(upper, "AIP_") || starts_with(upper, "LLAMA_") || starts_with(upper, "HF_") ||
        starts_with(upper, "HUGGING_FACE_") || starts_with(upper, "HUGGINGFACE_") || starts_with(upper, "MCP_") ||
        starts_with(upper, "LD_") || starts_with(upper, "DYLD_") || is_credential_suffix(upper) ||
        is_gpu_loader_or_log_hook_upper(upper)) {
        return true;
    }
    static constexpr std::string_view kForbiddenExact[] = {
        "MODEL_ENDPOINT", "APPDATA",         "PROGRAMDATA",     "LOCALAPPDATA",      "USERPROFILE",
        "XDG_CONFIG_HOME", "XDG_CONFIG_DIRS", "HOME",           "PATH",              "PATHEXT",
        "COMSPEC",        "SYSTEMROOT",      "WINDIR",          "SHELL",             "HTTP_PROXY",
        "HTTPS_PROXY",    "ALL_PROXY",       "NO_PROXY",        "FTP_PROXY",         "SSL_CERT_FILE",
        "SSL_CERT_DIR",   "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE",
    };
    for (const std::string_view exact : kForbiddenExact) {
        if (upper == exact) {
            return true;
        }
    }
    return false;
}

inline bool is_allowed_inherited_upper(std::string_view upper) noexcept {
    if (is_credential_suffix(upper) || is_gpu_loader_or_log_hook_upper(upper) || starts_with(upper, "LLAMA_") ||
        starts_with(upper, "HF_") || starts_with(upper, "HUGGING_FACE_") || starts_with(upper, "HUGGINGFACE_") ||
        starts_with(upper, "AIP_") || starts_with(upper, "MCP_") || starts_with(upper, "SONDER_")) {
        return false;
    }
    if (is_gpu_runtime_prefix(upper) || is_gpu_runtime_exact(upper) || starts_with(upper, "LC_") ||
        starts_with(upper, "OMP_")) {
        return true;
    }
    static constexpr std::string_view kAllowedExact[] = {
        "PATH",
        "SYSTEMROOT",
        "SYSTEMDRIVE",
        "WINDIR",
        "COMSPEC",
        "PATHEXT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "HOME",
        "USERPROFILE",
        "APPDATA",
        "LOCALAPPDATA",
        "TZ",
        "LANG",
        "TERM",
        "NUMBER_OF_PROCESSORS",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_IDENTIFIER",
        "PROCESSOR_LEVEL",
        "PROCESSOR_REVISION",
        "OS",
        "LD_LIBRARY_PATH",
        "DYLD_LIBRARY_PATH",
        "DYLD_FALLBACK_LIBRARY_PATH",
        "DYLD_FRAMEWORK_PATH",
        "DYLD_FALLBACK_FRAMEWORK_PATH",
        "VULKAN_SDK",
        "ONEAPI_ROOT",
        "XDG_RUNTIME_DIR",
        "XDG_CACHE_HOME",
        "DISPLAY",
        "WAYLAND_DISPLAY",
    };
    for (const std::string_view exact : kAllowedExact) {
        if (upper == exact) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

// True if an inherited parent environment variable is on the minimal explicit
// allowlist of OS/GPU/runtime variables permitted to reach a spawned child.
// Windows environment names are matched case-insensitively; POSIX names are
// matched against exact uppercase allowlist entries.
template <class StringView>
inline bool is_allowed_inherited_env_name(StringView key) {
    std::string upper;
    if (!detail::ascii_upper_into(key, upper)) {
        return false;
    }
#if !defined(_WIN32)
    // On POSIX, environment names are case-sensitive: require the key itself
    // to be uppercase so lowercase variants like http_proxy never slip in.
    for (std::size_t i = 0; i < key.size(); ++i) {
        if (static_cast<char>(key[i]) != upper[i]) {
            return false;
        }
    }
#endif
    return detail::is_allowed_inherited_upper(upper);
}

// Validates a list of configured child-only environment overrides. Rejects
// malformed names, bounds violations, and case-normalized llama-server
// arguments, credentials, download/tool/proxy/config-home/loader overrides.
inline Status check_child_environment_overrides(const std::vector<std::pair<std::string, std::string>>& env) {
    if (env.size() > 128) {
        return {ErrorCode::invalid_argument, "child environment overrides exceed bounds"};
    }
    std::size_t bytes = 0;
    std::string upper;
    for (const auto& [key, value] : env) {
        if (!detail::ascii_upper_into(key, upper) || value.find('\0') != std::string::npos) {
            return {ErrorCode::invalid_argument, "invalid child environment entry"};
        }
        if (detail::starts_with(upper, "AIP_")) {
            return {ErrorCode::invalid_argument,
                    "child environment override '" + key +
                        "' is refused: AIP_* variables enable llama-server's Vertex AI mode, which would replace the "
                        "supervisor-owned port"};
        }
        if (detail::is_forbidden_override_upper(upper)) {
            return {ErrorCode::invalid_argument,
                    "child environment override '" + key +
                        "' is refused: llama-server arguments, credentials, downloads, proxy, loader and config-home "
                        "variables cannot be set via environment overrides"};
        }
        bytes += key.size() + value.size() + 2;
        if (bytes > 32760) {
            return {ErrorCode::invalid_argument, "child environment overrides exceed bounds"};
        }
    }
    return {};
}

// Stricter allowlist for launch-profile launches: only explicit GPU/runtime
// tuning variables may be combined with `--profile`.
inline bool is_allowed_profile_env_override(std::string_view key) {
    std::string upper;
    if (!detail::ascii_upper_into(key, upper)) {
        return false;
    }
    if (detail::is_forbidden_override_upper(upper)) {
        return false;
    }
    return detail::is_gpu_runtime_prefix(upper) || detail::is_gpu_runtime_exact(upper) ||
           detail::starts_with(upper, "OMP_");
}

}  // namespace sonder::inference::llamaserver
