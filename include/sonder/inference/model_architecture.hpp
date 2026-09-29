// Model architecture classification used by cache and scheduling policy.
#pragma once

#include <algorithm>
#include <cctype>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::inference {

enum class ModelArchitecture {
    attention_only,
    hybrid,
    recurrent,
};

// A small, backend-neutral view of GGUF metadata. Values are strings because
// architecture classification only needs the metadata key/value names and
// must remain independent of llama.cpp's types.
using ModelMetadata = std::vector<std::pair<std::string, std::string>>;

namespace detail {

inline std::string lower_copy(std::string_view value) {
    std::string result(value);
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return result;
}

}  // namespace detail

inline ModelArchitecture classify_model_architecture(const ModelMetadata& metadata) {
    std::string architecture;
    bool recurrent_hint = false;
    bool attention_hint = false;
    for (const auto& [raw_key, raw_value] : metadata) {
        const std::string key = detail::lower_copy(raw_key);
        const std::string value = detail::lower_copy(raw_value);
        if (key == "general.architecture" || key == "architecture") architecture = value;
        recurrent_hint = recurrent_hint || key.find("deltanet") != std::string::npos ||
                         key.find(".ssm_") != std::string::npos || key.find(".ssm.") != std::string::npos ||
                         key.find("_ssm") != std::string::npos || key.find("recurrent") != std::string::npos;
        if (key.find("attention") != std::string::npos || key.find(".attn") != std::string::npos) {
            const bool head_count = key.find("head_count") != std::string::npos || key.find("heads") != std::string::npos;
            if (!head_count) {
                attention_hint = true;
            } else {
                try {
                    attention_hint = attention_hint || std::stoll(value) > 0;
                } catch (...) {
                    // A non-numeric head-count is not evidence of attention.
                }
            }
        }
    }

    const bool named_recurrent = architecture == "mamba" || architecture == "mamba2" ||
                                 architecture == "mamba3" || architecture == "rwkv" ||
                                 architecture == "rwkv6" || architecture == "rwkv7" ||
                                 architecture == "rwkv6qwen2" || architecture == "rwkv5";
    const bool named_hybrid = architecture == "jamba" || architecture == "qwen3_next" ||
                              architecture == "qwen3next" || architecture == "qwen3_next_moe" ||
                              architecture == "qwen3.8" || architecture == "qwen3_8";
    if (named_hybrid) return ModelArchitecture::hybrid;
    if (named_recurrent) return ModelArchitecture::recurrent;
    if (recurrent_hint) return attention_hint ? ModelArchitecture::hybrid : ModelArchitecture::recurrent;
    return ModelArchitecture::attention_only;
}

inline std::string_view to_string(ModelArchitecture architecture) noexcept {
    switch (architecture) {
        case ModelArchitecture::attention_only: return "attention_only";
        case ModelArchitecture::hybrid: return "hybrid";
        case ModelArchitecture::recurrent: return "recurrent";
    }
    return "attention_only";
}

}  // namespace sonder::inference
