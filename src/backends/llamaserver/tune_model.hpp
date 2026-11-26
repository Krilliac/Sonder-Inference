#pragma once
#include <cstdint>
#include <functional>
#include <istream>
#include <optional>
#include <string>
#include <string_view>
#include "sonder/inference/error.hpp"
#include "sonder/inference/model_architecture.hpp"

namespace sonder::inference::llamaserver::tune {
struct ModelInfo {
    ModelArchitecture architecture = ModelArchitecture::attention_only;
    std::string gguf_architecture;
    std::string tokenizer_model;
    std::optional<std::uint64_t> vocab_size;
    bool has_nextn = false;
    bool prism_hadamard = false;
    std::uint64_t split_count = 0;
    std::uint64_t file_bytes = 0;
};
// No weight loading. The optional embedded backend's reader takes a loaded
// llama_model, so the external-server build needs this bounded GGUF directory
// reader. Architecture classification itself is shared with that backend.
Result<ModelInfo> read_model_info(std::istream &input, const std::function<bool()> &cancelled = [] { return false; });
bool help_supports_spec_type(std::string_view help);
bool help_supports_mtp(std::string_view help);
bool help_supports_draft_model(std::string_view help);
bool help_supports_spec_type_value(std::string_view help, std::string_view spec_type);
} // namespace sonder::inference::llamaserver::tune
