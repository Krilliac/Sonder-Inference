#pragma once
#include <functional>
#include <istream>
#include "sonder/inference/error.hpp"
#include "sonder/inference/model_architecture.hpp"

namespace sonder::inference::llamaserver::tune {
struct ModelInfo {
    ModelArchitecture architecture = ModelArchitecture::attention_only;
    bool has_nextn = false;
};
// No weight loading. The optional embedded backend's reader takes a loaded
// llama_model, so the external-server build needs this bounded GGUF directory
// reader. Architecture classification itself is shared with that backend.
Result<ModelInfo> read_model_info(std::istream &input, const std::function<bool()> &cancelled = [] { return false; });
bool help_supports_mtp(std::string_view help);
} // namespace sonder::inference::llamaserver::tune
