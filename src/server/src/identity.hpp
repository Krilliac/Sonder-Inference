// Internal: backend identity for GET /v1/sonder/identity (docs/SERVER.md).
//
// The object has exactly the nine keys of Sonder Runtime's BackendIdentity
// (backend, model, model_digest, quantization, backend_version,
// tokenizer_digest, template_digest, context_tokens, hardware). Digests are
// lowercase 64-hex SHA-256 and context_tokens is > 0. When any value cannot
// be measured the identity is null and `reason` says why; nothing is
// fabricated.
#pragma once

#include <optional>
#include <string>

#include "sonder/inference/backend.hpp"
#include "sonder/inference/json.hpp"
#include "sonder/inference/model.hpp"

namespace sonder::inference::server::detail {

struct IdentityResult {
    json::Value backend_identity;  // object, or null
    std::optional<std::string> reason;
};

// `backend_version` is the backend's probe() result when it succeeded.
IdentityResult backend_identity(const Model& model, const std::optional<std::string>& backend_version);

// Same, from a backend name and the model descriptor. `descriptor` is null
// when the model has not been loaded yet (lazy residency); backends whose
// identity needs the descriptor then report null with a reason.
IdentityResult backend_identity(const std::string& backend, const std::string& model_id,
                                const ModelDescriptor* descriptor, const std::optional<std::string>& backend_version);

}  // namespace sonder::inference::server::detail
