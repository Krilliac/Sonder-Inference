#include "identity.hpp"

#include "sha256.hpp"
#include "sonder/inference/backends.hpp"
#include "sonder/inference/device.hpp"

namespace sonder::inference::server::detail {

namespace {

// MOCK backend descriptors. The mock performs no inference, so its identity
// is a set of digests over fixed strings that describe exactly what it does:
// synthetic, but stable and honest. The server labels it synthetic.
constexpr const char* kMockTokenizer =
    "sonder-inference mock tokenizer v1: whitespace-separated words; word list from src/backends/mock_backend.cpp";
constexpr const char* kGenericChatTemplate =
    "sonder-inference generic chat prompt v1 (format_chat_prompt): '<Role>: <content>\\n\\n' per message, "
    "then 'Assistant:'";

std::string cpu_hardware() {
    for (const auto& d : enumerate_devices()) {
        if (d.kind == DeviceKind::cpu) {
            return d.id + " " + d.name + " (" + host_platform() + ")";
        }
    }
    return host_platform();
}

}  // namespace

IdentityResult backend_identity(const Model& model, const std::optional<std::string>& backend_version) {
    IdentityResult out;
    const ModelDescriptor& d = model.descriptor();
    const std::string& backend = model.backend_name();
    if (backend == kMockBackendName) {
        if (d.context_length == 0) {
            out.reason = "mock model reports no context length";
            return out;
        }
        out.backend_identity = json::Object{
            {"backend", backend},
            {"model", d.name},
            {"model_digest", sha256_hex("sonder-inference mock model v1: name=" + d.name + " format=" + d.format +
                                        " family=" + d.family + " quantization=" + d.quantization)},
            {"quantization", d.quantization.empty() ? std::string("none") : d.quantization},
            {"backend_version", backend_version.value_or("mock")},
            {"tokenizer_digest", sha256_hex(kMockTokenizer)},
            {"template_digest", sha256_hex(kGenericChatTemplate)},
            {"context_tokens", d.context_length},
            {"hardware", "mock backend on " + cpu_hardware() + "; no inference is performed"}};
        return out;
    }
    if (backend == "ollama") {
        out.reason =
            "ollama: the Ollama API does not expose a measurable tokenizer digest, so a complete backend identity "
            "cannot be reported (API v1 returns null rather than a partial or guessed identity)";
        return out;
    }
    if (backend == "llamacpp") {
        out.reason = "llamacpp: GGUF file hashing is not implemented, so model, tokenizer and template digests are "
                     "not measured";
        return out;
    }
    out.reason = backend + ": backend identity is not implemented for this backend";
    return out;
}

}  // namespace sonder::inference::server::detail
