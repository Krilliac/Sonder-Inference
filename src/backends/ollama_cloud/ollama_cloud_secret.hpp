// Ollama Cloud API key holder (internal). The key is read once from an
// environment variable or a token file and only leaves this type as the
// Authorization header value handed to the transport.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "sonder/inference/error.hpp"

namespace sonder::inference::ollama_cloud {

using EnvLookup = std::function<std::optional<std::string>(const char* name)>;

// Overwrites the string's bytes (through a volatile pointer, so the stores
// are not elided) and clears it. Best effort: copies made elsewhere, e.g.
// inside the TLS library's record buffers, are not covered (the HTTP
// client wipes its own serialized request head).
void secure_wipe(std::string& text) noexcept;

// Move-only owner of the key. Not thread-safe to mutate; const use (header
// value, scrubbing) is safe from any thread once constructed.
class ApiKey {
public:
    ApiKey() = default;
    ~ApiKey();
    ApiKey(const ApiKey&) = delete;
    ApiKey& operator=(const ApiKey&) = delete;
    ApiKey(ApiKey&& other) noexcept;
    ApiKey& operator=(ApiKey&& other) noexcept;

    // Errors (invalid_argument) name the variable, never its value.
    static Result<ApiKey> from_env(const std::string& name, const EnvLookup& lookup);
    // Reads at most 4 KiB; surrounding whitespace (a trailing newline) is
    // trimmed. Errors name the path, never the content.
    static Result<ApiKey> from_file(const std::string& path);
    // Validates a raw value: 1..4096 bytes of printable ASCII (0x21..0x7e).
    // `source` names where it came from, for the error message only.
    static Result<ApiKey> from_value(std::string&& value, std::string_view source);

    [[nodiscard]] bool empty() const noexcept { return value_.empty(); }
    // "Bearer <key>". The caller owns the copy and should secure_wipe() it.
    [[nodiscard]] std::string bearer_header_value() const;
    // Replaces every occurrence of the key in `text`, raw or in its JSON
    // string-escaped form, with "[redacted]". Apply it to the whole text
    // before cutting it: a partial key is not recognised.
    void scrub(std::string& text) const;
    [[nodiscard]] Status scrub(const Status& status) const;

private:
    explicit ApiKey(std::string&& value) noexcept { value_.swap(value); }
    std::string value_;
};

}  // namespace sonder::inference::ollama_cloud
