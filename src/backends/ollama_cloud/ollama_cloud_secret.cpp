// Ollama Cloud API key loading and redaction.
#include "ollama_cloud_secret.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <utility>

namespace sonder::inference::ollama_cloud {

namespace {

constexpr std::size_t kMaxKeyBytes = 4096;
constexpr const char* kRedacted = "[redacted]";

bool valid_env_name(const std::string& name) {
    return !name.empty() && name.size() <= 128 && std::isdigit(static_cast<unsigned char>(name.front())) == 0 &&
           std::all_of(name.begin(), name.end(),
                       [](unsigned char c) { return std::isalnum(c) != 0 || c == '_'; });
}

std::optional<std::string> process_env(const char* name) {
#if defined(_WIN32)
    char* buffer = nullptr;
    std::size_t length = 0;
    if (_dupenv_s(&buffer, &length, name) != 0 || buffer == nullptr) {
        return std::nullopt;
    }
    std::string value(buffer);
    // Wipe the CRT copy before freeing it.
    volatile char* p = buffer;
    for (std::size_t i = 0; i < value.size(); ++i) {
        p[i] = '\0';
    }
    std::free(buffer);
    return value;
#else
    const char* raw = std::getenv(name);
    if (raw == nullptr) {
        return std::nullopt;
    }
    return std::string(raw);
#endif
}

}  // namespace

void secure_wipe(std::string& text) noexcept {
    volatile char* p = text.data();
    for (std::size_t i = 0; i < text.size(); ++i) {
        p[i] = '\0';
    }
    text.clear();
}

ApiKey::~ApiKey() { secure_wipe(value_); }

// Moves swap the buffers and then wipe the source: a plain std::string move
// can leave the bytes of a short (SSO) key behind in the moved-from object.
ApiKey::ApiKey(ApiKey&& other) noexcept {
    value_.swap(other.value_);
    secure_wipe(other.value_);
}

ApiKey& ApiKey::operator=(ApiKey&& other) noexcept {
    if (this != &other) {
        value_.swap(other.value_);
        secure_wipe(other.value_);
    }
    return *this;
}

Result<ApiKey> ApiKey::from_value(std::string&& input, std::string_view source) {
    std::string value;
    value.swap(input);  // no copy, so no unwiped residue in the caller's string
    if (value.empty()) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: " + std::string(source) + " is empty");
    }
    const bool printable = std::all_of(value.begin(), value.end(),
                                       [](unsigned char c) { return c >= 0x21 && c <= 0x7e; });
    if (value.size() > kMaxKeyBytes || !printable) {
        secure_wipe(value);
        return Status(ErrorCode::invalid_argument,
                      "ollama_cloud: " + std::string(source) + " does not hold a valid API key");
    }
    return ApiKey(std::move(value));
}

Result<ApiKey> ApiKey::from_env(const std::string& name, const EnvLookup& lookup) {
    if (!valid_env_name(name)) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: api_key_env is not a valid variable name");
    }
    std::optional<std::string> value = lookup ? lookup(name.c_str()) : process_env(name.c_str());
    if (!value || value->empty()) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: environment variable " + name + " is not set");
    }
    std::string raw;
    raw.swap(*value);
    return from_value(std::move(raw), "environment variable " + name);
}

Result<ApiKey> ApiKey::from_file(const std::string& path) {
    if (path.empty() || path.find('\0') != std::string::npos) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: api_key_file is not a valid path");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        return Status(ErrorCode::invalid_argument, "ollama_cloud: cannot read api_key_file " + path);
    }
    std::string text(kMaxKeyBytes + 2, '\0');
    in.read(text.data(), static_cast<std::streamsize>(text.size()));
    if (in.bad()) {
        secure_wipe(text);
        return Status(ErrorCode::invalid_argument, "ollama_cloud: cannot read api_key_file " + path);
    }
    text.resize(static_cast<std::size_t>(in.gcount()));
    const auto is_space = [](unsigned char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
    std::size_t begin = 0;
    std::size_t end = text.size();
    while (begin < end && is_space(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    while (end > begin && is_space(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    std::string value = text.substr(begin, end - begin);
    secure_wipe(text);
    return from_value(std::move(value), "api_key_file " + path);
}

std::string ApiKey::bearer_header_value() const { return "Bearer " + value_; }

void ApiKey::scrub(std::string& text) const {
    if (value_.empty()) {
        return;
    }
    std::size_t pos = 0;
    while ((pos = text.find(value_, pos)) != std::string::npos) {
        text.replace(pos, value_.size(), kRedacted);
        pos += std::char_traits<char>::length(kRedacted);
    }
}

Status ApiKey::scrub(const Status& status) const {
    if (status.ok()) {
        return status;
    }
    std::string message = status.message();
    scrub(message);
    return Status(status.code(), std::move(message));
}

}  // namespace sonder::inference::ollama_cloud
