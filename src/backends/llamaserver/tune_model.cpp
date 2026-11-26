#include "tune_model.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace sonder::inference::llamaserver::tune {
namespace {
bool is_positive_gguf_int(std::uint64_t type, std::uint64_t raw, std::uint64_t *positive_out = nullptr) noexcept {
    std::uint64_t val = 0;
    switch (type) {
    case 0:  // uint8
    case 2:  // uint16
    case 4:  // uint32
    case 10: // uint64
        val = raw;
        break;
    case 1: { // int8
        const auto s = static_cast<std::int8_t>(raw);
        if (s <= 0) return false;
        val = static_cast<std::uint64_t>(s);
        break;
    }
    case 3: { // int16
        const auto s = static_cast<std::int16_t>(raw);
        if (s <= 0) return false;
        val = static_cast<std::uint64_t>(s);
        break;
    }
    case 5: { // int32
        const auto s = static_cast<std::int32_t>(raw);
        if (s <= 0) return false;
        val = static_cast<std::uint64_t>(s);
        break;
    }
    case 11: { // int64
        const auto s = static_cast<std::int64_t>(raw);
        if (s <= 0) return false;
        val = static_cast<std::uint64_t>(s);
        break;
    }
    default:
        return false;
    }
    if (val == 0) return false;
    if (positive_out) *positive_out = val;
    return true;
}

bool find_flag_token(std::string_view help, std::string_view flag, std::size_t *pos_out = nullptr) noexcept {
    std::size_t pos = 0;
    while ((pos = help.find(flag, pos)) != std::string_view::npos) {
        const bool start_ok = pos == 0 || help[pos - 1] == ' ' || help[pos - 1] == '\t' ||
                              help[pos - 1] == '\n' || help[pos - 1] == '\r' || help[pos - 1] == ',';
        const std::size_t end = pos + flag.size();
        const bool end_ok = end == help.size() || help[end] == ' ' || help[end] == '\t' ||
                            help[end] == '=' || help[end] == '\n' || help[end] == '\r' || help[end] == ',';
        if (start_ok && end_ok) {
            if (pos_out) *pos_out = pos;
            return true;
        }
        pos = end;
    }
    return false;
}

std::string_view spec_type_help_section(std::string_view help) noexcept {
    std::size_t begin = 0;
    if (!find_flag_token(help, "--spec-type", &begin)) return {};
    auto end = help.find('\n', begin);
    while (end != std::string_view::npos) {
        auto line = help.find_first_not_of(" \t\r", end + 1);
        if (line != std::string_view::npos && help[line] == '-') break;
        end = help.find('\n', end + 1);
    }
    return help.substr(begin, end == std::string_view::npos ? 2048 : end - begin);
}

bool is_ident_char(char c) noexcept {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_';
}

bool section_contains_token(std::string_view section, std::string_view token) noexcept {
    if (token.empty()) return false;
    std::size_t pos = 0;
    while ((pos = section.find(token, pos)) != std::string_view::npos) {
        const bool left_ok = pos == 0 || !is_ident_char(section[pos - 1]);
        const std::size_t end = pos + token.size();
        const bool right_ok = end == section.size() || !is_ident_char(section[end]);
        if (left_ok && right_ok) return true;
        pos = end;
    }
    return false;
}

// GGUF v2/v3 directory fields are little endian. All reads/seeks are checked
// against the actual file length and a 256 MiB metadata work limit. Never
// allocate tokenizer arrays or touch the tensor payload.
class DirectoryReader {
  public:
    DirectoryReader(std::istream &in, const std::function<bool()> &cancelled) : in_(in), cancelled_(cancelled) {
        in_.seekg(0, std::ios::end);
        const auto end = in_.tellg();
        if (end < 0) throw std::runtime_error("GGUF stream is not seekable");
        size_ = static_cast<std::uint64_t>(end);
        in_.seekg(0);
    }
    std::uint64_t size() const noexcept { return size_; }
    std::uint64_t integer(std::size_t bytes) {
        std::array<unsigned char, 8> data{};
        bounds(bytes);
        in_.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(bytes));
        if (!in_) throw std::runtime_error("truncated GGUF integer");
        std::uint64_t n = 0;
        for (std::size_t i = 0; i < bytes; ++i) n |= static_cast<std::uint64_t>(data[i]) << (8 * i);
        return n;
    }
    std::string string() {
        const auto n = integer(8);
        if (n > 1024 * 1024) throw std::runtime_error("GGUF string exceeds 1 MiB");
        bounds(n);
        std::string s(static_cast<std::size_t>(n), '\0');
        in_.read(s.data(), static_cast<std::streamsize>(n));
        if (!in_) throw std::runtime_error("truncated GGUF string");
        return s;
    }
    void skip(std::uint64_t n) {
        bounds(n);
        in_.seekg(static_cast<std::streamoff>(n), std::ios::cur);
        if (!in_) throw std::runtime_error("truncated GGUF value");
    }
    std::string value(std::uint64_t type, bool retain, std::uint64_t *raw_scalar = nullptr,
                      std::optional<std::uint64_t> *string_array_count = nullptr) {
        if (type == 8) {
            if (retain) return string();
            const auto n = integer(8);
            skip(n);
            return {};
        }
        if (type == 9) {
            const auto element = integer(4), count = integer(8);
            if (element == 9 || element > 12)
                throw std::runtime_error("invalid or oversized GGUF array");
            if (element == 8) {
                if (count > 2000000)
                    throw std::runtime_error("invalid or oversized GGUF array");
                if (string_array_count) *string_array_count = count;
                for (std::uint64_t i = 0; i < count; ++i) value(8, false);
            } else {
                const auto w = width(element);
                // Scalar arrays (such as prism.hadamard.sign_values) skip in O(1)
                // without allocation; check bounds before multiplying.
                if (count > (1ull << 24) || count > (256ull * 1024 * 1024) / w)
                    throw std::runtime_error("invalid or oversized GGUF array");
                skip(count * w);
            }
            return {};
        }
        const auto n = integer(width(type));
        if (raw_scalar) *raw_scalar = n;
        return retain ? std::to_string(n) : std::string{};
    }
  private:
    static std::size_t width(std::uint64_t type) {
        switch (type) {
        case 0: case 1: case 7: return 1;
        case 2: case 3: return 2;
        case 4: case 5: case 6: return 4;
        case 10: case 11: case 12: return 8;
        default: throw std::runtime_error("unknown GGUF metadata type");
        }
    }
    void bounds(std::uint64_t n) {
        if (cancelled_()) throw std::runtime_error("GGUF inspection cancelled or budget exhausted");
        if (position_ > size_ || n > size_ - position_ || n > 256ull * 1024 * 1024 - position_)
            throw std::runtime_error("truncated GGUF or metadata exceeds 256 MiB");
        position_ += n;
    }
    std::istream &in_;
    const std::function<bool()> &cancelled_;
    std::uint64_t size_ = 0;
    std::uint64_t position_ = 0;
};
} // namespace
Result<ModelInfo> read_model_info(std::istream &input, const std::function<bool()> &cancelled) {
    try {
        DirectoryReader reader(input, cancelled);
        if (reader.integer(4) != 0x46554747) return Status(ErrorCode::invalid_argument, "model is not little-endian GGUF");
        const auto version = reader.integer(4);
        if (version != 2 && version != 3) return Status(ErrorCode::unsupported, "only GGUF v2/v3 are supported");
        const auto tensors = reader.integer(8), entries = reader.integer(8);
        if (tensors > 1000000 || entries > 100000) return Status(ErrorCode::invalid_argument, "GGUF directory exceeds bounds");
        ModelInfo info;
        info.file_bytes = reader.size();
        ModelMetadata metadata;
        std::string arch;
        std::vector<std::pair<std::string, bool>> nextn_predict_keys;
        std::vector<std::pair<std::string, std::uint64_t>> vocab_size_keys;
        std::optional<std::uint64_t> tokens_vocab_size;
        static constexpr std::string_view kNextnSuffix = ".nextn_predict_layers";
        static constexpr std::string_view kVocabSuffix = ".vocab_size";
        static constexpr std::string_view kPrismPrefix = "prism.hadamard.";
        for (std::uint64_t i = 0; i < entries; ++i) {
            auto key = reader.string();
            const auto type = reader.integer(4);
            if (key.size() >= kPrismPrefix.size() && key.compare(0, kPrismPrefix.size(), kPrismPrefix) == 0) {
                info.prism_hadamard = true;
            }
            const bool is_nextn_predict =
                key.size() > kNextnSuffix.size() &&
                key.compare(key.size() - kNextnSuffix.size(), kNextnSuffix.size(), kNextnSuffix) == 0;
            const bool is_vocab_size =
                key.size() > kVocabSuffix.size() &&
                key.compare(key.size() - kVocabSuffix.size(), kVocabSuffix.size(), kVocabSuffix) == 0;
            const bool relevant = key == "general.architecture" || key == "split.count" ||
                key == "tokenizer.ggml.model" ||
                key.find("ssm") != std::string::npos || key.find("attention") != std::string::npos ||
                key.find("attn") != std::string::npos || key.find("deltanet") != std::string::npos ||
                key.find("recurrent") != std::string::npos;
            std::uint64_t raw_scalar = 0;
            std::optional<std::uint64_t> string_array_count;
            auto value = reader.value(type, relevant, (is_nextn_predict || is_vocab_size) ? &raw_scalar : nullptr,
                                      key == "tokenizer.ggml.tokens" ? &string_array_count : nullptr);
            if (key == "general.architecture" && type == 8) {
                arch = value;
                info.gguf_architecture = value;
            }
            if (key == "tokenizer.ggml.model" && type == 8) {
                info.tokenizer_model = value;
            }
            if (key == "tokenizer.ggml.tokens" && string_array_count && *string_array_count > 0) {
                tokens_vocab_size = *string_array_count;
            }
            if (is_nextn_predict) {
                nextn_predict_keys.emplace_back(key, is_positive_gguf_int(type, raw_scalar));
            }
            if (is_vocab_size) {
                std::uint64_t positive_vocab = 0;
                if (is_positive_gguf_int(type, raw_scalar, &positive_vocab)) {
                    vocab_size_keys.emplace_back(key, positive_vocab);
                }
            }
            if (key == "split.count") {
                if (value != "0" && value != "1") {
                    return Status(ErrorCode::unsupported,
                                  "calibration requires a single-file GGUF (nextn/weights may be in another shard)");
                }
                info.split_count = value == "1" ? 1 : 0;
            }
            if (relevant) metadata.emplace_back(std::move(key), std::move(value));
        }
        info.architecture = classify_model_architecture(metadata);
        if (tokens_vocab_size) {
            info.vocab_size = *tokens_vocab_size;
        } else if (!arch.empty()) {
            const std::string expected_vocab_key = arch + std::string(kVocabSuffix);
            for (const auto &[key, sz] : vocab_size_keys) {
                if (key == expected_vocab_key) {
                    info.vocab_size = sz;
                    break;
                }
            }
        }
        if (!arch.empty()) {
            const std::string expected_key = arch + std::string(kNextnSuffix);
            for (const auto &[key, positive] : nextn_predict_keys) {
                if (key == expected_key && positive) {
                    info.has_nextn = true;
                    break;
                }
            }
        }
        for (std::uint64_t i = 0; i < tensors; ++i) {
            const auto name = reader.string();
            const auto dimensions = reader.integer(4);
            if (dimensions == 0 || dimensions > 4) return Status(ErrorCode::invalid_argument, "invalid GGUF tensor dimensions");
            for (std::uint64_t d = 0; d < dimensions; ++d)
                if (reader.integer(8) == 0) return Status(ErrorCode::invalid_argument, "empty GGUF tensor dimension");
            reader.skip(12); // tensor type (u32), data offset (u64)
            if (name.find(".nextn.") != std::string::npos) info.has_nextn = true;
        }
        return info;
    } catch (const std::exception &e) {
        return Status(ErrorCode::invalid_argument, e.what());
    }
}
bool help_supports_spec_type(std::string_view help) {
    return find_flag_token(help, "--spec-type");
}
bool help_supports_mtp(std::string_view help) {
    // Restrict draft-mtp to the --spec-type option's section. Seeing the term
    // somewhere else (e.g. a changelog) is insufficient.
    const auto section = spec_type_help_section(help);
    if (section.empty()) return false;
    return section_contains_token(section, "draft-mtp") && find_flag_token(help, "--spec-draft-n-max");
}
bool help_supports_draft_model(std::string_view help) {
    if (!help_supports_spec_type(help) || !find_flag_token(help, "--spec-draft-n-max")) return false;
    return find_flag_token(help, "--spec-draft-model") || find_flag_token(help, "--model-draft") ||
           find_flag_token(help, "-md");
}
bool help_supports_spec_type_value(std::string_view help, std::string_view spec_type) {
    if (spec_type.empty()) return false;
    const auto section = spec_type_help_section(help);
    if (section.empty()) return false;
    const bool enumerates_types =
        section_contains_token(section, "none") || section.find("draft-") != std::string_view::npos ||
        section.find("ngram-") != std::string_view::npos;
    if (!enumerates_types) return true;

    std::size_t start = 0;
    while (start <= spec_type.size()) {
        const auto comma = spec_type.find(',', start);
        const std::string_view tok =
            spec_type.substr(start, comma == std::string_view::npos ? std::string_view::npos : comma - start);
        if (tok.empty()) return false;
        bool ok = section_contains_token(section, tok);
        if (!ok && tok == "dflash") ok = section_contains_token(section, "draft-dflash");
        if (!ok && tok == "draft") ok = section_contains_token(section, "draft-simple");
        if (!ok) return false;
        if (comma == std::string_view::npos) break;
        start = comma + 1;
    }
    return true;
}
} // namespace sonder::inference::llamaserver::tune
