#include "tune_model.hpp"
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace sonder::inference::llamaserver::tune {
namespace {
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
    std::string value(std::uint64_t type, bool retain) {
        if (type == 8) {
            if (retain) return string();
            const auto n = integer(8);
            skip(n);
            return {};
        }
        if (type == 9) {
            const auto element = integer(4), count = integer(8);
            if (count > 2000000 || element == 9 || element > 12)
                throw std::runtime_error("invalid or oversized GGUF array");
            if (element == 8) {
                for (std::uint64_t i = 0; i < count; ++i) value(8, false);
            } else skip(count * width(element));
            return {};
        }
        const auto n = integer(width(type));
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
        ModelMetadata metadata;
        for (std::uint64_t i = 0; i < entries; ++i) {
            auto key = reader.string();
            const auto type = reader.integer(4);
            const bool relevant = key == "general.architecture" || key == "split.count" ||
                key.find("ssm") != std::string::npos || key.find("attention") != std::string::npos ||
                key.find("attn") != std::string::npos || key.find("deltanet") != std::string::npos ||
                key.find("recurrent") != std::string::npos;
            auto value = reader.value(type, relevant);
            if (key == "split.count" && value != "0" && value != "1")
                return Status(ErrorCode::unsupported, "calibration requires a single-file GGUF (nextn may be in another shard)");
            if (relevant) metadata.emplace_back(std::move(key), std::move(value));
        }
        ModelInfo info;
        info.architecture = classify_model_architecture(metadata);
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
bool help_supports_mtp(std::string_view help) {
    // Restrict draft-mtp to the --spec-type option's section. Seeing the term
    // somewhere else (e.g. a changelog) is insufficient.
    const auto begin = help.find("--spec-type");
    if (begin == std::string_view::npos) return false;
    const auto flag_end = begin + std::string_view("--spec-type").size();
    if (flag_end < help.size() && help[flag_end] != ' ' && help[flag_end] != '\t' && help[flag_end] != '=') return false;
    auto end = help.find('\n', begin);
    while (end != std::string_view::npos) {
        auto line = help.find_first_not_of(" \t\r", end + 1);
        if (line != std::string_view::npos && help[line] == '-') break;
        end = help.find('\n', end + 1);
    }
    const auto section = help.substr(begin, end == std::string_view::npos ? 2048 : end - begin);
    return section.find("draft-mtp") != std::string_view::npos && help.find("--spec-draft-n-max") != std::string_view::npos;
}
} // namespace sonder::inference::llamaserver::tune
