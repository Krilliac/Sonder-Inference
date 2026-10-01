#pragma once
#include <filesystem>
#include <string_view>

namespace sonder::inference::llamaserver {
inline std::filesystem::path utf8_path(std::string_view text) {
    return std::filesystem::path(std::u8string(text.begin(), text.end()));
}
} // namespace sonder::inference::llamaserver
