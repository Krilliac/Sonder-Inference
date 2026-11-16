#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace sonder::inference::detail {

// Returns `text` limited to `max_bytes`, cut on a UTF-8 code point boundary,
// with "..." appended when anything was dropped. Used to echo
// caller-controlled config keys in error messages without unbounded growth.
inline std::string bounded_for_message(std::string_view text, std::size_t max_bytes = 64) {
    if (text.size() <= max_bytes)
        return std::string(text);
    std::size_t cut = max_bytes;
    // Step back over UTF-8 continuation bytes (10xxxxxx) so no sequence is split.
    while (cut > 0 && (static_cast<unsigned char>(text[cut]) & 0xC0u) == 0x80u)
        --cut;
    std::string out(text.substr(0, cut));
    out += "...";
    return out;
}

}  // namespace sonder::inference::detail
