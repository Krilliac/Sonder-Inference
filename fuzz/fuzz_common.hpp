// Shared helpers for Sonder libFuzzer targets (fuzz/README.md).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace sonder_fuzz {

inline std::string_view as_view(const std::uint8_t* data, std::size_t size) {
    return {reinterpret_cast<const char*>(data), size};
}

// Invariant violation: print and abort so libFuzzer records a crash artifact.
[[noreturn]] inline void fail(const char* what) {
    std::fprintf(stderr, "sonder fuzz invariant violated: %s\n", what);
    std::abort();
}

#define SONDER_FUZZ_CHECK(cond) \
    do {                        \
        if (!(cond)) ::sonder_fuzz::fail(#cond); \
    } while (0)

// True if `s` is well-formed UTF-8 (no overlongs, surrogates or > U+10FFFF).
inline bool is_valid_utf8(std::string_view s) {
    std::size_t i = 0;
    while (i < s.size()) {
        const auto c = static_cast<unsigned char>(s[i]);
        std::size_t n = 0;
        std::uint32_t cp = 0;
        if (c < 0x80) { ++i; continue; }
        if ((c & 0xE0) == 0xC0) { n = 1; cp = c & 0x1F; }
        else if ((c & 0xF0) == 0xE0) { n = 2; cp = c & 0x0F; }
        else if ((c & 0xF8) == 0xF0) { n = 3; cp = c & 0x07; }
        else return false;
        if (i + n >= s.size()) return false;
        for (std::size_t k = 1; k <= n; ++k) {
            const auto cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if ((n == 1 && cp < 0x80) || (n == 2 && cp < 0x800) || (n == 3 && cp < 0x10000)) return false;
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += n + 1;
    }
    return true;
}

}  // namespace sonder_fuzz
