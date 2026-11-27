#include "slot_store.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <random>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <unistd.h>
#endif

// Upstream llama.cpp v1.0.0 (b1b770d) C headers:
//   - ggml/include/gguf.h:72 (gguf_init_params), 80 (gguf_reader_callback_t),
//     88 (gguf_init_from_callback), 90 (gguf_free), 96 (gguf_get_data_offset)
//   - include/llama.h:70 (llama_seq_id), 856 (llama_state_seq_get_size),
//     861 (llama_state_seq_get_data), 871 (llama_state_seq_set_data)
#include "gguf.h"
#include "llama.h"

namespace sonder::backends::llamacpp {

namespace {

namespace fs = std::filesystem;

constexpr std::size_t kIoChunkBytes = 65536u;

// Self-contained FIPS 180-4 SHA-256 so `sonder_backend_llamacpp` has no reverse
// static-archive link dependency on `sonder_inference`.
class Sha256 {
public:
    Sha256() noexcept
        : h_{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
             0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u} {}

    void Update(std::span<const std::uint8_t> data) noexcept {
        total_bytes_ += data.size();
        const std::uint8_t* p = data.data();
        std::size_t n = data.size();
        while (n > 0) {
            const std::size_t take = std::min<std::size_t>(64u - buf_len_, n);
            std::memcpy(buf_.data() + buf_len_, p, take);
            buf_len_ += take;
            p += take;
            n -= take;
            if (buf_len_ == 64u) {
                ProcessBlock(buf_.data());
                buf_len_ = 0;
            }
        }
    }

    [[nodiscard]] std::array<std::uint8_t, 32> Finish() noexcept {
        const std::uint64_t bits = total_bytes_ * 8u;
        const std::uint8_t pad = 0x80u;
        Update(std::span<const std::uint8_t>(&pad, 1u));
        const std::uint8_t zero = 0u;
        while (buf_len_ != 56u) {
            Update(std::span<const std::uint8_t>(&zero, 1u));
        }
        std::array<std::uint8_t, 8> len_be{};
        for (std::size_t i = 0; i < 8u; ++i) {
            len_be[i] = static_cast<std::uint8_t>(bits >> (56u - 8u * i));
        }
        Update(len_be);

        std::array<std::uint8_t, 32> digest{};
        for (std::size_t i = 0; i < 8u; ++i) {
            digest[4u * i] = static_cast<std::uint8_t>(h_[i] >> 24u);
            digest[4u * i + 1u] = static_cast<std::uint8_t>(h_[i] >> 16u);
            digest[4u * i + 2u] = static_cast<std::uint8_t>(h_[i] >> 8u);
            digest[4u * i + 3u] = static_cast<std::uint8_t>(h_[i]);
        }
        return digest;
    }

private:
    static constexpr std::array<std::uint32_t, 64> kRoundConstants = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
        0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
        0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
        0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
        0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
        0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
        0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
    };

    static constexpr std::uint32_t Rotr(std::uint32_t x, unsigned n) noexcept {
        return (x >> n) | (x << (32u - n));
    }

    void ProcessBlock(const std::uint8_t* p) noexcept {
        std::array<std::uint32_t, 64> w{};
        for (std::size_t i = 0; i < 16u; ++i) {
            w[i] = (static_cast<std::uint32_t>(p[4u * i]) << 24u) |
                   (static_cast<std::uint32_t>(p[4u * i + 1u]) << 16u) |
                   (static_cast<std::uint32_t>(p[4u * i + 2u]) << 8u) |
                   static_cast<std::uint32_t>(p[4u * i + 3u]);
        }
        for (std::size_t i = 16u; i < 64u; ++i) {
            const std::uint32_t s0 = Rotr(w[i - 15u], 7u) ^ Rotr(w[i - 15u], 18u) ^ (w[i - 15u] >> 3u);
            const std::uint32_t s1 = Rotr(w[i - 2u], 17u) ^ Rotr(w[i - 2u], 19u) ^ (w[i - 2u] >> 10u);
            w[i] = w[i - 16u] + s0 + w[i - 7u] + s1;
        }

        std::uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        std::uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (std::size_t i = 0; i < 64u; ++i) {
            const std::uint32_t s1 = Rotr(e, 6u) ^ Rotr(e, 11u) ^ Rotr(e, 25u);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + s1 + ch + kRoundConstants[i] + w[i];
            const std::uint32_t s0 = Rotr(a, 2u) ^ Rotr(a, 13u) ^ Rotr(a, 22u);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + t1;
            d = c;
            c = b;
            b = a;
            a = t1 + t2;
        }
        h_[0] += a;
        h_[1] += b;
        h_[2] += c;
        h_[3] += d;
        h_[4] += e;
        h_[5] += f;
        h_[6] += g;
        h_[7] += h;
    }

    std::array<std::uint32_t, 8> h_;
    std::array<std::uint8_t, 64> buf_{};
    std::size_t buf_len_ = 0;
    std::uint64_t total_bytes_ = 0;
};

[[nodiscard]] bool DigestEqualsConstantTime(const std::array<std::uint8_t, 32>& a,
                                            const std::array<std::uint8_t, 32>& b) noexcept {
    std::uint8_t diff = 0;
    for (std::size_t i = 0; i < 32u; ++i) {
        diff = static_cast<std::uint8_t>(diff | (a[i] ^ b[i]));
    }
    return diff == 0;
}

void WriteLe32(std::uint8_t* dst, std::uint32_t v) noexcept {
    dst[0] = static_cast<std::uint8_t>(v & 0xFFu);
    dst[1] = static_cast<std::uint8_t>((v >> 8u) & 0xFFu);
    dst[2] = static_cast<std::uint8_t>((v >> 16u) & 0xFFu);
    dst[3] = static_cast<std::uint8_t>((v >> 24u) & 0xFFu);
}

void WriteLe64(std::uint8_t* dst, std::uint64_t v) noexcept {
    for (unsigned i = 0; i < 8u; ++i) {
        dst[i] = static_cast<std::uint8_t>((v >> (8u * i)) & 0xFFu);
    }
}

[[nodiscard]] std::uint32_t ReadLe32(const std::uint8_t* src) noexcept {
    return static_cast<std::uint32_t>(src[0]) |
           (static_cast<std::uint32_t>(src[1]) << 8u) |
           (static_cast<std::uint32_t>(src[2]) << 16u) |
           (static_cast<std::uint32_t>(src[3]) << 24u);
}

[[nodiscard]] std::uint64_t ReadLe64(const std::uint8_t* src) noexcept {
    std::uint64_t out = 0;
    for (unsigned i = 0; i < 8u; ++i) {
        out |= static_cast<std::uint64_t>(src[i]) << (8u * i);
    }
    return out;
}

[[nodiscard]] bool EncodeKvCacheTypeWire(KvCacheType type, std::uint32_t& out_wire) noexcept {
    switch (type) {
        case KvCacheType::kF16: out_wire = 1u; return true;
        case KvCacheType::kF32: out_wire = 2u; return true;
        case KvCacheType::kBF16: out_wire = 3u; return true;
        case KvCacheType::kQ8_0: out_wire = 4u; return true;
        case KvCacheType::kQ5_1: out_wire = 5u; return true;
        case KvCacheType::kQ5_0: out_wire = 6u; return true;
        case KvCacheType::kQ4_1: out_wire = 7u; return true;
        case KvCacheType::kQ4_0: out_wire = 8u; return true;
        case KvCacheType::kIQ4_NL: out_wire = 9u; return true;
        case KvCacheType::kF8_E4M3: out_wire = 10u; return true;
        case KvCacheType::kQ2_0: out_wire = 11u; return true;
        case KvCacheType::kPQ2_0: out_wire = 12u; return true;
    }
    return false;
}

[[nodiscard]] bool DecodeKvCacheTypeWire(std::uint32_t wire, KvCacheType& out_type) noexcept {
    switch (wire) {
        case 1u: out_type = KvCacheType::kF16; return true;
        case 2u: out_type = KvCacheType::kF32; return true;
        case 3u: out_type = KvCacheType::kBF16; return true;
        case 4u: out_type = KvCacheType::kQ8_0; return true;
        case 5u: out_type = KvCacheType::kQ5_1; return true;
        case 6u: out_type = KvCacheType::kQ5_0; return true;
        case 7u: out_type = KvCacheType::kQ4_1; return true;
        case 8u: out_type = KvCacheType::kQ4_0; return true;
        case 9u: out_type = KvCacheType::kIQ4_NL; return true;
        case 10u: out_type = KvCacheType::kF8_E4M3; return true;
        case 11u: out_type = KvCacheType::kQ2_0; return true;
        case 12u: out_type = KvCacheType::kPQ2_0; return true;
        default: return false;
    }
}

[[nodiscard]] SlotStoreStatus ValidateLimits(const SlotStoreLimits& limits,
                                             std::uint64_t& out_effective_max_payload,
                                             std::uint64_t& out_effective_max_file) {
    if (limits.max_payload_bytes == 0u || limits.max_payload_bytes > kHardMaxSlotPayloadBytes) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "max_payload_bytes must be in 1..kHardMaxSlotPayloadBytes");
    }
    if (limits.max_file_bytes == 0u || limits.max_file_bytes > kHardMaxSlotFileBytes) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "max_file_bytes must be in 1..kHardMaxSlotFileBytes");
    }
    if (limits.max_file_bytes <= kSlotFileHeaderBytes) {
        out_effective_max_payload = 0u;
        out_effective_max_file = limits.max_file_bytes;
        return SlotStoreStatus::Ok();
    }
    out_effective_max_payload = std::min<std::uint64_t>(
        limits.max_payload_bytes, limits.max_file_bytes - static_cast<std::size_t>(kSlotFileHeaderBytes));
    out_effective_max_file = static_cast<std::uint64_t>(kSlotFileHeaderBytes) + out_effective_max_payload;
    return SlotStoreStatus::Ok();
}

[[nodiscard]] bool HasEmbeddedNul(const fs::path& p) noexcept {
    const auto& native = p.native();
    return native.find(static_cast<fs::path::value_type>(0)) != fs::path::string_type::npos;
}

// Strict RFC 3629 / Unicode Table 3-7 UTF-8 validation (rejects NUL-free invalid
// continuations, overlong encodings, UTF-16 surrogates U+D800..U+DFFF, and > U+10FFFF).
[[nodiscard]] bool IsValidUtf8(std::string_view s) noexcept {
    const auto* b = reinterpret_cast<const unsigned char*>(s.data());
    const std::size_t n = s.size();
    std::size_t i = 0;
    while (i < n) {
        const unsigned char c0 = b[i];
        if (c0 <= 0x7Fu) {
            ++i;
            continue;
        }
        if (c0 >= 0xC2u && c0 <= 0xDFu) {
            if (i + 1u >= n) return false;
            const unsigned char c1 = b[i + 1u];
            if (c1 < 0x80u || c1 > 0xBFu) return false;
            i += 2u;
            continue;
        }
        if (c0 == 0xE0u) {
            if (i + 2u >= n) return false;
            const unsigned char c1 = b[i + 1u];
            const unsigned char c2 = b[i + 2u];
            if (c1 < 0xA0u || c1 > 0xBFu || c2 < 0x80u || c2 > 0xBFu) return false;
            i += 3u;
            continue;
        }
        if ((c0 >= 0xE1u && c0 <= 0xECu) || (c0 >= 0xEEu && c0 <= 0xEFu)) {
            if (i + 2u >= n) return false;
            const unsigned char c1 = b[i + 1u];
            const unsigned char c2 = b[i + 2u];
            if (c1 < 0x80u || c1 > 0xBFu || c2 < 0x80u || c2 > 0xBFu) return false;
            i += 3u;
            continue;
        }
        if (c0 == 0xEDu) {
            if (i + 2u >= n) return false;
            const unsigned char c1 = b[i + 1u];
            const unsigned char c2 = b[i + 2u];
            // 0xA0..0xBF encode UTF-16 surrogates U+D800..U+DFFF (ill-formed UTF-8).
            if (c1 < 0x80u || c1 > 0x9Fu || c2 < 0x80u || c2 > 0xBFu) return false;
            i += 3u;
            continue;
        }
        if (c0 == 0xF0u) {
            if (i + 3u >= n) return false;
            const unsigned char c1 = b[i + 1u];
            const unsigned char c2 = b[i + 2u];
            const unsigned char c3 = b[i + 3u];
            if (c1 < 0x90u || c1 > 0xBFu || c2 < 0x80u || c2 > 0xBFu ||
                c3 < 0x80u || c3 > 0xBFu) {
                return false;
            }
            i += 4u;
            continue;
        }
        if (c0 >= 0xF1u && c0 <= 0xF3u) {
            if (i + 3u >= n) return false;
            const unsigned char c1 = b[i + 1u];
            const unsigned char c2 = b[i + 2u];
            const unsigned char c3 = b[i + 3u];
            if (c1 < 0x80u || c1 > 0xBFu || c2 < 0x80u || c2 > 0xBFu ||
                c3 < 0x80u || c3 > 0xBFu) {
                return false;
            }
            i += 4u;
            continue;
        }
        if (c0 == 0xF4u) {
            if (i + 3u >= n) return false;
            const unsigned char c1 = b[i + 1u];
            const unsigned char c2 = b[i + 2u];
            const unsigned char c3 = b[i + 3u];
            if (c1 < 0x80u || c1 > 0x8Fu || c2 < 0x80u || c2 > 0xBFu ||
                c3 < 0x80u || c3 > 0xBFu) {
                return false;
            }
            i += 4u;
            continue;
        }
        return false;
    }
    return true;
}

[[nodiscard]] constexpr char AsciiLower(char c) noexcept {
    return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

// Rejects Win32 DOS device stems (CON, PRN, AUX, NUL, COM0..COM9, LPT0..LPT9,
// plus superscript ¹²³ variants), case-insensitively and before any '.' extension
// or Win32-stripped trailing spaces before '.'.
[[nodiscard]] bool IsWindowsReservedDeviceName(std::string_view name) noexcept {
    const std::size_t dot = name.find('.');
    std::string_view stem = (dot == std::string_view::npos) ? name : name.substr(0, dot);
    while (!stem.empty() && stem.back() == ' ') {
        stem.remove_suffix(1);
    }
    if (stem.size() == 3u) {
        const char c0 = AsciiLower(stem[0]);
        const char c1 = AsciiLower(stem[1]);
        const char c2 = AsciiLower(stem[2]);
        return (c0 == 'c' && c1 == 'o' && c2 == 'n') ||
               (c0 == 'p' && c1 == 'r' && c2 == 'n') ||
               (c0 == 'a' && c1 == 'u' && c2 == 'x') ||
               (c0 == 'n' && c1 == 'u' && c2 == 'l');
    }
    if (stem.size() == 4u || stem.size() == 5u) {
        const char c0 = AsciiLower(stem[0]);
        const char c1 = AsciiLower(stem[1]);
        const char c2 = AsciiLower(stem[2]);
        const bool is_com_or_lpt =
            (c0 == 'c' && c1 == 'o' && c2 == 'm') ||
            (c0 == 'l' && c1 == 'p' && c2 == 't');
        if (!is_com_or_lpt) return false;
        if (stem.size() == 4u) {
            return stem[3] >= '0' && stem[3] <= '9';
        }
        const auto hi = static_cast<unsigned char>(stem[3]);
        const auto lo = static_cast<unsigned char>(stem[4]);
        // UTF-8 U+00B2 ('²' = C2 B2), U+00B3 ('³' = C2 B3), U+00B9 ('¹' = C2 B9).
        return hi == 0xC2u && (lo == 0xB2u || lo == 0xB3u || lo == 0xB9u);
    }
    return false;
}

[[nodiscard]] bool IsValidFilenameComponentUtf8(std::string_view comp) noexcept {
    if (comp.empty() || comp.size() > 255u || comp == "." || comp == "..") {
        return false;
    }
    if (!IsValidUtf8(comp)) {
        return false;
    }
    if (comp.back() == '.' || comp.back() == ' ') {
        return false;
    }
    for (const char raw : comp) {
        const auto ch = static_cast<unsigned char>(raw);
        if (ch < 0x20u || ch == 0x7Fu) {
            return false;
        }
        if (ch == '<' || ch == '>' || ch == ':' || ch == '"' ||
            ch == '/' || ch == '\\' || ch == '|' || ch == '?' || ch == '*') {
            return false;
        }
    }
    if (IsWindowsReservedDeviceName(comp)) {
        return false;
    }
    return true;
}

[[nodiscard]] bool PathToUtf8Safe(const fs::path& p, std::string& out_utf8) noexcept {
    out_utf8.clear();
    if (HasEmbeddedNul(p)) {
        return false;
    }
    try {
        const std::u8string u8 = p.u8string();
        out_utf8.assign(reinterpret_cast<const char*>(u8.data()), u8.size());
    } catch (...) {
        out_utf8.clear();
        return false;
    }
    return IsValidUtf8(out_utf8);
}

[[nodiscard]] bool ConstructPathFromValidUtf8(std::string_view utf8, fs::path& out_path) noexcept {
    out_path.clear();
    if (utf8.empty() || utf8.find('\0') != std::string_view::npos || !IsValidUtf8(utf8)) {
        return false;
    }
    try {
        const std::u8string u8(utf8.begin(), utf8.end());
        out_path = fs::path(u8);
        return true;
    } catch (...) {
        out_path.clear();
        return false;
    }
}

[[nodiscard]] bool HasInvalidDirectorySegments(const fs::path& p) noexcept {
    std::string utf8_comp;
    try {
        for (const fs::path& comp : p.relative_path()) {
            if (!PathToUtf8Safe(comp, utf8_comp)) {
                return true;
            }
            if (utf8_comp.empty() || utf8_comp == ".") {
                continue;
            }
            if (utf8_comp == "..") {
                return true;
            }
            if (!IsValidFilenameComponentUtf8(utf8_comp)) {
                return true;
            }
        }
    } catch (...) {
        return true;
    }
    return false;
}

#if defined(_WIN32)
[[nodiscard]] bool IsWin32ReparseOrDirectory(const fs::path& p, bool& out_exists) noexcept {
    const DWORD attrs = ::GetFileAttributesW(p.c_str());
    if (attrs == INVALID_FILE_ATTRIBUTES) {
        out_exists = false;
        return false;
    }
    out_exists = true;
    return (attrs & (FILE_ATTRIBUTE_REPARSE_POINT | FILE_ATTRIBUTE_DIRECTORY)) != 0u;
}
#endif

struct BoundedGgufReaderState {
    std::ifstream* stream = nullptr;
    std::uint64_t max_bytes = 0;
};

std::size_t BoundedGgufReadCallback(void* userdata, void* output, std::uint64_t offset, std::size_t len) {
    if (userdata == nullptr || output == nullptr || len == 0u) {
        return 0u;
    }
    auto& state = *static_cast<BoundedGgufReaderState*>(userdata);
    if (state.stream == nullptr || offset > state.max_bytes || len > state.max_bytes - offset) {
        return 0u;
    }
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max()) ||
        len > static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        return 0u;
    }
    state.stream->clear();
    state.stream->seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!state.stream->good()) {
        return 0u;
    }
    state.stream->read(static_cast<char*>(output), static_cast<std::streamsize>(len));
    const std::streamsize got = state.stream->gcount();
    return got > 0 ? static_cast<std::size_t>(got) : 0u;
}

[[nodiscard]] SlotStoreStatus WriteSlotAtomically(
    const fs::path& out_path,
    const std::array<std::uint8_t, kSlotFileHeaderBytes>& header_wire,
    std::span<const std::uint8_t> token_table_wire,
    std::span<const std::uint8_t> state_bytes) {
    if (SlotStoreStatus pv = ValidateSlotFilePath(out_path); !pv.ok()) {
        return pv;
    }
    fs::path parent = out_path.parent_path();
    if (parent.empty()) {
        parent = fs::path(".");
    }
    std::error_code ec;
    if (!fs::exists(parent, ec) || ec || !fs::is_directory(parent, ec)) {
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "slot parent directory does not exist or is not a directory");
    }

    std::random_device rd;
    fs::path tmp_path;

#if defined(_WIN32)
    HANDLE h = INVALID_HANDLE_VALUE;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const std::uint64_t nonce =
            (static_cast<std::uint64_t>(rd()) << 32u) ^ static_cast<std::uint64_t>(rd());
        char name[48];
        std::snprintf(name, sizeof(name), ".sonder-slot-%016llx.tmp",
                      static_cast<unsigned long long>(nonce));
        tmp_path = parent / name;
        h = ::CreateFileW(tmp_path.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h != INVALID_HANDLE_VALUE) {
            break;
        }
        if (::GetLastError() != ERROR_FILE_EXISTS) {
            return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                          "failed to create temporary slot file");
        }
    }
    if (h == INVALID_HANDLE_VALUE) {
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "exhausted temporary slot filename attempts");
    }

    auto cleanup_tmp = [&]() noexcept {
        std::error_code rm_ec;
        fs::remove(tmp_path, rm_ec);
    };

    auto write_span = [&](std::span<const std::uint8_t> span) noexcept -> bool {
        std::size_t off = 0;
        while (off < span.size()) {
            const DWORD chunk =
                static_cast<DWORD>(std::min<std::size_t>(span.size() - off, 0x40000000u));
            DWORD wrote = 0;
            if (!::WriteFile(h, span.data() + off, chunk, &wrote, nullptr) || wrote == 0) {
                return false;
            }
            off += static_cast<std::size_t>(wrote);
        }
        return true;
    };

    bool write_ok = write_span(header_wire) && write_span(token_table_wire) && write_span(state_bytes);
    if (write_ok) {
        write_ok = (::FlushFileBuffers(h) != 0);
    }
    ::CloseHandle(h);
    if (!write_ok) {
        cleanup_tmp();
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "failed writing or flushing temporary slot file");
    }

    if (SlotStoreStatus recheck = ValidateSlotFilePath(out_path); !recheck.ok()) {
        cleanup_tmp();
        return recheck;
    }
    if (!::MoveFileExW(tmp_path.c_str(), out_path.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        cleanup_tmp();
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "atomic rename of temporary slot file failed");
    }
#else
    int fd = -1;
    for (int attempt = 0; attempt < 32; ++attempt) {
        const std::uint64_t nonce =
            (static_cast<std::uint64_t>(rd()) << 32u) ^ static_cast<std::uint64_t>(rd());
        char name[48];
        std::snprintf(name, sizeof(name), ".sonder-slot-%016llx.tmp",
                      static_cast<unsigned long long>(nonce));
        tmp_path = parent / name;
        int flags = O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC;
#ifdef O_NOFOLLOW
        flags |= O_NOFOLLOW;
#endif
        fd = ::open(tmp_path.c_str(), flags, 0600);
        if (fd >= 0) {
            break;
        }
        if (errno != EEXIST) {
            return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                          "failed to create temporary slot file");
        }
    }
    if (fd < 0) {
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "exhausted temporary slot filename attempts");
    }

    auto cleanup_tmp = [&]() noexcept {
        std::error_code rm_ec;
        fs::remove(tmp_path, rm_ec);
    };

    auto write_span = [&](std::span<const std::uint8_t> span) noexcept -> bool {
        std::size_t off = 0;
        while (off < span.size()) {
            const std::size_t chunk = std::min<std::size_t>(span.size() - off, 0x40000000u);
            const ssize_t wrote = ::write(fd, span.data() + off, chunk);
            if (wrote < 0) {
                if (errno == EINTR) continue;
                return false;
            }
            if (wrote == 0) {
                return false;
            }
            off += static_cast<std::size_t>(wrote);
        }
        return true;
    };

    bool write_ok = write_span(header_wire) && write_span(token_table_wire) && write_span(state_bytes);
    if (write_ok && ::fsync(fd) != 0 && errno != EINVAL && errno != ENOTSUP) {
        write_ok = false;
    }
    if (::close(fd) != 0) {
        write_ok = false;
    }
    if (!write_ok) {
        cleanup_tmp();
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "failed writing or syncing temporary slot file");
    }

    if (SlotStoreStatus recheck = ValidateSlotFilePath(out_path); !recheck.ok()) {
        cleanup_tmp();
        return recheck;
    }
    if (::rename(tmp_path.c_str(), out_path.c_str()) != 0) {
        cleanup_tmp();
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "atomic rename of temporary slot file failed");
    }
#endif

    return SlotStoreStatus::Ok();
}

}  // namespace

std::string_view ToString(SlotStoreError code) noexcept {
    switch (code) {
        case SlotStoreError::kOk: return "ok";
        case SlotStoreError::kInvalidArgument: return "invalid_argument";
        case SlotStoreError::kInvalidPath: return "invalid_path";
        case SlotStoreError::kIoError: return "io_error";
        case SlotStoreError::kInvalidGguf: return "invalid_gguf";
        case SlotStoreError::kTruncatedFile: return "truncated_file";
        case SlotStoreError::kFileTooLarge: return "file_too_large";
        case SlotStoreError::kBadMagic: return "bad_magic";
        case SlotStoreError::kUnsupportedVersion: return "unsupported_version";
        case SlotStoreError::kInvalidHeader: return "invalid_header";
        case SlotStoreError::kModelMismatch: return "model_mismatch";
        case SlotStoreError::kKvTypeMismatch: return "kv_type_mismatch";
        case SlotStoreError::kContextLengthMismatch: return "context_length_mismatch";
        case SlotStoreError::kTokenCountMismatch: return "token_count_mismatch";
        case SlotStoreError::kChecksumMismatch: return "checksum_mismatch";
        case SlotStoreError::kBackendError: return "backend_error";
    }
    return "unknown";
}

Status SlotStoreStatus::ToBackendStatus() const {
    if (ok()) {
        return Status::Ok();
    }
    switch (code) {
        case SlotStoreError::kOk:
            return Status::Ok();
        case SlotStoreError::kInvalidArgument:
        case SlotStoreError::kInvalidPath:
        case SlotStoreError::kInvalidGguf:
        case SlotStoreError::kTruncatedFile:
        case SlotStoreError::kFileTooLarge:
        case SlotStoreError::kBadMagic:
        case SlotStoreError::kUnsupportedVersion:
        case SlotStoreError::kInvalidHeader:
        case SlotStoreError::kModelMismatch:
        case SlotStoreError::kKvTypeMismatch:
        case SlotStoreError::kContextLengthMismatch:
        case SlotStoreError::kTokenCountMismatch:
        case SlotStoreError::kChecksumMismatch:
            return Status::Error(ErrorCode::kInvalidArgument,
                                 std::string(ToString(code)) + ": " + message);
        case SlotStoreError::kIoError:
        case SlotStoreError::kBackendError:
            return Status::Error(ErrorCode::kDecodeFailed,
                                 std::string(ToString(code)) + ": " + message);
    }
    return Status::Error(ErrorCode::kDecodeFailed, message);
}

SlotStoreStatus LlamaCppSlotStateBackend::GetSequenceStateSize(std::int32_t seq_id,
                                                               std::size_t& out_size) const {
    out_size = 0;
    if (ctx_ == nullptr) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "llama_context is null");
    }
    if (seq_id < 0) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "seq_id must be >= 0");
    }
    std::size_t sz = 0;
    try {
        // llama.cpp v1.0.0: llama_state_seq_get_size (include/llama.h:856-858).
        sz = llama_state_seq_get_size(ctx_, static_cast<llama_seq_id>(seq_id));
    } catch (...) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "llama_state_seq_get_size threw an exception");
    }
    if (sz == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "llama_state_seq_get_size returned 0");
    }
    out_size = sz;
    return SlotStoreStatus::Ok();
}

SlotStoreStatus LlamaCppSlotStateBackend::GetSequenceStateData(std::int32_t seq_id,
                                                               std::span<std::uint8_t> dst,
                                                               std::size_t& out_written) const {
    out_written = 0;
    if (ctx_ == nullptr) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "llama_context is null");
    }
    if (seq_id < 0) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "seq_id must be >= 0");
    }
    if (dst.empty()) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "destination state buffer is empty");
    }
    std::size_t written = 0;
    try {
        // llama.cpp v1.0.0: llama_state_seq_get_data (include/llama.h:861-865).
        written = llama_state_seq_get_data(ctx_, dst.data(), dst.size(),
                                           static_cast<llama_seq_id>(seq_id));
    } catch (...) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "llama_state_seq_get_data threw an exception");
    }
    if (written == 0u || written > dst.size()) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "llama_state_seq_get_data failed");
    }
    out_written = written;
    return SlotStoreStatus::Ok();
}

SlotStoreStatus LlamaCppSlotStateBackend::SetSequenceStateData(
    std::int32_t seq_id, std::span<const std::uint8_t> src) {
    if (ctx_ == nullptr) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "llama_context is null");
    }
    if (seq_id < 0) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "seq_id must be >= 0");
    }
    if (src.empty()) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "source state buffer is empty");
    }
    std::size_t read_bytes = 0;
    try {
        // llama.cpp v1.0.0: llama_state_seq_set_data (include/llama.h:871-875).
        read_bytes = llama_state_seq_set_data(ctx_, src.data(), src.size(),
                                              static_cast<llama_seq_id>(seq_id));
    } catch (...) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "llama_state_seq_set_data threw an exception");
    }
    if (read_bytes == 0u || read_bytes != src.size()) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "llama_state_seq_set_data failed");
    }
    return SlotStoreStatus::Ok();
}

SlotStoreStatus ValidateSlotFilePath(const fs::path& path) {
    if (path.empty()) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath, "slot path is empty");
    }
    if (HasEmbeddedNul(path)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot path contains an embedded NUL");
    }
    // Reject Windows drive-relative paths such as "C:slot.bin".
    if (path.has_root_name() && !path.has_root_directory()) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "drive-relative slot paths are not allowed");
    }
    std::string full_utf8;
    if (!PathToUtf8Safe(path, full_utf8)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot path contains ill-formed UTF-8/UTF-16 sequences");
    }
    if (full_utf8.back() == '/' || full_utf8.back() == '\\') {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot path has a trailing directory separator");
    }
    if (HasInvalidDirectorySegments(path)) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidPath,
            "slot path contains '..', reserved device name, control char, or invalid component");
    }

    std::string fname_utf8;
    if (!PathToUtf8Safe(path.filename(), fname_utf8) || !IsValidFilenameComponentUtf8(fname_utf8)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot path does not name a valid regular file");
    }

#if defined(_WIN32)
    bool exists_win = false;
    if (IsWin32ReparseOrDirectory(path, exists_win)) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidPath,
            "slot path refers to a directory or symlink/reparse point");
    }
#endif

    std::error_code ec;
    const fs::file_status st = fs::symlink_status(path, ec);
    if (!ec && fs::status_known(st) && st.type() != fs::file_type::not_found) {
        if (fs::is_symlink(st) || fs::is_directory(st) || !fs::is_regular_file(st)) {
            return SlotStoreStatus::Error(
                SlotStoreError::kInvalidPath,
                "slot path exists and is not a regular non-symlink file");
        }
    }
    return SlotStoreStatus::Ok();
}

SlotStoreStatus ValidateSlotFilePathUtf8(std::string_view utf8_path, fs::path& out_path) {
    out_path.clear();
    fs::path p;
    if (!ConstructPathFromValidUtf8(utf8_path, p)) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidPath,
            "slot path is empty, contains NUL, or is not valid UTF-8");
    }
    if (SlotStoreStatus s = ValidateSlotFilePath(p); !s.ok()) {
        return s;
    }
    out_path = std::move(p);
    return SlotStoreStatus::Ok();
}

SlotStoreStatus ResolveSlotFilePath(const fs::path& root_dir, std::string_view slot_filename,
                                    fs::path& out_path) {
    out_path.clear();
    std::string root_utf8;
    if (root_dir.empty() || !PathToUtf8Safe(root_dir, root_utf8)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath, "invalid slot root directory");
    }
    if (root_dir.has_root_name() && !root_dir.has_root_directory()) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "drive-relative slot root directory is not allowed");
    }
    if (HasInvalidDirectorySegments(root_dir)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot root directory contains '..' or invalid segment");
    }
    if (!IsValidFilenameComponentUtf8(slot_filename)) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidPath,
            "slot filename is invalid, malformed UTF-8, or a reserved device name");
    }

    fs::path leaf_path;
    if (!ConstructPathFromValidUtf8(slot_filename, leaf_path)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "failed to decode UTF-8 slot filename");
    }

#if defined(_WIN32)
    const DWORD root_attrs = ::GetFileAttributesW(root_dir.c_str());
    if (root_attrs == INVALID_FILE_ATTRIBUTES ||
        (root_attrs & FILE_ATTRIBUTE_DIRECTORY) == 0u ||
        (root_attrs & FILE_ATTRIBUTE_REPARSE_POINT) != 0u) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidPath,
            "slot root directory must be an existing non-reparse directory");
    }
#endif

    std::error_code ec;
    const fs::file_status root_st = fs::symlink_status(root_dir, ec);
    if (ec || !fs::is_directory(root_st) || fs::is_symlink(root_st)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot root directory must be an existing non-symlink directory");
    }
    const fs::path canon_root = fs::weakly_canonical(root_dir, ec);
    if (ec || canon_root.empty()) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "failed to canonicalize slot root directory");
    }

    fs::path candidate;
    try {
        candidate = canon_root / leaf_path;
    } catch (...) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "failed to join slot filename with root directory");
    }
    if (candidate.parent_path() != canon_root) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot filename escaped root directory");
    }
    if (SlotStoreStatus s = ValidateSlotFilePath(candidate); !s.ok()) {
        return s;
    }
    out_path = std::move(candidate);
    return SlotStoreStatus::Ok();
}

SlotStoreStatus ComputeGgufModelIdentity(const fs::path& gguf_path, SlotModelIdentity& out_identity,
                                         std::size_t max_header_bytes,
                                         std::uint64_t* out_hashed_header_bytes) {
    out_identity = {};
    if (out_hashed_header_bytes != nullptr) {
        *out_hashed_header_bytes = 0u;
    }
    if (max_header_bytes == 0u || max_header_bytes > (std::size_t{512} << 20)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "max_header_bytes must be in 1..512 MiB");
    }
    if (SlotStoreStatus pv = ValidateSlotFilePath(gguf_path); !pv.ok()) {
        return pv;
    }

    std::error_code ec;
    if (!fs::is_regular_file(gguf_path, ec) || ec) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidGguf,
                                      "GGUF path does not exist or is not a regular file");
    }

    std::ifstream in(gguf_path, std::ios::binary);
    if (!in.good()) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidGguf, "cannot open GGUF file");
    }
    in.seekg(0, std::ios::end);
    const std::streamoff end_pos = in.tellg();
    if (!in.good() || end_pos < 24) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidGguf,
                                      "GGUF file is smaller than minimal 24-byte header");
    }
    const auto file_size = static_cast<std::uint64_t>(end_pos);
    const std::uint64_t header_cap =
        std::min<std::uint64_t>(file_size, static_cast<std::uint64_t>(max_header_bytes));
    if (header_cap < 24u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidGguf,
                                      "max_header_bytes is smaller than minimal GGUF header");
    }

    BoundedGgufReaderState reader_state{&in, header_cap};
    const gguf_init_params params{/*no_alloc=*/true, /*ctx=*/nullptr};
    std::unique_ptr<gguf_context, decltype(&gguf_free)> ctx(
        gguf_init_from_callback(&BoundedGgufReadCallback, &reader_state, kIoChunkBytes, header_cap,
                                params),
        &gguf_free);
    if (!ctx) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidGguf,
            "failed to parse GGUF header/metadata within configured byte cap");
    }

    const std::size_t data_offset = gguf_get_data_offset(ctx.get());
    if (data_offset < 24u || static_cast<std::uint64_t>(data_offset) > header_cap ||
        static_cast<std::uint64_t>(data_offset) > file_size) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidGguf,
                                      "GGUF data offset is out of bounds");
    }

    in.clear();
    in.seekg(0, std::ios::beg);
    if (!in.good()) {
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "failed to rewind GGUF stream for header hashing");
    }

    Sha256 sha;
    std::array<std::uint8_t, kIoChunkBytes> buf{};
    std::size_t remaining = data_offset;
    while (remaining > 0u) {
        const std::size_t chunk = std::min<std::size_t>(remaining, buf.size());
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(chunk));
        const std::streamsize got = in.gcount();
        if (got <= 0 || static_cast<std::size_t>(got) != chunk) {
            return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                          "short read while hashing GGUF header region");
        }
        sha.Update(std::span<const std::uint8_t>(buf.data(), chunk));
        remaining -= chunk;
    }

    out_identity.file_size = file_size;
    out_identity.header_sha256 = sha.Finish();
    if (out_hashed_header_bytes != nullptr) {
        *out_hashed_header_bytes = static_cast<std::uint64_t>(data_offset);
    }
    return SlotStoreStatus::Ok();
}

SlotStoreStatus SerializeSlotHeader(const SlotFileHeader& header,
                                    std::array<std::uint8_t, kSlotFileHeaderBytes>& out_bytes,
                                    const SlotStoreLimits& limits) {
    out_bytes.fill(0u);
    std::uint64_t max_payload = 0;
    std::uint64_t max_file = 0;
    if (SlotStoreStatus lv = ValidateLimits(limits, max_payload, max_file); !lv.ok()) {
        return lv;
    }
    (void)max_file;

    if (header.magic != kSlotFileMagic) {
        return SlotStoreStatus::Error(SlotStoreError::kBadMagic, "invalid slot header magic");
    }
    if (header.format_version != kSlotFormatVersion) {
        return SlotStoreStatus::Error(SlotStoreError::kUnsupportedVersion,
                                      "unsupported slot header format version");
    }
    if (header.header_bytes != kSlotFileHeaderBytes || header.reserved != 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "invalid slot header size or non-zero reserved field");
    }
    if (header.model_identity.file_size == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "model_identity.file_size must be > 0");
    }
    std::uint32_t wire_k = 0;
    std::uint32_t wire_v = 0;
    if (!EncodeKvCacheTypeWire(header.type_k, wire_k) ||
        !EncodeKvCacheTypeWire(header.type_v, wire_v)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "invalid KvCacheType enum value");
    }
    if (header.n_ctx == 0u || header.n_ctx > kMaxSlotContextTokens) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "n_ctx must be in 1..kMaxSlotContextTokens");
    }
    if (header.token_count > static_cast<std::uint64_t>(header.n_ctx)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "token_count exceeds n_ctx");
    }
    const std::uint64_t exact_tok_bytes = header.token_count * sizeof(Token);
    if (header.token_table_bytes != 0u && header.token_table_bytes != exact_tok_bytes) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidHeader,
            "token_table_bytes must be 0 or token_count * sizeof(Token)");
    }
    if (header.state_bytes == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader, "state_bytes must be > 0");
    }
    if (header.state_bytes > max_payload ||
        header.token_table_bytes > max_payload - header.state_bytes) {
        return SlotStoreStatus::Error(SlotStoreError::kFileTooLarge,
                                      "declared slot payload exceeds configured limit");
    }

    std::memcpy(out_bytes.data(), header.magic.data(), header.magic.size());
    WriteLe32(out_bytes.data() + 8u, header.format_version);
    WriteLe32(out_bytes.data() + 12u, header.header_bytes);
    WriteLe64(out_bytes.data() + 16u, header.model_identity.file_size);
    std::memcpy(out_bytes.data() + 24u, header.model_identity.header_sha256.data(), 32u);
    WriteLe32(out_bytes.data() + 56u, wire_k);
    WriteLe32(out_bytes.data() + 60u, wire_v);
    WriteLe32(out_bytes.data() + 64u, header.n_ctx);
    WriteLe32(out_bytes.data() + 68u, header.reserved);
    WriteLe64(out_bytes.data() + 72u, header.token_count);
    WriteLe64(out_bytes.data() + 80u, header.token_table_bytes);
    WriteLe64(out_bytes.data() + 88u, header.state_bytes);
    std::memcpy(out_bytes.data() + 96u, header.payload_sha256.data(), 32u);
    return SlotStoreStatus::Ok();
}

SlotStoreStatus DeserializeSlotHeader(std::span<const std::uint8_t> bytes,
                                      SlotFileHeader& out_header,
                                      const SlotStoreLimits& limits) {
    out_header = {};
    std::uint64_t max_payload = 0;
    std::uint64_t max_file = 0;
    if (SlotStoreStatus lv = ValidateLimits(limits, max_payload, max_file); !lv.ok()) {
        return lv;
    }
    (void)max_file;

    if (bytes.size() < kSlotFileHeaderBytes) {
        return SlotStoreStatus::Error(SlotStoreError::kTruncatedFile,
                                      "slot file header is truncated");
    }

    SlotFileHeader hdr{};
    std::memcpy(hdr.magic.data(), bytes.data(), hdr.magic.size());
    if (hdr.magic != kSlotFileMagic) {
        return SlotStoreStatus::Error(SlotStoreError::kBadMagic,
                                      "slot file magic does not match SNDRSLOT");
    }

    hdr.format_version = ReadLe32(bytes.data() + 8u);
    if (hdr.format_version != kSlotFormatVersion) {
        return SlotStoreStatus::Error(
            SlotStoreError::kUnsupportedVersion,
            "unsupported slot format version: " + std::to_string(hdr.format_version));
    }

    hdr.header_bytes = ReadLe32(bytes.data() + 12u);
    if (hdr.header_bytes != kSlotFileHeaderBytes) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "invalid header_bytes length in slot header");
    }

    hdr.model_identity.file_size = ReadLe64(bytes.data() + 16u);
    if (hdr.model_identity.file_size == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "slot header model_file_size is zero");
    }
    std::memcpy(hdr.model_identity.header_sha256.data(), bytes.data() + 24u, 32u);

    const std::uint32_t wire_k = ReadLe32(bytes.data() + 56u);
    const std::uint32_t wire_v = ReadLe32(bytes.data() + 60u);
    if (!DecodeKvCacheTypeWire(wire_k, hdr.type_k) ||
        !DecodeKvCacheTypeWire(wire_v, hdr.type_v)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "unknown KvCacheType wire code in slot header");
    }

    hdr.n_ctx = ReadLe32(bytes.data() + 64u);
    if (hdr.n_ctx == 0u || hdr.n_ctx > kMaxSlotContextTokens) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "slot header n_ctx is out of bounds");
    }

    hdr.reserved = ReadLe32(bytes.data() + 68u);
    if (hdr.reserved != 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "slot header reserved field must be zero");
    }

    hdr.token_count = ReadLe64(bytes.data() + 72u);
    if (hdr.token_count > static_cast<std::uint64_t>(hdr.n_ctx)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "slot header token_count exceeds n_ctx");
    }

    hdr.token_table_bytes = ReadLe64(bytes.data() + 80u);
    const std::uint64_t exact_tok_bytes = hdr.token_count * sizeof(Token);
    if (hdr.token_table_bytes != 0u && hdr.token_table_bytes != exact_tok_bytes) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidHeader,
            "slot header token_table_bytes is inconsistent with token_count");
    }

    hdr.state_bytes = ReadLe64(bytes.data() + 88u);
    if (hdr.state_bytes == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidHeader,
                                      "slot header state_bytes is zero");
    }
    if (hdr.state_bytes > max_payload ||
        hdr.token_table_bytes > max_payload - hdr.state_bytes) {
        return SlotStoreStatus::Error(
            SlotStoreError::kFileTooLarge,
            "slot header declared payload exceeds configured byte bound");
    }

    std::memcpy(hdr.payload_sha256.data(), bytes.data() + 96u, 32u);
    out_header = hdr;
    return SlotStoreStatus::Ok();
}

SlotStoreStatus SaveSlotFile(const fs::path& path, std::int32_t seq_id,
                             const SlotSaveMetadata& meta, const SlotStateBackend& backend,
                             const SlotStoreLimits& limits, SlotFileHeader* out_header) {
    if (out_header != nullptr) {
        *out_header = {};
    }
    std::uint64_t max_payload = 0;
    std::uint64_t max_file = 0;
    if (SlotStoreStatus lv = ValidateLimits(limits, max_payload, max_file); !lv.ok()) {
        return lv;
    }
    (void)max_file;

    if (seq_id < 0) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "seq_id must be >= 0");
    }
    if (SlotStoreStatus pv = ValidateSlotFilePath(path); !pv.ok()) {
        return pv;
    }
    if (meta.model_identity.file_size == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "meta.model_identity.file_size must be > 0");
    }
    std::uint32_t dummy_wire = 0;
    if (!EncodeKvCacheTypeWire(meta.type_k, dummy_wire) ||
        !EncodeKvCacheTypeWire(meta.type_v, dummy_wire)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "invalid KvCacheType in SlotSaveMetadata");
    }
    if (meta.n_ctx == 0u || meta.n_ctx > kMaxSlotContextTokens) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "meta.n_ctx must be in 1..kMaxSlotContextTokens");
    }
    if (!meta.tokens.empty() && meta.token_count != 0u &&
        meta.token_count != meta.tokens.size()) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidArgument,
            "meta.token_count disagrees with meta.tokens.size()");
    }
    const std::uint64_t effective_token_count =
        !meta.tokens.empty() ? static_cast<std::uint64_t>(meta.tokens.size()) : meta.token_count;
    if (effective_token_count > static_cast<std::uint64_t>(meta.n_ctx)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "effective token_count exceeds meta.n_ctx");
    }

    for (const Token tok : meta.tokens) {
        if (tok < 0) {
            return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                          "token IDs in SlotSaveMetadata must be >= 0");
        }
    }

    std::size_t raw_state_size = 0;
    if (SlotStoreStatus sz_st = backend.GetSequenceStateSize(seq_id, raw_state_size); !sz_st.ok()) {
        return sz_st;
    }
    if (raw_state_size == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "sequence state size reported as 0");
    }

    const std::uint64_t token_table_bytes =
        static_cast<std::uint64_t>(meta.tokens.size()) * sizeof(Token);
    if (static_cast<std::uint64_t>(raw_state_size) > max_payload ||
        token_table_bytes > max_payload - static_cast<std::uint64_t>(raw_state_size)) {
        return SlotStoreStatus::Error(SlotStoreError::kFileTooLarge,
                                      "sequence state payload exceeds configured byte cap");
    }

    std::vector<std::uint8_t> token_table_wire;
    std::vector<std::uint8_t> state_buf;
    try {
        token_table_wire.resize(static_cast<std::size_t>(token_table_bytes));
        state_buf.resize(raw_state_size);
    } catch (const std::bad_alloc&) {
        return SlotStoreStatus::Error(SlotStoreError::kFileTooLarge,
                                      "allocation failed while buffering sequence slot state");
    }

    for (std::size_t i = 0; i < meta.tokens.size(); ++i) {
        WriteLe32(token_table_wire.data() + i * sizeof(Token),
                  static_cast<std::uint32_t>(meta.tokens[i]));
    }

    std::size_t written_state = 0;
    if (SlotStoreStatus gd_st = backend.GetSequenceStateData(seq_id, state_buf, written_state);
        !gd_st.ok()) {
        return gd_st;
    }
    if (written_state == 0u || written_state > raw_state_size) {
        return SlotStoreStatus::Error(SlotStoreError::kBackendError,
                                      "GetSequenceStateData returned invalid byte count");
    }
    state_buf.resize(written_state);

    Sha256 sha;
    if (!token_table_wire.empty()) {
        sha.Update(token_table_wire);
    }
    sha.Update(state_buf);

    SlotFileHeader hdr{};
    hdr.magic = kSlotFileMagic;
    hdr.format_version = kSlotFormatVersion;
    hdr.header_bytes = kSlotFileHeaderBytes;
    hdr.model_identity = meta.model_identity;
    hdr.type_k = meta.type_k;
    hdr.type_v = meta.type_v;
    hdr.n_ctx = meta.n_ctx;
    hdr.reserved = 0u;
    hdr.token_count = effective_token_count;
    hdr.token_table_bytes = token_table_bytes;
    hdr.state_bytes = static_cast<std::uint64_t>(written_state);
    hdr.payload_sha256 = sha.Finish();

    std::array<std::uint8_t, kSlotFileHeaderBytes> hdr_wire{};
    if (SlotStoreStatus ser = SerializeSlotHeader(hdr, hdr_wire, limits); !ser.ok()) {
        return ser;
    }

    if (SlotStoreStatus wr = WriteSlotAtomically(path, hdr_wire, token_table_wire, state_buf);
        !wr.ok()) {
        return wr;
    }

    if (out_header != nullptr) {
        *out_header = hdr;
    }
    return SlotStoreStatus::Ok();
}

SlotStoreStatus RestoreSlotFile(const fs::path& path, std::int32_t dest_seq_id,
                                const SlotRestoreExpectation& expected,
                                SlotStateBackend& backend, SlotRestoreResult* out_result,
                                const SlotStoreLimits& limits) {
    if (out_result != nullptr) {
        *out_result = {};
    }
    std::uint64_t max_payload = 0;
    std::uint64_t max_file = 0;
    if (SlotStoreStatus lv = ValidateLimits(limits, max_payload, max_file); !lv.ok()) {
        return lv;
    }
    (void)max_payload;

    if (dest_seq_id < 0) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "dest_seq_id must be >= 0");
    }
    if (expected.model_identity.file_size == 0u) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "expected.model_identity.file_size must be > 0");
    }
    std::uint32_t dummy_wire = 0;
    if (!EncodeKvCacheTypeWire(expected.type_k, dummy_wire) ||
        !EncodeKvCacheTypeWire(expected.type_v, dummy_wire)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "invalid KvCacheType in SlotRestoreExpectation");
    }
    if (expected.n_ctx == 0u || expected.n_ctx > kMaxSlotContextTokens) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "expected.n_ctx must be in 1..kMaxSlotContextTokens");
    }
    if (expected.expected_token_count.has_value() &&
        *expected.expected_token_count > static_cast<std::uint64_t>(expected.n_ctx)) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument,
                                      "expected_token_count exceeds expected.n_ctx");
    }
    if (SlotStoreStatus pv = ValidateSlotFilePath(path); !pv.ok()) {
        return pv;
    }

    std::error_code ec;
    if (!fs::exists(path, ec) || ec) {
        return SlotStoreStatus::Error(SlotStoreError::kIoError, "slot file does not exist");
    }
    if (!fs::is_regular_file(path, ec) || ec) {
        return SlotStoreStatus::Error(SlotStoreError::kInvalidPath,
                                      "slot path is not a regular file");
    }

    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "cannot open slot file for reading");
    }
    in.seekg(0, std::ios::end);
    const std::streamoff end_pos = in.tellg();
    if (!in.good() || end_pos < 0) {
        return SlotStoreStatus::Error(SlotStoreError::kIoError,
                                      "failed to determine slot file size");
    }
    const auto actual_file_size = static_cast<std::uint64_t>(end_pos);
    if (actual_file_size > max_file) {
        return SlotStoreStatus::Error(SlotStoreError::kFileTooLarge,
                                      "slot file size exceeds configured max_file_bytes");
    }
    if (actual_file_size < kSlotFileHeaderBytes) {
        return SlotStoreStatus::Error(SlotStoreError::kTruncatedFile,
                                      "slot file is smaller than the 128-byte header");
    }

    in.seekg(0, std::ios::beg);
    std::array<std::uint8_t, kSlotFileHeaderBytes> hdr_wire{};
    in.read(reinterpret_cast<char*>(hdr_wire.data()),
            static_cast<std::streamsize>(hdr_wire.size()));
    if (in.gcount() != static_cast<std::streamsize>(hdr_wire.size())) {
        return SlotStoreStatus::Error(SlotStoreError::kTruncatedFile,
                                      "short read on slot file header");
    }

    // Validates magic, version, and all length bounds BEFORE any allocation.
    SlotFileHeader hdr{};
    if (SlotStoreStatus dh = DeserializeSlotHeader(hdr_wire, hdr, limits); !dh.ok()) {
        return dh;
    }

    const std::uint64_t declared_payload_bytes = hdr.token_table_bytes + hdr.state_bytes;
    const std::uint64_t declared_total_bytes =
        static_cast<std::uint64_t>(kSlotFileHeaderBytes) + declared_payload_bytes;
    if (declared_total_bytes > max_file) {
        return SlotStoreStatus::Error(SlotStoreError::kFileTooLarge,
                                      "slot declared file size exceeds configured max_file_bytes");
    }
    if (actual_file_size < declared_total_bytes) {
        return SlotStoreStatus::Error(
            SlotStoreError::kTruncatedFile,
            "slot file is truncated relative to declared payload length");
    }
    if (actual_file_size > declared_total_bytes) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidHeader,
            "slot file contains trailing unauthenticated bytes beyond declared payload");
    }

    // Check every compatibility invariant BEFORE allocating or reading payload bytes.
    if (hdr.model_identity.file_size != expected.model_identity.file_size ||
        !DigestEqualsConstantTime(hdr.model_identity.header_sha256,
                                  expected.model_identity.header_sha256)) {
        return SlotStoreStatus::Error(
            SlotStoreError::kModelMismatch,
            "slot model identity (file size or GGUF header SHA-256) does not match loaded model");
    }
    if (hdr.type_k != expected.type_k || hdr.type_v != expected.type_v) {
        return SlotStoreStatus::Error(SlotStoreError::kKvTypeMismatch,
                                      "slot K/V cache dtype does not match context K/V dtype");
    }
    if (hdr.n_ctx != expected.n_ctx) {
        return SlotStoreStatus::Error(SlotStoreError::kContextLengthMismatch,
                                      "slot n_ctx does not match context n_ctx");
    }
    if (expected.expected_token_count.has_value() &&
        hdr.token_count != *expected.expected_token_count) {
        return SlotStoreStatus::Error(SlotStoreError::kTokenCountMismatch,
                                      "slot token_count does not match expected token count");
    }

    std::vector<Token> tokens;
    std::vector<std::uint8_t> state_buf;
    try {
        if (hdr.token_table_bytes > 0u) {
            tokens.resize(static_cast<std::size_t>(hdr.token_count));
        }
        state_buf.resize(static_cast<std::size_t>(hdr.state_bytes));
    } catch (const std::bad_alloc&) {
        return SlotStoreStatus::Error(SlotStoreError::kFileTooLarge,
                                      "allocation failed while reading slot payload");
    }

    Sha256 sha;
    if (hdr.token_table_bytes > 0u) {
        std::array<std::uint8_t, kIoChunkBytes> tok_chunk{};
        std::size_t tok_idx = 0;
        std::size_t rem = static_cast<std::size_t>(hdr.token_table_bytes);
        while (rem > 0u) {
            // Keep chunk aligned to sizeof(Token) (4 bytes).
            const std::size_t take = std::min<std::size_t>(rem, tok_chunk.size());
            in.read(reinterpret_cast<char*>(tok_chunk.data()),
                    static_cast<std::streamsize>(take));
            if (in.gcount() != static_cast<std::streamsize>(take)) {
                return SlotStoreStatus::Error(SlotStoreError::kTruncatedFile,
                                              "short read on slot token table");
            }
            sha.Update(std::span<const std::uint8_t>(tok_chunk.data(), take));
            const std::size_t count_in_chunk = take / sizeof(Token);
            for (std::size_t i = 0; i < count_in_chunk; ++i) {
                tokens[tok_idx++] =
                    static_cast<Token>(ReadLe32(tok_chunk.data() + i * sizeof(Token)));
            }
            rem -= take;
        }
    }

    std::size_t off = 0;
    while (off < state_buf.size()) {
        const std::size_t take = std::min<std::size_t>(state_buf.size() - off, kIoChunkBytes);
        in.read(reinterpret_cast<char*>(state_buf.data() + off),
                static_cast<std::streamsize>(take));
        if (in.gcount() != static_cast<std::streamsize>(take)) {
            return SlotStoreStatus::Error(SlotStoreError::kTruncatedFile,
                                          "short read on slot sequence state payload");
        }
        sha.Update(std::span<const std::uint8_t>(state_buf.data() + off, take));
        off += take;
    }

    // Confirm no concurrent append occurred while streaming.
    const int extra = in.peek();
    if (extra != std::char_traits<char>::eof()) {
        return SlotStoreStatus::Error(
            SlotStoreError::kInvalidHeader,
            "slot file grew beyond declared length during restore");
    }

    const std::array<std::uint8_t, 32> actual_sha = sha.Finish();
    if (!DigestEqualsConstantTime(actual_sha, hdr.payload_sha256)) {
        return SlotStoreStatus::Error(SlotStoreError::kChecksumMismatch,
                                      "slot payload SHA-256 checksum mismatch");
    }
    for (const Token tok : tokens) {
        if (tok < 0) {
            return SlotStoreStatus::Error(
                SlotStoreError::kInvalidHeader,
                "slot token table contains negative token ID");
        }
    }

    // Only after every header, length, identity, and checksum check passes do we
    // touch the live sequence state in `backend`.
    if (SlotStoreStatus st = backend.SetSequenceStateData(dest_seq_id, state_buf); !st.ok()) {
        return st;
    }

    if (out_result != nullptr) {
        out_result->header = hdr;
        out_result->tokens = std::move(tokens);
        out_result->restored_state_bytes = state_buf.size();
    }
    return SlotStoreStatus::Ok();
}

SlotStoreStatus SaveLlamaSequenceSlot(const fs::path& path, llama_context* ctx,
                                      std::int32_t seq_id, const SlotSaveMetadata& meta,
                                      const SlotStoreLimits& limits, SlotFileHeader* out_header) {
    if (ctx == nullptr) {
        if (out_header != nullptr) *out_header = {};
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "llama_context is null");
    }
    const LlamaCppSlotStateBackend backend(ctx);
    return SaveSlotFile(path, seq_id, meta, backend, limits, out_header);
}

SlotStoreStatus RestoreLlamaSequenceSlot(const fs::path& path, llama_context* ctx,
                                         std::int32_t dest_seq_id,
                                         const SlotRestoreExpectation& expected,
                                         SlotRestoreResult* out_result,
                                         const SlotStoreLimits& limits) {
    if (ctx == nullptr) {
        if (out_result != nullptr) *out_result = {};
        return SlotStoreStatus::Error(SlotStoreError::kInvalidArgument, "llama_context is null");
    }
    LlamaCppSlotStateBackend backend(ctx);
    return RestoreSlotFile(path, dest_seq_id, expected, backend, out_result, limits);
}

}  // namespace sonder::backends::llamacpp
