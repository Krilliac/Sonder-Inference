// Lossless PQ2_0 (group 128, type 142) -> stock Q2_0 (group 64, type 42)
// GGUF re-blocking and dequantized parity verification.
#include "pq2_reblock.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "ggml.h"

namespace sonder::backends::llamacpp {

namespace {

constexpr std::uint32_t kGgufTypeUint8 = 0;
constexpr std::uint32_t kGgufTypeInt8 = 1;
constexpr std::uint32_t kGgufTypeUint16 = 2;
constexpr std::uint32_t kGgufTypeInt16 = 3;
constexpr std::uint32_t kGgufTypeUint32 = 4;
constexpr std::uint32_t kGgufTypeInt32 = 5;
constexpr std::uint32_t kGgufTypeFloat32 = 6;
constexpr std::uint32_t kGgufTypeBool = 7;
constexpr std::uint32_t kGgufTypeString = 8;
constexpr std::uint32_t kGgufTypeArray = 9;
constexpr std::uint32_t kGgufTypeUint64 = 10;
constexpr std::uint32_t kGgufTypeInt64 = 11;
constexpr std::uint32_t kGgufTypeFloat64 = 12;

constexpr std::uint64_t kMaxHeaderCount = 1'000'000ULL;
constexpr std::uint64_t kMaxStringBytes = 64ULL << 20;
constexpr std::size_t kStreamChunkPq2Blocks = 1024;
constexpr std::size_t kRawCopyChunkBytes = 64ULL << 10;

// Self-contained streaming SHA-256 (FIPS 180-4) for output file digests.
class Sha256Hasher {
public:
    Sha256Hasher() noexcept { reset(); }

    void reset() noexcept {
        state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        total_bytes_ = 0;
        buf_len_ = 0;
    }

    void update(const void* data, std::size_t len) noexcept {
        const auto* p = static_cast<const std::uint8_t*>(data);
        total_bytes_ += len;
        if (buf_len_ > 0) {
            const std::size_t take = std::min<std::size_t>(64u - buf_len_, len);
            std::memcpy(buf_.data() + buf_len_, p, take);
            buf_len_ += take;
            p += take;
            len -= take;
            if (buf_len_ == 64u) {
                compress(buf_.data());
                buf_len_ = 0;
            }
        }
        while (len >= 64u) {
            compress(p);
            p += 64u;
            len -= 64u;
        }
        if (len > 0) {
            std::memcpy(buf_.data(), p, len);
            buf_len_ = len;
        }
    }

    [[nodiscard]] std::string finish_hex() noexcept {
        const std::uint64_t bit_len = total_bytes_ * 8ULL;
        const std::uint8_t pad_one = 0x80u;
        update(&pad_one, 1);
        const std::uint8_t zero = 0x00u;
        while (buf_len_ != 56u) {
            update(&zero, 1);
        }
        std::uint8_t len_be[8];
        for (int i = 0; i < 8; ++i) {
            len_be[7 - i] = static_cast<std::uint8_t>((bit_len >> (i * 8)) & 0xFFu);
        }
        update(len_be, 8);

        static constexpr char kHex[] = "0123456789abcdef";
        std::string hex;
        hex.reserve(64);
        for (const std::uint32_t word : state_) {
            for (int shift = 24; shift >= 0; shift -= 8) {
                const auto byte = static_cast<std::uint8_t>((word >> shift) & 0xFFu);
                hex.push_back(kHex[byte >> 4]);
                hex.push_back(kHex[byte & 0x0Fu]);
            }
        }
        return hex;
    }

private:
    static std::uint32_t rotr(std::uint32_t x, unsigned n) noexcept {
        return (x >> n) | (x << (32u - n));
    }

    void compress(const std::uint8_t block[64]) noexcept {
        static constexpr std::uint32_t kRound[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcU, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
        };
        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                   static_cast<std::uint32_t>(block[i * 4 + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ ((~e) & g);
            const std::uint32_t temp1 = h + s1 + ch + kRound[i] + w[i];
            const std::uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{};
    std::uint64_t total_bytes_ = 0;
    std::array<std::uint8_t, 64> buf_{};
    std::size_t buf_len_ = 0;
};

[[nodiscard]] bool IsFp16Finite(std::uint16_t h) noexcept {
    return ((h >> 10) & 0x1Fu) != 0x1Fu;
}

// Exact IEEE-754 binary16 -> binary32 conversion (finite inputs).
[[nodiscard]] float Fp16ToFp32(std::uint16_t h) noexcept {
    const std::uint32_t sign = static_cast<std::uint32_t>(h >> 15) & 0x01u;
    const std::uint32_t exp = static_cast<std::uint32_t>(h >> 10) & 0x1Fu;
    std::uint32_t mant = static_cast<std::uint32_t>(h) & 0x03FFu;
    std::uint32_t bits = 0;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign << 31;
        } else {
            int e = -14;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                --e;
            }
            mant &= 0x03FFu;
            bits = (sign << 31) | (static_cast<std::uint32_t>(e + 127) << 23) | (mant << 13);
        }
    } else {
        bits = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

[[nodiscard]] bool CheckedMulU64(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > std::numeric_limits<std::uint64_t>::max() / b) {
        return false;
    }
    out = a * b;
    return true;
}

[[nodiscard]] bool CheckedAddU64(std::uint64_t a, std::uint64_t b, std::uint64_t& out) noexcept {
    if (a > std::numeric_limits<std::uint64_t>::max() - b) {
        return false;
    }
    out = a + b;
    return true;
}

[[nodiscard]] bool AlignUpU64(std::uint64_t value, std::uint64_t alignment, std::uint64_t& out) noexcept {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        return false;
    }
    const std::uint64_t mask = alignment - 1;
    if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
        return false;
    }
    out = (value + mask) & ~mask;
    return true;
}

struct KvSpan {
    std::string key;
    std::uint32_t val_type = 0;
    std::uint64_t raw_offset = 0;  // start of entire KV record in input file
    std::uint64_t raw_bytes = 0;   // full byte length of KV record (key + type + value)
};

struct TensorEntry {
    std::string name;
    std::vector<std::uint64_t> dims;
    std::uint32_t type = 0;
    std::uint64_t offset = 0;       // relative to data_section_offset
    std::uint64_t n_elements = 0;
    std::uint64_t byte_size = 0;
};

struct ParsedGguf {
    std::uint32_t version = 0;
    std::uint64_t alignment = 32;
    std::uint64_t file_size = 0;
    std::uint64_t data_section_offset = 0;
    bool prism_hadamard = false;
    std::optional<std::uint32_t> file_type;
    std::vector<KvSpan> kvs;
    std::vector<TensorEntry> tensors;
};

class BinaryReader {
public:
    BinaryReader(std::ifstream& in, std::uint64_t file_size) : in_(in), file_size_(file_size) {}

    [[nodiscard]] std::uint64_t pos() const noexcept { return pos_; }
    [[nodiscard]] std::uint64_t remaining() const noexcept {
        return pos_ <= file_size_ ? (file_size_ - pos_) : 0;
    }

    [[nodiscard]] bool read_exact(void* dst, std::size_t n) {
        if (n > remaining()) return false;
        in_.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(n));
        if (!in_.good() && !in_.eof()) return false;
        if (static_cast<std::size_t>(in_.gcount()) != n) return false;
        pos_ += n;
        return true;
    }

    [[nodiscard]] bool skip_bytes(std::uint64_t n) {
        if (n > remaining()) return false;
        const std::uint64_t next = pos_ + n;
        if (next > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
            return false;
        }
        in_.seekg(static_cast<std::streamoff>(next), std::ios::beg);
        if (!in_.good()) return false;
        pos_ = next;
        return true;
    }

    [[nodiscard]] bool read_u32(std::uint32_t& out) {
        std::uint8_t b[4];
        if (!read_exact(b, 4)) return false;
        out = static_cast<std::uint32_t>(b[0]) |
              (static_cast<std::uint32_t>(b[1]) << 8) |
              (static_cast<std::uint32_t>(b[2]) << 16) |
              (static_cast<std::uint32_t>(b[3]) << 24);
        return true;
    }

    [[nodiscard]] bool read_i32(std::int32_t& out) {
        std::uint32_t u = 0;
        if (!read_u32(u)) return false;
        std::memcpy(&out, &u, sizeof(out));
        return true;
    }

    [[nodiscard]] bool read_u64(std::uint64_t& out) {
        std::uint8_t b[8];
        if (!read_exact(b, 8)) return false;
        out = 0;
        for (int i = 0; i < 8; ++i) {
            out |= static_cast<std::uint64_t>(b[i]) << (i * 8);
        }
        return true;
    }

    [[nodiscard]] bool read_str(std::string& out, std::uint64_t max_len) {
        std::uint64_t len = 0;
        if (!read_u64(len)) return false;
        if (len > max_len || len > remaining()) return false;
        out.assign(static_cast<std::size_t>(len), '\0');
        if (len > 0 && !read_exact(out.data(), static_cast<std::size_t>(len))) {
            return false;
        }
        return true;
    }

private:
    std::ifstream& in_;
    std::uint64_t file_size_ = 0;
    std::uint64_t pos_ = 0;
};

[[nodiscard]] std::uint64_t ScalarGgufValueSize(std::uint32_t vtype) noexcept {
    switch (vtype) {
        case kGgufTypeUint8:
        case kGgufTypeInt8:
        case kGgufTypeBool: return 1;
        case kGgufTypeUint16:
        case kGgufTypeInt16: return 2;
        case kGgufTypeUint32:
        case kGgufTypeInt32:
        case kGgufTypeFloat32: return 4;
        case kGgufTypeUint64:
        case kGgufTypeInt64:
        case kGgufTypeFloat64: return 8;
        default: return 0;
    }
}

[[nodiscard]] bool SkipGgufValue(BinaryReader& r, std::uint32_t vtype, int depth = 0) {
    if (depth > 4) return false;
    const std::uint64_t scalar = ScalarGgufValueSize(vtype);
    if (scalar > 0) {
        if (vtype == kGgufTypeBool) {
            std::uint8_t bv = 0;
            if (!r.read_exact(&bv, 1)) return false;
            return bv <= 1u;
        }
        return r.skip_bytes(scalar);
    }
    if (vtype == kGgufTypeString) {
        std::uint64_t slen = 0;
        if (!r.read_u64(slen) || slen > kMaxStringBytes) return false;
        return r.skip_bytes(slen);
    }
    if (vtype == kGgufTypeArray) {
        std::uint32_t elem_type = 0;
        std::uint64_t count = 0;
        if (!r.read_u32(elem_type) || !r.read_u64(count)) return false;
        const std::uint64_t elem_scalar = ScalarGgufValueSize(elem_type);
        if (elem_scalar > 0) {
            if (elem_type == kGgufTypeBool) {
                if (count > r.remaining()) return false;
                for (std::uint64_t i = 0; i < count; ++i) {
                    std::uint8_t bv = 0;
                    if (!r.read_exact(&bv, 1) || bv > 1u) return false;
                }
                return true;
            }
            std::uint64_t bytes = 0;
            if (!CheckedMulU64(count, elem_scalar, bytes)) return false;
            return r.skip_bytes(bytes);
        }
        if (count > r.remaining() / 8u) return false;
        for (std::uint64_t i = 0; i < count; ++i) {
            if (!SkipGgufValue(r, elem_type, depth + 1)) return false;
        }
        return true;
    }
    return false;
}

[[nodiscard]] Status TensorTypeLayout(std::uint32_t type, std::string_view tensor_name,
                                      std::uint64_t& out_block_elems, std::uint64_t& out_block_bytes) {
    if (type == kGgmlTypePtq1_0 || type == kGgmlTypePq1_0 || type == kGgmlTypePq2_0m) {
        return Status::Error(
            ErrorCode::kUnsupported,
            "tensor '" + std::string(tensor_name) + "' uses unsupported private Prism tensor type " +
                std::to_string(type) + " (only PQ2_0 type 142 re-blocks losslessly to stock Q2_0)");
    }
    if (type == kGgmlTypePq2_0) {
        out_block_elems = kPq2BlockElements;
        out_block_bytes = kPq2BlockBytes;
        return Status::Ok();
    }
    if (type >= static_cast<std::uint32_t>(GGML_TYPE_COUNT)) {
        return Status::Error(
            ErrorCode::kUnsupported,
            "tensor '" + std::string(tensor_name) + "' uses unknown ggml type " + std::to_string(type));
    }
    const auto* traits = ggml_get_type_traits(static_cast<ggml_type>(type));
    if (traits == nullptr || traits->blck_size <= 0 || traits->type_size == 0) {
        return Status::Error(
            ErrorCode::kUnsupported,
            "tensor '" + std::string(tensor_name) + "' uses uninitialised ggml type " + std::to_string(type));
    }
    out_block_elems = static_cast<std::uint64_t>(traits->blck_size);
    out_block_bytes = static_cast<std::uint64_t>(traits->type_size);
    return Status::Ok();
}

[[nodiscard]] Status ParseGgufLayout(const std::filesystem::path& path, ParsedGguf& out) {
    out = {};
    if (path.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "GGUF path is empty");
    }
    std::error_code ec;
    const auto sym_st = std::filesystem::symlink_status(path, ec);
    if (ec || sym_st.type() == std::filesystem::file_type::not_found) {
        return Status::Error(ErrorCode::kFileNotFound, "GGUF file not found: " + path.generic_string());
    }
    if (std::filesystem::is_symlink(sym_st)) {
        return Status::Error(ErrorCode::kInvalidArgument, "GGUF symlink path refused: " + path.generic_string());
    }
    if (!std::filesystem::is_regular_file(sym_st)) {
        return Status::Error(ErrorCode::kFileNotFound, "GGUF path is not a regular file: " + path.generic_string());
    }
    const std::uint64_t fsize = std::filesystem::file_size(path, ec);
    if (ec || fsize < 24u) {
        return Status::Error(ErrorCode::kInvalidArgument, "truncated or unreadable GGUF header");
    }
    out.file_size = fsize;

    std::ifstream in(path, std::ios::binary);
    if (!in.good()) {
        return Status::Error(ErrorCode::kFileNotFound, "failed to open GGUF file: " + path.generic_string());
    }

    BinaryReader r(in, fsize);
    char magic[4] = {};
    if (!r.read_exact(magic, 4) || std::memcmp(magic, "GGUF", 4) != 0) {
        return Status::Error(ErrorCode::kInvalidArgument, "not a GGUF file (bad magic)");
    }
    if (!r.read_u32(out.version) || (out.version != 2u && out.version != 3u)) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "unsupported GGUF version (expected 2 or 3)");
    }

    std::uint64_t n_tensors = 0;
    std::uint64_t n_kv = 0;
    if (!r.read_u64(n_tensors) || !r.read_u64(n_kv)) {
        return Status::Error(ErrorCode::kInvalidArgument, "truncated GGUF counts");
    }
    // Bound counts against remaining file bytes BEFORE reserving vectors.
    if (n_kv > kMaxHeaderCount || n_kv > r.remaining() / 13u) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "corrupt GGUF metadata count (" + std::to_string(n_kv) + ")");
    }
    if (n_tensors > kMaxHeaderCount || n_tensors > r.remaining() / 24u) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "corrupt GGUF tensor count (" + std::to_string(n_tensors) + ")");
    }

    out.kvs.reserve(static_cast<std::size_t>(n_kv));
    std::unordered_set<std::string> seen_keys;
    seen_keys.reserve(static_cast<std::size_t>(n_kv));
    for (std::uint64_t i = 0; i < n_kv; ++i) {
        const std::uint64_t kv_start = r.pos();
        std::string key;
        std::uint32_t vtype = 0;
        if (!r.read_str(key, 65536u) || key.empty() || !r.read_u32(vtype)) {
            return Status::Error(ErrorCode::kInvalidArgument, "corrupt or truncated GGUF KV entry");
        }
        if (!seen_keys.insert(key).second) {
            return Status::Error(ErrorCode::kInvalidArgument, "duplicate GGUF metadata key: " + key);
        }
        if (key.rfind("prism.hadamard.", 0) == 0) {
            out.prism_hadamard = true;
        }
        if (key == "general.alignment") {
            if (vtype != kGgufTypeUint32) {
                return Status::Error(ErrorCode::kInvalidArgument, "general.alignment must be UINT32");
            }
            std::uint32_t align_val = 0;
            if (!r.read_u32(align_val) || align_val == 0 || (align_val & (align_val - 1)) != 0 ||
                align_val > (1u << 20)) {
                return Status::Error(ErrorCode::kInvalidArgument, "invalid general.alignment value");
            }
            out.alignment = align_val;
        } else if (key == "general.file_type" && (vtype == kGgufTypeUint32 || vtype == kGgufTypeInt32)) {
            if (vtype == kGgufTypeUint32) {
                std::uint32_t ft = 0;
                if (!r.read_u32(ft)) {
                    return Status::Error(ErrorCode::kInvalidArgument, "truncated general.file_type");
                }
                out.file_type = ft;
            } else {
                std::int32_t ft = 0;
                if (!r.read_i32(ft) || ft < 0) {
                    return Status::Error(ErrorCode::kInvalidArgument, "invalid general.file_type");
                }
                out.file_type = static_cast<std::uint32_t>(ft);
            }
        } else {
            if (!SkipGgufValue(r, vtype)) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "malformed or truncated GGUF value for key '" + key + "'");
            }
        }
        const std::uint64_t kv_end = r.pos();
        out.kvs.push_back(KvSpan{std::move(key), vtype, kv_start, kv_end - kv_start});
    }

    out.tensors.reserve(static_cast<std::size_t>(n_tensors));
    std::unordered_set<std::string> seen_tensors;
    seen_tensors.reserve(static_cast<std::size_t>(n_tensors));
    for (std::uint64_t i = 0; i < n_tensors; ++i) {
        TensorEntry t;
        std::uint32_t n_dims = 0;
        if (!r.read_str(t.name, 1024u) || t.name.empty() || !r.read_u32(n_dims)) {
            return Status::Error(ErrorCode::kInvalidArgument, "corrupt GGUF tensor descriptor");
        }
        if (!seen_tensors.insert(t.name).second) {
            return Status::Error(ErrorCode::kInvalidArgument, "duplicate GGUF tensor name: " + t.name);
        }
        if (n_dims == 0 || n_dims > 4u) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "invalid dimension count on tensor '" + t.name + "'");
        }
        t.dims.resize(n_dims);
        std::uint64_t elems = 1;
        for (std::uint32_t d = 0; d < n_dims; ++d) {
            if (!r.read_u64(t.dims[d]) || t.dims[d] == 0) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "zero or unreadable dimension on tensor '" + t.name + "'");
            }
            if (!CheckedMulU64(elems, t.dims[d], elems)) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "element count overflow on tensor '" + t.name + "'");
            }
        }
        if (!r.read_u32(t.type) || !r.read_u64(t.offset)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "truncated type/offset on tensor '" + t.name + "'");
        }
        std::uint64_t block_elems = 0;
        std::uint64_t block_bytes = 0;
        const Status layout_st = TensorTypeLayout(t.type, t.name, block_elems, block_bytes);
        if (!layout_st.ok()) return layout_st;

        if ((t.dims[0] % block_elems) != 0) {
            return Status::Error(
                ErrorCode::kInvalidArgument,
                "tensor '" + t.name + "' ne[0]=" + std::to_string(t.dims[0]) +
                    " is not divisible by block size " + std::to_string(block_elems));
        }
        const std::uint64_t n_blocks = elems / block_elems;
        if (!CheckedMulU64(n_blocks, block_bytes, t.byte_size)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "byte size overflow on tensor '" + t.name + "'");
        }
        t.n_elements = elems;
        out.tensors.push_back(std::move(t));
    }

    if (!AlignUpU64(r.pos(), out.alignment, out.data_section_offset) || out.data_section_offset > fsize) {
        return Status::Error(ErrorCode::kInvalidArgument, "truncated GGUF data section padding");
    }
    const std::uint64_t data_avail = fsize - out.data_section_offset;

    // Validate tensor offsets: aligned, in-bounds, and non-overlapping.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
    spans.reserve(out.tensors.size());
    for (const auto& t : out.tensors) {
        if ((t.offset % out.alignment) != 0) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "unaligned offset on tensor '" + t.name + "'");
        }
        std::uint64_t end = 0;
        if (!CheckedAddU64(t.offset, t.byte_size, end) || end > data_avail) {
            return Status::Error(
                ErrorCode::kInvalidArgument,
                "tensor '" + t.name + "' exceeds GGUF payload bounds (possible truncated file or legacy g128-as-Q2_0 layout)");
        }
        spans.emplace_back(t.offset, end);
    }
    std::sort(spans.begin(), spans.end());
    for (std::size_t i = 1; i < spans.size(); ++i) {
        if (spans[i - 1].second > spans[i].first) {
            return Status::Error(ErrorCode::kInvalidArgument, "overlapping tensor regions in GGUF");
        }
    }
    return Status::Ok();
}

Pq2GgufInspection SummarizeParsedGguf(const ParsedGguf& g) {
    Pq2GgufInspection insp;
    insp.gguf_version = g.version;
    insp.tensor_count = g.tensors.size();
    insp.prism_hadamard = g.prism_hadamard;
    insp.file_type = g.file_type;
    for (const auto& t : g.tensors) {
        if (t.type == kGgmlTypePq2_0) {
            ++insp.pq2_tensor_count;
            insp.pq2_input_bytes += t.byte_size;
            insp.q2_output_bytes += (t.n_elements / kQ2BlockElements) * kQ2BlockBytes;
        } else {
            if (t.type == kGgmlTypeQ2_0) {
                ++insp.q2_tensor_count;
            }
            insp.other_tensor_bytes += t.byte_size;
        }
    }
    return insp;
}

class HashingWriter {
public:
    explicit HashingWriter(std::ofstream& out) : out_(out) {}

    [[nodiscard]] bool write_exact(const void* data, std::size_t len) {
        if (len == 0) return true;
        out_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(len));
        if (!out_.good()) return false;
        hasher_.update(data, len);
        bytes_written_ += len;
        return true;
    }

    [[nodiscard]] bool write_zeros(std::uint64_t len) {
        static constexpr std::array<std::uint8_t, 4096> kZeros{};
        while (len > 0) {
            const std::size_t chunk = static_cast<std::size_t>(std::min<std::uint64_t>(len, kZeros.size()));
            if (!write_exact(kZeros.data(), chunk)) return false;
            len -= chunk;
        }
        return true;
    }

    [[nodiscard]] bool write_u32(std::uint32_t v) {
        const std::uint8_t b[4] = {
            static_cast<std::uint8_t>(v & 0xFFu),
            static_cast<std::uint8_t>((v >> 8) & 0xFFu),
            static_cast<std::uint8_t>((v >> 16) & 0xFFu),
            static_cast<std::uint8_t>((v >> 24) & 0xFFu),
        };
        return write_exact(b, 4);
    }

    [[nodiscard]] bool write_i32(std::int32_t v) {
        std::uint32_t u = 0;
        std::memcpy(&u, &v, sizeof(u));
        return write_u32(u);
    }

    [[nodiscard]] bool write_u64(std::uint64_t v) {
        std::uint8_t b[8];
        for (int i = 0; i < 8; ++i) {
            b[i] = static_cast<std::uint8_t>((v >> (i * 8)) & 0xFFu);
        }
        return write_exact(b, 8);
    }

    [[nodiscard]] bool write_str(std::string_view s) {
        return write_u64(s.size()) && write_exact(s.data(), s.size());
    }

    [[nodiscard]] std::uint64_t bytes_written() const noexcept { return bytes_written_; }
    [[nodiscard]] std::string finish_sha256() noexcept { return hasher_.finish_hex(); }

private:
    std::ofstream& out_;
    Sha256Hasher hasher_;
    std::uint64_t bytes_written_ = 0;
};

[[nodiscard]] bool CopyRange(std::ifstream& in, std::uint64_t offset, std::uint64_t length, HashingWriter& out) {
    if (offset > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) return false;
    in.clear();
    in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!in.good()) return false;
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(std::min<std::uint64_t>(length, kRawCopyChunkBytes)));
    std::uint64_t left = length;
    while (left > 0) {
        const std::size_t chunk = static_cast<std::size_t>(std::min<std::uint64_t>(left, buf.size()));
        in.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(chunk));
        if (static_cast<std::size_t>(in.gcount()) != chunk) return false;
        if (!out.write_exact(buf.data(), chunk)) return false;
        left -= chunk;
    }
    return true;
}

[[nodiscard]] std::uint64_t CurrentPid() noexcept {
#if defined(_WIN32)
    return static_cast<std::uint64_t>(_getpid());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

struct TempFileGuard {
    std::filesystem::path path;
    bool committed = false;
    ~TempFileGuard() noexcept {
        if (!committed && !path.empty()) {
            std::error_code ec;
            std::filesystem::remove(path, ec);
        }
    }
};

}  // namespace

Status DequantizePq2Row(std::span<const std::uint8_t> pq2_bytes,
                        std::size_t n_elements,
                        std::vector<float>& out_floats) {
    out_floats.clear();
    if (n_elements == 0 || (n_elements % kPq2BlockElements) != 0) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "PQ2_0 element count must be a positive multiple of 128");
    }
    const std::size_t n_blocks = n_elements / kPq2BlockElements;
    if (n_blocks > std::numeric_limits<std::size_t>::max() / kPq2BlockBytes ||
        pq2_bytes.size() != n_blocks * kPq2BlockBytes) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "PQ2_0 buffer size does not match (n_elements / 128) * 34");
    }

    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* blk = pq2_bytes.data() + b * kPq2BlockBytes;
        const auto d_half = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(blk[0]) | (static_cast<std::uint16_t>(blk[1]) << 8));
        if (!IsFp16Finite(d_half)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "non-finite FP16 scale in PQ2_0 block " + std::to_string(b));
        }
    }

    out_floats.resize(n_elements);
    for (std::size_t b = 0; b < n_blocks; ++b) {
        const std::uint8_t* blk = pq2_bytes.data() + b * kPq2BlockBytes;
        const auto d_half = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(blk[0]) | (static_cast<std::uint16_t>(blk[1]) << 8));
        const float d = Fp16ToFp32(d_half);
        float* dst = out_floats.data() + b * kPq2BlockElements;
        for (std::size_t j = 0; j < kPq2BlockElements; ++j) {
            const std::size_t byte_idx = 2u + (j >> 2);
            const unsigned bit_shift = static_cast<unsigned>((j & 3u) * 2u);
            const std::uint8_t q = (blk[byte_idx] >> bit_shift) & 0x03u;
            dst[j] = static_cast<float>(static_cast<int>(q) - 1) * d;
        }
    }
    return Status::Ok();
}

Status ReblockPq2RowToQ2(std::span<const std::uint8_t> pq2_bytes,
                         std::size_t n_elements,
                         std::vector<std::uint8_t>& out_q2_bytes) {
    out_q2_bytes.clear();
    if (n_elements == 0 || (n_elements % kPq2BlockElements) != 0) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "PQ2_0 element count must be a positive multiple of 128");
    }
    const std::size_t n_pq2_blocks = n_elements / kPq2BlockElements;
    if (n_pq2_blocks > std::numeric_limits<std::size_t>::max() / (2u * kQ2BlockBytes) ||
        pq2_bytes.size() != n_pq2_blocks * kPq2BlockBytes) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "PQ2_0 buffer size does not match (n_elements / 128) * 34");
    }

    for (std::size_t b = 0; b < n_pq2_blocks; ++b) {
        const std::uint8_t* src = pq2_bytes.data() + b * kPq2BlockBytes;
        const auto d_half = static_cast<std::uint16_t>(
            static_cast<std::uint16_t>(src[0]) | (static_cast<std::uint16_t>(src[1]) << 8));
        if (!IsFp16Finite(d_half)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "non-finite FP16 scale in PQ2_0 block " + std::to_string(b));
        }
    }

    out_q2_bytes.resize(n_pq2_blocks * 2u * kQ2BlockBytes);
    for (std::size_t b = 0; b < n_pq2_blocks; ++b) {
        const std::uint8_t* src = pq2_bytes.data() + b * kPq2BlockBytes;
        std::uint8_t* dst0 = out_q2_bytes.data() + (b * 2u) * kQ2BlockBytes;
        std::uint8_t* dst1 = dst0 + kQ2BlockBytes;
        dst0[0] = src[0];
        dst0[1] = src[1];
        std::memcpy(dst0 + 2, src + 2, 16u);
        dst1[0] = src[0];
        dst1[1] = src[1];
        std::memcpy(dst1 + 2, src + 18, 16u);
    }
    return Status::Ok();
}

Status InspectPq2Gguf(const std::filesystem::path& path,
                      Pq2GgufInspection& out_inspection) {
    ParsedGguf parsed;
    const Status st = ParseGgufLayout(path, parsed);
    if (!st.ok()) {
        out_inspection = {};
        return st;
    }
    out_inspection = SummarizeParsedGguf(parsed);
    return Status::Ok();
}

Status ReblockPq2ToQ2Gguf(const std::filesystem::path& input_path,
                          const std::filesystem::path& output_path,
                          Pq2ReblockReport* out_report) {
    if (out_report != nullptr) *out_report = {};
    if (input_path.empty() || output_path.empty()) {
        return Status::Error(ErrorCode::kInvalidArgument, "input and output GGUF paths must be non-empty");
    }

    ParsedGguf src;
    const Status parse_st = ParseGgufLayout(input_path, src);
    if (!parse_st.ok()) return parse_st;

    std::error_code ec;
    const auto out_sym = std::filesystem::symlink_status(output_path, ec);
    if (!ec && std::filesystem::is_symlink(out_sym)) {
        return Status::Error(ErrorCode::kInvalidArgument, "output GGUF symlink path refused");
    }
    ec.clear();
    if (std::filesystem::is_directory(output_path, ec)) {
        return Status::Error(ErrorCode::kInvalidArgument, "output GGUF path is a directory");
    }
    if (std::filesystem::exists(output_path, ec) && !ec) {
        if (std::filesystem::equivalent(input_path, output_path, ec) && !ec) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "output GGUF path must not alias input GGUF path");
        }
    }

    const Pq2GgufInspection insp = SummarizeParsedGguf(src);
    if (insp.pq2_tensor_count == 0) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "GGUF contains no PQ2_0 (type 142) tensors to re-block");
    }

    // Order tensors by source offset so output payload follows original tensor order.
    struct OutTensorPlan {
        std::size_t index = 0;
        std::uint32_t out_type = 0;
        std::uint64_t out_bytes = 0;
        std::uint64_t out_offset = 0;
    };
    std::vector<OutTensorPlan> plan(src.tensors.size());
    std::vector<std::size_t> order(src.tensors.size());
    for (std::size_t i = 0; i < src.tensors.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return src.tensors[a].offset < src.tensors[b].offset;
    });

    std::uint64_t running_off = 0;
    for (const std::size_t idx : order) {
        const auto& t = src.tensors[idx];
        std::uint64_t aligned = 0;
        if (!AlignUpU64(running_off, src.alignment, aligned)) {
            return Status::Error(ErrorCode::kInvalidArgument, "output tensor offset overflow");
        }
        const bool is_pq2 = (t.type == kGgmlTypePq2_0);
        const std::uint32_t out_type = is_pq2 ? kGgmlTypeQ2_0 : t.type;
        const std::uint64_t out_bytes =
            is_pq2 ? ((t.n_elements / kQ2BlockElements) * kQ2BlockBytes) : t.byte_size;
        if (!CheckedAddU64(aligned, out_bytes, running_off)) {
            return Status::Error(ErrorCode::kInvalidArgument, "output GGUF size overflow");
        }
        plan[idx] = OutTensorPlan{idx, out_type, out_bytes, aligned};
    }

    const std::filesystem::path parent = output_path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "failed to create output directory: " + parent.generic_string());
        }
    }

    static std::atomic<std::uint64_t> tmp_seq{0};
    const auto ticks = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    TempFileGuard tmp_guard{
        output_path.parent_path() /
        (output_path.filename().string() + ".tmp." + std::to_string(CurrentPid()) + "." +
         std::to_string(ticks) + "." + std::to_string(tmp_seq.fetch_add(1, std::memory_order_relaxed)))};

    std::ifstream in(input_path, std::ios::binary);
    if (!in.good()) {
        return Status::Error(ErrorCode::kFileNotFound, "failed to reopen input GGUF");
    }
    std::ofstream out_stream(tmp_guard.path, std::ios::binary | std::ios::trunc);
    if (!out_stream.good()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "failed to open temporary output GGUF: " + tmp_guard.path.generic_string());
    }

    HashingWriter w(out_stream);
    if (!w.write_exact("GGUF", 4) || !w.write_u32(src.version) ||
        !w.write_u64(src.tensors.size()) || !w.write_u64(src.kvs.size())) {
        return Status::Error(ErrorCode::kInvalidArgument, "failed writing output GGUF header");
    }

    for (const auto& kv : src.kvs) {
        const bool rewrite_ftype =
            (kv.key == "general.file_type") &&
            (kv.val_type == kGgufTypeUint32 || kv.val_type == kGgufTypeInt32) &&
            src.file_type.has_value() &&
            (*src.file_type == kPrismFtypeMostlyPq2_0 || *src.file_type == kPrismFtypeMostlyPq2_0Legacy);
        if (rewrite_ftype) {
            if (!w.write_str(kv.key) || !w.write_u32(kv.val_type)) {
                return Status::Error(ErrorCode::kInvalidArgument, "failed writing general.file_type key");
            }
            if (kv.val_type == kGgufTypeUint32) {
                if (!w.write_u32(kLlamaFtypeMostlyQ2_0)) {
                    return Status::Error(ErrorCode::kInvalidArgument, "failed writing general.file_type value");
                }
            } else {
                if (!w.write_i32(static_cast<std::int32_t>(kLlamaFtypeMostlyQ2_0))) {
                    return Status::Error(ErrorCode::kInvalidArgument, "failed writing general.file_type value");
                }
            }
        } else {
            if (!CopyRange(in, kv.raw_offset, kv.raw_bytes, w)) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "failed copying metadata key '" + kv.key + "'");
            }
        }
    }

    for (std::size_t i = 0; i < src.tensors.size(); ++i) {
        const auto& t = src.tensors[i];
        const auto& tp = plan[i];
        if (!w.write_str(t.name) || !w.write_u32(static_cast<std::uint32_t>(t.dims.size()))) {
            return Status::Error(ErrorCode::kInvalidArgument, "failed writing tensor descriptor");
        }
        for (const std::uint64_t d : t.dims) {
            if (!w.write_u64(d)) {
                return Status::Error(ErrorCode::kInvalidArgument, "failed writing tensor dimensions");
            }
        }
        if (!w.write_u32(tp.out_type) || !w.write_u64(tp.out_offset)) {
            return Status::Error(ErrorCode::kInvalidArgument, "failed writing tensor type/offset");
        }
    }

    std::uint64_t out_data_start = 0;
    if (!AlignUpU64(w.bytes_written(), src.alignment, out_data_start)) {
        return Status::Error(ErrorCode::kInvalidArgument, "output data alignment overflow");
    }
    if (!w.write_zeros(out_data_start - w.bytes_written())) {
        return Status::Error(ErrorCode::kInvalidArgument, "failed writing header alignment padding");
    }

    std::vector<std::uint8_t> pq2_chunk_buf;
    std::vector<std::uint8_t> q2_chunk_buf;
    std::uint64_t reblocked_count = 0;
    std::uint64_t copied_count = 0;

    for (const std::size_t idx : order) {
        const auto& t = src.tensors[idx];
        const auto& tp = plan[idx];
        const std::uint64_t target_file_pos = out_data_start + tp.out_offset;
        if (target_file_pos < w.bytes_written()) {
            return Status::Error(ErrorCode::kInvalidArgument, "internal output tensor offset regression");
        }
        if (!w.write_zeros(target_file_pos - w.bytes_written())) {
            return Status::Error(ErrorCode::kInvalidArgument, "failed writing tensor alignment padding");
        }

        const std::uint64_t src_pos = src.data_section_offset + t.offset;
        if (t.type == kGgmlTypePq2_0) {
            if (src_pos > static_cast<std::uint64_t>(std::numeric_limits<std::streamoff>::max())) {
                return Status::Error(ErrorCode::kInvalidArgument, "input tensor offset exceeds streamoff");
            }
            in.clear();
            in.seekg(static_cast<std::streamoff>(src_pos), std::ios::beg);
            if (!in.good()) {
                return Status::Error(ErrorCode::kInvalidArgument, "failed seeking to PQ2_0 tensor payload");
            }
            std::uint64_t blocks_left = t.n_elements / kPq2BlockElements;
            while (blocks_left > 0) {
                const std::size_t chunk_blocks =
                    static_cast<std::size_t>(std::min<std::uint64_t>(blocks_left, kStreamChunkPq2Blocks));
                const std::size_t in_bytes = chunk_blocks * kPq2BlockBytes;
                pq2_chunk_buf.resize(in_bytes);
                in.read(reinterpret_cast<char*>(pq2_chunk_buf.data()), static_cast<std::streamsize>(in_bytes));
                if (static_cast<std::size_t>(in.gcount()) != in_bytes) {
                    return Status::Error(ErrorCode::kInvalidArgument,
                                         "truncated read on PQ2_0 tensor '" + t.name + "'");
                }
                const Status rb_st =
                    ReblockPq2RowToQ2(pq2_chunk_buf, chunk_blocks * kPq2BlockElements, q2_chunk_buf);
                if (!rb_st.ok()) {
                    return Status::Error(rb_st.code, "tensor '" + t.name + "': " + rb_st.message);
                }
                if (!w.write_exact(q2_chunk_buf.data(), q2_chunk_buf.size())) {
                    return Status::Error(ErrorCode::kInvalidArgument,
                                         "failed writing Q2_0 payload for tensor '" + t.name + "'");
                }
                blocks_left -= chunk_blocks;
            }
            ++reblocked_count;
        } else {
            if (!CopyRange(in, src_pos, t.byte_size, w)) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "failed copying tensor '" + t.name + "'");
            }
            ++copied_count;
        }
    }

    out_stream.flush();
    if (!out_stream.good()) {
        return Status::Error(ErrorCode::kInvalidArgument, "failed flushing output GGUF");
    }
    const std::uint64_t final_bytes = w.bytes_written();
    const std::string sha256_hex = w.finish_sha256();
    out_stream.close();

    std::filesystem::rename(tmp_guard.path, output_path, ec);
    if (ec) {
        std::error_code rm_ec;
        std::filesystem::remove(output_path, rm_ec);
        ec.clear();
        std::filesystem::rename(tmp_guard.path, output_path, ec);
        if (ec) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "failed moving temporary GGUF into place: " + ec.message());
        }
    }
    tmp_guard.committed = true;

    if (out_report != nullptr) {
        out_report->inspection = insp;
        out_report->reblocked_tensors = reblocked_count;
        out_report->copied_tensors = copied_count;
        out_report->output_file_bytes = final_bytes;
        out_report->output_sha256 = sha256_hex;
    }
    return Status::Ok();
}

Status VerifyPq2ToQ2Gguf(const std::filesystem::path& pq2_path,
                         const std::filesystem::path& q2_path) {
    ParsedGguf src;
    const Status s1 = ParseGgufLayout(pq2_path, src);
    if (!s1.ok()) return s1;

    ParsedGguf dst;
    const Status s2 = ParseGgufLayout(q2_path, dst);
    if (!s2.ok()) return s2;

    if (src.version != dst.version || src.prism_hadamard != dst.prism_hadamard ||
        src.kvs.size() != dst.kvs.size() || src.tensors.size() != dst.tensors.size()) {
        return Status::Error(ErrorCode::kInvalidArgument,
                             "GGUF header structure mismatch between PQ2_0 source and Q2_0 output");
    }

    if (src.file_type.has_value() &&
        (*src.file_type == kPrismFtypeMostlyPq2_0 || *src.file_type == kPrismFtypeMostlyPq2_0Legacy)) {
        if (dst.file_type != std::optional<std::uint32_t>(kLlamaFtypeMostlyQ2_0)) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "output general.file_type was not updated to LLAMA_FTYPE_MOSTLY_Q2_0 (41)");
        }
    }

    std::ifstream in_src(pq2_path, std::ios::binary);
    std::ifstream in_dst(q2_path, std::ios::binary);
    if (!in_src.good() || !in_dst.good()) {
        return Status::Error(ErrorCode::kFileNotFound, "failed opening GGUF files for verification");
    }

    // Verify metadata byte-for-byte on every key except general.file_type.
    std::vector<std::uint8_t> kv_a;
    std::vector<std::uint8_t> kv_b;
    for (std::size_t i = 0; i < src.kvs.size(); ++i) {
        if (src.kvs[i].key != dst.kvs[i].key || src.kvs[i].val_type != dst.kvs[i].val_type) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "metadata key/type mismatch at index " + std::to_string(i));
        }
        if (src.kvs[i].key == "general.file_type") continue;
        if (src.kvs[i].raw_bytes != dst.kvs[i].raw_bytes) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "metadata byte length mismatch on key '" + src.kvs[i].key + "'");
        }
        std::uint64_t left = src.kvs[i].raw_bytes;
        std::uint64_t off_a = src.kvs[i].raw_offset;
        std::uint64_t off_b = dst.kvs[i].raw_offset;
        kv_a.resize(static_cast<std::size_t>(std::min<std::uint64_t>(left, kRawCopyChunkBytes)));
        kv_b.resize(kv_a.size());
        while (left > 0) {
            const std::size_t chunk = static_cast<std::size_t>(std::min<std::uint64_t>(left, kv_a.size()));
            in_src.clear();
            in_dst.clear();
            in_src.seekg(static_cast<std::streamoff>(off_a), std::ios::beg);
            in_dst.seekg(static_cast<std::streamoff>(off_b), std::ios::beg);
            in_src.read(reinterpret_cast<char*>(kv_a.data()), static_cast<std::streamsize>(chunk));
            in_dst.read(reinterpret_cast<char*>(kv_b.data()), static_cast<std::streamsize>(chunk));
            if (static_cast<std::size_t>(in_src.gcount()) != chunk ||
                static_cast<std::size_t>(in_dst.gcount()) != chunk ||
                std::memcmp(kv_a.data(), kv_b.data(), chunk) != 0) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "metadata value mismatch on key '" + src.kvs[i].key + "'");
            }
            off_a += chunk;
            off_b += chunk;
            left -= chunk;
        }
    }

    const auto* q2_traits = ggml_get_type_traits(GGML_TYPE_Q2_0);
    if (q2_traits == nullptr || q2_traits->to_float == nullptr) {
        return Status::Error(ErrorCode::kUnsupported, "GGML_TYPE_Q2_0 to_float traits unavailable");
    }

    std::vector<std::uint8_t> buf_a;
    std::vector<std::uint8_t> buf_b;
    std::vector<float> floats_a;
    std::vector<float> floats_b;

    for (std::size_t i = 0; i < src.tensors.size(); ++i) {
        const auto& ta = src.tensors[i];
        const auto& tb = dst.tensors[i];
        if (ta.name != tb.name || ta.dims != tb.dims || ta.n_elements != tb.n_elements) {
            return Status::Error(ErrorCode::kInvalidArgument,
                                 "tensor shape/name mismatch at '" + ta.name + "'");
        }
        const std::uint64_t pos_a = src.data_section_offset + ta.offset;
        const std::uint64_t pos_b = dst.data_section_offset + tb.offset;

        if (ta.type == kGgmlTypePq2_0) {
            if (tb.type != kGgmlTypeQ2_0) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "tensor '" + ta.name + "' was not converted to Q2_0");
            }
            in_src.clear();
            in_dst.clear();
            in_src.seekg(static_cast<std::streamoff>(pos_a), std::ios::beg);
            in_dst.seekg(static_cast<std::streamoff>(pos_b), std::ios::beg);
            std::uint64_t blocks_left = ta.n_elements / kPq2BlockElements;
            while (blocks_left > 0) {
                const std::size_t chunk_blocks =
                    static_cast<std::size_t>(std::min<std::uint64_t>(blocks_left, kStreamChunkPq2Blocks));
                const std::size_t elems = chunk_blocks * kPq2BlockElements;
                buf_a.resize(chunk_blocks * kPq2BlockBytes);
                buf_b.resize(chunk_blocks * 2u * kQ2BlockBytes);
                in_src.read(reinterpret_cast<char*>(buf_a.data()), static_cast<std::streamsize>(buf_a.size()));
                in_dst.read(reinterpret_cast<char*>(buf_b.data()), static_cast<std::streamsize>(buf_b.size()));
                if (static_cast<std::size_t>(in_src.gcount()) != buf_a.size() ||
                    static_cast<std::size_t>(in_dst.gcount()) != buf_b.size()) {
                    return Status::Error(ErrorCode::kInvalidArgument,
                                         "truncated read verifying tensor '" + ta.name + "'");
                }
                const Status dq_st = DequantizePq2Row(buf_a, elems, floats_a);
                if (!dq_st.ok()) return dq_st;

                // Also verify finite FP16 scales in every output Q2_0 block before calling ggml to_float.
                for (std::size_t qb = 0; qb < chunk_blocks * 2u; ++qb) {
                    const std::uint8_t* qptr = buf_b.data() + qb * kQ2BlockBytes;
                    const auto d_half = static_cast<std::uint16_t>(
                        static_cast<std::uint16_t>(qptr[0]) | (static_cast<std::uint16_t>(qptr[1]) << 8));
                    if (!IsFp16Finite(d_half)) {
                        return Status::Error(ErrorCode::kInvalidArgument,
                                             "non-finite Q2_0 scale in output tensor '" + ta.name + "'");
                    }
                }
                floats_b.resize(elems);
                q2_traits->to_float(buf_b.data(), floats_b.data(), static_cast<std::int64_t>(elems));
                if (std::memcmp(floats_a.data(), floats_b.data(), elems * sizeof(float)) != 0) {
                    return Status::Error(ErrorCode::kInvalidArgument,
                                         "dequantized numeric mismatch in tensor '" + ta.name + "'");
                }
                blocks_left -= chunk_blocks;
            }
        } else {
            if (tb.type != ta.type || tb.byte_size != ta.byte_size) {
                return Status::Error(ErrorCode::kInvalidArgument,
                                     "non-PQ2_0 tensor type/size changed on '" + ta.name + "'");
            }
            in_src.clear();
            in_dst.clear();
            in_src.seekg(static_cast<std::streamoff>(pos_a), std::ios::beg);
            in_dst.seekg(static_cast<std::streamoff>(pos_b), std::ios::beg);
            std::uint64_t left = ta.byte_size;
            buf_a.resize(static_cast<std::size_t>(std::min<std::uint64_t>(left, kRawCopyChunkBytes)));
            buf_b.resize(buf_a.size());
            while (left > 0) {
                const std::size_t chunk = static_cast<std::size_t>(std::min<std::uint64_t>(left, buf_a.size()));
                in_src.read(reinterpret_cast<char*>(buf_a.data()), static_cast<std::streamsize>(chunk));
                in_dst.read(reinterpret_cast<char*>(buf_b.data()), static_cast<std::streamsize>(chunk));
                if (static_cast<std::size_t>(in_src.gcount()) != chunk ||
                    static_cast<std::size_t>(in_dst.gcount()) != chunk ||
                    std::memcmp(buf_a.data(), buf_b.data(), chunk) != 0) {
                    return Status::Error(ErrorCode::kInvalidArgument,
                                         "byte mismatch in copied tensor '" + ta.name + "'");
                }
                left -= chunk;
            }
        }
    }
    return Status::Ok();
}

}  // namespace sonder::backends::llamacpp
