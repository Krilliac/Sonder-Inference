// Internal: SHA-256 (FIPS 180-4), used for the synthetic MOCK backend
// identity digests and fixed descriptor hashes. Not a security primitive
// here: nothing secret is hashed.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <string_view>

namespace sonder::inference::server::detail {

class Sha256 {
public:
    Sha256();
    void update(std::string_view data);
    std::array<std::uint8_t, 32> finish();

private:
    void block(const std::uint8_t* p);

    std::array<std::uint32_t, 8> h_;
    std::array<std::uint8_t, 64> buf_{};
    std::size_t buf_len_ = 0;
    std::uint64_t total_bytes_ = 0;
};

// Lowercase 64-character hex digest of `data`.
std::string sha256_hex(std::string_view data);

}  // namespace sonder::inference::server::detail
