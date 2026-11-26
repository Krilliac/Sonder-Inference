// Internal: FIPS 180-4 SHA-256 shared across core, artifact manifests, server identity,
// and calibration provenance checks.
#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>

namespace sonder::inference::detail {

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

// Validates and normalizes a 64-character hex SHA-256 string to lowercase.
bool normalize_sha256_hex(std::string_view input, std::string& normalized_out);

// Streams `path` in bounded chunks to compute its lowercase SHA-256 hex digest.
bool sha256_file_hex(const std::filesystem::path& path, std::string& hex_out, std::string& error_out,
                     const std::function<bool()>& cancelled = {});

}  // namespace sonder::inference::detail
