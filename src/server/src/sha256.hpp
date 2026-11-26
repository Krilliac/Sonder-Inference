// Internal: forwards to common SHA-256 (FIPS 180-4) under server::detail so
// existing server/artifact consumers and tests keep their exact types.
#pragma once

#include "common/sha256.hpp"

namespace sonder::inference::server::detail {

using Sha256 = sonder::inference::detail::Sha256;

std::string sha256_hex(std::string_view data);

}  // namespace sonder::inference::server::detail
