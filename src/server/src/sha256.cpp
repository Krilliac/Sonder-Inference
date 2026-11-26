#include "sha256.hpp"

namespace sonder::inference::server::detail {

std::string sha256_hex(std::string_view data) {
    return sonder::inference::detail::sha256_hex(data);
}

}  // namespace sonder::inference::server::detail
