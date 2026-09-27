// Internal: the one loopback rule shared by the server bind check, the Ollama
// client's remote guard and the CLI's plain-HTTP guard.
//
// Loopback means a name that can only reach this machine without trusting
// DNS: "localhost" (exact spelling, as before), the IPv6 literal ::1 (optionally bracketed),
// or a dotted-quad IPv4 literal in 127.0.0.0/8. A DNS name that merely
// starts with "127." (for example "127.0.0.1.attacker.example") is NOT
// loopback: it resolves wherever its owner says.
#pragma once

#include <cstddef>
#include <string_view>

namespace sonder::inference::detail {

// True for a strict dotted-quad IPv4 literal (four decimal octets 0..255, no
// leading zeros, nothing else) whose first octet is 127.
constexpr bool is_ipv4_loopback_literal(std::string_view host) noexcept {
    int octets = 0;
    int first = -1;
    std::size_t i = 0;
    while (true) {
        if (i >= host.size() || host[i] < '0' || host[i] > '9') {
            return false;
        }
        const std::size_t start = i;
        int value = 0;
        while (i < host.size() && host[i] >= '0' && host[i] <= '9') {
            if (i - start >= 3) {
                return false;
            }
            value = value * 10 + (host[i] - '0');
            ++i;
        }
        if (value > 255 || (i - start > 1 && host[start] == '0')) {
            return false;
        }
        if (octets == 0) {
            first = value;
        }
        ++octets;
        if (i == host.size()) {
            break;
        }
        if (host[i] != '.' || octets == 4) {
            return false;
        }
        ++i;
    }
    return octets == 4 && first == 127;
}

// `host` is a bare host (no port). Brackets around an IPv6 literal are allowed.
constexpr bool is_loopback_literal(std::string_view host) noexcept {
    if (host.size() > 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
        return host == "::1";
    }
    return host == "localhost" || host == "::1" || is_ipv4_loopback_literal(host);
}

}  // namespace sonder::inference::detail
