#include "rpc_worker.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace sonder::inference::llamaserver {
namespace {

// Echoes at most 64 bytes of operator input, printable ASCII only, so a
// hostile config cannot push control characters or megabytes into logs.
std::string printable(std::string_view text) {
    constexpr std::size_t kMax = 64;
    std::string out;
    out.reserve(std::min(text.size(), kMax) + 3);
    for (std::size_t i = 0; i < text.size() && i < kMax; ++i) {
        const auto c = static_cast<unsigned char>(text[i]);
        out.push_back(c >= 0x20 && c < 0x7f ? static_cast<char>(c) : '?');
    }
    if (text.size() > kMax)
        out += "...";
    return out;
}

Status bad(std::string message) { return Status(ErrorCode::invalid_argument, "llamaserver: " + std::move(message)); }

// Plain decimal, 1..max_digits digits, no sign, no leading zero (except "0").
std::optional<std::uint32_t> parse_decimal(std::string_view text, std::size_t max_digits) {
    if (text.empty() || text.size() > max_digits)
        return std::nullopt;
    if (text.size() > 1 && text.front() == '0')
        return std::nullopt;
    std::uint32_t value = 0;
    for (const char c : text) {
        if (c < '0' || c > '9')
            return std::nullopt;
        value = value * 10u + static_cast<std::uint32_t>(c - '0');
    }
    return value;
}

std::optional<std::array<std::uint32_t, 4>> parse_ipv4(std::string_view host) {
    std::array<std::uint32_t, 4> octets{};
    std::size_t start = 0;
    for (std::size_t i = 0; i < octets.size(); ++i) {
        const std::size_t dot = i + 1 < octets.size() ? host.find('.', start) : host.size();
        if (dot == std::string_view::npos)
            return std::nullopt;
        const auto value = parse_decimal(host.substr(start, dot - start), 3);
        if (!value || *value > 255)
            return std::nullopt;
        octets[i] = *value;
        start = dot + 1;
    }
    return octets;
}

enum class AddressClass { local, publicly_routed, refused };

AddressClass classify(const std::array<std::uint32_t, 4> &a) {
    if (a[0] == 0 || a[0] >= 224)
        return AddressClass::refused;  // "this network", multicast, reserved, broadcast
    if (a[0] == 127 || a[0] == 10)
        return AddressClass::local;  // loopback, RFC 1918
    if (a[0] == 172 && a[1] >= 16 && a[1] <= 31)
        return AddressClass::local;  // RFC 1918
    if (a[0] == 192 && a[1] == 168)
        return AddressClass::local;  // RFC 1918
    if (a[0] == 169 && a[1] == 254)
        return AddressClass::local;  // link-local
    if (a[0] == 100 && a[1] >= 64 && a[1] <= 127)
        return AddressClass::local;  // RFC 6598 shared (CGNAT, overlay VPNs)
    return AddressClass::publicly_routed;
}

} // namespace

bool is_rpc_flag(std::string_view arg) noexcept {
    // llama-server turns every '_' of a "--" argument into '-' before the
    // exact lookup; a future parser may also accept "--flag=value".
    const std::string_view name = arg.substr(0, arg.find('='));
    constexpr std::string_view kRpc = "--rpc";
    if (name.size() != kRpc.size())
        return false;
    for (std::size_t i = 0; i < kRpc.size(); ++i) {
        const char c = name[i] == '_' && i >= 2 ? '-' : name[i];
        if (c != kRpc[i])
            return false;
    }
    return true;
}

Status refuse_operator_rpc_arguments(const std::vector<std::string> &arguments) {
    for (const auto &arg : arguments) {
        if (is_rpc_flag(arg))
            return bad("args must not contain --rpc: the llama.cpp RPC client runs only in an isolated worker child; "
                       "configure rpc_worker.servers instead (docs/integration/rpc-worker.md)");
    }
    return Status::success();
}

Status validate_rpc_server(std::string_view server, bool allow_public) {
    const std::size_t colon = server.rfind(':');
    if (server.empty() || server.size() > 21 || colon == std::string_view::npos || colon == 0)
        return bad("rpc_worker.servers entry '" + printable(server) + "' must be IPV4:PORT");
    const auto octets = parse_ipv4(server.substr(0, colon));
    if (!octets)
        return bad("rpc_worker.servers entry '" + printable(server) +
                   "' must use a dotted-decimal IPv4 address (host names and IPv6 are refused)");
    const auto port = parse_decimal(server.substr(colon + 1), 5);
    if (!port || *port == 0 || *port > 65535)
        return bad("rpc_worker.servers entry '" + printable(server) + "' needs a port in 1..65535");
    switch (classify(*octets)) {
    case AddressClass::local:
        return Status::success();
    case AddressClass::publicly_routed:
        if (allow_public)
            return Status::success();
        return bad("rpc_worker.servers entry '" + printable(server) +
                   "' is a public address; RPC is unauthenticated clear text, set "
                   "rpc_worker.allow_public_addresses to accept it");
    case AddressClass::refused:
        break;
    }
    return bad("rpc_worker.servers entry '" + printable(server) +
               "' is an unspecified, multicast, broadcast or reserved address");
}

Status validate_rpc_worker(const LlamaServerRpcWorkerOptions &options, LlamaServerMode mode) {
    if (!options.enabled) {
        if (!options.servers.empty())
            return bad("rpc_worker.servers is set but rpc_worker.enabled is false");
        return Status::success();
    }
    if (mode != LlamaServerMode::spawn)
        return bad("rpc_worker needs spawn mode: the RPC client must run in a child this backend supervises");
    if (options.servers.empty())
        return bad("rpc_worker.enabled needs at least one rpc_worker.servers entry");
    if (options.servers.size() > LlamaServerRpcWorkerOptions::kMaxServers)
        return bad("rpc_worker.servers has more than " + std::to_string(LlamaServerRpcWorkerOptions::kMaxServers) +
                   " entries");
    for (std::size_t i = 0; i < options.servers.size(); ++i) {
        if (auto st = validate_rpc_server(options.servers[i], options.allow_public_addresses); !st.ok())
            return st;
        // Strict parsing leaves one spelling per endpoint, so string equality
        // is endpoint equality.
        for (std::size_t j = 0; j < i; ++j) {
            if (options.servers[j] == options.servers[i])
                return bad("rpc_worker.servers lists '" + printable(options.servers[i]) + "' twice");
        }
    }
    return Status::success();
}

std::vector<std::string> rpc_worker_arguments(const LlamaServerRpcWorkerOptions &options,
                                              const std::vector<std::string> &arguments) {
    if (!options.enabled || options.servers.empty())
        return arguments;
    std::string joined;
    for (const auto &server : options.servers) {
        if (!joined.empty())
            joined.push_back(',');
        joined += server;
    }
    std::vector<std::string> out;
    out.reserve(arguments.size() + 2);
    out.emplace_back("--rpc");
    out.push_back(std::move(joined));
    out.insert(out.end(), arguments.begin(), arguments.end());
    return out;
}

} // namespace sonder::inference::llamaserver
