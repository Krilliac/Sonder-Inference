// Test support for the server module: an in-process Server on an ephemeral
// loopback port with the MOCK backend, and a raw-socket HTTP client (so tests
// control every header and can send malformed or partial requests).
#pragma once

#include <doctest/doctest.h>

#include <cctype>
#include <chrono>
#include <cstdlib>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#else
#  include <arpa/inet.h>
#  include <netinet/in.h>
#  include <sys/socket.h>
#endif

#include "sonder/inference.hpp"
#include "sonder/inference/server.hpp"
#include "src/socket.hpp"

namespace server_test {

namespace si = sonder::inference;
namespace srv = sonder::inference::server;
namespace det = sonder::inference::server::detail;

struct Reply {
    int status = 0;
    std::map<std::string, std::string> headers;  // lowercased names
    std::string body;
    bool complete = false;  // connection closed normally after the response

    [[nodiscard]] std::string header(const std::string& lower_name) const {
        auto it = headers.find(lower_name);
        return it == headers.end() ? std::string() : it->second;
    }
    [[nodiscard]] bool has_header(const std::string& lower_name) const { return headers.count(lower_name) != 0; }
    [[nodiscard]] si::json::Value json() const {
        auto v = si::json::parse(body);
        REQUIRE_MESSAGE(v.ok(), "body is not JSON: " << body);
        return v.value();
    }
};

inline std::string lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

// Parses "HTTP/1.1 200 OK\r\nK: V\r\n\r\nbody".
inline Reply parse_reply(const std::string& raw) {
    Reply r;
    const auto end = raw.find("\r\n\r\n");
    if (end == std::string::npos || raw.rfind("HTTP/1.1 ", 0) != 0) {
        return r;
    }
    r.status = std::atoi(raw.c_str() + 9);
    std::size_t pos = raw.find("\r\n") + 2;
    while (pos < end) {
        const auto le = raw.find("\r\n", pos);
        const std::string line = raw.substr(pos, le - pos);
        const auto colon = line.find(':');
        if (colon != std::string::npos) {
            std::string v = line.substr(colon + 1);
            v.erase(0, v.find_first_not_of(' '));
            r.headers[lower(line.substr(0, colon))] = v;
        }
        pos = le + 2;
    }
    r.body = raw.substr(end + 4);
    return r;
}

// Raw connection to the server.
class Conn {
public:
    explicit Conn(std::uint16_t port) {
        auto c = det::connect_tcp("127.0.0.1", port, std::chrono::milliseconds(3000));
        REQUIRE_MESSAGE(c.ok(), c.status().to_string());
        sock_ = std::move(c.value());
    }
    // A connection whose receive buffer is limited to `receive_buffer` bytes
    // from the start (set before connect, so the TCP window is negotiated
    // for it; shrinking it later can stall the peer on zero-window probes).
    Conn(std::uint16_t port, int receive_buffer) {
        det::Socket s(static_cast<det::native_socket>(::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP)));
        REQUIRE(s.valid());
        setsockopt(static_cast<decltype(::socket(0, 0, 0))>(s.get()), SOL_SOCKET, SO_RCVBUF,
                   reinterpret_cast<const char*>(&receive_buffer), sizeof(receive_buffer));
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        REQUIRE(::connect(static_cast<decltype(::socket(0, 0, 0))>(s.get()), reinterpret_cast<const sockaddr*>(&addr),
                          sizeof(addr)) == 0);
        sock_ = std::move(s);
    }
    void send(const std::string& data) {
        auto st = det::send_all(sock_.get(), data, std::chrono::milliseconds(5000), nullptr);
        REQUIRE_MESSAGE(st.ok(), st.to_string());
    }
    // Reads until the peer closes (or `timeout`). Returns everything read.
    std::string read_all(std::chrono::milliseconds timeout = std::chrono::milliseconds(15000)) {
        std::string out;
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        char buf[8192];
        while (std::chrono::steady_clock::now() < deadline) {
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
            if (det::wait_socket(sock_.get(), false, left, nullptr) != det::WaitResult::ready) {
                break;
            }
            const long long n = det::recv_some(sock_.get(), buf, sizeof(buf));
            if (n == 0) {
                closed_ = true;
                break;
            }
            if (n == -1) {
                closed_ = true;
                break;
            }
            if (n > 0) out.append(buf, static_cast<std::size_t>(n));
        }
        return out;
    }
    // Reads until `needle` appears in the accumulated stream (or timeout).
    bool read_until(std::string& acc, const std::string& needle,
                    std::chrono::milliseconds timeout = std::chrono::milliseconds(10000)) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        char buf[8192];
        while (acc.find(needle) == std::string::npos) {
            const auto now = std::chrono::steady_clock::now();
            if (now >= deadline) return false;
            const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            if (det::wait_socket(sock_.get(), false, left, nullptr) != det::WaitResult::ready) return false;
            const long long n = det::recv_some(sock_.get(), buf, sizeof(buf));
            if (n == 0 || n == -1) {
                closed_ = true;
                return acc.find(needle) != std::string::npos;
            }
            if (n > 0) acc.append(buf, static_cast<std::size_t>(n));
        }
        return true;
    }
    // Shrinks the client's receive buffer so a peer that stops reading fills
    // the path quickly (backpressure tests).
    void shrink_receive_buffer(int bytes = 4096) {
        setsockopt(static_cast<decltype(::socket(0, 0, 0))>(sock_.get()), SOL_SOCKET, SO_RCVBUF,
                   reinterpret_cast<const char*>(&bytes), sizeof(bytes));
    }
    [[nodiscard]] bool closed() const { return closed_; }
    void close() { sock_.close(); }

private:
    det::Socket sock_;
    bool closed_ = false;
};

inline std::string build_request(const std::string& method, const std::string& target, std::uint16_t port,
                                 const std::vector<std::pair<std::string, std::string>>& headers = {},
                                 const std::optional<std::string>& body = std::nullopt, bool host_header = true) {
    std::string req = method + " " + target + " HTTP/1.1\r\n";
    if (host_header) {
        req += "Host: 127.0.0.1:" + std::to_string(port) + "\r\n";
    }
    for (const auto& [k, v] : headers) {
        req += k + ": " + v + "\r\n";
    }
    if (body) {
        req += "Content-Type: application/json\r\nContent-Length: " + std::to_string(body->size()) + "\r\n";
    }
    req += "\r\n";
    if (body) {
        req += *body;
    }
    return req;
}

inline Reply roundtrip(std::uint16_t port, const std::string& raw) {
    Conn c(port);
    c.send(raw);
    const std::string got = c.read_all();
    Reply r = parse_reply(got);
    r.complete = c.closed();
    return r;
}

inline Reply get(std::uint16_t port, const std::string& target,
                 const std::vector<std::pair<std::string, std::string>>& headers = {}) {
    return roundtrip(port, build_request("GET", target, port, headers));
}

inline Reply post(std::uint16_t port, const std::string& target, const std::string& body,
                  const std::vector<std::pair<std::string, std::string>>& headers = {}) {
    return roundtrip(port, build_request("POST", target, port, headers, body));
}

// Server with the mock backend on an ephemeral port; every event also lands
// in `events` (a MemoryTelemetrySink).
struct Fixture {
    std::shared_ptr<si::MemoryTelemetrySink> events = std::make_shared<si::MemoryTelemetrySink>();
    std::unique_ptr<srv::Server> server;
    std::uint16_t port = 0;

    static srv::ServerOptions defaults() {
        srv::ServerOptions o;
        o.host = "127.0.0.1";
        o.port = 0;
        o.backend.backend = "mock";
        o.models = {"mock:tiny"};
        o.heartbeat_interval = std::chrono::milliseconds(15000);
        o.shutdown_grace = std::chrono::milliseconds(2000);
        return o;
    }

    explicit Fixture(srv::ServerOptions o = defaults()) {
        o.extra_sinks.push_back(events);
        server = std::make_unique<srv::Server>(std::move(o));
        auto st = server->start();
        REQUIRE_MESSAGE(st.ok(), st.to_string());
        port = server->port();
    }

    // Parsed envelopes received so far (flushes the bus first).
    std::vector<si::json::Value> envelopes() {
        if (auto* e = server->engine()) {
            e->telemetry().flush();
        }
        std::vector<si::json::Value> out;
        for (const auto& line : events->lines()) {
            auto v = si::json::parse(line);
            if (v.ok()) out.push_back(v.value());
        }
        return out;
    }
    std::vector<si::json::Value> of_type(const std::string& type) {
        std::vector<si::json::Value> out;
        for (auto& e : envelopes()) {
            if (e.find("event_type")->as_string() == type) out.push_back(e);
        }
        return out;
    }
};

inline std::string chat_body(const std::string& extra = "", const std::string& content = "hello sonder") {
    return R"({"model":"default","messages":[{"role":"user","content":")" + content + R"("}])" + extra + "}";
}

// Polls `pred` until it holds or `timeout` elapses.
template <class Pred>
bool eventually(Pred pred, std::chrono::milliseconds timeout = std::chrono::milliseconds(5000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

}  // namespace server_test
