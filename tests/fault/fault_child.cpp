// Fake upstream server used as a REAL child process by the fault-injection
// tests (test_fault_injection.cpp). It is not a model: it speaks just enough
// HTTP to look like a llama-server to the supervisor (GET /health, a streamed
// POST /v1/chat/completions) and can be told to misbehave in specific ways.
//
// Every observable step is appended to an events file ("name pid=N k=v ...")
// so a test can prove the injected fault actually fired, not just that the
// supervisor survived. The supervisor appends "--host 127.0.0.1 --port N".
//
// Options (all optional):
//   --events FILE            event log (append, one open/write/close per line)
//   --once-marker FILE       die-* faults fire only while FILE does not exist,
//                            and create it when they fire (so the restarted
//                            child behaves)
//   --warmup-ms N            /health answers 503 for N ms after start
//   --die-warmup-ms N        die (exit 9) N ms after start while still warming
//   --exit-at-start CODE     exit with CODE immediately, before listening
//   --first-byte-delay-ms N  delay before the response headers (fake prefill)
//   --delay-requests K       only the first K chat requests are delayed (1)
//   --chunks N               SSE events per chat reply (5)
//   --chunk-interval-ms N    pause between events (5)
//   --die-after-chunks N     die (exit 7) after sending event N mid-stream
//   --stall-after-chunks N   after sending event N, stop sending (no terminator,
//                            connection held open) until the client leaves or
//                            --stall-hold-ms expires; fires once like die-*
//   --stall-hold-ms N        longest a stall is held (30000)
//   --flood-timeout-ms N     send timeout for POST /flood (300)
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>

#include "fault_sock.hpp"

#if defined(_WIN32)
#include <process.h>
#else
#include <csignal>
#endif

namespace {
using namespace std::chrono;
using fault_net::Sock;

struct Config {
    std::string events;
    std::string marker;
    int warmup_ms = 0;
    int die_warmup_ms = 0;
    int exit_at_start = 0;
    int first_byte_delay_ms = 0;
    int delay_requests = 1;
    int chunks = 5;
    int chunk_interval_ms = 5;
    int die_after_chunks = 0;
    int stall_after_chunks = 0;
    int stall_hold_ms = 30000;
    int flood_timeout_ms = 300;
    int port = 0;
};

Config g_cfg;
steady_clock::time_point g_started;
std::mutex g_log_mutex;
std::atomic<int> g_chat_requests{0};
std::atomic<bool> g_health_503_seen{false};
std::atomic<bool> g_ready_seen{false};

long current_pid() {
#if defined(_WIN32)
    return static_cast<long>(_getpid());
#else
    return static_cast<long>(::getpid());
#endif
}

void log_event(const std::string &name, const std::string &fields = {}) {
    if (g_cfg.events.empty())
        return;
    std::string line = name + " pid=" + std::to_string(current_pid());
    if (!fields.empty())
        line += " " + fields;
    line += "\n";
    std::lock_guard<std::mutex> lock(g_log_mutex);
    std::FILE *file = std::fopen(g_cfg.events.c_str(), "ab");
    if (!file)
        return;
    (void)std::fwrite(line.data(), 1, line.size(), file);
    (void)std::fclose(file);
}

// True when a die-* fault should fire now. Creates the marker so it fires once.
bool arm_die_fault() {
    if (g_cfg.marker.empty())
        return true;
    {
        std::ifstream exists(g_cfg.marker);
        if (exists.good())
            return false;
    }
    std::ofstream create(g_cfg.marker);
    create << "fired\n";
    return true;
}

bool parse_args(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc)
            return false;
        const std::string value = argv[++i];
        if (arg == "--host") continue;
        else if (arg == "--events") g_cfg.events = value;
        else if (arg == "--once-marker") g_cfg.marker = value;
        else if (arg == "--warmup-ms") g_cfg.warmup_ms = std::atoi(value.c_str());
        else if (arg == "--die-warmup-ms") g_cfg.die_warmup_ms = std::atoi(value.c_str());
        else if (arg == "--exit-at-start") g_cfg.exit_at_start = std::atoi(value.c_str());
        else if (arg == "--first-byte-delay-ms") g_cfg.first_byte_delay_ms = std::atoi(value.c_str());
        else if (arg == "--delay-requests") g_cfg.delay_requests = std::atoi(value.c_str());
        else if (arg == "--chunks") g_cfg.chunks = std::atoi(value.c_str());
        else if (arg == "--chunk-interval-ms") g_cfg.chunk_interval_ms = std::atoi(value.c_str());
        else if (arg == "--die-after-chunks") g_cfg.die_after_chunks = std::atoi(value.c_str());
        else if (arg == "--stall-after-chunks") g_cfg.stall_after_chunks = std::atoi(value.c_str());
        else if (arg == "--stall-hold-ms") g_cfg.stall_hold_ms = std::atoi(value.c_str());
        else if (arg == "--flood-timeout-ms") g_cfg.flood_timeout_ms = std::atoi(value.c_str());
        else if (arg == "--port") g_cfg.port = std::atoi(value.c_str());
        else return false;
    }
    return g_cfg.port > 0 && g_cfg.port < 65536;
}

struct Request {
    std::string method;
    std::string path;
    std::size_t body_bytes = 0;
};

bool read_request(Sock s, Request &out) {
    std::string buffer;
    char tmp[4096];
    std::size_t header_end = std::string::npos;
    while ((header_end = buffer.find("\r\n\r\n")) == std::string::npos) {
        if (buffer.size() > 65536)
            return false;
        const long n = fault_net::recv_some(s, tmp, sizeof(tmp));
        if (n <= 0)
            return false;
        buffer.append(tmp, static_cast<std::size_t>(n));
    }
    const auto line_end = buffer.find("\r\n");
    const std::string first = buffer.substr(0, line_end);
    const auto sp1 = first.find(' ');
    if (sp1 == std::string::npos)
        return false;
    const auto sp2 = first.find(' ', sp1 + 1);
    if (sp2 == std::string::npos)
        return false;
    out.method = first.substr(0, sp1);
    out.path = first.substr(sp1 + 1, sp2 - sp1 - 1);
    std::string lowered = buffer.substr(0, header_end);
    for (auto &c : lowered)
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
    std::size_t length = 0;
    const auto cl = lowered.find("content-length:");
    if (cl != std::string::npos)
        length = static_cast<std::size_t>(std::strtoull(lowered.c_str() + cl + 15, nullptr, 10));
    std::size_t have = buffer.size() - (header_end + 4);
    while (have < length) {
        const long n = fault_net::recv_some(s, tmp, sizeof(tmp));
        if (n <= 0)
            return false;
        have += static_cast<std::size_t>(n);
    }
    out.body_bytes = length;
    return true;
}

void send_text(Sock s, const std::string &text) { (void)fault_net::send_all(s, text.data(), text.size()); }

bool send_chunk(Sock s, const std::string &payload) {
    char size[32];
    std::snprintf(size, sizeof(size), "%zx\r\n", payload.size());
    const std::string framed = std::string(size) + payload + "\r\n";
    return fault_net::send_all(s, framed.data(), framed.size()) == framed.size();
}

void handle_health(Sock s) {
    const bool warm = duration_cast<milliseconds>(steady_clock::now() - g_started).count() >= g_cfg.warmup_ms;
    if (!warm) {
        if (!g_health_503_seen.exchange(true))
            log_event("health_503");
        const std::string body = "{\"error\":\"Loading model\"}";
        send_text(s, "HTTP/1.1 503 Service Unavailable\r\nContent-Type: application/json\r\nContent-Length: " +
                         std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body);
        return;
    }
    if (!g_ready_seen.exchange(true))
        log_event("ready");
    const std::string body = "{\"status\":\"ok\"}";
    send_text(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: " +
                     std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body);
}

void handle_chat(Sock s, std::size_t body_bytes) {
    const int index = g_chat_requests.fetch_add(1);
    log_event("request_received", "n=" + std::to_string(index) + " body_bytes=" + std::to_string(body_bytes));
    if (index < g_cfg.delay_requests && g_cfg.first_byte_delay_ms > 0) {
        // Fake long prefill: no bytes go out. Watch for the client leaving.
        const auto begin = steady_clock::now();
        while (duration_cast<milliseconds>(steady_clock::now() - begin).count() < g_cfg.first_byte_delay_ms) {
            if (fault_net::peer_closed(s)) {
                log_event("client_gone",
                          "phase=prefill waited_ms=" +
                              std::to_string(duration_cast<milliseconds>(steady_clock::now() - begin).count()));
                return;
            }
            std::this_thread::sleep_for(milliseconds(10));
        }
    }
    send_text(s, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n"
                 "Connection: close\r\n\r\n");
    log_event("first_byte_sent");
    for (int i = 1; i <= g_cfg.chunks; ++i) {
        if (!send_chunk(s, "data: {\"i\":" + std::to_string(i) + ",\"tok\":\"t" + std::to_string(i) + "\"}\n\n")) {
            log_event("client_gone", "phase=stream chunks_sent=" + std::to_string(i - 1));
            return;
        }
        if (g_cfg.die_after_chunks == i && arm_die_fault()) {
            log_event("die_midstream", "after_chunks=" + std::to_string(i));
            // Let the client drain what is already queued before the abrupt
            // exit so an RST cannot discard it.
            std::this_thread::sleep_for(milliseconds(100));
            std::_Exit(7);
        }
        if (g_cfg.stall_after_chunks == i && arm_die_fault()) {
            log_event("stall_midstream", "after_chunks=" + std::to_string(i));
            const auto begin = steady_clock::now();
            for (;;) {
                const auto waited = duration_cast<milliseconds>(steady_clock::now() - begin).count();
                if (fault_net::peer_closed(s)) {
                    log_event("client_gone", "phase=stalled waited_ms=" + std::to_string(waited));
                    return;
                }
                if (waited >= g_cfg.stall_hold_ms) {
                    log_event("stall_released", "waited_ms=" + std::to_string(waited));
                    return;
                }
                std::this_thread::sleep_for(milliseconds(10));
            }
        }
        std::this_thread::sleep_for(milliseconds(g_cfg.chunk_interval_ms));
    }
    if (!send_chunk(s, "data: [DONE]\n\n") || fault_net::send_all(s, "0\r\n\r\n", 5) != 5) {
        log_event("client_gone", "phase=stream_end");
        return;
    }
    log_event("stream_complete", "chunks=" + std::to_string(g_cfg.chunks));
}

// Writes an unbounded body so a client that never reads exhausts the socket
// buffers. The send timeout turns "client stalled" into a recorded event.
void handle_flood(Sock s) {
    (void)fault_net::set_int_opt(s, SOL_SOCKET, SO_SNDBUF, 4096);
    (void)fault_net::set_send_timeout_ms(s, g_cfg.flood_timeout_ms);
    log_event("request_received", "n=flood");
    send_text(s, "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nConnection: close\r\n\r\n");
    const std::string block(16384, 'x');
    constexpr std::size_t kCap = 64u * 1024u * 1024u;
    std::size_t sent = 0;
    while (sent < kCap) {
        const std::size_t n = fault_net::send_all(s, block.data(), block.size());
        sent += n;
        if (n != block.size()) {
            log_event("write_timeout", "bytes_sent=" + std::to_string(sent));
            return;
        }
    }
    log_event("flood_cap_reached", "bytes_sent=" + std::to_string(sent));
}

void serve_connection(Sock s) {
    Request request;
    if (read_request(s, request)) {
        if (request.method == "GET" && request.path == "/health")
            handle_health(s);
        else if (request.method == "POST" && request.path == "/flood")
            handle_flood(s);
        else if (request.method == "POST" && request.path == "/v1/chat/completions")
            handle_chat(s, request.body_bytes);
        else
            send_text(s, "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
    }
    fault_net::close_sock(s);
}
} // namespace

int main(int argc, char **argv) {
#if !defined(_WIN32)
    std::signal(SIGPIPE, SIG_IGN);
#endif
    g_started = steady_clock::now();
    if (!parse_args(argc, argv))
        return 64;
    log_event("start");
    if (g_cfg.exit_at_start != 0) {
        log_event("exit_at_start", "code=" + std::to_string(g_cfg.exit_at_start));
        std::_Exit(g_cfg.exit_at_start);
    }
    const Sock listener = fault_net::listen_loopback(static_cast<std::uint16_t>(g_cfg.port));
    if (listener == fault_net::kBad) {
        log_event("bind_failed", "port=" + std::to_string(g_cfg.port));
        return 3;
    }
    log_event("listening", "port=" + std::to_string(g_cfg.port));
    if (g_cfg.die_warmup_ms > 0) {
        std::thread([] {
            std::this_thread::sleep_for(milliseconds(g_cfg.die_warmup_ms));
            if (!g_ready_seen.load() && arm_die_fault()) {
                log_event("die_warmup", "after_ms=" + std::to_string(g_cfg.die_warmup_ms));
                std::_Exit(9);
            }
        }).detach();
    }
    for (;;) {
        const Sock s = fault_net::accept_conn(listener);
        if (s == fault_net::kBad)
            continue;
        std::thread(serve_connection, s).detach();
    }
}
