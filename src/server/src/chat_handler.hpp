// Internal: shared synchronous chat lifecycle and streaming transport for
// the OpenAI and Anthropic endpoints. Server owns sockets and admission;
// this layer owns cancellation, callbacks and protocol response framing.
#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include "openai.hpp"
#include "priority_admission.hpp"
#include "socket.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

enum class ChatProtocol { openai, anthropic };

// Callbacks execute synchronously on the connection thread except when
// SseStreamWriter serializes periodic `: keep-alive\n\n` heartbeats from the
// request watcher thread while a streaming request waits in prefill/idle.
struct ChatExchange {
    const ServerOptions& options;
    native_socket socket;
    std::string& request_id;
    int& status;
    bool synthetic;
    std::chrono::steady_clock::time_point started;
    PriorityAdmission& admission;
    const std::atomic<bool>& hard_stop;
    TelemetryBus& telemetry;
    std::function<void()> leave_inflight;
    std::function<bool(std::string_view)> write;
    std::function<bool()> start_stream;
    std::function<void(int, const json::Object&)> send_json;
    std::function<void(const ApiError&)> send_error;
    std::function<void()> watcher_failure;
};

// Thread-safe SSE transport coordinator shared by `/v1/chat/completions`,
// `/v1/messages`, `/v1/completions`, and `/v1/responses`. Serializes socket
// writes between the connection thread and the disconnect watcher so
// `: keep-alive\n\n` heartbeats during long prefill waits never interleave
// with HTTP headers, protocol preambles, or data frames.
class SseStreamWriter {
public:
    using Clock = std::chrono::steady_clock;

    SseStreamWriter(ChatExchange& ex, std::chrono::milliseconds keepalive_interval)
        : ex_(ex), keepalive_interval_(keepalive_interval), last_activity_(Clock::now()) {}

    void arm_keepalive() {
        std::lock_guard<std::mutex> lock(mu_);
        if (disarmed_) return;
        armed_ = true;
        last_activity_ = Clock::now();
    }

    void disarm_keepalive() {
        std::lock_guard<std::mutex> lock(mu_);
        disarmed_ = true;
        armed_ = false;
    }

    template <class MakePreamble>
    bool ensure_started(MakePreamble&& make_preamble) {
        std::lock_guard<std::mutex> lock(mu_);
        if (write_failed_) {
            return false;
        }
        if (!headers_sent_) {
            headers_sent_ = true;
            if (!ex_.start_stream()) {
                write_failed_ = true;
                return false;
            }
            last_activity_ = Clock::now();
        }
        if (!preamble_sent_) {
            preamble_sent_ = true;
            const std::string preamble = make_preamble();
            if (!preamble.empty()) {
                if (!ex_.write(preamble)) {
                    write_failed_ = true;
                    return false;
                }
                last_activity_ = Clock::now();
            }
        }
        return true;
    }

    bool write_frame(std::string_view frame) {
        std::lock_guard<std::mutex> lock(mu_);
        if (write_failed_) {
            return false;
        }
        if (!frame.empty()) {
            if (!ex_.write(frame)) {
                write_failed_ = true;
                return false;
            }
            last_activity_ = Clock::now();
        }
        return true;
    }

    bool maybe_send_keepalive() {
        if (keepalive_interval_.count() <= 0) {
            return true;
        }
        std::lock_guard<std::mutex> lock(mu_);
        if (!armed_ || disarmed_ || write_failed_) {
            return true;
        }
        if (Clock::now() - last_activity_ < keepalive_interval_) {
            return true;
        }
        if (!headers_sent_) {
            headers_sent_ = true;
            if (!ex_.start_stream()) {
                write_failed_ = true;
                return false;
            }
        }
        if (!ex_.write(": keep-alive\n\n")) {
            write_failed_ = true;
            return false;
        }
        last_activity_ = Clock::now();
        return true;
    }

    [[nodiscard]] bool headers_sent() const {
        std::lock_guard<std::mutex> lock(mu_);
        return headers_sent_;
    }

    [[nodiscard]] bool write_failed() const {
        std::lock_guard<std::mutex> lock(mu_);
        return write_failed_;
    }

private:
    ChatExchange& ex_;
    std::chrono::milliseconds keepalive_interval_;
    mutable std::mutex mu_;
    Clock::time_point last_activity_;
    bool armed_ = false;
    bool disarmed_ = false;
    bool headers_sent_ = false;
    bool preamble_sent_ = false;
    bool write_failed_ = false;
};

// The caller holds the model's residency pin through this synchronous call
// (including response writes), and destroys the session before releasing it.
void execute_chat(ChatExchange& exchange, const ChatJob& job, const Correlation& correlation,
                  const std::string& model, const std::string& backend,
                  const std::shared_ptr<Session>& session, ChatProtocol protocol,
                  bool backend_native_tools = true);

// Synchronous execution for POST /v1/completions (non-streaming and SSE streaming).
void execute_completion(ChatExchange& exchange, const CompletionJob& job, const Correlation& correlation,
                        const std::string& model, const std::string& backend,
                        const std::shared_ptr<Session>& session);

// Synchronous execution for POST /v1/embeddings.
void execute_embeddings(ChatExchange& exchange, const EmbedJob& job, const Correlation& correlation,
                        const std::string& model, const std::string& backend,
                        const std::shared_ptr<Session>& session);

} // namespace sonder::inference::server::detail
