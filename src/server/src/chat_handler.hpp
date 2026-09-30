// Internal: shared synchronous chat lifecycle and streaming transport for
// the OpenAI and Anthropic endpoints. Server owns sockets and admission;
// this layer owns cancellation, callbacks and protocol response framing.
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "openai.hpp"
#include "socket.hpp"
#include "sonder/inference/server.hpp"

namespace sonder::inference::server::detail {

enum class ChatProtocol { openai, anthropic };

// Callbacks execute synchronously on the connection thread. The disconnect
// watcher only reads socket and cancels the session, never writes responses.
struct ChatExchange {
    const ServerOptions& options;
    native_socket socket;
    std::string& request_id;
    int& status;
    bool synthetic;
    std::function<void()> leave_inflight;
    std::function<bool(std::string_view)> write;
    std::function<bool()> start_stream;
    std::function<void(int, const json::Object&)> send_json;
    std::function<void(const ApiError&)> send_error;
    std::function<void()> watcher_failure;
};

void execute_chat(ChatExchange& exchange, const ChatJob& job, const Correlation& correlation,
                  const std::string& model, const std::string& backend,
                  const std::shared_ptr<Session>& session, ChatProtocol protocol);

} // namespace sonder::inference::server::detail
