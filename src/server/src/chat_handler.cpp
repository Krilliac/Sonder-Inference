#include "chat_handler.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

#include "anthropic.hpp"
#include "request_path.hpp"
#include "test_hooks.hpp"

namespace sonder::inference::server::detail {

void execute_chat(ChatExchange& ex, const ChatJob& job, const Correlation& corr, const std::string& model,
                  const std::string& backend, const std::shared_ptr<Session>& session,
                  ChatProtocol protocol) {
    const bool anthropic = protocol == ChatProtocol::anthropic;
    RequestOptions ro;
    ro.request_id = make_id("req");
    ro.parent_request_id = corr.parent_request_id;
    ro.session_key = chat_session_key(job, corr);
    ro.thinking = job.thinking;
    ro.separate_reasoning = anthropic;
    const std::vector<std::string> warnings = apply_thinking_pins(ex.options, ro.thinking);
    const std::string completion_id = (anthropic ? "msg_" : "chatcmpl-") + *ro.request_id;
    // X-Sonder-Request-Id and the access log carry the engine request id.
    ex.request_id = *ro.request_id;
    const std::int64_t created_at = static_cast<std::int64_t>(std::time(nullptr));

    // Client disconnect cancels the request (request.cancelled).
    std::mutex watch_mu;
    std::condition_variable watch_cv;
    bool watch_done = false;
    std::atomic<bool> disconnected{false};
    std::thread watcher;
    // Stops and joins the watcher and leaves `inflight` on every exit,
    // including an exception out of the chat call: a joinable std::thread
    // destroyed during unwinding would call std::terminate.
    bool left_inflight = false;
    const auto end_request = [&] {
        {
            std::lock_guard<std::mutex> lock(watch_mu);
            watch_done = true;
        }
        watch_cv.notify_all();
        if (watcher.joinable()) {
            watcher.join();
        }
        if (!left_inflight) {
            left_inflight = true;
            ex.leave_inflight();
        }
    };
    struct EndRequestGuard {
        const decltype(end_request)& end;
        ~EndRequestGuard() { end(); }
    } end_guard{end_request};
    try {
        if (take_watcher_spawn_failure()) {
            throw std::system_error(std::make_error_code(std::errc::resource_unavailable_try_again),
                                    "test hook: watcher thread creation refused");
        }
        watcher = std::thread([&] {
            std::unique_lock<std::mutex> lock(watch_mu);
            while (!watch_done) {
                lock.unlock();
                if (peer_gone(ex.socket)) {
                    disconnected.store(true);
                    session->cancel();
                    return;
                }
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });
    } catch (const std::system_error&) {
        // The OS refused a thread (thread or memory limits). Nothing ran:
        // answer 503 for this request instead of taking the server down.
        end_request();
        session->close();
        ex.watcher_failure();
        ApiError e =
            make_error(503, "overloaded",
                       "sonder-inference cannot start a request thread right now; nothing was executed, "
                       "retry later");
        e.retry_after = 1;
        ex.send_error(e);
        return;
    }

    AnthropicStream messages_stream(completion_id, model);
    bool headers_sent = false;
    bool write_failed = false;
    const auto chunk_object = [&](json::Object delta, json::Value finish) {
        return json::Object{{"id", completion_id},
                            {"object", "chat.completion.chunk"},
                            {"created", created_at},
                            {"model", model},
                            {"choices", json::Array{json::Object{{"index", 0},
                                                                 {"delta", std::move(delta)},
                                                                 {"finish_reason", std::move(finish)}}}}};
    };
    const auto send_frame = [&](std::string_view frame) {
        if (!write_failed && !frame.empty() && !ex.write(frame)) {
            write_failed = true;
        }
        return !write_failed;
    };
    const auto send_event = [&](const json::Object& obj) {
        return send_frame("data: " + json::Value(obj).dump() + "\n\n");
    };
    const auto ensure_stream_head = [&]() {
        if (headers_sent) {
            return !write_failed;
        }
        headers_sent = true;
        if (!ex.start_stream()) {
            write_failed = true;
            return false;
        }
        if (anthropic) {
            return send_frame(messages_stream.start());
        }
        return send_event(chunk_object(json::Object{{"role", "assistant"}, {"content", ""}}, nullptr));
    };

    TokenCallback on_chunk;
    if (job.stream) {
        on_chunk = [&](const TokenChunk& chunk) {
            if (!ensure_stream_head() ||
                !(anthropic ? send_frame(messages_stream.chunk(chunk))
                            : send_event(chunk_object(json::Object{{"content", std::string(chunk.text)}},
                                                      nullptr)))) {
                session->cancel();
                return false;
            }
            return true;
        };
    }

    auto result = session->chat(job.messages, on_chunk, job.sampling, ro);
    end_request();
    const bool rejected = session->last_scheduler_rejected();

    if (disconnected.load() || write_failed) {
        ex.status = 499; // client closed the request (access log only)
        session->close();
        return;
    }
    if (!result.ok()) {
        session->close();
        const ApiError e = map_session_failure(result.status(), rejected);
        if (headers_sent) {
            if (anthropic) {
                (void)send_frame("event: error\ndata: " + json::Value(anthropic_error_body(e)).dump() +
                                 "\n\n");
            } else {
                (void)send_event(error_body(e));
            }
        } else {
            ex.send_error(e);
        }
        return;
    }
    const GenerationResult& r = result.value();
    json::Object meta{
        {"api_version", kApiVersion},  {"request_id", r.request_id},
        {"session_id", session->id()}, {"backend", backend},
        {"synthetic", ex.synthetic},   {"token_counts_from_backend", r.stats.token_counts_from_backend}};
    if (!warnings.empty()) {
        meta.set("warnings", json::Array(warnings.begin(), warnings.end()));
    }
    session->close();
    if (!job.stream) {
        if (anthropic) {
            json::Object doc = anthropic_message_response(completion_id, model, r, warnings);
            doc.set("sonder", std::move(meta));
            ex.send_json(200, doc);
            return;
        }
        json::Object doc{
            {"id", completion_id},
            {"object", "chat.completion"},
            {"created", created_at},
            {"model", model},
            {"choices",
             json::Array{json::Object{{"index", 0},
                                      {"message", json::Object{{"role", "assistant"}, {"content", r.text}}},
                                      {"finish_reason", finish_reason(r)}}}},
            {"usage", usage_json(r)},
            {"timings", timings_json(r)},
            {"sonder", std::move(meta)}};
        ex.send_json(200, doc);
        return;
    }
    if (!ensure_stream_head()) {
        return;
    }
    if (anthropic) {
        (void)send_frame(messages_stream.finish(r, warnings, std::move(meta)));
        return;
    }
    json::Object last = chunk_object(json::Object{}, finish_reason(r));
    if (job.include_usage) {
        last.set("usage", usage_json(r));
    }
    last.set("timings", timings_json(r));
    last.set("sonder", std::move(meta));
    if (send_event(last)) {
        (void)ex.write("data: [DONE]\n\n");
    }
}

} // namespace sonder::inference::server::detail
