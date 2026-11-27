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
#include "reasoning_budget.hpp"
#include "test_hooks.hpp"
#include "tool_call_parser.hpp"

namespace sonder::inference::server::detail {

namespace {

json::Object openai_tool_call_json(const ToolCall& tc) {
    return json::Object{{"id", tc.id},
                        {"type", "function"},
                        {"function", json::Object{{"name", tc.name},
                                                  {"arguments", tc.arguments_json.empty() ? "{}" : tc.arguments_json}}}};
}

json::Object openai_tool_call_delta_json(std::uint32_t index, const std::string& id, const std::string& name,
                                         const std::string& args_delta) {
    json::Object fn;
    if (!name.empty()) fn.set("name", name);
    fn.set("arguments", args_delta);
    json::Object entry{{"index", static_cast<std::int64_t>(index)}, {"function", std::move(fn)}};
    if (!id.empty()) {
        entry.set("id", id);
        entry.set("type", "function");
    } else if (!name.empty()) {
        entry.set("type", "function");
    }
    return entry;
}

std::string bounded_model_label(std::string_view text) {
    if (text.size() > kMaxEvidenceStringBytes) {
        std::size_t n = kMaxEvidenceStringBytes;
        while (n > 0 && (static_cast<unsigned char>(text[n]) & 0xC0u) == 0x80u) {
            --n;
        }
        text = text.substr(0, n);
    }
    return sanitize_bounded_evidence_string(text, kMaxEvidenceStringBytes);
}

}  // namespace

void emit_pre_execution_failure(TelemetryBus& bus, TelemetryContext ctx, std::string_view kind,
                                std::string_view error_code, const Correlation& corr,
                                std::optional<RequestPriority> job_priority,
                                std::string_view resolved_model,
                                std::string_view requested_model,
                                bool cancelled,
                                std::optional<double> admission_queue_ms) {
    if (corr.run_id) ctx.run_id = corr.run_id;
    if (corr.agent_id) ctx.agent_id = corr.agent_id;
    if (corr.task_id) ctx.task_id = corr.task_id;
    const auto prio = corr.priority_class ? corr.priority_class : job_priority;
    json::Object attrs{{"kind", std::string(kind)},
                       {"outcome", cancelled ? "cancelled" : "failed"}};
    if (!resolved_model.empty()) {
        attrs.set("model", bounded_model_label(resolved_model));
    }
    if (!requested_model.empty()) {
        attrs.set("requested_model", bounded_model_label(requested_model));
    }
    if (prio) {
        attrs.set("priority_class", to_string(*prio));
        if (admission_queue_ms.has_value()) {
            attrs.set("admission", json::Object{{"priority", to_string(*prio)},
                                                {"queue_ms", *admission_queue_ms}});
        }
    }
    attrs.set("error_code", std::string(error_code));
    attrs.set("error", std::string(error_code));
    attrs.set("backend_started", false);
    if (corr.parent_request_id) {
        attrs.set("parent_request_id", *corr.parent_request_id);
    }
    bus.emit(cancelled ? "request.cancelled" : "request.failed", std::move(ctx), std::move(attrs),
             TelemetryLevel::metrics);
}

std::optional<ApiError> prepare_chat_grammar(
    ChatJob& job, std::string_view tokenizer_identity,
    bool backend_supports_structured_output, bool backend_native_tools,
    std::string_view backend_name, CompiledGrammarCache* cache,
    const StructuredGrammarEngine& engine) {
    auto resolved = resolve_structured_grammar(
        "response_format.json_schema.schema", job.grammar, job.response_format_type,
        job.pending_response_schema, job.response_format_schema, job.tools, job.tool_choice,
        backend_supports_structured_output, backend_native_tools, backend_name, tokenizer_identity,
        cache, engine);
    if (!resolved.ok()) {
        const std::string code = resolved.status().code() == ErrorCode::unsupported
                                     ? "unsupported_parameter"
                                     : "invalid_request";
        const bool from_tools = !job.response_format_type.has_value() && job.grammar.empty() && !job.tools.empty();
        const std::optional<std::string> param =
            from_tools
                ? std::optional<std::string>("tools")
                : (job.response_format_type && *job.response_format_type == "json_schema"
                       ? std::optional<std::string>("response_format.json_schema.schema")
                       : (job.response_format_type ? std::optional<std::string>("response_format") : std::nullopt));
        const std::string msg =
            from_tools ? resolved.status().message() : ("invalid response_format: " + resolved.status().message());
        return make_error(400, code, msg, param);
    }
    job.compiled_grammar = std::move(resolved.value().compiled);
    job.grammar = std::move(resolved.value().grammar);
    return std::nullopt;
}

void execute_chat(ChatExchange& ex, const ChatJob& job, const Correlation& corr, const std::string& model,
                  const std::string& backend, const std::shared_ptr<Session>& session,
                  ChatProtocol protocol, bool backend_native_tools,
                  CompiledGrammarCache* grammar_cache, std::string_view tokenizer_identity,
                  bool backend_supports_structured_output) {
    const bool anthropic = protocol == ChatProtocol::anthropic;
    const RequestPriority priority = corr.priority_class.value_or(
        job.priority_class.value_or(RequestPriority::interactive));
    const auto deadline_ms = corr.deadline_ms ? corr.deadline_ms : job.deadline_ms;
    const auto deadline = deadline_ms
        ? ex.started + std::chrono::milliseconds(static_cast<std::int64_t>(*deadline_ms))
        : std::chrono::steady_clock::time_point::max();
    RequestOptions ro;
    ro.request_id = make_id("req");
    ro.parent_request_id = corr.parent_request_id;
    ro.session_key = chat_session_key(job, corr);
    ro.thinking = job.thinking;
    // Both protocols ask the backend to split reasoning from visible text.
    // Anthropic renders it as a thinking block, OpenAI as reasoning_content
    // (llama-server's convention); neither adds a field when it is empty.
    ro.separate_reasoning = true;
    ro.retain_text = !job.stream;
    if (ex.request_id.rfind("req-", 0) == 0) ro.request_id = ex.request_id;
    ex.request_id = *ro.request_id;
    ro.served_model = model;
    ro.requested_model = job.model;
    std::shared_ptr<const CompiledGrammar> compiled_grammar = job.compiled_grammar;
    std::string fallback_grammar = job.grammar;
    if (!compiled_grammar && grammar_cache != nullptr &&
        (job.response_format_type.has_value() || (!job.tools.empty() && job.tool_choice))) {
        const std::string fallback_tok =
            tokenizer_identity.empty() ? build_tokenizer_identity_key(backend, model) : std::string(tokenizer_identity);
        auto resolved = resolve_structured_grammar(
            "response_format.json_schema.schema", job.grammar, job.response_format_type,
            job.pending_response_schema, job.response_format_schema, job.tools, job.tool_choice,
            backend_supports_structured_output, backend_native_tools, backend, fallback_tok, grammar_cache);
        if (!resolved.ok()) {
            const std::string code = resolved.status().code() == ErrorCode::unsupported
                                         ? "unsupported_parameter"
                                         : "invalid_request";
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "chat", code, corr,
                                       priority, model, job.model);
            ex.leave_inflight();
            session->close();
            const bool from_tools =
                !job.response_format_type.has_value() && job.grammar.empty() && !job.tools.empty();
            ex.send_error(make_error(400, code, resolved.status().message(),
                                     from_tools ? "tools" : "response_format"));
            return;
        }
        compiled_grammar = std::move(resolved.value().compiled);
        fallback_grammar = std::move(resolved.value().grammar);
    }
    auto matcher_res = bind_request_grammar_matcher(compiled_grammar, ro.grammar, fallback_grammar);
    if (!matcher_res.ok()) {
        const std::string code = matcher_res.status().code() == ErrorCode::unsupported
                                     ? "unsupported_parameter"
                                     : "invalid_request";
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "chat", code, corr,
                                   priority, model, job.model);
        ex.leave_inflight();
        session->close();
        ex.send_error(make_error(400, code, matcher_res.status().message(), "response_format"));
        return;
    }
    std::unique_ptr<GrammarMatcher> grammar_matcher = std::move(matcher_res).value();
    ro.response_format_type = job.response_format_type;
    ro.response_format_schema = job.response_format_schema;
    ro.tools = job.tools;
    ro.tool_choice = job.tool_choice;
    ro.parallel_tool_calls = job.parallel_tool_calls;
    std::vector<std::string> warnings = apply_thinking_pins(ex.options, ro.thinking);
    reasoning_budget::apply(job, corr, ex.options, backend, ro, warnings);
    ro.priority_class = priority;
    ro.preserve_numeric_priority = corr.numeric_priority;
    if (deadline_ms) ro.deadline = deadline;
    std::atomic<bool> backend_started{false};
    const std::string completion_id = (anthropic ? "msg_" : "chatcmpl-") + *ro.request_id;
    // X-Sonder-Request-Id and the access log carry the engine request id.
    ex.request_id = *ro.request_id;
    const std::int64_t created_at = static_cast<std::int64_t>(std::time(nullptr));

    // Client disconnect cancels the request (request.cancelled).
    SseStreamWriter sse(ex, ex.options.stream_keepalive_interval);
    std::mutex watch_mu;
    std::condition_variable watch_cv;
    bool watch_done = false;
    std::atomic<bool> disconnected{false};
    ro.on_backend_start = [&] {
        if (disconnected.load() || ex.hard_stop.load() || std::chrono::steady_clock::now() >= deadline) {
            session->cancel();
            return;
        }
        backend_started.store(true);
        if (job.stream) {
            sse.arm_keepalive();
        }
    };
    std::shared_ptr<PriorityAdmission::Ticket> ticket;
    std::thread watcher;
    // Stops and joins the watcher and leaves `inflight` on every exit,
    // including an exception out of the chat call: a joinable std::thread
    // destroyed during unwinding would call std::terminate.
    bool left_inflight = false;
    const auto end_request = [&] {
        sse.disarm_keepalive();
        if (ticket) ticket->release();
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
                }
                const auto now = std::chrono::steady_clock::now();
                if (now >= deadline) {
                    // Only cancel via the watcher once the backend has started; pre-start
                    // queue/scheduler waits enforce ro.deadline directly so queued deadline
                    // expirations report ErrorCode::timeout (request.failed) instead of
                    // racing into request.cancelled.
                    if (backend_started.load()) {
                        session->cancel();
                    }
                } else if (job.stream && !disconnected.load() && !ex.hard_stop.load() &&
                           !sse.maybe_send_keepalive()) {
                    disconnected.store(true);
                    session->cancel();
                }
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });
    } catch (const std::system_error&) {
        // The OS refused a thread (thread or memory limits). Nothing ran:
        // answer 503 for this request instead of taking the server down.
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "chat", "overloaded",
                                   corr, priority, model, job.model);
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

    if (priority_admission_enabled(ex.options)) {
        ticket = ex.admission.enqueue(priority, [&] {
            return disconnected.load() || session->state() == SessionState::closed || ex.hard_stop.load();
        }, deadline);
        if (ticket->outcome() != PriorityAdmission::Outcome::admitted) {
            const char* err_code = ticket->outcome() == PriorityAdmission::Outcome::full
                                       ? "overloaded"
                                       : ticket->outcome() == PriorityAdmission::Outcome::expired
                                       ? "deadline_exceeded" : "cancelled";
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "chat", err_code,
                                       corr, priority, model, job.model, disconnected.load(), ticket->queue_ms());
            end_request();
            if (disconnected.load()) {
                ex.status = 499;
            } else {
                ApiError error = ticket->outcome() == PriorityAdmission::Outcome::expired
                    ? make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed")
                    : ticket->outcome() == PriorityAdmission::Outcome::full
                    ? make_error(429, "overloaded", "request priority queue is full; nothing was executed")
                    : make_error(503, "not_ready", "request cancelled while queued; nothing was executed");
                if (error.status == 429) error.retry_after = 1;
                ex.send_error(error);
            }
            session->close();
            return;
        }
        ro.admission = ticket;
    }

    AnthropicStream messages_stream(completion_id, model);
    const auto chunk_object = [&](json::Object delta, json::Value finish, json::Value logprobs_val = nullptr) {
        json::Object choice{{"index", 0}, {"delta", std::move(delta)}, {"finish_reason", std::move(finish)}};
        if (job.sampling.logprobs) {
            choice.set("logprobs", std::move(logprobs_val));
        }
        return json::Object{{"id", completion_id},
                            {"object", "chat.completion.chunk"},
                            {"created", created_at},
                            {"model", model},
                            {"choices", json::Array{std::move(choice)}}};
    };
    const auto send_frame = [&](std::string_view frame) { return sse.write_frame(frame); };
    const auto send_event = [&](const json::Object& obj) {
        return sse.write_frame("data: " + json::Value(obj).dump() + "\n\n");
    };
    const auto ensure_stream_head = [&]() {
        return sse.ensure_started([&]() -> std::string {
            if (anthropic) {
                return messages_stream.start();
            }
            return "data: " +
                   json::Value(chunk_object(json::Object{{"role", "assistant"}, {"content", ""}}, nullptr)).dump() +
                   "\n\n";
        });
    };

    const bool tool_choice_none = job.tool_choice && job.tool_choice->mode == ToolChoiceMode::none;
    const bool tools_enabled = !job.tools.empty() && !backend_native_tools && !tool_choice_none;
    const bool allow_parallel = job.parallel_tool_calls.value_or(true);
    ToolCallStreamParser tool_parser("call_" + *ro.request_id + "_", allow_parallel);
    std::vector<bool> seen_native_tool_indices;
    std::optional<std::uint32_t> single_native_tool_index;
    bool saw_text = false;
    bool emitted_tool_calls = false;

    const auto emit_parsed_tool = [&](const ParsedToolEvent& ev) -> bool {
        emitted_tool_calls = true;
        if (!ensure_stream_head()) return false;
        if (anthropic) {
            return send_frame(messages_stream.tool_call(ToolCall{ev.id, ev.name, ev.arguments_json}));
        }
        json::Array tcs;
        tcs.emplace_back(openai_tool_call_delta_json(ev.index, ev.id, ev.name, ev.arguments_json));
        return send_event(chunk_object(json::Object{{"tool_calls", std::move(tcs)}}, nullptr));
    };

    TokenCallback on_chunk;
    if (job.stream) {
        on_chunk = [&](const TokenChunk& chunk) {
            if (grammar_matcher && !chunk.text.empty()) {
                (void)grammar_matcher->accept_chunk(chunk.text);
            }
            // 1. Native upstream tool call deltas (e.g. llama-server, Ollama).
            if (!chunk.tool_call_deltas.empty()) {
                std::vector<const ToolCallDelta*> accepted;
                accepted.reserve(chunk.tool_call_deltas.size());
                for (const auto& tcd : chunk.tool_call_deltas) {
                    if (!allow_parallel) {
                        if (!single_native_tool_index) {
                            single_native_tool_index = tcd.index;
                        } else if (tcd.index != *single_native_tool_index) {
                            continue;
                        }
                    }
                    accepted.push_back(&tcd);
                }
                if (!accepted.empty()) {
                    emitted_tool_calls = true;
                    if (!ensure_stream_head()) {
                        session->cancel();
                        return false;
                    }
                    if (anthropic) {
                        for (const ToolCallDelta* tcd : accepted) {
                            if (!send_frame(messages_stream.tool_call_delta(*tcd))) {
                                session->cancel();
                                return false;
                            }
                        }
                    } else {
                        json::Array tcs;
                        tcs.reserve(accepted.size());
                        for (const ToolCallDelta* tcd : accepted) {
                            std::string id = tcd->id;
                            if (tcd->index < 64u) {
                                if (tcd->index >= seen_native_tool_indices.size()) {
                                    seen_native_tool_indices.resize(static_cast<std::size_t>(tcd->index) + 1, false);
                                }
                                if (!seen_native_tool_indices[tcd->index]) {
                                    seen_native_tool_indices[tcd->index] = true;
                                    if (id.empty()) {
                                        id = "call_" + *ro.request_id + "_" + std::to_string(tcd->index);
                                    }
                                }
                            }
                            tcs.emplace_back(
                                openai_tool_call_delta_json(tcd->index, id, tcd->name, tcd->arguments_delta));
                        }
                        if (!send_event(chunk_object(json::Object{{"tool_calls", std::move(tcs)}}, nullptr))) {
                            session->cancel();
                            return false;
                        }
                    }
                }
            }

            // 2. Visible text & reasoning (with incremental <tool_call> tag parsing when tools are active).
            if (tools_enabled) {
                const ToolCallStreamOutput parsed = tool_parser.feed(chunk.text);
                if (!parsed.visible_text.empty() || !chunk.reasoning.empty()) {
                    if (!parsed.visible_text.empty()) saw_text = true;
                    if (!ensure_stream_head()) {
                        session->cancel();
                        return false;
                    }
                    if (anthropic) {
                        TokenChunk vis{parsed.visible_text, chunk.index, chunk.reasoning};
                        if (!send_frame(messages_stream.chunk(vis))) {
                            session->cancel();
                            return false;
                        }
                    } else {
                        json::Object delta;
                        if (!parsed.visible_text.empty() || chunk.reasoning.empty()) {
                            delta.set("content", parsed.visible_text);
                        }
                        if (!chunk.reasoning.empty()) {
                            delta.set("reasoning_content", std::string(chunk.reasoning));
                        }
                        json::Value lp_val = nullptr;
                        if (job.sampling.logprobs && chunk.logprob.has_value() && !parsed.visible_text.empty()) {
                            lp_val = chat_logprobs_json(
                                {TokenLogprob{parsed.visible_text, *chunk.logprob, chunk.top_logprobs}});
                        }
                        if (!send_event(chunk_object(std::move(delta), nullptr, std::move(lp_val)))) {
                            session->cancel();
                            return false;
                        }
                    }
                }
                for (const auto& ev : parsed.tool_calls) {
                    if (!emit_parsed_tool(ev)) {
                        session->cancel();
                        return false;
                    }
                }
                return true;
            }

            if (chunk.text.empty() && chunk.reasoning.empty()) {
                return true;
            }
            if (!chunk.text.empty()) {
                saw_text = true;
            }
            const auto openai_delta = [&] {
                json::Object delta;
                if (!chunk.text.empty() || chunk.reasoning.empty()) delta.set("content", std::string(chunk.text));
                if (!chunk.reasoning.empty()) delta.set("reasoning_content", std::string(chunk.reasoning));
                return delta;
            };
            json::Value lp_val = nullptr;
            if (!anthropic && job.sampling.logprobs && chunk.logprob.has_value() && !chunk.text.empty()) {
                lp_val = chat_logprobs_json({TokenLogprob{std::string(chunk.text), *chunk.logprob, chunk.top_logprobs}});
            }
            if (!ensure_stream_head() ||
                !(anthropic ? send_frame(messages_stream.chunk(chunk))
                            : send_event(chunk_object(openai_delta(), nullptr, std::move(lp_val))))) {
                session->cancel();
                return false;
            }
            return true;
        };
    }

    std::vector<ChatMessage> prepared_messages;
    const std::vector<ChatMessage>* messages_to_send = &job.messages;
    if (tools_enabled) {
        prepared_messages = job.messages;
        const std::string tool_prompt = format_tools_system_prompt(job.tools, job.tool_choice, job.parallel_tool_calls);
        if (!prepared_messages.empty() && prepared_messages.front().role == "system") {
            append_chat_message_text(prepared_messages.front(), "\n\n" + tool_prompt);
        } else {
            prepared_messages.insert(prepared_messages.begin(), ChatMessage{"system", tool_prompt});
        }
        messages_to_send = &prepared_messages;
    }

    auto result = session->chat(*messages_to_send, on_chunk, job.sampling, ro);
    end_request();
    const bool rejected = session->last_scheduler_rejected();

    if (disconnected.load() || sse.write_failed()) {
        ex.status = 499; // client closed the request (access log only)
        session->close();
        return;
    }
    if (deadline_ms && !backend_started.load() && std::chrono::steady_clock::now() >= deadline) {
        session->close();
        ex.send_error(make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed"));
        return;
    }
    if (!result.ok()) {
        if (session->requests_started() == 0) {
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "chat",
                                       to_string(result.status().code()), corr, priority, model, job.model);
        }
        session->close();
        const ApiError e = map_session_failure(result.status(), rejected);
        if (sse.headers_sent()) {
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
    if (grammar_matcher) {
        if (!job.stream && !r.text.empty()) {
            (void)grammar_matcher->accept_chunk(r.text);
        }
        grammar_matcher->mark_finished();
    }
    if (job.stream && tools_enabled) {
        const ToolCallStreamOutput tail = tool_parser.flush();
        if (!tail.visible_text.empty()) {
            saw_text = true;
            if (ensure_stream_head()) {
                if (anthropic) {
                    (void)send_frame(messages_stream.chunk(TokenChunk{tail.visible_text, 0}));
                } else {
                    (void)send_event(chunk_object(json::Object{{"content", tail.visible_text}}, nullptr));
                }
            }
        }
        for (const auto& ev : tail.tool_calls) {
            (void)emit_parsed_tool(ev);
        }
    }

    std::string visible_text;
    std::vector<ToolCall> final_tool_calls;
    if (!job.stream) {
        if (!r.tool_calls.empty()) {
            visible_text = r.text;
            for (const auto& tc : r.tool_calls) {
                if (tc.name.empty() && tc.arguments_json.empty()) continue;
                if (!allow_parallel && !final_tool_calls.empty()) break;
                ToolCall normalized = tc;
                if (normalized.id.empty()) {
                    normalized.id = "call_" + *ro.request_id + "_" + std::to_string(final_tool_calls.size());
                }
                final_tool_calls.push_back(std::move(normalized));
            }
        } else if (tools_enabled) {
            ExtractedToolOutput extracted = extract_tool_calls(r.text, "call_" + *ro.request_id + "_", allow_parallel);
            visible_text = std::move(extracted.visible_text);
            final_tool_calls = std::move(extracted.tool_calls);
        } else {
            visible_text = r.text;
        }
    }
    const bool has_tool_calls = job.stream ? emitted_tool_calls : !final_tool_calls.empty();
    const char* effective_finish = has_tool_calls ? "tool_calls" : finish_reason(r);

    const bool empty_text = job.stream ? (!saw_text && !has_tool_calls) : (visible_text.empty() && !has_tool_calls);
    if (empty_text && std::string_view(effective_finish) == "length") warnings.emplace_back("empty_content_at_length");
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
            json::Object doc = anthropic_message_response(completion_id, model, r, warnings, final_tool_calls,
                                                          std::string_view(visible_text));
            doc.set("sonder", std::move(meta));
            ex.send_json(200, doc);
            return;
        }
        json::Object message{{"role", "assistant"},
                             {"content", (!final_tool_calls.empty() && visible_text.empty())
                                             ? json::Value(nullptr)
                                             : json::Value(visible_text)}};
        if (!r.reasoning.empty()) message.set("reasoning_content", r.reasoning);
        if (!final_tool_calls.empty()) {
            json::Array tcs;
            tcs.reserve(final_tool_calls.size());
            for (const auto& tc : final_tool_calls) {
                tcs.emplace_back(openai_tool_call_json(tc));
            }
            message.set("tool_calls", std::move(tcs));
        }
        json::Object choice{{"index", 0}, {"message", std::move(message)}, {"finish_reason", effective_finish}};
        if (job.sampling.logprobs) {
            choice.set("logprobs", chat_logprobs_json(r.logprobs));
        }
        json::Object doc{
            {"id", completion_id},
            {"object", "chat.completion"},
            {"created", created_at},
            {"model", model},
            {"choices", json::Array{std::move(choice)}},
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
    json::Object last = chunk_object(json::Object{}, effective_finish);
    if (job.include_usage) {
        last.set("usage", usage_json(r));
    }
    last.set("timings", timings_json(r));
    last.set("sonder", std::move(meta));
    if (send_event(last)) {
        (void)send_frame("data: [DONE]\n\n");
    }
}

void execute_completion(ChatExchange& ex, const CompletionJob& job, const Correlation& corr, const std::string& model,
                        const std::string& backend, const std::shared_ptr<Session>& session) {
    const RequestPriority priority = corr.priority_class.value_or(
        job.priority_class.value_or(RequestPriority::interactive));
    const auto deadline_ms = corr.deadline_ms ? corr.deadline_ms : job.deadline_ms;
    const auto deadline = deadline_ms
        ? ex.started + std::chrono::milliseconds(static_cast<std::int64_t>(*deadline_ms))
        : std::chrono::steady_clock::time_point::max();
    RequestOptions ro;
    ro.request_id = make_id("req");
    ro.parent_request_id = corr.parent_request_id;
    if (job.session_key) {
        ro.session_key = *job.session_key;
    } else {
        ChatJob key_stub;
        ro.session_key = chat_session_key(key_stub, corr);
    }
    ro.retain_text = !job.stream;
    ro.grammar = job.grammar;
    ro.priority_class = priority;
    ro.preserve_numeric_priority = corr.numeric_priority;
    if (deadline_ms) ro.deadline = deadline;
    if (ex.request_id.rfind("req-", 0) == 0) ro.request_id = ex.request_id;
    ro.served_model = model;
    ro.requested_model = job.model;
    std::atomic<bool> backend_started{false};
    const std::string completion_id = "cmpl-" + *ro.request_id;
    ex.request_id = *ro.request_id;
    const std::int64_t created_at = static_cast<std::int64_t>(std::time(nullptr));

    SseStreamWriter sse(ex, ex.options.stream_keepalive_interval);
    std::mutex watch_mu;
    std::condition_variable watch_cv;
    bool watch_done = false;
    std::atomic<bool> disconnected{false};
    ro.on_backend_start = [&] {
        if (disconnected.load() || ex.hard_stop.load() || std::chrono::steady_clock::now() >= deadline) {
            session->cancel();
            return;
        }
        backend_started.store(true);
        if (job.stream) {
            sse.arm_keepalive();
        }
    };
    std::shared_ptr<PriorityAdmission::Ticket> ticket;
    std::thread watcher;
    bool left_inflight = false;
    const auto end_request = [&] {
        sse.disarm_keepalive();
        if (ticket) ticket->release();
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
                }
                const auto now = std::chrono::steady_clock::now();
                if (now >= deadline) {
                    // Only cancel via the watcher once the backend has started; pre-start
                    // queue/scheduler waits enforce ro.deadline directly so queued deadline
                    // expirations report ErrorCode::timeout (request.failed) instead of
                    // racing into request.cancelled.
                    if (backend_started.load()) {
                        session->cancel();
                    }
                } else if (job.stream && !disconnected.load() && !ex.hard_stop.load() &&
                           !sse.maybe_send_keepalive()) {
                    disconnected.store(true);
                    session->cancel();
                }
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });
    } catch (const std::system_error&) {
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "generate", "overloaded",
                                   corr, priority, model, job.model);
        end_request();
        session->close();
        ex.watcher_failure();
        ApiError e =
            make_error(503, "overloaded",
                       "sonder-inference cannot start a request thread right now; nothing was executed, retry later");
        e.retry_after = 1;
        ex.send_error(e);
        return;
    }

    if (priority_admission_enabled(ex.options)) {
        ticket = ex.admission.enqueue(priority, [&] {
            return disconnected.load() || session->state() == SessionState::closed || ex.hard_stop.load();
        }, deadline);
        if (ticket->outcome() != PriorityAdmission::Outcome::admitted) {
            const char* err_code = ticket->outcome() == PriorityAdmission::Outcome::full
                                       ? "overloaded"
                                       : ticket->outcome() == PriorityAdmission::Outcome::expired
                                       ? "deadline_exceeded"
                                       : "cancelled";
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "generate", err_code,
                                       corr, priority, model, job.model, disconnected.load(), ticket->queue_ms());
            end_request();
            if (disconnected.load()) {
                ex.status = 499;
            } else {
                ApiError error = ticket->outcome() == PriorityAdmission::Outcome::expired
                    ? make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed")
                    : ticket->outcome() == PriorityAdmission::Outcome::full
                    ? make_error(429, "overloaded", "request priority queue is full; nothing was executed")
                    : make_error(503, "not_ready", "request cancelled while queued; nothing was executed");
                if (error.status == 429) error.retry_after = 1;
                ex.send_error(error);
            }
            session->close();
            return;
        }
        ro.admission = ticket;
    }

    const auto chunk_object = [&](std::string text, json::Value finish, json::Value logprobs_val = nullptr) {
        return json::Object{{"id", completion_id},
                            {"object", "text_completion"},
                            {"created", created_at},
                            {"model", model},
                            {"choices", json::Array{json::Object{{"text", std::move(text)},
                                                                 {"index", 0},
                                                                 {"logprobs", std::move(logprobs_val)},
                                                                 {"finish_reason", std::move(finish)}}}}};
    };
    const auto send_frame = [&](std::string_view frame) { return sse.write_frame(frame); };
    const auto send_event = [&](const json::Object& obj) {
        return sse.write_frame("data: " + json::Value(obj).dump() + "\n\n");
    };
    const auto ensure_stream_head = [&]() {
        return sse.ensure_started([&]() -> std::string {
            if (job.echo) {
                return "data: " + json::Value(chunk_object(job.prompt, nullptr)).dump() + "\n\n";
            }
            return {};
        });
    };

    std::size_t text_offset = job.echo ? job.prompt.size() : 0;
    TokenCallback on_chunk;
    if (job.stream) {
        on_chunk = [&](const TokenChunk& chunk) {
            if (chunk.text.empty()) {
                return true;
            }
            json::Value lp_val = nullptr;
            if (job.sampling.logprobs && chunk.logprob.has_value()) {
                lp_val = completion_logprobs_json(
                    {TokenLogprob{std::string(chunk.text), *chunk.logprob, chunk.top_logprobs}}, text_offset);
            }
            text_offset += chunk.text.size();
            if (!ensure_stream_head() || !send_event(chunk_object(std::string(chunk.text), nullptr, std::move(lp_val)))) {
                session->cancel();
                return false;
            }
            return true;
        };
    }

    auto result = session->generate(job.prompt, on_chunk, job.sampling, ro);
    end_request();
    const bool rejected = session->last_scheduler_rejected();

    if (disconnected.load() || sse.write_failed()) {
        ex.status = 499;
        session->close();
        return;
    }
    if (deadline_ms && !backend_started.load() && std::chrono::steady_clock::now() >= deadline) {
        session->close();
        ex.send_error(make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed"));
        return;
    }
    if (!result.ok()) {
        if (session->requests_started() == 0) {
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "generate",
                                       to_string(result.status().code()), corr, priority, model, job.model);
        }
        session->close();
        const ApiError e = map_session_failure(result.status(), rejected);
        if (sse.headers_sent()) {
            (void)send_event(error_body(e));
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
    session->close();
    if (!job.stream) {
        std::string text = job.echo ? (job.prompt + r.text) : r.text;
        json::Value lp_val =
            job.sampling.logprobs
                ? json::Value(completion_logprobs_json(r.logprobs, job.echo ? job.prompt.size() : 0))
                : json::Value(nullptr);
        json::Object doc{
            {"id", completion_id},
            {"object", "text_completion"},
            {"created", created_at},
            {"model", model},
            {"choices",
             json::Array{json::Object{{"text", std::move(text)},
                                      {"index", 0},
                                      {"logprobs", std::move(lp_val)},
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
    json::Object last = chunk_object(std::string{}, finish_reason(r));
    if (job.include_usage) {
        last.set("usage", usage_json(r));
    }
    last.set("timings", timings_json(r));
    last.set("sonder", std::move(meta));
    if (send_event(last)) {
        (void)send_frame("data: [DONE]\n\n");
    }
}

void execute_embeddings(ChatExchange& ex, const EmbedJob& job, const Correlation& corr, const std::string& model,
                        const std::string& backend, const std::shared_ptr<Session>& session) {
    const RequestPriority priority = corr.priority_class.value_or(
        job.priority_class.value_or(RequestPriority::interactive));
    const auto deadline_ms = corr.deadline_ms ? corr.deadline_ms : job.deadline_ms;
    const auto deadline = deadline_ms
        ? ex.started + std::chrono::milliseconds(static_cast<std::int64_t>(*deadline_ms))
        : std::chrono::steady_clock::time_point::max();
    RequestOptions ro;
    ro.request_id = make_id("req");
    ro.parent_request_id = corr.parent_request_id;
    ro.priority_class = priority;
    ro.preserve_numeric_priority = corr.numeric_priority;
    if (deadline_ms) ro.deadline = deadline;
    if (ex.request_id.rfind("req-", 0) == 0) ro.request_id = ex.request_id;
    ro.served_model = model;
    ro.requested_model = job.model;
    std::atomic<bool> backend_started{false};
    ex.request_id = *ro.request_id;

    std::mutex watch_mu;
    std::condition_variable watch_cv;
    bool watch_done = false;
    std::atomic<bool> disconnected{false};
    ro.on_backend_start = [&] {
        if (disconnected.load() || ex.hard_stop.load() || std::chrono::steady_clock::now() >= deadline) {
            session->cancel();
            return;
        }
        backend_started.store(true);
    };
    std::shared_ptr<PriorityAdmission::Ticket> ticket;
    std::thread watcher;
    bool left_inflight = false;
    const auto end_request = [&] {
        if (ticket) ticket->release();
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
                }
                if (std::chrono::steady_clock::now() >= deadline && backend_started.load()) {
                    session->cancel();
                }
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });
    } catch (const std::system_error&) {
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "embed", "overloaded",
                                   corr, priority, model, job.model);
        end_request();
        session->close();
        ex.watcher_failure();
        ApiError e =
            make_error(503, "overloaded",
                       "sonder-inference cannot start a request thread right now; nothing was executed, retry later");
        e.retry_after = 1;
        ex.send_error(e);
        return;
    }

    if (priority_admission_enabled(ex.options)) {
        ticket = ex.admission.enqueue(priority, [&] {
            return disconnected.load() || session->state() == SessionState::closed || ex.hard_stop.load();
        }, deadline);
        if (ticket->outcome() != PriorityAdmission::Outcome::admitted) {
            const char* err_code = ticket->outcome() == PriorityAdmission::Outcome::full
                                       ? "overloaded"
                                       : ticket->outcome() == PriorityAdmission::Outcome::expired
                                       ? "deadline_exceeded"
                                       : "cancelled";
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "embed", err_code,
                                       corr, priority, model, job.model, disconnected.load(), ticket->queue_ms());
            end_request();
            if (disconnected.load()) {
                ex.status = 499;
            } else {
                ApiError error = ticket->outcome() == PriorityAdmission::Outcome::expired
                    ? make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed")
                    : ticket->outcome() == PriorityAdmission::Outcome::full
                    ? make_error(429, "overloaded", "request priority queue is full; nothing was executed")
                    : make_error(503, "not_ready", "request cancelled while queued; nothing was executed");
                if (error.status == 429) error.retry_after = 1;
                ex.send_error(error);
            }
            session->close();
            return;
        }
        ro.admission = ticket;
    }

    EmbedRequest embed_req;
    embed_req.request_id = *ro.request_id;
    embed_req.inputs = job.inputs;
    embed_req.normalize = job.normalize;
    embed_req.dimensions = job.dimensions;
    embed_req.pooling_type = job.pooling_type;
    embed_req.encoding_format = job.encoding_format;
    auto result = session->embed(embed_req, ro);
    end_request();

    if (disconnected.load()) {
        ex.status = 499;
        session->close();
        return;
    }
    if (!result.ok() && session->requests_started() == 0) {
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "embed",
                                   to_string(result.status().code()), corr, priority, model, job.model);
    }
    if (deadline_ms && !backend_started.load() && std::chrono::steady_clock::now() >= deadline) {
        session->close();
        ex.send_error(make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed"));
        return;
    }
    if (!result.ok()) {
        session->close();
        if (result.status().code() == ErrorCode::unsupported) {
            ex.send_error(make_error(501, "not_implemented", result.status().message()));
        } else {
            ex.send_error(map_session_failure(result.status(), false));
        }
        return;
    }

    const EmbedResult& r = result.value();
    json::Array data;
    data.reserve(r.embeddings.size());
    for (std::size_t i = 0; i < r.embeddings.size(); ++i) {
        json::Array vec;
        vec.reserve(r.embeddings[i].size());
        for (const float v : r.embeddings[i]) {
            vec.emplace_back(static_cast<double>(v));
        }
        data.emplace_back(json::Object{{"object", "embedding"},
                                       {"index", static_cast<std::int64_t>(i)},
                                       {"embedding", std::move(vec)}});
    }
    const std::size_t effective_dims =
        r.dimensions > 0 ? r.dimensions : (!r.embeddings.empty() ? r.embeddings.front().size() : 0u);
    json::Object meta{{"api_version", kApiVersion},
                      {"request_id", *ro.request_id},
                      {"backend", backend},
                      {"model", model},
                      {"synthetic", ex.synthetic},
                      {"dimensions", static_cast<std::uint64_t>(effective_dims)},
                      {"pooling_type", to_string(r.pooling_type)},
                      {"encoding_format", r.encoding_format.empty() ? job.encoding_format : r.encoding_format}};
    json::Object doc{{"object", "list"},
                     {"data", std::move(data)},
                     {"model", model},
                     {"usage", json::Object{{"prompt_tokens", r.prompt_tokens}, {"total_tokens", r.prompt_tokens}}},
                     {"sonder", std::move(meta)}};
    session->close();
    ex.send_json(200, doc);
}

void execute_rerank(ChatExchange& ex, const RerankJob& job, const Correlation& corr, const std::string& model,
                    const std::string& backend, const std::shared_ptr<Session>& session) {
    const RequestPriority priority = corr.priority_class.value_or(
        job.priority_class.value_or(RequestPriority::interactive));
    const auto deadline_ms = corr.deadline_ms ? corr.deadline_ms : job.deadline_ms;
    const auto deadline = deadline_ms
        ? ex.started + std::chrono::milliseconds(static_cast<std::int64_t>(*deadline_ms))
        : std::chrono::steady_clock::time_point::max();
    RequestOptions ro;
    ro.request_id = make_id("req");
    ro.parent_request_id = corr.parent_request_id;
    ro.priority_class = priority;
    ro.preserve_numeric_priority = corr.numeric_priority;
    if (deadline_ms) ro.deadline = deadline;
    if (ex.request_id.rfind("req-", 0) == 0) ro.request_id = ex.request_id;
    ro.served_model = model;
    ro.requested_model = job.model;
    std::atomic<bool> backend_started{false};
    ex.request_id = *ro.request_id;

    std::mutex watch_mu;
    std::condition_variable watch_cv;
    bool watch_done = false;
    std::atomic<bool> disconnected{false};
    ro.on_backend_start = [&] {
        if (disconnected.load() || ex.hard_stop.load() || std::chrono::steady_clock::now() >= deadline) {
            session->cancel();
            return;
        }
        backend_started.store(true);
    };
    std::shared_ptr<PriorityAdmission::Ticket> ticket;
    std::thread watcher;
    bool left_inflight = false;
    const auto end_request = [&] {
        if (ticket) ticket->release();
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
                }
                if (std::chrono::steady_clock::now() >= deadline && backend_started.load()) {
                    session->cancel();
                }
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });
    } catch (const std::system_error&) {
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "rerank", "overloaded",
                                   corr, priority, model, job.model);
        end_request();
        session->close();
        ex.watcher_failure();
        ApiError e =
            make_error(503, "overloaded",
                       "sonder-inference cannot start a request thread right now; nothing was executed, retry later");
        e.retry_after = 1;
        ex.send_error(e);
        return;
    }

    if (priority_admission_enabled(ex.options)) {
        ticket = ex.admission.enqueue(priority, [&] {
            return disconnected.load() || session->state() == SessionState::closed || ex.hard_stop.load();
        }, deadline);
        if (ticket->outcome() != PriorityAdmission::Outcome::admitted) {
            const char* err_code = ticket->outcome() == PriorityAdmission::Outcome::full
                                       ? "overloaded"
                                       : ticket->outcome() == PriorityAdmission::Outcome::expired
                                       ? "deadline_exceeded"
                                       : "cancelled";
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "rerank", err_code,
                                       corr, priority, model, job.model, disconnected.load(), ticket->queue_ms());
            end_request();
            if (disconnected.load()) {
                ex.status = 499;
            } else {
                ApiError error = ticket->outcome() == PriorityAdmission::Outcome::expired
                    ? make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed")
                    : ticket->outcome() == PriorityAdmission::Outcome::full
                    ? make_error(429, "overloaded", "request priority queue is full; nothing was executed")
                    : make_error(503, "not_ready", "request cancelled while queued; nothing was executed");
                if (error.status == 429) error.retry_after = 1;
                ex.send_error(error);
            }
            session->close();
            return;
        }
        ro.admission = ticket;
    }

    RerankRequest req;
    req.request_id = *ro.request_id;
    req.query = job.query;
    req.documents = job.documents;
    req.top_n = job.top_n;
    req.return_documents = job.return_documents;
    auto result = session->rerank(req, ro);
    end_request();

    if (disconnected.load()) {
        ex.status = 499;
        session->close();
        return;
    }
    if (!result.ok() && session->requests_started() == 0) {
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "rerank",
                                   to_string(result.status().code()), corr, priority, model, job.model);
    }
    if (deadline_ms && !backend_started.load() && std::chrono::steady_clock::now() >= deadline) {
        session->close();
        ex.send_error(make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed"));
        return;
    }
    if (!result.ok()) {
        session->close();
        if (result.status().code() == ErrorCode::unsupported) {
            ex.send_error(make_error(501, "not_implemented", result.status().message()));
        } else {
            ex.send_error(map_session_failure(result.status(), false));
        }
        return;
    }

    const RerankResult& r = result.value();
    json::Array results_arr;
    results_arr.reserve(r.results.size());
    for (const auto& item : r.results) {
        json::Object entry{{"index", static_cast<std::int64_t>(item.index)},
                           {"relevance_score", item.relevance_score}};
        if (item.document.has_value()) {
            entry.set("document", json::Object{{"text", *item.document}});
        }
        results_arr.emplace_back(std::move(entry));
    }
    json::Object meta{{"api_version", kApiVersion},
                      {"request_id", *ro.request_id},
                      {"backend", backend},
                      {"model", model},
                      {"synthetic", ex.synthetic}};
    json::Object doc{{"object", "list"},
                     {"model", model},
                     {"results", std::move(results_arr)},
                     {"usage", json::Object{{"prompt_tokens", r.prompt_tokens}, {"total_tokens", r.prompt_tokens}}},
                     {"sonder", std::move(meta)}};
    session->close();
    ex.send_json(200, doc);
}

void execute_score(ChatExchange& ex, const ScoreJob& job, const Correlation& corr, const std::string& model,
                   const std::string& backend, const std::shared_ptr<Session>& session) {
    const RequestPriority priority = corr.priority_class.value_or(
        job.priority_class.value_or(RequestPriority::interactive));
    const auto deadline_ms = corr.deadline_ms ? corr.deadline_ms : job.deadline_ms;
    const auto deadline = deadline_ms
        ? ex.started + std::chrono::milliseconds(static_cast<std::int64_t>(*deadline_ms))
        : std::chrono::steady_clock::time_point::max();
    RequestOptions ro;
    ro.request_id = make_id("req");
    ro.parent_request_id = corr.parent_request_id;
    ro.priority_class = priority;
    ro.preserve_numeric_priority = corr.numeric_priority;
    if (deadline_ms) ro.deadline = deadline;
    if (ex.request_id.rfind("req-", 0) == 0) ro.request_id = ex.request_id;
    ro.served_model = model;
    ro.requested_model = job.model;
    std::atomic<bool> backend_started{false};
    ex.request_id = *ro.request_id;

    std::mutex watch_mu;
    std::condition_variable watch_cv;
    bool watch_done = false;
    std::atomic<bool> disconnected{false};
    ro.on_backend_start = [&] {
        if (disconnected.load() || ex.hard_stop.load() || std::chrono::steady_clock::now() >= deadline) {
            session->cancel();
            return;
        }
        backend_started.store(true);
    };
    std::shared_ptr<PriorityAdmission::Ticket> ticket;
    std::thread watcher;
    bool left_inflight = false;
    const auto end_request = [&] {
        if (ticket) ticket->release();
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
                }
                if (std::chrono::steady_clock::now() >= deadline && backend_started.load()) {
                    session->cancel();
                }
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });
    } catch (const std::system_error&) {
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "score", "overloaded",
                                   corr, priority, model, job.model);
        end_request();
        session->close();
        ex.watcher_failure();
        ApiError e =
            make_error(503, "overloaded",
                       "sonder-inference cannot start a request thread right now; nothing was executed, retry later");
        e.retry_after = 1;
        ex.send_error(e);
        return;
    }

    if (priority_admission_enabled(ex.options)) {
        ticket = ex.admission.enqueue(priority, [&] {
            return disconnected.load() || session->state() == SessionState::closed || ex.hard_stop.load();
        }, deadline);
        if (ticket->outcome() != PriorityAdmission::Outcome::admitted) {
            const char* err_code = ticket->outcome() == PriorityAdmission::Outcome::full
                                       ? "overloaded"
                                       : ticket->outcome() == PriorityAdmission::Outcome::expired
                                       ? "deadline_exceeded"
                                       : "cancelled";
            emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "score", err_code,
                                       corr, priority, model, job.model, disconnected.load(), ticket->queue_ms());
            end_request();
            if (disconnected.load()) {
                ex.status = 499;
            } else {
                ApiError error = ticket->outcome() == PriorityAdmission::Outcome::expired
                    ? make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed")
                    : ticket->outcome() == PriorityAdmission::Outcome::full
                    ? make_error(429, "overloaded", "request priority queue is full; nothing was executed")
                    : make_error(503, "not_ready", "request cancelled while queued; nothing was executed");
                if (error.status == 429) error.retry_after = 1;
                ex.send_error(error);
            }
            session->close();
            return;
        }
        ro.admission = ticket;
    }

    ScoreRequest req;
    req.request_id = *ro.request_id;
    req.pairs = job.pairs;
    auto result = session->score(req, ro);
    end_request();

    if (disconnected.load()) {
        ex.status = 499;
        session->close();
        return;
    }
    if (!result.ok() && session->requests_started() == 0) {
        emit_pre_execution_failure(ex.telemetry, session->telemetry_context(ro.request_id), "score",
                                   to_string(result.status().code()), corr, priority, model, job.model);
    }
    if (deadline_ms && !backend_started.load() && std::chrono::steady_clock::now() >= deadline) {
        session->close();
        ex.send_error(make_error(504, "deadline_exceeded", "request deadline expired while queued; nothing was executed"));
        return;
    }
    if (!result.ok()) {
        session->close();
        if (result.status().code() == ErrorCode::unsupported) {
            ex.send_error(make_error(501, "not_implemented", result.status().message()));
        } else {
            ex.send_error(map_session_failure(result.status(), false));
        }
        return;
    }

    const ScoreResult& r = result.value();
    json::Array data;
    data.reserve(r.scores.size());
    for (const auto& item : r.scores) {
        data.emplace_back(json::Object{{"object", "score"},
                                       {"index", static_cast<std::int64_t>(item.index)},
                                       {"score", item.score}});
    }
    json::Object meta{{"api_version", kApiVersion},
                      {"request_id", *ro.request_id},
                      {"backend", backend},
                      {"model", model},
                      {"synthetic", ex.synthetic}};
    json::Object doc{{"object", "list"},
                     {"model", model},
                     {"data", std::move(data)},
                     {"usage", json::Object{{"prompt_tokens", r.prompt_tokens}, {"total_tokens", r.prompt_tokens}}},
                     {"sonder", std::move(meta)}};
    session->close();
    ex.send_json(200, doc);
}

} // namespace sonder::inference::server::detail
