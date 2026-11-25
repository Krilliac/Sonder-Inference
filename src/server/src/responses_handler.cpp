#include "responses_handler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <ctime>
#include <limits>
#include <mutex>
#include <system_error>
#include <thread>
#include <utility>

#include "json_schema_gbnf.hpp"
#include "request_path.hpp"
#include "test_hooks.hpp"
#include "tool_call_parser.hpp"

namespace sonder::inference::server::detail {

namespace {

bool present(const json::Value* v) { return v != nullptr && !v->is_null(); }

ApiError bad(std::string code, std::string message, std::optional<std::string> param = std::nullopt) {
    return make_error(400, std::move(code), std::move(message), std::move(param));
}

std::string sse_named_event(std::string_view event, const json::Object& payload) {
    return "event: " + std::string(event) + "\ndata: " + json::Value(payload).dump() + "\n\n";
}

json::Array build_output_items(const std::string& resp_id, const std::string& reasoning,
                               const std::string& visible_text, const std::vector<ToolCall>& tool_calls) {
    json::Array output;
    if (!reasoning.empty()) {
        output.emplace_back(json::Object{
            {"id", "rs_" + resp_id},
            {"type", "reasoning"},
            {"status", "completed"},
            {"summary", json::Array{json::Object{{"type", "summary_text"}, {"text", reasoning}}}}});
    }
    if (!visible_text.empty() || tool_calls.empty()) {
        output.emplace_back(json::Object{
            {"id", "msg_" + resp_id},
            {"type", "message"},
            {"status", "completed"},
            {"role", "assistant"},
            {"content", json::Array{json::Object{{"type", "output_text"},
                                                 {"text", visible_text},
                                                 {"annotations", json::Array{}}}}}});
    }
    for (std::size_t i = 0; i < tool_calls.size(); ++i) {
        const auto& tc = tool_calls[i];
        output.emplace_back(json::Object{
            {"id", "fc_" + resp_id + "_" + std::to_string(i)},
            {"type", "function_call"},
            {"status", "completed"},
            {"call_id", tc.id.empty() ? ("call_" + resp_id + "_" + std::to_string(i)) : tc.id},
            {"name", tc.name},
            {"arguments", tc.arguments_json.empty() ? "{}" : tc.arguments_json}});
    }
    return output;
}

json::Object build_response_object(const std::string& resp_id, std::int64_t created_at, const std::string& status,
                                   const std::string& model, const std::optional<std::string>& prev_id,
                                   const std::string& reasoning, const std::string& visible_text,
                                   const std::vector<ToolCall>& tool_calls, const GenerationResult* result,
                                   json::Object sonder_meta) {
    const std::uint64_t input_tokens = result ? result->stats.prompt_tokens : 0u;
    const std::uint64_t output_tokens = result ? result->stats.completion_tokens : 0u;
    json::Object usage{{"input_tokens", input_tokens},
                       {"output_tokens", output_tokens},
                       {"total_tokens", input_tokens + output_tokens}};
    if (result && result->stats.cached_tokens) {
        usage.set("input_tokens_details", json::Object{{"cached_tokens", *result->stats.cached_tokens}});
    }
    json::Object obj{
        {"id", resp_id},
        {"object", "response"},
        {"created_at", created_at},
        {"status", status},
        {"model", model},
        {"previous_response_id", prev_id},
        {"output", status == "in_progress" ? json::Array{}
                                           : build_output_items(resp_id, reasoning, visible_text, tool_calls)},
        {"output_text", visible_text},
        {"usage", std::move(usage)},
        {"sonder", std::move(sonder_meta)}};
    return obj;
}

std::optional<ApiError> read_text_format(const json::Object& body, ResponsesJob& job) {
    const json::Value* text_v = body.find("text");
    if (!present(text_v)) {
        return std::nullopt;
    }
    if (!text_v->is_object()) {
        return bad("invalid_request", "text must be an object", "text");
    }
    const json::Value* fmt = text_v->find("format");
    if (!present(fmt)) {
        return std::nullopt;
    }
    if (!fmt->is_object()) {
        return bad("invalid_request", "text.format must be an object", "text.format");
    }
    const json::Value* type_v = fmt->find("type");
    if (!present(type_v) || !type_v->is_string()) {
        return bad("invalid_request", "text.format.type must be a string", "text.format.type");
    }
    const std::string& type = type_v->as_string();
    if (type == "text") {
        return std::nullopt;
    }
    if (type == "json_object") {
        auto compiled = default_structured_grammar_engine().compile_json_object();
        if (!compiled.ok()) {
            return bad("invalid_request", compiled.status().message(), "text.format");
        }
        job.grammar = std::move(compiled.value().gbnf);
        job.response_format_type = "json_object";
        return std::nullopt;
    }
    if (type == "json_schema") {
        const json::Value* schema_v = fmt->find("schema");
        if (!present(schema_v)) {
            if (const json::Value* nested = fmt->find("json_schema"); present(nested) && nested->is_object()) {
                schema_v = nested->find("schema");
            }
        }
        if (!present(schema_v) || !schema_v->is_object()) {
            return bad("invalid_request", "text.format.schema must be a JSON object", "text.format.schema");
        }
        auto compiled = default_structured_grammar_engine().compile_json_schema(*schema_v);
        if (!compiled.ok()) {
            return bad("invalid_request", "invalid text.format.schema: " + compiled.status().message(),
                       "text.format.schema");
        }
        job.grammar = std::move(compiled.value().gbnf);
        job.response_format_type = "json_schema";
        job.response_format_schema = schema_v->dump();
        return std::nullopt;
    }
    return bad("unsupported_parameter",
               "text.format.type '" + type + "' is not supported (expected text, json_object or json_schema)",
               "text.format.type");
}

}  // namespace

std::size_t ResponseHistoryStore::estimate_entry_bytes(const Entry& e) noexcept {
    std::size_t total = e.id.size() + e.session_key.size() + e.model.size() + 64;
    for (const auto& m : e.messages) {
        total += m.role.size() + m.content.size() + 32;
        if (m.reasoning_content) total += m.reasoning_content->size();
        if (m.tool_call_id) total += m.tool_call_id->size();
        if (m.name) total += m.name->size();
        for (const auto& tc : m.tool_calls) {
            total += tc.id.size() + tc.name.size() + tc.arguments_json.size() + 24;
        }
        for (const auto& blob : m.media) {
            total += blob.mime_type.size() + blob.bytes.size() + 16;
        }
    }
    return total;
}

void ResponseHistoryStore::evict_back_locked() noexcept {
    if (lru_.empty()) return;
    const std::size_t b = lru_.back().approx_bytes;
    used_bytes_ = used_bytes_ >= b ? (used_bytes_ - b) : 0;
    index_.erase(lru_.back().id);
    lru_.pop_back();
}

ResponseHistoryStore::ResponseHistoryStore(std::size_t capacity, std::size_t max_bytes)
    : capacity_(std::max<std::size_t>(1, capacity)), max_bytes_(std::max<std::size_t>(4096, max_bytes)) {}

std::optional<ResponseHistoryStore::Entry> ResponseHistoryStore::get(const std::string& id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = index_.find(id);
    if (it == index_.end()) {
        return std::nullopt;
    }
    lru_.splice(lru_.begin(), lru_, it->second);
    return *it->second;
}

void ResponseHistoryStore::put(Entry entry) {
    // Strip raw binary media blobs from retained history to prevent O(N^2) RAM
    // amplification across chained previous_response_id turns.
    for (auto& m : entry.messages) {
        m.media.clear();
        m.media.shrink_to_fit();
    }
    if (entry.messages.size() > kMaxMessagesPerEntry) {
        return;
    }
    entry.approx_bytes = estimate_entry_bytes(entry);
    if (entry.approx_bytes > kMaxEntryBytes || entry.approx_bytes > max_bytes_) {
        return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    auto it = index_.find(entry.id);
    if (it != index_.end()) {
        const std::size_t old_b = it->second->approx_bytes;
        used_bytes_ = used_bytes_ >= old_b ? (used_bytes_ - old_b) : 0;
        index_.erase(it);
        lru_.erase(it->second);
    }
    while (!lru_.empty() && (lru_.size() >= capacity_ || used_bytes_ > max_bytes_ - entry.approx_bytes)) {
        evict_back_locked();
    }
    used_bytes_ += entry.approx_bytes;
    const std::string key = entry.id;
    lru_.push_front(std::move(entry));
    index_[key] = lru_.begin();
}

std::size_t ResponseHistoryStore::size() const {
    std::lock_guard<std::mutex> lock(mu_);
    return lru_.size();
}

std::variant<ResponsesJob, ApiError> parse_responses_request(std::string_view body_text) {
    auto parsed = json::parse(body_text, json::ParseLimits{128, 65536});
    if (!parsed.ok()) {
        return bad("invalid_json", "request body is not valid JSON: " + parsed.status().message());
    }
    if (!parsed.value().is_object()) {
        return bad("invalid_json", "request body must be a JSON object");
    }
    const json::Object& body = parsed.value().as_object();
    ResponsesJob job;

    if (const json::Value* model = body.find("model"); present(model)) {
        if (!model->is_string() || model->as_string().empty()) {
            return bad("invalid_request", "model must be a non-empty string", "model");
        }
        job.model = model->as_string();
    }
    if (const json::Value* instr = body.find("instructions"); present(instr)) {
        if (!instr->is_string()) {
            return bad("invalid_request", "instructions must be a string", "instructions");
        }
        job.instructions = instr->as_string();
    }
    if (const json::Value* prev = body.find("previous_response_id"); present(prev)) {
        if (!prev->is_string() || prev->as_string().empty() || prev->as_string().size() > 128) {
            return bad("invalid_request", "previous_response_id must be a non-empty string of at most 128 bytes",
                       "previous_response_id");
        }
        job.previous_response_id = prev->as_string();
    }
    if (const json::Value* stream = body.find("stream"); present(stream)) {
        if (!stream->is_bool()) {
            return bad("invalid_request", "stream must be a boolean", "stream");
        }
        job.stream = stream->as_bool();
    }
    if (const json::Value* store = body.find("store"); present(store)) {
        if (!store->is_bool()) {
            return bad("invalid_request", "store must be a boolean", "store");
        }
        job.store = store->as_bool();
    }
    if (const json::Value* key = body.find("prompt_cache_key"); present(key)) {
        if (!key->is_string() || key->as_string().empty() || key->as_string().size() > 256) {
            return bad("invalid_request", "prompt_cache_key must be a string of 1 to 256 bytes", "prompt_cache_key");
        }
        job.session_key = key->as_string();
    }
    if (const json::Value* tools_v = body.find("tools"); present(tools_v)) {
        if (auto e = parse_openai_tools(*tools_v, job.tools, true)) {
            return *e;
        }
    }
    if (const json::Value* tc_v = body.find("tool_choice"); present(tc_v)) {
        ToolChoice tc;
        if (auto e = parse_openai_tool_choice(*tc_v, job.tools, tc, true)) {
            return *e;
        }
        job.tool_choice = std::move(tc);
    }

    const json::Value* input = body.find("input");
    if (!present(input)) {
        return bad("invalid_request", "input is required (string or non-empty array of input items)", "input");
    }
    std::size_t total_images = 0;
    if (input->is_string()) {
        job.input_messages.push_back(ChatMessage{"user", input->as_string()});
    } else if (input->is_array() && !input->as_array().empty()) {
        const auto& items = input->as_array();
        for (std::size_t i = 0; i < items.size(); ++i) {
            const std::string where = "input[" + std::to_string(i) + "]";
            if (!items[i].is_object()) {
                return bad("invalid_request", where + " must be an object", where);
            }
            const json::Value* type_v = items[i].find("type");
            const std::string item_type =
                (present(type_v) && type_v->is_string()) ? type_v->as_string() : std::string("message");

            if (item_type == "function_call_output") {
                const json::Value* cid = items[i].find("call_id");
                if (!present(cid) || !cid->is_string() || cid->as_string().empty()) {
                    return bad("invalid_request", where + ".call_id must be a non-empty string", where + ".call_id");
                }
                const json::Value* out_v = items[i].find("output");
                if (!present(out_v) || !out_v->is_string()) {
                    return bad("invalid_request", where + ".output must be a string", where + ".output");
                }
                ChatMessage tm;
                tm.role = "tool";
                tm.content = out_v->as_string();
                tm.tool_call_id = cid->as_string();
                job.input_messages.push_back(std::move(tm));
                continue;
            }
            if (item_type == "function_call") {
                const json::Value* cid = items[i].find("call_id");
                const json::Value* name = items[i].find("name");
                const json::Value* args = items[i].find("arguments");
                if (!present(cid) || !cid->is_string() || cid->as_string().empty() || !present(name) ||
                    !name->is_string() || name->as_string().empty()) {
                    return bad("invalid_request", where + " requires non-empty string call_id and name", where);
                }
                std::string args_str = "{}";
                if (present(args)) {
                    if (!args->is_string()) {
                        return bad("invalid_request", where + ".arguments must be a string", where + ".arguments");
                    }
                    args_str = args->as_string();
                }
                ChatMessage am;
                am.role = "assistant";
                am.tool_calls.push_back(ToolCall{cid->as_string(), name->as_string(), std::move(args_str)});
                job.input_messages.push_back(std::move(am));
                continue;
            }
            if (item_type != "message") {
                return bad("unsupported_parameter", where + ".type '" + item_type + "' is not supported",
                           where + ".type");
            }
            const json::Value* role_v = items[i].find("role");
            if (!present(role_v) || !role_v->is_string()) {
                return bad("invalid_request", where + ".role must be a string", where + ".role");
            }
            std::string role = role_v->as_string();
            if (role == "developer") role = "system";
            if (!is_known_chat_role(role)) {
                return bad("invalid_request", where + ".role must be system, developer, user, assistant, or tool",
                           where + ".role");
            }
            const json::Value* content = items[i].find("content");
            if (!present(content)) {
                return bad("invalid_request", where + ".content is required", where + ".content");
            }
            ChatMessage msg;
            msg.role = std::move(role);
            if (content->is_string()) {
                msg.content = content->as_string();
            } else if (content->is_array()) {
                for (std::size_t j = 0; j < content->as_array().size(); ++j) {
                    const std::string pwhere = where + ".content[" + std::to_string(j) + "]";
                    const json::Value& part = content->as_array()[j];
                    if (!part.is_object()) {
                        return bad("invalid_request", pwhere + " must be an object", pwhere);
                    }
                    const json::Value* ptype = part.find("type");
                    if (!present(ptype) || !ptype->is_string()) {
                        return bad("invalid_request", pwhere + ".type must be a string", pwhere + ".type");
                    }
                    const std::string& pt = ptype->as_string();
                    if (pt == "input_text" || pt == "output_text" || pt == "text") {
                        const json::Value* tv = part.find("text");
                        if (!present(tv) || !tv->is_string()) {
                            return bad("invalid_request", pwhere + ".text must be a string", pwhere + ".text");
                        }
                        msg.content += tv->as_string();
                    } else if (pt == "input_image" || pt == "image_url") {
                        const json::Value* iu = part.find("image_url");
                        std::string_view url_str;
                        if (present(iu) && iu->is_string()) {
                            url_str = iu->as_string();
                        } else if (present(iu) && iu->is_object()) {
                            const json::Value* u = iu->find("url");
                            if (present(u) && u->is_string()) url_str = u->as_string();
                        }
                        if (url_str.empty()) {
                            return bad("invalid_request", pwhere + ".image_url must be a data URI string",
                                       pwhere + ".image_url");
                        }
                        if (++total_images > kMaxImagesPerRequest) {
                            return bad("invalid_request",
                                       "request exceeds maximum of " + std::to_string(kMaxImagesPerRequest) + " images",
                                       "input");
                        }
                        auto media = parse_data_image_uri(url_str, pwhere + ".image_url");
                        if (auto* err = std::get_if<ApiError>(&media)) {
                            return *err;
                        }
                        msg.media.push_back(std::move(std::get<MediaContent>(media)));
                    } else {
                        return bad("unsupported_parameter", pwhere + ".type '" + pt + "' is not supported",
                                   pwhere + ".type");
                    }
                }
            } else {
                return bad("invalid_request", where + ".content must be a string or array", where + ".content");
            }
            job.input_messages.push_back(std::move(msg));
        }
    } else {
        return bad("invalid_request", "input must be a string or a non-empty array of input items", "input");
    }

    if (auto e = read_text_format(body, job)) {
        return *e;
    }

    if (const json::Value* r = body.find("reasoning"); present(r)) {
        if (!r->is_object()) {
            return bad("invalid_request", "reasoning must be an object", "reasoning");
        }
        if (const json::Value* eff = r->find("effort"); present(eff)) {
            if (!eff->is_string()) {
                return bad("invalid_request", "reasoning.effort must be a string", "reasoning.effort");
            }
            const std::string& v = eff->as_string();
            if (v == "none" || v == "off") {
                job.thinking.enable_thinking = false;
            } else if (v == "low" || v == "medium" || v == "high" || v == "xhigh") {
                job.thinking.enable_thinking = true;
                job.thinking.reasoning_effort = v;
            } else {
                return bad("invalid_request", "reasoning.effort must be none, low, medium, high, or xhigh",
                           "reasoning.effort");
            }
        }
    }

    if (const json::Value* p = body.find("priority"); present(p)) {
        if (!p->is_string() || !parse_request_priority(p->as_string())) {
            return bad("invalid_request", "priority must be interactive, subagent or background", "priority");
        }
        job.priority_class = parse_request_priority(p->as_string());
    }
    if (const json::Value* d = body.find("deadline_ms"); present(d)) {
        job.deadline_ms = parse_deadline_ms(*d);
        if (!job.deadline_ms) {
            return bad("invalid_request", "deadline_ms must be a positive integer", "deadline_ms");
        }
    }

    job.sampling = SamplingConfig{};
    job.sampling.max_tokens = kDefaultMaxTokens;
    job.sampling.explicit_only = true;
    if (const json::Value* t = body.find("temperature"); present(t)) {
        if (!t->is_number() || !std::isfinite(t->as_double()) || std::fabs(t->as_double()) > 1e6) {
            return bad("invalid_sampling", "temperature must be a finite number", "temperature");
        }
        job.sampling.temperature = static_cast<float>(t->as_double());
        job.sampling.explicit_fields |= SamplingConfig::kTemperature;
    }
    if (const json::Value* tp = body.find("top_p"); present(tp)) {
        if (!tp->is_number() || !std::isfinite(tp->as_double()) || std::fabs(tp->as_double()) > 1e6) {
            return bad("invalid_sampling", "top_p must be a finite number", "top_p");
        }
        job.sampling.top_p = static_cast<float>(tp->as_double());
        job.sampling.explicit_fields |= SamplingConfig::kTopP;
    }
    if (const json::Value* mot = body.find("max_output_tokens"); present(mot)) {
        if (!mot->is_integer() || mot->as_int() < 1 || mot->as_int() > SamplingConfig::kMaxTokensLimit) {
            return bad("invalid_sampling",
                       "max_output_tokens must be between 1 and " + std::to_string(SamplingConfig::kMaxTokensLimit),
                       "max_output_tokens");
        }
        job.sampling.max_tokens = static_cast<std::int32_t>(mot->as_int());
        job.sampling.explicit_fields |= SamplingConfig::kMaxTokens;
    }
    if (const json::Value* seed = body.find("seed"); present(seed)) {
        if (!seed->is_integer() || seed->as_int() < 0) {
            return bad("invalid_sampling", "seed must be a non-negative integer", "seed");
        }
        job.sampling.seed = seed->as_uint();
        job.sampling.explicit_fields |= SamplingConfig::kSeed;
    }
    if (Status st = validate(job.sampling); !st.ok()) {
        return bad("invalid_sampling", st.message());
    }
    return job;
}

void execute_responses(ChatExchange& ex, const ResponsesJob& job, const Correlation& corr, const std::string& model,
                       const std::string& backend, const std::shared_ptr<Session>& session,
                       ResponseHistoryStore& history, bool backend_native_tools, bool backend_supports_multimodal) {
    // 1. Assemble conversation history from optional previous_response_id.
    std::vector<ChatMessage> conversation;
    std::string inherited_session_key;
    if (job.previous_response_id) {
        auto prev = history.get(*job.previous_response_id);
        if (!prev) {
            ex.leave_inflight();
            session->close();
            ex.send_error(make_error(404, "not_found",
                                     "previous_response_id '" + *job.previous_response_id + "' was not found",
                                     "previous_response_id"));
            return;
        }
        if (!prev->model.empty() && prev->model != model) {
            ex.leave_inflight();
            session->close();
            ex.send_error(bad("invalid_request",
                              "previous_response_id belongs to a different model ('" + prev->model + "')",
                              "previous_response_id"));
            return;
        }
        conversation = std::move(prev->messages);
        inherited_session_key = std::move(prev->session_key);
    }
    if (job.instructions && !job.instructions->empty()) {
        if (!conversation.empty() && conversation.front().role == "system") {
            conversation.front().content = *job.instructions;
        } else {
            conversation.insert(conversation.begin(), ChatMessage{"system", *job.instructions});
        }
    }
    for (const auto& m : job.input_messages) {
        conversation.push_back(m);
    }
    if (conversation.size() > ResponseHistoryStore::kMaxMessagesPerEntry) {
        ex.leave_inflight();
        session->close();
        ex.send_error(bad("invalid_messages",
                          "conversation exceeds maximum of " +
                              std::to_string(ResponseHistoryStore::kMaxMessagesPerEntry) + " messages",
                          "input"));
        return;
    }
    std::size_t accumulated_images = 0;
    for (const auto& m : conversation) {
        accumulated_images += m.media.size();
    }
    if (accumulated_images > 8u) {
        ex.leave_inflight();
        session->close();
        ex.send_error(bad("invalid_messages", "conversation exceeds maximum of 8 images", "input"));
        return;
    }
    if (accumulated_images > 0 && !backend_supports_multimodal) {
        ex.leave_inflight();
        session->close();
        ex.send_error(make_error(400, "unsupported_parameter",
                                 "backend '" + backend + "' does not support multimodal image inputs", "input"));
        return;
    }
    if (Status st = validate_chat_messages(conversation); !st.ok()) {
        ex.leave_inflight();
        session->close();
        ex.send_error(bad("invalid_messages", st.message(), "input"));
        return;
    }

    // Keep clean history copy before any non-native tool prompt injection.
    const std::vector<ChatMessage> clean_history = conversation;
    if (!job.tools.empty() && !backend_native_tools) {
        const std::string tool_prompt = format_tools_system_prompt(job.tools, job.tool_choice);
        if (!conversation.empty() && conversation.front().role == "system") {
            conversation.front().content += "\n\n" + tool_prompt;
        } else {
            conversation.insert(conversation.begin(), ChatMessage{"system", tool_prompt});
        }
    }

    const RequestPriority priority = corr.priority_class.value_or(
        job.priority_class.value_or(RequestPriority::interactive));
    const auto deadline_ms = corr.deadline_ms ? corr.deadline_ms : job.deadline_ms;
    const auto deadline = deadline_ms
        ? ex.started + std::chrono::milliseconds(static_cast<std::int64_t>(*deadline_ms))
        : std::chrono::steady_clock::time_point::max();

    RequestOptions ro;
    ro.request_id = make_id("req");
    ro.parent_request_id = corr.parent_request_id;
    const std::string response_id = "resp_" + *ro.request_id;
    if (!inherited_session_key.empty()) {
        ro.session_key = inherited_session_key;
    } else if (job.session_key) {
        ro.session_key = *job.session_key;
    } else {
        ChatJob stub;
        const std::string corr_key = chat_session_key(stub, corr);
        ro.session_key = corr_key.empty() ? response_id : corr_key;
    }
    ro.thinking = job.thinking;
    ro.separate_reasoning = true;
    ro.retain_text = true;  // retain for response.completed + previous_response_id history
    ro.grammar = job.grammar;
    ro.response_format_type = job.response_format_type;
    ro.response_format_schema = job.response_format_schema;
    ro.tools = job.tools;
    ro.tool_choice = job.tool_choice;
    ro.priority_class = priority;
    ro.preserve_numeric_priority = corr.numeric_priority;
    if (deadline_ms) ro.deadline = deadline;
    std::vector<std::string> warnings = apply_thinking_pins(ex.options, ro.thinking);

    ex.request_id = *ro.request_id;
    const std::int64_t created_at = static_cast<std::int64_t>(std::time(nullptr));

    std::atomic<bool> backend_started{false};
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
        if (watcher.joinable()) watcher.join();
        if (!left_inflight) {
            left_inflight = true;
            ex.leave_inflight();
        }
    };
    struct EndGuard {
        const decltype(end_request)& end;
        ~EndGuard() { end(); }
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
                if (std::chrono::steady_clock::now() >= deadline) session->cancel();
                lock.lock();
                watch_cv.wait_for(lock, std::chrono::milliseconds(20), [&] { return watch_done; });
            }
        });
    } catch (const std::system_error&) {
        end_request();
        session->close();
        ex.watcher_failure();
        ApiError e = make_error(503, "overloaded",
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

    const bool tools_enabled = !job.tools.empty();
    ToolCallStreamParser stream_parser("call_" + *ro.request_id + "_");
    std::string streamed_visible_text;
    std::vector<ToolCall> streamed_tool_calls;
    bool headers_sent = false;
    bool write_failed = false;

    const auto send_frame = [&](std::string_view frame) {
        if (!write_failed && !frame.empty() && !ex.write(frame)) {
            write_failed = true;
        }
        return !write_failed;
    };
    const auto make_meta = [&](const GenerationResult* r) {
        json::Object meta{{"api_version", kApiVersion},
                          {"request_id", *ro.request_id},
                          {"session_id", session->id()},
                          {"session_key", ro.session_key},
                          {"backend", backend},
                          {"synthetic", ex.synthetic}};
        if (r != nullptr) {
            meta.set("token_counts_from_backend", r->stats.token_counts_from_backend);
        }
        if (!warnings.empty()) {
            meta.set("warnings", json::Array(warnings.begin(), warnings.end()));
        }
        return meta;
    };
    const auto ensure_stream_head = [&]() {
        if (headers_sent) return !write_failed;
        headers_sent = true;
        if (!ex.start_stream()) {
            write_failed = true;
            return false;
        }
        json::Object created_resp =
            build_response_object(response_id, created_at, "in_progress", model, job.previous_response_id, "", "", {},
                                  nullptr, make_meta(nullptr));
        return send_frame(sse_named_event(
            "response.created",
            json::Object{{"type", "response.created"}, {"response", std::move(created_resp)}}));
    };

    const auto emit_text_delta = [&](std::string_view delta) -> bool {
        if (delta.empty()) return true;
        streamed_visible_text.append(delta.data(), delta.size());
        return ensure_stream_head() &&
               send_frame(sse_named_event("response.output_text.delta",
                                          json::Object{{"type", "response.output_text.delta"},
                                                       {"item_id", "msg_" + response_id},
                                                       {"output_index", 0},
                                                       {"content_index", 0},
                                                       {"delta", std::string(delta)}}));
    };
    const auto emit_fn_delta = [&](std::uint32_t idx, const std::string& call_id, const std::string& name,
                                   std::string_view args_delta) -> bool {
        const std::int64_t out_idx =
            static_cast<std::int64_t>(idx) + (streamed_visible_text.empty() ? 0 : 1);
        return ensure_stream_head() &&
               send_frame(sse_named_event("response.function_call_arguments.delta",
                                          json::Object{{"type", "response.function_call_arguments.delta"},
                                                       {"item_id", "fc_" + response_id + "_" + std::to_string(idx)},
                                                       {"output_index", out_idx},
                                                       {"call_id", call_id},
                                                       {"name", name},
                                                       {"delta", std::string(args_delta)}}));
    };

    const bool parse_text_tools = tools_enabled && !backend_native_tools;
    TokenCallback on_chunk;
    if (job.stream) {
        on_chunk = [&](const TokenChunk& chunk) -> bool {
            for (const auto& tcd : chunk.tool_call_deltas) {
                if (tcd.index >= 64u) {
                    continue;
                }
                if (tcd.index >= streamed_tool_calls.size()) {
                    streamed_tool_calls.resize(static_cast<std::size_t>(tcd.index) + 1);
                }
                ToolCall& dst = streamed_tool_calls[tcd.index];
                if (!tcd.id.empty()) dst.id = tcd.id;
                else if (dst.id.empty()) dst.id = "call_" + *ro.request_id + "_" + std::to_string(tcd.index);
                if (!tcd.name.empty()) dst.name = tcd.name;
                dst.arguments_json += tcd.arguments_delta;
                if (!emit_fn_delta(tcd.index, dst.id, dst.name, tcd.arguments_delta)) {
                    session->cancel();
                    return false;
                }
            }
            if (!chunk.text.empty()) {
                if (parse_text_tools) {
                    const ToolCallStreamOutput out = stream_parser.feed(chunk.text);
                    if (!emit_text_delta(out.visible_text)) {
                        session->cancel();
                        return false;
                    }
                    for (const auto& ev : out.tool_calls) {
                        streamed_tool_calls.push_back(ToolCall{ev.id, ev.name, ev.arguments_json});
                        if (!emit_fn_delta(ev.index, ev.id, ev.name, ev.arguments_json)) {
                            session->cancel();
                            return false;
                        }
                    }
                } else {
                    if (!emit_text_delta(chunk.text)) {
                        session->cancel();
                        return false;
                    }
                }
            }
            return true;
        };
    }

    auto result = session->chat(conversation, on_chunk, job.sampling, ro);
    end_request();
    const bool rejected = session->last_scheduler_rejected();

    if (disconnected.load() || write_failed) {
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
        session->close();
        const ApiError e = map_session_failure(result.status(), rejected);
        if (headers_sent) {
            (void)send_frame(sse_named_event("error", error_body(e)));
        } else {
            ex.send_error(e);
        }
        return;
    }

    const GenerationResult& r = result.value();
    std::string final_visible;
    std::vector<ToolCall> final_calls;
    if (job.stream) {
        if (parse_text_tools) {
            const ToolCallStreamOutput tail = stream_parser.flush();
            if (!tail.visible_text.empty()) {
                (void)emit_text_delta(tail.visible_text);
            }
            for (const auto& ev : tail.tool_calls) {
                streamed_tool_calls.push_back(ToolCall{ev.id, ev.name, ev.arguments_json});
                (void)emit_fn_delta(ev.index, ev.id, ev.name, ev.arguments_json);
            }
        }
        final_visible = streamed_visible_text;
        final_calls = streamed_tool_calls;
    } else if (!r.tool_calls.empty()) {
        final_visible = r.text;
        final_calls = r.tool_calls;
    } else if (parse_text_tools) {
        ExtractedToolOutput ext = extract_tool_calls(r.text, "call_" + *ro.request_id + "_");
        final_visible = std::move(ext.visible_text);
        final_calls = std::move(ext.tool_calls);
    } else {
        final_visible = r.text;
    }

    if (job.store && r.outcome == RequestOutcome::completed) {
        ResponseHistoryStore::Entry entry;
        entry.id = response_id;
        entry.session_key = ro.session_key;
        entry.model = model;
        entry.messages = clean_history;
        ChatMessage assistant_turn;
        assistant_turn.role = "assistant";
        assistant_turn.content = final_visible;
        if (!r.reasoning.empty()) assistant_turn.reasoning_content = r.reasoning;
        assistant_turn.tool_calls = final_calls;
        entry.messages.push_back(std::move(assistant_turn));
        history.put(std::move(entry));
    }

    const std::string status_str = r.outcome == RequestOutcome::cancelled ? "incomplete" : "completed";
    json::Object completed_resp =
        build_response_object(response_id, created_at, status_str, model, job.previous_response_id, r.reasoning,
                              final_visible, final_calls, &r, make_meta(&r));
    session->close();

    if (!job.stream) {
        ex.send_json(200, completed_resp);
        return;
    }
    if (!ensure_stream_head()) {
        return;
    }
    (void)send_frame(sse_named_event(
        "response.completed",
        json::Object{{"type", "response.completed"}, {"response", std::move(completed_resp)}}));
}

}  // namespace sonder::inference::server::detail
