#include "sonder/inference/backends/llamaserver.hpp"

#include <algorithm>
#include <limits>
#include <mutex>
#include <utility>

#include "llamaserver_protocol.hpp"
#include "net/http_client.hpp"
#include "slot_affinity.hpp"
#include "supervisor.hpp"

namespace sonder::inference {
namespace {

Status protocol_error(std::string message) { return {ErrorCode::protocol_error, "llamaserver: " + message}; }

Status status_for_http(int code) {
    const std::string message = "llamaserver: upstream HTTP " + std::to_string(code);
    // Do not include arbitrary upstream bodies (which can contain prompts or credentials).
    if (code == 400 || code == 422)
        return {ErrorCode::invalid_argument, message};
    if (code == 404)
        return {ErrorCode::not_found, message};
    if (code == 408 || code == 504)
        return {ErrorCode::timeout, message};
    if (code == 503 || code == 429)
        return {ErrorCode::unavailable, message};
    return {ErrorCode::backend_error, message};
}

llamaserver::SpillGuardOptions spill_guard_options(const LlamaServerSpillGuardOptions &o) {
    llamaserver::SpillGuardOptions g;
    g.enabled = o.enabled;
    // An out-of-range enum value stays invalid and is rejected by validate_spill_guard.
    g.policy = static_cast<llamaserver::SpillPolicy>(-1);
    switch (o.policy) {
    case LlamaServerSpillPolicy::warn:
        g.policy = llamaserver::SpillPolicy::warn;
        break;
    case LlamaServerSpillPolicy::refuse:
        g.policy = llamaserver::SpillPolicy::refuse;
        break;
    case LlamaServerSpillPolicy::auto_fit:
        g.policy = llamaserver::SpillPolicy::auto_fit;
        break;
    }
    g.threshold_bytes = o.threshold_bytes;
    g.baseline_bytes = o.baseline_bytes;
    g.baseline_bytes_per_1k_ctx = o.baseline_bytes_per_1k_ctx;
    g.sample_interval = o.sample_interval;
    g.step_factor = o.fit_step_factor;
    g.step_align = o.fit_step_align;
    g.min_ctx = o.fit_min_ctx;
    g.max_attempts = o.fit_max_attempts;
    return g;
}

Status validate_options(const LlamaServerBackendOptions &options) {
    const auto duration_ok = [](std::chrono::milliseconds ms) {
        return ms.count() > 0 && ms <= std::chrono::hours(24);
    };
    for (const auto value : {options.connect_timeout, options.request_timeout, options.startup_timeout,
                             options.poll_interval, options.shutdown_timeout, options.restart_backoff,
                             options.max_restart_backoff, options.tls.handshake_timeout}) {
        if (!duration_ok(value))
            return {ErrorCode::invalid_argument, "llamaserver: timeouts must be in (0, 24h]"};
    }
    if (options.max_restarts > 1000 || options.restart_backoff > options.max_restart_backoff) {
        return {ErrorCode::invalid_argument, "llamaserver: invalid restart policy"};
    }
    if (options.context_length > (std::uint64_t{1} << 32))
        return {ErrorCode::invalid_argument, "llamaserver: context_length must be at most 4294967296"};
    if (options.mode == LlamaServerMode::spawn) {
        if (options.executable.empty() || options.executable.find('\0') != std::string::npos) {
            return {ErrorCode::invalid_argument, "llamaserver: spawn needs an executable path"};
        }
        if (options.diagnostics.log_file.find('\0') != std::string::npos)
            return {ErrorCode::invalid_argument, "llamaserver: log file path contains NUL"};
        if (options.spill_guard.enabled) {
            if (auto st = llamaserver::validate_spill_guard(spill_guard_options(options.spill_guard)); !st.ok())
                return st;
            if (options.spill_guard.policy == LlamaServerSpillPolicy::auto_fit &&
                !llamaserver::find_context_size(options.args))
                return {ErrorCode::invalid_argument,
                        "llamaserver: spill_guard.policy=auto_fit needs an explicit --ctx-size/-c argument"};
        }
        return llamaserver::validate_process_arguments(options.args);
    }
    if (options.mode != LlamaServerMode::attach)
        return {ErrorCode::invalid_argument, "llamaserver: invalid mode"};
    auto url = net::parse_url(options.base_url);
    if (!url.ok())
        return url.status();
    if (!net::is_loopback_host(url->host) && (!options.allow_remote || url->scheme != "https")) {
        return {ErrorCode::invalid_argument, "llamaserver: remote upstream requires allow_remote and HTTPS"};
    }
    return {};
}

// Bounded SSE event/line framing, also accepting native newline-delimited JSON.
// Frame callbacks own JSON validation; there is no unbounded LineSplitter.
class EventDecoder {
  public:
    using Callback = std::function<bool(std::string_view)>;
    explicit EventDecoder(Callback callback) : callback_(std::move(callback)) {}
    bool feed(std::string_view bytes) {
        while (!bytes.empty()) {
            const auto newline = bytes.find('\n');
            const auto part = bytes.substr(0, newline);
            if (line_.size() + part.size() > kLimit)
                return fail();
            line_.append(part);
            if (newline == std::string_view::npos)
                return true;
            bytes.remove_prefix(newline + 1);
            if (!line_.empty() && line_.back() == '\r')
                line_.pop_back();
            if (!line())
                return false;
            line_.clear();
        }
        return true;
    }
    bool finish() {
        if (!line_.empty()) {
            if (line_.back() == '\r')
                line_.pop_back();
            if (!line())
                return false;
            line_.clear();
        }
        return dispatch();
    }
    bool oversized() const { return oversized_; }

  private:
    bool fail() {
        oversized_ = true;
        return false;
    }
    bool dispatch() {
        if (event_.empty())
            return true;
        const bool keep_going = callback_(event_);
        event_.clear();
        return keep_going;
    }
    bool line() {
        if (line_.empty())
            return dispatch();
        if (line_.front() == ':')
            return true;
        if (line_.rfind("data:", 0) == 0) {
            std::string_view data(line_);
            data.remove_prefix(5);
            if (!data.empty() && data.front() == ' ')
                data.remove_prefix(1);
            if (event_.size() + data.size() + 1 > kLimit)
                return fail();
            if (!event_.empty())
                event_.push_back('\n');
            event_.append(data);
            return true;
        }
        if (line_.rfind("event:", 0) == 0 || line_.rfind("id:", 0) == 0 || line_.rfind("retry:", 0) == 0)
            return true;
        // Native completions on some upstream versions use NDJSON.
        if (!event_.empty())
            return callback_("invalid mixed SSE/JSON framing");
        return callback_(line_);
    }
    static constexpr std::size_t kLimit = 1024 * 1024;
    Callback callback_;
    std::string line_;
    std::string event_;
    bool oversized_ = false;
};

class LlamaServerModel;
class LlamaServerBackendImpl final : public LlamaServerBackend,
                                     public std::enable_shared_from_this<LlamaServerBackendImpl> {
  public:
    explicit LlamaServerBackendImpl(LlamaServerBackendOptions options)
        : options_(std::move(options)), validation_(validate_options(options_)) {
        if (validation_.ok() && options_.mode == LlamaServerMode::spawn) {
            llamaserver::SupervisorOptions s;
            s.environment = options_.environment;
            s.executable = options_.executable;
            s.arguments = options_.args;
            s.readiness_timeout = options_.startup_timeout;
            s.health_poll_interval = options_.poll_interval;
            s.restart_initial_backoff = options_.restart_backoff;
            s.restart_max_backoff = options_.max_restart_backoff;
            s.shutdown_timeout = options_.shutdown_timeout;
            s.max_restarts = options_.max_restarts;
            s.spill_guard = spill_guard_options(options_.spill_guard);
            s.log_file = options_.diagnostics.log_file;
            s.kv_pairing_check = options_.diagnostics.kv_pairing_check;
            supervisor_ = std::make_unique<llamaserver::Supervisor>(std::move(s));
        }
    }
    std::string name() const override { return kLlamaServerBackendName; }
    std::string description() const override {
        return options_.mode == LlamaServerMode::spawn ? "Supervised external llama-server"
                                                       : "OpenAI-compatible HTTP upstream";
    }
    BackendCapabilities capabilities() const override {
        BackendCapabilities c;
        c.add(Capability::streaming).add(Capability::remote_process);
        if (!options_.grammar.empty())
            c.add(Capability::structured_output);
        // These indicate delegation support, not measured acceleration or KV ownership.
        if (options_.native_completion)
            c.add(Capability::speculative_decode).add(Capability::prefix_reuse);
        return c;
    }
    Result<std::string> probe() override {
        auto models = list_models();
        if (!models.ok())
            return models.status();
        return std::string("OpenAI-compatible HTTP API"); // no fabricated upstream version
    }
    Result<std::vector<ModelDescriptor>> list_models() override {
        auto response = request("GET", "/v1/models", {});
        if (!response.ok())
            return response.status();
        auto parsed = json::parse(response.value());
        if (!parsed.ok())
            return protocol_error("malformed /v1/models response");
        const auto *data = parsed->find("data");
        if (!data || !data->is_array())
            return protocol_error("missing model data array");
        std::vector<ModelDescriptor> models;
        for (const auto &item : data->as_array()) {
            const auto *id = item.find("id");
            if (!id || !id->is_string() || id->as_string().empty())
                return protocol_error("missing model id");
            ModelDescriptor model;
            model.name = id->as_string();
            model.backend = kLlamaServerBackendName;
            model.resident = false;
            models.push_back(std::move(model));
        }
        return models;
    }
    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions &options) override;
    Status save_slot(std::uint32_t slot, const std::string &filename) override {
        return slot_action(slot, filename, "save");
    }
    Status restore_slot(std::uint32_t slot, const std::string &filename) override {
        return slot_action(slot, filename, "restore");
    }
    const LlamaServerBackendOptions &options() const { return options_; }
    // Spawn mode only: attach mode has no child to measure, so it reports
    // nothing and its responses stay unchanged.
    std::optional<BackendRuntimeStatus> runtime_status() const override {
        if (!supervisor_)
            return std::nullopt;
        return supervisor_->runtime_status();
    }

    Result<std::string> endpoint(const CancellationToken &cancel) {
        if (!validation_.ok())
            return validation_;
        if (cancel.cancelled())
            return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
        if (!supervisor_)
            return options_.base_url;
        auto port = supervisor_->start(cancel);
        if (!port.ok())
            return port.status();
        return "http://127.0.0.1:" + std::to_string(port.value());
    }
    Result<net::HttpRequest> prepare(std::string_view method, std::string_view path, std::string body,
                                     const CancellationToken &cancel) {
        auto base = endpoint(cancel);
        if (!base.ok())
            return base.status();
        auto url = net::parse_url(base.value());
        if (!url.ok())
            return url.status();
        net::HttpRequest request;
        request.method = method;
        request.host = url->host;
        request.port = url->port;
        request.target = (url->path == "/" ? "" : url->path) + std::string(path);
        request.body = std::move(body);
        request.connect_timeout = options_.connect_timeout;
        request.total_timeout = options_.request_timeout;
        request.use_tls = url->scheme == "https";
        request.tls.ca_bundle_path = options_.tls.ca_bundle_path;
        request.tls.pinned_sha256 = options_.tls.pinned_sha256;
        request.tls.pinned_cert_path = options_.tls.pinned_cert_path;
        request.tls.insecure_skip_verify = options_.tls.insecure_skip_verify;
        request.tls.server_name = options_.tls.server_name;
        request.tls.handshake_timeout = options_.tls.handshake_timeout;
        return request;
    }
    Result<std::string> request(std::string_view method, std::string_view path, std::string body) {
        auto prepared = prepare(method, path, std::move(body), {});
        if (!prepared.ok())
            return prepared.status();
        std::string response_body;
        const auto response =
            net::http_request_buffered(prepared.value(), response_body, {}, 4 * 1024 * 1024);
        if (!response.ok())
            return response.status();
        if (response->status != 200)
            return status_for_http(response->status);
        return response_body;
    }
    Result<GenerateStats> stream(std::string_view path, std::string body, const CancellationToken &cancel,
                                 const TokenCallback &callback, bool chat);

    // Served context and slot count: GET /props once per upstream instance
    // (endpoint, plus the launch argv in spawn mode, so a restart or an
    // auto_fit relaunch re-reads it). A missing or malformed /props caches
    // "unknown" (0) for that instance; options.context_length fills in n_ctx.
    llamaserver::ServerProps served_props(const CancellationToken &cancel) {
        auto ep = endpoint(cancel);
        llamaserver::ServerProps props;
        if (ep.ok()) {
            std::string key = ep.value();
            if (supervisor_) {
                for (const auto &arg : supervisor_->launch_arguments()) {
                    key += '\n';
                    key += arg;
                }
            }
            bool cached = false;
            {
                std::lock_guard<std::mutex> lock(props_mutex_);
                if (props_fetched_ && props_key_ == key) {
                    props = props_;
                    cached = true;
                }
            }
            if (!cached) {
                props = fetch_props(cancel);
                if (!cancel.cancelled()) {
                    std::lock_guard<std::mutex> lock(props_mutex_);
                    props_ = props;
                    props_key_ = std::move(key);
                    props_fetched_ = true;
                }
            }
        }
        if (props.n_ctx == 0)
            props.n_ctx = options_.context_length;
        return props;
    }
    // Request fields only llama.cpp understands; generic OpenAI mode gets none.
    llamaserver::RequestExtras extras(const llamaserver::ServerProps &props, std::string_view session_key,
                                      llamaserver::SlotAffinity::Lease &lease) {
        llamaserver::RequestExtras e;
        if (!options_.native_completion)
            return e;
        e.cache_prompt = true;
        if (options_.slot_affinity) {
            lease = slots_.acquire(session_key, props.total_slots);
            e.id_slot = lease.slot();
        }
        return e;
    }

  private:
    llamaserver::ServerProps fetch_props(const CancellationToken &cancel) {
        llamaserver::ServerProps props;
        auto prepared = prepare("GET", "/props", {}, cancel);
        if (!prepared.ok())
            return props;
        prepared->total_timeout = std::min(options_.request_timeout, std::chrono::milliseconds(30000));
        std::string body;
        const auto response = net::http_request_buffered(prepared.value(), body, cancel, 1024 * 1024);
        if (!response.ok() || response->status != 200)
            return props;
        auto parsed = json::parse(body);
        if (!parsed.ok() || !llamaserver::parse_props(parsed.value(), props).ok())
            return {};
        return props;
    }
    Status slot_action(std::uint32_t slot, const std::string &filename, std::string_view action) {
        if (!validation_.ok())
            return validation_;
        if (!options_.native_completion)
            return {ErrorCode::unsupported, "llamaserver: slot API requires native mode"};
        if (filename.empty() || filename == "." || filename == ".." || filename.size() > 255 ||
            filename.find_first_of("/\\:") != std::string::npos || filename.back() == '.' ||
            filename.back() == ' ' || std::any_of(filename.begin(), filename.end(), [](unsigned char c) {
                return c < 32 || c == 127;
            })) {
            return {ErrorCode::invalid_argument, "llamaserver: slot filename must be a basename"};
        }
        const auto response =
            request("POST", "/slots/" + std::to_string(slot) + "?action=" + std::string(action),
                    json::Value(json::Object{{"filename", filename}}).dump());
        if (!response.ok())
            return response.status();
        auto result = json::parse(response.value());
        if (!result.ok() || !result->is_object())
            return protocol_error("malformed slot response");
        if (result->find("error"))
            return {ErrorCode::backend_error, "llamaserver: slot operation failed"};
        return {};
    }
    LlamaServerBackendOptions options_;
    Status validation_;
    std::unique_ptr<llamaserver::Supervisor> supervisor_;
    std::mutex props_mutex_;
    bool props_fetched_ = false;
    std::string props_key_;
    llamaserver::ServerProps props_;
    llamaserver::SlotAffinity slots_;
};

class LlamaServerModel final : public BackendModel {
  public:
    LlamaServerModel(std::shared_ptr<LlamaServerBackendImpl> backend, ModelDescriptor descriptor)
        : backend_(std::move(backend)), descriptor_(std::move(descriptor)) {}
    const ModelDescriptor &descriptor() const override { return descriptor_; }
    bool has_native_chat() const override { return true; }
    Result<GenerateStats> generate(const GenerateRequest &request, const CancellationToken &cancel,
                                   const TokenCallback &callback) override {
        if (auto status = validate(request.sampling); !status.ok())
            return status;
        const auto props = backend_->served_props(cancel);
        if (auto status = llamaserver::validate_sampling(request.sampling, props.n_ctx); !status.ok())
            return status;
        const bool native = backend_->options().native_completion;
        llamaserver::SlotAffinity::Lease lease;
        auto body = llamaserver::build_completion_body(descriptor_.name, request, native,
                                                       backend_->options().grammar,
                                                       backend_->extras(props, {}, lease));
        return backend_->stream(native ? "/completion" : "/v1/completions", body.dump(), cancel, callback,
                                false);
    }
    Result<GenerateStats> chat(const ChatRequest &request, const CancellationToken &cancel,
                               const TokenCallback &callback) override {
        if (auto status = validate_chat_messages(request.messages); !status.ok())
            return status;
        if (auto status = validate(request.sampling); !status.ok())
            return status;
        const auto props = backend_->served_props(cancel);
        if (auto status = llamaserver::validate_sampling(request.sampling, props.n_ctx); !status.ok())
            return status;
        // The lease keeps the slot marked busy until the stream ends.
        llamaserver::SlotAffinity::Lease lease;
        auto body = llamaserver::build_chat_body(descriptor_.name, request, backend_->options().grammar,
                                                 backend_->extras(props, request.session_key, lease));
        return backend_->stream("/v1/chat/completions", body.dump(), cancel, callback, true);
    }

  private:
    std::shared_ptr<LlamaServerBackendImpl> backend_;
    ModelDescriptor descriptor_;
};

Result<std::shared_ptr<BackendModel>> LlamaServerBackendImpl::load_model(const ModelLoadOptions &options) {
    if (!validation_.ok())
        return validation_;
    if (options.model.empty())
        return Status(ErrorCode::invalid_argument, "llamaserver: model is required");
    // Verify readiness and the alias through discovery; never download or load
    // an unrelated model implicitly on a generic OpenAI upstream.
    auto models = list_models();
    if (!models.ok())
        return models.status();
    for (const auto &model : models.value()) {
        if (model.name == options.model) {
            // Readiness is proven: read the served context and slot count now,
            // so the first request does not pay for it.
            ModelDescriptor descriptor = model;
            descriptor.context_length = served_props({}).n_ctx;
            return std::shared_ptr<BackendModel>(
                std::make_shared<LlamaServerModel>(shared_from_this(), std::move(descriptor)));
        }
    }
    return Status(ErrorCode::not_found, "llamaserver: model is not advertised by upstream");
}

Result<GenerateStats> LlamaServerBackendImpl::stream(std::string_view path, std::string body,
                                                     const CancellationToken &cancel,
                                                     const TokenCallback &callback, bool chat) {
    auto prepared = prepare("POST", path, std::move(body), cancel);
    if (!prepared.ok())
        return prepared.status();
    auto request = std::move(prepared.value());
    int http_status = 0;
    request.on_status = [&](int status) { http_status = status; };
    const bool native = !chat && options_.native_completion;
    GenerateStats stats;
    Status error;
    bool stopped = false, terminal = false, done = false;
    bool usage_prompt = false, usage_completion = false, usage_cache = false;
    std::uint64_t observed_chunks = 0;
    const auto count = [&](const json::Value &object, const char *key, std::uint64_t &dest) {
        const auto *value = object.find(key);
        if (!value)
            return false;
        if (!value->is_integer() || value->as_double() < 0) {
            error = protocol_error(std::string("invalid count: ") + key);
            return false;
        }
        dest = value->as_uint();
        return true;
    };
    const auto frame = [&](std::string_view text) {
        if (cancel.cancelled())
            return false;
        if (text == "[DONE]") {
            if (!terminal) {
                error = protocol_error("DONE without a finish reason");
                return false;
            }
            done = true;
            return true;
        }
        if (done) {
            error = protocol_error("data after DONE");
            return false;
        }
        auto parsed = json::parse(text);
        if (!parsed.ok() || !parsed->is_object()) {
            error = protocol_error("malformed completion frame");
            return false;
        }
        const auto &object = parsed.value();
        if (const auto *upstream_error = object.find("error"); upstream_error && !upstream_error->is_null()) {
            error = {ErrorCode::backend_error, "llamaserver: upstream reported a completion error"};
            return false;
        }
        llamaserver::Timings timing;
        error = llamaserver::parse_timings(object, timing);
        if (!error.ok())
            return false;
        if (timing.prompt_present && !usage_prompt) {
            stats.prompt_tokens = timing.prompt_tokens + timing.cached_tokens;
            stats.token_counts_from_backend = true;
        }
        if (timing.completion_present && !usage_completion) {
            stats.completion_tokens = timing.completion_tokens;
            stats.token_counts_from_backend = true;
        }
        if (timing.cache_present && !usage_cache)
            stats.cached_tokens = timing.cached_tokens;
        if (timing.draft_present)
            stats.draft_tokens = timing.draft_tokens;
        if (timing.draft_accepted_present)
            stats.draft_accepted_tokens = timing.draft_accepted_tokens;
        if (timing.speed_present)
            stats.predicted_tokens_per_second = timing.predicted_tokens_per_second;
        if (timing.prompt_ns)
            stats.prompt_eval_ns = timing.prompt_ns;
        if (timing.eval_ns)
            stats.eval_ns = timing.eval_ns;

        if (const auto *usage = object.find("usage"); usage && !usage->is_null()) {
            if (!usage->is_object()) {
                error = protocol_error("usage must be an object");
                return false;
            }
            usage_prompt = count(*usage, "prompt_tokens", stats.prompt_tokens) || usage_prompt;
            usage_completion =
                count(*usage, "completion_tokens", stats.completion_tokens) || usage_completion;
            if (const auto *details = usage->find("prompt_tokens_details")) {
                if (!details->is_object()) {
                    error = protocol_error("prompt_tokens_details must be an object");
                    return false;
                }
                std::uint64_t cached = 0;
                if (count(*details, "cached_tokens", cached)) {
                    stats.cached_tokens = cached;
                    usage_cache = true;
                }
            }
            if (usage_prompt || usage_completion)
                stats.token_counts_from_backend = true;
        }
        std::string piece;
        std::string reason;
        if (native) {
            if (const auto *content = object.find("content")) {
                if (!content->is_string()) {
                    error = protocol_error("content must be text");
                    return false;
                }
                piece = content->as_string();
            }
            if (!usage_prompt && count(object, "tokens_evaluated", stats.prompt_tokens))
                stats.token_counts_from_backend = true;
            if (!usage_completion && count(object, "tokens_predicted", stats.completion_tokens))
                stats.token_counts_from_backend = true;
            if (const auto *cached = object.find("tokens_cached"); cached) {
                std::uint64_t value = 0;
                if (count(object, "tokens_cached", value) && !timing.cache_present && !usage_cache)
                    stats.cached_tokens = value;
            }
            const auto *stop = object.find("stop");
            if (stop && stop->is_bool() && stop->as_bool()) {
                terminal = true;
                const auto *type = object.find("stop_type");
                const auto is_true = [&](const char *key) {
                    const auto *v = object.find(key);
                    return v && v->is_bool() && v->as_bool();
                };
                if (is_true("stopped_limit"))
                    reason = "limit";
                else if (is_true("stopped_word"))
                    reason = "word";
                else
                    reason = type && type->is_string() ? type->as_string() : "eos";
            }
        } else if (const auto *choices = object.find("choices")) {
            if (!choices->is_array() || choices->as_array().size() > 1) {
                error = protocol_error("invalid choices");
                return false;
            }
            if (!choices->as_array().empty()) {
                const auto &choice = choices->as_array().front();
                const auto *content =
                    chat ? (choice.find("delta") ? choice.find("delta")->find("content") : nullptr)
                         : choice.find("text");
                if (content && !content->is_null()) {
                    if (!content->is_string()) {
                        error = protocol_error("completion content must be text");
                        return false;
                    }
                    piece = content->as_string();
                }
                if (const auto *finish = choice.find("finish_reason"); finish && !finish->is_null()) {
                    if (!finish->is_string()) {
                        error = protocol_error("invalid finish reason");
                        return false;
                    }
                    reason = finish->as_string();
                    terminal = true;
                }
            }
        }
        if (!error.ok())
            return false;
        if (!reason.empty()) {
            if (reason == "tool_calls" || reason == "function_call") {
                error = {ErrorCode::unsupported,
                         "llamaserver: tool calls are not represented by the text backend API"};
                return false;
            }
            const auto mapped = llamaserver::map_finish_reason(reason);
            if (mapped == StopReason::none) {
                error = protocol_error("unsupported finish reason");
                return false;
            }
            stats.stop_reason = mapped;
        }
        if (!piece.empty()) {
            ++observed_chunks;
            if (callback) {
                const bool keep = callback(TokenChunk{piece, stats.chunks});
                ++stats.chunks;
                if (!keep) {
                    stopped = true;
                    return false;
                }
            }
        }
        return true;
    };
    EventDecoder decoder(frame);
    const auto response = net::http_request(
        request,
        [&](std::string_view bytes) {
            if (http_status != 200)
                return false;
            return decoder.feed(bytes);
        },
        cancel);
    if (cancel.cancelled())
        return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
    if (http_status != 0 && http_status != 200)
        return status_for_http(http_status);
    if (!error.ok())
        return error;
    if (decoder.oversized())
        return protocol_error("completion frame exceeds 1 MiB");
    if (stopped) {
        stats.stop_reason = StopReason::callback;
        return stats;
    }
    if (!response.ok())
        return response.status();
    const bool finished = decoder.finish();
    if (cancel.cancelled())
        return Status(ErrorCode::cancelled, "llamaserver: request cancelled");
    if (!error.ok())
        return error;
    if (stopped) {
        stats.stop_reason = StopReason::callback;
        return stats;
    }
    if (!finished || !terminal)
        return protocol_error("stream ended without a valid completion");
    if (!stats.token_counts_from_backend)
        stats.completion_tokens = observed_chunks;
    return stats;
}

} // namespace

std::shared_ptr<LlamaServerBackend> make_llamaserver_backend(LlamaServerBackendOptions options) {
    return std::make_shared<LlamaServerBackendImpl>(std::move(options));
}

} // namespace sonder::inference
