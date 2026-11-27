// Ollama Cloud provider: Backend/BackendModel over a pluggable HTTPS
// transport, with the opt-in cloud allowlist, local-only refusal, API key
// redaction and remote-provider telemetry (docs/integration/ollama-cloud.md).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "backends/ollama_cloud/ollama_cloud_protocol.hpp"
#include "backends/ollama_cloud/ollama_cloud_secret.hpp"
#include "net/http_client.hpp"
#include "net/tls.hpp"
#include "sonder/inference/backend_runtime.hpp"
#include "sonder/inference/backends/ollama_cloud.hpp"

namespace sonder::inference {

namespace {

using Clock = std::chrono::steady_clock;
using ollama_cloud::ApiKey;
using ollama_cloud::BaseUrl;
using ollama_cloud::StreamDecoder;
using ollama_cloud::StreamEvent;
using ollama_cloud::StreamKind;
using ollama_cloud::TagEntry;

constexpr std::size_t kMaxQueuedEvents = 256;
constexpr const char* kProvider = "ollama_cloud";
constexpr const char* kLocality = "cloud";

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

std::uint64_t to_u64(std::int64_t v) { return v > 0 ? static_cast<std::uint64_t>(v) : 0u; }

StopReason map_done_reason(const std::string& reason) {
    if (reason == "length") {
        return StopReason::max_tokens;
    }
    if (reason == "stop") {
        return StopReason::end_of_sequence;
    }
    return StopReason::none;
}

// ---------------------------------------------------------------------------
// Default transport: the internal HTTP client over verified TLS.

class HttpsTransport final : public OllamaCloudTransport {
public:
    Status send(const OllamaCloudHttpRequest& request, const std::function<void(int)>& on_status,
                const std::function<bool(std::string_view)>& on_body, const CancellationToken& cancel) override {
        if constexpr (!net::kTlsCompiledIn) {
            (void)request;
            (void)on_status;
            (void)on_body;
            (void)cancel;
            return Status(ErrorCode::unsupported,
                          "ollama_cloud: this build has no TLS; configure with -DSONDER_WITH_TLS=ON");
        } else {
            net::HttpRequest r;
            r.method = request.method;
            r.host = request.host;
            r.port = request.port;
            r.target = request.target;
            r.body = request.body;
            r.content_type = "application/json";
            r.connect_timeout = request.connect_timeout;
            r.total_timeout = request.total_timeout;
            r.idle_timeout = request.idle_timeout;
            r.use_tls = true;
            r.tls.ca_bundle_path = request.ca_bundle_path;
            // Verification is never relaxed: no insecure mode, no pin-only mode
            // (which would skip chain and host-name checks), no SNI override.
            r.tls.insecure_skip_verify = false;
            r.tls.pinned_sha256.clear();
            r.tls.pinned_cert_path.clear();
            r.tls.server_name.clear();
            r.headers = request.headers;
            r.on_status = on_status;
            auto res = net::http_request(r, on_body, cancel);
            for (auto& header : r.headers) {
                ollama_cloud::secure_wipe(header.second);
            }
            if (!res.ok()) {
                return res.status();
            }
            return Status::success();
        }
    }
};

// ---------------------------------------------------------------------------
// Shared, immutable-after-construction provider state. Counters are atomics
// and the event queue is mutex-guarded, so every member function is safe to
// call from any thread.

struct RequestRecord {
    std::string endpoint;
    std::string model;
    std::string outcome;  // completed, failed, cancelled, refused
    int http_status = 0;
    ErrorCode error = ErrorCode::ok;
    double wall_ms = 0.0;
    double ttfb_ms = -1.0;
    double ttft_ms = -1.0;
    bool has_tokens = false;
    std::uint64_t prompt_tokens = 0;
    std::uint64_t completion_tokens = 0;
};

class CloudState {
public:
    CloudState(OllamaCloudOptions options, BaseUrl base, ApiKey key)
        : options_(std::move(options)), base_(std::move(base)), key_(std::move(key)) {
        transport_ = options_.transport ? options_.transport : make_ollama_cloud_https_transport();
        options_.transport.reset();
        options_.env_lookup = nullptr;
    }

    [[nodiscard]] const OllamaCloudOptions& options() const noexcept { return options_; }
    [[nodiscard]] const BaseUrl& base() const noexcept { return base_; }

    [[nodiscard]] bool is_cloud_model(const std::string& name) const {
        return std::find(options_.models.begin(), options_.models.end(), name) != options_.models.end();
    }

    // One HTTPS exchange. The Authorization header is built here and wiped
    // after the call; the returned status is scrubbed of the key.
    Status send(const std::string& method, std::string_view api_path, std::string body,
                const std::function<void(int)>& on_status, const std::function<bool(std::string_view)>& on_body,
                const CancellationToken& cancel, std::chrono::milliseconds total_timeout,
                std::chrono::milliseconds idle_timeout) const {
        OllamaCloudHttpRequest req;
        req.method = method;
        req.host = base_.host;
        req.port = base_.port;
        req.target = ollama_cloud::join_target(base_.path, api_path);
        req.body = std::move(body);
        req.connect_timeout = options_.connect_timeout;
        req.total_timeout = total_timeout;
        req.idle_timeout = idle_timeout;
        req.ca_bundle_path = options_.ca_bundle_path;
        req.headers.emplace_back("Authorization", key_.bearer_header_value());
        Status st = transport_->send(req, on_status, on_body, cancel);
        for (auto& header : req.headers) {
            ollama_cloud::secure_wipe(header.second);
        }
        return key_.scrub(st);
    }

    // GET with a buffered, bounded body. Non-2xx maps through http_status_error.
    Result<std::string> get_buffered(std::string_view api_path) const {
        int status = 0;
        std::string body;
        bool overflow = false;
        Status st = send(
            "GET", api_path, {}, [&status](int s) { status = s; },
            [&](std::string_view data) {
                if (body.size() + data.size() > ollama_cloud::kMaxBufferedBodyBytes) {
                    overflow = true;
                    return false;
                }
                body.append(data);
                return true;
            },
            {}, std::min(options_.request_timeout, std::chrono::milliseconds(60000)), std::chrono::milliseconds(0));
        if (!st.ok()) {
            return st;
        }
        if (overflow) {
            return Status(ErrorCode::protocol_error, "ollama_cloud: response body exceeds 16 MiB");
        }
        if (status < 200 || status >= 300) {
            return key_.scrub(ollama_cloud::http_status_error(status, body, scrubber()));
        }
        return body;
    }

    Result<std::vector<TagEntry>> tags() const {
        auto body = get_buffered("/api/tags");
        if (!body.ok()) {
            return body.status();
        }
        auto parsed = ollama_cloud::parse_tags(body.value());
        if (!parsed.ok()) {
            return key_.scrub(parsed.status());
        }
        return parsed;
    }

    [[nodiscard]] Status scrub(const Status& st) const { return key_.scrub(st); }
    // Redacts the key in upstream text before the protocol layer bounds it.
    // Captures `this`: use only while this CloudState is alive (callers hold
    // the shared_ptr for the duration of the request).
    [[nodiscard]] ollama_cloud::TextScrubber scrubber() const {
        return [this](std::string& text) { key_.scrub(text); };
    }

    void record(const RequestRecord& r) {
        requests_.fetch_add(1, std::memory_order_relaxed);
        if (r.outcome == "failed") {
            failures_.fetch_add(1, std::memory_order_relaxed);
        } else if (r.outcome == "cancelled") {
            cancelled_.fetch_add(1, std::memory_order_relaxed);
        } else if (r.outcome == "refused") {
            refused_local_only_.fetch_add(1, std::memory_order_relaxed);
        }
        json::Object attrs{{"provider", kProvider},
                           {"locality", kLocality},
                           {"remote", true},
                           {"host", base_.host},
                           {"endpoint", r.endpoint},
                           {"model", r.model},
                           {"outcome", r.outcome},
                           {"wall_ms", r.wall_ms}};
        if (r.http_status != 0) {
            attrs.set("http_status", r.http_status);
        }
        if (r.error != ErrorCode::ok) {
            // The code only: upstream messages never reach telemetry.
            attrs.set("error_code", to_string(r.error));
        }
        if (r.ttfb_ms >= 0) {
            attrs.set("ttfb_ms", r.ttfb_ms);
        }
        if (r.ttft_ms >= 0) {
            attrs.set("ttft_ms", r.ttft_ms);
        }
        if (r.has_tokens) {
            attrs.set("prompt_tokens", r.prompt_tokens);
            attrs.set("completion_tokens", r.completion_tokens);
        }
        std::lock_guard<std::mutex> lock(events_mu_);
        if (events_.size() >= kMaxQueuedEvents) {
            ++dropped_events_;  // drop-and-count; never block the request path
            return;
        }
        events_.emplace_back("backend.remote.request", std::move(attrs));
    }

    std::vector<std::pair<std::string, json::Object>> drain_events() {
        std::vector<std::pair<std::string, json::Object>> out;
        std::lock_guard<std::mutex> lock(events_mu_);
        out.reserve(events_.size());
        while (!events_.empty()) {
            out.push_back(std::move(events_.front()));
            events_.pop_front();
        }
        return out;
    }

    [[nodiscard]] json::Object diagnostics() const {
        json::Object d = redacted_config(options_);
        d.set("provider", kProvider);
        d.set("locality", kLocality);
        d.set("remote", true);
        d.set("tls_compiled_in", net::kTlsCompiledIn);
        d.set("requests", requests_.load(std::memory_order_relaxed));
        d.set("failures", failures_.load(std::memory_order_relaxed));
        d.set("cancelled", cancelled_.load(std::memory_order_relaxed));
        d.set("refused_local_only", refused_local_only_.load(std::memory_order_relaxed));
        std::uint64_t dropped = 0;
        {
            std::lock_guard<std::mutex> lock(events_mu_);
            dropped = dropped_events_;
        }
        d.set("telemetry_dropped", dropped);
        return d;
    }

private:
    OllamaCloudOptions options_;
    BaseUrl base_;
    ApiKey key_;
    std::shared_ptr<OllamaCloudTransport> transport_;
    std::atomic<std::uint64_t> requests_{0};
    std::atomic<std::uint64_t> failures_{0};
    std::atomic<std::uint64_t> cancelled_{0};
    std::atomic<std::uint64_t> refused_local_only_{0};
    mutable std::mutex events_mu_;
    std::deque<std::pair<std::string, json::Object>> events_;
    std::uint64_t dropped_events_ = 0;
};

Status local_only_refusal(const std::string& model) {
    return Status(ErrorCode::invalid_argument,
                  "ollama_cloud: model '" + model +
                      "' runs on Ollama Cloud (remote) and the request is local-only (X-Sonder-Local-Only)");
}

// ---------------------------------------------------------------------------

class OllamaCloudModel final : public BackendModel {
public:
    OllamaCloudModel(std::shared_ptr<CloudState> state, ModelDescriptor descriptor)
        : state_(std::move(state)), descriptor_(std::move(descriptor)) {}

    const ModelDescriptor& descriptor() const override { return descriptor_; }
    bool has_native_chat() const override { return true; }

    Result<GenerateStats> generate(const GenerateRequest& request, const CancellationToken& cancel,
                                   const TokenCallback& on_chunk) override {
        const auto start = Clock::now();
        if (request.local_only) {
            return refuse("/api/generate", start);
        }
        if (Status st = common_refusals(request.sampling, request.grammar); !st.ok()) {
            return st;
        }
        return stream(StreamKind::generate, "/api/generate",
                      ollama_cloud::build_generate_body(descriptor_.name, request), cancel, on_chunk, false, start);
    }

    Result<GenerateStats> chat(const ChatRequest& request, const CancellationToken& cancel,
                               const TokenCallback& on_chunk) override {
        const auto start = Clock::now();
        if (request.local_only) {
            return refuse("/api/chat", start);
        }
        if (Status st = validate_chat_messages(request.messages); !st.ok()) {
            return st;
        }
        if (Status st = common_refusals(request.sampling, request.grammar); !st.ok()) {
            return st;
        }
        if (request.response_format_type && *request.response_format_type != "text") {
            // https://docs.ollama.com/cloud: "Cloud models currently do not
            // support structured outputs."
            return Status(ErrorCode::unsupported,
                          "ollama_cloud: cloud models do not support structured outputs (response_format)");
        }
        if (has_chat_media(request.messages)) {
            return Status(ErrorCode::unsupported, "ollama_cloud: image inputs are not supported by this provider");
        }
        return stream(StreamKind::chat, "/api/chat", ollama_cloud::build_chat_body(descriptor_.name, request), cancel,
                      on_chunk, request.separate_reasoning, start);
    }

private:
    static Status common_refusals(const SamplingConfig& sampling, const std::string& grammar) {
        if (!sampling.logit_bias.empty()) {
            return Status(ErrorCode::invalid_argument, "the ollama_cloud backend does not support logit_bias");
        }
        if (sampling.typical_p != 1.0f) {
            return Status(ErrorCode::invalid_argument, "the ollama_cloud backend does not support typical_p");
        }
        if (!grammar.empty()) {
            return Status(ErrorCode::unsupported, "the ollama_cloud backend does not support GBNF grammar constraints");
        }
        return Status::success();
    }

    Status refuse(const char* endpoint, Clock::time_point start) {
        RequestRecord r;
        r.endpoint = endpoint;
        r.model = descriptor_.name;
        r.outcome = "refused";
        r.error = ErrorCode::invalid_argument;
        r.wall_ms = ms_since(start);
        state_->record(r);
        return local_only_refusal(descriptor_.name);
    }

    Result<GenerateStats> stream(StreamKind kind, const char* endpoint, std::string body,
                                 const CancellationToken& cancel, const TokenCallback& on_chunk,
                                 bool separate_reasoning, Clock::time_point start) {
        RequestRecord rec;
        rec.endpoint = endpoint;
        rec.model = descriptor_.name;
        const auto finish = [&](Result<GenerateStats> result) -> Result<GenerateStats> {
            rec.wall_ms = ms_since(start);
            if (result.ok()) {
                rec.outcome = "completed";
            } else if (result.status().code() == ErrorCode::cancelled) {
                rec.outcome = "cancelled";
                rec.error = ErrorCode::cancelled;
            } else {
                rec.outcome = "failed";
                rec.error = result.status().code();
            }
            state_->record(rec);
            return result;
        };
        if (cancel.cancelled()) {
            return finish(Status(ErrorCode::cancelled, "ollama_cloud: request cancelled"));
        }
        const bool emit_thinking = state_->options().emit_thinking_chunks;
        std::uint64_t index = 0;
        bool stopped_by_callback = false;
        StreamDecoder decoder(
            kind,
            [&](const StreamEvent& ev) {
                if (cancel.cancelled()) {
                    return false;
                }
                if (rec.ttft_ms < 0) {
                    rec.ttft_ms = ms_since(start);
                }
                if (!on_chunk) {
                    return true;
                }
                if (!ev.tool_calls.empty()) {
                    TokenChunk tc;
                    tc.index = index++;
                    tc.tool_call_deltas = ev.tool_calls;
                    if (!on_chunk(tc)) {
                        stopped_by_callback = true;
                        return false;
                    }
                }
                const auto deliver = [&](std::string_view text, std::string_view reasoning) {
                    if ((text.empty() && reasoning.empty()) || stopped_by_callback) {
                        return;
                    }
                    if (!on_chunk(TokenChunk{text, index++, reasoning})) {
                        stopped_by_callback = true;
                    }
                };
                if (separate_reasoning) {
                    deliver(ev.content, ev.thinking);
                } else {
                    if (emit_thinking) {
                        deliver(ev.thinking, {});
                    }
                    deliver(ev.content, {});
                }
                return !stopped_by_callback;
            },
            state_->options().max_stream_output_bytes, state_->scrubber());

        int http_status = 0;
        std::string error_body;
        Status st = state_->send(
            "POST", endpoint, std::move(body), [&http_status](int s) { http_status = s; },
            [&](std::string_view data) {
                if (rec.ttfb_ms < 0) {
                    rec.ttfb_ms = ms_since(start);
                }
                if (http_status < 200 || http_status >= 300) {
                    if (error_body.size() < ollama_cloud::kMaxErrorBodyBytes) {
                        error_body.append(data.substr(0, ollama_cloud::kMaxErrorBodyBytes - error_body.size()));
                    }
                    return true;
                }
                return decoder.feed(data);
            },
            cancel, state_->options().request_timeout, state_->options().idle_timeout);
        rec.http_status = http_status;
        if (cancel.cancelled()) {
            return finish(Status(ErrorCode::cancelled, "ollama_cloud: request cancelled"));
        }
        if (!st.ok()) {
            return finish(st);
        }
        if (http_status < 200 || http_status >= 300) {
            return finish(
                state_->scrub(ollama_cloud::http_status_error(http_status, error_body, state_->scrubber())));
        }
        if (Status fin = decoder.finish(); !fin.ok()) {
            return finish(state_->scrub(fin));
        }
        const auto& sum = decoder.summary();
        GenerateStats stats;
        stats.chunks = index;
        if (sum.has_timings) {
            stats.prompt_tokens = to_u64(sum.prompt_eval_count);
            stats.completion_tokens = to_u64(sum.eval_count);
            stats.load_ns = to_u64(sum.load_duration_ns);
            stats.prompt_eval_ns = to_u64(sum.prompt_eval_duration_ns);
            stats.eval_ns = to_u64(sum.eval_duration_ns);
            stats.token_counts_from_backend = true;
            if (sum.prompt_eval_cached_count) {
                stats.cached_tokens = to_u64(*sum.prompt_eval_cached_count);
            }
            rec.has_tokens = true;
            rec.prompt_tokens = stats.prompt_tokens;
            rec.completion_tokens = stats.completion_tokens;
        } else {
            stats.completion_tokens = sum.content_events;
        }
        if (stopped_by_callback || decoder.stopped_by_callback()) {
            stats.stop_reason = StopReason::callback;
        } else if (sum.saw_tool_calls) {
            stats.stop_reason = StopReason::tool_calls;
        } else {
            stats.stop_reason = map_done_reason(sum.done_reason);
        }
        return finish(stats);
    }

    std::shared_ptr<CloudState> state_;
    ModelDescriptor descriptor_;
};

class OllamaCloudBackend final : public Backend {
public:
    explicit OllamaCloudBackend(std::shared_ptr<CloudState> state) : state_(std::move(state)) {}

    std::string name() const override { return kOllamaCloudBackendName; }
    std::string description() const override {
        return "Ollama Cloud (remote provider) at https://" + state_->base().host;
    }
    BackendCapabilities capabilities() const override {
        BackendCapabilities caps;
        caps.add(Capability::streaming).add(Capability::remote_process).add(Capability::tool_calling);
        return caps;
    }

    // GET /api/tags: reachability of the cloud endpoint over verified TLS.
    Result<std::string> probe() override {
        auto tags = state_->tags();
        if (!tags.ok()) {
            return tags.status();
        }
        return std::string("ollama-cloud (") + std::to_string(tags.value().size()) + " models listed)";
    }

    // The configured cloud models that the cloud currently lists.
    Result<std::vector<ModelDescriptor>> list_models() override {
        auto tags = state_->tags();
        if (!tags.ok()) {
            return tags.status();
        }
        std::vector<ModelDescriptor> out;
        for (const auto& name : state_->options().models) {
            for (const auto& t : tags.value()) {
                if (t.name == name) {
                    out.push_back(descriptor_from(t));
                    break;
                }
            }
        }
        return out;
    }

    Result<std::shared_ptr<BackendModel>> load_model(const ModelLoadOptions& options) override {
        if (options.model.empty()) {
            return Status(ErrorCode::invalid_argument, "ollama_cloud: model name is required");
        }
        if (!state_->is_cloud_model(options.model)) {
            return Status(ErrorCode::invalid_argument,
                          "ollama_cloud: model '" + options.model +
                              "' is not configured as a cloud model (add it to the ollama_cloud 'models' list)");
        }
        auto tags = state_->tags();
        if (!tags.ok()) {
            return tags.status();
        }
        for (const auto& t : tags.value()) {
            if (t.name == options.model) {
                return std::shared_ptr<BackendModel>(std::make_shared<OllamaCloudModel>(state_, descriptor_from(t)));
            }
        }
        return Status(ErrorCode::not_found, "ollama_cloud: model '" + options.model + "' is not offered by " +
                                                state_->base().host + " (GET /api/tags)");
    }

    std::optional<BackendRuntimeStatus> runtime_status() const override {
        BackendRuntimeStatus st;
        st.gpu_memory.probe = "none";
        st.gpu_memory.status = "unsupported";
        st.gpu_memory.error = "remote provider (ollama_cloud): no local GPU";
        st.diagnostics = state_->diagnostics();
        st.take_events = [state = state_] { return state->drain_events(); };
        return st;
    }

private:
    static ModelDescriptor descriptor_from(const TagEntry& t) {
        ModelDescriptor d;
        d.name = t.name;
        d.backend = kOllamaCloudBackendName;
        d.format = t.format;
        d.family = t.family;
        d.parameter_size = t.parameter_size;
        d.quantization = t.quantization;
        d.size_bytes = t.size_bytes;
        d.resident = false;  // weights live on the provider
        return d;
    }

    std::shared_ptr<CloudState> state_;
};

}  // namespace

std::shared_ptr<OllamaCloudTransport> make_ollama_cloud_https_transport() {
    return std::make_shared<HttpsTransport>();
}

json::Object redacted_config(const OllamaCloudOptions& options) {
    json::Array models;
    for (const auto& m : options.models) {
        models.emplace_back(m);
    }
    json::Object key;
    if (!options.api_key_env.empty()) {
        key = json::Object{{"source", "env"}, {"name", options.api_key_env}};
    } else if (!options.api_key_file.empty()) {
        key = json::Object{{"source", "file"}, {"path", options.api_key_file}};
    } else {
        key = json::Object{{"source", "none"}};
    }
    json::Object out{{"base_url", options.base_url},
                     {"api_key", std::move(key)},
                     {"models", std::move(models)},
                     {"connect_timeout_ms", static_cast<std::int64_t>(options.connect_timeout.count())},
                     {"request_timeout_ms", static_cast<std::int64_t>(options.request_timeout.count())},
                     {"idle_timeout_ms", static_cast<std::int64_t>(options.idle_timeout.count())},
                     {"tls_verify", true}};
    if (!options.ca_bundle_path.empty()) {
        out.set("ca_bundle_path", options.ca_bundle_path);
    }
    return out;
}

Result<std::shared_ptr<Backend>> make_ollama_cloud_backend(OllamaCloudOptions options) {
    auto base = ollama_cloud::parse_base_url(options.base_url);
    if (!base.ok()) {
        return base.status();
    }
    const bool has_env = !options.api_key_env.empty();
    const bool has_file = !options.api_key_file.empty();
    if (has_env == has_file) {
        return Status(ErrorCode::invalid_argument,
                      "ollama_cloud: set exactly one of api_key_env (a variable name) or api_key_file (a path)");
    }
    std::set<std::string> seen;
    for (const auto& m : options.models) {
        if (m.empty() || m.size() > 256 ||
            std::any_of(m.begin(), m.end(), [](unsigned char c) { return c < 0x21 || c >= 0x7f; })) {
            return Status(ErrorCode::invalid_argument, "ollama_cloud: model names must be 1-256 printable ASCII bytes");
        }
        if (!seen.insert(m).second) {
            return Status(ErrorCode::invalid_argument, "ollama_cloud: duplicate model '" + m + "'");
        }
    }
    if (options.connect_timeout.count() <= 0 || options.request_timeout.count() <= 0 ||
        options.idle_timeout.count() < 0) {
        return Status(ErrorCode::invalid_argument,
                      "ollama_cloud: connect_timeout and request_timeout must be positive, idle_timeout >= 0");
    }
    auto key = has_env ? ApiKey::from_env(options.api_key_env, options.env_lookup)
                       : ApiKey::from_file(options.api_key_file);
    if (!key.ok()) {
        return key.status();
    }
    auto state = std::make_shared<CloudState>(std::move(options), std::move(base).value(), std::move(key).value());
    return std::shared_ptr<Backend>(std::make_shared<OllamaCloudBackend>(std::move(state)));
}

}  // namespace sonder::inference
