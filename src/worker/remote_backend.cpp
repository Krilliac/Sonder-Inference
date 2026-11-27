#include "remote_backend.hpp"
#include "ipc.hpp"
#include "process.hpp"
#include "protocol.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <mutex>
#include <thread>

namespace sonder::inference::worker {
using namespace std::chrono_literals;
namespace {
Status lost() { return {ErrorCode::backend_error, "worker connection lost; request was not replayed"}; }
Status malformed() { return {ErrorCode::protocol_error, "worker sent an unexpected response"}; }
Result<json::Value> receive(Channel &channel, std::chrono::milliseconds timeout) {
    auto frame = channel.read_frame(timeout);
    if (!frame)
        return frame.status();
    return parse_message(frame.value());
}
Status send(Channel &channel, json::Value value) { return channel.write_frame(value.dump(), 5s); }
} // namespace

RestartPolicy::RestartPolicy(std::size_t limit, std::chrono::milliseconds initial, std::chrono::milliseconds maximum)
    : limit_(limit), initial_(initial), maximum_(maximum) {}
std::size_t RestartPolicy::recent_restarts(Clock::time_point now) {
    while (!restarts_.empty() && now - restarts_.front() >= 1h)
        restarts_.pop_front();
    return restarts_.size();
}
bool RestartPolicy::permit_restart(Clock::time_point now) {
    if (waiting_for_reap())
        return false;
    if (recent_restarts(now) >= limit_)
        return false;
    restarts_.push_back(now);
    ++attempts_;
    return true;
}
bool RestartPolicy::waiting_for_reap() const {
    return predecessor_ && !predecessor_->reaped.load();
}
std::chrono::milliseconds RestartPolicy::backoff() const {
    auto delay = initial_;
    for (std::size_t i = 1; i < attempts_ && delay < maximum_; ++i) {
        if (delay > maximum_ / 2)
            return maximum_;
        delay *= 2;
    }
    return std::min(delay, maximum_);
}

class RemoteBackend::Impl {
  public:
    explicit Impl(RemoteBackendOptions options)
        : options_(std::move(options)),
          policy_(options_.max_restarts_per_hour, options_.restart_initial_backoff, options_.restart_max_backoff) {}
    ~Impl() { stop(); }
    void start_monitor() {
        monitor_ = std::thread([this] { monitor(); });
    }
    void stop() {
        std::lock_guard stop_lock(stop_mutex_);
        {
            std::lock_guard lock(state_mutex_);
            shutdown_.cancel();
        }
        wake_.notify_all();
        if (monitor_.joinable())
            monitor_.join();
    }
    std::uint32_t pid() const { return pid_.load(); }
    std::size_t restarts() const { return restarts_.load(); }
    bool circuit() const { return circuit_.load(); }

    Result<json::Value> command(std::string_view type, const json::Object &payload) {
        const auto until = std::chrono::steady_clock::now() +
                           std::clamp(options_.request_timeout, 1ms, std::chrono::milliseconds(24h));
        std::unique_lock<std::timed_mutex> operation(operation_mutex_, std::defer_lock);
        while (!operation.try_lock_for(20ms)) {
            if (shutdown_.cancelled())
                return Status(ErrorCode::cancelled, "worker command cancelled before admission");
            if (std::chrono::steady_clock::now() >= until)
                return Status(ErrorCode::timeout, "worker command admission timed out");
        }
        if (shutdown_.cancelled())
            return Status(ErrorCode::cancelled, "worker command cancelled before admission");
        if (std::chrono::steady_clock::now() >= until)
            return Status(ErrorCode::timeout, "worker command admission timed out");
        // Admission is bounded separately from the documented startup wait.
        auto ch = ready({});
        if (!ch)
            return ch.status();
        const auto id = ++next_id_;
        auto request = message(type, id);
        for (const auto &[key, value] : payload)
            request.set(key, value);
        if (auto st = send(*ch.value(), std::move(request)); !st.ok()) {
            invalidate(ch.value());
            return lost();
        }
        auto response = receive(*ch.value(), options_.startup_timeout);
        if (!response || response->find("id")->as_uint() != id) {
            invalidate(ch.value());
            return response ? malformed() : lost();
        }
        if (response->find("type")->as_string() == "error") {
            auto st = status_from_message(response.value());
            if (st.code() == ErrorCode::protocol_error)
                invalidate(ch.value());
            return st;
        }
        if (response->find("type")->as_string() != "result" ||
            !fields(response.value(), {"v", "type", "id", "value"}).ok()) {
            invalidate(ch.value());
            return malformed();
        }
        const auto &value = *response->find("value");
        bool valid = true;
        if (type == "probe")
            valid = value.is_string() && value.as_string().size() <= 256;
        else if (type == "load") {
            auto descriptor = descriptor_from_json(value);
            const auto *requested = payload.find("model");
            valid = descriptor && requested && requested->is_string() && descriptor->name == requested->as_string();
        } else if (type == "list") {
            valid = value.is_array() && value.as_array().size() <= 1024;
            if (valid)
                for (const auto &model : value.as_array())
                    if (!descriptor_from_json(model)) {
                        valid = false;
                        break;
                    }
        }
        if (!valid) {
            invalidate(ch.value());
            return malformed();
        }
        return *response->find("value");
    }
    Result<GenerateStats> generate(const ModelLoadOptions &model, std::string_view kind, json::Value payload,
                                   const CancellationToken &cancel, const TokenCallback &callback) {
        using Clock = std::chrono::steady_clock;
        const auto deadline = Clock::now() + std::clamp(options_.request_timeout, 1ms, std::chrono::milliseconds(24h));
        std::unique_lock<std::timed_mutex> operation(operation_mutex_, std::defer_lock);
        while (!operation.try_lock_for(20ms)) {
            if (cancel.cancelled() || shutdown_.cancelled())
                return Status(ErrorCode::cancelled, "worker request cancelled before admission");
            if (Clock::now() >= deadline)
                return Status(ErrorCode::timeout, "worker request admission timed out");
        }
        if (cancel.cancelled())
            return Status(ErrorCode::cancelled, "worker request cancelled");
        auto channel = ready(cancel, deadline);
        if (!channel)
            return channel.status();
        if (Clock::now() >= deadline)
            return Status(ErrorCode::timeout, "worker request readiness timed out");
        const auto ch = channel.value();
        const auto *sampler = payload.find("sampling");
        if (!sampler)
            return Status(ErrorCode::invalid_argument, "missing worker sampling policy");
        auto sampling = sampling_from_json(*sampler);
        if (!sampling)
            return Status(ErrorCode::invalid_argument, sampling.status().message());
        const auto max_tokens = static_cast<std::uint64_t>(sampling->max_tokens);
        // max_tokens was checked against kMaxTokensLimit, so this product fits.
        const auto byte_limit = std::min(kMaxRequestOutputBytes, max_tokens * kMaxTokenPayloadBytes);
        const auto id = ++next_id_;
        auto request = message(kind, id);
        request.set("model", model.model);
        request.set("device", model.device_id);
        request.set("request", std::move(payload));
        const auto bytes = json::Value(std::move(request)).dump();
        if (bytes.size() > kMaxFrameBytes)
            return Status(ErrorCode::invalid_argument, "worker request exceeds frame limit");
        if (auto parsed = parse_message(bytes); !parsed)
            return Status(ErrorCode::invalid_argument, parsed.status().message());
        const auto io_timeout = [&] {
            return std::max(1ms,
                            std::min(std::chrono::milliseconds(5s),
                                     std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now())));
        };
        // Anchor initial silence at dispatch, before the worker can respond.
        // Creating the cancellation thread must not reset this deadline.
        auto last_activity = Clock::now();
        if (auto st = ch->write_frame(bytes, io_timeout(), cancel); !st.ok()) {
            invalidate(ch);
            return cancel.cancelled() ? Status(ErrorCode::cancelled, "worker request cancelled") : lost();
        }

        std::atomic<bool> done{false}, control_sent{false};
        std::atomic<Clock::duration::rep> control_started{0};
        std::atomic<bool> deadline_expired{false};
        std::mutex writer;
        // All credit transitions share writer with ACK/cancel dispatch. The
        // request itself grants the first credit, before any token is received.
        bool worker_has_credit = true;
        bool cancelled_credit = false;
        const auto grace_expired = [&] {
            return control_sent.load() &&
                   Clock::now() - Clock::time_point(Clock::duration(control_started.load())) >= options_.cancel_grace;
        };
        const auto failure = [&]() -> Status {
            if (cancel.cancelled() || shutdown_.cancelled())
                return {ErrorCode::cancelled, "worker request cancelled"};
            if (deadline_expired.load())
                return {ErrorCode::timeout, "worker request deadline or control grace exceeded"};
            return lost();
        };
        // writer held: timestamp publication precedes control_sent. Grace starts
        // at dispatch, never at the previous token or subsequent worker activity.
        const auto control = [&](std::string_view type) {
            if (control_sent.load())
                return true;
            cancelled_credit = worker_has_credit;
            worker_has_credit = false;
            control_started.store(Clock::now().time_since_epoch().count());
            control_sent.store(true);
            return ch
                ->write_frame(json::Value(message(type, id)).dump(), std::min(io_timeout(), options_.cancel_grace),
                              shutdown_.token())
                .ok();
        };
        // Even a worker flooding valid frames, ignoring cancel, or blocking a
        // write cannot defeat the absolute/grace deadlines. Consumer callbacks
        // remain trusted caller code; closing IPC cannot forcibly unwind them.
        std::thread canceller([&] {
            while (!done.load()) {
                if (Clock::now() >= deadline || grace_expired()) {
                    deadline_expired.store(true);
                    invalidate(ch);
                    return;
                }
                if (cancel.cancelled() || shutdown_.cancelled()) {
                    std::lock_guard lock(writer);
                    if (!done.load() && !control("cancel")) {
                        invalidate(ch);
                        return;
                    }
                }
                std::this_thread::sleep_for(5ms);
            }
        });
        struct Join {
            std::atomic<bool> &done;
            std::thread &thread;
            ~Join() {
                done.store(true);
                thread.join();
            }
        } join{done, canceller};
        std::uint64_t expected = 0, output_bytes = 0;
        bool callback_stop = false;
        for (;;) {
            if (Clock::now() >= deadline || grace_expired() || deadline_expired.load()) {
                deadline_expired.store(true);
                invalidate(ch);
                return failure();
            }
            if (!control_sent.load() && Clock::now() - last_activity >= options_.inactivity_timeout) {
                invalidate(ch);
                return Status(ErrorCode::timeout, "worker inactivity timeout");
            }
            auto response = receive(*ch, 50ms);
            if (!response) {
                if (response.status().code() == ErrorCode::timeout)
                    continue;
                invalidate(ch);
                return failure();
            }
            if (Clock::now() >= deadline || grace_expired() || deadline_expired.load()) {
                deadline_expired.store(true);
                invalidate(ch);
                return failure();
            }
            if (!control_sent.load() && Clock::now() - last_activity >= options_.inactivity_timeout) {
                invalidate(ch);
                return Status(ErrorCode::timeout, "worker inactivity timeout");
            }
            last_activity = Clock::now();
            if (response->find("id")->as_uint() != id) {
                invalidate(ch);
                return malformed();
            }
            const auto type = response->find("type")->as_string();
            if (type == "error") {
                done.store(true);
                auto st = status_from_message(response.value());
                if (st.code() == ErrorCode::protocol_error)
                    invalidate(ch);
                return st;
            }
            if (type == "done" && fields(response.value(), {"v", "type", "id", "stats"}).ok()) {
                done.store(true);
                auto stats = stats_from_json(*response->find("stats"));
                if (!stats || stats->chunks != expected || stats->completion_tokens != expected ||
                    expected > max_tokens || stats->stop_reason == StopReason::none ||
                    stats->stop_reason == StopReason::cancelled || stats->stop_reason == StopReason::error ||
                    (stats->stop_reason == StopReason::max_tokens && expected != max_tokens) ||
                    (callback_stop && stats->stop_reason != StopReason::callback) ||
                    (stats->stop_reason == StopReason::callback && !callback_stop) ||
                    (stats->matched_stop && (stats->stop_reason != StopReason::stop_sequence ||
                                             std::find(sampling->stop.begin(), sampling->stop.end(),
                                                       *stats->matched_stop) == sampling->stop.end()))) {
                    invalidate(ch);
                    return malformed();
                }
                if (cancel.cancelled() || shutdown_.cancelled() || (control_sent.load() && !callback_stop))
                    return Status(ErrorCode::cancelled, "worker request cancelled");
                // Counts used by Session come from observed frames, not unchecked
                // worker arithmetic. Timings remain bounded worker-reported diagnostics.
                stats->chunks = expected;
                stats->completion_tokens = expected;
                return stats;
            }
            if (type != "token" || !fields(response.value(), {"v", "type", "id", "chunk"}).ok() ||
                expected >= max_tokens) {
                invalidate(ch);
                return malformed();
            }
            auto chunk = chunk_from_json(*response->find("chunk"));
            if (!chunk || chunk->index != expected ||
                (!sampling->logprobs && (chunk->logprob.has_value() || !chunk->top_logprobs.empty())) ||
                chunk->top_logprobs.size() > static_cast<std::size_t>(sampling->top_logprobs)) {
                invalidate(ch);
                return malformed();
            }
            std::uint64_t token_bytes = 0;
            const auto account = [&](std::size_t size) {
                if (size > kMaxTokenPayloadBytes - token_bytes)
                    return false;
                token_bytes += static_cast<std::uint64_t>(size);
                return true;
            };
            bool bounded = account(chunk->text.size()) && account(chunk->reasoning.size()) &&
                           account(chunk->logprob ? sizeof(float) : 0);
            for (const auto &candidate : chunk->top_logprobs)
                bounded = bounded && account(candidate.token.size()) && account(sizeof(float));
            if (!bounded || token_bytes > byte_limit - output_bytes) {
                invalidate(ch);
                return malformed();
            }
            bool drain_cancelled_token = false;
            {
                std::lock_guard lock(writer);
                drain_cancelled_token = control_sent.load();
                if (drain_cancelled_token) {
                    // The peer can have sent its one credited token before it
                    // read our asynchronous cancel. Drain exactly that token;
                    // a cancel replacing an ACK never grants this allowance.
                    if (!cancelled_credit) {
                        invalidate(ch);
                        return malformed();
                    }
                    cancelled_credit = false;
                }
                worker_has_credit = false;
                output_bytes += token_bytes;
                ++expected;
            }
            if (drain_cancelled_token)
                continue; // no callback or ACK; terminal response still owes the fixed grace deadline
            bool keep = true;
            try {
                if (!cancel.cancelled() && callback)
                    keep = callback(chunk.value());
            } catch (...) {
                invalidate(ch);
                throw;
            }
            {
                std::lock_guard lock(writer);
                if (!keep || cancel.cancelled()) {
                    callback_stop = !keep;
                    if (!control(!keep && !cancel.cancelled() ? "stop" : "cancel")) {
                        invalidate(ch);
                        return failure();
                    }
                } else if (!control_sent.load()) {
                    auto ack = message("ack", id);
                    ack.set("index", chunk->index);
                    if (!ch->write_frame(json::Value(std::move(ack)).dump(), io_timeout(), shutdown_.token()).ok()) {
                        invalidate(ch);
                        return failure();
                    }
                    worker_has_credit = true;
                }
            }
            // Consumer backpressure is not worker silence. The independent
            // absolute/grace deadlines continue to run throughout the callback.
            last_activity = Clock::now();
        }
    }

  private:
    Result<std::shared_ptr<Channel>>
    ready(const CancellationToken &cancel,
          std::chrono::steady_clock::time_point request_deadline = std::chrono::steady_clock::time_point::max()) {
        std::unique_lock lock(state_mutex_);
        requested_ = true;
        wake_.notify_all();
        const auto until = std::min(request_deadline,
                                    std::chrono::steady_clock::now() +
                                        std::clamp(options_.startup_timeout, 1ms, std::chrono::milliseconds(60s)) +
                                        std::clamp(options_.restart_max_backoff, 1ms, std::chrono::milliseconds(60s)));
        while (!channel_ && !shutdown_.cancelled() && !cancel.cancelled()) {
            if (unreaped_.load())
                return Status(ErrorCode::unavailable, "worker predecessor is unreaped; replacement is blocked");
            if (circuit_.load())
                return Status(ErrorCode::unavailable, "worker restart circuit open (rolling hour cap)");
            if (!configuration_error_.ok())
                return configuration_error_;
            if (std::chrono::steady_clock::now() >= until)
                return Status(ErrorCode::timeout, "worker readiness timed out");
            wake_.wait_for(lock, 20ms);
        }
        if (shutdown_.cancelled() || cancel.cancelled())
            return Status(ErrorCode::cancelled, "worker stopped or request cancelled");
        return channel_;
    }
    void invalidate(const std::shared_ptr<Channel> &ch) {
        ch->close();
        std::lock_guard lock(state_mutex_);
        if (channel_ == ch)
            channel_.reset();
        wake_.notify_all();
    }
    bool pause(std::chrono::milliseconds delay) {
        std::unique_lock lock(state_mutex_);
        return !wake_.wait_for(lock, delay, [&] { return shutdown_.cancelled(); });
    }
    void monitor() {
#if !defined(SONDER_WORKER_TESTING)
        if (!options_.test_worker_mode.empty()) {
            std::lock_guard lock(state_mutex_);
            configuration_error_ = {ErrorCode::invalid_argument, "worker fault injection is test-only"};
            wake_.notify_all();
            return;
        }
#endif
        if (options_.executable.empty()) {
            auto executable = current_executable();
            if (executable)
                options_.executable = std::move(executable).value();
            else {
                std::lock_guard lock(state_mutex_);
                configuration_error_ = executable.status();
                wake_.notify_all();
                return;
            }
        }
        if (options_.engine != "mock" || options_.token_delay < 0ms || options_.token_delay > 60s ||
            options_.restart_initial_backoff <= 0ms ||
            options_.restart_max_backoff < options_.restart_initial_backoff || options_.restart_max_backoff > 60s ||
            options_.startup_timeout <= 0ms || options_.startup_timeout > 60s ||
            options_.max_restarts_per_hour > 1000 || options_.inactivity_timeout <= 0ms ||
            options_.inactivity_timeout > 24h || options_.cancel_grace <= 0ms || options_.cancel_grace > 60s ||
            options_.request_timeout <= 0ms || options_.request_timeout > 24h) {
            std::lock_guard lock(state_mutex_);
            configuration_error_ = {ErrorCode::invalid_argument, "invalid worker options"};
            wake_.notify_all();
            return;
        }
        bool first = true;
        while (!shutdown_.cancelled()) {
            {
                std::unique_lock lock(state_mutex_);
                wake_.wait(lock, [&] { return requested_ || shutdown_.cancelled(); });
            }
            if (shutdown_.cancelled())
                break;
            // Keep the reap barrier ahead of accounting and endpoint/spawn
            // creation, including after failed exec/handshake startup.
            unreaped_.store(policy_.waiting_for_reap());
            wake_.notify_all();
            while (!shutdown_.cancelled() && policy_.waiting_for_reap())
                if (!pause(20ms))
                    break;
            if (shutdown_.cancelled())
                break;
            unreaped_.store(false);
            if (!first) {
                if (!policy_.permit_restart(RestartPolicy::Clock::now())) {
                    circuit_.store(true);
                    wake_.notify_all();
                    while (!shutdown_.cancelled() &&
                           policy_.recent_restarts(RestartPolicy::Clock::now()) >= options_.max_restarts_per_hour)
                        if (!pause(100ms))
                            break;
                    if (shutdown_.cancelled())
                        break;
                    circuit_.store(false);
                    continue;
                }
                restarts_.fetch_add(1);
                if (!pause(policy_.backoff()))
                    break;
            }
            first = false;
            auto listener = make_listener();
            if (!listener)
                continue;
            std::vector<std::string> args{"worker",
                                          "--endpoint",
                                          listener.value()->endpoint(),
                                          "--engine",
                                          options_.engine,
                                          "--token-delay-ms",
                                          std::to_string(options_.token_delay.count())};
#if defined(SONDER_WORKER_TESTING)
            if (!options_.test_worker_mode.empty())
                args.push_back("--misbehave=" + options_.test_worker_mode);
#endif
            auto child = spawn_process(options_.executable, args, &policy_.predecessor());
            if (!child)
                continue;
            pid_.store(child.value()->pid());
            auto connected = listener.value()->accept(options_.startup_timeout, shutdown_.token());
            bool accepted = false;
            if (connected) {
                auto hello = receive(*connected.value(), options_.startup_timeout);
                if (hello && fields(hello.value(), {"v", "type", "id", "engine"}).ok() &&
                    hello->find("type")->as_string() == "hello" && hello->find("id")->as_uint() == 0 &&
                    hello->find("engine")->is_string() && hello->find("engine")->as_string() == options_.engine) {
                    auto response = message("ready", 0);
                    response.set("engine", options_.engine);
                    accepted = send(*connected.value(), std::move(response)).ok();
                }
            }
            if (accepted) {
                {
                    std::lock_guard lock(state_mutex_);
                    channel_ = connected.value();
                }
                wake_.notify_all();
                while (!shutdown_.cancelled() && child.value()->running()) {
                    {
                        std::lock_guard lock(state_mutex_);
                        if (channel_ != connected.value())
                            break;
                    }
                    // No response is legal between operations. This also catches
                    // delayed post-terminal tokens without ever delivering them
                    // to another request. Active requests own the sole reader.
                    {
                        std::unique_lock<std::timed_mutex> idle(operation_mutex_, std::try_to_lock);
                        if (idle.owns_lock()) {
                            auto unsolicited = connected.value()->read_frame(1ms, shutdown_.token());
                            if (unsolicited || unsolicited.status().code() != ErrorCode::timeout) {
                                invalidate(connected.value());
                                break;
                            }
                        }
                    }
                    if (!pause(20ms))
                        break;
                }
                // A shutdown frame lets idle workers exit before forced reap.
                if (shutdown_.cancelled()) {
                    // Request ownership also covers its ACK/cancel writer.
                    // Give cancellation a bounded grace interval, then send
                    // shutdown only with exclusive ownership of the channel.
                    std::unique_lock<std::timed_mutex> operation(operation_mutex_, std::defer_lock);
                    if (operation.try_lock_for(options_.cancel_grace))
                        (void)send(*connected.value(), message("shutdown", 0));
                }
                invalidate(connected.value());
            } else if (connected)
                connected.value()->close();
            child.value()->stop(1s);
            pid_.store(0);
            wake_.notify_all();
        }
    }
    RemoteBackendOptions options_;
    RestartPolicy policy_; // monitor-only
    mutable std::mutex state_mutex_;
    std::mutex stop_mutex_;
    std::condition_variable wake_;
    std::timed_mutex operation_mutex_;
    CancellationSource shutdown_;
    std::thread monitor_;
    std::shared_ptr<Channel> channel_; // state_mutex_
    bool requested_ = false;
    Status configuration_error_;
    std::uint64_t next_id_ = 0; // operation_mutex_
    std::atomic<std::uint32_t> pid_{0};
    std::atomic<std::size_t> restarts_{0};
    std::atomic<bool> circuit_{false};
    std::atomic<bool> unreaped_{false};
};

namespace {
// Immutable descriptor/load options; shared implementation keeps requests safe
// if a Model outlives its Backend. Each request is re-bound after worker
// restart.
class RemoteModel final : public BackendModel {
  public:
    RemoteModel(ModelDescriptor descriptor,
                std::function<Result<GenerateStats>(std::string_view, json::Value, const CancellationToken &,
                                                    const TokenCallback &)>
                    execute)
        : descriptor_(std::move(descriptor)), execute_(std::move(execute)) {}
    const ModelDescriptor &descriptor() const override { return descriptor_; }
    bool has_native_chat() const override { return true; }
    Result<GenerateStats> generate(const GenerateRequest &r, const CancellationToken &c,
                                   const TokenCallback &cb) override {
        auto value = generate_to_json(r);
        if (!value)
            return value.status();
        return execute_("generate", std::move(value).value(), c, cb);
    }
    Result<GenerateStats> chat(const ChatRequest &r, const CancellationToken &c, const TokenCallback &cb) override {
        auto value = chat_to_json(r);
        if (!value)
            return value.status();
        return execute_("chat", std::move(value).value(), c, cb);
    }

  private:
    const ModelDescriptor descriptor_;
    const std::function<Result<GenerateStats>(std::string_view, json::Value, const CancellationToken &,
                                              const TokenCallback &)>
        execute_;
};
} // namespace

RemoteBackend::RemoteBackend(RemoteBackendOptions options) : impl_(std::make_shared<Impl>(std::move(options))) {
    impl_->start_monitor();
}
RemoteBackend::~RemoteBackend() { impl_->stop(); }
std::string RemoteBackend::name() const { return "remote-mock"; }
std::string RemoteBackend::description() const { return "Isolated synthetic mock worker (no model inference)"; }
BackendCapabilities RemoteBackend::capabilities() const { return advertised_capabilities(); }
BackendCapabilities RemoteBackend::advertised_capabilities() {
    return BackendCapabilities{}
        .add(Capability::streaming)
        .add(Capability::remote_process)
        .add(Capability::deterministic)
        .add(Capability::logprobs);
}
Result<std::string> RemoteBackend::probe() {
    auto result = impl_->command("probe", {});
    if (!result)
        return result.status();
    if (!result->is_string())
        return malformed();
    return result->as_string();
}
Result<std::vector<ModelDescriptor>> RemoteBackend::list_models() {
    auto result = impl_->command("list", {});
    if (!result)
        return result.status();
    if (!result->is_array() || result->as_array().size() > 1024)
        return malformed();
    std::vector<ModelDescriptor> models;
    for (const auto &value : result->as_array()) {
        auto d = descriptor_from_json(value);
        if (!d)
            return d.status();
        models.push_back(std::move(d).value());
    }
    return models;
}
Result<std::shared_ptr<BackendModel>> RemoteBackend::load_model(const ModelLoadOptions &options) {
    if (options.model.empty() || options.model.size() > 4096 || options.model.find('\0') != std::string::npos ||
        options.device_id.size() > 4096 || options.device_id.find('\0') != std::string::npos)
        return Status(ErrorCode::invalid_argument, "worker model identity exceeds protocol limits");
    auto result = impl_->command("load", {{"model", options.model}, {"device", options.device_id}});
    if (!result)
        return result.status();
    auto descriptor = descriptor_from_json(result.value());
    if (!descriptor)
        return descriptor.status();
    auto execute = [impl = impl_, options](std::string_view kind, json::Value value, const CancellationToken &cancel,
                                           const TokenCallback &callback) {
        return impl->generate(options, kind, std::move(value), cancel, callback);
    };
    return std::shared_ptr<BackendModel>(new RemoteModel(std::move(descriptor).value(), std::move(execute)));
}
std::uint32_t RemoteBackend::worker_pid() const { return impl_->pid(); }
std::size_t RemoteBackend::restart_count() const { return impl_->restarts(); }
bool RemoteBackend::circuit_open() const { return impl_->circuit(); }
std::shared_ptr<Backend> make_remote_backend(RemoteBackendOptions options) {
    return std::make_shared<RemoteBackend>(std::move(options));
}
} // namespace sonder::inference::worker
