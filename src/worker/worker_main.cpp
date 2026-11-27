#include "ipc.hpp"
#include "protocol.hpp"
#include "remote_backend.hpp"

#include <atomic>
#include <charconv>
#include <condition_variable>
#include <mutex>
#include <thread>

#include "sonder/inference/backends.hpp"

namespace sonder::inference::worker {
namespace {
using namespace std::chrono_literals;

// Engine selection is intentionally independent of IPC and lifecycle. Native
// backends can be registered here without exposing a logits transport.
Result<std::shared_ptr<Backend>> worker_engine(std::string_view name, std::chrono::milliseconds delay) {
    if (name != "mock")
        return Status(ErrorCode::unsupported, "worker engine is not available");
    MockBackendOptions options;
    options.token_delay = delay;
    return make_mock_backend(options);
}

// Single control-thread owner; the generation thread only uses stream state
// under mutex and emits data through the serialized writer. There is exactly
// one outstanding token, so output memory never grows with consumer lag.
class Worker {
  public:
    Worker(std::shared_ptr<Channel> channel, std::shared_ptr<Backend> backend)
        : channel_(std::move(channel)), backend_(std::move(backend)) {}
    ~Worker() {
        {
            std::lock_guard lock(stream_mutex_);
            cancel_.cancel();
        }
        credit_.notify_all();
        channel_->close();
        if (generation_.joinable())
            generation_.join();
    }
    int run() {
        for (;;) {
            auto frame = channel_->read_frame(1s);
            if (!frame) {
                if (frame.status().code() == ErrorCode::timeout)
                    continue;
                return 1;
            }
            auto parsed = parse_message(frame.value());
            if (!parsed) {
                // Framing is intact, so reject this payload without losing the
                // engine. ID 0 denotes an envelope that cannot be correlated.
                if (!reject(0, parsed.status()))
                    return 1;
                continue;
            }
            const auto &value = parsed.value();
            const auto id = value.find("id")->as_uint();
            const auto type = value.find("type")->as_string();
            if (type == "shutdown") {
                if (id != 0 || !fields(value, {"v", "type", "id"}).ok()) {
                    if (!reject(id, {ErrorCode::invalid_argument, "invalid shutdown request"}))
                        return 1;
                    continue;
                }
                {
                    std::lock_guard lock(stream_mutex_);
                    cancel_.cancel();
                }
                credit_.notify_all();
                return 0;
            }
            if (type == "cancel" || type == "stop") {
                std::lock_guard lock(stream_mutex_);
                // Stale controls, including extra stale fields, must never
                // enqueue an error that the next request could consume.
                if (id != 0 && id <= last_id_ && (id != active_id_ || !active_.load()))
                    continue;
                if (!fields(value, {"v", "type", "id"}).ok()) {
                    if (!reject(id, {ErrorCode::invalid_argument, "invalid cancellation fields"}))
                        return 1;
                    continue;
                }
                if (id == 0 || id != active_id_) {
                    if (!reject(id, {ErrorCode::invalid_argument, "unknown cancellation request"}))
                        return 1;
                    continue;
                }
                callback_stop_.store(type == "stop");
                cancel_.cancel();
                credit_.notify_all();
                continue;
            }
            if (type == "ack") {
                if (!fields(value, {"v", "type", "id", "index"}).ok()) {
                    if (!reject(id, {ErrorCode::invalid_argument, "invalid acknowledgement fields"}))
                        return 1;
                    continue;
                }
                auto index = unsigned_field(value, "index");
                std::lock_guard lock(stream_mutex_);
                if (!index || id != active_id_ || !awaiting_ || index.value() != awaiting_index_) {
                    if (!reject(id, {ErrorCode::invalid_argument, "unexpected acknowledgement"}))
                        return 1;
                    continue;
                }
                awaiting_ = false;
                credit_.notify_all();
                continue;
            }
            if (id == 0 || id <= last_id_ || active_.load()) {
                if (!reject(id, {ErrorCode::invalid_argument, "request id is stale or worker is busy"}))
                    return 1;
                continue;
            }
            if (generation_.joinable())
                generation_.join();
            last_id_ = id;
            if (type == "probe" || type == "list") {
                if (!fields(value, {"v", "type", "id"}).ok()) {
                    if (!reject(id, {ErrorCode::invalid_argument, "invalid discovery fields"}))
                        return 1;
                    continue;
                }
                if (type == "probe") {
                    auto result = backend_->probe();
                    if (!result) {
                        if (!write(error_message(id, result.status())))
                            return 1;
                    } else if (!result_message(id, result.value()))
                        return 1;
                } else {
                    auto result = backend_->list_models();
                    if (!result) {
                        if (!write(error_message(id, result.status())))
                            return 1;
                    } else {
                        json::Array models;
                        for (const auto &m : result.value())
                            models.emplace_back(descriptor_to_json(m));
                        if (!result_message(id, std::move(models)))
                            return 1;
                    }
                }
                continue;
            }
            if (type == "load") {
                if (!fields(value, {"v", "type", "id", "model", "device"}).ok()) {
                    if (!reject(id, {ErrorCode::invalid_argument, "invalid load fields"}))
                        return 1;
                    continue;
                }
                auto model = load(value);
                if (!model) {
                    if (!reject(id, model.status()))
                        return 1;
                } else if (!result_message(id, descriptor_to_json(model.value()->descriptor())))
                    return 1;
                continue;
            }
            if (type != "generate" && type != "chat") {
                if (!reject(id, {ErrorCode::invalid_argument, "unknown worker operation"}))
                    return 1;
                continue;
            }
            if (!fields(value, {"v", "type", "id", "model", "device", "request"}).ok()) {
                if (!reject(id, {ErrorCode::invalid_argument, "invalid execution request fields"}))
                    return 1;
                continue;
            }
            auto generate = type == "generate" ? generate_from_json(*value.find("request"))
                                               : Result<GenerateRequest>(GenerateRequest{});
            auto chat = type == "chat" ? chat_from_json(*value.find("request")) : Result<ChatRequest>(ChatRequest{});
            if (!generate || !chat) {
                if (!reject(id, !generate ? generate.status() : chat.status()))
                    return 1;
                continue;
            }
            // Decode first: rejected input must not load/evict a model either.
            auto model = load(value);
            if (!model) {
                if (!reject(id, model.status()))
                    return 1;
                continue;
            }
            {
                std::lock_guard lock(stream_mutex_);
                active_id_ = id;
                awaiting_ = false;
                cancel_ = CancellationSource{};
                callback_stop_.store(false);
                active_.store(true);
            }
            generation_ = std::thread([this, id, kind = type, model = model.value(),
                                       generate = std::move(generate).value(), chat = std::move(chat).value()] {
                try {
                    const TokenCallback callback = [this, id](const TokenChunk &chunk) {
                        if (cancel_.cancelled())
                            return false;
                        if (chunk.top_logprobs.size() > static_cast<std::size_t>(SamplingConfig::kMaxTopLogprobs) ||
                            !chunk.tool_call_deltas.empty()) {
                            cancel_.cancel();
                            return false;
                        }
                        {
                            std::lock_guard lock(stream_mutex_);
                            awaiting_ = true;
                            awaiting_index_ = chunk.index;
                        }
                        auto token = message("token", id);
                        token.set("chunk", chunk_to_json(chunk));
                        if (!write(std::move(token))) {
                            cancel_.cancel();
                            channel_->close();
                            return false;
                        }
                        std::unique_lock lock(stream_mutex_);
                        credit_.wait(lock, [this] { return !awaiting_ || cancel_.cancelled(); });
                        return !cancel_.cancelled();
                    };
                    auto result = kind == "generate" ? model->generate(generate, cancel_.token(), callback)
                                                     : model->chat(chat, cancel_.token(), callback);
                    // Publish inactive before terminal frame. A next request
                    // can then join this thread after its terminal read.
                    active_.store(false);
                    if (cancel_.cancelled() && !(callback_stop_.load() && result.ok()))
                        (void)write(error_message(id, {ErrorCode::cancelled, "worker request cancelled"}));
                    else if (!result)
                        (void)write(error_message(id, result.status()));
                    else {
                        auto done = message("done", id);
                        done.set("stats", stats_to_json(result.value()));
                        (void)write(std::move(done));
                    }
                } catch (...) {
                    active_.store(false);
                    channel_->close();
                }
            });
        }
    }

  private:
    bool reject(std::uint64_t id, const Status &status) {
        // Protocol errors from a request decoder describe caller input, not a
        // broken connection. The parent must not retire the worker for them.
        const Status error = status.code() == ErrorCode::protocol_error
                                 ? Status(ErrorCode::invalid_argument, status.message())
                                 : status;
        return write(error_message(id, error));
    }
    bool write(json::Value value) {
        std::lock_guard lock(writer_);
        const bool ok = channel_->write_frame(value.dump(), 5s).ok();
        if (!ok)
            channel_->close();
        return ok;
    }
    bool result_message(std::uint64_t id, json::Value value) {
        auto response = message("result", id);
        response.set("value", std::move(value));
        return write(std::move(response));
    }
    Result<std::shared_ptr<BackendModel>> load(const json::Value &value) {
        auto name = string_field(value, "model"), device = string_field(value, "device");
        if (!name || !device)
            return Status(ErrorCode::protocol_error, "invalid model selection");
        if (loaded_ && loaded_name_ == name.value() && loaded_device_ == device.value())
            return loaded_;
        ModelLoadOptions options;
        options.model = name.value();
        options.device_id = device.value();
        auto model = backend_->load_model(options);
        if (!model)
            return model.status();
        loaded_ = model.value();
        loaded_name_ = name.value();
        loaded_device_ = device.value();
        return loaded_;
    }
    const std::shared_ptr<Channel> channel_;
    const std::shared_ptr<Backend> backend_;
    std::shared_ptr<BackendModel> loaded_; // control-thread only
    std::string loaded_name_, loaded_device_;
    std::uint64_t last_id_ = 0;
    std::mutex writer_, stream_mutex_;
    std::condition_variable credit_;
    CancellationSource cancel_;
    std::thread generation_;
    std::atomic<bool> active_{false};
    std::atomic<bool> callback_stop_{false};
    std::uint64_t active_id_ = 0, awaiting_index_ = 0; // stream_mutex_
    bool awaiting_ = false;                            // stream_mutex_
};
} // namespace

int worker_main(const std::vector<std::string> &arguments) {
    std::string endpoint, engine = "mock";
    std::chrono::milliseconds delay{0};
    bool seen_endpoint = false, seen_engine = false, seen_delay = false;
    for (std::size_t i = 0; i < arguments.size(); ++i) {
        if (i + 1 >= arguments.size())
            return 2;
        const auto key = arguments[i++];
        const auto &value = arguments[i];
        if (key == "--endpoint" && !seen_endpoint) {
            endpoint = value;
            seen_endpoint = true;
        } else if (key == "--engine" && !seen_engine) {
            engine = value;
            seen_engine = true;
        } else if (key == "--token-delay-ms" && !seen_delay) {
            std::int64_t n = 0;
            auto parsed = std::from_chars(value.data(), value.data() + value.size(), n);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || n < 0 || n > 60000)
                return 2;
            delay = std::chrono::milliseconds(n);
            seen_delay = true;
        } else
            return 2;
    }
    if (endpoint.empty())
        return 2;
    try {
        auto backend = worker_engine(engine, delay);
        if (!backend)
            return 2;
        auto channel = connect(endpoint, 5s);
        if (!channel)
            return 1;
        auto hello = message("hello", 0);
        hello.set("engine", engine);
        if (!channel.value()->write_frame(json::Value(std::move(hello)).dump(), 5s).ok())
            return 1;
        auto frame = channel.value()->read_frame(5s);
        if (!frame)
            return 1;
        auto ready = parse_message(frame.value());
        if (!ready || !fields(ready.value(), {"v", "type", "id", "engine"}).ok() ||
            ready->find("type")->as_string() != "ready" || ready->find("id")->as_uint() != 0 ||
            !ready->find("engine")->is_string() || ready->find("engine")->as_string() != engine)
            return 1;
        return Worker(channel.value(), backend.value()).run();
    } catch (...) {
        return 1;
    }
}
} // namespace sonder::inference::worker
