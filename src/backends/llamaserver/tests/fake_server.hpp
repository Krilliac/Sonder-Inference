#pragma once

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <httplib.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace sonder_test {
class FakeLlamaServer {
  public:
    explicit FakeLlamaServer(int port = 0, unsigned health_failures = 0) {
        svr_.Get("/health", [this, health_failures](const httplib::Request &, httplib::Response &r) {
            r.status = health_requests_.fetch_add(1) < health_failures ? 503 : 200;
            r.set_content(r.status == 200 ? R"({"status":"ok"})" : R"({"error":"Loading model"})",
                          "application/json");
        });
        // Absent (404) until set_props(): generic OpenAI upstreams have none.
        svr_.Get("/props", [this](const httplib::Request &, httplib::Response &r) {
            std::lock_guard<std::mutex> lock(mu_);
            ++props_requests_;
            if (props_.empty()) {
                r.status = 404;
                return;
            }
            r.set_content(props_, "application/json");
        });
        svr_.Get("/v1/models", [](const httplib::Request &, httplib::Response &r) {
            r.set_content(R"({"object":"list","data":[{"id":"fake-model","object":"model"}]})",
                          "application/json");
        });
        auto stream = [this](const httplib::Request &req, httplib::Response &r) {
            const bool nonstream = req.body.find(R"("stream":false)") != std::string::npos ||
                                   req.body.find(R"("stream": false)") != std::string::npos;
            {
                std::lock_guard<std::mutex> lock(mu_);
                last_body_ = req.body;
                bodies_.push_back(req.body);
                if (nonstream)
                    nonstream_bodies_.push_back(req.body);
            }
            changed_.notify_all();
            if (nonstream) {
                std::unique_lock lock(mu_);
                changed_.wait(lock, [this] { return !hold_nonstream_; });
            }
            std::lock_guard<std::mutex> lock(mu_);
            r.status = status_;
            if (status_ != 200) {
                r.set_content(body_, nonstream ? "application/json" : "text/event-stream");
                return;
            }
            const auto payload = body_;
            const auto initial_delay = initial_delay_;
            if (nonstream) {
                r.set_content(nonstream_body_.empty() ?
                                  R"({"choices":[{"finish_reason":"stop"}],"timings":{"prompt_n":3,"cache_n":9}})" :
                                  nonstream_body_,
                              "application/json");
                return;
            }
            r.set_chunked_content_provider("text/event-stream",
                                           [this, payload, initial_delay](std::size_t, httplib::DataSink &sink) {
                                               if (initial_delay.count() > 0)
                                                   std::this_thread::sleep_for(initial_delay);
                                               const std::size_t step = payload.size() > 4096 ? 4093 : 7;
                                               for (std::size_t i = 0; i < payload.size(); i += step) {
                                                   const auto n = std::min(step, payload.size() - i);
                                                   if (!sink.write(payload.data() + i, n)) {
                                                       disconnects_.fetch_add(1);
                                                       return false;
                                                   }
                                               }
                                               if (initial_delay.count() > 0) {
                                                   // Probe after the delayed
                                                   // first write; a tiny fake
                                                   // response can otherwise be
                                                   // buffered after the peer
                                                   // has already closed.
                                                   for (int i = 0; i < 100; ++i) {
                                                       std::this_thread::sleep_for(std::chrono::milliseconds(10));
                                                       if (!sink.write("\n", 1)) {
                                                           disconnects_.fetch_add(1);
                                                           return false;
                                                       }
                                                   }
                                                   return false;
                                               }
                                               sink.done();
                                               return true;
                                           });
        };
        svr_.Post("/completion", stream);
        svr_.Post("/v1/completions", stream);
        svr_.Post("/v1/chat/completions", stream);
        svr_.Post(R"(/slots/(\d+))", [this](const httplib::Request &req, httplib::Response &r) {
            std::lock_guard<std::mutex> lock(mu_);
            last_slot_ = req.target;
            last_slot_body_ = req.body;
            r.set_content("{}", "application/json");
        });
        port_ = port == 0 ? svr_.bind_to_any_port("127.0.0.1")
                          : (svr_.bind_to_port("127.0.0.1", port) ? port : -1);
        if (port_ <= 0)
            throw std::runtime_error("fake llama server bind failed");
        thread_ = std::thread([this] { svr_.listen_after_bind(); });
        svr_.wait_until_ready();
    }
    ~FakeLlamaServer() {
        stop();
    }
    void stop() {
        // Release a held nonstream handler before asking httplib to stop and
        // joining its listener thread; otherwise a cancellation test can
        // leave the handler waiting forever while the destructor joins.
        hold_nonstream(false);
        svr_.stop();
        if (thread_.joinable())
            thread_.join();
    }
    std::string url() const { return "http://127.0.0.1:" + std::to_string(port_); }
    void set_body(std::string body) {
        std::lock_guard<std::mutex> lock(mu_);
        body_ = std::move(body);
    }
    void set_status(int status) {
        std::lock_guard<std::mutex> lock(mu_);
        status_ = status;
    }
    void set_initial_delay(std::chrono::milliseconds delay) {
        std::lock_guard<std::mutex> lock(mu_);
        initial_delay_ = delay;
    }
    void set_nonstream_body(std::string body) {
        std::lock_guard<std::mutex> lock(mu_);
        nonstream_body_ = std::move(body);
    }
    std::string last_body() const {
        std::lock_guard<std::mutex> lock(mu_);
        return last_body_;
    }
    std::vector<std::string> bodies() const {
        std::lock_guard<std::mutex> lock(mu_);
        return bodies_;
    }
    void set_props(std::string props) {
        std::lock_guard<std::mutex> lock(mu_);
        props_ = std::move(props);
    }
    unsigned props_requests() const {
        std::lock_guard<std::mutex> lock(mu_);
        return props_requests_;
    }
    std::string last_slot() const {
        std::lock_guard<std::mutex> lock(mu_);
        return last_slot_;
    }
    std::string last_slot_body() const {
        std::lock_guard<std::mutex> lock(mu_);
        return last_slot_body_;
    }
    unsigned health_requests() const { return health_requests_.load(); }
    unsigned disconnects() const { return disconnects_.load(); }
    std::vector<std::string> nonstream_bodies() const {
        std::lock_guard<std::mutex> lock(mu_);
        return nonstream_bodies_;
    }
    void hold_nonstream(bool hold = true) {
        std::lock_guard<std::mutex> lock(mu_);
        hold_nonstream_ = hold;
        if (!hold)
            changed_.notify_all();
    }
    bool wait_for_nonstream(std::size_t count) const {
        std::unique_lock lock(mu_);
        return changed_.wait_for(lock, std::chrono::seconds(3),
                                 [&] { return nonstream_bodies_.size() >= count; });
    }

  private:
    httplib::Server svr_;
    std::thread thread_;
    int port_ = 0;
    mutable std::mutex mu_;
    std::string body_;
    std::string nonstream_body_;
    std::string last_body_;
    std::vector<std::string> bodies_;
    std::vector<std::string> nonstream_bodies_;
    mutable std::condition_variable changed_;
    bool hold_nonstream_ = false;
    std::string props_;
    unsigned props_requests_ = 0;
    std::string last_slot_;
    std::string last_slot_body_;
    int status_ = 200;
    std::chrono::milliseconds initial_delay_{0};
    std::atomic<unsigned> health_requests_{0};
    std::atomic<unsigned> disconnects_{0};
};
} // namespace sonder_test
