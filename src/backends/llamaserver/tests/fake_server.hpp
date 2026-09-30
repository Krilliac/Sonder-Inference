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
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

namespace sonder_test {
class FakeLlamaServer {
  public:
    explicit FakeLlamaServer(int port = 0, unsigned health_failures = 0) {
        svr_.Get("/health", [this, health_failures](const httplib::Request &, httplib::Response &r) {
            r.status = health_requests_.fetch_add(1) < health_failures ? 503 : 200;
            r.set_content(r.status == 200 ? R"({"status":"ok"})" : R"({"error":"Loading model"})",
                          "application/json");
        });
        svr_.Get("/v1/models", [](const httplib::Request &, httplib::Response &r) {
            r.set_content(R"({"object":"list","data":[{"id":"fake-model","object":"model"}]})",
                          "application/json");
        });
        auto stream = [this](const httplib::Request &req, httplib::Response &r) {
            std::lock_guard<std::mutex> lock(mu_);
            last_body_ = req.body;
            r.status = status_;
            if (status_ != 200) {
                r.set_content(body_, "text/event-stream");
                return;
            }
            const auto payload = body_;
            r.set_chunked_content_provider("text/event-stream",
                                           [payload](std::size_t, httplib::DataSink &sink) {
                                               const std::size_t step = payload.size() > 4096 ? 4093 : 7;
                                               for (std::size_t i = 0; i < payload.size(); i += step) {
                                                   const auto n = std::min(step, payload.size() - i);
                                                   if (!sink.write(payload.data() + i, n))
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
    std::string last_body() const {
        std::lock_guard<std::mutex> lock(mu_);
        return last_body_;
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

  private:
    httplib::Server svr_;
    std::thread thread_;
    int port_ = 0;
    mutable std::mutex mu_;
    std::string body_;
    std::string last_body_;
    std::string last_slot_;
    std::string last_slot_body_;
    int status_ = 200;
    std::atomic<unsigned> health_requests_{0};
};
} // namespace sonder_test
