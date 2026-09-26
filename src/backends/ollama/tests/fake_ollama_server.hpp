// In-process fake Ollama HTTP server for unit tests (no live Ollama needed).
// Replays recorded NDJSON fixtures over chunked transfer encoding, optionally
// splitting lines at awkward byte boundaries, delaying, or stalling so
// cancellation can be exercised.
#pragma once

#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#include <httplib.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <atomic>
#include <chrono>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace sonder_test {

inline std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open fixture: " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

struct StreamScript {
  std::string body;              // full NDJSON payload
  int status = 200;
  std::size_t chunk_bytes = 0;   // 0 = write whole body at once
  std::chrono::milliseconds delay_per_chunk{0};
  // After writing `stall_after_bytes`, keep the connection open (sending blank
  // keepalive lines) until the client goes away or `stall_max` elapses.
  std::size_t stall_after_bytes = 0;
  std::chrono::milliseconds stall_max{5000};
  std::string content_type = "application/x-ndjson";
};

class FakeOllamaServer {
 public:
  FakeOllamaServer() {
    svr_.Get("/api/version", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"version":"0.12.0-fake"})", "application/json");
    });
    svr_.Get("/api/tags", [this](const httplib::Request&, httplib::Response& res) {
      std::lock_guard<std::mutex> l(mu_);
      res.set_content(tags_, "application/json");
    });
    svr_.Get("/api/ps", [](const httplib::Request&, httplib::Response& res) {
      res.set_content(R"({"models":[{"name":"llama3.2:3b","model":"llama3.2:3b","size":3400000000,"size_vram":3400000000,"expires_at":"2026-09-26T09:00:00Z"}]})", "application/json");
    });
    svr_.Post("/api/show", [this](const httplib::Request& req, httplib::Response& res) {
      std::lock_guard<std::mutex> l(mu_);
      last_show_body_ = req.body;
      if (req.body.find(show_model_) == std::string::npos) {
        res.status = 404;
        res.set_content(R"({"error":"model not found"})", "application/json");
        return;
      }
      res.set_content(show_, "application/json");
    });
    auto stream = [this](const httplib::Request& req, httplib::Response& res, bool chat) {
      StreamScript s;
      {
        std::lock_guard<std::mutex> l(mu_);
        (chat ? last_chat_body_ : last_generate_body_) = req.body;
        s = chat ? chat_ : generate_;
        ++request_count_;
      }
      res.status = s.status;
      if (s.status != 200) {
        res.set_content(s.body, s.content_type);
        return;
      }
      res.set_chunked_content_provider(
          s.content_type, [this, s](std::size_t, httplib::DataSink& sink) {
            const std::size_t step = s.chunk_bytes == 0 ? s.body.size() : s.chunk_bytes;
            std::size_t off = 0;
            const std::size_t limit = s.stall_after_bytes ? std::min(s.stall_after_bytes, s.body.size())
                                                          : s.body.size();
            while (off < limit) {
              const std::size_t n = std::min(step, limit - off);
              if (!sink.write(s.body.data() + off, n)) return false;
              off += n;
              if (s.delay_per_chunk.count() > 0) std::this_thread::sleep_for(s.delay_per_chunk);
            }
            if (s.stall_after_bytes) {
              const auto deadline = std::chrono::steady_clock::now() + s.stall_max;
              while (!stopping_ && std::chrono::steady_clock::now() < deadline) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                if (!sink.write("\n", 1)) return false;
              }
              return false;  // never completes normally
            }
            sink.done();
            return true;
          });
    };
    svr_.Post("/api/generate", [stream](const httplib::Request& q, httplib::Response& r) { stream(q, r, false); });
    svr_.Post("/api/chat", [stream](const httplib::Request& q, httplib::Response& r) { stream(q, r, true); });

    port_ = svr_.bind_to_any_port("127.0.0.1");
    if (port_ <= 0) throw std::runtime_error("fake server: bind failed");
    thread_ = std::thread([this] { svr_.listen_after_bind(); });
    svr_.wait_until_ready();
  }

  ~FakeOllamaServer() {
    stopping_ = true;
    svr_.stop();
    if (thread_.joinable()) thread_.join();
  }

  std::string base_url() const { return "http://127.0.0.1:" + std::to_string(port_); }
  int port() const { return port_; }

  void set_generate(StreamScript s) { std::lock_guard<std::mutex> l(mu_); generate_ = std::move(s); }
  void set_chat(StreamScript s) { std::lock_guard<std::mutex> l(mu_); chat_ = std::move(s); }
  void set_tags(std::string j) { std::lock_guard<std::mutex> l(mu_); tags_ = std::move(j); }
  void set_show(std::string model, std::string j) {
    std::lock_guard<std::mutex> l(mu_);
    show_model_ = std::move(model);
    show_ = std::move(j);
  }
  std::string last_generate_body() { std::lock_guard<std::mutex> l(mu_); return last_generate_body_; }
  std::string last_chat_body() { std::lock_guard<std::mutex> l(mu_); return last_chat_body_; }
  std::string last_show_body() { std::lock_guard<std::mutex> l(mu_); return last_show_body_; }
  int request_count() { std::lock_guard<std::mutex> l(mu_); return request_count_; }

 private:
  httplib::Server svr_;
  std::thread thread_;
  int port_ = 0;
  std::atomic<bool> stopping_{false};
  std::mutex mu_;
  StreamScript generate_;
  StreamScript chat_;
  std::string tags_ = R"({"models":[]})";
  std::string show_model_ = "\x01";
  std::string show_;
  std::string last_generate_body_, last_chat_body_, last_show_body_;
  int request_count_ = 0;
};

}  // namespace sonder_test
