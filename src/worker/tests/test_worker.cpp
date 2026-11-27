#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "server_test_support.hpp"
#include "sonder/inference/backends.hpp"
#include "worker/ipc.hpp"
#include "worker/process.hpp"
#include "worker/protocol.hpp"
#include "worker/remote_backend.hpp"

#if defined(_WIN32)
#include <windows.h>
#else
#include <csignal>
#include <fcntl.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#endif

namespace {
namespace si = sonder::inference;
namespace wk = si::worker;
using namespace std::chrono_literals;
using server_test::eventually;

wk::RemoteBackendOptions worker_options(std::chrono::milliseconds delay = 0ms) {
    wk::RemoteBackendOptions options;
    options.executable = SONDER_WORKER_EXECUTABLE;
    options.token_delay = delay;
    options.restart_initial_backoff = 20ms;
    options.restart_max_backoff = 100ms;
    return options;
}

si::server::ServerOptions serve_options(const std::shared_ptr<wk::RemoteBackend>& backend) {
    auto options = server_test::Fixture::defaults();
    options.backend.backend = "remote-mock";
    options.backend_instance = backend;
    return options;
}

bool event_for(server_test::Fixture& fixture, const std::string& type, const std::string& run) {
    for (const auto& event : fixture.of_type(type)) {
        const auto* value = event.find("run_id");
        if (value && value->as_string() == run) return true;
    }
    return false;
}

std::vector<si::json::Value> stream_objects(const std::string& body) {
    std::vector<si::json::Value> output;
    std::size_t offset = 0;
    while ((offset = body.find("data: ", offset)) != std::string::npos) {
        const auto end = body.find('\n', offset);
        const auto data = body.substr(offset + 6, end - offset - 6);
        if (data != "[DONE]") {
            auto value = si::json::parse(data);
            REQUIRE_MESSAGE(value.ok(), data);
            output.push_back(std::move(value.value()));
        }
        offset = end == std::string::npos ? body.size() : end + 1;
    }
    return output;
}

void kill_worker(std::uint32_t pid) {
    REQUIRE(pid != 0);
#if defined(_WIN32)
    HANDLE process = OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE, FALSE, pid);
    REQUIRE(process != nullptr);
    const bool terminated = TerminateProcess(process, 91) != FALSE;
    if (terminated) (void)WaitForSingleObject(process, 5000);
    CloseHandle(process);
    REQUIRE(terminated);
#else
    REQUIRE(::kill(static_cast<pid_t>(pid), SIGKILL) == 0);
#endif
}

bool process_alive(std::uint32_t pid) {
#if defined(_WIN32)
    HANDLE process = OpenProcess(SYNCHRONIZE, FALSE, pid);
    if (!process) return false;
    const bool alive = WaitForSingleObject(process, 0) == WAIT_TIMEOUT;
    CloseHandle(process);
    return alive;
#else
    if (::kill(static_cast<pid_t>(pid), 0) != 0) return false;
    // A reparented zombie has released engine/GPU resources and cannot run.
    std::ifstream status("/proc/" + std::to_string(pid) + "/stat");
    const std::string text((std::istreambuf_iterator<char>(status)), std::istreambuf_iterator<char>());
    const auto end = text.rfind(')');
    return end == std::string::npos || text.size() <= end + 2 || text[end + 2] != 'Z';
#endif
}

// Native client intentionally bypasses Channel's writer validation so the
// receiver sees the exact invalid length prefix specified by ARCH-01.
class RawPeer {
public:
    explicit RawPeer(const std::string& endpoint) {
#if defined(_WIN32)
        const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, endpoint.data(),
                                              static_cast<int>(endpoint.size()), nullptr, 0);
        REQUIRE(count > 0);
        std::wstring path(static_cast<std::size_t>(count), L'\0');
        REQUIRE(MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, endpoint.data(),
                                   static_cast<int>(endpoint.size()), path.data(), count) == count);
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                              OPEN_EXISTING, 0, nullptr);
        REQUIRE(handle_ != INVALID_HANDLE_VALUE);
#else
        handle_ = ::socket(AF_UNIX, SOCK_STREAM, 0);
        REQUIRE(handle_ >= 0);
        sockaddr_un address{};
        address.sun_family = AF_UNIX;
        REQUIRE(endpoint.size() < sizeof(address.sun_path));
        std::copy(endpoint.begin(), endpoint.end(), address.sun_path);
        REQUIRE(::connect(handle_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
#endif
    }
    ~RawPeer() { close(); }
    RawPeer(const RawPeer&) = delete;
    RawPeer& operator=(const RawPeer&) = delete;
    void write(const std::vector<unsigned char>& bytes) {
#if defined(_WIN32)
        DWORD written = 0;
        REQUIRE(WriteFile(handle_, bytes.data(), static_cast<DWORD>(bytes.size()), &written, nullptr));
        REQUIRE(written == bytes.size());
#else
        REQUIRE(::send(handle_, bytes.data(), bytes.size(), MSG_NOSIGNAL) == static_cast<ssize_t>(bytes.size()));
#endif
    }
    void close() {
#if defined(_WIN32)
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
#else
        if (handle_ >= 0) ::close(handle_);
        handle_ = -1;
#endif
    }
private:
#if defined(_WIN32)
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int handle_ = -1;
#endif
};

class EnvironmentOverride {
public:
    explicit EnvironmentOverride(std::string key) : key_(std::move(key)) {
        if (const char* previous = std::getenv(key_.c_str())) previous_ = previous;
        set("sonder-worker-parent-only-marker");
    }
    ~EnvironmentOverride() {
#if defined(_WIN32)
        (void)_putenv_s(key_.c_str(), previous_ ? previous_->c_str() : "");
#else
        if (previous_) (void)::setenv(key_.c_str(), previous_->c_str(), 1);
        else (void)::unsetenv(key_.c_str());
#endif
    }
    EnvironmentOverride(const EnvironmentOverride&) = delete;
    EnvironmentOverride& operator=(const EnvironmentOverride&) = delete;
private:
    void set(const char* value) {
#if defined(_WIN32)
        REQUIRE(_putenv_s(key_.c_str(), value) == 0);
#else
        REQUIRE(::setenv(key_.c_str(), value, 1) == 0);
#endif
    }
    std::string key_;
    std::optional<std::string> previous_;
};

// Deliberately opt this object into OS inheritance. The worker launcher must
// still exclude it, independently of its exact environment block.
class InheritableSentinel {
public:
    InheritableSentinel() {
#if defined(_WIN32)
        SECURITY_ATTRIBUTES attributes{};
        attributes.nLength = sizeof(attributes);
        attributes.bInheritHandle = TRUE;
        handle_ = CreateEventW(&attributes, TRUE, FALSE, nullptr);
        REQUIRE(handle_ != nullptr);
        DWORD flags = 0;
        REQUIRE(GetHandleInformation(handle_, &flags));
        REQUIRE((flags & HANDLE_FLAG_INHERIT) != 0);
#else
        int descriptors[2]{};
        REQUIRE(::pipe(descriptors) == 0);
        handle_ = fcntl(descriptors[0], F_DUPFD, 256);
        ::close(descriptors[0]);
        ::close(descriptors[1]);
        REQUIRE(handle_ >= 256);
        REQUIRE(fcntl(handle_, F_SETFD, 0) == 0);
        REQUIRE((fcntl(handle_, F_GETFD) & FD_CLOEXEC) == 0);
#endif
    }
    ~InheritableSentinel() {
#if defined(_WIN32)
        if (handle_) CloseHandle(handle_);
#else
        if (handle_ >= 0) ::close(handle_);
#endif
    }
    InheritableSentinel(const InheritableSentinel&) = delete;
    InheritableSentinel& operator=(const InheritableSentinel&) = delete;
    std::string argument() const {
#if defined(_WIN32)
        return std::to_string(reinterpret_cast<std::uintptr_t>(handle_));
#else
        return std::to_string(handle_);
#endif
    }
#if defined(_WIN32)
    bool signaled() const {
        const DWORD result = WaitForSingleObject(handle_, 0);
        REQUIRE((result == WAIT_OBJECT_0 || result == WAIT_TIMEOUT));
        return result == WAIT_OBJECT_0;
    }
#endif
private:
#if defined(_WIN32)
    HANDLE handle_ = nullptr;
#else
    int handle_ = -1;
#endif
};

} // namespace

TEST_CASE("worker: real serve CLI creates the mock worker through its backend factory") {
    const auto ready = std::filesystem::temp_directory_path() /
        ("sonder-worker-ready-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".json");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code ec;
            std::filesystem::remove(path, ec);
            std::filesystem::remove(path.string() + ".tmp", ec);
        }
    } cleanup{ready};
    auto child = wk::spawn_process(SONDER_WORKER_EXECUTABLE,
        {"serve", "--backend", "remote-mock", "--model", "mock:tiny", "--port", "0", "--ready-file", ready.string()});
    REQUIRE_MESSAGE(child.ok(), child.status().to_string());
    REQUIRE(eventually([&] { return std::filesystem::exists(ready); }, 10000ms));
    std::ifstream input(ready, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    const auto parsed = si::json::parse(text);
    REQUIRE_MESSAGE(parsed.ok(), text);
    const auto* url = parsed->find("url");
    REQUIRE(url != nullptr);
    REQUIRE(url->as_string().rfind("http://127.0.0.1:", 0) == 0);
    const auto port = static_cast<std::uint16_t>(std::stoi(url->as_string().substr(url->as_string().rfind(':') + 1)));
    REQUIRE(port != 0);
    si::server::detail::ensure_network();
    REQUIRE(eventually([&] { return server_test::get(port, "/v1/sonder/health").status == 200; }));
    const auto reply = server_test::post(port, "/v1/chat/completions",
        server_test::chat_body(R"(,"max_tokens":3,"stream":true)"));
    REQUIRE_MESSAGE(reply.status == 200, reply.body);
    CHECK(reply.complete);
    CHECK(reply.body.find("data: [DONE]") != std::string::npos);
    const auto objects = stream_objects(reply.body);
    REQUIRE(objects.size() >= 3);
    REQUIRE(objects.back().find("sonder") != nullptr);
    CHECK(objects.back().find("sonder")->find("synthetic")->as_bool());
    child.value()->stop(0ms);
    CHECK_FALSE(child.value()->running());
}

TEST_CASE("worker: abrupt parent exit kills its engine worker without destructor cleanup") {
    const auto pid_file = std::filesystem::temp_directory_path() /
        ("sonder-worker-parent-death-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{pid_file};
    auto parent = wk::spawn_process(SONDER_WORKER_PARENT_PROBE, {SONDER_WORKER_EXECUTABLE, pid_file.string()});
    REQUIRE_MESSAGE(parent.ok(), parent.status().to_string());
    REQUIRE(eventually([&] { return !parent.value()->running(); }, 10000ms));
    std::ifstream input(pid_file);
    std::uint32_t worker_pid = 0;
    REQUIRE(static_cast<bool>(input >> worker_pid));
    REQUIRE(worker_pid != 0);
    CHECK(eventually([&] { return !process_alive(worker_pid); }));
    parent.value()->stop(0ms);
}

TEST_CASE("worker: serve streams generation and chat through an isolated mock") {
    auto backend = std::make_shared<wk::RemoteBackend>(worker_options());
    server_test::Fixture fixture(serve_options(backend));
    REQUIRE(backend->worker_pid() != 0);
    CHECK_FALSE(backend->capabilities().has(si::Capability::token_logits));
    CHECK(backend->capabilities().has(si::Capability::remote_process));

    const auto generation = server_test::post(fixture.port, "/v1/completions",
        R"({"model":"default","prompt":"hello worker","max_tokens":6,"seed":19,"stream":true,"stream_options":{"include_usage":true}})");
    REQUIRE_MESSAGE(generation.status == 200, generation.body);
    CHECK(generation.complete);
    CHECK(generation.body.find("data: [DONE]") != std::string::npos);
    const auto generate_objects = stream_objects(generation.body);
    REQUIRE(generate_objects.size() >= 2);
    std::string generated;
    for (const auto& object : generate_objects) {
        const auto* choices = object.find("choices");
        REQUIRE(choices != nullptr);
        if (!choices->as_array().empty()) {
            if (const auto* text = choices->as_array()[0].find("text")) generated += text->as_string();
        }
    }
    CHECK_FALSE(generated.empty());
    const auto plain_generation = server_test::post(fixture.port, "/v1/completions",
        R"({"model":"default","prompt":"hello worker","max_tokens":6,"seed":19})");
    REQUIRE(plain_generation.status == 200);
    CHECK(plain_generation.json().find("choices")->as_array()[0].find("text")->as_string() == generated);
    REQUIRE(generate_objects.back().find("usage") != nullptr);
    CHECK(generate_objects.back().find("usage")->find("completion_tokens")->as_int() == 6);

    const auto chat = server_test::post(fixture.port, "/v1/chat/completions",
        server_test::chat_body(R"(,"max_tokens":6,"seed":19,"stream":true,"stream_options":{"include_usage":true})"));
    REQUIRE_MESSAGE(chat.status == 200, chat.body);
    CHECK(chat.complete);
    CHECK(chat.body.find("data: [DONE]") != std::string::npos);
    const auto chat_objects = stream_objects(chat.body);
    REQUIRE(chat_objects.size() >= 3);
    std::string chatted;
    for (const auto& object : chat_objects) {
        const auto& choice = object.find("choices")->as_array()[0];
        if (const auto* delta = choice.find("delta")) {
            if (const auto* text = delta->find("content")) chatted += text->as_string();
        }
    }
    CHECK_FALSE(chatted.empty());
    const auto plain_chat = server_test::post(fixture.port, "/v1/chat/completions",
        server_test::chat_body(R"(,"max_tokens":6,"seed":19)"));
    REQUIRE(plain_chat.status == 200);
    CHECK(plain_chat.json().find("choices")->as_array()[0].find("message")->find("content")->as_string() == chatted);
    REQUIRE(chat_objects.back().find("usage") != nullptr);
    CHECK(chat_objects.back().find("usage")->find("completion_tokens")->as_int() == 6);
    REQUIRE(chat_objects.back().find("sonder") != nullptr);
    CHECK(chat_objects.back().find("sonder")->find("synthetic")->as_bool());
    const auto token_events = fixture.of_type("inference.token.generated");
    REQUIRE_FALSE(token_events.empty());
    for (const auto& event : token_events) {
        const auto* producer = event.find("producer");
        REQUIRE(producer != nullptr);
        REQUIRE(producer->find("synthetic") != nullptr);
        CHECK(producer->find("synthetic")->as_bool());
    }
}

TEST_CASE("worker: oversized HTTP chat is rejected without spending restart budget") {
    const auto options = worker_options();
    auto backend = std::make_shared<wk::RemoteBackend>(options);
    server_test::Fixture fixture(serve_options(backend));
    const auto pid = backend->worker_pid();
    const auto restarts = backend->restart_count();
    REQUIRE(pid != 0);
    si::json::Array messages;
    for (std::size_t i = 0; i <= wk::kMaxChatMessages; ++i)
        messages.emplace_back(si::json::Object{{"role", "user"}, {"content", "x"}});
    const auto body = si::json::Value(si::json::Object{
        {"model", "default"}, {"messages", std::move(messages)}, {"max_tokens", 1}}).dump();
    // More invalid requests than the entire restart budget must remain harmless.
    for (std::size_t i = 0; i <= options.max_restarts_per_hour; ++i) {
        const auto reply = server_test::post(fixture.port, "/v1/chat/completions", body);
        REQUIRE_MESSAGE((reply.status >= 400 && reply.status < 500), reply.body);
        CHECK(backend->worker_pid() == pid);
        CHECK(backend->restart_count() == restarts);
        CHECK_FALSE(backend->circuit_open());
    }
    const auto valid = server_test::post(fixture.port, "/v1/chat/completions",
                                        server_test::chat_body(R"(,"max_tokens":2)"));
    REQUIRE_MESSAGE(valid.status == 200, valid.body);
    CHECK(backend->worker_pid() == pid);
    CHECK(backend->restart_count() == restarts);
}

TEST_CASE("worker: callback stop preserves successful stats and permits the next request") {
    auto backend = std::make_shared<wk::RemoteBackend>(worker_options());
    auto model = backend->load_model({"mock:tiny"});
    REQUIRE_MESSAGE(model.ok(), model.status().to_string());
    const auto initial_pid = backend->worker_pid();
    for (const std::uint64_t stop_after : {1u, 3u}) {
        CAPTURE(stop_after);
        si::GenerateRequest request;
        request.prompt = "hello from isolated worker";
        request.sampling = si::SamplingConfig::greedy(8);
        std::uint64_t callbacks = 0;
        const auto stopped = model.value()->generate(request, {}, [&](const si::TokenChunk& chunk) {
            CHECK(chunk.index == callbacks);
            ++callbacks;
            return callbacks < stop_after;
        });
        REQUIRE_MESSAGE(stopped.ok(), stopped.status().to_string());
        CHECK(callbacks == stop_after);
        CHECK(stopped->stop_reason == si::StopReason::callback);
        // Mock generation counts prompt words; this four-word prompt must
        // retain its actual count even when delivery stops on the first token.
        CHECK(stopped->prompt_tokens == 4);
        CHECK(stopped->token_counts_from_backend);
        CHECK(stopped->completion_tokens == stop_after);
        CHECK(stopped->chunks == stop_after);
        request.sampling.max_tokens = 2;
        std::uint64_t next_callbacks = 0;
        const auto next = model.value()->generate(request, {}, [&](const si::TokenChunk&) {
            ++next_callbacks;
            return true;
        });
        REQUIRE_MESSAGE(next.ok(), next.status().to_string());
        CHECK(next_callbacks == 2);
        CHECK(next->stop_reason == si::StopReason::max_tokens);
        CHECK(next->completion_tokens == 2);
        CHECK(backend->worker_pid() == initial_pid);
        CHECK(backend->restart_count() == 0);
    }
}

TEST_CASE("worker: token logprobs and top candidates match direct mock execution") {
    auto backend = std::make_shared<wk::RemoteBackend>(worker_options());
    auto remote_model = backend->load_model({"mock:tiny"});
    REQUIRE_MESSAGE(remote_model.ok(), remote_model.status().to_string());
    auto reference_backend = si::make_mock_backend();
    auto reference_model = reference_backend->load_model({"mock:tiny"});
    REQUIRE(reference_model.ok());
    si::GenerateRequest request;
    request.prompt = "preserve candidate observations";
    request.sampling = si::SamplingConfig::greedy(4);
    request.sampling.logprobs = true;
    request.sampling.top_logprobs = 3;
    struct Observation {
        std::string text;
        std::optional<float> logprob;
        std::vector<si::TopLogprob> candidates;
    };
    std::vector<Observation> reference;
    const auto expected = reference_model.value()->generate(request, {}, [&](const si::TokenChunk& chunk) {
        reference.push_back({std::string(chunk.text), chunk.logprob, chunk.top_logprobs});
        return true;
    });
    REQUIRE(expected.ok());
    std::size_t delivered = 0;
    const auto actual = remote_model.value()->generate(request, {}, [&](const si::TokenChunk& chunk) {
        REQUIRE(delivered < reference.size());
        const auto& baseline = reference[delivered++];
        CHECK(chunk.text == baseline.text);
        REQUIRE(chunk.logprob.has_value());
        CHECK(chunk.logprob == baseline.logprob);
        REQUIRE(chunk.top_logprobs.size() == baseline.candidates.size());
        CHECK(chunk.top_logprobs.size() == 3);
        for (std::size_t i = 0; i < baseline.candidates.size(); ++i) {
            CHECK(chunk.top_logprobs[i].token == baseline.candidates[i].token);
            CHECK(chunk.top_logprobs[i].logprob == baseline.candidates[i].logprob);
        }
        return true;
    });
    REQUIRE_MESSAGE(actual.ok(), actual.status().to_string());
    CHECK(delivered == reference.size());
    CHECK(actual->completion_tokens == expected->completion_tokens);
}

TEST_CASE("worker: HTTP disconnect cancels generation and chat without losing the worker") {
    auto backend = std::make_shared<wk::RemoteBackend>(worker_options(40ms));
    server_test::Fixture fixture(serve_options(backend));
    const auto initial_pid = backend->worker_pid();
    for (const bool chat : {false, true}) {
        CAPTURE(chat);
        const std::string run = chat ? "worker-cancel-chat" : "worker-cancel-generate";
        const std::string target = chat ? "/v1/chat/completions" : "/v1/completions";
        const std::string body = chat ? server_test::chat_body(R"(,"max_tokens":200,"stream":true)")
            : R"({"model":"default","prompt":"cancel this generation","max_tokens":200,"stream":true})";
        server_test::Conn client(fixture.port);
        client.send(server_test::build_request("POST", target, fixture.port, {{"X-Sonder-Run-Id", run}}, body));
        REQUIRE(eventually([&] { return event_for(fixture, "inference.token.generated", run); }));
        client.close();
        REQUIRE(eventually([&] { return event_for(fixture, "request.cancelled", run); }));
        CHECK_FALSE(event_for(fixture, "request.completed", run));
    }
    CHECK(server_test::post(fixture.port, "/v1/chat/completions",
                           server_test::chat_body(R"(,"max_tokens":2)")).status == 200);
    CHECK(backend->worker_pid() == initial_pid);
    CHECK(backend->restart_count() == 0);
}

TEST_CASE("worker: a crash fails the active HTTP request and restarts without replay") {
    auto backend = std::make_shared<wk::RemoteBackend>(worker_options(50ms));
    server_test::Fixture fixture(serve_options(backend));
    const auto crashed_pid = backend->worker_pid();
    const std::string run = "worker-crash-active";
    server_test::Conn client(fixture.port);
    client.send(server_test::build_request("POST", "/v1/chat/completions", fixture.port,
        {{"X-Sonder-Run-Id", run}}, server_test::chat_body(R"(,"max_tokens":200,"stream":true)")));
    REQUIRE(eventually([&] { return event_for(fixture, "inference.token.generated", run); }));
    kill_worker(crashed_pid);
    const auto failed_stream = client.read_all(5000ms);
    CHECK(client.closed());
    CHECK(failed_stream.find("error") != std::string::npos);
    REQUIRE(eventually([&] { return event_for(fixture, "request.failed", run); }));
    CHECK_FALSE(event_for(fixture, "request.completed", run));
    REQUIRE(eventually([&] { return backend->worker_pid() != 0 && backend->worker_pid() != crashed_pid; }));
    CHECK(backend->restart_count() == 1);
    CHECK(server_test::get(fixture.port, "/v1/sonder/health").status == 200);
    const auto recovered = server_test::post(fixture.port, "/v1/chat/completions",
                                            server_test::chat_body(R"(,"max_tokens":2)"));
    REQUIRE_MESSAGE(recovered.status == 200, recovered.body);
    CHECK_FALSE(recovered.json().find("choices")->as_array()[0].find("message")->find("content")->as_string().empty());
    CHECK_FALSE(event_for(fixture, "request.completed", run));
}

TEST_CASE("worker: restart budget is a rolling hour and uptime cannot reset it") {
    wk::RestartPolicy policy(2, 100ms, 250ms);
    const auto start = wk::RestartPolicy::Clock::time_point{};
    CHECK(policy.permit_restart(start));
    CHECK(policy.backoff() == 100ms);
    // Fifty minutes of healthy uptime does not erase the first restart.
    CHECK(policy.permit_restart(start + 50min));
    CHECK(policy.backoff() == 200ms);
    CHECK_FALSE(policy.permit_restart(start + 59min));
    CHECK(policy.recent_restarts(start + 59min) == 2);
    // Only the first timestamp expires at its hour boundary.
    CHECK(policy.permit_restart(start + 60min));
    CHECK(policy.recent_restarts(start + 60min) == 2);
    CHECK(policy.backoff() == 250ms);
    CHECK_FALSE(policy.permit_restart(start + 109min));
    CHECK(policy.permit_restart(start + 110min));
    CHECK(policy.backoff() == 250ms);
}

TEST_CASE("worker: repeated process crashes open the supervisor circuit") {
    auto options = worker_options();
    options.max_restarts_per_hour = 1;
    auto backend = std::make_shared<wk::RemoteBackend>(options);
    REQUIRE(backend->probe().ok());
    auto pid = backend->worker_pid();
    kill_worker(pid);
    REQUIRE(eventually([&] { return backend->worker_pid() != 0 && backend->worker_pid() != pid; }));
    REQUIRE(backend->probe().ok());
    CHECK(backend->restart_count() == 1);
    pid = backend->worker_pid();
    kill_worker(pid);
    REQUIRE(eventually([&] { return backend->circuit_open(); }));
    CHECK_FALSE(backend->probe().ok());
    CHECK(backend->restart_count() == 1);
}

TEST_CASE("worker: spawned process isolates environment cwd and inherited handles") {
    std::vector<std::unique_ptr<EnvironmentOverride>> poisoned;
    for (const char* key : {"PATH", "PATHEXT", "COMSPEC", "LD_LIBRARY_PATH", "LD_PRELOAD",
                            "DYLD_LIBRARY_PATH", "DYLD_INSERT_LIBRARIES", "VK_ICD_FILENAMES",
                            "VK_LAYER_PATH", "PYTHONPATH", "PYTHONHOME", "SONDER_WORKER_SECRET"}) {
        poisoned.push_back(std::make_unique<EnvironmentOverride>(key));
    }
    const auto path = std::filesystem::temp_directory_path() /
        ("sonder-worker-env-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt");
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ec; std::filesystem::remove(path, ec); }
    } cleanup{path};
    InheritableSentinel sentinel;
    auto child = wk::spawn_process(SONDER_WORKER_ENV_PROBE, {path.string(), sentinel.argument()});
    REQUIRE_MESSAGE(child.ok(), child.status().to_string());
    REQUIRE(eventually([&] { return !child.value()->running(); }));
    child.value()->stop(0ms);
    std::ifstream output(path, std::ios::binary);
    REQUIRE(output.good());
    std::string cwd;
    REQUIRE(static_cast<bool>(std::getline(output, cwd)));
    CHECK(std::filesystem::equivalent(std::filesystem::path(cwd),
                                     std::filesystem::path(SONDER_WORKER_ENV_PROBE).parent_path()));
    std::string inheritance;
    REQUIRE(static_cast<bool>(std::getline(output, inheritance)));
#if defined(_WIN32)
    // The helper may find an unrelated startup HANDLE with this numeric value.
    // It attempts SetEvent; only the original event changing state would prove
    // that the parent object itself crossed the inheritance boundary.
    CHECK((inheritance == "inherited=0" || inheritance == "inherited=1"));
    CHECK_FALSE(sentinel.signaled());
#else
    CHECK(inheritance == "inherited=0");
#endif
    std::map<std::string, std::string> observed;
    std::string entry;
    while (std::getline(output, entry)) {
        const auto equals = entry.find('=');
        REQUIRE(equals != std::string::npos);
        CHECK(observed.emplace(entry.substr(0, equals), entry.substr(equals + 1)).second);
    }
    const auto expected = wk::worker_environment();
    CHECK(observed.size() == expected.size());
    for (const auto& [key, value] : expected) {
        CAPTURE(key);
        const auto found = observed.find(key);
        REQUIRE(found != observed.end());
        CHECK(found->second == value);
    }
#if defined(_WIN32)
    CHECK(observed.size() == 2);
    CHECK(observed.count("SystemRoot") == 1);
    CHECK(observed["NoDefaultCurrentDirectoryInExePath"] == "1");
#else
    CHECK(observed.size() == 1);
    CHECK(observed["LANG"] == "C.UTF-8");
#endif
}

TEST_CASE("worker: framing accepts the size boundary and preserves embedded bytes") {
    auto listener = wk::make_listener();
    REQUIRE(listener.ok());
    auto client = wk::connect(listener.value()->endpoint(), 1000ms);
    REQUIRE_MESSAGE(client.ok(), client.status().to_string());
    auto server = listener.value()->accept(1000ms);
    REQUIRE(server.ok());
    std::string payload(wk::kMaxFrameBytes, 'x');
    payload[0] = '\0';
    payload.back() = '\xff';
    si::Status sent;
    std::thread writer([&] { sent = client.value()->write_frame(payload, 3000ms); });
    const auto received = server.value()->read_frame(3000ms);
    writer.join();
    REQUIRE_MESSAGE(sent.ok(), sent.to_string());
    REQUIRE(received.ok());
    CHECK(received.value() == payload);
    CHECK(server.value()->write_frame("reply", 1000ms).ok());
    const auto reply = client.value()->read_frame(1000ms);
    REQUIRE(reply.ok());
    CHECK(reply.value() == "reply");
}

TEST_CASE("worker: zero oversized and truncated length prefixes fail closed") {
    // ARCH-01 framing: unsigned 32-bit big-endian length, 1..1 MiB.
    for (const std::vector<unsigned char>& prefix : {
            std::vector<unsigned char>{0, 0, 0, 0},
            std::vector<unsigned char>{0, 0x10, 0, 1},
            std::vector<unsigned char>{0, 0x10}}) {
        CAPTURE(prefix.size());
        auto listener = wk::make_listener();
        REQUIRE(listener.ok());
        RawPeer peer(listener.value()->endpoint());
        auto receiver = listener.value()->accept(1000ms);
        REQUIRE(receiver.ok());
        peer.write(prefix);
        if (prefix.size() != 4) peer.close();
        const auto result = receiver.value()->read_frame(1000ms);
        CHECK_FALSE(result.ok());
        CHECK(result.status().code() != si::ErrorCode::timeout);
        CHECK_FALSE(receiver.value()->write_frame("must not continue", 100ms).ok());
    }
}

TEST_CASE("worker: incomplete frame timeout closes the channel while idle timeout is retryable") {
    auto listener = wk::make_listener();
    REQUIRE(listener.ok());
    RawPeer peer(listener.value()->endpoint());
    auto receiver = listener.value()->accept(1000ms);
    REQUIRE(receiver.ok());
    const auto idle = receiver.value()->read_frame(20ms);
    REQUIRE_FALSE(idle.ok());
    CHECK(idle.status().code() == si::ErrorCode::timeout);
    // A declared two-byte payload with only its first byte cannot be resumed
    // after the timeout; the framing boundary has become untrustworthy.
    peer.write({0, 0, 0, 2, 'x'});
    CHECK_FALSE(receiver.value()->read_frame(20ms).ok());
    CHECK_FALSE(receiver.value()->write_frame("must not continue", 100ms).ok());
}

TEST_CASE("worker: payload timeout starts after a delayed header arrives") {
    auto listener = wk::make_listener();
    REQUIRE(listener.ok());
    RawPeer peer(listener.value()->endpoint());
    auto receiver = listener.value()->accept(1000ms);
    REQUIRE(receiver.ok());
    si::Result<std::string> received = si::Status(si::ErrorCode::internal, "read not started");
    std::jthread reader([&] { received = receiver.value()->read_frame(1500ms); });
    std::this_thread::sleep_for(1000ms);
    peer.write({0, 0, 0, 2});
    std::this_thread::sleep_for(1000ms);
    peer.write({'o', 'k'});
    reader.join();
    REQUIRE_MESSAGE(received.ok(), received.status().to_string());
    CHECK(received.value() == "ok");
}

TEST_CASE("worker: malformed protocol envelopes and incompatible versions are rejected") {
    // The v1 envelope requires exactly a supported version, message type and
    // unsigned correlation ID; these are intentionally invalid instances.
    for (const char* payload : {"{", "null", "[]", "{}", "{\"v\":2,\"type\":\"hello\",\"id\":0}",
                               "{\"v\":1,\"type\":false,\"id\":0}",
                               "{\"v\":1,\"type\":\"hello\",\"id\":-1}"}) {
        CAPTURE(payload);
        CHECK_FALSE(wk::parse_message(payload).ok());
    }
}

TEST_CASE("worker: malformed handshake terminates the actual worker connection") {
    for (const char* payload : {"{", R"({"v":2,"type":"ready","id":0,"engine":"mock"})"}) {
        CAPTURE(payload);
        auto listener = wk::make_listener();
        REQUIRE(listener.ok());
        auto process = wk::spawn_process(SONDER_WORKER_EXECUTABLE,
            {"worker", "--endpoint", listener.value()->endpoint(), "--engine", "mock"});
        REQUIRE_MESSAGE(process.ok(), process.status().to_string());
        auto channel = listener.value()->accept(5000ms);
        REQUIRE(channel.ok());
        auto hello_frame = channel.value()->read_frame(5000ms);
        REQUIRE(hello_frame.ok());
        auto hello = wk::parse_message(hello_frame.value());
        REQUIRE(hello.ok());
        REQUIRE(hello->find("type")->as_string() == "hello");
        REQUIRE(channel.value()->write_frame(payload, 1000ms).ok());
        const auto closed = channel.value()->read_frame(5000ms);
        CHECK_FALSE(closed.ok());
        CHECK(closed.status().code() != si::ErrorCode::timeout);
        CHECK(eventually([&] { return !process.value()->running(); }));
        process.value()->stop(0ms);
    }
}

TEST_CASE("worker: invalid request payloads return typed errors and keep the worker usable") {
    auto listener = wk::make_listener();
    REQUIRE(listener.ok());
    auto process = wk::spawn_process(SONDER_WORKER_EXECUTABLE,
        {"worker", "--endpoint", listener.value()->endpoint(), "--engine", "mock"});
    REQUIRE_MESSAGE(process.ok(), process.status().to_string());
    const auto pid = process.value()->pid();
    auto channel = listener.value()->accept(5000ms);
    REQUIRE(channel.ok());
    const auto hello_frame = channel.value()->read_frame(5000ms);
    REQUIRE(hello_frame.ok());
    const auto hello = wk::parse_message(hello_frame.value());
    REQUIRE(hello.ok());
    REQUIRE(hello->find("type")->as_string() == "hello");
    auto ready = wk::message("ready", 0);
    ready.set("engine", "mock");
    REQUIRE(channel.value()->write_frame(si::json::Value(std::move(ready)).dump(), 1000ms).ok());

    std::uint64_t id = 0;
    const auto reject = [&](const char* kind, si::json::Value payload) {
        auto request = wk::message(kind, ++id);
        request.set("model", "mock:tiny");
        request.set("device", "cpu:0");
        request.set("request", std::move(payload));
        REQUIRE(channel.value()->write_frame(si::json::Value(std::move(request)).dump(), 1000ms).ok());
        const auto frame = channel.value()->read_frame(5000ms);
        REQUIRE_MESSAGE(frame.ok(), frame.status().to_string());
        const auto reply = wk::parse_message(frame.value());
        REQUIRE(reply.ok());
        CHECK(reply->find("id")->as_uint() == id);
        REQUIRE(reply->find("type")->as_string() == "error");
        CHECK(wk::status_from_message(reply.value()).code() == si::ErrorCode::invalid_argument);
        CHECK(process.value()->running());
        CHECK(process.value()->pid() == pid);
    };

    si::ChatRequest chat;
    si::ChatMessage message;
    message.role = "user";
    message.content = "x";
    chat.messages.push_back(message);
    const auto encoded_chat = wk::chat_to_json(chat);
    REQUIRE(encoded_chat.ok());
    auto bad_chat = encoded_chat->as_object();
    si::json::Array oversized;
    for (std::size_t i = 0; i <= wk::kMaxChatMessages; ++i)
        oversized.emplace_back(si::json::Object{{"role", "user"}, {"content", "x"}});
    bad_chat.set("messages", std::move(oversized));
    reject("chat", std::move(bad_chat)); // bypass the parent encoder deliberately

    si::GenerateRequest generate;
    generate.prompt = "still usable";
    generate.sampling = si::SamplingConfig::greedy(2);
    const auto encoded_generate = wk::generate_to_json(generate);
    REQUIRE(encoded_generate.ok());
    auto bad_generate = encoded_generate->as_object();
    bad_generate.set("target_itl_ms", -1);
    reject("generate", std::move(bad_generate));
    auto missing_fields = encoded_generate->as_object();
    missing_fields.set("sampling", nullptr);
    reject("generate", std::move(missing_fields));

    auto request = wk::message("generate", ++id);
    request.set("model", "mock:tiny");
    request.set("device", "cpu:0");
    request.set("request", encoded_generate.value());
    REQUIRE(channel.value()->write_frame(si::json::Value(std::move(request)).dump(), 1000ms).ok());
    for (std::uint64_t token = 0; token < 2; ++token) {
        const auto frame = channel.value()->read_frame(5000ms);
        REQUIRE(frame.ok());
        const auto reply = wk::parse_message(frame.value());
        REQUIRE(reply.ok());
        REQUIRE(reply->find("type")->as_string() == "token");
        CHECK(reply->find("id")->as_uint() == id);
        REQUIRE(reply->find("chunk") != nullptr);
        CHECK(reply->find("chunk")->find("index")->as_uint() == token);
        auto ack = wk::message("ack", id);
        ack.set("index", token);
        REQUIRE(channel.value()->write_frame(si::json::Value(std::move(ack)).dump(), 1000ms).ok());
    }
    const auto done_frame = channel.value()->read_frame(5000ms);
    REQUIRE(done_frame.ok());
    const auto done = wk::parse_message(done_frame.value());
    REQUIRE(done.ok());
    CHECK(done->find("type")->as_string() == "done");
    CHECK(done->find("id")->as_uint() == id);
    CHECK(process.value()->running());
    CHECK(process.value()->pid() == pid);
    REQUIRE(channel.value()->write_frame(si::json::Value(wk::message("shutdown", 0)).dump(), 1000ms).ok());
    CHECK(eventually([&] { return !process.value()->running(); }));
}

TEST_CASE("worker: parent encoders enforce worker metadata and flattened chat limits") {
    si::GenerateRequest generate;
    generate.prompt = "valid prompt";
    si::ChatRequest chat;
    si::ChatMessage message;
    message.role = "user";
    message.content = "legacy content";
    message.parts.emplace_back(si::TextContentPart{"first "});
    message.parts.emplace_back(si::TextContentPart{"second"});
    chat.messages.assign(wk::kMaxChatMessages, message);
    const auto boundary = wk::chat_to_json(chat);
    REQUIRE(boundary.ok());
    const auto decoded = wk::chat_from_json(boundary.value());
    REQUIRE(decoded.ok());
    CHECK(decoded->messages.size() == wk::kMaxChatMessages);
    CHECK(decoded->messages.front().content == "first second");
    chat.messages.push_back(message);
    const auto oversized = wk::chat_to_json(chat);
    REQUIRE_FALSE(oversized.ok());
    CHECK(oversized.status().code() == si::ErrorCode::invalid_argument);
    chat.messages.resize(1);
    chat.messages.front().parts.emplace_back(si::TextContentPart{std::string("\0", 1)});
    CHECK_FALSE(wk::chat_to_json(chat).ok());
    chat.messages.front() = message;

    const auto rejects_both = [&](auto change) {
        auto g = generate;
        auto c = chat;
        change(g);
        change(c);
        const auto encoded_g = wk::generate_to_json(g);
        const auto encoded_c = wk::chat_to_json(c);
        REQUIRE_FALSE(encoded_g.ok());
        REQUIRE_FALSE(encoded_c.ok());
        CHECK(encoded_g.status().code() == si::ErrorCode::invalid_argument);
        CHECK(encoded_c.status().code() == si::ErrorCode::invalid_argument);
    };
    rejects_both([](auto& r) { r.priority_class = static_cast<si::RequestPriority>(-1); });
    rejects_both([](auto& r) { r.priority_class = static_cast<si::RequestPriority>(99); });
    rejects_both([](auto& r) { r.target_ttft_ms = -1; });
    rejects_both([](auto& r) { r.target_itl_ms = -1; });
    rejects_both([](auto& r) { r.target_ttft_ms = std::numeric_limits<double>::infinity(); });
    rejects_both([](auto& r) { r.target_itl_ms = std::numeric_limits<double>::quiet_NaN(); });
    rejects_both([](auto& r) { r.sampling.explicit_fields = 0x10000u; });
    generate.target_ttft_ms = 0;
    generate.target_itl_ms = 0;
    const auto valid_generate = wk::generate_to_json(generate);
    REQUIRE(valid_generate.ok());
    CHECK(wk::generate_from_json(valid_generate.value()).ok());
}

TEST_CASE("worker: plain chat preserves the server reasoning split hint") {
    // execute_chat() always requests separate_reasoning, even when the client
    // requested no thinking. This output-routing flag must reach the worker.
    for (const bool separate : {false, true}) {
        si::ChatRequest request;
        si::ChatMessage message;
        message.role = "user";
        message.content = "plain text chat";
        request.messages.push_back(message);
        request.separate_reasoning = separate;
        const auto encoded = wk::chat_to_json(request);
        REQUIRE_MESSAGE(encoded.ok(), encoded.status().to_string());
        const auto parsed = si::json::parse(encoded->dump());
        REQUIRE(parsed.ok());
        const auto decoded = wk::chat_from_json(parsed.value());
        REQUIRE_MESSAGE(decoded.ok(), decoded.status().to_string());
        CHECK(decoded->separate_reasoning == separate);
        REQUIRE(decoded->messages.size() == 1);
        CHECK(decoded->messages.front().content == message.content);
        CHECK(decoded->thinking.empty());
        CHECK_FALSE(decoded->enable_stage_profiling);

        for (const auto& invalid : {si::json::Value(nullptr), si::json::Value(1), si::json::Value("true")}) {
            auto malformed = encoded->as_object();
            malformed.set("separate_reasoning", invalid);
            const auto rejected = wk::chat_from_json(si::json::Value(std::move(malformed)));
            REQUIRE_FALSE(rejected.ok());
            CHECK(rejected.status().code() == si::ErrorCode::protocol_error);
        }
    }
}

TEST_CASE("worker: server reasoning split hint does not enable unsupported chat features") {
    si::ChatRequest plain;
    si::ChatMessage message;
    message.role = "user";
    message.content = "plain text chat";
    plain.messages.push_back(message);
    plain.separate_reasoning = true;
    REQUIRE(wk::chat_to_json(plain).ok());
    const auto rejects = [&](auto change) {
        auto request = plain;
        change(request);
        const auto result = wk::chat_to_json(request);
        REQUIRE_FALSE(result.ok());
        CHECK(result.status().code() == si::ErrorCode::unsupported);
    };
    rejects([](auto& r) { r.thinking.enable_thinking = true; });
    rejects([](auto& r) { r.thinking.enable_thinking = false; });
    rejects([](auto& r) { r.thinking.reasoning_effort = "low"; });
    rejects([](auto& r) { r.reasoning_budget_tokens = 0; });
    rejects([](auto& r) { r.reasoning_budget_message = "budget"; });
    rejects([](auto& r) { r.grammar = "root ::= \"hello\""; });
    rejects([](auto& r) { r.enable_stage_profiling = true; });
    rejects([](auto& r) { r.tools.emplace_back(); });
    rejects([](auto& r) { r.tool_choice = si::ToolChoice{si::ToolChoiceMode::none, {}}; });
    rejects([](auto& r) { r.parallel_tool_calls = false; });
    rejects([](auto& r) { r.parallel_tool_calls = true; });
    rejects([](auto& r) { r.response_format_type = "json_object"; });
    rejects([](auto& r) { r.response_format_schema = "{}"; });
}

TEST_CASE("worker: sampling round trip retains uint64 seeds bans and explicit policy") {
    auto sampling = si::SamplingConfig::greedy(7, std::numeric_limits<std::uint64_t>::max());
    sampling.top_p = 0.75f;
    sampling.top_k = 13;
    sampling.min_p = 0.125f;
    sampling.repeat_penalty = 1.25f;
    sampling.typical_p = 0.875f;
    sampling.repeat_last_n = -1;
    sampling.presence_penalty = -0.5f;
    sampling.frequency_penalty = 0.5f;
    sampling.logit_bias = {{3, -std::numeric_limits<float>::infinity()}, {4, 2.0f}};
    sampling.num_ctx = 2048;
    sampling.stop = {"stop", "\nend"};
    sampling.logprobs = true;
    sampling.top_logprobs = 3;
    sampling.explicit_only = true;
    sampling.explicit_fields = si::SamplingConfig::kSeed | si::SamplingConfig::kTopK |
                               si::SamplingConfig::kLogitBias | si::SamplingConfig::kStop;
    const auto wire = wk::sampling_to_json(sampling).dump();
    const auto parsed = si::json::parse(wire);
    REQUIRE(parsed.ok());
    const auto decoded = wk::sampling_from_json(parsed.value());
    REQUIRE_MESSAGE(decoded.ok(), decoded.status().to_string());
    CHECK(decoded->seed == sampling.seed);
    CHECK(decoded->temperature == sampling.temperature);
    CHECK(decoded->top_p == sampling.top_p);
    CHECK(decoded->top_k == sampling.top_k);
    CHECK(decoded->min_p == sampling.min_p);
    CHECK(decoded->repeat_penalty == sampling.repeat_penalty);
    CHECK(decoded->typical_p == sampling.typical_p);
    CHECK(decoded->repeat_last_n == sampling.repeat_last_n);
    CHECK(decoded->presence_penalty == sampling.presence_penalty);
    CHECK(decoded->frequency_penalty == sampling.frequency_penalty);
    REQUIRE(decoded->logit_bias.size() == sampling.logit_bias.size());
    CHECK(decoded->logit_bias[0].token == 3);
    CHECK(decoded->logit_bias[0].bias == -std::numeric_limits<float>::infinity());
    CHECK(decoded->logit_bias[1].bias == 2.0f);
    CHECK(decoded->num_ctx == sampling.num_ctx);
    CHECK(decoded->max_tokens == sampling.max_tokens);
    CHECK(decoded->stop == sampling.stop);
    CHECK(decoded->logprobs == sampling.logprobs);
    CHECK(decoded->top_logprobs == sampling.top_logprobs);
    CHECK(decoded->explicit_only == sampling.explicit_only);
    CHECK(decoded->explicit_fields == sampling.explicit_fields);
}

TEST_CASE("worker: numeric wire values cannot overflow into bans or infinite logprobs") {
    auto sampling = si::SamplingConfig::greedy(2);
    sampling.logit_bias = {{3, -std::numeric_limits<float>::infinity()}};
    auto sampler_wire = wk::sampling_to_json(sampling);
    // Null is the protocol's explicit ban representation. A finite double
    // that overflows float must never be silently converted into that ban.
    const auto explicit_ban = wk::sampling_from_json(sampler_wire);
    REQUIRE(explicit_ban.ok());
    REQUIRE(explicit_ban->logit_bias.size() == 1);
    CHECK(explicit_ban->logit_bias[0].bias == -std::numeric_limits<float>::infinity());
    for (const double bias : {-1e300, 1e300, -101.0, 101.0}) {
        CAPTURE(bias);
        sampler_wire.object().set("logit_bias", si::json::Array{si::json::Object{{"token", 3}, {"bias", bias}}});
        const auto decoded = wk::sampling_from_json(sampler_wire);
        CHECK_FALSE(decoded.ok());
    }

    si::TokenChunk token{"x", 0};
    token.logprob = -0.25f;
    token.top_logprobs = {{"x", -0.25f}};
    const auto valid_token = wk::chunk_to_json(token);
    REQUIRE(wk::chunk_from_json(valid_token).ok());
    for (const double overflow : {-1e300, 1e300}) {
        CAPTURE(overflow);
        auto primary = valid_token;
        primary.object().set("logprob", overflow);
        CHECK_FALSE(wk::chunk_from_json(primary).ok());
        auto alternative = valid_token;
        alternative.object().set("top_logprobs",
            si::json::Array{si::json::Object{{"token", "x"}, {"logprob", overflow}}});
        CHECK_FALSE(wk::chunk_from_json(alternative).ok());
    }
}
