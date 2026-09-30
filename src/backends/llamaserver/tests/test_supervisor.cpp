#include "../process.hpp"
#include "../supervisor.hpp"
#include "fake_process.hpp"
#include "fake_server.hpp"
#include <atomic>
#include <condition_variable>
#include <doctest/doctest.h>
#include <mutex>
#include <thread>

#if !defined(_WIN32)
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace sonder::inference;
using namespace sonder::inference::llamaserver;
using namespace sonder_test;

TEST_CASE("supervisor polls readiness false false true and appends loopback binding") {
    auto state = std::make_shared<LaunchState>();
    auto probes = std::make_shared<std::atomic<unsigned>>(0);
    auto o = base_options();
    o.health_check = [probes](std::uint16_t port, std::chrono::milliseconds) {
        CHECK(port != 0);
        return probes->fetch_add(1) >= 2;
    };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    REQUIRE(state->specs.size() == 1);
    const auto &a = state->specs.front().arguments;
    CHECK(a[a.size() - 4] == "--host");
    CHECK(a[a.size() - 3] == "127.0.0.1");
    CHECK(a[a.size() - 2] == "--port");
    s.stop();
}

TEST_CASE("supervisor waits through real HTTP health 503 responses before 200") {
    struct HttpChild final : Process {
        std::shared_ptr<sonder_test::FakeLlamaServer> server;
        explicit HttpChild(std::shared_ptr<sonder_test::FakeLlamaServer> s) : server(std::move(s)) {}
        bool running() const override { return server != nullptr; }
        void stop(std::chrono::milliseconds) override { server.reset(); }
    };
    struct HttpLauncher final : ProcessLauncher {
        std::weak_ptr<sonder_test::FakeLlamaServer> server;
        Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
            auto upstream = std::make_shared<sonder_test::FakeLlamaServer>(spec.port, 2);
            server = upstream;
            return std::unique_ptr<Process>(new HttpChild(std::move(upstream)));
        }
    };
    auto launcher = std::make_unique<HttpLauncher>();
    auto *handle = launcher.get();
    Supervisor supervisor(base_options(), std::move(launcher));
    auto ready = supervisor.start();
    REQUIRE_MESSAGE(ready.ok(), ready.status().to_string());
    auto server = handle->server.lock();
    REQUIRE(server);
    CHECK(server->health_requests() == 3);
    server.reset();
    supervisor.stop();
    CHECK(handle->server.expired());
}

TEST_CASE("supervisor timeout stops child") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.readiness_timeout = std::chrono::milliseconds(15);
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return false; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    CHECK_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::timeout);
    REQUIRE(state->children.size() == 1);
    CHECK(state->children.front()->stops.load() >= 1);
}

TEST_CASE("supervisor readiness timeout waits for the monitor to stop a slow-probed child") {
    // Deterministic form of a CI race: the health probe outlives start()'s own
    // deadline, so start() must wait for the monitor to stop the child and
    // publish its failure instead of returning while the child still runs.
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.readiness_timeout = std::chrono::milliseconds(15);
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        return false;
    };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    const auto r = s.start();
    CHECK_FALSE(r.ok());
    CHECK(r.status().code() == ErrorCode::timeout);
    REQUIRE(state->children.size() == 1);
    CHECK(state->children.front()->stops.load() >= 1);
    CHECK_FALSE(state->children.front()->alive.load());
}

TEST_CASE("supervisor crash retries twice, caps backoff, then exhausts") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.max_restarts = 2;
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    REQUIRE(s.start().ok());
    child_at(state, 0)->alive.store(false);
    REQUIRE(wait_for_starts(state, 2));
    child_at(state, 1)->alive.store(false);
    REQUIRE(wait_for_starts(state, 3));
    child_at(state, 2)->alive.store(false);
    {
        std::lock_guard lock(state->mutex);
        REQUIRE(state->starts.size() == 3);
        CHECK(state->starts[1] - state->starts[0] >= o.restart_initial_backoff);
        CHECK(state->starts[2] - state->starts[1] >= o.restart_max_backoff);
    }
    // Wait for the final state, not an earlier transient backoff failure.
    for (int i = 0; i < 1000 && child_at(state, 2)->stops.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    CHECK_FALSE(s.start().ok());
    CHECK_FALSE(s.start().ok());
    s.stop();
    CHECK(state->starts.size() == 3);
}

TEST_CASE("supervisor stop during readiness prevents later launches") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return false; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    std::thread t([&] { (void)s.start(); });
    const bool launched = wait_for_starts(state, 1);
    s.stop();
    t.join();
    REQUIRE(launched);
    CHECK(state->starts.size() == 1);
    CHECK_FALSE(s.start().ok());
    CHECK(child_at(state, 0)->stops.load() == 1);
}

TEST_CASE("concurrent start calls launch only one child") {
    auto state = std::make_shared<LaunchState>();
    auto o = base_options();
    o.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    Supervisor s(o, std::make_unique<FakeLauncher>(state));
    Result<std::uint16_t> a(Status(ErrorCode::internal, "unset")), b(Status(ErrorCode::internal, "unset"));
    std::thread x([&] { a = s.start(); });
    std::thread y([&] { b = s.start(); });
    x.join();
    y.join();
    CHECK(a.ok());
    CHECK(b.ok());
    CHECK(state->starts.size() == 1);
    s.stop();
}

TEST_CASE("startup death fails promptly without treating a healthy port as the child") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.max_restarts = 0;
    options.health_check = [state](std::uint16_t, std::chrono::milliseconds) {
        child_at(state, 0)->alive.store(false);
        return true;
    };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    auto result = supervisor.start();
    CHECK_FALSE(result.ok());
    CHECK(result.status().code() == ErrorCode::unavailable);
    CHECK(child_at(state, 0)->stops.load() == 1);
}

TEST_CASE("shutdown interrupts restart backoff and is terminal") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    options.restart_initial_backoff = std::chrono::milliseconds(5000);
    options.restart_max_backoff = options.restart_initial_backoff;
    options.health_check = [](std::uint16_t, std::chrono::milliseconds) { return true; };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    REQUIRE(supervisor.start().ok());
    auto child = child_at(state, 0);
    child->alive.store(false);
    for (int i = 0; i < 1000 && child->stops.load() == 0; ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const auto before = std::chrono::steady_clock::now();
    supervisor.stop();
    CHECK(std::chrono::steady_clock::now() - before < std::chrono::seconds(2));
    CHECK(state->starts.size() == 1);
    CHECK_FALSE(supervisor.start().ok());
}

TEST_CASE("cancelled readiness waiter leaves supervisor usable for other requests") {
    auto state = std::make_shared<LaunchState>();
    auto options = base_options();
    std::atomic<bool> ready{false};
    options.health_check = [&](std::uint16_t, std::chrono::milliseconds) { return ready.load(); };
    Supervisor supervisor(options, std::make_unique<FakeLauncher>(state));
    CancellationSource cancel;
    Result<std::uint16_t> result(Status(ErrorCode::internal, "unset"));
    std::thread waiter([&] { result = supervisor.start(cancel.token()); });
    const bool launched = wait_for_starts(state, 1);
    cancel.cancel();
    waiter.join();
    REQUIRE(launched);
    CHECK(result.status().code() == ErrorCode::cancelled);
    ready.store(true);
    CHECK(supervisor.start().ok());
    supervisor.stop();
    CHECK(state->starts.size() == 1);
}

TEST_CASE("argument validation rejects NUL and host or port overrides") {
    CHECK_FALSE(validate_process_arguments({"--"}).ok());
    CHECK_FALSE(validate_process_arguments({"--host=0.0.0.0"}).ok());
    CHECK_FALSE(validate_process_arguments({"--port", "1"}).ok());
    CHECK_FALSE(validate_process_arguments({std::string("--model\0hidden", 13)}).ok());
    CHECK(validate_process_arguments({"-p", "prompt"}).ok());
}

#if !defined(_WIN32)
namespace {
std::string temporary_pid_file() {
    char path[] = "/tmp/sonder-llamaserver-pid-XXXXXX";
    const int fd = mkstemp(path);
    if (fd < 0)
        return {};
    close(fd);
    unlink(path);
    return path;
}

bool read_pid(const std::string &path, pid_t &pid) {
    std::ifstream input(path);
    long value = 0;
    if (!(input >> value) || value <= 0)
        return false;
    pid = static_cast<pid_t>(value);
    return true;
}

bool wait_for_pid_file(const std::string &path, pid_t &pid, std::chrono::milliseconds timeout) {
    const auto until = std::chrono::steady_clock::now() + timeout;
    while (std::chrono::steady_clock::now() < until) {
        if (read_pid(path, pid))
            return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return read_pid(path, pid);
}

struct PidFileCleanup {
    std::vector<std::string> paths;
    ~PidFileCleanup() {
        for (const auto &path : paths)
            unlink(path.c_str());
    }
};

bool process_gone(pid_t pid) {
    if (kill(pid, 0) != 0)
        return errno == ESRCH;

    // A killed, reparented child can briefly remain as a zombie while init
    // (or launchd) reaps it. It no longer owns model/GPU resources, so accept
    // that state as gone while still rejecting a live process.
    char command[64]{};
    std::snprintf(command, sizeof(command), "ps -o state= -p %ld", static_cast<long>(pid));
    FILE *pipe = popen(command, "r");
    if (!pipe)
        return false;
    char state[16]{};
    const bool zombie =
        std::fgets(state, sizeof(state), pipe) != nullptr && (state[0] == 'Z' || state[1] == 'Z');
    (void)pclose(pipe);
    return zombie;
}

bool wait_for_exit(pid_t pid, int &status, std::chrono::milliseconds timeout) {
    const auto until = std::chrono::steady_clock::now() + timeout;
    for (;;) {
        const pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid)
            return true;
        if (result < 0 && errno != EINTR && errno != ECHILD)
            return false;
        if (std::chrono::steady_clock::now() >= until)
            return false;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void helper_spawn_long_running_children(const std::string &first_file, const std::string &second_file,
                                        const std::string &ready_file) {
    std::unique_ptr<Process> first;
    std::unique_ptr<Process> second;
    std::thread launcher([&] {
        ProcessSpec first_spec;
        first_spec.executable = "/bin/sh";
        first_spec.arguments = {"-c", "trap '' TERM; echo $$ > '" + first_file + "'; while :; do :; done"};
        first_spec.port = 1;
        first_spec.shutdown_timeout = std::chrono::seconds(30);
        auto first_result = make_process_launcher()->start(first_spec);
        if (!first_result.ok())
            return;
        first = std::move(first_result.value());

        ProcessSpec second_spec = first_spec;
        second_spec.arguments = {"-c", "trap '' TERM; echo $$ > '" + second_file + "'; while :; do :; done"};
        auto second_result = make_process_launcher()->start(second_spec);
        if (second_result.ok())
            second = std::move(second_result.value());
    });
    launcher.join();
    if (!first || !second)
        _exit(111);
    // Publish readiness only after the launching thread has exited. A
    // thread-bound PDEATHSIG implementation must not pass this regression.
    {
        std::ofstream ready(ready_file);
        ready << getpid() << '\n';
    }
    for (;;) {
        pause();
    }
}
} // namespace

TEST_CASE("POSIX child is gone when launcher parent is SIGKILLed") {
    PidFileCleanup files;
    const auto first_file = temporary_pid_file();
    const auto second_file = temporary_pid_file();
    const auto ready_file = temporary_pid_file();
    REQUIRE_FALSE(first_file.empty());
    REQUIRE_FALSE(second_file.empty());
    REQUIRE_FALSE(ready_file.empty());
    files.paths = {first_file, second_file, ready_file};

    const pid_t helper = fork();
    REQUIRE(helper >= 0);
    if (helper == 0)
        helper_spawn_long_running_children(first_file, second_file, ready_file);

    pid_t first_child = -1;
    pid_t second_child = -1;
    pid_t ready_parent = -1;
    const bool started = wait_for_pid_file(ready_file, ready_parent, std::chrono::seconds(3)) &&
                         wait_for_pid_file(first_file, first_child, std::chrono::seconds(2)) &&
                         wait_for_pid_file(second_file, second_child, std::chrono::seconds(2));
    CHECK(started);
    if (started) {
        CHECK(ready_parent == helper);
        CHECK_FALSE(process_gone(first_child));
        CHECK_FALSE(process_gone(second_child));
        CHECK(kill(helper, SIGKILL) == 0);
        int status = 0;
        CHECK(wait_for_exit(helper, status, std::chrono::seconds(2)));
        CHECK(WIFSIGNALED(status));

        bool first_gone = false;
        bool second_gone = false;
        const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
        while (std::chrono::steady_clock::now() < until) {
            first_gone = first_gone || process_gone(first_child);
            second_gone = second_gone || process_gone(second_child);
            if (first_gone && second_gone)
                break;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        CHECK(first_gone);
        CHECK(second_gone);
        if (!first_gone) {
            const pid_t group = getpgid(first_child);
            if (group > 0 && group != getpgrp())
                kill(-group, SIGKILL);
            else
                kill(first_child, SIGKILL);
        }
        if (!second_gone) {
            const pid_t group = getpgid(second_child);
            if (group > 0 && group != getpgrp())
                kill(-group, SIGKILL);
            else
                kill(second_child, SIGKILL);
        }
    } else {
        kill(helper, SIGKILL);
        int status = 0;
        (void)wait_for_exit(helper, status, std::chrono::seconds(2));
        // Also clean up partial startup when running against a broken
        // launcher; do not leave a busy fake server behind after a failure.
        if (read_pid(first_file, first_child) && !process_gone(first_child))
            kill(first_child, SIGKILL);
        if (read_pid(second_file, second_child) && !process_gone(second_child))
            kill(second_child, SIGKILL);
    }
}

TEST_CASE("POSIX SIGTERM shutdown returns when child exits") {
    PidFileCleanup files;
    const auto pid_file = temporary_pid_file();
    REQUIRE_FALSE(pid_file.empty());
    files.paths.push_back(pid_file);
    ProcessSpec spec;
    spec.executable = "/bin/sh";
    spec.arguments = {"-c", "trap 'exit 0' TERM; echo $$ > '" + pid_file + "'; while :; do :; done"};
    spec.port = 1;
    spec.shutdown_timeout = std::chrono::seconds(2);
    auto result = make_process_launcher()->start(spec);
    REQUIRE_MESSAGE(result.ok(), result.status().to_string());
    auto process = std::move(result.value());
    pid_t child = -1;
    CHECK(wait_for_pid_file(pid_file, child, std::chrono::seconds(1)));

    const auto started = std::chrono::steady_clock::now();
    process->stop(std::chrono::seconds(2));
    const auto elapsed = std::chrono::steady_clock::now() - started;
    CHECK(elapsed < std::chrono::milliseconds(500));
    CHECK_FALSE(process->running());
    if (child > 0)
        CHECK(process_gone(child));
}
#endif
