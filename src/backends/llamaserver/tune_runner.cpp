#include "tune_runner.hpp"
#include "llamaserver_protocol.hpp"
#include "log_diagnostics.hpp"
#include "supervisor.hpp"
#include "utf8_path.hpp"
#include "net/http_client.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <thread>

namespace sonder::inference::llamaserver::tune {
namespace {
struct TrackedChild {
    std::atomic<std::uint32_t> pid{0};
    std::atomic<bool> launched{false};
};
class TrackingLauncher final : public ProcessLauncher {
  public:
    TrackingLauncher(std::unique_ptr<ProcessLauncher> launcher, std::shared_ptr<TrackedChild> child)
        : launcher_(std::move(launcher)), child_(std::move(child)) {}
    Result<std::unique_ptr<Process>> start(const ProcessSpec &spec) override {
        auto result = launcher_->start(spec);
        if (result.ok() && result.value()) {
            child_->pid.store(result.value()->pid());
            child_->launched.store(true);
        }
        return result;
    }
  private:
    std::unique_ptr<ProcessLauncher> launcher_;
    std::shared_ptr<TrackedChild> child_;
};
std::chrono::milliseconds remaining(Deadline deadline) {
    return std::max(std::chrono::milliseconds(1),
                    std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()));
}
Result<json::Value> http_exchange(std::uint16_t port, const std::string &method, const std::string &path,
                                const json::Value &body, Deadline deadline, const CancellationToken &cancel) {
    if (Clock::now() >= deadline) return Status(ErrorCode::timeout, "calibration budget exhausted");
    net::HttpRequest request;
    request.host = "127.0.0.1";
    request.port = port;
    request.method = method;
    request.target = path;
    if (method != "GET") request.body = body.dump();
    request.total_timeout = remaining(deadline);
    request.connect_timeout = std::min(request.total_timeout, std::chrono::milliseconds(2000));
    std::string response;
    auto received = net::http_request_buffered(request, response, cancel, 8 * 1024 * 1024);
    if (!received.ok()) return received.status();
    if (received.value().status != 200)
        return Status(ErrorCode::backend_error, path + " returned HTTP " + std::to_string(received.value().status));
    auto parsed = json::parse(response);
    if (!parsed.ok()) return parsed.status();
    if (!parsed.value().is_object() || parsed.value().find("error"))
        return Status(ErrorCode::protocol_error, path + " returned an error or non-object response");
    return parsed;
}
bool released(std::uint32_t pid, GpuCounterSource &source, std::chrono::milliseconds timeout) {
    const auto until = Clock::now() + timeout;
    unsigned consecutive = 0;
    do {
        const auto counters = source.read(pid);
        if (!counters.ok()) return false; // errors are never evidence of zero usage
        const auto sample = summarize_gpu_counters(pid, counters.value());
        if (sample.dedicated_bytes == 0 && sample.shared_bytes == 0) {
            if (++consecutive == 2) return true;
        } else consecutive = 0;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    } while (Clock::now() < until);
    return false;
}
Status workload(const Exchange &exchange, std::uint16_t port, const json::Array &pattern,
                std::size_t prompt_size, std::uint32_t decode_size, Deadline deadline,
                const CancellationToken &cancel, Timings &timings) {
    json::Array tokens;
    tokens.reserve(prompt_size);
    for (std::size_t i = 0; i < prompt_size; ++i) tokens.push_back(pattern[i % pattern.size()]);
    const json::Value body = json::Object{{"prompt", std::move(tokens)}, {"n_predict", decode_size},
        {"temperature", 0}, {"seed", 42}, {"ignore_eos", true}, {"cache_prompt", false},
        {"id_slot", 0}, {"stream", false}};
    auto response = exchange(port, "POST", "/completion", body, deadline, cancel);
    if (!response.ok()) return response.status();
    if (auto st = parse_timings(response.value(), timings); !st.ok()) return st;
    if (!timings.prompt_present || timings.prompt_tokens < prompt_size || timings.prompt_tokens > prompt_size + 4 ||
        !timings.completion_present || timings.completion_tokens != decode_size ||
        timings.cached_tokens != 0 || timings.prompt_ns == 0 || timings.eval_ns == 0)
        return {ErrorCode::protocol_error, "workload was truncated/cached or did not return complete positive timings"};
    return {};
}
struct DecodeWorkload {
    const char *name;
    double weight;
    std::string prompt;
    std::uint32_t max_tokens;
};
std::array<DecodeWorkload, 4> decode_workloads() {
    // A complete, fixed 120-line file. Keep real edit traffic (including the
    // unchanged surroundings) instead of a repeated token sequence.
    const std::string code = R"(from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    price: Decimal
    category: str


@dataclass(frozen=True)
class LineItem:
    product: Product
    quantity: int


CATALOG = (
    Product("A01", "Notebook", Decimal("4.50"), "paper"),
    Product("A02", "Sketchbook", Decimal("8.25"), "paper"),
    Product("A03", "Envelope pack", Decimal("3.00"), "paper"),
    Product("A04", "Index cards", Decimal("2.75"), "paper"),
    Product("A05", "Graph paper", Decimal("5.00"), "paper"),
    Product("A06", "Writing pad", Decimal("3.40"), "paper"),
    Product("B01", "Blue pen", Decimal("1.20"), "writing"),
    Product("B02", "Black pen", Decimal("1.20"), "writing"),
    Product("B03", "Red pen", Decimal("1.20"), "writing"),
    Product("B04", "Pencil", Decimal("0.80"), "writing"),
    Product("B05", "Marker", Decimal("2.30"), "writing"),
    Product("B06", "Highlighter", Decimal("1.90"), "writing"),
    Product("C01", "Ruler", Decimal("2.00"), "tools"),
    Product("C02", "Eraser", Decimal("0.90"), "tools"),
    Product("C03", "Sharpener", Decimal("1.50"), "tools"),
    Product("C04", "Scissors", Decimal("4.80"), "tools"),
    Product("C05", "Glue stick", Decimal("1.70"), "tools"),
    Product("C06", "Tape", Decimal("2.60"), "tools"),
    Product("D01", "Folder", Decimal("1.10"), "storage"),
    Product("D02", "Binder", Decimal("3.80"), "storage"),
    Product("D03", "Paper clips", Decimal("1.40"), "storage"),
    Product("D04", "Document tray", Decimal("6.50"), "storage"),
    Product("D05", "Archive box", Decimal("7.20"), "storage"),
    Product("D06", "Label pack", Decimal("2.40"), "storage"),
)


def lookup_product(sku: str) -> Product:
    for product in CATALOG:
        if product.sku == sku:
            return product
    raise KeyError(sku)


def line_subtotal(item: LineItem) -> Decimal:
    if item.quantity < 0:
        raise ValueError("quantity must be nonnegative")
    return item.product.price * item.quantity


def calculate_total(items: Iterable[LineItem]) -> Decimal:
    total = Decimal("0.00")
    for item in items:
        total += line_subtotal(item)
    return total.quantize(Decimal("0.01"))


def apply_discount(amount: Decimal, percent: int) -> Decimal:
    if not 0 <= percent <= 100:
        raise ValueError("discount must be between zero and one hundred")
    multiplier = Decimal(100 - percent) / Decimal(100)
    return (amount * multiplier).quantize(Decimal("0.01"))


def category_names() -> list[str]:
    return sorted({product.category for product in CATALOG})


def products_in_category(category: str) -> list[Product]:
    return [product for product in CATALOG if product.category == category]


def format_money(amount: Decimal) -> str:
    return f"${amount:.2f}"


def receipt_lines(items: list[LineItem]) -> list[str]:
    lines = ["Stationery order", "---------------"]
    for item in items:
        amount = format_money(line_subtotal(item))
        lines.append(f"{item.quantity} x {item.product.name}: {amount}")
    lines.append(f"Total: {format_money(calculate_total(items))}")
    return lines


def create_order(entries: list[tuple[str, int]]) -> list[LineItem]:
    result = []
    for sku, quantity in entries:
        if quantity <= 0:
            raise ValueError("order quantities must be positive")
        result.append(LineItem(lookup_product(sku), quantity))
    return result


def order_summary(items: list[LineItem], discount: int = 0) -> dict:
    subtotal = calculate_total(items)
    due = apply_discount(subtotal, discount)
    return {"lines": len(items), "subtotal": subtotal, "due": due}


def main() -> None:
    order = create_order([("A01", 3), ("B01", 2), ("C02", 1)])
    for line in receipt_lines(order):
        print(line)
    summary = order_summary(order, discount=10)
    print(f"Discounted total: {format_money(summary['due'])}")


if __name__ == "__main__":
    main()
)";
    return {{
        {"prose", 0.4, "Write a 200-token essay explaining how a town can conserve water during a dry summer. "
            "Discuss household use, farms, and a practical tradeoff. Use connected prose with a conclusion.", 256},
        {"code_edit", 0.3, "In this 120-line Python file, rename calculate_total to order_total, including every call. "
            "Return the complete edited file, keeping all other behaviour unchanged.\n```python\n" + code + "```", 2048},
        {"tool_json", 0.2, "You are taking one step in a coding agent loop. Task: rename calculate_total to order_total "
            "in invoice.py and update its callers. You have not read the file yet. Available tools: "
            "file_read(path: string), text_patch(path: string, old: string, new: string), test_run(path: string). "
            "Choose the next action. Return only a JSON object with keys tool and arguments. "
            "Do not invent file contents or claim that tests ran.", 256},
        {"reasoning", 0.1, "A cyclist rides 18 kilometres to a village at 12 kilometres per hour, rests for 15 minutes, "
            "then returns along the same route at 18 kilometres per hour. What is the average speed over the entire "
            "trip, including the rest? Give a short calculation and the answer in kilometres per hour.", 256}
    }};
}
Status decode_workload(const Exchange &exchange, std::uint16_t port, const DecodeWorkload &work,
                       bool mtp, Deadline deadline, const CancellationToken &cancel, WorkloadMeasurement &out) {
    const json::Value body = json::Object{
        {"messages", json::Array{
            json::Object{{"role", "system"}, {"content", "Follow the user's instructions and answer directly."}},
            json::Object{{"role", "user"}, {"content", work.prompt}}}},
        {"chat_template_kwargs", json::Object{{"enable_thinking", false}}},
        {"max_tokens", work.max_tokens}, {"temperature", 0}, {"seed", 42},
        {"cache_prompt", false}, {"stream", false}};
    auto response = exchange(port, "POST", "/v1/chat/completions", body, deadline, cancel);
    if (!response.ok()) return response.status();
    const auto *choices = response.value().find("choices");
    if (!choices || !choices->is_array() || choices->as_array().size() != 1)
        return {ErrorCode::protocol_error, "chat workload did not return one choice"};
    const auto &choice = choices->as_array().front();
    const auto *message = choice.find("message");
    const auto *content = message ? message->find("content") : nullptr;
    const auto *finish = choice.find("finish_reason");
    if (!content || !content->is_string() || content->as_string().find_first_not_of(" \t\r\n") == std::string::npos ||
        !finish || (finish->as_string() != "stop" && finish->as_string() != "length"))
        return {ErrorCode::protocol_error, "chat workload returned no usable text or finish reason"};
    Timings timings;
    if (auto status = parse_timings(response.value(), timings); !status.ok()) return status;
    if (!timings.prompt_present || timings.prompt_tokens == 0 || !timings.completion_present ||
        timings.completion_tokens == 0 || timings.completion_tokens > work.max_tokens ||
        timings.cached_tokens != 0 || timings.prompt_ns == 0 || timings.eval_ns == 0)
        return {ErrorCode::protocol_error, "chat workload was cached or did not return positive bounded timings"};
    if (mtp && (!timings.draft_present || timings.draft_tokens == 0 || !timings.draft_accepted_present))
        return {ErrorCode::protocol_error, "MTP requested but draft activity and acceptance were not measured"};
    if (!mtp && (timings.draft_tokens != 0 || timings.draft_accepted_tokens != 0))
        return {ErrorCode::protocol_error, "baseline unexpectedly used speculative decoding"};
    out.name = work.name;
    out.weight = work.weight;
    out.tok_s = static_cast<double>(timings.completion_tokens) * 1e9 / static_cast<double>(timings.eval_ns);
    if (timings.draft_present) out.draft_n = timings.draft_tokens;
    if (timings.draft_accepted_present) out.draft_n_accepted = timings.draft_accepted_tokens;
    return {};
}
} // namespace

Measurement run_candidate(const Options &o, const Candidate &candidate, Phase phase, Deadline deadline,
                          const std::string &log_path, const Dependencies &d) {
    Measurement result;
    result.candidate = candidate;
    result.phase = phase;
    result.memory_released = true; // nothing launched yet
    if (!d.counters || !d.counters->supported()) {
        result.verdict = Verdict::unsupported;
        result.detail = "a supported per-process GPU probe is required; no child started";
        result.stop_search = true;
        return result;
    }
    if (Clock::now() >= deadline || d.interrupted()) {
        result.verdict = d.interrupted() ? Verdict::cancelled : Verdict::timed_out;
        return result;
    }
    { // Ensure fake launchers and production both start with an empty log.
        std::ofstream log(utf8_path(log_path), std::ios::binary | std::ios::trunc);
        if (!log) { result.detail = "cannot create candidate log"; result.stop_search = true; return result; }
    }
    const Exchange exchange = d.exchange ? d.exchange : http_exchange;
    CancellationSource cancellation;
    const auto token = cancellation.token();
    auto tracked = std::make_shared<TrackedChild>();
    SupervisorOptions supervisor_options;
    supervisor_options.executable = o.executable;
    supervisor_options.arguments = arguments(o, candidate);
    supervisor_options.environment = o.environment;
    supervisor_options.output_file = log_path;
    supervisor_options.max_restarts = 0;
    supervisor_options.readiness_timeout = std::min(remaining(deadline), std::chrono::milliseconds(120000));
    supervisor_options.shutdown_timeout = std::chrono::milliseconds(2000);
    supervisor_options.spill_guard.enabled = false; // sampler below owns the same probe, including workload peaks
    supervisor_options.health_check = [&](std::uint16_t port, std::chrono::milliseconds timeout) {
        return exchange(port, "GET", "/health", {}, std::min(deadline, Clock::now() + timeout), token).ok();
    };
    auto launcher = d.launcher();
    if (!launcher) { result.detail = "no process launcher"; result.stop_search = true; return result; }
    Supervisor supervisor(std::move(supervisor_options), std::make_unique<TrackingLauncher>(std::move(launcher), tracked));
    LogTail log(log_path, true);
    LogDiagnostics diagnostics;
    bool sampled = false, spilled = false;
    std::atomic<bool> ready{false};
    Status probe_status;
    std::string fatal_log;
    const auto sample = [&] {
        const auto pid = tracked->pid.load();
        if (!pid) return;
        const auto raw = d.counters->read(pid);
        if (!raw.ok()) { probe_status = raw.status(); cancellation.cancel(); return; }
        const auto memory = summarize_gpu_counters(pid, raw.value());
        const auto matching = [pid](const CounterInstance &i) { return is_process_instance(i.name, pid); };
        const bool complete = std::any_of(raw.value().dedicated.begin(), raw.value().dedicated.end(), matching) &&
                              std::any_of(raw.value().shared.begin(), raw.value().shared.end(), matching);
        if (!complete && ready.load()) {
            probe_status = {ErrorCode::unavailable, "live GPU counter instances disappeared"};
            cancellation.cancel();
        }
        if (complete) {
            sampled = true;
            result.dedicated_bytes = std::max(result.dedicated_bytes.value_or(0), memory.dedicated_bytes);
            result.shared_bytes = std::max(result.shared_bytes.value_or(0), memory.shared_bytes);
            spilled = spilled || is_spilled(memory, spill_guard_for(o.grid, candidate), candidate.ctx);
            if (spilled) cancellation.cancel();
        }
        std::error_code ec;
        const auto size = std::filesystem::file_size(utf8_path(log_path), ec);
        if (ec || size > 8 * 1024 * 1024) {
            probe_status = {ErrorCode::io_error, "candidate log unavailable or exceeds 8 MiB"};
            cancellation.cancel();
            return;
        }
        if (auto st = log.poll(diagnostics); !st.ok()) {
            probe_status = st;
            cancellation.cancel();
        }
        for (const auto &warning : diagnostics.warnings()) {
            if (warning.code == "kv_kernel_f16_fallback") { result.slow_kernel = true; cancellation.cancel(); }
            if (warning.code == "no_gpu_device" || warning.code == "gpu_init_failed" ||
                (candidate.mtp != 0 && warning.code == "mtp_tensors_ignored")) {
                fatal_log = warning.code;
                cancellation.cancel();
            }
        }
    };
    // One thread owns the probe and log until joined. It also cancels blocked
    // readiness/HTTP operations at the deadline or first unsafe observation.
    std::jthread monitor([&](std::stop_token stop) {
        try {
            while (!stop.stop_requested()) {
                if (Clock::now() >= deadline || d.interrupted()) cancellation.cancel();
                sample();
                const auto until = Clock::now() + d.sample_interval;
                while (!stop.stop_requested() && Clock::now() < until)
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        } catch (const std::exception &e) {
            probe_status = {ErrorCode::io_error, e.what()};
            cancellation.cancel();
        }
    });
    Status status;
    try {
        auto started = supervisor.start(token);
        if (!started.ok()) status = started.status();
        else {
            const auto port = started.value();
            auto props = exchange(port, "GET", "/props", {}, deadline, token);
            ServerProps parsed;
            if (!props.ok()) status = props.status();
            else status = parse_props(props.value(), parsed);
            result.served_ctx = parsed.n_ctx;
            if (status.ok() && (parsed.n_ctx != candidate.ctx || parsed.total_slots != 1))
                status = {ErrorCode::protocol_error, "/props did not confirm the requested context and single slot"};
            if (status.ok()) ready.store(true);
            // Exact-token prefill probes qualify memory, independently of the
            // natural chat decode suite. They never contribute to decode rank.
            json::Array pattern;
            if (status.ok()) {
                auto tokens = exchange(port, "POST", "/tokenize", json::Object{
                    {"content", "Calibration prefill measures allocated context and memory usage."},
                    {"add_special", false}}, deadline, token);
                if (!tokens.ok()) status = tokens.status();
                else {
                    const auto *list = tokens.value().find("tokens");
                    if (!list || !list->is_array() || list->as_array().empty() || list->as_array().size() > 4096)
                        status = {ErrorCode::protocol_error, "/tokenize returned no bounded token array"};
                    else {
                        for (const auto &id : list->as_array())
                            if (!id.is_integer() || id.as_int(-1) < 0 || id.as_uint() > 2147483647)
                                status = {ErrorCode::protocol_error, "/tokenize returned an invalid token id"};
                        pattern = list->as_array();
                    }
                }
            }
            Timings timings;
            if (status.ok() && phase == Phase::benchmark) {
                for (const auto &work : decode_workloads()) {
                    if (token.cancelled() || d.interrupted() || Clock::now() >= deadline) {
                        status = {ErrorCode::cancelled, "decode suite cancelled before the next class"};
                        break;
                    }
                    WorkloadMeasurement measured;
                    status = decode_workload(exchange, port, work, candidate.mtp != 0, deadline, token, measured);
                    if (!status.ok()) {
                        status = {status.code(), std::string(work.name) + ": " + status.message()};
                        break;
                    }
                    result.workloads.push_back(std::move(measured));
                }
                if (status.ok()) result.short_tps = weighted_decode_tps(result.workloads);
            }
            if (status.ok()) {
                status = workload(exchange, port, pattern, 8192, 1, deadline, token, timings);
                if (status.ok()) result.prefill_tps = static_cast<double>(timings.prompt_tokens) * 1e9 / static_cast<double>(timings.prompt_ns);
            }
            if (status.ok() && phase == Phase::benchmark && o.thorough) {
                status = workload(exchange, port, pattern, static_cast<std::size_t>(candidate.ctx - 1024), 1, deadline, token, timings);
                if (status.ok()) result.long_prefill_tps = static_cast<double>(timings.prompt_tokens) * 1e9 / static_cast<double>(timings.prompt_ns);
            }
            if (status.ok() && !supervisor.running()) status = {ErrorCode::backend_error, "child exited during calibration"};
        }
    } catch (const std::exception &e) { status = {ErrorCode::internal, e.what()}; }
    monitor.request_stop();
    monitor.join();
    try {
        sample(); // explicit final sample also makes very short fake workloads observable
        // Drain the bounded remainder, including a non-newline-terminated warning.
        for (int i = 0; i < 32; ++i) {
            if (auto st = log.poll(diagnostics); !st.ok()) { probe_status = st; break; }
        }
        diagnostics.finish();
        for (const auto &warning : diagnostics.warnings()) {
            if (warning.code == "kv_kernel_f16_fallback") result.slow_kernel = true;
            if (warning.code == "no_gpu_device" || warning.code == "gpu_init_failed" ||
                (candidate.mtp != 0 && warning.code == "mtp_tensors_ignored")) fatal_log = warning.code;
        }
    } catch (const std::exception &e) { probe_status = {ErrorCode::io_error, e.what()}; }
    supervisor.stop();
    const auto pid = tracked->pid.load();
    try {
        result.memory_released = !tracked->launched.load() || (pid != 0 && released(pid, *d.counters, d.release_timeout));
    } catch (...) { result.memory_released = false; }
    if (!result.memory_released) {
        result.verdict = Verdict::release_failed;
        result.stop_search = true;
        result.detail = "child stopped but GPU memory release could not be confirmed; no next candidate";
    } else if (d.interrupted()) result.verdict = Verdict::cancelled;
    else if (!probe_status.ok()) {
        result.verdict = Verdict::unsupported;
        result.detail = probe_status.message();
        result.stop_search = true;
    } else if (spilled) result.verdict = Verdict::spill;
    else if (result.slow_kernel) result.verdict = Verdict::slow_kernel;
    else if (!fatal_log.empty()) { result.verdict = Verdict::error; result.detail = fatal_log; }
    else if (Clock::now() >= deadline) result.verdict = Verdict::timed_out;
    else if (!status.ok()) { result.verdict = Verdict::error; result.detail = status.message(); }
    else if (!sampled || !result.dedicated_bytes || *result.dedicated_bytes == 0) {
        result.verdict = Verdict::unsupported;
        result.detail = "no live dedicated GPU allocation observed; cannot establish non-spilling operation";
        result.stop_search = true;
    } else result.verdict = Verdict::clean;
    return result;
}

Result<std::string> executable_help(const Options &o, Deadline deadline, const std::string &path, const Dependencies &d) {
    if (Clock::now() >= deadline || d.interrupted()) return Status(ErrorCode::cancelled, "help probe cancelled");
    ProcessSpec spec;
    spec.executable = o.executable;
    spec.arguments = {"--help"};
    spec.port = 1; // launcher requires a nonzero identity; --help never receives a port argument
    spec.environment = o.environment;
    spec.output_file = path;
    auto launcher = d.launcher();
    if (!launcher) return Status(ErrorCode::internal, "no process launcher");
    auto process = launcher->start(spec);
    if (!process.ok()) return process.status();
    if (!process.value()) return Status(ErrorCode::internal, "launcher returned no process");
    const auto until = std::min(deadline, Clock::now() + std::chrono::seconds(15));
    bool oversized = false;
    while (process.value()->running() && Clock::now() < until && !d.interrupted()) {
        std::error_code ec;
        const auto size = std::filesystem::file_size(utf8_path(path), ec);
        if (!ec && size > 1024 * 1024) { oversized = true; break; }
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    const bool timed_out = process.value()->running();
    process.value()->stop();
    if (oversized || timed_out || d.interrupted()) return Status(ErrorCode::unavailable, "--help probe exceeded its bounds or was cancelled");
    std::ifstream file(utf8_path(path), std::ios::binary);
    if (!file) return Status(ErrorCode::io_error, "cannot read executable help");
    std::string text(1024 * 1024 + 1, '\0');
    file.read(text.data(), static_cast<std::streamsize>(text.size()));
    text.resize(static_cast<std::size_t>(file.gcount()));
    if (file.bad() || text.size() > 1024 * 1024) return Status(ErrorCode::io_error, "executable help exceeds bounds");
    return text;
}
} // namespace sonder::inference::llamaserver::tune
