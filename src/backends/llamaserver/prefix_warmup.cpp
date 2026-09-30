#include "prefix_warmup.hpp"

#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <utility>

#include "llamaserver_protocol.hpp"
#include "slot_affinity.hpp"

namespace sonder::inference::llamaserver {
namespace {
using Clock = std::chrono::steady_clock;
using Milliseconds = std::chrono::milliseconds;

Status error(std::string message) { return {ErrorCode::backend_error, std::move(message)}; }
Status cancelled() { return {ErrorCode::cancelled, "warm-up cancelled"}; }

Result<json::Value> fetch(net::HttpRequest request, const CancellationToken &cancel) {
    std::string body;
    const auto response = net::http_request_buffered(request, body, cancel, 1024 * 1024);
    if (cancel.cancelled())
        return cancelled();
    if (!response.ok())
        return error("warm-up HTTP request failed (" + std::string(to_string(response.status().code())) + ")");
    if (response->status != 200)
        return error("warm-up upstream HTTP " + std::to_string(response->status));
    auto value = json::parse(body);
    if (!value.ok() || !value->is_object() || value->find("error"))
        return error("warm-up upstream returned an invalid response");
    return value;
}

Result<json::Value> read_messages(const LlamaServerWarmupOptions &options, const CancellationToken &cancel) {
    if (options.max_prefix_chars == 0 || options.max_prefix_chars > 16777216)
        return error("warm-up max_prefix_chars must be in [1, 16777216]");
    if (options.messages_file.find('\0') != std::string::npos)
        return error("warm-up messages_file is not a valid path");
    std::error_code ec;
    const std::filesystem::path path(options.messages_file);
    if (!std::filesystem::is_regular_file(path, ec))
        return error("warm-up messages_file must be a readable regular file");
    std::ifstream file(path, std::ios::binary);
    if (!file)
        return error("warm-up messages_file could not be opened");
    std::string text;
    std::array<char, 4096> buffer{};
    while (file) {
        if (cancel.cancelled())
            return cancelled();
        file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const auto count = static_cast<std::size_t>(file.gcount());
        if (count > options.max_prefix_chars - text.size())
            return error("warm-up prefix exceeds max_prefix_chars");
        text.append(buffer.data(), count);
    }
    if (!file.eof())
        return error("warm-up messages_file read failed");
    auto messages = json::parse(text);
    if (!messages.ok() || !messages->is_array() || messages->as_array().empty())
        return error("warm-up messages_file must contain a nonempty JSON array of chat messages");
    for (const auto &message : messages->as_array()) {
        const auto *role = message.find("role");
        const auto *content = message.find("content");
        if (!role || !role->is_string() || !is_known_chat_role(role->as_string()) || !content ||
            !content->is_string())
            return error("warm-up messages must have a known role and string content");
    }
    return messages;
}

Status parse_result(const json::Value &value, BackendWarmupSlotStatus &slot) {
    const auto *choices = value.find("choices");
    if (!choices || !choices->is_array() || choices->as_array().size() != 1 ||
        !choices->as_array().front().find("finish_reason") ||
        !choices->as_array().front().find("finish_reason")->is_string())
        return error("warm-up response has no completed choice");
    Timings timing;
    if (!parse_timings(value, timing).ok())
        return error("warm-up response has invalid timings");
    if (timing.prompt_present)
        slot.prompt_tokens = timing.prompt_tokens + timing.cached_tokens;
    if (timing.cache_present)
        slot.cache_n = timing.cached_tokens;
    if (const auto *usage = value.find("usage")) {
        if (!usage->is_object())
            return error("warm-up response has invalid usage");
        if (const auto *count = usage->find("prompt_tokens")) {
            if (!count->is_integer() || count->as_double() < 0)
                return error("warm-up response has invalid prompt_tokens");
            slot.prompt_tokens = count->as_uint();
        }
        if (!slot.cache_n) {
            if (const auto *details = usage->find("prompt_tokens_details")) {
                if (const auto *count = details->find("cached_tokens")) {
                    if (!count->is_integer() || count->as_double() < 0)
                        return error("warm-up response has invalid cached_tokens");
                    slot.cache_n = count->as_uint();
                }
            }
        }
    }
    if (slot.prompt_tokens && slot.cache_n && *slot.cache_n > *slot.prompt_tokens)
        return error("warm-up response has inconsistent cache counts");
    return {};
}
} // namespace

PrefixWarmup::PrefixWarmup(const LlamaServerBackendOptions &options)
    : options_(options.warmup), native_(options.native_completion), startup_timeout_(options.startup_timeout),
      poll_interval_(options.poll_interval) {
    request_.connect_timeout = options.connect_timeout;
    request_.total_timeout = options.request_timeout;
    request_.tls.ca_bundle_path = options.tls.ca_bundle_path;
    request_.tls.pinned_sha256 = options.tls.pinned_sha256;
    request_.tls.pinned_cert_path = options.tls.pinned_cert_path;
    request_.tls.insecure_skip_verify = options.tls.insecure_skip_verify;
    request_.tls.server_name = options.tls.server_name;
    request_.tls.handshake_timeout = options.tls.handshake_timeout;
    if (!options_.messages_file.empty()) {
        status_.emplace();
        status_->status = "pending";
    }
}

PrefixWarmup::~PrefixWarmup() { stop(); }

void PrefixWarmup::stop_locked() {
    if (cancel_)
        cancel_->cancel();
    if (worker_.joinable())
        worker_.join();
    std::lock_guard lock(mutex_);
    warmed_.clear(); // the old child's checkpoints are no longer usable
}

void PrefixWarmup::stop() noexcept {
    std::lock_guard control(control_mutex_);
    stop_locked();
}

void PrefixWarmup::start(const std::string &base_url, bool wait_for_health) noexcept {
    if (options_.messages_file.empty())
        return;
    try {
        std::lock_guard control(control_mutex_);
        stop_locked();
        {
            std::lock_guard lock(mutex_);
            const auto generation = status_->generation + 1;
            status_ = BackendWarmupStatus{generation, "warming", {}};
            warnings_.clear();
            if (generation > 1 && !options_.on_restart) {
                status_->status = "skipped";
                return;
            }
        }
        cancel_ = std::make_unique<CancellationSource>();
        worker_ = std::thread(&PrefixWarmup::run, this, base_url, wait_for_health, cancel_->token());
    } catch (...) {
        fail("warm-up worker could not be started");
    }
}

void PrefixWarmup::fail(const std::string &message) {
    std::lock_guard lock(mutex_);
    status_->status = "error";
    for (auto &slot : status_->slots) {
        if (slot.status == "pending" || slot.status == "warming") {
            slot.status = "error";
            slot.error = message;
        }
    }
    if (warnings_.size() < SlotAffinity::kMaxSlots + 1u)
        warnings_.push_back(BackendWarning{"warmup_failed", "warning", "warmup", message, {}, 1});
}

void PrefixWarmup::run(std::string base_url, bool wait_for_health, CancellationToken cancel) noexcept {
    try {
        const auto result = replay(base_url, wait_for_health, cancel);
        if (cancel.cancelled()) {
            std::lock_guard lock(mutex_);
            status_->status = "cancelled";
            for (auto &slot : status_->slots) {
                if (slot.status == "pending" || slot.status == "warming")
                    slot.status = "cancelled";
            }
        } else if (!result.ok()) {
            fail(result.message());
        }
    } catch (...) {
        fail("warm-up worker failed");
    }
}

Status PrefixWarmup::replay(const std::string &base_url, bool wait_for_health, const CancellationToken &cancel) {
    if (!native_)
        return error("warm-up requires native_completion=true (llama-server slot API)");
    auto messages = read_messages(options_, cancel);
    if (!messages.ok())
        return messages.status();
    auto url = net::parse_url(base_url);
    if (!url.ok())
        return error("warm-up upstream URL is invalid");
    auto request = request_;
    request.host = url->host;
    request.port = url->port;
    request.use_tls = url->scheme == "https";
    const auto target = [&](const char *path) { return (url->path == "/" ? "" : url->path) + path; };
    if (wait_for_health) {
        const auto deadline = Clock::now() + startup_timeout_;
        bool ready = false;
        while (!cancel.cancelled() && Clock::now() < deadline) {
            auto probe = request;
            probe.target = target("/health");
            probe.total_timeout = std::min(Milliseconds(250),
                std::chrono::duration_cast<Milliseconds>(deadline - Clock::now()));
            probe.connect_timeout = std::min(probe.connect_timeout, probe.total_timeout);
            std::string body;
            const auto response = net::http_request_buffered(probe, body, cancel, 4096);
            if (response.ok() && response->status == 200) {
                ready = true;
                break;
            }
            const auto next = std::min(deadline, Clock::now() + poll_interval_);
            while (!cancel.cancelled() && Clock::now() < next)
                std::this_thread::sleep_for(Milliseconds(5));
        }
        if (!ready)
            return cancel.cancelled() ? cancelled() : error("warm-up readiness timed out");
    }
    request.target = target("/props");
    auto props_value = fetch(request, cancel);
    if (!props_value.ok())
        return props_value.status();
    ServerProps props;
    if (!parse_props(props_value.value(), props).ok() || props.total_slots == 0 ||
        props.total_slots > SlotAffinity::kMaxSlots)
        return error("warm-up requires a valid /props total_slots in [1, 1024]");
    std::vector<std::uint32_t> selected;
    if (options_.all_slots) {
        for (std::uint32_t i = 0; i < props.total_slots; ++i)
            selected.push_back(i);
    } else {
        selected = options_.slots;
        std::sort(selected.begin(), selected.end());
        if (selected.size() > SlotAffinity::kMaxSlots ||
            std::adjacent_find(selected.begin(), selected.end()) != selected.end() ||
            (!selected.empty() && selected.back() >= props.total_slots))
            return error("warm-up slots must be unique ids within /props total_slots");
    }
    if (cancel.cancelled())
        return cancelled();
    {
        std::lock_guard lock(mutex_);
        for (const auto id : selected) {
            BackendWarmupSlotStatus slot;
            slot.id_slot = id;
            slot.status = "pending";
            status_->slots.push_back(std::move(slot));
        }
    }
    request.target = target("/v1/models");
    auto models = fetch(request, cancel);
    if (!models.ok())
        return models.status();
    const auto *data = models->find("data");
    const auto *id = data && data->is_array() && !data->as_array().empty()
                         ? data->as_array().front().find("id") : nullptr;
    if (!id || !id->is_string() || id->as_string().empty())
        return error("warm-up upstream reported no model");
    json::Object body{{"model", id->as_string()}, {"messages", std::move(messages.value())},
                      {"cache_prompt", true}, {"max_tokens", 1}, {"stream", false},
                      {"chat_template_kwargs", options_.chat_template_kwargs}};
    request.method = "POST";
    request.target = target("/v1/chat/completions");
    bool failed = false;
    for (std::size_t i = 0; i < selected.size(); ++i) {
        if (cancel.cancelled())
            return cancelled();
        BackendWarmupSlotStatus slot;
        slot.id_slot = selected[i];
        slot.status = "warming";
        {
            std::lock_guard lock(mutex_);
            status_->slots[i] = slot;
        }
        body.set("id_slot", slot.id_slot);
        request.body = json::Value(body).dump();
        const auto began = Clock::now();
        const auto response = fetch(request, cancel);
        const auto result = response.ok() ? parse_result(response.value(), slot) : response.status();
        slot.milliseconds = std::chrono::duration<double, std::milli>(Clock::now() - began).count();
        slot.status = cancel.cancelled() ? "cancelled" : (result.ok() ? "complete" : "error");
        if (!result.ok() && !cancel.cancelled()) {
            slot.error = result.message();
            failed = true;
        }
        {
            std::lock_guard lock(mutex_);
            status_->slots[i] = slot;
            if (slot.status == "complete")
                warmed_.push_back(slot.id_slot);
            if (slot.status == "error")
                warnings_.push_back(BackendWarning{"warmup_failed", "warning", "warmup", slot.error,
                                                   {{"id_slot", std::to_string(slot.id_slot)}}, 1});
        }
    }
    std::lock_guard lock(mutex_);
    status_->status = failed ? "error" : "complete";
    return {};
}

std::optional<BackendWarmupStatus> PrefixWarmup::status() const {
    std::lock_guard lock(mutex_);
    return status_;
}
std::vector<BackendWarning> PrefixWarmup::warnings() const {
    std::lock_guard lock(mutex_);
    return warnings_;
}
std::vector<std::uint32_t> PrefixWarmup::warmed_slots() const {
    std::lock_guard lock(mutex_);
    return warmed_;
}

} // namespace sonder::inference::llamaserver
