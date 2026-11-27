#include "sonder/inference/engine.hpp"

#include <algorithm>
#include <chrono>
#include <limits>

#include "engine/request_runtime.hpp"
#include "sonder/inference/backends.hpp"

// Defined in the generated build_commit.cpp (cmake/SonderBuildInfo.cmake).
const char* sonder_build_commit_string();

namespace sonder::inference {

namespace {

std::uint64_t saturating_add(std::uint64_t a, std::uint64_t b) noexcept {
    return a > std::numeric_limits<std::uint64_t>::max() - b
               ? std::numeric_limits<std::uint64_t>::max()
               : a + b;
}

}  // namespace

#ifndef SONDER_INFERENCE_VERSION
#define SONDER_INFERENCE_VERSION "0.0.0"
#endif

const char* version_string() noexcept { return SONDER_INFERENCE_VERSION; }
const char* build_commit() noexcept { return ::sonder_build_commit_string(); }

const char* to_string(SchedulerMode mode) noexcept {
    switch (mode) {
        case SchedulerMode::automatic: return "automatic";
        case SchedulerMode::gate: return "gate";
        case SchedulerMode::account: return "account";
    }
    return "unknown";
}

json::Object gpu_memory_sample_attributes(const GpuDeviceInfo& gpu) {
    json::Object attrs{{"kind", "gpu"},
                       {"name", gpu.name},
                       {"vendor", to_string(gpu.vendor)},
                       {"vendor_id", gpu.vendor_id},
                       {"device_id", gpu.device_id},
                       {"is_integrated_uma", gpu.is_integrated_uma},
                       {"backend_hint", gpu.backend_hint},
                       {"source", gpu.source}};
    if (!gpu.id.empty()) {
        attrs.set("gpu_id", gpu.id);
    }
    if (gpu.dedicated_total.is_known()) {
        attrs.set("dedicated_total_bytes", gpu.dedicated_total.bytes_or(0));
    }
    if (gpu.dedicated_free.is_known()) {
        attrs.set("dedicated_free_bytes", gpu.dedicated_free.bytes_or(0));
    }
    if (gpu.shared_total.is_known()) {
        attrs.set("shared_total_bytes", gpu.shared_total.bytes_or(0));
    }

    if (!gpu.is_integrated_uma) {
        if (gpu.dedicated_total.is_known()) {
            attrs.set("total_bytes", gpu.dedicated_total.bytes_or(0));
        }
        if (gpu.dedicated_free.is_known()) {
            const std::uint64_t free_bytes = gpu.dedicated_free.bytes_or(0);
            attrs.set("available_bytes", free_bytes);
            if (gpu.dedicated_total.is_known()) {
                const std::uint64_t total_bytes = gpu.dedicated_total.bytes_or(0);
                if (total_bytes >= free_bytes) {
                    attrs.set("used_bytes", total_bytes - free_bytes);
                }
            }
        }
    } else {
        // Integrated UMA GPUs (e.g. AMD Radeon 780M APU) pool dedicated BIOS
        // carveout and shared/GTT system memory together (matching
        // gpu_effective_vram_mib()). Emit total/available/used bytes against
        // that unified pool (preserving the total_pool cap on free bytes and
        // omitting used_bytes on anomalous free > total readings) so
        // Observatory never drops UMA GPU samples.
        const bool has_pool = gpu.dedicated_total.is_known() || gpu.shared_total.is_known();
        const std::uint64_t total_pool =
            has_pool ? saturating_add(gpu.dedicated_total.bytes_or(0), gpu.shared_total.bytes_or(0)) : 0;
        if (has_pool && total_pool > 0) {
            attrs.set("total_bytes", total_pool);
        }
        if (gpu.dedicated_free.is_known()) {
            const std::uint64_t raw_free = gpu.dedicated_free.bytes_or(0);
            const std::uint64_t capped_free =
                (has_pool && total_pool > 0) ? std::min(raw_free, total_pool) : raw_free;
            attrs.set("available_bytes", capped_free);
            if (has_pool && total_pool > 0 && raw_free <= total_pool) {
                attrs.set("used_bytes", total_pool - raw_free);
            }
        }
    }

    if (const auto effective_mib = gpu_effective_vram_mib(gpu, false)) {
        attrs.set("effective_vram_mib", *effective_mib);
    }
    if (gpu.dedicated_free.is_known()) {
        if (const auto effective_free_mib = gpu_effective_vram_mib(gpu, true)) {
            attrs.set("effective_free_vram_mib", *effective_free_mib);
        }
    }
    return attrs;
}

Engine::Engine(EngineOptions options)
    : options_(std::move(options)), engine_id_(make_id("engine")),
      telemetry_(std::make_unique<TelemetryBus>(options_.telemetry)), devices_(enumerate_devices()),
      checkpoint_store_(options_.checkpoints) {
    for (auto& sink : options_.telemetry_sinks) {
        telemetry_->add_sink(sink);
    }
    options_.telemetry_sinks.clear();
    std::vector<GpuDeviceInfo> initial_gpus;
    if (options_.sample_gpu_devices) {
        initial_gpus = enumerate_gpu_devices();
        std::lock_guard<std::mutex> lock(gpu_devices_mutex_);
        gpu_devices_ = initial_gpus;
    }
    auto ctx = engine_context();
    json::Object started{{"version", version_string()},
                         {"commit", build_commit()},
                         {"platform", host_platform()},
                         {"device_count", devices_.size()},
                         {"gpu_device_count", initial_gpus.size()},
                         {"text_capture", options_.telemetry.capture_text ? "on" : "off"}};
    if (options_.server) {
        started.set("server", json::Object{{"host", options_.server->host},
                                           {"port", options_.server->port},
                                           {"api_version", options_.server->api_version}});
    }
    telemetry_->emit("engine.started", ctx, std::move(started), TelemetryLevel::metrics);
    runtime_ = detail::make_request_runtime(options_.scheduling, *telemetry_, ctx);
    checkpoint_store_.set_evict_callback([this](const ExecutionCheckpointRecord& rec) {
        if (runtime_ && rec.logical_kv_sequence_id != 0) {
            runtime_->release_logical_sequence(rec.logical_kv_sequence_id);
        }
    });
    if (runtime_) {
        const auto& so = options_.scheduling;
        telemetry_->emit("scheduler.configured", ctx,
                         json::Object{{"mode", to_string(so.mode)},
                                      {"kv_block_size_tokens", so.kv_block_size_tokens},
                                      {"kv_num_blocks", so.kv_num_blocks},
                                      {"prefix_caching", so.prefix_caching},
                                      {"max_running_sequences", so.max_running_sequences},
                                      {"max_step_sequences", so.max_step_sequences},
                                      {"max_step_tokens", so.max_step_tokens},
                                      {"prefill_chunk_tokens", so.prefill_chunk_tokens},
                                      {"admission_watermark_blocks", so.admission_watermark_blocks},
                                      {"max_requeue_count", so.max_requeue_count},
                                      {"step_stall_timeout_ms", so.step_stall_timeout_ms}},
                         TelemetryLevel::metrics);
    }
    if (options_.sample_devices_on_start) {
        sample_devices(devices_);
        if (options_.sample_gpu_devices) {
            sample_gpu_devices(initial_gpus);
        }
    }
    if (options_.device_sample_interval.count() > 0) {
        sampler_ = std::thread([this] { device_sampler_loop(); });
    }
}

std::vector<GpuDeviceInfo> Engine::gpu_devices() const {
    std::lock_guard<std::mutex> lock(gpu_devices_mutex_);
    return gpu_devices_;
}

void Engine::sample_devices(const std::vector<DeviceInfo>& devices) {
    for (std::size_t i = 0; i < devices.size(); ++i) {
        const auto& d = devices[i];
        auto dctx = engine_context();
        const std::string kind_str = to_string(d.kind);
        dctx.device_id = d.id.empty() ? (kind_str + ":" + std::to_string(i)) : d.id;
        json::Object attrs{{"kind", kind_str},
                           {"name", d.name},
                           {"logical_cores", d.logical_cores},
                           {"total_bytes", d.total_memory_bytes},
                           {"available_bytes", d.available_memory_bytes}};
        if (d.total_memory_bytes >= d.available_memory_bytes && d.total_memory_bytes > 0) {
            attrs.set("used_bytes", d.total_memory_bytes - d.available_memory_bytes);
        }
        telemetry_->emit("device.memory.sample", dctx, std::move(attrs), TelemetryLevel::metrics);
    }
}

void Engine::sample_gpu_devices(const std::vector<GpuDeviceInfo>& gpus) {
    for (std::size_t i = 0; i < gpus.size(); ++i) {
        const auto& gpu = gpus[i];
        auto dctx = engine_context();
        const std::string resolved_id = gpu.id.empty() ? ("gpu:" + std::to_string(i)) : gpu.id;
        dctx.device_id = resolved_id;
        json::Object attrs = gpu_memory_sample_attributes(gpu);
        if (attrs.find("gpu_id") == nullptr) {
            attrs.set("gpu_id", resolved_id);
        }
        telemetry_->emit("device.memory.sample", dctx, std::move(attrs), TelemetryLevel::metrics);
    }
}

void Engine::sample_backend_runtime() {
    std::vector<std::shared_ptr<Backend>> backends;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& kv : backends_) {
            backends.push_back(kv.second);
        }
    }
    for (const auto& backend : backends) {
        const auto status = backend->runtime_status();  // cheap snapshot
        if (!status) {
            continue;
        }
        const std::string name = backend->name();
        if (status->take_events) {
            for (auto& [type, attrs] : status->take_events()) {
                attrs.set("backend", name);
                telemetry_->emit(std::move(type), engine_context(), std::move(attrs), TelemetryLevel::metrics);
            }
        }
        if (status->warmup) {
            const auto& warmup = *status->warmup;
            auto& [generation, slots] = runtime_warmups_seen_[name];
            if (generation != warmup.generation) {
                generation = warmup.generation;
                slots.clear();
            }
            for (const auto& slot : warmup.slots) {
                if (slot.status != "complete" && slot.status != "error" && slot.status != "cancelled") continue;
                if (std::find(slots.begin(), slots.end(), slot.id_slot) != slots.end() || slots.size() >= 1024) continue;
                slots.push_back(slot.id_slot);
                auto attrs = to_json(slot);
                attrs.set("backend", name);
                attrs.set("generation", generation);
                telemetry_->emit("backend.warmup", engine_context(), std::move(attrs), TelemetryLevel::metrics);
            }
        }
        const auto& gpu = status->gpu_memory;
        auto& seen = runtime_samples_seen_[name];
        if (gpu.status == "ok" && gpu.samples != seen) {
            seen = gpu.samples;
            json::Object attrs = to_json(gpu);
            attrs.set("backend", name);
            attrs.set("fitted_ctx", json::Value(status->context.fitted_ctx));  // null when unknown
            attrs.set("fit_outcome", status->context.outcome);
            telemetry_->emit("backend.gpu_memory.sample", engine_context(), std::move(attrs), TelemetryLevel::metrics);
        }
        auto& emitted = runtime_warnings_seen_[name];
        for (const auto& w : status->warnings) {
            // These observations are queued per episode/poll, independently
            // of the slower engine sampler and health snapshot frequency.
            if (status->take_events && (w.code == "backend_stalled" || w.code == "diagnostics_blind" ||
                                       w.code == "metrics_unavailable" || w.code == "metrics_scrape_failed"))
                continue;
            std::string key = w.code + '\n' + w.source + '\n' + w.message;
            if (std::find(emitted.begin(), emitted.end(), key) != emitted.end()) {
                continue;
            }
            if (emitted.size() >= 256) {
                break;  // bounded: a backend cannot grow this without limit
            }
            emitted.push_back(std::move(key));
            json::Object attrs = to_json(w);
            attrs.set("backend", name);
            telemetry_->emit("backend.warning", engine_context(), std::move(attrs), TelemetryLevel::metrics);
        }
    }
}

void Engine::device_sampler_loop() {
    std::unique_lock<std::mutex> lock(sampler_mutex_);
    while (!sampler_cv_.wait_for(lock, options_.device_sample_interval, [this] { return sampler_stop_; })) {
        lock.unlock();
        sample_devices(enumerate_devices());  // re-reads available memory
        if (options_.sample_gpu_devices) {
            auto gpus = enumerate_gpu_devices();
            {
                std::lock_guard<std::mutex> gpu_lock(gpu_devices_mutex_);
                gpu_devices_ = gpus;
            }
            sample_gpu_devices(gpus);
        }
        sample_backend_runtime();
        lock.lock();
    }
}

Engine::~Engine() {
    {
        std::lock_guard<std::mutex> lock(sampler_mutex_);
        sampler_stop_ = true;
    }
    sampler_cv_.notify_all();
    if (sampler_.joinable()) {
        sampler_.join();
    }
    // Release checkpoint-owned logical KV sequences before tearing down runtime_.
    checkpoint_store_.clear();
    checkpoint_store_.set_evict_callback({});
    // Stop the scheduler thread first; it emits telemetry and references models.
    runtime_.reset();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        models_.clear();
        backends_.clear();
    }
    telemetry_->emit("engine.stopped", engine_context(), json::Object{}, TelemetryLevel::metrics);
    telemetry_->shutdown();
}

KvUsage Engine::kv_usage() const { return runtime_ ? runtime_->kv_usage() : KvUsage{}; }

TelemetryContext Engine::engine_context() const {
    TelemetryContext ctx;
    ctx.session_id = engine_id_;
    ctx.run_id = engine_id_;
    return ctx;
}

Status Engine::register_backend(std::shared_ptr<Backend> backend) {
    if (!backend) {
        return Status(ErrorCode::invalid_argument, "backend is null");
    }
    const std::string name = backend->name();
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (backends_.count(name) != 0) {
            return Status(ErrorCode::invalid_state, "backend already registered: " + name);
        }
        backends_[name] = backend;
    }
    auto ctx = engine_context();
    ctx.synthetic_work = is_mock_backend(name);
    telemetry_->emit("backend.registered", ctx,
                     json::Object{{"backend", name}, {"description", backend->description()},
                                  {"capabilities", [&] {
                                       json::Array caps;
                                       for (auto& c : backend->capabilities().names()) caps.emplace_back(c);
                                       return caps;
                                   }()}},
                     TelemetryLevel::metrics);
    return Status::success();
}

std::shared_ptr<Backend> Engine::find_backend(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = backends_.find(name);
    return it == backends_.end() ? nullptr : it->second;
}

std::vector<std::string> Engine::backend_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    for (const auto& kv : backends_) {
        names.push_back(kv.first);
    }
    return names;
}

Result<std::shared_ptr<Model>> Engine::load_model(const std::string& backend_name, const ModelLoadOptions& options) {
    auto backend = find_backend(backend_name);
    if (!backend) {
        return Status(ErrorCode::not_found, "unknown backend: " + backend_name);
    }
    if (options.model.empty()) {
        return Status(ErrorCode::invalid_argument, "model name is empty");
    }
    const std::string instance_id = make_id("model");
    auto ctx = engine_context();
    ctx.model_instance_id = instance_id;
    ctx.device_id = options.device_id;
    ctx.synthetic_work = is_mock_backend(backend_name);
    telemetry_->emit("model.load.started", ctx, json::Object{{"backend", backend_name}, {"model", options.model}},
                     TelemetryLevel::metrics);
    const auto t0 = std::chrono::steady_clock::now();
    auto loaded = backend->load_model(options);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!loaded.ok()) {
        const auto st = loaded.status();
        // Never emit raw upstream error text into shared Observatory telemetry (M-15).
        telemetry_->emit("model.load.failed", ctx,
                         json::Object{{"backend", backend_name}, {"model", options.model}, {"duration_ms", ms},
                                      {"error_code", to_string(st.code())}, {"error", to_string(st.code())}},
                         TelemetryLevel::metrics);
        return st;
    }
    auto model = std::make_shared<Model>(instance_id, backend_name, options.device_id, std::move(loaded).value());
    const auto& d = model->descriptor();
    telemetry_->emit("model.load.completed", ctx,
                     json::Object{{"backend", backend_name},
                                  {"model", d.name},
                                  {"format", d.format},
                                  {"family", d.family},
                                  {"parameter_size", d.parameter_size},
                                  {"quantization", d.quantization},
                                  {"size_bytes", d.size_bytes},
                                  {"resident", d.resident},
                                  {"duration_ms", ms}},
                     TelemetryLevel::metrics);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        models_[instance_id] = model;
    }
    return model;
}

Status Engine::unload_model(const std::string& model_instance_id) {
    std::shared_ptr<Model> model;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = models_.find(model_instance_id);
        if (it == models_.end()) {
            return Status(ErrorCode::not_found, "unknown model instance: " + model_instance_id);
        }
        model = it->second;
        models_.erase(it);
    }
    auto ctx = engine_context();
    ctx.model_instance_id = model_instance_id;
    ctx.device_id = model->device_id();
    ctx.synthetic_work = is_mock_backend(model->backend_name());
    // Sessions may still hold the handle; the backend model is released when
    // the last reference drops.
    telemetry_->emit("model.unload", ctx,
                     json::Object{{"backend", model->backend_name()}, {"model", model->descriptor().name},
                                  {"outstanding_references", static_cast<std::int64_t>(model.use_count() - 1)}},
                     TelemetryLevel::metrics);
    return Status::success();
}

std::vector<std::shared_ptr<Model>> Engine::loaded_models() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::shared_ptr<Model>> out;
    for (const auto& kv : models_) {
        out.push_back(kv.second);
    }
    return out;
}

Result<std::shared_ptr<Session>> Engine::create_session(const std::shared_ptr<Model>& model, SessionOptions options) {
    if (!model) {
        return Status(ErrorCode::invalid_argument, "model is null");
    }
    if (auto st = validate(options.sampling); !st.ok()) {
        return st;
    }
    if (options.session_id.empty()) {
        options.session_id = make_id("sess");
    }
    return std::make_shared<Session>(*this, model, std::move(options));
}

}  // namespace sonder::inference
