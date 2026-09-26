#include "sonder/inference/engine.hpp"

#include <chrono>

#include "engine/request_runtime.hpp"

namespace sonder::inference {

#ifndef SONDER_INFERENCE_VERSION
#define SONDER_INFERENCE_VERSION "0.0.0"
#endif
#ifndef SONDER_INFERENCE_GIT_COMMIT
#define SONDER_INFERENCE_GIT_COMMIT "unknown"
#endif

const char* version_string() noexcept { return SONDER_INFERENCE_VERSION; }
const char* build_commit() noexcept { return SONDER_INFERENCE_GIT_COMMIT; }

Engine::Engine(EngineOptions options)
    : options_(std::move(options)), engine_id_(make_id("engine")),
      telemetry_(std::make_unique<TelemetryBus>(options_.telemetry)), devices_(enumerate_devices()) {
    for (auto& sink : options_.telemetry_sinks) {
        telemetry_->add_sink(sink);
    }
    options_.telemetry_sinks.clear();
    auto ctx = engine_context();
    telemetry_->emit("engine.started", ctx,
                     json::Object{{"version", version_string()},
                                  {"commit", build_commit()},
                                  {"platform", host_platform()},
                                  {"device_count", devices_.size()}},
                     TelemetryLevel::metrics);
    runtime_ = detail::make_request_runtime(options_.scheduling, *telemetry_, ctx);
    if (runtime_) {
        const auto& so = options_.scheduling;
        telemetry_->emit("scheduler.configured", ctx,
                         json::Object{{"kv_block_size_tokens", so.kv_block_size_tokens},
                                      {"kv_num_blocks", so.kv_num_blocks},
                                      {"prefix_caching", so.prefix_caching},
                                      {"max_running_sequences", so.max_running_sequences},
                                      {"max_step_sequences", so.max_step_sequences},
                                      {"max_step_tokens", so.max_step_tokens},
                                      {"prefill_chunk_tokens", so.prefill_chunk_tokens},
                                      {"admission_watermark_blocks", so.admission_watermark_blocks},
                                      {"max_requeue_count", so.max_requeue_count}},
                         TelemetryLevel::metrics);
    }
    if (options_.sample_devices_on_start) {
        for (const auto& d : devices_) {
            auto dctx = ctx;
            dctx.device_id = d.id;
            telemetry_->emit("device.memory.sample", dctx,
                             json::Object{{"kind", to_string(d.kind)},
                                          {"name", d.name},
                                          {"logical_cores", d.logical_cores},
                                          {"total_bytes", d.total_memory_bytes},
                                          {"available_bytes", d.available_memory_bytes}},
                             TelemetryLevel::metrics);
        }
    }
}

Engine::~Engine() {
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
    telemetry_->emit("backend.registered", engine_context(),
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
    telemetry_->emit("model.load.started", ctx, json::Object{{"backend", backend_name}, {"model", options.model}},
                     TelemetryLevel::metrics);
    const auto t0 = std::chrono::steady_clock::now();
    auto loaded = backend->load_model(options);
    const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (!loaded.ok()) {
        const auto st = loaded.status();
        telemetry_->emit("model.load.failed", ctx,
                         json::Object{{"backend", backend_name}, {"model", options.model}, {"duration_ms", ms},
                                      {"error_code", to_string(st.code())}, {"error", st.message()}},
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
