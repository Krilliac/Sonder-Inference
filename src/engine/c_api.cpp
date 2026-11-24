// C ABI implementation (include/sonder_inference.h).
#include "sonder_inference.h"

#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "sonder/inference/backends.hpp"
#include "sonder/inference/device.hpp"
#include "sonder/inference/engine.hpp"
#include "sonder/inference/telemetry.hpp"
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif

using namespace sonder::inference;

namespace sonder::inference::detail {

// Thread-safe C ABI telemetry callback sink. Keeps a recursive mutex around
// callback invocation and updates so replacing/clearing or destroying the
// engine blocks until any in-flight callback finishes (preventing dangling
// callback invocations) without deadlocking on self-clearing from the writer
// thread. Exceptions from callbacks are contained so the sink is never retired
// and exceptions never cross the C ABI.
class CabiTelemetryCallbackSink final : public TelemetrySink {
public:
    CabiTelemetryCallbackSink(sonder_telemetry_callback callback, void* user_data) noexcept
        : callback_(callback), user_data_(user_data) {}

    void write(std::string_view json_line) override {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        if (!callback_) {
            return;
        }
        std::thread::id prev{};
        const std::thread::id self = std::this_thread::get_id();
        {
            std::lock_guard<std::mutex> tid_lock(thread_id_mutex_);
            prev = active_thread_;
            active_thread_ = self;
        }
        struct ThreadReset {
            std::mutex& mu;
            std::thread::id& slot;
            std::thread::id previous;
            ~ThreadReset() noexcept {
                try {
                    std::lock_guard<std::mutex> tid_lock(mu);
                    slot = previous;
                } catch (...) {
                }
            }
        } reset{thread_id_mutex_, active_thread_, prev};
        try {
            const char* data = json_line.empty() ? "" : json_line.data();
            callback_(user_data_, data, json_line.size());
        } catch (...) {
            // Contain exceptions from C/C++ callbacks; never escape into TelemetryBus.
        }
    }

    void set(sonder_telemetry_callback callback, void* user_data) {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        callback_ = callback;
        user_data_ = callback ? user_data : nullptr;
    }

    void clear() { set(nullptr, nullptr); }

    [[nodiscard]] bool has_callback() const {
        std::lock_guard<std::recursive_mutex> lock(mutex_);
        return callback_ != nullptr;
    }

    [[nodiscard]] bool is_in_callback_on_this_thread() const {
        std::lock_guard<std::mutex> tid_lock(thread_id_mutex_);
        return active_thread_ == std::this_thread::get_id();
    }

private:
    mutable std::recursive_mutex mutex_;
    sonder_telemetry_callback callback_ = nullptr;
    void* user_data_ = nullptr;
    mutable std::mutex thread_id_mutex_;
    std::thread::id active_thread_{};
};

}  // namespace sonder::inference::detail

// The engine is shared: every session handle holds a reference, because a
// Session keeps a plain Engine& and uses the engine's telemetry bus and
// request runtime until it is destroyed. sonder_engine_destroy() therefore
// flushes and clears any C telemetry callback attached to this handle, then
// drops the caller's reference; the engine stops when the last session handle
// is destroyed (or at once when there are none).
struct sonder_engine {
    std::shared_ptr<Engine> engine;
    std::mutex telemetry_sink_mutex;
    std::shared_ptr<detail::CabiTelemetryCallbackSink> telemetry_sink;
};
struct sonder_model {
    std::shared_ptr<Model> model;
};
struct sonder_session {
    std::shared_ptr<Engine> engine;  // declared first: destroyed after `session`
    std::shared_ptr<Session> session;
};

namespace {

thread_local std::string g_last_error;

sonder_status to_c(ErrorCode code) { return static_cast<sonder_status>(static_cast<int>(code)); }

sonder_status fail(const Status& st) {
    g_last_error = st.to_string();
    return to_c(st.code());
}

sonder_status fail(sonder_status code, const char* message) {
    g_last_error = message;
    return code;
}

sonder_status ok() {
    g_last_error.clear();
    return SONDER_OK;
}

// Original (first) layout of sonder_sampling_config. Callers compiled against
// it pass struct_size == sizeof(sonder_sampling_config_v1); the appended
// fields are only read when struct_size covers the whole current struct.
struct sonder_sampling_config_v1 {
    uint32_t struct_size;
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    float repeat_penalty;
    int32_t has_seed;
    uint64_t seed;
    int32_t max_tokens;
};
static_assert(offsetof(sonder_sampling_config, seed) == offsetof(sonder_sampling_config_v1, seed));
static_assert(offsetof(sonder_sampling_config, max_tokens) == offsetof(sonder_sampling_config_v1, max_tokens));
static_assert(sizeof(sonder_sampling_config) > sizeof(sonder_sampling_config_v1));

constexpr std::size_t kSamplingConfigMinSize = sizeof(sonder_sampling_config_v1);

bool has_extended_sampling_fields(const sonder_sampling_config& c) {
    return c.struct_size >= sizeof(sonder_sampling_config);
}

// Returns an error message, or nullptr when `c` can be converted.
const char* check_c_sampling(const sonder_sampling_config& c) {
    if (c.struct_size < kSamplingConfigMinSize) {
        return "sonder_sampling_config.struct_size too small";
    }
    if (has_extended_sampling_fields(c) && c.logit_bias_count > 0 && c.logit_bias == nullptr) {
        return "sonder_sampling_config.logit_bias is null but logit_bias_count > 0";
    }
    if (has_extended_sampling_fields(c) && c.logit_bias_count > SamplingConfig::kMaxLogitBiasEntries) {
        return "sonder_sampling_config.logit_bias_count exceeds the maximum";
    }
    return nullptr;
}

SamplingConfig from_c(const sonder_sampling_config& c) {
    SamplingConfig s;
    s.temperature = c.temperature;
    s.top_p = c.top_p;
    s.top_k = c.top_k;
    s.min_p = c.min_p;
    s.repeat_penalty = c.repeat_penalty;
    if (c.has_seed) {
        s.seed = c.seed;
    }
    s.max_tokens = c.max_tokens;
    if (has_extended_sampling_fields(c)) {
        s.typical_p = c.typical_p;
        s.presence_penalty = c.presence_penalty;
        s.frequency_penalty = c.frequency_penalty;
        s.repeat_last_n = c.repeat_last_n;
        s.num_ctx = c.num_ctx;
        s.logit_bias.reserve(c.logit_bias_count);
        for (std::size_t i = 0; i < c.logit_bias_count; ++i) {
            s.logit_bias.push_back(TokenLogitBias{c.logit_bias[i].token, c.logit_bias[i].bias});
        }
    }
    return s;
}

// Mirror the existing HTTP correlation-header policy, without echoing input.
// Read at most the 128-byte identifier plus its required terminator.
bool valid_correlation_id(const char* value) {
    if (!value) {
        return true;
    }
    for (std::size_t i = 0; i <= SONDER_MAX_CORRELATION_ID_BYTES; ++i) {
        const auto ch = static_cast<unsigned char>(value[i]);
        if (ch == 0) {
            return i > 0;
        }
        if (!((ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') ||
              (ch >= '0' && ch <= '9') || ch == '.' || ch == '_' || ch == ':' || ch == '-')) {
            return false;
        }
    }
    return false;
}

// Original (v1) layout of sonder_request_metadata: struct_size + parent_request_id.
// Appended scheduling fields (workload_class, priority) are only read when
// struct_size >= sizeof(sonder_request_metadata).
struct sonder_request_metadata_v1 {
    uint32_t struct_size;
    const char* parent_request_id;
};
static_assert(std::is_standard_layout_v<sonder_request_metadata>);
static_assert(std::is_trivial_v<sonder_request_metadata>);
static_assert(std::is_trivially_copyable_v<sonder_request_metadata>);
static_assert(offsetof(sonder_request_metadata, struct_size) ==
              offsetof(sonder_request_metadata_v1, struct_size));
static_assert(offsetof(sonder_request_metadata, parent_request_id) ==
              offsetof(sonder_request_metadata_v1, parent_request_id));
static_assert(sizeof(sonder_request_metadata) > sizeof(sonder_request_metadata_v1));
static_assert(offsetof(sonder_request_metadata, workload_class) >= sizeof(sonder_request_metadata_v1));

static_assert(std::is_standard_layout_v<sonder_gpu_info>);
static_assert(std::is_trivial_v<sonder_gpu_info>);
static_assert(std::is_trivially_copyable_v<sonder_gpu_info>);

constexpr std::size_t kRequestMetadataMinSize = sizeof(sonder_request_metadata_v1);
constexpr std::size_t kRequestMetadataExtendedMinSize =
    offsetof(sonder_request_metadata, priority) + sizeof(int32_t);
static_assert(kRequestMetadataExtendedMinSize > kRequestMetadataMinSize);
static_assert(kRequestMetadataExtendedMinSize <= sizeof(sonder_request_metadata));

constexpr int32_t kMinRequestPriority = -16;
constexpr int32_t kMaxRequestPriority = 16;

bool has_extended_request_metadata(const sonder_request_metadata& m) noexcept {
    return m.struct_size >= kRequestMetadataExtendedMinSize;
}

RequestPriority map_scheduling_to_priority_class(int32_t workload_class, int32_t priority) noexcept {
    const int base_rank = (workload_class == static_cast<int32_t>(SONDER_WORKLOAD_DEFAULT))
                              ? static_cast<int>(WorkloadClass::implementation_worker)
                              : static_cast<int>(workload_class - 1);
    const int effective_rank = base_rank - static_cast<int>(priority);
    if (effective_rank <= static_cast<int>(WorkloadClass::owner_orchestrator)) {
        return RequestPriority::interactive;
    }
    if (effective_rank <= static_cast<int>(WorkloadClass::research_worker)) {
        return RequestPriority::subagent;
    }
    return RequestPriority::background;
}

template <std::size_t N>
void copy_utf8_c_string(char (&dst)[N], std::string_view src) noexcept {
    static_assert(N > 0);
    std::size_t n = src.size() < (N - 1) ? src.size() : (N - 1);
    if (src.size() > n && n > 0 && (static_cast<unsigned char>(src[n]) & 0xC0u) == 0x80u) {
        while (n > 0 && (static_cast<unsigned char>(src[n - 1]) & 0xC0u) == 0x80u) {
            --n;
        }
        if (n > 0 && (static_cast<unsigned char>(src[n - 1]) & 0x80u) != 0u) {
            --n;
        }
    }
    if (n > 0) {
        std::memcpy(dst, src.data(), n);
    }
    dst[n] = '\0';
}

const char* request_options_from_c(const sonder_request_metadata* metadata, RequestOptions& options) {
    if (!metadata) {
        return nullptr;
    }
    if (metadata->struct_size < kRequestMetadataMinSize) {
        return "sonder_request_metadata.struct_size too small";
    }
    if (!valid_correlation_id(metadata->parent_request_id)) {
        return "parent_request_id must match [A-Za-z0-9._:-]{1,128}";
    }
    if (has_extended_request_metadata(*metadata)) {
        if (metadata->workload_class < static_cast<int32_t>(SONDER_WORKLOAD_DEFAULT) ||
            metadata->workload_class > static_cast<int32_t>(SONDER_WORKLOAD_MAINTENANCE)) {
            return "workload_class must be SONDER_WORKLOAD_DEFAULT (0) or in [1, 7]";
        }
        if (metadata->priority < kMinRequestPriority || metadata->priority > kMaxRequestPriority) {
            return "priority must be in [-16, 16]";
        }
        if (metadata->workload_class != static_cast<int32_t>(SONDER_WORKLOAD_DEFAULT) ||
            metadata->priority != 0) {
            options.priority_class = map_scheduling_to_priority_class(metadata->workload_class, metadata->priority);
        }
    }
    if (metadata->parent_request_id) {
        options.parent_request_id = metadata->parent_request_id;
    }
    return nullptr;
}

TokenCallback c_callback(sonder_token_callback callback, void* user_data) {
    if (!callback) {
        return {};
    }
    return [callback, user_data](const TokenChunk& chunk) {
        return callback(user_data, chunk.text.data(), chunk.text.size()) == 0;
    };
}

sonder_status finish_request(const Result<GenerationResult>& res, sonder_generation_stats* out_stats) {
    if (!res.ok()) {
        if (out_stats) {
            out_stats->outcome = SONDER_OUTCOME_FAILED;
        }
        return fail(res.status());
    }
    if (out_stats) {
        const auto& r = res.value();
        out_stats->outcome = static_cast<sonder_outcome>(static_cast<int>(r.outcome));
        out_stats->prompt_tokens = r.stats.prompt_tokens;
        out_stats->completion_tokens = r.stats.completion_tokens;
        out_stats->chunks = r.stats.chunks;
        out_stats->ttft_ms = r.ttft_ms;
        out_stats->total_ms = r.total_ms;
    }
    return ok();
}

template <class F>
sonder_status guarded(F&& f) noexcept {
    try {
        return f();
    } catch (const std::exception& e) {
        g_last_error = std::string("internal: ") + e.what();
        return SONDER_ERROR_INTERNAL;
    } catch (...) {
        g_last_error = "internal: unknown exception";
        return SONDER_ERROR_INTERNAL;
    }
}

}  // namespace

extern "C" {

uint32_t sonder_abi_version(void) { return SONDER_ABI_VERSION; }
const char* sonder_version_string(void) { return version_string(); }
const char* sonder_status_string(sonder_status status) { return to_string(static_cast<ErrorCode>(status)); }
const char* sonder_last_error_message(void) { return g_last_error.c_str(); }

void sonder_engine_options_init(sonder_engine_options* options) {
    if (!options) {
        return;
    }
    *options = sonder_engine_options{};
    options->struct_size = sizeof(sonder_engine_options);
    options->telemetry_level = SONDER_TELEMETRY_STANDARD;
}

sonder_status sonder_engine_create(const sonder_engine_options* options, sonder_engine** out_engine) {
    return guarded([&] {
        if (!out_engine) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "out_engine is null");
        }
        *out_engine = nullptr;
        EngineOptions eo;
        if (options) {
            if (options->struct_size < sizeof(sonder_engine_options)) {
                return fail(SONDER_ERROR_INVALID_ARGUMENT, "sonder_engine_options.struct_size too small");
            }
            eo.telemetry.level = static_cast<TelemetryLevel>(options->telemetry_level);
            eo.telemetry.capture_text = options->capture_text != 0;
            if (options->telemetry_jsonl_path && options->telemetry_jsonl_path[0] != '\0') {
                Status st;
                auto sink = make_jsonl_file_sink(options->telemetry_jsonl_path, true, &st);
                if (!sink) {
                    return fail(st);
                }
                eo.telemetry_sinks.push_back(std::shared_ptr<TelemetrySink>(std::move(sink)));
            }
        }
        auto handle = std::make_unique<sonder_engine>();
        handle->engine = std::make_shared<Engine>(std::move(eo));
        *out_engine = handle.release();
        return ok();
    });
}

void sonder_engine_destroy(sonder_engine* engine) {
    if (!engine) {
        return;
    }
    std::shared_ptr<detail::CabiTelemetryCallbackSink> sink;
    try {
        std::lock_guard<std::mutex> lock(engine->telemetry_sink_mutex);
        sink = engine->telemetry_sink;
    } catch (...) {
        // Best-effort lock acquisition; never let exceptions escape destroy.
    }
    if (sink) {
        try {
            if (!sink->is_in_callback_on_this_thread() && sink->has_callback()) {
                engine->engine->telemetry().flush();
            }
        } catch (...) {
            // Best-effort flush; always proceed to clear the callback sink below.
        }
        try {
            sink->clear();
        } catch (...) {
            // Best-effort clear; never let exceptions escape destroy.
        }
    }
    delete engine;
}

size_t sonder_engine_device_count(const sonder_engine* engine) {
    return engine ? engine->engine->devices().size() : 0;
}

size_t sonder_gpu_device_count(void) {
    try {
        return enumerate_gpu_devices().size();
    } catch (...) {
        return 0;
    }
}

sonder_status sonder_gpu_device_info(size_t index, sonder_gpu_info* out_info) {
    return guarded([&] {
        if (!out_info) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "out_info is null");
        }
        if (out_info->struct_size < sizeof(sonder_gpu_info)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "sonder_gpu_info.struct_size too small");
        }
        const std::vector<GpuDeviceInfo> gpus = enumerate_gpu_devices();
        if (index >= gpus.size()) {
            return fail(SONDER_ERROR_NOT_FOUND, "gpu device index out of range");
        }
        const GpuDeviceInfo& gpu = gpus[index];
        // Zero all bytes including implicit padding (offsets 428..431) and copy
        // via memcpy so uninitialized stack or stale caller bytes never leak.
        sonder_gpu_info info;
        std::memset(&info, 0, sizeof(info));
        info.struct_size = sizeof(sonder_gpu_info);
        copy_utf8_c_string(info.id, gpu.id);
        copy_utf8_c_string(info.name, gpu.name);
        copy_utf8_c_string(info.vendor, to_string(gpu.vendor));
        copy_utf8_c_string(info.backend_hint, gpu.backend_hint);
        copy_utf8_c_string(info.source, gpu.source);
        info.vendor_id = gpu.vendor_id;
        info.device_id = gpu.device_id;
        info.has_dedicated_total = gpu.dedicated_total.is_known() ? 1 : 0;
        info.dedicated_total_bytes = gpu.dedicated_total.bytes_or(0);
        info.has_dedicated_free = gpu.dedicated_free.is_known() ? 1 : 0;
        info.dedicated_free_bytes = gpu.dedicated_free.bytes_or(0);
        info.has_shared_total = gpu.shared_total.is_known() ? 1 : 0;
        info.shared_total_bytes = gpu.shared_total.bytes_or(0);
        const auto effective_total = gpu_effective_vram_mib(gpu, false);
        info.has_effective_vram = effective_total.has_value() ? 1 : 0;
        info.effective_vram_mib = effective_total.value_or(0);
        std::optional<std::uint64_t> effective_free;
        if (gpu.dedicated_free.is_known()) {
            effective_free = gpu_effective_vram_mib(gpu, true);
        }
        info.has_effective_free_vram = effective_free.has_value() ? 1 : 0;
        info.effective_free_vram_mib = effective_free.value_or(0);
        info.is_integrated_uma = gpu.is_integrated_uma ? 1 : 0;
        std::memcpy(out_info, &info, sizeof(info));
        return ok();
    });
}

sonder_status sonder_engine_set_telemetry_callback(sonder_engine* engine, sonder_telemetry_callback callback,
                                                   void* user_data) {
    return guarded([&] {
        if (!engine) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "engine is null");
        }
        std::shared_ptr<detail::CabiTelemetryCallbackSink> sink;
        {
            std::lock_guard<std::mutex> lock(engine->telemetry_sink_mutex);
            sink = engine->telemetry_sink;
            if (!sink) {
                if (!callback) {
                    return ok();
                }
                sink = std::make_shared<detail::CabiTelemetryCallbackSink>(callback, user_data);
                engine->engine->telemetry().add_sink(sink);
                engine->telemetry_sink = sink;
                return ok();
            }
        }
        if (!sink->is_in_callback_on_this_thread() && sink->has_callback()) {
            engine->engine->telemetry().flush();
        }
        sink->set(callback, user_data);
        return ok();
    });
}

sonder_status sonder_engine_register_mock_backend(sonder_engine* engine) {
    return guarded([&] {
        if (!engine) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "engine is null");
        }
        auto st = engine->engine->register_backend(make_mock_backend());
        return st.ok() ? ok() : fail(st);
    });
}

sonder_status sonder_engine_register_ollama_backend(sonder_engine* engine, const char* base_url) {
    return guarded([&] {
        if (!engine) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "engine is null");
        }
#if defined(SONDER_HAS_OLLAMA_BACKEND)
        OllamaBackendOptions o;
        if (base_url && base_url[0] != '\0') {
            o.base_url = base_url;
        }
        auto st = engine->engine->register_backend(make_ollama_backend(o));
        return st.ok() ? ok() : fail(st);
#else
        (void)base_url;
        return fail(SONDER_ERROR_UNSUPPORTED, "built without the ollama module (src/backends/ollama)");
#endif
    });
}

sonder_status sonder_model_load(sonder_engine* engine, const char* backend_name, const char* model_name,
                                sonder_model** out_model) {
    return guarded([&] {
        if (!engine || !backend_name || !model_name || !out_model) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "null argument");
        }
        *out_model = nullptr;
        ModelLoadOptions lo;
        lo.model = model_name;
        auto loaded = engine->engine->load_model(backend_name, lo);
        if (!loaded.ok()) {
            return fail(loaded.status());
        }
        *out_model = new sonder_model{std::move(loaded).value()};
        return ok();
    });
}

void sonder_model_release(sonder_model* model) { delete model; }

// Writes the defaults into the first `size` bytes the caller owns. The whole
// current struct only when `size` covers it; otherwise the original layout,
// with struct_size saying so (the appended fields then read as defaults).
// Smaller than the original layout: nothing is written.
void sonder_sampling_config_init_sized(sonder_sampling_config* config, size_t size) {
    if (!config || size < kSamplingConfigMinSize) {
        return;
    }
    const SamplingConfig d;
    sonder_sampling_config full{};
    full.struct_size = sizeof(sonder_sampling_config);
    full.temperature = d.temperature;
    full.top_p = d.top_p;
    full.top_k = d.top_k;
    full.min_p = d.min_p;
    full.repeat_penalty = d.repeat_penalty;
    full.has_seed = 0;
    full.seed = 0;
    full.max_tokens = d.max_tokens;
    full.typical_p = d.typical_p;
    full.presence_penalty = d.presence_penalty;
    full.frequency_penalty = d.frequency_penalty;
    full.repeat_last_n = d.repeat_last_n;
    full.num_ctx = d.num_ctx;
    full.logit_bias = nullptr;
    full.logit_bias_count = 0;
    if (size >= sizeof(sonder_sampling_config)) {
        *config = full;
        return;
    }
    full.struct_size = static_cast<uint32_t>(kSamplingConfigMinSize);
    std::memcpy(config, &full, kSamplingConfigMinSize);
}

// The exported symbol that binaries built against the original header call
// on their original-layout allocation: it must never write past that layout.
// Current callers reach sonder_sampling_config_init_sized() through the
// header macro instead. (Parenthesised name: the header defines a macro.)
void(sonder_sampling_config_init)(sonder_sampling_config* config) {
    sonder_sampling_config_init_sized(config, kSamplingConfigMinSize);
}

sonder_status sonder_sampling_config_validate(const sonder_sampling_config* config) {
    return guarded([&] {
        if (!config) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "config is null");
        }
        if (const char* err = check_c_sampling(*config)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, err);
        }
        auto st = validate(from_c(*config));
        return st.ok() ? ok() : fail(st);
    });
}

sonder_status sonder_session_create(sonder_engine* engine, sonder_model* model, const sonder_sampling_config* sampling,
                                    sonder_session** out_session) {
    return sonder_session_create_with_metadata(engine, model, sampling, nullptr, out_session);
}

sonder_status sonder_session_create_with_metadata(sonder_engine* engine, sonder_model* model,
                                                   const sonder_sampling_config* sampling,
                                                   const sonder_session_metadata* metadata,
                                                   sonder_session** out_session) {
    return guarded([&] {
        if (!engine || !model || !out_session) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "null argument");
        }
        *out_session = nullptr;
        SessionOptions so;
        if (metadata) {
            constexpr std::size_t min_size = offsetof(sonder_session_metadata, task_id) + sizeof(const char*);
            if (metadata->struct_size < min_size) {
                return fail(SONDER_ERROR_INVALID_ARGUMENT, "sonder_session_metadata.struct_size too small");
            }
            if (!valid_correlation_id(metadata->session_id) || !valid_correlation_id(metadata->run_id) ||
                !valid_correlation_id(metadata->agent_id) || !valid_correlation_id(metadata->task_id)) {
                return fail(SONDER_ERROR_INVALID_ARGUMENT,
                            "session metadata identifiers must match [A-Za-z0-9._:-]{1,128}");
            }
            if (metadata->session_id) {
                so.session_id = metadata->session_id;
            }
            if (metadata->run_id) {
                so.run_id = metadata->run_id;
            }
            if (metadata->agent_id) {
                so.agent_id = metadata->agent_id;
            }
            if (metadata->task_id) {
                so.task_id = metadata->task_id;
            }
        }
        if (sampling) {
            if (const char* err = check_c_sampling(*sampling)) {
                return fail(SONDER_ERROR_INVALID_ARGUMENT, err);
            }
            so.sampling = from_c(*sampling);
        }
        auto created = engine->engine->create_session(model->model, std::move(so));
        if (!created.ok()) {
            return fail(created.status());
        }
        *out_session = new sonder_session{engine->engine, std::move(created).value()};
        return ok();
    });
}

void sonder_session_destroy(sonder_session* session) { delete session; }

sonder_status sonder_session_generate(sonder_session* session, const char* prompt, sonder_token_callback callback,
                                      void* user_data, sonder_generation_stats* out_stats) {
    return sonder_session_generate_with_metadata(session, prompt, nullptr, callback, user_data, out_stats);
}

sonder_status sonder_session_generate_with_metadata(sonder_session* session, const char* prompt,
                                                    const sonder_request_metadata* metadata,
                                                    sonder_token_callback callback, void* user_data,
                                                    sonder_generation_stats* out_stats) {
    return guarded([&] {
        if (!session || !prompt) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "null argument");
        }
        if (out_stats && out_stats->struct_size < sizeof(sonder_generation_stats)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "sonder_generation_stats.struct_size too small");
        }
        RequestOptions options;
        if (const char* err = request_options_from_c(metadata, options)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, err);
        }
        return finish_request(session->session->generate(prompt, c_callback(callback, user_data), std::nullopt, options),
                              out_stats);
    });
}

sonder_status sonder_session_chat(sonder_session* session, const sonder_chat_message* const* messages,
                                  size_t message_count, sonder_token_callback callback, void* user_data,
                                  sonder_generation_stats* out_stats) {
    return sonder_session_chat_with_metadata(session, messages, message_count, nullptr, callback, user_data, out_stats);
}

sonder_status sonder_session_chat_with_metadata(sonder_session* session, const sonder_chat_message* const* messages,
                                               size_t message_count, const sonder_request_metadata* metadata,
                                               sonder_token_callback callback, void* user_data,
                                               sonder_generation_stats* out_stats) {
    return guarded([&] {
        if (!session || !messages || message_count == 0 || message_count > SONDER_MAX_CHAT_MESSAGES) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "chat requires a session and 1 to 1024 messages");
        }
        if (out_stats && out_stats->struct_size < sizeof(sonder_generation_stats)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "sonder_generation_stats.struct_size too small");
        }
        RequestOptions options;
        if (const char* err = request_options_from_c(metadata, options)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, err);
        }
        // This prefix remains fixed even if future headers append fields.
        constexpr std::size_t min_size = offsetof(sonder_chat_message, content) + sizeof(const char*);
        std::vector<ChatMessage> copied;
        copied.reserve(message_count);
        for (std::size_t i = 0; i < message_count; ++i) {
            const auto* message = messages[i];
            if (!message || message->struct_size < min_size) {
                return fail(SONDER_ERROR_INVALID_ARGUMENT, "null chat message or struct_size too small");
            }
            if (!message->role || !message->content) {
                return fail(SONDER_ERROR_INVALID_ARGUMENT, "chat role and content must be non-null");
            }
            copied.push_back(ChatMessage{message->role, message->content});
        }
        return finish_request(session->session->chat(copied, c_callback(callback, user_data), std::nullopt, options),
                              out_stats);
    });
}

sonder_status sonder_session_cancel(sonder_session* session) {
    if (!session) {
        return fail(SONDER_ERROR_INVALID_ARGUMENT, "session is null");
    }
    session->session->cancel();
    return ok();
}

}  // extern "C"
