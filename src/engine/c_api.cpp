// C ABI implementation (include/sonder_inference.h).
#include "sonder_inference.h"

#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "sonder/inference/backends.hpp"
#include "sonder/inference/engine.hpp"
#if defined(SONDER_HAS_OLLAMA_BACKEND)
#include "sonder/inference/backends/ollama.hpp"
#endif

using namespace sonder::inference;

// The engine is shared: every session handle holds a reference, because a
// Session keeps a plain Engine& and uses the engine's telemetry bus and
// request runtime until it is destroyed. sonder_engine_destroy() therefore
// only drops the caller's reference; the engine stops when the last session
// handle is destroyed (or at once when there are none).
struct sonder_engine {
    std::shared_ptr<Engine> engine;
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

void sonder_engine_destroy(sonder_engine* engine) { delete engine; }

size_t sonder_engine_device_count(const sonder_engine* engine) {
    return engine ? engine->engine->devices().size() : 0;
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
    return guarded([&] {
        if (!session || !prompt) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "null argument");
        }
        if (out_stats && out_stats->struct_size < sizeof(sonder_generation_stats)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "sonder_generation_stats.struct_size too small");
        }
        return finish_request(session->session->generate(prompt, c_callback(callback, user_data)), out_stats);
    });
}

sonder_status sonder_session_chat(sonder_session* session, const sonder_chat_message* const* messages,
                                  size_t message_count, sonder_token_callback callback, void* user_data,
                                  sonder_generation_stats* out_stats) {
    return guarded([&] {
        if (!session || !messages || message_count == 0 || message_count > SONDER_MAX_CHAT_MESSAGES) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "chat requires a session and 1 to 1024 messages");
        }
        if (out_stats && out_stats->struct_size < sizeof(sonder_generation_stats)) {
            return fail(SONDER_ERROR_INVALID_ARGUMENT, "sonder_generation_stats.struct_size too small");
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
        return finish_request(session->session->chat(copied, c_callback(callback, user_data)), out_stats);
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
