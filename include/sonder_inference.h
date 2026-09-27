/*
 * Sonder Inference - stable C ABI.
 *
 * This is the language-neutral boundary for bindings (Rust, Python, C#, ...).
 * Rules:
 *   - Opaque handles only; no C++ types cross this boundary.
 *   - Structs passed in carry `struct_size` so fields can be appended later.
 *   - sonder_status values are append-only and mirror sonder::inference::ErrorCode.
 *   - Strings are UTF-8, NUL-terminated on input; output text is (ptr, len).
 *   - Error detail for the calling thread is available from
 *     sonder_last_error_message() until the next call on that thread.
 *   - SONDER_ABI_VERSION increments on any incompatible change.
 */
#ifndef SONDER_INFERENCE_H
#define SONDER_INFERENCE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(SONDER_INFERENCE_SHARED)
#  if defined(SONDER_INFERENCE_BUILDING)
#    define SONDER_API __declspec(dllexport)
#  else
#    define SONDER_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(SONDER_INFERENCE_SHARED)
#  define SONDER_API __attribute__((visibility("default")))
#else
#  define SONDER_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SONDER_ABI_VERSION 1u

typedef enum sonder_status {
    SONDER_OK = 0,
    SONDER_ERROR_INVALID_ARGUMENT = 1,
    SONDER_ERROR_INVALID_STATE = 2,
    SONDER_ERROR_NOT_FOUND = 3,
    SONDER_ERROR_UNAVAILABLE = 4,
    SONDER_ERROR_CANCELLED = 5,
    SONDER_ERROR_TIMEOUT = 6,
    SONDER_ERROR_BACKEND = 7,
    SONDER_ERROR_PROTOCOL = 8,
    SONDER_ERROR_IO = 9,
    SONDER_ERROR_UNSUPPORTED = 10,
    SONDER_ERROR_INTERNAL = 11
} sonder_status;

typedef enum sonder_telemetry_level {
    SONDER_TELEMETRY_OFF = 0,
    SONDER_TELEMETRY_METRICS = 1,
    SONDER_TELEMETRY_STANDARD = 2,
    SONDER_TELEMETRY_DEEP = 3
} sonder_telemetry_level;

typedef enum sonder_outcome {
    SONDER_OUTCOME_NONE = 0,
    SONDER_OUTCOME_COMPLETED = 1,
    SONDER_OUTCOME_CANCELLED = 2,
    SONDER_OUTCOME_FAILED = 3
} sonder_outcome;

typedef struct sonder_engine sonder_engine;
typedef struct sonder_model sonder_model;
typedef struct sonder_session sonder_session;

typedef struct sonder_engine_options {
    uint32_t struct_size;               /* sizeof(sonder_engine_options) */
    sonder_telemetry_level telemetry_level;
    const char* telemetry_jsonl_path;   /* NULL: no telemetry sink */
    int32_t capture_text;               /* nonzero: include token text in events */
} sonder_engine_options;

/* One additive logit bias entry (see sonder_sampling_config.logit_bias). */
typedef struct sonder_logit_bias {
    int32_t token;  /* vocabulary index, >= 0 */
    float bias;     /* within [-100, 100], or -INFINITY to ban the token */
} sonder_logit_bias;

typedef struct sonder_sampling_config {
    uint32_t struct_size;  /* sizeof(sonder_sampling_config) */
    float temperature;
    float top_p;
    int32_t top_k;
    float min_p;
    float repeat_penalty;
    int32_t has_seed;
    uint64_t seed;
    int32_t max_tokens;
    /* ---- Appended fields (still ABI version 1). The library reads them only
     * when struct_size >= sizeof(sonder_sampling_config) of this header;
     * callers built against the original layout (struct_size ending after
     * max_tokens) keep working and get the defaults below. Initialise with
     * sonder_sampling_config_init(). ---- */
    float typical_p;          /* (0, 1]; 1 disables */
    float presence_penalty;   /* [-2, 2]; 0 disables */
    float frequency_penalty;  /* [-2, 2]; 0 disables */
    int32_t repeat_last_n;    /* penalty window: -1 whole context, 0 off, default 64 */
    int32_t num_ctx;          /* context window request; 0 = backend default */
    /* Borrowed for the duration of the call only (copied by the library).
     * May be NULL when logit_bias_count is 0. */
    const sonder_logit_bias* logit_bias;
    size_t logit_bias_count;
} sonder_sampling_config;

typedef struct sonder_generation_stats {
    uint32_t struct_size;  /* sizeof(sonder_generation_stats) */
    sonder_outcome outcome;
    uint64_t prompt_tokens;
    uint64_t completion_tokens;
    uint64_t chunks;
    double ttft_ms;        /* -1 when no output was produced */
    double total_ms;
} sonder_generation_stats;

/* Return nonzero to stop generation early. `text` is not NUL-terminated. */
typedef int (*sonder_token_callback)(void* user_data, const char* text, size_t length);

SONDER_API uint32_t sonder_abi_version(void);
SONDER_API const char* sonder_version_string(void);
SONDER_API const char* sonder_status_string(sonder_status status);
SONDER_API const char* sonder_last_error_message(void);

SONDER_API void sonder_engine_options_init(sonder_engine_options* options);
SONDER_API sonder_status sonder_engine_create(const sonder_engine_options* options, sonder_engine** out_engine);
/* Releases the caller's handle. The engine stops now, or when its last
 * session handle is destroyed if sessions remain. */
SONDER_API void sonder_engine_destroy(sonder_engine* engine);
SONDER_API size_t sonder_engine_device_count(const sonder_engine* engine);

/* Registers the deterministic MOCK backend (tests only) under name "mock". */
SONDER_API sonder_status sonder_engine_register_mock_backend(sonder_engine* engine);
/* Registers the Ollama adapter under name "ollama". base_url NULL = default.
 * Returns SONDER_ERROR_UNSUPPORTED when built without the ollama module. */
SONDER_API sonder_status sonder_engine_register_ollama_backend(sonder_engine* engine, const char* base_url);

SONDER_API sonder_status sonder_model_load(sonder_engine* engine, const char* backend_name, const char* model_name,
                                           sonder_model** out_model);
SONDER_API void sonder_model_release(sonder_model* model);

/* Fills `config` with the library defaults, writing at most `size` bytes
 * (pass sizeof(*config)). A size covering this header's struct sets every
 * field and struct_size = sizeof(sonder_sampling_config); a size covering only
 * the original layout (fields up to max_tokens) sets those and a matching
 * struct_size; anything smaller is left untouched. */
SONDER_API void sonder_sampling_config_init_sized(sonder_sampling_config* config, size_t size);
/* Original entry point, kept for binaries built against the original header:
 * it initialises only the original layout (it cannot know how much the caller
 * allocated). Source that includes this header calls the sized form through
 * the macro below, so it still gets every field. */
SONDER_API void sonder_sampling_config_init(sonder_sampling_config* config);
#define sonder_sampling_config_init(config) sonder_sampling_config_init_sized((config), sizeof(*(config)))
SONDER_API sonder_status sonder_sampling_config_validate(const sonder_sampling_config* config);

/* sampling may be NULL for defaults. The session keeps the model and the
 * engine alive: sonder_engine_destroy() may run before sonder_session_destroy()
 * (or while a generate call is in progress on another thread); the engine then
 * stops when its last session is destroyed. */
SONDER_API sonder_status sonder_session_create(sonder_engine* engine, sonder_model* model,
                                               const sonder_sampling_config* sampling,
                                               sonder_session** out_session);
SONDER_API void sonder_session_destroy(sonder_session* session);
/* Blocks until done. A cancelled request returns SONDER_OK with outcome CANCELLED. */
SONDER_API sonder_status sonder_session_generate(sonder_session* session, const char* prompt,
                                                 sonder_token_callback callback, void* user_data,
                                                 sonder_generation_stats* out_stats);
/* Thread-safe; cancels the in-flight request. */
SONDER_API sonder_status sonder_session_cancel(sonder_session* session);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SONDER_INFERENCE_H */
