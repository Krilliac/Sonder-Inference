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


/* Additive ABI-v1 session correlation metadata. NULL fields retain defaults.
 * Identifiers are caller-supplied metadata, not prompt text: each non-NULL
 * value must match [A-Za-z0-9._:-]{1,128}. Values are copied before returning.
 * Larger records are accepted; unknown tail fields are ignored. */
#define SONDER_MAX_CORRELATION_ID_BYTES 128u
typedef struct sonder_session_metadata {
    uint32_t struct_size;  /* sizeof(sonder_session_metadata) */
    const char* session_id; /* NULL: generated session id */
    const char* run_id;     /* NULL: engine correlation default */
    const char* agent_id;
    const char* task_id;
} sonder_session_metadata;

/* metadata may be NULL for the existing defaults. Sampling and engine/model
 * lifetime contracts are identical to sonder_session_create(). */
SONDER_API sonder_status sonder_session_create_with_metadata(
    sonder_engine* engine, sonder_model* model, const sonder_sampling_config* sampling,
    const sonder_session_metadata* metadata, sonder_session** out_session);


/* Text-only chat, additive in ABI version 1. Each record is independently
 * versioned; the pointer array allows future fields without changing its
 * stride. Larger records are accepted and unknown tail fields are ignored. */
#define SONDER_MAX_CHAT_MESSAGES 1024u
typedef struct sonder_chat_message {
    uint32_t struct_size;  /* sizeof(sonder_chat_message) */
    const char* role;      /* system, user, assistant or tool */
    const char* content;   /* UTF-8, NUL-terminated; may be empty */
} sonder_chat_message;

/* Blocks like generate, using native backend chat when available and the
 * engine's existing role-labelled prompt fallback otherwise. Requires 1 to
 * SONDER_MAX_CHAT_MESSAGES records, ending in user or tool. All record/string
 * pointers must be valid and non-NULL, borrowed until this call returns, and
 * must not be changed during the call. Inputs are copied before submission.
 * Callback, stats, cancellation and handle lifetime rules match generate.
 * No reasoning, tool-call objects or images are carried by this interface. */
SONDER_API sonder_status sonder_session_chat(sonder_session* session,
                                             const sonder_chat_message* const* messages, size_t message_count,
                                             sonder_token_callback callback, void* user_data,
                                             sonder_generation_stats* out_stats);

/* Additive ABI-v1 per-request parent lineage and ADR-020 scheduling metadata.
 * NULL parent_request_id retains no parent. Non-NULL IDs match
 * [A-Za-z0-9._:-]{1,128} and are explicit telemetry metadata independent of
 * text capture, not prompts or secrets. Records and strings must remain valid
 * and unchanged until the call returns; the parent is copied before submission.
 * Complete prefixes are required, larger records ignore unknown tails.
 * Session/run IDs and engine request IDs are unchanged. */
typedef enum sonder_workload_class {
    SONDER_WORKLOAD_DEFAULT = 0,
    SONDER_WORKLOAD_INTERACTIVE_USER = 1,
    SONDER_WORKLOAD_OWNER_ORCHESTRATOR = 2,
    SONDER_WORKLOAD_CRITIC_VERIFICATION = 3,
    SONDER_WORKLOAD_IMPLEMENTATION_WORKER = 4,
    SONDER_WORKLOAD_RESEARCH_WORKER = 5,
    SONDER_WORKLOAD_BACKGROUND_INDEXING = 6,
    SONDER_WORKLOAD_MAINTENANCE = 7
} sonder_workload_class;

typedef struct sonder_request_metadata {
    uint32_t struct_size;  /* sizeof(sonder_request_metadata) */
    const char* parent_request_id;
    /* ---- Appended ADR-020 scheduling metadata (still ABI version 1).
     * Read only when struct_size covers these fields; callers built against
     * the original layout (struct_size ending after parent_request_id) keep
     * working with session defaults. Zero-initialization selects
     * SONDER_WORKLOAD_DEFAULT (0) and priority 0 (session default).
     * workload_class: SONDER_WORKLOAD_DEFAULT (0) or 1..7.
     * priority: [-16, 16]; 0 keeps the class default, positive = more urgent.
     * When non-default, mapped onto native RequestOptions::priority_class
     * (emitted in request telemetry and enforced when strict priority
     * admission is enabled). ---- */
    int32_t workload_class;
    int32_t priority;
} sonder_request_metadata;

/* metadata may be NULL. Request, callback, stats, cancellation and handle
 * lifetime contracts match the original generate/chat exports. The parent is
 * attributes.parent_request_id on the five request lifecycle events only. */
SONDER_API sonder_status sonder_session_generate_with_metadata(
    sonder_session* session, const char* prompt, const sonder_request_metadata* metadata,
    sonder_token_callback callback, void* user_data, sonder_generation_stats* out_stats);
SONDER_API sonder_status sonder_session_chat_with_metadata(
    sonder_session* session, const sonder_chat_message* const* messages, size_t message_count,
    const sonder_request_metadata* metadata, sonder_token_callback callback, void* user_data,
    sonder_generation_stats* out_stats);

/* Additive ABI-v1 GPU device inventory.
 * Fixed UTF-8 NUL-terminated strings avoid caller-freed heap allocations.
 * Set out_info->struct_size = sizeof(sonder_gpu_info) before calling
 * sonder_gpu_device_info(). Memory counters are 0 when the matching has_*
 * flag is 0 (unknown is distinct from a known zero reading). */
typedef struct sonder_gpu_info {
    uint32_t struct_size;             /* sizeof(sonder_gpu_info) */
    char id[64];                      /* e.g. "gpu:0" */
    char name[256];                   /* human-readable adapter name */
    char vendor[32];                  /* "nvidia", "amd", "intel", "apple", "unknown" */
    char backend_hint[32];            /* "cuda", "rocm", "vulkan", "metal", or "" */
    char source[32];                  /* "dxgi", "amdgpu-sysfs", "nvml", etc. */
    uint32_t vendor_id;               /* PCI vendor id (e.g. 0x10DE) */
    uint32_t device_id;               /* PCI device id */
    uint64_t dedicated_total_bytes;   /* 0 when has_dedicated_total == 0 */
    uint64_t dedicated_free_bytes;    /* 0 when has_dedicated_free == 0 */
    uint64_t shared_total_bytes;      /* 0 when has_shared_total == 0 */
    uint64_t effective_vram_mib;      /* 0 when has_effective_vram == 0 */
    uint64_t effective_free_vram_mib; /* 0 when has_effective_free_vram == 0 */
    int32_t has_dedicated_total;
    int32_t has_dedicated_free;
    int32_t has_shared_total;
    int32_t has_effective_vram;
    int32_t has_effective_free_vram;
    int32_t is_integrated_uma;
} sonder_gpu_info;

SONDER_API size_t sonder_gpu_device_count(void);
SONDER_API sonder_status sonder_gpu_device_info(size_t index, sonder_gpu_info* out_info);

/* Additive ABI-v1 live Observatory telemetry callback sink.
 * Receives one serialized sonder.observatory.event/1 JSON line (no trailing
 * newline) from the engine's telemetry writer thread; must not block
 * indefinitely and must not call sonder_engine_destroy() from inside the
 * callback. `json_line` is `length` bytes of UTF-8 and is not guaranteed
 * NUL-terminated. Passing callback == NULL flushes queued events and clears
 * the sink; replacing or destroying the engine waits for any in-flight
 * callback invocation to finish so no dangling callback is ever invoked. */
typedef void (*sonder_telemetry_callback)(void* user_data, const char* json_line, size_t length);
SONDER_API sonder_status sonder_engine_set_telemetry_callback(sonder_engine* engine,
                                                              sonder_telemetry_callback callback,
                                                              void* user_data);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SONDER_INFERENCE_H */
