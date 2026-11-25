/* Compiled as C11: proves sonder_inference.h is a valid C header. */
#include <stdio.h>
#include <string.h>

#include "sonder_inference.h"

static int count_tokens(void* user_data, const char* text, size_t length) {
    (void)text;
    (void)length;
    ++*(int*)user_data;
    return 0;
}

typedef struct c_telemetry_state {
    int events;
    int saw_schema;
} c_telemetry_state;

static int bytes_contain(const char* haystack, size_t haystack_len, const char* needle) {
    const size_t needle_len = strlen(needle);
    size_t i;
    if (needle_len == 0 || needle_len > haystack_len) {
        return 0;
    }
    for (i = 0; i <= haystack_len - needle_len; ++i) {
        if (memcmp(haystack + i, needle, needle_len) == 0) {
            return 1;
        }
    }
    return 0;
}

static void on_telemetry(void* user_data, const char* json_line, size_t length) {
    c_telemetry_state* state = (c_telemetry_state*)user_data;
    ++state->events;
    if (bytes_contain(json_line, length, "\"schema\":\"sonder.observatory.event/1\"")) {
        state->saw_schema = 1;
    }
}

int main(void) {
    sonder_engine_options options;
    sonder_engine* engine = NULL;
    sonder_model* model = NULL;
    sonder_session* session = NULL;
    sonder_sampling_config sampling;
    sonder_generation_stats stats;
    int tokens = 0;

    if (sonder_abi_version() != SONDER_ABI_VERSION) return 1;
    {
        /* Exercise GPU inventory exports from C11. */
        const size_t gpu_count = sonder_gpu_device_count();
        size_t i;
        sonder_gpu_info gpu;
        memset(&gpu, 0, sizeof(gpu));
        if (sonder_gpu_device_info(0, NULL) != SONDER_ERROR_INVALID_ARGUMENT) return 17;
        gpu.struct_size = (uint32_t)(sizeof(sonder_gpu_info) - 1);
        if (sonder_gpu_device_info(0, &gpu) != SONDER_ERROR_INVALID_ARGUMENT) return 18;
        for (i = 0; i < gpu_count; ++i) {
            memset(&gpu, 0, sizeof(gpu));
            gpu.struct_size = sizeof(sonder_gpu_info);
            if (sonder_gpu_device_info(i, &gpu) != SONDER_OK) return 19;
            if (gpu.struct_size != sizeof(sonder_gpu_info)) return 20;
            if (memchr(gpu.id, '\0', sizeof(gpu.id)) == NULL) return 21;
            if (memchr(gpu.name, '\0', sizeof(gpu.name)) == NULL) return 22;
        }
        memset(&gpu, 0, sizeof(gpu));
        gpu.struct_size = sizeof(sonder_gpu_info);
        if (sonder_gpu_device_info(gpu_count, &gpu) != SONDER_ERROR_NOT_FOUND) return 23;
    }

    sonder_engine_options_init(&options);
    options.telemetry_level = SONDER_TELEMETRY_OFF;
    if (sonder_engine_create(&options, &engine) != SONDER_OK) return 2;
    if (sonder_engine_register_mock_backend(engine) != SONDER_OK) return 3;
    if (sonder_model_load(engine, "mock", "mock:tiny", &model) != SONDER_OK) return 4;
    sonder_sampling_config_init(&sampling);
    sampling.temperature = 0.0f;
    sampling.max_tokens = 4;
    if (sonder_session_create(engine, model, &sampling, &session) != SONDER_OK) return 5;
    memset(&stats, 0, sizeof(stats));
    stats.struct_size = sizeof(stats);
    if (sonder_session_generate(session, "c smoke", count_tokens, &tokens, &stats) != SONDER_OK) {
        fprintf(stderr, "generate failed: %s\n", sonder_last_error_message());
        return 6;
    }
    if (tokens != 4 || stats.completion_tokens != 4 || stats.outcome != SONDER_OUTCOME_COMPLETED) return 7;
    {
        const sonder_chat_message message = {sizeof(sonder_chat_message), "user", "c chat smoke"};
        const sonder_chat_message* messages[] = {&message};
        tokens = 0;
        if (sonder_session_chat(session, messages, 1, count_tokens, &tokens, &stats) != SONDER_OK) return 8;
        if (tokens != 4 || stats.chunks != 4 || stats.outcome != SONDER_OUTCOME_COMPLETED) return 9;
    }
    {
        sonder_request_metadata metadata;
        const sonder_chat_message message = {sizeof(sonder_chat_message), "user", "c parent chat"};
        const sonder_chat_message* messages[] = {&message};
        sonder_request_metadata zero_meta;
        sonder_request_metadata sched_meta;
        memset(&metadata, 0, sizeof(metadata));
        metadata.struct_size = sizeof(sonder_request_metadata);
        metadata.parent_request_id = "c-parent:1";
        metadata.workload_class = SONDER_WORKLOAD_DEFAULT;
        metadata.priority = 0;
        tokens = 0;
        if (sonder_session_generate_with_metadata(session, "c parent generate", &metadata,
                                                 count_tokens, &tokens, &stats) != SONDER_OK) return 13;
        if (tokens != 4 || stats.outcome != SONDER_OUTCOME_COMPLETED) return 14;
        tokens = 0;
        if (sonder_session_chat_with_metadata(session, messages, 1, &metadata,
                                             count_tokens, &tokens, &stats) != SONDER_OK) return 15;
        if (tokens != 4 || stats.outcome != SONDER_OUTCOME_COMPLETED) return 16;

        /* Zero-initialization selects SONDER_WORKLOAD_DEFAULT (0) and priority 0. */
        memset(&zero_meta, 0, sizeof(zero_meta));
        zero_meta.struct_size = sizeof(zero_meta);
        if (sonder_session_generate_with_metadata(session, "c zero meta", &zero_meta,
                                                 NULL, NULL, &stats) != SONDER_OK) return 24;

        /* Explicit ADR-020 workload_class and priority values. */
        memset(&sched_meta, 0, sizeof(sched_meta));
        sched_meta.struct_size = sizeof(sched_meta);
        sched_meta.parent_request_id = "c-sched:1";
        sched_meta.workload_class = SONDER_WORKLOAD_INTERACTIVE_USER;
        sched_meta.priority = 4;
        if (sonder_session_generate_with_metadata(session, "c interactive", &sched_meta,
                                                 NULL, NULL, &stats) != SONDER_OK) return 25;
        sched_meta.workload_class = SONDER_WORKLOAD_BACKGROUND_INDEXING;
        sched_meta.priority = -2;
        if (sonder_session_chat_with_metadata(session, messages, 1, &sched_meta,
                                             NULL, NULL, &stats) != SONDER_OK) return 26;
        sched_meta.workload_class = -1;
        sched_meta.priority = 0;
        if (sonder_session_generate_with_metadata(session, "c bad wl", &sched_meta,
                                                 NULL, NULL, &stats) != SONDER_ERROR_INVALID_ARGUMENT) return 27;
        sched_meta.workload_class = SONDER_WORKLOAD_MAINTENANCE + 1;
        if (sonder_session_generate_with_metadata(session, "c bad wl2", &sched_meta,
                                                 NULL, NULL, &stats) != SONDER_ERROR_INVALID_ARGUMENT) return 28;
    }
    sonder_session_destroy(session);
    {
        const sonder_session_metadata metadata = {sizeof(sonder_session_metadata),
                                                  "c-session:1", "c-run:1", NULL, NULL};
        session = NULL;
        if (sonder_session_create_with_metadata(engine, model, &sampling, &metadata, &session) != SONDER_OK) return 10;
        tokens = 0;
        if (sonder_session_generate(session, "c metadata smoke", count_tokens, &tokens, &stats) != SONDER_OK) return 11;
        if (tokens != 4 || stats.outcome != SONDER_OUTCOME_COMPLETED) return 12;
        sonder_session_destroy(session);
    }
    sonder_model_release(model);
    sonder_engine_destroy(engine);

    /* Exercise live telemetry callback from C11. */
    {
        sonder_engine* tel_engine = NULL;
        sonder_model* tel_model = NULL;
        sonder_session* tel_session = NULL;
        c_telemetry_state tel_state = {0, 0};
        int events_after_clear = 0;
        sonder_engine_options_init(&options);
        options.telemetry_level = SONDER_TELEMETRY_METRICS;
        if (sonder_engine_create(&options, &tel_engine) != SONDER_OK) return 29;
        if (sonder_engine_set_telemetry_callback(tel_engine, on_telemetry, &tel_state) != SONDER_OK) return 30;
        if (sonder_engine_register_mock_backend(tel_engine) != SONDER_OK) return 31;
        if (sonder_model_load(tel_engine, "mock", "mock:tiny", &tel_model) != SONDER_OK) return 32;
        if (sonder_session_create(tel_engine, tel_model, &sampling, &tel_session) != SONDER_OK) return 33;
        if (sonder_session_generate(tel_session, "c telemetry", NULL, NULL, &stats) != SONDER_OK) return 34;
        if (sonder_engine_set_telemetry_callback(tel_engine, NULL, NULL) != SONDER_OK) return 35;
        if (tel_state.events <= 0 || !tel_state.saw_schema) return 36;
        events_after_clear = tel_state.events;
        if (sonder_session_generate(tel_session, "c telemetry after clear", NULL, NULL, &stats) != SONDER_OK) return 37;
        sonder_session_destroy(tel_session);
        sonder_model_release(tel_model);
        sonder_engine_destroy(tel_engine);
        if (tel_state.events != events_after_clear) return 38;
    }

    printf("c abi smoke ok (%d tokens)\n", tokens);
    return 0;
}
