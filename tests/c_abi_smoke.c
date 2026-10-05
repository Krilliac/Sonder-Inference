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

int main(void) {
    sonder_engine_options options;
    sonder_engine* engine = NULL;
    sonder_model* model = NULL;
    sonder_session* session = NULL;
    sonder_sampling_config sampling;
    sonder_generation_stats stats;
    int tokens = 0;

    if (sonder_abi_version() != SONDER_ABI_VERSION) return 1;
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
        const sonder_request_metadata metadata = {sizeof(sonder_request_metadata), "c-parent:1"};
        const sonder_chat_message message = {sizeof(sonder_chat_message), "user", "c parent chat"};
        const sonder_chat_message* messages[] = {&message};
        tokens = 0;
        if (sonder_session_generate_with_metadata(session, "c parent generate", &metadata,
                                                 count_tokens, &tokens, &stats) != SONDER_OK) return 13;
        if (tokens != 4 || stats.outcome != SONDER_OUTCOME_COMPLETED) return 14;
        tokens = 0;
        if (sonder_session_chat_with_metadata(session, messages, 1, &metadata,
                                             count_tokens, &tokens, &stats) != SONDER_OK) return 15;
        if (tokens != 4 || stats.outcome != SONDER_OUTCOME_COMPLETED) return 16;
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
    printf("c abi smoke ok (%d tokens)\n", tokens);
    return 0;
}
