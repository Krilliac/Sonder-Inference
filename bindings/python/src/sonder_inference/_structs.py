"""ctypes mirrors of the structs in include/sonder_inference.h."""

from __future__ import annotations

import ctypes

c = ctypes


class EngineOptions(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("telemetry_level", c.c_int),
        ("telemetry_jsonl_path", c.c_char_p),
        ("capture_text", c.c_int32),
    ]


ENGINE_OPTIONS_SIZE = c.sizeof(EngineOptions)


class CLogitBias(c.Structure):
    _fields_ = [("token", c.c_int32), ("bias", c.c_float)]


LOGIT_BIAS_SIZE = c.sizeof(CLogitBias)


# Original layout (ABI v1 as first published): fields up to max_tokens.
class CSamplingConfigV1(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("temperature", c.c_float),
        ("top_p", c.c_float),
        ("top_k", c.c_int32),
        ("min_p", c.c_float),
        ("repeat_penalty", c.c_float),
        ("has_seed", c.c_int32),
        ("seed", c.c_uint64),
        ("max_tokens", c.c_int32),
    ]


class CSamplingConfig(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("temperature", c.c_float),
        ("top_p", c.c_float),
        ("top_k", c.c_int32),
        ("min_p", c.c_float),
        ("repeat_penalty", c.c_float),
        ("has_seed", c.c_int32),
        ("seed", c.c_uint64),
        ("max_tokens", c.c_int32),
        ("typical_p", c.c_float),
        ("presence_penalty", c.c_float),
        ("frequency_penalty", c.c_float),
        ("repeat_last_n", c.c_int32),
        ("num_ctx", c.c_int32),
        ("logit_bias", c.POINTER(CLogitBias)),
        ("logit_bias_count", c.c_size_t),
    ]


#: struct_size a caller built against the original layout would send.
SAMPLING_CONFIG_V1_SIZE = c.sizeof(CSamplingConfigV1)
SAMPLING_CONFIG_SIZE = c.sizeof(CSamplingConfig)


class CGenerationStatsV1(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("outcome", c.c_int),
        ("prompt_tokens", c.c_uint64),
        ("completion_tokens", c.c_uint64),
        ("chunks", c.c_uint64),
        ("ttft_ms", c.c_double),
        ("total_ms", c.c_double),
    ]


class CGenerationStats(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("outcome", c.c_int),
        ("prompt_tokens", c.c_uint64),
        ("completion_tokens", c.c_uint64),
        ("chunks", c.c_uint64),
        ("ttft_ms", c.c_double),
        ("total_ms", c.c_double),
        ("cached_tokens", c.c_uint64),
        ("has_cached_tokens", c.c_int32),
        ("stop_reason", c.c_int32),
        ("load_ns", c.c_uint64),
        ("prompt_eval_ns", c.c_uint64),
        ("eval_ns", c.c_uint64),
        ("prompt_ms", c.c_double),
        ("predicted_ms", c.c_double),
        ("predicted_tokens_per_second", c.c_double),
        ("draft_tokens", c.c_uint64),
        ("draft_accepted_tokens", c.c_uint64),
        ("has_draft_tokens", c.c_int32),
        ("reserved_flags", c.c_int32),
    ]


GENERATION_STATS_V1_SIZE = c.sizeof(CGenerationStatsV1)
GENERATION_STATS_SIZE = c.sizeof(CGenerationStats)


# int (*)(void* user_data, const char* text, size_t length). The text is not
# NUL-terminated, so it is received as a raw pointer (c_void_p), not c_char_p.
TOKEN_CALLBACK = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_void_p, c.c_size_t)


class CChunkEvent(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("kind", c.c_int),
        ("index", c.c_uint64),
        ("text", c.c_void_p),
        ("text_length", c.c_size_t),
        ("tool_call_id", c.c_char_p),
        ("tool_call_name", c.c_char_p),
    ]


CHUNK_EVENT_SIZE = c.sizeof(CChunkEvent)
CHUNK_CALLBACK = c.CFUNCTYPE(c.c_int, c.c_void_p, c.POINTER(CChunkEvent))


class CEmbedStats(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("reserved", c.c_uint32),
        ("prompt_tokens", c.c_uint64),
        ("eval_ns", c.c_uint64),
        ("embedding_count", c.c_size_t),
        ("dimensions", c.c_size_t),
    ]


EMBED_STATS_SIZE = c.sizeof(CEmbedStats)
EMBED_CALLBACK = c.CFUNCTYPE(c.c_int, c.c_void_p, c.c_size_t, c.POINTER(c.c_float), c.c_size_t)

# void (*)(void* user_data, const char* json_line, size_t length).
# The JSON line is not guaranteed NUL-terminated, so receive as c_void_p.
TELEMETRY_CALLBACK = c.CFUNCTYPE(None, c.c_void_p, c.c_void_p, c.c_size_t)


MAX_CORRELATION_ID_BYTES = 128
MAX_CHAT_MESSAGES = 1024
MAX_TOOLS = 256
MAX_EMBED_INPUTS = 1024


class CChatMessage(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("role", c.c_char_p),
        ("content", c.c_char_p),
    ]


CHAT_MESSAGE_SIZE = c.sizeof(CChatMessage)


class CSessionMetadata(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("session_id", c.c_char_p),
        ("run_id", c.c_char_p),
        ("agent_id", c.c_char_p),
        ("task_id", c.c_char_p),
    ]


SESSION_METADATA_SIZE = c.sizeof(CSessionMetadata)


class CToolDefinition(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("reserved", c.c_uint32),
        ("name", c.c_char_p),
        ("description", c.c_char_p),
        ("parameters_json", c.c_char_p),
    ]


TOOL_DEFINITION_SIZE = c.sizeof(CToolDefinition)


class CRequestMetadataV1(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("parent_request_id", c.c_char_p),
    ]


class CRequestMetadataV1Sched(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("parent_request_id", c.c_char_p),
        ("workload_class", c.c_int32),
        ("priority", c.c_int32),
    ]


class CRequestMetadata(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("parent_request_id", c.c_char_p),
        ("workload_class", c.c_int32),
        ("priority", c.c_int32),
        ("grammar", c.c_char_p),
        ("response_format_type", c.c_char_p),
        ("response_format_schema", c.c_char_p),
        ("separate_reasoning", c.c_int32),
        ("enable_thinking", c.c_int32),
        ("reasoning_effort", c.c_char_p),
        ("has_reasoning_budget", c.c_int32),
        ("tool_choice_mode", c.c_int32),
        ("reasoning_budget_tokens", c.c_int64),
        ("tool_choice_function", c.c_char_p),
        ("tools", c.POINTER(CToolDefinition)),
        ("tool_count", c.c_size_t),
    ]


REQUEST_METADATA_V1_SIZE = c.sizeof(CRequestMetadataV1)
REQUEST_METADATA_V1_SCHED_SIZE = c.sizeof(CRequestMetadataV1Sched)
REQUEST_METADATA_SIZE = c.sizeof(CRequestMetadata)


class CGpuInfo(c.Structure):
    _fields_ = [
        ("struct_size", c.c_uint32),
        ("id", c.c_char * 64),
        ("name", c.c_char * 256),
        ("vendor", c.c_char * 32),
        ("backend_hint", c.c_char * 32),
        ("source", c.c_char * 32),
        ("vendor_id", c.c_uint32),
        ("device_id", c.c_uint32),
        ("dedicated_total_bytes", c.c_uint64),
        ("dedicated_free_bytes", c.c_uint64),
        ("shared_total_bytes", c.c_uint64),
        ("effective_vram_mib", c.c_uint64),
        ("effective_free_vram_mib", c.c_uint64),
        ("has_dedicated_total", c.c_int32),
        ("has_dedicated_free", c.c_int32),
        ("has_shared_total", c.c_int32),
        ("has_effective_vram", c.c_int32),
        ("has_effective_free_vram", c.c_int32),
        ("is_integrated_uma", c.c_int32),
    ]


GPU_INFO_SIZE = c.sizeof(CGpuInfo)
